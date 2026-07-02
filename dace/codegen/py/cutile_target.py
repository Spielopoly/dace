"""Schedule-based cuTile Python code generation target.

AccessNode-centric design: MapEntry emits only PIDs and map variable
bindings; each CuTile_Tile AccessNode handles its own ``ct.load`` /
``ct.store`` (or ``ct.gather`` / ``ct.scatter`` for non-aligned
tiles).  MapExit is a no-op.
"""

import warnings
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

import sympy as sp

from dace import data, dtypes, registry, subsets
import dace.codegen.dispatcher as dispatcher_mod
from dace.codegen.py import control_flow as py_cflow
from dace.codegen.py.framecode import codeblock_to_python
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.codegen.py.target import PythonTargetCodeGenerator
from dace.sdfg import nodes
from dace.symbolic import symstr

if TYPE_CHECKING:
    from dace.codegen.py.framecode import DaCePythonCodeGenerator
    from dace.sdfg import SDFG, SDFGState

# ---------------------------------------------------------------------------
# Constants — avoid magic strings throughout the file.
# ---------------------------------------------------------------------------

#: Prefix for outer (input) scope connectors on a MapEntry / MapExit.
_SCOPE_IN_PREFIX: str = "IN_"

#: Prefix for inner (output) scope connectors on a MapEntry / MapExit.
_SCOPE_OUT_PREFIX: str = "OUT_"


def _matching_inner_connector(outer_conn: str) -> str:
    """Convert an outer (input) scope connector name to the matching inner (output) name.

    :param outer_conn: The outer connector name, e.g. ``"IN_A"``.
    :returns: The matching inner connector, e.g. ``"OUT_A"``.
    :raises ValueError: If *outer_conn* does not start with the expected prefix.
    """
    if not outer_conn.startswith(_SCOPE_IN_PREFIX):
        raise ValueError(f"Expected connector starting with {_SCOPE_IN_PREFIX!r}, got {outer_conn!r}")
    return _SCOPE_OUT_PREFIX + outer_conn[len(_SCOPE_IN_PREFIX):]


def _matching_outer_connector(inner_conn: str) -> str:
    """Convert an inner (output) scope connector name to the matching outer (input) name.

    :param inner_conn: The inner connector name, e.g. ``"OUT_A"``.
    :returns: The matching outer connector, e.g. ``"IN_A"``.
    :raises ValueError: If *inner_conn* does not start with the expected prefix.
    """
    if not inner_conn.startswith(_SCOPE_OUT_PREFIX):
        raise ValueError(f"Expected connector starting with {_SCOPE_OUT_PREFIX!r}, got {inner_conn!r}")
    return _SCOPE_IN_PREFIX + inner_conn[len(_SCOPE_OUT_PREFIX):]


def _array_runtime_name(sdfg: "SDFG", name: str) -> str:
    """Return the runtime variable name for a data array.

    Handles global/persistent/external arrays that live in ``globals()``.

    :param sdfg: The SDFG containing the array.
    :param name: The array name (possibly dotted).
    :returns: The runtime variable expression.
    """
    root_name, sep, suffix = name.partition(".")
    desc = sdfg.arrays.get(root_name)
    if desc is None:
        return name
    if desc.lifetime in (
            dtypes.AllocationLifetime.Global,
            dtypes.AllocationLifetime.Persistent,
            dtypes.AllocationLifetime.External,
    ):
        base = f"globals()[{root_name!r}]"
        return f"{base}.{suffix}" if sep else base
    return name


def _grid_exprs_from_map_entry(entry: nodes.MapEntry) -> List[str]:
    """Return per-dimension grid size (number of tiles) expressions from a map entry.

    Each grid dimension is the number of tiles ``ceil(extent / step)`` where
    ``extent = end - start + 1`` (the map range end is inclusive). This is
    emitted as a structural integer ceil-division ``int_ceil(extent, step)``
    rather than ``symstr(range.size())``.

    The latter is unsound for the Python/cuTile backend: ``symstr`` rewrites a
    symbolic ceiling using C integer-division semantics (e.g.
    ``ceiling((N-2)/8)`` becomes ``int_ceil(int_floor(N, 8) - 1/4, 1)``). Under
    Python's true division the residual rational ``1/4`` is the float ``0.25``,
    which both yields a non-integer grid dimension (rejected by ``ct.launch``)
    and is off-by-one for non-divisible extents. Building ``int_ceil`` directly
    from ``(start, end, step)`` keeps both operands integral.

    :param entry: The map entry node.
    :returns: List of grid-dimension expression strings (one per map dimension).
    """
    # Delegate to the single shared implementation so the map-entry grid and the
    # tile-op ``cutile`` expansions cannot drift apart (divergent grid strings
    # would silently desynchronize the folded launch-grid PIDs).
    from dace.libraries.tileops._pure_codegen import cutile_grid_size_exprs
    return cutile_grid_size_exprs(entry)


def _fold_grid_to_launch(grid_exprs: List[str]) -> Tuple[List[str], List[str]]:
    """Fold a ``K``-dimensional tile grid onto the cuTile launch grid (rank <= 3).

    The ``cuda.tile`` runtime caps the launch grid at three axes (``Dim3`` in
    ``ct.launch``; ``ct.bid(axis)`` only accepts ``axis in {0, 1, 2}``). When a
    tiled map nest has more than three dimensions (e.g. 4-D ``softmax``, 5-D
    ``conv2d``), the extra dimensions are linearized onto the available axes and
    recovered inside the kernel via integer div/mod.

    The fold layout is the single canonical contract defined in
    :mod:`dace.libraries.tileops._pure_codegen` (``cutile_launch_grid_dims`` /
    ``cutile_bid_expr``), shared with every tile-op ``cutile`` expansion so the
    map-entry block IDs and the tile-op block IDs agree: the two innermost map
    dimensions map to grid axes 1 and 2, and the leading ``K-2`` dimensions are
    folded row-major onto grid axis 0. For ``K <= 3`` the identity mapping is
    used (``__pid{d} = ct.bid(d)``), unchanged from the pre-folding behavior.

    :param grid_exprs: Per-dimension grid-size (tile-count) expression strings,
        in map order (dimension 0 is outermost). Length is ``K``.
    :returns: A tuple ``(launch_dims, pid_stmts)`` where ``launch_dims`` is the
        list of at most three launch-grid dimension expressions passed to
        ``ct.launch``, and ``pid_stmts`` is the list of Python statement strings
        that bind ``__pid{d}`` for every map dimension ``d`` inside the kernel.
    """
    from dace.libraries.tileops._pure_codegen import cutile_bid_expr, cutile_launch_grid_dims
    num_dims = len(grid_exprs)
    launch_dims = cutile_launch_grid_dims(grid_exprs)
    pid_stmts = [f"__pid{d} = {cutile_bid_expr(d, num_dims, grid_exprs)}" for d in range(num_dims)]
    return launch_dims, pid_stmts


def _map_index_exprs(entry: nodes.MapEntry) -> List[str]:
    """Return per-dimension element-coordinate expressions.

    Each expression computes the global element index from the PID
    (block ID) and the map's start/step parameters.

    :param entry: The map entry node.
    :returns: List of index expression strings.
    """
    result: List[str] = []
    for d, (start, _, step) in enumerate(entry.map.range):
        pid = f"__pid{d}"
        start_s = symstr(start)
        step_s = symstr(step)
        if start_s == "0" and step_s == "1":
            result.append(pid)
        elif step_s == "1":
            result.append(f"({start_s} + {pid})")
        else:
            result.append(f"({start_s} + {pid} * {step_s})")
    return result


def _ordered_unique(items: Iterable[str]) -> List[str]:
    """Deduplicate items while preserving insertion order, then sort.

    :param items: Iterable of strings.
    :returns: Sorted list of unique strings.
    """
    return sorted(set(items))


def _collect_free_symbols(entry: nodes.MapEntry, dfg_scope: object, sdfg: "SDFG") -> List[str]:
    """Collect free symbols used in a map scope.

    Returns symbols that appear in the map range, memlet subsets, or
    NestedSDFG symbol mappings and are declared in the SDFG's symbol
    table (but not constants).

    :param entry: The map entry node.
    :param dfg_scope: The scope subgraph view.
    :param sdfg: The SDFG.
    :returns: Sorted list of free symbol names.
    """
    syms = {str(s) for s in entry.map.range.free_symbols}
    for edge in dfg_scope.edges():
        memlet = edge.data
        if memlet is None:
            continue
        syms |= {str(s) for s in memlet.free_symbols}
    # Also collect symbols referenced by NestedSDFG symbol_mapping values,
    # so that symbols like ``cond_val`` that are forwarded into a conditional
    # NestedSDFG become kernel parameters.
    for scope_node in dfg_scope.nodes():
        if isinstance(scope_node, nodes.NestedSDFG):
            for expr in scope_node.symbol_mapping.values():
                if hasattr(expr, 'free_symbols'):
                    syms |= {str(s) for s in expr.free_symbols}
                else:
                    syms.add(str(expr))
        elif isinstance(scope_node, nodes.Tasklet):
            # Symbols referenced only in tasklet code (e.g. a TileBinop
            # Symbol-operand expansion emitting ``_a / N``) appear in no
            # memlet; without this they are undefined inside the kernel.
            syms |= scope_node.free_symbols
    # Loop induction variables and interstate-assigned names are module-level
    # Python locals in the generated code but not necessarily in
    # ``sdfg.symbols``; they must still become kernel parameters.
    runtime_defined = set()
    for region in sdfg.all_control_flow_regions():
        loop_var = getattr(region, 'loop_variable', None)
        if loop_var:
            runtime_defined.add(str(loop_var))
    for isedge in sdfg.all_interstate_edges():
        runtime_defined |= set(isedge.data.assignments.keys())
    syms = {
        s
        for s in syms
        if (s in sdfg.symbols or s in runtime_defined) and s not in sdfg.constants and s not in sdfg.arrays
    }
    return sorted(syms)


def _enclosing_cutile_entry(state: "SDFGState", node: nodes.Node) -> Optional[nodes.MapEntry]:
    """Find the nearest enclosing CuTile-scheduled MapEntry for a node.

    :param state: The SDFG state.
    :param node: The node to check.
    :returns: The enclosing CuTile MapEntry, or ``None`` if not inside one.
    """
    scope = state.scope_dict()
    cur = scope.get(node)
    while cur is not None:
        if isinstance(cur, nodes.MapEntry) and cur.map.schedule == dtypes.ScheduleType.CuTile:
            return cur
        cur = scope.get(cur)
    return None


def _is_cutile_node(state: "SDFGState", node: nodes.Node) -> bool:
    """Check whether a node belongs to a CuTile scope (entry, exit, or inside).

    :param state: The SDFG state.
    :param node: The node to check.
    :returns: ``True`` if the node is part of a CuTile scope.
    """
    if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.CuTile:
        return True
    if isinstance(node, nodes.MapExit):
        entry = state.entry_node(node)
        if (entry is not None and isinstance(entry, nodes.MapEntry)
                and entry.map.schedule == dtypes.ScheduleType.CuTile):
            return True
    return _enclosing_cutile_entry(state, node) is not None


# ---------------------------------------------------------------------------
# Code generator class
# ---------------------------------------------------------------------------


@registry.autoregister_params(name="cutile_python")
class CuTilePythonCodeGen(PythonTargetCodeGenerator):
    """Python target for CuTile-scheduled map scopes.

    Uses an AccessNode-centric design where each ``CuTile_Tile``
    AccessNode handles its own ``ct.load`` / ``ct.store`` operations,
    rather than centralizing loads at MapEntry and stores at MapExit.
    """

    title = "CuTilePython"
    target_name = "cutile_python"
    language = "python"

    def __init__(self, frame_codegen: "DaCePythonCodeGenerator", sdfg: "SDFG") -> None:
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher
        #: Tracks already-generated nested functions by position key to avoid duplicates.
        self._generated_nested_functions: Dict[str, str] = {}
        # Register as the handler for CuTile map scopes.
        self._dispatcher.register_map_dispatcher(dtypes.ScheduleType.CuTile, self)
        # Register as node handler for all nodes inside CuTile scopes.
        self._dispatcher.register_node_dispatcher(self, predicate=self._is_in_cutile_scope)
        # Register copy dispatchers for transfers involving CuTile_Tile storage.
        # The actual loads/stores are emitted by _generate_AccessNode, not by
        # copy_memory, but the dispatcher's target-collection phase needs to
        # find a handler for every memlet-tree edge pair.  Register for all
        # storage types that can appear in SDFG edges touching tile transients.
        _tile_peer_storages = [
            dtypes.StorageType.GPU_Global,
            dtypes.StorageType.Register,
        ]
        _copy_schedules = [dtypes.ScheduleType.CuTile, None]
        for peer_storage in _tile_peer_storages:
            for sched in _copy_schedules:
                self._dispatcher.register_copy_dispatcher(peer_storage, dtypes.StorageType.CuTile_Tile, sched, self)
                self._dispatcher.register_copy_dispatcher(dtypes.StorageType.CuTile_Tile, peer_storage, sched, self)
        # Also register tile-to-tile copies within a CuTile scope.
        for sched in _copy_schedules:
            self._dispatcher.register_copy_dispatcher(dtypes.StorageType.CuTile_Tile, dtypes.StorageType.CuTile_Tile,
                                                      sched, self)
        # Register for cross-storage copies between CPU_Heap and GPU_Global.
        # These arise from apply_gpu_transformations() copy-in/copy-out states
        # that transfer data between host and device outside any map scope.
        self._dispatcher.register_copy_dispatcher(dtypes.StorageType.CPU_Heap, dtypes.StorageType.GPU_Global, None,
                                                  self)
        self._dispatcher.register_copy_dispatcher(dtypes.StorageType.GPU_Global, dtypes.StorageType.CPU_Heap, None,
                                                  self)
        # Register array dispatcher for CuTile_Tile storage (allocation is a no-op).
        self._dispatcher.register_array_dispatcher(dtypes.StorageType.CuTile_Tile, self)

    def get_generated_codeobjects(self) -> list:
        """Return generated code objects (none for this target).

        :returns: Empty list.
        """
        return []

    def get_includes(self) -> Dict[str, List[str]]:
        """Return import statements needed for cuTile kernels.

        :returns: Mapping from code section to list of import lines.
        """
        return {"frame": ["import cuda.tile as ct", "import cupy"]}

    def preprocess(self, sdfg: "SDFG") -> None:
        """Preprocessing hook (no-op for cuTile).

        :param sdfg: The SDFG to preprocess.
        """
        pass

    @property
    def has_initializer(self) -> bool:
        """Whether this target has an initialization function."""
        return False

    @property
    def has_finalizer(self) -> bool:
        """Whether this target has a finalization function."""
        return False

    # ------------------------------------------------------------------
    # Dispatcher predicates
    # ------------------------------------------------------------------

    def _sdfg_is_cutile_body(self, sdfg: "SDFG") -> bool:
        """Whether *sdfg* is a NestedSDFG body emitted as a cuTile function.

        A NestedSDFG body has no CuTile MapEntry in its own states, so this is
        determined structurally by walking the nested-SDFG parent chain: the
        body is a cuTile function iff the NestedSDFG node that owns it (at any
        level) lives inside a CuTile-scheduled map.

        :param sdfg: The (possibly nested) SDFG to check.
        :returns: ``True`` if *sdfg* is a cuTile NestedSDFG body.
        """
        cur = sdfg
        while cur.parent_nsdfg_node is not None:
            if _is_cutile_node(cur.parent, cur.parent_nsdfg_node):
                return True
            cur = cur.parent_sdfg
        return False

    def _is_in_cutile_scope(self, sdfg: "SDFG", state: "SDFGState", node: nodes.Node) -> bool:
        """Predicate for the node dispatcher: return True for CuTile nodes.

        A node is in a cuTile scope if it lives (transitively) inside a
        CuTile-scheduled map, or if it belongs to a NestedSDFG body emitted as
        a cuTile function (see :meth:`_sdfg_is_cutile_body`).

        :param sdfg: The SDFG.
        :param state: The SDFG state.
        :param node: The node to check.
        :returns: ``True`` if the node belongs to a CuTile scope.
        """
        return _is_cutile_node(state, node) or self._sdfg_is_cutile_body(sdfg)

    def _in_cutile_context(self, sdfg: "SDFG", state: "SDFGState", node: nodes.Node) -> bool:
        """Whether *node* is inside a cuTile scope, counting NestedSDFG bodies.

        :param sdfg: The SDFG containing the node.
        :param state: The state containing the node.
        :param node: The node to check.
        :returns: ``True`` if generating cuTile code is valid for this node.
        """
        return (_enclosing_cutile_entry(state, node) is not None or self._sdfg_is_cutile_body(sdfg))

    # ------------------------------------------------------------------
    # Array dispatcher methods (no-ops for tile arrays)
    # ------------------------------------------------------------------

    def allocate_array(self, sdfg: "SDFG", cfg: object, dfg: object, state_id: int, node: nodes.Node, nodedesc: object,
                       global_stream: PythonCodeIOStream, declaration_stream: PythonCodeIOStream,
                       allocation_stream: PythonCodeIOStream) -> None:
        """Tile arrays inside cuTile kernels are Python locals -- no allocation needed.

        :param sdfg: The SDFG.
        :param cfg: The control flow graph.
        :param dfg: The dataflow graph.
        :param state_id: The state ID.
        :param node: The access node.
        :param nodedesc: The data descriptor.
        :param global_stream: Stream for global code.
        :param declaration_stream: Stream for declarations.
        :param allocation_stream: Stream for allocations.
        """
        pass

    def deallocate_array(self, sdfg: "SDFG", cfg: object, dfg: object, state_id: int, node: nodes.Node,
                         nodedesc: object, function_stream: PythonCodeIOStream,
                         callsite_stream: PythonCodeIOStream) -> None:
        """Tile arrays inside cuTile kernels are Python locals -- no deallocation needed.

        :param sdfg: The SDFG.
        :param cfg: The control flow graph.
        :param dfg: The dataflow graph.
        :param state_id: The state ID.
        :param node: The access node.
        :param nodedesc: The data descriptor.
        :param function_stream: Stream for function code.
        :param callsite_stream: Stream for call-site code.
        """
        pass

    def declare_array(self, sdfg: "SDFG", cfg: object, dfg: object, state_id: int, node: nodes.Node, nodedesc: object,
                      global_stream: PythonCodeIOStream, declaration_stream: PythonCodeIOStream) -> None:
        """Tile arrays are not declared -- they are created by ct.load or tasklet output.

        :param sdfg: The SDFG.
        :param cfg: The control flow graph.
        :param dfg: The dataflow graph.
        :param state_id: The state ID.
        :param node: The access node.
        :param nodedesc: The data descriptor.
        :param global_stream: Stream for global code.
        :param declaration_stream: Stream for declarations.
        """
        pass

    # ------------------------------------------------------------------
    # Copy dispatcher method
    # ------------------------------------------------------------------

    def copy_memory(self, sdfg: "SDFG", cfg: object, dfg: object, state_id: int, src_node: nodes.Node,
                    dst_node: nodes.Node, edge: object, function_stream: PythonCodeIOStream,
                    callsite_stream: PythonCodeIOStream) -> None:
        """Handle copy operations between arrays.

        Supports three categories of copies:

        1. **Cross-storage CPU_Heap/Default <-> GPU_Global** (from
           ``apply_gpu_transformations()`` copy-in/copy-out states): emits
           ``.set()`` (host-to-device) or ``.get(out=...)``
           (device-to-host) transfers.  ``StorageType.Default`` is
           treated as host-side since it resolves to ``CPU_Heap``.
        2. **CuTile_Tile <-> other storage** inside a scope: data flows
           through MapEntry/MapExit and is handled by
           :meth:`_generate_AccessNode`; this path is a fallback for
           direct AccessNode-to-AccessNode edges routed here.
        3. **Same-storage copies**: emits plain numpy-compatible
           assignment.

        :param sdfg: The SDFG.
        :param cfg: The control flow graph.
        :param dfg: The dataflow graph.
        :param state_id: The state ID.
        :param src_node: The source node.
        :param dst_node: The destination node.
        :param edge: The connecting edge.
        :param function_stream: Stream for function code.
        :param callsite_stream: Stream for call-site code.
        :raises NotImplementedError: If the copy is not between two
            AccessNodes.
        """
        memlet = edge.data
        if not isinstance(src_node, nodes.AccessNode) or not isinstance(dst_node, nodes.AccessNode):
            raise NotImplementedError(f"CuTile copy_memory only supports AccessNode-to-AccessNode "
                                      f"copies, got {type(src_node).__name__} -> "
                                      f"{type(dst_node).__name__}")

        src_storage = sdfg.arrays[src_node.data].storage
        dst_storage = sdfg.arrays[dst_node.data].storage

        # --- Cross-storage CPU <-> GPU transfers ---
        # Register counts as host-side: Register transients (e.g. staged
        # scalars) live in host memory in the Python backend.
        _HOST_STORAGES = (dtypes.StorageType.CPU_Heap, dtypes.StorageType.Default, dtypes.StorageType.Register)
        src_on_host = src_storage in _HOST_STORAGES
        dst_on_host = dst_storage in _HOST_STORAGES
        src_on_gpu = src_storage == dtypes.StorageType.GPU_Global
        dst_on_gpu = dst_storage == dtypes.StorageType.GPU_Global
        is_cpu_to_gpu = src_on_host and dst_on_gpu
        is_gpu_to_cpu = src_on_gpu and dst_on_host

        if is_cpu_to_gpu or is_gpu_to_cpu:
            self._emit_cross_storage_copy(sdfg, cfg, state_id, src_node, dst_node, memlet, is_cpu_to_gpu,
                                          callsite_stream)
            return

        # --- Same-storage / CuTile_Tile fallback copies ---
        # Build source expression
        src_expr = src_node.data
        if memlet.src_subset is not None:
            src_subset_str = self._subset_to_python(memlet.src_subset)
            if src_subset_str:
                src_expr = f"{src_node.data}[{src_subset_str}]"

        # Build destination expression
        dst_expr = dst_node.data
        if memlet.dst_subset is not None:
            dst_subset_str = self._subset_to_python(memlet.dst_subset)
            if dst_subset_str:
                dst_expr = f"{dst_node.data}[{dst_subset_str}]"

        # Emit assignment (works for both numpy and cupy)
        callsite_stream.write(f"{dst_expr} = {src_expr}", cfg, state_id)

    def _emit_cross_storage_copy(
        self,
        sdfg: "SDFG",
        cfg: object,
        state_id: int,
        src_node: nodes.AccessNode,
        dst_node: nodes.AccessNode,
        memlet: object,
        cpu_to_gpu: bool,
        callsite_stream: PythonCodeIOStream,
    ) -> None:
        """Emit a cross-storage copy between CPU_Heap and GPU_Global arrays.

        For CPU_Heap -> GPU_Global (copy-in), emits::

            dst.set(src)

        For GPU_Global -> CPU_Heap (copy-out), emits::

            src.get(out=dst)

        When the memlet carries subsets, the subset is applied to both
        source and destination expressions.  For full-array copies
        (typical of ``apply_gpu_transformations()``), the ``[:]`` ensures
        the data is copied into the pre-allocated array.

        :param sdfg: The SDFG.
        :param cfg: The control flow graph.
        :param state_id: The state ID.
        :param src_node: The source AccessNode.
        :param dst_node: The destination AccessNode.
        :param memlet: The memlet on the connecting edge.
        :param cpu_to_gpu: ``True`` for CPU_Heap -> GPU_Global,
            ``False`` for GPU_Global -> CPU_Heap.
        :param callsite_stream: Stream for call-site code.
        """
        # Build source expression (with optional subset).
        src_expr = src_node.data
        if memlet.src_subset is not None:
            src_subset_str = self._subset_to_python(memlet.src_subset)
            if src_subset_str:
                src_expr = f"{src_node.data}[{src_subset_str}]"

        # Host-side SCALAR endpoints need value semantics: ``.set``/``.get``
        # (and plain assignment) require arrays, and assigning a cupy 0-d into
        # a host scalar (or a numpy value into a device slice) fails at runtime.
        src_desc = sdfg.arrays[src_node.data]
        dst_desc = sdfg.arrays[dst_node.data]
        if not cpu_to_gpu and isinstance(dst_desc, data.Scalar):
            value_expr = f"{src_expr}.item()"
            if dst_desc.transient:
                # Plain Python local: rebinding is the correct write.
                callsite_stream.write(f"{dst_node.data} = {value_expr}", cfg, state_id)
            else:
                # Non-transient scalars are 0-d numpy buffers (caller-aliased).
                callsite_stream.write(f"{dst_node.data}[...] = {value_expr}", cfg, state_id)
            return
        if cpu_to_gpu and isinstance(src_desc, data.Scalar):
            value_expr = src_node.data if src_desc.transient else f"{src_node.data}.item()"
            dst_subset_str = self._subset_to_python(memlet.dst_subset) if memlet.dst_subset is not None else ""
            dst_expr = f"{dst_node.data}[{dst_subset_str or '...'}]"
            callsite_stream.write(f"{dst_expr} = {value_expr}", cfg, state_id)
            return

        # Build destination LHS (with optional subset, or [:] for full copy).
        if memlet.dst_subset is not None:
            dst_subset_str = self._subset_to_python(memlet.dst_subset)
            if dst_subset_str:
                dst_lhs = f"{dst_node.data}[{dst_subset_str}]"
            else:
                dst_lhs = f"{dst_node.data}"
        else:
            dst_lhs = f"{dst_node.data}"

        if cpu_to_gpu:
            callsite_stream.write(f"{dst_lhs}.set({src_expr})", cfg, state_id)
        else:
            callsite_stream.write(f"{src_expr}.get(out={dst_lhs})", cfg, state_id)

    @staticmethod
    def _subset_to_python(subset: "subsets.Subset") -> str:
        """Convert a subset to a Python indexing string.

        :param subset: The subset to convert.
        :returns: Python-style indexing string (e.g., ``"0:16, 0:8"``),
            or an empty string if *subset* is ``None``.
        """
        if subset is None:
            return ""
        if isinstance(subset, subsets.Range):
            parts: List[str] = []
            for start, end, step in subset:
                start_s = symstr(start) if start != 0 else ""
                end_s = symstr(end + 1)
                step_s = f":{symstr(step)}" if step != 1 else ""
                parts.append(f"{start_s}:{end_s}{step_s}")
            return ", ".join(parts)
        return str(subset)

    # ------------------------------------------------------------------
    # Per-tile alignment check
    # ------------------------------------------------------------------

    @staticmethod
    def _needs_gather_for_tile(entry: nodes.MapEntry, tile_shape: Tuple[int, ...], sdfg: "SDFG") -> bool:
        """Check if gather/scatter is needed for a specific tile.

        Returns ``True`` if the outer map's start is non-zero or the
        outer map's step doesn't match the given tile shape in any
        dimension.

        :param entry: The outer :class:`~dace.sdfg.nodes.MapEntry`.
        :param tile_shape: Shape of the specific tile being loaded/stored.
        :param sdfg: The SDFG (for symbol resolution).
        :returns: ``True`` if gather/scatter is needed for this tile.
        """
        for d, (start, _, step) in enumerate(entry.map.range):
            start_val = sp.sympify(start)
            step_val = sp.sympify(step)
            if start_val != 0:
                return True
            if d < len(tile_shape):
                tile_dim = tile_shape[d]
                if sp.sympify(step_val) != sp.sympify(tile_dim):
                    return True
        return False

    # ------------------------------------------------------------------
    # Single-tile shape resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_single_tile_shape(entry: nodes.MapEntry, tile_name: str, sdfg: "SDFG") -> Tuple[int, ...]:
        """Resolve the shape of a single tile transient to concrete integers.

        Reads the shape from the tile's array descriptor and substitutes
        map parameters and SDFG symbols to obtain integer dimensions.

        :param entry: The enclosing :class:`~dace.sdfg.nodes.MapEntry`.
        :param tile_name: Name of the tile transient in the SDFG.
        :param sdfg: The SDFG.
        :returns: Tuple of resolved integer dimensions.
        :raises RuntimeError: If the tile is not found or shape cannot be resolved.
        """
        desc = sdfg.arrays.get(tile_name)
        if desc is None:
            raise RuntimeError(f"Tile array {tile_name!r} not found in SDFG.")

        _tile_subs = {sp.Symbol(p): r[0] for p, r in zip(entry.map.params, entry.map.range)}
        _sym_subs = {sp.Symbol(s): sp.Integer(2**31) for s in sdfg.symbols}

        resolved: List[int] = []
        for s in desc.shape:
            val = sp.sympify(s).subs(_tile_subs).subs(_sym_subs)
            if val.is_Number:
                resolved.append(int(val))
            else:
                resolved.append(int(str(symstr(val))))  # best effort
        return tuple(resolved)

    # ------------------------------------------------------------------
    # Shape validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_tile_shape(tile_shape: Tuple[int, ...], entry: nodes.MapEntry, tile_name: str, sdfg: "SDFG") -> None:
        """Validate that the resolved tile shape is consistent with the map.

        Checks:

        1. All tile dimensions are positive integers.
        2. The tile does not have more dimensions than the enclosing map.
        3. For the aligned path (``ct.load``), warns if any tile dimension
           does not match the corresponding map step (since such cases
           require ``ct.gather`` instead).

        :param tile_shape: The resolved tile shape.
        :param entry: The enclosing MapEntry.
        :param tile_name: Name of the tile transient.
        :param sdfg: The SDFG.
        :raises RuntimeError: If the tile shape is invalid.
        """
        # 1. All dimensions must be positive.
        for d, dim in enumerate(tile_shape):
            if dim <= 0:
                raise RuntimeError(f"Tile {tile_name!r} has non-positive dimension {dim} "
                                   f"at axis {d}. All tile dimensions must be positive.")

        # 2. Tile dimensionality must not exceed map dimensionality.
        map_ndim = len(entry.map.range)
        if len(tile_shape) > map_ndim:
            raise RuntimeError(f"Tile {tile_name!r} has {len(tile_shape)} dimensions but "
                               f"the enclosing map has only {map_ndim} dimensions.")

        # 3. Advisory: warn when tile dims don't match map steps.
        for d, (_, _, step) in enumerate(entry.map.range):
            if d >= len(tile_shape):
                break
            step_val = int(sp.sympify(step))
            if tile_shape[d] != step_val:
                warnings.warn(
                    f"Tile {tile_name!r} dimension {d} has size "
                    f"{tile_shape[d]} but the enclosing map step is "
                    f"{step_val}. The gather/scatter path will be used.",
                    stacklevel=2)

    # ------------------------------------------------------------------
    # Trace load/store sources and targets through scope boundaries
    # ------------------------------------------------------------------

    @staticmethod
    def _trace_load_source_edge(state: "SDFGState", entry: nodes.MapEntry, in_edge: object) -> Optional[object]:
        """Trace through a MapEntry to the outer edge feeding a scope input.

        Given an edge MapEntry -> (inner node), finds the corresponding outer
        edge AccessNode(global) -> MapEntry (which carries the source memlet,
        including its subset).

        :param state: The SDFG state.
        :param entry: The MapEntry node.
        :param in_edge: The edge from MapEntry to the inner node.
        :returns: The outer edge, or ``None`` if not found.
        """
        src_conn = in_edge.src_conn
        if src_conn is None or not src_conn.startswith(_SCOPE_OUT_PREFIX):
            return None
        outer_conn = _matching_outer_connector(src_conn)
        for outer_edge in state.in_edges_by_connector(entry, outer_conn):
            if isinstance(outer_edge.src, nodes.AccessNode):
                return outer_edge
        return None

    @staticmethod
    def _trace_load_source(state: "SDFGState", entry: nodes.MapEntry, tile_node: nodes.AccessNode,
                           in_edge: object) -> Optional[str]:
        """Trace through a MapEntry to find the global array feeding a tile.

        Given an edge MapEntry -> AccessNode(tile), finds the corresponding
        outer edge AccessNode(global) -> MapEntry and returns the global
        array name.

        :param state: The SDFG state.
        :param entry: The MapEntry node.
        :param tile_node: The tile AccessNode inside the scope.
        :param in_edge: The edge from MapEntry to tile_node.
        :returns: The global array name, or ``None`` if not found.
        """
        outer_edge = CuTilePythonCodeGen._trace_load_source_edge(state, entry, in_edge)
        if outer_edge is None:
            return None
        return outer_edge.data.data if outer_edge.data else outer_edge.src.data

    @staticmethod
    def _trace_store_target(state: "SDFGState", exit_node: nodes.MapExit, tile_node: nodes.AccessNode,
                            out_edge: object) -> Optional[str]:
        """Trace through a MapExit to find the global array receiving a tile.

        Given an edge AccessNode(tile) -> MapExit, finds the corresponding
        outer edge MapExit -> AccessNode(global) and returns the global
        array name.

        :param state: The SDFG state.
        :param exit_node: The MapExit node.
        :param tile_node: The tile AccessNode inside the scope.
        :param out_edge: The edge from tile_node to MapExit.
        :returns: The global array name, or ``None`` if not found.
        """
        dst_conn = out_edge.dst_conn
        if dst_conn is None or not dst_conn.startswith(_SCOPE_IN_PREFIX):
            return None
        outer_conn = _matching_inner_connector(dst_conn)
        for outer_edge in state.out_edges_by_connector(exit_node, outer_conn):
            if isinstance(outer_edge.dst, nodes.AccessNode):
                return outer_edge.data.data if outer_edge.data else outer_edge.dst.data
        return None

    @staticmethod
    def _load_source_begins(state: "SDFGState", entry: nodes.MapEntry, in_edge: object) -> Optional[List[str]]:
        """Per-dim begin expressions of the outer memlet feeding a tile load.

        :param state: The SDFG state.
        :param entry: The MapEntry node.
        :param in_edge: The edge from MapEntry to the tile AccessNode.
        :returns: Per-dim begin expression strings of the outer (global-array)
            memlet, or ``None`` if not resolvable.
        """
        src_conn = in_edge.src_conn
        if src_conn is None or not src_conn.startswith(_SCOPE_OUT_PREFIX):
            return None
        outer_conn = _matching_outer_connector(src_conn)
        for outer_edge in state.in_edges_by_connector(entry, outer_conn):
            if (isinstance(outer_edge.src, nodes.AccessNode) and outer_edge.data is not None
                    and outer_edge.data.subset is not None):
                return [symstr(r[0]) for r in outer_edge.data.subset.ranges]
        return None

    @staticmethod
    def _store_target_begins(state: "SDFGState", exit_node: nodes.MapExit, out_edge: object) -> Optional[List[str]]:
        """Per-dim begin expressions of the outer memlet receiving a tile store.

        :param state: The SDFG state.
        :param exit_node: The MapExit node.
        :param out_edge: The edge from the tile AccessNode to the MapExit.
        :returns: Per-dim begin expression strings of the outer (global-array)
            memlet, or ``None`` if not resolvable.
        """
        dst_conn = out_edge.dst_conn
        if dst_conn is None or not dst_conn.startswith(_SCOPE_IN_PREFIX):
            return None
        outer_conn = _matching_inner_connector(dst_conn)
        for outer_edge in state.out_edges_by_connector(exit_node, outer_conn):
            if (isinstance(outer_edge.dst, nodes.AccessNode) and outer_edge.data is not None
                    and outer_edge.data.subset is not None):
                return [symstr(r[0]) for r in outer_edge.data.subset.ranges]
        return None

    @staticmethod
    def _const_begin_offsets(begins: Optional[List[str]], entry: nodes.MapEntry) -> List[object]:
        """Constant element offset per dim carried by the outer memlet begin.

        The outer memlet begin has the form ``<iter-var> + c`` (e.g. an offset
        slice ``B[1:-1]`` yields begin ``tile_i + 1``). Substituting every map
        iteration variable with ``0`` leaves the constant offset ``c`` that the
        block-id / map-range index reconstruction drops -- it must be added back
        to the ``ct.load`` / ``ct.gather`` / ``ct.scatter`` element index.

        :param begins: Per-dim begin expression strings (or ``None``).
        :param entry: The enclosing MapEntry (for its iteration variables).
        :returns: Per-dim symbolic offsets (``0`` where the begin is unusable or
            anchored at the block-aligned start).
        """
        if begins is None:
            return []
        subs = {sp.Symbol(str(p)): sp.Integer(0) for p in entry.map.params}
        offsets: List[object] = []
        for b in begins:
            try:
                offsets.append(sp.simplify(sp.sympify(b).subs(subs)))
            except Exception:  # noqa: BLE001 - non-symbolic begin -> assume anchored
                offsets.append(sp.Integer(0))
        return offsets

    @staticmethod
    def _apply_index_offsets(map_index_exprs: List[str], offsets: List[object]) -> List[str]:
        """Add the constant per-dim offsets to the per-dim map index expressions.

        :param map_index_exprs: Per-dim element-index expression strings.
        :param offsets: Per-dim constant offsets from :meth:`_const_begin_offsets`.
        :returns: Per-dim index expression strings with non-zero offsets folded in.
        """
        adjusted: List[str] = []
        for d, expr in enumerate(map_index_exprs):
            if d < len(offsets) and offsets[d] != 0:
                adjusted.append(f"(({expr}) + ({symstr(offsets[d])}))")
            else:
                adjusted.append(expr)
        return adjusted

    # ------------------------------------------------------------------
    # Gather / scatter emission helpers
    # ------------------------------------------------------------------

    def _emit_gather_load(self, callsite_stream: PythonCodeIOStream, arr: str, tile_var: str,
                          map_index_exprs: List[str], tile_shape: Tuple[int,
                                                                        ...], cfg: object, state_id: int) -> List[str]:
        """Emit ``ct.gather`` with computed index tiles for non-aligned loads.

        Generates per-dimension index tiles via ``ct.arange`` and
        ``ct.broadcast_to``, then calls ``ct.gather`` to load elements
        at arbitrary global positions.  This handles tiles that don't
        align with ``ct.load``'s implicit grid for examples strided maps
        or maps with non-zero start.

        :param callsite_stream: Code output stream.
        :param arr: Global array variable name.
        :param tile_var: Destination tile variable name.
        :param map_index_exprs: Per-dimension map variable expressions
            (e.g. ``["(2 + __pid0 * 32)"]``).
        :param tile_shape: Tile shape tuple (resolved to ints).
        :param cfg: The control flow graph.
        :param state_id: The state ID.
        :returns: List of index variable names (for reuse by scatter store).
        """
        ndim = len(tile_shape)
        idx_vars: List[str] = []

        for d in range(ndim):
            idx_var = f"__dace_ct_gidx_{tile_var}_{d}"
            callsite_stream.write(f"{idx_var} = {map_index_exprs[d]} + ct.arange({tile_shape[d]}, dtype=ct.int32)", cfg,
                                  state_id)
            idx_vars.append(idx_var)

        if ndim > 1:
            broadcast_vars: List[str] = []
            shape_str = ", ".join(str(s) for s in tile_shape)
            for d, idx_var in enumerate(idx_vars):
                reshape_dims = tuple(tile_shape[d] if i == d else 1 for i in range(ndim))
                reshape_str = ", ".join(str(x) for x in reshape_dims)
                bcast_var = f"{idx_var}_nd"
                callsite_stream.write(
                    f"{bcast_var} = ct.broadcast_to("
                    f"ct.reshape({idx_var}, ({reshape_str},)), "
                    f"({shape_str},))", cfg, state_id)
                broadcast_vars.append(bcast_var)
            indices_str = ", ".join(broadcast_vars)
        else:
            indices_str = idx_vars[0]
            broadcast_vars = idx_vars

        callsite_stream.write(f"{tile_var} = ct.gather({arr}, ({indices_str},), "
                              f"padding_value=0)", cfg, state_id)

        return broadcast_vars if ndim > 1 else idx_vars

    def _emit_scatter_store(self, callsite_stream: PythonCodeIOStream, arr: str, tile_expr: str,
                            gather_idx_vars: List[str], cfg: object, state_id: int) -> None:
        """Emit ``ct.scatter`` with precomputed index tiles.

        Reuses the index tile variables generated by
        :meth:`_emit_gather_load` to scatter tile elements back to
        the global array at the correct positions.

        :param callsite_stream: Code output stream.
        :param arr: Global array variable name.
        :param tile_expr: Tile expression to store.
        :param gather_idx_vars: Index variable names from gather load.
        :param cfg: The control flow graph.
        :param state_id: The state ID.
        """
        indices_str = ", ".join(gather_idx_vars)
        callsite_stream.write(f"ct.scatter({arr}, ({indices_str},), {tile_expr})", cfg, state_id)

    # ------------------------------------------------------------------
    # Node dispatch entry point
    # ------------------------------------------------------------------

    def generate_node(self, sdfg: "SDFG", cfg: object, dfg: object, state_id: int, node: nodes.Node,
                      function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        """Dispatch code generation for a single node inside a CuTile scope.

        :param sdfg: The SDFG.
        :param cfg: The control flow graph.
        :param dfg: The dataflow graph.
        :param state_id: The state ID.
        :param node: The node to generate code for.
        :param function_stream: Stream for function-level code.
        :param callsite_stream: Stream for call-site code.
        :raises NotImplementedError: If there is no handler for the node type.
        """
        method = getattr(self, f"_generate_{type(node).__name__}", None)
        if method is None:
            raise NotImplementedError(f"CuTile backend has no handler for {type(node).__name__}.")
        method(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)

    # ------------------------------------------------------------------
    # MapEntry: PIDs + map variable bindings only
    # ------------------------------------------------------------------

    def _generate_MapEntry(self, sdfg: "SDFG", cfg: object, dfg: object, state_id: int, node: nodes.MapEntry,
                           function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        """Emit block IDs and map variable bindings for a cuTile kernel.

        In the AccessNode-centric design, MapEntry only sets up the
        tile-coordinate PIDs and the element-coordinate map variables.
        Tile loads are handled by each AccessNode individually.

        :param sdfg: The SDFG.
        :param cfg: The control flow graph.
        :param dfg: The dataflow graph.
        :param state_id: The state ID.
        :param node: The MapEntry node.
        :param function_stream: Stream for function-level code.
        :param callsite_stream: Stream for call-site code.
        """
        map_index_exprs = _map_index_exprs(node)

        # Bind __pid{d} for every map dimension. When the grid rank exceeds the
        # cuTile launch-grid cap of 3, the extra dimensions are folded onto
        # axis 0 and recovered here via integer div/mod (see
        # ``_fold_grid_to_launch``); the launch site must fold identically.
        grid_exprs = _grid_exprs_from_map_entry(node)
        _, pid_stmts = _fold_grid_to_launch(grid_exprs)
        for stmt in pid_stmts:
            callsite_stream.write(stmt, cfg, state_id)
        for var, expr in zip(node.map.params, map_index_exprs):
            callsite_stream.write(f"{var} = {expr}", cfg, state_id)

    # ------------------------------------------------------------------
    # MapExit: no-op (stores handled at AccessNode)
    # ------------------------------------------------------------------

    def _generate_MapExit(self, sdfg: "SDFG", cfg: object, dfg: object, state_id: int, node: nodes.MapExit,
                          function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        """No-op -- tile stores are handled at each AccessNode.

        :param sdfg: The SDFG.
        :param cfg: The control flow graph.
        :param dfg: The dataflow graph.
        :param state_id: The state ID.
        :param node: The MapExit node.
        :param function_stream: Stream for function-level code.
        :param callsite_stream: Stream for call-site code.
        """
        pass

    # ------------------------------------------------------------------
    # AccessNode: THE CORE of the AccessNode-centric redesign
    # ------------------------------------------------------------------

    def _emit_scalar_bridge_binding(self, sdfg: "SDFG", state: "SDFGState", node: nodes.AccessNode, cfg: object,
                                    state_id: int, callsite_stream: PythonCodeIOStream) -> None:
        """Bind a Register-storage scalar bridge to its source kernel parameter.

        A loop-invariant scalar (e.g. ``alpha``) that the vectorizer staged via
        :func:`~dace.transformation.passes.vectorization.insert_tile_load_store.stage_constant_access`
        appears inside the cuTile scope as a fresh ``Register`` scalar transient
        (``alpha_const``) fed by an edge from the MapEntry. The scalar itself is
        passed into the kernel as a plain parameter (``alpha``), so the bridge is
        just a rename: emit ``alpha_const = alpha``. Without this the kernel body
        references the undefined ``alpha_const`` and the ``cuda.tile`` compiler
        raises ``Undefined variable alpha_const``.

        When the traced source is an *array* rather than a scalar, the bridge
        stages a single element of it (``stage_constant_access`` with a
        ``src_subset`` like ``aa[0, j]``); binding the bare name would alias
        the whole tensor, so the memlet subset is emitted as an index:
        ``aa_const = aa[0, j]``.

        :param sdfg: The SDFG.
        :param state: The state holding ``node``.
        :param node: The Register-storage scalar bridge AccessNode.
        :param cfg: The control flow graph.
        :param state_id: The state ID.
        :param callsite_stream: Stream for call-site (kernel body) code.
        """
        from dace.codegen.py.utils import data_access_expression

        in_edges = list(state.in_edges(node))
        map_entry_edges = [e for e in in_edges if isinstance(e.src, nodes.MapEntry)]
        if in_edges and not map_entry_edges:
            # A code-node producer (tasklet / nested SDFG) binds the name in its
            # own emission; anything else leaves the bridge undefined in the
            # kernel body -- surface it instead of silently emitting nothing.
            if not any(isinstance(e.src, nodes.CodeNode) for e in in_edges):
                srcs = sorted({type(e.src).__name__ for e in in_edges})
                warnings.warn(f"cuTile codegen: scalar bridge {node.data!r} is fed by {srcs} instead of a "
                              f"MapEntry; no binding emitted (the kernel may reference an undefined name).")
            return
        for in_edge in map_entry_edges:
            outer_edge = self._trace_load_source_edge(state, in_edge.src, in_edge)
            if outer_edge is None:
                warnings.warn(f"cuTile codegen: could not trace the source of scalar bridge {node.data!r} "
                              f"through MapEntry {in_edge.src.map.label!r}; no binding emitted.")
                continue
            src_name = outer_edge.data.data if outer_edge.data else outer_edge.src.data
            src_desc = sdfg.arrays.get(src_name)
            if src_desc is None:
                source_expr = src_name
            elif isinstance(src_desc, data.Scalar):
                src_subset = outer_edge.data.subset if outer_edge.data else None
                source_expr = data_access_expression(src_name, src_desc, src_subset)
            else:
                # Array-element source: cuda.tile arrays are not subscriptable
                # inside kernels ("Use load() or gather()"), and the element may
                # vary per block, so index by the INNER (per-iteration) memlet
                # subset -- the outer subset spans the whole map range.
                inner_subset = in_edge.data.subset if in_edge.data is not None else None
                source_expr = self._scalar_bridge_load_expr(src_name, inner_subset)
                if source_expr is None:
                    warnings.warn(f"cuTile codegen: scalar bridge {node.data!r} stages a non-single-element "
                                  f"subset of array {src_name!r}; no binding emitted.")
                    continue
            if source_expr != node.data:
                callsite_stream.write(f"{node.data} = {source_expr}", cfg, state_id)

    @staticmethod
    def _scalar_bridge_load_expr(src_name: str, subset) -> Optional[str]:
        """``ct.load`` expression for a single staged array element, as a
        ``(1,)`` tile (broadcastable against any tile operand), or None when
        ``subset`` does not select exactly one element per dimension.

        :param src_name: The (kernel-parameter) array name.
        :param subset: The inner-memlet subset selecting the element.
        :returns: The load expression, or None.
        """
        if isinstance(subset, subsets.Indices):
            indices = [symstr(index) for index in subset.indices]
        elif isinstance(subset, subsets.Range):
            if any(sp.simplify(size) != 1 for size in subset.size()):
                return None
            indices = [symstr(start) for start, _end, _step in subset.ranges]
        else:
            return None
        index_str = ', '.join(f'({index})' for index in indices)
        shape_str = ', '.join('1' for _ in indices)
        return f'ct.reshape(ct.load({src_name}, index=({index_str},), shape=({shape_str},)), (1,))'

    def _generate_AccessNode(self, sdfg: "SDFG", cfg: object, dfg: object, state_id: int, node: nodes.AccessNode,
                             function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        """Generate code for an AccessNode inside a cuTile scope.

        Handles four cases:

        1. **Tile loaded from global array** (via MapEntry): emit
           ``ct.load`` / ``ct.gather``.
        2. **Tile written by tasklet**: bind Python variable.
        3. **Tile stored to global array** (via MapExit): emit
           ``ct.store`` / ``ct.scatter``.
        4. **Tile read by tasklet**: no code needed (downstream reads
           the variable).

        :param sdfg: The SDFG.
        :param cfg: The control flow graph.
        :param dfg: The dataflow graph.
        :param state_id: The state ID.
        :param node: The AccessNode.
        :param function_stream: Stream for function-level code.
        :param callsite_stream: Stream for call-site code.
        :raises RuntimeError: For unsupported patterns (e.g. AccessNode -> AccessNode).
        """
        state = cfg.state(state_id)
        desc = sdfg.arrays.get(node.data)

        if desc is None:
            return  # Unknown array, skip

        # Only handle CuTile_Tile storage nodes in the AccessNode-centric path.
        # Non-tile AccessNodes (e.g. global arrays) are outside the scope
        # and connected via MapEntry/MapExit.
        if desc.storage != dtypes.StorageType.CuTile_Tile:
            # A Register-storage scalar bridge (a loop-invariant scalar such as
            # ``alpha`` staged by ``stage_constant_access``) enters the cuTile
            # kernel through the MapEntry, where it is a kernel parameter. Emit
            # the rename ``<bridge> = <scalar_param>`` so the kernel body can
            # read it; without this the tasklet references an undefined
            # ``*_const`` name and the cuda.tile compiler raises
            # ``Undefined variable <name>``.
            if desc.storage == dtypes.StorageType.Register and isinstance(desc, data.Scalar):
                self._emit_scalar_bridge_binding(sdfg, state, node, cfg, state_id, callsite_stream)
            return

        entry = _enclosing_cutile_entry(state, node)
        if entry is None and not self._in_cutile_context(sdfg, state, node):
            raise RuntimeError(f"CuTile_Tile AccessNode {node.data!r} found outside a CuTile scope.")

        # Process incoming edges -- handle loads and variable bindings
        for in_edge in state.in_edges(node):
            src = in_edge.src

            if isinstance(src, nodes.MapEntry):
                # Load from global array through MapEntry
                global_arr = self._trace_load_source(state, src, node, in_edge)
                if global_arr is None:
                    continue

                tile_shape = self._resolve_single_tile_shape(entry, node.data, sdfg)
                self._validate_tile_shape(tile_shape, entry, node.data, sdfg)
                map_index_exprs = _map_index_exprs(entry)
                cutile_index = ", ".join(f"__pid{d}" for d in range(len(entry.map.range)))

                # Fold the outer memlet's constant begin offset (e.g. ``A[1:-1]``
                # -> ``+ 1``) into the element index; a non-zero offset is not
                # block-aligned, so it forces the per-element ``ct.gather`` path.
                begins = self._load_source_begins(state, src, in_edge)
                offsets = self._const_begin_offsets(begins, entry)
                has_offset = any(o != 0 for o in offsets)
                gather_index_exprs = self._apply_index_offsets(map_index_exprs, offsets)

                if has_offset or self._needs_gather_for_tile(entry, tile_shape, sdfg):
                    self._emit_gather_load(callsite_stream, global_arr, node.data, gather_index_exprs, tile_shape, cfg,
                                           state_id)
                else:
                    shape_str = ", ".join(str(s) for s in tile_shape)
                    callsite_stream.write(
                        f"{node.data} = ct.load({global_arr}, "
                        f"index=({cutile_index},), shape=({shape_str},))", cfg, state_id)

            elif isinstance(src, nodes.NestedSDFG):
                # NestedSDFG output -> tile: bind variable. The nested-function
                # call site only assigns the *connector* name (``conn = func(...)``),
                # so the rename to this tile's data name must happen here.
                src_conn = in_edge.src_conn
                if src_conn is not None and src_conn != node.data:
                    callsite_stream.write(f"{node.data} = {src_conn}", cfg, state_id)

            elif isinstance(src, nodes.Tasklet):
                # Tasklet output -> tile: the binding (``tile = conn``) is already
                # emitted by ``_generate_Tasklet``'s output post-bind. Emitting it
                # here too would duplicate the assignment.
                pass

            elif isinstance(src, nodes.AccessNode):
                # Tile-to-tile dataflow is a rename of an immutable value.
                src_desc = sdfg.arrays.get(src.data)
                if (src_desc is not None
                        and src_desc.storage in (dtypes.StorageType.CuTile_Tile, dtypes.StorageType.Register)):
                    # TODO: Add check that shpaes match and we move the full tile
                    if src.data != node.data:
                        callsite_stream.write(f"{node.data} = {src.data}", cfg, state_id)
                else:
                    raise RuntimeError(f"AccessNode-to-AccessNode copy ({src.data!r} -> {node.data!r}) "
                                       f"is not supported in CuTile scope. Use library nodes for copies.")

        # Process outgoing edges -- handle stores
        for out_edge in state.out_edges(node):
            dst = out_edge.dst

            if isinstance(dst, nodes.MapExit):
                # Store tile to global array through MapExit
                global_arr = self._trace_store_target(state, dst, node, out_edge)
                if global_arr is None:
                    continue

                tile_shape = self._resolve_single_tile_shape(entry, node.data, sdfg)
                self._validate_tile_shape(tile_shape, entry, node.data, sdfg)
                map_index_exprs = _map_index_exprs(entry)
                cutile_index = ", ".join(f"__pid{d}" for d in range(len(entry.map.range)))

                # Fold the outer memlet's constant begin offset (e.g. ``B[1:-1]``
                # -> ``+ 1``) into the element index; a non-zero offset is not
                # block-aligned, so it forces the per-element ``ct.scatter`` path.
                begins = self._store_target_begins(state, dst, out_edge)
                offsets = self._const_begin_offsets(begins, entry)
                has_offset = any(o != 0 for o in offsets)
                scatter_index_exprs = self._apply_index_offsets(map_index_exprs, offsets)

                if has_offset or self._needs_gather_for_tile(entry, tile_shape, sdfg):
                    # Build index tiles for scatter
                    idx_vars = self._emit_gather_load(callsite_stream, global_arr, f"__ct_scatter_{node.data}",
                                                      scatter_index_exprs, tile_shape, cfg, state_id)
                    self._emit_scatter_store(callsite_stream, global_arr, node.data, idx_vars, cfg, state_id)
                else:
                    callsite_stream.write(f"ct.store({global_arr}, index=({cutile_index},), "
                                          f"tile={node.data})", cfg, state_id)

            elif isinstance(dst, (nodes.Tasklet, nodes.NestedSDFG)):
                # Tile -> tasklet: no code needed (tasklet reads the variable)
                pass

            elif isinstance(dst, nodes.AccessNode):
                # Tile-to-tile dataflow is a rename emitted at the destination
                # AccessNode's incoming edge; nothing to do here.  Reject only
                # genuinely unsupported cross-storage copies.
                dst_desc = sdfg.arrays.get(dst.data)
                if (dst_desc is None
                        or dst_desc.storage not in (dtypes.StorageType.CuTile_Tile, dtypes.StorageType.Register)):
                    # TODO: Add check that shpaes match and we move the full tile
                    raise RuntimeError(f"AccessNode-to-AccessNode copy ({node.data!r} -> {dst.data!r}) "
                                       f"is not supported in CuTile scope. Use library nodes for copies.")

    # ------------------------------------------------------------------
    # Tasklet
    # ------------------------------------------------------------------

    def _generate_Tasklet(self, sdfg: "SDFG", cfg: object, dfg: object, state_id: int, node: nodes.Tasklet,
                          function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        """Generate code for a Tasklet inside a cuTile scope.

        Binds input connectors from tile variables, emits the tasklet
        body, then binds outputs to downstream tile variables.

        :param sdfg: The SDFG.
        :param cfg: The control flow graph.
        :param dfg: The dataflow graph.
        :param state_id: The state ID.
        :param node: The Tasklet node.
        :param function_stream: Stream for function-level code.
        :param callsite_stream: Stream for call-site code.
        :raises NotImplementedError: If the tasklet uses a non-Python language.
        :raises RuntimeError: If the tasklet is not inside a CuTile scope.
        """
        if node.code.language != dtypes.Language.Python:
            raise NotImplementedError("CuTile backend only supports Python tasklets.")
        state = cfg.state(state_id)
        entry = _enclosing_cutile_entry(state, node)
        if entry is None and not self._in_cutile_context(sdfg, state, node):
            raise RuntimeError("CuTile tasklet handler invoked outside a CuTile scope.")

        if node.instrument != dtypes.InstrumentationType.No_Instrumentation:
            raise RuntimeError("Node-level instrumentation is not supported inside cuTile kernels; "
                               "instrument the enclosing cuTile map (kernel) instead.")

        init_code = codeblock_to_python(node.code_init).strip()
        if init_code:
            self._frame._initcode.write(init_code, sdfg)
        exit_code = codeblock_to_python(node.code_exit).strip()
        if exit_code:
            self._frame._exitcode.write(exit_code, sdfg)

        self._dispatcher.defined_vars.enter_scope(node)
        try:
            # Bind inputs from tile AccessNodes
            for edge in state.in_edges(node):
                if not edge.dst_conn:
                    continue
                rhs: Optional[str] = None
                if isinstance(edge.src, nodes.AccessNode):
                    rhs = edge.src.data
                elif isinstance(edge.src, (nodes.MapEntry, nodes.ConsumeEntry)):
                    # Trace through the scope entries to the root AccessNode.
                    # The memlet path walks ALL enclosing entries (a one-level
                    # connector hop breaks for doubly-nested scopes, and the
                    # connector name may be a stale transient name that
                    # differs from the array actually flowing through).
                    root = state.memlet_path(edge)[0].src
                    if isinstance(root, nodes.AccessNode):
                        rhs = root.data
                    elif edge.data is not None and edge.data.data is not None:
                        rhs = edge.data.data
                    elif edge.src_conn is not None:
                        rhs = edge.src_conn  # fallback
                elif edge.src_conn is not None:
                    rhs = edge.src_conn
                if rhs is None:
                    continue
                callsite_stream.write(f"{edge.dst_conn} = {rhs}", cfg, state_id)
                self._dispatcher.defined_vars.add(edge.dst_conn, dispatcher_mod.DefinedType.Scalar, "object")

            # Pre-bind output connectors that trace to actual (global) arrays.
            # This is needed for in-place operations like ct.scatter(_dst, ...)
            # where _dst must be bound to the global array before the body runs.
            # We must NOT pre-bind outputs that go to tile-local AccessNodes
            # (Register or CuTile_Tile storage) because those variables don't
            # exist yet -- they are created by the tasklet body.
            _LOCAL_STORAGES = {
                dtypes.StorageType.Register,
                dtypes.StorageType.CuTile_Tile,
            }
            _prebind_outputs: set = set()
            for edge in state.out_edges(node):
                if not edge.src_conn:
                    continue
                dst_name: Optional[str] = None
                if isinstance(edge.dst, nodes.AccessNode):
                    # Check if this is a local tile variable -- skip pre-bind
                    dst_desc = sdfg.arrays.get(edge.dst.data)
                    if dst_desc is not None and dst_desc.storage in _LOCAL_STORAGES:
                        continue
                    dst_name = edge.dst.data
                elif isinstance(edge.dst, (nodes.MapExit, nodes.ConsumeExit)):
                    # Trace through ALL enclosing scope exits (see input
                    # binding above for why a one-level hop is insufficient).
                    leaf = state.memlet_path(edge)[-1].dst
                    if isinstance(leaf, nodes.AccessNode):
                        dst_name = leaf.data
                if dst_name and dst_name != edge.src_conn:
                    callsite_stream.write(f"{edge.src_conn} = {dst_name}", cfg, state_id)
                    self._dispatcher.defined_vars.add(edge.src_conn, dispatcher_mod.DefinedType.Scalar, "object")
                    _prebind_outputs.add(edge.src_conn)

            # Emit tasklet body
            callsite_stream.write(f"\n####### Tasklet: {node.label}\n\n", cfg, state_id)
            callsite_stream.write(codeblock_to_python(node.code).strip() or "pass")
            callsite_stream.write(f"\n####### End of tasklet: {node.label}\n\n", cfg, state_id)

            # Bind outputs to downstream tile AccessNodes or through MapExit.
            # Skip connectors that were already pre-bound above — the in-place
            # operation (e.g. ct.scatter) already modified the array directly.
            for edge in state.out_edges(node):
                if not edge.src_conn:
                    continue
                if isinstance(edge.dst, nodes.AccessNode):
                    if edge.dst.data == edge.src_conn:
                        continue
                    if edge.src_conn in _prebind_outputs:
                        continue
                    callsite_stream.write(f"{edge.dst.data} = {edge.src_conn}", cfg, state_id)
                    self._dispatcher.defined_vars.add(edge.dst.data, dispatcher_mod.DefinedType.Scalar, "object")
                elif isinstance(edge.dst, (nodes.MapExit, nodes.ConsumeExit)):
                    # Trace through ALL enclosing scope exits to the actual
                    # destination AccessNode (see input binding above for why
                    # a one-level connector hop is insufficient).
                    leaf = state.memlet_path(edge)[-1].dst
                    if isinstance(leaf, nodes.AccessNode):
                        dst_name = leaf.data
                        if dst_name != edge.src_conn and edge.src_conn not in _prebind_outputs:
                            callsite_stream.write(f"{dst_name} = {edge.src_conn}", cfg, state_id)
                            self._dispatcher.defined_vars.add(dst_name, dispatcher_mod.DefinedType.Scalar, "object")
        finally:
            self._dispatcher.defined_vars.exit_scope(node)

    # ------------------------------------------------------------------
    # NestedSDFG — emitted as a module-level function
    # ------------------------------------------------------------------
    #
    # A user helper function compiled by DaCe becomes a NestedSDFG.  It is
    # emitted as a plain module-level Python function that the cuTile kernel
    # (or an enclosing nested function) calls.  The body is generated by the
    # SAME shared node dispatcher used for the kernel body, so every tile op
    # routes back to ``_generate_Tasklet`` / ``_generate_AccessNode`` — there
    # is no bespoke node walking here.  The boundary follows cuTile's value
    # model (https://docs.nvidia.com/cuda/cutile-python/execution.html):
    #
    #   * Tiles (and registers) are immutable, so tile-valued outputs are
    #     *returned* from the function.
    #   * Global arrays are read/write views, so global-array-valued outputs
    #     are passed in as destination parameters and written in place
    #     (``ct.store`` / ``ct.scatter``); they are not returned.
    #
    # The inner (map-less) nodes are recognised as living in a cuTile scope
    # structurally, via :meth:`_sdfg_is_cutile_body`.

    #: Inner-SDFG storage types whose values are immutable Python locals
    #: (returned from the generated function rather than written in place).
    _RETURNED_OUTPUT_STORAGES = frozenset({
        dtypes.StorageType.CuTile_Tile,
        dtypes.StorageType.Register,
    })

    def _is_returned_output(self, inner_sdfg: "SDFG", conn: str) -> bool:
        """Whether an output connector is an immutable value (returned) rather
        than a global-memory destination view (passed in, written in place).

        :param inner_sdfg: The nested SDFG.
        :param conn: The output connector name (== inner array name).
        :returns: ``True`` if the value must be returned from the function.
        """
        desc = inner_sdfg.arrays.get(conn)
        return desc is not None and desc.storage in self._RETURNED_OUTPUT_STORAGES

    def _generate_NestedSDFG(self, sdfg: "SDFG", cfg: object, dfg: object, state_id: int, node: nodes.NestedSDFG,
                             function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        """Emit a module-level function for a NestedSDFG and call it.

        :param sdfg: The containing SDFG.
        :param cfg: The containing control-flow region.
        :param dfg: The dataflow graph (unused; kept for the dispatcher signature).
        :param state_id: The containing state ID.
        :param node: The NestedSDFG node.
        :param function_stream: Stream for module-level code (the function def).
        :param callsite_stream: Stream for the call site.
        """
        state = cfg.state(state_id)
        inner_sdfg = node.sdfg

        # Lower any tile-op library nodes to their cuTile tasklets first.
        inner_sdfg.expand_library_nodes(recursive=True)

        func_name = (f"__dace_nested_{inner_sdfg.name}_{cfg.cfg_id}_"
                     f"{state_id}_{state.node_id(node)}")

        input_conns: List[str] = sorted({e.dst_conn for e in state.in_edges(node) if e.dst_conn is not None})
        output_conns: List[str] = sorted({e.src_conn for e in state.out_edges(node) if e.src_conn is not None})
        returned_outputs = [c for c in output_conns if self._is_returned_output(inner_sdfg, c)]
        dest_outputs = [c for c in output_conns if not self._is_returned_output(inner_sdfg, c)]
        symbol_names = self._nsdfg_runtime_symbols(node)

        # Parameters: inputs, then global-array destinations, then symbols.
        param_conns = list(dict.fromkeys(input_conns + dest_outputs))
        params = param_conns + symbol_names

        if func_name not in self._generated_nested_functions:
            self._emit_nsdfg_function(inner_sdfg, func_name, params, returned_outputs, function_stream)
            self._generated_nested_functions[func_name] = func_name

        # --- Call site ---
        # Mirror the deduplicated ``param_conns`` order so an in-out array
        # (a connector that is both an input and a destination output) is
        # passed exactly once, matching the function's parameter list.
        input_conn_set = set(input_conns)
        call_args: List[str] = [
            self._resolve_nsdfg_input_var(state, node, c) if c in input_conn_set else self._resolve_nsdfg_output_var(
                state, node, c) for c in param_conns
        ]
        for sym_name in symbol_names:
            mapping_expr = node.symbol_mapping.get(sym_name)
            call_args.append(symstr(mapping_expr) if mapping_expr is not None else sym_name)
        args_str = ", ".join(call_args)

        if returned_outputs:
            lhs = ", ".join(returned_outputs)
            callsite_stream.write(f"{lhs} = {func_name}({args_str})", cfg, state_id)
            for c in returned_outputs:
                self._dispatcher.defined_vars.add(c, dispatcher_mod.DefinedType.Scalar, "object")
        else:
            callsite_stream.write(f"{func_name}({args_str})", cfg, state_id)

    def _emit_nsdfg_function(self, inner_sdfg: "SDFG", func_name: str, params: List[str], returned_outputs: List[str],
                             function_stream: PythonCodeIOStream) -> None:
        """Write the module-level function definition for a NestedSDFG.

        The body is generated through the shared node dispatcher — every node
        routes back to the cuTile per-node handlers, since
        :meth:`_sdfg_is_cutile_body` recognises *inner_sdfg* as a cuTile body.

        :param inner_sdfg: The nested SDFG.
        :param func_name: The generated function name.
        :param params: Ordered parameter names (inputs, destinations, symbols).
        :param returned_outputs: Output connectors returned (immutable values).
        :param function_stream: Stream to append the function definition to.
        """
        body_stream = PythonCodeIOStream()

        def dispatch_state(inner_state: "SDFGState") -> str:
            tmp = PythonCodeIOStream()
            self._dispatcher.dispatch_subgraph(inner_sdfg,
                                               inner_state.parent_graph,
                                               inner_state,
                                               inner_state.block_id,
                                               function_stream,
                                               tmp,
                                               skip_entry_node=False)
            return tmp.getvalue()

        py_cflow.control_flow_region_to_code(inner_sdfg, dispatch_state, self._frame, inner_sdfg.symbols, body_stream)

        if returned_outputs:
            body_stream.write(f"return {', '.join(returned_outputs)}")

        function_stream.write("")
        function_stream.write(f"def {func_name}({', '.join(params)}):")
        with function_stream.indented():
            body_code = body_stream.getvalue().rstrip("\n")
            function_stream.write(body_code if body_code.strip() else "pass")
        function_stream.write("")

    # ------------------------------------------------------------------
    # NestedSDFG helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _nsdfg_runtime_symbols(node: nodes.NestedSDFG) -> List[str]:
        """Return sorted list of symbol names that must be passed at runtime.

        Symbols that are free in the inner SDFG and not constants are
        included.

        :param node: The NestedSDFG node.
        :returns: Sorted list of symbol names.
        """
        inner_sdfg = node.sdfg
        free_symbols = set(str(s) for s in inner_sdfg.used_symbols(all_symbols=False, keep_defined_in_mapping=True))
        return [
            sym_name for sym_name in sorted(node.symbol_mapping.keys())
            if sym_name in free_symbols and sym_name not in inner_sdfg.constants
        ]

    @staticmethod
    def _resolve_nsdfg_input_var(state: "SDFGState", node: nodes.NestedSDFG, conn_name: str) -> str:
        """Resolve the variable name for a NestedSDFG input connector at the call site.

        Traces edges to find the actual source variable name.

        :param state: The containing state.
        :param node: The NestedSDFG node.
        :param conn_name: The input connector name.
        :returns: The variable name to pass as argument.
        """
        for edge in state.in_edges(node):
            if edge.dst_conn == conn_name:
                if isinstance(edge.src, nodes.AccessNode):
                    return edge.src.data
                elif edge.src_conn is not None:
                    # Through a MapEntry — trace to the outer edge.
                    if isinstance(edge.src, nodes.MapEntry):
                        outer_conn = _matching_outer_connector(edge.src_conn)
                        for outer_edge in state.in_edges_by_connector(edge.src, outer_conn):
                            if isinstance(outer_edge.src, nodes.AccessNode):
                                return outer_edge.src.data
                    return edge.src_conn
        return conn_name  # fallback: use connector name itself

    @staticmethod
    def _resolve_nsdfg_output_var(state: "SDFGState", node: nodes.NestedSDFG, conn_name: str) -> str:
        """Resolve the variable name for a NestedSDFG output connector at the call site.

        Traces edges to find the actual destination variable name.

        :param state: The containing state.
        :param node: The NestedSDFG node.
        :param conn_name: The output connector name.
        :returns: The variable name to assign the return value to.
        """
        for edge in state.out_edges(node):
            if edge.src_conn == conn_name:
                if isinstance(edge.dst, nodes.AccessNode):
                    return edge.dst.data
                elif edge.dst_conn is not None:
                    # Through a MapExit — trace to the outer edge.
                    if isinstance(edge.dst, nodes.MapExit):
                        outer_conn = _matching_inner_connector(edge.dst_conn)
                        for outer_edge in state.out_edges_by_connector(edge.dst, outer_conn):
                            if isinstance(outer_edge.dst, nodes.AccessNode):
                                return outer_edge.dst.data
                    return edge.dst_conn
        return conn_name  # fallback: use connector name itself

    # ------------------------------------------------------------------
    # Scope generation (kernel wrapper + launch)
    # ------------------------------------------------------------------

    def generate_scope(self, sdfg: "SDFG", cfg: object, dfg_scope: object, state_id: int,
                       function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        """Generate the cuTile kernel wrapper and launch call for a map scope.

        Emits a ``@ct.kernel``-decorated function containing the scope body,
        then emits a ``ct.launch(...)`` call at the call site.

        :param sdfg: The SDFG.
        :param cfg: The control flow graph.
        :param dfg_scope: The scope subgraph view.
        :param state_id: The state ID.
        :param function_stream: Stream for function-level code.
        :param callsite_stream: Stream for call-site code.
        :raises ValueError: If the scope source is not a MapEntry.
        """
        entry = dfg_scope.source_nodes()[0]
        if not isinstance(entry, nodes.MapEntry):
            raise ValueError("CuTilePythonCodeGen expects a map scope")

        state = cfg.state(state_id)
        exit_node = state.exit_node(entry)
        grid_exprs = _grid_exprs_from_map_entry(entry)

        input_arrays = _ordered_unique(e.data.data for e in state.in_edges(entry)
                                       if e.data and e.data.data and isinstance(e.src, nodes.AccessNode))
        output_arrays = _ordered_unique(e.data.data for e in state.out_edges(exit_node)
                                        if e.data and e.data.data and isinstance(e.dst, nodes.AccessNode))
        free_syms = _collect_free_symbols(entry, dfg_scope, sdfg)
        kernel_params = list(dict.fromkeys(input_arrays + output_arrays + free_syms))

        kernel_name = (f"__dace_cutile_{sdfg.name}_{cfg.cfg_id}_"
                       f"{state.block_id}_{state.node_id(entry)}")

        kernel_stream = PythonCodeIOStream()
        kernel_stream.write("@ct.kernel")
        kernel_stream.write(f"def {kernel_name}({', '.join(kernel_params)}):")
        with kernel_stream.indented():
            # Emit MapEntry (pid setup) ourselves; the dispatcher's
            # topological walk treats MapEntry specially (dispatch_scope), so
            # we cannot rely on dispatch_subgraph to invoke our handler for it.
            self.generate_node(sdfg, cfg, dfg_scope, state_id, entry, function_stream, kernel_stream)
            # Walk the rest of the scope. Tasklets, MapExit, AccessNodes, and
            # NestedSDFGs are routed to our predicated handlers.
            self._dispatcher.dispatch_subgraph(
                sdfg,
                cfg,
                dfg_scope,
                state_id,
                function_stream,
                kernel_stream,
                skip_entry_node=True,
            )

        function_stream.write("")
        function_stream.write(kernel_stream.getvalue())
        function_stream.write("")

        # The cuTile launch grid is capped at 3 axes by the runtime. Grids with
        # more than 3 tiled dimensions are folded onto the 3 available axes (the
        # kernel recovers per-dim block IDs via div/mod in ``_generate_MapEntry``,
        # using the same ``_fold_grid_to_launch`` layout).
        launch_dims, _ = _fold_grid_to_launch(grid_exprs)
        grid_tuple = f"({', '.join(launch_dims)})"
        deduped_arrays = list(dict.fromkeys(input_arrays + output_arrays))
        launch_args = ([_array_runtime_name(sdfg, n) for n in deduped_arrays] + free_syms)
        args_tuple = (f"({', '.join(launch_args)},)" if len(launch_args) == 1 else f"({', '.join(launch_args)})")
        instrumented = (entry.map.instrument != dtypes.InstrumentationType.No_Instrumentation)

        # Instrumentation: kernel-scope begin (before launch)
        if instrumented:
            for instr in self._dispatcher.instrumentation.values():
                if instr is not None:
                    instr.on_scope_entry(sdfg, cfg, state, entry, callsite_stream, callsite_stream, function_stream)

        # Zero-trip maps (e.g. a loop-dependent range ``0:i`` at ``i == 0``)
        # yield a grid dimension of 0, which the cuTile runtime rejects
        # ("invalid argument"); the launch is a no-op then, so skip it.
        callsite_stream.write(
            f"if 0 not in {grid_tuple}: ct.launch(cupy.cuda.get_current_stream(), {grid_tuple}, "
            f"{kernel_name}, {args_tuple})",
            cfg,
            state_id,
        )

        # Always synchronize so the kernel completes before the host
        # continues (and before any timing measurement ends).
        callsite_stream.write("cupy.cuda.get_current_stream().synchronize()", cfg, state_id)

        # Instrumentation: kernel-scope end (after synchronize). The exit node
        # is passed so the provider resolves the matching entry node (and thus
        # the matching timer-variable id) via ``state.entry_node(exit)``.
        if instrumented:
            for instr in self._dispatcher.instrumentation.values():
                if instr is not None:
                    instr.on_scope_exit(sdfg, cfg, state, exit_node, callsite_stream, callsite_stream, function_stream)
