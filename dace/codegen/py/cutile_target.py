"""Schedule-based cuTile Python code generation target.

AccessNode-centric design: MapEntry emits only PIDs and map variable
bindings; each CuTile_Tile AccessNode handles its own ``ct.load`` /
``ct.store`` (or ``ct.gather`` / ``ct.scatter`` for non-aligned
tiles).  MapExit is a no-op.
"""

import warnings
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

import sympy as sp

from dace import dtypes, registry, subsets
import dace.codegen.dispatcher as dispatcher_mod
from dace.codegen.py.framecode import codeblock_to_python
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.codegen.py.target import PythonTargetCodeGenerator
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
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


# ---------------------------------------------------------------------------
# Connector helpers
# ---------------------------------------------------------------------------

def _matching_inner_connector(outer_conn: str) -> str:
    """Convert an outer (input) scope connector name to the matching inner (output) name.

    :param outer_conn: The outer connector name, e.g. ``"IN_A"``.
    :returns: The matching inner connector, e.g. ``"OUT_A"``.
    :raises ValueError: If *outer_conn* does not start with the expected prefix.
    """
    if not outer_conn.startswith(_SCOPE_IN_PREFIX):
        raise ValueError(
            f"Expected connector starting with {_SCOPE_IN_PREFIX!r}, got {outer_conn!r}")
    return _SCOPE_OUT_PREFIX + outer_conn[len(_SCOPE_IN_PREFIX):]


def _matching_outer_connector(inner_conn: str) -> str:
    """Convert an inner (output) scope connector name to the matching outer (input) name.

    :param inner_conn: The inner connector name, e.g. ``"OUT_A"``.
    :returns: The matching outer connector, e.g. ``"IN_A"``.
    :raises ValueError: If *inner_conn* does not start with the expected prefix.
    """
    if not inner_conn.startswith(_SCOPE_OUT_PREFIX):
        raise ValueError(
            f"Expected connector starting with {_SCOPE_OUT_PREFIX!r}, got {inner_conn!r}")
    return _SCOPE_IN_PREFIX + inner_conn[len(_SCOPE_OUT_PREFIX):]


# ---------------------------------------------------------------------------
# Existing helper functions (kept with type hint updates)
# ---------------------------------------------------------------------------

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
    """Return per-dimension grid size expressions from a map entry.

    :param entry: The map entry node.
    :returns: List of symbolic expressions for the grid dimensions.
    """
    return [symstr(s) for s in entry.map.range.size()]


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
    seen: Dict[str, None] = {}
    for x in items:
        if x not in seen:
            seen[x] = None
    return sorted(list(seen.keys()))


def _collect_free_symbols(entry: nodes.MapEntry, dfg_scope: object,
                          sdfg: "SDFG") -> List[str]:
    """Collect free symbols used in a map scope.

    Returns symbols that appear in the map range or memlet subsets and
    are declared in the SDFG's symbol table (but not constants).

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
    syms = {s for s in syms if s in sdfg.symbols and s not in sdfg.constants}
    return sorted(syms)


def _enclosing_cutile_entry(state: "SDFGState",
                            node: nodes.Node) -> Optional[nodes.MapEntry]:
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

    def __init__(self, frame_codegen: "DaCePythonCodeGenerator",
                 sdfg: "SDFG") -> None:
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher
        # Register as the handler for CuTile map scopes.
        self._dispatcher.register_map_dispatcher(dtypes.ScheduleType.CuTile, self)
        # Register as node handler for all nodes inside CuTile scopes.
        self._dispatcher.register_node_dispatcher(
            self, predicate=self._is_in_cutile_scope)
        # Register copy dispatchers for transfers involving CuTile_Tile storage.
        # The actual loads/stores are emitted by _generate_AccessNode, not by
        # copy_memory, but the dispatcher's target-collection phase needs to
        # find a handler for every memlet-tree edge pair.  Register for all
        # storage types that can appear in SDFG edges touching tile transients.
        _tile_peer_storages = [
            dtypes.StorageType.GPU_Global,
            dtypes.StorageType.CPU_Heap,
            dtypes.StorageType.CPU_Pinned,
            dtypes.StorageType.Register,
        ]
        _copy_schedules = [dtypes.ScheduleType.CuTile, None]
        for peer_storage in _tile_peer_storages:
            for sched in _copy_schedules:
                self._dispatcher.register_copy_dispatcher(
                    peer_storage, dtypes.StorageType.CuTile_Tile, sched, self)
                self._dispatcher.register_copy_dispatcher(
                    dtypes.StorageType.CuTile_Tile, peer_storage, sched, self)
        # Also register tile-to-tile copies within a CuTile scope.
        for sched in _copy_schedules:
            self._dispatcher.register_copy_dispatcher(
                dtypes.StorageType.CuTile_Tile, dtypes.StorageType.CuTile_Tile,
                sched, self)
        # Register array dispatcher for CuTile_Tile storage (allocation is a no-op).
        self._dispatcher.register_array_dispatcher(
            dtypes.StorageType.CuTile_Tile, self)

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

    @staticmethod
    def _is_in_cutile_scope(sdfg: "SDFG", state: "SDFGState",
                            node: nodes.Node) -> bool:
        """Predicate for the node dispatcher: return True for CuTile nodes.

        :param sdfg: The SDFG.
        :param state: The SDFG state.
        :param node: The node to check.
        :returns: ``True`` if the node belongs to a CuTile scope.
        """
        return _is_cutile_node(state, node)

    # ------------------------------------------------------------------
    # Array dispatcher methods (no-ops for tile arrays)
    # ------------------------------------------------------------------

    def allocate_array(self, sdfg: "SDFG", cfg: object, dfg: object,
                       state_id: int, node: nodes.Node, nodedesc: object,
                       global_stream: PythonCodeIOStream,
                       declaration_stream: PythonCodeIOStream,
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

    def deallocate_array(self, sdfg: "SDFG", cfg: object, dfg: object,
                         state_id: int, node: nodes.Node, nodedesc: object,
                         function_stream: PythonCodeIOStream,
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

    def declare_array(self, sdfg: "SDFG", cfg: object, dfg: object,
                      state_id: int, node: nodes.Node, nodedesc: object,
                      global_stream: PythonCodeIOStream,
                      declaration_stream: PythonCodeIOStream) -> None:
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

    def copy_memory(self, sdfg: "SDFG", cfg: object, dfg: object,
                    state_id: int, src_node: nodes.Node,
                    dst_node: nodes.Node, edge: object,
                    function_stream: PythonCodeIOStream,
                    callsite_stream: PythonCodeIOStream) -> None:
        """Handle copy operations between global arrays and tile arrays.

        For CuTile_Tile arrays inside a scope, data flows through
        MapEntry/MapExit boundaries and is handled by
        :meth:`_generate_AccessNode`.  This method is a fallback for
        any remaining direct AccessNode-to-AccessNode edges that the
        dispatcher routes here.

        Uses numpy-compatible assignment (works with cupy arrays on GPU).

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
        if not isinstance(src_node, nodes.AccessNode) or not isinstance(
                dst_node, nodes.AccessNode):
            raise NotImplementedError(
                f"CuTile copy_memory only supports AccessNode-to-AccessNode "
                f"copies, got {type(src_node).__name__} -> "
                f"{type(dst_node).__name__}")

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
    def _needs_gather_for_tile(entry: nodes.MapEntry,
                               tile_shape: Tuple[int, ...],
                               sdfg: "SDFG") -> bool:
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
    def _resolve_single_tile_shape(entry: nodes.MapEntry, tile_name: str,
                                   sdfg: "SDFG") -> Tuple[int, ...]:
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

        _tile_subs = {sp.Symbol(p): r[0]
                      for p, r in zip(entry.map.params, entry.map.range)}
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
    def _validate_tile_shape(tile_shape: Tuple[int, ...],
                             entry: nodes.MapEntry,
                             tile_name: str,
                             sdfg: "SDFG") -> None:
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
                raise RuntimeError(
                    f"Tile {tile_name!r} has non-positive dimension {dim} "
                    f"at axis {d}. All tile dimensions must be positive.")

        # 2. Tile dimensionality must not exceed map dimensionality.
        map_ndim = len(entry.map.range)
        if len(tile_shape) > map_ndim:
            raise RuntimeError(
                f"Tile {tile_name!r} has {len(tile_shape)} dimensions but "
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
    def _trace_load_source(state: "SDFGState", entry: nodes.MapEntry,
                           tile_node: nodes.AccessNode,
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
        src_conn = in_edge.src_conn
        if src_conn is None or not src_conn.startswith(_SCOPE_OUT_PREFIX):
            return None
        outer_conn = _matching_outer_connector(src_conn)
        for outer_edge in state.in_edges_by_connector(entry, outer_conn):
            if isinstance(outer_edge.src, nodes.AccessNode):
                return outer_edge.data.data if outer_edge.data else outer_edge.src.data
        return None

    @staticmethod
    def _trace_store_target(state: "SDFGState", exit_node: nodes.MapExit,
                            tile_node: nodes.AccessNode,
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

    # ------------------------------------------------------------------
    # Gather / scatter emission helpers
    # ------------------------------------------------------------------

    def _emit_gather_load(self, callsite_stream: PythonCodeIOStream,
                          arr: str, tile_var: str,
                          map_index_exprs: List[str],
                          tile_shape: Tuple[int, ...],
                          cfg: object, state_id: int) -> List[str]:
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
            callsite_stream.write(
                f"{idx_var} = {map_index_exprs[d]} + ct.arange({tile_shape[d]}, dtype=ct.int32)",
                cfg, state_id)
            idx_vars.append(idx_var)

        if ndim > 1:
            broadcast_vars: List[str] = []
            shape_str = ", ".join(str(s) for s in tile_shape)
            for d, idx_var in enumerate(idx_vars):
                reshape_dims = tuple(
                    tile_shape[d] if i == d else 1
                    for i in range(ndim))
                reshape_str = ", ".join(str(x) for x in reshape_dims)
                bcast_var = f"{idx_var}_nd"
                callsite_stream.write(
                    f"{bcast_var} = ct.broadcast_to("
                    f"ct.reshape({idx_var}, ({reshape_str},)), "
                    f"({shape_str},))",
                    cfg, state_id)
                broadcast_vars.append(bcast_var)
            indices_str = ", ".join(broadcast_vars)
        else:
            indices_str = idx_vars[0]
            broadcast_vars = idx_vars

        callsite_stream.write(
            f"{tile_var} = ct.gather({arr}, ({indices_str},), "
            f"padding_value=0)",
            cfg, state_id)

        return broadcast_vars if ndim > 1 else idx_vars

    def _emit_scatter_store(self, callsite_stream: PythonCodeIOStream,
                            arr: str, tile_expr: str,
                            gather_idx_vars: List[str],
                            cfg: object, state_id: int) -> None:
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
        callsite_stream.write(
            f"ct.scatter({arr}, ({indices_str},), {tile_expr})",
            cfg, state_id)

    # ------------------------------------------------------------------
    # Node dispatch entry point
    # ------------------------------------------------------------------

    def generate_node(self, sdfg: "SDFG", cfg: object, dfg: object,
                      state_id: int, node: nodes.Node,
                      function_stream: PythonCodeIOStream,
                      callsite_stream: PythonCodeIOStream) -> None:
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
            raise NotImplementedError(
                f"CuTile backend has no handler for {type(node).__name__}.")
        method(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)

    # ------------------------------------------------------------------
    # MapEntry: PIDs + map variable bindings only
    # ------------------------------------------------------------------

    def _generate_MapEntry(self, sdfg: "SDFG", cfg: object, dfg: object,
                           state_id: int, node: nodes.MapEntry,
                           function_stream: PythonCodeIOStream,
                           callsite_stream: PythonCodeIOStream) -> None:
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

        for d in range(len(node.map.range)):
            callsite_stream.write(f"__pid{d} = ct.bid({d})", cfg, state_id)
        for var, expr in zip(node.map.params, map_index_exprs):
            callsite_stream.write(f"{var} = {expr}", cfg, state_id)

    # ------------------------------------------------------------------
    # MapExit: no-op (stores handled at AccessNode)
    # ------------------------------------------------------------------

    def _generate_MapExit(self, sdfg: "SDFG", cfg: object, dfg: object,
                          state_id: int, node: nodes.MapExit,
                          function_stream: PythonCodeIOStream,
                          callsite_stream: PythonCodeIOStream) -> None:
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

    def _generate_AccessNode(self, sdfg: "SDFG", cfg: object, dfg: object,
                             state_id: int, node: nodes.AccessNode,
                             function_stream: PythonCodeIOStream,
                             callsite_stream: PythonCodeIOStream) -> None:
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
            return

        entry = _enclosing_cutile_entry(state, node)
        if entry is None:
            raise RuntimeError(
                f"CuTile_Tile AccessNode {node.data!r} found outside a CuTile scope.")

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
                cutile_index = ", ".join(
                    f"__pid{d}" for d in range(len(entry.map.range)))

                if self._needs_gather_for_tile(entry, tile_shape, sdfg):
                    self._emit_gather_load(
                        callsite_stream, global_arr, node.data,
                        map_index_exprs, tile_shape, cfg, state_id)
                else:
                    shape_str = ", ".join(str(s) for s in tile_shape)
                    callsite_stream.write(
                        f"{node.data} = ct.load({global_arr}, "
                        f"index=({cutile_index},), shape=({shape_str},))",
                        cfg, state_id)

            elif isinstance(src, (nodes.Tasklet, nodes.NestedSDFG)):
                # Tasklet/NestedSDFG output -> tile: bind variable
                src_conn = in_edge.src_conn
                if src_conn is not None and src_conn != node.data:
                    callsite_stream.write(
                        f"{node.data} = {src_conn}", cfg, state_id)

            elif isinstance(src, nodes.AccessNode):
                raise RuntimeError(
                    f"AccessNode-to-AccessNode copy ({src.data!r} -> {node.data!r}) "
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
                cutile_index = ", ".join(
                    f"__pid{d}" for d in range(len(entry.map.range)))

                if self._needs_gather_for_tile(entry, tile_shape, sdfg):
                    # Build index tiles for scatter
                    idx_vars = self._emit_gather_load(
                        callsite_stream, global_arr,
                        f"__ct_scatter_{node.data}",
                        map_index_exprs, tile_shape, cfg, state_id)
                    self._emit_scatter_store(
                        callsite_stream, global_arr, node.data,
                        idx_vars, cfg, state_id)
                else:
                    callsite_stream.write(
                        f"ct.store({global_arr}, index=({cutile_index},), "
                        f"tile={node.data})",
                        cfg, state_id)

            elif isinstance(dst, (nodes.Tasklet, nodes.NestedSDFG)):
                # Tile -> tasklet: no code needed (tasklet reads the variable)
                pass

            elif isinstance(dst, nodes.AccessNode):
                raise RuntimeError(
                    f"AccessNode-to-AccessNode copy ({node.data!r} -> {dst.data!r}) "
                    f"is not supported in CuTile scope. Use library nodes for copies.")

    # ------------------------------------------------------------------
    # Tasklet
    # ------------------------------------------------------------------

    def _generate_Tasklet(self, sdfg: "SDFG", cfg: object, dfg: object,
                          state_id: int, node: nodes.Tasklet,
                          function_stream: PythonCodeIOStream,
                          callsite_stream: PythonCodeIOStream) -> None:
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
            raise NotImplementedError(
                "CuTile backend only supports Python tasklets.")
        state = cfg.state(state_id)
        entry = _enclosing_cutile_entry(state, node)
        if entry is None:
            raise RuntimeError(
                "CuTile tasklet handler invoked outside a CuTile scope.")

        if node.instrument != dtypes.InstrumentationType.No_Instrumentation:
            raise RuntimeError(
                "Node-level instrumentation is not supported inside cuTile kernels; "
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
                    # Trace through the scope entry to find the actual array name.
                    # The inner connector (e.g. "OUT_A") maps to the outer
                    # connector ("IN_A") which receives from the real AccessNode.
                    inner_conn = edge.src_conn  # e.g., "OUT_A"
                    outer_conn = _matching_outer_connector(inner_conn) if inner_conn else None
                    if outer_conn:
                        for outer_edge in state.in_edges_by_connector(edge.src, outer_conn):
                            if isinstance(outer_edge.src, nodes.AccessNode):
                                rhs = outer_edge.src.data
                                break
                    if rhs is None and edge.src_conn is not None:
                        rhs = edge.src_conn  # fallback
                elif edge.src_conn is not None:
                    rhs = edge.src_conn
                if rhs is None:
                    continue
                callsite_stream.write(
                    f"{edge.dst_conn} = {rhs}", cfg, state_id)
                self._dispatcher.defined_vars.add(
                    edge.dst_conn, dispatcher_mod.DefinedType.Scalar, "object")

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
                    inner_conn = edge.dst_conn
                    outer_conn = _matching_inner_connector(inner_conn) if inner_conn else None
                    if outer_conn:
                        for outer_edge in state.out_edges_by_connector(edge.dst, outer_conn):
                            if isinstance(outer_edge.dst, nodes.AccessNode):
                                dst_name = outer_edge.dst.data
                                break
                if dst_name and dst_name != edge.src_conn:
                    callsite_stream.write(
                        f"{edge.src_conn} = {dst_name}", cfg, state_id)
                    self._dispatcher.defined_vars.add(
                        edge.src_conn, dispatcher_mod.DefinedType.Scalar, "object")
                    _prebind_outputs.add(edge.src_conn)

            # Emit tasklet body
            callsite_stream.write(
                f"\n####### Tasklet: {node.label}\n\n", cfg, state_id)
            callsite_stream.write(
                codeblock_to_python(node.code).strip() or "pass")
            callsite_stream.write(
                f"\n####### End of tasklet: {node.label}\n\n", cfg, state_id)

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
                    callsite_stream.write(
                        f"{edge.dst.data} = {edge.src_conn}", cfg, state_id)
                    self._dispatcher.defined_vars.add(
                        edge.dst.data, dispatcher_mod.DefinedType.Scalar,
                        "object")
                elif isinstance(edge.dst, (nodes.MapExit, nodes.ConsumeExit)):
                    # Trace through the scope exit to find the actual
                    # destination array.  The inner connector (e.g. "IN_C")
                    # maps to the outer connector ("OUT_C") which feeds
                    # the real AccessNode.
                    inner_conn = edge.dst_conn  # e.g., "IN_C"
                    outer_conn = _matching_inner_connector(inner_conn) if inner_conn else None
                    if outer_conn:
                        for outer_edge in state.out_edges_by_connector(edge.dst, outer_conn):
                            if isinstance(outer_edge.dst, nodes.AccessNode):
                                dst_name = outer_edge.dst.data
                                if dst_name != edge.src_conn:
                                    if edge.src_conn in _prebind_outputs:
                                        break
                                    callsite_stream.write(
                                        f"{dst_name} = {edge.src_conn}", cfg, state_id)
                                    self._dispatcher.defined_vars.add(
                                        dst_name, dispatcher_mod.DefinedType.Scalar,
                                        "object")
                                break
        finally:
            self._dispatcher.defined_vars.exit_scope(node)

    # ------------------------------------------------------------------
    # NestedSDFG
    # ------------------------------------------------------------------

    def _generate_NestedSDFG(self, sdfg: "SDFG", cfg: object, dfg: object,
                             state_id: int, node: nodes.NestedSDFG,
                             function_stream: PythonCodeIOStream,
                             callsite_stream: PythonCodeIOStream) -> None:
        """Inline a NestedSDFG produced by library-node expansion.

        Library-node expansions (e.g. :class:`TileIfElseOpLibraryNode`)
        return an SDFG that the framework wraps in a
        :class:`~dace.sdfg.nodes.NestedSDFG`.  Rather than generating a
        separate function call, this handler *inlines* the nested graph
        by binding connectors to the surrounding tile variables and
        emitting the inner Tasklet code directly.

        :param sdfg: The SDFG.
        :param cfg: The control flow graph.
        :param dfg: The dataflow graph.
        :param state_id: The state ID.
        :param node: The NestedSDFG node.
        :param function_stream: Stream for function-level code.
        :param callsite_stream: Stream for call-site code.
        """
        state = cfg.state(state_id)
        self._inline_nsdfg(state, node, function_stream,
                           callsite_stream, state_id, cfg)

    def _inline_nsdfg(self, containing_state: "SDFGState",
                      nsdfg_node: nodes.NestedSDFG,
                      function_stream: PythonCodeIOStream,
                      callsite_stream: PythonCodeIOStream,
                      state_id: int, cfg: object) -> None:
        """Recursively inline a NestedSDFG into the call-site stream.

        :param containing_state: The state that contains *nsdfg_node*.
        :param nsdfg_node: The :class:`~dace.sdfg.nodes.NestedSDFG` to
            inline.
        :param function_stream: Stream for top-level function code.
        :param callsite_stream: Stream for call-site (inline) code.
        :param state_id: State ID in the parent CFG.
        :param cfg: Parent control-flow region.
        """
        inner_sdfg = nsdfg_node.sdfg
        # Ensure every library node inside has been expanded.
        inner_sdfg.expand_library_nodes(recursive=True)

        states = inner_sdfg.states()
        if len(states) > 1:
            raise NotImplementedError(
                "CuTile _inline_nsdfg does not support multi-state "
                "NestedSDFGs; library node expansions must produce a "
                "single-state SDFG."
            )

        # -- bind input connectors --
        for edge in containing_state.in_edges(nsdfg_node):
            if edge.dst_conn is None:
                continue
            if isinstance(edge.src, nodes.AccessNode):
                src = edge.src.data
            elif edge.src_conn is not None:
                src = edge.src_conn
            else:
                continue
            if edge.dst_conn != src:
                callsite_stream.write(f"{edge.dst_conn} = {src}",
                                      cfg, state_id)

        # -- emit inner state(s) --
        for inner_state in states:
            for inner_node in sdutil.dfs_topological_sort(inner_state):
                if isinstance(inner_node, nodes.Tasklet):
                    self._emit_inline_tasklet(
                        inner_state, inner_node,
                        callsite_stream, state_id, cfg)
                elif isinstance(inner_node, nodes.NestedSDFG):
                    # Recurse for expansions-within-expansions.
                    self._inline_nsdfg(
                        inner_state, inner_node, function_stream,
                        callsite_stream, state_id, cfg)
                # AccessNode -> Python local; no code needed.

        # -- bind output connectors --
        for edge in containing_state.out_edges(nsdfg_node):
            if edge.src_conn is None:
                continue
            if isinstance(edge.dst, nodes.AccessNode):
                dst = edge.dst.data
            elif edge.dst_conn is not None:
                dst = edge.dst_conn
            else:
                continue
            if edge.src_conn != dst:
                callsite_stream.write(f"{dst} = {edge.src_conn}",
                                      cfg, state_id)

    def _emit_inline_tasklet(self, inner_state: "SDFGState",
                             tasklet: nodes.Tasklet,
                             callsite_stream: PythonCodeIOStream,
                             state_id: int, cfg: object) -> None:
        """Emit code for a Tasklet inside an inlined NestedSDFG.

        Binds input edges, emits the tasklet body, then binds outputs
        to downstream :class:`~dace.sdfg.nodes.AccessNode` locals.

        :param inner_state: The inner SDFG state containing the tasklet.
        :param tasklet: The Tasklet node.
        :param callsite_stream: Stream for call-site code.
        :param state_id: The state ID in the parent CFG.
        :param cfg: The parent control-flow region.
        """
        # Bind inputs
        for edge in inner_state.in_edges(tasklet):
            if edge.dst_conn is None:
                continue
            if isinstance(edge.src, nodes.AccessNode):
                src = edge.src.data
            elif edge.data is not None and edge.data.data is not None:
                src = edge.data.data
            else:
                continue
            if edge.dst_conn != src:
                callsite_stream.write(f"{edge.dst_conn} = {src}",
                                      cfg, state_id)

        # Emit tasklet body
        code = codeblock_to_python(tasklet.code).strip()
        if code:
            callsite_stream.write(code, cfg, state_id)

        # Bind outputs to access-node locals
        for edge in inner_state.out_edges(tasklet):
            if edge.src_conn is None:
                continue
            if (isinstance(edge.dst, nodes.AccessNode)
                    and edge.dst.data != edge.src_conn):
                callsite_stream.write(
                    f"{edge.dst.data} = {edge.src_conn}",
                    cfg, state_id)

    # ------------------------------------------------------------------
    # Scope generation (kernel wrapper + launch)
    # ------------------------------------------------------------------

    def generate_scope(self, sdfg: "SDFG", cfg: object, dfg_scope: object,
                       state_id: int, function_stream: PythonCodeIOStream,
                       callsite_stream: PythonCodeIOStream) -> None:
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

        input_arrays = _ordered_unique(
            e.data.data for e in state.in_edges(entry)
            if e.data and e.data.data
            and isinstance(e.src, nodes.AccessNode))
        output_arrays = _ordered_unique(
            e.data.data for e in state.out_edges(exit_node)
            if e.data and e.data.data
            and isinstance(e.dst, nodes.AccessNode))
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
            self.generate_node(sdfg, cfg, dfg_scope, state_id, entry,
                               function_stream, kernel_stream)
            # Walk the rest of the scope. Tasklets, MapExit, AccessNodes, and
            # NestedSDFGs are routed to our predicated handlers.
            self._dispatcher.dispatch_subgraph(
                sdfg, cfg, dfg_scope, state_id,
                function_stream, kernel_stream,
                skip_entry_node=True,
            )

        function_stream.write("")
        function_stream.write(kernel_stream.getvalue())
        function_stream.write("")

        padded_grid = (grid_exprs + ["1", "1", "1"])[:3]
        if len(grid_exprs) > 3:
            # TODO: support >3D grids by flattening extra dimensions into the 3D grid or via multiple kernel launches.
            raise NotImplementedError("CuTile backend does not support >3D grids yet.")
        grid_tuple = f"({', '.join(padded_grid)})"
        deduped_arrays = list(dict.fromkeys(input_arrays + output_arrays))
        launch_args = ([_array_runtime_name(sdfg, n) for n in deduped_arrays]
                       + free_syms)
        args_tuple = (f"({', '.join(launch_args)},)" if len(launch_args) == 1
                      else f"({', '.join(launch_args)})")
        instrumented = (entry.map.instrument
                        != dtypes.InstrumentationType.No_Instrumentation)

        # Instrumentation: kernel-scope begin (before launch)
        if instrumented:
            for instr in self._dispatcher.instrumentation.values():
                if instr is not None:
                    instr.on_scope_entry(sdfg, cfg, state, entry,
                                         callsite_stream, callsite_stream,
                                         function_stream)

        callsite_stream.write(
            f"ct.launch(cupy.cuda.get_current_stream(), {grid_tuple}, "
            f"{kernel_name}, {args_tuple})",
            cfg, state_id,
        )

        # Always synchronize so the kernel completes before the host
        # continues (and before any timing measurement ends).
        callsite_stream.write(
            "cupy.cuda.get_current_stream().synchronize()", cfg, state_id)

        # Instrumentation: kernel-scope end (after synchronize). The exit node
        # is passed so the provider resolves the matching entry node (and thus
        # the matching timer-variable id) via ``state.entry_node(exit)``.
        if instrumented:
            for instr in self._dispatcher.instrumentation.values():
                if instr is not None:
                    instr.on_scope_exit(sdfg, cfg, state, exit_node,
                                        callsite_stream, callsite_stream,
                                        function_stream)
