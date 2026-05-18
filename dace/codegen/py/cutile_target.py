"""Schedule-based cuTile Python code generation target."""

import re
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

from dace import dtypes, registry
import dace.codegen.dispatcher as dispatcher_mod
from dace.codegen.py.framecode import codeblock_to_python
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.codegen.py.target import PythonTargetCodeGenerator
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion
from dace.symbolic import symstr

if TYPE_CHECKING:
    from dace.codegen.py.framecode import DaCePythonCodeGenerator
    from dace.sdfg import SDFG, SDFGState


def _array_runtime_name(sdfg: "SDFG", name: str) -> str:
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
    return [symstr(s) for s in entry.map.range.size()]


def _map_index_exprs(entry: nodes.MapEntry) -> List[str]:
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
    seen: Dict[str, None] = {}
    for x in items:
        if x not in seen:
            seen[x] = None
    return list(seen.keys())


def _collect_free_symbols(entry: nodes.MapEntry, dfg_scope, sdfg: "SDFG") -> List[str]:
    syms = {str(s) for s in entry.map.range.free_symbols}
    for edge in dfg_scope.edges():
        memlet = edge.data
        if memlet is None:
            continue
        syms |= {str(s) for s in memlet.free_symbols}
    syms = {s for s in syms if s in sdfg.symbols and s not in sdfg.constants}
    return sorted(syms)


def _outer_endpoint(state: "SDFGState", edge, downstream: bool = False):
    return edge.dst if downstream else edge.src


def _inner_subset_for_entry_edge(state: "SDFGState", entry: nodes.MapEntry, in_edge):
    if in_edge.dst_conn and in_edge.dst_conn.startswith("IN_"):
        outer_conn = "OUT_" + in_edge.dst_conn[len("IN_"):]
        for out_edge in state.out_edges_by_connector(entry, outer_conn):
            if out_edge.data is not None and out_edge.data.subset is not None:
                return out_edge.data.subset
    return in_edge.data.subset


def _enclosing_cutile_entry(state: "SDFGState", node: nodes.Node) -> Optional[nodes.MapEntry]:
    scope = state.scope_dict()
    cur = scope.get(node)
    while cur is not None:
        if isinstance(cur, nodes.MapEntry) and cur.map.schedule == dtypes.ScheduleType.CuTile:
            return cur
        cur = scope.get(cur)
    return None


_SYMBOLIC_MASK_GUARD_RE = re.compile(r"__m\d+_guard")


def _tasklet_contains_symbolic_mask(node: nodes.Tasklet) -> bool:
    code = node.code.as_string if node.code is not None else ""
    return bool(_SYMBOLIC_MASK_GUARD_RE.search(code or ""))


def _is_cutile_node(state: "SDFGState", node: nodes.Node) -> bool:
    if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.CuTile:
        return True
    if isinstance(node, nodes.MapExit):
        entry = state.entry_node(node)
        if (entry is not None and isinstance(entry, nodes.MapEntry)
                and entry.map.schedule == dtypes.ScheduleType.CuTile):
            return True
    return _enclosing_cutile_entry(state, node) is not None


@registry.autoregister_params(name="cutile_python")
class CuTilePythonCodeGen(PythonTargetCodeGenerator):
    """Python target for CuTile-scheduled map scopes."""

    title = "CuTilePython"
    target_name = "cutile_python"
    language = "python"

    def __init__(self, frame_codegen: "DaCePythonCodeGenerator", sdfg: "SDFG") -> None:
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher
        self._tile_loads_by_entry: Dict[int, Tuple[Dict[str, str], List[str]]] = {}
        self._dispatcher.register_map_dispatcher(dtypes.ScheduleType.CuTile, self)
        self._dispatcher.register_node_dispatcher(self, predicate=self._is_in_cutile_scope)

    def get_generated_codeobjects(self):
        return []

    def get_includes(self) -> Dict[str, List[str]]:
        return {"frame": ["import cuda.tile as ct", "import cupy"]}

    def preprocess(self, sdfg: "SDFG") -> None:
        pass

    @property
    def has_initializer(self) -> bool:
        return False

    @property
    def has_finalizer(self) -> bool:
        return False

    @staticmethod
    def _is_in_cutile_scope(sdfg: "SDFG", state: "SDFGState", node: nodes.Node) -> bool:
        return _is_cutile_node(state, node)

    def generate_node(self, sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream):
        method = getattr(self, f"_generate_{type(node).__name__}", None)
        if method is None:
            raise NotImplementedError(
                f"CuTile backend has no handler for {type(node).__name__}; "
                f"extend CuTilePythonCodeGen with a _generate_{type(node).__name__} method.")
        method(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)

    def _generate_MapEntry(self, sdfg, cfg, dfg, state_id, node: nodes.MapEntry,
                           function_stream, callsite_stream) -> None:
        state = cfg.state(state_id)
        map_index_exprs = _map_index_exprs(node)
        # cuTile API uses tile-coordinate indices (raw block IDs), independent
        # of the map's start/step which describe global element positions.
        cutile_index = ", ".join(f"__pid{d}" for d in range(len(node.map.range)))

        for d in range(len(node.map.range)):
            callsite_stream.write(f"__pid{d} = ct.bid({d})", cfg, state_id)
        for var, expr in zip(node.map.params, map_index_exprs):
            callsite_stream.write(f"{var} = {expr}", cfg, state_id)

        mapping: Dict[str, str] = {}
        next_idx = 0
        for in_edge in state.in_edges(node):
            if in_edge.data is None or in_edge.data.data is None:
                continue
            outer = _outer_endpoint(state, in_edge)
            if not isinstance(outer, nodes.AccessNode):
                continue
            arr = in_edge.data.data
            inner_dst = self._inner_dst_for_entry_edge(state, node, in_edge)
            if isinstance(inner_dst, nodes.AccessNode):
                # Tile transient between MapEntry and the consumer (e.g.
                # post-MapTiling pipeline). Use its data name so downstream
                # memlets referencing it resolve naturally.
                tile_key = inner_dst.data
                tile_var = inner_dst.data
            else:
                tile_key = arr
                tile_var = f"__ct_t{next_idx}"
                next_idx += 1
            if tile_key in mapping:
                continue
            mapping[tile_key] = tile_var
            inner_subset = _inner_subset_for_entry_edge(state, node, in_edge)
            shape = ", ".join(symstr(s) for s in inner_subset.size())
            callsite_stream.write(
                f"{tile_var} = ct.load({arr}, index=({cutile_index},), shape=({shape},))",
                cfg, state_id,
            )

        self._tile_loads_by_entry[id(node)] = (mapping, map_index_exprs)

    @staticmethod
    def _inner_dst_for_entry_edge(state: "SDFGState", entry: nodes.MapEntry, in_edge):
        if not in_edge.dst_conn or not in_edge.dst_conn.startswith("IN_"):
            return None
        outer_conn = "OUT_" + in_edge.dst_conn[len("IN_"):]
        for out_edge in state.out_edges_by_connector(entry, outer_conn):
            return out_edge.dst
        return None

    def _generate_MapExit(self, sdfg, cfg, dfg, state_id, node: nodes.MapExit,
                          function_stream, callsite_stream) -> None:
        state = cfg.state(state_id)
        entry = state.entry_node(node)
        cutile_index = ", ".join(f"__pid{d}" for d in range(len(entry.map.range)))

        seen = set()
        for in_edge in state.in_edges(node):
            if in_edge.dst_conn is None or not in_edge.dst_conn.startswith("IN_"):
                continue
            if in_edge.data is None or in_edge.data.data is None:
                continue
            outer_conn = "OUT_" + in_edge.dst_conn[len("IN_"):]
            tile_expr = self._tile_expr_for_exit_in_edge(in_edge)
            if tile_expr is None:
                continue
            for out_edge in state.out_edges_by_connector(node, outer_conn):
                outer_dst = _outer_endpoint(state, out_edge, downstream=True)
                if not isinstance(outer_dst, nodes.AccessNode):
                    continue
                arr = out_edge.data.data
                if arr in seen:
                    continue
                seen.add(arr)
                callsite_stream.write(
                    f"ct.store({arr}, index=({cutile_index},), tile={tile_expr})",
                    cfg, state_id,
                )

    @staticmethod
    def _tile_expr_for_exit_in_edge(in_edge) -> Optional[str]:
        if isinstance(in_edge.src, nodes.AccessNode):
            return in_edge.src.data
        if in_edge.src_conn is not None:
            return in_edge.src_conn
        return None

    def _generate_Tasklet(self, sdfg, cfg, dfg, state_id, node: nodes.Tasklet,
                          function_stream, callsite_stream) -> None:
        if node.code.language != dtypes.Language.Python:
            raise NotImplementedError("CuTile backend only supports Python tasklets.")
        if _tasklet_contains_symbolic_mask(node):
            raise ValueError(
                f"CuTile backend cannot lower tasklet '{node.label}' containing a "
                f"symbolic mask guard marker; symbolic-masked scopes must be "
                f"expanded before reaching codegen.")
        state = cfg.state(state_id)
        entry = _enclosing_cutile_entry(state, node)
        if entry is None:
            raise RuntimeError("CuTile tasklet handler invoked outside a CuTile scope.")
        mapping, _ = self._tile_loads_by_entry[id(entry)]

        init_code = codeblock_to_python(node.code_init).strip()
        if init_code:
            self._frame._initcode.write(init_code, sdfg)
        exit_code = codeblock_to_python(node.code_exit).strip()
        if exit_code:
            self._frame._exitcode.write(exit_code, sdfg)

        self._dispatcher.defined_vars.enter_scope(node)
        try:
            for edge in state.in_edges(node):
                if not edge.dst_conn:
                    continue
                rhs: Optional[str] = None
                if isinstance(edge.src, nodes.AccessNode):
                    # Tile read from an internal transient: the value lives
                    # in a Python local that shares the AccessNode's name
                    # (emitted by either MapEntry's ct.load or a prior tasklet).
                    rhs = edge.src.data
                else:
                    arr = edge.data.data if edge.data is not None else None
                    if arr is not None and arr in mapping:
                        rhs = mapping[arr]
                    elif edge.src_conn is not None:
                        rhs = edge.src_conn
                if rhs is None:
                    continue
                callsite_stream.write(f"{edge.dst_conn} = {rhs}", cfg, state_id)
                self._dispatcher.defined_vars.add(
                    edge.dst_conn, dispatcher_mod.DefinedType.Scalar, "object")

            callsite_stream.write(f"\n####### Tasklet: {node.label}\n\n", cfg, state_id)
            callsite_stream.write(codeblock_to_python(node.code).strip() or "pass")
            callsite_stream.write(f"\n####### End of tasklet: {node.label}\n\n", cfg, state_id)

            for edge in state.out_edges(node):
                if not edge.src_conn:
                    continue
                if isinstance(edge.dst, nodes.AccessNode):
                    # Write to an internal tile transient: bind the
                    # AccessNode's name to the tasklet output connector so
                    # MapExit / downstream tasklets can reference it by name.
                    if edge.dst.data == edge.src_conn:
                        continue
                    callsite_stream.write(
                        f"{edge.dst.data} = {edge.src_conn}", cfg, state_id)
                    self._dispatcher.defined_vars.add(
                        edge.dst.data, dispatcher_mod.DefinedType.Scalar, "object")
        finally:
            self._dispatcher.defined_vars.exit_scope(node)

    def _generate_AccessNode(self, sdfg, cfg, dfg, state_id, node: nodes.AccessNode,
                             function_stream, callsite_stream) -> None:
        # Internal transient access nodes inside a cuTile kernel are pure
        # Python locals — no code needed. ct.load / ct.store are handled at
        # the MapEntry / MapExit boundary.
        return

    def generate_scope(self, sdfg, cfg, dfg_scope, state_id, function_stream, callsite_stream):
        entry = dfg_scope.source_nodes()[0]
        if not isinstance(entry, nodes.MapEntry):
            raise ValueError("CuTilePythonCodeGen expects a map scope")

        state = cfg.state(state_id)
        exit_node = state.exit_node(entry)
        grid_exprs = _grid_exprs_from_map_entry(entry)

        input_arrays = _ordered_unique(
            e.data.data for e in state.in_edges(entry)
            if e.data and e.data.data
            and isinstance(_outer_endpoint(state, e), nodes.AccessNode))
        output_arrays = _ordered_unique(
            e.data.data for e in state.out_edges(exit_node)
            if e.data and e.data.data
            and isinstance(_outer_endpoint(state, e, downstream=True), nodes.AccessNode))
        free_syms = _collect_free_symbols(entry, dfg_scope, sdfg)
        kernel_params = input_arrays + output_arrays + free_syms

        kernel_name = (f"__dace_cutile_{sdfg.name}_{cfg.cfg_id}_"
                       f"{state.block_id}_{state.node_id(entry)}")

        kernel_stream = PythonCodeIOStream()
        kernel_stream.write("@ct.kernel")
        kernel_stream.write(f"def {kernel_name}({', '.join(kernel_params)}):")
        with kernel_stream.indented():
            # Emit MapEntry (loads + pid setup) ourselves; the dispatcher's
            # topological walk treats MapEntry specially (dispatch_scope), so
            # we cannot rely on dispatch_subgraph to invoke our handler for it.
            self.generate_node(sdfg, cfg, dfg_scope, state_id, entry,
                               function_stream, kernel_stream)
            # Walk the rest of the scope. Tasklets, MapExit, and any inner
            # AccessNodes are routed to our predicated handlers.
            self._dispatcher.dispatch_subgraph(
                sdfg, cfg, dfg_scope, state_id,
                function_stream, kernel_stream,
                skip_entry_node=True,
            )

        function_stream.write("")
        function_stream.write(kernel_stream.getvalue())
        function_stream.write("")

        padded_grid = (grid_exprs + ["1", "1", "1"])[:3]
        grid_tuple = f"({', '.join(padded_grid)})"
        launch_args = ([_array_runtime_name(sdfg, n) for n in input_arrays + output_arrays]
                       + free_syms)
        args_tuple = (f"({', '.join(launch_args)},)" if len(launch_args) == 1
                      else f"({', '.join(launch_args)})")
        callsite_stream.write(
            f"ct.launch(cupy.cuda.get_current_stream(), {grid_tuple}, "
            f"{kernel_name}, {args_tuple})",
            cfg, state_id,
        )

        self._tile_loads_by_entry.pop(id(entry), None)
