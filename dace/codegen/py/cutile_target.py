"""Schedule-based cuTile Python code generation target."""

import ast
import re
from typing import TYPE_CHECKING, Dict, List

from dace import dtypes, registry
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.codegen.py.target import PythonTargetCodeGenerator
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion
from dace.symbolic import symstr

if TYPE_CHECKING:
    from dace.codegen.py.framecode import DaCePythonCodeGenerator
    from dace.sdfg import SDFG


def _tile_var(conn: str) -> str:
    return "__ct_t" + conn


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


def _substitute_tile_vars(expr_str: str, conn_names: List[str]) -> str:
    result = expr_str
    for conn in sorted(conn_names, key=len, reverse=True):
        result = re.sub(rf"\b{re.escape(conn)}\b", _tile_var(conn), result)
    return result


def _extract_tasklet_expr(tasklet: nodes.Tasklet, out_conn: str) -> str:
    code = (tasklet.code.as_string or "").strip()
    if not code:
        raise ValueError(f"CuTilePythonCodeGen: empty tasklet code in '{tasklet.label}'")

    module = ast.parse(code)
    for stmt in module.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == out_conn:
                    return ast.unparse(stmt.value)

    if "=" in code:
        lhs, rhs = code.split("=", 1)
        if lhs.strip() == out_conn:
            return rhs.strip()

    raise ValueError(
        f"CuTilePythonCodeGen: expected assignment to '{out_conn}' in tasklet '{tasklet.label}'"
    )


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


def _infer_tile_shape(sdfg: "SDFG", state, tasklet: nodes.Tasklet, out_conn: str) -> List[str]:
    for edge in state.out_edges(tasklet):
        if edge.src_conn != out_conn:
            continue
        if edge.data.subset is not None:
            try:
                return [symstr(s) for s in edge.data.subset.size()]
            except Exception:
                pass
        if edge.data.data is not None and edge.data.data in sdfg.arrays:
            return [symstr(s) for s in sdfg.arrays[edge.data.data].shape]
    raise ValueError(f"CuTilePythonCodeGen: could not infer tile shape for tasklet '{tasklet.label}'")


def _find_outer_input_array(state, edge) -> str:
    """Trace an input memlet edge back through the enclosing MapEntry to find the outer source array.

    When the cuTile backend generates kernels, array arguments should be the
    outer (non-transient) arrays that live on the GPU, not the intermediate
    transient tile AccessNodes that live inside the map scope.  We look at
    the single in-edge of the source AccessNode: if it comes from a MapEntry,
    the data name on that edge is the outer array name.
    """
    src = edge.src
    if isinstance(src, nodes.AccessNode):
        in_edges = list(state.in_edges(src))
        if len(in_edges) == 1 and isinstance(in_edges[0].src, nodes.MapEntry):
            outer_data = in_edges[0].data.data
            if outer_data is not None:
                return outer_data
    # Fallback: use whatever data name is on the direct edge
    return edge.data.data


def _find_outer_output_array(state, edge) -> str:
    """Trace an output memlet edge forward through the enclosing MapExit to find the outer destination array.

    Same rationale as _find_outer_input_array — the kernel's output argument
    should be the outer GPU array, not the transient tile AccessNode inside the
    map scope.  We look at the single out-edge of the destination AccessNode: if
    it goes to a MapExit, the data name on that edge is the outer array name.
    """
    dst = edge.dst
    if isinstance(dst, nodes.AccessNode):
        out_edges = list(state.out_edges(dst))
        if len(out_edges) == 1 and isinstance(out_edges[0].dst, nodes.MapExit):
            outer_data = out_edges[0].data.data
            if outer_data is not None:
                return outer_data
    # Fallback: use whatever data name is on the direct edge
    return edge.data.data


@registry.autoregister_params(name="cutile_python")
class CuTilePythonCodeGen(PythonTargetCodeGenerator):
    """Python target for CuTile-scheduled map scopes."""

    title = "CuTilePython"
    target_name = "cutile_python"
    language = "python"

    def __init__(self, frame_codegen: "DaCePythonCodeGenerator", sdfg: "SDFG") -> None:
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher
        self._dispatcher.register_map_dispatcher(dtypes.ScheduleType.CuTile, self)

    def get_generated_codeobjects(self):
        return []

    def get_includes(self) -> Dict[str, List[str]]:
        return {"frame": ["import cuda.tile as ct", "import cupy as cp"]}

    def preprocess(self, sdfg: "SDFG") -> None:
        pass

    @property
    def has_initializer(self) -> bool:
        return False

    @property
    def has_finalizer(self) -> bool:
        return False

    def write_and_resolve_expr(self, memlet, current_expr, new_expr):
        raise NotImplementedError("CuTilePythonCodeGen does not handle conflict resolution")

    def generate_node(
        self,
        sdfg,
        cfg: ControlFlowRegion,
        dfg,
        state_id: int,
        node,
        function_stream: PythonCodeIOStream,
        callsite_stream: PythonCodeIOStream,
    ) -> None:
        raise NotImplementedError("CuTilePythonCodeGen is map-scope based")

    def generate_scope(self, sdfg, cfg, dfg_scope, state_id, function_stream, callsite_stream):
        entry = dfg_scope.source_nodes()[0]
        if not isinstance(entry, nodes.MapEntry):
            raise ValueError("CuTilePythonCodeGen expects a map scope")

        state = cfg.state(state_id)
        map_index_exprs = _map_index_exprs(entry)
        grid_exprs = _grid_exprs_from_map_entry(entry)

        tasklets = [n for n in dfg_scope.nodes() if isinstance(n, nodes.Tasklet)]
        tasklets.sort(key=lambda t: state.node_id(t))

        for tasklet in tasklets:
            out_edges = list(state.out_edges(tasklet))
            if len(out_edges) != 1:
                raise ValueError(
                    f"CuTilePythonCodeGen only supports single-output tasklets, got {len(out_edges)} in '{tasklet.label}'"
                )

            out_edge = out_edges[0]
            out_conn = out_edge.src_conn
            if out_conn is None or out_edge.data.data is None:
                raise ValueError(f"CuTilePythonCodeGen: invalid output edge for tasklet '{tasklet.label}'")

            in_arrays: Dict[str, str] = {}
            for edge in state.in_edges(tasklet):
                if edge.dst_conn is None or edge.data.data is None:
                    continue
                outer_name = _find_outer_input_array(state, edge)
                in_arrays[edge.dst_conn] = _array_runtime_name(sdfg, outer_name)

            out_array_name = _find_outer_output_array(state, out_edge)
            out_array = _array_runtime_name(sdfg, out_array_name)
            out_array_param = out_conn
            uid = f"{cfg.cfg_id}_{state_id}_{state.node_id(tasklet)}"
            kernel_name = f"__ct_kernel_{uid}"

            tile_shape = _infer_tile_shape(sdfg, state, tasklet, out_conn)
            expr = _extract_tasklet_expr(tasklet, out_conn)
            if "__m" in expr:
                raise ValueError(
                    "CuTilePythonCodeGen: symbolic mask tasklets are not supported in CuTile schedule scopes"
                )

            conn_names = sorted(list(in_arrays.keys()) + [out_conn], key=len, reverse=True)
            substituted_expr = _substitute_tile_vars(expr, conn_names)

            shape_tuple = "(" + ", ".join(tile_shape) + ",)"
            pid_tuple = "(" + ", ".join(map_index_exprs) + ",)"

            params = list(in_arrays.keys()) + [out_array_param]
            params_str = ", ".join(params)

            pid_lines = [f"__pid{d} = ct.bid({d})" for d in range(len(map_index_exprs))]
            load_lines = [
                f"{_tile_var(conn)} = ct.load({conn}, index={pid_tuple}, shape={shape_tuple})"
                for conn in in_arrays.keys()
            ]
            out_tvar = _tile_var(out_conn)
            store_line = f"ct.store({out_array_param}, index={pid_tuple}, tile={out_tvar})"
            body_lines = pid_lines + load_lines + [f"{out_tvar} = {substituted_expr}", store_line]

            kernel_code = (
                f"@ct.kernel\n"
                f"def {kernel_name}({params_str}):\n"
                f"    " + "\n    ".join(body_lines) + "\n"
            )

            padded_grid = list(grid_exprs)
            while len(padded_grid) < 3:
                padded_grid.append("1")
            grid_expr = "(" + ", ".join(padded_grid[:3]) + ")"

            args = [in_arrays[conn] for conn in in_arrays.keys()] + [out_array]
            args_tuple = "(" + ", ".join(args) + ",)"
            launch_code = (
                f"__ct_stream_{kernel_name} = cp.cuda.get_current_stream()\n"
                f"__ct_grid_{kernel_name} = {grid_expr}\n"
                f"ct.launch(__ct_stream_{kernel_name}, __ct_grid_{kernel_name}, {kernel_name}, {args_tuple})"
            )

            function_stream.write(kernel_code, cfg, state_id)
            callsite_stream.write(launch_code, cfg, state_id)

    def declare_array(self, sdfg, cfg, dfg, state_id, node, nodedesc, global_stream, declaration_stream):
        raise NotImplementedError("CuTilePythonCodeGen does not handle arrays")

    def allocate_array(
        self,
        sdfg,
        cfg,
        dfg,
        state_id,
        node,
        nodedesc,
        global_stream,
        declaration_stream,
        allocation_stream,
    ):
        raise NotImplementedError("CuTilePythonCodeGen does not handle arrays")

    def deallocate_array(self, sdfg, cfg, dfg, state_id, node, nodedesc, function_stream, callsite_stream):
        raise NotImplementedError("CuTilePythonCodeGen does not handle arrays")
