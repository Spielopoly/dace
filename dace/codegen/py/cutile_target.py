"""CuTilePythonCodeGen – Python cuTile code generation target.

Registers a predicated node dispatcher that intercepts Python-language
Tasklets containing a ``__CUTILE_SPEC__`` marker and emits:

* A ``@ct.kernel`` function definition in the global (``function_stream``)
  scope of the generated ``.py`` file.
* A ``ct.launch(...)`` call at the call site (``callsite_stream``).

The marker is produced by the ``cutile_python`` expansions of the cuTile
library nodes (see :mod:`dace.libraries.cutile.nodes`).
"""

import re
from typing import TYPE_CHECKING, Dict, List, Optional

from dace import dtypes, registry
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.codegen.py.target import PythonTargetCodeGenerator
from dace.sdfg import nodes
from dace.sdfg.state import ControlFlowRegion

if TYPE_CHECKING:
    from dace.codegen.py.framecode import DaCePythonCodeGenerator
    from dace.sdfg import SDFG

# ---------------------------------------------------------------------------
# Marker constant – must match ``CUTILE_MARKER`` in python_spec.py
# ---------------------------------------------------------------------------

_CUTILE_MARKER: str = "__CUTILE_SPEC__"


# ---------------------------------------------------------------------------
# Predicate
# ---------------------------------------------------------------------------

def _is_cutile_tasklet(sdfg, state, node) -> bool:
    """Return True for Python Tasklets that carry a ``__CUTILE_SPEC__`` marker."""
    return (
        isinstance(node, nodes.Tasklet)
        and node.language == dtypes.Language.Python
        and _CUTILE_MARKER in (node.code.as_string or "")
    )


# ---------------------------------------------------------------------------
# Code-generation helpers
# ---------------------------------------------------------------------------

def _tile_var(conn: str) -> str:
    """Return the tile-level local variable name for connector *conn*."""
    # Example: "_a" → "__ct_t_a", "_in0" → "__ct_t_in0"
    return "__ct_t" + conn


def _op_python_expr(op: str, left: str, right: Optional[str] = None) -> str:
    """Build a Python expression applying *op* to tile objects."""
    if right is not None:
        return f"({left} {op} {right})"
    # Unary
    if op in ("-", "+"):
        return f"({op}{left})"
    _CT_FUNCS = {"sin", "cos", "exp", "sqrt", "log", "ceil", "floor"}
    if op == "abs":
        return f"abs({left})"
    if op in _CT_FUNCS:
        return f"ct.{op}({left})"
    return f"{op}({left})"


def _substitute_tile_vars(expr_str: str, conn_names: List[str]) -> str:
    """Replace whole-word connector names with their tile variable names."""
    result = expr_str
    for conn in sorted(conn_names, key=len, reverse=True):
        tvar = _tile_var(conn)
        result = re.sub(rf"\b{re.escape(conn)}\b", tvar, result)
    return result


def _array_runtime_name(sdfg: "SDFG", name: str) -> str:
    """Mirror ``PythonCodeGen._runtime_data_name`` without a reference to it."""
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


# ---------------------------------------------------------------------------
# Code generator
# ---------------------------------------------------------------------------

@registry.autoregister_params(name="cutile_python")
class CuTilePythonCodeGen(PythonTargetCodeGenerator):
    """Python cuTile code generation target.

    Handles Python Tasklets with a ``__CUTILE_SPEC__`` marker, emitting
    ``@ct.kernel`` definitions and ``ct.launch`` calls.
    """

    title = "CuTilePython"
    target_name = "cutile_python"
    language = "python"

    def __init__(
        self,
        frame_codegen: "DaCePythonCodeGenerator",
        sdfg: "SDFG",
    ) -> None:
        self._frame = frame_codegen
        self._dispatcher = frame_codegen.dispatcher
        self._dispatcher.register_node_dispatcher(self, predicate=_is_cutile_tasklet)

    # ── PythonTargetCodeGenerator interface ───────────────────────────

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

    def generate_scope(self, sdfg, cfg, dfg_scope, state_id, function_stream, callsite_stream):
        raise NotImplementedError("CuTilePythonCodeGen does not handle scopes")

    def declare_array(self, sdfg, cfg, dfg, state_id, node, nodedesc, global_stream, declaration_stream):
        raise NotImplementedError("CuTilePythonCodeGen does not handle arrays")

    def allocate_array(self, sdfg, cfg, dfg, state_id, node, nodedesc, global_stream, declaration_stream,
                       allocation_stream):
        raise NotImplementedError("CuTilePythonCodeGen does not handle arrays")

    def deallocate_array(self, sdfg, cfg, dfg, state_id, node, nodedesc, function_stream, callsite_stream):
        raise NotImplementedError("CuTilePythonCodeGen does not handle arrays")

    # ── Node generation ───────────────────────────────────────────────

    def generate_node(
        self,
        sdfg: "SDFG",
        cfg: ControlFlowRegion,
        dfg,
        state_id: int,
        node: nodes.Tasklet,
        function_stream: PythonCodeIOStream,
        callsite_stream: PythonCodeIOStream,
    ) -> None:
        from dace.libraries.cutile.nodes.python_spec import decode_spec, CuTileSpec  # lazy

        spec = decode_spec(node.code.as_string or "")
        if spec is None:
            raise ValueError(
                f"CuTilePythonCodeGen: no '{_CUTILE_MARKER}' marker "
                f"in tasklet '{node.label}'"
            )

        state = cfg.state(state_id)

        # Collect connector → SDFG array name mappings
        in_arrays: Dict[str, str] = {}
        out_array: Optional[str] = None
        out_conn: Optional[str] = None

        for edge in state.in_edges(node):
            if edge.dst_conn and edge.data.data:
                in_arrays[edge.dst_conn] = _array_runtime_name(sdfg, edge.data.data)
        for edge in state.out_edges(node):
            if edge.src_conn and edge.data.data:
                out_array = _array_runtime_name(sdfg, edge.data.data)
                out_conn = edge.src_conn

        if out_array is None or out_conn is None:
            raise ValueError(
                f"CuTilePythonCodeGen: no output connector for tasklet '{node.label}'"
            )

        uid = f"{cfg.cfg_id}_{state_id}_{state.node_id(node)}"
        kernel_name = f"__ct_kernel_{uid}"

        kernel_code = self._build_kernel(spec, kernel_name, in_arrays, out_array, out_conn, sdfg)
        launch_code = self._build_launch(spec, kernel_name, in_arrays, out_array, out_conn, sdfg)

        function_stream.write(kernel_code, cfg, state_id)
        callsite_stream.write(launch_code, cfg, state_id)

    # ── Kernel builder ────────────────────────────────────────────────

    def _resolve_tile_shape(
        self, spec, out_array: str, sdfg: "SDFG"
    ) -> List[int]:
        if spec.tile_shape:
            return list(spec.tile_shape)
        desc = sdfg.arrays.get(out_array)
        if desc is not None:
            try:
                return [int(s) for s in desc.shape]
            except (TypeError, ValueError):
                pass
        return [16]

    def _build_kernel(
        self,
        spec,
        kernel_name: str,
        in_arrays: Dict[str, str],
        out_array: str,
        out_conn: str,
        sdfg: "SDFG",
    ) -> str:
        tile_shape = self._resolve_tile_shape(spec, out_array, sdfg)
        ndim = len(tile_shape)
        shape_tuple = "(" + ", ".join(str(s) for s in tile_shape) + ",)"

        # Kernel parameter list: input connectors, then output connector
        params_str = ", ".join(list(in_arrays.keys()) + [out_conn])

        # Block-index lines
        pid_lines = [f"__pid{d} = ct.bid({d})" for d in range(ndim)]
        pid_tuple = "(" + ", ".join(f"__pid{d}" for d in range(ndim)) + ",)"

        # Load each input array as a tile
        load_lines = [
            f"{_tile_var(conn)} = ct.load({conn}, index={pid_tuple}, shape={shape_tuple})"
            for conn in in_arrays
        ]

        # Compute the result expression
        out_tvar = _tile_var(out_conn)
        result_expr = self._compute_expression(spec, list(in_arrays.keys()))
        store_line = f"ct.store({out_conn}, index={pid_tuple}, tile={out_tvar})"

        body_lines: List[str] = []
        body_lines.extend(pid_lines)
        body_lines.extend(load_lines)
        body_lines.append(f"{out_tvar} = {result_expr}")
        body_lines.append(store_line)

        indented_body = "\n    ".join(body_lines)
        kernel = (
            f"@ct.kernel\n"
            f"def {kernel_name}({params_str}):\n"
            f"    {indented_body}\n"
        )
        return kernel

    # ── Expression builders ───────────────────────────────────────────

    def _compute_expression(self, spec, conn_names: List[str]) -> str:
        kind = spec.kind

        if kind == "unmasked":
            return self._unmasked_expr(spec, conn_names)

        if kind == "runtime_mask":
            return self._runtime_mask_expr(spec, conn_names)

        if kind == "symbolic_mask":
            # Symbolic masks cannot be faithfully reproduced in cuTile.
            # Emit the unmasked operation; masked-out elements are left at
            # whatever value was computed by the operation.
            active_conns = [c for c in conn_names if c not in ("_m", "_c_in")]
            return self._unmasked_expr(spec, active_conns)

        if kind == "where_select":
            t_cond = _tile_var("_cond")
            t_x = _tile_var("_x")
            t_y = _tile_var("_y")
            return f"ct.where({t_cond}, {t_x}, {t_y})"

        if kind == "if_else":
            return self._if_else_expr(spec, conn_names)

        raise ValueError(f"CuTilePythonCodeGen: unknown spec kind {kind!r}")

    def _unmasked_expr(self, spec, conn_names: List[str]) -> str:
        if spec.expr_str is not None:
            return _substitute_tile_vars(spec.expr_str, conn_names)

        op = spec.op
        left = (
            spec.constant1
            if spec.constant1 is not None
            else (_tile_var("_a") if "_a" in conn_names else None)
        )
        right = (
            spec.constant2
            if spec.constant2 is not None
            else (_tile_var("_b") if "_b" in conn_names else None)
        )

        if left is None:
            raise ValueError(
                "CuTilePythonCodeGen: cannot determine left operand "
                f"for unmasked expression (op={op!r})"
            )
        return _op_python_expr(op, left, right)

    def _runtime_mask_expr(self, spec, conn_names: List[str]) -> str:
        active_conns = [c for c in conn_names if c not in ("_m", "_c_in")]
        op_part = self._unmasked_expr(spec, active_conns)
        t_mask = _tile_var("_m")
        if "_c_in" in conn_names:
            return f"ct.where({t_mask}, {op_part}, {_tile_var('_c_in')})"
        return f"ct.where({t_mask}, {op_part}, 0)"

    def _if_else_expr(self, spec, conn_names: List[str]) -> str:
        if spec.cond_str is None or spec.true_str is None or spec.false_str is None:
            raise ValueError(
                "CuTilePythonCodeGen: if_else spec requires "
                "cond_str, true_str, and false_str"
            )
        cond = _substitute_tile_vars(spec.cond_str, conn_names)
        true_val = _substitute_tile_vars(spec.true_str, conn_names)
        false_val = _substitute_tile_vars(spec.false_str, conn_names)
        return f"ct.where({cond}, {true_val}, {false_val})"

    # ── Launch builder ────────────────────────────────────────────────

    def _build_launch(
        self,
        spec,
        kernel_name: str,
        in_arrays: Dict[str, str],
        out_array: str,
        out_conn: str,
        sdfg: "SDFG",
    ) -> str:
        tile_shape = self._resolve_tile_shape(spec, out_array, sdfg)
        ndim = len(tile_shape)

        # Grid dimensions: ceil(output_size_d / tile_size_d)
        grid_parts = [
            f"max(1, ({out_array}.shape[{d}] + {ts} - 1) // {ts})"
            for d, ts in enumerate(tile_shape[:ndim])
        ]
        # CUDA grids are always 3-D
        while len(grid_parts) < 3:
            grid_parts.append("1")
        grid_str = "(" + ", ".join(grid_parts) + ")"

        # Kernel arguments: preserve connector order for inputs, then output
        args = list(in_arrays.values()) + [out_array]
        args_tuple = "(" + ", ".join(args) + ",)"

        stream_var = f"__ct_stream_{kernel_name}"
        grid_var = f"__ct_grid_{kernel_name}"
        return (
            f"{stream_var} = cp.cuda.get_current_stream()\n"
            f"{grid_var} = {grid_str}\n"
            f"ct.launch({stream_var}, {grid_var}, {kernel_name}, {args_tuple})"
        )
