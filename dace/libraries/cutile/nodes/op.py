"""
TileOpLibraryNode – unified element-wise operation library node.

Handles both binary and unary operations on tiles:
    Binary:
        C = A op B               (two arrays)
        C = A op CONST2          (array + constant on right)
        C = CONST1 op B          (constant on left + array)
        C = CONST1 op CONST2     (two constants)
    Unary:
        C = op(A)                (array operand)
        C = op(CONST1)           (constant operand)

The ``op`` property selects the operation.
``constant1`` replaces the left/first operand, ``constant2`` the right/second.
When neither ``constant2`` nor ``_b`` is present the node is unary.
"""
from __future__ import annotations

from typing import List, Optional

import dace
from dace import dtypes, properties
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg.nodes import LibraryNode
from dace import library
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from dace.sdfg.validation import InvalidSDFGNodeError
from ..op_registry import register_op, MaskType, TaskletType


# ── Supported operations ─────────────────────────────────────────────
_BINARY_OPS = ["+", "-", "*", "/"]
_UNARY_OPS = ["-", "abs", "sin", "cos", "exp", "sqrt", "log"]
_ALL_OPS = sorted(set(_BINARY_OPS + _UNARY_OPS))


def _op_cpp_expr(op: str, left: str, right: str | None = None) -> str:
    """Return a C++ expression for a binary or unary operation."""
    if right is not None:
        return f"({left} {op} {right})"
    # Unary
    if op in ("-", "+"):
        return f"({op}{left})"
    return f"{op}({left})"


@library.node
class TileOpLibraryNode(LibraryNode):
    """
    Unified library node for element-wise operations on tiles.

    Binary:  C = (constant1 or _a) op (constant2 or _b)
    Unary:   C = op(constant1 or _a)

    Connectors (presence depends on constants and arity)
    ----------
    _a  (in, optional)  : left / first operand tile
    _b  (in, optional)  : right / second operand tile (binary only)
    _c  (out)           : result tile
    """

    implementations: dict = {}
    default_implementation = "pure"

    op = properties.Property(
        dtype=str,
        default="+",
        desc="Operation symbol, e.g. '+', '-', '*', '/', 'abs', 'sin', …",
    )

    constant1 = properties.Property(
        dtype=str,
        default=None,
        desc="Left / first constant operand. None means operand comes from _a.",
        allow_none=True,
    )

    constant2 = properties.Property(
        dtype=str,
        default=None,
        desc="Right / second constant operand. None means operand comes from _b (binary) or absent (unary).",
        allow_none=True,
    )

    tile_shape = properties.ListProperty(
        element_type=int,
        default=None,
        desc=(
            "Tile dimensions, e.g. [128] for a 1-D tile or [32, 32] for 2-D. "
            "0-D (scalar) tiles can be represented with an empty list []. "
            "When None, the shape is inferred from the incoming array descriptors "
            "at expansion time."
        ),
        allow_none=True,
    )

    def __init__(self, name: str = "TileOp", op: str = "+",
                 tile_shape: Optional[List[int]] = None,
                 constant1: Optional[str] = None,
                 constant2: Optional[str] = None,
                 **kwargs):
        inputs: set[str] = set()
        if constant1 is None:
            inputs.add("_a")
        # Binary ops get _b unless constant2 replaces the right operand.
        # For ops in both _BINARY_OPS and _UNARY_OPS (e.g. "-"), the default
        # is binary; callers wanting unary must remove_in_connector("_b").
        if op in _BINARY_OPS and constant2 is None:
            inputs.add("_b")

        super().__init__(
            name,
            inputs=inputs,
            outputs={"_c"},
            **kwargs,
        )
        self.op = op
        self.tile_shape = tile_shape
        self.constant1 = constant1
        self.constant2 = constant2

    # ------------------------------------------------------------------
    @property
    def is_binary(self) -> bool:
        return self.constant2 is not None or "_b" in self.in_connectors

    def validate(self, sdfg: SDFG, state: SDFGState):
        if self.op not in _ALL_OPS:
            raise InvalidSDFGNodeError(
                f"TileOp '{self.name}': unsupported op '{self.op}'. "
                f"Supported: {_ALL_OPS}",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )

        if self.op not in _BINARY_OPS and self.is_binary:
            raise InvalidSDFGNodeError(
                f"TileOp '{self.name}': op '{self.op}' is unary-only "
                f"but has binary connectors.",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )

        c_node = None
        for edge in state.out_edges(self):
            if edge.src_conn == "_c":
                c_node = edge.dst
        if c_node is None:
            raise InvalidSDFGNodeError(
                f"TileOp '{self.name}': output connector _c must be connected.",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )

        # Check input connectors match expectation
        if self.constant1 is None:
            a_node = None
            for edge in state.in_edges(self):
                if edge.dst_conn == "_a":
                    a_node = edge.src
            if a_node is None:
                raise InvalidSDFGNodeError(
                    f"TileOp '{self.name}': connector _a must be connected when constant1 is not set.",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )

        if "_b" in self.in_connectors and self.constant2 is None:
            b_node = None
            for edge in state.in_edges(self):
                if edge.dst_conn == "_b":
                    b_node = edge.src
            if b_node is None:
                raise InvalidSDFGNodeError(
                    f"TileOp '{self.name}': connector _b must be connected when constant2 is not set.",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )


# ── C++ expansion (``pure``) ─────────────────────────────────────────

def _get_tile_descriptors(node, state, sdfg):
    """Return (a_desc, b_desc, c_desc). a_desc/b_desc may be None for constant operands."""
    a_desc = b_desc = c_desc = None
    for edge in state.in_edges(node):
        arr_name = edge.data.data
        if edge.dst_conn == "_a":
            a_desc = sdfg.arrays[arr_name]
        elif edge.dst_conn == "_b":
            b_desc = sdfg.arrays[arr_name]
    for edge in state.out_edges(node):
        arr_name = edge.data.data
        if edge.src_conn == "_c":
            c_desc = sdfg.arrays[arr_name]
    if c_desc is None:
        raise ValueError(
            f"TileOp expansion: _c not connected for node '{node.name}'."
        )
    return a_desc, b_desc, c_desc


@library.register_expansion(TileOpLibraryNode, "pure")
class ExpandTileOpPure(ExpandTransformation):
    """Expand TileOpLibraryNode into a C++ element-wise tasklet."""

    environments: list = []

    @staticmethod
    def expansion(node: TileOpLibraryNode, state: SDFGState, sdfg: SDFG) -> nodes.Tasklet:
        op = node.op
        constant1 = node.constant1
        constant2 = node.constant2

        a_desc, b_desc, c_desc = _get_tile_descriptors(node, state, sdfg)
        is_binary = (constant2 is not None) or (b_desc is not None)

        ref_desc = a_desc or b_desc or c_desc
        shape = tuple(node.tile_shape) if node.tile_shape is not None else ref_desc.shape
        ndim = len(shape)

        try:
            n_total = 1
            for s in shape:
                n_total *= int(s)
            use_scalar_form = (ndim == 0) or (n_total == 1)
        except (TypeError, ValueError):
            use_scalar_form = (ndim == 0)

        inputs: set[str] = set()
        if a_desc is not None:
            inputs.add("_a")
        if b_desc is not None:
            inputs.add("_b")

        # Determine operand values for scalar and indexed forms
        left_scalar = constant1 if constant1 is not None else "_a"
        right_scalar = constant2 if constant2 is not None else ("_b" if is_binary else None)
        left_indexed = constant1 if constant1 is not None else "_a[ia]"
        right_indexed = constant2 if constant2 is not None else ("_b[ib]" if is_binary else None)

        if use_scalar_form:
            code = f"_c = {_op_cpp_expr(op, left_scalar, right_scalar)};"
        else:
            shape_expr = ", ".join(symstr(s) for s in shape)
            c_strides_expr = ", ".join(symstr(s) for s in c_desc.strides)

            # Collect array descriptors that need stride computation
            array_descs: dict[str, object] = {}
            if a_desc is not None:
                array_descs["a"] = a_desc
            if b_desc is not None:
                array_descs["b"] = b_desc

            stride_decls = ""
            index_decls = ""
            index_updates = ""
            for key, desc in array_descs.items():
                stride_decls += f"const std::ptrdiff_t {key}_strides[ndim] = {{{', '.join(symstr(s) for s in desc.strides)}}};\n"
                index_decls += f"    std::size_t i{key} = 0;\n"
                index_updates += f"        i{key} += coord * {key}_strides[d];\n"

            if not array_descs:
                # Both operands are constants – fill output tile
                expr = _op_cpp_expr(op, left_indexed, right_indexed)
                code = f"""
constexpr int ndim = {ndim};
const std::size_t shape[ndim] = {{{shape_expr}}};
const std::ptrdiff_t c_strides[ndim] = {{{c_strides_expr}}};
const auto _val = {expr};

std::size_t n = 1;
for (int d = 0; d < ndim; ++d) {{
    n *= shape[d];
}}
for (std::size_t i = 0; i < n; ++i) {{
    std::size_t rem = i;
    std::size_t ic = 0;
    for (int d = ndim - 1; d >= 0; --d) {{
        const auto extent = shape[d];
        const std::size_t coord = rem % extent;
        rem /= extent;
        ic += coord * c_strides[d];
    }}
    _c[ic] = _val;
}}
"""
            else:
                expr = _op_cpp_expr(op, left_indexed, right_indexed)
                code = f"""
constexpr int ndim = {ndim};
const std::size_t shape[ndim] = {{{shape_expr}}};
{stride_decls}const std::ptrdiff_t c_strides[ndim] = {{{c_strides_expr}}};

std::size_t n = 1;
for (int d = 0; d < ndim; ++d) {{
    n *= shape[d];
}}
for (std::size_t i = 0; i < n; ++i) {{
    std::size_t rem = i;
{index_decls}    std::size_t ic = 0;
    for (int d = ndim - 1; d >= 0; --d) {{
        const auto extent = shape[d];
        const std::size_t coord = rem % extent;
        rem /= extent;
{index_updates}        ic += coord * c_strides[d];
    }}
    _c[ic] = {expr};
}}
"""

        return nodes.Tasklet(
            label=node.name + "_cutile",
            inputs=inputs,
            outputs={"_c"},
            code=code,
            language=dtypes.Language.CPP,
        )


# ── Register all supported ops ──────────────────────────────────────

_BINARY_DISPLAY_NAMES = {"+": "TileAdd", "-": "TileSubtract", "*": "TileMultiply", "/": "TileDivide"}
_CONST_BINARY_DISPLAY_NAMES = {
    "+": "TileConstAdd", "-": "TileConstSubtract",
    "*": "TileConstMultiply", "/": "TileConstDivide",
}
_SYMBOL_BINARY_DISPLAY_NAMES = {
    "+": "TileSymbolAdd", "-": "TileSymbolSubtract",
    "*": "TileSymbolMultiply", "/": "TileSymbolDivide",
}
_UNARY_DISPLAY_NAMES = {
    "-": "TileNegate", "abs": "TileAbs", "sin": "TileSin",
    "cos": "TileCos", "exp": "TileExp", "sqrt": "TileSqrt", "log": "TileLog",
}

for _op in _BINARY_OPS:
    # Two-array binary
    register_op(
        op=_op,
        tasklet_type=TaskletType.ARRAY_ARRAY,
        mask=MaskType.UNMASKED,
        node_type=TileOpLibraryNode,
        node_name=_BINARY_DISPLAY_NAMES.get(_op, f"TileOp_{_op}"),
        out="_c",
        rhs1="_a",
        rhs2="_b",
    )
    # Array + constant
    register_op(
        op=_op,
        tasklet_type=TaskletType.ARRAY_SYMBOL,
        mask=MaskType.UNMASKED,
        node_type=TileOpLibraryNode,
        node_name=_CONST_BINARY_DISPLAY_NAMES.get(_op, f"TileConstOp_{_op}"),
        out="_c",
        rhs1="_a",
        rhs2="_b",
    )
    # Two constants
    register_op(
        op=_op,
        tasklet_type=TaskletType.SYMBOL_SYMBOL,
        mask=MaskType.UNMASKED,
        node_type=TileOpLibraryNode,
        node_name=_SYMBOL_BINARY_DISPLAY_NAMES.get(_op, f"TileSymbolOp_{_op}"),
        out="_c",
    )

for _op in _UNARY_OPS:
    # Array operand
    register_op(
        op=_op,
        tasklet_type=TaskletType.UNARY_ARRAY,
        mask=MaskType.UNMASKED,
        node_type=TileOpLibraryNode,
        node_name=_UNARY_DISPLAY_NAMES.get(_op, f"TileUnaryOp_{_op}"),
        out="_c",
        rhs1="_a",
    )
    # Constant operand
    register_op(
        op=_op,
        tasklet_type=TaskletType.UNARY_SYMBOL,
        mask=MaskType.UNMASKED,
        node_type=TileOpLibraryNode,
        node_name=_UNARY_DISPLAY_NAMES.get(_op, f"TileUnaryOp_{_op}") + "Const",
        out="_c",
    )
