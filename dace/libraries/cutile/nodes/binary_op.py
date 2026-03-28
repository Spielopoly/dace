"""
TileBinaryOpLibraryNode Library Node for DaCe -> cuTile (TileIR) backend.

Represents element-wise binary operation on tiles:
    C[subset] = A[subset] OP B[subset]       (two arrays)
    C[subset] = A[subset] OP CONST           (array + constant, constant_position="right")
    C[subset] = CONST OP A[subset]           (constant + array, constant_position="left")
    C[subset] = CONST1 OP CONST2             (two constants, no array connectors)

The ``op`` property selects the operation (``+``, ``-``, ``*``, ``/``).
Optional ``constant`` / ``constant_position`` properties allow one or both
operands to be literal constants instead of array connectors.

Two expansion implementations are foreseen:
  - **pure** (C++ tasklet) — current default, works on any DaCe backend.
  - **cutile** (Nvidia CuTile Python) — future, targets CuTile directly.
"""
from __future__ import annotations

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


# ── Supported binary operations (add more entries to extend) ─────────
_BINARY_OPS = ["+", "-", "*", "/"]


def _binary_cpp_expr(op: str, left: str, right: str) -> str:
    """Return a C++ expression for ``left op right``."""
    return f"({left} {op} {right})"


@library.node
class TileBinaryOpLibraryNode(LibraryNode):
    """
    Generic library node for element-wise binary operations on tiles.

    C = A op B, C = A op CONST, C = CONST op A, or C = CONST1 op CONST2.

    Connectors (presence depends on whether constants replace operands)
    ----------
    _a  (in, optional)  : tile A — absent when constant_position=="left" and constant2 is set
    _b  (in, optional)  : tile B — absent when constant_position=="right" or constant2 is set
    _c  (out)           : tile C
    """

    implementations: dict = {}
    default_implementation = "pure"

    op = properties.Property(
        dtype=str,
        default="+",
        desc="Binary operation symbol, e.g. '+', '-', '*', '/'.",
    )

    constant = properties.Property(
        dtype=str,
        default=None,
        desc="First constant operand (e.g. '2', '3.14'). None means operand comes from a connector.",
        allow_none=True,
    )

    constant_position = properties.Property(
        dtype=str,
        default=None,
        desc="Position of 'constant': 'left' or 'right'. None when no constant.",
        allow_none=True,
    )

    constant2 = properties.Property(
        dtype=str,
        default=None,
        desc="Second constant operand. When both constant and constant2 are set, no array connectors are needed.",
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

    def __init__(self, name: str = "TileBinaryOp", op: str = "+",
                 tile_shape: list[int] | None = None,
                 constant: str | None = None,
                 constant_position: str | None = None,
                 constant2: str | None = None,
                 **kwargs):
        # Determine which connectors are needed
        inputs: set[str] = set()
        if constant2 is not None:
            # Both operands are constants — no array connectors
            pass
        elif constant is not None:
            # One operand is a constant — only one array connector
            inputs.add("_a")
        else:
            # Both operands are arrays
            inputs.add("_a")
            inputs.add("_b")

        super().__init__(
            name,
            inputs=inputs,
            outputs={"_c"},
            **kwargs,
        )
        self.op = op
        self.tile_shape = tile_shape
        self.constant = constant
        self.constant_position = constant_position
        self.constant2 = constant2

    def validate(self, sdfg: SDFG, state: SDFGState):
        has_const = self.constant is not None
        has_const2 = self.constant2 is not None

        if self.op not in _BINARY_OPS:
            raise InvalidSDFGNodeError(
                f"TileBinaryOp '{self.name}': unsupported op '{self.op}'. "
                f"Supported: {_BINARY_OPS}",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )

        if has_const and self.constant_position not in ("left", "right"):
            raise InvalidSDFGNodeError(
                f"TileBinaryOp '{self.name}': constant_position must be "
                f"'left' or 'right', got '{self.constant_position}'.",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )

        a_node = b_node = c_node = None
        for edge in state.in_edges(self):
            if edge.dst_conn == "_a":
                a_node = edge.src
            elif edge.dst_conn == "_b":
                b_node = edge.src
        for edge in state.out_edges(self):
            if edge.src_conn == "_c":
                c_node = edge.dst

        if c_node is None:
            raise InvalidSDFGNodeError(
                f"TileBinaryOp '{self.name}': output connector _c must be connected.",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )

        if has_const2:
            # Both constants — no array inputs required
            pass
        elif has_const:
            # One constant — need exactly _a
            if a_node is None:
                raise InvalidSDFGNodeError(
                    f"TileBinaryOp '{self.name}': connector _a must be connected when using a constant.",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )
        else:
            # No constants — need both _a and _b
            if a_node is None or b_node is None:
                raise InvalidSDFGNodeError(
                    f"TileBinaryOp '{self.name}': connectors _a and _b must be connected.",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )


# ── C++ expansion (``pure``) ─────────────────────────────────────────

def _get_tile_descriptors(node, state, sdfg):
    """Return (a_desc, b_desc, c_desc) for a binary op node.
    a_desc and/or b_desc may be None when constants replace them."""
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
            f"TileBinaryOp expansion: _c not connected for node '{node.name}'."
        )
    return a_desc, b_desc, c_desc


@library.register_expansion(TileBinaryOpLibraryNode, "pure")
class ExpandTileBinaryOpPure(ExpandTransformation):
    """Expand any TileBinaryOpLibraryNode into a C++ element-wise tasklet."""

    environments: list = []

    @staticmethod
    def expansion(node: TileBinaryOpLibraryNode, state: SDFGState, sdfg: SDFG) -> nodes.Tasklet:
        op = node.op
        constant = node.constant
        const_pos = node.constant_position
        constant2 = node.constant2

        a_desc, b_desc, c_desc = _get_tile_descriptors(node, state, sdfg)

        # Determine which descriptor to use for shape/strides
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

        # Figure out which connectors are arrays vs constants
        inputs: set[str] = set()
        if a_desc is not None:
            inputs.add("_a")
        if b_desc is not None:
            inputs.add("_b")

        # Build the expression generator based on what operands are present
        if constant2 is not None:
            # Both constants — no array reads
            left_val = constant if const_pos == "left" else constant2
            right_val = constant2 if const_pos == "left" else constant

            if use_scalar_form:
                code = f"_c = {_binary_cpp_expr(op, left_val, right_val)};"
            else:
                shape_expr = ", ".join(symstr(s) for s in shape)
                c_strides_expr = ", ".join(symstr(s) for s in c_desc.strides)
                expr = _binary_cpp_expr(op, left_val, right_val)
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
        elif constant is not None:
            # One constant, one array (_a)
            if use_scalar_form:
                if const_pos == "left":
                    code = f"_c = {_binary_cpp_expr(op, constant, '_a')};"
                else:
                    code = f"_c = {_binary_cpp_expr(op, '_a', constant)};"
            else:
                shape_expr = ", ".join(symstr(s) for s in shape)
                a_strides_expr = ", ".join(symstr(s) for s in a_desc.strides)
                c_strides_expr = ", ".join(symstr(s) for s in c_desc.strides)
                if const_pos == "left":
                    loop_expr = _binary_cpp_expr(op, constant, '_a[ia]')
                else:
                    loop_expr = _binary_cpp_expr(op, '_a[ia]', constant)
                code = f"""
constexpr int ndim = {ndim};
const std::size_t shape[ndim] = {{{shape_expr}}};
const std::ptrdiff_t a_strides[ndim] = {{{a_strides_expr}}};
const std::ptrdiff_t c_strides[ndim] = {{{c_strides_expr}}};

std::size_t n = 1;
for (int d = 0; d < ndim; ++d) {{
    n *= shape[d];
}}
for (std::size_t i = 0; i < n; ++i) {{
    std::size_t rem = i;
    std::size_t ia = 0;
    std::size_t ic = 0;
    for (int d = ndim - 1; d >= 0; --d) {{
        const auto extent = shape[d];
        const std::size_t coord = rem % extent;
        rem /= extent;
        ia += coord * a_strides[d];
        ic += coord * c_strides[d];
    }}
    _c[ic] = {loop_expr};
}}
"""
        else:
            # Two arrays
            if use_scalar_form:
                code = f"_c = {_binary_cpp_expr(op, '_a', '_b')};"
            else:
                shape_expr = ", ".join(symstr(s) for s in shape)
                a_strides_expr = ", ".join(symstr(s) for s in a_desc.strides)
                b_strides_expr = ", ".join(symstr(s) for s in b_desc.strides)
                c_strides_expr = ", ".join(symstr(s) for s in c_desc.strides)
                code = f"""
constexpr int ndim = {ndim};
const std::size_t shape[ndim] = {{{shape_expr}}};
const std::ptrdiff_t a_strides[ndim] = {{{a_strides_expr}}};
const std::ptrdiff_t b_strides[ndim] = {{{b_strides_expr}}};
const std::ptrdiff_t c_strides[ndim] = {{{c_strides_expr}}};

std::size_t n = 1;
for (int d = 0; d < ndim; ++d) {{
    n *= shape[d];
}}
for (std::size_t i = 0; i < n; ++i) {{
    std::size_t rem = i;
    std::size_t ia = 0;
    std::size_t ib = 0;
    std::size_t ic = 0;
    for (int d = ndim - 1; d >= 0; --d) {{
        const auto extent = shape[d];
        const std::size_t coord = rem % extent;
        rem /= extent;
        ia += coord * a_strides[d];
        ib += coord * b_strides[d];
        ic += coord * c_strides[d];
    }}
    _c[ic] = {_binary_cpp_expr(op, '_a[ia]', '_b[ib]')};
}}
"""
        return nodes.Tasklet(
            label=node.name + "_cutile",
            inputs=inputs,
            outputs={"_c"},
            code=code,
            language=dtypes.Language.CPP,
        )


# ── Register all supported binary ops ───────────────────────────────

_OP_DISPLAY_NAMES = {"+": "TileAdd", "-": "TileSubtract", "*": "TileMultiply", "/": "TileDivide"}
_CONST_OP_DISPLAY_NAMES = {
    "+": "TileConstAdd", "-": "TileConstSubtract",
    "*": "TileConstMultiply", "/": "TileConstDivide",
}
_SYMBOL_OP_DISPLAY_NAMES = {
    "+": "TileSymbolAdd", "-": "TileSymbolSubtract",
    "*": "TileSymbolMultiply", "/": "TileSymbolDivide",
}

for _op in _BINARY_OPS:
    # Two-array binary
    register_op(
        op=_op,
        tasklet_type=TaskletType.ARRAY_ARRAY,
        mask=MaskType.UNMASKED,
        node_type=TileBinaryOpLibraryNode,
        node_name=_OP_DISPLAY_NAMES.get(_op, f"TileBinaryOp_{_op}"),
        out="_c",
        rhs1="_a",
        rhs2="_b",
    )
    # Array + constant (both rhs map to _a since there's only one array connector)
    register_op(
        op=_op,
        tasklet_type=TaskletType.ARRAY_SYMBOL,
        mask=MaskType.UNMASKED,
        node_type=TileBinaryOpLibraryNode,
        node_name=_CONST_OP_DISPLAY_NAMES.get(_op, f"TileConstBinaryOp_{_op}"),
        out="_c",
        rhs1="_a",
        rhs2=None,
    )
    # Two constants (no array connectors)
    register_op(
        op=_op,
        tasklet_type=TaskletType.SYMBOL_SYMBOL,
        mask=MaskType.UNMASKED,
        node_type=TileBinaryOpLibraryNode,
        node_name=_SYMBOL_OP_DISPLAY_NAMES.get(_op, f"TileSymbolBinaryOp_{_op}"),
        out="_c",
    )
