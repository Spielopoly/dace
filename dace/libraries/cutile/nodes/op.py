"""
TileOpLibraryNode – unified element-wise operation library node (unmasked).

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

from typing import List, Optional, cast

from dace import dtypes
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes
from dace import library
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from ..op_registry import register_op, MaskType, TaskletType
from .base import (
    TileOpBase,
    get_output_connector_name,
    op_cpp_expr, get_tile_descriptors, resolve_shape_and_scalar_form,
    build_stride_decls, resolve_operands, collect_array_descs,
    get_all_input_descs, build_multi_op_code,
    _BINARY_OPS, _UNARY_OPS,
)


@library.node
class TileOpLibraryNode(TileOpBase):
    """
    Unified library node for element-wise operations on tiles.

    Binary:  C = (constant1 or _a) op (constant2 or _b)
    Unary:   C = op(constant1 or _a)

    Connectors (presence depends on constants and arity)
    ----------
    _a  (in, optional)  : left / first operand tile
    _b  (in, optional)  : right / second operand tile (binary only)
    _out (out)          : result tile
    """

    implementations: dict = {}
    default_implementation = "pure"

    def __init__(self, name: str = "TileOp", op: str = "+",
                 tile_shape: Optional[List[int]] = None,
                 constant1: Optional[str] = None,
                 constant2: Optional[str] = None,
                 expr=None,
                 out_connector: str = "_out",
                 **kwargs) -> None:
        """Initialise the unmasked element-wise tile operation node.

        Args:
            name: Node display name in the SDFG (default ``"TileOp"``).
            op: Operation symbol or function name (e.g. ``"+"``, ``"abs"``).
            tile_shape: Fixed tile extents, or ``None`` to infer at expansion.
            constant1: Literal C++ value replacing the left/first operand.
                When ``None`` the ``_a`` input connector is created.
            constant2: Literal C++ value replacing the right/second operand.
                When ``None`` the ``_b`` input connector is created for binary ops.
            expr: Optional SymPy expression for multi-op mode.
            out_connector: Name of the single output connector (default
                ``"_out"``).
            **kwargs: Forwarded to :class:`TileOpBase`.
        """
        super().__init__(name, op=op, tile_shape=tile_shape,
                         constant1=constant1, constant2=constant2,
                         expr=expr, out_connector=out_connector,
                         **kwargs)

    def validate(self, sdfg: SDFG, state: SDFGState) -> None:
        """Validate the node before code generation.

        Args:
            sdfg: The SDFG containing this node.
            state: The state containing this node.

        Raises:
            :class:`~dace.sdfg.validation.InvalidSDFGNodeError`: If any
                connector or operation constraint is violated.
        """
        self._validate_common(sdfg, state, "TileOp")


# ── C++ expansion (``pure``) ─────────────────────────────────────────

@library.register_expansion(TileOpLibraryNode, "pure")
class ExpandTileOpPure(ExpandTransformation):
    """Expand TileOpLibraryNode into a C++ element-wise tasklet."""

    environments: list = []

    @staticmethod
    def expansion(node: TileOpLibraryNode, state: SDFGState, sdfg: SDFG) -> nodes.Tasklet:
        """Expand the node into a C++ element-wise tasklet.

        Generates a C++ loop (or scalar assignment for single-element tiles)
        that applies the configured operation to every element of the input
        tile(s) and writes the result to the output tile.

        Args:
            node: The :class:`TileOpLibraryNode` to expand.
            state: The SDFG state containing *node*.
            sdfg: The SDFG owning the state.

        Returns:
            A :class:`~dace.sdfg.nodes.Tasklet` implementing the operation in
            C++.
        """
        out_conn = get_output_connector_name(node)
        if node.expr is not None:
            # ── Multi-op expression mode ──────────────────────────────────
            input_descs, c_desc = get_all_input_descs(node, state, sdfg)
            code = build_multi_op_code(
                node.expr, input_descs, c_desc, node.tile_shape, out_conn=out_conn)
            return nodes.Tasklet(
                label=node.name + "_cutile",
                inputs=set(input_descs.keys()),
                outputs={out_conn},
                code=code,
                language=dtypes.Language.CPP,
            )

        # ── Single-op mode ────────────────────────────────────────────────
        op = node.op
        constant1 = node.constant1
        constant2 = node.constant2

        a_desc, b_desc, c_desc, _, _ = get_tile_descriptors(node, state, sdfg)
        is_binary = (constant2 is not None) or (b_desc is not None)

        ref_desc = a_desc or b_desc or c_desc
        shape, ndim, use_scalar_form = resolve_shape_and_scalar_form(node, ref_desc)

        inputs: set[str] = set()
        if a_desc is not None:
            inputs.add("_a")
        if b_desc is not None:
            inputs.add("_b")

        # Determine operand values for scalar and indexed forms
        left_scalar, right_scalar, left_indexed, right_indexed = resolve_operands(
            constant1, constant2, is_binary)

        if use_scalar_form:
            code = f"{out_conn} = {op_cpp_expr(op, left_scalar, right_scalar)};"
        else:
            shape_expr = ", ".join(symstr(s) for s in shape)
            c_strides_expr = ", ".join(symstr(s) for s in c_desc.strides)

            # Collect array descriptors that need stride computation
            array_descs = collect_array_descs(a_desc, b_desc)

            stride_decls, index_decls, index_updates = build_stride_decls(array_descs)

            if not array_descs:
                # Both operands are constants – fill output tile
                expr = op_cpp_expr(op, left_indexed, right_indexed)
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
    {out_conn}[ic] = _val;
}}
"""
            else:
                expr = op_cpp_expr(op, left_indexed, right_indexed)
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
    {out_conn}[ic] = {expr};
}}
"""

        return nodes.Tasklet(
            label=node.name + "_cutile",
            inputs=inputs,
            outputs={out_conn},
            code=code,
            language=dtypes.Language.CPP,
        )


# ── Register all supported ops ──────────────────────────────────────
_BINARY_DISPLAY_NAMES = {
    "+": "TileAdd", "-": "TileSubtract", "*": "TileMultiply", "/": "TileDivide",
    ">": "TileGreaterThan", "<": "TileLessThan",
    ">=": "TileGreaterEqual", "<=": "TileLessEqual",
    "==": "TileEqual", "!=": "TileNotEqual",
}
_CONST_BINARY_DISPLAY_NAMES = {
    "+": "TileConstAdd", "-": "TileConstSubtract",
    "*": "TileConstMultiply", "/": "TileConstDivide",
    ">": "TileConstGreaterThan", "<": "TileConstLessThan",
    ">=": "TileConstGreaterEqual", "<=": "TileConstLessEqual",
    "==": "TileConstEqual", "!=": "TileConstNotEqual",
}
_SYMBOL_BINARY_DISPLAY_NAMES = {
    "+": "TileSymbolAdd", "-": "TileSymbolSubtract",
    "*": "TileSymbolMultiply", "/": "TileSymbolDivide",
    ">": "TileSymbolGreaterThan", "<": "TileSymbolLessThan",
    ">=": "TileSymbolGreaterEqual", "<=": "TileSymbolLessEqual",
    "==": "TileSymbolEqual", "!=": "TileSymbolNotEqual",
}
_UNARY_DISPLAY_NAMES = {
    "-": "TileNegate", "abs": "TileAbs", "sin": "TileSin",
    "cos": "TileCos", "exp": "TileExp", "sqrt": "TileSqrt", "log": "TileLog",
}

for _op in [*_BINARY_OPS]:
    # Two-array binary/comparison
    register_op(
        op=_op,
        tasklet_type=TaskletType.ARRAY_ARRAY,
        mask=MaskType.UNMASKED,
        node_type=TileOpLibraryNode,
        node_name=_BINARY_DISPLAY_NAMES.get(_op, f"TileOp_{_op}"),
        out="_out",
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
        out="_out",
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
        out="_out",
    )

for _op in _UNARY_OPS:
    # Array operand
    register_op(
        op=_op,
        tasklet_type=TaskletType.UNARY_ARRAY,
        mask=MaskType.UNMASKED,
        node_type=TileOpLibraryNode,
        node_name=_UNARY_DISPLAY_NAMES.get(_op, f"TileUnaryOp_{_op}"),
        out="_out",
        rhs1="_a",
    )
    # Constant operand
    register_op(
        op=_op,
        tasklet_type=TaskletType.UNARY_SYMBOL,
        mask=MaskType.UNMASKED,
        node_type=TileOpLibraryNode,
        node_name=_UNARY_DISPLAY_NAMES.get(_op, f"TileUnaryOp_{_op}") + "Const",
        out="_out",
    )
