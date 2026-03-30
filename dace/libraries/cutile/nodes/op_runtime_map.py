"""
Masked cuTile operation library node – unified binary and unary.

Implements element-wise masked operations:
    Binary:
        if mask[idx]: C[idx] = OP(A[idx], B[idx])
        if mask[idx]: C[idx] = OP(A[idx], CONST2)
        if mask[idx]: C[idx] = OP(CONST1, B[idx])
        if mask[idx]: C[idx] = OP(CONST1, CONST2)
    Unary:
        if mask[idx]: C[idx] = OP(A[idx])
        if mask[idx]: C[idx] = OP(CONST1)

When the mask is false, the output element is left untouched.
``constant1`` replaces the left/first operand, ``constant2`` the right/second.
"""
from __future__ import annotations

from typing import List, Optional

import dace
from dace import dtypes
from dace import library
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg.validation import InvalidSDFGNodeError
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from ..op_registry import TaskletType, MaskType, register_op
from ._base import (
    _TileOpBase,
    _op_cpp_expr, _get_tile_descriptors, _resolve_shape_and_scalar_form,
    _build_stride_decls, _resolve_operands, _collect_array_descs,
    _BINARY_OPS, _UNARY_OPS, SUPPORTED_MASK_DTYPES,
)


@library.node
class TileRuntimeMaskedOpLibraryNode(_TileOpBase):
    """
    Unified library node for masked element-wise operations on tiles.

    Binary:  if M: C = (constant1 or _a) op (constant2 or _b); else: C unchanged
    Unary:   if M: C = op(constant1 or _a); else: C unchanged

    Connectors
    ----------
    _a     (in, optional)  : left / first operand tile
    _b     (in, optional)  : right / second operand tile (binary only)
    _m     (in)            : boolean mask tile
    _c_in  (in, optional)  : initial C values used when mask is false
    _c     (out)           : result tile
    """

    implementations: dict = {}
    default_implementation = "pure"

    def __init__(self,
                 name: str = "TileMaskedOp",
                 op: str = "+",
                 tile_shape: Optional[List[int]] = None,
                 constant1: Optional[str] = None,
                 constant2: Optional[str] = None,
                 **kwargs):
        super().__init__(name, op=op, tile_shape=tile_shape,
                         constant1=constant1, constant2=constant2,
                         extra_inputs={"_m"}, **kwargs)

    def validate(self, sdfg: SDFG, state: SDFGState):
        self._validate_common(sdfg, state, "TileMaskedOp")

        # Additional mask-specific validation
        m_node = c_node = None
        a_node = b_node = None
        for edge in state.in_edges(self):
            if edge.dst_conn == "_m":
                m_node = edge.src
            elif edge.dst_conn == "_a":
                a_node = edge.src
            elif edge.dst_conn == "_b":
                b_node = edge.src
        for edge in state.out_edges(self):
            if edge.src_conn == "_c":
                c_node = edge.dst

        if m_node is None:
            raise InvalidSDFGNodeError(
                f"TileMaskedOp '{self.name}': connector _m must be connected.",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )

        c_desc = sdfg.arrays[c_node.data]
        m_desc = sdfg.arrays[m_node.data]

        if a_node is not None:
            a_desc = sdfg.arrays[a_node.data]
            if a_desc.shape != c_desc.shape:
                raise InvalidSDFGNodeError(
                    f"TileMaskedOp '{self.name}': shape mismatch — A={a_desc.shape}, C={c_desc.shape}",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )
        if b_node is not None:
            b_desc = sdfg.arrays[b_node.data]
            if b_desc.shape != c_desc.shape:
                raise InvalidSDFGNodeError(
                    f"TileMaskedOp '{self.name}': shape mismatch — B={b_desc.shape}, C={c_desc.shape}",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )
        if m_desc.shape != c_desc.shape:
            raise InvalidSDFGNodeError(
                f"TileMaskedOp '{self.name}': mask shape mismatch — M={m_desc.shape}, C={c_desc.shape}",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )

        if m_desc.dtype not in SUPPORTED_MASK_DTYPES:
            raise InvalidSDFGNodeError(
                f"TileMaskedOp '{self.name}': mask dtype must be bool or integer, got M={m_desc.dtype}",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )


# ── C++ expansion ────────────────────────────────────────────────────

@library.register_expansion(TileRuntimeMaskedOpLibraryNode, "pure")
class ExpandTileRuntimeMaskedOpPure(ExpandTransformation):
    """Expand TileRuntimeMaskedOpLibraryNode into a C++ tasklet."""

    environments: list = []

    @staticmethod
    def expansion(node: TileRuntimeMaskedOpLibraryNode, state: SDFGState,
                  sdfg: SDFG) -> nodes.Tasklet:
        op = node.op
        constant1 = node.constant1
        constant2 = node.constant2

        a_desc, b_desc, c_desc, m_desc, c_in_desc = _get_tile_descriptors(
            node, state, sdfg)
        if m_desc is None:
            raise ValueError(
                f"TileMaskedOp expansion: _m must be connected for node '{node.name}'."
            )
        has_c_in = c_in_desc is not None
        is_binary = (constant2 is not None) or (b_desc is not None)

        ref_desc = a_desc or b_desc or c_desc
        shape, ndim, use_scalar_form = _resolve_shape_and_scalar_form(node, ref_desc)

        inputs: set[str] = {"_m"}
        if a_desc is not None:
            inputs.add("_a")
        if b_desc is not None:
            inputs.add("_b")
        if has_c_in:
            inputs.add("_c_in")

        # Determine operand values
        left_scalar, right_scalar, left_indexed, right_indexed = _resolve_operands(
            constant1, constant2, is_binary)

        scalar_expr = _op_cpp_expr(op, left_scalar, right_scalar)
        indexed_expr = _op_cpp_expr(op, left_indexed, right_indexed)

        if use_scalar_form:
            if has_c_in:
                code = f"if (_m) {{ _c = {scalar_expr}; }} else {{ _c = _c_in; }}"
            else:
                code = f"if (_m) {{ _c = {scalar_expr}; }}"
        else:
            shape_expr = ", ".join(symstr(s) for s in shape)
            m_strides_expr = ", ".join(symstr(s) for s in m_desc.strides)
            c_strides_expr = ", ".join(symstr(s) for s in c_desc.strides)

            # Collect array descriptors for stride computation
            array_descs = _collect_array_descs(a_desc, b_desc)

            stride_decls, index_decls, index_updates = _build_stride_decls(array_descs)

            c_in_stride_decl = ""
            c_in_index_decl = ""
            c_in_index_update = ""
            c_in_else = ""
            if has_c_in:
                c_in_stride_decl = f"const std::ptrdiff_t c_in_strides[ndim] = {{{', '.join(symstr(s) for s in c_in_desc.strides)}}};"
                c_in_index_decl =   "    std::size_t iin = 0;"
                c_in_index_update = "        iin += coord * c_in_strides[d];"
                c_in_else =         "else { _c[ic] = _c_in[iin]; }"

            if not array_descs:
                # Both operands are constants – only mask + output iteration
                code = f"""
constexpr int ndim = {ndim};
const std::size_t shape[ndim] = {{{shape_expr}}};
const std::ptrdiff_t m_strides[ndim] = {{{m_strides_expr}}};
const std::ptrdiff_t c_strides[ndim] = {{{c_strides_expr}}};
{c_in_stride_decl}
const auto _val = {indexed_expr};

std::size_t n = 1;
for (int d = 0; d < ndim; ++d) {{
    n *= shape[d];
}}
for (std::size_t i = 0; i < n; ++i) {{
    std::size_t rem = i;
    std::size_t im = 0;
    std::size_t ic = 0;
{c_in_index_decl}
    for (int d = ndim - 1; d >= 0; --d) {{
        const auto extent = shape[d];
        const std::size_t coord = rem % extent;
        rem /= extent;
        im += coord * m_strides[d];
        ic += coord * c_strides[d];
{c_in_index_update}
    }}
    if (_m[im]) {{
        _c[ic] = _val;
    }}
    {c_in_else}
}}
"""
            else:
                code = f"""
constexpr int ndim = {ndim};
const std::size_t shape[ndim] = {{{shape_expr}}};
{stride_decls}const std::ptrdiff_t m_strides[ndim] = {{{m_strides_expr}}};
const std::ptrdiff_t c_strides[ndim] = {{{c_strides_expr}}};
{c_in_stride_decl}

std::size_t n = 1;
for (int d = 0; d < ndim; ++d) {{
    n *= shape[d];
}}

for (std::size_t i = 0; i < n; ++i) {{
    std::size_t rem = i;
{index_decls}    std::size_t im = 0;
    std::size_t ic = 0;
{c_in_index_decl}
    for (int d = ndim - 1; d >= 0; --d) {{
        const auto extent = shape[d];
        const std::size_t coord = rem % extent;
        rem /= extent;
{index_updates}        im += coord * m_strides[d];
        ic += coord * c_strides[d];
{c_in_index_update}
    }}
    if (_m[im]) {{
        _c[ic] = {indexed_expr};
    }}
    {c_in_else}
}}
"""

        return nodes.Tasklet(
            label=node.name + "_cutile",
            inputs=inputs,
            outputs={"_c"},
            code=code,
            language=dtypes.Language.CPP,
        )


# ── Register all masked ops ─────────────────────────────────────────

_MASKED_BINARY_DISPLAY_NAMES = {"+": "TileMaskedAdd", "-": "TileMaskedSubtract",
                                "*": "TileMaskedMultiply", "/": "TileMaskedDivide"}
_MASKED_CONST_DISPLAY_NAMES = {
    "+": "TileMaskedConstAdd", "-": "TileMaskedConstSubtract",
    "*": "TileMaskedConstMultiply", "/": "TileMaskedConstDivide",
}
_MASKED_UNARY_DISPLAY_NAMES = {
    "-": "TileMaskedNegate", "abs": "TileMaskedAbs", "sin": "TileMaskedSin",
    "cos": "TileMaskedCos", "exp": "TileMaskedExp", "sqrt": "TileMaskedSqrt",
    "log": "TileMaskedLog",
}

for _op in _BINARY_OPS:
    # Two-array masked binary
    register_op(
        op=_op,
        tasklet_type=TaskletType.ARRAY_ARRAY,
        mask=MaskType.RUNTIME,
        node_type=TileRuntimeMaskedOpLibraryNode,
        node_name=_MASKED_BINARY_DISPLAY_NAMES.get(_op, f"TileMaskedOp_{_op}"),
        out="_c",
        rhs1="_a",
        rhs2="_b",
        mask_in="_m",
        out_in="_c_in",
    )
    # Array + constant masked binary
    register_op(
        op=_op,
        tasklet_type=TaskletType.ARRAY_SYMBOL,
        mask=MaskType.RUNTIME,
        node_type=TileRuntimeMaskedOpLibraryNode,
        node_name=_MASKED_CONST_DISPLAY_NAMES.get(_op, f"TileMaskedConstOp_{_op}"),
        out="_c",
        rhs1="_a",
        rhs2="_b",
        mask_in="_m",
        out_in="_c_in",
    )

for _op in _UNARY_OPS:
    register_op(
        op=_op,
        tasklet_type=TaskletType.UNARY_ARRAY,
        mask=MaskType.RUNTIME,
        node_type=TileRuntimeMaskedOpLibraryNode,
        node_name=_MASKED_UNARY_DISPLAY_NAMES.get(_op, f"TileMaskedUnaryOp_{_op}"),
        out="_c",
        rhs1="_a",
        mask_in="_m",
        out_in="_c_in",
    )
