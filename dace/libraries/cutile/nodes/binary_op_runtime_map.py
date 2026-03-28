"""
Masked cuTile binary operation library nodes.

Implements element-wise masked binary operations:
    if mask[idx]:
        C[idx] = OP(A[idx], B[idx])        (two arrays)
        C[idx] = OP(A[idx], CONST)         (array + constant)
        C[idx] = OP(CONST, A[idx])         (constant + array)
        C[idx] = OP(CONST1, CONST2)        (two constants)

When the mask is false, the output element is left untouched.
The ``op`` property selects the operation (``+``, ``-``, ``*``, ``/``).
Optional ``constant`` / ``constant_position`` / ``constant2`` properties
allow one or both operands to be literal constants.
"""
from __future__ import annotations

from typing import Optional, cast

import dace
from dace import dtypes, properties
from dace import library
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg.nodes import LibraryNode
from dace.sdfg.validation import InvalidSDFGNodeError
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from ..op_registry import TaskletType, MaskType, register_op
from .binary_op import _binary_cpp_expr
from . import binary_op


_MASKED_BINARY_OPS = binary_op._BINARY_OPS


@library.node
class TileRuntimeMaskedBinaryOpLibraryNode(LibraryNode):
    """
    Generic library node for masked binary operation on tiles:
        if M: C = A op B  (or A op CONST, CONST op A, CONST1 op CONST2)
        else: C unchanged

    Connectors (presence depends on whether constants replace operands)
    ----------
    _a     (in, optional)  : tile A
    _b     (in, optional)  : tile B
    _m     (in)            : tile mask (same shape as output)
    _c_in  (in, optional)  : initial tile C values used when mask is false
    _c     (out)           : tile C
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
        desc="First constant operand. None means operand comes from a connector.",
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

    def __init__(self,
                 name: str = "TileMaskedBinaryOp",
                 op: str = "+",
                 tile_shape: list[int] | None = None,
                 constant: str | None = None,
                 constant_position: str | None = None,
                 constant2: str | None = None,
                 **kwargs):
        inputs: set[str] = {"_m"}
        if constant2 is not None:
            pass  # both constants — no array connectors
        elif constant is not None:
            inputs.add("_a")
        else:
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

        if m_node is None or c_node is None:
            raise InvalidSDFGNodeError(
                f"TileMaskedBinaryOp '{self.name}': connectors _m and _c must be connected.",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )

        c_desc = sdfg.arrays[c_node.data]
        m_desc = sdfg.arrays[m_node.data]

        # Shape checks for connected array operands
        if a_node is not None:
            a_desc = sdfg.arrays[a_node.data]
            if a_desc.shape != c_desc.shape:
                raise InvalidSDFGNodeError(
                    f"TileMaskedBinaryOp '{self.name}': shape mismatch — "
                    f"A={a_desc.shape}, C={c_desc.shape}",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )
        if b_node is not None:
            b_desc = sdfg.arrays[b_node.data]
            if b_desc.shape != c_desc.shape:
                raise InvalidSDFGNodeError(
                    f"TileMaskedBinaryOp '{self.name}': shape mismatch — "
                    f"B={b_desc.shape}, C={c_desc.shape}",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )
        if m_desc.shape != c_desc.shape:
            raise InvalidSDFGNodeError(
                f"TileMaskedBinaryOp '{self.name}': mask shape mismatch — "
                f"M={m_desc.shape}, C={c_desc.shape}",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )

        supported_mask_dtypes = {
            dace.bool,
            dace.int8, dace.uint8, dace.int16, dace.uint16,
            dace.int32, dace.uint32, dace.int64, dace.uint64,
        }
        if m_desc.dtype not in supported_mask_dtypes:
            raise InvalidSDFGNodeError(
                f"TileMaskedBinaryOp '{self.name}': mask dtype must be bool or integer, got M={m_desc.dtype}",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )


# ── C++ expansion ────────────────────────────────────────────────────

def _get_masked_tile_descriptors(node, state, sdfg):
    """Return (a_desc, b_desc, m_desc, c_desc, c_in_desc) for a masked binary node.
    a_desc and/or b_desc may be None when constants replace them."""
    a_desc = b_desc = m_desc = c_desc = c_in_desc = None
    for edge in state.in_edges(node):
        arr_name = edge.data.data
        if arr_name is None:
            continue
        if edge.dst_conn == "_a":
            a_desc = sdfg.arrays[arr_name]
        elif edge.dst_conn == "_b":
            b_desc = sdfg.arrays[arr_name]
        elif edge.dst_conn == "_m":
            m_desc = sdfg.arrays[arr_name]
        elif edge.dst_conn == "_c_in":
            c_in_desc = sdfg.arrays[arr_name]
    for edge in state.out_edges(node):
        arr_name = edge.data.data
        if arr_name is None:
            continue
        if edge.src_conn == "_c":
            c_desc = sdfg.arrays[arr_name]
    if None in (m_desc, c_desc):
        raise ValueError(
            f"TileMaskedBinaryOp expansion: _m and _c must be connected for node '{node.name}'."
        )
    return (
        cast(Optional[dace.data.Data], a_desc),
        cast(Optional[dace.data.Data], b_desc),
        cast(dace.data.Data, m_desc),
        cast(dace.data.Data, c_desc),
        cast(Optional[dace.data.Data], c_in_desc),
    )


@library.register_expansion(TileRuntimeMaskedBinaryOpLibraryNode, "pure")
class ExpandTileRuntimeMaskedBinaryOpPure(ExpandTransformation):
    """Expand any TileRuntimeMaskedBinaryOpLibraryNode into a C++ tasklet."""

    environments: list = []

    @staticmethod
    def expansion(node: TileRuntimeMaskedBinaryOpLibraryNode, state: SDFGState,
                  sdfg: SDFG) -> nodes.Tasklet:
        op = node.op
        constant = node.constant
        const_pos = node.constant_position
        constant2 = node.constant2

        a_desc, b_desc, m_desc, c_desc, c_in_desc = _get_masked_tile_descriptors(
            node, state, sdfg)
        has_c_in = c_in_desc is not None
        ref_desc = a_desc or b_desc or c_desc

        tile_shape = getattr(node, "tile_shape", None)
        shape = tuple(tile_shape) if tile_shape is not None else ref_desc.shape
        ndim = len(shape)

        try:
            n_total = 1
            for s in shape:
                n_total *= int(s)
            use_scalar_form = (ndim == 0) or (n_total == 1)
        except (TypeError, ValueError):
            use_scalar_form = (ndim == 0)

        inputs: set[str] = {"_m"}
        if a_desc is not None:
            inputs.add("_a")
        if b_desc is not None:
            inputs.add("_b")
        if has_c_in:
            inputs.add("_c_in")

        # Determine left/right expression atoms
        if constant2 is not None:
            left_val = constant if const_pos == "left" else constant2
            right_val = constant2 if const_pos == "left" else constant

            def scalar_expr():
                return _binary_cpp_expr(op, left_val, right_val)

            def indexed_expr():
                return _binary_cpp_expr(op, left_val, right_val)

            array_descs = {}  # no array stride computation needed
        elif constant is not None:
            def scalar_expr():
                if const_pos == "left":
                    return _binary_cpp_expr(op, constant, '_a')
                return _binary_cpp_expr(op, '_a', constant)

            def indexed_expr():
                if const_pos == "left":
                    return _binary_cpp_expr(op, constant, '_a[ia]')
                return _binary_cpp_expr(op, '_a[ia]', constant)

            array_descs = {"a": a_desc}
        else:
            def scalar_expr():
                return _binary_cpp_expr(op, '_a', '_b')

            def indexed_expr():
                return _binary_cpp_expr(op, '_a[ia]', '_b[ib]')

            array_descs = {"a": a_desc, "b": b_desc}

        if use_scalar_form:
            if has_c_in:
                code = f"if (_m) {{ _c = {scalar_expr()}; }} else {{ _c = _c_in; }}"
            else:
                code = f"if (_m) {{ _c = {scalar_expr()}; }}"
        else:
            shape_expr = ", ".join(symstr(s) for s in shape)
            m_strides_expr = ", ".join(symstr(s) for s in m_desc.strides)
            c_strides_expr = ", ".join(symstr(s) for s in c_desc.strides)

            stride_decls = ""
            index_decls = ""
            index_updates = ""
            for key, desc in array_descs.items():
                stride_decls += f"const std::ptrdiff_t {key}_strides[ndim] = {{{', '.join(symstr(s) for s in desc.strides)}}};\n"
                index_decls += f"    std::size_t i{key} = 0;\n"
                index_updates += f"        i{key} += coord * {key}_strides[d];\n"

            c_in_stride_decl = ""
            c_in_index_decl = ""
            c_in_index_update = ""
            c_in_else = ""
            if has_c_in:
                c_in_stride_decl = f"const std::ptrdiff_t c_in_strides[ndim] = {{{', '.join(symstr(s) for s in c_in_desc.strides)}}};"
                c_in_index_decl = "    std::size_t iin = 0;"
                c_in_index_update = "        iin += coord * c_in_strides[d];"
                c_in_else = "else { _c[ic] = _c_in[iin]; }"

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
        _c[ic] = {indexed_expr()};
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


# ── Register all masked binary ops ──────────────────────────────────

_OP_DISPLAY_NAMES = {"+": "TileMaskedAdd", "-": "TileMaskedSubtract",
                     "*": "TileMaskedMultiply", "/": "TileMaskedDivide"}
_CONST_OP_DISPLAY_NAMES = {
    "+": "TileMaskedConstAdd", "-": "TileMaskedConstSubtract",
    "*": "TileMaskedConstMultiply", "/": "TileMaskedConstDivide",
}

for _op in _MASKED_BINARY_OPS:
    # Two-array masked binary
    register_op(
        op=_op,
        tasklet_type=TaskletType.ARRAY_ARRAY,
        mask=MaskType.RUNTIME,
        node_type=TileRuntimeMaskedBinaryOpLibraryNode,
        node_name=_OP_DISPLAY_NAMES.get(_op, f"TileMaskedBinaryOp_{_op}"),
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
        node_type=TileRuntimeMaskedBinaryOpLibraryNode,
        node_name=_CONST_OP_DISPLAY_NAMES.get(_op, f"TileMaskedConstBinaryOp_{_op}"),
        out="_c",
        rhs1="_a",
        rhs2=None,
        mask_in="_m",
        out_in="_c_in",
    )
