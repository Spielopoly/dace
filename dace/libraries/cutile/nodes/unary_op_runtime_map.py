"""
Masked cuTile unary operation library nodes.

Implements element-wise masked unary operations:
    if mask[idx]:
        C[idx] = OP(A[idx])       (array operand)
        C[idx] = OP(CONST)        (constant operand)

When the mask is false, the output element is left untouched.
The ``op`` property selects the operation (prefix or function-style).
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
from .unary_op import _unary_cpp_expr, _UNARY_OPS


@library.node
class TileRuntimeMaskedUnaryOpLibraryNode(LibraryNode):
    """
    Generic library node for masked unary operation on a tile:
        if M: C = op(A)   or   if M: C = op(CONST)
        else: C unchanged

    Connectors
    ----------
    _a     (in, optional) : tile A — absent when constant is set
    _m     (in)           : tile mask (same shape as C)
    _c_in  (in, optional) : initial tile C values used when mask is false
    _c     (out)          : tile C
    """

    implementations: dict = {}
    default_implementation = "pure"

    op = properties.Property(
        dtype=str,
        default="-",
        desc="Unary operation symbol, e.g. '-', 'abs', 'sin', 'exp', 'sqrt', 'log'.",
    )

    constant = properties.Property(
        dtype=str,
        default=None,
        desc="Constant operand. When set, no _a connector is needed.",
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
                 name: str = "TileMaskedUnaryOp",
                 op: str = "-",
                 tile_shape: list[int] | None = None,
                 constant: str | None = None,
                 **kwargs):
        inputs: set[str] = {"_m"}
        if constant is None:
            inputs.add("_a")

        super().__init__(
            name,
            inputs=inputs,
            outputs={"_c"},
            **kwargs,
        )
        self.op = op
        self.tile_shape = tile_shape
        self.constant = constant

    def validate(self, sdfg: SDFG, state: SDFGState):
        m_node = c_node = None
        for edge in state.in_edges(self):
            if edge.dst_conn == "_m":
                m_node = edge.src
        for edge in state.out_edges(self):
            if edge.src_conn == "_c":
                c_node = edge.dst

        if m_node is None or c_node is None:
            raise InvalidSDFGNodeError(
                f"TileMaskedUnaryOp '{self.name}': connectors (_m, _c) must be connected.",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )

        if self.constant is None:
            a_node = None
            for edge in state.in_edges(self):
                if edge.dst_conn == "_a":
                    a_node = edge.src
            if a_node is None:
                raise InvalidSDFGNodeError(
                    f"TileMaskedUnaryOp '{self.name}': connector _a must be connected when no constant is set.",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )

        m_desc = sdfg.arrays[m_node.data]
        supported_mask_dtypes = {
            dace.bool,
            dace.int8, dace.uint8, dace.int16, dace.uint16,
            dace.int32, dace.uint32, dace.int64, dace.uint64,
        }
        if m_desc.dtype not in supported_mask_dtypes:
            raise InvalidSDFGNodeError(
                f"TileMaskedUnaryOp '{self.name}': mask dtype must be bool or integer, got M={m_desc.dtype}",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )


# ── C++ expansion ────────────────────────────────────────────────────

def _get_masked_unary_descriptors(node, state, sdfg):
    """Return (a_desc, m_desc, c_desc, c_in_desc) for a masked unary node.
    a_desc may be None when constant is set."""
    a_desc = m_desc = c_desc = c_in_desc = None
    for edge in state.in_edges(node):
        arr_name = edge.data.data
        if arr_name is None:
            continue
        if edge.dst_conn == "_a":
            a_desc = sdfg.arrays[arr_name]
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
            f"TileMaskedUnaryOp expansion: _m and _c must be connected for "
            f"node '{node.name}'."
        )
    return (a_desc, m_desc, c_desc, c_in_desc)


@library.register_expansion(TileRuntimeMaskedUnaryOpLibraryNode, "pure")
class ExpandTileRuntimeMaskedUnaryOpPure(ExpandTransformation):
    """Expand TileRuntimeMaskedUnaryOpLibraryNode into a C++ tasklet."""

    environments: list = []

    @staticmethod
    def expansion(node: TileRuntimeMaskedUnaryOpLibraryNode, state: SDFGState,
                  sdfg: SDFG) -> nodes.Tasklet:
        op = node.op
        constant = node.constant
        a_desc, m_desc, c_desc, c_in_desc = _get_masked_unary_descriptors(node, state, sdfg)
        has_c_in = c_in_desc is not None

        ref_desc = a_desc or c_desc
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
        if has_c_in:
            inputs.add("_c_in")

        if constant is not None:
            # Constant operand — the value is a literal, not from an array
            expr = _unary_cpp_expr(op, constant)
            if use_scalar_form:
                if has_c_in:
                    code = f"if (_m) {{ _c = {expr}; }} else {{ _c = _c_in; }}"
                else:
                    code = f"if (_m) {{ _c = {expr}; }}"
            else:
                shape_expr = ", ".join(symstr(s) for s in shape)
                m_strides_expr = ", ".join(symstr(s) for s in m_desc.strides)
                c_strides_expr = ", ".join(symstr(s) for s in c_desc.strides)
                c_in_strides_expr = ", ".join(symstr(s) for s in c_in_desc.strides) if has_c_in else ""

                code = f"""
constexpr int ndim = {ndim};
const std::size_t shape[ndim] = {{{shape_expr}}};
const std::ptrdiff_t m_strides[ndim] = {{{m_strides_expr}}};
const std::ptrdiff_t c_strides[ndim] = {{{c_strides_expr}}};
{"const std::ptrdiff_t c_in_strides[ndim] = {" + c_in_strides_expr + "};" if has_c_in else ""}
const auto _val = {expr};

std::size_t n = 1;
for (int d = 0; d < ndim; ++d) {{
    n *= shape[d];
}}
for (std::size_t i = 0; i < n; ++i) {{
    std::size_t rem = i;
    std::size_t im = 0;
    std::size_t ic = 0;
    {"std::size_t iin = 0;" if has_c_in else ""}
    for (int d = ndim - 1; d >= 0; --d) {{
        const auto extent = shape[d];
        const std::size_t coord = rem % extent;
        rem /= extent;
        im += coord * m_strides[d];
        ic += coord * c_strides[d];
        {"iin += coord * c_in_strides[d];" if has_c_in else ""}
    }}
    if (_m[im]) {{
        _c[ic] = _val;
    }}
    {"else { _c[ic] = _c_in[iin]; }" if has_c_in else ""}
}}
"""
        elif use_scalar_form:
            expr = _unary_cpp_expr(op, '_a')
            if has_c_in:
                code = f"if (_m) {{ _c = {expr}; }} else {{ _c = _c_in; }}"
            else:
                code = f"if (_m) {{ _c = {expr}; }}"
        else:
            shape_expr = ", ".join(symstr(s) for s in shape)
            a_strides_expr = ", ".join(symstr(s) for s in a_desc.strides)
            m_strides_expr = ", ".join(symstr(s) for s in m_desc.strides)
            c_strides_expr = ", ".join(symstr(s) for s in c_desc.strides)
            c_in_strides_expr = ", ".join(symstr(s) for s in c_in_desc.strides) if has_c_in else ""

            code = f"""
constexpr int ndim = {ndim};
const std::size_t shape[ndim] = {{{shape_expr}}};
const std::ptrdiff_t a_strides[ndim] = {{{a_strides_expr}}};
const std::ptrdiff_t m_strides[ndim] = {{{m_strides_expr}}};
const std::ptrdiff_t c_strides[ndim] = {{{c_strides_expr}}};
{"const std::ptrdiff_t c_in_strides[ndim] = {" + c_in_strides_expr + "};" if has_c_in else ""}

std::size_t n = 1;
for (int d = 0; d < ndim; ++d) {{
    n *= shape[d];
}}
for (std::size_t i = 0; i < n; ++i) {{
    std::size_t rem = i;
    std::size_t ia = 0;
    std::size_t im = 0;
    std::size_t ic = 0;
    {"std::size_t iin = 0;" if has_c_in else ""}
    for (int d = ndim - 1; d >= 0; --d) {{
        const auto extent = shape[d];
        const std::size_t coord = rem % extent;
        rem /= extent;
        ia += coord * a_strides[d];
        im += coord * m_strides[d];
        ic += coord * c_strides[d];
        {"iin += coord * c_in_strides[d];" if has_c_in else ""}
    }}
    if (_m[im]) {{
        _c[ic] = {_unary_cpp_expr(op, '_a[ia]')};
    }}
    {"else { _c[ic] = _c_in[iin]; }" if has_c_in else ""}
}}
"""

        return nodes.Tasklet(
            label=node.name + "_cutile",
            inputs=inputs,
            outputs={"_c"},
            code=code,
            language=dtypes.Language.CPP,
        )


# ── Register masked unary ops ───────────────────────────────────────

_MASKED_UNARY_DISPLAY_NAMES = {
    "-": "TileMaskedNegate", "abs": "TileMaskedAbs", "sin": "TileMaskedSin",
    "cos": "TileMaskedCos", "exp": "TileMaskedExp", "sqrt": "TileMaskedSqrt",
    "log": "TileMaskedLog",
}

for _op in _UNARY_OPS:
    register_op(
        op=_op,
        tasklet_type=TaskletType.UNARY_ARRAY,
        mask=MaskType.RUNTIME,
        node_type=TileRuntimeMaskedUnaryOpLibraryNode,
        node_name=_MASKED_UNARY_DISPLAY_NAMES.get(_op, f"TileMaskedUnaryOp_{_op}"),
        out="_c",
        rhs1="_a",
        mask_in="_m",
        out_in="_c_in",
    )
