"""
Masked cuTile binary operation library nodes.

Implements element-wise masked binary operations:
    if mask[idx]:
        C[idx] = OP(A[idx], B[idx])

When the mask is false, the output element is left untouched.
"""
from __future__ import annotations

from typing import Callable, Optional, cast

import dace
from dace import dtypes, properties
from dace import library
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg.nodes import LibraryNode
from dace.sdfg.validation import InvalidSDFGNodeError
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from ..op_registry import TaskletType, MaskType, register_matcher


@library.node
class TileRuntimeMaskedBinaryOPLibraryNode(LibraryNode):
    """
    Library node for masked binary operation on 2 tiles:
        if M: C = foo(A, B)
        else: C unchanged
    
    The mask is read at runtime, so this library node can be used for patterns where the mask condition cannot be resolved at compile time.
    

    Connectors
    ----------
    _a  (in)  : tile A
    _b  (in)  : tile B
    _m  (in)  : tile mask (same shape as A/B/C)
    _c_in (in, optional): initial tile C values used when mask is false
    _c  (out) : tile C
    """

    implementations: dict = {}
    default_implementation = "pure"

    tile_shape = properties.ListProperty(
        element_type=int,
        default=None,
        desc=(
            "Tile dimensions, e.g. [128] for a 1-D tile or [32, 32] for 2-D."
            "0-D (scalar) tiles can be represented with an empty list []."
            "When None, the shape is inferred from the incoming array descriptors "
            "at expansion time."
        ),
        allow_none=True,
    )

    def __init__(self,
                 name: str = "TileMaskedBinaryOP",
                 tile_shape: list[int] | None = None,
                 **kwargs):
        super().__init__(
            name,
            inputs={"_a", "_b", "_m"},
            outputs={"_c"},
            **kwargs,
        )
        self.tile_shape = tile_shape

    def validate(self, sdfg: SDFG, state: SDFGState):
        a_node = b_node = m_node = c_node = c_in_node = None
        for edge in state.in_edges(self):
            if edge.dst_conn == "_a":
                a_node = edge.src
            elif edge.dst_conn == "_b":
                b_node = edge.src
            elif edge.dst_conn == "_m":
                m_node = edge.src
            elif edge.dst_conn == "_c_in":
                c_in_node = edge.src
        for edge in state.out_edges(self):
            if edge.src_conn == "_c":
                c_node = edge.dst

        if a_node is None or b_node is None or m_node is None or c_node is None:
            raise InvalidSDFGNodeError(
                f"TileMaskedBinaryOP '{self.name}': connectors (_a, _b, _m, _c) must all be connected.",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )

        a_desc = sdfg.arrays[a_node.data]
        b_desc = sdfg.arrays[b_node.data]
        m_desc = sdfg.arrays[m_node.data]
        c_desc = sdfg.arrays[c_node.data]

        if a_desc.shape != b_desc.shape or a_desc.shape != c_desc.shape or a_desc.shape != m_desc.shape:
            raise InvalidSDFGNodeError(
                f"TileMaskedBinaryOP '{self.name}': shape mismatch - "
                f"A={a_desc.shape}, B={b_desc.shape}, M={m_desc.shape}, C={c_desc.shape}",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )

        if a_desc.dtype != b_desc.dtype:
            raise InvalidSDFGNodeError(
                f"TileMaskedBinaryOP '{self.name}': dtype mismatch - A={a_desc.dtype}, B={b_desc.dtype}",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )
        if c_desc.dtype != a_desc.dtype:
            raise InvalidSDFGNodeError(
                f"TileMaskedBinaryOP '{self.name}': dtype mismatch - A={a_desc.dtype}, C={c_desc.dtype}",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )

        if c_in_node is not None:
            c_in_desc = sdfg.arrays[c_in_node.data]
            if c_in_desc.shape != c_desc.shape:
                raise InvalidSDFGNodeError(
                    f"TileMaskedBinaryOP '{self.name}': shape mismatch - "
                    f"C_in={c_in_desc.shape}, C={c_desc.shape}",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )
            if c_in_desc.dtype != c_desc.dtype:
                raise InvalidSDFGNodeError(
                    f"TileMaskedBinaryOP '{self.name}': dtype mismatch - "
                    f"C_in={c_in_desc.dtype}, C={c_desc.dtype}",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )

        supported_mask_dtypes = {
            dace.bool,
            dace.int8,
            dace.uint8,
            dace.int16,
            dace.uint16,
            dace.int32,
            dace.uint32,
            dace.int64,
            dace.uint64,
        }
        if m_desc.dtype not in supported_mask_dtypes:
            raise InvalidSDFGNodeError(
                f"TileMaskedBinaryOP '{self.name}': mask dtype must be bool or integer, got M={m_desc.dtype}",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )


class ExpandTileElementWiseRuntimeMaskedBinaryOPPure(ExpandTransformation):
    """Expand TileRuntimeMaskedBinaryOPLibraryNode into a C++ tasklet."""

    environments: list = []

def _expansion_with_op(node: LibraryNode, parent_state: SDFGState,
                        parent_sdfg: SDFG,
                        binary_op_string_generator: Callable[[str, str], str]):
    masked_node = cast(TileRuntimeMaskedBinaryOPLibraryNode, node)
    a_desc, b_desc, m_desc, c_desc, c_in_desc = _get_masked_tile_descriptors(masked_node, parent_state, parent_sdfg)
    has_c_in = c_in_desc is not None

    tile_shape = getattr(masked_node, "tile_shape", None)
    shape: tuple[int] = tuple(tile_shape) if tile_shape is not None else a_desc.shape
    ndim: int = len(shape)

    try:
        n_total = 1
        for s in shape:
            n_total *= int(s)
        use_scalar_form = (ndim == 0) or (n_total == 1)
    except (TypeError, ValueError):
        use_scalar_form = (ndim == 0)

    if use_scalar_form:
        if has_c_in:
            code = (
                f"if (_m) {{ _c = {binary_op_string_generator('_a', '_b')}; }} "
                f"else {{ _c = _c_in; }}"
            )
        else:
            code = f"if (_m) {{ _c = {binary_op_string_generator('_a', '_b')}; }}"
    else:
        shape_expr = ", ".join(symstr(s) for s in shape)
        a_strides_expr = ", ".join(symstr(s) for s in a_desc.strides)
        b_strides_expr = ", ".join(symstr(s) for s in b_desc.strides)
        m_strides_expr = ", ".join(symstr(s) for s in m_desc.strides)
        c_strides_expr = ", ".join(symstr(s) for s in c_desc.strides)
        c_in_strides_expr = ", ".join(symstr(s) for s in c_in_desc.strides) if has_c_in else ""

        code = f"""
constexpr int ndim = {ndim};
const std::size_t shape[ndim] = {{{shape_expr}}};
const std::ptrdiff_t a_strides[ndim] = {{{a_strides_expr}}};
const std::ptrdiff_t b_strides[ndim] = {{{b_strides_expr}}};
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
    std::size_t ib = 0;
    std::size_t im = 0;
    std::size_t ic = 0;
    {"std::size_t iin = 0;" if has_c_in else ""}
    for (int d = ndim - 1; d >= 0; --d) {{
        const auto extent = shape[d];
        const std::size_t coord = rem % extent;
        rem /= extent;
        ia += coord * a_strides[d];
        ib += coord * b_strides[d];
        im += coord * m_strides[d];
        ic += coord * c_strides[d];
        {"iin += coord * c_in_strides[d];" if has_c_in else ""}
    }}
    if (_m[im]) {{
        _c[ic] = {binary_op_string_generator('_a[ia]', '_b[ib]')};
    }}
    {"else { _c[ic] = _c_in[iin]; }" if has_c_in else ""}
}}
"""

    inputs = {"_a", "_b", "_m"}
    if has_c_in:
        inputs.add("_c_in")
    tasklet = nodes.Tasklet(
        label=masked_node.name + "_cutile",
        inputs=inputs,
        outputs={"_c"},
        code=code,
        language=dtypes.Language.CPP,
    )
    return tasklet


def _get_masked_tile_descriptors(node: TileRuntimeMaskedBinaryOPLibraryNode, state: SDFGState,
                                 sdfg: SDFG) -> tuple[dace.data.Data, dace.data.Data, dace.data.Data, dace.data.Data, Optional[dace.data.Data]]:
    """Return (a_desc, b_desc, m_desc, c_desc, c_in_desc) array descriptors for node."""
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
    if None in (a_desc, b_desc, m_desc, c_desc):
        raise ValueError(
            f"TileMaskedAdd expansion: could not resolve all array descriptors for "
            f"node '{node.name}'. Make sure _a, _b, _m, _c are all connected."
        )
    return (
        cast(dace.data.Data, a_desc),
        cast(dace.data.Data, b_desc),
        cast(dace.data.Data, m_desc),
        cast(dace.data.Data, c_desc),
        cast(Optional[dace.data.Data], c_in_desc),
    )


@register_matcher(op="+", tasklet_type=TaskletType.ARRAY_ARRAY, mask=MaskType.RUNTIME,
                  node_name="TileAdd", out="_c", rhs1="_a", rhs2="_b", mask_in="_m", out_in="_c_in")
@library.node
class TileRuntimeMaskedAddLibraryNode(TileRuntimeMaskedBinaryOPLibraryNode):
    """Masked tile addition: if M then C = A + B else keep C unchanged."""

    def __init__(self,
                 name: str = "TileMaskedAdd",
                 tile_shape: list[int] | None = None,
                 **kwargs):
        super().__init__(
            name=name,
            tile_shape=tile_shape,
            **kwargs,
        )


@library.register_expansion(TileRuntimeMaskedAddLibraryNode, "pure")  # type: ignore[arg-type]
class ExpandTileRuntimeMaskedAddPure(ExpandTileElementWiseRuntimeMaskedBinaryOPPure):
    """Expand TileRuntimeMaskedAddLibraryNode into a C++ tasklet."""

    @staticmethod
    def expansion(node: TileRuntimeMaskedAddLibraryNode, state: SDFGState, sdfg: SDFG) -> nodes.Tasklet:
        return _expansion_with_op(
            node,
            state,
            sdfg,
            binary_op_string_generator=lambda a, b: f"{a} + {b}",
        )


@register_matcher(op="-", tasklet_type=TaskletType.ARRAY_ARRAY, mask=MaskType.RUNTIME,
                  node_name="TileSubtract", out="_c", rhs1="_a", rhs2="_b", mask_in="_m", out_in="_c_in")
@library.node
class TileRuntimeMaskedSubtractLibraryNode(TileRuntimeMaskedBinaryOPLibraryNode):
    """Masked tile subtraction: if M then C = A - B else keep C unchanged."""

    def __init__(self,
                 name: str = "TileRuntimeMaskedSubtract",
                 tile_shape: list[int] | None = None,
                 **kwargs):
        super().__init__(
            name=name,
            tile_shape=tile_shape,
            **kwargs,
        )


@library.register_expansion(TileRuntimeMaskedSubtractLibraryNode, "pure")  # type: ignore[arg-type]
class ExpandTileRuntimeMaskedSubtractPure(ExpandTileElementWiseRuntimeMaskedBinaryOPPure):
    """Expand TileRuntimeMaskedSubtractLibraryNode into a C++ tasklet."""

    @staticmethod
    def expansion(node: TileRuntimeMaskedSubtractLibraryNode, state: SDFGState, sdfg: SDFG) -> nodes.Tasklet:
        return _expansion_with_op(
            node,
            state,
            sdfg,
            binary_op_string_generator=lambda a, b: f"{a} - {b}",
        )
