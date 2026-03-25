"""
TileBinaryOPLibraryNode Library Node for DaCe -> cuTile (TileIR) backend.

Represents element-wise binary operation on two array tiles:
    C[subset] = A[subset] OP B[subset]

Expansion emits C++ tasklet code so the node can be expanded and validated
with DaCe's supported C++ code generation path.
"""
from __future__ import annotations
from typing import Callable

import dace
from dace import dtypes, properties
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg.nodes import LibraryNode
from dace import library
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from dace.sdfg.validation import (InvalidSDFGError, InvalidSDFGNodeError, InvalidSDFGEdgeError,
                                  InvalidSDFGInterstateEdgeError, NodeNotExpandedError)
from ..op_registry import register_matcher, MaskType, TaskletType

@library.node
class TileBinaryOPLibraryNode(LibraryNode):
    """
    Library node for any binary operation on 2 tiles: C = foo(A, B)
    where all tiles have the same shape and dtype.
    
    Connectors
    ----------
    _a  (in)  : tile A
    _b  (in)  : tile B
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
    
    

    def __init__(self, name: str = "TileBinaryOP", binary_op = None, tile_shape: list[int] | None = None, **kwargs):
        super().__init__(
            name,
            inputs={"_a", "_b"},
            outputs={"_c"},
            **kwargs,
        )
        self.tile_shape = tile_shape

    def validate(self, sdfg: SDFG, state: SDFGState):
        a_node = b_node = c_node = None
        for edge in state.in_edges(self):
            if edge.dst_conn == "_a":
                a_node = edge.src
            elif edge.dst_conn == "_b":
                b_node = edge.src
        for edge in state.out_edges(self):
            if edge.src_conn == "_c":
                c_node = edge.dst

        if a_node is None or b_node is None or c_node is None:
            raise InvalidSDFGNodeError(
                f"TileBinaryOP '{self.name}': all three connectors (_a, _b, _c) must be connected.",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )

        a_desc = sdfg.arrays[a_node.data]
        b_desc = sdfg.arrays[b_node.data]
        c_desc = sdfg.arrays[c_node.data]

        if a_desc.shape != b_desc.shape or a_desc.shape != c_desc.shape:
            raise InvalidSDFGNodeError(
                f"TileBinaryOP '{self.name}': shape mismatch – "
                f"A={a_desc.shape}, B={b_desc.shape}, C={c_desc.shape}",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )
        # TODO: cuTile actually has some flexibility here, which we could implement in the future.
        if a_desc.dtype != b_desc.dtype:
            raise InvalidSDFGNodeError(
                f"TileBinaryOP '{self.name}': dtype mismatch – A={a_desc.dtype}, B={b_desc.dtype}",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )
        if c_desc.dtype != a_desc.dtype:
            raise InvalidSDFGNodeError(
                f"TileBinaryOP '{self.name}': dtype mismatch – A={a_desc.dtype}, C={c_desc.dtype}",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )

# This does not need to get registered as it is only used as a base class for the actual expansions (TileAdd, TileSubtract, etc.) which are registered.
class ExpandTileElementWiseBinaryOPPure(ExpandTransformation):
    """Expands TileBinaryOPLibraryNode into a C++ tasklet for element-wise binary operations."""

    environments: list = []

def _tile_binary_op_cpp_expansion(node: TileBinaryOPLibraryNode, parent_state: SDFGState, parent_sdfg: SDFG, binary_op_string_generator: Callable[[str, str], str]):
    """Expand a ``TileBinaryOPLibraryNode`` into a C++ tasklet that performs
    element-wise binary computation over a tile.
    This method builds tasklet source code dynamically based on tile dimensionality:
    for scalar tiles (0D), it emits a single assignment; for N-D tiles, it generates
    index-linearization logic using shape and stride metadata, then applies the
    provided binary operation to each element pair from inputs ``_a`` and ``_b``,
    writing results to ``_c``.
    
    Parameters
    ----------
    node : TileBinaryOPLibraryNode
        The library node being expanded. Supplies tile metadata (e.g., ``tile_shape``)
        and naming information used for the generated tasklet.
    state : SDFGState
        The SDFG state that contains ``node``. Used to resolve data descriptors and
        context needed for expansion.
    sdfg : SDFG
        The parent SDFG graph. Used together with ``state``/``node`` to retrieve
        array descriptors (shape/strides) for input and output tiles.
    binary_op_string_generator : Callable[[str, str], str]
        Callback that receives two C++ operand expressions (for elements from ``_a``
        and ``_b``) and returns a C++ expression string implementing the desired
        binary operation. Example: lambda a, b: f"{a} + {b}" for addition.
    Returns
    -------
    nodes.Tasklet
        A C++ tasklet node with inputs ``_a``, ``_b`` and output ``_c`` containing
        generated code for element-wise tile-wise binary operation.
    """
    a_desc, b_desc, c_desc = _get_tile_descriptors(node, parent_state, parent_sdfg)
    shape: tuple[int] = tuple(node.tile_shape) if node.tile_shape is not None else a_desc.shape
    ndim: int = len(shape)

    # Determine whether DaCe will collapse the tile connector to a scalar.
    # This happens when the tile is 0-D (ndim == 0) OR when every dimension
    # is concretely 1 (single element). In both cases the C++ connector
    # variable is `double _a`, not `double *_a`, so we must NOT subscript it.
    try:
        n_total = 1
        for s in shape:
            n_total *= int(s)
        use_scalar_form = (ndim == 0) or (n_total == 1)
    except (TypeError, ValueError):
        use_scalar_form = (ndim == 0)

    if use_scalar_form:
        code = f"_c = {binary_op_string_generator('_a', '_b')};"
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

// Calculate total number of elements in the tile
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
    _c[ic] = {binary_op_string_generator('_a[ia]', '_b[ib]')};
}}
"""
    tasklet = nodes.Tasklet(
        label=node.name + "_cutile",
        inputs={"_a", "_b"},
        outputs={"_c"},
        code=code,
        language=dtypes.Language.CPP,
    )
    return tasklet


def _get_tile_descriptors(node: TileBinaryOPLibraryNode, state: SDFGState, sdfg: SDFG) -> tuple[dace.data.Data, dace.data.Data, dace.data.Data]:
    """Return (a_desc, b_desc, c_desc) array descriptors for a TileBinaryOPLibraryNode node."""
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
    if None in (a_desc, b_desc, c_desc):
        raise ValueError(
            f"TileAdd expansion: could not resolve all array descriptors for "
            f"node '{node.name}'. Make sure _a, _b, _c are all connected."
        )
    return a_desc, b_desc, c_desc

@register_matcher(op="+", tasklet_type=TaskletType.ARRAY_ARRAY, mask=MaskType.UNMASKED,
                  node_name="TileAdd", out="_c", rhs1="_a", rhs2="_b")
@library.node
class TileAddLibraryNode(TileBinaryOPLibraryNode):
    """Tile addition: C = A + B"""

    def __init__(self, name: str = "TileAdd", tile_shape: list[int] | None = None, **kwargs):
        super().__init__(
            name=name,
            tile_shape=tile_shape,
            **kwargs,
        )

@library.register_expansion(TileAddLibraryNode, "pure") # type: ignore
class ExpandTileAddPure(ExpandTileElementWiseBinaryOPPure):
    """Expands TileAddLibraryNode into a C++ tasklet that performs element-wise addition over a tile."""

    @staticmethod
    def expansion(node: TileAddLibraryNode, state: SDFGState, sdfg: SDFG) -> nodes.Tasklet:
        return _tile_binary_op_cpp_expansion(
            node, state, sdfg,
            binary_op_string_generator=lambda a, b: f"{a} + {b}"
        )

@register_matcher(op="-", tasklet_type=TaskletType.ARRAY_ARRAY, mask=MaskType.UNMASKED,
                  node_name="TileSubtract", out="_c", rhs1="_a", rhs2="_b")
@library.node
class TileSubtractLibraryNode(TileBinaryOPLibraryNode):
    """Tile subtraction: C = A - B"""

    def __init__(self, name: str = "TileSubtract", tile_shape: list[int] | None = None, **kwargs):
        super().__init__(
            name=name,
            tile_shape=tile_shape,
            **kwargs,
        )

@library.register_expansion(TileSubtractLibraryNode, "pure") # type: ignore
class ExpandTileSubtractPure(ExpandTileElementWiseBinaryOPPure):
    """Expands TileSubtractLibraryNode into a C++ tasklet that performs element-wise subtraction over a tile."""

    @staticmethod
    def expansion(node: TileSubtractLibraryNode, state: SDFGState, sdfg: SDFG) -> nodes.Tasklet:
        return _tile_binary_op_cpp_expansion(
            node, state, sdfg,
            binary_op_string_generator=lambda a, b: f"{a} - {b}"
        )
