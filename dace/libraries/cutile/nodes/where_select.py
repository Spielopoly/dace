"""
TileWhereSelectLibraryNode – element-wise conditional selection on tiles.

Implements::

    C[idx] = cond[idx] ? X[idx] : Y[idx]

for every element index *idx* in the tile. This mirrors ``cuda.tile.where``
from the NVIDIA cuTile library and is the tile-level analogue of
``numpy.where(cond, x, y)``.

All three input tiles must have the same shape, and the condition tile must
be of boolean or integer type.
"""


from typing import List, Optional

import dace
from dace import dtypes, library, properties
from dace.data.core import Array
from dace.sdfg import SDFG, SDFGState, nodes
from dace.sdfg.validation import InvalidSDFGNodeError
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation

from .base import resolve_shape_and_scalar_form, TileNodeBase, SUPPORTED_MASK_DTYPES


# ── Helper to read tile descriptors for where-select ─────────────────

def _get_where_descriptors(
    node: nodes.LibraryNode,
    state: SDFGState,
    sdfg: SDFG,
) -> tuple[Array, Array, Array, Array]:
    """Return ``(cond_desc, x_desc, y_desc, c_desc)`` for *node*.

    Walks the incoming and outgoing edges of the where-select *node* to
    resolve the array descriptors for all four connectors.

    Args:
        node: A :class:`TileWhereSelectLibraryNode` instance.
        state: The SDFG state containing *node*.
        sdfg: The SDFG owning the state.

    Returns:
        A 4-tuple ``(cond_desc, x_desc, y_desc, c_desc)``.

    Raises:
        :class:`dace.sdfg.validation.InvalidSDFGNodeError`: If any of the
            four connectors is not connected.
    """
    cond_desc = x_desc = y_desc = c_desc = None
    for edge in state.in_edges(node):
        arr_name = edge.data.data
        if arr_name is None:
            continue
        if edge.dst_conn == "_cond":
            cond_desc = sdfg.arrays[arr_name]
        elif edge.dst_conn == "_x":
            x_desc = sdfg.arrays[arr_name]
        elif edge.dst_conn == "_y":
            y_desc = sdfg.arrays[arr_name]
    for edge in state.out_edges(node):
        arr_name = edge.data.data
        if arr_name is None:
            continue
        if edge.src_conn == "_c":
            c_desc = sdfg.arrays[arr_name]
    if any(desc is None for desc in [cond_desc, x_desc, y_desc, c_desc]):
        raise InvalidSDFGNodeError(
            f"TileWhereSelect expansion: Not all inputs connected for node '{node.name}'.",
            sdfg=sdfg,
            state_id=state.parent_graph.node_id(state),
            node_id=state.node_id(node),
        )
    return cond_desc, x_desc, y_desc, c_desc # type: ignore


# ── Library node ─────────────────────────────────────────────────────

@library.node
class TileWhereSelectLibraryNode(TileNodeBase):
    """
    Element-wise conditional selection: ``C = where(cond, X, Y)``.

    For each tile element::

        C[idx] = cond[idx] ? X[idx] : Y[idx]

    Connectors
    ----------
    _cond (in)  : boolean / integer mask tile
    _x    (in)  : values selected when ``cond`` is true
    _y    (in)  : values selected when ``cond`` is false
    _c    (out) : result tile
    """

    implementations: dict = {}
    default_implementation = "pure"

    def __init__(self, name: str = "TileWhereSelect",
                 tile_shape: Optional[List[int]] = None,
                 **kwargs) -> None:
        """Initialise the element-wise conditional selection node.

        Args:
            name: Node display name in the SDFG (default
                ``"TileWhereSelect"``).
            tile_shape: Fixed tile extents, or ``None`` to infer at expansion
                from the connected input descriptors.
            **kwargs: Forwarded to :class:`TileNodeBase`.
        """
        super().__init__(
            name,
            inputs={"_cond", "_x", "_y"},
            outputs={"_c"},
            **kwargs,
        )
        self.tile_shape = tile_shape

    def validate(self, sdfg: SDFG, state: SDFGState) -> None:
        """Validate the where-select node before code generation.

        Checks that all four connectors are wired, that every input tile
        shares the same shape as the output, that the condition tile has a
        boolean or integer dtype, and that ``X`` and ``Y`` share the output
        dtype.

        Args:
            sdfg: The SDFG containing this node.
            state: The state containing this node.

        Raises:
            :class:`~dace.sdfg.validation.InvalidSDFGNodeError`: If any
                shape or dtype constraint is violated.
        """
        cond_desc, x_desc, y_desc, c_desc = _get_where_descriptors(
            self, state, sdfg)

        # All shapes must match
        for tag, desc in [("X", x_desc), ("Y", y_desc), ("cond", cond_desc)]:
            if desc.shape != c_desc.shape:
                raise InvalidSDFGNodeError(
                    f"TileWhereSelect '{self.name}': shape mismatch — "
                    f"{tag}={desc.shape}, C={c_desc.shape}",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )

        # Condition must be boolean or integer
        if cond_desc.dtype not in SUPPORTED_MASK_DTYPES:
            raise InvalidSDFGNodeError(
                f"TileWhereSelect '{self.name}': cond dtype must be bool or "
                f"integer, got {cond_desc.dtype}",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )

        # X, Y, C dtypes must match
        if x_desc.dtype != c_desc.dtype:
            raise InvalidSDFGNodeError(
                f"TileWhereSelect '{self.name}': dtype mismatch — "
                f"X={x_desc.dtype}, C={c_desc.dtype}",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )
        if y_desc.dtype != c_desc.dtype:
            raise InvalidSDFGNodeError(
                f"TileWhereSelect '{self.name}': dtype mismatch — "
                f"Y={y_desc.dtype}, C={c_desc.dtype}",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )


# ── C++ expansion ────────────────────────────────────────────────────

@library.register_expansion(TileWhereSelectLibraryNode, "pure")
class ExpandTileWhereSelectPure(ExpandTransformation):
    """Expand TileWhereSelectLibraryNode into a C++ element-wise tasklet."""

    environments: list = []

    @staticmethod
    def expansion(node: TileWhereSelectLibraryNode, state: SDFGState,
                  sdfg: SDFG) -> nodes.Tasklet:
        """Expand the node into a C++ element-wise selection tasklet.

        Generates ``_c[i] = _cond[i] ? _x[i] : _y[i]`` for every element
        index *i*, or the scalar form ``_c = _cond ? _x : _y`` for
        single-element tiles.

        Args:
            node: The :class:`TileWhereSelectLibraryNode` to expand.
            state: The SDFG state containing *node*.
            sdfg: The SDFG owning the state.

        Returns:
            A :class:`~dace.sdfg.nodes.Tasklet` implementing the where-select
            in C++.
        """
        cond_desc, x_desc, y_desc, c_desc = _get_where_descriptors(
            node, state, sdfg)

        ref_desc = c_desc
        shape, ndim, use_scalar_form = resolve_shape_and_scalar_form(
            node, ref_desc)

        if use_scalar_form:
            code = "_c = _cond ? _x : _y;"
        else:
            shape_expr = ", ".join(symstr(s) for s in shape)
            cond_strides = ", ".join(symstr(s) for s in cond_desc.strides)
            x_strides = ", ".join(symstr(s) for s in x_desc.strides)
            y_strides = ", ".join(symstr(s) for s in y_desc.strides)
            c_strides = ", ".join(symstr(s) for s in c_desc.strides)

            code = f"""
constexpr int ndim = {ndim};
const std::size_t shape[ndim] = {{{shape_expr}}};
const std::ptrdiff_t cond_strides[ndim] = {{{cond_strides}}};
const std::ptrdiff_t x_strides[ndim] = {{{x_strides}}};
const std::ptrdiff_t y_strides[ndim] = {{{y_strides}}};
const std::ptrdiff_t c_strides[ndim] = {{{c_strides}}};

std::size_t n = 1;
for (int d = 0; d < ndim; ++d) {{
    n *= shape[d];
}}

for (std::size_t i = 0; i < n; ++i) {{
    std::size_t rem = i;
    std::size_t i_cond = 0, ix = 0, iy = 0, ic = 0;
    for (int d = ndim - 1; d >= 0; --d) {{
        const auto extent = shape[d];
        const std::size_t coord = rem % extent;
        rem /= extent;
        i_cond += coord * cond_strides[d];
        ix += coord * x_strides[d];
        iy += coord * y_strides[d];
        ic += coord * c_strides[d];
    }}
    _c[ic] = _cond[i_cond] ? _x[ix] : _y[iy];
}}
"""

        return nodes.Tasklet(
            label=node.name + "_cutile",
            inputs={"_cond", "_x", "_y"},
            outputs={"_c"},
            code=code,
            language=dtypes.Language.CPP,
        )
