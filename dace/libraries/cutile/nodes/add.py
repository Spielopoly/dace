"""
TileAdd Library Node for DaCe -> cuTile (TileIR) backend.

Represents element-wise addition of two array tiles:
    C[subset] = A[subset] + B[subset]

Expansion emits C++ tasklet code so the node can be expanded and validated
with DaCe's supported C++ code generation path.
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


@library.node
class TileAdd(LibraryNode):
    """
    DaCe Library Node: element-wise tile addition  C = A + B.

    Connectors
    ----------
    _a  (in)  : tile A
    _b  (in)  : tile B
    _c  (out) : tile C
    """

    implementations: dict = {}
    default_implementation = "cuTile"

    tile_size = properties.ListProperty(
        element_type=int,
        default=[],
        desc=(
            "Tile dimensions, e.g. [128] for a 1-D tile or [32, 32] for 2-D. "
            "When empty the shape is inferred from the incoming array descriptors "
            "at expansion time."
        ),
    )

    def __init__(self, name: str = "tile_add", tile_size: list[int] | None = None, **kwargs):
        super().__init__(
            name,
            inputs={"_a", "_b"},
            outputs={"_c"},
            **kwargs,
        )
        if tile_size is not None:
            self.tile_size = tile_size

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
            raise ValueError(
                f"TileAdd '{self.name}': all three connectors (_a, _b, _c) must be connected."
            )

        a_desc = sdfg.arrays[a_node.data]
        b_desc = sdfg.arrays[b_node.data]
        c_desc = sdfg.arrays[c_node.data]

        if a_desc.shape != b_desc.shape or a_desc.shape != c_desc.shape:
            raise ValueError(
                f"TileAdd '{self.name}': shape mismatch – "
                f"A={a_desc.shape}, B={b_desc.shape}, C={c_desc.shape}"
            )
        if a_desc.dtype != b_desc.dtype:
            raise ValueError(
                f"TileAdd '{self.name}': dtype mismatch – A={a_desc.dtype}, B={b_desc.dtype}"
            )
        if c_desc.dtype != a_desc.dtype:
            raise ValueError(
                f"TileAdd '{self.name}': dtype mismatch – A={a_desc.dtype}, C={c_desc.dtype}"
            )


@library.register_expansion(TileAdd, "cuTile")
class ExpandTileAddCuTile(ExpandTransformation):
    """Expands TileAdd into a C++ tasklet for element-wise tile addition."""

    environments: list = []

    @staticmethod
    def expansion(node: TileAdd, state: SDFGState, sdfg: SDFG):
        a_desc, b_desc, c_desc = _get_tile_descriptors(node, state, sdfg)
        shape: tuple = a_desc.shape
        ndim: int = len(shape)

        if ndim == 0:
            code = "_c[0] = _a[0] + _b[0];"
        else:
            shape_expr = ", ".join(symstr(s) for s in shape)
            a_strides_expr = ", ".join(symstr(s) for s in a_desc.strides)
            b_strides_expr = ", ".join(symstr(s) for s in b_desc.strides)
            c_strides_expr = ", ".join(symstr(s) for s in c_desc.strides)

            # Keep the operation generic via an operator policy lambda.
            code = f"""
constexpr int ndim = {ndim};
const std::size_t shape[ndim] = {{{shape_expr}}};
const std::size_t a_strides[ndim] = {{{a_strides_expr}}};
const std::size_t b_strides[ndim] = {{{b_strides_expr}}};
const std::size_t c_strides[ndim] = {{{c_strides_expr}}};

std::size_t n = 1;
for (int d = 0; d < ndim; ++d) {{
    n *= shape[d];
}}

auto apply_binary = [&](auto op) {{
    for (std::size_t linear = 0; linear < n; ++linear) {{
        std::size_t rem = linear;
        long long ia = 0;
        long long ib = 0;
        long long ic = 0;
        for (int d = ndim - 1; d >= 0; --d) {{
            const auto extent = static_cast<std::size_t>(shape[ d]);
            const long long coord = static_cast<long long>(rem % extent);
            rem /= extent;
            ia += coord * a_strides[ d];
            ib += coord * b_strides[ d];
            ic += coord * c_strides[ d];
        }}
        _c[ic] = op(_a[ia], _b[ib]);
    }}
}};

apply_binary([](const auto& lhs, const auto& rhs) {{
    return lhs + rhs;
}});
"""
        tasklet = nodes.Tasklet(
            label=node.name + "_cutile",
            inputs={"_a", "_b"},
            outputs={"_c"},
            code=code,
            language=dtypes.Language.CPP,
        )
        return tasklet


def _get_tile_descriptors(node: TileAdd, state: SDFGState, sdfg: SDFG):
    """Return (a_desc, b_desc, c_desc) array descriptors for a TileAdd node."""
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