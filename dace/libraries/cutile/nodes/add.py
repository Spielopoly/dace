"""
TileAdd Library Node for DaCe → cuTile (TileIR) backend.

Represents element-wise addition of two array tiles:
    C[subset] = A[subset] + B[subset]

Expansion emits a Python-language Tasklet that calls into cuTile Python
(cuda.tile) to perform the addition as a single tile operation on the GPU.
"""
from __future__ import annotations

import dace
from dace import dtypes, properties, Memlet
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg.nodes import LibraryNode
from dace import library
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


@library.register_expansion(TileAdd, "cuTile")
class ExpandTileAddCuTile(ExpandTransformation):
    """Expands TileAdd into a Python-language Tasklet emitting a cuTile kernel."""

    environments: list = []

    @staticmethod
    def expansion(node: TileAdd, state: SDFGState, sdfg: SDFG):
        a_desc, b_desc, c_desc = _get_tile_descriptors(node, state, sdfg)
        shape: tuple = a_desc.shape
        ndim: int = len(shape)

        if node.tile_size:
            tile_size_expr = repr(tuple(node.tile_size))
        else:
            tile_size_expr = "(" + ", ".join(str(s) for s in shape) + ",)"

        zero_index = "(" + ", ".join(["0"] * ndim) + ",)"

        code = f"""\
import cuda.tile as __ct
import cupy as __cp

@__ct.kernel
def _tile_add_kernel(_ka, _kb, _kc, _tile_shape: __ct.Constant[tuple]):
    _a_tile = __ct.load(_ka, index={zero_index}, shape=_tile_shape)
    _b_tile = __ct.load(_kb, index={zero_index}, shape=_tile_shape)
    _c_tile = _a_tile + _b_tile
    __ct.store(_kc, index={zero_index}, tile=_c_tile)

_tile_shape = {tile_size_expr}
_grid = (1, 1, 1)
__ct.launch(
    __cp.cuda.get_current_stream(),
    _grid,
    _tile_add_kernel,
    (_a, _b, _c, _tile_shape),
)
"""
        tasklet = nodes.Tasklet(
            label=node.name + "_cutile",
            inputs={"_a", "_b"},
            outputs={"_c"},
            code=code,
            language=dtypes.Language.Python,
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