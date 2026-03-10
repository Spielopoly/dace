"""
TileAdd Library Node for DaCe → cuTile (TileIR) backend.

Represents element-wise addition of two array tiles:
    C[subset] = A[subset] + B[subset]

The arrays A, B, C are standard DaCe arrays whose shape matches the tile
dimensions exactly (the caller is responsible for scoping this node inside
a Map that iterates over tiles, so that each invocation sees a tile-shaped
slice of the full arrays).

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


# ---------------------------------------------------------------------------
# Library node
# ---------------------------------------------------------------------------

@library.node
class TileAdd(LibraryNode):
    """
    DaCe Library Node: element-wise tile addition  C = A + B.

    Connectors
    ----------
    _a  (in)  : tile A  – shape must equal the tile dimensions
    _b  (in)  : tile B  – same shape as A
    _c  (out) : tile C  – same shape as A/B
    """

    # Registry of available implementations (populated by @library.register_expansion).
    implementations: dict = {}
    default_implementation = "cuTile"

    # --- node properties ---
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
        # Declare the three connectors up-front.
        super().__init__(
            name,
            inputs={"_a", "_b"},
            outputs={"_c"},
            **kwargs,
        )
        if tile_size is not None:
            self.tile_size = tile_size

    def validate(self, sdfg: SDFG, state: SDFGState):
        """
        Light-weight validation: check that both inputs and the output have
        the same shape and dtype.
        """
        a_node, b_node, c_node = None, None, None

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


# ---------------------------------------------------------------------------
# cuTile expansion
# ---------------------------------------------------------------------------

@library.register_expansion(TileAdd, "cuTile")
class ExpandTileAddCuTile(ExpandTransformation):
    """
    Expands TileAdd into a Python-language Tasklet that emits a cuTile kernel.

    The generated Tasklet
    ---------------------
    * defines a ``@ct.kernel`` function ``_tile_add_kernel`` that loads two
      tiles from global arrays, adds them element-wise, and stores the result,
    * calls ``ct.launch`` from the Tasklet body (host-side wrapper),
    * is tagged with ``dtypes.Language.Python`` so DaCe's Python codegen
      emits it verbatim into the generated module.

    Connector naming convention
    ---------------------------
    DaCe uses the connector name as the local variable name inside the Tasklet,
    so the kernel receives the raw array objects under ``_a``, ``_b``, ``_c``.
    """

    environments: list = []  # no external environment declarations needed yet

    @staticmethod
    def expansion(node: TileAdd, state: SDFGState, sdfg: SDFG):
        """
        Return a Tasklet node that replaces the library node in the SDFG.

        The caller (LibraryNode.expand) will re-wire the surrounding memlets
        onto the returned node's connectors automatically.
        """

        # ------------------------------------------------------------------
        # 1.  Recover array descriptors to determine shape / dtype.
        # ------------------------------------------------------------------
        a_desc, b_desc, c_desc = _get_tile_descriptors(node, state, sdfg)
        shape: tuple = a_desc.shape          # symbolic or concrete sizes
        ndim: int = len(shape)

        # ------------------------------------------------------------------
        # 2.  Derive tile_size list: prefer node property, else use shape.
        # ------------------------------------------------------------------
        if node.tile_size:
            tile_size_expr = repr(tuple(node.tile_size))
        else:
            # Build a Python tuple literal from the (possibly symbolic) shape.
            # At runtime the shape will be concrete integers.
            tile_size_expr = "(" + ", ".join(str(s) for s in shape) + ",)"

        # ------------------------------------------------------------------
        # 3.  Build the Tasklet code string.
        #
        #     cuTile convention:
        #       - index argument to ct.load / ct.store is the *block id* tuple
        #         (one entry per dimension).  Because each invocation of this
        #         Tasklet already operates on exactly one tile, the block id is
        #         (0, 0, ...) and the shape equals the full array shape.
        #       - The grid is therefore (1, 1, 1).
        #
        #     We wrap the kernel in a nested function so the Tasklet body is
        #     self-contained and can be emitted verbatim.
        # ------------------------------------------------------------------

        # Build a string like "(0,)" for 1-D or "(0, 0,)" for 2-D
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

        # ------------------------------------------------------------------
        # 4.  Create the Tasklet with the correct in/out connectors.
        # ------------------------------------------------------------------
        tasklet = nodes.Tasklet(
            label=node.name + "_cutile",
            inputs={"_a", "_b"},
            outputs={"_c"},
            code=code,
            language=dtypes.Language.Python,
        )

        return tasklet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_tile_descriptors(node: TileAdd, state: SDFGState, sdfg: SDFG):
    """
    Walk the edges attached to *node* in *state* and return the
    (a_desc, b_desc, c_desc) array descriptors.
    """
    a_desc = b_desc = c_desc = None

    for edge in state.in_edges(node):
        arr_name = edge.data.data          # name of the array on this memlet
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


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def add_tile_add_to_state(
    sdfg: SDFG,
    state: SDFGState,
    a_name: str,
    b_name: str,
    c_name: str,
    tile_size: list[int] | None = None,
) -> TileAdd:
    """
    Add a TileAdd library node to *state* and connect it to existing arrays
    *a_name*, *b_name* (inputs) and *c_name* (output) with full-array memlets.

    Parameters
    ----------
    sdfg       : The SDFG that owns the arrays.
    state      : The SDFGState to add the node into.
    a_name     : Name of array A in sdfg.arrays.
    b_name     : Name of array B in sdfg.arrays.
    c_name     : Name of array C in sdfg.arrays.
    tile_size  : Optional explicit tile dimensions.  If None, inferred from
                 the array shape at expansion time.

    Returns
    -------
    The TileAdd node that was added to the state.
    """
    a_desc = sdfg.arrays[a_name]
    b_desc = sdfg.arrays[b_name]
    c_desc = sdfg.arrays[c_name]

    # Create access nodes
    a_access = state.add_read(a_name)
    b_access = state.add_read(b_name)
    c_access = state.add_write(c_name)

    # Create library node
    lib_node = TileAdd(name="tile_add", tile_size=tile_size)
    state.add_node(lib_node)

    # Build full-shape subsets  (e.g. "0:N, 0:M" for a 2-D tile)
    def full_subset(desc) -> str:
        return ", ".join(f"0:{s}" for s in desc.shape)

    # Wire edges: AccessNode --> LibraryNode connector (inputs)
    state.add_edge(
        a_access, None,
        lib_node, "_a",
        Memlet(data=a_name, subset=full_subset(a_desc)),
    )
    state.add_edge(
        b_access, None,
        lib_node, "_b",
        Memlet(data=b_name, subset=full_subset(b_desc)),
    )
    # Wire output: LibraryNode connector --> AccessNode
    state.add_edge(
        lib_node, "_c",
        c_access, None,
        Memlet(data=c_name, subset=full_subset(c_desc)),
    )

    return lib_node


# ---------------------------------------------------------------------------
# Quick smoke-test (run with: python tile_add.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N = 128  # tile length

    # --- Build SDFG ---
    sdfg = SDFG("tile_add_test")

    # Declare arrays (GPU, float32, shape [N])
    sdfg.add_array("A", shape=[N], dtype=dace.float32,
                   storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("B", shape=[N], dtype=dace.float32,
                   storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("C", shape=[N], dtype=dace.float32,
                   storage=dtypes.StorageType.GPU_Global)

    state = sdfg.add_state("main")

    lib_node = add_tile_add_to_state(
        sdfg, state,
        a_name="A", b_name="B", c_name="C",
        tile_size=[N],
    )

    # --- Validate before expansion ---
    sdfg.validate()
    print("[OK] SDFG validates before expansion.")
    sdfg.save("tile_add_before_expansion.sdfg")  # for inspection in GraphViz or DaCe GUI

    # --- Expand library nodes ---
    sdfg.expand_library_nodes()
    sdfg.validate()
    print("[OK] SDFG validates after expansion.")
    sdfg.save("tile_add_expanded.sdfg")  # for inspection in GraphViz or DaCe GUI

    # Print the generated tasklet code for inspection
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.Tasklet):
            print("\n--- Expanded Tasklet code ---")
            print(node.code.as_string)

    print("\nDone. (Compile + run requires a Blackwell GPU with CUDA 13.1+)")