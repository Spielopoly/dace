# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
import collections
import dace
import dace.libraries.linalg.environments as environments

from dace import library, nodes, properties
from dace.utils import prod as _prod
from dace.libraries.blas import blas_helpers
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation


@library.expansion
class ExpandPure(ExpandTransformation):
    """ Implements the pure expansion of Addition on tiles."""

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        ...


@library.expansion
class ExpandCuTile(ExpandTransformation):
    environments = []  # TODO: add cuTile environment

    @staticmethod
    def expansion(node, state, sdfg):
        # Build a nested SDFG or Tasklet that calls cuTile Python
        # The tasklet will contain inline Python/CUDA via cuTile
        tasklet = state.add_tasklet(
            name="tile_add_cutile",
            inputs={"_a", "_b"},
            outputs={"_c"},
            code="""
# TODO: actual code for tile addition
""",
            language=dace.dtypes.Language.Python
        )
        return tasklet

@library.node
class AddCutile(nodes.LibraryNode):
    """ Implements tile addition. """


    implementations = {"pure": ExpandPure, "cutile": ExpandCuTile}
    default_implementation = "pure"

    tile_dimensions = properties.ListProperty(element_type=int, default=[], desc="Tile dimensions. Equal for all inputs/outputs.")

    def __init__(self, name, tile_dimensions=[], *args, **kwargs):
        super().__init__(name, *args, inputs={"_in_tile_a", "_in_tile_b"}, outputs={"_out_tile"}, **kwargs)
        self.tile_dimensions = tile_dimensions
    