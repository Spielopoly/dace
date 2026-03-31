"""
Pipeline for applying cuTile transformations to an SDFG.

Usage::

    from dace.libraries.cutile.transformations.pipeline import apply_cutile_pipeline
    count = apply_cutile_pipeline(sdfg)
"""
from __future__ import annotations
from typing import Tuple

from dace.sdfg import SDFG
from dace.sdfg import nodes as sdfg_nodes
from dace.sdfg.construction_utils import normalize_conditional_blocks_in_nsdfg
from dace.libraries.cutile.transformations.scalar_to_tile_library import (
    ScalarToTileCanonical,
    ScalarToTileMasked,
)
from dace.libraries.cutile.transformations.if_else_to_where_select import (
    IfElseMapToTileWhere,
)
from dace.transformation.dataflow import TrivialChainElimination, MapTiling



def apply_cutile_pipeline(sdfg: SDFG, *,
                          validate: bool = True,
                          validate_all: bool = True,
                          apply_map_tiling: bool = True,
                          tile_shape: Tuple[int, ...] = (16, 16, 16)) -> int:
    """
    Apply the full cuTile transformation pipeline to an SDFG.

    Currently applies:

     1. **ScalarToTileLibraryCanonical** – replace canonical inner scalar maps
         with unmasked cuTile library nodes.
     2. **ScalarToTileLibraryMasked** – replace non-canonical inner scalar maps
         with runtime-masked cuTile library nodes.

    Parameters
    ----------
    sdfg : SDFG
        The SDFG to transform (modified in-place).
    validate : bool
        Validate the SDFG after the full pipeline.
    validate_all : bool
        Validate after every single transformation application.
    apply_map_tiling : bool
        Whether to apply MapTiling first

    Returns
    -------
    int
        Total number of transformations applied.
    """
    count = 0
    
    # Apply MapTiling to create tiled patterns
    if apply_map_tiling:
        options = {
            "tile_sizes": tile_shape,
            "skew": True,
        }
        count += sdfg.apply_transformations(
            [MapTiling],
            validate=validate_all,
            validate_all=validate_all,
            options=options,
        )


    # Phase 1: Replace scalar tasklets with library nodes
    count += sdfg.apply_transformations_once_everywhere(
        [
            TrivialChainElimination,
            ScalarToTileCanonical,
            ScalarToTileMasked,
        ],
        validate=validate_all,
        validate_all=validate_all,
    )

    # Phase 1.5: Normalize conditional blocks in NestedSDFGs
    # Split 2-branch if-else blocks into sequential single-branch blocks
    # and duplicate conditions across top-level nodes for each branch.
    # This pre-normalizes the structure so IfElseMapToTileWhere operates
    # on a simpler, uniform pattern.
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, sdfg_nodes.NestedSDFG):
                normalize_conditional_blocks_in_nsdfg(node.sdfg)

    # Phase 2: Replace if-else patterns with where-select library nodes
    count += sdfg.apply_transformations_once_everywhere(
        [IfElseMapToTileWhere],
        validate=validate_all,
        validate_all=validate_all,
    )

    if validate or validate_all:
        sdfg.validate()

    return count
