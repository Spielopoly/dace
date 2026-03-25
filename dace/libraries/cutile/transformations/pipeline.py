"""
Pipeline for applying cuTile transformations to an SDFG.

Usage::

    from dace.libraries.cutile.transformations.pipeline import apply_cutile_pipeline
    count = apply_cutile_pipeline(sdfg)
"""
from __future__ import annotations
from typing import Optional, Tuple

from dace.sdfg import SDFG
from dace.libraries.cutile.transformations.scalar_to_tile_library import (
    ScalarToTileLibraryCanonical,
    ScalarToTileLibraryMasked,
)
from dace.transformation.dataflow import TrivialChainElimination, MapTiling



def apply_cutile_pipeline(sdfg: SDFG, *,
                          validate: bool = True,
                          validate_all: bool = True,
                          apply_map_tiling: bool = True,
                          tile_shape: Optional[Tuple[int, ...]] = None) -> int:
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
            ScalarToTileLibraryCanonical,
            ScalarToTileLibraryMasked,
        ],
        validate=validate_all,
        validate_all=validate_all,
    )

    if validate or validate_all:
        sdfg.validate()

    return count
