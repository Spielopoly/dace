"""
Pipeline for applying cuTile transformations to an SDFG.

Usage::

    from dace.libraries.cutile.transformations.pipeline import apply_cutile_pipeline
    count = apply_cutile_pipeline(sdfg)
"""
from __future__ import annotations

from dace.sdfg import SDFG
from dace.libraries.cutile.transformations.scalar_to_tile_library import ScalarToTileLibrary


def apply_cutile_pipeline(sdfg: SDFG, validate: bool = True,
                          validate_all: bool = False) -> int:
    """
    Apply the full cuTile transformation pipeline to an SDFG.

    Currently applies:

    1. **ScalarToTileLibrary** – replace inner maps + scalar tasklets inside
       tile maps with cuTile library nodes.

    Parameters
    ----------
    sdfg : SDFG
        The SDFG to transform (modified in-place).
    validate : bool
        Validate the SDFG after the full pipeline.
    validate_all : bool
        Validate after every single transformation application.

    Returns
    -------
    int
        Total number of transformations applied.
    """
    count = 0

    # Phase 1: Replace scalar tasklets with library nodes
    count += sdfg.apply_transformations_repeated(
        ScalarToTileLibrary,
        validate=validate_all,
        validate_all=validate_all,
    )

    if validate:
        sdfg.validate()

    return count
