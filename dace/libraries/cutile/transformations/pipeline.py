"""
Pipeline for applying cuTile transformations to an SDFG.

Usage::

    from dace.libraries.cutile.transformations.pipeline import apply_cutile_pipeline
    count = apply_cutile_pipeline(sdfg)
"""
from __future__ import annotations

from .utils import duplicate_conditions_for_whole_sdfgs
from dace.sdfg import SDFG
from dace.sdfg import nodes as sdfg_nodes
from dace.sdfg.state import ConditionalBlock
from .scalar_to_tile_library import (
    ScalarToTileCanonical,
    ScalarToTileMasked,
)
from .if_else_to_where_select import (
    IfElseMapToTileWhere,
)
from dace.transformation.dataflow import MapTiling, TrivialTaskletElimination, TrivialChainElimination
from dace.transformation.interstate.loop_lifting import LoopLifting
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes.split_tasklets import SplitTasklets
from dace.transformation.dataflow.map_fission import MapFission
from dace.transformation import pass_pipeline as ppl


def _simplify(sdfg: SDFG):
    """Helper function to apply a few simplification transformations before the main pipeline."""
    sdfg.apply_transformations_repeated([TrivialChainElimination])
    sdfg.apply_transformations_repeated([TrivialTaskletElimination])
    sdfg.simplify()


def apply_cutile_pipeline(sdfg: SDFG, *,
                          validate: bool = True,
                          validate_all: bool = True,
                          apply_map_tiling: bool = True,
                          tile_shape: tuple[int, ...] = (16, 16, 16),
                          debug_save_sdfg_steps: bool = False) -> int:
    """
    Apply the full cuTile transformation pipeline to an SDFG.

    Pipeline stages:

     1. **Simplify** – trivial tasklet/chain elimination and standard simplify.
     2. **LoopLifting / LoopToMap** – convert state-machine loops into maps.
     3. **SplitTasklets** – split multi-statement tasklets into single ops.
     4. **MapFission** – fission maps into single-operation maps.
     5. **Simplify** – clean up after preprocessing.
     6. **MapTiling** – tile maps to the given tile shape.
     7. **Normalize conditional blocks** in NestedSDFGs.
     8. **ScalarToTileCanonical / ScalarToTileMasked / IfElseMapToTileWhere**
         – replace scalar tasklets with cuTile library nodes.

    Parameters
    ----------
    sdfg : SDFG
        The SDFG to transform (modified in-place).
    validate : bool
        Validate the SDFG after the full pipeline.
    validate_all : bool
        Validate after every single transformation application.
    apply_map_tiling : bool
        Whether to apply MapTiling.
    tile_shape : tuple[int, ...]
        Tile sizes for MapTiling.

    Returns
    -------
    int
        Total number of transformations applied.
    """
    count = 0
    
    _pipeline_step = -1
    import time
    current_time = time.strftime("%Y%m%d-%H%M%S")
    dirname = f"/workspace/cutile_pipeline_debug_sdfgs/{current_time}"
    import os
    os.makedirs(dirname, exist_ok=True)
    def debug_save_sdfg():
        if debug_save_sdfg_steps:
            nonlocal _pipeline_step
            _pipeline_step += 1
            name = f"{dirname}/step_{_pipeline_step}.sdfg"
            
            sdfg.save(name)
    
    debug_save_sdfg()
    
    # Step 1: Initial simplification
    # Removes trivial tasklets and unnecessary access nodes
    _simplify(sdfg)
    debug_save_sdfg()

    # ── Preprocessing: Canonicalize the SDFG for tiling ──────────────

    # Step 2: Convert loops to maps where possible
    # LoopLifting promotes detected state-machine loops into explicit
    # LoopRegion constructs; LoopToMap then converts eligible loops to maps.
    count += sdfg.apply_transformations_repeated(
        [LoopLifting], validate=False, validate_all=False,
    )
    count += sdfg.apply_transformations_repeated(
        [LoopToMap], validate=False, validate_all=False,
    )
    debug_save_sdfg()

    # Step 3: Split multi-statement tasklets into single-operation tasklets
    # This enables pattern matching against individual operations for
    # tile library node replacement.
    split_result = SplitTasklets().apply_pass(sdfg, {})
    if split_result is not None:
        count += split_result if isinstance(split_result, int) else 1
    debug_save_sdfg()

    # Step 4: Fission maps with complex subgraphs into single-operation maps
    # Each resulting map should have exactly one computational node,
    # which can then be matched against tile library node patterns.
    # NOTE: Skip MapFission when the SDFG contains NestedSDFGs with
    # ConditionalBlocks (if-else patterns).  MapFission enters an infinite
    # loop on those because each fission creates new matchable patterns.
    _has_conditional = any(
        isinstance(block, ConditionalBlock)
        for state in sdfg.all_states()
        for node in state.nodes()
        if isinstance(node, sdfg_nodes.NestedSDFG)
        for block in node.sdfg.all_control_flow_regions()
    )
    if not _has_conditional:
        count += sdfg.apply_transformations_repeated(
            [MapFission], validate=False, validate_all=False,
        )
    debug_save_sdfg()

    # Step 5: Clean up after preprocessing
    _simplify(sdfg)
    debug_save_sdfg()

    # Step 6: Apply MapTiling to create tiled patterns
    if apply_map_tiling:
        options = {
            "tile_sizes": tile_shape,
            "skew": True,
        }
        count += sdfg.apply_transformations_once_everywhere(
            [MapTiling],
            validate=validate_all,
            validate_all=validate_all,
            options=options,
        )
    debug_save_sdfg()

    # Step 7: Normalize conditional blocks in NestedSDFGs
    # Duplicate conditions across top-level nodes for each branch.
    # TODO: Not sure if this is actually useful for this pipeline
    # We'll need more tests for this, and actually implement it properly
    # So far we only have tests for sdfgs where this does not apply
    duplicate_conditions_for_whole_sdfgs(sdfg)
    debug_save_sdfg()

    # Step 8: Replace scalar tasklets with library nodes
    count += sdfg.apply_transformations_repeated(
        [
            ScalarToTileCanonical,
            ScalarToTileMasked,
            IfElseMapToTileWhere
        ],
        validate=validate_all,
        validate_all=validate_all,
    )
    debug_save_sdfg()

    if validate or validate_all:
        sdfg.validate()

    return count
