"""
Pipeline for applying cuTile transformations to an SDFG.

Usage::

    from dace.libraries.cutile.transformations.pipeline import apply_cutile_pipeline
    count = apply_cutile_pipeline(sdfg)

The pipeline is also available as a :class:`CuTilePipeline` Pass that can be
composed with other DaCe passes::

    from dace.libraries.cutile.transformations.pipeline import CuTilePipeline
    pipeline = CuTilePipeline(apply_map_collapse_and_tiling=True, tile_shape=(16, 16, 16))
    result = pipeline.apply_pass(sdfg, {})
"""


import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple, Type, Union

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
from dace.transformation.dataflow import MapTiling, TrivialTaskletElimination, MapFission, MapCollapse
from dace.transformation.interstate.loop_lifting import LoopLifting
from dace.transformation.interstate.loop_to_map import LoopToMap
from dace.transformation.passes.split_tasklets import SplitTasklets
from dace.transformation import pass_pipeline as ppl
from .remove_intermediate_transient import RemoveIntermediateTransient



# ---------------------------------------------------------------------------
# CuTilePipeline — the main pipeline Pass
# ---------------------------------------------------------------------------


@dataclass(unsafe_hash=True)
class CuTilePipeline(ppl.Pass):
    """Full cuTile transformation pipeline as a DaCe :class:`~dace.transformation.pass_pipeline.Pass`.
    """

    CATEGORY: str = 'cuTile'

    apply_map_collapse_and_tiling: bool = True  # Controls steps 3 (MapCollapse) and 7 (MapTiling)
    tile_shape: Tuple[int, ...] = (16, 16, 16)
    validate: bool = True
    validate_all: bool = True
    debug_save_sdfg_steps: bool = False

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> Set[Union[Type[ppl.Pass], ppl.Pass]]:
        return set()

    def _simplify(self, sdfg: SDFG) -> None:
        """Trivial tasklet elimination followed by standard simplification."""
        sdfg.apply_transformations_repeated([TrivialTaskletElimination])
        sdfg.simplify()
        sdfg.apply_transformations_repeated([RemoveIntermediateTransient])

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]):
        # Debug snapshot setup
        debug_step = -1
        debug_dirname: Optional[str] = None
        if self.debug_save_sdfg_steps:
            debug_dirname = f"/workspace/cutile_pipeline_debug_sdfgs/{time.strftime('%Y%m%d-%H%M%S')}"
            os.makedirs(debug_dirname, exist_ok=True)

        def debug_save() -> None:
            nonlocal debug_step
            if self.debug_save_sdfg_steps:
                debug_step += 1
                sdfg.save(f"{debug_dirname}/step_{debug_step}.sdfg")

        count = 0
        debug_save()  # Save initial state

        # Step 1: Initial simplification
        self._simplify(sdfg)
        debug_save()

        # Step 2: Loop canonicalization (LoopLifting + LoopToMap)
        count += sdfg.apply_transformations_repeated(
            [LoopLifting, LoopToMap], validate=self.validate_all, validate_all=self.validate_all,
        )
        debug_save()

        # Step 3: MapCollapse
        if self.apply_map_collapse_and_tiling:
            count += sdfg.apply_transformations_repeated(
                [MapCollapse], validate=self.validate_all, validate_all=self.validate_all,
            )
        debug_save()

        # Step 4: Early handling of ConditionalBlock maps.
        # Tile maps containing ConditionalBlock NestedSDFGs and convert
        # them to TileIfElseOpLibraryNode BEFORE SplitTasklets breaks
        # multi-statement tasklets inside the branches.
        if self.apply_map_collapse_and_tiling:
            count += _apply_map_tiling_to_all_maps(
                sdfg, tile_shape=self.tile_shape, validate=self.validate_all,
                only_conditional_block_maps=True,
            )
            duplicate_conditions_for_whole_sdfgs(sdfg)
            count += sdfg.apply_transformations_repeated(
                [IfElseMapToTileWhere],
                validate=self.validate_all,
                validate_all=self.validate_all,
            )
        debug_save()

        # Step 5: Split multi-statement tasklets
        SplitTasklets().apply_pass(sdfg, pipeline_results)
        debug_save()

        # Step 6: Fission maps
        count += sdfg.apply_transformations_repeated(
            [MapFission], validate=self.validate_all, validate_all=self.validate_all,
        )
        debug_save()
        
        # Step 7
        # MapFission may produce directly-nested 1-D maps (e.g. fission of
        # an outer loop that contained an inner loop map).  Collapse them so
        # that the subsequent tiling step sees a single multi-dimensional map.
        if self.apply_map_collapse_and_tiling:
            count += sdfg.apply_transformations_repeated(
                [MapCollapse], validate=self.validate_all, validate_all=self.validate_all,
            )
        debug_save()

        # Step 8: Clean up after preprocessing
        self._simplify(sdfg)
        debug_save()

        # Step 9: Map tiling (optional)
        if self.apply_map_collapse_and_tiling:
            count += _apply_map_tiling_to_all_maps(
                sdfg, tile_shape=self.tile_shape, validate=self.validate_all,
            )
        debug_save()

        # Step 10: Normalize conditional blocks in NestedSDFGs
        duplicate_conditions_for_whole_sdfgs(sdfg)
        debug_save()

        # Step 11: Apply IfElseMapToTileWhere for patterns that weren't
        # handled in Step 3b (e.g. frontend patterns that needed MapFission first).
        count += sdfg.apply_transformations_repeated(
            [IfElseMapToTileWhere],
            validate=self.validate_all,
            validate_all=self.validate_all,
        )
        debug_save()

        # Step 12: ScalarToTile transformations
        count += sdfg.apply_transformations_repeated(
            [ScalarToTileCanonical, ScalarToTileMasked],
            validate=self.validate_all,
            validate_all=self.validate_all,
        )
        debug_save()

        # Step 13: Final clean up
        self._simplify(sdfg)
        debug_save()

        if self.validate or self.validate_all:
            sdfg.validate()

        pipeline_results['cutile_pipeline_count'] = count
        return pipeline_results


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _apply_map_tiling_to_all_maps(sdfg: SDFG,
                                   tile_shape: Tuple[int, ...],
                                   validate: bool = False,
                                   only_conditional_block_maps: bool = False) -> int:
    """Apply MapTiling to all MapEntry nodes in the SDFG.

    Collects all MapEntry nodes before tiling begins, then applies MapTiling
    to each one individually.  This ensures that all original maps are tiled
    without re-tiling newly created tile maps.

    Exception handling: ``ValueError`` raised by ``MapTiling.apply_to()``
    (e.g. incompatible map structure) is caught and silently skipped.
    Unexpected exceptions propagate for visibility.

    Args:
        sdfg: The SDFG to transform.
        tile_shape: Tile sizes for :class:`~dace.transformation.dataflow.MapTiling`.
        validate: Whether to validate the SDFG after each transformation.

    Returns:
        Total number of :class:`~dace.transformation.dataflow.MapTiling`
        transformations applied.
    """
    count = 0

    # Collect all MapEntry nodes from all states before any tiling
    map_entries_to_tile = []
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, sdfg_nodes.MapEntry):
                map_entries_to_tile.append((state, node))
            elif isinstance(node, sdfg_nodes.NestedSDFG):
                count += _apply_map_tiling_to_all_maps(node.sdfg, tile_shape, validate,
                                                       only_conditional_block_maps)

    # Apply MapTiling to each original MapEntry exactly once
    options = {
        "tile_sizes": tile_shape,
        "skew": True,
        "tile_trivial": True,
    }

    for state, map_entry in map_entries_to_tile:
        # Check that the map entry still exists in the state
        # (it may have been removed or transformed by previous tiling)
        if map_entry not in state.nodes():
            continue

        # If only tiling maps that contain ConditionalBlock NestedSDFGs, skip others
        if only_conditional_block_maps:
            scope_nodes = state.scope_subgraph(map_entry).nodes()
            has_conditional = False
            for sn in scope_nodes:
                if isinstance(sn, sdfg_nodes.NestedSDFG):
                    for cfr in sn.sdfg.all_control_flow_regions():
                        if isinstance(cfr, ConditionalBlock):
                            has_conditional = True
                            break
                if has_conditional:
                    break
            if not has_conditional:
                continue

        # Verify the node is still a valid MapEntry before attempting
        if not isinstance(map_entry, sdfg_nodes.MapEntry):
            continue

        try:
            # Apply MapTiling directly to this map entry.
            # MapTiling.apply_to() raises ValueError if the transformation
            # cannot be applied (e.g., map structure incompatible with tiling).
            MapTiling.apply_to(sdfg, options=options, map_entry=map_entry, verify=True)
            count += 1
            if validate:
                sdfg.validate()
        except ValueError:
            # MapTiling.apply_to() raises ValueError when can_be_applied() fails
            # or when transformation preconditions are not met. Skip this map.
            continue

    return count


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def apply_cutile_pipeline(sdfg: SDFG, *,
                          validate: bool = True,
                          validate_all: bool = True,
                          apply_map_collapse_and_tiling: bool = True,
                          tile_shape: Tuple[int, ...] = (16, 16, 16),
                          debug_save_sdfg_steps: bool = False) -> int:
    """Apply the full cuTile transformation pipeline to an SDFG.

    This is a convenience wrapper around :class:`CuTilePipeline`.  See that
    class for the full list of pipeline stages.

    Args:
        sdfg: The SDFG to transform (modified in-place).
        validate: Validate the SDFG after the full pipeline.
        validate_all: Validate after every single transformation application.
        apply_map_collapse_and_tiling: Whether to apply
            :class:`~dace.transformation.dataflow.MapCollapse` (step 3) and
            :class:`~dace.transformation.dataflow.MapTiling` (step 7).
        tile_shape: Tile sizes for
            :class:`~dace.transformation.dataflow.MapTiling`.
        debug_save_sdfg_steps: When ``True``, save the SDFG to disk after
            each pipeline step for debugging.

    Returns:
        Total number of transformations applied.
    """
    pipeline = CuTilePipeline(
        apply_map_collapse_and_tiling=apply_map_collapse_and_tiling,
        tile_shape=tile_shape,
        validate=validate,
        validate_all=validate_all,
        debug_save_sdfg_steps=debug_save_sdfg_steps,
    )
    result = pipeline.apply_pass(sdfg, {})
    return result['cutile_pipeline_count'] if result is not None else 0
