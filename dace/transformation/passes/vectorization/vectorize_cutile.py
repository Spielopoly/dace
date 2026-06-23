# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``VectorizeCuTile`` — the cuTile front-door orchestrator.

Composes the K-dim tile-op vectorizer
(:class:`~dace.transformation.passes.vectorization.vectorize_cpu_multi_dim.VectorizeCPUMultiDim`
with ``target_isa="CUTILE"`` and ``expand_tile_nodes=False``) with the six
cuTile lowering passes from
:mod:`~dace.transformation.passes.vectorization.cutile_lowering`, library-node
expansion, and the Python backend stamp. The result is an SDFG the cuTile code
generator (``dace/codegen/py/cutile_target.py``) compiles into ``cuda.tile``
Python kernels.
"""
from typing import Any, Dict, Literal, Optional, Set, Tuple, Type

from dace import SDFG, dtypes, properties, transformation
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.vectorization.cutile_lowering import (
    CuTileInsertDataCopies,
    CuTileSetGlobalStorage,
    CuTileSetImplementations,
    CuTileSetSchedules,
    CuTileSetTileStorage,
    CuTileValidateTiles,
)
from dace.transformation.passes.vectorization.remove_unused_per_lane_symbols import RemoveUnusedPerLaneSymbols
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim


@properties.make_properties
@transformation.explicit_cf_compatible
class VectorizeCuTile(ppl.Pass):
    """Vectorize an SDFG into cuTile kernels for the Python backend.

    Imperative stages, run once, in order (:class:`CanonicalizationPipeline`
    style — this is a :class:`~dace.transformation.pass_pipeline.Pass`, not a
    ``Pipeline``, because the lowering passes must run between the vectorizer
    and library-node expansion and the vectorizer is itself a Pipeline):

    1. ``VectorizeCPUMultiDim(widths=..., target_isa="CUTILE",
       expand_tile_nodes=False, ...)`` — emit ``tileops`` library nodes.
    2. :class:`CuTileValidateTiles` — anchors exist; widths are powers of 2.
    3. :class:`CuTileSetSchedules` — outermost tiled map -> ``CuTile``,
       inner chain -> ``Sequential``.
    4. :class:`CuTileSetTileStorage` — tile transients -> ``CuTile_Tile``.
    5. :class:`CuTileSetGlobalStorage` — kernel-touched non-transients ->
       ``GPU_Global``.
    6. :class:`CuTileInsertDataCopies` (optional, default on) — clone
       ``GPU_Global`` non-transients to device transients and insert
       copy-in/copy-out states so callers can pass NumPy host arrays.
    7. :class:`CuTileSetImplementations` — lib nodes -> ``target_isa="CUTILE"``,
       ``implementation="cutile"``.
    8. ``sdfg.expand_library_nodes()``
    9. ``sdfg.backend = dtypes.BackendLanguage.Python``

    Steps 2–9 are exactly the manual escape-hatch recipe: run the six
    lowering passes in that order with ``strict`` of your choice, then the two
    explicit core-API calls of steps 8 and 9.

    **Canonicalization is NOT run** (parity with ``VectorizeCPUMultiDim``):
    callers wanting the full front-door flow run
    ``dace.transformation.passes.canonicalize.canonicalize(sdfg)`` first.

    The cuTile configuration is pinned: ``target_isa`` is always ``"CUTILE"``
    and ``expand_tile_nodes`` is always ``False`` internally (expansion happens
    in step 7, after implementation stamping); neither is exposed as a knob.
    The remaining meaningful ``VectorizeCPUMultiDim`` knobs are forwarded
    verbatim. The inner vectorizer is constructed eagerly in ``__init__`` so
    an invalid configuration (non-power-of-2 widths, K outside 1..3, bad knob
    combination) raises ``NotImplementedError`` at construction time.
    """

    CATEGORY: str = "Vectorization"

    strict = properties.Property(dtype=bool,
                                 default=False,
                                 desc="When True, lowering-pass precondition violations raise "
                                 "ValueError instead of emitting a UserWarning.")

    insert_data_copies = properties.Property(
        dtype=bool,
        default=True,
        desc="When True, insert copy-in/copy-out states so the SDFG "
        "interface accepts numpy (host) arrays instead of requiring "
        "cupy (device) arrays.")

    def __init__(self,
                 widths: Tuple[int, ...],
                 *,
                 remainder_strategy: Literal["full_mask", "masked_tail", "scalar_postamble"] = "full_mask",
                 branch_mode: Literal["merge", "fp_factor"] = "merge",
                 loop_to_map_permissive: bool = False,
                 nest_map_bodies: bool = False,
                 strict: bool = False,
                 insert_data_copies: bool = True,
                 debug_save: bool = False):
        """Build the orchestrator (validates the configuration eagerly).

        :param widths: Per-dim tile widths, innermost-last (1..3 entries, all
            powers of 2 — a hard ``cuda.tile`` runtime requirement).
        :param remainder_strategy: Tile remainder handling, forwarded to
            :class:`VectorizeCPUMultiDim` (``"full_mask"`` default).
        :param branch_mode: Branch lowering, forwarded to
            :class:`VectorizeCPUMultiDim` (``"merge"`` default).
        :param loop_to_map_permissive: Forwarded; lets ``LoopToMap``
            parallelise scatter-style loops.
        :param nest_map_bodies: Forwarded; ``True`` routes every innermost map
            body through the NestedSDFG tile descent.
        :param strict: When ``True``, lowering-pass precondition violations
            (e.g. a partially-vectorized SDFG) raise ``ValueError`` instead of
            emitting a ``UserWarning``.
        :param insert_data_copies: When ``True`` (default), run
            :class:`CuTileInsertDataCopies` to insert copy-in/copy-out states
            so callers can pass NumPy host arrays instead of CuPy device arrays.
        :raises NotImplementedError: On any configuration
            :class:`VectorizeCPUMultiDim` rejects (raised here, at
            construction, not at ``apply_pass`` time).
        """
        super().__init__()
        self.strict = strict
        self.insert_data_copies = insert_data_copies
        self._debug_save = debug_save
        # Eager construction: VectorizeCPUMultiDim.__init__ validates the
        # whole knob row (widths count/powers of 2, remainder/branch combos),
        # so a bad config fails fast at orchestrator construction.
        self._vectorizer = VectorizeCPUMultiDim(widths=widths,
                                                target_isa="CUTILE",
                                                remainder_strategy=remainder_strategy,
                                                branch_mode=branch_mode,
                                                loop_to_map_permissive=loop_to_map_permissive,
                                                nest_map_bodies=nest_map_bodies,
                                                expand_tile_nodes=False)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> Set[Type[ppl.Pass]]:
        return set()

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Run the full cuTile pipeline on ``sdfg`` in place.

        :param sdfg: The SDFG to vectorize and lower.
        :param pipeline_results: Unused pipeline results.
        :returns: The number of cuTile kernels created (the
            :class:`CuTileSetSchedules` count), or ``None`` if nothing was
            tiled.
        :raises ValueError: When ``strict`` and a lowering precondition fails,
            or unconditionally on non-power-of-2 widths / a tile-op node type
            without a ``'cutile'`` implementation.
        """
        
        import os
        import time
        debug_dir = os.environ.get("DACE_CUTILE_DEBUG_DIR", ".cutile_pipeline_debug")
        DEBUG_SAVE_NAME = os.path.join(debug_dir, str(int(time.time())), "stage_{stage}.sdfg")
        
        stage = 0
        
        def debug_save_sdfg():
            nonlocal stage
            if self._debug_save:
                sdfg.save(DEBUG_SAVE_NAME.format(stage=stage))
                stage += 1

        debug_save_sdfg()
        self._vectorizer.apply_pass(sdfg, {})
        # The inner vectorizer runs with ``expand_tile_nodes=False``, so it
        # skips its own post-expansion ``RemoveUnusedPerLaneSymbols`` sweep.
        # The cuTile gather lowering materialises indices as ``_idx_<d>`` index
        # tiles (consumed by ``ct.gather``), which makes the per-lane scalar
        # fan-out symbols dead -- their *defining* interstate-edge assignments
        # still read the index arrays in host code, which validation rejects
        # once those arrays become ``GPU_Global``. Sweep them here. The pass
        # only removes symbols with zero remaining references, so it is a no-op
        # for paths where the per-lane symbols are genuinely used.
        RemoveUnusedPerLaneSymbols(sweep_dead_index_symbols=True).apply_pass(sdfg, {})
        debug_save_sdfg()
        CuTileValidateTiles(strict=self.strict).apply_pass(sdfg, {})
        debug_save_sdfg()
        num_kernels = CuTileSetSchedules(strict=self.strict).apply_pass(sdfg, {})
        CuTileSetTileStorage(strict=self.strict).apply_pass(sdfg, {})
        CuTileSetGlobalStorage(strict=self.strict).apply_pass(sdfg, {})
        debug_save_sdfg()
        if self.insert_data_copies:
            CuTileInsertDataCopies(strict=self.strict).apply_pass(sdfg, {})
        CuTileSetImplementations(strict=self.strict).apply_pass(sdfg, {})
        sdfg.backend = dtypes.BackendLanguage.Python
        debug_save_sdfg()
        return num_kernels
