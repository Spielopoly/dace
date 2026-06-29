# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``VectorizeCuTile`` — the cuTile front-door orchestrator.

Composes the K-dim tile-op vectorizer
(:class:`~dace.transformation.passes.vectorization.vectorize_cpu_multi_dim.VectorizeCPUMultiDim`
with ``target_isa="CUTILE"`` and ``expand_tile_nodes=False``) with
``sdfg.apply_gpu_transformations()`` for GPU scheduling/storage/data-copy
handling, the tileops-specific lowering passes from
:mod:`~dace.transformation.passes.vectorization.cutile_lowering`, and the
Python backend stamp.  The result is an SDFG the cuTile code generator
(``dace/codegen/py/cutile_target.py``) compiles into ``cuda.tile`` Python
kernels.
"""
from typing import Any, Dict, Literal, Optional, Set, Tuple, Type

from dace import SDFG, dtypes, properties, transformation
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies
from dace.transformation.passes.vectorization.cutile_lowering import (
    CuTileSetImplementations,
    CuTileSetTileStorage,
    CuTileValidateTiles,
    GPUDeviceToCuTile,
)
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
    3. ``sdfg.apply_gpu_transformations(...)`` — GPU scheduling, storage
       stamping, and host/device data copies (replaces the old manual
       ``CuTileSetSchedules``, ``CuTileSetGlobalStorage``, and
       ``CuTileInsertDataCopies`` passes).
    4. :class:`GPUDeviceToCuTile` — adapter: re-stamp tileops-anchored
       outermost maps from ``GPU_Device`` to ``CuTile``.
    5. :class:`CuTileSetTileStorage` — ``Register`` tile transients inside
       CuTile scopes become ``CuTile_Tile``.
    6. :class:`CuTileSetImplementations` — lib nodes ->
       ``target_isa="CUTILE"``, ``implementation="cutile"``.
    6b. :class:`InsertExplicitCopies` (when ``insert_explicit_copies``, the
        default) — lift the implicit host<->device ``AccessNode -> AccessNode``
        copy edges that ``apply_gpu_transformations()`` created into explicit
        ``CopyLibraryNode`` instances, giving a uniform IR where every copy is an
        explicit node.  The copy nodes expand to bare Tasklets:
        ``_cpy_out = _cpy_in`` for same-storage (``ExpandPython``),
        ``dst.set(src)`` for H2D (``ExpandPythonH2D``),
        ``src.get(out=dst)`` for D2H (``ExpandPythonD2H``).  The
        tileops-anchored tile loads/stores (``CuTile_Tile`` storage) are excluded
        from :class:`InsertExplicitCopies` and are unaffected. Set
        ``insert_explicit_copies=False`` to keep the raw copy edges instead.
    7. ``sdfg.backend = dtypes.BackendLanguage.Python``

    **Canonicalization is NOT run** (parity with ``VectorizeCPUMultiDim``):
    callers wanting the full front-door flow run
    ``dace.transformation.passes.canonicalize.canonicalize(sdfg)`` first.

    The cuTile configuration is pinned: ``target_isa`` is always ``"CUTILE"``
    and ``expand_tile_nodes`` is always ``False`` internally (expansion happens
    in step 6, after implementation stamping); neither is exposed as a knob.
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

    insert_explicit_copies = properties.Property(
        dtype=bool,
        default=True,
        desc="When True (the default), run InsertExplicitCopies after "
        "implementation stamping to lift the implicit host<->device copy edges "
        "created by apply_gpu_transformations() into explicit CopyLibraryNode "
        "instances (uniform explicit-copy IR; runtime unchanged). Set False to "
        "keep the raw copy edges, which the cuTile/Python backend lowers natively.")

    def __init__(self,
                 widths: Tuple[int, ...],
                 *,
                 remainder_strategy: Literal["full_mask", "masked_tail", "scalar_postamble"] = "full_mask",
                 branch_mode: Literal["merge", "fp_factor"] = "merge",
                 loop_to_map_permissive: bool = False,
                 nest_map_bodies: bool = False,
                 strict: bool = False,
                 insert_explicit_copies: bool = True,
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
        :param insert_explicit_copies: When ``True`` (the default), run
            :class:`~dace.transformation.passes.insert_explicit_copies.InsertExplicitCopies`
            after implementation stamping, lifting the implicit host<->device
            ``AccessNode -> AccessNode`` copies created by
            ``apply_gpu_transformations()`` into explicit ``CopyLibraryNode``
            instances.  The copy nodes expand to bare Tasklets:
            ``_cpy_out = _cpy_in`` for same-storage (``ExpandPython``),
            ``dst.set(src)`` for H2D (``ExpandPythonH2D``),
            ``src.get(out=dst)`` for D2H (``ExpandPythonD2H``).
            Set ``False`` to keep the raw copy edges instead.
        :param debug_save: When ``True``, save intermediate SDFG files
            after each pipeline stage for debugging.
        :raises NotImplementedError: On any configuration
            :class:`VectorizeCPUMultiDim` rejects (raised here, at
            construction, not at ``apply_pass`` time).
        """
        super().__init__()
        self.strict = strict
        self.insert_explicit_copies = insert_explicit_copies
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
            :class:`GPUDeviceToCuTile` count), or ``None`` if nothing was
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

        # Step 1: Vectorize — emit tileops library nodes
        debug_save_sdfg()
        self._vectorizer.apply_pass(sdfg, {})
        debug_save_sdfg()

        # Step 2: Validate — anchors exist, widths are powers of 2
        CuTileValidateTiles(strict=self.strict).apply_pass(sdfg, {})
        debug_save_sdfg()

        # Step 3: GPU transform — scheduling, storage, data copies
        sdfg.apply_gpu_transformations(
            sequential_innermaps=True,
            register_transients=True,
            simplify=True,
        )
        debug_save_sdfg()

        # Step 4: Adapter — re-stamp tileops-anchored maps GPU_Device -> CuTile
        num_kernels = GPUDeviceToCuTile(strict=self.strict).apply_pass(sdfg, {})
        debug_save_sdfg()

        # Step 5: Tile storage — Register transients in CuTile scopes -> CuTile_Tile
        CuTileSetTileStorage(strict=self.strict).apply_pass(sdfg, {})
        debug_save_sdfg()

        # Step 6: Stamp cuTile implementations on library nodes
        CuTileSetImplementations(strict=self.strict).apply_pass(sdfg, {})
        debug_save_sdfg()

        # Step 6b (default on): lift implicit host<->device copies into explicit
        # CopyLibraryNode instances. tileops tile loads/stores use CuTile_Tile
        # storage, which InsertExplicitCopies excludes, so only the
        # apply_gpu_transformations()-created CPU_Heap<->GPU_Global copies are
        # lifted. Their expansions emit: _cpy_out = _cpy_in for same-storage
        # (ExpandPython, bare Tasklet), dst.set(src) for H2D (ExpandPythonH2D,
        # NestedSDFG), src.get(out=dst) for D2H (ExpandPythonD2H, NestedSDFG).
        if self.insert_explicit_copies:
            InsertExplicitCopies().apply_pass(sdfg, {})
            debug_save_sdfg()

        # Step 7: Python backend stamp
        sdfg.backend = dtypes.BackendLanguage.Python
        debug_save_sdfg()

        return num_kernels
