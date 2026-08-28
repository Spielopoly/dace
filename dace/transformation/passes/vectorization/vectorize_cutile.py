# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``VectorizeCuTile`` — the cuTile front-door orchestrator.

Composes the K-dim tile-op vectorizer
(:class:`~dace.transformation.passes.vectorization.vectorize_multi_dim.VectorizeMultiDim`
with a cuTile-tailored
:class:`~dace.transformation.passes.vectorization.config.VectorizeConfig`:
``target_isa=ISA.CUTILE``, ``expand_tile_nodes=False``, ``assume_even=False``,
``assumption_guard=False``) with ``sdfg.apply_gpu_transformations()`` for GPU
scheduling/storage/data-copy handling, the tileops-specific lowering passes from
:mod:`~dace.transformation.passes.vectorization.cutile_lowering`, and the
Python backend stamp.  The result is an SDFG the cuTile code generator
(``dace/codegen/py/cutile_target.py``) compiles into ``cuda.tile`` Python
kernels.
"""
import warnings
from typing import Any, Dict, Optional, Set, Tuple, Type, Union

from dace import SDFG, data, dtypes, properties, transformation
from dace.sdfg import nodes
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.cutile_lowering import (
    CuTileSetImplementations,
    CuTileSetLibraryImplementations,
    CuTileSetTileStorage,
    CuTileValidateTiles,
    GPUDeviceToCuTile,
    _collect_tile_nodes,
    clamp_propagated_oob_memlets,
    mark_tile_op_memlets_allow_oob,
)
from dace.transformation.passes.vectorization.enums import ISA, BranchMode, RemainderStrategy
from dace.transformation.passes.vectorization.vectorize_multi_dim import VectorizeMultiDim, _has_gpu_device_map as has_gpu_device_map


@properties.make_properties
@transformation.explicit_cf_compatible
class VectorizeCuTile(ppl.Pass):
    """
    Vectorization pass for cuTile
    """

    CATEGORY: str = "Vectorization"

    strict = properties.Property(dtype=bool,
                                 default=False,
                                 desc="When True, lowering-pass precondition violations raise "
                                 "ValueError instead of emitting a UserWarning.")

    run_canonicalize = properties.Property(dtype=bool,
                                           default=True,
                                           desc="When True, run the canonicalize pipeline as step 0 "
                                           "(before vectorization).")

    use_gpu_storage = properties.Property(dtype=bool,
                                          default=False,
                                          desc="When True, apply auto_optimize's apply_gpu_storage before GPU "
                                          "scheduling: only non-transients with Default storage (written "
                                          "non-transient scalars included) become GPU_Global, so no "
                                          "host<->device copy states are generated and the compiled SDFG "
                                          "requires device (cupy) arrays. On an SDFG already GPU-transformed "
                                          "with existing staging clones the copy states remain (a UserWarning "
                                          "is emitted).")

    def __init__(self,
                 widths: Tuple[int, ...],
                 *,
                 remainder_strategy: Union[RemainderStrategy, str] = RemainderStrategy.FULL_MASK,
                 branch_mode: Union[BranchMode, str] = BranchMode.MERGE,
                 loop_to_map_permissive: bool = False,
                 strict: bool = False,
                 run_canonicalize: bool = True,
                 use_gpu_storage: bool = False,
                 debug_save: bool = False):
        """Build the orchestrator (validates the configuration eagerly).

        :param widths: Per-dim tile widths, innermost-last (1..3 entries, all
            powers of 2 — a hard ``cuda.tile`` runtime requirement).
        :param remainder_strategy: Tile remainder handling, forwarded to the
            inner vectorizer (``full_mask`` default).
        :param branch_mode: Branch lowering, forwarded to the inner
            vectorizer (``merge`` default).
        :param loop_to_map_permissive: Forwarded; lets ``LoopToMap``
            parallelise scatter-style loops.
        :param strict: When ``True``, lowering-pass precondition violations
            (e.g. a partially-vectorized SDFG) raise ``ValueError`` instead of
            emitting a ``UserWarning``.
        :param run_canonicalize: When ``True`` (default), run the canonicalize
            pipeline as step 0 before vectorization.
        :param use_gpu_storage: When ``True``, apply ``auto_optimize``'s
            ``apply_gpu_storage``: only non-transients with ``Default``
            storage (written non-transient scalars included) become
            ``GPU_Global``, so no host<->device copy states are generated and
            the compiled SDFG requires device (cupy) arrays. On an SDFG
            already GPU-transformed with existing staging clones the copy
            states remain (a ``UserWarning`` is emitted).
        :param debug_save: When ``True``, save intermediate SDFG files
            after each pipeline stage for debugging.
        :raises NotImplementedError: On any configuration
            :class:`VectorizeMultiDim` rejects (raised here, at construction,
            not at ``apply_pass`` time).
        """
        super().__init__()
        self.strict = strict
        self.run_canonicalize = run_canonicalize
        self.use_gpu_storage = use_gpu_storage
        self._debug_save = debug_save
        # Eager construction: VectorizeMultiDim.__init__ validates the whole
        # knob row (widths count/powers of 2, remainder/branch combos), so a
        # bad config fails fast at orchestrator construction.
        self._config = VectorizeConfig(
            widths=widths,
            target_isa=ISA.CUTILE,
            remainder_strategy=remainder_strategy,
            branch_mode=branch_mode,
            loop_to_map_permissive=loop_to_map_permissive,
            expand_tile_nodes=False,
            validate=True,
            assume_even=False,
            assumption_guard=False,
        )
        self._vectorizer = VectorizeMultiDim(self._config)

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> Set[Type[ppl.Pass]]:
        return set()

    @staticmethod
    def canonicalize_for_cutile(sdfg: SDFG) -> SDFG:
        """Run canonicalize with the cuTile knob row (step 0 of this pass).

        The canonical form is width-independent, so callers compiling one
        program at several tile widths should call this ONCE on the base SDFG
        and then run ``VectorizeCuTile(..., run_canonicalize=False)`` on
        deep copies — canonicalize dominates pipeline time.

        Knobs: ``target="gpu"`` (cuTile is a GPU backend);
        ``semantic_lifting=False`` keeps raw maps the vectorizer can lower;
        ``assumption_guard=False`` drops the terminal CPP ``__builtin_trap``
        guard the Python backend cannot codegen; ``reduction_to_wcr_map=False``
        keeps accumulator loops out of the privatized-WCR-map shape the tiler
        refuses; ``peel_limit=0`` skips ``BestEffortLoopPeeling`` — its
        per-stuck-loop probing dominates canonicalize time (~105 of 117s on
        adi) and the boundary-conflict loops it unblocks don't occur on the
        cuTile corpus.

        :param sdfg: The SDFG to canonicalize in place.
        :returns: The same ``sdfg`` instance.
        """
        # Deferred import: the canonicalize pipeline imports from this
        # (vectorization) subpackage, so a module-level import would be
        # circular.
        from dace.transformation.passes.canonicalize import canonicalize
        return canonicalize(sdfg,
                            target="gpu",
                            semantic_lifting=False,
                            assumption_guard=False,
                            reduction_to_wcr_map=False,
                            peel_limit=0)

    @staticmethod
    def _apply_gpu_transformations(sdfg: SDFG):
        return sdfg.apply_gpu_transformations(
            sequential_innermaps=True,
            register_transients=True,
            simplify=False,
        )

    @staticmethod
    def _has_device_staging_copy(sdfg: SDFG, name: str) -> bool:
        """Return whether ``name`` has a direct device-staging copy.

        :param sdfg: The SDFG to inspect.
        :param name: The non-transient array name.
        :returns: Whether a host/device copy edge connects the array to a
            transient ``GPU_Global`` descriptor.
        """
        for state in sdfg.states():
            for edge in state.edges():
                if not (isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.AccessNode)):
                    continue
                if edge.src.data == name:
                    clone_name = edge.dst.data
                elif edge.dst.data == name:
                    clone_name = edge.src.data
                else:
                    continue
                clone = sdfg.arrays.get(clone_name)
                if clone is not None and clone.transient and clone.storage == dtypes.StorageType.GPU_Global:
                    return True
        return False

    @staticmethod
    def _remove_trivial_gpu_maps(sdfg: SDFG) -> int:
        """Strip provably single-iteration ``GPU_Device`` maps.

        ``GPUTransformSDFG`` wraps every free (map-less) tasklet in a trivial
        ``0:1`` one-param ``*_gmap`` kernel map. In the GPU-first order those
        wrappers exist BEFORE the vectorizer runs and poison it twice:
        ``MarkTileDims`` (``require_gpu_resident=True``) rejects them for
        ``K >= 2`` widths ("has only 1 params"), and for ``K = 1`` the
        conversion passes tile-convert their scalar bodies even though the
        map is not a tile candidate — emitting ``ct.*`` tile ops into what
        the tail later demotes to a HOST step. Eliminating the wrapper
        recreates the map-less shape the legacy (device=CPU) order vectorizes;
        a wrapper ``TrivialMapElimination`` declines stays ``Sequential``
        (the tail treats it as a tiny host step).

        Order matters: ``TrivialMapElimination`` refuses GPU-scheduled maps
        ("syntactically needed"), so the trivial ``GPU_Device`` maps are
        demoted to ``Sequential`` first and eliminated second. Elimination is
        applied ONLY to the just-demoted entries (not SDFG-wide), so
        legitimate pre-existing ``0:1`` maps are left alone.

        :param sdfg: The GPU-scheduled SDFG to fix up in place.
        :returns: The number of maps demoted (before elimination).
        """
        from dace.transformation.dataflow import TrivialMapElimination

        demoted = []
        for node, parent in sdfg.all_nodes_recursive():
            if not (isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_Device):
                continue
            try:
                trivial = int(node.map.range.num_elements()) == 1
            except (TypeError, ValueError):
                trivial = False  # Symbolic volume: not provably trivial.
            if trivial:
                node.map.schedule = dtypes.ScheduleType.Sequential
                demoted.append((parent, node))
        for graph, entry in demoted:
            try:
                TrivialMapElimination.apply_to(graph.sdfg, map_entry=entry, verify=True, save=False)
            except ValueError:
                pass  # Declined: the wrapper stays Sequential (tiny host step).
        return len(demoted)

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

        sdfg.backend = dtypes.BackendLanguage.Python

        # Step 0: Canonicalize — rewrite into canonical loop/map form before
        # vectorization (knob rationale: see canonicalize_for_cutile).
        debug_save_sdfg()
        if self.run_canonicalize:
            self.canonicalize_for_cutile(sdfg)
            debug_save_sdfg()

        # Device-resident calling convention: mark non-transient arrays
        # GPU_Global so GPUTransformSDFG creates no clones/copy states.
        if self.use_gpu_storage:
            # Detect legacy/pre-offloaded staging while the original argument
            # storage still distinguishes it from ordinary device dataflow.
            preexisting_staging = []
            for name, desc in sdfg.arrays.items():
                if (not desc.transient and isinstance(desc, data.Array)
                        and desc.storage != dtypes.StorageType.GPU_Global
                        and self._has_device_staging_copy(sdfg, name)):
                    preexisting_staging.append(name)

            # Deferred import: a module-level import would be circular
            # (auto_optimize -> dace.transformation.passes.__init__ ->
            # canonicalize -> this vectorization subpackage).
            from dace.transformation.auto.auto_optimize import apply_gpu_storage
            apply_gpu_storage(sdfg)
            # Loud no-op detection (read-only scalars legitimately stay host).
            not_promoted = []
            for name, desc in sdfg.arrays.items():
                if desc.transient or not isinstance(desc, data.Array):
                    continue
                if desc.storage != dtypes.StorageType.GPU_Global:
                    not_promoted.append(name)
            ineffective = list(dict.fromkeys(not_promoted + preexisting_staging))
            if ineffective:
                details = []
                if not_promoted:
                    details.append(f"not promoted: {not_promoted}")
                if preexisting_staging:
                    details.append(f"pre-existing staging copies: {preexisting_staging}")
                warnings.warn(
                    f"use_gpu_storage could not establish direct device arguments for arrays {ineffective}; "
                    f"{'; '.join(details)}.", UserWarning)
            debug_save_sdfg()

        # GPU-first order: GPU-schedule BEFORE vectorizing so
        # MarkTileDims(require_gpu_resident=True) sees GPU-resident maps.
        # Skipped when the caller already GPU-scheduled.
        if not has_gpu_device_map(sdfg):
            self._apply_gpu_transformations(sdfg)
            # GPUTransformSDFG wraps free tasklets in trivial 0:1 kernel
            # maps that poison the vectorizer; strip them (see helper).
            self._remove_trivial_gpu_maps(sdfg)
            debug_save_sdfg()

        # Vectorize — emit tileops library nodes inside the GPU kernels.
        self._vectorizer.apply_pass(sdfg, {})
        debug_save_sdfg()

        # Anchor census: zero anchors is either the supported BLAS-only
        # configuration (all reductions became BLAS library nodes, lowered
        # in the tail) or a genuinely un-vectorized SDFG; the lowering
        # passes diagnose which. No anchors can appear after this point.
        has_anchors = bool(_collect_tile_nodes(sdfg))

        # Validate — anchors exist, widths are powers of 2.
        CuTileValidateTiles(strict=self.strict).apply_pass(sdfg, {})
        debug_save_sdfg()

        # Masked tile ops address a full W-wide window whose tail lanes
        # are inactive; at non-divisible boundaries the memlet SUBSET
        # exceeds the array bounds even though the masked runtime accesses
        # do not. Mark those memlets allow_oob before any validation of
        # the tiled SDFG.
        mark_tile_op_memlets_allow_oob(sdfg)

        # Defensive: propagation/simplify inside the vectorizer can union
        # tile windows into provably-OOB outer subsets and drop allow_oob
        # (see the caveat in mark_tile_op_memlets_allow_oob).
        clamp_propagated_oob_memlets(sdfg)

        # Simplify + validate the tiled SDFG (after allow_oob marking).
        sdfg.simplify()
        debug_save_sdfg()

        # Adapter — re-stamp tileops-anchored maps GPU_Device -> CuTile.
        num_kernels = GPUDeviceToCuTile(strict=self.strict).apply_pass(sdfg, {})
        debug_save_sdfg()

        # Tile storage — Register transients in CuTile scopes -> CuTile_Tile.
        CuTileSetTileStorage(strict=self.strict).apply_pass(sdfg, {})
        debug_save_sdfg()

        # Select + expand non-tileops library nodes (e.g. BLAS MatMul -> CuPy)
        # so no GPU_Device-scheduled library node survives to codegen.
        CuTileSetLibraryImplementations(strict=self.strict).apply_pass(sdfg, {})
        debug_save_sdfg()

        # Stamp cuTile implementations on tileops library nodes. Skipped when
        # the vectorizer produced no anchors: the pass would be a no-op, and
        # its "already expanded?" diagnostic is misleading here (in the
        # BLAS-only configuration the previous step legitimately expanded
        # everything).
        if has_anchors:
            CuTileSetImplementations(strict=self.strict).apply_pass(sdfg, {})

        debug_save_sdfg()

        return num_kernels
