# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Structure-only unit tests (no GPU) for the six cuTile lowering passes.

Each pass in :mod:`dace.transformation.passes.vectorization.cutile_lowering`
is tested in isolation on SDFGs produced by
``VectorizeCPUMultiDim(target_isa="CUTILE", expand_tile_nodes=False)``, plus
hand-built SDFGs for the anchor-construction corner cases. Ordering /
out-of-order precondition behavior and the full documented sequence through
code generation (text assertions only — no compilation, no kernel launches)
are covered at the end.
"""
import warnings
from typing import Dict, List, Set, Tuple

import pytest

import dace
from dace import data, dtypes
from dace.libraries.tileops import TileBinop
from dace.libraries.tileops.nodes import TileIota
from dace.sdfg import SDFG, nodes
from dace.transformation.passes.vectorization.cutile_lowering import (
    CuTileInsertDataCopies,
    CuTileSetGlobalStorage,
    CuTileSetImplementations,
    CuTileSetSchedules,
    CuTileSetTileStorage,
    CuTileValidateTiles,
    _tile_node_types,
)
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import (
    VectorizeCPUMultiDim,
)

# ============================================================
# Fixture builders
# ============================================================


def _vectorize_cutile(sdfg: SDFG, widths: Tuple[int, ...], **kwargs) -> None:
    """Run the building-block vectorizer config for the cuTile lowering.

    :param sdfg: The SDFG to vectorize in place.
    :param widths: Per-dim tile widths, innermost-last.
    :param kwargs: Extra ``VectorizeCPUMultiDim`` knobs (e.g.
        ``nest_map_bodies=True``).
    """
    VectorizeCPUMultiDim(widths=widths, target_isa="CUTILE", expand_tile_nodes=False, **kwargs).apply_pass(sdfg, {})


def _build_unvectorized_vadd_sdfg() -> SDFG:
    """Symbolic-size K=1 vadd, NOT vectorized (zero tile-op anchors)."""
    N = dace.symbol("N")

    @dace.program
    def cutile_lowering_vadd_plain(A: dace.float64[N], B: dace.float64[N], C: dace.float64[N]):
        for i in dace.map[0:N]:
            C[i] = A[i] + B[i]

    return cutile_lowering_vadd_plain.to_sdfg()


def _build_vadd_k1_sdfg(**vectorizer_kwargs) -> SDFG:
    """Symbolic-size K=1 vadd, vectorized with ``widths=(8,)``."""
    sdfg = _build_unvectorized_vadd_sdfg()
    _vectorize_cutile(sdfg, (8, ), **vectorizer_kwargs)
    return sdfg


def _build_vadd_k2_sdfg() -> SDFG:
    """Symbolic-size K=2 vadd, vectorized with ``widths=(8, 4)``."""
    M = dace.symbol("M")
    N = dace.symbol("N")

    @dace.program
    def cutile_lowering_vadd_k2(A: dace.float64[M, N], B: dace.float64[M, N], C: dace.float64[M, N]):
        for i, j in dace.map[0:M, 0:N]:
            C[i, j] = A[i, j] + B[i, j]

    sdfg = cutile_lowering_vadd_k2.to_sdfg()
    _vectorize_cutile(sdfg, (8, 4))
    return sdfg


def _build_vadd_concrete_sdfg() -> SDFG:
    """Concrete-size vadd with a non-divisible boundary (100 % 8 != 0)."""

    @dace.program
    def cutile_lowering_vadd_n100(A: dace.float64[100], B: dace.float64[100], C: dace.float64[100]):
        for i in dace.map[0:100]:
            C[i] = A[i] + B[i]

    sdfg = cutile_lowering_vadd_n100.to_sdfg()
    _vectorize_cutile(sdfg, (8, ))
    return sdfg


def _build_strides_sdfg() -> SDFG:
    """The ``TileIR/vectorized_pipeline/strides.py`` kernel, inner dim only.

    ``C[i, j] = A[i*2, j] + B[i*2, j]`` vectorized with ``widths=(8,)`` —
    the strided ``i*2`` access is a pre-existing stride that the old
    step>1 schedule heuristic would have keyed on.
    """

    @dace.program
    def cutile_lowering_strides(A: dace.float32[128, 128], B: dace.float32[128, 128], C: dace.float32[64, 128]):
        for i in range(64):
            for j in range(128):
                C[i, j] = A[i * 2, j] + B[i * 2, j]

    sdfg = cutile_lowering_strides.to_sdfg()
    _vectorize_cutile(sdfg, (8, ))
    return sdfg


def _build_partially_vectorized_sdfg() -> Tuple[SDFG, nodes.MapEntry]:
    """K=1 vadd kernel plus an independent, un-vectorized strided map on D.

    Models a partially-vectorized SDFG: the vadd loop vectorized into
    tile-op anchors, while a second independent loop (touching only the
    non-transient ``D``) carries no anchors. The extra map deliberately has
    a non-unit step (``0:N:2``) — the old step>1 heuristic would have
    mis-stamped it ``CuTile``.

    :returns: ``(sdfg, host_map_entry)`` where ``host_map_entry`` is the
        un-anchored map's entry node.
    """
    sdfg = _build_vadd_k1_sdfg()
    sdfg.add_array("D", (dace.symbol("N"), ), dace.float64, transient=False)
    host_state = sdfg.add_state_after(list(sdfg.states())[-1], "host_loop")
    host_state.add_mapped_tasklet(
        "host",
        {"j": "0:N:2"},
        {"_d": dace.Memlet("D[j]")},
        "_o = _d + 1.0",
        {"_o": dace.Memlet("D[j]")},
        external_edges=True,
    )
    host_entry = next(n for n in host_state.nodes() if isinstance(n, nodes.MapEntry))
    return sdfg, host_entry


def _build_map_chain_sdfg() -> Tuple[SDFG, nodes.MapEntry, nodes.MapEntry]:
    """Hand-built two-level map nest around a ``TileBinop`` anchor.

    The flat vectorizer output has a single tiled map, so the
    inner-chain-``Sequential`` stamping is exercised on this explicit nest.

    :returns: ``(sdfg, outer_entry, inner_entry)``.
    """
    sdfg = dace.SDFG("cutile_lowering_map_chain")
    sdfg.add_array("A", (4, 8), dace.float64)
    sdfg.add_array("B", (4, 8), dace.float64)
    sdfg.add_array("C", (4, 8), dace.float64)
    state = sdfg.add_state("main")
    outer_entry, outer_exit = state.add_map("outer", {"i": "0:4"})
    inner_entry, inner_exit = state.add_map("inner", {"t": "0:1"})
    a = state.add_access("A")
    b = state.add_access("B")
    c = state.add_access("C")
    node = TileBinop(name="tb", widths=(8, ), op="+")
    state.add_memlet_path(a, outer_entry, inner_entry, node, dst_conn="_a", memlet=dace.Memlet("A[i, 0:8]"))
    state.add_memlet_path(b, outer_entry, inner_entry, node, dst_conn="_b", memlet=dace.Memlet("B[i, 0:8]"))
    state.add_memlet_path(node, inner_exit, outer_exit, c, src_conn="_c", memlet=dace.Memlet("C[i, 0:8]"))
    sdfg.validate()
    return sdfg, outer_entry, inner_entry


def _build_bare_tile_binop_sdfg(widths: Tuple[int, ...]) -> SDFG:
    """Hand-built single-state SDFG with one ``TileBinop`` and no maps.

    :param widths: Tile widths of the node (possibly non-power-of-2, for
        the validation tests).
    """
    sdfg = dace.SDFG("cutile_lowering_bare_binop")
    shape = tuple(max(w, 1) for w in widths)
    sdfg.add_array("A", shape, dace.float64)
    sdfg.add_array("B", shape, dace.float64)
    sdfg.add_array("C", shape, dace.float64)
    state = sdfg.add_state("main")
    a = state.add_access("A")
    b = state.add_access("B")
    c = state.add_access("C")
    node = TileBinop(name="tb", widths=widths, op="+")
    state.add_node(node)
    full = ",".join(f"0:{w}" for w in shape)
    state.add_edge(a, None, node, "_a", dace.Memlet(f"A[{full}]"))
    state.add_edge(b, None, node, "_b", dace.Memlet(f"B[{full}]"))
    state.add_edge(node, "_c", c, None, dace.Memlet(f"C[{full}]"))
    return sdfg


def _build_tile_iota_sdfg() -> SDFG:
    """Hand-built SDFG holding a ``TileIota`` with a 'cutile' expansion."""
    sdfg = dace.SDFG("cutile_lowering_iota")
    sdfg.add_array("I", (8, ), dace.int64, transient=False)
    state = sdfg.add_state("main")
    iota = TileIota(name="ti", widths=(8, ), expr="__l0")
    state.add_node(iota)
    state.add_edge(iota, "_dst", state.add_access("I"), None, dace.Memlet("I[0:8]"))
    return sdfg


# ============================================================
# Inspection helpers
# ============================================================


def _all_map_entries(sdfg: SDFG) -> List[Tuple[nodes.MapEntry, dace.SDFGState]]:
    """All MapEntry nodes in ``sdfg``, recursively."""
    return [(n, g) for n, g in sdfg.all_nodes_recursive() if isinstance(n, nodes.MapEntry)]


def _cutile_map_entries(sdfg: SDFG) -> List[nodes.MapEntry]:
    """All MapEntry nodes with ``ScheduleType.CuTile``, recursively."""
    return [n for n, _ in _all_map_entries(sdfg) if n.map.schedule == dtypes.ScheduleType.CuTile]


def _tileops_nodes(sdfg: SDFG) -> List[Tuple[nodes.LibraryNode, dace.SDFGState]]:
    """All tileops library nodes in ``sdfg``, recursively.

    The tile-node class list is sourced from
    :func:`cutile_lowering._tile_node_types` so it cannot drift when a new
    tile-node type is added; the scope-walk in
    :func:`_outermost_anchored_entries` remains an independent
    reimplementation by design.
    """
    tile_types = _tile_node_types()
    return [(n, g) for n, g in sdfg.all_nodes_recursive() if isinstance(n, tile_types)]


def _outermost_anchored_entries(sdfg: SDFG) -> Set[nodes.MapEntry]:
    """Outermost map entries enclosing tileops anchors (top-level walk).

    Independent re-implementation of the anchored scope walk (in-state
    only — sufficient for the flat-path fixtures used here) so the tests
    do not lean on the module under test's own helpers.
    """
    outermost: Set[nodes.MapEntry] = set()
    for node, state in _tileops_nodes(sdfg):
        scope_dict = state.scope_dict()
        scope = scope_dict.get(node)
        last = None
        while scope is not None:
            last = scope
            scope = scope_dict.get(scope)
        if last is not None:
            outermost.add(last)
    return outermost


def _storage_snapshot(sdfg: SDFG) -> Dict[Tuple[str, str], dtypes.StorageType]:
    """Map ``(sdfg_label, array_name) -> storage`` over all nested SDFGs."""
    snapshot: Dict[Tuple[str, str], dtypes.StorageType] = {}
    for nested in sdfg.all_sdfgs_recursive():
        for name, desc in nested.arrays.items():
            snapshot[(nested.label, name)] = desc.storage
    return snapshot


def _single_nested_sdfg(sdfg: SDFG) -> SDFG:
    """Return the single NestedSDFG body of a ``nest_map_bodies`` fixture."""
    nsdfgs = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG)]
    assert len(nsdfgs) == 1, f"expected exactly one NestedSDFG, found {len(nsdfgs)}"
    return nsdfgs[0].sdfg


def _apply_schedules_quietly(sdfg: SDFG) -> None:
    """Run ``CuTileSetSchedules`` swallowing partial-vectorization warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        CuTileSetSchedules().apply_pass(sdfg, {})


# ============================================================
# 1. CuTileValidateTiles
# ============================================================


class TestValidateTiles:

    def test_counts_anchors_on_vadd_k1(self):
        """K=1 vadd carries 5 anchors (maskgen, 2 loads, binop, store)."""
        sdfg = _build_vadd_k1_sdfg()
        assert CuTileValidateTiles().apply_pass(sdfg, {}) == 5

    def test_counts_anchors_on_vadd_k2(self):
        """K=2 vadd carries the same 5-anchor chain."""
        sdfg = _build_vadd_k2_sdfg()
        assert CuTileValidateTiles().apply_pass(sdfg, {}) == 5

    def test_zero_anchors_warns_and_returns_none(self):
        """An un-vectorized SDFG has no anchors: warn + return None."""
        sdfg = _build_unvectorized_vadd_sdfg()
        with pytest.warns(UserWarning, match="CuTileValidateTiles: no tileops library nodes found"):
            assert CuTileValidateTiles().apply_pass(sdfg, {}) is None

    def test_zero_anchors_strict_raises(self):
        """``strict=True`` turns the zero-anchor warning into ValueError."""
        sdfg = _build_unvectorized_vadd_sdfg()
        with pytest.raises(ValueError, match="no tileops library nodes found"):
            CuTileValidateTiles(strict=True).apply_pass(sdfg, {})

    def test_non_power_of_two_widths_raise_even_without_strict(self):
        """Non-power-of-2 widths are a hard error regardless of ``strict``."""
        sdfg = _build_bare_tile_binop_sdfg((6, ))
        with pytest.raises(ValueError, match="not a power of 2"):
            CuTileValidateTiles(strict=False).apply_pass(sdfg, {})

    def test_power_of_two_widths_pass_hand_built(self):
        """A hand-built pow2 anchor validates fine (count 1)."""
        sdfg = _build_bare_tile_binop_sdfg((8, ))
        assert CuTileValidateTiles().apply_pass(sdfg, {}) == 1


# ============================================================
# 2. CuTileSetSchedules
# ============================================================


class TestSetSchedules:

    def test_k1_vadd_single_cutile_map(self):
        """K=1: exactly the (single) anchored tiled map becomes CuTile."""
        sdfg = _build_vadd_k1_sdfg()
        assert CuTileSetSchedules().apply_pass(sdfg, {}) == 1
        cutile_maps = _cutile_map_entries(sdfg)
        assert len(cutile_maps) == 1
        # It is the tileops-anchored outermost map, and it is tiled (step 8).
        assert set(cutile_maps) == _outermost_anchored_entries(sdfg)
        assert any(str(step) == "8" for _, _, step in cutile_maps[0].map.range)

    def test_k2_vadd_single_cutile_map(self):
        """K=2 (widths (8, 4)): the single 2-D tiled map becomes CuTile."""
        sdfg = _build_vadd_k2_sdfg()
        assert CuTileSetSchedules().apply_pass(sdfg, {}) == 1
        cutile_maps = _cutile_map_entries(sdfg)
        assert len(cutile_maps) == 1
        assert set(cutile_maps) == _outermost_anchored_entries(sdfg)
        steps = [str(step) for _, _, step in cutile_maps[0].map.range]
        assert steps == ["8", "4"]

    def test_nested_chain_outer_cutile_inner_sequential(self):
        """Hand-built nest: outermost map CuTile, inner chain Sequential."""
        sdfg, outer_entry, inner_entry = _build_map_chain_sdfg()
        assert CuTileSetSchedules().apply_pass(sdfg, {}) == 1
        assert outer_entry.map.schedule == dtypes.ScheduleType.CuTile
        assert inner_entry.map.schedule == dtypes.ScheduleType.Sequential

    def test_idempotent_second_run(self):
        """A second run restamps nothing new and emits no warnings."""
        sdfg = _build_vadd_k1_sdfg()
        CuTileSetSchedules().apply_pass(sdfg, {})
        schedules_before = {n: n.map.schedule for n, _ in _all_map_entries(sdfg)}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert CuTileSetSchedules().apply_pass(sdfg, {}) == 1
        assert caught == []
        assert {n: n.map.schedule for n, _ in _all_map_entries(sdfg)} == schedules_before

    def test_zero_anchors_warns_and_returns_none(self):
        """Run before the vectorizer: no anchors -> warn + None, no stamps."""
        sdfg = _build_unvectorized_vadd_sdfg()
        with pytest.warns(UserWarning, match="CuTileSetSchedules: no tileops library nodes found"):
            assert CuTileSetSchedules().apply_pass(sdfg, {}) is None
        assert _cutile_map_entries(sdfg) == []

    def test_zero_anchors_strict_raises(self):
        sdfg = _build_unvectorized_vadd_sdfg()
        with pytest.raises(ValueError, match="no tileops library nodes found"):
            CuTileSetSchedules(strict=True).apply_pass(sdfg, {})

    def test_anchor_without_enclosing_map_warns(self):
        """A bare anchor (no map at any level) cannot become a kernel."""
        sdfg = _build_bare_tile_binop_sdfg((8, ))
        with pytest.warns(UserWarning, match="has no enclosing map at any level"):
            assert CuTileSetSchedules().apply_pass(sdfg, {}) is None

    def test_anchor_without_enclosing_map_strict_raises(self):
        sdfg = _build_bare_tile_binop_sdfg((8, ))
        with pytest.raises(ValueError, match="has no enclosing map at any level"):
            CuTileSetSchedules(strict=True).apply_pass(sdfg, {})

    def test_strides_kernel_only_anchored_map_is_cutile(self):
        """Step>1 false-positive regression on the strides.py kernel.

        ``C[i, j] = A[i*2, j] + B[i*2, j]`` with only the inner dim
        vectorized: the only CuTile map must be the tileops-anchored one
        (stamping is anchor-driven, never step-driven).
        """
        sdfg = _build_strides_sdfg()
        assert CuTileSetSchedules().apply_pass(sdfg, {}) == 1
        cutile_maps = _cutile_map_entries(sdfg)
        assert len(cutile_maps) == 1
        assert set(cutile_maps) == _outermost_anchored_entries(sdfg)

    def test_unanchored_strided_map_not_stamped(self):
        """Step>1 false-positive regression, explicit form.

        A map with a non-unit step (``0:N:2``) that encloses no tileops
        anchor must NOT be stamped CuTile merely for its stride (the old
        heuristic keyed on ``step != 1``); instead the partial-vectorization
        audit warns about it.
        """
        sdfg, host_entry = _build_partially_vectorized_sdfg()
        with pytest.warns(UserWarning, match="encloses no tileops node"):
            assert CuTileSetSchedules().apply_pass(sdfg, {}) == 1
        assert host_entry.map.schedule == dtypes.ScheduleType.Default
        cutile_maps = _cutile_map_entries(sdfg)
        assert len(cutile_maps) == 1
        assert host_entry not in cutile_maps

    def test_partially_vectorized_warns_and_leaves_map_untouched(self):
        """Partially-vectorized SDFG: the un-anchored map keeps its schedule
        and the trailing audit emits the partially-vectorized warning."""
        sdfg, host_entry = _build_partially_vectorized_sdfg()
        with pytest.warns(UserWarning, match="partially-vectorized SDFG"):
            CuTileSetSchedules().apply_pass(sdfg, {})
        assert host_entry.map.schedule == dtypes.ScheduleType.Default

    def test_partially_vectorized_strict_raises(self):
        sdfg, _ = _build_partially_vectorized_sdfg()
        with pytest.raises(ValueError, match="encloses no tileops node"):
            CuTileSetSchedules(strict=True).apply_pass(sdfg, {})


# ============================================================
# 3. CuTileSetTileStorage
# ============================================================


class TestSetTileStorage:

    def test_k1_tile_and_mask_transients_stamped(self):
        """K=1: the 4 Register tile/mask transients become CuTile_Tile."""
        sdfg = _build_vadd_k1_sdfg()
        CuTileSetSchedules().apply_pass(sdfg, {})
        assert CuTileSetTileStorage().apply_pass(sdfg, {}) == 4
        stamped = {name for name, desc in sdfg.arrays.items() if desc.storage == dtypes.StorageType.CuTile_Tile}
        assert len(stamped) == 4
        assert "_tile_iter_mask" in stamped
        for name in stamped:
            assert sdfg.arrays[name].transient, f"{name} is not transient"
            assert isinstance(sdfg.arrays[name], data.Array)

    def test_scalar_descriptors_untouched(self):
        """No Scalar descriptor changes storage (and none becomes a tile)."""
        sdfg = _build_vadd_k1_sdfg()
        CuTileSetSchedules().apply_pass(sdfg, {})
        scalar_storage_before = {
            (owner, name): storage
            for (owner, name), storage in _storage_snapshot(sdfg).items()
        }
        CuTileSetTileStorage().apply_pass(sdfg, {})
        for nested in sdfg.all_sdfgs_recursive():
            for name, desc in nested.arrays.items():
                if isinstance(desc, data.Scalar):
                    assert desc.storage == scalar_storage_before[(nested.label, name)], \
                        f"Scalar {name} changed storage"
                    assert desc.storage != dtypes.StorageType.CuTile_Tile

    def test_nsdfg_descent_inner_descriptors_stamped(self):
        """``nest_map_bodies=True``: inner connector-bound tile descriptors
        (both the transient tiles and the non-transient mask view) are
        stamped CuTile_Tile through the NSDFG boundary."""
        sdfg = _build_vadd_k1_sdfg(nest_map_bodies=True)
        CuTileSetSchedules().apply_pass(sdfg, {})
        assert CuTileSetTileStorage().apply_pass(sdfg, {}) == 5
        inner = _single_nested_sdfg(sdfg)
        # Every inner transient Array is a tile of the kernel body.
        inner_transient_arrays = [
            name for name, desc in inner.arrays.items() if isinstance(desc, data.Array) and desc.transient
        ]
        assert inner_transient_arrays, "descent fixture has no inner tile transients"
        for name in inner_transient_arrays:
            assert inner.arrays[name].storage == dtypes.StorageType.CuTile_Tile, \
                f"inner transient {name} not stamped"
        # The outer mask tile propagates to its inner (non-transient)
        # connector descriptor too.
        assert sdfg.arrays["_tile_iter_mask"].storage == dtypes.StorageType.CuTile_Tile
        assert inner.arrays["_tile_iter_mask"].storage == dtypes.StorageType.CuTile_Tile

    def test_without_schedules_warns_and_changes_nothing(self):
        """Precondition: no CuTile map -> warn + None + zero mutations."""
        sdfg = _build_vadd_k1_sdfg()
        before = _storage_snapshot(sdfg)
        with pytest.warns(UserWarning, match="CuTileSetTileStorage: no CuTile-scheduled map found"):
            assert CuTileSetTileStorage().apply_pass(sdfg, {}) is None
        assert _storage_snapshot(sdfg) == before

    def test_without_schedules_strict_raises(self):
        sdfg = _build_vadd_k1_sdfg()
        with pytest.raises(ValueError, match="run CuTileSetSchedules first"):
            CuTileSetTileStorage(strict=True).apply_pass(sdfg, {})

    def test_idempotent_second_run(self):
        """A second run stamps nothing (returns None), storages unchanged."""
        sdfg = _build_vadd_k1_sdfg()
        CuTileSetSchedules().apply_pass(sdfg, {})
        CuTileSetTileStorage().apply_pass(sdfg, {})
        before = _storage_snapshot(sdfg)
        assert CuTileSetTileStorage().apply_pass(sdfg, {}) is None
        assert _storage_snapshot(sdfg) == before


# ============================================================
# 4. CuTileSetGlobalStorage
# ============================================================


class TestSetGlobalStorage:

    def test_kernel_touched_globals_stamped(self):
        """K=1: the three kernel operands A, B, C become GPU_Global."""
        sdfg = _build_vadd_k1_sdfg()
        CuTileSetSchedules().apply_pass(sdfg, {})
        assert CuTileSetGlobalStorage().apply_pass(sdfg, {}) == 3
        for name in ("A", "B", "C"):
            assert sdfg.arrays[name].storage == dtypes.StorageType.GPU_Global, f"{name} not GPU_Global"

    def test_non_kernel_global_not_stamped(self):
        """Per-kernel-use contract: a non-transient touched only outside the
        kernel (D, in an independent host map) keeps its storage."""
        sdfg, _ = _build_partially_vectorized_sdfg()
        d_storage_before = sdfg.arrays["D"].storage
        _apply_schedules_quietly(sdfg)
        assert CuTileSetGlobalStorage().apply_pass(sdfg, {}) == 3
        assert sdfg.arrays["D"].storage == d_storage_before
        assert sdfg.arrays["D"].storage != dtypes.StorageType.GPU_Global
        for name in ("A", "B", "C"):
            assert sdfg.arrays[name].storage == dtypes.StorageType.GPU_Global

    def test_nsdfg_propagation_stamps_inner_and_outer(self):
        """``nest_map_bodies=True``: globals reached only through the NSDFG
        are stamped on both the inner and the outer descriptor."""
        sdfg = _build_vadd_k1_sdfg(nest_map_bodies=True)
        CuTileSetSchedules().apply_pass(sdfg, {})
        CuTileSetTileStorage().apply_pass(sdfg, {})
        assert CuTileSetGlobalStorage().apply_pass(sdfg, {}) is not None
        inner = _single_nested_sdfg(sdfg)
        for name in ("A", "B", "C"):
            assert sdfg.arrays[name].storage == dtypes.StorageType.GPU_Global, f"outer {name} not GPU_Global"
            assert inner.arrays[name].storage == dtypes.StorageType.GPU_Global, f"inner {name} not GPU_Global"

    def test_inner_tile_connector_storage_preserved(self):
        """A tile passed through an NSDFG connector must keep CuTile_Tile
        storage; GPU_Global stamping is for global arrays only."""
        sdfg = _build_vadd_k1_sdfg(nest_map_bodies=True)
        CuTileSetSchedules().apply_pass(sdfg, {})
        CuTileSetTileStorage().apply_pass(sdfg, {})
        inner = _single_nested_sdfg(sdfg)
        inner_tile_connectors = [
            name for name, desc in inner.arrays.items()
            if not desc.transient and desc.storage == dtypes.StorageType.CuTile_Tile
        ]
        assert inner_tile_connectors, "fixture lost its inner tile connector arrays"
        CuTileSetGlobalStorage().apply_pass(sdfg, {})
        for name in inner_tile_connectors:
            assert inner.arrays[name].storage == dtypes.StorageType.CuTile_Tile, \
                f"inner tile connector {name} clobbered to {inner.arrays[name].storage}"

    def test_without_schedules_warns_and_changes_nothing(self):
        sdfg = _build_vadd_k1_sdfg()
        before = _storage_snapshot(sdfg)
        with pytest.warns(UserWarning, match="CuTileSetGlobalStorage: no CuTile-scheduled map found"):
            assert CuTileSetGlobalStorage().apply_pass(sdfg, {}) is None
        assert _storage_snapshot(sdfg) == before

    def test_without_schedules_strict_raises(self):
        sdfg = _build_vadd_k1_sdfg()
        with pytest.raises(ValueError, match="run CuTileSetSchedules first"):
            CuTileSetGlobalStorage(strict=True).apply_pass(sdfg, {})

    def test_idempotent_second_run(self):
        sdfg = _build_vadd_k1_sdfg()
        CuTileSetSchedules().apply_pass(sdfg, {})
        CuTileSetGlobalStorage().apply_pass(sdfg, {})
        before = _storage_snapshot(sdfg)
        assert CuTileSetGlobalStorage().apply_pass(sdfg, {}) is None
        assert _storage_snapshot(sdfg) == before


# ============================================================
# 5. CuTileSetImplementations
# ============================================================


class TestSetImplementations:

    def test_all_nodes_stamped_pre_expansion(self):
        """Every tileops node carries CUTILE/cutile before expansion."""
        sdfg = _build_vadd_k1_sdfg()
        assert CuTileSetImplementations().apply_pass(sdfg, {}) == 5
        lib_nodes = _tileops_nodes(sdfg)
        assert len(lib_nodes) == 5
        for node, _ in lib_nodes:
            assert node.target_isa == "CUTILE", f"{node.label}: target_isa {node.target_isa}"
            assert node.implementation == "cutile", f"{node.label}: implementation {node.implementation}"

    def test_k2_nodes_stamped(self):
        """K=2 nodes are stamped 'cutile' directly (no K>=2 'pure' dispatch)."""
        sdfg = _build_vadd_k2_sdfg()
        assert CuTileSetImplementations().apply_pass(sdfg, {}) == 5
        for node, _ in _tileops_nodes(sdfg):
            assert node.implementation == "cutile"

    def test_after_expansion_warns(self):
        """Run after ``expand_library_nodes()``: no lib nodes -> warn + None."""
        sdfg = _build_vadd_k1_sdfg()
        CuTileSetImplementations().apply_pass(sdfg, {})
        sdfg.expand_library_nodes()
        assert _tileops_nodes(sdfg) == []
        with pytest.warns(UserWarning, match="already expanded"):
            assert CuTileSetImplementations().apply_pass(sdfg, {}) is None

    def test_after_expansion_strict_raises(self):
        sdfg = _build_vadd_k1_sdfg()
        CuTileSetImplementations().apply_pass(sdfg, {})
        sdfg.expand_library_nodes()
        with pytest.raises(ValueError, match="already expanded"):
            CuTileSetImplementations(strict=True).apply_pass(sdfg, {})

    def test_tile_iota_stamped_cutile(self):
        """``TileIota`` ships 'cutile': stamping succeeds."""
        sdfg = _build_tile_iota_sdfg()
        assert CuTileSetImplementations().apply_pass(sdfg, {}) == 1
        for node, _ in _tileops_nodes(sdfg):
            assert node.target_isa == "CUTILE"
            assert node.implementation == "cutile"

    def test_tile_iota_expands_to_python_tasklet(self):
        """``TileIota`` stamped 'cutile' expands to a Python tasklet."""
        sdfg = _build_tile_iota_sdfg()
        CuTileSetImplementations().apply_pass(sdfg, {})
        sdfg.expand_library_nodes()
        assert not any(isinstance(n, nodes.LibraryNode) for n, _ in sdfg.all_nodes_recursive())
        tasklets = [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.Tasklet)]
        assert len(tasklets) == 1
        body = tasklets[0].code.as_string
        assert "ct.arange" in body
        assert "_dst" in body

    def test_node_without_cutile_implementation_raises(self):
        """A tileops node that lacks ``'cutile'`` in its implementations
        dict must fail loudly, not fall back silently."""
        from unittest.mock import patch
        sdfg = _build_tile_iota_sdfg()
        # Temporarily remove 'cutile' from TileIota.implementations so
        # the error path triggers.  patch.dict restores the original on
        # exit, so other tests are not affected.
        with patch.dict(TileIota.implementations, {"pure": TileIota.implementations["pure"]}, clear=True):
            with pytest.raises(ValueError, match="has no 'cutile' implementation"):
                CuTileSetImplementations().apply_pass(sdfg, {})

    def test_idempotent_second_run(self):
        sdfg = _build_vadd_k1_sdfg()
        CuTileSetImplementations().apply_pass(sdfg, {})
        assert CuTileSetImplementations().apply_pass(sdfg, {}) == 5
        for node, _ in _tileops_nodes(sdfg):
            assert node.target_isa == "CUTILE"
            assert node.implementation == "cutile"


# ============================================================
# 6. CuTileInsertDataCopies
# ============================================================


def _apply_up_to_global_storage(sdfg: SDFG) -> None:
    """Run passes up to and including CuTileSetGlobalStorage."""
    CuTileValidateTiles().apply_pass(sdfg, {})
    CuTileSetSchedules().apply_pass(sdfg, {})
    CuTileSetTileStorage().apply_pass(sdfg, {})
    CuTileSetGlobalStorage().apply_pass(sdfg, {})


class TestInsertDataCopies:

    def test_k1_vadd_clones_three_arrays(self):
        """K=1 vadd: A, B, C are cloned and copy states are inserted."""
        sdfg = _build_vadd_k1_sdfg()
        _apply_up_to_global_storage(sdfg)
        assert CuTileInsertDataCopies().apply_pass(sdfg, {}) == 3

    def test_originals_reverted_to_cpu_heap(self):
        """After the pass, the original A, B, C descriptors are CPU_Heap."""
        sdfg = _build_vadd_k1_sdfg()
        _apply_up_to_global_storage(sdfg)
        CuTileInsertDataCopies().apply_pass(sdfg, {})
        for name in ("A", "B", "C"):
            assert sdfg.arrays[name].storage == dtypes.StorageType.CPU_Heap, \
                f"{name} not CPU_Heap"

    def test_gpu_clones_are_gpu_global_transients(self):
        """gpu_* clones exist and are GPU_Global transients."""
        sdfg = _build_vadd_k1_sdfg()
        _apply_up_to_global_storage(sdfg)
        CuTileInsertDataCopies().apply_pass(sdfg, {})
        for orig_name in ("A", "B", "C"):
            gpu_name = f"gpu_{orig_name}"
            assert gpu_name in sdfg.arrays, f"{gpu_name} not found"
            desc = sdfg.arrays[gpu_name]
            assert desc.storage == dtypes.StorageType.GPU_Global, \
                f"{gpu_name} storage is {desc.storage}"
            assert desc.transient, f"{gpu_name} is not transient"

    def test_copyin_state_exists(self):
        """A copy-in state is inserted as the new start block."""
        sdfg = _build_vadd_k1_sdfg()
        _apply_up_to_global_storage(sdfg)
        CuTileInsertDataCopies().apply_pass(sdfg, {})
        copyin = sdfg.start_block
        assert copyin.label.endswith("_copyin"), \
            f"start block label is {copyin.label}, expected *_copyin"
        # It should have AccessNodes for both original and GPU names.
        access_names = {n.data for n in copyin.nodes()
                        if isinstance(n, nodes.AccessNode)}
        for orig in ("A", "B", "C"):
            assert orig in access_names, f"{orig} not in copy-in state"
            assert f"gpu_{orig}" in access_names, \
                f"gpu_{orig} not in copy-in state"

    def test_copyout_state_exists_with_written_arrays(self):
        """A copy-out state is inserted after all sink nodes, containing
        only written arrays (C for vadd)."""
        sdfg = _build_vadd_k1_sdfg()
        _apply_up_to_global_storage(sdfg)
        CuTileInsertDataCopies().apply_pass(sdfg, {})
        # The copy-out state is the sole sink of the SDFG.
        sinks = sdfg.sink_nodes()
        assert len(sinks) == 1
        copyout = sinks[0]
        assert copyout.label.endswith("_copyout"), \
            f"sink label is {copyout.label}, expected *_copyout"
        # Copy-out should contain the written arrays.  For vadd, C is written.
        access_names = {n.data for n in copyout.nodes()
                        if isinstance(n, nodes.AccessNode)}
        assert "C" in access_names, "C (written) not in copy-out"
        assert "gpu_C" in access_names, "gpu_C not in copy-out"

    def test_internal_references_use_gpu_clones(self):
        """All computation states reference gpu_* clones, not the originals."""
        sdfg = _build_vadd_k1_sdfg()
        _apply_up_to_global_storage(sdfg)
        CuTileInsertDataCopies().apply_pass(sdfg, {})
        copyin = sdfg.start_block
        sinks = sdfg.sink_nodes()
        for state in sdfg.states():
            if state is copyin or state in sinks:
                continue
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode):
                    assert node.data not in ("A", "B", "C"), \
                        f"computation state still references original {node.data}"

    def test_copyin_has_full_array_memlets(self):
        """Copy-in edges use Memlet.from_array (full-array copies)."""
        sdfg = _build_vadd_k1_sdfg()
        _apply_up_to_global_storage(sdfg)
        CuTileInsertDataCopies().apply_pass(sdfg, {})
        copyin = sdfg.start_block
        edges = list(copyin.edges())
        assert len(edges) == 3, f"expected 3 copy-in edges, got {len(edges)}"
        for edge in edges:
            assert edge.data.data is not None
            # Source should be the CPU original, dst should be GPU clone.
            assert isinstance(edge.src, nodes.AccessNode)
            assert isinstance(edge.dst, nodes.AccessNode)
            assert edge.dst.data.startswith("gpu_"), \
                f"copy-in dst {edge.dst.data} should start with gpu_"

    def test_k2_vadd_clones_three_arrays(self):
        """K=2 vadd: same three arrays are cloned."""
        sdfg = _build_vadd_k2_sdfg()
        _apply_up_to_global_storage(sdfg)
        assert CuTileInsertDataCopies().apply_pass(sdfg, {}) == 3

    def test_concrete_non_divisible_boundary(self):
        """Concrete-size (100) with non-divisible tile boundary works."""
        sdfg = _build_vadd_concrete_sdfg()
        _apply_up_to_global_storage(sdfg)
        assert CuTileInsertDataCopies().apply_pass(sdfg, {}) == 3
        for orig in ("A", "B", "C"):
            assert sdfg.arrays[orig].storage == dtypes.StorageType.CPU_Heap

    def test_idempotent_second_run(self):
        """A second run finds no GPU_Global non-transients and returns None."""
        sdfg = _build_vadd_k1_sdfg()
        _apply_up_to_global_storage(sdfg)
        CuTileInsertDataCopies().apply_pass(sdfg, {})
        snapshot = _storage_snapshot(sdfg)
        num_states_before = len(list(sdfg.states()))
        assert CuTileInsertDataCopies().apply_pass(sdfg, {}) is None
        assert _storage_snapshot(sdfg) == snapshot
        assert len(list(sdfg.states())) == num_states_before

    def test_without_schedules_warns_and_returns_none(self):
        """Precondition: no CuTile map -> warn + None."""
        sdfg = _build_vadd_k1_sdfg()
        with pytest.warns(UserWarning,
                          match="CuTileInsertDataCopies: no CuTile-scheduled map found"):
            assert CuTileInsertDataCopies().apply_pass(sdfg, {}) is None

    def test_without_schedules_strict_raises(self):
        """``strict=True`` turns the precondition warning into ValueError."""
        sdfg = _build_vadd_k1_sdfg()
        with pytest.raises(ValueError, match="run CuTileSetSchedules first"):
            CuTileInsertDataCopies(strict=True).apply_pass(sdfg, {})

    def test_non_kernel_array_not_cloned(self):
        """A non-transient array D that is NOT GPU_Global is left untouched."""
        sdfg, _ = _build_partially_vectorized_sdfg()
        _apply_schedules_quietly(sdfg)
        CuTileSetTileStorage().apply_pass(sdfg, {})
        CuTileSetGlobalStorage().apply_pass(sdfg, {})
        d_storage = sdfg.arrays["D"].storage
        CuTileInsertDataCopies().apply_pass(sdfg, {})
        # D was never GPU_Global, so it should not be cloned.
        assert "gpu_D" not in sdfg.arrays
        assert sdfg.arrays["D"].storage == d_storage

    def test_scalars_not_cloned(self):
        """Scalar descriptors are never cloned (even if somehow GPU_Global)."""
        sdfg = _build_vadd_k1_sdfg()
        _apply_up_to_global_storage(sdfg)
        # Check no scalar is in the candidates (sanity).
        for name, desc in sdfg.arrays.items():
            if isinstance(desc, data.Scalar):
                assert desc.storage != dtypes.StorageType.GPU_Global
        CuTileInsertDataCopies().apply_pass(sdfg, {})
        # After the pass, no gpu_ clone of a scalar should exist.
        for name, desc in sdfg.arrays.items():
            if name.startswith("gpu_"):
                assert isinstance(desc, data.Array), \
                    f"{name} is a cloned Scalar, but scalars should not be cloned"

    def test_state_count_increases_by_two(self):
        """Exactly two new states (copy-in, copy-out) are added."""
        sdfg = _build_vadd_k1_sdfg()
        _apply_up_to_global_storage(sdfg)
        n_before = len(list(sdfg.states()))
        CuTileInsertDataCopies().apply_pass(sdfg, {})
        n_after = len(list(sdfg.states()))
        assert n_after == n_before + 2

    def test_nsdfg_inner_refs_unchanged(self):
        """``nest_map_bodies=True``: inner NSDFG connector arrays keep their
        names (they pick up the gpu_ data via outer memlets)."""
        sdfg = _build_vadd_k1_sdfg(nest_map_bodies=True)
        CuTileSetSchedules().apply_pass(sdfg, {})
        CuTileSetTileStorage().apply_pass(sdfg, {})
        CuTileSetGlobalStorage().apply_pass(sdfg, {})
        inner = _single_nested_sdfg(sdfg)
        inner_names_before = set(inner.arrays.keys())
        CuTileInsertDataCopies().apply_pass(sdfg, {})
        inner_names_after = set(inner.arrays.keys())
        # The inner SDFG arrays should not change (no gpu_ names added inside).
        assert inner_names_before == inner_names_after


# ============================================================
# 7. Ordering / full pipeline
# ============================================================


class TestPipelineOrdering:

    def _run_full_sequence(self, sdfg: SDFG) -> None:
        """Apply the six passes in documented order, then expand + backend."""
        CuTileValidateTiles().apply_pass(sdfg, {})
        CuTileSetSchedules().apply_pass(sdfg, {})
        CuTileSetTileStorage().apply_pass(sdfg, {})
        CuTileSetGlobalStorage().apply_pass(sdfg, {})
        CuTileInsertDataCopies().apply_pass(sdfg, {})
        CuTileSetImplementations().apply_pass(sdfg, {})
        sdfg.expand_library_nodes()
        sdfg.backend = dtypes.BackendLanguage.Python

    def test_full_sequence_k2_generates_cutile_code(self):
        """Documented sequence on K=2 vadd -> Python backend cuTile code."""
        sdfg = _build_vadd_k2_sdfg()
        self._run_full_sequence(sdfg)
        assert not any(isinstance(n, nodes.LibraryNode) for n, _ in sdfg.all_nodes_recursive())
        assert sdfg.backend == dtypes.BackendLanguage.Python
        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        assert "import cuda.tile as ct" in code
        assert "@ct.kernel" in code
        assert "ct.load(" in code

    def test_full_sequence_k1_concrete_non_divisible(self):
        """Concrete non-divisible size (100 % 8 != 0) through the sequence."""
        sdfg = _build_vadd_concrete_sdfg()
        assert CuTileValidateTiles().apply_pass(sdfg, {}) == 5
        assert CuTileSetSchedules().apply_pass(sdfg, {}) == 1
        assert CuTileSetTileStorage().apply_pass(sdfg, {}) == 4
        assert CuTileSetGlobalStorage().apply_pass(sdfg, {}) == 3
        assert CuTileSetImplementations().apply_pass(sdfg, {}) == 5
        sdfg.expand_library_nodes()
        sdfg.backend = dtypes.BackendLanguage.Python
        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        assert "import cuda.tile as ct" in code

    def test_storage_before_schedules_is_warned_noop_then_recovers(self):
        """Out-of-order: a storage pass run first warns and is a no-op; the
        documented order afterwards still works on the same SDFG."""
        sdfg = _build_vadd_k1_sdfg()
        before = _storage_snapshot(sdfg)
        with pytest.warns(UserWarning, match="run CuTileSetSchedules first"):
            assert CuTileSetTileStorage().apply_pass(sdfg, {}) is None
        with pytest.warns(UserWarning, match="run CuTileSetSchedules first"):
            assert CuTileSetGlobalStorage().apply_pass(sdfg, {}) is None
        assert _storage_snapshot(sdfg) == before
        # Recovery: correct order succeeds afterwards.
        assert CuTileSetSchedules().apply_pass(sdfg, {}) == 1
        assert CuTileSetTileStorage().apply_pass(sdfg, {}) == 4
        assert CuTileSetGlobalStorage().apply_pass(sdfg, {}) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
