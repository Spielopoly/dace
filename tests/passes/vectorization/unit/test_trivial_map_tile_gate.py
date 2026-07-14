# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared tile-candidate gate: trivial wrapper maps (all dims 1-trip, step 1) are never tiled.

``GPUTransformSDFG`` wraps free tasklets in trivial one-param ``0:1`` ``*_gmap``
kernel maps. When such a wrapper reached the vectorizer (caller-pre-scheduled
SDFGs, ``VectorizeGPUMultiDim``), ``MarkTileDims`` crashed at K>=2 ("has only 1
params") and at K=1 tile machinery was inserted into a body with nothing to
widen (isolated ``_tile_iter_mask``). The refusal lives in ``is_tile_eligible``
(flowing through ``is_vectorizable_map``) so every tile pass skips such maps
consistently. NOT refused: symbolic trip counts (masking handles short trips)
and single-trip maps with step > 1 -- those are legitimate W-strided tile maps
(``StrideMapByTileWidths`` remainder tails like ``16:20:8``), and downstream
passes re-consult the gate after striding.
"""
from typing import Dict, Tuple

import dace
from dace.dtypes import ScheduleType
from dace.transformation.passes.vectorization.mark_tile_dims import MarkTileDims
from dace.transformation.passes.vectorization.utils.map_predicates import is_tile_eligible, is_vectorizable_map


def _map_sdfg(
        name: str,
        ranges: Dict[str, str],
        schedule: ScheduleType = ScheduleType.Sequential,
        symbols: Tuple[str, ...] = (),
        size: int = 8,
) -> Tuple[dace.SDFGState, dace.nodes.MapEntry]:
    """A single innermost copy map ``B[idx] = A[idx]`` with the given ranges.

    :param name: SDFG name.
    :param ranges: Map parameter -> range string.
    :param schedule: Map schedule.
    :param symbols: Symbol names to declare on the SDFG.
    :param size: Array extent per dimension.
    :returns: The state and the map entry.
    """
    sdfg = dace.SDFG(name)
    for sym in symbols:
        sdfg.add_symbol(sym, dace.int64)
    ndim = len(ranges)
    sdfg.add_array("A", (size, ) * ndim, dace.float64)
    sdfg.add_array("B", (size, ) * ndim, dace.float64)
    state = sdfg.add_state()
    a, b = state.add_read("A"), state.add_write("B")
    me, mx = state.add_map("m", ranges, schedule=schedule)
    t = state.add_tasklet("c", {"inp"}, {"out"}, "out = inp")
    idx = ", ".join(ranges.keys())
    state.add_memlet_path(a, me, t, dst_conn="inp", memlet=dace.Memlet(f"A[{idx}]"))
    state.add_memlet_path(t, mx, b, src_conn="out", memlet=dace.Memlet(f"B[{idx}]"))
    return state, me


def test_trivial_single_param_map_refused():
    """A literal one-param ``0:1`` map (the ``*_gmap`` wrapper shape) is refused."""
    state, me = _map_sdfg("trivial_1p", {"i": "0:1"})
    assert not is_tile_eligible(state, me)
    assert not is_vectorizable_map(state, me)


def test_trivial_gpu_wrapper_map_refused():
    """The exact GPU-scheduled ``0:1`` wrapper shape is refused too."""
    state, me = _map_sdfg("trivial_gmap", {"assign__gmapi": "0:1"}, schedule=ScheduleType.GPU_Device)
    assert not is_tile_eligible(state, me)
    assert not is_vectorizable_map(state, me)


def test_all_trivial_multi_dim_map_refused():
    """A multi-dim map where ALL dims are provably 1-trip step-1 is refused."""
    state, me = _map_sdfg("trivial_2p", {"i": "0:1", "j": "0:1"})
    assert not is_tile_eligible(state, me)
    assert not is_vectorizable_map(state, me)


def test_symbolic_range_map_accepted():
    """A symbolic trip count is not provably trivial -> stays a candidate."""
    state, me = _map_sdfg("symbolic_1p", {"i": "0:N"}, symbols=("N", ))
    assert is_tile_eligible(state, me)
    assert is_vectorizable_map(state, me)


def test_mixed_trivial_symbolic_map_accepted():
    """One trivial dim next to a symbolic dim: total volume is symbolic -> accepted."""
    state, me = _map_sdfg("mixed_2p", {"i": "0:1", "j": "0:N"}, symbols=("N", ))
    assert is_tile_eligible(state, me)
    assert is_vectorizable_map(state, me)


def test_single_trip_strided_tail_map_accepted():
    """A single-trip map with step > 1 (masked remainder tail ``16:20:8``) is a
    legitimate W-strided tile map -> stays a candidate (refusing it left the
    tail scalar and computed one element instead of the masked tile)."""
    state, me = _map_sdfg("strided_tail", {"i": "16:20:8"}, size=32)
    assert is_tile_eligible(state, me)
    assert is_vectorizable_map(state, me)


def test_multi_dim_trivial_and_strided_tail_accepted():
    """A ``0:1`` dim next to a strided single-trip tail dim is not the wrapper
    shape (not ALL dims are 1-trip step-1) -> stays a candidate."""
    state, me = _map_sdfg("mixed_tail_2p", {"i": "0:1", "j": "16:20:8"}, size=32)
    assert is_tile_eligible(state, me)
    assert is_vectorizable_map(state, me)


def test_symbolic_step_map_accepted():
    """A ``0:1`` range with a SYMBOLIC step is not provably 1-trip step-1 ->
    stays a candidate (only literal step-1 wrappers are refused)."""
    state, me = _map_sdfg("symbolic_step", {"i": "0:1:s"}, symbols=("s", ))
    assert is_tile_eligible(state, me)
    assert is_vectorizable_map(state, me)


def test_normal_range_map_accepted():
    """A plain literal multi-trip map stays a candidate (gate sanity check)."""
    state, me = _map_sdfg("normal_1p", {"i": "0:8"})
    assert is_tile_eligible(state, me)
    assert is_vectorizable_map(state, me)


def test_mark_tile_dims_skips_trivial_gpu_wrapper():
    """MarkTileDims no longer crashes on a 1-param ``0:1`` GPU wrapper at K=2.

    Before the gate this raised ``NotImplementedError: ... has only 1 params
    (< K=2)`` (the wrapper passed the candidate gates as an innermost
    GPU-resident map); now the shared gate refuses it -> no spec, no crash.
    """
    state, me = _map_sdfg("gmap_k2", {"assign__gmapi": "0:1"}, schedule=ScheduleType.GPU_Device)
    assert MarkTileDims(widths=(8, 8), require_gpu_resident=True).apply_pass(state.sdfg, {}) is None
    # K=1: the wrapper must not become a tile candidate either.
    assert MarkTileDims(widths=(8, ), require_gpu_resident=True).apply_pass(state.sdfg, {}) is None


if __name__ == "__main__":
    test_trivial_single_param_map_refused()
    test_trivial_gpu_wrapper_map_refused()
    test_all_trivial_multi_dim_map_refused()
    test_symbolic_range_map_accepted()
    test_mixed_trivial_symbolic_map_accepted()
    test_single_trip_strided_tail_map_accepted()
    test_multi_dim_trivial_and_strided_tail_accepted()
    test_symbolic_step_map_accepted()
    test_normal_range_map_accepted()
    test_mark_tile_dims_skips_trivial_gpu_wrapper()
    print("ok")
