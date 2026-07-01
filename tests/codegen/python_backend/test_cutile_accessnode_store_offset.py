# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression tests for the AccessNode-centric cuTile store/load element offset (Bug 16).

The ``CuTilePythonCodeGen._generate_AccessNode`` load/store path derives the
``ct.load`` / ``ct.gather`` / ``ct.scatter`` element index from the map range and
block ids only; it dropped the constant begin offset carried by the outer
(global-array) memlet, so a tile written/read through an offset slice
(``B[1:33]``) landed one element too low -- silently wrong. The fix folds the
outer memlet's constant begin offset into the index (forcing the per-element
gather/scatter path when the offset is non-zero).

These build the AccessNode-centric SDFG directly (``A -> MapEntry -> tile ->
tasklet -> tile -> MapExit -> B``) so they exercise ``cutile_target.py`` rather
than the tile-op library-node expansion, and run on the GPU with cupy arrays.
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.dtypes import StorageType, ScheduleType, Language
from dace.memlet import Memlet

pytestmark = pytest.mark.gpu


def _build_offset_copy_sdfg(name, n, begin, tile_w):
    """A[begin:begin+tile_w] -> *2 -> B[begin:begin+tile_w] via CuTile_Tile nodes."""
    sdfg = dace.SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array("A", [n], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("B", [n], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("_tA", [tile_w], dace.float64, storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array("_tB", [tile_w], dace.float64, storage=StorageType.CuTile_Tile, transient=True)
    state = sdfg.add_state("main")
    me, mx = state.add_map("cutile_map", {"tile_i": "0:1"}, schedule=ScheduleType.CuTile)
    a = state.add_read("A")
    b = state.add_write("B")
    tA = state.add_access("_tA")
    tB = state.add_access("_tB")
    tk = state.add_tasklet("c", {"inp"}, {"out"}, "out = inp * 2.0", language=Language.Python)
    sub = f"{begin}:{begin + tile_w}"
    state.add_memlet_path(a, me, tA, memlet=Memlet(data="A", subset=sub))
    state.add_edge(tA, None, tk, "inp", Memlet(data="_tA", subset=f"0:{tile_w}"))
    state.add_edge(tk, "out", tB, None, Memlet(data="_tB", subset=f"0:{tile_w}"))
    state.add_memlet_path(tB, mx, b, memlet=Memlet(data="B", subset=sub))
    sdfg.fill_scope_connectors()
    return sdfg


@pytest.mark.parametrize("begin", [0, 1, 3])
def test_accessnode_offset_copy(begin):
    """Tile write/read through an offset slice must land at the right position."""
    cupy = pytest.importorskip("cupy")
    n, tile_w = 40, 32
    sdfg = _build_offset_copy_sdfg(f"an_offset_{begin}", n, begin, tile_w)
    a_host = np.arange(n, dtype=np.float64) + 0.5
    A = cupy.asarray(a_host)
    B = cupy.full(n, -9.0, dtype=cupy.float64)
    sdfg.compile()(A=A, B=B)
    expected = np.full(n, -9.0)
    expected[begin:begin + tile_w] = 2.0 * a_host[begin:begin + tile_w]
    np.testing.assert_allclose(cupy.asnumpy(B), expected, rtol=1e-14)


def test_accessnode_offset_differs_from_base0():
    """Before/after guard: an offset copy must differ from the base-0 copy."""
    cupy = pytest.importorskip("cupy")
    n, tile_w = 40, 32
    sdfg = _build_offset_copy_sdfg("an_offset_guard", n, 1, tile_w)
    a_host = np.arange(n, dtype=np.float64) + 0.5
    A = cupy.asarray(a_host)
    B = cupy.zeros(n, dtype=cupy.float64)
    sdfg.compile()(A=A, B=B)
    got = cupy.asnumpy(B)
    offset_ref = np.zeros(n)
    offset_ref[1:1 + tile_w] = 2.0 * a_host[1:1 + tile_w]
    base0_ref = np.zeros(n)
    base0_ref[0:tile_w] = 2.0 * a_host[0:tile_w]
    np.testing.assert_allclose(got, offset_ref, rtol=1e-14)
    assert not np.allclose(got, base0_ref)


if __name__ == "__main__":
    for bg in (0, 1, 3):
        test_accessnode_offset_copy(bg)
    test_accessnode_offset_differs_from_base0()
    print("accessnode offset regression tests passed")
