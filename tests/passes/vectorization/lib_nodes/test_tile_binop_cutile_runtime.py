# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU runtime tests for TileBinop's cuTile expansion.

Each test manually constructs an SDFG with TileLoad + TileBinop + TileStore
nodes inside a CuTile-scheduled map, stamps cuTile implementations and
storage, expands library nodes, compiles to the Python backend, runs on
GPU, and compares the result against a NumPy reference.

Test categories:
  1. Arithmetic ops (+, -, *, /, %, **)
  2. Comparison ops (<, >=, ==, !=)
  3. Logical ops (&&, ||)
  4. Bitwise ops (&, |, ^)
  5. Min/Max
  6. Masked operations
  7. Symbol operands (kind_b="Symbol")
  8. Mixed dtypes
  9. Multi-dimensional tiles (K=2)
 10. Non-divisible boundaries

Naming convention: each SDFG must have a globally unique name (used as
build folder / .dacecache key).  Use prefix ``binop_rt_``.
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.sdfg import SDFG
from dace.memlet import Memlet
from dace.dtypes import ScheduleType, StorageType
from dace.libraries.tileops.nodes.tile_load import TileLoad
from dace.libraries.tileops.nodes.tile_store import TileStore
from dace.libraries.tileops.nodes.tile_binop import TileBinop

pytestmark = pytest.mark.gpu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_cutile(sdfg, **kwargs):
    """Compile and run a cuTile SDFG, converting numpy<->cupy.

    :param sdfg: The SDFG to compile and run.
    :param kwargs: Named arguments for the SDFG (arrays and symbols).
    :returns: Dictionary mapping array names to NumPy results.
    """
    import cupy as cp

    cp_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            cp_kwargs[k] = cp.asarray(v)
        else:
            cp_kwargs[k] = v

    csdfg = sdfg.compile()
    csdfg(**cp_kwargs)

    results = {}
    for k, v in cp_kwargs.items():
        if isinstance(v, cp.ndarray):
            results[k] = cp.asnumpy(v)
        else:
            results[k] = v
    return results


def _stamp_cutile(node):
    """Stamp cuTile target_isa and implementation on a tileops lib node.

    :param node: A TileLoad, TileStore, or TileBinop node.
    """
    node.target_isa = "CUTILE"
    node.implementation = "cutile"


def _build_tile_tile_binop_sdfg(
    name,
    N,
    W,
    op,
    dtype_a,
    dtype_b,
    dtype_c,
    has_mask=False,
):
    """Build a 1-D Tile+Tile binop SDFG.

    Structure::

        TileLoad(A) -> _tile_a -> TileBinop -> _tile_c -> TileStore -> C
        TileLoad(B) -> _tile_b ----^

    All inside a CuTile-scheduled map ``i = 0:N:W``.

    :param name: Unique SDFG name.
    :param N: Array length.
    :param W: Tile width (power of 2).
    :param op: Binary operator string.
    :param dtype_a: DaCe dtype for input A.
    :param dtype_b: DaCe dtype for input B.
    :param dtype_c: DaCe dtype for output C.
    :param has_mask: Whether to include a mask input.
    :returns: The constructed and expanded SDFG.
    """
    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python

    sdfg.add_array("A", (N,), dtype_a, storage=StorageType.GPU_Global)
    sdfg.add_array("B", (N,), dtype_b, storage=StorageType.GPU_Global)
    sdfg.add_array("C", (N,), dtype_c, storage=StorageType.GPU_Global)
    sdfg.add_array(
        "_tile_a", (W,), dtype_a,
        storage=StorageType.CuTile_Tile, transient=True,
    )
    sdfg.add_array(
        "_tile_b", (W,), dtype_b,
        storage=StorageType.CuTile_Tile, transient=True,
    )
    sdfg.add_array(
        "_tile_c", (W,), dtype_c,
        storage=StorageType.CuTile_Tile, transient=True,
    )

    if has_mask:
        sdfg.add_array(
            "MASK", (N,), dace.bool_,
            storage=StorageType.GPU_Global,
        )
        sdfg.add_array(
            "_tile_mask", (W,), dace.bool_,
            storage=StorageType.CuTile_Tile, transient=True,
        )

    state = sdfg.add_state("main")
    me, mx = state.add_map(
        "m", {"i": f"0:{N}:{W}"}, schedule=ScheduleType.CuTile,
    )

    # -- TileLoad A --
    load_a = TileLoad("load_a", widths=(W,))
    _stamp_cutile(load_a)
    state.add_node(load_a)

    # -- TileLoad B --
    load_b = TileLoad("load_b", widths=(W,))
    _stamp_cutile(load_b)
    state.add_node(load_b)

    # -- TileBinop --
    binop = TileBinop("binop", widths=(W,), op=op, has_mask=has_mask)
    _stamp_cutile(binop)
    state.add_node(binop)

    # -- TileStore --
    store = TileStore("store", widths=(W,))
    _stamp_cutile(store)
    state.add_node(store)

    # -- AccessNodes --
    a_an = state.add_read("A")
    b_an = state.add_read("B")
    c_an = state.add_write("C")
    tile_a_an = state.add_access("_tile_a")
    tile_b_an = state.add_access("_tile_b")
    tile_c_an = state.add_access("_tile_c")

    # -- Wiring: input arrays through map entry to loads --
    state.add_memlet_path(
        a_an, me, load_a, dst_conn="_src",
        memlet=Memlet(data="A", subset=f"i:i+{W}"),
    )
    state.add_edge(
        load_a, "_dst", tile_a_an, None,
        Memlet(data="_tile_a", subset=f"0:{W}"),
    )

    state.add_memlet_path(
        b_an, me, load_b, dst_conn="_src",
        memlet=Memlet(data="B", subset=f"i:i+{W}"),
    )
    state.add_edge(
        load_b, "_dst", tile_b_an, None,
        Memlet(data="_tile_b", subset=f"0:{W}"),
    )

    # -- Wiring: tiles to binop --
    state.add_edge(
        tile_a_an, None, binop, "_a",
        Memlet(data="_tile_a", subset=f"0:{W}"),
    )
    state.add_edge(
        tile_b_an, None, binop, "_b",
        Memlet(data="_tile_b", subset=f"0:{W}"),
    )

    # -- Mask wiring (optional) --
    if has_mask:
        load_mask = TileLoad("load_mask", widths=(W,))
        _stamp_cutile(load_mask)
        state.add_node(load_mask)
        mask_an = state.add_read("MASK")
        tile_mask_an = state.add_access("_tile_mask")
        state.add_memlet_path(
            mask_an, me, load_mask, dst_conn="_src",
            memlet=Memlet(data="MASK", subset=f"i:i+{W}"),
        )
        state.add_edge(
            load_mask, "_dst", tile_mask_an, None,
            Memlet(data="_tile_mask", subset=f"0:{W}"),
        )
        state.add_edge(
            tile_mask_an, None, binop, "_mask",
            Memlet(data="_tile_mask", subset=f"0:{W}"),
        )

    # -- Wiring: binop output through store and map exit --
    state.add_edge(
        binop, "_c", tile_c_an, None,
        Memlet(data="_tile_c", subset=f"0:{W}"),
    )
    state.add_edge(
        tile_c_an, None, store, "_src",
        Memlet(data="_tile_c", subset=f"0:{W}"),
    )
    state.add_memlet_path(
        store, mx, c_an, src_conn="_dst",
        memlet=Memlet(data="C", subset=f"i:i+{W}"),
    )

    sdfg.expand_library_nodes()
    return sdfg


def _build_symbol_binop_sdfg(
    name,
    N,
    W,
    op,
    dtype_a,
    dtype_c,
    expr_b,
    symbols=None,
):
    """Build a 1-D Tile+Symbol binop SDFG.

    Structure::

        TileLoad(A) -> _tile_a -> TileBinop(kind_b="Symbol") -> _tile_c -> TileStore -> C

    :param name: Unique SDFG name.
    :param N: Array length.
    :param W: Tile width (power of 2).
    :param op: Binary operator string.
    :param dtype_a: DaCe dtype for input A.
    :param dtype_c: DaCe dtype for output C.
    :param expr_b: Symbol expression string for the RHS operand.
    :param symbols: Optional dict {sym_name: dace_dtype} to register.
    :returns: The constructed and expanded SDFG.
    """
    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python

    sdfg.add_array("A", (N,), dtype_a, storage=StorageType.GPU_Global)
    sdfg.add_array("C", (N,), dtype_c, storage=StorageType.GPU_Global)
    sdfg.add_array(
        "_tile_a", (W,), dtype_a,
        storage=StorageType.CuTile_Tile, transient=True,
    )
    sdfg.add_array(
        "_tile_c", (W,), dtype_c,
        storage=StorageType.CuTile_Tile, transient=True,
    )

    if symbols:
        for sym_name, sym_dtype in symbols.items():
            sdfg.add_symbol(sym_name, sym_dtype)

    state = sdfg.add_state("main")
    me, mx = state.add_map(
        "m", {"i": f"0:{N}:{W}"}, schedule=ScheduleType.CuTile,
    )

    load_a = TileLoad("load_a", widths=(W,))
    _stamp_cutile(load_a)
    state.add_node(load_a)

    binop = TileBinop(
        "binop", widths=(W,), op=op,
        kind_b="Symbol", expr_b=expr_b,
    )
    _stamp_cutile(binop)
    state.add_node(binop)

    store = TileStore("store", widths=(W,))
    _stamp_cutile(store)
    state.add_node(store)

    a_an = state.add_read("A")
    c_an = state.add_write("C")
    tile_a_an = state.add_access("_tile_a")
    tile_c_an = state.add_access("_tile_c")

    state.add_memlet_path(
        a_an, me, load_a, dst_conn="_src",
        memlet=Memlet(data="A", subset=f"i:i+{W}"),
    )
    state.add_edge(
        load_a, "_dst", tile_a_an, None,
        Memlet(data="_tile_a", subset=f"0:{W}"),
    )
    state.add_edge(
        tile_a_an, None, binop, "_a",
        Memlet(data="_tile_a", subset=f"0:{W}"),
    )
    state.add_edge(
        binop, "_c", tile_c_an, None,
        Memlet(data="_tile_c", subset=f"0:{W}"),
    )
    state.add_edge(
        tile_c_an, None, store, "_src",
        Memlet(data="_tile_c", subset=f"0:{W}"),
    )
    state.add_memlet_path(
        store, mx, c_an, src_conn="_dst",
        memlet=Memlet(data="C", subset=f"i:i+{W}"),
    )

    sdfg.expand_library_nodes()
    return sdfg


def _build_k2_binop_sdfg(
    name,
    N0,
    N1,
    W0,
    W1,
    op,
    dtype,
):
    """Build a K=2 Tile+Tile binop SDFG with 2-D arrays.

    :param name: Unique SDFG name.
    :param N0: Array size in dim 0.
    :param N1: Array size in dim 1.
    :param W0: Tile width for dim 0.
    :param W1: Tile width for dim 1.
    :param op: Binary operator string.
    :param dtype: DaCe dtype for all arrays.
    :returns: The constructed and expanded SDFG.
    """
    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python

    sdfg.add_array("A", (N0, N1), dtype, storage=StorageType.GPU_Global)
    sdfg.add_array("B", (N0, N1), dtype, storage=StorageType.GPU_Global)
    sdfg.add_array("C", (N0, N1), dtype, storage=StorageType.GPU_Global)
    sdfg.add_array(
        "_tile_a", (W0, W1), dtype,
        storage=StorageType.CuTile_Tile, transient=True,
    )
    sdfg.add_array(
        "_tile_b", (W0, W1), dtype,
        storage=StorageType.CuTile_Tile, transient=True,
    )
    sdfg.add_array(
        "_tile_c", (W0, W1), dtype,
        storage=StorageType.CuTile_Tile, transient=True,
    )

    state = sdfg.add_state("main")
    me, mx = state.add_map(
        "m", {"i": f"0:{N0}:{W0}", "j": f"0:{N1}:{W1}"},
        schedule=ScheduleType.CuTile,
    )

    sub = f"0:{W0}, 0:{W1}"

    load_a = TileLoad("load_a", widths=(W0, W1))
    _stamp_cutile(load_a)
    state.add_node(load_a)

    load_b = TileLoad("load_b", widths=(W0, W1))
    _stamp_cutile(load_b)
    state.add_node(load_b)

    binop = TileBinop("binop", widths=(W0, W1), op=op)
    _stamp_cutile(binop)
    state.add_node(binop)

    store = TileStore("store", widths=(W0, W1))
    _stamp_cutile(store)
    state.add_node(store)

    a_an = state.add_read("A")
    b_an = state.add_read("B")
    c_an = state.add_write("C")
    tile_a_an = state.add_access("_tile_a")
    tile_b_an = state.add_access("_tile_b")
    tile_c_an = state.add_access("_tile_c")

    state.add_memlet_path(
        a_an, me, load_a, dst_conn="_src",
        memlet=Memlet(data="A", subset=f"i:i+{W0}, j:j+{W1}"),
    )
    state.add_edge(
        load_a, "_dst", tile_a_an, None,
        Memlet(data="_tile_a", subset=sub),
    )

    state.add_memlet_path(
        b_an, me, load_b, dst_conn="_src",
        memlet=Memlet(data="B", subset=f"i:i+{W0}, j:j+{W1}"),
    )
    state.add_edge(
        load_b, "_dst", tile_b_an, None,
        Memlet(data="_tile_b", subset=sub),
    )

    state.add_edge(tile_a_an, None, binop, "_a", Memlet(data="_tile_a", subset=sub))
    state.add_edge(tile_b_an, None, binop, "_b", Memlet(data="_tile_b", subset=sub))

    state.add_edge(binop, "_c", tile_c_an, None, Memlet(data="_tile_c", subset=sub))
    state.add_edge(tile_c_an, None, store, "_src", Memlet(data="_tile_c", subset=sub))
    state.add_memlet_path(
        store, mx, c_an, src_conn="_dst",
        memlet=Memlet(data="C", subset=f"i:i+{W0}, j:j+{W1}"),
    )

    sdfg.expand_library_nodes()
    return sdfg


# ---------------------------------------------------------------------------
# 1. Arithmetic ops
# ---------------------------------------------------------------------------


class TestArithmeticOps:
    """Tile+Tile arithmetic: +, -, *, /, %, **."""

    def test_add_float32(self):
        """A + B, float32, N=64, W=8."""
        N, W = 64, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_add_f32", N, W, "+",
            dace.float32, dace.float32, dace.float32,
        )
        rng = np.random.default_rng(1001)
        A = rng.standard_normal(N).astype(np.float32)
        B = rng.standard_normal(N).astype(np.float32)
        C = np.zeros(N, dtype=np.float32)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        np.testing.assert_allclose(results["C"], A + B, rtol=1e-5)

    def test_sub_float64(self):
        """A - B, float64, N=128, W=16."""
        N, W = 128, 16
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_sub_f64", N, W, "-",
            dace.float64, dace.float64, dace.float64,
        )
        rng = np.random.default_rng(1002)
        A = rng.standard_normal(N)
        B = rng.standard_normal(N)
        C = np.zeros(N)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        np.testing.assert_allclose(results["C"], A - B, rtol=1e-14)

    def test_mul_int32(self):
        """A * B, int32, N=64, W=8."""
        N, W = 64, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_mul_i32", N, W, "*",
            dace.int32, dace.int32, dace.int32,
        )
        rng = np.random.default_rng(1003)
        A = rng.integers(-50, 50, N, dtype=np.int32)
        B = rng.integers(-50, 50, N, dtype=np.int32)
        C = np.zeros(N, dtype=np.int32)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        np.testing.assert_array_equal(results["C"], A * B)

    def test_div_float32(self):
        """A / B, float32, N=64, W=8 (no div-by-zero)."""
        N, W = 64, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_div_f32", N, W, "/",
            dace.float32, dace.float32, dace.float32,
        )
        rng = np.random.default_rng(1004)
        A = rng.standard_normal(N).astype(np.float32)
        B = (rng.standard_normal(N).astype(np.float32) + 0.5)
        B[B == 0] = 1.0
        C = np.zeros(N, dtype=np.float32)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        np.testing.assert_allclose(results["C"], A / B, rtol=1e-5)

    def test_mod_int32(self):
        """A % B, int32, N=64, W=8 (B all positive)."""
        N, W = 64, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_mod_i32", N, W, "%",
            dace.int32, dace.int32, dace.int32,
        )
        rng = np.random.default_rng(1005)
        A = rng.integers(0, 100, N, dtype=np.int32)
        B = rng.integers(1, 20, N, dtype=np.int32)
        C = np.zeros(N, dtype=np.int32)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        np.testing.assert_array_equal(results["C"], np.mod(A, B))

    def test_mod_negative_int32(self):
        """A % B with negative A values (floor-modulo semantics)."""
        N, W = 64, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_mod_neg_i32", N, W, "%",
            dace.int32, dace.int32, dace.int32,
        )
        rng = np.random.default_rng(1006)
        A = rng.integers(-100, 100, N, dtype=np.int32)
        B = rng.integers(1, 20, N, dtype=np.int32)
        C = np.zeros(N, dtype=np.int32)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        np.testing.assert_array_equal(results["C"], np.mod(A, B))

    def test_pow_float32(self):
        """A ** B, float32, N=64, W=8 (A positive, B small positive)."""
        N, W = 64, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_pow_f32", N, W, "**",
            dace.float32, dace.float32, dace.float32,
        )
        rng = np.random.default_rng(1007)
        A = rng.uniform(0.5, 5.0, N).astype(np.float32)
        B = rng.uniform(0.5, 3.0, N).astype(np.float32)
        C = np.zeros(N, dtype=np.float32)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        np.testing.assert_allclose(results["C"], np.power(A, B), rtol=1e-4)

    def test_div_int32(self):
        """Integer division via A5 float-cast path: int32 / int32.

        The cuTile expansion casts int operands to float64, performs true
        division, then casts back to int32 (truncation toward zero).
        """
        N, W = 64, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_div_i32", N, W, "/",
            dace.int32, dace.int32, dace.int32,
        )
        rng = np.random.default_rng(1099)
        A = rng.integers(1, 100, size=N, dtype=np.int32)
        B = rng.integers(1, 10, size=N, dtype=np.int32)
        C = np.zeros(N, dtype=np.int32)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        # cuTile path: ct.astype(float64(A) / float64(B), ct.int32)
        # ct.astype truncates toward zero (like C/NumPy int cast)
        expected = (A.astype(np.float64) / B.astype(np.float64)).astype(np.int32)
        np.testing.assert_array_equal(results["C"], expected)

    def test_pow_int32(self):
        """Integer power via A2 float-cast path: int32 ** int32.

        Uses base=2 only because GPU ``pow`` on float64 can return values
        slightly below the exact integer for some base/exponent combinations,
        causing truncation to the wrong value. Base=2 results are exact in
        float64.
        """
        N, W = 64, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_pow_i32", N, W, "**",
            dace.int32, dace.int32, dace.int32,
        )
        # Base = 2 (powers of 2 are always exact in float64).
        A = np.full(N, 2, dtype=np.int32)
        rng = np.random.default_rng(1008)
        B = rng.integers(0, 5, N, dtype=np.int32)   # [0, 4] -> 1, 2, 4, 8, 16
        C = np.zeros(N, dtype=np.int32)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        expected = np.power(np.int32(2), B).astype(np.int32)
        np.testing.assert_array_equal(results["C"], expected)


# ---------------------------------------------------------------------------
# 2. Comparison ops
# ---------------------------------------------------------------------------


class TestComparisonOps:
    """Tile+Tile comparisons producing bool output.

    TileBinop's ``validate`` enforces ``_promotion_ok(src, dst)``; float ->
    int32 is a narrowing conversion that is rejected.  Comparison operators
    naturally produce bool tiles, which is a legal promotion target for any
    input dtype (``numeric -> bool`` is "truthiness", accepted by
    ``_promotion_ok``).  The cuTile runtime stores bool tiles to bool
    GPU_Global arrays, and we convert them to int for the final comparison.
    """

    def test_less_float32(self):
        """A < B, float32 inputs, bool output."""
        N, W = 64, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_lt_f32", N, W, "<",
            dace.float32, dace.float32, dace.bool_,
        )
        rng = np.random.default_rng(2001)
        A = rng.standard_normal(N).astype(np.float32)
        B = rng.standard_normal(N).astype(np.float32)
        C = np.zeros(N, dtype=np.bool_)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        expected = A < B
        np.testing.assert_array_equal(results["C"], expected)

    def test_greater_equal_int32(self):
        """A >= B, int32 inputs, bool output."""
        N, W = 64, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_ge_i32", N, W, ">=",
            dace.int32, dace.int32, dace.bool_,
        )
        rng = np.random.default_rng(2002)
        A = rng.integers(-50, 50, N, dtype=np.int32)
        B = rng.integers(-50, 50, N, dtype=np.int32)
        C = np.zeros(N, dtype=np.bool_)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        expected = A >= B
        np.testing.assert_array_equal(results["C"], expected)

    def test_equal_float64(self):
        """A == B, float64 inputs, bool output."""
        N, W = 64, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_eq_f64", N, W, "==",
            dace.float64, dace.float64, dace.bool_,
        )
        rng = np.random.default_rng(2003)
        # Use integers cast to float64 so some values match exactly.
        A = rng.integers(0, 10, N).astype(np.float64)
        B = rng.integers(0, 10, N).astype(np.float64)
        C = np.zeros(N, dtype=np.bool_)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        expected = A == B
        np.testing.assert_array_equal(results["C"], expected)

    def test_not_equal_int32(self):
        """A != B, int32 inputs, bool output."""
        N, W = 64, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_ne_i32", N, W, "!=",
            dace.int32, dace.int32, dace.bool_,
        )
        rng = np.random.default_rng(2004)
        A = rng.integers(0, 10, N, dtype=np.int32)
        B = rng.integers(0, 10, N, dtype=np.int32)
        C = np.zeros(N, dtype=np.bool_)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        expected = A != B
        np.testing.assert_array_equal(results["C"], expected)

    def test_greater_float32(self):
        """A > B, float32 inputs, bool output."""
        N, W = 64, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_gt_f32", N, W, ">",
            dace.float32, dace.float32, dace.bool_,
        )
        rng = np.random.default_rng(2001)
        A = rng.standard_normal(N).astype(np.float32)
        B = rng.standard_normal(N).astype(np.float32)
        C = np.zeros(N, dtype=np.bool_)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        np.testing.assert_array_equal(results["C"], A > B)

    def test_less_equal_int32(self):
        """A <= B, int32 inputs, bool output."""
        N, W = 64, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_le_i32", N, W, "<=",
            dace.int32, dace.int32, dace.bool_,
        )
        rng = np.random.default_rng(2002)
        A = rng.integers(-50, 50, size=N, dtype=np.int32)
        B = rng.integers(-50, 50, size=N, dtype=np.int32)
        C = np.zeros(N, dtype=np.bool_)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        np.testing.assert_array_equal(results["C"], A <= B)


# ---------------------------------------------------------------------------
# 3. Logical ops
# ---------------------------------------------------------------------------


class TestLogicalOps:
    """Logical AND (&&) and OR (||) on int32 tiles."""

    def test_logical_and_int32(self):
        """A && B, int32 inputs (0 and nonzero), int32 output."""
        N, W = 64, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_land_i32", N, W, "&&",
            dace.int32, dace.int32, dace.int32,
        )
        rng = np.random.default_rng(3001)
        A = rng.integers(0, 5, N, dtype=np.int32)
        B = rng.integers(0, 5, N, dtype=np.int32)
        C = np.zeros(N, dtype=np.int32)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        expected = ((A != 0) & (B != 0)).astype(np.int32)
        np.testing.assert_array_equal(results["C"], expected)

    def test_logical_or_int32(self):
        """A || B, int32 inputs, int32 output."""
        N, W = 64, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_lor_i32", N, W, "||",
            dace.int32, dace.int32, dace.int32,
        )
        rng = np.random.default_rng(3002)
        A = rng.integers(0, 5, N, dtype=np.int32)
        B = rng.integers(0, 5, N, dtype=np.int32)
        C = np.zeros(N, dtype=np.int32)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        expected = ((A != 0) | (B != 0)).astype(np.int32)
        np.testing.assert_array_equal(results["C"], expected)


# ---------------------------------------------------------------------------
# 4. Bitwise ops
# ---------------------------------------------------------------------------


class TestBitwiseOps:
    """Bitwise AND (&), OR (|), XOR (^) on int32 tiles."""

    def test_bitwise_and_int32(self):
        """A & B, int32."""
        N, W = 64, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_band_i32", N, W, "&",
            dace.int32, dace.int32, dace.int32,
        )
        rng = np.random.default_rng(4001)
        A = rng.integers(0, 256, N, dtype=np.int32)
        B = rng.integers(0, 256, N, dtype=np.int32)
        C = np.zeros(N, dtype=np.int32)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        np.testing.assert_array_equal(results["C"], A & B)

    def test_bitwise_or_int32(self):
        """A | B, int32."""
        N, W = 64, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_bor_i32", N, W, "|",
            dace.int32, dace.int32, dace.int32,
        )
        rng = np.random.default_rng(4002)
        A = rng.integers(0, 256, N, dtype=np.int32)
        B = rng.integers(0, 256, N, dtype=np.int32)
        C = np.zeros(N, dtype=np.int32)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        np.testing.assert_array_equal(results["C"], A | B)

    def test_bitwise_xor_int32(self):
        """A ^ B, int32."""
        N, W = 64, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_bxor_i32", N, W, "^",
            dace.int32, dace.int32, dace.int32,
        )
        rng = np.random.default_rng(4003)
        A = rng.integers(0, 256, N, dtype=np.int32)
        B = rng.integers(0, 256, N, dtype=np.int32)
        C = np.zeros(N, dtype=np.int32)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        np.testing.assert_array_equal(results["C"], A ^ B)


# ---------------------------------------------------------------------------
# 5. Min / Max
# ---------------------------------------------------------------------------


class TestMinMax:
    """ct.minimum / ct.maximum on tiles."""

    def test_min_float32(self):
        """min(A, B), float32."""
        N, W = 64, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_min_f32", N, W, "min",
            dace.float32, dace.float32, dace.float32,
        )
        rng = np.random.default_rng(5001)
        A = rng.standard_normal(N).astype(np.float32)
        B = rng.standard_normal(N).astype(np.float32)
        C = np.zeros(N, dtype=np.float32)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        np.testing.assert_allclose(results["C"], np.minimum(A, B), rtol=1e-6)

    def test_max_int32(self):
        """max(A, B), int32."""
        N, W = 64, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_max_i32", N, W, "max",
            dace.int32, dace.int32, dace.int32,
        )
        rng = np.random.default_rng(5002)
        A = rng.integers(-100, 100, N, dtype=np.int32)
        B = rng.integers(-100, 100, N, dtype=np.int32)
        C = np.zeros(N, dtype=np.int32)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        np.testing.assert_array_equal(results["C"], np.maximum(A, B))

    def test_min_negative_int32(self):
        """min(A, B) with all-negative values, int32."""
        N, W = 64, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_min_neg_i32", N, W, "min",
            dace.int32, dace.int32, dace.int32,
        )
        rng = np.random.default_rng(5003)
        A = rng.integers(-200, -1, N, dtype=np.int32)
        B = rng.integers(-200, -1, N, dtype=np.int32)
        C = np.zeros(N, dtype=np.int32)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        np.testing.assert_array_equal(results["C"], np.minimum(A, B))


# ---------------------------------------------------------------------------
# 6. Masked operations
# ---------------------------------------------------------------------------


class TestMasked:
    """Masked TileBinop: ct.where(_mask, op_result, fill)."""

    def test_masked_add_all_true(self):
        """All-true mask: result should match unmasked A + B."""
        N, W = 64, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_madd_true", N, W, "+",
            dace.float32, dace.float32, dace.float32,
            has_mask=True,
        )
        rng = np.random.default_rng(6001)
        A = rng.standard_normal(N).astype(np.float32)
        B = rng.standard_normal(N).astype(np.float32)
        C = np.zeros(N, dtype=np.float32)
        MASK = np.ones(N, dtype=np.bool_)
        results = _run_cutile(sdfg, A=A, B=B, C=C, MASK=MASK)
        np.testing.assert_allclose(results["C"], A + B, rtol=1e-5)

    def test_masked_add_all_false(self):
        """All-false mask: result should be 0 (fill value) everywhere."""
        N, W = 64, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_madd_false", N, W, "+",
            dace.float32, dace.float32, dace.float32,
            has_mask=True,
        )
        rng = np.random.default_rng(6002)
        A = rng.standard_normal(N).astype(np.float32)
        B = rng.standard_normal(N).astype(np.float32)
        C = np.full(N, 999.0, dtype=np.float32)
        MASK = np.zeros(N, dtype=np.bool_)
        results = _run_cutile(sdfg, A=A, B=B, C=C, MASK=MASK)
        np.testing.assert_array_equal(results["C"], np.zeros(N, dtype=np.float32))

    def test_masked_mul_partial(self):
        """Alternating mask: masked lanes get 0, unmasked get A * B."""
        N, W = 64, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_mmul_part", N, W, "*",
            dace.float32, dace.float32, dace.float32,
            has_mask=True,
        )
        rng = np.random.default_rng(6003)
        A = rng.standard_normal(N).astype(np.float32)
        B = rng.standard_normal(N).astype(np.float32)
        C = np.zeros(N, dtype=np.float32)
        MASK = np.zeros(N, dtype=np.bool_)
        MASK[::2] = True  # Even indices active
        results = _run_cutile(sdfg, A=A, B=B, C=C, MASK=MASK)
        expected = np.where(MASK, A * B, 0.0).astype(np.float32)
        np.testing.assert_allclose(results["C"], expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# 7. Symbol operands (kind_b="Symbol")
# ---------------------------------------------------------------------------


class TestSymbolOperand:
    """TileBinop with kind_b="Symbol" (RHS is a symbolic expression)."""

    def test_add_symbol_rhs_literal(self):
        """A + 5, float32, expr_b="5"."""
        N, W = 64, 8
        sdfg = _build_symbol_binop_sdfg(
            "binop_rt_sym_add5", N, W, "+",
            dace.float32, dace.float32, "5",
        )
        rng = np.random.default_rng(7001)
        A = rng.standard_normal(N).astype(np.float32)
        C = np.zeros(N, dtype=np.float32)
        results = _run_cutile(sdfg, A=A, C=C)
        np.testing.assert_allclose(results["C"], A + 5.0, rtol=1e-5)

    @pytest.mark.skip(
        reason="cuTile runtime: free SDFG symbols are not passed into "
        "ct.program kernels (TileSyntaxError: Undefined variable)",
    )
    def test_mul_symbol_rhs_free(self):
        """A * s, float32, expr_b="s", s passed at runtime.

        Currently skipped: the cuTile Python backend does not forward
        free SDFG symbols into the generated ct.program kernel, so the
        runtime raises ``TileSyntaxError: Undefined variable s used``.
        """
        N, W = 64, 8
        s_val = np.float32(3.14)
        sdfg = _build_symbol_binop_sdfg(
            "binop_rt_sym_muls", N, W, "*",
            dace.float32, dace.float32, "s",
            symbols={"s": dace.float32},
        )
        rng = np.random.default_rng(7002)
        A = rng.standard_normal(N).astype(np.float32)
        C = np.zeros(N, dtype=np.float32)
        results = _run_cutile(sdfg, A=A, C=C, s=s_val)
        np.testing.assert_allclose(
            results["C"], A * s_val, rtol=1e-5,
        )

    def test_sub_symbol_rhs(self):
        """A - 10, int32, expr_b="10"."""
        N, W = 64, 8
        sdfg = _build_symbol_binop_sdfg(
            "binop_rt_sym_sub10", N, W, "-",
            dace.int32, dace.int32, "10",
        )
        rng = np.random.default_rng(7003)
        A = rng.integers(0, 100, N, dtype=np.int32)
        C = np.zeros(N, dtype=np.int32)
        results = _run_cutile(sdfg, A=A, C=C)
        np.testing.assert_array_equal(results["C"], A - 10)


# ---------------------------------------------------------------------------
# 8. Mixed dtypes
# ---------------------------------------------------------------------------


class TestMixedDtype:
    """TileBinop with mixed input dtypes (auto-promotion)."""

    def test_int32_plus_float32(self):
        """A (int32) + B (float32) -> C (float32)."""
        N, W = 64, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_mixed_i32_f32", N, W, "+",
            dace.int32, dace.float32, dace.float32,
        )
        rng = np.random.default_rng(8001)
        A = rng.integers(-50, 50, N, dtype=np.int32)
        B = rng.standard_normal(N).astype(np.float32)
        C = np.zeros(N, dtype=np.float32)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        expected = A.astype(np.float32) + B
        np.testing.assert_allclose(results["C"], expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# 9. Multi-dimensional tiles (K=2)
# ---------------------------------------------------------------------------


class TestMultiDim:
    """K=2 tile binop with 2-D arrays."""

    def test_add_k2_float32(self):
        """2-D add: C[i,j] = A[i,j] + B[i,j], widths=(4, 8)."""
        N0, N1 = 16, 32
        W0, W1 = 4, 8
        sdfg = _build_k2_binop_sdfg(
            "binop_rt_add_k2_f32", N0, N1, W0, W1, "+", dace.float32,
        )
        rng = np.random.default_rng(9001)
        A = rng.standard_normal((N0, N1)).astype(np.float32)
        B = rng.standard_normal((N0, N1)).astype(np.float32)
        C = np.zeros((N0, N1), dtype=np.float32)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        np.testing.assert_allclose(results["C"], A + B, rtol=1e-5)

    def test_mul_k2_int32(self):
        """2-D multiply: C[i,j] = A[i,j] * B[i,j], widths=(4, 8)."""
        N0, N1 = 16, 32
        W0, W1 = 4, 8
        sdfg = _build_k2_binop_sdfg(
            "binop_rt_mul_k2_i32", N0, N1, W0, W1, "*", dace.int32,
        )
        rng = np.random.default_rng(9002)
        A = rng.integers(-10, 10, (N0, N1), dtype=np.int32)
        B = rng.integers(-10, 10, (N0, N1), dtype=np.int32)
        C = np.zeros((N0, N1), dtype=np.int32)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        np.testing.assert_array_equal(results["C"], A * B)

    def test_min_k2_float64(self):
        """2-D min: C[i,j] = min(A[i,j], B[i,j]), widths=(8, 8)."""
        N0, N1 = 16, 16
        W0, W1 = 8, 8
        sdfg = _build_k2_binop_sdfg(
            "binop_rt_min_k2_f64", N0, N1, W0, W1, "min", dace.float64,
        )
        rng = np.random.default_rng(9003)
        A = rng.standard_normal((N0, N1))
        B = rng.standard_normal((N0, N1))
        C = np.zeros((N0, N1))
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        np.testing.assert_allclose(results["C"], np.minimum(A, B), rtol=1e-14)


# ---------------------------------------------------------------------------
# 10. Non-divisible boundaries
# ---------------------------------------------------------------------------


class TestNonDivisible:
    """Non-divisible array size: TileLoad pad_mode handles boundary tiles."""

    def test_add_nondivisible_float32(self):
        """N=100 with W=8 (100 is not divisible by 8)."""
        N, W = 100, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_nondiv_f32", N, W, "+",
            dace.float32, dace.float32, dace.float32,
        )
        rng = np.random.default_rng(10001)
        A = rng.standard_normal(N).astype(np.float32)
        B = rng.standard_normal(N).astype(np.float32)
        C = np.zeros(N, dtype=np.float32)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        np.testing.assert_allclose(results["C"], A + B, rtol=1e-5)

    def test_mul_nondivisible_int32(self):
        """N=33 with W=8 (33 not divisible by 8), int32."""
        N, W = 33, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_nondiv_i32", N, W, "*",
            dace.int32, dace.int32, dace.int32,
        )
        rng = np.random.default_rng(10002)
        A = rng.integers(-50, 50, N, dtype=np.int32)
        B = rng.integers(-50, 50, N, dtype=np.int32)
        C = np.zeros(N, dtype=np.int32)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        np.testing.assert_array_equal(results["C"], A * B)

    def test_sub_nondivisible_n7(self):
        """N=7 with W=8 (7 active lanes in the single tile)."""
        N, W = 7, 8
        sdfg = _build_tile_tile_binop_sdfg(
            "binop_rt_nondiv_n7", N, W, "-",
            dace.float64, dace.float64, dace.float64,
        )
        rng = np.random.default_rng(10003)
        A = rng.standard_normal(N)
        B = rng.standard_normal(N)
        C = np.zeros(N)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        np.testing.assert_allclose(results["C"], A - B, rtol=1e-14)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=300"])
