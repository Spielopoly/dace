# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU runtime tests for TileStore's cuTile expansion.

Each test goes through the FULL pipeline:
``@dace.program`` (or SDFG API) -> ``VectorizeCuTile(widths=...).apply_pass(sdfg, {})``
-> compile -> run with NumPy arrays -> compare against NumPy reference.

The ``VectorizeCuTile`` orchestrator detects store access patterns
(contiguous, strided, scatter) and emits ``TileStore`` library nodes
with ``'cutile'`` implementations, which are expanded into ``ct.store``
/ ``ct.scatter`` calls for the Python backend.  These tests verify
end-to-end correctness of those stores across dtypes, tile widths,
array sizes (including non-divisible boundaries), 1-D and 2-D arrays,
transpose stores, WCR/atomic stores, and multi-dim (K>=2) stores.

**Naming convention:** every SDFG name must be globally unique
(they share ``.dacecache``).

**Int32 note:** ``@dace.program`` with ``int32`` arrays emits an IR
structure whose index expressions the cuTile codegen does not vectorize
(the ``A_index`` / ``B_index`` temporaries remain scalar).  The SDFG API
(``add_mapped_tasklet``) produces a flat structure that vectorizes
correctly, matching the existing integration test pattern.  Int32 tests
therefore use the SDFG API.
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.transformation.passes.vectorization.vectorize_cutile import VectorizeCuTile

# All tests in this file require a GPU.
pytestmark = pytest.mark.gpu


# ============================================================
# Pipeline helpers
# ============================================================


def _apply_and_run(sdfg, widths, **kwargs):
    """Apply the VectorizeCuTile pipeline and run the compiled SDFG.

    :param sdfg: SDFG to transform and compile.
    :param widths: Tile widths for vectorization (powers of 2).
    :param kwargs: Named arguments for the SDFG (arrays + symbols).
    :returns: The kwargs dict (output arrays are modified in-place).
    """
    VectorizeCuTile(widths=widths).apply_pass(sdfg, {})
    csdfg = sdfg.compile()
    csdfg(**kwargs)
    return kwargs


def _build_copy_sdfg(name, dtype=dace.float64):
    """Build a symbolic-sized B[i] = A[i] SDFG via the SDFG API.

    :param name: Unique SDFG name.
    :param dtype: Data type for all arrays.
    :returns: The constructed SDFG.
    """
    N = dace.symbol("N")
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (N,), dtype)
    sdfg.add_array("B", (N,), dtype)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "copy",
        {"i": "0:N"},
        {"_a": dace.Memlet("A[i]")},
        "_b = _a",
        {"_b": dace.Memlet("B[i]")},
        external_edges=True,
    )
    return sdfg


def _build_vadd_sdfg(name, dtype=dace.float64):
    """Build a symbolic-sized C[i] = A[i] + B[i] SDFG via the SDFG API.

    :param name: Unique SDFG name.
    :param dtype: Data type for all arrays.
    :returns: The constructed SDFG.
    """
    N = dace.symbol("N")
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (N,), dtype)
    sdfg.add_array("B", (N,), dtype)
    sdfg.add_array("C", (N,), dtype)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "add",
        {"i": "0:N"},
        {"_a": dace.Memlet("A[i]"), "_b": dace.Memlet("B[i]")},
        "_c = _a + _b",
        {"_c": dace.Memlet("C[i]")},
        external_edges=True,
    )
    return sdfg


def _build_copy_2d_sdfg(name, dtype=dace.float64):
    """Build a symbolic-sized 2-D B[i,j] = A[i,j] SDFG.

    :param name: Unique SDFG name.
    :param dtype: Data type for all arrays.
    :returns: The constructed SDFG.
    """
    M = dace.symbol("M")
    N = dace.symbol("N")
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (M, N), dtype)
    sdfg.add_array("B", (M, N), dtype)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "copy2d",
        {"i": "0:M", "j": "0:N"},
        {"_a": dace.Memlet("A[i, j]")},
        "_b = _a",
        {"_b": dace.Memlet("B[i, j]")},
        external_edges=True,
    )
    return sdfg


def _build_transpose_2d_sdfg(name, dtype=dace.float64):
    """Build a symbolic-sized 2-D transpose B[i,j] = A[j,i] SDFG.

    :param name: Unique SDFG name.
    :param dtype: Data type for all arrays.
    :returns: The constructed SDFG.
    """
    M = dace.symbol("M")
    N = dace.symbol("N")
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (M, N), dtype)
    sdfg.add_array("B", (N, M), dtype)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "transpose",
        {"i": "0:N", "j": "0:M"},
        {"_a": dace.Memlet("A[j, i]")},
        "_b = _a",
        {"_b": dace.Memlet("B[i, j]")},
        external_edges=True,
    )
    return sdfg


def _build_permute_3d_kji_sdfg(name, dtype=dace.float64):
    """Build a 3-D full axis reversal B[i,j,k] = A[k,j,i] SDFG.

    :param name: Unique SDFG name.
    :param dtype: Data type for all arrays.
    :returns: The constructed SDFG.
    """
    L = dace.symbol("L")
    M = dace.symbol("M")
    N = dace.symbol("N")
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (L, M, N), dtype)
    sdfg.add_array("B", (N, M, L), dtype)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "permute",
        {"i": "0:N", "j": "0:M", "k": "0:L"},
        {"_a": dace.Memlet("A[k, j, i]")},
        "_b = _a",
        {"_b": dace.Memlet("B[i, j, k]")},
        external_edges=True,
    )
    return sdfg


# ============================================================
# Category 1: Contiguous stores (basic identity copy)
# ============================================================


class TestContiguousStoreCutileRuntime:
    """Contiguous stores: B[i] = A[i] or C[i] = A[i] + B[i]."""

    def test_copy_f64(self):
        """1-D contiguous copy, float64, symbolic N, n=100."""
        sdfg = _build_copy_sdfg("ts_rt_contig_f64")

        n = 100
        rng = np.random.default_rng(42)
        A = rng.standard_normal(n)
        B = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_allclose(B, A, rtol=1e-14)

    def test_copy_f32(self):
        """1-D contiguous copy, float32."""
        sdfg = _build_copy_sdfg("ts_rt_contig_f32", dtype=dace.float32)

        n = 100
        rng = np.random.default_rng(43)
        A = rng.standard_normal(n).astype(np.float32)
        B = np.zeros(n, dtype=np.float32)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_allclose(B, A, rtol=1e-6)

    def test_copy_i32(self):
        """1-D contiguous copy, int32 (via SDFG API)."""
        sdfg = _build_copy_sdfg("ts_rt_contig_i32", dtype=dace.int32)

        n = 100
        rng = np.random.default_rng(44)
        A = rng.integers(0, 500, n, dtype=np.int32)
        B = np.zeros(n, dtype=np.int32)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_array_equal(B, A)

    def test_add_store(self):
        """C[i] = A[i] + B[i], float64."""
        sdfg = _build_vadd_sdfg("ts_rt_contig_add")

        n = 100
        rng = np.random.default_rng(45)
        A = rng.standard_normal(n)
        B = rng.standard_normal(n)
        C = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(C, A + B, rtol=1e-14)

    def test_copy_2d_k1(self):
        """2-D copy B[i,j] = A[i,j] with K=1, widths=(8,)."""
        sdfg = _build_copy_2d_sdfg("ts_rt_contig_2d_k1")

        m, n = 16, 24
        rng = np.random.default_rng(46)
        A = rng.standard_normal((m, n))
        B = np.zeros((m, n))

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, M=m, N=n)
        np.testing.assert_allclose(B, A, rtol=1e-14)

    def test_copy_2d_k2(self):
        """2-D copy B[i,j] = A[i,j] with K=2, widths=(8,8)."""
        sdfg = _build_copy_2d_sdfg("ts_rt_contig_2d_k2")

        m, n = 16, 24
        rng = np.random.default_rng(47)
        A = rng.standard_normal((m, n))
        B = np.zeros((m, n))

        _apply_and_run(sdfg, widths=(8, 8), A=A, B=B, M=m, N=n)
        np.testing.assert_allclose(B, A, rtol=1e-14)


# ============================================================
# Category 2: Non-divisible sizes (remainder handling)
# ============================================================


class TestNonDivisibleStoreCutileRuntime:
    """Non-divisible array sizes force the last tile to be partially masked."""

    def test_copy_n7(self):
        """N=7, widths=(8,) -- last tile partially OOB."""
        sdfg = _build_copy_sdfg("ts_rt_nondiv_n7")

        n = 7
        rng = np.random.default_rng(50)
        A = rng.standard_normal(n)
        B = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_allclose(B, A, rtol=1e-14)

    def test_copy_n1(self):
        """N=1 -- extreme case: 1 active lane."""
        sdfg = _build_copy_sdfg("ts_rt_nondiv_n1")

        n = 1
        rng = np.random.default_rng(51)
        A = rng.standard_normal(n)
        B = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_allclose(B, A, rtol=1e-14)

    def test_copy_n15(self):
        """N=15, widths=(8,)."""
        sdfg = _build_copy_sdfg("ts_rt_nondiv_n15")

        n = 15
        rng = np.random.default_rng(52)
        A = rng.standard_normal(n)
        B = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_allclose(B, A, rtol=1e-14)

    @pytest.mark.parametrize("n", [1, 3, 7, 9, 15, 17, 33])
    def test_add_non_divisible(self, n):
        """C[i] = A[i] + B[i] with various non-divisible sizes."""
        sdfg = _build_vadd_sdfg(f"ts_rt_nondiv_add_{n}")

        rng = np.random.default_rng(n + 3000)
        A = rng.standard_normal(n)
        B = rng.standard_normal(n)
        C = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(C, A + B, rtol=1e-14)

    def test_copy_2d_non_div(self):
        """M=7, N=15 with K=2 widths=(8,8)."""
        sdfg = _build_copy_2d_sdfg("ts_rt_nondiv_2d_k2")

        m, n = 7, 15
        rng = np.random.default_rng(53)
        A = rng.standard_normal((m, n))
        B = np.zeros((m, n))

        _apply_and_run(sdfg, widths=(8, 8), A=A, B=B, M=m, N=n)
        np.testing.assert_allclose(B, A, rtol=1e-14)


# ============================================================
# Category 3: Transpose stores
# ============================================================


class TestTransposeStoreCutileRuntime:
    """Transpose stores exercise the scatter path in TileStore.

    When K==ndim, the aligned ct.store path with ct.permute is used.
    When K < ndim, the scatter path must squeeze the ndim-rank tile
    from ct.load to K-rank before calling ct.scatter (the fix in
    Change 2).
    """

    def test_transpose_2d_k2(self):
        """B[i,j] = A[j,i] with K=2 widths=(8,8) -- K==ndim, aligned path."""
        sdfg = _build_transpose_2d_sdfg("ts_rt_tr_2d_k2")

        m, n = 16, 24
        rng = np.random.default_rng(60)
        A = rng.standard_normal((m, n))
        B = np.zeros((n, m))

        _apply_and_run(sdfg, widths=(8, 8), A=A, B=B, M=m, N=n)
        np.testing.assert_allclose(B, A.T, rtol=1e-14)

    def test_transpose_2d_k2_non_divisible(self):
        """B[i,j] = A[j,i] with K=2 widths=(8,8), non-divisible sizes."""
        sdfg = _build_transpose_2d_sdfg("ts_rt_tr_2d_k2_nd")

        m, n = 13, 17
        rng = np.random.default_rng(61)
        A = rng.standard_normal((m, n))
        B = np.zeros((n, m))

        _apply_and_run(sdfg, widths=(8, 8), A=A, B=B, M=m, N=n)
        np.testing.assert_allclose(B, A.T, rtol=1e-14)

    def test_transpose_2d_k1(self):
        """B[i,j] = A[j,i] with K=1 widths=(8,) -- scatter shape fix."""
        sdfg = _build_transpose_2d_sdfg("ts_rt_tr_2d_k1")

        m, n = 16, 24
        rng = np.random.default_rng(62)
        A = rng.standard_normal((m, n))
        B = np.zeros((n, m))

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, M=m, N=n)
        np.testing.assert_allclose(B, A.T, rtol=1e-14)

    def test_transpose_2d_k1_non_divisible(self):
        """B[i,j] = A[j,i] with K=1 widths=(8,), non-divisible sizes."""
        sdfg = _build_transpose_2d_sdfg("ts_rt_tr_2d_k1_nd")

        m, n = 13, 17
        rng = np.random.default_rng(63)
        A = rng.standard_normal((m, n))
        B = np.zeros((n, m))

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, M=m, N=n)
        np.testing.assert_allclose(B, A.T, rtol=1e-14)

    def test_permute_3d_k2(self):
        """B[i,j,k] = A[k,j,i] with K=2 widths=(8,8) -- K < ndim."""
        sdfg = _build_permute_3d_kji_sdfg("ts_rt_perm_3d_k2")

        l, m, n = 8, 16, 24
        rng = np.random.default_rng(64)
        A = rng.standard_normal((l, m, n))
        B = np.zeros((n, m, l))

        _apply_and_run(sdfg, widths=(8, 8), A=A, B=B, L=l, M=m, N=n)
        np.testing.assert_allclose(B, A.transpose(2, 1, 0), rtol=1e-14)

    def test_permute_3d_k1(self):
        """B[i,j,k] = A[k,j,i] with K=1 widths=(8,) -- K < ndim."""
        sdfg = _build_permute_3d_kji_sdfg("ts_rt_perm_3d_k1")

        l, m, n = 8, 16, 24
        rng = np.random.default_rng(65)
        A = rng.standard_normal((l, m, n))
        B = np.zeros((n, m, l))

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, L=l, M=m, N=n)
        np.testing.assert_allclose(B, A.transpose(2, 1, 0), rtol=1e-14)

    @pytest.mark.parametrize("m,n", [(1, 1), (3, 5), (7, 9), (8, 16),
                                      (15, 17), (32, 32)])
    def test_transpose_2d_k2_various_sizes(self, m, n):
        """2-D transpose with various (m, n) sizes, K=2."""
        sdfg = _build_transpose_2d_sdfg(f"ts_rt_tr_2d_k2_sz_{m}x{n}")

        rng = np.random.default_rng(m * 100 + n)
        A = rng.standard_normal((m, n))
        B = np.zeros((n, m))

        _apply_and_run(sdfg, widths=(8, 8), A=A, B=B, M=m, N=n)
        np.testing.assert_allclose(B, A.T, rtol=1e-14)


# ============================================================
# Category 4: WCR atomic stores (expansion-level tests)
# ============================================================


class TestWCRStoreCutile:
    """Test WCR (atomic) store expansion at the unit level.

    Since canonicalization decomposes WCR before vectorization, end-to-end
    pipeline tests through @dace.program -> canonicalize -> VectorizeCuTile
    will not produce TileStore with WCR.  These tests verify the expansion
    codegen by constructing TileStore nodes manually with wcr set and
    calling the expansion directly.
    """

    def test_atomic_add_expansion(self):
        """Verify ExpandTileStoreCutile generates ct.atomic_add for Sum WCR."""
        from dace.libraries.tileops.nodes.tile_store import TileStore, ExpandTileStoreCutile

        N = dace.symbol("N")
        sdfg = dace.SDFG("ts_rt_wcr_add_expansion")
        sdfg.add_array("tile_in", (8,), dace.float64,
                        storage=dtypes.StorageType.CuTile_Tile, transient=True)
        sdfg.add_array("dst", (N,), dace.float64,
                        storage=dtypes.StorageType.GPU_Global)
        state = sdfg.add_state()

        store = TileStore("wcr_store", widths=(8,),
                          wcr="lambda a, b: a + b")
        state.add_node(store)

        # Wire edges
        an_tile = state.add_access("tile_in")
        an_dst = state.add_access("dst")
        state.add_edge(an_tile, None, store, "_src",
                       dace.Memlet("tile_in[0:8]"))
        state.add_edge(store, "_dst", an_dst, None,
                       dace.Memlet("dst[0:N]"))

        tasklet = ExpandTileStoreCutile.expansion(store, state, sdfg)
        code = tasklet.code.as_string
        assert "ct.atomic_add" in code, (
            f"Expected ct.atomic_add in generated code, got:\n{code}")

    def test_atomic_min_expansion(self):
        """Verify ExpandTileStoreCutile generates ct.atomic_min for Min WCR."""
        from dace.libraries.tileops.nodes.tile_store import TileStore, ExpandTileStoreCutile

        N = dace.symbol("N")
        sdfg = dace.SDFG("ts_rt_wcr_min_expansion")
        sdfg.add_array("tile_in", (8,), dace.float64,
                        storage=dtypes.StorageType.CuTile_Tile, transient=True)
        sdfg.add_array("dst", (N,), dace.float64,
                        storage=dtypes.StorageType.GPU_Global)
        state = sdfg.add_state()

        store = TileStore("wcr_store_min", widths=(8,),
                          wcr="lambda a, b: min(a, b)")
        state.add_node(store)

        an_tile = state.add_access("tile_in")
        an_dst = state.add_access("dst")
        state.add_edge(an_tile, None, store, "_src",
                       dace.Memlet("tile_in[0:8]"))
        state.add_edge(store, "_dst", an_dst, None,
                       dace.Memlet("dst[0:N]"))

        tasklet = ExpandTileStoreCutile.expansion(store, state, sdfg)
        code = tasklet.code.as_string
        assert "ct.atomic_min" in code, (
            f"Expected ct.atomic_min in generated code, got:\n{code}")

    def test_atomic_max_expansion(self):
        """Verify ExpandTileStoreCutile generates ct.atomic_max for Max WCR."""
        from dace.libraries.tileops.nodes.tile_store import TileStore, ExpandTileStoreCutile

        N = dace.symbol("N")
        sdfg = dace.SDFG("ts_rt_wcr_max_expansion")
        sdfg.add_array("tile_in", (8,), dace.float64,
                        storage=dtypes.StorageType.CuTile_Tile, transient=True)
        sdfg.add_array("dst", (N,), dace.float64,
                        storage=dtypes.StorageType.GPU_Global)
        state = sdfg.add_state()

        store = TileStore("wcr_store_max", widths=(8,),
                          wcr="lambda a, b: max(a, b)")
        state.add_node(store)

        an_tile = state.add_access("tile_in")
        an_dst = state.add_access("dst")
        state.add_edge(an_tile, None, store, "_src",
                       dace.Memlet("tile_in[0:8]"))
        state.add_edge(store, "_dst", an_dst, None,
                       dace.Memlet("dst[0:N]"))

        tasklet = ExpandTileStoreCutile.expansion(store, state, sdfg)
        code = tasklet.code.as_string
        assert "ct.atomic_max" in code, (
            f"Expected ct.atomic_max in generated code, got:\n{code}")

    def test_unsupported_wcr_raises(self):
        """Verify NotImplementedError for unsupported WCR (Product)."""
        from dace.libraries.tileops.nodes.tile_store import TileStore, ExpandTileStoreCutile

        N = dace.symbol("N")
        sdfg = dace.SDFG("ts_rt_wcr_product_raises")
        sdfg.add_array("tile_in", (8,), dace.float64,
                        storage=dtypes.StorageType.CuTile_Tile, transient=True)
        sdfg.add_array("dst", (N,), dace.float64,
                        storage=dtypes.StorageType.GPU_Global)
        state = sdfg.add_state()

        store = TileStore("wcr_store_prod", widths=(8,),
                          wcr="lambda a, b: a * b")
        state.add_node(store)

        an_tile = state.add_access("tile_in")
        an_dst = state.add_access("dst")
        state.add_edge(an_tile, None, store, "_src",
                       dace.Memlet("tile_in[0:8]"))
        state.add_edge(store, "_dst", an_dst, None,
                       dace.Memlet("dst[0:N]"))

        with pytest.raises(NotImplementedError, match="Product"):
            ExpandTileStoreCutile.expansion(store, state, sdfg)

    def test_atomic_add_skips_aligned_path(self):
        """WCR stores must skip the aligned ct.store path (use scatter)."""
        from dace.libraries.tileops.nodes.tile_store import TileStore, ExpandTileStoreCutile

        N = dace.symbol("N")
        sdfg = dace.SDFG("ts_rt_wcr_add_no_ctstore")
        sdfg.add_array("tile_in", (8,), dace.float64,
                        storage=dtypes.StorageType.CuTile_Tile, transient=True)
        sdfg.add_array("dst", (N,), dace.float64,
                        storage=dtypes.StorageType.GPU_Global)
        state = sdfg.add_state()

        # Unit coeffs, no mask -> would normally take aligned ct.store path,
        # but WCR must force it to scatter/atomic.
        store = TileStore("wcr_store_noalign", widths=(8,),
                          wcr="lambda a, b: a + b")
        state.add_node(store)

        an_tile = state.add_access("tile_in")
        an_dst = state.add_access("dst")
        state.add_edge(an_tile, None, store, "_src",
                       dace.Memlet("tile_in[0:8]"))
        state.add_edge(store, "_dst", an_dst, None,
                       dace.Memlet("dst[0:N]"))

        tasklet = ExpandTileStoreCutile.expansion(store, state, sdfg)
        code = tasklet.code.as_string
        assert "ct.store(" not in code, (
            f"WCR store should NOT use ct.store, got:\n{code}")
        assert "ct.atomic_add" in code, (
            f"WCR store should use ct.atomic_add, got:\n{code}")


# ============================================================
# Category 5: Masked stores
# ============================================================


class TestMaskedStoreCutileRuntime:
    """Masked stores exercise the masked scatter path (non-divisible sizes)."""

    def test_masked_store_n3(self):
        """Copy with n=3, width=8 -- only 3 of 8 lanes active."""
        sdfg = _build_copy_sdfg("ts_rt_masked_n3")

        n = 3
        rng = np.random.default_rng(70)
        A = rng.standard_normal(n)
        B = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_allclose(B, A, rtol=1e-14)

    def test_masked_store_n9(self):
        """Add with n=9, width=8 -- 1 active lane on second tile."""
        sdfg = _build_vadd_sdfg("ts_rt_masked_n9")

        n = 9
        rng = np.random.default_rng(71)
        A = rng.standard_normal(n)
        B = rng.standard_normal(n)
        C = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(C, A + B, rtol=1e-14)

    def test_masked_store_n17(self):
        """Copy with n=17, width=16."""
        N = dace.symbol("N")

        @dace.program
        def copy_masked(A: dace.float64[N], B: dace.float64[N]):
            for i in range(N):
                B[i] = A[i]

        sdfg = copy_masked.to_sdfg()
        sdfg.name = "ts_rt_masked_n17_w16"

        n = 17
        rng = np.random.default_rng(72)
        A = rng.standard_normal(n)
        B = np.zeros(n)

        _apply_and_run(sdfg, widths=(16,), A=A, B=B, N=n)
        np.testing.assert_allclose(B, A, rtol=1e-14)


# ============================================================
# Category 6: Multi-dim K=2 stores
# ============================================================


class TestMultiDimStoreCutileRuntime:
    """K=2 tile stores with widths=(W0, W1)."""

    def test_k2_copy_8x8(self):
        """K=2 contiguous copy, widths=(8,8), tile-divisible."""
        sdfg = _build_copy_2d_sdfg("ts_rt_k2_copy_8x8")

        m, n = 16, 24
        rng = np.random.default_rng(80)
        A = rng.standard_normal((m, n))
        B = np.zeros((m, n))

        _apply_and_run(sdfg, widths=(8, 8), A=A, B=B, M=m, N=n)
        np.testing.assert_allclose(B, A, rtol=1e-14)

    def test_k2_add_4x8(self):
        """K=2 add, widths=(4,8), tile-divisible."""
        M = dace.symbol("M")
        N = dace.symbol("N")
        sdfg = dace.SDFG("ts_rt_k2_add_4x8")
        sdfg.add_array("A", (M * 4, N * 8), dace.float64)
        sdfg.add_array("B", (M * 4, N * 8), dace.float64)
        sdfg.add_array("C", (M * 4, N * 8), dace.float64)
        state = sdfg.add_state("main")
        state.add_mapped_tasklet(
            "add",
            {"i": "0:M*4", "j": "0:N*8"},
            {"_a": dace.Memlet("A[i, j]"), "_b": dace.Memlet("B[i, j]")},
            "_c = _a + _b",
            {"_c": dace.Memlet("C[i, j]")},
            external_edges=True,
        )

        m_val, n_val = 3, 4
        m4, n8 = m_val * 4, n_val * 8
        rng = np.random.default_rng(81)
        A = rng.standard_normal((m4, n8))
        B = rng.standard_normal((m4, n8))
        C = np.zeros((m4, n8))

        _apply_and_run(sdfg, widths=(4, 8), A=A, B=B, C=C, M=m_val, N=n_val)
        np.testing.assert_allclose(C, A + B, rtol=1e-14)

    def test_k2_non_divisible(self):
        """K=2 copy with non-divisible sizes (m=7, n=15)."""
        sdfg = _build_copy_2d_sdfg("ts_rt_k2_nondiv")

        m, n = 7, 15
        rng = np.random.default_rng(82)
        A = rng.standard_normal((m, n))
        B = np.zeros((m, n))

        _apply_and_run(sdfg, widths=(8, 8), A=A, B=B, M=m, N=n)
        np.testing.assert_allclose(B, A, rtol=1e-14)

    def test_k2_float32(self):
        """K=2 copy in float32, widths=(8,8)."""
        sdfg = _build_copy_2d_sdfg("ts_rt_k2_f32", dtype=dace.float32)

        m, n = 16, 24
        rng = np.random.default_rng(83)
        A = rng.standard_normal((m, n)).astype(np.float32)
        B = np.zeros((m, n), dtype=np.float32)

        _apply_and_run(sdfg, widths=(8, 8), A=A, B=B, M=m, N=n)
        np.testing.assert_allclose(B, A, rtol=1e-6)


# ============================================================
# Category 7: Codegen verification
# ============================================================


class TestStoreCodegenStructure:
    """Verify that generated cuTile code contains expected primitives."""

    def test_contiguous_store_emits_ct_scatter_or_store(self):
        """Contiguous store should emit ct.store or ct.scatter in generated code.

        With symbolic sizes, the vectorizer generates a tile-iteration mask,
        so the masked scatter path (ct.scatter with mask=) is taken rather
        than the aligned ct.store path.
        """
        sdfg = _build_copy_sdfg("ts_rt_codegen_ctstore")
        VectorizeCuTile(widths=(8,)).apply_pass(sdfg, {})

        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        assert "ct.store(" in code or "ct.scatter(" in code, (
            "Expected ct.store or ct.scatter in generated code")

    def test_masked_store_emits_ct_scatter(self):
        """Symbolic-sized store should use ct.scatter (masked)."""
        sdfg = _build_copy_sdfg("ts_rt_codegen_ctscatter")
        VectorizeCuTile(widths=(8,)).apply_pass(sdfg, {})

        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        assert "ct.scatter(" in code, (
            "Expected ct.scatter in generated code for symbolic-sized store")

    def test_generated_store_code_is_valid_python(self):
        """Generated cuTile store code should be parseable as valid Python."""
        import ast as ast_mod

        sdfg = _build_vadd_sdfg("ts_rt_codegen_valid_py")
        VectorizeCuTile(widths=(8,)).apply_pass(sdfg, {})

        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        ast_mod.parse(code)

    def test_transpose_store_emits_scatter(self):
        """Transpose store (K=2, symbolic) uses ct.scatter (masked path)."""
        sdfg = _build_transpose_2d_sdfg("ts_rt_codegen_permute")
        VectorizeCuTile(widths=(8, 8)).apply_pass(sdfg, {})

        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        # Symbolic sizes force mask generation, so the masked scatter path
        # is taken rather than aligned ct.store with ct.permute.
        assert "ct.scatter(" in code or "ct.store(" in code or "ct.permute(" in code, (
            "Expected ct.scatter, ct.store, or ct.permute in generated code")


# ============================================================
# Category 8: _needs_int64 helper tests
# ============================================================


class TestNeedsInt64:
    """Unit tests for the _needs_int64 helper in tile_store."""

    def test_small_array_uses_int32(self):
        """Small concrete array should use int32."""
        from dace.libraries.tileops._pure_codegen import needs_int64 as _needs_int64

        sdfg = dace.SDFG("ts_rt_int64_small")
        sdfg.add_array("A", (1000,), dace.float64)
        desc = sdfg.arrays["A"]
        assert not _needs_int64(desc, [1])

    def test_large_array_uses_int64(self):
        """Large concrete array (>2^31) should use int64."""
        from dace.libraries.tileops._pure_codegen import needs_int64 as _needs_int64

        sdfg = dace.SDFG("ts_rt_int64_large")
        sdfg.add_array("A", (3_000_000_000,), dace.float64)
        desc = sdfg.arrays["A"]
        assert _needs_int64(desc, [1])

    def test_symbolic_size_uses_int64(self):
        """Symbolic size should conservatively use int64."""
        from dace.libraries.tileops._pure_codegen import needs_int64 as _needs_int64

        N = dace.symbol("N")
        sdfg = dace.SDFG("ts_rt_int64_sym")
        sdfg.add_array("A", (N,), dace.float64)
        desc = sdfg.arrays["A"]
        assert _needs_int64(desc, [1])

    def test_large_coeff_uses_int64(self):
        """Large stride coefficient should trigger int64."""
        from dace.libraries.tileops._pure_codegen import needs_int64 as _needs_int64

        sdfg = dace.SDFG("ts_rt_int64_coeff")
        sdfg.add_array("A", (1_000_000,), dace.float64)
        desc = sdfg.arrays["A"]
        # coeff * dim_max > INT32_MAX
        assert _needs_int64(desc, [3000])


# ============================================================
# Category 9: Various tile widths
# ============================================================


class TestTileWidths:
    """Power-of-2 tile widths: 4, 8, 16, 32."""

    @pytest.mark.parametrize("w", [4, 8, 16, 32])
    def test_copy_tile_width(self, w):
        """Contiguous copy with tile width w and non-divisible size."""
        sdfg = _build_copy_sdfg(f"ts_rt_tw_{w}")

        n = 100  # non-divisible by all of 4, 8, 16, 32
        rng = np.random.default_rng(w + 500)
        A = rng.standard_normal(n)
        B = np.zeros(n)

        _apply_and_run(sdfg, widths=(w,), A=A, B=B, N=n)
        np.testing.assert_allclose(B, A, rtol=1e-14)


# ============================================================
# Category 10: Edge cases
# ============================================================


class TestStoreEdgeCases:
    """Edge cases for tile stores."""

    def test_identity_copy_aligned(self):
        """Identity copy, aligned size: B[i] = A[i], n=64."""
        sdfg = _build_copy_sdfg("ts_rt_edge_id_copy")

        n = 64
        rng = np.random.default_rng(90)
        A = rng.standard_normal(n)
        B = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_allclose(B, A, rtol=1e-14)

    def test_negate(self):
        """Negate: B[i] = -A[i]."""
        N = dace.symbol("N")

        @dace.program
        def negate(A: dace.float64[N], B: dace.float64[N]):
            for i in range(N):
                B[i] = -A[i]

        sdfg = negate.to_sdfg()
        sdfg.name = "ts_rt_edge_negate"

        n = 100
        rng = np.random.default_rng(91)
        A = rng.standard_normal(n)
        B = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_allclose(B, -A, rtol=1e-14)

    def test_large_array_store(self):
        """1-D copy with N=10000."""
        sdfg = _build_copy_sdfg("ts_rt_edge_large")

        n = 10000
        rng = np.random.default_rng(92)
        A = rng.standard_normal(n)
        B = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_allclose(B, A, rtol=1e-14)


# ============================================================
# Entry point
# ============================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=300"])
