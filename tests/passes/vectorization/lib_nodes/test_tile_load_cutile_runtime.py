# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU runtime tests for TileLoad's cuTile expansion.

Each test goes through the FULL pipeline:
``@dace.program`` (or SDFG API) -> ``VectorizeCuTile(widths=...).apply_pass(sdfg, {})``
-> compile -> run with NumPy arrays -> compare against NumPy reference.

The ``VectorizeCuTile`` orchestrator detects load access patterns
(contiguous, strided, gather) and emits ``TileLoad`` library nodes
with ``'cutile'`` implementations, which are expanded into ``ct.load``
/ ``ct.gather`` calls for the Python backend.  These tests verify
end-to-end correctness of those loads across dtypes, tile widths,
array sizes (including non-divisible boundaries), 1-D and 2-D arrays,
gather (indirect) accesses, scalar broadcast loads, and multi-input
kernels that exercise multiple loads per kernel.

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


def _build_vadd2d_sdfg(name, dtype=dace.float64):
    """Build a symbolic-sized 2-D C[i,j] = A[i,j] + B[i,j] SDFG.

    Array shapes are ``(M*W0, N*W1)`` so tile-divisible at runtime.

    :param name: Unique SDFG name.
    :param dtype: Data type for all arrays.
    :returns: The constructed SDFG.
    """
    M = dace.symbol("M")
    N = dace.symbol("N")
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (M * 8, N * 8), dtype)
    sdfg.add_array("B", (M * 8, N * 8), dtype)
    sdfg.add_array("C", (M * 8, N * 8), dtype)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "add2d",
        {"i": "0:M*8", "j": "0:N*8"},
        {"_a": dace.Memlet("A[i, j]"), "_b": dace.Memlet("B[i, j]")},
        "_c = _a + _b",
        {"_c": dace.Memlet("C[i, j]")},
        external_edges=True,
    )
    return sdfg


# ============================================================
# Category 1: Contiguous Aligned Loads (basics)
# ============================================================


class TestContiguousLoads:
    """Contiguous loads: ``B[i] = A[i]`` or ``C[i,j] = A[i,j] + B[i,j]``."""

    def test_contiguous_1d_float64(self):
        """1-D contiguous copy, float64, symbolic N, n=100 (non-divisible by 8)."""
        N = dace.symbol("N")

        @dace.program
        def copy_f64(A: dace.float64[N], B: dace.float64[N]):
            for i in range(N):
                B[i] = A[i]

        sdfg = copy_f64.to_sdfg()
        sdfg.name = "tl_rt_contig_1d_f64"

        n = 100
        rng = np.random.default_rng(42)
        A = rng.standard_normal(n)
        B = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_allclose(B, A, rtol=1e-14)

    def test_contiguous_1d_float32(self):
        """1-D contiguous copy, float32."""
        N = dace.symbol("N")

        @dace.program
        def copy_f32(A: dace.float32[N], B: dace.float32[N]):
            for i in range(N):
                B[i] = A[i]

        sdfg = copy_f32.to_sdfg()
        sdfg.name = "tl_rt_contig_1d_f32"

        n = 100
        rng = np.random.default_rng(43)
        A = rng.standard_normal(n).astype(np.float32)
        B = np.zeros(n, dtype=np.float32)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_allclose(B, A, rtol=1e-6)

    def test_contiguous_1d_int32(self):
        """1-D contiguous add, int32 (via SDFG API for cuTile compatibility)."""
        sdfg = _build_vadd_sdfg("tl_rt_contig_1d_i32", dtype=dace.int32)

        n = 100
        rng = np.random.default_rng(44)
        A = rng.integers(0, 500, n, dtype=np.int32)
        B = rng.integers(0, 500, n, dtype=np.int32)
        C = np.zeros(n, dtype=np.int32)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, C=C, N=n)
        np.testing.assert_array_equal(C, A + B)

    def test_contiguous_2d_k2(self):
        """2-D contiguous add, K=2 vectorization, widths=(8,8)."""
        sdfg = _build_vadd2d_sdfg("tl_rt_contig_2d_k2")

        m_val, n_val = 3, 4
        m8, n8 = m_val * 8, n_val * 8
        rng = np.random.default_rng(45)
        A = rng.standard_normal((m8, n8))
        B = rng.standard_normal((m8, n8))
        C = np.zeros((m8, n8))

        _apply_and_run(sdfg, widths=(8, 8), A=A, B=B, C=C, M=m_val, N=n_val)
        np.testing.assert_allclose(C, A + B, rtol=1e-14)

    @pytest.mark.parametrize("n", [1, 3, 7, 9, 15, 17, 31, 33, 63, 65, 100, 127])
    def test_contiguous_non_divisible_sizes(self, n):
        """Contiguous add with various non-divisible sizes (remainder masking)."""
        N = dace.symbol("N")

        @dace.program
        def add_1d(A: dace.float64[N], B: dace.float64[N],
                   C: dace.float64[N]):
            for i in range(N):
                C[i] = A[i] + B[i]

        sdfg = add_1d.to_sdfg()
        sdfg.name = f"tl_rt_contig_nondiv_{n}"

        rng = np.random.default_rng(n + 1000)
        A = rng.standard_normal(n)
        B = rng.standard_normal(n)
        C = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(C, A + B, rtol=1e-14)


# ============================================================
# Category 2: Strided Loads
# ============================================================


class TestStridedLoads:
    """Strided load access patterns: ``B[i] = A[2*i]``.

    The cuTile runtime does not support tile indexing (``ct.TileTypeError:
    Directly indexing a tile is not supported``), so strided loads through
    the VectorizeCuTile pipeline are expected to fail at runtime.  These
    tests document the current limitation.
    """

    @pytest.mark.xfail(
        reason="cuTile runtime: strided tile indexing not supported",
        strict=True,
    )
    def test_strided_1d_stride2(self):
        """1-D stride-2 load: ``B[i] = A[2*i]``."""
        N = dace.symbol("N")

        @dace.program
        def strided_load(A: dace.float64[2 * N], B: dace.float64[N]):
            for i in range(N):
                B[i] = A[2 * i]

        sdfg = strided_load.to_sdfg()
        sdfg.name = "tl_rt_stride2_1d"

        n = 100
        rng = np.random.default_rng(50)
        A = rng.standard_normal(2 * n)
        B = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_allclose(B, A[::2], rtol=1e-14)

    def test_strided_2d_stride2(self):
        """2-D stride-2 load on first dimension: ``C[i,j] = A[2*i, j]``."""
        M = dace.symbol("M")
        N = dace.symbol("N")

        @dace.program
        def strided_2d(A: dace.float64[2 * M, N], C: dace.float64[M, N]):
            for i in range(M):
                for j in range(N):
                    C[i, j] = A[2 * i, j]

        sdfg = strided_2d.to_sdfg()
        sdfg.name = "tl_rt_stride2_2d"

        m, n = 10, 17
        rng = np.random.default_rng(51)
        A = rng.standard_normal((2 * m, n))
        C = np.zeros((m, n))

        _apply_and_run(sdfg, widths=(8,), A=A, C=C, M=m, N=n)
        np.testing.assert_allclose(C, A[::2, :], rtol=1e-14)


# ============================================================
# Category 3: Indirect Gather Loads
# ============================================================


class TestGatherLoads:
    """Indirect gather loads: ``B[i] = A[idx[i]]``."""

    def test_gather_1d(self):
        """1-D gather via random permutation indices."""
        N = dace.symbol("N")

        @dace.program
        def gather_1d(A: dace.float64[N], idx: dace.int32[N],
                      B: dace.float64[N]):
            for i in range(N):
                B[i] = A[idx[i]]

        sdfg = gather_1d.to_sdfg()
        sdfg.name = "tl_rt_gather_1d"

        n = 100
        rng = np.random.default_rng(60)
        A = rng.standard_normal(n)
        idx = rng.permutation(n).astype(np.int32)
        B = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, idx=idx, B=B, N=n)
        np.testing.assert_allclose(B, A[idx], rtol=1e-14)

    def test_gather_1d_with_computation(self):
        """1-D gather with a multiply: ``B[i] = A[idx[i]] * 2.0``."""
        N = dace.symbol("N")

        @dace.program
        def gather_mul(A: dace.float64[N], idx: dace.int32[N],
                       B: dace.float64[N]):
            for i in range(N):
                B[i] = A[idx[i]] * 2.0

        sdfg = gather_mul.to_sdfg()
        sdfg.name = "tl_rt_gather_mul"

        n = 64
        rng = np.random.default_rng(61)
        A = rng.standard_normal(n)
        idx = rng.permutation(n).astype(np.int32)
        B = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, idx=idx, B=B, N=n)
        np.testing.assert_allclose(B, A[idx] * 2.0, rtol=1e-14)

    def test_gather_1d_non_divisible(self):
        """1-D gather with n=17 (non-divisible by 8) -- tests masked gather."""
        N = dace.symbol("N")

        @dace.program
        def gather_nd(A: dace.float64[N], idx: dace.int32[N],
                      B: dace.float64[N]):
            for i in range(N):
                B[i] = A[idx[i]]

        sdfg = gather_nd.to_sdfg()
        sdfg.name = "tl_rt_gather_nd"

        n = 17
        rng = np.random.default_rng(62)
        A = rng.standard_normal(n)
        idx = rng.permutation(n).astype(np.int32)
        B = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, idx=idx, B=B, N=n)
        np.testing.assert_allclose(B, A[idx], rtol=1e-14)

    def test_gather_1d_aligned(self):
        """1-D gather with n=64 (aligned to 8)."""
        N = dace.symbol("N")

        @dace.program
        def gather_al(A: dace.float64[N], idx: dace.int32[N],
                      B: dace.float64[N]):
            for i in range(N):
                B[i] = A[idx[i]]

        sdfg = gather_al.to_sdfg()
        sdfg.name = "tl_rt_gather_aligned"

        n = 64
        rng = np.random.default_rng(63)
        A = rng.standard_normal(n)
        idx = rng.permutation(n).astype(np.int32)
        B = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, idx=idx, B=B, N=n)
        np.testing.assert_allclose(B, A[idx], rtol=1e-14)


# ============================================================
# Category 4: Masked Remainder via Non-divisible Boundaries
# ============================================================


class TestMaskedRemainder:
    """Tests that exercise the tile load mask for remainder handling.

    Non-divisible array sizes force the last tile to be partially masked.
    The load padding mode ensures out-of-bounds lanes read the correct
    identity value.
    """

    def test_masked_remainder_copy_small(self):
        """Copy with n=3, width=8 -- only 3 of 8 lanes are active on the single tile."""
        N = dace.symbol("N")

        @dace.program
        def copy_small(A: dace.float64[N], B: dace.float64[N]):
            for i in range(N):
                B[i] = A[i]

        sdfg = copy_small.to_sdfg()
        sdfg.name = "tl_rt_masked_copy_n3"

        n = 3
        rng = np.random.default_rng(70)
        A = rng.standard_normal(n)
        B = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_allclose(B, A, rtol=1e-14)

    def test_masked_remainder_add_n1(self):
        """Add with n=1, width=8 -- extreme case: 1 active lane."""
        N = dace.symbol("N")

        @dace.program
        def add_n1(A: dace.float64[N], B: dace.float64[N],
                   C: dace.float64[N]):
            for i in range(N):
                C[i] = A[i] + B[i]

        sdfg = add_n1.to_sdfg()
        sdfg.name = "tl_rt_masked_add_n1"

        n = 1
        rng = np.random.default_rng(71)
        A = rng.standard_normal(n)
        B = rng.standard_normal(n)
        C = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(C, A + B, rtol=1e-14)

    def test_masked_remainder_add_n7(self):
        """Add with n=7, width=8 -- 7 of 8 lanes active (one inactive)."""
        N = dace.symbol("N")

        @dace.program
        def add_n7(A: dace.float64[N], B: dace.float64[N],
                   C: dace.float64[N]):
            for i in range(N):
                C[i] = A[i] + B[i]

        sdfg = add_n7.to_sdfg()
        sdfg.name = "tl_rt_masked_add_n7"

        n = 7
        rng = np.random.default_rng(72)
        A = rng.standard_normal(n)
        B = rng.standard_normal(n)
        C = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(C, A + B, rtol=1e-14)

    def test_masked_remainder_add_n15(self):
        """Add with n=15, width=16 -- 15 of 16 lanes active."""
        N = dace.symbol("N")

        @dace.program
        def add_n15(A: dace.float64[N], B: dace.float64[N],
                    C: dace.float64[N]):
            for i in range(N):
                C[i] = A[i] + B[i]

        sdfg = add_n15.to_sdfg()
        sdfg.name = "tl_rt_masked_add_n15_w16"

        n = 15
        rng = np.random.default_rng(73)
        A = rng.standard_normal(n)
        B = rng.standard_normal(n)
        C = np.zeros(n)

        _apply_and_run(sdfg, widths=(16,), A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(C, A + B, rtol=1e-14)


# ============================================================
# Category 5: Multiple dtypes
# ============================================================


class TestDtypes:
    """Elementwise add across different data types."""

    def test_dtype_float32_elementwise(self):
        """float32 elementwise add."""
        sdfg = _build_vadd_sdfg("tl_rt_dtype_f32", dtype=dace.float32)

        n = 100
        rng = np.random.default_rng(80)
        A = rng.standard_normal(n).astype(np.float32)
        B = rng.standard_normal(n).astype(np.float32)
        C = np.zeros(n, dtype=np.float32)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(C, A + B, rtol=1e-6)

    def test_dtype_float64_elementwise(self):
        """float64 elementwise add."""
        sdfg = _build_vadd_sdfg("tl_rt_dtype_f64", dtype=dace.float64)

        n = 100
        rng = np.random.default_rng(81)
        A = rng.standard_normal(n)
        B = rng.standard_normal(n)
        C = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(C, A + B, rtol=1e-14)

    def test_dtype_int32_elementwise(self):
        """int32 elementwise add (via SDFG API for cuTile compatibility)."""
        sdfg = _build_vadd_sdfg("tl_rt_dtype_i32", dtype=dace.int32)

        n = 100
        rng = np.random.default_rng(82)
        A = rng.integers(0, 500, n, dtype=np.int32)
        B = rng.integers(0, 500, n, dtype=np.int32)
        C = np.zeros(n, dtype=np.int32)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, C=C, N=n)
        np.testing.assert_array_equal(C, A + B)

    def test_dtype_float32_non_divisible(self):
        """float32 add, n=17 (non-divisible by 8)."""
        sdfg = _build_vadd_sdfg("tl_rt_dtype_f32_nd", dtype=dace.float32)

        n = 17
        rng = np.random.default_rng(83)
        A = rng.standard_normal(n).astype(np.float32)
        B = rng.standard_normal(n).astype(np.float32)
        C = np.zeros(n, dtype=np.float32)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(C, A + B, rtol=1e-6)

    def test_dtype_int32_non_divisible(self):
        """int32 add, n=17 (non-divisible by 8)."""
        sdfg = _build_vadd_sdfg("tl_rt_dtype_i32_nd", dtype=dace.int32)

        n = 17
        rng = np.random.default_rng(84)
        A = rng.integers(0, 500, n, dtype=np.int32)
        B = rng.integers(0, 500, n, dtype=np.int32)
        C = np.zeros(n, dtype=np.int32)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, C=C, N=n)
        np.testing.assert_array_equal(C, A + B)


# ============================================================
# Category 6: Symbolic sizes
# ============================================================


class TestSymbolicSizes:
    """Symbolic sizes with various concrete instantiations."""

    def test_symbolic_1d(self):
        """Symbolic 1-D size."""
        N = dace.symbol("N")

        @dace.program
        def sym_copy(A: dace.float64[N], B: dace.float64[N]):
            for i in range(N):
                B[i] = A[i]

        sdfg = sym_copy.to_sdfg()
        sdfg.name = "tl_rt_symbolic_1d"

        n = 37
        rng = np.random.default_rng(90)
        A = rng.standard_normal(n)
        B = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_allclose(B, A, rtol=1e-14)

    def test_symbolic_2d_k2(self):
        """Symbolic 2-D sizes with K=2, widths=(8,8)."""
        sdfg = _build_vadd2d_sdfg("tl_rt_symbolic_2d_k2")

        m_val, n_val = 2, 3
        m8, n8 = m_val * 8, n_val * 8
        rng = np.random.default_rng(91)
        A = rng.standard_normal((m8, n8))
        B = rng.standard_normal((m8, n8))
        C = np.zeros((m8, n8))

        _apply_and_run(sdfg, widths=(8, 8), A=A, B=B, C=C, M=m_val, N=n_val)
        np.testing.assert_allclose(C, A + B, rtol=1e-14)

    @pytest.mark.parametrize("n", [8, 37, 64, 100, 128, 255])
    def test_symbolic_various_sizes(self, n):
        """Symbolic 1-D add with various concrete N values."""
        sdfg = _build_vadd_sdfg(f"tl_rt_sym_n{n}")

        rng = np.random.default_rng(n + 2000)
        A = rng.standard_normal(n)
        B = rng.standard_normal(n)
        C = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(C, A + B, rtol=1e-14)


# ============================================================
# Category 7: Various Tile Widths
# ============================================================


class TestTileWidths:
    """Power-of-2 tile widths: 4, 8, 16, 32."""

    @pytest.mark.parametrize("w", [4, 8, 16, 32])
    def test_tile_width(self, w):
        """Contiguous add with tile width w and non-divisible size."""
        N = dace.symbol("N")

        @dace.program
        def add_tw(A: dace.float64[N], B: dace.float64[N],
                   C: dace.float64[N]):
            for i in range(N):
                C[i] = A[i] + B[i]

        sdfg = add_tw.to_sdfg()
        sdfg.name = f"tl_rt_tw_{w}"

        n = 100  # non-divisible by all of 4, 8, 16, 32
        rng = np.random.default_rng(w + 100)
        A = rng.standard_normal(n)
        B = rng.standard_normal(n)
        C = np.zeros(n)

        _apply_and_run(sdfg, widths=(w,), A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(C, A + B, rtol=1e-14)

    @pytest.mark.parametrize("w", [4, 16, 32])
    def test_tile_width_aligned(self, w):
        """Contiguous add with tile width w and aligned size (n=128)."""
        N = dace.symbol("N")

        @dace.program
        def add_tw_aligned(A: dace.float64[N], B: dace.float64[N],
                           C: dace.float64[N]):
            for i in range(N):
                C[i] = A[i] + B[i]

        sdfg = add_tw_aligned.to_sdfg()
        sdfg.name = f"tl_rt_tw_aligned_{w}"

        n = 128
        rng = np.random.default_rng(w + 200)
        A = rng.standard_normal(n)
        B = rng.standard_normal(n)
        C = np.zeros(n)

        _apply_and_run(sdfg, widths=(w,), A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(C, A + B, rtol=1e-14)


# ============================================================
# Category 8: Larger / Stress Tests
# ============================================================


class TestLargeArrays:
    """Larger array sizes for stress testing."""

    def test_large_array_1d(self):
        """1-D contiguous add with N=10000."""
        N = dace.symbol("N")

        @dace.program
        def add_large(A: dace.float64[N], B: dace.float64[N],
                      C: dace.float64[N]):
            for i in range(N):
                C[i] = A[i] + B[i]

        sdfg = add_large.to_sdfg()
        sdfg.name = "tl_rt_large_1d"

        n = 10000
        rng = np.random.default_rng(110)
        A = rng.standard_normal(n)
        B = rng.standard_normal(n)
        C = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(C, A + B, rtol=1e-14)

    def test_large_2d_k2(self):
        """2-D contiguous add with K=2, 256x256 (32*8 x 32*8)."""
        sdfg = _build_vadd2d_sdfg("tl_rt_large_2d_k2")

        m_val, n_val = 32, 32
        m8, n8 = m_val * 8, n_val * 8
        rng = np.random.default_rng(111)
        A = rng.standard_normal((m8, n8))
        B = rng.standard_normal((m8, n8))
        C = np.zeros((m8, n8))

        _apply_and_run(sdfg, widths=(8, 8), A=A, B=B, C=C, M=m_val, N=n_val)
        np.testing.assert_allclose(C, A + B, rtol=1e-14)

    def test_large_1d_non_divisible(self):
        """1-D contiguous add with N=9999 (non-divisible by 8)."""
        N = dace.symbol("N")

        @dace.program
        def add_large_nd(A: dace.float64[N], B: dace.float64[N],
                         C: dace.float64[N]):
            for i in range(N):
                C[i] = A[i] + B[i]

        sdfg = add_large_nd.to_sdfg()
        sdfg.name = "tl_rt_large_1d_nd"

        n = 9999
        rng = np.random.default_rng(112)
        A = rng.standard_normal(n)
        B = rng.standard_normal(n)
        C = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(C, A + B, rtol=1e-14)


# ============================================================
# Category 9: Scalar Broadcast
# ============================================================


class TestScalarBroadcast:
    """Scalar constants/parameters loaded (broadcast) into tile lanes."""

    def test_scalar_multiply(self):
        """``B[i] = A[i] * 2.0`` -- the constant 2.0 is a scalar broadcast."""
        N = dace.symbol("N")

        @dace.program
        def scale_by_2(A: dace.float64[N], B: dace.float64[N]):
            for i in range(N):
                B[i] = A[i] * 2.0

        sdfg = scale_by_2.to_sdfg()
        sdfg.name = "tl_rt_scalar_mul2"

        n = 100
        rng = np.random.default_rng(120)
        A = rng.standard_normal(n)
        B = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_allclose(B, A * 2.0, rtol=1e-14)

    def test_scalar_add(self):
        """``B[i] = A[i] + 42.0`` -- scalar add broadcast."""
        N = dace.symbol("N")

        @dace.program
        def add_42(A: dace.float64[N], B: dace.float64[N]):
            for i in range(N):
                B[i] = A[i] + 42.0

        sdfg = add_42.to_sdfg()
        sdfg.name = "tl_rt_scalar_add42"

        n = 100
        rng = np.random.default_rng(121)
        A = rng.standard_normal(n)
        B = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_allclose(B, A + 42.0, rtol=1e-14)

    def test_scalar_multiply_float32(self):
        """``B[i] = A[i] * 3.0`` -- float32 scalar broadcast."""
        N = dace.symbol("N")

        @dace.program
        def scale_f32(A: dace.float32[N], B: dace.float32[N]):
            for i in range(N):
                B[i] = A[i] * 3.0

        sdfg = scale_f32.to_sdfg()
        sdfg.name = "tl_rt_scalar_mul3_f32"

        n = 100
        rng = np.random.default_rng(122)
        A = rng.standard_normal(n).astype(np.float32)
        B = np.zeros(n, dtype=np.float32)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_allclose(B, (A * np.float32(3.0)), rtol=1e-6)

    def test_scalar_multiply_non_divisible(self):
        """``B[i] = A[i] * 5.0``, n=33 (non-divisible by 8)."""
        N = dace.symbol("N")

        @dace.program
        def scale_by_5(A: dace.float64[N], B: dace.float64[N]):
            for i in range(N):
                B[i] = A[i] * 5.0

        sdfg = scale_by_5.to_sdfg()
        sdfg.name = "tl_rt_scalar_mul5_nd"

        n = 33
        rng = np.random.default_rng(123)
        A = rng.standard_normal(n)
        B = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_allclose(B, A * 5.0, rtol=1e-14)


# ============================================================
# Category 10: Combined Operations Exercising Multiple Loads
# ============================================================


class TestMultipleLoads:
    """Kernels that load from multiple arrays in a single tile iteration."""

    def test_fma_like(self):
        """FMA-like: ``D[i] = A[i] * B[i] + C[i]`` -- 3 tile loads per iteration."""
        N = dace.symbol("N")

        @dace.program
        def fma(A: dace.float64[N], B: dace.float64[N],
                C: dace.float64[N], D: dace.float64[N]):
            for i in range(N):
                D[i] = A[i] * B[i] + C[i]

        sdfg = fma.to_sdfg()
        sdfg.name = "tl_rt_fma"

        n = 100
        rng = np.random.default_rng(130)
        A = rng.standard_normal(n)
        B = rng.standard_normal(n)
        C = rng.standard_normal(n)
        D = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, C=C, D=D, N=n)
        np.testing.assert_allclose(D, A * B + C, rtol=1e-14)

    def test_weighted_sum_2(self):
        """Weighted sum: ``C[i] = 0.5*A[i] + 0.5*B[i]`` -- two loads + scalar broadcasts."""
        N = dace.symbol("N")

        @dace.program
        def wavg(A: dace.float64[N], B: dace.float64[N],
                 C: dace.float64[N]):
            for i in range(N):
                C[i] = 0.5 * A[i] + 0.5 * B[i]

        sdfg = wavg.to_sdfg()
        sdfg.name = "tl_rt_weighted_sum"

        n = 100
        rng = np.random.default_rng(131)
        A = rng.standard_normal(n)
        B = rng.standard_normal(n)
        C = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(C, 0.5 * A + 0.5 * B, rtol=1e-14)

    def test_four_input_chain(self):
        """``E[i] = A[i] + B[i] + C[i] + D[i]`` -- four loads."""
        N = dace.symbol("N")

        @dace.program
        def add4(A: dace.float64[N], B: dace.float64[N],
                 C: dace.float64[N], D: dace.float64[N],
                 E: dace.float64[N]):
            for i in range(N):
                E[i] = A[i] + B[i] + C[i] + D[i]

        sdfg = add4.to_sdfg()
        sdfg.name = "tl_rt_add4"

        n = 65
        rng = np.random.default_rng(132)
        A = rng.standard_normal(n)
        B = rng.standard_normal(n)
        C = rng.standard_normal(n)
        D = rng.standard_normal(n)
        E = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, C=C, D=D, E=E, N=n)
        np.testing.assert_allclose(E, A + B + C + D, rtol=1e-14)

    def test_fma_float32(self):
        """FMA-like in float32: ``D[i] = A[i] * B[i] + C[i]``."""
        N = dace.symbol("N")

        @dace.program
        def fma_f32(A: dace.float32[N], B: dace.float32[N],
                    C: dace.float32[N], D: dace.float32[N]):
            for i in range(N):
                D[i] = A[i] * B[i] + C[i]

        sdfg = fma_f32.to_sdfg()
        sdfg.name = "tl_rt_fma_f32"

        n = 100
        rng = np.random.default_rng(134)
        A = rng.standard_normal(n).astype(np.float32)
        B = rng.standard_normal(n).astype(np.float32)
        C = rng.standard_normal(n).astype(np.float32)
        D = np.zeros(n, dtype=np.float32)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, C=C, D=D, N=n)
        np.testing.assert_allclose(D, A * B + C, rtol=1e-6)

    def test_fma_non_divisible(self):
        """FMA-like with n=17 (non-divisible by 8)."""
        N = dace.symbol("N")

        @dace.program
        def fma_nd(A: dace.float64[N], B: dace.float64[N],
                   C: dace.float64[N], D: dace.float64[N]):
            for i in range(N):
                D[i] = A[i] * B[i] + C[i]

        sdfg = fma_nd.to_sdfg()
        sdfg.name = "tl_rt_fma_nd"

        n = 17
        rng = np.random.default_rng(135)
        A = rng.standard_normal(n)
        B = rng.standard_normal(n)
        C = rng.standard_normal(n)
        D = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, C=C, D=D, N=n)
        np.testing.assert_allclose(D, A * B + C, rtol=1e-14)


# ============================================================
# Category 11: K=2 Multidimensional Tile Loads
# ============================================================


class TestMultiDimTileLoads:
    """K=2 tile loads with ``widths=(W0, W1)``."""

    def test_k2_contiguous_add(self):
        """K=2 contiguous add: ``C[i,j] = A[i,j] + B[i,j]``, widths=(8,8)."""
        sdfg = _build_vadd2d_sdfg("tl_rt_k2_contig_add")

        m_val, n_val = 3, 4
        m8, n8 = m_val * 8, n_val * 8
        rng = np.random.default_rng(140)
        A = rng.standard_normal((m8, n8))
        B = rng.standard_normal((m8, n8))
        C = np.zeros((m8, n8))

        _apply_and_run(sdfg, widths=(8, 8), A=A, B=B, C=C, M=m_val, N=n_val)
        np.testing.assert_allclose(C, A + B, rtol=1e-14)

    def test_k2_contiguous_mul(self):
        """K=2 contiguous multiply via SDFG API, widths=(4,8)."""
        M = dace.symbol("M")
        N = dace.symbol("N")
        sdfg = dace.SDFG("tl_rt_k2_contig_mul")
        sdfg.add_array("A", (M * 4, N * 8), dace.float64)
        sdfg.add_array("B", (M * 4, N * 8), dace.float64)
        sdfg.add_array("C", (M * 4, N * 8), dace.float64)
        state = sdfg.add_state("main")
        state.add_mapped_tasklet(
            "mul",
            {"i": "0:M*4", "j": "0:N*8"},
            {"_a": dace.Memlet("A[i, j]"), "_b": dace.Memlet("B[i, j]")},
            "_c = _a * _b",
            {"_c": dace.Memlet("C[i, j]")},
            external_edges=True,
        )

        m_val, n_val = 3, 4
        m4, n8 = m_val * 4, n_val * 8
        rng = np.random.default_rng(141)
        A = rng.standard_normal((m4, n8))
        B = rng.standard_normal((m4, n8))
        C = np.zeros((m4, n8))

        _apply_and_run(sdfg, widths=(4, 8), A=A, B=B, C=C, M=m_val, N=n_val)
        np.testing.assert_allclose(C, A * B, rtol=1e-14)

    def test_k2_float32(self):
        """K=2 contiguous add in float32, widths=(8,8)."""
        M = dace.symbol("M")
        N = dace.symbol("N")
        sdfg = dace.SDFG("tl_rt_k2_f32")
        sdfg.add_array("A", (M * 8, N * 8), dace.float32)
        sdfg.add_array("B", (M * 8, N * 8), dace.float32)
        sdfg.add_array("C", (M * 8, N * 8), dace.float32)
        state = sdfg.add_state("main")
        state.add_mapped_tasklet(
            "add",
            {"i": "0:M*8", "j": "0:N*8"},
            {"_a": dace.Memlet("A[i, j]"), "_b": dace.Memlet("B[i, j]")},
            "_c = _a + _b",
            {"_c": dace.Memlet("C[i, j]")},
            external_edges=True,
        )

        m_val, n_val = 2, 3
        m8, n8 = m_val * 8, n_val * 8
        rng = np.random.default_rng(142)
        A = rng.standard_normal((m8, n8)).astype(np.float32)
        B = rng.standard_normal((m8, n8)).astype(np.float32)
        C = np.zeros((m8, n8), dtype=np.float32)

        _apply_and_run(sdfg, widths=(8, 8), A=A, B=B, C=C, M=m_val, N=n_val)
        np.testing.assert_allclose(C, A + B, rtol=1e-6)


# ============================================================
# Category 12: Codegen Verification
# ============================================================


class TestCodegenStructure:
    """Verify that the generated code contains expected cuTile primitives.

    These tests exercise the pipeline through code generation and inspect the
    generated Python source for ct.load / ct.gather calls.
    """

    def test_contiguous_load_emits_ct_load(self):
        """Contiguous load pattern should emit ``ct.load`` in generated code."""
        N = dace.symbol("N")

        @dace.program
        def copy_cg(A: dace.float64[N], B: dace.float64[N]):
            for i in range(N):
                B[i] = A[i]

        sdfg = copy_cg.to_sdfg()
        sdfg.name = "tl_rt_codegen_ct_load"
        VectorizeCuTile(widths=(8,)).apply_pass(sdfg, {})

        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        assert "ct.load(" in code, "Expected ct.load in generated code"

    def test_gather_emits_ct_gather(self):
        """Gather load pattern should emit ``ct.gather`` in generated code."""
        N = dace.symbol("N")

        @dace.program
        def gather_cg(A: dace.float64[N], idx: dace.int32[N],
                      B: dace.float64[N]):
            for i in range(N):
                B[i] = A[idx[i]]

        sdfg = gather_cg.to_sdfg()
        sdfg.name = "tl_rt_codegen_ct_gather"
        VectorizeCuTile(widths=(8,)).apply_pass(sdfg, {})

        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        assert "ct.gather(" in code, "Expected ct.gather in generated code"

    def test_generated_code_is_valid_python(self):
        """Generated cuTile kernel code should be parseable as valid Python."""
        import ast as ast_mod

        N = dace.symbol("N")

        @dace.program
        def add_cg(A: dace.float64[N], B: dace.float64[N],
                   C: dace.float64[N]):
            for i in range(N):
                C[i] = A[i] + B[i]

        sdfg = add_cg.to_sdfg()
        sdfg.name = "tl_rt_codegen_valid_py"
        VectorizeCuTile(widths=(8,)).apply_pass(sdfg, {})

        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        ast_mod.parse(code)

    def test_mask_gen_present_for_symbolic_size(self):
        """Symbolic-sized SDFG should generate mask comparison in code."""
        N = dace.symbol("N")

        @dace.program
        def copy_mask_cg(A: dace.float64[N], B: dace.float64[N]):
            for i in range(N):
                B[i] = A[i]

        sdfg = copy_mask_cg.to_sdfg()
        sdfg.name = "tl_rt_codegen_mask"
        VectorizeCuTile(widths=(8,)).apply_pass(sdfg, {})

        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        # Mask generation uses arange and comparison against the bound
        assert "ct.arange(8" in code, "Expected ct.arange(8 for mask generation"

    def test_k2_generates_valid_python(self):
        """K=2 SDFG should generate valid Python."""
        import ast as ast_mod

        sdfg = _build_vadd2d_sdfg("tl_rt_codegen_k2_valid")
        VectorizeCuTile(widths=(8, 8)).apply_pass(sdfg, {})

        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        ast_mod.parse(code)


# ============================================================
# Category 13: Edge Cases
# ============================================================


class TestEdgeCases:
    """Edge cases for tile loads."""

    def test_identity_copy_aligned(self):
        """Simple identity copy, aligned size: ``B[i] = A[i]``, n=64."""
        N = dace.symbol("N")

        @dace.program
        def id_copy(A: dace.float64[N], B: dace.float64[N]):
            for i in range(N):
                B[i] = A[i]

        sdfg = id_copy.to_sdfg()
        sdfg.name = "tl_rt_edge_id_copy"

        n = 64
        rng = np.random.default_rng(150)
        A = rng.standard_normal(n)
        B = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_allclose(B, A, rtol=1e-14)

    def test_negate(self):
        """Negate: ``B[i] = -A[i]``."""
        N = dace.symbol("N")

        @dace.program
        def negate(A: dace.float64[N], B: dace.float64[N]):
            for i in range(N):
                B[i] = -A[i]

        sdfg = negate.to_sdfg()
        sdfg.name = "tl_rt_edge_negate"

        n = 100
        rng = np.random.default_rng(151)
        A = rng.standard_normal(n)
        B = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_allclose(B, -A, rtol=1e-14)

    def test_self_add(self):
        """Self-add: ``B[i] = A[i] + A[i]`` -- same source loaded twice."""
        N = dace.symbol("N")

        @dace.program
        def self_add(A: dace.float64[N], B: dace.float64[N]):
            for i in range(N):
                B[i] = A[i] + A[i]

        sdfg = self_add.to_sdfg()
        sdfg.name = "tl_rt_edge_self_add"

        n = 100
        rng = np.random.default_rng(152)
        A = rng.standard_normal(n)
        B = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_allclose(B, 2 * A, rtol=1e-14)

    def test_subtraction_non_commutative(self):
        """Non-commutative: ``C[i] = A[i] - B[i]`` (order matters)."""
        N = dace.symbol("N")

        @dace.program
        def sub_nc(A: dace.float64[N], B: dace.float64[N],
                   C: dace.float64[N]):
            for i in range(N):
                C[i] = A[i] - B[i]

        sdfg = sub_nc.to_sdfg()
        sdfg.name = "tl_rt_edge_sub_nc"

        n = 100
        rng = np.random.default_rng(153)
        A = rng.standard_normal(n)
        B = rng.standard_normal(n)
        C = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(C, A - B, rtol=1e-14)

    def test_division(self):
        """Division: ``C[i] = A[i] / B[i]``."""
        N = dace.symbol("N")

        @dace.program
        def div_op(A: dace.float64[N], B: dace.float64[N],
                   C: dace.float64[N]):
            for i in range(N):
                C[i] = A[i] / B[i]

        sdfg = div_op.to_sdfg()
        sdfg.name = "tl_rt_edge_div"

        n = 100
        rng = np.random.default_rng(154)
        A = rng.standard_normal(n)
        B = rng.standard_normal(n) + 0.1  # avoid division by zero
        C = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(C, A / B, rtol=1e-14)

    def test_mul_sub_chain(self):
        """Chained ops: ``C[i] = A[i] * B[i] - A[i]``."""
        N = dace.symbol("N")

        @dace.program
        def mul_sub(A: dace.float64[N], B: dace.float64[N],
                    C: dace.float64[N]):
            for i in range(N):
                C[i] = A[i] * B[i] - A[i]

        sdfg = mul_sub.to_sdfg()
        sdfg.name = "tl_rt_edge_mul_sub"

        n = 100
        rng = np.random.default_rng(155)
        A = rng.standard_normal(n)
        B = rng.standard_normal(n)
        C = np.zeros(n)

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(C, A * B - A, rtol=1e-14)


# ============================================================
# Category 14: 2-D Transposed Loads
# ============================================================


def _build_transpose_2d_sdfg(name, dtype=dace.float64):
    """Build a symbolic-sized 2-D transpose ``B[i,j] = A[j,i]`` SDFG.

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


class TestTransposedLoads:
    """2-D matrix transpose: ``B[i,j] = A[j,i]``.

    Transpose requires K=2 vectorization (``widths=(W0, W1)``) so that both
    axes are tiled.  The vectorizer detects the swapped index pattern and
    emits a ``TileLoad`` with ``src_dims=[1, 0]``, which expands to
    ``ct.load(..., order=(1, 0))`` in cuTile code.

    K=1 fails at runtime because the cuTile scatter receives a tile whose
    shape is incompatible with the 1-D index tuple (shape ``(1, W)`` vs.
    index shape ``(W,)``).
    """

    # ---- K=2: working transpose tests ----

    def test_transpose_2d_float64(self):
        """2-D transpose, float64, K=2 widths=(8,8)."""
        M = dace.symbol("M")
        N = dace.symbol("N")

        @dace.program
        def tr_f64(A: dace.float64[M, N], B: dace.float64[N, M]):
            for i in range(N):
                for j in range(M):
                    B[i, j] = A[j, i]

        sdfg = tr_f64.to_sdfg()
        sdfg.name = "tl_rt_tr_2d_f64"

        m, n = 16, 24
        rng = np.random.default_rng(200)
        A = rng.standard_normal((m, n))
        B = np.zeros((n, m))

        _apply_and_run(sdfg, widths=(8, 8), A=A, B=B, M=m, N=n)
        np.testing.assert_allclose(B, A.T, rtol=1e-14)

    def test_transpose_2d_float32(self):
        """2-D transpose, float32, K=2 widths=(8,8)."""
        M = dace.symbol("M")
        N = dace.symbol("N")

        @dace.program
        def tr_f32(A: dace.float32[M, N], B: dace.float32[N, M]):
            for i in range(N):
                for j in range(M):
                    B[i, j] = A[j, i]

        sdfg = tr_f32.to_sdfg()
        sdfg.name = "tl_rt_tr_2d_f32"

        m, n = 16, 24
        rng = np.random.default_rng(201)
        A = rng.standard_normal((m, n)).astype(np.float32)
        B = np.zeros((n, m), dtype=np.float32)

        _apply_and_run(sdfg, widths=(8, 8), A=A, B=B, M=m, N=n)
        np.testing.assert_allclose(B, A.T, rtol=1e-6)

    def test_transpose_2d_int32(self):
        """2-D transpose, int32, K=2 widths=(8,8) via SDFG API."""
        sdfg = _build_transpose_2d_sdfg("tl_rt_tr_2d_i32", dtype=dace.int32)

        m, n = 16, 24
        rng = np.random.default_rng(202)
        A = rng.integers(0, 500, (m, n), dtype=np.int32)
        B = np.zeros((n, m), dtype=np.int32)

        _apply_and_run(sdfg, widths=(8, 8), A=A, B=B, M=m, N=n)
        np.testing.assert_array_equal(B, A.T)

    def test_transpose_2d_non_divisible(self):
        """2-D transpose with non-divisible sizes (m=13, n=17)."""
        M = dace.symbol("M")
        N = dace.symbol("N")

        @dace.program
        def tr_nd(A: dace.float64[M, N], B: dace.float64[N, M]):
            for i in range(N):
                for j in range(M):
                    B[i, j] = A[j, i]

        sdfg = tr_nd.to_sdfg()
        sdfg.name = "tl_rt_tr_2d_nd"

        m, n = 13, 17
        rng = np.random.default_rng(203)
        A = rng.standard_normal((m, n))
        B = np.zeros((n, m))

        _apply_and_run(sdfg, widths=(8, 8), A=A, B=B, M=m, N=n)
        np.testing.assert_allclose(B, A.T, rtol=1e-14)

    @pytest.mark.parametrize("m,n", [(1, 1), (3, 5), (7, 9), (8, 16),
                                      (15, 17), (32, 32), (33, 31)])
    def test_transpose_2d_various_sizes(self, m, n):
        """2-D transpose with various (m, n) sizes."""
        sdfg = _build_transpose_2d_sdfg(f"tl_rt_tr_2d_sz_{m}x{n}")

        rng = np.random.default_rng(m * 100 + n)
        A = rng.standard_normal((m, n))
        B = np.zeros((n, m))

        _apply_and_run(sdfg, widths=(8, 8), A=A, B=B, M=m, N=n)
        np.testing.assert_allclose(B, A.T, rtol=1e-14)

    def test_transpose_2d_widths_4_4(self):
        """2-D transpose with widths=(4,4), non-divisible sizes."""
        sdfg = _build_transpose_2d_sdfg("tl_rt_tr_2d_w4")

        m, n = 13, 17
        rng = np.random.default_rng(204)
        A = rng.standard_normal((m, n))
        B = np.zeros((n, m))

        _apply_and_run(sdfg, widths=(4, 4), A=A, B=B, M=m, N=n)
        np.testing.assert_allclose(B, A.T, rtol=1e-14)

    def test_transpose_2d_sdfg_api(self):
        """2-D transpose via SDFG API (not @dace.program), float64."""
        sdfg = _build_transpose_2d_sdfg("tl_rt_tr_2d_api")

        m, n = 16, 24
        rng = np.random.default_rng(205)
        A = rng.standard_normal((m, n))
        B = np.zeros((n, m))

        _apply_and_run(sdfg, widths=(8, 8), A=A, B=B, M=m, N=n)
        np.testing.assert_allclose(B, A.T, rtol=1e-14)

    def test_transpose_2d_square(self):
        """2-D transpose of a square matrix (m == n)."""
        sdfg = _build_transpose_2d_sdfg("tl_rt_tr_2d_sq")

        m, n = 16, 16
        rng = np.random.default_rng(206)
        A = rng.standard_normal((m, n))
        B = np.zeros((n, m))

        _apply_and_run(sdfg, widths=(8, 8), A=A, B=B, M=m, N=n)
        np.testing.assert_allclose(B, A.T, rtol=1e-14)

    def test_transpose_2d_codegen_has_order(self):
        """Verify that the generated cuTile code contains ``order=(1, 0)``."""
        sdfg = _build_transpose_2d_sdfg("tl_rt_tr_2d_cg_order")
        VectorizeCuTile(widths=(8, 8)).apply_pass(sdfg, {})
        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        assert "order=(1, 0)" in code, (
            "Expected order=(1, 0) in generated code for 2-D transpose"
        )

    def test_transpose_2d_tile_load_src_dims(self):
        """Verify the TileLoad node has src_dims=[1, 0] for transpose."""
        from dace.libraries.tileops.nodes.tile_load import TileLoad

        sdfg = _build_transpose_2d_sdfg("tl_rt_tr_2d_src_dims")
        VectorizeCuTile(widths=(8, 8)).apply_pass(sdfg, {})
        found = False
        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, TileLoad):
                    assert node.src_dims == [1, 0], (
                        f"Expected src_dims=[1, 0], got {node.src_dims}"
                    )
                    found = True
        assert found, "No TileLoad node found in SDFG"

    # ---- K=1: documents the scatter shape mismatch limitation ----

    def test_transpose_2d_k1_xfail(self):
        """2-D transpose with K=1 fails at runtime due to scatter shape."""
        M = dace.symbol("M")
        N = dace.symbol("N")

        @dace.program
        def tr_k1(A: dace.float64[M, N], B: dace.float64[N, M]):
            for i in range(N):
                for j in range(M):
                    B[i, j] = A[j, i]

        sdfg = tr_k1.to_sdfg()
        sdfg.name = "tl_rt_tr_2d_k1_xfail"

        m, n = 16, 24
        rng = np.random.default_rng(210)
        A = rng.standard_normal((m, n))
        B = np.zeros((n, m))

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, M=m, N=n)
        np.testing.assert_allclose(B, A.T, rtol=1e-14)


# ============================================================
# Category 15: 3-D Axis Permutation Loads
# ============================================================


def _build_permute_3d_kji_sdfg(name, dtype=dace.float64):
    """Build a 3-D full axis reversal ``B[i,j,k] = A[k,j,i]`` SDFG.

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


def _build_permute_3d_ikj_sdfg(name, dtype=dace.float64):
    """Build a 3-D partial permutation ``B[i,j,k] = A[i,k,j]`` SDFG.

    This swaps axes 1 and 2, keeping axis 0 in place.

    :param name: Unique SDFG name.
    :param dtype: Data type for all arrays.
    :returns: The constructed SDFG.
    """
    L = dace.symbol("L")
    M = dace.symbol("M")
    N = dace.symbol("N")
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (L, M, N), dtype)
    sdfg.add_array("B", (L, N, M), dtype)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "permute",
        {"i": "0:L", "j": "0:N", "k": "0:M"},
        {"_a": dace.Memlet("A[i, k, j]")},
        "_b = _a",
        {"_b": dace.Memlet("B[i, j, k]")},
        external_edges=True,
    )
    return sdfg


class TestAxisPermutationLoads:
    """3-D axis permutation loads: ``B[i,j,k] = A[perm(i,j,k)]``.

    These tests exercise the ``src_dims`` / ``order=`` path in TileLoad
    for 3-D arrays.  Axis permutation requires K=3 vectorization
    (``widths=(W0, W1, W2)``) so that all three axes are tiled.

    Two permutations are tested:

    * **Full reversal** ``(k,j,i)``: ``src_dims=[2, 1, 0]``
    * **Partial swap** ``(i,k,j)``: ``src_dims=[0, 2, 1]``

    K=1 and K=2 fail at runtime due to scatter shape mismatches when
    K < ndim for permuted accesses.
    """

    # ---- K=3: working 3-D permutation tests ----

    def test_permute_3d_kji_float64(self):
        """3-D full reversal ``B[i,j,k] = A[k,j,i]``, float64, K=3."""
        sdfg = _build_permute_3d_kji_sdfg("tl_rt_perm_kji_f64")

        l, m, n = 8, 16, 24
        rng = np.random.default_rng(300)
        A = rng.standard_normal((l, m, n))
        B = np.zeros((n, m, l))

        _apply_and_run(sdfg, widths=(4, 4, 4), A=A, B=B, L=l, M=m, N=n)
        np.testing.assert_allclose(B, A.transpose(2, 1, 0), rtol=1e-14)

    def test_permute_3d_ikj_float64(self):
        """3-D partial swap ``B[i,j,k] = A[i,k,j]``, float64, K=3."""
        sdfg = _build_permute_3d_ikj_sdfg("tl_rt_perm_ikj_f64")

        l, m, n = 8, 16, 24
        rng = np.random.default_rng(301)
        A = rng.standard_normal((l, m, n))
        B = np.zeros((l, n, m))

        _apply_and_run(sdfg, widths=(4, 4, 4), A=A, B=B, L=l, M=m, N=n)
        np.testing.assert_allclose(B, A.transpose(0, 2, 1), rtol=1e-14)

    def test_permute_3d_kji_non_divisible(self):
        """3-D full reversal with non-divisible sizes (5, 7, 11)."""
        sdfg = _build_permute_3d_kji_sdfg("tl_rt_perm_kji_nd")

        l, m, n = 5, 7, 11
        rng = np.random.default_rng(302)
        A = rng.standard_normal((l, m, n))
        B = np.zeros((n, m, l))

        _apply_and_run(sdfg, widths=(4, 4, 4), A=A, B=B, L=l, M=m, N=n)
        np.testing.assert_allclose(B, A.transpose(2, 1, 0), rtol=1e-14)

    def test_permute_3d_ikj_non_divisible(self):
        """3-D partial swap with non-divisible sizes (5, 7, 11)."""
        sdfg = _build_permute_3d_ikj_sdfg("tl_rt_perm_ikj_nd")

        l, m, n = 5, 7, 11
        rng = np.random.default_rng(303)
        A = rng.standard_normal((l, m, n))
        B = np.zeros((l, n, m))

        _apply_and_run(sdfg, widths=(4, 4, 4), A=A, B=B, L=l, M=m, N=n)
        np.testing.assert_allclose(B, A.transpose(0, 2, 1), rtol=1e-14)

    def test_permute_3d_kji_float32(self):
        """3-D full reversal, float32, K=3."""
        sdfg = _build_permute_3d_kji_sdfg("tl_rt_perm_kji_f32",
                                           dtype=dace.float32)

        l, m, n = 8, 16, 24
        rng = np.random.default_rng(304)
        A = rng.standard_normal((l, m, n)).astype(np.float32)
        B = np.zeros((n, m, l), dtype=np.float32)

        _apply_and_run(sdfg, widths=(4, 4, 4), A=A, B=B, L=l, M=m, N=n)
        np.testing.assert_allclose(B, A.transpose(2, 1, 0), rtol=1e-6)

    def test_permute_3d_kji_dace_program(self):
        """3-D full reversal via @dace.program (not SDFG API)."""
        L = dace.symbol("L")
        M = dace.symbol("M")
        N = dace.symbol("N")

        @dace.program
        def perm_kji(A: dace.float64[L, M, N], B: dace.float64[N, M, L]):
            for i in range(N):
                for j in range(M):
                    for k in range(L):
                        B[i, j, k] = A[k, j, i]

        sdfg = perm_kji.to_sdfg()
        sdfg.name = "tl_rt_perm_kji_dp"

        l, m, n = 8, 16, 24
        rng = np.random.default_rng(305)
        A = rng.standard_normal((l, m, n))
        B = np.zeros((n, m, l))

        _apply_and_run(sdfg, widths=(4, 4, 4), A=A, B=B, L=l, M=m, N=n)
        np.testing.assert_allclose(B, A.transpose(2, 1, 0), rtol=1e-14)

    def test_permute_3d_ikj_dace_program(self):
        """3-D partial swap via @dace.program (not SDFG API)."""
        L = dace.symbol("L")
        M = dace.symbol("M")
        N = dace.symbol("N")

        @dace.program
        def perm_ikj(A: dace.float64[L, M, N], B: dace.float64[L, N, M]):
            for i in range(L):
                for j in range(N):
                    for k in range(M):
                        B[i, j, k] = A[i, k, j]

        sdfg = perm_ikj.to_sdfg()
        sdfg.name = "tl_rt_perm_ikj_dp"

        l, m, n = 8, 16, 24
        rng = np.random.default_rng(306)
        A = rng.standard_normal((l, m, n))
        B = np.zeros((l, n, m))

        _apply_and_run(sdfg, widths=(4, 4, 4), A=A, B=B, L=l, M=m, N=n)
        np.testing.assert_allclose(B, A.transpose(0, 2, 1), rtol=1e-14)

    def test_permute_3d_kji_tile_load_src_dims(self):
        """Verify TileLoad has src_dims=[2, 1, 0] for full reversal."""
        from dace.libraries.tileops.nodes.tile_load import TileLoad

        sdfg = _build_permute_3d_kji_sdfg("tl_rt_perm_kji_src_dims")
        VectorizeCuTile(widths=(4, 4, 4)).apply_pass(sdfg, {})
        found = False
        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, TileLoad):
                    assert node.src_dims == [2, 1, 0], (
                        f"Expected src_dims=[2, 1, 0], got {node.src_dims}"
                    )
                    found = True
        assert found, "No TileLoad node found in SDFG"

    def test_permute_3d_ikj_tile_load_src_dims(self):
        """Verify TileLoad has src_dims=[0, 2, 1] for partial swap."""
        from dace.libraries.tileops.nodes.tile_load import TileLoad

        sdfg = _build_permute_3d_ikj_sdfg("tl_rt_perm_ikj_src_dims")
        VectorizeCuTile(widths=(4, 4, 4)).apply_pass(sdfg, {})
        found = False
        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, TileLoad):
                    assert node.src_dims == [0, 2, 1], (
                        f"Expected src_dims=[0, 2, 1], got {node.src_dims}"
                    )
                    found = True
        assert found, "No TileLoad node found in SDFG"

    # ---- K<ndim: documents scatter shape mismatch limitation ----

    def test_permute_3d_k1_xfail(self):
        """3-D permutation with K=1 fails at runtime."""
        L = dace.symbol("L")
        M = dace.symbol("M")
        N = dace.symbol("N")

        @dace.program
        def perm_k1(A: dace.float64[L, M, N], B: dace.float64[N, M, L]):
            for i in range(N):
                for j in range(M):
                    for k in range(L):
                        B[i, j, k] = A[k, j, i]

        sdfg = perm_k1.to_sdfg()
        sdfg.name = "tl_rt_perm_3d_k1_xfail"

        l, m, n = 8, 16, 24
        rng = np.random.default_rng(310)
        A = rng.standard_normal((l, m, n))
        B = np.zeros((n, m, l))

        _apply_and_run(sdfg, widths=(8,), A=A, B=B, L=l, M=m, N=n)
        np.testing.assert_allclose(B, A.transpose(2, 1, 0), rtol=1e-14)

    def test_permute_3d_k2_xfail(self):
        """3-D permutation with K=2 fails at runtime."""
        L = dace.symbol("L")
        M = dace.symbol("M")
        N = dace.symbol("N")

        @dace.program
        def perm_k2(A: dace.float64[L, M, N], B: dace.float64[N, M, L]):
            for i in range(N):
                for j in range(M):
                    for k in range(L):
                        B[i, j, k] = A[k, j, i]

        sdfg = perm_k2.to_sdfg()
        sdfg.name = "tl_rt_perm_3d_k2_xfail"

        l, m, n = 8, 16, 24
        rng = np.random.default_rng(311)
        A = rng.standard_normal((l, m, n))
        B = np.zeros((n, m, l))

        _apply_and_run(sdfg, widths=(8, 8), A=A, B=B, L=l, M=m, N=n)
        np.testing.assert_allclose(B, A.transpose(2, 1, 0), rtol=1e-14)


# ============================================================
# Entry point
# ============================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=300"])
