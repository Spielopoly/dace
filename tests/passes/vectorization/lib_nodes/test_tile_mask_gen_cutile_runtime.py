# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU runtime tests for the cuTile expansion of :class:`TileMaskGen`.

Two test groups:

**Group 1 (GPU):** End-to-end through the ``VectorizeCuTile`` pipeline.
``@dace.program`` (or SDFG API) -> ``VectorizeCuTile(widths=...)`` ->
compile -> run on GPU -> assert vs NumPy.  TileMaskGen is exercised
indirectly: the pipeline generates it for non-divisible array sizes
(remainder masking).

**Group 2 (no GPU):** Validation tests for the cuTile expansion:
C++-expression guard, int64/int32 dtype selection.

**Naming convention:** every SDFG name must be globally unique
(they share ``.dacecache``).
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.transformation.passes.vectorization.vectorize_cutile import VectorizeCuTile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_counter = 0


def _unique(prefix: str) -> str:
    """Return a globally-unique SDFG name to avoid .dacecache collisions.

    :param prefix: Name prefix.
    :returns: Unique SDFG name.
    """
    global _counter
    _counter += 1
    return f"{prefix}_{_counter}"


def _apply_and_run(sdfg, widths, **kwargs):
    """Apply VectorizeCuTile and run the compiled SDFG.

    :param sdfg: SDFG to transform and compile.
    :param widths: Tile widths for vectorization (powers of 2).
    :param kwargs: Named arguments for the SDFG (arrays + symbols).
    :returns: The kwargs dict (output arrays are modified in-place).
    """
    VectorizeCuTile(widths=widths).apply_pass(sdfg, {})
    csdfg = sdfg.compile()
    csdfg(**kwargs)
    return kwargs


# ---------------------------------------------------------------------------
# Group 1: End-to-end through VectorizeCuTile (GPU runtime tests)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestMaskGenK1NonDivisible:
    """K=1 tests with non-divisible array sizes exercising TileMaskGen."""

    def test_copy_n13_w8(self):
        """K=1, copy A->B with N=13, W=8.  Last tile has 5 active lanes."""
        N = dace.symbol("N")

        @dace.program
        def copy_nd(A: dace.float32[N], B: dace.float32[N]):
            for i in range(N):
                B[i] = A[i]

        sdfg = copy_nd.to_sdfg()
        sdfg.name = _unique("mg_k1_nondiv_copy")

        n = 13
        A = np.random.default_rng(42).random(n).astype(np.float32)
        B = np.zeros(n, dtype=np.float32)
        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_array_equal(B, A)

    def test_single_element_n1_w8(self):
        """K=1, N=1, W=8.  Only 1 active lane out of 8 -- extreme tail."""
        N = dace.symbol("N")

        @dace.program
        def copy_n1(A: dace.float32[N], B: dace.float32[N]):
            for i in range(N):
                B[i] = A[i]

        sdfg = copy_n1.to_sdfg()
        sdfg.name = _unique("mg_k1_single")

        A = np.array([42.0], dtype=np.float32)
        B = np.zeros(1, dtype=np.float32)
        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=1)
        np.testing.assert_array_equal(B, A)

    def test_axpy_n22_w8(self):
        """K=1, B = A*2+1 with N=22, W=8.  Tests mask + binop."""
        N = dace.symbol("N")

        @dace.program
        def axpy(A: dace.float32[N], B: dace.float32[N]):
            for i in range(N):
                B[i] = A[i] * 2.0 + 1.0

        sdfg = axpy.to_sdfg()
        sdfg.name = _unique("mg_k1_axpy")

        n = 22
        A = np.random.default_rng(44).random(n).astype(np.float32)
        B = np.zeros(n, dtype=np.float32)
        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_allclose(B, A * 2.0 + 1.0, rtol=1e-6)

    def test_negate_n7_w8(self):
        """K=1, B = -A with N=7, W=8.  7 of 8 lanes active."""
        N = dace.symbol("N")

        @dace.program
        def neg(A: dace.float64[N], B: dace.float64[N]):
            for i in range(N):
                B[i] = -A[i]

        sdfg = neg.to_sdfg()
        sdfg.name = _unique("mg_k1_neg_n7")

        n = 7
        A = np.random.default_rng(45).standard_normal(n)
        B = np.zeros(n)
        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_allclose(B, -A, rtol=1e-14)

    def test_fma_n33_w8(self):
        """K=1, D = A*B + C with N=33, W=8.  Tests mask + 3 loads + binops."""
        N = dace.symbol("N")

        @dace.program
        def fma(A: dace.float64[N], B: dace.float64[N],
                C: dace.float64[N], D: dace.float64[N]):
            for i in range(N):
                D[i] = A[i] * B[i] + C[i]

        sdfg = fma.to_sdfg()
        sdfg.name = _unique("mg_k1_fma")

        n = 33
        rng = np.random.default_rng(46)
        A = rng.standard_normal(n)
        B = rng.standard_normal(n)
        C = rng.standard_normal(n)
        D = np.zeros(n)
        _apply_and_run(sdfg, widths=(8,), A=A, B=B, C=C, D=D, N=n)
        np.testing.assert_allclose(D, A * B + C, rtol=1e-14)


@pytest.mark.gpu
class TestMaskGenK1AllActive:
    """K=1 tests with divisible sizes -- all mask lanes True."""

    def test_copy_n16_w8(self):
        """K=1, copy with N=16 (divisible by W=8).  All masks all-True."""
        N = dace.symbol("N")

        @dace.program
        def copy_al(A: dace.float32[N], B: dace.float32[N]):
            for i in range(N):
                B[i] = A[i]

        sdfg = copy_al.to_sdfg()
        sdfg.name = _unique("mg_k1_aligned_copy")

        n = 16
        A = np.random.default_rng(43).random(n).astype(np.float32)
        B = np.zeros(n, dtype=np.float32)
        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_array_equal(B, A)

    def test_copy_n64_w8(self):
        """K=1, copy with N=64, W=8.  Multiple full tiles, no remainder."""
        N = dace.symbol("N")

        @dace.program
        def copy_64(A: dace.float64[N], B: dace.float64[N]):
            for i in range(N):
                B[i] = A[i]

        sdfg = copy_64.to_sdfg()
        sdfg.name = _unique("mg_k1_aligned64")

        n = 64
        A = np.random.default_rng(50).standard_normal(n)
        B = np.zeros(n)
        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_allclose(B, A, rtol=1e-14)


@pytest.mark.gpu
class TestMaskGenK1SymbolicSize:
    """K=1 tests with symbolic sizes."""

    def test_symbolic_n13(self):
        """K=1, symbolic N, W=8.  Tested with N=13 at runtime."""
        N = dace.symbol("N")

        @dace.program
        def copy_sym(A: dace.float32[N], B: dace.float32[N]):
            for i in range(N):
                B[i] = A[i]

        sdfg = copy_sym.to_sdfg()
        sdfg.name = _unique("mg_k1_symbolic")

        n = 13
        A = np.random.default_rng(47).random(n).astype(np.float32)
        B = np.zeros(n, dtype=np.float32)
        _apply_and_run(sdfg, widths=(8,), A=A, B=B, N=n)
        np.testing.assert_array_equal(B, A)

    @pytest.mark.parametrize("n", [1, 3, 7, 9, 15, 17, 31, 33, 63, 65])
    def test_symbolic_various_sizes(self, n):
        """K=1, symbolic add with various N (including non-divisible)."""
        N = dace.symbol("N")

        @dace.program
        def add_sym(A: dace.float64[N], B: dace.float64[N],
                    C: dace.float64[N]):
            for i in range(N):
                C[i] = A[i] + B[i]

        sdfg = add_sym.to_sdfg()
        sdfg.name = _unique(f"mg_k1_sym_n{n}")

        rng = np.random.default_rng(n + 3000)
        A = rng.standard_normal(n)
        B = rng.standard_normal(n)
        C = np.zeros(n)
        _apply_and_run(sdfg, widths=(8,), A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(C, A + B, rtol=1e-14)


@pytest.mark.gpu
class TestMaskGenK2:
    """K=2 tests exercising 2D mask generation."""

    def _build_copy2d_sdfg(self, name, dtype=dace.float64):
        """Build a symbolic-sized 2-D C[i,j] = A[i,j] SDFG.

        Array shapes use ``M*W0`` and ``N*W1`` so tile-aligned sizes
        can be tested; non-divisible sizes are tested via symbolic
        free-form shapes.

        :param name: Unique SDFG name.
        :param dtype: DaCe dtype.
        :returns: Constructed SDFG.
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

    def test_k2_non_divisible_both_dims(self):
        """K=2, copy with M=5, N=13, widths=(4,8).  Both dims non-divisible."""
        sdfg = self._build_copy2d_sdfg(_unique("mg_k2_nondiv"))

        m, n = 5, 13
        rng = np.random.default_rng(46)
        A = rng.standard_normal((m, n))
        B = np.zeros((m, n))
        _apply_and_run(sdfg, widths=(4, 8), A=A, B=B, M=m, N=n)
        np.testing.assert_allclose(B, A, rtol=1e-14)

    def test_k2_aligned_both_dims(self):
        """K=2, copy with M=8, N=16, widths=(4,8).  Both dims aligned."""
        sdfg = self._build_copy2d_sdfg(_unique("mg_k2_aligned"))

        m, n = 8, 16
        rng = np.random.default_rng(47)
        A = rng.standard_normal((m, n))
        B = np.zeros((m, n))
        _apply_and_run(sdfg, widths=(4, 8), A=A, B=B, M=m, N=n)
        np.testing.assert_allclose(B, A, rtol=1e-14)

    def test_k2_one_dim_non_divisible(self):
        """K=2, copy with M=8 (aligned), N=13 (non-div by 8), widths=(4,8)."""
        sdfg = self._build_copy2d_sdfg(_unique("mg_k2_onenondiv"))

        m, n = 8, 13
        rng = np.random.default_rng(48)
        A = rng.standard_normal((m, n))
        B = np.zeros((m, n))
        _apply_and_run(sdfg, widths=(4, 8), A=A, B=B, M=m, N=n)
        np.testing.assert_allclose(B, A, rtol=1e-14)

    def test_k2_add_non_divisible(self):
        """K=2, add C=A+B with M=5, N=13, widths=(4,8)."""
        M = dace.symbol("M")
        N = dace.symbol("N")
        sdfg = dace.SDFG(_unique("mg_k2_add_nondiv"))
        sdfg.add_array("A", (M, N), dace.float64)
        sdfg.add_array("B", (M, N), dace.float64)
        sdfg.add_array("C", (M, N), dace.float64)
        state = sdfg.add_state("main")
        state.add_mapped_tasklet(
            "add2d",
            {"i": "0:M", "j": "0:N"},
            {"_a": dace.Memlet("A[i, j]"), "_b": dace.Memlet("B[i, j]")},
            "_c = _a + _b",
            {"_c": dace.Memlet("C[i, j]")},
            external_edges=True,
        )

        m, n = 5, 13
        rng = np.random.default_rng(49)
        A = rng.standard_normal((m, n))
        B = rng.standard_normal((m, n))
        C = np.zeros((m, n))
        _apply_and_run(sdfg, widths=(4, 8), A=A, B=B, C=C, M=m, N=n)
        np.testing.assert_allclose(C, A + B, rtol=1e-14)

    def test_k2_single_element(self):
        """K=2, copy with M=1, N=1, widths=(4,8).  Extreme tail: 1 active lane."""
        sdfg = self._build_copy2d_sdfg(_unique("mg_k2_1x1"))

        m, n = 1, 1
        rng = np.random.default_rng(50)
        A = rng.standard_normal((m, n))
        B = np.zeros((m, n))
        _apply_and_run(sdfg, widths=(4, 8), A=A, B=B, M=m, N=n)
        np.testing.assert_allclose(B, A, rtol=1e-14)


@pytest.mark.gpu
class TestMaskGenTileWidths:
    """Tests with various tile widths (all powers of 2)."""

    @pytest.mark.parametrize("w", [4, 8, 16, 32])
    def test_k1_non_divisible_various_widths(self, w):
        """K=1 copy with N=100, various tile widths."""
        N = dace.symbol("N")

        @dace.program
        def copy_tw(A: dace.float64[N], B: dace.float64[N]):
            for i in range(N):
                B[i] = A[i]

        sdfg = copy_tw.to_sdfg()
        sdfg.name = _unique(f"mg_k1_tw{w}")

        n = 100  # not divisible by any of 4, 8, 16, 32
        rng = np.random.default_rng(w + 500)
        A = rng.standard_normal(n)
        B = np.zeros(n)
        _apply_and_run(sdfg, widths=(w,), A=A, B=B, N=n)
        np.testing.assert_allclose(B, A, rtol=1e-14)


@pytest.mark.gpu
class TestMaskGenCodegenVerification:
    """Verify that generated code contains mask-related cuTile primitives."""

    def test_mask_gen_emits_ct_arange(self):
        """Symbolic-sized SDFG should emit ct.arange for mask generation."""
        N = dace.symbol("N")

        @dace.program
        def copy_cg(A: dace.float64[N], B: dace.float64[N]):
            for i in range(N):
                B[i] = A[i]

        sdfg = copy_cg.to_sdfg()
        sdfg.name = _unique("mg_codegen_arange")
        VectorizeCuTile(widths=(8,)).apply_pass(sdfg, {})

        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        assert "ct.arange(8" in code, (
            "Expected ct.arange(8 in generated code for mask generation"
        )

    def test_mask_gen_emits_comparison(self):
        """Symbolic-sized SDFG should emit < comparison for mask bound check."""
        N = dace.symbol("N")

        @dace.program
        def copy_cg2(A: dace.float64[N], B: dace.float64[N]):
            for i in range(N):
                B[i] = A[i]

        sdfg = copy_cg2.to_sdfg()
        sdfg.name = _unique("mg_codegen_cmp")
        VectorizeCuTile(widths=(8,)).apply_pass(sdfg, {})

        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        # TileMaskGen emits: __offsets0 + __pid0 * W < (ub)
        assert "< (" in code or "< N" in code, (
            "Expected upper-bound comparison in generated mask code"
        )

    def test_mask_gen_emits_ct_bid(self):
        """Symbolic-sized SDFG should emit ct.bid for mask block index."""
        N = dace.symbol("N")

        @dace.program
        def copy_cg3(A: dace.float64[N], B: dace.float64[N]):
            for i in range(N):
                B[i] = A[i]

        sdfg = copy_cg3.to_sdfg()
        sdfg.name = _unique("mg_codegen_bid")
        VectorizeCuTile(widths=(8,)).apply_pass(sdfg, {})

        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        assert "ct.bid(" in code, (
            "Expected ct.bid in generated code for mask block index"
        )


# ---------------------------------------------------------------------------
# Group 2: Validation tests (no GPU required)
# ---------------------------------------------------------------------------


class TestMaskGenCppExprRejection:
    """ExpandTileMaskGenCutile rejects C++-flavored upper bounds."""

    def test_rejects_std_namespace(self):
        """``std::min(N, M)`` in global_ubs is rejected."""
        from dace.libraries.tileops.nodes.tile_mask_gen import (
            TileMaskGen,
            ExpandTileMaskGenCutile,
        )

        node = TileMaskGen(
            "test_mask_cpp",
            widths=(8,),
            iter_vars=("i",),
            global_ubs=("std::min(N, M)",),
        )
        node.implementation = "cutile"

        with pytest.raises(ValueError, match="C\\+\\+-flavored"):
            ExpandTileMaskGenCutile.expansion(node, None, None)

    def test_rejects_pointer_deref(self):
        """``ptr->size`` in global_ubs is rejected."""
        from dace.libraries.tileops.nodes.tile_mask_gen import (
            TileMaskGen,
            ExpandTileMaskGenCutile,
        )

        node = TileMaskGen(
            "test_mask_arrow",
            widths=(8,),
            iter_vars=("i",),
            global_ubs=("ptr->size",),
        )
        node.implementation = "cutile"

        with pytest.raises(ValueError, match="C\\+\\+-flavored"):
            ExpandTileMaskGenCutile.expansion(node, None, None)

    def test_rejects_sizeof(self):
        """``sizeof(int)`` in global_ubs is rejected."""
        from dace.libraries.tileops.nodes.tile_mask_gen import (
            TileMaskGen,
            ExpandTileMaskGenCutile,
        )

        node = TileMaskGen(
            "test_mask_sizeof",
            widths=(8,),
            iter_vars=("i",),
            global_ubs=("sizeof(int)",),
        )
        node.implementation = "cutile"

        with pytest.raises(ValueError, match="C\\+\\+-flavored"):
            ExpandTileMaskGenCutile.expansion(node, None, None)

    def test_rejects_trailing_semicolon(self):
        """Trailing semicolon in global_ubs is rejected."""
        from dace.libraries.tileops.nodes.tile_mask_gen import (
            TileMaskGen,
            ExpandTileMaskGenCutile,
        )

        node = TileMaskGen(
            "test_mask_semi",
            widths=(8,),
            iter_vars=("i",),
            global_ubs=("N;",),
        )
        node.implementation = "cutile"

        with pytest.raises(ValueError, match="C\\+\\+-flavored"):
            ExpandTileMaskGenCutile.expansion(node, None, None)


class TestMaskGenDtypeSelection:
    """Expansion uses ct.int64 or ct.int32 based on upper-bound analysis."""

    def test_int64_for_symbolic_ubs(self):
        """Symbolic upper bound -> ct.int64 (conservative)."""
        from dace.libraries.tileops.nodes.tile_mask_gen import (
            TileMaskGen,
            ExpandTileMaskGenCutile,
        )

        node = TileMaskGen(
            "test_mask_sym64",
            widths=(8,),
            iter_vars=("i",),
            global_ubs=("N",),
        )
        node.implementation = "cutile"

        tasklet = ExpandTileMaskGenCutile.expansion(node, None, None)
        code = tasklet.code.as_string
        assert "ct.int64" in code, (
            "Expected ct.int64 for symbolic upper bound"
        )
        assert "ct.int32" not in code

    def test_int64_for_large_concrete_ubs(self):
        """Upper bound exceeding 2^31-1 -> ct.int64."""
        from dace.libraries.tileops.nodes.tile_mask_gen import (
            TileMaskGen,
            ExpandTileMaskGenCutile,
        )

        big = str(2**31)  # 2147483648, just above int32 max
        node = TileMaskGen(
            "test_mask_big64",
            widths=(8,),
            iter_vars=("i",),
            global_ubs=(big,),
        )
        node.implementation = "cutile"

        tasklet = ExpandTileMaskGenCutile.expansion(node, None, None)
        code = tasklet.code.as_string
        assert "ct.int64" in code, (
            "Expected ct.int64 for large upper bound"
        )
        assert "ct.int32" not in code

    def test_int32_for_small_concrete_ubs(self):
        """Small concrete upper bound -> ct.int32."""
        from dace.libraries.tileops.nodes.tile_mask_gen import (
            TileMaskGen,
            ExpandTileMaskGenCutile,
        )

        node = TileMaskGen(
            "test_mask_small32",
            widths=(8,),
            iter_vars=("i",),
            global_ubs=("128",),
        )
        node.implementation = "cutile"

        tasklet = ExpandTileMaskGenCutile.expansion(node, None, None)
        code = tasklet.code.as_string
        assert "ct.int32" in code, (
            "Expected ct.int32 for small concrete upper bound"
        )
        assert "ct.int64" not in code

    def test_int64_for_expression_ubs(self):
        """Expression upper bound (e.g. ``N + 1``) -> ct.int64 (symbolic)."""
        from dace.libraries.tileops.nodes.tile_mask_gen import (
            TileMaskGen,
            ExpandTileMaskGenCutile,
        )

        node = TileMaskGen(
            "test_mask_expr64",
            widths=(8,),
            iter_vars=("i",),
            global_ubs=("N + 1",),
        )
        node.implementation = "cutile"

        tasklet = ExpandTileMaskGenCutile.expansion(node, None, None)
        code = tasklet.code.as_string
        assert "ct.int64" in code
        assert "ct.int32" not in code

    def test_int32_for_max_int32_ubs(self):
        """Upper bound exactly at 2^31-1 -> ct.int32."""
        from dace.libraries.tileops.nodes.tile_mask_gen import (
            TileMaskGen,
            ExpandTileMaskGenCutile,
        )

        node = TileMaskGen(
            "test_mask_maxint32",
            widths=(8,),
            iter_vars=("i",),
            global_ubs=(str(2**31 - 1),),
        )
        node.implementation = "cutile"

        tasklet = ExpandTileMaskGenCutile.expansion(node, None, None)
        code = tasklet.code.as_string
        assert "ct.int32" in code
        assert "ct.int64" not in code

    def test_k2_mixed_ubs(self):
        """K=2: one symbolic, one small concrete -> ct.int64 (conservative)."""
        from dace.libraries.tileops.nodes.tile_mask_gen import (
            TileMaskGen,
            ExpandTileMaskGenCutile,
        )

        node = TileMaskGen(
            "test_mask_k2_mixed",
            widths=(4, 8),
            iter_vars=("i", "j"),
            global_ubs=("128", "N"),
        )
        node.implementation = "cutile"

        tasklet = ExpandTileMaskGenCutile.expansion(node, None, None)
        code = tasklet.code.as_string
        assert "ct.int64" in code
        assert "ct.int32" not in code

    def test_k2_all_small_concrete_ubs(self):
        """K=2: both upper bounds small concrete -> ct.int32."""
        from dace.libraries.tileops.nodes.tile_mask_gen import (
            TileMaskGen,
            ExpandTileMaskGenCutile,
        )

        node = TileMaskGen(
            "test_mask_k2_small",
            widths=(4, 8),
            iter_vars=("i", "j"),
            global_ubs=("16", "32"),
        )
        node.implementation = "cutile"

        tasklet = ExpandTileMaskGenCutile.expansion(node, None, None)
        code = tasklet.code.as_string
        assert "ct.int32" in code
        assert "ct.int64" not in code


class TestMaskGenExpansionStructure:
    """Verify structural properties of the expansion output."""

    def test_k1_output_is_python_tasklet(self):
        """K=1 expansion returns a Python-language tasklet."""
        from dace.libraries.tileops.nodes.tile_mask_gen import (
            TileMaskGen,
            ExpandTileMaskGenCutile,
        )

        node = TileMaskGen(
            "test_struct_k1",
            widths=(8,),
            iter_vars=("i",),
            global_ubs=("N",),
        )
        node.implementation = "cutile"

        tasklet = ExpandTileMaskGenCutile.expansion(node, None, None)
        assert tasklet.language == dtypes.Language.Python

    def test_k1_output_connector(self):
        """K=1 expansion tasklet has ``_o`` output."""
        from dace.libraries.tileops.nodes.tile_mask_gen import (
            TileMaskGen,
            ExpandTileMaskGenCutile,
        )

        node = TileMaskGen(
            "test_struct_conn",
            widths=(8,),
            iter_vars=("i",),
            global_ubs=("N",),
        )
        node.implementation = "cutile"

        tasklet = ExpandTileMaskGenCutile.expansion(node, None, None)
        assert "_o" in tasklet.out_connectors

    def test_k2_broadcasts_and_combines(self):
        """K=2 expansion uses broadcast_to and bitwise-and (&)."""
        from dace.libraries.tileops.nodes.tile_mask_gen import (
            TileMaskGen,
            ExpandTileMaskGenCutile,
        )

        node = TileMaskGen(
            "test_struct_k2",
            widths=(4, 8),
            iter_vars=("i", "j"),
            global_ubs=("M", "N"),
        )
        node.implementation = "cutile"

        tasklet = ExpandTileMaskGenCutile.expansion(node, None, None)
        code = tasklet.code.as_string
        assert "ct.broadcast_to" in code, (
            "K=2 expansion should use ct.broadcast_to"
        )
        assert " & " in code, (
            "K=2 expansion should combine per-dim masks with &"
        )

    def test_k1_no_broadcast(self):
        """K=1 expansion should NOT use broadcast (single dim -> direct assign)."""
        from dace.libraries.tileops.nodes.tile_mask_gen import (
            TileMaskGen,
            ExpandTileMaskGenCutile,
        )

        node = TileMaskGen(
            "test_struct_k1_nobcast",
            widths=(8,),
            iter_vars=("i",),
            global_ubs=("N",),
        )
        node.implementation = "cutile"

        tasklet = ExpandTileMaskGenCutile.expansion(node, None, None)
        code = tasklet.code.as_string
        assert "ct.broadcast_to" not in code
        assert "_o = __mask0" in code

    def test_k3_three_dim_mask(self):
        """K=3 expansion produces three per-dim masks combined with &."""
        from dace.libraries.tileops.nodes.tile_mask_gen import (
            TileMaskGen,
            ExpandTileMaskGenCutile,
        )

        node = TileMaskGen(
            "test_struct_k3",
            widths=(2, 4, 8),
            iter_vars=("i", "j", "k"),
            global_ubs=("M", "N", "P"),
        )
        node.implementation = "cutile"

        tasklet = ExpandTileMaskGenCutile.expansion(node, None, None)
        code = tasklet.code.as_string
        # Should have 3 per-dim masks and 2 & operators
        assert "__mask0" in code
        assert "__mask1" in code
        assert "__mask2" in code
        assert code.count("ct.broadcast_to") == 3
        assert code.count(" & ") == 2


class TestMaskGenNodeValidation:
    """Validate TileMaskGen constructor constraints."""

    def test_rejects_k0(self):
        """K=0 (empty widths) is rejected."""
        from dace.libraries.tileops.nodes.tile_mask_gen import TileMaskGen

        with pytest.raises(ValueError, match="widths length"):
            TileMaskGen("bad_k0", widths=(), iter_vars=(), global_ubs=())

    def test_rejects_k4(self):
        """K=4 (widths too long) is rejected."""
        from dace.libraries.tileops.nodes.tile_mask_gen import TileMaskGen

        with pytest.raises(ValueError, match="widths length"):
            TileMaskGen(
                "bad_k4",
                widths=(2, 4, 8, 16),
                iter_vars=("i", "j", "k", "l"),
                global_ubs=("M", "N", "P", "Q"),
            )

    def test_rejects_length_mismatch(self):
        """Mismatched lengths between widths, iter_vars, global_ubs."""
        from dace.libraries.tileops.nodes.tile_mask_gen import TileMaskGen

        with pytest.raises(ValueError, match="lengths must agree"):
            TileMaskGen(
                "bad_mismatch",
                widths=(8,),
                iter_vars=("i", "j"),
                global_ubs=("N",),
            )


class TestMaskGenTrailingKGuard:
    """Verify the trailing-K grid-axis binding guard."""

    @staticmethod
    def _build_map_with_maskgen(map_params, map_ranges, iter_vars, global_ubs, widths):
        """Build a minimal SDFG with a CuTile map containing a TileMaskGen.

        :param map_params: list of str, the map parameter names.
        :param map_ranges: dict mapping param name to range string.
        :param iter_vars: tuple of str, TileMaskGen iter_vars.
        :param global_ubs: tuple of str, TileMaskGen global_ubs.
        :param widths: tuple of int, TileMaskGen widths.
        :returns: (node, state, sdfg) tuple for calling expansion.
        """
        from dace.libraries.tileops.nodes.tile_mask_gen import TileMaskGen

        sdfg = dace.SDFG(_unique("trailing_k_test"))
        sdfg.add_array("_mask", list(widths), dace.bool_,
                        storage=dace.StorageType.CuTile_Tile, transient=True)
        state = sdfg.add_state("s0")

        me, mx = state.add_map(
            "cutile_map",
            map_ranges,
            schedule=dace.ScheduleType.CuTile,
        )
        # Ensure params are in the specified order
        me.map.params = list(map_params)

        mask_node = TileMaskGen(
            "test_mask",
            widths=widths,
            iter_vars=iter_vars,
            global_ubs=global_ubs,
        )
        mask_node.implementation = "cutile"
        state.add_node(mask_node)

        mask_access = state.add_access("_mask")

        # Wire: map_entry -> mask_node -> mask_access -> map_exit
        # The empty memlet from me to mask_node establishes scope membership
        state.add_nedge(me, mask_node, dace.Memlet())
        state.add_edge(mask_node, "_o", mask_access, None,
                       dace.Memlet(data="_mask", subset=", ".join(f"0:{w}" for w in widths)))
        state.add_nedge(mask_access, mx, dace.Memlet())

        return mask_node, state, sdfg

    def test_rejects_non_trailing_iter_vars(self):
        """Expansion raises ValueError when iter_vars are not the trailing
        K params of the enclosing CuTile map."""
        from dace.libraries.tileops.nodes.tile_mask_gen import ExpandTileMaskGenCutile

        # Map has params ["i", "jb"] but iter_vars is ("i",)
        # The trailing param is "jb", not "i" -- mismatch
        node, state, sdfg = self._build_map_with_maskgen(
            map_params=["i", "jb"],
            map_ranges={"i": "0:16:8", "jb": "0:4"},
            iter_vars=("i",),
            global_ubs=("16",),
            widths=(8,),
        )

        with pytest.raises(ValueError, match="trailing grid axes"):
            ExpandTileMaskGenCutile.expansion(node, state, sdfg)

    def test_accepts_trailing_iter_vars(self):
        """Expansion succeeds when iter_vars are the trailing K params."""
        from dace.libraries.tileops.nodes.tile_mask_gen import ExpandTileMaskGenCutile

        # Map has params ["jb", "i"] and iter_vars is ("i",)
        # The trailing param IS "i" -- match
        node, state, sdfg = self._build_map_with_maskgen(
            map_params=["jb", "i"],
            map_ranges={"jb": "0:4", "i": "0:16:8"},
            iter_vars=("i",),
            global_ubs=("16",),
            widths=(8,),
        )

        tasklet = ExpandTileMaskGenCutile.expansion(node, state, sdfg)
        assert tasklet is not None
        assert hasattr(tasklet, 'code')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=300"])
