# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for return-value handling in the Python backend (PythonCompiledSDFG).

Covers single return, multiple returns, symbolic sizes, positional arg
mapping, explicit __return passing, do_not_execute, and convert_return_values.
"""

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.codegen.py.compiled_sdfg import PythonCompiledSDFG, _is_return_array_name

N = dace.symbol('N')


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _compile_python(sdfg):
    """Set the backend to Python and compile."""
    sdfg.backend = dtypes.BackendLanguage.Python
    return sdfg.compile()


# ---------------------------------------------------------------------------
# Single return value
# ---------------------------------------------------------------------------

class TestSingleReturn:
    """Tests for SDFGs that return a single array."""

    def test_single_return_keyword_args(self):
        """Single return via keyword arguments."""
        @dace.program
        def add_one(A: dace.float64[8]):
            return A + 1.0

        sdfg = add_one.to_sdfg()
        csdfg = _compile_python(sdfg)
        A = np.arange(8, dtype=np.float64)
        result = csdfg(A=A)
        np.testing.assert_allclose(result, A + 1.0)
        assert isinstance(result, np.ndarray)

    def test_single_return_positional_args(self):
        """Single return via positional arguments."""
        @dace.program
        def add_one(A: dace.float64[8]):
            return A + 1.0

        sdfg = add_one.to_sdfg()
        csdfg = _compile_python(sdfg)
        A = np.arange(8, dtype=np.float64)
        result = csdfg(A)
        np.testing.assert_allclose(result, A + 1.0)

    def test_single_return_scalar_constant(self):
        """Return a constant scalar value (shape (1,))."""
        @dace.program
        def return_42():
            return 42

        sdfg = return_42.to_sdfg(simplify=False)
        csdfg = _compile_python(sdfg)
        result = csdfg()
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert result[0] == 42

    def test_single_return_symbolic_size(self):
        """Return array with symbolic size."""
        @dace.program
        def scale(A: dace.float64[N]):
            return A * 3.0

        sdfg = scale.to_sdfg()
        csdfg = _compile_python(sdfg)
        for size in [1, 5, 16]:
            A = np.arange(size, dtype=np.float64)
            result = csdfg(A=A, N=size)
            np.testing.assert_allclose(result, A * 3.0)

    def test_single_return_does_not_modify_input(self):
        """The input array should not be modified."""
        @dace.program
        def add_one(A: dace.float64[8]):
            return A + 1.0

        sdfg = add_one.to_sdfg()
        csdfg = _compile_python(sdfg)
        A = np.arange(8, dtype=np.float64)
        A_copy = A.copy()
        csdfg(A=A)
        np.testing.assert_array_equal(A, A_copy)

    def test_single_return_explicit_return_buffer(self):
        """User passes __return explicitly as a keyword argument."""
        @dace.program
        def add_one(A: dace.float64[8]):
            return A + 1.0

        sdfg = add_one.to_sdfg()
        csdfg = _compile_python(sdfg)
        A = np.arange(8, dtype=np.float64)
        ret = np.zeros(8, dtype=np.float64)
        result = csdfg(A=A, __return=ret)
        # The user-supplied buffer should be used
        np.testing.assert_allclose(ret, A + 1.0)
        assert result is ret


# ---------------------------------------------------------------------------
# Multiple return values
# ---------------------------------------------------------------------------

class TestMultipleReturns:
    """Tests for SDFGs that return multiple arrays."""

    def test_two_returns(self):
        """Return two arrays as a tuple."""
        @dace.program
        def two_ret(A: dace.float64[8]):
            return A + 1.0, A * 2.0

        sdfg = two_ret.to_sdfg()
        csdfg = _compile_python(sdfg)
        A = np.arange(8, dtype=np.float64)
        result = csdfg(A=A)
        assert isinstance(result, tuple)
        assert len(result) == 2
        np.testing.assert_allclose(result[0], A + 1.0)
        np.testing.assert_allclose(result[1], A * 2.0)

    def test_three_returns(self):
        """Return three arrays as a tuple."""
        @dace.program
        def three_ret(A: dace.float64[4]):
            return A + 1.0, A * 2.0, A - 1.0

        sdfg = three_ret.to_sdfg()
        csdfg = _compile_python(sdfg)
        A = np.array([1.0, 2.0, 3.0, 4.0])
        result = csdfg(A=A)
        assert isinstance(result, tuple)
        assert len(result) == 3
        np.testing.assert_allclose(result[0], A + 1.0)
        np.testing.assert_allclose(result[1], A * 2.0)
        np.testing.assert_allclose(result[2], A - 1.0)

    def test_multiple_returns_symbolic_size(self):
        """Multiple returns with symbolic sizes."""
        @dace.program
        def sym_multi(A: dace.float64[N]):
            return A + 1.0, A * 2.0

        sdfg = sym_multi.to_sdfg()
        csdfg = _compile_python(sdfg)
        for size in [1, 5, 16]:
            A = np.arange(size, dtype=np.float64)
            result = csdfg(A=A, N=size)
            assert isinstance(result, tuple)
            assert len(result) == 2
            np.testing.assert_allclose(result[0], A + 1.0)
            np.testing.assert_allclose(result[1], A * 2.0)


# ---------------------------------------------------------------------------
# No return value
# ---------------------------------------------------------------------------

class TestNoReturn:
    """Tests for SDFGs that have no return values."""

    def test_no_return_modifies_output_in_place(self):
        """SDFGs without return values write to output arrays in-place."""
        @dace.program
        def no_ret(A: dace.float64[4], B: dace.float64[4]):
            for i in range(4):
                B[i] = A[i] + 1.0

        sdfg = no_ret.to_sdfg(simplify=False)
        csdfg = _compile_python(sdfg)
        A = np.array([1.0, 2.0, 3.0, 4.0])
        B = np.zeros(4, dtype=np.float64)
        result = csdfg(A=A, B=B)
        # No return values -> returns None
        assert result is None
        np.testing.assert_allclose(B, A + 1.0)


# ---------------------------------------------------------------------------
# do_not_execute
# ---------------------------------------------------------------------------

class TestDoNotExecute:
    """Tests for do_not_execute flag with return values."""

    def test_do_not_execute_returns_allocated_arrays(self):
        """With do_not_execute, return values are allocated but not computed."""
        @dace.program
        def add_one(A: dace.float64[4]):
            return A + 1.0

        sdfg = add_one.to_sdfg()
        csdfg = _compile_python(sdfg)
        csdfg.do_not_execute = True
        A = np.array([1.0, 2.0, 3.0, 4.0])
        result = csdfg(A=A)
        # Return value is still an array (allocated), even though not computed
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
        assert result.dtype == np.float64

    def test_do_not_execute_no_return(self):
        """With do_not_execute and no return arrays, returns None."""
        @dace.program
        def no_ret(A: dace.float64[4], B: dace.float64[4]):
            for i in range(4):
                B[i] = A[i] + 1.0

        sdfg = no_ret.to_sdfg(simplify=False)
        csdfg = _compile_python(sdfg)
        csdfg.do_not_execute = True
        A = np.array([1.0, 2.0, 3.0, 4.0])
        B = np.zeros(4, dtype=np.float64)
        result = csdfg(A=A, B=B)
        assert result is None

    def test_do_not_execute_multiple_returns(self):
        """With do_not_execute and multiple returns, returns tuple of arrays."""
        @dace.program
        def two_ret(A: dace.float64[4]):
            return A + 1.0, A * 2.0

        sdfg = two_ret.to_sdfg()
        csdfg = _compile_python(sdfg)
        csdfg.do_not_execute = True
        A = np.array([1.0, 2.0, 3.0, 4.0])
        result = csdfg(A=A)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Positional argument mapping
# ---------------------------------------------------------------------------

class TestPositionalArgs:
    """Tests for positional argument conversion with return values."""

    def test_positional_single_return(self):
        """Positional arg for input, return auto-allocated."""
        @dace.program
        def add_one(A: dace.float64[4]):
            return A + 1.0

        sdfg = add_one.to_sdfg()
        csdfg = _compile_python(sdfg)
        A = np.array([1.0, 2.0, 3.0, 4.0])
        result = csdfg(A)
        np.testing.assert_allclose(result, A + 1.0)

    def test_positional_explicit_return(self):
        """Positional arg maps to __return if no other args exist."""
        @dace.program
        def return_42():
            return 42

        sdfg = return_42.to_sdfg(simplify=False)
        csdfg = _compile_python(sdfg)
        ret = np.zeros(1, dtype=np.int64)
        # Positional arg maps to __return via arglist keys
        csdfg(ret)
        assert ret[0] == 42

    def test_positional_and_keyword_overlap_error(self):
        """Error when same arg is both positional and keyword."""
        @dace.program
        def add_one(A: dace.float64[4]):
            return A + 1.0

        sdfg = add_one.to_sdfg()
        csdfg = _compile_python(sdfg)
        A = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="positional and keyword"):
            csdfg(A, A=A)


# ---------------------------------------------------------------------------
# _has_returns (cached attribute)
# ---------------------------------------------------------------------------

class TestHasReturns:
    """Tests for the _has_returns cached attribute."""

    def test_no_returns(self):
        """SDFG with no __return arrays."""
        @dace.program
        def no_ret(A: dace.float64[4], B: dace.float64[4]):
            for i in range(4):
                B[i] = A[i]

        sdfg = no_ret.to_sdfg(simplify=False)
        csdfg = _compile_python(sdfg)
        assert not csdfg._has_returns

    def test_single_return(self):
        """SDFG with __return."""
        @dace.program
        def single():
            return 42

        sdfg = single.to_sdfg(simplify=False)
        csdfg = _compile_python(sdfg)
        assert csdfg._has_returns

    def test_multiple_returns(self):
        """SDFG with __return_0, __return_1."""
        @dace.program
        def multi(A: dace.float64[4]):
            return A + 1.0, A * 2.0

        sdfg = multi.to_sdfg()
        csdfg = _compile_python(sdfg)
        assert csdfg._has_returns


# ---------------------------------------------------------------------------
# Integration: end-to-end through @dace.program
# ---------------------------------------------------------------------------

class TestIntegration:
    """End-to-end integration tests through @dace.program."""

    def test_return_via_sdfg_call(self):
        """SDFG.__call__ returns the computed value (single return)."""
        @dace.program
        def add_one(A: dace.float64[8]):
            return A + 1.0

        sdfg = add_one.to_sdfg()
        sdfg.backend = dtypes.BackendLanguage.Python
        A = np.arange(8, dtype=np.float64)
        result = sdfg(A=A)
        np.testing.assert_allclose(result, A + 1.0)

    def test_multi_return_via_sdfg_call(self):
        """SDFG.__call__ returns a tuple for multiple returns."""
        @dace.program
        def two_ret(A: dace.float64[4]):
            return A + 1.0, A * 2.0

        sdfg = two_ret.to_sdfg()
        sdfg.backend = dtypes.BackendLanguage.Python
        A = np.array([1.0, 2.0, 3.0, 4.0])
        result = sdfg(A=A)
        assert isinstance(result, tuple)
        assert len(result) == 2
        np.testing.assert_allclose(result[0], A + 1.0)
        np.testing.assert_allclose(result[1], A * 2.0)

    def test_return_with_computation(self):
        """Return after non-trivial computation."""
        @dace.program
        def compute(A: dace.float64[N], B: dace.float64[N]):
            return A + B

        sdfg = compute.to_sdfg()
        sdfg.backend = dtypes.BackendLanguage.Python
        A = np.array([1.0, 2.0, 3.0])
        B = np.array([4.0, 5.0, 6.0])
        result = sdfg(A=A, B=B, N=3)
        np.testing.assert_allclose(result, A + B)

    def test_return_preserves_dtype_int64(self):
        """Return preserves integer dtypes (int64)."""
        @dace.program
        def int_op(A: dace.int64[4]):
            return A + np.int64(1)

        sdfg = int_op.to_sdfg()
        sdfg.backend = dtypes.BackendLanguage.Python
        A = np.array([1, 2, 3, 4], dtype=np.int64)
        result = sdfg(A=A)
        assert result.dtype == np.int64
        np.testing.assert_array_equal(result, A + 1)

    def test_repeated_calls_same_compiled_sdfg(self):
        """Multiple calls to the same compiled SDFG produce correct results."""
        @dace.program
        def add_one(A: dace.float64[4]):
            return A + 1.0

        sdfg = add_one.to_sdfg()
        csdfg = _compile_python(sdfg)

        for i in range(3):
            A = np.array([float(i), float(i+1), float(i+2), float(i+3)])
            result = csdfg(A=A)
            np.testing.assert_allclose(result, A + 1.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for return-value handling."""

    def test_size_one_array_return(self):
        """Return a single-element array."""
        @dace.program
        def ret_one():
            return 42

        sdfg = ret_one.to_sdfg(simplify=False)
        csdfg = _compile_python(sdfg)
        result = csdfg()
        assert result.shape == (1,)
        assert result[0] == 42

    def test_empty_sdfg_no_crash(self):
        """SDFG with no arrays and no return values doesn't crash."""
        sdfg = dace.SDFG('empty_test')
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.add_state('s')
        code = "def empty_test(): pass\n"
        csdfg = PythonCompiledSDFG(sdfg, code)
        result = csdfg()
        assert result is None

    def test_return_different_symbolic_sizes(self):
        """Return array size changes between calls with different symbols."""
        @dace.program
        def scale(A: dace.float64[N]):
            return A * 2.0

        sdfg = scale.to_sdfg()
        csdfg = _compile_python(sdfg)

        # Call with N=4
        A4 = np.arange(4, dtype=np.float64)
        r4 = csdfg(A=A4, N=4)
        np.testing.assert_allclose(r4, A4 * 2.0)

        # Call again with N=8 (different symbolic size)
        csdfg.finalize()
        A8 = np.arange(8, dtype=np.float64)
        r8 = csdfg(A=A8, N=8)
        np.testing.assert_allclose(r8, A8 * 2.0)
        assert r8.shape == (8,)


# ---------------------------------------------------------------------------
# Parity with C++ backend return_value_test.py
# ---------------------------------------------------------------------------

class TestCppBackendParity:
    """Mirror test cases from tests/python_frontend/return_value_test.py
    to verify the Python backend produces identical results."""

    def test_return_scalar_constant_5(self):
        """C++ parity: return scalar constant 5."""
        @dace.program
        def return_scalar():
            return 5

        sdfg = return_scalar.to_sdfg(simplify=False)
        csdfg = _compile_python(sdfg)
        res = csdfg()
        assert res == 5
        assert isinstance(res, np.ndarray)
        assert res.shape == (1,)
        assert res.dtype == np.int64

    def test_return_tuple_scalars(self):
        """C++ parity: return tuple of scalars (5, 6)."""
        @dace.program
        def return_tuple():
            return 5, 6

        sdfg = return_tuple.to_sdfg(simplify=False)
        csdfg = _compile_python(sdfg)
        res = csdfg()
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert res[0] == 5
        assert res[1] == 6

    def test_return_array_tuple_different_sizes(self):
        """C++ parity: return tuple of arrays with different sizes."""
        @dace.program
        def return_array_tuple(A: dace.float64[5], B: dace.float64[6]):
            return A + 1.0, B + 2.0

        sdfg = return_array_tuple.to_sdfg()
        csdfg = _compile_python(sdfg)
        A = np.ones(5, dtype=np.float64)
        B = np.ones(6, dtype=np.float64)
        res = csdfg(A=A, B=B)
        assert isinstance(res, tuple)
        assert len(res) == 2
        np.testing.assert_allclose(res[0], A + 1.0)
        np.testing.assert_allclose(res[1], B + 2.0)
        assert res[0].shape == (5,)
        assert res[1].shape == (6,)

    def test_return_constant_array(self):
        """C++ parity: return array computed from input."""
        @dace.program
        def return_array(A: dace.float64[5]):
            return A + 1.0

        sdfg = return_array.to_sdfg()
        csdfg = _compile_python(sdfg)
        A = np.ones(5, dtype=np.float64) * 4.0
        res = csdfg(A=A)
        np.testing.assert_allclose(res, A + 1.0)

    def test_return_tuple_1_element(self):
        """C++ parity: return single-element tuple (not scalar)."""
        @dace.program
        def return_one_element_tuple(a: dace.float64[20]):
            return (a + 3.5,)

        sdfg = return_one_element_tuple.to_sdfg()
        csdfg = _compile_python(sdfg)
        a = np.random.default_rng(42).random(20)
        ref = a + 3.5
        res = csdfg(a=a)
        assert isinstance(res, tuple)
        assert len(res) == 1
        np.testing.assert_allclose(res[0], ref)

    def test_return_void_early_return(self):
        """C++ parity: void return with in-place mutation."""
        @dace.program
        def return_void(a: dace.float64[20]):
            a[:] += 1
            return
            a[:] = 5

        sdfg = return_void.to_sdfg(simplify=False)
        csdfg = _compile_python(sdfg)
        a = np.random.default_rng(42).random(20)
        ref = a + 1
        res = csdfg(a=a)
        assert res is None
        np.testing.assert_allclose(a, ref)


# ---------------------------------------------------------------------------
# Additional coverage: dtype, caching, multi-arg positional
# ---------------------------------------------------------------------------

class TestAdditionalCoverage:
    """Additional tests for edge cases and coverage gaps."""

    def test_return_float32(self):
        """Return preserves float32 dtype."""
        @dace.program
        def f32_op(A: dace.float32[4], B: dace.float32[4]):
            return A + B

        sdfg = f32_op.to_sdfg()
        csdfg = _compile_python(sdfg)
        A = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        B = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        result = csdfg(A=A, B=B)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, A + B, rtol=1e-6)

    def test_two_positional_inputs_with_return(self):
        """Two positional input arrays with an auto-allocated return."""
        @dace.program
        def add_arrays(A: dace.float64[4], B: dace.float64[4]):
            return A + B

        sdfg = add_arrays.to_sdfg()
        csdfg = _compile_python(sdfg)
        A = np.array([1.0, 2.0, 3.0, 4.0])
        B = np.array([10.0, 20.0, 30.0, 40.0])
        result = csdfg(A, B)
        np.testing.assert_allclose(result, A + B)

    def test_repeated_calls_same_symbols_correct(self):
        """Repeated calls with same symbols produce correct results."""
        @dace.program
        def scale(A: dace.float64[N]):
            return A * 2.0

        sdfg = scale.to_sdfg()
        csdfg = _compile_python(sdfg)

        A1 = np.arange(5, dtype=np.float64)
        r1 = csdfg(A=A1, N=5)
        np.testing.assert_allclose(r1, A1 * 2.0)

        # Second call with same N=5
        A2 = np.ones(5, dtype=np.float64)
        r2 = csdfg(A=A2, N=5)
        np.testing.assert_allclose(r2, A2 * 2.0)

        # Results should be independent
        np.testing.assert_allclose(r1, A1 * 2.0)

    def test_is_single_value_ret_flag(self):
        """_is_single_value_ret is True for single return, False for multi."""
        @dace.program
        def single(A: dace.float64[4]):
            return A + 1.0

        @dace.program
        def multi(A: dace.float64[4]):
            return A + 1.0, A * 2.0

        csdfg_single = _compile_python(single.to_sdfg())
        csdfg_multi = _compile_python(multi.to_sdfg())

        assert csdfg_single._is_single_value_ret is True
        assert csdfg_multi._is_single_value_ret is False

    def test_return_large_array(self):
        """Return a larger array to verify allocation works at scale."""
        @dace.program
        def identity(A: dace.float64[N]):
            return A + 0.0

        sdfg = identity.to_sdfg()
        csdfg = _compile_python(sdfg)
        A = np.random.default_rng(42).random(10000)
        result = csdfg(A=A, N=10000)
        np.testing.assert_allclose(result, A)
        assert result.shape == (10000,)

    def test_return_2d_array(self):
        """Return a 2D array."""
        M = dace.symbol('M')

        @dace.program
        def add_2d(A: dace.float64[M, N]):
            return A + 1.0

        sdfg = add_2d.to_sdfg()
        csdfg = _compile_python(sdfg)
        A = np.arange(12, dtype=np.float64).reshape(3, 4)
        result = csdfg(A=A, M=3, N=4)
        np.testing.assert_allclose(result, A + 1.0)
        assert result.shape == (3, 4)

    def test_return_result_is_independent_copy(self):
        """Each call returns a new independent array (not aliasing prior results)."""
        @dace.program
        def add_one(A: dace.float64[4]):
            return A + 1.0

        sdfg = add_one.to_sdfg()
        csdfg = _compile_python(sdfg)

        A1 = np.array([1.0, 2.0, 3.0, 4.0])
        r1 = csdfg(A=A1)
        r1_copy = r1.copy()

        A2 = np.array([10.0, 20.0, 30.0, 40.0])
        r2 = csdfg(A=A2)

        # r1 should not be modified by the second call
        np.testing.assert_allclose(r1, r1_copy)
        np.testing.assert_allclose(r2, A2 + 1.0)


# ---------------------------------------------------------------------------
# GPU return value tests
# ---------------------------------------------------------------------------

class TestGpuReturn:
    """GPU return value tests (require cupy and GPU hardware)."""

    @pytest.mark.gpu
    def test_allocate_return_array_gpu_storage(self):
        """_allocate_return_array allocates a cupy array for GPU_Global storage."""
        try:
            import cupy
        except ImportError:
            pytest.skip("cupy not installed")

        sdfg = dace.SDFG('gpu_ret_test')
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.add_array('__return', [8], dace.float64,
                        storage=dtypes.StorageType.GPU_Global)
        sdfg.add_state('s')
        code = "def gpu_ret_test(**kwargs): pass\n"
        csdfg = PythonCompiledSDFG(sdfg, code)

        arr = csdfg._allocate_return_array('__return', {})
        assert isinstance(arr, cupy.ndarray)
        assert arr.shape == (8,)
        assert arr.dtype == np.float64


# ---------------------------------------------------------------------------
# Regression: __return coexisting with __return_tile* transients (Bug 13)
# ---------------------------------------------------------------------------

class TestReturnTilePrefixTransients:
    """Regression tests for Bug 13.

    The cuTile vectorizer introduces transients that share the ``__return``
    prefix (e.g. ``__return_tile_out``) when a program's return value is
    written through a tile kernel.  These are NOT genuine return values and
    must not (a) trip the single-vs-tuple return assertion in
    ``PythonCompiledSDFG.__init__`` nor (b) be marshaled as return values.
    """

    def test_is_return_array_name_classification(self):
        """Only ``__return`` / ``__return_<int>`` count as return arrays."""
        assert _is_return_array_name('__return')
        assert _is_return_array_name('__return_0')
        assert _is_return_array_name('__return_12')
        # Tile transients sharing the prefix are excluded.
        assert not _is_return_array_name('__return_tile')
        assert not _is_return_array_name('__return_tile_out')
        assert not _is_return_array_name('__return_tile_0')
        # Unrelated names.
        assert not _is_return_array_name('A')
        assert not _is_return_array_name('return')

    def test_return_plus_tile_transient_no_assertion(self):
        """``__return`` next to a ``__return_tile_out`` transient must compile.

        Before Bug 13 was fixed the ``startswith('__return_')`` check treated
        the tile transient as a conflicting tuple-return element and tripped an
        ``AssertionError`` during ``PythonCompiledSDFG.__init__``.
        """
        sdfg = dace.SDFG('ret_tile_test')
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.add_array('__return', [8], dace.float64)
        sdfg.add_transient('__return_tile_out', [8], dace.float64)
        sdfg.add_state('s')
        code = "def ret_tile_test(**kwargs): pass\n"

        # Must not raise AssertionError.
        csdfg = PythonCompiledSDFG(sdfg, code)

        # The tile transient is not a return value: single-return semantics
        # hold and only __return is marshaled.
        assert csdfg._is_single_value_ret is True
        assert csdfg._has_returns is True
        assert csdfg._get_return_names() == ['__return']

    def test_return_tile_transient_only_is_not_a_return(self):
        """A lone ``__return_tile*`` transient does not count as a return."""
        sdfg = dace.SDFG('tile_only_test')
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.add_transient('__return_tile_out', [8], dace.float64)
        sdfg.add_state('s')
        code = "def tile_only_test(**kwargs): pass\n"
        csdfg = PythonCompiledSDFG(sdfg, code)
        assert csdfg._has_returns is False
        assert csdfg._get_return_names() == []


class TestCuTileReturnIntegration:
    """End-to-end cuTile pipeline: a returned value written through a tile
    kernel must marshal correctly despite the ``__return_tile_out`` transient.
    """

    @pytest.mark.gpu
    def test_cutile_single_return_2d(self):
        """`compute`-style kernel lowered to cuTile returns the right array."""
        from dace.transformation.passes.vectorization import VectorizeCuTile

        M = dace.symbol('M')
        Nn = dace.symbol('N')

        @dace.program
        def compute_kernel(array_1: dace.int64[M, Nn], array_2: dace.int64[M, Nn],
                           a: dace.int64, b: dace.int64, c: dace.int64):
            return np.minimum(np.maximum(array_1, 2), 10) * a + array_2 * b + c

        sdfg = compute_kernel.to_sdfg(simplify=False)
        VectorizeCuTile(widths=(8, 8)).apply_pass(sdfg, {})

        # The offending dual naming must be present to exercise the regression.
        return_prefixed = [n for n in sdfg.arrays if n.startswith('__return')]
        assert '__return' in return_prefixed
        assert any(n.startswith('__return_tile') for n in return_prefixed)

        csdfg = sdfg.compile()

        rng = np.random.default_rng(0)
        mm, nn = 16, 16
        a1 = rng.integers(0, 20, (mm, nn)).astype(np.int64)
        a2 = rng.integers(0, 20, (mm, nn)).astype(np.int64)
        a, b, c = 3, 4, 5
        res = csdfg(array_1=a1.copy(), array_2=a2.copy(), a=a, b=b, c=c, M=mm, N=nn)
        r = np.asarray(res.get() if hasattr(res, 'get') else res)
        exp = np.minimum(np.maximum(a1, 2), 10) * a + a2 * b + c
        np.testing.assert_array_equal(r, exp)
        assert r.shape == (mm, nn)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
