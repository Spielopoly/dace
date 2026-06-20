# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Integration tests for CuPy expansions of all BLAS library nodes in DaCe.

Each test builds an SDFG (manually or via @dace.program), sets the library
node's implementation to 'CuPy', applies GPU transformations, compiles,
runs with concrete numpy arrays, and compares the result to a numpy reference.
"""
import numpy as np
import pytest

import dace
from dace.libraries.blas import Gemm, Gemv, Dot, BatchedMatMul
from dace.libraries.blas.nodes.ger import Ger
from dace.libraries.blas.nodes.axpy import Axpy
from dace.libraries.blas.nodes.einsum import Einsum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_and_set_impl(sdfg, node_type, impl):
    """Find all nodes of *node_type* in *sdfg* and stamp *impl*."""
    for n, _ in sdfg.all_nodes_recursive():
        if isinstance(n, node_type):
            n.implementation = impl


def _create_gemm_sdfg(dtype, A_shape, B_shape, Y_shape, transA, transB,
                       alpha, beta, C_shape=None):
    """Build a Gemm SDFG following the pattern in gemm_test.py.

    When *beta* != 0 and *C_shape* is not None the C matrix is broadcast-added.
    When *C_shape* is None and *beta* != 0 the output itself serves as C.
    """
    sdfg_name = 'cupy_gemm_test'
    sdfg = dace.SDFG(sdfg_name)
    state = sdfg.add_state()

    dace_dtype = dace.dtype_to_typeclass(dtype)

    A, A_arr = sdfg.add_array('A', A_shape, dace_dtype)
    B, B_arr = sdfg.add_array('B', B_shape, dace_dtype)
    # Output is always Y_shape
    actual_C_shape = C_shape if C_shape is not None else Y_shape
    C, C_arr = sdfg.add_array('C', Y_shape, dace_dtype)

    rA = state.add_read('A')
    rB = state.add_read('B')
    wC = state.add_write('C')

    libnode = Gemm('_Gemm_', transA=transA, transB=transB, alpha=alpha,
                   beta=beta)
    libnode.implementation = 'CuPy'
    state.add_node(libnode)

    state.add_edge(rA, None, libnode, '_a',
                   dace.Memlet.from_array(A, A_arr))
    state.add_edge(rB, None, libnode, '_b',
                   dace.Memlet.from_array(B, B_arr))
    state.add_edge(libnode, '_c', wC, None,
                   dace.Memlet.from_array(C, C_arr))
    if beta != 0.0:
        rC = state.add_read('C')
        state.add_edge(rC, None, libnode, '_c',
                       dace.Memlet.from_array(C, C_arr))

    return sdfg


def _numpy_gemm(A, B, C, transA, transB, alpha, beta):
    A_t = A.T if transA else A
    B_t = B.T if transB else B
    result = alpha * (A_t @ B_t)
    if C is not None and beta != 0:
        result = result + beta * C
    return result


# ---------------------------------------------------------------------------
# Gemm tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestCuPyGemm:

    def test_basic(self):
        """A @ B, no transpose, alpha=1, beta=0."""
        M, K, N = 25, 23, 24
        A = np.random.rand(M, K).astype(np.float32)
        B = np.random.rand(K, N).astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)

        sdfg = _create_gemm_sdfg(np.float32, [M, K], [K, N], [M, N],
                                  transA=False, transB=False, alpha=1.0,
                                  beta=0.0)
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, Gemm, 'CuPy')

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C)
        del csdfg

        expected = A @ B
        assert np.allclose(C, expected, rtol=1e-5)

    def test_transA(self):
        """transA=True: C = A^T @ B."""
        M, K, N = 25, 23, 24
        A = np.random.rand(K, M).astype(np.float32)
        B = np.random.rand(K, N).astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)

        sdfg = _create_gemm_sdfg(np.float32, [K, M], [K, N], [M, N],
                                  transA=True, transB=False, alpha=1.0,
                                  beta=0.0)
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, Gemm, 'CuPy')

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C)
        del csdfg

        expected = A.T @ B
        assert np.allclose(C, expected, rtol=1e-5)

    def test_transB(self):
        """transB=True: C = A @ B^T."""
        M, K, N = 25, 23, 24
        A = np.random.rand(M, K).astype(np.float32)
        B = np.random.rand(N, K).astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)

        sdfg = _create_gemm_sdfg(np.float32, [M, K], [N, K], [M, N],
                                  transA=False, transB=True, alpha=1.0,
                                  beta=0.0)
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, Gemm, 'CuPy')

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C)
        del csdfg

        expected = A @ B.T
        assert np.allclose(C, expected, rtol=1e-5)

    def test_transAB(self):
        """transA=True, transB=True: C = A^T @ B^T."""
        M, K, N = 25, 23, 24
        A = np.random.rand(K, M).astype(np.float32)
        B = np.random.rand(N, K).astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)

        sdfg = _create_gemm_sdfg(np.float32, [K, M], [N, K], [M, N],
                                  transA=True, transB=True, alpha=1.0,
                                  beta=0.0)
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, Gemm, 'CuPy')

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C)
        del csdfg

        expected = A.T @ B.T
        assert np.allclose(C, expected, rtol=1e-5)

    @pytest.mark.parametrize('alpha,beta', [
        (1.0, 0.0),
        (1.0, 1.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (2.0, 3.0),
    ])
    def test_alpha_beta(self, alpha, beta):
        """Various alpha/beta scalars."""
        M, K, N = 25, 23, 24
        A = np.random.rand(M, K).astype(np.float32)
        B = np.random.rand(K, N).astype(np.float32)
        C_init = np.random.rand(M, N).astype(np.float32) if beta != 0 else None
        C = C_init.copy() if C_init is not None else np.zeros((M, N),
                                                               dtype=np.float32)

        sdfg = _create_gemm_sdfg(np.float32, [M, K], [K, N], [M, N],
                                  transA=False, transB=False, alpha=alpha,
                                  beta=beta)
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, Gemm, 'CuPy')

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C)
        del csdfg

        expected = _numpy_gemm(A, B, C_init, False, False, alpha, beta)
        assert np.allclose(C, expected, rtol=1e-5)

    def test_complex64(self):
        """complex64 dtype."""
        M, K, N = 10, 8, 12
        A = (np.random.rand(M, K) + 1j * np.random.rand(M, K)).astype(
            np.complex64)
        B = (np.random.rand(K, N) + 1j * np.random.rand(K, N)).astype(
            np.complex64)
        C = np.zeros((M, N), dtype=np.complex64)

        sdfg = _create_gemm_sdfg(np.complex64, [M, K], [K, N], [M, N],
                                  transA=False, transB=False, alpha=1.0,
                                  beta=0.0)
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, Gemm, 'CuPy')

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C)
        del csdfg

        expected = A @ B
        assert np.allclose(C, expected, rtol=1e-5)

    def test_float64(self):
        """float64 dtype."""
        M, K, N = 10, 8, 12
        A = np.random.rand(M, K).astype(np.float64)
        B = np.random.rand(K, N).astype(np.float64)
        C = np.zeros((M, N), dtype=np.float64)

        sdfg = _create_gemm_sdfg(np.float64, [M, K], [K, N], [M, N],
                                  transA=False, transB=False, alpha=1.0,
                                  beta=0.0)
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, Gemm, 'CuPy')

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C)
        del csdfg

        expected = A @ B
        assert np.allclose(C, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Gemv tests
# ---------------------------------------------------------------------------

def _create_gemv_sdfg(dtype, M, N, transA, alpha, beta):
    """Build a Gemv SDFG manually."""
    dace_dtype = dace.dtype_to_typeclass(dtype)
    sdfg = dace.SDFG('cupy_gemv_test')
    state = sdfg.add_state()

    A_shape = [M, N]
    x_shape = [N] if not transA else [M]
    y_shape = [M] if not transA else [N]

    sdfg.add_array('A', A_shape, dace_dtype)
    sdfg.add_array('x', x_shape, dace_dtype)
    sdfg.add_array('y', y_shape, dace_dtype)

    rA = state.add_read('A')
    rx = state.add_read('x')
    wy = state.add_write('y')

    libnode = Gemv('gemv', transA=transA, alpha=alpha, beta=beta)
    libnode.implementation = 'CuPy'
    state.add_node(libnode)

    state.add_edge(rA, None, libnode, '_A',
                   dace.Memlet.from_array('A', sdfg.arrays['A']))
    state.add_edge(rx, None, libnode, '_x',
                   dace.Memlet.from_array('x', sdfg.arrays['x']))
    state.add_edge(libnode, '_y', wy, None,
                   dace.Memlet.from_array('y', sdfg.arrays['y']))

    if beta != 0:
        ry = state.add_read('y')
        state.add_edge(ry, None, libnode, '_y',
                       dace.Memlet.from_array('y', sdfg.arrays['y']))

    return sdfg


@pytest.mark.gpu
class TestCuPyGemv:

    def test_basic(self):
        """y = A @ x."""
        M, N = 20, 15
        A = np.random.rand(M, N).astype(np.float32)
        x = np.random.rand(N).astype(np.float32)
        y = np.zeros(M, dtype=np.float32)

        sdfg = _create_gemv_sdfg(np.float32, M, N, transA=False, alpha=1,
                                  beta=0)
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, Gemv, 'CuPy')

        csdfg = sdfg.compile()
        csdfg(A=A, x=x, y=y)
        del csdfg

        expected = A @ x
        assert np.allclose(y, expected, rtol=1e-5)

    def test_transA(self):
        """y = A^T @ x."""
        M, N = 20, 15
        A = np.random.rand(M, N).astype(np.float32)
        x = np.random.rand(M).astype(np.float32)
        y = np.zeros(N, dtype=np.float32)

        sdfg = _create_gemv_sdfg(np.float32, M, N, transA=True, alpha=1,
                                  beta=0)
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, Gemv, 'CuPy')

        csdfg = sdfg.compile()
        csdfg(A=A, x=x, y=y)
        del csdfg

        expected = A.T @ x
        assert np.allclose(y, expected, rtol=1e-5)

    def test_alpha_beta(self):
        """y = 2.0 * A @ x + 3.0 * y."""
        M, N = 20, 15
        A = np.random.rand(M, N).astype(np.float32)
        x = np.random.rand(N).astype(np.float32)
        y_init = np.random.rand(M).astype(np.float32)
        y = y_init.copy()

        sdfg = _create_gemv_sdfg(np.float32, M, N, transA=False, alpha=2.0,
                                  beta=3.0)
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, Gemv, 'CuPy')

        csdfg = sdfg.compile()
        csdfg(A=A, x=x, y=y)
        del csdfg

        expected = 2.0 * (A @ x) + 3.0 * y_init
        assert np.allclose(y, expected, rtol=1e-5)

    def test_float64(self):
        """float64 dtype."""
        M, N = 20, 15
        A = np.random.rand(M, N).astype(np.float64)
        x = np.random.rand(N).astype(np.float64)
        y = np.zeros(M, dtype=np.float64)

        sdfg = _create_gemv_sdfg(np.float64, M, N, transA=False, alpha=1,
                                  beta=0)
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, Gemv, 'CuPy')

        csdfg = sdfg.compile()
        csdfg(A=A, x=x, y=y)
        del csdfg

        expected = A @ x
        assert np.allclose(y, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Dot tests
# ---------------------------------------------------------------------------

def _create_dot_sdfg(dtype, N):
    """Build a Dot SDFG manually."""
    dace_dtype = dace.dtype_to_typeclass(dtype)
    sdfg = dace.SDFG('cupy_dot_test')
    state = sdfg.add_state()

    sdfg.add_array('x', [N], dace_dtype)
    sdfg.add_array('y', [N], dace_dtype)
    sdfg.add_array('result', [1], dace_dtype)

    rx = state.add_read('x')
    ry = state.add_read('y')
    wres = state.add_write('result')

    libnode = Dot('dot')
    libnode.implementation = 'CuPy'
    state.add_node(libnode)

    state.add_edge(rx, None, libnode, '_x',
                   dace.Memlet.from_array('x', sdfg.arrays['x']))
    state.add_edge(ry, None, libnode, '_y',
                   dace.Memlet.from_array('y', sdfg.arrays['y']))
    state.add_edge(libnode, '_result', wres, None,
                   dace.Memlet.from_array('result', sdfg.arrays['result']))

    return sdfg


@pytest.mark.gpu
class TestCuPyDot:

    def test_basic(self):
        """result = x . y (float32)."""
        N = 100
        x = np.random.rand(N).astype(np.float32)
        y = np.random.rand(N).astype(np.float32)
        result = np.zeros(1, dtype=np.float32)

        sdfg = _create_dot_sdfg(np.float32, N)
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, Dot, 'CuPy')

        csdfg = sdfg.compile()
        csdfg(x=x, y=y, result=result)
        del csdfg

        expected = np.dot(x, y)
        assert np.allclose(result[0], expected, rtol=1e-5)

    def test_float64(self):
        """result = x . y (float64)."""
        N = 100
        x = np.random.rand(N).astype(np.float64)
        y = np.random.rand(N).astype(np.float64)
        result = np.zeros(1, dtype=np.float64)

        sdfg = _create_dot_sdfg(np.float64, N)
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, Dot, 'CuPy')

        csdfg = sdfg.compile()
        csdfg(x=x, y=y, result=result)
        del csdfg

        expected = np.dot(x, y)
        assert np.allclose(result[0], expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# BatchedMatMul tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestCuPyBatchedMatMul:

    def test_basic_3d(self):
        """[batch, M, K] @ [batch, K, N]."""
        batch, M, K, N = 4, 10, 15, 8

        @dace.program
        def batched_mm(A: dace.float32[4, 10, 15],
                       B: dace.float32[4, 15, 8],
                       C: dace.float32[4, 10, 8]):
            C[:] = A @ B

        sdfg = batched_mm.to_sdfg()
        _find_and_set_impl(sdfg, BatchedMatMul, 'CuPy')
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, BatchedMatMul, 'CuPy')

        A = np.random.rand(batch, M, K).astype(np.float32)
        B = np.random.rand(batch, K, N).astype(np.float32)
        C = np.zeros((batch, M, N), dtype=np.float32)

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C)
        del csdfg

        expected = A @ B
        assert np.allclose(C, expected, rtol=1e-5)

    def test_broadcast(self):
        """[batch, M, K] @ [K, N] via matmul broadcasting."""
        batch, M, K, N = 4, 10, 15, 8

        @dace.program
        def batched_mm_bcast(A: dace.float32[4, 10, 15],
                             B: dace.float32[15, 8],
                             C: dace.float32[4, 10, 8]):
            C[:] = A @ B

        sdfg = batched_mm_bcast.to_sdfg()

        # Check if a BatchedMatMul node was produced; the frontend may
        # lower this differently (e.g., via Gemm inside a map).
        has_bmm = any(isinstance(n, BatchedMatMul)
                      for n, _ in sdfg.all_nodes_recursive())
        if not has_bmm:
            pytest.skip('Frontend did not produce a BatchedMatMul node '
                        'for this broadcast pattern')

        _find_and_set_impl(sdfg, BatchedMatMul, 'CuPy')
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, BatchedMatMul, 'CuPy')

        A = np.random.rand(batch, M, K).astype(np.float32)
        B = np.random.rand(K, N).astype(np.float32)
        C = np.zeros((batch, M, N), dtype=np.float32)

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C)
        del csdfg

        expected = A @ B
        assert np.allclose(C, expected, rtol=1e-5)

    def test_float64(self):
        """float64 batched matmul."""
        batch, M, K, N = 3, 7, 11, 5

        @dace.program
        def batched_mm_f64(A: dace.float64[3, 7, 11],
                           B: dace.float64[3, 11, 5],
                           C: dace.float64[3, 7, 5]):
            C[:] = A @ B

        sdfg = batched_mm_f64.to_sdfg()
        _find_and_set_impl(sdfg, BatchedMatMul, 'CuPy')
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, BatchedMatMul, 'CuPy')

        A = np.random.rand(batch, M, K).astype(np.float64)
        B = np.random.rand(batch, K, N).astype(np.float64)
        C = np.zeros((batch, M, N), dtype=np.float64)

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C)
        del csdfg

        expected = A @ B
        assert np.allclose(C, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Ger tests
# ---------------------------------------------------------------------------

def _create_ger_sdfg(dtype, M, N, alpha):
    """Build a Ger SDFG manually."""
    dace_dtype = dace.dtype_to_typeclass(dtype)
    sdfg = dace.SDFG('cupy_ger_test')
    state = sdfg.add_state()

    sdfg.add_array('x', [M], dace_dtype)
    sdfg.add_array('y', [N], dace_dtype)
    sdfg.add_array('A', [M, N], dace_dtype)
    sdfg.add_array('res', [M, N], dace_dtype)

    rx = state.add_read('x')
    ry = state.add_read('y')
    rA = state.add_read('A')
    wres = state.add_write('res')

    libnode = Ger('ger', n=N, m=M, alpha=alpha)
    libnode.implementation = 'CuPy'
    state.add_node(libnode)

    state.add_edge(rA, None, libnode, '_A',
                   dace.Memlet.from_array('A', sdfg.arrays['A']))
    state.add_edge(rx, None, libnode, '_x',
                   dace.Memlet.from_array('x', sdfg.arrays['x']))
    state.add_edge(ry, None, libnode, '_y',
                   dace.Memlet.from_array('y', sdfg.arrays['y']))
    state.add_edge(libnode, '_res', wres, None,
                   dace.Memlet.from_array('res', sdfg.arrays['res']))

    return sdfg


@pytest.mark.gpu
class TestCuPyGer:

    def test_basic(self):
        """res = outer(x, y) + A (alpha=1)."""
        M, N = 12, 10
        x = np.random.rand(M).astype(np.float32)
        y = np.random.rand(N).astype(np.float32)
        A = np.random.rand(M, N).astype(np.float32)
        res = np.zeros((M, N), dtype=np.float32)

        sdfg = _create_ger_sdfg(np.float32, M, N, alpha=1)
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, Ger, 'CuPy')

        csdfg = sdfg.compile()
        csdfg(x=x, y=y, A=A, res=res)
        del csdfg

        expected = np.outer(x, y) + A
        assert np.allclose(res, expected, rtol=1e-5)

    def test_alpha(self):
        """res = 2.0 * outer(x, y) + A."""
        M, N = 12, 10
        x = np.random.rand(M).astype(np.float32)
        y = np.random.rand(N).astype(np.float32)
        A = np.random.rand(M, N).astype(np.float32)
        res = np.zeros((M, N), dtype=np.float32)

        sdfg = _create_ger_sdfg(np.float32, M, N, alpha=2)
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, Ger, 'CuPy')

        csdfg = sdfg.compile()
        csdfg(x=x, y=y, A=A, res=res)
        del csdfg

        expected = 2.0 * np.outer(x, y) + A
        assert np.allclose(res, expected, rtol=1e-5)

    def test_float64(self):
        """float64 ger."""
        M, N = 12, 10
        x = np.random.rand(M).astype(np.float64)
        y = np.random.rand(N).astype(np.float64)
        A = np.random.rand(M, N).astype(np.float64)
        res = np.zeros((M, N), dtype=np.float64)

        sdfg = _create_ger_sdfg(np.float64, M, N, alpha=1)
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, Ger, 'CuPy')

        csdfg = sdfg.compile()
        csdfg(x=x, y=y, A=A, res=res)
        del csdfg

        expected = np.outer(x, y) + A
        assert np.allclose(res, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Axpy tests
# ---------------------------------------------------------------------------

def _create_axpy_sdfg(dtype, N, a):
    """Build an Axpy SDFG manually."""
    dace_dtype = dace.dtype_to_typeclass(dtype)
    sdfg = dace.SDFG('cupy_axpy_test')
    state = sdfg.add_state()

    sdfg.add_array('x', [N], dace_dtype)
    sdfg.add_array('y', [N], dace_dtype)
    sdfg.add_array('res', [N], dace_dtype)

    rx = state.add_read('x')
    ry = state.add_read('y')
    wres = state.add_write('res')

    libnode = Axpy('axpy', a=a, n=N)
    libnode.implementation = 'CuPy'
    state.add_node(libnode)

    state.add_edge(rx, None, libnode, '_x',
                   dace.Memlet.from_array('x', sdfg.arrays['x']))
    state.add_edge(ry, None, libnode, '_y',
                   dace.Memlet.from_array('y', sdfg.arrays['y']))
    state.add_edge(libnode, '_res', wres, None,
                   dace.Memlet.from_array('res', sdfg.arrays['res']))

    return sdfg


@pytest.mark.gpu
class TestCuPyAxpy:

    def test_basic(self):
        """res = 2.0 * x + y."""
        N = 100
        x = np.random.rand(N).astype(np.float32)
        y = np.random.rand(N).astype(np.float32)
        res = np.zeros(N, dtype=np.float32)

        sdfg = _create_axpy_sdfg(np.float32, N, a=2.0)
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, Axpy, 'CuPy')

        csdfg = sdfg.compile()
        csdfg(x=x, y=y, res=res)
        del csdfg

        expected = 2.0 * x + y
        assert np.allclose(res, expected, rtol=1e-5)

    def test_a_zero(self):
        """res = y (a=0)."""
        N = 100
        x = np.random.rand(N).astype(np.float32)
        y = np.random.rand(N).astype(np.float32)
        res = np.zeros(N, dtype=np.float32)

        sdfg = _create_axpy_sdfg(np.float32, N, a=0)
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, Axpy, 'CuPy')

        csdfg = sdfg.compile()
        csdfg(x=x, y=y, res=res)
        del csdfg

        expected = y.copy()
        assert np.allclose(res, expected, rtol=1e-5)

    def test_a_one(self):
        """res = x + y (a=1)."""
        N = 100
        x = np.random.rand(N).astype(np.float32)
        y = np.random.rand(N).astype(np.float32)
        res = np.zeros(N, dtype=np.float32)

        sdfg = _create_axpy_sdfg(np.float32, N, a=1)
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, Axpy, 'CuPy')

        csdfg = sdfg.compile()
        csdfg(x=x, y=y, res=res)
        del csdfg

        expected = x + y
        assert np.allclose(res, expected, rtol=1e-5)

    def test_float64(self):
        """float64 axpy."""
        N = 100
        x = np.random.rand(N).astype(np.float64)
        y = np.random.rand(N).astype(np.float64)
        res = np.zeros(N, dtype=np.float64)

        sdfg = _create_axpy_sdfg(np.float64, N, a=2.0)
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, Axpy, 'CuPy')

        csdfg = sdfg.compile()
        csdfg(x=x, y=y, res=res)
        del csdfg

        expected = 2.0 * x + y
        assert np.allclose(res, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Einsum tests
# ---------------------------------------------------------------------------

def _create_einsum_sdfg(einsum_str, arrays_spec, output_name):
    """Build an Einsum SDFG using the @dace.program frontend.

    :param einsum_str:  e.g., 'ij,jk->ik'
    :param arrays_spec: dict mapping array name -> (shape, dtype)
    :param output_name: name of the output array
    :returns: compiled SDFG
    """
    # Use the @dace.program approach to generate a proper Einsum node.
    # This is more robust than manually creating the node since the
    # frontend handles connector naming.
    pass


@pytest.mark.gpu
class TestCuPyEinsum:

    def test_matmul(self):
        """ij,jk->ik (matrix multiply via einsum)."""
        M, K, N = 10, 8, 12

        @dace.program
        def einsum_matmul(A: dace.float32[10, 8], B: dace.float32[8, 12],
                          C: dace.float32[10, 12]):
            C[:] = np.einsum('ij,jk->ik', A, B)

        sdfg = einsum_matmul.to_sdfg()
        _find_and_set_impl(sdfg, Einsum, 'CuPy')
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, Einsum, 'CuPy')

        A = np.random.rand(M, K).astype(np.float32)
        B = np.random.rand(K, N).astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C)
        del csdfg

        expected = A @ B
        assert np.allclose(C, expected, rtol=1e-5)

    def test_trace(self):
        """ii-> (trace)."""
        N = 10

        @dace.program
        def einsum_trace(A: dace.float32[10, 10],
                         result: dace.float32[1]):
            result[:] = np.einsum('ii->', A)

        sdfg = einsum_trace.to_sdfg()
        _find_and_set_impl(sdfg, Einsum, 'CuPy')
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, Einsum, 'CuPy')

        A = np.random.rand(N, N).astype(np.float32)
        result = np.zeros(1, dtype=np.float32)

        csdfg = sdfg.compile()
        csdfg(A=A, result=result)
        del csdfg

        expected = np.trace(A)
        assert np.allclose(result[0], expected, rtol=1e-5)

    def test_outer(self):
        """i,j->ij (outer product)."""
        M, N = 8, 12

        @dace.program
        def einsum_outer(x: dace.float32[8], y: dace.float32[12],
                         C: dace.float32[8, 12]):
            C[:] = np.einsum('i,j->ij', x, y)

        sdfg = einsum_outer.to_sdfg()
        _find_and_set_impl(sdfg, Einsum, 'CuPy')
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, Einsum, 'CuPy')

        x = np.random.rand(M).astype(np.float32)
        y = np.random.rand(N).astype(np.float32)
        C = np.zeros((M, N), dtype=np.float32)

        csdfg = sdfg.compile()
        csdfg(x=x, y=y, C=C)
        del csdfg

        expected = np.outer(x, y)
        assert np.allclose(C, expected, rtol=1e-5)

    def test_batch_matmul_einsum(self):
        """aij,ajk->aik (batched matmul via einsum)."""
        batch, M, K, N = 3, 6, 5, 7

        @dace.program
        def einsum_bmm(A: dace.float32[3, 6, 5], B: dace.float32[3, 5, 7],
                       C: dace.float32[3, 6, 7]):
            C[:] = np.einsum('aij,ajk->aik', A, B)

        sdfg = einsum_bmm.to_sdfg()
        _find_and_set_impl(sdfg, Einsum, 'CuPy')
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, Einsum, 'CuPy')

        A = np.random.rand(batch, M, K).astype(np.float32)
        B = np.random.rand(batch, K, N).astype(np.float32)
        C = np.zeros((batch, M, N), dtype=np.float32)

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C)
        del csdfg

        expected = np.einsum('aij,ajk->aik', A, B)
        assert np.allclose(C, expected, rtol=1e-5)

    def test_transpose(self):
        """ij->ji (matrix transpose via einsum)."""
        M, N = 10, 12

        @dace.program
        def einsum_transpose(A: dace.float32[10, 12],
                             B: dace.float32[12, 10]):
            B[:] = np.einsum('ij->ji', A)

        sdfg = einsum_transpose.to_sdfg()
        _find_and_set_impl(sdfg, Einsum, 'CuPy')
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, Einsum, 'CuPy')

        A = np.random.rand(M, N).astype(np.float32)
        B = np.zeros((N, M), dtype=np.float32)

        csdfg = sdfg.compile()
        csdfg(A=A, B=B)
        del csdfg

        expected = A.T
        assert np.allclose(B, expected, rtol=1e-5)

    def test_float64(self):
        """ij,jk->ik with float64."""
        M, K, N = 10, 8, 12

        @dace.program
        def einsum_matmul_f64(A: dace.float64[10, 8], B: dace.float64[8, 12],
                              C: dace.float64[10, 12]):
            C[:] = np.einsum('ij,jk->ik', A, B)

        sdfg = einsum_matmul_f64.to_sdfg()
        _find_and_set_impl(sdfg, Einsum, 'CuPy')
        sdfg.apply_gpu_transformations()
        _find_and_set_impl(sdfg, Einsum, 'CuPy')

        A = np.random.rand(M, K).astype(np.float64)
        B = np.random.rand(K, N).astype(np.float64)
        C = np.zeros((M, N), dtype=np.float64)

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C)
        del csdfg

        expected = A @ B
        assert np.allclose(C, expected, rtol=1e-10)


if __name__ == '__main__':
    # Quick smoke test runner (requires GPU)
    import sys

    tests = [
        ('Gemm.basic', TestCuPyGemm().test_basic),
        ('Gemm.transA', TestCuPyGemm().test_transA),
        ('Gemm.transB', TestCuPyGemm().test_transB),
        ('Gemm.transAB', TestCuPyGemm().test_transAB),
        ('Gemm.float64', TestCuPyGemm().test_float64),
        ('Gemv.basic', TestCuPyGemv().test_basic),
        ('Gemv.transA', TestCuPyGemv().test_transA),
        ('Gemv.float64', TestCuPyGemv().test_float64),
        ('Dot.basic', TestCuPyDot().test_basic),
        ('Dot.float64', TestCuPyDot().test_float64),
        ('BatchedMatMul.basic', TestCuPyBatchedMatMul().test_basic_3d),
        ('Ger.basic', TestCuPyGer().test_basic),
        ('Ger.alpha', TestCuPyGer().test_alpha),
        ('Axpy.basic', TestCuPyAxpy().test_basic),
        ('Axpy.a_zero', TestCuPyAxpy().test_a_zero),
        ('Axpy.a_one', TestCuPyAxpy().test_a_one),
        ('Einsum.matmul', TestCuPyEinsum().test_matmul),
        ('Einsum.outer', TestCuPyEinsum().test_outer),
        ('Einsum.transpose', TestCuPyEinsum().test_transpose),
    ]

    for name, fn in tests:
        try:
            fn()
            print(f'PASS: {name}')
        except Exception as e:
            print(f'FAIL: {name}: {e}')
            if '--stop' in sys.argv:
                sys.exit(1)
