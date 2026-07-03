# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression tests for Python-backend argument marshalling and error reporting.

Covers the NPBench ``dace_cutile`` failure classes:

1. numpy scalar arguments (``np.int64`` etc.) for Scalar args and symbols
   (NPBench ``compute``): marshalled to 0-d numpy buffers of the declared
   dtype / native Python symbol values; ``ct.launch`` receives native scalars.
2. plain Python floats for Scalar args (NPBench ``channel_flow``): generated
   host code must not index scalar values that were never wrapped.
3. GPU element -> host scalar copies (NPBench ``syrk``/``syr2k``): emitted as
   ``.item()`` transfers instead of ``.get(out=scalar[:1])``.
4. ``InvalidSDFG*Error.__str__`` with stale state/node/edge ids (NPBench
   ``adi``/``cavity_flow``): degrades to numeric ids instead of raising.
5. Tail-tile single-column writes and column-slice gather loads through
   ``VectorizeCuTile`` (NPBench ``adi``/``cavity_flow`` true errors):
   provably-OOB propagated memlets are clamped, and gather results are not
   permuted with source-rank axes.
"""
import itertools

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.sdfg import SDFG
from dace.sdfg.validation import (InvalidSDFGEdgeError, InvalidSDFGError, InvalidSDFGInterstateEdgeError,
                                  InvalidSDFGNodeError)

_COUNTER = itertools.count()

N = dace.symbol('N')
M = dace.symbol('M')


def _python_backend(program) -> 'dace.codegen.py.compiled_sdfg.PythonCompiledSDFG':
    """Lower a ``@dace.program`` to the pure Python backend and compile it.

    :param program: The DaCe program to compile.
    :returns: The compiled Python-backend SDFG.
    """
    sdfg = program.to_sdfg(simplify=False)
    sdfg.name = f'{sdfg.name}_{next(_COUNTER)}'
    sdfg.backend = dtypes.BackendLanguage.Python
    return sdfg.compile()


def _cutile_compile(program, widths=(32, )):
    """Lower a ``@dace.program`` through the full cuTile pipeline.

    :param program: The DaCe program to compile.
    :param widths: Tile widths for :class:`VectorizeCuTile`.
    :returns: The compiled Python-backend SDFG.
    """
    from dace.transformation.passes.vectorization import VectorizeCuTile
    sdfg = program.to_sdfg(simplify=False)
    sdfg.name = f'{sdfg.name}_{next(_COUNTER)}'
    VectorizeCuTile(widths=widths).apply_pass(sdfg, {})
    return sdfg.compile()


# =============================================================================
# 1 + 2: Scalar argument marshalling (pure Python backend, CPU)
# =============================================================================


@dace.program
def _axpb(x: dace.float64[N], a: dace.float64, b: dace.float64):
    return x * a + b


@dace.program
def _int_scale(x: dace.int64[N], a: dace.int64, b: dace.int64):
    return x * a + b


@dace.program
def _inplace_scale(x: dace.float64[N], a: dace.float64):
    x *= a


class TestScalarArgumentMarshalling:

    def test_numpy_float_scalar_args(self):
        """np.float64 scalar arguments are accepted (compute class)."""
        csdfg = _python_backend(_axpb)
        x = np.arange(8, dtype=np.float64)
        out = csdfg(x=x, a=np.float64(2.5), b=np.float64(1.0), N=8)
        np.testing.assert_allclose(np.asarray(out), x * 2.5 + 1.0)

    def test_numpy_int_scalar_args_and_symbol(self):
        """np.int64 Scalar args and np.int64 symbol values (compute class)."""
        csdfg = _python_backend(_int_scale)
        x = np.arange(8, dtype=np.int64)
        out = csdfg(x=x, a=np.int64(4), b=np.int64(9), N=np.int64(8))
        np.testing.assert_array_equal(np.asarray(out), x * 4 + 9)

    def test_python_float_scalar_args(self):
        """Plain Python floats for Scalar args (channel_flow class)."""
        csdfg = _python_backend(_axpb)
        x = np.arange(8, dtype=np.float64)
        out = csdfg(x=x, a=2.0, b=0.5, N=8)
        np.testing.assert_allclose(np.asarray(out), x * 2.0 + 0.5)

    def test_mixed_dtype_scalar_coerced_to_declared(self):
        """A numpy scalar of a different dtype is cast to the declared one."""
        csdfg = _python_backend(_int_scale)
        x = np.arange(8, dtype=np.int64)
        # float64 values passed for int64 scalars: truncated like the C++ backend.
        out = csdfg(x=x, a=np.float64(4.0), b=np.float64(9.0), N=8)
        np.testing.assert_array_equal(np.asarray(out), x * 4 + 9)

    def test_zero_d_array_scalar_arg(self):
        """A 0-d numpy array bound to a Scalar argument works."""
        csdfg = _python_backend(_axpb)
        x = np.arange(8, dtype=np.float64)
        out = csdfg(x=x, a=np.asarray(3.0), b=np.asarray(1.5), N=8)
        np.testing.assert_allclose(np.asarray(out), x * 3.0 + 1.5)

    def test_nonscalar_array_for_scalar_arg_raises(self):
        """Passing a length-3 array for a Scalar argument raises TypeError."""
        csdfg = _python_backend(_axpb)
        x = np.arange(8, dtype=np.float64)
        with pytest.raises(TypeError, match='expected a scalar'):
            csdfg(x=x, a=np.ones(3), b=1.0, N=8)

    def test_positional_numpy_scalar_args(self):
        """Positional numpy scalars are marshalled too (fast path, no returns).

        Positional binding follows ``arglist()`` order, here ``(x, N, a)``.
        """
        csdfg = _python_backend(_inplace_scale)
        x = np.arange(8, dtype=np.float64)
        expected = x * 2.0
        csdfg(x, np.int64(8), np.float64(2.0))
        np.testing.assert_allclose(x, expected)


# =============================================================================
# 4: InvalidSDFG*Error.__str__ with stale ids
# =============================================================================


def _tiny_sdfg() -> SDFG:
    sdfg = SDFG(f'stale_id_sdfg_{next(_COUNTER)}')
    sdfg.add_state('only_state')
    return sdfg


class TestInvalidSDFGErrorStr:
    """__str__ must degrade to numeric ids when ids no longer resolve."""

    def test_invalid_sdfg_error_stale_state_id(self):
        err = InvalidSDFGError('boom', _tiny_sdfg(), state_id=6)
        assert 'boom' in str(err)
        assert 'state with id 6' in str(err)

    def test_invalid_sdfg_node_error_stale_state_id(self):
        err = InvalidSDFGNodeError('boom', _tiny_sdfg(), state_id=6, node_id=3)
        s = str(err)
        assert 'boom' in s
        assert 'state with id 6' in s
        assert 'node with id 3' in s

    def test_invalid_sdfg_node_error_stale_node_id(self):
        err = InvalidSDFGNodeError('boom', _tiny_sdfg(), state_id=0, node_id=99)
        s = str(err)
        assert 'boom' in s
        assert 'only_state' in s
        assert 'node with id 99' in s

    def test_invalid_sdfg_edge_error_stale_state_id(self):
        err = InvalidSDFGEdgeError('boom', _tiny_sdfg(), state_id=6, edge_id=0)
        s = str(err)
        assert 'boom' in s
        assert 'state with id 6' in s
        assert 'edge with id 0' in s

    def test_invalid_sdfg_edge_error_stale_edge_id(self):
        err = InvalidSDFGEdgeError('boom', _tiny_sdfg(), state_id=0, edge_id=42)
        s = str(err)
        assert 'boom' in s
        assert 'edge with id 42' in s

    def test_invalid_interstate_edge_error_stale_edge_id(self):
        err = InvalidSDFGInterstateEdgeError('boom', _tiny_sdfg(), edge_id=7)
        s = str(err)
        assert 'boom' in s
        assert 'edge with id 7' in s

    def test_valid_ids_still_render_labels(self):
        err = InvalidSDFGError('boom', _tiny_sdfg(), state_id=0)
        assert 'only_state' in str(err)


# =============================================================================
# 1 + 3 + 5: cuTile end-to-end regressions (GPU)
# =============================================================================


@dace.program
def _compute_like(array_1: dace.int64[M, N], array_2: dace.int64[M, N], a: dace.int64, b: dace.int64, c: dace.int64):
    return np.minimum(np.maximum(array_1, 2), 10) * a + array_2 * b + c


@dace.program
def _syrk_like(alpha: dace.float64, beta: dace.float64, C: dace.float64[N, N], A: dace.float64[N, M]):
    for i in range(N):
        C[i, :i + 1] *= beta
        for k in range(M):
            C[i, :i + 1] += alpha * A[i, k] * A[:i + 1, k]


@dace.program
def _adi_like(u: dace.float64[N, N], p: dace.float64[N, N]):
    u[1:N - 1, N - 1] = 1.0
    for j in range(1, N - 1):
        u[1:N - 1, j] = p[1:N - 1, j - 1] + 0.5


@pytest.mark.gpu
class TestCuTileArgumentMarshalling:

    def test_numpy_int_scalars_reach_ct_launch(self):
        """np.int64 Scalar args through the cuTile pipeline (compute)."""
        csdfg = _cutile_compile(_compute_like)
        rng = np.random.default_rng(42)
        m, n = 64, 70  # non-divisible tail on the tiled dim
        a1 = rng.integers(0, 1000, size=(m, n)).astype(np.int64)
        a2 = rng.integers(0, 1000, size=(m, n)).astype(np.int64)
        a, b, c = np.int64(4), np.int64(3), np.int64(9)
        out = csdfg(array_1=a1, array_2=a2, a=a, b=b, c=c, M=np.int64(m), N=np.int64(n))
        ref = np.minimum(np.maximum(a1, 2), 10) * a + a2 * b + c
        np.testing.assert_array_equal(np.asarray(out), ref)

    def test_gpu_element_to_host_scalar_copy(self):
        """Host scalars fed from GPU elements + scalar launch args (syrk)."""
        csdfg = _cutile_compile(_syrk_like)
        n, m = 40, 30
        C = np.fromfunction(lambda i, j: ((i * j + 2) % n) / m, (n, n), dtype=np.float64)
        A = np.fromfunction(lambda i, j: ((i * j + 1) % n) / n, (n, m), dtype=np.float64)
        alpha, beta = np.float64(1.5), np.float64(1.2)

        ref = C.copy()
        for i in range(n):
            ref[i, :i + 1] *= beta
            for k in range(m):
                ref[i, :i + 1] += alpha * A[i, k] * A[:i + 1, k]

        csdfg(alpha=alpha, beta=beta, C=C, A=A, N=n, M=m)
        # rtol matches the NPBench harness: the cuda.tile runtime packs float
        # kernel scalars as float32 (a runtime limitation that predates the
        # marshalling fix -- np.float64 subclasses float and was truncated
        # the same way), so float64 scalar paths carry ~1e-8 relative error.
        np.testing.assert_allclose(C, ref, rtol=1e-5)

    def test_tail_tile_column_assignment_and_gather_load(self):
        """Single-column tail writes + column gather loads (adi/cavity_flow)."""
        csdfg = _cutile_compile(_adi_like)
        n = 70  # non-power-of-2: tail tiles + provably-OOB propagated memlets
        rng = np.random.default_rng(7)
        u = rng.random((n, n))
        p = rng.random((n, n))

        ref = u.copy()
        ref[1:n - 1, n - 1] = 1.0
        for j in range(1, n - 1):
            ref[1:n - 1, j] = p[1:n - 1, j - 1] + 0.5

        csdfg(u=u, p=p, N=n)
        np.testing.assert_allclose(u, ref)


# =============================================================================
# 3: Cross-storage scalar copy emission (structural, no GPU needed)
# =============================================================================


class TestScalarCrossStorageCopyEmission:
    """GPU <-> host copies with a Scalar endpoint must not emit array APIs."""

    @staticmethod
    def _emit(src_scalar: bool):
        from dace.codegen import dispatcher as dispatcher_mod
        from dace.codegen.py.cutile_target import CuTilePythonCodeGen
        from dace.codegen.py.prettycode import PythonCodeIOStream
        from dace.memlet import Memlet

        sdfg = SDFG(f'scalar_copy_{next(_COUNTER)}')
        sdfg.backend = dtypes.BackendLanguage.Python
        if src_scalar:
            sdfg.add_scalar('s', dace.float64, storage=dtypes.StorageType.CPU_Heap, transient=True)
            sdfg.add_array('g', [1], dace.float64, storage=dtypes.StorageType.GPU_Global, transient=True)
            src_name, dst_name = 's', 'g'
            memlet = Memlet(data='g', subset='0')
        else:
            sdfg.add_array('g', [4, 4], dace.float64, storage=dtypes.StorageType.GPU_Global, transient=True)
            sdfg.add_scalar('s', dace.float64, storage=dtypes.StorageType.CPU_Heap, transient=True)
            src_name, dst_name = 'g', 's'
            memlet = Memlet(data='g', subset='1, 2', other_subset='0')
        state = sdfg.add_state('s0')
        src = state.add_read(src_name)
        dst = state.add_write(dst_name)
        edge = state.add_edge(src, None, dst, None, memlet)

        class _Frame:

            def __init__(self):
                self.dispatcher = dispatcher_mod.TargetDispatcher(self)
                self._initcode = PythonCodeIOStream()
                self._exitcode = PythonCodeIOStream()

        codegen = CuTilePythonCodeGen(_Frame(), sdfg)
        stream = PythonCodeIOStream()
        codegen.copy_memory(sdfg, sdfg, state, 0, src, dst, edge, PythonCodeIOStream(), stream)
        return stream.getvalue()

    def test_gpu_element_to_host_scalar_emits_item(self):
        code = self._emit(src_scalar=False)
        assert '.item()' in code
        assert '.get(' not in code

    def test_host_scalar_to_gpu_emits_assignment(self):
        code = self._emit(src_scalar=True)
        assert '.set(' not in code
        assert '=' in code


if __name__ == '__main__':
    import sys
    pytest.main([__file__, '-v'] + sys.argv[1:])
