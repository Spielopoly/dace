# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Integration tests for the RemoveViews preprocess safety net in the Python backend.

The Python backend's ``preprocess()`` method
(:meth:`dace.codegen.py.python_target.PythonCodegen.preprocess`) runs
:class:`~dace.transformation.passes.remove_views.RemoveViews` on every nested
SDFG before code generation begins.  This ensures that View access nodes --
which the Python backend does not support natively -- are eliminated before
any code is emitted.

Each test constructs an SDFG containing one or more View nodes, sets the Python
backend, compiles, runs, and compares the result against NumPy.
"""
import itertools

import numpy as np
import pytest

import dace
from dace import data, Memlet, nodes
from dace.dtypes import BackendLanguage
from dace.sdfg import SDFG

_SDFG_COUNTER = itertools.count()


def _new_sdfg(prefix: str) -> SDFG:
    """Create a fresh SDFG with a unique name and Python backend selected."""
    sdfg = SDFG(f"{prefix}_{next(_SDFG_COUNTER)}")
    sdfg.backend = BackendLanguage.Python
    return sdfg


def _count_views(sdfg: SDFG) -> int:
    """Count View access nodes in the top-level SDFG only (non-recursive)."""
    count = 0
    for state in sdfg.states():
        for n in state.nodes():
            if isinstance(n, nodes.AccessNode) and isinstance(sdfg.arrays.get(n.data, None), data.View):
                count += 1
    return count


def _count_views_recursive(sdfg: SDFG) -> int:
    """Count View access nodes across the SDFG and all nested SDFGs."""
    count = 0
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            for n in state.nodes():
                if isinstance(n, nodes.AccessNode) and isinstance(nsdfg.arrays.get(n.data, None), data.View):
                    count += 1
    return count


# ---------------------------------------------------------------------------
# Test 1: Reshape view -- programmatic SDFG
# ---------------------------------------------------------------------------


def test_reshape_view_python_backend():
    """Reshape view: A[20] -> V[4,5], mapped tasklet B[i,j] = V[i,j] * 2.

    Builds the SDFG programmatically with a View node, sets the Python
    backend, compiles, runs, and verifies that RemoveViews in preprocess
    transparently handles the view so that the result matches NumPy.
    """
    sdfg = _new_sdfg('reshape_view_pyback')
    sdfg.add_array('A', [20], dace.float64)
    sdfg.add_array('B', [4, 5], dace.float64)
    sdfg.add_view('V', [4, 5], dace.float64)

    state = sdfg.add_state(is_start_block=True)
    a = state.add_read('A')
    v = state.add_access('V')

    # View edge: A (flat) -> V (reshaped)
    state.add_edge(a, None, v, 'views', Memlet(data='A', subset='0:20', other_subset='0:4, 0:5'))

    state.add_mapped_tasklet(
        'double',
        {
            'i': '0:4',
            'j': '0:5'
        },
        {'inp': Memlet('V[i, j]')},
        'out = inp * 2.0',
        {'out': Memlet('B[i, j]')},
        input_nodes={'V': v},
        external_edges=True,
    )

    sdfg.validate()

    # Sanity: view exists before compilation
    assert _count_views(sdfg) >= 1

    compiled = sdfg.compile()

    A = np.arange(20, dtype=np.float64)
    B = np.zeros((4, 5), dtype=np.float64)
    compiled(A=A, B=B)

    expected = A.reshape(4, 5) * 2.0
    np.testing.assert_allclose(B, expected)


# ---------------------------------------------------------------------------
# Test 2: Slice view -- programmatic SDFG
# ---------------------------------------------------------------------------


def test_slice_view_python_backend():
    """Slice view: A[10] -> V[5] viewing A[2:7], mapped tasklet B[i] = V[i] + 1.

    Builds a 1-D contiguous slice view programmatically and verifies
    that the Python backend preprocess eliminates it correctly.
    """
    sdfg = _new_sdfg('slice_view_pyback')
    sdfg.add_array('A', [10], dace.float64)
    sdfg.add_array('B', [5], dace.float64)
    sdfg.add_view('V', [5], dace.float64)

    state = sdfg.add_state(is_start_block=True)
    a = state.add_read('A')
    v = state.add_access('V')

    # View edge: A[2:7] -> V[0:5]
    state.add_edge(a, None, v, 'views', Memlet(data='A', subset='2:7', other_subset='0:5'))

    state.add_mapped_tasklet(
        'add_one',
        {'i': '0:5'},
        {'inp': Memlet('V[i]')},
        'out = inp + 1.0',
        {'out': Memlet('B[i]')},
        input_nodes={'V': v},
        external_edges=True,
    )

    sdfg.validate()
    assert _count_views(sdfg) >= 1

    compiled = sdfg.compile()

    A = np.arange(10, dtype=np.float64)
    B = np.zeros(5, dtype=np.float64)
    compiled(A=A, B=B)

    expected = A[2:7] + 1.0
    np.testing.assert_allclose(B, expected)


# ---------------------------------------------------------------------------
# Test 3: Nested SDFG with view
# ---------------------------------------------------------------------------


def test_nested_sdfg_with_view_python_backend():
    """NestedSDFG containing a reshape view: outer A[20] -> inner V[4,5] -> outer B[4,5].

    Tests that ``all_sdfgs_recursive()`` in the Python backend's preprocess
    catches and removes views inside nested SDFGs.
    """
    # -- Inner SDFG: X[20] -> V[4,5] (reshape view) -> Y[4,5] = V * 2 --
    inner = SDFG('inner_with_view')
    inner.add_array('X', [20], dace.float64)
    inner.add_array('Y', [4, 5], dace.float64)
    inner.add_view('V', [4, 5], dace.float64)

    inner_state = inner.add_state(is_start_block=True)
    x = inner_state.add_read('X')
    v = inner_state.add_access('V')

    inner_state.add_edge(x, None, v, 'views', Memlet(data='X', subset='0:20', other_subset='0:4, 0:5'))

    inner_state.add_mapped_tasklet(
        'double',
        {
            'i': '0:4',
            'j': '0:5'
        },
        {'inp': Memlet('V[i, j]')},
        'out = inp * 2.0',
        {'out': Memlet('Y[i, j]')},
        input_nodes={'V': v},
        external_edges=True,
    )

    # -- Outer SDFG --
    outer = _new_sdfg('outer_nested_view')
    outer.add_array('A', [20], dace.float64)
    outer.add_array('B', [4, 5], dace.float64)

    state = outer.add_state(is_start_block=True)
    nested = state.add_nested_sdfg(inner, {'X'}, {'Y'})
    state.add_edge(state.add_read('A'), None, nested, 'X', Memlet('A[0:20]'))
    state.add_edge(nested, 'Y', state.add_write('B'), None, Memlet('B[0:4, 0:5]'))

    outer.validate()

    # The inner SDFG should contain a view
    assert _count_views_recursive(outer) >= 1

    compiled = outer.compile()

    A = np.arange(20, dtype=np.float64)
    B = np.zeros((4, 5), dtype=np.float64)
    compiled(A=A, B=B)

    expected = A.reshape(4, 5) * 2.0
    np.testing.assert_allclose(B, expected)


# ---------------------------------------------------------------------------
# Test 4: @dace.program reshape via np.reshape
# ---------------------------------------------------------------------------


def test_dace_program_reshape_python_backend():
    """Frontend @dace.program using np.reshape produces a View; Python backend handles it.

    Uses ``to_sdfg(simplify=False)`` to preserve the view created by the
    frontend, then sets the Python backend and verifies correct execution.
    """

    @dace.program
    def reshape_prog(A: dace.float64[20]):
        V = np.reshape(A, (4, 5))
        return V * 2.0

    sdfg = reshape_prog.to_sdfg(simplify=False)
    sdfg.backend = BackendLanguage.Python

    # The frontend should have created at least one view
    assert _count_views_recursive(sdfg) >= 1

    compiled = sdfg.compile()

    A = np.arange(20, dtype=np.float64)
    ret = np.zeros((4, 5), dtype=np.float64)
    compiled(A=A, __return=ret)

    expected = A.reshape(4, 5) * 2.0
    np.testing.assert_allclose(ret, expected)


# ---------------------------------------------------------------------------
# Test 5: @dace.program slice
# ---------------------------------------------------------------------------


def test_dace_program_slice_python_backend():
    """Frontend @dace.program using slicing produces a View; Python backend handles it.

    Uses ``to_sdfg(simplify=False)`` to preserve the slice view, then
    sets the Python backend and verifies correct execution.
    """

    @dace.program
    def slice_prog(A: dace.float64[10]):
        V = A[2:7]
        return V + 1.0

    sdfg = slice_prog.to_sdfg(simplify=False)
    sdfg.backend = BackendLanguage.Python

    assert _count_views_recursive(sdfg) >= 1

    compiled = sdfg.compile()

    A = np.arange(10, dtype=np.float64)
    ret = np.zeros(5, dtype=np.float64)
    compiled(A=A, __return=ret)

    expected = A[2:7] + 1.0
    np.testing.assert_allclose(ret, expected)


# ---------------------------------------------------------------------------
# Test 6: cuTile pipeline with views (GPU)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_cutile_pipeline_with_views():
    """VectorizeCuTile pipeline on an SDFG that originally contains views.

    The cuTile pipeline calls ``apply_gpu_transformations()`` which internally
    calls ``simplify()``, which can re-introduce views.  The Python backend's
    preprocess safety net must strip them before codegen.

    This test:
    1. Creates a @dace.program that generates views (reshape).
    2. Runs canonicalize.
    3. Runs VectorizeCuTile.
    4. Compiles (Python backend, set by VectorizeCuTile).
    5. Runs on GPU and compares against NumPy.
    """
    from dace.transformation.passes.canonicalize import canonicalize
    from dace.transformation.passes.vectorization.vectorize_cutile import VectorizeCuTile

    @dace.program
    def reshape_add(A: dace.float64[16], B: dace.float64[4, 4]):
        V = np.reshape(A, (4, 4))
        B[:] = V + 1.0

    sdfg = reshape_add.to_sdfg(simplify=False)

    # Step 1: Canonicalize (the caller's job before vectorization)
    canonicalize(sdfg)

    # Step 2: Run the cuTile pipeline (widths must be powers of 2)
    pipeline = VectorizeCuTile(widths=(4, ))
    pipeline.apply_pass(sdfg, {})

    # VectorizeCuTile sets the Python backend automatically
    assert sdfg.backend == BackendLanguage.Python

    compiled = sdfg.compile()

    A = np.arange(16, dtype=np.float64)
    B = np.zeros((4, 4), dtype=np.float64)
    compiled(A=A, B=B)

    expected = A.reshape(4, 4) + 1.0
    np.testing.assert_allclose(B, expected)


# ---------------------------------------------------------------------------
# Scope-crossing views (mlp regression): the view edge attaches to a
# MapEntry/MapExit ``views`` connector rather than directly to the viewed
# AccessNode.  RemoveViews must reconnect to the IMMEDIATE scope-node
# endpoint; reconnecting to the distant viewed AccessNode bypasses the
# scope node and orphans it (MapExit with in_degree 0 ->
# "Leftover nodes in queue" in scope_dict()).
# ---------------------------------------------------------------------------


def _build_write_view_through_mapexit(n: int = 16) -> SDFG:
    """``B[i] = A[i] + 1`` where the write goes through a View of ``B`` whose
    view edge attaches to the MapExit's ``views`` connector."""
    sdfg = _new_sdfg('scope_write_view')
    sdfg.add_array('A', (n, ), dace.float64)
    sdfg.add_array('B', (n, ), dace.float64)
    sdfg.add_view('V', (n, ), dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map('m', {'i': f'0:{n}'})
    t = state.add_tasklet('body', {'_a'}, {'_b'}, '_b = _a + 1.0')
    state.add_memlet_path(state.add_read('A'), me, t, dst_conn='_a', memlet=Memlet(f'A[i]'))
    v = state.add_access('V')
    v.add_out_connector('views')
    state.add_edge(t, '_b', v, None, Memlet('V[i]'))
    mx.add_in_connector('IN_B')
    mx.add_out_connector('OUT_B')
    state.add_edge(v, 'views', mx, 'IN_B', Memlet(f'B[0:{n}]'))
    state.add_edge(mx, 'OUT_B', state.add_write('B'), None, Memlet(f'B[0:{n}]'))
    return sdfg


def test_scope_crossing_write_view_keeps_mapexit_connected():
    """RemoveViews on a write-side scope-crossing view must keep the MapExit
    on the dataflow path (in_degree > 0) and leave a valid scope tree."""
    from dace.transformation.passes.remove_views import RemoveViews

    sdfg = _build_write_view_through_mapexit()
    state = next(iter(sdfg.states()))
    RemoveViews().apply_pass(sdfg, {})
    assert _count_views(sdfg) == 0
    exits = [n for n in state.nodes() if isinstance(n, nodes.MapExit)]
    assert exits and all(state.in_degree(x) > 0 for x in exits), \
        'MapExit was orphaned by the view removal'
    state.scope_dict()  # raised "Leftover nodes in queue" before the fix


def test_scope_crossing_write_view_runs():
    """End-to-end compile + run of the scope-crossing write view."""
    n = 16
    sdfg = _build_write_view_through_mapexit(n)
    a = np.arange(n, dtype=np.float64)
    b = np.zeros(n)
    sdfg.compile()(A=a.copy(), B=b)
    np.testing.assert_allclose(b, a + 1.0)


def _build_read_view_through_mapentry(n: int = 16) -> SDFG:
    """``B[i] = 2 * A[i]`` where the read comes through a View of ``A`` whose
    view edge attaches to the MapEntry's ``views`` connector."""
    sdfg = _new_sdfg('scope_read_view')
    sdfg.add_array('A', (n, ), dace.float64)
    sdfg.add_array('B', (n, ), dace.float64)
    sdfg.add_view('V', (n, ), dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map('m', {'i': f'0:{n}'})
    me.add_in_connector('IN_A')
    me.add_out_connector('OUT_A')
    state.add_edge(state.add_read('A'), None, me, 'IN_A', Memlet(f'A[0:{n}]'))
    v = state.add_access('V')
    v.add_in_connector('views')
    state.add_edge(me, 'OUT_A', v, 'views', Memlet(f'A[0:{n}]'))
    t = state.add_tasklet('body', {'_a'}, {'_b'}, '_b = 2.0 * _a')
    state.add_edge(v, None, t, '_a', Memlet('V[i]'))
    state.add_memlet_path(t, mx, state.add_write('B'), src_conn='_b', memlet=Memlet('B[i]'))
    return sdfg


def test_scope_crossing_read_view_keeps_mapentry_connected():
    """RemoveViews on a read-side scope-crossing view must keep the MapEntry
    on the dataflow path (out_degree > 0) and leave a valid scope tree."""
    from dace.transformation.passes.remove_views import RemoveViews

    sdfg = _build_read_view_through_mapentry()
    state = next(iter(sdfg.states()))
    RemoveViews().apply_pass(sdfg, {})
    assert _count_views(sdfg) == 0
    entries = [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]
    assert entries and all(state.out_degree(x) > 0 for x in entries), \
        'MapEntry was orphaned by the view removal'
    state.scope_dict()


def test_scope_crossing_read_view_runs():
    """End-to-end compile + run of the scope-crossing read view."""
    n = 16
    sdfg = _build_read_view_through_mapentry(n)
    a = np.arange(n, dtype=np.float64)
    b = np.zeros(n)
    sdfg.compile()(A=a.copy(), B=b)
    np.testing.assert_allclose(b, 2.0 * a)


def _build_symbolic_nested_flatten_views() -> SDFG:
    """Build symbolic read/write flatten views around a NestedSDFG."""
    n = dace.symbol('N', dtype=dace.int64)

    inner = SDFG('symbolic_flatten_inner')
    inner.add_symbol('N', dace.int64)
    inner.add_array('X', (n, 4), dace.float64)
    inner.add_array('Y', (n, 4), dace.float64)
    inner_state = inner.add_state(is_start_block=True)
    inner_state.add_mapped_tasklet('double', {
        'i': '0:N',
        'j': '0:4'
    }, {'x': Memlet('X[i, j]')},
                                   'y = 2.0 * x', {'y': Memlet('Y[i, j]')},
                                   external_edges=True)

    sdfg = _new_sdfg('symbolic_nested_flatten_views')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', (n, 2, 2), dace.float64)
    sdfg.add_array('B', (n, 2, 2), dace.float64)
    sdfg.add_view('Vin', (n, 4), dace.float64)
    sdfg.add_view('Vout', (n, 4), dace.float64)

    state = sdfg.add_state(is_start_block=True)
    a = state.add_read('A')
    b = state.add_write('B')
    vin = state.add_access('Vin')
    vout = state.add_access('Vout')
    nested = state.add_nested_sdfg(inner, ['X'], ['Y'], symbol_mapping={'N': n})
    state.add_edge(a, None, vin, 'views', Memlet('A[0:N, 0:2, 0:2]', other_subset='0:N, 0:4'))
    state.add_edge(vin, None, nested, 'X', Memlet('Vin[0:N, 0:4]'))
    state.add_edge(nested, 'Y', vout, None, Memlet('Vout[0:N, 0:4]'))
    state.add_edge(vout, 'views', b, None, Memlet('B[0:N, 0:2, 0:2]', other_subset='0:N, 0:4'))
    return sdfg


def test_symbolic_nested_flatten_views_python_backend():
    """Surviving symbolic flatten views alias their backing NumPy arrays."""
    from dace.transformation.passes.remove_views import RemoveViews

    sdfg = _build_symbolic_nested_flatten_views()
    RemoveViews().apply_pass(sdfg, {})
    assert _count_views(sdfg) == 2

    compiled = sdfg.compile()
    a = np.arange(12, dtype=np.float64).reshape(3, 2, 2)
    b = np.zeros_like(a)
    compiled(A=a, B=b, N=3)
    np.testing.assert_allclose(b, 2.0 * a)


@pytest.mark.gpu
def test_cutile_symbolic_flatten_matmul_view():
    """A Lenet-shaped flatten -> matmul -> tile kernel runs on the GPU."""
    import cupy as cp
    from dace.transformation.passes.vectorization import VectorizeCuTile

    n = dace.symbol('N', dtype=dace.int64)
    o = dace.symbol('O', dtype=dace.int64)

    @dace.program
    def flatten_matmul_bias(A: dace.float32[n, 2, 2, 4], B: dace.float32[16, o], bias: dace.float32[o]):
        flat = np.reshape(A, (n, 16))
        return flat @ B + bias

    sdfg = flatten_matmul_bias.to_sdfg(simplify=False)
    VectorizeCuTile(widths=(8, 8), use_gpu_storage=True).apply_pass(sdfg, {})
    assert any(isinstance(desc, data.View) for nsdfg in sdfg.all_sdfgs_recursive() for desc in nsdfg.arrays.values())

    compiled = sdfg.compile()
    rng = np.random.default_rng(42)
    a = rng.random((3, 2, 2, 4), dtype=np.float32)
    b = rng.random((16, 19), dtype=np.float32)
    bias = rng.random(19, dtype=np.float32)
    result = compiled(A=cp.asarray(a), B=cp.asarray(b), bias=cp.asarray(bias), N=3, O=19)
    np.testing.assert_allclose(cp.asnumpy(result), a.reshape(3, 16) @ b + bias, rtol=2e-5, atol=2e-5)


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    test_reshape_view_python_backend()
    test_slice_view_python_backend()
    test_nested_sdfg_with_view_python_backend()
    test_dace_program_reshape_python_backend()
    test_dace_program_slice_python_backend()
    test_scope_crossing_write_view_keeps_mapexit_connected()
    test_scope_crossing_write_view_runs()
    test_scope_crossing_read_view_keeps_mapentry_connected()
    test_scope_crossing_read_view_runs()
    # GPU test skipped in __main__; run via pytest with --gpu
