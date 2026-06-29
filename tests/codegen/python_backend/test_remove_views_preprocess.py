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
            if isinstance(n, nodes.AccessNode) and isinstance(
                    sdfg.arrays.get(n.data, None), data.View):
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
        {'i': '0:4', 'j': '0:5'},
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
        {'i': '0:4', 'j': '0:5'},
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
    pipeline = VectorizeCuTile(widths=(4,))
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


if __name__ == '__main__':
    test_reshape_view_python_backend()
    test_slice_view_python_backend()
    test_nested_sdfg_with_view_python_backend()
    test_dace_program_reshape_python_backend()
    test_dace_program_slice_python_backend()
    # GPU test skipped in __main__; run via pytest with --gpu
