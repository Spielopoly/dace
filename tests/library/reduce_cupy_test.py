# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the CuPy expansion of the Reduce library node.

The CuPy expansion emits a Python-language tasklet that calls
``cupy.sum``/``cupy.prod``/etc.  It requires the **Python backend**
(``sdfg.backend = BackendLanguage.Python``) — NOT ``apply_gpu_transformations``
which is for the C++/CUDA codegen path.  CuPy itself manages GPU execution
internally.
"""
import itertools
import numpy as np
import pytest

import dace
import dace.libraries.standard as std
from dace import SDFG, Memlet, dtypes
from dace.frontend.operations import detect_reduction_type
from dace.libraries.standard.nodes.reduce import (
    ExpandReduceCuPy,
    ExpandReducePure,
    _REDUCTION_TYPE_TO_CUPY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_reduce_sdfg(in_shape, axes, wcr, dtype=dace.float64, identity=None):
    """Build a minimal SDFG with a single Reduce node.

    :param in_shape: shape of the input array (list of ints).
    :param axes:     axes to reduce over (list of ints or None for all).
    :param wcr:      WCR lambda string.
    :param dtype:    DaCe data type.
    :param identity: optional identity value for initialisation.
    :returns:        (sdfg, reduce_node)
    """
    out_shape = [s for i, s in enumerate(in_shape) if i not in (axes or [])]
    if not out_shape:
        out_shape = [1]

    g = SDFG('reduce_cupy_test')
    g.add_array('A', in_shape, dtype)
    g.add_array('B', out_shape, dtype)
    st = g.add_state('main', is_start_block=True)

    a = st.add_access('A')
    b = st.add_access('B')
    r = st.add_reduce(wcr, axes, identity)
    st.add_nedge(a, r, Memlet.from_array('A', g.arrays['A']))
    st.add_nedge(r, b, Memlet.from_array('B', g.arrays['B']))

    return g, r


N = dace.symbol('N', dace.int64)


def _make_reduce_sdfg_symbolic(axes, wcr, dtype=dace.float64, identity=None):
    """Build a Reduce SDFG with a symbolic leading dimension."""
    in_shape = [N, 16]
    out_shape_list = [s for i, s in enumerate(in_shape) if i not in (axes or [])]
    if not out_shape_list:
        out_shape_list = [1]

    g = SDFG('reduce_cupy_sym')
    g.add_array('A', in_shape, dtype)
    g.add_array('B', out_shape_list, dtype)
    st = g.add_state('main', is_start_block=True)

    a = st.add_access('A')
    b = st.add_access('B')
    r = st.add_reduce(wcr, axes, identity)
    st.add_nedge(a, r, Memlet.from_array('A', g.arrays['A']))
    st.add_nedge(r, b, Memlet.from_array('B', g.arrays['B']))

    return g, r


def _compile_cupy_sdfg(sdfg):
    """Set up SDFG for CuPy (Python backend) and compile.

    :param sdfg: An SDFG with Reduce nodes whose implementation is 'CuPy'.
    :returns:    A compiled SDFG callable.
    """
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.expand_library_nodes()
    return sdfg.compile()


# ---------------------------------------------------------------------------
# Unit tests -- expansion structure (no GPU required)
# ---------------------------------------------------------------------------

class TestExpandReduceCuPyStructure:
    """Structural / unit tests that do NOT need a GPU."""

    def test_registered(self):
        """CuPy must appear in the Reduce implementations dict."""
        assert 'CuPy' in std.Reduce.implementations
        assert std.Reduce.implementations['CuPy'] is ExpandReduceCuPy

    def test_expansion_returns_nsdfg_sum(self):
        """Expanding a sum reduction must return a nested SDFG."""
        sdfg, rnode = _make_reduce_sdfg([8, 4], [1], 'lambda a, b: a + b')
        rnode.implementation = 'CuPy'
        state = sdfg.start_block
        nsdfg = ExpandReduceCuPy.expansion(rnode, state, sdfg)
        assert isinstance(nsdfg, SDFG)
        assert nsdfg.name == 'reduce_cupy'
        # Must contain a Python-language tasklet
        tasklets = [
            n for n in nsdfg.start_block.nodes()
            if isinstance(n, dace.sdfg.nodes.Tasklet)
        ]
        assert len(tasklets) == 1
        assert tasklets[0].language == dace.Language.Python

    def test_expansion_returns_nsdfg_all_axes(self):
        """Reducing all axes (scalar output) must work."""
        sdfg, rnode = _make_reduce_sdfg([4, 6], [0, 1], 'lambda a, b: a + b')
        rnode.implementation = 'CuPy'
        state = sdfg.start_block
        nsdfg = ExpandReduceCuPy.expansion(rnode, state, sdfg)
        assert isinstance(nsdfg, SDFG)
        tasklets = [
            n for n in nsdfg.start_block.nodes()
            if isinstance(n, dace.sdfg.nodes.Tasklet)
        ]
        assert len(tasklets) == 1
        # The CuPy call must use axis=None for full reduction
        code_str = tasklets[0].code.as_string
        assert 'None' in code_str

    def test_fallback_on_unsupported_type(self):
        """Custom WCR should fall back to ExpandReducePure."""
        sdfg, rnode = _make_reduce_sdfg(
            [8, 4], [1], 'lambda a, b: a if a > b else b + 1'
        )
        rnode.implementation = 'CuPy'
        state = sdfg.start_block
        # Should NOT raise -- falls back to pure
        nsdfg = ExpandReduceCuPy.expansion(rnode, state, sdfg)
        assert isinstance(nsdfg, SDFG)
        # The fallback SDFG should NOT be named 'reduce_cupy'
        assert nsdfg.name != 'reduce_cupy'

    def test_fallback_on_degenerate(self):
        """A degenerate reduction (no squeezed axes) must fall back to pure."""
        sdfg, rnode = _make_reduce_sdfg([1, 4], [0], 'lambda a, b: a + b')
        rnode.implementation = 'CuPy'
        state = sdfg.start_block
        nsdfg = ExpandReduceCuPy.expansion(rnode, state, sdfg)
        assert isinstance(nsdfg, SDFG)

    def test_all_supported_reduction_types_in_map(self):
        """Sanity: every supported type has a CuPy template entry."""
        for rtype in _REDUCTION_TYPE_TO_CUPY:
            assert '{inp}' in _REDUCTION_TYPE_TO_CUPY[rtype]
            assert '{axes}' in _REDUCTION_TYPE_TO_CUPY[rtype]

    @pytest.mark.parametrize('redtype', list(_REDUCTION_TYPE_TO_CUPY.keys()))
    def test_expansion_for_each_supported_type(self, redtype):
        """Each supported reduction type must expand without error."""
        # Map reduction type back to a WCR string
        wcr_map = {
            dtypes.ReductionType.Sum: 'lambda a, b: a + b',
            dtypes.ReductionType.Product: 'lambda a, b: a * b',
            dtypes.ReductionType.Min: 'lambda a, b: min(a, b)',
            dtypes.ReductionType.Max: 'lambda a, b: max(a, b)',
            dtypes.ReductionType.Logical_And: 'lambda a, b: a and b',
            dtypes.ReductionType.Logical_Or: 'lambda a, b: a or b',
        }
        wcr = wcr_map.get(redtype)
        if wcr is None:
            pytest.skip(f'No WCR string for {redtype}')
        sdfg, rnode = _make_reduce_sdfg([8, 4], [1], wcr, dtype=dace.float64)
        rnode.implementation = 'CuPy'
        state = sdfg.start_block
        nsdfg = ExpandReduceCuPy.expansion(rnode, state, sdfg)
        assert isinstance(nsdfg, SDFG)
        assert nsdfg.name == 'reduce_cupy'

    def test_fallback_on_bitwise(self):
        """Bitwise reductions should fall back to ExpandReducePure."""
        for wcr in ['lambda a, b: a & b', 'lambda a, b: a | b',
                     'lambda a, b: a ^ b']:
            sdfg, rnode = _make_reduce_sdfg(
                [8, 4], [1], wcr, dtype=dace.int32)
            rnode.implementation = 'CuPy'
            state = sdfg.start_block
            nsdfg = ExpandReduceCuPy.expansion(rnode, state, sdfg)
            assert isinstance(nsdfg, SDFG)
            # Should have fallen back to pure (different name)
            assert nsdfg.name != 'reduce_cupy'


# ---------------------------------------------------------------------------
# Integration tests -- compile + run via the Python backend.
# CuPy internally uses the GPU, so these need @pytest.mark.gpu.
# ---------------------------------------------------------------------------

_wcr_and_np = [
    ('lambda a, b: a + b', np.sum),
    ('lambda a, b: a * b', np.prod),
    ('lambda a, b: min(a, b)', np.min),
    ('lambda a, b: max(a, b)', np.max),
]

_shapes_axes = [
    # (in_shape, axes, description)
    ([64], [0], 'full_1d'),
    ([8, 16], [1], 'partial_2d_last'),
    ([8, 16], [0], 'partial_2d_first'),
    ([8, 16], [0, 1], 'full_2d'),
    ([4, 6, 8], [1], 'partial_3d_middle'),
    ([4, 6, 8], [0, 2], 'partial_3d_noncontiguous'),
    ([4, 6, 8], [0, 1, 2], 'full_3d'),
    ([100], [0], 'large_1d'),
    ([7, 13], [0], 'non_power_of_2'),
]


@pytest.mark.gpu
class TestExpandReduceCuPyGPU:
    """End-to-end tests: expand with CuPy, compile via Python backend, run."""

    @pytest.mark.parametrize(
        'wcr_np,shape_axes',
        list(itertools.product(_wcr_and_np, _shapes_axes)),
    )
    def test_cupy_reduce_e2e(self, wcr_np, shape_axes):
        wcr_str, np_fn = wcr_np
        in_shape, axes, _desc = shape_axes

        sdfg, rnode = _make_reduce_sdfg(
            in_shape, axes, wcr_str, dtype=dace.float64, identity=None,
        )
        rnode.implementation = 'CuPy'

        csdfg = _compile_cupy_sdfg(sdfg)

        a = np.random.rand(*in_shape).astype(np.float64)
        expected = np_fn(a, axis=tuple(axes))

        out_shape = list(expected.shape) if expected.shape else [1]
        b = np.zeros(out_shape, dtype=np.float64)

        csdfg(A=a, B=b)
        del csdfg

        np.testing.assert_allclose(
            b.reshape(expected.shape), expected, rtol=1e-12)

    @pytest.mark.skip(
        reason='Python backend does not propagate symbols that appear '
               'only in nested-SDFG memlet subsets (pre-existing limitation)'
    )
    def test_cupy_reduce_symbolic(self):
        """Reduce with a symbolic leading dimension."""
        sdfg, rnode = _make_reduce_sdfg_symbolic(
            [0], 'lambda a, b: a + b', dtype=dace.float64,
        )
        rnode.implementation = 'CuPy'

        csdfg = _compile_cupy_sdfg(sdfg)

        n_val = 32
        a = np.random.rand(n_val, 16).astype(np.float64)
        expected = np.sum(a, axis=0)
        b = np.zeros(16, dtype=np.float64)

        csdfg(A=a, B=b, N=n_val)
        del csdfg

        np.testing.assert_allclose(b, expected, rtol=1e-12)

    def test_cupy_reduce_min_int32(self):
        """Min reduction on int32 data."""
        sdfg, rnode = _make_reduce_sdfg(
            [8, 4], [1], 'lambda a, b: min(a, b)', dtype=dace.int32,
        )
        rnode.implementation = 'CuPy'

        csdfg = _compile_cupy_sdfg(sdfg)

        a = np.random.randint(0, 256, size=(8, 4), dtype=np.int32)
        expected = np.min(a, axis=1)
        b = np.zeros(8, dtype=np.int32)

        csdfg(A=a, B=b)
        del csdfg

        np.testing.assert_array_equal(b, expected)

    def test_cupy_reduce_float32(self):
        """Sum reduction on float32 data."""
        sdfg, rnode = _make_reduce_sdfg(
            [16, 8], [0], 'lambda a, b: a + b', dtype=dace.float32,
        )
        rnode.implementation = 'CuPy'

        csdfg = _compile_cupy_sdfg(sdfg)

        a = np.random.rand(16, 8).astype(np.float32)
        expected = np.sum(a, axis=0)
        b = np.zeros(8, dtype=np.float32)

        csdfg(A=a, B=b)
        del csdfg

        np.testing.assert_allclose(b, expected, rtol=1e-5)


if __name__ == '__main__':
    # Structural tests
    t = TestExpandReduceCuPyStructure()
    t.test_registered()
    t.test_expansion_returns_nsdfg_sum()
    t.test_expansion_returns_nsdfg_all_axes()
    t.test_fallback_on_unsupported_type()
    t.test_fallback_on_degenerate()
    t.test_all_supported_reduction_types_in_map()
    print('All structural tests passed.')

    # GPU tests (will fail without GPU + CuPy)
    tg = TestExpandReduceCuPyGPU()
    tg.test_cupy_reduce_min_int32()
    tg.test_cupy_reduce_float32()
    print('GPU tests passed.')
