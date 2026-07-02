# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the Python backend's nested-SDFG connector shape reconciliation.

A nested connector whose shape differs from the outer memlet subset (e.g. a
reshape View rewritten by RemoveViews into a differently-shaped slice) is
reconciled with a flat C-order reshape -- but only when that is provably
order-preserving (same element count and C-contiguous connector strides).
Permutation-type mismatches must raise loudly instead of silently transposing
data, and output-side mismatches must land writes in the outer array.
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.memlet import Memlet


def _build_scale_sdfg(name: str,
                      conn_in_shape,
                      conn_in_strides,
                      conn_out_shape,
                      conn_out_strides,
                      outer_shape=(2, 3, 4)) -> dace.SDFG:
    """Outer arrays ``X``/``Y`` of ``outer_shape``; a nested SDFG computes
    ``_y = 2 * _x`` with the given connector shapes/strides.

    :param name: SDFG name.
    :param conn_in_shape: Nested input connector shape.
    :param conn_in_strides: Nested input connector strides.
    :param conn_out_shape: Nested output connector shape.
    :param conn_out_strides: Nested output connector strides.
    :param outer_shape: Shape of the outer arrays.
    :returns: The configured (Python-backend) SDFG.
    """
    sdfg = dace.SDFG(name)
    state = sdfg.add_state()
    sdfg.add_array('X', outer_shape, dace.float64)
    sdfg.add_array('Y', outer_shape, dace.float64)

    nsdfg = dace.SDFG(name + '_inner')
    nstate = nsdfg.add_state()
    nsdfg.add_array('_x', conn_in_shape, dace.float64, strides=conn_in_strides)
    nsdfg.add_array('_y', conn_out_shape, dace.float64, strides=conn_out_strides)
    t = nstate.add_tasklet('scale', {'__i'}, {'__o'}, '__o = 2 * __i')
    nstate.add_edge(nstate.add_read('_x'), None, t, '__i', Memlet.from_array('_x', nsdfg.arrays['_x']))
    nstate.add_edge(t, '__o', nstate.add_write('_y'), None, Memlet.from_array('_y', nsdfg.arrays['_y']))

    node = state.add_nested_sdfg(nsdfg, {'_x'}, {'_y'})
    state.add_edge(state.add_read('X'), None, node, '_x', Memlet.from_array('X', sdfg.arrays['X']))
    state.add_edge(node, '_y', state.add_write('Y'), None, Memlet.from_array('Y', sdfg.arrays['Y']))
    sdfg.backend = dtypes.BackendLanguage.Python
    return sdfg


def test_output_side_flat_reshape():
    """A flat-order-preserving OUTPUT connector mismatch writes back correctly.

    Outer ``(2, 3, 4)`` vs C-contiguous ``(6, 4)`` connectors on both sides:
    writes through the reshaped output must land in the outer array (bug 08b:
    they were previously unhandled).
    """
    sdfg = _build_scale_sdfg('nested_out_reshape', (6, 4), (4, 1), (6, 4), (4, 1))
    X = np.random.rand(2, 3, 4)
    Y = np.zeros((2, 3, 4))
    sdfg(X=X, Y=Y)
    assert np.allclose(Y, 2 * X), f"max diff = {np.max(np.abs(Y - 2 * X))}"


def test_input_permutation_mismatch_raises():
    """A permutation-type INPUT mismatch raises instead of transposing data.

    Connector shape ``(4, 6)`` with transposed strides ``(1, 4)`` is not a flat
    C-order reinterpretation of the outer ``(2, 3, 4)`` slice; a flat reshape
    would silently produce transposed data (bug 08b).
    """
    sdfg = _build_scale_sdfg('nested_perm_mismatch', (4, 6), (1, 4), (6, 4), (4, 1))
    with pytest.raises(NotImplementedError, match='flat-order-preserving'):
        sdfg.compile()


def test_output_permutation_mismatch_raises():
    """A permutation-type OUTPUT mismatch raises loudly too."""
    sdfg = _build_scale_sdfg('nested_perm_out_mismatch', (6, 4), (4, 1), (4, 6), (1, 4))
    with pytest.raises(NotImplementedError, match='flat-order-preserving'):
        sdfg.compile()


def test_element_count_mismatch_raises():
    """A same-rank different-size mismatch raises (no silent wrong indexing)."""
    sdfg = _build_scale_sdfg('nested_count_mismatch', (5, 4), (4, 1), (6, 4), (4, 1))
    with pytest.raises(NotImplementedError, match='flat-order-preserving'):
        sdfg.compile()


def test_matching_shapes_unchanged():
    """Identical connector/outer shapes need no reconciliation."""
    sdfg = _build_scale_sdfg('nested_matching', (2, 3, 4), (12, 4, 1), (2, 3, 4), (12, 4, 1))
    X = np.random.rand(2, 3, 4)
    Y = np.zeros((2, 3, 4))
    sdfg(X=X, Y=Y)
    assert np.allclose(Y, 2 * X)


def test_inout_connector_flat_reshape():
    """An in/out connector with a flat reshape mismatch reads AND writes back.

    ``_y`` is both input and output of the nested SDFG (``_y = 2*_x + _y``),
    bound through the same contiguous bridge with a copy-back.
    """
    sdfg = dace.SDFG('nested_inout_reshape')
    state = sdfg.add_state()
    sdfg.add_array('X', (2, 3, 4), dace.float64)
    sdfg.add_array('Y', (2, 3, 4), dace.float64)

    nsdfg = dace.SDFG('nested_inout_reshape_inner')
    nstate = nsdfg.add_state()
    nsdfg.add_array('_x', (6, 4), dace.float64, strides=(4, 1))
    nsdfg.add_array('_y', (6, 4), dace.float64, strides=(4, 1))
    t = nstate.add_tasklet('axpy', {'__i', '__c'}, {'__o'}, '__o = 2 * __i + __c')
    nstate.add_edge(nstate.add_read('_x'), None, t, '__i', Memlet.from_array('_x', nsdfg.arrays['_x']))
    nstate.add_edge(nstate.add_read('_y'), None, t, '__c', Memlet.from_array('_y', nsdfg.arrays['_y']))
    nstate.add_edge(t, '__o', nstate.add_write('_y'), None, Memlet.from_array('_y', nsdfg.arrays['_y']))

    node = state.add_nested_sdfg(nsdfg, {'_x', '_y'}, {'_y'})
    state.add_edge(state.add_read('X'), None, node, '_x', Memlet.from_array('X', sdfg.arrays['X']))
    state.add_edge(state.add_read('Y'), None, node, '_y', Memlet.from_array('Y', sdfg.arrays['Y']))
    state.add_edge(node, '_y', state.add_write('Y'), None, Memlet.from_array('Y', sdfg.arrays['Y']))
    sdfg.backend = dtypes.BackendLanguage.Python

    X = np.random.rand(2, 3, 4)
    Y0 = np.random.rand(2, 3, 4)
    Y = Y0.copy()
    sdfg(X=X, Y=Y)
    ref = 2 * X + Y0
    assert np.allclose(Y, ref), f"max diff = {np.max(np.abs(Y - ref))}"


def test_rendered_column_slice_needs_no_reshape():
    """A strided column slice whose RENDERED shape matches the connector
    (correlation regression).

    The outer memlet ``X[0:5, 2]`` has subset size ``(5, 1)`` but renders to
    a shape-``(5,)`` view (the size-1 dim collapses to an index), exactly
    matching the 1-D connector ``_x`` with column stride ``(7,)``.  Before
    the fix this raised ``NotImplementedError`` (non-contiguous strides
    rejected the flat reshape that is not actually needed).
    """
    sdfg = dace.SDFG('nested_column_slice')
    state = sdfg.add_state()
    sdfg.add_array('X', (5, 7), dace.float64)
    sdfg.add_array('Y', (5, ), dace.float64)

    nsdfg = dace.SDFG('nested_column_slice_inner')
    nstate = nsdfg.add_state()
    nsdfg.add_array('_x', (5, ), dace.float64, strides=(7, ))
    nsdfg.add_array('_y', (5, ), dace.float64)
    t = nstate.add_tasklet('scale', {'__i'}, {'__o'}, '__o = 2 * __i')
    nstate.add_edge(nstate.add_read('_x'), None, t, '__i', Memlet.from_array('_x', nsdfg.arrays['_x']))
    nstate.add_edge(t, '__o', nstate.add_write('_y'), None, Memlet.from_array('_y', nsdfg.arrays['_y']))

    node = state.add_nested_sdfg(nsdfg, {'_x'}, {'_y'})
    state.add_edge(state.add_read('X'), None, node, '_x', Memlet('X[0:5, 2]'))
    state.add_edge(node, '_y', state.add_write('Y'), None, Memlet.from_array('Y', sdfg.arrays['Y']))
    sdfg.backend = dtypes.BackendLanguage.Python

    X = np.random.rand(5, 7)
    Y = np.zeros(5)
    sdfg(X=X, Y=Y)
    assert np.allclose(Y, 2 * X[:, 2])


def test_writeback_through_collapsed_dim():
    """A reshape-bridge writeback into a target with a collapsed size-1 dim
    (mlp regression).

    The outer memlet ``Y[0:6, 0]`` renders to a shape-``(6,)`` target, so
    the bridge writeback must reshape to ``(6,)``, not the raw subset size
    ``(6, 1)`` (which raised a cupy/numpy shape mismatch).
    """
    sdfg = dace.SDFG('nested_collapsed_writeback')
    state = sdfg.add_state()
    sdfg.add_array('X', (2, 3), dace.float64)
    sdfg.add_array('Y', (6, 2), dace.float64)

    nsdfg = dace.SDFG('nested_collapsed_writeback_inner')
    nstate = nsdfg.add_state()
    nsdfg.add_array('_x', (2, 3), dace.float64, strides=(3, 1))
    nsdfg.add_array('_y', (2, 3), dace.float64, strides=(3, 1))
    t = nstate.add_tasklet('scale', {'__i'}, {'__o'}, '__o = 2 * __i')
    nstate.add_edge(nstate.add_read('_x'), None, t, '__i', Memlet.from_array('_x', nsdfg.arrays['_x']))
    nstate.add_edge(t, '__o', nstate.add_write('_y'), None, Memlet.from_array('_y', nsdfg.arrays['_y']))

    node = state.add_nested_sdfg(nsdfg, {'_x'}, {'_y'})
    state.add_edge(state.add_read('X'), None, node, '_x', Memlet.from_array('X', sdfg.arrays['X']))
    state.add_edge(node, '_y', state.add_write('Y'), None, Memlet('Y[0:6, 0]'))
    sdfg.backend = dtypes.BackendLanguage.Python

    X = np.random.rand(2, 3)
    Y = np.zeros((6, 2))
    sdfg(X=X, Y=Y)
    assert np.allclose(Y[:, 0], (2 * X).reshape(6))
    assert np.allclose(Y[:, 1], 0.0)


if __name__ == '__main__':
    test_output_side_flat_reshape()
    test_input_permutation_mismatch_raises()
    test_output_permutation_mismatch_raises()
    test_element_count_mismatch_raises()
    test_matching_shapes_unchanged()
    test_inout_connector_flat_reshape()
    test_rendered_column_slice_needs_no_reshape()
    test_writeback_through_collapsed_dim()
