# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for DaCe C++ codegen handling of negative and symbolic map steps."""

import numpy as np

import dace
from dace import dtypes


def _make_add_sdfg(name, map_range_dict, array_size=6):
    """Build SDFG: C[i] = A[i] + B[i] under a single Sequential map."""
    sdfg = dace.SDFG(name)
    state = sdfg.add_state()
    sdfg.add_array('A', [array_size], dace.float64)
    sdfg.add_array('B', [array_size], dace.float64)
    sdfg.add_array('C', [array_size], dace.float64)

    me, mx = state.add_map('compute', map_range_dict,
                           schedule=dtypes.ScheduleType.Sequential)
    tasklet = state.add_tasklet('add', {'a', 'b'}, {'c'}, 'c = a + b')

    a_read = state.add_read('A')
    b_read = state.add_read('B')
    c_write = state.add_write('C')

    state.add_memlet_path(a_read, me, tasklet, dst_conn='a',
                          memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(b_read, me, tasklet, dst_conn='b',
                          memlet=dace.Memlet('B[i]'))
    state.add_memlet_path(tasklet, mx, c_write, src_conn='c',
                          memlet=dace.Memlet('C[i]'))
    return sdfg


def _run_add(sdfg, expected_indices, **kwargs):
    """Compile, run, and verify that only expected_indices were written."""
    A = np.arange(6, dtype=np.float64)
    B = np.arange(6, dtype=np.float64) * 10
    C = np.full(6, -1.0, dtype=np.float64)

    sdfg(A=A, B=B, C=C, **kwargs)

    expected = np.full(6, -1.0)
    for idx in expected_indices:
        expected[idx] = A[idx] + B[idx]
    np.testing.assert_array_equal(C, expected)


def test_map_positive_step_fixed():
    """Fixed positive step '1:6:2' -> iterations [1, 3, 5]."""
    sdfg = _make_add_sdfg('pos_step_fixed', {'i': '1:6:2'})
    _run_add(sdfg, [1, 3, 5])


def test_map_negative_step_fixed():
    """Fixed negative step '5:0:-2' -> iterations [5, 3, 1]."""
    sdfg = _make_add_sdfg('neg_step_fixed', {'i': '5:0:-2'})
    _run_add(sdfg, [5, 3, 1])


def test_map_symbolic_step_positive():
    """Symbolic step '1:6:S' with S=2 -> iterations [1, 3, 5]."""
    sdfg = _make_add_sdfg('sym_step_pos', {'i': '1:6:S'})
    sdfg.add_symbol('S', dace.int32)
    _run_add(sdfg, [1, 3, 5], S=np.int32(2))


def test_map_symbolic_step_negative():
    """Symbolic step with negative runtime value.

    Construct range (5, 1, S) programmatically so the inclusive end is correct
    for downward iteration.  With S=-2 -> iterations [5, 3, 1].
    """
    sdfg = _make_add_sdfg('sym_step_neg', {'i': '5:0:-1'})  # placeholder
    sdfg.add_symbol('S', dace.int32)
    # Overwrite map range to (5, 1, S) — inclusive on both ends
    state = sdfg.states()[0]
    for node in state.nodes():
        if isinstance(node, dace.sdfg.nodes.MapEntry):
            node.map.range = dace.subsets.Range([(5, 1, dace.symbol('S'))])
            break
    _run_add(sdfg, [5, 3, 1], S=np.int32(-2))


def test_map_symbolic_step_unknown_sign():
    """Symbolic step '1:6:S' tested with positive and negative S.

    S=2:  iterations [1, 3, 5]
    S=-2: empty loop (start < end, stepping backward -> no iterations)
    """
    sdfg = _make_add_sdfg('sym_step_unknown', {'i': '1:6:S'})
    sdfg.add_symbol('S', dace.int32)
    # Positive step
    _run_add(sdfg, [1, 3, 5], S=np.int32(2))
    # Negative step — should produce an empty loop
    _run_add(sdfg, [], S=np.int32(-2))


def test_map_step_one():
    """Map with step=1 (default). Baseline regression test."""
    sdfg = _make_add_sdfg('step_one', {'i': '0:6'})
    _run_add(sdfg, [0, 1, 2, 3, 4, 5])


if __name__ == '__main__':
    test_map_step_one()
    test_map_positive_step_fixed()
    test_map_negative_step_fixed()
    test_map_symbolic_step_positive()
    test_map_symbolic_step_negative()
    test_map_symbolic_step_unknown_sign()
