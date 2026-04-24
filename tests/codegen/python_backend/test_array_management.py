# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for PythonCodeGen array management (python_target.py).

Covers allocate_array, deallocate_array, declare_array, copy_memory,
define_out_memlet, and emit_interstate_variable_declaration.
"""
import pytest
import numpy as np

import dace
from dace import dtypes, data
from dace.dtypes import ScheduleType
from dace.sdfg import SDFG, nodes
from dace.memlet import Memlet

def _MAP_XFAIL(func):
    return func


def _make_python_sdfg(name: str) -> SDFG:
    """Create an SDFG with backend set to Python."""
    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    return sdfg


# =============================================================================
# allocate_array -- structural tests
# =============================================================================


class TestAllocateArray:
    """Test allocate_array code generation paths."""

    def test_allocate_scalar_transient(self):
        """Transient scalar -> 'name = 0'."""
        sdfg = _make_python_sdfg('test_alloc_scalar')
        sdfg.add_scalar('tmp', dace.float64, transient=True)
        sdfg.add_scalar('x', dace.float64)
        sdfg.add_scalar('y', dace.float64)
        state = sdfg.add_state('s')
        r = state.add_read('x')
        w = state.add_write('y')
        tmp = state.add_access('tmp')
        t1 = state.add_tasklet('t1', {'a'}, {'b'}, 'b = a + 1')
        t2 = state.add_tasklet('t2', {'c'}, {'d'}, 'd = c')
        state.add_edge(r, None, t1, 'a', Memlet(data='x'))
        state.add_edge(t1, 'b', tmp, None, Memlet(data='tmp'))
        state.add_edge(tmp, None, t2, 'c', Memlet(data='tmp'))
        state.add_edge(t2, 'd', w, None, Memlet(data='y'))
        code = sdfg.generate_code()[0].code
        assert 'tmp = 0' in code

    def test_allocate_array_transient_1d(self):
        """1D float64 transient -> numpy.zeros with correct shape."""
        sdfg = _make_python_sdfg('test_alloc_1d')
        sdfg.add_transient('tmp', [10], dace.float64)
        sdfg.add_array('A', [10], dace.float64)
        sdfg.add_array('B', [10], dace.float64)
        state = sdfg.add_state('s')
        a = state.add_read('A')
        tmp = state.add_access('tmp')
        b = state.add_write('B')
        state.add_edge(a, None, tmp, None, Memlet(data='A', subset='0:10', other_subset='0:10'))
        state.add_edge(tmp, None, b, None, Memlet(data='tmp', subset='0:10', other_subset='0:10'))
        code = sdfg.generate_code()[0].code
        assert 'numpy.zeros' in code
        assert 'float64' in code

    def test_allocate_array_transient_2d(self):
        """2D transient -> correct shape in numpy.zeros."""
        sdfg = _make_python_sdfg('test_alloc_2d')
        sdfg.add_transient('tmp', [3, 5], dace.float64)
        sdfg.add_array('A', [3, 5], dace.float64)
        sdfg.add_array('B', [3, 5], dace.float64)
        state = sdfg.add_state('s')
        a = state.add_read('A')
        tmp = state.add_access('tmp')
        b = state.add_write('B')
        state.add_edge(a, None, tmp, None, Memlet(data='A', subset='0:3, 0:5', other_subset='0:3, 0:5'))
        state.add_edge(tmp, None, b, None, Memlet(data='tmp', subset='0:3, 0:5', other_subset='0:3, 0:5'))
        code = sdfg.generate_code()[0].code
        assert 'numpy.zeros' in code
        assert '3' in code and '5' in code

    def test_allocate_array_transient_int32(self):
        """int32 transient -> numpy.int32 dtype."""
        sdfg = _make_python_sdfg('test_alloc_int32')
        sdfg.add_transient('tmp', [8], dace.int32)
        sdfg.add_array('A', [8], dace.int32)
        sdfg.add_array('B', [8], dace.int32)
        state = sdfg.add_state('s')
        a = state.add_read('A')
        tmp = state.add_access('tmp')
        b = state.add_write('B')
        state.add_edge(a, None, tmp, None, Memlet(data='A', subset='0:8', other_subset='0:8'))
        state.add_edge(tmp, None, b, None, Memlet(data='tmp', subset='0:8', other_subset='0:8'))
        code = sdfg.generate_code()[0].code
        assert 'int32' in code

    def test_allocate_non_transient(self):
        """Non-transient arrays are not allocated -- no numpy.zeros for them."""
        sdfg = _make_python_sdfg('test_alloc_non_trans')
        sdfg.add_array('A', [10], dace.float64)
        sdfg.add_array('B', [10], dace.float64)
        state = sdfg.add_state('s')
        a = state.add_read('A')
        b = state.add_write('B')
        state.add_edge(a, None, b, None, Memlet(data='A', subset='0:10', other_subset='0:10'))
        code = sdfg.generate_code()[0].code
        assert 'A = numpy.zeros' not in code
        assert 'B = numpy.zeros' not in code

    def test_allocate_unsupported_type(self):
        """Unsupported data type (Stream) raises NotImplementedError."""
        sdfg = _make_python_sdfg('test_alloc_stream')
        sdfg.add_stream('S', dace.float64, transient=True)
        sdfg.add_scalar('x', dace.float64)
        state = sdfg.add_state('s')
        s_node = state.add_access('S')
        r = state.add_read('x')
        state.add_edge(r, None, s_node, None, Memlet(data='x'))
        with pytest.raises(NotImplementedError, match="Stream descriptors"):
            sdfg.generate_code()


# =============================================================================
# copy_memory -- structural tests
# =============================================================================


class TestCopyMemory:
    """Test copy_memory code generation paths."""

    def test_copy_array_to_array_full(self):
        """Full array copy with both subsets generates indexed assignment."""
        sdfg = _make_python_sdfg('test_full_copy')
        sdfg.add_array('A', [10], dace.float64)
        sdfg.add_array('B', [10], dace.float64)
        state = sdfg.add_state('s')
        a = state.add_read('A')
        b = state.add_write('B')
        state.add_edge(a, None, b, None, Memlet(data='A', subset='0:10', other_subset='0:10'))
        code = sdfg.generate_code()[0].code
        assert 'A[0:10]' in code and 'B[0:10]' in code

    def test_copy_array_to_array_subset(self):
        """Subset copy A[2:5] -> B[2:5]."""
        sdfg = _make_python_sdfg('test_subset_copy')
        sdfg.add_array('A', [10], dace.float64)
        sdfg.add_array('B', [10], dace.float64)
        state = sdfg.add_state('s')
        a = state.add_read('A')
        b = state.add_write('B')
        state.add_edge(a, None, b, None, Memlet(data='A', subset='2:5', other_subset='2:5'))
        code = sdfg.generate_code()[0].code
        assert 'A[2:5]' in code or 'A[2:4]' in code

    def test_copy_scalar_to_scalar(self):
        """Scalar copy -> plain assignment 'y = x'."""
        sdfg = _make_python_sdfg('test_scalar_copy')
        sdfg.add_scalar('x', dace.float64)
        sdfg.add_scalar('y', dace.float64)
        state = sdfg.add_state('s')
        rx = state.add_read('x')
        wy = state.add_write('y')
        state.add_edge(rx, None, wy, None, Memlet(data='x'))
        code = sdfg.generate_code()[0].code
        assert 'y[...] = x' in code

    def test_copy_array_element_to_scalar(self):
        """Array element to scalar -> 'y = A[3]'."""
        sdfg = _make_python_sdfg('test_elem_to_scalar')
        sdfg.add_array('A', [10], dace.float64)
        sdfg.add_scalar('y', dace.float64)
        state = sdfg.add_state('s')
        a = state.add_read('A')
        w = state.add_write('y')
        state.add_edge(a, None, w, None, Memlet(data='A', subset='3'))
        code = sdfg.generate_code()[0].code
        assert 'A[3]' in code

    def test_copy_no_dst_subset(self):
        """dst_subset is None -> plain name for dst."""
        sdfg = _make_python_sdfg('test_no_dst_sub')
        sdfg.add_array('A', [10], dace.float64)
        sdfg.add_array('B', [10], dace.float64)
        state = sdfg.add_state('s')
        a = state.add_read('A')
        b = state.add_write('B')
        state.add_edge(a, None, b, None, Memlet(data='A', subset='0:10'))
        code = sdfg.generate_code()[0].code
        assert 'A[0:10]' in code
        assert 'numpy.copyto(B, A[0:10])' in code


# =============================================================================
# No-op methods -- verify they don't crash
# =============================================================================


class TestNoOps:
    """Test that no-op methods (declare, deallocate, etc.) don't crash."""

    def test_declare_array_noop(self):
        """declare_array doesn't crash when called via code gen."""
        sdfg = _make_python_sdfg('test_declare')
        sdfg.add_scalar('x', dace.float64)
        sdfg.add_scalar('y', dace.float64)
        state = sdfg.add_state('s')
        r = state.add_read('x')
        w = state.add_write('y')
        t = state.add_tasklet('t', {'a'}, {'b'}, 'b = a')
        state.add_edge(r, None, t, 'a', Memlet(data='x'))
        state.add_edge(t, 'b', w, None, Memlet(data='y'))
        code_objs = sdfg.generate_code()
        assert len(code_objs) > 0

    def test_deallocate_array_noop(self):
        """deallocate_array doesn't crash when called via code gen."""
        sdfg = _make_python_sdfg('test_dealloc')
        sdfg.add_transient('tmp', [5], dace.float64)
        sdfg.add_array('A', [5], dace.float64)
        sdfg.add_array('B', [5], dace.float64)
        state = sdfg.add_state('s')
        a = state.add_read('A')
        tmp = state.add_access('tmp')
        b = state.add_write('B')
        state.add_edge(a, None, tmp, None, Memlet(data='A', subset='0:5', other_subset='0:5'))
        state.add_edge(tmp, None, b, None, Memlet(data='tmp', subset='0:5', other_subset='0:5'))
        code_objs = sdfg.generate_code()
        assert len(code_objs) > 0

    def test_define_out_memlet_noop(self):
        """define_out_memlet is a no-op, should not affect code gen."""
        sdfg = _make_python_sdfg('test_def_out')
        sdfg.add_scalar('x', dace.float64)
        sdfg.add_scalar('y', dace.float64)
        state = sdfg.add_state('s')
        r = state.add_read('x')
        w = state.add_write('y')
        t = state.add_tasklet('t', {'a'}, {'b'}, 'b = a')
        state.add_edge(r, None, t, 'a', Memlet(data='x'))
        state.add_edge(t, 'b', w, None, Memlet(data='y'))
        code_objs = sdfg.generate_code()
        assert len(code_objs) > 0

    def test_emit_interstate_variable_declaration_noop(self):
        """emit_interstate_variable_declaration is a no-op, code gen works."""
        sdfg = _make_python_sdfg('test_emit_isv')
        sdfg.add_array('y', [1], dace.float64)

        s1 = sdfg.add_state('s1')
        s2 = sdfg.add_state('s2')
        sdfg.add_edge(s1, s2, dace.InterstateEdge(assignments={'tmp': '42'}))

        wy = s2.add_write('y')
        t = s2.add_tasklet('t', set(), {'out'}, 'out = tmp')
        s2.add_edge(t, 'out', wy, None, Memlet(data='y', subset='0'))

        code_objs = sdfg.generate_code()
        assert len(code_objs) > 0


# =============================================================================
# Correctness tests -- compile, run, verify against numpy
# =============================================================================


class TestArrayManagementCorrectness:
    """End-to-end correctness tests for array allocation, copy, and management."""

    def test_copy_correctness(self):
        """Array copy via access nodes: compile, run, verify against numpy."""
        N = 10
        sdfg = _make_python_sdfg('test_copy_corr')
        sdfg.add_array('A', [N], dace.float64)
        sdfg.add_array('B', [N], dace.float64)
        state = sdfg.add_state('s')
        a = state.add_read('A')
        b = state.add_write('B')
        state.add_edge(a, None, b, None, Memlet(data='A', subset='0:10', other_subset='0:10'))
        csdfg = sdfg.compile()
        A = np.arange(N, dtype=np.float64)
        B = np.zeros(N, dtype=np.float64)
        csdfg(A=A, B=B)
        np.testing.assert_array_equal(B, A)

    def test_transient_scalar_correctness(self):
        """Transient scalar used as intermediate: compile and verify."""
        sdfg = _make_python_sdfg('test_trans_scalar')
        sdfg.add_array('x', [1], dace.float64)
        sdfg.add_array('y', [1], dace.float64)
        sdfg.add_scalar('tmp', dace.float64, transient=True)
        state = sdfg.add_state('s')
        rx = state.add_read('x')
        wy = state.add_write('y')
        tmp_node = state.add_access('tmp')
        t1 = state.add_tasklet('t1', {'a'}, {'b'}, 'b = a * 5')
        t2 = state.add_tasklet('t2', {'c'}, {'d'}, 'd = c + 3')
        state.add_edge(rx, None, t1, 'a', Memlet(data='x', subset='0'))
        state.add_edge(t1, 'b', tmp_node, None, Memlet(data='tmp'))
        state.add_edge(tmp_node, None, t2, 'c', Memlet(data='tmp'))
        state.add_edge(t2, 'd', wy, None, Memlet(data='y', subset='0'))
        csdfg = sdfg.compile()
        x = np.array([4.0], dtype=np.float64)
        y = np.array([0.0], dtype=np.float64)
        csdfg(x=x, y=y)
        assert y[0] == 23.0

    def test_transient_array_correctness(self):
        """Transient array as intermediate: compile and verify."""
        N = 8
        sdfg = _make_python_sdfg('test_trans_array')
        sdfg.add_array('A', [N], dace.float64)
        sdfg.add_array('B', [N], dace.float64)
        sdfg.add_transient('tmp', [N], dace.float64)
        state = sdfg.add_state('s')
        a = state.add_read('A')
        tmp = state.add_access('tmp')
        b = state.add_write('B')
        state.add_edge(a, None, tmp, None, Memlet(data='A', subset='0:8', other_subset='0:8'))
        state.add_edge(tmp, None, b, None, Memlet(data='tmp', subset='0:8', other_subset='0:8'))
        csdfg = sdfg.compile()
        A = np.arange(N, dtype=np.float64) * 2
        B = np.zeros(N, dtype=np.float64)
        csdfg(A=A, B=B)
        np.testing.assert_array_equal(B, A)

    def test_2d_transient_array_correctness(self):
        """2D transient array: compile and verify."""
        R, C = 3, 4
        sdfg = _make_python_sdfg('test_trans_2d')
        sdfg.add_array('A', [R, C], dace.float64)
        sdfg.add_array('B', [R, C], dace.float64)
        sdfg.add_transient('tmp', [R, C], dace.float64)
        state = sdfg.add_state('s')
        a = state.add_read('A')
        tmp = state.add_access('tmp')
        b = state.add_write('B')
        state.add_edge(a, None, tmp, None, Memlet(data='A', subset='0:3, 0:4', other_subset='0:3, 0:4'))
        state.add_edge(tmp, None, b, None, Memlet(data='tmp', subset='0:3, 0:4', other_subset='0:3, 0:4'))
        csdfg = sdfg.compile()
        A = np.random.rand(R, C)
        B = np.zeros((R, C))
        csdfg(A=A, B=B)
        np.testing.assert_array_equal(B, A)

    def test_int32_transient_correctness(self):
        """int32 transient array: compile and verify dtype preserved."""
        N = 6
        sdfg = _make_python_sdfg('test_int32_trans')
        sdfg.add_array('A', [N], dace.int32)
        sdfg.add_array('B', [N], dace.int32)
        sdfg.add_transient('tmp', [N], dace.int32)
        state = sdfg.add_state('s')
        a = state.add_read('A')
        tmp = state.add_access('tmp')
        b = state.add_write('B')
        state.add_edge(a, None, tmp, None, Memlet(data='A', subset='0:6', other_subset='0:6'))
        state.add_edge(tmp, None, b, None, Memlet(data='tmp', subset='0:6', other_subset='0:6'))
        csdfg = sdfg.compile()
        A = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
        B = np.zeros(N, dtype=np.int32)
        csdfg(A=A, B=B)
        np.testing.assert_array_equal(B, A)

    @_MAP_XFAIL
    def test_map_with_transient_intermediate(self):
        """Map with transient intermediate: A->map->tmp->map->B."""
        N = 10
        sdfg = _make_python_sdfg('test_map_trans')
        sdfg.add_array('A', [N], dace.float64)
        sdfg.add_array('B', [N], dace.float64)
        sdfg.add_transient('tmp', [N], dace.float64)

        s1 = sdfg.add_state('s1')
        me1, mx1 = s1.add_map('m1', {'i': '0:10'}, schedule=ScheduleType.Sequential)
        a = s1.add_read('A')
        tmp_w = s1.add_write('tmp')
        t1 = s1.add_tasklet('double', {'x'}, {'y'}, 'y = x * 2')
        s1.add_memlet_path(a, me1, t1, dst_conn='x', memlet=Memlet(data='A', subset='i'))
        s1.add_memlet_path(t1, mx1, tmp_w, src_conn='y', memlet=Memlet(data='tmp', subset='i'))

        s2 = sdfg.add_state('s2')
        me2, mx2 = s2.add_map('m2', {'i': '0:10'}, schedule=ScheduleType.Sequential)
        tmp_r = s2.add_read('tmp')
        b = s2.add_write('B')
        t2 = s2.add_tasklet('inc', {'x'}, {'y'}, 'y = x + 1')
        s2.add_memlet_path(tmp_r, me2, t2, dst_conn='x', memlet=Memlet(data='tmp', subset='i'))
        s2.add_memlet_path(t2, mx2, b, src_conn='y', memlet=Memlet(data='B', subset='i'))

        sdfg.add_edge(s1, s2, dace.InterstateEdge())
        csdfg = sdfg.compile()
        A = np.random.rand(N)
        B = np.zeros(N)
        csdfg(A=A, B=B)
        np.testing.assert_allclose(B, A * 2 + 1)

    def test_multi_state_copy_correctness(self):
        """Multi-state with copies: S1 copies A->tmp, S2 copies tmp->B."""
        N = 5
        sdfg = _make_python_sdfg('test_multi_state_copy')
        sdfg.add_array('A', [N], dace.float64)
        sdfg.add_array('B', [N], dace.float64)
        sdfg.add_transient('tmp', [N], dace.float64)

        s1 = sdfg.add_state('s1')
        a = s1.add_read('A')
        tmp_w = s1.add_write('tmp')
        s1.add_edge(a, None, tmp_w, None, Memlet(data='A', subset='0:5', other_subset='0:5'))

        s2 = sdfg.add_state('s2')
        tmp_r = s2.add_read('tmp')
        b = s2.add_write('B')
        s2.add_edge(tmp_r, None, b, None, Memlet(data='tmp', subset='0:5', other_subset='0:5'))

        sdfg.add_edge(s1, s2, dace.InterstateEdge())
        csdfg = sdfg.compile()
        A = np.array([10, 20, 30, 40, 50], dtype=np.float64)
        B = np.zeros(N, dtype=np.float64)
        csdfg(A=A, B=B)
        np.testing.assert_array_equal(B, A)

    def test_scalar_copy_correctness(self):
        """Scalar copy: x -> y via access nodes, compile and verify."""
        sdfg = _make_python_sdfg('test_scalar_copy_corr')
        sdfg.add_array('x', [1], dace.float64)
        sdfg.add_array('y', [1], dace.float64)
        state = sdfg.add_state('s')
        rx = state.add_read('x')
        wy = state.add_write('y')
        state.add_edge(rx, None, wy, None, Memlet(data='x', subset='0', other_subset='0'))
        csdfg = sdfg.compile()
        x = np.array([99.0], dtype=np.float64)
        y = np.array([0.0], dtype=np.float64)
        csdfg(x=x, y=y)
        assert y[0] == 99.0
