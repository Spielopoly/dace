# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for PythonCodeGen scope/map generation (python_target.py).

Covers generate_scope for single/multi-param maps, stepped maps,
symbolic bounds, nested maps, and end-to-end correctness.
"""
import pytest
import numpy as np

import dace
from dace import dtypes, data
from dace.dtypes import ScheduleType
from dace.sdfg import SDFG
from dace.memlet import Memlet

def _MAP_XFAIL(func):
    return func


def _make_python_sdfg(name: str) -> SDFG:
    """Create an SDFG with backend set to Python."""
    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    return sdfg


# =============================================================================
# Structural tests -- verify generated code patterns
# =============================================================================


class TestMapCodeGeneration:
    """Test that map generation produces correct Python loop structures."""

    @_MAP_XFAIL
    def test_single_param_map(self):
        """1D map generates a single for loop."""
        sdfg = _make_python_sdfg('test_1d_map')
        sdfg.add_array('A', [10], dace.float64)
        sdfg.add_array('B', [10], dace.float64)
        state = sdfg.add_state('s')
        me, mx = state.add_map('m', {'i': '0:10'}, schedule=ScheduleType.Sequential)
        a = state.add_read('A')
        b = state.add_write('B')
        t = state.add_tasklet('t', {'inp'}, {'out'}, 'out = inp')
        state.add_memlet_path(a, me, t, dst_conn='inp', memlet=Memlet(data='A', subset='i'))
        state.add_memlet_path(t, mx, b, src_conn='out', memlet=Memlet(data='B', subset='i'))
        code = sdfg.generate_code()[0].code
        assert 'for i in range' in code

    @_MAP_XFAIL
    def test_multi_param_map(self):
        """2D map generates nested for loops."""
        sdfg = _make_python_sdfg('test_2d_map')
        sdfg.add_array('A', [4, 6], dace.float64)
        sdfg.add_array('B', [4, 6], dace.float64)
        state = sdfg.add_state('s')
        me, mx = state.add_map('m', {'i': '0:4', 'j': '0:6'}, schedule=ScheduleType.Sequential)
        a = state.add_read('A')
        b = state.add_write('B')
        t = state.add_tasklet('t', {'inp'}, {'out'}, 'out = inp + 1')
        state.add_memlet_path(a, me, t, dst_conn='inp', memlet=Memlet(data='A', subset='i, j'))
        state.add_memlet_path(t, mx, b, src_conn='out', memlet=Memlet(data='B', subset='i, j'))
        code = sdfg.generate_code()[0].code
        assert 'for i in range' in code
        assert 'for j in range' in code

    @_MAP_XFAIL
    def test_map_with_step(self):
        """Non-unit step generates correct range."""
        sdfg = _make_python_sdfg('test_step_map')
        sdfg.add_array('A', [10], dace.float64)
        sdfg.add_array('B', [5], dace.float64)
        state = sdfg.add_state('s')
        me, mx = state.add_map('m', {'i': '0:10:2'}, schedule=ScheduleType.Sequential)
        a = state.add_read('A')
        b = state.add_write('B')
        t = state.add_tasklet('t', {'inp'}, {'out'}, 'out = inp')
        state.add_memlet_path(a, me, t, dst_conn='inp', memlet=Memlet(data='A', subset='i'))
        state.add_memlet_path(t, mx, b, src_conn='out', memlet=Memlet(data='B', subset='i // 2'))
        code = sdfg.generate_code()[0].code
        assert '2)' in code or ', 2)' in code

    @_MAP_XFAIL
    def test_map_with_symbolic_bounds(self):
        """Symbolic bounds appear in generated range."""
        sdfg = _make_python_sdfg('test_sym_map')
        sdfg.add_symbol('N', dace.int64)
        sdfg.add_array('A', [dace.symbol('N')], dace.float64)
        sdfg.add_array('B', [dace.symbol('N')], dace.float64)
        state = sdfg.add_state('s')
        me, mx = state.add_map('m', {'i': '0:N'}, schedule=ScheduleType.Sequential)
        a = state.add_read('A')
        b = state.add_write('B')
        t = state.add_tasklet('t', {'inp'}, {'out'}, 'out = inp')
        state.add_memlet_path(a, me, t, dst_conn='inp', memlet=Memlet(data='A', subset='i'))
        state.add_memlet_path(t, mx, b, src_conn='out', memlet=Memlet(data='B', subset='i'))
        code = sdfg.generate_code()[0].code
        assert 'N' in code
        assert 'for i in range' in code

    @_MAP_XFAIL
    def test_map_body_with_tasklet(self):
        """Map body with tasklet generates tasklet code inside loop."""
        sdfg = _make_python_sdfg('test_map_tasklet')
        sdfg.add_array('A', [8], dace.float64)
        sdfg.add_array('B', [8], dace.float64)
        state = sdfg.add_state('s')
        me, mx = state.add_map('m', {'i': '0:8'}, schedule=ScheduleType.Sequential)
        a = state.add_read('A')
        b = state.add_write('B')
        t = state.add_tasklet('square', {'x'}, {'y'}, 'y = x * x')
        state.add_memlet_path(a, me, t, dst_conn='x', memlet=Memlet(data='A', subset='i'))
        state.add_memlet_path(t, mx, b, src_conn='y', memlet=Memlet(data='B', subset='i'))
        code = sdfg.generate_code()[0].code
        assert 'y = x * x' in code or 'y = (x * x)' in code

    @_MAP_XFAIL
    def test_nested_maps(self):
        """Map inside a map (nested maps) generates nested for loops."""
        N, M = 4, 6
        sdfg = _make_python_sdfg('test_nested')
        sdfg.add_array('A', [N, M], dace.float64)
        sdfg.add_array('B', [N, M], dace.float64)
        state = sdfg.add_state('s')
        ome, omx = state.add_map('outer', {'i': '0:4'}, schedule=ScheduleType.Sequential)
        ime, imx = state.add_map('inner', {'j': '0:6'}, schedule=ScheduleType.Sequential)
        a = state.add_read('A')
        b = state.add_write('B')
        t = state.add_tasklet('t', {'inp'}, {'out'}, 'out = inp + 1')
        state.add_memlet_path(a, ome, ime, t, dst_conn='inp', memlet=Memlet(data='A', subset='i, j'))
        state.add_memlet_path(t, imx, omx, b, src_conn='out', memlet=Memlet(data='B', subset='i, j'))
        code = sdfg.generate_code()[0].code
        assert 'for i in range' in code
        assert 'for j in range' in code

    @_MAP_XFAIL
    def test_map_with_multiple_access_nodes(self):
        """Map reading/writing multiple arrays."""
        N = 10
        sdfg = _make_python_sdfg('test_multi_access')
        sdfg.add_array('A', [N], dace.float64)
        sdfg.add_array('B', [N], dace.float64)
        sdfg.add_array('C', [N], dace.float64)
        state = sdfg.add_state('s')
        me, mx = state.add_map('m', {'i': '0:10'}, schedule=ScheduleType.Sequential)
        ra = state.add_read('A')
        rb = state.add_read('B')
        wc = state.add_write('C')
        t = state.add_tasklet('add', {'x', 'y'}, {'z'}, 'z = x + y')
        state.add_memlet_path(ra, me, t, dst_conn='x', memlet=Memlet(data='A', subset='i'))
        state.add_memlet_path(rb, me, t, dst_conn='y', memlet=Memlet(data='B', subset='i'))
        state.add_memlet_path(t, mx, wc, src_conn='z', memlet=Memlet(data='C', subset='i'))
        code = sdfg.generate_code()[0].code
        assert 'z = x + y' in code or 'z = (x + y)' in code


# =============================================================================
# Correctness tests -- compile, run, verify against numpy
# =============================================================================


class TestMapCorrectness:
    """End-to-end correctness tests for maps."""

    @_MAP_XFAIL
    def test_map_correctness_1d(self):
        """1D map: element-wise double, verify against numpy."""
        N = 15
        sdfg = _make_python_sdfg('test_1d_corr')
        sdfg.add_array('A', [N], dace.float64)
        sdfg.add_array('B', [N], dace.float64)
        state = sdfg.add_state('s')
        me, mx = state.add_map('m', {'i': '0:15'}, schedule=ScheduleType.Sequential)
        a = state.add_read('A')
        b = state.add_write('B')
        t = state.add_tasklet('t', {'inp'}, {'out'}, 'out = inp * 2.0')
        state.add_memlet_path(a, me, t, dst_conn='inp', memlet=Memlet(data='A', subset='i'))
        state.add_memlet_path(t, mx, b, src_conn='out', memlet=Memlet(data='B', subset='i'))
        csdfg = sdfg.compile()
        A = np.random.rand(N)
        B = np.zeros(N)
        csdfg(A=A, B=B)
        np.testing.assert_allclose(B, A * 2.0)

    @_MAP_XFAIL
    def test_map_correctness_2d(self):
        """2D map: element-wise add with constant, verify against numpy."""
        R, C = 5, 7
        sdfg = _make_python_sdfg('test_2d_corr')
        sdfg.add_array('A', [R, C], dace.float64)
        sdfg.add_array('B', [R, C], dace.float64)
        state = sdfg.add_state('s')
        me, mx = state.add_map('m', {'i': '0:5', 'j': '0:7'}, schedule=ScheduleType.Sequential)
        a = state.add_read('A')
        b = state.add_write('B')
        t = state.add_tasklet('t', {'inp'}, {'out'}, 'out = inp + 3.14')
        state.add_memlet_path(a, me, t, dst_conn='inp', memlet=Memlet(data='A', subset='i, j'))
        state.add_memlet_path(t, mx, b, src_conn='out', memlet=Memlet(data='B', subset='i, j'))
        csdfg = sdfg.compile()
        A = np.random.rand(R, C)
        B = np.zeros((R, C))
        csdfg(A=A, B=B)
        np.testing.assert_allclose(B, A + 3.14)

    @_MAP_XFAIL
    def test_map_correctness_with_step(self):
        """Stepped map (step=2): process even indices, verify against numpy."""
        N = 10
        sdfg = _make_python_sdfg('test_step_corr')
        sdfg.add_array('A', [N], dace.float64)
        sdfg.add_array('B', [N], dace.float64)
        state = sdfg.add_state('s')
        me, mx = state.add_map('m', {'i': '0:10:2'}, schedule=ScheduleType.Sequential)
        a = state.add_read('A')
        b = state.add_write('B')
        t = state.add_tasklet('t', {'inp'}, {'out'}, 'out = inp * 10')
        state.add_memlet_path(a, me, t, dst_conn='inp', memlet=Memlet(data='A', subset='i'))
        state.add_memlet_path(t, mx, b, src_conn='out', memlet=Memlet(data='B', subset='i'))
        csdfg = sdfg.compile()
        A = np.arange(N, dtype=np.float64)
        B = np.zeros(N, dtype=np.float64)
        csdfg(A=A, B=B)
        for i in range(0, N, 2):
            assert B[i] == A[i] * 10

    @_MAP_XFAIL
    def test_nested_map_correctness(self):
        """Nested maps: B[i,j] = A[i,j] + i + j, verify against numpy."""
        R, C = 3, 5
        sdfg = _make_python_sdfg('test_nested_corr')
        sdfg.add_array('A', [R, C], dace.float64)
        sdfg.add_array('B', [R, C], dace.float64)
        state = sdfg.add_state('s')
        ome, omx = state.add_map('outer', {'i': '0:3'}, schedule=ScheduleType.Sequential)
        ime, imx = state.add_map('inner', {'j': '0:5'}, schedule=ScheduleType.Sequential)
        a = state.add_read('A')
        b = state.add_write('B')
        t = state.add_tasklet('t', {'inp'}, {'out'}, 'out = inp + i + j')
        state.add_memlet_path(a, ome, ime, t, dst_conn='inp', memlet=Memlet(data='A', subset='i, j'))
        state.add_memlet_path(t, imx, omx, b, src_conn='out', memlet=Memlet(data='B', subset='i, j'))
        csdfg = sdfg.compile()
        A = np.random.rand(R, C)
        B = np.zeros((R, C))
        csdfg(A=A, B=B)
        expected = A.copy()
        for i in range(R):
            for j in range(C):
                expected[i, j] = A[i, j] + i + j
        np.testing.assert_allclose(B, expected)

    @_MAP_XFAIL
    def test_map_add_two_arrays(self):
        """Map adding two arrays: C = A + B, verify against numpy."""
        N = 12
        sdfg = _make_python_sdfg('test_add_arrays')
        sdfg.add_array('A', [N], dace.float64)
        sdfg.add_array('B', [N], dace.float64)
        sdfg.add_array('C', [N], dace.float64)
        state = sdfg.add_state('s')
        me, mx = state.add_map('m', {'i': '0:12'}, schedule=ScheduleType.Sequential)
        ra = state.add_read('A')
        rb = state.add_read('B')
        wc = state.add_write('C')
        t = state.add_tasklet('add', {'x', 'y'}, {'z'}, 'z = x + y')
        state.add_memlet_path(ra, me, t, dst_conn='x', memlet=Memlet(data='A', subset='i'))
        state.add_memlet_path(rb, me, t, dst_conn='y', memlet=Memlet(data='B', subset='i'))
        state.add_memlet_path(t, mx, wc, src_conn='z', memlet=Memlet(data='C', subset='i'))
        csdfg = sdfg.compile()
        A = np.random.rand(N)
        B = np.random.rand(N)
        C = np.zeros(N)
        csdfg(A=A, B=B, C=C)
        np.testing.assert_allclose(C, A + B)

    @_MAP_XFAIL
    def test_map_symbolic_correctness(self):
        """Map with symbolic bounds: compile and run with concrete N."""
        sdfg = _make_python_sdfg('test_sym_corr')
        N_sym = dace.symbol('N')
        sdfg.add_array('A', [N_sym], dace.float64)
        sdfg.add_array('B', [N_sym], dace.float64)
        state = sdfg.add_state('s')
        me, mx = state.add_map('m', {'i': '0:N'}, schedule=ScheduleType.Sequential)
        a = state.add_read('A')
        b = state.add_write('B')
        t = state.add_tasklet('t', {'inp'}, {'out'}, 'out = inp + 1')
        state.add_memlet_path(a, me, t, dst_conn='inp', memlet=Memlet(data='A', subset='i'))
        state.add_memlet_path(t, mx, b, src_conn='out', memlet=Memlet(data='B', subset='i'))
        csdfg = sdfg.compile()
        N_val = 8
        A = np.arange(N_val, dtype=np.float64)
        B = np.zeros(N_val, dtype=np.float64)
        csdfg(A=A, B=B, N=N_val)
        np.testing.assert_allclose(B, A + 1)

    @_MAP_XFAIL
    def test_map_symbolic_negative_step_runtime(self):
        """A symbolic map step can be negative at runtime and still iterates correctly."""
        sdfg = _make_python_sdfg('test_symbolic_negative_step_runtime')
        sdfg.add_symbol('START', dace.int64)
        sdfg.add_symbol('STOP', dace.int64)
        sdfg.add_symbol('STEP', dace.int64)
        sdfg.add_array('A', [6], dace.float64)
        sdfg.add_array('B', [6], dace.float64)
        state = sdfg.add_state('s')
        map_entry, map_exit = state.add_map('m', {'i': 'START:STOP:STEP'}, schedule=ScheduleType.Sequential)
        tasklet = state.add_tasklet('copy', {'inp'}, {'out'}, 'out = inp')
        state.add_memlet_path(state.add_read('A'), map_entry, tasklet, dst_conn='inp', memlet=Memlet(data='A', subset='i'))
        state.add_memlet_path(tasklet, map_exit, state.add_write('B'), src_conn='out', memlet=Memlet(data='B', subset='i'))

        generated_code = sdfg.generate_code()[0].code
        assert 'if (STEP) > 0 else' in generated_code

        compiled_sdfg = sdfg.compile()
        a = np.arange(6, dtype=np.float64)
        b = np.zeros(6, dtype=np.float64)
        compiled_sdfg(A=a, B=b, START=5, STOP=-1, STEP=-2)

        expected = np.zeros(6, dtype=np.float64)
        expected[[5, 3, 1]] = a[[5, 3, 1]]
        np.testing.assert_allclose(b, expected)

    @_MAP_XFAIL
    def test_map_square_elements(self):
        """Map squaring elements: B[i] = A[i]^2, verify against numpy."""
        N = 10
        sdfg = _make_python_sdfg('test_square')
        sdfg.add_array('A', [N], dace.float64)
        sdfg.add_array('B', [N], dace.float64)
        state = sdfg.add_state('s')
        me, mx = state.add_map('m', {'i': '0:10'}, schedule=ScheduleType.Sequential)
        a = state.add_read('A')
        b = state.add_write('B')
        t = state.add_tasklet('sq', {'x'}, {'y'}, 'y = x * x')
        state.add_memlet_path(a, me, t, dst_conn='x', memlet=Memlet(data='A', subset='i'))
        state.add_memlet_path(t, mx, b, src_conn='y', memlet=Memlet(data='B', subset='i'))
        csdfg = sdfg.compile()
        A = np.random.rand(N)
        B = np.zeros(N)
        csdfg(A=A, B=B)
        np.testing.assert_allclose(B, A ** 2)
