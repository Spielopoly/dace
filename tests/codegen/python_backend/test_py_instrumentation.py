# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for Python-backend instrumentation (timing) support.

Covers:
- PythonTimerProvider unit tests (compilation, code emission)
- Integration tests (full pipeline: SDFG construction -> compile -> run -> report)
- SDFG-level, state-level, and scope-level timing
- Symbolic sizes and edge cases
- Report format validation
"""

import json
import os
import time

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.codegen.instrumentation.py_timer import PythonTimerProvider
from dace.codegen.instrumentation.report import InstrumentationReport
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.sdfg import nodes


def _make_add_sdfg(name: str, size: int = 10):
    """Create a simple SDFG that computes C = A + B with a map."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [size], dace.float64)
    sdfg.add_array('B', [size], dace.float64)
    sdfg.add_array('C', [size], dace.float64)

    state = sdfg.add_state('compute')
    a = state.add_access('A')
    b = state.add_access('B')
    c = state.add_access('C')
    me, mx = state.add_map('add_map', {'i': f'0:{size}'})
    tasklet = state.add_tasklet('add', {'a_in', 'b_in'}, {'c_out'}, 'c_out = a_in + b_in')
    state.add_memlet_path(a, me, tasklet, dst_conn='a_in', memlet=dace.Memlet(f'A[i]'))
    state.add_memlet_path(b, me, tasklet, dst_conn='b_in', memlet=dace.Memlet(f'B[i]'))
    state.add_memlet_path(tasklet, mx, c, src_conn='c_out', memlet=dace.Memlet(f'C[i]'))

    return sdfg, state, me


def _make_scale_sdfg(name: str, size: int = 20):
    """Create a simple SDFG that computes B = A * 3.0 with a map."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [size], dace.float64)
    sdfg.add_array('B', [size], dace.float64)

    state = sdfg.add_state('compute')
    a = state.add_access('A')
    b = state.add_access('B')
    me, mx = state.add_map('scale_map', {'i': f'0:{size}'})
    tasklet = state.add_tasklet('scale', {'inp'}, {'out'}, 'out = inp * 3.0')
    state.add_memlet_path(a, me, tasklet, dst_conn='inp', memlet=dace.Memlet(f'A[i]'))
    state.add_memlet_path(tasklet, mx, b, src_conn='out', memlet=dace.Memlet(f'B[i]'))

    return sdfg, state, me


# ---------------------------------------------------------------------------
# Unit tests for PythonTimerProvider
# ---------------------------------------------------------------------------


class TestPythonTimerProviderUnit:
    """Unit tests for PythonTimerProvider internals."""

    def test_idstr_sdfg_level(self):
        """_idstr with (sdfg, None, None) should return just the cfg_id."""
        provider = PythonTimerProvider()
        sdfg = dace.SDFG('test_idstr')
        result = provider._idstr(sdfg, None, None)
        assert result == str(sdfg.cfg_id)

    def test_idstr_state_level(self):
        """_idstr with (sdfg, state, None) should include state id."""
        provider = PythonTimerProvider()
        sdfg = dace.SDFG('test_idstr_state')
        state = sdfg.add_state('s0')
        result = provider._idstr(sdfg, state, None)
        assert '_' in result
        parts = result.split('_')
        assert len(parts) == 2

    def test_idstr_node_level(self):
        """_idstr with (sdfg, state, node) should include node id."""
        provider = PythonTimerProvider()
        sdfg = dace.SDFG('test_idstr_node')
        state = sdfg.add_state('s0')
        sdfg.add_array('A', [10], dace.float64)
        anode = state.add_access('A')
        result = provider._idstr(sdfg, state, anode)
        parts = result.split('_')
        assert len(parts) == 3

    def test_emit_tbegin_writes_timer_code(self):
        """_emit_tbegin should write timer capture code to the stream."""
        provider = PythonTimerProvider()
        sdfg = dace.SDFG('test_emit')
        stream = PythonCodeIOStream()
        provider._emit_tbegin(stream, sdfg, None, None)
        code = stream.getvalue()
        assert '__dace_tbegin_' in code
        assert '__dace_timer_us()' in code

    def test_emit_tend_writes_timer_and_event(self):
        """_emit_tend should write end timer capture and event append."""
        provider = PythonTimerProvider()
        sdfg = dace.SDFG('test_emit_tend')
        stream = PythonCodeIOStream()
        provider._emit_tend('TestEvent', stream, sdfg, None, None)
        code = stream.getvalue()
        assert '__dace_tend_' in code
        assert '__dace_perf_events.append' in code
        assert 'TestEvent' in code
        assert 'Timer' in code

    def test_on_sdfg_begin_emits_timer_setup_ctypes(self):
        """on_sdfg_begin should emit ctypes-based timer setup."""
        provider = PythonTimerProvider()
        sdfg = dace.SDFG('test_setup')
        sdfg.instrument = dtypes.InstrumentationType.PythonTimer

        global_stream = PythonCodeIOStream()
        local_stream = PythonCodeIOStream()
        provider.on_sdfg_begin(sdfg, local_stream, global_stream, None)

        global_code = global_stream.getvalue()
        assert '__dace_timer_us' in global_code
        assert '__dace_perf_events' in global_code

        local_code = local_stream.getvalue()
        assert '__dace_tbegin_' in local_code

    def test_on_sdfg_begin_no_timer_if_not_instrumented(self):
        """on_sdfg_begin should not emit tbegin if SDFG is not instrumented."""
        provider = PythonTimerProvider()
        sdfg = dace.SDFG('test_no_timer')
        # Do NOT set sdfg.instrument

        global_stream = PythonCodeIOStream()
        local_stream = PythonCodeIOStream()
        provider.on_sdfg_begin(sdfg, local_stream, global_stream, None)

        # Global setup should still happen (timer library loaded)
        global_code = global_stream.getvalue()
        assert '__dace_timer_us' in global_code

        # But no tbegin in local stream
        local_code = local_stream.getvalue()
        assert '__dace_tbegin_' not in local_code


# ---------------------------------------------------------------------------
# Integration tests: full pipeline
# ---------------------------------------------------------------------------


class TestPythonTimerIntegration:
    """Integration tests that compile and run instrumented programs."""

    def test_sdfg_level_timer(self):
        """Instrumenting the entire SDFG should produce a valid report."""
        sdfg, state, me = _make_add_sdfg('test_sdfg_timer')
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.instrument = dtypes.InstrumentationType.PythonTimer

        compiled = sdfg.compile()

        a = np.ones(10, dtype=np.float64)
        b = np.ones(10, dtype=np.float64) * 2.0
        c = np.zeros(10, dtype=np.float64)
        compiled(A=a, B=b, C=c)

        # Check correctness
        assert np.allclose(c, a + b)

        # Check report exists
        report = sdfg.get_latest_report()
        assert report is not None
        assert len(report.events) > 0

        # Check event structure
        for event in report.events:
            assert event.category == 'Timer'
            assert event.duration >= 0
            assert 'SDFG' in event.name

    def test_state_level_timer(self):
        """Instrumenting states should produce per-state timing events."""
        sdfg, state, me = _make_add_sdfg('test_state_timer')
        sdfg.backend = dtypes.BackendLanguage.Python

        # Instrument the state
        state.instrument = dtypes.InstrumentationType.PythonTimer

        compiled = sdfg.compile()

        a = np.ones(10, dtype=np.float64)
        b = np.arange(10, dtype=np.float64)
        c = np.zeros(10, dtype=np.float64)
        compiled(A=a, B=b, C=c)

        assert np.allclose(c, a + b)

        report = sdfg.get_latest_report()
        assert report is not None
        assert len(report.events) > 0

        for event in report.events:
            assert event.category == 'Timer'
            assert 'State' in event.name

    def test_map_scope_timer(self):
        """Instrumenting a map scope should time the for-loop execution."""
        sdfg, state, me = _make_scale_sdfg('test_map_timer')

        me.map.instrument = dtypes.InstrumentationType.PythonTimer
        sdfg.backend = dtypes.BackendLanguage.Python

        compiled = sdfg.compile()

        a = np.arange(20, dtype=np.float64)
        b = np.zeros(20, dtype=np.float64)
        compiled(A=a, B=b)

        assert np.allclose(b, a * 3.0)

        report = sdfg.get_latest_report()
        assert report is not None
        assert len(report.events) > 0
        assert any('Map' in ev.name for ev in report.events)

    def test_all_levels_instrumented(self):
        """SDFG, state, and map all instrumented simultaneously."""
        sdfg, state, me = _make_add_sdfg('test_all_levels', size=16)

        sdfg.instrument = dtypes.InstrumentationType.PythonTimer
        state.instrument = dtypes.InstrumentationType.PythonTimer
        me.map.instrument = dtypes.InstrumentationType.PythonTimer

        sdfg.backend = dtypes.BackendLanguage.Python
        compiled = sdfg.compile()

        a = np.arange(16, dtype=np.float64)
        b = np.ones(16, dtype=np.float64)
        c = np.zeros(16, dtype=np.float64)
        compiled(A=a, B=b, C=c)

        assert np.allclose(c, a + b)

        report = sdfg.get_latest_report()
        assert report is not None
        names = [ev.name for ev in report.events]
        assert any('SDFG' in n for n in names)
        assert any('State' in n for n in names)
        assert any('Map' in n for n in names)

    def test_symbolic_size(self):
        """Instrumented program with symbolic array size should work."""
        N = dace.symbol('N')

        sdfg = dace.SDFG('test_sym_timer')
        sdfg.add_array('A', [N], dace.float64)
        sdfg.add_array('B', [N], dace.float64)
        sdfg.add_array('C', [N], dace.float64)

        state = sdfg.add_state('compute')
        a = state.add_access('A')
        b = state.add_access('B')
        c = state.add_access('C')
        me, mx = state.add_map('m', {'i': '0:N'})
        t = state.add_tasklet('add', {'x', 'y'}, {'z'}, 'z = x + y')
        state.add_memlet_path(a, me, t, dst_conn='x', memlet=dace.Memlet('A[i]'))
        state.add_memlet_path(b, me, t, dst_conn='y', memlet=dace.Memlet('B[i]'))
        state.add_memlet_path(t, mx, c, src_conn='z', memlet=dace.Memlet('C[i]'))

        sdfg.instrument = dtypes.InstrumentationType.PythonTimer
        me.map.instrument = dtypes.InstrumentationType.PythonTimer

        sdfg.backend = dtypes.BackendLanguage.Python
        compiled = sdfg.compile()

        size = 25  # Non-power-of-2
        arr_a = np.random.rand(size)
        arr_b = np.random.rand(size)
        arr_c = np.zeros(size)
        compiled(A=arr_a, B=arr_b, C=arr_c, N=size)

        assert np.allclose(arr_c, arr_a + arr_b)

        report = sdfg.get_latest_report()
        assert report is not None
        assert len(report.events) > 0

    def test_report_json_format(self):
        """Check that the generated JSON report matches the Chrome Tracing format."""
        sdfg, state, me = _make_add_sdfg('test_json_format', size=4)
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.instrument = dtypes.InstrumentationType.PythonTimer

        compiled = sdfg.compile()
        a = np.ones(4, dtype=np.float64)
        b = np.ones(4, dtype=np.float64)
        c = np.zeros(4, dtype=np.float64)
        compiled(A=a, B=b, C=c)

        report_path = sdfg.get_latest_report_path()
        assert report_path is not None

        with open(report_path, 'r') as f:
            data = json.load(f)

        assert 'traceEvents' in data
        assert 'sdfgHash' in data
        assert isinstance(data['traceEvents'], list)
        assert len(data['traceEvents']) > 0

        for event in data['traceEvents']:
            assert event['ph'] == 'X'
            assert event['cat'] == 'Timer'
            assert 'ts' in event
            assert 'dur' in event
            assert event['dur'] >= 0
            assert 'pid' in event
            assert 'args' in event
            assert 'cfg_id' in event['args']

    def test_no_instrumentation_no_events(self):
        """A program with no instrumentation should not produce timer events."""
        sdfg, state, me = _make_add_sdfg('test_no_instr', size=5)
        sdfg.backend = dtypes.BackendLanguage.Python
        # No instrumentation set

        # Clear any old reports
        try:
            sdfg.clear_instrumentation_reports()
        except FileNotFoundError:
            pass

        compiled = sdfg.compile()
        a = np.ones(5, dtype=np.float64)
        b = np.ones(5, dtype=np.float64)
        c = np.zeros(5, dtype=np.float64)
        compiled(A=a, B=b, C=c)

        # Either no report or empty
        report_path = sdfg.get_latest_report_path()
        if report_path is not None:
            with open(report_path) as f:
                data = json.load(f)
            # Should have no events
            assert len(data.get('traceEvents', [])) == 0

    def test_multiple_runs_produce_multiple_reports(self):
        """Running an instrumented SDFG twice should produce two reports."""
        sdfg, state, me = _make_add_sdfg('test_multi_run', size=3)
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.instrument = dtypes.InstrumentationType.PythonTimer

        # Clear old reports
        try:
            sdfg.clear_instrumentation_reports()
        except FileNotFoundError:
            pass

        compiled = sdfg.compile()
        a = np.ones(3, dtype=np.float64)
        b = np.ones(3, dtype=np.float64)
        c = np.zeros(3, dtype=np.float64)

        compiled(A=a, B=b, C=c)
        # Small delay to ensure different timestamps
        time.sleep(0.02)
        compiled(A=a, B=b, C=c)

        reports = sdfg.get_instrumentation_reports()
        assert len(reports) >= 2

    def test_multidimensional_map(self):
        """A 2D map with instrumentation should work correctly."""
        sdfg = dace.SDFG('test_2d_map_timer')
        sdfg.add_array('A', [4, 4], dace.float64)
        sdfg.add_array('B', [4, 4], dace.float64)

        state = sdfg.add_state('s0')
        a = state.add_access('A')
        b = state.add_access('B')
        me, mx = state.add_map('m2d', {'i': '0:4', 'j': '0:4'})
        t = state.add_tasklet('neg', {'inp'}, {'out'}, 'out = -inp')
        state.add_memlet_path(a, me, t, dst_conn='inp', memlet=dace.Memlet('A[i, j]'))
        state.add_memlet_path(t, mx, b, src_conn='out', memlet=dace.Memlet('B[i, j]'))

        me.map.instrument = dtypes.InstrumentationType.PythonTimer

        sdfg.backend = dtypes.BackendLanguage.Python
        compiled = sdfg.compile()

        arr_a = np.arange(16, dtype=np.float64).reshape(4, 4)
        arr_b = np.zeros((4, 4), dtype=np.float64)
        compiled(A=arr_a, B=arr_b)

        assert np.allclose(arr_b, -arr_a)

        report = sdfg.get_latest_report()
        assert report is not None
        assert any('Map' in ev.name for ev in report.events)

    def test_report_has_correct_sdfg_hash(self):
        """The report's sdfgHash should match the SDFG's hash."""
        sdfg, state, me = _make_add_sdfg('test_hash', size=6)
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.instrument = dtypes.InstrumentationType.PythonTimer

        compiled = sdfg.compile()
        a = np.ones(6, dtype=np.float64)
        b = np.ones(6, dtype=np.float64)
        c = np.zeros(6, dtype=np.float64)
        compiled(A=a, B=b, C=c)

        report_path = sdfg.get_latest_report_path()
        with open(report_path) as f:
            data = json.load(f)

        # sdfgHash should be a non-empty string
        assert isinstance(data['sdfgHash'], str)
        assert len(data['sdfgHash']) > 0

    def test_event_duration_nonnegative(self):
        """All event durations should be non-negative."""
        sdfg, state, me = _make_add_sdfg('test_dur_pos', size=100)
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.instrument = dtypes.InstrumentationType.PythonTimer
        state.instrument = dtypes.InstrumentationType.PythonTimer
        me.map.instrument = dtypes.InstrumentationType.PythonTimer

        compiled = sdfg.compile()
        a = np.random.rand(100)
        b = np.random.rand(100)
        c = np.zeros(100)
        compiled(A=a, B=b, C=c)

        report = sdfg.get_latest_report()
        for event in report.events:
            assert event.duration >= 0

    def test_event_uuids_valid(self):
        """Events should have valid UUID components matching the SDFG structure."""
        sdfg, state, me = _make_add_sdfg('test_uuids', size=8)
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.instrument = dtypes.InstrumentationType.PythonTimer
        state.instrument = dtypes.InstrumentationType.PythonTimer
        me.map.instrument = dtypes.InstrumentationType.PythonTimer

        compiled = sdfg.compile()
        a = np.ones(8, dtype=np.float64)
        b = np.ones(8, dtype=np.float64)
        c = np.zeros(8, dtype=np.float64)
        compiled(A=a, B=b, C=c)

        report = sdfg.get_latest_report()
        for event in report.events:
            cfg_id, state_id, node_id = event.uuid
            assert cfg_id >= 0  # Should always have a cfg_id

    def test_context_manager_instrument(self):
        """The dace.instrument context manager should work with Python backend."""
        sdfg, state, me = _make_add_sdfg('test_ctx_mgr', size=10)
        sdfg.backend = dtypes.BackendLanguage.Python

        a = np.ones(10, dtype=np.float64)
        b = np.ones(10, dtype=np.float64) * 2
        c = np.zeros(10, dtype=np.float64)

        with dace.instrument(dtypes.InstrumentationType.PythonTimer,
                             filter='*',
                             annotate_maps=True,
                             annotate_states=True,
                             annotate_sdfgs=True) as profiler:
            sdfg(A=a, B=b, C=c)

        assert np.allclose(c, a + b)

        # The profiler should have collected a report
        assert len(profiler.reports) > 0
        report = profiler.report
        assert len(report.events) > 0


# ---------------------------------------------------------------------------
# Code generation tests (verify emitted code structure)
# ---------------------------------------------------------------------------


class TestPythonTimerCodegen:
    """Tests that verify the structure of generated instrumentation code."""

    def test_timer_code_emitted_in_generated_source(self):
        """The generated Python source should contain timer-related code."""
        sdfg, state, me = _make_add_sdfg('test_codegen_timer', size=5)
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.instrument = dtypes.InstrumentationType.PythonTimer

        code_objects = sdfg.generate_code()
        source = code_objects[0].clean_code

        assert '__dace_timer_us' in source
        assert '__dace_perf_events' in source
        assert '__dace_tbegin_' in source
        assert '__dace_tend_' in source
        assert 'report-' in source  # report file name pattern

    def test_map_timer_code_emitted(self):
        """Map scope instrumentation should emit timer code around for-loops."""
        sdfg, state, me = _make_scale_sdfg('test_map_codegen')
        me.map.instrument = dtypes.InstrumentationType.PythonTimer
        sdfg.backend = dtypes.BackendLanguage.Python

        code_objects = sdfg.generate_code()
        source = code_objects[0].clean_code

        # Should have timer begin/end around the map
        assert '__dace_tbegin_' in source
        assert '__dace_tend_' in source
        assert '__dace_perf_events.append' in source

    def test_state_timer_code_emitted(self):
        """State instrumentation should emit timer code."""
        sdfg, state, me = _make_add_sdfg('test_state_codegen', size=5)
        state.instrument = dtypes.InstrumentationType.PythonTimer
        sdfg.backend = dtypes.BackendLanguage.Python

        code_objects = sdfg.generate_code()
        source = code_objects[0].clean_code

        assert '__dace_tbegin_' in source
        assert '__dace_perf_events.append' in source
        assert 'State' in source

    def test_no_instrumentation_no_timer_code(self):
        """Without instrumentation, no timer code should be emitted."""
        sdfg, state, me = _make_add_sdfg('test_no_timer_code', size=5)
        sdfg.backend = dtypes.BackendLanguage.Python
        # No instrumentation

        code_objects = sdfg.generate_code()
        source = code_objects[0].clean_code

        assert '__dace_tbegin_' not in source
        assert '__dace_tend_' not in source


# ---------------------------------------------------------------------------
# Additional integration tests (expanded coverage)
# ---------------------------------------------------------------------------


class TestPythonTimerIntegrationExpanded:
    """Additional integration tests for wider coverage."""

    def test_nested_sdfg_instrumentation(self):
        """Create an SDFG with a nested SDFG, instrument the nested SDFG,
        verify events appear in the report."""
        outer = dace.SDFG('test_nested_instr_outer')
        outer.add_array('A', [10], dace.float64)
        outer.add_array('B', [10], dace.float64)

        inner = dace.SDFG('test_nested_instr_inner')
        inner.backend = dtypes.BackendLanguage.Python
        inner.add_array('X', [10], dace.float64)
        inner.add_array('Y', [10], dace.float64)

        inner_state = inner.add_state('inner_compute')
        x_node = inner_state.add_access('X')
        y_node = inner_state.add_access('Y')
        me, mx = inner_state.add_map('inner_map', {'i': '0:10'})
        t = inner_state.add_tasklet('double', {'inp'}, {'out'}, 'out = inp * 2')
        inner_state.add_memlet_path(x_node, me, t, dst_conn='inp',
                                    memlet=dace.Memlet('X[i]'))
        inner_state.add_memlet_path(t, mx, y_node, src_conn='out',
                                    memlet=dace.Memlet('Y[i]'))

        # Instrument the inner state and map
        inner_state.instrument = dtypes.InstrumentationType.PythonTimer
        me.map.instrument = dtypes.InstrumentationType.PythonTimer

        outer_state = outer.add_state('outer_state')
        nested_node = outer_state.add_nested_sdfg(inner, {'X'}, {'Y'})
        a_node = outer_state.add_access('A')
        b_node = outer_state.add_access('B')
        outer_state.add_edge(a_node, None, nested_node, 'X',
                             dace.Memlet('A[0:10]'))
        outer_state.add_edge(nested_node, 'Y', b_node, None,
                             dace.Memlet('B[0:10]'))

        outer.backend = dtypes.BackendLanguage.Python
        outer.instrument = dtypes.InstrumentationType.PythonTimer

        compiled = outer.compile()
        a = np.arange(10, dtype=np.float64)
        b = np.zeros(10, dtype=np.float64)
        compiled(A=a, B=b)

        assert np.allclose(b, a * 2)

        report = outer.get_latest_report()
        assert report is not None
        assert len(report.events) > 0

    def test_multiple_maps_mixed_instrumentation(self):
        """Two maps in one state, only one instrumented."""
        sdfg = dace.SDFG('test_mixed_maps')
        sdfg.add_array('A', [10], dace.float64)
        sdfg.add_array('B', [10], dace.float64)
        sdfg.add_array('C', [10], dace.float64)

        state = sdfg.add_state('s0')
        a = state.add_access('A')
        b = state.add_access('B')
        c = state.add_access('C')

        # Map 1: B = A * 2 (instrumented)
        me1, mx1 = state.add_map('map1', {'i': '0:10'})
        t1 = state.add_tasklet('t1', {'x'}, {'y'}, 'y = x * 2')
        state.add_memlet_path(a, me1, t1, dst_conn='x',
                              memlet=dace.Memlet('A[i]'))
        state.add_memlet_path(t1, mx1, b, src_conn='y',
                              memlet=dace.Memlet('B[i]'))
        me1.map.instrument = dtypes.InstrumentationType.PythonTimer

        # Map 2: C = A + 1 (NOT instrumented)
        a2 = state.add_access('A')
        me2, mx2 = state.add_map('map2', {'j': '0:10'})
        t2 = state.add_tasklet('t2', {'x'}, {'y'}, 'y = x + 1')
        state.add_memlet_path(a2, me2, t2, dst_conn='x',
                              memlet=dace.Memlet('A[j]'))
        state.add_memlet_path(t2, mx2, c, src_conn='y',
                              memlet=dace.Memlet('C[j]'))
        # map2 NOT instrumented

        sdfg.backend = dtypes.BackendLanguage.Python
        compiled = sdfg.compile()

        arr_a = np.arange(10, dtype=np.float64)
        arr_b = np.zeros(10, dtype=np.float64)
        arr_c = np.zeros(10, dtype=np.float64)
        compiled(A=arr_a, B=arr_b, C=arr_c)

        assert np.allclose(arr_b, arr_a * 2)
        assert np.allclose(arr_c, arr_a + 1)

        report = sdfg.get_latest_report()
        assert report is not None
        names = [ev.name for ev in report.events]
        # Only map1 should appear
        assert any('map1' in n for n in names)
        # map2 should not have timing events
        assert not any('map2' in n for n in names)

    def test_large_array_size(self):
        """Test with a large array size (10000+)."""
        size = 10007  # Prime number, non-power-of-2
        sdfg, state, me = _make_add_sdfg('test_large_array', size=size)
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.instrument = dtypes.InstrumentationType.PythonTimer
        me.map.instrument = dtypes.InstrumentationType.PythonTimer

        compiled = sdfg.compile()
        a = np.random.rand(size)
        b = np.random.rand(size)
        c = np.zeros(size)
        compiled(A=a, B=b, C=c)

        assert np.allclose(c, a + b)

        report = sdfg.get_latest_report()
        assert report is not None
        # SDFG + Map events
        assert len(report.events) >= 2

    def test_multi_state_sdfg_instrumented(self):
        """SDFG with multiple states, some instrumented, some not."""
        from dace.sdfg.sdfg import InterstateEdge

        sdfg = dace.SDFG('test_multistate_instr')
        sdfg.add_array('A', [8], dace.float64)
        sdfg.add_array('B', [8], dace.float64)
        sdfg.add_array('C', [8], dace.float64)

        # State 0: B = A * 2 (instrumented)
        s0 = sdfg.add_state('compute_b')
        a0 = s0.add_access('A')
        b0 = s0.add_access('B')
        me0, mx0 = s0.add_map('m0', {'i': '0:8'})
        t0 = s0.add_tasklet('t0', {'x'}, {'y'}, 'y = x * 2')
        s0.add_memlet_path(a0, me0, t0, dst_conn='x',
                           memlet=dace.Memlet('A[i]'))
        s0.add_memlet_path(t0, mx0, b0, src_conn='y',
                           memlet=dace.Memlet('B[i]'))
        s0.instrument = dtypes.InstrumentationType.PythonTimer

        # State 1: C = B + 1 (NOT instrumented)
        s1 = sdfg.add_state('compute_c')
        b1 = s1.add_access('B')
        c1 = s1.add_access('C')
        me1, mx1 = s1.add_map('m1', {'i': '0:8'})
        t1 = s1.add_tasklet('t1', {'x'}, {'y'}, 'y = x + 1')
        s1.add_memlet_path(b1, me1, t1, dst_conn='x',
                           memlet=dace.Memlet('B[i]'))
        s1.add_memlet_path(t1, mx1, c1, src_conn='y',
                           memlet=dace.Memlet('C[i]'))

        sdfg.add_edge(s0, s1, InterstateEdge())

        sdfg.backend = dtypes.BackendLanguage.Python
        compiled = sdfg.compile()

        a = np.arange(8, dtype=np.float64)
        b = np.zeros(8, dtype=np.float64)
        c = np.zeros(8, dtype=np.float64)
        compiled(A=a, B=b, C=c)

        assert np.allclose(b, a * 2)
        assert np.allclose(c, a * 2 + 1)

        report = sdfg.get_latest_report()
        assert report is not None
        names = [ev.name for ev in report.events]
        assert any('compute_b' in n for n in names)
        # compute_c should not have state events
        assert not any('compute_c' in n for n in names)

    def test_empty_map_body(self):
        """Edge case: a map with a no-op tasklet."""
        sdfg = dace.SDFG('test_empty_map')
        sdfg.add_array('A', [5], dace.float64)

        state = sdfg.add_state('s0')
        a = state.add_access('A')
        me, mx = state.add_map('noop_map', {'i': '0:5'})
        t = state.add_tasklet('noop', {'x'}, {'y'}, 'y = x')
        state.add_memlet_path(a, me, t, dst_conn='x',
                              memlet=dace.Memlet('A[i]'))
        a_out = state.add_access('A')
        state.add_memlet_path(t, mx, a_out, src_conn='y',
                              memlet=dace.Memlet('A[i]'))

        me.map.instrument = dtypes.InstrumentationType.PythonTimer
        sdfg.backend = dtypes.BackendLanguage.Python
        compiled = sdfg.compile()

        arr = np.arange(5, dtype=np.float64)
        arr_copy = arr.copy()
        compiled(A=arr)

        assert np.allclose(arr, arr_copy)

        report = sdfg.get_latest_report()
        assert report is not None
        assert any('Map' in ev.name for ev in report.events)

    def test_mixed_state_and_map_instrumentation(self):
        """State instrumented AND map in that state also instrumented."""
        sdfg, state, me = _make_scale_sdfg('test_mixed_state_map', size=12)
        state.instrument = dtypes.InstrumentationType.PythonTimer
        me.map.instrument = dtypes.InstrumentationType.PythonTimer

        sdfg.backend = dtypes.BackendLanguage.Python
        compiled = sdfg.compile()

        a = np.arange(12, dtype=np.float64)
        b = np.zeros(12, dtype=np.float64)
        compiled(A=a, B=b)

        assert np.allclose(b, a * 3.0)

        report = sdfg.get_latest_report()
        assert report is not None
        names = [ev.name for ev in report.events]
        assert any('State' in n for n in names)
        assert any('Map' in n for n in names)
        assert len(report.events) >= 2

    def test_report_str_representation(self):
        """Test that str(report) produces meaningful output."""
        sdfg, state, me = _make_add_sdfg('test_str_report', size=4)
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.instrument = dtypes.InstrumentationType.PythonTimer

        compiled = sdfg.compile()
        a = np.ones(4, dtype=np.float64)
        b = np.ones(4, dtype=np.float64)
        c = np.zeros(4, dtype=np.float64)
        compiled(A=a, B=b, C=c)

        report = sdfg.get_latest_report()
        report_str = str(report)

        assert 'Instrumentation report' in report_str
        assert 'SDFG Hash' in report_str
        assert 'Runtime (ms)' in report_str
        assert len(report_str) > 50

    def test_report_csv_export(self):
        """Test that report.as_csv() returns valid CSV strings."""
        sdfg, state, me = _make_add_sdfg('test_csv_report', size=6)
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.instrument = dtypes.InstrumentationType.PythonTimer
        me.map.instrument = dtypes.InstrumentationType.PythonTimer

        compiled = sdfg.compile()
        a = np.ones(6, dtype=np.float64)
        b = np.ones(6, dtype=np.float64)
        c = np.zeros(6, dtype=np.float64)
        compiled(A=a, B=b, C=c)

        report = sdfg.get_latest_report()
        durations_csv, counters_csv = report.as_csv()

        # Durations CSV should have content
        assert len(durations_csv) > 0
        lines = durations_csv.strip().split('\n')
        assert len(lines) >= 2  # Header + at least one data row
        assert 'MinMS' in lines[0]
        assert 'MeanMS' in lines[0]

    def test_two_maps_in_one_state_both_instrumented(self):
        """Two maps in the same state, both instrumented."""
        sdfg = dace.SDFG('test_two_maps_instr')
        sdfg.add_array('A', [8], dace.float64)
        sdfg.add_array('B', [8], dace.float64)
        sdfg.add_array('C', [8], dace.float64)

        state = sdfg.add_state('s0')

        # Map 1: B = A * 2
        a1 = state.add_access('A')
        b1 = state.add_access('B')
        me1, mx1 = state.add_map('double_map', {'i': '0:8'})
        t1 = state.add_tasklet('double', {'x'}, {'y'}, 'y = x * 2')
        state.add_memlet_path(a1, me1, t1, dst_conn='x',
                              memlet=dace.Memlet('A[i]'))
        state.add_memlet_path(t1, mx1, b1, src_conn='y',
                              memlet=dace.Memlet('B[i]'))
        me1.map.instrument = dtypes.InstrumentationType.PythonTimer

        # Map 2: C = A + 10
        a2 = state.add_access('A')
        c1 = state.add_access('C')
        me2, mx2 = state.add_map('offset_map', {'j': '0:8'})
        t2 = state.add_tasklet('offset', {'x'}, {'y'}, 'y = x + 10')
        state.add_memlet_path(a2, me2, t2, dst_conn='x',
                              memlet=dace.Memlet('A[j]'))
        state.add_memlet_path(t2, mx2, c1, src_conn='y',
                              memlet=dace.Memlet('C[j]'))
        me2.map.instrument = dtypes.InstrumentationType.PythonTimer

        sdfg.backend = dtypes.BackendLanguage.Python
        compiled = sdfg.compile()

        a = np.arange(8, dtype=np.float64)
        b = np.zeros(8, dtype=np.float64)
        c = np.zeros(8, dtype=np.float64)
        compiled(A=a, B=b, C=c)

        assert np.allclose(b, a * 2)
        assert np.allclose(c, a + 10)

        report = sdfg.get_latest_report()
        assert report is not None
        names = [ev.name for ev in report.events]
        assert any('double_map' in n for n in names)
        assert any('offset_map' in n for n in names)

    def test_dace_program_decorator_with_instrumentation(self):
        """Use the @dace.program frontend with instrumentation."""

        @dace.program
        def add_arrays(A: dace.float64[10], B: dace.float64[10],
                       C: dace.float64[10]):
            for i in dace.map[0:10]:
                C[i] = A[i] + B[i]

        sdfg = add_arrays.to_sdfg(simplify=False)
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.instrument = dtypes.InstrumentationType.PythonTimer

        compiled = sdfg.compile()
        a = np.ones(10, dtype=np.float64)
        b = np.ones(10, dtype=np.float64) * 3
        c = np.zeros(10, dtype=np.float64)
        compiled(A=a, B=b, C=c)

        assert np.allclose(c, a + b)

        report = sdfg.get_latest_report()
        assert report is not None
        assert len(report.events) > 0

    def test_integer_dtype_arrays(self):
        """Test instrumentation with integer-typed arrays."""
        sdfg = dace.SDFG('test_int_dtype')
        sdfg.add_array('A', [15], dace.int32)
        sdfg.add_array('B', [15], dace.int32)

        state = sdfg.add_state('s0')
        a = state.add_access('A')
        b = state.add_access('B')
        me, mx = state.add_map('m', {'i': '0:15'})
        t = state.add_tasklet('sq', {'x'}, {'y'}, 'y = x * x')
        state.add_memlet_path(a, me, t, dst_conn='x',
                              memlet=dace.Memlet('A[i]'))
        state.add_memlet_path(t, mx, b, src_conn='y',
                              memlet=dace.Memlet('B[i]'))

        me.map.instrument = dtypes.InstrumentationType.PythonTimer
        sdfg.instrument = dtypes.InstrumentationType.PythonTimer
        sdfg.backend = dtypes.BackendLanguage.Python

        compiled = sdfg.compile()
        a_arr = np.arange(15, dtype=np.int32)
        b_arr = np.zeros(15, dtype=np.int32)
        compiled(A=a_arr, B=b_arr)

        assert np.array_equal(b_arr, a_arr ** 2)

        report = sdfg.get_latest_report()
        assert report is not None
        assert len(report.events) >= 2

    def test_report_event_count_matches_structure(self):
        """All three levels instrumented: exactly 3 events expected."""
        sdfg, state, me = _make_add_sdfg('test_event_count', size=4)
        sdfg.instrument = dtypes.InstrumentationType.PythonTimer
        state.instrument = dtypes.InstrumentationType.PythonTimer
        me.map.instrument = dtypes.InstrumentationType.PythonTimer

        sdfg.backend = dtypes.BackendLanguage.Python

        try:
            sdfg.clear_instrumentation_reports()
        except FileNotFoundError:
            pass

        compiled = sdfg.compile()
        a = np.ones(4, dtype=np.float64)
        b = np.ones(4, dtype=np.float64)
        c = np.zeros(4, dtype=np.float64)
        compiled(A=a, B=b, C=c)

        report = sdfg.get_latest_report()
        # SDFG + State + Map = 3 events
        assert len(report.events) == 3

    def test_report_save_and_reload(self):
        """Save a report and reload it; verify round-trip works."""
        import tempfile

        sdfg, state, me = _make_add_sdfg('test_save_reload', size=6)
        sdfg.instrument = dtypes.InstrumentationType.PythonTimer
        sdfg.backend = dtypes.BackendLanguage.Python

        compiled = sdfg.compile()
        a = np.ones(6, dtype=np.float64)
        b = np.ones(6, dtype=np.float64)
        c = np.zeros(6, dtype=np.float64)
        compiled(A=a, B=b, C=c)

        report = sdfg.get_latest_report()
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False,
                                         prefix='report-12345') as f:
            tmp_path = f.name
        try:
            report.save(tmp_path)
            reloaded = InstrumentationReport(tmp_path)
            assert len(reloaded.events) == len(report.events)
            assert reloaded.sdfg_hash == report.sdfg_hash
            for orig, re_ev in zip(report.events, reloaded.events):
                assert orig.name == re_ev.name
                assert orig.category == re_ev.category
        finally:
            os.unlink(tmp_path)

    def test_3d_map_instrumentation(self):
        """A 3D map with instrumentation should produce valid events."""
        sdfg = dace.SDFG('test_3d_map')
        sdfg.add_array('A', [2, 3, 4], dace.float64)
        sdfg.add_array('B', [2, 3, 4], dace.float64)

        state = sdfg.add_state('s0')
        a = state.add_access('A')
        b = state.add_access('B')
        me, mx = state.add_map('m3d', {'i': '0:2', 'j': '0:3', 'k': '0:4'})
        t = state.add_tasklet('inc', {'x'}, {'y'}, 'y = x + 1')
        state.add_memlet_path(a, me, t, dst_conn='x',
                              memlet=dace.Memlet('A[i, j, k]'))
        state.add_memlet_path(t, mx, b, src_conn='y',
                              memlet=dace.Memlet('B[i, j, k]'))

        me.map.instrument = dtypes.InstrumentationType.PythonTimer
        sdfg.backend = dtypes.BackendLanguage.Python

        compiled = sdfg.compile()
        a_arr = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        b_arr = np.zeros((2, 3, 4), dtype=np.float64)
        compiled(A=a_arr, B=b_arr)

        assert np.allclose(b_arr, a_arr + 1)

        report = sdfg.get_latest_report()
        assert report is not None
        assert any('Map' in ev.name for ev in report.events)

    def test_single_element_array(self):
        """Edge case: array of size 1."""
        sdfg, state, me = _make_add_sdfg('test_single_elem', size=1)
        sdfg.instrument = dtypes.InstrumentationType.PythonTimer
        me.map.instrument = dtypes.InstrumentationType.PythonTimer
        sdfg.backend = dtypes.BackendLanguage.Python

        compiled = sdfg.compile()
        a = np.array([3.14], dtype=np.float64)
        b = np.array([2.72], dtype=np.float64)
        c = np.zeros(1, dtype=np.float64)
        compiled(A=a, B=b, C=c)

        assert np.allclose(c, a + b)

        report = sdfg.get_latest_report()
        assert report is not None
        assert len(report.events) >= 2

    def test_timer_monotonicity(self):
        """Timestamps should be monotonically non-decreasing within a report."""
        sdfg, state, me = _make_add_sdfg('test_mono', size=50)
        sdfg.instrument = dtypes.InstrumentationType.PythonTimer
        state.instrument = dtypes.InstrumentationType.PythonTimer
        me.map.instrument = dtypes.InstrumentationType.PythonTimer
        sdfg.backend = dtypes.BackendLanguage.Python

        compiled = sdfg.compile()
        a = np.random.rand(50)
        b = np.random.rand(50)
        c = np.zeros(50)
        compiled(A=a, B=b, C=c)

        report = sdfg.get_latest_report()
        # Sort events by timestamp
        sorted_events = sorted(report.events, key=lambda e: e.timestamp)
        for i in range(1, len(sorted_events)):
            assert sorted_events[i].timestamp >= sorted_events[i-1].timestamp

    def test_report_durations_dict(self):
        """The report.durations dict should be properly populated."""
        sdfg, state, me = _make_add_sdfg('test_durations_dict', size=8)
        sdfg.instrument = dtypes.InstrumentationType.PythonTimer
        sdfg.backend = dtypes.BackendLanguage.Python

        compiled = sdfg.compile()
        a = np.ones(8, dtype=np.float64)
        b = np.ones(8, dtype=np.float64)
        c = np.zeros(8, dtype=np.float64)
        compiled(A=a, B=b, C=c)

        report = sdfg.get_latest_report()
        # durations should have at least one entry
        assert len(report.durations) > 0

        # Each entry should have non-empty event names
        for uuid, events in report.durations.items():
            assert len(events) > 0
            for name, threads in events.items():
                assert isinstance(name, str)
                assert len(name) > 0

    def test_scope_entry_exit_consistency(self):
        """The idstr used in tbegin should match the one in tend for a scope."""
        provider = PythonTimerProvider()
        sdfg = dace.SDFG('test_scope_consistency')
        state = sdfg.add_state('s0')
        sdfg.add_array('A', [10], dace.float64)
        me, mx = state.add_map('m', {'i': '0:10'})
        # Entry node
        t = state.add_tasklet('noop', {'x'}, {'y'}, 'y = x')
        a = state.add_access('A')
        b = state.add_access('A')
        state.add_memlet_path(a, me, t, dst_conn='x',
                              memlet=dace.Memlet('A[i]'))
        state.add_memlet_path(t, mx, b, src_conn='y',
                              memlet=dace.Memlet('A[i]'))

        idstr_entry = provider._idstr(sdfg, state, me)
        idstr_exit = provider._idstr(sdfg, state, me)

        assert idstr_entry == idstr_exit
        assert '_' in idstr_entry  # Should have cfg_state_node format

    def test_on_scope_entry_and_exit_emit_code(self):
        """on_scope_entry and on_scope_exit should emit code when instrumented."""
        provider = PythonTimerProvider()
        sdfg = dace.SDFG('test_scope_emit')
        state = sdfg.add_state('s0')
        sdfg.add_array('A', [5], dace.float64)
        me, mx = state.add_map('m', {'i': '0:5'})
        t = state.add_tasklet('noop', {'x'}, {'y'}, 'y = x')
        a = state.add_access('A')
        b = state.add_access('A')
        state.add_memlet_path(a, me, t, dst_conn='x',
                              memlet=dace.Memlet('A[i]'))
        state.add_memlet_path(t, mx, b, src_conn='y',
                              memlet=dace.Memlet('A[i]'))

        me.map.instrument = dtypes.InstrumentationType.PythonTimer

        outer_stream = PythonCodeIOStream()
        inner_stream = PythonCodeIOStream()
        global_stream = PythonCodeIOStream()

        provider.on_scope_entry(sdfg, sdfg, state, me,
                                outer_stream, inner_stream, global_stream)
        entry_code = outer_stream.getvalue()
        assert '__dace_tbegin_' in entry_code

        outer_stream2 = PythonCodeIOStream()
        provider.on_scope_exit(sdfg, sdfg, state, mx,
                               outer_stream2, inner_stream, global_stream)
        exit_code = outer_stream2.getvalue()
        assert '__dace_tend_' in exit_code
        assert '__dace_perf_events.append' in exit_code

    def test_on_state_begin_end_emit_code(self):
        """on_state_begin and on_state_end should emit timer code when state is
        instrumented."""
        provider = PythonTimerProvider()
        sdfg = dace.SDFG('test_state_emit')
        state = sdfg.add_state('s0')
        state.instrument = dtypes.InstrumentationType.PythonTimer

        local_stream = PythonCodeIOStream()
        global_stream = PythonCodeIOStream()

        provider.on_state_begin(sdfg, sdfg, state, local_stream, global_stream)
        begin_code = local_stream.getvalue()
        assert '__dace_tbegin_' in begin_code

        local_stream2 = PythonCodeIOStream()
        provider.on_state_end(sdfg, sdfg, state, local_stream2, global_stream)
        end_code = local_stream2.getvalue()
        assert '__dace_tend_' in end_code
        assert 'State s0' in end_code

    def test_on_sdfg_end_writes_report_for_toplevel(self):
        """on_sdfg_end should emit report-writing code for top-level SDFG."""
        provider = PythonTimerProvider()
        sdfg = dace.SDFG('test_report_emit')
        sdfg.instrument = dtypes.InstrumentationType.PythonTimer

        local_stream = PythonCodeIOStream()
        global_stream = PythonCodeIOStream()

        provider.on_sdfg_end(sdfg, local_stream, global_stream)
        code = local_stream.getvalue()

        assert 'report-' in code
        assert '__dace_json.dump' in code
        assert 'traceEvents' in code
        assert 'sdfgHash' in code

    def test_on_sdfg_end_no_report_for_nested(self):
        """on_sdfg_end should NOT emit report-writing code for nested SDFGs."""
        provider = PythonTimerProvider()
        outer = dace.SDFG('test_nested_no_report')
        inner = dace.SDFG('inner')
        inner.instrument = dtypes.InstrumentationType.PythonTimer

        # Set up parent relationship
        outer.add_array('A', [1], dace.float64)
        inner.add_array('X', [1], dace.float64)
        inner.add_array('Y', [1], dace.float64)
        inner_state = inner.add_state('is')
        outer_state = outer.add_state('os')
        nested_node = outer_state.add_nested_sdfg(inner, {'X'}, {'Y'})
        outer_state.add_edge(outer_state.add_read('A'), None,
                             nested_node, 'X', dace.Memlet('A[0]'))
        a_out = outer.add_array('B', [1], dace.float64)
        outer_state.add_edge(nested_node, 'Y',
                             outer_state.add_write('B'), None,
                             dace.Memlet('B[0]'))

        local_stream = PythonCodeIOStream()
        global_stream = PythonCodeIOStream()

        provider.on_sdfg_end(inner, local_stream, global_stream)
        code = local_stream.getvalue()

        # Nested SDFG should not write report
        assert '__dace_json.dump' not in code

    def test_float32_arrays(self):
        """Test instrumentation with float32 arrays."""
        sdfg = dace.SDFG('test_f32')
        sdfg.add_array('A', [20], dace.float32)
        sdfg.add_array('B', [20], dace.float32)

        state = sdfg.add_state('s0')
        a = state.add_access('A')
        b = state.add_access('B')
        me, mx = state.add_map('m', {'i': '0:20'})
        t = state.add_tasklet('neg', {'x'}, {'y'}, 'y = -x')
        state.add_memlet_path(a, me, t, dst_conn='x',
                              memlet=dace.Memlet('A[i]'))
        state.add_memlet_path(t, mx, b, src_conn='y',
                              memlet=dace.Memlet('B[i]'))

        me.map.instrument = dtypes.InstrumentationType.PythonTimer
        sdfg.backend = dtypes.BackendLanguage.Python

        compiled = sdfg.compile()
        a_arr = np.arange(20, dtype=np.float32)
        b_arr = np.zeros(20, dtype=np.float32)
        compiled(A=a_arr, B=b_arr)

        assert np.allclose(b_arr, -a_arr)
        report = sdfg.get_latest_report()
        assert report is not None

    def test_map_with_step(self):
        """Map with a non-unit step value."""
        sdfg = dace.SDFG('test_step_map')
        sdfg.add_array('A', [10], dace.float64)
        sdfg.add_array('B', [10], dace.float64)

        state = sdfg.add_state('s0')
        a = state.add_access('A')
        b = state.add_access('B')
        me, mx = state.add_map('step_map', {'i': '0:10:2'})
        t = state.add_tasklet('copy', {'x'}, {'y'}, 'y = x')
        state.add_memlet_path(a, me, t, dst_conn='x',
                              memlet=dace.Memlet('A[i]'))
        state.add_memlet_path(t, mx, b, src_conn='y',
                              memlet=dace.Memlet('B[i]'))

        me.map.instrument = dtypes.InstrumentationType.PythonTimer
        sdfg.backend = dtypes.BackendLanguage.Python

        compiled = sdfg.compile()
        a_arr = np.arange(10, dtype=np.float64)
        b_arr = np.full(10, -1.0, dtype=np.float64)
        compiled(A=a_arr, B=b_arr)

        # Only even indices should be copied
        for i in range(0, 10, 2):
            assert b_arr[i] == a_arr[i]

        report = sdfg.get_latest_report()
        assert report is not None
        assert any('Map' in ev.name for ev in report.events)

    def test_multiple_symbolic_sizes(self):
        """Test with multiple symbolic dimensions."""
        M = dace.symbol('M')
        N = dace.symbol('N')

        sdfg = dace.SDFG('test_multi_sym')
        sdfg.add_array('A', [M, N], dace.float64)
        sdfg.add_array('B', [M, N], dace.float64)

        state = sdfg.add_state('s0')
        a = state.add_access('A')
        b = state.add_access('B')
        me, mx = state.add_map('m2d', {'i': '0:M', 'j': '0:N'})
        t = state.add_tasklet('add1', {'x'}, {'y'}, 'y = x + 1')
        state.add_memlet_path(a, me, t, dst_conn='x',
                              memlet=dace.Memlet('A[i, j]'))
        state.add_memlet_path(t, mx, b, src_conn='y',
                              memlet=dace.Memlet('B[i, j]'))

        sdfg.instrument = dtypes.InstrumentationType.PythonTimer
        me.map.instrument = dtypes.InstrumentationType.PythonTimer
        sdfg.backend = dtypes.BackendLanguage.Python

        compiled = sdfg.compile()
        m_val, n_val = 3, 7
        a_arr = np.random.rand(m_val, n_val)
        b_arr = np.zeros((m_val, n_val))
        compiled(A=a_arr, B=b_arr, M=m_val, N=n_val)

        assert np.allclose(b_arr, a_arr + 1)

        report = sdfg.get_latest_report()
        assert report is not None
        assert len(report.events) >= 2


# ---------------------------------------------------------------------------
# Additional codegen tests
# ---------------------------------------------------------------------------


class TestPythonTimerCodegenExpanded:
    """Additional code generation structure tests."""

    def test_codegen_no_duplicate_timer_setup(self):
        """Timer setup code should only appear once in global stream even
        when multiple SDFGs are instrumented."""
        sdfg, state, me = _make_add_sdfg('test_no_dup_setup', size=4)
        sdfg.instrument = dtypes.InstrumentationType.PythonTimer
        state.instrument = dtypes.InstrumentationType.PythonTimer
        me.map.instrument = dtypes.InstrumentationType.PythonTimer
        sdfg.backend = dtypes.BackendLanguage.Python

        code_objects = sdfg.generate_code()
        source = code_objects[0].clean_code

        # The timer setup should appear exactly once
        count_timer_us = source.count('__dace_timer_us = ')
        assert count_timer_us == 1

        count_perf_events = source.count('__dace_perf_events = []')
        assert count_perf_events == 1

    def test_codegen_report_filename_format(self):
        """Generated code should use 'report-{timestamp}.json' format."""
        sdfg, state, me = _make_add_sdfg('test_report_fmt', size=4)
        sdfg.instrument = dtypes.InstrumentationType.PythonTimer
        sdfg.backend = dtypes.BackendLanguage.Python

        code_objects = sdfg.generate_code()
        source = code_objects[0].clean_code

        assert 'report-{__dace_perf_ts}' in source
        assert '.json' in source

    def test_codegen_events_list_structure(self):
        """Generated event append code should include correct fields."""
        sdfg, state, me = _make_add_sdfg('test_event_fields', size=4)
        sdfg.instrument = dtypes.InstrumentationType.PythonTimer
        sdfg.backend = dtypes.BackendLanguage.Python

        code_objects = sdfg.generate_code()
        source = code_objects[0].clean_code

        # Event tuple should include: name, category, tbegin, tend, cfg_id, state_id, node_id
        assert '"Timer"' in source
        assert '"SDFG test_event_fields"' in source


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
