# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Python-backend instrumentation provider for wall-clock timing.

Emits Python code that uses a compiled C++ shared library (via ctypes) for
high-resolution timestamps.  Falls back to :func:`time.perf_counter_ns` when
compilation is not possible.

This provider is **not** auto-registered because it would conflict with the
C++ :class:`TimerProvider`.  It is swapped in explicitly by
:func:`dace.codegen.codegen.generate_code` when the Python backend is active.
"""

import os
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Optional

from dace import dtypes
from dace.config import Config
from dace.codegen.instrumentation.provider import InstrumentationProvider
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.sdfg import SDFG, nodes
from dace.sdfg.state import ControlFlowRegion, SDFGState


class PythonTimerProvider(InstrumentationProvider):
    """Timing instrumentation provider for the DaCe Python backend.

    Generates Python code that captures wall-clock timestamps before and after
    instrumented regions (SDFGs, states, map scopes) and writes a Chrome
    Tracing JSON report at the end of the top-level SDFG execution.

    .. note::
        Node-level instrumentation (individual tasklets/library nodes) is not
        supported. Setting ``node.instrument = Timer`` on a tasklet will have
        no effect in the Python backend. Only SDFG-level, state-level, and
        scope-level (map) instrumentation are supported.
    """

    # ------------------------------------------------------------------
    # Shared-library compilation
    # ------------------------------------------------------------------

    _LIB_DIR = Path.home() / '.dace' / 'lib'
    _CPP_SRC = Path(__file__).parent / '_dace_timer.cpp'

    @staticmethod
    def _ensure_timer_compiled() -> str:
        """Compile ``_dace_timer.cpp`` into a shared library, cached under
        ``~/.dace/lib/``.

        :returns: Absolute path to the compiled shared library.
        """
        lib_dir = PythonTimerProvider._LIB_DIR
        lib_dir.mkdir(parents=True, exist_ok=True)
        lib_path = lib_dir / '_dace_timer.so'

        # Re-compile only if the library does not exist or the source is newer
        if lib_path.exists():
            src_mtime = PythonTimerProvider._CPP_SRC.stat().st_mtime
            lib_mtime = lib_path.stat().st_mtime
            if lib_mtime >= src_mtime:
                return str(lib_path)

        compiler = Config.get('compiler', 'cpu', 'executable')
        try:
            fd, tmp_path = tempfile.mkstemp(suffix='.so', dir=str(lib_dir))
            os.close(fd)
            subprocess.run(
                [compiler, '-shared', '-fPIC', '-O2',
                 str(PythonTimerProvider._CPP_SRC), '-o', tmp_path],
                check=True,
                capture_output=True,
            )
            os.replace(tmp_path, str(lib_path))  # atomic on POSIX
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
            try:
                os.unlink(tmp_path)
            except (OSError, UnboundLocalError):
                pass
            warnings.warn(
                f'Failed to compile _dace_timer.cpp ({exc}). '
                'Falling back to time.perf_counter_ns for instrumentation.',
                stacklevel=2,
            )
            return ''

        return str(lib_path)

    # ------------------------------------------------------------------
    # Helper: emit timer-begin code
    # ------------------------------------------------------------------

    def _emit_tbegin(self, stream: PythonCodeIOStream, cfg: ControlFlowRegion,
                     state: Optional[SDFGState], node: Optional[nodes.Node]) -> None:
        """Emit a timer-begin statement that captures the current timestamp.

        :param stream: The output stream to write the generated Python code to.
        :param cfg: The control flow region containing the instrumented element.
        :param state: The SDFG state, or ``None`` for SDFG-level instrumentation.
        :param node: The scope entry node, or ``None`` for state/SDFG-level.
        """
        idstr = self._idstr(cfg, state, node)
        stream.write(f'__dace_tbegin_{idstr} = __dace_timer_us()')

    # ------------------------------------------------------------------
    # Helper: emit timer-end + event append code
    # ------------------------------------------------------------------

    def _emit_tend(self, timer_name: str, stream: PythonCodeIOStream,
                   cfg: ControlFlowRegion, state: Optional[SDFGState],
                   node: Optional[nodes.Node]) -> None:
        """Emit timer-end code and append a performance event to the trace list.

        :param timer_name: Human-readable name for the traced event.
        :param stream: The output stream to write the generated Python code to.
        :param cfg: The control flow region containing the instrumented element.
        :param state: The SDFG state, or ``None`` for SDFG-level instrumentation.
        :param node: The scope entry node, or ``None`` for state/SDFG-level.
        """
        idstr = self._idstr(cfg, state, node)

        cfg_id = cfg.cfg_id
        state_id = -1
        node_id = -1
        if state is not None:
            state_id = state.block_id
            if node is not None:
                node_id = state.node_id(node)

        stream.write(f'__dace_tend_{idstr} = __dace_timer_us()')
        stream.write(
            f'__dace_perf_events.append(("{timer_name}", "Timer", '
            f'__dace_tbegin_{idstr}, __dace_tend_{idstr}, '
            f'{cfg_id}, {state_id}, {node_id}))'
        )

    # ------------------------------------------------------------------
    # InstrumentationProvider hooks
    # ------------------------------------------------------------------

    def on_sdfg_begin(self, sdfg: SDFG, local_stream: PythonCodeIOStream,
                      global_stream: PythonCodeIOStream, codegen) -> None:
        # Emit timer setup into the global stream (once per SDFG file)
        lib_path = self._ensure_timer_compiled()
        if lib_path:
            global_stream.write('import ctypes as __dace_ctypes')
            global_stream.write(
                f'__dace_timer_lib = __dace_ctypes.CDLL({lib_path!r})'
            )
            global_stream.write(
                '__dace_timer_lib.timer_us.restype = __dace_ctypes.c_ulonglong'
            )
            global_stream.write('__dace_timer_us = __dace_timer_lib.timer_us')
        else:
            # Fallback: Python-based timer (microseconds)
            global_stream.write('import time as __dace_time')
            global_stream.write(
                '__dace_timer_us = lambda: __dace_time.perf_counter_ns() // 1000'
            )

        global_stream.write('__dace_perf_events = []')

        if sdfg.instrument == dtypes.InstrumentationType.Timer:
            self._emit_tbegin(local_stream, sdfg, None, None)

    def on_sdfg_end(self, sdfg: SDFG, local_stream: PythonCodeIOStream,
                    global_stream: PythonCodeIOStream) -> None:
        if sdfg.instrument == dtypes.InstrumentationType.Timer:
            self._emit_tend(f'SDFG {sdfg.name}', local_stream, sdfg, None, None)

        # For the top-level SDFG, emit the report-saving code
        if sdfg.parent is None:
            build_folder = os.path.abspath(sdfg.build_folder)
            # Hash is computed at code-generation time (frozen), matching C++ backend behavior
            sdfg_hash = sdfg.hash_sdfg()

            local_stream.write('import os as __dace_os')
            local_stream.write('import json as __dace_json')
            local_stream.write('import time as __dace_time_mod')
            local_stream.write(
                f'__dace_perf_dir = __dace_os.path.join({build_folder!r}, "perf")'
            )
            local_stream.write(
                '__dace_os.makedirs(__dace_perf_dir, exist_ok=True)'
            )
            local_stream.write(
                '__dace_perf_ts = int(__dace_time_mod.time() * 1000)'
            )
            local_stream.write(
                '__dace_perf_path = __dace_os.path.join('
                '__dace_perf_dir, f"report-{__dace_perf_ts}.json")'
            )
            local_stream.write(
                '__dace_perf_report = {'
                '"traceEvents": ['
                '{"name": ev[0], "cat": ev[1], "ph": "X", '
                '"ts": ev[2], "dur": ev[3] - ev[2], '
                '"pid": __dace_os.getpid(), "tid": -1, '
                '"args": {"cfg_id": ev[4], "state_id": ev[5], "id": ev[6]}} '
                'for ev in __dace_perf_events], '
                f'"sdfgHash": {sdfg_hash!r}'
                '}'
            )
            local_stream.write(
                'with open(__dace_perf_path, "w") as __dace_perf_fp:'
            )
            local_stream.indent()
            local_stream.write(
                '__dace_json.dump(__dace_perf_report, __dace_perf_fp)'
            )
            local_stream.dedent()

    # ------------------------------------------------------------------
    # State hooks
    # ------------------------------------------------------------------

    def on_state_begin(self, sdfg: SDFG, cfg: ControlFlowRegion,
                       state: SDFGState, local_stream: PythonCodeIOStream,
                       global_stream: PythonCodeIOStream) -> None:
        if state.instrument == dtypes.InstrumentationType.Timer:
            self._emit_tbegin(local_stream, cfg, state, None)

    def on_state_end(self, sdfg: SDFG, cfg: ControlFlowRegion,
                     state: SDFGState, local_stream: PythonCodeIOStream,
                     global_stream: PythonCodeIOStream) -> None:
        if state.instrument == dtypes.InstrumentationType.Timer:
            self._emit_tend(f'State {state.label}', local_stream, cfg, state, None)

    # ------------------------------------------------------------------
    # Scope hooks
    # ------------------------------------------------------------------

    def _get_sobj(self, node: nodes.Node) -> object:
        """Return the scope object (Map or Consume) behind an entry/exit node."""
        if hasattr(node, 'consume'):
            return node.consume
        return node.map

    def on_scope_entry(self, sdfg: SDFG, cfg: ControlFlowRegion,
                       state: SDFGState, node: nodes.EntryNode,
                       outer_stream: PythonCodeIOStream,
                       inner_stream: PythonCodeIOStream,
                       global_stream: PythonCodeIOStream) -> None:
        s = self._get_sobj(node)
        if s.instrument == dtypes.InstrumentationType.Timer:
            self._emit_tbegin(outer_stream, cfg, state, node)

    def on_scope_exit(self, sdfg: SDFG, cfg: ControlFlowRegion,
                      state: SDFGState, node: nodes.ExitNode,
                      outer_stream: PythonCodeIOStream,
                      inner_stream: PythonCodeIOStream,
                      global_stream: PythonCodeIOStream) -> None:
        entry_node = state.entry_node(node)
        s = self._get_sobj(node)
        if s.instrument == dtypes.InstrumentationType.Timer:
            self._emit_tend(
                f'{type(s).__name__} {s.label}',
                outer_stream, cfg, state, entry_node,
            )
