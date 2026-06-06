# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Python-backend instrumentation provider for wall-clock timing.

Emits Python code that uses :func:`time.perf_counter_ns` for high-resolution
timestamps.

This provider is registered against its own
:attr:`~dace.dtypes.InstrumentationType.PythonTimer` type (rather than the
C++ :class:`TimerProvider`'s :attr:`~dace.dtypes.InstrumentationType.Timer`) so
the two never collide. Set ``element.instrument = InstrumentationType.PythonTimer``
on a Python-backend SDFG/state/map/node to time it.
"""

import os
from typing import Optional, Union

from dace import dtypes, registry
from dace.config import Config
from dace.codegen.instrumentation.provider import InstrumentationProvider
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.sdfg import SDFG, nodes
from dace.sdfg.nodes import CodeNode
from dace.sdfg.state import ControlFlowRegion, SDFGState


@registry.autoregister_params(type=dtypes.InstrumentationType.PythonTimer)
class PythonTimerProvider(InstrumentationProvider):
    """Timing instrumentation provider for the DaCe Python backend.

    Generates Python code that captures wall-clock timestamps before and after
    instrumented regions (SDFGs, states, map scopes, and code nodes) and writes
    a Chrome Tracing JSON report at the end of the top-level SDFG execution.

    Selected via the dedicated
    :attr:`~dace.dtypes.InstrumentationType.PythonTimer` instrumentation type
    (the Python-backend counterpart of the C++ ``Timer``).

    .. note::
        Node-level instrumentation (individual tasklets/library nodes) is
        supported for the plain Python target. Tasklets inside cuTile kernels
        are rejected by the cuTile target.
    """

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
        # Retain the frame code generator so on_sdfg_end can emit the
        # finalization report into its exit-code stream (false report mode).
        self._codegen = codegen

        # Emit the timer and event-list setup once, only for the top-level SDFG.
        if sdfg.parent is None:
            # Import the report dependencies at module scope so the report-writer
            # (which may run inside ``__dace_exit_<name>()`` at finalization) only
            # references already-bound globals. Re-importing inside the exit
            # function would route through the frame's custom ``__import__`` hook
            # and fail at interpreter shutdown (``sys.meta_path is None``).
            global_stream.write('import time as __dace_time')
            global_stream.write('import os as __dace_os')
            global_stream.write('import json as __dace_json')
            global_stream.write(
                '__dace_timer_us = lambda: __dace_time.perf_counter_ns() // 1000'
            )
            global_stream.write('__dace_perf_events = []')

            # With report-per-invocation (the default), reset the event list at
            # the start of each call so repeated invocations do not accumulate
            # (mirrors the C++ ``report.reset()``). ``.clear()`` mutates the
            # module-level list in place, so no ``global`` declaration is needed.
            # When disabled, events accumulate across all invocations and a
            # single report is saved at interpreter exit (see ``on_sdfg_end``),
            # mirroring the C++ behavior of saving once in ``__dace_exit``.
            if Config.get_bool('instrumentation', 'report_each_invocation'):
                local_stream.write('__dace_perf_events.clear()')

        if sdfg.instrument == dtypes.InstrumentationType.PythonTimer:
            self._emit_tbegin(local_stream, sdfg, None, None)

    def on_sdfg_end(self, sdfg: SDFG, local_stream: PythonCodeIOStream,
                    global_stream: PythonCodeIOStream) -> None:
        if sdfg.instrument == dtypes.InstrumentationType.PythonTimer:
            self._emit_tend(f'SDFG {sdfg.name}', local_stream, sdfg, None, None)

        # For the top-level SDFG, emit the report-saving code.
        if sdfg.parent is None:
            if Config.get_bool('instrumentation', 'report_each_invocation'):
                # Save a fresh, timestamped report at the end of every
                # invocation (the default; matches the C++ per-invocation save).
                self._emit_report_writer(local_stream, sdfg)
            else:
                # Accumulate events across all invocations and save a single
                # report at finalization. The report-writer is emitted into the
                # frame's exit-code stream, which becomes the body of the
                # generated ``__dace_exit_<name>()`` function -- run by
                # ``PythonCompiledSDFG.finalize()`` (and its ``__del__``). This
                # mirrors the C++ backend, which saves once in ``__dace_exit``.
                # ``__dace``-prefixed names stay local to the exit function, and
                # it reads the module-global ``__dace_perf_events`` / ``__dace_time``.
                #
                # TODO: Once the Python frame code dispatches the
                # ``on_sdfg_exit_*`` instrumentation hooks (see
                # ``framecode._build_lifecycle_functions``), move this emission
                # into ``on_sdfg_exit_end`` instead of reaching into
                # ``self._codegen._exitcode`` directly.
                self._emit_report_writer(self._codegen._exitcode, sdfg)

    def _emit_report_writer(self, stream: PythonCodeIOStream, sdfg: SDFG) -> None:
        """Emit self-contained Python code that writes the accumulated
        ``__dace_perf_events`` to a timestamped Chrome-Tracing JSON report.

        The emitted block is position-independent: it references only the
        module-global ``__dace_os`` / ``__dace_json`` / ``__dace_time`` (imported
        once in :meth:`on_sdfg_begin`) and ``__dace_perf_events``, so it works
        both inside the program function (per invocation) and inside the
        generated ``__dace_exit_<name>()`` finalizer. It performs no imports of
        its own, so it is safe to run during interpreter shutdown.

        :param stream: The output stream to write the generated Python code to.
        :param sdfg: The top-level SDFG whose report is being written.
        """
        build_folder = os.path.abspath(sdfg.build_folder)
        # Hash is computed at code-generation time (frozen), matching C++ backend behavior
        sdfg_hash = sdfg.hash_sdfg()

        # ``__dace_os`` / ``__dace_json`` / ``__dace_time`` are imported at module
        # scope in ``on_sdfg_begin`` for the top-level SDFG, so they are already
        # globals in scope here (no local imports -- see on_sdfg_begin).
        stream.write(
            f'__dace_perf_dir = __dace_os.path.join({build_folder!r}, "perf")'
        )
        stream.write('__dace_os.makedirs(__dace_perf_dir, exist_ok=True)')
        stream.write('__dace_perf_ts = int(__dace_time.time() * 1000)')
        stream.write(
            '__dace_perf_path = __dace_os.path.join('
            '__dace_perf_dir, f"report-{__dace_perf_ts}.json")'
        )
        # The ``state_id``/``id`` args are intentionally always emitted (even
        # when ``-1``): the report reader (``report.py``
        # ``get_event_uuid_and_other_info``) keys off their *presence*, so a
        # future "omit -1 keys" optimization must NOT be applied here -- doing
        # so would break parity with the C++ reader.
        stream.write(
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
        stream.write('with open(__dace_perf_path, "w") as __dace_perf_fp:')
        stream.indent()
        stream.write('__dace_json.dump(__dace_perf_report, __dace_perf_fp)')
        stream.dedent()

    # ------------------------------------------------------------------
    # State hooks
    # ------------------------------------------------------------------

    def on_state_begin(self, sdfg: SDFG, cfg: ControlFlowRegion,
                       state: SDFGState, local_stream: PythonCodeIOStream,
                       global_stream: PythonCodeIOStream) -> None:
        if state.instrument == dtypes.InstrumentationType.PythonTimer:
            self._emit_tbegin(local_stream, cfg, state, None)

    def on_state_end(self, sdfg: SDFG, cfg: ControlFlowRegion,
                     state: SDFGState, local_stream: PythonCodeIOStream,
                     global_stream: PythonCodeIOStream) -> None:
        if state.instrument == dtypes.InstrumentationType.PythonTimer:
            self._emit_tend(f'State {state.label}', local_stream, cfg, state, None)

    # ------------------------------------------------------------------
    # Scope hooks
    # ------------------------------------------------------------------

    def _get_sobj(self, node: nodes.Node) -> Union[nodes.Map, nodes.Consume]:
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
        if s.instrument == dtypes.InstrumentationType.PythonTimer:
            self._emit_tbegin(outer_stream, cfg, state, node)

    def on_scope_exit(self, sdfg: SDFG, cfg: ControlFlowRegion,
                      state: SDFGState, node: nodes.ExitNode,
                      outer_stream: PythonCodeIOStream,
                      inner_stream: PythonCodeIOStream,
                      global_stream: PythonCodeIOStream) -> None:
        entry_node = state.entry_node(node)
        s = self._get_sobj(node)
        if s.instrument == dtypes.InstrumentationType.PythonTimer:
            self._emit_tend(
                f'{type(s).__name__} {s.label}',
                outer_stream, cfg, state, entry_node,
            )

    # ------------------------------------------------------------------
    # Node hooks
    # ------------------------------------------------------------------

    def on_node_begin(self, sdfg: SDFG, cfg: ControlFlowRegion,
                      state: SDFGState, node: nodes.Node,
                      outer_stream: PythonCodeIOStream,
                      inner_stream: PythonCodeIOStream,
                      global_stream: PythonCodeIOStream) -> None:
        if not isinstance(node, CodeNode):
            return
        if node.instrument == dtypes.InstrumentationType.PythonTimer:
            self._emit_tbegin(outer_stream, cfg, state, node)

    def on_node_end(self, sdfg: SDFG, cfg: ControlFlowRegion,
                    state: SDFGState, node: nodes.Node,
                    outer_stream: PythonCodeIOStream,
                    inner_stream: PythonCodeIOStream,
                    global_stream: PythonCodeIOStream) -> None:
        if not isinstance(node, CodeNode):
            return
        if node.instrument == dtypes.InstrumentationType.PythonTimer:
            idstr = self._idstr(cfg, state, node)
            self._emit_tend(
                f'{type(node).__name__} {idstr}',
                outer_stream, cfg, state, node,
            )
