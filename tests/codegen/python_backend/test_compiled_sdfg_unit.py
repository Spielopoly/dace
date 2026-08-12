# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for PythonCompiledSDFG and compile_python_sdfg."""

from contextlib import contextmanager
import gc

import pytest
import numpy as np

import dace
from dace.codegen.py.compiled_sdfg import PythonCompiledSDFG, compile_python_sdfg
from dace.codegen.codeobject import CodeObject


def _make_sdfg(name: str = "test_func") -> dace.SDFG:
    """Create a minimal SDFG with the given name."""
    sdfg = dace.SDFG(name)
    sdfg.backend = dace.dtypes.BackendLanguage.Python
    state = sdfg.add_state("init")
    return sdfg


def _compile_code(sdfg: dace.SDFG, code: str) -> PythonCompiledSDFG:
    """Compile a focused Cython host fixture through the native compiler."""
    code_object = CodeObject(
        name=sdfg.name,
        code=code,
        language='pyx',
        target=None,
        title='Frame',
    )
    return compile_python_sdfg(sdfg, [code_object])


# ---------------------------------------------------------------------------
# PythonCompiledSDFG.__init__
# ---------------------------------------------------------------------------


def test_init_success():
    """Valid code defining the expected function → function extracted."""
    sdfg = _make_sdfg("my_func")
    code = "def my_func(x):\n    return x + 1\n"
    csdfg = _compile_code(sdfg, code)
    assert csdfg._func is not None
    assert callable(csdfg._func)


def test_init_missing_function():
    """Code without matching function → RuntimeError."""
    sdfg = _make_sdfg("expected_name")
    code = "def wrong_name():\n    pass\n"
    with pytest.raises(RuntimeError, match="does not define function 'expected_name'"):
        _compile_code(sdfg, code)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_sdfg_property():
    """.sdfg returns the original SDFG."""
    sdfg = _make_sdfg("f")
    code = "def f(): pass\n"
    csdfg = _compile_code(sdfg, code)
    assert csdfg.sdfg is sdfg


def test_code_property():
    """.code returns the code string."""
    sdfg = _make_sdfg("f")
    code = "def f(): pass\n"
    csdfg = _compile_code(sdfg, code)
    assert csdfg.code == code


def test_finalize_allows_reinitialize():
    """finalize tears down runtime state so a later call reinitializes it."""
    sdfg = _make_sdfg("lifecycle")
    code = ("init_calls = []\n"
            "exit_calls = []\n"
            "def __dace_init_lifecycle(value):\n"
            "    init_calls.append(value)\n"
            "def __dace_exit_lifecycle():\n"
            "    exit_calls.append(len(init_calls))\n"
            "def lifecycle(value):\n"
            "    return len(init_calls), len(exit_calls)\n")
    csdfg = _compile_code(sdfg, code)

    # Keyword call: the SDFG declares no arguments, so positional args are
    # rejected (see test_call_positional_to_argless_sdfg_raises).
    assert csdfg(value=1) == (1, 0)
    csdfg.finalize()
    assert csdfg._namespace['exit_calls'] == [1]

    assert csdfg(value=2) == (2, 1)
    csdfg.finalize()
    assert csdfg._namespace['init_calls'] == [1, 2]
    assert csdfg._namespace['exit_calls'] == [1, 2]


def test_cached_handles_have_independent_native_state():
    """Two handles for one cached extension must not share module globals."""
    sdfg = _make_sdfg("isolated_handles")
    code = ("counter = 0\n"
            "__dace_persistent_transients = {}\n"
            "def __dace_init_isolated_handles():\n"
            "    global counter\n"
            "    counter = 0\n"
            "def isolated_handles():\n"
            "    global counter\n"
            "    counter += 1\n"
            "    return counter\n")
    first = _compile_code(sdfg, code)
    second = _compile_code(sdfg, code)

    assert first.module is not second.module
    assert [first(), first(), second(), first()] == [1, 2, 1, 3]
    first.finalize()
    assert second() == 2


def test_failing_exit_finalizes_handle_once():
    """A failing exit hook still clears state and reaches a terminal state."""
    sdfg = _make_sdfg("failing_exit")
    code = ("exit_calls = 0\n"
            "__dace_persistent_transients = {'value': 1}\n"
            "def __dace_exit_failing_exit():\n"
            "    global exit_calls\n"
            "    exit_calls += 1\n"
            "    raise RuntimeError('exit failed')\n"
            "def failing_exit():\n"
            "    pass\n")
    compiled = _compile_code(sdfg, code)
    compiled()

    with pytest.raises(RuntimeError, match='exit failed'):
        compiled.finalize()
    assert compiled._initialized is False
    assert compiled._finalized is True
    assert compiled._namespace['__dace_persistent_transients'] == {}
    assert compiled._namespace['exit_calls'] == 1

    compiled.finalize()
    assert compiled._namespace['exit_calls'] == 1


def test_finalize_before_initialize_does_not_run_exit():
    """A fresh handle must not run user exit code, including from __del__."""
    sdfg = _make_sdfg("never_initialized")
    code = ("exit_calls = 0\n"
            "def __dace_exit_never_initialized():\n"
            "    global exit_calls\n"
            "    exit_calls += 1\n"
            "def never_initialized():\n"
            "    pass\n")
    compiled = _compile_code(sdfg, code)
    module = compiled.module

    compiled.finalize()
    assert module.exit_calls == 0

    del compiled
    gc.collect()
    assert module.exit_calls == 0


def test_failed_initialize_does_not_run_exit():
    """A failed initializer must not make explicit or destructor cleanup call exit."""
    sdfg = _make_sdfg("failed_initialize")
    code = ("exit_calls = 0\n"
            "def __dace_init_failed_initialize():\n"
            "    raise RuntimeError('init failed')\n"
            "def __dace_exit_failed_initialize():\n"
            "    global exit_calls\n"
            "    exit_calls += 1\n"
            "def failed_initialize():\n"
            "    pass\n")
    compiled = _compile_code(sdfg, code)
    module = compiled.module

    with pytest.raises(RuntimeError, match='init failed'):
        compiled()
    assert compiled._initialized is False

    compiled.finalize()
    del compiled
    gc.collect()
    assert module.exit_calls == 0


# ---------------------------------------------------------------------------
# __call__
# ---------------------------------------------------------------------------


def test_call_with_args():
    """__call__ binds positional args onto arglist() order."""
    sdfg = _make_sdfg("add")
    sdfg.add_scalar("a", dace.int64)
    sdfg.add_scalar("b", dace.int64)
    code = "def add(a, b):\n    return a[()] + b[()]\n"
    csdfg = _compile_code(sdfg, code)
    assert csdfg(2, 3) == 5


def test_call_positional_to_argless_sdfg_raises():
    """Positional args to an SDFG that declares none raise (not swallowed)."""
    sdfg = _make_sdfg("add")
    code = "def add(a, b):\n    return a + b\n"
    csdfg = _compile_code(sdfg, code)
    with pytest.raises(KeyError, match="does not accept them"):
        csdfg(2, 3)


def test_call_excess_positional_args_raise():
    """More positional args than arglist() entries raise TypeError."""
    sdfg = _make_sdfg("add")
    sdfg.add_scalar("a", dace.int64)
    sdfg.add_scalar("b", dace.int64)
    code = "def add(a, b):\n    return a[()] + b[()]\n"
    csdfg = _compile_code(sdfg, code)
    with pytest.raises(TypeError, match="accepts at most 2"):
        csdfg(2, 3, 4)


def test_call_duplicate_positional_and_keyword_raises():
    """An argument passed both ways raises ValueError (not swallowed)."""
    sdfg = _make_sdfg("add")
    sdfg.add_scalar("a", dace.int64)
    sdfg.add_scalar("b", dace.int64)
    code = "def add(a, b):\n    return a[()] + b[()]\n"
    csdfg = _compile_code(sdfg, code)
    with pytest.raises(ValueError, match="both positional and keyword"):
        csdfg(2, a=2, b=3)


def test_call_with_kwargs():
    """__call__ passes keyword args correctly."""
    sdfg = _make_sdfg("add")
    code = "def add(a, b):\n    return a + b\n"
    csdfg = _compile_code(sdfg, code)
    assert csdfg(a=10, b=20) == 30


def test_call_with_return_value():
    """__call__ returns function result."""
    sdfg = _make_sdfg("get_val")
    code = "def get_val():\n    return 42\n"
    csdfg = _compile_code(sdfg, code)
    assert csdfg() == 42


# ---------------------------------------------------------------------------
# __del__
# ---------------------------------------------------------------------------


def test_del_no_crash():
    """Delete PythonCompiledSDFG without error."""
    sdfg = _make_sdfg("f")
    code = "def f(): pass\n"
    csdfg = _compile_code(sdfg, code)
    del csdfg  # should not raise


# ---------------------------------------------------------------------------
# compile_python_sdfg
# ---------------------------------------------------------------------------


def test_compile_python_sdfg_empty_code_objects():
    """Empty list → RuntimeError."""
    sdfg = _make_sdfg("f")
    with pytest.raises(RuntimeError, match="No code objects generated"):
        compile_python_sdfg(sdfg, [])


def test_compile_python_sdfg_success():
    """Full pipeline: create SDFG, generate code, compile, verify PythonCompiledSDFG returned."""
    sdfg = dace.SDFG("compile_test")
    sdfg.backend = dace.dtypes.BackendLanguage.Python
    sdfg.add_array("A", [1], dace.float64)
    sdfg.add_array("B", [1], dace.float64)
    state = sdfg.add_state("compute")
    a_node = state.add_access("A")
    b_node = state.add_access("B")
    tasklet = state.add_tasklet("add1", {"a"}, {"b"}, "b = a + 1")
    state.add_edge(a_node, None, tasklet, "a", dace.Memlet("A[0]"))
    state.add_edge(tasklet, "b", b_node, None, dace.Memlet("B[0]"))

    code_objects = sdfg.generate_code()
    csdfg = compile_python_sdfg(sdfg, code_objects)

    assert isinstance(csdfg, PythonCompiledSDFG)
    A = np.array([5.0], dtype=np.float64)
    B = np.array([0.0], dtype=np.float64)
    csdfg(A=A, B=B)
    np.testing.assert_array_equal(B, [6.0])


def test_compile_python_sdfg_rejects_auxiliary_module():
    """The native backend accepts one host source, not runtime helper modules."""
    sdfg = _make_sdfg("my_fn")
    host = CodeObject(name='my_fn', code='def my_fn(): return 1\n', language='pyx', target=None, title='Frame')
    auxiliary = CodeObject(name='helper', code='VALUE = 1\n', language='py', target=None, title='Helper')
    with pytest.raises(RuntimeError, match='Unexpected Python-backend CodeObject'):
        compile_python_sdfg(sdfg, [host, auxiliary])


def test_compiled_extension_uses_hashed_sys_modules_key():
    """The loaded extension is registered only under its hashed internal name."""
    import sys
    csdfg = _compile_code(_make_sdfg('clean_fn'), 'def clean_fn(): return 99\n')
    assert csdfg() == 99
    assert sys.modules[csdfg.module.__name__] is csdfg.module


def test_compile_python_sdfg_rejects_non_linkable_auxiliary_source():
    """Non-linkable arbitrary sources are not native host inputs."""
    sdfg = _make_sdfg("fn")
    host = CodeObject(name='fn', code='def fn(): return 1\n', language='pyx', target=None, title='Frame')
    auxiliary = CodeObject(name='sample',
                           code='VALUE = 1\n',
                           language='py',
                           target=None,
                           title='Sample',
                           linkable=False)
    with pytest.raises(RuntimeError, match='Unexpected Python-backend CodeObject'):
        compile_python_sdfg(sdfg, [host, auxiliary])


def test_compile_python_sdfg_stdlib_imports_still_work():
    """Imports of stdlib modules fall through to the real importer."""
    sdfg = _make_sdfg("uses_stdlib")
    co_frame = CodeObject(
        name="uses_stdlib",
        code="import math\ndef uses_stdlib():\n    return math.floor(3.7)\n",
        language="pyx",
        target=None,
        title="Frame",
    )
    csdfg = compile_python_sdfg(sdfg, [co_frame])
    assert csdfg() == 3


# ---------------------------------------------------------------------------
# Profiler compatibility (do_not_execute, _libhandle, _cfunc)
# ---------------------------------------------------------------------------


def test_do_not_execute_default():
    """do_not_execute defaults to False."""
    sdfg = _make_sdfg("f")
    code = "def f(): return 42\n"
    csdfg = _compile_code(sdfg, code)
    assert csdfg.do_not_execute is False


def test_libhandle_default():
    """_libhandle defaults to None."""
    sdfg = _make_sdfg("f")
    code = "def f(): return 42\n"
    csdfg = _compile_code(sdfg, code)
    assert csdfg._libhandle is None


def test_do_not_execute_skips_execution():
    """When do_not_execute is True, __call__ initializes but does not run the function."""
    call_log = []
    sdfg = _make_sdfg("tracked")
    code = ("call_log = []\n"
            "def tracked():\n"
            "    call_log.append('called')\n"
            "    return 99\n")
    csdfg = _compile_code(sdfg, code)
    # Inject the same log list so we can inspect it
    csdfg._namespace['call_log'] = call_log

    csdfg.do_not_execute = True
    result = csdfg()
    assert result is None
    assert call_log == []  # Function was NOT called


def test_do_not_execute_still_initializes():
    """When do_not_execute is True, __call__ still runs initialization."""
    sdfg = _make_sdfg("init_test")
    code = ("init_count = [0]\n"
            "def __dace_init_init_test():\n"
            "    init_count[0] += 1\n"
            "def init_test():\n"
            "    return init_count[0]\n")
    csdfg = _compile_code(sdfg, code)
    csdfg.do_not_execute = True
    csdfg()
    # Initialization should have happened
    assert csdfg._initialized is True
    assert csdfg._namespace['init_count'][0] == 1


def test_do_not_execute_toggle():
    """do_not_execute can be toggled on and off, matching profiler save/restore pattern."""
    sdfg = _make_sdfg("toggle")
    sdfg.add_scalar("x", dace.int64)
    code = "def toggle(x):\n    return x * 2\n"
    csdfg = _compile_code(sdfg, code)

    # Normal call
    assert csdfg(5) == 10

    # Save old value, set True (as profiler does)
    old_dne = csdfg.do_not_execute
    csdfg.do_not_execute = True
    result = csdfg(5)
    assert result is None

    # Restore (as profiler does)
    csdfg.do_not_execute = old_dne
    assert csdfg(5) == 10


def test_cfunc_returns_callable():
    """_cfunc returns a callable."""
    sdfg = _make_sdfg("f")
    code = "def f(x):\n    return x + 1\n"
    csdfg = _compile_code(sdfg, code)
    assert callable(csdfg._cfunc)


def test_cfunc_ignores_handle():
    """_cfunc ignores the first argument (handle), passes the rest through."""
    sdfg = _make_sdfg("add")
    code = "def add(a, b):\n    return a + b\n"
    csdfg = _compile_code(sdfg, code)
    # Call with None as handle (matching Python backend _libhandle)
    assert csdfg._cfunc(None, 3, 7) == 10


def test_cfunc_ignores_handle_with_kwargs():
    """_cfunc passes kwargs through correctly."""
    sdfg = _make_sdfg("add")
    code = "def add(a, b):\n    return a + b\n"
    csdfg = _compile_code(sdfg, code)
    assert csdfg._cfunc(None, a=10, b=20) == 30


def test_cfunc_with_no_args():
    """_cfunc works with handle-only call (no additional args)."""
    sdfg = _make_sdfg("noop")
    code = "def noop():\n    return 'done'\n"
    csdfg = _compile_code(sdfg, code)
    assert csdfg._cfunc(None) == 'done'


def test_cfunc_with_libhandle():
    """_cfunc(csdfg._libhandle, *args) works -- the profiler's calling convention."""
    sdfg = _make_sdfg("mul")
    code = "def mul(a, b):\n    return a * b\n"
    csdfg = _compile_code(sdfg, code)
    result = csdfg._cfunc(csdfg._libhandle, 6, 7)
    assert result == 42


def test_profiler_interface_complete():
    """PythonCompiledSDFG exposes all attributes the CompiledSDFGProfiler needs."""
    sdfg = _make_sdfg("profiled")
    sdfg.add_scalar("x", dace.int64)
    code = "def profiled(x):\n    return x\n"
    csdfg = _compile_code(sdfg, code)

    # The profiler accesses these attributes:
    assert hasattr(csdfg, '_cfunc')
    assert hasattr(csdfg, '_libhandle')
    assert hasattr(csdfg, 'do_not_execute')
    assert hasattr(csdfg, 'sdfg')

    # Simulate exactly what CompiledSDFGProfiler does:
    #   compiled_sdfg._cfunc(compiled_sdfg._libhandle, *args)
    args = (42, )
    result = csdfg._cfunc(csdfg._libhandle, *args)
    assert result == 42

    #   old_dne = compiled_sdfg.do_not_execute
    #   compiled_sdfg.do_not_execute = True
    old_dne = csdfg.do_not_execute
    csdfg.do_not_execute = True
    assert csdfg() is None  # call should be suppressed

    #   compiled_sdfg.do_not_execute = old_dne
    csdfg.do_not_execute = old_dne
    assert csdfg(42) == 42  # back to normal


def test_public_call_runs_compiled_sdfg_profiler_in_generated_arg_order():
    """The public adapter must enter hooks and let the profiler replace execution."""
    from dace.frontend.operations import CompiledSDFGProfiler

    sdfg = _make_sdfg("profiled_public_call")
    sdfg.add_scalar("z_value", dace.int64)
    sdfg.add_scalar("a_value", dace.int64)
    argnames = list(sdfg.arglist().keys())
    parameters = ", ".join(argnames)
    recorded_values = ", ".join(f"int({name}[()])" for name in argnames)
    code = ("calls = []\n"
            f"def profiled_public_call({parameters}):\n"
            f"    calls.append(({recorded_values},))\n")
    compiled = _compile_code(sdfg, code)

    values = {argnames[0]: 17, argnames[1]: 29}
    reversed_kwargs = {argnames[1]: values[argnames[1]], argnames[0]: values[argnames[0]]}
    profiler = CompiledSDFGProfiler(repetitions=2, warmup=1, print_results=False)
    with dace.hooks.on_compiled_sdfg_call(context_manager=profiler):
        result = compiled(**reversed_kwargs)

    assert result is None
    assert compiled.module.calls == [tuple(values[name] for name in argnames)] * 3
    assert len(profiler.times) == 1


def test_public_call_honors_hook_replacement_handle():
    """A hook-provided handle replaces the original native call."""
    sdfg = _make_sdfg("replace_public_call")
    sdfg.add_scalar("z_value", dace.int64)
    sdfg.add_scalar("a_value", dace.int64)
    argnames = list(sdfg.arglist().keys())
    parameters = ", ".join(argnames)
    code = ("original_calls = 0\n"
            f"def replace_public_call({parameters}):\n"
            "    global original_calls\n"
            "    original_calls += 1\n")
    compiled = _compile_code(sdfg, code)
    replacement_calls = []
    sentinel = object()

    class Replacement:

        do_not_execute = False
        _libhandle = sentinel

        def _cfunc(self, handle, *args):
            assert handle is sentinel
            replacement_calls.append(tuple(int(value[()]) for value in args))
            return "replacement"

    replacement = Replacement()

    @contextmanager
    def replace_handle(_compiled, _hook_args):
        yield replacement

    values = {argnames[0]: 31, argnames[1]: 47}
    reversed_kwargs = {argnames[1]: values[argnames[1]], argnames[0]: values[argnames[0]]}
    with dace.hooks.on_compiled_sdfg_call(context_manager=replace_handle):
        result = compiled(**reversed_kwargs)

    assert result == "replacement"
    assert replacement_calls == [tuple(values[name] for name in argnames)]
    assert compiled.module.original_calls == 0
