# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for PythonCompiledSDFG and compile_python_sdfg."""

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


# ---------------------------------------------------------------------------
# PythonCompiledSDFG.__init__
# ---------------------------------------------------------------------------

def test_init_success():
    """Valid code defining the expected function → function extracted."""
    sdfg = _make_sdfg("my_func")
    code = "def my_func(x):\n    return x + 1\n"
    csdfg = PythonCompiledSDFG(sdfg, code)
    assert csdfg._func is not None
    assert callable(csdfg._func)


def test_init_missing_function():
    """Code without matching function → RuntimeError."""
    sdfg = _make_sdfg("expected_name")
    code = "def wrong_name():\n    pass\n"
    with pytest.raises(RuntimeError, match="does not define function 'expected_name'"):
        PythonCompiledSDFG(sdfg, code)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

def test_sdfg_property():
    """.sdfg returns the original SDFG."""
    sdfg = _make_sdfg("f")
    code = "def f(): pass\n"
    csdfg = PythonCompiledSDFG(sdfg, code)
    assert csdfg.sdfg is sdfg


def test_code_property():
    """.code returns the code string."""
    sdfg = _make_sdfg("f")
    code = "def f(): pass\n"
    csdfg = PythonCompiledSDFG(sdfg, code)
    assert csdfg.code == code


def test_finalize_allows_reinitialize():
    """finalize tears down runtime state so a later call reinitializes it."""
    sdfg = _make_sdfg("lifecycle")
    code = (
        "init_calls = []\n"
        "exit_calls = []\n"
        "def __dace_init_lifecycle(value):\n"
        "    init_calls.append(value)\n"
        "def __dace_exit_lifecycle():\n"
        "    exit_calls.append(len(init_calls))\n"
        "def lifecycle(value):\n"
        "    return len(init_calls), len(exit_calls)\n"
    )
    csdfg = PythonCompiledSDFG(sdfg, code)

    # Keyword call: the SDFG declares no arguments, so positional args are
    # rejected (see test_call_positional_to_argless_sdfg_raises).
    assert csdfg(value=1) == (1, 0)
    csdfg.finalize()
    assert csdfg._namespace['exit_calls'] == [1]

    assert csdfg(value=2) == (2, 1)
    csdfg.finalize()
    assert csdfg._namespace['init_calls'] == [1, 2]
    assert csdfg._namespace['exit_calls'] == [1, 2]


# ---------------------------------------------------------------------------
# __call__
# ---------------------------------------------------------------------------

def test_call_with_args():
    """__call__ binds positional args onto arglist() order."""
    sdfg = _make_sdfg("add")
    sdfg.add_scalar("a", dace.int64)
    sdfg.add_scalar("b", dace.int64)
    code = "def add(a, b):\n    return a[()] + b[()]\n"
    csdfg = PythonCompiledSDFG(sdfg, code)
    assert csdfg(2, 3) == 5


def test_call_positional_to_argless_sdfg_raises():
    """Positional args to an SDFG that declares none raise (not swallowed)."""
    sdfg = _make_sdfg("add")
    code = "def add(a, b):\n    return a + b\n"
    csdfg = PythonCompiledSDFG(sdfg, code)
    with pytest.raises(KeyError, match="does not accept them"):
        csdfg(2, 3)


def test_call_excess_positional_args_raise():
    """More positional args than arglist() entries raise TypeError."""
    sdfg = _make_sdfg("add")
    sdfg.add_scalar("a", dace.int64)
    sdfg.add_scalar("b", dace.int64)
    code = "def add(a, b):\n    return a[()] + b[()]\n"
    csdfg = PythonCompiledSDFG(sdfg, code)
    with pytest.raises(TypeError, match="accepts at most 2"):
        csdfg(2, 3, 4)


def test_call_duplicate_positional_and_keyword_raises():
    """An argument passed both ways raises ValueError (not swallowed)."""
    sdfg = _make_sdfg("add")
    sdfg.add_scalar("a", dace.int64)
    sdfg.add_scalar("b", dace.int64)
    code = "def add(a, b):\n    return a[()] + b[()]\n"
    csdfg = PythonCompiledSDFG(sdfg, code)
    with pytest.raises(ValueError, match="both positional and keyword"):
        csdfg(2, a=2, b=3)


def test_call_with_kwargs():
    """__call__ passes keyword args correctly."""
    sdfg = _make_sdfg("add")
    code = "def add(a, b):\n    return a + b\n"
    csdfg = PythonCompiledSDFG(sdfg, code)
    assert csdfg(a=10, b=20) == 30


def test_call_with_return_value():
    """__call__ returns function result."""
    sdfg = _make_sdfg("get_val")
    code = "def get_val():\n    return 42\n"
    csdfg = PythonCompiledSDFG(sdfg, code)
    assert csdfg() == 42


# ---------------------------------------------------------------------------
# __del__
# ---------------------------------------------------------------------------

def test_del_no_crash():
    """Delete PythonCompiledSDFG without error."""
    sdfg = _make_sdfg("f")
    code = "def f(): pass\n"
    csdfg = PythonCompiledSDFG(sdfg, code)
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


def test_compile_python_sdfg_auxiliary_module_importable():
    """Linkable auxiliary code objects are resolvable via the import hook."""
    sdfg = _make_sdfg("my_fn")
    co_aux = CodeObject(
        name="my_helper",
        code="HELPER_VALUE = 42\n",
        language="Python",
        target=None,
        title="Helper",
        linkable=True,
    )
    co_frame = CodeObject(
        name="my_fn",
        code="from my_helper import HELPER_VALUE\ndef my_fn():\n    return HELPER_VALUE\n",
        language="Python",
        target=None,
        title="Frame",
    )
    csdfg = compile_python_sdfg(sdfg, [co_frame, co_aux])
    assert csdfg() == 42


def test_compile_python_sdfg_does_not_touch_sys_modules():
    """Auxiliary modules are resolved without ever being placed in sys.modules."""
    import sys
    sdfg = _make_sdfg("clean_fn")
    co_aux = CodeObject(
        name="clean_aux_module",
        code="AUX = 99\n",
        language="Python",
        target=None,
        title="Aux",
        linkable=True,
    )
    co_frame = CodeObject(
        name="clean_fn",
        code="from clean_aux_module import AUX\ndef clean_fn():\n    return AUX\n",
        language="Python",
        target=None,
        title="Frame",
    )
    assert "clean_aux_module" not in sys.modules
    csdfg = compile_python_sdfg(sdfg, [co_frame, co_aux])
    assert csdfg() == 99
    assert "clean_aux_module" not in sys.modules
    csdfg.finalize()
    assert "clean_aux_module" not in sys.modules


def test_compile_python_sdfg_non_linkable_not_imported():
    """Non-linkable code objects are excluded from the import hook."""
    sdfg = _make_sdfg("fn")
    co_frame = CodeObject(
        name="fn",
        code="def fn():\n    return 1\n",
        language="Python",
        target=None,
        title="Frame",
    )
    co_nonlinkable = CodeObject(
        name="sample_main_module",
        code="SHOULD_NOT_EXIST = True\n",
        language="Python",
        target=None,
        title="SampleMain",
        linkable=False,
    )
    csdfg = compile_python_sdfg(sdfg, [co_frame, co_nonlinkable])
    assert csdfg() == 1
    assert "sample_main_module" not in csdfg._aux_modules


def test_compile_python_sdfg_stdlib_imports_still_work():
    """Imports of stdlib modules fall through to the real importer."""
    sdfg = _make_sdfg("uses_stdlib")
    co_frame = CodeObject(
        name="uses_stdlib",
        code="import math\ndef uses_stdlib():\n    return math.floor(3.7)\n",
        language="Python",
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
    csdfg = PythonCompiledSDFG(sdfg, code)
    assert csdfg.do_not_execute is False


def test_libhandle_default():
    """_libhandle defaults to None."""
    sdfg = _make_sdfg("f")
    code = "def f(): return 42\n"
    csdfg = PythonCompiledSDFG(sdfg, code)
    assert csdfg._libhandle is None


def test_do_not_execute_skips_execution():
    """When do_not_execute is True, __call__ initializes but does not run the function."""
    call_log = []
    sdfg = _make_sdfg("tracked")
    code = (
        "call_log = []\n"
        "def tracked():\n"
        "    call_log.append('called')\n"
        "    return 99\n"
    )
    csdfg = PythonCompiledSDFG(sdfg, code)
    # Inject the same log list so we can inspect it
    csdfg._namespace['call_log'] = call_log

    csdfg.do_not_execute = True
    result = csdfg()
    assert result is None
    assert call_log == []  # Function was NOT called


def test_do_not_execute_still_initializes():
    """When do_not_execute is True, __call__ still runs initialization."""
    sdfg = _make_sdfg("init_test")
    code = (
        "init_count = [0]\n"
        "def __dace_init_init_test():\n"
        "    init_count[0] += 1\n"
        "def init_test():\n"
        "    return init_count[0]\n"
    )
    csdfg = PythonCompiledSDFG(sdfg, code)
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
    csdfg = PythonCompiledSDFG(sdfg, code)

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
    csdfg = PythonCompiledSDFG(sdfg, code)
    assert callable(csdfg._cfunc)


def test_cfunc_ignores_handle():
    """_cfunc ignores the first argument (handle), passes the rest through."""
    sdfg = _make_sdfg("add")
    code = "def add(a, b):\n    return a + b\n"
    csdfg = PythonCompiledSDFG(sdfg, code)
    # Call with None as handle (matching Python backend _libhandle)
    assert csdfg._cfunc(None, 3, 7) == 10


def test_cfunc_ignores_handle_with_kwargs():
    """_cfunc passes kwargs through correctly."""
    sdfg = _make_sdfg("add")
    code = "def add(a, b):\n    return a + b\n"
    csdfg = PythonCompiledSDFG(sdfg, code)
    assert csdfg._cfunc(None, a=10, b=20) == 30


def test_cfunc_with_no_args():
    """_cfunc works with handle-only call (no additional args)."""
    sdfg = _make_sdfg("noop")
    code = "def noop():\n    return 'done'\n"
    csdfg = PythonCompiledSDFG(sdfg, code)
    assert csdfg._cfunc(None) == 'done'


def test_cfunc_with_libhandle():
    """_cfunc(csdfg._libhandle, *args) works -- the profiler's calling convention."""
    sdfg = _make_sdfg("mul")
    code = "def mul(a, b):\n    return a * b\n"
    csdfg = PythonCompiledSDFG(sdfg, code)
    result = csdfg._cfunc(csdfg._libhandle, 6, 7)
    assert result == 42


def test_profiler_interface_complete():
    """PythonCompiledSDFG exposes all attributes the CompiledSDFGProfiler needs."""
    sdfg = _make_sdfg("profiled")
    sdfg.add_scalar("x", dace.int64)
    code = "def profiled(x):\n    return x\n"
    csdfg = PythonCompiledSDFG(sdfg, code)

    # The profiler accesses these attributes:
    assert hasattr(csdfg, '_cfunc')
    assert hasattr(csdfg, '_libhandle')
    assert hasattr(csdfg, 'do_not_execute')
    assert hasattr(csdfg, 'sdfg')

    # Simulate exactly what CompiledSDFGProfiler does:
    #   compiled_sdfg._cfunc(compiled_sdfg._libhandle, *args)
    args = (42,)
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
