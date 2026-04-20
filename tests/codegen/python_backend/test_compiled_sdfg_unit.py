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


# ---------------------------------------------------------------------------
# __call__
# ---------------------------------------------------------------------------

def test_call_with_args():
    """__call__ passes positional args correctly."""
    sdfg = _make_sdfg("add")
    code = "def add(a, b):\n    return a + b\n"
    csdfg = PythonCompiledSDFG(sdfg, code)
    assert csdfg(2, 3) == 5


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


def test_compile_python_sdfg_multiple_objects():
    """Uses first code object only; second is ignored."""
    sdfg = _make_sdfg("my_fn")
    co1 = CodeObject(
        name="first",
        code="def my_fn(x):\n    return x * 2\n",
        language="Python",
        target=None,
        title="Frame",
    )
    co2 = CodeObject(
        name="second",
        code="def my_fn(x):\n    return x * 3\n",
        language="Python",
        target=None,
        title="Frame",
    )
    csdfg = compile_python_sdfg(sdfg, [co1, co2])
    # First code object defines x*2
    assert csdfg(5) == 10
