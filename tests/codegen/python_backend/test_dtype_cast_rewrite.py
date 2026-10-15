# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.codegen.py.utils import rewrite_dtype_casts


def test_shadowed_bare_dtype_name_is_not_rewritten():
    body = """def int64(value):
    return value + 2
result = int64(value)"""

    assert rewrite_dtype_casts(body) == body


def test_dace_bool_uses_numpy_bool_alias():
    rewritten = rewrite_dtype_casts("result = dace.bool(value)")

    assert "numpy.bool_" in rewritten
    assert "numpy.bool(" not in rewritten
    namespace = {"dace": dace, "numpy": np, "value": np.array(True)}
    exec(rewritten, namespace)
    assert namespace["result"] == np.bool_(True)


@pytest.mark.parametrize(
    ("dtype_name", "dtype"),
    [
        ("bfloat16", dace.bfloat16),
        ("float8_e4m3fn", dace.float8_e4m3fn),
        ("float8_e5m2", dace.float8_e5m2),
    ],
)
def test_registered_extended_dtype_uses_dace_typeclass(dtype_name, dtype):
    rewritten = rewrite_dtype_casts(f"result = {dtype_name}(value)")

    assert f"dace.{dtype_name}" in rewritten
    assert f"numpy.{dtype_name}" not in rewritten
    namespace = {"dace": dace, "numpy": np, "value": np.array(1.25)}
    exec(rewritten, namespace)
    assert namespace["result"] == dtype(1.25)


def test_shadowed_dtype_tasklet_runs_user_function():
    sdfg = dace.SDFG("shadowed_dtype_tasklet")
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array("x", [1], dace.float64)
    sdfg.add_array("y", [1], dace.float64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet(
        "shadowed_cast",
        {"value"},
        {"result"},
        """def int64(value):
    return value + 2
result = int64(value)""",
    )
    state.add_edge(state.add_read("x"), None, tasklet, "value", dace.Memlet("x[0]"))
    state.add_edge(tasklet, "result", state.add_write("y"), None, dace.Memlet("y[0]"))

    result = np.zeros(1)
    sdfg.compile()(x=np.array([3.75]), y=result)

    assert result[0] == 5.75


def test_array_dtype_cast_preserves_array():
    """A dtype constructor on an array must not unwrap it as a scalar."""
    @dace.program
    def cast_array(A: dace.float32[3], B: dace.float64[3]):
        with dace.tasklet:
            cast_input << A[:]
            cast_output >> B[:]
            cast_output = np.float64(cast_input)

    sdfg = cast_array.to_sdfg(simplify=False)
    sdfg.backend = dtypes.BackendLanguage.Python
    a = np.array([1.25, -2.5, 3.75], dtype=np.float32)
    b = np.full(a.shape, np.nan, dtype=np.float64)
    sdfg.compile()(A=a, B=b)
    np.testing.assert_array_equal(b, np.float64(a))
