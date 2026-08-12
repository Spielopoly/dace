# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Integration tests for the Python backend's math-name aliases.

Host-side tasklet code is emitted verbatim with the C/sympy math names DaCe
uses (``atan2``, ``fmin``, ``pow``, ``cpp_mod``, ...); the generated frame does
inlines explicit NumPy-backed aliases so Cython can resolve them. These tests
compile and run Python-backend SDFGs whose tasklets use the aliased names and
compare against NumPy (regression: the arc_distance ``atan2`` failure).
"""
import itertools

import numpy as np
import pytest

import dace
from dace.dtypes import BackendLanguage
from dace.sdfg import SDFG

_SDFG_COUNTER = itertools.count()


def _new_sdfg(prefix: str) -> SDFG:
    """Create a Python-backend SDFG with a unique name."""
    sdfg = SDFG(f"{prefix}_{next(_SDFG_COUNTER)}")
    sdfg.backend = BackendLanguage.Python
    return sdfg


def _run_binary_tasklet(code: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Build/compile/run a 1-element-per-tasklet binary-op SDFG.

    :param code: Tasklet code using connectors ``lhs``, ``rhs`` -> ``out``.
    :param a: First operand array.
    :param b: Second operand array.
    :returns: The result array.
    """
    size = len(a)
    sdfg = _new_sdfg("binary_alias")
    sdfg.add_array('A', [size], dace.float64)
    sdfg.add_array('B', [size], dace.float64)
    sdfg.add_array('C', [size], dace.float64)
    state = sdfg.add_state(is_start_block=True)
    a_read = state.add_read('A')
    b_read = state.add_read('B')
    c_write = state.add_write('C')
    for idx in range(size):
        tasklet = state.add_tasklet(f'op_{idx}', {'lhs', 'rhs'}, {'out'}, code)
        state.add_edge(a_read, None, tasklet, 'lhs', dace.Memlet(f'A[{idx}]'))
        state.add_edge(b_read, None, tasklet, 'rhs', dace.Memlet(f'B[{idx}]'))
        state.add_edge(tasklet, 'out', c_write, None, dace.Memlet(f'C[{idx}]'))
    out = np.zeros(size)
    sdfg(A=a.copy(), B=b.copy(), C=out)
    return out


def _run_unary_tasklet(code: str, a: np.ndarray) -> np.ndarray:
    """Build/compile/run a 1-element-per-tasklet unary-op SDFG.

    :param code: Tasklet code using connector ``inp`` -> ``out``.
    :param a: Operand array.
    :returns: The result array.
    """
    size = len(a)
    sdfg = _new_sdfg("unary_alias")
    sdfg.add_array('A', [size], dace.float64)
    sdfg.add_array('C', [size], dace.float64)
    state = sdfg.add_state(is_start_block=True)
    a_read = state.add_read('A')
    c_write = state.add_write('C')
    for idx in range(size):
        tasklet = state.add_tasklet(f'op_{idx}', {'inp'}, {'out'}, code)
        state.add_edge(a_read, None, tasklet, 'inp', dace.Memlet(f'A[{idx}]'))
        state.add_edge(tasklet, 'out', c_write, None, dace.Memlet(f'C[{idx}]'))
    out = np.zeros(size)
    sdfg(A=a.copy(), C=out)
    return out


class TestExplicitAliasTasklets:
    """Tasklets calling the C-style names directly (the emitted spellings)."""

    @pytest.mark.parametrize("code,ref", [
        ("out = atan2(lhs, rhs)", np.arctan2),
        ("out = fmin(lhs, rhs)", np.fmin),
        ("out = fmax(lhs, rhs)", np.fmax),
        ("out = pow(lhs, rhs)", np.power),
        ("out = hypot(lhs, rhs)", np.hypot),
        ("out = copysign(lhs, rhs)", np.copysign),
        ("out = cpp_mod(lhs, rhs)", np.fmod),
        ("out = py_mod(lhs, rhs)", np.mod),
        ("out = np_float_pow(lhs, rhs)", np.float_power),
    ])
    def test_binary_alias(self, code, ref):
        rng = np.random.default_rng(11)
        a = rng.uniform(-3.0, 3.0, 8)
        b = rng.uniform(0.5, 3.0, 8) * np.where(rng.random(8) < 0.5, -1.0, 1.0)
        out = _run_binary_tasklet(code, a, b)
        np.testing.assert_allclose(out, ref(a, b), rtol=1e-14)

    @pytest.mark.parametrize("code,ref", [
        ("out = asin(inp)", np.arcsin),
        ("out = acos(inp)", np.arccos),
        ("out = atan(inp)", np.arctan),
        ("out = round(inp)", np.round),
        ("out = trunc(inp)", np.trunc),
        ("out = cbrt(inp)", np.cbrt),
        ("out = expm1(inp)", np.expm1),
        ("out = log1p(inp)", np.log1p),
    ])
    def test_unary_alias(self, code, ref):
        rng = np.random.default_rng(12)
        a = rng.uniform(-0.99, 0.99, 8)
        out = _run_unary_tasklet(code, a)
        np.testing.assert_allclose(out, ref(a), rtol=1e-14)


class TestFrontendPrograms:
    """End-to-end ``@dace.program`` -> Python backend -> run vs NumPy."""

    def test_arctan2_program(self):
        """The documented arc_distance failure: ``np.arctan2`` on host."""
        N = dace.symbol("N")

        @dace.program
        def prog(X: dace.float64[N], Y: dace.float64[N], Z: dace.float64[N]):
            Z[:] = np.arctan2(X, Y)

        sdfg = prog.to_sdfg()
        sdfg.backend = BackendLanguage.Python
        rng = np.random.default_rng(21)
        X = rng.uniform(-2, 2, 16)
        Y = rng.uniform(-2, 2, 16)
        Z = np.zeros(16)
        sdfg(X=X, Y=Y, Z=Z, N=16)
        np.testing.assert_allclose(Z, np.arctan2(X, Y), rtol=1e-14)

    def test_fmin_fmax_program(self):
        N = dace.symbol("N")

        @dace.program
        def prog(X: dace.float64[N], Y: dace.float64[N], Z: dace.float64[N]):
            Z[:] = np.fmin(X, Y) + np.fmax(X, Y)

        sdfg = prog.to_sdfg()
        sdfg.backend = BackendLanguage.Python
        rng = np.random.default_rng(22)
        X = rng.uniform(-2, 2, 16)
        Y = rng.uniform(-2, 2, 16)
        Z = np.zeros(16)
        sdfg(X=X, Y=Y, Z=Z, N=16)
        np.testing.assert_allclose(Z, np.fmin(X, Y) + np.fmax(X, Y), rtol=1e-14)

    def test_fmod_mod_mixed_signs_program(self):
        """``np.fmod`` (C-sign) vs ``np.mod`` (Python-sign) on host."""
        N = dace.symbol("N")

        @dace.program
        def prog(X: dace.float64[N], Y: dace.float64[N], F: dace.float64[N], M: dace.float64[N]):
            F[:] = np.fmod(X, Y)
            M[:] = np.mod(X, Y)

        sdfg = prog.to_sdfg()
        sdfg.backend = BackendLanguage.Python
        X = np.array([-7.5, 7.5, -7.5, 7.5, 5.0, -5.0, 0.0, 1.25] * 2)
        Y = np.array([3.0, 3.0, -3.0, -3.0, 2.5, 2.5, 3.0, -0.5] * 2)
        F = np.zeros(16)
        M = np.zeros(16)
        sdfg(X=X, Y=Y, F=F, M=M, N=16)
        np.testing.assert_allclose(F, np.fmod(X, Y), rtol=1e-14)
        np.testing.assert_allclose(M, np.mod(X, Y), rtol=1e-14)

    def test_power_program(self):
        N = dace.symbol("N")

        @dace.program
        def prog(X: dace.float64[N], Y: dace.float64[N], Z: dace.float64[N]):
            Z[:] = np.power(X, Y)

        sdfg = prog.to_sdfg()
        sdfg.backend = BackendLanguage.Python
        rng = np.random.default_rng(23)
        X = rng.uniform(0.1, 3.0, 16)
        Y = rng.uniform(-2.0, 2.0, 16)
        Z = np.zeros(16)
        sdfg(X=X, Y=Y, Z=Z, N=16)
        np.testing.assert_allclose(Z, np.power(X, Y), rtol=1e-14)


class TestIntCeilFloorSemantics:
    """``int_ceil`` / ``int_floor`` stay integer-exact (no float cast)."""

    def test_int_ceil_int_floor_are_exact(self):
        import importlib.util
        import pathlib
        path = pathlib.Path(dace.__file__).parent / "codegen" / "py" / "sympy_function_redefinitions.py"
        spec = importlib.util.spec_from_file_location("sfr_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        big = 2**62 + 3  # would lose precision under any float cast
        assert mod.int_ceil(big, 2) == (big + 1) // 2
        assert mod.int_floor(big, 2) == big // 2
        assert isinstance(mod.int_ceil(7, 2), int) and mod.int_ceil(7, 2) == 4
        assert isinstance(mod.int_floor(7, 2), int) and mod.int_floor(7, 2) == 3


class TestITE:
    """``ITE`` branches scalars directly and dispatches array conditions
    through ``np.where`` (which reaches cupy via __array_function__)."""

    @staticmethod
    def _load_mod():
        import importlib.util
        import pathlib
        path = pathlib.Path(dace.__file__).parent / "codegen" / "py" / "sympy_function_redefinitions.py"
        spec = importlib.util.spec_from_file_location("sfr_ite_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_scalar_condition_branches(self):
        mod = self._load_mod()
        assert mod.ITE(True, 1, 2) == 1
        assert mod.ITE(False, 1, 2) == 2

    def test_host_array_condition(self):
        mod = self._load_mod()
        cond = np.array([True, False, True])
        out = mod.ITE(cond, np.array([1, 1, 1]), np.array([2, 2, 2]))
        np.testing.assert_array_equal(out, [1, 2, 1])

    @pytest.mark.gpu
    def test_host_condition_device_arms_coerced(self):
        """A host (numpy) condition with device (cupy) arms: the condition is
        coerced into the arms' namespace so np.where dispatches to cupy."""
        cupy = pytest.importorskip("cupy")
        mod = self._load_mod()
        cond = np.array([True, False, True])  # host
        then_v = cupy.asarray([10.0, 10.0, 10.0])  # device
        else_v = cupy.asarray([20.0, 20.0, 20.0])
        out = mod.ITE(cond, then_v, else_v)
        assert isinstance(out, cupy.ndarray)
        np.testing.assert_array_equal(cupy.asnumpy(out), [10.0, 20.0, 10.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
