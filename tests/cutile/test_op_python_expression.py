"""Unit tests for :func:`op_python_expression` and :data:`_CT_PREFIXED_FUNCS`
in :mod:`dace.libraries.cutile.nodes.base`.
"""

import pytest

from dace.libraries.cutile.nodes.base import (
    _CT_PREFIXED_FUNCS,
    op_python_expression,
)


# ── Backward compatibility (ct_prefix=False, the default) ───────────


class TestOpPythonExpressionDefault:
    """Tests with default ``ct_prefix=False`` ensuring backward compatibility."""

    def test_binary_add(self):
        assert op_python_expression("+", "_a", "_b") == "(_a + _b)"

    def test_binary_sub(self):
        assert op_python_expression("-", "_a", "_b") == "(_a - _b)"

    def test_binary_mul(self):
        assert op_python_expression("*", "_a", "_b") == "(_a * _b)"

    def test_binary_div(self):
        assert op_python_expression("/", "_a", "_b") == "(_a / _b)"

    def test_binary_mod(self):
        assert op_python_expression("%", "x", "y") == "(x % y)"

    def test_unary_negate(self):
        assert op_python_expression("-", "_a") == "(-_a)"

    def test_unary_positive(self):
        assert op_python_expression("+", "_a") == "(+_a)"

    def test_unary_abs(self):
        assert op_python_expression("abs", "_a") == "abs(_a)"

    def test_unary_sin(self):
        assert op_python_expression("sin", "_a") == "sin(_a)"

    def test_unary_cos(self):
        assert op_python_expression("cos", "_a") == "cos(_a)"

    def test_unary_exp(self):
        assert op_python_expression("exp", "_a") == "exp(_a)"

    def test_unary_sqrt(self):
        assert op_python_expression("sqrt", "_a") == "sqrt(_a)"

    def test_unary_log(self):
        assert op_python_expression("log", "_a") == "log(_a)"

    def test_unary_ceil(self):
        assert op_python_expression("ceil", "_a") == "ceil(_a)"

    def test_unary_floor(self):
        assert op_python_expression("floor", "_a") == "floor(_a)"


# ── ct_prefix=True ──────────────────────────────────────────────────


class TestOpPythonExpressionCtPrefix:
    """Tests with ``ct_prefix=True``."""

    # -- Prefixed unary math functions --

    def test_sin_prefixed(self):
        assert op_python_expression("sin", "_a", ct_prefix=True) == "ct.sin(_a)"

    def test_cos_prefixed(self):
        assert op_python_expression("cos", "_a", ct_prefix=True) == "ct.cos(_a)"

    def test_exp_prefixed(self):
        assert op_python_expression("exp", "_a", ct_prefix=True) == "ct.exp(_a)"

    def test_sqrt_prefixed(self):
        assert op_python_expression("sqrt", "_a", ct_prefix=True) == "ct.sqrt(_a)"

    def test_log_prefixed(self):
        assert op_python_expression("log", "_a", ct_prefix=True) == "ct.log(_a)"

    def test_ceil_prefixed(self):
        assert op_python_expression("ceil", "_a", ct_prefix=True) == "ct.ceil(_a)"

    def test_floor_prefixed(self):
        assert op_python_expression("floor", "_a", ct_prefix=True) == "ct.floor(_a)"

    # -- Built-in functions NOT prefixed even with ct_prefix=True --

    def test_abs_not_prefixed(self):
        assert op_python_expression("abs", "_a", ct_prefix=True) == "abs(_a)"

    # -- Unary sign operators unaffected by ct_prefix --

    def test_negate_with_prefix(self):
        assert op_python_expression("-", "_a", ct_prefix=True) == "(-_a)"

    def test_positive_with_prefix(self):
        assert op_python_expression("+", "_a", ct_prefix=True) == "(+_a)"

    # -- Binary ops unaffected by ct_prefix --

    def test_binary_add_with_prefix(self):
        assert op_python_expression("+", "_a", "_b", ct_prefix=True) == "(_a + _b)"

    def test_binary_sub_with_prefix(self):
        assert op_python_expression("-", "_a", "_b", ct_prefix=True) == "(_a - _b)"

    def test_binary_mul_with_prefix(self):
        assert op_python_expression("*", "_a", "_b", ct_prefix=True) == "(_a * _b)"


# ── _CT_PREFIXED_FUNCS constant ────────────────────────────────────


class TestCtPrefixedFuncsConstant:
    """Tests for the :data:`_CT_PREFIXED_FUNCS` constant."""

    def test_is_frozenset(self):
        assert isinstance(_CT_PREFIXED_FUNCS, frozenset)

    def test_expected_members(self):
        expected = {"sin", "cos", "exp", "sqrt", "log", "ceil", "floor"}
        assert _CT_PREFIXED_FUNCS == expected

    def test_abs_not_in_prefixed(self):
        assert "abs" not in _CT_PREFIXED_FUNCS

    def test_immutable(self):
        with pytest.raises(AttributeError):
            _CT_PREFIXED_FUNCS.add("tanh")


# ── Edge cases ──────────────────────────────────────────────────────


class TestOpPythonExpressionEdgeCases:
    """Edge cases for :func:`op_python_expression`."""

    def test_custom_operand_names(self):
        result = op_python_expression("sin", "my_var", ct_prefix=True)
        assert result == "ct.sin(my_var)"

    def test_complex_operand_expression(self):
        result = op_python_expression("sqrt", "(_a + _b)", ct_prefix=True)
        assert result == "ct.sqrt((_a + _b))"

    def test_binary_with_constants(self):
        result = op_python_expression("+", "3.14", "2.71")
        assert result == "(3.14 + 2.71)"

    def test_unknown_unary_func_no_prefix(self):
        """An unknown function name should never get the ct. prefix."""
        result = op_python_expression("tanh", "_a", ct_prefix=True)
        assert result == "tanh(_a)"

    def test_unknown_unary_func_default(self):
        result = op_python_expression("tanh", "_a")
        assert result == "tanh(_a)"

    def test_explicit_ct_prefix_false(self):
        """Explicitly passing ct_prefix=False matches default behavior."""
        result = op_python_expression("sin", "_a", ct_prefix=False)
        assert result == "sin(_a)"

    def test_right_none_explicit(self):
        """Explicitly passing right=None is the same as omitting it."""
        assert op_python_expression("sin", "_a", None, ct_prefix=True) == "ct.sin(_a)"

    def test_all_ct_prefixed_funcs_get_prefix(self):
        """Every function in _CT_PREFIXED_FUNCS gets the ct. prefix."""
        for func in _CT_PREFIXED_FUNCS:
            result = op_python_expression(func, "x", ct_prefix=True)
            assert result == f"ct.{func}(x)", f"Failed for {func}"

    def test_all_ct_prefixed_funcs_no_prefix_by_default(self):
        """No function gets ct. prefix when ct_prefix is False (default)."""
        for func in _CT_PREFIXED_FUNCS:
            result = op_python_expression(func, "x")
            assert result == f"{func}(x)", f"Failed for {func}"
