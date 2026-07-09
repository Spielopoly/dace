# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the in-kernel tasklet rewrites in the cuTile backend.

``_rewrite_cutile_tasklet_code`` adapts a Python tasklet body for execution
inside a ``@ct.kernel``:

* bare-name math calls (host numpy aliases) become ``ct.*`` spellings — a
  kernel cannot capture a numpy ufunc as a constant;
* ternaries whose condition reads a tile connector become ``ct.where`` —
  tiles cannot be branched on — with numeric-literal arms cast to the
  opposite tile arm's dtype (mixed where-arm dtypes are rejected).
"""
import pytest

from dace.codegen.py.cutile_target import _rewrite_cutile_tasklet_code


class TestMathNameMapping:

    def test_binary_math_call(self):
        code = _rewrite_cutile_tasklet_code("__out = atan2(__in1, __in2)", {})
        assert code == "__out = ct.atan2(__in1, __in2)"

    def test_unary_math_call(self):
        code = _rewrite_cutile_tasklet_code("__out = sinh(__in1)", {})
        assert code == "__out = ct.sinh(__in1)"

    def test_alias_spellings(self):
        assert "ct.atan2" in _rewrite_cutile_tasklet_code("o = arctan2(a, b)", {})
        assert "ct.ceil" in _rewrite_cutile_tasklet_code("o = ceiling(a)", {})
        assert "ct.abs" in _rewrite_cutile_tasklet_code("o = fabs(a)", {})
        assert "ct.minimum" in _rewrite_cutile_tasklet_code("o = fmin(a, b)", {})

    def test_nested_calls(self):
        code = _rewrite_cutile_tasklet_code("o = sqrt(sin(a) ** 2 + cos(a) ** 2)", {})
        assert "ct.sqrt(ct.sin(a) ** 2 + ct.cos(a) ** 2)" in code

    def test_attribute_calls_untouched(self):
        code = _rewrite_cutile_tasklet_code("o = ct.load(a, (0,), shape=())", {})
        assert code == "o = ct.load(a, (0,), shape=())"

    def test_unmapped_host_alias_warns(self):
        with pytest.warns(UserWarning, match="arcsin"):
            _rewrite_cutile_tasklet_code("o = arcsin(a)", {})

    def test_unknown_function_untouched(self):
        code = _rewrite_cutile_tasklet_code("o = my_helper(a)", {})
        assert code == "o = my_helper(a)"


class TestTernaryToWhere:

    def test_tile_condition_rewritten(self):
        code = _rewrite_cutile_tasklet_code("__out = (__in1 if __incond else __in2)",
                                            {"__incond": "bool_", "__in1": "float64", "__in2": "float64"})
        assert code == "__out = ct.where(__incond, __in1, __in2)"

    def test_scalar_condition_untouched(self):
        code = _rewrite_cutile_tasklet_code("__out = (1 if k > 0 else 2)", {})
        assert "ct.where" not in code

    def test_literal_arm_cast_to_tile_dtype(self):
        code = _rewrite_cutile_tasklet_code("__out = (0 if __incond else __in2)",
                                            {"__incond": "bool_", "__in2": "float64"})
        assert code == "__out = ct.where(__incond, ct.astype(0, ct.float64), __in2)"

    def test_bool_literal_arm_not_cast(self):
        code = _rewrite_cutile_tasklet_code("__out = (True if __incond else __in2)",
                                            {"__incond": "bool_", "__in2": "bool_"})
        assert "astype" not in code
        assert "ct.where(__incond, True, __in2)" in code

    def test_syntax_error_passthrough(self):
        bad = "this is not python ("
        assert _rewrite_cutile_tasklet_code(bad, {}) == bad

    # --- Literal-arm casting edge cases (negative / fractional literals) ---

    def test_negative_literal_arm_cast(self):
        """-1.0 is ast.UnaryOp(USub, Constant); it must be cast like 1.0."""
        code = _rewrite_cutile_tasklet_code("__out = (-1.0 if __incond else __in2)",
                                            {"__incond": "bool_", "__in2": "float64"})
        assert code == "__out = ct.where(__incond, ct.astype(-1.0, ct.float64), __in2)"

    def test_positive_signed_literal_arm_cast(self):
        code = _rewrite_cutile_tasklet_code("__out = (+2 if __incond else __in2)",
                                            {"__incond": "bool_", "__in2": "int64"})
        assert code == "__out = ct.where(__incond, ct.astype(+2, ct.int64), __in2)"

    def test_fractional_literal_vs_int_tile_not_cast(self):
        """ct.astype(0.5, ct.int32) silently truncates to 0 — leave uncast."""
        code = _rewrite_cutile_tasklet_code("__out = (0.5 if __incond else __in2)",
                                            {"__incond": "bool_", "__in2": "int32"})
        assert "astype" not in code
        assert "ct.where(__incond, 0.5, __in2)" in code

    def test_negative_fractional_literal_vs_int_tile_not_cast(self):
        code = _rewrite_cutile_tasklet_code("__out = (-0.5 if __incond else __in2)",
                                            {"__incond": "bool_", "__in2": "int32"})
        assert "astype" not in code

    def test_integral_float_literal_vs_int_tile_cast(self):
        """2.0 -> int is lossless, so the cast is applied."""
        code = _rewrite_cutile_tasklet_code("__out = (2.0 if __incond else __in2)",
                                            {"__incond": "bool_", "__in2": "int32"})
        assert code == "__out = ct.where(__incond, ct.astype(2.0, ct.int32), __in2)"

    def test_fractional_literal_vs_float_tile_cast(self):
        code = _rewrite_cutile_tasklet_code("__out = (0.5 if __incond else __in2)",
                                            {"__incond": "bool_", "__in2": "float32"})
        assert code == "__out = ct.where(__incond, ct.astype(0.5, ct.float32), __in2)"


class TestTaintedLocalCondition:
    """Ternaries conditioned on a tasklet-LOCAL name holding a tile value
    (assigned earlier in the body) must also be rewritten to ct.where."""

    def test_local_assigned_from_tile_conn(self):
        code = _rewrite_cutile_tasklet_code(
            "cond = __in1 > 0\n__out = (__in2 if cond else __in3)",
            {"__in1": "float64", "__in2": "float64", "__in3": "float64"})
        assert "ct.where(cond, __in2, __in3)" in code

    def test_transitively_tainted_local(self):
        code = _rewrite_cutile_tasklet_code(
            "a = __in1 * 2\nb = a > 1\n__out = (0.0 if b else __in2)",
            {"__in1": "float64", "__in2": "float64"})
        assert "ct.where(b, ct.astype(0.0, ct.float64), __in2)" in code

    def test_augassigned_local_tainted(self):
        code = _rewrite_cutile_tasklet_code(
            "acc = 0\nacc += __in1\n__out = (__in2 if acc > 0 else __in3)",
            {"__in1": "int64", "__in2": "int64", "__in3": "int64"})
        assert "ct.where(acc > 0, __in2, __in3)" in code

    def test_untainted_local_condition_untouched(self):
        code = _rewrite_cutile_tasklet_code(
            "k = 3\n__out = (__in1 if k > 0 else 2)", {"__in1": "float64"})
        assert "ct.where" not in code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
