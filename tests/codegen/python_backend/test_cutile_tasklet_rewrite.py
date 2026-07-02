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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
