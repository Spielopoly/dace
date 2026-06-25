# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for TileIota cuTile expansion robustness: dtype generality and
C++ expr validation.

These tests verify:
1. ``_resolve_dst_cutile_dtype`` resolves the correct cuTile dtype from the
   ``_dst`` output descriptor, falling back to ``ct.int32`` when no
   descriptor is available.
2. ``validate_cutile_expr`` rejects obviously-C++ patterns (``std::``,
   ``->``, trailing ``;``, ``sizeof``) and accepts valid Python exprs.
3. The ``ExpandTileIotaCutile.expansion()`` method emits the correct dtype
   in ``ct.arange()`` calls for different output array dtypes.
4. The expansion raises ``ValueError`` for C++ expressions.
"""
import ast

import pytest

import dace
from dace.libraries.tileops import TileIota
from dace.libraries.tileops._cutile_dtypes import _DACE_TO_CUTILE_DTYPE
from dace.libraries.tileops._pure_codegen import (
    _CPP_PATTERN,
    validate_cutile_expr,
)
from dace.libraries.tileops.nodes.tile_iota import (
    _resolve_dst_cutile_dtype,
)


# ============================================================
# _DACE_TO_CUTILE_DTYPE mapping
# ============================================================


def test_dtype_mapping_covers_all_expected_types():
    """The mapping covers all float, signed/unsigned integer, and bool dtypes."""
    expected = {
        "float16", "float32", "float64",
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "bool",
    }
    assert set(_DACE_TO_CUTILE_DTYPE.keys()) == expected


def test_dtype_mapping_values_use_ct_prefix():
    """All values in the mapping use the ``ct.`` prefix and match the key.

    The sole exception is ``bool``, which maps to cuTile's ``ct.bool_``
    (cuTile exposes the boolean type as ``bool_``, not ``bool``).
    """
    for key, val in _DACE_TO_CUTILE_DTYPE.items():
        assert val.startswith("ct."), f"Value for {key!r} should start with 'ct.': {val!r}"
        if key == "bool":
            assert val == "ct.bool_"
        else:
            assert val == f"ct.{key}"


# ============================================================
# validate_cutile_expr
# ============================================================


def test_validate_accepts_simple_python_expr():
    """A simple Python expression raises nothing."""
    validate_cutile_expr("i + __l0")
    validate_cutile_expr("__l0 * 4 + __l1")
    validate_cutile_expr("_idx[__l0]")
    validate_cutile_expr("__l0 * 32 + __l1 * 8 + __l2")
    validate_cutile_expr("i + 2 * __l0")


def test_validate_accepts_python_builtins():
    """Python-style expressions with valid function calls pass."""
    validate_cutile_expr("max(__l0, 0)")
    validate_cutile_expr("abs(__l0 - 5)")
    validate_cutile_expr("int(__l0 / 2)")


def test_validate_rejects_std_namespace():
    """Expressions with ``std::`` are rejected."""
    with pytest.raises(ValueError, match="C\\+\\+-flavored"):
        validate_cutile_expr("std::max(__l0, 0)")


def test_validate_rejects_arrow_operator():
    """Expressions with ``->`` are rejected."""
    with pytest.raises(ValueError, match="C\\+\\+-flavored"):
        validate_cutile_expr("ptr->field + __l0")


def test_validate_rejects_trailing_semicolon():
    """Expressions with a trailing ``;`` are rejected."""
    with pytest.raises(ValueError, match="C\\+\\+-flavored"):
        validate_cutile_expr("i + __l0;")


def test_validate_rejects_sizeof():
    """Expressions with ``sizeof`` are rejected."""
    with pytest.raises(ValueError, match="C\\+\\+-flavored"):
        validate_cutile_expr("sizeof(int) * __l0")


def test_validate_error_message_includes_match_and_expr():
    """The error message includes the matched pattern and the full expr."""
    expr = "std::min(__l0, N)"
    with pytest.raises(ValueError) as exc_info:
        validate_cutile_expr(expr)
    msg = str(exc_info.value)
    assert "std::" in msg
    assert expr in msg


def test_validate_does_not_reject_colon_in_python_slice():
    """Colons in Python slice syntax should NOT trigger the C++ check."""
    validate_cutile_expr("_src[__l0:__l0+4]")


def test_validate_does_not_reject_minus_greater_in_comment():
    """The ``->`` check matches literally; a valid Python expression
    without ``->`` should pass."""
    validate_cutile_expr("__l0 - 1")


# ============================================================
# _resolve_dst_cutile_dtype
# ============================================================


def _make_iota_sdfg_with_dtype(dtype, widths=(8,)):
    """Build a minimal SDFG with a TileIota wired to a ``_dst`` array
    of the given dtype, and return ``(node, state, sdfg)``."""
    sdfg = dace.SDFG(f"iota_dtype_{dtype.as_numpy_dtype().name}")
    arr_name = "_tile"
    sdfg.add_array(arr_name, list(widths), dtype,
                   storage=dace.dtypes.StorageType.Register, transient=True)
    state = sdfg.add_state()
    node = TileIota("iota", widths=widths, expr="__l0")
    state.add_node(node)
    tile_acc = state.add_access(arr_name)
    memlet_str = f"{arr_name}[" + ", ".join(f"0:{w}" for w in widths) + "]"
    state.add_edge(node, "_dst", tile_acc, None, dace.Memlet(memlet_str))
    return node, state, sdfg


def test_resolve_dtype_fallback_none_state():
    """When parent_state is None, fallback to ct.int32."""
    node = TileIota("iota", widths=(8,), expr="__l0")
    result = _resolve_dst_cutile_dtype(node, None, None)
    assert result == "ct.int32"


def test_resolve_dtype_fallback_none_sdfg():
    """When parent_sdfg is None, fallback to ct.int32."""
    node = TileIota("iota", widths=(8,), expr="__l0")
    sdfg = dace.SDFG("test")
    state = sdfg.add_state()
    state.add_node(node)
    result = _resolve_dst_cutile_dtype(node, state, None)
    assert result == "ct.int32"


def test_resolve_dtype_fallback_no_edges():
    """When node has no output edges, fallback to ct.int32."""
    sdfg = dace.SDFG("test")
    state = sdfg.add_state()
    node = TileIota("iota", widths=(8,), expr="__l0")
    state.add_node(node)
    result = _resolve_dst_cutile_dtype(node, state, sdfg)
    assert result == "ct.int32"


def test_resolve_dtype_int32():
    """Resolve int32 from wired _dst descriptor."""
    node, state, sdfg = _make_iota_sdfg_with_dtype(dace.int32)
    result = _resolve_dst_cutile_dtype(node, state, sdfg)
    assert result == "ct.int32"


def test_resolve_dtype_int64():
    """Resolve int64 from wired _dst descriptor."""
    node, state, sdfg = _make_iota_sdfg_with_dtype(dace.int64)
    result = _resolve_dst_cutile_dtype(node, state, sdfg)
    assert result == "ct.int64"


def test_resolve_dtype_float32():
    """Resolve float32 from wired _dst descriptor."""
    node, state, sdfg = _make_iota_sdfg_with_dtype(dace.float32)
    result = _resolve_dst_cutile_dtype(node, state, sdfg)
    assert result == "ct.float32"


def test_resolve_dtype_float64():
    """Resolve float64 from wired _dst descriptor."""
    node, state, sdfg = _make_iota_sdfg_with_dtype(dace.float64)
    result = _resolve_dst_cutile_dtype(node, state, sdfg)
    assert result == "ct.float64"


def test_resolve_dtype_float16():
    """Resolve float16 from wired _dst descriptor."""
    node, state, sdfg = _make_iota_sdfg_with_dtype(dace.float16)
    result = _resolve_dst_cutile_dtype(node, state, sdfg)
    assert result == "ct.float16"


def test_resolve_dtype_uint8():
    """uint8 is a supported dtype and resolves to ct.uint8."""
    node, state, sdfg = _make_iota_sdfg_with_dtype(dace.uint8)
    result = _resolve_dst_cutile_dtype(node, state, sdfg)
    assert result == "ct.uint8"


def test_resolve_dtype_falls_back_to_int32_without_descriptor():
    """With no wired descriptor (bare expansion), resolution falls back to ct.int32."""
    node, state, sdfg = _make_iota_sdfg_with_dtype(dace.int32)
    result = _resolve_dst_cutile_dtype(node, None, None)
    assert result == "ct.int32"


# ============================================================
# ExpandTileIotaCutile.expansion() — dtype-aware ct.arange
# ============================================================


def _expand_cutile_with_dst_dtype(dtype, widths=(8,), expr="__l0"):
    """Expand a TileIota with cutile implementation and return the
    tasklet code body as a string."""
    node, state, sdfg = _make_iota_sdfg_with_dtype(dtype, widths=widths)
    node.implementation = "cutile"
    # Need a map entry so that validation passes.
    me, mx = state.add_map("m", {"i": "0:1"})
    state.add_nedge(me, node, dace.Memlet())
    # Wire map exit.
    tile_acc = None
    for e in state.out_edges(node):
        if e.src_conn == "_dst":
            tile_acc = e.dst
            break
    state.add_nedge(tile_acc, mx, dace.Memlet())
    out_name = "OUT"
    sdfg.add_array(out_name, list(widths), dtype)
    out_acc = state.add_access(out_name)
    state.add_nedge(mx, out_acc, dace.Memlet())

    cls = node.implementations["cutile"]
    tasklet = cls.expansion(node, state, sdfg)
    return tasklet.code.as_string, tasklet.language


def test_expansion_k1_int32_emits_ct_int32():
    """K=1 with int32 output: arange uses ct.int32."""
    body, lang = _expand_cutile_with_dst_dtype(dace.int32)
    assert "ct.arange(8, dtype=ct.int32)" in body
    assert lang == dace.dtypes.Language.Python


def test_expansion_k1_int64_emits_ct_int64():
    """K=1 with int64 output: arange uses ct.int64."""
    body, lang = _expand_cutile_with_dst_dtype(dace.int64)
    assert "ct.arange(8, dtype=ct.int64)" in body
    assert lang == dace.dtypes.Language.Python


def test_expansion_k1_float32_emits_ct_float32():
    """K=1 with float32 output: arange uses ct.float32."""
    body, lang = _expand_cutile_with_dst_dtype(dace.float32)
    assert "ct.arange(8, dtype=ct.float32)" in body
    assert lang == dace.dtypes.Language.Python


def test_expansion_k1_float64_emits_ct_float64():
    """K=1 with float64 output: arange uses ct.float64."""
    body, lang = _expand_cutile_with_dst_dtype(dace.float64)
    assert "ct.arange(8, dtype=ct.float64)" in body
    assert lang == dace.dtypes.Language.Python


def test_expansion_k2_int64_emits_ct_int64_in_broadcast():
    """K=2 with int64 output: broadcast_to(arange(..., dtype=ct.int64)...)."""
    body, lang = _expand_cutile_with_dst_dtype(
        dace.int64, widths=(2, 4), expr="__l0 * 4 + __l1")
    assert "ct.arange(2, dtype=ct.int64)" in body
    assert "ct.arange(4, dtype=ct.int64)" in body
    assert "ct.broadcast_to" in body
    assert lang == dace.dtypes.Language.Python


def test_expansion_k2_float32_emits_ct_float32_in_broadcast():
    """K=2 with float32 output: broadcast_to(arange(..., dtype=ct.float32)...)."""
    body, lang = _expand_cutile_with_dst_dtype(
        dace.float32, widths=(4, 8), expr="__l0 * 8 + __l1")
    assert "ct.arange(4, dtype=ct.float32)" in body
    assert "ct.arange(8, dtype=ct.float32)" in body
    assert lang == dace.dtypes.Language.Python


def test_expansion_k3_int64_emits_ct_int64_in_all_broadcasts():
    """K=3 with int64 output: all three arange calls use ct.int64."""
    body, lang = _expand_cutile_with_dst_dtype(
        dace.int64, widths=(2, 4, 8), expr="__l0 * 32 + __l1 * 8 + __l2")
    assert body.count("dtype=ct.int64") == 3
    assert lang == dace.dtypes.Language.Python


def test_expansion_single_lane_does_not_emit_arange_regardless_of_dtype():
    """Single-lane (all widths=1): no ct.arange regardless of dtype."""
    body, lang = _expand_cutile_with_dst_dtype(
        dace.float64, widths=(1,), expr="__l0")
    assert "ct.arange" not in body
    # Lane var substituted to 0.
    assert "0" in body
    assert lang == dace.dtypes.Language.Python


# ============================================================
# ExpandTileIotaCutile.expansion() — expr validation
# ============================================================


def test_expansion_rejects_cpp_std_namespace():
    """Expansion with a ``std::`` expression raises ValueError."""
    node, state, sdfg = _make_iota_sdfg_with_dtype(dace.int32)
    node.implementation = "cutile"
    # Forcefully set a C++ expression.
    node.expr = "std::max(__l0, 0)"
    cls = node.implementations["cutile"]
    with pytest.raises(ValueError, match="C\\+\\+-flavored"):
        cls.expansion(node, state, sdfg)


def test_expansion_rejects_cpp_arrow_operator():
    """Expansion with a ``->`` expression raises ValueError."""
    node, state, sdfg = _make_iota_sdfg_with_dtype(dace.int32)
    node.implementation = "cutile"
    node.expr = "ptr->field + __l0"
    cls = node.implementations["cutile"]
    with pytest.raises(ValueError, match="C\\+\\+-flavored"):
        cls.expansion(node, state, sdfg)


def test_expansion_rejects_cpp_sizeof():
    """Expansion with ``sizeof`` expression raises ValueError."""
    node, state, sdfg = _make_iota_sdfg_with_dtype(dace.int32)
    node.implementation = "cutile"
    node.expr = "sizeof(int) * __l0"
    cls = node.implementations["cutile"]
    with pytest.raises(ValueError, match="C\\+\\+-flavored"):
        cls.expansion(node, state, sdfg)


def test_expansion_rejects_trailing_semicolon():
    """Expansion with trailing ``;`` expression raises ValueError."""
    node, state, sdfg = _make_iota_sdfg_with_dtype(dace.int32)
    node.implementation = "cutile"
    node.expr = "__l0 + 1;"
    cls = node.implementations["cutile"]
    with pytest.raises(ValueError, match="C\\+\\+-flavored"):
        cls.expansion(node, state, sdfg)


def test_expansion_accepts_valid_python_expr():
    """Expansion with a valid Python expression succeeds."""
    body, lang = _expand_cutile_with_dst_dtype(dace.int32, expr="__l0 + 1")
    ast.parse(body)
    assert "__l0" in body
    assert lang == dace.dtypes.Language.Python


def test_expansion_single_lane_rejects_cpp_expr():
    """Even the single-lane (all widths=1) path validates the expr."""
    node, state, sdfg = _make_iota_sdfg_with_dtype(dace.int32, widths=(1,))
    node.implementation = "cutile"
    node.expr = "std::max(0, 1)"
    cls = node.implementations["cutile"]
    with pytest.raises(ValueError, match="C\\+\\+-flavored"):
        cls.expansion(node, state, sdfg)


# ============================================================
# _CPP_PATTERN regex — exhaustive coverage
# ============================================================


def test_cpp_pattern_matches_std():
    """Regex matches ``std::``."""
    assert _CPP_PATTERN.search("std::max(a, b)")


def test_cpp_pattern_matches_arrow():
    """Regex matches ``->``."""
    assert _CPP_PATTERN.search("p->x")


def test_cpp_pattern_matches_trailing_semicolon():
    """Regex matches trailing ``;``."""
    assert _CPP_PATTERN.search("x + 1;")


def test_cpp_pattern_matches_semicolon_with_trailing_spaces():
    """Regex matches ``;`` followed by whitespace only."""
    assert _CPP_PATTERN.search("x + 1;  ")


def test_cpp_pattern_matches_sizeof():
    """Regex matches ``sizeof``."""
    assert _CPP_PATTERN.search("sizeof(int)")


def test_cpp_pattern_no_match_on_python():
    """Regex does NOT match a clean Python expression."""
    assert _CPP_PATTERN.search("__l0 * 4 + __l1") is None


def test_cpp_pattern_no_match_on_python_slices():
    """Regex does NOT match Python slice syntax with colons."""
    assert _CPP_PATTERN.search("a[0:4]") is None


def test_cpp_pattern_no_match_on_subtraction():
    """``-`` followed by non-``>`` is NOT flagged."""
    assert _CPP_PATTERN.search("__l0 - 1") is None


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
