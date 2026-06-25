# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Structural and integration tests for the ``cutile`` expansion of
:class:`TileBinop`.

These tests verify the robustness improvements to
``ExpandTileBinopCutile.expansion()``:

- A1: ``validate()`` is called at expansion time
- A2: ``**`` on integer operands wraps in float cast
- A3: masked fill value is dtype-aware (``False`` for bool, ``0`` for numeric)
- A4: symbol operands get explicit ``ct.astype`` casts
- A5: ``/`` on integer operands wraps in float cast

Structural tests inspect the generated tasklet code without requiring a GPU.
Integration tests (marked ``@pytest.mark.gpu``) compile and run on GPU.
"""
import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileBinop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_cutile_sdfg(widths, op, a_dtype, b_dtype, c_dtype,
                        has_mask=False,
                        kind_a="Tile", kind_b="Tile",
                        expr_a=None, expr_b=None,
                        free_symbols=None,
                        name_suffix=""):
    """Build a minimal single-state SDFG wiring operands through a
    ``TileBinop`` node with ``implementation="cutile"``.

    Returns ``(sdfg, state, node)`` *before* expansion so the caller can
    inspect or expand.
    """
    tag = f"cutile_binop_{op.replace('*', 'star').replace('/', 'div').replace('<', 'lt').replace('>', 'gt').replace('=', 'eq').replace('!', 'ne').replace('&', 'and').replace('|', 'or').replace('^', 'xor').replace('%', 'mod').replace('+', 'add').replace('-', 'sub')}"
    tag += f"_{'x'.join(str(w) for w in widths)}"
    if name_suffix:
        tag += f"_{name_suffix}"
    sdfg = dace.SDFG(tag)
    if free_symbols:
        for sym, dt in free_symbols.items():
            sdfg.add_symbol(sym, dt)

    full = ",".join(f"0:{w}" for w in widths)
    state = sdfg.add_state("main")

    # Create the TileBinop node
    node = TileBinop(name="tb", widths=widths, op=op, has_mask=has_mask,
                     kind_a=kind_a, kind_b=kind_b,
                     expr_a=expr_a, expr_b=expr_b)
    node.implementation = "cutile"
    state.add_node(node)

    # Wire inputs
    if kind_a in ("Tile", "Scalar"):
        if kind_a == "Scalar":
            sdfg.add_scalar("A", a_dtype, transient=False)
            a = state.add_access("A")
            state.add_edge(a, None, node, "_a", dace.Memlet("A"))
        else:
            sdfg.add_array("A", widths, a_dtype, transient=False)
            a = state.add_access("A")
            state.add_edge(a, None, node, "_a", dace.Memlet(f"A[{full}]"))

    if kind_b in ("Tile", "Scalar"):
        if kind_b == "Scalar":
            sdfg.add_scalar("B", b_dtype, transient=False)
            b = state.add_access("B")
            state.add_edge(b, None, node, "_b", dace.Memlet("B"))
        else:
            sdfg.add_array("B", widths, b_dtype, transient=False)
            b = state.add_access("B")
            state.add_edge(b, None, node, "_b", dace.Memlet(f"B[{full}]"))

    # Output
    sdfg.add_array("C", widths, c_dtype, transient=False)
    c = state.add_access("C")
    state.add_edge(node, "_c", c, None, dace.Memlet(f"C[{full}]"))

    # Mask
    if has_mask:
        sdfg.add_array("M", widths, dace.bool_, transient=False)
        m = state.add_access("M")
        state.add_edge(m, None, node, "_mask", dace.Memlet(f"M[{full}]"))

    return sdfg, state, node


def _expand_and_get_code(sdfg):
    """Expand library nodes and return the tasklet code string."""
    sdfg.expand_library_nodes()
    sdfg.validate()
    # Find the tasklet in the expanded SDFG
    for s in sdfg.states():
        for n in s.nodes():
            if isinstance(n, dace.sdfg.nodes.Tasklet):
                return n.code.as_string
    raise RuntimeError("No tasklet found after expansion")


# ---------------------------------------------------------------------------
# A1: validate() is called
# ---------------------------------------------------------------------------

class TestA1Validate:
    """``ExpandTileBinopCutile.expansion()`` calls ``node.validate()``."""

    def test_cutile_expansion_calls_validate_rejects_narrowing(self):
        """A narrowing conversion (float64 -> int32) is caught at expansion
        time because ``validate()`` runs first."""
        sdfg, state, node = _build_cutile_sdfg(
            widths=(8,), op="+",
            a_dtype=dace.float64, b_dtype=dace.float64, c_dtype=dace.int32,
            name_suffix="narrow",
        )
        with pytest.raises(NotImplementedError, match="narrowing"):
            sdfg.expand_library_nodes()

    def test_cutile_expansion_calls_validate_unconnected_output(self):
        """Validate catches unconnected ``_c`` output."""
        sdfg = dace.SDFG("cutile_validate_no_c")
        sdfg.add_array("A", [8], dace.float32, transient=False)
        sdfg.add_array("B", [8], dace.float32, transient=False)
        state = sdfg.add_state("main")
        # Manually build a node but don't connect _c
        node = TileBinop(name="tb_no_c", widths=(8,), op="+")
        node.implementation = "cutile"
        state.add_node(node)
        a = state.add_access("A")
        b = state.add_access("B")
        state.add_edge(a, None, node, "_a", dace.Memlet("A[0:8]"))
        state.add_edge(b, None, node, "_b", dace.Memlet("B[0:8]"))
        with pytest.raises(ValueError, match="'_c' not connected"):
            sdfg.expand_library_nodes()


# ---------------------------------------------------------------------------
# A2: ** on integer operands
# ---------------------------------------------------------------------------

class TestA2PowerInteger:
    """``**`` with integer operands wraps in ``ct.astype(..., ct.float64)``
    and casts back if the output is integer."""

    def test_pow_int_operands_float_cast_in_code(self):
        """Both int32 operands -> code contains float64 casts."""
        sdfg, _, _ = _build_cutile_sdfg(
            widths=(8,), op="**",
            a_dtype=dace.int32, b_dtype=dace.int32, c_dtype=dace.int32,
            name_suffix="pow_int",
        )
        code = _expand_and_get_code(sdfg)
        # Both operands should be float-cast
        assert "ct.astype(_a, ct.float64)" in code
        assert "ct.astype(_b, ct.float64)" in code
        # Output is int32 -> cast back
        assert "ct.astype(" in code
        assert "ct.int32" in code

    def test_pow_int_lhs_float_rhs_casts_only_lhs(self):
        """int32 ** float64: only LHS is cast to float64."""
        sdfg, _, _ = _build_cutile_sdfg(
            widths=(8,), op="**",
            a_dtype=dace.int32, b_dtype=dace.float64, c_dtype=dace.float64,
            name_suffix="pow_int_lhs",
        )
        code = _expand_and_get_code(sdfg)
        assert "ct.astype(_a, ct.float64)" in code
        # RHS is already float64, should not be additionally wrapped
        # (the raw connector _b is used)
        assert "ct.astype(_b, ct.float64)" not in code

    def test_pow_float_operands_no_cast(self):
        """float64 ** float64: no float-cast wrapping needed."""
        sdfg, _, _ = _build_cutile_sdfg(
            widths=(8,), op="**",
            a_dtype=dace.float64, b_dtype=dace.float64, c_dtype=dace.float64,
            name_suffix="pow_float",
        )
        code = _expand_and_get_code(sdfg)
        assert "ct.float64" not in code

    def test_pow_int_output_float_has_castback(self):
        """int64 ** int64 with int64 output: cast-back to ct.int64."""
        sdfg, _, _ = _build_cutile_sdfg(
            widths=(4, 8), op="**",
            a_dtype=dace.int64, b_dtype=dace.int64, c_dtype=dace.int64,
            name_suffix="pow_int64_castback",
        )
        code = _expand_and_get_code(sdfg)
        assert "ct.int64" in code

    def test_pow_int_to_float_output_no_castback(self):
        """int32 ** int32 with float64 output: no cast-back needed."""
        sdfg, _, _ = _build_cutile_sdfg(
            widths=(8,), op="**",
            a_dtype=dace.int32, b_dtype=dace.int32, c_dtype=dace.float64,
            name_suffix="pow_int_float_out",
        )
        code = _expand_and_get_code(sdfg)
        # Both operands float-cast
        assert "ct.astype(_a, ct.float64)" in code
        assert "ct.astype(_b, ct.float64)" in code
        # No cast-back because output is float64
        # Count ct.int32 occurrences - should be zero
        assert "ct.int32" not in code


# ---------------------------------------------------------------------------
# A3: dtype-aware masked fill
# ---------------------------------------------------------------------------

class TestA3MaskedFill:
    """Masked ``ct.where`` uses dtype-aware fill: ``False`` for bool,
    ``0`` for numeric."""

    def test_masked_float_fill_is_zero(self):
        """Float output with mask -> fill value is ``0``."""
        sdfg, _, _ = _build_cutile_sdfg(
            widths=(8,), op="+",
            a_dtype=dace.float64, b_dtype=dace.float64, c_dtype=dace.float64,
            has_mask=True, name_suffix="mask_float",
        )
        code = _expand_and_get_code(sdfg)
        assert "ct.where(_mask," in code
        assert ", 0)" in code
        assert ", False)" not in code

    def test_masked_int_fill_is_zero(self):
        """Integer output with mask -> fill value is ``0``."""
        sdfg, _, _ = _build_cutile_sdfg(
            widths=(8,), op="+",
            a_dtype=dace.int32, b_dtype=dace.int32, c_dtype=dace.int32,
            has_mask=True, name_suffix="mask_int",
        )
        code = _expand_and_get_code(sdfg)
        assert "ct.where(_mask," in code
        assert ", 0)" in code

    def test_masked_bool_fill_is_false(self):
        """Bool output with mask -> fill value is ``False``."""
        sdfg, _, _ = _build_cutile_sdfg(
            widths=(8,), op="==",
            a_dtype=dace.float64, b_dtype=dace.float64, c_dtype=dace.bool_,
            has_mask=True, name_suffix="mask_bool",
        )
        code = _expand_and_get_code(sdfg)
        assert "ct.where(_mask," in code
        assert ", False)" in code

    def test_unmasked_no_ct_where(self):
        """Without mask, no ``ct.where`` is emitted."""
        sdfg, _, _ = _build_cutile_sdfg(
            widths=(8,), op="+",
            a_dtype=dace.float64, b_dtype=dace.float64, c_dtype=dace.float64,
            has_mask=False, name_suffix="no_mask",
        )
        code = _expand_and_get_code(sdfg)
        assert "ct.where" not in code


# ---------------------------------------------------------------------------
# A4: symbol operand ct.astype cast
# ---------------------------------------------------------------------------

class TestA4SymbolCast:
    """Symbol operands get an explicit ``ct.astype`` cast to the operand
    dtype."""

    def test_symbol_rhs_gets_astype_cast(self):
        """Tile + Symbol: symbol is wrapped in ``ct.astype``."""
        sdfg, _, _ = _build_cutile_sdfg(
            widths=(8,), op="+",
            a_dtype=dace.float64, b_dtype=dace.float64, c_dtype=dace.float64,
            kind_b="Symbol", expr_b="alpha",
            free_symbols={"alpha": dace.float64},
            name_suffix="sym_rhs",
        )
        code = _expand_and_get_code(sdfg)
        # Symbol should be wrapped in ct.astype with the operand dtype
        assert "ct.astype(alpha, ct.float64)" in code

    def test_symbol_lhs_gets_astype_cast(self):
        """Symbol + Tile: symbol on LHS is wrapped in ``ct.astype``."""
        sdfg, _, _ = _build_cutile_sdfg(
            widths=(8,), op="*",
            a_dtype=dace.float64, b_dtype=dace.float64, c_dtype=dace.float64,
            kind_a="Symbol", expr_a="beta",
            free_symbols={"beta": dace.float64},
            name_suffix="sym_lhs",
        )
        code = _expand_and_get_code(sdfg)
        assert "ct.astype(beta, ct.float64)" in code

    def test_both_symbols_cast_to_output_dtype(self):
        """Symbol + Symbol: both cast to the output dtype (no Tile/Scalar
        operand to derive from)."""
        sdfg, _, _ = _build_cutile_sdfg(
            widths=(8,), op="+",
            a_dtype=dace.float64, b_dtype=dace.float64, c_dtype=dace.float64,
            kind_a="Symbol", kind_b="Symbol",
            expr_a="alpha", expr_b="beta",
            free_symbols={"alpha": dace.float64, "beta": dace.float64},
            name_suffix="sym_sym",
        )
        code = _expand_and_get_code(sdfg)
        assert "ct.astype(alpha, ct.float64)" in code
        assert "ct.astype(beta, ct.float64)" in code

    def test_symbol_cast_uses_tile_dtype_for_comparison(self):
        """For comparison ops, symbol is cast to the Tile operand's dtype,
        not the bool output dtype."""
        sdfg, _, _ = _build_cutile_sdfg(
            widths=(8,), op=">",
            a_dtype=dace.float64, b_dtype=dace.float64, c_dtype=dace.bool_,
            kind_b="Symbol", expr_b="threshold",
            free_symbols={"threshold": dace.float64},
            name_suffix="sym_cmp",
        )
        code = _expand_and_get_code(sdfg)
        # Cast to float64 (the tile operand's type), NOT bool_
        assert "ct.astype(threshold, ct.float64)" in code
        assert "ct.bool_" not in code or "ct.astype(threshold, ct.bool_)" not in code

    def test_symbol_cast_int32_operand(self):
        """Symbol paired with int32 Tile -> cast to ct.int32."""
        sdfg, _, _ = _build_cutile_sdfg(
            widths=(8,), op="+",
            a_dtype=dace.int32, b_dtype=dace.int32, c_dtype=dace.int32,
            kind_b="Symbol", expr_b="offset",
            free_symbols={"offset": dace.int32},
            name_suffix="sym_int32",
        )
        code = _expand_and_get_code(sdfg)
        assert "ct.astype(offset, ct.int32)" in code


# ---------------------------------------------------------------------------
# A5: / on integer operands
# ---------------------------------------------------------------------------

class TestA5DivisionInteger:
    """``/`` with integer operands wraps in ``ct.astype(..., ct.float64)``."""

    def test_div_int_operands_float_cast(self):
        """Both int32 operands -> code contains float64 casts."""
        sdfg, _, _ = _build_cutile_sdfg(
            widths=(8,), op="/",
            a_dtype=dace.int32, b_dtype=dace.int32, c_dtype=dace.float64,
            name_suffix="div_int",
        )
        code = _expand_and_get_code(sdfg)
        assert "ct.astype(_a, ct.float64)" in code
        assert "ct.astype(_b, ct.float64)" in code

    def test_div_float_operands_no_cast(self):
        """float64 / float64: no additional float-cast wrapping."""
        sdfg, _, _ = _build_cutile_sdfg(
            widths=(8,), op="/",
            a_dtype=dace.float64, b_dtype=dace.float64, c_dtype=dace.float64,
            name_suffix="div_float",
        )
        code = _expand_and_get_code(sdfg)
        # No ct.astype wrapping for float operands in division
        assert "ct.astype(_a" not in code
        assert "ct.astype(_b" not in code

    def test_div_int_output_casts_back(self):
        """int32 / int32 -> int32 output: result is cast back to ct.int32."""
        sdfg, _, _ = _build_cutile_sdfg(
            widths=(8,), op="/",
            a_dtype=dace.int32, b_dtype=dace.int32, c_dtype=dace.int32,
            name_suffix="div_int_out",
        )
        code = _expand_and_get_code(sdfg)
        # Float-cast operands
        assert "ct.astype(_a, ct.float64)" in code
        assert "ct.astype(_b, ct.float64)" in code
        # Cast-back to int
        assert "ct.int32" in code

    def test_div_mixed_int_float_casts_only_int(self):
        """int32 / float64: only int side gets float-cast."""
        sdfg, _, _ = _build_cutile_sdfg(
            widths=(8,), op="/",
            a_dtype=dace.int32, b_dtype=dace.float64, c_dtype=dace.float64,
            name_suffix="div_mixed",
        )
        code = _expand_and_get_code(sdfg)
        assert "ct.astype(_a, ct.float64)" in code
        assert "ct.astype(_b, ct.float64)" not in code


# ---------------------------------------------------------------------------
# Combined / edge-case tests
# ---------------------------------------------------------------------------

class TestCombined:
    """Edge cases that exercise multiple robustness features together."""

    def test_pow_symbol_int_rhs(self):
        """Tile(int32) ** Symbol(int): both sides get float64 cast, plus
        symbol gets ct.astype."""
        sdfg, _, _ = _build_cutile_sdfg(
            widths=(8,), op="**",
            a_dtype=dace.int32, b_dtype=dace.int32, c_dtype=dace.int32,
            kind_b="Symbol", expr_b="exp",
            free_symbols={"exp": dace.int32},
            name_suffix="pow_sym_int",
        )
        code = _expand_and_get_code(sdfg)
        # LHS tile: float-cast
        assert "ct.astype(_a, ct.float64)" in code
        # RHS symbol: first gets ct.astype to operand dtype, then float-cast
        assert "ct.float64" in code
        # Output is int32: cast-back
        assert "ct.int32" in code

    def test_basic_add_float64_code_shape(self):
        """Smoke test: float64 + float64 produces clean code with no
        unnecessary casts."""
        sdfg, _, _ = _build_cutile_sdfg(
            widths=(8,), op="+",
            a_dtype=dace.float64, b_dtype=dace.float64, c_dtype=dace.float64,
            name_suffix="smoke_add",
        )
        code = _expand_and_get_code(sdfg)
        assert "_c = (_a + _b)" == code

    def test_masked_pow_int_all_features(self):
        """Masked int ** int exercises A1 + A2 + A3 together."""
        sdfg, _, _ = _build_cutile_sdfg(
            widths=(8,), op="**",
            a_dtype=dace.int32, b_dtype=dace.int32, c_dtype=dace.int32,
            has_mask=True, name_suffix="pow_int_masked",
        )
        code = _expand_and_get_code(sdfg)
        # A2: float-cast
        assert "ct.astype(_a, ct.float64)" in code
        assert "ct.astype(_b, ct.float64)" in code
        # A2: cast-back to int
        assert "ct.int32" in code
        # A3: fill value is 0 (not False)
        assert ", 0)" in code
        assert "ct.where(_mask," in code

    def test_multidim_k2_basic(self):
        """K=2 tiles produce valid code."""
        sdfg, _, _ = _build_cutile_sdfg(
            widths=(4, 8), op="+",
            a_dtype=dace.float64, b_dtype=dace.float64, c_dtype=dace.float64,
            name_suffix="k2",
        )
        code = _expand_and_get_code(sdfg)
        assert "_c = (_a + _b)" == code

    def test_min_max_no_float_cast(self):
        """min/max on float operands should not trigger float-cast."""
        for op in ("min", "max"):
            sdfg, _, _ = _build_cutile_sdfg(
                widths=(8,), op=op,
                a_dtype=dace.float64, b_dtype=dace.float64,
                c_dtype=dace.float64,
                name_suffix=f"{op}_float",
            )
            code = _expand_and_get_code(sdfg)
            ct_fn = "ct.minimum" if op == "min" else "ct.maximum"
            assert ct_fn in code
            assert "ct.astype" not in code
