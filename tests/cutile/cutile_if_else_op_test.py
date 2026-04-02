"""Tests for TileIfElseOpLibraryNode."""
import numpy as np
import pytest
import sympy as sp

import dace
from dace.libraries.cutile.nodes.if_else_op import (
    TileIfElseOpLibraryNode,
)
from dace.sdfg.validation import InvalidSDFGNodeError

# Shorthand symbols used across tests
_in0 = sp.Symbol("_in0")
_in1 = sp.Symbol("_in1")
_in2 = sp.Symbol("_in2")


# ── helpers ──────────────────────────────────────────────────────────

def _make_simple_sdfg(node, shape=(4,), dtype=dace.float64):
    """Build a minimal SDFG with *node* wired up and return (sdfg, state)."""
    sdfg = dace.SDFG("test_if_else")
    state = sdfg.add_state("s")

    # Create arrays for each input connector
    for conn in sorted(node.in_connectors):
        sdfg.add_array(conn, shape=shape, dtype=dtype)
    # Output
    sdfg.add_array("out", shape=shape, dtype=dtype)

    state.add_node(node)

    for conn in sorted(node.in_connectors):
        r = state.add_read(conn)
        state.add_edge(r, None, node, conn,
                        dace.Memlet.from_array(conn, sdfg.arrays[conn]))

    w = state.add_write("out")
    state.add_edge(node, "_out", w, None,
                    dace.Memlet.from_array("out", sdfg.arrays["out"]))
    return sdfg, state


# ── construction tests ───────────────────────────────────────────────

def test_basic_construction():
    node = TileIfElseOpLibraryNode(
        "test",
        condition=_in0 > 0,
        true_expr=_in0 + 1,
        false_expr=_in0 * 2,
    )
    assert node.condition == (_in0 > 0)
    assert node.true_expr == _in0 + 1
    assert node.false_expr == _in0 * 2
    assert "_in0" in node.in_connectors
    assert "_out" in node.out_connectors


def test_multiple_inputs():
    node = TileIfElseOpLibraryNode(
        "test",
        condition=_in0 >= _in1,
        true_expr=_in0 - _in2,
        false_expr=_in1 / _in2,
    )
    assert "_in0" in node.in_connectors
    assert "_in1" in node.in_connectors
    assert "_in2" in node.in_connectors


def test_connectors_derived_from_all_expressions():
    """Connectors are the union of free symbols across all 3 expressions."""
    node = TileIfElseOpLibraryNode(
        "test",
        condition=_in0 > 0,          # only _in0
        true_expr=_in1 + 1,          # only _in1
        false_expr=_in2 * 3,         # only _in2
    )
    assert set(node.in_connectors) == {"_in0", "_in1", "_in2"}


# ── validation tests ─────────────────────────────────────────────────

def test_validate_missing_condition():
    node = TileIfElseOpLibraryNode(
        "t", condition=None,
        true_expr=_in0 + 1,
        false_expr=_in0 * 2,
        num_inputs=1,
    )
    sdfg, state = _make_simple_sdfg(node)
    with pytest.raises(InvalidSDFGNodeError, match="condition must be set"):
        node.validate(sdfg, state)


def test_validate_missing_true_expr():
    node = TileIfElseOpLibraryNode(
        "t", condition=_in0 > 0,
        true_expr=None,
        false_expr=_in0 * 2,
        num_inputs=1,
    )
    sdfg, state = _make_simple_sdfg(node)
    with pytest.raises(InvalidSDFGNodeError, match="true_expr must be set"):
        node.validate(sdfg, state)


def test_validate_missing_false_expr():
    node = TileIfElseOpLibraryNode(
        "t", condition=_in0 > 0,
        true_expr=_in0 + 1,
        false_expr=None,
        num_inputs=1,
    )
    sdfg, state = _make_simple_sdfg(node)
    with pytest.raises(InvalidSDFGNodeError, match="false_expr must be set"):
        node.validate(sdfg, state)


def test_validate_bad_condition_symbol():
    """Condition references a symbol not in the connectors."""
    node = TileIfElseOpLibraryNode(
        "t",
        condition=sp.Symbol("unknown") > 0,
        true_expr=_in0 + 1,
        false_expr=_in0 * 2,
        num_inputs=1,
    )
    sdfg, state = _make_simple_sdfg(node)
    with pytest.raises(InvalidSDFGNodeError, match="condition symbol"):
        node.validate(sdfg, state)


def test_validate_bad_true_expr_symbol():
    """true_expr references a symbol not in the connectors."""
    node = TileIfElseOpLibraryNode(
        "t",
        condition=_in0 > 0,
        true_expr=sp.Symbol("unknown") + 1,
        false_expr=_in0 * 2,
        num_inputs=1,
    )
    sdfg, state = _make_simple_sdfg(node)
    with pytest.raises(InvalidSDFGNodeError, match="true_expr symbol"):
        node.validate(sdfg, state)


def test_validate_bad_false_expr_symbol():
    """false_expr references a symbol not in the connectors."""
    node = TileIfElseOpLibraryNode(
        "t",
        condition=_in0 > 0,
        true_expr=_in0 + 1,
        false_expr=sp.Symbol("unknown") * 2,
        num_inputs=1,
    )
    sdfg, state = _make_simple_sdfg(node)
    with pytest.raises(InvalidSDFGNodeError, match="false_expr symbol"):
        node.validate(sdfg, state)


def test_validate_connector_not_connected():
    """An input connector that is not wired should fail validation."""
    node = TileIfElseOpLibraryNode(
        "t",
        condition=_in0 > 0,
        true_expr=_in0 + 1,
        false_expr=_in0 * 2,
    )
    sdfg = dace.SDFG("test_if_else")
    state = sdfg.add_state("s")
    sdfg.add_array("_in0", shape=(4,), dtype=dace.float64)
    sdfg.add_array("out", shape=(4,), dtype=dace.float64)
    state.add_node(node)
    # Wire only the output — leave _in0 disconnected
    w = state.add_write("out")
    state.add_edge(node, "_out", w, None,
                    dace.Memlet.from_array("out", sdfg.arrays["out"]))
    with pytest.raises(InvalidSDFGNodeError, match="must be connected"):
        node.validate(sdfg, state)


def test_validate_success():
    node = TileIfElseOpLibraryNode(
        "t",
        condition=_in0 > 0,
        true_expr=_in0 + 1,
        false_expr=_in0 * 2,
    )
    sdfg, state = _make_simple_sdfg(node)
    node.validate(sdfg, state)  # should not raise


def test_validate_constant_only_expr():
    """An expression with no free symbols (e.g. sp.Integer(5)) should validate."""
    node = TileIfElseOpLibraryNode(
        "t",
        condition=_in0 > 0,
        true_expr=sp.Integer(5),
        false_expr=_in0 * 2,
    )
    sdfg, state = _make_simple_sdfg(node)
    node.validate(sdfg, state)  # should not raise


# ── expansion tests ──────────────────────────────────────────────────

def test_expansion_creates_sdfg():
    """Expansion should produce an SDFG (not a Tasklet)."""
    from dace.libraries.cutile.nodes.if_else_op import ExpandTileIfElseOpPure

    node = TileIfElseOpLibraryNode(
        "t",
        condition=_in0 > 0,
        true_expr=_in0 + 1,
        false_expr=_in0 * 2,
    )
    sdfg, state = _make_simple_sdfg(node)
    result = ExpandTileIfElseOpPure.expansion(node, state, sdfg)
    assert isinstance(result, dace.SDFG)
    # Inner SDFG should have the transient arrays
    assert "cond_tile" in result.arrays
    assert "true_tile" in result.arrays
    assert "false_tile" in result.arrays
    assert result.arrays["cond_tile"].transient is True


def test_expansion_two_inputs():
    """Expansion with two input arrays."""
    from dace.libraries.cutile.nodes.if_else_op import ExpandTileIfElseOpPure

    node = TileIfElseOpLibraryNode(
        "t",
        condition=_in0 > _in1,
        true_expr=_in0 + _in1,
        false_expr=_in0 * _in1,
    )
    sdfg, state = _make_simple_sdfg(node)
    result = ExpandTileIfElseOpPure.expansion(node, state, sdfg)
    assert isinstance(result, dace.SDFG)
    # Both input arrays should be non-transient in the inner SDFG
    assert "_in0" in result.arrays
    assert "_in1" in result.arrays
    assert result.arrays["_in0"].transient is False
    assert result.arrays["_in1"].transient is False


# ── end-to-end compilation test ──────────────────────────────────────

def test_expand_and_compile():
    """The full SDFG should compile after library node expansion."""
    node = TileIfElseOpLibraryNode(
        "ie",
        condition=_in0 > 0,
        true_expr=_in0 + 1,
        false_expr=_in0 * 2,
    )
    sdfg, state = _make_simple_sdfg(node, shape=(4,), dtype=dace.float64)
    sdfg.expand_library_nodes()
    sdfg.compile()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
