"""Tests for TileIfElseOpLibraryNode."""
import numpy as np
import pytest
import sympy as sp

import dace
from dace.libraries.cutile.nodes.if_else_op import (
    TileIfElseOpLibraryNode,
    _required_roles,
)
from dace.sdfg.validation import InvalidSDFGNodeError


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
        condition=sp.Symbol("_in0") > 0,
        true_op="+", true_constant2="1",
        false_op="*", false_constant2="2",
        num_inputs=1,
        input_roles={"_in0": ["true_rhs1", "false_rhs1"]},
    )
    assert node.condition == (sp.Symbol("_in0") > 0)
    assert "_in0" in node.in_connectors
    assert "_out" in node.out_connectors


def test_multiple_inputs():
    node = TileIfElseOpLibraryNode(
        "test",
        condition=sp.Symbol("_in0") >= sp.Symbol("_in1"),
        true_op="-",
        false_op="/",
        num_inputs=3,
        input_roles={
            "_in0": ["true_rhs1", "false_rhs1"],
            "_in1": ["true_rhs2"],
            "_in2": ["false_rhs2"],
        },
    )
    assert "_in0" in node.in_connectors
    assert "_in1" in node.in_connectors
    assert "_in2" in node.in_connectors


# ── required_roles tests ─────────────────────────────────────────────

def test_required_roles_all_array():
    node = TileIfElseOpLibraryNode(
        "t", condition=sp.Symbol("_in0") > 0,
        true_op="+", false_op="*", num_inputs=1,
        input_roles={},
    )
    roles = _required_roles(node)
    assert roles == {
        "true_rhs1", "true_rhs2",
        "false_rhs1", "false_rhs2",
    }


def test_required_roles_with_constants():
    node = TileIfElseOpLibraryNode(
        "t", condition=sp.Symbol("_in0") > 0,
        true_op="abs",  # unary – no rhs2
        false_op="+", false_constant2="5",
        num_inputs=1,
        input_roles={},
    )
    roles = _required_roles(node)
    # true_rhs2 not needed (unary),
    # false_rhs2 not needed (constant)
    assert roles == {"true_rhs1", "false_rhs1"}


# ── validation tests ─────────────────────────────────────────────────

def test_validate_missing_role():
    node = TileIfElseOpLibraryNode(
        "t", condition=sp.Symbol("_in0") > 0,
        true_op="+", true_constant2="1",
        false_op="*", false_constant2="2",
        num_inputs=1, input_roles={},  # missing true_rhs1 and false_rhs1
    )
    sdfg, state = _make_simple_sdfg(node)
    with pytest.raises(InvalidSDFGNodeError, match="missing"):
        node.validate(sdfg, state)


def test_validate_missing_condition():
    node = TileIfElseOpLibraryNode(
        "t", condition=None,
        true_op="+", true_constant2="1",
        false_op="*", false_constant2="2",
        num_inputs=1,
        input_roles={"_in0": ["true_rhs1", "false_rhs1"]},
    )
    sdfg, state = _make_simple_sdfg(node)
    with pytest.raises(InvalidSDFGNodeError, match="condition must be set"):
        node.validate(sdfg, state)


def test_validate_bad_condition_symbol():
    node = TileIfElseOpLibraryNode(
        "t",
        condition=sp.Symbol("unknown") > 0,
        true_op="+", true_constant2="1",
        false_op="+", false_constant2="1",
        num_inputs=1,
        input_roles={"_in0": ["true_rhs1", "false_rhs1"]},
    )
    sdfg, state = _make_simple_sdfg(node)
    with pytest.raises(InvalidSDFGNodeError, match="condition symbol"):
        node.validate(sdfg, state)


def test_validate_unknown_input_roles_key():
    node = TileIfElseOpLibraryNode(
        "t", condition=sp.Symbol("_in0") > 0,
        true_op="+", true_constant2="1",
        false_op="*", false_constant2="2",
        num_inputs=1,
        input_roles={"_in9": ["true_rhs1", "false_rhs1"]},
    )
    sdfg, state = _make_simple_sdfg(node)
    with pytest.raises(InvalidSDFGNodeError, match="unknown connector"):
        node.validate(sdfg, state)


def test_validate_duplicate_role_assignment():
    node = TileIfElseOpLibraryNode(
        "t", condition=sp.Symbol("_in0") > 0,
        true_op="+", false_op="*",
        num_inputs=2,
        input_roles={
            "_in0": ["true_rhs1", "false_rhs1"],
            "_in1": ["true_rhs1", "true_rhs2", "false_rhs2"],
        },
    )
    sdfg, state = _make_simple_sdfg(node)
    with pytest.raises(InvalidSDFGNodeError, match="assigned more than once"):
        node.validate(sdfg, state)


def test_validate_success():
    node = TileIfElseOpLibraryNode(
        "t", condition=sp.Symbol("_in0") > 0,
        true_op="+", true_constant2="1",
        false_op="*", false_constant2="2",
        num_inputs=1,
        input_roles={"_in0": ["true_rhs1", "false_rhs1"]},
    )
    sdfg, state = _make_simple_sdfg(node)
    node.validate(sdfg, state)  # should not raise


# ── expansion tests ──────────────────────────────────────────────────

def test_expansion_creates_sdfg():
    """Expansion should produce an SDFG (not a Tasklet)."""
    from dace.libraries.cutile.nodes.if_else_op import ExpandTileIfElseOpPure

    node = TileIfElseOpLibraryNode(
        "t", condition=sp.Symbol("_in0") > 0,
        true_op="+", true_constant2="1",
        false_op="*", false_constant2="2",
        num_inputs=1,
        input_roles={"_in0": ["true_rhs1", "false_rhs1"]},
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
        "t", condition=sp.Symbol("_in0") > sp.Symbol("_in1"),
        true_op="+",
        false_op="*",
        num_inputs=2,
        input_roles={
            "_in0": ["true_rhs1", "false_rhs1"],
            "_in1": ["true_rhs2", "false_rhs2"],
        },
    )
    sdfg, state = _make_simple_sdfg(node)
    result = ExpandTileIfElseOpPure.expansion(node, state, sdfg)
    assert isinstance(result, dace.SDFG)
    # Both input arrays should be non-transient in the inner SDFG
    assert "_in0" in result.arrays
    assert "_in1" in result.arrays
    assert result.arrays["_in0"].transient is False
    assert result.arrays["_in1"].transient is False


def test_expansion_missing_condition_raises():
    """Expansion should fail when no condition is provided."""
    from dace.libraries.cutile.nodes.if_else_op import ExpandTileIfElseOpPure

    node = TileIfElseOpLibraryNode(
        "t", condition=None,
        true_op="+", true_constant2="1",
        false_op="*", false_constant2="2",
        num_inputs=1,
        input_roles={"_in0": ["true_rhs1", "false_rhs1"]},
    )
    sdfg, state = _make_simple_sdfg(node)
    with pytest.raises(ValueError, match="condition is missing"):
        ExpandTileIfElseOpPure.expansion(node, state, sdfg)


# ── end-to-end compilation test ──────────────────────────────────────

def test_expand_and_compile():
    """The full SDFG should compile after library node expansion."""
    node = TileIfElseOpLibraryNode(
        "ie", condition=sp.Symbol("_in0") > 0,
        true_op="+", true_constant2="1",
        false_op="*", false_constant2="2",
        num_inputs=1,
        input_roles={"_in0": ["true_rhs1", "false_rhs1"]},
    )
    sdfg, state = _make_simple_sdfg(node, shape=(4,), dtype=dace.float64)
    sdfg.expand_library_nodes()
    sdfg.compile()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
