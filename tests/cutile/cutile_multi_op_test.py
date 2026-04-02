"""Tests for TileOpLibraryNode with multi-op (expr) support.

DaCe requires that tasklet connector names differ from SDFG array names.
The convention used here is:
  - SDFG arrays:  uppercase letters  (A, B, C, D, …)
  - Connector/symbol names:  lowercase with _ prefix  (_a, _b, _d, …)
    - Note: ``_out`` is reserved for the output connector and must NOT be used
    as an expression input symbol.
"""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

import dace
from dace.libraries.cutile.nodes.op import TileOpLibraryNode
from dace.sdfg.validation import InvalidSDFGNodeError


# ── helpers ──────────────────────────────────────────────────────────

def _make_sdfg(node, conn_to_array: dict, shape=(8,), dtype=dace.float64):
    """Build a minimal SDFG and return *(sdfg, state)*.

    *conn_to_array* maps each input connector name of *node* to the SDFG
    array name it should read from, e.g. ``{"_a": "A", "_b": "B"}``.
    An ``"Out"`` array is always created and wired to output connector ``_out``.
    """
    sdfg = dace.SDFG("test_multi_op")
    state = sdfg.add_state("s")

    for arr_name in conn_to_array.values():
        sdfg.add_array(arr_name, shape=shape, dtype=dtype)
    sdfg.add_array("Out", shape=shape, dtype=dtype)

    state.add_node(node)

    for conn_name, arr_name in conn_to_array.items():
        r = state.add_read(arr_name)
        state.add_edge(r, None, node, conn_name,
                       dace.Memlet.from_array(arr_name, sdfg.arrays[arr_name]))

    w = state.add_write("Out")
    state.add_edge(node, "_out", w, None,
                   dace.Memlet.from_array("Out", sdfg.arrays["Out"]))
    return sdfg, state


def _compile_and_run(node, conn_to_array: dict, arrays_np: dict, shape=(8,)):
    """Expand, compile, and execute the SDFG; return the ``Out`` array."""
    sdfg, _ = _make_sdfg(node, conn_to_array, shape=shape)
    sdfg.expand_library_nodes()
    compiled = sdfg.compile()
    out = np.zeros(shape, dtype=np.float64)
    compiled(**arrays_np, Out=out)
    return out


# ── construction & connector inference ───────────────────────────────

def test_expr_connectors_three_inputs():
    _a, _b, _d = sp.symbols("_a _b _d")
    node = TileOpLibraryNode("t", expr=_a * (_b + _d))
    assert set(node.in_connectors) == {"_a", "_b", "_d"}
    assert "_out" in node.out_connectors


def test_expr_connectors_with_unary():
    _a, _d = sp.symbols("_a _d")
    node = TileOpLibraryNode("t", expr=_a - sp.sin(_d))
    assert set(node.in_connectors) == {"_a", "_d"}


def test_expr_connectors_with_literal():
    """Numeric literals in the expression do not become connectors."""
    _a = sp.Symbol("_a")
    node = TileOpLibraryNode("t", expr=_a * 2 + 1)
    assert set(node.in_connectors) == {"_a"}


def test_expr_single_input_op_field_ignored():
    """When expr is set, op/constant1/constant2 are ignored for connectors."""
    _a, _b = sp.symbols("_a _b")
    node = TileOpLibraryNode("t", expr=_a + _b, op="*", constant1="99")
    assert set(node.in_connectors) == {"_a", "_b"}


# ── validation ────────────────────────────────────────────────────────

def test_validate_success():
    _a, _b, _d = sp.symbols("_a _b _d")
    node = TileOpLibraryNode("t", expr=_a * (_b + _d))
    sdfg, state = _make_sdfg(node, {"_a": "A", "_b": "B", "_d": "D"})
    node.validate(sdfg, state)  # must not raise


def test_validate_reserved_output_connector_raises():
    """Using '_out' as an expr symbol should be rejected since it is the output connector."""
    _a = sp.Symbol("_a")
    _out_sym = sp.Symbol("_out")
    node = TileOpLibraryNode("t", expr=_a + _out_sym)
    sdfg, state = _make_sdfg(node, {"_a": "A", "_out": "C"})
    with pytest.raises(InvalidSDFGNodeError, match="reserved for the output connector"):
        node.validate(sdfg, state)


def test_validate_missing_connector_raises():
    _a, _b = sp.symbols("_a _b")
    node = TileOpLibraryNode("t", expr=_a + _b)
    # Wire only _a, leave _b unconnected
    sdfg = dace.SDFG("bad")
    state = sdfg.add_state("s")
    sdfg.add_array("A", shape=(4,), dtype=dace.float64)
    sdfg.add_array("Out", shape=(4,), dtype=dace.float64)
    state.add_node(node)
    r = state.add_read("A")
    state.add_edge(r, None, node, "_a",
                   dace.Memlet.from_array("A", sdfg.arrays["A"]))
    w = state.add_write("Out")
    state.add_edge(node, "_out", w, None,
                   dace.Memlet.from_array("Out", sdfg.arrays["Out"]))
    with pytest.raises(InvalidSDFGNodeError, match="must be connected in multi-op mode"):
        node.validate(sdfg, state)


# ── expansion (tasklet production) ───────────────────────────────────

def test_expansion_produces_tasklet():
    from dace.libraries.cutile.nodes.op import ExpandTileOpPure
    _a, _b = sp.symbols("_a _b")
    node = TileOpLibraryNode("t", expr=_a * _b)
    sdfg, state = _make_sdfg(node, {"_a": "A", "_b": "B"})
    result = ExpandTileOpPure.expansion(node, state, sdfg)
    assert isinstance(result, dace.nodes.Tasklet)
    assert "_a" in result.in_connectors
    assert "_b" in result.in_connectors
    assert "_out" in result.out_connectors


def test_expansion_three_inputs_produces_tasklet():
    from dace.libraries.cutile.nodes.op import ExpandTileOpPure
    _a, _b, _d = sp.symbols("_a _b _d")
    node = TileOpLibraryNode("t", expr=_a * (_b + _d))
    sdfg, state = _make_sdfg(node, {"_a": "A", "_b": "B", "_d": "D"})
    result = ExpandTileOpPure.expansion(node, state, sdfg)
    assert isinstance(result, dace.nodes.Tasklet)
    assert {"_a", "_b", "_d"} == set(result.in_connectors)


# ── end-to-end compile & run ──────────────────────────────────────────

@pytest.mark.parametrize("n", [1, 4, 16])
def test_binary_multiply(n):
    _a, _b = sp.symbols("_a _b")
    node = TileOpLibraryNode("t", expr=_a * _b)
    a = np.arange(n, dtype=np.float64)
    b = np.arange(n, dtype=np.float64) + 1.0
    out = _compile_and_run(node, {"_a": "A", "_b": "B"},
                           {"A": a, "B": b}, shape=(n,))
    np.testing.assert_allclose(out, a * b)


@pytest.mark.parametrize("n", [4, 16])
def test_three_input_fma(n):
    """out = a * (b + d)"""
    _a, _b, _d = sp.symbols("_a _b _d")
    node = TileOpLibraryNode("t", expr=_a * (_b + _d))
    a = np.ones(n, dtype=np.float64) * 2
    b = np.arange(n, dtype=np.float64)
    d = np.arange(n, dtype=np.float64) * 0.5
    out = _compile_and_run(node, {"_a": "A", "_b": "B", "_d": "D"},
                           {"A": a, "B": b, "D": d}, shape=(n,))
    np.testing.assert_allclose(out, a * (b + d))


@pytest.mark.parametrize("n", [4, 16])
def test_four_input_expression(n):
    """out = a * (b + d) - sin(e)"""
    _a, _b, _d, _e = sp.symbols("_a _b _d _e")
    node = TileOpLibraryNode("t", expr=_a * (_b + _d) - sp.sin(_e))
    a = np.ones(n, dtype=np.float64) * 3
    b = np.arange(n, dtype=np.float64)
    d = np.ones(n, dtype=np.float64)
    e = np.linspace(0, np.pi, n).copy()
    out = _compile_and_run(
        node,
        {"_a": "A", "_b": "B", "_d": "D", "_e": "E"},
        {"A": a, "B": b, "D": d, "E": e},
        shape=(n,),
    )
    np.testing.assert_allclose(out, a * (b + d) - np.sin(e), rtol=1e-10)


@pytest.mark.parametrize("n", [4, 16])
def test_literal_in_expression(n):
    """out = 2 * a + 1"""
    _a = sp.Symbol("_a")
    node = TileOpLibraryNode("t", expr=2 * _a + 1)
    a = np.arange(n, dtype=np.float64)
    out = _compile_and_run(node, {"_a": "A"}, {"A": a}, shape=(n,))
    np.testing.assert_allclose(out, 2 * a + 1)


@pytest.mark.parametrize("n", [4, 16])
def test_unary_chain(n):
    """out = sin(abs(a))"""
    _a = sp.Symbol("_a")
    node = TileOpLibraryNode("t", expr=sp.sin(sp.Abs(_a)))
    a = np.linspace(-np.pi, np.pi, n).copy()
    out = _compile_and_run(node, {"_a": "A"}, {"A": a}, shape=(n,))
    np.testing.assert_allclose(out, np.sin(np.abs(a)), rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
