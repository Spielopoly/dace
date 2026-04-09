"""Regression tests for expr support in masked cuTile op nodes."""


import numpy as np
import pytest
import sympy as sp

import dace
from dace.libraries.cutile.nodes.op_runtime_map import TileRuntimeMaskedOpLibraryNode
from dace.libraries.cutile.nodes.op_symbolic_mask import TileSymbolicMaskedOpLibraryNode
from dace.sdfg.validation import InvalidSDFGNodeError


def _build_runtime_expr_sdfg(node, *, shape=(8,), with_c_in=False):
    sdfg = dace.SDFG("test_runtime_masked_expr")
    state = sdfg.add_state("s")

    sdfg.add_array("A", shape=shape, dtype=dace.float64)
    sdfg.add_array("B", shape=shape, dtype=dace.float64)
    sdfg.add_array("M", shape=shape, dtype=dace.bool)
    if with_c_in:
        sdfg.add_array("Cin", shape=shape, dtype=dace.float64)
    sdfg.add_array("Out", shape=shape, dtype=dace.float64)

    state.add_node(node)
    state.add_edge(state.add_read("A"), None, node, "_a",
                   dace.Memlet.from_array("A", sdfg.arrays["A"]))
    state.add_edge(state.add_read("B"), None, node, "_b",
                   dace.Memlet.from_array("B", sdfg.arrays["B"]))
    state.add_edge(state.add_read("M"), None, node, "_m",
                   dace.Memlet.from_array("M", sdfg.arrays["M"]))
    if with_c_in:
        state.add_edge(state.add_read("Cin"), None, node, "_c_in",
                       dace.Memlet.from_array("Cin", sdfg.arrays["Cin"]))

    state.add_edge(node, "_out", state.add_write("Out"), None,
                   dace.Memlet.from_array("Out", sdfg.arrays["Out"]))
    return sdfg


def _build_symbolic_expr_sdfg(node, *, shape=(8,), with_c_in=False):
    sdfg = dace.SDFG("test_symbolic_masked_expr")
    state = sdfg.add_state("s")

    sdfg.add_array("A", shape=shape, dtype=dace.float64)
    sdfg.add_array("B", shape=shape, dtype=dace.float64)
    if with_c_in:
        sdfg.add_array("Cin", shape=shape, dtype=dace.float64)
    sdfg.add_array("Out", shape=shape, dtype=dace.float64)

    state.add_node(node)
    state.add_edge(state.add_read("A"), None, node, "_a",
                   dace.Memlet.from_array("A", sdfg.arrays["A"]))
    state.add_edge(state.add_read("B"), None, node, "_b",
                   dace.Memlet.from_array("B", sdfg.arrays["B"]))
    if with_c_in:
        state.add_edge(state.add_read("Cin"), None, node, "_c_in",
                       dace.Memlet.from_array("Cin", sdfg.arrays["Cin"]))

    state.add_edge(node, "_out", state.add_write("Out"), None,
                   dace.Memlet.from_array("Out", sdfg.arrays["Out"]))
    return sdfg


def test_runtime_masked_expr_with_cin():
    _a, _b = sp.symbols("_a _b")
    node = TileRuntimeMaskedOpLibraryNode("mexpr", expr=_a * (_b + 1))
    sdfg = _build_runtime_expr_sdfg(node, shape=(8,), with_c_in=True)
    sdfg.expand_library_nodes()
    prog = sdfg.compile()

    a = np.arange(8, dtype=np.float64)
    b = np.arange(8, dtype=np.float64) * 2
    m = np.array([True, False, True, False, True, False, True, False], dtype=np.bool_)
    cin = np.full(8, 7.0, dtype=np.float64)
    out = np.zeros(8, dtype=np.float64)

    prog(A=a, B=b, M=m, Cin=cin, Out=out)
    np.testing.assert_allclose(out, np.where(m, a * (b + 1), cin))


def test_runtime_masked_expr_without_cin_preserves_zero_init():
    _a, _b = sp.symbols("_a _b")
    node = TileRuntimeMaskedOpLibraryNode("mexpr", expr=_a + _b)
    sdfg = _build_runtime_expr_sdfg(node, shape=(8,), with_c_in=False)
    sdfg.expand_library_nodes()
    prog = sdfg.compile()

    a = np.arange(8, dtype=np.float64)
    b = np.ones(8, dtype=np.float64)
    m = np.array([True, False, True, False, True, False, True, False], dtype=np.bool_)
    out = np.zeros(8, dtype=np.float64)

    prog(A=a, B=b, M=m, Out=out)
    np.testing.assert_allclose(out, np.where(m, a + b, 0.0))


def test_symbolic_masked_expr_with_cin():
    _a, _b = sp.symbols("_a _b")
    cond = sp.Eq(sp.Mod(sp.Symbol("__m0"), 2), 0)
    node = TileSymbolicMaskedOpLibraryNode(
        "smexpr",
        expr=_a + _b,
        mask_condition=cond,
    )
    sdfg = _build_symbolic_expr_sdfg(node, shape=(10,), with_c_in=True)
    sdfg.expand_library_nodes()
    prog = sdfg.compile()

    a = np.arange(10, dtype=np.float64)
    b = np.ones(10, dtype=np.float64) * 3
    cin = np.full(10, -2.0, dtype=np.float64)
    out = np.zeros(10, dtype=np.float64)

    prog(A=a, B=b, Cin=cin, Out=out)
    expected = np.where((np.arange(10) % 2) == 0, a + b, cin)
    np.testing.assert_allclose(out, expected)


def test_runtime_masked_expr_rejects_reserved_output_symbol():
    _a = sp.Symbol("_a")
    _out_sym = sp.Symbol("_out")
    node = TileRuntimeMaskedOpLibraryNode("bad", expr=_a + _out_sym)
    sdfg = _build_runtime_expr_sdfg(node, shape=(4,), with_c_in=False)
    with pytest.raises(InvalidSDFGNodeError, match="reserved for the output connector"):
        sdfg.validate()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
