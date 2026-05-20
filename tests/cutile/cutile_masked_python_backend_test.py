"""
Comprehensive tests for the cuTile Python backend expansion of masked operations.

Covers three node types:
- TileRuntimeMaskedOpLibraryNode  (runtime boolean mask tile)
- TileSymbolicMaskedOpLibraryNode (compile-time symbolic predicate)
- TileOpLibraryNode               (unmasked, ct. prefix checks)

Tests are split into five sections:

  Section 1 – Runtime masked op: code structure (cutile_python expansion)
  Section 2 – Runtime masked op: end-to-end C++ correctness
  Section 3 – Symbolic masked op: code structure (cutile_python expansion)
  Section 4 – Symbolic masked op: end-to-end C++ correctness
  Section 5 – Unmasked op cutile_python: ct_prefix checks

NOTE: cuTile requires power-of-2 tile shapes per dimension (e.g. 16, 32, [4,4], [8,8]).
Code-structure tests use `sdfg.backend = dtypes.BackendLanguage.Python` and inspect the
expanded tasklet source directly; they do NOT run GPU code.
Correctness tests use the default ``"pure"`` (C++) expansion and compile/run on CPU.
"""

import numpy as np
import pytest
import sympy as sp

import dace
from dace import dtypes
from dace.sdfg import nodes
from dace.libraries.cutile.nodes.op import TileOpLibraryNode
from dace.libraries.cutile.nodes.op_runtime_map import TileRuntimeMaskedOpLibraryNode
from dace.libraries.cutile.nodes.op_symbolic_mask import TileSymbolicMaskedOpLibraryNode


# ── shared helpers ────────────────────────────────────────────────────────────

def _expanded_tasklet_code(
    lib_node,
    arrays: dict,
    out_name: str = "Out",
    out_shape=None,
    out_dtype=dace.float64,
) -> str:
    """Build SDFG, wire lib_node, expand, return tasklet code string.

    :param lib_node: The library node to expand (implementation already set).
    :param arrays: Dict mapping array name to (connector_name, dtype, shape).
    :param out_name: Name for the output array.
    :param out_shape: Shape of the output array; defaults to ``[16]``.
    :param out_dtype: DaCe dtype for the output array.
    :returns: The ``.code.as_string`` of the single expanded tasklet.
    """
    shape = out_shape or [16]
    sdfg = dace.SDFG("expand_only")
    sdfg.backend = dtypes.BackendLanguage.Python

    for arr_name, (_, dtype, arr_shape) in arrays.items():
        sdfg.add_array(arr_name, shape=arr_shape, dtype=dtype)
    sdfg.add_array(out_name, shape=shape, dtype=out_dtype)

    state = sdfg.add_state("main")
    state.add_node(lib_node)

    for arr_name, (conn, _, arr_shape) in arrays.items():
        r = state.add_read(arr_name)
        state.add_edge(
            r, None, lib_node, conn,
            dace.Memlet.from_array(arr_name, sdfg.arrays[arr_name]),
        )

    w = state.add_write(out_name)
    out_conn = next(iter(lib_node.out_connectors.keys()))
    state.add_edge(
        lib_node, out_conn, w, None,
        dace.Memlet.from_array(out_name, sdfg.arrays[out_name]),
    )

    sdfg.expand_library_nodes()
    tasklets = [
        n
        for st in sdfg.states()
        for n in st.nodes()
        if isinstance(n, nodes.Tasklet)
    ]
    assert len(tasklets) == 1, f"Expected 1 tasklet, got {len(tasklets)}"
    return tasklets[0].code.as_string


def _build_runtime_masked_sdfg(
    shape,
    op,
    constant1=None,
    constant2=None,
    with_c_in=True,
    tile_shape=None,
    expr=None,
):
    """Build a runnable SDFG with a TileRuntimeMaskedOpLibraryNode (pure/C++ expansion).

    :param shape: Array and tile shape list, e.g. ``[16]`` or ``[4, 4]``.
    :param op: Operation string, e.g. ``"+"`` or ``"sin"``.
    :param constant1: Optional left constant (str literal).
    :param constant2: Optional right constant (str literal).
    :param with_c_in: Whether to wire the ``_c_in`` connector.
    :param tile_shape: Explicit tile shape; defaults to *shape*.
    :param expr: Optional SymPy expression for multi-op mode.
    :returns: Compiled SDFG callable.
    """
    effective_tile = tile_shape or shape
    node = TileRuntimeMaskedOpLibraryNode(
        "rmasked",
        op=op,
        tile_shape=effective_tile,
        constant1=constant1,
        constant2=constant2,
        expr=expr,
    )

    sdfg = dace.SDFG("test_runtime_masked")
    state = sdfg.add_state("s")

    # Derive which operand connectors are actually present on the node
    has_a = "_a" in node.in_connectors
    has_b = "_b" in node.in_connectors

    if expr is not None:
        for sym_name in sorted(str(s) for s in expr.free_symbols
                               if isinstance(s, sp.Symbol)):
            sdfg.add_array(sym_name.lstrip("_").upper(), shape=shape,
                           dtype=dace.float64)
    else:
        if has_a:
            sdfg.add_array("A", shape=shape, dtype=dace.float64)
        if has_b:
            sdfg.add_array("B", shape=shape, dtype=dace.float64)

    sdfg.add_array("M", shape=shape, dtype=dace.bool_)
    if with_c_in:
        sdfg.add_array("Cin", shape=shape, dtype=dace.float64)
    sdfg.add_array("Out", shape=shape, dtype=dace.float64)

    state.add_node(node)

    if expr is not None:
        for sym_name in sorted(str(s) for s in expr.free_symbols
                               if isinstance(s, sp.Symbol)):
            arr_name = sym_name.lstrip("_").upper()
            state.add_edge(
                state.add_read(arr_name), None, node, sym_name,
                dace.Memlet.from_array(arr_name, sdfg.arrays[arr_name]),
            )
    else:
        if has_a:
            state.add_edge(state.add_read("A"), None, node, "_a",
                           dace.Memlet.from_array("A", sdfg.arrays["A"]))
        if has_b:
            state.add_edge(state.add_read("B"), None, node, "_b",
                           dace.Memlet.from_array("B", sdfg.arrays["B"]))

    state.add_edge(state.add_read("M"), None, node, "_m",
                   dace.Memlet.from_array("M", sdfg.arrays["M"]))
    if with_c_in:
        state.add_edge(state.add_read("Cin"), None, node, "_c_in",
                       dace.Memlet.from_array("Cin", sdfg.arrays["Cin"]))
    state.add_edge(node, "_out", state.add_write("Out"), None,
                   dace.Memlet.from_array("Out", sdfg.arrays["Out"]))

    sdfg.expand_library_nodes()
    return sdfg.compile()


def _build_symbolic_masked_sdfg(
    shape,
    op,
    mask_condition=None,
    constant1=None,
    constant2=None,
    with_c_in=True,
    tile_shape=None,
    expr=None,
):
    """Build a runnable SDFG with a TileSymbolicMaskedOpLibraryNode (pure/C++ expansion).

    :param shape: Array shape, e.g. ``[16]`` or ``[4, 4]``.
    :param op: Operation string.
    :param mask_condition: SymPy boolean predicate or ``None`` (always-true).
    :param constant1: Optional left constant (str literal).
    :param constant2: Optional right constant (str literal).
    :param with_c_in: Whether to wire ``_c_in``.
    :param tile_shape: Explicit tile shape; defaults to *shape*.
    :param expr: Optional SymPy expression for multi-op mode.
    :returns: Compiled SDFG callable.
    """
    effective_tile = tile_shape or shape
    node = TileSymbolicMaskedOpLibraryNode(
        "symmasked",
        op=op,
        tile_shape=effective_tile,
        mask_condition=mask_condition,
        constant1=constant1,
        constant2=constant2,
        expr=expr,
    )

    sdfg = dace.SDFG("test_symbolic_masked")
    state = sdfg.add_state("s")

    # Derive which connectors are actually present on the node
    has_a = "_a" in node.in_connectors
    has_b = "_b" in node.in_connectors

    if expr is not None:
        for sym_name in sorted(str(s) for s in expr.free_symbols
                               if isinstance(s, sp.Symbol)):
            sdfg.add_array(sym_name.lstrip("_").upper(), shape=shape,
                           dtype=dace.float64)
    else:
        if has_a:
            sdfg.add_array("A", shape=shape, dtype=dace.float64)
        if has_b:
            sdfg.add_array("B", shape=shape, dtype=dace.float64)

    if with_c_in:
        sdfg.add_array("Cin", shape=shape, dtype=dace.float64)
    sdfg.add_array("Out", shape=shape, dtype=dace.float64)

    state.add_node(node)

    if expr is not None:
        for sym_name in sorted(str(s) for s in expr.free_symbols
                               if isinstance(s, sp.Symbol)):
            arr_name = sym_name.lstrip("_").upper()
            state.add_edge(
                state.add_read(arr_name), None, node, sym_name,
                dace.Memlet.from_array(arr_name, sdfg.arrays[arr_name]),
            )
    else:
        if has_a:
            state.add_edge(state.add_read("A"), None, node, "_a",
                           dace.Memlet.from_array("A", sdfg.arrays["A"]))
        if has_b:
            state.add_edge(state.add_read("B"), None, node, "_b",
                           dace.Memlet.from_array("B", sdfg.arrays["B"]))

    if with_c_in:
        state.add_edge(state.add_read("Cin"), None, node, "_c_in",
                       dace.Memlet.from_array("Cin", sdfg.arrays["Cin"]))
    state.add_edge(node, "_out", state.add_write("Out"), None,
                   dace.Memlet.from_array("Out", sdfg.arrays["Out"]))

    sdfg.expand_library_nodes()
    return sdfg.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Runtime Masked op — code structure (cutile_python expansion)
# ─────────────────────────────────────────────────────────────────────────────

def test_runtime_masked_add_emits_ct_where():
    """Binary +: generated code must reference ct.where and the mask _m."""
    lib = TileRuntimeMaskedOpLibraryNode("MaskedAdd", op="+", tile_shape=[16])
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float64, [16]),
            "B": ("_b", dace.float64, [16]),
            "M": ("_m", dace.bool_, [16]),
            "Cin": ("_c_in", dace.float64, [16]),
        },
    )
    assert "ct.where" in code
    assert "_m" in code


def test_runtime_masked_subtract_emits_ct_where():
    """Binary -: generated code must reference ct.where."""
    lib = TileRuntimeMaskedOpLibraryNode("MaskedSub", op="-", tile_shape=[16])
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float64, [16]),
            "B": ("_b", dace.float64, [16]),
            "M": ("_m", dace.bool_, [16]),
            "Cin": ("_c_in", dace.float64, [16]),
        },
    )
    assert "ct.where" in code
    assert "_m" in code


def test_runtime_masked_multiply_emits_ct_where():
    """Binary *: generated code must reference ct.where."""
    lib = TileRuntimeMaskedOpLibraryNode("MaskedMul", op="*", tile_shape=[16])
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float64, [16]),
            "B": ("_b", dace.float64, [16]),
            "M": ("_m", dace.bool_, [16]),
            "Cin": ("_c_in", dace.float64, [16]),
        },
    )
    assert "ct.where" in code
    assert "_m" in code


def test_runtime_masked_divide_emits_ct_where():
    """Binary /: generated code must reference ct.where."""
    lib = TileRuntimeMaskedOpLibraryNode("MaskedDiv", op="/", tile_shape=[16])
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float64, [16]),
            "B": ("_b", dace.float64, [16]),
            "M": ("_m", dace.bool_, [16]),
            "Cin": ("_c_in", dace.float64, [16]),
        },
    )
    assert "ct.where" in code
    assert "_m" in code


def test_runtime_masked_unary_negate_emits_ct_where():
    """Unary negate: code must reference ct.where and the negation of _a."""
    lib = TileRuntimeMaskedOpLibraryNode("MaskedNeg", op="-", tile_shape=[16])
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float64, [16]),
            "M": ("_m", dace.bool_, [16]),
            "Cin": ("_c_in", dace.float64, [16]),
        },
    )
    assert "ct.where" in code
    # op_python_expression("-", "_a") emits "(- _a)" (with space inserted by sympy)
    assert "_a" in code
    assert "-" in code


def test_runtime_masked_unary_abs_emits_ct_where():
    """Unary abs: code must use plain abs( (not ct.abs) and ct.where."""
    lib = TileRuntimeMaskedOpLibraryNode("MaskedAbs", op="abs", tile_shape=[16])
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float64, [16]),
            "M": ("_m", dace.bool_, [16]),
            "Cin": ("_c_in", dace.float64, [16]),
        },
    )
    assert "ct.where" in code
    assert "abs(" in code
    assert "ct.abs" not in code


def test_runtime_masked_unary_sin_emits_ct_sin():
    """Unary sin: code must reference ct.sin and ct.where."""
    lib = TileRuntimeMaskedOpLibraryNode("MaskedSin", op="sin", tile_shape=[16])
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float64, [16]),
            "M": ("_m", dace.bool_, [16]),
            "Cin": ("_c_in", dace.float64, [16]),
        },
    )
    assert "ct.where" in code
    assert "ct.sin(" in code


def test_runtime_masked_unary_cos_emits_ct_cos():
    """Unary cos: code must reference ct.cos."""
    lib = TileRuntimeMaskedOpLibraryNode("MaskedCos", op="cos", tile_shape=[16])
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float64, [16]),
            "M": ("_m", dace.bool_, [16]),
            "Cin": ("_c_in", dace.float64, [16]),
        },
    )
    assert "ct.where" in code
    assert "ct.cos(" in code


def test_runtime_masked_unary_exp_emits_ct_exp():
    """Unary exp: code must reference ct.exp."""
    lib = TileRuntimeMaskedOpLibraryNode("MaskedExp", op="exp", tile_shape=[16])
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float64, [16]),
            "M": ("_m", dace.bool_, [16]),
            "Cin": ("_c_in", dace.float64, [16]),
        },
    )
    assert "ct.where" in code
    assert "ct.exp(" in code


def test_runtime_masked_unary_sqrt_emits_ct_sqrt():
    """Unary sqrt: code must reference ct.sqrt."""
    lib = TileRuntimeMaskedOpLibraryNode("MaskedSqrt", op="sqrt", tile_shape=[16])
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float64, [16]),
            "M": ("_m", dace.bool_, [16]),
            "Cin": ("_c_in", dace.float64, [16]),
        },
    )
    assert "ct.where" in code
    assert "ct.sqrt(" in code


def test_runtime_masked_unary_log_emits_ct_log():
    """Unary log: code must reference ct.log."""
    lib = TileRuntimeMaskedOpLibraryNode("MaskedLog", op="log", tile_shape=[16])
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float64, [16]),
            "M": ("_m", dace.bool_, [16]),
            "Cin": ("_c_in", dace.float64, [16]),
        },
    )
    assert "ct.where" in code
    assert "ct.log(" in code


def test_runtime_masked_const_right_emits_ct_where():
    """A + 2.0 (constant2): code must reference ct.where."""
    lib = TileRuntimeMaskedOpLibraryNode(
        "MaskedConstAdd", op="+", tile_shape=[16], constant2="2.0"
    )
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float64, [16]),
            "M": ("_m", dace.bool_, [16]),
            "Cin": ("_c_in", dace.float64, [16]),
        },
    )
    assert "ct.where" in code
    assert "2.0" in code


def test_runtime_masked_const_left_emits_ct_where():
    """1.0 - B (constant1): code must reference ct.where."""
    lib = TileRuntimeMaskedOpLibraryNode(
        "MaskedConstSub", op="-", tile_shape=[16], constant1="1.0"
    )
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {
            "B": ("_b", dace.float64, [16]),
            "M": ("_m", dace.bool_, [16]),
            "Cin": ("_c_in", dace.float64, [16]),
        },
    )
    assert "ct.where" in code
    assert "1.0" in code


def test_runtime_masked_expr_emits_ct_where():
    """Multi-op sympy expr _a * _b: code must reference ct.where."""
    _a, _b = sp.symbols("_a _b")
    lib = TileRuntimeMaskedOpLibraryNode(
        "MaskedExpr", expr=_a * _b, tile_shape=[16]
    )
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float64, [16]),
            "B": ("_b", dace.float64, [16]),
            "M": ("_m", dace.bool_, [16]),
            "Cin": ("_c_in", dace.float64, [16]),
        },
    )
    assert "ct.where" in code


def test_runtime_masked_requires_c_in():
    """Missing _c_in connector must raise ValueError mentioning requires '_c_in'."""
    lib = TileRuntimeMaskedOpLibraryNode("MaskedNoCin", op="+", tile_shape=[16])
    lib.implementation = "cutile_python"
    with pytest.raises(ValueError, match="requires '_c_in'"):
        _expanded_tasklet_code(
            lib,
            {
                "A": ("_a", dace.float64, [16]),
                "B": ("_b", dace.float64, [16]),
                "M": ("_m", dace.bool_, [16]),
            },
        )


def test_runtime_masked_code_uses_c_in_as_fallback():
    """Generated code must include the _c_in symbol (used as the else-branch)."""
    lib = TileRuntimeMaskedOpLibraryNode("MaskedCinFallback", op="+", tile_shape=[16])
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float64, [16]),
            "B": ("_b", dace.float64, [16]),
            "M": ("_m", dace.bool_, [16]),
            "Cin": ("_c_in", dace.float64, [16]),
        },
    )
    assert "_c_in" in code


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Runtime Masked op — end-to-end C++ execution (correctness)
# ─────────────────────────────────────────────────────────────────────────────

def test_runtime_masked_add_correctness():
    """if mask: out = a + b, else out = cin."""
    shape = [16]
    sdfg = dace.SDFG("test_rma")
    sdfg.add_array("A", shape=shape, dtype=dace.float64)
    sdfg.add_array("B", shape=shape, dtype=dace.float64)
    sdfg.add_array("M", shape=shape, dtype=dace.bool_)
    sdfg.add_array("Cin", shape=shape, dtype=dace.float64)
    sdfg.add_array("Out", shape=shape, dtype=dace.float64)

    state = sdfg.add_state("s")
    node = TileRuntimeMaskedOpLibraryNode("add", op="+", tile_shape=[16])
    state.add_node(node)
    state.add_edge(state.add_read("A"), None, node, "_a",
                   dace.Memlet.from_array("A", sdfg.arrays["A"]))
    state.add_edge(state.add_read("B"), None, node, "_b",
                   dace.Memlet.from_array("B", sdfg.arrays["B"]))
    state.add_edge(state.add_read("M"), None, node, "_m",
                   dace.Memlet.from_array("M", sdfg.arrays["M"]))
    state.add_edge(state.add_read("Cin"), None, node, "_c_in",
                   dace.Memlet.from_array("Cin", sdfg.arrays["Cin"]))
    state.add_edge(node, "_out", state.add_write("Out"), None,
                   dace.Memlet.from_array("Out", sdfg.arrays["Out"]))

    sdfg.expand_library_nodes()
    prog = sdfg.compile()

    rng = np.random.default_rng(1)
    a = rng.uniform(-5, 5, size=shape).astype(np.float64)
    b = rng.uniform(-5, 5, size=shape).astype(np.float64)
    mask = rng.integers(0, 2, size=shape).astype(np.bool_)
    cin = rng.uniform(-10, 10, size=shape).astype(np.float64)
    out = cin.copy()

    prog(A=a, B=b, M=mask, Cin=cin, Out=out)
    np.testing.assert_allclose(out, np.where(mask, a + b, cin), rtol=1e-12)


def test_runtime_masked_subtract_correctness():
    """if mask: out = a - b, else out = cin."""
    shape = [16]
    prog = _build_runtime_masked_sdfg(shape, op="-")
    rng = np.random.default_rng(2)
    a = rng.uniform(-5, 5, size=shape).astype(np.float64)
    b = rng.uniform(-5, 5, size=shape).astype(np.float64)
    mask = rng.integers(0, 2, size=shape).astype(np.bool_)
    cin = rng.uniform(-10, 10, size=shape).astype(np.float64)
    out = cin.copy()
    prog(A=a, B=b, M=mask, Cin=cin, Out=out)
    np.testing.assert_allclose(out, np.where(mask, a - b, cin), rtol=1e-12)


def test_runtime_masked_multiply_correctness():
    """if mask: out = a * b, else out = cin."""
    shape = [16]
    prog = _build_runtime_masked_sdfg(shape, op="*")
    rng = np.random.default_rng(3)
    a = rng.uniform(-5, 5, size=shape).astype(np.float64)
    b = rng.uniform(-5, 5, size=shape).astype(np.float64)
    mask = rng.integers(0, 2, size=shape).astype(np.bool_)
    cin = rng.uniform(-10, 10, size=shape).astype(np.float64)
    out = cin.copy()
    prog(A=a, B=b, M=mask, Cin=cin, Out=out)
    np.testing.assert_allclose(out, np.where(mask, a * b, cin), rtol=1e-12)


def test_runtime_masked_unary_sin_correctness():
    """if mask: out = sin(a), else out = cin."""
    shape = [16]
    prog = _build_runtime_masked_sdfg(shape, op="sin")
    rng = np.random.default_rng(4)
    a = rng.uniform(-np.pi, np.pi, size=shape).astype(np.float64)
    mask = rng.integers(0, 2, size=shape).astype(np.bool_)
    cin = rng.uniform(-10, 10, size=shape).astype(np.float64)
    out = cin.copy()
    prog(A=a, M=mask, Cin=cin, Out=out)
    np.testing.assert_allclose(out, np.where(mask, np.sin(a), cin), rtol=1e-12)


def test_runtime_masked_unary_abs_correctness():
    """if mask: out = abs(a), else out = cin."""
    shape = [16]
    prog = _build_runtime_masked_sdfg(shape, op="abs")
    rng = np.random.default_rng(5)
    a = rng.uniform(-5, 5, size=shape).astype(np.float64)
    mask = rng.integers(0, 2, size=shape).astype(np.bool_)
    cin = rng.uniform(-10, 10, size=shape).astype(np.float64)
    out = cin.copy()
    prog(A=a, M=mask, Cin=cin, Out=out)
    np.testing.assert_allclose(out, np.where(mask, np.abs(a), cin), rtol=1e-12)


def test_runtime_masked_const_correctness():
    """A + 3.0 with mask (constant2)."""
    shape = [16]
    prog = _build_runtime_masked_sdfg(shape, op="+", constant2="3.0")
    rng = np.random.default_rng(6)
    a = rng.uniform(-5, 5, size=shape).astype(np.float64)
    mask = rng.integers(0, 2, size=shape).astype(np.bool_)
    cin = rng.uniform(-10, 10, size=shape).astype(np.float64)
    out = cin.copy()
    prog(A=a, M=mask, Cin=cin, Out=out)
    np.testing.assert_allclose(out, np.where(mask, a + 3.0, cin), rtol=1e-12)


def test_runtime_masked_expr_correctness():
    """_a * (_b + 1) expression with mask."""
    shape = [16]
    _a, _b = sp.symbols("_a _b")
    prog = _build_runtime_masked_sdfg(shape, op="+", expr=_a * (_b + 1))
    rng = np.random.default_rng(7)
    a = rng.uniform(-5, 5, size=shape).astype(np.float64)
    b = rng.uniform(-5, 5, size=shape).astype(np.float64)
    mask = rng.integers(0, 2, size=shape).astype(np.bool_)
    cin = rng.uniform(-10, 10, size=shape).astype(np.float64)
    out = cin.copy()
    prog(A=a, B=b, M=mask, Cin=cin, Out=out)
    np.testing.assert_allclose(out, np.where(mask, a * (b + 1), cin), rtol=1e-12)


def test_runtime_masked_all_true_mask():
    """All-True mask: result should equal op applied to every element."""
    shape = [16]
    prog = _build_runtime_masked_sdfg(shape, op="+")
    rng = np.random.default_rng(8)
    a = rng.uniform(-5, 5, size=shape).astype(np.float64)
    b = rng.uniform(-5, 5, size=shape).astype(np.float64)
    mask = np.ones(shape, dtype=np.bool_)
    cin = rng.uniform(-10, 10, size=shape).astype(np.float64)
    out = cin.copy()
    prog(A=a, B=b, M=mask, Cin=cin, Out=out)
    np.testing.assert_allclose(out, a + b, rtol=1e-12)


def test_runtime_masked_all_false_mask():
    """All-False mask: output should equal cin unchanged."""
    shape = [16]
    prog = _build_runtime_masked_sdfg(shape, op="+")
    rng = np.random.default_rng(9)
    a = rng.uniform(-5, 5, size=shape).astype(np.float64)
    b = rng.uniform(-5, 5, size=shape).astype(np.float64)
    mask = np.zeros(shape, dtype=np.bool_)
    cin = rng.uniform(-10, 10, size=shape).astype(np.float64)
    out = cin.copy()
    prog(A=a, B=b, M=mask, Cin=cin, Out=out)
    np.testing.assert_allclose(out, cin, rtol=1e-12)


def test_runtime_masked_2d_tile():
    """2D tile shape [4,4] verify masked add works element-wise."""
    shape = [4, 4]
    prog = _build_runtime_masked_sdfg(shape, op="+", tile_shape=[4, 4])
    rng = np.random.default_rng(10)
    a = rng.uniform(-5, 5, size=shape).astype(np.float64)
    b = rng.uniform(-5, 5, size=shape).astype(np.float64)
    mask = rng.integers(0, 2, size=shape).astype(np.bool_)
    cin = rng.uniform(-10, 10, size=shape).astype(np.float64)
    out = cin.copy()
    prog(A=a, B=b, M=mask, Cin=cin, Out=out)
    np.testing.assert_allclose(out, np.where(mask, a + b, cin), rtol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Symbolic Masked op — code structure (cutile_python expansion)
# ─────────────────────────────────────────────────────────────────────────────

def test_symbolic_masked_emits_ct_where():
    """Basic symbolic mask: code must contain ct.where."""
    cond = sp.Symbol("__m0") < 8
    lib = TileSymbolicMaskedOpLibraryNode(
        "SymMasked", op="+", tile_shape=[16], mask_condition=cond
    )
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float64, [16]),
            "B": ("_b", dace.float64, [16]),
            "Cin": ("_c_in", dace.float64, [16]),
        },
    )
    assert "ct.where" in code


def test_symbolic_masked_emits_ct_arange_1d():
    """1D tile with symbolic mask: code must use ct.arange."""
    cond = sp.Symbol("__m0") < 8
    lib = TileSymbolicMaskedOpLibraryNode(
        "SymMasked1D", op="+", tile_shape=[16], mask_condition=cond
    )
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float64, [16]),
            "B": ("_b", dace.float64, [16]),
            "Cin": ("_c_in", dace.float64, [16]),
        },
    )
    assert "ct.arange" in code


def test_symbolic_masked_emits_ct_broadcast_2d():
    """2D tile: code must use ct.broadcast_to and ct.arange."""
    cond = sp.Symbol("__m0") < 2
    lib = TileSymbolicMaskedOpLibraryNode(
        "SymMasked2D", op="+", tile_shape=[4, 4], mask_condition=cond
    )
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float64, [4, 4]),
            "B": ("_b", dace.float64, [4, 4]),
            "Cin": ("_c_in", dace.float64, [4, 4]),
        },
        out_shape=[4, 4],
    )
    assert "ct.broadcast_to" in code
    assert "ct.arange" in code


def test_symbolic_masked_add_code_structure():
    """__m0 < 8 condition: generated code has arange(16), condition, and ct.where."""
    cond = sp.Symbol("__m0") < 8
    lib = TileSymbolicMaskedOpLibraryNode(
        "SymMaskedAdd", op="+", tile_shape=[16], mask_condition=cond
    )
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float64, [16]),
            "B": ("_b", dace.float64, [16]),
            "Cin": ("_c_in", dace.float64, [16]),
        },
    )
    assert "ct.arange" in code
    assert "ct.where" in code
    # The condition operand 8 must appear somewhere in the code
    assert "8" in code


def test_symbolic_masked_unary_sin_emits_ct_sin():
    """sin with symbolic mask: code must contain ct.sin and ct.where."""
    cond = sp.Symbol("__m0") < 8
    lib = TileSymbolicMaskedOpLibraryNode(
        "SymMaskedSin", op="sin", tile_shape=[16], mask_condition=cond
    )
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float64, [16]),
            "Cin": ("_c_in", dace.float64, [16]),
        },
    )
    assert "ct.sin(" in code
    assert "ct.where" in code


def test_symbolic_masked_expr_mode_emits_ct_where():
    """SymPy expression mode with symbolic mask must produce ct.where."""
    _a, _b = sp.symbols("_a _b")
    cond = sp.Symbol("__m0") < 8
    lib = TileSymbolicMaskedOpLibraryNode(
        "SymMaskedExpr", expr=_a + _b, tile_shape=[16], mask_condition=cond
    )
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float64, [16]),
            "B": ("_b", dace.float64, [16]),
            "Cin": ("_c_in", dace.float64, [16]),
        },
    )
    assert "ct.where" in code


def test_symbolic_masked_no_mask_condition():
    """mask_condition=None means direct assignment without ct.where."""
    lib = TileSymbolicMaskedOpLibraryNode(
        "SymMaskedNoMask", op="+", tile_shape=[16], mask_condition=None
    )
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float64, [16]),
            "B": ("_b", dace.float64, [16]),
            "Cin": ("_c_in", dace.float64, [16]),
        },
    )
    # No mask condition: should be a direct assignment (no ct.where)
    assert "ct.where" not in code
    # The output connector should appear on the left-hand side of an assignment
    assert "_out" in code


def test_symbolic_masked_requires_c_in():
    """Missing _c_in must raise ValueError mentioning requires '_c_in'."""
    cond = sp.Symbol("__m0") < 8
    lib = TileSymbolicMaskedOpLibraryNode(
        "SymNoCin", op="+", tile_shape=[16], mask_condition=cond
    )
    lib.implementation = "cutile_python"
    with pytest.raises(ValueError, match="requires '_c_in'"):
        _expanded_tasklet_code(
            lib,
            {
                "A": ("_a", dace.float64, [16]),
                "B": ("_b", dace.float64, [16]),
            },
        )


def test_symbolic_masked_multidim_uses_correct_dim():
    """Condition on __m1 in a 2D tile must emit only the __m1 arange, not __m0."""
    cond = sp.Symbol("__m1") < 2
    lib = TileSymbolicMaskedOpLibraryNode(
        "SymMasked2DM1", op="+", tile_shape=[4, 4], mask_condition=cond
    )
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float64, [4, 4]),
            "B": ("_b", dace.float64, [4, 4]),
            "Cin": ("_c_in", dace.float64, [4, 4]),
        },
        out_shape=[4, 4],
    )
    # __m1 should be defined but NOT __m0 (only used dim is emitted)
    assert "__m1" in code
    assert "__m0" not in code


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Symbolic Masked op — end-to-end C++ execution (correctness)
# ─────────────────────────────────────────────────────────────────────────────

def test_symbolic_masked_add_correctness_1d():
    """__m0 < 8 on 1D tile [16]: first 8 elements get a+b, rest get cin."""
    shape = [16]
    cond = sp.Symbol("__m0") < 8
    prog = _build_symbolic_masked_sdfg(shape, op="+", mask_condition=cond)
    rng = np.random.default_rng(11)
    a = rng.uniform(-5, 5, size=shape).astype(np.float64)
    b = rng.uniform(-5, 5, size=shape).astype(np.float64)
    cin = rng.uniform(-10, 10, size=shape).astype(np.float64)
    out = cin.copy()
    prog(A=a, B=b, Cin=cin, Out=out)
    idx_mask = np.arange(16) < 8
    np.testing.assert_allclose(out, np.where(idx_mask, a + b, cin), rtol=1e-12)


def test_symbolic_masked_add_correctness_2d():
    """__m0 < 2 on 2D tile [4,4]: first 2 rows get a+b, rest get cin."""
    shape = [4, 4]
    cond = sp.Symbol("__m0") < 2
    prog = _build_symbolic_masked_sdfg(
        shape, op="+", mask_condition=cond, tile_shape=[4, 4]
    )
    rng = np.random.default_rng(12)
    a = rng.uniform(-5, 5, size=shape).astype(np.float64)
    b = rng.uniform(-5, 5, size=shape).astype(np.float64)
    cin = rng.uniform(-10, 10, size=shape).astype(np.float64)
    out = cin.copy()
    prog(A=a, B=b, Cin=cin, Out=out)
    row_mask = (np.arange(4) < 2)[:, np.newaxis] * np.ones((4, 4), dtype=bool)
    np.testing.assert_allclose(out, np.where(row_mask, a + b, cin), rtol=1e-12)


def test_symbolic_masked_unary_neg_correctness():
    """Negate with __m0 % 2 == 0 condition: even indices get negated.

    Since op="-" is ambiguous (in _BINARY_OPS and _UNARY_OPS), we use the
    expr mode with ``-sp.Symbol("_a")`` to express unary negation unambiguously.
    """
    shape = [16]
    _a = sp.Symbol("_a")
    m0 = sp.Symbol("__m0")
    cond = sp.Eq(sp.Mod(m0, 2), 0)
    # Use expr mode to express unary negate unambiguously
    prog = _build_symbolic_masked_sdfg(
        shape, op="-", mask_condition=cond, expr=-_a
    )
    rng = np.random.default_rng(13)
    a = rng.uniform(-5, 5, size=shape).astype(np.float64)
    cin = rng.uniform(-10, 10, size=shape).astype(np.float64)
    out = cin.copy()
    prog(A=a, Cin=cin, Out=out)
    idx_mask = (np.arange(16) % 2) == 0
    np.testing.assert_allclose(out, np.where(idx_mask, -a, cin), rtol=1e-12)


def test_symbolic_masked_const_right_correctness():
    """A + 5.0 with __m0 >= 4: indices >=4 get a+5, rest get cin."""
    shape = [16]
    cond = sp.Symbol("__m0") >= 4
    prog = _build_symbolic_masked_sdfg(
        shape, op="+", mask_condition=cond, constant2="5.0"
    )
    rng = np.random.default_rng(14)
    a = rng.uniform(-5, 5, size=shape).astype(np.float64)
    cin = rng.uniform(-10, 10, size=shape).astype(np.float64)
    out = cin.copy()
    prog(A=a, Cin=cin, Out=out)
    idx_mask = np.arange(16) >= 4
    np.testing.assert_allclose(out, np.where(idx_mask, a + 5.0, cin), rtol=1e-12)


def test_symbolic_masked_expr_correctness():
    """_a * _b expr with __m0 < 4 condition: first 4 elements get a*b."""
    shape = [16]
    _a, _b = sp.symbols("_a _b")
    cond = sp.Symbol("__m0") < 4
    prog = _build_symbolic_masked_sdfg(
        shape, op="+", mask_condition=cond, expr=_a * _b
    )
    rng = np.random.default_rng(15)
    a = rng.uniform(-5, 5, size=shape).astype(np.float64)
    b = rng.uniform(-5, 5, size=shape).astype(np.float64)
    cin = rng.uniform(-10, 10, size=shape).astype(np.float64)
    out = cin.copy()
    prog(A=a, B=b, Cin=cin, Out=out)
    idx_mask = np.arange(16) < 4
    np.testing.assert_allclose(out, np.where(idx_mask, a * b, cin), rtol=1e-12)


def test_symbolic_masked_eq_mod_condition():
    """__m0 % 2 == 0 even-element masking with binary add."""
    shape = [16]
    m0 = sp.Symbol("__m0")
    cond = sp.Eq(sp.Mod(m0, 2), 0)
    prog = _build_symbolic_masked_sdfg(shape, op="+", mask_condition=cond)
    rng = np.random.default_rng(16)
    a = rng.uniform(-5, 5, size=shape).astype(np.float64)
    b = rng.uniform(-5, 5, size=shape).astype(np.float64)
    cin = rng.uniform(-10, 10, size=shape).astype(np.float64)
    out = cin.copy()
    prog(A=a, B=b, Cin=cin, Out=out)
    idx_mask = (np.arange(16) % 2) == 0
    np.testing.assert_allclose(out, np.where(idx_mask, a + b, cin), rtol=1e-12)


def test_symbolic_masked_none_mask_condition_acts_as_all_true():
    """mask_condition=None (always true): all elements get the operation applied.

    Note: mask_condition=None is the correct way to express "no masking" —
    the expansion applies the operation unconditionally without ct.where.
    Using sp.S.true is incorrect because symstr converts it to Python's
    'True' (capitalised) which is invalid C++.
    """
    shape = [16]
    prog = _build_symbolic_masked_sdfg(
        shape, op="+", mask_condition=None
    )
    rng = np.random.default_rng(17)
    a = rng.uniform(-5, 5, size=shape).astype(np.float64)
    b = rng.uniform(-5, 5, size=shape).astype(np.float64)
    cin = rng.uniform(-10, 10, size=shape).astype(np.float64)
    out = cin.copy()
    prog(A=a, B=b, Cin=cin, Out=out)
    np.testing.assert_allclose(out, a + b, rtol=1e-12)


def test_symbolic_masked_with_c_in_preserved():
    """Masked-off elements must equal the original cin values."""
    shape = [16]
    # Only mask elements at index < 4; the rest must keep cin exactly.
    cond = sp.Symbol("__m0") < 4
    prog = _build_symbolic_masked_sdfg(shape, op="*", mask_condition=cond)
    rng = np.random.default_rng(18)
    a = rng.uniform(-5, 5, size=shape).astype(np.float64)
    b = rng.uniform(-5, 5, size=shape).astype(np.float64)
    cin = rng.uniform(-10, 10, size=shape).astype(np.float64)
    out = cin.copy()
    prog(A=a, B=b, Cin=cin, Out=out)
    idx_mask = np.arange(16) < 4
    # Masked-off region must be exactly cin
    np.testing.assert_array_equal(out[~idx_mask], cin[~idx_mask])
    # Masked-on region must be a*b
    np.testing.assert_allclose(out[idx_mask], (a * b)[idx_mask], rtol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Unmasked op cutile_python — ct_prefix tests
# ─────────────────────────────────────────────────────────────────────────────

def test_unmasked_sin_emits_ct_sin():
    """TileOpLibraryNode sin + cutile_python -> ct.sin in generated code."""
    lib = TileOpLibraryNode("Sin", op="sin", tile_shape=[16])
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {"A": ("_a", dace.float64, [16])},
    )
    assert "ct.sin(" in code


def test_unmasked_cos_emits_ct_cos():
    """TileOpLibraryNode cos + cutile_python -> ct.cos in generated code."""
    lib = TileOpLibraryNode("Cos", op="cos", tile_shape=[16])
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {"A": ("_a", dace.float64, [16])},
    )
    assert "ct.cos(" in code


def test_unmasked_exp_emits_ct_exp():
    """TileOpLibraryNode exp + cutile_python -> ct.exp in generated code."""
    lib = TileOpLibraryNode("Exp", op="exp", tile_shape=[16])
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {"A": ("_a", dace.float64, [16])},
    )
    assert "ct.exp(" in code


def test_unmasked_sqrt_emits_ct_sqrt():
    """TileOpLibraryNode sqrt + cutile_python -> ct.sqrt in generated code."""
    lib = TileOpLibraryNode("Sqrt", op="sqrt", tile_shape=[16])
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {"A": ("_a", dace.float64, [16])},
    )
    assert "ct.sqrt(" in code


def test_unmasked_abs_no_ct_prefix():
    """TileOpLibraryNode abs + cutile_python -> abs( not ct.abs(."""
    lib = TileOpLibraryNode("Abs", op="abs", tile_shape=[16])
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {"A": ("_a", dace.float64, [16])},
    )
    assert "abs(" in code
    assert "ct.abs" not in code


def test_unmasked_add_no_ct_prefix():
    """TileOpLibraryNode binary + + cutile_python -> (_a + _b) not ct.add."""
    lib = TileOpLibraryNode("Add", op="+", tile_shape=[16])
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float64, [16]),
            "B": ("_b", dace.float64, [16]),
        },
    )
    assert "(_a + _b)" in code
    assert "ct.add" not in code


def test_symbolic_masked_multidim_both_dims_used():
    """Condition referencing __m0 AND __m1 simultaneously emits two arange lines."""
    cond = sp.And(sp.Symbol("__m0") < 2, sp.Symbol("__m1") < 2)
    lib = TileSymbolicMaskedOpLibraryNode(
        "BothDims", op="+", tile_shape=[4, 4],
        mask_condition=cond,
    )
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float64, [4, 4]),
            "B": ("_b", dace.float64, [4, 4]),
            "Cin": ("_c_in", dace.float64, [4, 4]),
        },
        out_shape=[4, 4],
    )
    # Both coordinate tiles should be generated
    assert "ct.arange(4" in code  # __m0 with shape[0]=4
    assert "__m0" in code
    assert "__m1" in code
    assert "ct.where" in code
    assert "ct.broadcast_to" in code


def test_symbolic_masked_scalar_tile_form():
    """Scalar tile (tile_shape=[1]) uses ternary expression, not ct.where."""
    cond = sp.Lt(sp.Symbol("__m0"), 1)  # always true for single element
    lib = TileSymbolicMaskedOpLibraryNode(
        "ScalarTile", op="+", tile_shape=[1],
        mask_condition=cond,
    )
    lib.implementation = "cutile_python"
    code = _expanded_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float64, [1]),
            "B": ("_b", dace.float64, [1]),
            "Cin": ("_c_in", dace.float64, [1]),
        },
        out_shape=[1],
    )
    # Scalar form uses ternary "if" not ct.where
    assert "if __mask" in code
    assert "ct.where" not in code
    # Coordinate is set to 0
    assert "__m0 = 0" in code


def test_symbolic_masked_none_condition_does_not_require_c_in():
    """mask_condition=None: _c_in is optional, no ct.where emitted."""
    lib = TileSymbolicMaskedOpLibraryNode("NoMask", op="+", tile_shape=[16])
    lib.implementation = "cutile_python"
    # No _c_in provided — should not raise
    code = _expanded_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float64, [16]),
            "B": ("_b", dace.float64, [16]),
        },
    )
    assert "ct.where" not in code
    assert "_out = " in code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
