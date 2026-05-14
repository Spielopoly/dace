"""Tests for the cuTile Python backend code generation.

Verifies that:
- Library node ``cutile_python`` expansions produce Python Tasklets with the
  correct ``__CUTILE_SPEC__`` marker.
- ``CuTilePythonCodeGen`` predicate correctly identifies marker tasklets.
- Full code generation via ``codegen.generate_code`` produces ``@ct.kernel``
  definitions and ``ct.launch`` calls.
- ``scalar_to_tile_library.py`` auto-selects ``cutile_python`` for Python-
  backend SDFGs.
"""

import pytest
import sympy as sp
import dace
from dace import dtypes
from dace.sdfg import SDFG, nodes
from dace.codegen import codegen as dace_codegen
from dace.libraries.cutile.nodes.op import TileOpLibraryNode
from dace.libraries.cutile.nodes.op_runtime_map import TileRuntimeMaskedOpLibraryNode
from dace.libraries.cutile.nodes.where_select import TileWhereSelectLibraryNode
from dace.libraries.cutile.nodes.if_else_op import TileIfElseOpLibraryNode
from dace.libraries.cutile.nodes.python_spec import (
    CUTILE_MARKER,
    CuTileSpec,
    decode_spec,
    encode_spec,
    make_marker_code,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_add_sdfg(implementation: str = "cutile_python") -> SDFG:
    """Build a minimal SDFG: C = A + B using TileOpLibraryNode."""
    sdfg = SDFG("simple_add")
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array("A", shape=[128], dtype=dace.float32)
    sdfg.add_array("B", shape=[128], dtype=dace.float32)
    sdfg.add_array("C", shape=[128], dtype=dace.float32)
    state = sdfg.add_state()
    a, b, c = state.add_read("A"), state.add_read("B"), state.add_write("C")
    lib = TileOpLibraryNode("TileAdd", op="+", tile_shape=[16])
    lib.implementation = implementation
    state.add_node(lib)
    state.add_edge(a, None, lib, "_a", dace.Memlet.from_array("A", sdfg.arrays["A"]))
    state.add_edge(b, None, lib, "_b", dace.Memlet.from_array("B", sdfg.arrays["B"]))
    state.add_edge(lib, "_out", c, None, dace.Memlet.from_array("C", sdfg.arrays["C"]))
    return sdfg


def _get_tasklets(sdfg: SDFG):
    return [
        node
        for st in sdfg.states()
        for node in st.nodes()
        if isinstance(node, nodes.Tasklet)
    ]


# ---------------------------------------------------------------------------
# Spec encode/decode
# ---------------------------------------------------------------------------

def test_encode_decode_roundtrip():
    spec = CuTileSpec(kind="unmasked", op="+", tile_shape=[16], ndim=1)
    code = make_marker_code(spec)
    assert CUTILE_MARKER in code
    recovered = decode_spec(code)
    assert recovered is not None
    assert recovered.kind == "unmasked"
    assert recovered.op == "+"
    assert recovered.tile_shape == [16]


def test_decode_spec_no_marker():
    assert decode_spec("x = 1\ny = 2") is None


def test_decode_spec_survives_ast_roundtrip():
    """decode_spec must work on code that has been through ast.parse/unparse."""
    import ast
    spec = CuTileSpec(kind="runtime_mask", op="*", tile_shape=[32], ndim=1)
    code = make_marker_code(spec)
    # Simulate the CodeBlock roundtrip
    unparsed = ast.unparse(ast.parse(code))
    recovered = decode_spec(unparsed)
    assert recovered is not None
    assert recovered.kind == "runtime_mask"
    assert recovered.op == "*"
    assert recovered.tile_shape == [32]


# ---------------------------------------------------------------------------
# Predicate
# ---------------------------------------------------------------------------

def test_is_cutile_tasklet_true():
    from dace.codegen.py.cutile_target import _is_cutile_tasklet
    sdfg = _simple_add_sdfg()
    sdfg.expand_library_nodes()
    tasklets = _get_tasklets(sdfg)
    assert len(tasklets) == 1
    state = next(iter(sdfg.states()))
    assert _is_cutile_tasklet(sdfg, state, tasklets[0])


def test_is_cutile_tasklet_false_plain():
    """Plain Python tasklets without the marker are not cuTile tasklets."""
    from dace.codegen.py.cutile_target import _is_cutile_tasklet
    sdfg = SDFG("plain")
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_scalar("x", dace.float32, transient=True)
    state = sdfg.add_state()
    t = nodes.Tasklet("plain", inputs={"_in"}, outputs={"_out"}, code="__dace_out = _in + 1")
    state.add_node(t)
    assert not _is_cutile_tasklet(sdfg, state, t)


# ---------------------------------------------------------------------------
# Expansion – cutile_python
# ---------------------------------------------------------------------------

def test_expansion_produces_marker_tasklet():
    sdfg = _simple_add_sdfg()
    sdfg.expand_library_nodes()
    tasklets = _get_tasklets(sdfg)
    assert len(tasklets) == 1
    code = tasklets[0].code.as_string
    assert CUTILE_MARKER in code
    spec = decode_spec(code)
    assert spec is not None
    assert spec.kind == "unmasked"
    assert spec.op == "+"
    assert spec.tile_shape == [16]


def test_expansion_runtime_mask():
    sdfg = SDFG("rt_mask")
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array("A", shape=[128], dtype=dace.float32)
    sdfg.add_array("mask", shape=[128], dtype=dace.bool_)
    sdfg.add_array("C", shape=[128], dtype=dace.float32)
    state = sdfg.add_state()
    a = state.add_read("A")
    m = state.add_read("mask")
    c = state.add_write("C")
    lib = TileRuntimeMaskedOpLibraryNode("MaskedAdd", op="+", tile_shape=[16])
    lib.implementation = "cutile_python"
    state.add_node(lib)
    state.add_edge(a, None, lib, "_a", dace.Memlet.from_array("A", sdfg.arrays["A"]))
    state.add_edge(m, None, lib, "_m", dace.Memlet.from_array("mask", sdfg.arrays["mask"]))
    state.add_edge(lib, "_out", c, None, dace.Memlet.from_array("C", sdfg.arrays["C"]))
    sdfg.expand_library_nodes()
    tasklets = _get_tasklets(sdfg)
    assert len(tasklets) == 1
    spec = decode_spec(tasklets[0].code.as_string)
    assert spec is not None
    assert spec.kind == "runtime_mask"


def test_expansion_where_select():
    sdfg = SDFG("where_sel")
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array("cond", shape=[128], dtype=dace.bool_)
    sdfg.add_array("x", shape=[128], dtype=dace.float32)
    sdfg.add_array("y", shape=[128], dtype=dace.float32)
    sdfg.add_array("out", shape=[128], dtype=dace.float32)
    state = sdfg.add_state()
    nc = state.add_read("cond")
    nx = state.add_read("x")
    ny = state.add_read("y")
    no = state.add_write("out")
    lib = TileWhereSelectLibraryNode("WhereSelect", tile_shape=[16])
    lib.implementation = "cutile_python"
    state.add_node(lib)
    state.add_edge(nc, None, lib, "_cond", dace.Memlet.from_array("cond", sdfg.arrays["cond"]))
    state.add_edge(nx, None, lib, "_x", dace.Memlet.from_array("x", sdfg.arrays["x"]))
    state.add_edge(ny, None, lib, "_y", dace.Memlet.from_array("y", sdfg.arrays["y"]))
    state.add_edge(lib, "_c", no, None, dace.Memlet.from_array("out", sdfg.arrays["out"]))
    sdfg.expand_library_nodes()
    tasklets = _get_tasklets(sdfg)
    assert len(tasklets) == 1
    spec = decode_spec(tasklets[0].code.as_string)
    assert spec is not None
    assert spec.kind == "where_select"


def test_expansion_if_else():
    sdfg = SDFG("if_else")
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array("A", shape=[128], dtype=dace.float32)
    sdfg.add_array("B", shape=[128], dtype=dace.float32)
    sdfg.add_array("C", shape=[128], dtype=dace.float32)
    state = sdfg.add_state()
    a, b, c = state.add_read("A"), state.add_read("B"), state.add_write("C")
    lib = TileIfElseOpLibraryNode(
        "IfElse",
        condition=sp.sympify("_a > 0"),
        true_expr=sp.sympify("_a"),
        false_expr=sp.sympify("_b"),
        tile_shape=[16],
    )
    lib.implementation = "cutile_python"
    state.add_node(lib)
    state.add_edge(a, None, lib, "_a", dace.Memlet.from_array("A", sdfg.arrays["A"]))
    state.add_edge(b, None, lib, "_b", dace.Memlet.from_array("B", sdfg.arrays["B"]))
    state.add_edge(lib, "_out", c, None, dace.Memlet.from_array("C", sdfg.arrays["C"]))
    sdfg.expand_library_nodes()
    tasklets = _get_tasklets(sdfg)
    assert len(tasklets) == 1
    spec = decode_spec(tasklets[0].code.as_string)
    assert spec is not None
    assert spec.kind == "if_else"
    assert sp.sympify(spec.cond_str) == sp.sympify("_a > 0")
    assert sp.sympify(spec.true_str) == sp.sympify("_a")
    assert sp.sympify(spec.false_str) == sp.sympify("_b")


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

def test_codegen_produces_ct_kernel():
    sdfg = _simple_add_sdfg()
    sdfg.expand_library_nodes()
    code_objects = dace_codegen.generate_code(sdfg)
    frame_code = next(co.clean_code for co in code_objects if co.name == "simple_add")
    assert "@ct.kernel" in frame_code
    assert "ct.load" in frame_code
    assert "ct.store" in frame_code
    assert "ct.launch" in frame_code


def test_codegen_imports():
    sdfg = _simple_add_sdfg()
    sdfg.expand_library_nodes()
    code_objects = dace_codegen.generate_code(sdfg)
    frame_code = next(co.clean_code for co in code_objects if co.name == "simple_add")
    assert "import cuda.tile as ct" in frame_code
    assert "import cupy as cp" in frame_code


def test_codegen_kernel_name_contains_ids():
    sdfg = _simple_add_sdfg()
    sdfg.expand_library_nodes()
    code_objects = dace_codegen.generate_code(sdfg)
    frame_code = next(co.clean_code for co in code_objects if co.name == "simple_add")
    assert "__ct_kernel_" in frame_code


def test_codegen_grid_expression():
    sdfg = _simple_add_sdfg()
    sdfg.expand_library_nodes()
    code_objects = dace_codegen.generate_code(sdfg)
    frame_code = next(co.clean_code for co in code_objects if co.name == "simple_add")
    # Grid computation should reference output array shape
    assert "C.shape[0]" in frame_code or "shape[0]" in frame_code


# ---------------------------------------------------------------------------
# Auto-selection via sdfg.backend
# ---------------------------------------------------------------------------

def test_auto_select_cutile_python_implementation():
    """TileOpLibraryNode default implementation should be 'cutile_python' when
    sdfg.backend == Python."""
    from dace.libraries.cutile.transformations.scalar_to_tile_library import (
        ScalarToTileCanonical,
    )

    sdfg = SDFG("auto_select")
    sdfg.backend = dtypes.BackendLanguage.Python
    for sym in ("M", "T0"):
        sdfg.add_symbol(sym, dace.int32)
    sdfg.add_array("A", shape=[dace.symbol("M") // dace.symbol("T0"), dace.symbol("T0")],
                   dtype=dace.float32)
    sdfg.add_array("B", shape=[dace.symbol("M") // dace.symbol("T0"), dace.symbol("T0")],
                   dtype=dace.float32)
    sdfg.add_array("C", shape=[dace.symbol("M") // dace.symbol("T0"), dace.symbol("T0")],
                   dtype=dace.float32)

    state = sdfg.add_state()
    a, b, c = state.add_read("A"), state.add_read("B"), state.add_write("C")
    outer_entry, outer_exit = state.add_map(
        "tile_map",
        {"t": "0:M//T0"},
        schedule=dtypes.ScheduleType.Sequential,
    )
    inner_entry, inner_exit = state.add_map(
        "elem_map",
        {"i": "0:T0"},
        schedule=dtypes.ScheduleType.Sequential,
    )
    tasklet = state.add_tasklet("add", {"a", "b"}, {"c"}, "c = a + b")

    state.add_memlet_path(
        a,
        outer_entry,
        inner_entry,
        tasklet,
        dst_conn="a",
        memlet=dace.Memlet("A[t, i]"),
    )
    state.add_memlet_path(
        b,
        outer_entry,
        inner_entry,
        tasklet,
        dst_conn="b",
        memlet=dace.Memlet("B[t, i]"),
    )
    state.add_memlet_path(
        tasklet,
        inner_exit,
        outer_exit,
        c,
        src_conn="c",
        memlet=dace.Memlet("C[t, i]"),
    )

    sdfg.validate()

    result = sdfg.apply_transformations_repeated([ScalarToTileCanonical])
    assert result >= 1

    lib_nodes = [
        n
        for st in sdfg.states()
        for n in st.nodes()
        if isinstance(n, TileOpLibraryNode)
    ]
    assert len(lib_nodes) >= 1
    for lib_node in lib_nodes:
        assert lib_node.implementation == "cutile_python", (
            f"Expected cutile_python but got {lib_node.implementation!r}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
