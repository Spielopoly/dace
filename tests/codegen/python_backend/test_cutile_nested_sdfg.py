# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for CuTile NestedSDFG code generation as module-level functions.

The refactored _generate_NestedSDFG generates separate module-level
functions (with return values) instead of inlining.  Tiles are immutable
in cuTile, so output connectors must be returned, not passed as arguments.
"""
import pytest

import dace
from dace import dtypes
from dace.codegen import dispatcher as dispatcher_mod
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.codegen.py.cutile_target import CuTilePythonCodeGen
from dace.dtypes import ScheduleType, StorageType, Language
from dace.sdfg import nodes, SDFG
from dace.memlet import Memlet


# =============================================================================
# Test infrastructure
# =============================================================================


def _make_cutile_python_sdfg(name: str) -> SDFG:
    """Create an SDFG with backend set to Python for cuTile testing."""
    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    return sdfg


class _StubFrameCodegen:
    """Minimal stub for the frame codegen, just providing a dispatcher."""

    def __init__(self):
        self.dispatcher = dispatcher_mod.TargetDispatcher(self)
        self._initcode = PythonCodeIOStream()
        self._exitcode = PythonCodeIOStream()


def _make_cutile_codegen(sdfg: SDFG) -> CuTilePythonCodeGen:
    """Instantiate a CuTilePythonCodeGen with a stub frame codegen."""
    frame = _StubFrameCodegen()
    codegen = CuTilePythonCodeGen(frame, sdfg)
    return codegen


def _generate_nested_code(sdfg: SDFG, state, nested_node):
    """Call _generate_NestedSDFG and return (function_code, callsite_code)."""
    codegen = _make_cutile_codegen(sdfg)
    function_stream = PythonCodeIOStream()
    callsite_stream = PythonCodeIOStream()
    state_id = sdfg.node_id(state)
    codegen._generate_NestedSDFG(
        sdfg, sdfg, state, state_id, nested_node,
        function_stream, callsite_stream)
    return function_stream.getvalue(), callsite_stream.getvalue()


# =============================================================================
# SDFG builders
# =============================================================================


def _build_simple_binop_nested_sdfg(name: str = "simple_binop_nsdfg"):
    """Build an SDFG with a NestedSDFG that does tile_a + tile_b -> tile_out.

    Outer structure:
        AccessNode(_tile_a) -> NestedSDFG(a, b -> out) <- AccessNode(_tile_b)
                                   |
                                   v
                            AccessNode(_tile_out)

    Inside a CuTile map scope.
    """
    sdfg = _make_cutile_python_sdfg(name)
    sdfg.add_array("A", [32], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("B", [32], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("C", [32], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("_tile_a", [32], dace.float64,
                   storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array("_tile_b", [32], dace.float64,
                   storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array("_tile_out", [32], dace.float64,
                   storage=StorageType.CuTile_Tile, transient=True)

    state = sdfg.add_state("main")
    me, mx = state.add_map("cutile_map", {"tile_i": "0:1"},
                           schedule=ScheduleType.CuTile)

    a_node = state.add_read("A")
    b_node = state.add_read("B")
    c_node = state.add_write("C")
    tile_a = state.add_access("_tile_a")
    tile_b = state.add_access("_tile_b")
    tile_out = state.add_access("_tile_out")

    # Build inner nested SDFG: a + b -> out
    inner_sdfg = SDFG("binop_add")
    inner_sdfg.add_array("a", [32], dace.float64,
                         storage=StorageType.CuTile_Tile)
    inner_sdfg.add_array("b", [32], dace.float64,
                         storage=StorageType.CuTile_Tile)
    inner_sdfg.add_array("out", [32], dace.float64,
                         storage=StorageType.CuTile_Tile)
    inner_state = inner_sdfg.add_state("compute")
    in_a = inner_state.add_read("a")
    in_b = inner_state.add_read("b")
    out_node = inner_state.add_write("out")
    tasklet = inner_state.add_tasklet("add", {"_a", "_b"}, {"_out"},
                                      "_out = _a + _b",
                                      language=Language.Python)
    inner_state.add_edge(in_a, None, tasklet, "_a", Memlet(data="a", subset="0:32"))
    inner_state.add_edge(in_b, None, tasklet, "_b", Memlet(data="b", subset="0:32"))
    inner_state.add_edge(tasklet, "_out", out_node, None,
                         Memlet(data="out", subset="0:32"))

    nsdfg = state.add_nested_sdfg(inner_sdfg, {"a", "b"}, {"out"})

    # Wire up: A -> MapEntry -> tile_a -> NestedSDFG
    state.add_memlet_path(a_node, me, tile_a,
                          memlet=Memlet(data="A", subset="0:32"))
    state.add_memlet_path(b_node, me, tile_b,
                          memlet=Memlet(data="B", subset="0:32"))
    state.add_edge(tile_a, None, nsdfg, "a",
                   Memlet(data="_tile_a", subset="0:32"))
    state.add_edge(tile_b, None, nsdfg, "b",
                   Memlet(data="_tile_b", subset="0:32"))
    state.add_edge(nsdfg, "out", tile_out, None,
                   Memlet(data="_tile_out", subset="0:32"))
    state.add_memlet_path(tile_out, mx, c_node,
                          memlet=Memlet(data="C", subset="0:32"))

    return sdfg, state, nsdfg


def _build_unop_nested_sdfg(name: str = "unop_nsdfg"):
    """Build a NestedSDFG that does -tile_a -> tile_out (unary negation).

    Single input, single output.
    """
    sdfg = _make_cutile_python_sdfg(name)
    sdfg.add_array("A", [16], dace.float32, storage=StorageType.GPU_Global)
    sdfg.add_array("B", [16], dace.float32, storage=StorageType.GPU_Global)
    sdfg.add_array("_tile_a", [16], dace.float32,
                   storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array("_tile_out", [16], dace.float32,
                   storage=StorageType.CuTile_Tile, transient=True)

    state = sdfg.add_state("main")
    me, mx = state.add_map("cutile_map", {"tile_i": "0:1"},
                           schedule=ScheduleType.CuTile)

    a_node = state.add_read("A")
    b_node = state.add_write("B")
    tile_a = state.add_access("_tile_a")
    tile_out = state.add_access("_tile_out")

    # Build inner nested SDFG: -a -> out
    inner_sdfg = SDFG("unop_neg")
    inner_sdfg.add_array("inp", [16], dace.float32,
                         storage=StorageType.CuTile_Tile)
    inner_sdfg.add_array("result", [16], dace.float32,
                         storage=StorageType.CuTile_Tile)
    inner_state = inner_sdfg.add_state("compute")
    in_node = inner_state.add_read("inp")
    out_node = inner_state.add_write("result")
    tasklet = inner_state.add_tasklet("neg", {"x"}, {"y"},
                                      "y = -x",
                                      language=Language.Python)
    inner_state.add_edge(in_node, None, tasklet, "x",
                         Memlet(data="inp", subset="0:16"))
    inner_state.add_edge(tasklet, "y", out_node, None,
                         Memlet(data="result", subset="0:16"))

    nsdfg = state.add_nested_sdfg(inner_sdfg, {"inp"}, {"result"})

    state.add_memlet_path(a_node, me, tile_a,
                          memlet=Memlet(data="A", subset="0:16"))
    state.add_edge(tile_a, None, nsdfg, "inp",
                   Memlet(data="_tile_a", subset="0:16"))
    state.add_edge(nsdfg, "result", tile_out, None,
                   Memlet(data="_tile_out", subset="0:16"))
    state.add_memlet_path(tile_out, mx, b_node,
                          memlet=Memlet(data="B", subset="0:16"))

    return sdfg, state, nsdfg


def _build_multi_output_nested_sdfg(name: str = "multi_output_nsdfg"):
    """Build a NestedSDFG with two outputs: out1 = a + b, out2 = a - b."""
    sdfg = _make_cutile_python_sdfg(name)
    sdfg.add_array("A", [32], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("B", [32], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("C", [32], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("D", [32], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("_tile_a", [32], dace.float64,
                   storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array("_tile_b", [32], dace.float64,
                   storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array("_tile_c", [32], dace.float64,
                   storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array("_tile_d", [32], dace.float64,
                   storage=StorageType.CuTile_Tile, transient=True)

    state = sdfg.add_state("main")
    me, mx = state.add_map("cutile_map", {"tile_i": "0:1"},
                           schedule=ScheduleType.CuTile)

    a_node = state.add_read("A")
    b_node = state.add_read("B")
    c_node = state.add_write("C")
    d_node = state.add_write("D")
    tile_a = state.add_access("_tile_a")
    tile_b = state.add_access("_tile_b")
    tile_c = state.add_access("_tile_c")
    tile_d = state.add_access("_tile_d")

    # Build inner nested SDFG
    inner_sdfg = SDFG("dual_op")
    inner_sdfg.add_array("a", [32], dace.float64,
                         storage=StorageType.CuTile_Tile)
    inner_sdfg.add_array("b", [32], dace.float64,
                         storage=StorageType.CuTile_Tile)
    inner_sdfg.add_array("out1", [32], dace.float64,
                         storage=StorageType.CuTile_Tile)
    inner_sdfg.add_array("out2", [32], dace.float64,
                         storage=StorageType.CuTile_Tile)
    inner_state = inner_sdfg.add_state("compute")
    in_a = inner_state.add_read("a")
    in_b = inner_state.add_read("b")
    out1_node = inner_state.add_write("out1")
    out2_node = inner_state.add_write("out2")
    # Tasklet 1: out1 = a + b
    t1 = inner_state.add_tasklet("add", {"_a", "_b"}, {"_sum"},
                                 "_sum = _a + _b", language=Language.Python)
    inner_state.add_edge(in_a, None, t1, "_a", Memlet(data="a", subset="0:32"))
    inner_state.add_edge(in_b, None, t1, "_b", Memlet(data="b", subset="0:32"))
    inner_state.add_edge(t1, "_sum", out1_node, None,
                         Memlet(data="out1", subset="0:32"))
    # Tasklet 2: out2 = a - b
    # We need a second read of 'a' and 'b' for the second tasklet
    in_a2 = inner_state.add_read("a")
    in_b2 = inner_state.add_read("b")
    t2 = inner_state.add_tasklet("sub", {"_a", "_b"}, {"_diff"},
                                 "_diff = _a - _b", language=Language.Python)
    inner_state.add_edge(in_a2, None, t2, "_a", Memlet(data="a", subset="0:32"))
    inner_state.add_edge(in_b2, None, t2, "_b", Memlet(data="b", subset="0:32"))
    inner_state.add_edge(t2, "_diff", out2_node, None,
                         Memlet(data="out2", subset="0:32"))

    nsdfg = state.add_nested_sdfg(inner_sdfg,
                                  {"a", "b"}, {"out1", "out2"})

    state.add_memlet_path(a_node, me, tile_a,
                          memlet=Memlet(data="A", subset="0:32"))
    state.add_memlet_path(b_node, me, tile_b,
                          memlet=Memlet(data="B", subset="0:32"))
    state.add_edge(tile_a, None, nsdfg, "a",
                   Memlet(data="_tile_a", subset="0:32"))
    state.add_edge(tile_b, None, nsdfg, "b",
                   Memlet(data="_tile_b", subset="0:32"))
    state.add_edge(nsdfg, "out1", tile_c, None,
                   Memlet(data="_tile_c", subset="0:32"))
    state.add_edge(nsdfg, "out2", tile_d, None,
                   Memlet(data="_tile_d", subset="0:32"))
    state.add_memlet_path(tile_c, mx, c_node,
                          memlet=Memlet(data="C", subset="0:32"))
    state.add_memlet_path(tile_d, mx, d_node,
                          memlet=Memlet(data="D", subset="0:32"))

    return sdfg, state, nsdfg


def _build_no_output_nested_sdfg(name: str = "no_output_nsdfg"):
    """Build a NestedSDFG with no output connectors (side-effect only)."""
    sdfg = _make_cutile_python_sdfg(name)
    sdfg.add_array("A", [32], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("_tile_a", [32], dace.float64,
                   storage=StorageType.CuTile_Tile, transient=True)

    state = sdfg.add_state("main")
    me, mx = state.add_map("cutile_map", {"tile_i": "0:1"},
                           schedule=ScheduleType.CuTile)

    a_node = state.add_read("A")
    tile_a = state.add_access("_tile_a")

    # Build inner nested SDFG with no output (just reads input)
    inner_sdfg = SDFG("noop")
    inner_sdfg.add_array("a", [32], dace.float64,
                         storage=StorageType.CuTile_Tile)
    inner_state = inner_sdfg.add_state("compute")
    in_a = inner_state.add_read("a")
    # Tasklet that does nothing meaningful
    tasklet = inner_state.add_tasklet("noop", {"x"}, {},
                                      "pass", language=Language.Python)
    inner_state.add_edge(in_a, None, tasklet, "x",
                         Memlet(data="a", subset="0:32"))

    nsdfg = state.add_nested_sdfg(inner_sdfg, {"a"}, set())

    state.add_memlet_path(a_node, me, tile_a,
                          memlet=Memlet(data="A", subset="0:32"))
    state.add_edge(tile_a, None, nsdfg, "a",
                   Memlet(data="_tile_a", subset="0:32"))
    # No output edges; map exit connects with empty memlet.
    state.add_memlet_path(nsdfg, mx, memlet=Memlet())

    return sdfg, state, nsdfg


def _build_symbol_passing_nested_sdfg(name: str = "symbol_nsdfg"):
    """Build a NestedSDFG that uses a symbol from the outer SDFG.

    Inner tasklet: out = inp * N (where N is a symbol).
    """
    N = dace.symbol("N")
    sdfg = _make_cutile_python_sdfg(name)
    sdfg.add_symbol("N", dace.int32)
    sdfg.add_array("A", [32], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("B", [32], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("_tile_a", [32], dace.float64,
                   storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array("_tile_out", [32], dace.float64,
                   storage=StorageType.CuTile_Tile, transient=True)

    state = sdfg.add_state("main")
    me, mx = state.add_map("cutile_map", {"tile_i": "0:1"},
                           schedule=ScheduleType.CuTile)

    a_node = state.add_read("A")
    b_node = state.add_write("B")
    tile_a = state.add_access("_tile_a")
    tile_out = state.add_access("_tile_out")

    # Build inner nested SDFG using symbol N
    inner_sdfg = SDFG("scale_by_N")
    inner_sdfg.add_symbol("N", dace.int32)
    inner_sdfg.add_array("inp", [32], dace.float64,
                         storage=StorageType.CuTile_Tile)
    inner_sdfg.add_array("out", [32], dace.float64,
                         storage=StorageType.CuTile_Tile)
    inner_state = inner_sdfg.add_state("compute")
    in_node = inner_state.add_read("inp")
    out_node = inner_state.add_write("out")
    tasklet = inner_state.add_tasklet("scale", {"x"}, {"y"},
                                      "y = x * N",
                                      language=Language.Python)
    inner_state.add_edge(in_node, None, tasklet, "x",
                         Memlet(data="inp", subset="0:32"))
    inner_state.add_edge(tasklet, "y", out_node, None,
                         Memlet(data="out", subset="0:32"))

    nsdfg = state.add_nested_sdfg(inner_sdfg, {"inp"}, {"out"},
                                  symbol_mapping={"N": N})

    state.add_memlet_path(a_node, me, tile_a,
                          memlet=Memlet(data="A", subset="0:32"))
    state.add_edge(tile_a, None, nsdfg, "inp",
                   Memlet(data="_tile_a", subset="0:32"))
    state.add_edge(nsdfg, "out", tile_out, None,
                   Memlet(data="_tile_out", subset="0:32"))
    state.add_memlet_path(tile_out, mx, b_node,
                          memlet=Memlet(data="B", subset="0:32"))

    return sdfg, state, nsdfg


def _build_recursive_nested_sdfg(name: str = "recursive_nsdfg"):
    """Build a NestedSDFG that itself contains another NestedSDFG.

    Outer NestedSDFG: a -> [inner NestedSDFG: negate] -> out
    Inner NestedSDFG: x -> -x -> y
    """
    sdfg = _make_cutile_python_sdfg(name)
    sdfg.add_array("A", [16], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("B", [16], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("_tile_a", [16], dace.float64,
                   storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array("_tile_out", [16], dace.float64,
                   storage=StorageType.CuTile_Tile, transient=True)

    state = sdfg.add_state("main")
    me, mx = state.add_map("cutile_map", {"tile_i": "0:1"},
                           schedule=ScheduleType.CuTile)

    a_node = state.add_read("A")
    b_node = state.add_write("B")
    tile_a = state.add_access("_tile_a")
    tile_out = state.add_access("_tile_out")

    # Build the deepest (innermost) nested SDFG: negate
    deep_sdfg = SDFG("deep_neg")
    deep_sdfg.add_array("x", [16], dace.float64,
                        storage=StorageType.CuTile_Tile)
    deep_sdfg.add_array("y", [16], dace.float64,
                        storage=StorageType.CuTile_Tile)
    deep_state = deep_sdfg.add_state("negate")
    deep_x = deep_state.add_read("x")
    deep_y = deep_state.add_write("y")
    deep_tasklet = deep_state.add_tasklet("neg", {"_in"}, {"_out"},
                                          "_out = -_in",
                                          language=Language.Python)
    deep_state.add_edge(deep_x, None, deep_tasklet, "_in",
                        Memlet(data="x", subset="0:16"))
    deep_state.add_edge(deep_tasklet, "_out", deep_y, None,
                        Memlet(data="y", subset="0:16"))

    # Build the outer nested SDFG that contains the deep one
    outer_inner_sdfg = SDFG("wrapper")
    outer_inner_sdfg.add_array("a", [16], dace.float64,
                               storage=StorageType.CuTile_Tile)
    outer_inner_sdfg.add_array("out", [16], dace.float64,
                               storage=StorageType.CuTile_Tile)
    oi_state = outer_inner_sdfg.add_state("wrap")
    oi_a = oi_state.add_read("a")
    oi_out = oi_state.add_write("out")
    # Add the deep nested SDFG
    deep_nsdfg = oi_state.add_nested_sdfg(deep_sdfg, {"x"}, {"y"})
    oi_state.add_edge(oi_a, None, deep_nsdfg, "x",
                      Memlet(data="a", subset="0:16"))
    oi_state.add_edge(deep_nsdfg, "y", oi_out, None,
                      Memlet(data="out", subset="0:16"))

    nsdfg = state.add_nested_sdfg(outer_inner_sdfg, {"a"}, {"out"})

    state.add_memlet_path(a_node, me, tile_a,
                          memlet=Memlet(data="A", subset="0:16"))
    state.add_edge(tile_a, None, nsdfg, "a",
                   Memlet(data="_tile_a", subset="0:16"))
    state.add_edge(nsdfg, "out", tile_out, None,
                   Memlet(data="_tile_out", subset="0:16"))
    state.add_memlet_path(tile_out, mx, b_node,
                          memlet=Memlet(data="B", subset="0:16"))

    return sdfg, state, nsdfg


# =============================================================================
# Tests: Function generation structure
# =============================================================================


class TestNestedSDFGFunctionGeneration:
    """Test that NestedSDFGs generate module-level functions."""

    def test_simple_binop_generates_function_def(self):
        """A simple binary op nested SDFG should produce a def statement."""
        sdfg, state, nsdfg = _build_simple_binop_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        assert "def __dace_nested_" in func_code
        assert "binop_add" in func_code

    def test_simple_binop_function_has_parameters(self):
        """The function should have input connectors as parameters."""
        sdfg, state, nsdfg = _build_simple_binop_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        # Input connectors are 'a' and 'b'
        assert "(a, b)" in func_code or "(a, b):" in func_code

    def test_simple_binop_function_has_return(self):
        """The function should return the output connector."""
        sdfg, state, nsdfg = _build_simple_binop_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        assert "return out" in func_code

    def test_simple_binop_call_site_assigns_output(self):
        """The call site should assign the returned value to the output tile."""
        sdfg, state, nsdfg = _build_simple_binop_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        # Output should be assigned to _tile_out
        assert "_tile_out = __dace_nested_" in call_code
        # Arguments should be the input tile variables
        assert "_tile_a" in call_code
        assert "_tile_b" in call_code

    def test_simple_binop_tasklet_body_in_function(self):
        """The tasklet body should appear in the function definition."""
        sdfg, state, nsdfg = _build_simple_binop_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        # codeblock_to_python may parenthesize: "_out = (_a + _b)"
        assert "_out = (_a + _b)" in func_code or "_out = _a + _b" in func_code

    def test_unop_single_input_single_output(self):
        """A unary operation nested SDFG should work with 1 input and 1 output."""
        sdfg, state, nsdfg = _build_unop_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        assert "def __dace_nested_" in func_code
        assert "return result" in func_code
        # codeblock_to_python may parenthesize: "y = (- x)"
        assert "y = -x" in func_code or "y = (- x)" in func_code

    def test_unop_call_site(self):
        """The call site for a unary op should pass 1 argument and receive 1 output."""
        sdfg, state, nsdfg = _build_unop_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        assert "_tile_out = __dace_nested_" in call_code
        assert "_tile_a" in call_code


class TestNestedSDFGMultipleOutputs:
    """Test NestedSDFGs with multiple output connectors."""

    def test_multi_output_returns_tuple(self):
        """A NestedSDFG with 2 outputs should return a tuple."""
        sdfg, state, nsdfg = _build_multi_output_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        assert "return out1, out2" in func_code

    def test_multi_output_call_site_unpacks(self):
        """The call site should unpack the tuple to separate variables."""
        sdfg, state, nsdfg = _build_multi_output_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        assert "_tile_c, _tile_d = __dace_nested_" in call_code


class TestNestedSDFGNoOutputs:
    """Test NestedSDFGs with no output connectors."""

    def test_no_output_no_return(self):
        """A NestedSDFG with no outputs should not have a return statement."""
        sdfg, state, nsdfg = _build_no_output_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        assert "return" not in func_code

    def test_no_output_call_site_no_assignment(self):
        """The call site for a no-output NestedSDFG should have no assignment."""
        sdfg, state, nsdfg = _build_no_output_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        assert "= __dace_nested_" not in call_code
        assert "__dace_nested_" in call_code


class TestNestedSDFGSymbolPassing:
    """Test that symbols are correctly passed to nested functions."""

    def test_symbol_in_parameters(self):
        """Symbols should appear as function parameters."""
        sdfg, state, nsdfg = _build_symbol_passing_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        # N should be a parameter
        assert "N" in func_code.split("def ")[1].split("):")[0]

    def test_symbol_in_call_args(self):
        """Symbols should be passed in the call arguments."""
        sdfg, state, nsdfg = _build_symbol_passing_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        # N should appear in the call
        assert "N" in call_code

    def test_symbol_used_in_body(self):
        """The tasklet body uses the symbol."""
        sdfg, state, nsdfg = _build_symbol_passing_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        # codeblock_to_python may parenthesize: "y = (x * N)"
        assert "y = x * N" in func_code or "y = (x * N)" in func_code


class TestNestedSDFGRecursive:
    """Test recursive NestedSDFGs (NestedSDFG within NestedSDFG)."""

    def test_recursive_generates_two_functions(self):
        """A recursive NestedSDFG should generate two function definitions."""
        sdfg, state, nsdfg = _build_recursive_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        # Count the number of 'def ' occurrences
        def_count = func_code.count("\ndef ") + (1 if func_code.startswith("def ") else 0)
        # Should have at least 2 functions (wrapper + deep_neg)
        assert def_count >= 2, f"Expected >= 2 defs, got {def_count}\nCode:\n{func_code}"

    def test_recursive_inner_has_negate(self):
        """The inner function should contain the negation tasklet."""
        sdfg, state, nsdfg = _build_recursive_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        # codeblock_to_python may parenthesize: "_out = (- _in)"
        assert "_out = -_in" in func_code or "_out = (- _in)" in func_code

    def test_recursive_outer_calls_inner(self):
        """The outer function should call the inner function."""
        sdfg, state, nsdfg = _build_recursive_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        # The outer function should contain a call to the deep_neg function
        assert "__dace_nested_deep_neg" in func_code


class TestNestedSDFGDeduplication:
    """Test that generating the same NestedSDFG twice doesn't duplicate functions."""

    def test_same_node_not_duplicated(self):
        """Calling _generate_NestedSDFG twice for the same node should not duplicate."""
        sdfg, state, nsdfg = _build_simple_binop_nested_sdfg()
        codegen = _make_cutile_codegen(sdfg)
        function_stream = PythonCodeIOStream()
        callsite_stream = PythonCodeIOStream()
        state_id = sdfg.node_id(state)

        # Generate twice
        codegen._generate_NestedSDFG(
            sdfg, sdfg, state, state_id, nsdfg,
            function_stream, callsite_stream)
        codegen._generate_NestedSDFG(
            sdfg, sdfg, state, state_id, nsdfg,
            function_stream, callsite_stream)

        func_code = function_stream.getvalue()
        # Only one function definition should exist
        def_count = func_code.count("def __dace_nested_binop_add")
        assert def_count == 1, f"Expected 1 def, got {def_count}"


class TestNestedSDFGInputResolution:
    """Test that input variable names are correctly resolved at the call site."""

    def test_input_from_access_node(self):
        """When connected from AccessNode, use the data name."""
        sdfg, state, nsdfg = _build_simple_binop_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        # Call should pass _tile_a and _tile_b
        assert "_tile_a" in call_code
        assert "_tile_b" in call_code


class TestNestedSDFGCodeValidity:
    """Test that generated code is valid Python."""

    def test_simple_binop_function_is_valid_python(self):
        """The generated function should be parseable Python."""
        import ast
        sdfg, state, nsdfg = _build_simple_binop_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        # Try parsing the function code
        ast.parse(func_code)

    def test_unop_function_is_valid_python(self):
        """The generated unop function should be parseable Python."""
        import ast
        sdfg, state, nsdfg = _build_unop_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        ast.parse(func_code)

    def test_multi_output_function_is_valid_python(self):
        """The multi-output function should be parseable Python."""
        import ast
        sdfg, state, nsdfg = _build_multi_output_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        ast.parse(func_code)

    def test_symbol_function_is_valid_python(self):
        """The symbol-passing function should be parseable Python."""
        import ast
        sdfg, state, nsdfg = _build_symbol_passing_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        ast.parse(func_code)

    def test_recursive_function_is_valid_python(self):
        """The recursive NestedSDFG functions should be parseable Python."""
        import ast
        sdfg, state, nsdfg = _build_recursive_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        ast.parse(func_code)

    def test_call_site_is_valid_python(self):
        """The call site code should be parseable Python."""
        import ast
        sdfg, state, nsdfg = _build_simple_binop_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        ast.parse(call_code)


class TestNestedSDFGConnectorBindings:
    """Test input/output connector binding inside the generated function."""

    def test_input_connector_bound_to_parameter(self):
        """Tasklet inputs should be bound from the AccessNode (which matches the parameter)."""
        sdfg, state, nsdfg = _build_simple_binop_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        # Inside the function, the tasklet has connectors _a and _b
        # which should be bound from the AccessNode data 'a' and 'b'
        assert "_a = a" in func_code
        assert "_b = b" in func_code

    def test_output_connector_bound_to_access_node(self):
        """Tasklet outputs should be bound to the downstream AccessNode name."""
        sdfg, state, nsdfg = _build_simple_binop_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        # The tasklet output _out should be bound to the AccessNode 'out'
        assert "out = _out" in func_code


# =============================================================================
# Additional SDFG builders (inner map scope, multi-state)
# =============================================================================


def _build_inner_map_nested_sdfg(name: str = "inner_map_nsdfg"):
    """Build a NestedSDFG containing a sequential map (element-wise square).

    Inner SDFG structure:
        AccessNode(inp) -> MapEntry[i:0:4] -> Tasklet(out[i] = inp[i]**2) -> MapExit -> AccessNode(out)

    The map applies an element-wise operation (squaring) over 4 elements.
    """
    sdfg = _make_cutile_python_sdfg(name)
    sdfg.add_array("A", [16], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("B", [16], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("_tile_a", [16], dace.float64,
                   storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array("_tile_out", [16], dace.float64,
                   storage=StorageType.CuTile_Tile, transient=True)

    state = sdfg.add_state("main")
    me, mx = state.add_map("cutile_map", {"tile_i": "0:1"},
                           schedule=ScheduleType.CuTile)

    a_node = state.add_read("A")
    b_node = state.add_write("B")
    tile_a = state.add_access("_tile_a")
    tile_out = state.add_access("_tile_out")

    # Build inner nested SDFG with a sequential map
    inner_sdfg = SDFG("square_map")
    inner_sdfg.add_array("inp", [4], dace.float64,
                         storage=StorageType.CuTile_Tile)
    inner_sdfg.add_array("out", [4], dace.float64,
                         storage=StorageType.CuTile_Tile)
    inner_state = inner_sdfg.add_state("compute")

    in_node = inner_state.add_read("inp")
    out_node = inner_state.add_write("out")

    # Sequential map: for i in range(4): out[i] = inp[i] ** 2
    inner_me, inner_mx = inner_state.add_map("square_loop", {"i": "0:4"},
                                             schedule=ScheduleType.Sequential)

    square_tasklet = inner_state.add_tasklet(
        "square", {"_x"}, {"_y"},
        "_y = _x * _x",
        language=Language.Python)

    inner_state.add_memlet_path(in_node, inner_me, square_tasklet,
                                dst_conn="_x",
                                memlet=Memlet(data="inp", subset="i"))
    inner_state.add_memlet_path(square_tasklet, inner_mx, out_node,
                                src_conn="_y",
                                memlet=Memlet(data="out", subset="i"))

    nsdfg = state.add_nested_sdfg(inner_sdfg, {"inp"}, {"out"})

    state.add_memlet_path(a_node, me, tile_a,
                          memlet=Memlet(data="A", subset="0:16"))
    state.add_edge(tile_a, None, nsdfg, "inp",
                   Memlet(data="_tile_a", subset="0:4"))
    state.add_edge(nsdfg, "out", tile_out, None,
                   Memlet(data="_tile_out", subset="0:4"))
    state.add_memlet_path(tile_out, mx, b_node,
                          memlet=Memlet(data="B", subset="0:16"))

    return sdfg, state, nsdfg


def _build_multi_state_nested_sdfg(name: str = "multi_state_nsdfg"):
    """Build a NestedSDFG with two states (should raise NotImplementedError).

    Inner SDFG has state1 -> state2 with an interstate edge.
    """
    sdfg = _make_cutile_python_sdfg(name)
    sdfg.add_array("A", [16], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("B", [16], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("_tile_a", [16], dace.float64,
                   storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array("_tile_out", [16], dace.float64,
                   storage=StorageType.CuTile_Tile, transient=True)

    state = sdfg.add_state("main")
    me, mx = state.add_map("cutile_map", {"tile_i": "0:1"},
                           schedule=ScheduleType.CuTile)

    a_node = state.add_read("A")
    b_node = state.add_write("B")
    tile_a = state.add_access("_tile_a")
    tile_out = state.add_access("_tile_out")

    # Build inner nested SDFG with 2 states
    inner_sdfg = SDFG("two_states")
    inner_sdfg.add_array("inp", [16], dace.float64,
                         storage=StorageType.CuTile_Tile)
    inner_sdfg.add_array("out", [16], dace.float64,
                         storage=StorageType.CuTile_Tile)
    state1 = inner_sdfg.add_state("first")
    state2 = inner_sdfg.add_state("second")
    inner_sdfg.add_edge(state1, state2, dace.InterstateEdge())

    # Add a trivial tasklet in state2
    in_node = state2.add_read("inp")
    out_node = state2.add_write("out")
    tasklet = state2.add_tasklet("copy", {"x"}, {"y"}, "y = x",
                                 language=Language.Python)
    state2.add_edge(in_node, None, tasklet, "x",
                    Memlet(data="inp", subset="0:16"))
    state2.add_edge(tasklet, "y", out_node, None,
                    Memlet(data="out", subset="0:16"))

    nsdfg = state.add_nested_sdfg(inner_sdfg, {"inp"}, {"out"})

    state.add_memlet_path(a_node, me, tile_a,
                          memlet=Memlet(data="A", subset="0:16"))
    state.add_edge(tile_a, None, nsdfg, "inp",
                   Memlet(data="_tile_a", subset="0:16"))
    state.add_edge(nsdfg, "out", tile_out, None,
                   Memlet(data="_tile_out", subset="0:16"))
    state.add_memlet_path(tile_out, mx, b_node,
                          memlet=Memlet(data="B", subset="0:16"))

    return sdfg, state, nsdfg


# =============================================================================
# Tests: Inner map scope
# =============================================================================


class TestNestedSDFGInnerMapScope:
    """Test NestedSDFGs that contain inner sequential maps."""

    def test_inner_map_generates_for_loop(self):
        """A sequential map inside a NestedSDFG should emit a for-loop."""
        sdfg, state, nsdfg = _build_inner_map_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        assert "for i in range(" in func_code

    def test_inner_map_tasklet_in_body(self):
        """The tasklet inside the inner map should appear in the generated code."""
        sdfg, state, nsdfg = _build_inner_map_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        # The squaring tasklet body
        assert "_y = _x * _x" in func_code or \
               "_y = (_x * _x)" in func_code

    def test_inner_map_valid_python(self):
        """The generated code with an inner map should be parseable Python."""
        import ast
        sdfg, state, nsdfg = _build_inner_map_nested_sdfg()
        func_code, call_code = _generate_nested_code(sdfg, state, nsdfg)
        ast.parse(func_code)


# =============================================================================
# Tests: Multi-state guard
# =============================================================================


class TestNestedSDFGMultiStateGuard:
    """Test that multi-state NestedSDFGs raise NotImplementedError."""

    def test_multi_state_raises_not_implemented(self):
        """A NestedSDFG with 2 states should raise NotImplementedError."""
        sdfg, state, nsdfg = _build_multi_state_nested_sdfg()
        with pytest.raises(NotImplementedError, match="single-state"):
            _generate_nested_code(sdfg, state, nsdfg)

    def test_multi_state_error_mentions_state_count(self):
        """The error message should mention the number of states."""
        sdfg, state, nsdfg = _build_multi_state_nested_sdfg()
        with pytest.raises(NotImplementedError, match="2 states"):
            _generate_nested_code(sdfg, state, nsdfg)


# =============================================================================
# Tests: Language check
# =============================================================================


class TestNestedSDFGLanguageCheck:
    """Test that non-Python tasklets in NestedSDFGs raise NotImplementedError."""

    def test_cpp_tasklet_raises_not_implemented(self):
        """A C++ tasklet inside a NestedSDFG should raise NotImplementedError."""
        sdfg = _make_cutile_python_sdfg("cpp_tasklet_nsdfg")
        sdfg.add_array("A", [16], dace.float64, storage=StorageType.GPU_Global)
        sdfg.add_array("B", [16], dace.float64, storage=StorageType.GPU_Global)
        sdfg.add_array("_tile_a", [16], dace.float64,
                       storage=StorageType.CuTile_Tile, transient=True)
        sdfg.add_array("_tile_out", [16], dace.float64,
                       storage=StorageType.CuTile_Tile, transient=True)

        state = sdfg.add_state("main")
        me, mx = state.add_map("cutile_map", {"tile_i": "0:1"},
                               schedule=ScheduleType.CuTile)

        a_node = state.add_read("A")
        b_node = state.add_write("B")
        tile_a = state.add_access("_tile_a")
        tile_out = state.add_access("_tile_out")

        # Build inner nested SDFG with a C++ tasklet
        inner_sdfg = SDFG("cpp_inner")
        inner_sdfg.add_array("inp", [16], dace.float64,
                             storage=StorageType.CuTile_Tile)
        inner_sdfg.add_array("out", [16], dace.float64,
                             storage=StorageType.CuTile_Tile)
        inner_state = inner_sdfg.add_state("compute")
        in_node = inner_state.add_read("inp")
        out_node = inner_state.add_write("out")
        tasklet = inner_state.add_tasklet("cpp_op", {"x"}, {"y"},
                                          "y = x;",
                                          language=Language.CPP)
        inner_state.add_edge(in_node, None, tasklet, "x",
                             Memlet(data="inp", subset="0:16"))
        inner_state.add_edge(tasklet, "y", out_node, None,
                             Memlet(data="out", subset="0:16"))

        nsdfg = state.add_nested_sdfg(inner_sdfg, {"inp"}, {"out"})

        state.add_memlet_path(a_node, me, tile_a,
                              memlet=Memlet(data="A", subset="0:16"))
        state.add_edge(tile_a, None, nsdfg, "inp",
                       Memlet(data="_tile_a", subset="0:16"))
        state.add_edge(nsdfg, "out", tile_out, None,
                       Memlet(data="_tile_out", subset="0:16"))
        state.add_memlet_path(tile_out, mx, b_node,
                              memlet=Memlet(data="B", subset="0:16"))

        with pytest.raises(NotImplementedError, match="Python tasklets"):
            _generate_nested_code(sdfg, state, nsdfg)
