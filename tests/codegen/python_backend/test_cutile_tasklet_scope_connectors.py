# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for CuTile scope connector resolution in _generate_Tasklet.

When a Tasklet inside a CuTile map scope is connected to MapEntry (inputs)
or MapExit (outputs) rather than directly to AccessNodes, the codegen must
trace through the scope connectors to find the actual array names.

Without the fix, the generated code would use connector names like ``OUT_A``
or ``IN_C`` instead of the actual array names ``A`` and ``C``.
"""
import pytest

import dace
from dace import dtypes
from dace.codegen import dispatcher as dispatcher_mod
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.codegen.py.cutile_target import (
    CuTilePythonCodeGen,
    _matching_inner_connector,
    _matching_outer_connector,
)
from dace.dtypes import ScheduleType, StorageType, Language
from dace.sdfg import nodes, SDFG
from dace.memlet import Memlet


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


# =============================================================================
# Helper function tests
# =============================================================================


class TestConnectorHelpers:
    """Test the _matching_inner_connector and _matching_outer_connector helpers."""

    def test_matching_inner_connector_basic(self):
        """IN_A -> OUT_A."""
        assert _matching_inner_connector("IN_A") == "OUT_A"

    def test_matching_outer_connector_basic(self):
        """OUT_A -> IN_A."""
        assert _matching_outer_connector("OUT_A") == "IN_A"

    def test_matching_inner_connector_multichar(self):
        """IN_my_array -> OUT_my_array."""
        assert _matching_inner_connector("IN_my_array") == "OUT_my_array"

    def test_matching_outer_connector_multichar(self):
        """OUT_my_array -> IN_my_array."""
        assert _matching_outer_connector("OUT_my_array") == "IN_my_array"

    def test_matching_inner_connector_invalid_prefix(self):
        """Non-IN_ prefix raises ValueError."""
        with pytest.raises(ValueError, match="Expected connector starting with"):
            _matching_inner_connector("OUT_A")

    def test_matching_outer_connector_invalid_prefix(self):
        """Non-OUT_ prefix raises ValueError."""
        with pytest.raises(ValueError, match="Expected connector starting with"):
            _matching_outer_connector("IN_A")

    def test_roundtrip_inner_outer(self):
        """Inner -> outer -> inner preserves the base name."""
        assert _matching_inner_connector(
            _matching_outer_connector("OUT_foo")) == "OUT_foo"

    def test_roundtrip_outer_inner(self):
        """Outer -> inner -> outer preserves the base name."""
        assert _matching_outer_connector(
            _matching_inner_connector("IN_bar")) == "IN_bar"


# =============================================================================
# Graph building helpers
# =============================================================================


def _build_direct_scope_sdfg(name: str = "direct_scope_conn_test"):
    """Build an SDFG where the tasklet connects directly through
    MapEntry/MapExit connectors (no intermediate tile AccessNodes).

    Graph structure:
        AccessNode(A) -> MapEntry --[OUT_A/inp]--> Tasklet --[out/IN_B]--> MapExit -> AccessNode(B)

    This is the case that triggers the scope connector resolution bug.
    """
    sdfg = _make_cutile_python_sdfg(name)
    sdfg.add_array("A", [32], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("B", [32], dace.float64, storage=StorageType.GPU_Global)

    state = sdfg.add_state("main")

    # Create CuTile map
    me, mx = state.add_map("cutile_map", {"tile_i": "0:1"},
                           schedule=ScheduleType.CuTile)

    a_node = state.add_read("A")
    b_node = state.add_write("B")

    tasklet = state.add_tasklet("compute", {"inp"}, {"out"},
                                "out = inp * 2.0",
                                language=Language.Python)

    # Connect directly through map scope connectors:
    # A -> MapEntry(IN_A -> OUT_A) -> Tasklet(inp) -> ... -> MapExit(IN_B -> OUT_B) -> B
    state.add_memlet_path(a_node, me, tasklet,
                          dst_conn="inp",
                          memlet=Memlet(data="A", subset="0:32"))
    state.add_memlet_path(tasklet, mx, b_node,
                          src_conn="out",
                          memlet=Memlet(data="B", subset="0:32"))

    return sdfg, state, tasklet, me, mx


def _build_tile_intermediary_sdfg(name: str = "tile_intermediary_test"):
    """Build an SDFG where the tasklet connects via tile AccessNodes
    (the normal cuTile pattern).

    Graph structure:
        AccessNode(A) -> MapEntry -> AccessNode(_tile_A) -> Tasklet
            -> AccessNode(_tile_B) -> MapExit -> AccessNode(B)
    """
    sdfg = _make_cutile_python_sdfg(name)
    sdfg.add_array("A", [32], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("B", [32], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("_tile_A", [32], dace.float64,
                   storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array("_tile_B", [32], dace.float64,
                   storage=StorageType.CuTile_Tile, transient=True)

    state = sdfg.add_state("main")

    me, mx = state.add_map("cutile_map", {"tile_i": "0:1"},
                           schedule=ScheduleType.CuTile)

    a_node = state.add_read("A")
    b_node = state.add_write("B")
    tile_a = state.add_access("_tile_A")
    tile_b = state.add_access("_tile_B")

    tasklet = state.add_tasklet("compute", {"inp"}, {"out"},
                                "out = inp * 2.0",
                                language=Language.Python)

    state.add_memlet_path(a_node, me, tile_a,
                          dst_conn=None,
                          memlet=Memlet(data="A", subset="0:32"))
    state.add_edge(tile_a, None, tasklet, "inp",
                   Memlet(data="_tile_A", subset="0:32"))
    state.add_edge(tasklet, "out", tile_b, None,
                   Memlet(data="_tile_B", subset="0:32"))
    state.add_memlet_path(tile_b, mx, b_node,
                          src_conn=None,
                          memlet=Memlet(data="B", subset="0:32"))

    return sdfg, state, tasklet


def _build_multi_input_direct_scope_sdfg(name: str = "multi_input_scope"):
    """Build an SDFG with multiple inputs going through MapEntry.

    Graph structure:
        AccessNode(X) -> MapEntry --[OUT_X/a]--> Tasklet(a,b -> c) --[c/IN_Z]--> MapExit -> AccessNode(Z)
        AccessNode(Y) -> MapEntry --[OUT_Y/b]-->
    """
    sdfg = _make_cutile_python_sdfg(name)
    sdfg.add_array("X", [32], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("Y", [32], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("Z", [32], dace.float64, storage=StorageType.GPU_Global)

    state = sdfg.add_state("main")
    me, mx = state.add_map("cutile_map", {"tile_i": "0:1"},
                           schedule=ScheduleType.CuTile)

    x_node = state.add_read("X")
    y_node = state.add_read("Y")
    z_node = state.add_write("Z")

    tasklet = state.add_tasklet("add", {"a", "b"}, {"c"},
                                "c = a + b",
                                language=Language.Python)

    state.add_memlet_path(x_node, me, tasklet,
                          dst_conn="a",
                          memlet=Memlet(data="X", subset="0:32"))
    state.add_memlet_path(y_node, me, tasklet,
                          dst_conn="b",
                          memlet=Memlet(data="Y", subset="0:32"))
    state.add_memlet_path(tasklet, mx, z_node,
                          src_conn="c",
                          memlet=Memlet(data="Z", subset="0:32"))

    return sdfg, state, tasklet


def _generate_tasklet_code(sdfg: SDFG, state, tasklet) -> str:
    """Call _generate_Tasklet and return the generated code string."""
    codegen = _make_cutile_codegen(sdfg)
    function_stream = PythonCodeIOStream()
    callsite_stream = PythonCodeIOStream()
    state_id = sdfg.node_id(state)
    codegen._generate_Tasklet(
        sdfg, sdfg, state, state_id, tasklet,
        function_stream, callsite_stream)
    return callsite_stream.getvalue()


# =============================================================================
# Graph structure verification tests
# =============================================================================


class TestGraphStructure:
    """Verify the test SDFGs have the expected graph topology."""

    def test_direct_scope_input_is_map_entry(self):
        """In direct scope SDFG, tasklet input comes from MapEntry."""
        sdfg, state, tasklet, me, mx = _build_direct_scope_sdfg()
        in_edges = state.in_edges(tasklet)
        assert len(in_edges) == 1
        edge = in_edges[0]
        assert isinstance(edge.src, nodes.MapEntry), \
            f"Expected MapEntry, got {type(edge.src).__name__}"
        assert edge.src_conn is not None
        assert edge.src_conn.startswith("OUT_")

    def test_direct_scope_output_is_map_exit(self):
        """In direct scope SDFG, tasklet output goes to MapExit."""
        sdfg, state, tasklet, me, mx = _build_direct_scope_sdfg()
        out_edges = state.out_edges(tasklet)
        assert len(out_edges) == 1
        edge = out_edges[0]
        assert isinstance(edge.dst, nodes.MapExit), \
            f"Expected MapExit, got {type(edge.dst).__name__}"
        assert edge.dst_conn is not None
        assert edge.dst_conn.startswith("IN_")

    def test_tile_intermediary_input_is_access_node(self):
        """In tile-intermediary SDFG, tasklet input comes from AccessNode."""
        sdfg, state, tasklet = _build_tile_intermediary_sdfg()
        in_edges = state.in_edges(tasklet)
        assert len(in_edges) == 1
        edge = in_edges[0]
        assert isinstance(edge.src, nodes.AccessNode)
        assert edge.src.data == "_tile_A"

    def test_tile_intermediary_output_is_access_node(self):
        """In tile-intermediary SDFG, tasklet output goes to AccessNode."""
        sdfg, state, tasklet = _build_tile_intermediary_sdfg()
        out_edges = state.out_edges(tasklet)
        assert len(out_edges) == 1
        edge = out_edges[0]
        assert isinstance(edge.dst, nodes.AccessNode)
        assert edge.dst.data == "_tile_B"

    def test_multi_input_both_from_map_entry(self):
        """Multiple inputs all come from MapEntry."""
        sdfg, state, tasklet = _build_multi_input_direct_scope_sdfg()
        in_edges = state.in_edges(tasklet)
        assert len(in_edges) == 2
        for edge in in_edges:
            assert isinstance(edge.src, nodes.MapEntry)
            assert edge.src_conn.startswith("OUT_")


# =============================================================================
# Input scope connector resolution tests
# =============================================================================


class TestScopeConnectorResolutionInputs:
    """Test that input bindings in _generate_Tasklet resolve scope connectors."""

    def test_input_from_access_node_uses_data_name(self):
        """When input comes directly from an AccessNode, use the data name."""
        sdfg, state, tasklet = _build_tile_intermediary_sdfg(
            "input_access_node")
        code = _generate_tasklet_code(sdfg, state, tasklet)
        # Should bind: inp = _tile_A
        assert "inp = _tile_A" in code
        # Should NOT use any OUT_ connector name
        assert "OUT_" not in code

    def test_input_from_map_entry_resolves_to_array_name(self):
        """When input comes from MapEntry, resolve to the actual array name."""
        sdfg, state, tasklet, _, _ = _build_direct_scope_sdfg(
            "input_map_entry_resolve")
        code = _generate_tasklet_code(sdfg, state, tasklet)
        # Should bind: inp = A (the actual array name)
        assert "inp = A" in code
        # Should NOT contain OUT_A (the scope connector name)
        assert "OUT_A" not in code, \
            f"Generated code contains 'OUT_A' instead of 'A':\n{code}"

    def test_multiple_inputs_from_map_entry_all_resolved(self):
        """Multiple inputs through MapEntry all resolve to actual array names."""
        sdfg, state, tasklet = _build_multi_input_direct_scope_sdfg(
            "multi_input_resolve")
        code = _generate_tasklet_code(sdfg, state, tasklet)
        # Should bind: a = X and b = Y
        assert "a = X" in code
        assert "b = Y" in code
        # Should NOT contain any OUT_ connector names
        assert "OUT_X" not in code, \
            f"Generated code contains 'OUT_X' instead of 'X':\n{code}"
        assert "OUT_Y" not in code, \
            f"Generated code contains 'OUT_Y' instead of 'Y':\n{code}"


# =============================================================================
# Output scope connector resolution tests
# =============================================================================


class TestScopeConnectorResolutionOutputs:
    """Test that output bindings in _generate_Tasklet resolve scope connectors."""

    def test_output_to_access_node_uses_data_name(self):
        """When output goes to a tile-local AccessNode, use post-bind.

        Tile-local outputs (CuTile_Tile / Register storage) are NOT
        pre-bound because the variable does not exist yet.  Instead, a
        post-bind ``_tile_B = out`` is emitted AFTER the tasklet body.
        """
        sdfg, state, tasklet = _build_tile_intermediary_sdfg(
            "output_access_node")
        code = _generate_tasklet_code(sdfg, state, tasklet)
        # Post-bind should emit: _tile_B = out (after the body)
        assert "_tile_B = out" in code
        # Should NOT use any IN_ connector name
        assert "IN_" not in code

    def test_output_to_map_exit_resolves_to_array_name(self):
        """When output goes to MapExit, resolve to the actual array name.

        With pre-binding, the output connector is bound to the array
        BEFORE the tasklet body: ``out = B``.
        """
        sdfg, state, tasklet, _, _ = _build_direct_scope_sdfg(
            "output_map_exit_resolve")
        code = _generate_tasklet_code(sdfg, state, tasklet)
        # Pre-bind should emit: out = B (before the body)
        assert "out = B" in code
        # Should NOT contain IN_B (the scope connector name)
        assert "IN_B" not in code, \
            f"Generated code contains 'IN_B' instead of 'B':\n{code}"

    def test_output_same_name_no_rebind(self):
        """When dst_name == src_conn, no assignment should be emitted."""
        sdfg = _make_cutile_python_sdfg("same_name_no_rebind")
        sdfg.add_array("A", [32], dace.float64, storage=StorageType.GPU_Global)

        state = sdfg.add_state("main")
        me, mx = state.add_map("cutile_map", {"tile_i": "0:1"},
                               schedule=ScheduleType.CuTile)

        a_in = state.add_read("A")
        a_out = state.add_write("A")

        # Tasklet with output connector named "A" writing through MapExit to AccessNode "A"
        tasklet = state.add_tasklet("identity", {"inp"}, {"A"},
                                    "A = inp",
                                    language=Language.Python)

        state.add_memlet_path(a_in, me, tasklet,
                              dst_conn="inp",
                              memlet=Memlet(data="A", subset="0:32"))
        state.add_memlet_path(tasklet, mx, a_out,
                              src_conn="A",
                              memlet=Memlet(data="A", subset="0:32"))

        code = _generate_tasklet_code(sdfg, state, tasklet)
        # The output should NOT generate "A = A" since dst_name == src_conn
        lines = [line.strip() for line in code.split("\n") if line.strip()]
        rebind_lines = [l for l in lines if l == "A = A"]
        assert len(rebind_lines) == 0, \
            f"Unnecessary 'A = A' rebinding found in:\n{code}"


# =============================================================================
# Combined input + output resolution tests
# =============================================================================


class TestScopeConnectorResolutionCombined:
    """Test both input and output resolution together."""

    def test_full_scope_pass_through(self):
        """Both input and output go through scope connectors -- both resolved."""
        sdfg, state, tasklet, _, _ = _build_direct_scope_sdfg(
            "full_scope_pass_through")
        code = _generate_tasklet_code(sdfg, state, tasklet)
        # Input: inp = A (not OUT_A)
        assert "inp = A" in code
        # Output pre-bind: out = B (not B = out)
        assert "out = B" in code
        # No scope connector names in generated code
        assert "OUT_A" not in code
        assert "IN_B" not in code

    def test_tasklet_body_is_emitted(self):
        """The tasklet body is emitted between input and output bindings."""
        sdfg, state, tasklet, _, _ = _build_direct_scope_sdfg(
            "body_emitted")
        code = _generate_tasklet_code(sdfg, state, tasklet)
        # The tasklet body should be present
        assert "out = (inp * 2.0)" in code or "out = inp * 2.0" in code

    def test_mixed_access_node_and_scope_connector(self):
        """One input from AccessNode (tile transient), another from MapEntry."""
        sdfg = _make_cutile_python_sdfg("mixed_input")
        sdfg.add_array("X", [32], dace.float64, storage=StorageType.GPU_Global)
        sdfg.add_array("Y", [32], dace.float64, storage=StorageType.GPU_Global)
        sdfg.add_array("_tile_Y", [32], dace.float64,
                       storage=StorageType.CuTile_Tile, transient=True)
        sdfg.add_array("Z", [32], dace.float64, storage=StorageType.GPU_Global)

        state = sdfg.add_state("main")
        me, mx = state.add_map("cutile_map", {"tile_i": "0:1"},
                               schedule=ScheduleType.CuTile)

        x_node = state.add_read("X")
        y_node = state.add_read("Y")
        tile_y = state.add_access("_tile_Y")
        z_node = state.add_write("Z")

        tasklet = state.add_tasklet("add", {"a", "b"}, {"c"},
                                    "c = a + b",
                                    language=Language.Python)

        # 'a' comes from MapEntry (scope connector)
        state.add_memlet_path(x_node, me, tasklet,
                              dst_conn="a",
                              memlet=Memlet(data="X", subset="0:32"))
        # 'b' comes from tile AccessNode (via MapEntry -> tile_Y -> tasklet)
        state.add_memlet_path(y_node, me, tile_y,
                              memlet=Memlet(data="Y", subset="0:32"))
        state.add_edge(tile_y, None, tasklet, "b",
                       Memlet(data="_tile_Y", subset="0:32"))
        state.add_memlet_path(tasklet, mx, z_node,
                              src_conn="c",
                              memlet=Memlet(data="Z", subset="0:32"))

        code = _generate_tasklet_code(sdfg, state, tasklet)
        # 'a' should be resolved to 'X' (from MapEntry)
        assert "a = X" in code
        # 'b' should be resolved to '_tile_Y' (from AccessNode)
        assert "b = _tile_Y" in code
        # No scope connector names
        assert "OUT_X" not in code


# =============================================================================
# Output pre-binding tests
# =============================================================================


class TestOutputPreBinding:
    """Test that output connectors are pre-bound BEFORE the tasklet body.

    This is essential for in-place operations like ``ct.scatter(_dst, ...)``
    where ``_dst`` must reference the actual destination array when the
    body executes.
    """

    def test_postbind_after_body_direct_access_node(self):
        """Post-bind appears after the tasklet body (tile-local AccessNode).

        Tile-local outputs (CuTile_Tile storage) must NOT be pre-bound
        because the destination variable does not exist yet.  Instead,
        ``_tile_B = out`` is emitted AFTER the body.
        """
        sdfg, state, tasklet = _build_tile_intermediary_sdfg(
            "postbind_after_body_an")
        code = _generate_tasklet_code(sdfg, state, tasklet)
        # Post-bind: _tile_B = out must appear AFTER the tasklet body
        postbind_pos = code.find("_tile_B = out")
        body_end_pos = code.find("End of tasklet: compute")
        assert postbind_pos != -1, f"Post-bind '_tile_B = out' not found:\n{code}"
        assert body_end_pos != -1, f"Tasklet body end marker not found:\n{code}"
        assert postbind_pos > body_end_pos, \
            f"Post-bind should appear after body. Post-bind at {postbind_pos}, body end at {body_end_pos}:\n{code}"

    def test_prebind_before_body_map_exit(self):
        """Pre-bind appears before the tasklet body (MapExit destination)."""
        sdfg, state, tasklet, _, _ = _build_direct_scope_sdfg(
            "prebind_before_body_me")
        code = _generate_tasklet_code(sdfg, state, tasklet)
        # Pre-bind: out = B must appear BEFORE the tasklet body
        prebind_pos = code.find("out = B")
        body_pos = code.find("Tasklet: compute")
        assert prebind_pos != -1, f"Pre-bind 'out = B' not found:\n{code}"
        assert body_pos != -1, f"Tasklet body marker not found:\n{code}"
        assert prebind_pos < body_pos, \
            f"Pre-bind should appear before body. Pre-bind at {prebind_pos}, body at {body_pos}:\n{code}"

    def test_no_duplicate_postbind_after_prebind(self):
        """Post-bind should NOT emit a duplicate assignment for pre-bound outputs."""
        sdfg, state, tasklet, _, _ = _build_direct_scope_sdfg(
            "no_dup_postbind")
        code = _generate_tasklet_code(sdfg, state, tasklet)
        # "out = B" should appear exactly once (the pre-bind), not twice
        occurrences = code.count("out = B")
        assert occurrences == 1, \
            f"Expected 'out = B' exactly once (pre-bind only), found {occurrences} times:\n{code}"
        # "B = out" (old post-bind format) should NOT appear at all
        assert "B = out" not in code, \
            f"Old post-bind 'B = out' should not appear:\n{code}"

    def test_no_duplicate_postbind_access_node(self):
        """Post-bind for tile-local outputs should appear exactly once."""
        sdfg, state, tasklet = _build_tile_intermediary_sdfg(
            "no_dup_postbind_an")
        code = _generate_tasklet_code(sdfg, state, tasklet)
        # "_tile_B = out" should appear exactly once (post-bind)
        occurrences = code.count("_tile_B = out")
        assert occurrences == 1, \
            f"Expected '_tile_B = out' exactly once, found {occurrences} times:\n{code}"
        # "out = _tile_B" (pre-bind) should NOT appear for tile-local outputs
        assert "out = _tile_B" not in code, \
            f"Pre-bind 'out = _tile_B' should not appear for tile-local outputs:\n{code}"

    def test_same_name_no_prebind(self):
        """When dst_name == src_conn, no pre-bind should be emitted."""
        sdfg = _make_cutile_python_sdfg("same_name_no_prebind")
        sdfg.add_array("A", [32], dace.float64, storage=StorageType.GPU_Global)

        state = sdfg.add_state("main")
        me, mx = state.add_map("cutile_map", {"tile_i": "0:1"},
                               schedule=ScheduleType.CuTile)

        a_in = state.add_read("A")
        a_out = state.add_write("A")

        tasklet = state.add_tasklet("identity", {"inp"}, {"A"},
                                    "A = inp",
                                    language=Language.Python)

        state.add_memlet_path(a_in, me, tasklet,
                              dst_conn="inp",
                              memlet=Memlet(data="A", subset="0:32"))
        state.add_memlet_path(tasklet, mx, a_out,
                              src_conn="A",
                              memlet=Memlet(data="A", subset="0:32"))

        code = _generate_tasklet_code(sdfg, state, tasklet)
        # Should NOT generate "A = A" as either pre-bind or post-bind
        lines = [line.strip() for line in code.split("\n") if line.strip()]
        rebind_lines = [l for l in lines if l == "A = A"]
        assert len(rebind_lines) == 0, \
            f"Unnecessary 'A = A' rebinding found in:\n{code}"

    def test_scatter_pattern_dst_bound_before_body(self):
        """Simulate a scatter expansion tasklet: _dst must be available in the body.

        This is the core bug being fixed: ct.scatter(_dst, ...) needs _dst
        to be bound to the actual array before the tasklet body runs.

        Graph structure (inside CuTile map scope):
            AccessNode(src_arr) -> MapEntry -> AccessNode(_tile_t) -> Tasklet(_src, _mask -> _dst)
            AccessNode(mask_arr) -> MapEntry -> AccessNode(_tile_iter_mask) ->    ^
            Tasklet -> MapExit -> AccessNode(C)
        """
        sdfg = _make_cutile_python_sdfg("scatter_prebind")
        sdfg.add_array("src_arr", [32], dace.float64, storage=StorageType.GPU_Global)
        sdfg.add_array("mask_arr", [32], dace.bool, storage=StorageType.GPU_Global)
        sdfg.add_array("C", [32], dace.float64, storage=StorageType.GPU_Global)
        sdfg.add_array("_tile_t", [32], dace.float64,
                       storage=StorageType.CuTile_Tile, transient=True)
        sdfg.add_array("_tile_iter_mask", [32], dace.bool,
                       storage=StorageType.CuTile_Tile, transient=True)

        state = sdfg.add_state("main")
        me, mx = state.add_map("cutile_map", {"tile_i": "0:1"},
                               schedule=ScheduleType.CuTile)

        src_node = state.add_read("src_arr")
        mask_node = state.add_read("mask_arr")
        tile_t = state.add_access("_tile_t")
        tile_mask = state.add_access("_tile_iter_mask")
        c_node = state.add_write("C")

        # Scatter tasklet: uses _dst (output connector) in the body
        tasklet = state.add_tasklet(
            "scatter_store", {"_src", "_mask"}, {"_dst"},
            "ct.scatter(_dst, (__idx0,), _src, mask=_mask)",
            language=Language.Python)

        # Connect inputs through MapEntry to tile AccessNodes, then to tasklet
        state.add_memlet_path(src_node, me, tile_t,
                              memlet=Memlet(data="src_arr", subset="0:32"))
        state.add_memlet_path(mask_node, me, tile_mask,
                              memlet=Memlet(data="mask_arr", subset="0:32"))
        state.add_edge(tile_t, None, tasklet, "_src",
                       Memlet(data="_tile_t", subset="0:32"))
        state.add_edge(tile_mask, None, tasklet, "_mask",
                       Memlet(data="_tile_iter_mask", subset="0:32"))
        # Connect tasklet output through MapExit to C
        state.add_memlet_path(tasklet, mx, c_node,
                              src_conn="_dst",
                              memlet=Memlet(data="C", subset="0:32"))

        code = _generate_tasklet_code(sdfg, state, tasklet)

        # _dst must be bound to C BEFORE the body
        prebind_pos = code.find("_dst = C")
        body_pos = code.find("ct.scatter(_dst")
        assert prebind_pos != -1, \
            f"Pre-bind '_dst = C' not found in generated code:\n{code}"
        assert body_pos != -1, \
            f"Body 'ct.scatter(_dst' not found in generated code:\n{code}"
        assert prebind_pos < body_pos, \
            (f"Pre-bind must appear BEFORE body. "
             f"Pre-bind at {prebind_pos}, body at {body_pos}:\n{code}")

    def test_multi_output_prebind(self):
        """Multiple output connectors should all be pre-bound."""
        sdfg = _make_cutile_python_sdfg("multi_output_prebind")
        sdfg.add_array("A", [32], dace.float64, storage=StorageType.GPU_Global)
        sdfg.add_array("B", [32], dace.float64, storage=StorageType.GPU_Global)
        sdfg.add_array("C", [32], dace.float64, storage=StorageType.GPU_Global)

        state = sdfg.add_state("main")
        me, mx = state.add_map("cutile_map", {"tile_i": "0:1"},
                               schedule=ScheduleType.CuTile)

        a_node = state.add_read("A")
        b_node = state.add_write("B")
        c_node = state.add_write("C")

        tasklet = state.add_tasklet("split", {"inp"}, {"out1", "out2"},
                                    "out1 = inp * 2\nout2 = inp * 3",
                                    language=Language.Python)

        state.add_memlet_path(a_node, me, tasklet,
                              dst_conn="inp",
                              memlet=Memlet(data="A", subset="0:32"))
        state.add_memlet_path(tasklet, mx, b_node,
                              src_conn="out1",
                              memlet=Memlet(data="B", subset="0:32"))
        state.add_memlet_path(tasklet, mx, c_node,
                              src_conn="out2",
                              memlet=Memlet(data="C", subset="0:32"))

        code = _generate_tasklet_code(sdfg, state, tasklet)
        # Both outputs should be pre-bound before body
        body_marker = "Tasklet: split"
        body_pos = code.find(body_marker)
        assert body_pos != -1

        out1_pos = code.find("out1 = B")
        out2_pos = code.find("out2 = C")
        assert out1_pos != -1, f"Pre-bind 'out1 = B' not found:\n{code}"
        assert out2_pos != -1, f"Pre-bind 'out2 = C' not found:\n{code}"
        assert out1_pos < body_pos, \
            f"Pre-bind 'out1 = B' should appear before body:\n{code}"
        assert out2_pos < body_pos, \
            f"Pre-bind 'out2 = C' should appear before body:\n{code}"

        # No old-style post-binds
        assert "B = out1" not in code
        assert "C = out2" not in code


class TestNonPythonTaskletRejection:
    """Test that non-Python tasklets raise NotImplementedError."""

    def test_cpp_tasklet_raises(self):
        """A C++ tasklet inside CuTile scope should raise NotImplementedError."""
        sdfg, state, tasklet, _, _ = _build_direct_scope_sdfg("cpp_tasklet")
        # Change the tasklet language to C++
        tasklet.code = dace.properties.CodeBlock("out = inp * 2.0",
                                                 language=Language.CPP)
        codegen = _make_cutile_codegen(sdfg)
        function_stream = PythonCodeIOStream()
        callsite_stream = PythonCodeIOStream()
        state_id = sdfg.node_id(state)
        with pytest.raises(NotImplementedError,
                           match="CuTile backend only supports Python tasklets"):
            codegen._generate_Tasklet(
                sdfg, sdfg, state, state_id, tasklet,
                function_stream, callsite_stream)
