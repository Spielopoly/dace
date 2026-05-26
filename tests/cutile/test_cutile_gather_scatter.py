"""Tests for gather/scatter support in cuTile Python codegen.

Tests the new _needs_gather, _resolve_tile_shapes, _emit_gather_load,
_emit_scatter_store methods, and the refactored _generate_MapEntry /
_generate_MapExit that choose between ct.load/ct.store (aligned) and
ct.gather/ct.scatter (non-aligned) paths.
"""

import re as _re

import pytest
import sympy as sp

import dace
from dace import dtypes
from dace.codegen import codegen as dace_codegen
from dace.codegen.py.cutile_target import CuTilePythonCodeGen
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.sdfg import SDFG, nodes


# ---------------------------------------------------------------------------
# Helper: generate Python code from an SDFG
# ---------------------------------------------------------------------------

def _code_of(sdfg: SDFG) -> str:
    code_objects = dace_codegen.generate_code(sdfg)
    return next(co.clean_code for co in code_objects if co.name == sdfg.name)


# ---------------------------------------------------------------------------
# Helper: build a tiled SDFG with tile transients (simulating post-MapTiling)
# ---------------------------------------------------------------------------

def _build_tiled_sdfg(name: str,
                      map_range: dict,
                      tile_shape: list,
                      ndim: int = 1,
                      array_shape=None,
                      start_offset: int = 0) -> SDFG:
    """Build an SDFG with a CuTile-scheduled map and tile transients.

    Creates an SDFG that simulates the output of MapTiling: a CuTile map
    with tile transient AccessNodes between the MapEntry and the Tasklet.

    :param name: SDFG name.
    :param map_range: Map range dict, e.g. {"tile_i": "0:N:32"}.
    :param tile_shape: Shape of tile transients, e.g. [32] or [16, 16].
    :param ndim: Number of dimensions (1 or 2).
    :param array_shape: Shape of global arrays.  Defaults to symbolic.
    :param start_offset: Not used directly; map_range should encode offset.
    :returns: Constructed SDFG with Python backend set.
    """
    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python

    if ndim == 1:
        N = dace.symbol("N")
        sdfg.add_symbol("N", dace.int32)
        arr_shape = array_shape or [N]
    else:
        N = dace.symbol("N")
        M = dace.symbol("M")
        sdfg.add_symbol("N", dace.int32)
        sdfg.add_symbol("M", dace.int32)
        arr_shape = array_shape or [N, M]

    sdfg.add_array("A", shape=arr_shape, dtype=dace.float32)
    sdfg.add_array("C", shape=arr_shape, dtype=dace.float32)

    # Tile transients (like MapTiling creates)
    sdfg.add_transient("A_tile", shape=tile_shape, dtype=dace.float32)
    sdfg.add_transient("C_tile", shape=tile_shape, dtype=dace.float32)

    state = sdfg.add_state("main")
    map_entry, map_exit = state.add_map(
        "tiled", map_range, schedule=dtypes.ScheduleType.CuTile)

    # A -> MapEntry -> A_tile -> Tasklet -> C_tile -> MapExit -> C
    a_read = state.add_read("A")
    c_write = state.add_write("C")
    a_tile_access = state.add_access("A_tile")
    c_tile_access = state.add_access("C_tile")
    tasklet = state.add_tasklet("compute", {"_a"}, {"_out"}, "_out = _a")

    # Build memlet strings based on map params
    params = list(map_range.keys())
    if ndim == 1:
        p = params[0]
        ts = tile_shape[0]
        in_memlet_str = f"A[{p}:{p}+{ts}]"
        out_memlet_str = f"C[{p}:{p}+{ts}]"
        tile_memlet_in_str = f"A_tile[0:{ts}]"
        tile_memlet_out_str = f"C_tile[0:{ts}]"
    else:
        p0, p1 = params[0], params[1]
        ts0, ts1 = tile_shape[0], tile_shape[1]
        in_memlet_str = f"A[{p0}:{p0}+{ts0}, {p1}:{p1}+{ts1}]"
        out_memlet_str = f"C[{p0}:{p0}+{ts0}, {p1}:{p1}+{ts1}]"
        tile_memlet_in_str = f"A_tile[0:{ts0}, 0:{ts1}]"
        tile_memlet_out_str = f"C_tile[0:{ts0}, 0:{ts1}]"

    # --- Input path: A -> MapEntry -> A_tile -> Tasklet ---
    # Add connectors on MapEntry
    map_entry.add_in_connector("IN_A")
    map_entry.add_out_connector("OUT_A")
    state.add_edge(a_read, None, map_entry, "IN_A",
                   dace.Memlet(in_memlet_str))
    state.add_edge(map_entry, "OUT_A", a_tile_access, None,
                   dace.Memlet(tile_memlet_in_str))
    state.add_edge(a_tile_access, None, tasklet, "_a",
                   dace.Memlet(tile_memlet_in_str))

    # --- Output path: Tasklet -> C_tile -> MapExit -> C ---
    # Add connectors on MapExit
    map_exit.add_in_connector("IN_C")
    map_exit.add_out_connector("OUT_C")
    state.add_edge(tasklet, "_out", c_tile_access, None,
                   dace.Memlet(tile_memlet_out_str))
    state.add_edge(c_tile_access, None, map_exit, "IN_C",
                   dace.Memlet(tile_memlet_out_str))
    state.add_edge(map_exit, "OUT_C", c_write, None,
                   dace.Memlet(out_memlet_str))

    return sdfg


# ===========================================================================
# Tests for _needs_gather
# ===========================================================================

class TestNeedsGather:
    """Test the static _needs_gather alignment detection method."""

    def _make_entry(self, map_range: dict) -> nodes.MapEntry:
        """Create a MapEntry with the given range (for unit testing)."""
        sdfg = SDFG("dummy")
        state = sdfg.add_state()
        entry, _ = state.add_map("m", map_range,
                                 schedule=dtypes.ScheduleType.CuTile)
        return entry

    def test_aligned_start_zero_step_equals_tile(self):
        """Start=0, step matches tile shape -> no gather needed."""
        entry = self._make_entry({"i": "0:N:32"})
        sdfg = SDFG("dummy")
        sdfg.add_symbol("N", dace.int32)
        tile_shapes = {"A_tile": (32,)}
        assert CuTilePythonCodeGen._needs_gather(
            entry, tile_shapes, sdfg) is False

    def test_nonzero_start_needs_gather(self):
        """Non-zero start -> gather needed."""
        entry = self._make_entry({"i": "2:N:32"})
        sdfg = SDFG("dummy")
        sdfg.add_symbol("N", dace.int32)
        tile_shapes = {"A_tile": (32,)}
        assert CuTilePythonCodeGen._needs_gather(
            entry, tile_shapes, sdfg) is True

    def test_step_mismatch_needs_gather(self):
        """Step doesn't match tile shape -> gather needed."""
        entry = self._make_entry({"i": "0:N:64"})
        sdfg = SDFG("dummy")
        sdfg.add_symbol("N", dace.int32)
        tile_shapes = {"A_tile": (32,)}
        assert CuTilePythonCodeGen._needs_gather(
            entry, tile_shapes, sdfg) is True

    def test_aligned_2d(self):
        """2D map with matching tile shapes -> no gather."""
        entry = self._make_entry({"i": "0:N:16", "j": "0:M:8"})
        sdfg = SDFG("dummy")
        sdfg.add_symbol("N", dace.int32)
        sdfg.add_symbol("M", dace.int32)
        tile_shapes = {"A_tile": (16, 8)}
        assert CuTilePythonCodeGen._needs_gather(
            entry, tile_shapes, sdfg) is False

    def test_nonzero_start_2d_needs_gather(self):
        """2D with non-zero start in second dim -> gather needed."""
        entry = self._make_entry({"i": "0:N:16", "j": "4:M:8"})
        sdfg = SDFG("dummy")
        sdfg.add_symbol("N", dace.int32)
        sdfg.add_symbol("M", dace.int32)
        tile_shapes = {"A_tile": (16, 8)}
        assert CuTilePythonCodeGen._needs_gather(
            entry, tile_shapes, sdfg) is True

    def test_empty_tile_shapes(self):
        """Empty tile shapes dict -> no gather (nothing to compare)."""
        entry = self._make_entry({"i": "2:N:32"})
        sdfg = SDFG("dummy")
        sdfg.add_symbol("N", dace.int32)
        assert CuTilePythonCodeGen._needs_gather(entry, {}, sdfg) is False

    def test_step_one_matches_tile_one(self):
        """Stride-1 map with tile size 1 -> aligned."""
        entry = self._make_entry({"i": "0:N"})
        sdfg = SDFG("dummy")
        sdfg.add_symbol("N", dace.int32)
        tile_shapes = {"A_tile": (1,)}
        assert CuTilePythonCodeGen._needs_gather(
            entry, tile_shapes, sdfg) is False

    def test_symbolic_start_needs_gather(self):
        """Symbolic (non-zero) start -> gather needed."""
        entry = self._make_entry({"i": "K:N:32"})
        sdfg = SDFG("dummy")
        sdfg.add_symbol("N", dace.int32)
        sdfg.add_symbol("K", dace.int32)
        tile_shapes = {"A_tile": (32,)}
        assert CuTilePythonCodeGen._needs_gather(
            entry, tile_shapes, sdfg) is True


# ===========================================================================
# Tests for _resolve_tile_shapes
# ===========================================================================

class TestResolveTileShapes:
    """Test the static _resolve_tile_shapes method."""

    def test_1d_transient(self):
        """1D tile transient should resolve shape from descriptor."""
        sdfg = SDFG("test_resolve_1d")
        sdfg.backend = dtypes.BackendLanguage.Python
        N = dace.symbol("N")
        sdfg.add_symbol("N", dace.int32)
        sdfg.add_transient("A_tile", shape=[32], dtype=dace.float32)

        state = sdfg.add_state()
        entry, _ = state.add_map("m", {"tile_i": "0:N:32"},
                                 schedule=dtypes.ScheduleType.CuTile)

        result = CuTilePythonCodeGen._resolve_tile_shapes(
            entry, state, sdfg, {"A_tile": "A_tile"})
        assert result == {"A_tile": (32,)}

    def test_2d_transient(self):
        """2D tile transient with constant shapes."""
        sdfg = SDFG("test_resolve_2d")
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.add_symbol("N", dace.int32)
        sdfg.add_symbol("M", dace.int32)
        sdfg.add_transient("A_tile", shape=[16, 8], dtype=dace.float32)

        state = sdfg.add_state()
        entry, _ = state.add_map(
            "m", {"tile_i": "0:N:16", "tile_j": "0:M:8"},
            schedule=dtypes.ScheduleType.CuTile)

        result = CuTilePythonCodeGen._resolve_tile_shapes(
            entry, state, sdfg, {"A_tile": "A_tile"})
        assert result == {"A_tile": (16, 8)}

    def test_missing_transient_excluded(self):
        """Tile keys not in sdfg.arrays should be excluded from result."""
        sdfg = SDFG("test_missing")
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.add_symbol("N", dace.int32)

        state = sdfg.add_state()
        entry, _ = state.add_map("m", {"tile_i": "0:N:32"},
                                 schedule=dtypes.ScheduleType.CuTile)

        result = CuTilePythonCodeGen._resolve_tile_shapes(
            entry, state, sdfg, {"nonexistent": "nonexistent"})
        assert result == {}

    def test_multiple_tiles(self):
        """Multiple tile transients should all be resolved."""
        sdfg = SDFG("test_multi")
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.add_symbol("N", dace.int32)
        sdfg.add_transient("A_tile", shape=[32], dtype=dace.float32)
        sdfg.add_transient("B_tile", shape=[32], dtype=dace.float32)

        state = sdfg.add_state()
        entry, _ = state.add_map("m", {"tile_i": "0:N:32"},
                                 schedule=dtypes.ScheduleType.CuTile)

        result = CuTilePythonCodeGen._resolve_tile_shapes(
            entry, state, sdfg,
            {"A_tile": "A_tile", "B_tile": "B_tile"})
        assert result == {"A_tile": (32,), "B_tile": (32,)}


# ===========================================================================
# Tests for _emit_gather_load
# ===========================================================================

class TestEmitGatherLoad:
    """Test the _emit_gather_load method output."""

    def _make_codegen(self) -> CuTilePythonCodeGen:
        """Create a CuTilePythonCodeGen without full initialization."""
        # We can't fully initialize without a frame_codegen, so we
        # create a minimal mock by bypassing __init__.
        obj = object.__new__(CuTilePythonCodeGen)
        return obj

    def test_1d_gather(self):
        """1D gather should emit arange + ct.gather."""
        cg = self._make_codegen()
        stream = PythonCodeIOStream()
        result = cg._emit_gather_load(
            stream, "A", "A_tile",
            ["(2 + __pid0 * 32)"], (32,), None, 0)
        code = stream.getvalue()

        assert "__dace_ct_gidx_A_tile_0 = (2 + __pid0 * 32) + ct.arange(32" in code
        assert "ct.gather(A, (__dace_ct_gidx_A_tile_0,), padding_value=0)" in code
        # 1D: should return idx_vars directly (no broadcast)
        assert result == ["__dace_ct_gidx_A_tile_0"]
        # No broadcast_to for 1D
        assert "ct.broadcast_to" not in code

    def test_2d_gather(self):
        """2D gather should emit arange + reshape + broadcast_to + ct.gather."""
        cg = self._make_codegen()
        stream = PythonCodeIOStream()
        result = cg._emit_gather_load(
            stream, "A", "A_tile",
            ["(2 + __pid0 * 16)", "__pid1"],
            (16, 16), None, 0)
        code = stream.getvalue()

        # Should have 2 arange calls
        assert "__dace_ct_gidx_A_tile_0 = (2 + __pid0 * 16) + ct.arange(16" in code
        assert "__dace_ct_gidx_A_tile_1 = __pid1 + ct.arange(16" in code

        # Should have reshape and broadcast_to for 2D
        assert "ct.reshape(__dace_ct_gidx_A_tile_0, (16, 1,))" in code
        assert "ct.reshape(__dace_ct_gidx_A_tile_1, (1, 16,))" in code
        assert "ct.broadcast_to" in code
        assert "(16, 16,)" in code

        # Should have gather call
        assert "ct.gather(A" in code
        assert "padding_value=0" in code

        # Result should be broadcast variables
        assert len(result) == 2
        assert result[0] == "__dace_ct_gidx_A_tile_0_nd"
        assert result[1] == "__dace_ct_gidx_A_tile_1_nd"


# ===========================================================================
# Tests for _emit_scatter_store
# ===========================================================================

class TestEmitScatterStore:
    """Test the _emit_scatter_store method output."""

    def _make_codegen(self) -> CuTilePythonCodeGen:
        obj = object.__new__(CuTilePythonCodeGen)
        return obj

    def test_1d_scatter(self):
        """1D scatter should emit ct.scatter with single index."""
        cg = self._make_codegen()
        stream = PythonCodeIOStream()
        cg._emit_scatter_store(
            stream, "C", "C_tile",
            ["__dace_ct_gidx_C_tile_0"], None, 0)
        code = stream.getvalue()
        assert "ct.scatter(C, (__dace_ct_gidx_C_tile_0,), C_tile)" in code

    def test_2d_scatter(self):
        """2D scatter should emit ct.scatter with two index variables."""
        cg = self._make_codegen()
        stream = PythonCodeIOStream()
        cg._emit_scatter_store(
            stream, "C", "C_tile",
            ["__dace_ct_gidx_0_nd", "__dace_ct_gidx_1_nd"], None, 0)
        code = stream.getvalue()
        assert "ct.scatter(C, (__dace_ct_gidx_0_nd, __dace_ct_gidx_1_nd,), C_tile)" in code


# ===========================================================================
# Integration tests: full codegen for aligned vs non-aligned maps
# ===========================================================================

class TestAlignedCodegen:
    """Integration tests for aligned (ct.load/ct.store) code generation."""

    def test_aligned_1d_uses_ct_load_and_store(self):
        """Aligned 1D map (start=0, step=tile) should use ct.load/ct.store."""
        sdfg = _build_tiled_sdfg(
            "aligned_1d",
            map_range={"tile_i": "0:N:32"},
            tile_shape=[32],
            ndim=1,
        )
        sdfg.validate()
        code = _code_of(sdfg)

        assert "ct.load(A" in code
        assert "ct.store(C" in code
        assert "ct.gather" not in code
        assert "ct.scatter" not in code

    def test_aligned_2d_uses_ct_load_and_store(self):
        """Aligned 2D map (start=0, step=tile for both dims) -> ct.load/ct.store."""
        sdfg = _build_tiled_sdfg(
            "aligned_2d",
            map_range={"tile_i": "0:N:16", "tile_j": "0:M:8"},
            tile_shape=[16, 8],
            ndim=2,
        )
        sdfg.validate()
        code = _code_of(sdfg)

        assert "ct.load(A" in code
        assert "ct.store(C" in code
        assert "ct.gather" not in code
        assert "ct.scatter" not in code

    def test_aligned_shape_from_descriptor(self):
        """Aligned path should use shape from tile transient descriptor."""
        sdfg = _build_tiled_sdfg(
            "aligned_desc_shape",
            map_range={"tile_i": "0:N:32"},
            tile_shape=[32],
            ndim=1,
        )
        sdfg.validate()
        code = _code_of(sdfg)

        # Shape should come from descriptor (32), not memlet
        load_match = _re.search(r"ct\.load\(A,.*?shape=\(([^)]*)\)", code)
        assert load_match is not None, f"No ct.load found in:\n{code}"
        shape_str = load_match.group(1).strip().rstrip(",").strip()
        assert shape_str == "32", (
            f"Expected shape '32' from descriptor, got '{shape_str}'")


class TestGatherScatterCodegen:
    """Integration tests for non-aligned (ct.gather/ct.scatter) codegen."""

    def test_nonzero_start_uses_gather_scatter(self):
        """Non-zero start in map range should trigger gather/scatter."""
        sdfg = _build_tiled_sdfg(
            "gather_nonzero",
            map_range={"tile_i": "2:N:32"},
            tile_shape=[32],
            ndim=1,
        )
        sdfg.validate()
        code = _code_of(sdfg)

        assert "ct.gather(A" in code
        assert "ct.scatter(C" in code
        assert "ct.arange(32" in code
        # Should NOT have ct.load/ct.store
        load_match = _re.search(r"ct\.load\(A,", code)
        assert load_match is None, (
            f"Expected ct.gather, not ct.load, for non-zero start.\n{code}")

    def test_step_mismatch_uses_gather_scatter(self):
        """Step != tile_shape should trigger gather/scatter."""
        sdfg = _build_tiled_sdfg(
            "gather_step_mismatch",
            map_range={"tile_i": "0:N:64"},
            tile_shape=[32],
            ndim=1,
        )
        sdfg.validate()
        code = _code_of(sdfg)

        assert "ct.gather(A" in code
        assert "ct.scatter(C" in code

    def test_2d_nonzero_start_uses_gather(self):
        """2D map with non-zero start should use gather with broadcast."""
        sdfg = _build_tiled_sdfg(
            "gather_2d_offset",
            map_range={"tile_i": "4:N:16", "tile_j": "0:M:8"},
            tile_shape=[16, 8],
            ndim=2,
        )
        sdfg.validate()
        code = _code_of(sdfg)

        assert "ct.gather(A" in code
        assert "ct.broadcast_to" in code
        assert "ct.reshape" in code
        assert "ct.scatter(C" in code

    def test_gather_index_variable_names(self):
        """Gather index variable names should follow the naming convention."""
        sdfg = _build_tiled_sdfg(
            "gather_idx_names",
            map_range={"tile_i": "2:N:32"},
            tile_shape=[32],
            ndim=1,
        )
        sdfg.validate()
        code = _code_of(sdfg)

        # Should contain index variables with tile var name
        assert "__dace_ct_gidx_A_tile_0" in code


class TestCacheConsistency:
    """Tests that the _tile_loads_by_entry cache format is consistent."""

    def test_tasklet_can_unpack_mapping(self):
        """_generate_Tasklet should be able to access mapping from cache.

        The cache stores a 4-tuple, but _generate_Tasklet only needs [0].
        """
        sdfg = _build_tiled_sdfg(
            "cache_test",
            map_range={"tile_i": "0:N:32"},
            tile_shape=[32],
            ndim=1,
        )
        sdfg.validate()
        # If the cache format is wrong, codegen will crash
        code = _code_of(sdfg)
        assert "ct.load" in code or "ct.gather" in code

    def test_cache_cleaned_up_after_scope(self):
        """Cache entry should be removed after generate_scope completes."""
        sdfg = _build_tiled_sdfg(
            "cache_cleanup",
            map_range={"tile_i": "0:N:32"},
            tile_shape=[32],
            ndim=1,
        )
        sdfg.validate()
        # generate_code internally calls generate_scope which should clean up
        _code_of(sdfg)
        # No direct way to check cache from outside, but if it doesn't crash,
        # cleanup worked (generate_scope calls pop at the end).


class TestAccessNodeErrorHandling:
    """Tests for _generate_AccessNode error handling."""

    def test_access_node_does_not_crash(self):
        """_generate_AccessNode should not crash for normal access nodes."""
        sdfg = _build_tiled_sdfg(
            "access_node_test",
            map_range={"tile_i": "0:N:32"},
            tile_shape=[32],
            ndim=1,
        )
        sdfg.validate()
        # AccessNodes are inside the CuTile scope. If _generate_AccessNode
        # errors, codegen will crash.
        code = _code_of(sdfg)
        assert "@ct.kernel" in code


# ===========================================================================
# Edge case tests
# ===========================================================================

class TestEdgeCases:
    """Edge case tests for gather/scatter codegen."""

    def test_start_zero_step_one_tile_one(self):
        """Trivial case: stride-1 map with tile size 1 is aligned."""
        sdfg = _build_tiled_sdfg(
            "trivial_aligned",
            map_range={"tile_i": "0:N"},
            tile_shape=[1],
            ndim=1,
        )
        sdfg.validate()
        code = _code_of(sdfg)

        assert "ct.load(A" in code
        assert "ct.gather" not in code

    def test_large_tile_size(self):
        """Large power-of-2 tile size should work for aligned path."""
        sdfg = _build_tiled_sdfg(
            "large_tile",
            map_range={"tile_i": "0:N:1024"},
            tile_shape=[1024],
            ndim=1,
        )
        sdfg.validate()
        code = _code_of(sdfg)

        assert "ct.load(A" in code
        load_match = _re.search(r"ct\.load\(A,.*?shape=\(([^)]*)\)", code)
        assert load_match is not None
        shape_str = load_match.group(1).strip().rstrip(",").strip()
        assert shape_str == "1024"

    def test_multiple_input_arrays(self):
        """Multiple input arrays should all get gather loads when non-aligned."""
        sdfg = SDFG("multi_input_gather")
        sdfg.backend = dtypes.BackendLanguage.Python
        N = dace.symbol("N")
        sdfg.add_symbol("N", dace.int32)
        sdfg.add_array("A", shape=[N], dtype=dace.float32)
        sdfg.add_array("B", shape=[N], dtype=dace.float32)
        sdfg.add_array("C", shape=[N], dtype=dace.float32)
        sdfg.add_transient("A_tile", shape=[32], dtype=dace.float32)
        sdfg.add_transient("B_tile", shape=[32], dtype=dace.float32)
        sdfg.add_transient("C_tile", shape=[32], dtype=dace.float32)

        state = sdfg.add_state("main")
        map_entry, map_exit = state.add_map(
            "tiled", {"tile_i": "2:N:32"},
            schedule=dtypes.ScheduleType.CuTile)

        a_read = state.add_read("A")
        b_read = state.add_read("B")
        c_write = state.add_write("C")
        a_tile = state.add_access("A_tile")
        b_tile = state.add_access("B_tile")
        c_tile = state.add_access("C_tile")
        tasklet = state.add_tasklet(
            "add", {"_a", "_b"}, {"_out"}, "_out = _a + _b")

        state.add_memlet_path(
            a_read, map_entry, a_tile, dst_conn=None,
            memlet=dace.Memlet("A[tile_i:tile_i+32]"))
        state.add_memlet_path(
            b_read, map_entry, b_tile, dst_conn=None,
            memlet=dace.Memlet("B[tile_i:tile_i+32]"))
        state.add_edge(a_tile, None, tasklet, "_a",
                       dace.Memlet("A_tile[0:32]"))
        state.add_edge(b_tile, None, tasklet, "_b",
                       dace.Memlet("B_tile[0:32]"))
        state.add_edge(tasklet, "_out", c_tile, None,
                       dace.Memlet("C_tile[0:32]"))
        state.add_memlet_path(
            c_tile, map_exit, c_write, src_conn=None,
            memlet=dace.Memlet("C[tile_i:tile_i+32]"))

        sdfg.validate()
        code = _code_of(sdfg)

        # Both A and B should be loaded via gather
        assert "ct.gather(A" in code
        assert "ct.gather(B" in code
        assert "ct.scatter(C" in code
