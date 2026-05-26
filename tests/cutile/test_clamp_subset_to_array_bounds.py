"""Unit and integration tests for Min-clamped array-bound subsets in ScalarToTileMasked.

The ``_clamp_subset_to_array_bounds`` helper clamps each dimension's upper
bound of a memlet subset to ``Min(unclamped_upper, array_dim_size - 1)``.
This prevents DaCe validation from flagging out-of-bounds memlets when
tile shapes exceed the remaining array extent at boundary tiles.

Tests cover:
  - Direct unit tests of ``_clamp_subset_to_array_bounds``
  - Integration tests verifying that ``_build_memlet`` and
    ``_add_output_preload`` produce Min-clamped subsets after the
    full cuTile pipeline is applied to non-canonical (masked) SDFGs.
"""

import sympy as sp
import pytest

import dace
from dace import dtypes, Memlet, subsets
from dace.sdfg import SDFG, nodes
from dace.libraries.cutile.transformations.pipeline import apply_cutile_pipeline
from dace.libraries.cutile.transformations.scalar_to_tile_library import (
    ScalarToTileMasked,
)
from dace.libraries.cutile.nodes import TileSymbolicMaskedOpLibraryNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_skewed_masked_sdfg_1d(
    array_size: int,
    inner_end: int,
    inner_step: int,
    outer_step: int,
    name: str = "test_skewed_clamp_1d",
) -> SDFG:
    """Build a 2D SDFG with a skewed inner map (start=0).

    Skewed maps are produced by MapTiling(skew=True) and always start at 0.
    The outer step encodes tile_size * abs(inner_step), allowing the
    override in _calculate_tile_shape to recover the original tile_size.

    Arrays have shape ``[outer_tiles, array_size]``.  The tile shape
    derived from the override is ``outer_step / abs(inner_step)``,
    which can exceed ``array_size`` -- triggering Min-clamping.
    """
    outer_tiles = 2
    sdfg = SDFG(name)
    sdfg.add_array("A", shape=[outer_tiles, array_size], dtype=dace.float64)
    sdfg.add_array("B", shape=[outer_tiles, array_size], dtype=dace.float64)
    sdfg.add_array("C", shape=[outer_tiles, array_size], dtype=dace.float64)

    state = sdfg.add_state("main")
    a_acc = state.add_read("A")
    b_acc = state.add_read("B")
    c_acc = state.add_write("C")

    outer_entry, outer_exit = state.add_map(
        "tile_map",
        {"i": f"0:{outer_tiles}"},
        schedule=dtypes.ScheduleType.Sequential,
    )
    # Override the outer step for skew detection.
    outer_entry.map.range[0] = (
        outer_entry.map.range[0][0],
        outer_entry.map.range[0][1],
        outer_step,
    )

    inner_entry, inner_exit = state.add_map(
        "elem_map",
        {"ii": f"0:{inner_end}:{inner_step}"},
        schedule=dtypes.ScheduleType.Sequential,
    )

    tasklet = state.add_tasklet("add", {"a", "b"}, {"c"}, "c = a + b")

    state.add_memlet_path(
        a_acc, outer_entry, inner_entry, tasklet,
        dst_conn="a", memlet=Memlet("A[i, ii]"),
    )
    state.add_memlet_path(
        b_acc, outer_entry, inner_entry, tasklet,
        dst_conn="b", memlet=Memlet("B[i, ii]"),
    )
    state.add_memlet_path(
        tasklet, inner_exit, outer_exit, c_acc,
        src_conn="c", memlet=Memlet("C[i, ii]"),
    )

    sdfg.validate()
    return sdfg


def _build_noncanonical_masked_sdfg_2d(
    shape: tuple[int, int],
    ii_range: str,
    jj_range: str,
    name: str = "test_clamp_2d",
) -> SDFG:
    """Build a 4D SDFG with non-canonical inner maps in two dimensions.

    Arrays have shape ``[2, 2, shape[0], shape[1]]``.  Inner map ranges
    are given as strings (e.g. ``"1:6:2"``, ``"0:4"``).
    """
    mt, nt = 2, 2
    t0, t1 = shape

    sdfg = SDFG(name)
    sdfg.add_array("A", shape=[mt, nt, t0, t1], dtype=dace.float64)
    sdfg.add_array("B", shape=[mt, nt, t0, t1], dtype=dace.float64)
    sdfg.add_array("C", shape=[mt, nt, t0, t1], dtype=dace.float64)

    state = sdfg.add_state("main")
    a_acc = state.add_read("A")
    b_acc = state.add_read("B")
    c_acc = state.add_write("C")

    outer_entry, outer_exit = state.add_map(
        "tile_map",
        {"i": f"0:{mt}", "j": f"0:{nt}"},
        schedule=dtypes.ScheduleType.Sequential,
    )
    inner_entry, inner_exit = state.add_map(
        "elem_map",
        {"ii": ii_range, "jj": jj_range},
        schedule=dtypes.ScheduleType.Sequential,
    )

    tasklet = state.add_tasklet("add", {"a", "b"}, {"c"}, "c = a + b")

    state.add_memlet_path(
        a_acc, outer_entry, inner_entry, tasklet,
        dst_conn="a", memlet=Memlet("A[i, j, ii, jj]"),
    )
    state.add_memlet_path(
        b_acc, outer_entry, inner_entry, tasklet,
        dst_conn="b", memlet=Memlet("B[i, j, ii, jj]"),
    )
    state.add_memlet_path(
        tasklet, inner_exit, outer_exit, c_acc,
        src_conn="c", memlet=Memlet("C[i, j, ii, jj]"),
    )

    sdfg.validate()
    return sdfg


def _build_symbolic_masked_sdfg(
    name: str = "test_clamp_symbolic",
) -> SDFG:
    """Build a masked SDFG with symbolic array sizes.

    Arrays have shape ``[2, N]``.  Inner map range is ``1:N:2`` (offset,
    strided), which produces a tile shape of ``N`` (from Max(1,N-1)+1).
    Since the array's second dimension is also ``N``, the upper bound
    ``N - 1`` is clamped to ``Min(N - 1, N - 1) = N - 1``.  This tests
    that symbolic sizes produce Min expressions before simplification.
    """
    N = dace.symbol("N", dtype=dace.int32)
    sdfg = SDFG(name)
    sdfg.add_symbol("N", dace.int32)
    sdfg.add_array("A", shape=[2, N], dtype=dace.float64)
    sdfg.add_array("B", shape=[2, N], dtype=dace.float64)
    sdfg.add_array("C", shape=[2, N], dtype=dace.float64)

    state = sdfg.add_state("main")
    a_acc = state.add_read("A")
    b_acc = state.add_read("B")
    c_acc = state.add_write("C")

    outer_entry, outer_exit = state.add_map(
        "tile_map", {"i": "0:2"},
        schedule=dtypes.ScheduleType.Sequential,
    )
    inner_entry, inner_exit = state.add_map(
        "elem_map", {"ii": f"1:{N}:2"},
        schedule=dtypes.ScheduleType.Sequential,
    )

    tasklet = state.add_tasklet("add", {"a", "b"}, {"c"}, "c = a + b")

    state.add_memlet_path(
        a_acc, outer_entry, inner_entry, tasklet,
        dst_conn="a", memlet=Memlet("A[i, ii]"),
    )
    state.add_memlet_path(
        b_acc, outer_entry, inner_entry, tasklet,
        dst_conn="b", memlet=Memlet("B[i, ii]"),
    )
    state.add_memlet_path(
        tasklet, inner_exit, outer_exit, c_acc,
        src_conn="c", memlet=Memlet("C[i, ii]"),
    )

    sdfg.validate()
    return sdfg


def _find_staging_memlet_subsets(sdfg: SDFG) -> list[subsets.Range]:
    """Find memlet subsets on edges from outer MapEntry to tile transient AccessNodes.

    After the ScalarToTileMasked transformation, staging memlets go from
    the outer MapEntry to tile transient AccessNodes.  The edge's data.data
    refers to the global array (not the transient), and the destination
    is a transient AccessNode.
    """
    result = []
    for state in sdfg.states():
        for edge in state.edges():
            if (isinstance(edge.src, nodes.MapEntry)
                    and isinstance(edge.dst, nodes.AccessNode)
                    and edge.data.subset is not None):
                # Check if destination is a tile transient
                dst_desc = sdfg.arrays.get(edge.dst.data)
                if dst_desc is not None and dst_desc.transient:
                    result.append(edge.data.subset)
    return result


def _find_store_memlet_subsets(sdfg: SDFG) -> list[subsets.Range]:
    """Find memlet subsets on edges from tile transient AccessNodes to outer MapExit.

    After transformation, store memlets go from tile transient AccessNodes
    to the outer MapExit.  The edge's data.data refers to the global array.
    """
    result = []
    for state in sdfg.states():
        for edge in state.edges():
            if (isinstance(edge.src, nodes.AccessNode)
                    and isinstance(edge.dst, nodes.MapExit)
                    and edge.data.subset is not None):
                # Check if source is a tile transient
                src_desc = sdfg.arrays.get(edge.src.data)
                if src_desc is not None and src_desc.transient:
                    result.append(edge.data.subset)
    return result


def _find_preload_memlet_subsets(sdfg: SDFG) -> list[subsets.Range]:
    """Find preload memlet subsets (global array -> outer_entry with preload connector)."""
    result = []
    for state in sdfg.states():
        for edge in state.edges():
            if (isinstance(edge.src, nodes.AccessNode)
                    and isinstance(edge.dst, nodes.MapEntry)
                    and edge.data.subset is not None):
                if edge.dst_conn and "preload" in edge.dst_conn.lower():
                    result.append(edge.data.subset)
    return result


# ---------------------------------------------------------------------------
# Unit tests: _clamp_subset_to_array_bounds
# ---------------------------------------------------------------------------


class TestClampSubsetToArrayBounds:
    """Direct tests for ScalarToTileMasked._clamp_subset_to_array_bounds."""

    def _make_instance(self, sdfg: SDFG, state) -> ScalarToTileMasked:
        """Create a ScalarToTileMasked instance with _sdfg set."""
        instance = ScalarToTileMasked.__new__(ScalarToTileMasked)
        instance._sdfg = sdfg
        instance._graph = state
        return instance

    def test_clamp_concrete_within_bounds(self):
        """When upper bound is within array size, no Min is needed."""
        sdfg = SDFG("test_within_bounds")
        sdfg.add_array("X", shape=[100], dtype=dace.float64)
        state = sdfg.add_state("s")

        inst = self._make_instance(sdfg, state)
        original = subsets.Range([(0, 50, 1)])
        result = inst._clamp_subset_to_array_bounds(original, "X")

        # Min(50, 99) should simplify to 50
        _, high, _ = result[0]
        assert int(sp.sympify(high)) == 50

    def test_clamp_concrete_exceeds_bounds(self):
        """When upper bound exceeds array size, Min clamps it."""
        sdfg = SDFG("test_exceeds_bounds")
        sdfg.add_array("X", shape=[10], dtype=dace.float64)
        state = sdfg.add_state("s")

        inst = self._make_instance(sdfg, state)
        original = subsets.Range([(0, 15, 1)])
        result = inst._clamp_subset_to_array_bounds(original, "X")

        # Min(15, 9) should simplify to 9
        _, high, _ = result[0]
        assert int(sp.sympify(high)) == 9

    def test_clamp_exact_bounds(self):
        """When upper bound equals array_size - 1, no change needed."""
        sdfg = SDFG("test_exact_bounds")
        sdfg.add_array("X", shape=[10], dtype=dace.float64)
        state = sdfg.add_state("s")

        inst = self._make_instance(sdfg, state)
        original = subsets.Range([(0, 9, 1)])
        result = inst._clamp_subset_to_array_bounds(original, "X")

        _, high, _ = result[0]
        assert int(sp.sympify(high)) == 9

    def test_clamp_symbolic_upper_bound(self):
        """Symbolic upper bounds should produce Min(symbolic, array_max)."""
        sdfg = SDFG("test_symbolic")
        sdfg.add_array("X", shape=[10], dtype=dace.float64)
        state = sdfg.add_state("s")

        inst = self._make_instance(sdfg, state)
        tile_i = sp.Symbol("tile_i")
        original = subsets.Range([(tile_i, tile_i + 15, 1)])
        result = inst._clamp_subset_to_array_bounds(original, "X")

        _, high, _ = result[0]
        high_expr = sp.sympify(high)
        assert high_expr.has(sp.Min), f"Expected Min in {high_expr}"
        # Should be Min(tile_i + 15, 9)
        assert high_expr == sp.Min(tile_i + 15, 9)

    def test_clamp_symbolic_array_size(self):
        """Symbolic array sizes produce Min(upper, N - 1)."""
        N = dace.symbol("N", dtype=dace.int32)
        sdfg = SDFG("test_symbolic_array")
        sdfg.add_symbol("N", dace.int32)
        sdfg.add_array("X", shape=[N], dtype=dace.float64)
        state = sdfg.add_state("s")

        inst = self._make_instance(sdfg, state)
        original = subsets.Range([(0, 15, 1)])
        result = inst._clamp_subset_to_array_bounds(original, "X")

        _, high, _ = result[0]
        high_expr = sp.sympify(high)
        assert high_expr.has(sp.Min), f"Expected Min in {high_expr}"

    def test_clamp_multidimensional(self):
        """Each dimension is clamped independently."""
        sdfg = SDFG("test_multidim")
        sdfg.add_array("X", shape=[10, 20], dtype=dace.float64)
        state = sdfg.add_state("s")

        inst = self._make_instance(sdfg, state)
        # dim 0: 15 > 9 (clamped), dim 1: 5 < 19 (not clamped)
        original = subsets.Range([(0, 15, 1), (0, 5, 1)])
        result = inst._clamp_subset_to_array_bounds(original, "X")

        _, high0, _ = result[0]
        _, high1, _ = result[1]
        assert int(sp.sympify(high0)) == 9
        assert int(sp.sympify(high1)) == 5

    def test_clamp_preserves_low_and_step(self):
        """Clamping only modifies upper bounds; low and step are unchanged."""
        sdfg = SDFG("test_preserve")
        sdfg.add_array("X", shape=[10], dtype=dace.float64)
        state = sdfg.add_state("s")

        inst = self._make_instance(sdfg, state)
        original = subsets.Range([(3, 20, 2)])
        result = inst._clamp_subset_to_array_bounds(original, "X")

        low, high, step = result[0]
        assert int(sp.sympify(low)) == 3
        assert int(sp.sympify(high)) == 9  # Min(20, 9) = 9
        assert int(sp.sympify(step)) == 2

    def test_clamp_unknown_array_returns_unchanged(self):
        """If data_name is not in sdfg.arrays, return the original subset."""
        sdfg = SDFG("test_unknown")
        state = sdfg.add_state("s")

        inst = self._make_instance(sdfg, state)
        original = subsets.Range([(0, 100, 1)])
        result = inst._clamp_subset_to_array_bounds(original, "nonexistent")

        assert result is original  # Should be the same object

    def test_clamp_more_dims_than_array(self):
        """Subset dimensions beyond array shape are left unclamped."""
        sdfg = SDFG("test_extra_dims")
        sdfg.add_array("X", shape=[10], dtype=dace.float64)
        state = sdfg.add_state("s")

        inst = self._make_instance(sdfg, state)
        # 2D subset on 1D array: first dim clamped, second untouched
        original = subsets.Range([(0, 20, 1), (0, 50, 1)])
        result = inst._clamp_subset_to_array_bounds(original, "X")

        _, high0, _ = result[0]
        _, high1, _ = result[1]
        assert int(sp.sympify(high0)) == 9   # clamped
        assert int(sp.sympify(high1)) == 50   # untouched

    def test_clamp_both_symbolic(self):
        """Both upper bound and array size symbolic produces Min expression."""
        N = dace.symbol("N", dtype=dace.int32)
        sdfg = SDFG("test_both_symbolic")
        sdfg.add_symbol("N", dace.int32)
        sdfg.add_array("X", shape=[N], dtype=dace.float64)
        state = sdfg.add_state("s")

        inst = self._make_instance(sdfg, state)
        tile_i = sp.Symbol("tile_i")
        original = subsets.Range([(0, tile_i + 7, 1)])
        result = inst._clamp_subset_to_array_bounds(original, "X")

        _, high, _ = result[0]
        high_expr = sp.sympify(high)
        assert high_expr.has(sp.Min), f"Expected Min in {high_expr}"


# ---------------------------------------------------------------------------
# Integration tests: Min-clamped subsets after pipeline
# ---------------------------------------------------------------------------


class TestMinClampedMemletsAfterPipeline:
    """Verify that the cuTile pipeline produces correctly clamped memlets
    for masked transformations where tile shape can exceed array bounds."""

    def test_skewed_map_memlets_clamped_to_array_bounds(self):
        """Skewed maps with tile shape > array dim should produce clamped memlets.

        Inner map: 0:7:2 (values {0,2,4,6}), max index 6, array_size=7.
        Base tile shape = Max(0,6)+1 = 7.
        With outer_step=16, override extends to 16/2 = 8.
        Tile shape 8 > array_size 7, so dim 1 upper bound should be
        clamped from 7 (tile_shape-1) to Min(7, 6) = 6.
        """
        sdfg = _build_skewed_masked_sdfg_1d(
            array_size=7,
            inner_end=7,   # DaCe "0:7:2" -> inclusive end=6
            inner_step=2,
            outer_step=16,  # tile_size = 16/2 = 8 > array_size=7
            name="test_skewed_min_clamp",
        )

        count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
        assert count == 1, f"Expected 1 transformation, got {count}"

        # After transformation, staging memlets should exist
        staging_subsets = _find_staging_memlet_subsets(sdfg)
        assert len(staging_subsets) > 0, "Should have staging memlets"

        # The inner-map dimension (dim 1) should be clamped.
        # Unclamped would be 0:7 (tile_shape-1), clamped is 0:6 (array_size-1).
        for subset in staging_subsets:
            _, high, _ = subset[1]  # dim 1 is the inner-map dimension
            high_val = int(sp.sympify(high))
            assert high_val == 6, (
                f"Expected dim 1 upper bound to be clamped to 6 (array_size-1), "
                f"got {high_val}"
            )

    def test_skewed_map_sdfg_validates(self):
        """SDFG with skewed map should pass validation after clamping."""
        sdfg = _build_skewed_masked_sdfg_1d(
            array_size=7,
            inner_end=7,
            inner_step=2,
            outer_step=16,
            name="test_skewed_validates",
        )

        count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
        assert count == 1

        # This should not raise.
        sdfg.validate()

    def test_store_memlets_also_clamped(self):
        """Output store memlets should also be clamped to array bounds."""
        sdfg = _build_skewed_masked_sdfg_1d(
            array_size=7,
            inner_end=7,
            inner_step=2,
            outer_step=16,
            name="test_store_clamp",
        )

        count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
        assert count == 1

        store_subsets = _find_store_memlet_subsets(sdfg)
        assert len(store_subsets) > 0, "Should have store memlets"

        for subset in store_subsets:
            _, high, _ = subset[1]
            high_val = int(sp.sympify(high))
            assert high_val == 6, (
                f"Expected store dim 1 upper bound to be 6 (array_size-1), got {high_val}"
            )

    def test_preload_memlets_clamped(self):
        """Preload memlets (for _c_in) should also be clamped."""
        sdfg = _build_skewed_masked_sdfg_1d(
            array_size=7,
            inner_end=7,
            inner_step=2,
            outer_step=16,
            name="test_preload_clamp",
        )

        count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
        assert count == 1

        preload_subsets = _find_preload_memlet_subsets(sdfg)
        # Masked ops should have preload paths.
        assert len(preload_subsets) > 0, "Should have preload memlets"

        for subset in preload_subsets:
            _, high, _ = subset[1]
            high_val = int(sp.sympify(high))
            assert high_val == 6, (
                f"Expected preload dim 1 upper bound to be 6 (array_size-1), got {high_val}"
            )

    def test_within_bounds_no_unnecessary_min(self):
        """When tile shape fits within array bounds, Min simplifies away.

        Inner map: 0:4:2 (values {0,2}), max index 2, array_size=10.
        Base tile shape = Max(0,2)+1 = 3.  With outer_step=6, override
        gives 6/2 = 3.  Tile shape 3 < array_size 10.
        Min(2, 9) = 2, so sp.Min should not appear in the result.
        """
        sdfg = _build_skewed_masked_sdfg_1d(
            array_size=10,
            inner_end=4,
            inner_step=2,
            outer_step=6,  # tile_size = 6/2 = 3 <= array_size=10
            name="test_within_bounds_no_min",
        )

        count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
        assert count == 1

        staging_subsets = _find_staging_memlet_subsets(sdfg)
        assert len(staging_subsets) > 0

        for subset in staging_subsets:
            _, high, _ = subset[1]
            expr = sp.sympify(high)
            assert not expr.has(sp.Min), (
                f"Expected no Min when tile fits in array, got {expr}"
            )

    def test_noncanonical_offset_stride_2d(self):
        """2D non-canonical inner map validates after clamping.

        Inner map: ii=1:6:2 (values {1,3,5}), jj=0:4 (values {0,1,2,3}).
        Array shape: (6, 5).  Tile shapes: (6, 4).
        dim 2: tile_shape=6 == array_size=6, upper = 5 == 5.  No clamp needed.
        dim 3: tile_shape=4 < array_size=5, upper = 3 < 4.  No clamp needed.
        But the SDFG should still validate.
        """
        sdfg = _build_noncanonical_masked_sdfg_2d(
            shape=(6, 5),
            ii_range="1:6:2",
            jj_range="0:4",
            name="test_2d_offset_stride",
        )

        count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
        assert count == 1

        # Should pass validation.
        sdfg.validate()

    def test_symbolic_array_size_memlets_valid(self):
        """Symbolic array sizes should produce valid memlets after clamping."""
        sdfg = _build_symbolic_masked_sdfg(name="test_symbolic_clamp")

        count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
        assert count == 1

        # Should pass validation.
        sdfg.validate()

    def test_large_tile_extension_clamped(self):
        """Large tile extension (outer_step >> array_size) is properly clamped.

        Inner map: 0:5:3 (values {0,3}), max index 3, array_size=5.
        Base tile shape = Max(0,3)+1 = 4.
        With outer_step=48, override gives 48/3 = 16.
        Tile shape 16 >> array_size 5.
        Upper bound should be clamped from 15 to Min(15, 4) = 4.
        """
        sdfg = _build_skewed_masked_sdfg_1d(
            array_size=5,
            inner_end=5,   # DaCe "0:5:3" -> inclusive end=3
            inner_step=3,
            outer_step=48,  # tile_size = 48/3 = 16 >> array_size=5
            name="test_large_extension_clamp",
        )

        count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
        assert count == 1

        staging_subsets = _find_staging_memlet_subsets(sdfg)
        assert len(staging_subsets) > 0

        for subset in staging_subsets:
            _, high, _ = subset[1]
            high_val = int(sp.sympify(high))
            assert high_val == 4, (
                f"Expected dim 1 upper bound to be 4 (array_size-1), got {high_val}"
            )

        # Must validate.
        sdfg.validate()

    def test_library_node_is_symbolic_masked(self):
        """Verify the transformation produces a TileSymbolicMaskedOpLibraryNode."""
        sdfg = _build_skewed_masked_sdfg_1d(
            array_size=7,
            inner_end=7,
            inner_step=2,
            outer_step=16,
            name="test_lib_node_type",
        )

        count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
        assert count == 1

        state = sdfg.states()[0]
        lib_nodes = [n for n in state.nodes() if isinstance(n, nodes.LibraryNode)]
        assert len(lib_nodes) == 1
        assert isinstance(lib_nodes[0], TileSymbolicMaskedOpLibraryNode)
