"""
Tests for the cuTile transformation pipeline.

Tests the ScalarToTileLibrary transformation that replaces inner maps
with scalar tasklets by cuTile library nodes.
"""
from __future__ import annotations

import dace
import numpy as np
from dace import dtypes, Memlet
from dace.sdfg import SDFG, nodes
from dace.libraries.cutile.transformations.pipeline import apply_cutile_pipeline
from dace.libraries.cutile.transformations.scalar_to_tile_library import ScalarToTileLibrary
from dace.libraries.cutile.nodes.add import TileAdd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

M  = dace.symbol("M",  dtype=dace.int32)
N  = dace.symbol("N",  dtype=dace.int32)
T0 = dace.symbol("T0", dtype=dace.int32)
T1 = dace.symbol("T1", dtype=dace.int32)


def _add_blocked_arrays(sdfg: SDFG, names=("A", "B", "C")):
    """Add 4-D blocked arrays [M//T0, N//T1, T0, T1] to sdfg."""
    tile_shape = [M // T0, N // T1, T0, T1]
    for name in names:
        sdfg.add_array(
            name, shape=tile_shape, dtype=dace.float64,
            storage=dtypes.StorageType.GPU_Global,
        )


def build_tiled_scalar_add_sdfg() -> SDFG:
    """Build an SDFG with tiled scalar add (the 'before' state)."""
    sdfg = SDFG("tile_add_before")
    for sym in ("M", "N", "T0", "T1"):
        sdfg.add_symbol(sym, dace.int32)

    _add_blocked_arrays(sdfg)
    state = sdfg.add_state("main")

    A_acc = state.add_read("A")
    B_acc = state.add_read("B")
    C_acc = state.add_write("C")

    outer_entry, outer_exit = state.add_map(
        "tile_map",
        {"i": "0:M//T0", "j": "0:N//T1"},
        schedule=dtypes.ScheduleType.Sequential,
    )
    inner_entry, inner_exit = state.add_map(
        "elem_map",
        {"ii": "0:T0", "jj": "0:T1"},
        schedule=dtypes.ScheduleType.GPU_ThreadBlock,
    )
    tasklet = state.add_tasklet("add", {"a", "b"}, {"c"}, "c = a + b")

    state.add_memlet_path(
        A_acc, outer_entry, inner_entry, tasklet,
        dst_conn="a", memlet=Memlet("A[i, j, ii, jj]"),
    )
    state.add_memlet_path(
        B_acc, outer_entry, inner_entry, tasklet,
        dst_conn="b", memlet=Memlet("B[i, j, ii, jj]"),
    )
    state.add_memlet_path(
        tasklet, inner_exit, outer_exit, C_acc,
        src_conn="c", memlet=Memlet("C[i, j, ii, jj]"),
    )

    sdfg.validate()
    return sdfg


def build_runtime_tiled_scalar_add_sdfg(
    outer_shape=(2, 3),
    tile_shape=(2, 2),
    dtype=dace.float64,
) -> SDFG:
    """Build a CPU-friendly 'before pipeline' tiled scalar add SDFG."""
    mt, nt = outer_shape
    t0, t1 = tile_shape

    sdfg = SDFG("tile_add_before_runtime")
    sdfg.add_array("A", shape=[mt, nt, t0, t1], dtype=dtype)
    sdfg.add_array("B", shape=[mt, nt, t0, t1], dtype=dtype)
    sdfg.add_array("C", shape=[mt, nt, t0, t1], dtype=dtype)

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
        {"ii": f"0:{t0}", "jj": f"0:{t1}"},
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_scalar_to_tile_add():
    """Transform tiled scalar add → TileAdd library node."""
    sdfg = build_tiled_scalar_add_sdfg()

    count = apply_cutile_pipeline(sdfg, validate=True)
    assert count == 1, f"Expected 1 transformation, got {count}"

    state = sdfg.states()[0]

    # Exactly one library node
    lib_nodes = [n for n in state.nodes() if isinstance(n, nodes.LibraryNode)]
    assert len(lib_nodes) == 1
    assert isinstance(lib_nodes[0], TileAdd)

    # Only the outer map remains (no inner map)
    map_entries = [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]
    assert len(map_entries) == 1, f"Expected 1 map entry, got {len(map_entries)}"

    # Three tile transients created
    transients = {name for name, desc in sdfg.arrays.items() if desc.transient}
    assert len(transients) == 3, f"Expected 3 transients, got {transients}"

    # No tasklets remain
    tasklets = [n for n in state.nodes() if isinstance(n, nodes.Tasklet)]
    assert len(tasklets) == 0

    # Library node has correct connectors
    lib = lib_nodes[0]
    assert set(lib.in_connectors.keys()) == {"_a", "_b"}
    assert set(lib.out_connectors.keys()) == {"_c"}

    # Validate final SDFG
    sdfg.validate()


def test_can_be_applied_rejects_non_scalar():
    """Transformation must NOT match when tasklet accesses are not scalar."""
    sdfg = SDFG("non_scalar")
    for sym in ("M", "N", "T0", "T1"):
        sdfg.add_symbol(sym, dace.int32)
    _add_blocked_arrays(sdfg)
    state = sdfg.add_state("main")

    A_acc = state.add_read("A")
    C_acc = state.add_write("C")

    outer_entry, outer_exit = state.add_map(
        "tile_map", {"i": "0:M//T0", "j": "0:N//T1"},
        schedule=dtypes.ScheduleType.Sequential,
    )
    # Inner map NOT starting at 0 – should be rejected
    inner_entry, inner_exit = state.add_map(
        "elem_map", {"ii": "1:T0", "jj": "0:T1"},
        schedule=dtypes.ScheduleType.GPU_ThreadBlock,
    )
    tasklet = state.add_tasklet("copy", {"a"}, {"c"}, "c = a")

    state.add_memlet_path(
        A_acc, outer_entry, inner_entry, tasklet,
        dst_conn="a", memlet=Memlet("A[i, j, ii, jj]"),
    )
    state.add_memlet_path(
        tasklet, inner_exit, outer_exit, C_acc,
        src_conn="c", memlet=Memlet("C[i, j, ii, jj]"),
    )

    count = sdfg.apply_transformations(ScalarToTileLibrary)
    assert count == 0, "Should not apply to inner map not starting at 0"


def test_can_be_applied_rejects_unknown_op():
    """Transformation must NOT match when tasklet code is not registered."""
    sdfg = SDFG("unknown_op")
    for sym in ("M", "N", "T0", "T1"):
        sdfg.add_symbol(sym, dace.int32)
    _add_blocked_arrays(sdfg)
    state = sdfg.add_state("main")

    A_acc = state.add_read("A")
    B_acc = state.add_read("B")
    C_acc = state.add_write("C")

    outer_entry, outer_exit = state.add_map(
        "tile_map", {"i": "0:M//T0", "j": "0:N//T1"},
        schedule=dtypes.ScheduleType.Sequential,
    )
    inner_entry, inner_exit = state.add_map(
        "elem_map", {"ii": "0:T0", "jj": "0:T1"},
        schedule=dtypes.ScheduleType.GPU_ThreadBlock,
    )
    # Unknown operation – not registered in op registry
    tasklet = state.add_tasklet("unknown", {"a", "b"}, {"c"},
                                "c = a ** b")

    state.add_memlet_path(
        A_acc, outer_entry, inner_entry, tasklet,
        dst_conn="a", memlet=Memlet("A[i, j, ii, jj]"),
    )
    state.add_memlet_path(
        B_acc, outer_entry, inner_entry, tasklet,
        dst_conn="b", memlet=Memlet("B[i, j, ii, jj]"),
    )
    state.add_memlet_path(
        tasklet, inner_exit, outer_exit, C_acc,
        src_conn="c", memlet=Memlet("C[i, j, ii, jj]"),
    )

    count = sdfg.apply_transformations(ScalarToTileLibrary)
    assert count == 0, "Should not apply to unknown operation"


def test_pipeline_idempotent():
    """Running the pipeline twice should not change a transformed SDFG."""
    sdfg = build_tiled_scalar_add_sdfg()

    count1 = apply_cutile_pipeline(sdfg, validate=True)
    assert count1 == 1

    count2 = apply_cutile_pipeline(sdfg, validate=True)
    assert count2 == 0, "Pipeline should be idempotent"


def test_structure_matches_expected():
    """Verify the transformed SDFG has the expected node/edge structure."""
    sdfg = build_tiled_scalar_add_sdfg()
    apply_cutile_pipeline(sdfg, validate=True)

    state = sdfg.states()[0]
    outer_entry = None
    outer_exit = None
    lib_node = None

    for n in state.nodes():
        if isinstance(n, nodes.MapEntry):
            outer_entry = n
        elif isinstance(n, nodes.MapExit):
            outer_exit = n
        elif isinstance(n, nodes.LibraryNode):
            lib_node = n

    assert outer_entry is not None
    assert outer_exit is not None
    assert lib_node is not None

    # Library node should be inside outer map scope
    scope = state.scope_dict()
    assert scope[lib_node] == outer_entry

    # Check transient access nodes are in scope
    trans_nodes = [n for n in state.nodes()
                   if isinstance(n, nodes.AccessNode) and sdfg.arrays[n.data].transient]
    assert len(trans_nodes) == 3
    for tn in trans_nodes:
        assert scope[tn] == outer_entry

    # Library node has inputs from tile transients
    in_edges = state.in_edges(lib_node)
    assert len(in_edges) == 2
    for e in in_edges:
        assert isinstance(e.src, nodes.AccessNode)
        assert sdfg.arrays[e.src.data].transient

    # Library node has output to tile transient
    out_edges = state.out_edges(lib_node)
    assert len(out_edges) == 1
    assert isinstance(out_edges[0].dst, nodes.AccessNode)
    assert sdfg.arrays[out_edges[0].dst.data].transient


def test_tileadd_runtime_numeric_correctness():
    """Execute TileAdd with concrete data and compare against NumPy addition."""
    sdfg = SDFG("tile_add_runtime")
    sdfg.add_array("A", shape=[3, 4], dtype=dace.float64)
    sdfg.add_array("B", shape=[3, 4], dtype=dace.float64)
    sdfg.add_array("C", shape=[3, 4], dtype=dace.float64)

    state = sdfg.add_state("main")
    a_read = state.add_read("A")
    b_read = state.add_read("B")
    c_write = state.add_write("C")
    add_node = TileAdd("tile_add_runtime_node")
    state.add_node(add_node)

    state.add_edge(a_read, None, add_node, "_a", Memlet("A[0:3, 0:4]"))
    state.add_edge(b_read, None, add_node, "_b", Memlet("B[0:3, 0:4]"))
    state.add_edge(add_node, "_c", c_write, None, Memlet("C[0:3, 0:4]"))

    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(1234)
    a = rng.standard_normal((3, 4), dtype=np.float64)
    b = rng.standard_normal((3, 4), dtype=np.float64)
    c = np.zeros((3, 4), dtype=np.float64)

    sdfg(A=a, B=b, C=c)
    sdfg.save("cutile_test_tileadd_runtime.sdfg")

    np.testing.assert_allclose(c, a + b, rtol=0.0, atol=1e-12)


def test_pipeline_runtime_numeric_correctness_float64():
    """End-to-end: before-pipeline SDFG -> pipeline -> run -> numeric check."""
    sdfg = build_runtime_tiled_scalar_add_sdfg(
        outer_shape=(2, 3),
        tile_shape=(2, 2),
        dtype=dace.float64,
    )
    
    sdfg.save("cutile_test_before_pipeline_float64.sdfg")

    count = apply_cutile_pipeline(sdfg, validate=True)
    sdfg.save("cutile_test_after_pipeline_float64.sdfg")
    assert count == 1
    sdfg.expand_library_nodes()
    sdfg.save("cutile_test_after_expansion_float64.sdfg")
    sdfg.validate()

    shape = (2, 3, 2, 2)
    rng = np.random.default_rng(2026)
    a = rng.uniform(-50.0, 50.0, size=shape).astype(np.float64)
    b = rng.uniform(-50.0, 50.0, size=shape).astype(np.float64)
    c = np.zeros(shape, dtype=np.float64)

    sdfg(A=a, B=b, C=c)

    np.testing.assert_allclose(c, a + b, rtol=0.0, atol=1e-12)


def test_pipeline_runtime_numeric_correctness_float32():
    """End-to-end check on a second shape/type configuration."""
    sdfg = build_runtime_tiled_scalar_add_sdfg(
        outer_shape=(3, 2),
        tile_shape=(1, 4),
        dtype=dace.float32,
    )
    
    sdfg.save("cutile_test_before_pipeline_float32.sdfg")

    count = apply_cutile_pipeline(sdfg, validate=True)
    sdfg.save("cutile_test_after_pipeline_float32.sdfg")
    assert count == 1
    sdfg.expand_library_nodes()
    sdfg.save("cutile_test_after_expansion_float32.sdfg")
    sdfg.validate()

    shape = (3, 2, 1, 4)
    rng = np.random.default_rng(7)
    a = rng.uniform(-10.0, 10.0, size=shape).astype(np.float32)
    b = rng.uniform(-10.0, 10.0, size=shape).astype(np.float32)
    c = np.zeros(shape, dtype=np.float32)

    sdfg(A=a, B=b, C=c)

    np.testing.assert_allclose(c, a + b, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    test_scalar_to_tile_add()
    print("[PASS] test_scalar_to_tile_add")

    test_can_be_applied_rejects_non_scalar()
    print("[PASS] test_can_be_applied_rejects_non_scalar")

    test_can_be_applied_rejects_unknown_op()
    print("[PASS] test_can_be_applied_rejects_unknown_op")

    test_pipeline_idempotent()
    print("[PASS] test_pipeline_idempotent")

    test_structure_matches_expected()
    print("[PASS] test_structure_matches_expected")

    test_tileadd_runtime_numeric_correctness()
    print("[PASS] test_tileadd_runtime_numeric_correctness")

    test_pipeline_runtime_numeric_correctness_float64()
    print("[PASS] test_pipeline_runtime_numeric_correctness_float64")

    test_pipeline_runtime_numeric_correctness_float32()
    print("[PASS] test_pipeline_runtime_numeric_correctness_float32")

    print("\nAll tests passed!")
