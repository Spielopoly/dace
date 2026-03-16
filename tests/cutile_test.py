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
from dace.libraries.cutile.nodes import *


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
    print(f"Applied {count} transformations.")
    assert count == 1, f"Expected 1 transformation, got {count}"

    state = sdfg.states()[0]

    # Exactly one library node
    lib_nodes = [n for n in state.nodes() if isinstance(n, nodes.LibraryNode)]
    assert len(lib_nodes) == 1
    assert isinstance(lib_nodes[0], TileAddLibraryNode)

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
    add_node = TileAddLibraryNode("tile_add_runtime_node")
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


def test_tileadd_runtime_write_mask_numeric_correctness():
    """Execute TileAdd with write-mask and verify masked-out elements remain unchanged."""
    sdfg = SDFG("tile_add_runtime_write_mask")
    sdfg.add_array("A", shape=[3, 4], dtype=dace.float64)
    sdfg.add_array("B", shape=[3, 4], dtype=dace.float64)
    sdfg.add_array("C", shape=[3, 4], dtype=dace.float64)

    state = sdfg.add_state("main")
    a_read = state.add_read("A")
    b_read = state.add_read("B")
    c_write = state.add_write("C")
    add_node = TileAddLibraryNode(
        "tile_add_runtime_masked_node",
        write_mask="__i0 < 2",
    )
    state.add_node(add_node)

    state.add_edge(a_read, None, add_node, "_a", Memlet("A[0:3, 0:4]"))
    state.add_edge(b_read, None, add_node, "_b", Memlet("B[0:3, 0:4]"))
    state.add_edge(add_node, "_c", c_write, None, Memlet("C[0:3, 0:4]"))

    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(4321)
    a = rng.standard_normal((3, 4), dtype=np.float64)
    b = rng.standard_normal((3, 4), dtype=np.float64)
    c = np.full((3, 4), 17.0, dtype=np.float64)

    sdfg(A=a, B=b, C=c)

    expected = np.full((3, 4), 17.0, dtype=np.float64)
    expected[:2, :] = a[:2, :] + b[:2, :]
    np.testing.assert_allclose(c, expected, rtol=0.0, atol=1e-12)


def test_tileadd_runtime_write_mask_multidim_condition():
    """Write-mask with a combined 2-D condition should only update selected coordinates."""
    shape = (4, 5)
    sdfg = SDFG("tile_add_runtime_write_mask_multidim")
    sdfg.add_array("A", shape=list(shape), dtype=dace.float64)
    sdfg.add_array("B", shape=list(shape), dtype=dace.float64)
    sdfg.add_array("C", shape=list(shape), dtype=dace.float64)

    state = sdfg.add_state("main")
    add_node = TileAddLibraryNode(
        "tile_add_runtime_masked_multidim",
        write_mask="(__i0 < 3) and ((__i1 % 2) == 0)",
    )
    state.add_node(add_node)
    state.add_edge(state.add_read("A"), None, add_node, "_a", Memlet("A[0:4, 0:5]"))
    state.add_edge(state.add_read("B"), None, add_node, "_b", Memlet("B[0:4, 0:5]"))
    state.add_edge(add_node, "_c", state.add_write("C"), None, Memlet("C[0:4, 0:5]"))

    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(9901)
    a = rng.standard_normal(shape, dtype=np.float64)
    b = rng.standard_normal(shape, dtype=np.float64)
    c = np.full(shape, -7.0, dtype=np.float64)

    sdfg(A=a, B=b, C=c)

    mask = np.fromfunction(lambda i, j: (i < 3) & ((j % 2) == 0), shape, dtype=int)
    expected = np.full(shape, -7.0, dtype=np.float64)
    expected[mask] = (a + b)[mask]
    np.testing.assert_allclose(c, expected, rtol=0.0, atol=1e-12)


def test_tileadd_runtime_write_mask_arbitrary_index_names():
    """Write-mask should support arbitrary symbolic index names mapped via mask_indices."""
    shape = (3, 4)
    sdfg = SDFG("tile_add_runtime_write_mask_aliases")
    sdfg.add_array("A", shape=list(shape), dtype=dace.float64)
    sdfg.add_array("B", shape=list(shape), dtype=dace.float64)
    sdfg.add_array("C", shape=list(shape), dtype=dace.float64)

    state = sdfg.add_state("main")
    add_node = TileAddLibraryNode(
        "tile_add_runtime_masked_aliases",
        write_mask="((row + 2 * col) % 3) == 1",
        mask_indices=["row", "col"],
    )
    state.add_node(add_node)
    state.add_edge(state.add_read("A"), None, add_node, "_a", Memlet("A[0:3, 0:4]"))
    state.add_edge(state.add_read("B"), None, add_node, "_b", Memlet("B[0:3, 0:4]"))
    state.add_edge(add_node, "_c", state.add_write("C"), None, Memlet("C[0:3, 0:4]"))

    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(9902)
    a = rng.standard_normal(shape, dtype=np.float64)
    b = rng.standard_normal(shape, dtype=np.float64)
    c = np.full(shape, 13.0, dtype=np.float64)

    sdfg(A=a, B=b, C=c)

    mask = np.fromfunction(lambda i, j: ((i + 2 * j) % 3) == 1, shape, dtype=int)
    expected = np.full(shape, 13.0, dtype=np.float64)
    expected[mask] = (a + b)[mask]
    np.testing.assert_allclose(c, expected, rtol=0.0, atol=1e-12)


def test_tileadd_runtime_write_mask_flat_index_expression():
    """Write-mask can reference __i_flat for flattened-index selection."""
    shape = (2, 3, 4)
    sdfg = SDFG("tile_add_runtime_write_mask_flat")
    sdfg.add_array("A", shape=list(shape), dtype=dace.float64)
    sdfg.add_array("B", shape=list(shape), dtype=dace.float64)
    sdfg.add_array("C", shape=list(shape), dtype=dace.float64)

    state = sdfg.add_state("main")
    add_node = TileAddLibraryNode(
        "tile_add_runtime_masked_flat",
        write_mask="(__i_flat % 4) < 2",
    )
    state.add_node(add_node)
    state.add_edge(state.add_read("A"), None, add_node, "_a", Memlet("A[0:2, 0:3, 0:4]"))
    state.add_edge(state.add_read("B"), None, add_node, "_b", Memlet("B[0:2, 0:3, 0:4]"))
    state.add_edge(add_node, "_c", state.add_write("C"), None, Memlet("C[0:2, 0:3, 0:4]"))

    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(9903)
    a = rng.standard_normal(shape, dtype=np.float64)
    b = rng.standard_normal(shape, dtype=np.float64)
    c = np.full(shape, 101.0, dtype=np.float64)

    sdfg(A=a, B=b, C=c)

    flat_mask = (np.arange(np.prod(shape)) % 4) < 2
    mask = flat_mask.reshape(shape)
    expected = np.full(shape, 101.0, dtype=np.float64)
    expected[mask] = (a + b)[mask]
    np.testing.assert_allclose(c, expected, rtol=0.0, atol=1e-12)


def test_scalar_to_tile_sets_symbolic_mask_metadata():
    """Transformation should encode inner-map masking symbols on the produced libnode."""
    sdfg = build_tiled_scalar_add_sdfg()
    count = apply_cutile_pipeline(sdfg, validate=True)
    assert count == 1

    state = sdfg.states()[0]
    lib_nodes = [n for n in state.nodes() if isinstance(n, TileAddLibraryNode)]
    assert len(lib_nodes) == 1

    lib_node = lib_nodes[0]
    assert lib_node.mask_indices == ["ii", "jj"]

    free_symbols = {str(s) for s in dace.symbolic.pystr_to_symbolic(lib_node.write_mask).free_symbols}
    assert "ii" in free_symbols
    assert "jj" in free_symbols


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


# ---------------------------------------------------------------------------
# Helpers for subtraction and N-D dimension tests
# ---------------------------------------------------------------------------

def build_tiled_scalar_subtract_sdfg() -> SDFG:
    """Build an SDFG with tiled scalar subtract (the 'before' state)."""
    sdfg = SDFG("tile_subtract_before")
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
    tasklet = state.add_tasklet("subtract", {"a", "b"}, {"c"}, "c = a - b")

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


def build_runtime_tiled_scalar_subtract_sdfg(
    outer_shape=(2, 3),
    tile_shape=(2, 2),
    dtype=dace.float64,
) -> SDFG:
    """Build a CPU-friendly 'before pipeline' tiled scalar subtract SDFG."""
    mt, nt = outer_shape
    t0, t1 = tile_shape

    sdfg = SDFG("tile_subtract_before_runtime")
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
    tasklet = state.add_tasklet("subtract", {"a", "b"}, {"c"}, "c = a - b")

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


# Tile shapes for N-D dimension tests (0-D through 6-D).
# 0-D is represented via a 1-element 1-D array with tile_shape=[] on the node.
_NDIM_SHAPES = {
    0: (),
    1: (16,),
    2: (4, 8),
    3: (3, 4, 5),
    4: (2, 3, 4, 5),
    5: (2, 2, 3, 4, 5),
    6: (2, 2, 2, 3, 4, 5),
}


def _make_direct_binary_sdfg(shape, node_class, name, tile_shape_override=None):
    """Build a single-state SDFG with one binary library node for a given tile shape.

    For shape=() (0-D), a 1-element 1-D backing array is used and tile_shape=[]
    is forced on the node so the 0-D code path in the expansion is exercised.
    """
    if not shape:
        actual_shape = [1]
        node_tile_shape = [] if tile_shape_override is None else tile_shape_override
    else:
        actual_shape = list(shape)
        node_tile_shape = tile_shape_override

    sdfg = SDFG(name)
    sdfg.add_array("A", shape=actual_shape, dtype=dace.float64)
    sdfg.add_array("B", shape=actual_shape, dtype=dace.float64)
    sdfg.add_array("C", shape=actual_shape, dtype=dace.float64)

    state = sdfg.add_state("main")
    a_read = state.add_read("A")
    b_read = state.add_read("B")
    c_write = state.add_write("C")

    op_node = node_class(name + "_node", tile_shape=node_tile_shape)
    state.add_node(op_node)

    subset = ", ".join(f"0:{s}" for s in actual_shape)
    state.add_edge(a_read, None, op_node, "_a", Memlet(f"A[{subset}]"))
    state.add_edge(b_read, None, op_node, "_b", Memlet(f"B[{subset}]"))
    state.add_edge(op_node, "_c", c_write, None, Memlet(f"C[{subset}]"))

    sdfg.validate()
    return sdfg, actual_shape


def _run_direct_binary_op_test(shape, node_class, np_op, name,
                                tile_shape_override=None, seed=42,
                                dtype=np.float64, atol=1e-12, rtol=1e-12):
    """Expand + execute a direct binary library node and verify against NumPy."""
    sdfg, actual_shape = _make_direct_binary_sdfg(shape, node_class, name, tile_shape_override)
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(seed)
    a = rng.uniform(-10.0, 10.0, size=actual_shape).astype(dtype)
    b = rng.uniform(-10.0, 10.0, size=actual_shape).astype(dtype)
    c = np.zeros(actual_shape, dtype=dtype)

    sdfg(A=a, B=b, C=c)
    np.testing.assert_allclose(c, np_op(a, b), rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Subtraction transformation tests
# ---------------------------------------------------------------------------

def test_scalar_to_tile_subtract():
    """Transform tiled scalar subtract → TileSubtractLibraryNode."""
    sdfg = build_tiled_scalar_subtract_sdfg()

    count = apply_cutile_pipeline(sdfg, validate=True)
    print(f"Applied {count} transformations.")
    assert count == 1, f"Expected 1 transformation, got {count}"

    state = sdfg.states()[0]

    lib_nodes = [n for n in state.nodes() if isinstance(n, nodes.LibraryNode)]
    assert len(lib_nodes) == 1
    assert isinstance(lib_nodes[0], TileSubtractLibraryNode)

    map_entries = [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]
    assert len(map_entries) == 1, f"Expected 1 map entry, got {len(map_entries)}"

    transients = {name for name, desc in sdfg.arrays.items() if desc.transient}
    assert len(transients) == 3, f"Expected 3 transients, got {transients}"

    tasklets = [n for n in state.nodes() if isinstance(n, nodes.Tasklet)]
    assert len(tasklets) == 0

    lib = lib_nodes[0]
    assert set(lib.in_connectors.keys()) == {"_a", "_b"}
    assert set(lib.out_connectors.keys()) == {"_c"}

    sdfg.validate()


def test_subtract_pipeline_idempotent():
    """Running the pipeline twice on a subtract SDFG should be idempotent."""
    sdfg = build_tiled_scalar_subtract_sdfg()

    count1 = apply_cutile_pipeline(sdfg, validate=True)
    assert count1 == 1

    count2 = apply_cutile_pipeline(sdfg, validate=True)
    assert count2 == 0, "Pipeline should be idempotent for subtract"


def test_subtract_structure_matches_expected():
    """Transformed subtract SDFG has the expected node/edge structure."""
    sdfg = build_tiled_scalar_subtract_sdfg()
    apply_cutile_pipeline(sdfg, validate=True)

    state = sdfg.states()[0]
    outer_entry = outer_exit = lib_node = None

    for n in state.nodes():
        if isinstance(n, nodes.MapEntry):
            outer_entry = n
        elif isinstance(n, nodes.MapExit):
            outer_exit = n
        elif isinstance(n, nodes.LibraryNode):
            lib_node = n

    assert outer_entry is not None and outer_exit is not None and lib_node is not None
    assert isinstance(lib_node, TileSubtractLibraryNode)

    scope = state.scope_dict()
    assert scope[lib_node] == outer_entry

    trans_nodes = [n for n in state.nodes()
                   if isinstance(n, nodes.AccessNode) and sdfg.arrays[n.data].transient]
    assert len(trans_nodes) == 3
    for tn in trans_nodes:
        assert scope[tn] == outer_entry

    in_edges = state.in_edges(lib_node)
    assert len(in_edges) == 2
    for e in in_edges:
        assert isinstance(e.src, nodes.AccessNode)
        assert sdfg.arrays[e.src.data].transient

    out_edges = state.out_edges(lib_node)
    assert len(out_edges) == 1
    assert isinstance(out_edges[0].dst, nodes.AccessNode)
    assert sdfg.arrays[out_edges[0].dst.data].transient


# ---------------------------------------------------------------------------
# Subtraction runtime and pipeline tests
# ---------------------------------------------------------------------------

def test_tilesubtract_runtime_numeric_correctness():
    """Execute TileSubtract directly and compare against NumPy subtraction."""
    sdfg = SDFG("tile_subtract_runtime")
    sdfg.add_array("A", shape=[3, 4], dtype=dace.float64)
    sdfg.add_array("B", shape=[3, 4], dtype=dace.float64)
    sdfg.add_array("C", shape=[3, 4], dtype=dace.float64)

    state = sdfg.add_state("main")
    a_read = state.add_read("A")
    b_read = state.add_read("B")
    c_write = state.add_write("C")
    sub_node = TileSubtractLibraryNode("tile_subtract_runtime_node")
    state.add_node(sub_node)

    state.add_edge(a_read, None, sub_node, "_a", Memlet("A[0:3, 0:4]"))
    state.add_edge(b_read, None, sub_node, "_b", Memlet("B[0:3, 0:4]"))
    state.add_edge(sub_node, "_c", c_write, None, Memlet("C[0:3, 0:4]"))

    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(9876)
    a = rng.standard_normal((3, 4))
    b = rng.standard_normal((3, 4))
    c = np.zeros((3, 4), dtype=np.float64)

    sdfg(A=a, B=b, C=c)

    np.testing.assert_allclose(c, a - b, rtol=0.0, atol=1e-12)


def test_pipeline_runtime_subtraction_float64():
    """End-to-end: before-pipeline subtract SDFG -> pipeline -> run -> numeric check."""
    sdfg = build_runtime_tiled_scalar_subtract_sdfg(
        outer_shape=(2, 3),
        tile_shape=(2, 2),
        dtype=dace.float64,
    )

    count = apply_cutile_pipeline(sdfg, validate=True)
    assert count == 1
    sdfg.expand_library_nodes()
    sdfg.validate()

    shape = (2, 3, 2, 2)
    rng = np.random.default_rng(1001)
    a = rng.uniform(-50.0, 50.0, size=shape).astype(np.float64)
    b = rng.uniform(-50.0, 50.0, size=shape).astype(np.float64)
    c = np.zeros(shape, dtype=np.float64)

    sdfg(A=a, B=b, C=c)

    np.testing.assert_allclose(c, a - b, rtol=0.0, atol=1e-12)


def test_pipeline_runtime_subtraction_float32():
    """End-to-end subtract pipeline test with float32."""
    sdfg = build_runtime_tiled_scalar_subtract_sdfg(
        outer_shape=(3, 2),
        tile_shape=(1, 4),
        dtype=dace.float32,
    )

    count = apply_cutile_pipeline(sdfg, validate=True)
    assert count == 1
    sdfg.expand_library_nodes()
    sdfg.validate()

    shape = (3, 2, 1, 4)
    rng = np.random.default_rng(2002)
    a = rng.uniform(-10.0, 10.0, size=shape).astype(np.float32)
    b = rng.uniform(-10.0, 10.0, size=shape).astype(np.float32)
    c = np.zeros(shape, dtype=np.float32)

    sdfg(A=a, B=b, C=c)

    np.testing.assert_allclose(c, a - b, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# N-D dimension tests: TileAdd (0-D through 6-D)
# ---------------------------------------------------------------------------

def test_tileadd_ndim_0():
    """TileAdd: 0-D expansion path (tile_shape=[]) on a 1-element array."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[0], node_class=TileAddLibraryNode, np_op=np.add,
        name="tileadd_ndim0",
    )


def test_tileadd_ndim_1():
    """TileAdd with a 1-D tile of shape (16,)."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[1], node_class=TileAddLibraryNode, np_op=np.add,
        name="tileadd_ndim1",
    )


def test_tileadd_ndim_2():
    """TileAdd with a 2-D tile of shape (4, 8)."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[2], node_class=TileAddLibraryNode, np_op=np.add,
        name="tileadd_ndim2",
    )


def test_tileadd_ndim_3():
    """TileAdd with a 3-D tile of shape (3, 4, 5)."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[3], node_class=TileAddLibraryNode, np_op=np.add,
        name="tileadd_ndim3",
    )


def test_tileadd_ndim_4():
    """TileAdd with a 4-D tile of shape (2, 3, 4, 5)."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[4], node_class=TileAddLibraryNode, np_op=np.add,
        name="tileadd_ndim4",
    )


def test_tileadd_ndim_5():
    """TileAdd with a 5-D tile of shape (2, 2, 3, 4, 5)."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[5], node_class=TileAddLibraryNode, np_op=np.add,
        name="tileadd_ndim5",
    )


def test_tileadd_ndim_6():
    """TileAdd with a 6-D tile of shape (2, 2, 2, 3, 4, 5)."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[6], node_class=TileAddLibraryNode, np_op=np.add,
        name="tileadd_ndim6",
    )


# ---------------------------------------------------------------------------
# N-D dimension tests: TileSubtract (0-D through 6-D)
# ---------------------------------------------------------------------------

def test_tilesubtract_ndim_0():
    """TileSubtract: 0-D expansion path (tile_shape=[]) on a 1-element array."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[0], node_class=TileSubtractLibraryNode, np_op=np.subtract,
        name="tilesubtract_ndim0",
    )


def test_tilesubtract_ndim_1():
    """TileSubtract with a 1-D tile of shape (16,)."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[1], node_class=TileSubtractLibraryNode, np_op=np.subtract,
        name="tilesubtract_ndim1",
    )


def test_tilesubtract_ndim_2():
    """TileSubtract with a 2-D tile of shape (4, 8)."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[2], node_class=TileSubtractLibraryNode, np_op=np.subtract,
        name="tilesubtract_ndim2",
    )


def test_tilesubtract_ndim_3():
    """TileSubtract with a 3-D tile of shape (3, 4, 5)."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[3], node_class=TileSubtractLibraryNode, np_op=np.subtract,
        name="tilesubtract_ndim3",
    )


def test_tilesubtract_ndim_4():
    """TileSubtract with a 4-D tile of shape (2, 3, 4, 5)."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[4], node_class=TileSubtractLibraryNode, np_op=np.subtract,
        name="tilesubtract_ndim4",
    )


def test_tilesubtract_ndim_5():
    """TileSubtract with a 5-D tile of shape (2, 2, 3, 4, 5)."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[5], node_class=TileSubtractLibraryNode, np_op=np.subtract,
        name="tilesubtract_ndim5",
    )


def test_tilesubtract_ndim_6():
    """TileSubtract with a 6-D tile of shape (2, 2, 2, 3, 4, 5)."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[6], node_class=TileSubtractLibraryNode, np_op=np.subtract,
        name="tilesubtract_ndim6",
    )


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

def test_subtract_self_gives_zero():
    """A - A must be exactly zero for all elements."""
    shape = [5, 6]
    sdfg = SDFG("subtract_self")
    sdfg.add_array("A", shape=shape, dtype=dace.float64)
    sdfg.add_array("C", shape=shape, dtype=dace.float64)

    state = sdfg.add_state("main")
    a_read_1 = state.add_read("A")
    a_read_2 = state.add_read("A")
    c_write = state.add_write("C")

    sub_node = TileSubtractLibraryNode("sub_self")
    state.add_node(sub_node)
    state.add_edge(a_read_1, None, sub_node, "_a", Memlet("A[0:5, 0:6]"))
    state.add_edge(a_read_2, None, sub_node, "_b", Memlet("A[0:5, 0:6]"))
    state.add_edge(sub_node, "_c", c_write, None, Memlet("C[0:5, 0:6]"))

    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(11)
    a = rng.uniform(-100.0, 100.0, size=shape).astype(np.float64)
    c = np.ones(shape, dtype=np.float64)  # non-zero init to catch unwritten elements

    sdfg(A=a, C=c)
    np.testing.assert_array_equal(c, np.zeros(shape))


def test_add_subtract_roundtrip():
    """(A + B) - B must recover A to floating-point precision."""
    shape = [6, 7]
    rng = np.random.default_rng(22)
    a = rng.uniform(-50.0, 50.0, size=shape).astype(np.float64)
    b = rng.uniform(-50.0, 50.0, size=shape).astype(np.float64)

    # Step 1: T = A + B
    sdfg_add = SDFG("add_roundtrip")
    sdfg_add.add_array("A", shape=shape, dtype=dace.float64)
    sdfg_add.add_array("B", shape=shape, dtype=dace.float64)
    sdfg_add.add_array("T", shape=shape, dtype=dace.float64)
    st = sdfg_add.add_state()
    add_node = TileAddLibraryNode("add_rt_node")
    st.add_node(add_node)
    st.add_edge(st.add_read("A"), None, add_node, "_a", Memlet("A[0:6, 0:7]"))
    st.add_edge(st.add_read("B"), None, add_node, "_b", Memlet("B[0:6, 0:7]"))
    st.add_edge(add_node, "_c", st.add_write("T"), None, Memlet("T[0:6, 0:7]"))
    sdfg_add.expand_library_nodes()
    t = np.zeros(shape, dtype=np.float64)
    sdfg_add(A=a, B=b, T=t)

    # Step 2: O = T - B
    sdfg_sub = SDFG("sub_roundtrip")
    sdfg_sub.add_array("T", shape=shape, dtype=dace.float64)
    sdfg_sub.add_array("B", shape=shape, dtype=dace.float64)
    sdfg_sub.add_array("O", shape=shape, dtype=dace.float64)
    st2 = sdfg_sub.add_state()
    sub_node = TileSubtractLibraryNode("sub_rt_node")
    st2.add_node(sub_node)
    st2.add_edge(st2.add_read("T"), None, sub_node, "_a", Memlet("T[0:6, 0:7]"))
    st2.add_edge(st2.add_read("B"), None, sub_node, "_b", Memlet("B[0:6, 0:7]"))
    st2.add_edge(sub_node, "_c", st2.add_write("O"), None, Memlet("O[0:6, 0:7]"))
    sdfg_sub.expand_library_nodes()
    out = np.zeros(shape, dtype=np.float64)
    sdfg_sub(T=t, B=b, O=out)

    np.testing.assert_allclose(out, a, rtol=1e-12, atol=1e-12)


def test_add_zero_identity():
    """A + zeros must equal A exactly."""
    shape = [8, 8]
    sdfg, actual_shape = _make_direct_binary_sdfg(shape, TileAddLibraryNode, "add_zero_identity")
    sdfg.expand_library_nodes()

    rng = np.random.default_rng(33)
    a = rng.uniform(-1.0, 1.0, size=actual_shape).astype(np.float64)
    b = np.zeros(actual_shape, dtype=np.float64)
    c = np.empty(actual_shape, dtype=np.float64)

    sdfg(A=a, B=b, C=c)
    np.testing.assert_array_equal(c, a)


def test_subtract_zero_identity():
    """A - zeros must equal A exactly."""
    shape = [8, 8]
    sdfg, actual_shape = _make_direct_binary_sdfg(shape, TileSubtractLibraryNode, "sub_zero_identity")
    sdfg.expand_library_nodes()

    rng = np.random.default_rng(44)
    a = rng.uniform(-1.0, 1.0, size=actual_shape).astype(np.float64)
    b = np.zeros(actual_shape, dtype=np.float64)
    c = np.empty(actual_shape, dtype=np.float64)

    sdfg(A=a, B=b, C=c)
    np.testing.assert_array_equal(c, a)


def test_tilesubtract_negative_result():
    """TileSubtract produces correct negative values when a < 0 and b > 0."""
    shape = [4, 5]
    sdfg, actual_shape = _make_direct_binary_sdfg(shape, TileSubtractLibraryNode, "sub_negative")
    sdfg.expand_library_nodes()

    rng = np.random.default_rng(55)
    a = rng.uniform(-10.0, -1.0, size=actual_shape).astype(np.float64)
    b = rng.uniform(1.0, 10.0, size=actual_shape).astype(np.float64)
    c = np.zeros(actual_shape, dtype=np.float64)

    sdfg(A=a, B=b, C=c)
    assert np.all(c < 0), "All results should be negative when a < 0 and b > 0"
    np.testing.assert_allclose(c, a - b, rtol=0.0, atol=1e-12)


def test_large_tile_subtract():
    """TileSubtract on a large 2-D tile (64 × 128)."""
    _run_direct_binary_op_test(
        shape=[64, 128], node_class=TileSubtractLibraryNode, np_op=np.subtract,
        name="sub_large_tile", seed=66,
    )


def test_large_tile_add():
    """TileAdd on a large 2-D tile (64 × 128)."""
    _run_direct_binary_op_test(
        shape=[64, 128], node_class=TileAddLibraryNode, np_op=np.add,
        name="add_large_tile", seed=77,
    )


def test_tileadd_int32_dtype():
    """TileAdd with int32 tiles — exact integer result expected."""
    shape = [5, 6]
    sdfg = SDFG("tileadd_int32")
    sdfg.add_array("A", shape=shape, dtype=dace.int32)
    sdfg.add_array("B", shape=shape, dtype=dace.int32)
    sdfg.add_array("C", shape=shape, dtype=dace.int32)

    state = sdfg.add_state("main")
    add_node = TileAddLibraryNode("add_int32")
    state.add_node(add_node)
    state.add_edge(state.add_read("A"), None, add_node, "_a", Memlet("A[0:5, 0:6]"))
    state.add_edge(state.add_read("B"), None, add_node, "_b", Memlet("B[0:5, 0:6]"))
    state.add_edge(add_node, "_c", state.add_write("C"), None, Memlet("C[0:5, 0:6]"))

    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(88)
    a = rng.integers(-100, 100, size=shape, dtype=np.int32)
    b = rng.integers(-100, 100, size=shape, dtype=np.int32)
    c = np.zeros(shape, dtype=np.int32)

    sdfg(A=a, B=b, C=c)
    np.testing.assert_array_equal(c, a + b)


def test_tilesubtract_int32_dtype():
    """TileSubtract with int32 tiles — exact integer result expected."""
    shape = [5, 6]
    sdfg = SDFG("tilesubtract_int32")
    sdfg.add_array("A", shape=shape, dtype=dace.int32)
    sdfg.add_array("B", shape=shape, dtype=dace.int32)
    sdfg.add_array("C", shape=shape, dtype=dace.int32)

    state = sdfg.add_state("main")
    sub_node = TileSubtractLibraryNode("sub_int32")
    state.add_node(sub_node)
    state.add_edge(state.add_read("A"), None, sub_node, "_a", Memlet("A[0:5, 0:6]"))
    state.add_edge(state.add_read("B"), None, sub_node, "_b", Memlet("B[0:5, 0:6]"))
    state.add_edge(sub_node, "_c", state.add_write("C"), None, Memlet("C[0:5, 0:6]"))

    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(99)
    a = rng.integers(-100, 100, size=shape, dtype=np.int32)
    b = rng.integers(-100, 100, size=shape, dtype=np.int32)
    c = np.zeros(shape, dtype=np.int32)

    sdfg(A=a, B=b, C=c)
    np.testing.assert_array_equal(c, a - b)


def test_subtract_pipeline_various_tile_sizes():
    """Pipeline subtract test across multiple outer/tile size configurations."""
    configs = [
        ((1, 1), (4, 4)),
        ((4, 4), (1, 1)),
        ((2, 5), (3, 3)),
    ]
    for outer_shape, tile_shape in configs:
        sdfg = build_runtime_tiled_scalar_subtract_sdfg(
            outer_shape=outer_shape,
            tile_shape=tile_shape,
            dtype=dace.float64,
        )
        count = apply_cutile_pipeline(sdfg, validate=True)
        assert count == 1, f"Expected 1 for {outer_shape=} {tile_shape=}, got {count}"
        sdfg.expand_library_nodes()
        sdfg.validate()

        full_shape = outer_shape + tile_shape
        rng = np.random.default_rng(100)
        a = rng.uniform(-5.0, 5.0, size=full_shape).astype(np.float64)
        b = rng.uniform(-5.0, 5.0, size=full_shape).astype(np.float64)
        c = np.zeros(full_shape, dtype=np.float64)

        sdfg(A=a, B=b, C=c)
        np.testing.assert_allclose(c, a - b, rtol=0.0, atol=1e-12,
                                   err_msg=f"Failed for {outer_shape=} {tile_shape=}")


def test_add_pipeline_various_tile_sizes():
    """Pipeline add test across multiple outer/tile size configurations."""
    configs = [
        ((1, 1), (4, 4)),
        ((4, 4), (1, 1)),
        ((2, 5), (3, 3)),
    ]
    for outer_shape, tile_shape in configs:
        sdfg = build_runtime_tiled_scalar_add_sdfg(
            outer_shape=outer_shape,
            tile_shape=tile_shape,
            dtype=dace.float64,
        )
        count = apply_cutile_pipeline(sdfg, validate=True)
        assert count == 1
        sdfg.expand_library_nodes()
        sdfg.validate()

        full_shape = outer_shape + tile_shape
        rng = np.random.default_rng(101)
        a = rng.uniform(-5.0, 5.0, size=full_shape).astype(np.float64)
        b = rng.uniform(-5.0, 5.0, size=full_shape).astype(np.float64)
        c = np.zeros(full_shape, dtype=np.float64)

        sdfg(A=a, B=b, C=c)
        np.testing.assert_allclose(c, a + b, rtol=0.0, atol=1e-12,
                                   err_msg=f"Failed for {outer_shape=} {tile_shape=}")


if __name__ == "__main__":
    # --- original add tests ---
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

    # --- subtraction transformation tests ---
    test_scalar_to_tile_subtract()
    print("[PASS] test_scalar_to_tile_subtract")

    test_subtract_pipeline_idempotent()
    print("[PASS] test_subtract_pipeline_idempotent")

    test_subtract_structure_matches_expected()
    print("[PASS] test_subtract_structure_matches_expected")

    # --- subtraction runtime/pipeline tests ---
    test_tilesubtract_runtime_numeric_correctness()
    print("[PASS] test_tilesubtract_runtime_numeric_correctness")

    test_pipeline_runtime_subtraction_float64()
    print("[PASS] test_pipeline_runtime_subtraction_float64")

    test_pipeline_runtime_subtraction_float32()
    print("[PASS] test_pipeline_runtime_subtraction_float32")

    # --- N-D dimension tests: TileAdd ---
    for _d in range(7):
        _fn = globals()[f"test_tileadd_ndim_{_d}"]
        _fn()
        print(f"[PASS] test_tileadd_ndim_{_d}")

    # --- N-D dimension tests: TileSubtract ---
    for _d in range(7):
        _fn = globals()[f"test_tilesubtract_ndim_{_d}"]
        _fn()
        print(f"[PASS] test_tilesubtract_ndim_{_d}")

    # --- Edge case tests ---
    test_subtract_self_gives_zero()
    print("[PASS] test_subtract_self_gives_zero")

    test_add_subtract_roundtrip()
    print("[PASS] test_add_subtract_roundtrip")

    test_add_zero_identity()
    print("[PASS] test_add_zero_identity")

    test_subtract_zero_identity()
    print("[PASS] test_subtract_zero_identity")

    test_tilesubtract_negative_result()
    print("[PASS] test_tilesubtract_negative_result")

    test_large_tile_subtract()
    print("[PASS] test_large_tile_subtract")

    test_large_tile_add()
    print("[PASS] test_large_tile_add")

    test_tileadd_int32_dtype()
    print("[PASS] test_tileadd_int32_dtype")

    test_tilesubtract_int32_dtype()
    print("[PASS] test_tilesubtract_int32_dtype")

    test_subtract_pipeline_various_tile_sizes()
    print("[PASS] test_subtract_pipeline_various_tile_sizes")

    test_add_pipeline_various_tile_sizes()
    print("[PASS] test_add_pipeline_various_tile_sizes")

    print("\nAll tests passed!")
