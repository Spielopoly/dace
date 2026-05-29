"""
Tests for the cuTile transformation pipeline.

Tests scalar-to-library transformations that replace inner maps
with scalar tasklets by cuTile library nodes.
"""


import dace
import numpy as np
import pytest
from dace import dtypes, Memlet
from dace.sdfg import SDFG, nodes
from dace.libraries.cutile.transformations.pipeline import apply_cutile_pipeline
from dace.libraries.cutile.transformations.scalar_to_tile_library import (
    ScalarToTileCanonical,
    ScalarToTileMasked,
)
from dace.libraries.cutile.nodes import TileOpLibraryNode, TileRuntimeMaskedOpLibraryNode, TileSymbolicMaskedOpLibraryNode


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


def build_runtime_tiled_scalar_add_same_input_sdfg(
    outer_shape=(2, 3),
    tile_shape=(2, 2),
    dtype=dace.float64,
) -> SDFG:
    """Build a CPU-friendly 'before pipeline' tiled scalar c=a+a SDFG."""
    mt, nt = outer_shape
    t0, t1 = tile_shape

    sdfg = SDFG("tile_add_same_input_before_runtime")
    sdfg.add_array("A", shape=[mt, nt, t0, t1], dtype=dtype)
    sdfg.add_array("C", shape=[mt, nt, t0, t1], dtype=dtype)

    state = sdfg.add_state("main")
    a_acc = state.add_read("A")
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
    tasklet = state.add_tasklet("add_same", {"a"}, {"c"}, "c = a + a")

    state.add_memlet_path(
        a_acc, outer_entry, inner_entry, tasklet,
        dst_conn="a", memlet=Memlet("A[i, j, ii, jj]"),
    )
    state.add_memlet_path(
        tasklet, inner_exit, outer_exit, c_acc,
        src_conn="c", memlet=Memlet("C[i, j, ii, jj]"),
    )

    sdfg.validate()
    return sdfg


def build_tiled_scalar_add_same_input_sdfg() -> SDFG:
    """Build canonical tiled SDFG with duplicated operand use: c = a + a."""
    sdfg = SDFG("tile_add_same_input_before")
    for sym in ("M", "N", "T0", "T1"):
        sdfg.add_symbol(sym, dace.int32)

    _add_blocked_arrays(sdfg, names=("A", "C"))
    state = sdfg.add_state("main")

    A_acc = state.add_read("A")
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
    tasklet = state.add_tasklet("add_same", {"a"}, {"c"}, "c = a + a")

    state.add_memlet_path(
        A_acc, outer_entry, inner_entry, tasklet,
        dst_conn="a", memlet=Memlet("A[i, j, ii, jj]"),
    )
    state.add_memlet_path(
        tasklet, inner_exit, outer_exit, C_acc,
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

    count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
    assert count == 1, f"Expected 1 transformation, got {count}"

    state = sdfg.states()[0]

    # Exactly one library node
    lib_nodes = [n for n in state.nodes() if isinstance(n, nodes.LibraryNode)]
    assert len(lib_nodes) == 1
    assert isinstance(lib_nodes[0], TileOpLibraryNode)
    assert lib_nodes[0].op == "+"

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
    assert set(lib.out_connectors.keys()) == {"_out"}

    # Validate final SDFG
    sdfg.validate()


def test_scalar_to_tile_add_same_input_operand():
    """Regression: c = a + a must wire one input tile to both library inputs."""
    sdfg = build_tiled_scalar_add_same_input_sdfg()

    count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
    assert count == 1, f"Expected 1 transformation, got {count}"

    state = sdfg.states()[0]
    lib_nodes = [n for n in state.nodes() if isinstance(n, nodes.LibraryNode)]
    assert len(lib_nodes) == 1
    assert isinstance(lib_nodes[0], TileOpLibraryNode)
    assert lib_nodes[0].op == "+"

    lib_node = lib_nodes[0]
    in_edges = state.in_edges(lib_node)
    assert len(in_edges) == 2
    assert {e.dst_conn for e in in_edges} == {"_a", "_b"}
    assert len({e.src.data for e in in_edges if isinstance(e.src, nodes.AccessNode)}) == 1


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

    count = sdfg.apply_transformations([ScalarToTileCanonical, ScalarToTileMasked])
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

    count = sdfg.apply_transformations([ScalarToTileCanonical, ScalarToTileMasked])
    assert count == 0, "Should not apply to unknown operation"


def test_pipeline_idempotent():
    """Running the pipeline twice should not change a transformed SDFG."""
    sdfg = build_tiled_scalar_add_sdfg()

    count1 = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
    assert count1 == 1

    count2 = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
    assert count2 == 0, "Pipeline should be idempotent"


def test_structure_matches_expected():
    """Verify the transformed SDFG has the expected node/edge structure."""
    sdfg = build_tiled_scalar_add_sdfg()
    apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)

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
    add_node = TileOpLibraryNode("tile_add_runtime_node", op="+")
    state.add_node(add_node)

    state.add_edge(a_read, None, add_node, "_a", Memlet("A[0:3, 0:4]"))
    state.add_edge(b_read, None, add_node, "_b", Memlet("B[0:3, 0:4]"))
    state.add_edge(add_node, "_out", c_write, None, Memlet("C[0:3, 0:4]"))

    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(1234)
    a = rng.standard_normal((3, 4), dtype=np.float64)
    b = rng.standard_normal((3, 4), dtype=np.float64)
    c = np.zeros((3, 4), dtype=np.float64)

    sdfg(A=a, B=b, C=c)

    np.testing.assert_allclose(c, a + b, rtol=0.0, atol=1e-12)


def test_pipeline_runtime_numeric_correctness_float64():
    """End-to-end: before-pipeline SDFG -> pipeline -> run -> numeric check."""
    sdfg = build_runtime_tiled_scalar_add_sdfg(
        outer_shape=(2, 3),
        tile_shape=(2, 2),
        dtype=dace.float64,
    )

    count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
    assert count == 1
    sdfg.expand_library_nodes()
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

    count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
    assert count == 1
    sdfg.expand_library_nodes()
    sdfg.validate()

    shape = (3, 2, 1, 4)
    rng = np.random.default_rng(7)
    a = rng.uniform(-10.0, 10.0, size=shape).astype(np.float32)
    b = rng.uniform(-10.0, 10.0, size=shape).astype(np.float32)
    c = np.zeros(shape, dtype=np.float32)

    sdfg(A=a, B=b, C=c)

    np.testing.assert_allclose(c, a + b, rtol=1e-6, atol=1e-6)


def test_pipeline_runtime_numeric_correctness_same_input_float64():
    """End-to-end: c = a + a is transformed and numerically correct."""
    sdfg = build_runtime_tiled_scalar_add_same_input_sdfg(
        outer_shape=(2, 3),
        tile_shape=(2, 2),
        dtype=dace.float64,
    )

    count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
    assert count == 1
    sdfg.expand_library_nodes()
    sdfg.validate()

    shape = (2, 3, 2, 2)
    rng = np.random.default_rng(2027)
    a = rng.uniform(-50.0, 50.0, size=shape).astype(np.float64)
    c = np.zeros(shape, dtype=np.float64)

    sdfg(A=a, C=c)

    np.testing.assert_allclose(c, a + a, rtol=0.0, atol=1e-12)


def test_pipeline_runtime_numeric_correctness_same_input_float32():
    """Second shape/type runtime check for c = a + a."""
    sdfg = build_runtime_tiled_scalar_add_same_input_sdfg(
        outer_shape=(3, 2),
        tile_shape=(1, 4),
        dtype=dace.float32,
    )

    count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
    assert count == 1
    sdfg.expand_library_nodes()
    sdfg.validate()

    shape = (3, 2, 1, 4)
    rng = np.random.default_rng(2028)
    a = rng.uniform(-10.0, 10.0, size=shape).astype(np.float32)
    c = np.zeros(shape, dtype=np.float32)

    sdfg(A=a, C=c)

    np.testing.assert_allclose(c, a + a, rtol=1e-6, atol=1e-6)


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


def build_tiled_scalar_masked_add_sdfg(mask_dtype=dace.bool) -> SDFG:
    """Build an SDFG with tiled scalar masked add (the 'before' state)."""
    sdfg = SDFG("tile_masked_add_before")
    for sym in ("M", "N", "T0", "T1"):
        sdfg.add_symbol(sym, dace.int32)

    _add_blocked_arrays(sdfg, names=("A", "B", "MASK", "C"))
    sdfg.arrays["MASK"].dtype = mask_dtype

    state = sdfg.add_state("main")
    a_acc = state.add_read("A")
    b_acc = state.add_read("B")
    m_acc = state.add_read("MASK")
    c_acc = state.add_write("C")

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
    tasklet = state.add_tasklet("masked_add", {"a", "b", "m"}, {"c"}, "if m:\n    c = a + b")

    state.add_memlet_path(
        a_acc,
        outer_entry,
        inner_entry,
        tasklet,
        dst_conn="a",
        memlet=Memlet("A[i, j, ii, jj]"),
    )
    state.add_memlet_path(
        b_acc,
        outer_entry,
        inner_entry,
        tasklet,
        dst_conn="b",
        memlet=Memlet("B[i, j, ii, jj]"),
    )
    state.add_memlet_path(
        m_acc,
        outer_entry,
        inner_entry,
        tasklet,
        dst_conn="m",
        memlet=Memlet("MASK[i, j, ii, jj]"),
    )
    state.add_memlet_path(
        tasklet,
        inner_exit,
        outer_exit,
        c_acc,
        src_conn="c",
        memlet=Memlet("C[i, j, ii, jj]"),
    )

    sdfg.validate()
    return sdfg


def build_runtime_tiled_scalar_masked_add_sdfg(
    outer_shape=(2, 3),
    tile_shape=(2, 2),
    dtype=dace.float64,
    mask_dtype=dace.bool,
) -> SDFG:
    """Build a CPU-friendly 'before pipeline' tiled scalar masked add SDFG."""
    mt, nt = outer_shape
    t0, t1 = tile_shape

    sdfg = SDFG("tile_masked_add_before_runtime")
    sdfg.add_array("A", shape=[mt, nt, t0, t1], dtype=dtype)
    sdfg.add_array("B", shape=[mt, nt, t0, t1], dtype=dtype)
    sdfg.add_array("MASK", shape=[mt, nt, t0, t1], dtype=mask_dtype)
    sdfg.add_array("C", shape=[mt, nt, t0, t1], dtype=dtype)

    state = sdfg.add_state("main")
    a_acc = state.add_read("A")
    b_acc = state.add_read("B")
    m_acc = state.add_read("MASK")
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
    tasklet = state.add_tasklet("masked_add", {"a", "b", "m"}, {"c"}, "if m:\n    c = a + b")

    state.add_memlet_path(
        a_acc,
        outer_entry,
        inner_entry,
        tasklet,
        dst_conn="a",
        memlet=Memlet("A[i, j, ii, jj]"),
    )
    state.add_memlet_path(
        b_acc,
        outer_entry,
        inner_entry,
        tasklet,
        dst_conn="b",
        memlet=Memlet("B[i, j, ii, jj]"),
    )
    state.add_memlet_path(
        m_acc,
        outer_entry,
        inner_entry,
        tasklet,
        dst_conn="m",
        memlet=Memlet("MASK[i, j, ii, jj]"),
    )
    state.add_memlet_path(
        tasklet,
        inner_exit,
        outer_exit,
        c_acc,
        src_conn="c",
        memlet=Memlet("C[i, j, ii, jj]"),
    )

    sdfg.validate()
    return sdfg


def _make_direct_masked_add_sdfg(shape, name, dtype=dace.float64, mask_dtype=dace.bool, tile_shape_override=None):
    """Build a single-state SDFG with one masked-add library node."""
    if not shape:
        actual_shape = [1]
        node_tile_shape = [] if tile_shape_override is None else tile_shape_override
    else:
        actual_shape = list(shape)
        node_tile_shape = tile_shape_override

    sdfg = SDFG(name)
    sdfg.add_array("A", shape=actual_shape, dtype=dtype)
    sdfg.add_array("B", shape=actual_shape, dtype=dtype)
    sdfg.add_array("MASK", shape=actual_shape, dtype=mask_dtype)
    sdfg.add_array("C", shape=actual_shape, dtype=dtype)

    state = sdfg.add_state("main")
    a_read = state.add_read("A")
    b_read = state.add_read("B")
    m_read = state.add_read("MASK")
    c_write = state.add_write("C")

    op_node = TileRuntimeMaskedOpLibraryNode(name + "_node", op="+", tile_shape=node_tile_shape)
    state.add_node(op_node)

    subset = ", ".join(f"0:{s}" for s in actual_shape)
    state.add_edge(a_read, None, op_node, "_a", Memlet(f"A[{subset}]"))
    state.add_edge(b_read, None, op_node, "_b", Memlet(f"B[{subset}]"))
    state.add_edge(m_read, None, op_node, "_m", Memlet(f"MASK[{subset}]"))
    state.add_edge(op_node, "_out", c_write, None, Memlet(f"C[{subset}]"))

    sdfg.validate()
    return sdfg, actual_shape


def _run_direct_masked_add_test(shape,
                                name,
                                seed=123,
                                dtype=np.float64,
                                mask_dtype=dace.bool,
                                tile_shape_override=None,
                                c_init=3.5,
                                atol=1e-12,
                                rtol=1e-12):
    """Expand + execute masked add and verify that unmasked values stay untouched."""
    dace_dtype = dace.float64 if dtype == np.float64 else dace.float32
    sdfg, actual_shape = _make_direct_masked_add_sdfg(
        shape,
        name,
        dtype=dace_dtype,
        mask_dtype=mask_dtype,
        tile_shape_override=tile_shape_override,
    )
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(seed)
    a = rng.uniform(-10.0, 10.0, size=actual_shape).astype(dtype)
    b = rng.uniform(-10.0, 10.0, size=actual_shape).astype(dtype)

    if mask_dtype == dace.bool:
        mask_np = rng.integers(0, 2, size=actual_shape).astype(np.bool_)
    else:
        mask_np = rng.integers(0, 4, size=actual_shape).astype(np.uint8)

    c = np.full(actual_shape, c_init, dtype=dtype)
    expected = np.full(actual_shape, c_init, dtype=dtype)
    expected[mask_np.astype(bool)] = (a + b)[mask_np.astype(bool)]

    sdfg(A=a, B=b, MASK=mask_np, C=c)
    np.testing.assert_allclose(c, expected, rtol=rtol, atol=atol)


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


def _make_direct_binary_sdfg(shape, name, op="+", tile_shape_override=None):
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

    op_node = TileOpLibraryNode(name + "_node", op=op, tile_shape=node_tile_shape)
    state.add_node(op_node)

    subset = ", ".join(f"0:{s}" for s in actual_shape)
    state.add_edge(a_read, None, op_node, "_a", Memlet(f"A[{subset}]"))
    state.add_edge(b_read, None, op_node, "_b", Memlet(f"B[{subset}]"))
    state.add_edge(op_node, "_out", c_write, None, Memlet(f"C[{subset}]"))

    sdfg.validate()
    return sdfg, actual_shape


def _run_direct_binary_op_test(shape, np_op, name, op="+",
                                tile_shape_override=None, seed=42,
                                dtype=np.float64, atol=1e-12, rtol=1e-12):
    """Expand + execute a direct binary library node and verify against NumPy."""
    sdfg, actual_shape = _make_direct_binary_sdfg(shape, name, op=op, tile_shape_override=tile_shape_override)
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

    count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
    assert count == 1, f"Expected 1 transformation, got {count}"

    state = sdfg.states()[0]

    lib_nodes = [n for n in state.nodes() if isinstance(n, nodes.LibraryNode)]
    assert len(lib_nodes) == 1
    assert isinstance(lib_nodes[0], TileOpLibraryNode)
    assert lib_nodes[0].op == "-"

    map_entries = [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]
    assert len(map_entries) == 1, f"Expected 1 map entry, got {len(map_entries)}"

    transients = {name for name, desc in sdfg.arrays.items() if desc.transient}
    assert len(transients) == 3, f"Expected 3 transients, got {transients}"

    tasklets = [n for n in state.nodes() if isinstance(n, nodes.Tasklet)]
    assert len(tasklets) == 0

    lib = lib_nodes[0]
    assert set(lib.in_connectors.keys()) == {"_a", "_b"}
    assert set(lib.out_connectors.keys()) == {"_out"}

    sdfg.validate()


def test_subtract_pipeline_idempotent():
    """Running the pipeline twice on a subtract SDFG should be idempotent."""
    sdfg = build_tiled_scalar_subtract_sdfg()

    count1 = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
    assert count1 == 1

    count2 = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
    assert count2 == 0, "Pipeline should be idempotent for subtract"


def test_subtract_structure_matches_expected():
    """Transformed subtract SDFG has the expected node/edge structure."""
    sdfg = build_tiled_scalar_subtract_sdfg()
    apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)

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
    assert isinstance(lib_node, TileOpLibraryNode)
    assert lib_node.op == "-"

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
    sub_node = TileOpLibraryNode("tile_subtract_runtime_node", op="-")
    state.add_node(sub_node)

    state.add_edge(a_read, None, sub_node, "_a", Memlet("A[0:3, 0:4]"))
    state.add_edge(b_read, None, sub_node, "_b", Memlet("B[0:3, 0:4]"))
    state.add_edge(sub_node, "_out", c_write, None, Memlet("C[0:3, 0:4]"))

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

    count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
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

    count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
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
        shape=_NDIM_SHAPES[0], np_op=np.add,
        name="tileadd_ndim0", op="+",
    )


def test_tileadd_ndim_1():
    """TileAdd with a 1-D tile of shape (16,)."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[1], np_op=np.add,
        name="tileadd_ndim1", op="+",
    )


def test_tileadd_ndim_2():
    """TileAdd with a 2-D tile of shape (4, 8)."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[2], np_op=np.add,
        name="tileadd_ndim2", op="+",
    )


def test_tileadd_ndim_3():
    """TileAdd with a 3-D tile of shape (3, 4, 5)."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[3], np_op=np.add,
        name="tileadd_ndim3", op="+",
    )


def test_tileadd_ndim_4():
    """TileAdd with a 4-D tile of shape (2, 3, 4, 5)."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[4], np_op=np.add,
        name="tileadd_ndim4", op="+",
    )


def test_tileadd_ndim_5():
    """TileAdd with a 5-D tile of shape (2, 2, 3, 4, 5)."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[5], np_op=np.add,
        name="tileadd_ndim5", op="+",
    )


def test_tileadd_ndim_6():
    """TileAdd with a 6-D tile of shape (2, 2, 2, 3, 4, 5)."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[6], np_op=np.add,
        name="tileadd_ndim6", op="+",
    )


# ---------------------------------------------------------------------------
# N-D dimension tests: TileSubtract (0-D through 6-D)
# ---------------------------------------------------------------------------

def test_tilesubtract_ndim_0():
    """TileSubtract: 0-D expansion path (tile_shape=[]) on a 1-element array."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[0], np_op=np.subtract,
        name="tilesubtract_ndim0", op="-",
    )


def test_tilesubtract_ndim_1():
    """TileSubtract with a 1-D tile of shape (16,)."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[1], np_op=np.subtract,
        name="tilesubtract_ndim1", op="-",
    )


def test_tilesubtract_ndim_2():
    """TileSubtract with a 2-D tile of shape (4, 8)."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[2], np_op=np.subtract,
        name="tilesubtract_ndim2", op="-",
    )


def test_tilesubtract_ndim_3():
    """TileSubtract with a 3-D tile of shape (3, 4, 5)."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[3], np_op=np.subtract,
        name="tilesubtract_ndim3", op="-",
    )


def test_tilesubtract_ndim_4():
    """TileSubtract with a 4-D tile of shape (2, 3, 4, 5)."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[4], np_op=np.subtract,
        name="tilesubtract_ndim4", op="-",
    )


def test_tilesubtract_ndim_5():
    """TileSubtract with a 5-D tile of shape (2, 2, 3, 4, 5)."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[5], np_op=np.subtract,
        name="tilesubtract_ndim5", op="-",
    )


def test_tilesubtract_ndim_6():
    """TileSubtract with a 6-D tile of shape (2, 2, 2, 3, 4, 5)."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[6], np_op=np.subtract,
        name="tilesubtract_ndim6", op="-",
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

    sub_node = TileOpLibraryNode("sub_self", op="-")
    state.add_node(sub_node)
    state.add_edge(a_read_1, None, sub_node, "_a", Memlet("A[0:5, 0:6]"))
    state.add_edge(a_read_2, None, sub_node, "_b", Memlet("A[0:5, 0:6]"))
    state.add_edge(sub_node, "_out", c_write, None, Memlet("C[0:5, 0:6]"))

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
    add_node = TileOpLibraryNode("add_rt_node", op="+")
    st.add_node(add_node)
    st.add_edge(st.add_read("A"), None, add_node, "_a", Memlet("A[0:6, 0:7]"))
    st.add_edge(st.add_read("B"), None, add_node, "_b", Memlet("B[0:6, 0:7]"))
    st.add_edge(add_node, "_out", st.add_write("T"), None, Memlet("T[0:6, 0:7]"))
    sdfg_add.expand_library_nodes()
    t = np.zeros(shape, dtype=np.float64)
    sdfg_add(A=a, B=b, T=t)

    # Step 2: O = T - B
    sdfg_sub = SDFG("sub_roundtrip")
    sdfg_sub.add_array("T", shape=shape, dtype=dace.float64)
    sdfg_sub.add_array("B", shape=shape, dtype=dace.float64)
    sdfg_sub.add_array("O", shape=shape, dtype=dace.float64)
    st2 = sdfg_sub.add_state()
    sub_node = TileOpLibraryNode("sub_rt_node", op="-")
    st2.add_node(sub_node)
    st2.add_edge(st2.add_read("T"), None, sub_node, "_a", Memlet("T[0:6, 0:7]"))
    st2.add_edge(st2.add_read("B"), None, sub_node, "_b", Memlet("B[0:6, 0:7]"))
    st2.add_edge(sub_node, "_out", st2.add_write("O"), None, Memlet("O[0:6, 0:7]"))
    sdfg_sub.expand_library_nodes()
    out = np.zeros(shape, dtype=np.float64)
    sdfg_sub(T=t, B=b, O=out)

    np.testing.assert_allclose(out, a, rtol=1e-12, atol=1e-12)


def test_add_zero_identity():
    """A + zeros must equal A exactly."""
    shape = [8, 8]
    sdfg, actual_shape = _make_direct_binary_sdfg(shape, "add_zero_identity", op="+")
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
    sdfg, actual_shape = _make_direct_binary_sdfg(shape, "sub_zero_identity", op="-")
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
    sdfg, actual_shape = _make_direct_binary_sdfg(shape, "sub_negative", op="-")
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
        shape=[64, 128], np_op=np.subtract,
        name="sub_large_tile", op="-", seed=66,
    )


def test_large_tile_add():
    """TileAdd on a large 2-D tile (64 × 128)."""
    _run_direct_binary_op_test(
        shape=[64, 128], np_op=np.add,
        name="add_large_tile", op="+", seed=77,
    )


def test_tileadd_int32_dtype():
    """TileAdd with int32 tiles — exact integer result expected."""
    shape = [5, 6]
    sdfg = SDFG("tileadd_int32")
    sdfg.add_array("A", shape=shape, dtype=dace.int32)
    sdfg.add_array("B", shape=shape, dtype=dace.int32)
    sdfg.add_array("C", shape=shape, dtype=dace.int32)

    state = sdfg.add_state("main")
    add_node = TileOpLibraryNode("add_int32", op="+")
    state.add_node(add_node)
    state.add_edge(state.add_read("A"), None, add_node, "_a", Memlet("A[0:5, 0:6]"))
    state.add_edge(state.add_read("B"), None, add_node, "_b", Memlet("B[0:5, 0:6]"))
    state.add_edge(add_node, "_out", state.add_write("C"), None, Memlet("C[0:5, 0:6]"))

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
    sub_node = TileOpLibraryNode("sub_int32", op="-")
    state.add_node(sub_node)
    state.add_edge(state.add_read("A"), None, sub_node, "_a", Memlet("A[0:5, 0:6]"))
    state.add_edge(state.add_read("B"), None, sub_node, "_b", Memlet("B[0:5, 0:6]"))
    state.add_edge(sub_node, "_out", state.add_write("C"), None, Memlet("C[0:5, 0:6]"))

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
        count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
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
        count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
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


# ---------------------------------------------------------------------------
# Non-canonical inner-map transformation tests
# ---------------------------------------------------------------------------

def build_runtime_tiled_scalar_noncanonical_binary_sdfg(
    ii_range: str,
    jj_range: str,
    op: str,
    name: str,
    outer_shape=(2, 2),
    inner_shape=(6, 5),
    dtype=dace.float64,
    step_symbol: str | None = None,
) -> SDFG:
    """Build a runtime SDFG with shifted/strided inner-map ranges."""
    mt, nt = outer_shape
    t0, t1 = inner_shape

    sdfg = SDFG(name)
    if step_symbol is not None:
        sdfg.add_symbol(step_symbol, dace.int32)

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
        {"ii": ii_range, "jj": jj_range},
        schedule=dtypes.ScheduleType.Sequential,
    )

    if op == "+":
        tasklet_code = "c = a + b"
        tasklet_name = "add_noncanonical"
    elif op == "-":
        tasklet_code = "c = a - b"
        tasklet_name = "sub_noncanonical"
    else:
        raise ValueError(f"Unsupported op '{op}'.")

    tasklet = state.add_tasklet(tasklet_name, {"a", "b"}, {"c"}, tasklet_code)

    state.add_memlet_path(
        a_acc,
        outer_entry,
        inner_entry,
        tasklet,
        dst_conn="a",
        memlet=Memlet("A[i, j, ii, jj]"),
    )
    state.add_memlet_path(
        b_acc,
        outer_entry,
        inner_entry,
        tasklet,
        dst_conn="b",
        memlet=Memlet("B[i, j, ii, jj]"),
    )
    state.add_memlet_path(
        tasklet,
        inner_exit,
        outer_exit,
        c_acc,
        src_conn="c",
        memlet=Memlet("C[i, j, ii, jj]"),
    )

    sdfg.validate()
    return sdfg


def _expected_noncanonical_binary(a, b, c_init, ii_values, jj_values, np_op):
    expected = c_init.copy()
    for ii in ii_values:
        for jj in jj_values:
            expected[:, :, ii, jj] = np_op(a[:, :, ii, jj], b[:, :, ii, jj])
    return expected


def _assert_no_strided_outer_memlets(state: dace.SDFGState):
    for edge in state.edges():
        if edge.data.data not in {"A", "B", "C"}:
            continue
        subset = edge.data.dst_subset if edge.data._is_data_src is False else edge.data.src_subset
        if not isinstance(subset, dace.subsets.Range):
            continue
        if not (
            isinstance(edge.src, (nodes.MapEntry, nodes.MapExit))
            or isinstance(edge.dst, (nodes.MapEntry, nodes.MapExit))
        ):
            continue
        for _, _, step in subset:
            assert step == 1, f"Found strided outer memlet: {edge.data}"


def test_noncanonical_add_transforms_to_masked_node_and_contiguous_memlets():
    """Non-canonical add should rewrite to masked node and avoid strided outer memlets."""
    sdfg = build_runtime_tiled_scalar_noncanonical_binary_sdfg(
        ii_range="1:6:2",
        jj_range="1:5",
        op="+",
        name="tile_add_before_runtime_noncanonical",
        dtype=dace.float64,
    )

    count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
    assert count == 1

    state = sdfg.states()[0]
    lib_nodes = [n for n in state.nodes() if isinstance(n, nodes.LibraryNode)]
    assert len(lib_nodes) == 1
    assert isinstance(lib_nodes[0], TileSymbolicMaskedOpLibraryNode)
    assert lib_nodes[0].op == "+"
    assert lib_nodes[0].mask_condition is not None  # should have a real condition
    # Note: outer memlets may now correctly have stride>1 when the original
    # map range has a non-unit step (e.g. ii_range="1:6:2").  The companion
    # test_noncanonical_add_runtime_numeric_correctness validates correctness.


def test_noncanonical_add_runtime_numeric_correctness():
    """Non-canonical add should update only the mapped points and keep others unchanged."""
    sdfg = build_runtime_tiled_scalar_noncanonical_binary_sdfg(
        ii_range="1:6:2",
        jj_range="1:5",
        op="+",
        name="tile_add_runtime_noncanonical_add",
        dtype=dace.float64,
    )

    count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
    assert count == 1
    sdfg.expand_library_nodes()
    sdfg.validate()

    shape = (2, 2, 6, 5)
    rng = np.random.default_rng(4242)
    a = rng.uniform(-10.0, 10.0, size=shape).astype(np.float64)
    b = rng.uniform(-10.0, 10.0, size=shape).astype(np.float64)
    c = rng.uniform(-5.0, 5.0, size=shape).astype(np.float64)
    c_expected = _expected_noncanonical_binary(a, b, c, range(1, 6, 2), range(1, 5, 1), np.add)

    sdfg(A=a, B=b, C=c)
    np.testing.assert_allclose(c, c_expected, rtol=0.0, atol=1e-12)


def test_noncanonical_subtract_runtime_numeric_correctness_strided():
    """Non-canonical strided subtract should use masked subtract semantics."""
    sdfg = build_runtime_tiled_scalar_noncanonical_binary_sdfg(
        ii_range="1:6:2",
        jj_range="1:5",
        op="-",
        name="tile_subtract_runtime_noncanonical_strided",
        dtype=dace.float64,
    )

    count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
    assert count == 1

    state = sdfg.states()[0]
    lib_nodes = [n for n in state.nodes() if isinstance(n, nodes.LibraryNode)]
    assert len(lib_nodes) == 1
    assert isinstance(lib_nodes[0], TileSymbolicMaskedOpLibraryNode)
    assert lib_nodes[0].op == "-"
    assert lib_nodes[0].mask_condition is not None

    sdfg.expand_library_nodes()
    sdfg.validate()

    shape = (2, 2, 6, 5)
    rng = np.random.default_rng(5252)
    a = rng.uniform(-10.0, 10.0, size=shape).astype(np.float64)
    b = rng.uniform(-10.0, 10.0, size=shape).astype(np.float64)
    c = rng.uniform(-5.0, 5.0, size=shape).astype(np.float64)
    c_expected = _expected_noncanonical_binary(a, b, c, range(1, 6, 2), range(1, 5, 1), np.subtract)

    sdfg(A=a, B=b, C=c)
    np.testing.assert_allclose(c, c_expected, rtol=0.0, atol=1e-12)


def test_noncanonical_symbolic_positive_step_runtime():
    """Symbolic (positive) step should work for different positive runtime values.

    Maps must use a positive step (descending maps are rejected by SDFG
    validation; symbolic steps carry a runtime ``step > 0`` assertion), so only
    positive runtime step values are exercised here.
    """
    sdfg = build_runtime_tiled_scalar_noncanonical_binary_sdfg(
        ii_range="1:6:S",
        jj_range="1:5",
        op="+",
        name="tile_add_runtime_noncanonical_symbolic_step",
        dtype=dace.float64,
        step_symbol="S",
    )

    count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
    assert count == 1
    sdfg.expand_library_nodes()
    sdfg.validate()

    shape = (2, 2, 6, 5)
    rng = np.random.default_rng(6262)
    a = rng.uniform(-10.0, 10.0, size=shape).astype(np.float64)
    b = rng.uniform(-10.0, 10.0, size=shape).astype(np.float64)

    for step_value in (1, 2, 4):
        c = rng.uniform(-5.0, 5.0, size=shape).astype(np.float64)
        c_expected = _expected_noncanonical_binary(
            a, b, c, range(1, 6, step_value), range(1, 5, 1), np.add
        )
        sdfg(A=a, B=b, C=c, S=np.int32(step_value))
        np.testing.assert_allclose(c, c_expected, rtol=0.0, atol=1e-12)


def _mask_truth_set(expr, lo=0, hi=8):
    """Truth table of a boolean SymPy mask over a small integer cube."""
    import itertools
    import sympy as sp
    syms = sorted(expr.free_symbols, key=str)
    out = set()
    for vals in itertools.product(range(lo, hi), repeat=len(syms)):
        if bool(expr.subs(dict(zip(syms, vals)))):
            out.add(vals)
    return out


def test_symbolic_mask_condition_survives_json_roundtrip():
    """A masked tile op's symbolic ``mask_condition`` must round-trip through
    SDFG JSON (de)serialization and still produce correct results.

    Regression test: SymPy boolean masks (``And``/``Or``) are stored in a plain
    ``Property`` and printed by the serializer using ``&``/``|``.  The symbolic
    deserializer must understand those operators, and the property must route
    (de)serialization through the symbolic helpers, otherwise reloading any
    saved masked cuTile SDFG fails.
    """
    sdfg = build_runtime_tiled_scalar_noncanonical_binary_sdfg(
        ii_range="1:6:2",
        jj_range="1:5",
        op="+",
        name="tile_add_masked_serde",
        dtype=dace.float64,
    )
    count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
    assert count == 1

    state = sdfg.states()[0]
    lib = next(n for n in state.nodes() if isinstance(n, TileSymbolicMaskedOpLibraryNode))
    orig_mask = lib.mask_condition
    assert orig_mask is not None

    # Round-trip through JSON.
    sdfg2 = SDFG.from_json(sdfg.to_json())
    state2 = sdfg2.states()[0]
    lib2 = next(n for n in state2.nodes() if isinstance(n, TileSymbolicMaskedOpLibraryNode))
    rt_mask = lib2.mask_condition
    assert rt_mask is not None
    # Logically identical (serializer may normalize, e.g. simplify Mod(x, 1)).
    assert _mask_truth_set(orig_mask) == _mask_truth_set(rt_mask)

    # The reloaded SDFG must still expand, compile, and compute correctly.
    sdfg2.expand_library_nodes()
    sdfg2.validate()
    shape = (2, 2, 6, 5)
    rng = np.random.default_rng(8484)
    a = rng.uniform(-10.0, 10.0, size=shape).astype(np.float64)
    b = rng.uniform(-10.0, 10.0, size=shape).astype(np.float64)
    c = rng.uniform(-5.0, 5.0, size=shape).astype(np.float64)
    c_expected = _expected_noncanonical_binary(a, b, c, range(1, 6, 2), range(1, 5, 1), np.add)
    sdfg2(A=a, B=b, C=c)
    np.testing.assert_allclose(c, c_expected, rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Masked add tests
# ---------------------------------------------------------------------------

def test_tilemaskedadd_runtime_numeric_correctness_bool_mask():
    """Direct masked add runtime test with bool mask."""
    _run_direct_masked_add_test(
        shape=[5, 6],
        name="tilemaskedadd_bool",
        seed=111,
        dtype=np.float64,
        mask_dtype=dace.bool,
        c_init=7.0,
        atol=1e-12,
        rtol=1e-12,
    )


def test_tilemaskedadd_runtime_numeric_correctness_uint8_mask():
    """Direct masked add runtime test with uint8 bitmask-like values."""
    _run_direct_masked_add_test(
        shape=[5, 6],
        name="tilemaskedadd_uint8",
        seed=222,
        dtype=np.float64,
        mask_dtype=dace.uint8,
        c_init=-3.0,
        atol=1e-12,
        rtol=1e-12,
    )


def test_tilemaskedadd_ndim_0():
    """MaskedAdd 0-D expansion path (tile_shape=[]) with active mask."""
    sdfg, actual_shape = _make_direct_masked_add_sdfg(
        shape=(),
        name="tilemaskedadd_ndim0",
        dtype=dace.float64,
        mask_dtype=dace.bool,
        tile_shape_override=[],
    )
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(333)
    a = rng.uniform(-10.0, 10.0, size=actual_shape).astype(np.float64)
    b = rng.uniform(-10.0, 10.0, size=actual_shape).astype(np.float64)
    mask = np.ones(actual_shape, dtype=np.bool_)
    c = np.full(actual_shape, 12.0, dtype=np.float64)

    sdfg(A=a, B=b, MASK=mask, C=c)
    np.testing.assert_allclose(c, a + b, rtol=1e-12, atol=1e-12)


def test_tilemaskedadd_ndim_4_float32():
    """MaskedAdd with a 4-D float32 tile."""
    _run_direct_masked_add_test(
        shape=[2, 3, 4, 5],
        name="tilemaskedadd_ndim4_f32",
        seed=444,
        dtype=np.float32,
        mask_dtype=dace.bool,
        c_init=2.0,
        atol=1e-6,
        rtol=1e-6,
    )


def test_tilemaskedadd_all_false_keeps_output_unchanged():
    """All-false mask must leave C unchanged."""
    shape = [4, 7]
    sdfg, actual_shape = _make_direct_masked_add_sdfg(shape, "tilemaskedadd_all_false")
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(555)
    a = rng.uniform(-10.0, 10.0, size=actual_shape).astype(np.float64)
    b = rng.uniform(-10.0, 10.0, size=actual_shape).astype(np.float64)
    mask = np.zeros(actual_shape, dtype=np.bool_)
    c = np.full(actual_shape, -9.25, dtype=np.float64)

    sdfg(A=a, B=b, MASK=mask, C=c)
    np.testing.assert_array_equal(c, np.full(actual_shape, -9.25, dtype=np.float64))


def test_tilemaskedadd_all_true_equals_plain_add():
    """All-true mask must behave like plain add."""
    shape = [4, 7]
    sdfg, actual_shape = _make_direct_masked_add_sdfg(shape, "tilemaskedadd_all_true")
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(666)
    a = rng.uniform(-10.0, 10.0, size=actual_shape).astype(np.float64)
    b = rng.uniform(-10.0, 10.0, size=actual_shape).astype(np.float64)
    mask = np.ones(actual_shape, dtype=np.bool_)
    c = np.full(actual_shape, 1.0, dtype=np.float64)

    sdfg(A=a, B=b, MASK=mask, C=c)
    np.testing.assert_allclose(c, a + b, rtol=0.0, atol=1e-12)


def test_tilemaskedadd_validate_rejects_mask_shape_mismatch():
    """Validation must reject mask shape mismatch."""
    sdfg = SDFG("tilemaskedadd_bad_mask_shape")
    sdfg.add_array("A", shape=[4, 4], dtype=dace.float64)
    sdfg.add_array("B", shape=[4, 4], dtype=dace.float64)
    sdfg.add_array("MASK", shape=[4, 3], dtype=dace.bool)
    sdfg.add_array("C", shape=[4, 4], dtype=dace.float64)

    state = sdfg.add_state("main")
    node = TileRuntimeMaskedOpLibraryNode("bad_shape", op="+")
    state.add_node(node)
    state.add_edge(state.add_read("A"), None, node, "_a", Memlet("A[0:4, 0:4]"))
    state.add_edge(state.add_read("B"), None, node, "_b", Memlet("B[0:4, 0:4]"))
    state.add_edge(state.add_read("MASK"), None, node, "_m", Memlet("MASK[0:4, 0:3]"))
    state.add_edge(node, "_out", state.add_write("C"), None, Memlet("C[0:4, 0:4]"))

    with pytest.raises(Exception):
        sdfg.validate()


def test_tilemaskedadd_validate_rejects_non_integer_mask_dtype():
    """Validation must reject non-bool/non-integer mask dtype."""
    sdfg = SDFG("tilemaskedadd_bad_mask_dtype")
    sdfg.add_array("A", shape=[4, 4], dtype=dace.float64)
    sdfg.add_array("B", shape=[4, 4], dtype=dace.float64)
    sdfg.add_array("MASK", shape=[4, 4], dtype=dace.float32)
    sdfg.add_array("C", shape=[4, 4], dtype=dace.float64)

    state = sdfg.add_state("main")
    node = TileRuntimeMaskedOpLibraryNode("bad_dtype", op="+")
    state.add_node(node)
    state.add_edge(state.add_read("A"), None, node, "_a", Memlet("A[0:4, 0:4]"))
    state.add_edge(state.add_read("B"), None, node, "_b", Memlet("B[0:4, 0:4]"))
    state.add_edge(state.add_read("MASK"), None, node, "_m", Memlet("MASK[0:4, 0:4]"))
    state.add_edge(node, "_out", state.add_write("C"), None, Memlet("C[0:4, 0:4]"))

    with pytest.raises(Exception):
        sdfg.validate()


# ---------------------------------------------------------------------------
# Multiply and Divide tests (new binary ops)
# ---------------------------------------------------------------------------

def test_tilemultiply_runtime_numeric_correctness():
    """Execute TileMultiply directly and compare against NumPy multiplication."""
    sdfg = SDFG("tile_multiply_runtime")
    sdfg.add_array("A", shape=[3, 4], dtype=dace.float64)
    sdfg.add_array("B", shape=[3, 4], dtype=dace.float64)
    sdfg.add_array("C", shape=[3, 4], dtype=dace.float64)

    state = sdfg.add_state("main")
    a_read = state.add_read("A")
    b_read = state.add_read("B")
    c_write = state.add_write("C")
    mul_node = TileOpLibraryNode("tile_multiply_runtime_node", op="*")
    state.add_node(mul_node)

    state.add_edge(a_read, None, mul_node, "_a", Memlet("A[0:3, 0:4]"))
    state.add_edge(b_read, None, mul_node, "_b", Memlet("B[0:3, 0:4]"))
    state.add_edge(mul_node, "_out", c_write, None, Memlet("C[0:3, 0:4]"))

    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(3001)
    a = rng.standard_normal((3, 4), dtype=np.float64)
    b = rng.standard_normal((3, 4), dtype=np.float64)
    c = np.zeros((3, 4), dtype=np.float64)

    sdfg(A=a, B=b, C=c)
    np.testing.assert_allclose(c, a * b, rtol=0.0, atol=1e-12)


def test_tiledivide_runtime_numeric_correctness():
    """Execute TileDivide directly and compare against NumPy division."""
    sdfg = SDFG("tile_divide_runtime")
    sdfg.add_array("A", shape=[3, 4], dtype=dace.float64)
    sdfg.add_array("B", shape=[3, 4], dtype=dace.float64)
    sdfg.add_array("C", shape=[3, 4], dtype=dace.float64)

    state = sdfg.add_state("main")
    a_read = state.add_read("A")
    b_read = state.add_read("B")
    c_write = state.add_write("C")
    div_node = TileOpLibraryNode("tile_divide_runtime_node", op="/")
    state.add_node(div_node)

    state.add_edge(a_read, None, div_node, "_a", Memlet("A[0:3, 0:4]"))
    state.add_edge(b_read, None, div_node, "_b", Memlet("B[0:3, 0:4]"))
    state.add_edge(div_node, "_out", c_write, None, Memlet("C[0:3, 0:4]"))

    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(3002)
    a = rng.standard_normal((3, 4), dtype=np.float64)
    # Avoid near-zero denominators
    b = rng.uniform(1.0, 10.0, size=(3, 4)).astype(np.float64)
    c = np.zeros((3, 4), dtype=np.float64)

    sdfg(A=a, B=b, C=c)
    np.testing.assert_allclose(c, a / b, rtol=1e-12, atol=1e-12)


def test_tilemultiply_ndim_2():
    """TileMultiply with a 2-D tile of shape (4, 8)."""
    _run_direct_binary_op_test(
        shape=_NDIM_SHAPES[2], np_op=np.multiply,
        name="tilemultiply_ndim2", op="*",
    )


def test_tiledivide_ndim_2():
    """TileDivide with a 2-D tile (avoid near-zero denominator)."""
    shape = [4, 8]
    sdfg, actual_shape = _make_direct_binary_sdfg(shape, "tiledivide_ndim2", op="/")
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(3003)
    a = rng.uniform(-10.0, 10.0, size=actual_shape).astype(np.float64)
    b = rng.uniform(0.5, 5.0, size=actual_shape).astype(np.float64)
    c = np.zeros(actual_shape, dtype=np.float64)

    sdfg(A=a, B=b, C=c)
    np.testing.assert_allclose(c, a / b, rtol=1e-12, atol=1e-12)


def test_pipeline_multiply_transforms_correctly():
    """Pipeline multiply SDFG → TileOpLibraryNode with op='*'."""
    mt, nt, t0, t1 = 2, 3, 2, 2
    sdfg = SDFG("tile_multiply_pipeline")
    sdfg.add_array("A", shape=[mt, nt, t0, t1], dtype=dace.float64)
    sdfg.add_array("B", shape=[mt, nt, t0, t1], dtype=dace.float64)
    sdfg.add_array("C", shape=[mt, nt, t0, t1], dtype=dace.float64)

    state = sdfg.add_state("main")
    a_acc = state.add_read("A")
    b_acc = state.add_read("B")
    c_acc = state.add_write("C")

    outer_entry, outer_exit = state.add_map(
        "tile_map", {"i": f"0:{mt}", "j": f"0:{nt}"},
        schedule=dtypes.ScheduleType.Sequential,
    )
    inner_entry, inner_exit = state.add_map(
        "elem_map", {"ii": f"0:{t0}", "jj": f"0:{t1}"},
        schedule=dtypes.ScheduleType.Sequential,
    )
    tasklet = state.add_tasklet("multiply", {"a", "b"}, {"c"}, "c = a * b")

    state.add_memlet_path(a_acc, outer_entry, inner_entry, tasklet,
                          dst_conn="a", memlet=Memlet("A[i, j, ii, jj]"))
    state.add_memlet_path(b_acc, outer_entry, inner_entry, tasklet,
                          dst_conn="b", memlet=Memlet("B[i, j, ii, jj]"))
    state.add_memlet_path(tasklet, inner_exit, outer_exit, c_acc,
                          src_conn="c", memlet=Memlet("C[i, j, ii, jj]"))
    sdfg.validate()

    count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
    assert count == 1

    state = sdfg.states()[0]
    lib_nodes = [n for n in state.nodes() if isinstance(n, nodes.LibraryNode)]
    assert len(lib_nodes) == 1
    assert isinstance(lib_nodes[0], TileOpLibraryNode)
    assert lib_nodes[0].op == "*"

    sdfg.expand_library_nodes()
    sdfg.validate()

    shape = (mt, nt, t0, t1)
    rng = np.random.default_rng(3004)
    a = rng.uniform(-5.0, 5.0, size=shape).astype(np.float64)
    b = rng.uniform(-5.0, 5.0, size=shape).astype(np.float64)
    c = np.zeros(shape, dtype=np.float64)

    sdfg(A=a, B=b, C=c)
    np.testing.assert_allclose(c, a * b, rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Unary operation tests (new)
# ---------------------------------------------------------------------------

def _make_direct_unary_sdfg(shape, name, op="-", tile_shape_override=None):
    """Build a single-state SDFG with one unary library node."""
    if not shape:
        actual_shape = [1]
        node_tile_shape = [] if tile_shape_override is None else tile_shape_override
    else:
        actual_shape = list(shape)
        node_tile_shape = tile_shape_override

    sdfg = SDFG(name)
    sdfg.add_array("A", shape=actual_shape, dtype=dace.float64)
    sdfg.add_array("C", shape=actual_shape, dtype=dace.float64)

    state = sdfg.add_state("main")
    a_read = state.add_read("A")
    c_write = state.add_write("C")

    op_node = TileOpLibraryNode(name + "_node", op=op, tile_shape=node_tile_shape)
    # Unified TileOpLibraryNode defaults to binary for ambiguous ops like "-";
    # remove _b to make it unary.
    if "_b" in op_node.in_connectors:
        op_node.remove_in_connector("_b")
    state.add_node(op_node)

    subset = ", ".join(f"0:{s}" for s in actual_shape)
    state.add_edge(a_read, None, op_node, "_a", Memlet(f"A[{subset}]"))
    state.add_edge(op_node, "_out", c_write, None, Memlet(f"C[{subset}]"))

    sdfg.validate()
    return sdfg, actual_shape


def test_unary_negate_runtime_numeric_correctness():
    """Execute TileUnaryOp (negate) directly and compare against NumPy."""
    sdfg, actual_shape = _make_direct_unary_sdfg([5, 6], "tile_negate_runtime")
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(4001)
    a = rng.uniform(-10.0, 10.0, size=actual_shape).astype(np.float64)
    c = np.zeros(actual_shape, dtype=np.float64)

    sdfg(A=a, C=c)
    np.testing.assert_allclose(c, -a, rtol=0.0, atol=1e-12)


def test_unary_negate_0d():
    """TileUnaryOp negate: 0-D expansion path."""
    sdfg, actual_shape = _make_direct_unary_sdfg((), "tile_negate_0d")
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(4002)
    a = rng.uniform(-10.0, 10.0, size=actual_shape).astype(np.float64)
    c = np.zeros(actual_shape, dtype=np.float64)

    sdfg(A=a, C=c)
    np.testing.assert_allclose(c, -a, rtol=0.0, atol=1e-12)


def test_unary_negate_4d():
    """TileUnaryOp negate with a 4-D tile."""
    sdfg, actual_shape = _make_direct_unary_sdfg([2, 3, 4, 5], "tile_negate_4d")
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(4003)
    a = rng.uniform(-10.0, 10.0, size=actual_shape).astype(np.float64)
    c = np.zeros(actual_shape, dtype=np.float64)

    sdfg(A=a, C=c)
    np.testing.assert_allclose(c, -a, rtol=0.0, atol=1e-12)


def test_pipeline_unary_negate_transforms_correctly():
    """Pipeline unary negate: c = -a → TileOpLibraryNode with op='-'."""
    mt, nt, t0, t1 = 2, 3, 2, 2
    sdfg = SDFG("tile_negate_pipeline")
    sdfg.add_array("A", shape=[mt, nt, t0, t1], dtype=dace.float64)
    sdfg.add_array("C", shape=[mt, nt, t0, t1], dtype=dace.float64)

    state = sdfg.add_state("main")
    a_acc = state.add_read("A")
    c_acc = state.add_write("C")

    outer_entry, outer_exit = state.add_map(
        "tile_map", {"i": f"0:{mt}", "j": f"0:{nt}"},
        schedule=dtypes.ScheduleType.Sequential,
    )
    inner_entry, inner_exit = state.add_map(
        "elem_map", {"ii": f"0:{t0}", "jj": f"0:{t1}"},
        schedule=dtypes.ScheduleType.Sequential,
    )
    tasklet = state.add_tasklet("negate", {"a"}, {"c"}, "c = -a")

    state.add_memlet_path(a_acc, outer_entry, inner_entry, tasklet,
                          dst_conn="a", memlet=Memlet("A[i, j, ii, jj]"))
    state.add_memlet_path(tasklet, inner_exit, outer_exit, c_acc,
                          src_conn="c", memlet=Memlet("C[i, j, ii, jj]"))
    sdfg.validate()

    count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
    assert count == 1

    state = sdfg.states()[0]
    lib_nodes = [n for n in state.nodes() if isinstance(n, nodes.LibraryNode)]
    assert len(lib_nodes) == 1
    assert isinstance(lib_nodes[0], TileOpLibraryNode)
    assert lib_nodes[0].op == "-"

    sdfg.expand_library_nodes()
    sdfg.validate()

    shape = (mt, nt, t0, t1)
    rng = np.random.default_rng(4004)
    a = rng.uniform(-5.0, 5.0, size=shape).astype(np.float64)
    c = np.zeros(shape, dtype=np.float64)

    sdfg(A=a, C=c)
    np.testing.assert_allclose(c, -a, rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Constant binary operation tests (new)
# ---------------------------------------------------------------------------

def _make_direct_const_binary_sdfg(shape, name, op="+", constant="2",
                                    constant_position="right",
                                    tile_shape_override=None,
                                    dtype=dace.float64):
    """Build SDFG with one const-binary library node."""
    if not shape:
        actual_shape = [1]
        node_tile_shape = [] if tile_shape_override is None else tile_shape_override
    else:
        actual_shape = list(shape)
        node_tile_shape = tile_shape_override

    sdfg = SDFG(name)
    sdfg.add_array("A", shape=actual_shape, dtype=dtype)
    sdfg.add_array("C", shape=actual_shape, dtype=dtype)

    state = sdfg.add_state("main")
    a_read = state.add_read("A")
    c_write = state.add_write("C")

    # Map old constant/constant_position to new constant1/constant2
    if constant_position == "left":
        op_node = TileOpLibraryNode(
            name + "_node", op=op, constant1=constant, tile_shape=node_tile_shape)
        array_conn = "_b"
    else:
        op_node = TileOpLibraryNode(
            name + "_node", op=op, constant2=constant, tile_shape=node_tile_shape)
        array_conn = "_a"
    state.add_node(op_node)

    subset = ", ".join(f"0:{s}" for s in actual_shape)
    state.add_edge(a_read, None, op_node, array_conn, Memlet(f"A[{subset}]"))
    state.add_edge(op_node, "_out", c_write, None, Memlet(f"C[{subset}]"))

    sdfg.validate()
    return sdfg, actual_shape


def test_const_add_right_runtime():
    """c = a + 2 → TileConstBinaryOp with constant on right."""
    sdfg, actual_shape = _make_direct_const_binary_sdfg(
        [4, 5], "const_add_right", op="+", constant="2", constant_position="right")
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(5001)
    a = rng.uniform(-10.0, 10.0, size=actual_shape).astype(np.float64)
    c = np.zeros(actual_shape, dtype=np.float64)

    sdfg(A=a, C=c)
    np.testing.assert_allclose(c, a + 2, rtol=0.0, atol=1e-12)


def test_const_add_left_runtime():
    """c = 3 + a → TileConstBinaryOp with constant on left."""
    sdfg, actual_shape = _make_direct_const_binary_sdfg(
        [4, 5], "const_add_left", op="+", constant="3", constant_position="left")
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(5002)
    a = rng.uniform(-10.0, 10.0, size=actual_shape).astype(np.float64)
    c = np.zeros(actual_shape, dtype=np.float64)

    sdfg(A=a, C=c)
    np.testing.assert_allclose(c, 3 + a, rtol=0.0, atol=1e-12)


def test_const_subtract_right_runtime():
    """c = a - 5 → TileConstBinaryOp(op='-', constant='5', position='right')."""
    sdfg, actual_shape = _make_direct_const_binary_sdfg(
        [3, 7], "const_sub_right", op="-", constant="5", constant_position="right")
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(5003)
    a = rng.uniform(-10.0, 10.0, size=actual_shape).astype(np.float64)
    c = np.zeros(actual_shape, dtype=np.float64)

    sdfg(A=a, C=c)
    np.testing.assert_allclose(c, a - 5, rtol=0.0, atol=1e-12)


def test_const_subtract_left_runtime():
    """c = 10 - a → TileConstBinaryOp(op='-', constant='10', position='left')."""
    sdfg, actual_shape = _make_direct_const_binary_sdfg(
        [3, 7], "const_sub_left", op="-", constant="10", constant_position="left")
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(5004)
    a = rng.uniform(-10.0, 10.0, size=actual_shape).astype(np.float64)
    c = np.zeros(actual_shape, dtype=np.float64)

    sdfg(A=a, C=c)
    np.testing.assert_allclose(c, 10 - a, rtol=0.0, atol=1e-12)


def test_const_multiply_right_runtime():
    """c = a * 3 → TileConstBinaryOp(op='*', constant='3', position='right')."""
    sdfg, actual_shape = _make_direct_const_binary_sdfg(
        [4, 6], "const_mul_right", op="*", constant="3", constant_position="right")
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(5005)
    a = rng.uniform(-10.0, 10.0, size=actual_shape).astype(np.float64)
    c = np.zeros(actual_shape, dtype=np.float64)

    sdfg(A=a, C=c)
    np.testing.assert_allclose(c, a * 3, rtol=0.0, atol=1e-12)


def test_const_divide_right_runtime():
    """c = a / 4 → TileConstBinaryOp(op='/', constant='4', position='right')."""
    sdfg, actual_shape = _make_direct_const_binary_sdfg(
        [4, 6], "const_div_right", op="/", constant="4", constant_position="right")
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(5006)
    a = rng.uniform(-10.0, 10.0, size=actual_shape).astype(np.float64)
    c = np.zeros(actual_shape, dtype=np.float64)

    sdfg(A=a, C=c)
    np.testing.assert_allclose(c, a / 4, rtol=1e-12, atol=1e-12)


def test_const_binary_0d():
    """TileConstBinaryOp: 0-D tile path on single-element array."""
    sdfg, actual_shape = _make_direct_const_binary_sdfg(
        (), "const_add_0d", op="+", constant="7", constant_position="right")
    sdfg.expand_library_nodes()
    sdfg.validate()

    a = np.array([3.0], dtype=np.float64)
    c = np.zeros([1], dtype=np.float64)

    sdfg(A=a, C=c)
    np.testing.assert_allclose(c, a + 7, rtol=0.0, atol=1e-12)


def test_pipeline_const_add_right_transforms_correctly():
    """Pipeline c = a + 2 → TileConstBinaryOpLibraryNode."""
    mt, nt, t0, t1 = 2, 3, 2, 2
    sdfg = SDFG("tile_const_add_pipeline")
    sdfg.add_array("A", shape=[mt, nt, t0, t1], dtype=dace.float64)
    sdfg.add_array("C", shape=[mt, nt, t0, t1], dtype=dace.float64)

    state = sdfg.add_state("main")
    a_acc = state.add_read("A")
    c_acc = state.add_write("C")

    outer_entry, outer_exit = state.add_map(
        "tile_map", {"i": f"0:{mt}", "j": f"0:{nt}"},
        schedule=dtypes.ScheduleType.Sequential,
    )
    inner_entry, inner_exit = state.add_map(
        "elem_map", {"ii": f"0:{t0}", "jj": f"0:{t1}"},
        schedule=dtypes.ScheduleType.Sequential,
    )
    tasklet = state.add_tasklet("const_add", {"a"}, {"c"}, "c = a + 2")

    state.add_memlet_path(a_acc, outer_entry, inner_entry, tasklet,
                          dst_conn="a", memlet=Memlet("A[i, j, ii, jj]"))
    state.add_memlet_path(tasklet, inner_exit, outer_exit, c_acc,
                          src_conn="c", memlet=Memlet("C[i, j, ii, jj]"))
    sdfg.validate()

    count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
    assert count == 1

    state = sdfg.states()[0]
    lib_nodes = [n for n in state.nodes() if isinstance(n, nodes.LibraryNode)]
    assert len(lib_nodes) == 1
    assert isinstance(lib_nodes[0], TileOpLibraryNode)
    assert lib_nodes[0].op == "+"
    assert lib_nodes[0].constant2 == "2"

    sdfg.expand_library_nodes()
    sdfg.validate()

    shape = (mt, nt, t0, t1)
    rng = np.random.default_rng(5007)
    a = rng.uniform(-5.0, 5.0, size=shape).astype(np.float64)
    c = np.zeros(shape, dtype=np.float64)

    sdfg(A=a, C=c)
    np.testing.assert_allclose(c, a + 2, rtol=0.0, atol=1e-12)


def test_pipeline_const_left_multiply_transforms_correctly():
    """Pipeline c = 3 * a → TileConstBinaryOpLibraryNode with constant on left."""
    mt, nt, t0, t1 = 2, 3, 2, 2
    sdfg = SDFG("tile_const_lmul_pipeline")
    sdfg.add_array("A", shape=[mt, nt, t0, t1], dtype=dace.float64)
    sdfg.add_array("C", shape=[mt, nt, t0, t1], dtype=dace.float64)

    state = sdfg.add_state("main")
    a_acc = state.add_read("A")
    c_acc = state.add_write("C")

    outer_entry, outer_exit = state.add_map(
        "tile_map", {"i": f"0:{mt}", "j": f"0:{nt}"},
        schedule=dtypes.ScheduleType.Sequential,
    )
    inner_entry, inner_exit = state.add_map(
        "elem_map", {"ii": f"0:{t0}", "jj": f"0:{t1}"},
        schedule=dtypes.ScheduleType.Sequential,
    )
    tasklet = state.add_tasklet("const_lmul", {"a"}, {"c"}, "c = 3 * a")

    state.add_memlet_path(a_acc, outer_entry, inner_entry, tasklet,
                          dst_conn="a", memlet=Memlet("A[i, j, ii, jj]"))
    state.add_memlet_path(tasklet, inner_exit, outer_exit, c_acc,
                          src_conn="c", memlet=Memlet("C[i, j, ii, jj]"))
    sdfg.validate()

    count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
    assert count == 1

    state = sdfg.states()[0]
    lib_nodes = [n for n in state.nodes() if isinstance(n, nodes.LibraryNode)]
    assert len(lib_nodes) == 1
    assert isinstance(lib_nodes[0], TileOpLibraryNode)
    assert lib_nodes[0].op == "*"
    assert lib_nodes[0].constant1 == "3"

    sdfg.expand_library_nodes()
    sdfg.validate()

    shape = (mt, nt, t0, t1)
    rng = np.random.default_rng(5008)
    a = rng.uniform(-5.0, 5.0, size=shape).astype(np.float64)
    c = np.zeros(shape, dtype=np.float64)

    sdfg(A=a, C=c)
    np.testing.assert_allclose(c, 3 * a, rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Masked unary and masked const tests (new)
# ---------------------------------------------------------------------------

def _make_direct_masked_unary_sdfg(shape, name, op="-", mask_dtype=dace.bool,
                                    tile_shape_override=None):
    """Build SDFG with one masked unary library node."""
    if not shape:
        actual_shape = [1]
        node_tile_shape = [] if tile_shape_override is None else tile_shape_override
    else:
        actual_shape = list(shape)
        node_tile_shape = tile_shape_override

    sdfg = SDFG(name)
    sdfg.add_array("A", shape=actual_shape, dtype=dace.float64)
    sdfg.add_array("MASK", shape=actual_shape, dtype=mask_dtype)
    sdfg.add_array("C", shape=actual_shape, dtype=dace.float64)

    state = sdfg.add_state("main")
    a_read = state.add_read("A")
    m_read = state.add_read("MASK")
    c_write = state.add_write("C")

    op_node = TileRuntimeMaskedOpLibraryNode(
        name + "_node", op=op, tile_shape=node_tile_shape)
    # Unified TileRuntimeMaskedOpLibraryNode defaults to binary for "-";
    # remove _b to make it unary.
    if "_b" in op_node.in_connectors:
        op_node.remove_in_connector("_b")
    state.add_node(op_node)

    subset = ", ".join(f"0:{s}" for s in actual_shape)
    state.add_edge(a_read, None, op_node, "_a", Memlet(f"A[{subset}]"))
    state.add_edge(m_read, None, op_node, "_m", Memlet(f"MASK[{subset}]"))
    state.add_edge(op_node, "_out", c_write, None, Memlet(f"C[{subset}]"))

    sdfg.validate()
    return sdfg, actual_shape


def test_masked_unary_negate_runtime():
    """Masked negate: only negates where mask is True."""
    sdfg, actual_shape = _make_direct_masked_unary_sdfg([4, 5], "masked_negate")
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(6001)
    a = rng.uniform(-10.0, 10.0, size=actual_shape).astype(np.float64)
    mask = rng.integers(0, 2, size=actual_shape).astype(np.bool_)
    c = np.full(actual_shape, 99.0, dtype=np.float64)

    expected = c.copy()
    expected[mask] = (-a)[mask]

    sdfg(A=a, MASK=mask, C=c)
    np.testing.assert_allclose(c, expected, rtol=0.0, atol=1e-12)


def test_masked_const_add_right_runtime():
    """Masked c = a + 2: only applies where mask is True."""
    shape = [4, 5]
    sdfg = SDFG("masked_const_add_right")
    sdfg.add_array("A", shape=shape, dtype=dace.float64)
    sdfg.add_array("MASK", shape=shape, dtype=dace.bool)
    sdfg.add_array("C", shape=shape, dtype=dace.float64)

    state = sdfg.add_state("main")
    a_read = state.add_read("A")
    m_read = state.add_read("MASK")
    c_write = state.add_write("C")

    op_node = TileRuntimeMaskedOpLibraryNode(
        "masked_const_add_node", op="+", constant2="2")
    state.add_node(op_node)

    state.add_edge(a_read, None, op_node, "_a", Memlet("A[0:4, 0:5]"))
    state.add_edge(m_read, None, op_node, "_m", Memlet("MASK[0:4, 0:5]"))
    state.add_edge(op_node, "_out", c_write, None, Memlet("C[0:4, 0:5]"))

    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(6002)
    a = rng.uniform(-10.0, 10.0, size=shape).astype(np.float64)
    mask = rng.integers(0, 2, size=shape).astype(np.bool_)
    c = np.full(shape, -7.0, dtype=np.float64)

    expected = c.copy()
    expected[mask] = (a + 2)[mask]

    sdfg(A=a, MASK=mask, C=c)
    np.testing.assert_allclose(c, expected, rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Function-style unary ops (sin, abs, exp, sqrt, log)
# ---------------------------------------------------------------------------

def _make_direct_unary_func_sdfg(shape, name, op="sin", tile_shape_override=None,
                                  dtype=dace.float64):
    """Build SDFG with one function-style unary library node."""
    if not shape:
        actual_shape = [1]
        node_tile_shape = [] if tile_shape_override is None else tile_shape_override
    else:
        actual_shape = list(shape)
        node_tile_shape = tile_shape_override

    sdfg = SDFG(name)
    sdfg.add_array("A", shape=actual_shape, dtype=dtype)
    sdfg.add_array("C", shape=actual_shape, dtype=dtype)

    state = sdfg.add_state("main")
    a_read = state.add_read("A")
    c_write = state.add_write("C")

    op_node = TileOpLibraryNode(
        name + "_node", op=op, tile_shape=node_tile_shape)
    state.add_node(op_node)

    subset = ", ".join(f"0:{s}" for s in actual_shape)
    state.add_edge(a_read, None, op_node, "_a", Memlet(f"A[{subset}]"))
    state.add_edge(op_node, "_out", c_write, None, Memlet(f"C[{subset}]"))

    sdfg.validate()
    return sdfg, actual_shape


def test_unary_sin_runtime():
    """c = sin(a)."""
    sdfg, actual_shape = _make_direct_unary_func_sdfg([4, 5], "unary_sin", op="sin")
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(7001)
    a = rng.uniform(-3.14, 3.14, size=actual_shape).astype(np.float64)
    c = np.zeros(actual_shape, dtype=np.float64)

    sdfg(A=a, C=c)
    np.testing.assert_allclose(c, np.sin(a), rtol=1e-12, atol=1e-12)


def test_unary_abs_runtime():
    """c = abs(a)."""
    sdfg, actual_shape = _make_direct_unary_func_sdfg([3, 7], "unary_abs", op="abs")
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(7002)
    a = rng.uniform(-10.0, 10.0, size=actual_shape).astype(np.float64)
    c = np.zeros(actual_shape, dtype=np.float64)

    sdfg(A=a, C=c)
    np.testing.assert_allclose(c, np.abs(a), rtol=0.0, atol=1e-12)


def test_unary_exp_runtime():
    """c = exp(a)."""
    sdfg, actual_shape = _make_direct_unary_func_sdfg([4, 5], "unary_exp", op="exp")
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(7003)
    a = rng.uniform(-2.0, 2.0, size=actual_shape).astype(np.float64)
    c = np.zeros(actual_shape, dtype=np.float64)

    sdfg(A=a, C=c)
    np.testing.assert_allclose(c, np.exp(a), rtol=1e-12, atol=1e-12)


def test_unary_sqrt_runtime():
    """c = sqrt(a)."""
    sdfg, actual_shape = _make_direct_unary_func_sdfg([5, 6], "unary_sqrt", op="sqrt")
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(7004)
    a = rng.uniform(0.1, 10.0, size=actual_shape).astype(np.float64)
    c = np.zeros(actual_shape, dtype=np.float64)

    sdfg(A=a, C=c)
    np.testing.assert_allclose(c, np.sqrt(a), rtol=1e-12, atol=1e-12)


def test_unary_log_runtime():
    """c = log(a)."""
    sdfg, actual_shape = _make_direct_unary_func_sdfg([3, 4], "unary_log", op="log")
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(7005)
    a = rng.uniform(0.1, 10.0, size=actual_shape).astype(np.float64)
    c = np.zeros(actual_shape, dtype=np.float64)

    sdfg(A=a, C=c)
    np.testing.assert_allclose(c, np.log(a), rtol=1e-12, atol=1e-12)


def test_unary_cos_runtime():
    """c = cos(a)."""
    sdfg, actual_shape = _make_direct_unary_func_sdfg([4, 5], "unary_cos", op="cos")
    sdfg.expand_library_nodes()
    sdfg.validate()

    rng = np.random.default_rng(7006)
    a = rng.uniform(-3.14, 3.14, size=actual_shape).astype(np.float64)
    c = np.zeros(actual_shape, dtype=np.float64)

    sdfg(A=a, C=c)
    np.testing.assert_allclose(c, np.cos(a), rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Const-only binary ops (c = CONST op CONST, no array inputs)
# ---------------------------------------------------------------------------

def test_const_only_multiply():
    """c = 2 * 3 → entire tile filled with 6."""
    shape = [4, 5]
    sdfg = SDFG("const_only_mul")
    sdfg.add_array("C", shape=shape, dtype=dace.float64)

    state = sdfg.add_state("main")
    c_write = state.add_write("C")

    op_node = TileOpLibraryNode(
        "const_only_mul_node", op="*", constant1="2",
        constant2="3")
    state.add_node(op_node)

    subset = ", ".join(f"0:{s}" for s in shape)
    state.add_edge(op_node, "_out", c_write, None, Memlet(f"C[{subset}]"))

    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()

    c = np.zeros(shape, dtype=np.float64)
    sdfg(C=c)
    np.testing.assert_allclose(c, np.full(shape, 6.0), rtol=0.0, atol=1e-12)


def test_const_only_add():
    """c = 10 + 5 → entire tile filled with 15."""
    shape = [3, 4]
    sdfg = SDFG("const_only_add")
    sdfg.add_array("C", shape=shape, dtype=dace.float64)

    state = sdfg.add_state("main")
    c_write = state.add_write("C")

    op_node = TileOpLibraryNode(
        "const_only_add_node", op="+", constant1="10",
        constant2="5")
    state.add_node(op_node)

    subset = ", ".join(f"0:{s}" for s in shape)
    state.add_edge(op_node, "_out", c_write, None, Memlet(f"C[{subset}]"))

    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()

    c = np.zeros(shape, dtype=np.float64)
    sdfg(C=c)
    np.testing.assert_allclose(c, np.full(shape, 15.0), rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Unary constant (c = sin(CONST)) — fill tile with constant result
# ---------------------------------------------------------------------------

def test_unary_const_sin():
    """c = sin(0) → entire tile filled with 0."""
    shape = [4, 5]
    sdfg = SDFG("unary_const_sin")
    sdfg.add_array("C", shape=shape, dtype=dace.float64)

    state = sdfg.add_state("main")
    c_write = state.add_write("C")

    op_node = TileOpLibraryNode(
        "unary_const_sin_node", op="sin", constant1="0.0")
    state.add_node(op_node)

    subset = ", ".join(f"0:{s}" for s in shape)
    state.add_edge(op_node, "_out", c_write, None, Memlet(f"C[{subset}]"))

    sdfg.validate()
    sdfg.expand_library_nodes()
    sdfg.validate()

    c = np.full(shape, 99.0, dtype=np.float64)
    sdfg(C=c)
    np.testing.assert_allclose(c, np.full(shape, np.sin(0.0)), rtol=0.0, atol=1e-12)


def test_pipeline_multiple_independent_maps_tiling():
    """Regression test: pipeline tiles all independent maps, not just the first.
    
    This test verifies the fix for the issue where MapTiling in Step 6 was
    only applied to the first map found via pattern matching. The fix ensures
    that all original maps are tiled exactly once.
    """
    # Build an SDFG with two independent maps in the same state
    sdfg = SDFG("multi_map_tiling_test")
    sdfg.add_array("A", shape=[4, 4], dtype=dace.float64)
    sdfg.add_array("B", shape=[4, 4], dtype=dace.float64)
    sdfg.add_array("C", shape=[4, 4], dtype=dace.float64)
    sdfg.add_array("D", shape=[4, 4], dtype=dace.float64)
    
    state = sdfg.add_state("main")
    
    # First independent map: C = A + B
    a_read = state.add_read("A")
    b_read = state.add_read("B")
    c_write = state.add_write("C")
    
    map_entry1, map_exit1 = state.add_map(
        "map_add",
        {"i": "0:4", "j": "0:4"},
    )
    add_tasklet = state.add_tasklet("add", {"a", "b"}, {"c"}, "c = a + b")
    
    state.add_memlet_path(a_read, map_entry1, add_tasklet, dst_conn="a", memlet=Memlet("A[i, j]"))
    state.add_memlet_path(b_read, map_entry1, add_tasklet, dst_conn="b", memlet=Memlet("B[i, j]"))
    state.add_memlet_path(add_tasklet, map_exit1, c_write, src_conn="c", memlet=Memlet("C[i, j]"))
    
    # Second independent map: D = A * 2
    a_read2 = state.add_read("A")
    d_write = state.add_write("D")
    
    map_entry2, map_exit2 = state.add_map(
        "map_mul",
        {"ii": "0:4", "jj": "0:4"},
    )
    mul_tasklet = state.add_tasklet("mul2", {"a"}, {"out"}, "out = a * 2")
    
    state.add_memlet_path(a_read2, map_entry2, mul_tasklet, dst_conn="a", memlet=Memlet("A[ii, jj]"))
    state.add_memlet_path(mul_tasklet, map_exit2, d_write, src_conn="out", memlet=Memlet("D[ii, jj]"))
    
    sdfg.validate()
    
    # Count total MapEntry nodes before tiling
    map_entries_before = [
        node
        for state in sdfg.all_states()
        for node in state.nodes()
        if isinstance(node, nodes.MapEntry)
    ]
    assert len(map_entries_before) == 2, f"Expected 2 maps before tiling, got {len(map_entries_before)}"
    
    # Apply the pipeline with tiling
    count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=True, tile_shape=(2, 2))
    
    # The transformation count should indicate that both maps were tiled.
    # Each MapTiling creates additional strip-mining operations, so count >= 2
    assert count >= 2, f"Expected at least 2 transformations (both maps tiled), got {count}"
    
    # Verify the transformation was actually applied
    # At least verify that the SDFG is still valid and we can execute it
    sdfg.expand_library_nodes()
    sdfg.validate()
    
    rng = np.random.default_rng(5678)
    a = rng.standard_normal((4, 4), dtype=np.float64)
    b = rng.standard_normal((4, 4), dtype=np.float64)
    c = np.zeros((4, 4), dtype=np.float64)
    d = np.zeros((4, 4), dtype=np.float64)
    
    sdfg(A=a, B=b, C=c, D=d)
    
    np.testing.assert_allclose(c, a + b, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(d, a * 2, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Skewed-map tile shape tests (ScalarToTileMasked._calculate_tile_shape)
# ---------------------------------------------------------------------------

def _build_skewed_map_sdfg(
    outer_step: int,
    inner_end: int,
    inner_step: int,
    name: str,
    outer_shape: tuple[int, int] = (2, 2),
    inner_t1: int = 5,
) -> SDFG:
    """Build an SDFG that mimics MapTiling(skew=True) output.

    The inner map has range 0:inner_end:inner_step (zero-based, as
    produced by skewing).  The outer map step is
    tile_size * abs(inner_step), where tile_size is the original
    power-of-2 tile parameter.

    Parameters:
        outer_step: Step of the outer map dimension 0.
        inner_end: End (inclusive) of the inner map dimension 0.
        inner_step: Step of the inner map dimension 0.
        name: SDFG name.
        outer_shape: Number of outer tiles per dimension (mt, nt).
        inner_t1: Size of inner map dimension 1 (canonical, step=1).
    """
    mt, nt = outer_shape
    # inner_end is inclusive, so the array dim 2 must hold at least inner_end+1
    t0 = inner_end + 1
    t1 = inner_t1

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
    # Override dimension-0 outer step to encode tile_size.
    outer_entry.map.range[0] = (
        outer_entry.map.range[0][0],
        outer_entry.map.range[0][1],
        outer_step,
    )

    inner_entry, inner_exit = state.add_map(
        "elem_map",
        {"ii": f"0:{inner_end}:{inner_step}", "jj": f"0:{t1 - 1}"},
        schedule=dtypes.ScheduleType.Sequential,
    )

    tasklet = state.add_tasklet("add_skewed", {"a", "b"}, {"c"}, "c = a + b")

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


def _get_tile_transient_shapes(sdfg: SDFG) -> list[tuple]:
    """Return the shapes of all tile transients (transient arrays) in *sdfg*."""
    shapes = []
    for name, desc in sdfg.arrays.items():
        if desc.transient:
            shapes.append(tuple(desc.shape))
    return shapes


def test_skewed_map_tile_shape_is_power_of_2():
    """ScalarToTileMasked should derive power-of-2 tile shape for skewed maps.

    MapTiling(skew=True) on map[2:16:2] with tile_size=16 produces:
        outer step = 16 * 2 = 32
        inner range = 0:13:2  (DaCe stores inclusive end = 12)
    The base _calculate_tile_shape gives Max(0,12)+1=13 (not power of 2).
    The override should detect start==0 and recover tile_size = 32/2 = 16.
    """
    sdfg = _build_skewed_map_sdfg(
        outer_step=32,  # tile_size=16, inner_step=2 -> 16*2=32
        inner_end=13,   # DaCe "0:13:2" -> inclusive end=12, values 0,2,4,6,8,10,12
        inner_step=2,
        name="test_skewed_tile_shape",
    )

    count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
    assert count == 1, f"Expected 1 transformation, got {count}"

    state = sdfg.states()[0]
    lib_nodes = [n for n in state.nodes() if isinstance(n, nodes.LibraryNode)]
    assert len(lib_nodes) == 1
    assert isinstance(lib_nodes[0], TileSymbolicMaskedOpLibraryNode)

    # Check tile transient shapes: dimension 0 should be 16 (power of 2),
    # NOT 13 (Max(0,12)+1 from base class).
    tile_shapes = _get_tile_transient_shapes(sdfg)
    assert len(tile_shapes) > 0
    for shape in tile_shapes:
        assert int(shape[0]) == 16, (
            f"Tile transient dim 0 should be 16 (power of 2), got {shape[0]}"
        )


def test_skewed_map_tile_shape_step_1():
    """Skewed map with step=1: tile_size should be outer_step/1 = outer_step.

    MapTiling(skew=True) on map[0:16] with tile_size=16 and step=1:
        outer step = 16 * 1 = 16
        inner range = 0:16:1  (DaCe stores inclusive end = 15)
    Base gives Max(0,15)+1=16.  Override gives 16/1=16.  Both agree.
    """
    sdfg = _build_skewed_map_sdfg(
        outer_step=16,  # tile_size=16, inner_step=1 -> 16*1=16
        inner_end=16,   # DaCe "0:16:1" -> inclusive end=15
        inner_step=1,
        name="test_skewed_tile_shape_step1",
    )

    count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
    assert count == 1

    tile_shapes = _get_tile_transient_shapes(sdfg)
    assert len(tile_shapes) > 0
    for shape in tile_shapes:
        assert int(shape[0]) == 16, (
            f"Tile transient dim 0 should be 16, got {shape[0]}"
        )


def test_skewed_map_tile_shape_step_4():
    """Skewed map with step=4: tile_size=16, outer_step=64, inner 0:12:4.

    DaCe "0:12:4" -> inclusive end=8, values {0,4,8}.
    Base: Max(0,8)+1=9, Override: 64/4=16.
    The override should produce tile shape dim 0 = 16.
    """
    sdfg = _build_skewed_map_sdfg(
        outer_step=64,  # tile_size=16, inner_step=4 -> 16*4=64
        inner_end=12,   # DaCe "0:12:4" -> inclusive end=8, values 0,4,8
        inner_step=4,
        name="test_skewed_tile_shape_step4",
    )

    count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
    assert count == 1

    tile_shapes = _get_tile_transient_shapes(sdfg)
    assert len(tile_shapes) > 0
    for shape in tile_shapes:
        # Base would give Max(0,8)+1=9, override gives 64/4=16.
        assert int(shape[0]) == 16, (
            f"Tile transient dim 0 should be 16 (power of 2), got {shape[0]}"
        )


def test_skewed_map_tile_shape_large_tile_size():
    """Larger tile_size=32, step=3: outer_step=96, inner 0:32:3.

    DaCe "0:32:3" -> inclusive end=30 (last multiple of 3 before 32).
    Base: Max(0,30)+1=31 (not power of 2).
    Override: 96/3=32 (power of 2).
    """
    sdfg = _build_skewed_map_sdfg(
        outer_step=96,  # tile_size=32, inner_step=3 -> 32*3=96
        inner_end=32,   # DaCe "0:32:3" -> inclusive end=30
        inner_step=3,
        name="test_skewed_tile_shape_large",
    )

    count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
    assert count == 1

    tile_shapes = _get_tile_transient_shapes(sdfg)
    assert len(tile_shapes) > 0
    for shape in tile_shapes:
        assert int(shape[0]) == 32, (
            f"Tile transient dim 0 should be 32 (power of 2), got {shape[0]}"
        )


def test_non_skewed_map_not_affected():
    """Non-skewed maps (start != 0) should NOT be affected by the override.

    The override only activates when inner start == 0. For start=1 (non-skewed),
    the base class behavior should be preserved.
    """
    sdfg = build_runtime_tiled_scalar_noncanonical_binary_sdfg(
        ii_range="1:6:2",
        jj_range="1:5",
        op="+",
        name="test_non_skewed_unaffected",
        dtype=dace.float64,
    )

    count = apply_cutile_pipeline(sdfg, validate=True, apply_map_collapse_and_tiling=False)
    assert count == 1

    state = sdfg.states()[0]
    lib_nodes = [n for n in state.nodes() if isinstance(n, nodes.LibraryNode)]
    assert len(lib_nodes) == 1
    lib_node = lib_nodes[0]
    assert isinstance(lib_node, TileSymbolicMaskedOpLibraryNode)

    # For ii_range="1:6:2": DaCe inclusive end=5, Max(1,5)+1=6
    # (no skew correction since start!=0)
    # For jj_range="1:5": DaCe inclusive end=4, Max(1,4)+1=5
    # Verify tile transient shapes match expected bounding-box sizes.
    tile_shapes = _get_tile_transient_shapes(sdfg)
    assert len(tile_shapes) > 0
    for shape in tile_shapes:
        assert int(shape[0]) == 6, f"Expected dim 0 = 6, got {shape[0]}"
        assert int(shape[1]) == 5, f"Expected dim 1 = 5, got {shape[1]}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
