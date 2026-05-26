"""Tests for schedule-based cuTile Python backend code generation."""

import pytest
import dace
from dace import dtypes
from dace.codegen import codegen as dace_codegen
from dace.sdfg import SDFG, nodes

from dace.libraries.cutile.nodes.op_runtime_map import TileRuntimeMaskedOpLibraryNode
from dace.libraries.cutile.nodes.op_symbolic_mask import TileSymbolicMaskedOpLibraryNode
from dace.libraries.cutile.nodes.where_select import TileWhereSelectLibraryNode


def _code_of(sdfg: SDFG) -> str:
    code_objects = dace_codegen.generate_code(sdfg)
    return next(co.clean_code for co in code_objects if co.name == sdfg.name)


def _build_schedule_tasklet_sdfg(name: str, schedule: dtypes.ScheduleType, tasklet_code: str,
                                 input_arrays: dict[str, tuple[str, dace.typeclass]],
                                 output_dtype=dace.float32) -> SDFG:
    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_symbol("N", dace.int32)

    for arr_name, (_, dtype) in input_arrays.items():
        sdfg.add_array(arr_name, shape=[dace.symbol("N")], dtype=dtype)
    sdfg.add_array("C", shape=[dace.symbol("N")], dtype=output_dtype)

    state = sdfg.add_state("main")
    map_entry, map_exit = state.add_map("m", {"i": "0:N"}, schedule=schedule)

    tasklet_inputs = set(conn for conn, _ in input_arrays.values())
    tasklet = state.add_tasklet("t", tasklet_inputs, {"_out"}, tasklet_code)
    c_write = state.add_write("C")

    for arr_name, (conn, _) in input_arrays.items():
        r = state.add_read(arr_name)
        state.add_memlet_path(r, map_entry, tasklet, dst_conn=conn, memlet=dace.Memlet(f"{arr_name}[i]"))

    state.add_memlet_path(tasklet, map_exit, c_write, src_conn="_out", memlet=dace.Memlet("C[i]"))

    sdfg.validate()
    return sdfg


def _expanded_single_tasklet_code(lib_node, arrays: dict[str, tuple[str, dace.typeclass]], out_name: str = "C") -> str:
    sdfg = SDFG("expand_only")
    sdfg.backend = dtypes.BackendLanguage.Python

    for arr_name, (_, dtype) in arrays.items():
        sdfg.add_array(arr_name, shape=[16], dtype=dtype)
    sdfg.add_array(out_name, shape=[16], dtype=dace.float32)

    state = sdfg.add_state("main")
    state.add_node(lib_node)

    for arr_name, (conn, _) in arrays.items():
        r = state.add_read(arr_name)
        state.add_edge(r, None, lib_node, conn, dace.Memlet.from_array(arr_name, sdfg.arrays[arr_name]))

    w = state.add_write(out_name)
    out_conn = next(iter(lib_node.out_connectors.keys()))
    state.add_edge(lib_node, out_conn, w, None, dace.Memlet.from_array(out_name, sdfg.arrays[out_name]))

    sdfg.expand_library_nodes()

    tasklets = [
        n
        for st in sdfg.states()
        for n in st.nodes()
        if isinstance(n, nodes.Tasklet)
    ]
    assert len(tasklets) == 1
    return tasklets[0].code.as_string


def test_schedule_cutile_generates_kernel_and_launch():
    sdfg = _build_schedule_tasklet_sdfg(
        "schedule_cutile_add",
        dtypes.ScheduleType.CuTile,
        "_out = _a + _b",
        {"A": ("_a", dace.float32), "B": ("_b", dace.float32)},
    )

    frame_code = _code_of(sdfg)
    assert "@ct.kernel" in frame_code
    assert "ct.launch" in frame_code
    assert "ct.load" in frame_code
    assert "ct.store" in frame_code
    assert "import cuda.tile as ct" in frame_code
    assert "import cupy" in frame_code


def test_non_cutile_schedule_not_dispatched_to_cutile_target():
    sdfg = _build_schedule_tasklet_sdfg(
        "schedule_seq_add",
        dtypes.ScheduleType.Sequential,
        "_out = _a + _b",
        {"A": ("_a", dace.float32), "B": ("_b", dace.float32)},
    )

    frame_code = _code_of(sdfg)
    assert "@ct.kernel" not in frame_code
    assert "ct.launch" not in frame_code


def test_marker_like_tasklet_code_does_not_trigger_cutile_target_without_schedule():
    sdfg = _build_schedule_tasklet_sdfg(
        "schedule_seq_marker_like",
        dtypes.ScheduleType.Sequential,
        "_out = _a + _b\n__CUTILE_SPEC__ = 'legacy_marker'",
        {"A": ("_a", dace.float32), "B": ("_b", dace.float32)},
    )

    frame_code = _code_of(sdfg)
    assert "@ct.kernel" not in frame_code
    assert "ct.launch" not in frame_code


def test_runtime_mask_expansion_emits_where_expression():
    lib = TileRuntimeMaskedOpLibraryNode("MaskedAdd", op="+", tile_shape=[16])
    lib.implementation = "cutile_python"
    code = _expanded_single_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float32),
            "B": ("_b", dace.float32),
            "M": ("_m", dace.bool_),
            "Cin": ("_c_in", dace.float32),
        },
    )
    assert "ct.where" in code
    assert "_m" in code


def test_runtime_mask_cutile_python_requires_c_in():
    lib = TileRuntimeMaskedOpLibraryNode("MaskedAddNoCin", op="+", tile_shape=[16])
    lib.implementation = "cutile_python"

    with pytest.raises(ValueError, match="requires '_c_in'"):
        _expanded_single_tasklet_code(
            lib,
            {
                "A": ("_a", dace.float32),
                "B": ("_b", dace.float32),
                "M": ("_m", dace.bool_),
            },
        )


def test_where_select_expansion_emits_where_expression():
    lib = TileWhereSelectLibraryNode("Where", tile_shape=[16])
    lib.implementation = "cutile_python"
    code = _expanded_single_tasklet_code(
        lib,
        {
            "Cond": ("_cond", dace.bool_),
            "X": ("_x", dace.float32),
            "Y": ("_y", dace.float32),
        },
    )
    assert "ct.where" in code


def test_cutile_codegen_rejects_symbolic_mask_tasklet():
    sdfg = _build_schedule_tasklet_sdfg(
        "schedule_cutile_symbolic",
        dtypes.ScheduleType.CuTile,
        "_out = ('__m0_guard', _a + _b)[1]",
        {
            "A": ("_a", dace.float32),
            "B": ("_b", dace.float32),
            "Cin": ("_c_in", dace.float32),
        },
    )

    with pytest.raises(ValueError, match="symbolic mask"):
        _code_of(sdfg)


# TODO: The following four tests might become outdated rather quickly

def test_symbolic_mask_no_condition_expands_without_where():
    """With mask_condition=None, expansion should succeed without ct.where."""
    lib = TileSymbolicMaskedOpLibraryNode("SymMasked", op="+", tile_shape=[16])
    lib.implementation = "cutile_python"

    code = _expanded_single_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float32),
            "B": ("_b", dace.float32),
            "Cin": ("_c_in", dace.float32),
        },
    )
    # With mask_condition=None, ct.where should not be emitted
    assert "ct.where" not in code
    assert "_out" in code or "_a" in code


def test_strided_map_shape_resolves_min_max():
    """Strided maps produce Min/Max in subset.size(); codegen must resolve them to integers.

    A map with range ``0:N:32`` and inner memlet ``A[tile_i:tile_i+32]``
    yields ``Max(0, -tile_i + Min(N-1, tile_i+31)) - Min(0, ...) + 1`` for
    the tile size.  The cuTile runtime requires constant integer shape tuples
    in ``ct.load(..., shape=(...,))``.  This test verifies that substituting
    the map param with its start value resolves the expression to ``32``.
    """
    import re as _re

    sdfg = SDFG("strided_shape_resolve")
    sdfg.backend = dtypes.BackendLanguage.Python
    N = dace.symbol("N")
    sdfg.add_symbol("N", dace.int32)
    sdfg.add_array("A", shape=[N], dtype=dace.float32)
    sdfg.add_array("C", shape=[N], dtype=dace.float32)

    state = sdfg.add_state("main")
    # Outer tiled map: stride-32 steps (as produced by MapTiling)
    map_entry, map_exit = state.add_map(
        "tiled", {"tile_i": "0:N:32"}, schedule=dtypes.ScheduleType.CuTile)
    tasklet = state.add_tasklet("copy", {"_a"}, {"_out"}, "_out = _a")
    a_read = state.add_read("A")
    c_write = state.add_write("C")

    # Inner subset: tile_i : min(tile_i+32, N) (strided tile access)
    state.add_memlet_path(
        a_read, map_entry, tasklet, dst_conn="_a",
        memlet=dace.Memlet(f"A[tile_i:Min(tile_i + 32, N)]"))
    state.add_memlet_path(
        tasklet, map_exit, c_write, src_conn="_out",
        memlet=dace.Memlet(f"C[tile_i:Min(tile_i + 32, N)]"))
    sdfg.validate()

    frame_code = _code_of(sdfg)
    # Extract the shape argument from the ct.load() call
    load_match = _re.search(r"ct\.load\(A,.*?shape=\(([^)]*)\)", frame_code)
    assert load_match is not None, f"No ct.load(A, ...) found in:\n{frame_code}"
    shape_str = load_match.group(1).strip().rstrip(",").strip()
    # The shape must be a plain integer (32), not a symbolic Min/Max expression
    assert shape_str == "32", (
        f"Expected resolved shape '32', got '{shape_str}'.\n"
        f"Generated code:\n{frame_code}"
    )


def test_strided_2d_map_shape_resolves():
    """2-D strided map should resolve both dimensions to constant integers."""
    import re as _re

    sdfg = SDFG("strided_2d_shape")
    sdfg.backend = dtypes.BackendLanguage.Python
    N = dace.symbol("N")
    M = dace.symbol("M")
    sdfg.add_symbol("N", dace.int32)
    sdfg.add_symbol("M", dace.int32)
    sdfg.add_array("A", shape=[N, M], dtype=dace.float32)
    sdfg.add_array("C", shape=[N, M], dtype=dace.float32)

    state = sdfg.add_state("main")
    map_entry, map_exit = state.add_map(
        "tiled2d",
        {"tile_i": "0:N:16", "tile_j": "0:M:8"},
        schedule=dtypes.ScheduleType.CuTile,
    )
    tasklet = state.add_tasklet("copy", {"_a"}, {"_out"}, "_out = _a")
    a_read = state.add_read("A")
    c_write = state.add_write("C")

    state.add_memlet_path(
        a_read, map_entry, tasklet, dst_conn="_a",
        memlet=dace.Memlet(f"A[tile_i:Min(tile_i + 16, N), tile_j:Min(tile_j + 8, M)]"))
    state.add_memlet_path(
        tasklet, map_exit, c_write, src_conn="_out",
        memlet=dace.Memlet(f"C[tile_i:Min(tile_i + 16, N), tile_j:Min(tile_j + 8, M)]"))
    sdfg.validate()

    frame_code = _code_of(sdfg)
    load_match = _re.search(r"ct\.load\(A,.*?shape=\(([^)]*)\)", frame_code)
    assert load_match is not None, f"No ct.load(A, ...) found in:\n{frame_code}"
    shape_str = load_match.group(1).strip().rstrip(",").strip()
    # Both dimensions must resolve to plain integers
    parts = [p.strip() for p in shape_str.split(",")]
    assert parts == ["16", "8"], (
        f"Expected resolved shape ['16', '8'], got {parts}.\n"
        f"Generated code:\n{frame_code}"
    )


def test_non_strided_map_shape_unchanged():
    """Non-strided (stride-1) maps should continue to produce simple shapes."""
    import re as _re

    sdfg = SDFG("non_strided_shape")
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_symbol("N", dace.int32)
    N = dace.symbol("N")
    sdfg.add_array("A", shape=[N], dtype=dace.float32)
    sdfg.add_array("C", shape=[N], dtype=dace.float32)

    state = sdfg.add_state("main")
    map_entry, map_exit = state.add_map(
        "m", {"i": "0:N"}, schedule=dtypes.ScheduleType.CuTile)
    tasklet = state.add_tasklet("copy", {"_a"}, {"_out"}, "_out = _a")
    a_read = state.add_read("A")
    c_write = state.add_write("C")

    state.add_memlet_path(
        a_read, map_entry, tasklet, dst_conn="_a",
        memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(
        tasklet, map_exit, c_write, src_conn="_out",
        memlet=dace.Memlet("C[i]"))
    sdfg.validate()

    frame_code = _code_of(sdfg)
    # For a stride-1 map with scalar access, the shape should be 1
    load_match = _re.search(r"ct\.load\(A,.*?shape=\(([^)]*)\)", frame_code)
    assert load_match is not None, f"No ct.load(A, ...) found in:\n{frame_code}"
    shape_str = load_match.group(1).strip().rstrip(",").strip()
    assert shape_str == "1", (
        f"Expected shape '1' for scalar access, got '{shape_str}'.\n"
        f"Generated code:\n{frame_code}"
    )


def test_tile_transient_padded_shape_used_for_load():
    """When a tile transient has a power-of-2 padded shape, ct.load must use
    the transient descriptor shape, not the memlet subset size.

    This simulates the ScalarToTileMasked pattern where a strided inner map
    has a non-power-of-2 tile size (e.g. 14) but the transient is padded to
    the next power of 2 (16).  The memlet still covers only the actual data
    range (0:13 = 14 elements), but ct.load must emit shape=(16,) because
    the cuTile runtime requires power-of-2 tile dimensions.
    """
    import re as _re

    sdfg = SDFG("padded_tile_transient")
    sdfg.backend = dtypes.BackendLanguage.Python
    N = dace.symbol("N")
    sdfg.add_symbol("N", dace.int32)
    sdfg.add_array("A", shape=[N], dtype=dace.float32)
    sdfg.add_array("C", shape=[N], dtype=dace.float32)
    # Tile transient with power-of-2 padded shape (16, not 14)
    sdfg.add_transient("tile_A", shape=[16], dtype=dace.float32)

    state = sdfg.add_state("main")
    map_entry, map_exit = state.add_map(
        "tiled", {"tile_i": "0:N:14"}, schedule=dtypes.ScheduleType.CuTile)
    tasklet = state.add_tasklet("copy", {"_a"}, {"_out"}, "_out = _a")
    a_read = state.add_read("A")
    c_write = state.add_write("C")
    tile_a = state.add_access("tile_A")

    # Outer edge: full array -> map entry -> tile transient
    # add_memlet_path through the map entry automatically creates IN_/OUT_
    # connectors; then the inner leg lands on the tile transient.
    state.add_memlet_path(
        a_read, map_entry, tile_a,
        memlet=dace.Memlet(f"A[tile_i:Min(tile_i + 14, N)]"))
    # Tile transient -> tasklet
    state.add_edge(tile_a, None, tasklet, "_a",
                   dace.Memlet("tile_A[0:14]"))
    # Tasklet -> map exit -> output
    state.add_memlet_path(
        tasklet, map_exit, c_write, src_conn="_out",
        memlet=dace.Memlet(f"C[tile_i:Min(tile_i + 14, N)]"))
    sdfg.validate()

    frame_code = _code_of(sdfg)
    load_match = _re.search(r"ct\.load\(A,.*?shape=\(([^)]*)\)", frame_code)
    assert load_match is not None, f"No ct.load(A, ...) found in:\n{frame_code}"
    shape_str = load_match.group(1).strip().rstrip(",").strip()
    # Must be 16 (padded descriptor shape), not 14 (memlet subset size)
    assert shape_str == "16", (
        f"Expected padded shape '16', got '{shape_str}'.\n"
        f"Generated code:\n{frame_code}"
    )


def test_tile_transient_2d_padded_shape():
    """2-D tile transient with padded shape: both dimensions should come from
    the transient descriptor, not from the memlet subset."""
    import re as _re

    sdfg = SDFG("padded_2d_tile_transient")
    sdfg.backend = dtypes.BackendLanguage.Python
    N = dace.symbol("N")
    M = dace.symbol("M")
    sdfg.add_symbol("N", dace.int32)
    sdfg.add_symbol("M", dace.int32)
    sdfg.add_array("A", shape=[N, M], dtype=dace.float32)
    sdfg.add_array("C", shape=[N, M], dtype=dace.float32)
    # 2-D transient padded to powers of 2: actual tile is (14, 12)
    # but transient is padded to (16, 16)
    sdfg.add_transient("tile_A", shape=[16, 16], dtype=dace.float32)

    state = sdfg.add_state("main")
    map_entry, map_exit = state.add_map(
        "tiled2d",
        {"tile_i": "0:N:14", "tile_j": "0:M:12"},
        schedule=dtypes.ScheduleType.CuTile,
    )
    tasklet = state.add_tasklet("copy", {"_a"}, {"_out"}, "_out = _a")
    a_read = state.add_read("A")
    c_write = state.add_write("C")
    tile_a = state.add_access("tile_A")

    state.add_memlet_path(
        a_read, map_entry, tile_a,
        memlet=dace.Memlet(f"A[tile_i:Min(tile_i + 14, N), tile_j:Min(tile_j + 12, M)]"))
    state.add_edge(tile_a, None, tasklet, "_a",
                   dace.Memlet("tile_A[0:14, 0:12]"))
    state.add_memlet_path(
        tasklet, map_exit, c_write, src_conn="_out",
        memlet=dace.Memlet(f"C[tile_i:Min(tile_i + 14, N), tile_j:Min(tile_j + 12, M)]"))
    sdfg.validate()

    frame_code = _code_of(sdfg)
    
    load_match = _re.search(r"ct\.gather\(A,.*?shape=\(([^)]*)\)", frame_code)
    assert load_match is not None, f"No ct.gather(A, ...) found in:\n{frame_code}"
    shape_str = load_match.group(1).strip().rstrip(",").strip()
    parts = [p.strip() for p in shape_str.split(",")]
    assert parts == ["16", "16"], (
        f"Expected padded shape ['16', '16'], got {parts}.\n"
        f"Generated code:\n{frame_code}"
    )
