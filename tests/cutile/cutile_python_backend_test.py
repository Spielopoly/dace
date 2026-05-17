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
    assert "import cupy as cp" in frame_code


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


def test_symbolic_mask_cutile_python_expansion_rejected():
    lib = TileSymbolicMaskedOpLibraryNode("SymMasked", op="+", tile_shape=[16])
    lib.implementation = "cutile_python"

    _expanded_single_tasklet_code(
        lib,
        {
            "A": ("_a", dace.float32),
            "B": ("_b", dace.float32),
            "Cin": ("_c_in", dace.float32),
        },
    )
