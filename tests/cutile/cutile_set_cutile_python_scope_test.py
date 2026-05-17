"""Tests for SetCuTilePythonScope transformation."""

import dace
from dace import dtypes
from dace.sdfg import SDFG, nodes

from dace.libraries.cutile.nodes import (
    TileOpLibraryNode,
    TileSymbolicMaskedOpLibraryNode,
)
from dace.libraries.cutile.transformations.set_cutile_python_scope import SetCuTilePythonScope


def _build_map_with_library_node(lib_node: nodes.LibraryNode, with_extra_tasklet: bool = False) -> SDFG:
    sdfg = SDFG("scope_test")
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_symbol("MT", dace.int32)
    sdfg.add_symbol("TS", dace.int32)

    sdfg.add_array("A", shape=[dace.symbol("MT"), dace.symbol("TS")], dtype=dace.float32)
    sdfg.add_array("B", shape=[dace.symbol("MT"), dace.symbol("TS")], dtype=dace.float32)
    sdfg.add_array("C", shape=[dace.symbol("MT"), dace.symbol("TS")], dtype=dace.float32)
    if "_m" in lib_node.in_connectors:
        sdfg.add_array("M", shape=[dace.symbol("MT"), dace.symbol("TS")], dtype=dace.bool_)

    state = sdfg.add_state("main")
    me, mx = state.add_map("tile", {"t": "0:MT"}, schedule=dtypes.ScheduleType.Sequential)

    a = state.add_read("A")
    b = state.add_read("B")
    c = state.add_write("C")
    m = state.add_read("M") if "_m" in lib_node.in_connectors else None

    state.add_node(lib_node)
    if "_a" in lib_node.in_connectors:
        state.add_memlet_path(a, me, lib_node, dst_conn="_a", memlet=dace.Memlet("A[t, 0:TS]"))
    if "_b" in lib_node.in_connectors:
        state.add_memlet_path(b, me, lib_node, dst_conn="_b", memlet=dace.Memlet("B[t, 0:TS]"))
    if "_m" in lib_node.in_connectors:
        assert m is not None
        state.add_memlet_path(m, me, lib_node, dst_conn="_m", memlet=dace.Memlet("M[t, 0:TS]"))
    if "_c_in" in lib_node.in_connectors:
        state.add_memlet_path(c, me, lib_node, dst_conn="_c_in", memlet=dace.Memlet("C[t, 0:TS]"))

    out_conn = next(iter(lib_node.out_connectors.keys()))
    state.add_memlet_path(lib_node, mx, c, src_conn=out_conn, memlet=dace.Memlet("C[t, 0:TS]"))

    if with_extra_tasklet:
        temp = state.add_tasklet("foreign", {"x"}, {"y"}, "y = x")
        state.add_memlet_path(a, me, temp, dst_conn="x", memlet=dace.Memlet("A[t, 0]"))
        state.add_memlet_path(temp, mx, c, src_conn="y", memlet=dace.Memlet("C[t, 0]"))

    sdfg.validate()
    return sdfg


def test_set_cutile_python_scope_positive_and_idempotent():
    lib = TileOpLibraryNode("Add", op="+", tile_shape=[16])
    sdfg = _build_map_with_library_node(lib)

    first = sdfg.apply_transformations_repeated([SetCuTilePythonScope])
    assert first == 1

    state = sdfg.states()[0]
    map_entry = next(n for n in state.nodes() if isinstance(n, nodes.MapEntry))
    lib_nodes = [n for n in state.nodes() if isinstance(n, nodes.LibraryNode)]

    assert map_entry.map.schedule == dtypes.ScheduleType.CuTile
    assert all(n.implementation == "cutile_python" for n in lib_nodes)

    second = sdfg.apply_transformations_repeated([SetCuTilePythonScope])
    assert second == 0


def test_set_cutile_python_scope_rejects_foreign_tasklet():
    lib = TileOpLibraryNode("Add", op="+", tile_shape=[16])
    sdfg = _build_map_with_library_node(lib, with_extra_tasklet=True)

    applied = sdfg.apply_transformations_repeated([SetCuTilePythonScope])
    assert applied == 0


def test_set_cutile_python_scope_rejects_symbolic_mask_scope():
    lib = TileSymbolicMaskedOpLibraryNode("SymMasked", op="+", tile_shape=[16])
    sdfg = _build_map_with_library_node(lib)

    applied = sdfg.apply_transformations_repeated([SetCuTilePythonScope])
    assert applied == 0
