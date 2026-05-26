"""Tests that cuTile transformations correctly handle StorageType.CuTile_Tile
for tile transient arrays.

Covers:
  - create_tile_transient() in utils.py preserves source storage
  - SetCuTilePythonScope.apply() in set_cutile_python_scope.py sets CuTile_Tile
    on transient AccessNodes within the scope (Python backend only)
  - ScalarToTileMasked preload transients inherit source storage
"""

import dace
from dace import dtypes
from dace.sdfg import SDFG, nodes
from dace.sdfg.state import SDFGState

from dace.libraries.cutile.nodes import TileOpLibraryNode
from dace.libraries.cutile.transformations.utils import create_tile_transient
from dace.libraries.cutile.transformations.set_cutile_python_scope import SetCuTilePythonScope


# =========================================================================
# Tests for create_tile_transient (utils.py)
# =========================================================================


def _make_sdfg_with_array(
    name: str = "test_sdfg",
    storage: dtypes.StorageType = dtypes.StorageType.GPU_Global,
) -> tuple[SDFG, SDFGState]:
    """Helper: build SDFG with an array named 'A' using the given storage."""
    sdfg = SDFG(name)
    sdfg.add_array("A", shape=[64, 64], dtype=dace.float32, storage=storage)
    state = sdfg.add_state("s")
    return sdfg, state


def test_create_tile_transient_preserves_storage() -> None:
    """create_tile_transient must inherit the original array's storage type."""
    sdfg, state = _make_sdfg_with_array(storage=dtypes.StorageType.GPU_Global)
    tile_name, _ = create_tile_transient(sdfg, state, "A", (16, 16))

    desc = sdfg.arrays[tile_name]
    assert desc.storage == dtypes.StorageType.GPU_Global, (
        f"Expected GPU_Global but got {desc.storage}"
    )


def test_create_tile_transient_preserves_cpu_storage() -> None:
    """create_tile_transient on a CPU_Heap array should produce CPU_Heap transient."""
    sdfg = SDFG("cpu_test")
    sdfg.add_array("X", shape=[100], dtype=dace.float64,
                   storage=dtypes.StorageType.CPU_Heap)
    state = sdfg.add_state("s")

    tile_name, _ = create_tile_transient(sdfg, state, "X", (32,))
    desc = sdfg.arrays[tile_name]
    assert desc.storage == dtypes.StorageType.CPU_Heap


def test_create_tile_transient_preserves_dtype() -> None:
    """create_tile_transient must keep the original array's dtype."""
    sdfg, state = _make_sdfg_with_array()
    tile_name, _ = create_tile_transient(sdfg, state, "A", (8, 8))

    desc = sdfg.arrays[tile_name]
    assert desc.dtype == dace.float32


def test_create_tile_transient_shape() -> None:
    """create_tile_transient must set the correct tile shape."""
    sdfg, state = _make_sdfg_with_array()
    tile_shape = (32, 16)
    tile_name, _ = create_tile_transient(sdfg, state, "A", tile_shape)

    desc = sdfg.arrays[tile_name]
    assert tuple(desc.shape) == tile_shape


def test_create_tile_transient_is_transient_and_scope_lifetime() -> None:
    """The created array must be transient with Scope lifetime."""
    sdfg, state = _make_sdfg_with_array()
    tile_name, _ = create_tile_transient(sdfg, state, "A", (8,))

    desc = sdfg.arrays[tile_name]
    assert desc.transient
    assert desc.lifetime == dtypes.AllocationLifetime.Scope


def test_create_tile_transient_node_in_state() -> None:
    """The returned AccessNode must be in the state."""
    sdfg, state = _make_sdfg_with_array()
    _, tile_node = create_tile_transient(sdfg, state, "A", (8,))

    assert tile_node in state.nodes()
    assert isinstance(tile_node, nodes.AccessNode)


# =========================================================================
# Tests for SetCuTilePythonScope.apply() (set_cutile_python_scope.py)
# =========================================================================


def _build_map_with_transient_and_lib_node() -> SDFG:
    """Build an SDFG with a map containing transients and a cuTile lib node.

    Layout:
        A(read) -> MapEntry -> T_a(transient) -> LibNode -> MapExit -> C(write)
        B(read) -> MapEntry -> T_b(transient) ->

    The transients start with GPU_Global storage; SetCuTilePythonScope should
    change them to CuTile_Tile.
    """
    sdfg = SDFG("scope_storage_test")
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_symbol("MT", dace.int32)
    sdfg.add_symbol("TS", dace.int32)

    sdfg.add_array("A", shape=[dace.symbol("MT"), dace.symbol("TS")], dtype=dace.float32)
    sdfg.add_array("B", shape=[dace.symbol("MT"), dace.symbol("TS")], dtype=dace.float32)
    sdfg.add_array("C", shape=[dace.symbol("MT"), dace.symbol("TS")], dtype=dace.float32)
    sdfg.add_transient("T_a", shape=[16], dtype=dace.float32)
    sdfg.add_transient("T_b", shape=[16], dtype=dace.float32)

    state = sdfg.add_state("main")
    me, mx = state.add_map("tile", {"t": "0:MT"}, schedule=dtypes.ScheduleType.Sequential)

    a = state.add_read("A")
    b = state.add_read("B")
    c = state.add_write("C")
    t_a = state.add_access("T_a")
    t_b = state.add_access("T_b")

    lib = TileOpLibraryNode("Add", op="+", tile_shape=[16])
    state.add_node(lib)

    out_conn = next(iter(lib.out_connectors.keys()))

    # A -> MapEntry -> T_a -> LibNode._a
    state.add_memlet_path(a, me, t_a, memlet=dace.Memlet("A[t, 0:TS]"))
    state.add_edge(t_a, None, lib, "_a", dace.Memlet("T_a[0:16]"))

    # B -> MapEntry -> T_b -> LibNode._b
    state.add_memlet_path(b, me, t_b, memlet=dace.Memlet("B[t, 0:TS]"))
    state.add_edge(t_b, None, lib, "_b", dace.Memlet("T_b[0:16]"))

    # LibNode -> MapExit -> C
    state.add_memlet_path(lib, mx, c, src_conn=out_conn, memlet=dace.Memlet("C[t, 0:TS]"))

    sdfg.validate()
    return sdfg


def test_set_cutile_python_scope_sets_transient_storage() -> None:
    """SetCuTilePythonScope.apply() must set transient AccessNodes in scope
    to CuTile_Tile storage."""
    sdfg = _build_map_with_transient_and_lib_node()

    # Before: transients have GPU_Global storage
    for tname in ("T_a", "T_b"):
        assert sdfg.arrays[tname].storage == dtypes.StorageType.Default

    applied = sdfg.apply_transformations_repeated([SetCuTilePythonScope])
    assert applied == 1

    # After: transients should have CuTile_Tile storage
    for tname in ("T_a", "T_b"):
        desc = sdfg.arrays[tname]
        assert desc.storage == dtypes.StorageType.CuTile_Tile, (
            f"Expected CuTile_Tile for '{tname}' but got {desc.storage}"
        )


def test_set_cutile_python_scope_does_not_change_non_transient() -> None:
    """Non-transient arrays referenced by AccessNodes in scope must NOT have
    their storage changed."""
    sdfg = _build_map_with_transient_and_lib_node()

    # Before: A, B, C are non-transient
    original_storages = {
        name: sdfg.arrays[name].storage
        for name in ("A", "B", "C")
    }

    sdfg.apply_transformations_repeated([SetCuTilePythonScope])

    # Non-transients should be unchanged
    for name, orig_storage in original_storages.items():
        assert sdfg.arrays[name].storage == orig_storage, (
            f"Non-transient '{name}' storage changed from {orig_storage} "
            f"to {sdfg.arrays[name].storage}"
        )


def test_set_cutile_python_scope_also_sets_schedule_and_impl() -> None:
    """Verify that the original functionality (schedule + impl) still works
    alongside the new storage update."""
    sdfg = _build_map_with_transient_and_lib_node()
    sdfg.apply_transformations_repeated([SetCuTilePythonScope])

    state = sdfg.states()[0]
    map_entry = next(n for n in state.nodes() if isinstance(n, nodes.MapEntry))
    lib_nodes = [n for n in state.nodes() if isinstance(n, nodes.LibraryNode)]

    assert map_entry.map.schedule == dtypes.ScheduleType.CuTile
    assert all(n.implementation == "cutile_python" for n in lib_nodes)


def test_set_cutile_python_scope_idempotent_storage() -> None:
    """Applying SetCuTilePythonScope twice should not change the result."""
    sdfg = _build_map_with_transient_and_lib_node()

    first = sdfg.apply_transformations_repeated([SetCuTilePythonScope])
    assert first == 1

    for tname in ("T_a", "T_b"):
        assert sdfg.arrays[tname].storage == dtypes.StorageType.CuTile_Tile

    # Second application should be a no-op (already marked)
    second = sdfg.apply_transformations_repeated([SetCuTilePythonScope])
    assert second == 0

    # Storage should still be CuTile_Tile
    for tname in ("T_a", "T_b"):
        assert sdfg.arrays[tname].storage == dtypes.StorageType.CuTile_Tile


def test_set_cutile_python_scope_not_applied_without_python_backend() -> None:
    """SetCuTilePythonScope must not apply when backend is not Python."""
    sdfg = _build_map_with_transient_and_lib_node()
    sdfg.backend = dtypes.BackendLanguage.CPP  # Not Python backend

    applied = sdfg.apply_transformations_repeated([SetCuTilePythonScope])
    assert applied == 0

    # Storage should remain GPU_Global
    for tname in ("T_a", "T_b"):
        assert sdfg.arrays[tname].storage == dtypes.StorageType.Default


# =========================================================================
# Tests for ScalarToTileMasked preload transients (scalar_to_tile_library.py)
# =========================================================================


def test_scalar_to_tile_masked_preload_preserves_original_storage() -> None:
    """Verify preload transients created by ScalarToTileMasked inherit
    the original array's storage type (not CuTile_Tile), since the
    transformation runs for both C++ and Python backends.

    The SetCuTilePythonScope transformation is responsible for converting
    to CuTile_Tile specifically for the Python backend.
    """
    from dace.libraries.cutile.transformations.scalar_to_tile_library import ScalarToTileMasked

    sdfg = SDFG("masked_preload_test")
    sdfg.add_array("A", shape=[64], dtype=dace.float32,
                   storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("C", shape=[64], dtype=dace.float32,
                   storage=dtypes.StorageType.GPU_Global)

    state = sdfg.add_state("main")

    # Outer map: tiles of 8
    ome, omx = state.add_map("outer", {"ti": "0:64:8"},
                             schedule=dtypes.ScheduleType.GPU_Device)
    # Inner map: non-canonical (stride 2, so masked)
    ime, imx = state.add_map("inner", {"i": "0:7:2"},
                             schedule=dtypes.ScheduleType.Sequential)

    a_read = state.add_read("A")
    c_write = state.add_write("C")

    tasklet = state.add_tasklet("negate", {"_in"}, {"_out"}, "_out = -_in")

    state.add_memlet_path(a_read, ome, ime, tasklet,
                          dst_conn="_in",
                          memlet=dace.Memlet("A[ti + i]"))
    state.add_memlet_path(tasklet, imx, omx, c_write,
                          src_conn="_out",
                          memlet=dace.Memlet("C[ti + i]"))

    sdfg.validate()

    applied = sdfg.apply_transformations([ScalarToTileMasked])
    assert applied == 1, "ScalarToTileMasked should have been applied"

    # After transformation, all tile transients should inherit GPU_Global
    # (the original array storage). CuTile_Tile is set later by
    # SetCuTilePythonScope for the Python backend.
    for name, desc in sdfg.arrays.items():
        if desc.transient:
            assert desc.storage == dtypes.StorageType.GPU_Global, (
                f"Transient '{name}' has storage {desc.storage}, "
                f"expected GPU_Global (inherited from original arrays)"
            )
