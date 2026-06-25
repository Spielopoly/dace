# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU runtime tests for TileIota's cuTile expansion.

Each test manually constructs an SDFG with TileIota + TileStore nodes inside a
CuTile-scheduled map, stamps cuTile implementations and storage, expands
library nodes, compiles to the Python backend, runs on GPU, and compares the
result against a NumPy reference.

TileIota is NOT wired into the production ``VectorizeCuTile`` pipeline — it is
only constructed manually. Therefore these tests hand-stamp schedule, storage,
and implementations rather than using the orchestrator.

Naming convention: each SDFG must have a globally unique name (used as build
folder).
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.sdfg import SDFG
from dace.memlet import Memlet
from dace.dtypes import ScheduleType, StorageType
from dace.libraries.tileops.nodes.tile_iota import TileIota
from dace.libraries.tileops.nodes.tile_store import TileStore
from dace.libraries.tileops.nodes.tile_binop import TileBinop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_cutile(sdfg: SDFG, **kwargs):
    """Compile and run a cuTile SDFG, returning results as NumPy arrays.

    Converts NumPy inputs to CuPy for GPU execution and converts outputs
    back to NumPy.

    :param sdfg: The SDFG to compile and run.
    :param kwargs: Named arguments for the SDFG (arrays and symbols).
    :returns: Dictionary mapping array names to NumPy results.
    """
    import cupy as cp

    cp_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            cp_kwargs[k] = cp.asarray(v)
        else:
            cp_kwargs[k] = v

    csdfg = sdfg.compile()
    csdfg(**cp_kwargs)

    results = {}
    for k, v in cp_kwargs.items():
        if isinstance(v, cp.ndarray):
            results[k] = cp.asnumpy(v)
        else:
            results[k] = v
    return results


def _build_iota_store_sdfg_k1(
    name: str,
    N: int,
    W: int,
    expr: str,
    out_dtype: dace.typeclass,
) -> SDFG:
    """Build a K=1 SDFG: TileIota -> tile transient -> TileStore -> OUT.

    The map iterates ``i = 0:N:W``.  TileIota fills a tile with ``expr``
    (which may reference ``i`` and ``__l0``).  TileStore writes the tile
    to ``OUT[i:i+W]``.

    :param name: Unique SDFG name.
    :param N: Array length.
    :param W: Tile width (power of 2).
    :param expr: Per-lane body expression.
    :param out_dtype: DaCe dtype for the output array.
    :returns: The constructed SDFG.
    """
    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array("OUT", (N,), out_dtype, storage=StorageType.GPU_Global)
    sdfg.add_array(
        "_tile", (W,), out_dtype,
        storage=StorageType.CuTile_Tile, transient=True,
    )

    state = sdfg.add_state("main")
    me, mx = state.add_map(
        "iota_map", {"i": f"0:{N}:{W}"},
        schedule=ScheduleType.CuTile,
    )

    # TileIota node
    iota = TileIota("iota", widths=(W,), expr=expr)
    iota.target_isa = "CUTILE"
    iota.implementation = "cutile"
    state.add_node(iota)

    # Tile transient AccessNode
    tile_an = state.add_access("_tile")

    # TileStore node
    store = TileStore("store", widths=(W,))
    store.target_isa = "CUTILE"
    store.implementation = "cutile"
    state.add_node(store)

    # Output AccessNode (outside the map scope)
    out_an = state.add_write("OUT")

    # Wiring: MapEntry -> TileIota -> _tile -> TileStore -> MapExit -> OUT
    state.add_edge(me, None, iota, None, Memlet())
    state.add_edge(iota, "_dst", tile_an, None,
                   Memlet(data="_tile", subset=f"0:{W}"))
    state.add_edge(tile_an, None, store, "_src",
                   Memlet(data="_tile", subset=f"0:{W}"))
    state.add_memlet_path(
        store, mx, out_an,
        src_conn="_dst",
        memlet=Memlet(data="OUT", subset=f"i:i+{W}"),
    )

    sdfg.expand_library_nodes()
    return sdfg


def _build_iota_store_sdfg_k2(
    name: str,
    M: int,
    N: int,
    W0: int,
    W1: int,
    expr: str,
    out_dtype: dace.typeclass,
) -> SDFG:
    """Build a K=2 SDFG: TileIota -> tile transient -> TileStore -> OUT.

    The map iterates ``i = 0:M:W0, j = 0:N:W1``.  TileIota fills a 2-D
    tile with ``expr`` (which may reference ``i``, ``j``, ``__l0``,
    ``__l1``).  TileStore writes the tile to ``OUT[i:i+W0, j:j+W1]``.

    :param name: Unique SDFG name.
    :param M: First array dimension.
    :param N: Second array dimension.
    :param W0: Tile width for dim 0.
    :param W1: Tile width for dim 1.
    :param expr: Per-lane body expression.
    :param out_dtype: DaCe dtype for the output array.
    :returns: The constructed SDFG.
    """
    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array("OUT", (M, N), out_dtype, storage=StorageType.GPU_Global)
    sdfg.add_array(
        "_tile", (W0, W1), out_dtype,
        storage=StorageType.CuTile_Tile, transient=True,
    )

    state = sdfg.add_state("main")
    me, mx = state.add_map(
        "iota_map", {"i": f"0:{M}:{W0}", "j": f"0:{N}:{W1}"},
        schedule=ScheduleType.CuTile,
    )

    iota = TileIota("iota", widths=(W0, W1), expr=expr)
    iota.target_isa = "CUTILE"
    iota.implementation = "cutile"
    state.add_node(iota)

    tile_an = state.add_access("_tile")

    store = TileStore("store", widths=(W0, W1))
    store.target_isa = "CUTILE"
    store.implementation = "cutile"
    state.add_node(store)

    # Output AccessNode (outside the map scope)
    out_an = state.add_write("OUT")

    # Wiring
    state.add_edge(me, None, iota, None, Memlet())
    state.add_edge(iota, "_dst", tile_an, None,
                   Memlet(data="_tile", subset=f"0:{W0}, 0:{W1}"))
    state.add_edge(tile_an, None, store, "_src",
                   Memlet(data="_tile", subset=f"0:{W0}, 0:{W1}"))
    state.add_memlet_path(
        store, mx, out_an,
        src_conn="_dst",
        memlet=Memlet(data="OUT", subset=f"i:i+{W0}, j:j+{W1}"),
    )

    sdfg.expand_library_nodes()
    return sdfg


# ---------------------------------------------------------------------------
# GPU runtime tests
# ---------------------------------------------------------------------------


# All tests in this class require a GPU.
class TestTileIotaCutileRuntime:
    """GPU runtime tests for TileIota cuTile expansion."""

    pytestmark = pytest.mark.gpu

    # ---------------------------------------------------------------
    # Test 1: K=1 identity iota (int32)
    # ---------------------------------------------------------------

    def test_k1_identity_int32(self):
        """K=1 identity iota: OUT[i] = i + __l0, producing np.arange(64)."""
        W, N = 8, 64
        sdfg = _build_iota_store_sdfg_k1(
            "iota_rt_k1_id_i32", N, W, "i + __l0", dace.int32,
        )
        out = np.zeros(N, dtype=np.int32)
        results = _run_cutile(sdfg, OUT=out)
        expected = np.arange(N, dtype=np.int32)
        np.testing.assert_array_equal(results["OUT"], expected)

    # ---------------------------------------------------------------
    # Test 2: K=1 affine iota (int32)
    # ---------------------------------------------------------------

    def test_k1_affine_int32(self):
        """K=1 affine iota: OUT[i] = 2 * (i + __l0) + 1."""
        W, N = 8, 64
        sdfg = _build_iota_store_sdfg_k1(
            "iota_rt_k1_aff_i32", N, W, "2 * (i + __l0) + 1", dace.int32,
        )
        out = np.zeros(N, dtype=np.int32)
        results = _run_cutile(sdfg, OUT=out)
        expected = (2 * np.arange(N) + 1).astype(np.int32)
        np.testing.assert_array_equal(results["OUT"], expected)

    # ---------------------------------------------------------------
    # Test 3: K=1 identity iota (int64)
    # ---------------------------------------------------------------

    def test_k1_identity_int64(self):
        """K=1 identity iota with int64 dtype."""
        W, N = 8, 64
        sdfg = _build_iota_store_sdfg_k1(
            "iota_rt_k1_id_i64", N, W, "i + __l0", dace.int64,
        )
        out = np.zeros(N, dtype=np.int64)
        results = _run_cutile(sdfg, OUT=out)
        expected = np.arange(N, dtype=np.int64)
        np.testing.assert_array_equal(results["OUT"], expected)

    # ---------------------------------------------------------------
    # Test 4: K=1 identity iota (float32)
    # ---------------------------------------------------------------

    def test_k1_identity_float32(self):
        """K=1 identity iota with float32 dtype."""
        W, N = 8, 64
        sdfg = _build_iota_store_sdfg_k1(
            "iota_rt_k1_id_f32", N, W, "i + __l0", dace.float32,
        )
        out = np.zeros(N, dtype=np.float32)
        results = _run_cutile(sdfg, OUT=out)
        expected = np.arange(N, dtype=np.float32)
        np.testing.assert_allclose(results["OUT"], expected, rtol=1e-6)

    # ---------------------------------------------------------------
    # Test 5: K=1 identity iota (float64)
    # ---------------------------------------------------------------

    def test_k1_identity_float64(self):
        """K=1 identity iota with float64 dtype."""
        W, N = 8, 64
        sdfg = _build_iota_store_sdfg_k1(
            "iota_rt_k1_id_f64", N, W, "i + __l0", dace.float64,
        )
        out = np.zeros(N, dtype=np.float64)
        results = _run_cutile(sdfg, OUT=out)
        expected = np.arange(N, dtype=np.float64)
        np.testing.assert_allclose(results["OUT"], expected, rtol=1e-14)

    # ---------------------------------------------------------------
    # Test 6: K=2 multi-dim iota
    # ---------------------------------------------------------------

    def test_k2_flat_index(self):
        """K=2 iota: OUT[i, j] = __l0 * 8 + __l1 (flat index within tile)."""
        W0, W1 = 4, 8
        M, N = 16, 64
        sdfg = _build_iota_store_sdfg_k2(
            "iota_rt_k2_flat", M, N, W0, W1,
            "__l0 * 8 + __l1", dace.int32,
        )
        out = np.zeros((M, N), dtype=np.int32)
        results = _run_cutile(sdfg, OUT=out)
        # Each (W0, W1) tile should contain __l0 * 8 + __l1 independently.
        expected = np.zeros((M, N), dtype=np.int32)
        for ti in range(0, M, W0):
            for tj in range(0, N, W1):
                for l0 in range(W0):
                    for l1 in range(W1):
                        expected[ti + l0, tj + l1] = l0 * 8 + l1
        np.testing.assert_array_equal(results["OUT"], expected)

    # ---------------------------------------------------------------
    # Test 7: Parametrized tile widths
    # ---------------------------------------------------------------

    @pytest.mark.parametrize("W", [4, 8, 16, 32])
    def test_k1_parametrized_widths(self, W: int):
        """K=1 identity iota with various tile widths."""
        N = 128  # Divisible by all test widths
        sdfg = _build_iota_store_sdfg_k1(
            f"iota_rt_k1_w{W}", N, W, "i + __l0", dace.int32,
        )
        out = np.zeros(N, dtype=np.int32)
        results = _run_cutile(sdfg, OUT=out)
        expected = np.arange(N, dtype=np.int32)
        np.testing.assert_array_equal(results["OUT"], expected)

    # ---------------------------------------------------------------
    # Test 8: K=2 global-index iota
    # ---------------------------------------------------------------

    def test_k2_global_index(self):
        """K=2 iota computing global row-major index: (i + __l0) * N + (j + __l1)."""
        W0, W1 = 4, 8
        M, N = 16, 32
        sdfg = _build_iota_store_sdfg_k2(
            "iota_rt_k2_glob", M, N, W0, W1,
            f"(i + __l0) * {N} + (j + __l1)", dace.int32,
        )
        out = np.zeros((M, N), dtype=np.int32)
        results = _run_cutile(sdfg, OUT=out)
        expected = np.zeros((M, N), dtype=np.int32)
        for r in range(M):
            for c in range(N):
                expected[r, c] = r * N + c
        np.testing.assert_array_equal(results["OUT"], expected)

    # ---------------------------------------------------------------
    # Test 9: Iota -> TileBinop -> TileStore chain
    # ---------------------------------------------------------------

    def test_iota_binop_chain(self):
        """TileIota -> TileBinop (multiply by 2) -> TileStore."""
        W, N = 8, 64
        sdfg = SDFG("iota_rt_binop_chain")
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.add_array("OUT", (N,), dace.int32, storage=StorageType.GPU_Global)
        sdfg.add_array(
            "_tile_iota", (W,), dace.int32,
            storage=StorageType.CuTile_Tile, transient=True,
        )
        sdfg.add_array(
            "_tile_result", (W,), dace.int32,
            storage=StorageType.CuTile_Tile, transient=True,
        )

        state = sdfg.add_state("main")
        me, mx = state.add_map(
            "iota_map", {"i": f"0:{N}:{W}"},
            schedule=ScheduleType.CuTile,
        )

        # TileIota: produces lane indices
        iota = TileIota("iota", widths=(W,), expr="i + __l0")
        iota.target_isa = "CUTILE"
        iota.implementation = "cutile"
        state.add_node(iota)

        tile_iota_an = state.add_access("_tile_iota")

        # TileBinop: multiply by 2 (Symbol on the RHS)
        binop = TileBinop(
            "mul2", widths=(W,), op="*",
            kind_b="Symbol", expr_b="2",
        )
        binop.target_isa = "CUTILE"
        binop.implementation = "cutile"
        state.add_node(binop)

        tile_result_an = state.add_access("_tile_result")

        # TileStore: write result to OUT
        store = TileStore("store", widths=(W,))
        store.target_isa = "CUTILE"
        store.implementation = "cutile"
        state.add_node(store)

        out_an = state.add_write("OUT")

        # Wiring
        state.add_edge(me, None, iota, None, Memlet())
        state.add_edge(iota, "_dst", tile_iota_an, None,
                       Memlet(data="_tile_iota", subset=f"0:{W}"))
        state.add_edge(tile_iota_an, None, binop, "_a",
                       Memlet(data="_tile_iota", subset=f"0:{W}"))
        state.add_edge(binop, "_c", tile_result_an, None,
                       Memlet(data="_tile_result", subset=f"0:{W}"))
        state.add_edge(tile_result_an, None, store, "_src",
                       Memlet(data="_tile_result", subset=f"0:{W}"))
        state.add_memlet_path(
            store, mx, out_an,
            src_conn="_dst",
            memlet=Memlet(data="OUT", subset=f"i:i+{W}"),
        )

        sdfg.expand_library_nodes()

        out = np.zeros(N, dtype=np.int32)
        results = _run_cutile(sdfg, OUT=out)
        expected = (np.arange(N, dtype=np.int32) * 2)
        np.testing.assert_array_equal(results["OUT"], expected)

    # ---------------------------------------------------------------
    # Test 10: K=1 strided expression
    # ---------------------------------------------------------------

    def test_k1_strided_expr(self):
        """K=1 strided iota: OUT[i] = i + 2 * __l0 (even indices)."""
        W, N = 8, 64
        sdfg = _build_iota_store_sdfg_k1(
            "iota_rt_k1_stride2", N, W, "i + 2 * __l0", dace.int32,
        )
        out = np.zeros(N, dtype=np.int32)
        results = _run_cutile(sdfg, OUT=out)
        # Within each tile of W elements starting at map position `ti`,
        # element l gets value ti + 2*l.
        expected = np.zeros(N, dtype=np.int32)
        for ti in range(0, N, W):
            for l0 in range(W):
                expected[ti + l0] = ti + 2 * l0
        np.testing.assert_array_equal(results["OUT"], expected)

    # ---------------------------------------------------------------
    # Test 11: K=1 with larger array, wider tile
    # ---------------------------------------------------------------

    def test_k1_large_array_w16(self):
        """K=1 identity iota with W=16, N=256."""
        W, N = 16, 256
        sdfg = _build_iota_store_sdfg_k1(
            "iota_rt_k1_w16_n256", N, W, "i + __l0", dace.int32,
        )
        out = np.zeros(N, dtype=np.int32)
        results = _run_cutile(sdfg, OUT=out)
        expected = np.arange(N, dtype=np.int32)
        np.testing.assert_array_equal(results["OUT"], expected)

    # ---------------------------------------------------------------
    # Test 12: K=2 non-trivial expr with addition
    # ---------------------------------------------------------------

    def test_k2_add_lanes(self):
        """K=2 iota: OUT[i, j] = __l0 + __l1."""
        W0, W1 = 4, 8
        M, N = 16, 32
        sdfg = _build_iota_store_sdfg_k2(
            "iota_rt_k2_addl", M, N, W0, W1,
            "__l0 + __l1", dace.int32,
        )
        out = np.zeros((M, N), dtype=np.int32)
        results = _run_cutile(sdfg, OUT=out)
        expected = np.zeros((M, N), dtype=np.int32)
        for ti in range(0, M, W0):
            for tj in range(0, N, W1):
                for l0 in range(W0):
                    for l1 in range(W1):
                        expected[ti + l0, tj + l1] = l0 + l1
        np.testing.assert_array_equal(results["OUT"], expected)

    # ---------------------------------------------------------------
    # Test 13: Constant fill (no lane variable dependency)
    # ---------------------------------------------------------------

    def test_k1_constant_fill(self):
        """K=1 constant fill: OUT[:] = 42 (via ``42 + 0 * __l0``).

        A bare ``"42"`` would produce a Python scalar (not a cuTile tile),
        which ``ct.store`` cannot write.  Adding ``0 * __l0`` forces the
        expression through ``ct.arange`` and yields a proper tile-shaped
        constant.
        """
        W, N = 8, 64
        sdfg = _build_iota_store_sdfg_k1(
            "iota_rt_k1_const42", N, W, "42 + 0 * __l0", dace.int32,
        )
        out = np.zeros(N, dtype=np.int32)
        results = _run_cutile(sdfg, OUT=out)
        expected = np.full(N, 42, dtype=np.int32)
        np.testing.assert_array_equal(results["OUT"], expected)

    # ---------------------------------------------------------------
    # Test 14: K=1 identity iota with W=4, N=4 (single tile)
    # ---------------------------------------------------------------

    def test_k1_single_tile(self):
        """K=1 identity iota with exactly one tile (N=W)."""
        W, N = 4, 4
        sdfg = _build_iota_store_sdfg_k1(
            "iota_rt_k1_single", N, W, "i + __l0", dace.int32,
        )
        out = np.zeros(N, dtype=np.int32)
        results = _run_cutile(sdfg, OUT=out)
        expected = np.arange(N, dtype=np.int32)
        np.testing.assert_array_equal(results["OUT"], expected)

    # ---------------------------------------------------------------
    # Test 15: K=1 square expression (nonlinear in lane index)
    # ---------------------------------------------------------------

    def test_k1_square_expr(self):
        """K=1 nonlinear: OUT[i] = (i + __l0) * (i + __l0).

        The cuTile runtime does not support ``pow`` on integers, so the
        square is expressed as a multiplication instead of ``** 2``.
        """
        W, N = 8, 64
        sdfg = _build_iota_store_sdfg_k1(
            "iota_rt_k1_sq", N, W, "(i + __l0) * (i + __l0)", dace.int32,
        )
        out = np.zeros(N, dtype=np.int32)
        results = _run_cutile(sdfg, OUT=out)
        expected = (np.arange(N, dtype=np.int32) ** 2).astype(np.int32)
        np.testing.assert_array_equal(results["OUT"], expected)


# Mark all methods in the runtime class as GPU tests.
TestTileIotaCutileRuntime.pytestmark = pytest.mark.gpu


# ---------------------------------------------------------------------------
# Expression validation tests (no GPU required)
# ---------------------------------------------------------------------------


class TestTileIotaExprValidation:
    """Tests for ``validate_cutile_expr`` rejecting C++ patterns.

    These tests verify that the cuTile expansion raises ``ValueError``
    when the expression contains obviously-C++ constructs that would
    produce invalid Python code.
    """

    def test_rejects_std_namespace(self):
        """``std::min(...)`` is rejected."""
        from dace.libraries.tileops._pure_codegen import validate_cutile_expr
        with pytest.raises(ValueError, match="C\\+\\+"):
            validate_cutile_expr("std::min(__l0, 7)")

    def test_rejects_pointer_deref(self):
        """``ptr->field`` is rejected."""
        from dace.libraries.tileops._pure_codegen import validate_cutile_expr
        with pytest.raises(ValueError, match="C\\+\\+"):
            validate_cutile_expr("ptr->field")

    def test_rejects_sizeof(self):
        """``sizeof(int)`` is rejected."""
        from dace.libraries.tileops._pure_codegen import validate_cutile_expr
        with pytest.raises(ValueError, match="C\\+\\+"):
            validate_cutile_expr("sizeof(int)")

    def test_rejects_trailing_semicolon(self):
        """``__l0;`` (trailing semicolon) is rejected."""
        from dace.libraries.tileops._pure_codegen import validate_cutile_expr
        with pytest.raises(ValueError, match="C\\+\\+"):
            validate_cutile_expr("__l0;")

    def test_accepts_valid_python_expr(self):
        """Valid Python expressions are accepted without error."""
        from dace.libraries.tileops._pure_codegen import validate_cutile_expr
        # These should NOT raise
        validate_cutile_expr("i + __l0")
        validate_cutile_expr("2 * (i + __l0) + 1")
        validate_cutile_expr("__l0 * 8 + __l1")
        validate_cutile_expr("min(__l0, 7)")
        validate_cutile_expr("_idx[__l0]")

    def test_validation_called_during_expansion(self):
        """The cuTile expansion calls ``validate_cutile_expr``."""
        iota = TileIota("bad", widths=(8,), expr="std::min(__l0, 7)")
        iota.implementation = "cutile"
        sdfg = dace.SDFG("validate_during_expand")
        state = sdfg.add_state("main")
        state.add_node(iota)
        cls = iota.implementations["cutile"]
        with pytest.raises(ValueError, match="C\\+\\+"):
            cls.expansion(iota, state, sdfg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=300"])
