# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU runtime tests for the cuTile expansion of :class:`TileReduce`.

Two test patterns are used:

**Pattern A (hand-written tasklet):** A manually written Python tasklet
containing cuTile reduction calls (``ct.sum``, ``ct.prod``, etc.) is
placed inside a NestedSDFG within a CuTile-scheduled map.  This proves
the cuTile runtime handles reductions correctly, independent of our
expansion machinery.

**Pattern B (expansion-based):** A :class:`TileReduce` library node
with ``implementation='cutile'`` is placed in the NestedSDFG, then
``expand_library_nodes()`` is called.  This tests the full expansion
path through to GPU execution.

Both patterns use the AccessNode-centric codegen structure:
``A (GPU_Global) -> CuTile map -> _tile_src (CuTile_Tile) -> NestedSDFG
-> _tile_dst (CuTile_Tile) -> CuTile map exit -> B (GPU_Global)``.

Single-block maps are used (each dim iterates exactly once with
``step == width``) so the output is deterministic without cross-tile
accumulation.
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.memlet import Memlet
from dace.dtypes import ScheduleType, StorageType, Language
from dace.sdfg import SDFG
from dace.libraries.tileops.nodes.tile_reduce import (
    TileReduce,
    _identity_literal_cutile,
    _dace_dtype_to_cutile_str,
)

# All tests in this file require GPU.
pytestmark = pytest.mark.gpu


# ============================================================
# Helpers
# ============================================================


def _run_cutile(sdfg, **kwargs):
    """Compile and run a cuTile SDFG, converting numpy<->cupy.

    :param sdfg: The SDFG to compile and run.
    :param kwargs: Named arguments for the SDFG (arrays and symbols).
    :returns: Dictionary mapping array names to numpy results.
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


def _np_reduce(op, arr, axis=None):
    """NumPy reference reduction.

    :param op: One of ``+``, ``*``, ``min``, ``max``.
    :param arr: Input array.
    :param axis: Reduction axis (``None`` for full reduction).
    :returns: Reduced array.
    """
    if op == '+':
        return np.sum(arr, axis=axis)
    if op == '*':
        return np.prod(arr, axis=axis)
    if op == 'min':
        return np.min(arr, axis=axis)
    if op == 'max':
        return np.max(arr, axis=axis)
    raise ValueError(f"Unknown op: {op}")


def _ct_reduce_fn(op):
    """Return the cuTile reduction function name string.

    :param op: One of ``+``, ``*``, ``min``, ``max``.
    :returns: String like ``"ct.sum"``.
    """
    return {'+': "ct.sum", '*': "ct.prod", 'min': "ct.min", 'max': "ct.max"}[op]


def _dst_shape(widths, axis):
    """Compute the output shape after reducing along *axis*.

    :param widths: Tuple of tile widths.
    :param axis: Reduction axis (``None`` for full reduction).
    :returns: Output shape tuple.
    """
    if axis is None:
        return (1,)
    return tuple(w for i, w in enumerate(widths) if i != axis)


# ============================================================
# SDFG builder — Pattern A (hand-written tasklet)
# ============================================================


def _build_handwritten_sdfg(widths, op, axis, has_mask, dtype, name):
    """Build a cuTile reduction SDFG with a hand-written Python tasklet.

    The inner NestedSDFG contains a single Python tasklet with cuTile
    reduction code (e.g. ``_dst = ct.sum(_src, axis=None)``).

    :param widths: Tuple of tile widths (per-dim).
    :param op: Reduction op (``+``, ``*``, ``min``, ``max``).
    :param axis: Reduction axis (``None`` for full).
    :param has_mask: Whether a boolean mask input is present.
    :param dtype: DaCe data type for the data arrays.
    :param name: Unique SDFG name.
    :returns: The constructed SDFG.
    """
    K = len(widths)
    dst_sh = _dst_shape(widths, axis)
    np_dtype = dtype.as_numpy_dtype()

    # -- Outer SDFG --
    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python

    sdfg.add_array("A", widths, dtype, storage=StorageType.GPU_Global)
    sdfg.add_array("B", dst_sh, dtype, storage=StorageType.GPU_Global)
    if has_mask:
        sdfg.add_array("M", widths, dace.bool, storage=StorageType.GPU_Global)

    sdfg.add_array(
        "_tile_src", widths, dtype,
        storage=StorageType.CuTile_Tile, transient=True,
    )
    sdfg.add_array(
        "_tile_dst", dst_sh, dtype,
        storage=StorageType.CuTile_Tile, transient=True,
    )
    if has_mask:
        sdfg.add_array(
            "_tile_mask", widths, dace.bool,
            storage=StorageType.CuTile_Tile, transient=True,
        )

    state = sdfg.add_state("main")

    # CuTile map: single-block (each dim iterates once).
    map_params = {f'__i{d}': f'0:{w}:{w}' for d, w in enumerate(widths)}
    me, mx = state.add_map('cutile_map', map_params, schedule=ScheduleType.CuTile)

    # -- Inner SDFG --
    inner_sdfg = SDFG(f"{name}_inner")
    inner_sdfg.add_array(
        "_tile_src", widths, dtype, storage=StorageType.CuTile_Tile,
    )
    inner_sdfg.add_array(
        "_tile_dst", dst_sh, dtype, storage=StorageType.CuTile_Tile,
    )
    if has_mask:
        inner_sdfg.add_array(
            "_tile_mask", widths, dace.bool, storage=StorageType.CuTile_Tile,
        )

    inner_state = inner_sdfg.add_state("compute")

    # Build tasklet code.
    if has_mask:
        identity = _identity_literal_cutile(op, dtype)
        code = (
            f"_rhs = ct.where(_mask, _src, {identity})\n"
            f"_dst = {_ct_reduce_fn(op)}(_rhs, axis={axis})"
        )
    else:
        code = f"_dst = {_ct_reduce_fn(op)}(_src, axis={axis})"

    in_connectors = {"_src"}
    if has_mask:
        in_connectors.add("_mask")

    tasklet = inner_state.add_tasklet(
        "reduce", in_connectors, {"_dst"}, code, language=Language.Python,
    )

    src_subset = ','.join(f'0:{w}' for w in widths)
    dst_subset = ','.join(f'0:{s}' for s in dst_sh)

    in_src = inner_state.add_read("_tile_src")
    out_dst = inner_state.add_write("_tile_dst")
    inner_state.add_edge(
        in_src, None, tasklet, "_src",
        Memlet(data="_tile_src", subset=src_subset),
    )
    inner_state.add_edge(
        tasklet, "_dst", out_dst, None,
        Memlet(data="_tile_dst", subset=dst_subset),
    )
    if has_mask:
        in_mask = inner_state.add_read("_tile_mask")
        inner_state.add_edge(
            in_mask, None, tasklet, "_mask",
            Memlet(data="_tile_mask", subset=src_subset),
        )

    # -- Wire outer SDFG --
    nsdfg_inputs = {"_tile_src"}
    if has_mask:
        nsdfg_inputs.add("_tile_mask")
    nsdfg = state.add_nested_sdfg(
        inner_sdfg, nsdfg_inputs, {"_tile_dst"},
        symbol_mapping={},
    )

    a_node = state.add_read("A")
    b_node = state.add_write("B")
    tile_src = state.add_access("_tile_src")
    tile_dst = state.add_access("_tile_dst")

    a_subset = ','.join(f'__i{d}:__i{d}+{w}' for d, w in enumerate(widths))
    state.add_memlet_path(
        a_node, me, tile_src,
        memlet=Memlet(data="A", subset=a_subset),
    )
    state.add_edge(
        tile_src, None, nsdfg, "_tile_src",
        Memlet(data="_tile_src", subset=src_subset),
    )
    state.add_edge(
        nsdfg, "_tile_dst", tile_dst, None,
        Memlet(data="_tile_dst", subset=dst_subset),
    )
    b_subset = ','.join(f'0:{s}' for s in dst_sh)
    state.add_memlet_path(
        tile_dst, mx, b_node,
        memlet=Memlet(data="B", subset=b_subset),
    )

    if has_mask:
        m_node = state.add_read("M")
        tile_mask = state.add_access("_tile_mask")
        state.add_memlet_path(
            m_node, me, tile_mask,
            memlet=Memlet(data="M", subset=a_subset),
        )
        state.add_edge(
            tile_mask, None, nsdfg, "_tile_mask",
            Memlet(data="_tile_mask", subset=src_subset),
        )

    sdfg.validate()
    return sdfg


# ============================================================
# SDFG builder — Pattern B (expansion-based)
# ============================================================


def _build_expansion_sdfg(widths, op, axis, has_mask, dtype, name):
    """Build a cuTile reduction SDFG using :class:`TileReduce` lib-node expansion.

    The inner NestedSDFG contains a ``TileReduce`` library node with
    ``implementation='cutile'`` and ``target_isa='CUTILE'``.  After
    construction, ``inner_sdfg.expand_library_nodes()`` is called so
    the expansion path is tested end-to-end.

    :param widths: Tuple of tile widths (per-dim).
    :param op: Reduction op (``+``, ``*``, ``min``, ``max``).
    :param axis: Reduction axis (``None`` for full).
    :param has_mask: Whether a boolean mask input is present.
    :param dtype: DaCe data type for the data arrays.
    :param name: Unique SDFG name.
    :returns: The constructed SDFG.
    """
    K = len(widths)
    dst_sh = _dst_shape(widths, axis)

    # -- Outer SDFG --
    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python

    sdfg.add_array("A", widths, dtype, storage=StorageType.GPU_Global)
    sdfg.add_array("B", dst_sh, dtype, storage=StorageType.GPU_Global)
    if has_mask:
        sdfg.add_array("M", widths, dace.bool, storage=StorageType.GPU_Global)

    sdfg.add_array(
        "_tile_src", widths, dtype,
        storage=StorageType.CuTile_Tile, transient=True,
    )
    sdfg.add_array(
        "_tile_dst", dst_sh, dtype,
        storage=StorageType.CuTile_Tile, transient=True,
    )
    if has_mask:
        sdfg.add_array(
            "_tile_mask", widths, dace.bool,
            storage=StorageType.CuTile_Tile, transient=True,
        )

    state = sdfg.add_state("main")

    # CuTile map: single-block.
    map_params = {f'__i{d}': f'0:{w}:{w}' for d, w in enumerate(widths)}
    me, mx = state.add_map('cutile_map', map_params, schedule=ScheduleType.CuTile)

    # -- Inner SDFG --
    inner_sdfg = SDFG(f"{name}_inner")
    inner_sdfg.add_array(
        "_tile_src", widths, dtype, storage=StorageType.CuTile_Tile,
    )
    inner_sdfg.add_array(
        "_tile_dst", dst_sh, dtype, storage=StorageType.CuTile_Tile,
    )
    if has_mask:
        inner_sdfg.add_array(
            "_tile_mask", widths, dace.bool, storage=StorageType.CuTile_Tile,
        )

    inner_state = inner_sdfg.add_state("compute")

    # Place TileReduce lib node.
    reduce_node = TileReduce(
        name='tile_reduce', widths=list(widths),
        op=op, axis=axis, has_mask=has_mask,
    )
    reduce_node.implementation = 'cutile'
    reduce_node.target_isa = 'CUTILE'
    inner_state.add_node(reduce_node)

    src_subset = ','.join(f'0:{w}' for w in widths)
    dst_subset = ','.join(f'0:{s}' for s in dst_sh)

    in_src = inner_state.add_read("_tile_src")
    out_dst = inner_state.add_write("_tile_dst")
    inner_state.add_edge(
        in_src, None, reduce_node, "_src",
        Memlet(data="_tile_src", subset=src_subset),
    )
    inner_state.add_edge(
        reduce_node, "_dst", out_dst, None,
        Memlet(data="_tile_dst", subset=dst_subset),
    )
    if has_mask:
        in_mask = inner_state.add_read("_tile_mask")
        inner_state.add_edge(
            in_mask, None, reduce_node, "_mask",
            Memlet(data="_tile_mask", subset=src_subset),
        )

    # Expand the lib node into a cuTile tasklet.
    inner_sdfg.expand_library_nodes()

    # -- Wire outer SDFG --
    nsdfg_inputs = {"_tile_src"}
    if has_mask:
        nsdfg_inputs.add("_tile_mask")
    nsdfg = state.add_nested_sdfg(
        inner_sdfg, nsdfg_inputs, {"_tile_dst"},
        symbol_mapping={},
    )

    a_node = state.add_read("A")
    b_node = state.add_write("B")
    tile_src = state.add_access("_tile_src")
    tile_dst = state.add_access("_tile_dst")

    a_subset = ','.join(f'__i{d}:__i{d}+{w}' for d, w in enumerate(widths))
    state.add_memlet_path(
        a_node, me, tile_src,
        memlet=Memlet(data="A", subset=a_subset),
    )
    state.add_edge(
        tile_src, None, nsdfg, "_tile_src",
        Memlet(data="_tile_src", subset=src_subset),
    )
    state.add_edge(
        nsdfg, "_tile_dst", tile_dst, None,
        Memlet(data="_tile_dst", subset=dst_subset),
    )
    b_subset = ','.join(f'0:{s}' for s in dst_sh)
    state.add_memlet_path(
        tile_dst, mx, b_node,
        memlet=Memlet(data="B", subset=b_subset),
    )

    if has_mask:
        m_node = state.add_read("M")
        tile_mask = state.add_access("_tile_mask")
        state.add_memlet_path(
            m_node, me, tile_mask,
            memlet=Memlet(data="M", subset=a_subset),
        )
        state.add_edge(
            tile_mask, None, nsdfg, "_tile_mask",
            Memlet(data="_tile_mask", subset=src_subset),
        )

    sdfg.validate()
    return sdfg


# ============================================================
# Pattern A: Hand-written cuTile reduction tasklets (unmasked)
# ============================================================


class TestTileReduceCutileBasic:
    """Hand-written cuTile reduction tasklets -- proves the runtime works."""

    def test_k1_sum_full(self):
        """K=1 full sum reduction to scalar."""
        widths = (8,)
        sdfg = _build_handwritten_sdfg(widths, '+', None, False, dace.float64, 'hw_k1_sum_full')
        A = np.random.default_rng(42).random(widths)
        B = np.zeros((1,), dtype=np.float64)
        result = _run_cutile(sdfg, A=A, B=B)
        np.testing.assert_allclose(result['B'], [np.sum(A)], rtol=1e-12)

    def test_k1_prod_full(self):
        """K=1 full product reduction to scalar."""
        widths = (8,)
        sdfg = _build_handwritten_sdfg(widths, '*', None, False, dace.float64, 'hw_k1_prod_full')
        A = np.random.default_rng(43).random(widths) + 0.5
        B = np.zeros((1,), dtype=np.float64)
        result = _run_cutile(sdfg, A=A, B=B)
        np.testing.assert_allclose(result['B'], [np.prod(A)], rtol=1e-10)

    def test_k1_min_full(self):
        """K=1 full min reduction to scalar."""
        widths = (8,)
        sdfg = _build_handwritten_sdfg(widths, 'min', None, False, dace.float64, 'hw_k1_min_full')
        A = np.random.default_rng(44).random(widths)
        B = np.zeros((1,), dtype=np.float64)
        result = _run_cutile(sdfg, A=A, B=B)
        np.testing.assert_allclose(result['B'], [np.min(A)], rtol=1e-12)

    def test_k1_max_full(self):
        """K=1 full max reduction to scalar."""
        widths = (8,)
        sdfg = _build_handwritten_sdfg(widths, 'max', None, False, dace.float64, 'hw_k1_max_full')
        A = np.random.default_rng(45).random(widths)
        B = np.zeros((1,), dtype=np.float64)
        result = _run_cutile(sdfg, A=A, B=B)
        np.testing.assert_allclose(result['B'], [np.max(A)], rtol=1e-12)

    def test_k2_sum_full(self):
        """K=2 full sum reduction to scalar."""
        widths = (4, 8)
        sdfg = _build_handwritten_sdfg(widths, '+', None, False, dace.float64, 'hw_k2_sum_full')
        A = np.random.default_rng(46).random(widths)
        B = np.zeros((1,), dtype=np.float64)
        result = _run_cutile(sdfg, A=A, B=B)
        np.testing.assert_allclose(result['B'], [np.sum(A)], rtol=1e-12)

    def test_k2_sum_axis0(self):
        """K=2 sum along axis 0 -> output shape (8,)."""
        widths = (4, 8)
        sdfg = _build_handwritten_sdfg(widths, '+', 0, False, dace.float64, 'hw_k2_sum_ax0')
        A = np.random.default_rng(47).random(widths)
        B = np.zeros((8,), dtype=np.float64)
        result = _run_cutile(sdfg, A=A, B=B)
        np.testing.assert_allclose(result['B'], np.sum(A, axis=0), rtol=1e-12)

    def test_k2_sum_axis1(self):
        """K=2 sum along axis 1 -> output shape (4,)."""
        widths = (4, 8)
        sdfg = _build_handwritten_sdfg(widths, '+', 1, False, dace.float64, 'hw_k2_sum_ax1')
        A = np.random.default_rng(48).random(widths)
        B = np.zeros((4,), dtype=np.float64)
        result = _run_cutile(sdfg, A=A, B=B)
        np.testing.assert_allclose(result['B'], np.sum(A, axis=1), rtol=1e-12)

    def test_k2_min_axis0(self):
        """K=2 min along axis 0 -> output shape (8,)."""
        widths = (4, 8)
        sdfg = _build_handwritten_sdfg(widths, 'min', 0, False, dace.float64, 'hw_k2_min_ax0')
        A = np.random.default_rng(49).random(widths)
        B = np.zeros((8,), dtype=np.float64)
        result = _run_cutile(sdfg, A=A, B=B)
        np.testing.assert_allclose(result['B'], np.min(A, axis=0), rtol=1e-12)

    def test_k2_max_axis1(self):
        """K=2 max along axis 1 -> output shape (4,)."""
        widths = (4, 8)
        sdfg = _build_handwritten_sdfg(widths, 'max', 1, False, dace.float64, 'hw_k2_max_ax1')
        A = np.random.default_rng(50).random(widths)
        B = np.zeros((4,), dtype=np.float64)
        result = _run_cutile(sdfg, A=A, B=B)
        np.testing.assert_allclose(result['B'], np.max(A, axis=1), rtol=1e-12)

    def test_k2_prod_full(self):
        """K=2 full product reduction to scalar (narrow value range)."""
        widths = (4, 8)
        sdfg = _build_handwritten_sdfg(widths, '*', None, False, dace.float64, 'hw_k2_prod_full')
        A = np.random.default_rng(51).random(widths) * 0.5 + 0.75
        B = np.zeros((1,), dtype=np.float64)
        result = _run_cutile(sdfg, A=A, B=B)
        np.testing.assert_allclose(result['B'], [np.prod(A)], rtol=1e-10)


# ============================================================
# Pattern A: Hand-written cuTile reduction tasklets (masked)
# ============================================================


class TestTileReduceCutileMasked:
    """Masked cuTile reductions with identity pre-fill."""

    def test_k1_sum_masked(self):
        """K=1 masked sum reduction."""
        widths = (8,)
        rng = np.random.default_rng(60)
        A = rng.random(widths)
        M = np.array([True, False, True, True, False, True, False, True],
                     dtype=np.bool_)
        sdfg = _build_handwritten_sdfg(widths, '+', None, True, dace.float64, 'hw_k1_sum_mask')
        B = np.zeros((1,), dtype=np.float64)
        result = _run_cutile(sdfg, A=A, M=M, B=B)
        np.testing.assert_allclose(result['B'], [np.sum(A[M])], rtol=1e-12)

    def test_k1_min_masked(self):
        """K=1 masked min reduction."""
        widths = (8,)
        rng = np.random.default_rng(61)
        A = rng.random(widths)
        M = np.array([True, False, True, True, False, True, False, True],
                     dtype=np.bool_)
        sdfg = _build_handwritten_sdfg(widths, 'min', None, True, dace.float64, 'hw_k1_min_mask')
        B = np.zeros((1,), dtype=np.float64)
        result = _run_cutile(sdfg, A=A, M=M, B=B)
        np.testing.assert_allclose(result['B'], [np.min(A[M])], rtol=1e-12)

    def test_k1_max_masked(self):
        """K=1 masked max reduction."""
        widths = (8,)
        rng = np.random.default_rng(62)
        A = rng.random(widths)
        M = np.array([True, False, True, True, False, True, False, True],
                     dtype=np.bool_)
        sdfg = _build_handwritten_sdfg(widths, 'max', None, True, dace.float64, 'hw_k1_max_mask')
        B = np.zeros((1,), dtype=np.float64)
        result = _run_cutile(sdfg, A=A, M=M, B=B)
        np.testing.assert_allclose(result['B'], [np.max(A[M])], rtol=1e-12)

    def test_k1_prod_masked(self):
        """K=1 masked product reduction."""
        widths = (8,)
        rng = np.random.default_rng(63)
        A = rng.random(widths) + 0.5
        M = np.array([True, False, True, True, False, True, False, True],
                     dtype=np.bool_)
        sdfg = _build_handwritten_sdfg(widths, '*', None, True, dace.float64, 'hw_k1_prod_mask')
        B = np.zeros((1,), dtype=np.float64)
        result = _run_cutile(sdfg, A=A, M=M, B=B)
        np.testing.assert_allclose(result['B'], [np.prod(A[M])], rtol=1e-10)

    def test_k2_sum_axis0_masked(self):
        """K=2 masked sum along axis 0."""
        widths = (4, 8)
        rng = np.random.default_rng(64)
        A = rng.random(widths)
        M = rng.choice([True, False], size=widths)
        sdfg = _build_handwritten_sdfg(widths, '+', 0, True, dace.float64, 'hw_k2_sum_ax0_mask')
        B = np.zeros((8,), dtype=np.float64)
        result = _run_cutile(sdfg, A=A, M=M, B=B)
        expected = np.sum(np.where(M, A, 0.0), axis=0)
        np.testing.assert_allclose(result['B'], expected, rtol=1e-12)

    def test_k1_sum_all_false_mask(self):
        """All-false mask -> result is identity (0.0 for sum)."""
        widths = (8,)
        A = np.ones(widths, dtype=np.float64) * 99.0
        M = np.zeros(widths, dtype=np.bool_)
        sdfg = _build_handwritten_sdfg(widths, '+', None, True, dace.float64, 'hw_k1_sum_allfalse')
        B = np.full((1,), -1.0, dtype=np.float64)
        result = _run_cutile(sdfg, A=A, M=M, B=B)
        np.testing.assert_allclose(result['B'], [0.0], rtol=1e-12)

    def test_k1_sum_all_true_mask(self):
        """All-true mask -> same as unmasked."""
        widths = (8,)
        rng = np.random.default_rng(66)
        A = rng.random(widths)
        M = np.ones(widths, dtype=np.bool_)
        sdfg = _build_handwritten_sdfg(widths, '+', None, True, dace.float64, 'hw_k1_sum_alltrue')
        B = np.zeros((1,), dtype=np.float64)
        result = _run_cutile(sdfg, A=A, M=M, B=B)
        np.testing.assert_allclose(result['B'], [np.sum(A)], rtol=1e-12)


# ============================================================
# Pattern A: Data type variations
# ============================================================


class TestTileReduceCutileDtypes:
    """Multiple dtypes through the hand-written cuTile path."""

    def test_float32_sum(self):
        """float32 full sum."""
        widths = (8,)
        A = np.random.default_rng(70).random(widths).astype(np.float32)
        sdfg = _build_handwritten_sdfg(widths, '+', None, False, dace.float32, 'hw_f32_sum')
        B = np.zeros((1,), dtype=np.float32)
        result = _run_cutile(sdfg, A=A, B=B)
        np.testing.assert_allclose(result['B'], [np.sum(A)], rtol=1e-5)

    def test_float64_sum(self):
        """float64 full sum."""
        widths = (8,)
        A = np.random.default_rng(71).random(widths)
        sdfg = _build_handwritten_sdfg(widths, '+', None, False, dace.float64, 'hw_f64_sum')
        B = np.zeros((1,), dtype=np.float64)
        result = _run_cutile(sdfg, A=A, B=B)
        np.testing.assert_allclose(result['B'], [np.sum(A)], rtol=1e-12)

    def test_int32_sum(self):
        """int32 full sum."""
        widths = (8,)
        A = np.random.default_rng(72).integers(0, 100, size=widths, dtype=np.int32)
        sdfg = _build_handwritten_sdfg(widths, '+', None, False, dace.int32, 'hw_i32_sum')
        B = np.zeros((1,), dtype=np.int32)
        result = _run_cutile(sdfg, A=A, B=B)
        np.testing.assert_array_equal(result['B'], [np.sum(A)])

    def test_int32_min(self):
        """int32 min reduction."""
        widths = (8,)
        A = np.random.default_rng(73).integers(-100, 100, size=widths, dtype=np.int32)
        sdfg = _build_handwritten_sdfg(widths, 'min', None, False, dace.int32, 'hw_i32_min')
        B = np.zeros((1,), dtype=np.int32)
        result = _run_cutile(sdfg, A=A, B=B)
        np.testing.assert_array_equal(result['B'], [np.min(A)])

    def test_int32_max(self):
        """int32 max reduction."""
        widths = (8,)
        A = np.random.default_rng(74).integers(-100, 100, size=widths, dtype=np.int32)
        sdfg = _build_handwritten_sdfg(widths, 'max', None, False, dace.int32, 'hw_i32_max')
        B = np.zeros((1,), dtype=np.int32)
        result = _run_cutile(sdfg, A=A, B=B)
        np.testing.assert_array_equal(result['B'], [np.max(A)])

    def test_float32_min_masked(self):
        """float32 masked min reduction."""
        widths = (8,)
        A = np.random.default_rng(75).random(widths).astype(np.float32)
        M = np.array([True, False, True, True, False, True, False, True],
                     dtype=np.bool_)
        sdfg = _build_handwritten_sdfg(widths, 'min', None, True, dace.float32, 'hw_f32_min_mask')
        B = np.zeros((1,), dtype=np.float32)
        result = _run_cutile(sdfg, A=A, M=M, B=B)
        np.testing.assert_allclose(result['B'], [np.min(A[M])], rtol=1e-5)

    def test_float64_max_masked(self):
        """float64 masked max reduction."""
        widths = (8,)
        A = np.random.default_rng(76).random(widths)
        M = np.array([True, False, True, True, False, True, False, True],
                     dtype=np.bool_)
        sdfg = _build_handwritten_sdfg(widths, 'max', None, True, dace.float64, 'hw_f64_max_mask')
        B = np.zeros((1,), dtype=np.float64)
        result = _run_cutile(sdfg, A=A, M=M, B=B)
        np.testing.assert_allclose(result['B'], [np.max(A[M])], rtol=1e-12)


# ============================================================
# Pattern B: Expansion-based (TileReduce lib node -> expand -> run)
# ============================================================


class TestTileReduceCutileExpansion:
    """Full expansion path: TileReduce lib node -> expand -> cuTile codegen -> GPU runtime."""

    def test_expansion_k1_sum_full(self):
        """K=1 full sum via expansion."""
        widths = (8,)
        sdfg = _build_expansion_sdfg(widths, '+', None, False, dace.float64, 'exp_k1_sum')
        A = np.random.default_rng(80).random(widths)
        B = np.zeros((1,), dtype=np.float64)
        result = _run_cutile(sdfg, A=A, B=B)
        np.testing.assert_allclose(result['B'], [np.sum(A)], rtol=1e-12)

    def test_expansion_k1_sum_masked(self):
        """K=1 masked sum via expansion."""
        widths = (8,)
        sdfg = _build_expansion_sdfg(widths, '+', None, True, dace.float64, 'exp_k1_sum_m')
        A = np.random.default_rng(81).random(widths)
        M = np.array([True, False, True, True, False, True, False, True],
                     dtype=np.bool_)
        B = np.zeros((1,), dtype=np.float64)
        result = _run_cutile(sdfg, A=A, M=M, B=B)
        np.testing.assert_allclose(result['B'], [np.sum(A[M])], rtol=1e-12)

    def test_expansion_k1_min_masked(self):
        """Regression: K=1 masked min via expansion (identity was broken)."""
        widths = (8,)
        sdfg = _build_expansion_sdfg(widths, 'min', None, True, dace.float64, 'exp_k1_min_m')
        A = np.random.default_rng(82).random(widths)
        M = np.array([True, False, True, True, False, True, False, True],
                     dtype=np.bool_)
        B = np.zeros((1,), dtype=np.float64)
        result = _run_cutile(sdfg, A=A, M=M, B=B)
        np.testing.assert_allclose(result['B'], [np.min(A[M])], rtol=1e-12)

    def test_expansion_k1_max_masked(self):
        """Regression: K=1 masked max via expansion (identity was broken)."""
        widths = (8,)
        sdfg = _build_expansion_sdfg(widths, 'max', None, True, dace.float64, 'exp_k1_max_m')
        A = np.random.default_rng(83).random(widths)
        M = np.array([True, False, True, True, False, True, False, True],
                     dtype=np.bool_)
        B = np.zeros((1,), dtype=np.float64)
        result = _run_cutile(sdfg, A=A, M=M, B=B)
        np.testing.assert_allclose(result['B'], [np.max(A[M])], rtol=1e-12)

    def test_expansion_k2_sum_axis0(self):
        """K=2 sum along axis 0 via expansion."""
        widths = (4, 8)
        sdfg = _build_expansion_sdfg(widths, '+', 0, False, dace.float64, 'exp_k2_sum_ax0')
        A = np.random.default_rng(84).random(widths)
        B = np.zeros((8,), dtype=np.float64)
        result = _run_cutile(sdfg, A=A, B=B)
        np.testing.assert_allclose(result['B'], np.sum(A, axis=0), rtol=1e-12)

    def test_expansion_k2_prod_full(self):
        """K=2 full product via expansion (narrow range)."""
        widths = (4, 8)
        sdfg = _build_expansion_sdfg(widths, '*', None, False, dace.float64, 'exp_k2_prod')
        A = np.random.default_rng(85).random(widths) * 0.5 + 0.75
        B = np.zeros((1,), dtype=np.float64)
        result = _run_cutile(sdfg, A=A, B=B)
        np.testing.assert_allclose(result['B'], [np.prod(A)], rtol=1e-10)

    def test_expansion_k1_sum_float32(self):
        """K=1 float32 sum via expansion."""
        widths = (8,)
        sdfg = _build_expansion_sdfg(widths, '+', None, False, dace.float32, 'exp_k1_sum_f32')
        A = np.random.default_rng(86).random(widths).astype(np.float32)
        B = np.zeros((1,), dtype=np.float32)
        result = _run_cutile(sdfg, A=A, B=B)
        np.testing.assert_allclose(result['B'], [np.sum(A)], rtol=1e-5)

    def test_expansion_k1_min_int32_masked(self):
        """Integer min with identity = INT32_MAX via expansion."""
        widths = (8,)
        sdfg = _build_expansion_sdfg(widths, 'min', None, True, dace.int32, 'exp_i32_min_m')
        A = np.random.default_rng(87).integers(-100, 100, size=widths, dtype=np.int32)
        M = np.array([True, False, True, True, False, True, False, True],
                     dtype=np.bool_)
        B = np.zeros((1,), dtype=np.int32)
        result = _run_cutile(sdfg, A=A, M=M, B=B)
        np.testing.assert_array_equal(result['B'], [np.min(A[M])])

    def test_expansion_k2_sum_axis1(self):
        """K=2 sum along axis 1 via expansion."""
        widths = (4, 8)
        sdfg = _build_expansion_sdfg(widths, '+', 1, False, dace.float64, 'exp_k2_sum_ax1')
        A = np.random.default_rng(88).random(widths)
        B = np.zeros((4,), dtype=np.float64)
        result = _run_cutile(sdfg, A=A, B=B)
        np.testing.assert_allclose(result['B'], np.sum(A, axis=1), rtol=1e-12)

    def test_expansion_k2_max_axis0_masked(self):
        """K=2 masked max along axis 0 via expansion."""
        widths = (4, 8)
        rng = np.random.default_rng(89)
        A = rng.random(widths)
        M = rng.choice([True, False], size=widths)
        sdfg = _build_expansion_sdfg(widths, 'max', 0, True, dace.float64, 'exp_k2_max_ax0_m')
        B = np.zeros((8,), dtype=np.float64)
        result = _run_cutile(sdfg, A=A, M=M, B=B)
        expected = np.max(np.where(M, A, -np.inf), axis=0)
        np.testing.assert_allclose(result['B'], expected, rtol=1e-12)

    def test_expansion_k1_prod_masked(self):
        """K=1 masked product via expansion."""
        widths = (8,)
        rng = np.random.default_rng(90)
        A = rng.random(widths) + 0.5
        M = np.array([True, False, True, True, False, True, False, True],
                     dtype=np.bool_)
        sdfg = _build_expansion_sdfg(widths, '*', None, True, dace.float64, 'exp_k1_prod_m')
        B = np.zeros((1,), dtype=np.float64)
        result = _run_cutile(sdfg, A=A, M=M, B=B)
        np.testing.assert_allclose(result['B'], [np.prod(A[M])], rtol=1e-10)


# ============================================================
# Identity literal unit tests (no GPU needed -- but module
# pytestmark applies, so these run with the gpu marker)
# ============================================================


class TestTileReduceCutileIdentityLiterals:
    """Unit tests for _identity_literal_cutile (no GPU needed)."""

    def test_identity_sum_float64(self):
        assert _identity_literal_cutile('+', dace.float64) == "ct.astype(0, ct.float64)"

    def test_identity_prod_float32(self):
        assert _identity_literal_cutile('*', dace.float32) == "ct.astype(1, ct.float32)"

    def test_identity_min_float64(self):
        assert _identity_literal_cutile('min', dace.float64) == "ct.astype(float('inf'), ct.float64)"

    def test_identity_max_float64(self):
        assert _identity_literal_cutile('max', dace.float64) == "ct.astype(float('-inf'), ct.float64)"

    def test_identity_min_int32(self):
        assert _identity_literal_cutile('min', dace.int32) == "ct.astype(2147483647, ct.int32)"

    def test_identity_max_int32(self):
        assert _identity_literal_cutile('max', dace.int32) == "ct.astype(-2147483648, ct.int32)"

    def test_identity_min_uint32(self):
        assert _identity_literal_cutile('min', dace.uint32) == "ct.astype(4294967295, ct.uint32)"

    def test_identity_max_uint32(self):
        assert _identity_literal_cutile('max', dace.uint32) == "ct.astype(0, ct.uint32)"

    def test_identity_unknown_op_raises(self):
        with pytest.raises(NotImplementedError):
            _identity_literal_cutile('^', dace.float64)

    def test_dace_dtype_to_cutile_str_mapping(self):
        """Verify the dtype-to-cuTile-string mapping for common types."""
        assert _dace_dtype_to_cutile_str(dace.float64) == "ct.float64"
        assert _dace_dtype_to_cutile_str(dace.float32) == "ct.float32"
        assert _dace_dtype_to_cutile_str(dace.int32) == "ct.int32"
        assert _dace_dtype_to_cutile_str(dace.bool) == "ct.bool_"


# ============================================================
# Entry point
# ============================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=300"])
