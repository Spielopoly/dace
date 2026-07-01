# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression tests for cuTile lowering of BLAS ``MatMul`` library nodes.

``@`` / ``np.matmul`` in a ``@dace.program`` becomes a BLAS ``MatMul`` library
node that the tileops-anchored passes leave untouched. Before
``CuTileSetLibraryImplementations`` was added to the pipeline, such a node kept
the ``GPU_Device`` schedule stamped by ``apply_gpu_transformations()`` and was
never expanded, so the Python/cuTile backend raised
``KeyError: ScheduleType.GPU_Device`` at code generation.

The structural test guards that regression without a GPU; the runtime test
validates numerics on a GPU.
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.sdfg import SDFG, nodes
from dace.transformation.passes.vectorization.cutile_lowering import _collect_non_tile_library_nodes
from dace.transformation.passes.vectorization.vectorize_cutile import VectorizeCuTile


def _gpu_device_scheduled_nodes(sdfg: SDFG):
    """Return every node still scheduled ``GPU_Device`` after lowering.

    :param sdfg: The lowered SDFG to inspect (NestedSDFGs included).
    :returns: List of ``(node, schedule-carrier)`` for diagnostics.
    """
    out = []
    for node, _ in sdfg.all_nodes_recursive():
        carrier = node.map if isinstance(node, nodes.MapEntry) else node
        if getattr(carrier, "schedule", None) == dtypes.ScheduleType.GPU_Device:
            out.append(node)
    return out


def test_matmul_lowering_leaves_no_blas_libnode_or_gpu_device():
    """A matmul kernel lowers with no residual BLAS node / GPU_Device schedule.

    This is the core Bug-A guard: ``CuTileSetLibraryImplementations`` must
    select+expand the ``MatMul`` (via ``specialize`` -> ``Gemv``/``Gemm`` ->
    ``CuPy``) so nothing survives that the Python backend cannot code-generate.
    Pure structural check -- no GPU required.
    """
    M, N = (dace.symbol(s, dtype=dace.int64) for s in ("M", "N"))

    @dace.program
    def atax(A: dace.float64[M, N], x: dace.float64[N]):
        return (A @ x) @ A

    sdfg = atax.to_sdfg(simplify=False)
    VectorizeCuTile(widths=(8, 8)).apply_pass(sdfg, {})

    leftover_blas = _collect_non_tile_library_nodes(sdfg)
    assert not leftover_blas, f"unexpanded non-tileops library nodes: {[type(n).__name__ for n, _ in leftover_blas]}"

    leftover_gpu_device = _gpu_device_scheduled_nodes(sdfg)
    assert not leftover_gpu_device, f"leftover GPU_Device nodes: {leftover_gpu_device}"


def test_matmul_reshaped_contraction_lowers():
    """A reshape-based tensor contraction (doitgen) lowers without error.

    doitgen's ``np.reshape(A[r], (NQ, 1, NP)) @ C4`` specializes ``MatMul`` to
    ``Gemm`` over operands that carry a redundant singleton dimension from the
    reshape. The strict 2-D ``Gemm`` validation used to raise
    ``ValueError: matrix-matrix product only supported on matrices``; the CuPy
    expansion now squeezes the unit dims and lowers cleanly. Pure structural
    check -- no GPU required.
    """
    NR, NQ, NP = (dace.symbol(s, dtype=dace.int64) for s in ("NR", "NQ", "NP"))

    @dace.program
    def doitgen(A: dace.float64[NR, NQ, NP], C4: dace.float64[NP, NP]):
        for r in range(NR):
            A[r, :, :] = np.reshape(np.reshape(A[r], (NQ, 1, NP)) @ C4, (NQ, NP))

    sdfg = doitgen.to_sdfg(simplify=False)
    VectorizeCuTile(widths=(8, 8, 8)).apply_pass(sdfg, {})

    assert not _collect_non_tile_library_nodes(sdfg)
    assert not _gpu_device_scheduled_nodes(sdfg)


@pytest.mark.gpu
def test_matmul_cutile_runtime_validates():
    """Compile+run a matmul kernel on GPU and validate against NumPy.

    Regression for Bug 02: the BLAS 'CuPy' expansion used to round-trip its
    result through host (``cupy.asnumpy(...)``), which cannot be assigned into
    the ``GPU_Global`` (cupy) output the cuTile pipeline places arrays on. The
    expansion is now device-resident when operands live on GPU storage, so the
    matmul stays on the device and the kernel produces correct numerics.
    """
    M, N = (dace.symbol(s, dtype=dace.int64) for s in ("M", "N"))

    @dace.program
    def atax(A: dace.float64[M, N], x: dace.float64[N]):
        return (A @ x) @ A

    sdfg = atax.to_sdfg(simplify=False)
    VectorizeCuTile(widths=(8, 8)).apply_pass(sdfg, {})
    csdfg = sdfg.compile()

    m, n = 40, 48
    rng = np.random.default_rng(0)
    A = rng.random((m, n))
    x = rng.random(n)
    result = csdfg(A=A, x=x, M=m, N=n)

    expected = (A @ x) @ A
    assert np.allclose(np.asarray(result), expected, atol=1e-10)


@pytest.mark.gpu
def test_matmul_cutile_runtime_non_divisible():
    """atax with array sizes that are not multiples of the tile width.

    Exercises remainder handling of the device-resident matmul (Bug 02) on a
    non-divisible boundary (``30 x 20`` with width 8).
    """
    M, N = (dace.symbol(s, dtype=dace.int64) for s in ("M", "N"))

    @dace.program
    def atax(A: dace.float64[M, N], x: dace.float64[N]):
        return (A @ x) @ A

    sdfg = atax.to_sdfg(simplify=False)
    VectorizeCuTile(widths=(8, 8)).apply_pass(sdfg, {})
    csdfg = sdfg.compile()

    m, n = 30, 20
    rng = np.random.default_rng(7)
    A = rng.random((m, n))
    x = rng.random(n)
    result = csdfg(A=A, x=x, M=m, N=n)

    assert np.allclose(np.asarray(result), (A @ x) @ A, atol=1e-9)


@pytest.mark.gpu
@pytest.mark.parametrize("nr,nq,np_", [(6, 8, 8), (5, 10, 8)])
def test_matmul_reshaped_contraction_runtime(nr: int, nq: int, np_: int):
    """Compile+run doitgen's reshape contraction on GPU and validate numerics.

    Regression for Bug 08: the singleton-bearing ``(NQ, 1, NP) @ (NP, NP)``
    product must expand and execute (device-resident) rather than raise at
    ``Gemm`` validation. Covers a divisible and a non-divisible ``NQ``.
    """
    NR, NQ, NP = (dace.symbol(s, dtype=dace.int64) for s in ("NR", "NQ", "NP"))

    @dace.program
    def doitgen(A: dace.float64[NR, NQ, NP], C4: dace.float64[NP, NP]):
        for r in range(NR):
            A[r, :, :] = np.reshape(np.reshape(A[r], (NQ, 1, NP)) @ C4, (NQ, NP))

    sdfg = doitgen.to_sdfg(simplify=False)
    VectorizeCuTile(widths=(8, 8, 8)).apply_pass(sdfg, {})
    csdfg = sdfg.compile()

    rng = np.random.default_rng(11)
    A = rng.random((nr, nq, np_))
    C4 = rng.random((np_, np_))

    expected = A.copy()
    for r in range(nr):
        expected[r] = np.reshape(np.reshape(expected[r], (nq, 1, np_)) @ C4, (nq, np_))

    result = A.copy()
    csdfg(A=result, C4=C4, NR=nr, NQ=nq, NP=np_)

    assert np.allclose(result, expected, atol=1e-9)


if __name__ == "__main__":
    test_matmul_lowering_leaves_no_blas_libnode_or_gpu_device()
    test_matmul_reshaped_contraction_lowers()
    test_matmul_cutile_runtime_validates()
    test_matmul_cutile_runtime_non_divisible()
    test_matmul_reshaped_contraction_runtime(6, 8, 8)
    test_matmul_reshaped_contraction_runtime(5, 10, 8)
