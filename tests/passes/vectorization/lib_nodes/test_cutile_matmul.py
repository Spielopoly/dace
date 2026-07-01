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


@pytest.mark.gpu
def test_matmul_cutile_runtime_validates():
    """Compile+run a matmul kernel on GPU and validate against NumPy.

    The ``CuTileSetLibraryImplementations`` pass resolves Bug A (the
    previously-fatal ``KeyError: GPU_Device`` is gone and the kernel compiles --
    see the structural test above). This test currently FAILS at execution on a
    separate, unfixed storage-model mismatch: the BLAS 'CuPy' expansion
    round-trips the result through host (``cupy.asnumpy(...)``), which cannot be
    assigned into the GPU_Global (cupy) output.
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


if __name__ == "__main__":
    test_matmul_lowering_leaves_no_blas_libnode_or_gpu_device()
    test_matmul_cutile_runtime_validates()
