# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Caller-pre-scheduled entry into ``VectorizeCuTile``.

The ``VectorizeCuTile`` GPU-first order skips its own
``apply_gpu_transformations`` (and the trivial-wrapper cleanup) when the caller
already GPU-scheduled the SDFG. A free tasklet (``B[0, 0] = 3.0`` outside any
map) then arrives wrapped in the trivial ``0:1`` ``*_gmap`` kernel map
``GPUTransformSDFG`` mints, and before the shared-gate refusal in
``is_tile_eligible`` the pipeline crashed on it: ``NotImplementedError:
MarkTileDims ... has only 1 params (< K=2)`` at K=2, and an invalid SDFG
(isolated ``_tile_iter_mask``) at K=1. These tests pin the entry path:
structure without a GPU, numerics (symbolic sizes, non-divisible boundary)
on one.
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.sdfg import SDFG, nodes
from dace.transformation.passes.vectorization.vectorize_cutile import VectorizeCuTile


def _prescheduled_sdfg() -> SDFG:
    """Canonicalize + GPU-schedule a kernel with a free tasklet, as a caller would.

    :returns: A GPU-scheduled (not yet vectorized) SDFG containing a trivial
        ``0:1`` wrapper map around the free ``B[0, 0] = 3.0`` tasklet.
    """
    N = dace.symbol("N", dtype=dace.int64)
    M = dace.symbol("M", dtype=dace.int64)

    @dace.program
    def scale_with_seed(A: dace.float64[N, M], B: dace.float64[N, M]):
        for i, j in dace.map[0:N, 0:M]:
            B[i, j] = A[i, j] * 2.0
        B[0, 0] = 3.0

    sdfg = scale_with_seed.to_sdfg(simplify=True)
    VectorizeCuTile.canonicalize_for_cutile(sdfg)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.apply_gpu_transformations(validate=False, sequential_innermaps=True, register_transients=True, simplify=False)
    return sdfg


def _cutile_maps(sdfg: SDFG):
    """Every map scheduled ``CuTile`` after lowering (NestedSDFGs included)."""
    return [
        node for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.CuTile
    ]


@pytest.mark.parametrize("widths", [(8, ), (8, 8)], ids=["k1", "k2"])
def test_prescheduled_entry_lowers(widths):
    """The pre-scheduled entry path completes and produces a CuTile kernel.

    Crashed before the shared-gate refusal (K=2: ``MarkTileDims ... only 1
    params``; K=1: isolated ``_tile_iter_mask``). Structural check, no GPU.
    """
    sdfg = _prescheduled_sdfg()
    kernels = VectorizeCuTile(widths=widths, run_canonicalize=False).apply_pass(sdfg, {})
    assert kernels is not None and kernels >= 1
    assert len(_cutile_maps(sdfg)) == kernels
    assert sdfg.backend == dtypes.BackendLanguage.Python
    sdfg.validate()


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(16, 32), (13, 21)], ids=["divisible", "remainder"])
def test_prescheduled_entry_runtime(shape):
    """Compile + run the pre-scheduled entry on GPU and compare against NumPy.

    Symbolic sizes; ``(13, 21)`` with widths ``(8, 8)`` covers non-divisible
    boundaries in both dims. The free tasklet's seed write (``B[0, 0] = 3.0``)
    lands AFTER the map, so the wrapper-kernel path is numerically pinned:
    the reference is ``2 * A`` with ``ref[0, 0] = 3.0``.
    """
    sdfg = _prescheduled_sdfg()
    VectorizeCuTile(widths=(8, 8), run_canonicalize=False).apply_pass(sdfg, {})
    csdfg = sdfg.compile()

    n, m = shape
    rng = np.random.default_rng(0)
    A = rng.random((n, m))
    B = np.zeros((n, m))
    csdfg(A=A, B=B, N=n, M=m)
    ref = 2 * A
    ref[0, 0] = 3.0
    assert np.allclose(B, ref), f"max diff = {np.max(np.abs(B - ref))}"


if __name__ == "__main__":
    test_prescheduled_entry_lowers((8, ))
    test_prescheduled_entry_lowers((8, 8))
    test_prescheduled_entry_runtime((16, 32))
    test_prescheduled_entry_runtime((13, 21))
    print("ok")
