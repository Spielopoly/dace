# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Integration tests for the ``VectorizeCuTile`` public API after the config redesign.

Covers the ``device`` knob (GPU-first default order vs the legacy CPU order),
the assumption-guard regression (no CPP tasklets in a Python-backend SDFG),
and the ``ISA.CUTILE`` save/load round-trip. GPU-executing tests are marked
``@pytest.mark.gpu``.
"""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.sdfg import SDFG, nodes
from dace.transformation.passes.vectorization import VectorizeCuTile
from dace.transformation.passes.vectorization.cutile_lowering import _collect_tile_nodes
from dace.transformation.passes.vectorization.enums import ISA

N = dace.symbol("N")
M = dace.symbol("M")


@dace.program
def _vadd(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N]:
        c[i] = a[i] + b[i]


@dace.program
def _row_sum(A: dace.float64[M, N], y: dace.float64[M]):
    for i in dace.map[0:M]:
        s = 0.0
        for j in range(N):
            s = s + A[i, j]
        y[i] = s


def _vadd_sdfg(name: str) -> SDFG:
    sdfg = _vadd.to_sdfg()
    sdfg.name = name
    return sdfg


def _cpp_tasklets(sdfg: SDFG) -> list:
    return [
        node for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.Tasklet) and node.language == dtypes.Language.CPP
    ]


def _gpu_device_maps(sdfg: SDFG) -> list:
    return [
        node for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_Device
    ]


@pytest.mark.gpu
@pytest.mark.parametrize("n", [64, 100])
def test_gpu_first_default_vadd(n):
    """Default (device=GPU) order: anchors, a CuTile kernel, correct numerics.

    ``n=100`` with ``widths=(8,)`` exercises the non-divisible masked boundary.
    """
    sdfg = _vadd_sdfg(f"cutile_knob_gpu_vadd_{n}")
    num_kernels = VectorizeCuTile(widths=(8, )).apply_pass(sdfg, {})
    assert num_kernels is not None and num_kernels >= 1
    assert sdfg.backend == dtypes.BackendLanguage.Python
    a = np.random.rand(n)
    b = np.random.rand(n)
    c = np.zeros(n)
    sdfg(a=a, b=b, c=c, N=n)
    np.testing.assert_allclose(c, a + b)


@pytest.mark.gpu
def test_gpu_first_lifted_reduction():
    """A row-sum (atax-like) reduction runs end-to-end under device=GPU.

    Covers the GPU reduction finalize (``_gpu_place_reductions``) interplay
    with ``CuTileSetLibraryImplementations``: any lifted ``Reduce`` / BLAS
    node must be re-stamped and expanded so nothing GPU_Device-scheduled
    survives to the Python backend. (``n=100`` with ``widths=(8,)`` covers
    the non-divisible masked boundary.)
    """
    sdfg = _row_sum.to_sdfg()
    sdfg.name = "cutile_knob_gpu_rowsum"
    num_kernels = VectorizeCuTile(widths=(8, )).apply_pass(sdfg, {})
    assert num_kernels is not None and num_kernels >= 1
    assert sdfg.backend == dtypes.BackendLanguage.Python
    # The Python backend has no GPU_Device dispatcher; none may survive.
    assert not _gpu_device_maps(sdfg)
    m, n = 24, 100
    A = np.random.rand(m, n)
    y = np.zeros(m)
    sdfg(A=A, y=y, M=m, N=n)
    np.testing.assert_allclose(y, A.sum(axis=1))


def test_no_cpp_assumption_guard_tasklets():
    """Symbolic-size kernel: the pipeline must not emit CPP trap tasklets.

    Regression check for the vectorizer's ``insert_assumption_guards`` stage
    (a CPP ``__builtin_trap`` tasklet the Python backend cannot codegen);
    the cuTile config disables it via ``assumption_guard=False``.
    """
    sdfg = _vadd_sdfg("cutile_knob_no_cpp_guard")
    VectorizeCuTile(widths=(8, )).apply_pass(sdfg, {})
    assert not _cpp_tasklets(sdfg)


def test_isa_cutile_save_load_roundtrip(tmp_path):
    """Tile nodes stamped ``target_isa=ISA.CUTILE`` survive a save/load cycle."""
    sdfg = _vadd_sdfg("cutile_knob_roundtrip")
    VectorizeCuTile(widths=(8, )).apply_pass(sdfg, {})
    anchors = _collect_tile_nodes(sdfg)
    assert anchors
    assert all(node.target_isa == ISA.CUTILE for node, _ in anchors)
    assert all(node.implementation == "cutile" for node, _ in anchors)

    path = str(tmp_path / "roundtrip.sdfg")
    sdfg.save(path)
    loaded = SDFG.from_file(path)
    loaded_anchors = _collect_tile_nodes(loaded)
    assert len(loaded_anchors) == len(anchors)
    # ISA is a str-enum: the reloaded value must compare equal to the member.
    assert all(node.target_isa == "CUTILE" for node, _ in loaded_anchors)
    assert all(node.implementation == "cutile" for node, _ in loaded_anchors)


if __name__ == "__main__":
    import pathlib
    import tempfile
    test_gpu_first_default_vadd(100)
    test_cpu_legacy_order_vadd(100)
    test_gpu_first_lifted_reduction()
    test_no_cpp_assumption_guard_tasklets()
    with tempfile.TemporaryDirectory() as td:
        test_isa_cutile_save_load_roundtrip(pathlib.Path(td))
