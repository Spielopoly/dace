# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression tests for symbolic body-local containers in ``WidenAccesses``."""

import numpy as np
import pytest

import dace

from dace.memlet import Memlet
from dace.sdfg import propagation
from dace.sdfg.nodes import LibraryNode
from dace.transformation.dataflow import MapCollapse, MapFusion
from dace.transformation.interstate import LoopToMap
from dace.transformation.passes.vectorization.config import VectorizeConfig
from dace.transformation.passes.vectorization.enums import ISA
from dace.transformation.passes.vectorization.utils.pass_invariants import lane_dep_transients_widened
from dace.transformation.passes.vectorization.vectorize_cutile import VectorizeCuTile
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import VectorizeCPUMultiDim
from dace.transformation.passes.vectorization.widen_accesses import WidenAccesses

N_LU = dace.symbol("N_LU", dtype=dace.int64)


@dace.program
def lu_symbolic_view_kernel(A: dace.float64[N_LU, N_LU]):
    """LU update containing the symbolic ``A[i, :j]`` and ``A[:j, j]`` views."""
    for i in range(N_LU):
        for j in range(i):
            A[i, j] -= A[i, :j] @ A[:j, j]
            A[i, j] /= A[j, j]
        for j in range(i, N_LU):
            A[i, j] -= A[i, :i] @ A[:i, j]


def _parallelize_loops(sdfg: dace.SDFG) -> None:
    """Apply the same loop-to-map preparation used by the NPBench parallel track."""
    sdfg.simplify()
    for nested in sdfg.all_sdfgs_recursive():
        propagation.propagate_states(nested)
    applied = 1
    while applied > 0:
        applied = sdfg.apply_transformations_repeated([LoopToMap, MapCollapse])
        sdfg.simplify()
    sdfg.apply_transformations_repeated(MapFusion)
    sdfg.simplify()


def _lu_reference(initial: np.ndarray) -> np.ndarray:
    """Evaluate the small LU fixture with NumPy."""
    result = initial.copy()
    size = result.shape[0]
    for i in range(size):
        for j in range(i):
            result[i, j] -= result[i, :j] @ result[:j, j]
            result[i, j] /= result[j, j]
        for j in range(i, size):
            result[i, j] -= result[i, :i] @ result[:i, j]
    return result


def test_symbolic_body_local_containers_are_not_assumed_lane_dependent() -> None:
    """Only propagated values widen; symbolic scratch buffers and views retain their shapes."""
    n = dace.symbol("N")
    sdfg = dace.SDFG("symbolic_body_locals")
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_array("A", (n, ), dace.float64)
    state = sdfg.add_state("state")
    map_entry, map_exit = state.add_map("tile_map", {"i": "0:N"})

    inner = dace.SDFG("body")
    inner.add_symbol("N", dace.int64)
    inner.add_symbol("i", dace.int64)
    inner.add_array("A", (n, ), dace.float64)
    inner.add_array("scratch", (n, ), dace.float64, transient=True)
    inner.add_view("slice_view", (n, ), dace.float64)
    inner.add_scalar("lane_value", dace.float64, transient=True)
    inner_state = inner.add_state("body")
    read = inner_state.add_access("A")
    lane_value = inner_state.add_access("lane_value")
    tasklet = inner_state.add_tasklet("copy", {"value"}, {"result"}, "result = value")
    inner_state.add_edge(read, None, tasklet, "value", Memlet("A[i]"))
    inner_state.add_edge(tasklet, "result", lane_value, None, Memlet("lane_value[0]"))

    nested = state.add_nested_sdfg(inner, {"A"}, set(), symbol_mapping={"N": "N", "i": "i"})
    outer_read = state.add_access("A")
    state.add_memlet_path(outer_read, map_entry, nested, dst_conn="A", memlet=Memlet("A[0:N]"))
    state.add_nedge(nested, map_exit, Memlet())

    WidenAccesses(widths=(8, )).apply_pass(sdfg, {})

    assert tuple(inner.arrays["scratch"].shape) == (n, )
    assert isinstance(inner.arrays["slice_view"], dace.data.View)
    assert tuple(inner.arrays["slice_view"].shape) == (n, )
    assert tuple(inner.arrays["lane_value"].shape) == (8, )


def test_lane_dependent_transient_invariant_checks_only_classified_names() -> None:
    """Unrelated symbolic locals are accepted while a classified narrow value is rejected."""
    n = dace.symbol("N")
    inner = dace.SDFG("body")
    inner.add_symbol("N", dace.int64)
    inner.add_array("scratch", (n, ), dace.float64, transient=True)
    inner.add_view("slice_view", (n, ), dace.float64)
    inner.add_array("wide", (8, ), dace.float64, transient=True)
    inner.add_array("narrow", (1, ), dace.float64, transient=True)

    assert lane_dep_transients_widened(inner, {"wide"}, (8, )) is None
    violation = lane_dep_transients_widened(inner, {"narrow"}, (8, ))
    assert violation is not None
    assert "narrow" in violation
    assert "shape (1,) != widths (8,)" in violation


def test_lu_symbolic_views_vectorize_and_run_with_pure_backend() -> None:
    """The affected LU shape compiles through tileops ``pure`` and matches NumPy."""
    size = 7
    rng = np.random.default_rng(42)
    initial = rng.random((size, size)) + np.eye(size) * size
    expected = _lu_reference(initial)
    actual = initial.copy()

    vectorized = lu_symbolic_view_kernel.to_sdfg(simplify=False)
    vectorized.name = "lu_symbolic_view_vectorized"
    _parallelize_loops(vectorized)
    VectorizeCPUMultiDim(VectorizeConfig(widths=(4, ), target_isa=ISA.SCALAR,
                                         remainder_strategy="masked_tail")).apply_pass(vectorized, {})

    for node, _ in vectorized.all_nodes_recursive():
        if isinstance(node, LibraryNode) and "pure" in node.implementations:
            node.implementation = "pure"
    vectorized.compile()(A=actual, N_LU=size)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.gpu
def test_lu_symbolic_views_vectorize_and_run_with_cutile() -> None:
    """The affected LU shape lowers through the Python/cuTile backend and matches NumPy."""
    pytest.importorskip("cupy")
    pytest.importorskip("cuda.tile")
    size = 7
    rng = np.random.default_rng(43)
    initial = rng.random((size, size)) + np.eye(size) * size
    expected = _lu_reference(initial)
    actual = initial.copy()

    vectorized = lu_symbolic_view_kernel.to_sdfg(simplify=False)
    vectorized.name = "lu_symbolic_view_cutile"
    _parallelize_loops(vectorized)
    VectorizeCuTile(widths=(4, ), run_canonicalize=False, use_gpu_storage=False).apply_pass(vectorized, {})

    vectorized.compile()(A=actual, N_LU=size)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
