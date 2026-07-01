# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression tests for cuTile launch-grid size codegen (pipeline bug 01).

The grid size (number of tiles) for a cuTile map dimension is
``ceil(extent / step)`` where ``extent = end - start + 1`` (the map range end is
inclusive). It must be emitted as a *structural integer* ceil-division
``int_ceil(extent, step)``.

Historically it was ``symstr(range.size())``, which rewrites a symbolic ceiling
using C integer-division semantics -- e.g. ``ceiling((N-2)/8)`` becomes
``int_ceil(int_floor(N, 8) - 1/4, 1)``. Under Python true division the residual
``1/4`` is the float ``0.25``, so the grid dimension is a non-integer (rejected
by ``ct.launch`` with ``TypeError: an integer is required``) *and* off-by-one for
non-divisible extents (``N=204`` -> 25 tiles instead of 26). These tests pin the
integer/ceil-div behaviour and, on a GPU, the end-to-end correctness.
"""
import numpy as np
import pytest

import dace
from dace import dtypes, subsets
from dace.sdfg import nodes
from dace.codegen.py.cutile_target import _grid_exprs_from_map_entry


def _int_ceil(x, y=1):
    """Reference for the backend's generated ``int_ceil`` (see
    ``dace/codegen/py/sympy_function_redefinitions.py``)."""
    return -(-x // y)


def _map_entry_with_range(rng: subsets.Range) -> nodes.MapEntry:
    """Build a standalone MapEntry carrying the given range."""
    params = [f"i{d}" for d in range(len(rng))]
    return nodes.MapEntry(nodes.Map("m", params, rng))


def _norm(expr: str) -> str:
    """Normalize a grid expression for comparison (drop spaces/parentheses)."""
    return expr.replace(" ", "").replace("(", "").replace(")", "")


class TestGridExprStructural:
    """``_grid_exprs_from_map_entry`` emits integer ceil-div, never a float."""

    def test_symbolic_residual_rational_range(self):
        """Range ``0:N-2:8`` (extent ``N-2``) -> ``int_ceil(N - 2, 8)``.

        This is exactly the jacobi_1d case that produced ``1/4``.
        """
        N = dace.symbol("N", dtype=dace.int64)
        entry = _map_entry_with_range(subsets.Range([(0, N - 3, 8)]))
        exprs = _grid_exprs_from_map_entry(entry)
        assert len(exprs) == 1
        assert _norm(exprs[0]) == "int_ceilN-2,8"
        # No leftover rational / float artifacts from the old symstr path.
        joined = exprs[0]
        assert "int_floor" not in joined
        assert "/4" not in joined and "1/4" not in joined
        assert "." not in joined

    def test_step_one_is_bare_extent(self):
        """A unit step needs no ceil-div: extent is emitted directly."""
        N = dace.symbol("N", dtype=dace.int64)
        entry = _map_entry_with_range(subsets.Range([(1, N - 2, 1)]))
        exprs = _grid_exprs_from_map_entry(entry)
        assert len(exprs) == 1
        assert _norm(exprs[0]) == "N-2"
        assert "int_ceil" not in exprs[0]

    def test_concrete_offbyone_count(self):
        """Non-divisible concrete extent launches the correct tile count.

        Extent 202, step 8 -> ceil(202/8) = 26 (the old path gave 25).
        """
        entry = _map_entry_with_range(subsets.Range([(0, 201, 8)]))
        exprs = _grid_exprs_from_map_entry(entry)
        assert _norm(exprs[0]) == "int_ceil202,8"
        assert eval(exprs[0], {"int_ceil": _int_ceil}) == 26

    def test_multi_dim(self):
        """Per-dimension grid expressions are produced in order."""
        M = dace.symbol("M", dtype=dace.int64)
        entry = _map_entry_with_range(subsets.Range([(0, M - 3, 4), (0, 61, 8)]))
        exprs = _grid_exprs_from_map_entry(entry)
        assert [_norm(e) for e in exprs] == ["int_ceilM-2,4", "int_ceil62,8"]


def _lower_head_sdfg() -> dace.SDFG:
    """Lower a residual-rational elementwise kernel through ``VectorizeCuTile``.

    ``B[:-2] = 2*A[:-2]`` writes from index 0 (no store-offset shift) with a
    tiled extent of ``N-2`` -- isolating the grid-size path.
    """
    from dace.transformation.passes.vectorization import VectorizeCuTile

    N = dace.symbol("N", dtype=dace.int64)

    @dace.program
    def head(A: dace.float64[N], B: dace.float64[N]):
        B[:-2] = 2.0 * A[:-2]

    sdfg = head.to_sdfg(simplify=False)
    VectorizeCuTile(widths=(8, )).apply_pass(sdfg, {})
    return sdfg


def test_lowered_grid_is_integer_ceil_div():
    """After lowering, the cuTile map's grid expr is integer ceil-div (no GPU)."""
    sdfg = _lower_head_sdfg()
    found = False
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.states():
            for node in state.nodes():
                if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.CuTile:
                    exprs = _grid_exprs_from_map_entry(node)
                    for e in exprs:
                        assert "int_floor" not in e, e
                        assert "1/4" not in e and "/4" not in e, e
                        assert e.startswith("int_ceil(") or e[0].isalnum(), e
                        found = True
    assert found, "no CuTile map entry found in lowered SDFG"


@pytest.mark.gpu
@pytest.mark.parametrize("n", [64, 66, 50, 204])
def test_grid_end_to_end(n):
    """End-to-end on GPU: non-divisible sizes match NumPy (grid launches all
    tiles, integer grid accepted by ``ct.launch``)."""
    sdfg = _lower_head_sdfg()
    csdfg = sdfg.compile()
    rng = np.random.default_rng(0)
    A = rng.random(n)
    B = np.full(n, -1.0)
    ref = B.copy()
    ref[:-2] = 2.0 * A[:-2]
    csdfg(A=A, B=B, N=n)
    assert np.allclose(B, ref), f"n={n}: max abs err {np.abs(B - ref).max()}"


if __name__ == "__main__":
    TestGridExprStructural().test_symbolic_residual_rational_range()
    TestGridExprStructural().test_step_one_is_bare_extent()
    TestGridExprStructural().test_concrete_offbyone_count()
    TestGridExprStructural().test_multi_dim()
    test_lowered_grid_is_integer_ceil_div()
    print("non-GPU grid-size tests passed")
