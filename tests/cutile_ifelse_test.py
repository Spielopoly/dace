"""
Tests for the cuTile if-else → TileWhereSelect transformation.

Tests ``IfElseMapToTileWhere`` which converts tiled maps containing
NestedSDFGs with if-else ``ConditionalBlock`` patterns into tile-level
unconditional TileOp branches + a ``TileWhereSelectLibraryNode``.
"""

import math

import numpy as np
import pytest

import dace
from dace.sdfg import nodes
from dace.transformation.dataflow import TrivialChainElimination, MapTiling
from dace.libraries.cutile.transformations.if_else_to_where_select import (
    IfElseMapToTileWhere,
)
from dace.libraries.cutile.nodes.where_select import TileWhereSelectLibraryNode
from dace.libraries.cutile.nodes.op import TileOpLibraryNode
from dace.libraries.cutile.transformations.pipeline import apply_cutile_pipeline


# ---------------------------------------------------------------------------
# DaCe programs under test
# ---------------------------------------------------------------------------

SZ = dace.symbol("SZ", dtype=dace.int32)


@dace.program
def if_else_add_constant(A: dace.float64[SZ, SZ], B: dace.float64[SZ, SZ]):
    for i, j in dace.map[0:SZ, 0:SZ]:
        if A[i, j] > 0:
            B[i, j] = B[i, j] - 1.0
        else:
            B[i, j] = B[i, j] + 1.0


@dace.program
def if_else_add(
    A: dace.float64[SZ, SZ],
    B: dace.float64[SZ, SZ],
    C: dace.float64[SZ, SZ],
):
    for i, j in dace.map[0:SZ, 0:SZ]:
        if A[i, j] > 0:
            C[i, j] = C[i, j] + B[i, j]
        else:
            C[i, j] = C[i, j] - B[i, j]


# ── Two-array comparison (A < B), write-only output, different arrays ──

@dace.program
def if_else_lt_add_sub(
    A: dace.float64[SZ, SZ],
    B: dace.float64[SZ, SZ],
    C: dace.float64[SZ, SZ],
    D: dace.float64[SZ, SZ],
    E: dace.float64[SZ, SZ],
):
    for i, j in dace.map[0:SZ, 0:SZ]:
        if A[i, j] < B[i, j]:
            C[i, j] = D[i, j] + E[i, j]
        else:
            C[i, j] = D[i, j] - E[i, j]


# ── Two-array comparison (A < B), output overwrites condition input ──

@dace.program
def if_else_lt_self_write(
    A: dace.float64[SZ, SZ],
    B: dace.float64[SZ, SZ],
):
    for i, j in dace.map[0:SZ, 0:SZ]:
        if A[i, j] < B[i, j]:
            A[i, j] = 2.0 * A[i, j]
        else:
            A[i, j] = A[i, j] + 1.0


# ── Two-array comparison, binary + unary (sin) in branches ──

@dace.program
def if_else_lt_sin(
    A: dace.float64[SZ, SZ],
    B: dace.float64[SZ, SZ],
    C: dace.float64[SZ, SZ],
    D: dace.float64[SZ, SZ],
):
    for i, j in dace.map[0:SZ, 0:SZ]:
        if A[i, j] < B[i, j]:
            C[i, j] = D[i, j] + A[i, j]
        else:
            C[i, j] = math.sin(D[i, j])


# ── >= comparison against constant, mul vs abs ──

@dace.program
def if_else_ge_mul_abs(
    A: dace.float64[SZ, SZ],
    B: dace.float64[SZ, SZ],
):
    for i, j in dace.map[0:SZ, 0:SZ]:
        if A[i, j] >= 0:
            B[i, j] = A[i, j] * A[i, j]
        else:
            B[i, j] = abs(A[i, j])


# ── <= comparison, constant factor in both branches ──

@dace.program
def if_else_le_const_ops(
    A: dace.float64[SZ, SZ],
    B: dace.float64[SZ, SZ],
):
    for i, j in dace.map[0:SZ, 0:SZ]:
        if A[i, j] <= 0:
            B[i, j] = A[i, j] * 3.0
        else:
            B[i, j] = A[i, j] / 2.0


# ── != comparison, negation in true branch, add constant in false ──

@dace.program
def if_else_ne_neg(
    A: dace.float64[SZ, SZ],
    B: dace.float64[SZ, SZ],
):
    for i, j in dace.map[0:SZ, 0:SZ]:
        if A[i, j] != 0:
            B[i, j] = -A[i, j]
        else:
            B[i, j] = A[i, j] + 1.0


# ── == comparison on two arrays, add vs mul ──

@dace.program
def if_else_eq_add_mul(
    A: dace.float64[SZ, SZ],
    B: dace.float64[SZ, SZ],
    C: dace.float64[SZ, SZ],
):
    for i, j in dace.map[0:SZ, 0:SZ]:
        if A[i, j] == B[i, j]:
            C[i, j] = A[i, j] + B[i, j]
        else:
            C[i, j] = A[i, j] * B[i, j]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tile_and_transform(program, *, tile_sizes=(16, 16)):
    """Parse → TrivialChainElimination → Tiling → IfElseMapToTileWhere."""
    sdfg = program.to_sdfg()
    sdfg.apply_transformations_once_everywhere(TrivialChainElimination)
    sdfg.apply_transformations(
        [MapTiling], options={"tile_sizes": list(tile_sizes), "skew": True}
    )
    n_applied = sdfg.apply_transformations_once_everywhere(IfElseMapToTileWhere)
    return sdfg, n_applied


def _collect_lib_nodes(sdfg, cls):
    """Return all library nodes of a given type across all states."""
    return [
        n
        for state in sdfg.states()
        for n in state.nodes()
        if isinstance(n, cls)
    ]


# ---------------------------------------------------------------------------
# Structure / applicability tests
# ---------------------------------------------------------------------------

class TestIfElseStructure:
    """Verify the transformation applies and produces the expected graph."""

    def test_constant_branches_apply(self):
        sdfg, n = _tile_and_transform(if_else_add_constant)
        assert n == 1, f"Expected 1 application, got {n}"

    def test_array_branches_apply(self):
        sdfg, n = _tile_and_transform(if_else_add)
        assert n == 1, f"Expected 1 application, got {n}"

    def test_produces_tile_ops(self):
        sdfg, _ = _tile_and_transform(if_else_add_constant)
        tile_ops = _collect_lib_nodes(sdfg, TileOpLibraryNode)
        assert len(tile_ops) == 3, (
            f"Expected 3 TileOps (condition + true + false), got {len(tile_ops)}"
        )

    def test_produces_where_select(self):
        sdfg, _ = _tile_and_transform(if_else_add_constant)
        ws = _collect_lib_nodes(sdfg, TileWhereSelectLibraryNode)
        assert len(ws) == 1, (
            f"Expected 1 TileWhereSelect, got {len(ws)}"
        )

    def test_no_nested_sdfg_remains(self):
        sdfg, _ = _tile_and_transform(if_else_add_constant)
        nsdfgs = [
            n
            for state in sdfg.states()
            for n in state.nodes()
            if isinstance(n, nodes.NestedSDFG)
        ]
        assert len(nsdfgs) == 0, "NestedSDFG should be removed after transformation"

    def test_validates_after_transform(self):
        sdfg, _ = _tile_and_transform(if_else_add_constant)
        sdfg.validate()

    def test_validates_after_expansion(self):
        sdfg, _ = _tile_and_transform(if_else_add_constant)
        sdfg.expand_library_nodes()
        sdfg.validate()

    @pytest.mark.parametrize("program", [
        if_else_lt_add_sub,
        if_else_lt_self_write,
        if_else_lt_sin,
        if_else_ge_mul_abs,
        if_else_le_const_ops,
        if_else_ne_neg,
        if_else_eq_add_mul,
    ], ids=lambda p: p.name)
    def test_new_conditions_apply(self, program):
        sdfg, n = _tile_and_transform(program)
        assert n == 1, f"Expected 1 application, got {n}"

    @pytest.mark.parametrize("program", [
        if_else_lt_add_sub,
        if_else_lt_self_write,
        if_else_lt_sin,
        if_else_ge_mul_abs,
        if_else_le_const_ops,
        if_else_ne_neg,
        if_else_eq_add_mul,
    ], ids=lambda p: p.name)
    def test_new_conditions_validate(self, program):
        sdfg, _ = _tile_and_transform(program)
        sdfg.expand_library_nodes()
        sdfg.validate()


# ---------------------------------------------------------------------------
# Numeric correctness tests
# ---------------------------------------------------------------------------

class TestIfElseCorrectness:
    """End-to-end correctness: compile + run + compare to numpy."""

    @pytest.mark.parametrize("n", [32, 37, 5])
    def test_constant_branches(self, n):
        sdfg, _ = _tile_and_transform(if_else_add_constant)
        sdfg.expand_library_nodes()
        compiled = sdfg.compile()

        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, n))
        B = rng.standard_normal((n, n))
        B_copy = B.copy()
        expected = np.where(A > 0, B_copy - 1.0, B_copy + 1.0)

        compiled(A=A, B=B, SZ=n)
        np.testing.assert_allclose(B, expected)

    @pytest.mark.parametrize("n", [32, 33, 8])
    def test_array_branches(self, n):
        sdfg, _ = _tile_and_transform(if_else_add)
        sdfg.expand_library_nodes()
        compiled = sdfg.compile()

        rng = np.random.default_rng(99)
        A = rng.standard_normal((n, n))
        B = rng.standard_normal((n, n))
        C = rng.standard_normal((n, n))
        C_copy = C.copy()
        expected = np.where(A > 0, C_copy + B, C_copy - B)

        compiled(A=A, B=B, C=C, SZ=n)
        np.testing.assert_allclose(C, expected)


# ---------------------------------------------------------------------------
# Extended correctness tests – various conditions and branch ops
# ---------------------------------------------------------------------------

class TestIfElseConditionVariants:
    """
    Correctness tests for different comparison operators, branch operations,
    and array access patterns in the if-else transformation.
    """

    @pytest.mark.parametrize("n", [32, 35])
    def test_lt_two_array_cond_write_only_output(self, n):
        """if A < B:  C = D + E  else:  C = D - E  (C is write-only)."""
        sdfg, _ = _tile_and_transform(if_else_lt_add_sub)
        sdfg.expand_library_nodes()
        compiled = sdfg.compile()

        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, n))
        B = rng.standard_normal((n, n))
        C = np.zeros((n, n))
        D = rng.standard_normal((n, n))
        E = rng.standard_normal((n, n))
        expected = np.where(A < B, D + E, D - E)

        compiled(A=A, B=B, C=C, D=D, E=E, SZ=n)
        np.testing.assert_allclose(C, expected)

    @pytest.mark.parametrize("n", [32, 35])
    def test_lt_self_write(self, n):
        """if A < B:  A = 2*A  else:  A = A + 1  (output = condition input)."""
        sdfg, _ = _tile_and_transform(if_else_lt_self_write)
        sdfg.expand_library_nodes()
        compiled = sdfg.compile()

        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, n))
        B = rng.standard_normal((n, n))
        A_copy = A.copy()
        expected = np.where(A_copy < B, 2.0 * A_copy, A_copy + 1.0)

        compiled(A=A, B=B, SZ=n)
        np.testing.assert_allclose(A, expected)

    @pytest.mark.parametrize("n", [32, 35])
    def test_lt_binary_vs_unary_sin(self, n):
        """if A < B:  C = D + A  else:  C = sin(D)  (binary vs unary)."""
        sdfg, _ = _tile_and_transform(if_else_lt_sin)
        sdfg.expand_library_nodes()
        compiled = sdfg.compile()

        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, n))
        B = rng.standard_normal((n, n))
        C = np.zeros((n, n))
        D = rng.standard_normal((n, n))
        expected = np.where(A < B, D + A, np.sin(D))

        compiled(A=A, B=B, C=C, D=D, SZ=n)
        np.testing.assert_allclose(C, expected)

    @pytest.mark.parametrize("n", [32, 35])
    def test_ge_const_cond_mul_vs_abs(self, n):
        """if A >= 0:  B = A * A  else:  B = abs(A)."""
        sdfg, _ = _tile_and_transform(if_else_ge_mul_abs)
        sdfg.expand_library_nodes()
        compiled = sdfg.compile()

        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, n))
        B = np.zeros((n, n))
        expected = np.where(A >= 0, A * A, np.abs(A))

        compiled(A=A, B=B, SZ=n)
        np.testing.assert_allclose(B, expected)

    @pytest.mark.parametrize("n", [32, 35])
    def test_le_const_cond_const_ops(self, n):
        """if A <= 0:  B = A * 3  else:  B = A / 2."""
        sdfg, _ = _tile_and_transform(if_else_le_const_ops)
        sdfg.expand_library_nodes()
        compiled = sdfg.compile()

        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, n))
        B = np.zeros((n, n))
        expected = np.where(A <= 0, A * 3.0, A / 2.0)

        compiled(A=A, B=B, SZ=n)
        np.testing.assert_allclose(B, expected)

    @pytest.mark.parametrize("n", [32, 35])
    def test_ne_const_cond_negation(self, n):
        """if A != 0:  B = -A  else:  B = A + 1."""
        sdfg, _ = _tile_and_transform(if_else_ne_neg)
        sdfg.expand_library_nodes()
        compiled = sdfg.compile()

        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, n))
        B = np.zeros((n, n))
        expected = np.where(A != 0, -A, A + 1.0)

        compiled(A=A, B=B, SZ=n)
        np.testing.assert_allclose(B, expected)

    @pytest.mark.parametrize("n", [32, 35])
    def test_eq_two_array_cond_add_vs_mul(self, n):
        """if A == B:  C = A + B  else:  C = A * B."""
        sdfg, _ = _tile_and_transform(if_else_eq_add_mul)
        sdfg.expand_library_nodes()
        compiled = sdfg.compile()

        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, n))
        B = rng.standard_normal((n, n))
        C = np.zeros((n, n))
        expected = np.where(A == B, A + B, A * B)

        compiled(A=A, B=B, C=C, SZ=n)
        np.testing.assert_allclose(C, expected)


# ---------------------------------------------------------------------------
# Pipeline integration test
# ---------------------------------------------------------------------------

class TestIfElsePipeline:
    """Verify that the pipeline applies the if-else transformation."""

    def test_pipeline_applies_if_else(self):
        sdfg = if_else_add_constant.to_sdfg()
        sdfg.apply_transformations_once_everywhere(TrivialChainElimination)
        count = apply_cutile_pipeline(sdfg, apply_map_tiling=True)
        assert count > 0

        ws = _collect_lib_nodes(sdfg, TileWhereSelectLibraryNode)
        assert len(ws) == 1

    def test_pipeline_numeric_correctness(self):
        sdfg = if_else_add_constant.to_sdfg()
        sdfg.apply_transformations_once_everywhere(TrivialChainElimination)
        apply_cutile_pipeline(sdfg, apply_map_tiling=True)
        sdfg.expand_library_nodes()
        compiled = sdfg.compile()

        n = 40
        rng = np.random.default_rng(7)
        A = rng.standard_normal((n, n))
        B = rng.standard_normal((n, n))
        B_copy = B.copy()
        expected = np.where(A > 0, B_copy - 1.0, B_copy + 1.0)

        compiled(A=A, B=B, SZ=n)
        np.testing.assert_allclose(B, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
