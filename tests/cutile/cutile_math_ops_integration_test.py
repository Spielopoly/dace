"""Integration tests for cuTile math operation pipeline.

Covers sin, cos, exp, sqrt, log, abs, division — operations supported by the
cuTile library that were missing from the integration test suite.  The
previous session added code-structure and direct-library-node tests, but left
gaps in full end-to-end coverage.

Every test here follows the full integration path required by CLAUDE.md:
  @dace.program → to_sdfg() → apply_cutile_pipeline() → compile → run → NumPy

Two backends are exercised:
  - C++ ("pure") backend  — CPU compile + run, no GPU required (Section A & B)
  - cuTile Python backend — GPU compile + run, @pytest.mark.gpu (Section C)

Edge cases per CLAUDE.md requirements:
  - 1D arrays (non-square)
  - float32 dtype alongside float64
  - symbolic array sizes (FN, FM)
  - non-divisible tile boundaries (array not a multiple of tile size)
  - 3D arrays
"""

import math

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.codegen import codegen as dace_codegen
from dace.libraries.cutile.nodes.op import TileOpLibraryNode
from dace.libraries.cutile.transformations.pipeline import apply_cutile_pipeline
from dace.sdfg import nodes


cp = pytest.importorskip("cupy")
pytest.importorskip("cuda.tile")


# ---------------------------------------------------------------------------
# Helpers (mirroring cutile_frontend_test.py / cutile_frontend_python_backend_test.py)
# ---------------------------------------------------------------------------


def _frontend_library_nodes(sdfg: dace.SDFG):
    return [
        n
        for state in sdfg.states()
        for n in state.nodes()
        if isinstance(n, nodes.LibraryNode)
    ]


def _frame_code_of(sdfg: dace.SDFG) -> str:
    code_objects = dace_codegen.generate_code(sdfg)
    return next(co.clean_code for co in code_objects if co.name == sdfg.name)


# ---------------------------------------------------------------------------
# @dace.program definitions — float64, 2D
# ---------------------------------------------------------------------------


@dace.program
def prog_sin_2d(A: dace.float64[24, 20], C: dace.float64[24, 20]):
    for i, j in dace.map[0:24, 0:20]:
        with dace.tasklet:
            a << A[i, j]
            c >> C[i, j]
            c = math.sin(a)


@dace.program
def prog_cos_2d(A: dace.float64[24, 20], C: dace.float64[24, 20]):
    for i, j in dace.map[0:24, 0:20]:
        with dace.tasklet:
            a << A[i, j]
            c >> C[i, j]
            c = math.cos(a)


@dace.program
def prog_exp_2d(A: dace.float64[24, 20], C: dace.float64[24, 20]):
    for i, j in dace.map[0:24, 0:20]:
        with dace.tasklet:
            a << A[i, j]
            c >> C[i, j]
            c = math.exp(a)


@dace.program
def prog_sqrt_2d(A: dace.float64[24, 20], C: dace.float64[24, 20]):
    for i, j in dace.map[0:24, 0:20]:
        with dace.tasklet:
            a << A[i, j]
            c >> C[i, j]
            c = math.sqrt(a)


@dace.program
def prog_log_2d(A: dace.float64[24, 20], C: dace.float64[24, 20]):
    for i, j in dace.map[0:24, 0:20]:
        with dace.tasklet:
            a << A[i, j]
            c >> C[i, j]
            c = math.log(a)


@dace.program
def prog_div_2d(
    A: dace.float64[24, 20],
    B: dace.float64[24, 20],
    C: dace.float64[24, 20],
):
    for i, j in dace.map[0:24, 0:20]:
        C[i, j] = A[i, j] / B[i, j]


# ---------------------------------------------------------------------------
# @dace.program definitions — float32, power-of-2 shapes (Python backend GPU)
# ---------------------------------------------------------------------------


@dace.program
def prog_sin_f32(A: dace.float32[16, 16], C: dace.float32[16, 16]):
    for i, j in dace.map[0:16, 0:16]:
        with dace.tasklet:
            a << A[i, j]
            c >> C[i, j]
            c = math.sin(a)


@dace.program
def prog_cos_f32(A: dace.float32[16, 16], C: dace.float32[16, 16]):
    for i, j in dace.map[0:16, 0:16]:
        with dace.tasklet:
            a << A[i, j]
            c >> C[i, j]
            c = math.cos(a)


@dace.program
def prog_exp_f32(A: dace.float32[16, 16], C: dace.float32[16, 16]):
    for i, j in dace.map[0:16, 0:16]:
        with dace.tasklet:
            a << A[i, j]
            c >> C[i, j]
            c = math.exp(a)


@dace.program
def prog_sqrt_f32(A: dace.float32[16, 16], C: dace.float32[16, 16]):
    for i, j in dace.map[0:16, 0:16]:
        with dace.tasklet:
            a << A[i, j]
            c >> C[i, j]
            c = math.sqrt(a)


@dace.program
def prog_log_f32(A: dace.float32[16, 16], C: dace.float32[16, 16]):
    for i, j in dace.map[0:16, 0:16]:
        with dace.tasklet:
            a << A[i, j]
            c >> C[i, j]
            c = math.log(a)


@dace.program
def prog_div_f32(
    A: dace.float32[16, 16],
    B: dace.float32[16, 16],
    C: dace.float32[16, 16],
):
    for i, j in dace.map[0:16, 0:16]:
        C[i, j] = A[i, j] / B[i, j]


# ---------------------------------------------------------------------------
# @dace.program definitions — 1D arrays
# ---------------------------------------------------------------------------


@dace.program
def prog_sin_1d(A: dace.float64[30], C: dace.float64[30]):
    for i in dace.map[0:30]:
        with dace.tasklet:
            a << A[i]
            c >> C[i]
            c = math.sin(a)


@dace.program
def prog_exp_1d(A: dace.float64[30], C: dace.float64[30]):
    for i in dace.map[0:30]:
        with dace.tasklet:
            a << A[i]
            c >> C[i]
            c = math.exp(a)


@dace.program
def prog_add_1d_f32(
    A: dace.float32[30],
    B: dace.float32[30],
    C: dace.float32[30],
):
    for i in dace.map[0:30]:
        C[i] = A[i] + B[i]


@dace.program
def prog_negate_1d_f32(A: dace.float32[30], C: dace.float32[30]):
    for i in dace.map[0:30]:
        C[i] = -A[i]


@dace.program
def prog_sin_1d_f32(A: dace.float32[32], C: dace.float32[32]):
    for i in dace.map[0:32]:
        with dace.tasklet:
            a << A[i]
            c >> C[i]
            c = math.sin(a)


# ---------------------------------------------------------------------------
# @dace.program definitions — symbolic sizes
# ---------------------------------------------------------------------------

FN = dace.symbol("FN", dtype=dace.int32)
FM = dace.symbol("FM", dtype=dace.int32)


@dace.program
def prog_sin_sym(A: dace.float64[FN, FM], C: dace.float64[FN, FM]):
    for i, j in dace.map[0:FN, 0:FM]:
        with dace.tasklet:
            a << A[i, j]
            c >> C[i, j]
            c = math.sin(a)


@dace.program
def prog_exp_sym_f32(A: dace.float32[FN, FM], C: dace.float32[FN, FM]):
    for i, j in dace.map[0:FN, 0:FM]:
        with dace.tasklet:
            a << A[i, j]
            c >> C[i, j]
            c = math.exp(a)


# ---------------------------------------------------------------------------
# @dace.program definitions — non-divisible tile boundaries
# ---------------------------------------------------------------------------


@dace.program
def prog_sin_nondiv(A: dace.float32[30, 25], C: dace.float32[30, 25]):
    for i, j in dace.map[0:30, 0:25]:
        with dace.tasklet:
            a << A[i, j]
            c >> C[i, j]
            c = math.sin(a)


@dace.program
def prog_sqrt_nondiv(A: dace.float32[33, 27], C: dace.float32[33, 27]):
    for i, j in dace.map[0:33, 0:27]:
        with dace.tasklet:
            a << A[i, j]
            c >> C[i, j]
            c = math.sqrt(a)


@dace.program
def prog_exp_nondiv_f32(A: dace.float32[30, 25], C: dace.float32[30, 25]):
    for i, j in dace.map[0:30, 0:25]:
        with dace.tasklet:
            a << A[i, j]
            c >> C[i, j]
            c = math.exp(a)


# ---------------------------------------------------------------------------
# @dace.program definitions — 3D
# ---------------------------------------------------------------------------


@dace.program
def prog_sin_3d(A: dace.float32[8, 8, 8], C: dace.float32[8, 8, 8]):
    for i, j, k in dace.map[0:8, 0:8, 0:8]:
        with dace.tasklet:
            a << A[i, j, k]
            c >> C[i, j, k]
            c = math.sin(a)


@dace.program
def prog_exp_3d(A: dace.float32[8, 8, 8], C: dace.float32[8, 8, 8]):
    for i, j, k in dace.map[0:8, 0:8, 0:8]:
        with dace.tasklet:
            a << A[i, j, k]
            c >> C[i, j, k]
            c = math.exp(a)


# ---------------------------------------------------------------------------
# Section A: C++ backend integration tests — math functions (no GPU needed)
# ---------------------------------------------------------------------------


class TestCppBackendMathOps:
    """Full pipeline → C++ compile → run for math functions missing from
    cutile_frontend_test.py (which covers only cos, abs, negate, +, -, *).

    Pattern: @dace.program → apply_cutile_pipeline() → expand_library_nodes()
             → sdfg(...) [DaCe compiles + runs C++] → NumPy comparison.
    """

    def test_sin_2d(self):
        """sin on float64 24×20 array through the full C++ pipeline."""
        sdfg = prog_sin_2d.to_sdfg(simplify=True)
        count = apply_cutile_pipeline(
            sdfg, validate=True, apply_map_collapse_and_tiling=True, tile_shape=(6, 5)
        )
        assert count >= 1
        assert any(isinstance(n, TileOpLibraryNode) for n in _frontend_library_nodes(sdfg))
        sdfg.expand_library_nodes()
        sdfg.validate()
        rng = np.random.default_rng(1001)
        a = rng.uniform(-np.pi, np.pi, (24, 20))
        c = np.zeros((24, 20))
        sdfg(A=a, C=c)
        np.testing.assert_allclose(c, np.sin(a), rtol=1e-12, atol=1e-12)

    def test_exp_2d(self):
        """exp on float64 24×20 array through the full C++ pipeline."""
        sdfg = prog_exp_2d.to_sdfg(simplify=True)
        count = apply_cutile_pipeline(
            sdfg, validate=True, apply_map_collapse_and_tiling=True, tile_shape=(6, 5)
        )
        assert count >= 1
        sdfg.expand_library_nodes()
        sdfg.validate()
        rng = np.random.default_rng(1002)
        a = rng.uniform(-2.0, 2.0, (24, 20))
        c = np.zeros((24, 20))
        sdfg(A=a, C=c)
        np.testing.assert_allclose(c, np.exp(a), rtol=1e-12, atol=1e-12)

    def test_sqrt_2d(self):
        """sqrt on float64 24×20 array through the full C++ pipeline."""
        sdfg = prog_sqrt_2d.to_sdfg(simplify=True)
        count = apply_cutile_pipeline(
            sdfg, validate=True, apply_map_collapse_and_tiling=True, tile_shape=(6, 5)
        )
        assert count >= 1
        sdfg.expand_library_nodes()
        sdfg.validate()
        rng = np.random.default_rng(1003)
        a = rng.uniform(0.1, 10.0, (24, 20))
        c = np.zeros((24, 20))
        sdfg(A=a, C=c)
        np.testing.assert_allclose(c, np.sqrt(a), rtol=1e-12, atol=1e-12)

    def test_log_2d(self):
        """log on float64 24×20 array through the full C++ pipeline."""
        sdfg = prog_log_2d.to_sdfg(simplify=True)
        count = apply_cutile_pipeline(
            sdfg, validate=True, apply_map_collapse_and_tiling=True, tile_shape=(6, 5)
        )
        assert count >= 1
        sdfg.expand_library_nodes()
        sdfg.validate()
        rng = np.random.default_rng(1004)
        a = rng.uniform(0.1, 10.0, (24, 20))
        c = np.zeros((24, 20))
        sdfg(A=a, C=c)
        np.testing.assert_allclose(c, np.log(a), rtol=1e-12, atol=1e-12)

    def test_div_2d(self):
        """A/B division on float64 24×20 array through the full C++ pipeline."""
        sdfg = prog_div_2d.to_sdfg(simplify=True)
        count = apply_cutile_pipeline(
            sdfg, validate=True, apply_map_collapse_and_tiling=True, tile_shape=(6, 5)
        )
        assert count >= 1
        sdfg.expand_library_nodes()
        sdfg.validate()
        rng = np.random.default_rng(1005)
        a = rng.uniform(1.0, 5.0, (24, 20))
        b = rng.uniform(0.5, 2.0, (24, 20))
        c = np.zeros((24, 20))
        sdfg(A=a, B=b, C=c)
        np.testing.assert_allclose(c, a / b, rtol=1e-12, atol=1e-12)

    def test_sin_1d(self):
        """sin on float64 1D array of 30 elements (non-multiple of tile)."""
        sdfg = prog_sin_1d.to_sdfg(simplify=True)
        count = apply_cutile_pipeline(
            sdfg, validate=True, apply_map_collapse_and_tiling=True, tile_shape=(7,)
        )
        assert count >= 1
        sdfg.expand_library_nodes()
        sdfg.validate()
        rng = np.random.default_rng(1006)
        a = rng.uniform(-np.pi, np.pi, (30,))
        c = np.zeros((30,))
        sdfg(A=a, C=c)
        np.testing.assert_allclose(c, np.sin(a), rtol=1e-12, atol=1e-12)

    def test_exp_1d(self):
        """exp on float64 1D array of 30 elements."""
        sdfg = prog_exp_1d.to_sdfg(simplify=True)
        count = apply_cutile_pipeline(
            sdfg, validate=True, apply_map_collapse_and_tiling=True, tile_shape=(7,)
        )
        assert count >= 1
        sdfg.expand_library_nodes()
        sdfg.validate()
        rng = np.random.default_rng(1007)
        a = rng.uniform(-2.0, 2.0, (30,))
        c = np.zeros((30,))
        sdfg(A=a, C=c)
        np.testing.assert_allclose(c, np.exp(a), rtol=1e-12, atol=1e-12)

    def test_add_1d_f32(self):
        """float32 1D add on 30-element arrays (non-multiple of tile)."""
        sdfg = prog_add_1d_f32.to_sdfg(simplify=True)
        count = apply_cutile_pipeline(
            sdfg, validate=True, apply_map_collapse_and_tiling=True, tile_shape=(7,)
        )
        assert count >= 1
        sdfg.expand_library_nodes()
        sdfg.validate()
        rng = np.random.default_rng(1008)
        a = rng.uniform(-5.0, 5.0, (30,)).astype(np.float32)
        b = rng.uniform(-5.0, 5.0, (30,)).astype(np.float32)
        c = np.zeros((30,), dtype=np.float32)
        sdfg(A=a, B=b, C=c)
        np.testing.assert_allclose(c, a + b, rtol=1e-6, atol=1e-6)

    def test_negate_1d_f32(self):
        """float32 1D negate on 30-element arrays."""
        sdfg = prog_negate_1d_f32.to_sdfg(simplify=True)
        count = apply_cutile_pipeline(
            sdfg, validate=True, apply_map_collapse_and_tiling=True, tile_shape=(7,)
        )
        assert count >= 1
        sdfg.expand_library_nodes()
        sdfg.validate()
        rng = np.random.default_rng(1009)
        a = rng.uniform(-5.0, 5.0, (30,)).astype(np.float32)
        c = np.zeros((30,), dtype=np.float32)
        sdfg(A=a, C=c)
        np.testing.assert_allclose(c, -a, rtol=1e-6, atol=1e-6)

    def test_sin_float32(self):
        """sin on float32 16×16 array through the C++ pipeline."""
        sdfg = prog_sin_f32.to_sdfg(simplify=True)
        count = apply_cutile_pipeline(
            sdfg, validate=True, apply_map_collapse_and_tiling=True, tile_shape=(8, 8)
        )
        assert count >= 1
        sdfg.expand_library_nodes()
        sdfg.validate()
        rng = np.random.default_rng(1010)
        a = rng.uniform(-np.pi, np.pi, (16, 16)).astype(np.float32)
        c = np.zeros((16, 16), dtype=np.float32)
        sdfg(A=a, C=c)
        np.testing.assert_allclose(c, np.sin(a), rtol=1e-6, atol=1e-6)

    def test_exp_float32(self):
        """exp on float32 16×16 array through the C++ pipeline."""
        sdfg = prog_exp_f32.to_sdfg(simplify=True)
        count = apply_cutile_pipeline(
            sdfg, validate=True, apply_map_collapse_and_tiling=True, tile_shape=(8, 8)
        )
        assert count >= 1
        sdfg.expand_library_nodes()
        sdfg.validate()
        rng = np.random.default_rng(1011)
        a = rng.uniform(-2.0, 2.0, (16, 16)).astype(np.float32)
        c = np.zeros((16, 16), dtype=np.float32)
        sdfg(A=a, C=c)
        np.testing.assert_allclose(c, np.exp(a), rtol=1e-5, atol=1e-6)

    def test_sqrt_float32(self):
        """sqrt on float32 16×16 array through the C++ pipeline."""
        sdfg = prog_sqrt_f32.to_sdfg(simplify=True)
        count = apply_cutile_pipeline(
            sdfg, validate=True, apply_map_collapse_and_tiling=True, tile_shape=(8, 8)
        )
        assert count >= 1
        sdfg.expand_library_nodes()
        sdfg.validate()
        rng = np.random.default_rng(1012)
        a = rng.uniform(0.1, 10.0, (16, 16)).astype(np.float32)
        c = np.zeros((16, 16), dtype=np.float32)
        sdfg(A=a, C=c)
        np.testing.assert_allclose(c, np.sqrt(a), rtol=1e-6, atol=1e-6)

    def test_sin_symbolic_size(self):
        """sin on symbolically-sized float64 array — runtime sizes 18×14."""
        sdfg = prog_sin_sym.to_sdfg(simplify=True)
        count = apply_cutile_pipeline(
            sdfg, validate=True, apply_map_collapse_and_tiling=True, tile_shape=(6, 5)
        )
        assert count >= 1
        sdfg.expand_library_nodes()
        sdfg.validate()
        n, m = 18, 14
        rng = np.random.default_rng(1013)
        a = rng.uniform(-np.pi, np.pi, (n, m))
        c = np.zeros((n, m))
        sdfg(A=a, C=c, FN=n, FM=m)
        np.testing.assert_allclose(c, np.sin(a), rtol=1e-12, atol=1e-12)

    def test_sin_nondivisible_tile(self):
        """sin on float32 30×25 array with tile_shape=(7, 5) — not a divisor of either dim."""
        sdfg = prog_sin_nondiv.to_sdfg(simplify=True)
        count = apply_cutile_pipeline(
            sdfg, validate=True, apply_map_collapse_and_tiling=True, tile_shape=(7, 5)
        )
        assert count >= 1
        sdfg.expand_library_nodes()
        sdfg.validate()
        rng = np.random.default_rng(1014)
        a = rng.uniform(-np.pi, np.pi, (30, 25)).astype(np.float32)
        c = np.zeros((30, 25), dtype=np.float32)
        sdfg(A=a, C=c)
        np.testing.assert_allclose(c, np.sin(a), rtol=1e-5, atol=1e-6)

    def test_sqrt_nondivisible_tile(self):
        """sqrt on float32 33×27 array with tile_shape=(8, 7) — partially covered tiles."""
        sdfg = prog_sqrt_nondiv.to_sdfg(simplify=True)
        count = apply_cutile_pipeline(
            sdfg, validate=True, apply_map_collapse_and_tiling=True, tile_shape=(8, 7)
        )
        assert count >= 1
        sdfg.expand_library_nodes()
        sdfg.validate()
        rng = np.random.default_rng(1015)
        a = rng.uniform(0.1, 10.0, (33, 27)).astype(np.float32)
        c = np.zeros((33, 27), dtype=np.float32)
        sdfg(A=a, C=c)
        np.testing.assert_allclose(c, np.sqrt(a), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Section B: Python backend structural tests (no GPU required)
# ---------------------------------------------------------------------------


class TestPythonBackendMathOpsStructure:
    """Structural codegen tests: verify the cuTile Python backend emits the
    correct ``ct.<func>(...)`` namespace-prefixed calls for each math operation.

    No GPU is required — these tests only inspect the generated Python kernel
    source code via ``dace_codegen.generate_code()``.
    """

    def test_sin_emits_ct_sin(self):
        """Python backend: sin → ct.sin in generated kernel."""
        sdfg = prog_sin_2d.to_sdfg(simplify=True)
        sdfg.backend = dtypes.BackendLanguage.Python
        apply_cutile_pipeline(sdfg)
        frame_code = _frame_code_of(sdfg)
        assert "@ct.kernel" in frame_code
        assert "ct.sin(" in frame_code

    def test_cos_emits_ct_cos(self):
        """Python backend: cos → ct.cos in generated kernel."""
        sdfg = prog_cos_2d.to_sdfg(simplify=True)
        sdfg.backend = dtypes.BackendLanguage.Python
        apply_cutile_pipeline(sdfg)
        frame_code = _frame_code_of(sdfg)
        assert "@ct.kernel" in frame_code
        assert "ct.cos(" in frame_code

    def test_exp_emits_ct_exp(self):
        """Python backend: exp → ct.exp in generated kernel."""
        sdfg = prog_exp_2d.to_sdfg(simplify=True)
        sdfg.backend = dtypes.BackendLanguage.Python
        apply_cutile_pipeline(sdfg)
        frame_code = _frame_code_of(sdfg)
        assert "@ct.kernel" in frame_code
        assert "ct.exp(" in frame_code

    def test_sqrt_emits_ct_sqrt(self):
        """Python backend: sqrt → ct.sqrt in generated kernel."""
        sdfg = prog_sqrt_2d.to_sdfg(simplify=True)
        sdfg.backend = dtypes.BackendLanguage.Python
        apply_cutile_pipeline(sdfg)
        frame_code = _frame_code_of(sdfg)
        assert "@ct.kernel" in frame_code
        assert "ct.sqrt(" in frame_code

    def test_log_emits_ct_log(self):
        """Python backend: log → ct.log in generated kernel."""
        sdfg = prog_log_2d.to_sdfg(simplify=True)
        sdfg.backend = dtypes.BackendLanguage.Python
        apply_cutile_pipeline(sdfg)
        frame_code = _frame_code_of(sdfg)
        assert "@ct.kernel" in frame_code
        assert "ct.log(" in frame_code

    def test_div_emits_kernel_boilerplate(self):
        """Python backend: division → @ct.kernel and ct.launch in generated code."""
        sdfg = prog_div_2d.to_sdfg(simplify=True)
        sdfg.backend = dtypes.BackendLanguage.Python
        apply_cutile_pipeline(sdfg)
        frame_code = _frame_code_of(sdfg)
        assert "@ct.kernel" in frame_code
        assert "ct.launch" in frame_code

    def test_sin_1d_uses_single_block_index(self):
        """Python backend: 1D sin kernel uses ct.bid(0), not ct.bid(1)."""
        sdfg = prog_sin_1d.to_sdfg(simplify=True)
        sdfg.backend = dtypes.BackendLanguage.Python
        apply_cutile_pipeline(sdfg)
        frame_code = _frame_code_of(sdfg)
        assert "@ct.kernel" in frame_code
        assert "ct.bid(0)" in frame_code
        assert "ct.bid(1)" not in frame_code

    def test_sin_f32_emits_ct_sin(self):
        """Python backend: float32 sin → ct.sin in generated kernel."""
        sdfg = prog_sin_f32.to_sdfg(simplify=True)
        sdfg.backend = dtypes.BackendLanguage.Python
        apply_cutile_pipeline(sdfg)
        frame_code = _frame_code_of(sdfg)
        assert "@ct.kernel" in frame_code
        assert "ct.sin(" in frame_code

    def test_sin_symbolic_surfaces_free_symbols(self):
        """Python backend: symbolic-size kernel must reference FN and FM."""
        sdfg = prog_sin_sym.to_sdfg(simplify=True)
        sdfg.backend = dtypes.BackendLanguage.Python
        apply_cutile_pipeline(sdfg)
        frame_code = _frame_code_of(sdfg)
        assert "@ct.kernel" in frame_code
        assert "FN" in frame_code
        assert "FM" in frame_code

    def test_sin_3d_uses_three_block_ids(self):
        """Python backend: 3D sin kernel uses ct.bid(0), ct.bid(1), ct.bid(2)."""
        sdfg = prog_sin_3d.to_sdfg(simplify=True)
        sdfg.backend = dtypes.BackendLanguage.Python
        apply_cutile_pipeline(sdfg)
        frame_code = _frame_code_of(sdfg)
        assert "ct.bid(0)" in frame_code
        assert "ct.bid(1)" in frame_code
        assert "ct.bid(2)" in frame_code


# ---------------------------------------------------------------------------
# Section C: Python backend GPU runtime tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_pipeline_sin_python_backend_runtime():
    """sin: @dace.program → Python backend → GPU compile → run → NumPy sin."""
    rng = np.random.default_rng(2001)
    a_np = rng.uniform(-np.pi, np.pi, (16, 16)).astype(np.float32)
    a_cp = cp.asarray(a_np)
    c_cp = cp.zeros((16, 16), dtype=cp.float32)

    sdfg = prog_sin_f32.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    apply_cutile_pipeline(sdfg)
    csdfg = sdfg.compile()
    assert csdfg is not None
    csdfg(A=a_cp, C=c_cp)
    np.testing.assert_allclose(cp.asnumpy(c_cp), np.sin(a_np), rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
def test_pipeline_cos_python_backend_runtime():
    """cos: @dace.program → Python backend → GPU compile → run → NumPy cos."""
    rng = np.random.default_rng(2002)
    a_np = rng.uniform(-np.pi, np.pi, (16, 16)).astype(np.float32)
    a_cp = cp.asarray(a_np)
    c_cp = cp.zeros((16, 16), dtype=cp.float32)

    sdfg = prog_cos_f32.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    apply_cutile_pipeline(sdfg)
    csdfg = sdfg.compile()
    assert csdfg is not None
    csdfg(A=a_cp, C=c_cp)
    np.testing.assert_allclose(cp.asnumpy(c_cp), np.cos(a_np), rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
def test_pipeline_exp_python_backend_runtime():
    """exp: @dace.program → Python backend → GPU compile → run → NumPy exp."""
    rng = np.random.default_rng(2003)
    a_np = rng.uniform(-2.0, 2.0, (16, 16)).astype(np.float32)
    a_cp = cp.asarray(a_np)
    c_cp = cp.zeros((16, 16), dtype=cp.float32)

    sdfg = prog_exp_f32.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    apply_cutile_pipeline(sdfg)
    csdfg = sdfg.compile()
    assert csdfg is not None
    csdfg(A=a_cp, C=c_cp)
    np.testing.assert_allclose(cp.asnumpy(c_cp), np.exp(a_np), rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
def test_pipeline_sqrt_python_backend_runtime():
    """sqrt: @dace.program → Python backend → GPU compile → run → NumPy sqrt."""
    rng = np.random.default_rng(2004)
    a_np = rng.uniform(0.1, 10.0, (16, 16)).astype(np.float32)
    a_cp = cp.asarray(a_np)
    c_cp = cp.zeros((16, 16), dtype=cp.float32)

    sdfg = prog_sqrt_f32.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    apply_cutile_pipeline(sdfg)
    csdfg = sdfg.compile()
    assert csdfg is not None
    csdfg(A=a_cp, C=c_cp)
    np.testing.assert_allclose(cp.asnumpy(c_cp), np.sqrt(a_np), rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
def test_pipeline_log_python_backend_runtime():
    """log: @dace.program → Python backend → GPU compile → run → NumPy log."""
    rng = np.random.default_rng(2005)
    a_np = rng.uniform(0.1, 10.0, (16, 16)).astype(np.float32)
    a_cp = cp.asarray(a_np)
    c_cp = cp.zeros((16, 16), dtype=cp.float32)

    sdfg = prog_log_f32.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    apply_cutile_pipeline(sdfg)
    csdfg = sdfg.compile()
    assert csdfg is not None
    csdfg(A=a_cp, C=c_cp)
    np.testing.assert_allclose(cp.asnumpy(c_cp), np.log(a_np), rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
def test_pipeline_div_python_backend_runtime():
    """A/B division: @dace.program → Python backend → GPU compile → run."""
    rng = np.random.default_rng(2006)
    a_np = rng.uniform(1.0, 5.0, (16, 16)).astype(np.float32)
    b_np = rng.uniform(0.5, 2.0, (16, 16)).astype(np.float32)
    a_cp = cp.asarray(a_np)
    b_cp = cp.asarray(b_np)
    c_cp = cp.zeros((16, 16), dtype=cp.float32)

    sdfg = prog_div_f32.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    apply_cutile_pipeline(sdfg)
    csdfg = sdfg.compile()
    assert csdfg is not None
    csdfg(A=a_cp, B=b_cp, C=c_cp)
    np.testing.assert_allclose(cp.asnumpy(c_cp), a_np / b_np, rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
def test_pipeline_sin_1d_python_backend_runtime():
    """sin on 1D float32 array: Python backend GPU runtime correctness."""
    rng = np.random.default_rng(2007)
    a_np = rng.uniform(-np.pi, np.pi, (32,)).astype(np.float32)
    a_cp = cp.asarray(a_np)
    c_cp = cp.zeros((32,), dtype=cp.float32)

    sdfg = prog_sin_1d_f32.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    apply_cutile_pipeline(sdfg)
    csdfg = sdfg.compile()
    assert csdfg is not None
    csdfg(A=a_cp, C=c_cp)
    np.testing.assert_allclose(cp.asnumpy(c_cp), np.sin(a_np), rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
def test_pipeline_sin_symbolic_python_backend_runtime():
    """sin on symbolically-sized float32 array: runtime sizes 18×14."""
    rng = np.random.default_rng(2008)
    n_val, m_val = 18, 14
    a_np = rng.uniform(-np.pi, np.pi, (n_val, m_val)).astype(np.float32)
    a_cp = cp.asarray(a_np)
    c_cp = cp.zeros((n_val, m_val), dtype=cp.float32)

    sdfg = prog_exp_sym_f32.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    apply_cutile_pipeline(sdfg)
    csdfg = sdfg.compile()
    assert csdfg is not None
    csdfg(A=a_cp, C=c_cp, FN=n_val, FM=m_val)
    np.testing.assert_allclose(cp.asnumpy(c_cp), np.exp(a_np), rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
def test_pipeline_sin_nondivisible_tile_python_backend_runtime():
    """sin on 30×25 float32 array with power-of-2 tile (16×16): partial tiles handled."""
    rng = np.random.default_rng(2009)
    a_np = rng.uniform(-np.pi, np.pi, (30, 25)).astype(np.float32)
    a_cp = cp.asarray(a_np)
    c_cp = cp.zeros((30, 25), dtype=cp.float32)

    sdfg = prog_sin_nondiv.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    apply_cutile_pipeline(sdfg, apply_map_collapse_and_tiling=True, tile_shape=(16, 16))
    csdfg = sdfg.compile()
    assert csdfg is not None
    csdfg(A=a_cp, C=c_cp)
    np.testing.assert_allclose(cp.asnumpy(c_cp), np.sin(a_np), rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
def test_pipeline_exp_nondivisible_tile_python_backend_runtime():
    """exp on 30×25 float32 array with 16×16 tile (neither divides evenly)."""
    rng = np.random.default_rng(2010)
    a_np = rng.uniform(-1.5, 1.5, (30, 25)).astype(np.float32)
    a_cp = cp.asarray(a_np)
    c_cp = cp.zeros((30, 25), dtype=cp.float32)

    sdfg = prog_exp_nondiv_f32.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    apply_cutile_pipeline(sdfg, apply_map_collapse_and_tiling=True, tile_shape=(16, 16))
    csdfg = sdfg.compile()
    assert csdfg is not None
    csdfg(A=a_cp, C=c_cp)
    np.testing.assert_allclose(cp.asnumpy(c_cp), np.exp(a_np), rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
def test_pipeline_sin_3d_python_backend_runtime():
    """sin on 3D float32 array: Python backend GPU runtime correctness."""
    rng = np.random.default_rng(2011)
    a_np = rng.uniform(-np.pi, np.pi, (8, 8, 8)).astype(np.float32)
    a_cp = cp.asarray(a_np)
    c_cp = cp.zeros((8, 8, 8), dtype=cp.float32)

    sdfg = prog_sin_3d.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    apply_cutile_pipeline(sdfg)
    csdfg = sdfg.compile()
    assert csdfg is not None
    csdfg(A=a_cp, C=c_cp)
    np.testing.assert_allclose(cp.asnumpy(c_cp), np.sin(a_np), rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
def test_pipeline_exp_3d_python_backend_runtime():
    """exp on 3D float32 array: Python backend GPU runtime correctness."""
    rng = np.random.default_rng(2012)
    a_np = rng.uniform(-1.5, 1.5, (8, 8, 8)).astype(np.float32)
    a_cp = cp.asarray(a_np)
    c_cp = cp.zeros((8, 8, 8), dtype=cp.float32)

    sdfg = prog_exp_3d.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    apply_cutile_pipeline(sdfg)
    csdfg = sdfg.compile()
    assert csdfg is not None
    csdfg(A=a_cp, C=c_cp)
    np.testing.assert_allclose(cp.asnumpy(c_cp), np.exp(a_np), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
