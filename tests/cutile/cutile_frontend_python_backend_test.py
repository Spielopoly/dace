"""Frontend integration tests for schedule-based cuTile Python backend codegen."""

import pytest
import numpy as np

import dace
from dace import dtypes
from dace.codegen import codegen as dace_codegen
from dace.libraries.cutile.transformations.pipeline import apply_cutile_pipeline
from dace.sdfg import nodes


cp = pytest.importorskip("cupy")
pytest.importorskip("cuda.tile")


@dace.program
def frontend_add(
    A: dace.float32[16, 16],
    B: dace.float32[16, 16],
    C: dace.float32[16, 16],
):
    for i, j in dace.map[0:16, 0:16]:
        C[i, j] = A[i, j] + B[i, j]


@dace.program
def frontend_selfadd(
    A: dace.float32[16, 16],
    C: dace.float32[16, 16],
):
    for i, j in dace.map[0:16, 0:16]:
        C[i, j] = A[i, j] + A[i, j]


def _set_cutile_schedule(sdfg: dace.SDFG) -> None:
    map_entries = [
        n
        for state in sdfg.states()
        for n in state.nodes()
        if isinstance(n, nodes.MapEntry)
    ]
    assert map_entries
    for me in map_entries:
        me.map.schedule = dtypes.ScheduleType.CuTile


def _all_map_schedules(sdfg: dace.SDFG) -> list[dtypes.ScheduleType]:
    return [
        n.map.schedule
        for state in sdfg.states()
        for n in state.nodes()
        if isinstance(n, nodes.MapEntry)
    ]


def _frame_code_of(sdfg: dace.SDFG) -> str:
    code_objects = dace_codegen.generate_code(sdfg)
    return next(co.clean_code for co in code_objects if co.name == sdfg.name)


def test_frontend_add_python_backend_codegen():
    sdfg = frontend_add.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    _set_cutile_schedule(sdfg)
    assert _all_map_schedules(sdfg)
    assert all(s == dtypes.ScheduleType.CuTile for s in _all_map_schedules(sdfg))

    frame_code = _frame_code_of(sdfg)
    assert "@ct.kernel" in frame_code
    assert "ct.launch" in frame_code
    assert "__ct_t" in frame_code


def test_frontend_selfadd_python_backend_codegen():
    sdfg = frontend_selfadd.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    _set_cutile_schedule(sdfg)

    frame_code = _frame_code_of(sdfg)
    assert "@ct.kernel" in frame_code
    assert "ct.launch" in frame_code
    # self-add should reference the same input tile variable multiple times
    assert frame_code.count("__ct_t") >= 2


def test_frontend_add_without_cutile_schedule_no_cutile_codegen():
    sdfg = frontend_add.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    assert _all_map_schedules(sdfg)
    assert all(s != dtypes.ScheduleType.CuTile for s in _all_map_schedules(sdfg))

    frame_code = _frame_code_of(sdfg)
    assert "@ct.kernel" not in frame_code
    assert "ct.launch" not in frame_code


# ---------------------------------------------------------------------------
# End-to-end pipeline integration tests: @dace.program → apply_cutile_pipeline
# ---------------------------------------------------------------------------


@dace.program
def pipeline_vadd(
    A: dace.float32[16, 16],
    B: dace.float32[16, 16],
    C: dace.float32[16, 16],
):
    """Simple element-wise add used by the pipeline integration tests."""
    for i, j in dace.map[0:16, 0:16]:
        C[i, j] = A[i, j] + B[i, j]


@dace.program
def pipeline_selfadd(
    A: dace.float32[24, 20],
    C: dace.float32[24, 20],
):
    """Self-add program (single input referenced twice) for pipeline tests."""
    for i, j in dace.map[0:24, 0:20]:
        C[i, j] = A[i, j] + A[i, j]


@dace.program
def pipeline_scalar_multiply(
    A: dace.float32[16, 16],
    B: dace.float32[16, 16],
    C: dace.float32[16, 16],
):
    """Simple element-wise multiply used by runtime correctness tests."""
    for i, j in dace.map[0:16, 0:16]:
        C[i, j] = A[i, j] * B[i, j]


def test_pipeline_sets_cutile_schedule_and_generates_kernel():
    """End-to-end: @dace.program → to_sdfg → set Python backend → apply_cutile_pipeline.

    Verifies that:
    - ``apply_cutile_pipeline`` triggers ``SetCuTilePythonScope``, which marks
      outer maps with ``ScheduleType.CuTile`` (because ``sdfg.backend`` is Python).
    - The subsequent code generation emits ``@ct.kernel`` and ``ct.launch`` in
      the frame code, confirming the schedule-based dispatch path is active.
    """
    sdfg = pipeline_vadd.to_sdfg(simplify=True)
    # Enable the Python cuTile backend so SetCuTilePythonScope fires inside the pipeline.
    sdfg.backend = dtypes.BackendLanguage.Python

    apply_cutile_pipeline(sdfg)

    # After the pipeline, at least one map must have the CuTile schedule.
    schedules = _all_map_schedules(sdfg)
    assert schedules, "No map entries found after pipeline — pipeline may have failed silently"
    assert any(
        s == dtypes.ScheduleType.CuTile for s in schedules
    ), f"Expected at least one CuTile-scheduled map, got: {schedules}"

    # Code generation must produce the cuTile Python kernel boilerplate.
    frame_code = _frame_code_of(sdfg)
    assert "@ct.kernel" in frame_code, "Expected '@ct.kernel' decorator in generated code"
    assert "ct.launch" in frame_code, "Expected 'ct.launch' call in generated code"


def test_pipeline_without_python_backend_does_not_set_cutile_schedule():
    """End-to-end: pipeline without Python backend must NOT set CuTile schedules.

    ``SetCuTilePythonScope`` is a no-op when ``sdfg.backend`` is not
    ``BackendLanguage.Python``.  This test guards against the pipeline
    accidentally tagging maps as CuTile in a non-Python-backend context.
    """
    sdfg = pipeline_vadd.to_sdfg(simplify=True)
    # Deliberately do NOT set sdfg.backend — leave it at the default (C++/CUDA).

    apply_cutile_pipeline(sdfg)

    schedules = _all_map_schedules(sdfg)
    assert schedules, "No map entries found after pipeline"
    assert not any(
        s == dtypes.ScheduleType.CuTile for s in schedules
    ), f"No maps should have CuTile schedule without Python backend, got: {schedules}"


def test_pipeline_frontend_selfadd_schedule_and_codegen():
    """End-to-end: selfadd program (single input used twice) through the pipeline.

    Uses a non-square 24×20 array to exercise the pipeline with shapes that
    differ from the default tile size.  Verifies both that the CuTile schedule
    is set and that code generation produces the expected kernel boilerplate.
    """
    sdfg = pipeline_selfadd.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python

    apply_cutile_pipeline(sdfg)

    # Outer maps must be tagged CuTile.
    schedules = _all_map_schedules(sdfg)
    assert schedules, "No map entries found after pipeline"
    assert any(
        s == dtypes.ScheduleType.CuTile for s in schedules
    ), f"Expected at least one CuTile-scheduled map, got: {schedules}"

    # Code generation must include cuTile kernel boilerplate.
    frame_code = _frame_code_of(sdfg)
    assert "@ct.kernel" in frame_code, "Expected '@ct.kernel' decorator in generated code"
    assert "ct.launch" in frame_code, "Expected 'ct.launch' call in generated code"


@pytest.mark.gpu
def test_pipeline_vadd_runtime_correctness():
    """Pipeline add program should compile, execute, and match NumPy reference."""
    rng = np.random.default_rng(42)
    a_np = rng.random((16, 16)).astype(np.float32)
    b_np = rng.random((16, 16)).astype(np.float32)

    a_cp = cp.asarray(a_np)
    b_cp = cp.asarray(b_np)
    c_cp = cp.zeros((16, 16), dtype=cp.float32)

    sdfg = pipeline_vadd.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    apply_cutile_pipeline(sdfg)

    csdfg = sdfg.compile()
    assert csdfg is not None
    csdfg(A=a_cp, B=b_cp, C=c_cp)

    np.testing.assert_allclose(cp.asnumpy(c_cp), a_np + b_np, rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
def test_pipeline_selfadd_runtime_correctness():
    """Pipeline self-add program should compile, execute, and match NumPy reference."""
    rng = np.random.default_rng(42)
    a_np = rng.random((24, 20)).astype(np.float32)

    a_cp = cp.asarray(a_np)
    c_cp = cp.zeros((24, 20), dtype=cp.float32)

    sdfg = pipeline_selfadd.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    apply_cutile_pipeline(sdfg)

    csdfg = sdfg.compile()
    assert csdfg is not None
    csdfg(A=a_cp, C=c_cp)

    np.testing.assert_allclose(cp.asnumpy(c_cp), 2.0 * a_np, rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
def test_pipeline_scalar_multiply_runtime_correctness():
    """Pipeline multiply program should compile, execute, and match NumPy reference."""
    rng = np.random.default_rng(42)
    a_np = rng.random((16, 16)).astype(np.float32)
    b_np = rng.random((16, 16)).astype(np.float32)

    a_cp = cp.asarray(a_np)
    b_cp = cp.asarray(b_np)
    c_cp = cp.zeros((16, 16), dtype=cp.float32)

    sdfg = pipeline_scalar_multiply.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    apply_cutile_pipeline(sdfg)

    csdfg = sdfg.compile()
    assert csdfg is not None
    csdfg(A=a_cp, B=b_cp, C=c_cp)

    np.testing.assert_allclose(cp.asnumpy(c_cp), a_np * b_np, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Additional pipeline integration programs
# ---------------------------------------------------------------------------


FN = dace.symbol("FN", dtype=dace.int32)
FM = dace.symbol("FM", dtype=dace.int32)


@dace.program
def pipeline_negate(
    A: dace.float32[24, 20],
    C: dace.float32[24, 20],
):
    """Unary negation program — exercises single-input tile load + unary tasklet."""
    for i, j in dace.map[0:24, 0:20]:
        C[i, j] = -A[i, j]


@dace.program
def pipeline_sub(
    A: dace.float32[16, 16],
    B: dace.float32[16, 16],
    C: dace.float32[16, 16],
):
    """Subtraction — exercises an op other than ``+`` / ``*``."""
    for i, j in dace.map[0:16, 0:16]:
        C[i, j] = A[i, j] - B[i, j]


@dace.program
def pipeline_symbolic_vadd(
    A: dace.float32[FN, FM],
    B: dace.float32[FN, FM],
    C: dace.float32[FN, FM],
):
    """Symbolic-shape add — exercises free-symbol surfacing in kernel signature."""
    for i, j in dace.map[0:FN, 0:FM]:
        C[i, j] = A[i, j] + B[i, j]


@dace.program
def pipeline_3d_vadd(
    A: dace.float32[8, 8, 8],
    B: dace.float32[8, 8, 8],
    C: dace.float32[8, 8, 8],
):
    """3-D add — exercises multi-dim ``ct.bid`` / index tuple construction."""
    for i, j, k in dace.map[0:8, 0:8, 0:8]:
        C[i, j, k] = A[i, j, k] + B[i, j, k]


# ---------------------------------------------------------------------------
# Structural codegen tests (no GPU required)
# ---------------------------------------------------------------------------


def test_pipeline_negate_codegen_structure():
    """Unary negate should generate a kernel with one load, one store, no second input arg."""
    sdfg = pipeline_negate.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    apply_cutile_pipeline(sdfg)

    frame_code = _frame_code_of(sdfg)
    assert "@ct.kernel" in frame_code
    assert "ct.launch" in frame_code
    assert frame_code.count("ct.load(A,") == 1
    assert frame_code.count("ct.store(C,") == 1


def test_pipeline_subtraction_codegen_structure():
    """Subtraction (non-add/mul op) should generate a kernel with two loads and one store."""
    sdfg = pipeline_sub.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    apply_cutile_pipeline(sdfg)

    frame_code = _frame_code_of(sdfg)
    assert "@ct.kernel" in frame_code
    assert "ct.launch" in frame_code
    assert "ct.load(A," in frame_code
    assert "ct.load(B," in frame_code
    assert "ct.store(C," in frame_code


def test_pipeline_selfadd_emits_one_load_per_unique_array():
    """Self-add: the same outer array A is read twice, but loaded only once."""
    sdfg = pipeline_selfadd.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    apply_cutile_pipeline(sdfg)

    frame_code = _frame_code_of(sdfg)
    assert frame_code.count("ct.load(A,") == 1, (
        "Expected exactly one ct.load(A, ...) call for self-add (deduplicated input).")


def test_pipeline_symbolic_shape_codegen_includes_free_symbols():
    """Symbolic-shape program: kernel signature and launch args must surface FN/FM."""
    sdfg = pipeline_symbolic_vadd.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    apply_cutile_pipeline(sdfg)

    frame_code = _frame_code_of(sdfg)
    assert "@ct.kernel" in frame_code
    assert "FN" in frame_code
    assert "FM" in frame_code


def test_pipeline_3d_vadd_codegen_uses_three_block_ids():
    """3-D map should emit ``ct.bid(0)``, ``ct.bid(1)``, and ``ct.bid(2)``."""
    sdfg = pipeline_3d_vadd.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    apply_cutile_pipeline(sdfg)

    frame_code = _frame_code_of(sdfg)
    assert "ct.bid(0)" in frame_code
    assert "ct.bid(1)" in frame_code
    assert "ct.bid(2)" in frame_code


def test_pipeline_rectangular_tile_codegen_structure():
    """Non-divisible tile shape (24×20 with tile_shape=(6, 5)) should still codegen."""
    sdfg = pipeline_negate.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    apply_cutile_pipeline(sdfg, apply_map_collapse_and_tiling=True, tile_shape=(6, 5))

    frame_code = _frame_code_of(sdfg)
    assert "@ct.kernel" in frame_code
    assert "ct.load(A," in frame_code
    assert "ct.store(C," in frame_code


# ---------------------------------------------------------------------------
# Runtime correctness tests (require GPU + cupy + cuda.tile)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_pipeline_negate_runtime_correctness():
    """Unary negate should compile, execute, and match NumPy reference."""
    rng = np.random.default_rng(43)
    a_np = rng.random((24, 20)).astype(np.float32)

    a_cp = cp.asarray(a_np)
    c_cp = cp.zeros((24, 20), dtype=cp.float32)

    sdfg = pipeline_negate.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    apply_cutile_pipeline(sdfg)

    csdfg = sdfg.compile()
    assert csdfg is not None
    csdfg(A=a_cp, C=c_cp)

    np.testing.assert_allclose(cp.asnumpy(c_cp), -a_np, rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
def test_pipeline_subtraction_runtime_correctness():
    """A - B should compile, execute, and match NumPy reference."""
    rng = np.random.default_rng(44)
    a_np = rng.random((16, 16)).astype(np.float32)
    b_np = rng.random((16, 16)).astype(np.float32)

    a_cp = cp.asarray(a_np)
    b_cp = cp.asarray(b_np)
    c_cp = cp.zeros((16, 16), dtype=cp.float32)

    sdfg = pipeline_sub.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    apply_cutile_pipeline(sdfg)

    csdfg = sdfg.compile()
    assert csdfg is not None
    csdfg(A=a_cp, B=b_cp, C=c_cp)

    np.testing.assert_allclose(cp.asnumpy(c_cp), a_np - b_np, rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
def test_pipeline_symbolic_shape_runtime_correctness():
    """Symbolic-shape add should compile, execute with runtime sizes, and match."""
    rng = np.random.default_rng(45)
    n_val = np.int32(18)
    m_val = np.int32(14)
    a_np = rng.random((n_val, m_val)).astype(np.float32)
    b_np = rng.random((n_val, m_val)).astype(np.float32)

    a_cp = cp.asarray(a_np)
    b_cp = cp.asarray(b_np)
    c_cp = cp.zeros((n_val, m_val), dtype=cp.float32)

    sdfg = pipeline_symbolic_vadd.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    apply_cutile_pipeline(sdfg)

    csdfg = sdfg.compile()
    assert csdfg is not None
    csdfg(A=a_cp, B=b_cp, C=c_cp, FN=n_val, FM=m_val)

    np.testing.assert_allclose(cp.asnumpy(c_cp), a_np + b_np, rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
def test_pipeline_3d_vadd_runtime_correctness():
    """3-D add should compile, execute, and match NumPy reference."""
    rng = np.random.default_rng(46)
    a_np = rng.random((8, 8, 8)).astype(np.float32)
    b_np = rng.random((8, 8, 8)).astype(np.float32)

    a_cp = cp.asarray(a_np)
    b_cp = cp.asarray(b_np)
    c_cp = cp.zeros((8, 8, 8), dtype=cp.float32)

    sdfg = pipeline_3d_vadd.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    apply_cutile_pipeline(sdfg)

    csdfg = sdfg.compile()
    assert csdfg is not None
    csdfg(A=a_cp, B=b_cp, C=c_cp)

    np.testing.assert_allclose(cp.asnumpy(c_cp), a_np + b_np, rtol=1e-5, atol=1e-6)


@pytest.mark.gpu
def test_pipeline_rectangular_tile_runtime_correctness():
    """Non-divisible tile shape should still produce correct results."""
    rng = np.random.default_rng(47)
    a_np = rng.random((24, 20)).astype(np.float32)

    a_cp = cp.asarray(a_np)
    c_cp = cp.zeros((24, 20), dtype=cp.float32)

    sdfg = pipeline_negate.to_sdfg(simplify=True)
    sdfg.backend = dtypes.BackendLanguage.Python
    apply_cutile_pipeline(sdfg, apply_map_collapse_and_tiling=True, tile_shape=(6, 5))

    csdfg = sdfg.compile()
    assert csdfg is not None
    csdfg(A=a_cp, C=c_cp)

    np.testing.assert_allclose(cp.asnumpy(c_cp), -a_np, rtol=1e-5, atol=1e-6)
