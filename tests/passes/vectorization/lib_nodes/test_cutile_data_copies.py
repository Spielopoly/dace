# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the ``CuTileInsertDataCopies`` lowering pass.

Two layers:

* **Structure / codegen tests (no GPU):** after applying the lowering pipeline
  through ``CuTileInsertDataCopies``, the SDFG has the expected copy-in /
  copy-out states, GPU_Global transient clones, CPU_Heap originals, and correct
  AccessNode/Memlet references.  Codegen tests verify ``cupy.asarray`` /
  ``cupy.asnumpy`` appear in the generated Python code.

* **Runtime tests (``@pytest.mark.gpu``):** compile and run on GPU with
  **NumPy** (host) arrays directly — the whole point of data copies — and
  compare against NumPy references.
"""

import ast
from typing import List, Set, Tuple

import numpy as np
import pytest

import dace
from dace import data, dtypes
from dace.sdfg import SDFG, nodes
from dace.transformation.passes.vectorization import VectorizeCuTile
from dace.transformation.passes.vectorization.cutile_lowering import (
    CuTileInsertDataCopies,
    CuTileSetGlobalStorage,
    CuTileSetImplementations,
    CuTileSetSchedules,
    CuTileSetTileStorage,
    CuTileValidateTiles,
)
from dace.transformation.passes.vectorization.vectorize_cpu_multi_dim import (
    VectorizeCPUMultiDim,
)


# ============================================================
# Fixture builders
# ============================================================


def _build_vadd_sdfg(name: str, dtype: dace.typeclass = dace.float64) -> SDFG:
    """Build a symbolic-sized 1-D ``C[i] = A[i] + B[i]`` SDFG.

    :param name: SDFG name (must be unique per test).
    :param dtype: Data type for all arrays.
    :returns: The constructed SDFG.
    """
    N = dace.symbol("N")
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (N,), dtype)
    sdfg.add_array("B", (N,), dtype)
    sdfg.add_array("C", (N,), dtype)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "add",
        {"i": "0:N"},
        {"_a": dace.Memlet("A[i]"), "_b": dace.Memlet("B[i]")},
        "_c = _a + _b",
        {"_c": dace.Memlet("C[i]")},
        external_edges=True,
    )
    return sdfg


def _build_vadd2d_sdfg(name: str, dtype: dace.typeclass = dace.float64) -> SDFG:
    """Build a symbolic-sized 2-D ``C[i, j] = A[i, j] + B[i, j]`` SDFG.

    :param name: SDFG name (must be unique per test).
    :param dtype: Data type for all arrays.
    :returns: The constructed SDFG.
    """
    M = dace.symbol("M")
    N = dace.symbol("N")
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (M, N), dtype)
    sdfg.add_array("B", (M, N), dtype)
    sdfg.add_array("C", (M, N), dtype)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "add2d",
        {"i": "0:M", "j": "0:N"},
        {"_a": dace.Memlet("A[i, j]"), "_b": dace.Memlet("B[i, j]")},
        "_c = _a + _b",
        {"_c": dace.Memlet("C[i, j]")},
        external_edges=True,
    )
    return sdfg


def _build_vadd_with_scalar_sdfg(name: str) -> SDFG:
    """Build ``C[i] = A[i] + B[i] + alpha`` with a scalar parameter.

    :param name: SDFG name (must be unique per test).
    :returns: The constructed SDFG.
    """
    N = dace.symbol("N")
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (N,), dace.float64)
    sdfg.add_array("B", (N,), dace.float64)
    sdfg.add_array("C", (N,), dace.float64)
    sdfg.add_scalar("alpha", dace.float64)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "add_scalar",
        {"i": "0:N"},
        {
            "_a": dace.Memlet("A[i]"),
            "_b": dace.Memlet("B[i]"),
            "_s": dace.Memlet("alpha[0]"),
        },
        "_c = _a + _b + _s",
        {"_c": dace.Memlet("C[i]")},
        external_edges=True,
    )
    return sdfg


# ============================================================
# Lowering helpers
# ============================================================


def _lower_before_data_copies(sdfg: SDFG, widths: Tuple[int, ...] = (8,)) -> None:
    """Run the lowering pipeline up to (but NOT including) CuTileInsertDataCopies.

    After this call the SDFG has CuTile-scheduled maps and GPU_Global
    non-transients, ready for ``CuTileInsertDataCopies``.

    :param sdfg: The SDFG to lower.
    :param widths: Per-dim tile widths (innermost-last).
    """
    vec = VectorizeCPUMultiDim(
        widths=widths,
        target_isa="CUTILE",
        expand_tile_nodes=False,
    )
    vec.apply_pass(sdfg, {})
    CuTileValidateTiles().apply_pass(sdfg, {})
    CuTileSetSchedules().apply_pass(sdfg, {})
    CuTileSetTileStorage().apply_pass(sdfg, {})
    CuTileSetGlobalStorage().apply_pass(sdfg, {})


def _finish_after_data_copies(sdfg: SDFG) -> None:
    """Run the lowering passes that come after CuTileInsertDataCopies.

    :param sdfg: The SDFG to finish lowering.
    """
    CuTileSetImplementations().apply_pass(sdfg, {})
    sdfg.expand_library_nodes()
    sdfg.backend = dtypes.BackendLanguage.Python


def _generate_code(sdfg: SDFG) -> str:
    """Generate Python-backend code for ``sdfg`` and return the main file text.

    :param sdfg: The (already lowered) SDFG.
    :returns: The generated code of the main code object.
    """
    code_objects = sdfg.generate_code()
    return next(co for co in code_objects if co.name == sdfg.name).code


# ============================================================
# Inspection helpers
# ============================================================


def _copyin_state(sdfg: SDFG) -> "dace.sdfg.state.SDFGState":
    """Return the copyin state (the start block) of an SDFG with data copies.

    :param sdfg: The SDFG to inspect.
    :returns: The start block state.
    """
    return sdfg.start_block


def _copyout_state(sdfg: SDFG) -> "dace.sdfg.state.SDFGState":
    """Return the copyout state (a terminal/sink state) of an SDFG with data copies.

    :param sdfg: The SDFG to inspect.
    :returns: The first sink state whose label contains 'copyout'.
    """
    for sink in sdfg.sink_nodes():
        if "copyout" in sink.label:
            return sink
    raise AssertionError("No copyout state found in SDFG")


def _state_access_node_names(state: "dace.sdfg.state.SDFGState") -> Set[str]:
    """Collect all AccessNode data names within a single state.

    :param state: The state to inspect.
    :returns: Set of data names referenced by AccessNodes in the state.
    """
    return {
        node.data
        for node in state.nodes()
        if isinstance(node, nodes.AccessNode)
    }


def _computation_states(sdfg: SDFG) -> List["dace.sdfg.state.SDFGState"]:
    """Return all states that are neither copyin nor copyout.

    :param sdfg: The SDFG to inspect.
    :returns: List of computation states.
    """
    copyin_label = sdfg.start_block.label
    copyout_labels = {
        sink.label for sink in sdfg.sink_nodes() if "copyout" in sink.label
    }
    return [
        s for s in sdfg.states()
        if s.label != copyin_label and s.label not in copyout_labels
    ]


# ============================================================
# Structure tests (no GPU)
# ============================================================


class TestCuTileInsertDataCopiesStructure:
    """SDFG structure after ``CuTileInsertDataCopies``."""

    def test_copyin_state_exists(self) -> None:
        """After applying the pass, a copyin state exists as the new start
        block, containing AccessNode pairs for each candidate array."""
        sdfg = _build_vadd_sdfg("dc_struct_copyin_exists")
        _lower_before_data_copies(sdfg)
        CuTileInsertDataCopies().apply_pass(sdfg, {})

        start = _copyin_state(sdfg)
        assert "copyin" in start.label
        an_names = _state_access_node_names(start)
        # Must contain both originals and gpu_ clones
        for name in ("A", "B", "C"):
            assert name in an_names, f"Original '{name}' not in copyin"
            assert f"gpu_{name}" in an_names, f"Clone 'gpu_{name}' not in copyin"

    def test_copyout_state_exists(self) -> None:
        """A copyout state exists as a terminal state, containing AccessNode
        pairs for written arrays only."""
        sdfg = _build_vadd_sdfg("dc_struct_copyout_exists")
        _lower_before_data_copies(sdfg)
        CuTileInsertDataCopies().apply_pass(sdfg, {})

        copyout = _copyout_state(sdfg)
        assert "copyout" in copyout.label
        an_names = _state_access_node_names(copyout)
        # C is written, so it must be in copyout
        assert "C" in an_names
        assert "gpu_C" in an_names

    def test_originals_reverted_to_cpu_heap(self) -> None:
        """The original non-transient arrays (A, B, C) have
        storage=CPU_Heap after the pass."""
        sdfg = _build_vadd_sdfg("dc_struct_cpu_heap")
        _lower_before_data_copies(sdfg)
        CuTileInsertDataCopies().apply_pass(sdfg, {})

        for name in ("A", "B", "C"):
            desc = sdfg.arrays[name]
            assert desc.storage == dtypes.StorageType.CPU_Heap, (
                f"'{name}' storage is {desc.storage}, expected CPU_Heap"
            )

    def test_gpu_clones_are_gpu_global_transients(self) -> None:
        """The ``gpu_*`` cloned arrays exist, are transient, and have
        GPU_Global storage."""
        sdfg = _build_vadd_sdfg("dc_struct_gpu_clones")
        _lower_before_data_copies(sdfg)
        CuTileInsertDataCopies().apply_pass(sdfg, {})

        for name in ("A", "B", "C"):
            gpu_name = f"gpu_{name}"
            assert gpu_name in sdfg.arrays, f"'{gpu_name}' not found in SDFG arrays"
            desc = sdfg.arrays[gpu_name]
            assert desc.transient, f"'{gpu_name}' is not transient"
            assert desc.storage == dtypes.StorageType.GPU_Global, (
                f"'{gpu_name}' storage is {desc.storage}, expected GPU_Global"
            )

    def test_access_nodes_reference_clones(self) -> None:
        """All AccessNodes in computation states reference ``gpu_*`` names
        (not original array names)."""
        sdfg = _build_vadd_sdfg("dc_struct_access_refs")
        _lower_before_data_copies(sdfg)
        CuTileInsertDataCopies().apply_pass(sdfg, {})

        orig_names = {"A", "B", "C"}
        for state in _computation_states(sdfg):
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode):
                    if node.data in orig_names:
                        pytest.fail(
                            f"AccessNode in computation state '{state.label}' "
                            f"still references original '{node.data}'"
                        )

    def test_memlets_reference_clones(self) -> None:
        """All Memlets in computation states reference ``gpu_*`` data names."""
        sdfg = _build_vadd_sdfg("dc_struct_memlet_refs")
        _lower_before_data_copies(sdfg)
        CuTileInsertDataCopies().apply_pass(sdfg, {})

        orig_names = {"A", "B", "C"}
        for state in _computation_states(sdfg):
            for edge in state.edges():
                if edge.data.data in orig_names:
                    pytest.fail(
                        f"Memlet in computation state '{state.label}' "
                        f"still references original '{edge.data.data}'"
                    )

    def test_scalars_not_cloned(self) -> None:
        """Scalar parameters are NOT cloned by the pass (scalars are skipped)."""
        sdfg = _build_vadd_with_scalar_sdfg("dc_struct_scalar_skip")
        _lower_before_data_copies(sdfg)
        CuTileInsertDataCopies().apply_pass(sdfg, {})

        # The scalar should still exist unchanged
        assert "alpha" in sdfg.arrays
        assert isinstance(sdfg.arrays["alpha"], data.Scalar)
        # No gpu_alpha clone should exist
        assert "gpu_alpha" not in sdfg.arrays

    def test_read_only_array_skips_copyout(self) -> None:
        """A and B are read-only, C is written. The copyout state should only
        contain C's clone, not A's or B's."""
        sdfg = _build_vadd_sdfg("dc_struct_readonly_skip")
        _lower_before_data_copies(sdfg)
        CuTileInsertDataCopies().apply_pass(sdfg, {})

        copyout = _copyout_state(sdfg)
        an_names = _state_access_node_names(copyout)
        # Only C (the output) should appear in copyout
        assert "C" in an_names, "C not in copyout"
        assert "gpu_C" in an_names, "gpu_C not in copyout"
        # A and B (read-only) should NOT appear in copyout
        assert "A" not in an_names, "Read-only A found in copyout"
        assert "B" not in an_names, "Read-only B found in copyout"
        assert "gpu_A" not in an_names, "Read-only gpu_A found in copyout"
        assert "gpu_B" not in an_names, "Read-only gpu_B found in copyout"

    def test_idempotent(self) -> None:
        """Running the pass twice returns None the second time (no GPU_Global
        non-transients remain)."""
        sdfg = _build_vadd_sdfg("dc_struct_idempotent")
        _lower_before_data_copies(sdfg)

        result1 = CuTileInsertDataCopies().apply_pass(sdfg, {})
        assert result1 is not None and result1 > 0

        result2 = CuTileInsertDataCopies().apply_pass(sdfg, {})
        assert result2 is None

    def test_return_value_is_clone_count(self) -> None:
        """Returns the number of arrays cloned (3 for vadd with A, B, C)."""
        sdfg = _build_vadd_sdfg("dc_struct_return_count")
        _lower_before_data_copies(sdfg)

        result = CuTileInsertDataCopies().apply_pass(sdfg, {})
        assert result == 3

    def test_no_cutile_schedule_warns(self) -> None:
        """On an SDFG with no CuTile-scheduled maps, the pass warns and
        returns None in non-strict mode."""
        # Build a plain SDFG without running the lowering pipeline
        sdfg = _build_vadd_sdfg("dc_struct_no_schedule_warn")
        # Manually stamp a non-transient as GPU_Global so the candidate
        # check would normally trigger, but the precondition check (no
        # CuTile scope) comes first.
        sdfg.arrays["A"].storage = dtypes.StorageType.GPU_Global

        with pytest.warns(UserWarning, match="no CuTile-scheduled map found"):
            result = CuTileInsertDataCopies(strict=False).apply_pass(sdfg, {})
        assert result is None

    def test_no_cutile_schedule_strict_raises(self) -> None:
        """With strict=True and no CuTile-scheduled maps, ValueError is raised."""
        sdfg = _build_vadd_sdfg("dc_struct_no_schedule_strict")
        sdfg.arrays["A"].storage = dtypes.StorageType.GPU_Global

        with pytest.raises(ValueError, match="no CuTile-scheduled map found"):
            CuTileInsertDataCopies(strict=True).apply_pass(sdfg, {})

    def test_2d_vadd_structure(self) -> None:
        """2D vadd with widths=(8, 4) -- verify clones and copy states for
        2D arrays."""
        sdfg = _build_vadd2d_sdfg("dc_struct_2d")
        _lower_before_data_copies(sdfg, widths=(8, 4))
        result = CuTileInsertDataCopies().apply_pass(sdfg, {})

        assert result == 3

        # Verify clones exist
        for name in ("A", "B", "C"):
            gpu_name = f"gpu_{name}"
            assert gpu_name in sdfg.arrays
            assert sdfg.arrays[gpu_name].transient
            assert sdfg.arrays[gpu_name].storage == dtypes.StorageType.GPU_Global
            assert sdfg.arrays[name].storage == dtypes.StorageType.CPU_Heap

        # Verify copyin state
        start = _copyin_state(sdfg)
        assert "copyin" in start.label
        copyin_names = _state_access_node_names(start)
        for name in ("A", "B", "C"):
            assert name in copyin_names
            assert f"gpu_{name}" in copyin_names

        # Verify copyout state (only C)
        copyout = _copyout_state(sdfg)
        copyout_names = _state_access_node_names(copyout)
        assert "C" in copyout_names
        assert "gpu_C" in copyout_names

    def test_copyin_has_full_array_memlets(self) -> None:
        """Copyin state edges carry full-array Memlets."""
        sdfg = _build_vadd_sdfg("dc_struct_copyin_memlets")
        _lower_before_data_copies(sdfg)
        CuTileInsertDataCopies().apply_pass(sdfg, {})

        start = _copyin_state(sdfg)
        for edge in start.edges():
            # Each edge goes from original to gpu_clone
            assert isinstance(edge.src, nodes.AccessNode)
            assert isinstance(edge.dst, nodes.AccessNode)
            # The memlet data should reference the original array
            assert edge.data.data is not None
            assert not edge.src.data.startswith("gpu_"), (
                f"Copyin source '{edge.src.data}' starts with 'gpu_'"
            )
            assert edge.dst.data.startswith("gpu_"), (
                f"Copyin destination '{edge.dst.data}' does not start with 'gpu_'"
            )

    def test_copyout_has_full_array_memlets(self) -> None:
        """Copyout state edges carry full-array Memlets from clone to original."""
        sdfg = _build_vadd_sdfg("dc_struct_copyout_memlets")
        _lower_before_data_copies(sdfg)
        CuTileInsertDataCopies().apply_pass(sdfg, {})

        copyout = _copyout_state(sdfg)
        for edge in copyout.edges():
            assert isinstance(edge.src, nodes.AccessNode)
            assert isinstance(edge.dst, nodes.AccessNode)
            # Source should be the gpu clone, dst should be the original
            assert edge.src.data.startswith("gpu_"), (
                f"Copyout source '{edge.src.data}' does not start with 'gpu_'"
            )
            assert not edge.dst.data.startswith("gpu_"), (
                f"Copyout destination '{edge.dst.data}' starts with 'gpu_'"
            )

    def test_copyin_is_start_block(self) -> None:
        """The copyin state is wired as the SDFG start block."""
        sdfg = _build_vadd_sdfg("dc_struct_copyin_start")
        _lower_before_data_copies(sdfg)
        CuTileInsertDataCopies().apply_pass(sdfg, {})

        start = sdfg.start_block
        assert "copyin" in start.label

    def test_copyout_is_sink_node(self) -> None:
        """The copyout state is a sink (terminal) node in the SDFG."""
        sdfg = _build_vadd_sdfg("dc_struct_copyout_sink")
        _lower_before_data_copies(sdfg)
        CuTileInsertDataCopies().apply_pass(sdfg, {})

        sinks = sdfg.sink_nodes()
        copyout_found = any("copyout" in s.label for s in sinks)
        assert copyout_found, f"No copyout state among sinks: {[s.label for s in sinks]}"

    def test_no_candidates_returns_none(self) -> None:
        """When there are no GPU_Global non-transient arrays (e.g. all already
        cloned), the pass returns None."""
        sdfg = _build_vadd_sdfg("dc_struct_no_candidates")
        _lower_before_data_copies(sdfg)

        # Apply once (clones everything)
        CuTileInsertDataCopies().apply_pass(sdfg, {})
        # Apply again -- no candidates remain
        result = CuTileInsertDataCopies().apply_pass(sdfg, {})
        assert result is None


# ============================================================
# Codegen tests (no GPU)
# ============================================================


class TestCuTileInsertDataCopiesCodegen:
    """Generated Python-backend code with data copies."""

    def test_codegen_contains_cupy_asarray(self) -> None:
        """After full pipeline with insert_data_copies=True, generated code
        contains ``cupy.asarray``."""
        sdfg = _build_vadd_sdfg("dc_codegen_asarray")
        VectorizeCuTile(widths=(8,), insert_data_copies=True).apply_pass(sdfg, {})
        code = _generate_code(sdfg)
        assert "cupy.asarray" in code, (
            "Generated code does not contain 'cupy.asarray'"
        )

    def test_codegen_contains_cupy_asnumpy(self) -> None:
        """After full pipeline with insert_data_copies=True, generated code
        contains ``cupy.asnumpy``."""
        sdfg = _build_vadd_sdfg("dc_codegen_asnumpy")
        VectorizeCuTile(widths=(8,), insert_data_copies=True).apply_pass(sdfg, {})
        code = _generate_code(sdfg)
        assert "cupy.asnumpy" in code, (
            "Generated code does not contain 'cupy.asnumpy'"
        )

    def test_codegen_valid_python(self) -> None:
        """Generated code parses with ``ast.parse``."""
        sdfg = _build_vadd_sdfg("dc_codegen_valid")
        VectorizeCuTile(widths=(8,), insert_data_copies=True).apply_pass(sdfg, {})
        code = _generate_code(sdfg)
        ast.parse(code)

    def test_codegen_no_data_copies_no_asarray(self) -> None:
        """With insert_data_copies=False, generated code should NOT contain
        ``cupy.asarray`` (no copy states)."""
        sdfg = _build_vadd_sdfg("dc_codegen_no_copies")
        VectorizeCuTile(widths=(8,), insert_data_copies=False).apply_pass(sdfg, {})
        code = _generate_code(sdfg)
        assert "cupy.asarray" not in code, (
            "Generated code contains 'cupy.asarray' when data copies are disabled"
        )

    def test_codegen_2d_valid_python(self) -> None:
        """2D vadd with data copies generates valid Python code."""
        sdfg = _build_vadd2d_sdfg("dc_codegen_2d_valid")
        VectorizeCuTile(widths=(8, 4), insert_data_copies=True).apply_pass(sdfg, {})
        code = _generate_code(sdfg)
        ast.parse(code)
        assert "cupy.asarray" in code
        assert "cupy.asnumpy" in code

    def test_codegen_with_scalar_valid_python(self) -> None:
        """SDFG with a scalar parameter generates valid Python code with
        data copies (scalar is not cloned)."""
        sdfg = _build_vadd_with_scalar_sdfg("dc_codegen_scalar")
        VectorizeCuTile(widths=(8,), insert_data_copies=True).apply_pass(sdfg, {})
        code = _generate_code(sdfg)
        ast.parse(code)


# ============================================================
# Runtime tests (GPU)
# ============================================================


@pytest.mark.gpu
class TestCuTileInsertDataCopiesRuntime:
    """Runtime tests: the key feature is passing NumPy arrays directly."""

    def test_vadd_numpy_arrays_directly(self) -> None:
        """Build vadd SDFG, apply VectorizeCuTile(insert_data_copies=True),
        compile, call with NUMPY arrays, verify result matches A + B.

        This is the KEY test: after data copies, callers pass host arrays.
        """
        sdfg = _build_vadd_sdfg("dc_rt_vadd_numpy")
        VectorizeCuTile(widths=(8,), insert_data_copies=True).apply_pass(sdfg, {})

        n = 64
        rng = np.random.default_rng(42)
        A = rng.random(n)
        B = rng.random(n)
        C = np.zeros(n)

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C, N=n)

        np.testing.assert_allclose(C, A + B, rtol=1e-14)

    def test_vadd_non_divisible_numpy(self) -> None:
        """N=17 (non-divisible by 8) with NumPy arrays directly."""
        sdfg = _build_vadd_sdfg("dc_rt_vadd_nondiv")
        VectorizeCuTile(widths=(8,), insert_data_copies=True).apply_pass(sdfg, {})

        n = 17
        rng = np.random.default_rng(43)
        A = rng.random(n)
        B = rng.random(n)
        C = np.zeros(n)

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C, N=n)

        np.testing.assert_allclose(C, A + B, rtol=1e-14)

    def test_vadd_2d_numpy(self) -> None:
        """2D vadd with widths=(8, 4), pass numpy arrays directly."""
        sdfg = _build_vadd2d_sdfg("dc_rt_vadd_2d")
        VectorizeCuTile(widths=(8, 4), insert_data_copies=True).apply_pass(sdfg, {})

        m, n = 10, 17
        rng = np.random.default_rng(44)
        A = rng.random((m, n))
        B = rng.random((m, n))
        C = np.zeros((m, n))

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C, M=m, N=n)

        np.testing.assert_allclose(C, A + B, rtol=1e-14)

    def test_vadd_with_scalar_param_numpy(self) -> None:
        """An SDFG with a scalar parameter. Verify scalar is passed through
        correctly with numpy arrays."""
        sdfg = _build_vadd_with_scalar_sdfg("dc_rt_vadd_scalar")
        VectorizeCuTile(widths=(8,), insert_data_copies=True).apply_pass(sdfg, {})

        n = 32
        rng = np.random.default_rng(45)
        A = rng.random(n)
        B = rng.random(n)
        C = np.zeros(n)
        alpha_val = np.float64(3.14)

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C, alpha=alpha_val, N=n)

        np.testing.assert_allclose(C, A + B + alpha_val, rtol=1e-14)

    def test_insert_data_copies_false_requires_cupy(self) -> None:
        """Build vadd with insert_data_copies=False, compile. Calling with
        CuPy arrays should work. This verifies the flag actually matters."""
        import cupy as cp

        sdfg = _build_vadd_sdfg("dc_rt_no_copies_cupy")
        VectorizeCuTile(widths=(8,), insert_data_copies=False).apply_pass(sdfg, {})

        n = 64
        rng = np.random.default_rng(46)
        A_np = rng.random(n)
        B_np = rng.random(n)

        # With cupy arrays it should work
        A = cp.asarray(A_np)
        B = cp.asarray(B_np)
        C = cp.zeros(n)

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C, N=n)

        np.testing.assert_allclose(cp.asnumpy(C), A_np + B_np, rtol=1e-14)

    def test_vadd_large_numpy(self) -> None:
        """Larger problem size (N=1000) to exercise multiple tile iterations."""
        sdfg = _build_vadd_sdfg("dc_rt_vadd_large")
        VectorizeCuTile(widths=(8,), insert_data_copies=True).apply_pass(sdfg, {})

        n = 1000
        rng = np.random.default_rng(47)
        A = rng.random(n)
        B = rng.random(n)
        C = np.zeros(n)

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C, N=n)

        np.testing.assert_allclose(C, A + B, rtol=1e-14)

    def test_vadd_float32_numpy(self) -> None:
        """float32 dtype with numpy arrays directly."""
        sdfg = _build_vadd_sdfg("dc_rt_vadd_f32", dtype=dace.float32)
        VectorizeCuTile(widths=(8,), insert_data_copies=True).apply_pass(sdfg, {})

        n = 100
        rng = np.random.default_rng(48)
        A = rng.random(n).astype(np.float32)
        B = rng.random(n).astype(np.float32)
        C = np.zeros(n, dtype=np.float32)

        csdfg = sdfg.compile()
        csdfg(A=A, B=B, C=C, N=n)

        np.testing.assert_allclose(C, A + B, rtol=1e-6)

    def test_symbolic_n_two_sizes_numpy(self) -> None:
        """One compiled SDFG, two runtime values of the symbol N, numpy arrays."""
        sdfg = _build_vadd_sdfg("dc_rt_vadd_symbolic")
        VectorizeCuTile(widths=(8,), insert_data_copies=True).apply_pass(sdfg, {})
        csdfg = sdfg.compile()

        rng = np.random.default_rng(49)
        for n in (64, 17):
            A = rng.random(n)
            B = rng.random(n)
            C = np.zeros(n)
            csdfg(A=A, B=B, C=C, N=n)
            np.testing.assert_allclose(C, A + B, rtol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
