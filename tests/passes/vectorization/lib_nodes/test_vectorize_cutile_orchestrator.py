# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Integration tests for the ``VectorizeCuTile`` orchestrator pass.

Two layers:

* **Structure / codegen tests (no GPU):** after
  ``VectorizeCuTile(widths=...).apply_pass(sdfg, {})`` the SDFG must carry no
  unexpanded tileops library nodes, use the Python backend, and generate
  ``cuda.tile`` code (``import cuda.tile``, a ``@ct.kernel`` function,
  ``ct.load(`` — masked stores under the default ``full_mask`` remainder
  strategy emit ``ct.scatter(`` rather than ``ct.store(``). The orchestrator
  API surface (eager config validation, ``strict`` forwarding) is covered here
  too.
* **Runtime tests (``@pytest.mark.gpu``):** compile and run on GPU with CuPy
  arrays and compare against NumPy references — aligned and non-divisible
  sizes (masked remainder), K=2 ``(8, 4)`` 2-D kernels, symbolic sizes,
  the ``TileIR/vectorized_pipeline/strides.py`` kernel, dtype crosses, and a
  mixed SDFG (cuTile kernel + non-kernel host array).

Note: a reduction runtime fixture is deliberately absent — ``np.sum`` with
``widths=(8, 4)`` currently hits a ``MarkTileDims`` failure and with
``widths=(8,)`` no ``TileReduce`` is emitted (see the Step 5 notes in the
implementation log); per-node reduction coverage lives in the tileops
library-node tests.
"""
import ast
from typing import Dict, Sequence, Tuple, Union

import numpy as np
import pytest

import dace
from dace import dtypes, symbolic
from dace.sdfg import SDFG, nodes
from dace.transformation.passes.vectorization import VectorizeCuTile


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
    sdfg.add_array("A", (N, ), dtype)
    sdfg.add_array("B", (N, ), dtype)
    sdfg.add_array("C", (N, ), dtype)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "add",
        {"i": "0:N"},
        {
            "_a": dace.Memlet("A[i]"),
            "_b": dace.Memlet("B[i]")
        },
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
        {
            "i": "0:M",
            "j": "0:N"
        },
        {
            "_a": dace.Memlet("A[i, j]"),
            "_b": dace.Memlet("B[i, j]")
        },
        "_c = _a + _b",
        {"_c": dace.Memlet("C[i, j]")},
        external_edges=True,
    )
    return sdfg


@dace.program
def _strides_prog(A: dace.float32[128, 128], B: dace.float32[128, 128], C: dace.float32[64, 128]):
    for i in range(64):
        for j in range(128):
            C[i, j] = A[i * 2, j] + B[i * 2, j]


def _build_strides_sdfg(name: str) -> SDFG:
    """The ``TileIR/vectorized_pipeline/strides.py`` kernel.

    ``C[i, j] = A[i*2, j] + B[i*2, j]`` over concrete 128x128 inputs — the
    pre-existing ``i*2`` stride must not confuse the anchor-driven schedule
    stamping.

    :param name: SDFG name (must be unique per test).
    :returns: The constructed SDFG.
    """
    sdfg = _strides_prog.to_sdfg()
    sdfg.name = name
    return sdfg


def _build_mixed_sdfg(name: str) -> SDFG:
    """A vadd kernel state plus an independent host state touching only D.

    The second state holds a bare (un-mapped) tasklet ``D[0] += 5.0`` — the
    vectorizer leaves it alone, so after the pipeline the SDFG mixes a cuTile
    kernel (A, B, C — GPU_Global) with host work (D — never stamped).

    :param name: SDFG name (must be unique per test).
    :returns: The constructed SDFG.
    """
    sdfg = _build_vadd_sdfg(name)
    sdfg.add_array("D", (1, ), dace.float64)
    host_state = sdfg.add_state_after(list(sdfg.states())[-1], "host")
    tasklet = host_state.add_tasklet("bump", {"_d"}, {"_o"}, "_o = _d + 5.0")
    host_state.add_edge(host_state.add_access("D"), None, tasklet, "_d", dace.Memlet("D[0]"))
    host_state.add_edge(tasklet, "_o", host_state.add_access("D"), None, dace.Memlet("D[0]"))
    return sdfg


def _build_no_anchor_sdfg(name: str) -> SDFG:
    """An SDFG the vectorizer cannot vectorize (a bare tasklet, no maps).

    Running ``VectorizeCuTile`` on it yields zero tileops anchors, exercising
    the warn-vs-strict behavior of the lowering passes.

    :param name: SDFG name (must be unique per test).
    :returns: The constructed SDFG.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (1, ), dace.float64)
    state = sdfg.add_state("main")
    tasklet = state.add_tasklet("set", {}, {"_o"}, "_o = 1.0")
    state.add_edge(tasklet, "_o", state.add_access("A"), None, dace.Memlet("A[0]"))
    return sdfg


# ============================================================
# Inspection / execution helpers
# ============================================================


def _generate_code(sdfg: SDFG) -> str:
    """Generate Python-backend code for ``sdfg`` and return the main file text.

    :param sdfg: The (already lowered) SDFG.
    :returns: The generated code of the main code object.
    """
    code_objects = sdfg.generate_code()
    return next(co for co in code_objects if co.name == sdfg.name).code


def _library_nodes(sdfg: SDFG) -> Sequence[nodes.LibraryNode]:
    """All remaining (unexpanded) library nodes in ``sdfg``, recursively.

    :param sdfg: The SDFG to inspect.
    :returns: List of library nodes.
    """
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.LibraryNode)]


def _cutile_map_entries(sdfg: SDFG) -> Sequence[nodes.MapEntry]:
    """All MapEntry nodes with ``ScheduleType.CuTile``, recursively.

    :param sdfg: The SDFG to inspect.
    :returns: List of CuTile-scheduled map entries.
    """
    return [
        n for n, _ in sdfg.all_nodes_recursive()
        if isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.CuTile
    ]


def _run_cutile(sdfg: SDFG, host_arrays: Tuple[str, ...] = (), **kwargs) -> Dict[str, Union[np.ndarray, int]]:
    """Compile and run a cuTile SDFG, returning results as numpy arrays.

    NumPy array arguments are converted to CuPy for GPU execution (except
    those named in ``host_arrays``, which stay on the host) and converted
    back afterwards.

    :param sdfg: The SDFG to compile and run.
    :param host_arrays: Names of arrays that must remain NumPy (host-side).
    :param kwargs: Named arguments for the SDFG (arrays and symbols).
    :returns: Dictionary mapping argument names to results.
    """
    import cupy as cp

    cp_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray) and k not in host_arrays:
            cp_kwargs[k] = cp.asarray(v)
        else:
            cp_kwargs[k] = v

    csdfg = sdfg.compile()
    csdfg(**cp_kwargs)

    results = {}
    for k, v in cp_kwargs.items():
        if isinstance(v, cp.ndarray):
            results[k] = cp.asnumpy(v)
        else:
            results[k] = v
    return results


# ============================================================
# Structure tests (no GPU)
# ============================================================


class TestOrchestratorStructure:
    """SDFG structure after a single ``apply_pass`` call."""

    def test_returns_kernel_count(self):
        """apply_pass returns the number of cuTile kernels created (1)."""
        sdfg = _build_vadd_sdfg("vcutile_struct_ret")
        assert VectorizeCuTile(widths=(8, ), insert_data_copies=False).apply_pass(sdfg, {}) == 1

    def test_backend_is_python(self):
        """The pipeline stamps the Python backend."""
        sdfg = _build_vadd_sdfg("vcutile_struct_backend")
        VectorizeCuTile(widths=(8, ), insert_data_copies=False).apply_pass(sdfg, {})
        assert sdfg.backend == dtypes.BackendLanguage.Python

    def test_single_cutile_map_with_tile_step(self):
        """Exactly one CuTile-scheduled map exists, stepped by the width."""
        sdfg = _build_vadd_sdfg("vcutile_struct_schedule")
        VectorizeCuTile(widths=(8, ), insert_data_copies=False).apply_pass(sdfg, {})
        cutile_maps = _cutile_map_entries(sdfg)
        assert len(cutile_maps) == 1
        assert any(str(step) == "8" for _, _, step in cutile_maps[0].map.range)

    def test_globals_are_gpu_global(self):
        """Kernel-touched non-transients become GPU_Global."""
        sdfg = _build_vadd_sdfg("vcutile_struct_globals")
        VectorizeCuTile(widths=(8, ), insert_data_copies=False).apply_pass(sdfg, {})
        for name in ("A", "B", "C"):
            assert sdfg.arrays[name].storage == dtypes.StorageType.GPU_Global, f"{name} not GPU_Global"

    def test_tile_transients_are_cutile_tile(self):
        """At least one transient carries CuTile_Tile storage."""
        sdfg = _build_vadd_sdfg("vcutile_struct_tiles")
        VectorizeCuTile(widths=(8, ), insert_data_copies=False).apply_pass(sdfg, {})
        assert any(desc.transient and desc.storage == dtypes.StorageType.CuTile_Tile
                   for desc in sdfg.arrays.values()), "No CuTile_Tile transient after pipeline"

    def test_k2_structure(self):
        """K=2 widths (8, 4): one 2-D CuTile map stepped (8, 4), all expanded."""
        sdfg = _build_vadd2d_sdfg("vcutile_struct_k2")
        assert VectorizeCuTile(widths=(8, 4), insert_data_copies=False).apply_pass(sdfg, {}) == 1
        cutile_maps = _cutile_map_entries(sdfg)
        assert len(cutile_maps) == 1
        steps = [str(step) for _, _, step in cutile_maps[0].map.range]
        assert steps == ["8", "4"]
        assert _library_nodes(sdfg) == []
        assert sdfg.backend == dtypes.BackendLanguage.Python

    def test_strides_only_anchored_map_is_cutile(self):
        """strides.py kernel: the pre-existing i*2 stride never triggers a
        CuTile stamp on its own — only the anchored tiled map is stamped."""
        sdfg = _build_strides_sdfg("vcutile_struct_strides")
        assert VectorizeCuTile(widths=(8, 4), insert_data_copies=False).apply_pass(sdfg, {}) == 1
        assert len(_cutile_map_entries(sdfg)) == 1
        assert _library_nodes(sdfg) == []

    def test_mixed_sdfg_host_array_not_stamped(self):
        """Mixed SDFG: kernel operands GPU_Global, host-only D untouched."""
        sdfg = _build_mixed_sdfg("vcutile_struct_mixed")
        assert VectorizeCuTile(widths=(8, ), insert_data_copies=False).apply_pass(sdfg, {}) == 1
        for name in ("A", "B", "C"):
            assert sdfg.arrays[name].storage == dtypes.StorageType.GPU_Global, f"{name} not GPU_Global"
        assert sdfg.arrays["D"].storage != dtypes.StorageType.GPU_Global


# ============================================================
# Codegen tests (no GPU)
# ============================================================


class TestOrchestratorCodegen:
    """Generated Python-backend code after the orchestrator."""

    def test_code_is_valid_python(self):
        """Generated code parses with ast.parse."""
        sdfg = _build_vadd_sdfg("vcutile_code_valid")
        VectorizeCuTile(widths=(8, ), insert_data_copies=False).apply_pass(sdfg, {})
        ast.parse(_generate_code(sdfg))

    def test_code_imports_cuda_tile(self):
        """Generated code imports cuda.tile."""
        sdfg = _build_vadd_sdfg("vcutile_code_import")
        VectorizeCuTile(widths=(8, ), insert_data_copies=False).apply_pass(sdfg, {})
        code = _generate_code(sdfg)
        assert "import cuda.tile as ct" in code

    def test_code_has_kernel_function_and_launch(self):
        """Generated code has a @ct.kernel function and a ct.launch call."""
        sdfg = _build_vadd_sdfg("vcutile_code_kernel")
        VectorizeCuTile(widths=(8, ), insert_data_copies=False).apply_pass(sdfg, {})
        code = _generate_code(sdfg)
        assert "@ct.kernel" in code
        kernel_idx = code.index("@ct.kernel")
        assert code.index("def ", kernel_idx) > kernel_idx
        assert "ct.launch(" in code
        assert "ct.bid(0)" in code

    def test_code_has_tile_load_and_store(self):
        """Generated code loads tiles; under the default full_mask remainder
        strategy with symbolic N the store is deterministically ct.scatter."""
        sdfg = _build_vadd_sdfg("vcutile_code_loadstore")
        VectorizeCuTile(widths=(8, ), insert_data_copies=False).apply_pass(sdfg, {})
        code = _generate_code(sdfg)
        assert "ct.load(" in code
        assert "ct.scatter(" in code

    def test_code_masks_against_symbolic_bound(self):
        """The iteration mask uses arange(width) compared against N."""
        sdfg = _build_vadd_sdfg("vcutile_code_mask")
        VectorizeCuTile(widths=(8, ), insert_data_copies=False).apply_pass(sdfg, {})
        code = _generate_code(sdfg)
        assert "ct.arange(8" in code
        assert "< N" in code

    def test_k2_codegen(self):
        """K=2 widths (8, 4): valid Python with cuTile primitives."""
        sdfg = _build_vadd2d_sdfg("vcutile_code_k2")
        VectorizeCuTile(widths=(8, 4), insert_data_copies=False).apply_pass(sdfg, {})
        code = _generate_code(sdfg)
        ast.parse(code)
        assert "import cuda.tile as ct" in code
        assert "@ct.kernel" in code
        assert "ct.load(" in code

    def test_strides_codegen(self):
        """The strides.py kernel generates valid cuTile code.

        The strided ``A[i*2, j]`` access lowers to masked ``ct.gather(``
        loads (not contiguous ``ct.load(``); the masked store to C is a
        ``ct.scatter(``.
        """
        sdfg = _build_strides_sdfg("vcutile_code_strides")
        VectorizeCuTile(widths=(8, 4), insert_data_copies=False).apply_pass(sdfg, {})
        code = _generate_code(sdfg)
        ast.parse(code)
        assert "import cuda.tile as ct" in code
        assert "@ct.kernel" in code
        assert "ct.gather(" in code
        assert "ct.scatter(" in code

    def test_int32_codegen(self):
        """int32 vadd generates valid cuTile code."""
        sdfg = _build_vadd_sdfg("vcutile_code_i32", dtype=dace.int32)
        VectorizeCuTile(widths=(8, ), insert_data_copies=False).apply_pass(sdfg, {})
        code = _generate_code(sdfg)
        ast.parse(code)
        assert "ct.load(" in code

    def test_mixed_sdfg_codegen(self):
        """The mixed SDFG generates valid code containing a kernel."""
        sdfg = _build_mixed_sdfg("vcutile_code_mixed")
        VectorizeCuTile(widths=(8, ), insert_data_copies=False).apply_pass(sdfg, {})
        code = _generate_code(sdfg)
        ast.parse(code)
        assert "@ct.kernel" in code

    def test_nest_map_bodies_codegen(self):
        """nest_map_bodies=True drives the map body through a NestedSDFG.

        The cuTile codegen emits NSDFGs as module-level functions, so this
        exercises the NSDFG-body path through the full orchestrator. The
        result must still be valid Python with cuTile primitives.
        """
        sdfg = _build_vadd_sdfg("vcutile_code_nest")
        VectorizeCuTile(widths=(8, ), nest_map_bodies=True, insert_data_copies=False).apply_pass(sdfg, {})
        code = _generate_code(sdfg)
        ast.parse(code)
        assert "import cuda.tile as ct" in code
        assert "@ct.kernel" in code
        assert "ct.load(" in code


# ============================================================
# API surface tests (no GPU)
# ============================================================


class TestAPISurface:
    """Eager config validation and ``strict`` forwarding."""

    def test_widths_length_4_raises_eagerly(self):
        """K=4 widths are rejected at construction (not at apply_pass)."""
        with pytest.raises(NotImplementedError):
            VectorizeCuTile(widths=(8, 8, 8, 8))

    def test_widths_empty_raises_eagerly(self):
        """Zero-length widths are rejected at construction."""
        with pytest.raises(NotImplementedError):
            VectorizeCuTile(widths=())

    def test_non_power_of_two_width_raises_eagerly(self):
        """Non-power-of-2 widths are rejected at construction."""
        with pytest.raises(NotImplementedError):
            VectorizeCuTile(widths=(6, ))

    def test_bad_remainder_strategy_raises_eagerly(self):
        """An unknown remainder_strategy is rejected at construction."""
        with pytest.raises(NotImplementedError):
            VectorizeCuTile(widths=(8, ), remainder_strategy="bogus")

    def test_zero_anchor_warns_and_returns_none(self):
        """Non-strict: an unvectorizable SDFG warns and returns None; the
        pipeline still runs to completion (Python backend stamped)."""
        sdfg = _build_no_anchor_sdfg("vcutile_api_zero_anchor_warn")
        with pytest.warns(UserWarning, match="no tileops library nodes found"):
            assert VectorizeCuTile(widths=(8, ), insert_data_copies=False).apply_pass(sdfg, {}) is None
        assert sdfg.backend == dtypes.BackendLanguage.Python

    def test_zero_anchor_strict_raises(self):
        """strict=True forwards to the lowering passes: ValueError."""
        sdfg = _build_no_anchor_sdfg("vcutile_api_zero_anchor_strict")
        with pytest.raises(ValueError, match="no tileops library nodes found"):
            VectorizeCuTile(widths=(8, ), strict=True, insert_data_copies=False).apply_pass(sdfg, {})

    def test_strict_success_path_returns_kernel_count(self):
        """strict=True on a cleanly vectorizable SDFG does NOT raise: the
        lowering passes find their tileops anchors and the pass returns the
        kernel count (1)."""
        sdfg = _build_vadd_sdfg("vcutile_api_strict_success")
        assert VectorizeCuTile(widths=(8, ), strict=True, insert_data_copies=False).apply_pass(sdfg, {}) == 1


# ============================================================
# Runtime tests (GPU)
# ============================================================


@pytest.mark.gpu
class TestRuntimeVadd:
    """1-D vadd kernels: aligned, masked remainder, symbolic N, dtypes."""

    def test_vadd_aligned(self):
        """N=64 aligned to width 8."""
        sdfg = _build_vadd_sdfg("vcutile_rt_aligned")
        VectorizeCuTile(widths=(8, ), insert_data_copies=False).apply_pass(sdfg, {})

        n = 64
        rng = np.random.default_rng(1)
        A, B, C = rng.random(n), rng.random(n), np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(results["C"], A + B, rtol=1e-14)

    def test_vadd_non_divisible(self):
        """N=17 not divisible by 8 (masked remainder)."""
        sdfg = _build_vadd_sdfg("vcutile_rt_unaligned")
        VectorizeCuTile(widths=(8, ), insert_data_copies=False).apply_pass(sdfg, {})

        n = 17
        rng = np.random.default_rng(2)
        A, B, C = rng.random(n), rng.random(n), np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(results["C"], A + B, rtol=1e-14)

    def test_symbolic_n_one_compile_two_sizes(self):
        """One compiled SDFG, two runtime values of the symbol N."""
        import cupy as cp

        sdfg = _build_vadd_sdfg("vcutile_rt_symbolic")
        VectorizeCuTile(widths=(8, ), insert_data_copies=False).apply_pass(sdfg, {})
        csdfg = sdfg.compile()

        rng = np.random.default_rng(3)
        for n in (64, 17):
            A, B = rng.random(n), rng.random(n)
            dA, dB, dC = cp.asarray(A), cp.asarray(B), cp.zeros(n)
            csdfg(A=dA, B=dB, C=dC, N=n)
            np.testing.assert_allclose(cp.asnumpy(dC), A + B, rtol=1e-14)

    def test_float32(self):
        """float32 dtype cross."""
        sdfg = _build_vadd_sdfg("vcutile_rt_f32", dtype=dace.float32)
        VectorizeCuTile(widths=(8, ), insert_data_copies=False).apply_pass(sdfg, {})

        n = 100
        rng = np.random.default_rng(4)
        A = rng.random(n).astype(np.float32)
        B = rng.random(n).astype(np.float32)
        C = np.zeros(n, dtype=np.float32)
        results = _run_cutile(sdfg, A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(results["C"], A + B, rtol=1e-6)

    def test_int32(self):
        """int32 dtype cross (exact comparison)."""
        sdfg = _build_vadd_sdfg("vcutile_rt_i32", dtype=dace.int32)
        VectorizeCuTile(widths=(8, ), insert_data_copies=False).apply_pass(sdfg, {})

        n = 100
        rng = np.random.default_rng(5)
        A = rng.integers(0, 1000, n, dtype=np.int32)
        B = rng.integers(0, 1000, n, dtype=np.int32)
        C = np.zeros(n, dtype=np.int32)
        results = _run_cutile(sdfg, A=A, B=B, C=C, N=n)
        np.testing.assert_array_equal(results["C"], A + B)


@pytest.mark.gpu
class TestRuntimeMultiDim:
    """K=2 kernels and the strides example."""

    def test_k2_aligned(self):
        """Widths (8, 4) on an aligned 16x32 problem."""
        sdfg = _build_vadd2d_sdfg("vcutile_rt_k2_aligned")
        VectorizeCuTile(widths=(8, 4), insert_data_copies=False).apply_pass(sdfg, {})

        m, n = 16, 32
        rng = np.random.default_rng(6)
        A, B, C = rng.random((m, n)), rng.random((m, n)), np.zeros((m, n))
        results = _run_cutile(sdfg, A=A, B=B, C=C, M=m, N=n)
        np.testing.assert_allclose(results["C"], A + B, rtol=1e-14)

    def test_k2_non_divisible(self):
        """Widths (8, 4) on a 10x17 problem (masked remainder in both dims)."""
        sdfg = _build_vadd2d_sdfg("vcutile_rt_k2_unaligned")
        VectorizeCuTile(widths=(8, 4), insert_data_copies=False).apply_pass(sdfg, {})

        m, n = 10, 17
        rng = np.random.default_rng(7)
        A, B, C = rng.random((m, n)), rng.random((m, n)), np.zeros((m, n))
        results = _run_cutile(sdfg, A=A, B=B, C=C, M=m, N=n)
        np.testing.assert_allclose(results["C"], A + B, rtol=1e-14)

    def test_strides_kernel(self):
        """The TileIR strides.py kernel: C[i, j] = A[i*2, j] + B[i*2, j]."""
        sdfg = _build_strides_sdfg("vcutile_rt_strides")
        VectorizeCuTile(widths=(8, 4), insert_data_copies=False).apply_pass(sdfg, {})

        rng = np.random.default_rng(8)
        A = rng.random((128, 128)).astype(np.float32)
        B = rng.random((128, 128)).astype(np.float32)
        C = np.zeros((64, 128), dtype=np.float32)
        results = _run_cutile(sdfg, A=A, B=B, C=C)
        np.testing.assert_allclose(results["C"], A[::2, :] + B[::2, :], rtol=1e-6)


@pytest.mark.gpu
class TestRuntimeMixed:
    """Mixed cuTile-kernel + host-state SDFG end-to-end."""

    def test_mixed_sdfg_end_to_end(self):
        """A/B/C run through the kernel on GPU; D stays a host NumPy array."""
        sdfg = _build_mixed_sdfg("vcutile_rt_mixed")
        VectorizeCuTile(widths=(8, ), insert_data_copies=False).apply_pass(sdfg, {})

        n = 24
        rng = np.random.default_rng(9)
        A, B, C = rng.random(n), rng.random(n), np.zeros(n)
        D = np.array([1.5])
        results = _run_cutile(sdfg, host_arrays=("D", ), A=A, B=B, C=C, D=D, N=n)
        np.testing.assert_allclose(results["C"], A + B, rtol=1e-14)
        np.testing.assert_allclose(results["D"], np.array([6.5]), rtol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
