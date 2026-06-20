# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for CuTile cross-storage CPU_Heap <-> GPU_Global copy support.

Verifies that CuTilePythonCodeGen correctly:
1. Registers copy dispatchers for CPU_Heap <-> GPU_Global copies.
2. Emits ``.set()`` for CPU_Heap -> GPU_Global transfers.
3. Emits ``.get(out=...)`` for GPU_Global -> CPU_Heap transfers.
4. Handles memlet subsets in cross-storage copies.
5. Falls through to plain assignment for same-storage copies.
"""
import pytest

import dace
from dace import dtypes, subsets
from dace.codegen import dispatcher as dispatcher_mod
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.codegen.py.cutile_target import CuTilePythonCodeGen
from dace.sdfg import nodes, SDFG
from dace.memlet import Memlet


def _make_cutile_python_sdfg(name: str) -> SDFG:
    """Create an SDFG with backend set to Python for cuTile testing."""
    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    return sdfg


class _StubFrameCodegen:
    """Minimal stub for the frame codegen, just providing a dispatcher."""

    def __init__(self):
        self.dispatcher = dispatcher_mod.TargetDispatcher(self)
        self._initcode = PythonCodeIOStream()
        self._exitcode = PythonCodeIOStream()


def _make_cutile_codegen(sdfg: SDFG) -> CuTilePythonCodeGen:
    """Instantiate a CuTilePythonCodeGen with a stub frame codegen."""
    frame = _StubFrameCodegen()
    codegen = CuTilePythonCodeGen(frame, sdfg)
    return codegen


# =============================================================================
# Dispatcher registration tests
# =============================================================================


class TestCrossStorageDispatcherRegistration:
    """Verify that __init__ registers cross-storage copy dispatchers."""

    def test_cpu_heap_to_gpu_global_registered(self):
        """CPU_Heap -> GPU_Global with schedule=None is registered."""
        sdfg = _make_cutile_python_sdfg("test_reg_cpu_gpu")
        frame = _StubFrameCodegen()
        codegen = CuTilePythonCodeGen(frame, sdfg)
        key = (dtypes.StorageType.CPU_Heap,
               dtypes.StorageType.GPU_Global,
               None)
        dispatcher = frame.dispatcher
        assert key in dispatcher._generic_copy_dispatchers
        assert dispatcher._generic_copy_dispatchers[key] is codegen

    def test_gpu_global_to_cpu_heap_registered(self):
        """GPU_Global -> CPU_Heap with schedule=None is registered."""
        sdfg = _make_cutile_python_sdfg("test_reg_gpu_cpu")
        frame = _StubFrameCodegen()
        codegen = CuTilePythonCodeGen(frame, sdfg)
        key = (dtypes.StorageType.GPU_Global,
               dtypes.StorageType.CPU_Heap,
               None)
        dispatcher = frame.dispatcher
        assert key in dispatcher._generic_copy_dispatchers
        assert dispatcher._generic_copy_dispatchers[key] is codegen

    def test_cutile_tile_dispatchers_still_registered(self):
        """Existing CuTile_Tile copy dispatchers are still present."""
        sdfg = _make_cutile_python_sdfg("test_reg_tile")
        frame = _StubFrameCodegen()
        codegen = CuTilePythonCodeGen(frame, sdfg)
        dispatcher = frame.dispatcher
        # CuTile_Tile <-> GPU_Global, schedule=None
        key = (dtypes.StorageType.GPU_Global,
               dtypes.StorageType.CuTile_Tile,
               None)
        assert key in dispatcher._generic_copy_dispatchers
        assert dispatcher._generic_copy_dispatchers[key] is codegen


# =============================================================================
# copy_memory -- cross-storage copy code generation tests
# =============================================================================


def _build_copy_sdfg_and_edge(
    name: str,
    src_name: str,
    dst_name: str,
    src_storage: dtypes.StorageType,
    dst_storage: dtypes.StorageType,
    shape: list,
    dtype: dace.typeclass = dace.float64,
    src_subset_str: str = None,
    dst_subset_str: str = None,
):
    """Build a minimal SDFG with two arrays and a copy edge.

    Returns (sdfg, codegen, state, src_node, dst_node, edge, callsite_stream).
    """
    sdfg = _make_cutile_python_sdfg(name)
    sdfg.add_array(src_name, shape, dtype, storage=src_storage)
    sdfg.add_array(dst_name, shape, dtype, storage=dst_storage, transient=True)
    state = sdfg.add_state("copy_state")
    src_node = state.add_read(src_name)
    dst_node = state.add_write(dst_name)

    # Build memlet
    if src_subset_str and dst_subset_str:
        memlet = Memlet(data=src_name, subset=src_subset_str,
                        other_subset=dst_subset_str)
    elif src_subset_str:
        memlet = Memlet(data=src_name, subset=src_subset_str)
    else:
        memlet = Memlet(data=src_name)

    edge = state.add_edge(src_node, None, dst_node, None, memlet)

    codegen = _make_cutile_codegen(sdfg)
    fn_stream = PythonCodeIOStream()
    cs_stream = PythonCodeIOStream()
    return sdfg, codegen, state, src_node, dst_node, edge, fn_stream, cs_stream


class TestCopyMemoryCPUToGPU:
    """Test copy_memory for CPU_Heap -> GPU_Global transfers."""

    def test_full_array_copy_cpu_to_gpu(self):
        """Full-array CPU_Heap -> GPU_Global emits .set()."""
        (sdfg, codegen, state, src, dst, edge,
         fn_stream, cs_stream) = _build_copy_sdfg_and_edge(
            "test_cpu_to_gpu_full",
            "A_cpu", "A_gpu",
            dtypes.StorageType.CPU_Heap,
            dtypes.StorageType.GPU_Global,
            [16],
        )
        codegen.copy_memory(sdfg, sdfg, state, 0, src, dst, edge,
                            fn_stream, cs_stream)
        code = cs_stream.getvalue()
        assert ".set(" in code
        assert "A_gpu[" in code  # dst has indexing
        assert "A_cpu" in code

    def test_full_array_copy_cpu_to_gpu_no_subset(self):
        """Full copy with auto-inferred subset uses [:] on destination."""
        (sdfg, codegen, state, src, dst, edge,
         fn_stream, cs_stream) = _build_copy_sdfg_and_edge(
            "test_cpu_to_gpu_no_sub",
            "X", "X_dev",
            dtypes.StorageType.CPU_Heap,
            dtypes.StorageType.GPU_Global,
            [32],
        )
        codegen.copy_memory(sdfg, sdfg, state, 0, src, dst, edge,
                            fn_stream, cs_stream)
        code = cs_stream.getvalue()
        # Memlet(data="X") auto-infers src_subset from array shape,
        # so the source expression will include the subset.
        assert ".set(" in code
        assert "X" in code
        assert "X_dev[" in code

    def test_subset_copy_cpu_to_gpu(self):
        """Subset copy CPU_Heap -> GPU_Global applies subset indexing."""
        (sdfg, codegen, state, src, dst, edge,
         fn_stream, cs_stream) = _build_copy_sdfg_and_edge(
            "test_cpu_to_gpu_sub",
            "B_cpu", "B_gpu",
            dtypes.StorageType.CPU_Heap,
            dtypes.StorageType.GPU_Global,
            [64],
            src_subset_str="0:32",
            dst_subset_str="0:32",
        )
        codegen.copy_memory(sdfg, sdfg, state, 0, src, dst, edge,
                            fn_stream, cs_stream)
        code = cs_stream.getvalue()
        assert ".set(" in code
        assert "B_cpu[" in code
        assert "B_gpu[" in code

    def test_2d_array_copy_cpu_to_gpu(self):
        """2D full-array CPU_Heap -> GPU_Global."""
        (sdfg, codegen, state, src, dst, edge,
         fn_stream, cs_stream) = _build_copy_sdfg_and_edge(
            "test_cpu_to_gpu_2d",
            "M_host", "M_dev",
            dtypes.StorageType.CPU_Heap,
            dtypes.StorageType.GPU_Global,
            [8, 16],
        )
        codegen.copy_memory(sdfg, sdfg, state, 0, src, dst, edge,
                            fn_stream, cs_stream)
        code = cs_stream.getvalue()
        assert ".set(" in code
        assert "M_host" in code
        assert "M_dev[" in code


class TestCopyMemoryGPUToCPU:
    """Test copy_memory for GPU_Global -> CPU_Heap transfers."""

    def test_full_array_copy_gpu_to_cpu(self):
        """Full-array GPU_Global -> CPU_Heap emits .get(out=...)."""
        (sdfg, codegen, state, src, dst, edge,
         fn_stream, cs_stream) = _build_copy_sdfg_and_edge(
            "test_gpu_to_cpu_full",
            "C_gpu", "C_cpu",
            dtypes.StorageType.GPU_Global,
            dtypes.StorageType.CPU_Heap,
            [16],
        )
        codegen.copy_memory(sdfg, sdfg, state, 0, src, dst, edge,
                            fn_stream, cs_stream)
        code = cs_stream.getvalue()
        assert ".get(out=" in code
        assert "C_cpu[" in code
        assert "C_gpu" in code

    def test_full_array_copy_gpu_to_cpu_no_subset(self):
        """Full copy with auto-inferred subset uses [:] on destination."""
        (sdfg, codegen, state, src, dst, edge,
         fn_stream, cs_stream) = _build_copy_sdfg_and_edge(
            "test_gpu_to_cpu_no_sub",
            "Y_dev", "Y",
            dtypes.StorageType.GPU_Global,
            dtypes.StorageType.CPU_Heap,
            [32],
        )
        codegen.copy_memory(sdfg, sdfg, state, 0, src, dst, edge,
                            fn_stream, cs_stream)
        code = cs_stream.getvalue()
        # Memlet auto-infers src_subset from array shape.
        assert "Y_dev" in code
        assert ".get(out=" in code
        assert "Y[" in code

    def test_subset_copy_gpu_to_cpu(self):
        """Subset copy GPU_Global -> CPU_Heap applies subset indexing."""
        (sdfg, codegen, state, src, dst, edge,
         fn_stream, cs_stream) = _build_copy_sdfg_and_edge(
            "test_gpu_to_cpu_sub",
            "D_gpu", "D_cpu",
            dtypes.StorageType.GPU_Global,
            dtypes.StorageType.CPU_Heap,
            [64],
            src_subset_str="4:20",
            dst_subset_str="4:20",
        )
        codegen.copy_memory(sdfg, sdfg, state, 0, src, dst, edge,
                            fn_stream, cs_stream)
        code = cs_stream.getvalue()
        assert ".get(out=" in code
        assert "D_gpu[" in code
        assert "D_cpu[" in code

    def test_2d_array_copy_gpu_to_cpu(self):
        """2D full-array GPU_Global -> CPU_Heap."""
        (sdfg, codegen, state, src, dst, edge,
         fn_stream, cs_stream) = _build_copy_sdfg_and_edge(
            "test_gpu_to_cpu_2d",
            "N_dev", "N_host",
            dtypes.StorageType.GPU_Global,
            dtypes.StorageType.CPU_Heap,
            [4, 8],
        )
        codegen.copy_memory(sdfg, sdfg, state, 0, src, dst, edge,
                            fn_stream, cs_stream)
        code = cs_stream.getvalue()
        assert "N_dev" in code
        assert ".get(out=" in code
        assert "N_host[" in code


# =============================================================================
# copy_memory -- same-storage fallback (existing behavior preserved)
# =============================================================================


class TestCopyMemorySameStorage:
    """Verify same-storage copies still use plain assignment."""

    def test_gpu_to_gpu_same_storage(self):
        """GPU_Global -> GPU_Global emits plain assignment, no cupy call."""
        (sdfg, codegen, state, src, dst, edge,
         fn_stream, cs_stream) = _build_copy_sdfg_and_edge(
            "test_gpu_to_gpu",
            "P_gpu", "Q_gpu",
            dtypes.StorageType.GPU_Global,
            dtypes.StorageType.GPU_Global,
            [16],
        )
        codegen.copy_memory(sdfg, sdfg, state, 0, src, dst, edge,
                            fn_stream, cs_stream)
        code = cs_stream.getvalue()
        assert "cupy.asarray" not in code
        assert "cupy.asnumpy" not in code
        assert ".set(" not in code
        assert ".get(out=" not in code
        # Should have a plain assignment
        assert "P_gpu" in code
        assert "Q_gpu" in code

    def test_cpu_to_cpu_same_storage(self):
        """CPU_Heap -> CPU_Heap emits plain assignment, no cupy call."""
        (sdfg, codegen, state, src, dst, edge,
         fn_stream, cs_stream) = _build_copy_sdfg_and_edge(
            "test_cpu_to_cpu",
            "R_cpu", "S_cpu",
            dtypes.StorageType.CPU_Heap,
            dtypes.StorageType.CPU_Heap,
            [16],
        )
        codegen.copy_memory(sdfg, sdfg, state, 0, src, dst, edge,
                            fn_stream, cs_stream)
        code = cs_stream.getvalue()
        assert "cupy.asarray" not in code
        assert "cupy.asnumpy" not in code
        assert ".set(" not in code
        assert ".get(out=" not in code
        assert "R_cpu" in code
        assert "S_cpu" in code


# =============================================================================
# copy_memory -- error handling
# =============================================================================


class TestCopyMemoryErrors:
    """Verify error handling in copy_memory."""

    def test_non_access_node_raises(self):
        """Non-AccessNode source or destination raises NotImplementedError."""
        sdfg = _make_cutile_python_sdfg("test_error_non_access")
        sdfg.add_array("A", [16], dace.float64,
                       storage=dtypes.StorageType.CPU_Heap)
        state = sdfg.add_state("s")
        a_node = state.add_read("A")
        tasklet = state.add_tasklet("t", {"inp"}, set(), "pass")
        edge = state.add_edge(a_node, None, tasklet, "inp",
                              Memlet(data="A"))

        codegen = _make_cutile_codegen(sdfg)
        fn_stream = PythonCodeIOStream()
        cs_stream = PythonCodeIOStream()

        with pytest.raises(NotImplementedError,
                           match="AccessNode-to-AccessNode"):
            codegen.copy_memory(sdfg, sdfg, state, 0, a_node, tasklet, edge,
                                fn_stream, cs_stream)


# =============================================================================
# copy_memory -- dtype variations
# =============================================================================


class TestCopyMemoryDtypes:
    """Verify cross-storage copies work with different data types."""

    @pytest.mark.parametrize("dtype", [
        dace.float32,
        dace.float64,
        dace.int32,
        dace.int64,
    ])
    def test_cpu_to_gpu_various_dtypes(self, dtype):
        """CPU -> GPU copy works for various numeric types."""
        (sdfg, codegen, state, src, dst, edge,
         fn_stream, cs_stream) = _build_copy_sdfg_and_edge(
            f"test_dtype_{dtype.as_numpy_dtype().name}",
            "arr_cpu", "arr_gpu",
            dtypes.StorageType.CPU_Heap,
            dtypes.StorageType.GPU_Global,
            [16],
            dtype=dtype,
        )
        codegen.copy_memory(sdfg, sdfg, state, 0, src, dst, edge,
                            fn_stream, cs_stream)
        code = cs_stream.getvalue()
        assert ".set(" in code
        assert "arr_cpu" in code

    @pytest.mark.parametrize("dtype", [
        dace.float32,
        dace.float64,
        dace.int32,
        dace.int64,
    ])
    def test_gpu_to_cpu_various_dtypes(self, dtype):
        """GPU -> CPU copy works for various numeric types."""
        (sdfg, codegen, state, src, dst, edge,
         fn_stream, cs_stream) = _build_copy_sdfg_and_edge(
            f"test_dtype_{dtype.as_numpy_dtype().name}_out",
            "arr_gpu", "arr_cpu",
            dtypes.StorageType.GPU_Global,
            dtypes.StorageType.CPU_Heap,
            [16],
            dtype=dtype,
        )
        codegen.copy_memory(sdfg, sdfg, state, 0, src, dst, edge,
                            fn_stream, cs_stream)
        code = cs_stream.getvalue()
        assert "arr_gpu" in code
        assert ".get(out=" in code


# =============================================================================
# _emit_cross_storage_copy -- internal method tests
# =============================================================================


class TestEmitCrossStorageCopy:
    """Direct tests of _emit_cross_storage_copy."""

    def test_cpu_to_gpu_direct_no_subset(self):
        """Direct call with no-subset memlet emits .set() with [:] dst."""
        sdfg = _make_cutile_python_sdfg("test_direct_cpu_gpu_no_sub")
        sdfg.add_array("src", [10], dace.float64,
                       storage=dtypes.StorageType.CPU_Heap)
        sdfg.add_array("dst", [10], dace.float64,
                       storage=dtypes.StorageType.GPU_Global, transient=True)
        state = sdfg.add_state("s")
        src_node = state.add_read("src")
        dst_node = state.add_write("dst")
        # Memlet auto-infers src_subset from array shape.
        memlet = Memlet(data="src")
        state.add_edge(src_node, None, dst_node, None, memlet)

        codegen = _make_cutile_codegen(sdfg)
        cs_stream = PythonCodeIOStream()
        codegen._emit_cross_storage_copy(
            sdfg, sdfg, 0, src_node, dst_node, memlet,
            cpu_to_gpu=True, callsite_stream=cs_stream)
        code = cs_stream.getvalue()
        assert ".set(" in code
        assert "src" in code
        assert "dst[" in code

    def test_gpu_to_cpu_direct_no_subset(self):
        """Direct call with no-subset memlet emits .get(out=...) with [:] dst."""
        sdfg = _make_cutile_python_sdfg("test_direct_gpu_cpu_no_sub")
        sdfg.add_array("src", [10], dace.float64,
                       storage=dtypes.StorageType.GPU_Global)
        sdfg.add_array("dst", [10], dace.float64,
                       storage=dtypes.StorageType.CPU_Heap, transient=True)
        state = sdfg.add_state("s")
        src_node = state.add_read("src")
        dst_node = state.add_write("dst")
        memlet = Memlet(data="src")
        state.add_edge(src_node, None, dst_node, None, memlet)

        codegen = _make_cutile_codegen(sdfg)
        cs_stream = PythonCodeIOStream()
        codegen._emit_cross_storage_copy(
            sdfg, sdfg, 0, src_node, dst_node, memlet,
            cpu_to_gpu=False, callsite_stream=cs_stream)
        code = cs_stream.getvalue()
        assert "src" in code
        assert ".get(out=" in code
        assert "dst[" in code

    def test_cpu_to_gpu_direct_truly_no_subset(self):
        """Direct call with a memlet that has no subsets emits bare src name."""
        sdfg = _make_cutile_python_sdfg("test_direct_cpu_gpu")
        sdfg.add_array("src", [10], dace.float64,
                       storage=dtypes.StorageType.CPU_Heap)
        sdfg.add_array("dst", [10], dace.float64,
                       storage=dtypes.StorageType.GPU_Global, transient=True)
        state = sdfg.add_state("s")
        src_node = state.add_read("src")
        dst_node = state.add_write("dst")
        # Create memlet without adding edge (add_edge auto-infers subsets).
        memlet = Memlet(data="src")
        assert memlet.src_subset is None  # confirm no auto-inference

        codegen = _make_cutile_codegen(sdfg)
        cs_stream = PythonCodeIOStream()
        codegen._emit_cross_storage_copy(
            sdfg, sdfg, 0, src_node, dst_node, memlet,
            cpu_to_gpu=True, callsite_stream=cs_stream)
        code = cs_stream.getvalue()
        assert "dst[:].set(src)" in code

    def test_gpu_to_cpu_direct_truly_no_subset(self):
        """Direct call with a memlet that has no subsets emits bare src name."""
        sdfg = _make_cutile_python_sdfg("test_direct_gpu_cpu")
        sdfg.add_array("src", [10], dace.float64,
                       storage=dtypes.StorageType.GPU_Global)
        sdfg.add_array("dst", [10], dace.float64,
                       storage=dtypes.StorageType.CPU_Heap, transient=True)
        state = sdfg.add_state("s")
        src_node = state.add_read("src")
        dst_node = state.add_write("dst")
        memlet = Memlet(data="src")
        assert memlet.src_subset is None

        codegen = _make_cutile_codegen(sdfg)
        cs_stream = PythonCodeIOStream()
        codegen._emit_cross_storage_copy(
            sdfg, sdfg, 0, src_node, dst_node, memlet,
            cpu_to_gpu=False, callsite_stream=cs_stream)
        code = cs_stream.getvalue()
        assert "src.get(out=dst[:])" in code

    def test_subset_applied_to_source(self):
        """When memlet has src_subset, it is applied to the source expression."""
        sdfg = _make_cutile_python_sdfg("test_direct_src_sub")
        sdfg.add_array("src", [32], dace.float64,
                       storage=dtypes.StorageType.CPU_Heap)
        sdfg.add_array("dst", [32], dace.float64,
                       storage=dtypes.StorageType.GPU_Global, transient=True)
        state = sdfg.add_state("s")
        src_node = state.add_read("src")
        dst_node = state.add_write("dst")
        memlet = Memlet(data="src", subset="0:16")
        state.add_edge(src_node, None, dst_node, None, memlet)

        codegen = _make_cutile_codegen(sdfg)
        cs_stream = PythonCodeIOStream()
        codegen._emit_cross_storage_copy(
            sdfg, sdfg, 0, src_node, dst_node, memlet,
            cpu_to_gpu=True, callsite_stream=cs_stream)
        code = cs_stream.getvalue()
        assert ".set(" in code
        assert "src[" in code
        assert "dst[:]" in code

    def test_subset_applied_to_destination(self):
        """When memlet has dst_subset, it appears on the LHS."""
        sdfg = _make_cutile_python_sdfg("test_direct_dst_sub")
        sdfg.add_array("src", [32], dace.float64,
                       storage=dtypes.StorageType.CPU_Heap)
        sdfg.add_array("dst", [32], dace.float64,
                       storage=dtypes.StorageType.GPU_Global, transient=True)
        state = sdfg.add_state("s")
        src_node = state.add_read("src")
        dst_node = state.add_write("dst")
        memlet = Memlet(data="src", subset="0:16", other_subset="0:16")
        state.add_edge(src_node, None, dst_node, None, memlet)

        codegen = _make_cutile_codegen(sdfg)
        cs_stream = PythonCodeIOStream()
        codegen._emit_cross_storage_copy(
            sdfg, sdfg, 0, src_node, dst_node, memlet,
            cpu_to_gpu=True, callsite_stream=cs_stream)
        code = cs_stream.getvalue()
        assert "dst[" in code
        assert ".set(" in code
        assert "src[" in code


# =============================================================================
# copy_memory -- Default storage treated as host-side
# =============================================================================


class TestCopyMemoryDefaultStorage:
    """Verify that StorageType.Default is treated as host-side (CPU_Heap).

    Default storage resolves to CPU_Heap in practice.  Cross-storage
    copies between Default and GPU_Global should still emit
    ``.set()`` / ``.get(out=...)``.
    """

    def test_default_to_gpu_global_emits_set(self):
        """Default -> GPU_Global emits .set() (treated as host)."""
        (sdfg, codegen, state, src, dst, edge,
         fn_stream, cs_stream) = _build_copy_sdfg_and_edge(
            "test_default_to_gpu",
            "A_default", "A_gpu",
            dtypes.StorageType.Default,
            dtypes.StorageType.GPU_Global,
            [16],
        )
        codegen.copy_memory(sdfg, sdfg, state, 0, src, dst, edge,
                            fn_stream, cs_stream)
        code = cs_stream.getvalue()
        assert ".set(" in code
        assert "A_default" in code
        assert "A_gpu[" in code

    def test_gpu_global_to_default_emits_get_out(self):
        """GPU_Global -> Default emits .get(out=...) (treated as host)."""
        (sdfg, codegen, state, src, dst, edge,
         fn_stream, cs_stream) = _build_copy_sdfg_and_edge(
            "test_gpu_to_default",
            "B_gpu", "B_default",
            dtypes.StorageType.GPU_Global,
            dtypes.StorageType.Default,
            [16],
        )
        codegen.copy_memory(sdfg, sdfg, state, 0, src, dst, edge,
                            fn_stream, cs_stream)
        code = cs_stream.getvalue()
        assert "B_gpu" in code
        assert ".get(out=" in code
        assert "B_default[" in code

    def test_default_to_default_is_same_storage(self):
        """Default -> Default is same-storage (both host), plain assignment."""
        (sdfg, codegen, state, src, dst, edge,
         fn_stream, cs_stream) = _build_copy_sdfg_and_edge(
            "test_default_to_default",
            "C_default", "D_default",
            dtypes.StorageType.Default,
            dtypes.StorageType.Default,
            [16],
        )
        codegen.copy_memory(sdfg, sdfg, state, 0, src, dst, edge,
                            fn_stream, cs_stream)
        code = cs_stream.getvalue()
        assert "cupy.asarray" not in code
        assert "cupy.asnumpy" not in code
        assert ".set(" not in code
        assert ".get(out=" not in code
        assert "C_default" in code
        assert "D_default" in code

    def test_default_to_gpu_with_subset(self):
        """Default -> GPU_Global with subsets applies indexing correctly."""
        (sdfg, codegen, state, src, dst, edge,
         fn_stream, cs_stream) = _build_copy_sdfg_and_edge(
            "test_default_to_gpu_sub",
            "E_def", "E_gpu",
            dtypes.StorageType.Default,
            dtypes.StorageType.GPU_Global,
            [64],
            src_subset_str="0:32",
            dst_subset_str="0:32",
        )
        codegen.copy_memory(sdfg, sdfg, state, 0, src, dst, edge,
                            fn_stream, cs_stream)
        code = cs_stream.getvalue()
        assert ".set(" in code
        assert "E_def[" in code
        assert "E_gpu[" in code

    def test_cpu_heap_to_default_is_same_storage(self):
        """CPU_Heap -> Default is same-storage (both host), plain assignment."""
        (sdfg, codegen, state, src, dst, edge,
         fn_stream, cs_stream) = _build_copy_sdfg_and_edge(
            "test_cpu_to_default",
            "F_cpu", "F_default",
            dtypes.StorageType.CPU_Heap,
            dtypes.StorageType.Default,
            [16],
        )
        codegen.copy_memory(sdfg, sdfg, state, 0, src, dst, edge,
                            fn_stream, cs_stream)
        code = cs_stream.getvalue()
        assert "cupy.asarray" not in code
        assert "cupy.asnumpy" not in code
        assert ".set(" not in code
        assert ".get(out=" not in code
