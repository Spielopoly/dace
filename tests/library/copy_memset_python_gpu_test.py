# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""GPU and integration tests for Python-backend CopyLibraryNode / MemsetLibraryNode expansions.

Tests in this file require a GPU and CuPy, and verify:
- Cross-storage copy dispatch (CPU_Heap <-> GPU_Global) via the Python backend,
  which emits ``.set()`` (host->device) / ``.get(out=...)`` (device->host)
- Same-storage GPU_Global -> GPU_Global copies
- MemsetLibraryNode on GPU_Global arrays
- InsertExplicitCopies pass + Python backend end-to-end

All GPU runtime tests are marked ``@pytest.mark.gpu``.  The Python backend's
cuTile codegen target registers cross-storage copy dispatchers unconditionally,
so any Python-backend SDFG with CPU_Heap <-> GPU_Global copies emits CuPy
``.set()`` / ``.get()`` calls.  This means GPU_Global arrays must be CuPy
ndarrays at runtime.
"""
from typing import Optional, Sequence, Tuple

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.libraries.standard.nodes.copy_node import CopyLibraryNode
from dace.libraries.standard.nodes.memset_node import MemsetLibraryNode
from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies

# CuPy is required for GPU runtime tests.  Import with a fallback so
# structural (non-GPU) tests in the same file still work.
try:
    import cupy as cp
except ImportError:
    cp = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_copy_sdfg(
    name: str,
    shape: Sequence[int],
    dtype: dace.typeclass,
    src_storage: dtypes.StorageType,
    dst_storage: dtypes.StorageType,
    src_subset: Optional[str] = None,
    dst_subset: Optional[str] = None,
) -> Tuple[dace.SDFG, CopyLibraryNode]:
    """Build an SDFG with a CopyLibraryNode copying between two arrays.

    :param name: SDFG name (should be unique across tests).
    :param shape: array shape for both src and dst.
    :param dtype: element type.
    :param src_storage: source storage type.
    :param dst_storage: destination storage type.
    :param src_subset: source memlet subset string (default: full range).
    :param dst_subset: destination memlet subset string (default: full range).
    :returns: ``(sdfg, libnode)``.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('src', list(shape), dtype, storage=src_storage)
    sdfg.add_array('dst', list(shape), dtype, storage=dst_storage)

    state = sdfg.add_state('copy_state')
    src_node = state.add_access('src')
    dst_node = state.add_access('dst')

    src_sub = src_subset or ', '.join(f'0:{s}' for s in shape)
    dst_sub = dst_subset or ', '.join(f'0:{s}' for s in shape)

    cpy = CopyLibraryNode(name='_copy_')
    state.add_node(cpy)
    state.add_edge(src_node, None, cpy, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.Memlet(f'src[{src_sub}]'))
    state.add_edge(cpy, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, dst_node, None, dace.Memlet(f'dst[{dst_sub}]'))

    sdfg.backend = dtypes.BackendLanguage.Python
    return sdfg, cpy


def _make_memset_sdfg(
    name: str,
    shape: Sequence[int],
    dtype: dace.typeclass,
    storage: dtypes.StorageType,
    subset: Optional[str] = None,
) -> Tuple[dace.SDFG, MemsetLibraryNode]:
    """Build an SDFG with a MemsetLibraryNode zeroing an array.

    :param name: SDFG name (should be unique across tests).
    :param shape: array shape.
    :param dtype: element type.
    :param storage: array storage type.
    :param subset: memlet subset string (default: full range).
    :returns: ``(sdfg, libnode)``.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('arr', list(shape), dtype, storage=storage)

    sub = subset or ', '.join(f'0:{s}' for s in shape)

    state = sdfg.add_state('memset_state')
    arr_node = state.add_access('arr')
    mset = MemsetLibraryNode(name='_memset_')
    state.add_node(mset)
    state.add_edge(mset, MemsetLibraryNode.OUTPUT_CONNECTOR_NAME, arr_node, None, dace.Memlet(f'arr[{sub}]'))

    sdfg.backend = dtypes.BackendLanguage.Python
    return sdfg, mset


def _make_an_copy_sdfg(
    name: str,
    shape: Sequence[int],
    dtype: dace.typeclass,
    src_storage: dtypes.StorageType,
    dst_storage: dtypes.StorageType,
) -> dace.SDFG:
    """Build an SDFG with a direct AccessNode->AccessNode copy edge (no CopyLibraryNode).

    Used to test ``InsertExplicitCopies`` which converts these to CopyLibraryNodes.

    :param name: SDFG name (should be unique across tests).
    :param shape: array shape for both src and dst.
    :param dtype: element type.
    :param src_storage: source storage type.
    :param dst_storage: destination storage type.
    :returns: the SDFG (no backend set -- caller must set it).
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('src', list(shape), dtype, storage=src_storage)
    sdfg.add_array('dst', list(shape), dtype, storage=dst_storage)

    state = sdfg.add_state('copy_state')
    src_node = state.add_access('src')
    dst_node = state.add_access('dst')

    sub = ', '.join(f'0:{s}' for s in shape)
    state.add_edge(src_node, None, dst_node, None, dace.Memlet(data='src', subset=sub, other_subset=sub))
    return sdfg


def _require_cupy():
    """Skip the calling test if CuPy is not available."""
    if cp is None:
        pytest.skip('CuPy is required for GPU runtime tests')


# ---------------------------------------------------------------------------
# GPU Cross-Storage Copy Tests (CopyLibraryNode)
# ---------------------------------------------------------------------------


class TestCopyPythonGPUCrossStorage:
    """Prove that the Python expansion dispatches cross-storage copies via
    ``.set()`` (host->device, ``ExpandPythonH2D``) and ``.get(out=...)``
    (device->host, ``ExpandPythonD2H``)."""

    @pytest.mark.gpu
    def test_host_to_device_1d(self):
        """CPU_Heap -> GPU_Global 1D copy."""
        _require_cupy()
        sdfg, libnode = _make_copy_sdfg('gpu_h2d_1d', (64, ), dace.float64, dtypes.StorageType.CPU_Heap,
                                        dtypes.StorageType.GPU_Global)
        sdfg.validate()
        sdfg.expand_library_nodes()
        assert libnode.implementation == 'PythonH2D'
        sdfg.validate()
        csdfg = sdfg.compile()

        src = np.arange(64, dtype=np.float64)
        dst = cp.zeros(64, dtype=np.float64)
        csdfg(src=src, dst=dst)
        np.testing.assert_array_equal(cp.asnumpy(dst), src)

    @pytest.mark.gpu
    def test_device_to_host_1d(self):
        """GPU_Global -> CPU_Heap 1D copy."""
        _require_cupy()
        sdfg, libnode = _make_copy_sdfg('gpu_d2h_1d', (64, ), dace.float64, dtypes.StorageType.GPU_Global,
                                        dtypes.StorageType.CPU_Heap)
        sdfg.validate()
        sdfg.expand_library_nodes()
        assert libnode.implementation == 'PythonD2H'
        sdfg.validate()
        csdfg = sdfg.compile()

        src = cp.asarray(np.arange(64, dtype=np.float64))
        dst = np.zeros(64, dtype=np.float64)
        csdfg(src=src, dst=dst)
        np.testing.assert_array_equal(dst, np.arange(64, dtype=np.float64))

    @pytest.mark.gpu
    def test_host_to_device_2d(self):
        """CPU_Heap -> GPU_Global 2D copy."""
        _require_cupy()
        sdfg, libnode = _make_copy_sdfg('gpu_h2d_2d', (8, 16), dace.float64, dtypes.StorageType.CPU_Heap,
                                        dtypes.StorageType.GPU_Global)
        sdfg.validate()
        sdfg.expand_library_nodes()
        assert libnode.implementation == 'PythonH2D'
        sdfg.validate()
        csdfg = sdfg.compile()

        src = np.arange(128, dtype=np.float64).reshape(8, 16)
        dst = cp.zeros((8, 16), dtype=np.float64)
        csdfg(src=src, dst=dst)
        np.testing.assert_array_equal(cp.asnumpy(dst), src)

    @pytest.mark.gpu
    def test_device_to_host_2d(self):
        """GPU_Global -> CPU_Heap 2D copy."""
        _require_cupy()
        sdfg, libnode = _make_copy_sdfg('gpu_d2h_2d', (8, 16), dace.float64, dtypes.StorageType.GPU_Global,
                                        dtypes.StorageType.CPU_Heap)
        sdfg.validate()
        sdfg.expand_library_nodes()
        assert libnode.implementation == 'PythonD2H'
        sdfg.validate()
        csdfg = sdfg.compile()

        ref = np.arange(128, dtype=np.float64).reshape(8, 16)
        src = cp.asarray(ref)
        dst = np.zeros((8, 16), dtype=np.float64)
        csdfg(src=src, dst=dst)
        np.testing.assert_array_equal(dst, ref)

    @pytest.mark.gpu
    def test_host_to_device_float32(self):
        """CPU_Heap -> GPU_Global with float32 dtype."""
        _require_cupy()
        sdfg, libnode = _make_copy_sdfg('gpu_h2d_f32', (32, ), dace.float32, dtypes.StorageType.CPU_Heap,
                                        dtypes.StorageType.GPU_Global)
        sdfg.validate()
        sdfg.expand_library_nodes()
        assert libnode.implementation == 'PythonH2D'
        sdfg.validate()
        csdfg = sdfg.compile()

        src = np.arange(32, dtype=np.float32)
        dst = cp.zeros(32, dtype=np.float32)
        csdfg(src=src, dst=dst)
        np.testing.assert_array_equal(cp.asnumpy(dst), src)

    @pytest.mark.gpu
    def test_host_to_device_int32(self):
        """CPU_Heap -> GPU_Global with int32 dtype."""
        _require_cupy()
        sdfg, libnode = _make_copy_sdfg('gpu_h2d_i32', (32, ), dace.int32, dtypes.StorageType.CPU_Heap,
                                        dtypes.StorageType.GPU_Global)
        sdfg.validate()
        sdfg.expand_library_nodes()
        assert libnode.implementation == 'PythonH2D'
        sdfg.validate()
        csdfg = sdfg.compile()

        src = np.arange(32, dtype=np.int32)
        dst = cp.zeros(32, dtype=np.int32)
        csdfg(src=src, dst=dst)
        np.testing.assert_array_equal(cp.asnumpy(dst), src)

    @pytest.mark.gpu
    def test_roundtrip_h2d_then_d2h(self):
        """Full roundtrip: CPU -> GPU -> CPU through two states.

        The intermediate GPU buffer is a non-transient argument so the
        test can supply a CuPy array (the Python backend's transient
        allocator does not use CuPy unless the SDFG uses CuTile).
        """
        _require_cupy()
        sdfg = dace.SDFG('gpu_roundtrip')
        sdfg.add_array('host_in', (64, ), dace.float64, storage=dtypes.StorageType.CPU_Heap)
        sdfg.add_array('gpu_buf', (64, ), dace.float64, storage=dtypes.StorageType.GPU_Global)
        sdfg.add_array('host_out', (64, ), dace.float64, storage=dtypes.StorageType.CPU_Heap)

        # State 1: host_in -> gpu_buf
        s1 = sdfg.add_state('h2d')
        src1 = s1.add_access('host_in')
        dst1 = s1.add_access('gpu_buf')
        cpy1 = CopyLibraryNode(name='_h2d_')
        s1.add_node(cpy1)
        s1.add_edge(src1, None, cpy1, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.Memlet('host_in[0:64]'))
        s1.add_edge(cpy1, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, dst1, None, dace.Memlet('gpu_buf[0:64]'))

        # State 2: gpu_buf -> host_out
        s2 = sdfg.add_state('d2h')
        src2 = s2.add_access('gpu_buf')
        dst2 = s2.add_access('host_out')
        cpy2 = CopyLibraryNode(name='_d2h_')
        s2.add_node(cpy2)
        s2.add_edge(src2, None, cpy2, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.Memlet('gpu_buf[0:64]'))
        s2.add_edge(cpy2, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, dst2, None, dace.Memlet('host_out[0:64]'))

        sdfg.add_edge(s1, s2, dace.InterstateEdge())
        sdfg.backend = dtypes.BackendLanguage.Python

        sdfg.validate()
        sdfg.expand_library_nodes()
        sdfg.validate()
        csdfg = sdfg.compile()

        host_in = np.arange(64, dtype=np.float64)
        gpu_buf = cp.zeros(64, dtype=np.float64)
        host_out = np.zeros(64, dtype=np.float64)
        csdfg(host_in=host_in, gpu_buf=gpu_buf, host_out=host_out)
        np.testing.assert_array_equal(host_out, host_in)


# ---------------------------------------------------------------------------
# GPU Same-Storage Copy Tests
# ---------------------------------------------------------------------------


class TestCopyPythonGPUSameStorage:
    """GPU_Global -> GPU_Global copies under the Python backend."""

    @pytest.mark.gpu
    def test_device_to_device_1d(self):
        """GPU_Global -> GPU_Global 1D copy."""
        _require_cupy()
        sdfg, libnode = _make_copy_sdfg('gpu_d2d_1d', (64, ), dace.float64, dtypes.StorageType.GPU_Global,
                                        dtypes.StorageType.GPU_Global)
        sdfg.validate()
        sdfg.expand_library_nodes()
        assert libnode.implementation == 'Python'
        sdfg.validate()
        csdfg = sdfg.compile()

        src = cp.asarray(np.arange(64, dtype=np.float64))
        dst = cp.zeros(64, dtype=np.float64)
        csdfg(src=src, dst=dst)
        np.testing.assert_array_equal(cp.asnumpy(dst), np.arange(64, dtype=np.float64))

    @pytest.mark.gpu
    def test_device_to_device_2d(self):
        """GPU_Global -> GPU_Global 2D copy."""
        _require_cupy()
        sdfg, libnode = _make_copy_sdfg('gpu_d2d_2d', (8, 16), dace.float64, dtypes.StorageType.GPU_Global,
                                        dtypes.StorageType.GPU_Global)
        sdfg.validate()
        sdfg.expand_library_nodes()
        assert libnode.implementation == 'Python'
        sdfg.validate()
        csdfg = sdfg.compile()

        ref = np.arange(128, dtype=np.float64).reshape(8, 16)
        src = cp.asarray(ref)
        dst = cp.zeros((8, 16), dtype=np.float64)
        csdfg(src=src, dst=dst)
        np.testing.assert_array_equal(cp.asnumpy(dst), ref)


# ---------------------------------------------------------------------------
# GPU Memset Tests
# ---------------------------------------------------------------------------


class TestMemsetPythonGPU:
    """MemsetLibraryNode on GPU_Global arrays under the Python backend.

    The ``ExpandPython`` expansion returns a bare Tasklet for GPU-resident
    arrays, avoiding the host-side Sequential map that would fail validation.
    The Python backend emits ``arr[...] = 0`` via slice assignment, which
    CuPy handles through broadcasting.
    """

    @pytest.mark.gpu
    def test_gpu_memset_1d(self):
        """GPU_Global 1D memset."""
        _require_cupy()
        sdfg, libnode = _make_memset_sdfg('gpu_mset_1d', (64, ), dace.float64, dtypes.StorageType.GPU_Global)
        sdfg.validate()
        sdfg.expand_library_nodes()
        assert libnode.implementation == 'Python'
        sdfg.validate()
        csdfg = sdfg.compile()

        arr = cp.ones(64, dtype=np.float64)
        csdfg(arr=arr)
        np.testing.assert_array_equal(cp.asnumpy(arr), np.zeros(64, dtype=np.float64))

    @pytest.mark.gpu
    def test_gpu_memset_2d(self):
        """GPU_Global 2D memset."""
        _require_cupy()
        sdfg, libnode = _make_memset_sdfg('gpu_mset_2d', (8, 16), dace.float64, dtypes.StorageType.GPU_Global)
        sdfg.validate()
        sdfg.expand_library_nodes()
        assert libnode.implementation == 'Python'
        sdfg.validate()
        csdfg = sdfg.compile()

        arr = cp.ones((8, 16), dtype=np.float64)
        csdfg(arr=arr)
        np.testing.assert_array_equal(cp.asnumpy(arr), np.zeros((8, 16), dtype=np.float64))

    @pytest.mark.gpu
    def test_gpu_memset_float32(self):
        """GPU_Global memset with float32 dtype."""
        _require_cupy()
        sdfg, libnode = _make_memset_sdfg('gpu_mset_f32', (32, ), dace.float32, dtypes.StorageType.GPU_Global)
        sdfg.validate()
        sdfg.expand_library_nodes()
        assert libnode.implementation == 'Python'
        sdfg.validate()
        csdfg = sdfg.compile()

        arr = cp.ones(32, dtype=np.float32)
        csdfg(arr=arr)
        np.testing.assert_array_equal(cp.asnumpy(arr), np.zeros(32, dtype=np.float32))

    @pytest.mark.gpu
    def test_gpu_memset_slice(self):
        """GPU_Global partial memset: zeros arr[10:50], rest stays."""
        _require_cupy()
        sdfg, libnode = _make_memset_sdfg('gpu_mset_slice', (64, ),
                                          dace.float64,
                                          dtypes.StorageType.GPU_Global,
                                          subset='10:50')
        sdfg.validate()
        sdfg.expand_library_nodes()
        assert libnode.implementation == 'Python'
        sdfg.validate()
        csdfg = sdfg.compile()

        arr = cp.ones(64, dtype=np.float64)
        csdfg(arr=arr)
        result = cp.asnumpy(arr)
        assert np.all(result[:10] == 1)
        assert np.all(result[50:] == 1)
        assert np.all(result[10:50] == 0)


# ---------------------------------------------------------------------------
# InsertExplicitCopies + Python Backend Integration Tests
# ---------------------------------------------------------------------------


class TestInsertExplicitCopiesPythonBackend:
    """Verify the core use case: InsertExplicitCopies on a Python-backend SDFG."""

    def test_cpu_same_storage(self):
        """CPU_Heap -> CPU_Heap: InsertExplicitCopies + Python backend."""
        sdfg = _make_an_copy_sdfg('iec_cpu', (32, ), dace.float64, dtypes.StorageType.CPU_Heap,
                                  dtypes.StorageType.CPU_Heap)
        sdfg.backend = dtypes.BackendLanguage.Python
        result = InsertExplicitCopies().apply_pass(sdfg, {})
        assert result is not None, "Pass should have inserted a copy node"
        # Verify CopyLibraryNode was inserted
        state = sdfg.states()[0]
        copy_nodes = [n for n in state.nodes() if isinstance(n, CopyLibraryNode)]
        assert len(copy_nodes) == 1
        # Compile and run
        sdfg.validate()
        sdfg.expand_library_nodes()
        sdfg.validate()
        csdfg = sdfg.compile()

        src = np.arange(32, dtype=np.float64)
        dst = np.zeros(32, dtype=np.float64)
        csdfg(src=src, dst=dst)
        np.testing.assert_array_equal(dst, src)

    @pytest.mark.gpu
    def test_host_to_device(self):
        """CPU_Heap -> GPU_Global: InsertExplicitCopies + Python backend."""
        _require_cupy()
        sdfg = _make_an_copy_sdfg('iec_h2d', (64, ), dace.float64, dtypes.StorageType.CPU_Heap,
                                  dtypes.StorageType.GPU_Global)
        sdfg.backend = dtypes.BackendLanguage.Python
        result = InsertExplicitCopies().apply_pass(sdfg, {})
        assert result is not None
        state = sdfg.states()[0]
        copy_nodes = [n for n in state.nodes() if isinstance(n, CopyLibraryNode)]
        assert len(copy_nodes) == 1
        sdfg.validate()
        sdfg.expand_library_nodes()
        sdfg.validate()
        csdfg = sdfg.compile()

        src = np.arange(64, dtype=np.float64)
        dst = cp.zeros(64, dtype=np.float64)
        csdfg(src=src, dst=dst)
        np.testing.assert_array_equal(cp.asnumpy(dst), src)

    @pytest.mark.gpu
    def test_device_to_host(self):
        """GPU_Global -> CPU_Heap: InsertExplicitCopies + Python backend."""
        _require_cupy()
        sdfg = _make_an_copy_sdfg('iec_d2h', (64, ), dace.float64, dtypes.StorageType.GPU_Global,
                                  dtypes.StorageType.CPU_Heap)
        sdfg.backend = dtypes.BackendLanguage.Python
        result = InsertExplicitCopies().apply_pass(sdfg, {})
        assert result is not None
        state = sdfg.states()[0]
        copy_nodes = [n for n in state.nodes() if isinstance(n, CopyLibraryNode)]
        assert len(copy_nodes) == 1
        sdfg.validate()
        sdfg.expand_library_nodes()
        sdfg.validate()
        csdfg = sdfg.compile()

        ref = np.arange(64, dtype=np.float64)
        src = cp.asarray(ref)
        dst = np.zeros(64, dtype=np.float64)
        csdfg(src=src, dst=dst)
        np.testing.assert_array_equal(dst, ref)

    @pytest.mark.gpu
    def test_device_to_device(self):
        """GPU_Global -> GPU_Global: InsertExplicitCopies + Python backend."""
        _require_cupy()
        sdfg = _make_an_copy_sdfg('iec_d2d', (64, ), dace.float64, dtypes.StorageType.GPU_Global,
                                  dtypes.StorageType.GPU_Global)
        sdfg.backend = dtypes.BackendLanguage.Python
        result = InsertExplicitCopies().apply_pass(sdfg, {})
        assert result is not None
        state = sdfg.states()[0]
        copy_nodes = [n for n in state.nodes() if isinstance(n, CopyLibraryNode)]
        assert len(copy_nodes) == 1
        sdfg.validate()
        sdfg.expand_library_nodes()
        sdfg.validate()
        csdfg = sdfg.compile()

        ref = np.arange(64, dtype=np.float64)
        src = cp.asarray(ref)
        dst = cp.zeros(64, dtype=np.float64)
        csdfg(src=src, dst=dst)
        np.testing.assert_array_equal(cp.asnumpy(dst), ref)

    def test_2d_cpu(self):
        """2D CPU_Heap -> CPU_Heap: InsertExplicitCopies + Python backend."""
        sdfg = _make_an_copy_sdfg('iec_2d_cpu', (8, 16), dace.float64, dtypes.StorageType.CPU_Heap,
                                  dtypes.StorageType.CPU_Heap)
        sdfg.backend = dtypes.BackendLanguage.Python
        InsertExplicitCopies().apply_pass(sdfg, {})
        sdfg.validate()
        sdfg.expand_library_nodes()
        sdfg.validate()
        csdfg = sdfg.compile()

        src = np.arange(128, dtype=np.float64).reshape(8, 16)
        dst = np.zeros((8, 16), dtype=np.float64)
        csdfg(src=src, dst=dst)
        np.testing.assert_array_equal(dst, src)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Additional edge cases: odd shapes, non-power-of-2, 3D."""

    @pytest.mark.gpu
    def test_odd_shape_h2d(self):
        """Non-power-of-2 shape: CPU -> GPU."""
        _require_cupy()
        sdfg, libnode = _make_copy_sdfg('gpu_odd_h2d', (37, ), dace.float64, dtypes.StorageType.CPU_Heap,
                                        dtypes.StorageType.GPU_Global)
        sdfg.validate()
        sdfg.expand_library_nodes()
        assert libnode.implementation == 'PythonH2D'
        sdfg.validate()
        csdfg = sdfg.compile()

        src = np.arange(37, dtype=np.float64)
        dst = cp.zeros(37, dtype=np.float64)
        csdfg(src=src, dst=dst)
        np.testing.assert_array_equal(cp.asnumpy(dst), src)

    @pytest.mark.gpu
    def test_3d_h2d(self):
        """3D array: CPU -> GPU."""
        _require_cupy()
        sdfg, libnode = _make_copy_sdfg('gpu_3d_h2d', (4, 8, 16), dace.float64, dtypes.StorageType.CPU_Heap,
                                        dtypes.StorageType.GPU_Global)
        sdfg.validate()
        sdfg.expand_library_nodes()
        assert libnode.implementation == 'PythonH2D'
        sdfg.validate()
        csdfg = sdfg.compile()

        src = np.arange(512, dtype=np.float64).reshape(4, 8, 16)
        dst = cp.zeros((4, 8, 16), dtype=np.float64)
        csdfg(src=src, dst=dst)
        np.testing.assert_array_equal(cp.asnumpy(dst), src)

    @pytest.mark.gpu
    def test_single_element_h2d(self):
        """Single-element cross-storage copy."""
        _require_cupy()
        sdfg, libnode = _make_copy_sdfg('gpu_single_h2d', (1, ), dace.float64, dtypes.StorageType.CPU_Heap,
                                        dtypes.StorageType.GPU_Global)
        sdfg.validate()
        sdfg.expand_library_nodes()
        assert libnode.implementation == 'PythonH2D'
        sdfg.validate()
        csdfg = sdfg.compile()

        src = np.array([42.0], dtype=np.float64)
        dst = cp.zeros(1, dtype=np.float64)
        csdfg(src=src, dst=dst)
        np.testing.assert_array_equal(cp.asnumpy(dst), src)

    @pytest.mark.gpu
    def test_int64_d2h(self):
        """int64 device to host copy."""
        _require_cupy()
        sdfg, libnode = _make_copy_sdfg('gpu_i64_d2h', (48, ), dace.int64, dtypes.StorageType.GPU_Global,
                                        dtypes.StorageType.CPU_Heap)
        sdfg.validate()
        sdfg.expand_library_nodes()
        assert libnode.implementation == 'PythonD2H'
        sdfg.validate()
        csdfg = sdfg.compile()

        ref = np.arange(48, dtype=np.int64)
        src = cp.asarray(ref)
        dst = np.zeros(48, dtype=np.int64)
        csdfg(src=src, dst=dst)
        np.testing.assert_array_equal(dst, ref)

    @pytest.mark.gpu
    def test_gpu_memset_3d(self):
        """3D GPU_Global memset."""
        _require_cupy()
        sdfg, libnode = _make_memset_sdfg('gpu_mset_3d', (4, 5, 6), dace.float64, dtypes.StorageType.GPU_Global)
        sdfg.validate()
        sdfg.expand_library_nodes()
        assert libnode.implementation == 'Python'
        sdfg.validate()
        csdfg = sdfg.compile()

        arr = cp.ones((4, 5, 6), dtype=np.float64)
        csdfg(arr=arr)
        assert np.all(cp.asnumpy(arr) == 0)

    @pytest.mark.gpu
    def test_iec_2d_h2d(self):
        """2D CPU_Heap -> GPU_Global: InsertExplicitCopies + Python backend."""
        _require_cupy()
        sdfg = _make_an_copy_sdfg('iec_2d_h2d', (8, 16), dace.float64, dtypes.StorageType.CPU_Heap,
                                  dtypes.StorageType.GPU_Global)
        sdfg.backend = dtypes.BackendLanguage.Python
        result = InsertExplicitCopies().apply_pass(sdfg, {})
        assert result is not None
        sdfg.validate()
        sdfg.expand_library_nodes()
        sdfg.validate()
        csdfg = sdfg.compile()

        src = np.arange(128, dtype=np.float64).reshape(8, 16)
        dst = cp.zeros((8, 16), dtype=np.float64)
        csdfg(src=src, dst=dst)
        np.testing.assert_array_equal(cp.asnumpy(dst), src)


if __name__ == '__main__':
    pytest.main([__file__])
