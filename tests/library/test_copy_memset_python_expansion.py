# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for Python-backend-friendly expansions of CopyLibraryNode and MemsetLibraryNode.

Covers:
- ``ExpandPython`` on ``CopyLibraryNode``: structure, auto-selection, compile+run.
- ``ExpandPython`` on ``MemsetLibraryNode``: structure, auto-selection, compile+run.
- ``is_python_backend`` helper.
"""
from typing import Optional, Sequence, Tuple

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.libraries.standard.nodes.copy_node import (
    CopyLibraryNode,
    select_copy_implementation,
)
from dace.libraries.standard.nodes.memset_node import (
    MemsetLibraryNode,
    select_memset_implementation,
)
from dace.libraries.standard.helper import is_python_backend
from dace.sdfg import nodes

# =============================================================================
# Helpers
# =============================================================================


def _make_copy_sdfg(
    src_shape: Sequence[int],
    dst_shape: Sequence[int],
    src_storage: dtypes.StorageType = dtypes.StorageType.CPU_Heap,
    dst_storage: dtypes.StorageType = dtypes.StorageType.CPU_Heap,
    src_subset: Optional[str] = None,
    dst_subset: Optional[str] = None,
    implementation: Optional[str] = None,
    name: str = "copy_sdfg",
    dtype: dace.dtypes.typeclass = dace.float64,
    python_backend: bool = False,
) -> Tuple[dace.SDFG, CopyLibraryNode]:
    """Build a one-state SDFG that copies ``src`` -> ``dst`` via CopyLibraryNode.

    :param src_shape: source array shape.
    :param dst_shape: destination array shape.
    :param src_storage: source storage type.
    :param dst_storage: destination storage type.
    :param src_subset: source memlet subset string (default: full range).
    :param dst_subset: destination memlet subset string (default: full range).
    :param implementation: pinned implementation name (None keeps 'Auto').
    :param name: SDFG name.
    :param dtype: element type.
    :param python_backend: if True, set backend to Python.
    :returns: ``(sdfg, libnode)``.
    """
    sdfg = dace.SDFG(name)
    if python_backend:
        sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array("src", list(src_shape), dtype, storage=src_storage)
    sdfg.add_array("dst", list(dst_shape), dtype, storage=dst_storage)

    state = sdfg.add_state("main")
    src_acc = state.add_access("src")
    dst_acc = state.add_access("dst")

    src_sub = src_subset or ", ".join(f"0:{s}" for s in src_shape)
    dst_sub = dst_subset or ", ".join(f"0:{s}" for s in dst_shape)

    libnode = CopyLibraryNode(name="cp")
    if implementation is not None:
        libnode.implementation = implementation
    state.add_edge(src_acc, None, libnode, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.memlet.Memlet(f"src[{src_sub}]"))
    state.add_edge(libnode, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, dst_acc, None, dace.memlet.Memlet(f"dst[{dst_sub}]"))
    return sdfg, libnode


def _make_memset_sdfg(
    shape: Sequence[int],
    subset: str,
    storage: dtypes.StorageType = dtypes.StorageType.CPU_Heap,
    implementation: Optional[str] = None,
    name: str = "memset_sdfg",
    dtype: dace.dtypes.typeclass = dace.float64,
    python_backend: bool = False,
) -> Tuple[dace.SDFG, MemsetLibraryNode]:
    """Build a one-state SDFG that zeros a sub-region via MemsetLibraryNode.

    :param shape: array shape.
    :param subset: memlet subset string for the memset output edge.
    :param storage: array storage type.
    :param implementation: pinned implementation name (None keeps 'Auto').
    :param name: SDFG name.
    :param dtype: element type.
    :param python_backend: if True, set backend to Python.
    :returns: ``(sdfg, libnode)``.
    """
    sdfg = dace.SDFG(name)
    if python_backend:
        sdfg.backend = dtypes.BackendLanguage.Python
    arr_name = "B"
    sdfg.add_array(arr_name, list(shape), dtype, storage=storage)

    state = sdfg.add_state("main")
    out_acc = state.add_access(arr_name)
    libnode = MemsetLibraryNode(name="mset")
    if implementation is not None:
        libnode.implementation = implementation
    state.add_edge(libnode, MemsetLibraryNode.OUTPUT_CONNECTOR_NAME, out_acc, None,
                   dace.memlet.Memlet(f"{arr_name}[{subset}]"))
    return sdfg, libnode


# =============================================================================
# is_python_backend helper
# =============================================================================


class TestIsPythonBackend:
    """Tests for :func:`is_python_backend`."""

    def test_returns_true_for_python_backend(self):
        sdfg = dace.SDFG("test_is_py")
        sdfg.backend = dtypes.BackendLanguage.Python
        state = sdfg.add_state("s")
        assert is_python_backend(state) is True

    def test_returns_false_for_default_backend(self):
        sdfg = dace.SDFG("test_not_py")
        state = sdfg.add_state("s")
        assert is_python_backend(state) is False

    def test_returns_false_for_cpp_backend(self):
        sdfg = dace.SDFG("test_cpp")
        sdfg.backend = dtypes.BackendLanguage.CPP
        state = sdfg.add_state("s")
        assert is_python_backend(state) is False


# =============================================================================
# CopyLibraryNode ExpandPython -- registration
# =============================================================================


class TestCopyPythonRegistration:
    """Verify that all Python expansions are in CopyLibraryNode.implementations."""

    def test_python_in_implementations(self):
        assert "Python" in CopyLibraryNode.implementations

    def test_python_h2d_in_implementations(self):
        assert "PythonH2D" in CopyLibraryNode.implementations

    def test_python_d2h_in_implementations(self):
        assert "PythonD2H" in CopyLibraryNode.implementations


# =============================================================================
# CopyLibraryNode -- auto-selector picks 'Python' for Python backend
# =============================================================================


class TestCopyAutoSelectorPythonBackend:
    """``select_copy_implementation`` returns ``'Python'`` when backend is Python."""

    def test_same_storage_cpu(self):
        sdfg, libnode = _make_copy_sdfg([100], [100], python_backend=True, name="sel_py_cpu")
        state = sdfg.start_state
        result = select_copy_implementation(libnode, state)
        assert result == "Python"

    def test_single_element(self):
        """Single-element copies route to 'Tasklet' under Python backend."""
        sdfg, libnode = _make_copy_sdfg([100], [100],
                                        src_subset="42",
                                        dst_subset="7",
                                        python_backend=True,
                                        name="sel_py_single")
        state = sdfg.start_state
        result = select_copy_implementation(libnode, state)
        assert result == "Tasklet"

    def test_multi_dim(self):
        sdfg, libnode = _make_copy_sdfg([10, 20], [10, 20], python_backend=True, name="sel_py_md")
        state = sdfg.start_state
        result = select_copy_implementation(libnode, state)
        assert result == "Python"

    def test_cross_storage_cpu_gpu(self):
        """Cross-storage CPU->GPU returns 'PythonH2D' under Python backend."""
        sdfg, libnode = _make_copy_sdfg([64], [64],
                                        src_storage=dtypes.StorageType.CPU_Heap,
                                        dst_storage=dtypes.StorageType.GPU_Global,
                                        python_backend=True,
                                        name="sel_py_cross")
        state = sdfg.start_state
        result = select_copy_implementation(libnode, state)
        assert result == "PythonH2D"

    def test_cross_storage_gpu_cpu(self):
        """Cross-storage GPU->CPU returns 'PythonD2H' under Python backend."""
        sdfg, libnode = _make_copy_sdfg([64], [64],
                                        src_storage=dtypes.StorageType.GPU_Global,
                                        dst_storage=dtypes.StorageType.CPU_Heap,
                                        python_backend=True,
                                        name="sel_py_cross_d2h")
        state = sdfg.start_state
        result = select_copy_implementation(libnode, state)
        assert result == "PythonD2H"

    def test_non_python_backend_does_not_return_python(self):
        """Without Python backend, auto-selector picks something else."""
        sdfg, libnode = _make_copy_sdfg([100], [100], name="sel_not_py")
        state = sdfg.start_state
        result = select_copy_implementation(libnode, state)
        assert result != "Python"


# =============================================================================
# CopyLibraryNode ExpandPython -- structural tests
# =============================================================================


class TestCopyPythonExpansionStructure:
    """Verify the expansion's NestedSDFG structure (no compile/run).

    ``ExpandPython`` returns a :class:`~dace.SDFG` (NestedSDFG wrapper)
    containing an ``AccessNode -> Tasklet -> AccessNode`` graph whose
    Tasklet code is ``_out = _in``.
    """

    def test_expansion_produces_nsdfg(self):
        """ExpandPython.expansion returns an SDFG (NestedSDFG wrapper)."""
        sdfg, libnode = _make_copy_sdfg([50], [50], implementation="Python", name="struc_tasklet")
        state = sdfg.start_state
        from dace.libraries.standard.nodes.copy_node import ExpandPython
        result = ExpandPython.expansion(libnode, state, sdfg)
        assert isinstance(result, dace.SDFG)

    def test_nsdfg_contains_tasklet_with_correct_connectors(self):
        """The inner NestedSDFG has a Tasklet with _in/_out connectors."""
        sdfg, libnode = _make_copy_sdfg([50], [50], implementation="Python", name="struc_conn")
        state = sdfg.start_state
        from dace.libraries.standard.nodes.copy_node import ExpandPython
        result = ExpandPython.expansion(libnode, state, sdfg)
        # Find the compute state (the one with nodes; skip the symbol guard)
        compute_states = [s for s in result.states() if len(s.nodes()) > 0]
        assert len(compute_states) == 1
        tasklets = [n for n in compute_states[0].nodes() if isinstance(n, nodes.Tasklet)]
        assert len(tasklets) == 1
        assert "_in" in tasklets[0].in_connectors
        assert "_out" in tasklets[0].out_connectors

    def test_tasklet_code_is_assignment(self):
        """The inner Tasklet code is a simple assignment ``_out = _in``."""
        sdfg, libnode = _make_copy_sdfg([50], [50], implementation="Python", name="struc_code")
        state = sdfg.start_state
        from dace.libraries.standard.nodes.copy_node import ExpandPython
        result = ExpandPython.expansion(libnode, state, sdfg)
        compute_states = [s for s in result.states() if len(s.nodes()) > 0]
        tasklets = [n for n in compute_states[0].nodes() if isinstance(n, nodes.Tasklet)]
        assert "_out = _in" in tasklets[0].code.as_string

    def test_tasklet_uses_python_language(self):
        """The inner Tasklet language is Python."""
        sdfg, libnode = _make_copy_sdfg([50], [50], implementation="Python", name="struc_lang")
        state = sdfg.start_state
        from dace.libraries.standard.nodes.copy_node import ExpandPython
        result = ExpandPython.expansion(libnode, state, sdfg)
        compute_states = [s for s in result.states() if len(s.nodes()) > 0]
        tasklets = [n for n in compute_states[0].nodes() if isinstance(n, nodes.Tasklet)]
        assert tasklets[0].language == dace.Language.Python

    def test_expansion_multi_dim(self):
        """ExpandPython handles multi-dimensional copies (NestedSDFG)."""
        sdfg, libnode = _make_copy_sdfg([10, 20], [10, 20], implementation="Python", name="struc_md")
        state = sdfg.start_state
        from dace.libraries.standard.nodes.copy_node import ExpandPython
        result = ExpandPython.expansion(libnode, state, sdfg)
        assert isinstance(result, dace.SDFG)

    def test_d2h_expansion_produces_nsdfg_with_get_out(self):
        """ExpandPythonD2H returns a NestedSDFG with .get(out=...) in the Tasklet."""
        sdfg, libnode = _make_copy_sdfg([64], [64],
                                        src_storage=dtypes.StorageType.GPU_Global,
                                        dst_storage=dtypes.StorageType.CPU_Heap,
                                        implementation="PythonD2H",
                                        name="struc_d2h")
        state = sdfg.start_state
        from dace.libraries.standard.nodes.copy_node import ExpandPythonD2H
        result = ExpandPythonD2H.expansion(libnode, state, sdfg)
        assert isinstance(result, dace.SDFG)
        inner_state = result.start_state
        tasklets = [n for n in inner_state.nodes() if isinstance(n, nodes.Tasklet)]
        assert len(tasklets) == 1
        assert ".get(out=" in tasklets[0].code.as_string

    def test_h2d_expansion_produces_nsdfg_with_set(self):
        """ExpandPythonH2D returns a NestedSDFG with .set() in the Tasklet."""
        sdfg, libnode = _make_copy_sdfg([64], [64],
                                        src_storage=dtypes.StorageType.CPU_Heap,
                                        dst_storage=dtypes.StorageType.GPU_Global,
                                        implementation="PythonH2D",
                                        name="struc_h2d")
        state = sdfg.start_state
        from dace.libraries.standard.nodes.copy_node import ExpandPythonH2D
        result = ExpandPythonH2D.expansion(libnode, state, sdfg)
        assert isinstance(result, dace.SDFG)
        inner_state = result.start_state
        tasklets = [n for n in inner_state.nodes() if isinstance(n, nodes.Tasklet)]
        assert len(tasklets) == 1
        assert ".set(" in tasklets[0].code.as_string


# =============================================================================
# CopyLibraryNode ExpandPython -- compile + run (CPU_Heap, Python backend)
# =============================================================================


class TestCopyPythonCompileRun:
    """End-to-end compile+run tests with the Python backend."""

    def test_1d_full_copy(self):
        """1D full-array CPU_Heap -> CPU_Heap copy via Python expansion."""
        sdfg, libnode = _make_copy_sdfg([100], [100], python_backend=True, name="py_copy_1d")
        sdfg.validate()
        sdfg.expand_library_nodes()
        assert libnode.implementation == "Python"
        sdfg.validate()
        exe = sdfg.compile()

        src = np.arange(100, dtype=np.float64)
        dst = np.zeros(100, dtype=np.float64)
        exe(src=src, dst=dst)
        np.testing.assert_array_equal(dst, src)

    def test_1d_slice_copy(self):
        """1D slice copy: src[50:100] -> dst[0:50]."""
        sdfg, libnode = _make_copy_sdfg([200], [200],
                                        src_subset="50:100",
                                        dst_subset="0:50",
                                        python_backend=True,
                                        name="py_copy_1d_slice")
        sdfg.validate()
        sdfg.expand_library_nodes()
        assert libnode.implementation == "Python"
        sdfg.validate()
        exe = sdfg.compile()

        src = np.arange(200, dtype=np.float64)
        dst = np.zeros(200, dtype=np.float64)
        exe(src=src, dst=dst)
        np.testing.assert_array_equal(dst[:50], src[50:100])
        assert np.all(dst[50:] == 0)

    def test_2d_full_copy(self):
        """2D full-array CPU_Heap -> CPU_Heap copy via Python expansion."""
        sdfg, libnode = _make_copy_sdfg([8, 16], [8, 16], python_backend=True, name="py_copy_2d")
        sdfg.validate()
        sdfg.expand_library_nodes()
        assert libnode.implementation == "Python"
        sdfg.validate()
        exe = sdfg.compile()

        src = np.arange(128, dtype=np.float64).reshape(8, 16)
        dst = np.zeros((8, 16), dtype=np.float64)
        exe(src=src, dst=dst)
        np.testing.assert_array_equal(dst, src)

    def test_2d_subblock_copy(self):
        """2D sub-block copy: src[2:6, 4:12] -> dst[2:6, 4:12]."""
        sdfg, libnode = _make_copy_sdfg([10, 20], [10, 20],
                                        src_subset="2:6, 4:12",
                                        dst_subset="2:6, 4:12",
                                        python_backend=True,
                                        name="py_copy_2d_sub")
        sdfg.validate()
        sdfg.expand_library_nodes()
        assert libnode.implementation == "Python"
        sdfg.validate()
        exe = sdfg.compile()

        src = np.arange(200, dtype=np.float64).reshape(10, 20)
        dst = np.zeros((10, 20), dtype=np.float64)
        exe(src=src, dst=dst)
        np.testing.assert_array_equal(dst[2:6, 4:12], src[2:6, 4:12])
        # Untouched region stays zero
        mask = np.ones((10, 20), dtype=bool)
        mask[2:6, 4:12] = False
        assert np.all(dst[mask] == 0)

    def test_float32_dtype(self):
        """Copy with float32 dtype."""
        sdfg, libnode = _make_copy_sdfg([64], [64], dtype=dace.float32, python_backend=True, name="py_copy_f32")
        sdfg.validate()
        sdfg.expand_library_nodes()
        sdfg.validate()
        exe = sdfg.compile()

        src = np.arange(64, dtype=np.float32)
        dst = np.zeros(64, dtype=np.float32)
        exe(src=src, dst=dst)
        np.testing.assert_array_equal(dst, src)

    def test_int32_dtype(self):
        """Copy with int32 dtype."""
        sdfg, libnode = _make_copy_sdfg([32], [32], dtype=dace.int32, python_backend=True, name="py_copy_i32")
        sdfg.validate()
        sdfg.expand_library_nodes()
        sdfg.validate()
        exe = sdfg.compile()

        src = np.arange(32, dtype=np.int32)
        dst = np.zeros(32, dtype=np.int32)
        exe(src=src, dst=dst)
        np.testing.assert_array_equal(dst, src)

    def test_single_element_copy(self):
        """Single-element copy: auto-selector routes to 'Tasklet' under Python backend."""
        sdfg, libnode = _make_copy_sdfg([100], [100],
                                        src_subset="42",
                                        dst_subset="7",
                                        python_backend=True,
                                        name="py_copy_single_elem")
        sdfg.validate()
        sdfg.expand_library_nodes()
        # Auto-dispatch routes single-element to 'Tasklet', not 'Python'
        assert libnode.implementation == "Tasklet"
        sdfg.validate()
        exe = sdfg.compile()

        src = np.arange(100, dtype=np.float64)
        dst = np.zeros(100, dtype=np.float64)
        exe(src=src, dst=dst)
        assert dst[7] == src[42]
        # All other elements stay zero
        expected = np.zeros(100, dtype=np.float64)
        expected[7] = src[42]
        np.testing.assert_array_equal(dst, expected)

    def test_concrete_size_copy(self):
        """Copy with concrete array size (primary use case for bare Tasklet expansion).

        Note: symbolic sizes (``dace.symbol('N')``) are NOT supported with
        ``ExpandPython``'s bare Tasklet expansion on the Python backend,
        because ``used_symbols(all_symbols=False)`` does not discover
        symbols from AccessNode->Tasklet "view edge" memlets.  The cuTile
        pipeline (the primary consumer) always uses concrete sizes.
        """
        sdfg = dace.SDFG('py_copy_concrete')
        sdfg.add_array('src', [64], dace.float64)
        sdfg.add_array('dst', [64], dace.float64)
        state = sdfg.add_state()
        src_node = state.add_access('src')
        dst_node = state.add_access('dst')
        cpy = CopyLibraryNode(name='_copy_')
        state.add_node(cpy)
        state.add_edge(src_node, None, cpy, CopyLibraryNode.INPUT_CONNECTOR_NAME, dace.Memlet('src[0:64]'))
        state.add_edge(cpy, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, dst_node, None, dace.Memlet('dst[0:64]'))
        sdfg.backend = dtypes.BackendLanguage.Python
        csdfg = sdfg.compile()
        src = np.arange(64, dtype=np.float64)
        dst = np.zeros(64, dtype=np.float64)
        csdfg(src=src, dst=dst)
        np.testing.assert_array_equal(dst, src)


# =============================================================================
# MemsetLibraryNode ExpandPython -- registration
# =============================================================================


class TestMemsetPythonRegistration:
    """Verify that 'Python' is in MemsetLibraryNode.implementations."""

    def test_python_in_implementations(self):
        assert "Python" in MemsetLibraryNode.implementations


# =============================================================================
# MemsetLibraryNode -- auto-selector picks 'Python' for Python backend
# =============================================================================


class TestMemsetAutoSelectorPythonBackend:
    """``select_memset_implementation`` returns the right impl for Python backend."""

    def test_multi_element_returns_python(self):
        sdfg, libnode = _make_memset_sdfg([100], "0:100", python_backend=True, name="msel_py")
        state = sdfg.start_state
        result = select_memset_implementation(libnode, state)
        assert result == "Python"

    def test_single_element_returns_tasklet(self):
        """Single-element memset returns 'tasklet' under Python backend."""
        sdfg, libnode = _make_memset_sdfg([100], "42", python_backend=True, name="msel_py_single")
        state = sdfg.start_state
        result = select_memset_implementation(libnode, state)
        assert result == "tasklet"

    def test_multi_dim_returns_python(self):
        sdfg, libnode = _make_memset_sdfg([10, 20], "2:8, 5:15", python_backend=True, name="msel_py_md")
        state = sdfg.start_state
        result = select_memset_implementation(libnode, state)
        assert result == "Python"

    def test_non_python_backend_does_not_return_python(self):
        """Without Python backend, auto-selector picks something else."""
        sdfg, libnode = _make_memset_sdfg([100], "0:100", name="msel_not_py")
        state = sdfg.start_state
        result = select_memset_implementation(libnode, state)
        assert result != "Python"


# =============================================================================
# MemsetLibraryNode ExpandPython -- structural tests
# =============================================================================


class TestMemsetPythonExpansionStructure:
    """Verify the expansion's SDFG structure (no compile/run)."""

    def test_multi_element_expansion_produces_sdfg(self):
        """Multi-element ExpandPython returns an SDFG."""
        sdfg, libnode = _make_memset_sdfg([50], "0:50", implementation="Python", name="mstruc_sdfg")
        state = sdfg.start_state
        from dace.libraries.standard.nodes.memset_node import ExpandPython
        result = ExpandPython.expansion(libnode, state, sdfg)
        assert isinstance(result, dace.SDFG)

    def test_multi_element_has_mapped_tasklet(self):
        """Multi-element expansion contains a MapEntry + Tasklet."""
        sdfg, libnode = _make_memset_sdfg([50], "0:50", implementation="Python", name="mstruc_map")
        state = sdfg.start_state
        from dace.libraries.standard.nodes.memset_node import ExpandPython
        inner_sdfg = ExpandPython.expansion(libnode, state, sdfg)
        inner_state = inner_sdfg.start_state
        map_entries = [n for n in inner_state.nodes() if isinstance(n, nodes.MapEntry)]
        tasklets = [n for n in inner_state.nodes() if isinstance(n, nodes.Tasklet)]
        assert len(map_entries) == 1
        assert len(tasklets) == 1

    def test_multi_element_sequential_schedule(self):
        """The map in the expansion uses Sequential schedule."""
        sdfg, libnode = _make_memset_sdfg([50], "0:50", implementation="Python", name="mstruc_sched")
        state = sdfg.start_state
        from dace.libraries.standard.nodes.memset_node import ExpandPython
        inner_sdfg = ExpandPython.expansion(libnode, state, sdfg)
        inner_state = inner_sdfg.start_state
        map_entries = [n for n in inner_state.nodes() if isinstance(n, nodes.MapEntry)]
        assert len(map_entries) == 1
        assert map_entries[0].schedule == dtypes.ScheduleType.Sequential

    def test_single_element_returns_tasklet(self):
        """Single-element ExpandPython returns a bare Tasklet."""
        sdfg, libnode = _make_memset_sdfg([100], "42", implementation="Python", name="mstruc_single")
        state = sdfg.start_state
        from dace.libraries.standard.nodes.memset_node import ExpandPython
        result = ExpandPython.expansion(libnode, state, sdfg)
        assert isinstance(result, nodes.Tasklet)

    def test_single_element_tasklet_code_assigns_zero(self):
        """The bare Tasklet's code assigns 0 to the output connector."""
        sdfg, libnode = _make_memset_sdfg([100], "42", implementation="Python", name="mstruc_single_code")
        state = sdfg.start_state
        from dace.libraries.standard.nodes.memset_node import ExpandPython
        result = ExpandPython.expansion(libnode, state, sdfg)
        assert "= 0" in result.code.as_string

    def test_multi_dim_expansion(self):
        """Multi-dimensional memset expansion has the right number of map params."""
        sdfg, libnode = _make_memset_sdfg([10, 20], "2:8, 5:15", implementation="Python", name="mstruc_md")
        state = sdfg.start_state
        from dace.libraries.standard.nodes.memset_node import ExpandPython
        inner_sdfg = ExpandPython.expansion(libnode, state, sdfg)
        inner_state = inner_sdfg.start_state
        map_entries = [n for n in inner_state.nodes() if isinstance(n, nodes.MapEntry)]
        assert len(map_entries) == 1
        # 2D non-trivial subset -> 2 map params
        assert len(map_entries[0].params) == 2


# =============================================================================
# MemsetLibraryNode ExpandPython -- compile + run (CPU_Heap, Python backend)
# =============================================================================


class TestMemsetPythonCompileRun:
    """End-to-end compile+run tests with the Python backend."""

    def test_1d_full_memset(self):
        """1D full-array memset zeros all elements."""
        sdfg, libnode = _make_memset_sdfg([100], "0:100", python_backend=True, name="py_mset_1d")
        sdfg.validate()
        sdfg.expand_library_nodes()
        assert libnode.implementation == "Python"
        sdfg.validate()
        exe = sdfg.compile()

        B = np.ones(100, dtype=np.float64)
        exe(B=B)
        assert np.all(B == 0)

    def test_1d_slice_memset(self):
        """1D slice memset: zeros B[25:75], rest stays untouched."""
        sdfg, libnode = _make_memset_sdfg([200], "25:75", python_backend=True, name="py_mset_1d_slice")
        sdfg.validate()
        sdfg.expand_library_nodes()
        assert libnode.implementation == "Python"
        sdfg.validate()
        exe = sdfg.compile()

        B = np.ones(200, dtype=np.float64)
        exe(B=B)
        assert np.all(B[:25] == 1)
        assert np.all(B[75:] == 1)
        assert np.all(B[25:75] == 0)

    def test_2d_subblock_memset(self):
        """2D sub-block memset zeros only the specified region."""
        sdfg, libnode = _make_memset_sdfg([10, 20], "2:8, 5:15", python_backend=True, name="py_mset_2d")
        sdfg.validate()
        sdfg.expand_library_nodes()
        assert libnode.implementation == "Python"
        sdfg.validate()
        exe = sdfg.compile()

        B = np.ones((10, 20), dtype=np.float64)
        exe(B=B)
        # Zeroed region
        assert np.all(B[2:8, 5:15] == 0)
        # Untouched border
        expected = np.ones((10, 20), dtype=np.float64)
        for i in range(2, 8):
            for j in range(5, 15):
                expected[i, j] = 0
        np.testing.assert_array_equal(B, expected)

    def test_single_element_memset(self):
        """Single-element memset via 'tasklet' expansion on Python backend."""
        sdfg, libnode = _make_memset_sdfg([100], "42", python_backend=True, name="py_mset_single")
        sdfg.validate()
        sdfg.expand_library_nodes()
        # Single element routes to 'tasklet', not 'Python'
        assert libnode.implementation == "tasklet"
        sdfg.validate()
        exe = sdfg.compile()

        B = np.ones(100, dtype=np.float64)
        exe(B=B)
        assert B[42] == 0
        assert np.all(B[:42] == 1)
        assert np.all(B[43:] == 1)

    def test_float32_dtype(self):
        """Memset with float32 dtype."""
        sdfg, libnode = _make_memset_sdfg([64], "0:64", dtype=dace.float32, python_backend=True, name="py_mset_f32")
        sdfg.validate()
        sdfg.expand_library_nodes()
        sdfg.validate()
        exe = sdfg.compile()

        B = np.ones(64, dtype=np.float32)
        exe(B=B)
        assert np.all(B == 0)

    def test_3d_full_memset(self):
        """3D full-array memset zeros all elements."""
        sdfg, libnode = _make_memset_sdfg([4, 5, 6], "0:4, 0:5, 0:6", python_backend=True, name="py_mset_3d")
        sdfg.validate()
        sdfg.expand_library_nodes()
        assert libnode.implementation == "Python"
        sdfg.validate()
        exe = sdfg.compile()

        B = np.ones((4, 5, 6), dtype=np.float64)
        exe(B=B)
        assert np.all(B == 0)

    def test_symbolic_size_memset(self):
        """Memset with symbolic array size N."""
        N = dace.symbol('N')
        sdfg = dace.SDFG('py_memset_symbolic')
        sdfg.add_array('arr', [N], dace.float64)
        state = sdfg.add_state()
        arr_node = state.add_access('arr')
        mset = MemsetLibraryNode(name='_memset_')
        state.add_node(mset)
        state.add_edge(mset, MemsetLibraryNode.OUTPUT_CONNECTOR_NAME, arr_node, None, dace.Memlet('arr[0:N]'))
        sdfg.backend = dtypes.BackendLanguage.Python
        csdfg = sdfg.compile()
        arr = np.ones(48, dtype=np.float64)
        csdfg(arr=arr, N=48)
        np.testing.assert_array_equal(arr, np.zeros(48))


# =============================================================================
# CopyLibraryNode ExpandPython via Auto dispatch (end-to-end)
# =============================================================================


class TestCopyPythonAutoDispatch:
    """Auto-dispatched copy with Python backend calls ExpandPython."""

    def test_auto_dispatch_picks_python_and_runs(self):
        """Auto dispatch selects Python expansion and the result is correct."""
        sdfg, libnode = _make_copy_sdfg([50], [50], implementation=None, python_backend=True, name="py_copy_auto")
        sdfg.validate()
        sdfg.expand_library_nodes()
        assert libnode.implementation == "Python"
        sdfg.validate()
        exe = sdfg.compile()

        src = np.arange(50, dtype=np.float64)
        dst = np.zeros(50, dtype=np.float64)
        exe(src=src, dst=dst)
        np.testing.assert_array_equal(dst, src)


# =============================================================================
# MemsetLibraryNode ExpandPython via Auto dispatch (end-to-end)
# =============================================================================


class TestMemsetPythonAutoDispatch:
    """Auto-dispatched memset with Python backend calls ExpandPython."""

    def test_auto_dispatch_picks_python_and_runs(self):
        """Auto dispatch selects Python expansion and the result is correct."""
        sdfg, libnode = _make_memset_sdfg([80], "10:70", implementation=None, python_backend=True, name="py_mset_auto")
        sdfg.validate()
        sdfg.expand_library_nodes()
        assert libnode.implementation == "Python"
        sdfg.validate()
        exe = sdfg.compile()

        B = np.ones(80, dtype=np.float64)
        exe(B=B)
        assert np.all(B[:10] == 1)
        assert np.all(B[70:] == 1)
        assert np.all(B[10:70] == 0)


if __name__ == "__main__":
    pytest.main([__file__])
