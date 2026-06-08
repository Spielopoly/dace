# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Runtime integration tests for cuTile NestedSDFG code generation.

Tests compile and execute cuTile kernels containing NestedSDFGs on GPU,
comparing results against NumPy references.  NestedSDFGs are built manually
inside CuTile map scopes with tile transient storage.

Single-state and multi-state inner SDFGs are covered, as well as symbolic
sizes, multiple dtypes, interstate assignments, and recursive nesting.

**Key naming convention:** inner SDFG array names match the outer tile
transient names (e.g. ``_tile_a``, ``_tile_out``) so the AccessNode-centric
codegen does not emit a spurious rebinding line after the NestedSDFG call.
"""
import ast

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.sdfg import SDFG
from dace.memlet import Memlet
from dace.dtypes import ScheduleType, StorageType, Language

# All tests in this file require GPU.
pytestmark = pytest.mark.gpu


# ============================================================
# Helpers
# ============================================================


def _run_cutile(sdfg: SDFG, **kwargs):
    """Compile and run a cuTile SDFG, returning results as numpy arrays.

    Uses cupy arrays for GPU execution.  Converts numpy inputs to cupy,
    runs, and converts cupy outputs back to numpy.

    :param sdfg: The SDFG to compile and run.
    :param kwargs: Named arguments for the SDFG (arrays and symbols).
    :returns: Dictionary mapping array names to numpy results.
    """
    import cupy as cp

    cp_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
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


def _build_single_state_nsdfg_sdfg(
    name: str,
    tasklet_code: str,
    dtype=dace.float64,
    W: int = 8,
):
    """Build an SDFG with a single-state NestedSDFG inside a CuTile map.

    The NestedSDFG has one input (``_tile_a``) and one output (``_tile_out``)
    with a single tasklet whose body is *tasklet_code*.

    The outer SDFG has arrays ``A`` (input) and ``B`` (output) of shape
    ``(N,)`` on GPU_Global, plus tile transients on CuTile_Tile.

    :param name: SDFG name (must be unique per test).
    :param tasklet_code: Python tasklet body with ``x`` as input and
        ``y`` as output (e.g. ``"y = x * 2"``).
    :param dtype: Data type for all arrays.
    :param W: Tile width (must be power of 2).
    :returns: The constructed SDFG.
    """
    N = dace.symbol("N")

    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array("A", (N,), dtype, storage=StorageType.GPU_Global)
    sdfg.add_array("B", (N,), dtype, storage=StorageType.GPU_Global)
    sdfg.add_array(
        "_tile_a", (W,), dtype,
        storage=StorageType.CuTile_Tile, transient=True,
    )
    sdfg.add_array(
        "_tile_out", (W,), dtype,
        storage=StorageType.CuTile_Tile, transient=True,
    )

    state = sdfg.add_state("main")
    me, mx = state.add_map(
        "cutile_map", {"i": f"0:N:{W}"},
        schedule=ScheduleType.CuTile,
    )

    # Inner SDFG: _tile_a -> tasklet -> _tile_out  (single state)
    inner_sdfg = SDFG(f"{name}_inner")
    inner_sdfg.add_array(
        "_tile_a", (W,), dtype, storage=StorageType.CuTile_Tile,
    )
    inner_sdfg.add_array(
        "_tile_out", (W,), dtype, storage=StorageType.CuTile_Tile,
    )
    inner_state = inner_sdfg.add_state("compute")
    in_node = inner_state.add_read("_tile_a")
    out_node = inner_state.add_write("_tile_out")
    tasklet = inner_state.add_tasklet(
        "op", {"x"}, {"y"}, tasklet_code, language=Language.Python,
    )
    inner_state.add_edge(
        in_node, None, tasklet, "x",
        Memlet(data="_tile_a", subset=f"0:{W}"),
    )
    inner_state.add_edge(
        tasklet, "y", out_node, None,
        Memlet(data="_tile_out", subset=f"0:{W}"),
    )

    nsdfg = state.add_nested_sdfg(inner_sdfg, {"_tile_a"}, {"_tile_out"})

    a_node = state.add_read("A")
    b_node = state.add_write("B")
    tile_a = state.add_access("_tile_a")
    tile_out = state.add_access("_tile_out")

    state.add_memlet_path(
        a_node, me, tile_a,
        memlet=Memlet(data="A", subset=f"i:i+{W}"),
    )
    state.add_edge(
        tile_a, None, nsdfg, "_tile_a",
        Memlet(data="_tile_a", subset=f"0:{W}"),
    )
    state.add_edge(
        nsdfg, "_tile_out", tile_out, None,
        Memlet(data="_tile_out", subset=f"0:{W}"),
    )
    state.add_memlet_path(
        tile_out, mx, b_node,
        memlet=Memlet(data="B", subset=f"i:i+{W}"),
    )
    return sdfg


def _build_two_input_nsdfg_sdfg(
    name: str,
    tasklet_code: str,
    dtype=dace.float64,
    W: int = 8,
):
    """Build an SDFG with a single-state NestedSDFG that takes two inputs.

    The NestedSDFG has inputs ``_tile_a``, ``_tile_b`` and output
    ``_tile_out``.  The tasklet body uses ``x`` and ``z`` as inputs,
    ``y`` as output.

    :param name: SDFG name.
    :param tasklet_code: Tasklet body with ``x``, ``z`` -> ``y``.
    :param dtype: Data type.
    :param W: Tile width.
    :returns: The constructed SDFG.
    """
    N = dace.symbol("N")

    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array("A", (N,), dtype, storage=StorageType.GPU_Global)
    sdfg.add_array("B", (N,), dtype, storage=StorageType.GPU_Global)
    sdfg.add_array("C", (N,), dtype, storage=StorageType.GPU_Global)
    sdfg.add_array(
        "_tile_a", (W,), dtype,
        storage=StorageType.CuTile_Tile, transient=True,
    )
    sdfg.add_array(
        "_tile_b", (W,), dtype,
        storage=StorageType.CuTile_Tile, transient=True,
    )
    sdfg.add_array(
        "_tile_out", (W,), dtype,
        storage=StorageType.CuTile_Tile, transient=True,
    )

    state = sdfg.add_state("main")
    me, mx = state.add_map(
        "cutile_map", {"i": f"0:N:{W}"},
        schedule=ScheduleType.CuTile,
    )

    # Inner SDFG: (_tile_a, _tile_b) -> tasklet -> _tile_out
    inner_sdfg = SDFG(f"{name}_inner")
    inner_sdfg.add_array(
        "_tile_a", (W,), dtype, storage=StorageType.CuTile_Tile,
    )
    inner_sdfg.add_array(
        "_tile_b", (W,), dtype, storage=StorageType.CuTile_Tile,
    )
    inner_sdfg.add_array(
        "_tile_out", (W,), dtype, storage=StorageType.CuTile_Tile,
    )
    inner_state = inner_sdfg.add_state("compute")
    in_a = inner_state.add_read("_tile_a")
    in_b = inner_state.add_read("_tile_b")
    out_node = inner_state.add_write("_tile_out")
    tasklet = inner_state.add_tasklet(
        "op", {"x", "z"}, {"y"}, tasklet_code, language=Language.Python,
    )
    inner_state.add_edge(
        in_a, None, tasklet, "x",
        Memlet(data="_tile_a", subset=f"0:{W}"),
    )
    inner_state.add_edge(
        in_b, None, tasklet, "z",
        Memlet(data="_tile_b", subset=f"0:{W}"),
    )
    inner_state.add_edge(
        tasklet, "y", out_node, None,
        Memlet(data="_tile_out", subset=f"0:{W}"),
    )

    nsdfg = state.add_nested_sdfg(
        inner_sdfg, {"_tile_a", "_tile_b"}, {"_tile_out"},
    )

    a_node = state.add_read("A")
    b_node = state.add_read("B")
    c_node = state.add_write("C")
    tile_a = state.add_access("_tile_a")
    tile_b = state.add_access("_tile_b")
    tile_out = state.add_access("_tile_out")

    state.add_memlet_path(
        a_node, me, tile_a,
        memlet=Memlet(data="A", subset=f"i:i+{W}"),
    )
    state.add_memlet_path(
        b_node, me, tile_b,
        memlet=Memlet(data="B", subset=f"i:i+{W}"),
    )
    state.add_edge(
        tile_a, None, nsdfg, "_tile_a",
        Memlet(data="_tile_a", subset=f"0:{W}"),
    )
    state.add_edge(
        tile_b, None, nsdfg, "_tile_b",
        Memlet(data="_tile_b", subset=f"0:{W}"),
    )
    state.add_edge(
        nsdfg, "_tile_out", tile_out, None,
        Memlet(data="_tile_out", subset=f"0:{W}"),
    )
    state.add_memlet_path(
        tile_out, mx, c_node,
        memlet=Memlet(data="C", subset=f"i:i+{W}"),
    )
    return sdfg


# ============================================================
# 1. Single-state NestedSDFG runtime tests
# ============================================================


class TestSingleStateNestedSDFGRuntime:
    """Single-state NestedSDFGs inside CuTile scopes execute correctly."""

    def test_double_aligned(self):
        """B[i] = A[i] * 2, N=64 (aligned to tile width 8)."""
        sdfg = _build_single_state_nsdfg_sdfg(
            "rt_ss_double_64", "y = x * 2",
        )
        n = 64
        rng = np.random.default_rng(100)
        A = rng.random(n)
        B = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, N=n)
        np.testing.assert_allclose(results["B"], A * 2, rtol=1e-14)

    def test_negate(self):
        """B[i] = -A[i], N=64."""
        sdfg = _build_single_state_nsdfg_sdfg(
            "rt_ss_neg_64", "y = -x",
        )
        n = 64
        rng = np.random.default_rng(101)
        A = rng.random(n)
        B = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, N=n)
        np.testing.assert_allclose(results["B"], -A, rtol=1e-14)

    def test_add_constant(self):
        """B[i] = A[i] + 42.0, N=128."""
        sdfg = _build_single_state_nsdfg_sdfg(
            "rt_ss_add42_128", "y = x + 42.0",
        )
        n = 128
        rng = np.random.default_rng(102)
        A = rng.random(n)
        B = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, N=n)
        np.testing.assert_allclose(results["B"], A + 42.0, rtol=1e-14)

    def test_binop_add(self):
        """C[i] = A[i] + B[i] via two-input NestedSDFG, N=64."""
        sdfg = _build_two_input_nsdfg_sdfg(
            "rt_ss_add_64", "y = x + z",
        )
        n = 64
        rng = np.random.default_rng(103)
        A = rng.random(n)
        B = rng.random(n)
        C = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(results["C"], A + B, rtol=1e-14)

    def test_binop_mul(self):
        """C[i] = A[i] * B[i] via two-input NestedSDFG, N=64."""
        sdfg = _build_two_input_nsdfg_sdfg(
            "rt_ss_mul_64", "y = x * z",
        )
        n = 64
        rng = np.random.default_rng(104)
        A = rng.random(n)
        B = rng.random(n)
        C = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, C=C, N=n)
        np.testing.assert_allclose(results["C"], A * B, rtol=1e-14)

    @pytest.mark.parametrize("n", [8, 16, 32, 64, 128, 256])
    def test_various_aligned_sizes(self, n):
        """B[i] = A[i] * 2 for several aligned sizes."""
        sdfg = _build_single_state_nsdfg_sdfg(
            f"rt_ss_double_{n}", "y = x * 2",
        )
        rng = np.random.default_rng(n + 200)
        A = rng.random(n)
        B = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, N=n)
        np.testing.assert_allclose(results["B"], A * 2, rtol=1e-14)

    def test_codegen_contains_nested_function(self):
        """The generated code should contain a module-level NestedSDFG function."""
        sdfg = _build_single_state_nsdfg_sdfg(
            "rt_ss_codegen", "y = x * 2",
        )
        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        assert "def __dace_nested_" in code
        assert "@ct.kernel" in code
        # The generated code should be valid Python.
        ast.parse(code)

    def test_codegen_function_called_inside_kernel(self):
        """The NestedSDFG function should be called inside the kernel."""
        sdfg = _build_single_state_nsdfg_sdfg(
            "rt_ss_codegen_call", "y = x + 1",
        )
        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        # Find the kernel function body.
        kernel_start = code.index("@ct.kernel")
        kernel_body = code[kernel_start:]
        assert "__dace_nested_" in kernel_body


# ============================================================
# 2. Multi-state NestedSDFG runtime tests
# ============================================================


def _build_two_state_nsdfg_sdfg(name: str, dtype=dace.float64, W: int = 8):
    """Build SDFG with a two-state NestedSDFG: B = A * 2 + 1.

    Inner SDFG:
        state1: _tmp = _tile_a * 2
        state2: _tile_out = _tmp + 1

    :param name: SDFG name.
    :param dtype: Data type.
    :param W: Tile width.
    :returns: The constructed SDFG.
    """
    N = dace.symbol("N")

    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array("A", (N,), dtype, storage=StorageType.GPU_Global)
    sdfg.add_array("B", (N,), dtype, storage=StorageType.GPU_Global)
    sdfg.add_array(
        "_tile_a", (W,), dtype,
        storage=StorageType.CuTile_Tile, transient=True,
    )
    sdfg.add_array(
        "_tile_out", (W,), dtype,
        storage=StorageType.CuTile_Tile, transient=True,
    )

    state = sdfg.add_state("main")
    me, mx = state.add_map(
        "cutile_map", {"i": f"0:N:{W}"},
        schedule=ScheduleType.CuTile,
    )

    # Inner SDFG: 2 states
    inner_sdfg = SDFG(f"{name}_inner")
    inner_sdfg.add_array(
        "_tile_a", (W,), dtype, storage=StorageType.CuTile_Tile,
    )
    inner_sdfg.add_array(
        "_tmp", (W,), dtype,
        storage=StorageType.CuTile_Tile, transient=True,
    )
    inner_sdfg.add_array(
        "_tile_out", (W,), dtype, storage=StorageType.CuTile_Tile,
    )

    s1 = inner_sdfg.add_state("double")
    s2 = inner_sdfg.add_state("add_one")
    inner_sdfg.add_edge(s1, s2, dace.InterstateEdge())

    # state1: _tmp = _tile_a * 2
    s1_in = s1.add_read("_tile_a")
    s1_out = s1.add_write("_tmp")
    t1 = s1.add_tasklet(
        "mul2", {"x"}, {"y"}, "y = x * 2", language=Language.Python,
    )
    s1.add_edge(s1_in, None, t1, "x", Memlet(data="_tile_a", subset=f"0:{W}"))
    s1.add_edge(t1, "y", s1_out, None, Memlet(data="_tmp", subset=f"0:{W}"))

    # state2: _tile_out = _tmp + 1
    s2_in = s2.add_read("_tmp")
    s2_out = s2.add_write("_tile_out")
    t2 = s2.add_tasklet(
        "add1", {"x"}, {"y"}, "y = x + 1", language=Language.Python,
    )
    s2.add_edge(s2_in, None, t2, "x", Memlet(data="_tmp", subset=f"0:{W}"))
    s2.add_edge(t2, "y", s2_out, None, Memlet(data="_tile_out", subset=f"0:{W}"))

    nsdfg = state.add_nested_sdfg(inner_sdfg, {"_tile_a"}, {"_tile_out"})

    a_node = state.add_read("A")
    b_node = state.add_write("B")
    tile_a = state.add_access("_tile_a")
    tile_out = state.add_access("_tile_out")

    state.add_memlet_path(
        a_node, me, tile_a,
        memlet=Memlet(data="A", subset=f"i:i+{W}"),
    )
    state.add_edge(
        tile_a, None, nsdfg, "_tile_a",
        Memlet(data="_tile_a", subset=f"0:{W}"),
    )
    state.add_edge(
        nsdfg, "_tile_out", tile_out, None,
        Memlet(data="_tile_out", subset=f"0:{W}"),
    )
    state.add_memlet_path(
        tile_out, mx, b_node,
        memlet=Memlet(data="B", subset=f"i:i+{W}"),
    )
    return sdfg


def _build_three_state_chain_sdfg(name: str, dtype=dace.float64, W: int = 8):
    """Build SDFG with a three-state chain: B = (A + 1) * 2 - 3.

    Inner SDFG:
        state1: _t1 = _tile_a + 1
        state2: _t2 = _t1 * 2
        state3: _tile_out = _t2 - 3

    :param name: SDFG name.
    :param dtype: Data type.
    :param W: Tile width.
    :returns: The constructed SDFG.
    """
    N = dace.symbol("N")

    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array("A", (N,), dtype, storage=StorageType.GPU_Global)
    sdfg.add_array("B", (N,), dtype, storage=StorageType.GPU_Global)
    sdfg.add_array(
        "_tile_a", (W,), dtype,
        storage=StorageType.CuTile_Tile, transient=True,
    )
    sdfg.add_array(
        "_tile_out", (W,), dtype,
        storage=StorageType.CuTile_Tile, transient=True,
    )

    state = sdfg.add_state("main")
    me, mx = state.add_map(
        "cutile_map", {"i": f"0:N:{W}"},
        schedule=ScheduleType.CuTile,
    )

    # Inner SDFG: 3 states
    inner_sdfg = SDFG(f"{name}_inner")
    inner_sdfg.add_array(
        "_tile_a", (W,), dtype, storage=StorageType.CuTile_Tile,
    )
    inner_sdfg.add_array(
        "_t1", (W,), dtype,
        storage=StorageType.CuTile_Tile, transient=True,
    )
    inner_sdfg.add_array(
        "_t2", (W,), dtype,
        storage=StorageType.CuTile_Tile, transient=True,
    )
    inner_sdfg.add_array(
        "_tile_out", (W,), dtype, storage=StorageType.CuTile_Tile,
    )

    s1 = inner_sdfg.add_state("add_one")
    s2 = inner_sdfg.add_state("mul_two")
    s3 = inner_sdfg.add_state("sub_three")
    inner_sdfg.add_edge(s1, s2, dace.InterstateEdge())
    inner_sdfg.add_edge(s2, s3, dace.InterstateEdge())

    # state1: _t1 = _tile_a + 1
    n1_in = s1.add_read("_tile_a")
    n1_out = s1.add_write("_t1")
    t1 = s1.add_tasklet(
        "add1", {"x"}, {"y"}, "y = x + 1", language=Language.Python,
    )
    s1.add_edge(n1_in, None, t1, "x", Memlet(data="_tile_a", subset=f"0:{W}"))
    s1.add_edge(t1, "y", n1_out, None, Memlet(data="_t1", subset=f"0:{W}"))

    # state2: _t2 = _t1 * 2
    n2_in = s2.add_read("_t1")
    n2_out = s2.add_write("_t2")
    t2 = s2.add_tasklet(
        "mul2", {"x"}, {"y"}, "y = x * 2", language=Language.Python,
    )
    s2.add_edge(n2_in, None, t2, "x", Memlet(data="_t1", subset=f"0:{W}"))
    s2.add_edge(t2, "y", n2_out, None, Memlet(data="_t2", subset=f"0:{W}"))

    # state3: _tile_out = _t2 - 3
    n3_in = s3.add_read("_t2")
    n3_out = s3.add_write("_tile_out")
    t3 = s3.add_tasklet(
        "sub3", {"x"}, {"y"}, "y = x - 3", language=Language.Python,
    )
    s3.add_edge(n3_in, None, t3, "x", Memlet(data="_t2", subset=f"0:{W}"))
    s3.add_edge(t3, "y", n3_out, None, Memlet(data="_tile_out", subset=f"0:{W}"))

    nsdfg = state.add_nested_sdfg(inner_sdfg, {"_tile_a"}, {"_tile_out"})

    a_node = state.add_read("A")
    b_node = state.add_write("B")
    tile_a = state.add_access("_tile_a")
    tile_out = state.add_access("_tile_out")

    state.add_memlet_path(
        a_node, me, tile_a,
        memlet=Memlet(data="A", subset=f"i:i+{W}"),
    )
    state.add_edge(
        tile_a, None, nsdfg, "_tile_a",
        Memlet(data="_tile_a", subset=f"0:{W}"),
    )
    state.add_edge(
        nsdfg, "_tile_out", tile_out, None,
        Memlet(data="_tile_out", subset=f"0:{W}"),
    )
    state.add_memlet_path(
        tile_out, mx, b_node,
        memlet=Memlet(data="B", subset=f"i:i+{W}"),
    )
    return sdfg


def _build_empty_first_state_sdfg(name: str, dtype=dace.float64, W: int = 8):
    """Build SDFG with an empty first state then a compute state.

    Inner SDFG:
        state1: (empty)
        state2: _tile_out = _tile_a + 1.0

    :param name: SDFG name.
    :param dtype: Data type.
    :param W: Tile width.
    :returns: The constructed SDFG.
    """
    N = dace.symbol("N")

    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array("A", (N,), dtype, storage=StorageType.GPU_Global)
    sdfg.add_array("B", (N,), dtype, storage=StorageType.GPU_Global)
    sdfg.add_array(
        "_tile_a", (W,), dtype,
        storage=StorageType.CuTile_Tile, transient=True,
    )
    sdfg.add_array(
        "_tile_out", (W,), dtype,
        storage=StorageType.CuTile_Tile, transient=True,
    )

    state = sdfg.add_state("main")
    me, mx = state.add_map(
        "cutile_map", {"i": f"0:N:{W}"},
        schedule=ScheduleType.CuTile,
    )

    # Inner SDFG: empty first state, compute second state
    inner_sdfg = SDFG(f"{name}_inner")
    inner_sdfg.add_array(
        "_tile_a", (W,), dtype, storage=StorageType.CuTile_Tile,
    )
    inner_sdfg.add_array(
        "_tile_out", (W,), dtype, storage=StorageType.CuTile_Tile,
    )

    s1 = inner_sdfg.add_state("empty")
    s2 = inner_sdfg.add_state("compute")
    inner_sdfg.add_edge(s1, s2, dace.InterstateEdge())

    # state2: _tile_out = _tile_a + 1.0
    s2_in = s2.add_read("_tile_a")
    s2_out = s2.add_write("_tile_out")
    t = s2.add_tasklet(
        "add1", {"x"}, {"y"}, "y = x + 1.0", language=Language.Python,
    )
    s2.add_edge(s2_in, None, t, "x", Memlet(data="_tile_a", subset=f"0:{W}"))
    s2.add_edge(t, "y", s2_out, None, Memlet(data="_tile_out", subset=f"0:{W}"))

    nsdfg = state.add_nested_sdfg(inner_sdfg, {"_tile_a"}, {"_tile_out"})

    a_node = state.add_read("A")
    b_node = state.add_write("B")
    tile_a = state.add_access("_tile_a")
    tile_out = state.add_access("_tile_out")

    state.add_memlet_path(
        a_node, me, tile_a,
        memlet=Memlet(data="A", subset=f"i:i+{W}"),
    )
    state.add_edge(
        tile_a, None, nsdfg, "_tile_a",
        Memlet(data="_tile_a", subset=f"0:{W}"),
    )
    state.add_edge(
        nsdfg, "_tile_out", tile_out, None,
        Memlet(data="_tile_out", subset=f"0:{W}"),
    )
    state.add_memlet_path(
        tile_out, mx, b_node,
        memlet=Memlet(data="B", subset=f"i:i+{W}"),
    )
    return sdfg


def _build_interstate_assignment_sdfg(
    name: str, dtype=dace.float64, W: int = 8,
):
    """Build SDFG with an interstate assignment in the NestedSDFG.

    Inner SDFG:
        state1 (empty) --[scale = 3]--> state2: _tile_out = _tile_a * scale

    Expected: B = A * 3.

    :param name: SDFG name.
    :param dtype: Data type.
    :param W: Tile width.
    :returns: The constructed SDFG.
    """
    N = dace.symbol("N")

    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array("A", (N,), dtype, storage=StorageType.GPU_Global)
    sdfg.add_array("B", (N,), dtype, storage=StorageType.GPU_Global)
    sdfg.add_array(
        "_tile_a", (W,), dtype,
        storage=StorageType.CuTile_Tile, transient=True,
    )
    sdfg.add_array(
        "_tile_out", (W,), dtype,
        storage=StorageType.CuTile_Tile, transient=True,
    )

    state = sdfg.add_state("main")
    me, mx = state.add_map(
        "cutile_map", {"i": f"0:N:{W}"},
        schedule=ScheduleType.CuTile,
    )

    # Inner SDFG with interstate assignment
    inner_sdfg = SDFG(f"{name}_inner")
    inner_sdfg.add_symbol("scale", dace.int64)
    inner_sdfg.add_array(
        "_tile_a", (W,), dtype, storage=StorageType.CuTile_Tile,
    )
    inner_sdfg.add_array(
        "_tile_out", (W,), dtype, storage=StorageType.CuTile_Tile,
    )

    s1 = inner_sdfg.add_state("init")
    s2 = inner_sdfg.add_state("compute")
    inner_sdfg.add_edge(
        s1, s2, dace.InterstateEdge(assignments={"scale": "3"}),
    )

    s2_in = s2.add_read("_tile_a")
    s2_out = s2.add_write("_tile_out")
    t = s2.add_tasklet(
        "scale_mul", {"x"}, {"y"}, "y = x * scale",
        language=Language.Python,
    )
    s2.add_edge(
        s2_in, None, t, "x", Memlet(data="_tile_a", subset=f"0:{W}"),
    )
    s2.add_edge(
        t, "y", s2_out, None, Memlet(data="_tile_out", subset=f"0:{W}"),
    )

    nsdfg = state.add_nested_sdfg(
        inner_sdfg, {"_tile_a"}, {"_tile_out"},
        symbol_mapping={"scale": 0},
    )

    a_node = state.add_read("A")
    b_node = state.add_write("B")
    tile_a = state.add_access("_tile_a")
    tile_out = state.add_access("_tile_out")

    state.add_memlet_path(
        a_node, me, tile_a,
        memlet=Memlet(data="A", subset=f"i:i+{W}"),
    )
    state.add_edge(
        tile_a, None, nsdfg, "_tile_a",
        Memlet(data="_tile_a", subset=f"0:{W}"),
    )
    state.add_edge(
        nsdfg, "_tile_out", tile_out, None,
        Memlet(data="_tile_out", subset=f"0:{W}"),
    )
    state.add_memlet_path(
        tile_out, mx, b_node,
        memlet=Memlet(data="B", subset=f"i:i+{W}"),
    )
    return sdfg


def _build_conditional_nsdfg_sdfg(
    name: str, dtype=dace.float64, W: int = 8, cond_val: int = 1,
):
    """Build SDFG with a conditional-branching NestedSDFG.

    Inner SDFG:
        init (empty) --[cond_val > 0]--> branch_a: _tile_out = _tile_a * 2.0
        init (empty) --[not (cond_val > 0)]--> branch_b: _tile_out = _tile_a + 1.0
        branch_a --> merge (empty)
        branch_b --> merge (empty)

    The symbol ``cond_val`` is passed from the outer SDFG.

    :param name: SDFG name.
    :param dtype: Data type.
    :param W: Tile width.
    :param cond_val: Default value for the condition symbol mapping.
    :returns: The constructed SDFG.
    """
    N = dace.symbol("N")

    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_symbol("cond_val", dace.int32)
    sdfg.add_array("A", (N,), dtype, storage=StorageType.GPU_Global)
    sdfg.add_array("B", (N,), dtype, storage=StorageType.GPU_Global)
    sdfg.add_array(
        "_tile_a", (W,), dtype,
        storage=StorageType.CuTile_Tile, transient=True,
    )
    sdfg.add_array(
        "_tile_out", (W,), dtype,
        storage=StorageType.CuTile_Tile, transient=True,
    )

    state = sdfg.add_state("main")
    me, mx = state.add_map(
        "cutile_map", {"i": f"0:N:{W}"},
        schedule=ScheduleType.CuTile,
    )

    # Inner SDFG: conditional branching
    inner_sdfg = SDFG(f"{name}_inner")
    inner_sdfg.add_symbol("cond_val", dace.int32)
    inner_sdfg.add_array(
        "_tile_a", (W,), dtype, storage=StorageType.CuTile_Tile,
    )
    inner_sdfg.add_array(
        "_tile_out", (W,), dtype, storage=StorageType.CuTile_Tile,
    )

    s_init = inner_sdfg.add_state("init")
    s_branch_a = inner_sdfg.add_state("branch_a")
    s_branch_b = inner_sdfg.add_state("branch_b")
    s_merge = inner_sdfg.add_state("merge")

    # Conditional edges from init
    inner_sdfg.add_edge(
        s_init, s_branch_a,
        dace.InterstateEdge(condition="cond_val > 0"),
    )
    inner_sdfg.add_edge(
        s_init, s_branch_b,
        dace.InterstateEdge(condition="not (cond_val > 0)"),
    )
    # Unconditional edges to merge
    inner_sdfg.add_edge(s_branch_a, s_merge, dace.InterstateEdge())
    inner_sdfg.add_edge(s_branch_b, s_merge, dace.InterstateEdge())

    # branch_a: _tile_out = _tile_a * 2.0
    ba_in = s_branch_a.add_read("_tile_a")
    ba_out = s_branch_a.add_write("_tile_out")
    ta = s_branch_a.add_tasklet(
        "mul2", {"x"}, {"y"}, "y = x * 2.0", language=Language.Python,
    )
    s_branch_a.add_edge(
        ba_in, None, ta, "x",
        Memlet(data="_tile_a", subset=f"0:{W}"),
    )
    s_branch_a.add_edge(
        ta, "y", ba_out, None,
        Memlet(data="_tile_out", subset=f"0:{W}"),
    )

    # branch_b: _tile_out = _tile_a + 1.0
    bb_in = s_branch_b.add_read("_tile_a")
    bb_out = s_branch_b.add_write("_tile_out")
    tb = s_branch_b.add_tasklet(
        "add1", {"x"}, {"y"}, "y = x + 1.0", language=Language.Python,
    )
    s_branch_b.add_edge(
        bb_in, None, tb, "x",
        Memlet(data="_tile_a", subset=f"0:{W}"),
    )
    s_branch_b.add_edge(
        tb, "y", bb_out, None,
        Memlet(data="_tile_out", subset=f"0:{W}"),
    )

    nsdfg = state.add_nested_sdfg(
        inner_sdfg, {"_tile_a"}, {"_tile_out"},
        symbol_mapping={"cond_val": dace.symbol("cond_val")},
    )

    a_node = state.add_read("A")
    b_node = state.add_write("B")
    tile_a = state.add_access("_tile_a")
    tile_out = state.add_access("_tile_out")

    state.add_memlet_path(
        a_node, me, tile_a,
        memlet=Memlet(data="A", subset=f"i:i+{W}"),
    )
    state.add_edge(
        tile_a, None, nsdfg, "_tile_a",
        Memlet(data="_tile_a", subset=f"0:{W}"),
    )
    state.add_edge(
        nsdfg, "_tile_out", tile_out, None,
        Memlet(data="_tile_out", subset=f"0:{W}"),
    )
    state.add_memlet_path(
        tile_out, mx, b_node,
        memlet=Memlet(data="B", subset=f"i:i+{W}"),
    )
    return sdfg


class TestMultiStateNestedSDFGRuntime:
    """Multi-state NestedSDFGs inside CuTile scopes execute correctly."""

    def test_two_state_sequential(self):
        """B = A * 2 + 1 via two-state NestedSDFG, N=64."""
        sdfg = _build_two_state_nsdfg_sdfg("rt_ms_twost_64")
        n = 64
        rng = np.random.default_rng(200)
        A = rng.random(n)
        B = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, N=n)
        np.testing.assert_allclose(results["B"], A * 2 + 1, rtol=1e-14)

    def test_two_state_large(self):
        """B = A * 2 + 1 via two-state NestedSDFG, N=256."""
        sdfg = _build_two_state_nsdfg_sdfg("rt_ms_twost_256")
        n = 256
        rng = np.random.default_rng(201)
        A = rng.random(n)
        B = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, N=n)
        np.testing.assert_allclose(results["B"], A * 2 + 1, rtol=1e-14)

    def test_empty_first_state(self):
        """B = A + 1.0 via empty-first-state NestedSDFG, N=64."""
        sdfg = _build_empty_first_state_sdfg("rt_ms_empty_64")
        n = 64
        rng = np.random.default_rng(202)
        A = rng.random(n)
        B = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, N=n)
        np.testing.assert_allclose(results["B"], A + 1.0, rtol=1e-14)

    def test_three_state_chain(self):
        """B = (A + 1) * 2 - 3 via three-state chain, N=64."""
        sdfg = _build_three_state_chain_sdfg("rt_ms_chain3_64")
        n = 64
        rng = np.random.default_rng(203)
        A = rng.random(n)
        B = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, N=n)
        np.testing.assert_allclose(results["B"], (A + 1) * 2 - 3, rtol=1e-14)

    def test_three_state_chain_large(self):
        """B = (A + 1) * 2 - 3 via three-state chain, N=256."""
        sdfg = _build_three_state_chain_sdfg("rt_ms_chain3_256")
        n = 256
        rng = np.random.default_rng(204)
        A = rng.random(n)
        B = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, N=n)
        np.testing.assert_allclose(results["B"], (A + 1) * 2 - 3, rtol=1e-14)

    def test_interstate_assignment(self):
        """B = A * 3 via interstate assignment (scale=3), N=64."""
        sdfg = _build_interstate_assignment_sdfg("rt_ms_assign_64")
        n = 64
        rng = np.random.default_rng(205)
        A = rng.random(n)
        B = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, N=n)
        np.testing.assert_allclose(results["B"], A * 3, rtol=1e-14)

    def test_two_state_codegen_valid_python(self):
        """The generated code for a two-state NestedSDFG should be valid Python."""
        sdfg = _build_two_state_nsdfg_sdfg("rt_ms_codegen_twost")
        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        ast.parse(code)

    def test_three_state_codegen_valid_python(self):
        """The generated code for a three-state NestedSDFG should be valid Python."""
        sdfg = _build_three_state_chain_sdfg("rt_ms_codegen_chain3")
        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        ast.parse(code)

    def test_interstate_assignment_codegen_valid_python(self):
        """The generated code for interstate assignment should be valid Python."""
        sdfg = _build_interstate_assignment_sdfg("rt_ms_codegen_assign")
        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        ast.parse(code)

    def test_conditional_true_branch(self):
        """Conditional NestedSDFG with cond_val=1 takes the true branch (B = A * 2)."""
        sdfg = _build_conditional_nsdfg_sdfg("rt_ms_cond_true")
        n = 64
        rng = np.random.default_rng(206)
        A = rng.random(n)
        B = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, N=n, cond_val=1)
        np.testing.assert_allclose(results["B"], A * 2.0, rtol=1e-14)

    def test_conditional_false_branch(self):
        """Conditional NestedSDFG with cond_val=0 takes the false branch (B = A + 1)."""
        sdfg = _build_conditional_nsdfg_sdfg("rt_ms_cond_false")
        n = 64
        rng = np.random.default_rng(207)
        A = rng.random(n)
        B = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, N=n, cond_val=0)
        np.testing.assert_allclose(results["B"], A + 1.0, rtol=1e-14)

    def test_conditional_codegen_valid_python(self):
        """The generated code for a conditional NestedSDFG should be valid Python."""
        sdfg = _build_conditional_nsdfg_sdfg("rt_ms_codegen_cond")
        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        ast.parse(code)

    def test_conditional_codegen_has_if(self):
        """The generated code for a conditional NestedSDFG should contain an if."""
        sdfg = _build_conditional_nsdfg_sdfg("rt_ms_codegen_cond_if")
        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        assert "if " in code


# ============================================================
# 3. Symbolic sizes and edge cases
# ============================================================


class TestNestedSDFGSymbolicRuntime:
    """Symbolic sizes and edge cases for NestedSDFGs in CuTile scopes."""

    @pytest.mark.parametrize("n", [8, 64, 128])
    def test_symbolic_size(self, n):
        """Run with various symbolic N values (aligned to tile width)."""
        sdfg = _build_single_state_nsdfg_sdfg(
            f"rt_sym_{n}", "y = x * 2",
        )
        rng = np.random.default_rng(n + 300)
        A = rng.random(n)
        B = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, N=n)
        np.testing.assert_allclose(results["B"], A * 2, rtol=1e-14)

    @pytest.mark.parametrize("n", [8, 64, 128])
    def test_symbolic_size_two_state(self, n):
        """Two-state NestedSDFG with various symbolic N values."""
        sdfg = _build_two_state_nsdfg_sdfg(f"rt_sym_ms_{n}")
        rng = np.random.default_rng(n + 400)
        A = rng.random(n)
        B = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, N=n)
        np.testing.assert_allclose(results["B"], A * 2 + 1, rtol=1e-14)


# ============================================================
# 4. Data type tests
# ============================================================


class TestNestedSDFGDtypeRuntime:
    """Test NestedSDFGs with different data types."""

    def test_float32(self):
        """B[i] = A[i] * 2 in float32, N=64."""
        sdfg = _build_single_state_nsdfg_sdfg(
            "rt_dt_f32", "y = x * 2", dtype=dace.float32,
        )
        n = 64
        rng = np.random.default_rng(500)
        A = rng.random(n).astype(np.float32)
        B = np.zeros(n, dtype=np.float32)
        results = _run_cutile(sdfg, A=A, B=B, N=n)
        np.testing.assert_allclose(results["B"], A * 2, rtol=1e-6)

    def test_float64(self):
        """B[i] = A[i] * 2 in float64 (explicit), N=64."""
        sdfg = _build_single_state_nsdfg_sdfg(
            "rt_dt_f64", "y = x * 2", dtype=dace.float64,
        )
        n = 64
        rng = np.random.default_rng(501)
        A = rng.random(n)
        B = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, N=n)
        np.testing.assert_allclose(results["B"], A * 2, rtol=1e-14)

    def test_int32(self):
        """B[i] = A[i] * 2 in int32, N=64."""
        sdfg = _build_single_state_nsdfg_sdfg(
            "rt_dt_i32", "y = x * 2", dtype=dace.int32,
        )
        n = 64
        rng = np.random.default_rng(502)
        A = rng.integers(0, 500, n, dtype=np.int32)
        B = np.zeros(n, dtype=np.int32)
        results = _run_cutile(sdfg, A=A, B=B, N=n)
        np.testing.assert_array_equal(results["B"], A * 2)

    def test_float32_two_state(self):
        """B = A * 2 + 1 in float32 via two-state NestedSDFG, N=64."""
        sdfg = _build_two_state_nsdfg_sdfg(
            "rt_dt_f32_ms", dtype=dace.float32,
        )
        n = 64
        rng = np.random.default_rng(503)
        A = rng.random(n).astype(np.float32)
        B = np.zeros(n, dtype=np.float32)
        results = _run_cutile(sdfg, A=A, B=B, N=n)
        np.testing.assert_allclose(
            results["B"], (A * 2 + 1).astype(np.float32), rtol=1e-6,
        )

    def test_int32_two_input(self):
        """C[i] = A[i] + B[i] in int32, N=64."""
        sdfg = _build_two_input_nsdfg_sdfg(
            "rt_dt_i32_2in", "y = x + z", dtype=dace.int32,
        )
        n = 64
        rng = np.random.default_rng(504)
        A = rng.integers(0, 500, n, dtype=np.int32)
        B = rng.integers(0, 500, n, dtype=np.int32)
        C = np.zeros(n, dtype=np.int32)
        results = _run_cutile(sdfg, A=A, B=B, C=C, N=n)
        np.testing.assert_array_equal(results["C"], A + B)


# ============================================================
# 5. Tile width tests
# ============================================================


class TestNestedSDFGTileWidths:
    """Test different power-of-2 tile widths for NestedSDFGs."""

    @pytest.mark.parametrize("w", [4, 8, 16, 32])
    def test_single_state_tile_widths(self, w):
        """B[i] = A[i] * 2 with various tile widths."""
        sdfg = _build_single_state_nsdfg_sdfg(
            f"rt_tw_ss_w{w}", "y = x * 2", W=w,
        )
        n = 128  # Aligned for all widths up to 32
        rng = np.random.default_rng(w + 600)
        A = rng.random(n)
        B = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, N=n)
        np.testing.assert_allclose(results["B"], A * 2, rtol=1e-14)

    @pytest.mark.parametrize("w", [4, 8, 16, 32])
    def test_two_state_tile_widths(self, w):
        """B = A * 2 + 1 with various tile widths, N=128."""
        sdfg = _build_two_state_nsdfg_sdfg(
            f"rt_tw_ms_w{w}", W=w,
        )
        n = 128
        rng = np.random.default_rng(w + 700)
        A = rng.random(n)
        B = np.zeros(n)
        results = _run_cutile(sdfg, A=A, B=B, N=n)
        np.testing.assert_allclose(results["B"], A * 2 + 1, rtol=1e-14)


# ============================================================
# 6. Codegen structure tests (no GPU needed at runtime,
#    but module-level pytestmark applies gpu)
# ============================================================


class TestNestedSDFGCodegenStructure:
    """Verify generated code structure for NestedSDFG variants."""

    def test_single_state_has_nested_function(self):
        """Single-state NestedSDFG generates a module-level function."""
        sdfg = _build_single_state_nsdfg_sdfg(
            "cg_ss_func", "y = x * 2",
        )
        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        assert "def __dace_nested_" in code

    def test_single_state_function_has_return(self):
        """The generated function should have a return statement."""
        sdfg = _build_single_state_nsdfg_sdfg(
            "cg_ss_return", "y = x * 2",
        )
        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        assert "return _tile_out" in code

    def test_two_state_has_sequential_ops(self):
        """Two-state NestedSDFG has both operations in the generated function."""
        sdfg = _build_two_state_nsdfg_sdfg("cg_ms_twost")
        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        # Both tasklet bodies should be present in the function.
        assert "* 2" in code
        assert "+ 1" in code

    def test_interstate_assignment_has_scale(self):
        """Interstate assignment generates ``scale = 3`` in function body."""
        sdfg = _build_interstate_assignment_sdfg("cg_ms_assign")
        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        assert "scale = 3" in code

    def test_ct_load_and_store_present(self):
        """The generated kernel should have ct.load and ct.store."""
        sdfg = _build_single_state_nsdfg_sdfg(
            "cg_ct_loadstore", "y = x * 2",
        )
        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        assert "ct.load(" in code
        assert "ct.store(" in code

    def test_ct_launch_present(self):
        """The generated code should have ct.launch."""
        sdfg = _build_single_state_nsdfg_sdfg(
            "cg_ct_launch", "y = x * 2",
        )
        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        assert "ct.launch(" in code

    def test_kernel_decorator_present(self):
        """The generated code should have @ct.kernel."""
        sdfg = _build_single_state_nsdfg_sdfg(
            "cg_ct_kernel", "y = x * 2",
        )
        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        assert "@ct.kernel" in code

    def test_two_input_nsdfg_has_both_params(self):
        """Two-input NestedSDFG function should have both parameters."""
        sdfg = _build_two_input_nsdfg_sdfg(
            "cg_2in_params", "y = x + z",
        )
        code_objects = sdfg.generate_code()
        code = next(co for co in code_objects if co.name == sdfg.name).code
        # Find the function definition.
        func_start = code.index("def __dace_nested_")
        func_line = code[func_start:code.index(":", func_start) + 1]
        assert "_tile_a" in func_line
        assert "_tile_b" in func_line


# ============================================================
# Entry point
# ============================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=300"])
