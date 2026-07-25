# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import sys
import types

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.codegen.instrumentation.provider import InstrumentationProvider
from dace.sdfg import SDFG

_SYNC_CALL = 'cupy.cuda.get_current_stream().synchronize()'


class _CountingStream:

    def __init__(self) -> None:
        self.synchronization_count = 0

    def synchronize(self) -> None:
        self.synchronization_count += 1


class _ScopeMutationInstrumentation(InstrumentationProvider):
    """Synthetic provider that mutates begin-hook locals in the end hook."""

    def on_sdfg_begin(self, sdfg, local_stream, global_stream, codegen) -> None:
        global_stream.write("__scope_results = []")
        local_stream.write("__scope_token = 40")
        local_stream.write("__nested_token = 7")
        local_stream.write("global __begin_global")
        local_stream.write("__begin_global = 3")
        local_stream.write("__end_global = 5")

    def on_sdfg_end(self, sdfg, local_stream, global_stream) -> None:
        local_stream.write("__scope_token += 1")
        local_stream.write("__scope_token = __scope_token + 1")
        local_stream.write("def __ignored_scope():")
        with local_stream.indented():
            local_stream.write("__nested_token = 99")
        local_stream.write("__ignored_values = [__nested_token for __nested_token in range(1)]")
        local_stream.write("__begin_global += 1")
        local_stream.write("global __end_global")
        local_stream.write("__end_global += 2")
        local_stream.write("__scope_results.append((__scope_token, __nested_token, __begin_global, __end_global))")


def _execute_generated(source: str, function_name: str, monkeypatch: pytest.MonkeyPatch, **kwargs) -> _CountingStream:
    """Execute generated frame code with a minimal fake CuPy module."""
    stream = _CountingStream()
    cupy_module = types.ModuleType("cupy")
    cupy_module.cuda = types.SimpleNamespace(get_current_stream=lambda: stream)
    monkeypatch.setitem(sys.modules, "cupy", cupy_module)
    monkeypatch.setitem(sys.modules, "sympy_function_redefinitions", types.ModuleType("sympy_function_redefinitions"))
    namespace: dict[str, object] = {}
    exec(source, namespace)
    namespace[function_name](**kwargs)
    return stream


def _build_unused_gpu_argument(name: str) -> SDFG:
    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array("A", [1], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_state("empty", is_start_block=True)
    return sdfg


def _build_gpu_return(name: str) -> SDFG:
    sdfg = _build_unused_gpu_argument(name)
    state = sdfg.start_block
    return_block = sdfg.add_return("successful_return")
    sdfg.add_edge(state, return_block, dace.InterstateEdge())
    return sdfg


def _build_gpu_exception(name: str) -> SDFG:
    sdfg = _build_unused_gpu_argument(name)
    sdfg.start_block.add_tasklet("fail", {}, {}, 'raise RuntimeError("expected failure")')
    return sdfg


def _build_cpu_return(name: str, tasklet_literal: bool = False) -> SDFG:
    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    state = sdfg.add_state("empty", is_start_block=True)
    if tasklet_literal:
        routed_exit = f"return __dace_successful_exit_{sdfg.cfg_id}()"
        state.add_tasklet("literal", {}, {}, f"marker = {routed_exit!r}")
    return_block = sdfg.add_return("successful_return")
    sdfg.add_edge(state, return_block, dace.InterstateEdge())
    return sdfg


def _build_nested_region_return(name: str) -> SDFG:
    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array("A", [1], dace.float64, storage=dtypes.StorageType.GPU_Global)
    region = dace.sdfg.state.ControlFlowRegion("nested_region", sdfg=sdfg)
    sdfg.add_node(region, is_start_block=True)
    state = region.add_state("empty", is_start_block=True)
    return_block = region.add_return("successful_return")
    region.add_edge(state, return_block, dace.InterstateEdge())
    return sdfg


def _build_cutile_program(name: str) -> SDFG:
    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array("A", [32], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("B", [32], dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("_tile_A", [32], dace.float64, storage=dtypes.StorageType.CuTile_Tile, transient=True)
    sdfg.add_array("_tile_B", [32], dace.float64, storage=dtypes.StorageType.CuTile_Tile, transient=True)

    state = sdfg.add_state("main", is_start_block=True)
    entry, exit_node = state.add_map("cutile_map", {"tile_i": "0:32:32"}, schedule=dtypes.ScheduleType.CuTile)
    tile_a = state.add_access("_tile_A")
    tile_b = state.add_access("_tile_B")
    tasklet = state.add_tasklet("compute", {"inp"}, {"out"}, "out = inp * 2.0")
    state.add_memlet_path(state.add_read("A"), entry, tile_a, memlet=dace.Memlet("A[0:32]"))
    state.add_edge(tile_a, None, tasklet, "inp", dace.Memlet("_tile_A[0:32]"))
    state.add_edge(tasklet, "out", tile_b, None, dace.Memlet("_tile_B[0:32]"))
    state.add_memlet_path(tile_b, exit_node, state.add_write("B"), memlet=dace.Memlet("B[0:32]"))
    return sdfg


def _build_copy_chain(name: str, storage: dtypes.StorageType) -> SDFG:
    """Build a two-state copy chain with one transient array."""
    n = dace.symbol('N')
    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array('A', [n], dace.float64, storage=storage)
    sdfg.add_array('B', [n], dace.float64, storage=storage)
    sdfg.add_transient('tmp', [n], dace.float64, storage=storage)

    first = sdfg.add_state('first', is_start_block=True)
    first.add_edge(first.add_read('A'), None, first.add_write('tmp'), None, dace.Memlet('A[0:N]'))
    second = sdfg.add_state('second')
    second.add_edge(second.add_read('tmp'), None, second.add_write('B'), None, dace.Memlet('tmp[0:N]'))
    sdfg.add_edge(first, second, dace.InterstateEdge())
    return sdfg


def _build_nested_gpu_copy(name: str) -> SDFG:
    """Build a top-level SDFG whose nested helper uses GPU arrays."""
    storage = dtypes.StorageType.GPU_Global
    inner = SDFG(f'{name}_inner')
    inner.backend = dtypes.BackendLanguage.Python
    inner.add_array('X', [8], dace.float64, storage=storage)
    inner.add_array('Y', [8], dace.float64, storage=storage)
    inner_state = inner.add_state('copy', is_start_block=True)
    inner_state.add_edge(inner_state.add_read('X'), None, inner_state.add_write('Y'), None, dace.Memlet('X[0:8]'))
    inner_return = inner.add_return('successful_return')
    inner.add_edge(inner_state, inner_return, dace.InterstateEdge())

    outer = SDFG(name)
    outer.backend = dtypes.BackendLanguage.Python
    outer.add_array('A', [8], dace.float64, storage=storage)
    outer.add_array('B', [8], dace.float64, storage=storage)
    outer_state = outer.add_state('nested', is_start_block=True)
    nested = outer_state.add_nested_sdfg(inner, {'X'}, {'Y'})
    outer_state.add_edge(outer_state.add_read('A'), None, nested, 'X', dace.Memlet('A[0:8]'))
    outer_state.add_edge(nested, 'Y', outer_state.add_write('B'), None, dace.Memlet('B[0:8]'))
    return outer


def _generated_source(sdfg: SDFG) -> str:
    """Return the generated Python frame source."""
    return sdfg.generate_code()[0].code


def _function_nodes(source: str) -> dict[str, ast.FunctionDef]:
    module = ast.parse(source)
    return {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}


def _sync_count(function: ast.FunctionDef) -> int:
    return sum(ast.unparse(node) == _SYNC_CALL for node in ast.walk(function) if isinstance(node, ast.Call))


def test_cpu_program_emits_no_boundary_sync() -> None:
    sdfg = _build_copy_chain('cpu_boundary_no_sync', dtypes.StorageType.CPU_Heap)
    source = _generated_source(sdfg)

    assert _SYNC_CALL not in source
    assert _sync_count(_function_nodes(source)[sdfg.name]) == 0
    assert "__dace_successful_exit_" not in ast.unparse(_function_nodes(source)[sdfg.name])


def test_gpu_copy_chain_emits_one_success_epilogue_sync() -> None:
    sdfg = _build_copy_chain('gpu_boundary_one_sync', dtypes.StorageType.GPU_Global)
    source = _generated_source(sdfg)
    top_level = _function_nodes(source)[sdfg.name]

    assert source.count(_SYNC_CALL) == 1
    assert _sync_count(top_level) == 1
    assert "__dace_successful_exit_" not in ast.unparse(top_level)
    assert not any(isinstance(node, ast.Try) for node in ast.walk(top_level))
    assert 'try:' not in ast.unparse(top_level)
    assert 'finally:' not in ast.unparse(top_level)


def test_nested_gpu_helper_has_no_boundary_sync() -> None:
    sdfg = _build_nested_gpu_copy('nested_gpu_boundary_sync')
    source = _generated_source(sdfg)
    functions = _function_nodes(source)
    helper_functions = [
        function for name, function in functions.items() if name != sdfg.name and not name.startswith('__dace_')
    ]

    assert len(helper_functions) == 1
    assert source.count(_SYNC_CALL) == 1
    assert _sync_count(functions[sdfg.name]) == 1
    assert _sync_count(helper_functions[0]) == 0
    assert any(
        isinstance(node, ast.FunctionDef) and node.name.startswith("__dace_successful_exit_")
        for node in ast.walk(helper_functions[0]))
    assert any(
        isinstance(node, ast.Return) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
        and node.value.func.id.startswith("__dace_successful_exit_") for node in ast.walk(helper_functions[0]))


def test_boundary_sync_precedes_sdfg_timer_end() -> None:
    sdfg = _build_copy_chain('gpu_boundary_timer_order', dtypes.StorageType.GPU_Global)
    sdfg.instrument = dtypes.InstrumentationType.PythonTimer
    source = _generated_source(sdfg)
    top_level_source = ast.unparse(_function_nodes(source)[sdfg.name])

    assert top_level_source.index(_SYNC_CALL) < top_level_source.index('__dace_tend_')
    assert not any(isinstance(node, ast.Try) for node in ast.walk(_function_nodes(source)[sdfg.name]))


def test_successful_return_routes_through_sync_before_timer_end(monkeypatch: pytest.MonkeyPatch) -> None:
    sdfg = _build_gpu_return("gpu_boundary_return")
    sdfg.instrument = dtypes.InstrumentationType.PythonTimer
    with dace.config.set_temporary("instrumentation", "report_each_invocation", value=False):
        source = _generated_source(sdfg)
    top_level = _function_nodes(source)[sdfg.name]
    top_level_source = ast.unparse(top_level)
    exit_calls = [
        node for node in ast.walk(top_level) if isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name) and node.value.func.id.startswith("__dace_successful_exit_")
    ]

    assert len(exit_calls) == 2
    assert any(
        isinstance(node, ast.FunctionDef) and node.name.startswith("__dace_successful_exit_")
        for node in ast.walk(top_level))
    assert _sync_count(top_level) == 1
    assert top_level_source.index(_SYNC_CALL) < top_level_source.index("__dace_tend_")
    assert not any(isinstance(node, ast.Try) for node in ast.walk(top_level))

    stream = _execute_generated(source, sdfg.name, monkeypatch, A=np.empty(1))
    assert stream.synchronization_count == 1


def test_exception_skips_boundary_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    sdfg = _build_gpu_exception("gpu_boundary_exception")
    source = _generated_source(sdfg)
    stream = _CountingStream()
    cupy_module = types.ModuleType("cupy")
    cupy_module.cuda = types.SimpleNamespace(get_current_stream=lambda: stream)
    monkeypatch.setitem(sys.modules, "cupy", cupy_module)
    monkeypatch.setitem(sys.modules, "sympy_function_redefinitions", types.ModuleType("sympy_function_redefinitions"))
    namespace: dict[str, object] = {}
    exec(source, namespace)

    with pytest.raises(RuntimeError, match="expected failure"):
        namespace[sdfg.name](A=np.empty(1))

    assert stream.synchronization_count == 0


def test_unused_gpu_argument_imports_cupy_and_synchronizes(monkeypatch: pytest.MonkeyPatch) -> None:
    sdfg = _build_unused_gpu_argument("unused_gpu_argument")
    source = _generated_source(sdfg)

    assert "import cupy" in source
    stream = _execute_generated(source, sdfg.name, monkeypatch, A=np.empty(1))
    assert stream.synchronization_count == 1


def test_cutile_keeps_kernel_and_boundary_synchronization() -> None:
    sdfg = _build_cutile_program("cutile_boundary_sync")
    source = _generated_source(sdfg)
    top_level = _function_nodes(source)[sdfg.name]
    top_level_source = ast.unparse(top_level)

    assert "ct.launch(" in source
    assert _sync_count(top_level) == 2
    assert top_level_source.rindex(_SYNC_CALL) > top_level_source.index("ct.launch(")


def test_empty_cpu_return_uses_empty_success_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    sdfg = _build_cpu_return("cpu_empty_success_helper")
    source = _generated_source(sdfg)
    top_level = _function_nodes(source)[sdfg.name]
    epilogue = next(node for node in top_level.body
                    if isinstance(node, ast.FunctionDef) and node.name.startswith("__dace_successful_exit_"))

    assert len(epilogue.body) == 1
    assert isinstance(epilogue.body[0], ast.Pass)
    assert any(
        isinstance(node, ast.Return) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
        and node.value.func.id == epilogue.name for node in ast.walk(top_level))
    _execute_generated(source, sdfg.name, monkeypatch)


def test_return_routing_preserves_tasklet_literal(monkeypatch: pytest.MonkeyPatch) -> None:
    sdfg = _build_cpu_return("cpu_return_literal", tasklet_literal=True)
    routed_exit = f"return __dace_successful_exit_{sdfg.cfg_id}()"
    source = _generated_source(sdfg)
    top_level = _function_nodes(source)[sdfg.name]

    routed_calls = sum(
        isinstance(node, ast.Return) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
        and node.value.func.id == f"__dace_successful_exit_{sdfg.cfg_id}" for node in ast.walk(top_level))
    assert any(isinstance(node, ast.Constant) and node.value == routed_exit for node in ast.walk(top_level))
    assert source.count(routed_exit) == routed_calls + 1
    _execute_generated(source, sdfg.name, monkeypatch)


def test_nested_region_return_routes_through_boundary_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    sdfg = _build_nested_region_return("nested_region_boundary_return")
    source = _generated_source(sdfg)
    top_level = _function_nodes(source)[sdfg.name]

    assert _sync_count(top_level) == 1
    assert any(
        isinstance(node, ast.FunctionDef) and node.name.startswith("__dace_successful_exit_")
        for node in ast.walk(top_level))
    stream = _execute_generated(source, sdfg.name, monkeypatch, A=np.empty(1))
    assert stream.synchronization_count == 1


def test_instrumentation_end_mutates_begin_scope_on_return(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_mapping = InstrumentationProvider.get_provider_mapping().copy()
    provider_mapping[dtypes.InstrumentationType.PythonTimer] = _ScopeMutationInstrumentation
    monkeypatch.setattr(InstrumentationProvider, "get_provider_mapping", staticmethod(lambda: provider_mapping))

    sdfg = _build_cpu_return("instrumentation_scope_return")
    sdfg.instrument = dtypes.InstrumentationType.PythonTimer
    source = _generated_source(sdfg)
    top_level = _function_nodes(source)[sdfg.name]
    epilogue = next(node for node in top_level.body
                    if isinstance(node, ast.FunctionDef) and node.name.startswith("__dace_successful_exit_"))
    nonlocal_names = {name for node in epilogue.body if isinstance(node, ast.Nonlocal) for name in node.names}
    helper_globals = [name for node in epilogue.body if isinstance(node, ast.Global) for name in node.names]
    function_globals = [name for node in top_level.body if isinstance(node, ast.Global) for name in node.names]

    assert nonlocal_names == {"__scope_token"}
    assert "__nested_token" not in nonlocal_names
    assert "__begin_global" not in nonlocal_names
    assert "__end_global" not in nonlocal_names
    assert helper_globals.count("__begin_global") == 1
    assert helper_globals.count("__end_global") == 1
    assert function_globals.count("__begin_global") == 1
    assert function_globals.count("__end_global") == 1
    assert isinstance(top_level.body[0], ast.Global)
    assert top_level.body[0].names == ["__end_global"]

    monkeypatch.setitem(sys.modules, "sympy_function_redefinitions", types.ModuleType("sympy_function_redefinitions"))
    namespace: dict[str, object] = {}
    exec(source, namespace)
    namespace[sdfg.name]()

    assert namespace["__scope_results"] == [(42, 7, 4, 7)]


@pytest.mark.gpu
def test_gpu_copy_chain_runs_with_boundary_sync() -> None:
    cupy = pytest.importorskip('cupy')
    sdfg = _build_copy_chain('gpu_boundary_runtime', dtypes.StorageType.GPU_Global)
    compiled = sdfg.compile()
    n = 17
    source = compiled.code
    a = cupy.arange(n, dtype=cupy.float64)
    b = cupy.zeros(n, dtype=cupy.float64)

    compiled(A=a, B=b, N=n)

    assert source.count(_SYNC_CALL) == 1
    np.testing.assert_array_equal(cupy.asnumpy(b), np.arange(n, dtype=np.float64))
