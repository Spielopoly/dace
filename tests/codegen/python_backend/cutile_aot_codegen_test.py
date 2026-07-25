# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Generated-source tests for cuTile JIT/AOT mode selection and typing."""
import numpy as np
import pytest

import dace
from dace import dtypes
from dace.codegen import codegen
from dace.codegen.exceptions import CodegenError
from dace.dtypes import Language, ScheduleType, StorageType
from dace.memlet import Memlet


def _runtime_symbol_sdfg(name: str, expression: str) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array('x', [64], dace.int64, storage=StorageType.GPU_Global)
    sdfg.add_array('y', [64], dace.int64, storage=StorageType.GPU_Global)
    sdfg.add_array('_tx', [32], dace.int64, storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array('_ty', [32], dace.int64, storage=StorageType.CuTile_Tile, transient=True)
    init = sdfg.add_state('init')
    state = sdfg.add_state('main')
    sdfg.add_edge(init, state, dace.InterstateEdge(assignments={'runtime_value': expression}))
    me, mx = state.add_map('cutile_map', {'ti': '0:64:32'}, schedule=ScheduleType.CuTile)
    tasklet = state.add_tasklet('add', {'inp'}, {'out'}, 'out = inp + runtime_value', language=Language.Python)
    tx, ty = state.add_access('_tx'), state.add_access('_ty')
    state.add_memlet_path(state.add_read('x'), me, tx, memlet=Memlet('x[0:64]'))
    state.add_edge(tx, None, tasklet, 'inp', Memlet('_tx[0:32]'))
    state.add_edge(tasklet, 'out', ty, None, Memlet('_ty[0:32]'))
    state.add_memlet_path(ty, mx, state.add_write('y'), memlet=Memlet('y[0:64]'))
    sdfg.fill_scope_connectors()
    return sdfg


def _simple_sdfg(name: str) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array('x', [64], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array('y', [64], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array('_tx', [32], dace.float64, storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array('_ty', [32], dace.float64, storage=StorageType.CuTile_Tile, transient=True)
    state = sdfg.add_state()
    me, mx = state.add_map('cutile_map', {'ti': '0:64:32'}, schedule=ScheduleType.CuTile)
    tasklet = state.add_tasklet('scale', {'inp'}, {'out'}, 'out = inp * 2.0', language=Language.Python)
    tx, ty = state.add_access('_tx'), state.add_access('_ty')
    state.add_memlet_path(state.add_read('x'), me, tx, memlet=Memlet('x[0:64]'))
    state.add_edge(tx, None, tasklet, 'inp', Memlet('_tx[0:32]'))
    state.add_edge(tasklet, 'out', ty, None, Memlet('_ty[0:32]'))
    state.add_memlet_path(ty, mx, state.add_write('y'), memlet=Memlet('y[0:64]'))
    sdfg.fill_scope_connectors()
    return sdfg


def _add_runtime_kernel(sdfg: dace.SDFG, state: dace.SDFGState, suffix: str) -> None:
    """Add one cuTile kernel that consumes ``runtime_value``."""
    x_name, y_name = f'x_{suffix}', f'y_{suffix}'
    tx_name, ty_name = f'_tx_{suffix}', f'_ty_{suffix}'
    sdfg.add_array(x_name, [64], dace.int64, storage=StorageType.GPU_Global)
    sdfg.add_array(y_name, [64], dace.int64, storage=StorageType.GPU_Global)
    sdfg.add_array(tx_name, [32], dace.int64, storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array(ty_name, [32], dace.int64, storage=StorageType.CuTile_Tile, transient=True)
    me, mx = state.add_map(f'cutile_{suffix}', {f'ti_{suffix}': '0:64:32'}, schedule=ScheduleType.CuTile)
    tasklet = state.add_tasklet(f'add_{suffix}', {'inp'}, {'out'},
                                'out = inp + runtime_value',
                                language=Language.Python)
    tx, ty = state.add_access(tx_name), state.add_access(ty_name)
    state.add_memlet_path(state.add_read(x_name), me, tx, memlet=Memlet(f'{x_name}[0:64]'))
    state.add_edge(tx, None, tasklet, 'inp', Memlet(f'{tx_name}[0:32]'))
    state.add_edge(tasklet, 'out', ty, None, Memlet(f'{ty_name}[0:32]'))
    state.add_memlet_path(ty, mx, state.add_write(y_name), memlet=Memlet(f'{y_name}[0:64]'))
    state.fill_scope_connectors()


def _nested_helper_sdfg(name: str) -> dace.SDFG:
    """Build a cuTile scope that calls a generated nested helper."""
    sdfg = dace.SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array('x', [32], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array('y', [32], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array('_tx', [32], dace.float64, storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array('_ty', [32], dace.float64, storage=StorageType.CuTile_Tile, transient=True)
    state = sdfg.add_state('main')
    me, mx = state.add_map('cutile_map', {'ti': '0:32:32'}, schedule=ScheduleType.CuTile)

    inner = dace.SDFG('aot_nested_scale')
    inner.add_array('inp', [32], dace.float64, storage=StorageType.CuTile_Tile)
    inner.add_array('out', [32], dace.float64, storage=StorageType.CuTile_Tile)
    inner_state = inner.add_state('compute')
    tasklet = inner_state.add_tasklet('scale', {'value'}, {'result'}, 'result = value * 2.0', language=Language.Python)
    inner_state.add_edge(inner_state.add_read('inp'), None, tasklet, 'value', Memlet('inp[0:32]'))
    inner_state.add_edge(tasklet, 'result', inner_state.add_write('out'), None, Memlet('out[0:32]'))

    nested = state.add_nested_sdfg(inner, {'inp'}, {'out'})
    tx, ty = state.add_access('_tx'), state.add_access('_ty')
    state.add_memlet_path(state.add_read('x'), me, tx, memlet=Memlet('x[0:32]'))
    state.add_edge(tx, None, nested, 'inp', Memlet('_tx[0:32]'))
    state.add_edge(nested, 'out', ty, None, Memlet('_ty[0:32]'))
    state.add_memlet_path(ty, mx, state.add_write('y'), memlet=Memlet('y[0:32]'))
    state.fill_scope_connectors()
    return sdfg


def test_default_mode_is_jit():
    from dace.config import Config
    assert Config.get('compiler', 'cutile', 'mode') == 'jit'


def test_jit_frame_keeps_runtime_kernel(monkeypatch):
    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    objects = codegen.generate_code(_simple_sdfg('cutile_jit_source'))
    frame = objects[0].code
    assert '@ct.kernel' in frame
    assert 'ct.launch(' in frame
    assert not any(obj.name.startswith('__dace_cutile_aot_') for obj in objects[1:])


@pytest.mark.gpu
def test_aot_frame_uses_generated_launcher(monkeypatch):
    monkeypatch.setenv('DACE_compiler_cutile_mode', 'aot')
    objects = codegen.generate_code(_simple_sdfg('cutile_aot_source'))
    frame = objects[0].code
    auxiliary = next(obj for obj in objects[1:] if obj.name.startswith('__dace_cutile_aot_'))
    assert '@ct.kernel' not in frame
    assert 'ct.launch(' not in frame
    assert '._compile' not in frame
    assert '__dace_cutile_aot_runtime.launch(' in frame
    assert 'Module()' in auxiliary.code and '.load(' in auxiliary.code
    assert 'ct.launch(' not in auxiliary.code and '._compile' not in auxiliary.code


def test_aot_nested_helpers_are_export_only(monkeypatch):
    from dace.codegen.py import cutile_aot

    captured = []

    def fake_generate_aot_module(kernels, module_name):
        captured.extend(kernels)
        return '_KERNELS = {}\ndef launch(kernel_name, grid, args): pass\n'

    monkeypatch.setenv('DACE_compiler_cutile_mode', 'aot')
    monkeypatch.setattr(cutile_aot, 'generate_aot_module', fake_generate_aot_module)
    objects = codegen.generate_code(_nested_helper_sdfg('cutile_aot_nested_source'))
    frame = objects[0].code

    assert 'def __dace_nested_' not in frame
    assert len(captured) == 1
    assert 'def __dace_nested_' in captured[0]['source']
    assert captured[0]['abi_version'] == cutile_aot.AOT_ABI_VERSION
    assert all(isinstance(param, cutile_aot.AOTParam) for param in captured[0]['params'])


def test_runtime_unsigned_type_is_preserved(monkeypatch):
    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    code = _runtime_symbol_sdfg('cutile_runtime_uint', str(2**63 + 7)).generate_code()[0].code
    compact = code.replace(' ', '').replace('\n', '')
    assert 'cupy.asarray(runtime_value,dtype=numpy.uint64).reshape(1)' in compact
    assert 'dtype=numpy.int64' not in compact


def test_runtime_true_division_is_local_python_typing(monkeypatch):
    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = _runtime_symbol_sdfg('cutile_runtime_div', 'N / 2')
    sdfg.add_symbol('N', dace.int64)
    code = sdfg.generate_code()[0].code.replace(' ', '').replace('\n', '')
    assert 'cupy.asarray(runtime_value,dtype=numpy.float64).reshape(1)' in code


def test_python_assignment_type_respects_final_expression():
    from dace.codegen.py.cutile_target import _python_assignment_type

    symbols = {'N': dace.int64}
    assert _python_assignment_type('N / 2', symbols, {}) == dace.float64
    assert _python_assignment_type('N / 2 > 0', symbols, {}) == dace.bool


@pytest.mark.parametrize('expression, expected', [
    (str(-(2**63)), dace.int64),
    (str(2**63 - 1), dace.int64),
    (str(2**63), dace.uint64),
    (str(2**64 - 1), dace.uint64),
])
def test_python_assignment_type_static_integer_bounds(expression, expected):
    from dace.codegen.py.cutile_target import _python_assignment_type

    assert _python_assignment_type(expression, {}, {}) == expected


@pytest.mark.parametrize('expression', [str(2**64), str(-(2**63) - 1)])
def test_python_assignment_type_rejects_out_of_range_integer(expression):
    from dace.codegen.py.cutile_target import _python_assignment_type

    with pytest.raises(CodegenError, match='outside the representable 64-bit range'):
        _python_assignment_type(expression, {}, {})


def test_python_assignment_type_preserves_declared_uint64():
    from dace.codegen.py.cutile_target import _python_assignment_type

    assert _python_assignment_type('U', {'U': dace.uint64}, {}) == dace.uint64


def test_ordered_states_keep_distinct_runtime_types(monkeypatch):
    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = dace.SDFG('cutile_ordered_runtime_types')
    sdfg.backend = dtypes.BackendLanguage.Python
    init = sdfg.add_state('init')
    first = sdfg.add_state('first')
    second = sdfg.add_state('second')
    _add_runtime_kernel(sdfg, first, 'first')
    _add_runtime_kernel(sdfg, second, 'second')
    sdfg.add_edge(init, first, dace.InterstateEdge(assignments={'runtime_value': '7'}))
    sdfg.add_edge(first, second, dace.InterstateEdge(assignments={'runtime_value': '1.5'}))

    code = ''.join(sdfg.generate_code()[0].code.split())
    assert code.count('cupy.asarray(runtime_value,dtype=numpy.int64).reshape(1)') == 1
    assert code.count('cupy.asarray(runtime_value,dtype=numpy.float64).reshape(1)') == 1


def test_loop_region_runtime_assignment_uses_induction_type(monkeypatch):
    from dace.sdfg.state import LoopRegion

    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = dace.SDFG('cutile_loop_runtime_type')
    sdfg.backend = dtypes.BackendLanguage.Python
    loop = LoopRegion('loop',
                      condition_expr='i < 3',
                      loop_var='i',
                      initialize_expr='i = 0',
                      update_expr='i = i + 1',
                      sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    assign = loop.add_state('assign', is_start_block=True)
    kernel = loop.add_state('kernel')
    _add_runtime_kernel(sdfg, kernel, 'loop')
    loop.add_edge(assign, kernel, dace.InterstateEdge(assignments={'runtime_value': 'i + 1'}))

    code = ''.join(sdfg.generate_code()[0].code.split())
    assert 'cupy.asarray(runtime_value,dtype=numpy.int64).reshape(1)' in code
    assert 'cupy.asarray(runtime_value,dtype=numpy.uint64).reshape(1)' not in code


def test_loop_induction_range_rejects_later_iteration_overflow(monkeypatch):
    from dace.sdfg.state import LoopRegion

    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = dace.SDFG('cutile_loop_runtime_overflow')
    sdfg.backend = dtypes.BackendLanguage.Python
    loop = LoopRegion('loop',
                      condition_expr='i < 3',
                      loop_var='i',
                      initialize_expr='i = 0',
                      update_expr='i = i + 1',
                      sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    assign = loop.add_state('assign', is_start_block=True)
    kernel = loop.add_state('kernel')
    _add_runtime_kernel(sdfg, kernel, 'loop_overflow')
    loop.add_edge(assign, kernel, dace.InterstateEdge(assignments={'runtime_value': f'{2**64 - 2} + i'}))

    with pytest.raises(CodegenError, match="no unambiguous reaching dtype.*runtime_value"):
        sdfg.generate_code()


def test_conflicting_merge_runtime_types_raise(monkeypatch):
    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = dace.SDFG('cutile_conflicting_runtime_types')
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_symbol('flag', dace.bool)
    init = sdfg.add_state('init')
    left = sdfg.add_state('left')
    right = sdfg.add_state('right')
    merge = sdfg.add_state('merge')
    _add_runtime_kernel(sdfg, merge, 'merge')
    sdfg.add_edge(init, left, dace.InterstateEdge('flag', assignments={'runtime_value': '7'}))
    sdfg.add_edge(init, right, dace.InterstateEdge('not flag', assignments={'runtime_value': '1.5'}))
    sdfg.add_edge(left, merge, dace.InterstateEdge())
    sdfg.add_edge(right, merge, dace.InterstateEdge())

    with pytest.raises(CodegenError, match="conflicting reaching dtypes.*runtime_value"):
        sdfg.generate_code()


def test_python_assignment_int_cast_after_division_uses_proven_safe_range():
    from dace.codegen.py.cutile_target import _python_assignment_type

    assert _python_assignment_type('int(N / 2)', {'N': dace.int64}, {}) == dace.int64


@pytest.mark.parametrize('target', ['round', 'numpy.round'])
def test_generated_numpy_round_preserves_input_dtype(target):
    from dace.codegen.py.cutile_target import _python_assignment_type

    assert _python_assignment_type(f'{target}(N / 2)', {'N': dace.int64}, {}) == dace.float64


def test_builtin_and_numpy_abs_have_distinct_bool_results():
    from dace.codegen.py.cutile_target import _python_assignment_type

    assert _python_assignment_type('abs(True)', {}, {}) == dace.int64
    assert _python_assignment_type('numpy.abs(True)', {}, {}) == dace.bool


@pytest.mark.parametrize('expression', [
    'np.round(N)',
    'other.round(N)',
    'other.abs(N)',
    'other.int64(N)',
    'dace.int64(N)',
])
def test_unknown_or_unavailable_dotted_call_targets_are_rejected(expression):
    from dace.codegen.py.cutile_target import _python_assignment_type

    with pytest.raises(CodegenError, match='no unambiguous supported Python result type'):
        _python_assignment_type(expression, {'N': dace.int64}, {})


def test_explicit_numpy_dtype_cast_is_supported():
    from dace.codegen.py.cutile_target import _python_assignment_type

    assert _python_assignment_type('numpy.int32(N)', {'N': dace.int64}, {}) == dace.int32


@pytest.mark.parametrize('expression', ['True and 2**63', '2**63 if True else 0'])
def test_python_assignment_static_control_flow_preserves_uint64(expression):
    from dace.codegen.py.cutile_target import _python_assignment_type

    assert _python_assignment_type(expression, {}, {}) == dace.uint64


def test_python_assignment_boolop_returns_operand_types():
    from dace.codegen.py.cutile_target import _python_assignment_types

    types = {'N': frozenset({dace.int64})}
    assert _python_assignment_types('N / 2 and 1', types, {}) == frozenset({dace.float64, dace.int64})


def test_many_ambiguous_symbols_use_bounded_candidate_propagation():
    from dace.codegen.py.cutile_target import _python_assignment_types

    types = {f'value_{index}': frozenset({dace.int8, dace.float64}) for index in range(80)}
    expression = ' + '.join(types)
    assert _python_assignment_types(expression, types, {}) == frozenset({dace.int64, dace.float64})


@pytest.mark.parametrize('expression, expected_dtype', [
    ('B + B', dace.int64),
    ('B - B', dace.int64),
    ('B * B', dace.int64),
    ('B & B', dace.bool),
    ('B | B', dace.bool),
    ('B ^ B', dace.bool),
])
def test_native_bool_binary_operations_follow_python_semantics(expression, expected_dtype):
    from dace.codegen.py.cutile_target import _python_assignment_type

    assert _python_assignment_type(expression, {'B': dace.bool}, {}) == expected_dtype


@pytest.mark.parametrize('expression, symbols', [
    ('L + R', {
        'L': dace.int64,
        'R': dace.int64
    }),
    ('L - R', {
        'L': dace.int64,
        'R': dace.int64
    }),
    ('L * R', {
        'L': dace.int64,
        'R': dace.int64
    }),
    ('L + R', {
        'L': dace.uint64,
        'R': dace.uint64
    }),
])
def test_native_full_width_integer_overflow_is_rejected(expression, symbols):
    from dace.codegen.py.cutile_target import _python_assignment_type

    with pytest.raises(CodegenError, match='no unambiguous supported Python result type'):
        _python_assignment_type(expression, symbols, {})


@pytest.mark.parametrize('expression', ['L + R', 'L - R', 'L * R', 'L // 2', 'L % 7', 'L << 2', 'L >> 2'])
def test_native_narrow_integer_ranges_choose_safe_int64_abi(expression):
    from dace.codegen.py.cutile_target import _python_assignment_type

    assert _python_assignment_type(expression, {'L': dace.int8, 'R': dace.uint8}, {}) == dace.int64


def test_native_bool_add_candidate_covers_python_value_two():
    from dace.codegen.py.cutile_target import _native_dynamic_candidate, _python_assignment_candidates

    incoming = {'B': frozenset({_native_dynamic_candidate(dace.bool)})}
    candidates = _python_assignment_candidates('B + B', incoming, {})

    assert {(candidate.dtype, candidate.numeric_range) for candidate in candidates} == {(dace.int64, (0, 2))}


def test_exact_numpy_scalar_candidates_preserve_object_dtype_and_singleton_range():
    from dace.codegen.py.cutile_target import (_HOST_NUMPY, _python_assignment_candidates)

    int_candidate = next(iter(_python_assignment_candidates('numpy.int8(-128)', {}, {})))
    abs_candidate = next(iter(_python_assignment_candidates('numpy.abs(numpy.int8(-128))', {}, {})))
    bool_candidate = next(iter(_python_assignment_candidates('numpy.bool_(True) + numpy.bool_(True)', {}, {})))

    assert isinstance(int_candidate.value, np.int8)
    assert (int_candidate.dtype, int_candidate.host_category, int_candidate.numeric_range) == (dace.int8, _HOST_NUMPY,
                                                                                               (-128, -128))
    assert isinstance(abs_candidate.value, np.int8)
    assert (abs_candidate.dtype, int(abs_candidate.value), abs_candidate.numeric_range) == (dace.int8, -128, (-128,
                                                                                                              -128))
    assert isinstance(bool_candidate.value, np.bool_)
    assert (bool_candidate.dtype, bool(bool_candidate.value), bool_candidate.numeric_range) == (dace.bool, True, (1, 1))


@pytest.mark.parametrize('expression', [
    'int(numpy.int16(I) + 1)',
    'int(numpy.int16(I) // True)',
    'int(numpy.int16(I) % True)',
])
def test_numpy_integer_result_ranges_support_safe_native_int_cast(expression):
    from dace.codegen.py.cutile_target import _python_assignment_type

    assert _python_assignment_type(expression, {'I': dace.int16}, {}) == dace.int64


@pytest.mark.parametrize('expression, expected_dtype', [
    ('numpy.float32(F) + 1.0', dace.float32),
    ('numpy.int16(I) + 1', dace.int16),
])
def test_numpy_operations_apply_proven_weak_scalar_promotion(expression, expected_dtype):
    from dace.codegen.py.cutile_target import _python_assignment_type

    assert _python_assignment_type(expression, {'F': dace.float32, 'I': dace.int16}, {}) == expected_dtype


def test_numpy_operation_with_dynamic_native_scalar_fails_closed():
    from dace.codegen.py.cutile_target import _python_assignment_type

    with pytest.raises(CodegenError, match='no unambiguous supported Python result type'):
        _python_assignment_type('numpy.float32(F) + N', {'F': dace.float32, 'N': dace.int8}, {})


def test_sequential_assignments_preserve_numpy_candidate_metadata():
    import ast
    from dace.codegen.py.cutile_target import (_HOST_NUMPY, _native_dynamic_candidate, _transfer_python_statements)

    incoming = {'F': frozenset({_native_dynamic_candidate(dace.float32)})}
    statements = ast.parse('temporary = numpy.float32(F)\nabsolute = numpy.abs(temporary)\npositive = +temporary').body
    result = _transfer_python_statements(statements, incoming, {})

    for name in ('temporary', 'absolute', 'positive'):
        assert {candidate.dtype for candidate in result[name]} == {dace.float32}
        assert {candidate.host_category for candidate in result[name]} == {_HOST_NUMPY}


@pytest.mark.parametrize('expression', [
    'A is B',
    'A is not B',
    'A == B is C',
    'A is B != C',
])
def test_identity_comparisons_fail_closed(expression):
    from dace.codegen.py.cutile_target import _python_assignment_type

    with pytest.raises(CodegenError, match='no unambiguous supported Python result type'):
        _python_assignment_type(expression, {'A': dace.int64, 'B': dace.int64, 'C': dace.int64}, {})


@pytest.mark.parametrize('expression, expected_dtype', [
    ('int(N / 2)', 'int64'),
    ('round(N / 2)', 'float64'),
    ('True and 2**63', 'uint64'),
    ('2**63 if True else 0', 'uint64'),
])
def test_aot_signature_uses_local_python_expression_dtype(monkeypatch, expression, expected_dtype):
    from dace.codegen.py import cutile_aot

    captured = []

    def fake_generate_aot_module(kernels, module_name):
        captured.extend(kernels)
        return '_KERNELS = {}\ndef launch(kernel_name, grid, args): pass\n'

    monkeypatch.setenv('DACE_compiler_cutile_mode', 'aot')
    monkeypatch.setattr(cutile_aot, 'generate_aot_module', fake_generate_aot_module)
    sdfg = _runtime_symbol_sdfg(f'cutile_aot_expr_{expected_dtype}', expression)
    if 'N' in expression:
        sdfg.add_symbol('N', dace.int64)

    codegen.generate_code(sdfg)

    assert len(captured) == 1
    assert captured[0]['params'][-1].dtype == expected_dtype


def test_operand_returning_boolop_is_rejected_at_kernel(monkeypatch):
    monkeypatch.setenv('DACE_compiler_cutile_mode', 'aot')
    sdfg = _runtime_symbol_sdfg('cutile_aot_boolop_ambiguous', 'N / 2 and 1')
    sdfg.add_symbol('N', dace.int64)

    with pytest.raises(CodegenError, match="conflicting reaching dtypes.*runtime_value"):
        codegen.generate_code(sdfg)


@pytest.mark.parametrize('expression', ['+B', '-B', '~B'])
def test_dynamic_bool_unary_operations_are_int64(expression):
    from dace.codegen.py.cutile_target import _python_assignment_type

    assert _python_assignment_type(expression, {'B': dace.bool}, {}) == dace.int64


def test_negative_integer_power_is_float64():
    from dace.codegen.py.cutile_target import _python_assignment_type

    assert _python_assignment_type('N ** -1', {'N': dace.int64}, {}) == dace.float64


@pytest.mark.parametrize('expression', ['N ** 2', 'N ** M'])
def test_value_dependent_dynamic_integer_power_is_rejected(expression):
    from dace.codegen.py.cutile_target import _python_assignment_type

    with pytest.raises(CodegenError, match='no unambiguous supported Python result type'):
        _python_assignment_type(expression, {'N': dace.int64, 'M': dace.int64}, {})


@pytest.mark.parametrize('dtype', [dace.bool, dace.uint8, dace.uint16, dace.uint32])
def test_dynamic_builtin_int_uses_int64_when_source_range_fits(dtype):
    from dace.codegen.py.cutile_target import _python_assignment_type

    assert _python_assignment_type('int(U)', {'U': dtype}, {}) == dace.int64


def test_dynamic_uint64_int_cast_preserves_unsigned_dtype():
    from dace.codegen.py.cutile_target import _python_assignment_type

    assert _python_assignment_type('int(U)', {'U': dace.uint64}, {}) == dace.uint64


def test_dynamic_float_to_builtin_int_is_rejected_without_range_information():
    from dace.codegen.py.cutile_target import _python_assignment_type

    with pytest.raises(CodegenError, match='no unambiguous supported Python result type'):
        _python_assignment_type('int(F)', {'F': dace.float64}, {})


@pytest.mark.parametrize('target', ['round', 'numpy.round'])
def test_generated_numpy_round_promotes_dynamic_bool_to_float16(target):
    from dace.codegen.py.cutile_target import _python_assignment_type

    assert _python_assignment_type(f'{target}(B)', {'B': dace.bool}, {}) == dace.float16


@pytest.mark.parametrize('expression, symbols', [
    ('-U', {
        'U': dace.uint64
    }),
    ('~U', {
        'U': dace.uint64
    }),
    ('-I', {
        'I': dace.int64
    }),
])
def test_dynamic_unary_integer_boundary_cases_are_rejected(expression, symbols):
    from dace.codegen.py.cutile_target import _python_assignment_type

    with pytest.raises(CodegenError, match='no unambiguous supported Python result type'):
        _python_assignment_type(expression, symbols, {})


def test_dynamic_unary_plus_uint64_remains_uint64():
    from dace.codegen.py.cutile_target import _python_assignment_type

    assert _python_assignment_type('+U', {'U': dace.uint64}, {}) == dace.uint64


@pytest.mark.parametrize('expression', ['X ** 0.5', 'X ** Y'])
def test_dynamic_real_power_that_may_be_complex_is_rejected(expression):
    from dace.codegen.py.cutile_target import _python_assignment_type

    with pytest.raises(CodegenError, match='no unambiguous supported Python result type'):
        _python_assignment_type(expression, {'X': dace.float64, 'Y': dace.float64}, {})


def test_marshaled_native_complex_abs_returns_python_float64():
    from dace.codegen.py.cutile_target import _python_assignment_type

    assert _python_assignment_type('abs(Z)', {'Z': dace.complex128}, {}) == dace.float64
    assert _python_assignment_type('abs(Z)', {'Z': dace.complex64}, {}) == dace.float64


@pytest.mark.parametrize('expression', ['+I', '-I', '~I'])
def test_marshaled_narrow_native_integer_unary_results_use_int64(expression):
    from dace.codegen.py.cutile_target import _python_assignment_type

    assert _python_assignment_type(expression, {'I': dace.int8}, {}) == dace.int64


@pytest.mark.parametrize('expression, symbol_name, symbol_dtype, symbol_value, expected_dtype', [
    ('round(F)', 'F', dace.float32, np.float32(1.25), 'float64'),
    ('abs(Z)', 'Z', dace.complex64, np.complex64(1 + 2j), 'float64'),
    ('+I', 'I', dace.int8, np.int8(-128), 'int64'),
    ('-I', 'I', dace.int8, np.int8(-128), 'int64'),
    ('~I', 'I', dace.int8, np.int8(-128), 'int64'),
])
def test_generated_frame_matches_native_symbol_marshalling(monkeypatch, expression, symbol_name, symbol_dtype,
                                                           symbol_value, expected_dtype):
    from dace.codegen.py.compiled_sdfg import PythonCompiledSDFG

    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = _runtime_symbol_sdfg(f'cutile_marshaled_{symbol_name}_{expected_dtype}', expression)
    sdfg.add_symbol(symbol_name, symbol_dtype)

    code = sdfg.generate_code()[0].code
    compiled = object.__new__(PythonCompiledSDFG)
    compiled._sdfg = sdfg
    marshalled = compiled._marshal_arguments({symbol_name: symbol_value})[symbol_name]

    assert type(marshalled) in (bool, int, float, complex)
    assert f'runtime_value = {expression}' in code
    assert f'cupy.asarray(runtime_value, dtype=numpy.{expected_dtype}).reshape(1)' in code


def test_unsupported_log_call_is_rejected_at_kernel(monkeypatch):
    monkeypatch.setenv('DACE_compiler_cutile_mode', 'aot')
    sdfg = _runtime_symbol_sdfg('cutile_aot_unsupported_log', 'log(N)')
    sdfg.add_symbol('N', dace.int64)

    with pytest.raises(CodegenError, match="no unambiguous reaching dtype.*runtime_value"):
        codegen.generate_code(sdfg)


@pytest.mark.parametrize('expression', ['2 ** 1000000', '1 << 1000000'])
def test_huge_static_power_and_shift_are_rejected_before_evaluation(expression):
    from dace.codegen.py.cutile_target import _python_assignment_type

    with pytest.raises(CodegenError, match='bounded|intermediate bits'):
        _python_assignment_type(expression, {}, {})


def test_exact_large_intermediate_is_checked_only_after_final_operation():
    from dace.codegen.py.cutile_target import _python_assignment_type

    assert _python_assignment_type('2**64 - 1', {}, {}) == dace.uint64


@pytest.mark.parametrize('expression, expected_dtype', [
    ('not 2**100', dace.bool),
    ('1 if 2**100 else 1.5', dace.int64),
])
def test_oversized_exact_truth_tests_do_not_require_materialization(expression, expected_dtype):
    from dace.codegen.py.cutile_target import _python_assignment_type

    assert _python_assignment_type(expression, {}, {}) == expected_dtype


@pytest.mark.parametrize('expression, expected_dtype', [
    ('1 ** 1000000', dace.int64),
    ('(-1) ** 1000001', dace.int64),
    ('123 >> 1000000', dace.int64),
    ('-123 >> 1000000', dace.int64),
    ('0 << 1000000', dace.int64),
])
def test_trivial_huge_static_operations_are_evaluated_without_large_intermediates(expression, expected_dtype):
    from dace.codegen.py.cutile_target import _python_assignment_type

    assert _python_assignment_type(expression, {}, {}) == expected_dtype


def _finite_recurrence_sdfg(name: str, runtime_update: str) -> dace.SDFG:
    from dace.sdfg.state import LoopRegion

    sdfg = dace.SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    loop = LoopRegion('loop',
                      condition_expr='i < 1000',
                      loop_var='i',
                      initialize_expr='i = 0\nruntime_value = 10',
                      update_expr=f'i = i + 1\nruntime_value = {runtime_update}',
                      sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    kernel = loop.add_state('kernel', is_start_block=True)
    _add_runtime_kernel(sdfg, kernel, name)
    return sdfg


@pytest.mark.parametrize('runtime_update', ['runtime_value + 1', 'runtime_value - 1'])
def test_generated_large_finite_integer_recurrence_converges(monkeypatch, runtime_update):
    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = _finite_recurrence_sdfg('cutile_finite_recurrence', runtime_update)

    code = ''.join(sdfg.generate_code()[0].code.split())

    assert 'cupy.asarray(runtime_value,dtype=numpy.int64).reshape(1)' in code


@pytest.mark.parametrize('seed', [2**63 - 3, 2**64 - 3], ids=['int64', 'uint64'])
@pytest.mark.parametrize('update_expression', [
    'i = i + 1\nruntime_value += i',
    'runtime_value += i\ni = i + 1',
],
                         ids=['induction-first', 'recurrence-first'])
def test_generated_loop_variable_recurrence_is_not_stabilized(monkeypatch, seed, update_expression):
    from dace.sdfg.state import LoopRegion

    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = dace.SDFG('cutile_variable_recurrence')
    sdfg.backend = dtypes.BackendLanguage.Python
    loop = LoopRegion('loop',
                      condition_expr='i < 3',
                      loop_var='i',
                      initialize_expr=f'i = 1\nruntime_value = {seed}',
                      update_expr=update_expression,
                      sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    kernel = loop.add_state('kernel', is_start_block=True)
    _add_runtime_kernel(sdfg, kernel, 'variable_recurrence')

    with pytest.raises(CodegenError):
        sdfg.generate_code()


def test_generated_bool_recurrence_retains_seed_type(monkeypatch):
    from dace.sdfg.state import LoopRegion

    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = dace.SDFG('cutile_bool_recurrence')
    sdfg.backend = dtypes.BackendLanguage.Python
    loop = LoopRegion('loop',
                      condition_expr='i < 3',
                      loop_var='i',
                      initialize_expr='i = 0\nruntime_value = True',
                      update_expr='i = i + 1\nruntime_value += 1',
                      sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    kernel = loop.add_state('kernel', is_start_block=True)
    _add_runtime_kernel(sdfg, kernel, 'bool_recurrence')

    with pytest.raises(CodegenError, match='conflicting reaching dtypes.*runtime_value'):
        sdfg.generate_code()


def test_generated_bool_induction_seed_is_not_stabilized(monkeypatch):
    from dace.sdfg.state import LoopRegion

    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = dace.SDFG('cutile_bool_induction')
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_symbol('i', dace.bool)
    loop = LoopRegion('loop',
                      condition_expr='i < 3',
                      loop_var='i',
                      initialize_expr='i = True',
                      update_expr='i = i + 1',
                      sdfg=sdfg)
    assign = loop.add_state('assign', is_start_block=True)
    kernel = loop.add_state('kernel')
    init_statement = loop.init_statement
    loop.init_statement = None
    _add_runtime_kernel(sdfg, kernel, 'bool_induction')
    loop.init_statement = init_statement
    loop.add_edge(assign, kernel, dace.InterstateEdge(assignments={'runtime_value': 'i'}))
    sdfg.add_node(loop, is_start_block=True)

    with pytest.raises(CodegenError, match='conflicting reaching dtypes.*runtime_value'):
        sdfg.generate_code()


def test_numpy_scalar_induction_initializer_falls_back_without_symbolic_error():
    from dace.sdfg.state import LoopRegion

    sdfg = dace.SDFG('cutile_numpy_induction_proof')
    loop = LoopRegion('loop',
                      condition_expr='i < 3',
                      loop_var='i',
                      initialize_expr='i = numpy.int8(0)',
                      update_expr='i = i + 1',
                      sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)

    assert _induction_info(loop) is None


def test_generated_numpy_int8_recurrence_preserves_numpy_semantics(monkeypatch):
    from dace.sdfg.state import LoopRegion

    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = dace.SDFG('cutile_numpy_int8_recurrence')
    sdfg.backend = dtypes.BackendLanguage.Python
    loop = LoopRegion('loop',
                      condition_expr='i < 3',
                      loop_var='i',
                      initialize_expr='i = 0\nruntime_value = numpy.int8(0)',
                      update_expr='i = i + 1\nruntime_value = runtime_value + 1',
                      sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    kernel = loop.add_state('kernel', is_start_block=True)
    _add_runtime_kernel(sdfg, kernel, 'numpy_counter')

    code = ''.join(sdfg.generate_code()[0].code.split())

    assert 'cupy.asarray(runtime_value,dtype=numpy.int8).reshape(1)' in code


def _induction_info(loop, incoming=None):
    from dace.codegen.py.cutile_target import _loop_induction_info, _transfer_codeblock

    incoming = {} if incoming is None else incoming
    initialized = _transfer_codeblock(loop.init_statement, incoming, loop.sdfg.constants)
    return _loop_induction_info(loop, initialized, loop.sdfg.constants)


def test_float_step_is_not_treated_as_canonical_integer_induction():
    from dace.sdfg.state import LoopRegion

    sdfg = dace.SDFG('cutile_float_step_proof')
    loop = LoopRegion('loop',
                      condition_expr='i < 2.0',
                      loop_var='i',
                      initialize_expr='i = 0.0',
                      update_expr='i = i + 0.5',
                      sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)

    assert _induction_info(loop) is None


def test_generated_float_step_uses_analyzed_float_update(monkeypatch):
    from dace.sdfg.state import LoopRegion

    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = dace.SDFG('cutile_float_step_generated')
    sdfg.backend = dtypes.BackendLanguage.Python
    loop = LoopRegion('loop',
                      condition_expr='i < 2.0',
                      loop_var='i',
                      initialize_expr='i = 0.0',
                      update_expr='i = i + 0.5',
                      sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    assign = loop.add_state('assign', is_start_block=True)
    kernel = loop.add_state('kernel')
    _add_runtime_kernel(sdfg, kernel, 'float_step')
    loop.add_edge(assign, kernel, dace.InterstateEdge(assignments={'runtime_value': 'i'}))

    code = ''.join(sdfg.generate_code()[0].code.split())

    assert 'cupy.asarray(runtime_value,dtype=numpy.float64).reshape(1)' in code


def test_inverted_post_condition_is_not_induction_clamped():
    from dace.sdfg.state import LoopRegion

    sdfg = dace.SDFG('cutile_inverted_proof')
    loop = LoopRegion('loop',
                      condition_expr='i < 2',
                      loop_var='i',
                      initialize_expr='i = 0',
                      update_expr='i = i + 1',
                      inverted=True,
                      update_before_condition=False,
                      sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)

    assert _induction_info(loop) is None


def test_mutated_condition_bound_is_not_induction_clamped():
    from dace.codegen.py.cutile_target import _native_dynamic_candidate
    from dace.sdfg.state import LoopRegion

    sdfg = dace.SDFG('cutile_mutated_bound_proof')
    sdfg.add_symbol('limit', dace.int64)
    loop = LoopRegion('loop',
                      condition_expr='i < limit',
                      loop_var='i',
                      initialize_expr='i = 0',
                      update_expr='i = i + 1',
                      sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    start = loop.add_state('start', is_start_block=True)
    end = loop.add_state('end')
    loop.add_edge(start, end, dace.InterstateEdge(assignments={'limit': 'limit + 1'}))
    incoming = {'limit': frozenset({_native_dynamic_candidate(dace.int64)})}

    assert _induction_info(loop, incoming) is None


def test_uint64_exit_overshoot_is_not_induction_clamped():
    from dace.sdfg.state import LoopRegion

    sdfg = dace.SDFG('cutile_uint64_overshoot_proof')
    loop = LoopRegion('loop',
                      condition_expr=f'i <= {2**64 - 1}',
                      loop_var='i',
                      initialize_expr=f'i = {2**64 - 2}',
                      update_expr='i = i + 2',
                      sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)

    assert _induction_info(loop) is None


@pytest.mark.parametrize('case', ['inverted', 'uint64_overshoot'])
def test_generated_unproven_induction_counterexamples_fail_closed(monkeypatch, case):
    from dace.sdfg.state import LoopRegion

    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = dace.SDFG(f'cutile_unproven_{case}')
    sdfg.backend = dtypes.BackendLanguage.Python
    if case == 'inverted':
        loop = LoopRegion('loop',
                          condition_expr='i < 2',
                          loop_var='i',
                          initialize_expr='i = 0',
                          update_expr='i = i + 1',
                          inverted=True,
                          update_before_condition=False,
                          sdfg=sdfg)
    elif case == 'mutated_bound':
        sdfg.add_symbol('limit', dace.int64)
        loop = LoopRegion('loop',
                          condition_expr='i < limit',
                          loop_var='i',
                          initialize_expr='i = 0',
                          update_expr='i = i + 1',
                          sdfg=sdfg)
    else:
        loop = LoopRegion('loop',
                          condition_expr=f'i <= {2**64 - 1}',
                          loop_var='i',
                          initialize_expr=f'i = {2**64 - 2}',
                          update_expr='i = i + 2',
                          sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    assign = loop.add_state('assign', is_start_block=True)
    kernel = loop.add_state('kernel')
    _add_runtime_kernel(sdfg, kernel, case)
    assignments = {'runtime_value': 'i'}
    if case == 'mutated_bound':
        assignments['limit'] = 'limit + 1'
    loop.add_edge(assign, kernel, dace.InterstateEdge(assignments=assignments))

    with pytest.raises(CodegenError):
        sdfg.generate_code()


def _loop_local_expression_sdfg(name: str, init_expression: str) -> dace.SDFG:
    from dace.sdfg.state import LoopRegion

    sdfg = dace.SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    loop = LoopRegion('loop',
                      condition_expr='i < 1',
                      loop_var='i',
                      initialize_expr=f'i = 0\n{init_expression}',
                      update_expr='i = i + 1',
                      sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    kernel = loop.add_state('kernel', is_start_block=True)
    _add_runtime_kernel(sdfg, kernel, name)
    return sdfg


@pytest.mark.parametrize('expression, expected_dtype', [
    ('temporary = numpy.int8(-128)\nruntime_value = numpy.abs(temporary)', 'int8'),
    ('temporary = numpy.int8(-128)\nruntime_value = abs(temporary)', 'int8'),
    ('temporary = numpy.bool_(True)\nruntime_value = temporary + temporary', 'bool_'),
])
def test_generated_exact_numpy_scalar_semantics(monkeypatch, expression, expected_dtype):
    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = _loop_local_expression_sdfg(f'cutile_numpy_exact_{expected_dtype}', expression)

    code = ''.join(sdfg.generate_code()[0].code.split())

    assert f'cupy.asarray(runtime_value,dtype=numpy.{expected_dtype}).reshape(1)' in code


@pytest.mark.parametrize('expression, symbol_name, symbol_dtype, expected_dtype', [
    ('temporary = numpy.int8(I)\nruntime_value = numpy.abs(temporary)', 'I', dace.int8, 'int8'),
    ('temporary = numpy.bool_(B)\nruntime_value = temporary + temporary', 'B', dace.bool, 'bool_'),
])
def test_generated_dynamic_numpy_scalar_semantics(monkeypatch, expression, symbol_name, symbol_dtype, expected_dtype):
    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = _loop_local_expression_sdfg(f'cutile_numpy_dynamic_{expected_dtype}', expression)
    sdfg.add_symbol(symbol_name, symbol_dtype)

    code = ''.join(sdfg.generate_code()[0].code.split())

    assert f'cupy.asarray(runtime_value,dtype=numpy.{expected_dtype}).reshape(1)' in code


def test_generated_identity_assignment_is_rejected(monkeypatch):
    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = _loop_local_expression_sdfg('cutile_identity_rejected', 'runtime_value = i is i')

    with pytest.raises(CodegenError, match="no unambiguous reaching dtype.*runtime_value"):
        sdfg.generate_code()


def _loop_carried_runtime_sdfg(name: str, tail_expression: str) -> dace.SDFG:
    from dace.sdfg.state import LoopRegion

    sdfg = dace.SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    init = sdfg.add_state('init')
    loop = LoopRegion('loop',
                      condition_expr='i < 3',
                      loop_var='i',
                      initialize_expr='i = 0',
                      update_expr='i = i + 1',
                      sdfg=sdfg)
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={'runtime_value': '1'}))
    kernel = loop.add_state('kernel', is_start_block=True)
    tail = loop.add_state('tail')
    _add_runtime_kernel(sdfg, kernel, 'loop_top')
    loop.add_edge(kernel, tail, dace.InterstateEdge(assignments={'runtime_value': tail_expression}))
    return sdfg


def test_numpy_metadata_crosses_multiple_reaching_edges():
    from types import SimpleNamespace
    from dace.codegen.py.cutile_target import (_HOST_NUMPY, _native_dynamic_candidate, _transfer_reaching_types)

    incoming = {'F': frozenset({_native_dynamic_candidate(dace.float32)})}
    cast_edge = SimpleNamespace(data=SimpleNamespace(assignments={'temporary': 'numpy.float32(F)'}))
    abs_edge = SimpleNamespace(data=SimpleNamespace(assignments={'runtime_value': 'numpy.abs(temporary)'}))

    after_cast = _transfer_reaching_types(cast_edge, incoming, {})
    after_abs = _transfer_reaching_types(abs_edge, after_cast, {})

    assert {candidate.dtype for candidate in after_abs['runtime_value']} == {dace.float32}
    assert {candidate.host_category for candidate in after_abs['runtime_value']} == {_HOST_NUMPY}


def test_loop_carried_safe_division_then_negation_remains_int64(monkeypatch):
    from dace.sdfg.state import LoopRegion

    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = dace.SDFG('cutile_loop_carried_safe_range')
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_symbol('N', dace.int64)
    loop = LoopRegion('loop',
                      condition_expr='i < 3',
                      loop_var='i',
                      initialize_expr='i = 0\nruntime_value = int(N / 2)',
                      update_expr='i = i + 1\nruntime_value = -runtime_value',
                      sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    kernel = loop.add_state('kernel', is_start_block=True)
    _add_runtime_kernel(sdfg, kernel, 'safe_range')

    code = ''.join(sdfg.generate_code()[0].code.split())

    assert 'cupy.asarray(runtime_value,dtype=numpy.int64).reshape(1)' in code


def test_loop_top_accepts_same_loop_carried_type(monkeypatch):
    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = _loop_carried_runtime_sdfg('cutile_loop_carried_same', '2')

    code = ''.join(sdfg.generate_code()[0].code.split())

    assert 'cupy.asarray(runtime_value,dtype=numpy.int64).reshape(1)' in code


def test_loop_top_rejects_conflicting_later_iteration_type(monkeypatch):
    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = _loop_carried_runtime_sdfg('cutile_loop_carried_conflict', '1.5')

    with pytest.raises(CodegenError, match="conflicting reaching dtypes.*runtime_value"):
        sdfg.generate_code()


def _loop_update_runtime_sdfg(name: str, update_expression: str) -> dace.SDFG:
    from dace.sdfg.state import LoopRegion

    sdfg = dace.SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    init = sdfg.add_state('init')
    loop = LoopRegion('loop',
                      condition_expr='i < 3',
                      loop_var='i',
                      initialize_expr='i = 0\nruntime_value = 2',
                      update_expr=f'i = i + 1\nruntime_value = {update_expression}',
                      sdfg=sdfg)
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={'runtime_value': '1'}))
    kernel = loop.add_state('kernel', is_start_block=True)
    _add_runtime_kernel(sdfg, kernel, 'loop_update')
    return sdfg


def test_loop_init_only_runtime_symbol_is_local_in_generated_frame(monkeypatch):
    from dace.sdfg.state import LoopRegion

    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = dace.SDFG('cutile_loop_init_only_symbol')
    sdfg.backend = dtypes.BackendLanguage.Python
    loop = LoopRegion('loop',
                      condition_expr='i < 3',
                      loop_var='i',
                      initialize_expr='i = 0\nruntime_value = 2',
                      update_expr='i = i + 1',
                      sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    kernel = loop.add_state('kernel', is_start_block=True)
    _add_runtime_kernel(sdfg, kernel, 'init_only')

    code = sdfg.generate_code()[0].code
    function_line = next(line for line in code.splitlines() if line.startswith('def cutile_loop_init_only_symbol('))

    assert 'runtime_value' not in function_line
    assert 'runtime_value = 2' in code
    assert 'cupy.asarray(runtime_value, dtype=numpy.int64).reshape(1)' in code


def test_loop_local_discovery_excludes_conditional_augmented_and_nonlocal_stores():
    from dace.codegen.py.framecode import _definite_plain_codeblock_stores
    from dace.properties import CodeBlock

    assert _definite_plain_codeblock_stores(CodeBlock('plain = 1\ntyped: int = 2')) == {'plain', 'typed'}
    assert _definite_plain_codeblock_stores(CodeBlock('if flag:\n    conditional = 1')) == set()
    assert _definite_plain_codeblock_stores(CodeBlock('augmented += 1')) == set()
    assert _definite_plain_codeblock_stores(CodeBlock('global global_name\nglobal_name = 1')) == set()
    assert _definite_plain_codeblock_stores(CodeBlock('nonlocal closure_name\nclosure_name = 1')) == set()


def test_loop_init_and_update_accept_same_runtime_type(monkeypatch):
    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = _loop_update_runtime_sdfg('cutile_loop_update_same', '3')

    code = ''.join(sdfg.generate_code()[0].code.split())

    assert 'cupy.asarray(runtime_value,dtype=numpy.int64).reshape(1)' in code


def test_loop_update_rejects_conflicting_later_iteration_type(monkeypatch):
    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = _loop_update_runtime_sdfg('cutile_loop_update_conflict', '3.5')

    with pytest.raises(CodegenError, match="conflicting reaching dtypes.*runtime_value"):
        sdfg.generate_code()


def test_cyclic_reaching_flow_widens_and_terminates():
    from dace.codegen.py.cutile_target import _state_symbol_types

    sdfg = dace.SDFG('cutile_cyclic_reaching')
    start = sdfg.add_state('start', is_start_block=True)
    body = sdfg.add_state('body')
    sdfg.add_edge(start, body, dace.InterstateEdge(assignments={'runtime_value': '0'}))
    sdfg.add_edge(body, body, dace.InterstateEdge(assignments={'runtime_value': 'runtime_value + 1'}))

    candidates = _state_symbol_types(sdfg, sdfg, body)['runtime_value']

    assert {candidate.dtype for candidate in candidates} == {None, dace.int64}
    assert len(candidates) == 2


def test_candidate_lattice_joins_ranges_and_widens_without_enumeration():
    from dace.codegen.py.cutile_target import (_DYNAMIC_VALUE, _HOST_NATIVE, _ExpressionCandidate, _join_candidate_sets,
                                               _undefined_candidate)

    zero = _ExpressionCandidate(dace.int64, 0, _HOST_NATIVE, (0, 0))
    one = _ExpressionCandidate(dace.int64, 1, _HOST_NATIVE, (1, 1))
    joined = _join_candidate_sets(frozenset({zero}), frozenset({one}), 'value')
    candidate = next(iter(joined))
    assert candidate.value is _DYNAMIC_VALUE
    assert candidate.numeric_range == (0, 1)

    two = _ExpressionCandidate(dace.int64, 2, _HOST_NATIVE, (0, 2))
    widened = _join_candidate_sets(joined, frozenset({two, _undefined_candidate()}), 'value', widen=True)
    defined = next(candidate for candidate in widened if candidate.dtype is not None)
    assert defined.numeric_range == (-(2**63), 2**63 - 1)
    assert any(candidate.dtype is None for candidate in widened)
    assert len(widened) == 2


def test_undefined_candidate_merge_is_structural_and_idempotent():
    from dace.codegen.py.cutile_target import (_merge_reaching_types, _native_dynamic_candidate, _undefined_candidate)

    first_undefined = _undefined_candidate()
    second_undefined = _undefined_candidate()
    assert first_undefined == second_undefined
    assert hash(first_undefined) == hash(second_undefined)
    assert len({first_undefined, second_undefined}) == 1

    initial = {'runtime_value': frozenset({_native_dynamic_candidate(dace.int8)})}
    merged, changed = _merge_reaching_types(initial, {})
    repeated, repeated_changed = _merge_reaching_types(merged, {})

    assert changed
    assert not repeated_changed
    assert repeated == merged
    assert {candidate.dtype for candidate in merged['runtime_value']} == {None, dace.int64}


def test_partial_python_definition_merges_undefined_candidate():
    import ast
    from dace.codegen.py.cutile_target import _native_dynamic_candidate, _transfer_python_statements

    incoming = {'flag': frozenset({_native_dynamic_candidate(dace.bool)})}
    result = _transfer_python_statements(ast.parse('if flag:\n    runtime_value = 1').body, incoming, {})

    assert {candidate.dtype for candidate in result['runtime_value']} == {None, dace.int64}


def test_conditional_only_loop_exit_contributes_to_summary(monkeypatch):
    from dace.sdfg.state import LoopRegion

    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = dace.SDFG('cutile_conditional_only_loop_exit')
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_symbol('flag', dace.bool)
    init = sdfg.add_state('init')
    loop = LoopRegion('loop',
                      condition_expr='i < 2',
                      loop_var='i',
                      initialize_expr='i = 0',
                      update_expr='i = i + 1',
                      inverted=True,
                      sdfg=sdfg)
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={'runtime_value': '1'}))
    branch = loop.add_state('branch', is_start_block=True)
    explicit_exit = loop.add_state('explicit_exit')
    loop.add_edge(branch, explicit_exit, dace.InterstateEdge('flag', assignments={'runtime_value': '1.5'}))
    after = sdfg.add_state('after')
    sdfg.add_edge(loop, after, dace.InterstateEdge())
    _add_runtime_kernel(sdfg, after, 'conditional_exit')

    with pytest.raises(CodegenError, match="conflicting reaching dtypes.*runtime_value"):
        sdfg.generate_code()


@pytest.mark.parametrize('abrupt_kind', ['break', 'continue', 'return'])
def test_abrupt_loop_blocks_fail_closed(monkeypatch, abrupt_kind):
    from dace.sdfg.state import LoopRegion

    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = dace.SDFG(f'cutile_abrupt_{abrupt_kind}')
    sdfg.backend = dtypes.BackendLanguage.Python
    init = sdfg.add_state('init')
    loop = LoopRegion('loop',
                      condition_expr='i < 2',
                      loop_var='i',
                      initialize_expr='i = 0',
                      update_expr='i = i + 1',
                      sdfg=sdfg)
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={'runtime_value': '1'}))
    start = loop.add_state('start', is_start_block=True)
    if abrupt_kind == 'break':
        abrupt = loop.add_break('abrupt')
    elif abrupt_kind == 'continue':
        abrupt = loop.add_continue('abrupt')
    else:
        abrupt = loop.add_return('abrupt')
    loop.add_edge(start, abrupt, dace.InterstateEdge())
    after = sdfg.add_state('after')
    sdfg.add_edge(loop, after, dace.InterstateEdge())
    _add_runtime_kernel(sdfg, after, f'abrupt_{abrupt_kind}')

    with pytest.raises(CodegenError, match='abrupt control-flow block'):
        sdfg.generate_code()


def test_unsupported_compound_loop_update_fails_closed(monkeypatch):
    from dace.sdfg.state import LoopRegion

    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = dace.SDFG('cutile_unsupported_loop_update')
    sdfg.backend = dtypes.BackendLanguage.Python
    init = sdfg.add_state('init')
    loop = LoopRegion('loop',
                      condition_expr='i < 2',
                      loop_var='i',
                      initialize_expr='i = 0',
                      update_expr='for j in range(2):\n    runtime_value = j\ni = i + 1',
                      sdfg=sdfg)
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={'runtime_value': '1'}))
    kernel = loop.add_state('kernel', is_start_block=True)
    _add_runtime_kernel(sdfg, kernel, 'unsupported_update')

    with pytest.raises(CodegenError, match='unsupported loop init/update statement For'):
        sdfg.generate_code()


def test_side_effecting_loop_assignment_target_fails_closed(monkeypatch):
    from dace.sdfg.state import LoopRegion

    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = dace.SDFG('cutile_side_effecting_loop_assignment')
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array('runtime_array', [1], dace.int64)
    init = sdfg.add_state('init')
    loop = LoopRegion('loop',
                      condition_expr='i < 2',
                      loop_var='i',
                      initialize_expr='i = 0',
                      update_expr='runtime_array[0] = 2\ni = i + 1',
                      sdfg=sdfg)
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={'runtime_value': '1'}))
    kernel = loop.add_state('kernel', is_start_block=True)
    _add_runtime_kernel(sdfg, kernel, 'side_effecting_assignment')

    with pytest.raises(CodegenError, match='unsupported loop assignment target'):
        sdfg.generate_code()


@pytest.mark.parametrize('initialize_expr, update_expr, loop_var', [
    ('runtime_value = 1.5', None, 'i'),
    (None, 'runtime_value = 1.5', 'i'),
])
def test_partial_loop_components_are_not_emitted_or_analyzed(monkeypatch, initialize_expr, update_expr, loop_var):
    from dace.properties import CodeBlock
    from dace.sdfg.state import LoopRegion

    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = dace.SDFG('cutile_partial_loop_components')
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_symbol('i', dace.int64)
    init = sdfg.add_state('init')
    loop = LoopRegion('loop',
                      condition_expr='i < 2',
                      loop_var='i',
                      initialize_expr='i = 0',
                      update_expr='i = i + 1',
                      sdfg=sdfg)
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={'runtime_value': '1'}))
    kernel = loop.add_state('kernel', is_start_block=True)
    _add_runtime_kernel(sdfg, kernel, 'partial_components')
    loop.init_statement = CodeBlock(initialize_expr) if initialize_expr is not None else None
    loop.update_statement = CodeBlock(update_expr) if update_expr is not None else None
    loop.loop_variable = loop_var or ''

    code = ''.join(sdfg.generate_code()[0].code.split())

    assert 'cupy.asarray(runtime_value,dtype=numpy.int64).reshape(1)' in code
    assert 'runtime_value=1.5' not in code


def test_loop_without_variable_disables_init_and_update_analysis():
    from dace.codegen.py.cutile_target import _loop_components_enabled
    from dace.sdfg.state import LoopRegion

    loop = LoopRegion('loop',
                      condition_expr='True',
                      initialize_expr='runtime_value = 1.5',
                      update_expr='runtime_value = 2.5')

    assert not _loop_components_enabled(loop)


def test_nested_conditional_merge_rejects_conflicting_runtime_type(monkeypatch):
    from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion

    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    sdfg = dace.SDFG('cutile_nested_conditional_runtime_type')
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_symbol('flag', dace.bool)
    init = sdfg.add_state('init')
    loop = LoopRegion('loop',
                      condition_expr='i < 2',
                      loop_var='i',
                      initialize_expr='i = 0',
                      update_expr='i = i + 1',
                      sdfg=sdfg)
    sdfg.add_node(loop)
    sdfg.add_edge(init, loop, dace.InterstateEdge(assignments={'runtime_value': '1'}))
    conditional = ConditionalBlock('conditional', sdfg=sdfg)
    loop.add_node(conditional, is_start_block=True)
    for label, condition, expression in [('left', 'flag', '1'), ('right', None, '1.5')]:
        branch = ControlFlowRegion(label, sdfg=sdfg)
        start = branch.add_state(f'{label}_start', is_start_block=True)
        end = branch.add_state(f'{label}_end')
        branch.add_edge(start, end, dace.InterstateEdge(assignments={'runtime_value': expression}))
        conditional.add_branch(condition, branch)
    kernel = loop.add_state('kernel')
    _add_runtime_kernel(sdfg, kernel, 'nested_conditional')
    loop.add_edge(conditional, kernel, dace.InterstateEdge())

    with pytest.raises(CodegenError, match="conflicting reaching dtypes.*runtime_value"):
        sdfg.generate_code()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
