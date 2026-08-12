# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for native types in generated Cython hosts."""

from pathlib import Path

import numpy as np
import pytest

import dace
from dace import data, dtypes
from dace.codegen.py.framecode import _cython_memoryview_type, _cython_scalar_type

NATIVE_N = dace.symbol('NATIVE_N', dtype=dace.int64)
NATIVE_ALPHA = dace.symbol('NATIVE_ALPHA', dtype=dace.float64)


@dace.program
def _native_axpy(A: dace.float64[NATIVE_N], B: dace.float64[NATIVE_N]):
    for i in dace.map[0:NATIVE_N]:
        tmp = A[i] * NATIVE_ALPHA
        B[i] = tmp + i


def _annotation_line(source: str, function_name: str) -> str:
    """Return the native-locals decorator for a generated function.

    :param source: Generated Cython host source.
    :param function_name: Function whose decorator should be returned.
    :returns: The matching ``@cython.locals`` decorator block.
    """
    lines = source.splitlines()
    function_index = next(index for index, line in enumerate(lines) if line.startswith(f'def {function_name}('))
    cursor = function_index - 1
    while cursor >= 0 and not lines[cursor].strip():
        cursor -= 1
    if cursor < 0 or lines[cursor].strip() != ')':
        return ''
    decorator_index = max(index for index, line in enumerate(lines[:cursor + 1]) if line.startswith('@cython.locals('))
    return '\n'.join(lines[decorator_index:function_index])


def test_generated_host_annotates_stable_native_types():
    sdfg = _native_axpy.to_sdfg(simplify=False)
    sdfg.backend = dtypes.BackendLanguage.Python
    source = next(code.code for code in sdfg.generate_code() if code.language == 'pyx')

    assert 'cimport cython' in source
    assert 'from libc.stdint cimport' in source
    annotation = _annotation_line(source, sdfg.name)
    assert annotation.startswith('@cython.locals(')
    for expected in ('A=__dace_const_double[:]', 'B=double[:]', 'NATIVE_ALPHA=double', 'NATIVE_N=int64_t', 'i=int64_t'):
        assert expected in annotation


def test_tasklet_scalar_temporaries_are_native():
    sdfg = dace.SDFG('native_tasklet_locals')
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array('A', (1, ), dace.float64)
    sdfg.add_array('B', (1, ), dace.float64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet('typed', {'inp'}, {'out'}, 'tmp = inp * 2.0\nout = tmp + 1.0')
    state.add_edge(state.add_read('A'), None, tasklet, 'inp', dace.Memlet('A[0]'))
    state.add_edge(tasklet, 'out', state.add_write('B'), None, dace.Memlet('B[0]'))

    source = next(code.code for code in sdfg.generate_code() if code.language == 'pyx')
    annotation = _annotation_line(source, sdfg.name)
    for expected in ('inp=double', 'out=double', 'tmp=double'):
        assert expected in annotation


def test_whole_array_tasklet_remains_dynamic(tmp_path):
    sdfg = dace.SDFG('native_whole_array_tasklet')
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.build_folder = str(tmp_path / 'native_whole_array_tasklet')
    sdfg.add_array('X', (6, 4), dace.float64)
    sdfg.add_array('Y', (6, 4), dace.float64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet('scale', {'value'}, {'result'}, 'result = 2.0 * value')
    state.add_edge(state.add_read('X'), None, tasklet, 'value', dace.Memlet.from_array('X', sdfg.arrays['X']))
    state.add_edge(tasklet, 'result', state.add_write('Y'), None, dace.Memlet.from_array('Y', sdfg.arrays['Y']))

    source = next(code.code for code in sdfg.generate_code() if code.language == 'pyx')
    function_header = f'def {sdfg.name}('
    function_index = source.index(function_header)
    preceding_source = source[:function_index].rsplit('\ndef ', maxsplit=1)[-1]
    for forbidden in ('X=', 'Y=', 'value=', 'result='):
        assert forbidden not in preceding_source

    compiled = sdfg.compile()
    x = np.arange(24, dtype=np.float64).reshape(6, 4)
    x.flags.writeable = False
    y = np.empty_like(x)
    compiled(X=x, Y=y)
    np.testing.assert_array_equal(y, 2.0 * x)


def test_bulk_tasklet_names_reserve_function_scope(tmp_path):
    sdfg = dace.SDFG('native_bulk_scalar_name_collision')
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.build_folder = str(tmp_path / 'native_bulk_scalar_name_collision')
    sdfg.add_array('X', (3, 4), dace.float64)
    sdfg.add_array('Y', (3, 4), dace.float64)
    sdfg.add_array('A', (8, ), dace.float64)
    sdfg.add_array('B', (8, ), dace.float64)
    state = sdfg.add_state()

    bulk = state.add_tasklet('bulk', {'x'}, {'y'}, 'tmp = 2.0 * x\ny = tmp + 1.0')
    state.add_edge(state.add_read('X'), None, bulk, 'x', dace.Memlet.from_array('X', sdfg.arrays['X']))
    state.add_edge(bulk, 'y', state.add_write('Y'), None, dace.Memlet.from_array('Y', sdfg.arrays['Y']))

    entry, exit_node = state.add_map('scalar_map', {'i': '0:8'}, schedule=dtypes.ScheduleType.Sequential)
    scalar = state.add_tasklet('scalar', {'x'}, {'y'}, 'tmp = 3.0 * x\ny = tmp - 2.0')
    state.add_memlet_path(state.add_read('A'), entry, scalar, dst_conn='x', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(scalar, exit_node, state.add_write('B'), src_conn='y', memlet=dace.Memlet('B[i]'))

    compiled = sdfg.compile()
    annotation = _annotation_line(compiled.code, sdfg.name)
    assert 'i=int64_t' in annotation
    for name in ('X', 'Y', 'x', 'y', 'tmp'):
        assert f'    {name}=' not in annotation

    x_bulk = np.arange(12, dtype=np.float64).reshape(3, 4)
    y_bulk = np.empty_like(x_bulk)
    x_scalar = np.arange(8, dtype=np.float64)
    y_scalar = np.empty_like(x_scalar)
    compiled(X=x_bulk, Y=y_bulk, A=x_scalar, B=y_scalar)
    np.testing.assert_array_equal(y_bulk, 2.0 * x_bulk + 1.0)
    np.testing.assert_array_equal(y_scalar, 3.0 * x_scalar - 2.0)


def test_nested_whole_array_tasklet_has_no_native_array_annotations():
    sdfg = dace.SDFG('native_nested_whole_array')
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array('X', (6, 4), dace.float64)
    sdfg.add_array('Y', (6, 4), dace.float64)
    state = sdfg.add_state()

    nested = dace.SDFG('native_nested_whole_array_inner')
    nested.add_array('_x', (6, 4), dace.float64)
    nested.add_array('_y', (6, 4), dace.float64)
    nested_state = nested.add_state()
    tasklet = nested_state.add_tasklet('scale', {'__i'}, {'__o'}, '__o = 2.0 * __i')
    nested_state.add_edge(nested_state.add_read('_x'), None, tasklet, '__i',
                          dace.Memlet.from_array('_x', nested.arrays['_x']))
    nested_state.add_edge(tasklet, '__o', nested_state.add_write('_y'), None,
                          dace.Memlet.from_array('_y', nested.arrays['_y']))

    nested_node = state.add_nested_sdfg(nested, {'_x'}, {'_y'})
    state.add_edge(state.add_read('X'), None, nested_node, '_x', dace.Memlet.from_array('X', sdfg.arrays['X']))
    state.add_edge(nested_node, '_y', state.add_write('Y'), None, dace.Memlet.from_array('Y', sdfg.arrays['Y']))

    source = next(code.code for code in sdfg.generate_code() if code.language == 'pyx')
    helper_line = next(line for line in source.splitlines() if line.startswith('def native_nested_whole_array_inner_'))
    helper_name = helper_line.removeprefix('def ').split('(', maxsplit=1)[0]
    assert _annotation_line(source, helper_name) == ''
    assert _annotation_line(source, sdfg.name) == ''
    for expected in ('__i = _x[0:6, 0:4]', '__o = (2.0 * __i)', '_y[0:6, 0:4] = __o'):
        assert expected in source


def test_public_compile_runs_with_strided_cpu_memoryviews(tmp_path):
    sdfg = _native_axpy.to_sdfg(simplify=False)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.build_folder = str(tmp_path / 'native_axpy')
    compiled = sdfg.compile()

    size = 37
    input_storage = np.linspace(-2.0, 3.0, size * 2, dtype=np.float64)
    output_storage = np.full(size * 3, np.nan, dtype=np.float64)
    input_view = input_storage[1::2]
    output_view = output_storage[1::3]
    input_view.flags.writeable = False
    alpha = np.float64(1.75)

    compiled(A=input_view, B=output_view, NATIVE_ALPHA=alpha, NATIVE_N=size)

    expected = input_view * alpha + np.arange(size, dtype=np.float64)
    np.testing.assert_allclose(output_view, expected)
    annotation = _annotation_line(compiled.code, sdfg.name)
    assert 'A=__dace_const_double[:]' in annotation
    assert 'B=double[:]' in annotation
    assert 'NATIVE_ALPHA=double' in annotation
    assert 'NATIVE_N=int64_t' in annotation

    generated_c = list(Path(sdfg.build_folder).rglob('*.c'))
    assert len(generated_c) == 1
    c_source = generated_c[0].read_text(encoding='utf-8')
    assert '__Pyx_memviewslice __pyx_v_A' in c_source
    assert 'int64_t __pyx_v_NATIVE_N' in c_source
    assert 'int64_t __pyx_v_i' in c_source
    assert 'double __pyx_v_tmp' in c_source

    readonly_output = np.empty(size, dtype=np.float64)
    readonly_output.flags.writeable = False
    with pytest.raises((BufferError, ValueError), match='read-only|writable'):
        compiled(A=input_view, B=readonly_output, NATIVE_ALPHA=alpha, NATIVE_N=size)


def test_readonly_integer_input_uses_const_typedef(tmp_path):
    sdfg = dace.SDFG('native_const_int32')
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.build_folder = str(tmp_path / 'native_const_int32')
    sdfg.add_array('source', (9, ), dace.int32)
    sdfg.add_array('destination', (9, ), dace.int32)
    state = sdfg.add_state()
    entry, exit_node = state.add_map('copy', {'i': '0:9'}, schedule=dtypes.ScheduleType.Sequential)
    tasklet = state.add_tasklet('increment', {'value'}, {'result'}, 'result = value + 1')
    state.add_memlet_path(state.add_read('source'), entry, tasklet, dst_conn='value', memlet=dace.Memlet('source[i]'))
    state.add_memlet_path(tasklet,
                          exit_node,
                          state.add_write('destination'),
                          src_conn='result',
                          memlet=dace.Memlet('destination[i]'))

    compiled = sdfg.compile()
    source = np.arange(9, dtype=np.int32)
    source.flags.writeable = False
    destination = np.empty_like(source)
    compiled(source=source, destination=destination)

    np.testing.assert_array_equal(destination, np.arange(1, 10, dtype=np.int32))
    annotation = _annotation_line(compiled.code, sdfg.name)
    assert 'source=__dace_const_int32_t[:]' in annotation
    assert 'destination=int32_t[:]' in annotation


def test_gpu_and_unsupported_arrays_remain_python_objects():
    cpu = data.Array(dace.float32, (4, 8), storage=dtypes.StorageType.CPU_Heap)
    gpu = data.Array(dace.float32, (4, 8), storage=dtypes.StorageType.GPU_Global)
    half = data.Array(dace.float16, (8, ), storage=dtypes.StorageType.CPU_Heap)

    assert _cython_memoryview_type(cpu) == 'float[:, :]'
    assert _cython_memoryview_type(cpu, readonly=True) == '__dace_const_float[:, :]'
    assert _cython_memoryview_type(gpu) is None
    assert _cython_memoryview_type(half) is None
    assert _cython_scalar_type(dace.int32) == 'int32_t'
    assert _cython_scalar_type(dace.uint32) == 'uint32_t'


def test_cpu_array_crossing_to_gpu_remains_ndarray():
    size = dace.symbol('SIZE', dtype=dace.int64)
    sdfg = dace.SDFG('native_gpu_copy')
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_symbol('SIZE', dace.int64)
    sdfg.add_array('host', (size, ), dace.float64)
    sdfg.add_array('device', (size, ), dace.float64, storage=dtypes.StorageType.GPU_Global, transient=True)
    state = sdfg.add_state()
    state.add_edge(state.add_read('host'), None, state.add_write('device'), None, dace.Memlet('host[0:SIZE]'))

    source = next(code.code for code in sdfg.generate_code() if code.language == 'pyx')
    annotation = _annotation_line(source, sdfg.name)
    assert 'SIZE=int64_t' in annotation
    assert 'host=' not in annotation


def test_cpu_bulk_copy_destination_remains_ndarray():
    """Cython memoryviews cannot receive a NumPy slice directly."""
    sdfg = dace.SDFG('native_bulk_copy')
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array('A', (8, ), dace.float64)
    sdfg.add_array('B', (8, ), dace.float64)
    state = sdfg.add_state()
    state.add_edge(state.add_read('A'), None, state.add_write('B'), None, dace.Memlet('A[0:8]'))
    source = next(code.code for code in sdfg.generate_code() if code.language == 'pyx')
    annotation = _annotation_line(source, sdfg.name)
    assert 'A=__dace_const_double[:]' in annotation
    assert 'B=' not in annotation


def test_public_scalar_data_remains_a_writable_buffer():
    sdfg = dace.SDFG('native_scalar_buffer')
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_scalar('acc', dace.float64)
    state = sdfg.add_state()
    acc_read = state.add_read('acc')
    acc_write = state.add_write('acc')
    tasklet = state.add_tasklet('increment', {'value'}, {'result'}, 'result = value + 1.0')
    state.add_edge(acc_read, None, tasklet, 'value', dace.Memlet('acc[0]'))
    state.add_edge(tasklet, 'result', acc_write, None, dace.Memlet('acc[0]'))

    source = next(code.code for code in sdfg.generate_code() if code.language == 'pyx')
    annotation = _annotation_line(source, sdfg.name)
    assert 'acc=' not in annotation
