import itertools

import numpy as np
import pytest

import dace
from dace.dtypes import BackendLanguage, ScheduleType
from dace.properties import CodeBlock
from dace.sdfg import InterstateEdge, SDFG
from dace.sdfg.state import LoopRegion


_SDFG_COUNTER = itertools.count()


def _new_sdfg(prefix: str) -> SDFG:
    sdfg = SDFG(f"{prefix}_{next(_SDFG_COUNTER)}")
    sdfg.backend = BackendLanguage.Python
    return sdfg


def _run_sdfg(sdfg: SDFG, **kwargs):
    csdfg = sdfg.compile()
    csdfg(**kwargs)


def _assert_same(actual: np.ndarray, expected: np.ndarray) -> None:
    if np.issubdtype(actual.dtype, np.floating) or np.issubdtype(actual.dtype, np.complexfloating):
        np.testing.assert_allclose(actual, expected)
    else:
        np.testing.assert_array_equal(actual, expected)


def _build_fixed_index_binary_sdfg(prefix: str, dtype, size: int, code: str) -> SDFG:
    sdfg = _new_sdfg(prefix)
    sdfg.add_array('A', [size], dtype)
    sdfg.add_array('B', [size], dtype)
    sdfg.add_array('C', [size], dtype)

    state = sdfg.add_state(is_start_block=True)
    a_read = state.add_read('A')
    b_read = state.add_read('B')
    c_write = state.add_write('C')

    for idx in range(size):
        tasklet = state.add_tasklet(f'op_{idx}', {'lhs', 'rhs'}, {'out'}, code)
        state.add_edge(a_read, None, tasklet, 'lhs', dace.Memlet(f'A[{idx}]'))
        state.add_edge(b_read, None, tasklet, 'rhs', dace.Memlet(f'B[{idx}]'))
        state.add_edge(tasklet, 'out', c_write, None, dace.Memlet(f'C[{idx}]'))

    return sdfg


def _build_fixed_index_array_scalar_sdfg(prefix: str, dtype, size: int, code: str) -> SDFG:
    sdfg = _new_sdfg(prefix)
    sdfg.add_array('A', [size], dtype)
    sdfg.add_scalar('scale', dtype)
    sdfg.add_array('B', [size], dtype)

    state = sdfg.add_state(is_start_block=True)
    a_read = state.add_read('A')
    scale_read = state.add_read('scale')
    b_write = state.add_write('B')

    for idx in range(size):
        tasklet = state.add_tasklet(f'op_{idx}', {'inp', 'scale_value'}, {'out'}, code)
        state.add_edge(a_read, None, tasklet, 'inp', dace.Memlet(f'A[{idx}]'))
        state.add_edge(scale_read, None, tasklet, 'scale_value', dace.Memlet('scale'))
        state.add_edge(tasklet, 'out', b_write, None, dace.Memlet(f'B[{idx}]'))

    return sdfg


def _build_fixed_index_unary_sdfg(prefix: str, dtype, size: int, code: str) -> SDFG:
    sdfg = _new_sdfg(prefix)
    sdfg.add_array('A', [size], dtype)
    sdfg.add_array('B', [size], dtype)

    state = sdfg.add_state(is_start_block=True)
    a_read = state.add_read('A')
    b_write = state.add_write('B')

    for idx in range(size):
        tasklet = state.add_tasklet(f'op_{idx}', {'inp'}, {'out'}, code)
        state.add_edge(a_read, None, tasklet, 'inp', dace.Memlet(f'A[{idx}]'))
        state.add_edge(tasklet, 'out', b_write, None, dace.Memlet(f'B[{idx}]'))

    return sdfg


@pytest.mark.parametrize(
    ('dtype', 'np_dtype', 'lhs', 'rhs', 'code', 'expected'),
    [
        (dace.float32, np.float32, 1.25, 2.5, 'out = lhs + rhs', np.array([3.75], dtype=np.float32)),
        (dace.float32, np.float32, 6.0, 1.5, 'out = lhs * rhs', np.array([9.0], dtype=np.float32)),
        (dace.float64, np.float64, 3.0, 4.5, 'out = lhs + rhs', np.array([7.5], dtype=np.float64)),
        (dace.float64, np.float64, -2.0, 3.5, 'out = lhs * rhs', np.array([-7.0], dtype=np.float64)),
        (dace.int32, np.int32, 7, -4, 'out = lhs + rhs', np.array([3], dtype=np.int32)),
        (dace.int32, np.int32, -3, 5, 'out = lhs * rhs', np.array([-15], dtype=np.int32)),
        (dace.int64, np.int64, 11, 8, 'out = lhs + rhs', np.array([19], dtype=np.int64)),
        (dace.int64, np.int64, -6, 9, 'out = lhs * rhs', np.array([-54], dtype=np.int64)),
    ],
)
def test_singleton_scalar_like_arithmetic(dtype, np_dtype, lhs, rhs, code, expected):
    sdfg = _build_fixed_index_binary_sdfg('singleton_binary', dtype, 1, code)
    a = np.array([lhs], dtype=np_dtype)
    b = np.array([rhs], dtype=np_dtype)
    c = np.zeros(1, dtype=np_dtype)

    _run_sdfg(sdfg, A=a, B=b, C=c)

    _assert_same(c, expected)


@pytest.mark.parametrize(
    ('dtype', 'np_dtype', 'code', 'reference'),
    [
        (dace.float64, np.float64, 'out = lhs + rhs', lambda a, b: a + b),
        (dace.float64, np.float64, 'out = lhs * rhs', lambda a, b: a * b),
        (dace.int64, np.int64, 'out = lhs + rhs', lambda a, b: a + b),
        (dace.complex128, np.complex128, 'out = lhs + rhs', lambda a, b: a + b),
    ],
)
def test_array_elementwise_fixed_indices(dtype, np_dtype, code, reference):
    sdfg = _build_fixed_index_binary_sdfg('array_binary', dtype, 4, code)
    a = np.array([1, -2, 3, -4], dtype=np_dtype)
    b = np.array([4, 5, -6, 7], dtype=np_dtype)
    c = np.zeros(4, dtype=np_dtype)

    _run_sdfg(sdfg, A=a, B=b, C=c)

    _assert_same(c, reference(a, b))


@pytest.mark.parametrize(
    ('dtype', 'np_dtype', 'scale', 'code', 'reference'),
    [
        (dace.float32, np.float32, np.float32(2.0), 'out = inp * scale_value', lambda a, s: a * s),
        (dace.float64, np.float64, np.float64(-1.5), 'out = inp + scale_value', lambda a, s: a + s),
        (dace.int32, np.int32, np.int32(4), 'out = inp * scale_value', lambda a, s: a * s),
        (dace.int64, np.int64, np.int64(3), 'out = inp - scale_value', lambda a, s: a - s),
    ],
)
def test_array_scalar_fixed_indices(dtype, np_dtype, scale, code, reference):
    sdfg = _build_fixed_index_array_scalar_sdfg('array_scalar', dtype, 4, code)
    a = np.array([2, -1, 5, 7], dtype=np_dtype)
    b = np.zeros(4, dtype=np_dtype)

    _run_sdfg(sdfg, A=a, B=b, scale=scale)

    _assert_same(b, reference(a, scale))


@pytest.mark.parametrize(
    ('dtype', 'np_dtype', 'values', 'code', 'reference'),
    [
        (dace.float32, np.float32, np.array([1.5, -2.0, 4.0], dtype=np.float32), 'out = -inp', lambda a: -a),
        (dace.float64, np.float64, np.array([-3.5, 2.25, -9.0], dtype=np.float64), 'out = abs(inp)', lambda a: np.abs(a)),
        (dace.bool_, np.bool_, np.array([True, False, True], dtype=np.bool_), 'out = (not inp)', lambda a: np.logical_not(a)),
        (
            dace.complex128,
            np.complex128,
            np.array([1 + 2j, -3 + 1j, 0.5 - 4j], dtype=np.complex128),
            'out = inp.conjugate()',
            lambda a: np.conjugate(a),
        ),
    ],
)
def test_unary_fixed_indices(dtype, np_dtype, values, code, reference):
    sdfg = _build_fixed_index_unary_sdfg('array_unary', dtype, len(values), code)
    result = np.zeros_like(values)

    _run_sdfg(sdfg, A=values, B=result)

    _assert_same(result, reference(values))


@pytest.mark.parametrize(('dtype', 'np_dtype'), [(dace.float32, np.float32), (dace.float64, np.float64), (dace.int64, np.int64)])
def test_symbolic_slice_copy_1d(dtype, np_dtype):
    n_symbol = dace.symbol('N')
    sdfg = _new_sdfg('symbolic_slice_1d')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [n_symbol], dtype)
    sdfg.add_array('B', [n_symbol - 2], dtype)

    state = sdfg.add_state(is_start_block=True)
    a_read = state.add_read('A')
    b_write = state.add_write('B')
    state.add_edge(a_read, None, b_write, None, dace.Memlet('A[1:N-1] -> [0:N-2]'))

    n = 7
    a = np.arange(n, dtype=np_dtype)
    b = np.zeros(n - 2, dtype=np_dtype)

    _run_sdfg(sdfg, A=a, B=b, N=n)

    _assert_same(b, a[1:-1])


@pytest.mark.parametrize(('dtype', 'np_dtype'), [(dace.float64, np.float64), (dace.int64, np.int64)])
def test_symbolic_slice_copy_2d(dtype, np_dtype):
    n_symbol = dace.symbol('N')
    m_symbol = dace.symbol('M')
    sdfg = _new_sdfg('symbolic_slice_2d')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_symbol('M', dace.int64)
    sdfg.add_array('A', [n_symbol, m_symbol], dtype)
    sdfg.add_array('B', [n_symbol, m_symbol], dtype)

    state = sdfg.add_state(is_start_block=True)
    a_read = state.add_read('A')
    b_write = state.add_write('B')
    state.add_edge(a_read, None, b_write, None, dace.Memlet('A[0:N, 0:M] -> [0:N, 0:M]'))

    n = 3
    m = 4
    a = np.arange(n * m, dtype=np_dtype).reshape(n, m)
    b = np.zeros_like(a)

    _run_sdfg(sdfg, A=a, B=b, N=n, M=m)

    _assert_same(b, a)


def test_loop_region_symbolic_fill():
    n_symbol = dace.symbol('N')
    sdfg = _new_sdfg('loop_fill')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [n_symbol], dace.int64)

    loop = LoopRegion('loop', condition_expr='i < N', loop_var='i', initialize_expr='i = 0', update_expr='i = i + 1', sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('body', is_start_block=True)
    a_write = body.add_write('A')
    tasklet = body.add_tasklet('fill', {}, {'out'}, 'out = i * 2')
    body.add_edge(tasklet, 'out', a_write, None, dace.Memlet('A[i]'))

    n = 6
    a = np.zeros(n, dtype=np.int64)
    _run_sdfg(sdfg, A=a, N=n)

    np.testing.assert_array_equal(a, np.arange(n, dtype=np.int64) * 2)


def test_loop_region_symbolic_offset_fill_with_constant():
    n_symbol = dace.symbol('N')
    sdfg = _new_sdfg('loop_fill_offset')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_constant('SHIFT', 5)
    sdfg.add_array('A', [n_symbol], dace.int64)

    loop = LoopRegion('loop', condition_expr='i < N', loop_var='i', initialize_expr='i = 0', update_expr='i = i + 1', sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('body', is_start_block=True)
    a_write = body.add_write('A')
    tasklet = body.add_tasklet('fill', {}, {'out'}, 'out = i + SHIFT')
    body.add_edge(tasklet, 'out', a_write, None, dace.Memlet('A[i]'))

    n = 5
    a = np.zeros(n, dtype=np.int64)
    _run_sdfg(sdfg, A=a, N=n)

    np.testing.assert_array_equal(a, np.arange(n, dtype=np.int64) + 5)


def test_interstate_variable_assignment_used_in_tasklet():
    sdfg = _new_sdfg('interstate_var')
    sdfg.add_array('A', [1], dace.int64)

    init = sdfg.add_state('init', is_start_block=True)
    compute = sdfg.add_state('compute')
    sdfg.add_edge(init, compute, InterstateEdge(assignments={'k': '5'}))

    a_write = compute.add_write('A')
    tasklet = compute.add_tasklet('compute', {}, {'out'}, 'out = k + 2')
    compute.add_edge(tasklet, 'out', a_write, None, dace.Memlet('A[0]'))

    a = np.zeros(1, dtype=np.int64)
    _run_sdfg(sdfg, A=a)

    np.testing.assert_array_equal(a, np.array([7], dtype=np.int64))


def test_scalar_transient_pipeline():
    sdfg = _new_sdfg('scalar_transient_pipeline')
    sdfg.add_array('A', [1], dace.int64)
    sdfg.add_array('B', [1], dace.int64)
    sdfg.add_scalar('tmp', dace.int64, transient=True)

    state1 = sdfg.add_state('state1', is_start_block=True)
    state2 = sdfg.add_state('state2')
    sdfg.add_edge(state1, state2, InterstateEdge())

    a_read = state1.add_read('A')
    tmp_write = state1.add_write('tmp')
    add_tasklet = state1.add_tasklet('add', {'inp'}, {'out'}, 'out = inp + 5')
    state1.add_edge(a_read, None, add_tasklet, 'inp', dace.Memlet('A[0]'))
    state1.add_edge(add_tasklet, 'out', tmp_write, None, dace.Memlet('tmp'))

    tmp_read = state2.add_read('tmp')
    b_write = state2.add_write('B')
    mul_tasklet = state2.add_tasklet('mul', {'inp'}, {'out'}, 'out = inp * 2')
    state2.add_edge(tmp_read, None, mul_tasklet, 'inp', dace.Memlet('tmp'))
    state2.add_edge(mul_tasklet, 'out', b_write, None, dace.Memlet('B[0]'))

    a = np.array([3], dtype=np.int64)
    b = np.zeros(1, dtype=np.int64)
    _run_sdfg(sdfg, A=a, B=b)

    np.testing.assert_array_equal(b, np.array([16], dtype=np.int64))


def test_three_state_copy_pipeline():
    n_symbol = dace.symbol('N')
    sdfg = _new_sdfg('three_state_copy_pipeline')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [n_symbol], dace.float64)
    sdfg.add_array('B', [n_symbol], dace.float64)
    sdfg.add_array('C', [n_symbol], dace.float64)

    state1 = sdfg.add_state('state1', is_start_block=True)
    state2 = sdfg.add_state('state2')
    state3 = sdfg.add_state('state3')
    sdfg.add_edge(state1, state2, InterstateEdge())
    sdfg.add_edge(state2, state3, InterstateEdge())

    state1.add_edge(state1.add_read('A'), None, state1.add_write('B'), None, dace.Memlet('A[0:N] -> [0:N]'))
    state2.add_edge(state2.add_read('B'), None, state2.add_write('C'), None, dace.Memlet('B[0:N] -> [0:N]'))

    n = 8
    a = np.linspace(-1.0, 2.0, n)
    b = np.zeros(n, dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)
    _run_sdfg(sdfg, A=a, B=b, C=c, N=n)

    np.testing.assert_allclose(b, a)
    np.testing.assert_allclose(c, a)


def test_multiple_output_arrays():
    sdfg = _new_sdfg('multiple_outputs')
    sdfg.add_array('A', [2], dace.int64)
    sdfg.add_array('B', [2], dace.int64)

    state = sdfg.add_state(is_start_block=True)
    a_write = state.add_write('A')
    b_write = state.add_write('B')
    tasklet_a = state.add_tasklet('ta', {}, {'out'}, 'out = 3')
    tasklet_b = state.add_tasklet('tb', {}, {'out'}, 'out = 7')
    state.add_edge(tasklet_a, 'out', a_write, None, dace.Memlet('A[0]'))
    state.add_edge(tasklet_b, 'out', b_write, None, dace.Memlet('B[1]'))

    a = np.zeros(2, dtype=np.int64)
    b = np.zeros(2, dtype=np.int64)
    _run_sdfg(sdfg, A=a, B=b)

    np.testing.assert_array_equal(a, np.array([3, 0], dtype=np.int64))
    np.testing.assert_array_equal(b, np.array([0, 7], dtype=np.int64))


def test_in_place_fixed_index_update():
    sdfg = _new_sdfg('in_place_update')
    sdfg.add_array('A', [3], dace.int64)

    state = sdfg.add_state(is_start_block=True)
    a_read = state.add_read('A')
    a_write = state.add_write('A')
    for idx, delta in enumerate((2, -1, 5)):
        tasklet = state.add_tasklet(f'update_{idx}', {'inp'}, {'out'}, f'out = inp + {delta}')
        state.add_edge(a_read, None, tasklet, 'inp', dace.Memlet(f'A[{idx}]'))
        state.add_edge(tasklet, 'out', a_write, None, dace.Memlet(f'A[{idx}]'))

    a = np.array([10, 20, 30], dtype=np.int64)
    _run_sdfg(sdfg, A=a)

    np.testing.assert_array_equal(a, np.array([12, 19, 35], dtype=np.int64))


def test_bool_copy_and_not():
    sdfg = _new_sdfg('bool_copy_not')
    sdfg.add_array('A', [4], dace.bool_)
    sdfg.add_array('B', [4], dace.bool_)

    state = sdfg.add_state(is_start_block=True)
    a_read = state.add_read('A')
    b_write = state.add_write('B')
    state.add_edge(a_read, None, b_write, None, dace.Memlet('A[0:4] -> [0:4]'))

    tasklet = state.add_tasklet('flip', {'inp'}, {'out'}, 'out = (not inp)')
    state.add_edge(a_read, None, tasklet, 'inp', dace.Memlet('A[2]'))
    state.add_edge(tasklet, 'out', b_write, None, dace.Memlet('B[2]'))

    a = np.array([True, False, True, False], dtype=np.bool_)
    b = np.zeros(4, dtype=np.bool_)
    _run_sdfg(sdfg, A=a, B=b)

    expected = a.copy()
    expected[2] = not expected[2]
    np.testing.assert_array_equal(b, expected)


def test_complex_fixed_index_computation():
    sdfg = _new_sdfg('complex_fixed_index')
    sdfg.add_array('A', [2], dace.complex128)
    sdfg.add_array('B', [2], dace.complex128)

    state = sdfg.add_state(is_start_block=True)
    a_read = state.add_read('A')
    b_write = state.add_write('B')
    tasklet = state.add_tasklet('complex_op', {'inp'}, {'out'}, 'out = inp.conjugate() + (1 - 2j)')
    state.add_edge(a_read, None, tasklet, 'inp', dace.Memlet('A[1]'))
    state.add_edge(tasklet, 'out', b_write, None, dace.Memlet('B[0]'))

    a = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
    b = np.zeros(2, dtype=np.complex128)
    _run_sdfg(sdfg, A=a, B=b)

    expected = np.zeros(2, dtype=np.complex128)
    expected[0] = np.conjugate(a[1]) + (1 - 2j)
    np.testing.assert_allclose(b, expected)


@pytest.mark.xfail(strict=True, reason='Python backend dispatches MapExit nodes and raises NotImplementedError for mapped computations.')
def test_map_elementwise_add_xfail():
    sdfg = _new_sdfg('map_elementwise')
    sdfg.add_array('A', [4], dace.float64)
    sdfg.add_array('B', [4], dace.float64)
    sdfg.add_array('C', [4], dace.float64)

    state = sdfg.add_state(is_start_block=True)
    map_entry, map_exit = state.add_map('m', {'i': '0:4'}, schedule=ScheduleType.Sequential)
    tasklet = state.add_tasklet('add', {'lhs', 'rhs'}, {'out'}, 'out = lhs + rhs')
    state.add_memlet_path(state.add_read('A'), map_entry, tasklet, dst_conn='lhs', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(state.add_read('B'), map_entry, tasklet, dst_conn='rhs', memlet=dace.Memlet('B[i]'))
    state.add_memlet_path(tasklet, map_exit, state.add_write('C'), src_conn='out', memlet=dace.Memlet('C[i]'))

    a = np.arange(4, dtype=np.float64)
    b = np.arange(4, dtype=np.float64)
    c = np.zeros(4, dtype=np.float64)
    _run_sdfg(sdfg, A=a, B=b, C=c)


@pytest.mark.xfail(strict=True, reason='Python backend reduction or WCR code generation currently fails for end-to-end execution.')
def test_sum_reduction_xfail():
    sdfg = _new_sdfg('reduction_sum')
    sdfg.add_array('A', [4], dace.float64)
    sdfg.add_array('out', [1], dace.float64)

    state = sdfg.add_state(is_start_block=True)
    init = state.add_tasklet('init', {}, {'out'}, 'out = 0.0')
    state.add_edge(init, 'out', state.add_write('out'), None, dace.Memlet('out[0]'))

    map_entry, map_exit = state.add_map('m', {'i': '0:4'}, schedule=ScheduleType.Sequential)
    tasklet = state.add_tasklet('accumulate', {'inp'}, {'out'}, 'out = inp')
    state.add_memlet_path(state.add_read('A'), map_entry, tasklet, dst_conn='inp', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(
        tasklet,
        map_exit,
        state.add_write('out'),
        src_conn='out',
        memlet=dace.Memlet('out[0]', wcr='lambda x, y: x + y'),
    )

    a = np.arange(4, dtype=np.float64)
    out = np.zeros(1, dtype=np.float64)
    _run_sdfg(sdfg, A=a, out=out)


@pytest.mark.xfail(strict=True, reason='Python backend allocates transient arrays with numpy.zeros(...) but the generated function lacks import numpy.')
def test_transient_array_intermediate_xfail():
    n_symbol = dace.symbol('N')
    sdfg = _new_sdfg('transient_array_pipeline')
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [n_symbol], dace.float64)
    sdfg.add_array('B', [n_symbol], dace.float64)
    sdfg.add_transient('tmp', [n_symbol], dace.float64)

    state1 = sdfg.add_state('state1', is_start_block=True)
    state2 = sdfg.add_state('state2')
    sdfg.add_edge(state1, state2, InterstateEdge())
    state1.add_edge(state1.add_read('A'), None, state1.add_access('tmp'), None, dace.Memlet('A[0:N] -> [0:N]'))
    state2.add_edge(state2.add_read('tmp'), None, state2.add_write('B'), None, dace.Memlet('tmp[0:N] -> [0:N]'))

    n = 5
    a = np.arange(n, dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    _run_sdfg(sdfg, A=a, B=b, N=n)


@pytest.mark.xfail(strict=True, reason='Python backend has no code generator for NestedSDFG nodes.')
def test_nested_sdfg_xfail():
    outer = _new_sdfg('outer_nested')
    outer.add_array('A', [1], dace.float64)
    outer.add_array('B', [1], dace.float64)

    inner = SDFG('inner_nested')
    inner.add_array('X', [1], dace.float64)
    inner.add_array('Y', [1], dace.float64)
    inner_state = inner.add_state(is_start_block=True)
    tasklet = inner_state.add_tasklet('copy', {'inp'}, {'out'}, 'out = inp + 1')
    inner_state.add_edge(inner_state.add_read('X'), None, tasklet, 'inp', dace.Memlet('X[0]'))
    inner_state.add_edge(tasklet, 'out', inner_state.add_write('Y'), None, dace.Memlet('Y[0]'))

    state = outer.add_state(is_start_block=True)
    nested = state.add_nested_sdfg(inner, {'X'}, {'Y'})
    state.add_edge(state.add_read('A'), None, nested, 'X', dace.Memlet('A[0]'))
    state.add_edge(nested, 'Y', state.add_write('B'), None, dace.Memlet('B[0]'))

    a = np.array([3.0], dtype=np.float64)
    b = np.zeros(1, dtype=np.float64)
    _run_sdfg(outer, A=a, B=b)


@pytest.mark.xfail(strict=True, reason='Python backend rebinds scalar outputs locally and does not write them back to the caller.')
def test_scalar_descriptor_output_xfail():
    sdfg = _new_sdfg('scalar_output')
    sdfg.add_scalar('lhs', dace.float64)
    sdfg.add_scalar('rhs', dace.float64)
    sdfg.add_scalar('out', dace.float64)

    state = sdfg.add_state(is_start_block=True)
    tasklet = state.add_tasklet('add', {'lhs_value', 'rhs_value'}, {'out_value'}, 'out_value = lhs_value + rhs_value')
    state.add_edge(state.add_read('lhs'), None, tasklet, 'lhs_value', dace.Memlet('lhs'))
    state.add_edge(state.add_read('rhs'), None, tasklet, 'rhs_value', dace.Memlet('rhs'))
    state.add_edge(tasklet, 'out_value', state.add_write('out'), None, dace.Memlet('out'))

    out = np.zeros(1, dtype=np.float64)
    _run_sdfg(sdfg, lhs=np.float64(2.0), rhs=np.float64(4.5), out=out)
    np.testing.assert_allclose(out, np.array([6.5], dtype=np.float64))


@pytest.mark.xfail(strict=True, reason='Python backend does not execute SDFG-level global Python code in the generated runtime namespace.')
def test_global_code_xfail():
    sdfg = _new_sdfg('global_code')
    sdfg.add_array('A', [1], dace.int64)
    sdfg.global_code[None] = CodeBlock('GLOBAL_SENTINEL = 9')

    state = sdfg.add_state(is_start_block=True)
    tasklet = state.add_tasklet('write', {}, {'out'}, 'out = GLOBAL_SENTINEL')
    state.add_edge(tasklet, 'out', state.add_write('A'), None, dace.Memlet('A[0]'))

    a = np.zeros(1, dtype=np.int64)
    _run_sdfg(sdfg, A=a)