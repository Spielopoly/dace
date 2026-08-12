# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression tests for cuTile ``generate_scope`` / kernel-parameter codegen.

Covers the code-review findings on the scope generator:

1. **Bound map params never become launch args**: a kernel map parameter that
   collides with a host loop variable or an interstate-assignment key must not
   pass the ``runtime_defined`` filter in ``_collect_free_symbols`` (it would
   become an undefined-at-callsite launch argument -> ``NameError``).
2. **Mutable scalar storage**: it must already be materialized as a one-element
   device array; an unlowered ``Scalar`` fails clearly.
3. **Numeric symbols**: declared and inferred host values receive exact
   ``ScalarConstraint`` objects and matching fixed-width host packing.
4. **Launch guard**: non-positive grid dimensions return before lookup and
   launch from the dedicated Cython helper.
"""
import json

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.dtypes import Language, ScheduleType, StorageType
from dace.memlet import Memlet
from dace.sdfg import nodes

# ---------------------------------------------------------------------------
# 1: _collect_free_symbols excludes scope-bound map parameters
# ---------------------------------------------------------------------------


def _scope_sdfg_with_colliding_param(param: str):
    """A CuTile map whose parameter collides with an interstate-assigned name.

    :param param: The map-parameter name (also used as an interstate
        assignment key on a later edge).
    :returns: ``(sdfg, state, map_entry)``.
    """
    N = dace.symbol('N', dtype=dace.int64)
    sdfg = dace.SDFG(f'collide_{param}')
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_array('A', [N], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array('B', [N], dace.float64, storage=StorageType.GPU_Global)
    state = sdfg.add_state('main')
    state.add_mapped_tasklet('t', {param: '0:N:8'}, {'_a': Memlet(f'A[{param}]')},
                             '_b = _a + 1.0', {'_b': Memlet(f'B[{param}]')},
                             external_edges=True,
                             schedule=ScheduleType.CuTile)
    # A later interstate edge assigns the same name, making it
    # "runtime-defined" from _collect_free_symbols' point of view.
    tail = sdfg.add_state('tail')
    sdfg.add_edge(state, tail, dace.InterstateEdge(assignments={param: '0'}))
    entry = next(n for n in state.nodes() if isinstance(n, nodes.MapEntry))
    return sdfg, state, entry


def test_map_param_colliding_with_interstate_key_not_a_kernel_symbol():
    """The map param is scope-bound and must not appear in the free symbols."""
    from dace.codegen.py.cutile_target import _collect_free_symbols
    sdfg, state, entry = _scope_sdfg_with_colliding_param('ii')
    syms = _collect_free_symbols(entry, state.scope_subgraph(entry), sdfg)
    assert 'ii' not in syms
    assert 'N' in syms


def test_map_param_colliding_with_loop_variable_not_a_kernel_symbol():
    """Same collision through a LoopRegion loop variable."""
    from dace.codegen.py.cutile_target import _collect_free_symbols
    from dace.sdfg.state import LoopRegion
    sdfg, state, entry = _scope_sdfg_with_colliding_param('jj')
    loop = LoopRegion('loop', 'jj < 4', 'jj', 'jj = 0', 'jj = jj + 1')
    sdfg.add_node(loop)
    loop.add_state('body')
    syms = _collect_free_symbols(entry, state.scope_subgraph(entry), sdfg)
    assert 'jj' not in syms


# ---------------------------------------------------------------------------
# 2: numeric Scalar in input AND output -> NotImplementedError
# ---------------------------------------------------------------------------


def _inout_scalar_sdfg(dtype) -> dace.SDFG:
    """A CuTile map reading and writing the same non-transient Scalar."""
    sdfg = dace.SDFG(f'inout_scalar_{dtype.to_string()}')
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_scalar('s', dtype, storage=StorageType.GPU_Global)
    state = sdfg.add_state('main')
    me, mx = state.add_map('cutile_map', {'tile_i': '0:1'}, schedule=ScheduleType.CuTile)
    r = state.add_read('s')
    w = state.add_write('s')
    tk = state.add_tasklet('t', {'inp'}, {'out'}, 'out = inp + 1', language=Language.Python)
    state.add_memlet_path(r, me, tk, dst_conn='inp', memlet=Memlet('s[0]'))
    state.add_memlet_path(tk, mx, w, src_conn='out', memlet=Memlet('s[0]'))
    sdfg.fill_scope_connectors()
    return sdfg


def test_float_scalar_inout_raises_not_implemented():
    """Mutable float storage must be materialized before cuTile codegen."""
    sdfg = _inout_scalar_sdfg(dace.float64)
    with pytest.raises(NotImplementedError, match='materialize mutable scalar storage'):
        sdfg.generate_code()


def test_int_scalar_inout_raises_not_implemented():
    """Mutable integer storage must be materialized before cuTile codegen."""
    sdfg = _inout_scalar_sdfg(dace.int64)
    with pytest.raises(NotImplementedError, match='materialize mutable scalar storage'):
        sdfg.generate_code()


def test_bool_scalar_inout_raises_not_implemented():
    """Mutable Boolean storage must be materialized before cuTile codegen."""
    sdfg = _inout_scalar_sdfg(dace.bool)
    with pytest.raises(NotImplementedError, match='materialize mutable scalar storage'):
        sdfg.generate_code()


# ---------------------------------------------------------------------------
# 3 + 4: float64 symbols + launch guard
# ---------------------------------------------------------------------------


def _float_symbol_sdfg(name: str, n: int = 64, tile_w: int = 32) -> dace.SDFG:
    """``y = x * alpha_v`` with ``alpha_v`` a float64 SYMBOL, tiles built
    manually (the AccessNode-centric pattern used across these tests)."""
    sdfg = dace.SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_symbol('alpha_v', dace.float64)
    sdfg.add_array('x', [n], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array('y', [n], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array('_tx', [tile_w], dace.float64, storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array('_ty', [tile_w], dace.float64, storage=StorageType.CuTile_Tile, transient=True)
    state = sdfg.add_state('main')
    me, mx = state.add_map('cutile_map', {'tile_i': f'0:{n}:{tile_w}'}, schedule=ScheduleType.CuTile)
    xr = state.add_read('x')
    yw = state.add_write('y')
    tx = state.add_access('_tx')
    ty = state.add_access('_ty')
    tk = state.add_tasklet('scale', {'inp'}, {'out'}, 'out = inp * alpha_v', language=Language.Python)
    state.add_memlet_path(xr, me, tx, memlet=Memlet(data='x', subset=f'0:{n}'))
    state.add_edge(tx, None, tk, 'inp', Memlet(data='_tx', subset=f'0:{tile_w}'))
    state.add_edge(tk, 'out', ty, None, Memlet(data='_ty', subset=f'0:{tile_w}'))
    state.add_memlet_path(ty, mx, yw, memlet=Memlet(data='y', subset=f'0:{n}'))
    sdfg.fill_scope_connectors()
    return sdfg


def test_float64_symbol_uses_exact_exported_scalar():
    """The build signature and host helper agree on exact float64 packing."""
    sdfg = _float_symbol_sdfg('float_sym_structural')
    code_objects = sdfg.generate_code()
    host = next(code.code for code in code_objects if code.language == 'pyx').replace(' ', '')
    build = next(code.code for code in code_objects if code.target_type == 'cutile_build').replace(' ', '')
    assert 'compilation.ScalarConstraint(ct.float64)' in build
    assert 'numpy.float64(__dace_raw_' in host
    assert 'cupy.asarray(alpha_v' not in host
    assert 'ct.load(alpha_v' not in build


def test_launch_guard_skips_nonpositive_grid():
    """The helper returns before lookup and launch for nonpositive grids."""
    sdfg = _float_symbol_sdfg('launch_guard_structural')
    code = sdfg.generate_code()[0].code.replace(' ', '').replace('\n', '')
    assert 'ifmin(__dace_grid)<=0:return' in code
    assert code.index('ifmin(__dace_grid)<=0:return') < code.rindex('__dace_cutile_get_function')
    assert 'ct.launch' not in code


def _int_symbol_sdfg(name: str, n: int = 64, tile_w: int = 32) -> dace.SDFG:
    """``y = x + k_off`` with ``k_off`` an int64 SYMBOL (same manual-tile
    pattern as :func:`_float_symbol_sdfg`)."""
    sdfg = dace.SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_symbol('k_off', dace.int64)
    sdfg.add_array('x', [n], dace.int64, storage=StorageType.GPU_Global)
    sdfg.add_array('y', [n], dace.int64, storage=StorageType.GPU_Global)
    sdfg.add_array('_tx', [tile_w], dace.int64, storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array('_ty', [tile_w], dace.int64, storage=StorageType.CuTile_Tile, transient=True)
    state = sdfg.add_state('main')
    me, mx = state.add_map('cutile_map', {'tile_i': f'0:{n}:{tile_w}'}, schedule=ScheduleType.CuTile)
    xr = state.add_read('x')
    yw = state.add_write('y')
    tx = state.add_access('_tx')
    ty = state.add_access('_ty')
    tk = state.add_tasklet('shift', {'inp'}, {'out'}, 'out = inp + k_off', language=Language.Python)
    state.add_memlet_path(xr, me, tx, memlet=Memlet(data='x', subset=f'0:{n}'))
    state.add_edge(tx, None, tk, 'inp', Memlet(data='_tx', subset=f'0:{tile_w}'))
    state.add_edge(tk, 'out', ty, None, Memlet(data='_ty', subset=f'0:{tile_w}'))
    state.add_memlet_path(ty, mx, yw, memlet=Memlet(data='y', subset=f'0:{n}'))
    sdfg.fill_scope_connectors()
    return sdfg


def test_int64_symbol_uses_exact_exported_scalar():
    """The build signature and host helper agree on exact int64 packing."""
    sdfg = _int_symbol_sdfg('int_sym_structural')
    code_objects = sdfg.generate_code()
    host = next(code.code for code in code_objects if code.language == 'pyx').replace(' ', '')
    build = next(code.code for code in code_objects if code.target_type == 'cutile_build').replace(' ', '')
    assert 'compilation.ScalarConstraint(ct.int64)' in build
    assert 'numpy.int64(__dace_raw_' in host
    assert 'cupy.asarray(k_off' not in host
    assert 'ct.load(k_off' not in build


@pytest.mark.gpu
def test_int64_symbol_large_value_runtime():
    """End-to-end: an int64 symbol >= 2**31 (by-value would OverflowError)."""
    cupy = pytest.importorskip('cupy')
    n = 64
    sdfg = _int_symbol_sdfg('int_sym_runtime', n=n)
    k_off = 2**40 + 12345
    x_host = np.arange(n, dtype=np.int64)
    x = cupy.asarray(x_host)
    y = cupy.zeros(n, dtype=cupy.int64)
    sdfg.compile()(x=x, y=y, k_off=k_off)
    np.testing.assert_array_equal(cupy.asnumpy(y), x_host + k_off)


@pytest.mark.gpu
def test_float64_symbol_full_precision_runtime():
    """End-to-end: an f32-typed symbol would lose the 2**-40 component."""
    cupy = pytest.importorskip('cupy')
    n = 64
    sdfg = _float_symbol_sdfg('float_sym_runtime', n=n)
    alpha = 1.0 + 2.0**-40
    x_host = np.random.default_rng(11).random(n)
    x = cupy.asarray(x_host)
    y = cupy.zeros(n, dtype=cupy.float64)
    sdfg.compile()(x=x, y=y, alpha_v=alpha)
    ref = x_host * alpha
    # Tight tolerance: a float32-typed symbol argument shows ~1e-8 error.
    assert np.abs(cupy.asnumpy(y) - ref).max() < 1e-14


def _runtime_defined_symbol_sdfg(name: str,
                                 assign_expr: str,
                                 dtype,
                                 op: str,
                                 n: int = 64,
                                 tile_w: int = 32) -> dace.SDFG:
    """``y = x <op> c_rt`` with ``c_rt`` a RUNTIME-DEFINED name: assigned on an
    interstate edge and deliberately NOT declared in ``sdfg.symbols``."""
    sdfg = dace.SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array('x', [n], dtype, storage=StorageType.GPU_Global)
    sdfg.add_array('y', [n], dtype, storage=StorageType.GPU_Global)
    sdfg.add_array('_tx', [tile_w], dtype, storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array('_ty', [tile_w], dtype, storage=StorageType.CuTile_Tile, transient=True)
    init = sdfg.add_state('init')
    state = sdfg.add_state('main')
    sdfg.add_edge(init, state, dace.InterstateEdge(assignments={'c_rt': assign_expr}))
    me, mx = state.add_map('cutile_map', {'tile_i': f'0:{n}:{tile_w}'}, schedule=ScheduleType.CuTile)
    xr = state.add_read('x')
    yw = state.add_write('y')
    tx = state.add_access('_tx')
    ty = state.add_access('_ty')
    tk = state.add_tasklet('t', {'inp'}, {'out'}, f'out = inp {op} c_rt', language=Language.Python)
    state.add_memlet_path(xr, me, tx, memlet=Memlet(data='x', subset=f'0:{n}'))
    state.add_edge(tx, None, tk, 'inp', Memlet(data='_tx', subset=f'0:{tile_w}'))
    state.add_edge(tk, 'out', ty, None, Memlet(data='_ty', subset=f'0:{tile_w}'))
    state.add_memlet_path(ty, mx, yw, memlet=Memlet(data='y', subset=f'0:{n}'))
    sdfg.fill_scope_connectors()
    assert 'c_rt' not in sdfg.symbols
    return sdfg


def test_runtime_defined_symbol_type_is_inferred_for_exact_abi():
    """An interstate value gets one inferred fixed scalar ABI."""
    sdfg = _runtime_defined_symbol_sdfg('rt_sym_structural', '1.0 + 2.0**(-40)', dace.float64, '*')
    code_objects = sdfg.generate_code()
    host = next(code.code for code in code_objects if code.language == 'pyx').replace(' ', '')
    build = next(code.code for code in code_objects if code.target_type == 'cutile_build').replace(' ', '')
    assert 'compilation.ScalarConstraint(ct.float64)' in build
    assert 'numpy.float64(__dace_raw_' in host
    assert 'cupy.asarray(c_rt)' not in host
    assert 'ct.load(c_rt' not in build


def test_conflicting_runtime_defined_symbol_types_fail_during_compilation():
    """Branch assignments with incompatible scalar ABIs fail closed."""
    sdfg = _runtime_defined_symbol_sdfg('rt_conflicting_types', '1', dace.float64, '+')
    sdfg.add_symbol('choose_float', dace.bool_)
    init = sdfg.start_state
    main = sdfg.states()[1]
    for edge in list(sdfg.edges_between(init, main)):
        sdfg.remove_edge(edge)
    integer_path = sdfg.add_state('integer_path')
    float_path = sdfg.add_state('float_path')
    sdfg.add_edge(init, integer_path, dace.InterstateEdge(condition='not choose_float', assignments={'c_rt': '1'}))
    sdfg.add_edge(init, float_path,
                  dace.InterstateEdge(condition='choose_float', assignments={'c_rt': 'numpy.float64(1.5)'}))
    sdfg.add_edge(integer_path, main, dace.InterstateEdge())
    sdfg.add_edge(float_path, main, dace.InterstateEdge())

    with pytest.raises(TypeError, match="Conflicting cuTile ABI types for runtime value 'c_rt'"):
        sdfg.generate_code()


def test_runtime_defined_numpy_integer_keeps_numpy_semantics():
    """Explicit NumPy constructors retain their narrow scalar type."""
    sdfg = _runtime_defined_symbol_sdfg('rt_numpy_int8', 'numpy.int8(5)', dace.int8, '+')
    code_objects = sdfg.generate_code()
    host = next(code.code for code in code_objects if code.language == 'pyx').replace(' ', '')
    build = next(code.code for code in code_objects if code.target_type == 'cutile_build').replace(' ', '')
    assert 'compilation.ScalarConstraint(ct.int8)' in build
    assert 'numpy.int8(__dace_raw_' in host


@pytest.mark.gpu
def test_runtime_defined_float_symbol_runtime():
    """End-to-end: a runtime-defined float (``1 + 2**-40``, not f32-exact)
    survives the kernel exactly (by value it would round to float32)."""
    cupy = pytest.importorskip('cupy')
    n = 64
    sdfg = _runtime_defined_symbol_sdfg('rt_float_runtime', '1.0 + 2.0**(-40)', dace.float64, '*', n=n)
    x_host = np.random.default_rng(13).random(n)
    x = cupy.asarray(x_host)
    y = cupy.zeros(n, dtype=cupy.float64)
    sdfg.compile()(x=x, y=y)
    ref = x_host * (1.0 + 2.0**-40)
    np.testing.assert_array_equal(cupy.asnumpy(y), ref)


@pytest.mark.gpu
def test_runtime_defined_int_symbol_large_value_runtime():
    """End-to-end: a runtime-defined int >= 2**31 (by value cuda.tile types
    Python ints int32 -> OverflowError)."""
    cupy = pytest.importorskip('cupy')
    n = 64
    big = 2**40 + 12345
    sdfg = _runtime_defined_symbol_sdfg('rt_int_runtime', str(big), dace.int64, '+', n=n)
    x_host = np.arange(n, dtype=np.int64)
    x = cupy.asarray(x_host)
    y = cupy.zeros(n, dtype=cupy.int64)
    sdfg.compile()(x=x, y=y)
    np.testing.assert_array_equal(cupy.asnumpy(y), x_host + big)


# ---------------------------------------------------------------------------
# 5: scalar SDFG constants
# ---------------------------------------------------------------------------

_NON_F32_CONSTANT = np.float64(1.0 + 2.0**-40)


def _constant_sdfg(name: str, value: object = _NON_F32_CONSTANT) -> dace.SDFG:
    """Build the float64 symbol fixture with a specialized SDFG constant."""
    sdfg = _float_symbol_sdfg(name)
    sdfg.add_constant('alpha_v', float(value), dace.data.Scalar(dace.float64))
    return sdfg


def test_scalar_constant_is_typed_literal_and_changes_symbol_hash():
    """Specialized constants are absent from the ABI and enter both hashes."""
    first = _constant_sdfg('constant_hash', _NON_F32_CONSTANT)
    second = _constant_sdfg('constant_hash', np.float64(1.0 + 2.0**-39))
    first_object = next(code for code in first.generate_code() if code.target_type == 'cutile_build')
    second_object = next(code for code in second.generate_code() if code.target_type == 'cutile_build')
    first_symbols = set(json.loads(first_object.extra_compiler_kwargs['cutile_symbols']))
    second_symbols = set(json.loads(second_object.extra_compiler_kwargs['cutile_symbols']))

    assert first_symbols != second_symbols
    assert 'alpha_v = ct.bitcast(' in first_object.code
    assert '__dace_compile_time_constant constant_hash.alpha_v=' in first_object.code
    assert 'ScalarConstraint(ct.float64)' not in first_object.code

    from dace.codegen.py.cutile_target import _cutile_scalar_constant
    int8_literal = _cutile_scalar_constant('SMALL', dace.data.Scalar(dace.int8), -1)
    assert 'ct.uint8' in int8_literal and 'ct.int8' in int8_literal


def test_referenced_array_constant_fails_closed():
    """An array constant cannot silently become a kernel-module capture."""
    sdfg = _float_symbol_sdfg('unsupported_constant')
    sdfg.symbols.pop('alpha_v')
    sdfg.add_constant('alpha_v', np.asarray([1.0], dtype=np.float64))
    with pytest.raises(NotImplementedError, match=r"constant 'alpha_v' must be a scalar"):
        sdfg.generate_code()


@pytest.mark.gpu
def test_non_float32_exact_scalar_constant_runtime():
    """A specialized float64 constant retains its exact bit pattern."""
    cupy = pytest.importorskip('cupy')
    n = 64
    sdfg = _constant_sdfg('constant_runtime', _NON_F32_CONSTANT)
    x_host = np.random.default_rng(17).random(n)
    x = cupy.asarray(x_host)
    y = cupy.zeros(n, dtype=cupy.float64)
    sdfg.compile()(x=x, y=y)
    np.testing.assert_array_equal(cupy.asnumpy(y), x_host * _NON_F32_CONSTANT)


# ---------------------------------------------------------------------------
# 6: py_mod / int_floor in rendered element-index expressions
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
