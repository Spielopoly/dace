# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Codegen-side tests for cuTile AOT support (GPU-free).

Covers the two ``cutile_target`` changes:

1. **Symbol dtype pinning**: runtime-defined names (interstate-assigned /
   loop variables, absent from ``sdfg.symbols``) are staged with an explicit
   ``dtype=numpy.<inferred>`` from the frame's type inference — deterministic
   launch-arg dtypes for JIT and AOT alike.
2. **AOT spec registry**: with ``compiler.cutile.aot_compile`` on (the
   default), generated frame code carries ``__dace_cutile_aot_specs`` mapping
   kernel name -> per-launch-arg spec ``(kind, dtype, ndim, stride_constant)``
   in exact launch-arg order; untypeable parameters raise ``CodegenError``.
   With the config off the generated code is identical minus the registry.

Only generated source is inspected; no kernels are launched.
"""
import ast

import pytest

import dace
from dace import data, dtypes
from dace.codegen.exceptions import CodegenError
from dace.dtypes import Language, ScheduleType, StorageType
from dace.memlet import Memlet

_AOT_ENV = 'DACE_compiler_cutile_aot_compile'
_REGISTRY = '__dace_cutile_aot_specs'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_specs(code: str) -> dict:
    """Parse the AOT spec registry entries out of generated frame code.

    :param code: The generated Python source.
    :returns: Mapping kernel name -> spec dict.
    """
    specs = {}
    for node in ast.walk(ast.parse(code)):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.targets[0].value, ast.Name) and node.targets[0].value.id == _REGISTRY):
            specs[ast.literal_eval(node.targets[0].slice)] = ast.literal_eval(node.value)
    return specs


def _kernel_params(code: str, kernel_name: str) -> list:
    """Return the parameter names of the generated kernel function.

    :param code: The generated Python source.
    :param kernel_name: The ``__dace_cutile_...`` kernel name.
    :returns: List of parameter names in definition order.
    """
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.FunctionDef) and node.name == kernel_name:
            return [a.arg for a in node.args.args]
    raise AssertionError(f'kernel {kernel_name} not found in generated code')


def _is_registry_stmt(node: ast.stmt) -> bool:
    """Whether a module-level statement initializes or fills the registry."""
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return False
    target = node.targets[0]
    if isinstance(target, ast.Name):
        return target.id == _REGISTRY
    return (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == _REGISTRY)


def _normalized_dump(code: str, drop_registry: bool = False) -> tuple:
    """AST dump with module-level imports order-normalized (their emission
    order depends on set iteration over used targets — pre-existing
    nondeterminism, not config-dependent) and, optionally, registry
    statements removed.

    :param code: The generated Python source.
    :param drop_registry: Whether to strip registry statements first.
    :returns: ``(sorted import dumps, dump of the remaining module)``.
    """
    tree = ast.parse(code)
    if drop_registry:
        tree.body = [n for n in tree.body if not _is_registry_stmt(n)]
    imports = sorted(ast.dump(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)))
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    return imports, ast.dump(tree)


def _runtime_defined_symbol_sdfg(name: str, assign_expr: str, dtype, tasklet_code: str) -> dace.SDFG:
    """``y = f(x, c_rt)`` with ``c_rt`` a RUNTIME-DEFINED name (assigned on an
    interstate edge, not declared in ``sdfg.symbols``)."""
    n, tw = 64, 32
    sdfg = dace.SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array('x', [n], dtype, storage=StorageType.GPU_Global)
    sdfg.add_array('y', [n], dtype, storage=StorageType.GPU_Global)
    sdfg.add_array('_tx', [tw], dtype, storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array('_ty', [tw], dtype, storage=StorageType.CuTile_Tile, transient=True)
    init = sdfg.add_state('init')
    state = sdfg.add_state('main')
    sdfg.add_edge(init, state, dace.InterstateEdge(assignments={'c_rt': assign_expr}))
    me, mx = state.add_map('cutile_map', {'ti': f'0:{n}:{tw}'}, schedule=ScheduleType.CuTile)
    tk = state.add_tasklet('t', {'inp'}, {'out'}, tasklet_code, language=Language.Python)
    tx = state.add_access('_tx')
    ty = state.add_access('_ty')
    state.add_memlet_path(state.add_read('x'), me, tx, memlet=Memlet(data='x', subset=f'0:{n}'))
    state.add_edge(tx, None, tk, 'inp', Memlet(data='_tx', subset=f'0:{tw}'))
    state.add_edge(tk, 'out', ty, None, Memlet(data='_ty', subset=f'0:{tw}'))
    state.add_memlet_path(ty, mx, state.add_write('y'), memlet=Memlet(data='y', subset=f'0:{n}'))
    sdfg.fill_scope_connectors()
    assert 'c_rt' not in sdfg.symbols
    return sdfg


def _rich_sdfg(name: str) -> dace.SDFG:
    """A cuTile map exercising every spec kind: 2-D arrays with a symbolic
    outer stride, an input Scalar, declared int64/float64/bool symbols, and a
    runtime-defined name."""
    N = dace.symbol('N', dtype=dace.int64)
    M = dace.symbol('M', dtype=dace.int64)
    sdfg = dace.SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_symbol('M', dace.int64)
    sdfg.add_symbol('alpha', dace.float64)
    sdfg.add_symbol('flag', dace.bool)
    sdfg.add_array('A', [N, M], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array('B', [N, M], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_scalar('s_in', dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array('_ta', [8, 32], dace.float64, storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array('_tb', [8, 32], dace.float64, storage=StorageType.CuTile_Tile, transient=True)
    init = sdfg.add_state('init')
    state = sdfg.add_state('main')
    sdfg.add_edge(init, state, dace.InterstateEdge(assignments={'c_rt': '1.5'}))
    me, mx = state.add_map('cutile_map', {'ti': '0:N:8', 'tj': '0:32:32'}, schedule=ScheduleType.CuTile)
    tk = state.add_tasklet('t', {'inp', 'sv'}, {'out'},
                           'out = inp * alpha + sv + c_rt if flag else inp',
                           language=Language.Python)
    ta = state.add_access('_ta')
    tb = state.add_access('_tb')
    state.add_memlet_path(state.add_read('A'), me, ta, memlet=Memlet(data='A', subset='ti:ti+8, tj:tj+32'))
    state.add_memlet_path(state.add_read('s_in'), me, tk, dst_conn='sv', memlet=Memlet('s_in[0]'))
    state.add_edge(ta, None, tk, 'inp', Memlet(data='_ta', subset='0:8, 0:32'))
    state.add_edge(tk, 'out', tb, None, Memlet(data='_tb', subset='0:8, 0:32'))
    state.add_memlet_path(tb, mx, state.add_write('B'), memlet=Memlet(data='B', subset='ti:ti+8, tj:tj+32'))
    sdfg.fill_scope_connectors()
    return sdfg


# ---------------------------------------------------------------------------
# 1: runtime-defined symbol dtype pinning (unconditional, also with AOT off)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'assign_expr, dtype, expected',
    [
        ('1.0 + 2.0**(-40)', dace.float64, 'float64'),
        ('7', dace.int64, 'int64'),
        # Large positive literals infer uint64 (framecode inference); the pin
        # normalizes unsigned to same-width signed (bit-exact < 2**63) so the
        # kernel never mixes int64/uint64 -- an unsupported cuda.tile
        # promotion. Host values are Python ints (signed) anyway.
        (str(2**40 + 12345), dace.int64, 'int64'),
    ],
    ids=['float64', 'int64', 'big_literal_int64'])
def test_runtime_defined_symbol_dtype_pinned(monkeypatch, assign_expr, dtype, expected):
    """The staged launch arg carries the frame-inferred dtype explicitly."""
    monkeypatch.setenv(_AOT_ENV, '0')  # pinning must not depend on AOT
    sdfg = _runtime_defined_symbol_sdfg(f'rt_pin_{expected}', assign_expr, dtype, 'out = inp + c_rt')
    code = sdfg.generate_code()[0].code.replace(' ', '').replace('\n', '')
    assert f'cupy.asarray(c_rt,dtype=numpy.{expected}).reshape(1)' in code
    assert 'cupy.asarray(c_rt).reshape(1)' not in code


def test_runtime_defined_bool_symbol_dtype_pinned(monkeypatch):
    """A runtime-defined bool rides the device path with ``numpy.bool_``."""
    monkeypatch.setenv(_AOT_ENV, '0')
    sdfg = _runtime_defined_symbol_sdfg('rt_pin_bool', '1 > 0', dace.float64, 'out = inp if c_rt else -inp')
    code = sdfg.generate_code()[0].code.replace(' ', '').replace('\n', '')
    assert 'cupy.asarray(c_rt,dtype=numpy.bool_).reshape(1)' in code


def test_runtime_defined_true_division_pinned_as_float(monkeypatch):
    """Host true division and the staged kernel argument must both stay floating point."""
    monkeypatch.setenv(_AOT_ENV, '0')
    sdfg = _runtime_defined_symbol_sdfg('rt_pin_div', 'N / 2', dace.float64, 'out = inp + c_rt')
    sdfg.add_symbol('N', dace.int64)
    code = sdfg.generate_code()[0].code.replace(' ', '').replace('\n', '')
    assert 'cupy.asarray(c_rt,dtype=numpy.float64).reshape(1)' in code


# ---------------------------------------------------------------------------
# 2: spec registry emission (config on)
# ---------------------------------------------------------------------------


def test_registry_schema_and_order(monkeypatch):
    """One registry entry per kernel; params in exact launch-arg order with
    the plan's ``(kind, dtype, ndim, stride_constant)`` schema."""
    monkeypatch.setenv(_AOT_ENV, '1')
    sdfg = _rich_sdfg('aot_schema')
    code = sdfg.generate_code()[0].code
    specs = _extract_specs(code)
    assert len(specs) == 1
    kernel_name, spec = next(iter(specs.items()))
    assert kernel_name.startswith('__dace_cutile_aot_schema')
    params = spec['params']
    expected_by_name = {
        'A': ('array', 'float64', 2, None),  # runtime-derived layout
        'B': ('array', 'float64', 2, None),
        's_in': ('array', 'float64', 1, (1, )),  # device-staged input Scalar
        'N': ('array', 'int64', 1, (1, )),  # declared int64 symbol
        'alpha': ('array', 'float64', 1, (1, )),  # declared float64 symbol
        'c_rt': ('array', 'float64', 1, (1, )),  # runtime-defined, inferred f64
        'flag': ('scalar', 'bool', 0, None),  # declared bool symbol, by value
    }
    kparams = _kernel_params(code, kernel_name)
    assert sorted(kparams) == sorted(expected_by_name)
    # Spec order == kernel parameter order == launch-arg order.
    assert [tuple(p) if isinstance(p, (list, tuple)) else p for p in params] == \
        [expected_by_name[p] for p in kparams]
    # Arrays (deduped input+output) come before the free symbols.
    n_arrays = 3  # A, s_in, B
    assert set(kparams[:n_arrays]) == {'A', 'B', 's_in'}
    assert kparams[n_arrays:] == sorted(['N', 'alpha', 'c_rt', 'flag'])


def test_registry_default_on(monkeypatch):
    """Without any config override the registry is emitted (default on)."""
    monkeypatch.delenv(_AOT_ENV, raising=False)
    sdfg = _runtime_defined_symbol_sdfg('aot_default_on', '1.5', dace.float64, 'out = inp * c_rt')
    code = sdfg.generate_code()[0].code
    assert _REGISTRY in code
    assert len(_extract_specs(code)) == 1


def test_registry_literals_reconstructible(monkeypatch):
    """The emitted registry is plain literals (frozen interface for
    cutile_aot): every entry survives ``ast.literal_eval`` round-tripping."""
    monkeypatch.setenv(_AOT_ENV, '1')
    sdfg = _rich_sdfg('aot_literals')
    specs = _extract_specs(sdfg.generate_code()[0].code)
    for spec in specs.values():
        assert set(spec.keys()) == {'params'}
        for kind, dtype_name, ndim, stride in spec['params']:
            assert kind in ('array', 'scalar')
            assert isinstance(dtype_name, str) and isinstance(ndim, int)
            assert stride is None or isinstance(stride, tuple)


# ---------------------------------------------------------------------------
# 3: config off -> no registry, otherwise identical code
# ---------------------------------------------------------------------------


def test_config_off_no_registry_and_identical_code(monkeypatch):
    """Config off: no registry anywhere, and the module is identical to the
    config-on module minus the registry statements."""
    monkeypatch.setenv(_AOT_ENV, '1')
    code_on = _rich_sdfg('aot_off_cmp').generate_code()[0].code
    monkeypatch.setenv(_AOT_ENV, '0')
    code_off = _rich_sdfg('aot_off_cmp').generate_code()[0].code
    assert _REGISTRY in code_on
    assert _REGISTRY not in code_off
    assert _normalized_dump(code_on, drop_registry=True) == _normalized_dump(code_off)


# ---------------------------------------------------------------------------
# 4: untypeable parameters raise (fail loudly)
# ---------------------------------------------------------------------------


def test_output_scalar_supported_with_aot_on(monkeypatch):
    """A kernel-written numeric Scalar uses its one-element device-array convention."""
    monkeypatch.setenv(_AOT_ENV, '1')
    sdfg = dace.SDFG('aot_out_scalar')
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_scalar('s', dace.float64, storage=StorageType.GPU_Global)
    state = sdfg.add_state('main')
    me, mx = state.add_map('cutile_map', {'ti': '0:1'}, schedule=ScheduleType.CuTile)
    tk = state.add_tasklet('t', {}, {'out'}, 'out = 1.0', language=Language.Python)
    state.add_nedge(me, tk, dace.Memlet())
    state.add_memlet_path(tk, mx, state.add_write('s'), src_conn='out', memlet=Memlet('s[0]'))
    sdfg.fill_scope_connectors()
    specs = _extract_specs(sdfg.generate_code()[0].code)
    assert any(("array", "float64", 1, (1, )) in spec["params"] for spec in specs.values())


def test_build_aot_spec_unit_cases():
    """Direct unit coverage of the spec builder's raise paths and entries."""
    from dace.codegen.py.cutile_target import _build_aot_spec
    sdfg = dace.SDFG('aot_spec_unit')
    N = dace.symbol('N', dtype=dace.int64)
    sdfg.add_symbol('N', dace.int64)
    sdfg.add_symbol('flag', dace.bool)
    sdfg.add_array('A', [N, 64], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_scalar('s', dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_scalar('bs', dace.bool, storage=StorageType.GPU_Global)
    sdfg.add_datadesc('obj', data.Structure({'field': data.Array(dace.float32, [4])}, name='Obj'))

    # External array strides are runtime-derived; the input Scalar and device symbol become
    # 1-element arrays; bool scalar/symbol are by-value scalar entries.
    spec = _build_aot_spec(sdfg, 'k', ['A', 's', 'bs'], [], ['N', 'flag'], {'N': 'int64'})
    assert spec == [
        ('array', 'float64', 2, None),
        ('array', 'float64', 1, (1, )),
        ('scalar', 'bool', 0, None),
        ('array', 'int64', 1, (1, )),
        ('scalar', 'bool', 0, None),
    ]

    # Dotted structure members resolve to their actual descriptor.
    assert _build_aot_spec(sdfg, 'k', ['obj.field'], [], [], {}) == [
        ('array', 'float32', 1, None),
    ]

    # Kernel-written numeric Scalars use the same one-element array convention.
    assert _build_aot_spec(sdfg, 'k', ['s'], ['s'], [], {}) == [
        ('array', 'float64', 1, (1, )),
    ]
    # Complex symbols use the same device-staged array convention.
    assert _build_aot_spec(sdfg, 'k', [], [], ['z'], {'z': 'complex128'}) == [
        ('array', 'complex128', 1, (1, )),
    ]
    # Symbol that is neither device-staged nor a declared bool -> raise.
    with pytest.raises(CodegenError, match='aot_compile'):
        _build_aot_spec(sdfg, 'k', [], [], ['w'], {})


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
