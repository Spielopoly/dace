# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""cuTile AOT export and direct-launch integration tests."""
import base64
import os
import subprocess
import sys
import types
import zlib

import numpy as np
import pytest

import dace
from dace.codegen.py import cutile_aot
from dace.codegen.py.cutile_aot import (AOT_ABI_VERSION, AOTParam, CuTileAOTError, build_export_plan,
                                        check_prerequisites, resolve_arch)
from dace.config import set_temporary
from dace.transformation.passes.vectorization import VectorizeCuTile

ct = pytest.importorskip('cuda.tile')
compilation = pytest.importorskip('cuda.tile.compilation')

N = dace.symbol('N')
M = dace.symbol('M')
K = dace.symbol('K', dtype=dace.int64)
KU = dace.symbol('KU', dtype=dace.uint64)


@dace.program
def _vadd(x: dace.float64[N], y: dace.float64[N], z: dace.float64[N]):
    z[:] = x + y


@dace.program
def _vadd_f32(x: dace.float32[N], y: dace.float32[N], z: dace.float32[N]):
    z[:] = x + y


@dace.program
def _scale_add_2d(A: dace.float64[M, N], B: dace.float64[M, N]):
    B[:] = A * 2.0 + B


@dace.program
def _shift_i64(x: dace.int64[N], y: dace.int64[N]):
    y[:] = x + K


@dace.program
def _shift_u64(x: dace.uint64[N], y: dace.uint64[N]):
    y[:] = x + KU


@dace.program
def _two_kernels(x: dace.float64[N], y: dace.float64[N], z: dace.float64[M]):
    y[:] = x * 2.0
    z[:] = z + 1.0


def _lower(program, name: str, widths) -> dace.SDFG:
    sdfg = program.to_sdfg(simplify=False)
    sdfg.name = name
    VectorizeCuTile(widths=widths, use_gpu_storage=True).apply_pass(sdfg, {})
    return sdfg


class _FakeCUDAArray:

    dtype = np.dtype(np.float64)
    ndim = 2
    __cuda_array_interface__ = {}

    def __init__(self, shape, element_strides):
        self.shape = shape
        self.strides = tuple(stride * self.dtype.itemsize for stride in element_strides)


def _launcher_metadata(params):
    has_array = any(param.kind == 'array' for param in params)
    variants = [{'index_dtype': 'int32', 'symbol': 'kernel_i32'}]
    if has_array:
        variants.append({'index_dtype': 'int64', 'symbol': 'kernel_i64'})
    return {
        'kernel': {
            'abi_version': AOT_ABI_VERSION,
            'arch': 'sm_99',
            'cubin': base64.b85encode(zlib.compress(b'fake cubin')).decode('ascii'),
            'params': [param.to_metadata() for param in params],
            'variants': variants,
        }
    }


def _launcher_namespace(monkeypatch, params):
    records, modules = [], []
    context = [101]
    stream = object()

    class FakeFunction:

        def __init__(self, symbol):
            self.symbol = symbol

        def __call__(self, grid, block, args, stream=None):
            records.append((self.symbol, grid, block, args, stream, context[0]))

    class FakeModule:

        def __init__(self):
            self.loaded = None
            modules.append(self)

        def load(self, cubin):
            self.loaded = cubin

        def get_function(self, symbol):
            return FakeFunction(symbol)

    fake_cupy = types.SimpleNamespace(cuda=types.SimpleNamespace(
        Device=lambda: types.SimpleNamespace(id=7, compute_capability='99'),
        driver=types.SimpleNamespace(ctxGetCurrent=lambda: context[0]),
        function=types.SimpleNamespace(Module=FakeModule),
        get_current_stream=lambda: stream,
    ))
    monkeypatch.setitem(sys.modules, 'cupy', fake_cupy)
    namespace = {}
    exec(cutile_aot._render_module(_launcher_metadata(params)), namespace)
    return namespace, records, modules, context, stream


def test_public_v2_signature_plan():
    param = AOTParam('array', 'float64', 2)
    spec = {'abi_version': AOT_ABI_VERSION, 'params': [param]}
    signatures, variants = build_export_plan(spec, 'kernel')
    assert [v['index_dtype'] for v in variants] == ['int32', 'int64']
    assert {sig.calling_convention.version for sig in signatures} == {2}
    assert [sig.parameters[0].index_dtype for sig in signatures] == [ct.int32, ct.int64]
    assert all(sig.symbol.startswith('kernel') for sig in signatures)
    assert cutile_aot._validated_params(spec) == (param, )
    assert param.to_metadata() == {
        'kind': 'array',
        'dtype': 'float64',
        'ndim': 2,
        'strides': None,
    }


def test_scalar_only_signature_is_not_duplicated():
    spec = {
        'abi_version': AOT_ABI_VERSION,
        'params': [AOTParam('scalar', 'bool', 0)],
    }
    signatures, variants = build_export_plan(spec, 'scalar_kernel')
    assert len(signatures) == 1
    assert variants == [{'index_dtype': 'int32', 'symbol': signatures[0].symbol}]


def test_descriptor_abi_version_is_required():
    with pytest.raises(CuTileAOTError, match='ABI version'):
        build_export_plan({'abi_version': AOT_ABI_VERSION + 1, 'params': []}, 'kernel')


def test_prerequisites_only_require_public_api(monkeypatch):
    check_prerequisites()
    monkeypatch.delattr(compilation, 'export_kernel')
    with pytest.raises(CuTileAOTError, match='public APIs'):
        check_prerequisites()


def test_arch_config_override():
    with set_temporary('compiler', 'cutile', 'aot_arch', value='sm_99'):
        assert resolve_arch() == 'sm_99'


def test_aot_export_failure_has_no_fallback(monkeypatch):
    monkeypatch.setattr(compilation, 'export_kernel', lambda *args, **kwargs:
                        (_ for _ in ()).throw(RuntimeError('bad')))
    with pytest.raises(CuTileAOTError, match='AOT export failed'):
        cutile_aot._export_kernel(object(), [], 'sm_120', 'kernel')


def test_rendered_launcher_executes_documented_2d_abi(monkeypatch):
    params = [AOTParam('array', 'float64', 2)]
    namespace, records, modules, _, stream = _launcher_namespace(monkeypatch, params)
    array = _FakeCUDAArray((5, 7), (11, 2))

    namespace['launch']('kernel', (3, 4), (array, ))

    assert len(modules) == 1 and modules[0].loaded == b'fake cubin'
    symbol, grid, block, flattened, actual_stream, _ = records[0]
    assert symbol == 'kernel_i32'
    assert grid == (3, 4, 1)
    assert block == (1, 1, 1)
    assert actual_stream is stream
    assert flattened[0] is array
    assert flattened[1:] == (np.int32(5), np.int32(7), np.int32(11), np.int32(2))
    assert all(isinstance(value, np.int32) for value in flattened[1:])


@pytest.mark.parametrize('shape,element_strides,index_type,symbol', [
    ((2**31 - 1, 3), (3, 1), np.int32, 'kernel_i32'),
    ((2**31, 3), (3, 1), np.int64, 'kernel_i64'),
    ((3, 3), (2**31 - 1, 1), np.int32, 'kernel_i32'),
    ((3, 3), (2**31, 1), np.int64, 'kernel_i64'),
])
def test_rendered_launcher_selects_shape_and_stride_index_width(monkeypatch, shape, element_strides, index_type,
                                                                symbol):
    params = [AOTParam('array', 'float64', 2)]
    namespace, records, _, _, _ = _launcher_namespace(monkeypatch, params)
    array = _FakeCUDAArray(shape, element_strides)

    namespace['launch']('kernel', 1, (array, ))

    assert records[0][0] == symbol
    flattened = records[0][3]
    assert flattened[0] is array
    assert all(isinstance(value, index_type) for value in flattened[1:])


def test_rendered_launcher_rejects_negative_stride(monkeypatch):
    params = [AOTParam('array', 'float64', 2)]
    namespace, records, modules, _, _ = _launcher_namespace(monkeypatch, params)
    array = _FakeCUDAArray((5, 7), (-7, 1))

    with pytest.raises(ValueError, match='negative element strides'):
        namespace['launch']('kernel', 1, (array, ))
    assert records == []
    assert modules == []


def test_rendered_launcher_cache_includes_cuda_context(monkeypatch):
    params = [AOTParam('array', 'float64', 2)]
    namespace, records, modules, context, _ = _launcher_namespace(monkeypatch, params)
    array = _FakeCUDAArray((5, 7), (7, 1))

    namespace['launch']('kernel', 1, (array, ))
    namespace['launch']('kernel', 1, (array, ))
    context[0] = 202
    namespace['launch']('kernel', 1, (array, ))

    assert len(records) == 3
    assert len(modules) == 2
    assert set(namespace['_MODULE_CACHE']) == {
        (7, 101, 'kernel'),
        (7, 202, 'kernel'),
    }


@pytest.mark.gpu
def test_aot_2d_symbolic_remainder(monkeypatch):
    import cupy as cp
    monkeypatch.setenv('DACE_compiler_cutile_mode', 'aot')
    sdfg = _lower(_scale_add_2d, 'aot_direct_2d', (8, 8))
    csdfg = sdfg.compile()
    m, n = 19, 35
    rng = np.random.default_rng(1)
    a_h, b_h = rng.random((m, n)), rng.random((m, n))
    a, b = cp.asarray(a_h), cp.asarray(b_h)
    csdfg(A=a, B=b, M=m, N=n)
    np.testing.assert_allclose(cp.asnumpy(b), a_h * 2.0 + b_h, rtol=1e-14)
    assert '@ct.kernel' not in csdfg.code and 'ct.launch(' not in csdfg.code


@pytest.mark.gpu
def test_aot_strided_views(monkeypatch):
    import cupy as cp
    monkeypatch.setenv('DACE_compiler_cutile_mode', 'aot')
    csdfg = _lower(_vadd, 'aot_direct_views', (32, )).compile()
    n = 70
    x_base = cp.arange(n * 2 + 1, dtype=cp.float64)
    y_base = cp.arange(n * 2 + 1, dtype=cp.float64) * 3
    z_base = cp.zeros(n * 2 + 1, dtype=cp.float64)
    x_view = x_base[1:1 + 2 * n:2]
    y_view = y_base[1:1 + 2 * n:2]
    z_view = z_base[1:1 + 2 * n:2]
    csdfg(x=x_view, y=y_view, z=z_view, N=n)
    cp.testing.assert_array_equal(z_view, x_view + y_view)

    with pytest.raises(ValueError, match='negative element strides'):
        csdfg(x=x_view[::-1], y=y_view, z=z_view, N=n)


@pytest.mark.gpu
def test_aot_float32(monkeypatch):
    import cupy as cp
    monkeypatch.setenv('DACE_compiler_cutile_mode', 'aot')
    csdfg = _lower(_vadd_f32, 'aot_direct_f32', (32, )).compile()
    n = 70
    rng = np.random.default_rng(4)
    x_h = rng.random(n, dtype=np.float32)
    y_h = rng.random(n, dtype=np.float32)
    x, y, z = cp.asarray(x_h), cp.asarray(y_h), cp.zeros(n, dtype=cp.float32)
    csdfg(x=x, y=y, z=z, N=n)
    cp.testing.assert_array_equal(z, x + y)


@pytest.mark.gpu
def test_aot_uint64_symbol_above_signed_range(monkeypatch):
    import cupy as cp
    monkeypatch.setenv('DACE_compiler_cutile_mode', 'aot')
    csdfg = _lower(_shift_u64, 'aot_direct_u64', (32, )).compile()
    n = 70
    shift = np.uint64(2**63 + 17)
    inp = cp.arange(n, dtype=cp.uint64)
    out = cp.zeros(n, dtype=cp.uint64)
    csdfg(x=inp, y=out, N=n, KU=shift)
    cp.testing.assert_array_equal(out, inp + shift)


@pytest.mark.gpu
def test_aot_multiple_kernels_and_large_int64(monkeypatch):
    import cupy as cp
    monkeypatch.setenv('DACE_compiler_cutile_mode', 'aot')
    multi = _lower(_two_kernels, 'aot_direct_multi', (32, )).compile()
    x, y, z = cp.arange(70, dtype=cp.float64), cp.zeros(70), cp.zeros(37)
    multi(x=x, y=y, z=z, N=70, M=37)
    cp.testing.assert_array_equal(y, x * 2.0)
    cp.testing.assert_array_equal(z, cp.ones_like(z))
    modules = [m for n, m in multi._aux_modules.items() if n.startswith('__dace_cutile_aot_')]
    assert len(modules) == 1 and len(modules[0]._KERNELS) == 2

    shift = _lower(_shift_i64, 'aot_direct_i64', (32, )).compile()
    inp, out = cp.arange(70, dtype=cp.int64), cp.zeros(70, dtype=cp.int64)
    shift(x=inp, y=out, N=70, K=2**40 + 123)
    cp.testing.assert_array_equal(out, inp + (2**40 + 123))


@pytest.mark.gpu
def test_architecture_mismatch_raises(monkeypatch):
    import cupy as cp
    monkeypatch.setenv('DACE_compiler_cutile_mode', 'aot')
    csdfg = _lower(_vadd, 'aot_arch_mismatch', (32, )).compile()
    module = next(m for n, m in csdfg._aux_modules.items() if n.startswith('__dace_cutile_aot_'))
    metadata = next(iter(module._KERNELS.values()))
    metadata['arch'] = 'sm_00'
    with pytest.raises(RuntimeError, match='current device'):
        csdfg(x=cp.zeros(32), y=cp.zeros(32), z=cp.zeros(32), N=32)


@pytest.mark.gpu
def test_jit_mode_is_explicit_and_runs(monkeypatch):
    import cupy as cp
    monkeypatch.setenv('DACE_compiler_cutile_mode', 'jit')
    csdfg = _lower(_vadd, 'cutile_explicit_jit', (32, )).compile()
    x, y, z = cp.arange(37, dtype=cp.float64), cp.ones(37), cp.zeros(37)
    csdfg(x=x, y=y, z=z, N=37)
    cp.testing.assert_array_equal(z, x + y)
    assert '@ct.kernel' in csdfg.code and 'ct.launch(' in csdfg.code


@pytest.mark.gpu
def test_output_file_runs_without_compiled_sdfg_hooks(monkeypatch, tmp_path):
    monkeypatch.setenv('DACE_compiler_cutile_mode', 'aot')
    sdfg = _lower(_vadd, 'aot_output_module', (32, ))
    output = tmp_path / 'generated_program.py'
    sdfg.compile(output_file=str(output), return_program_handle=False)
    assert output.exists()
    frame = output.read_text()
    assert '@ct.kernel' not in frame and 'ct.launch(' not in frame and '._compile' not in frame
    auxiliary = next(tmp_path.glob('__dace_cutile_aot_*.py'))
    assert 'module.load(' in auxiliary.read_text()
    runner = '''import cupy as cp\nimport generated_program as generated\nn=70\nx=cp.arange(n,dtype=cp.float64)\ny=cp.ones(n,dtype=cp.float64)\nz=cp.zeros(n,dtype=cp.float64)\ngenerated.aot_output_module(x=x,y=y,z=z,N=n)\ncp.testing.assert_array_equal(z,x+y)\n'''
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(tmp_path), env.get('PYTHONPATH', '')])
    subprocess.run([sys.executable, '-B', '-c', runner], cwd=tmp_path, env=env, check=True, timeout=120)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
