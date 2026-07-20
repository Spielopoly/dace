# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for cuTile AOT compilation (``dace/codegen/py/cutile_aot.py``).

GPU-free unit tests cover signature construction from specs, structural keys,
the version gate and arch resolution. The ``@pytest.mark.gpu`` integration
tests run end to end -- ``@dace.program`` -> ``VectorizeCuTile`` ->
``sdfg.compile()`` (AOT export, on by default) -> launch -> compare vs NumPy --
and assert that launches never JIT-compile.
"""
import dataclasses
import os

import numpy as np
import pytest

import dace
from dace.codegen.py import cutile_aot
from dace.codegen.py.cutile_aot import CuTileAOTError, build_export_plan, check_prerequisites, resolve_arch
from dace.config import Config, set_temporary
from dace.transformation.passes.vectorization import VectorizeCuTile

ct = pytest.importorskip('cuda.tile')
compilation = pytest.importorskip('cuda.tile.compilation')

#: Typical kernel spec: 2-D data array, 1-element device-staged scalar, by-value bool.
SPEC = {
    'params': [
        ('array', 'float64', 2, (None, 1)),
        ('array', 'float64', 1, (1, )),
        ('scalar', 'bool', 0, None),
    ]
}

# =============================================================================
# Signature construction
# =============================================================================


class TestBuildExportPlan:

    def test_single_int32_signature(self):
        signatures, symbols = build_export_plan(SPEC, 'my_kernel')
        assert len(signatures) == 1
        assert len(symbols) == 1
        index_dtypes = set()
        for sig in signatures:
            assert isinstance(sig, compilation.KernelSignature)
            assert sig.calling_convention.version == 1
            arr2d, arr1d, scal = sig.parameters
            assert isinstance(arr2d, compilation.ArrayConstraint)
            assert arr2d.dtype is ct.float64 and arr2d.ndim == 2
            assert arr2d.stride_constant == (None, 1)
            assert isinstance(arr1d, compilation.ArrayConstraint)
            assert arr1d.dtype is ct.float64 and arr1d.ndim == 1
            assert arr1d.stride_constant == (1, )
            assert isinstance(scal, compilation.ScalarConstraint)
            assert scal.dtype is ct.bool_
            assert arr2d.index_dtype is arr1d.index_dtype
            index_dtypes.add(arr2d.index_dtype)
        assert index_dtypes == {ct.int32}

    def test_conservative_constraints(self):
        signatures, _ = build_export_plan(SPEC, 'my_kernel')
        for sig in signatures:
            for param in sig.parameters:
                if not isinstance(param, compilation.ArrayConstraint):
                    continue
                # Non-negative strides only where the stride is not already constant
                # (cuda-tile drops the redundant lower bound on constant-stride dims).
                for const, lb in zip(param.stride_constant, param.stride_lower_bound_incl):
                    assert lb == (None if const is not None else 0)
                # No divisibility/alignment assumptions.
                assert all(d == 1 for d in param.stride_divisible_by)
                assert all(d == 1 for d in param.shape_divisible_by)
                assert param.base_addr_divisible_by == 1
                assert all(s is None for s in param.shape_constant)
                assert param.may_alias_internally is True

    def test_shared_alias_group(self):
        signatures, _ = build_export_plan(SPEC, 'my_kernel')
        for sig in signatures:
            arrays = [p for p in sig.parameters if isinstance(p, compilation.ArrayConstraint)]
            assert all(a.alias_groups == ('dace', ) for a in arrays)

    def test_single_array_no_alias_group(self):
        # A shared alias group with one member is rejected by cuda-tile; must degrade to ().
        spec = {'params': [('array', 'float32', 1, (1, )), ('scalar', 'bool', 0, None)]}
        signatures, _ = build_export_plan(spec, 'k')
        for sig in signatures:
            (arr, ) = [p for p in sig.parameters if isinstance(p, compilation.ArrayConstraint)]
            assert arr.alias_groups == ()

    def test_symbols_mangled_and_keyed(self):
        signatures, symbols = build_export_plan(SPEC, 'my_kernel')
        assert len(set(symbols.values())) == 1
        for sig in signatures:
            assert sig.symbol is not None and sig.symbol.startswith('my_kernel')
            assert symbols[cutile_aot._structural_key(sig)] == sig.symbol

    def test_unknown_kind_raises(self):
        spec = {'params': [('gizmo', 'float64', 1, None)]}
        with pytest.raises(CuTileAOTError, match='kind'):
            build_export_plan(spec, 'k')

    def test_unknown_dtype_raises(self):
        spec = {'params': [('array', 'complex128', 1, None)]}
        with pytest.raises(CuTileAOTError, match='dtype'):
            build_export_plan(spec, 'k')

    def test_scalar_only_spec_exports_one_signature(self):
        """Without array parameters the index dtype has nothing to apply to, so both loop
        iterations build the same signature and symbol -- export it once, not twice."""
        spec = {'params': [('scalar', 'bool', 0, None)]}
        signatures, symbols = build_export_plan(spec, 'scalar_only_kernel')
        assert len(signatures) == 1
        assert len(symbols) == 1

    def test_bad_stride_constant_raises(self):
        spec = {'params': [('array', 'float64', 2, (1, ))]}  # length mismatch with ndim
        with pytest.raises(CuTileAOTError, match='kernel "k"'):
            build_export_plan(spec, 'k')


# =============================================================================
# Structural keys
# =============================================================================


class TestStructuralKey:

    def test_deterministic(self):
        sigs_a, _ = build_export_plan(SPEC, 'k')
        sigs_b, _ = build_export_plan(SPEC, 'k')
        for a, b in zip(sigs_a, sigs_b):
            assert cutile_aot._structural_key(a) == cutile_aot._structural_key(b)

    def test_index_dtype_is_int32(self):
        (signature, ), symbols = build_export_plan(SPEC, 'k')
        key = cutile_aot._structural_key(signature)
        assert signature.parameters[0].index_dtype is ct.int32
        assert key in symbols

    def test_dtype_and_ndim_distinguish(self):
        base = {'params': [('array', 'float64', 2, None), ('array', 'float64', 2, None)]}
        other_dtype = {'params': [('array', 'float32', 2, None), ('array', 'float64', 2, None)]}
        other_ndim = {'params': [('array', 'float64', 1, None), ('array', 'float64', 2, None)]}
        keys = [cutile_aot._structural_key(build_export_plan(s, 'k')[0][0]) for s in (base, other_dtype, other_ndim)]
        assert len(set(keys)) == 3

    def test_scalar_vs_array_distinguish(self):
        as_scalar = {'params': [('scalar', 'float64', 0, None)]}
        as_array = {'params': [('array', 'float64', 1, (1, ))]}
        key_s = cutile_aot._structural_key(build_export_plan(as_scalar, 'k')[0][0])
        key_a = cutile_aot._structural_key(build_export_plan(as_array, 'k')[0][0])
        assert key_s != key_a

    def test_conservative_key_accepts_runtime_specialization(self):
        # Runtime stride and alignment specialization must be accepted by a conservative export.
        cc = compilation.CallingConvention.cutile_python_v1()
        conservative = compilation.ArrayConstraint(ct.float64,
                                                   1,
                                                   index_dtype=ct.int32,
                                                   stride_lower_bound_incl=0,
                                                   alias_groups=(),
                                                   may_alias_internally=False)
        specialized = compilation.ArrayConstraint(ct.float64,
                                                  1,
                                                  index_dtype=ct.int32,
                                                  stride_lower_bound_incl=None,
                                                  alias_groups=(),
                                                  may_alias_internally=False,
                                                  stride_constant=(1, ),
                                                  shape_divisible_by=16,
                                                  base_addr_divisible_by=16)
        key_c = cutile_aot._structural_key(compilation.KernelSignature([conservative], cc))
        key_s = cutile_aot._structural_key(compilation.KernelSignature([specialized], cc))
        assert key_c != key_s
        assert cutile_aot._compatible_key(key_c, key_s)

    def test_unsupported_constraint_raises(self):
        cc = compilation.CallingConvention.cutile_python_v1()
        sig = compilation.KernelSignature([3], cc)  # ConstantConstraint shorthand
        with pytest.raises(CuTileAOTError, match='ConstantConstraint'):
            cutile_aot._structural_key(sig)

    def test_calling_convention_distinguishes(self):
        """Conventions differ in argument packing, so they must not share a key -- otherwise a
        launcher using one convention could be served a symbol compiled for the other."""
        param = compilation.ArrayConstraint(ct.float64,
                                            1,
                                            index_dtype=ct.int32,
                                            stride_lower_bound_incl=0,
                                            alias_groups=(),
                                            may_alias_internally=False)
        key_v1 = cutile_aot._structural_key(
            compilation.KernelSignature([param], compilation.CallingConvention.cutile_python_v1()))
        key_v2 = cutile_aot._structural_key(
            compilation.KernelSignature([param], compilation.CallingConvention.cutile_python_v2()))
        assert key_v1 != key_v2
        # The convention leads the key, ahead of the per-parameter entries.
        assert key_v1[0] == compilation.CallingConvention.cutile_python_v1().code
        assert key_v1[1:] == key_v2[1:]

    def test_built_signatures_carry_the_convention(self):
        """Built and derived keys are symmetric: build_export_plan's keys start with the same
        convention code that a launch-derived signature of ours would produce."""
        _, symbols = build_export_plan(SPEC, 'conv_kernel')
        expected = compilation.CallingConvention.cutile_python_v1().code
        assert all(key[0] == expected for key in symbols)


# =============================================================================
# Version gate
# =============================================================================


class TestVersionGate:

    def test_current_install_passes(self):
        check_prerequisites()

    def test_wrong_version_raises(self, monkeypatch):
        monkeypatch.setattr(ct, '__version__', '1.4.0')
        with pytest.raises(CuTileAOTError, match='1.5'):
            check_prerequisites()
        with pytest.raises(CuTileAOTError, match='aot_compile'):
            check_prerequisites()

    def test_missing_export_attr_raises(self, monkeypatch):
        monkeypatch.delattr(compilation, 'export_kernel')
        with pytest.raises(CuTileAOTError, match='export_kernel'):
            check_prerequisites()

    def test_wrong_compile_arity_raises(self, monkeypatch):
        monkeypatch.setattr(ct.kernel, '_compile', lambda self, signature: None)
        with pytest.raises(CuTileAOTError, match='_compile'):
            check_prerequisites()


# =============================================================================
# Arch resolution and config defaults
# =============================================================================


def test_compile_version_failure_has_no_jit_fallback(monkeypatch):
    """The compiled-SDFG integration must propagate AOT prerequisite failures."""
    monkeypatch.setattr(ct, "__version__", "1.4.0")
    sdfg = _lower(_vadd, "aot_version_failure", (32, ))
    with pytest.raises(CuTileAOTError, match="1.5"):
        sdfg.compile()


class TestResolveArch:

    def test_config_override(self):
        with set_temporary('compiler', 'cutile', 'aot_arch', value='sm_99'):
            assert resolve_arch() == 'sm_99'

    def test_auto_detect(self, monkeypatch):
        cupy = pytest.importorskip('cupy')

        class FakeDevice:

            compute_capability = '120'

        monkeypatch.setattr(cupy.cuda, 'Device', lambda *a, **kw: FakeDevice())
        with set_temporary('compiler', 'cutile', 'aot_arch', value=''):
            assert resolve_arch() == 'sm_120'

    def test_no_gpu_no_override_raises(self, monkeypatch):
        cupy = pytest.importorskip('cupy')

        def broken_device(*args, **kwargs):
            raise RuntimeError('no GPU')

        monkeypatch.setattr(cupy.cuda, 'Device', broken_device)
        with set_temporary('compiler', 'cutile', 'aot_arch', value=''):
            with pytest.raises(CuTileAOTError, match='aot_arch'):
                resolve_arch()


class TestExportCubin:
    """Every export and atomic-publication failure must surface as CuTileAOTError."""

    def test_export_failure_raises(self, monkeypatch, tmp_path):

        def fail(*args, **kwargs):
            raise RuntimeError("export broke")

        monkeypatch.setattr(compilation, 'export_kernel', fail)
        with pytest.raises(CuTileAOTError, match='AOT export failed') as excinfo:
            cutile_aot._export_cubin(None, [], str(tmp_path / 'k.cubin'), 'sm_120', 'my_kernel')
        assert 'aot_compile=False' in str(excinfo.value)

    def test_publish_failure_raises(self, monkeypatch, tmp_path):

        def write_bytes(kernel, signatures, output, **kwargs):
            output.write(b"cubin")

        monkeypatch.setattr(compilation, 'export_kernel', write_bytes)
        missing = str(tmp_path / 'nope' / 'k.cubin')
        with pytest.raises(CuTileAOTError, match='Cannot publish') as excinfo:
            cutile_aot._export_cubin(None, [], missing, 'sm_120', 'my_kernel')
        assert 'aot_compile=False' in str(excinfo.value)

    def test_empty_cubin_raises(self, monkeypatch, tmp_path):
        path = tmp_path / 'k.cubin'
        path.write_bytes(b'')  # e.g. a concurrent export truncating the file
        monkeypatch.setattr(compilation, 'export_kernel', lambda *a, **kw: None)
        with pytest.raises(CuTileAOTError, match='is empty') as excinfo:
            cutile_aot._export_cubin(None, [], str(path), 'sm_120', 'my_kernel')
        assert 'my_kernel' in str(excinfo.value)


class TestFailureContracts:

    def test_output_directory_failure_is_wrapped(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cutile_aot, "check_prerequisites", lambda: None)
        monkeypatch.setattr(cutile_aot, "resolve_arch", lambda: "sm_120")

        def fail_makedirs(*args, **kwargs):
            raise OSError("read only")

        monkeypatch.setattr(os, "makedirs", fail_makedirs)
        sdfg = type("FakeSDFG", (), {"build_folder": str(tmp_path)})()
        namespace = {cutile_aot.AOT_SPECS_NAME: {"k": {"params": []}}}
        with pytest.raises(CuTileAOTError, match="output directory") as excinfo:
            cutile_aot.precompile_kernels(namespace, sdfg)
        assert "aot_compile=False" in str(excinfo.value)

    def test_arch_mismatch_raises(self, monkeypatch):

        @ct.kernel
        def kernel_for_arch_test(x):
            pass

        entry = cutile_aot.AOTEntry(cubin=b"x", arch="sm_99", symbols={})
        kernel = cutile_aot._make_precompiled(kernel_for_arch_test, entry)
        monkeypatch.setattr(cutile_aot, "_current_arch", lambda: "sm_120")
        with pytest.raises(CuTileAOTError, match="exported for sm_99"):
            kernel._compile(None, None)


class TestConfigDefaults:

    def test_aot_compile_default_on(self):
        assert Config.get_bool('compiler', 'cutile', 'aot_compile') is True

    def test_aot_arch_default_empty(self):
        assert Config.get('compiler', 'cutile', 'aot_arch') == ''


# =============================================================================
# GPU integration: end-to-end through VectorizeCuTile with AOT on by default
# =============================================================================

N = dace.symbol('N')
M = dace.symbol('M')
K = dace.symbol('K', dtype=dace.int64)


@dace.program
def _vadd(x: dace.float64[N], y: dace.float64[N], z: dace.float64[N]):
    z[:] = x + y


@dace.program
def _scale_add_2d(A: dace.float64[M, N], B: dace.float64[M, N]):
    B[:] = A * 2.0 + B


@dace.program
def _vadd_f32(x: dace.float32[N], y: dace.float32[N], z: dace.float32[N]):
    z[:] = x + y


@dace.program
def _shift_i64(x: dace.int64[N], y: dace.int64[N]):
    y[:] = x + K


@dace.program
def _row_accumulate(x: dace.float64[M, N], y: dace.float64[N], steps: dace.int64):
    for i in range(steps):
        y[:] = y + x[i, :]


@dace.program
def _two_kernels(x: dace.float64[N], y: dace.float64[N], z: dace.float64[M]):
    y[:] = x * 2.0
    z[:] = z + 1.0


def _lower(program, name: str, widths) -> dace.SDFG:
    """Lower a ``@dace.program`` through the cuTile front door.

    :param program: The dace program.
    :param name: Unique SDFG name (isolated build folder).
    :param widths: Tile widths (powers of 2), forwarded to VectorizeCuTile.
    :returns: The lowered SDFG (Python backend, cupy-array calling convention).
    """
    sdfg = program.to_sdfg(simplify=False)
    sdfg.name = name
    VectorizeCuTile(widths=widths, use_gpu_storage=True).apply_pass(sdfg, {})
    return sdfg


def _aot_kernels(csdfg) -> dict:
    """Registered kernel name -> namespace object for a compiled SDFG."""
    specs = csdfg._namespace.get(cutile_aot.AOT_SPECS_NAME) or {}
    return {name: csdfg._namespace[name] for name in specs}


@pytest.fixture
def jit_compiles(monkeypatch):
    """Spy on JIT compilation: ``ct.kernel._compile`` is the entry ``ct.launch``
    uses on a dispatch-cache miss. ``PrecompiledKernel`` overrides ``_compile``
    in its own class, so AOT launches never reach this. AOT export calls
    ``compile_tile`` directly, not this method, so the spy counts JIT only."""
    calls = []
    orig = ct.kernel._compile

    def spy(self, signature, context):
        calls.append(signature)
        return orig(self, signature, context)

    monkeypatch.setattr(ct.kernel, '_compile', spy)
    return calls


@pytest.mark.gpu
def test_aot_zero_jit_and_artifacts(jit_compiles):
    """AOT is actually used: PrecompiledKernel in the namespace, cubin on disk,
    zero JIT compilations across compile + two launches, NumPy-exact."""
    import cupy as cp

    sdfg = _lower(_vadd, 'aot_it_vadd', (32, ))
    csdfg = sdfg.compile()

    kernels = _aot_kernels(csdfg)
    assert kernels, 'no AOT-registered kernels in the compiled namespace'
    for name, kernel in kernels.items():
        assert isinstance(kernel, cutile_aot.PrecompiledKernel)
        cubin = os.path.join(sdfg.build_folder, 'cutile_aot', f'{name}.cubin')
        assert os.path.isfile(cubin) and os.path.getsize(cubin) > 0

    rng = np.random.default_rng(0)
    n = 70  # not divisible by the tile width: remainder path
    x_h, y_h = rng.random(n), rng.random(n)
    x, y, z = cp.asarray(x_h), cp.asarray(y_h), cp.zeros(n)
    csdfg(x=x, y=y, z=z, N=n)
    np.testing.assert_allclose(cp.asnumpy(z), x_h + y_h, rtol=1e-14)

    n2 = 64  # second launch, different (divisible) size
    x2, y2, z2 = cp.asarray(x_h[:n2]), cp.asarray(y_h[:n2]), cp.zeros(n2)
    csdfg(x=x2, y=y2, z=z2, N=n2)
    np.testing.assert_allclose(cp.asnumpy(z2), x_h[:n2] + y_h[:n2], rtol=1e-14)

    assert jit_compiles == []


@pytest.mark.gpu
def test_aot_2d_symbolic_remainder(jit_compiles):
    """2-D float64 kernel, symbolic M/N with a remainder in both dims."""
    import cupy as cp

    sdfg = _lower(_scale_add_2d, 'aot_it_2d', (8, 8))
    csdfg = sdfg.compile()

    m, n = 20, 28
    rng = np.random.default_rng(1)
    A_h, B_h = rng.random((m, n)), rng.random((m, n))
    ref = A_h * 2.0 + B_h
    A, B = cp.asarray(A_h), cp.asarray(B_h)
    csdfg(A=A, B=B, M=m, N=n)
    np.testing.assert_allclose(cp.asnumpy(B), ref, rtol=1e-14)
    assert jit_compiles == []


@pytest.mark.gpu
def test_aot_int64_symbol_large_value(jit_compiles):
    """An int64 symbol >= 2**31 (by-value it would OverflowError) through AOT."""
    import cupy as cp

    sdfg = _lower(_shift_i64, 'aot_it_i64', (32, ))
    csdfg = sdfg.compile()

    n, k = 70, 2**40 + 12345
    x_h = np.arange(n, dtype=np.int64)
    x, y = cp.asarray(x_h), cp.zeros(n, dtype=cp.int64)
    csdfg(x=x, y=y, N=n, K=k)
    np.testing.assert_array_equal(cp.asnumpy(y), x_h + k)
    assert jit_compiles == []


@pytest.mark.gpu
def test_aot_unaligned_views(jit_compiles):
    """Unaligned cupy views (``a[1:]``) hit the conservative AOT signature."""
    import cupy as cp

    sdfg = _lower(_vadd, 'aot_it_views', (32, ))
    csdfg = sdfg.compile()

    n = 66
    rng = np.random.default_rng(2)
    x_f, y_f = cp.asarray(rng.random(n + 1)), cp.asarray(rng.random(n + 1))
    z_f = cp.zeros(n + 1)
    csdfg(x=x_f[1:], y=y_f[1:], z=z_f[1:], N=n)
    ref = cp.asnumpy(x_f)[1:] + cp.asnumpy(y_f)[1:]
    np.testing.assert_allclose(cp.asnumpy(z_f)[1:], ref, rtol=1e-14)
    assert jit_compiles == []


@pytest.mark.gpu
def test_aot_host_loop_induction_symbol(jit_compiles):
    """A host loop whose induction variable is a kernel argument.

    The loop variable is runtime-defined (absent from ``sdfg.symbols``), so its launch dtype comes
    from the frame's loop-bound inference rather than from the runtime value. That pinned dtype is
    what the AOT spec is built from, so the kernel must hit the exported signature with zero JIT.
    """
    cp = pytest.importorskip('cupy')

    sdfg = _lower(_row_accumulate, 'aot_it_hostloop', (32, ))

    # The induction variable reaches the kernel as a runtime-defined symbol with a pinned dtype.
    code = sdfg.generate_code()[0].code
    loop_syms = [s for s in sdfg.free_symbols | set(sdfg.symbols) if s.startswith('_loop_it')]
    assert not loop_syms, 'loop variable unexpectedly declared as an SDFG symbol'
    assert 'cupy.asarray(_loop_it_0, dtype=numpy.int64).reshape(1)' in ' '.join(code.split())

    csdfg = sdfg.compile()
    kernels = _aot_kernels(csdfg)
    assert kernels
    assert all(isinstance(k, cutile_aot.PrecompiledKernel) for k in kernels.values())
    # The induction symbol is staged as a 1-element int64 device array in the spec.
    specs = csdfg._namespace[cutile_aot.AOT_SPECS_NAME]
    assert any(('array', 'int64', 1, (1, )) in spec['params'] for spec in specs.values())

    m, n, steps = 6, 70, 4  # n not divisible by the tile width: remainder path
    rng = np.random.default_rng(6)
    x_h, y_h = rng.random((m, n)), rng.random(n)
    ref = y_h.copy()
    for i in range(steps):
        ref = ref + x_h[i, :]

    x, y = cp.asarray(x_h), cp.asarray(y_h)
    csdfg(x=x, y=y, steps=steps, N=n)
    np.testing.assert_allclose(cp.asnumpy(y), ref, rtol=1e-14)
    assert jit_compiles == []


@pytest.mark.gpu
def test_aot_multi_kernel_sdfg(jit_compiles):
    """Two cuTile kernels in one SDFG: both precompiled, both correct."""
    import cupy as cp

    sdfg = _lower(_two_kernels, 'aot_it_multi', (32, ))
    csdfg = sdfg.compile()

    kernels = _aot_kernels(csdfg)
    assert len(kernels) == 2
    assert all(isinstance(k, cutile_aot.PrecompiledKernel) for k in kernels.values())

    n, m = 70, 40
    rng = np.random.default_rng(3)
    x_h, z_h = rng.random(n), rng.random(m)
    x, y, z = cp.asarray(x_h), cp.zeros(n), cp.asarray(z_h)
    csdfg(x=x, y=y, z=z, N=n, M=m)
    np.testing.assert_allclose(cp.asnumpy(y), x_h * 2.0, rtol=1e-14)
    np.testing.assert_allclose(cp.asnumpy(z), z_h + 1.0, rtol=1e-14)
    assert jit_compiles == []


@pytest.mark.gpu
def test_aot_strided_views(jit_compiles):
    """Non-unit runtime strides remain correct under AOT."""
    import cupy as cp
    sdfg = _lower(_vadd, "aot_it_strided", (32, ))
    csdfg = sdfg.compile()
    n = 64
    x_base = cp.arange(n * 2, dtype=cp.float64)
    y_base = cp.arange(n * 2, dtype=cp.float64) * 3
    z_base = cp.zeros(n * 2, dtype=cp.float64)
    csdfg(x=x_base[::2], y=y_base[::2], z=z_base[::2], N=n)
    cp.testing.assert_array_equal(z_base[::2], x_base[::2] + y_base[::2])
    assert jit_compiles == []


@pytest.mark.gpu
def test_aot_float32_launch(jit_compiles):
    """Float32 launches from the exported cubin."""
    import cupy as cp
    sdfg = _lower(_vadd_f32, "aot_it_f32", (32, ))
    csdfg = sdfg.compile()
    n = 70
    x = cp.arange(n, dtype=cp.float32)
    y = cp.arange(n, dtype=cp.float32) * cp.float32(0.25)
    z = cp.zeros(n, dtype=cp.float32)
    csdfg(x=x, y=y, z=z, N=n)
    cp.testing.assert_array_equal(z, x + y)
    assert jit_compiles == []


@pytest.mark.gpu
def test_aot_and_jit_outputs_are_identical():
    """AOT and JIT produce bit-identical output for identical inputs."""
    import cupy as cp
    n = 70
    rng = np.random.default_rng(9)
    x = cp.asarray(rng.random(n))
    y = cp.asarray(rng.random(n))
    aot_out = cp.zeros(n)
    jit_out = cp.zeros(n)
    _lower(_vadd, "aot_bit_equal", (32, )).compile()(x=x, y=y, z=aot_out, N=n)
    with set_temporary("compiler", "cutile", "aot_compile", value=False):
        _lower(_vadd, "jit_bit_equal", (32, )).compile()(x=x, y=y, z=jit_out, N=n)
    cp.testing.assert_array_equal(aot_out, jit_out)


@pytest.mark.gpu
def test_aot_config_off_pure_jit(jit_compiles, monkeypatch):
    """Config off: no registry, no cubin dir, no precompile call, plain
    ``ct.kernel`` JIT (compile counter > 0), results still correct."""
    import cupy as cp

    precompile_calls = []
    monkeypatch.setattr(cutile_aot, 'precompile_kernels', lambda *a, **kw: precompile_calls.append(a))

    with set_temporary('compiler', 'cutile', 'aot_compile', value=False):
        sdfg = _lower(_vadd, 'aot_it_off', (32, ))
        csdfg = sdfg.compile()

        assert cutile_aot.AOT_SPECS_NAME not in csdfg._namespace
        assert precompile_calls == []
        assert not os.path.exists(os.path.join(sdfg.build_folder, 'cutile_aot'))
        kernels = [v for v in csdfg._namespace.values() if isinstance(v, ct.kernel)]
        assert kernels
        assert not any(isinstance(k, cutile_aot.PrecompiledKernel) for k in kernels)

        n = 70
        rng = np.random.default_rng(4)
        x_h, y_h = rng.random(n), rng.random(n)
        x, y, z = cp.asarray(x_h), cp.asarray(y_h), cp.zeros(n)
        csdfg(x=x, y=y, z=z, N=n)
        np.testing.assert_allclose(cp.asnumpy(z), x_h + y_h, rtol=1e-14)
        assert len(jit_compiles) > 0  # the JIT path really compiled


@pytest.mark.gpu
def test_aot_recompile_overwrites(jit_compiles):
    """Recompiling re-exports (no caching): cubin mtimes advance, still correct."""
    import cupy as cp

    sdfg = _lower(_vadd, 'aot_it_recompile', (32, ))
    csdfg1 = sdfg.compile()
    paths = [
        os.path.join(sdfg.build_folder, 'cutile_aot', f'{name}.cubin')
        for name in csdfg1._namespace[cutile_aot.AOT_SPECS_NAME]
    ]
    assert paths
    first = {p: os.stat(p).st_mtime_ns for p in paths}

    csdfg2 = sdfg.compile()
    for p in paths:
        assert os.stat(p).st_mtime_ns > first[p], f'{p} was not re-exported'

    n = 70
    rng = np.random.default_rng(5)
    x_h, y_h = rng.random(n), rng.random(n)
    x, y, z = cp.asarray(x_h), cp.asarray(y_h), cp.zeros(n)
    csdfg2(x=x, y=y, z=z, N=n)
    np.testing.assert_allclose(cp.asnumpy(z), x_h + y_h, rtol=1e-14)
    assert jit_compiles == []


@pytest.mark.gpu
def test_aot_corrupted_symbol_map_raises(jit_compiles):
    """Fail loudly: an emptied symbol map raises CuTileAOTError at launch (with
    the derived signature in the message), never falling back to JIT."""
    import cupy as cp

    sdfg = _lower(_vadd, 'aot_it_corrupt', (32, ))
    csdfg = sdfg.compile()
    kernels = _aot_kernels(csdfg)
    assert kernels
    for kernel in kernels.values():
        kernel._aot_entry = dataclasses.replace(kernel._aot_entry, symbols={})

    n = 64
    x, y, z = cp.zeros(n), cp.zeros(n), cp.zeros(n)
    with pytest.raises(CuTileAOTError, match='signature mismatch') as excinfo:
        csdfg(x=x, y=y, z=z, N=n)
    assert 'Derived signature' in str(excinfo.value)
    assert jit_compiles == []  # no silent JIT fallback
