# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Runtime scalars use exact exported cuTile scalar constraints.

The build-only module assigns an explicit ``ScalarConstraint`` for every
read-only numeric scalar. The Cython host packs the matching fixed-width NumPy
scalar, avoiding the demotion performed by the old runtime ``ct.launch`` path.

Covers: f64 bit-exactness, int64 >= 2**31 (the old OverflowError case),
uint64 >= 2**63, narrow ints (int8/int16 -- probed: cuda.tile accepts
narrow ints, int symbols as sizes with non-divisible boundaries, bool args,
and the structural exported-ABI contract.
"""
import itertools

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.dtypes import Language, ScheduleType, StorageType
from dace.memlet import Memlet

_COUNTER = itertools.count()

N = dace.symbol('N')

#: A float64 value that is NOT exactly representable in float32 -- an
#: f32-demoted scalar shows up as a ~1.6e-8 relative error.
_F64_PROBE = 0.3333333333333333


def _cutile_compile(program, widths=(32, )):
    """Lower a ``@dace.program`` through the full cuTile pipeline.

    :param program: The DaCe program to compile.
    :param widths: Tile widths for :class:`VectorizeCuTile`.
    :returns: The compiled Python-backend SDFG.
    """
    from dace.transformation.passes.vectorization import VectorizeCuTile
    sdfg = program.to_sdfg(simplify=False)
    sdfg.name = f'{sdfg.name}_{next(_COUNTER)}'
    VectorizeCuTile(widths=widths).apply_pass(sdfg, {})
    return sdfg.compile()


@dace.program
def _scale_f64(x: dace.float64[N], a: dace.float64):
    return x * a


@dace.program
def _shift_i64(x: dace.int64[N], a: dace.int64):
    return x + a


@dace.program
def _shift_u64(x: dace.uint64[N], a: dace.uint64):
    return x + a


# =============================================================================
# End-to-end runtime tests (GPU)
# =============================================================================


@pytest.mark.gpu
class TestScalarABIRuntime:

    def test_f64_scalar_bit_exact(self):
        """A non-f32-exact float64 scalar survives the kernel bit-exactly
        (by value it would round to f32: ~1.6e-8 relative error)."""
        csdfg = _cutile_compile(_scale_f64)
        n = 70  # non-divisible tail
        x = np.random.default_rng(3).random(n)
        out = np.asarray(csdfg(x=x, a=np.float64(_F64_PROBE), N=n))
        ref = x * np.float64(_F64_PROBE)
        np.testing.assert_array_equal(out, ref)  # bit-exact

    def test_python_float_scalar_bit_exact(self):
        """Same through a plain Python float argument."""
        csdfg = _cutile_compile(_scale_f64)
        n = 64
        x = np.random.default_rng(4).random(n)
        out = np.asarray(csdfg(x=x, a=_F64_PROBE, N=n))
        np.testing.assert_array_equal(out, x * np.float64(_F64_PROBE))

    def test_int64_scalar_above_int32_range(self):
        """An int64 scalar >= 2**31: by value this raised OverflowError
        (cuda.tile types Python ints as int32)."""
        csdfg = _cutile_compile(_shift_i64)
        n = 70
        big = 2**40 + 12345
        x = np.arange(n, dtype=np.int64)
        out = np.asarray(csdfg(x=x, a=np.int64(big), N=n))
        np.testing.assert_array_equal(out, x + big)

    def test_int64_scalar_boundary_value(self):
        """Exactly 2**31 -- the first value the old by-value path rejected."""
        csdfg = _cutile_compile(_shift_i64)
        n = 64
        x = np.arange(n, dtype=np.int64)
        out = np.asarray(csdfg(x=x, a=np.int64(2**31), N=n))
        np.testing.assert_array_equal(out, x + 2**31)

    def test_uint64_scalar_above_int64_range(self):
        """A uint64 scalar >= 2**63 keeps its exact unsigned exported ABI."""
        csdfg = _cutile_compile(_shift_u64)
        n = 64
        big = np.uint64(2**63 + 12345)
        x = np.arange(n, dtype=np.uint64)
        out = np.asarray(csdfg(x=x, a=big, N=n))
        np.testing.assert_array_equal(out, x + big)

    def test_int_symbol_as_size_non_divisible(self):
        """The symbolic size N uses an exact scalar ABI in the mask bound."""
        csdfg = _cutile_compile(_scale_f64)
        for n in (1, 31, 100):  # all-tail, sub-width, and multi-tile+tail
            x = np.random.default_rng(n).random(n)
            out = np.asarray(csdfg(x=x, a=np.float64(2.5), N=n))
            np.testing.assert_array_equal(out, x * 2.5)

    @pytest.mark.parametrize('dtype, np_dtype', [(dace.int8, np.int8), (dace.int16, np.int16)])
    def test_narrow_int_scalar_arg(self, dtype, np_dtype):
        """int8/int16 scalars use matching narrow exported constraints."""
        cupy = pytest.importorskip('cupy')
        sdfg = _scalar_arg_sdfg(f'narrow_{np_dtype.__name__}_{next(_COUNTER)}', dtype)
        csdfg = sdfg.compile()
        x_host = np.random.default_rng(7).random(64)
        x = cupy.asarray(x_host)
        y = cupy.zeros(64, dtype=cupy.float64)
        csdfg(x=x, y=y, a=np_dtype(7))
        np.testing.assert_array_equal(cupy.asnumpy(y), x_host + 7.0)

    def test_bool_scalar_arg(self):
        """Bool scalars keep the by-value path and still work."""
        cupy = pytest.importorskip('cupy')
        sdfg = _bool_scalar_sdfg('bool_arg_runtime')
        csdfg = sdfg.compile()
        x_host = np.random.default_rng(9).random(64)
        for flag in (True, False):
            x = cupy.asarray(x_host)
            y = cupy.zeros(64, dtype=cupy.float64)
            csdfg(x=x, y=y, flag=flag)
            ref = x_host * 2.0 if flag else x_host
            np.testing.assert_array_equal(cupy.asnumpy(y), ref)


# =============================================================================
# Structural tests (no GPU): explicit constraints + exact host packing
# =============================================================================


def _scalar_arg_sdfg(name: str, dtype, n: int = 64, tile_w: int = 32) -> dace.SDFG:
    """``y = x + a`` with ``a`` a non-transient Scalar argument feeding the
    kernel tasklet (the AccessNode-centric manual-tile pattern)."""
    sdfg = dace.SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_scalar('a', dtype)
    sdfg.add_array('x', [n], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array('y', [n], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array('_tx', [tile_w], dace.float64, storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array('_ty', [tile_w], dace.float64, storage=StorageType.CuTile_Tile, transient=True)
    state = sdfg.add_state('main')
    me, mx = state.add_map('cutile_map', {'tile_i': f'0:{n}:{tile_w}'}, schedule=ScheduleType.CuTile)
    xr = state.add_read('x')
    ar = state.add_read('a')
    yw = state.add_write('y')
    tx = state.add_access('_tx')
    ty = state.add_access('_ty')
    tk = state.add_tasklet('shift', {'inp', 'a_in'}, {'out'}, 'out = inp + a_in', language=Language.Python)
    state.add_memlet_path(xr, me, tx, memlet=Memlet(data='x', subset=f'0:{n}'))
    state.add_memlet_path(ar, me, tk, dst_conn='a_in', memlet=Memlet(data='a', subset='0'))
    state.add_edge(tx, None, tk, 'inp', Memlet(data='_tx', subset=f'0:{tile_w}'))
    state.add_edge(tk, 'out', ty, None, Memlet(data='_ty', subset=f'0:{tile_w}'))
    state.add_memlet_path(ty, mx, yw, memlet=Memlet(data='y', subset=f'0:{n}'))
    sdfg.fill_scope_connectors()
    return sdfg


def _bool_scalar_sdfg(name: str, n: int = 64, tile_w: int = 32) -> dace.SDFG:
    """``y = x * 2 if flag else x`` with ``flag`` a bool Scalar argument."""
    sdfg = dace.SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_scalar('flag', dace.bool)
    sdfg.add_array('x', [n], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array('y', [n], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array('_tx', [tile_w], dace.float64, storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array('_ty', [tile_w], dace.float64, storage=StorageType.CuTile_Tile, transient=True)
    state = sdfg.add_state('main')
    me, mx = state.add_map('cutile_map', {'tile_i': f'0:{n}:{tile_w}'}, schedule=ScheduleType.CuTile)
    xr = state.add_read('x')
    fr = state.add_read('flag')
    yw = state.add_write('y')
    tx = state.add_access('_tx')
    ty = state.add_access('_ty')
    tk = state.add_tasklet('sel', {'inp', 'f_in'}, {'out'},
                           'out = inp * 2.0 if f_in else inp',
                           language=Language.Python)
    state.add_memlet_path(xr, me, tx, memlet=Memlet(data='x', subset=f'0:{n}'))
    state.add_memlet_path(fr, me, tk, dst_conn='f_in', memlet=Memlet(data='flag', subset='0'))
    state.add_edge(tx, None, tk, 'inp', Memlet(data='_tx', subset=f'0:{tile_w}'))
    state.add_edge(tk, 'out', ty, None, Memlet(data='_ty', subset=f'0:{tile_w}'))
    state.add_memlet_path(ty, mx, yw, memlet=Memlet(data='y', subset=f'0:{n}'))
    sdfg.fill_scope_connectors()
    return sdfg


class TestExportedScalarABI:
    """Generated code uses matching explicit constraints and host packing."""

    @pytest.mark.parametrize('dtype, np_name', [
        (dace.float64, 'float64'),
        (dace.float32, 'float32'),
        (dace.float16, 'float16'),
        (dace.int64, 'int64'),
        (dace.int32, 'int32'),
        (dace.int16, 'int16'),
        (dace.int8, 'int8'),
        (dace.uint32, 'uint32'),
        (dace.uint16, 'uint16'),
        (dace.uint8, 'uint8'),
        (dace.uint64, 'uint64'),
    ])
    def test_numeric_scalar_has_exact_constraint_and_packing(self, dtype, np_name):
        sdfg = _scalar_arg_sdfg(f'stage_{np_name}_{next(_COUNTER)}', dtype)
        code_objects = sdfg.generate_code()
        host = next(code.code for code in code_objects if code.language == 'pyx').replace(' ', '')
        build = next(code.code for code in code_objects if code.target_type == 'cutile_build').replace(' ', '')
        assert f'compilation.ScalarConstraint(ct.{np_name})' in build
        assert f'numpy.{np_name}(__dace_raw_' in host
        assert 'cupy.asarray(a' not in host
        assert 'a_in=ct.load(a,(0,),shape=()).item()' not in build
        assert 'a_in=a' in build

    def test_bool_scalar_has_exact_constraint(self):
        sdfg = _bool_scalar_sdfg(f'stage_bool_{next(_COUNTER)}')
        code_objects = sdfg.generate_code()
        host = next(code.code for code in code_objects if code.language == 'pyx').replace(' ', '')
        build = next(code.code for code in code_objects if code.target_type == 'cutile_build').replace(' ', '')
        assert 'compilation.ScalarConstraint(ct.bool_)' in build
        assert 'numpy.bool_(__dace_raw_' in host
        assert 'cupy.asarray(flag' not in host
        assert 'ct.load(flag' not in build


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
