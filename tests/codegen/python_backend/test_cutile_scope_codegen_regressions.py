# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression tests for cuTile ``generate_scope`` / kernel-parameter codegen.

Covers the code-review findings on the scope generator:

1. **Bound map params never become launch args**: a kernel map parameter that
   collides with a host loop variable or an interstate-assignment key must not
   pass the ``runtime_defined`` filter in ``_collect_free_symbols`` (it would
   become an undefined-at-callsite launch argument -> ``NameError``).
2. **Float Scalar in kernel input AND output**: the launch site passes
   kernel-written scalars raw while in-kernel reads bind float scalars as 0-d
   tiles -- incompatible conventions, so codegen must raise loudly.
3. **Float64 SYMBOLS**: by-value float kernel arguments are typed float32 by
   the cuda.tile frontend; float symbols must ride the same device-memory
   path as float Scalars (1-element device array + 0-d tile load).
4. **Launch guard**: non-positive grid dims (zero- or negative-trip maps) skip
   the launch (``min((...)) > 0``), not just zero dims.
"""
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
# 2: float Scalar in input AND output -> NotImplementedError
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
    sdfg = _inout_scalar_sdfg(dace.float64)
    with pytest.raises(NotImplementedError, match='both a kernel input and a kernel output'):
        sdfg.generate_code()


def test_int_scalar_inout_still_generates():
    """Only FLOAT scalars are affected; integer in/out scalars pass through."""
    sdfg = _inout_scalar_sdfg(dace.int64)
    code = sdfg.generate_code()[0].code
    assert 'ct.launch' in code


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


def test_float64_symbol_staged_through_device_memory():
    """The launch site wraps the float symbol in a 1-element device array and
    the kernel rebinds it as a 0-d tile (by-value floats are typed float32
    by cuda.tile, silently losing f64 precision)."""
    sdfg = _float_symbol_sdfg('float_sym_structural')
    code = sdfg.generate_code()[0].code.replace(' ', '')
    assert 'cupy.asarray(alpha_v,dtype=numpy.float64).reshape(1)' in code
    assert 'alpha_v=ct.load(alpha_v,(0,),shape=())' in code


def test_launch_guard_skips_nonpositive_grid():
    """The launch is guarded by ``min((...)) > 0`` (not merely ``0 not in``),
    so negative-trip grids (e.g. ``1:N-1`` at ``N == 1``) are skipped too."""
    sdfg = _float_symbol_sdfg('launch_guard_structural')
    code = sdfg.generate_code()[0].code.replace(' ', '').replace('\n', '')
    assert '>0:ct.launch' in code
    assert 'ifmin((' in code
    assert '0notin' not in code


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
