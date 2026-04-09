"""Tests for the RemoveIntermediateTransient transformation."""
import pytest
import numpy as np

import dace
from dace import SDFG, dtypes
from dace.sdfg import nodes
from dace.libraries.cutile.transformations.remove_intermediate_transient import (
    RemoveIntermediateTransient,
)


def _make_scalar_intermediate_sdfg():
    """Create a simple SDFG: Tasklet → scalar c_tmp → MapExit → c AccessNode.

    Equivalent to ``c[i] = a[i] + a[i]`` with an intermediate scalar.
    """
    N = dace.symbol('N')
    sdfg = SDFG('scalar_intermediate')
    sdfg.add_array('a', [N], dace.float32)
    sdfg.add_array('c', [N], dace.float32)
    sdfg.add_scalar('c_tmp', dace.float32, transient=True)

    state = sdfg.add_state('compute')

    a_read = state.add_access('a')
    c_tmp = state.add_access('c_tmp')
    c_write = state.add_access('c')

    me, mx = state.add_map('mymap', dict(i='0:N'))

    tasklet = state.add_tasklet('add', {'__in1', '__in2'}, {'__out'},
                                '__out = __in1 + __in2')

    state.add_memlet_path(a_read, me, tasklet,
                          memlet=dace.Memlet('a[i]'), dst_conn='__in1')
    state.add_memlet_path(a_read, me, tasklet,
                          memlet=dace.Memlet('a[i]'), dst_conn='__in2')
    state.add_edge(tasklet, '__out', c_tmp, None, dace.Memlet('c_tmp[0]'))
    mx.add_in_connector('IN_c')
    state.add_edge(c_tmp, None, mx, 'IN_c',
                   dace.Memlet(data='c', subset='i', other_subset='0'))
    state.add_memlet_path(mx, c_write,
                          memlet=dace.Memlet('c[i]'), src_conn='OUT_c')

    return sdfg


def _make_2d_tiled_sdfg():
    """Create a tiled 2D SDFG similar to cutile pipeline output.

    Pattern: Tasklet → c_slice scalar → InnerMapExit → OuterMapExit → c
    """
    N = dace.symbol('N')
    sdfg = SDFG('tiled_2d')
    sdfg.add_array('a', [N, N], dace.float32)
    sdfg.add_array('c', [N, N], dace.float32)
    sdfg.add_scalar('c_slice', dace.float32, transient=True)

    state = sdfg.add_state('compute')

    a_read = state.add_access('a')
    c_slice = state.add_access('c_slice')
    c_write = state.add_access('c')

    ome, omx = state.add_map('outer',
                              dict(tile_i='0:N:16', tile_j='0:N:16'))
    ime, imx = state.add_map(
        'inner',
        dict(i='0:Min(16, N-tile_i)', j='0:Min(16, N-tile_j)'))

    tasklet = state.add_tasklet('add', {'__in1', '__in2'}, {'__out'},
                                '__out = __in1 + __in2')

    state.add_memlet_path(a_read, ome, ime, tasklet,
                          memlet=dace.Memlet('a[i+tile_i, j+tile_j]'),
                          dst_conn='__in1')
    state.add_memlet_path(a_read, ome, ime, tasklet,
                          memlet=dace.Memlet('a[i+tile_i, j+tile_j]'),
                          dst_conn='__in2')
    state.add_edge(tasklet, '__out', c_slice, None,
                   dace.Memlet('c_slice[0]'))
    imx.add_in_connector('IN_c')
    state.add_edge(c_slice, None, imx, 'IN_c',
                   dace.Memlet(data='c', subset='i+tile_i, j+tile_j',
                               other_subset='0'))
    state.add_memlet_path(
        imx, omx, c_write,
        memlet=dace.Memlet(
            'c[tile_i:tile_i+Min(16,N-tile_i), '
            'tile_j:tile_j+Min(16,N-tile_j)]'),
        src_conn='OUT_c')

    return sdfg


# ---------------------------------------------------------------------------
# Positive tests
# ---------------------------------------------------------------------------


def test_scalar_tasklet_to_map_exit():
    """Basic test: scalar intermediate between tasklet and map exit."""
    sdfg = _make_scalar_intermediate_sdfg()

    count = sdfg.apply_transformations(RemoveIntermediateTransient)
    assert count == 1

    state = sdfg.states()[0]
    for node in state.data_nodes():
        assert node.data != 'c_tmp', "c_tmp should have been removed"
    assert 'c_tmp' not in sdfg.arrays, "c_tmp descriptor should be removed"


def test_scalar_tasklet_numerical():
    """Numerical correctness after transformation."""
    sdfg = _make_scalar_intermediate_sdfg()
    sdfg.apply_transformations(RemoveIntermediateTransient)

    N_val = 64
    a = np.random.rand(N_val).astype(np.float32)
    c = np.zeros(N_val, dtype=np.float32)
    sdfg(a=a, c=c, N=N_val)
    np.testing.assert_allclose(c, a + a, rtol=1e-5)


def test_2d_tiled():
    """Tiled 2D maps with scalar intermediate."""
    sdfg = _make_2d_tiled_sdfg()

    count = sdfg.apply_transformations(RemoveIntermediateTransient)
    assert count == 1

    state = sdfg.states()[0]
    for node in state.data_nodes():
        assert node.data != 'c_slice', "c_slice should have been removed"


def test_2d_tiled_numerical():
    """Numerical correctness for tiled 2D case after transformation."""
    sdfg = _make_2d_tiled_sdfg()
    sdfg.apply_transformations(RemoveIntermediateTransient)

    N_val = 64
    a = np.random.rand(N_val, N_val).astype(np.float32)
    c = np.zeros((N_val, N_val), dtype=np.float32)
    sdfg(a=a, c=c, N=N_val)
    np.testing.assert_allclose(c, a + a, rtol=1e-5)


# ---------------------------------------------------------------------------
# Negative tests
# ---------------------------------------------------------------------------


def test_does_not_apply_non_transient():
    """Should not apply if the access node is not transient."""
    sdfg = _make_scalar_intermediate_sdfg()
    sdfg.arrays['c_tmp'].transient = False

    count = sdfg.apply_transformations(RemoveIntermediateTransient)
    assert count == 0


def test_does_not_apply_multi_use():
    """Should not apply if the data is used by multiple access nodes."""
    N = dace.symbol('N')
    sdfg = SDFG('multi_use')
    sdfg.add_array('a', [N], dace.float32)
    sdfg.add_array('c', [N], dace.float32)
    sdfg.add_scalar('c_tmp', dace.float32, transient=True)

    # State 1: the pattern (tasklet → c_tmp → map_exit)
    state1 = sdfg.add_state('compute')
    a_read = state1.add_access('a')
    c_tmp = state1.add_access('c_tmp')
    c_write = state1.add_access('c')

    me, mx = state1.add_map('mymap', dict(i='0:N'))
    tasklet = state1.add_tasklet('add', {'__in1', '__in2'}, {'__out'},
                                 '__out = __in1 + __in2')

    state1.add_memlet_path(a_read, me, tasklet,
                           memlet=dace.Memlet('a[i]'), dst_conn='__in1')
    state1.add_memlet_path(a_read, me, tasklet,
                           memlet=dace.Memlet('a[i]'), dst_conn='__in2')
    state1.add_edge(tasklet, '__out', c_tmp, None, dace.Memlet('c_tmp[0]'))
    mx.add_in_connector('IN_c')
    state1.add_edge(c_tmp, None, mx, 'IN_c',
                    dace.Memlet(data='c', subset='i', other_subset='0'))
    state1.add_memlet_path(mx, c_write,
                           memlet=dace.Memlet('c[i]'), src_conn='OUT_c')

    # State 2: another use of c_tmp (making occurrences > 1)
    state2 = sdfg.add_state('other')
    c_tmp2 = state2.add_access('c_tmp')
    c_write2 = state2.add_access('c')
    t2 = state2.add_tasklet('read', {'__in'}, {'__out'}, '__out = __in')
    state2.add_edge(c_tmp2, None, t2, '__in', dace.Memlet('c_tmp[0]'))
    state2.add_edge(t2, '__out', c_write2, None, dace.Memlet('c[0]'))
    sdfg.add_edge(state1, state2, dace.InterstateEdge())

    count = sdfg.apply_transformations(RemoveIntermediateTransient)
    assert count == 0


def test_does_not_apply_multi_in_degree():
    """Should not apply if access node has more than one incoming edge."""
    N = dace.symbol('N')
    sdfg = SDFG('multi_in')
    sdfg.add_array('a', [N], dace.float32)
    sdfg.add_array('b', [N], dace.float32)
    sdfg.add_array('c', [N], dace.float32)
    sdfg.add_scalar('c_tmp', dace.float32, transient=True)

    state = sdfg.add_state('compute')
    a_read = state.add_access('a')
    b_read = state.add_access('b')
    c_tmp = state.add_access('c_tmp')
    c_write = state.add_access('c')

    me, mx = state.add_map('mymap', dict(i='0:N'))

    t1 = state.add_tasklet('t1', {'__in'}, {'__out'}, '__out = __in')
    t2 = state.add_tasklet('t2', {'__in'}, {'__out'}, '__out = __in')

    state.add_memlet_path(a_read, me, t1,
                          memlet=dace.Memlet('a[i]'), dst_conn='__in')
    state.add_memlet_path(b_read, me, t2,
                          memlet=dace.Memlet('b[i]'), dst_conn='__in')

    state.add_edge(t1, '__out', c_tmp, None, dace.Memlet('c_tmp[0]'))
    state.add_edge(t2, '__out', c_tmp, None, dace.Memlet('c_tmp[0]'))

    mx.add_in_connector('IN_c')
    state.add_edge(c_tmp, None, mx, 'IN_c',
                   dace.Memlet(data='c', subset='i', other_subset='0'))
    state.add_memlet_path(mx, c_write,
                          memlet=dace.Memlet('c[i]'), src_conn='OUT_c')

    count = sdfg.apply_transformations(RemoveIntermediateTransient)
    assert count == 0


def test_does_not_apply_multi_out_degree():
    """Should not apply if access node has more than one outgoing edge."""
    N = dace.symbol('N')
    sdfg = SDFG('multi_out')
    sdfg.add_array('a', [N], dace.float32)
    sdfg.add_array('c', [N], dace.float32)
    sdfg.add_scalar('c_tmp', dace.float32, transient=True)

    state = sdfg.add_state('compute')
    a_read = state.add_access('a')
    c_tmp = state.add_access('c_tmp')
    c_write = state.add_access('c')

    me, mx = state.add_map('mymap', dict(i='0:N'))

    t1 = state.add_tasklet('add', {'__in1', '__in2'}, {'__out'},
                           '__out = __in1 + __in2')

    state.add_memlet_path(a_read, me, t1,
                          memlet=dace.Memlet('a[i]'), dst_conn='__in1')
    state.add_memlet_path(a_read, me, t1,
                          memlet=dace.Memlet('a[i]'), dst_conn='__in2')

    state.add_edge(t1, '__out', c_tmp, None, dace.Memlet('c_tmp[0]'))
    # c_tmp has TWO outgoing edges to the map exit (out_degree > 1)
    mx.add_in_connector('IN_c')
    mx.add_in_connector('IN_c2')
    mx.add_out_connector('OUT_c')
    mx.add_out_connector('OUT_c2')
    state.add_edge(c_tmp, None, mx, 'IN_c',
                   dace.Memlet(data='c', subset='i', other_subset='0'))
    state.add_edge(c_tmp, None, mx, 'IN_c2',
                   dace.Memlet(data='c', subset='i', other_subset='0'))
    state.add_edge(mx, 'OUT_c', c_write, None, dace.Memlet('c[i]'))
    state.add_edge(mx, 'OUT_c2', c_write, None, dace.Memlet('c[i]'))

    count = sdfg.apply_transformations(RemoveIntermediateTransient)
    assert count == 0


def test_does_not_apply_interstate_use():
    """Should not apply if data is used in an interstate edge."""
    sdfg = _make_scalar_intermediate_sdfg()

    state2 = sdfg.add_state('other')
    sdfg.add_edge(sdfg.states()[0], state2,
                  dace.InterstateEdge(condition='c_tmp > 0'))

    count = sdfg.apply_transformations(RemoveIntermediateTransient)
    assert count == 0


# ---------------------------------------------------------------------------
# Feature tests
# ---------------------------------------------------------------------------


def test_wcr_preserved():
    """WCR from the outgoing edge should be preserved in the new memlet."""
    N = dace.symbol('N')
    sdfg = SDFG('wcr_test')
    sdfg.add_array('a', [N], dace.float32)
    sdfg.add_array('c', [N], dace.float32)
    sdfg.add_scalar('c_tmp', dace.float32, transient=True)

    state = sdfg.add_state('compute')
    a_read = state.add_access('a')
    c_tmp = state.add_access('c_tmp')
    c_write = state.add_access('c')

    me, mx = state.add_map('mymap', dict(i='0:N'))
    tasklet = state.add_tasklet('add', {'__in'}, {'__out'}, '__out = __in')

    state.add_memlet_path(a_read, me, tasklet,
                          memlet=dace.Memlet('a[i]'), dst_conn='__in')
    state.add_edge(tasklet, '__out', c_tmp, None,
                   dace.Memlet('c_tmp[0]'))
    mx.add_in_connector('IN_c')
    state.add_edge(c_tmp, None, mx, 'IN_c',
                   dace.Memlet(data='c', subset='i', other_subset='0',
                               wcr='lambda a, b: a + b'))
    state.add_memlet_path(mx, c_write,
                          memlet=dace.Memlet('c[i]',
                                             wcr='lambda a, b: a + b'),
                          src_conn='OUT_c')

    sdfg.apply_transformations(RemoveIntermediateTransient)

    state = sdfg.states()[0]
    for edge in state.edges():
        if (isinstance(edge.src, nodes.Tasklet)
                and isinstance(edge.dst, nodes.MapExit)):
            assert edge.data.wcr is not None, "WCR should be preserved"
            break
    else:
        pytest.fail("Expected edge from Tasklet to MapExit not found")


def test_accessnode_predecessor():
    """AccessNode as the predecessor of the intermediate transient."""
    N = dace.symbol('N')
    sdfg = SDFG('an_predecessor')
    sdfg.add_array('a', [N], dace.float32)
    sdfg.add_array('c', [N], dace.float32)
    sdfg.add_scalar('c_tmp', dace.float32, transient=True)

    state = sdfg.add_state('compute')
    a_read = state.add_access('a')
    a_inner = state.add_access('a')
    c_tmp = state.add_access('c_tmp')
    c_write = state.add_access('c')

    me, mx = state.add_map('mymap', dict(i='0:N'))

    state.add_memlet_path(a_read, me, a_inner, memlet=dace.Memlet('a[i]'))
    state.add_edge(a_inner, None, c_tmp, None,
                   dace.Memlet(data='c_tmp', subset='0', other_subset='i'))
    mx.add_in_connector('IN_c')
    state.add_edge(c_tmp, None, mx, 'IN_c',
                   dace.Memlet(data='c', subset='i', other_subset='0'))
    state.add_memlet_path(mx, c_write,
                          memlet=dace.Memlet('c[i]'), src_conn='OUT_c')

    count = sdfg.apply_transformations(RemoveIntermediateTransient)
    assert count == 1

    state = sdfg.states()[0]
    for node in state.data_nodes():
        assert node.data != 'c_tmp', "c_tmp should have been removed"
