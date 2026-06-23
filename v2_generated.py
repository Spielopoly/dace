# DaCe AUTO-GENERATED FILE. DO NOT MODIFY
import cuda.tile as ct
import cupy
import numpy
from dataclasses import dataclass
from sympy_function_redefinitions import *
ONE = 1
__dace_persistent_transients = {}

@ct.kernel
def __dace_cutile_icon_zekinh_gather_0_0_5(gpu_e_bln, gpu_edge_blk, gpu_edge_idx, gpu_z_kin_hor_e, gpu_z_ekinh, NB, NLEV, NPROMA):
    __pid0 = ct.bid(0)
    __pid1 = ct.bid(1)
    __pid2 = ct.bid(2)
    jb = __pid0
    jk = (0 + __pid1 * 8)
    jc = (0 + __pid2 * 8)

    ####### Tasklet: _tile_iter_mask_gen_cutile

    __pid0 = ct.bid(1)
    __pid1 = ct.bid(2)
    __offsets0 = ct.arange(8, dtype=ct.int32)
    __mask0 = ((__offsets0 + (__pid0 * 8)) < (8 * NLEV))
    __offsets1 = ct.arange(8, dtype=ct.int32)
    __mask1 = ((__offsets1 + (__pid1 * 8)) < (8 * NPROMA))
    _o = (ct.broadcast_to(__mask0[:, None], (8, 8)) & ct.broadcast_to(__mask1[None, :], (8, 8)))

    ####### End of tasklet: _tile_iter_mask_gen_cutile

    _tile_iter_mask = _o
    _tile_iter_mask = _o
    _mask = _tile_iter_mask
    _src = gpu_e_bln

    ####### Tasklet: load_e_bln_tile_cutile

    __pid0 = ct.bid(1)
    __pid1 = ct.bid(2)
    _dst = ct.where(_mask, ct.gather(_src, (jb, 0, ct.broadcast_to(((ct.arange(8, dtype=ct.int32) + (__pid1 * 8)) * 1)[None, :], (8, 8))), padding_value=0), 0)

    ####### End of tasklet: load_e_bln_tile_cutile

    e_bln_tile = _dst
    e_bln_tile = _dst
    e_bln_index = e_bln_tile
    _mask = _tile_iter_mask
    _src = gpu_e_bln

    ####### Tasklet: load_e_bln_tile_0_cutile

    __pid0 = ct.bid(1)
    __pid1 = ct.bid(2)
    _dst = ct.where(_mask, ct.gather(_src, (jb, 1, ct.broadcast_to(((ct.arange(8, dtype=ct.int32) + (__pid1 * 8)) * 1)[None, :], (8, 8))), padding_value=0), 0)

    ####### End of tasklet: load_e_bln_tile_0_cutile

    e_bln_tile_0 = _dst
    e_bln_tile_0 = _dst
    e_bln_index_0 = e_bln_tile_0
    _mask = _tile_iter_mask
    _src = gpu_e_bln

    ####### Tasklet: load_e_bln_tile_1_cutile

    __pid0 = ct.bid(1)
    __pid1 = ct.bid(2)
    _dst = ct.where(_mask, ct.gather(_src, (jb, 2, ct.broadcast_to(((ct.arange(8, dtype=ct.int32) + (__pid1 * 8)) * 1)[None, :], (8, 8))), padding_value=0), 0)

    ####### End of tasklet: load_e_bln_tile_1_cutile

    e_bln_tile_1 = _dst
    e_bln_tile_1 = _dst
    e_bln_index_1 = e_bln_tile_1
    _src = gpu_edge_blk

    ####### Tasklet: load__idx_z_kin_hor_e_0_load0_cutile

    __pid0 = ct.bid(2)
    _dst = ct.reshape(ct.permute(ct.load(_src, index=(jb, __pid0, 0), shape=(1, 8, 1), padding_mode=ct.PaddingMode.ZERO), axes=(0, 2, 1)), (1, 8))

    ####### End of tasklet: load__idx_z_kin_hor_e_0_load0_cutile

    _idx_z_kin_hor_e_0_load0 = _dst
    _idx_z_kin_hor_e_0_load0 = _dst
    _src = gpu_edge_idx

    ####### Tasklet: load__idx_z_kin_hor_e_2_load0_cutile

    __pid0 = ct.bid(2)
    _dst = ct.reshape(ct.permute(ct.load(_src, index=(jb, __pid0, 0), shape=(1, 8, 1), padding_mode=ct.PaddingMode.ZERO), axes=(0, 2, 1)), (1, 8))

    ####### End of tasklet: load__idx_z_kin_hor_e_2_load0_cutile

    _idx_z_kin_hor_e_2_load0 = _dst
    _idx_z_kin_hor_e_2_load0 = _dst
    _idx_0 = _idx_z_kin_hor_e_0_load0
    _idx_2 = _idx_z_kin_hor_e_2_load0
    _mask = _tile_iter_mask
    _src = gpu_z_kin_hor_e

    ####### Tasklet: load_z_kin_hor_e_gather_cutile

    __pid0 = ct.bid(1)
    __pid1 = ct.bid(2)
    _dst = ct.gather(_src, (_idx_0, ct.broadcast_to((ct.arange(8, dtype=ct.int32) + (__pid0 * 8))[:, None], (8, 8)), _idx_2), padding_value=0, mask=_mask)

    ####### End of tasklet: load_z_kin_hor_e_gather_cutile

    z_kin_hor_e_gather = _dst
    z_kin_hor_e_gather = _dst
    z_kin_hor_e_index = z_kin_hor_e_gather
    _b = z_kin_hor_e_index
    _a = e_bln_index
    _mask = _tile_iter_mask

    ####### Tasklet: _Mult__binop_cutile

    _c = ct.where(_mask, (_a * _b), False)

    ####### End of tasklet: _Mult__binop_cutile

    e_bln_slice_times_z_kin_hor_e_slice = _c
    e_bln_slice_times_z_kin_hor_e_slice = _c
    _src = gpu_edge_blk

    ####### Tasklet: load__idx_z_kin_hor_e_0_load0_0_cutile

    __pid0 = ct.bid(2)
    _dst = ct.reshape(ct.permute(ct.load(_src, index=(jb, __pid0, 1), shape=(1, 8, 1), padding_mode=ct.PaddingMode.ZERO), axes=(0, 2, 1)), (1, 8))

    ####### End of tasklet: load__idx_z_kin_hor_e_0_load0_0_cutile

    _idx_z_kin_hor_e_0_load0_0 = _dst
    _idx_z_kin_hor_e_0_load0_0 = _dst
    _src = gpu_edge_idx

    ####### Tasklet: load__idx_z_kin_hor_e_2_load0_0_cutile

    __pid0 = ct.bid(2)
    _dst = ct.reshape(ct.permute(ct.load(_src, index=(jb, __pid0, 1), shape=(1, 8, 1), padding_mode=ct.PaddingMode.ZERO), axes=(0, 2, 1)), (1, 8))

    ####### End of tasklet: load__idx_z_kin_hor_e_2_load0_0_cutile

    _idx_z_kin_hor_e_2_load0_0 = _dst
    _idx_z_kin_hor_e_2_load0_0 = _dst
    _idx_0 = _idx_z_kin_hor_e_0_load0_0
    _idx_2 = _idx_z_kin_hor_e_2_load0_0
    _mask = _tile_iter_mask
    _src = gpu_z_kin_hor_e

    ####### Tasklet: load_z_kin_hor_e_gather_0_cutile

    __pid0 = ct.bid(1)
    __pid1 = ct.bid(2)
    _dst = ct.gather(_src, (_idx_0, ct.broadcast_to((ct.arange(8, dtype=ct.int32) + (__pid0 * 8))[:, None], (8, 8)), _idx_2), padding_value=0, mask=_mask)

    ####### End of tasklet: load_z_kin_hor_e_gather_0_cutile

    z_kin_hor_e_gather_0 = _dst
    z_kin_hor_e_gather_0 = _dst
    z_kin_hor_e_index_0 = z_kin_hor_e_gather_0
    _a = e_bln_index_0
    _b = z_kin_hor_e_index_0
    _mask = _tile_iter_mask

    ####### Tasklet: _Mult__binop_cutile

    _c = ct.where(_mask, (_a * _b), False)

    ####### End of tasklet: _Mult__binop_cutile

    e_bln_slice_times_z_kin_hor_e_slice_0 = _c
    e_bln_slice_times_z_kin_hor_e_slice_0 = _c
    _a = e_bln_slice_times_z_kin_hor_e_slice
    _b = e_bln_slice_times_z_kin_hor_e_slice_0
    _mask = _tile_iter_mask

    ####### Tasklet: _Add__binop_cutile

    _c = ct.where(_mask, (_a + _b), False)

    ####### End of tasklet: _Add__binop_cutile

    e_bln_slice_z_kin_hor_e_slice_plus_e_bln_slice_z_kin_hor_e_slice = _c
    e_bln_slice_z_kin_hor_e_slice_plus_e_bln_slice_z_kin_hor_e_slice = _c
    _src = gpu_edge_blk

    ####### Tasklet: load__idx_z_kin_hor_e_0_load0_1_cutile

    __pid0 = ct.bid(2)
    _dst = ct.reshape(ct.permute(ct.load(_src, index=(jb, __pid0, 2), shape=(1, 8, 1), padding_mode=ct.PaddingMode.ZERO), axes=(0, 2, 1)), (1, 8))

    ####### End of tasklet: load__idx_z_kin_hor_e_0_load0_1_cutile

    _idx_z_kin_hor_e_0_load0_1 = _dst
    _idx_z_kin_hor_e_0_load0_1 = _dst
    _src = gpu_edge_idx

    ####### Tasklet: load__idx_z_kin_hor_e_2_load0_1_cutile

    __pid0 = ct.bid(2)
    _dst = ct.reshape(ct.permute(ct.load(_src, index=(jb, __pid0, 2), shape=(1, 8, 1), padding_mode=ct.PaddingMode.ZERO), axes=(0, 2, 1)), (1, 8))

    ####### End of tasklet: load__idx_z_kin_hor_e_2_load0_1_cutile

    _idx_z_kin_hor_e_2_load0_1 = _dst
    _idx_z_kin_hor_e_2_load0_1 = _dst
    _idx_0 = _idx_z_kin_hor_e_0_load0_1
    _idx_2 = _idx_z_kin_hor_e_2_load0_1
    _mask = _tile_iter_mask
    _src = gpu_z_kin_hor_e

    ####### Tasklet: load_z_kin_hor_e_gather_1_cutile

    __pid0 = ct.bid(1)
    __pid1 = ct.bid(2)
    _dst = ct.gather(_src, (_idx_0, ct.broadcast_to((ct.arange(8, dtype=ct.int32) + (__pid0 * 8))[:, None], (8, 8)), _idx_2), padding_value=0, mask=_mask)

    ####### End of tasklet: load_z_kin_hor_e_gather_1_cutile

    z_kin_hor_e_gather_1 = _dst
    z_kin_hor_e_gather_1 = _dst
    z_kin_hor_e_index_1 = z_kin_hor_e_gather_1
    _a = e_bln_index_1
    _b = z_kin_hor_e_index_1
    _mask = _tile_iter_mask

    ####### Tasklet: _Mult__binop_cutile

    _c = ct.where(_mask, (_a * _b), False)

    ####### End of tasklet: _Mult__binop_cutile

    e_bln_slice_times_z_kin_hor_e_slice_1 = _c
    e_bln_slice_times_z_kin_hor_e_slice_1 = _c
    _a = e_bln_slice_z_kin_hor_e_slice_plus_e_bln_slice_z_kin_hor_e_slice
    _b = e_bln_slice_times_z_kin_hor_e_slice_1
    _mask = _tile_iter_mask

    ####### Tasklet: _Add__binop_cutile

    _c = ct.where(_mask, (_a + _b), False)

    ####### End of tasklet: _Add__binop_cutile

    z_ekinh_tile_out = _c
    z_ekinh_tile_out = _c
    _src = z_ekinh_tile_out
    _mask = _tile_iter_mask
    _dst = gpu_z_ekinh

    ####### Tasklet: store_z_ekinh_tile_out_cutile

    __pid0 = ct.bid(1)
    __pid1 = ct.bid(2)
    __idx0 = ct.broadcast_to((ct.arange(8, dtype=ct.int32) + (__pid0 * 8))[:, None], (8, 8))
    __idx1 = ct.broadcast_to((ct.arange(8, dtype=ct.int32) + (__pid1 * 8))[None, :], (8, 8))
    ct.scatter(_dst, (jb, __idx0, __idx1), _src, mask=_mask)

    ####### End of tasklet: store_z_ekinh_tile_out_cutile



def icon_zekinh_gather(e_bln, edge_blk, edge_idx, z_ekinh, z_kin_hor_e, NB, NLEV, NPROMA):
    gpu_z_kin_hor_e = cupy.empty(((8*NB), (8*NLEV), (8*NPROMA)), dtype=numpy.float64)
    gpu_edge_idx = cupy.empty(((8*NB), (8*NPROMA), 3), dtype=numpy.intc)
    gpu_e_bln = cupy.empty(((8*NB), 3, (8*NPROMA)), dtype=numpy.float64)
    gpu_edge_blk = cupy.empty(((8*NB), (8*NPROMA), 3), dtype=numpy.intc)
    gpu_z_ekinh = cupy.empty(((8*NB), (8*NLEV), (8*NPROMA)), dtype=numpy.float64)

    gpu_e_bln[:].set(e_bln[:(8*NB), :3, :(8*NPROMA)])
    gpu_z_kin_hor_e[:].set(z_kin_hor_e[:(8*NB), :(8*NLEV), :(8*NPROMA)])
    gpu_edge_idx[:].set(edge_idx[:(8*NB), :(8*NPROMA), :3])
    gpu_edge_blk[:].set(edge_blk[:(8*NB), :(8*NPROMA), :3])
    gpu_z_ekinh[:].set(z_ekinh[:(8*NB), :(8*NLEV), :(8*NPROMA)])

    ct.launch(cupy.cuda.get_current_stream(), ((8*NB), NLEV, NPROMA), __dace_cutile_icon_zekinh_gather_0_0_5, (gpu_e_bln, gpu_edge_blk, gpu_edge_idx, gpu_z_kin_hor_e, gpu_z_ekinh, NB, NLEV, NPROMA))
    cupy.cuda.get_current_stream().synchronize()
    gpu_z_ekinh.get(out=z_ekinh[:(8*NB), :(8*NLEV), :(8*NPROMA)])
    del gpu_z_kin_hor_e
    del gpu_edge_idx
    del gpu_e_bln
    del gpu_edge_blk
    del gpu_z_ekinh
