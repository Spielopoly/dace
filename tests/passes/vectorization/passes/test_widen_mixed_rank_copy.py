# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression coverage for widening one endpoint of a mixed-rank AccessNode copy."""

import dace
from dace import subsets
from dace.transformation.passes.vectorization.widen_accesses import WidenAccesses


def test_widen_transient_rewrites_other_subset_when_memlet_names_destination():
    """The transient endpoint may be represented by ``other_subset``."""
    sdfg = dace.SDFG("widen_mixed_rank_copy")
    sdfg.add_symbol("i", dace.int64)
    sdfg.add_symbol("j", dace.int64)
    sdfg.add_array("bridge", (1, 1), dace.float64, transient=True)
    sdfg.add_array("output", (64, 64), dace.float64)
    state = sdfg.add_state()
    bridge = state.add_access("bridge")
    output = state.add_access("output")
    edge = state.add_edge(bridge, None, output, None,
                          dace.Memlet(data="output", subset="i, j:j + 32", other_subset="0, 0"))

    changed = WidenAccesses(widths=(32, ))._widen_transient(sdfg, "bridge", {"bridge"})

    assert changed
    assert tuple(sdfg.arrays["bridge"].shape) == (32, )
    assert edge.data.subset == subsets.Range.from_string("i, j:j + 32")
    assert edge.data.other_subset == subsets.Range.from_string("0:32")
    assert edge.data.volume == 32
    sdfg.validate()


def test_direct_tile_write_is_staged_when_output_is_also_read():
    """An in-place read must not leave a direct tile-to-global copy."""
    from dace.libraries.tileops import TileStore
    from dace.transformation.passes.vectorization.insert_tile_load_store import InsertTileLoadStore

    sdfg = dace.SDFG("stage_mixed_rank_copy")
    sdfg.add_symbol("i", dace.int64)
    sdfg.add_symbol("j", dace.int64)
    sdfg.add_array("bridge", (32, ), dace.float64, transient=True)
    sdfg.add_array("output", (64, 64), dace.float64)
    state = sdfg.add_state()
    bridge = state.add_access("bridge")
    output = state.add_access("output")
    state.add_edge(bridge, None, output, None, dace.Memlet(data="output", subset="i, j:j + 32", other_subset="0:32"))
    reader = state.add_tasklet("reader", {"value"}, set(), "pass")
    state.add_edge(output, None, reader, "value", dace.Memlet("output[i, j]"))

    staging = InsertTileLoadStore(widths=(32, ))
    assert staging._stage_reads_in_state(state, sdfg, ("j", ), None) == 1
    assert staging._stage_writes_in_state(state, sdfg, ("j", ), None) == 1

    stores = [node for node in state.nodes() if isinstance(node, TileStore)]
    assert len(stores) == 1
    store = stores[0]
    assert state.edges_between(bridge, output) == []
    assert len(state.edges_between(bridge, store)) == 1
    assert len(state.edges_between(store, output)) == 1
    sdfg.validate()


def test_singleton_array_write_broadcasts_through_tileload():
    """A singleton Array producer is splatted into the TileStore bridge."""
    from dace.libraries.tileops import TileLoad, TileStore
    from dace.transformation.passes.vectorization.insert_tile_load_store import InsertTileLoadStore

    sdfg = dace.SDFG("stage_singleton_array_broadcast")
    sdfg.add_symbol("i", dace.int64)
    sdfg.add_symbol("j", dace.int64)
    sdfg.add_array("singleton", (1, 1), dace.float64, transient=True)
    sdfg.add_array("output", (64, 64), dace.float64)
    state = sdfg.add_state()
    singleton = state.add_access("singleton")
    output = state.add_access("output")
    state.add_edge(singleton, None, output, None, dace.Memlet(data="output", subset="i, j:j + 32", other_subset="0, 0"))

    staged = InsertTileLoadStore(widths=(32, ))._stage_writes_in_state(state, sdfg, ("j", ), None)

    loads = [node for node in state.nodes() if isinstance(node, TileLoad)]
    stores = [node for node in state.nodes() if isinstance(node, TileStore)]
    assert staged == 1
    assert len(loads) == 1 and loads[0].src_kind == "Scalar"
    assert len(stores) == 1
    assert not any(
        isinstance(edge.src, dace.nodes.AccessNode) and isinstance(edge.dst, dace.nodes.AccessNode)
        for edge in state.edges())
    sdfg.validate()
