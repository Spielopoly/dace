# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace

from dace.sdfg import nodes
from dace.transformation.dataflow.trivial_chain_elimination import TrivialChainElimination


def _assert_no_direct_tasklet_edges(state: dace.SDFGState):
    for edge in state.edges():
        assert not (isinstance(edge.src, nodes.Tasklet) and isinstance(edge.dst, nodes.Tasklet))


def test_access_before_tasklet_chain_elimination():
    sdfg = dace.SDFG("access_before_tasklet_chain")
    sdfg.add_symbol("s", dace.int32)
    sdfg.add_array("A", (1, ), dace.int32)

    state = sdfg.add_state()

    tmp_before, _ = sdfg.add_scalar("tmp_before", dace.int32, transient=True)
    tmp_after, _ = sdfg.add_scalar("tmp_after", dace.int32, transient=True)

    producer = state.add_tasklet("producer", {}, {"out"}, "out = s")
    before_access = state.add_access(tmp_before)
    trivial_copy = state.add_tasklet("copy", {"inp"}, {"out"}, "out = inp")
    after_access = state.add_access(tmp_after)
    sink = state.add_tasklet("sink", {"inp"}, {"out"}, "out = inp")
    a_write = state.add_write("A")

    state.add_edge(producer, "out", before_access, None, dace.Memlet(before_access.data))
    state.add_edge(before_access, None, trivial_copy, "inp", dace.Memlet(before_access.data))
    state.add_edge(trivial_copy, "out", after_access, None, dace.Memlet(after_access.data))
    state.add_edge(after_access, None, sink, "inp", dace.Memlet(after_access.data))
    state.add_edge(sink, "out", a_write, None, dace.Memlet("A[0]"))

    sdfg.validate()
    count = sdfg.apply_transformations([TrivialChainElimination])
    assert count == 1

    assert before_access not in state.nodes()
    assert trivial_copy not in state.nodes()
    assert len(state.edges_between(producer, after_access)) == 1
    _assert_no_direct_tasklet_edges(state)


def test_tasklet_before_access_chain_elimination():
    sdfg = dace.SDFG("tasklet_before_access_chain")
    sdfg.add_array("A", (1, ), dace.int32)
    sdfg.add_array("B", (1, ), dace.int32)

    state = sdfg.add_state()

    a_read = state.add_read("A")
    trivial_copy = state.add_tasklet("copy", {"inp"}, {"out"}, "out = inp")
    middle_access, _ = sdfg.add_scalar("middle", dace.int32, transient=True)
    middle = state.add_access(middle_access)
    b_write = state.add_write("B")

    state.add_edge(a_read, None, trivial_copy, "inp", dace.Memlet("A[0]"))
    state.add_edge(trivial_copy, "out", middle, None, dace.Memlet(middle.data))
    state.add_edge(middle, None, b_write, None, dace.Memlet("B[0]"))

    sdfg.validate()
    count = sdfg.apply_transformations([TrivialChainElimination])
    assert count == 1

    assert trivial_copy not in state.nodes()
    assert middle not in state.nodes()
    assert len(state.edges_between(a_read, b_write)) == 1
    _assert_no_direct_tasklet_edges(state)


def test_nontrivial_tasklet_not_eliminated():
    sdfg = dace.SDFG("nontrivial_tasklet_chain")
    sdfg.add_array("A", (1, ), dace.int32)
    sdfg.add_array("B", (1, ), dace.int32)

    state = sdfg.add_state()

    a_read = state.add_read("A")
    nontrivial = state.add_tasklet("copy_plus", {"inp"}, {"out"}, "out = inp + 1")
    middle_access, _ = sdfg.add_scalar("middle2", dace.int32, transient=True)
    middle = state.add_access(middle_access)
    b_write = state.add_write("B")

    state.add_edge(a_read, None, nontrivial, "inp", dace.Memlet("A[0]"))
    state.add_edge(nontrivial, "out", middle, None, dace.Memlet(middle.data))
    state.add_edge(middle, None, b_write, None, dace.Memlet("B[0]"))

    sdfg.validate()
    count = sdfg.apply_transformations([TrivialChainElimination])
    assert count == 0


if __name__ == '__main__':
    test_access_before_tasklet_chain_elimination()
    test_tasklet_before_access_chain_elimination()
    test_nontrivial_tasklet_not_eliminated()
