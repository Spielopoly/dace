# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import dace

from dace.transformation.dataflow.trivial_access_elimination import TrivialAccessNodeElimination


N = 10


def test_trivial_access_node_simple():
    sdfg = dace.SDFG("trivial_access_node_simple")
    sdfg.add_array("A", (N, ), dace.int32)
    sdfg.add_array("C", (N, ), dace.int32)
    sdfg.add_transient("B", (N, ), dace.int32)

    state = sdfg.add_state()
    a_read = state.add_read("A")
    b_access = state.add_access("B")
    c_write = state.add_write("C")

    state.add_edge(a_read, None, b_access, None, dace.Memlet("A[0:N]"))
    state.add_edge(b_access, None, c_write, None, dace.Memlet("C[0:N]"))

    sdfg.validate()
    count = sdfg.apply_transformations_repeated(TrivialAccessNodeElimination)
    assert count == 1

    access_nodes = [n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode)]
    assert len(access_nodes) == 2
    assert b_access not in state.nodes()
    assert len(state.edges_between(a_read, c_write)) == 1


def test_trivial_access_node_with_map_entry_exit():
    sdfg = dace.SDFG("trivial_access_node_with_map_entry_exit")
    sdfg.add_array("A", (N, ), dace.int32)
    sdfg.add_array("C", (N, ), dace.int32)
    sdfg.add_transient("B", (N, ), dace.int32)

    state = sdfg.add_state()
    a_read = state.add_read("A")
    b_access = state.add_access("B")
    c_write = state.add_write("C")

    me, mx = state.add_map("m", dict(i="0:N"))

    state.add_memlet_path(a_read, me, b_access, memlet=dace.Memlet("A[i]"))
    state.add_memlet_path(b_access, mx, c_write, memlet=dace.Memlet("C[i]"))

    sdfg.validate()
    count = sdfg.apply_transformations_repeated(TrivialAccessNodeElimination)
    assert count == 1

    assert b_access not in state.nodes()
    assert len(state.edges_between(me, mx)) == 1


if __name__ == '__main__':
    test_trivial_access_node_simple()
    test_trivial_access_node_with_map_entry_exit()
