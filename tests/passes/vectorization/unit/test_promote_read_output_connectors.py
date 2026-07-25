# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for promoting internally-read NestedSDFG outputs to inout connectors."""

import dace
from dace.transformation.passes.vectorization.vectorize_multi_dim import (
    _promote_read_output_connectors_to_inout, )


def test_promoted_output_connector_is_valid_inout() -> None:
    """An output name must also be installed as an input before adding its in-edge."""
    inner = dace.SDFG("inner")
    inner.add_array("seed", (4, ), dace.float64)
    inner.add_array("result", (4, ), dace.float64)
    inner_state = inner.add_state("body", is_start_block=True)
    seed = inner_state.add_read("seed")
    old_result = inner_state.add_read("result")
    result = inner_state.add_write("result")
    add = inner_state.add_tasklet("add", {"_seed", "_old"}, {"_out"}, "_out = _seed + _old")
    inner_state.add_edge(seed, None, add, "_seed", dace.Memlet("seed[0:4]"))
    inner_state.add_edge(old_result, None, add, "_old", dace.Memlet("result[0:4]"))
    inner_state.add_edge(add, "_out", result, None, dace.Memlet("result[0:4]"))

    outer = dace.SDFG("outer")
    outer.add_array("A", (4, ), dace.float64)
    outer_state = outer.add_state("call", is_start_block=True)
    source = outer_state.add_read("A")
    destination = outer_state.add_write("A")
    nested = outer_state.add_nested_sdfg(inner, {"seed"}, {"result"})
    outer_state.add_edge(source, None, nested, "seed", dace.Memlet("A[0:4]"))
    outer_state.add_edge(nested, "result", destination, None, dace.Memlet("A[0:4]"))

    assert _promote_read_output_connectors_to_inout(outer) == 1
    assert "result" in nested.in_connectors
    assert "result" in nested.out_connectors
    promoted_edge = next(edge for edge in outer_state.in_edges(nested) if edge.dst_conn == "result")
    assert promoted_edge.data.data == "A"
    outer.validate()
