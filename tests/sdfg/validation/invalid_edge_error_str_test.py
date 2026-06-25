# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Regression tests for ``InvalidSDFGEdgeError.__str__`` robustness.

The formatter must never raise: a crash inside ``__str__`` masks the real
validation message. The edge/state ids can be stale if the graph was mutated
between raising and formatting the error (e.g. a pass validates, fails, and the
message is built later by a wrapping consumer). These tests use only core
``main`` APIs.
"""
import pytest

import dace
from dace.sdfg.validation import InvalidSDFGEdgeError


def _sdfg_with_one_state() -> dace.SDFG:
    sdfg = dace.SDFG("err_str_test")
    sdfg.add_state("s0")
    return sdfg


def test_str_with_out_of_range_edge_id_does_not_crash():
    """A stale (out-of-range) edge_id formats gracefully and keeps the message."""
    sdfg = _sdfg_with_one_state()
    msg = "Memlet subset does not match node dimension (expected 1, got 2)"
    err = InvalidSDFGEdgeError(msg, sdfg, 0, 23)  # state 0 has no edges
    s = str(err)  # must not raise
    assert msg in s
    assert "no longer present" in s


def test_str_with_stale_state_id_does_not_crash():
    """A stale state_id formats gracefully and keeps the message."""
    sdfg = _sdfg_with_one_state()
    err = InvalidSDFGEdgeError("some validation message", sdfg, 99, None)
    s = str(err)  # must not raise
    assert "some validation message" in s


def test_str_with_valid_edge_unchanged():
    """When indices are valid, the edge is described as before."""
    sdfg = dace.SDFG("err_str_valid")
    sdfg.add_array("A", (4,), dace.float64)
    sdfg.add_array("B", (4,), dace.float64)
    state = sdfg.add_state("s0")
    a = state.add_access("A")
    b = state.add_access("B")
    state.add_edge(a, None, b, None, dace.Memlet("A[0:4]"))
    err = InvalidSDFGEdgeError("boom", sdfg, sdfg.node_id(state), 0)
    s = str(err)
    assert "boom" in s
    assert "no longer present" not in s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
