# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``__str__`` rendering of validation errors.

Pins that the stale-id branches of ``InvalidSDFGNodeError`` /
``InvalidSDFGEdgeError`` fall through to the common suffix logic — in
particular the "saved for inspection" path suffix must not be skipped —
and that the fresh-id branches still render node/edge details.
"""
import os

import dace
from dace.sdfg.validation import InvalidSDFGEdgeError, InvalidSDFGNodeError


def _make_sdfg() -> dace.SDFG:
    """One-state SDFG with a single tasklet for the fresh-id cases."""
    sdfg = dace.SDFG("error_str_sdfg")
    state = sdfg.add_state("s0")
    state.add_tasklet("t", {}, {}, "pass")
    return sdfg


def test_node_error_stale_state_id_keeps_path_suffix(tmp_path):
    sdfg = _make_sdfg()
    err = InvalidSDFGNodeError("bad node", sdfg, 99, 5)
    err.path = str(tmp_path / "invalid.sdfgz")
    text = str(err)
    assert "bad node" in text
    assert "state with id 99" in text
    assert "node with id 5" in text
    assert f"Invalid SDFG saved for inspection in {os.path.abspath(err.path)}" in text


def test_node_error_stale_state_id_without_path_has_no_suffix():
    sdfg = _make_sdfg()
    err = InvalidSDFGNodeError("bad node", sdfg, 99, None)
    text = str(err)
    assert text == "bad node (at state with id 99)"


def test_node_error_fresh_ids_render_node_and_path_suffix(tmp_path):
    sdfg = _make_sdfg()
    err = InvalidSDFGNodeError("bad node", sdfg, 0, 0)
    err.path = str(tmp_path / "invalid.sdfgz")
    text = str(err)
    assert "at state s0" in text
    assert "node t" in text
    assert "saved for inspection" in text


def test_edge_error_stale_state_id_keeps_path_suffix(tmp_path):
    sdfg = _make_sdfg()
    err = InvalidSDFGEdgeError("bad edge", sdfg, 42, 7)
    err.path = str(tmp_path / "invalid.sdfgz")
    text = str(err)
    assert "bad edge" in text
    assert "state with id 42" in text
    assert "edge with id 7" in text
    assert f"Invalid SDFG saved for inspection in {os.path.abspath(err.path)}" in text


def test_edge_error_stale_edge_id_keeps_path_suffix(tmp_path):
    """Fresh state, stale edge id: the numeric edge fallback keeps the suffix."""
    sdfg = _make_sdfg()
    err = InvalidSDFGEdgeError("bad edge", sdfg, 0, 7)
    err.path = str(tmp_path / "invalid.sdfgz")
    text = str(err)
    assert "at state s0" in text
    assert "edge with id 7" in text
    assert "saved for inspection" in text


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
