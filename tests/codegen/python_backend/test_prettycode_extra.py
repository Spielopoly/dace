# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Additional unit tests for PythonCodeIOStream covering branches not in test_prettycode.py."""

import pytest
from unittest.mock import MagicMock

from dace.config import set_temporary
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.sdfg.graph import NodeNotFoundError


# ---------------------------------------------------------------------------
# lineinfo enabled
# ---------------------------------------------------------------------------

def test_lineinfo_enabled():
    """Enable codegen_lineinfo → #__CODEGEN annotation appears in output."""
    with set_temporary('compiler', 'codegen_lineinfo', value=True):
        s = PythonCodeIOStream()
        s.write("x = 1")
        output = s.getvalue()
        assert "#__CODEGEN" in output
        # Should contain filename:lineno of the caller
        assert "test_prettycode_extra.py" in output


# ---------------------------------------------------------------------------
# node_id handling
# ---------------------------------------------------------------------------

def test_node_id_list_with_integers():
    """write() with node_id=[0, 1] → comma-separated annotation."""
    cfg = MagicMock()
    cfg.cfg_id = 0
    s = PythonCodeIOStream()
    s.write("x = 1", cfg=cfg, state_id=0, node_id=[0, 1])
    output = s.getvalue()
    assert "#__DACE:0:0:0,1" in output


def test_node_id_list_with_node_objects():
    """Non-int node objects resolved via state.node_id()."""
    cfg = MagicMock()
    cfg.cfg_id = 0
    mock_node = MagicMock()
    mock_state = MagicMock()
    mock_state.node_id.return_value = 7
    cfg.state.return_value = mock_state

    s = PythonCodeIOStream()
    s.write("x = 1", cfg=cfg, state_id=0, node_id=[mock_node])
    output = s.getvalue()
    assert "#__DACE:0:0:7" in output
    mock_state.node_id.assert_called_once_with(mock_node)


def test_node_id_not_found():
    """NodeNotFoundError → -1 in annotation."""
    cfg = MagicMock()
    cfg.cfg_id = 0
    mock_node = MagicMock()
    mock_state = MagicMock()
    mock_state.node_id.side_effect = NodeNotFoundError("not found")
    cfg.state.return_value = mock_state

    s = PythonCodeIOStream()
    s.write("x = 1", cfg=cfg, state_id=0, node_id=[mock_node])
    output = s.getvalue()
    assert "#__DACE:0:0:-1" in output


def test_node_id_single_non_int():
    """Single non-int node_id wraps into list and resolves."""
    cfg = MagicMock()
    cfg.cfg_id = 0
    mock_node = MagicMock()
    mock_state = MagicMock()
    mock_state.node_id.return_value = 3
    cfg.state.return_value = mock_state

    s = PythonCodeIOStream()
    s.write("x = 1", cfg=cfg, state_id=0, node_id=mock_node)
    output = s.getvalue()
    assert "#__DACE:0:0:3" in output


# ---------------------------------------------------------------------------
# spaces config
# ---------------------------------------------------------------------------

def test_spaces_config_zero():
    """_spaces <= 0 defaults to 4."""
    with set_temporary('compiler', 'indentation_spaces', value=0):
        s = PythonCodeIOStream()
        assert s._spaces == 4
        s.indent()
        s.write("x = 1")
        assert s.getvalue() == "    x = 1\n"


def test_spaces_config_negative():
    """Negative _spaces defaults to 4."""
    with set_temporary('compiler', 'indentation_spaces', value=-2):
        s = PythonCodeIOStream()
        assert s._spaces == 4


# ---------------------------------------------------------------------------
# write with all location params
# ---------------------------------------------------------------------------

def test_write_with_all_location_params():
    """cfg + state_id + node_id complete annotation."""
    cfg = MagicMock()
    cfg.cfg_id = 5
    s = PythonCodeIOStream()
    s.write("y = 2", cfg=cfg, state_id=3, node_id=[42])
    output = s.getvalue()
    assert "#__DACE:5:3:42" in output


def test_write_cfg_only():
    """cfg without state_id → only cfg_id in annotation."""
    cfg = MagicMock()
    cfg.cfg_id = 2
    s = PythonCodeIOStream()
    s.write("z = 3", cfg=cfg)
    output = s.getvalue()
    assert "#__DACE:2" in output
    # No state_id or node_id suffix
    assert "#__DACE:2:" not in output


def test_write_cfg_and_state_id_no_node():
    """cfg + state_id but no node_id → annotation stops at state_id."""
    cfg = MagicMock()
    cfg.cfg_id = 1
    s = PythonCodeIOStream()
    s.write("w = 4", cfg=cfg, state_id=9)
    output = s.getvalue()
    assert "#__DACE:1:9" in output


# ---------------------------------------------------------------------------
# indented context manager yields stream
# ---------------------------------------------------------------------------

def test_indented_yields_stream():
    """with stream.indented() yields the stream itself."""
    s = PythonCodeIOStream()
    with s.indented() as inner:
        assert inner is s
        assert s.indent_level == 1
    assert s.indent_level == 0


def test_indented_multiple_levels():
    """indented(levels=3) increases and decreases by 3."""
    s = PythonCodeIOStream()
    with s.indented(levels=3):
        assert s.indent_level == 3
        s.write("deep")
    assert s.indent_level == 0
    output = s.getvalue()
    # 3 levels * 4 spaces = 12 spaces
    assert output.startswith(" " * 12 + "deep")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_write_none():
    """write(None) produces no output."""
    s = PythonCodeIOStream()
    s.write(None)
    assert s.getvalue() == ""


def test_write_empty_string():
    """write('') produces no output."""
    s = PythonCodeIOStream()
    s.write("")
    assert s.getvalue() == ""


def test_write_non_string():
    """write() with non-string value converts via str()."""
    s = PythonCodeIOStream()
    s.write(42)
    assert s.getvalue() == "42\n"


def test_lineinfo_disabled_by_default():
    """By default, #__CODEGEN does not appear."""
    with set_temporary('compiler', 'codegen_lineinfo', value=False):
        s = PythonCodeIOStream()
        s.write("x = 1")
        assert "#__CODEGEN" not in s.getvalue()
