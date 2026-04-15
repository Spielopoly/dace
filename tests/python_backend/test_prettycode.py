# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for PythonCodeIOStream."""

import pytest

from dace.codegen.py.prettycode import PythonCodeIOStream


def test_basic_write():
    s = PythonCodeIOStream()
    s.write("x = 1")
    assert s.getvalue() == "x = 1\n"


def test_indent_dedent():
    s = PythonCodeIOStream()
    s.write("if True:")
    s.indent()
    s.write("x = 1")
    s.dedent()
    s.write("y = 2")
    assert s.getvalue() == "if True:\n    x = 1\ny = 2\n"


def test_indented_context_manager():
    s = PythonCodeIOStream()
    s.write("for i in range(10):")
    with s.indented():
        s.write("print(i)")
    s.write("done()")
    expected = "for i in range(10):\n    print(i)\ndone()\n"
    assert s.getvalue() == expected


def test_nested_indentation():
    s = PythonCodeIOStream()
    s.write("if True:")
    with s.indented():
        s.write("for i in range(10):")
        with s.indented():
            s.write("print(i)")
    expected = "if True:\n    for i in range(10):\n        print(i)\n"
    assert s.getvalue() == expected


def test_empty_lines_in_multiline():
    s = PythonCodeIOStream()
    s.write("x = 1\n\ny = 2")
    result = s.getvalue()
    # Empty line should not get indentation
    assert "\n\n" in result


def test_multiline_write():
    s = PythonCodeIOStream()
    s.indent()
    s.write("x = 1\ny = 2")
    assert s.getvalue() == "    x = 1\n    y = 2\n"


def test_dedent_clamp():
    s = PythonCodeIOStream()
    s.dedent()  # Should not go below 0
    assert s.indent_level == 0
    s.write("x = 1")
    assert s.getvalue() == "x = 1\n"


def test_base_indentation():
    s = PythonCodeIOStream(base_indentation=2)
    s.write("x = 1")
    assert s.getvalue() == "        x = 1\n"  # 8 spaces (2 * 4)


def test_no_brace_counting():
    s = PythonCodeIOStream()
    s.write("d = {'key': 'value'}")
    s.write("x = 1")
    # Braces should NOT affect indentation
    result = s.getvalue()
    assert "d = {'key': 'value'}\n" in result
    assert "x = 1\n" in result
    assert "    x = 1" not in result


def test_location_annotation():
    s = PythonCodeIOStream()

    class MockCFG:
        cfg_id = 0

    s.write("x = 1", cfg=MockCFG(), state_id=0, node_id=1)
    result = s.getvalue()
    assert "# __DACE:0:0:1" in result
    assert result.startswith("x = 1")


def test_location_annotation_with_guid():
    s = PythonCodeIOStream()

    class MockCFG:
        cfg_id = 0

    class MockNode:
        guid = "abc-123"

    s.write("x = 1", cfg=MockCFG(), state_id=0, node_id=MockNode())
    assert "# __DACE:0:0:abc-123" in s.getvalue()


def test_empty_write():
    s = PythonCodeIOStream()
    s.write("")
    s.write(None)
    assert s.getvalue() == ""


def test_relative_indentation_preserved():
    """Multi-line content with internal indentation should be preserved."""
    s = PythonCodeIOStream()
    s.indent()
    # Simulate content from dispatch_state with its own indentation
    content = "for i in range(10):\n    x = i + 1"
    s.write(content)
    result = s.getvalue()
    lines = result.rstrip('\n').split('\n')
    assert lines[0] == "    for i in range(10):"
    assert lines[1] == "        x = i + 1"


def test_trailing_newline_stripped():
    """Content ending with newline should not produce extra blank lines."""
    s = PythonCodeIOStream()
    s.write("x = 1\n")
    s.write("y = 2\n")
    assert s.getvalue() == "x = 1\ny = 2\n"


def test_indent_level_property():
    s = PythonCodeIOStream()
    assert s.indent_level == 0
    s.indent(2)
    assert s.indent_level == 2
    s.dedent()
    assert s.indent_level == 1


def test_indented_restores_on_exception():
    """indented() context manager should restore indentation even on exception."""
    s = PythonCodeIOStream()
    try:
        with s.indented():
            assert s.indent_level == 1
            raise ValueError("test")
    except ValueError:
        pass
    assert s.indent_level == 0


def test_write_numeric_contents():
    """write() should accept non-string types via str() conversion."""
    s = PythonCodeIOStream()
    s.write(42)
    assert s.getvalue() == "42\n"


def test_multiple_indented_levels():
    """Deeply nested indented() calls stack correctly."""
    s = PythonCodeIOStream()
    s.write("level0")
    with s.indented():
        s.write("level1")
        with s.indented():
            s.write("level2")
            with s.indented():
                s.write("level3")
    s.write("back0")
    lines = s.getvalue().strip().split('\n')
    assert lines[0] == "level0"
    assert lines[1] == "    level1"
    assert lines[2] == "        level2"
    assert lines[3] == "            level3"
    assert lines[4] == "back0"


def test_cfg_only_annotation():
    """Location annotation with cfg but no state_id or node_id."""
    s = PythonCodeIOStream()

    class MockCFG:
        cfg_id = 5

    s.write("x = 1", cfg=MockCFG())
    result = s.getvalue()
    assert "# __DACE:5" in result
    # Should NOT have extra colons for state/node
    assert result.strip().endswith("# __DACE:5")


def test_annotation_only_first_line():
    """Location annotation should appear on the first non-empty line only."""
    s = PythonCodeIOStream()

    class MockCFG:
        cfg_id = 0

    s.write("a = 1\nb = 2\nc = 3", cfg=MockCFG(), state_id=0, node_id=1)
    result = s.getvalue()
    lines = result.strip().split('\n')
    assert "# __DACE" in lines[0]
    assert "# __DACE" not in lines[1]
    assert "# __DACE" not in lines[2]


def test_getvalue_returns_all_written():
    """getvalue() returns everything written, including across indent changes."""
    s = PythonCodeIOStream()
    s.write("a = 1")
    s.indent()
    s.write("b = 2")
    s.dedent()
    s.write("c = 3")
    val = s.getvalue()
    assert "a = 1\n" in val
    assert "    b = 2\n" in val
    assert "c = 3\n" in val


def test_indented_with_custom_levels():
    """indented() with levels > 1."""
    s = PythonCodeIOStream()
    s.write("outer")
    with s.indented(2):
        s.write("inner")
    s.write("outer_again")
    lines = s.getvalue().strip().split('\n')
    assert lines[1] == "        inner"  # 2 levels = 8 spaces
    assert lines[2] == "outer_again"
