# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for automatic formatting of generated Python-backend code.

The Python/cuTile backend emits long, densely composed lines. These are
reflowed with yapf at code-generation time (see
:func:`dace.codegen.py.prettycode.format_python_code`), so the final ``.code``
of every ``'py'`` code object is already formatted.
"""

import pytest

import dace
from dace import dtypes
from dace.codegen.py.prettycode import format_python_code
from dace.transformation.passes.canonicalize import canonicalize
from dace.transformation.passes.vectorization.vectorize_cutile import VectorizeCuTile

# yapf is an optional (linting-extra) dependency; formatting fails soft without
# it, so the integration assertions only hold when yapf is importable.
yapf = pytest.importorskip("yapf")

# Must match column_limit in the project's .style.yapf.
COLUMN_LIMIT = 120


def test_format_python_code_reflows_long_lines():
    """A long, single-line expression is wrapped to multiple lines."""
    long_arg = "some_scalar_value_with_a_really_long_name_to_force_a_wrap"
    ugly = ("def kernel(a, b):\n"
            f"    result = ct.store(ct.add(ct.load(a, offsets=(i0, i1)), ct.mul(ct.load(b), {long_arg})))\n")
    formatted = format_python_code(ugly)

    assert formatted != ugly
    # The expression must now span more than the original two physical lines.
    assert formatted.count("\n") > ugly.count("\n")
    # Formatting is semantics-preserving: the result still parses.
    compile(formatted, "<formatted>", "exec")


def test_format_python_code_is_idempotent():
    """Formatting already-formatted code is a no-op."""
    src = "x = 1\ny = 2\n"
    once = format_python_code(src)
    assert format_python_code(once) == once


def test_format_python_code_fails_soft_on_invalid_input():
    """Syntactically invalid input is returned unchanged, never raising."""
    bad = "this is (not valid python"
    assert format_python_code(bad) == bad


def _build_vadd_cutile_sdfg():
    """Build and lower a symbolic vector-add SDFG onto the cuTile backend.

    The cuTile codegen emits a ``ct.launch(...)`` call that, unformatted, is a
    single 128-character line — longer than the 120-column limit. This is code
    generation only (no GPU is required to emit the source strings).

    :returns: A Python-backend SDFG ready for ``generate_code()``.
    """
    N = dace.symbol("N")
    sdfg = dace.SDFG("vadd_fmt")
    for name in ("A", "B", "C"):
        sdfg.add_array(name, (N,), dace.float64)
    state = sdfg.add_state("main")
    state.add_mapped_tasklet(
        "add",
        {"i": "0:N"},
        {"_a": dace.Memlet("A[i]"), "_b": dace.Memlet("B[i]")},
        "_c = _a + _b",
        {"_c": dace.Memlet("C[i]")},
        external_edges=True,
    )
    canonicalize(sdfg)
    VectorizeCuTile(widths=(8,)).apply_pass(sdfg, {})
    return sdfg


def test_generated_cutile_code_has_no_overlong_lines():
    """End-to-end: the codegen pipeline reflows generated cuTile code.

    Without the formatting hook the generated ``ct.launch`` statement is a
    single 128-character line; this test fails in that case, so it genuinely
    exercises the integration (not just short, already-fitting lines).
    """
    sdfg = _build_vadd_cutile_sdfg()
    py_objects = [co for co in sdfg.generate_code() if co.language == "py"]
    assert py_objects, "expected at least one Python code object"

    for co in py_objects:
        overlong = [line for line in co.code.split("\n") if len(line) > COLUMN_LIMIT]
        assert not overlong, ("generated code has lines exceeding the column limit "
                              f"(formatting did not run): {overlong}")


def test_generated_cutile_launch_is_wrapped():
    """The long ``ct.launch(...)`` statement is wrapped across physical lines.

    A direct check that the over-long statement was reflowed: unformatted it
    occupies a single physical line; formatted it spans more than one.
    """
    sdfg = _build_vadd_cutile_sdfg()
    frame = next(co for co in sdfg.generate_code() if co.language == "py")
    lines = frame.code.split("\n")

    launch_idx = next((i for i, line in enumerate(lines) if "ct.launch(" in line), None)
    assert launch_idx is not None, "expected a ct.launch(...) call in generated code"

    # The statement continues onto the next physical line(s) until it balances.
    launch_line = lines[launch_idx]
    assert launch_line.count("(") > launch_line.count(")"), \
        "ct.launch(...) was not wrapped onto multiple lines"


if __name__ == "__main__":
    test_format_python_code_reflows_long_lines()
    test_format_python_code_is_idempotent()
    test_format_python_code_fails_soft_on_invalid_input()
    test_generated_cutile_code_has_no_overlong_lines()
    test_generated_cutile_launch_is_wrapped()
