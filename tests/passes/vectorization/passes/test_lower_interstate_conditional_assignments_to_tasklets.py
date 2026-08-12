# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for conditional-assignment symbol demotion safety."""

import dace
from dace.transformation.passes.vectorization.lower_interstate_conditional_assignments_to_tasklets import (
    LowerInterstateConditionalAssignmentsToTasklets, )


def test_preserves_public_and_symbolic_context_symbols() -> None:
    """Protect symbolic contexts and preserve a demoted symbol's integer type."""
    sdfg = dace.SDFG("conditional_symbol_contexts")
    for name in ("N", "i", "j", "k"):
        sdfg.add_symbol(name, dace.int64)
    sdfg.add_array("A", [dace.symbol("N")], dace.float64, offset=[dace.symbol("i")])
    sdfg.add_array("shape_uses_j", [dace.symbol("j")], dace.float64, transient=True)

    init = sdfg.add_state("init", is_start_block=True)
    body = sdfg.add_state("body")
    sdfg.add_edge(init, body, dace.InterstateEdge(assignments={"i": "0", "j": "N", "k": "1"}))

    tasklet = body.add_tasklet(
        "condition_symbol_to_scalar_test",
        inputs=set(),
        outputs={"_out"},
        code="_out = N + i + j + k",
    )
    output = body.add_access("A")
    body.add_edge(tasklet, "_out", output, None, dace.Memlet("A[0]"))

    assert "N" in {str(sym) for sym in sdfg.free_symbols}
    original_symbols = dict(sdfg.symbols)

    LowerInterstateConditionalAssignmentsToTasklets().apply_pass(sdfg, {})

    assert {
        name: sdfg.symbols[name]
        for name in ("N", "i", "j")
    } == {
        name: original_symbols[name]
        for name in ("N", "i", "j")
    }
    assert not {"N", "i", "j"} & set(sdfg.arrays)
    assert "k" not in sdfg.symbols
    assert isinstance(sdfg.arrays["k"], dace.data.Scalar)
    assert sdfg.arrays["k"].dtype == dace.int64
    assert sdfg.arrays["k"].transient
    sdfg.validate()
