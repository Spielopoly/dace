# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``restore_sdfg_in_place`` — the ``VectorizeUnsupported`` refusal rollback.

The restore adopts a snapshot's graph contents into the caller-owned SDFG
object. The adopted structure is internally consistent with the SNAPSHOT as
its root, so every pointer aimed at the snapshot object itself must be swapped
to the target — including on blocks nested inside LoopRegions and
ConditionalBlocks. The old top-level-only fix-up left inner blocks pointing at
the discarded snapshot, which broke ``parent_graph`` walks downstream
(``propagate_memlets_sdfg`` crashed with ``'NoneType' object has no attribute
'cfg_id'`` when ``apply_gpu_transformations`` ran after a refusal).
"""
import copy

import numpy as np
import pytest

import dace
from dace.transformation.passes.vectorization.vectorize_multi_dim import restore_sdfg_in_place

N = dace.symbol("N")
M = dace.symbol("M")


@dace.program
def _loop_nest(alpha: dace.float64, C: dace.float64[N, N], A: dace.float64[N, M]):
    for i in range(N):
        C[i, :i + 1] *= alpha
        for k in range(M):
            C[i, :i + 1] += alpha * A[i, k] * A[:i + 1, k]


def _build() -> dace.SDFG:
    """Unsimplified SDFG with nested LoopRegions (blocks below the top level)."""
    return _loop_nest.to_sdfg(simplify=False)


def _assert_rooting_invariants(sdfg: dace.SDFG) -> None:
    """Every block's ``sdfg`` matches its region's, and every non-root region
    has a parent graph."""
    for cfr in sdfg.all_control_flow_regions(recursive=True):
        if cfr is not sdfg:
            assert cfr.parent_graph is not None, f"region {cfr.label!r} lost its parent_graph"
        for block in cfr.nodes():
            assert block.sdfg is cfr.sdfg, (f"block {block.label!r} in {cfr.label!r} points at a stale SDFG "
                                            f"({block.sdfg} is not {cfr.sdfg})")
            assert block.parent_graph is cfr


def test_restore_reroots_nested_blocks():
    """Blocks inside LoopRegions must be re-rooted onto the target, not left
    pointing at the discarded snapshot."""
    sdfg = _build()
    snapshot = copy.deepcopy(sdfg)
    restore_sdfg_in_place(sdfg, copy.deepcopy(snapshot))
    _assert_rooting_invariants(sdfg)
    assert sdfg._sdfg is sdfg
    sdfg.validate()


def test_restore_preserves_identity_and_survives_propagation():
    """The caller's reference stays valid and memlet propagation (the crash
    site of the old bug) runs cleanly on the restored SDFG."""
    from dace.sdfg.propagation import propagate_memlets_sdfg

    sdfg = _build()
    ref = sdfg  # the caller-owned reference
    snapshot = copy.deepcopy(sdfg)
    restore_sdfg_in_place(sdfg, copy.deepcopy(snapshot))
    assert ref is sdfg
    propagate_memlets_sdfg(sdfg)  # crashed with NoneType.cfg_id before the fix
    sdfg.validate()


def test_restore_roundtrip_numerics():
    """A restored SDFG compiles and computes the same result as the original."""
    sdfg = _build()
    snapshot = copy.deepcopy(sdfg)
    restore_sdfg_in_place(sdfg, copy.deepcopy(snapshot))

    n, m = 12, 7
    rng = np.random.default_rng(0)
    C = rng.random((n, n))
    A = rng.random((n, m))
    alpha = np.float64(1.25)

    ref = C.copy()
    for i in range(n):
        ref[i, :i + 1] *= alpha
        for k in range(m):
            ref[i, :i + 1] += alpha * A[i, k] * A[:i + 1, k]

    sdfg(alpha=alpha, C=C, A=A, N=n, M=m)
    np.testing.assert_allclose(C, ref, rtol=1e-12)


def test_restore_rejects_nested_target():
    """Restoring into a nested (non-root) SDFG would null its parent pointers
    and orphan it from the enclosing SDFG — must raise instead."""
    outer = dace.SDFG("restore_outer")
    outer.add_array("x", (1, ), dace.float64)
    state = outer.add_state()
    inner = dace.SDFG("restore_inner")
    inner.add_array("x", (1, ), dace.float64)
    istate = inner.add_state()
    t = istate.add_tasklet("set", {}, {"_o"}, "_o = 1.0")
    istate.add_edge(t, "_o", istate.add_access("x"), None, dace.Memlet("x[0]"))
    nsdfg = state.add_nested_sdfg(inner, inputs={}, outputs={"x"})
    state.add_edge(nsdfg, "x", state.add_access("x"), None, dace.Memlet("x[0]"))
    outer.validate()

    with pytest.raises(ValueError, match="nested SDFG"):
        restore_sdfg_in_place(inner, copy.deepcopy(inner))
    # The enclosing SDFG is untouched and still valid.
    outer.validate()


if __name__ == "__main__":
    test_restore_reroots_nested_blocks()
    test_restore_preserves_identity_and_survives_propagation()
    test_restore_roundtrip_numerics()
    test_restore_rejects_nested_target()
