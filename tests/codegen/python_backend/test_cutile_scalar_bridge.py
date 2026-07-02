# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the cuTile scalar-bridge binding emission (bug 03 hardening).

``_emit_scalar_bridge_binding`` binds a Register scalar bridge (staged by the
vectorizer's ``stage_constant_access``) to its traced source. The vectorizer
can stage a single ELEMENT of an array (``src_subset`` like ``aa[0, j]``); the
binding must then index the source (``aa_const = aa[0, j]``) instead of
aliasing the whole tensor. Untraceable bridges must warn, not silently emit
nothing.
"""
import warnings as _warnings

import pytest

import dace
from dace import dtypes
from dace.codegen import dispatcher as dispatcher_mod
from dace.codegen.py.cutile_target import CuTilePythonCodeGen
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.dtypes import ScheduleType, StorageType
from dace.memlet import Memlet
from dace.sdfg import SDFG, nodes


class _StubFrameCodegen:
    """Minimal stub for the frame codegen, just providing a dispatcher."""

    def __init__(self):
        self.dispatcher = dispatcher_mod.TargetDispatcher(self)
        self._initcode = PythonCodeIOStream()
        self._exitcode = PythonCodeIOStream()


def _emit_bridge_binding(sdfg: SDFG, state: "dace.SDFGState", bridge: nodes.AccessNode) -> str:
    """Run ``_emit_scalar_bridge_binding`` on ``bridge`` and return the code."""
    codegen = CuTilePythonCodeGen(_StubFrameCodegen(), sdfg)
    stream = PythonCodeIOStream()
    codegen._emit_scalar_bridge_binding(sdfg, state, bridge, sdfg, sdfg.node_id(state), stream)
    return stream.getvalue()


def _make_sdfg_with_bridge(name: str, src_memlet: Memlet, add_symbol: str = None):
    """Build ``AccessNode(aa) -> MapEntry(CuTile) -> AccessNode(aa_const)``.

    :param name: SDFG name.
    :param src_memlet: Memlet for the staging edge (source side).
    :param add_symbol: Optional free symbol to register on the SDFG.
    :returns: ``(sdfg, state, bridge_access_node)``.
    """
    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    if add_symbol:
        sdfg.add_symbol(add_symbol, dace.int64)
    sdfg.add_array("aa", [4, 8], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_scalar("aa_const", dace.float64, storage=StorageType.Register, transient=True)

    state = sdfg.add_state("main")
    me, _mx = state.add_map("cutile_map", {"tile_i": "0:8"}, schedule=ScheduleType.CuTile)
    aa_node = state.add_read("aa")
    bridge = state.add_access("aa_const")
    state.add_memlet_path(aa_node, me, bridge, memlet=src_memlet)
    return sdfg, state, bridge


class TestArrayElementBridge:
    """Array-element sources must be bound with a ``ct.load`` at the staged
    element: cuda.tile arrays are not subscriptable inside kernels, so a raw
    ``aa[0, 1]`` subscript would raise at kernel compile time."""

    def test_constant_element_index(self):
        """``aa[0, 1]`` staged into the bridge -> a (1, 1)-shaped ``ct.load``."""
        sdfg, state, bridge = _make_sdfg_with_bridge("bridge_const_idx", Memlet(data="aa", subset="0, 1"))
        code = _emit_bridge_binding(sdfg, state, bridge)
        assert "aa_const = ct.reshape(ct.load(aa, index=((0), (1),), shape=(1, 1,)), (1,))" in code
        # Neither the whole-tensor alias nor a raw subscript may be emitted.
        assert "aa_const = aa\n" not in code
        assert "aa_const = aa[" not in code

    def test_symbolic_element_index(self):
        """``aa[0, j]`` with a free symbol loads at the symbolic index."""
        sdfg, state, bridge = _make_sdfg_with_bridge("bridge_sym_idx", Memlet(data="aa", subset="0, j"), add_symbol="j")
        code = _emit_bridge_binding(sdfg, state, bridge)
        assert "aa_const = ct.reshape(ct.load(aa, index=((0), (j),), shape=(1, 1,)), (1,))" in code

    def test_per_iteration_element_uses_inner_subset(self):
        """The binding must index by the INNER (per-iteration) memlet subset.

        The outer edge spans the whole map range (``aa[0:4, 0:8]``); the staged
        element varies per iteration (``aa[tile_i, 3]``). Using the outer
        subset would emit a non-constant slice of the wrong elements
        (softmax/conv2d_bias regression).
        """
        sdfg = SDFG("bridge_per_iter")
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.add_array("aa", [4, 8], dace.float64, storage=StorageType.GPU_Global)
        sdfg.add_scalar("aa_const", dace.float64, storage=StorageType.Register, transient=True)
        state = sdfg.add_state("main")
        me, _mx = state.add_map("cutile_map", {"tile_i": "0:4"}, schedule=ScheduleType.CuTile)
        aa_node = state.add_read("aa")
        bridge = state.add_access("aa_const")
        me.add_in_connector("IN_aa")
        me.add_out_connector("OUT_aa")
        state.add_edge(aa_node, None, me, "IN_aa", Memlet(data="aa", subset="0:4, 3"))
        state.add_edge(me, "OUT_aa", bridge, None, Memlet(data="aa", subset="tile_i, 3"))
        code = _emit_bridge_binding(sdfg, state, bridge)
        assert "aa_const = ct.reshape(ct.load(aa, index=((tile_i), (3),), shape=(1, 1,)), (1,))" in code

    def test_non_single_element_subset_warns(self):
        """A multi-element staged subset cannot be a scalar bridge: warn."""
        sdfg, state, bridge = _make_sdfg_with_bridge("bridge_multi_elem", Memlet(data="aa", subset="0:2, 1"))
        with pytest.warns(UserWarning, match="non-single-element"):
            code = _emit_bridge_binding(sdfg, state, bridge)
        assert "aa_const =" not in code

    def test_scalar_source_binds_bare_name(self):
        """A true scalar source keeps the plain rename (``alpha_const = alpha``)."""
        sdfg = SDFG("bridge_scalar_src")
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.add_scalar("alpha", dace.float64)
        sdfg.add_scalar("alpha_const", dace.float64, storage=StorageType.Register, transient=True)
        state = sdfg.add_state("main")
        me, _mx = state.add_map("cutile_map", {"tile_i": "0:8"}, schedule=ScheduleType.CuTile)
        a_node = state.add_read("alpha")
        bridge = state.add_access("alpha_const")
        state.add_memlet_path(a_node, me, bridge, memlet=Memlet(data="alpha", subset="0"))
        code = _emit_bridge_binding(sdfg, state, bridge)
        assert "alpha_const = alpha" in code
        assert "alpha[" not in code


class TestBridgeWarnings:
    """Untraceable bridges warn instead of silently emitting nothing."""

    def test_untraceable_map_entry_edge_warns(self):
        """A MapEntry in-edge without a scope connector cannot be traced."""
        sdfg = SDFG("bridge_untraceable")
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.add_scalar("aa_const", dace.float64, storage=StorageType.Register, transient=True)
        state = sdfg.add_state("main")
        me, _mx = state.add_map("cutile_map", {"tile_i": "0:8"}, schedule=ScheduleType.CuTile)
        bridge = state.add_access("aa_const")
        # Direct edge with no scope connector: tracing must fail.
        state.add_edge(me, None, bridge, None, Memlet())
        with pytest.warns(UserWarning, match="aa_const"):
            code = _emit_bridge_binding(sdfg, state, bridge)
        assert "aa_const =" not in code

    def test_non_map_entry_source_warns(self):
        """A bridge fed by a plain AccessNode (not MapEntry / code node) warns."""
        sdfg = SDFG("bridge_bad_source")
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.add_scalar("other", dace.float64, storage=StorageType.Register, transient=True)
        sdfg.add_scalar("aa_const", dace.float64, storage=StorageType.Register, transient=True)
        state = sdfg.add_state("main")
        other = state.add_access("other")
        bridge = state.add_access("aa_const")
        state.add_edge(other, None, bridge, None, Memlet(data="other", subset="0"))
        with pytest.warns(UserWarning, match="aa_const"):
            _emit_bridge_binding(sdfg, state, bridge)

    def test_tasklet_source_does_not_warn(self):
        """A tasklet producer binds the name itself -- no warning, no code."""
        sdfg = SDFG("bridge_tasklet_source")
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.add_scalar("aa_const", dace.float64, storage=StorageType.Register, transient=True)
        state = sdfg.add_state("main")
        tasklet = state.add_tasklet("produce", set(), {"out"}, "out = 1.0")
        bridge = state.add_access("aa_const")
        state.add_edge(tasklet, "out", bridge, None, Memlet(data="aa_const", subset="0"))
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            code = _emit_bridge_binding(sdfg, state, bridge)
        assert code.strip() == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
