# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Unit tests for the cuTile scalar-bridge binding emission (bug 03 hardening).

``_emit_scalar_bridge_binding`` binds a Register scalar bridge (staged by the
vectorizer's ``stage_constant_access``) to its traced source. The vectorizer
can stage a single ELEMENT of an array (``src_subset`` like ``aa[0, j]``); the
binding must then load exactly that element as a scalar tile
(``aa_const = ct.load(aa, (0, j), shape=()).item()``) — cuTile arrays are not
subscriptable in-kernel and the propagated outer subset is a non-constant
slice the cuda.tile compiler rejects. Numeric (float/int/uint) scalar sources
also bind via a 0-d tile load (the launch site passes them as 1-element
device arrays to keep f64 precision and full int64 range); bool scalars keep
the plain rename (passed by value). Untraceable bridges must warn, not
silently emit nothing.
"""
import warnings as _warnings

import numpy as np
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
    """Array-element sources must be bound with a 0-d ``ct.load`` at the
    memlet subset (arrays are not subscriptable inside a ct kernel)."""

    def test_constant_element_index(self):
        """``aa[0, 1]`` staged -> ``aa_const = ct.load(aa, (0, 1), shape=()).item()``."""
        sdfg, state, bridge = _make_sdfg_with_bridge("bridge_const_idx", Memlet(data="aa", subset="0, 1"))
        code = _emit_bridge_binding(sdfg, state, bridge).replace(" ", "")
        assert "aa_const=ct.load(aa,(0,1,),shape=()).item()" in code
        # The whole-tensor alias must NOT be emitted.
        assert "aa_const = aa\n" not in code
        assert "aa_const = aa[" not in code

    def test_symbolic_element_index(self):
        """``aa[0, j]`` -> ``aa_const = ct.load(aa, (0, j), shape=()).item()``."""
        sdfg, state, bridge = _make_sdfg_with_bridge("bridge_sym_idx", Memlet(data="aa", subset="0, j"), add_symbol="j")
        code = _emit_bridge_binding(sdfg, state, bridge).replace(" ", "")
        assert "aa_const=ct.load(aa,(0,j,),shape=()).item()" in code

    def test_per_iteration_element_uses_inner_subset(self):
        """The binding must index by the INNER (per-iteration) memlet subset.

        The outer edge spans the whole map range (``aa[0:4, 3]``); the staged
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
        code = _emit_bridge_binding(sdfg, state, bridge).replace(" ", "")
        assert "aa_const=ct.load(aa,(tile_i,3,),shape=()).item()" in code

    def test_non_single_element_subset_warns_and_uses_begins(self):
        """A multi-element staged subset warns and anchors at the per-dim
        begins (best-effort scalar bridge)."""
        sdfg, state, bridge = _make_sdfg_with_bridge("bridge_multi_elem", Memlet(data="aa", subset="0:2, 1"))
        with pytest.warns(UserWarning, match="non-element"):
            code = _emit_bridge_binding(sdfg, state, bridge)
        assert "ct.load(aa, (0, 1,), shape=()).item()" in code

    def test_float_scalar_source_binds_scalar_tile_load(self):
        """A float scalar source is a 1-element device array at runtime and
        binds via a 0-d tile load (f64 precision; by-value floats are f32)."""
        sdfg = SDFG("bridge_scalar_src")
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.add_scalar("alpha", dace.float64)
        sdfg.add_scalar("alpha_const", dace.float64, storage=StorageType.Register, transient=True)
        state = sdfg.add_state("main")
        me, _mx = state.add_map("cutile_map", {"tile_i": "0:8"}, schedule=ScheduleType.CuTile)
        a_node = state.add_read("alpha")
        bridge = state.add_access("alpha_const")
        state.add_memlet_path(a_node, me, bridge, memlet=Memlet(data="alpha", subset="0"))
        code = _emit_bridge_binding(sdfg, state, bridge).replace(" ", "")
        assert "alpha_const=ct.load(alpha,(0,),shape=()).item()" in code

    def test_int_scalar_source_binds_scalar_tile_load(self):
        """An integer scalar source is a 1-element device array at runtime too
        (by-value ints are typed int32: OverflowError >= 2**31) and binds via
        a 0-d tile load."""
        sdfg = SDFG("bridge_int_scalar_src")
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.add_scalar("kk", dace.int64)
        sdfg.add_scalar("kk_const", dace.int64, storage=StorageType.Register, transient=True)
        state = sdfg.add_state("main")
        me, _mx = state.add_map("cutile_map", {"tile_i": "0:8"}, schedule=ScheduleType.CuTile)
        a_node = state.add_read("kk")
        bridge = state.add_access("kk_const")
        state.add_memlet_path(a_node, me, bridge, memlet=Memlet(data="kk", subset="0"))
        code = _emit_bridge_binding(sdfg, state, bridge).replace(" ", "")
        assert "kk_const=ct.load(kk,(0,),shape=()).item()" in code

    def test_bool_scalar_source_binds_bare_name(self):
        """A bool scalar source keeps the plain rename (passed by value)."""
        sdfg = SDFG("bridge_bool_scalar_src")
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.add_scalar("flag", dace.bool)
        sdfg.add_scalar("flag_const", dace.bool, storage=StorageType.Register, transient=True)
        state = sdfg.add_state("main")
        me, _mx = state.add_map("cutile_map", {"tile_i": "0:8"}, schedule=ScheduleType.CuTile)
        a_node = state.add_read("flag")
        bridge = state.add_access("flag_const")
        state.add_memlet_path(a_node, me, bridge, memlet=Memlet(data="flag", subset="0"))
        code = _emit_bridge_binding(sdfg, state, bridge)
        assert "flag_const = flag" in code
        assert "ct.load(flag" not in code


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

    def test_register_access_node_source_binds_bare_name(self):
        """A bridge fed directly by a local Register scalar uses a rename."""
        sdfg = SDFG("bridge_bad_source")
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.add_scalar("other", dace.float64, storage=StorageType.Register, transient=True)
        sdfg.add_scalar("aa_const", dace.float64, storage=StorageType.Register, transient=True)
        state = sdfg.add_state("main")
        other = state.add_access("other")
        bridge = state.add_access("aa_const")
        state.add_edge(other, None, bridge, None, Memlet(data="other", subset="0"))
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            code = _emit_bridge_binding(sdfg, state, bridge)
        assert "aa_const = other" in code

    def test_numeric_scalar_access_node_source_binds_tile_load(self):
        """A direct numeric parameter source in a nested SDFG is device-loaded."""
        sdfg = SDFG("bridge_nested_scalar_source")
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.add_scalar("dt", dace.float64)
        sdfg.add_scalar("dt_const", dace.float64, storage=StorageType.Register, transient=True)
        state = sdfg.add_state("main")
        dt = state.add_access("dt")
        bridge = state.add_access("dt_const")
        state.add_edge(dt, None, bridge, None, Memlet(data="dt", subset="0"))
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            code = _emit_bridge_binding(sdfg, state, bridge)
        assert "dt_const = ct.load(dt, (0,), shape=()).item()" in code

    def test_array_access_node_source_uses_source_subset(self):
        """A direct array source loads the source-side element subset."""
        sdfg = SDFG("bridge_nested_array_source")
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.add_array("aa", [4, 8], dace.float64, storage=StorageType.GPU_Global)
        sdfg.add_scalar("aa_const", dace.float64, storage=StorageType.Register, transient=True)
        state = sdfg.add_state("main")
        aa = state.add_access("aa")
        bridge = state.add_access("aa_const")
        state.add_edge(aa, None, bridge, None, Memlet(data="aa_const", subset="0", other_subset="2, 3"))
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            code = _emit_bridge_binding(sdfg, state, bridge)
        assert "aa_const = ct.load(aa, (2, 3,), shape=()).item()" in code

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


class TestNestedBridgeGpuIntegration:
    """End-to-end coverage for a direct scalar bridge in a nested SDFG."""

    @pytest.mark.gpu
    def test_numeric_scalar_bridge_executes(self):
        """A nested ``dt -> dt_const`` bridge compiles and scales a tile."""
        import cupy

        sdfg = SDFG("nested_scalar_bridge_runtime")
        sdfg.backend = dtypes.BackendLanguage.Python
        sdfg.add_array("x", [32], dace.float64, storage=StorageType.GPU_Global)
        sdfg.add_array("y", [32], dace.float64, storage=StorageType.GPU_Global)
        sdfg.add_scalar("dt", dace.float64)
        sdfg.add_array("x_tile", [32], dace.float64, storage=StorageType.CuTile_Tile, transient=True)
        sdfg.add_array("y_tile", [32], dace.float64, storage=StorageType.CuTile_Tile, transient=True)

        inner = SDFG("nested_scalar_bridge_body")
        inner.add_array("inp", [32], dace.float64, storage=StorageType.CuTile_Tile)
        inner.add_array("out", [32], dace.float64, storage=StorageType.CuTile_Tile)
        inner.add_scalar("dt", dace.float64)
        inner.add_scalar("dt_const", dace.float64, storage=StorageType.Register, transient=True)
        inner_state = inner.add_state("compute")
        inp = inner_state.add_read("inp")
        dt = inner_state.add_read("dt")
        bridge = inner_state.add_access("dt_const")
        out = inner_state.add_write("out")
        scale = inner_state.add_tasklet("scale", {"value", "factor"}, {"result"}, "result = value * factor")
        inner_state.add_edge(dt, None, bridge, None, Memlet("dt[0]"))
        inner_state.add_edge(inp, None, scale, "value", Memlet("inp[0:32]"))
        inner_state.add_edge(bridge, None, scale, "factor", Memlet("dt_const[0]"))
        inner_state.add_edge(scale, "result", out, None, Memlet("out[0:32]"))

        state = sdfg.add_state("main")
        me, mx = state.add_map("cutile_map", {"tile_i": "0:32:32"}, schedule=ScheduleType.CuTile)
        nested = state.add_nested_sdfg(inner, {"inp", "dt"}, {"out"})
        x_tile = state.add_access("x_tile")
        y_tile = state.add_access("y_tile")
        state.add_memlet_path(state.add_read("x"), me, x_tile, memlet=Memlet("x[0:32]"))
        state.add_edge(x_tile, None, nested, "inp", Memlet("x_tile[0:32]"))
        state.add_memlet_path(state.add_read("dt"), me, nested, dst_conn="dt", memlet=Memlet("dt[0]"))
        state.add_edge(nested, "out", y_tile, None, Memlet("y_tile[0:32]"))
        state.add_memlet_path(y_tile, mx, state.add_write("y"), memlet=Memlet("y[0:32]"))
        sdfg.fill_scope_connectors()

        x = cupy.arange(32, dtype=cupy.float64)
        y = cupy.empty_like(x)
        sdfg(x=x, y=y, dt=np.float64(0.125))
        np.testing.assert_array_equal(cupy.asnumpy(y), np.arange(32, dtype=np.float64) * 0.125)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
