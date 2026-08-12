# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Mutable cuTile scalar storage and writeback tests."""

import numpy as np
import pytest

import dace
from dace import data, dtypes
import dace.codegen.dispatcher as dispatcher_mod
from dace.codegen.codeobject import CodeObject
from dace.codegen.py.cutile_target import CuTilePythonCodeGen
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.dtypes import ScheduleType, StorageType
from dace.memlet import Memlet
from dace.transformation.passes.vectorization.vectorize_cutile import VectorizeCuTile

N = dace.symbol("N")


@dace.program
def _sum_into_scalar(x: dace.float64[N], acc: dace.float64):
    for i in dace.map[0:N]:
        with dace.tasklet:
            value << x[i]
            result >> acc(1, lambda left, right: left + right)
            result = value


def _lower_sum(name: str) -> dace.SDFG:
    """Lower the scalar WCR program through the public cuTile pass."""
    sdfg = _sum_into_scalar.to_sdfg(simplify=False)
    sdfg.name = name
    VectorizeCuTile(widths=(32, )).apply_pass(sdfg, {})
    return sdfg


def _host_and_build(sdfg: dace.SDFG) -> tuple[CodeObject, CodeObject]:
    """Return the generated Cython host and aggregate cuTile build objects."""
    objects = sdfg.generate_code()
    host = next(obj for obj in objects if obj.language == "pyx")
    build = next(obj for obj in objects if obj.target_type == "cutile_build")
    return host, build


class _StubFrameCodegen:
    """Minimal frame-codegen surface required by the cuTile target."""

    def __init__(self):
        self.dispatcher = dispatcher_mod.TargetDispatcher(self)
        self._initcode = PythonCodeIOStream()
        self._exitcode = PythonCodeIOStream()


def _manual_writeback_sdfg(
        name: str,
        wcr: str | None,
        dtype: dtypes.typeclass = dace.float64) -> tuple[dace.SDFG, dace.SDFGState, dace.nodes.AccessNode]:
    """Build a minimal Register-scalar-to-device-array scope output."""
    sdfg = dace.SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array("out", (1, ), dtype, storage=StorageType.GPU_Global)
    sdfg.add_scalar("value", dtype, storage=StorageType.Register, transient=True)

    state = sdfg.add_state("main")
    _entry, exit_node = state.add_map("cutile_map", {"tile_i": "0:1"}, schedule=ScheduleType.CuTile)
    output = state.add_write("out")
    bridge = state.add_access("value")
    exit_node.add_in_connector("IN_out")
    exit_node.add_out_connector("OUT_out")
    state.add_edge(bridge, None, exit_node, "IN_out", Memlet("out[0]", wcr=wcr))
    state.add_edge(exit_node, "OUT_out", output, None, Memlet("out[0]", wcr=wcr))
    return sdfg, state, bridge


def _writeback_code(name: str, wcr: str | None, dtype: dtypes.typeclass = dace.float64) -> str:
    """Emit only the localized Register scalar writeback code."""
    sdfg, state, bridge = _manual_writeback_sdfg(name, wcr, dtype)
    codegen = CuTilePythonCodeGen(_StubFrameCodegen(), sdfg)
    stream = PythonCodeIOStream()
    codegen._emit_scalar_bridge_writeback(sdfg, state, bridge, sdfg, sdfg.node_id(state), stream)
    return stream.getvalue()


def test_lowering_promotes_device_scalar_and_emits_atomic_writeback():
    """The public scalar stays host-visible while its GPU clone is an array."""
    sdfg = _lower_sum("cutile_mutable_scalar_structure")

    public_desc = sdfg.arrays["acc"]
    assert isinstance(public_desc, data.Scalar)
    device_name = next(name for name in sdfg.arrays if name.startswith("gpu_acc"))
    device_desc = sdfg.arrays[device_name]
    assert isinstance(device_desc, data.Array)
    assert tuple(int(dim) for dim in device_desc.shape) == (1, )
    assert device_desc.storage == StorageType.GPU_Global

    copy_pairs = set()
    for state in sdfg.states():
        for edge in state.edges():
            if (isinstance(edge.src, dace.nodes.AccessNode) and isinstance(edge.dst, dace.nodes.AccessNode)):
                copy_pairs.add((edge.src.data, edge.dst.data))
    assert ("acc", device_name) in copy_pairs
    assert (device_name, "acc") in copy_pairs

    host, build = _host_and_build(sdfg)
    host_compact = "".join(host.code.split())
    build_compact = "".join(build.code.split())
    assert f"ct.atomic_add({device_name},(0,)," in build_compact
    assert "cupy.asarray(acc" not in host_compact
    assert f"{device_name}[...]=acc.item()" in host_compact
    assert f"acc[...]={device_name}.item()" in host_compact
    assert "compilation.ArrayConstraint(ct.float64,1," in build_compact
    assert "shape_constant=(1,)" in build_compact


def test_register_scalar_ordinary_write_uses_store():
    """A non-WCR Register scalar output uses an ordinary element store."""
    code = _writeback_code("cutile_scalar_store", None).replace(" ", "")
    assert "ct.store(out,index=(0,),tile=value)" in code


@pytest.mark.parametrize(
    ("wcr", "atomic", "dtype"),
    [
        ("lambda x, y: x + y", "atomic_add", dace.float64),
        ("lambda x, y: min(x, y)", "atomic_min", dace.float64),
        ("lambda x, y: max(x, y)", "atomic_max", dace.float64),
        ("lambda x, y: x & y", "atomic_and", dace.int64),
        ("lambda x, y: x | y", "atomic_or", dace.int64),
        ("lambda x, y: x ^ y", "atomic_xor", dace.int64),
        ("lambda x, y: y", "atomic_xchg", dace.float64),
    ],
    ids=("sum", "min", "max", "and", "or", "xor", "exchange"),
)
def test_supported_scalar_wcr_uses_atomic(wcr: str, atomic: str, dtype: dtypes.typeclass):
    """Every supported scalar WCR maps to the corresponding atomic."""
    code = _writeback_code(f"cutile_scalar_{atomic}", wcr, dtype)
    assert f"ct.{atomic}(out,(0,),value)" in code.replace(" ", "")


def test_unsupported_scalar_wcr_fails_closed():
    """Product has no cuda.tile atomic equivalent and must not race silently."""
    with pytest.raises(NotImplementedError, match="Product.*atomic writeback"):
        _writeback_code("cutile_scalar_product", "lambda x, y: x * y")


@pytest.mark.gpu
def test_mutable_scalar_wcr_cython_runtime():
    """Compile and run N=70 with a nonzero public scalar initial value."""
    n = 70
    initial = np.float64(7.25)
    rng = np.random.default_rng(42)
    x = rng.random(n, dtype=np.float64)
    acc = np.array(initial, dtype=np.float64)

    sdfg = _lower_sum("cutile_mutable_scalar_runtime")
    compiled = sdfg.compile()
    compiled(x=x, acc=acc, N=n)

    expected = initial + np.sum(x)
    np.testing.assert_allclose(acc.item(), expected, rtol=1e-14, atol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
