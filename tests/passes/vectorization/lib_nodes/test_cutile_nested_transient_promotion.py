# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Transient promotion inside demoted GPU scopes descends into NestedSDFGs.

``apply_gpu_transformations(register_transients=True)`` stamps in-scope
transients ``Register`` (host numpy in the Python backend). When
``GPUDeviceToCuTile`` demotes a non-tileops ``GPU_Device`` map to a host driver
loop, ``_promote_demoted_scope_transients`` re-stamps such transients
``GPU_Global`` so whole-array copies against the cupy operands stay
device-resident. A ``Register`` Array transient *inside a NestedSDFG* within
the demoted scope used to be skipped; its whole-array copy to a ``GPU_Global``
connector then raised ``ValueError: non-scalar numpy.ndarray cannot be used
for fill`` at runtime. These tests pin the recursive walk.
"""
import warnings

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.sdfg import SDFG
from dace.transformation.passes.vectorization.cutile_lowering import _demote_residual_gpu_device_maps


def _demoted_nested_transient_sdfg(name: str) -> SDFG:
    """A ``GPU_Device`` map over a NestedSDFG whose ``Register`` transient
    flows through a WHOLE-ARRAY copy to a ``GPU_Global`` connector.

    :param name: SDFG name.
    :returns: The SDFG, before demotion.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", (4, 32), dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("B", (4, 32), dace.float64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()

    inner = dace.SDFG(name + "_body")
    inner.add_array("a_in", (32, ), dace.float64)
    inner.add_array("b_out", (32, ), dace.float64)
    inner.add_array("t", (32, ), dace.float64, transient=True, storage=dtypes.StorageType.Register)
    ist = inner.add_state()
    tasklet = ist.add_tasklet("mul", {"_a"}, {"_t"}, "_t = _a * 2.0")
    read_a, acc_t, write_b = ist.add_read("a_in"), ist.add_access("t"), ist.add_write("b_out")
    ime, imx = ist.add_map("inner1", {"j": "0:32"})
    ist.add_memlet_path(read_a, ime, tasklet, dst_conn="_a", memlet=dace.Memlet("a_in[j]"))
    ist.add_memlet_path(tasklet, imx, acc_t, src_conn="_t", memlet=dace.Memlet("t[j]"))
    # Whole-array copy: fails at runtime when ``t`` stays a host numpy array.
    ist.add_nedge(acc_t, write_b, dace.Memlet("t[0:32] -> b_out[0:32]"))

    me, mx = state.add_map("outer", {"i": "0:4"}, schedule=dtypes.ScheduleType.GPU_Device)
    nsdfg = state.add_nested_sdfg(inner, {"a_in"}, {"b_out"})
    state.add_memlet_path(state.add_read("A"), me, nsdfg, dst_conn="a_in", memlet=dace.Memlet("A[i, 0:32]"))
    state.add_memlet_path(nsdfg, mx, state.add_write("B"), src_conn="b_out", memlet=dace.Memlet("B[i, 0:32]"))
    return sdfg


def _demote(sdfg: SDFG) -> None:
    """Demote the residual ``GPU_Device`` maps (suppressing the size diagnostic)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _demote_residual_gpu_device_maps(sdfg, strict=False, pass_name="Test")


def test_nested_register_transient_promoted():
    """Demotion re-stamps the NestedSDFG-internal Register transient GPU_Global.

    Before the recursive walk, only AccessNodes directly in the demoted scope
    were promoted and ``t`` stayed ``Register``. Structural check, no GPU.
    """
    sdfg = _demoted_nested_transient_sdfg("nested_promotion_struct")
    _demote(sdfg)
    inner = next(sd for sd in sdfg.all_sdfgs_recursive() if sd is not sdfg)
    assert inner.arrays["t"].storage == dtypes.StorageType.GPU_Global
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.validate()


def test_nested_scalar_and_nontransient_left_alone():
    """The walk touches only Register Array TRANSIENTS: connector arrays and
    Scalars keep their storage."""
    sdfg = _demoted_nested_transient_sdfg("nested_promotion_scope")
    inner = next(sd for sd in sdfg.all_sdfgs_recursive() if sd is not sdfg)
    inner.add_scalar("s", dace.float64, transient=True, storage=dtypes.StorageType.Register)
    ist = next(iter(inner.states()))
    seed = ist.add_tasklet("seed", {}, {"_s"}, "_s = 0.0")
    ist.add_edge(seed, "_s", ist.add_write("s"), None, dace.Memlet("s[0]"))
    _demote(sdfg)
    assert inner.arrays["t"].storage == dtypes.StorageType.GPU_Global
    assert inner.arrays["s"].storage == dtypes.StorageType.Register, "Scalars feed host control flow"
    assert not inner.arrays["a_in"].transient
    assert inner.arrays["a_in"].storage == dtypes.StorageType.Default, "connector arrays inherit parent storage"


def test_gpu_global_frame_imports_cupy():
    """The frame imports cupy whenever the module touches GPU arrays.

    Previously the import came only from the cuTile target's includes, i.e.
    only when that target was dispatch-'used' (a CuTile map or a host<->device
    copy); an SDFG with ``GPU_Global`` data but neither raised ``NameError``
    on the generated ``cupy.empty``. Codegen-only check, no GPU."""
    sdfg = dace.SDFG("cupy_import_gate")
    sdfg.add_array("A", (8, ), dace.float64, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("t", (8, ), dace.float64, transient=True, storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array("B", (8, ), dace.float64, storage=dtypes.StorageType.GPU_Global)
    state = sdfg.add_state()
    acc_t = state.add_access("t")
    state.add_nedge(state.add_read("A"), acc_t, dace.Memlet("A[0:8] -> t[0:8]"))
    state.add_nedge(acc_t, state.add_write("B"), dace.Memlet("t[0:8] -> B[0:8]"))
    sdfg.backend = dtypes.BackendLanguage.Python
    assert "import cupy" in sdfg.generate_code()[0].clean_code

    # A CPU-only module must stay cupy-free (works on cupy-less machines).
    cpu = dace.SDFG("cupy_import_gate_cpu")
    cpu.add_array("X", (8, ), dace.float64)
    cpu.add_array("Y", (8, ), dace.float64)
    st = cpu.add_state()
    st.add_nedge(st.add_read("X"), st.add_write("Y"), dace.Memlet("X[0:8] -> Y[0:8]"))
    cpu.backend = dtypes.BackendLanguage.Python
    assert "import cupy" not in cpu.generate_code()[0].clean_code


@pytest.mark.gpu
def test_nested_register_transient_runtime():
    """The promoted SDFG compiles and runs; without promotion the whole-array
    copy raised ``ValueError: non-scalar numpy.ndarray cannot be used for
    fill``."""
    import cupy
    sdfg = _demoted_nested_transient_sdfg("nested_promotion_runtime")
    _demote(sdfg)
    sdfg.backend = dtypes.BackendLanguage.Python

    A = cupy.asarray(np.random.default_rng(0).random((4, 32)))
    B = cupy.zeros((4, 32))
    sdfg(A=A, B=B)
    assert np.allclose(cupy.asnumpy(B), cupy.asnumpy(A) * 2.0)


if __name__ == "__main__":
    test_nested_register_transient_promoted()
    test_nested_scalar_and_nontransient_left_alone()
    test_gpu_global_frame_imports_cupy()
    test_nested_register_transient_runtime()
    print("ok")
