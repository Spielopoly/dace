# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Structure tests for aggregate cuTile build and Cython host artifacts."""

import dataclasses
import json
from pathlib import Path

import pytest

import dace
from dace import dtypes
from dace.codegen.codeobject import CodeObject
from dace.codegen.py import artifacts
from dace.codegen.py.compiled_sdfg import _load_native_module
from dace.codegen.py.compiler import build_python_extension
from dace.codegen.py.artifacts import (CuTileKernelSpec, CuTileModuleSpec, CuTileParameterKind, CuTileParameterSpec,
                                       make_kernel_symbol)
from dace.dtypes import Language, ScheduleType, StorageType
from dace.memlet import Memlet


def _array_parameter(name: str, dtype: str = "float32", rank: int = 1) -> CuTileParameterSpec:
    return CuTileParameterSpec(
        name=name,
        host_expression=name,
        kind=CuTileParameterKind.Array,
        dace_dtype=dtype,
        cutile_dtype=dtype,
        rank=rank,
        index_dtype="int64",
        shape_constraints=(None, ) * rank,
        stride_constraints=(None, ) * rank,
        alias_groups=("all", ),
        may_alias_internally=True,
        is_read=True,
        is_written=False,
        source=name,
    )


def _kernel(identity, body: str, parameter: CuTileParameterSpec) -> CuTileKernelSpec:
    symbol, helper = make_kernel_symbol(identity, body, (parameter, ))
    return CuTileKernelSpec(
        stable_identity=identity,
        map_identity=f"cfg={identity[0]}/state={identity[1]}/map={identity[2]}",
        source_location="test.py:1:0",
        branch_body=body,
        helper_source="",
        parameters=(parameter, ),
        exported_symbol=symbol,
        launch_helper=helper,
        grid=("1", ),
    )


def _manual_scalar_sdfg(name: str) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_symbol("alpha", dace.float64)
    sdfg.add_array("x", [64], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("y", [64], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("_tx", [32], dace.float64, storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array("_ty", [32], dace.float64, storage=StorageType.CuTile_Tile, transient=True)
    state = sdfg.add_state("main")
    entry, exit_node = state.add_map("cutile_map", {"tile_i": "0:64:32"}, schedule=ScheduleType.CuTile)
    x = state.add_read("x")
    y = state.add_write("y")
    tx = state.add_access("_tx")
    ty = state.add_access("_ty")
    tasklet = state.add_tasklet("scale", {"value"}, {"result"}, "result = value * alpha", language=Language.Python)
    state.add_memlet_path(x, entry, tx, memlet=Memlet("x[0:64]"))
    state.add_edge(tx, None, tasklet, "value", Memlet("_tx[0:32]"))
    state.add_edge(tasklet, "result", ty, None, Memlet("_ty[0:32]"))
    state.add_memlet_path(ty, exit_node, y, memlet=Memlet("y[0:64]"))
    sdfg.fill_scope_connectors()
    return sdfg


def _manual_two_map_sdfg(name: str) -> dace.SDFG:
    sdfg = _manual_scalar_sdfg(name)
    sdfg.add_array("z", [64], dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("_tz_in", [32], dace.float64, storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array("_tz_out", [32], dace.float64, storage=StorageType.CuTile_Tile, transient=True)
    first = sdfg.start_state
    second = sdfg.add_state("second")
    sdfg.add_edge(first, second, dace.InterstateEdge())
    entry, exit_node = second.add_map("cutile_map_2", {"tile_j": "0:64:32"}, schedule=ScheduleType.CuTile)
    y = second.add_read("y")
    z = second.add_write("z")
    tile_in = second.add_access("_tz_in")
    tile_out = second.add_access("_tz_out")
    tasklet = second.add_tasklet("shift", {"value"}, {"result"}, "result = value + 1.0", language=Language.Python)
    second.add_memlet_path(y, entry, tile_in, memlet=Memlet("y[0:64]"))
    second.add_edge(tile_in, None, tasklet, "value", Memlet("_tz_in[0:32]"))
    second.add_edge(tasklet, "result", tile_out, None, Memlet("_tz_out[0:32]"))
    second.add_memlet_path(tile_out, exit_node, z, memlet=Memlet("z[0:64]"))
    sdfg.fill_scope_connectors()
    return sdfg


def test_specs_are_frozen_and_module_is_empty_without_kernels():
    parameter = _array_parameter("x")
    kernel = _kernel((0, 1, 2, "sdfg"), "out = x", parameter)
    module = CuTileModuleSpec("empty")

    with pytest.raises(dataclasses.FrozenInstanceError):
        parameter.rank = 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        kernel.grid = ("2", )
    with pytest.raises(dataclasses.FrozenInstanceError):
        module.name = "other"
    assert module.render_codeobject(object) is None


def test_module_sorts_branches_and_attaches_symbol_metadata():
    parameter = _array_parameter("x")
    later = _kernel((2, 0, 0, "nested"), "later = x", parameter)
    earlier = _kernel((0, 3, 1, "top"), "earlier = x", parameter)
    module = CuTileModuleSpec("ordered").with_kernel(later).with_kernel(earlier)
    code_object = module.render_codeobject(object)

    assert code_object is not None
    assert code_object.language == "py"
    assert code_object.target_type == "cutile_build"
    assert code_object.linkable is False
    assert code_object.code.index("earlier = x") < code_object.code.index("later = x")
    assert code_object.code.count("@ct.kernel") == 1
    assert "compilation.ConstantConstraint(0)" in code_object.code
    assert "compilation.ConstantConstraint(1)" in code_object.code
    assert "compilation.TupleConstraint" in code_object.code
    assert "KernelSignature.from_kernel_args" not in code_object.code
    assert json.loads(code_object.extra_compiler_kwargs["cutile_symbols"]) == module.symbol_map


def test_generated_host_and_build_sources_have_separate_responsibilities():
    code_objects = _manual_scalar_sdfg("cutile_artifact_split").generate_code()
    host = next(code_object for code_object in code_objects if code_object.language == "pyx")
    build = next(code_object for code_object in code_objects if code_object.target_type == "cutile_build")

    assert len(code_objects) == 2
    assert "import cuda.tile" not in host.code
    assert "@ct.kernel" not in host.code
    assert "ct.launch(" not in host.code
    assert "cupy.asarray(alpha" not in host.code
    assert 'cdef extern from "dace_cutile_embedded.h"' in host.code
    assert "__dace_cutile_cubin" in host.code
    assert "__dace_cutile_cubin_size" in host.code
    assert "numpy.float64" in host.code
    assert "outside float64 range" in host.code
    assert "__cuda_array_interface__" in host.code
    assert "unsupported byte stride" in host.code
    assert "cupy.cuda.get_current_stream()" in host.code
    assert "(1, 1, 1)" in host.code

    assert build.code.count("@ct.kernel") == 1
    assert "def __dace_cutile_module(kernel_id: ct.Constant[int], args):" in build.code
    assert "compilation.ArrayConstraint" in build.code
    assert "index_dtype=ct.int64" in build.code
    assert "stride_lower_bound_incl=0" in build.code
    assert "compilation.ScalarConstraint(ct.float64)" in build.code
    assert "compilation.ConstantConstraint(0)" in build.code
    assert "compilation.TupleConstraint" in build.code
    assert "KernelSignature.from_kernel_args" not in build.code
    assert "parser.add_argument('--output', required=True)" in build.code
    assert "parser.add_argument('--arch', required=True)" in build.code


def test_float64_scalar_pack_accepts_huge_integer_and_reports_range(tmp_path: Path) -> None:
    """Generated float packing handles arbitrary-size Python integers."""
    parameter = CuTileParameterSpec(
        name="alpha",
        host_expression="alpha",
        kind=CuTileParameterKind.ReadOnlyScalar,
        dace_dtype="float64",
        cutile_dtype="float64",
        rank=0,
        index_dtype="int64",
        shape_constraints=(),
        stride_constraints=(),
        alias_groups=(),
        may_alias_internally=False,
        is_read=True,
        is_written=False,
        source="alpha",
    )
    pack_lines = artifacts._render_scalar_pack(0, parameter)
    source = "import numpy\n\n" + "\n".join([
        "def pack_float64(__dace_value_0):",
        "    cdef list __dace_args = []",
        "    cdef object __dace_packed_0",
        *pack_lines,
        "    return __dace_args[0]",
    ]) + "\n"
    sdfg = dace.SDFG("cutile_scalar_pack_test")
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_state("empty")
    sdfg.build_folder = str(tmp_path / "build")
    code_object = CodeObject(sdfg.name, source, "pyx", None, "Frame")
    build = build_python_extension(sdfg, [code_object])
    module = _load_native_module(build.extension_path, build.module_name)

    assert module.pack_float64(2**100) == float(2**100)
    with pytest.raises(OverflowError, match="cuTile scalar alpha is outside float64 range"):
        module.pack_float64(10**1000)


def test_multiple_maps_still_emit_one_build_module_with_unique_symbols():
    code_objects = _manual_two_map_sdfg("cutile_two_maps").generate_code()
    build_objects = [code_object for code_object in code_objects if code_object.target_type == "cutile_build"]
    assert len(build_objects) == 1

    build = build_objects[0]
    symbols = json.loads(build.extra_compiler_kwargs["cutile_symbols"])
    assert len(symbols) == 2
    assert len(set(symbols)) == 2
    assert build.code.count("@ct.kernel") == 1
    assert build.code.count("compilation.KernelSignature(") == 2
    assert "if kernel_id == 0:" in build.code
    assert "elif kernel_id == 1:" in build.code


def test_generated_artifacts_are_deterministic():
    first = _manual_two_map_sdfg("cutile_deterministic").generate_code()
    second = _manual_two_map_sdfg("cutile_deterministic").generate_code()
    first_build = next(code for code in first if code.target_type == "cutile_build")
    second_build = next(code for code in second if code.target_type == "cutile_build")
    assert first_build.code == second_build.code
    assert first_build.extra_compiler_kwargs == second_build.extra_compiler_kwargs
