# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Acceptance tests for aggregate cuTile artifacts and deployed extensions."""

import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import dace
from dace import dtypes
from dace.codegen.codeobject import CodeObject
from dace.codegen.py.compiler import (build_python_extension, copy_compiled_extension, deployed_extension_manifest_path)
from dace.dtypes import Language, ScheduleType, StorageType
from dace.memlet import Memlet


def _add_cutile_map(sdfg: dace.SDFG, state: dace.SDFGState, input_name: str, output_name: str,
                    tile_shape: tuple[int, ...], ranges: dict[str, str], operation: str) -> None:
    """Add one manually lowered cuTile map to ``state``."""
    input_tile_name = f"_{input_name}_tile"
    output_tile_name = f"_{output_name}_tile"
    dtype = sdfg.arrays[input_name].dtype
    sdfg.add_array(input_tile_name, tile_shape, dtype, storage=StorageType.CuTile_Tile, transient=True)
    sdfg.add_array(output_tile_name, tile_shape, dtype, storage=StorageType.CuTile_Tile, transient=True)

    entry, exit_node = state.add_map(f"{output_name}_cutile", ranges, schedule=ScheduleType.CuTile)
    input_node = state.add_read(input_name)
    output_node = state.add_write(output_name)
    input_tile = state.add_access(input_tile_name)
    output_tile = state.add_access(output_tile_name)
    tasklet = state.add_tasklet(
        f"{output_name}_operation",
        {"value"},
        {"result"},
        f"result = value {operation}",
        language=Language.Python,
    )
    full_subset = ", ".join(f"0:{dimension}" for dimension in sdfg.arrays[input_name].shape)
    tile_subset = ", ".join(f"0:{dimension}" for dimension in tile_shape)
    state.add_memlet_path(input_node, entry, input_tile, memlet=Memlet(f"{input_name}[{full_subset}]"))
    state.add_edge(input_tile, None, tasklet, "value", Memlet(f"{input_tile_name}[{tile_subset}]"))
    state.add_edge(tasklet, "result", output_tile, None, Memlet(f"{output_tile_name}[{tile_subset}]"))
    state.add_memlet_path(output_tile, exit_node, output_node, memlet=Memlet(f"{output_name}[{full_subset}]"))


def _multi_entry_nested_sdfg(name: str) -> dace.SDFG:
    """Build one outer map and one independently invoked nested-map entry."""
    sdfg = dace.SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.add_array("x", (64, ), dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("y", (64, ), dace.float64, storage=StorageType.GPU_Global)
    sdfg.add_array("matrix_in", (8, 8), dace.float32, storage=StorageType.GPU_Global)
    sdfg.add_array("matrix_out", (8, 8), dace.float32, storage=StorageType.GPU_Global)

    outer_state = sdfg.add_state("outer_map")
    _add_cutile_map(sdfg, outer_state, "x", "y", (32, ), {"tile_i": "0:64:32"}, "+ 3.0")

    nested = dace.SDFG(f"{name}_nested")
    nested.add_array("nested_in", (8, 8), dace.float32, storage=StorageType.GPU_Global)
    nested.add_array("nested_out", (8, 8), dace.float32, storage=StorageType.GPU_Global)
    nested_state = nested.add_state("nested_map")
    _add_cutile_map(
        nested,
        nested_state,
        "nested_in",
        "nested_out",
        (4, 4),
        {
            "tile_i": "0:8:4",
            "tile_j": "0:8:4",
        },
        "* 2.0",
    )
    nested.fill_scope_connectors()

    invoke_state = sdfg.add_state("invoke_nested")
    sdfg.add_edge(outer_state, invoke_state, dace.InterstateEdge())
    nested_node = invoke_state.add_nested_sdfg(nested, {"nested_in"}, {"nested_out"})
    matrix_in = invoke_state.add_read("matrix_in")
    matrix_out = invoke_state.add_write("matrix_out")
    invoke_state.add_edge(matrix_in, None, nested_node, "nested_in", Memlet("matrix_in[0:8, 0:8]"))
    invoke_state.add_edge(nested_node, "nested_out", matrix_out, None, Memlet("matrix_out[0:8, 0:8]"))
    sdfg.fill_scope_connectors()
    return sdfg


def _host_and_build(sdfg: dace.SDFG) -> tuple[list[CodeObject], CodeObject, CodeObject]:
    code_objects = sdfg.generate_code()
    host_objects = [code_object for code_object in code_objects if code_object.language == "pyx"]
    build_objects = [code_object for code_object in code_objects if code_object.target_type == "cutile_build"]
    assert len(host_objects) == 1
    assert len(build_objects) == 1
    assert len(code_objects) == 2
    return code_objects, host_objects[0], build_objects[0]


def test_outer_and_independent_nested_maps_share_one_aggregate_artifact() -> None:
    _, first_host, first_build = _host_and_build(_multi_entry_nested_sdfg("nested_aggregate"))
    _, second_host, second_build = _host_and_build(_multi_entry_nested_sdfg("nested_aggregate"))

    symbols = json.loads(first_build.extra_compiler_kwargs["cutile_symbols"])
    assert len(symbols) == 2
    assert len(set(symbols)) == 2
    assert any("nested_aggregate:cfg=" in identity for identity in symbols.values())
    assert any("nested_aggregate_nested:cfg=" in identity for identity in symbols.values())
    assert first_build.code.count("@ct.kernel") == 1
    assert first_build.code.count("compilation.KernelSignature(") == 2
    assert re.search(r"compilation\.ArrayConstraint\(ct\.float64,\s+1,", first_build.code)
    assert re.search(r"compilation\.ArrayConstraint\(ct\.float32,\s+2,", first_build.code)
    assert first_host.code.count("cdef void __dace_cutile_launch_") == 2
    assert first_build.code == second_build.code
    assert first_build.extra_compiler_kwargs == second_build.extra_compiler_kwargs
    assert first_host.code.count("cdef void __dace_cutile_launch_") == second_host.code.count(
        "cdef void __dace_cutile_launch_")


_DEPLOYED_PROCESS = textwrap.dedent(r"""
    import hashlib
    import importlib.abc
    import importlib.util
    import json
    import pathlib
    import sys

    extension_path = pathlib.Path(sys.argv[1]).resolve(strict=True)
    manifest_path = extension_path.with_name(extension_path.name + ".dace.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    module_name = manifest["module_name"]
    program_name = manifest["sdfg_name"]
    generated_build_module = sys.argv[2]
    if manifest["extension_file"] != extension_path.name:
        raise RuntimeError("deployment manifest extension filename does not match")
    extension_hash = hashlib.sha256(extension_path.read_bytes()).hexdigest()
    if manifest["extension_sha256"] != extension_hash:
        raise RuntimeError("deployment extension hash does not match")
    blocked_prefixes = (
        "cuda.tile",
        "dace.codegen.py.cutile_build",
        "dace.codegen.py.compiler",
        generated_build_module,
        "dace_cutile_build",
    )

    class _BlockBuildImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in blocked_prefixes):
                raise ImportError(f"runtime attempted forbidden build import {fullname!r}")
            return None

    sys.meta_path.insert(0, _BlockBuildImports())
    import cupy
    import numpy

    spec = importlib.util.spec_from_file_location(module_name, extension_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load deployed extension {extension_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    x_storage = cupy.arange(128, dtype=cupy.float64)
    y_storage = cupy.empty(128, dtype=cupy.float64)
    x = x_storage[::2]
    y = y_storage[::2]
    matrix_in = cupy.arange(64, dtype=cupy.float32).reshape(8, 8)
    matrix_out = cupy.empty_like(matrix_in)
    if x.strides[0] <= x.dtype.itemsize:
        raise AssertionError("test input is not a positive-stride CUDA view")

    getattr(module, program_name)(x=x, y=y, matrix_in=matrix_in, matrix_out=matrix_out)
    cupy.cuda.get_current_stream().synchronize()
    numpy.testing.assert_allclose(cupy.asnumpy(y), cupy.asnumpy(x) + 3.0, rtol=0.0, atol=0.0)
    numpy.testing.assert_allclose(cupy.asnumpy(matrix_out), cupy.asnumpy(matrix_in) * 2.0, rtol=0.0, atol=0.0)
    if any(
        name == prefix or name.startswith(prefix + ".")
        for name in sys.modules
        for prefix in blocked_prefixes
    ):
        raise AssertionError("a forbidden build-time module was imported")
""")


@pytest.mark.gpu
def test_deployed_multi_entry_extension_and_manifest_are_self_contained(tmp_path: Path) -> None:
    sdfg = _multi_entry_nested_sdfg("deployed_multi_entry")
    sdfg.build_folder = str(tmp_path / "build-cache")
    code_objects, _, build_object = _host_and_build(sdfg)
    build = build_python_extension(sdfg, code_objects)

    deployment_dir = tmp_path / "deployment"
    deployment_dir.mkdir()
    deployed_extension = copy_compiled_extension(build, deployment_dir, sdfg.name)
    deployed_manifest = deployed_extension_manifest_path(deployed_extension)
    deployed_files = list(deployment_dir.iterdir())
    assert set(deployed_files) == {deployed_extension, deployed_manifest}
    assert not list(deployment_dir.glob("*.py"))
    assert not list(deployment_dir.glob("*.cubin"))

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _DEPLOYED_PROCESS,
            str(deployed_extension),
            build_object.name,
        ],
        cwd=deployment_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"deployed process failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@pytest.mark.gpu
def test_embedded_launch_uses_nondefault_current_stream() -> None:
    """Every kernel launch and the successful-exit sync use the active stream."""
    import cupy

    sdfg = _multi_entry_nested_sdfg("nondefault_stream")
    compiled = sdfg.compile()
    observed_streams: list[int] = []
    real_cupy = cupy

    class _CudaProxy:

        def __getattr__(self, name: str):
            return getattr(real_cupy.cuda, name)

        def get_current_stream(self):
            current = real_cupy.cuda.get_current_stream()
            observed_streams.append(current.ptr)
            return current

    class _CupyProxy:

        cuda = _CudaProxy()

        def __getattr__(self, name: str):
            return getattr(real_cupy, name)

    compiled.module.cupy = _CupyProxy()
    stream = cupy.cuda.Stream(non_blocking=True)
    x = cupy.arange(64, dtype=cupy.float64)
    y = cupy.empty_like(x)
    matrix_in = cupy.arange(64, dtype=cupy.float32).reshape(8, 8)
    matrix_out = cupy.empty_like(matrix_in)

    with stream:
        compiled(x=x, y=y, matrix_in=matrix_in, matrix_out=matrix_out)

    assert observed_streams
    assert set(observed_streams) == {stream.ptr}
    cupy.testing.assert_array_equal(y, x + 3.0)
    cupy.testing.assert_array_equal(matrix_out, matrix_in * 2.0)
