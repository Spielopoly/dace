# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the native Python-backend compiler."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest
import dace
from dace import dtypes
from dace.codegen.codeobject import CodeObject
import dace.codegen.py.compiler as native_compiler
from dace.codegen.py.cutile_build import CuTileBuildArtifacts
from dace.codegen.py.compiler import build_python_extension
from dace.properties import CodeBlock


def _empty_sdfg(name: str, build_folder) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_state('state')
    sdfg.backend = dtypes.BackendLanguage.Python
    sdfg.build_folder = str(build_folder)
    return sdfg


def test_generated_host_is_one_self_contained_pyx(tmp_path):
    sdfg = _empty_sdfg('one_host', tmp_path / 'one_host')
    code_objects = sdfg.generate_code()

    assert len(code_objects) == 1
    assert code_objects[0].language == 'pyx'
    assert 'def int_ceil' in code_objects[0].code
    assert 'import *' not in code_objects[0].code
    assert 'sympy_function_redefinitions' not in code_objects[0].code


def test_build_manifest_and_cache_are_deterministic(tmp_path):
    sdfg = _empty_sdfg('native_cache', tmp_path / 'native_cache')
    code_objects = sdfg.generate_code()

    first = build_python_extension(sdfg, code_objects)
    second = build_python_extension(sdfg, code_objects)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.module_name == second.module_name
    assert first.extension_path == second.extension_path
    assert first.module_name.startswith('_dace_py_')
    assert first.extension_path.is_file()
    assert first.source_path.suffix == '.pyx'
    assert not list(first.manifest_path.parent.rglob('setup.py'))

    manifest = json.loads(first.manifest_path.read_text(encoding='utf-8'))
    assert manifest['module_name'] == first.module_name
    assert manifest['extension_file'] == first.extension_path.name
    assert manifest['host_source_file'] == first.source_path.name


def test_source_change_changes_internal_module_name(tmp_path):
    first_sdfg = _empty_sdfg('native_hash', tmp_path / 'native_hash')
    first_objects = first_sdfg.generate_code()
    first = build_python_extension(first_sdfg, first_objects)

    first_objects[0].code += '\nNATIVE_HASH_SENTINEL = 1\n'
    second = build_python_extension(first_sdfg, first_objects)
    assert first.module_name != second.module_name


@pytest.mark.parametrize('variable', native_compiler._SETUPTOOLS_ENVIRONMENT_OVERRIDES)
def test_setuptools_environment_override_changes_build_digest(monkeypatch, tmp_path, variable):
    sdfg = _empty_sdfg('toolchain_environment_hash', tmp_path / 'toolchain_environment_hash')
    code_objects = sdfg.generate_code()
    for name in native_compiler._SETUPTOOLS_ENVIRONMENT_OVERRIDES:
        monkeypatch.delenv(name, raising=False)

    baseline_inputs = native_compiler._build_inputs(sdfg, code_objects, (), ())
    baseline_digest = native_compiler._input_digest(baseline_inputs)

    override = f'dace-{variable.lower()}-override'
    monkeypatch.setenv(variable, override)
    overridden_inputs = native_compiler._build_inputs(sdfg, code_objects, (), ())

    assert native_compiler._input_digest(overridden_inputs) != baseline_digest
    compiler_identity = overridden_inputs['toolchain']['compiler']
    assert compiler_identity['environment_overrides'][variable] == override
    if variable == 'CC':
        assert compiler_identity['command'] == override


def test_cupy_provider_distribution_version_changes_build_digest(monkeypatch, tmp_path):
    sdfg = _empty_sdfg('cupy_provider_hash', tmp_path / 'cupy_provider_hash')
    code_objects = sdfg.generate_code()
    versions = {'cupy-cuda13x': '14.1.1'}
    package_version = native_compiler._package_version

    def _version(distribution):
        if distribution in versions:
            return versions[distribution]
        return package_version(distribution)

    monkeypatch.setattr(native_compiler.importlib.metadata, 'packages_distributions',
                        lambda: {'cupy': ['cupy-cuda13x']})
    monkeypatch.setattr(native_compiler, '_package_version', _version)

    first_inputs = native_compiler._build_inputs(sdfg, code_objects, (), ())
    assert first_inputs['toolchain']['cupy'] == {'cupy-cuda13x': '14.1.1'}

    versions['cupy-cuda13x'] = '14.1.2'
    second_inputs = native_compiler._build_inputs(sdfg, code_objects, (), ())

    assert second_inputs['toolchain']['cupy'] == {'cupy-cuda13x': '14.1.2'}
    assert native_compiler._input_digest(second_inputs) != native_compiler._input_digest(first_inputs)


def test_cutile_cache_hit_skips_both_compilers(monkeypatch, tmp_path):
    from dace.codegen.py import cutile_build

    sdfg = _empty_sdfg('cutile_cache', tmp_path / 'cutile_cache')
    host = CodeObject(
        name='host',
        code='def cutile_cache():\n    pass\n',
        language='pyx',
        target=None,
        title='host',
    )
    cutile = CodeObject(
        name='cutile',
        code='raise AssertionError("build source is mocked")\n',
        language='py',
        target=None,
        title='cuTile build',
        target_type='cutile_build',
        additional_compiler_kwargs={'cutile_symbols': '{"symbol": "map"}'},
        linkable=False,
    )
    calls = {'cutile': 0, 'cython': 0}

    def _fake_cutile(code_objects, build_dir, arch, *, sdfg_name):
        del code_objects, sdfg_name
        calls['cutile'] += 1
        artifact_dir = Path(build_dir) / 'artifacts'
        artifact_dir.mkdir(parents=True)
        cubin = artifact_dir / 'module.cubin'
        cubin.write_bytes(b'cubin')
        header = artifact_dir / 'dace_cutile_embedded.h'
        header.write_text('extern int cubin;\n', encoding='utf-8')
        source = artifact_dir / 'dace_cutile_embedded.c'
        source.write_text('int cubin = 1;\n', encoding='utf-8')
        return CuTileBuildArtifacts(arch=arch, cubin=cubin, header=header, source=source)

    def _fake_compile(module_name, pyx_path, extra_sources, include_dirs, build_dir):
        del pyx_path, extra_sources, include_dirs
        calls['cython'] += 1
        output = Path(build_dir) / f'{module_name}.so'
        output.parent.mkdir(parents=True)
        output.write_bytes(b'extension')
        return output

    monkeypatch.setattr(cutile_build, 'resolve_cutile_arch', lambda arch=None: 'sm_90')
    monkeypatch.setattr(cutile_build, 'build_cutile_artifacts', _fake_cutile)
    monkeypatch.setattr(native_compiler, '_compile_extension', _fake_compile)

    first = build_python_extension(sdfg, [host, cutile])
    second = build_python_extension(sdfg, [host, cutile])

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert calls == {'cutile': 1, 'cython': 1}
    manifest = json.loads(first.manifest_path.read_text(encoding='utf-8'))
    assert len(manifest['derived_files']) == 3


def test_public_compile_reuses_semantic_cache_across_deepcopies(monkeypatch, tmp_path):
    sdfg = _empty_sdfg('public_semantic_cache', tmp_path / 'public_semantic_cache')
    sdfg.add_array('A', [1], dace.int32)
    sdfg.add_array('B', [1], dace.int32)
    state = sdfg.states()[0]
    read = state.add_read('A')
    write = state.add_write('B')
    tasklet = state.add_tasklet('increment', {'value': None}, {'result': None}, 'result = value + 1')
    state.add_edge(read, None, tasklet, 'value', dace.Memlet('A[0]'))
    state.add_edge(tasklet, 'result', write, None, dace.Memlet('B[0]'))

    compiled_modules = []

    def _fake_compile(module_name, pyx_path, extra_sources, include_dirs, build_dir):
        del pyx_path, extra_sources, include_dirs
        compiled_modules.append(module_name)
        output = Path(build_dir) / f'{module_name}.so'
        output.parent.mkdir(parents=True)
        output.write_bytes(b'extension')
        return output

    monkeypatch.setattr(native_compiler, '_compile_extension', _fake_compile)

    assert sdfg.compile(return_program_handle=False) is None
    assert sdfg.compile(return_program_handle=False) is None
    assert len(compiled_modules) == 1

    cache_root = Path(sdfg.build_folder) / 'python'
    cache_dirs = [path for path in cache_root.iterdir() if path.is_dir() and not path.name.startswith('.')]
    assert len(cache_dirs) == 1

    sdfg.add_constant('meaningful_change', 1)
    assert sdfg.compile(return_program_handle=False) is None
    assert len(compiled_modules) == 2
    cache_dirs = [path for path in cache_root.iterdir() if path.is_dir() and not path.name.startswith('.')]
    assert len(cache_dirs) == 2


def test_deployed_output_is_self_describing_without_program_handle(tmp_path):
    sdfg = _empty_sdfg('self_describing_output', tmp_path / 'self_describing_output')
    output = tmp_path / 'renamed_extension.so'

    assert sdfg.compile(output_file=output, return_program_handle=False) is None
    assert output.is_file()

    manifest_path = native_compiler.deployed_extension_manifest_path(output)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['module_name'].startswith('_dace_py_')
    assert manifest['module_name'].isidentifier()
    assert manifest['extension_file'] == output.name
    assert manifest['extension_sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()

    module = native_compiler.load_deployed_extension(output)
    getattr(module, sdfg.name)()

    unsafe_manifest = dict(manifest)
    unsafe_manifest['extension_file'] = f'../{output.name}'
    manifest_path.write_text(json.dumps(unsafe_manifest), encoding='utf-8')
    with pytest.raises(native_compiler.NativePythonCompileError, match='Unsafe extension filename'):
        native_compiler.load_deployed_extension(output)

    tampered_manifest = dict(manifest)
    tampered_manifest['extension_sha256'] = '0' * 64
    manifest_path.write_text(json.dumps(tampered_manifest), encoding='utf-8')
    with pytest.raises(native_compiler.NativePythonCompileError, match='hash mismatch'):
        native_compiler.load_deployed_extension(output)


def test_native_failure_retains_source_output_and_cutile_context(monkeypatch, tmp_path):
    from dace.codegen.py import cutile_build

    sdfg = _empty_sdfg('native_failure', tmp_path / 'native_failure')
    host = CodeObject(
        name='host',
        code='def native_failure():\n    pass\n',
        language='pyx',
        target=None,
        title='host',
    )
    symbol_map = {'exported_kernel': 'native_failure:cfg=0/state=failing/map=broken'}
    cutile = CodeObject(
        name='cutile',
        code='raise AssertionError("cuTile export is mocked")\n',
        language='py',
        target=None,
        title='cuTile build',
        target_type='cutile_build',
        additional_compiler_kwargs={'cutile_symbols': json.dumps(symbol_map)},
        linkable=False,
    )

    def _fake_cutile(code_objects, build_dir, arch, *, sdfg_name):
        del code_objects, sdfg_name
        assert arch == 'sm_90'
        artifact_dir = Path(build_dir) / 'artifacts'
        artifact_dir.mkdir(parents=True)
        cubin = artifact_dir / 'module.cubin'
        cubin.write_bytes(b'cubin')
        header = artifact_dir / 'dace_cutile_embedded.h'
        header.write_text('extern int cubin;\n', encoding='utf-8')
        source = artifact_dir / 'dace_cutile_embedded.c'
        source.write_text('int cubin = 1;\n', encoding='utf-8')
        return CuTileBuildArtifacts(arch=arch, cubin=cubin, header=header, source=source)

    def _failing_compile(module_name, pyx_path, extra_sources, include_dirs, build_dir):
        del module_name, pyx_path, extra_sources, include_dirs, build_dir
        print('python compiler diagnostic sentinel')
        os.write(2, b'child compiler stderr sentinel\n')
        raise RuntimeError('compiler crashed sentinel')

    monkeypatch.setattr(cutile_build, 'resolve_cutile_arch', lambda arch=None: 'sm_90')
    monkeypatch.setattr(cutile_build, 'build_cutile_artifacts', _fake_cutile)
    monkeypatch.setattr(native_compiler, '_compile_extension', _failing_compile)

    with pytest.raises(native_compiler.NativePythonCompileError) as captured:
        build_python_extension(sdfg, [host, cutile])

    message = str(captured.value)
    assert 'sm_90' in message
    assert 'exported_kernel -> native_failure:cfg=0/state=failing/map=broken' in message
    assert 'python compiler diagnostic sentinel' in message
    assert 'child compiler stderr sentinel' in message
    assert 'compiler crashed sentinel' in message

    failure_root = Path(sdfg.build_folder) / 'python' / 'failed'
    failures = list(failure_root.iterdir())
    assert len(failures) == 1
    failure_dir = failures[0]
    retained_sources = list(failure_dir.glob('*.pyx'))
    assert len(retained_sources) == 1
    assert retained_sources[0].read_text(encoding='utf-8') == host.code
    assert not (failure_dir / 'manifest.json').exists()

    output = (failure_dir / 'compiler-output.txt').read_text(encoding='utf-8')
    assert 'python compiler diagnostic sentinel' in output
    assert 'child compiler stderr sentinel' in output
    record = json.loads((failure_dir / 'failure.json').read_text(encoding='utf-8'))
    assert record['cutile_arch'] == 'sm_90'
    assert record['cutile_symbols'] == 'exported_kernel -> native_failure:cfg=0/state=failing/map=broken'
    assert record['error_type'] == 'RuntimeError'
    assert str(retained_sources[0]) in message
    assert str(failure_dir / 'failure.json') in message


def test_cutile_export_failure_retains_source_in_durable_stage(monkeypatch, tmp_path):
    from dace.codegen.py import cutile_build

    sdfg = _empty_sdfg('cutile_export_failure', tmp_path / 'cutile_export_failure')
    host = CodeObject(
        name='host',
        code='def cutile_export_failure():\n    pass\n',
        language='pyx',
        target=None,
        title='host',
    )
    symbol_map = {'failed_export': 'cutile_export_failure:cfg=0/state=0/map=failed'}
    build_source = 'raise RuntimeError("aggregate export failure sentinel")\n'
    cutile = CodeObject(
        name='cutile',
        code=build_source,
        language='py',
        target=None,
        title='cuTile build',
        target_type='cutile_build',
        additional_compiler_kwargs={'cutile_symbols': json.dumps(symbol_map)},
        linkable=False,
    )
    real_run = subprocess.run

    def _run(command, **kwargs):
        if command[1:3] == ['-I', '-B']:
            return subprocess.CompletedProcess(command, 2, stdout='export stdout', stderr='export stderr')
        return real_run(command, **kwargs)

    monkeypatch.setattr(cutile_build, 'resolve_cutile_arch', lambda arch=None: 'sm_90')
    monkeypatch.setattr(cutile_build.subprocess, 'run', _run)

    with pytest.raises(native_compiler.NativePythonCompileError) as captured:
        build_python_extension(sdfg, [host, cutile])

    failure_root = Path(sdfg.build_folder) / 'python' / 'failed'
    failures = list(failure_root.iterdir())
    assert len(failures) == 1
    failure_dir = failures[0]
    retained_sources = list(failure_dir.rglob(cutile_build.CUTILE_BUILD_SOURCE))
    assert len(retained_sources) == 1
    retained_source = retained_sources[0]
    assert retained_source.is_file()
    assert retained_source.read_text(encoding='utf-8') == build_source

    message = str(captured.value)
    diagnostic_prefix = 'cuTile build source: '
    diagnostic_line = next(line for line in message.splitlines() if line.startswith(diagnostic_prefix))
    assert Path(diagnostic_line.removeprefix(diagnostic_prefix)) == retained_source
    assert Path(diagnostic_line.removeprefix(diagnostic_prefix)).is_file()
    assert 'export stdout' in message
    assert 'export stderr' in message
    assert 'failed_export -> cutile_export_failure:cfg=0/state=0/map=failed' in message

    record = json.loads((failure_dir / 'failure.json').read_text(encoding='utf-8'))
    assert record['cutile_build_source_files'] == [retained_source.relative_to(failure_dir).as_posix()]


def test_cache_rejects_mismatched_valid_module_name(monkeypatch, tmp_path):
    sdfg = _empty_sdfg('module_name_corruption', tmp_path / 'module_name_corruption')
    code_objects = sdfg.generate_code()
    compile_calls = 0

    def _fake_compile(module_name, pyx_path, extra_sources, include_dirs, build_dir):
        nonlocal compile_calls
        del pyx_path, extra_sources, include_dirs
        compile_calls += 1
        output = Path(build_dir) / f'{module_name}.so'
        output.parent.mkdir(parents=True)
        output.write_bytes(b'extension')
        return output

    monkeypatch.setattr(native_compiler, '_compile_extension', _fake_compile)

    first = build_python_extension(sdfg, code_objects)
    manifest = json.loads(first.manifest_path.read_text(encoding='utf-8'))
    manifest['module_name'] = '_dace_py_valid_but_incorrect'
    first.manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

    second = build_python_extension(sdfg, code_objects)

    assert compile_calls == 2
    assert second.cache_hit is False
    repaired = json.loads(second.manifest_path.read_text(encoding='utf-8'))
    assert repaired['module_name'] == first.module_name
    assert second.module_name == first.module_name


def test_cache_publication_lock_serializes_processes(tmp_path):
    """The persistent advisory lock excludes a separate Python process."""
    lock_path = tmp_path / 'shared.lock'
    marker_path = tmp_path / 'child-acquired'
    child_source = "\n".join([
        'from pathlib import Path',
        'import sys',
        'from dace.codegen.py.compiler import _cache_publication_lock',
        'with _cache_publication_lock(Path(sys.argv[1])):',
        '    Path(sys.argv[2]).write_text("acquired", encoding="utf-8")',
    ])

    with native_compiler._cache_publication_lock(lock_path):
        child = subprocess.Popen(
            [
                sys.executable,
                '-B',
                '-c',
                child_source,
                str(lock_path),
                str(marker_path),
            ],
            cwd=Path.cwd(),
        )
        time.sleep(0.5)
        assert child.poll() is None
        assert not marker_path.exists()
    assert child.wait(timeout=10) == 0
    assert marker_path.read_text(encoding='utf-8') == 'acquired'


def test_same_key_concurrent_builds_validate_winner_and_serialize_capture(monkeypatch, tmp_path):
    sdfg = _empty_sdfg('concurrent_native_cache', tmp_path / 'concurrent_native_cache')
    code_objects = sdfg.generate_code()
    initial_checks = threading.Barrier(2)
    thread_state = threading.local()
    original_cached_build = native_compiler._cached_build
    compile_lock = threading.Lock()
    compile_calls = 0
    active_compilers = 0
    maximum_active_compilers = 0

    def _synchronized_cached_build(cache_dir, expected_inputs, source):
        result = original_cached_build(cache_dir, expected_inputs, source)
        if not getattr(thread_state, 'completed_initial_check', False):
            thread_state.completed_initial_check = True
            assert result is None
            initial_checks.wait(timeout=10)
        return result

    def _fake_compile(module_name, pyx_path, extra_sources, include_dirs, build_dir):
        nonlocal compile_calls, active_compilers, maximum_active_compilers
        del pyx_path, extra_sources, include_dirs
        with compile_lock:
            compile_calls += 1
            active_compilers += 1
            maximum_active_compilers = max(maximum_active_compilers, active_compilers)
        try:
            time.sleep(0.05)
            output = Path(build_dir) / f'{module_name}.so'
            output.parent.mkdir(parents=True)
            output.write_bytes(b'extension')
            return output
        finally:
            with compile_lock:
                active_compilers -= 1

    monkeypatch.setattr(native_compiler, '_cached_build', _synchronized_cached_build)
    monkeypatch.setattr(native_compiler, '_compile_extension', _fake_compile)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(build_python_extension, sdfg, code_objects) for _ in range(2)]
        results = [future.result(timeout=20) for future in futures]

    assert compile_calls == 2
    assert maximum_active_compilers == 1
    assert sorted(result.cache_hit for result in results) == [False, True]
    assert results[0].module_name == results[1].module_name
    assert results[0].extension_path == results[1].extension_path
    assert original_cached_build(results[0].manifest_path.parent,
                                 json.loads(results[0].manifest_path.read_text(encoding='utf-8'))['inputs'],
                                 results[0].source) is not None
    assert not (Path(sdfg.build_folder) / 'python' / 'failed').exists()


def test_stable_hash_canonicalizes_mixed_none_and_string_keys():
    sdfg = _empty_sdfg('mixed_global_code_keys', '.dacecache/mixed_global_code_keys')
    sdfg.global_code['frame'] = CodeBlock('FRAME_SCOPE = 1')
    sdfg.global_code[None] = CodeBlock('DEFAULT_SCOPE = 2')

    first_hash = native_compiler._stable_sdfg_hash(sdfg)

    frame_code = sdfg.global_code.pop('frame')
    default_code = sdfg.global_code.pop(None)
    sdfg.global_code[None] = default_code
    sdfg.global_code['frame'] = frame_code
    assert native_compiler._stable_sdfg_hash(sdfg) == first_hash

    sdfg.global_code[None] = CodeBlock('DEFAULT_SCOPE = 3')
    assert native_compiler._stable_sdfg_hash(sdfg) != first_hash

    with pytest.raises(native_compiler.NativePythonCompileError, match='Unsupported SDFG JSON mapping key.*tuple'):
        native_compiler._canonicalize_sdfg_json({'payload': {('unsupported', ): 'value'}})
