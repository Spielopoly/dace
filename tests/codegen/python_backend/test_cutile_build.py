# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import stat
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from dace.codegen.codeobject import CodeObject
from dace.codegen.py import cutile_build
from dace.codegen.py.cutile_build import CuTileBuildError

_SYMBOLS = {
    '__dace_cutile_map_0': 'state 0/map 3',
    '__dace_cutile_map_1': 'nested/state 2/map 4',
}


def _build_code_object(*,
                       symbols=_SYMBOLS,
                       linkable=False,
                       language='py',
                       target_type='cutile_build',
                       code='raise RuntimeError("build source should run only in a subprocess")\n') -> CodeObject:
    kwargs = {} if symbols is None else {'cutile_symbols': json.dumps(symbols)}
    return CodeObject(name='aggregate_cutile',
                      code=code,
                      language=language,
                      target=None,
                      title='Aggregate cuTile build',
                      target_type=target_type,
                      additional_compiler_kwargs=kwargs,
                      linkable=linkable)


def test_select_cutile_build_codeobject_is_optional_and_decodes_symbols():
    assert cutile_build.select_cutile_build_codeobject([]) is None

    code_object = _build_code_object()
    selected = cutile_build.select_cutile_build_codeobject([code_object])

    assert selected.code_object is code_object
    assert selected.expected_symbols == _SYMBOLS


def test_select_cutile_build_codeobject_rejects_multiple_objects():
    with pytest.raises(CuTileBuildError, match='at most one'):
        cutile_build.select_cutile_build_codeobject([_build_code_object(), _build_code_object()])


@pytest.mark.parametrize(
    'code_object, message',
    [
        (_build_code_object(symbols=None), 'cutile_symbols'),
        (_build_code_object(symbols={}), 'nonempty JSON object'),
        (_build_code_object(symbols={'symbol': ''}), 'map identity'),
        (_build_code_object(linkable=True), 'linkable=False'),
        (_build_code_object(language='cpp'), 'Python source'),
    ],
)
def test_select_cutile_build_codeobject_rejects_invalid_input(code_object, message):
    with pytest.raises(CuTileBuildError, match=message):
        cutile_build.select_cutile_build_codeobject([code_object])


def test_select_cutile_build_codeobject_rejects_malformed_and_duplicate_metadata():
    malformed = _build_code_object()
    malformed.extra_compiler_kwargs['cutile_symbols'] = '{'
    with pytest.raises(CuTileBuildError, match='invalid cuTile symbol metadata'):
        cutile_build.select_cutile_build_codeobject([malformed])

    duplicate = _build_code_object()
    duplicate.extra_compiler_kwargs['cutile_symbols'] = '{"same": "map 0", "same": "map 1"}'
    with pytest.raises(CuTileBuildError, match='duplicate exported symbol'):
        cutile_build.select_cutile_build_codeobject([duplicate])


@pytest.mark.parametrize('arch', ['sm_75', 'sm_90', 'sm_120'])
def test_resolve_cutile_arch_accepts_explicit_arch_without_gpu(monkeypatch, arch):
    monkeypatch.setattr(cutile_build, '_query_active_device_arch', lambda: pytest.fail('queried the GPU'))
    assert cutile_build.resolve_cutile_arch(arch) == arch


@pytest.mark.parametrize('arch', ['90', 'compute_90', 'sm_9', 'sm_0900', 'sm_90a', 'sm_-1', '', 'auto '])
def test_resolve_cutile_arch_rejects_invalid_explicit_arch(arch):
    with pytest.raises(ValueError, match='expected'):
        cutile_build.resolve_cutile_arch(arch)


def test_resolve_cutile_arch_auto_queries_active_cupy_device(monkeypatch):
    fake_cupy = SimpleNamespace(cuda=SimpleNamespace(Device=lambda: SimpleNamespace(compute_capability='12.0')))
    monkeypatch.setitem(sys.modules, 'cupy', fake_cupy)

    assert cutile_build.resolve_cutile_arch('auto') == 'sm_120'


def test_resolve_cutile_arch_uses_auto_configuration(monkeypatch):
    monkeypatch.setattr(cutile_build.Config, 'get', lambda *path: 'auto')
    monkeypatch.setattr(cutile_build, '_query_active_device_arch', lambda: 'sm_90')

    assert cutile_build.resolve_cutile_arch() == 'sm_90'


def test_resolve_cutile_arch_auto_failure_is_actionable(monkeypatch):

    def _fail_device():
        raise RuntimeError('no CUDA device')

    fake_cupy = SimpleNamespace(cuda=SimpleNamespace(Device=_fail_device))
    monkeypatch.setitem(sys.modules, 'cupy', fake_cupy)

    with pytest.raises(CuTileBuildError, match=r"compiler\.cutile\.arch.*set compiler\.cutile\.arch"):
        cutile_build.resolve_cutile_arch('auto')


def _selected_input():
    return cutile_build.select_cutile_build_codeobject([_build_code_object()])


def test_export_cutile_cubin_runs_one_isolated_export_and_verifies_symbols(monkeypatch, tmp_path):
    calls = []

    def _run(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == sys.executable:
            output_path = Path(command[command.index('--output') + 1])
            output_path.write_bytes(b'aggregate-cubin')
            return subprocess.CompletedProcess(command, 0, stdout='exported once\n', stderr='')
        assert command[:2] == ['cuobjdump', '--dump-elf-symbols']
        symbols = '\n'.join(f'STT_FUNC STB_GLOBAL STO_ENTRY {symbol}' for symbol in _SYMBOLS)
        return subprocess.CompletedProcess(command, 0, stdout=symbols, stderr='')

    monkeypatch.setattr(cutile_build.subprocess, 'run', _run)
    monkeypatch.setenv('PYTHONPATH', '/untrusted/source')
    cubin_path = cutile_build.export_cutile_cubin(_selected_input(), tmp_path, 'sm_90', sdfg_name='multi_map')

    assert cubin_path.name == cutile_build.CUTILE_CUBIN
    assert cubin_path.parent.parent == tmp_path
    assert cubin_path.parent.name.startswith('.dace-cutile-artifacts-')
    assert cubin_path.is_absolute()
    assert cubin_path.read_bytes() == b'aggregate-cubin'
    assert (cubin_path.parent / cutile_build.CUTILE_BUILD_SOURCE).is_file()
    export_calls = [call for call in calls if call[0][0] == sys.executable]
    assert len(export_calls) == 1
    command, kwargs = export_calls[0]
    assert command[1:3] == ['-I', '-B']
    assert command[-2:] == ['--arch', 'sm_90']
    assert kwargs['cwd'] == str(Path(command[command.index('--output') + 1]).parent)
    assert 'PYTHONPATH' not in kwargs['env']
    assert kwargs['env']['PYTHONNOUSERSITE'] == '1'


def test_export_cutile_cubin_resolves_relative_paths_for_real_subprocess(monkeypatch, tmp_path):
    build_source = '''import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--arch", required=True)
arguments = parser.parse_args()
Path(arguments.output).write_bytes(("real-" + arguments.arch).encode("ascii"))
'''
    build_input = cutile_build.select_cutile_build_codeobject([_build_code_object(code=build_source)])
    verified_paths = []

    def _verify(cubin_path, expected_symbols, arch, *, sdfg_name):
        verified_paths.append(Path(cubin_path))

    monkeypatch.setattr(cutile_build, 'verify_cubin_symbols', _verify)
    monkeypatch.chdir(tmp_path)

    cubin_path = cutile_build.export_cutile_cubin(build_input, Path('relative-build'), 'sm_90')

    assert cubin_path.is_absolute()
    assert cubin_path.parent.parent == (tmp_path / 'relative-build').resolve()
    assert cubin_path.read_bytes() == b'real-sm_90'
    assert verified_paths == [cubin_path]


@pytest.mark.parametrize('contents, message', [(None, 'produced 0 cubins'), (b'', 'empty cubin')])
def test_export_cutile_cubin_rejects_missing_or_empty_output(monkeypatch, tmp_path, contents, message):

    def _run(command, **kwargs):
        if contents is not None:
            Path(command[command.index('--output') + 1]).write_bytes(contents)
        return subprocess.CompletedProcess(command, 0, stdout='', stderr='')

    monkeypatch.setattr(cutile_build.subprocess, 'run', _run)

    with pytest.raises(CuTileBuildError, match=message):
        cutile_build.export_cutile_cubin(_selected_input(), tmp_path, 'sm_90', sdfg_name='broken_sdfg')


def test_export_cutile_cubin_rejects_more_than_one_cubin(monkeypatch, tmp_path):

    def _run(command, **kwargs):
        output_path = Path(command[command.index('--output') + 1])
        output_path.write_bytes(b'aggregate')
        (output_path.parent / 'per_kernel.cubin').write_bytes(b'forbidden')
        return subprocess.CompletedProcess(command, 0, stdout='', stderr='')

    monkeypatch.setattr(cutile_build.subprocess, 'run', _run)

    with pytest.raises(CuTileBuildError, match='2 cubins instead of exactly one'):
        cutile_build.export_cutile_cubin(_selected_input(), tmp_path, 'sm_90')


def test_export_cutile_cubin_reports_source_map_symbol_arch_and_output(monkeypatch, tmp_path):

    def _run(command, **kwargs):
        return subprocess.CompletedProcess(command, 2, stdout='compiler stdout', stderr='compiler stderr')

    monkeypatch.setattr(cutile_build.subprocess, 'run', _run)

    with pytest.raises(CuTileBuildError) as error:
        cutile_build.export_cutile_cubin(_selected_input(), tmp_path, 'sm_120', sdfg_name='failing_sdfg')

    message = str(error.value)
    assert 'dace_cutile_build.py' in message
    assert '__dace_cutile_map_0' in message
    assert 'state 0/map 3' in message
    assert 'sm_120' in message
    assert 'compiler stdout' in message
    assert 'compiler stderr' in message
    retained_sources = list(tmp_path.rglob(cutile_build.CUTILE_BUILD_SOURCE))
    assert len(retained_sources) == 1
    retained_source = retained_sources[0]
    assert retained_source.is_file()
    assert 'build source should run only in a subprocess' in retained_source.read_text(encoding='utf-8')
    assert str(retained_source) in message


def test_verify_cubin_symbols_checks_exact_symbol_and_reports_map(monkeypatch, tmp_path):
    cubin = tmp_path / 'module.cubin'
    cubin.write_bytes(b'cubin')

    def _run(command, **kwargs):
        stdout = ('symbols:\n'
                  'STT_FUNC         STB_GLOBAL STO_ENTRY      __dace_cutile_map_0_suffix\n'
                  'STT_FUNC         STB_GLOBAL STO_ENTRY      __dace_cutile_map_1\n')
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr='')

    monkeypatch.setattr(cutile_build.subprocess, 'run', _run)

    with pytest.raises(CuTileBuildError) as error:
        cutile_build.verify_cubin_symbols(cubin, _SYMBOLS, 'sm_90', sdfg_name='symbol_sdfg')

    message = str(error.value)
    assert "map 'state 0/map 3'" in message
    assert "symbol '__dace_cutile_map_0'" in message
    assert "architecture 'sm_90'" in message
    assert "SDFG 'symbol_sdfg'" in message


def test_verify_cubin_symbols_accepts_realistic_global_entry_rows(monkeypatch, tmp_path):
    cubin = tmp_path / 'module.cubin'
    cubin.write_bytes(b'cubin')
    output = '''
symbols:
STT_CUDA_OBJECT  STB_LOCAL  STO_?          _param
STT_FUNC         STB_GLOBAL STO_ENTRY      __dace_cutile_map_0
STT_FUNC         STB_GLOBAL STO_ENTRY      __dace_cutile_map_1
STT_FUNC         STB_GLOBAL STV_DEFAULT  U unresolved_device_function
'''
    monkeypatch.setattr(cutile_build.subprocess, 'run',
                        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout=output, stderr=''))

    cutile_build.verify_cubin_symbols(cubin, _SYMBOLS, 'sm_90')


@pytest.mark.parametrize(
    'invalid_row',
    [
        'STT_OBJECT       STB_GLOBAL STO_ENTRY      __dace_cutile_map_0',
        'STT_FUNC         STB_LOCAL  STO_ENTRY      __dace_cutile_map_0',
        'STT_FUNC         STB_GLOBAL STV_DEFAULT  U __dace_cutile_map_0',
        'STT_FUNC         STB_GLOBAL STO_CONSTANT   __dace_cutile_map_0',
    ],
    ids=['object', 'local', 'undefined', 'non-entry'],
)
def test_verify_cubin_symbols_rejects_non_global_function_entries(monkeypatch, tmp_path, invalid_row):
    cubin = tmp_path / 'module.cubin'
    cubin.write_bytes(b'cubin')
    output = f'''symbols:
{invalid_row}
STT_FUNC         STB_GLOBAL STO_ENTRY      __dace_cutile_map_1
'''
    monkeypatch.setattr(cutile_build.subprocess, 'run',
                        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout=output, stderr=''))

    with pytest.raises(CuTileBuildError, match="map 'state 0/map 3'.*symbol '__dace_cutile_map_0'"):
        cutile_build.verify_cubin_symbols(cubin, _SYMBOLS, 'sm_90')


def test_write_cubin_embedding_is_fixed_name_and_deterministic(tmp_path):
    cubin = tmp_path / 'input.cubin'
    cubin.write_bytes(bytes([0, 1, 127, 128, 255]))

    header, source = cutile_build.write_cubin_embedding(cubin, tmp_path)
    first_header = header.read_text(encoding='utf-8')
    first_source = source.read_text(encoding='utf-8')
    second_header, second_source = cutile_build.write_cubin_embedding(cubin, tmp_path)

    assert header.name == 'dace_cutile_embedded.h'
    assert source.name == 'dace_cutile_embedded.c'
    assert second_header.read_text(encoding='utf-8') == first_header
    assert second_source.read_text(encoding='utf-8') == first_source
    assert 'extern const unsigned char __dace_cutile_cubin[];' in first_header
    assert 'extern const size_t __dace_cutile_cubin_size;' in first_header
    assert 'const unsigned char __dace_cutile_cubin[] = {' in first_source
    assert '0x00, 0x01, 0x7f, 0x80, 0xff,' in first_source
    assert 'const size_t __dace_cutile_cubin_size = sizeof(__dace_cutile_cubin);' in first_source


def test_write_cubin_embedding_rejects_empty_cubin(tmp_path):
    cubin = tmp_path / 'empty.cubin'
    cubin.write_bytes(b'')

    with pytest.raises(CuTileBuildError, match='cannot embed empty'):
        cutile_build.write_cubin_embedding(cubin, tmp_path)


def test_write_cubin_embedding_replaces_symlinks_without_following_them(tmp_path):
    cubin = tmp_path / 'input.cubin'
    cubin.write_bytes(b'cubin')
    external_header = tmp_path / 'external.h'
    external_source = tmp_path / 'external.c'
    external_header.write_text('header sentinel', encoding='utf-8')
    external_source.write_text('source sentinel', encoding='utf-8')
    header_link = tmp_path / cutile_build.CUTILE_EMBEDDED_HEADER
    source_link = tmp_path / cutile_build.CUTILE_EMBEDDED_SOURCE
    header_link.symlink_to(external_header)
    source_link.symlink_to(external_source)

    header, source = cutile_build.write_cubin_embedding(cubin, tmp_path)

    assert not header.is_symlink()
    assert not source.is_symlink()
    assert external_header.read_text(encoding='utf-8') == 'header sentinel'
    assert external_source.read_text(encoding='utf-8') == 'source sentinel'


def test_build_cutile_artifacts_isolates_concurrent_builds(monkeypatch, tmp_path):
    barrier = threading.Barrier(2)

    def _run(command, **kwargs):
        if command[0] == sys.executable:
            source_path = Path(command[3])
            marker = 'first' if '# first' in source_path.read_text(encoding='utf-8') else 'second'
            barrier.wait(timeout=5)
            output_path = Path(command[command.index('--output') + 1])
            output_path.write_bytes(marker.encode('ascii'))
            return subprocess.CompletedProcess(command, 0, stdout='', stderr='')
        output = '\n'.join(f'STT_FUNC STB_GLOBAL STO_ENTRY {symbol}' for symbol in _SYMBOLS)
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr='')

    monkeypatch.setattr(cutile_build.subprocess, 'run', _run)
    code_objects = ([_build_code_object(code='# first\n')], [_build_code_object(code='# second\n')])
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(cutile_build.build_cutile_artifacts, objects, tmp_path, 'sm_90') for objects in code_objects
        ]
        first, second = [future.result(timeout=10) for future in futures]

    assert first.cubin.parent != second.cubin.parent
    for artifacts, expected in ((first, b'first'), (second, b'second')):
        assert artifacts.cubin.parent == artifacts.header.parent == artifacts.source.parent
        assert artifacts.cubin.read_bytes() == expected
        assert artifacts.source.read_text(encoding='utf-8') == cutile_build._embedding_source(expected)
        assert stat.S_IMODE(artifacts.cubin.parent.stat().st_mode) == 0o700


def test_build_cutile_artifacts_ignores_preexisting_fixed_name_symlinks(monkeypatch, tmp_path):
    sentinels = {}
    for artifact_name in (cutile_build.CUTILE_BUILD_SOURCE, cutile_build.CUTILE_CUBIN,
                          cutile_build.CUTILE_EMBEDDED_HEADER, cutile_build.CUTILE_EMBEDDED_SOURCE):
        sentinel = tmp_path / f'{artifact_name}.sentinel'
        sentinel.write_bytes(f'outside:{artifact_name}'.encode('ascii'))
        fixed_path = tmp_path / artifact_name
        fixed_path.symlink_to(sentinel)
        sentinels[fixed_path] = sentinel

    def _run(command, **kwargs):
        if command[0] == sys.executable:
            Path(command[command.index('--output') + 1]).write_bytes(b'private-cubin')
            return subprocess.CompletedProcess(command, 0, stdout='', stderr='')
        output = '\n'.join(f'STT_FUNC STB_GLOBAL STO_ENTRY {symbol}' for symbol in _SYMBOLS)
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr='')

    monkeypatch.setattr(cutile_build.subprocess, 'run', _run)

    artifacts = cutile_build.build_cutile_artifacts([_build_code_object()], tmp_path, 'sm_90')

    assert artifacts.cubin.parent.parent == tmp_path
    for fixed_path, sentinel in sentinels.items():
        assert fixed_path.is_symlink()
        assert sentinel.read_bytes() == f'outside:{fixed_path.name}'.encode('ascii')


def test_build_cutile_artifacts_without_cutile_does_not_resolve_gpu(monkeypatch, tmp_path):
    monkeypatch.setattr(cutile_build, 'resolve_cutile_arch', lambda arch=None: pytest.fail('resolved architecture'))
    assert cutile_build.build_cutile_artifacts([], tmp_path) is None
