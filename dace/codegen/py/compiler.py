# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Native compiler for Python-backend SDFGs."""

import contextlib
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
from types import ModuleType
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

import dace

if TYPE_CHECKING:
    from dace.codegen.codeobject import CodeObject
    from dace.codegen.py.compiled_sdfg import PythonCompiledSDFG
    from dace.sdfg import SDFG

_MANIFEST_SCHEMA = 1
_DEPLOYMENT_MANIFEST_SCHEMA = 1
_CUTILE_TARGET_TYPE = 'cutile_build'
_PROCESS_OUTPUT_LOCK = threading.RLock()
_CACHE_PUBLICATION_THREAD_LOCK = threading.RLock()
_SETUPTOOLS_ENVIRONMENT_OVERRIDES = ('CC', 'CXX', 'CPP', 'LDSHARED', 'LDCXXSHARED', 'AR', 'ARFLAGS', 'RANLIB', 'CFLAGS',
                                     'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS')


class NativePythonCompileError(RuntimeError):
    """Raised when the generated Cython host cannot be compiled."""


@dataclass(frozen=True)
class PythonExtensionBuild:
    """A completed Python-backend extension build.

    :param module_name: Internal extension module name.
    :param extension_path: Managed path to the compiled extension.
    :param manifest_path: Path to the validated build manifest.
    :param source_path: Managed path to the generated Cython source.
    :param source: Generated Cython source.
    :param cache_hit: Whether the extension came from an existing cache entry.
    """

    module_name: str
    extension_path: Path
    manifest_path: Path
    source_path: Path
    source: str
    cache_hit: bool


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@contextlib.contextmanager
def _capture_process_output(path: Path) -> Iterator[None]:
    """Capture Python output and child-process file descriptors to ``path``."""
    with _PROCESS_OUTPUT_LOCK, path.open('w', encoding='utf-8') as stream:
        saved_stdout = os.dup(1)
        try:
            saved_stderr = os.dup(2)
        except Exception:
            os.close(saved_stdout)
            raise
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                yield
        finally:
            stream.flush()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


@contextlib.contextmanager
def _cache_publication_lock(path: Path) -> Iterator[None]:
    """Lock publication for one content-addressed cache key."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_PUBLICATION_THREAD_LOCK, path.open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b'\0')
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _package_version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _top_level_distribution_versions(package: str) -> Dict[str, Optional[str]]:
    """Return installed distributions that provide a top-level package.

    :param package: Importable top-level package name.
    :returns: Distribution names mapped to installed versions.
    """
    distributions = importlib.metadata.packages_distributions().get(package, ())
    return {name: _package_version(name) for name in sorted(set(distributions))}


def _compiler_identity() -> Dict[str, Any]:
    configured_compiler = sysconfig.get_config_var('CC') or ''
    compiler = os.environ.get('CC', configured_compiler)
    command = shlex.split(compiler)
    version = None
    if command:
        executable = shutil.which(command[0])
        if executable is not None:
            try:
                result = subprocess.run([executable, '--version'],
                                        check=False,
                                        capture_output=True,
                                        text=True,
                                        timeout=10)
                version = (result.stdout or result.stderr).splitlines()[0]
            except (OSError, subprocess.SubprocessError, IndexError):
                version = None
    return {
        'command': compiler,
        'version': version,
        'sysconfig_cc': configured_compiler,
        'sysconfig_cflags': sysconfig.get_config_var('CFLAGS'),
        'sysconfig_ldflags': sysconfig.get_config_var('LDFLAGS'),
        'environment_overrides': {
            name: os.environ.get(name)
            for name in _SETUPTOOLS_ENVIRONMENT_OVERRIDES
        },
    }


def _code_object_record(code_object: 'CodeObject') -> Dict[str, Any]:
    """Return stable cache data for a generated code object."""
    return {
        'name': code_object.name,
        'language': code_object.language,
        'target_type': code_object.target_type,
        'title': code_object.title,
        'linkable': code_object.linkable,
        'extra_compiler_kwargs': dict(sorted(code_object.extra_compiler_kwargs.items())),
        'code_sha256': _sha256_bytes(code_object.code.encode('utf-8')),
    }


def _canonical_mapping_key(key: Any) -> Tuple[str, str]:
    """Return a sortable, collision-free representation of a JSON mapping key.

    :param key: Mapping key emitted by SDFG serialization.
    :returns: Type tag and canonical key value.
    """
    if key is None:
        return ('none', '')
    if type(key) is str:
        return ('str', key)
    if type(key) is bool:
        return ('bool', 'true' if key else 'false')
    if type(key) is int:
        return ('int', str(key))
    if type(key) is float and math.isfinite(key):
        return ('float', key.hex())
    raise NativePythonCompileError(f'Unsupported SDFG JSON mapping key {key!r} of type {type(key).__name__}; '
                                   'expected None, str, bool, int, or a finite float')


def _canonicalize_sdfg_json(value: Any) -> Any:
    """Return type-tagged, deterministic SDFG JSON without GUID fields.

    :param value: Value emitted by SDFG serialization.
    :returns: Canonical JSON-compatible value.
    """
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            if type(key) is str and key == 'guid':
                continue
            canonical_key = _canonical_mapping_key(key)
            items.append((canonical_key, _canonicalize_sdfg_json(item)))
        items.sort(key=lambda item: json.dumps(item[0], separators=(',', ':')))
        return ('mapping', items)
    if isinstance(value, list):
        return ('list', [_canonicalize_sdfg_json(item) for item in value])
    if isinstance(value, tuple):
        return ('tuple', [_canonicalize_sdfg_json(item) for item in value])
    return value


def _stable_sdfg_hash(sdfg: 'SDFG') -> str:
    semantic_json = _canonicalize_sdfg_json(sdfg.to_json())
    serialized = json.dumps(semantic_json, separators=(',', ':'), default=str)
    return _sha256_bytes(serialized.encode('utf-8'))


def _include_dir_record(directory: Path) -> Dict[str, Any]:
    """Return path-independent hashes for native headers in an include directory."""
    headers = []
    for path in sorted(directory.rglob('*')):
        if path.is_file() and path.suffix.lower() in ('.h', '.hh', '.hpp', '.inc'):
            headers.append({
                'name': path.relative_to(directory).as_posix(),
                'sha256': _sha256_file(path),
            })
    return {'headers': headers}


def _validate_code_objects(code_objects: Sequence['CodeObject']) -> Tuple['CodeObject', Optional['CodeObject']]:
    """Validate and split Python-backend code objects.

    :param code_objects: Objects emitted for one top-level SDFG.
    :returns: The host object and optional build-only cuTile object.
    """
    if not code_objects:
        raise RuntimeError('No code objects generated for Python backend')

    cutile_objects = [obj for obj in code_objects if obj.target_type == _CUTILE_TARGET_TYPE]
    host_objects = [obj for obj in code_objects if obj.language == 'pyx' and obj.target_type != _CUTILE_TARGET_TYPE]
    recognized_ids = {id(obj) for obj in cutile_objects + host_objects}
    unexpected = [obj for obj in code_objects if id(obj) not in recognized_ids]

    if len(host_objects) != 1:
        raise RuntimeError(f'Expected exactly one Python host .pyx CodeObject, found {len(host_objects)}')
    if len(cutile_objects) > 1:
        raise RuntimeError(f'Expected at most one cuTile build CodeObject, found {len(cutile_objects)}')
    if unexpected:
        details = ', '.join(f'{obj.name}.{obj.language}' for obj in unexpected)
        raise RuntimeError(f'Unexpected Python-backend CodeObject(s): {details}')

    host = host_objects[0]
    if not host.linkable:
        raise RuntimeError('The Python host .pyx CodeObject must be linkable')
    if cutile_objects:
        cutile = cutile_objects[0]
        if cutile.language != 'py' or cutile.linkable:
            raise RuntimeError('The cuTile build CodeObject must use language py and linkable=False')
        return host, cutile
    return host, None


def _cutile_symbol_context(code_object: Optional['CodeObject']) -> str:
    """Return readable cuTile exported-symbol diagnostics."""
    if code_object is None:
        return 'not applicable'
    encoded = code_object.extra_compiler_kwargs.get('cutile_symbols')
    if not isinstance(encoded, str):
        return f'invalid metadata {encoded!r}'
    try:
        symbols = json.loads(encoded)
    except (TypeError, ValueError):
        return f'invalid metadata {encoded!r}'
    if not isinstance(symbols, dict):
        return f'invalid metadata {encoded!r}'
    if not symbols:
        return 'no exported symbols'
    return ', '.join(f'{symbol} -> {map_identity}' for symbol, map_identity in sorted(symbols.items()))


def _cutile_build_source_files(stage_path: Path) -> Tuple[str, ...]:
    """Return retained aggregate cuTile source paths relative to a build stage.

    :param stage_path: Private or durable native build stage.
    :returns: Sorted relative aggregate build-source paths.
    """
    from dace.codegen.py.cutile_build import CUTILE_BUILD_SOURCE
    return tuple(path.relative_to(stage_path).as_posix() for path in sorted(stage_path.rglob(CUTILE_BUILD_SOURCE)))


def _retain_failed_build(stage_path: Path, *, digest: str, module_name: str, sdfg_name: str, source_name: str,
                         output_name: str, cutile_arch: Optional[str], cutile_symbols: str, error: Exception) -> Path:
    """Move a failed build stage to a durable diagnostic directory."""
    record = {
        'schema': _MANIFEST_SCHEMA,
        'digest': digest,
        'module_name': module_name,
        'sdfg_name': sdfg_name,
        'host_source_file': source_name,
        'compiler_output_file': output_name,
        'cutile_arch': cutile_arch,
        'cutile_symbols': cutile_symbols,
        'cutile_build_source_files': _cutile_build_source_files(stage_path),
        'error_type': type(error).__name__,
        'error': str(error),
    }
    record_path = stage_path / 'failure.json'
    try:
        record_path.write_text(json.dumps(record, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    except OSError:
        pass

    failure_root = stage_path.parent / 'failed'
    try:
        failure_root.mkdir(exist_ok=True)
        failure_path = failure_root / stage_path.name.lstrip('.')
        os.replace(stage_path, failure_path)
        return failure_path
    except OSError:
        return stage_path


def _build_inputs(sdfg: 'SDFG', code_objects: Sequence['CodeObject'], extra_sources: Sequence[Path],
                  include_dirs: Sequence[Path]) -> Dict[str, Any]:
    from Cython import __version__ as cython_version

    return {
        'schema': _MANIFEST_SCHEMA,
        'sdfg_sha256': _stable_sdfg_hash(sdfg),
        'code_objects': [_code_object_record(obj) for obj in code_objects],
        'extra_sources': [{
            'name': source.name,
            'sha256': _sha256_file(source),
        } for source in extra_sources],
        'include_dirs': [_include_dir_record(path) for path in include_dirs],
        'toolchain': {
            'dace': dace.__version__,
            'python': sys.version,
            'soabi': sysconfig.get_config_var('SOABI'),
            'ext_suffix': sysconfig.get_config_var('EXT_SUFFIX'),
            'cython': cython_version,
            'setuptools': _package_version('setuptools'),
            'numpy': np.__version__,
            'cupy': _top_level_distribution_versions('cupy'),
            'cuda_tile': _package_version('cuda-tile'),
            'compiler': _compiler_identity(),
        },
    }


def _input_digest(inputs: Mapping[str, Any]) -> str:
    encoded = json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return _sha256_bytes(encoded)


def _cached_build(cache_dir: Path, expected_inputs: Mapping[str, Any], source: str) -> Optional[PythonExtensionBuild]:
    manifest_path = cache_dir / 'manifest.json'
    expected_module_name = f'_dace_py_{_input_digest(expected_inputs)[:24]}'
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('inputs') != expected_inputs:
            return None
        module_name = manifest['module_name']
        if module_name != expected_module_name:
            return None
        extension_name = manifest['extension_file']
        source_name = manifest['host_source_file']
        if Path(extension_name).name != extension_name or Path(source_name).name != source_name:
            return None
        extension_path = cache_dir / extension_name
        source_path = cache_dir / source_name
        if not extension_path.is_file() or not source_path.is_file():
            return None
        if _sha256_file(extension_path) != manifest['extension_sha256']:
            return None
        if _sha256_file(source_path) != manifest['host_source_sha256']:
            return None
        if source_path.read_text(encoding='utf-8') != source:
            return None
        derived_files = manifest.get('derived_files', [])
        needs_cutile = any(
            obj.get('target_type') == _CUTILE_TARGET_TYPE for obj in expected_inputs.get('code_objects', []))
        if needs_cutile != bool(derived_files):
            return None
        seen_derived = set()
        for record in derived_files:
            relative_path = Path(record['file'])
            if relative_path.is_absolute() or '..' in relative_path.parts or relative_path in seen_derived:
                return None
            seen_derived.add(relative_path)
            derived_path = cache_dir / relative_path
            if not derived_path.is_file() or _sha256_file(derived_path) != record['sha256']:
                return None
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return PythonExtensionBuild(module_name, extension_path, manifest_path, source_path, source, True)


def _publish_cache(stage_path: Path, cache_dir: Path, expected_inputs: Mapping[str, Any],
                   source: str) -> Optional[PythonExtensionBuild]:
    """Publish a completed stage or accept a validated concurrent winner.

    :param stage_path: Complete private build stage.
    :param cache_dir: Content-addressed destination directory.
    :param expected_inputs: Expected manifest build inputs.
    :param source: Expected generated host source.
    :returns: A concurrent winning build, or ``None`` when this stage won.
    """
    lock_path = cache_dir.parent / f'.{cache_dir.name}.lock'
    with _cache_publication_lock(lock_path):
        existing = _cached_build(cache_dir, expected_inputs, source)
        if existing is not None:
            shutil.rmtree(stage_path)
            return existing
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        os.replace(stage_path, cache_dir)
    return None


def _compile_extension(module_name: str, pyx_path: Path, extra_sources: Sequence[Path], include_dirs: Sequence[Path],
                       build_dir: Path) -> Path:
    """Compile one Cython extension with setuptools' programmatic API."""
    from Cython.Build import cythonize
    from setuptools import Distribution, Extension

    cython_dir = build_dir / 'cython'
    library_dir = build_dir / 'lib'
    temporary_dir = build_dir / 'temp'
    cython_dir.mkdir(parents=True)
    library_dir.mkdir(parents=True)
    temporary_dir.mkdir(parents=True)

    extension = Extension(module_name,
                          sources=[str(pyx_path), *(str(path) for path in extra_sources)],
                          include_dirs=[str(path) for path in include_dirs])
    extensions = cythonize([extension],
                           build_dir=str(cython_dir),
                           compiler_directives={'language_level': 3},
                           quiet=True)
    distribution = Distribution({'name': module_name, 'ext_modules': extensions})
    command = distribution.get_command_obj('build_ext')
    command.build_lib = str(library_dir)
    command.build_temp = str(temporary_dir)
    command.force = True
    command.inplace = False
    command.ensure_finalized()
    command.run()

    extension_path = Path(command.get_ext_fullpath(module_name))
    if not extension_path.is_absolute():
        extension_path = Path.cwd() / extension_path
    if not extension_path.is_file():
        candidates = list(library_dir.rglob(f'{module_name}*{sysconfig.get_config_var("EXT_SUFFIX") or ".so"}'))
        if len(candidates) != 1:
            raise NativePythonCompileError(
                f'Cython reported success but no extension was produced for module {module_name!r}')
        extension_path = candidates[0]
    return extension_path


def build_python_extension(
    sdfg: 'SDFG',
    code_objects: Sequence['CodeObject'],
    *,
    extra_sources: Sequence[os.PathLike] = (),
    include_dirs: Sequence[os.PathLike] = ()) -> PythonExtensionBuild:
    """Build or reuse the native extension for a Python-backend SDFG.

    ``extra_sources`` and ``include_dirs`` are the extension point for the
    optional embedded cuTile cubin translation unit.

    This backend always reuses validated content-addressed cache entries,
    independently of ``compiler.use_cache``. That option controls the legacy
    path-based compiler cache, which can skip code generation entirely.

    :param sdfg: Deep-copied SDFG used for code generation.
    :param code_objects: Generated host and optional build-only objects.
    :param extra_sources: Additional native sources linked into the extension.
    :param include_dirs: Additional native include directories.
    :returns: A completed, validated build.
    """
    host, cutile_build = _validate_code_objects(code_objects)

    cutile_arch = None
    if cutile_build is not None:
        from dace.codegen.py.cutile_build import resolve_cutile_arch
        cutile_arch = resolve_cutile_arch()
        cutile_build.extra_compiler_kwargs['cutile_arch'] = cutile_arch

    native_sources = tuple(Path(source).resolve() for source in extra_sources)
    native_include_dirs = tuple(Path(directory).resolve() for directory in include_dirs)
    missing_sources = [str(path) for path in native_sources if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f'Native extension source file(s) not found: {missing_sources}')
    missing_include_dirs = [str(path) for path in native_include_dirs if not path.is_dir()]
    if missing_include_dirs:
        raise FileNotFoundError(f'Native extension include directory/directories not found: {missing_include_dirs}')

    inputs = _build_inputs(sdfg, code_objects, native_sources, native_include_dirs)
    digest = _input_digest(inputs)
    module_name = f'_dace_py_{digest[:24]}'
    cache_root = Path(sdfg.build_folder).resolve() / 'python'
    cache_dir = cache_root / digest
    cached = _cached_build(cache_dir, inputs, host.code)
    if cached is not None:
        return cached

    cache_root.mkdir(parents=True, exist_ok=True)
    stage_path = Path(tempfile.mkdtemp(prefix=f'.{digest}.', dir=cache_root))
    pyx_path = stage_path / f'{module_name}.pyx'
    pyx_path.write_text(host.code, encoding='utf-8')
    compiler_output_path = stage_path / 'compiler-output.txt'
    cutile_symbols = _cutile_symbol_context(cutile_build)
    derived_files = []
    try:
        with _capture_process_output(compiler_output_path):
            compile_sources = native_sources
            compile_include_dirs = native_include_dirs
            if cutile_build is not None:
                from dace.codegen.py.cutile_build import build_cutile_artifacts
                cutile_artifacts = build_cutile_artifacts(code_objects,
                                                          stage_path / 'cutile',
                                                          cutile_arch,
                                                          sdfg_name=sdfg.name)
                if cutile_artifacts is None:
                    raise NativePythonCompileError('cuTile code generation did not produce build artifacts')
                compile_sources = (*compile_sources, cutile_artifacts.source)
                compile_include_dirs = (*compile_include_dirs, cutile_artifacts.header.parent)
                for path in sorted(cutile_artifacts.source.parent.iterdir()):
                    if path.is_file():
                        derived_files.append({
                            'file': path.relative_to(stage_path).as_posix(),
                            'sha256': _sha256_file(path),
                        })
            compiled_path = _compile_extension(module_name, pyx_path, compile_sources, compile_include_dirs,
                                               stage_path / 'build')
        extension_name = f'{module_name}{sysconfig.get_config_var("EXT_SUFFIX") or compiled_path.suffix}'
        staged_extension = stage_path / extension_name
        shutil.copy2(compiled_path, staged_extension)
        manifest = {
            'schema': _MANIFEST_SCHEMA,
            'inputs': inputs,
            'module_name': module_name,
            'extension_file': extension_name,
            'extension_sha256': _sha256_file(staged_extension),
            'host_source_file': pyx_path.name,
            'host_source_sha256': _sha256_file(pyx_path),
            'derived_files': derived_files,
        }
        manifest_path = stage_path / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n', encoding='utf-8')
        compiler_output_path.unlink(missing_ok=True)

        existing = _publish_cache(stage_path, cache_dir, inputs, host.code)
        if existing is not None:
            return existing
    except Exception as error:
        try:
            output = compiler_output_path.read_text(encoding='utf-8', errors='replace').strip()
        except OSError:
            output = ''
        if not output:
            output = f'{type(error).__name__}: {error}'
            try:
                compiler_output_path.write_text(output + '\n', encoding='utf-8')
            except OSError:
                pass
        retained_path = _retain_failed_build(stage_path,
                                             digest=digest,
                                             module_name=module_name,
                                             sdfg_name=sdfg.name,
                                             source_name=pyx_path.name,
                                             output_name=compiler_output_path.name,
                                             cutile_arch=cutile_arch,
                                             cutile_symbols=cutile_symbols,
                                             error=error)
        diagnostic = [
            f'Failed to build native Python extension for SDFG {sdfg.name!r}.',
            f'Source: {retained_path / pyx_path.name}',
            f'Failure record: {retained_path / "failure.json"}',
        ]
        diagnostic.extend(f'cuTile build source: {retained_path / relative_path}'
                          for relative_path in _cutile_build_source_files(retained_path))
        diagnostic.extend([
            f'cuTile architecture: {cutile_arch or "not applicable"}',
            f'cuTile symbols: {cutile_symbols}',
            f'Original error: {type(error).__name__}: {error}',
            f'Compiler output:\n{output}',
        ])
        raise NativePythonCompileError('\n'.join(diagnostic)) from error

    result = _cached_build(cache_dir, inputs, host.code)
    if result is None:
        raise NativePythonCompileError(f'Native Python build cache is incomplete: {cache_dir}')
    return PythonExtensionBuild(result.module_name, result.extension_path, result.manifest_path, result.source_path,
                                result.source, False)


def deployed_extension_manifest_path(extension_path: os.PathLike) -> Path:
    """Return the sidecar manifest path for a deployed extension.

    :param extension_path: Deployed native extension path.
    :returns: Adjacent deployment manifest path.
    """
    path = Path(extension_path)
    return path.with_name(f'{path.name}.dace.json')


def load_deployed_extension(extension_path: os.PathLike) -> ModuleType:
    """Load a deployed extension using its recorded internal module name.

    :param extension_path: Deployed native extension path.
    :returns: Loaded native extension module.
    """
    path = Path(extension_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Deployed Python extension does not exist: {path}')

    manifest_path = deployed_extension_manifest_path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except FileNotFoundError as error:
        raise NativePythonCompileError(f'Deployment manifest does not exist for {path}: {manifest_path}') from error
    except (OSError, json.JSONDecodeError) as error:
        raise NativePythonCompileError(f'Cannot read deployment manifest {manifest_path}: {error}') from error

    if not isinstance(manifest, dict) or manifest.get('schema') != _DEPLOYMENT_MANIFEST_SCHEMA:
        raise NativePythonCompileError(f'Unsupported deployment manifest schema in {manifest_path}')
    module_name = manifest.get('module_name')
    if not isinstance(module_name, str) or not module_name.isidentifier():
        raise NativePythonCompileError(f'Invalid internal module name in {manifest_path}')
    extension_file = manifest.get('extension_file')
    if not isinstance(extension_file, str) or Path(extension_file).name != extension_file:
        raise NativePythonCompileError(f'Unsafe extension filename in {manifest_path}')
    if extension_file != path.name:
        raise NativePythonCompileError(
            f'Deployment manifest {manifest_path} describes {extension_file!r}, not {path.name!r}')
    expected_hash = manifest.get('extension_sha256')
    if (not isinstance(expected_hash, str) or len(expected_hash) != 64
            or any(character not in '0123456789abcdef' for character in expected_hash)):
        raise NativePythonCompileError(f'Invalid extension hash in {manifest_path}')
    actual_hash = _sha256_file(path)
    if actual_hash != expected_hash:
        raise NativePythonCompileError(
            f'Deployed extension hash mismatch for {path}; the file or manifest was modified')

    from dace.codegen.py.compiled_sdfg import _load_native_module
    return _load_native_module(path, module_name)


def copy_compiled_extension(build: PythonExtensionBuild, output_file: os.PathLike, sdfg_name: str) -> Path:
    """Copy a compiled extension and its self-describing deployment manifest."""
    destination = Path(output_file)
    if destination.is_dir():
        destination = destination / f'{sdfg_name}{sysconfig.get_config_var("EXT_SUFFIX") or build.extension_path.suffix}'
    shutil.copy2(build.extension_path, destination)
    manifest = {
        'schema': _DEPLOYMENT_MANIFEST_SCHEMA,
        'module_name': build.module_name,
        'sdfg_name': sdfg_name,
        'extension_file': destination.name,
        'extension_sha256': _sha256_file(destination),
    }
    manifest_path = deployed_extension_manifest_path(destination)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    return destination


def compile_python_sdfg(
    sdfg: 'SDFG',
    code_objects: Sequence['CodeObject'],
    *,
    output_file: Optional[os.PathLike] = None,
    load: bool = True,
    extra_sources: Sequence[os.PathLike] = (),
    include_dirs: Sequence[os.PathLike] = ()
) -> Optional['PythonCompiledSDFG']:
    """Compile and optionally load a Python-backend SDFG as a native extension."""
    build = build_python_extension(sdfg, code_objects, extra_sources=extra_sources, include_dirs=include_dirs)
    if output_file is not None:
        copy_compiled_extension(build, output_file, sdfg.name)
    if not load:
        return None

    from dace.codegen.py.compiled_sdfg import PythonCompiledSDFG
    return PythonCompiledSDFG(sdfg, build.extension_path, build.module_name, code=build.source)
