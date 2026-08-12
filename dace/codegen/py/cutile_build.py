# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Build-time export and embedding helpers for aggregate cuTile kernels."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from dace.codegen.codeobject import CodeObject
from dace.config import Config

CUTILE_BUILD_TARGET_TYPE = 'cutile_build'
CUTILE_BUILD_SOURCE = 'dace_cutile_build.py'
CUTILE_CUBIN = 'dace_cutile_module.cubin'
CUTILE_EMBEDDED_HEADER = 'dace_cutile_embedded.h'
CUTILE_EMBEDDED_SOURCE = 'dace_cutile_embedded.c'
CUTILE_CUBIN_SYMBOL = '__dace_cutile_cubin'
CUTILE_CUBIN_SIZE_SYMBOL = '__dace_cutile_cubin_size'

_ARCH_PATTERN = re.compile(r'sm_[1-9][0-9]{1,2}')
_PYTHON_ENVIRONMENT_KEYS = ('PYTHONHOME', 'PYTHONPATH', 'PYTHONSTARTUP', 'PYTHONINSPECT')


class CuTileBuildError(RuntimeError):
    """Raised when the aggregate cuTile build artifact is invalid."""


@dataclass(frozen=True)
class CuTileBuildInput:
    """Validated aggregate cuTile build input.

    :param code_object: Build-only Python source.
    :param expected_symbols: Exported symbol to map-identity mapping.
    """

    code_object: CodeObject
    expected_symbols: Mapping[str, str]


@dataclass(frozen=True)
class CuTileBuildArtifacts:
    """Paths produced by an aggregate cuTile build.

    :param arch: GPU architecture used for export.
    :param cubin: Exported aggregate cubin.
    :param header: Header declaring the embedded cubin.
    :param source: C source defining the embedded cubin.
    """

    arch: str
    cubin: Path
    header: Path
    source: Path


def _decode_symbol_map(code_object: CodeObject) -> Mapping[str, str]:
    """Decode build-time symbol metadata from a code object.

    :param code_object: Aggregate cuTile build object.
    :returns: Exported symbol to map-identity mapping.
    """
    encoded = code_object.extra_compiler_kwargs.get('cutile_symbols')
    if not encoded:
        raise CuTileBuildError("cuTile build CodeObject is missing extra_compiler_kwargs['cutile_symbols']; "
                               'expected JSON mapping each exported symbol to its map identity')

    def _reject_duplicate_keys(pairs: Sequence[Tuple[str, object]]) -> Mapping[str, object]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate exported symbol {key!r}')
            result[key] = value
        return result

    try:
        decoded = json.loads(encoded, object_pairs_hook=_reject_duplicate_keys)
    except (TypeError, ValueError) as ex:
        raise CuTileBuildError(f'invalid cuTile symbol metadata: {ex}') from ex
    if not isinstance(decoded, dict) or not decoded:
        raise CuTileBuildError('cuTile symbol metadata must be a nonempty JSON object')
    if any(not isinstance(symbol, str) or not symbol for symbol in decoded):
        raise CuTileBuildError('every exported cuTile symbol must be a nonempty string')
    if any(not isinstance(map_identity, str) or not map_identity for map_identity in decoded.values()):
        raise CuTileBuildError('every cuTile map identity must be a nonempty string')
    return decoded


def select_cutile_build_codeobject(code_objects: Sequence[CodeObject]) -> Optional[CuTileBuildInput]:
    """Select and validate the optional aggregate cuTile build object.

    :param code_objects: Generated code objects for one SDFG build.
    :returns: The validated build input, or ``None`` when no cuTile code exists.
    """
    matches = [code_object for code_object in code_objects if code_object.target_type == CUTILE_BUILD_TARGET_TYPE]
    if len(matches) > 1:
        names = ', '.join(repr(code_object.name) for code_object in matches)
        raise CuTileBuildError(f'expected at most one cuTile build CodeObject, found {len(matches)}: {names}')
    if not matches:
        return None

    code_object = matches[0]
    if code_object.linkable:
        raise CuTileBuildError('cuTile build CodeObject must have linkable=False')
    if code_object.language.lower() not in ('py', 'python'):
        raise CuTileBuildError(f"cuTile build CodeObject must use Python source, got {code_object.language!r}")
    return CuTileBuildInput(code_object=code_object, expected_symbols=_decode_symbol_map(code_object))


def validate_cutile_arch(arch: str) -> str:
    """Validate a concrete cuTile GPU architecture.

    :param arch: Architecture in ``sm_XX`` form.
    :returns: The validated architecture unchanged.
    """
    if not isinstance(arch, str) or _ARCH_PATTERN.fullmatch(arch) is None:
        raise ValueError(f"invalid cuTile architecture {arch!r}; expected 'sm_XX' (for example, 'sm_90')")
    return arch


def _query_active_device_arch() -> str:
    """Query the active CuPy device architecture.

    :returns: Architecture in ``sm_XX`` form.
    """
    try:
        import cupy

        capability = cupy.cuda.Device().compute_capability
    except Exception as ex:
        raise CuTileBuildError("compiler.cutile.arch is 'auto', but the active CuPy GPU could not be queried; "
                               "set compiler.cutile.arch explicitly (for example, 'sm_90')") from ex

    if isinstance(capability, (tuple, list)) and len(capability) == 2:
        capability = f'{capability[0]}{capability[1]}'
    elif isinstance(capability, bytes):
        capability = capability.decode('ascii', errors='strict')
    capability_text = str(capability).replace('.', '')
    arch = capability_text if capability_text.startswith('sm_') else f'sm_{capability_text}'
    try:
        return validate_cutile_arch(arch)
    except ValueError as ex:
        raise CuTileBuildError(f'CuPy reported unsupported compute capability {capability!r}; '
                               "set compiler.cutile.arch explicitly (for example, 'sm_90')") from ex


def resolve_cutile_arch(arch: Optional[str] = None) -> str:
    """Resolve an explicit or configured cuTile architecture.

    :param arch: Explicit architecture, or ``None`` to read configuration.
    :returns: A validated concrete architecture.
    """
    configured_arch = Config.get('compiler', 'cutile', 'arch') if arch is None else arch
    if configured_arch == 'auto':
        return _query_active_device_arch()
    return validate_cutile_arch(configured_arch)


def _build_environment() -> Mapping[str, str]:
    """Create the controlled environment for the build-only process.

    :returns: Subprocess environment without Python path injection.
    """
    environment = os.environ.copy()
    for key in _PYTHON_ENVIRONMENT_KEYS:
        environment.pop(key, None)
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['PYTHONNOUSERSITE'] = '1'
    return environment


def _build_context(expected_symbols: Mapping[str, str], arch: str, sdfg_name: str) -> str:
    """Format identifying information for build errors.

    :param expected_symbols: Exported symbol to map-identity mapping.
    :param arch: Target GPU architecture.
    :param sdfg_name: Name of the compiled SDFG.
    :returns: Concise build context.
    """
    maps = ', '.join(f'{map_identity} ({symbol})' for symbol, map_identity in sorted(expected_symbols.items()))
    return f"SDFG {sdfg_name!r}, map(s) {maps}, architecture {arch!r}"


def _run_build_source(source_path: Path, output_path: Path, arch: str, expected_symbols: Mapping[str, str],
                      sdfg_name: str) -> None:
    """Run one aggregate build source in an isolated Python process.

    :param source_path: Build-only Python source path.
    :param output_path: Requested aggregate cubin path.
    :param arch: Target GPU architecture.
    :param expected_symbols: Exported symbol to map-identity mapping.
    :param sdfg_name: Name of the compiled SDFG.
    """
    absolute_source_path = Path(source_path).resolve(strict=True)
    absolute_output_path = Path(output_path).resolve(strict=False)
    command = [
        sys.executable, '-I', '-B',
        str(absolute_source_path), '--output',
        str(absolute_output_path), '--arch', arch
    ]
    try:
        result = subprocess.run(command,
                                cwd=str(absolute_output_path.parent),
                                env=_build_environment(),
                                capture_output=True,
                                text=True,
                                check=False)
    except OSError as ex:
        raise CuTileBuildError(
            f'could not start the cuTile build process for {_build_context(expected_symbols, arch, sdfg_name)}: {ex}'
        ) from ex
    if result.returncode != 0:
        output = '\n'.join(part.strip() for part in (result.stdout, result.stderr) if part and part.strip())
        detail = f'\n{output}' if output else ''
        raise CuTileBuildError(f'cuTile export failed for {_build_context(expected_symbols, arch, sdfg_name)} '
                               f'(source: {absolute_source_path}){detail}')


def _parse_cuobjdump_symbols(output: str) -> Mapping[str, Tuple[Tuple[str, str, str], ...]]:
    """Parse symbol attributes from ``cuobjdump --dump-elf-symbols`` output.

    :param output: Text emitted by ``cuobjdump``.
    :returns: Symbol names mapped to their type, binding, and other attributes.
    """
    rows = {}
    for line in output.splitlines():
        fields = line.split()
        if (len(fields) < 4 or not fields[0].startswith('STT_') or not fields[1].startswith('STB_')
                or not fields[2].startswith(('STO_', 'STV_'))):
            continue
        symbol = fields[-1]
        rows.setdefault(symbol, []).append((fields[0], fields[1], fields[2]))
    return {symbol: tuple(attributes) for symbol, attributes in rows.items()}


def verify_cubin_symbols(cubin_path: Path,
                         expected_symbols: Mapping[str, str],
                         arch: str,
                         *,
                         sdfg_name: str = '<unknown>') -> None:
    """Verify all expected entry symbols with ``cuobjdump``.

    :param cubin_path: Aggregate cubin to inspect.
    :param expected_symbols: Exported symbol to map-identity mapping.
    :param arch: Target GPU architecture.
    :param sdfg_name: Name of the compiled SDFG.
    """
    absolute_cubin_path = Path(cubin_path).resolve(strict=True)
    try:
        result = subprocess.run(
            ['cuobjdump', '--dump-elf-symbols', str(absolute_cubin_path)], capture_output=True, text=True, check=False)
    except OSError as ex:
        raise CuTileBuildError(
            f'could not run cuobjdump for {_build_context(expected_symbols, arch, sdfg_name)}: {ex}') from ex
    if result.returncode != 0:
        output = '\n'.join(part.strip() for part in (result.stdout, result.stderr) if part and part.strip())
        detail = f': {output}' if output else ''
        raise CuTileBuildError(f'cuobjdump could not inspect {absolute_cubin_path} for '
                               f'{_build_context(expected_symbols, arch, sdfg_name)}{detail}')

    required_attributes = ('STT_FUNC', 'STB_GLOBAL', 'STO_ENTRY')
    symbol_rows = _parse_cuobjdump_symbols(result.stdout)
    missing = [(symbol, expected_symbols[symbol], symbol_rows.get(symbol, ())) for symbol in sorted(expected_symbols)
               if required_attributes not in symbol_rows.get(symbol, ())]
    if missing:
        details = []
        for symbol, map_identity, observed_rows in missing:
            observed = ', '.join(' '.join(row) for row in observed_rows) if observed_rows else 'not present'
            details.append(f"map {map_identity!r} symbol {symbol!r} (observed: {observed})")
        raise CuTileBuildError(
            f'missing cuTile cubin entry for {", ".join(details)}, architecture {arch!r}, SDFG {sdfg_name!r}')


def export_cutile_cubin(build_input: CuTileBuildInput,
                        build_dir: Path,
                        arch: str,
                        *,
                        sdfg_name: str = '<unknown>') -> Path:
    """Export and verify one aggregate cubin in a subprocess.

    :param build_input: Validated aggregate source and expected symbols.
    :param build_dir: Parent directory for a unique private artifact directory.
    :param arch: Validated target GPU architecture.
    :param sdfg_name: Name of the compiled SDFG.
    :returns: Path to the verified cubin.
    """
    concrete_arch = validate_cutile_arch(arch)
    build_directory = Path(build_dir).resolve()
    build_directory.mkdir(parents=True, exist_ok=True)
    artifact_directory = Path(tempfile.mkdtemp(prefix='.dace-cutile-artifacts-', dir=build_directory))
    try:
        source_path = artifact_directory / CUTILE_BUILD_SOURCE
        source_path.write_text(build_input.code_object.clean_code, encoding='utf-8')
        requested_cubin = artifact_directory / CUTILE_CUBIN
        _run_build_source(source_path, requested_cubin, concrete_arch, build_input.expected_symbols, sdfg_name)
        cubins = sorted(artifact_directory.rglob('*.cubin'))
        if len(cubins) != 1:
            raise CuTileBuildError(f'cuTile export produced {len(cubins)} cubins instead of exactly one for '
                                   f'{_build_context(build_input.expected_symbols, concrete_arch, sdfg_name)}')
        if cubins[0] != requested_cubin:
            raise CuTileBuildError(
                f'cuTile export ignored the requested output path {requested_cubin} and produced {cubins[0]} for '
                f'{_build_context(build_input.expected_symbols, concrete_arch, sdfg_name)}')
        if requested_cubin.is_symlink() or not requested_cubin.is_file():
            raise CuTileBuildError(f'cuTile export did not produce a regular cubin file for '
                                   f'{_build_context(build_input.expected_symbols, concrete_arch, sdfg_name)}')
        if requested_cubin.stat().st_size == 0:
            raise CuTileBuildError(f'cuTile export produced an empty cubin for '
                                   f'{_build_context(build_input.expected_symbols, concrete_arch, sdfg_name)}')
        verify_cubin_symbols(requested_cubin, build_input.expected_symbols, concrete_arch, sdfg_name=sdfg_name)
        return requested_cubin
    except CuTileBuildError as error:
        # Keep the aggregate source inspectable. The native compiler moves its
        # enclosing private stage into the durable failure directory.
        source_detail = f'source: {source_path}'
        if source_detail in str(error):
            raise
        raise CuTileBuildError(f'{error} ({source_detail})') from error


def _embedding_header() -> str:
    """Render the fixed cubin embedding header.

    :returns: Deterministic C header source.
    """
    return f'''#ifndef DACE_CUTILE_EMBEDDED_H
#define DACE_CUTILE_EMBEDDED_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {{
#endif

extern const unsigned char {CUTILE_CUBIN_SYMBOL}[];
extern const size_t {CUTILE_CUBIN_SIZE_SYMBOL};

#ifdef __cplusplus
}}
#endif

#endif
'''


def _embedding_source(cubin: bytes) -> str:
    """Render a deterministic C byte-array definition.

    :param cubin: Nonempty cubin bytes.
    :returns: C translation unit source.
    """
    rows = []
    for start in range(0, len(cubin), 12):
        values = ', '.join(f'0x{value:02x}' for value in cubin[start:start + 12])
        rows.append(f'    {values},')
    contents = '\n'.join(rows)
    return f'''#include "{CUTILE_EMBEDDED_HEADER}"

const unsigned char {CUTILE_CUBIN_SYMBOL}[] = {{
{contents}
}};
const size_t {CUTILE_CUBIN_SIZE_SYMBOL} = sizeof({CUTILE_CUBIN_SYMBOL});
'''


def _atomic_write_text(path: Path, contents: str) -> None:
    """Atomically replace a text file without following an existing symlink."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w',
                                         encoding='utf-8',
                                         prefix=f'.{path.name}.',
                                         dir=path.parent,
                                         delete=False) as temporary_file:
            temporary_file.write(contents)
            temporary_path = Path(temporary_file.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_cubin_embedding(cubin_path: Path, build_dir: Path) -> Tuple[Path, Path]:
    """Write fixed-name C sources embedding an aggregate cubin.

    :param cubin_path: Nonempty cubin to embed.
    :param build_dir: Compiler build directory.
    :returns: Header path followed by C source path.
    """
    absolute_cubin_path = Path(cubin_path).resolve(strict=True)
    cubin = absolute_cubin_path.read_bytes()
    if not cubin:
        raise CuTileBuildError(f'cannot embed empty cuTile cubin {absolute_cubin_path}')
    build_directory = Path(build_dir).resolve()
    build_directory.mkdir(parents=True, exist_ok=True)
    header_path = build_directory / CUTILE_EMBEDDED_HEADER
    source_path = build_directory / CUTILE_EMBEDDED_SOURCE
    _atomic_write_text(header_path, _embedding_header())
    _atomic_write_text(source_path, _embedding_source(cubin))
    return header_path, source_path


def build_cutile_artifacts(code_objects: Sequence[CodeObject],
                           build_dir: Path,
                           arch: Optional[str] = None,
                           *,
                           sdfg_name: str = '<unknown>') -> Optional[CuTileBuildArtifacts]:
    """Build and embed the optional aggregate cuTile code object.

    :param code_objects: Generated code objects for one SDFG build.
    :param build_dir: Compiler build directory.
    :param arch: Explicit architecture, or ``None`` to use configuration.
    :param sdfg_name: Name of the compiled SDFG.
    :returns: Build artifacts, or ``None`` when no cuTile object exists.
    """
    build_input = select_cutile_build_codeobject(code_objects)
    if build_input is None:
        return None
    concrete_arch = resolve_cutile_arch(arch)
    cubin = export_cutile_cubin(build_input, build_dir, concrete_arch, sdfg_name=sdfg_name)
    try:
        header, source = write_cubin_embedding(cubin, cubin.parent)
    except BaseException:
        shutil.rmtree(cubin.parent, ignore_errors=True)
        raise
    return CuTileBuildArtifacts(arch=concrete_arch, cubin=cubin, header=header, source=source)
