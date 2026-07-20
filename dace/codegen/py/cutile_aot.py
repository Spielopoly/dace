# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Ahead-of-time (AOT) compilation of generated cuTile kernels.

At DaCe compile time, every generated ``@ct.kernel`` is exported to a cubin via
``cuda.tile.compilation.export_kernel`` with explicitly constructed conservative
signatures (two per kernel: ``index_dtype`` int32 and int64). At launch, a
:class:`PrecompiledKernel` serves the exported cubin through the private
``ct.kernel._compile`` hook, so no JIT compilation happens.

All failures raise :class:`CuTileAOTError` -- there is no silent JIT fallback.
Set the config ``compiler.cutile.aot_compile=False`` to disable AOT entirely.

The interface to the code generator is the registry ``__dace_cutile_aot_specs``
emitted into the generated frame code: ``{kernel_name: {"params": [(kind,
dtype_name, ndim, stride_constant), ...]}}`` in launch-argument order, where
``kind`` is ``"array"`` or ``"scalar"``.

cuda-tile and cupy are imported lazily inside functions; importing this module
never requires them.
"""
import dataclasses
import inspect
import os
import types
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from dace.config import Config

if TYPE_CHECKING:
    from dace.sdfg import SDFG

#: Name of the module-level registry emitted by the cuTile code generator.
AOT_SPECS_NAME = '__dace_cutile_aot_specs'

_DISABLE_HINT = ('Set the DaCe config entry compiler.cutile.aot_compile=False to disable AOT compilation '
                 '(pure JIT).')

#: The cuda-tile version series whose private ``_compile`` dispatch contract has been verified.
_SUPPORTED_VERSION_PREFIX = '1.5.'

_REQUIRED_COMPILATION_ATTRS = ('export_kernel', 'KernelSignature', 'ArrayConstraint', 'ScalarConstraint',
                               'CallingConvention')

#: Structural key of one kernel parameter: ("array", dtype, ndim, index_dtype) or ("scalar", dtype).
ParamKey = Tuple[Any, ...]
#: Structural key of a full signature: the calling-convention code followed by one ParamKey per parameter.
StructuralKey = Tuple[Any, ...]


class CuTileAOTError(RuntimeError):
    """Raised for any cuTile AOT failure (prerequisites, arch, export, launch-time mismatch)."""
    pass


@dataclasses.dataclass(frozen=True)
class AOTEntry:
    """Precompiled artifact for one kernel: cubin bytes, target arch, and symbol per structural key."""
    cubin: bytes
    arch: str
    symbols: Dict[StructuralKey, str]


def _compilation() -> types.ModuleType:
    """Return the ``cuda.tile.compilation`` module (lazy import).

    :returns: The imported module.
    :raises CuTileAOTError: If cuda-tile is not importable.
    """
    try:
        from cuda.tile import compilation
    except ImportError as exc:
        raise CuTileAOTError(f'cuda-tile is not importable: {exc}. {_DISABLE_HINT}') from exc
    return compilation


def check_prerequisites() -> None:
    """Verify that the installed cuda-tile supports the AOT path this module relies on.

    Requires cuda-tile 1.5.x (the version whose private ``_compile`` contract was verified), the
    ``cuda.tile.compilation`` exports used here, and ``ct.kernel._compile`` with the expected arity.

    :raises CuTileAOTError: If any prerequisite is missing.
    """
    try:
        import cuda.tile as ct
    except ImportError as exc:
        raise CuTileAOTError(f'cuda-tile is not importable: {exc}. {_DISABLE_HINT}') from exc
    compilation = _compilation()

    version = str(getattr(ct, '__version__', '<unknown>'))
    if not version.startswith(_SUPPORTED_VERSION_PREFIX):
        raise CuTileAOTError(f'cuTile AOT compilation requires cuda-tile {_SUPPORTED_VERSION_PREFIX}x '
                             f'(found {version}); the private ct.kernel._compile dispatch contract is only '
                             f'verified for that series. Install cuda-tile {_SUPPORTED_VERSION_PREFIX}x. '
                             f'{_DISABLE_HINT}')

    for attr in _REQUIRED_COMPILATION_ATTRS:
        if not hasattr(compilation, attr):
            raise CuTileAOTError(f'cuda.tile.compilation lacks the required attribute "{attr}". {_DISABLE_HINT}')
    if not hasattr(compilation.CallingConvention, 'cutile_python_v1'):
        raise CuTileAOTError(f'cuda.tile CallingConvention lacks "cutile_python_v1". {_DISABLE_HINT}')
    if not hasattr(compilation.KernelSignature, 'with_mangled_symbol'):
        raise CuTileAOTError(f'cuda.tile KernelSignature lacks "with_mangled_symbol", which is needed to '
                             f'name the exported symbols. {_DISABLE_HINT}')

    # Used at launch to detect a cubin exported for a different GPU.
    try:
        from cuda.tile._compile import get_sm_arch  # noqa: F401
    except ImportError as exc:
        raise CuTileAOTError(f'cuda.tile._compile lacks "get_sm_arch", which is needed to verify the '
                             f'target architecture at launch: {exc}. {_DISABLE_HINT}') from exc

    compile_fn = getattr(ct.kernel, '_compile', None)
    if compile_fn is None:
        raise CuTileAOTError(f'ct.kernel has no "_compile" method to override. {_DISABLE_HINT}')
    try:
        params = list(inspect.signature(compile_fn).parameters)
    except (TypeError, ValueError) as exc:
        raise CuTileAOTError(f'Cannot inspect ct.kernel._compile: {exc}. {_DISABLE_HINT}') from exc
    if len(params) != 3:
        raise CuTileAOTError(f'ct.kernel._compile has unexpected parameters {params} '
                             f'(expected (self, signature, context)). {_DISABLE_HINT}')


def resolve_arch() -> str:
    """Determine the GPU architecture to export cubins for.

    Uses the config ``compiler.cutile.aot_arch`` if set, otherwise auto-detects from the current GPU.

    :returns: Architecture string, e.g. ``"sm_120"``.
    :raises CuTileAOTError: If no override is set and no GPU is available.
    """
    arch = Config.get('compiler', 'cutile', 'aot_arch')
    if arch:
        return arch
    try:
        import cupy
        return f'sm_{cupy.cuda.Device().compute_capability}'
    except Exception as exc:
        raise CuTileAOTError('Cannot determine the GPU architecture for cuTile AOT export (no usable GPU: '
                             f'{exc}). Set compiler.cutile.aot_arch (e.g. "sm_120"). '
                             f'{_DISABLE_HINT}') from exc


def _current_arch() -> str:
    """Return the architecture of the current GPU, as the cuTile JIT would target it.

    :returns: Architecture string, e.g. ``"sm_120"``.
    :raises CuTileAOTError: If the GPU cannot be queried.
    """
    try:
        from cuda.tile._compile import get_sm_arch
        return get_sm_arch()
    except Exception as exc:
        raise CuTileAOTError(f'Cannot determine the current GPU architecture at launch: {exc}. '
                             f'{_DISABLE_HINT}') from exc


def _ct_dtype(name: str) -> Any:
    """Map a numpy dtype name from a spec to the corresponding ``cuda.tile`` dtype.

    :param name: Numpy dtype name, e.g. ``"float64"`` or ``"bool"``.
    :returns: The ``cuda.tile`` DType.
    :raises CuTileAOTError: If no such dtype exists.
    """
    import cuda.tile as ct
    dtype = getattr(ct, 'bool_' if name == 'bool' else name, None)
    if dtype is None:
        raise CuTileAOTError(f'cuda.tile has no dtype named "{name}"; the kernel cannot be AOT-typed. '
                             f'{_DISABLE_HINT}')
    return dtype


def _structural_key(signature: Any) -> StructuralKey:
    """Reduce a ``KernelSignature`` to its ABI-relevant structure.

    Covers the calling convention (argument packing differs between conventions, e.g.
    ``cutile_python_v1`` and ``cutile_python_v2``) and, per parameter: constraint class, dtype, ndim
    and index dtype (arrays). Incidental derived constants (alignment, divisibility, stride constants)
    are deliberately excluded -- they only ever strengthen a signature, never change the ABI.

    :param signature: A ``cuda.tile.compilation.KernelSignature``.
    :returns: Hashable structural key.
    :raises CuTileAOTError: On constraint types this module does not emit.
    """
    compilation = _compilation()
    key: List[ParamKey] = []
    for i, param in enumerate(signature.parameters):
        if isinstance(param, compilation.ArrayConstraint):
            key.append(('array', param.dtype.name, param.ndim, param.index_dtype.name))
        elif isinstance(param, compilation.ScalarConstraint):
            key.append(('scalar', param.dtype.name))
        else:
            raise CuTileAOTError(f'Unsupported constraint type {type(param).__name__} at parameter #{i} of '
                                 f'signature {signature!r}. {_DISABLE_HINT}')
    return (signature.calling_convention.code, ) + tuple(key)


def build_export_plan(spec: Dict[str, Any], func_name: str) -> Tuple[List[Any], Dict[StructuralKey, str]]:
    """Build the dual-index-dtype signatures and the structural-key-to-symbol map for one kernel.

    One conservative ``KernelSignature`` is built per ``index_dtype`` in (int32, int64): shared alias
    group across all arrays (or no aliasing constraint if there are fewer than two arrays), non-negative
    strides, no divisibility or alignment assumptions, ``stride_constant`` only as given by the spec.
    Symbols are mangled from ``func_name`` so the exported cubin contains exactly these symbols. A
    kernel without array parameters yields one signature only -- the index dtype then has nothing to
    apply to, so both variants would be identical.

    :param spec: Registry entry, ``{"params": [(kind, dtype_name, ndim, stride_constant), ...]}``.
    :param func_name: Python function name of the kernel (base of the mangled symbols).
    :returns: ``(signatures, {structural_key: symbol})``.
    :raises CuTileAOTError: If any parameter cannot be typed or signature construction fails.
    """
    compilation = _compilation()
    params = spec['params']
    num_arrays = sum(1 for kind, *_ in params if kind == 'array')
    # A shared alias group with a single member is rejected as redundant by cuda-tile.
    alias_groups: Tuple[str, ...] = ('dace', ) if num_arrays >= 2 else ()

    signatures: List[Any] = []
    symbols: Dict[StructuralKey, str] = {}
    for index_dtype_name in ('int32', 'int64'):
        index_dtype = _ct_dtype(index_dtype_name)
        constraints = []
        for i, (kind, dtype_name, ndim, stride_constant) in enumerate(params):
            if kind == 'array':
                try:
                    constraints.append(
                        compilation.ArrayConstraint(
                            _ct_dtype(dtype_name),
                            ndim,
                            index_dtype=index_dtype,
                            stride_lower_bound_incl=0,
                            alias_groups=alias_groups,
                            may_alias_internally=False,
                            stride_constant=(tuple(stride_constant) if stride_constant is not None else None)))
                except CuTileAOTError:
                    raise
                except Exception as exc:
                    raise CuTileAOTError(f'Cannot build the array constraint for parameter #{i} of kernel '
                                         f'"{func_name}" (dtype={dtype_name}, ndim={ndim}, '
                                         f'stride_constant={stride_constant}): {exc}. {_DISABLE_HINT}') from exc
            elif kind == 'scalar':
                constraints.append(compilation.ScalarConstraint(_ct_dtype(dtype_name)))
            else:
                raise CuTileAOTError(f'Unsupported parameter kind "{kind}" at parameter #{i} of kernel '
                                     f'"{func_name}". {_DISABLE_HINT}')
        try:
            signature = compilation.KernelSignature(constraints, compilation.CallingConvention.cutile_python_v1())
            signature = signature.with_mangled_symbol(func_name)
        except Exception as exc:
            raise CuTileAOTError(f'Cannot build the {index_dtype_name}-index signature for kernel '
                                 f'"{func_name}": {exc}. {_DISABLE_HINT}') from exc
        key = _structural_key(signature)
        if key in symbols:
            # No array parameter: the index dtype is not part of the signature, so both loop
            # iterations produce the same structure and symbol. Export it only once.
            continue
        signatures.append(signature)
        symbols[key] = signature.symbol
    return signatures, symbols


def _export_cubin(dispatcher: Any, signatures: List[Any], path: str, arch: str, kernel_name: str) -> bytes:
    """Export ``dispatcher`` for ``signatures`` to a cubin file and return its bytes.

    :param dispatcher: The ``ct.kernel`` to export.
    :param signatures: Signatures (with mangled symbols) to compile.
    :param path: Output cubin path (overwritten).
    :param arch: Target architecture, e.g. ``"sm_120"``.
    :param kernel_name: Kernel name for error messages.
    :returns: The cubin bytes.
    :raises CuTileAOTError: If the export fails, or if the cubin cannot be read back or is empty.
    """
    compilation = _compilation()
    try:
        compilation.export_kernel(dispatcher, signatures, path, gpu_code=arch, output_format='cubin')
    except Exception as exc:
        raise CuTileAOTError(f'AOT export failed for kernel "{kernel_name}" (arch {arch}): {exc}. '
                             f'{_DISABLE_HINT}') from exc
    try:
        with open(path, 'rb') as f:
            cubin = f.read()
    except OSError as exc:
        raise CuTileAOTError(f'Cannot read back the AOT cubin for kernel "{kernel_name}" from "{path}": '
                             f'{exc}. {_DISABLE_HINT}') from exc
    if not cubin:
        raise CuTileAOTError(f'The AOT cubin exported for kernel "{kernel_name}" at "{path}" is empty '
                             f'(a concurrent export may have truncated it). {_DISABLE_HINT}')
    return cubin


_precompiled_kernel_class: Optional[type] = None


def get_precompiled_kernel_class() -> type:
    """Return the :class:`PrecompiledKernel` class, creating it lazily.

    The class subclasses ``ct.kernel``, so it can only be created once cuda-tile is importable;
    importing this module alone never requires cuda-tile.

    :returns: The ``PrecompiledKernel`` class.
    :raises CuTileAOTError: If cuda-tile is not importable.
    """
    global _precompiled_kernel_class
    if _precompiled_kernel_class is not None:
        return _precompiled_kernel_class

    try:
        import cuda.tile as ct
    except ImportError as exc:
        raise CuTileAOTError(f'cuda-tile is not importable: {exc}. {_DISABLE_HINT}') from exc

    class PrecompiledKernel(ct.kernel):
        """A ``ct.kernel`` that serves an AOT-exported cubin instead of JIT-compiling.

        ``_aot_entry`` (an :class:`AOTEntry`) is attached after construction by
        :func:`precompile_kernels`. ``_compile`` receives the runtime-derived signature from
        ``ct.launch``; a structural-key miss means the DaCe signature builder is buggy and raises.
        """

        _aot_entry: AOTEntry

        def _compile(self, signature, context):
            entry = self._aot_entry
            arch = _current_arch()
            if arch != entry.arch:
                raise CuTileAOTError(f'AOT cubin for kernel "{self._pyfunc.__name__}" was exported for '
                                     f'{entry.arch}, but the current GPU is {arch}. Recompile the SDFG on '
                                     f'this machine or set compiler.cutile.aot_arch accordingly. '
                                     f'{_DISABLE_HINT}')
            key = _structural_key(signature)
            symbol = entry.symbols.get(key)
            if symbol is None:
                raise CuTileAOTError(f'cuTile AOT signature mismatch for kernel "{self._pyfunc.__name__}" '
                                     f'(this indicates a bug in the DaCe AOT signature builder). Derived '
                                     f'signature: {signature!r}; derived structural key: {key}; exported '
                                     f'keys: {list(entry.symbols)}. {_DISABLE_HINT}')
            return entry.cubin, symbol, None, []

    _precompiled_kernel_class = PrecompiledKernel
    return PrecompiledKernel


def _make_precompiled(dispatcher: Any, entry: AOTEntry) -> Any:
    """Create a :class:`PrecompiledKernel` mirroring ``dispatcher`` and attach the AOT entry.

    :param dispatcher: The original ``ct.kernel``.
    :param entry: The precompiled artifact to serve.
    :returns: The replacement ``PrecompiledKernel``.
    """
    cls = get_precompiled_kernel_class()
    kernel = cls(dispatcher._pyfunc, **dataclasses.asdict(dispatcher._compiler_options))
    kernel._aot_entry = entry
    return kernel


def precompile_kernels(namespace: Dict[str, Any], sdfg: 'SDFG') -> None:
    """AOT-compile all kernels registered in ``namespace[AOT_SPECS_NAME]`` and rebind them.

    For each kernel: build dual (int32/int64 index) conservative signatures from its spec, export one
    cubin to ``<build_folder>/cutile_aot/<kernel>.cubin`` (overwriting), and replace the dispatcher in
    ``namespace`` with a :class:`PrecompiledKernel` serving the cubin. All failures raise.

    :param namespace: The executed frame-code namespace of a compiled Python-backend SDFG.
    :param sdfg: The SDFG (for the build folder).
    :raises CuTileAOTError: On any prerequisite, arch, typing or export failure.
    """
    specs = namespace.get(AOT_SPECS_NAME)
    if not specs:
        return
    check_prerequisites()
    import cuda.tile as ct
    arch = resolve_arch()

    out_dir = os.path.join(sdfg.build_folder, 'cutile_aot')
    os.makedirs(out_dir, exist_ok=True)
    for kernel_name, spec in specs.items():
        dispatcher = namespace.get(kernel_name)
        if not isinstance(dispatcher, ct.kernel):
            raise CuTileAOTError(f'AOT spec refers to "{kernel_name}", which is not a ct.kernel in the '
                                 f'generated code (got {type(dispatcher).__name__}). {_DISABLE_HINT}')
        signatures, symbols = build_export_plan(spec, dispatcher._pyfunc.__name__)
        cubin_path = os.path.join(out_dir, f'{kernel_name}.cubin')
        cubin = _export_cubin(dispatcher, signatures, cubin_path, arch, kernel_name)
        namespace[kernel_name] = _make_precompiled(dispatcher, AOTEntry(cubin=cubin, arch=arch, symbols=symbols))


def __getattr__(name: str) -> Any:
    if name == 'PrecompiledKernel':
        return get_precompiled_kernel_class()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
