# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Contains functionality to load and invoke native Python-backend SDFGs."""

import atexit
import importlib.util
from pathlib import Path
import re
import shutil
import tempfile
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dace.codegen.codeobject import CodeObject

#: A return-value array is named exactly ``__return`` (single value) or
#: ``__return_<int>`` (tuple element).  Transients that merely share the
#: ``__return`` prefix -- e.g. the ``__return_tile`` / ``__return_tile_out``
#: buffers the tile vectorizer creates when the program's return value is
#: written through a cuTile kernel -- are NOT return values and must be
#: excluded from return marshaling.
_RETURN_ARRAY_RE = re.compile(r'^__return(_[0-9]+)?$')

# CPython extension state is tied to the loaded shared-library image. Loading
# the managed cache file twice may therefore alias module globals even when two
# distinct module objects are requested. Keep one private image per compiled
# handle for the process lifetime so persistent transients and instrumentation
# state cannot leak between handles. Process-lifetime retention also avoids
# deleting a loaded extension on platforms that lock shared libraries.
_PRIVATE_EXTENSION_DIRS: List[Path] = []


def _cleanup_private_extensions() -> None:
    for directory in _PRIVATE_EXTENSION_DIRS:
        shutil.rmtree(directory, ignore_errors=True)


atexit.register(_cleanup_private_extensions)


def _is_return_array_name(name: str) -> bool:
    """Whether ``name`` is a genuine SDFG return-value array name.

    :param name: A data-descriptor name from ``sdfg.arrays``.
    :returns: ``True`` for ``__return`` and ``__return_<int>`` only.
    """
    return _RETURN_ARRAY_RE.match(name) is not None


def _load_native_module(extension_path: Path, module_name: str) -> ModuleType:
    """Load a Cython extension using its hashed internal module name."""
    if not extension_path.is_file():
        raise FileNotFoundError(f'Compiled Python extension does not exist: {extension_path}')
    spec = importlib.util.spec_from_file_location(module_name, extension_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot create an import specification for {extension_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_isolated_native_module(extension_path: Path, module_name: str) -> ModuleType:
    """Load a private extension image for one compiled-program handle.

    :param extension_path: Validated extension in the managed build cache.
    :param module_name: Internal CPython initialization name.
    :returns: A module whose native globals are private to the caller.
    """
    private_dir = Path(tempfile.mkdtemp(prefix=f'dace-{module_name}-'))
    private_path = private_dir / extension_path.name
    try:
        shutil.copy2(extension_path, private_path)
        module = _load_native_module(private_path, module_name)
    except Exception:
        shutil.rmtree(private_dir, ignore_errors=True)
        raise
    _PRIVATE_EXTENSION_DIRS.append(private_dir)
    return module


class PythonCompiledSDFG:
    """A callable adapter around a compiled Python-backend extension."""

    def __init__(self, sdfg, extension_path, module_name: str, *, code: str = ''):
        from dace.sdfg import SDFG
        self._sdfg: SDFG = sdfg
        self._library_path = Path(extension_path).resolve()
        self._code: str = code
        self._initialized = False
        self._finalized = False
        self._module_name = module_name
        self._module = _load_isolated_native_module(self._library_path, module_name)
        # This module-dictionary view is retained for instrumentation consumers.
        self._namespace = self._module.__dict__

        func_name = sdfg.name
        if not hasattr(self._module, func_name):
            raise RuntimeError(f"Compiled Python extension does not define function '{func_name}'")
        self._func = getattr(self._module, func_name)
        self._init = getattr(self._module, f'__dace_init_{func_name}', None)
        self._exit = getattr(self._module, f'__dace_exit_{func_name}', None)

        # --- Return-value metadata ---
        # Single return (__return) vs tuple return (__return_0, __return_1, ...)
        self._is_single_value_ret: bool = False
        if '__return' in self._sdfg.arrays:
            assert not any(_is_return_array_name(aname) and aname != '__return' for aname in self._sdfg.arrays.keys())
            self._is_single_value_ret = True

        # Whether the SDFG has any genuine return arrays (cached for fast
        # __call__); tile transients sharing the __return prefix don't count.
        self._has_returns: bool = any(_is_return_array_name(aname) for aname in self._sdfg.arrays)

        # Argument name list for positional arg conversion (includes __return*).
        # Computed lazily by _get_argnames() because arglist() can fail on
        # nested SDFGs with undeclared runtime symbols.
        self._argnames: Optional[List[str]] = None

        # Sorted __return* array names (lazy cache).
        self._return_names: Optional[List[str]] = None

        # Create cached cfunc wrapper for profiler compatibility
        func = self._func

        def _cfunc_wrapper(_handle, *args, **kwargs):
            return func(*args, **kwargs)

        self._cfunc_cached = _cfunc_wrapper

        # Profiler compatibility: these attributes mirror the C++ CompiledSDFG
        # interface so that CompiledSDFGProfiler (and similar tools like
        # npbench) can use the same code path for both backends.
        self.do_not_execute: bool = False
        self._libhandle = None

    @property
    def sdfg(self):
        return self._sdfg

    @property
    def code(self) -> str:
        return self._code

    @property
    def library_path(self) -> Path:
        """Managed path to the loaded native extension."""
        return self._library_path

    @property
    def module(self) -> ModuleType:
        """Loaded native extension module."""
        return self._module

    @property
    def _cfunc(self):
        """Return a callable compatible with the C++ CompiledSDFG interface.

        The C++ backend exposes ``_cfunc`` as a ctypes function pointer whose
        first argument is ``_libhandle`` (a ``ctypes.c_void_p``).  Profiling
        tools such as ``CompiledSDFGProfiler`` call
        ``compiled_sdfg._cfunc(compiled_sdfg._libhandle, *args)`` to bypass
        the normal ``__call__`` path.

        This property returns a cached thin wrapper around ``self._func``
        that accepts (and ignores) the leading handle argument so the same
        calling convention works for the Python backend.
        """
        return self._cfunc_cached

    def _get_argnames(self) -> List[str]:
        """Return cached arglist keys, computing them lazily.

        Returns ``sdfg.arglist().keys()`` which is equivalent to
        ``CompiledSDFG.argnames`` for positional-arg mapping.
        Computed lazily because ``arglist()`` can raise on nested SDFGs
        with undeclared runtime symbols during ``__init__``.
        """
        if self._argnames is None:
            self._argnames = list(self._sdfg.arglist().keys())
        return self._argnames

    def initialize(self, *args, **kwargs):
        if self._initialized:
            return
        if self._init is not None:
            self._init(*args, **kwargs)
        self._initialized = True
        self._finalized = False

    def finalize(self):
        if self._finalized or not self._initialized:
            return
        try:
            if self._exit is not None:
                self._exit()
        finally:
            # Clear persistent transients even when user exit code raises.
            # TODO: Should be part of generated code, not here
            persistent = getattr(self._module, '__dace_persistent_transients', None)
            if persistent is not None:
                persistent.clear()
            self._initialized = False
            self._finalized = True

    def _ordered_call_arguments(self, args: tuple, kwargs: Dict[str, Any]) -> tuple:
        """Return call arguments in the generated function's positional order.

        Compiled-SDFG hooks receive a positional tuple and may invoke
        :attr:`_cfunc` repeatedly. Normal generated calls have already been
        bound to ``sdfg.arglist()`` order. The signature fallback preserves the
        existing focused-fixture and undeclared-symbol paths where that arglist
        cannot describe the native function.

        :param args: Positional arguments for the native function.
        :param kwargs: Marshalled keyword arguments for the native function.
        :returns: Arguments suitable for the compiled-call hook interface.
        """
        try:
            argnames = self._get_argnames()
        except Exception:
            argnames = []

        if not args and set(kwargs) == set(argnames):
            return tuple(kwargs[name] for name in argnames)

        try:
            import inspect
            bound = inspect.signature(self._func).bind(*args, **kwargs)
            bound.apply_defaults()
            if bound.kwargs:
                raise TypeError('Compiled-SDFG call hooks do not support keyword-only native arguments')
            return bound.args
        except (TypeError, ValueError):
            # Let the native function report the original binding error when
            # no hook consumes the arguments. Generated SDFG functions never
            # reach this fallback because their arglist is exact.
            return args

    def _invoke_with_hooks(self, args: tuple, kwargs: Dict[str, Any]) -> Any:
        """Invoke the native function through compiled-SDFG call hooks.

        :param args: Positional arguments for the native function.
        :param kwargs: Marshalled keyword arguments for the native function.
        :returns: The selected native function's return value.
        """
        from dace import hooks

        hook_args = self._ordered_call_arguments(args, kwargs)
        with hooks.invoke_compiled_sdfg_call_hooks(self, hook_args) as compiled_sdfg:
            if compiled_sdfg.do_not_execute:
                return None
            if compiled_sdfg is not self:
                return compiled_sdfg._cfunc(compiled_sdfg._libhandle, *hook_args)
            return self._func(*args, **kwargs)

    def _bind_positional(self, args: tuple, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert positional arguments to keyword arguments.

        :param args: Positional argument values, mapped onto ``arglist()`` order.
        :param kwargs: Keyword arguments (not modified).
        :returns: A new kwargs dict including the positional bindings.
        :raises KeyError: If the SDFG takes no arguments but ``args`` is nonempty.
        :raises TypeError: If more positional arguments are passed than the
            SDFG accepts.
        :raises ValueError: If an argument is passed both ways.
        """
        if not args:
            return dict(kwargs)
        argnames = self._get_argnames()
        if not argnames:
            raise KeyError("Passed positional arguments to an SDFG that does "
                           "not accept them.")
        if len(args) > len(argnames):
            raise TypeError(f"Passed {len(args)} positional arguments to an SDFG that "
                            f"accepts at most {len(argnames)} ({argnames}).")
        positional = dict(zip(argnames, args))
        if not positional.keys().isdisjoint(kwargs.keys()):
            raise ValueError("Arguments passed as both positional and keyword: "
                             f"{set(positional) & set(kwargs)}")
        merged = dict(kwargs)
        merged.update(positional)
        return merged

    def _marshal_arguments(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce runtime arguments to the Python-backend calling convention.

        Scalar data arguments are bound as 0-d numpy buffers of the declared
        dtype (generated code reads them as ``x[()]`` and writes ``x[...]``),
        and numpy scalars passed for symbols are converted to native Python
        values (kernel launch grids and ``range()`` bounds require them).

        :param kwargs: Keyword arguments as passed by the caller.
        :returns: A new dict with marshalled values.
        :raises TypeError: If a non-scalar array is passed for a Scalar
            argument, if an ndarray argument's dtype does not match the
            declared dtype, or if a scalar value cannot be losslessly
            converted to the declared dtype.
        """
        from dace import data

        marshalled = dict(kwargs)
        for name, value in kwargs.items():
            desc = self._sdfg.arrays.get(name)
            if isinstance(desc, data.Scalar):
                nptype = desc.dtype.as_numpy_dtype()
                if isinstance(value, np.ndarray):
                    if value.size != 1:
                        raise TypeError(f'Argument {name!r}: expected a scalar, '
                                        f'got an array of shape {value.shape}')
                    # A dtype-mismatched buffer would have to be copied
                    # (np.asarray), so in-SDFG writes would land in the copy
                    # and the caller's buffer would keep the stale value.
                    # Mirror the C backend's array behavior and reject it.
                    if value.dtype != nptype:
                        raise TypeError(f'Argument {name!r}: passed an ndarray of dtype '
                                        f'{value.dtype}, expected {np.dtype(nptype)}')
                    # Same-dtype reshape is a view: in-SDFG writes propagate.
                    marshalled[name] = value.reshape(())
                elif isinstance(value, (bool, int, float, complex, np.generic)):
                    coerced = np.asarray(value, dtype=nptype)
                    # Coercing a plain value to an integer/bool dtype must be
                    # exact (e.g. 3.9 -> int32 would silently truncate to 3,
                    # and out-of-range numpy ints silently wrap).
                    if coerced.dtype.kind in 'iub' and coerced[()] != value:
                        raise TypeError(f'Argument {name!r}: value {value!r} cannot be '
                                        f'losslessly converted to declared dtype '
                                        f'{np.dtype(nptype)}')
                    marshalled[name] = coerced
                # Anything else (e.g. a cupy scalar buffer) passes through.
            elif desc is None:
                # Symbol values: native Python scalars only.
                if isinstance(value, np.generic):
                    marshalled[name] = value.item()
                elif isinstance(value, np.ndarray) and value.ndim == 0:
                    marshalled[name] = value.item()
        return marshalled

    def _get_return_names(self) -> List[str]:
        """Sorted names of ``__return*`` arrays in the SDFG."""
        if self._return_names is None:
            self._return_names = sorted(n for n in self._sdfg.arrays if _is_return_array_name(n))
        return self._return_names

    def _allocate_return_array(self, name: str, syms: Dict[str, Any]) -> np.ndarray:
        """Allocate a return-value array, evaluating symbolic shapes.

        :param name: The ``__return*`` array name in the SDFG.
        :param syms: Symbol-to-value mapping for shape evaluation.
        :returns: A newly allocated numpy (or cupy) array.
        """
        from dace import dtypes, symbolic

        desc = self._sdfg.arrays[name]
        if desc.transient:
            raise ValueError(f'Used the special array name "{name}" as transient.')
        shape = tuple(int(symbolic.evaluate(s, syms)) for s in desc.shape)
        dtype = desc.dtype.as_numpy_dtype()
        if desc.storage is dtypes.StorageType.GPU_Global:
            try:
                import cupy
                return cupy.empty(shape, dtype=dtype)
            except (ImportError, ModuleNotFoundError):
                raise NotImplementedError('GPU return values require cupy to be installed')
        return np.empty(shape, dtype=dtype)

    def __call__(self, *args, **kwargs):
        # Fast path: no return values
        if not self._has_returns:
            if args:
                try:
                    self._get_argnames()
                except Exception:
                    # arglist() can fail on nested SDFGs with undeclared
                    # runtime symbols; fall back to raw positional args
                    # (keyword arguments are still marshalled below).
                    pass
                else:
                    # Only the argname resolution above is fallible; binding
                    # errors (duplicate/excess arguments) must propagate.
                    kwargs = self._bind_positional(args, kwargs)
                    args = ()
            kwargs = self._marshal_arguments(kwargs)
            self.initialize(*args, **kwargs)
            return self._invoke_with_hooks(args, kwargs)

        # Convert positional args to keyword args
        kwargs = self._bind_positional(args, kwargs)
        kwargs = self._marshal_arguments(kwargs)

        # Resolve symbols for shape evaluation
        syms = {k: v for k, v in kwargs.items() if k not in self._sdfg.arrays}
        syms.update(self._sdfg.constants)

        # Allocate (or reuse user-provided) return arrays
        return_arrays = []
        for name in self._get_return_names():
            if name not in kwargs:
                kwargs[name] = self._allocate_return_array(name, syms)
            return_arrays.append(kwargs[name])

        self.initialize(**kwargs)
        self._invoke_with_hooks((), kwargs)

        # Marshal return values
        if self._is_single_value_ret:
            return return_arrays[0]
        return tuple(return_arrays)

    def __del__(self):
        try:
            self.finalize()
        except Exception:
            # Destructors must not retry or report user cleanup failures.
            pass


def compile_python_sdfg(sdfg, code_objects: 'Sequence[CodeObject]', **kwargs) -> Optional[PythonCompiledSDFG]:
    """Compatibility entry point for native Python-backend compilation."""
    from dace.codegen.py.compiler import compile_python_sdfg as compile_native_python_sdfg
    return compile_native_python_sdfg(sdfg, code_objects, **kwargs)
