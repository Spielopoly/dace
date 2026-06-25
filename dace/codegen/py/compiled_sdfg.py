# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Contains functionality to compile and invoke Python-generated SDFG code."""

import builtins
import linecache
import types
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dace.codegen.codeobject import CodeObject


def _build_aux_module(co: 'CodeObject') -> types.ModuleType:
    """Build a Python module from a CodeObject without registering it in sys.modules."""
    pseudo_filename = f'<dace_generated_module_{co.name}_{id(co)}>'
    code_lines = co.code.splitlines(True)
    linecache.cache[pseudo_filename] = (len(co.code), None, code_lines, pseudo_filename)
    mod = types.ModuleType(co.name)
    mod.__file__ = pseudo_filename
    compiled = compile(co.code, pseudo_filename, 'exec')
    exec(compiled, mod.__dict__)
    return mod


def _make_import_hook(aux_modules: Dict[str, types.ModuleType]):
    """Build a replacement for ``__import__`` that resolves ``aux_modules`` first.

    Top-level absolute imports whose name matches an auxiliary module return
    that module directly. All other imports fall through to the real importer,
    so stdlib and third-party packages behave normally and ``sys.modules`` is
    never touched for our generated modules.
    """
    real_import = builtins.__import__

    def __import__(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in aux_modules:
            return aux_modules[name]
        return real_import(name, globals, locals, fromlist, level)

    return __import__


class PythonCompiledSDFG:
    """
    A compiled SDFG object for the Python backend.

    Unlike ``CompiledSDFG``, which loads a shared library via ctypes,
    this class executes the generated Python source code directly using
    ``exec()`` and extracts the callable function from the resulting
    namespace.
    """

    def __init__(self, sdfg, code: str, *,
                 aux_modules: Optional[Dict[str, types.ModuleType]] = None):
        from dace.sdfg import SDFG
        self._sdfg: SDFG = sdfg
        self._code: str = code
        self._initialized = False
        self._finalized = False
        self._aux_modules: Dict[str, types.ModuleType] = aux_modules or {}

        # Register generated source under a pseudo filename so inspect can
        # retrieve source lines for nested/generated functions.
        # This is necessary for features like CuTile that generate kernels and
        # rely on inspect.getsource() to retrieve their source code for compilation.
        pseudo_filename = (
            f'<dace_generated_python_sdfg_{sdfg.name}_{id(self)}>'
        )
        code_lines = self._code.splitlines(True)
        linecache.cache[pseudo_filename] = (len(self._code), None, code_lines,
                            pseudo_filename)

        # Build a custom builtins so that imports of auxiliary modules resolve
        # locally instead of polluting sys.modules.
        custom_builtins = dict(builtins.__dict__)
        custom_builtins['__import__'] = _make_import_hook(self._aux_modules)

        # Execute the generated code in an isolated namespace
        self._namespace: Dict[str, Any] = {
            '__builtins__': custom_builtins,
            '__file__': pseudo_filename,
        }
        compiled_code = compile(self._code, pseudo_filename, 'exec')
        exec(compiled_code, self._namespace)

        # Extract the generated function (name matches the SDFG name)
        func_name = sdfg.name
        if func_name not in self._namespace:
            raise RuntimeError(
                f"Generated Python code does not define function '{func_name}'"
            )
        self._func = self._namespace[func_name]
        self._init = self._namespace.get(f'__dace_init_{func_name}')
        self._exit = self._namespace.get(f'__dace_exit_{func_name}')

        # --- Return-value metadata ---
        # Single return (__return) vs tuple return (__return_0, __return_1, ...)
        self._is_single_value_ret: bool = False
        if '__return' in self._sdfg.arrays:
            assert not any(
                aname.startswith('__return_')
                for aname in self._sdfg.arrays.keys()
            )
            self._is_single_value_ret = True

        # Whether the SDFG has any __return* arrays (cached for fast __call__)
        self._has_returns: bool = any(
            aname.startswith('__return') for aname in self._sdfg.arrays
        )

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
        """Return the ordered argument names for positional-arg mapping.

        Prefers ``sdfg.arg_names`` (the ``@dace.program`` signature order),
        matching the C++ :class:`~dace.codegen.compiled_sdfg.CompiledSDFG`
        calling convention, so a positional call ``csdfg(a, b, c)`` binds the
        same way on both backends. Falls back to ``sdfg.arglist().keys()``
        (canonical/sorted order) only when ``arg_names`` is empty (e.g. an
        SDFG built directly via the API rather than from a ``@dace.program``).

        Computed lazily because ``arglist()`` can raise on nested SDFGs with
        undeclared runtime symbols during ``__init__``.
        """
        if self._argnames is None:
            if self._sdfg.arg_names:
                self._argnames = list(self._sdfg.arg_names)
            else:
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
        if self._finalized:
            return
        if self._exit is not None:
            self._exit()
        # Clear persistent transients so that the next initialize() cycle
        # starts fresh, matching the semantics of a full re-initialization.
        # TODO: This needs to be part of the generated code, not here
        if '__dace_persistent_transients' in self._namespace:
            self._namespace['__dace_persistent_transients'].clear()
        self._initialized = False
        self._finalized = True

    def _get_return_names(self) -> List[str]:
        """Sorted names of ``__return*`` arrays in the SDFG."""
        if self._return_names is None:
            self._return_names = sorted(
                n for n in self._sdfg.arrays if n.startswith('__return')
            )
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
            raise ValueError(
                f'Used the special array name "{name}" as transient.')
        shape = tuple(int(symbolic.evaluate(s, syms)) for s in desc.shape)
        dtype = desc.dtype.as_numpy_dtype()
        if desc.storage is dtypes.StorageType.GPU_Global:
            try:
                import cupy
                return cupy.empty(shape, dtype=dtype)
            except (ImportError, ModuleNotFoundError):
                raise NotImplementedError(
                    'GPU return values require cupy to be installed')
        return np.empty(shape, dtype=dtype)

    def __call__(self, *args, **kwargs):
        # Convert positional args to keyword args by signature order. This must
        # happen for BOTH the fast (no-returns) and slow paths: the generated
        # ``self._func`` lists its parameters in arglist (canonical) order, not
        # ``@dace.program`` signature order, so forwarding positional args raw
        # would misbind them. Binding by name via ``_get_argnames`` is
        # order-independent and matches the C++ backend's calling convention.
        # When the SDFG declares no argument names (an API-built SDFG / raw
        # code object), fall back to forwarding the positional args unchanged.
        if args:
            argnames = self._get_argnames()
            if argnames:
                positional = dict(zip(argnames, args))
                if not positional.keys().isdisjoint(kwargs.keys()):
                    raise ValueError(
                        "Arguments passed as both positional and keyword: "
                        f"{set(positional) & set(kwargs)}")
                kwargs.update(positional)
                args = ()

        # Fast path: no return values -- forward directly
        if not self._has_returns:
            self.initialize(*args, **kwargs)
            if self.do_not_execute:
                return None
            return self._func(*args, **kwargs)

        # Resolve symbols for shape evaluation
        syms = {k: v for k, v in kwargs.items()
                if k not in self._sdfg.arrays}
        syms.update(self._sdfg.constants)

        # Allocate (or reuse user-provided) return arrays
        return_arrays = []
        for name in self._get_return_names():
            if name not in kwargs:
                kwargs[name] = self._allocate_return_array(name, syms)
            return_arrays.append(kwargs[name])

        self.initialize(**kwargs)
        if not self.do_not_execute:
            self._func(**kwargs)

        # Marshal return values
        if self._is_single_value_ret:
            return return_arrays[0]
        return tuple(return_arrays)

    def __del__(self):
        try:
            self.finalize()
        except AttributeError: # can happen if __init__ raised an exception
            pass


def compile_python_sdfg(sdfg, code_objects: 'list[CodeObject]') -> PythonCompiledSDFG:
    """
    Compile a Python-backend SDFG from the generated code objects.

    The first code object is the frame (main SDFG function). Subsequent
    linkable code objects are built as in-memory Python modules that are
    resolvable by import statements in the frame code via a private
    ``__import__`` hook. ``sys.modules`` is never modified. Non-linkable
    objects (e.g. SampleMain) are skipped.

    :param sdfg: The SDFG that was compiled.
    :param code_objects: List of CodeObject instances from code generation.
    :return: A callable PythonCompiledSDFG.
    """
    if not code_objects:
        raise RuntimeError("No code objects generated for Python backend")

    frame_co = code_objects[0]

    aux_modules: Dict[str, types.ModuleType] = {}
    for co in code_objects[1:]:
        if not co.linkable:
            continue
        aux_modules[co.name] = _build_aux_module(co)

    return PythonCompiledSDFG(sdfg, frame_co.code, aux_modules=aux_modules)
