# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Contains functionality to compile and invoke Python-generated SDFG code."""

import builtins
import linecache
import types
from typing import Any, Dict, Optional, TYPE_CHECKING

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

    @property
    def sdfg(self):
        return self._sdfg

    @property
    def code(self) -> str:
        return self._code

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

    def __call__(self, *args, **kwargs):
        self.initialize(*args, **kwargs)
        return self._func(*args, **kwargs)

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
