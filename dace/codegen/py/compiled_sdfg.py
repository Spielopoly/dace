# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Contains functionality to compile and invoke Python-generated SDFG code."""

import linecache
import sys
import types
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from dace.codegen.codeobject import CodeObject


def _register_as_module(co: 'CodeObject') -> None:
    """Exec a code object's code into a fresh module and register it in sys.modules."""
    pseudo_filename = f'<dace_generated_module_{co.name}_{id(co)}>'
    code_lines = co.code.splitlines(True)
    linecache.cache[pseudo_filename] = (len(co.code), None, code_lines, pseudo_filename)
    mod = types.ModuleType(co.name)
    mod.__file__ = pseudo_filename
    compiled = compile(co.code, pseudo_filename, 'exec')
    exec(compiled, mod.__dict__)
    sys.modules[co.name] = mod


class PythonCompiledSDFG:
    """
    A compiled SDFG object for the Python backend.

    Unlike ``CompiledSDFG``, which loads a shared library via ctypes,
    this class executes the generated Python source code directly using
    ``exec()`` and extracts the callable function from the resulting
    namespace.
    """

    def __init__(self, sdfg, code: str, *, injected_module_names: Optional[List[str]] = None):
        from dace.sdfg import SDFG
        self._sdfg: SDFG = sdfg
        self._code: str = code
        self._initialized = False
        self._finalized = False
        self._injected_module_names: List[str] = injected_module_names or []

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

        # Execute the generated code in an isolated namespace
        self._namespace: Dict[str, Any] = {}
        self._namespace['__file__'] = pseudo_filename
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
        for name in self._injected_module_names:
            sys.modules.pop(name, None)
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
    linkable code objects are registered as importable Python modules in
    ``sys.modules`` so that import statements in the frame code resolve
    correctly. Non-linkable objects (e.g. SampleMain) are skipped.

    :param sdfg: The SDFG that was compiled.
    :param code_objects: List of CodeObject instances from code generation.
    :return: A callable PythonCompiledSDFG.
    """
    if not code_objects:
        raise RuntimeError("No code objects generated for Python backend")

    frame_co = code_objects[0]

    injected_module_names = []
    for co in code_objects[1:]:
        if not co.linkable:
            continue
        _register_as_module(co)
        injected_module_names.append(co.name)

    return PythonCompiledSDFG(sdfg, frame_co.code, injected_module_names=injected_module_names)
