# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Contains functionality to compile and invoke Python-generated SDFG code."""

from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from dace.codegen.codeobject import CodeObject


class PythonCompiledSDFG:
    """
    A compiled SDFG object for the Python backend.

    Unlike ``CompiledSDFG``, which loads a shared library via ctypes,
    this class executes the generated Python source code directly using
    ``exec()`` and extracts the callable function from the resulting
    namespace.
    """

    def __init__(self, sdfg, code: str):
        from dace.sdfg import SDFG
        self._sdfg: SDFG = sdfg
        self._code: str = code
        self._initialized = False
        self._finalized = False

        # Execute the generated code in an isolated namespace
        self._namespace: Dict[str, Any] = {}
        exec(self._code, self._namespace)

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
        self._initialized = False
        self._finalized = True

    def __call__(self, *args, **kwargs):
        self.initialize(*args, **kwargs)
        return self._func(*args, **kwargs)

    def __del__(self):
        try:
            self.finalize()
        except Exception:
            pass


def compile_python_sdfg(sdfg, code_objects: 'list[CodeObject]') -> PythonCompiledSDFG:
    """
    Compile a Python-backend SDFG from the generated code objects.

    :param sdfg: The SDFG that was compiled.
    :param code_objects: List of CodeObject instances from code generation.
    :return: A callable PythonCompiledSDFG.
    """
    if not code_objects:
        raise RuntimeError("No code objects generated for Python backend")
    code = code_objects[0].code
    return PythonCompiledSDFG(sdfg, code)
