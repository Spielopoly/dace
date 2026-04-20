"""Tests for Python backend compile() and execution."""
import numpy as np
import dace
from dace.codegen.py.compiled_sdfg import PythonCompiledSDFG


def test_empty_program_compiles_and_runs():
    """An empty program should compile and run without errors."""
    @dace.program
    def empty_program():
        pass

    sdfg = empty_program.to_sdfg(simplify=False)
    sdfg.backend = dace.dtypes.BackendLanguage.Python
    compiled = sdfg.compile()
    compiled()


def test_compile_returns_python_compiled_sdfg():
    """compile() should return a PythonCompiledSDFG for the Python backend."""
    @dace.program
    def empty_program():
        pass

    sdfg = empty_program.to_sdfg(simplify=False)
    sdfg.backend = dace.dtypes.BackendLanguage.Python
    compiled = sdfg.compile()
    assert isinstance(compiled, PythonCompiledSDFG)


def test_compiled_sdfg_has_sdfg_property():
    """The compiled object should expose the SDFG via a .sdfg property."""
    @dace.program
    def empty_program():
        pass

    sdfg = empty_program.to_sdfg(simplify=False)
    sdfg.backend = dace.dtypes.BackendLanguage.Python
    compiled = sdfg.compile()
    assert compiled.sdfg is not None


def test_compiled_sdfg_has_code_property():
    """The compiled object should expose the generated source via .code."""
    @dace.program
    def empty_program():
        pass

    sdfg = empty_program.to_sdfg(simplify=False)
    sdfg.backend = dace.dtypes.BackendLanguage.Python
    compiled = sdfg.compile()
    assert isinstance(compiled.code, str)
    assert 'def ' in compiled.code


def test_program_with_array_args():
    """A program with array arguments should compile and execute correctly."""
    sdfg = dace.SDFG('test_array_args')
    sdfg.add_array('A', [1], dace.float64)
    sdfg.add_array('B', [1], dace.float64)

    state = sdfg.add_state('compute')
    a_node = state.add_access('A')
    b_node = state.add_access('B')
    tasklet = state.add_tasklet('scale', {'inp'}, {'out'}, 'out = inp * 2.0')
    state.add_edge(a_node, None, tasklet, 'inp', dace.Memlet('A[0]'))
    state.add_edge(tasklet, 'out', b_node, None, dace.Memlet('B[0]'))

    sdfg.backend = dace.dtypes.BackendLanguage.Python
    compiled = sdfg.compile()

    a = np.array([3.0], dtype=np.float64)
    b = np.array([0.0], dtype=np.float64)
    compiled(A=a, B=b)
    assert np.allclose(b, [6.0])


def test_program_with_positional_args():
    """Positional arguments should work too."""
    sdfg = dace.SDFG('test_positional_args')
    sdfg.add_array('A', [1], dace.float64)
    sdfg.add_array('B', [1], dace.float64)

    state = sdfg.add_state('compute')
    a_node = state.add_access('A')
    b_node = state.add_access('B')
    tasklet = state.add_tasklet('scale', {'inp'}, {'out'}, 'out = inp * 2.0')
    state.add_edge(a_node, None, tasklet, 'inp', dace.Memlet('A[0]'))
    state.add_edge(tasklet, 'out', b_node, None, dace.Memlet('B[0]'))

    sdfg.backend = dace.dtypes.BackendLanguage.Python
    compiled = sdfg.compile()

    a = np.array([5.0], dtype=np.float64)
    b = np.array([0.0], dtype=np.float64)
    compiled(a, b)
    assert np.allclose(b, [10.0])


def test_return_program_handle_false():
    """compile(return_program_handle=False) should return None."""
    @dace.program
    def empty_program():
        pass

    sdfg = empty_program.to_sdfg(simplify=False)
    sdfg.backend = dace.dtypes.BackendLanguage.Python
    result = sdfg.compile(return_program_handle=False)
    assert result is None


def test_output_file(tmp_path):
    """compile(output_file=...) should write the generated Python source."""
    @dace.program
    def empty_program():
        pass

    sdfg = empty_program.to_sdfg(simplify=False)
    sdfg.backend = dace.dtypes.BackendLanguage.Python

    # output_file as a directory
    compiled = sdfg.compile(output_file=str(tmp_path))
    out_file = tmp_path / f'{compiled.sdfg.name}.py'
    assert out_file.exists()
    content = out_file.read_text()
    assert 'def ' in content

    # output_file as exact filename
    exact_path = tmp_path / 'my_output.py'
    sdfg2 = empty_program.to_sdfg(simplify=False)
    sdfg2.backend = dace.dtypes.BackendLanguage.Python
    sdfg2.compile(output_file=str(exact_path))
    assert exact_path.exists()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, '-q'])