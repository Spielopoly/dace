"""Detailed tests for Python backend code generation of an empty SDFG."""
import dace


def _generate_empty_program():
    """Helper: generate code for an empty program via the Python backend."""
    @dace.program
    def empty_program():
        pass

    sdfg = empty_program.to_sdfg(simplify=False)
    sdfg.backend = dace.dtypes.BackendLanguage.Python
    return sdfg, sdfg.generate_code()


def test_number_of_code_objects():
    """Exactly two CodeObjects should be returned."""
    _, code = _generate_empty_program()
    assert len(code) == 2  # One for the frame, one for the usage example


def test_language_is_py():
    """The CodeObject language must be 'py'."""
    _, code = _generate_empty_program()
    assert code[0].language == 'py'


def test_title_is_frame():
    """The CodeObject title must be 'Frame'."""
    _, code = _generate_empty_program()
    assert code[0].title == 'Frame'


def test_name_matches_sdfg():
    """The CodeObject name must match the SDFG name."""
    sdfg, code = _generate_empty_program()
    assert code[0].name == sdfg.name


def test_generated_code_contains_function_def():
    """Generated code must contain a proper Python function definition."""
    sdfg, code = _generate_empty_program()
    assert f'def {sdfg.name}()' in code[0].code


def test_generated_code_has_pass_body():
    """An empty program should have 'pass' as the function body."""
    _, code = _generate_empty_program()
    assert 'pass' in code[0].code


def test_generated_code_does_not_import_numpy():
    """An empty Python SDFG should not emit a numpy import in the frame header."""
    _, code = _generate_empty_program()
    assert 'import numpy' not in code[0].code


def test_generated_code_is_valid_python():
    """Generated code must be compilable Python."""
    _, code = _generate_empty_program()
    compile(code[0].code, '<generated>', 'exec')


def test_generated_code_is_executable():
    """Generated code must be executable — calling the function should not raise."""
    sdfg, code = _generate_empty_program()
    ns = {}
    exec(code[0].code, ns)
    # Calling the generated function should succeed
    ns[sdfg.name]()


if __name__ == "__main__":
    test_number_of_code_objects()
    test_language_is_py()
    test_title_is_frame()
    test_name_matches_sdfg()
    test_generated_code_contains_function_def()
    test_generated_code_has_pass_body()
    test_generated_code_is_valid_python()
    test_generated_code_is_executable()
    print("All tests passed!")
