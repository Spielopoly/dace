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
    """Exactly one CodeObject should be returned."""
    _, code = _generate_empty_program()
    assert len(code) == 1  # One for the frame


def test_language_is_pyx():
    """The host CodeObject language must be 'pyx'."""
    _, code = _generate_empty_program()
    assert code[0].language == 'pyx'


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


def test_generated_code_inlines_symbolic_helpers():
    """The one host source should contain helpers without wildcard imports."""
    _, code = _generate_empty_program()
    assert 'def int_ceil' in code[0].code
    assert 'import *' not in code[0].code
    assert 'sympy_function_redefinitions' not in code[0].code


def test_generated_code_compiles_as_native_extension():
    """The generated host should compile and execute through the public path."""
    sdfg, _ = _generate_empty_program()
    compiled = sdfg.compile()
    compiled()
    assert compiled.library_path.is_file()


if __name__ == "__main__":
    test_number_of_code_objects()
    test_language_is_pyx()
    test_title_is_frame()
    test_name_matches_sdfg()
    test_generated_code_contains_function_def()
    test_generated_code_has_pass_body()
    test_generated_code_inlines_symbolic_helpers()
    test_generated_code_compiles_as_native_extension()
    print("All tests passed!")
