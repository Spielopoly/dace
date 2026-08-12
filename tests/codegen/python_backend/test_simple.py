import dace
from dace.config import set_temporary
import numpy as np
import pytest

N = dace.symbol('N')
Modulo = dace.symbol('Modulo')

FIVE = 5.0


def foo(x: dace.float64[3, 7]):
    s = np.sum(x)
    return FIVE + s


@dace.program
def simple_program(x: dace.float64[N, 7], y: dace.float64[Modulo * N, 7]):
    for i in range(N):
        for j in range(7):
            if i % Modulo == 0 and j % 2 == 0:
                y[Modulo * i, j] = x[i, j] * foo(x)
    # print(y)
    z = np.zeros_like(x)
    for i in range(N):
        for j in range(0, 7, 3):
            z[i, j] = x[i, j] + y[i * Modulo, j]
    return z


def _reference_simple_program(x: np.ndarray, y: np.ndarray, modulo: int) -> np.ndarray:
    """Mirror the source program exactly for backend regression checks."""
    for row_index in range(x.shape[0]):
        for column_index in range(x.shape[1]):
            if row_index % modulo == 0 and column_index % 2 == 0:
                y[modulo * row_index, column_index] = x[row_index, column_index] * foo(x)
    # print(y)
    z = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(0, x.shape[1], 3):
            z[i, j] = x[i, j] + y[i * modulo, j]
    return z


@pytest.mark.parametrize(('n_value', 'modulo_value'), [(1, 1), (3, 2), (5, 3)])
def test_simple_program(n_value, modulo_value, save_generated_code=False):
    sdfg = simple_program.to_sdfg(simplify=False)
    sdfg.backend = dace.dtypes.BackendLanguage.Python

    rng = np.random.default_rng(7)
    x_start = rng.random((n_value, 7), dtype=np.float64)
    return_buffer = np.zeros_like(x_start)
    x_expected = np.copy(x_start)
    x_test = np.copy(x_start)
    y_start = rng.random((n_value * modulo_value, 7), dtype=np.float64)
    y_expected = np.copy(y_start)
    y_test = np.copy(y_start)
    expected = _reference_simple_program(x_expected, y_expected, modulo_value)

    with set_temporary('compiler', 'codegen_lineinfo', value=True):
        if save_generated_code:
            sdfg.save('simple_program.sdfg')
            sdfg.compile('simple_program.py', return_program_handle=False, validate=False)
        generated_code = sdfg.generate_code()[0].code
        sdfg(x=x_test, y=y_test, __return=return_buffer, N=n_value, Modulo=modulo_value)

    np.testing.assert_allclose(return_buffer, expected)
    np.testing.assert_allclose(x_test, x_expected)
    np.testing.assert_allclose(y_test, y_expected)
    assert 'x[0:3, 0:7]' not in generated_code
    assert ', s[0:1],' not in generated_code
    assert generated_code.count('# DaCe AUTO-GENERATED FILE. DO NOT MODIFY') == 1
    assert generated_code.splitlines().count('import numpy as _np') == 1


@dace.program
def very_simple_program():
    return 42


def test_very_simple_program():
    sdfg = very_simple_program.to_sdfg(simplify=False)
    sdfg.backend = dace.dtypes.BackendLanguage.Python
    ret = np.zeros(1, dtype=np.int64)
    sdfg(ret)
    assert ret[0] == 42


if __name__ == "__main__":
    test_very_simple_program()
    print("Very simple program test passed.")

    test_simple_program(3, 2, save_generated_code=True)
    print("simple_program test passed.")
