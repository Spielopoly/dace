import dace
from dace.config import set_temporary
import numpy as np
import pytest

def foo(x: dace.float64[3,7]):
    s = np.sum(x)
    return 5.0 + s
@dace.program
def simple_program(x: dace.float64[3,7], y: dace.float64[3,7]):
    for i in range(3):
        for j in range(7):
            if i % 2 == 0 and j % 2 == 0:
                y[i, j] = x[i, j] * foo(x[i, j])
    return x + y

@pytest.mark.xfail(reason="Python backend does not yet support nested sdfgs, or map scopes")
def test_simple_program():
    
    sdfg = simple_program.to_sdfg(simplify=False)
    sdfg.backend = dace.dtypes.BackendLanguage.Python
    sdfg.save('simple_program.sdfg')
    with set_temporary('compiler', 'codegen_lineinfo', value=False):
        code = sdfg.generate_code()
        
        x = np.random.rand(3, 7)
        y = np.random.rand(3, 7)
        ret = np.zeros((3, 7), dtype=np.float64)
        result = sdfg(x=x, y=y, __return=ret)
    
    return code

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
    
    code = test_simple_program()
    for c in code:
        print(f"Code Object: {c.name}, Language: {c.language}, Title: {c.title}")
        print(c.code)
        