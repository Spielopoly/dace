import dace
from dace.config import set_temporary

def test_simple_program():
    @dace.program
    def simple_program(x: dace.float64[3,7], y: dace.float64[3,7]):
        for i in range(3):
            for j in range(7):
                if i % 2 == 0 and j % 2 == 0:
                    y[i, j] = x[i, j] * 2.0
    
    sdfg = simple_program.to_sdfg(simplify=True)
    sdfg.backend = dace.dtypes.BackendLanguage.Python
    with set_temporary('compiler', 'codegen_lineinfo', value=False):
        code = sdfg.generate_code()
    
    return code

if __name__ == "__main__":
    code = test_simple_program()
    for c in code:
        print(f"Code Object: {c.name}, Language: {c.language}, Title: {c.title}")
        print(c.code)