import dace
from dace.config import set_temporary

def test_simple_program():
    @dace.program
    def simple_program(x: dace.float64):
        if x > 0:
            return x
        else:
            return -x
    
    sdfg = simple_program.to_sdfg(simplify=False)
    sdfg.backend = dace.dtypes.BackendLanguage.Python
    with set_temporary('compiler', 'codegen_lineinfo', value=False):
        code = sdfg.generate_code()
    
    return code

if __name__ == "__main__":
    code = test_simple_program()
    for c in code:
        print(f"Code Object: {c.name}, Language: {c.language}, Title: {c.title}")
        print(c.code)