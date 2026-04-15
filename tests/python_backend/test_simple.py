import dace


def test_simple_program():
    @dace.program
    def simple_program(x: dace.float64):
        x += 1.0
    
    sdfg = simple_program.to_sdfg(simplify=False)
    sdfg.backend = dace.dtypes.BackendLanguage.Python
    code = sdfg.generate_code()
    
    return code

if __name__ == "__main__":
    code = test_simple_program()
    for c in code:
        print(f"Code Object: {c.name}, Language: {c.language}, Title: {c.title}")
        print(c.code)