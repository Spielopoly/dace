import dace


def test_empty_program():
    @dace.program
    def empty_program():
        pass
    
    sdfg = empty_program.to_sdfg(simplify=False)
    sdfg.backend = dace.dtypes.BackendLanguage.Python
    code = sdfg.generate_code()
    
    return code

if __name__ == "__main__":
    code = test_empty_program()
    for c in code:
        print(f"Code Object: {c.name}, Language: {c.language}, Title: {c.title}")
        print(c.code)