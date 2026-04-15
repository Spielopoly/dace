import dace


def test_empty_program():
    @dace.program
    def empty_program():
        pass
    
    sdfg = empty_program.to_sdfg(simplify=False)
    sdfg.backend = dace.dtypes.BackendLanguage.Python
    code = sdfg.generate_code()
    
    return

if __name__ == "__main__":
    test_empty_program()