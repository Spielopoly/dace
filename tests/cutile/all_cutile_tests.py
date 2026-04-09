if __name__ == "__main__":
    import sys
    import os
    import glob
    import pytest

    # Get the directory of this file
    file_name = os.path.abspath(__file__)
    current_dir = os.path.dirname(file_name)
    
    # Find all test files in the directory
    test_files = glob.glob(os.path.join(current_dir, "test_*.py"))
    test_files += glob.glob(os.path.join(current_dir, "*_test.py"))
    test_files = [file for file in test_files if file != file_name]
    
    sys.exit(pytest.main(test_files + ["-q", "--tb=short"]))
else:
    from cutile_frontend_test import *
    from cutile_if_else_op_test import *
    from cutile_ifelse_test import *
    from cutile_masked_expr_test import *
    from cutile_multi_op_test import *
    from cutile_test import *
    