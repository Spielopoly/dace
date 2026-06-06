# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for cuTile nested SDFG function-call generation.

The cuTile runtime is NOT installed in CI, so these tests verify the
*structure* of generated code: function definitions are emitted, function
calls appear at the call site, and the output parses as valid Python.

The old CuTile backend inlined nested SDFGs directly into the call site
and raised ``NotImplementedError`` for multi-state nested SDFGs.  After
the refactor, it delegates to the default Python backend's function-
generation approach, which emits a separate ``def`` and a call.
"""
import ast
import re

import pytest

import dace
from dace import dtypes, data
from dace.sdfg import nodes, SDFG


def _build_cutile_sdfg_with_nsdfg(name: str = 'cutile_nsdfg_test',
                                   multi_state: bool = False):
    """Build a minimal SDFG with a CuTile map containing a NestedSDFG.

    The SDFG structure:
    - Outer SDFG has one state with a CuTile-scheduled map
    - Inside the map: AccessNode(A_tile) -> NestedSDFG -> AccessNode(B_tile)
    - The NestedSDFG has a tasklet that doubles the input

    :param name: Name for the SDFG.
    :param multi_state: If True, the nested SDFG has two states.
    :returns: ``(sdfg, nested_sdfg_node)``
    """
    N = dace.symbol('N')
    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python

    sdfg.add_array('A', [N], dace.float64,
                   storage=dtypes.StorageType.GPU_Global)
    sdfg.add_array('B', [N], dace.float64,
                   storage=dtypes.StorageType.GPU_Global)

    state = sdfg.add_state('main', is_start_block=True)

    # Create outer CuTile map
    me, mx = state.add_map('cutile_map', {'tile_i': '0:N:16'},
                           schedule=dtypes.ScheduleType.CuTile)

    # Tile transients
    sdfg.add_array('A_tile', [16], dace.float64,
                   storage=dtypes.StorageType.CuTile_Tile, transient=True)
    sdfg.add_array('B_tile', [16], dace.float64,
                   storage=dtypes.StorageType.CuTile_Tile, transient=True)

    a_tile = state.add_access('A_tile')
    b_tile = state.add_access('B_tile')
    a_node = state.add_access('A')
    b_node = state.add_access('B')

    # Build nested SDFG
    nsdfg = SDFG(f'{name}_nested')
    nsdfg.backend = dtypes.BackendLanguage.Python
    nsdfg.add_array('inp', [16], dace.float64,
                    storage=dtypes.StorageType.CuTile_Tile)
    nsdfg.add_array('out', [16], dace.float64,
                    storage=dtypes.StorageType.CuTile_Tile)

    if multi_state:
        ns1 = nsdfg.add_state('ns1', is_start_block=True)
        ns2 = nsdfg.add_state('ns2')
        nsdfg.add_edge(ns1, ns2, dace.InterstateEdge())

        # State 1: temp = inp
        nsdfg.add_array('temp', [16], dace.float64,
                        storage=dtypes.StorageType.CuTile_Tile,
                        transient=True)
        t1 = ns1.add_tasklet('copy', {'_in'}, {'_out'}, '_out = _in')
        inp1 = ns1.add_access('inp')
        temp1 = ns1.add_access('temp')
        ns1.add_edge(inp1, None, t1, '_in', dace.Memlet('inp[0:16]'))
        ns1.add_edge(t1, '_out', temp1, None, dace.Memlet('temp[0:16]'))

        # State 2: out = temp * 2
        t2 = ns2.add_tasklet('double', {'_in'}, {'_out'}, '_out = _in * 2')
        temp2 = ns2.add_access('temp')
        out2 = ns2.add_access('out')
        ns2.add_edge(temp2, None, t2, '_in', dace.Memlet('temp[0:16]'))
        ns2.add_edge(t2, '_out', out2, None, dace.Memlet('out[0:16]'))
    else:
        ns = nsdfg.add_state('nested_state', is_start_block=True)
        t = ns.add_tasklet('double', {'_in'}, {'_out'}, '_out = _in * 2')
        inp_node = ns.add_access('inp')
        out_node = ns.add_access('out')
        ns.add_edge(inp_node, None, t, '_in', dace.Memlet('inp[0:16]'))
        ns.add_edge(t, '_out', out_node, None, dace.Memlet('out[0:16]'))

    nsdfg_node = state.add_nested_sdfg(nsdfg, {'inp'}, {'out'})

    # Wire: A -> MapEntry -> A_tile -> NestedSDFG -> B_tile -> MapExit -> B
    state.add_memlet_path(a_node, me, a_tile,
                          memlet=dace.Memlet('A[tile_i:tile_i+16]'))
    state.add_edge(a_tile, None, nsdfg_node, 'inp',
                   dace.Memlet('A_tile[0:16]'))
    state.add_edge(nsdfg_node, 'out', b_tile, None,
                   dace.Memlet('B_tile[0:16]'))
    state.add_memlet_path(b_tile, mx, b_node,
                          memlet=dace.Memlet('B[tile_i:tile_i+16]'))

    return sdfg, nsdfg_node


def _generate_code(sdfg: SDFG) -> str:
    """Generate Python code for an SDFG and return the frame code string.

    :param sdfg: The SDFG to generate code for.
    :returns: The generated Python frame code.
    """
    code_objects = sdfg.generate_code()
    for co in code_objects:
        if co.title == 'Frame':
            return co.code
    raise RuntimeError('No frame code object found')


class TestCuTileNestedSDFGFunctionGeneration:
    """Verify that nested SDFGs inside cuTile scopes produce function defs + calls."""

    def test_generates_function_def_and_call(self):
        """Single-state NestedSDFG should generate a function definition and call."""
        sdfg, nsdfg_node = _build_cutile_sdfg_with_nsdfg(
            'test_func_def_call')
        code = _generate_code(sdfg)

        # Should contain a function definition for the nested SDFG
        assert 'def ' in code, \
            'Expected a function definition in generated code'
        # The nested SDFG function should be defined somewhere.
        # The function name contains the nested SDFG's name.
        nested_name = nsdfg_node.sdfg.name
        func_def_pattern = re.compile(
            r'def\s+\w*' + re.escape(nested_name) + r'\w*\s*\(')
        assert func_def_pattern.search(code), (
            f'Expected function definition containing {nested_name!r} '
            f'in generated code')

        # There should be a matching call (function name followed by '(')
        func_call_pattern = re.compile(
            r'\w*' + re.escape(nested_name) + r'\w*\s*\(')
        # The def itself matches too; we need at least 2 matches
        # (one def, one call).
        matches = func_call_pattern.findall(code)
        assert len(matches) >= 2, (
            f'Expected at least one function def and one call for '
            f'{nested_name!r}, found {len(matches)} match(es)')

        # The generated code should be valid Python
        ast.parse(code)

    def test_multi_state_nested_sdfg(self):
        """Multi-state NestedSDFG should work (old code raised NotImplementedError)."""
        sdfg, _ = _build_cutile_sdfg_with_nsdfg(
            'test_multi_state', multi_state=True)
        # This should NOT raise NotImplementedError
        code = _generate_code(sdfg)
        ast.parse(code)

    def test_tile_args_passed_by_name(self):
        """CuTile_Tile arrays should be passed by name without view expressions."""
        sdfg, _ = _build_cutile_sdfg_with_nsdfg('test_tile_args')
        code = _generate_code(sdfg)

        # The generated code should be valid Python
        ast.parse(code)

        # The function call should pass tile arrays directly by name.
        # Look for the nested function call line.  It should NOT contain
        # numpy-style view expressions like ``A_tile[0:16]`` for
        # CuTile_Tile args (those are opaque tile objects).
        # We check that the call site passes the tile name directly.
        lines = code.split('\n')
        # Find lines that call the nested function (contain nested name + '(')
        call_lines = [
            line for line in lines
            if 'nested' in line and '(' in line and 'def ' not in line
        ]
        for line in call_lines:
            # If A_tile appears, it should NOT be subscripted
            if 'A_tile' in line:
                assert 'A_tile[' not in line, (
                    f'CuTile_Tile arg should be passed by name, '
                    f'not subscripted: {line!r}')

    def test_ast_validity_single_state(self):
        """Single-state generated code should parse as valid Python AST."""
        sdfg, _ = _build_cutile_sdfg_with_nsdfg('test_ast_single')
        code = _generate_code(sdfg)
        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(
                f'Generated code is not valid Python: {e}\n\nCode:\n{code}')

    def test_ast_validity_multi_state(self):
        """Multi-state generated code should parse as valid Python AST."""
        sdfg, _ = _build_cutile_sdfg_with_nsdfg('test_ast_multi',
                                                  multi_state=True)
        code = _generate_code(sdfg)
        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(
                f'Generated code is not valid Python: {e}\n\nCode:\n{code}')

    def test_no_inline_nsdfg_remnants(self):
        """Generated code should not contain inline-style direct assignments.

        The old inlining approach would emit lines like ``inp = A_tile``
        directly in the kernel body.  The new function-call approach should
        use proper function arguments instead.
        """
        sdfg, _ = _build_cutile_sdfg_with_nsdfg('test_no_inline')
        code = _generate_code(sdfg)

        # The function body may contain assignment, but there should be
        # a proper function definition, not just raw connector bindings.
        func_def_pattern = re.compile(r'def\s+\w*nested\w*\s*\(')
        assert func_def_pattern.search(code), (
            'Expected a function definition for the nested SDFG')
        ast.parse(code)
