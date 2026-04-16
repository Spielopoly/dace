# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for the Python backend control flow code generation module.
"""

import ast
import pytest
import dace
from dace import dtypes
from dace.properties import CodeBlock
from dace.sdfg import SDFG, InterstateEdge
from dace.sdfg.state import (ControlFlowRegion, LoopRegion, ConditionalBlock, BreakBlock, ContinueBlock, ReturnBlock,
                              SDFGState)
from dace.codegen.py.control_flow import (control_flow_region_to_code, _unparse_py_expr, _unparse_codeblock)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeCodegen:
    """Minimal stub for the codegen parameter."""
    pass


def _make_dispatch():
    """Returns a dispatch_state callable that records calls and returns dummy code."""
    dispatched = []

    def dispatch_state(state: SDFGState) -> str:
        dispatched.append(state.label)
        return f'# state: {state.label}\n'

    return dispatch_state, dispatched


# ---------------------------------------------------------------------------
# Unit tests: helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_indent_empty(self):
        assert _indent('', 4) == ''

    def test_indent_single_line(self):
        assert _indent('x = 1', 4) == '    x = 1'

    def test_indent_multiline(self):
        result = _indent('a = 1\nb = 2\n', 8)
        lines = result.split('\n')
        assert lines[0] == '        a = 1'
        assert lines[1] == '        b = 2'

    def test_indent_preserves_blank_lines(self):
        result = _indent('a\n\nb', 4)
        lines = result.split('\n')
        assert lines[1] == ''  # blank line not indented

    def test_unparse_py_expr_string(self):
        assert _unparse_py_expr('x + 1', None) == 'x + 1'

    def test_unparse_py_expr_ast(self):
        node = ast.parse('x + 1').body[0]
        result = _unparse_py_expr(node, None)
        assert 'x' in result and '1' in result

    def test_unparse_py_expr_list(self):
        nodes = ast.parse('x = 1\ny = 2').body
        result = _unparse_py_expr(nodes, None)
        assert 'x' in result and 'y' in result

    def test_unparse_codeblock_none(self):
        assert _unparse_codeblock(None, None) == ''

    def test_unparse_codeblock_python(self):
        cb = CodeBlock('x + 1')
        result = _unparse_codeblock(cb, None)
        assert 'x' in result


# ---------------------------------------------------------------------------
# Unit tests: interstate assignments
# ---------------------------------------------------------------------------

class TestInterstateAssignments:
    def test_no_assignments(self):
        sdfg = SDFG('test')
        s0 = sdfg.add_state('s0')
        s1 = sdfg.add_state('s1')
        edge = sdfg.add_edge(s0, s1, InterstateEdge())
        result = _generate_interstate_assignments(edge, sdfg, 0)
        assert result == ''

    def test_single_assignment(self):
        sdfg = SDFG('test')
        s0 = sdfg.add_state('s0')
        s1 = sdfg.add_state('s1')
        edge = sdfg.add_edge(s0, s1, InterstateEdge(assignments={'i': '0'}))
        result = _generate_interstate_assignments(edge, sdfg, 0)
        assert result == 'i = 0'

    def test_multiple_assignments(self):
        sdfg = SDFG('test')
        s0 = sdfg.add_state('s0')
        s1 = sdfg.add_state('s1')
        edge = sdfg.add_edge(s0, s1, InterstateEdge(assignments={'i': '0', 'j': 'N'}))
        result = _generate_interstate_assignments(edge, sdfg, 4)
        assert '    i = 0' in result
        assert '    j = N' in result

    def test_indentation(self):
        sdfg = SDFG('test')
        s0 = sdfg.add_state('s0')
        s1 = sdfg.add_state('s1')
        edge = sdfg.add_edge(s0, s1, InterstateEdge(assignments={'x': '42'}))
        result = _generate_interstate_assignments(edge, sdfg, 8)
        assert result.startswith('        x = 42')


# ---------------------------------------------------------------------------
# Unit tests: loop region
# ---------------------------------------------------------------------------

class TestLoopRegion:
    def test_simple_while_loop(self):
        sdfg = SDFG('test_loop')
        loop = LoopRegion('myloop', condition_expr='i < N', sdfg=sdfg)
        sdfg.add_node(loop, is_start_block=True)
        loop.add_state('loop_body', is_start_block=True)

        dispatch, dispatched = _make_dispatch()
        codegen = _FakeCodegen()

        result = _loop_region_to_code(loop, dispatch, codegen, {}, indent=0)
        assert 'while' in result
        assert 'i < N' in result or 'i<N' in result

    def test_for_style_loop(self):
        sdfg = SDFG('test_for')
        loop = LoopRegion('forloop', condition_expr='i < 10', loop_var='i',
                          initialize_expr='i = 0', update_expr='i = i + 1', sdfg=sdfg)
        sdfg.add_node(loop, is_start_block=True)
        loop.add_state('loop_body', is_start_block=True)

        dispatch, dispatched = _make_dispatch()
        codegen = _FakeCodegen()

        result = _loop_region_to_code(loop, dispatch, codegen, {}, indent=0)
        assert 'i = 0' in result
        assert 'while' in result
        assert 'i = i + 1' in result or 'i = (i + 1)' in result

    def test_inverted_loop(self):
        sdfg = SDFG('test_inverted')
        loop = LoopRegion('dowhile', condition_expr='x > 0', inverted=True, sdfg=sdfg)
        sdfg.add_node(loop, is_start_block=True)
        loop.add_state('loop_body', is_start_block=True)

        dispatch, _ = _make_dispatch()
        codegen = _FakeCodegen()

        result = _loop_region_to_code(loop, dispatch, codegen, {}, indent=0)
        assert 'while True:' in result
        assert 'break' in result

    def test_loop_indentation(self):
        sdfg = SDFG('test_indent')
        loop = LoopRegion('myloop', condition_expr='True', sdfg=sdfg)
        sdfg.add_node(loop, is_start_block=True)
        loop.add_state('body', is_start_block=True)

        dispatch, _ = _make_dispatch()
        codegen = _FakeCodegen()

        result = _loop_region_to_code(loop, dispatch, codegen, {}, indent=4)
        lines = result.split('\n')
        while_line = [l for l in lines if 'while' in l][0]
        assert while_line.startswith('    ')


# ---------------------------------------------------------------------------
# Unit tests: conditional block
# ---------------------------------------------------------------------------

class TestConditionalBlock:
    def test_if_else(self):
        sdfg = SDFG('test_cond')
        cond_block = ConditionalBlock('mycond', sdfg=sdfg)
        sdfg.add_node(cond_block, is_start_block=True)

        then_region = ControlFlowRegion('then_branch', sdfg=sdfg)
        then_region.add_state('then_body', is_start_block=True)
        else_region = ControlFlowRegion('else_branch', sdfg=sdfg)
        else_region.add_state('else_body', is_start_block=True)

        cond_block.add_branch(CodeBlock('x > 0'), then_region)
        cond_block.add_branch(None, else_region)

        dispatch, dispatched = _make_dispatch()
        codegen = _FakeCodegen()

        result = _conditional_block_to_code(cond_block, dispatch, codegen, {}, indent=0)
        assert 'if' in result
        assert 'else:' in result
        assert 'then_body' in result or len(dispatched) == 2

    def test_if_elif_else(self):
        sdfg = SDFG('test_elif')
        cond_block = ConditionalBlock('mycond', sdfg=sdfg)
        sdfg.add_node(cond_block, is_start_block=True)

        r1 = ControlFlowRegion('branch1', sdfg=sdfg)
        r1.add_state('b1_body', is_start_block=True)
        r2 = ControlFlowRegion('branch2', sdfg=sdfg)
        r2.add_state('b2_body', is_start_block=True)
        r3 = ControlFlowRegion('branch3', sdfg=sdfg)
        r3.add_state('b3_body', is_start_block=True)

        cond_block.add_branch(CodeBlock('x == 1'), r1)
        cond_block.add_branch(CodeBlock('x == 2'), r2)
        cond_block.add_branch(None, r3)

        dispatch, _ = _make_dispatch()
        codegen = _FakeCodegen()

        result = _conditional_block_to_code(cond_block, dispatch, codegen, {}, indent=0)
        assert 'if' in result
        assert 'elif' in result
        assert 'else:' in result

    def test_conditional_indentation(self):
        sdfg = SDFG('test_indent')
        cond_block = ConditionalBlock('mycond', sdfg=sdfg)
        sdfg.add_node(cond_block, is_start_block=True)

        r1 = ControlFlowRegion('branch', sdfg=sdfg)
        r1.add_state('body', is_start_block=True)
        cond_block.add_branch(CodeBlock('True'), r1)

        dispatch, _ = _make_dispatch()
        codegen = _FakeCodegen()

        result = _conditional_block_to_code(cond_block, dispatch, codegen, {}, indent=8)
        lines = result.split('\n')
        if_line = [l for l in lines if 'if' in l][0]
        assert if_line.startswith('        ')


# ---------------------------------------------------------------------------
# Unit tests: full region code generation
# ---------------------------------------------------------------------------

class TestControlFlowRegionToCode:
    def test_single_state(self):
        sdfg = SDFG('test_single')
        sdfg.add_state('s0', is_start_block=True)

        dispatch, dispatched = _make_dispatch()
        codegen = _FakeCodegen()

        result = control_flow_region_to_code(sdfg, dispatch, codegen, {}, indent=0)
        assert 's0' in dispatched

    def test_linear_chain(self):
        sdfg = SDFG('test_chain')
        s0 = sdfg.add_state('s0', is_start_block=True)
        s1 = sdfg.add_state('s1')
        sdfg.add_edge(s0, s1, InterstateEdge())

        dispatch, dispatched = _make_dispatch()
        codegen = _FakeCodegen()

        result = control_flow_region_to_code(sdfg, dispatch, codegen, {}, indent=0)
        assert 's0' in dispatched
        assert 's1' in dispatched

    def test_conditional_edge(self):
        sdfg = SDFG('test_cond_edge')
        s0 = sdfg.add_state('s0', is_start_block=True)
        s1 = sdfg.add_state('s1')
        sdfg.add_edge(s0, s1, InterstateEdge(condition=CodeBlock('x > 0')))

        dispatch, _ = _make_dispatch()
        codegen = _FakeCodegen()

        result = control_flow_region_to_code(sdfg, dispatch, codegen, {}, indent=0)
        assert 'if' in result

    def test_assignment_edge(self):
        sdfg = SDFG('test_assign')
        s0 = sdfg.add_state('s0', is_start_block=True)
        s1 = sdfg.add_state('s1')
        sdfg.add_edge(s0, s1, InterstateEdge(assignments={'i': '0'}))

        dispatch, _ = _make_dispatch()
        codegen = _FakeCodegen()

        result = control_flow_region_to_code(sdfg, dispatch, codegen, {}, indent=0)
        assert 'i = 0' in result

    def test_branching_uses_state_machine(self):
        """A region with branching (out_degree > 1) should use the state-machine fallback."""
        sdfg = SDFG('test_branch')
        s0 = sdfg.add_state('s0', is_start_block=True)
        s1 = sdfg.add_state('s1')
        s2 = sdfg.add_state('s2')
        sdfg.add_edge(s0, s1, InterstateEdge(condition=CodeBlock('x > 0')))
        sdfg.add_edge(s0, s2, InterstateEdge(condition=CodeBlock('x <= 0')))

        dispatch, dispatched = _make_dispatch()
        codegen = _FakeCodegen()

        result = control_flow_region_to_code(sdfg, dispatch, codegen, {}, indent=0)
        # Should use state machine pattern
        assert '__state_' in result
        assert 'while True:' in result

    def test_no_goto_in_output(self):
        """Ensure no C++ constructs leak into the Python output."""
        sdfg = SDFG('test_no_cpp')
        s0 = sdfg.add_state('s0', is_start_block=True)
        s1 = sdfg.add_state('s1')
        sdfg.add_edge(s0, s1, InterstateEdge())

        dispatch, _ = _make_dispatch()
        codegen = _FakeCodegen()

        result = control_flow_region_to_code(sdfg, dispatch, codegen, {}, indent=0)
        assert 'goto' not in result
        assert '{' not in result
        assert '}' not in result
        assert ';' not in result


# ---------------------------------------------------------------------------
# Unit tests: break / continue / return blocks
# ---------------------------------------------------------------------------

class TestSpecialBlocks:
    def test_break_block(self):
        sdfg = SDFG('test_break')
        loop = LoopRegion('loop', condition_expr='True', sdfg=sdfg)
        sdfg.add_node(loop, is_start_block=True)
        s = loop.add_state('body', is_start_block=True)
        brk = loop.add_break('brk')
        loop.add_edge(s, brk, InterstateEdge())

        dispatch, _ = _make_dispatch()
        codegen = _FakeCodegen()

        result = _loop_region_to_code(loop, dispatch, codegen, {}, indent=0)
        assert 'break' in result

    def test_continue_block(self):
        """Verify the continue block is generated when it's not at the tail."""
        sdfg = SDFG('test_continue')
        loop = LoopRegion('loop', condition_expr='True', sdfg=sdfg)
        sdfg.add_node(loop, is_start_block=True)
        s = loop.add_state('body', is_start_block=True)
        cont = loop.add_continue('cont')
        s2 = loop.add_state('after_cont')
        loop.add_edge(s, cont, InterstateEdge(condition=CodeBlock('flag')))
        loop.add_edge(s, s2, InterstateEdge(condition=CodeBlock('not flag')))

        dispatch, _ = _make_dispatch()
        codegen = _FakeCodegen()

        result = _loop_region_to_code(loop, dispatch, codegen, {}, indent=0)
        assert 'continue' in result

    def test_return_block(self):
        sdfg = SDFG('test_return')
        s = sdfg.add_state('body', is_start_block=True)
        ret = sdfg.add_return('ret')
        sdfg.add_edge(s, ret, InterstateEdge())

        dispatch, _ = _make_dispatch()
        codegen = _FakeCodegen()

        result = control_flow_region_to_code(sdfg, dispatch, codegen, {}, indent=0)
        assert 'return' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
