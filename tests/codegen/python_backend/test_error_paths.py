# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Error-path tests for the DaCe Python backend.

These tests target reachable exception branches in the Python backend code
generation stack without modifying existing tests or forcing invalid full-SDFG
pipelines when a smaller unit-style construction is sufficient.
"""

import pytest

import dace
from dace import dtypes
from dace.codegen.py.compiled_sdfg import PythonCompiledSDFG, compile_python_sdfg
from dace.codegen.py.control_flow import (
    _unparse_codeblock,
    _write_conditional_block,
    _write_dispatch_block,
    _write_loop_region,
)
from dace.codegen.py.framecode import DaCePythonCodeGenerator, codeblock_to_python
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.codegen.py.python_target import PythonCodeGen
from dace.memlet import Memlet
from dace.properties import CodeBlock
from dace.sdfg import SDFG, NodeNotExpandedError, nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion, LoopRegion


class _FakeCodeObject:
    def __init__(self, code: str):
        self.code = code


class _DummyCodegen:
    pass


class _DummyLibraryNode(nodes.LibraryNode):
    _dace_library_name = 'dummy_test_library'
    implementations = {}
    default_implementation = None


def _make_sdfg(name: str) -> SDFG:
    sdfg = SDFG(name)
    sdfg.backend = dtypes.BackendLanguage.Python
    return sdfg


def _make_python_codegen(sdfg: SDFG):
    frame = DaCePythonCodeGenerator(sdfg)
    target = PythonCodeGen(frame, sdfg)
    return frame, target


def _make_streams():
    return PythonCodeIOStream(), PythonCodeIOStream()


def _make_dispatch(result: str = '# state\n'):
    def _dispatch(_state):
        return result

    return _dispatch


def _make_conditional_branch_region(label: str, sdfg: SDFG) -> ControlFlowRegion:
    region = ControlFlowRegion(label, sdfg=sdfg)
    region.add_state(f'{label}_state', is_start_block=True)
    return region


def test_compiled_sdfg_missing_generated_function():
    sdfg = _make_sdfg('expected_name')

    with pytest.raises(RuntimeError, match="does not define function 'expected_name'"):
        PythonCompiledSDFG(sdfg, 'def other_name():\n    return 1\n')


def test_compile_python_sdfg_empty_code_object_list():
    sdfg = _make_sdfg('empty_codegen')

    with pytest.raises(RuntimeError, match='No code objects generated'):
        compile_python_sdfg(sdfg, [])


def test_generate_node_unknown_type_raises_not_implemented():
    sdfg = _make_sdfg('unknown_node')
    sdfg.add_array('A', [1], dace.float64)
    state = sdfg.add_state('state')
    _, map_exit = state.add_map('m', {'i': '0:1'}, schedule=dtypes.ScheduleType.Sequential)
    _, target = _make_python_codegen(sdfg)
    function_stream, callsite_stream = _make_streams()

    with pytest.raises(NotImplementedError, match='MapExit'):
        target.generate_node(sdfg, sdfg, state, state.block_id, map_exit, function_stream, callsite_stream)


def test_generate_node_unexpanded_library_node_raises_node_not_expanded():
    sdfg = _make_sdfg('unexpanded_library')
    state = sdfg.add_state('state')
    libnode = _DummyLibraryNode('libnode')
    state.add_node(libnode)
    _, target = _make_python_codegen(sdfg)
    function_stream, callsite_stream = _make_streams()

    with pytest.raises(NodeNotExpandedError):
        target.generate_node(sdfg, sdfg, state, state.block_id, libnode, function_stream, callsite_stream)


def test_tasklet_non_python_language_raises_not_implemented():
    sdfg = _make_sdfg('cpp_tasklet')
    sdfg.add_scalar('x', dace.float64)
    sdfg.add_scalar('y', dace.float64)
    state = sdfg.add_state('state')
    read_x = state.add_read('x')
    write_y = state.add_write('y')
    tasklet = state.add_tasklet('task', {'inp'}, {'out'}, 'out = inp', language=dtypes.Language.CPP)
    state.add_edge(read_x, None, tasklet, 'inp', Memlet(data='x'))
    state.add_edge(tasklet, 'out', write_y, None, Memlet(data='y'))
    _, target = _make_python_codegen(sdfg)
    function_stream, callsite_stream = _make_streams()

    with pytest.raises(NotImplementedError, match='only supports Python tasklets'):
        target.generate_node(sdfg, sdfg, state, state.block_id, tasklet, function_stream, callsite_stream)


def test_tasklet_code_to_code_input_memlet_raises_not_implemented():
    sdfg = _make_sdfg('code_to_code_input')
    sdfg.add_scalar('x', dace.float64)
    sdfg.add_scalar('y', dace.float64)
    state = sdfg.add_state('state')
    read_x = state.add_read('x')
    write_y = state.add_write('y')
    tasklet = state.add_tasklet('task', {'inp'}, {'out'}, 'out = inp')
    state.add_edge(read_x, None, tasklet, 'inp', Memlet())
    state.add_edge(tasklet, 'out', write_y, None, Memlet(data='y'))
    _, target = _make_python_codegen(sdfg)
    function_stream, callsite_stream = _make_streams()

    with pytest.raises(NotImplementedError, match='Code-to-code memlets not supported'):
        target.generate_node(sdfg, sdfg, state, state.block_id, tasklet, function_stream, callsite_stream)


def test_tasklet_code_to_code_output_memlet_raises_not_implemented():
    sdfg = _make_sdfg('code_to_code_output')
    sdfg.add_scalar('x', dace.float64)
    sdfg.add_scalar('y', dace.float64)
    state = sdfg.add_state('state')
    read_x = state.add_read('x')
    write_y = state.add_write('y')
    tasklet = state.add_tasklet('task', {'inp'}, {'out'}, 'out = inp')
    state.add_edge(read_x, None, tasklet, 'inp', Memlet(data='x'))
    state.add_edge(tasklet, 'out', write_y, None, Memlet())
    _, target = _make_python_codegen(sdfg)
    function_stream, callsite_stream = _make_streams()

    with pytest.raises(NotImplementedError, match='Code-to-code memlets not supported'):
        target.generate_node(sdfg, sdfg, state, state.block_id, tasklet, function_stream, callsite_stream)


def test_allocate_array_unsupported_data_descriptor_raises_not_implemented():
    sdfg = _make_sdfg('stream_alloc')
    sdfg.add_stream('pipe', dace.float64, transient=True)
    state = sdfg.add_state('state')
    access = state.add_access('pipe')
    _, target = _make_python_codegen(sdfg)
    function_stream, declaration_stream = _make_streams()
    allocation_stream = PythonCodeIOStream()

    with pytest.raises(NotImplementedError, match='cannot allocate Stream'):
        target.allocate_array(
            sdfg,
            sdfg,
            state,
            state.block_id,
            access,
            sdfg.arrays['pipe'],
            function_stream,
            declaration_stream,
            allocation_stream,
        )


def test_generate_external_memory_management_raises_for_external_lifetime():
    sdfg = _make_sdfg('external_memory')
    sdfg.add_array('A', [4], dace.float64, lifetime=dtypes.AllocationLifetime.External)
    sdfg.add_state('state')
    codegen = DaCePythonCodeGenerator(sdfg)

    with pytest.raises(NotImplementedError, match='External memory management'):
        codegen.generate_external_memory_management(sdfg, PythonCodeIOStream())


def test_codeblock_to_python_rejects_non_python_codeblocks():
    with pytest.raises(ValueError, match='cannot be converted to Python'):
        codeblock_to_python(CodeBlock('int x = 0;', language=dtypes.Language.CPP))


def test_get_schedule_invalid_scope_type_raises_type_error():
    sdfg = _make_sdfg('invalid_schedule')
    sdfg.add_state('state')
    codegen = DaCePythonCodeGenerator(sdfg)

    with pytest.raises(TypeError):
        codegen._get_schedule(object())


def test_generate_code_raises_when_not_all_states_are_generated(monkeypatch):
    sdfg = _make_sdfg('missing_states')
    first = sdfg.add_state('first', is_start_block=True)
    second = sdfg.add_state('second')
    sdfg.add_edge(first, second, dace.InterstateEdge())
    codegen = DaCePythonCodeGenerator(sdfg)

    monkeypatch.setattr(codegen, 'determine_allocation_lifetime', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(codegen, 'allocate_arrays_in_scope', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(codegen, 'deallocate_arrays_in_scope', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(codegen, 'generate_states', lambda *_args, **_kwargs: {first})

    with pytest.raises(RuntimeError, match='Not all states were generated'):
        codegen.generate_code(sdfg, None)


def test_unparse_codeblock_rejects_non_python_code():
    sdfg = _make_sdfg('non_python_codeblock')

    with pytest.raises(NotImplementedError, match='cannot be unparsed to Python'):
        _unparse_codeblock(CodeBlock('x++;', language=dtypes.Language.CPP), sdfg)


def test_loop_region_rejects_non_python_condition_codeblock():
    sdfg = _make_sdfg('loop_non_python_condition')
    loop = LoopRegion('loop', condition_expr='i < 4', sdfg=sdfg)
    sdfg.add_node(loop, is_start_block=True)
    loop.add_state('body', is_start_block=True)
    loop.loop_condition = CodeBlock('i < 4', language=dtypes.Language.CPP)

    with pytest.raises(NotImplementedError, match='cannot be unparsed to Python'):
        _write_loop_region(loop, _make_dispatch(), _DummyCodegen(), {}, PythonCodeIOStream())


def test_conditional_block_none_condition_as_first_branch_raises_runtime_error():
    sdfg = _make_sdfg('conditional_none_first')
    conditional = ConditionalBlock('cond', sdfg=sdfg)
    sdfg.add_node(conditional, is_start_block=True)
    else_region = _make_conditional_branch_region('else_branch', sdfg)
    conditional.add_branch(None, else_region)

    with pytest.raises(RuntimeError, match='Missing branch condition'):
        _write_conditional_block(conditional, _make_dispatch(), _DummyCodegen(), {}, PythonCodeIOStream())


def test_conditional_block_none_condition_for_non_final_branch_raises_runtime_error():
    sdfg = _make_sdfg('conditional_none_nonfinal')
    conditional = ConditionalBlock('cond', sdfg=sdfg)
    sdfg.add_node(conditional, is_start_block=True)
    then_region = _make_conditional_branch_region('then_branch', sdfg)
    invalid_region = _make_conditional_branch_region('invalid_branch', sdfg)
    final_region = _make_conditional_branch_region('final_branch', sdfg)
    conditional.add_branch(CodeBlock('flag'), then_region)
    conditional.add_branch(None, invalid_region)
    conditional.add_branch(CodeBlock('other_flag'), final_region)

    with pytest.raises(RuntimeError, match='Missing branch condition'):
        _write_conditional_block(conditional, _make_dispatch(), _DummyCodegen(), {}, PythonCodeIOStream())


def test_dispatch_block_unknown_control_flow_type_raises_not_implemented():
    with pytest.raises(NotImplementedError, match='not implemented'):
        _write_dispatch_block(object(), _make_dispatch(), _DummyCodegen(), {}, PythonCodeIOStream())


def test_compile_python_sdfg_uses_first_code_object_only():
    sdfg = _make_sdfg('first_code_object')
    code_objects = [
        _FakeCodeObject('def first_code_object():\n    return 7\n'),
        _FakeCodeObject('def first_code_object():\n    return 99\n'),
    ]

    compiled = compile_python_sdfg(sdfg, code_objects)

    assert compiled() == 7