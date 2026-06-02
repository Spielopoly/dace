import ast

import dace
import pytest

from dace import dtypes
from dace.codegen.py.control_flow import (
    _is_child_of,
    _state_label,
    _unparse_codeblock,
    _unparse_py_expr,
    _write_conditional_block,
    _write_control_flow_region,
    _write_dispatch_block,
    _write_interstate_assignments,
    _write_state_machine,
    _write_structured_region,
)
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.properties import CodeBlock
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import (
    BreakBlock,
    ConditionalBlock,
    ContinueBlock,
    ControlFlowBlock,
    ControlFlowRegion,
    LoopRegion,
    ReturnBlock,
    UnstructuredControlFlow,
)


def _new_sdfg(name: str = "cf_test") -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.backend = dace.dtypes.BackendLanguage.Python
    return sdfg


def _dispatch_text(state) -> str:
    return f"visited.append('{state.label}')"


def _render_control_flow(region, dispatch_state=_dispatch_text) -> str:
    stream = PythonCodeIOStream()
    _write_control_flow_region(region, dispatch_state, None, {}, stream)
    return stream.getvalue()


def _render_state_machine(region, dispatch_state=_dispatch_text) -> str:
    stream = PythonCodeIOStream()
    _write_state_machine(region, dispatch_state, None, {}, stream)
    return stream.getvalue()


def _render_structured(region, dispatch_state=_dispatch_text) -> str:
    stream = PythonCodeIOStream()
    _write_structured_region(region, dispatch_state, None, {}, stream)
    return stream.getvalue()


def _make_region_with_two_states(label: str = "r"):
    sdfg = _new_sdfg(label)
    region = ControlFlowRegion(label, sdfg=sdfg)
    s0 = region.add_state("s0", is_start_block=True)
    s1 = region.add_state("s1")
    return sdfg, region, s0, s1


def _attach_to_parent(block, sdfg: dace.SDFG) -> ControlFlowRegion:
    parent = ControlFlowRegion("parent", sdfg=sdfg)
    parent.add_node(block, is_start_block=True)
    return parent


class _LenSkewedEdges(list):
    def __init__(self, edges, fake_len: int):
        super().__init__(edges)
        self._fake_len = fake_len

    def __len__(self):
        return self._fake_len


# -----------------------------------------------------------------------------
# 1) Structured vs state-machine routing
# -----------------------------------------------------------------------------


def test_structured_region_selected() -> None:
    _, region, s0, s1 = _make_region_with_two_states("structured_route")
    region.add_edge(s0, s1, InterstateEdge())

    code = _render_control_flow(region)

    assert "__state_" not in code
    assert "visited.append('s0')" in code
    assert "visited.append('s1')" in code


def test_state_machine_selected() -> None:
    _, region, s0, s1 = _make_region_with_two_states("sm_route")
    s2 = region.add_state("s2")
    region.add_edge(s0, s1, InterstateEdge("flag"))
    region.add_edge(s0, s2, InterstateEdge("not flag"))

    code = _render_control_flow(region)

    assert "while __state_" in code
    assert "if __state_" in code


def test_unstructured_control_flow_selected() -> None:
    sdfg = _new_sdfg("unstructured_route")
    region = UnstructuredControlFlow("u", sdfg=sdfg)
    s0 = region.add_state("s0", is_start_block=True)

    code = _render_control_flow(region)

    assert "while __state_" in code
    assert "visited.append('s0')" in code


# -----------------------------------------------------------------------------
# 2) State machine generation paths
# -----------------------------------------------------------------------------


def test_state_machine_no_outgoing_edges() -> None:
    sdfg = _new_sdfg("sm_no_out")
    region = ControlFlowRegion("r", sdfg=sdfg)
    region.add_state("terminal", is_start_block=True)

    code = _render_state_machine(region)

    assert "= '__exit_0'" in code


def test_state_machine_single_unconditional_edge() -> None:
    _, region, s0, s1 = _make_region_with_two_states("sm_single_uncond")
    region.add_edge(s0, s1, InterstateEdge())

    code = _render_state_machine(region)

    assert "__state_0 = 's1'" in code


def test_state_machine_single_conditional_edge() -> None:
    _, region, s0, s1 = _make_region_with_two_states("sm_single_cond")
    region.add_edge(s0, s1, InterstateEdge("cond"))

    code = _render_state_machine(region)

    assert "if cond:" in code
    assert "else:" in code
    assert "__state_0 = '__exit_0'" in code


def test_state_machine_multiple_conditional_edges() -> None:
    _, region, s0, s1 = _make_region_with_two_states("sm_multi_cond")
    s2 = region.add_state("s2")
    region.add_edge(s0, s1, InterstateEdge("a"))
    region.add_edge(s0, s2, InterstateEdge("b"))

    code = _render_state_machine(region)

    assert "if a:" in code
    assert "elif b:" in code
    assert "else:" in code


def test_state_machine_conditional_plus_unconditional_else() -> None:
    _, region, s0, s1 = _make_region_with_two_states("sm_cond_else")
    s2 = region.add_state("s2")
    region.add_edge(s0, s1, InterstateEdge("a"))
    region.add_edge(s0, s2, InterstateEdge())

    code = _render_state_machine(region)

    assert "if a:" in code
    assert "else:" in code
    assert "__state_0 = 's2'" in code


def test_state_machine_multiple_unconditional_edges_warning() -> None:
    _, region, s0, s1 = _make_region_with_two_states("sm_warn_uncond")
    s2 = region.add_state("s2")
    region.add_edge(s0, s1, InterstateEdge())
    region.add_edge(s0, s2, InterstateEdge())

    with pytest.warns(UserWarning, match="multiple unconditional edges"):
        code = _render_state_machine(region)

    assert "__state_0 = 's1'" in code


def test_state_machine_unconditional_edge_first_branch_with_len_skew(monkeypatch) -> None:
    _, region, s0, s1 = _make_region_with_two_states("sm_edge_first_len_skew")
    edge = region.add_edge(s0, s1, InterstateEdge(assignments={"x": "1"}))
    original_out_edges = region.out_edges

    def _patched_out_edges(node):
        if node is s0:
            return _LenSkewedEdges([edge], fake_len=2)
        return original_out_edges(node)

    monkeypatch.setattr(region, 'out_edges', _patched_out_edges)

    code = _render_state_machine(region)

    assert "x = 1" in code
    assert "__state_0 = 's1'" in code


def test_state_machine_no_edges_iterated_falls_back_to_exit_label(monkeypatch) -> None:
    _, region, s0, _ = _make_region_with_two_states("sm_empty_iter_len_skew")
    original_out_edges = region.out_edges

    def _patched_out_edges(node):
        if node is s0:
            return _LenSkewedEdges([], fake_len=2)
        return original_out_edges(node)

    monkeypatch.setattr(region, 'out_edges', _patched_out_edges)

    code = _render_state_machine(region)

    assert "__state_0 = '__exit_0'" in code


def test_state_machine_interstate_assignments_emitted() -> None:
    _, region, s0, s1 = _make_region_with_two_states("sm_assignments")
    region.add_edge(s0, s1, InterstateEdge(assignments={"x": "1", "y": "x + 1"}))

    code = _render_state_machine(region)

    assert "x = 1" in code
    assert "y = x + 1" in code


def test_state_machine_executable_correctness() -> None:
    _, region, s0, s1 = _make_region_with_two_states("sm_exec")
    s2 = region.add_state("s2")
    region.add_edge(s0, s1, InterstateEdge("flag", {"result": "1"}))
    region.add_edge(s0, s2, InterstateEdge(assignments={"result": "2"}))

    stream = PythonCodeIOStream()
    _write_state_machine(region, lambda _: "", None, {}, stream)
    code = stream.getvalue()

    namespace = {"flag": True}
    exec(code, namespace, namespace)
    assert namespace["result"] == 1

    namespace = {"flag": False}
    exec(code, namespace, namespace)
    assert namespace["result"] == 2


# -----------------------------------------------------------------------------
# 3) Structured region generation paths
# -----------------------------------------------------------------------------


def test_structured_single_state() -> None:
    sdfg = _new_sdfg("st_single")
    region = ControlFlowRegion("r", sdfg=sdfg)
    region.add_state("only", is_start_block=True)

    code = _render_structured(region)

    assert "visited.append('only')" in code


def test_structured_generate_children_of_filters_non_children() -> None:
    sdfg = _new_sdfg("st_generate_children_filter")
    region = ControlFlowRegion("r", sdfg=sdfg)
    start = region.add_state("start", is_start_block=True)
    parent = ControlFlowBlock("parent", sdfg=sdfg)
    ptree = {start: None, parent: None}

    stream = PythonCodeIOStream()
    _write_structured_region(
        region,
        _dispatch_text,
        None,
        {},
        stream,
        start=start,
        generate_children_of=parent,
        ptree=ptree,
    )

    assert stream.getvalue() == ""


def test_structured_stop_block_skips_processing() -> None:
    sdfg = _new_sdfg("st_stop_skip")
    region = ControlFlowRegion("r", sdfg=sdfg)
    start = region.add_state("start", is_start_block=True)

    stream = PythonCodeIOStream()
    _write_structured_region(region, _dispatch_text, None, {}, stream, start=start, stop=start)

    assert stream.getvalue() == ""


def test_structured_unconditional_edge_first_branch_with_len_skew(monkeypatch) -> None:
    _, region, s0, s1 = _make_region_with_two_states("st_len_skew_uncond")
    edge = region.add_edge(s0, s1, InterstateEdge(assignments={"x": "7"}))
    original_out_edges = region.out_edges

    def _patched_out_edges(node):
        if node is s0:
            return _LenSkewedEdges([edge], fake_len=2)
        return original_out_edges(node)

    monkeypatch.setattr(region, 'out_edges', _patched_out_edges)

    stream = PythonCodeIOStream()
    _write_structured_region(region, _dispatch_text, None, {}, stream, ptree={s0: None, s1: None})
    code = stream.getvalue()

    assert "x = 7" in code


def test_structured_linear_chain() -> None:
    _, region, s0, s1 = _make_region_with_two_states("st_linear")
    region.add_edge(s0, s1, InterstateEdge())

    code = _render_structured(region)

    assert "visited.append('s0')" in code
    assert "visited.append('s1')" in code


def test_structured_single_conditional_edge() -> None:
    _, region, s0, s1 = _make_region_with_two_states("st_one_cond")
    region.add_edge(s0, s1, InterstateEdge("c", {"x": "5"}))

    code = _render_structured(region)

    assert "if c:" in code
    assert "x = 5" in code


def test_structured_conditional_no_assignments_leads_to_pass() -> None:
    _, region, s0, s1 = _make_region_with_two_states("st_cond_pass")
    region.add_edge(s0, s1, InterstateEdge("c"))

    code = _render_structured(region)

    assert "if c:" in code
    assert "pass" in code


def test_structured_unconditional_with_assignments() -> None:
    _, region, s0, s1 = _make_region_with_two_states("st_uncond_assign")
    region.add_edge(s0, s1, InterstateEdge(assignments={"x": "42"}))

    code = _render_structured(region)

    assert "x = 42" in code


def test_structured_multi_edge_if_elif() -> None:
    _, region, s0, s1 = _make_region_with_two_states("st_if_elif")
    s2 = region.add_state("s2")
    region.add_edge(s0, s1, InterstateEdge("a", {"x": "1"}))
    region.add_edge(s0, s2, InterstateEdge("b", {"x": "2"}))

    code = _render_structured(region)

    assert "if a:" in code
    assert "elif b:" in code


def test_structured_multi_edge_else_branch() -> None:
    _, region, s0, s1 = _make_region_with_two_states("st_else")
    s2 = region.add_state("s2")
    region.add_edge(s0, s1, InterstateEdge("a", {"x": "1"}))
    region.add_edge(s0, s2, InterstateEdge(assignments={"x": "3"}))

    code = _render_structured(region)

    assert "if a:" in code
    assert "else:" in code
    assert "x = 3" in code


def test_structured_unconditional_only_multi_edge_case() -> None:
    _, region, s0, s1 = _make_region_with_two_states("st_uncond_only")
    s2 = region.add_state("s2")
    region.add_edge(s0, s1, InterstateEdge(assignments={"x": "1"}))
    region.add_edge(s0, s2, InterstateEdge(assignments={"x": "2"}))

    with pytest.warns(UserWarning, match="multiple unconditional edges"):
        code = _render_structured(region)

    assert "if 1:" in code
    assert "x = 1" in code
    assert "x = 2" in code


def test_structured_empty_else_body_pass() -> None:
    _, region, s0, s1 = _make_region_with_two_states("st_else_pass")
    s2 = region.add_state("s2")
    region.add_edge(s0, s1, InterstateEdge("a", {"x": "1"}))
    region.add_edge(s0, s2, InterstateEdge())

    code = _render_structured(region)

    assert "else:" in code
    assert "pass" in code


def test_structured_multiple_unconditional_warning_path() -> None:
    _, region, s0, s1 = _make_region_with_two_states("st_warn")
    s2 = region.add_state("s2")
    s3 = region.add_state("s3")
    region.add_edge(s0, s1, InterstateEdge("a", {"x": "1"}))
    region.add_edge(s0, s2, InterstateEdge(assignments={"x": "2"}))
    region.add_edge(s0, s3, InterstateEdge(assignments={"x": "3"}))

    with pytest.warns(UserWarning, match="multiple unconditional edges"):
        code = _render_structured(region)

    assert "if a:" in code
    assert "else:" in code


# -----------------------------------------------------------------------------
# 4) Dispatch block coverage
# -----------------------------------------------------------------------------


def test_dispatch_sdfgstate_empty_and_non_empty() -> None:
    sdfg = _new_sdfg("dispatch_states")
    empty_state = sdfg.add_state("empty", is_start_block=True)
    non_empty_state = sdfg.add_state("non_empty")
    sdfg.add_scalar("x", dace.int32)
    non_empty_state.add_access("x")

    stream = PythonCodeIOStream()
    _write_dispatch_block(empty_state, lambda _: "", None, {}, stream)
    _write_dispatch_block(non_empty_state, lambda _: "x = 7", None, {}, stream)
    code = stream.getvalue()

    assert "x = 7" in code


def test_dispatch_break_block() -> None:
    sdfg = _new_sdfg("dispatch_break")
    block = BreakBlock("b", sdfg=sdfg)
    _attach_to_parent(block, sdfg)
    stream = PythonCodeIOStream()

    _write_dispatch_block(block, lambda _: "", None, {}, stream)

    assert "break" in stream.getvalue()


def test_dispatch_continue_block() -> None:
    sdfg = _new_sdfg("dispatch_continue")
    block = ContinueBlock("c", sdfg=sdfg)
    _attach_to_parent(block, sdfg)
    stream = PythonCodeIOStream()

    _write_dispatch_block(block, lambda _: "", None, {}, stream)

    assert "continue" in stream.getvalue()


def test_dispatch_return_block() -> None:
    sdfg = _new_sdfg("dispatch_return")
    block = ReturnBlock("r", sdfg=sdfg)
    _attach_to_parent(block, sdfg)
    stream = PythonCodeIOStream()

    _write_dispatch_block(block, lambda _: "", None, {}, stream)

    assert "return" in stream.getvalue()


def test_dispatch_loop_region() -> None:
    sdfg = _new_sdfg("dispatch_loop")
    loop = LoopRegion("loop", condition_expr="i < 2", loop_var="i", initialize_expr="i = 0", update_expr="i = i + 1", sdfg=sdfg)
    loop.add_state("body", is_start_block=True)
    _attach_to_parent(loop, sdfg)

    stream = PythonCodeIOStream()
    _write_dispatch_block(loop, lambda _: "x = 1", None, {}, stream)

    code = stream.getvalue()
    assert "while (i < 2):" in code
    assert "i = (i + 1)" in code


def test_dispatch_conditional_block() -> None:
    sdfg = _new_sdfg("dispatch_cond")
    cond = ConditionalBlock("cond", sdfg=sdfg)
    br1 = ControlFlowRegion("b1", sdfg=sdfg)
    br1.add_state("s1", is_start_block=True)
    br2 = ControlFlowRegion("b2", sdfg=sdfg)
    br2.add_state("s2", is_start_block=True)
    cond.add_branch("flag", br1)
    cond.add_branch(None, br2)
    _attach_to_parent(cond, sdfg)

    stream = PythonCodeIOStream()
    _write_dispatch_block(cond, lambda _: "x = 1", None, {}, stream)

    code = stream.getvalue()
    assert "if flag:" in code
    assert "else:" in code


def test_dispatch_nested_control_flow_region() -> None:
    sdfg = _new_sdfg("dispatch_nested")
    nested = ControlFlowRegion("nested", sdfg=sdfg)
    nested.add_state("s", is_start_block=True)

    stream = PythonCodeIOStream()
    _write_dispatch_block(nested, lambda _: "x = 1", None, {}, stream)

    assert "x = 1" in stream.getvalue()


def test_dispatch_unknown_type_raises() -> None:
    stream = PythonCodeIOStream()

    with pytest.raises(NotImplementedError):
        _write_dispatch_block(object(), lambda _: "", None, {}, stream)


# -----------------------------------------------------------------------------
# 5) Helper coverage
# -----------------------------------------------------------------------------


def test_unparse_py_expr_string() -> None:
    sdfg = _new_sdfg("unparse_string")
    assert _unparse_py_expr("a + b", sdfg) == "a + b"


def test_unparse_py_expr_list() -> None:
    sdfg = _new_sdfg("unparse_list")
    code_ast = [ast.parse("x = 1").body[0], ast.parse("y = 2").body[0]]

    text = _unparse_py_expr(code_ast, sdfg)

    assert "x = 1" in text
    assert "y = 2" in text
    assert ";" in text


def test_unparse_py_expr_ast_node() -> None:
    sdfg = _new_sdfg("unparse_ast")
    node = ast.parse("a * (b + 1)").body[0].value

    assert _unparse_py_expr(node, sdfg) == "(a * (b + 1))"


def test_unparse_codeblock_none() -> None:
    sdfg = _new_sdfg("cb_none")
    assert _unparse_codeblock(None, sdfg) == ""


def test_unparse_codeblock_python() -> None:
    sdfg = _new_sdfg("cb_py")
    cb = CodeBlock("x + 1", dtypes.Language.Python)

    assert _unparse_codeblock(cb, sdfg) == "(x + 1)"


def test_unparse_codeblock_non_python_with_code_raises() -> None:
    sdfg = _new_sdfg("cb_cpp_error")
    cb = CodeBlock("x + 1", dtypes.Language.CPP)

    with pytest.raises(NotImplementedError):
        _unparse_codeblock(cb, sdfg)


def test_unparse_codeblock_non_python_empty() -> None:
    sdfg = _new_sdfg("cb_cpp_empty")
    cb = CodeBlock("", dtypes.Language.CPP)

    assert _unparse_codeblock(cb, sdfg) == ""


def test_write_interstate_assignments_none() -> None:
    sdfg = _new_sdfg("assign_none")
    region = ControlFlowRegion("r", sdfg=sdfg)
    s0 = region.add_state("s0", is_start_block=True)
    s1 = region.add_state("s1")
    edge = region.add_edge(s0, s1, InterstateEdge())

    stream = PythonCodeIOStream()
    wrote = _write_interstate_assignments(edge, sdfg, stream)

    assert wrote is False
    assert stream.getvalue() == ""


def test_write_interstate_assignments_multiple() -> None:
    sdfg = _new_sdfg("assign_multi")
    region = ControlFlowRegion("r", sdfg=sdfg)
    s0 = region.add_state("s0", is_start_block=True)
    s1 = region.add_state("s1")
    edge = region.add_edge(s0, s1, InterstateEdge(assignments={"x": "1", "y": "x + 1"}))

    stream = PythonCodeIOStream()
    wrote = _write_interstate_assignments(edge, sdfg, stream)
    code = stream.getvalue()

    assert wrote is True
    assert "x = 1" in code
    assert "y = x + 1" in code


def test_is_child_of_true_false_same() -> None:
    sdfg = _new_sdfg("child_of")
    parent = ControlFlowBlock("parent", sdfg=sdfg)
    child = ControlFlowBlock("child", sdfg=sdfg)
    other = ControlFlowBlock("other", sdfg=sdfg)

    ptree = {child: parent, parent: None, other: None}

    assert _is_child_of(child, parent, ptree) is True
    assert _is_child_of(other, parent, ptree) is False
    assert _is_child_of(parent, parent, ptree) is True


def test_state_label_sanitize_whitespace() -> None:
    sdfg = _new_sdfg("label_sanitize")
    node = ControlFlowBlock("state with spaces", sdfg=sdfg)

    assert _state_label(node) == "state_with_spaces"


def test_conditional_block_if_only() -> None:
    sdfg = _new_sdfg("cond_if_only")
    cond = ConditionalBlock("cond", sdfg=sdfg)
    br = ControlFlowRegion("br", sdfg=sdfg)
    br.add_state("s", is_start_block=True)
    cond.add_branch("x > 0", br)
    _attach_to_parent(cond, sdfg)

    stream = PythonCodeIOStream()
    _write_conditional_block(cond, lambda _: "x = 1", None, {}, stream)

    assert "if (x > 0):" in stream.getvalue()


def test_conditional_block_if_else() -> None:
    sdfg = _new_sdfg("cond_if_else")
    cond = ConditionalBlock("cond", sdfg=sdfg)
    br1 = ControlFlowRegion("br1", sdfg=sdfg)
    br1.add_state("s1", is_start_block=True)
    br2 = ControlFlowRegion("br2", sdfg=sdfg)
    br2.add_state("s2", is_start_block=True)
    cond.add_branch("x > 0", br1)
    cond.add_branch(None, br2)
    _attach_to_parent(cond, sdfg)

    stream = PythonCodeIOStream()
    _write_conditional_block(cond, lambda _: "x = 1", None, {}, stream)
    code = stream.getvalue()

    assert "if (x > 0):" in code
    assert "else:" in code


def test_conditional_block_if_elif_else() -> None:
    sdfg = _new_sdfg("cond_if_elif_else")
    cond = ConditionalBlock("cond", sdfg=sdfg)

    br1 = ControlFlowRegion("br1", sdfg=sdfg)
    br1.add_state("s1", is_start_block=True)
    br2 = ControlFlowRegion("br2", sdfg=sdfg)
    br2.add_state("s2", is_start_block=True)
    br3 = ControlFlowRegion("br3", sdfg=sdfg)
    br3.add_state("s3", is_start_block=True)

    cond.add_branch("a", br1)
    cond.add_branch("b", br2)
    cond.add_branch(None, br3)
    _attach_to_parent(cond, sdfg)

    stream = PythonCodeIOStream()
    _write_conditional_block(cond, lambda _: "x = 1", None, {}, stream)
    code = stream.getvalue()

    assert "if a:" in code
    assert "elif b:" in code
    assert "else:" in code


def test_conditional_none_condition_not_final_raises() -> None:
    sdfg = _new_sdfg("cond_none_mid")
    cond = ConditionalBlock("cond", sdfg=sdfg)

    br1 = ControlFlowRegion("br1", sdfg=sdfg)
    br1.add_state("s1", is_start_block=True)
    br2 = ControlFlowRegion("br2", sdfg=sdfg)
    br2.add_state("s2", is_start_block=True)
    br3 = ControlFlowRegion("br3", sdfg=sdfg)
    br3.add_state("s3", is_start_block=True)

    cond.add_branch("a", br1)
    cond.add_branch(None, br2)
    cond.add_branch("b", br3)
    _attach_to_parent(cond, sdfg)

    with pytest.raises(RuntimeError, match="Missing branch condition"):
        _write_conditional_block(cond, lambda _: "", None, {}, PythonCodeIOStream())


def test_conditional_none_condition_at_position_zero_raises() -> None:
    sdfg = _new_sdfg("cond_none_first")
    cond = ConditionalBlock("cond", sdfg=sdfg)

    br1 = ControlFlowRegion("br1", sdfg=sdfg)
    br1.add_state("s1", is_start_block=True)
    cond.add_branch(None, br1)
    _attach_to_parent(cond, sdfg)

    with pytest.raises(RuntimeError, match="Missing branch condition"):
        _write_conditional_block(cond, lambda _: "", None, {}, PythonCodeIOStream())


def test_conditional_empty_body_writes_pass() -> None:
    sdfg = _new_sdfg("cond_empty_body")
    cond = ConditionalBlock("cond", sdfg=sdfg)
    br = ControlFlowRegion("br", sdfg=sdfg)
    br.add_state("s", is_start_block=True)
    cond.add_branch("flag", br)
    _attach_to_parent(cond, sdfg)

    stream = PythonCodeIOStream()
    _write_conditional_block(cond, lambda _: "", None, {}, stream)

    assert "pass" in stream.getvalue()
