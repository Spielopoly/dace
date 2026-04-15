# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Functions for generating Python code for control flow in SDFGs using control flow regions.
"""

import re
import warnings
from typing import TYPE_CHECKING, Callable, Dict, Optional, Set

from dace import dtypes
from dace.frontend.python import astutils
from dace.properties import CodeBlock
from dace.sdfg.analysis import cfg as cfg_analysis
from dace.sdfg.graph import Edge
from dace.sdfg.sdfg import SDFG, InterstateEdge
from dace.sdfg.state import (AbstractControlFlowRegion, BreakBlock, ConditionalBlock, ContinueBlock,
                              ControlFlowBlock, ControlFlowRegion, LoopRegion, ReturnBlock, SDFGState,
                              UnstructuredControlFlow)

if TYPE_CHECKING:
    from dace.codegen.py.framecode import DaCePythonCodeGenerator


def _indent(code: str, spaces: int) -> str:
    """Indents every non-empty line in *code* by *spaces* spaces."""
    if spaces <= 0:
        return code
    prefix = ' ' * spaces
    return '\n'.join((prefix + line) if line.strip() else line for line in code.split('\n'))


def _unparse_py_expr(code_ast, sdfg: SDFG) -> str:
    """
    Converts an AST node (or string expression) to a Python expression string.

    Unlike the C++ ``unparse_interstate_edge``, this produces valid Python and
    does not need C++ casting or type-suffix handling.
    """
    if isinstance(code_ast, str):
        return code_ast
    if isinstance(code_ast, list):
        # CodeBlock.code is a list of AST statements; unparse each and join.
        return '; '.join(_unparse_py_expr(node, sdfg) for node in code_ast)
    # For AST nodes, use the DaCe-aware unparser that handles subscripts etc.
    return astutils.unparse(code_ast)


def _unparse_codeblock(cb: Optional[CodeBlock], sdfg: SDFG) -> str:
    """Unparses a CodeBlock to a Python string."""
    if cb is None:
        return ''
    if cb.language == dtypes.Language.Python:
        return _unparse_py_expr(cb.code, sdfg)
    # Fallback for non-Python code blocks
    return cb.as_string


# ---------------------------------------------------------------------------
# Interstate edge code generation
# ---------------------------------------------------------------------------

def _generate_interstate_assignments(edge: Edge[InterstateEdge], sdfg: SDFG, indent: int) -> str:
    """
    Generates Python assignment statements for an interstate edge.

    :param edge:   The interstate edge.
    :param sdfg:   The SDFG.
    :param indent:  Current indentation level (number of spaces).
    :return:       Python assignment lines.
    """
    prefix = ' ' * indent
    lines = []
    for variable, value in edge.data.assignments.items():
        val_str = _unparse_py_expr(value, sdfg)
        lines.append(f'{prefix}{variable} = {val_str}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Structured control flow helpers
# ---------------------------------------------------------------------------

def _clean_loop_body(body: str) -> str:
    """Strips a trailing bare ``continue`` from a loop body (mirrors C++ version)."""
    stripped = body.rstrip()
    if stripped.endswith('continue'):
        body = stripped[:-len('continue')]
    return body


def _loop_region_to_code(region: LoopRegion, dispatch_state: Callable[[SDFGState], str],
                          codegen: 'DaCePythonCodeGenerator', symbols: Dict[str, dtypes.typeclass],
                          indent: int) -> str:
    """
    Converts a LoopRegion to Python code.

    :param region:          The LoopRegion.
    :param dispatch_state:  Callback to generate code for a given SDFG state.
    :param codegen:         The Python code generator.
    :param symbols:         Symbol table.
    :param indent:          Current indentation level.
    :return:                Python code string.
    """
    sdfg = region.sdfg
    prefix = ' ' * indent
    body_indent = indent + 4

    cond = _unparse_codeblock(region.loop_condition, sdfg)

    body = _clean_loop_body(control_flow_region_to_code(region, dispatch_state, codegen, symbols, indent=body_indent))

    expr = ''

    if region.update_statement and region.init_statement and region.loop_variable:
        init = _unparse_codeblock(region.init_statement, sdfg)
        update = _unparse_codeblock(region.update_statement, sdfg)

        # Emit the initializer before the loop
        expr += f'{prefix}{init}\n'

        if region.inverted:
            # do-while equivalent: while True body, check condition at end
            expr += f'{prefix}while True:\n'
            expr += body
            if region.update_before_condition:
                expr += f'{" " * body_indent}{update}\n'
                expr += f'{" " * body_indent}if not ({cond}):\n'
                expr += f'{" " * (body_indent + 4)}break\n'
            else:
                expr += f'{" " * body_indent}if not ({cond}):\n'
                expr += f'{" " * (body_indent + 4)}break\n'
                expr += f'{" " * body_indent}{update}\n'
        else:
            expr += f'{prefix}while {cond}:\n'
            expr += body
            expr += f'{" " * body_indent}{update}\n'
    else:
        # Simple while loop (no init/update/loop_variable)
        if region.inverted:
            # do-while: execute body once, then check condition
            expr += f'{prefix}while True:\n'
            expr += body
            expr += f'{" " * body_indent}if not ({cond}):\n'
            expr += f'{" " * (body_indent + 4)}break\n'
        else:
            expr += f'{prefix}while {cond}:\n'
            expr += body

    return expr


def _conditional_block_to_code(region: ConditionalBlock, dispatch_state: Callable[[SDFGState], str],
                                codegen: 'DaCePythonCodeGenerator', symbols: Dict[str, dtypes.typeclass],
                                indent: int) -> str:
    """
    Converts a ConditionalBlock to Python if/elif/else code.

    :param region:          The ConditionalBlock.
    :param dispatch_state:  Callback to generate code for a given SDFG state.
    :param codegen:         The Python code generator.
    :param symbols:         Symbol table.
    :param indent:          Current indentation level.
    :return:                Python code string.
    """
    sdfg = region.sdfg
    prefix = ' ' * indent
    body_indent = indent + 4
    expr = ''

    for i, (cond, body_region) in enumerate(region.branches):
        if cond is not None:
            cond_str = _unparse_py_expr(cond.code, sdfg)
            if i == 0:
                expr += f'{prefix}if {cond_str}:\n'
            else:
                expr += f'{prefix}elif {cond_str}:\n'
        else:
            if i < len(region.branches) - 1 or i == 0:
                raise RuntimeError('Missing branch condition for non-final conditional branch')
            expr += f'{prefix}else:\n'
        expr += control_flow_region_to_code(body_region, dispatch_state, codegen, symbols, indent=body_indent)

    return expr


# ---------------------------------------------------------------------------
# State-machine fallback for unstructured / irreducible control flow
# ---------------------------------------------------------------------------

def _state_machine_to_code(region: AbstractControlFlowRegion, dispatch_state: Callable[[SDFGState], str],
                            codegen: 'DaCePythonCodeGenerator', symbols: Dict[str, dtypes.typeclass],
                            indent: int) -> str:
    """
    Generates a ``while True`` + state-variable dispatch loop for
    unstructured control flow that cannot be expressed with structured
    Python constructs.

    Each block gets a string label used as the state-variable value.
    """
    sdfg = region.sdfg
    prefix = ' ' * indent
    body_indent = indent + 4
    case_indent = body_indent + 4

    start_label = _state_label(region.start_block)
    exit_label = f'__exit_{region.cfg_id}'

    expr = f'{prefix}__state_{region.cfg_id} = {start_label!r}\n'
    expr += f'{prefix}while True:\n'

    first = True
    for node in region.nodes():
        label = _state_label(node)
        kw = 'if' if first else 'elif'
        first = False
        expr += f'{" " * body_indent}{kw} __state_{region.cfg_id} == {label!r}:\n'

        # Dispatch the block itself
        expr += _dispatch_block(node, dispatch_state, codegen, symbols, case_indent)

        # Generate outgoing edge transitions
        out_edges = region.out_edges(node)
        if len(out_edges) == 0:
            expr += f'{" " * case_indent}__state_{region.cfg_id} = {exit_label!r}\n'
        elif len(out_edges) == 1:
            e = out_edges[0]
            assigns = _generate_interstate_assignments(e, sdfg, case_indent)
            if assigns:
                if not e.data.is_unconditional():
                    cond_str = _unparse_py_expr(e.data.condition.code[0], sdfg)
                    expr += f'{" " * case_indent}if {cond_str}:\n'
                    assigns = _generate_interstate_assignments(e, sdfg, case_indent + 4)
                    if assigns:
                        expr += assigns + '\n'
                    expr += f'{" " * (case_indent + 4)}__state_{region.cfg_id} = {_state_label(e.dst)!r}\n'
                    expr += f'{" " * case_indent}else:\n'
                    expr += f'{" " * (case_indent + 4)}__state_{region.cfg_id} = {exit_label!r}\n'
                else:
                    expr += assigns + '\n'
                    expr += f'{" " * case_indent}__state_{region.cfg_id} = {_state_label(e.dst)!r}\n'
            else:
                if not e.data.is_unconditional():
                    cond_str = _unparse_py_expr(e.data.condition.code[0], sdfg)
                    expr += f'{" " * case_indent}if {cond_str}:\n'
                    expr += f'{" " * (case_indent + 4)}__state_{region.cfg_id} = {_state_label(e.dst)!r}\n'
                    expr += f'{" " * case_indent}else:\n'
                    expr += f'{" " * (case_indent + 4)}__state_{region.cfg_id} = {exit_label!r}\n'
                else:
                    expr += f'{" " * case_indent}__state_{region.cfg_id} = {_state_label(e.dst)!r}\n'
        else:
            # Multiple outgoing edges (branching)
            unconditional_edge = None
            edge_first = True
            for e in out_edges:
                if e.data.is_unconditional():
                    if unconditional_edge is not None:
                        warnings.warn(
                            f'Unstructured control flow region {region.label} has multiple unconditional edges '
                            f'leading out of block {node.label}.')
                    else:
                        unconditional_edge = e
                        continue
                cond_str = _unparse_py_expr(e.data.condition.code[0], sdfg)
                kw2 = 'if' if edge_first else 'elif'
                edge_first = False
                expr += f'{" " * case_indent}{kw2} {cond_str}:\n'
                assigns = _generate_interstate_assignments(e, sdfg, case_indent + 4)
                if assigns:
                    expr += assigns + '\n'
                expr += f'{" " * (case_indent + 4)}__state_{region.cfg_id} = {_state_label(e.dst)!r}\n'

            if unconditional_edge is not None:
                if edge_first:
                    # All edges were unconditional—shouldn't happen but handle it.
                    assigns = _generate_interstate_assignments(unconditional_edge, sdfg, case_indent)
                    if assigns:
                        expr += assigns + '\n'
                    expr += f'{" " * case_indent}__state_{region.cfg_id} = {_state_label(unconditional_edge.dst)!r}\n'
                else:
                    expr += f'{" " * case_indent}else:\n'
                    assigns = _generate_interstate_assignments(unconditional_edge, sdfg, case_indent + 4)
                    if assigns:
                        expr += assigns + '\n'
                    expr += f'{" " * (case_indent + 4)}__state_{region.cfg_id} = {_state_label(unconditional_edge.dst)!r}\n'
            else:
                # No unconditional edge — exit if no condition matched
                if not edge_first:
                    expr += f'{" " * case_indent}else:\n'
                    expr += f'{" " * (case_indent + 4)}__state_{region.cfg_id} = {exit_label!r}\n'
                else:
                    expr += f'{" " * case_indent}__state_{region.cfg_id} = {exit_label!r}\n'

    # Exit case
    expr += f'{" " * body_indent}{"elif" if not first else "if"} __state_{region.cfg_id} == {exit_label!r}:\n'
    expr += f'{" " * case_indent}break\n'

    return expr


def _state_label(node: ControlFlowBlock) -> str:
    """Returns a sanitised label for use as a state-machine value."""
    return re.sub(r'\s+', '_', node.label)


def _dispatch_block(node: ControlFlowBlock, dispatch_state: Callable[[SDFGState], str],
                     codegen: 'DaCePythonCodeGenerator', symbols: Dict[str, dtypes.typeclass],
                     indent: int) -> str:
    """Generates Python code for a single control-flow block (non-edge part)."""
    prefix = ' ' * indent
    expr = ''

    if isinstance(node, SDFGState):
        code = dispatch_state(node)
        if code and code.strip():
            expr += _indent(code, indent) + '\n'
    elif isinstance(node, BreakBlock):
        expr += f'{prefix}break\n'
    elif isinstance(node, ContinueBlock):
        expr += f'{prefix}continue\n'
    elif isinstance(node, ReturnBlock):
        expr += f'{prefix}return\n'
    elif isinstance(node, LoopRegion):
        expr += _loop_region_to_code(node, dispatch_state, codegen, symbols, indent)
    elif isinstance(node, ConditionalBlock):
        expr += _conditional_block_to_code(node, dispatch_state, codegen, symbols, indent)
    elif isinstance(node, ControlFlowRegion):
        expr += control_flow_region_to_code(node, dispatch_state, codegen, symbols, indent=indent)
    else:
        raise NotImplementedError(f'Control flow block {type(node)} not implemented')

    return expr


# ---------------------------------------------------------------------------
# Structured (reducible) control flow path
# ---------------------------------------------------------------------------

def _child_of(node: ControlFlowBlock, parent: ControlFlowBlock,
              ptree: Dict[ControlFlowBlock, ControlFlowBlock]) -> bool:
    curnode = node
    while curnode is not None:
        if curnode is parent:
            return True
        curnode = ptree.get(curnode)
    return False


def _structured_region_to_code(region: AbstractControlFlowRegion, dispatch_state: Callable[[SDFGState], str],
                                codegen: 'DaCePythonCodeGenerator', symbols: Dict[str, dtypes.typeclass],
                                indent: int, start: Optional[ControlFlowBlock] = None,
                                stop: Optional[ControlFlowBlock] = None,
                                generate_children_of: Optional[ControlFlowBlock] = None,
                                ptree: Optional[Dict[ControlFlowBlock, ControlFlowBlock]] = None,
                                visited: Optional[Set[ControlFlowBlock]] = None) -> str:
    """
    Generates Python code for a structured (reducible) control-flow region.

    Uses sequential block visitation (same order as the C++ backend) and
    generates assignments/conditions inline.  No ``goto`` labels are emitted.
    """
    sdfg = region.sdfg
    prefix = ' ' * indent

    if ptree is None:
        ptree = cfg_analysis.block_parent_tree(region, with_loops=False)
    start = start if start is not None else region.start_block
    visited = set() if visited is None else visited

    expr = ''
    stack = [start]
    while stack:
        node = stack.pop()
        if generate_children_of is not None and not _child_of(node, generate_children_of, ptree):
            continue
        if node in visited or node is stop:
            continue
        visited.add(node)

        # Dispatch the block itself
        expr += _dispatch_block(node, dispatch_state, codegen, symbols, indent)

        # Handle outgoing edges
        out_edges = region.out_edges(node)
        if len(out_edges) == 0:
            # Terminal block — nothing more to do.
            pass
        elif len(out_edges) == 1:
            e = out_edges[0]
            assigns = _generate_interstate_assignments(e, sdfg, indent)
            if not e.data.is_unconditional():
                cond_str = _unparse_py_expr(e.data.condition.code[0], sdfg)
                expr += f'{prefix}if {cond_str}:\n'
                a = _generate_interstate_assignments(e, sdfg, indent + 4)
                if a:
                    expr += a + '\n'
                # For a single conditional edge, fall through if condition not met
                # (no explicit else needed since subsequent code is the fallthrough).
            else:
                if assigns:
                    expr += assigns + '\n'
            stack.append(e.dst)
        else:
            # Multiple outgoing edges — generate if/elif/else chain
            unconditional_edge = None
            edge_first = True
            for e in out_edges:
                if e.data.is_unconditional():
                    if unconditional_edge is not None:
                        warnings.warn(
                            f'Structured control flow region {region.label} has multiple unconditional edges '
                            f'leading out of block {node.label}.')
                    else:
                        unconditional_edge = e
                        continue
                cond_str = _unparse_py_expr(e.data.condition.code[0], sdfg)
                kw = 'if' if edge_first else 'elif'
                edge_first = False
                expr += f'{prefix}{kw} {cond_str}:\n'
                a = _generate_interstate_assignments(e, sdfg, indent + 4)
                if a:
                    expr += a + '\n'
                stack.append(e.dst)

            if unconditional_edge is not None:
                if edge_first:
                    # Defensive: all edges were unconditional — shouldn't normally happen
                    # in structured control flow, but handle it for robustness / future-proofing.
                    assigns = _generate_interstate_assignments(unconditional_edge, sdfg, indent)
                    if assigns:
                        expr += assigns + '\n'
                else:
                    expr += f'{prefix}else:\n'
                    a = _generate_interstate_assignments(unconditional_edge, sdfg, indent + 4)
                    if a:
                        expr += a + '\n'
                stack.append(unconditional_edge.dst)

    return expr


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def control_flow_region_to_code(region: AbstractControlFlowRegion,
                                dispatch_state: Callable[[SDFGState], str],
                                codegen: 'DaCePythonCodeGenerator',
                                symbols: Dict[str, dtypes.typeclass],
                                start: Optional[ControlFlowBlock] = None,
                                stop: Optional[ControlFlowBlock] = None,
                                generate_children_of: Optional[ControlFlowBlock] = None,
                                ptree: Optional[Dict[ControlFlowBlock, ControlFlowBlock]] = None,
                                visited: Optional[Set[ControlFlowBlock]] = None,
                                indent: int = 0) -> str:
    """
    Converts a control flow region to Python code with the correct control
    flow expressions.

    :param region:               The control flow region to convert.
    :param dispatch_state:       Callback that generates Python code for a
                                 single SDFG state and returns it as a string.
    :param codegen:              The Python code generator object.
    :param symbols:              A dictionary of symbol names and their types.
    :param start:                Optional start block override.
    :param stop:                 Optional stop block (exclusive).
    :param generate_children_of: If set, only generate children of this block.
    :param ptree:                Pre-computed parent tree (or None).
    :param visited:              Set of already-visited blocks.
    :param indent:               Current indentation in number of spaces.
    :return:                     Python code string.
    """
    # Detect whether the region contains irreducible / unstructured control flow.
    contains_irreducible = (any(region.out_degree(node) > 1 for node in region.nodes())
                            or isinstance(region, UnstructuredControlFlow))

    if contains_irreducible:
        return _state_machine_to_code(region, dispatch_state, codegen, symbols, indent)
    else:
        return _structured_region_to_code(region, dispatch_state, codegen, symbols, indent,
                                          start=start, stop=stop, generate_children_of=generate_children_of,
                                          ptree=ptree, visited=visited)
