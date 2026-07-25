# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Functions for generating Python code for control flow in SDFGs using control flow regions.
"""

import re
import warnings
from typing import TYPE_CHECKING, Callable, Dict, Optional, Set

from dace import dtypes
from dace.codegen.py.prettycode import PythonCodeIOStream
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
    return astutils.unparse(code_ast)


def _unparse_codeblock(cb: Optional[CodeBlock], sdfg: SDFG) -> str:
    """Unparses a CodeBlock to a Python string."""
    if cb is None:
        return ''
    if cb.language == dtypes.Language.Python:
        return _unparse_py_expr(cb.code, sdfg)
    if cb.code:
        raise NotImplementedError(f'CodeBlock with language {cb.language} cannot be unparsed to Python')
    return ''


# ---------------------------------------------------------------------------
# Interstate edge code generation
# ---------------------------------------------------------------------------

def _write_interstate_assignments(edge: Edge[InterstateEdge], sdfg: SDFG,
                                  stream: PythonCodeIOStream) -> bool:
    """
    Writes Python assignment statements for an interstate edge to the stream.

    :return: True if any assignments were written.
    """
    # TODO: C++ backend generates gotos and conditions. Check if we need to do anything else here.
    if not edge.data.assignments:
        return False
    for variable, value in edge.data.assignments.items():
        val_str = _unparse_py_expr(value, sdfg)
        stream.write(f'{variable} = {val_str}', cfg=sdfg, state_id=edge.src.block_id)
    return True


# ---------------------------------------------------------------------------
# Stream-based control flow helpers
# ---------------------------------------------------------------------------

def _write_loop_region(region: LoopRegion, dispatch_state: Callable[[SDFGState], str],
                       codegen: 'DaCePythonCodeGenerator', symbols: Dict[str, dtypes.typeclass],
                       stream: PythonCodeIOStream) -> None:
    """Writes a LoopRegion as Python code to *stream*."""
    sdfg = region.sdfg
    cond = _unparse_codeblock(region.loop_condition, sdfg)

    # TODO: Check that doing a seperation here makes sense, and what happens if
    # only some of the components are present?
    if region.update_statement and region.init_statement and region.loop_variable:
        # This is a C-style for loop
        init = _unparse_codeblock(region.init_statement, sdfg)
        update = _unparse_codeblock(region.update_statement, sdfg)

        stream.write(init, cfg=sdfg, state_id=region.block_id)

        if region.inverted:
            stream.write('while True:', cfg=sdfg, state_id=region.block_id)
            with stream.indented():
                _write_control_flow_region(region, dispatch_state, codegen, symbols, stream)
                if region.update_before_condition:
                    stream.write(update, cfg=sdfg, state_id=region.block_id)
                    stream.write(f'if not ({cond}):', cfg=sdfg, state_id=region.block_id)
                    with stream.indented():
                        stream.write('break', cfg=sdfg, state_id=region.block_id)
                else:
                    stream.write(f'if not ({cond}):', cfg=sdfg, state_id=region.block_id)
                    with stream.indented():
                        stream.write('break', cfg=sdfg, state_id=region.block_id)
                    stream.write(update, cfg=sdfg, state_id=region.block_id)
        else:
            stream.write(f'while {cond}:', cfg=sdfg, state_id=region.block_id)
            with stream.indented():
                _write_control_flow_region(region, dispatch_state, codegen, symbols, stream)
                stream.write(update, cfg=sdfg, state_id=region.block_id)
    else:
        if region.inverted:
            stream.write('while True:', cfg=sdfg, state_id=region.block_id)
            with stream.indented():
                _write_control_flow_region(region, dispatch_state, codegen, symbols, stream)
                stream.write(f'if not ({cond}):', cfg=sdfg, state_id=region.block_id)
                with stream.indented():
                    stream.write('break', cfg=sdfg, state_id=region.block_id)
        else:
            stream.write(f'while {cond}:', cfg=sdfg, state_id=region.block_id)
            with stream.indented():
                if not _write_control_flow_region(region, dispatch_state, codegen, symbols, stream):
                    stream.write('pass', cfg=sdfg, state_id=region.block_id)


def _write_conditional_block(region: ConditionalBlock, dispatch_state: Callable[[SDFGState], str],
                             codegen: 'DaCePythonCodeGenerator', symbols: Dict[str, dtypes.typeclass],
                             stream: PythonCodeIOStream) -> None:
    """Writes a ConditionalBlock as Python if/elif/else code to *stream*."""
    sdfg = region.sdfg

    for i, (cond, body_region) in enumerate(region.branches):
        if cond is not None:
            cond_str = _unparse_py_expr(cond.code, sdfg)
            if i == 0:
                stream.write(f'if {cond_str}:', cfg=sdfg, state_id=region.block_id)
            else:
                stream.write(f'elif {cond_str}:', cfg=sdfg, state_id=region.block_id)
        else:
            if i < len(region.branches) - 1 or i == 0:
                raise RuntimeError('Missing branch condition for non-final conditional branch')
            stream.write('else:', cfg=sdfg, state_id=region.block_id)
        pos_before = stream.tell()
        with stream.indented():
            if not _write_control_flow_region(body_region, dispatch_state, codegen, symbols, stream):
                stream.write('pass', cfg=sdfg, state_id=region.block_id)


# ---------------------------------------------------------------------------
# State-machine fallback for unstructured / irreducible control flow
# ---------------------------------------------------------------------------

def _state_label(node: ControlFlowBlock) -> str:
    """Returns a sanitised label for use as a state-machine value."""
    return re.sub(r'\s+', '_', node.label)


def _write_state_machine(region: AbstractControlFlowRegion,
                         dispatch_state: Callable[[SDFGState], str],
                         codegen: 'DaCePythonCodeGenerator',
                         symbols: Dict[str, dtypes.typeclass],
                         stream: PythonCodeIOStream,
                         start: Optional[ControlFlowBlock] = None,
                         stop: Optional[ControlFlowBlock] = None,
                         generate_children_of: Optional[ControlFlowBlock] = None,
                         ptree: Optional[Dict[ControlFlowBlock, ControlFlowBlock]] = None,
                         visited: Optional[Set[ControlFlowBlock]] = None) -> None:
    """
    Writes a while loop with state-variable dispatch for
    unstructured control flow that cannot be expressed with structured
    Python constructs.
    """
    sdfg = region.sdfg

    start_label = _state_label(region.start_block if start is None else start)
    exit_label = f'__exit_{region.cfg_id}'
    current_state_label = f'__state_{region.cfg_id}'

    stream.write(f'{current_state_label} = {start_label!r}', cfg=sdfg)
    stream.write(f'while {current_state_label} != {exit_label!r}:', cfg=sdfg)

    with stream.indented():
        first = True
        for node in region.nodes():
            label = _state_label(node)
            kw = 'if' if first else 'elif'
            first = False
            stream.write(f'{kw} {current_state_label} == {label!r}:', cfg=sdfg, state_id=node.block_id)

            with stream.indented():
                # Dispatch the block itself
                _write_dispatch_block(node, dispatch_state, codegen, symbols, stream)

                # Generate outgoing edge transitions
                out_edges = region.out_edges(node)
                if len(out_edges) == 0:
                    # If no outgoing edges, this is the last block and we can exit the region.
                    stream.write(f'{current_state_label} = {exit_label!r}', cfg=sdfg, state_id=node.block_id)
                elif len(out_edges) == 1:
                    e = out_edges[0]
                    if not e.data.is_unconditional():
                        cond_str = _unparse_py_expr(e.data.condition.code[0], sdfg)
                        stream.write(f'if {cond_str}:', cfg=sdfg, state_id=node.block_id)
                        with stream.indented():
                            _write_interstate_assignments(e, sdfg, stream)
                            stream.write(f'{current_state_label} = {_state_label(e.dst)!r}', cfg=sdfg, state_id=node.block_id)
                        stream.write('else:', cfg=sdfg, state_id=node.block_id)
                        with stream.indented():
                            stream.write(f'{current_state_label} = {exit_label!r}', cfg=sdfg, state_id=node.block_id)
                    else:
                        _write_interstate_assignments(e, sdfg, stream)
                        stream.write(f'{current_state_label} = {_state_label(e.dst)!r}', cfg=sdfg, state_id=node.block_id)
                else:
                    # Multiple outgoing edges (branching)
                    unconditional_edge = None
                    edge_first = True
                    for e in out_edges:
                        if e.data.is_unconditional():
                            if unconditional_edge is not None:
                                warnings.warn(
                                    f'Unstructured control flow region {region.label} has multiple '
                                    f'unconditional edges leading out of block {node.label}.')
                            else:
                                unconditional_edge = e
                                continue
                        cond_str = _unparse_py_expr(e.data.condition.code[0], sdfg)
                        kw2 = 'if' if edge_first else 'elif'
                        edge_first = False
                        stream.write(f'{kw2} {cond_str}:', cfg=sdfg, state_id=node.block_id)
                        with stream.indented():
                            _write_interstate_assignments(e, sdfg, stream)
                            stream.write(f'{current_state_label} = {_state_label(e.dst)!r}', cfg=sdfg, state_id=node.block_id)

                    if unconditional_edge is not None:
                        if edge_first:
                            _write_interstate_assignments(unconditional_edge, sdfg, stream)
                            stream.write(
                                f'{current_state_label} = {_state_label(unconditional_edge.dst)!r}', cfg=sdfg, state_id=node.block_id)
                        else:
                            stream.write('else:', cfg=sdfg, state_id=node.block_id)
                            with stream.indented():
                                _write_interstate_assignments(unconditional_edge, sdfg, stream)
                                stream.write(
                                    f'{current_state_label} = {_state_label(unconditional_edge.dst)!r}', cfg=sdfg, state_id=node.block_id)
                    else:
                        if not edge_first:
                            stream.write('else:', cfg=sdfg, state_id=node.block_id)
                            with stream.indented():
                                stream.write(f'{current_state_label} = {exit_label!r}', cfg=sdfg, state_id=node.block_id)
                        else:
                            stream.write(f'{current_state_label} = {exit_label!r}', cfg=sdfg, state_id=node.block_id)


def _write_dispatch_block(node: ControlFlowBlock, dispatch_state: Callable[[SDFGState], str],
                          codegen: 'DaCePythonCodeGenerator', symbols: Dict[str, dtypes.typeclass],
                          stream: PythonCodeIOStream) -> None:
    """Writes Python code for a single control-flow block to *stream*."""
    if isinstance(node, SDFGState):
        code = dispatch_state(node)
        if code and code.strip():
            stream.write(code)
    elif isinstance(node, BreakBlock):
        stream.write('break', cfg=node.sdfg, state_id=node.block_id)
    elif isinstance(node, ContinueBlock):
        stream.write('continue', cfg=node.sdfg, state_id=node.block_id)
    elif isinstance(node, ReturnBlock):
        exit_statement = getattr(codegen, 'successful_exit_statement', lambda _: 'return')(node.sdfg)
        stream.write(exit_statement, cfg=node.sdfg, state_id=node.block_id)
    elif isinstance(node, LoopRegion):
        _write_loop_region(node, dispatch_state, codegen, symbols, stream)
    elif isinstance(node, ConditionalBlock):
        _write_conditional_block(node, dispatch_state, codegen, symbols, stream)
    elif isinstance(node, ControlFlowRegion):
        _write_control_flow_region(node, dispatch_state, codegen, symbols, stream)
    else:
        raise NotImplementedError(f'Control flow block {type(node)} not implemented')


# ---------------------------------------------------------------------------
# Structured (reducible) control flow path
# ---------------------------------------------------------------------------

def _is_child_of(node: ControlFlowBlock, parent: ControlFlowBlock,
              parent_tree: Dict[ControlFlowBlock, ControlFlowBlock]) -> bool:
    curnode = node
    while curnode is not None:
        if curnode is parent:
            return True
        curnode = parent_tree.get(curnode)
    return False


def _write_structured_region(region: AbstractControlFlowRegion, dispatch_state: Callable[[SDFGState], str],
                             codegen: 'DaCePythonCodeGenerator', symbols: Dict[str, dtypes.typeclass],
                             stream: PythonCodeIOStream,
                             start: Optional[ControlFlowBlock] = None,
                             stop: Optional[ControlFlowBlock] = None,
                             generate_children_of: Optional[ControlFlowBlock] = None,
                             ptree: Optional[Dict[ControlFlowBlock, ControlFlowBlock]] = None,
                             visited: Optional[Set[ControlFlowBlock]] = None) -> None:
    """
    Writes Python code for a structured (reducible) control-flow region to *stream*.

    Uses sequential block visitation (same order as the C++ backend) and
    generates assignments/conditions inline.  No ``goto`` labels are emitted.
    """
    sdfg = region.sdfg

    if ptree is None:
        ptree = cfg_analysis.block_parent_tree(region, with_loops=False)
    start = start if start is not None else region.start_block
    visited = set() if visited is None else visited

    stack = [start]
    while stack:
        node = stack.pop()
        if generate_children_of is not None and not _is_child_of(node, generate_children_of, ptree):
            continue
        if node in visited or node is stop:
            continue
        visited.add(node)

        # Dispatch the block itself
        _write_dispatch_block(node, dispatch_state, codegen, symbols, stream)

        # Handle outgoing edges
        out_edges = region.out_edges(node)
        if len(out_edges) == 0:
            # TODO: C++ backend generates goto here. Check what we should do in Python
            pass
        elif len(out_edges) == 1:
            e = out_edges[0]
            if not e.data.is_unconditional():
                cond_str = _unparse_py_expr(e.data.condition.code[0], sdfg)
                stream.write(f'if {cond_str}:', cfg=sdfg, state_id=node.block_id)
                pos_before = stream.tell()
                with stream.indented():
                    if not _write_interstate_assignments(e, sdfg, stream):
                        # If no assignments were written, we still need a statement in the body.
                        stream.write('pass', cfg=sdfg, state_id=node.block_id)
            else:
                _write_interstate_assignments(e, sdfg, stream)
            stack.append(e.dst)
        else:
            # Multiple outgoing edges — generate if/elif/else chain
            # TODO: C++ backend generates gotos and conditions. Check if we need to do anything else here.
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
                stream.write(f'{kw} {cond_str}:', cfg=sdfg, state_id=node.block_id)
                pos_before = stream.tell()
                with stream.indented():
                    if not _write_interstate_assignments(e, sdfg, stream):
                        # If no assignments were written, we still need a statement in the body.
                        stream.write('pass', cfg=sdfg, state_id=node.block_id)
                stack.append(e.dst)

            if unconditional_edge is not None:
                if edge_first:
                    _write_interstate_assignments(unconditional_edge, sdfg, stream)
                else:
                    stream.write('else:', cfg=sdfg, state_id=node.block_id)
                    pos_before = stream.tell()
                    with stream.indented():
                        _write_interstate_assignments(unconditional_edge, sdfg, stream)
                        if stream.tell() == pos_before:
                            stream.write('pass', cfg=sdfg, state_id=node.block_id)
                stack.append(unconditional_edge.dst)


# ---------------------------------------------------------------------------
# Internal stream-based entry point
# ---------------------------------------------------------------------------

def _write_control_flow_region(region: AbstractControlFlowRegion,
                               dispatch_state: Callable[[SDFGState], str],
                               codegen: 'DaCePythonCodeGenerator',
                               symbols: Dict[str, dtypes.typeclass],
                               stream: PythonCodeIOStream,
                               start: Optional[ControlFlowBlock] = None,
                               stop: Optional[ControlFlowBlock] = None,
                               generate_children_of: Optional[ControlFlowBlock] = None,
                               ptree: Optional[Dict[ControlFlowBlock, ControlFlowBlock]] = None,
                               visited: Optional[Set[ControlFlowBlock]] = None) -> int:
    """Writes control flow region code to *stream*."""
    contains_irreducible = (any(region.out_degree(node) > 1 for node in region.nodes())
                            or isinstance(region, UnstructuredControlFlow))

    tell = stream.tell()
    if contains_irreducible:
        _write_state_machine(region, dispatch_state, codegen, symbols, stream,
                                 start=start, stop=stop, generate_children_of=generate_children_of,
                                 ptree=ptree, visited=visited)
    else:
        _write_structured_region(region, dispatch_state, codegen, symbols, stream,
                                 start=start, stop=stop, generate_children_of=generate_children_of,
                                 ptree=ptree, visited=visited)
    return stream.tell() - tell  # Return how many characters were written

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def control_flow_region_to_code(region: AbstractControlFlowRegion,
                                dispatch_state: Callable[[SDFGState], str],
                                codegen: 'DaCePythonCodeGenerator',
                                symbols: Dict[str, dtypes.typeclass],
                                stream: PythonCodeIOStream,
                                start: Optional[ControlFlowBlock] = None,
                                stop: Optional[ControlFlowBlock] = None,
                                generate_children_of: Optional[ControlFlowBlock] = None,
                                parent_tree: Optional[Dict[ControlFlowBlock, ControlFlowBlock]] = None,
                                visited: Optional[Set[ControlFlowBlock]] = None,
                                ) -> None:
    """
    Converts a control flow region to Python code with the correct control
    flow expressions.

    :param region:               The control flow region to convert.
    :param dispatch_state:       Callback that generates Python code for a
                                 single SDFG state and returns it as a string.
    :param codegen:              The Python code generator object.
    :param symbols:              A dictionary of symbol names and their types.
    :param stream:               The output stream to write generated code into.
    :param start:                Optional start block override.
    :param stop:                 Optional stop block (exclusive).
    :param generate_children_of: If set, only generate children of this block.
    :param parent_tree:          Pre-computed parent tree (or None).
    :param visited:              Set of already-visited blocks.
    """
    assert isinstance(stream, PythonCodeIOStream)
    _write_control_flow_region(region, dispatch_state, codegen, symbols, stream,
                               start=start, stop=stop, generate_children_of=generate_children_of,
                               ptree=parent_tree, visited=visited)
