# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
SDFG construction utility functions.

Helpers for copying state contents, flattening conditional blocks, and related
graph-manipulation tasks used by cuTile transformations and other passes.
"""
import copy
from typing import Dict, Tuple

from sympy import Function

import dace
import dace.symbolic
import dace.sdfg.utils as sdutil
import dace.sdfg.tasklet_utils as tutil
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.symbolic import pystr_to_symbolic
from dace.properties import CodeBlock
from dace import InterstateEdge


def copy_state_contents(old_state: dace.SDFGState,
                        new_state: dace.SDFGState) -> Dict[dace.nodes.Node, dace.nodes.Node]:
    """
    Deep-copy all nodes and edges from one SDFG state into another.

    Parameters
    ----------
    old_state : dace.SDFGState
        The source SDFG state to copy from.
    new_state : dace.SDFGState
        The destination SDFG state to copy into.

    Returns
    -------
    dict
        A mapping from original nodes in *old_state* to their deep-copied
        counterparts in *new_state*.
    """
    node_map: Dict[dace.nodes.Node, dace.nodes.Node] = {}

    for n in old_state.nodes():
        c_n = copy.deepcopy(n)
        node_map[n] = c_n
        new_state.add_node(c_n)

    for e in old_state.edges():
        c_src = node_map[e.src]
        c_dst = node_map[e.dst]
        new_state.add_edge(c_src, e.src_conn, c_dst, e.dst_conn, copy.deepcopy(e.data))

    return node_map


def move_branch_cfg_up_discard_conditions(if_block: ConditionalBlock,
                                          body_to_take: ControlFlowRegion):
    """
    Move a branch of a conditional block up in the CFG, discarding the
    conditional check and other branches.

    This operation:
      - Copies all nodes and edges from the selected branch (*body_to_take*)
        into the parent graph of the conditional.
      - Reconnects all incoming edges of the original conditional block to the
        start of the selected branch.
      - Connects all outgoing edges of the original conditional block to the
        end of the selected branch.
      - Removes the original conditional block from the graph.

    Parameters
    ----------
    if_block : ConditionalBlock
        The conditional block whose branch is to be promoted.
    body_to_take : ControlFlowRegion
        The branch of *if_block* to keep.  Must be one of its branches.
    """
    bodies = {b for _, b in if_block.branches}
    assert body_to_take in bodies
    assert isinstance(if_block, ConditionalBlock)

    graph = if_block.parent_graph

    node_map: Dict = {}
    new_start_block = None
    new_end_block = None

    for node in body_to_take.nodes():
        copynode = copy.deepcopy(node)
        node_map[node] = copynode
        start_block_case = (body_to_take.start_block == node) and (graph.start_block == if_block)
        if body_to_take.start_block == node:
            assert new_start_block is None
            new_start_block = copynode
        if body_to_take.out_degree(node) == 0:
            assert new_end_block is None
            new_end_block = copynode
        graph.add_node(copynode, is_start_block=start_block_case)

    for edge in body_to_take.edges():
        src = node_map[edge.src]
        dst = node_map[edge.dst]
        graph.add_edge(src, dst, copy.deepcopy(edge.data))

    for ie in graph.in_edges(if_block):
        graph.add_edge(ie.src, new_start_block, copy.deepcopy(ie.data))
    for oe in graph.out_edges(if_block):
        graph.add_edge(new_end_block, oe.dst, copy.deepcopy(oe.data))

    graph.remove_node(if_block)


def extract_condition_var_and_assignment(parent_graph: ControlFlowRegion,
                                        conditional: ConditionalBlock) -> Tuple[str, str]:
    """
    Extract the condition variable and its resolved assignment from a ConditionalBlock.

    Performs a reverse BFS from *conditional* through *parent_graph* to find
    interstate-edge assignments for the condition's free symbols, substitutes
    them into the condition via ``replace_dict``, and returns the condition
    variable name together with the resolved condition string.

    Parameters
    ----------
    parent_graph : ControlFlowRegion
        The control-flow region that contains *conditional*.
    conditional : ConditionalBlock
        The conditional block whose condition is to be resolved.

    Returns
    -------
    tuple of (str, str)
        ``(cond_var, resolved_condition_string)``
    """
    non_none_conds = [cond for cond, _ in conditional.branches if cond is not None]
    assert len(non_none_conds) == 1
    cond = non_none_conds.pop()
    cond_code_str = cond.as_string
    cond_code_symexpr = pystr_to_symbolic(cond_code_str, simplify=False)

    # Find values assigned to the symbols
    free_syms = {str(s).strip() for s in cond_code_symexpr.free_symbols if str(s) in parent_graph.sdfg.symbols}
    sym_val_map = dict()
    nodes_to_check = {conditional}
    visited = set()

    # Do reverse BFS from the sink node to get all possible interstate assignments
    while nodes_to_check:
        node_to_check = nodes_to_check.pop()
        visited.add(node_to_check)
        ies = {ie for ie in parent_graph.in_edges(node_to_check)}
        for ie in ies:
            for k, v in ie.data.assignments.items():
                if k in free_syms and k not in sym_val_map:
                    sym_val_map[k] = v
        nodes_to_check = nodes_to_check.union({ie.src for ie in ies} - visited)

    # If 1 free symbol, easy it means it is condition variable,
    # otherwise get the left most
    if len(cond_code_symexpr.free_symbols) == 1:
        cond_var = str(next(iter(cond_code_symexpr.free_symbols)))
    else:
        tokens = tutil.token_split_variable_names(cond_code_str)
        if not tokens:
            raise ValueError(f"No valid identifier found in condition expression: {cond_code_str}")
        cond_var = tokens.pop()

    # If the sym_map has any functions, then we need to drop, e.g. array access
    new_sym_val_map = dict()
    for k, v in sym_val_map.items():
        vv = dace.symbolic.SymExpr(v)
        funcs = [e for e in vv.atoms(Function)]
        if len(funcs) == 0:
            new_sym_val_map[str(k)] = str(v)
    sym_val_map = new_sym_val_map

    # Substitute using replace dict to avoid problems
    conditional.replace_dict(sym_val_map)

    new_conds = {c.as_string for c, b in conditional.branches if c is not None}
    new_cond = new_conds.pop()

    return cond_var, new_cond


def _split_branches(parent_graph: ControlFlowRegion,
                   if_block: ConditionalBlock) -> Tuple['ConditionalBlock', 'ConditionalBlock']:
    """
    Split a two-branch ConditionalBlock into two sequential single-branch blocks.

    The original *if_block* keeps its first (true) branch.  A new
    ``ConditionalBlock`` is created for the second (false) branch with a
    negated condition (``(resolved_cond) == 0``).  The two blocks are
    connected by an ``InterstateEdge`` in *parent_graph*.

    Parameters
    ----------
    parent_graph : ControlFlowRegion
        The CFR that contains *if_block*.
    if_block : ConditionalBlock
        A conditional block with exactly two branches.

    Returns
    -------
    tuple of (ConditionalBlock, ConditionalBlock)
        ``(original_if_block, new_negated_if_block)``
    """
    # Create two new conditional blocks with single branches each
    tup0 = if_block.branches[0]
    tup1 = if_block.branches[1]
    (cond0, body0) = tup0[0], tup0[1]
    (cond1, body1) = tup1[0], tup1[1]

    cond = cond0 if cond0 is not None else cond1
    body = body0 if cond0 is None else body1

    if_block.remove_branch(body)
    assert cond.language == dace.dtypes.Language.Python

    if_out_edges = parent_graph.out_edges(if_block)

    new_if_block = ConditionalBlock(label=f"{if_block.label}_negated", sdfg=parent_graph.sdfg, parent=parent_graph)

    # Get the condition assignment of the if-block to copy the symbol type
    # We add its negation to the new branch (e.g. expr == 0 instead of expr == 1 which is the usual one)
    _, cond_assignment = extract_condition_var_and_assignment(parent_graph, if_block)

    new_if_block.add_branch(condition=CodeBlock(f"({cond_assignment}) == 0"), branch=body)

    parent_graph.add_node(new_if_block)

    for oe in if_out_edges:
        parent_graph.remove_edge(oe)
        parent_graph.add_edge(new_if_block, oe.dst, copy.deepcopy(oe.data))

    # Do not use negation assignments={f"{negated_name}": f"not ({cond.as_string})"}
    # Creates issue when simplifying with sympy
    parent_graph.add_edge(if_block, new_if_block, InterstateEdge())

    parent_graph.reset_cfg_list()

    return if_block, new_if_block


def duplicate_condition_across_top_level_nodes(parent_graph: ControlFlowRegion,
                                               conditional: ConditionalBlock) -> bool:
    """
    Duplicate a single-branch conditional's condition across each top-level node.

    If the branch body is a line graph (each node has in/out degree <= 1),
    each node after the first is wrapped in its own ``ConditionalBlock`` with
    a deep-copy of the original condition.  Interstate edge assignments are
    preserved as pre-/post-assignment states.

    Parameters
    ----------
    parent_graph : ControlFlowRegion
        The CFR that contains *conditional*.
    conditional : ConditionalBlock
        A single-branch conditional block.

    Returns
    -------
    bool
        ``True`` if duplication was applied, ``False`` otherwise.
    """
    applied = False
    if len(conditional.branches) == 1:
        cond, body = conditional.branches[0]

        nodes = [n for n in body.bfs_nodes()]
        if len(nodes) <= 1:
            return False

        in_degree_leq_one = all({body.in_degree(n) <= 1 for n in nodes})
        out_degree_leq_one = all({body.out_degree(n) <= 1 for n in nodes})
        edges = body.edges()
        # Can support if they are not fully empty
        all_edges_empty = all({e.data.assignments == dict() for e in edges})

        if in_degree_leq_one and out_degree_leq_one:
            # Put all nodes into their own if condition
            node_to_add_after = conditional
            # First node gets to stay
            for ci, node in enumerate(nodes[1:]):
                # Get edge data to copy

                if not all_edges_empty:
                    cfg_in_edges = body.in_edges(node)
                    assert len(cfg_in_edges) <= 1, f"{cfg_in_edges}"
                    cfg_in_edge = cfg_in_edges[0] if len(cfg_in_edges) == 1 else None
                    cfg_out_edges = body.out_edges(node)
                    assert len(cfg_out_edges) <= 1, f"{cfg_out_edges}"
                    cfg_out_edge = cfg_out_edges[0] if len(cfg_out_edges) == 1 else None

                body.remove_node(node)

                is_empty_state = isinstance(node, dace.SDFGState) and len(node.nodes()) == 0
                # If state is empty do not wrap it in a conditional region
                if not is_empty_state:
                    copy_conditional = ConditionalBlock(label=conditional.label + f"_v_{ci}",
                                                       sdfg=conditional.sdfg,
                                                       parent=parent_graph)

                    cfg = ControlFlowRegion(label=conditional.label + f"_v_{ci}_body",
                                            sdfg=conditional.sdfg,
                                            parent=copy_conditional)
                    cfg.add_node(copy.deepcopy(node))
                    copy_conditional.add_branch(condition=copy.deepcopy(cond), branch=cfg)
                else:
                    copy_conditional = copy.deepcopy(node)

                parent_graph.add_node(copy_conditional, False, False)

                for oe in parent_graph.out_edges(node_to_add_after):
                    parent_graph.remove_edge(oe)
                    parent_graph.add_edge(copy_conditional, oe.dst, copy.deepcopy(oe.data))

                # Find the edge between the
                parent_graph.add_edge(node_to_add_after, copy_conditional, InterstateEdge())

                if not all_edges_empty:
                    if cfg_in_edge is not None:
                        pre_assign = parent_graph.add_state_before(
                            state=copy_conditional,
                            label=f"pre_assign_{copy_conditional.label}",
                            is_start_block=parent_graph.start_block == copy_conditional,
                            assignments=cfg_in_edge.data.assignments)
                    if cfg_out_edge is not None:
                        post_assign = parent_graph.add_state_after(state=copy_conditional,
                                                                   label=f"post_assign_{copy_conditional.label}",
                                                                   is_start_block=False,
                                                                   assignments=cfg_out_edge.data.assignments)
                        node_to_add_after = post_assign
                    else:
                        node_to_add_after = copy_conditional
                else:
                    node_to_add_after = copy_conditional
            applied = True

            parent_graph.sdfg.reset_cfg_list()
            sdutil.set_nested_sdfg_parent_references(parent_graph.sdfg)
    return applied