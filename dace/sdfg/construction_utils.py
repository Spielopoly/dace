# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""
SDFG construction utility functions.

Helpers for copying state contents, flattening conditional blocks, and related
graph-manipulation tasks used by cuTile transformations and other passes.
"""
import copy
from typing import Dict

import dace
import dace.sdfg.utils as sdutil
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion


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
