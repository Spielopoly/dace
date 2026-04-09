"""
RemoveIntermediateTransient transformation.

Removes unnecessary intermediate transient access nodes in the pattern::

    predecessor → AccessNode(transient) → MapExit

where after the MapExit all paths must reach an AccessNode before reaching
a CodeNode (Tasklet or NestedSDFG).
"""

import copy
from typing import Set, Union, Optional

from dace import data, properties, SDFG
from dace.sdfg import nodes, SDFGState
from dace.sdfg import utils as sdutil
from dace.transformation import transformation as pm
from dace import memlet as mm
from dace import subsets


class RemoveIntermediateTransient(pm.SingleStateTransformation):
    """Remove an intermediate transient access node between a predecessor
    node and a MapExit.

    Pattern: predecessor → AccessNode(transient) → MapExit

    The access node is removed and the predecessor is connected directly
    to the MapExit with a composed memlet.

    Preconditions:
    - The access node's data descriptor must be transient.
    - The access node's data descriptor must not be a View, Reference, or
      StructureView.
    - The access node must have exactly one incoming and one outgoing edge.
    - No other access node in the SDFG may reference the same data.
    - The data must not appear in any interstate edge's free symbols.
    - All paths from the MapExit must reach an AccessNode before reaching
      a CodeNode (Tasklet or NestedSDFG).
    """

    access_node = pm.PatternNode(nodes.AccessNode)
    map_exit = pm.PatternNode(nodes.MapExit)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.access_node, cls.map_exit)]

    @staticmethod
    def _all_paths_reach_access_node(state: SDFGState, start_node) -> bool:
        """Check that all paths from successors of *start_node* reach an
        AccessNode before reaching a CodeNode (Tasklet or NestedSDFG).

        Uses DFS on the state graph.
        """
        visited: Set = set()
        stack = [start_node]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for edge in state.out_edges(node):
                succ = edge.dst
                if isinstance(succ, nodes.AccessNode):
                    continue  # Path satisfied — stop exploring
                if isinstance(succ, (nodes.Tasklet, nodes.NestedSDFG)):
                    return False  # CodeNode reached before AccessNode
                # Structural node (MapEntry, MapExit, etc.) — keep traversing
                stack.append(succ)
        return True

    def can_be_applied(self, graph: SDFGState, expr_index: int,
                       sdfg: SDFG, permissive: bool = False) -> bool:
        access_node = self.access_node
        map_exit = self.map_exit
        desc = sdfg.arrays.get(access_node.data)

        if desc is None:
            return False

        # Must be transient
        if not desc.transient:
            return False

        # Must not be a View, Reference, or StructureView
        if isinstance(desc, (data.View, data.Reference, data.StructureView)):
            return False

        # Must have exactly one incoming and one outgoing edge
        if graph.in_degree(access_node) != 1:
            return False
        if graph.out_degree(access_node) != 1:
            return False

        # No other access node in the SDFG may reference the same data
        occurrences = 0
        for state in sdfg.states():
            for node in state.data_nodes():
                if node.data == access_node.data:
                    occurrences += 1
                    if occurrences > 1:
                        return False

        # Data must not appear in any interstate edge's free symbols
        for isedge in sdfg.all_interstate_edges():
            if access_node.data in isedge.data.free_symbols:
                return False

        # All paths from MapExit must reach an AccessNode before a CodeNode
        if not self._all_paths_reach_access_node(graph, map_exit):
            return False

        # The out_edge must carry data (not just a dependency edge)
        out_edge = graph.edges_between(access_node, map_exit)[0]
        if out_edge.data.data is None:
            return False

        # out_edge must reference an outer array, not the intermediate itself
        if out_edge.data.data == access_node.data:
            return False

        # Validate memlet composition for AccessNode predecessors with non-scalar intermediates
        in_edge = graph.in_edges(access_node)[0]
        if isinstance(in_edge.src, nodes.AccessNode):
            in_other = in_edge.data.other_subset
            in_subset = in_edge.data.subset
            out_other = out_edge.data.other_subset
            if in_other is not None and in_subset is not None and out_other is not None:
                try:
                    import copy as _copy
                    relative = _copy.deepcopy(out_other)
                    relative.offset(in_subset, negative=True)
                    in_other.compose(relative)
                except (ValueError, TypeError, NotImplementedError):
                    return False

        return True

    def apply(self, graph: SDFGState, sdfg: SDFG):
        access_node = self.access_node
        map_exit = self.map_exit

        in_edge = graph.in_edges(access_node)[0]
        out_edge = graph.edges_between(access_node, map_exit)[0]

        # Build the new memlet that bypasses the intermediate access node.
        # out_edge.data.data is the outer array name (e.g., 'c')
        # out_edge.data.subset is the subset of that array (e.g., 'i')
        # out_edge.data.other_subset references the intermediate (being removed)
        out_data = out_edge.data.data
        out_subset = (copy.deepcopy(out_edge.data.subset)
                      if out_edge.data.subset is not None else None)

        # Determine the source-side subset (other_subset in the new memlet)
        new_other_subset = None
        if isinstance(in_edge.src, nodes.AccessNode):
            # Predecessor is an AccessNode: compose subsets through intermediate
            in_other = in_edge.data.other_subset
            in_subset = in_edge.data.subset
            out_other = out_edge.data.other_subset

            if in_other is not None and in_subset is not None and out_other is not None:
                relative = copy.deepcopy(out_other)
                relative.offset(in_subset, negative=True)
                new_other_subset = in_other.compose(relative)
            elif in_other is not None:
                new_other_subset = copy.deepcopy(in_other)

        # Merge WCR from both edges
        wcr = out_edge.data.wcr
        wcr_nonatomic = out_edge.data.wcr_nonatomic
        if in_edge.data.wcr is not None:
            if wcr is None:
                wcr = in_edge.data.wcr
                wcr_nonatomic = in_edge.data.wcr_nonatomic

        # Construct new memlet
        new_memlet = mm.Memlet(data=out_data,
                               subset=out_subset,
                               other_subset=new_other_subset,
                               volume=out_edge.data.volume,
                               dynamic=out_edge.data.dynamic,
                               wcr=wcr,
                               wcr_nonatomic=wcr_nonatomic)

        # Add new edge bypassing the access node
        graph.add_edge(in_edge.src, in_edge.src_conn,
                       map_exit, out_edge.dst_conn, new_memlet)

        # Remove the access node
        graph.remove_node(access_node)

        # Remove the data descriptor if no longer referenced
        try:
            sdfg.remove_data(access_node.data)
        except ValueError:
            pass
