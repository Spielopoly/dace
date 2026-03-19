# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Contains classes that implement trivial access node elimination. """

import copy

from dace import data, symbolic
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from dace.properties import make_properties
from dace.dtypes import AllocationLifetime, StorageType


@make_properties
class TrivialAccessNodeElimination(transformation.SingleStateTransformation):
    """Implements trivial AccessNode elimination.

    Removes transient access nodes that are only written to once and read from once
    """

    read = transformation.PatternNode(nodes.AccessNode)
    read_map = transformation.PatternNode(nodes.MapEntry)
    read_tasklet = transformation.PatternNode(nodes.Tasklet)
    access = transformation.PatternNode(nodes.AccessNode)
    write = transformation.PatternNode(nodes.AccessNode)
    write_map = transformation.PatternNode(nodes.MapExit)
    write_tasklet = transformation.PatternNode(nodes.Tasklet)

    @classmethod
    def expressions(cls):
        return [
            sdutil.node_path_graph(cls.read, cls.access, cls.write),
            sdutil.node_path_graph(cls.read_map, cls.access, cls.write),
            sdutil.node_path_graph(cls.read_tasklet, cls.access, cls.write),
            sdutil.node_path_graph(cls.read, cls.access, cls.write_map),
            sdutil.node_path_graph(cls.read_map, cls.access, cls.write_map),
            sdutil.node_path_graph(cls.read_tasklet, cls.access, cls.write_map),
            sdutil.node_path_graph(cls.read, cls.access, cls.write_tasklet),
            sdutil.node_path_graph(cls.read_map, cls.access, cls.write_tasklet),
            sdutil.node_path_graph(cls.read_tasklet, cls.access, cls.write_tasklet),
        ]

    def _source_for_expr(self, expr_index):
        if expr_index in (1, 4, 7):
            # Source is map entry
            return self.read_map
        if expr_index in (2, 5, 8):
            # Source is tasklet
            return self.read_tasklet
        # Source is access node
        return self.read

    def _dest_for_expr(self, expr_index):
        if expr_index in (3, 4, 5):
            # Destination is map exit
            return self.write_map
        if expr_index in (6, 7, 8):
            # Destination is tasklet
            return self.write_tasklet
        # Destination is access node
        return self.write

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        read = self._source_for_expr(expr_index)
        access: nodes.AccessNode = self.access
        write = self._dest_for_expr(expr_index)

        if graph.in_degree(access) != 1 or graph.out_degree(access) != 1:
            return False
        
        # Check that access node is transient
        access_desc = access.desc(sdfg)
        if not access_desc.transient:
            return False
        # Lifetime must be scope
        if access_desc.lifetime != AllocationLifetime.Scope:
            return False
        # Default storage location must be used
        if access_desc.storage != StorageType.Default:
            return False

        in_edge = graph.edges_between(read, access)[0]
        out_edge = graph.edges_between(access, write)[0]

        if out_edge.data.wcr:
            return False
        if in_edge.data.subset is None or out_edge.data.subset is None:
            return False

        in_subset_expr = in_edge.data.subset.num_elements()
        in_subset_expr_exact = in_edge.data.subset.num_elements_exact()
        out_subset_expr = out_edge.data.subset.num_elements()
        out_subset_expr_exact = out_edge.data.subset.num_elements_exact()
        if (in_subset_expr != out_subset_expr and symbolic.inequal_symbols(in_subset_expr_exact, out_subset_expr_exact)):
            return False

        access_desc = access.desc(sdfg)
        if not access_desc.transient:
            return False
        if isinstance(access_desc, data.Stream):
            return False
        if isinstance(access_desc, data.View):
            return False

        return True

    def apply(self, graph, sdfg):
        read = self._source_for_expr(self.expr_index)
        access = self.access
        write = self._dest_for_expr(self.expr_index)

        in_edge = graph.edges_between(read, access)[0]
        out_edge = graph.edges_between(access, write)[0]

        new_memlet = copy.deepcopy(out_edge.data)
        if isinstance(read, nodes.Tasklet):
            # Tasklet-source memlets must stay destination-attached so src_subset
            # remains None on the new edge.
            new_memlet._is_data_src = False
            new_memlet.subset = copy.deepcopy(out_edge.data.dst_subset or out_edge.data.subset)
            new_memlet.other_subset = None
        else:
            new_memlet.other_subset = copy.deepcopy(in_edge.data.subset)

        graph.remove_edge(in_edge)
        graph.remove_edge(out_edge)
        graph.add_edge(read, in_edge.src_conn, write, out_edge.dst_conn, new_memlet)
        graph.remove_node(access)
