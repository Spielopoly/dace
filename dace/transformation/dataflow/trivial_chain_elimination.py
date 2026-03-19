# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Conservative elimination of a trivial tasklet and one adjacent access node."""

import copy

from dace import data, symbolic
from dace.dtypes import AllocationLifetime, StorageType
from dace.properties import make_properties
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import transformation


@make_properties
class TrivialChainElimination(transformation.SingleStateTransformation):
    """Eliminates a trivial copy-tasklet together with one adjacent access node.

    This transformation is intentionally conservative and only matches local
    direct-neighbor chains with exactly one access node adjacent to the trivial
    tasklet.
    """

    src_access = transformation.PatternNode(nodes.AccessNode)
    src_map = transformation.PatternNode(nodes.MapEntry)
    src_tasklet = transformation.PatternNode(nodes.Tasklet)
    access = transformation.PatternNode(nodes.AccessNode)
    tasklet = transformation.PatternNode(nodes.Tasklet)
    dst_access = transformation.PatternNode(nodes.AccessNode)
    dst_map = transformation.PatternNode(nodes.MapExit)

    @classmethod
    def expressions(cls):
        # access -> tasklet chain
        exprs = [
            sdutil.node_path_graph(cls.src_access, cls.access, cls.tasklet, cls.dst_access),
            sdutil.node_path_graph(cls.src_access, cls.access, cls.tasklet, cls.dst_map),
            sdutil.node_path_graph(cls.src_map, cls.access, cls.tasklet, cls.dst_access),
            sdutil.node_path_graph(cls.src_map, cls.access, cls.tasklet, cls.dst_map),
            sdutil.node_path_graph(cls.src_tasklet, cls.access, cls.tasklet, cls.dst_access),
            sdutil.node_path_graph(cls.src_tasklet, cls.access, cls.tasklet, cls.dst_map),
        ]
        # tasklet -> access chain
        exprs.extend([
            sdutil.node_path_graph(cls.src_access, cls.tasklet, cls.access, cls.dst_access),
            sdutil.node_path_graph(cls.src_access, cls.tasklet, cls.access, cls.dst_map),
            sdutil.node_path_graph(cls.src_map, cls.tasklet, cls.access, cls.dst_access),
            sdutil.node_path_graph(cls.src_map, cls.tasklet, cls.access, cls.dst_map),
        ])
        return exprs

    def _access_before_tasklet(self, expr_index: int) -> bool:
        return expr_index < 6

    def _source_for_expr(self, expr_index: int):
        if expr_index in (0, 1, 6, 7):
            return self.src_access
        if expr_index in (2, 3, 8, 9):
            return self.src_map
        return self.src_tasklet

    def _dest_for_expr(self, expr_index: int):
        if expr_index in (0, 2, 4, 6, 8):
            return self.dst_access
        return self.dst_map

    def _is_trivial_tasklet(self, graph, sdfg, source, tasklet, dest, in_edge, out_edge):
        if len(tasklet.in_connectors) != 1:
            return False
        if len(tasklet.out_connectors) != 1:
            return False
        if len(graph.in_edges(tasklet)) != 1:
            return False
        if len(graph.out_edges(tasklet)) != 1:
            return False

        in_conn = list(tasklet.in_connectors.keys())[0]
        out_conn = list(tasklet.out_connectors.keys())[0]
        if tasklet.code.as_string != f'{out_conn} = {in_conn}':
            return False

        if out_edge.data.wcr:
            return False

        read_desc = sdfg.arrays[in_edge.data.data]
        write_desc = sdfg.arrays[out_edge.data.data]
        if isinstance(read_desc, data.Stream):
            return False
        if isinstance(write_desc, data.Stream):
            return False

        if (isinstance(source, nodes.MapEntry) or isinstance(dest, nodes.MapExit)) and read_desc.dtype != write_desc.dtype:
            return False

        return True

    def _is_eliminable_access(self, graph, sdfg, access, in_edge, out_edge):
        if graph.in_degree(access) != 1 or graph.out_degree(access) != 1:
            return False

        access_desc = access.desc(sdfg)
        if not access_desc.transient:
            return False
        if access_desc.lifetime != AllocationLifetime.Scope:
            return False
        if access_desc.storage != StorageType.Default:
            return False
        if isinstance(access_desc, data.Stream):
            return False
        if isinstance(access_desc, data.View):
            return False

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

        return True

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        source = self._source_for_expr(expr_index)
        access = self.access
        tasklet = self.tasklet
        dest = self._dest_for_expr(expr_index)

        if isinstance(source, nodes.CodeNode) and isinstance(dest, nodes.CodeNode):
            return False

        if self._access_before_tasklet(expr_index):
            access_in = graph.edges_between(source, access)
            access_out = graph.edges_between(access, tasklet)
            tasklet_out = graph.edges_between(tasklet, dest)
            if len(access_in) != 1 or len(access_out) != 1 or len(tasklet_out) != 1:
                return False

            if not self._is_eliminable_access(graph, sdfg, access, access_in[0], access_out[0]):
                return False
            if not self._is_trivial_tasklet(graph, sdfg, source, tasklet, dest, access_out[0], tasklet_out[0]):
                return False
        else:
            tasklet_in = graph.edges_between(source, tasklet)
            tasklet_out = graph.edges_between(tasklet, access)
            access_out = graph.edges_between(access, dest)
            if len(tasklet_in) != 1 or len(tasklet_out) != 1 or len(access_out) != 1:
                return False

            if not self._is_trivial_tasklet(graph, sdfg, source, tasklet, dest, tasklet_in[0], tasklet_out[0]):
                return False
            if not self._is_eliminable_access(graph, sdfg, access, tasklet_out[0], access_out[0]):
                return False

        return True

    def apply(self, graph, sdfg):
        source = self._source_for_expr(self.expr_index)
        access = self.access
        tasklet = self.tasklet
        dest = self._dest_for_expr(self.expr_index)

        if self._access_before_tasklet(self.expr_index):
            access_in_edge = graph.edges_between(source, access)[0]
            tasklet_in_edge = graph.edges_between(access, tasklet)[0]
            tasklet_out_edge = graph.edges_between(tasklet, dest)[0]

            tmp_memlet = copy.deepcopy(tasklet_out_edge.data)
            tmp_memlet.other_subset = copy.deepcopy(tasklet_in_edge.data.subset)

            final_memlet = copy.deepcopy(tmp_memlet)
            if isinstance(source, nodes.Tasklet):
                final_memlet._is_data_src = False
                final_memlet.subset = copy.deepcopy(tmp_memlet.dst_subset or tmp_memlet.subset)
                final_memlet.other_subset = None
            else:
                final_memlet.other_subset = copy.deepcopy(access_in_edge.data.subset)

            src_conn = access_in_edge.src_conn
            dst_conn = tasklet_out_edge.dst_conn

            graph.remove_edge(access_in_edge)
            graph.remove_edge(tasklet_in_edge)
            graph.remove_edge(tasklet_out_edge)
        else:
            tasklet_in_edge = graph.edges_between(source, tasklet)[0]
            tasklet_out_edge = graph.edges_between(tasklet, access)[0]
            access_out_edge = graph.edges_between(access, dest)[0]

            tmp_memlet = copy.deepcopy(tasklet_out_edge.data)
            tmp_memlet.other_subset = copy.deepcopy(tasklet_in_edge.data.subset)

            final_memlet = copy.deepcopy(access_out_edge.data)
            if isinstance(source, nodes.Tasklet):
                final_memlet._is_data_src = False
                final_memlet.subset = copy.deepcopy(access_out_edge.data.dst_subset or access_out_edge.data.subset)
                final_memlet.other_subset = None
            else:
                final_memlet.other_subset = copy.deepcopy(tmp_memlet.subset)

            src_conn = tasklet_in_edge.src_conn
            dst_conn = access_out_edge.dst_conn

            graph.remove_edge(tasklet_in_edge)
            graph.remove_edge(tasklet_out_edge)
            graph.remove_edge(access_out_edge)

        graph.add_edge(source, src_conn, dest, dst_conn, final_memlet)
        graph.remove_node(tasklet)
        graph.remove_node(access)
