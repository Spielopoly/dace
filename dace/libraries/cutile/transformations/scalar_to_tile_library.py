"""
ScalarToTileLibrary transformation.

Matches an inner map + scalar tasklet nested inside an outer (tile) map
and replaces them with a cuTile library node operating on tile-shaped
transient arrays.

Pattern (BEFORE)::

    OuterMapEntry ─→ InnerMapEntry ─→ Tasklet ─→ InnerMapExit ─→ OuterMapExit

Result (AFTER)::

    OuterMapEntry ─→ tile_transient_read ─→ LibNode ─→ tile_transient_write ─→ OuterMapExit
"""
from __future__ import annotations

import copy
from typing import Optional

import dace
from dace import Memlet, dtypes, subsets
from dace.sdfg import SDFG, SDFGState, nodes, utils as sdutil
from dace.transformation import transformation as xf

from dace.libraries.cutile._op_registry import match_tasklet


class ScalarToTileLibrary(xf.SingleStateTransformation):
    """
    Replace an inner element-wise map + scalar tasklet with a cuTile
    library node that operates on the whole tile.

    Expects nested maps where the outer map iterates over tile indices
    and the inner map iterates over elements within each tile, with a
    single scalar tasklet performing a recognised element-wise operation.
    """

    outer_map_entry = xf.PatternNode(nodes.MapEntry)
    inner_map_entry = xf.PatternNode(nodes.MapEntry)
    tasklet = xf.PatternNode(nodes.Tasklet)
    inner_map_exit = xf.PatternNode(nodes.MapExit)
    outer_map_exit = xf.PatternNode(nodes.MapExit)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.outer_map_entry,
                                       cls.inner_map_entry,
                                       cls.tasklet,
                                       cls.inner_map_exit,
                                       cls.outer_map_exit)]

    def can_be_applied(self, graph: SDFGState, expr_index: int,
                       sdfg: SDFG, permissive: bool = False) -> bool:
        outer_entry = self.outer_map_entry
        inner_entry = self.inner_map_entry
        tasklet = self.tasklet
        inner_exit = self.inner_map_exit
        outer_exit = self.outer_map_exit

        # 1. Inner map scope must contain only the tasklet
        inner_scope = graph.scope_subgraph(inner_entry,
                                           include_entry=False,
                                           include_exit=False)
        if len(inner_scope.nodes()) != 1:
            return False
        if inner_scope.nodes()[0] is not tasklet:
            return False

        # 2. Outer map scope must contain exactly inner_entry + inner_exit + tasklet
        outer_scope = graph.scope_subgraph(outer_entry,
                                           include_entry=False,
                                           include_exit=False)
        if set(outer_scope.nodes()) != {inner_entry, inner_exit, tasklet}:
            return False

        # 3. All tasklet memlets must be scalar (single-element access)
        for e in list(graph.in_edges(tasklet)) + list(graph.out_edges(tasklet)):
            for rng in e.data.subset:
                start, end, step = rng
                if start != end:
                    return False

        # 4. Inner map ranges must start at 0 with step 1
        for rng in inner_entry.map.range:
            start, _, step = rng
            if start != 0 or step != 1:
                return False

        # 5. Tasklet must match a registered operation
        if match_tasklet(tasklet) is None:
            return False

        return True

    def apply(self, graph: SDFGState, sdfg: SDFG) -> None:
        outer_entry = self.outer_map_entry
        inner_entry = self.inner_map_entry
        inner_exit = graph.exit_node(inner_entry)
        outer_exit = graph.exit_node(outer_entry)
        tasklet = self.tasklet

        op_match = match_tasklet(tasklet)
        assert op_match is not None

        # Tile shape from inner map (e.g. [T0, T1])
        tile_shape = inner_entry.map.range.size()
        tile_subset = subsets.Range([(0, d - 1, 1) for d in tile_shape])

        # Create the library node
        lib_node = op_match.library_node_class(name=op_match.op_name)
        graph.add_node(lib_node)

        # --- Process inputs ---
        for edge in list(graph.out_edges(inner_entry)):
            if edge.dst is not tasklet:
                continue
            tasklet_conn = edge.dst_conn
            lib_conn = op_match.in_conn_map.get(tasklet_conn)
            if lib_conn is None:
                continue

            # Trace back to the outer_entry → inner_entry edge
            inner_out_conn = edge.src_conn          # e.g. "OUT_A"
            inner_in_conn = inner_out_conn.replace("OUT_", "IN_")
            outer_to_inner = self._find_edge(graph, outer_entry, inner_entry,
                                             dst_conn=inner_in_conn)
            if outer_to_inner is None:
                continue

            data_name = outer_to_inner.data.data
            data_desc = sdfg.arrays[data_name]

            # Create tile transient
            trans_name = sdfg._find_new_name(data_name + "_tile")
            sdfg.add_transient(
                trans_name,
                shape=list(tile_shape),
                dtype=data_desc.dtype,
                storage=data_desc.storage,
                lifetime=dtypes.AllocationLifetime.Scope,
            )
            trans_read = graph.add_access(trans_name)

            # outer_entry → transient (tile-slice memlet with other_subset)
            new_memlet = copy.deepcopy(outer_to_inner.data)
            new_memlet.other_subset = tile_subset
            graph.add_edge(outer_entry, outer_to_inner.src_conn,
                           trans_read, None, new_memlet)

            # transient → library node (full tile memlet)
            graph.add_edge(trans_read, None, lib_node, lib_conn,
                           Memlet(data=trans_name, subset=tile_subset))

            graph.remove_edge(outer_to_inner)

        # --- Process outputs ---
        for edge in list(graph.in_edges(inner_exit)):
            if edge.src is not tasklet:
                continue
            tasklet_conn = edge.src_conn
            lib_conn = op_match.out_conn_map.get(tasklet_conn)
            if lib_conn is None:
                continue

            inner_in_conn = edge.dst_conn           # e.g. "IN_C"
            inner_out_conn = inner_in_conn.replace("IN_", "OUT_")
            inner_to_outer = self._find_edge(graph, inner_exit, outer_exit,
                                             src_conn=inner_out_conn)
            if inner_to_outer is None:
                continue

            data_name = inner_to_outer.data.data
            data_desc = sdfg.arrays[data_name]

            trans_name = sdfg._find_new_name(data_name + "_tile")
            sdfg.add_transient(
                trans_name,
                shape=list(tile_shape),
                dtype=data_desc.dtype,
                storage=data_desc.storage,
                lifetime=dtypes.AllocationLifetime.Scope,
            )
            trans_write = graph.add_access(trans_name)

            # library node → transient
            graph.add_edge(lib_node, lib_conn, trans_write, None,
                           Memlet(data=trans_name, subset=tile_subset))

            # transient → outer_exit (tile-slice memlet with other_subset)
            new_memlet = copy.deepcopy(inner_to_outer.data)
            new_memlet.other_subset = tile_subset
            graph.add_edge(trans_write, None, outer_exit,
                           inner_to_outer.dst_conn, new_memlet)

            graph.remove_edge(inner_to_outer)

        # --- Remove inner map scope ---
        graph.remove_node(tasklet)
        graph.remove_node(inner_entry)
        graph.remove_node(inner_exit)

    # ------------------------------------------------------------------
    @staticmethod
    def _find_edge(graph, src, dst, src_conn=None, dst_conn=None):
        """Find an edge between *src* and *dst* matching optional connectors."""
        for e in graph.edges_between(src, dst):
            if src_conn is not None and e.src_conn != src_conn:
                continue
            if dst_conn is not None and e.dst_conn != dst_conn:
                continue
            return e
        return None
