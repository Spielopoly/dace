"""
ScalarToTileLibrary transformation.

Matches an inner map + scalar tasklet nested inside an outer (tile) map
and replaces them with a cuTile library node operating on tile-shaped
transient arrays.

Pattern (BEFORE)::

    OuterMapEntry ─→ InnerMapEntry ─→ Tasklet ─→ InnerMapExit ─→ OuterMapExit

Result (AFTER, canonical)::

    OuterMapEntry ─→ tile_transient_read ─→ LibNode ─→ tile_transient_write ─→ OuterMapExit

Result (AFTER, non-canonical inner maps)::

    OuterMapEntry ─→ tile_transient_read ─→ MaskedLibNode ─→ tile_transient_write ─→ OuterMapExit
                    + generated mask transient used to select valid map points
"""
from __future__ import annotations

import copy
from typing import Optional

import dace
import sympy as sp
from dace import Memlet, dtypes, subsets
from dace.sdfg import SDFG, SDFGState, nodes, utils as sdutil
from dace.symbolic import symstr
from dace.transformation import transformation as xf

from dace.libraries.cutile._op_registry import TileOpMatch, match_tasklet


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

        # 3. All tasklet memlets must be scalar (single-element accesses)
        for e in list(graph.in_edges(tasklet)) + list(graph.out_edges(tasklet)):
            if not isinstance(e.data.subset, subsets.Range):
                return False
            for rng in e.data.subset:
                start, end, step = rng
                if start != end:
                    return False

        # 4. Reject invalid map ranges (zero increment)
        for start, end, step in inner_entry.map.range:
            if step == 0:
                return False

        # 5. Tasklet must match a registered operation
        op_match = match_tasklet(tasklet)
        if op_match is None:
            return False

        # 6. Non-canonical maps require a masked counterpart
        if self._is_canonical_inner_map(inner_entry.map):
            if op_match[0] is None:
                return False
        else:
            if op_match[1] is None:
                return False

        return True

    def apply(self, graph: SDFGState, sdfg: SDFG) -> None:
        outer_entry = self.outer_map_entry
        inner_entry = self.inner_map_entry
        inner_exit = self.inner_map_exit
        outer_exit = self.outer_map_exit
        tasklet = self.tasklet

        op_match = match_tasklet(tasklet)
        assert op_match is not None

        if self._is_canonical_inner_map(inner_entry.map):
            assert op_match[0] is not None
            self._apply_canonical(graph, sdfg, outer_entry, inner_entry,
                                  tasklet, inner_exit, outer_exit, op_match[0])
        else:
            assert op_match[1] is not None
            self._apply_noncanonical(graph, sdfg, outer_entry, inner_entry,
                                     tasklet, inner_exit, outer_exit, op_match[1])

    def _apply_canonical(self, graph: SDFGState, sdfg: SDFG,
                         outer_entry: nodes.MapEntry, inner_entry: nodes.MapEntry,
                         tasklet: nodes.Tasklet, inner_exit: nodes.MapExit,
                         outer_exit: nodes.MapExit, op_match: TileOpMatch) -> None:
        # Tile shape from inner map (e.g. [T0, T1])
        tile_shape = inner_entry.map.range.size()
        tile_subset = self._tile_subset_from_shape(tile_shape)

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

    def _apply_noncanonical(self, graph: SDFGState, sdfg: SDFG,
                            outer_entry: nodes.MapEntry, inner_entry: nodes.MapEntry,
                            tasklet: nodes.Tasklet, inner_exit: nodes.MapExit,
                            outer_exit: nodes.MapExit, op_match: TileOpMatch) -> None:
        masked_cls = op_match.library_node_class
        assert masked_cls is not None

        tile_shape = self._bounding_tile_shape(inner_entry.map)
        tile_subset = self._tile_subset_from_shape(tile_shape)

        lib_node = masked_cls(name=op_match.op_name)
        graph.add_node(lib_node)
        lib_node.add_in_connector("_c_in")

        mask_storage: Optional[dtypes.StorageType] = None

        # --- Process inputs ---
        for edge in list(graph.out_edges(inner_entry)):
            if edge.dst is not tasklet:
                continue
            tasklet_conn = edge.dst_conn
            lib_conn = op_match.in_conn_map.get(tasklet_conn)
            if lib_conn is None:
                continue

            inner_out_conn = edge.src_conn
            inner_in_conn = inner_out_conn.replace("OUT_", "IN_")
            outer_to_inner = self._find_edge(graph, outer_entry, inner_entry,
                                             dst_conn=inner_in_conn)
            if outer_to_inner is None:
                continue

            data_name = outer_to_inner.data.data
            data_desc = sdfg.arrays[data_name]
            if mask_storage is None:
                mask_storage = data_desc.storage

            load_subset = self._build_contiguous_outer_subset(edge.data.subset, inner_entry.map)

            trans_name = sdfg._find_new_name(data_name + "_tile")
            sdfg.add_transient(
                trans_name,
                shape=list(tile_shape),
                dtype=data_desc.dtype,
                storage=data_desc.storage,
                lifetime=dtypes.AllocationLifetime.Scope,
            )
            trans_read = graph.add_access(trans_name)

            graph.add_edge(
                outer_entry,
                outer_to_inner.src_conn,
                trans_read,
                None,
                Memlet(data=data_name, subset=load_subset, other_subset=tile_subset),
            )

            graph.add_edge(trans_read, None, lib_node, lib_conn,
                           Memlet(data=trans_name, subset=tile_subset))

            graph.remove_edge(outer_to_inner)

        if mask_storage is None:
            mask_storage = dtypes.StorageType.Default

        mask_name = sdfg._find_new_name("map_mask_tile")
        sdfg.add_transient(
            mask_name,
            shape=list(tile_shape),
            dtype=dace.bool,
            storage=mask_storage,
            lifetime=dtypes.AllocationLifetime.Scope,
        )

        mask_source = self._add_mask_fill_subgraph(graph, mask_name, tile_shape, inner_entry.map)
        mask_read = graph.add_access(mask_name)

        outer_entry.add_in_connector("IN_mask")
        outer_entry.add_out_connector("OUT_mask")
        graph.add_edge(mask_source, None, outer_entry, "IN_mask",
                       Memlet(data=mask_name, subset=tile_subset))
        graph.add_edge(outer_entry, "OUT_mask", mask_read, None,
                       Memlet(data=mask_name, subset=tile_subset))
        graph.add_edge(mask_read, None, lib_node, "_m",
                       Memlet(data=mask_name, subset=tile_subset))

        # --- Process outputs ---
        for edge in list(graph.in_edges(inner_exit)):
            if edge.src is not tasklet:
                continue
            tasklet_conn = edge.src_conn
            lib_conn = op_match.out_conn_map.get(tasklet_conn)
            if lib_conn is None:
                continue

            inner_in_conn = edge.dst_conn
            inner_out_conn = inner_in_conn.replace("IN_", "OUT_")
            inner_to_outer = self._find_edge(graph, inner_exit, outer_exit,
                                             src_conn=inner_out_conn)
            if inner_to_outer is None:
                continue

            data_name = inner_to_outer.data.data
            data_desc = sdfg.arrays[data_name]
            store_subset = self._build_contiguous_outer_subset(edge.data.subset, inner_entry.map)

            preload_name = sdfg._find_new_name(data_name + "_tile_in")
            sdfg.add_transient(
                preload_name,
                shape=list(tile_shape),
                dtype=data_desc.dtype,
                storage=data_desc.storage,
                lifetime=dtypes.AllocationLifetime.Scope,
            )
            preload_tile = graph.add_access(preload_name)

            preload_node = graph.add_access(data_name)
            preload_in_conn = f"IN_PRELOAD_{preload_name}"
            preload_out_conn = preload_in_conn.replace("IN_", "OUT_", 1)
            outer_entry.add_in_connector(preload_in_conn)
            outer_entry.add_out_connector(preload_out_conn)
            graph.add_edge(preload_node, None, outer_entry, preload_in_conn,
                           Memlet(data=data_name, subset=store_subset))
            graph.add_edge(
                outer_entry,
                preload_out_conn,
                preload_tile,
                None,
                Memlet(data=data_name, subset=store_subset, other_subset=tile_subset),
            )
            graph.add_edge(preload_tile, None, lib_node, "_c_in",
                           Memlet(data=preload_name, subset=tile_subset))

            trans_name = sdfg._find_new_name(data_name + "_tile")
            sdfg.add_transient(
                trans_name,
                shape=list(tile_shape),
                dtype=data_desc.dtype,
                storage=data_desc.storage,
                lifetime=dtypes.AllocationLifetime.Scope,
            )
            trans_write = graph.add_access(trans_name)

            graph.add_edge(lib_node, lib_conn, trans_write, None,
                           Memlet(data=trans_name, subset=tile_subset))

            graph.add_edge(
                trans_write,
                None,
                outer_exit,
                inner_to_outer.dst_conn,
                Memlet(data=data_name, subset=store_subset, other_subset=tile_subset),
            )

            graph.remove_edge(inner_to_outer)

        graph.remove_node(tasklet)
        graph.remove_node(inner_entry)
        graph.remove_node(inner_exit)

    @staticmethod
    def _tile_subset_from_shape(tile_shape) -> subsets.Range:
        return subsets.Range([(0, d - 1, 1) for d in tile_shape])

    @staticmethod
    def _bounding_tile_shape(inner_map: nodes.Map) -> list:
        shape = []
        for start, end, _ in inner_map.range:
            low = sp.Min(start, end)
            high = sp.Max(start, end)
            shape.append(high - low + 1)
        return shape

    @staticmethod
    def _build_contiguous_outer_subset(tasklet_subset: subsets.Subset,
                                       inner_map: nodes.Map) -> subsets.Range:
        if not isinstance(tasklet_subset, subsets.Range):
            raise TypeError("Expected range subset on tasklet memlet.")

        param_bounds = {
            pname: (sp.Min(start, end), sp.Max(start, end))
            for pname, (start, end, _) in zip(inner_map.params, inner_map.range)
        }

        new_ranges = []
        for rng in tasklet_subset:
            expr = sp.sympify(rng[0])
            expr_symbols = {str(s): s for s in expr.free_symbols}
            used_params = [p for p in inner_map.params if p in expr_symbols]

            if len(used_params) > 1:
                raise ValueError(
                    "Tasklet subset dimension depends on multiple inner-map parameters."
                )

            if not used_params:
                new_ranges.append((expr, expr, 1))
                continue

            pname = used_params[0]
            map_symbol = expr_symbols[pname]
            low, high = param_bounds[pname]

            low_expr = expr.subs(map_symbol, low)
            high_expr = expr.subs(map_symbol, high)
            new_ranges.append((sp.Min(low_expr, high_expr),
                               sp.Max(low_expr, high_expr),
                               1))

        return subsets.Range(new_ranges)

    @staticmethod
    def _add_mask_fill_subgraph(graph: SDFGState, mask_name: str,
                                tile_shape: list, inner_map: nodes.Map) -> nodes.AccessNode:
        params = [f"m{d}" for d in range(len(tile_shape))]
        map_ranges = {p: f"0:{symstr(extent)}"
                      for p, extent in zip(params, tile_shape)}
        fill_entry, fill_exit = graph.add_map(
            "fill_mask_map",
            map_ranges,
            schedule=dtypes.ScheduleType.Sequential,
        )

        cond_terms = []
        for p, (start, end, step) in zip(params, inner_map.range):
            low_expr = sp.Min(start, end)
            global_idx = f"(({symstr(low_expr)}) + ({p}))"

            start_s = symstr(start)
            end_s = symstr(end)
            step_s = symstr(step)

            cond_pos = (
                f"((({step_s}) > 0) and ({global_idx} >= ({start_s})) and "
                f"({global_idx} <= ({end_s})) and "
                f"((({global_idx} - ({start_s})) % ({step_s})) == 0))"
            )
            cond_neg = (
                f"((({step_s}) < 0) and ({global_idx} <= ({start_s})) and "
                f"({global_idx} >= ({end_s})) and "
                f"((((({start_s}) - ({global_idx})) % (-({step_s}))) == 0)))"
            )
            cond_terms.append(f"(({cond_pos}) or ({cond_neg}))")

        condition = " and ".join(cond_terms) if cond_terms else "True"
        fill_tasklet = graph.add_tasklet("fill_mask", {}, {"out"},
                                         f"out = {condition}")

        graph.add_edge(fill_entry, None, fill_tasklet, None, Memlet())

        mask_write = graph.add_access(mask_name)
        idx_symbols = [sp.Symbol(p) for p in params]
        idx_subset = subsets.Range([(s, s, 1) for s in idx_symbols])
        mask_in_conn = "IN_map_mask_tile"
        mask_out_conn = "OUT_map_mask_tile"
        fill_exit.add_in_connector(mask_in_conn)
        fill_exit.add_out_connector(mask_out_conn)
        graph.add_edge(fill_tasklet, "out", fill_exit, mask_in_conn,
                       Memlet(data=mask_name, subset=idx_subset))
        graph.add_edge(fill_exit, mask_out_conn, mask_write, None,
                       Memlet(data=mask_name, subset=idx_subset))
        return mask_write

    @staticmethod
    def _is_canonical_inner_map(inner_map: nodes.Map) -> bool:
        for start, _, step in inner_map.range:
            if start != 0 or step != 1:
                return False
        return True

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
