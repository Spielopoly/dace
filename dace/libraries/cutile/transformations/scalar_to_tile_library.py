"""
ScalarToTileLibrary transformation.

This pass recognizes a "scalarized tile kernel": the program iterates over a
tile with an outer map, then iterates element-by-element inside that tile with
an inner map and a scalar tasklet. When the tasklet matches a known element-wise
operation, we can replace the scalar loop body with one cuTile library node that
operates on a whole tile at once.

Why this exists:
- Library nodes encode backend-specific optimized implementations.
- Replacing scalar inner loops with tile ops reduces graph complexity.
- The transformation preserves semantics by staging data through tile transients
  and preserving original outer-map indexing on loads/stores.

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
from typing import Dict, Optional

import dace
import sympy as sp
from dace import Memlet, dtypes, subsets
from dace.sdfg import SDFG, SDFGState, nodes, utils as sdutil
from dace.symbolic import symstr
from dace.transformation import transformation as xf

from dace.libraries.cutile.op_registry import match_tasklet_to_tile_library_node, MaskType, TaskletLibraryNodeMatch


class _ScalarToTileLibraryBase(xf.SingleStateTransformation):
    """
    Replace an inner scalar element-wise map nest with a tile library call.

    Expected structure:
    - The outer map identifies which tile of the larger tensor is processed.
    - The inner map iterates over positions *inside* that tile.
    - A single tasklet computes a supported element-wise op on scalar memlets.

    The transformation keeps the outer map intact and rewires only the inner
    region. This keeps surrounding program structure stable while replacing
    per-element work with tile-wide library execution.
    """

    outer_map_entry = xf.PatternNode(nodes.MapEntry)
    inner_map_entry = xf.PatternNode(nodes.MapEntry)
    tasklet = xf.PatternNode(nodes.Tasklet)
    inner_map_exit = xf.PatternNode(nodes.MapExit)
    outer_map_exit = xf.PatternNode(nodes.MapExit)

    _MASK_TYPE: MaskType
    _REQUIRES_CANONICAL: bool

    @classmethod
    def expressions(cls): # type: ignore
        """
        Declare the exact path-shaped pattern this transformation looks for.

        DaCe uses this expression to find candidate subgraphs before
        `can_be_applied` performs stricter semantic checks.
        """
        return [sdutil.node_path_graph(cls.outer_map_entry,
                                       cls.inner_map_entry,
                                       cls.tasklet,
                                       cls.inner_map_exit,
                                       cls.outer_map_exit)]

    def can_be_applied(self, graph: SDFGState, expr_index: int,
                       sdfg: SDFG, permissive: bool = False) -> bool:
        """
        Validate that the matched subgraph is a pure element-wise inner loop.

        The transformation only applies when the inner map has no extra nodes,
        tasklet accesses are scalar, and there is a registered tile operator
        for the tasklet code (including masked support for non-canonical maps).
        """

        outer_entry = self.outer_map_entry
        inner_entry = self.inner_map_entry
        tasklet = self.tasklet
        inner_exit = self.inner_map_exit
        outer_exit = self.outer_map_exit

        # 1) Inner map scope must contain only the tasklet body.
        inner_scope = graph.scope_subgraph(inner_entry,
                                           include_entry=False,
                                           include_exit=False)
        if len(inner_scope.nodes()) != 1:
            return False
        if inner_scope.nodes()[0] is not tasklet:
            return False

        # 2) Outer scope may only hold this inner map pair and tasklet.
        outer_scope = graph.scope_subgraph(outer_entry,
                                           include_entry=False,
                                           include_exit=False)
        if set(outer_scope.nodes()) != {inner_entry, inner_exit, tasklet}:
            return False

        # 3) Every tasklet access must be scalar (single-point range).
        # The op matcher assumes scalar tasklet connectors correspond to one
        # element position inside the tile; non-scalar memlets violate that model.
        for e in list(graph.in_edges(tasklet)) + list(graph.out_edges(tasklet)):
            if not isinstance(e.data.subset, subsets.Range):
                return False
            for rng in e.data.subset:
                start, end, step = rng
                if start != end:
                    return False

        # 4) Reject invalid map ranges (zero increment)
        for start, end, step in inner_entry.map.range:
            if step == 0:
                return False

        is_canonical = self._is_canonical_inner_map(inner_entry.map)
        if is_canonical != self._REQUIRES_CANONICAL:
            return False

        op_match = match_tasklet_to_tile_library_node(graph, tasklet, self._MASK_TYPE)
        if op_match is None:
            return False

        return True

    def apply(self, graph: SDFGState, sdfg: SDFG) -> None:  # type: ignore[override]
        """
        Apply the selected lowering mode for this specialization.
        """
        # `can_be_applied` already validated pattern shape and operator support;
        # here we perform the actual graph rewrite.
        outer_entry = self.outer_map_entry
        inner_entry = self.inner_map_entry
        inner_exit = self.inner_map_exit
        outer_exit = self.outer_map_exit
        tasklet = self.tasklet

        op_match = match_tasklet_to_tile_library_node(graph, tasklet, self._MASK_TYPE)
        if op_match is None:
            raise RuntimeError("Operator became unsupported between can_be_applied and apply.")

        if self._REQUIRES_CANONICAL:
            self._apply_canonical(graph, sdfg, outer_entry, inner_entry,
                                  tasklet, inner_exit, outer_exit, op_match)
        else:
            self._apply_noncanonical(graph, sdfg, outer_entry, inner_entry,
                                     tasklet, inner_exit, outer_exit, op_match)

    @staticmethod
    def _build_input_slot_map(op_match: TaskletLibraryNodeMatch) -> dict[str, list[str]]:
        """
        Build tasklet-input to library-input slot mapping.

        A single tasklet connector can map to multiple library slots, e.g.,
        for expressions such as c = a + a where rhs1 == rhs2.
        """
        slot_map: dict[str, list[str]] = {}
        classification = op_match.tasklet_classification
        node_info = op_match.node_info

        if classification.rhs1 is not None and node_info.rhs1 is not None:
            slot_map.setdefault(classification.rhs1, []).append(node_info.rhs1)
        if classification.rhs2 is not None and node_info.rhs2 is not None:
            slot_map.setdefault(classification.rhs2, []).append(node_info.rhs2)

        return slot_map

    @staticmethod
    def _output_connector_for_tasklet(op_match: TaskletLibraryNodeMatch,
                                      tasklet_conn: Optional[str]) -> Optional[str]:
        if tasklet_conn is None:
            return None
        if op_match.tasklet_classification.lhs == tasklet_conn:
            return op_match.node_info.out
        return None

    @staticmethod
    def _tasklet_input_edges_from_inner(graph: SDFGState,
                                        inner_entry: nodes.MapEntry,
                                        tasklet: nodes.Tasklet) -> Dict[str, dace.graph.MultiConnectorEdge[Memlet]]:
        """Map tasklet input connector to the edge inner_entry -> tasklet."""
        edge_map = {}
        for edge in graph.out_edges(inner_entry):
            if edge.dst is not tasklet or edge.dst_conn is None:
                continue
            edge_map[edge.dst_conn] = edge
        return edge_map

    def _apply_canonical(self, graph: SDFGState, sdfg: SDFG,
                         outer_entry: nodes.MapEntry, inner_entry: nodes.MapEntry,
                         tasklet: nodes.Tasklet, inner_exit: nodes.MapExit,
                         outer_exit: nodes.MapExit, op_match: TaskletLibraryNodeMatch) -> None:
        """
        Lower a canonical inner map (0-based, unit-stride) to a tile op.

        Data is staged through tile-shaped transients to convert scalar memlets
        into full-tile memlets consumed/produced by the library node.
        """
        # Derive the exact tile extents from the inner map. In canonical mode,
        # this is the concrete domain the library node should process.
        tile_shape = inner_entry.map.range.size()
        tile_subset = self._tile_subset_from_shape(tile_shape)

        # Instantiate the target library node selected by the operation matcher.
        lib_node = op_match.node_info.type(name=op_match.node_info.node_name)
        graph.add_node(lib_node)

        # === Input lowering ===
        # For each tasklet input connector:
        #   outer map slice -> tile transient -> library node input.
        # We preserve original outer indexing by copying memlets and only adding
        # `other_subset` to describe how the outer slice maps into tile space.
        input_slots = self._build_input_slot_map(op_match)
        input_edges = self._tasklet_input_edges_from_inner(graph, inner_entry, tasklet)

        # Multiple tasklet inputs can be fed by the same map connector
        # (frontend c = a + a often yields __in1/__in2 both from OUT_A).
        # Group by map input connector and lower each producer edge once.
        input_plan: dict[str, list[str]] = {}
        plan_edges = {}
        for tasklet_conn, lib_conns in input_slots.items():
            edge = input_edges.get(tasklet_conn)
            if edge is None or edge.src_conn is None:
                continue
            inner_in_conn = edge.src_conn.replace("OUT_", "IN_")
            input_plan.setdefault(inner_in_conn, []).extend(lib_conns)
            if inner_in_conn not in plan_edges:
                plan_edges[inner_in_conn] = edge

        for inner_in_conn, lib_conns in input_plan.items():
            edge = plan_edges[inner_in_conn]
            outer_to_inner = self._find_edge(graph, outer_entry, inner_entry,
                                             dst_conn=inner_in_conn)
            if outer_to_inner is None:
                continue

            data_name = outer_to_inner.data.data
            data_desc = sdfg.arrays[data_name]

            # Create a scope-lifetime transient tile to stage this operand.
            # Scope lifetime ensures no state escapes outside this map rewrite.
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

            # The library node consumes the full tile domain.
            for lib_conn in lib_conns:
                graph.add_edge(trans_read, None, lib_node, lib_conn,
                               Memlet(data=trans_name, subset=tile_subset))

            # Remove the original scalar feed from outer map to inner map.
            graph.remove_edge(outer_to_inner)

        # === Output lowering ===
        # Symmetric to input lowering:
        #   library node output -> tile transient -> original outer destination.
        for edge in graph.in_edges(inner_exit):
            if edge.src is not tasklet:
                continue
            tasklet_conn = edge.src_conn
            lib_conn = self._output_connector_for_tasklet(op_match, tasklet_conn)
            if lib_conn is None:
                continue

            # Same connector translation in reverse for inner-exit -> outer-exit.
            inner_in_conn = edge.dst_conn           # e.g. "IN_C"
            inner_out_conn = inner_in_conn.replace("IN_", "OUT_")
            inner_to_outer = self._find_edge(graph, inner_exit, outer_exit,
                                             src_conn=inner_out_conn)
            if inner_to_outer is None:
                continue

            data_name = inner_to_outer.data.data
            data_desc = sdfg.arrays[data_name]

            # Create per-output staging transient tile.
            trans_name = sdfg._find_new_name(data_name + "_tile")
            sdfg.add_transient(
                trans_name,
                shape=list(tile_shape),
                dtype=data_desc.dtype,
                storage=data_desc.storage,
                lifetime=dtypes.AllocationLifetime.Scope,
            )
            trans_write = graph.add_access(trans_name)

            # Library writes full tile result into transient.
            graph.add_edge(lib_node, lib_conn, trans_write, None,
                           Memlet(data=trans_name, subset=tile_subset))

            # transient → outer_exit (tile-slice memlet with other_subset)
            new_memlet = copy.deepcopy(inner_to_outer.data)
            new_memlet.other_subset = tile_subset
            graph.add_edge(trans_write, None, outer_exit,
                           inner_to_outer.dst_conn, new_memlet)

            # Remove the original scalar store path.
            graph.remove_edge(inner_to_outer)

        # After all dataflow edges are rewired through the library node, the
        # scalar tasklet and inner-map nodes are dead and can be removed.
        graph.remove_node(tasklet)
        graph.remove_node(inner_entry)
        graph.remove_node(inner_exit)

    def _apply_noncanonical(self, graph: SDFGState, sdfg: SDFG,
                            outer_entry: nodes.MapEntry, inner_entry: nodes.MapEntry,
                            tasklet: nodes.Tasklet, inner_exit: nodes.MapExit,
                            outer_exit: nodes.MapExit, op_match: TaskletLibraryNodeMatch) -> None:
        """
        Lower non-canonical inner maps using a mask-aware tile library node.

        Non-canonical ranges (offset starts, negative/strided bounds) are
        mapped to a bounding tile. A boolean mask marks valid points and is
        provided to the masked library node.
        """
        
        # The bounding tile is the smallest axis-aligned tile that contains all
        # iteration points from the original inner map, regardless of direction.
        tile_shape = self._bounding_tile_shape(inner_entry.map)
        tile_subset = self._tile_subset_from_shape(tile_shape)

        # `_m` receives domain mask; `_c_in` receives preloaded old output values
        # so masked lanes can preserve original values.
        lib_node = op_match.node_info.type(name=op_match.node_info.node_name)
        graph.add_node(lib_node)
        out_in_conn = op_match.node_info.out_in or "_c_in"
        mask_in_conn = op_match.node_info.mask_in or "_m"
        if out_in_conn not in lib_node.in_connectors:
            lib_node.add_in_connector(out_in_conn)

        # Keep mask storage aligned with operand storage when possible to avoid
        # introducing unnecessary storage-space transitions.
        mask_storage: Optional[dtypes.StorageType] = None

        # === Input lowering (non-canonical) ===
        # Input memlets are expanded from scalar expressions to contiguous outer
        # ranges covering the complete bounding tile footprint.
        input_slots = self._build_input_slot_map(op_match)
        input_edges = self._tasklet_input_edges_from_inner(graph, inner_entry, tasklet)

        input_plan: dict[str, list[str]] = {}
        plan_edges = {}
        for tasklet_conn, lib_conns in input_slots.items():
            edge = input_edges.get(tasklet_conn)
            if edge is None or edge.src_conn is None:
                continue
            inner_in_conn = edge.src_conn.replace("OUT_", "IN_")
            input_plan.setdefault(inner_in_conn, []).extend(lib_conns)
            if inner_in_conn not in plan_edges:
                plan_edges[inner_in_conn] = edge

        for inner_in_conn, lib_conns in input_plan.items():
            edge = plan_edges[inner_in_conn]
            outer_to_inner = self._find_edge(graph, outer_entry, inner_entry,
                                             dst_conn=inner_in_conn)
            if outer_to_inner is None:
                continue

            data_name = outer_to_inner.data.data
            data_desc = sdfg.arrays[data_name]
            if mask_storage is None:
                mask_storage = data_desc.storage

            # Convert scalar index expression(s) to a contiguous outer tile
            # range that covers all points visited by the inner map.
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

            for lib_conn in lib_conns:
                graph.add_edge(trans_read, None, lib_node, lib_conn,
                               Memlet(data=trans_name, subset=tile_subset))

            # Remove scalar path now represented by tile staging edges.
            graph.remove_edge(outer_to_inner)

        # If no input dictated storage, use backend default.
        if mask_storage is None:
            mask_storage = dtypes.StorageType.Default

        # Allocate the mask tile once per outer-map iteration.
        mask_name = sdfg._find_new_name("map_mask_tile")
        sdfg.add_transient(
            mask_name,
            shape=list(tile_shape),
            dtype=dace.bool,
            storage=mask_storage,
            lifetime=dtypes.AllocationLifetime.Scope,
        )

        # Build producer subgraph that computes per-lane validity predicate.
        # It must execute inside the outer map, because mask predicates can
        # reference tiled-loop symbols (e.g., tile_i/tile_j).
        mask_source, fill_entry = self._add_mask_fill_subgraph(graph, mask_name, tile_shape, inner_entry.map)
        graph.add_edge(outer_entry, None, fill_entry, None, Memlet())
        graph.add_edge(mask_source, None, lib_node, mask_in_conn,
                       Memlet(data=mask_name, subset=tile_subset))

        # === Output lowering (non-canonical) ===
        # We preload destination tile values so masked lanes can pass through
        # unchanged. Then we write the updated tile back to original footprint.
        for edge in list(graph.in_edges(inner_exit)):
            if edge.src is not tasklet:
                continue
            tasklet_conn = edge.src_conn
            lib_conn = self._output_connector_for_tasklet(op_match, tasklet_conn)
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
            if not isinstance(edge.data.subset, subsets.Subset):
                raise TypeError("Expected subset on inner map output edge.")
            store_subset = self._build_contiguous_outer_subset(edge.data.subset, inner_entry.map)

            # Preload current destination tile values into _c_in so masked ops
            # can keep lanes where mask == False.
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
            graph.add_edge(preload_tile, None, lib_node, out_in_conn,
                           Memlet(data=preload_name, subset=tile_subset))

            # Temporary tile for the new output produced by the library node.
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

            # Store full tile back to outer map footprint.
            graph.add_edge(
                trans_write,
                None,
                outer_exit,
                inner_to_outer.dst_conn,
                Memlet(data=data_name, subset=store_subset, other_subset=tile_subset),
            )

            # Remove original scalar inner-exit path.
            graph.remove_edge(inner_to_outer)

        # As in canonical lowering, the rewritten inner scope is now redundant.
        graph.remove_node(tasklet)
        graph.remove_node(inner_entry)
        graph.remove_node(inner_exit)

    @staticmethod
    def _tile_subset_from_shape(tile_shape) -> subsets.Range:
        """
        Build a dense local tile range [0, extent-1] in every dimension.

        This is the canonical subset used for transient tiles and library-node
        memlets, independent of where the tile lives in global tensor space.
        """
        return subsets.Range([(0, d - 1, 1) for d in tile_shape])

    @staticmethod
    def _bounding_tile_shape(inner_map: nodes.Map) -> list:
        """
        Compute per-dimension extents of a bounding box for an inner map range.

        Using min/max handles both increasing and decreasing ranges uniformly.
        The result is symbolic and may include expressions.
        """
        shape = []
        for start, end, _ in inner_map.range:
            start = _ScalarToTileLibraryBase._to_sympy_expr(start)
            end = _ScalarToTileLibraryBase._to_sympy_expr(end)
            low = sp.Min(start, end)
            high = sp.Max(start, end)
            shape.append(high - low + 1)
        return shape

    @staticmethod
    def _build_contiguous_outer_subset(tasklet_subset: subsets.Subset,
                                       inner_map: nodes.Map) -> subsets.Range:
        """
        Lift scalar tasklet accesses to a contiguous outer subset.

        For each accessed dimension, substitute the inner-map parameter with
        its min/max reachable values and build a conservative contiguous range.
        """
        if not isinstance(tasklet_subset, subsets.Range):
            raise TypeError("Expected range subset on tasklet memlet.")

        # Precompute symbolic min/max bounds for each inner-map parameter so we
        # can safely evaluate accesses even for reversed iteration ranges.
        param_bounds = {
            str(pname): (
                sp.Min(
                    _ScalarToTileLibraryBase._to_sympy_expr(start),
                    _ScalarToTileLibraryBase._to_sympy_expr(end),
                ),
                sp.Max(
                    _ScalarToTileLibraryBase._to_sympy_expr(start),
                    _ScalarToTileLibraryBase._to_sympy_expr(end),
                ),
            )
            for pname, (start, end, _) in zip(inner_map.params, inner_map.range)
        }

        new_ranges = []
        for rng in tasklet_subset:
            expr = _ScalarToTileLibraryBase._to_sympy_expr(rng[0])
            expr_symbols = {str(s): s for s in expr.free_symbols}
            used_params = [str(p) for p in inner_map.params if str(p) in expr_symbols]

            if len(used_params) > 1:
                # We currently support one inner-map symbol per dimension.
                # Multiple symbols imply coupling that is not representable as a
                # simple axis-aligned contiguous range.
                raise ValueError(
                    "Tasklet subset dimension depends on multiple inner-map parameters."
                )

            if not used_params:
                # Dimension independent of inner-map params is already fixed.
                new_ranges.append((expr, expr, 1))
                continue

            pname = str(used_params[0])
            map_symbol = expr_symbols[pname]
            low, high = param_bounds[pname]

            # Evaluate access expression at both map extremes.
            low_expr = expr.subs(map_symbol, low)
            high_expr = expr.subs(map_symbol, high)
            new_ranges.append((sp.Min(low_expr, high_expr),
                               sp.Max(low_expr, high_expr),
                               1))

        return subsets.Range(new_ranges)

    @staticmethod
    def _add_mask_fill_subgraph(graph: SDFGState, mask_name: str,
                                tile_shape: list, inner_map: nodes.Map) -> tuple[nodes.AccessNode, nodes.MapEntry]:
        """
        Build a sequential map that fills the mask tile with domain validity.

        Each mask element corresponds to a point in the bounding tile and is
        True iff that point is part of the original (possibly strided/reversed)
        inner-map iteration domain.
        """
        # Create one index variable per tile dimension (m0, m1, ...).
        params = [f"m{d}" for d in range(len(tile_shape))]
        map_ranges: Dict[str, str | subsets.Subset] = {
            p: f"0:{symstr(extent)}" for p, extent in zip(params, tile_shape)
        }
        fill_entry, fill_exit = graph.add_map(
            "fill_mask_map",
            map_ranges,
            schedule=dtypes.ScheduleType.Sequential,
        )

        cond_terms = []
        for p, (start, end, step) in zip(params, inner_map.range):
            start = _ScalarToTileLibraryBase._to_sympy_expr(start)
            end = _ScalarToTileLibraryBase._to_sympy_expr(end)
            step = _ScalarToTileLibraryBase._to_sympy_expr(step)
            low_expr = sp.Min(start, end)
            global_idx = f"(({symstr(low_expr)}) + ({p}))"

            start_s = symstr(start)
            end_s = symstr(end)
            step_s = symstr(step)

            # Two direction-specific predicates:
            # - cond_pos handles positive steps.
            # - cond_neg handles negative steps.
            # Both include divisibility checks to enforce strided membership.
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
        # Materialize predicate directly in a tasklet to avoid extra branches.
        fill_tasklet = graph.add_tasklet("fill_mask", {}, {"out"},
                                         f"out = {condition}")

        graph.add_edge(fill_entry, None, fill_tasklet, None, Memlet())

        # Write mask value to exactly one tile element per map point.
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
        return mask_write, fill_entry

    @staticmethod
    def _to_sympy_expr(expr):
        """Convert DaCe symbolic values (including SymExpr) to plain SymPy."""
        if isinstance(expr, dace.symbolic.SymExpr):
            return expr.expr
        return sp.sympify(expr)

    @staticmethod
    def _is_canonical_inner_map(inner_map: nodes.Map) -> bool:
        """
        Canonical inner map predicate used to select lowering strategy.

        Canonical means: start==0 and step==1 in every dimension, which implies
        the tile is fully dense and does not need a validity mask.
        """
        for start, _, step in inner_map.range:
            if start != 0 or step != 1:
                return False
        return True

    # ------------------------------------------------------------------
    @staticmethod
    def _find_edge(graph, src, dst, src_conn=None, dst_conn=None):
        """Find one edge between `src` and `dst` matching optional connectors."""
        for e in graph.edges_between(src, dst):
            if src_conn is not None and e.src_conn != src_conn:
                continue
            if dst_conn is not None and e.dst_conn != dst_conn:
                continue
            return e
        return None


class ScalarToTileLibraryCanonical(_ScalarToTileLibraryBase):
    """Lower canonical scalar inner maps to unmasked cuTile library nodes."""

    _MASK_TYPE = MaskType.UNMASKED
    _REQUIRES_CANONICAL = True


class ScalarToTileLibraryMasked(_ScalarToTileLibraryBase):
    """Lower non-canonical scalar inner maps to masked cuTile library nodes."""

    _MASK_TYPE = MaskType.RUNTIME
    _REQUIRES_CANONICAL = False
