"""
ScalarToTile transformation.

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

    OuterMapEntry -> InnerMapEntry -> Tasklet -> InnerMapExit -> OuterMapExit

Result (AFTER, canonical)::

    OuterMapEntry -> tile_transient_read -> LibNode -> tile_transient_write -> OuterMapExit

Result (AFTER, non-canonical inner maps)::

    OuterMapEntry -> tile_transient_read  -> MaskedLibNode -> tile_transient_write -> OuterMapExit
                  -> preload_transient -> MaskedLibNode._c_in
                    + symbolic mask condition selects valid map points
                    (masked-out lanes are preserved via _c_in preload)
"""


import abc
import copy
import itertools
from typing import Optional, cast

import dace
import sympy as sp
from sympy.polys.polyerrors import PolynomialError
from dace import Memlet, dtypes, subsets
from dace.sdfg import SDFG, SDFGState, nodes, utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.scope import ScopeSubgraphView
from dace.symbolic import SymExpr
from dace.transformation import transformation as xf

from dace.libraries.cutile.op_registry import match_tasklet_to_tile_library_node, MaskType
from dace.libraries.cutile.transformations.utils import (
    tile_subset_from_shape,
    create_tile_transient,
    primary_memlet_subset,
    with_primary_subset,
    memlet_with_primary_subset,
)


class _ScalarToTileBase(xf.SingleStateTransformation, abc.ABC):
    """
    Base class for ScalarToTile transformations.
    Not intended to be used directly.

    Any function may be overridden by child classes to change the applicability
    conditions or transformation behavior, but the default implementation
    assumes a very strict and simple pattern of an inner map with exactly one
    tasklet, where all accesses in the tasklet are scalar and match a registered
    tile library node. This is sufficient for the canonical case, but
    non-canonical cases may need to relax some of these conditions.

    The ``apply`` method is structured as a template method, calling hooks that
    child classes override to inject specialised behaviour (e.g. mask creation,
    view-based output, non-canonical memlet construction).

    .. warning::

        ``PatternNode`` descriptors (``self.outer_map_entry`` etc.) resolve
        nodes by integer index in the state's node list.  Adding or removing
        graph nodes shifts those indices, making descriptor access return
        **wrong** nodes.  ``_set_convenience_variables`` therefore captures
        the actual node objects once, and all methods called from ``apply``
        must use the captured ``self._outer_entry`` / ``self._inner_entry`` /
        ``self._inner_exit`` / ``self._outer_exit`` / ``self._tasklet``
        references instead of the descriptors.
    """

    outer_map_entry = xf.PatternNode(nodes.MapEntry)
    inner_map_entry = xf.PatternNode(nodes.MapEntry)
    tasklet = xf.PatternNode(nodes.Tasklet)
    inner_map_exit = xf.PatternNode(nodes.MapExit)
    outer_map_exit = xf.PatternNode(nodes.MapExit)

    @classmethod
    def expressions(cls):  # type: ignore
        """
        Declare the exact path-shaped pattern this transformation looks for.

        DaCe uses this expression to find candidate subgraphs before
        ``can_be_applied`` performs stricter semantic checks.
        """
        return [sdutil.node_path_graph(cls.outer_map_entry,
                                       cls.inner_map_entry,
                                       cls.tasklet,
                                       cls.inner_map_exit,
                                       cls.outer_map_exit)]

    # ---- convenience helpers ------------------------------------------------

    def _set_convenience_variables(self, sdfg: SDFG, graph: SDFGState) -> None:
        """Capture the matched SDFG nodes as instance attributes.

        Must be called at the start of both :meth:`can_be_applied` and
        :meth:`apply` before any graph modifications occur.  The
        ``PatternNode`` descriptors resolve nodes by integer index in the
        state's node list; adding or removing nodes shifts those indices and
        makes subsequent descriptor accesses return wrong nodes.  Capturing
        the actual node objects here avoids that hazard.

        Args:
            sdfg: The SDFG being transformed.
            graph: The state containing the matched subgraph.
        """
        self._sdfg = sdfg
        self._graph = graph
        # Capture actual node objects NOW, before any graph modifications.
        # See class docstring for rationale.
        self._outer_entry: nodes.MapEntry = self.outer_map_entry
        self._inner_entry: nodes.MapEntry = self.inner_map_entry
        self._tasklet_node: nodes.Tasklet = self.tasklet
        self._inner_exit: nodes.MapExit = self.inner_map_exit
        self._outer_exit: nodes.MapExit = self.outer_map_exit

    def _get_scope(self, map_entry: nodes.MapEntry) -> ScopeSubgraphView:
        """Return the scope subgraph of *map_entry*, excluding entry and exit."""
        return self._graph.scope_subgraph(map_entry, include_entry=False, include_exit=False)

    def _get_all_edges_of_node(self, node: nodes.Node) -> list[MultiConnectorEdge[Memlet]]:
        """Return all incoming and outgoing edges of *node*."""
        return list(self._graph.in_edges(node)) + list(self._graph.out_edges(node))

    def _find_path_edge(self, anchor_edge: MultiConnectorEdge[Memlet], src: nodes.Node, dst: nodes.Node) -> Optional[MultiConnectorEdge[Memlet]]:
        """
        Find the path segment from *src* to *dst* on *anchor_edge*'s memlet path.

        Walk the full memlet path of *anchor_edge* and return the single edge
        whose source is *src* and destination is *dst*. Returns ``None`` when no
        such segment exists.
        """
        for e in self._graph.memlet_path(anchor_edge):
            if e.src is src and e.dst is dst:
                return e
        return None

    # ---- applicability checks -----------------------------------------------
    # These are called from can_be_applied()

    def _has_exactly_one_tasklet_in_inner_scope(self) -> bool:
        """Inner map scope must contain only the single matched tasklet body."""
        inner_scope = self._get_scope(self._inner_entry)
        return len(inner_scope.nodes()) == 1 \
            and self._tasklet_node in inner_scope.nodes()

    def _has_valid_inner_scope(self) -> bool:
        """Validate that the inner scope is a pure single-tasklet body."""
        return self._has_exactly_one_tasklet_in_inner_scope()

    def _has_valid_outer_scope(self) -> bool:
        """
        Validate that the outer scope holds only this inner map pair and tasklet.

        If additional nodes live between the outer map entry/exit (e.g. extra
        tasklets, access nodes, or other map nests), this is not a simple
        scalarized tile kernel and we must reject it.
        """
        outer_scope = self._get_scope(self._outer_entry)
        return set(outer_scope.nodes()) == {self._inner_entry, self._inner_exit, self._tasklet_node}

    def _has_valid_scope(self) -> bool:
        """Combined inner + outer scope validity check."""
        return self._has_valid_inner_scope() and self._has_valid_outer_scope()

    def _is_tasklet_scalar(self, tasklet: nodes.Tasklet) -> bool:
        """
        Every tasklet access must be scalar (single-point range).
        The op matcher assumes scalar tasklet connectors correspond to one
        element position inside the tile; non-scalar memlets violate that model.
        """
        for edge in self._get_all_edges_of_node(tasklet):
            edge_subset = primary_memlet_subset(edge.data)
            if not isinstance(edge_subset, subsets.Range):
                return False
            for rng in edge_subset:
                start, end, step = rng
                if start != end:
                    return False
        return True

    @abc.abstractmethod
    def _has_valid_inner_map_ranges(self) -> bool:
        """
        Validate inner-map ranges.  Implemented by each child class.

        The canonical child rejects non-zero starts and non-unit strides;
        the masked child rejects zero strides but accepts everything else
        (offset starts and non-unit positive strides).
        """
        ...

    def _has_valid_outer_map_ranges(self) -> bool:
        """Outer-map ranges are generally unconstrained."""
        return True

    def _has_valid_map_ranges(self) -> bool:
        """Combined inner + outer map range validity check."""
        return self._has_valid_inner_map_ranges() and self._has_valid_outer_map_ranges()

    @abc.abstractmethod
    def _get_mask_type(self) -> MaskType:
        """
        Return the mask type used to query the op registry.

        This determines which set of library nodes the matcher will consider.
        ``UNMASKED`` returns plain tile ops; ``RUNTIME`` returns masked ops
        that accept an additional boolean mask input.
        """
        ...

    def can_be_applied(self, graph: SDFGState, expr_index: int,
                       sdfg: SDFG, permissive: bool = False) -> bool:
        """Validate that the matched subgraph is a pure element-wise inner loop.

        The transformation only applies when the inner map has no extra nodes,
        tasklet accesses are scalar, and there is a registered tile operator
        for the tasklet code.

        Args:
            graph: The SDFG state containing the matched subgraph.
            expr_index: Index of the matched expression (always 0).
            sdfg: The top-level SDFG.
            permissive: Unused; present for API compatibility.

        Returns:
            ``True`` if all conditions are satisfied.
        """
        self._set_convenience_variables(sdfg, graph)

        if not self._has_valid_scope():
            return False
        if not self._is_tasklet_scalar(self._tasklet_node):
            return False
        if not self._has_valid_map_ranges():
            return False
        if match_tasklet_to_tile_library_node(graph, self._tasklet_node, self._get_mask_type(), promote_scalars=True) is None:
            return False
        if self._try_calculate_tile_shape() is None:
            return False

        return True

    # ---- tile shape / subset helpers ----------------------------------------

    def _calculate_tile_shape(self) -> tuple[sp.Basic | int, ...]:
        """
        Tile shape from inner-map range maxima.

        Per dimension, the tile shape is ``end + 1``.  Maps always use a
        positive step (descending maps are rejected by SDFG validation), so
        the inclusive ``end`` is the largest reachable coordinate and the tile
        covers from index 0 up to it.  This ensures power-of-2 friendly shapes
        and avoids bounding-box arithmetic.

        If the cutile pipeline is used, starting at 0 is (usually) not a big
        over-approximation, because Map-Tiling skews the inner maps to start at 0

        When the result still contains map-local parameters, conservative
        over-approximation is used to eliminate them, keeping descriptor
        shapes independent of map-local symbols.
        """
        assert isinstance(self._inner_entry.map.range, subsets.Range)
        base_shape: list[sp.Basic] = []
        for _start, end, _ in self._inner_entry.map.range:
            end_expr = self._to_sympy_expr(end)
            base_shape.append(end_expr + 1)

        local_params = {
            str(pname) for pname in self._outer_entry.map.params
        }
        local_params.update(str(pname) for pname in self._inner_entry.map.params)
        if not local_params:
            return tuple(base_shape)

        parameter_bounds = self._collect_local_map_parameter_bounds()
        descriptor_shape: list[sp.Basic] = []

        for extent in base_shape:
            if not self._expr_uses_symbol_names(extent, local_params):
                descriptor_shape.append(extent)
                continue

            resolved_extent = self._eliminate_local_map_parameters(
                extent,
                local_params,
                parameter_bounds,
            )
            resolved_extent = sp.sympify(dace.symbolic.overapproximate(resolved_extent))
            resolved_extent = self._eliminate_local_map_parameters(
                resolved_extent,
                local_params,
                parameter_bounds,
            )
            resolved_extent = sp.sympify(dace.symbolic.overapproximate(resolved_extent))

            if self._expr_uses_symbol_names(resolved_extent, local_params):
                raise ValueError(
                    "Tile shape still depends on map-local symbols after "
                    "conservative over-approximation"
                )

            descriptor_shape.append(resolved_extent)

        return tuple(descriptor_shape)

    def _try_calculate_tile_shape(self) -> Optional[tuple[sp.Basic | int, ...]]:
        """Return a symbol-safe tile shape or ``None`` if elimination is unsafe."""
        try:
            return self._calculate_tile_shape()
        except ValueError:
            return None

    @staticmethod
    def _expr_uses_symbol_names(expr: sp.Basic | int,
                                symbol_names: set[str]) -> bool:
        """Return whether *expr* references any symbol in *symbol_names*."""
        sym_expr = _ScalarToTileBase._to_sympy_expr(expr)
        return any(str(symbol) in symbol_names for symbol in sym_expr.free_symbols)

    def _collect_local_map_parameter_bounds(self) -> dict[str, tuple[sp.Basic, sp.Basic]]:
        """Collect conservative ``(low, high)`` bounds for inner/outer map params.

        Maps always use a positive step, so ``start`` is the lower bound and
        the inclusive ``end`` is the upper bound directly.
        """
        bounds: dict[str, tuple[sp.Basic, sp.Basic]] = {}
        for map_node in (self._outer_entry.map, self._inner_entry.map):
            for pname, (start, end, _) in zip(map_node.params, map_node.range):
                start_expr = self._to_sympy_expr(start)
                end_expr = self._to_sympy_expr(end)
                bounds[str(pname)] = (start_expr, end_expr)
        return bounds

    @staticmethod
    def _eliminate_local_map_parameters(expr: sp.Basic,
                                        local_params: set[str],
                                        parameter_bounds: dict[str, tuple[sp.Basic, sp.Basic]]) -> sp.Basic:
        """Eliminate map-local symbols from *expr* by bound substitution."""
        result = sp.sympify(expr)
        max_rounds = len(parameter_bounds) + 2
        for _ in range(max_rounds):
            active_symbols = [s for s in result.free_symbols if str(s) in local_params]
            if not active_symbols:
                break

            changed = False
            for symbol in active_symbols:
                symbol_bounds = parameter_bounds.get(str(symbol))
                if symbol_bounds is None:
                    continue
                low, high = symbol_bounds
                result = sp.Max(result.subs(symbol, low), result.subs(symbol, high))
                changed = True

            if not changed:
                break

        return result

    # ---- input/output slot mapping ------------------------------------------

    def _get_map_from_tasklet_input_to_libnode_inputs(self) -> dict[str, list[str]]:
        """
        Build tasklet-input -> library-input slot mapping by position.

        Maps ``classification.rhs1`` to ``node_info.rhs1`` and
        ``classification.rhs2`` to ``node_info.rhs2`` when both are non-None.
        A single tasklet connector can map to multiple library slots (e.g.
        ``c = a + a`` where ``rhs1 == rhs2``).

        Constants are handled separately via ``_configure_library_node``.
        """
        slot_map: dict[str, list[str]] = {}
        classification = self._tasklet_classification
        node_info = self._node_info

        if classification.rhs1 is not None and node_info.rhs1 is not None:
            slot_map.setdefault(classification.rhs1, []).append(node_info.rhs1)
        if classification.rhs2 is not None and node_info.rhs2 is not None:
            slot_map.setdefault(classification.rhs2, []).append(node_info.rhs2)

        return slot_map

    def _output_connector_for_tasklet(self, tasklet_conn: Optional[str]) -> Optional[str]:
        """Map a tasklet output connector to the corresponding library output."""
        if tasklet_conn is None:
            return None
        if self._tasklet_classification.lhs == tasklet_conn:
            return self._node_info.out
        return None

    # ---- input plan ---------------------------------------------------------

    def _create_input_plan(self, input_slots: dict[str, list[str]]) -> dict[tuple[Optional[str], Optional[str]], tuple[MultiConnectorEdge[Memlet], list[str], MultiConnectorEdge[Memlet]]]:
        """
        Build the input plan: a mapping from outer->inner edges to
        ``(outer_edge, lib_conns, tasklet_edge)`` tuples.

        Multiple tasklet inputs can be fed by the same outer-to-inner map path
        (the frontend ``c = a + a`` often yields two tasklet connectors from
        one map connector).  We group by the actual outer->inner edge on the
        memlet path instead of relying on connector name rewrites.

        The *tasklet_edge* is carried along so that child classes can use its
        subset for memlet lifting (e.g. contiguous outer subset in the masked
        case).
        """
        input_plan: dict[tuple, tuple] = {}
        for tasklet_conn, lib_conns in input_slots.items():
            tasklet_edges = list(self._graph.in_edges_by_connector(self._tasklet_node, tasklet_conn))
            if not tasklet_edges:
                continue
            tasklet_edge = tasklet_edges[0]
            outer_to_inner = self._find_path_edge(tasklet_edge, self._outer_entry, self._inner_entry)
            if outer_to_inner is None:
                continue
            # Connector names can be None for multiple distinct edges.
            # Include data name to avoid collapsing different inputs
            # (e.g., A and B in self-write kernels) into a single staging path.
            edge_key = (
                outer_to_inner.src_conn,
                outer_to_inner.dst_conn,
                cast(Optional[str], outer_to_inner.data.data),
            )
            if edge_key not in input_plan:
                input_plan[edge_key] = (outer_to_inner, [], tasklet_edge)
            input_plan[edge_key][1].extend(lib_conns)
        return input_plan

    # ---- transient creation -------------------------------------------------

    # ---- memlet construction hooks ------------------------------------------

    def _build_memlet(self, map_edge: MultiConnectorEdge[Memlet], tasklet_edge: MultiConnectorEdge[Memlet]) -> Memlet:
        """
        Build the memlet from *map_edge* to the tile transient.

        We preserve original outer indexing by deep-copying the memlet.
        The two-edge pattern (outer_entry -> transient -> lib_node) already
        ensures correct data mapping: each edge has its own data+subset
        pair, so ``other_subset`` is not needed.

        The canonical default deep-copies the original outer->inner memlet.
        Non-canonical child classes override this to build a contiguous outer
        subset that covers the full bounding tile footprint.
        """
        new_memlet = copy.deepcopy(map_edge.data)
        return new_memlet

    def _build_input_staging_memlet(self, outer_edge: MultiConnectorEdge[Memlet], tasklet_edge: MultiConnectorEdge[Memlet]) -> Memlet:
        """
        Build the memlet from *outer_entry* to the input tile transient.

        Delegates to ``_build_memlet`` by default.  Non-canonical child
        classes override ``_build_memlet`` to build a contiguous outer
        subset that covers the full bounding tile footprint.
        """
        return self._build_memlet(outer_edge, tasklet_edge)

    def _build_output_store_memlet(self, inner_to_outer_edge: MultiConnectorEdge[Memlet], tasklet_out_edge: MultiConnectorEdge[Memlet]) -> Memlet:
        """
        Build the memlet from the output tile transient to *outer_exit*.

        Symmetric to ``_build_input_staging_memlet``: delegates to
        ``_build_memlet`` by default.
        """
        return self._build_memlet(inner_to_outer_edge, tasklet_out_edge)

    # ---- input lowering -----------------------------------------------------

    def _add_input_transient_and_connect_edges(self, outer_edge: MultiConnectorEdge[Memlet], lib_conns: list[str], tasklet_edge: MultiConnectorEdge[Memlet]) -> tuple[str, nodes.AccessNode]:
        """
        Create an input tile transient and wire:
          outer_entry -> transient -> library_node (for each lib_conn).

        The outer map slice is staged through a tile-shaped transient so the
        library node receives a contiguous tile rather than individual scalars.
        The library node consumes the full tile domain via ``tile_subset``.

        Returns ``(trans_name, trans_read)``.
        """
        data_name = cast(str, outer_edge.data.data)
        trans_name, trans_read = create_tile_transient(self._sdfg, self._graph, data_name, self._tile_shape)

        # outer_entry -> transient (staging memlet preserves outer indexing).
        staging_memlet = self._build_input_staging_memlet(outer_edge, tasklet_edge)
        self._graph.add_edge(self._outer_entry, outer_edge.src_conn,
                             trans_read, None, staging_memlet)

        # The library node consumes the full tile domain.
        for lib_conn in lib_conns:
            self._graph.add_edge(trans_read, None, self._library_node, lib_conn,
                                 memlet_with_primary_subset(trans_name,
                                                            self._tile_subset,
                                                            data_on_src=True))

        return trans_name, trans_read

    # ---- output lowering ----------------------------------------------------

    def _add_output_preload(self, data_name: str, inner_to_outer_edge: MultiConnectorEdge[Memlet], tasklet_out_edge: MultiConnectorEdge[Memlet]) -> None:
        """
        Hook for adding a preload path of existing output values.

        No-op by default.  Child classes may override to inject additional
        dataflow before the output store.
        """
        pass

    def _add_output_transient_and_connect_edges(self, inner_to_outer_edge: MultiConnectorEdge[Memlet], lib_conn: str, tasklet_out_edge: MultiConnectorEdge[Memlet]) -> tuple[str, nodes.AccessNode]:
        """
        Create an output tile transient and wire:
          library_node -> transient -> outer_exit.

        Symmetric to input lowering: the library node writes the full tile
        result into a transient, which is then stored back to the outer map
        footprint.

        Returns ``(trans_name, trans_write)``.
        """
        data_name = cast(str, inner_to_outer_edge.data.data)
        trans_name, trans_write = create_tile_transient(self._sdfg, self._graph, data_name, self._tile_shape)

        # Library writes full tile result into transient.
        self._graph.add_edge(self._library_node, lib_conn, trans_write, None,
                             memlet_with_primary_subset(trans_name,
                                                        self._tile_subset,
                                                        data_on_src=False))

        # transient -> outer_exit (store memlet preserves outer indexing).
        store_memlet = self._build_output_store_memlet(inner_to_outer_edge, tasklet_out_edge)
        self._graph.add_edge(trans_write, None, self._outer_exit,
                             inner_to_outer_edge.dst_conn, store_memlet)

        return trans_name, trans_write

    # ---- hooks for child classes --------------------------------------------

    def _configure_library_node(self):
        """
        Configure the library node's properties from the tasklet classification.

        Sets ``op``, ``constant1``, and ``constant2`` directly from the
        classification.  Removes input connectors that are replaced by
        constants or not needed (unary ops have no rhs2).
        """
        self._library_node.op = self._tasklet_classification.op

        if self._tasklet_classification.constant1 is not None:
            self._library_node.constant1 = self._tasklet_classification.constant1
            if self._node_info.rhs1 and self._node_info.rhs1 in self._library_node.in_connectors:
                self._library_node.remove_in_connector(self._node_info.rhs1)
        if self._tasklet_classification.constant2 is not None:
            self._library_node.constant2 = self._tasklet_classification.constant2
            if self._node_info.rhs2 and self._node_info.rhs2 in self._library_node.in_connectors:
                self._library_node.remove_in_connector(self._node_info.rhs2)
        # Unary ops have no rhs2 in the registry entry; remove any
        # extra input connectors the constructor may have added.
        elif self._node_info.rhs2 is None:
            keep = {self._node_info.rhs1, self._node_info.mask_in, self._node_info.out_in} - {None}
            for conn in list(self._library_node.in_connectors):
                if conn not in keep:
                    self._library_node.remove_in_connector(conn)

    def _post_input_lowering(self):
        """
        Hook called after input lowering, before output lowering.

        Overridden by the masked child to build the mask fill subgraph and
        connect it to the library node.  The subgraph executes inside the
        outer map, because mask predicates can reference tiled-loop symbols
        (e.g. ``tile_i`` / ``tile_j``).  No-op in the canonical case.
        """
        pass

    # ---- cleanup ------------------------------------------------------------

    def _remove_original_nodes(self):
        """
        Remove the now-dead inner map and tasklet nodes.

        After all dataflow edges are rewired through the library node, the
        scalar tasklet and inner-map entry/exit nodes are dead and can be
        safely removed from the state.
        """
        self._graph.remove_node(self._tasklet_node)
        self._graph.remove_node(self._inner_entry)
        self._graph.remove_node(self._inner_exit)

    # ---- main apply (template method) ---------------------------------------

    def apply(self, graph: SDFGState, sdfg: SDFG) -> None:  # type: ignore
        """Apply the transformation: replace the inner scalar map with a tile
        library call.

        ``can_be_applied`` already validated pattern shape and operator
        support; here we perform the actual graph rewrite.  The method is
        structured as a template method — child classes inject behaviour
        through the hooks ``_configure_library_node``,
        ``_build_input_staging_memlet``, ``_build_output_store_memlet``,
        and ``_post_input_lowering``.

        Args:
            graph: The SDFG state containing the matched subgraph.
            sdfg: The top-level SDFG.
        """
        self._set_convenience_variables(sdfg, graph)

        # Resolve the matching library node and classification.
        # The op matcher maps the tasklet code to a concrete cuTile node type.
        op_match = match_tasklet_to_tile_library_node(graph, self._tasklet_node, self._get_mask_type(), promote_scalars=True)
        if op_match is None:
            raise ValueError("No matching library node found for tasklet")
        self._node_info = op_match.node_info
        self._tasklet_classification = op_match.tasklet_classification

        # Derive tile extents from inner-map range sizes using the shared
        # map-range-maximum strategy for both canonical and masked lowering.
        tile_shape = self._try_calculate_tile_shape()
        if tile_shape is None:
            return
        self._tile_shape = tile_shape
        self._tile_subset = tile_subset_from_shape(self._tile_shape)

        # Instantiate the target library node selected by the operation
        # matcher.  Child hook may add extra connectors (mask, preload).
        self._library_node = self._node_info.type(self._node_info.node_name)
        graph.add_node(self._library_node)
        self._configure_library_node()

        # === Input lowering ===
        # For each tasklet input connector:
        #   outer map slice -> tile transient -> library node input.
        self._input_slots = self._get_map_from_tasklet_input_to_libnode_inputs()
        self._input_plan = self._create_input_plan(self._input_slots)

        for outer_edge, lib_conns, tasklet_edge in self._input_plan.values():
            self._add_input_transient_and_connect_edges(outer_edge, lib_conns, tasklet_edge)
            # Remove the original scalar feed from outer map to inner map.
            self._graph.remove_edge(outer_edge)

        # Child hook (e.g. build mask fill subgraph for non-canonical maps).
        self._post_input_lowering()

        # === Output lowering ===
        # Symmetric to input lowering:
        #   library node output -> tile transient -> original outer destination.
        for edge in list(self._graph.in_edges(self._inner_exit)):
            if edge.src is not self._tasklet_node:
                continue
            tasklet_out_edge = edge
            tasklet_conn = edge.src_conn
            lib_conn = self._output_connector_for_tasklet(tasklet_conn)
            if lib_conn is None:
                continue

            # Follow the same memlet path and pick the inner-exit -> outer-exit
            # segment directly instead of inferring connector names.
            inner_to_outer = self._find_path_edge(edge, self._inner_exit, self._outer_exit)
            if inner_to_outer is None:
                continue

            data_name = cast(str, inner_to_outer.data.data)
            # Child hook (e.g. output preload for child classes that need it).
            self._add_output_preload(data_name, inner_to_outer, tasklet_out_edge)
            self._add_output_transient_and_connect_edges(inner_to_outer, lib_conn, tasklet_out_edge)
            # Remove the original scalar store path.
            self._graph.remove_edge(inner_to_outer)

        self._remove_original_nodes()

    # ---- static utility methods (available to all child classes) -------------

    @staticmethod
    def _to_sympy_expr(expr: sp.Basic | SymExpr | int,
                       use_approx: bool = False) -> sp.Basic:
        """Convert DaCe symbolic values (including ``SymExpr``) to plain SymPy.

        Args:
            expr: A SymPy expression, a DaCe :class:`~dace.symbolic.SymExpr`,
                or an integer literal to convert.
            use_approx: If ``True`` and *expr* is a
                :class:`~dace.symbolic.SymExpr`, use ``expr.approx`` instead
                of ``expr.expr``.

        Returns:
            An equivalent plain :class:`sympy.Basic` expression.
        """
        if isinstance(expr, dace.symbolic.SymExpr):
            return expr.approx if use_approx else expr.expr
        return sp.sympify(expr)

    @staticmethod
    def _build_contiguous_outer_subset(tasklet_subset: subsets.Range,
                                       inner_map: nodes.Map,
                                       upper_bounds: Optional[dict[str, sp.Basic]] = None) -> subsets.Range:
        """Lift scalar tasklet accesses to cover the full tile from 0 to max.

        For each accessed dimension, substitutes the inner-map parameter with
        bounds ``0`` (low) and ``end`` (high) and builds a contiguous range
        that covers the full tile footprint starting at index 0.  Maps always
        use a positive step, so the inclusive ``end`` is the high bound; this
        matches the tile shape ``end + 1`` used by ``_calculate_tile_shape``.

        Args:
            tasklet_subset: The scalar :class:`~dace.subsets.Range` on the
                tasklet access memlet to lift.
            inner_map: The inner :class:`~dace.sdfg.nodes.Map` whose parameter
                bounds are used for substitution.
            upper_bounds: Optional mapping from inner-map parameter names to
                explicit upper-bound expressions.  When provided, the given
                value is used instead of the inclusive ``end`` for the
                corresponding parameter.  This is used by the masked path to
                ensure memlets cover the full tile shape.

        Returns:
            A :class:`~dace.subsets.Range` covering the full tile from
            index 0 to the maximum reachable coordinate.

        Raises:
            TypeError: If *tasklet_subset* is not a
                :class:`~dace.subsets.Range`.
            ValueError: If a subset dimension depends on more than one
                inner-map parameter.
        """
        if not isinstance(tasklet_subset, subsets.Range):
            raise TypeError("Expected range subset on tasklet memlet.")

        # Tile covers from 0 to the upper bound per dimension.
        # When upper_bounds is provided, use the caller-supplied value;
        # otherwise fall back to the inclusive ``end`` (maps use a positive
        # step, so end is the high bound) to match the tile shape
        # ``end + 1`` from _calculate_tile_shape.
        param_bounds = {}
        for d, (pname, (start, end, _)) in enumerate(
                zip(inner_map.params, inner_map.range)):
            pname_str = str(pname)
            if upper_bounds is not None and pname_str in upper_bounds:
                high = upper_bounds[pname_str]
            else:
                high = _ScalarToTileBase._to_sympy_expr(end)
            param_bounds[pname_str] = (sp.Integer(0), high)

        new_ranges = []
        for rng in tasklet_subset:
            expr = _ScalarToTileBase._to_sympy_expr(rng[0])
            expr_symbols = {str(s): s for s in expr.free_symbols}
            used_params = [str(p) for p in inner_map.params if str(p) in expr_symbols]

            if len(used_params) > 1:
                # We currently support one inner-map symbol per dimension.
                # Multiple symbols imply coupling that is not representable as
                # a simple axis-aligned contiguous range.
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

            # Require affine dependence and preserve coefficient as stride.
            # TODO: Is second derivative actually a requirement?
            second_derivative = sp.simplify(sp.diff(expr, map_symbol, 2))
            if second_derivative != 0:
                raise ValueError(
                    "Tasklet subset dimension is not affine in inner-map parameter."
                )
            stride_expr = sp.Abs(sp.simplify(sp.diff(expr, map_symbol)))
            if stride_expr == 0:
                raise ValueError(
                    "Tasklet subset dimension is independent after simplification."
                )

            # Evaluate access expression at both map extremes.
            low_expr = expr.subs(map_symbol, low)
            high_expr = expr.subs(map_symbol, high)
            new_ranges.append((sp.Min(low_expr, high_expr),
                               sp.Max(low_expr, high_expr),
                               stride_expr))

        return subsets.Range(new_ranges)

    def _map_param_bounds(self) -> dict[str, tuple[sp.Basic, sp.Basic]]:
        """Return symbolic min/max bounds for inner and outer map parameters.

        Maps always use a positive step (descending maps are rejected by SDFG
        validation), so ``start`` is the lower bound and the inclusive ``end``
        is the upper bound directly.
        """
        bounds: dict[str, tuple[sp.Basic, sp.Basic]] = {}
        for params, rng in ((self._inner_entry.map.params, self._inner_entry.map.range),
                            (self._outer_entry.map.params, self._outer_entry.map.range)):
            for pname, (start, end, _step) in zip(params, rng):
                s = self._to_sympy_expr(start)
                e = self._to_sympy_expr(end)
                bounds[str(pname)] = (s, e)
        return bounds

    def _expr_uses_outer_map_param(self, expr: sp.Basic) -> bool:
        """Return True iff expr references at least one outer-map parameter."""
        expr_symbols = {str(s) for s in expr.free_symbols}
        return any(str(p) in expr_symbols for p in self._outer_entry.map.params)

    def _expr_range_over_map_params(self, expr: sp.Basic) -> tuple[sp.Basic, sp.Basic]:
        """Conservatively evaluate expr min/max over used outer+inner map params."""
        bounds = self._map_param_bounds()
        symtab = {str(s): s for s in expr.free_symbols}
        used = [name for name in bounds.keys() if name in symtab]
        if not used:
            return expr, expr

        candidates: list[sp.Basic] = []
        for corner in itertools.product((0, 1), repeat=len(used)):
            repl = {}
            for bit, name in zip(corner, used):
                lo, hi = bounds[name]
                repl[symtab[name]] = hi if bit else lo
            candidates.append(expr.subs(repl))

        return sp.Min(*candidates), sp.Max(*candidates)

    def _infer_implicit_stride_factor(self, expr: sp.Basic, data_dim: int) -> int:
        """Infer hidden affine stride from compressed iteration-space cardinality.

        Returns 1 when no reliable stride can be inferred.
        """
        if not self._expr_uses_outer_map_param(expr):
            return 1

        low, high = self._expr_range_over_map_params(expr)
        low_s = sp.simplify(low)
        extent = sp.simplify(high - low + 1)
        if not (isinstance(data_dim, int) and data_dim > 1 and extent.is_integer):
            return 1

        try:
            extent_i = int(extent)
        except TypeError:
            return 1
        if extent_i <= 1 or data_dim <= extent_i:
            return 1

        # Candidate stride from ceil(data_dim / extent).
        stride = (data_dim + extent_i - 1) // extent_i
        if stride <= 1:
            return 1
        # Validate by inverse cardinality and 0-based alignment.
        if (data_dim + stride - 1) // stride != extent_i:
            return 1
        if low_s != 0:
            return 1
        return stride

    @staticmethod
    def _build_mask_condition_symbolic(inner_map: nodes.Map,
                                       outer_map: Optional[nodes.Map] = None) -> sp.Basic:
        """Build a SymPy boolean expression for tile-point validity.

        Returns a conjunction of per-dimension predicates using ``__m0``,
        ``__m1``, ... as coordinate symbols.  Maps always use a positive step
        (descending maps are rejected by SDFG validation; symbolic steps carry
        a runtime ``step > 0`` assertion), so each dimension has a single
        predicate::

            m >= start  AND  m <= end  AND  (m - start) % step == 0

        where ``m = __m{d}``.  The tile covers ``0..end`` per dimension, so the
        mask checks ``m`` against ``start`` and ``end`` directly.

        Args:
            inner_map: The inner :class:`~dace.sdfg.nodes.Map` whose range
                encodes the valid coordinate set (after skewing).
            outer_map: Optional outer (tiled) :class:`~dace.sdfg.nodes.Map`.
                When given, its parameters are used as skew offsets.

        Returns:
            A SymPy boolean expression over ``__m0``, ``__m1``, ... symbols
            that is ``True`` exactly for tile coordinates that correspond to
            points in the original inner-map iteration space.
        """
        dim_conds: list[sp.Basic] = []

        for d, (start, end, step) in enumerate(inner_map.range):
            m = sp.Symbol(f"__m{d}")
            start = _ScalarToTileBase._to_sympy_expr(start)
            end = _ScalarToTileBase._to_sympy_expr(end)
            step = _ScalarToTileBase._to_sympy_expr(step)

            # m must lie within [start, end] and align to the step grid
            # starting at start.
            dim_conds.append(sp.And(
                sp.GreaterThan(m, start),          # m >= start
                sp.LessThan(m, end),               # m <= end
                sp.Eq(sp.Mod(m - start, step), 0),
            ))

        if not dim_conds:
            return sp.true
        if len(dim_conds) == 1:
            return dim_conds[0]
        return sp.And(*dim_conds)


# =========================================================================
# Concrete transformation classes
# =========================================================================


class ScalarToTileCanonical(_ScalarToTileBase):
    """
    Lower canonical scalar inner maps to unmasked cuTile library nodes.

    Canonical pattern: inner map is 0-based, unit-stride in every dimension.
    Data is staged through tile-shaped transients to convert scalar memlets
    into full-tile memlets consumed/produced by the library node.  All
    base-class defaults (memlet deep-copy, no mask, no preload) apply
    directly.
    """

    def _has_valid_inner_map_ranges(self) -> bool:
        """Return ``True`` only when all inner-map ranges are canonical.

        Canonical means every dimension starts at ``0`` with stride ``1``.
        Non-canonical ranges (offset start, non-unit stride) are handled by
        :class:`ScalarToTileMasked` instead.

        Returns:
            ``True`` if every dimension satisfies ``start == 0`` and
            ``step == 1``.
        """
        ranges = self._inner_entry.map.range
        if not isinstance(ranges, subsets.Range):
            return False
        for rng in ranges:
            start, end, step = rng
            if not (start == 0 and step == 1):
                return False
        return True

    def _has_identity_tasklet_indexing(self) -> bool:
        """Require identity indexing w.r.t. inner-map parameters.

        Canonical lowering assumes pointwise alignment between inner-map
        coordinates and array accesses. Scaled accesses (e.g. ``2*i``) are
        handled by the masked path.
        """
        for edge in self._get_all_edges_of_node(self._tasklet_node):
            subset = primary_memlet_subset(edge.data)
            if not isinstance(subset, subsets.Range):
                return False

            data_name = cast(Optional[str], edge.data.data)
            data_desc = self._sdfg.arrays.get(data_name) if data_name is not None else None

            for dim, rng in enumerate(subset):
                expr = self._to_sympy_expr(rng[0])
                expr_symbols = {str(s): s for s in expr.free_symbols}
                used_params = [str(p) for p in self._inner_entry.map.params if str(p) in expr_symbols]

                if len(used_params) > 1:
                    return False
                if not used_params:
                    continue

                p = expr_symbols[used_params[0]]
                if sp.simplify(sp.diff(expr, p, 2)) != 0:
                    return False
                if sp.simplify(sp.diff(expr, p)) != 1:
                    return False

                if data_desc is not None and dim < len(data_desc.shape):
                    try:
                        data_dim = int(data_desc.shape[dim])
                    except TypeError:
                        data_dim = -1
                    if data_dim > 0 and self._infer_implicit_stride_factor(expr, data_dim) > 1:
                        return False

        return True

    def can_be_applied(self, graph: SDFGState, expr_index: int,
                       sdfg: SDFG, permissive: bool = False) -> bool:
        if not super().can_be_applied(graph, expr_index, sdfg, permissive):
            return False
        return self._has_identity_tasklet_indexing()

    def _get_mask_type(self) -> MaskType:
        """Return ``MaskType.UNMASKED`` since canonical maps need no mask."""
        return MaskType.UNMASKED


class ScalarToTileMasked(_ScalarToTileBase):
    """
    Lower non-canonical scalar inner maps to symbolic-masked cuTile library
    nodes.

    Non-canonical ranges (offset starts, strided bounds) are mapped to a
    zero-based tile whose shape is ``end + 1`` per dimension.  For skewed maps
    (produced by ``MapTiling(skew=True)``), the tile shape is extended to
    ``outer_step / inner_step`` to recover the original ``tile_size``
    parameter, ensuring power-of-2 compatibility with the cuTile
    Python backend.  A symbolic condition is embedded directly in the library
    node and evaluated per element during expansion — no runtime mask array is
    allocated or filled.

    Masked-out lanes must preserve their original values.  This is achieved
    by preloading the current output tile into the library node's ``_c_in``
    connector before the operation executes.  The library node's expansion
    emits ``else { _c[i] = _c_in[i]; }`` for masked-out elements.

    The library node's ``mask_condition`` property is set to a SymPy expression
    that uses ``__m0``, ``__m1``, … as tile coordinate variables.
    """

    def _calculate_tile_shape(self) -> tuple[sp.Basic | int, ...]:
        """Tile shape from inner-map range, extended for skewed maps.

        Starts with the base-class shape (``end + 1`` per dimension), then for
        skewed dimensions (``start == 0``, indicating ``MapTiling(skew=True)``
        was applied) extends the tile to ``outer_step / inner_step``.  This
        recovers the original ``tile_size`` parameter from ``MapTiling``, which
        is guaranteed to be a power of 2 when the pipeline requests it.

        The extension is safe because:

        * Load/store memlets still cover only the valid range (from
          ``_build_contiguous_outer_subset``), so no out-of-bounds memory
          accesses occur.
        * The mask condition excludes the extra tile positions.
        * The extra positions in the tile transient are padding — they are
          never stored back to global memory.
        * For the C++ expansion path, the extra positions contain
          uninitialized memory.  The mask condition excludes them from
          computation, and the store memlet limits the writeback range,
          so they do not affect correctness.

        .. warning::

            The ``start == 0`` heuristic for detecting skewed maps is
            fragile.  It assumes that a zero-based inner-map start implies
            ``MapTiling(skew=True)`` was applied.  If a user manually
            constructs a zero-based non-unit-stride map without an outer
            tiling map, this heuristic would incorrectly try to derive a
            tile size from the outer map step — but the ``sp.Max`` guard
            ensures the tile never shrinks, only grows.
        """
        base_shape = list(super()._calculate_tile_shape())

        for d, (start, end, step) in enumerate(self._inner_entry.map.range):
            start_expr = self._to_sympy_expr(start)
            step_expr = self._to_sympy_expr(step)

            # Detect skewed dimension: start == 0 suggests MapTiling(skew=True)
            # subtracted the outer map parameter from the original start.
            if (start_expr == sp.Integer(0)
                    and d < len(self._outer_entry.map.range)):
                _, _, outer_step = self._outer_entry.map.range[d]
                outer_step_expr = self._to_sympy_expr(outer_step)
                # Recover original tile_size: outer_step = tile_size * inner_step
                # (inner_step is always positive; descending maps are rejected).
                tile_from_outer = outer_step_expr / step_expr
                if not tile_from_outer.is_Number:
                    tile_from_outer = sp.simplify(tile_from_outer)
                # Take the larger of the base shape and derived tile size.
                # This ensures we never shrink the tile, only extend it.
                base_shape[d] = sp.Max(base_shape[d], tile_from_outer)

        return tuple(base_shape)

    def _clamp_subset_to_array_bounds(
        self,
        subset: subsets.Range,
        data_name: str,
    ) -> subsets.Range:
        """Clamp each dimension's upper bound to the array descriptor's size.

        Global-to-tile staging memlets must stay within array bounds for
        DaCe validation.  The tile transient may be larger than the
        remaining array slice (e.g. boundary tiles or non-aligned
        offsets), so we clamp each range dimension with
        ``Min(unclamped_upper, array_dim_size - 1)``.

        The cuTile runtime handles actual boundary masking -- loads pad
        out-of-bounds lanes (``padding_mode``), and stores silently
        ignore them.

        :param subset: The unclamped :class:`~dace.subsets.Range`.
        :param data_name: Name of the global array in
            ``self._sdfg.arrays``.
        :returns: A new :class:`~dace.subsets.Range` with clamped upper
            bounds.
        """
        desc = self._sdfg.arrays.get(data_name)
        if desc is None:
            return subset
        clamped_ranges: list[tuple[sp.Basic, sp.Basic, sp.Basic]] = []
        for dim_i, (low, high, step) in enumerate(subset):
            if dim_i < len(desc.shape):
                array_max = self._to_sympy_expr(desc.shape[dim_i]) - 1
                clamped_high = sp.Min(
                    self._to_sympy_expr(high), array_max
                )
                clamped_ranges.append((low, clamped_high, step))
            else:
                clamped_ranges.append((low, high, step))
        return subsets.Range(clamped_ranges)

    def _has_valid_inner_map_ranges(self) -> bool:
        """Return ``True`` for inner maps with non-zero strides.

        Accepts any map range as long as no dimension has a zero step
        (which would represent an infinite loop).  Canonical maps are
        not explicitly rejected here; :class:`ScalarToTileCanonical`
        has priority via ``order_by_transformation`` in the pipeline.

        Returns:
            ``True`` if every dimension has a non-zero step.
        """
        for start, end, step in self._inner_entry.map.range:
            if step == 0:
                return False
        return True

    def _get_mask_type(self) -> MaskType:
        """Return ``MaskType.SYMBOLIC`` for symbolic-mask tile ops."""
        return MaskType.SYMBOLIC

    @staticmethod
    def _is_affine_in_symbol(expr: sp.Basic, symbol: sp.Symbol) -> bool:
        """Return whether *expr* is affine in *symbol*.

        Endpoint-only bounding in ``_build_contiguous_outer_subset`` is safe
        for affine access expressions because extrema lie on map endpoints.
        Non-affine accesses can have interior extrema and must be rejected.
        """
        try:
            poly = sp.Poly(sp.expand(expr), symbol)
        except (PolynomialError, TypeError, ValueError):
            return False
        return poly.total_degree() <= 1

    def _has_safe_contiguous_subset_bounding(self) -> bool:
        """Conservatively validate that endpoint-only subset bounding is safe.

        The masked path lifts scalar tasklet accesses to a contiguous outer
        subset by evaluating access expressions at map endpoints. That is only
        sound when each accessed dimension is affine in at most one inner-map
        parameter. If this check fails we skip the transformation.
        """
        inner_param_names = {str(pname) for pname in self._inner_entry.map.params}

        for edge in self._get_all_edges_of_node(self._tasklet_node):
            edge_subset = primary_memlet_subset(edge.data)
            if not isinstance(edge_subset, subsets.Range):
                return False

            for rng in edge_subset:
                expr = self._to_sympy_expr(rng[0])
                expr_symbols = {str(symbol): symbol for symbol in expr.free_symbols}
                used_inner_params = [pname for pname in inner_param_names if pname in expr_symbols]

                # Multiple inner-map parameters in one dimension are not
                # representable by a simple axis-aligned contiguous bound.
                if len(used_inner_params) > 1:
                    return False

                if len(used_inner_params) == 1:
                    map_symbol = expr_symbols[used_inner_params[0]]
                    if not self._is_affine_in_symbol(expr, map_symbol):
                        return False

        return True

    def can_be_applied(self, graph: SDFGState, expr_index: int,
                       sdfg: SDFG, permissive: bool = False) -> bool:
        """Apply base checks and reject masked lowering when bounding is unsafe.

        This is a fail-soft guard: unsafe non-affine/non-monotone index
        expressions simply keep the original scalar form.
        """
        if not super().can_be_applied(graph, expr_index, sdfg, permissive):
            return False

        return self._has_safe_contiguous_subset_bounding()

    def _build_memlet(self, map_edge: MultiConnectorEdge[Memlet], tasklet_edge: MultiConnectorEdge[Memlet]) -> Memlet:
        """Build a Min-clamped contiguous outer-subset memlet for the tile.

        cuTile always loads and stores the entire tile, so memlets must
        cover the full tile shape -- not just the inner-map's reachable
        range.  Upper bounds for each inner-map parameter are derived
        from ``self._tile_shape`` to ensure the memlet and transient
        agree.

        Because tile shapes may exceed the remaining array extent at
        boundary tiles (e.g. when the array size is not a multiple of
        the tile size), the resulting subset is clamped with
        ``Min(unclamped_upper, array_dim_size - 1)`` via
        :meth:`_clamp_subset_to_array_bounds`.  This keeps the memlet
        within array bounds for DaCe validation while the cuTile runtime
        handles actual boundary masking (load padding, store ignoring).

        Args:
            map_edge: The outer-entry-to-inner-entry edge whose ``data.data``
                names the source array.
            tasklet_edge: The tasklet access edge whose subset provides the
                scalar index expression to lift.

        Returns:
            A :class:`~dace.Memlet` with the lifted, Min-clamped contiguous
            outer subset covering the full tile shape.
        """
        tasklet_subset = primary_memlet_subset(tasklet_edge.data)
        if tasklet_subset is None:
            raise ValueError("ScalarToTileMasked expects tasklet memlets to define a subset")

        # Derive upper bounds from tile shape so memlets cover the full tile.
        upper_bounds = {
            str(pname): self._to_sympy_expr(self._tile_shape[d]) - 1
            for d, pname in enumerate(self._inner_entry.map.params)
            if d < len(self._tile_shape)
        }

        data_name = map_edge.data.data
        load_subset = self._build_contiguous_outer_subset(
            tasklet_subset, self._inner_entry.map, upper_bounds=upper_bounds)
        clamped_subset = self._clamp_subset_to_array_bounds(load_subset, data_name)
        return with_primary_subset(map_edge.data, clamped_subset)

    def _configure_library_node(self) -> None:
        """Configure the library node and set the symbolic mask condition.

        Calls the base class to set ``op`` and constants, then computes
        the SymPy mask condition from the inner map ranges and stores it
        on the library node.
        """
        super()._configure_library_node()
        condition = self._build_mask_condition_symbolic(
            self._inner_entry.map, outer_map=self._outer_entry.map)
        self._library_node.mask_condition = condition

    def _add_output_preload(self, data_name: str, inner_to_outer_edge: MultiConnectorEdge[Memlet], tasklet_out_edge: MultiConnectorEdge[Memlet]) -> None:
        """Preload existing output values so masked-out lanes are preserved.

        Creates a tile-shaped transient that reads the current values from
        the output array and feeds them to the library node's ``_c_in``
        connector.  The library node's expansion uses these values for
        masked-out elements (``else { _c[i] = _c_in[i]; }``).

        The preload read memlet is Min-clamped via
        :meth:`_clamp_subset_to_array_bounds` so that boundary tiles
        (where the tile extends past the array extent) stay within
        array bounds for DaCe validation.

        Wiring::

            output_array_read -> outer_entry[new_conn] -> preload_transient -> lib_node._c_in

        Args:
            data_name: Name of the output array whose current values are
                preloaded.
            inner_to_outer_edge: The inner-exit-to-outer-exit edge used to
                derive the outer subset for the preload read.
            tasklet_out_edge: The tasklet output edge used to derive the
                scalar subset for the contiguous-outer-subset computation.
        """
        preload_connector = self._node_info.out_in or "_c_in"

        # 1. Add preload connector to the library node.
        self._library_node.add_in_connector(preload_connector)

        # 2. Create a preload tile transient (same shape/dtype as output).
        #    We look up the descriptor from the original output array and use
        #    a distinct name to avoid confusion with the output tile transient.
        data_desc = self._sdfg.arrays[data_name]
        preload_name = self._sdfg._find_new_name(data_name + "_preload_tile")
        self._sdfg.add_transient(
            preload_name,
            shape=self._tile_shape,
            dtype=data_desc.dtype,
            storage=data_desc.storage,
            lifetime=dtypes.AllocationLifetime.Scope,
        )
        preload_node = self._graph.add_access(preload_name)

        # 3. Build the outer subset for reading current values (same range
        #    the store memlet will use).
        tasklet_subset = primary_memlet_subset(tasklet_out_edge.data)
        if tasklet_subset is None:
            raise ValueError("ScalarToTileMasked expects output tasklet memlets to define a subset")
        # Use tile-shape-derived upper bounds so the preload covers the
        # full tile -- matching the store memlet built by _build_memlet.
        upper_bounds = {
            str(pname): self._to_sympy_expr(self._tile_shape[d]) - 1
            for d, pname in enumerate(self._inner_entry.map.params)
            if d < len(self._tile_shape)
        }
        outer_subset = self._build_contiguous_outer_subset(
            tasklet_subset, self._inner_entry.map, upper_bounds=upper_bounds)
        outer_subset = self._clamp_subset_to_array_bounds(outer_subset, data_name)

        # 4. Add a new input connector pair on outer_entry.
        conn_base = self._outer_entry.next_connector("preload")
        in_conn = "IN_" + conn_base
        out_conn = "OUT_" + conn_base
        self._outer_entry.add_in_connector(in_conn)
        self._outer_entry.add_out_connector(out_conn)

        # 5. Wire: output_array_read -> outer_entry
        ext_read = self._graph.add_access(data_name)
        preload_memlet = memlet_with_primary_subset(data_name,
                                outer_subset,
                                data_on_src=True)
        self._graph.add_edge(ext_read, None, self._outer_entry, in_conn,
                     preload_memlet)

        # 6. Wire: outer_entry -> preload_transient
        self._graph.add_edge(self._outer_entry, out_conn, preload_node, None,
                     copy.deepcopy(preload_memlet))

        # 7. Wire: preload_transient -> library_node preload connector
        self._graph.add_edge(preload_node, None, self._library_node, preload_connector,
                     memlet_with_primary_subset(preload_name,
                                                self._tile_subset,
                                                data_on_src=True))
