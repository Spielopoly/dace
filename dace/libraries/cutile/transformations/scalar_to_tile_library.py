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
from typing import Dict, Optional, cast

import dace
import sympy as sp
from dace import Memlet, dtypes, subsets
from dace.sdfg import SDFG, SDFGState, nodes, utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.scope import ScopeSubgraphView
from dace.symbolic import SymExpr, symstr
from dace.transformation import transformation as xf

from dace.libraries.cutile.op_registry import match_tasklet_to_tile_library_node, MaskType
from dace.libraries.cutile.transformations.utils import tile_subset_from_shape, create_tile_transient, is_canonical_inner_map


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
            if not isinstance(edge.data.subset, subsets.Range):
                return False
            for rng in edge.data.subset:
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
        (including reversed and offset ranges).
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

        return True

    # ---- tile shape / subset helpers ----------------------------------------

    def _calculate_tile_shape(self) -> tuple[sp.Basic | int, ...]:
        """
        Tile shape from the inner map range.

        The canonical default uses the range sizes directly.  Non-canonical
        child classes override this with bounding-tile logic.
        """
        assert isinstance(self._inner_entry.map.range, subsets.Range)
        return tuple(self._inner_entry.map.range.size())

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
            edge_key = (outer_to_inner.src_conn, outer_to_inner.dst_conn)
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
                                 Memlet(data=trans_name, subset=self._tile_subset))

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
                             Memlet(data=trans_name, subset=self._tile_subset))

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

        # Derive the tile extents from the inner map.  In canonical mode this
        # is the concrete domain; in masked mode it is the bounding tile.
        self._tile_shape = self._calculate_tile_shape()
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
    def _to_sympy_expr(expr: sp.Basic | SymExpr | int) -> sp.Basic:
        """Convert DaCe symbolic values (including ``SymExpr``) to plain SymPy.

        Args:
            expr: A SymPy expression, a DaCe :class:`~dace.symbolic.SymExpr`,
                or an integer literal to convert.

        Returns:
            An equivalent plain :class:`sympy.Basic` expression.
        """
        if isinstance(expr, dace.symbolic.SymExpr):
            return expr.expr
        return sp.sympify(expr)

    @staticmethod
    def _bounding_tile_shape(inner_map: nodes.Map) -> list[sp.Basic | int]:
        """Compute per-dimension extents of a bounding box for an inner map range.

        Using ``Min``/``Max`` handles both increasing and decreasing ranges
        uniformly.  The result is symbolic and may include expressions that
        are only resolved at runtime.

        Args:
            inner_map: The inner :class:`~dace.sdfg.nodes.Map` whose range to
                compute the bounding box from.

        Returns:
            A list of per-dimension extents (symbolic or integer).
        """
        shape = []
        for start, end, _ in inner_map.range:
            start = _ScalarToTileBase._to_sympy_expr(start)
            end = _ScalarToTileBase._to_sympy_expr(end)
            low = sp.Min(start, end)
            high = sp.Max(start, end)
            shape.append(high - low + 1)
        return shape

    @staticmethod
    def _build_contiguous_outer_subset(tasklet_subset: subsets.Range,
                                       inner_map: nodes.Map) -> subsets.Range:
        """Lift scalar tasklet accesses to a contiguous outer subset.

        For each accessed dimension, substitutes the inner-map parameter with
        its min/max reachable values and builds a conservative contiguous range.
        This converts the scalar index expression(s) used in the inner map to
        a contiguous outer tile range that covers all points visited by the
        inner map.

        Args:
            tasklet_subset: The scalar :class:`~dace.subsets.Range` on the
                tasklet access memlet to lift.
            inner_map: The inner :class:`~dace.sdfg.nodes.Map` whose parameter
                bounds are used for substitution.

        Returns:
            A :class:`~dace.subsets.Range` covering all outer-space points
            reachable from the inner-map iteration.

        Raises:
            TypeError: If *tasklet_subset* is not a
                :class:`~dace.subsets.Range`.
            ValueError: If a subset dimension depends on more than one
                inner-map parameter.
        """
        if not isinstance(tasklet_subset, subsets.Range):
            raise TypeError("Expected range subset on tasklet memlet.")

        # Precompute symbolic min/max bounds for each inner-map parameter so
        # we can safely evaluate accesses even for reversed iteration ranges.
        param_bounds = {
            str(pname): (
                sp.Min(
                    _ScalarToTileBase._to_sympy_expr(start),
                    _ScalarToTileBase._to_sympy_expr(end),
                ),
                sp.Max(
                    _ScalarToTileBase._to_sympy_expr(start),
                    _ScalarToTileBase._to_sympy_expr(end),
                ),
            )
            for pname, (start, end, _) in zip(inner_map.params, inner_map.range)
        }

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

            # Evaluate access expression at both map extremes.
            low_expr = expr.subs(map_symbol, low)
            high_expr = expr.subs(map_symbol, high)
            new_ranges.append((sp.Min(low_expr, high_expr),
                               sp.Max(low_expr, high_expr),
                               1))

        return subsets.Range(new_ranges)

    @staticmethod
    def _build_mask_condition_symbolic(inner_map: nodes.Map) -> sp.Basic:
        """Build a SymPy boolean expression for tile-point validity.

        Returns a conjunction of per-dimension predicates using ``__m0``,
        ``__m1``, … as coordinate symbols.  Each sub-clause handles both
        positive and negative step directions (combined with ``Or``).

        For positive step (low = start)::

            __m <= end - start  AND  __m % step == 0

        For negative step (low = end)::

            __m <= start - end  AND  (start - end - __m) % (-step) == 0

        Args:
            inner_map: The inner :class:`~dace.sdfg.nodes.Map` whose range
                encodes the valid coordinate set.

        Returns:
            A SymPy boolean expression over ``__m0``, ``__m1``, … symbols
            that is ``True`` exactly for tile coordinates that correspond to
            points in the original inner-map iteration space.
        """
        dim_conds: list[sp.Basic] = []

        for d, (start, end, step) in enumerate(inner_map.range):
            m = sp.Symbol(f"__m{d}")
            start = _ScalarToTileBase._to_sympy_expr(start)
            end = _ScalarToTileBase._to_sympy_expr(end)
            step = _ScalarToTileBase._to_sympy_expr(step)

            # Positive-step sub-clause
            cond_pos = sp.And(
                sp.StrictGreaterThan(step, 0),
                sp.LessThan(m, end - start + 1),
                sp.Eq(sp.Mod(m, step), 0),
            )
            # Negative-step sub-clause
            cond_neg = sp.And(
                sp.StrictLessThan(step, 0),
                sp.LessThan(m, start - end + 1),
                sp.Eq(sp.Mod(start - end - m, -step), 0),
            )
            dim_conds.append(sp.Or(cond_pos, cond_neg))

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

    def _get_mask_type(self) -> MaskType:
        """Return ``MaskType.UNMASKED`` since canonical maps need no mask."""
        return MaskType.UNMASKED


class ScalarToTileMasked(_ScalarToTileBase):
    """
    Lower non-canonical scalar inner maps to symbolic-masked cuTile library
    nodes.

    Non-canonical ranges (offset starts, negative/strided bounds) are mapped
    to a bounding tile.  A symbolic condition is embedded directly in the
    library node and evaluated per element during expansion—no runtime mask
    array is allocated or filled.

    Masked-out lanes must preserve their original values.  This is achieved
    by preloading the current output tile into the library node's ``_c_in``
    connector before the operation executes.  The library node's expansion
    emits ``else { _c[i] = _c_in[i]; }`` for masked-out elements.

    The library node's ``mask_condition`` property is set to a SymPy expression
    that uses ``__m0``, ``__m1``, … as tile coordinate variables.
    """

    def _has_valid_inner_map_ranges(self) -> bool:
        """Return ``True`` for non-canonical inner maps with non-zero strides.

        Accepts any map range as long as no dimension has a zero step (which
        would represent an infinite loop).  Also rejects canonical maps so
        that :class:`ScalarToTileCanonical` takes priority over them.

        Returns:
            ``True`` if every dimension has a non-zero step and the map is
            non-canonical (i.e. :class:`ScalarToTileCanonical` would reject
            it).
        """
        for start, end, step in self._inner_entry.map.range:
            if step == 0:
                return False
        # Only match non-canonical maps.
        if is_canonical_inner_map(self._inner_entry.map):
            return False
        return True

    def _get_mask_type(self) -> MaskType:
        """Return ``MaskType.SYMBOLIC`` for symbolic-mask tile ops."""
        return MaskType.SYMBOLIC

    def _calculate_tile_shape(self) -> tuple:
        """Return the bounding tile shape derived from the inner map ranges.

        The bounding tile is the smallest axis-aligned tile that contains all
        iteration points from the original inner map, regardless of direction.

        Returns:
            A tuple of per-dimension extents (symbolic or integer).
        """
        return tuple(self._bounding_tile_shape(self._inner_entry.map))

    def _build_memlet(self, map_edge: MultiConnectorEdge[Memlet], tasklet_edge: MultiConnectorEdge[Memlet]) -> Memlet:
        """Build a contiguous outer-subset memlet covering the inner map's footprint.

        Converts scalar index expression(s) to a contiguous outer tile range
        that covers all points visited by the inner map.

        Args:
            map_edge: The outer-entry-to-inner-entry edge whose ``data.data``
                names the source array.
            tasklet_edge: The tasklet access edge whose subset provides the
                scalar index expression to lift.

        Returns:
            A :class:`~dace.Memlet` with a contiguous outer subset.
        """
        data_name = cast(str, map_edge.data.data)
        load_subset = self._build_contiguous_outer_subset(
            tasklet_edge.data.subset, self._inner_entry.map)
        return Memlet(data=data_name, subset=load_subset)

    def _configure_library_node(self) -> None:
        """Configure the library node and set the symbolic mask condition.

        Calls the base class to set ``op`` and constants, then computes
        the SymPy mask condition from the inner map ranges and stores it
        on the library node.
        """
        super()._configure_library_node()
        condition = self._build_mask_condition_symbolic(
            self._inner_entry.map)
        self._library_node.mask_condition = condition

    def _add_output_preload(self, data_name: str, inner_to_outer_edge: MultiConnectorEdge[Memlet], tasklet_out_edge: MultiConnectorEdge[Memlet]) -> None:
        """Preload existing output values so masked-out lanes are preserved.

        Creates a tile-shaped transient that reads the current values from
        the output array and feeds them to the library node's ``_c_in``
        connector.  The library node's expansion uses these values for
        masked-out elements (``else { _c[i] = _c_in[i]; }``).

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
        # 1. Add _c_in connector to the library node.
        self._library_node.add_in_connector("_c_in")

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
        outer_subset = self._build_contiguous_outer_subset(
            tasklet_out_edge.data.subset, self._inner_entry.map)

        # 4. Add a new input connector pair on outer_entry.
        conn_base = self._outer_entry.next_connector("preload")
        in_conn = "IN_" + conn_base
        out_conn = "OUT_" + conn_base
        self._outer_entry.add_in_connector(in_conn)
        self._outer_entry.add_out_connector(out_conn)

        # 5. Wire: output_array_read -> outer_entry
        ext_read = self._graph.add_access(data_name)
        self._graph.add_edge(ext_read, None, self._outer_entry, in_conn,
                             Memlet(data=data_name, subset=outer_subset))

        # 6. Wire: outer_entry -> preload_transient
        self._graph.add_edge(self._outer_entry, out_conn, preload_node, None,
                             Memlet(data=data_name, subset=outer_subset))

        # 7. Wire: preload_transient -> library_node._c_in
        self._graph.add_edge(preload_node, None, self._library_node, "_c_in",
                             Memlet(data=preload_name, subset=self._tile_subset))
