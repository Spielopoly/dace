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

    OuterMapEntry -> tile_transient_read -> MaskedLibNode -> tile_transient_write -> OuterMapExit
                    + generated mask transient used to select valid map points
"""
from __future__ import annotations

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
    preload paths, non-canonical memlet construction).

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

    def _set_convenience_variables(self, sdfg: SDFG, graph: SDFGState):
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
        """
        Validate that the matched subgraph is a pure element-wise inner loop.

        The transformation only applies when the inner map has no extra nodes,
        tasklet accesses are scalar, and there is a registered tile operator
        for the tasklet code.
        """
        self._set_convenience_variables(sdfg, graph)

        if not self._has_valid_scope():
            return False
        if not self._is_tasklet_scalar(self._tasklet_node):
            return False
        if not self._has_valid_map_ranges():
            return False
        if match_tasklet_to_tile_library_node(graph, self._tasklet_node, self._get_mask_type()) is None:
            return False

        return True

    # ---- tile shape / subset helpers ----------------------------------------

    def _calculate_tile_shape(self) -> tuple:
        """
        Tile shape from the inner map range.

        The canonical default uses the range sizes directly.  Non-canonical
        child classes override this with bounding-tile logic.
        """
        assert isinstance(self._inner_entry.map.range, subsets.Range)
        return tuple(self._inner_entry.map.range.size())

    @staticmethod
    def _tile_subset_from_shape(tile_shape: tuple[sp.Basic | int, ...]) -> subsets.Range:
        """Build a dense local tile range ``[0, extent-1]`` in every dimension."""
        return subsets.Range([(0, d - 1, 1) for d in tile_shape])

    # ---- input/output slot mapping ------------------------------------------

    def _get_map_from_tasklet_input_to_libnode_inputs(self) -> dict[str, list[str]]:
        """
        Build tasklet-input -> library-input slot mapping.

        A single tasklet connector can map to multiple library slots, e.g.,
        for expressions such as ``c = a + a`` where ``rhs1 == rhs2``.  In that
        case, one tasklet connector name fans out to two library input slots.
        """
        slot_map: dict[str, list[str]] = {}
        classification = self._tasklet_classification
        node_info = self._node_info

        for tasklet_conn, library_conn in zip(
            [classification.rhs1, classification.rhs2, classification.constant1, classification.constant2],
            [node_info.rhs1, node_info.rhs2, node_info.constant1, node_info.constant2]
        ):
            if tasklet_conn is not None and library_conn is not None:
                slot_map.setdefault(tasklet_conn, []).append(library_conn)
            elif tasklet_conn is not None and library_conn is None:
                raise ValueError(
                    f"Invalid mapping: tasklet connector {tasklet_conn}, "
                    f"library connector {library_conn}"
                )
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

    def _create_and_add_tile_transient(self, data_name: str, tile_shape: tuple[sp.Basic | int, ...]) -> tuple[str, nodes.AccessNode]:
        """
        Create a scope-lifetime transient tile to stage an operand.

        Scope lifetime ensures no state escapes outside this map rewrite.
        The dtype and storage are inherited from the original array
        descriptor *data_name*.

        Returns ``(trans_name, trans_node)``.
        """
        data_desc = self._sdfg.arrays[data_name]
        trans_name = self._sdfg._find_new_name(data_name + "_tile")
        self._sdfg.add_transient(
            trans_name,
            shape=tile_shape,
            dtype=data_desc.dtype,
            storage=data_desc.storage,
            lifetime=dtypes.AllocationLifetime.Scope,
        )
        trans_node = self._graph.add_access(trans_name)
        return trans_name, trans_node

    # ---- memlet construction hooks ------------------------------------------

    def _build_input_staging_memlet(self, outer_edge: MultiConnectorEdge[Memlet], tasklet_edge: MultiConnectorEdge[Memlet]) -> Memlet:
        """
        Build the memlet from *outer_entry* to the input tile transient.

        We preserve original outer indexing by copying the memlet and only
        adding ``other_subset`` to describe how the outer slice maps into
        tile space.

        The canonical default deep-copies the original outer->inner memlet.
        Non-canonical child classes override this to build a contiguous outer
        subset that covers the full bounding tile footprint.
        """
        new_memlet = copy.deepcopy(outer_edge.data)
        new_memlet.other_subset = self._tile_subset
        return new_memlet

    def _build_output_store_memlet(self, inner_to_outer_edge: MultiConnectorEdge[Memlet], tasklet_out_edge: MultiConnectorEdge[Memlet]) -> Memlet:
        """
        Build the memlet from the output tile transient to *outer_exit*.

        Symmetric to ``_build_input_staging_memlet``: the canonical default
        deep-copies the original inner->outer memlet and adds ``other_subset``.
        Non-canonical child classes override this to produce a contiguous
        store range.
        """
        new_memlet = copy.deepcopy(inner_to_outer_edge.data)
        new_memlet.other_subset = self._tile_subset
        return new_memlet

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
        trans_name, trans_read = self._create_and_add_tile_transient(data_name, self._tile_shape)

        # outer_entry -> transient (tile-slice memlet with other_subset).
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

        The masked variant overrides this to preload current destination tile
        values into the library node's ``_c_in`` connector, so lanes where
        ``mask == False`` can preserve their original values.

        No-op in the canonical case.
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
        trans_name, trans_write = self._create_and_add_tile_transient(data_name, self._tile_shape)

        # Library writes full tile result into transient.
        self._graph.add_edge(self._library_node, lib_conn, trans_write, None,
                             Memlet(data=trans_name, subset=self._tile_subset))

        # transient -> outer_exit (tile-slice memlet with other_subset).
        store_memlet = self._build_output_store_memlet(inner_to_outer_edge, tasklet_out_edge)
        self._graph.add_edge(trans_write, None, self._outer_exit,
                             inner_to_outer_edge.dst_conn, store_memlet)

        return trans_name, trans_write

    # ---- hooks for child classes --------------------------------------------

    def _configure_library_node(self):
        """
        Hook to configure extra connectors on the library node.

        Overridden by the masked child to add ``_m`` (mask) and ``_c_in``
        (preloaded output) input connectors.  No-op in the canonical case.
        """
        pass

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

    def apply(self, graph: SDFGState, sdfg: SDFG):  # type: ignore
        """
        Apply the transformation: replace the inner scalar map with a tile
        library call.

        ``can_be_applied`` already validated pattern shape and operator
        support; here we perform the actual graph rewrite.  The method is
        structured as a template method — child classes inject behaviour
        through the hooks ``_configure_library_node``,
        ``_build_input_staging_memlet``, ``_build_output_store_memlet``,
        ``_post_input_lowering``, and ``_add_output_preload``.
        """
        self._set_convenience_variables(sdfg, graph)

        # Resolve the matching library node and classification.
        # The op matcher maps the tasklet code to a concrete cuTile node type.
        op_match = match_tasklet_to_tile_library_node(graph, self._tasklet_node, self._get_mask_type())
        if op_match is None:
            raise ValueError("No matching library node found for tasklet")
        self._node_info = op_match.node_info
        self._tasklet_classification = op_match.tasklet_classification

        # Derive the tile extents from the inner map.  In canonical mode this
        # is the concrete domain; in masked mode it is the bounding tile.
        self._tile_shape = self._calculate_tile_shape()
        self._tile_subset = self._tile_subset_from_shape(self._tile_shape)

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
            # Child hook (e.g. preload existing output values for masked lanes).
            self._add_output_preload(data_name, inner_to_outer, tasklet_out_edge)
            self._add_output_transient_and_connect_edges(inner_to_outer, lib_conn, tasklet_out_edge)
            # Remove the original scalar store path.
            self._graph.remove_edge(inner_to_outer)

        self._remove_original_nodes()

    # ---- static utility methods (available to all child classes) -------------

    @staticmethod
    def _to_sympy_expr(expr: sp.Basic | SymExpr | int) -> sp.Basic:
        """Convert DaCe symbolic values (including ``SymExpr``) to plain SymPy."""
        if isinstance(expr, dace.symbolic.SymExpr):
            return expr.expr
        return sp.sympify(expr)

    @staticmethod
    def _is_canonical_inner_map(inner_map: nodes.Map) -> bool:
        """
        Canonical inner map predicate used to select lowering strategy.

        Canonical means: ``start == 0`` and ``step == 1`` in every dimension,
        which implies the tile is fully dense and does not need a validity
        mask.
        """
        for start, _, step in inner_map.range:
            if start != 0 or step != 1:
                return False
        return True

    @staticmethod
    def _bounding_tile_shape(inner_map: nodes.Map) -> list[sp.Basic | int]:
        """
        Compute per-dimension extents of a bounding box for an inner map range.

        Using ``Min``/``Max`` handles both increasing and decreasing ranges
        uniformly.  The result is symbolic and may include expressions that
        are only resolved at runtime.
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
    def _build_contiguous_outer_subset(tasklet_subset: subsets.Subset,
                                       inner_map: nodes.Map) -> subsets.Range:
        """
        Lift scalar tasklet accesses to a contiguous outer subset.

        For each accessed dimension, substitute the inner-map parameter with
        its min/max reachable values and build a conservative contiguous range.
        This converts the scalar index expression(s) used in the inner map to
        a contiguous outer tile range that covers all points visited by the
        inner map.
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
    def _build_mask_condition(tile_shape: list[sp.Basic | int], inner_map: nodes.Map) -> str:
        """
        Build a predicate string that checks if a tile point is within the
        original (possibly strided/reversed) inner-map iteration domain.

        For each dimension, generates direction-aware conditions for both
        positive and negative steps, including stride/offset divisibility
        checks.  Returns a conjunction of per-dimension predicates.
        """
        params = [f"m{d}" for d in range(len(tile_shape))]
        cond_terms = []

        for p, (start, end, step) in zip(params, inner_map.range):
            start = _ScalarToTileBase._to_sympy_expr(start)
            end = _ScalarToTileBase._to_sympy_expr(end)
            step = _ScalarToTileBase._to_sympy_expr(step)
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

        return " and ".join(cond_terms) if cond_terms else "True"

    @staticmethod
    def _add_mask_fill_subgraph(graph: SDFGState, mask_name: str,
                                tile_shape: list[sp.Basic | int],
                                inner_map: nodes.Map) -> tuple[nodes.AccessNode, nodes.MapEntry]:
        """
        Build a sequential map that fills the mask tile with domain validity.

        Each mask element corresponds to a point in the bounding tile and is
        ``True`` iff that point is part of the original (possibly
        strided/reversed) inner-map iteration domain.
        """
        # Create one index variable per tile dimension (m0, m1, ...).
        params = [f"m{d}" for d in range(len(tile_shape))]
        map_ranges: Dict[str, str] = {
            p: f"0:{symstr(extent)}" for p, extent in zip(params, tile_shape)
        }
        fill_entry, fill_exit = graph.add_map(
            "fill_mask_map",
            map_ranges,
            schedule=dtypes.ScheduleType.Sequential,
        )

        condition = _ScalarToTileBase._build_mask_condition(tile_shape, inner_map)
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
        ranges = self._inner_entry.map.range
        if not isinstance(ranges, subsets.Range):
            return False
        for rng in ranges:
            start, end, step = rng
            if not (start == 0 and step == 1):
                return False
        return True

    def _get_mask_type(self) -> MaskType:
        return MaskType.UNMASKED


class ScalarToTileMasked(_ScalarToTileBase):
    """
    Lower non-canonical scalar inner maps to masked cuTile library nodes.

    Non-canonical ranges (offset starts, negative/strided bounds) are mapped
    to a bounding tile.  A boolean mask marks valid points and is provided to
    the masked library node.  Existing output values are preloaded so
    unmasked lanes are preserved.

    Extra connectors on the library node:

    - ``_m`` receives the domain-validity mask.
    - ``_c_in`` receives preloaded old output values so masked ops can keep
      lanes where ``mask == False``.
    """

    def _has_valid_inner_map_ranges(self) -> bool:
        for start, end, step in self._inner_entry.map.range:
            if step == 0:
                return False
        # Only match non-canonical maps.
        if self._is_canonical_inner_map(self._inner_entry.map):
            return False
        return True

    def _get_mask_type(self) -> MaskType:
        return MaskType.RUNTIME

    def _calculate_tile_shape(self) -> tuple:
        """
        The bounding tile is the smallest axis-aligned tile that contains all
        iteration points from the original inner map, regardless of direction.
        """
        return tuple(self._bounding_tile_shape(self._inner_entry.map))

    def _configure_library_node(self):
        """
        Add ``_c_in`` connector for preloaded output values so masked lanes
        can preserve their original values.
        """
        out_in_conn = self._node_info.out_in or "_c_in"
        if out_in_conn not in self._library_node.in_connectors:
            self._library_node.add_in_connector(out_in_conn)

    def _build_input_staging_memlet(self, outer_edge: MultiConnectorEdge[Memlet], tasklet_edge: MultiConnectorEdge[Memlet]) -> Memlet:
        """
        Convert scalar index expression(s) to a contiguous outer tile range
        that covers all points visited by the inner map.
        """
        data_name = cast(str, outer_edge.data.data)
        load_subset = self._build_contiguous_outer_subset(
            tasklet_edge.data.subset, self._inner_entry.map)
        return Memlet(data=data_name, subset=load_subset, other_subset=self._tile_subset)

    def _build_output_store_memlet(self, inner_to_outer_edge: MultiConnectorEdge[Memlet], tasklet_out_edge: MultiConnectorEdge[Memlet]) -> Memlet:
        """Store full tile back to outer map footprint using a contiguous range."""
        data_name = cast(str, inner_to_outer_edge.data.data)
        store_subset = self._build_contiguous_outer_subset(
            tasklet_out_edge.data.subset, self._inner_entry.map)
        return Memlet(data=data_name, subset=store_subset, other_subset=self._tile_subset)

    def _post_input_lowering(self):
        """
        Create mask transient, build its fill subgraph, and connect to the
        library node.

        The mask fill subgraph executes inside the outer map so mask predicates
        can reference tiled-loop symbols.  Mask storage is aligned with operand
        storage when possible to avoid introducing unnecessary storage-space
        transitions.
        """
        # Keep mask storage aligned with operand storage when possible.
        mask_storage = dtypes.StorageType.Default
        for outer_edge, lib_conns, tasklet_edge in self._input_plan.values():
            data_name = cast(str, outer_edge.data.data)
            data_desc = self._sdfg.arrays[data_name]
            mask_storage = data_desc.storage
            break

        tile_shape = list(self._tile_shape)
        # Allocate the mask tile once per outer-map iteration.
        mask_name = self._sdfg._find_new_name("map_mask_tile")
        self._sdfg.add_transient(
            mask_name,
            shape=tile_shape,
            dtype=dace.bool,
            storage=mask_storage,
            lifetime=dtypes.AllocationLifetime.Scope,
        )

        # Build producer subgraph that computes per-lane validity predicate.
        mask_source, fill_entry = self._add_mask_fill_subgraph(
            self._graph, mask_name, tile_shape, self._inner_entry.map)
        self._graph.add_edge(self._outer_entry, None, fill_entry, None, Memlet())

        mask_in_conn = self._node_info.mask_in or "_m"
        self._graph.add_edge(mask_source, None, self._library_node, mask_in_conn,
                             Memlet(data=mask_name, subset=self._tile_subset))

    def _add_output_preload(self, data_name: str, inner_to_outer_edge: MultiConnectorEdge[Memlet], tasklet_out_edge: MultiConnectorEdge[Memlet]) -> None:
        """
        Preload current destination tile values into ``_c_in`` so masked ops
        can keep lanes where ``mask == False``.

        Creates a separate preload path:
        output_array -> outer_entry -> preload_transient -> library_node._c_in
        """
        data_desc = self._sdfg.arrays[data_name]
        store_subset = self._build_contiguous_outer_subset(
            tasklet_out_edge.data.subset, self._inner_entry.map)

        preload_name = self._sdfg._find_new_name(data_name + "_tile_in")
        self._sdfg.add_transient(
            preload_name,
            shape=list(self._tile_shape),
            dtype=data_desc.dtype,
            storage=data_desc.storage,
            lifetime=dtypes.AllocationLifetime.Scope,
        )
        preload_tile = self._graph.add_access(preload_name)

        # Read existing values from the output array.
        preload_node = self._graph.add_access(data_name)
        preload_in_conn = f"IN_PRELOAD_{preload_name}"
        preload_out_conn = preload_in_conn.replace("IN_", "OUT_", 1)
        self._outer_entry.add_in_connector(preload_in_conn)
        self._outer_entry.add_out_connector(preload_out_conn)
        self._graph.add_edge(preload_node, None, self._outer_entry, preload_in_conn,
                             Memlet(data=data_name, subset=store_subset))
        self._graph.add_edge(
            self._outer_entry, preload_out_conn, preload_tile, None,
            Memlet(data=data_name, subset=store_subset, other_subset=self._tile_subset),
        )

        out_in_conn = self._node_info.out_in or "_c_in"
        self._graph.add_edge(preload_tile, None, self._library_node, out_in_conn,
                             Memlet(data=preload_name, subset=self._tile_subset))