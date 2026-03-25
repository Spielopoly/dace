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

    OuterMapEntry ─→ InnerMapEntry ─→ Tasklet ─→ InnerMapExit ─→ OuterMapExit

Result (AFTER, canonical)::

    OuterMapEntry ─→ tile_transient_read ─→ LibNode ─→ tile_transient_write ─→ OuterMapExit

Result (AFTER, non-canonical inner maps)::

    OuterMapEntry ─→ tile_transient_read ─→ MaskedLibNode ─→ tile_transient_write ─→ OuterMapExit
                    + generated mask transient used to select valid map points
"""
from __future__ import annotations

import copy
from typing import Dict, Optional, cast
import abc

import dace
import sympy as sp
from dace import Memlet, dtypes, subsets
from dace.sdfg import SDFG, SDFGState, nodes, utils as sdutil
from dace.symbolic import symstr
from dace.transformation import transformation as xf

from dace.libraries.cutile.op_registry import match_tasklet_to_tile_library_node, MaskType, TaskletLibraryNodeMatch


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
    """

    outer_map_entry = xf.PatternNode(nodes.MapEntry)
    inner_map_entry = xf.PatternNode(nodes.MapEntry)
    tasklet = xf.PatternNode(nodes.Tasklet)
    inner_map_exit = xf.PatternNode(nodes.MapExit)
    outer_map_exit = xf.PatternNode(nodes.MapExit)

    @classmethod
    def expressions(cls): # type: ignore
        return [sdutil.node_path_graph(cls.outer_map_entry,
                                       cls.inner_map_entry,
                                       cls.tasklet,
                                       cls.inner_map_exit,
                                       cls.outer_map_exit)]
    
    def _set_convenience_variables(self, sdfg: SDFG, graph: SDFGState):
        self._sdfg = sdfg
        self._graph = graph
    
    def _get_scope(self, map_entry: nodes.MapEntry):
        return self._graph.scope_subgraph(map_entry, include_entry=False, include_exit=False)

    def _has_exactly_one_tasklet_in_inner_scope(self):
        inner_scope = self._get_scope(self.inner_map_entry)
        return len(inner_scope.nodes()) == 1 \
            and self.tasklet in inner_scope.nodes()

    def _has_valid_inner_scope(self):
        return self._has_exactly_one_tasklet_in_inner_scope()
    
    def _has_valid_outer_scope(self):
        outer_scope = self._get_scope(self.outer_map_entry)
        return set(outer_scope.nodes()) == {self.inner_map_entry, self.inner_map_exit, self.tasklet}

    def _has_valid_scope(self):
        return self._has_valid_inner_scope() and self._has_valid_outer_scope()
    
    def _get_all_edges_of_node(self, node: nodes.Node):
        return list(self._graph.in_edges(node)) + list(self._graph.out_edges(node))
    
    def _is_tasklet_scalar(self, tasklet: nodes.Tasklet):
        """
        Every tasklet access must be scalar (single-point range).
        The op matcher assumes scalar tasklet connectors correspond to one
        element position inside the tile non-scalar memlets violate that model.
        Note that this does not mean that the type returned by 
          `tasklet_utils.classify_tasklet`
        will be "scalar"
        For example, a tasklet that performs elementwise addition on two 
        vectors will likely be classified as "array-array" type, but it can 
        still be matched to a tile library node as long as its connectors are scalar.
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
        This depends on the type of tile library node we want to match
        and therefore this needs to be implemented by the child class.
        """
        ...
    
    def _has_valid_outer_map_ranges(self) -> bool:
        """
        We mostly don't care about the outer map ranges. Because the relevant
        movement will actually be handled by the rest of the code generation. 
        We just reconnect the edges to our access nodes
        """
        return True
    
    def _has_valid_map_ranges(self):
        return self._has_valid_inner_map_ranges() and self._has_valid_outer_map_ranges()
    
    @abc.abstractmethod
    def _get_mask_type(self) -> MaskType:
        """
        Determine the type of mask (if any) to use for the library node.

        This depends on the structure of the inner map and which library node
        variants are registered, and therefore this needs to be implemented by
        the child class.
        """
        ...
        

    def can_be_applied(self, graph: SDFGState, expr_index: int,
                       sdfg: SDFG, permissive: bool = False) -> bool:
        """
        Validate that the matched subgraph is a pure element-wise inner loop.

        The transformation only applies when the inner map has no extra nodes,
        tasklet accesses are scalar, and there is a registered tile operator
        for the tasklet code.
        
        These assumptions may not work for later transformations,
        those should override this method.
        """
        self._set_convenience_variables(sdfg, graph)
        
        if not self._has_valid_scope():
            return False
        if not self._is_tasklet_scalar(self.tasklet):
            return False
        if not self._has_valid_map_ranges():
            return False
        if match_tasklet_to_tile_library_node(graph, self.tasklet, self._get_mask_type()) is None:
            return False
        
        return True
    
    def _calculate_tile_shape(self):
        """
        Calculate the shape of the tile being processed by the inner map.

        This is needed to determine the shape of the library node
        and the transients used to connect it to the rest of the graph.
        """
        assert isinstance(self.inner_map_entry.map.range, subsets.Range)
        return self.inner_map_entry.map.range.size()

    def _tile_subset_from_shape(self, tile_shape) -> subsets.Range:
        """
        Build a dense local tile range [0, extent-1] in every dimension.
        """
        return subsets.Range([(0, d - 1, 1) for d in tile_shape])

    def _get_map_from_tasklet_input_to_libnode_inputs(self) -> dict[str, list[str]]:
        """
        Build tasklet-input to library-input slot mapping.

        A single tasklet connector can map to multiple library slots, e.g.,
        for expressions such as c = a + a where rhs1 == rhs2.
        
        Returns a dict mapping tasklet connector name to a list of library node
        connector names that should be fed by it.
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
                raise ValueError(f"Invalid mapping between tasklet and library node connectors: "
                                 f"tasklet connector {tasklet_conn}, library connector {library_conn}")
        return slot_map

    def _get_map_from_tasklet_input_to_edge(self) -> Dict[str, sdutil.gr.MultiConnectorEdge[Memlet]]:
        """Map tasklet input connector to the edge inner_entry -> tasklet."""
        edge_map = {}
        for edge in self._graph.edges_between(self.inner_map_entry, self.tasklet):
            if edge.dst_conn is None:
                continue
            edge_map[edge.dst_conn] = edge
        return edge_map
    
    def _create_input_plan_and_edges(self, input_slots: dict[str, list[str]], input_edges: dict[str, sdutil.gr.MultiConnectorEdge[Memlet]]):
        """
        Returns:
        - input_plan: dict mapping 
        """
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
        return input_plan, plan_edges

    def apply(self, graph: SDFGState, sdfg: SDFG): # type: ignore
        """
        This should be overwritten by child classes but then also called
        in the child class's apply method to set the necessary variables.
        
        This function already tries to calculate and create as much as possible
        so code duplication is minimized in the child classes
        """
        
        self._set_convenience_variables(sdfg, graph)
        
        # get the library node class to use for this tasklet
        op_match = match_tasklet_to_tile_library_node(graph, self.tasklet, self._get_mask_type())
        if op_match is None:
            raise ValueError("No matching library node found for tasklet, cannot apply transformation")
        self._node_info = op_match.node_info
        self._tasklet_classification = op_match.tasklet_classification
        
        # calculate the tile shape
        self._tile_shape = self._calculate_tile_shape()
        self._tile_subset = self._tile_subset_from_shape(self._tile_shape)
        
        # create the library node and add it to the graph
        # It will need to be properly connected and configured in the child class
        self._library_node = self._node_info.type(self._node_info.node_name)
        graph.add_node(self._library_node)
        
        # Multiple tasklet inputs can be fed by the same map connector
        # (frontend c = a + a often yields __in1/__in2 both from OUT_A).
        # Group by map input connector and lower each producer edge once.
        self._input_slots = self._get_map_from_tasklet_input_to_libnode_inputs()
        self._input_edges = self._get_map_from_tasklet_input_to_edge()
        
        


class ScalarToTileCanonical(_ScalarToTileBase):
    """
    Recognizes the canonical pattern of an inner map with exactly one tasklet,
    where all accesses in the tasklet are scalar and match a registered tile
    library node, and replaces it with a single library node that operates on
    the whole tile at once.
    
    A canonical pattern looks is the simplest possible version of a tiled map:
    OuterMapEntry ─→ InnerMapEntry ─→ Tasklet ─→ InnerMapExit ─→ OuterMapExit
    where the inner map iterates over all elements of the tile
    (i.e. its range is something like `0:tile_size:1`), and the tasklet performs
    a simple element-wise operation on those elements that can be directly
    matched to a tile library node.
    """
    
    def _has_valid_inner_map_ranges(self) -> bool:
        # The inner map must iterate over a full tile with unit stride in order 
        # to be directly replaced with a single tile library node.
        # This is because the library node will consume/produce a full tile at once,
        # so the inner map needs to correspond exactly to that tile.
        # If we need something more complicated (e.g. non-unit stride, partial tile, etc.)
        # then we need to use a masked library node variant and generate masks accordingly.
        ranges = self.inner_map_entry.map.range
        if not isinstance(ranges, subsets.Range):
            return False
        for rng in ranges:
            start, end, step = rng
            if not (start == 0 and step == 1):
                return False
        return True
    
    def _get_mask_type(self) -> MaskType:
        # The canonical pattern does not use any masks, so we return UNMASKED.
        return MaskType.UNMASKED
    
    def apply(self, graph: SDFGState, sdfg: SDFG):
        """
        Lower a canonical inner map (0-based, unit-stride) to a tile op.

        Data is staged through tile-shaped transients to convert scalar memlets
        into full-tile memlets consumed/produced by the library node.
        """
        super().apply(graph, sdfg)
        