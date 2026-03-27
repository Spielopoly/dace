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
    
    def _calculate_tile_shape(self) -> tuple:
        """
        Calculate the shape of the tile being processed by the inner map.

        This is needed to determine the shape of the library node
        and the transients used to connect it to the rest of the graph.
        """
        assert isinstance(self.inner_map_entry.map.range, subsets.Range)
        return tuple(self.inner_map_entry.map.range.size())

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

    def _find_path_edge(self, anchor_edge: sdutil.gr.MultiConnectorEdge[Memlet], src: nodes.Node, dst: nodes.Node) -> Optional[sdutil.gr.MultiConnectorEdge[Memlet]]:
        """
        Find the path segment from `src` to `dst` on an anchor edge memlet path.
        """
        for e in self._graph.memlet_path(anchor_edge):
            if e.src is src and e.dst is dst:
                return e
        return None

    def _create_input_plan(self, input_slots: dict[str, list[str]]):
        """
        The input plan represents a mapping from edges, that connect the outer map to the inner map, to a list of corresponding library node connectors.
        
        
        Parameters
        ----------
        input_slots : dict[str, list[str]]
            A dictionary mapping tasklet input connector names to lists of library node input connector names.

        Returns
        -------
        input_plan : dict[tuple[Optional[str], Optional[str]], tuple[sdutil.gr.MultiConnectorEdge[Memlet], list[str]]]
            A dictionary mapping edge keys to tuples of the corresponding edge and library node connectors.

        """
        input_plan: dict[tuple[Optional[str], Optional[str]], tuple[sdutil.gr.MultiConnectorEdge[Memlet], list[str]]] = {}
        for tasklet_conn, lib_conns in input_slots.items():
            tasklet_edges = list(self._graph.in_edges_by_connector(self.tasklet, tasklet_conn))
            if not tasklet_edges:
                continue
            outer_map_to_inner_map_edge = self._find_path_edge(tasklet_edges[0], self.outer_map_entry, self.inner_map_entry)
            if outer_map_to_inner_map_edge is None:
                continue
            edge_key = (outer_map_to_inner_map_edge.src_conn, outer_map_to_inner_map_edge.dst_conn)
            if edge_key not in input_plan:
                input_plan[edge_key] = (outer_map_to_inner_map_edge, [])
            input_plan[edge_key][1].extend(lib_conns)
        return input_plan
    
    def _create_and_add_tile_transient(self, data_name: str, tile_shape: tuple):
        """
        Create a transient array for tile data.

        The transient has scope lifetime and tile shape, and is used to
        connect the original scalar memlets to the library node's tile memlets.
        
        Parameters
        ----------
        data_name : str
            The base name of the data for the transient array (a unique name will be generated).
        tile_shape : tuple
            The shape of the tile being processed, used to set the transient shape.

        Returns
        -------
        trans_name : str
            The name of the created transient array.
        trans_node : AccessNode
            The graph node corresponding to the transient array.
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

    def _add_input_transient_and_connect_edges(self, outer_map_to_inner_map_edge: sdutil.gr.MultiConnectorEdge[Memlet], lib_conns: list[str]):
        """
        For a given edge connecting the outer map to the inner map, create a transient tile array and connect it to the library node.
        
        This involves:
        1. Creating a transient array for the tile data.
        2. Connecting the outer map entry to the transient with a memlet that has the same subset as the original edge but with `other_subset` set to the tile subset.
        3. Connecting the transient to the library node with memlets that have the tile subset as their subset.
        
        Parameters
        ----------
        outer_map_to_inner_map_edge : MultiConnectorEdge[Memlet]
            The edge connecting the outer map to the inner map.
        lib_conns : list[str]
            The list of library node connector names that should be fed by this edge.
        
        Returns
        -------
        trans_name : str
            The name of the created transient array.
        trans_read : AccessNode
            The graph node corresponding to the transient array.
        """
        
        data_name = cast(str, outer_map_to_inner_map_edge.data.data)
        trans_name, trans_read = self._create_and_add_tile_transient(data_name, self._tile_shape)

        # outer_entry → transient (tile-slice memlet with other_subset)
        new_memlet = copy.deepcopy(outer_map_to_inner_map_edge.data)
        new_memlet.other_subset = self._tile_subset
        self._graph.add_edge(self.outer_map_entry, outer_map_to_inner_map_edge.src_conn, trans_read, None, new_memlet)

        # Add transient → library node edges
        for lib_conn in lib_conns:
            self._graph.add_edge(trans_read, None, self._library_node, lib_conn,
                                 Memlet(data=trans_name, subset=self._tile_subset))
        
        return trans_name, trans_read
    
    def _output_connector_for_tasklet(self, tasklet_conn: Optional[str]) -> Optional[str]:
        if tasklet_conn is None:
            return None
        if self._tasklet_classification.lhs == tasklet_conn:
            return self._node_info.out
        return None

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
        
        # === Input lowering ===
        # For each tasklet input connector:
        #   outer map slice -> tile transient -> library node input.
        # We preserve original outer indexing by copying memlets and only adding
        # `other_subset` to describe how the outer slice maps into tile space

        # Multiple tasklet inputs can be fed by the same map path
        # (frontend c = a + a often yields two tasklet connectors from one map
        # connector)
        self._input_slots = self._get_map_from_tasklet_input_to_libnode_inputs()
        # The input plan represents a mapping from edges, that connect the outer map to the inner map, to a list of corresponding library node connectors.
        self._input_plan = self._create_input_plan(self._input_slots)
        
        # Create input transient accesses (tiles) and edges
        for outer_map_to_inner_map_edge, lib_conns in self._input_plan.values():
            self._add_input_transient_and_connect_edges(outer_map_to_inner_map_edge, lib_conns)

            # Remove the original scalar feed from outer map to inner map.
            self._graph.remove_edge(outer_map_to_inner_map_edge)
        
        # === Output lowering ===
        # Symmetric to input lowering:
        #   library node output -> tile transient -> original outer destination.
        # Fortunately there is only one output, so no need to worry about multiple
        # tasklet connectors mapping to the same library node connector and such.
        for edge in graph.in_edges(self.inner_map_exit):
            if edge.src is not self.tasklet:
                continue
            tasklet_conn = edge.src_conn
            lib_conn = self._output_connector_for_tasklet(tasklet_conn)
            if lib_conn is None:
                continue

            # Follow the same memlet path and pick the inner-exit -> outer-exit
            inner_to_outer = self._find_path_edge(edge, self.inner_map_exit, self.outer_map_exit)
            if inner_to_outer is None:
                continue

            data_name = cast(str, inner_to_outer.data.data)
            trans_name, trans_write = self._create_and_add_tile_transient(data_name, self._tile_shape)

            # Library writes full tile result into transient.
            graph.add_edge(self._library_node, lib_conn, trans_write, None,
                           Memlet(data=trans_name, subset=self._tile_subset))

            # transient → outer_exit (tile-slice memlet with other_subset)
            new_memlet = copy.deepcopy(inner_to_outer.data)
            new_memlet.other_subset = self._tile_subset
            graph.add_edge(trans_write, None, self.outer_map_exit,
                           inner_to_outer.dst_conn, new_memlet)

            # Remove the original scalar store path.
            graph.remove_edge(inner_to_outer)

        # After all dataflow edges are rewired through the library node, the
        # scalar tasklet and inner-map nodes are dead and can be removed.
        graph.remove_node(self.tasklet)
        graph.remove_node(self.inner_map_entry)
        graph.remove_node(self.inner_map_exit)


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
        