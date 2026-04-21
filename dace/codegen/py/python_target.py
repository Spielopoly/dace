# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Python target code generator for the DaCe Python backend.

Generates pure Python code for SDFG nodes, scopes, array allocation/deallocation,
and memory copies. Registers with the dispatcher as the node, map, array, and copy
handler for CPU storage types and sequential schedules.
"""
import ast
import itertools
from typing import TYPE_CHECKING

from dace import data, dtypes
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.codegen.target import TargetCodeGenerator
from dace.codegen.dispatcher import TargetDispatcher
from dace.sdfg import nodes, SDFG, ScopeSubgraphView, scope_contains_scope, NodeNotExpandedError
from dace.sdfg.state import ControlFlowRegion, StateSubgraphView
from dace.sdfg.graph import MultiConnectorEdge
from dace.memlet import Memlet

if TYPE_CHECKING:
    from dace.codegen.py.framecode import DaCePythonCodeGenerator


class PythonCodeGen(TargetCodeGenerator):
    """SDFG Python code generator.

    Generates pure Python code for nodes, scopes, and data management.
    """

    title = "Python"
    target_name = "python"
    language = "python"

    def get_includes(self) -> dict[str, list[str]]:
        return {'frame': ['numpy']}

    def __init__(self, frame_codegen: 'DaCePythonCodeGenerator', sdfg: SDFG):
        self._frame = frame_codegen
        self._dispatcher: TargetDispatcher = frame_codegen.dispatcher
        dispatcher = self._dispatcher

        # Register as generic node dispatcher
        dispatcher.register_node_dispatcher(self)

        # Register for sequential maps
        dispatcher.register_map_dispatcher([dtypes.ScheduleType.Sequential], self)

        # Register for CPU storage types
        cpu_storage = [dtypes.StorageType.CPU_Heap, dtypes.StorageType.Register]
        dispatcher.register_array_dispatcher(cpu_storage, self)

        # Register copy dispatchers for all CPU storage pairs
        for src_storage, dst_storage in itertools.product(cpu_storage, cpu_storage):
            dispatcher.register_copy_dispatcher(src_storage, dst_storage, None, self)

    # =========================================================================
    # Node dispatch
    # =========================================================================

    def generate_node(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                      node: nodes.Node, function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        gen = getattr(self, "_generate_" + type(node).__name__, None)
        if gen is None:
            if isinstance(node, nodes.LibraryNode):
                raise NodeNotExpandedError(sdfg, state_id, dfg.node_id(node))
            raise NotImplementedError(
                f"Python backend: no code generator for node type {type(node).__name__}")
        gen(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)

    # =========================================================================
    # AccessNode
    # =========================================================================

    def _generate_AccessNode(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                             node: nodes.AccessNode, function_stream: PythonCodeIOStream,
                             callsite_stream: PythonCodeIOStream) -> None:
        state = cfg.nodes()[state_id]
        sdict = state.scope_dict()

        # Incoming edges: dispatch copy when source is another AccessNode
        for edge in state.in_edges(node):
            memlet_path = state.memlet_path(edge)
            if memlet_path[-1].dst != node:
                continue
            src_node = memlet_path[0].src
            if isinstance(src_node, nodes.CodeNode):
                continue  # Handled by the code node's generator
            if isinstance(src_node, nodes.AccessNode):
                # Only generate copy at the innermost scope where both arrays exist
                if scope_contains_scope(sdict, src_node, node) and sdict[src_node] != sdict[node]:
                    self._dispatcher.dispatch_copy(src_node, node, edge, sdfg, cfg, dfg, state_id,
                                                   function_stream, callsite_stream)

        # Outgoing edges: dispatch copy when destination is another AccessNode
        for edge in state.out_edges(node):
            memlet_path = state.memlet_path(edge)
            dst_node = memlet_path[-1].dst
            if isinstance(dst_node, nodes.CodeNode):
                continue
            if dst_node == node:
                continue
            if isinstance(dst_node, nodes.AccessNode):
                # Skip if destination is in an inner scope (handled there)
                if sdict[node] != sdict[dst_node] and scope_contains_scope(sdict, node, dst_node):
                    continue
                self._dispatcher.dispatch_copy(node, dst_node, edge, sdfg, cfg, dfg, state_id,
                                               function_stream, callsite_stream)

    # =========================================================================
    # Tasklet
    # =========================================================================

    def _generate_Tasklet(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                          node: nodes.Tasklet, function_stream: PythonCodeIOStream,
                          callsite_stream: PythonCodeIOStream) -> None:
        state = cfg.nodes()[state_id]

        # --- Read inputs ---
        for edge in state.in_edges(node):
            if not edge.dst_conn:
                continue
            connector = edge.dst_conn
            memlet = edge.data
            if memlet.data is None:
                raise NotImplementedError("Code-to-code memlets not supported in Python backend")

            desc = sdfg.arrays[memlet.data]
            if isinstance(desc, data.Scalar):
                callsite_stream.write(f"{connector} = {memlet.data}", cfg, state_id, node)
            else:
                callsite_stream.write(f"{connector} = {memlet.data}[{memlet.subset}]", cfg, state_id, node)

        # --- Tasklet body ---
        if node.code.language != dtypes.Language.Python:
            raise NotImplementedError(
                f"Python backend only supports Python tasklets, got {node.code.language}")

        for stmt in node.code.code:
            callsite_stream.write(ast.unparse(stmt), cfg, state_id, node)

        # --- Write outputs ---
        for edge in state.out_edges(node):
            if not edge.src_conn:
                continue
            connector = edge.src_conn
            memlet = edge.data
            if memlet.data is None:
                raise NotImplementedError("Code-to-code memlets not supported in Python backend")

            desc = sdfg.arrays[memlet.data]
            if isinstance(desc, data.Scalar):
                callsite_stream.write(f"{memlet.data} = {connector}", cfg, state_id, node)
            else:
                callsite_stream.write(f"{memlet.data}[{memlet.subset}] = {connector}", cfg, state_id, node)

    # =========================================================================
    # Scope (map) generation
    # =========================================================================

    def generate_scope(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, state_id: int,
                       function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        entry_node = dfg_scope.source_nodes()[0]
        params = entry_node.map.params
        ranges = entry_node.map.range

        for param, (start, end, step) in zip(params, ranges):
            callsite_stream.write(f"for {param} in range({start}, {end} + 1, {step}):", cfg, state_id, entry_node)

        # Generate loop body with stream-managed indentation
        if isinstance(callsite_stream, PythonCodeIOStream) and isinstance(function_stream, PythonCodeIOStream):
            with callsite_stream.indented():
                pos_before = callsite_stream.tell()
                self._dispatcher.dispatch_subgraph(sdfg, cfg, dfg_scope, state_id, function_stream, callsite_stream,
                                                   skip_entry_node=True)
                if callsite_stream.tell() == pos_before:
                    callsite_stream.write('pass', cfg, state_id, entry_node)
        else:
            raise NotImplementedError("Python backend requires PythonCodeIOStream for scope generation")

    # =========================================================================
    # Array management
    # =========================================================================

    def declare_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg, state_id: int, node: nodes.AccessNode,
                      nodedesc: data.Data, global_stream: PythonCodeIOStream,
                      declaration_stream: PythonCodeIOStream) -> None:
        pass  # Python variables are created on assignment

    def allocate_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg, state_id: int, node: nodes.AccessNode,
                       nodedesc: data.Data, global_stream: PythonCodeIOStream, declaration_stream: PythonCodeIOStream,
                       allocation_stream: PythonCodeIOStream) -> None:
        if not nodedesc.transient:
            return  # Non-transient (argument) arrays are passed externally

        if isinstance(nodedesc, data.Scalar):
            allocation_stream.write(f"{node.data} = 0", cfg, state_id, node)
        elif isinstance(nodedesc, data.Array):
            shape = ", ".join(str(s) for s in nodedesc.shape)
            dtype_str = nodedesc.dtype.to_string()
            allocation_stream.write(
                f"{node.data} = numpy.zeros(({shape},), dtype=numpy.{dtype_str})",
                cfg, state_id, node)
        else:
            raise NotImplementedError(
                f"Python backend: cannot allocate {type(nodedesc).__name__}")

    def deallocate_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg, state_id: int, node: nodes.AccessNode,
                         nodedesc: data.Data, function_stream: PythonCodeIOStream,
                         callsite_stream: PythonCodeIOStream) -> None:
        pass  # Python garbage-collects

    # =========================================================================
    # Memory copy
    # =========================================================================

    def copy_memory(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg, state_id: int,
                    src_node: nodes.Node, dst_node: nodes.Node, edge: MultiConnectorEdge[Memlet],
                    function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        memlet = edge.data
        src_name = memlet.data if isinstance(src_node, nodes.AccessNode) else edge.src_conn
        dst_name = memlet.data if isinstance(dst_node, nodes.AccessNode) else edge.dst_conn

        src_desc = sdfg.arrays.get(src_node.data) if isinstance(src_node, nodes.AccessNode) else None
        dst_desc = sdfg.arrays.get(dst_node.data) if isinstance(dst_node, nodes.AccessNode) else None

        if src_desc is not None and not isinstance(src_desc, data.Scalar) and memlet.src_subset is not None:
            src_expr = f"{src_node.data}[{memlet.src_subset}]"
        elif src_desc is not None:
            src_expr = src_node.data
        else:
            src_expr = src_name

        if dst_desc is not None and not isinstance(dst_desc, data.Scalar) and memlet.dst_subset is not None:
            dst_expr = f"{dst_node.data}[{memlet.dst_subset}]"
        elif dst_desc is not None:
            dst_expr = dst_node.data
        else:
            dst_expr = dst_name

        callsite_stream.write(f"{dst_expr} = {src_expr}", cfg, state_id, src_node)

    def define_out_memlet(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg, state_id: int,
                          src_node: nodes.Node, dst_node: nodes.Node, edge: MultiConnectorEdge[Memlet],
                          function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        pass  # Python variables don't need pre-declaration

    # =========================================================================
    # Interstate variables
    # =========================================================================

    def emit_interstate_variable_declaration(self, name: str, dtype: dtypes.typeclass,
                                             callsite_stream: PythonCodeIOStream, sdfg: SDFG):
        pass  # Python variables are created on assignment
