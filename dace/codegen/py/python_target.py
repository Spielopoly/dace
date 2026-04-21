# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Python target code generator for the DaCe Python backend.

Generates pure Python code for SDFG nodes, scopes, array allocation/deallocation,
and memory copies. Registers with the dispatcher as the node, map, array, and copy
handler for CPU storage types and sequential schedules.
"""
import ast
import copy
import itertools
from typing import TYPE_CHECKING

from dace import data, dtypes, subsets, symbolic
from dace.config import Config
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
        return {'frame': ['numpy', 'from dataclasses import dataclass']}

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

        self._generated_nested_sdfg: dict[object, object] = {}

    def _python_read_expr(self, name: str, desc: data.Data, subset) -> str:
        if isinstance(desc, data.Scalar) or subset is None:
            return name
        return f"{name}[{subset}]"

    def _python_write_target(self, name: str, desc: data.Data, subset) -> str:
        if isinstance(desc, data.Scalar):
            return name
        if subset is None:
            return f"{name}[...]"
        return f"{name}[{subset}]"

    def _is_python_view(self, desc: data.Data, subset) -> bool:
        if isinstance(desc, data.Scalar):
            return False
        if subset is None:
            return True
        if isinstance(subset, subsets.SubsetUnion):
            return False
        return subset.data_dims() > 0

    def _nested_temp_name(self, cfg: ControlFlowRegion, state_id: int, dfg: StateSubgraphView, node: nodes.NestedSDFG,
                          connector: str) -> str:
        return f"__dace_nested_{cfg.cfg_id}_{state_id}_{dfg.node_id(node)}_{connector}"

    def _allocate_nested_temp(self, name: str, desc: data.Data, stream: PythonCodeIOStream, cfg: ControlFlowRegion,
                              state_id: int, node: nodes.NestedSDFG) -> None:
        if isinstance(desc, data.Scalar):
            stream.write(f"{name} = numpy.{desc.dtype.to_string()}(0)", cfg, state_id, node)
            return
        if isinstance(desc, data.Array):
            shape = ", ".join(symbolic.symstr(s) for s in desc.shape)
            stream.write(f"{name} = numpy.zeros(({shape},), dtype=numpy.{desc.dtype.to_string()})", cfg, state_id,
                         node)
            return
        raise NotImplementedError(f"Python backend: cannot create temporary for {type(desc).__name__}")

    def _same_outer_access(self, in_edge: MultiConnectorEdge[Memlet], out_edge: MultiConnectorEdge[Memlet]) -> bool:
        if in_edge.data.data != out_edge.data.data:
            return False
        src_subset = '' if in_edge.data.src_subset is None else str(in_edge.data.src_subset)
        dst_subset = '' if out_edge.data.dst_subset is None else str(out_edge.data.dst_subset)
        return src_subset == dst_subset

    def _is_single_value_array(self, desc: data.Data) -> bool:
        if not isinstance(desc, data.Array):
            return False
        return all((dim == 1) == True for dim in desc.shape)

    def _temp_init_target(self, name: str, desc: data.Data, subset) -> str:
        if isinstance(desc, data.Array) and subset is None and self._is_single_value_array(desc):
            return f"{name}.flat[0]"
        return self._python_write_target(name, desc, subset)

    def _temp_result_expr(self, name: str, desc: data.Data, subset, scalarize: bool) -> str:
        if isinstance(desc, data.Array) and subset is None and scalarize and self._is_single_value_array(desc):
            return f"{name}.flat[0]"
        return self._python_read_expr(name, desc, subset)

    def _nested_symbol_replacements(self, node: nodes.NestedSDFG) -> dict[object, object]:
        replacements: dict[object, object] = {}
        for name, value in node.symbol_mapping.items():
            replacements[symbolic.pystr_to_symbolic(name)] = symbolic.pystr_to_symbolic(value)
        return replacements

    def _mapped_rebased_connector_subset(self, desc: data.Data, node: nodes.NestedSDFG):
        if not hasattr(desc, 'shape'):
            return None

        replacements = self._nested_symbol_replacements(node)
        ranges = []
        for extent in desc.shape:
            mapped_extent = symbolic.pystr_to_symbolic(extent)
            if replacements and symbolic.issymbolic(mapped_extent):
                mapped_extent = mapped_extent.subs(replacements)
            ranges.append((0, mapped_extent - 1, 1))
        return subsets.Range(ranges)

    def _is_rebased_full_connector_view(self, desc: data.Data, node: nodes.NestedSDFG, subset) -> bool:
        if subset is None:
            return True
        if isinstance(subset, subsets.SubsetUnion):
            return False

        expected_subset = self._mapped_rebased_connector_subset(desc, node)
        if expected_subset is None:
            return False

        actual_subset = copy.deepcopy(subset)
        try:
            return expected_subset.covers_precise(actual_subset) and actual_subset.covers_precise(expected_subset)
        except (AttributeError, TypeError, ValueError):
            return False

    def _build_nested_data_binding(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                                   node: nodes.NestedSDFG, connector: str,
                                   in_edge: MultiConnectorEdge[Memlet] | None,
                                   out_edge: MultiConnectorEdge[Memlet] | None) -> tuple[str, list[str], list[str]]:
        inner_desc = node.sdfg.arrays[connector]
        prelude: list[str] = []
        postlude: list[str] = []

        if in_edge is not None:
            outer_src_desc = sdfg.arrays[in_edge.data.data]
            input_expr = self._python_read_expr(in_edge.data.data, outer_src_desc, in_edge.data.src_subset)
            input_is_direct = self._is_python_view(outer_src_desc, in_edge.data.src_subset)
            input_is_safe_view = isinstance(inner_desc, data.Array) and self._is_rebased_full_connector_view(
                inner_desc, node, in_edge.data.dst_subset)
        else:
            outer_src_desc = None
            input_expr = None
            input_is_direct = False
            input_is_safe_view = False

        if out_edge is not None:
            outer_dst_desc = sdfg.arrays[out_edge.data.data]
            output_expr = self._python_read_expr(out_edge.data.data, outer_dst_desc, out_edge.data.dst_subset)
            output_target = self._python_write_target(out_edge.data.data, outer_dst_desc, out_edge.data.dst_subset)
            output_is_direct = self._is_python_view(outer_dst_desc, out_edge.data.dst_subset)
            output_is_safe_view = isinstance(inner_desc, data.Array) and self._is_rebased_full_connector_view(
                inner_desc, node, out_edge.data.src_subset)
        else:
            outer_dst_desc = None
            output_expr = None
            output_target = None
            output_is_direct = False
            output_is_safe_view = False

        if in_edge is not None and out_edge is not None:
            if (isinstance(inner_desc, data.Array) and input_is_direct and output_is_direct
                    and input_is_safe_view and output_is_safe_view and self._same_outer_access(in_edge, out_edge)):
                return input_expr, prelude, postlude

            temp_name = self._nested_temp_name(cfg, state_id, dfg, node, connector)
            temp_stream = PythonCodeIOStream()
            self._allocate_nested_temp(temp_name, inner_desc, temp_stream, cfg, state_id, node)
            prelude.extend(line for line in temp_stream.getvalue().splitlines() if line.strip())
            init_target = self._temp_init_target(temp_name, inner_desc, in_edge.data.dst_subset)
            prelude.append(f"{init_target} = {input_expr}")
            result_expr = self._temp_result_expr(temp_name, inner_desc, out_edge.data.src_subset,
                                                 scalarize=not output_is_direct)
            postlude.append(f"{output_target} = {result_expr}")
            return temp_name, prelude, postlude

        if in_edge is not None:
            if isinstance(inner_desc, data.Scalar) or (input_is_direct and input_is_safe_view):
                return input_expr, prelude, postlude

            temp_name = self._nested_temp_name(cfg, state_id, dfg, node, connector)
            temp_stream = PythonCodeIOStream()
            self._allocate_nested_temp(temp_name, inner_desc, temp_stream, cfg, state_id, node)
            prelude.extend(line for line in temp_stream.getvalue().splitlines() if line.strip())
            init_target = self._temp_init_target(temp_name, inner_desc, in_edge.data.dst_subset)
            prelude.append(f"{init_target} = {input_expr}")
            return temp_name, prelude, postlude

        if out_edge is not None:
            if isinstance(inner_desc, data.Array) and output_is_direct and output_is_safe_view:
                return output_expr, prelude, postlude

            temp_name = self._nested_temp_name(cfg, state_id, dfg, node, connector)
            temp_stream = PythonCodeIOStream()
            self._allocate_nested_temp(temp_name, inner_desc, temp_stream, cfg, state_id, node)
            prelude.extend(line for line in temp_stream.getvalue().splitlines() if line.strip())
            result_expr = self._temp_result_expr(temp_name, inner_desc, out_edge.data.src_subset,
                                                 scalarize=not output_is_direct)
            postlude.append(f"{output_target} = {result_expr}")
            return temp_name, prelude, postlude

        raise KeyError(f"Connector {connector} is not connected on NestedSDFG {node.label}")

    def _nested_sdfg_label(self, cfg: ControlFlowRegion, state_id: int, dfg: StateSubgraphView,
                           node: nodes.NestedSDFG) -> tuple[str, bool]:
        unique_functions_conf = Config.get('compiler', 'unique_functions')

        if unique_functions_conf is True:
            unique_functions_conf = 'hash'
        elif unique_functions_conf is False:
            unique_functions_conf = 'none'

        if unique_functions_conf == 'hash':
            unique_functions = True
            unique_functions_hash = True
        elif unique_functions_conf == 'unique_name':
            unique_functions = True
            unique_functions_hash = False
        elif unique_functions_conf == 'none':
            unique_functions = False
            unique_functions_hash = False
        else:
            raise ValueError(f"Unknown unique_functions configuration: {unique_functions_conf}")

        if unique_functions and not unique_functions_hash and node.unique_name:
            sdfg_label = node.unique_name
        else:
            sdfg_label = f"{node.sdfg.name}_{cfg.cfg_id}_{state_id}_{dfg.node_id(node)}"

        code_already_generated = False
        if unique_functions:
            sdfg_hash = node.sdfg.hash_sdfg()
            if unique_functions_hash:
                if sdfg_hash in self._generated_nested_sdfg:
                    code_already_generated = True
                    sdfg_label = self._generated_nested_sdfg[sdfg_hash]
                else:
                    self._generated_nested_sdfg[sdfg_hash] = sdfg_label
            else:
                if sdfg_label in self._generated_nested_sdfg:
                    code_already_generated = True
                    if sdfg_hash != self._generated_nested_sdfg[sdfg_label]:
                        raise ValueError(f"Different Nested SDFGs have the same unique name: {sdfg_label}")
                else:
                    self._generated_nested_sdfg[sdfg_label] = sdfg_hash

        return sdfg_label, code_already_generated

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

    def _generate_NestedSDFG(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                             node: nodes.NestedSDFG, function_stream: PythonCodeIOStream,
                             callsite_stream: PythonCodeIOStream) -> None:
        state = cfg.nodes()[state_id]
        sdfg_label, code_already_generated = self._nested_sdfg_label(cfg, state_id, dfg, node)

        in_edges = {
            edge.dst_conn: edge
            for edge in sorted(state.in_edges(node), key=lambda edge: edge.dst_conn or '')
            if edge.dst_conn is not None and edge.data.data is not None
        }
        out_edges = {
            edge.src_conn: edge
            for edge in sorted(state.out_edges(node), key=lambda edge: edge.src_conn or '')
            if edge.src_conn is not None and edge.data.data is not None
        }

        fsyms = self._frame.free_symbols(node.sdfg)
        arglist = node.sdfg.arglist(scalars_only=False, free_symbols=fsyms)
        used_symbols = node.sdfg.used_symbols(all_symbols=False, keep_defined_in_mapping=True)

        signature_args: list[str] = []
        call_args: list[str] = []
        prelude: list[str] = []
        postlude: list[str] = []
        returned_scalars: list[str] = []

        for arg_name in arglist.keys():
            if arg_name in in_edges or arg_name in out_edges:
                arg_expr, arg_prelude, arg_postlude = self._build_nested_data_binding(sdfg, cfg, dfg, state_id, node,
                                                                                      arg_name, in_edges.get(arg_name),
                                                                                      out_edges.get(arg_name))
                signature_args.append(arg_name)
                call_args.append(arg_expr)
                prelude.extend(arg_prelude)
                postlude.extend(arg_postlude)
                if arg_name in out_edges and isinstance(node.sdfg.arrays[arg_name], data.Scalar):
                    returned_scalars.append(arg_expr)
            elif arg_name in node.symbol_mapping and arg_name in used_symbols and arg_name not in sdfg.constants:
                signature_args.append(arg_name)
                call_args.append(symbolic.symstr(node.symbol_mapping[arg_name]))

        for stmt in prelude:
            callsite_stream.write(stmt, cfg, state_id, node)

        if not code_already_generated:
            global_code, local_code, _, used_environments = self._frame.generate_code(node.sdfg, None, sdfg_label)
            self._dispatcher._used_environments |= used_environments

            function_stream.write(global_code)
            function_stream.write(f"def {sdfg_label}({', '.join(signature_args)}):", cfg, state_id, node)
            with function_stream.indented():
                self._frame.generate_constants(node.sdfg, function_stream)
                if local_code.strip():
                    function_stream.write(local_code)
                if returned_scalars:
                    function_stream.write(f"return {', '.join(name for name in signature_args if name in out_edges and isinstance(node.sdfg.arrays[name], data.Scalar))}", cfg, state_id, node)
                elif not local_code.strip():
                    function_stream.write('pass', cfg, state_id, node)
            function_stream.write('', cfg, state_id, node)

        call_expr = f"{sdfg_label}({', '.join(call_args)})"
        if returned_scalars:
            if len(returned_scalars) == 1:
                callsite_stream.write(f"{returned_scalars[0]} = {call_expr}", cfg, state_id, node)
            else:
                callsite_stream.write(f"{', '.join(returned_scalars)} = {call_expr}", cfg, state_id, node)
        else:
            callsite_stream.write(call_expr, cfg, state_id, node)

        for stmt in postlude:
            callsite_stream.write(stmt, cfg, state_id, node)

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
