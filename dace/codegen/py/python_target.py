# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Python target code generator for the DaCe Python backend."""

import ast
import contextlib
import copy
import itertools
import re
from typing import TYPE_CHECKING, Optional

from dace import data, dtypes, subsets
from dace.codegen.dispatcher import TargetDispatcher
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.codegen.target import TargetCodeGenerator
from dace.memlet import Memlet
from dace.sdfg import NodeNotExpandedError, SDFG, ScopeSubgraphView, nodes, scope_contains_scope
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import ControlFlowRegion, StateSubgraphView

if TYPE_CHECKING:
    from dace.codegen.py.framecode import DaCePythonCodeGenerator


class PythonCodeGen(TargetCodeGenerator):
    """Pure-Python code generator for SDFG nodes, scopes, and copies."""

    title = "Python"
    target_name = "python"
    language = "python"

    def get_includes(self) -> dict[str, list[str]]:
        return {'frame': ['numpy']}

    def preprocess(self, sdfg: SDFG) -> None:
        """Reject scopes that the Python backend cannot lower before dispatch starts."""
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(node, nodes.MapEntry) and node.map.schedule != dtypes.ScheduleType.Sequential:
                raise NotImplementedError(
                    f'Python backend only supports sequential maps, got {node.map.schedule}')

    def __init__(self, frame_codegen: 'DaCePythonCodeGenerator', sdfg: SDFG):
        self._frame = frame_codegen
        self._dispatcher: TargetDispatcher = frame_codegen.dispatcher
        self._generated_nested_sdfgs: dict[int, tuple[str, list[str]]] = {}
        dispatcher = self._dispatcher

        # Register as generic node dispatcher
        dispatcher.register_node_dispatcher(self)

        dispatcher.register_map_dispatcher([dtypes.ScheduleType.Sequential], self)

        cpu_storage = [dtypes.StorageType.CPU_Heap, dtypes.StorageType.CPU_ThreadLocal, dtypes.StorageType.Register]
        for storage in cpu_storage:
            dispatcher.register_array_dispatcher(storage, self)

        # Register copy dispatchers for all CPU storage pairs
        for src_storage, dst_storage in itertools.product(cpu_storage, cpu_storage):
            dispatcher.register_copy_dispatcher(src_storage, dst_storage, None, self)

    def generate_node(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                      node: nodes.Node, function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        gen = getattr(self, "_generate_" + type(node).__name__, None)
        if gen is None:
            if isinstance(node, nodes.LibraryNode):
                raise NodeNotExpandedError(sdfg, state_id, dfg.node_id(node))
            raise NotImplementedError(f'Python backend: no code generator for node type {type(node).__name__}')
        gen(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)

    def generate_scope(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, state_id: int,
                       function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        entry_node = dfg_scope.source_nodes()[0]
        if entry_node.map.schedule != dtypes.ScheduleType.Sequential:
            raise NotImplementedError(
                f'Python backend only supports sequential maps, got {entry_node.map.schedule}')

        if hasattr(self._frame, 'allocate_arrays_in_scope'):
            self._frame.allocate_arrays_in_scope(sdfg, cfg, entry_node, function_stream, callsite_stream)

        with contextlib.ExitStack() as loop_stack:
            for param, (start, end, step) in zip(entry_node.map.params, entry_node.map.range):
                callsite_stream.write(self._map_range_statement(str(param), start, end, step), cfg, state_id)
                loop_stack.enter_context(callsite_stream.indented())

            pos_before = callsite_stream.tell()
            self._dispatcher.dispatch_subgraph(sdfg,
                                               cfg,
                                               dfg_scope,
                                               state_id,
                                               function_stream,
                                               callsite_stream,
                                               skip_entry_node=True,
                                               skip_exit_node=True)
            if callsite_stream.tell() == pos_before:
                callsite_stream.write('pass', cfg, state_id)

        if hasattr(self._frame, 'deallocate_arrays_in_scope'):
            self._frame.deallocate_arrays_in_scope(sdfg, cfg, entry_node, function_stream, callsite_stream)

    def _generate_AccessNode(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                             node: nodes.AccessNode, function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        state = cfg.state(state_id)
        sdict = state.scope_dict()

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

    def _generate_Tasklet(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                          node: nodes.Tasklet, function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        state = cfg.state(state_id)

        # --- Read inputs ---
        for edge in state.in_edges(node):
            if not edge.dst_conn:
                continue
            if edge.data.data is None:
                raise NotImplementedError('Code-to-code memlets not supported in Python backend')

            desc = sdfg.arrays[edge.data.data]
            if isinstance(desc, data.Stream):
                raise NotImplementedError('Python backend does not support Stream descriptors')

            subset = self._source_subset(edge.data)
            expr = self._read_expr(edge.data.data, desc, subset, copy_value=not isinstance(desc, data.Scalar))
            callsite_stream.write(f'{edge.dst_conn} = {expr}', cfg, state_id)

        # --- Tasklet body ---
        if node.code.language != dtypes.Language.Python:
            raise NotImplementedError(f'Python backend only supports Python tasklets, got {node.code.language}')

        for statement in node.code.code:
            callsite_stream.write(ast.unparse(statement) if isinstance(statement, ast.AST) else str(statement), cfg, state_id)

        for edge in state.out_edges(node):
            if not edge.src_conn:
                continue
            if edge.data.data is None:
                raise NotImplementedError('Code-to-code memlets not supported in Python backend')

            self._write_memlet_value(sdfg,
                                     cfg,
                                     state_id,
                                     node,
                                     edge.data,
                                     edge.src_conn,
                             callsite_stream,
                                     subset=self._destination_subset(edge.data))

    def _generate_NestedSDFG(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                         node: nodes.NestedSDFG, function_stream: PythonCodeIOStream,
                         callsite_stream: PythonCodeIOStream) -> None:
        helper_name, argnames = self._get_or_generate_nested_helper(sdfg,
                                                                    cfg,
                                                                    dfg,
                                                                    state_id,
                                                                    node,
                                            function_stream)
        state = cfg.state(state_id)

        inputs: dict[str, str] = {}
        output_writebacks: list[tuple[Memlet, str]] = []
        for edge in state.in_edges(node):
            if not edge.dst_conn:
                continue
            if edge.data.data is None:
                raise NotImplementedError('Code-to-code memlets not supported in Python backend')
            desc = sdfg.arrays[edge.data.data]
            inputs[edge.dst_conn] = self._nested_input_expr(edge.data.data,
                                                            desc,
                                                            self._source_subset(edge.data),
                                                            node.sdfg.arrays[edge.dst_conn])

        outputs: dict[str, str] = {}
        for edge in state.out_edges(node):
            if not edge.src_conn:
                continue
            if edge.data.data is None:
                raise NotImplementedError('Code-to-code memlets not supported in Python backend')
            desc = sdfg.arrays[edge.data.data]
            output_argument, output_writeback = self._nested_output_expr(edge.data.data,
                                                                         desc,
                                                                         self._destination_subset(edge.data),
                                                                         node.sdfg.arrays[edge.src_conn],
                                                                         edge.src_conn,
                                                                         cfg,
                                                                         state_id,
                                                                         callsite_stream)
            outputs[edge.src_conn] = output_argument
            if output_writeback is not None:
                output_writebacks.append((edge.data, output_writeback))

        arguments: list[str] = []
        for argname in argnames:
            if argname in inputs:
                arguments.append(f'{argname}={inputs[argname]}')
            elif argname in outputs:
                arguments.append(f'{argname}={outputs[argname]}')
            elif argname in node.symbol_mapping:
                arguments.append(f'{argname}={node.symbol_mapping[argname]}')
            elif argname in self._frame.symbols_and_constants(sdfg):
                arguments.append(f'{argname}={argname}')
            else:
                raise NotImplementedError(f'Python backend could not map nested SDFG argument {argname!r}')

        callsite_stream.write(f'{helper_name}({", ".join(arguments)})', cfg, state_id)
        for memlet, value_expr in output_writebacks:
            self._write_memlet_value(sdfg,
                                     cfg,
                                     state_id,
                                     node,
                                     memlet,
                                     value_expr,
                                     callsite_stream,
                                     subset=self._destination_subset(memlet))

    def _generate_ConsumeEntry(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                               node: nodes.ConsumeEntry, function_stream: PythonCodeIOStream,
                               callsite_stream: PythonCodeIOStream) -> None:
        raise NotImplementedError('Python backend does not support Consume scopes')

    def _generate_ConsumeExit(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                              node: nodes.ConsumeExit, function_stream: PythonCodeIOStream,
                              callsite_stream: PythonCodeIOStream) -> None:
        raise NotImplementedError('Python backend does not support Consume scopes')

    def declare_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg, state_id: int, node: nodes.Node,
                      nodedesc: data.Data, global_stream: PythonCodeIOStream,
                      declaration_stream: PythonCodeIOStream) -> None:
        pass

    def allocate_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg, state_id: int, node: nodes.Node,
                       nodedesc: data.Data, global_stream: PythonCodeIOStream,
                       declaration_stream: PythonCodeIOStream, allocation_stream: PythonCodeIOStream) -> None:
        if not isinstance(node, nodes.AccessNode):
            raise TypeError('Python backend allocation expects AccessNode instances')
        if not nodedesc.transient:
            return
        if isinstance(nodedesc, data.Stream):
            raise NotImplementedError('Python backend does not support Stream descriptors')
        if isinstance(nodedesc, (data.View, data.Reference)):
            if state_id < 0:
                return
            state = cfg.state(state_id)
            if not state.in_edges(node):
                return
            edge = state.in_edges(node)[0]
            source_node = state.memlet_path(edge)[0].src
            if not isinstance(source_node, nodes.AccessNode):
                return
            source_desc = source_node.desc(sdfg)
            source_expr = self._reference_expr(source_node.data, source_desc, self._source_subset(edge.data))
            allocation_stream.write(f'{node.data} = {source_expr}', cfg, state_id)
            return
        if isinstance(nodedesc, data.Scalar):
            allocation_stream.write(f'{node.data} = {self._zero_value(nodedesc)}', cfg, state_id)
            return
        if isinstance(nodedesc, data.Array):
            allocation_stream.write(
                f'{node.data} = numpy.zeros({self._shape_expr(nodedesc.shape)}, dtype={self._dtype_expr(nodedesc)})',
                cfg,
                state_id,
            )
            return
        raise NotImplementedError(f'Python backend: cannot allocate {type(nodedesc).__name__}')

    def deallocate_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg, state_id: int, node: nodes.Node,
                         nodedesc: data.Data, function_stream: PythonCodeIOStream,
                         callsite_stream: PythonCodeIOStream) -> None:
        pass

    def copy_memory(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg, state_id: int, src_node: nodes.Node,
                    dst_node: nodes.Node, edge: MultiConnectorEdge[Memlet], function_stream: PythonCodeIOStream,
                    callsite_stream: PythonCodeIOStream) -> None:
        memlet = edge.data

        src_desc = src_node.desc(sdfg) if isinstance(src_node, nodes.AccessNode) else None
        dst_desc = dst_node.desc(sdfg) if isinstance(dst_node, nodes.AccessNode) else None
        if isinstance(src_desc, data.Stream) or isinstance(dst_desc, data.Stream):
            raise NotImplementedError('Python backend does not support Stream descriptors')

        if isinstance(src_node, nodes.AccessNode):
            assert src_desc is not None
            src_expr = self._reference_expr(src_node.data, src_desc, self._source_subset(memlet))
        else:
            src_expr = edge.src_conn

        if isinstance(dst_node, nodes.AccessNode):
            if isinstance(dst_desc, (data.View, data.Reference)):
                callsite_stream.write(f'{dst_node.data} = {src_expr}', cfg, state_id)
                return
            self._write_memlet_value(sdfg,
                                     cfg,
                                     state_id,
                                     dst_node,
                                     memlet,
                                     src_expr,
                                     callsite_stream,
                                     target_name=dst_node.data,
                                     target_desc=dst_desc,
                                     subset=self._destination_subset(memlet))
            return

        if edge.dst_conn is None:
            raise NotImplementedError('Python backend cannot lower copies to unnamed connectors')

        if memlet.wcr is not None:
            callsite_stream.write(f'{edge.dst_conn} = {self._wcr_expr(memlet, edge.dst_conn, src_expr)}', cfg,
                                  state_id)
        else:
            callsite_stream.write(f'{edge.dst_conn} = {src_expr}', cfg, state_id)

    def define_out_memlet(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg, state_id: int, src_node: nodes.Node,
                          dst_node: nodes.Node, edge: MultiConnectorEdge[Memlet],
                          function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        pass

    def emit_interstate_variable_declaration(self, name: str, dtype: dtypes.typeclass,
                                             callsite_stream: PythonCodeIOStream, sdfg: SDFG):
        pass

    def _get_or_generate_nested_helper(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: StateSubgraphView, state_id: int,
                                       node: nodes.NestedSDFG,
                                       function_stream: PythonCodeIOStream) -> tuple[str, list[str]]:
        helper = self._generated_nested_sdfgs.get(id(node))
        if helper is not None:
            return helper

        from dace.codegen.py.framecode import DaCePythonCodeGenerator

        helper_name = self._nested_helper_name(sdfg, state_id, dfg.node_id(node), node.sdfg.name)
        nested_sdfg = copy.deepcopy(node.sdfg)
        nested_sdfg.backend = dtypes.BackendLanguage.Python
        nested_sdfg.parent = None
        nested_sdfg.parent_nsdfg_node = None
        nested_sdfg.reset_cfg_list()
        nested_codegen = DaCePythonCodeGenerator(nested_sdfg)
        nested_target = PythonCodeGen(nested_codegen, nested_sdfg)
        nested_codegen.targets.add(nested_target)
        nested_target.preprocess(nested_sdfg)
        nested_preamble_stream = PythonCodeIOStream()
        nested_codegen.generate_embedded_function_preamble(nested_sdfg, nested_preamble_stream)
        nested_finalizer_stream = PythonCodeIOStream()
        nested_codegen.generate_embedded_function_finalizer(nested_sdfg, nested_finalizer_stream)
        nested_header, nested_body, nested_targets, nested_environments = nested_codegen.generate_code(
            nested_sdfg,
            None,
            function_name=helper_name,
            include_lifecycle=False,
            include_file_header=False,
            function_body_preamble=nested_preamble_stream.getvalue(),
            function_body_finally=nested_finalizer_stream.getvalue())
        self._dispatcher.used_targets.update(nested_targets)
        self._dispatcher.used_environments.update(nested_environments)
        nested_source = nested_header + nested_body
        if nested_source:
            function_stream.write(nested_source, node.sdfg)

        helper = (helper_name, list(nested_codegen.arglist.keys()))
        self._generated_nested_sdfgs[id(node)] = helper
        return helper

    def _write_memlet_value(self, sdfg: SDFG, cfg: ControlFlowRegion, state_id: int, anchor: nodes.Node,
                            memlet: Memlet, value_expr: str, callsite_stream: PythonCodeIOStream,
                            target_name: Optional[str] = None, target_desc: Optional[data.Data] = None,
                            subset=None) -> None:
        name = target_name or memlet.data
        assert name is not None
        desc = target_desc or sdfg.arrays[name]
        if isinstance(desc, data.Stream):
            raise NotImplementedError('Python backend does not support Stream descriptors')

        target_expr = self._reference_expr(name, desc, subset)
        statement: str
        if memlet.wcr is not None:
            current_expr = target_expr
            assign_expr = target_expr
            if isinstance(desc, data.Scalar) and not desc.transient:
                current_expr = f'{name}[...]'
                assign_expr = current_expr
            statement = f'{assign_expr} = {self._wcr_expr(memlet, current_expr, value_expr)}'
        elif isinstance(desc, (data.View, data.Reference)) and subset is None:
            statement = f'{name} = {value_expr}'
        elif isinstance(desc, data.Array) and subset is None:
            statement = f'numpy.copyto({name}, {value_expr})'
        elif isinstance(desc, data.Scalar) and not desc.transient:
            statement = f'{name}[...] = {value_expr}'
        else:
            statement = f'{target_expr} = {value_expr}'

        callsite_stream.write(statement, cfg, state_id)

    def _map_range_statement(self, param: str, start, end, step) -> str:
        """Lower a DaCe map range to Python's end-exclusive range semantics."""
        stop_expr = self._range_stop_expr(end, step)
        return f'for {param} in range({start}, {stop_expr}, {step}):'

    def _source_subset(self, memlet: Memlet):
        subset = getattr(memlet, 'src_subset', None)
        return subset if subset is not None else memlet.subset

    def _destination_subset(self, memlet: Memlet):
        subset = getattr(memlet, 'dst_subset', None)
        if subset is not None:
            return subset
        return memlet.other_subset

    def _reference_expr(self, name: str, desc: data.Data, subset) -> str:
        if isinstance(desc, data.Scalar) or subset is None:
            return name
        return f'{name}[{subset}]'

    def _nested_input_expr(self, name: str, parent_desc: data.Data, subset, nested_desc: data.Data) -> str:
        """Build a nested SDFG input expression that matches the callee descriptor."""
        if isinstance(parent_desc, data.Scalar) and isinstance(nested_desc, data.Array):
            self._require_single_value_nested_buffer(nested_desc, name)
            return f'numpy.asarray([{name}], dtype={self._dtype_expr(nested_desc)})'
        if isinstance(parent_desc, data.Scalar) or subset is None:
            return name
        point_indices = self._point_subset_indices(subset)
        if isinstance(nested_desc, data.Array) and point_indices is not None:
            slice_expr = ', '.join(f'{index}:{index} + 1' for index in point_indices)
            return f'{name}[{slice_expr}]'
        return self._reference_expr(name, parent_desc, subset)

    def _nested_output_expr(self,
                            name: str,
                            parent_desc: data.Data,
                            subset,
                            nested_desc: data.Data,
                            connector_name: str,
                            cfg: ControlFlowRegion,
                            state_id: int,
                            callsite_stream: PythonCodeIOStream) -> tuple[str, Optional[str]]:
        """Build a nested SDFG output argument and an optional scalar write-back expression."""
        if isinstance(parent_desc, data.Scalar):
            buffer_name = self._nested_buffer_name(name, connector_name, state_id)
            callsite_stream.write(self._nested_buffer_initialization(buffer_name, nested_desc), cfg, state_id)
            return buffer_name, self._nested_buffer_value_expr(buffer_name, nested_desc)

        if subset is None:
            return name, None

        point_indices = self._point_subset_indices(subset)
        if point_indices is not None:
            slice_expr = ', '.join(f'{index}:{index} + 1' for index in point_indices)
            return f'{name}[{slice_expr}]', None

        return self._reference_expr(name, parent_desc, subset), None

    def _point_subset_indices(self, subset):
        if isinstance(subset, subsets.Indices):
            return list(subset.indices)
        if isinstance(subset, subsets.Range):
            indices = []
            for start, end, step in subset.ranges:
                if str(start) != str(end) or str(step) != '1':
                    return None
                indices.append(start)
            return indices
        return None

    def _read_expr(self, name: str, desc: data.Data, subset, copy_value: bool) -> str:
        expr = self._reference_expr(name, desc, subset)
        if not copy_value or isinstance(desc, data.Scalar):
            return expr
        if self._point_subset_indices(subset) is not None:
            return expr
        return f'numpy.copy({expr})'

    def _wcr_expr(self, memlet: Memlet, current_expr: str, new_expr: str) -> str:
        reduction = memlet.wcr
        if isinstance(reduction, ast.AST):
            reduction_expr = ast.unparse(reduction)
        else:
            reduction_expr = str(reduction)
        return f'({reduction_expr})({current_expr}, {new_expr})'

    def _range_stop_expr(self, end, step) -> str:
        """Return the Python stop expression for an inclusive DaCe map bound."""
        step_direction = self._static_step_direction(step)
        if step_direction > 0:
            return f'({end}) + 1'
        if step_direction < 0:
            return f'({end}) - 1'
        return f'(({end}) + 1 if ({step}) > 0 else ({end}) - 1)'

    def _shape_expr(self, shape) -> str:
        dims = ', '.join(str(dim) for dim in shape)
        if len(shape) == 1:
            dims += ','
        return f'({dims})'

    def _dtype_expr(self, desc: data.Data) -> str:
        return f'numpy.{desc.dtype.to_string()}'

    def _zero_value(self, desc: data.Data) -> str:
        if desc.dtype == dtypes.bool_:
            return 'False'
        return '0'

    def _nested_helper_name(self, sdfg: SDFG, state_id: int, node_id: int, nested_name: str) -> str:
        return '__dace_nested_{root}_{nested}_{state}_{node}'.format(
            root=self._sanitize_name(sdfg.name),
            nested=self._sanitize_name(nested_name),
            state=state_id,
            node=node_id,
        )

    def _nested_buffer_name(self, data_name: str, connector_name: str, state_id: int) -> str:
        return '__dace_nested_buffer_{data}_{connector}_{state}'.format(
            data=self._sanitize_name(data_name),
            connector=self._sanitize_name(connector_name),
            state=state_id,
        )

    def _nested_buffer_initialization(self, buffer_name: str, nested_desc: data.Data) -> str:
        self._require_single_value_nested_buffer(nested_desc, buffer_name)
        return f'{buffer_name} = numpy.zeros((1,), dtype={self._dtype_expr(nested_desc)})'

    def _nested_buffer_value_expr(self, buffer_name: str, nested_desc: data.Data) -> str:
        self._require_single_value_nested_buffer(nested_desc, buffer_name)
        return f'{buffer_name}[0]'

    def _require_single_value_nested_buffer(self, nested_desc: data.Data, connector_name: str) -> None:
        if isinstance(nested_desc, data.Scalar):
            return
        if isinstance(nested_desc, data.Array) and len(nested_desc.shape) == 1 and str(nested_desc.shape[0]) == '1':
            return
        raise NotImplementedError(
            f'Python backend can only bridge scalar nested values through size-1 buffers, got {connector_name!r}')

    def _sanitize_name(self, name: str) -> str:
        return re.sub(r'\W|^(?=\d)', '_', name)

    def _static_step_direction(self, step) -> int:
        """Return -1, 0, or 1 when the map step direction is statically known.
        Returns 0 for unknown or dynamic step values."""
        try:
            numeric_step = int(step)
        except (TypeError, ValueError):
            return 0

        if numeric_step < 0:
            return -1
        if numeric_step > 0:
            return 1
        return 0
