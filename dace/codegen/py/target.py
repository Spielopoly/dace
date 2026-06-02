# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import os
import shutil  # which
from typing import List, TYPE_CHECKING
import warnings

from dace import memlet as mm, data as dt, dtypes
from dace.sdfg import nodes, SDFG, SDFGState, ScopeSubgraphView, graph as gr
from dace.registry import make_registry
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.codegen.codeobject import CodeObject
from dace.sdfg.state import ControlFlowRegion
from dace.codegen.target import TargetCodeGeneratorBase

if TYPE_CHECKING:
    from dace.codegen.targets.framecode import DaCeCodeGenerator


@make_registry
class PythonTargetCodeGenerator(TargetCodeGeneratorBase):
    """
    Interface dictating functions that generate code for:

        * Array allocation/deallocation/initialization/copying
        * Scope (map, consume) code generation
        
    This is the python version of the TargetCodeGenerator, which is used by the Python backend.
    """


    def generate_state(self, sdfg: SDFG, cfg: ControlFlowRegion, state: SDFGState, function_stream: PythonCodeIOStream,
                       callsite_stream: PythonCodeIOStream, generate_state_footer: bool) -> None:
        """ Generates code for an SDFG state, outputting it to the given
            code streams.

            :param sdfg: The SDFG to generate code from.
            :param state: The SDFGState to generate code from.
            :param function_stream: A `PythonCodeIOStream` object that will be
                                    generated outside the calling code, for
                                    use when generating global functions.
            :param callsite_stream: A `PythonCodeIOStream` object that points
                                    to the current location (call-site)
                                    in the code.
        """
        pass

    def write_and_resolve_expr(self,
                            memlet: mm.Memlet,
                            current_expr: str,
                            new_expr: str) -> str:
        """
        Emits a conflict resolution call from a memlet.
        """
        raise NotImplementedError('Abstract class')

    def generate_scope(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg_scope: ScopeSubgraphView, state_id: int,
                       function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        """ Generates code for an SDFG state scope (from a scope-entry node
            to its corresponding scope-exit node), outputting it to the given
            code streams.

            :param sdfg: The SDFG to generate code from.
            :param dfg_scope: The `ScopeSubgraphView` to generate code from.
            :param state_id: The node ID of the state in the given SDFG.
            :param function_stream: A `PythonCodeIOStream` object that will be
                                    generated outside the calling code, for
                                    use when generating global functions.
            :param callsite_stream: A `PythonCodeIOStream` object that points
                                    to the current location (call-site)
                                    in the code.
        """
        raise NotImplementedError('Abstract class')

    def generate_node(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: SDFGState, state_id: int, node: nodes.Node,
                      function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        """ Generates code for a single node, outputting it to the given
            code streams.

            :param sdfg: The SDFG to generate code from.
            :param dfg: The SDFG state to generate code from.
            :param state_id: The node ID of the state in the given SDFG.
            :param node: The node to generate code from.
            :param function_stream: A `PythonCodeIOStream` object that will be
                                    generated outside the calling code, for
                                    use when generating global functions.
            :param callsite_stream: A `PythonCodeIOStream` object that points
                                    to the current location (call-site)
                                    in the code.
        """
        raise NotImplementedError('Abstract class')

    def declare_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: SDFGState, state_id: int, node: nodes.Node,
                      nodedesc: dt.Data, global_stream: PythonCodeIOStream, declaration_stream: PythonCodeIOStream) -> None:
        """ Generates code for declaring an array without allocating it,
            outputting to the given code streams.

            :param sdfg: The SDFG to generate code from.
            :param dfg: The SDFG state to generate code from.
            :param state_id: The node ID of the state in the given SDFG.
            :param node: The data node to generate allocation for.
            :param nodedesc: The data descriptor to allocate.
            :param global_stream: A `PythonCodeIOStream` object that will be
                                    generated outside the calling code, for
                                    use when generating global functions.
            :param declaration_stream: A `PythonCodeIOStream` object that points
                                       to the point of array declaration.
        """
        raise NotImplementedError('Abstract class')

    def allocate_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: SDFGState, state_id: int, node: nodes.Node,
                       nodedesc: dt.Data, global_stream: PythonCodeIOStream, declaration_stream: PythonCodeIOStream,
                       allocation_stream: PythonCodeIOStream) -> None:
        """ Generates code for allocating an array, outputting to the given
            code streams.

            :param sdfg: The SDFG to generate code from.
            :param dfg: The SDFG state to generate code from.
            :param state_id: The node ID of the state in the given SDFG.
            :param node: The data node to generate allocation for.
            :param nodedesc: The data descriptor to allocate.
            :param global_stream: A `PythonCodeIOStream` object that will be
                                    generated outside the calling code, for
                                    use when generating global functions.
            :param declaration_stream: A `PythonCodeIOStream` object that points
                                       to the point of array declaration.
            :param allocation_stream: A `PythonCodeIOStream` object that points
                                       to the call-site of array allocation.
        """
        raise NotImplementedError('Abstract class')

    def deallocate_array(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: SDFGState, state_id: int, node: nodes.Node,
                         nodedesc: dt.Data, function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        """ Generates code for deallocating an array, outputting to the given
            code streams.

            :param sdfg: The SDFG to generate code from.
            :param dfg: The SDFG state to generate code from.
            :param state_id: The node ID of the state in the given SDFG.
            :param node: The data node to generate deallocation for.
            :param nodedesc: The data descriptor to deallocate.
            :param function_stream: A `PythonCodeIOStream` object that will be
                                    generated outside the calling code, for
                                    use when generating global functions.
            :param callsite_stream: A `PythonCodeIOStream` object that points
                                    to the current location (call-site)
                                    in the code.
        """
        raise NotImplementedError('Abstract class')

    def copy_memory(self, sdfg: SDFG, cfg: ControlFlowRegion, dfg: SDFGState, state_id: int, src_node: nodes.Node,
                    dst_node: nodes.Node, edge: gr.MultiConnectorEdge[mm.Memlet], function_stream: PythonCodeIOStream,
                    callsite_stream: PythonCodeIOStream) -> None:
        """ Generates code for copying memory, either from a data access
            node (array/stream) to another, a code node (tasklet/nested
            SDFG) to another, or a combination of the two.

            :param sdfg: The SDFG to generate code from.
            :param dfg: The SDFG state to generate code from.
            :param state_id: The node ID of the state in the given SDFG.
            :param src_node: The source node to generate copy code for.
            :param dst_node: The destination node to generate copy code for.
            :param edge: The edge representing the copy (in the innermost
                         scope, adjacent to either the source or destination
                         node).
            :param function_stream: A `PythonCodeIOStream` object that will be
                                    generated outside the calling code, for
                                    use when generating global functions.
            :param callsite_stream: A `PythonCodeIOStream` object that points
                                    to the current location (call-site)
                                    in the code.
        """
        raise NotImplementedError('Abstract class')

    def emit_interstate_variable_declaration(self, name: str, dtype: dtypes.typeclass, callsite_stream: PythonCodeIOStream,
                                             sdfg: SDFG):
        """ Emits the declaration of an interstate variable at the given
            call-site.

            :param name: The name of the variable.
            :param dtype: The data type of the variable.
            :param callsite_stream: A ``PythonCodeIOStream`` object that points
                                    to the current location (call-site)
                                    in the code.
            :param sdfg: The SDFG in which the variable is declared.
        """
        raise NotImplementedError('Abstract class')


class IllegalCopy(PythonTargetCodeGenerator):
    """ A code generator that is triggered when invalid copies are specified
        by the SDFG. Only raises an exception on failure. """

    def copy_memory(self, sdfg, cfg, dfg, state_id, src_node, dst_node, edge, function_stream, callsite_stream):
        raise TypeError('Illegal copy! (from ' + str(src_node) + ' to ' + str(dst_node) + ')')
