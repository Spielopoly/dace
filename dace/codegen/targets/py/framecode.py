# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import collections
import copy
import pathlib
import re
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Union

import numpy as np

import dace
from dace import config, data, dtypes
from dace.cli import progress
from dace.codegen import control_flow as cflow
from dace.codegen import dispatcher as disp
from dace.codegen.prettycode import CodeIOStream
from dace.codegen.target import TargetCodeGenerator
from dace.sdfg.type_inference import infer_expr_type
from dace.sdfg import SDFG, SDFGState, nodes
from dace.sdfg import scope as sdscope
from dace.sdfg import utils
from dace.sdfg.analysis import cfg as cfg_analysis
from dace.sdfg.state import ControlFlowBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.analysis import StateReachability, loop_analysis


def _get_or_eval_sdfg_first_arg(func, sdfg):
    if callable(func):
        return func(sdfg)
    return func


class DaCePythonCodeGenerator(object):
    """ DaCe code generator class that writes the generated code for SDFG
        state machines, and uses a dispatcher to generate code for
        individual states based on the target. """

    def __init__(self, sdfg: SDFG):
        self._dispatcher = disp.TargetDispatcher(self)
        self._dispatcher.register_state_dispatcher(self)
        self._initcode = CodeIOStream()
        self._exitcode = CodeIOStream()
        self.statestruct: List[str] = []
        self.environments: List[Any] = []
        self.targets: Set[TargetCodeGenerator] = set()
        self.to_allocate: DefaultDict[Union[SDFG, SDFGState, nodes.EntryNode],
                                      List[Tuple[SDFG, Optional[SDFGState], Optional[nodes.AccessNode], bool, bool,
                                                 bool]]] = collections.defaultdict(list)
        self.where_allocated: Dict[Tuple[SDFG, str], SDFG] = {}
        self.fsyms: Dict[int, Set[str]] = {}
        self._symbols_and_constants: Dict[int, Set[str]] = {}
        fsyms = self.free_symbols(sdfg)
        self.arglist = sdfg.arglist(scalars_only=False, free_symbols=fsyms)

        # resolve all symbols and constants
        # first handle root
        sdfg.reset_cfg_list()
        self._symbols_and_constants[sdfg.cfg_id] = sdfg.free_symbols.union(sdfg.constants_prop.keys())
        # then recurse
        for nested, state in sdfg.all_nodes_recursive():
            if isinstance(nested, nodes.NestedSDFG):
                state: SDFGState

                nsdfg = nested.sdfg

                # found a new nested sdfg: resolve symbols and constants
                result = nsdfg.free_symbols.union(nsdfg.constants_prop.keys())

                parent_constants = self._symbols_and_constants[nsdfg.parent_sdfg.cfg_id]
                result |= parent_constants

                # check for constant inputs
                for edge in state.in_edges(nested):
                    if edge.data.data in parent_constants:
                        # this edge is constant => propagate to nested sdfg
                        result.add(edge.dst_conn)

                self._symbols_and_constants[nsdfg.cfg_id] = result

    # Cached fields
    def symbols_and_constants(self, sdfg: SDFG):
        return self._symbols_and_constants[sdfg.cfg_id]

    def free_symbols(self, obj: Any):
        k = id(obj)
        if k in self.fsyms:
            return self.fsyms[k]
        if hasattr(obj, 'used_symbols'):
            result = obj.used_symbols(all_symbols=False)
        else:
            result = obj.free_symbols
        self.fsyms[k] = result
        return result

    ##################################################################
    # Target registry

    @property
    def dispatcher(self):
        return self._dispatcher

    ##################################################################
    # Code generation

    def preprocess(self, sdfg: SDFG) -> None:
        """
        Called before code generation. Used for making modifications on the SDFG prior to code generation.

        :note: Post-conditions assume that the SDFG will NOT be changed after this point.
        :param sdfg: The SDFG to modify in-place.
        """
        raise NotImplementedError()

    def generate_constants(self, sdfg: SDFG, callsite_stream: CodeIOStream):
        raise NotImplementedError()

    def generate_fileheader(self, sdfg: SDFG, global_stream: CodeIOStream, backend: str = 'frame'):
        """ Generate a header in every output file that includes custom types
            and constants.

            :param sdfg: The input SDFG.
            :param global_stream: Stream to write to (global).
            :param backend: Whose backend this header belongs to.
        """
        raise NotImplementedError()

    def generate_header(self, sdfg: SDFG, global_stream: CodeIOStream, callsite_stream: CodeIOStream):
        """ Generate the header of the frame-code. Code exists in a separate
            function for overriding purposes.

            :param sdfg: The input SDFG.
            :param global_stream: Stream to write to (global).
            :param callsite_stream: Stream to write to (at call site).
        """

    def generate_footer(self, sdfg: SDFG, global_stream: CodeIOStream, callsite_stream: CodeIOStream):
        """ Generate the footer of the frame-code. Code exists in a separate
            function for overriding purposes.

            :param sdfg: The input SDFG.
            :param global_stream: Stream to write to (global).
            :param callsite_stream: Stream to write to (at call site).
        """
        raise NotImplementedError()

    def generate_external_memory_management(self, sdfg: SDFG, callsite_stream: CodeIOStream):
        """
        If external data descriptors are found in the SDFG (or any nested SDFGs),
        this function will generate exported functions to (1) get the required memory size
        per storage location (``__dace_get_external_memory_size_<STORAGE>``, where ``<STORAGE>``
        can be ``CPU_Heap`` or any other ``dtypes.StorageType``); and (2) set the externally-allocated
        pointer to the generated code's internal state (``__dace_set_external_memory_<STORAGE>``).
        """
        raise NotImplementedError()

    def generate_state(self,
                       sdfg: SDFG,
                       cfg: ControlFlowRegion,
                       state: SDFGState,
                       global_stream: CodeIOStream,
                       callsite_stream: CodeIOStream,
                       generate_state_footer: bool = True):
        raise NotImplementedError()

    def generate_states(self, sdfg: SDFG, global_stream: CodeIOStream, callsite_stream: CodeIOStream) -> Set[SDFGState]:
        raise NotImplementedError()

    def _get_schedule(self, scope: Union[nodes.EntryNode, SDFGState, SDFG]) -> dtypes.ScheduleType:
        raise NotImplementedError()

    def _can_allocate(self, sdfg: SDFG, state: SDFGState, desc: data.Data, scope: Union[nodes.EntryNode, SDFGState,
                                                                                        SDFG]) -> bool:
        raise NotImplementedError()

    def determine_allocation_lifetime(self, top_sdfg: SDFG):
        """
        Determines where (at which scope/state/SDFG) each data descriptor will be allocated/deallocated.

        :param top_sdfg: The top-level SDFG to determine for.
        """
        raise NotImplementedError()

    def allocate_arrays_in_scope(self, sdfg: SDFG, cfg: ControlFlowRegion, scope: Union[nodes.EntryNode, SDFGState,
                                                                                        SDFG],
                                 function_stream: CodeIOStream, callsite_stream: CodeIOStream) -> None:
        raise NotImplementedError()

    def deallocate_arrays_in_scope(self, sdfg: SDFG, cfg: ControlFlowRegion, scope: Union[nodes.EntryNode, SDFGState,
                                                                                          SDFG],
                                   function_stream: CodeIOStream, callsite_stream: CodeIOStream):
        raise NotImplementedError()

    def generate_code(self,
                      sdfg: SDFG,
                      schedule: Optional[dtypes.ScheduleType],
                      cfg_id: str = "") -> Tuple[str, str, Set[TargetCodeGenerator], Set[str]]:
        """ Generate frame code for a given SDFG, calling registered targets'
            code generation callbacks for them to generate their own code.

            :param sdfg: The SDFG to generate code for.
            :param schedule: The schedule the SDFG is currently located, or
                             None if the SDFG is top-level.
            :param cfg_id An optional string id given to the SDFG label
            :return: A tuple of the generated global frame code, local frame
                     code, and a set of targets that have been used in the
                     generation of this SDFG.
        """
        raise NotImplementedError()
