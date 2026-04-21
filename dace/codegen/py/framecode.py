# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import ast
import collections
import copy
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Union

import numpy as np

import dace
from dace import data, dtypes
from dace.cli import progress
from dace.codegen.py import control_flow as py_cflow
from dace.codegen import dispatcher as disp
from dace.codegen.py.prettycode import PythonCodeIOStream
from dace.codegen.target import TargetCodeGenerator
from dace.frontend.python import astutils
from dace.sdfg.type_inference import infer_expr_type
from dace.sdfg import SDFG, SDFGState, nodes
from dace.sdfg import scope as sdscope
from dace.sdfg import utils
from dace.sdfg.analysis import cfg as cfg_analysis
from dace.sdfg.state import ControlFlowBlock, ControlFlowRegion, LoopRegion
from dace.transformation.passes.analysis import StateReachability, loop_analysis
from dace.properties import CodeBlock


def codeblock_to_python(cb: CodeBlock):
    if cb.language == dtypes.Language.Python:
        return cb.as_string or ""
    if cb.as_string:
        raise ValueError(f"CodeBlock language {cb.language} cannot be converted to Python.")
    # ignore empty code blocks
    return ""


def _normalize_python_import(import_entry: str) -> Optional[str]:
    import_entry = import_entry.strip()
    if not import_entry:
        return None
    if import_entry.startswith('import ') or import_entry.startswith('from '):
        return import_entry
    return f'import {import_entry}'


def _split_codeblock_imports(cb: CodeBlock) -> Tuple[List[str], str]:
    if cb.language != dtypes.Language.Python:
        return [], codeblock_to_python(cb)

    statements = cb.code if isinstance(cb.code, list) else ast.parse(cb.as_string).body
    import_statements: List[str] = []
    remaining_statements: List[ast.AST] = []
    for stmt in statements:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            import_statements.append(astutils.unparse(stmt))
        else:
            remaining_statements.append(stmt)

    remaining_code = astutils.unparse(remaining_statements) if remaining_statements else ""
    return import_statements, remaining_code


class DaCePythonCodeGenerator(object):
    """ DaCe code generator class that writes the generated code for SDFG
        state machines, and uses a dispatcher to generate code for
        individual states based on the target. """

    def __init__(self, sdfg: SDFG):
        self._dispatcher = disp.TargetDispatcher(self)
        self._dispatcher.register_state_dispatcher(self)
        self._initcode = PythonCodeIOStream()
        self._exitcode = PythonCodeIOStream()
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
        pass

    def generate_constants(self, sdfg: SDFG, callsite_stream: PythonCodeIOStream):
        # Write constants
        for cstname, (csttype, cstval) in sdfg.constants_prop.items():
            if isinstance(csttype, data.Array):
                const_str = cstname + " = ["
                it = np.nditer(cstval, order='C')
                for i in range(cstval.size - 1):
                    const_str += str(it[0]) + ", "
                    it.iternext()
                const_str += str(it[0]) + "]\n"
                callsite_stream.write(const_str, sdfg)
            else:
                callsite_stream.write(f"{cstname} = {cstval}\n", sdfg)

    def generate_fileheader(self, sdfg: SDFG, global_stream: PythonCodeIOStream, backend: str = 'frame'):
        """ Generate a header in every output file that includes custom types
            and constants.

            :param sdfg: The input SDFG.
            :param global_stream: Stream to write to (global).
            :param backend: Whose backend this header belongs to.
        """
        # TODO: mangle_dace_state_struct_name should be moved to a shared utility module
        #       instead of importing from the C++ target.
        from dace.codegen.targets.cpp import mangle_dace_state_struct_name  # Avoid circular import

        emitted_imports: Set[str] = set()

        def _write_imports(imports: List[str], import_sdfg: SDFG) -> None:
            lines: List[str] = []
            for import_entry in imports:
                normalized = _normalize_python_import(import_entry)
                if normalized is None or normalized in emitted_imports:
                    continue
                emitted_imports.add(normalized)
                lines.append(normalized)
            if lines:
                global_stream.write('\n'.join(lines), import_sdfg)

        def _write_global_code(codeblock: CodeBlock, code_sdfg: SDFG) -> None:
            import_statements, remaining_code = _split_codeblock_imports(codeblock)
            _write_imports(import_statements, code_sdfg)
            if remaining_code:
                global_stream.write(remaining_code, code_sdfg)
        
        #########################################################
        # Target-based includes
        for target in self._dispatcher.used_targets:
            headers = target.get_includes()
            if backend in headers:
                _write_imports(headers[backend], sdfg)

        # Environment-based includes
        for env in self.environments:
            if len(env.headers) > 0:
                if not isinstance(env.headers, dict):
                    headers = {'frame': env.headers}
                else:
                    headers = env.headers
                if backend in headers:
                    _write_imports(headers[backend], sdfg)

        #########################################################
        # Custom types
        datatypes = set()
        # Types of this SDFG
        for _, arrname, arr in sdfg.arrays_recursive():
            if arr is not None:
                datatypes.add(arr.dtype)

        emitted = set()

        def _emit_definitions(dtype: dtypes.typeclass, wrote_something: bool) -> bool:
            if isinstance(dtype, dtypes.pointer):
                wrote_something = _emit_definitions(dtype._typeclass, wrote_something)
            elif isinstance(dtype, dtypes.struct):
                for field in dtype.fields.values():
                    wrote_something = _emit_definitions(field, wrote_something)
            if hasattr(dtype, 'emit_definition'):
                if not wrote_something:
                    global_stream.write("", sdfg)
                if dtype not in emitted:
                    global_stream.write(dtype.emit_definition(), sdfg)
                    wrote_something = True
                    emitted.add(dtype)
            return wrote_something

        # Emit unique definitions
        wrote_something = False
        for typ in datatypes:
            wrote_something = _emit_definitions(typ, wrote_something)
        if wrote_something:
            global_stream.write("", sdfg)

        #########################################################
        # Write constants
        self.generate_constants(sdfg, global_stream)

        #########################################################
        # Write state struct (only if there are fields)
        if self.statestruct:
            structstr = '\n'.join(self.statestruct)
            global_stream.write(f'class {mangle_dace_state_struct_name(sdfg)}:', sdfg)
            with global_stream.indented():
                global_stream.write(structstr, sdfg)

        for sd in sdfg.all_sdfgs_recursive():
            if None in sd.global_code:
                _write_global_code(sd.global_code[None], sd)
            if 'python' in sd.global_code:
                _write_global_code(sd.global_code['python'], sd)
            if backend in sd.global_code:
                _write_global_code(sd.global_code[backend], sd)

    def generate_header(self, sdfg: SDFG, global_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream):
        """ Generate the header of the frame-code. Code exists in a separate
            function for overriding purposes.

            :param sdfg: The input SDFG.
            :param global_stream: Stream to write to (global).
            :param callsite_stream: Stream to write to (at call site).
        """
        # Write frame code - header
        global_stream.write('# DaCe AUTO-GENERATED FILE. DO NOT MODIFY\n', sdfg)

        # Write header required by environments
        for env in self.environments:
            self.statestruct.extend(env.state_fields)

        self.generate_fileheader(sdfg, global_stream, 'frame')

    def generate_footer(self, sdfg: SDFG, global_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream):
        """ Generate the footer of the frame-code. Code exists in a separate
            function for overriding purposes.

            :param sdfg: The input SDFG.
            :param global_stream: Stream to write to (global).
            :param callsite_stream: Stream to write to (at call site).
        """
        # Python backend: function wrapping is handled in generate_code().
        pass

    def generate_external_memory_management(self, sdfg: SDFG, callsite_stream: PythonCodeIOStream):
        """
        External memory management is not yet supported in the Python backend.
        """
        # Collect external arrays to check if any exist
        for subsdfg, aname, arr in sdfg.arrays_recursive():
            if arr.lifetime == dtypes.AllocationLifetime.External:
                raise NotImplementedError(
                    'External memory management is not yet supported in the Python backend.')
        # No external arrays — nothing to do.
        pass

    def generate_state(self,
                       sdfg: SDFG,
                       cfg: ControlFlowRegion,
                       state: SDFGState,
                       global_stream: PythonCodeIOStream,
                       callsite_stream: PythonCodeIOStream,
                       generate_state_footer: bool = True):
        sid = state.block_id

        # Emit internal transient array allocation
        self.allocate_arrays_in_scope(sdfg, cfg, state, global_stream, callsite_stream)

        callsite_stream.write('\n', cfg=cfg, state_id=sid)

        #####################
        # Create dataflow graph for state's children.

        # DFG to code scheme: Only generate code for nodes whose all
        # dependencies have been executed (topological sort).
        # For different connected components, run them concurrently.

        components = dace.sdfg.concurrent_subgraphs(state)

        if len(components) <= 1:
            self._dispatcher.dispatch_subgraph(sdfg,
                                               cfg,
                                               state,
                                               sid,
                                               global_stream,
                                               callsite_stream,
                                               skip_entry_node=False)
        else:
            # Python backend: no OpenMP support, dispatch components sequentially
            for c in components:
                self._dispatcher.dispatch_subgraph(sdfg,
                                                   cfg,
                                                   c,
                                                   sid,
                                                   global_stream,
                                                   callsite_stream,
                                                   skip_entry_node=False)

        #####################
        # Write state footer

        if generate_state_footer:
            # Emit internal transient array deallocation
            self.deallocate_arrays_in_scope(sdfg, state.parent_graph, state, global_stream, callsite_stream)

    def generate_states(self, sdfg: SDFG, global_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> Set[SDFGState]:
        states_generated = set()

        opbar = progress.OptionalProgressBar(len(sdfg.states()), title=f'Generating code (SDFG {sdfg.cfg_id})')

        # Create closure + function for state dispatcher
        def dispatch_state(state: SDFGState) -> str:
            stream = PythonCodeIOStream()
            self._dispatcher.dispatch_state(state, global_stream, stream)
            opbar.next()
            states_generated.add(state)  # For sanity check
            return stream.getvalue()

        py_cflow.control_flow_region_to_code(sdfg, dispatch_state, self, sdfg.symbols, callsite_stream)

        opbar.done()

        return states_generated

    def _get_schedule(self, scope: Union[nodes.EntryNode, SDFGState, SDFG]) -> dtypes.ScheduleType:
        TOP_SCHEDULE = dtypes.ScheduleType.Sequential
        if scope is None:
            return TOP_SCHEDULE
        elif isinstance(scope, nodes.EntryNode):
            return scope.schedule
        elif isinstance(scope, (SDFGState, SDFG)):
            sdfg: SDFG = (scope if isinstance(scope, SDFG) else scope.parent)
            if sdfg.parent_nsdfg_node is None:
                return TOP_SCHEDULE

            # Go one SDFG up
            pstate = sdfg.parent
            pscope = pstate.entry_node(sdfg.parent_nsdfg_node)
            if pscope is not None:
                return self._get_schedule(pscope)
            return self._get_schedule(pstate)
        else:
            raise TypeError

    def _can_allocate(self, sdfg: SDFG, state: SDFGState, desc: data.Data, scope: Union[nodes.EntryNode, SDFGState,
                                                                                        SDFG]) -> bool:
        # TODO: Check if whatever this does is actually correct
        # for python
        schedule = self._get_schedule(scope)
        # if not dtypes.can_allocate(desc.storage, schedule):
        #     return False
        if dtypes.can_allocate(desc.storage, schedule):
            return True

        # Check for device-level memory recursively
        node = scope if isinstance(scope, nodes.EntryNode) else None
        cstate = scope if isinstance(scope, SDFGState) else state
        csdfg = scope if isinstance(scope, SDFG) else sdfg

        if desc.storage in dtypes.GPU_STORAGES:
            return sdscope.is_devicelevel_gpu(csdfg, cstate, node)

        return False

    def determine_allocation_lifetime(self, top_sdfg: SDFG):
        """
        Determines where (at which scope/state/SDFG) each data descriptor will be allocated/deallocated.

        :param top_sdfg: The top-level SDFG to determine for.
        """
        # TODO: I don't believe this is correct for the python backend
        # Python only has one way to scope things and that is with functions
        
        # Gather shared transients, free symbols, and first/last appearance
        shared_transients = {}
        fsyms = {}
        reachability = StateReachability().apply_pass(top_sdfg, {})
        access_instances: Dict[int, Dict[str, List[Tuple[SDFGState, nodes.AccessNode]]]] = {}
        for sdfg in top_sdfg.all_sdfgs_recursive():
            shared_transients[sdfg.cfg_id] = sdfg.shared_transients(check_toplevel=False, include_nested_data=True)
            fsyms[sdfg.cfg_id] = self.symbols_and_constants(sdfg)

            #############################################
            # Look for all states in which a scope-allocated array is used in
            instances: Dict[str, List[Tuple[SDFGState, nodes.AccessNode]]] = collections.defaultdict(list)
            array_names = sdfg.arrays.keys(
            )  #set(k for k, v in sdfg.arrays.items() if v.lifetime == dtypes.AllocationLifetime.Scope)
            # Iterate topologically to get state-order
            for state in cfg_analysis.blockorder_topological_sort(sdfg, ignore_nonstate_blocks=True):
                for node in state.data_nodes():
                    if node.data not in array_names:
                        continue
                    instances[node.data].append((state, node))

                # Look in the surrounding edges for usage
                edge_fsyms: Set[str] = set()
                for e in state.parent_graph.all_edges(state):
                    edge_fsyms |= e.data.free_symbols
                for edge_array in edge_fsyms & array_names:
                    instances[edge_array].append((state, nodes.AccessNode(edge_array)))
            #############################################

            access_instances[sdfg.cfg_id] = instances

        for sdfg, name, desc in top_sdfg.arrays_recursive(include_nested_data=True):
            # NOTE: Assuming here that all Structure members share transient/storage/lifetime properties.
            # TODO: Study what is needed in the DaCe stack to ensure this assumption is correct.
            top_desc = sdfg.arrays[name.split('.')[0]]
            top_transient = top_desc.transient
            top_storage = top_desc.storage
            top_lifetime = top_desc.lifetime
            if not top_transient:
                continue
            if name in sdfg.constants_prop:
                # Constants do not need to be allocated
                continue

            # NOTE: In the code below we infer where a transient should be
            # declared, allocated, and deallocated. The information is stored
            # in the `to_allocate` dictionary. The key of each entry is the
            # scope where one of the above actions must occur, while the value
            # is a tuple containing the following information:
            # 1. The SDFG object that containts the transient.
            # 2. The State id where the action should (approx.) take place.
            # 3. The Access Node id of the transient in the above State.
            # 4. True if declaration should take place, otherwise False.
            # 5. True if allocation should take place, otherwise False.
            # 6. True if deallocation should take place, otherwise False.

            first_state_instance, first_node_instance = access_instances[sdfg.cfg_id].get(name, [(None, None)])[0]
            last_state_instance, last_node_instance = access_instances[sdfg.cfg_id].get(name, [(None, None)])[-1]

            # Cases
            if top_lifetime in (dtypes.AllocationLifetime.Persistent, dtypes.AllocationLifetime.External):
                # Persistent memory is allocated in initialization code and
                # exists in the library state structure

                # If unused, skip
                if first_node_instance is None:
                    continue

                # TODO: as_arg can return c-style definitions that may break python code
                definition = desc.as_arg(name=f'__{sdfg.cfg_id}_{name}')

                if top_storage != dtypes.StorageType.CPU_ThreadLocal:  # If thread-local, skip struct entry
                    self.statestruct.append(definition)

                self.to_allocate[top_sdfg].append((sdfg, first_state_instance, first_node_instance, True, True, True))
                self.where_allocated[(sdfg, name)] = top_sdfg
                continue
            elif top_lifetime is dtypes.AllocationLifetime.Global:
                # Global memory is allocated in the beginning of the program
                # exists in the library state structure (to be passed along
                # to the right SDFG)

                # If unused, skip
                if first_node_instance is None:
                    continue
                # TODO: as_arg can return c-style definitions that may break python code
                definition = desc.as_arg(name=f'__{sdfg.cfg_id}_{name}')
                self.statestruct.append(definition)

                self.to_allocate[top_sdfg].append((sdfg, first_state_instance, first_node_instance, True, True, True))
                self.where_allocated[(sdfg, name)] = top_sdfg
                continue

            # The rest of the cases change the starting scope we attempt to
            # allocate from, since the descriptors may only be allocated higher
            # in the hierarchy (e.g., in the case of GPU global memory inside
            # a kernel).
            alloc_scope: Union[nodes.EntryNode, SDFGState, SDFG] = None
            alloc_state: SDFGState = None
            if (name in shared_transients[sdfg.cfg_id] or top_lifetime is dtypes.AllocationLifetime.SDFG):
                # SDFG descriptors are allocated in the beginning of their SDFG
                alloc_scope = sdfg
                if first_state_instance is not None:
                    alloc_state = first_state_instance
                # If unused, skip
                if first_node_instance is None:
                    continue
            elif top_lifetime == dtypes.AllocationLifetime.State:
                # State memory is either allocated in the beginning of the
                # containing state or the SDFG (if used in more than one state)
                curstate: SDFGState = None
                multistate = False
                for state in sdfg.states():
                    if any(n.data == name for n in state.data_nodes()):
                        if curstate is not None:
                            multistate = True
                            break
                        curstate = state
                if multistate:
                    alloc_scope = sdfg
                else:
                    alloc_scope = curstate
                    alloc_state = curstate
            elif top_lifetime == dtypes.AllocationLifetime.Scope:
                # Scope memory (default) is either allocated in the innermost
                # scope (e.g., Map, Consume) it is used in (i.e., greatest
                # common denominator), or in the SDFG if used in multiple states
                curscope: Union[nodes.EntryNode, SDFGState] = None
                curstate: SDFGState = None
                multistate = False

                # Does the array appear in inter-state edges or loop / conditional block conditions etc.?
                for isedge in sdfg.all_interstate_edges():
                    if name in self.free_symbols(isedge.data):
                        multistate = True
                for cfg in sdfg.all_control_flow_regions():
                    block_syms = cfg.used_symbols(all_symbols=True, with_contents=False)
                    if name in block_syms:
                        multistate = True

                for state in sdfg.states():
                    if multistate:
                        break
                    sdict = state.scope_dict()
                    for node in state.nodes():
                        if not isinstance(node, nodes.AccessNode):
                            continue
                        if node.root_data != name:
                            continue

                        # If already found in another state, set scope to SDFG
                        if curstate is not None and curstate != state:
                            multistate = True
                            break
                        curstate = state

                        # Current scope (or state object if top-level)
                        scope = sdict[node] or state
                        if curscope is None:
                            curscope = scope
                            continue
                        # States always win
                        if isinstance(scope, SDFGState):
                            curscope = scope
                            continue
                        # Lower/Higher/Disjoint scopes: find common denominator
                        if isinstance(curscope, SDFGState):
                            if scope in curscope.nodes():
                                continue
                        curscope = sdscope.common_parent_scope(sdict, scope, curscope)

                    if multistate:
                        break

                if multistate:
                    alloc_scope = sdfg
                else:
                    alloc_scope = curscope
                    alloc_state = curstate
            else:
                raise TypeError('Unrecognized allocation lifetime "%s"' % desc.lifetime)

            if alloc_scope is None:  # No allocation necessary
                continue

            # If descriptor cannot be allocated in this scope, traverse up the
            # scope tree until it is possible
            cursdfg = sdfg
            curstate = alloc_state
            curscope = alloc_scope
            while not self._can_allocate(cursdfg, curstate, desc, curscope):
                if curscope is None:
                    break
                if isinstance(curscope, nodes.EntryNode):
                    # Go one scope up
                    curscope = curstate.entry_node(curscope)
                    if curscope is None:
                        curscope = curstate
                elif isinstance(curscope, (SDFGState, SDFG)):
                    cursdfg: SDFG = (curscope if isinstance(curscope, SDFG) else curscope.parent)
                    # Go one SDFG up
                    if cursdfg.parent_nsdfg_node is None:
                        curscope = None
                        curstate = None
                        cursdfg = None
                    else:
                        curstate = cursdfg.parent
                        curscope = curstate.entry_node(cursdfg.parent_nsdfg_node)
                        cursdfg = cursdfg.parent_sdfg
                else:
                    raise TypeError

            if curscope is None:
                curscope = top_sdfg

            # Check if Array/View is dependent on non-free SDFG symbols
            # NOTE: Tuple is (SDFG, State, Node, declare, allocate, deallocate)
            fsymbols = fsyms[sdfg.cfg_id]
            if (not isinstance(curscope, nodes.EntryNode)
                    and utils.is_nonfree_sym_dependent(first_node_instance, desc, first_state_instance, fsymbols)):
                # Allocate in first State, deallocate in last State
                if first_state_instance != last_state_instance:
                    # If any state is not reachable from first state, find common denominators in the form of
                    # dominator and postdominator.
                    instances: List[Tuple[SDFGState, nodes.AccessNode]] = access_instances[sdfg.cfg_id][name]

                    # A view gets "allocated" everywhere it appears
                    if isinstance(desc, data.View):
                        for s, n in instances:
                            self.to_allocate[s].append((sdfg, s, n, False, True, False))
                            self.to_allocate[s].append((sdfg, s, n, False, False, True))
                        self.where_allocated[(sdfg, name)] = cursdfg
                        continue

                    if any(inst not in reachability[sdfg.cfg_id][first_state_instance] for inst in instances):
                        first_state_instance, last_state_instance = _get_dominator_and_postdominator(sdfg, instances)
                        # Declare in SDFG scope
                        # NOTE: Even if we declare the data at a common dominator, we keep the first and last node
                        # instances. This is especially needed for Views which require both the SDFGState and the
                        # AccessNode.
                        self.to_allocate[curscope].append((sdfg, None, nodes.AccessNode(name), True, False, False))
                    else:
                        self.to_allocate[curscope].append(
                            (sdfg, first_state_instance, first_node_instance, True, False, False))

                    curscope = first_state_instance
                    self.to_allocate[curscope].append(
                        (sdfg, first_state_instance, first_node_instance, False, True, False))
                    curscope = last_state_instance
                    self.to_allocate[curscope].append(
                        (sdfg, last_state_instance, last_node_instance, False, False, True))
                else:
                    curscope = first_state_instance
                    self.to_allocate[curscope].append(
                        (sdfg, first_state_instance, first_node_instance, True, True, True))
            else:
                self.to_allocate[curscope].append((sdfg, first_state_instance, first_node_instance, True, True, True))
            if isinstance(curscope, SDFG):
                self.where_allocated[(sdfg, name)] = curscope
            else:
                self.where_allocated[(sdfg, name)] = cursdfg

    def allocate_arrays_in_scope(self, sdfg: SDFG, cfg: ControlFlowRegion, scope: Union[nodes.EntryNode, SDFGState,
                                                                                        SDFG],
                                 function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream) -> None:
        """ Dispatches allocation of all arrays in the given scope. """
        # TODO: Check what we should actually do for python
        if len(self.to_allocate[scope]) == 0:
            return
        for tsdfg, state, node, declare, allocate, _ in self.to_allocate[scope]:
            if state is not None:
                state_id = state.block_id
            else:
                state_id = -1

            desc = node.desc(tsdfg)

            self._dispatcher.dispatch_allocate(tsdfg, cfg if state is None else state.parent_graph, state, state_id,
                                               node, desc, function_stream, callsite_stream, declare, allocate)
        for instr in self._dispatcher.instrumentation.values():
            if instr is not None:
                instr.on_allocation_end(sdfg, scope, callsite_stream)

    def deallocate_arrays_in_scope(self, sdfg: SDFG, cfg: ControlFlowRegion, scope: Union[nodes.EntryNode, SDFGState,
                                                                                          SDFG],
                                   function_stream: PythonCodeIOStream, callsite_stream: PythonCodeIOStream):
        """ Dispatches deallocation of all arrays in the given scope. """
        # TODO: Check what we should actually do for python
        if len(self.to_allocate[scope]) == 0:
            return
        for tsdfg, state, node, _, _, deallocate in self.to_allocate[scope]:
            if not deallocate:
                continue
            if state is not None:
                state_id = state.block_id
            else:
                state_id = -1

            desc = node.desc(tsdfg)

            self._dispatcher.dispatch_deallocate(tsdfg, state.parent_graph, state, state_id, node, desc,
                                                 function_stream, callsite_stream)

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
        # TODO: This is not yet fully correct for a python implementation
        # Also quite a bit of code was removed compared to C++ 
        # version, so we should check that all necessary steps 
        # are still present and correct for python.
        
        if len(cfg_id) == 0 and sdfg.cfg_id != 0:
            cfg_id = '_%d' % sdfg.cfg_id

        global_stream = PythonCodeIOStream()
        callsite_stream = PythonCodeIOStream()

        is_top_level = sdfg.parent is None

        # Analyze allocation lifetime of SDFG and all nested SDFGs
        if is_top_level:
            # TODO: Check if this is actually needed for python
            self.determine_allocation_lifetime(sdfg)

        # Generate code
        ###########################

        # Allocate outer-level transients
        # TODO: Check if this is correct for python
        self.allocate_arrays_in_scope(sdfg, sdfg, sdfg, global_stream, callsite_stream)

        # Define constants as top-level-allocated
        # TODO: Check if this is correct for python
        for cname, (ctype, _) in sdfg.constants_prop.items():
            if isinstance(ctype, data.Array):
                self.dispatcher.defined_vars.add(cname, disp.DefinedType.Pointer, ctype.dtype.ctype) # TODO: Pointer is almost definitely not correct for python
            else:
                self.dispatcher.defined_vars.add(cname, disp.DefinedType.Scalar, ctype.dtype.ctype)

        # Allocate inter-state variables
        global_symbols = copy.deepcopy(sdfg.symbols)
        global_symbols.update({aname: arr.dtype for aname, arr in sdfg.arrays.items()})
        interstate_symbols = {}
        for cfr in sdfg.all_control_flow_regions():
            if isinstance(cfr, LoopRegion) and cfr.loop_variable is not None and cfr.init_statement is not None:
                if not cfr.loop_variable in interstate_symbols:
                    if cfr.loop_variable in global_symbols:
                        interstate_symbols[cfr.loop_variable] = global_symbols[cfr.loop_variable]
                    else:
                        l_end = loop_analysis.get_loop_end(cfr)
                        l_start = loop_analysis.get_init_assignment(cfr)
                        l_step = loop_analysis.get_loop_stride(cfr)
                        sym_type = dtypes.result_type_of(infer_expr_type(l_start, global_symbols),
                                                         infer_expr_type(l_step, global_symbols),
                                                         infer_expr_type(l_end, global_symbols))
                        interstate_symbols[cfr.loop_variable] = sym_type
                if not cfr.loop_variable in global_symbols:
                    global_symbols[cfr.loop_variable] = interstate_symbols[cfr.loop_variable]

            for e in cfr.dfs_edges(cfr.start_block):
                symbols = e.data.new_symbols(sdfg, global_symbols)
                # Inferred symbols only take precedence if global symbol not defined or None
                symbols = {
                    k: v if (k not in global_symbols or global_symbols[k] is None) else global_symbols[k]
                    for k, v in symbols.items()
                }
                interstate_symbols.update(symbols)
                global_symbols.update(symbols)

        # In Python, variables don't need explicit declaration — they are
        # created on first assignment.  We still record them so that
        # ``defined_vars`` stays in sync with the C++ backend expectations.
        for isvarName, isvarType in interstate_symbols.items():
            if isvarType is None:
                raise TypeError(f'Type inference failed for symbol {isvarName}')
            if not is_top_level and isvarName in sdfg.parent_nsdfg_node.symbol_mapping:
                continue
            # No emit needed: Python variables are created on assignment.
            # TODO: Check that this is true for all cases, and that no "undefined variable" errors can occur.

        #######################################################################
        # Generate actual program body

        states_generated = self.generate_states(sdfg, global_stream, callsite_stream)

        #######################################################################

        # Sanity check
        if len(states_generated) != len(sdfg.states()):
            raise RuntimeError(
                "Not all states were generated in SDFG {}!"
                "\n  Generated: {}\n  Missing: {}".format(sdfg.label, [s.label for s in states_generated],
                                                          [s.label for s in (set(sdfg.states()) - states_generated)]))

        # Deallocate transients
        self.deallocate_arrays_in_scope(sdfg, sdfg, sdfg, global_stream, callsite_stream)

        # Now that we have all the information about dependencies, generate
        # header and footer
        if is_top_level:
            # Get all environments used in the generated code, including
            # dependent environments
            self.environments = dace.library.get_environments_and_dependencies(self._dispatcher.used_environments)

            header_global_stream = PythonCodeIOStream()
            self.generate_header(sdfg, header_global_stream, PythonCodeIOStream())

            self.generate_footer(sdfg, PythonCodeIOStream(), PythonCodeIOStream())
            self.generate_external_memory_management(sdfg, PythonCodeIOStream())

            # Merge global streams
            header_global_stream.write(global_stream.getvalue())
            generated_header = header_global_stream.getvalue()

            # Build Python function wrapper
            params = ', '.join(self.arglist.keys())
            body = callsite_stream.getvalue().strip()

            func_code = PythonCodeIOStream()
            func_code.write(f'def {sdfg.name}({params}):\n', cfg=sdfg)
            if body:
                with func_code.indented():
                    func_code.write(body)
            else:
                with func_code.indented():
                    func_code.write('pass\n', cfg=sdfg)

            generated_code = func_code.getvalue()
        else:
            generated_header = global_stream.getvalue()
            generated_code = callsite_stream.getvalue()

        # Return the generated global and local code strings
        return (generated_header, generated_code, self._dispatcher.used_targets, self._dispatcher.used_environments)


def _get_dominator_and_postdominator(sdfg: SDFG, accesses: List[Tuple[SDFGState, nodes.AccessNode]]):
    """
    Gets the closest common dominator and post-dominator for a list of states.
    Used for determining allocation of data used in branched states.
    """
    # TODO: Find out what this does and if it's correct for python
    alldoms: Dict[ControlFlowBlock, Set[ControlFlowBlock]] = collections.defaultdict(lambda: set())
    allpostdoms: Dict[ControlFlowBlock, Set[ControlFlowBlock]] = collections.defaultdict(lambda: set())
    idom: Dict[ControlFlowRegion, Dict[ControlFlowBlock, ControlFlowBlock]] = {}
    ipostdom: Dict[ControlFlowRegion, Dict[ControlFlowBlock, ControlFlowBlock]] = {}
    utils.get_control_flow_block_dominators(sdfg, idom, alldoms, ipostdom, allpostdoms)

    states = [a for a, _ in accesses]
    data_name = accesses[0][1].data

    # All dominators and postdominators include the states themselves
    for state in states:
        alldoms[state].add(state)
        allpostdoms[state].add(state)

    start_state = states[0]
    while any(start_state not in alldoms[n] for n in states):
        if idom[start_state] is start_state:
            raise NotImplementedError(f'Could not find an appropriate dominator for allocation of "{data_name}"')
        start_state = idom[start_state]

    end_state = states[-1]
    while any(end_state not in allpostdoms[n] for n in states):
        if ipostdom[end_state] is end_state:
            raise NotImplementedError(f'Could not find an appropriate post-dominator for deallocation of "{data_name}"')
        end_state = ipostdom[end_state]

    # TODO(later): If any of the symbols were not yet defined, or have changed afterwards, fail
    # raise NotImplementedError

    return start_state, end_state
