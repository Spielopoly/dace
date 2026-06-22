# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
General purpose Einstein sum (einsum) library node.

Specialization expansions of this node convert it to fast BLAS operations (e.g., matrix multiplications) if possible.
"""

from copy import deepcopy

import dace
from dace import SDFG, SDFGState, dtypes, library, nodes, properties
from dace import transformation as xf
from dace.frontend.common import einsum
from dace.symbolic import equal_valued, symstr


# Define the library node itself
@library.node
class Einsum(nodes.LibraryNode):
    # Set the default expansion of the node to 'specialize' (registered below)
    implementations = {}
    default_implementation = 'specialize'

    # Configurable properties of the einsum node
    einsum_str = properties.Property(dtype=str,
                                     default='',
                                     desc='The Einstein notation string that describes this einsum')

    alpha = properties.SymbolicProperty(desc='The coefficient to multiply the inputs with', default=1.0)
    beta = properties.SymbolicProperty(desc='The coefficient to multiply the output with when added to the product',
                                       default=0.0)


# Define the expansion, which specializes the einsum by lowering it to either a BLAS operation or a direct contraction
@library.register_expansion(Einsum, 'specialize')
class SpecializeEinsum(xf.ExpandTransformation):
    # Define environments necessary for this expansion (optional, can be an empty list)
    environments = []

    # The following method returns the SDFG that results from expanding the library node.
    # Upon expansion, DaCe will insert the returned SDFG into the graph as a nested SDFG node (which can be inlined).
    @staticmethod
    def expansion(node: Einsum, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        # Make an SDFG for the expansion
        sdfg = SDFG('einsum')
        state = sdfg.add_state()

        # Add the given arrays (as given by memlets) to the expansion SDFG
        inputs = []
        output = None
        for e in parent_state.in_edges(node):
            inputs.append(e.dst_conn)
            desc = parent_sdfg.arrays[e.data.data]
            insubset = deepcopy(e.data.src_subset)
            isqdim = insubset.squeeze()
            sdfg.add_array(e.dst_conn,
                           insubset.size(),
                           desc.dtype,
                           strides=[s for i, s in enumerate(desc.strides) if i in isqdim],
                           storage=desc.storage)

        for e in parent_state.out_edges(node):
            output = e.src_conn
            desc = parent_sdfg.arrays[e.data.data]
            outsubset = deepcopy(e.data.dst_subset)
            osqdim = outsubset.squeeze()
            sdfg.add_array(output,
                           outsubset.size(),
                           desc.dtype,
                           strides=[s for i, s in enumerate(desc.strides) if i in osqdim],
                           storage=desc.storage)
        #######################################

        # Fill SDFG with einsum contents
        einsum.create_einsum_sdfg(sdfg,
                                  state,
                                  node.einsum_str,
                                  *sorted(inputs),
                                  output=output,
                                  output_name=output,
                                  alpha=node.alpha,
                                  beta=node.beta)
        return sdfg


@library.register_expansion(Einsum, 'CuPy')
class ExpandEinsumCuPy(xf.ExpandTransformation):
    """CuPy-based GPU einsum using ``cupy.einsum``.

    Produces a nested SDFG with a Python-language tasklet calling
    ``cupy.einsum`` with the node's einsum string.
    """

    environments = []

    @staticmethod
    def expansion(node: Einsum, parent_state: SDFGState,
                  parent_sdfg: SDFG) -> SDFG:
        sdfg = SDFG('einsum_cupy')
        state = sdfg.add_state()

        # Add arrays from parent (mirrors SpecializeEinsum).
        inputs = []
        output = None
        for e in parent_state.in_edges(node):
            inputs.append(e.dst_conn)
            desc = parent_sdfg.arrays[e.data.data]
            insubset = deepcopy(e.data.src_subset)
            isqdim = insubset.squeeze()
            sdfg.add_array(
                e.dst_conn,
                insubset.size(),
                desc.dtype,
                strides=[s for i, s in enumerate(desc.strides)
                         if i in isqdim],
                storage=desc.storage)

        for e in parent_state.out_edges(node):
            output = e.src_conn
            desc = parent_sdfg.arrays[e.data.data]
            outsubset = deepcopy(e.data.dst_subset)
            osqdim = outsubset.squeeze()
            sdfg.add_array(
                output,
                outsubset.size(),
                desc.dtype,
                strides=[s for i, s in enumerate(desc.strides)
                         if i in osqdim],
                storage=desc.storage)

        # Build tasklet code.
        sorted_inputs = sorted(inputs)
        input_args = ', '.join(
            f'cupy.asarray(__{inp})' for inp in sorted_inputs)

        code_lines = ['import cupy']

        alpha = node.alpha
        beta = node.beta
        einsum_str = node.einsum_str

        code_lines.append(
            f'__result = cupy.einsum("{einsum_str}", {input_args})')

        # Alpha scaling.
        if not equal_valued(1, alpha):
            code_lines.append(f'__result = {symstr(alpha)} * __result')

        # Beta scaling (add to existing output).
        if not equal_valued(0, beta):
            code_lines.append(
                f'__result = __result + {symstr(beta)}'
                f' * cupy.asarray(__{output})')

        code_lines.append(f'__{output}_out = cupy.asnumpy(__result)')

        code = '\n'.join(code_lines)

        # Connector setup.
        in_connectors = {f'__{inp}': None for inp in sorted_inputs}
        out_connectors = {f'__{output}_out': None}
        if not equal_valued(0, beta):
            in_connectors[f'__{output}'] = None

        tasklet = nodes.Tasklet(
            'einsum_cupy',
            in_connectors,
            out_connectors,
            code,
            language=dtypes.Language.Python,
        )
        state.add_node(tasklet)

        # Wire input edges.
        for inp in sorted_inputs:
            r = state.add_read(inp)
            state.add_edge(
                r, None, tasklet, f'__{inp}',
                dace.Memlet.from_array(inp, sdfg.arrays[inp]))

        # Wire output edge.
        w = state.add_write(output)
        state.add_edge(
            tasklet, f'__{output}_out', w, None,
            dace.Memlet.from_array(output, sdfg.arrays[output]))

        # Beta != 0 requires reading the current output.
        if not equal_valued(0, beta):
            r_out = state.add_read(output)
            state.add_edge(
                r_out, None, tasklet, f'__{output}',
                dace.Memlet.from_array(output, sdfg.arrays[output]))

        return sdfg
