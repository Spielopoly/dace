"""
IfElseMapToTileWhere – transform tiled maps with if-else nested SDFGs to
tile-level unconditional branches + where-select.

Pattern (BEFORE, after MapTiling)::

    OuterMapEntry → InnerMapEntry → NestedSDFG(if-else) → InnerMapExit → OuterMapExit

Result (AFTER)::

    OuterMapEntry → [input tile transients]
                  → TileIfElseOpLibraryNode
                  → [output tile]
                  → OuterMapExit

Both branches are executed unconditionally on full tiles inside the compound
node's expansion SDFG.  A where-select picks the correct result per element
based on the condition mask.
"""


import ast as _ast
import copy
from typing import Dict, List, Optional, Tuple

import dace
import sympy as sp
from dace import Memlet
from dace.sdfg import SDFG, SDFGState, nodes, utils as sdutil
from dace.sdfg.state import ConditionalBlock
from dace.transformation import transformation as xf

from dace.libraries.cutile.nodes.if_else_op import TileIfElseOpLibraryNode
from dace.libraries.cutile.op_registry import (
    TaskletLibraryNodeMatch,
    match_tasklet_to_tile_library_node,
    MaskType,
)
from dace.libraries.cutile.transformations.utils import (
    tile_subset_from_shape,
    create_tile_transient,
    is_canonical_inner_map,
    primary_memlet_subset,
    with_primary_subset,
    memlet_with_primary_subset,
)
from sympy.parsing.sympy_parser import parse_expr


# ── Helpers ──────────────────────────────────────────────────────────

def _to_sympy_condition(
    cond_expr: str,
    nsdfg_arrays: set,
) -> Optional[sp.Basic]:
    """Convert a condition expression string to a SymPy expression.

    All free symbols in the resulting expression must map to NSDFG array
    names. Returns ``None`` if parsing fails or unknown symbols appear.

    Args:
        cond_expr: A condition string from an interstate edge assignment (e.g.
            ``"A > 0"``, ``"mask"``).
        nsdfg_arrays: Set of array names declared in the nested SDFG.  Every
            free symbol in the parsed expression must appear in this set.

    Returns:
        A parsed :class:`sympy.Basic` expression, or ``None`` if the
        expression could not be parsed or references unknown symbols.
    """
    parsed_expression = dace.symbolic.pystr_to_symbolic(cond_expr, simplify=False)

    if not isinstance(parsed_expression, sp.Basic):
        return None

    free_symbol_names = {
        str(free_symbol) for free_symbol in parsed_expression.free_symbols
    }
    if not free_symbol_names.issubset(nsdfg_arrays):
        return None

    return parsed_expression


def _build_sympy_expr(
    op: str, left: sp.Basic, right: Optional[sp.Basic] = None,
) -> sp.Basic:
    """Build a SymPy expression from a classified operation.

    Args:
        op: Operation string (e.g. ``"+"``, ``"abs"``, ``"sin"``).
        left: Left / first operand (Symbol or Number).
        right: Right / second operand.  ``None`` for unary operations.

    Returns:
        A SymPy expression combining the operands with the operation.

    Raises:
        ValueError: If *left* or *op* is ``None``.
    """
    if left is None:
        raise ValueError("Left operand cannot be None")
    if op is None:
        raise ValueError("Operator cannot be None")
    if right is None:
        return parse_expr(f"{op}({left})")
    return parse_expr(f"({left}) {op} ({right})")


def _extract_base_array_name(expr_str: str) -> Optional[str]:
    """Extract the base array name from an array subscript like ``B[i, j]``."""
    try:
        tree = _ast.parse(expr_str.strip(), mode='eval')
        if isinstance(tree.body, _ast.Subscript) and isinstance(tree.body.value, _ast.Name):
            return tree.body.value.id
    except Exception:
        pass
    return None


def _tasklet_code_to_sympy(code_str: str, input_mapping: Dict[str, sp.Basic]) -> Optional[sp.Basic]:
    """Convert a simple assignment tasklet to a SymPy expression.

    Handles patterns like ``__out = __in1 * 3`` by replacing connector
    names with the SymPy expressions from *input_mapping*.  Also handles
    ``dace.float64(…)`` and similar typed-constant wrappers.
    """
    try:
        tree = _ast.parse(code_str.strip())
        if len(tree.body) != 1 or not isinstance(tree.body[0], _ast.Assign):
            return None
        rhs_str = _ast.unparse(tree.body[0].value)
        for conn, resolved in input_mapping.items():
            rhs_str = rhs_str.replace(conn, f'({resolved})')
        # Strip dace type wrappers: dace.float64(3) → 3
        import re
        rhs_str = re.sub(r'dace\.\w+\(([^)]+)\)', r'\1', rhs_str)
        return dace.symbolic.pystr_to_symbolic(rhs_str, simplify=False)
    except Exception:
        return None


def _resolve_symbol(
    inner_sdfg: SDFG,
    sym_name: str,
    ise_assignments: Dict[str, str],
    wired_inputs: set,
    visited: frozenset,
) -> Optional[sp.Basic]:
    """Resolve *sym_name* to a SymPy expression over wired input arrays.

    Handles wired inputs (identity), interstate-edge symbol assignments
    (e.g. ``B[i, j]`` → ``B``), and single-tasklet transient arrays.
    """
    if sym_name in visited:
        return None
    visited = visited | {sym_name}

    # Already a wired input → identity
    if sym_name in wired_inputs:
        return sp.Symbol(sym_name)

    # Interstate-edge symbol → extract base array name
    if sym_name in ise_assignments:
        base = _extract_base_array_name(ise_assignments[sym_name])
        if base is not None:
            return _resolve_symbol(inner_sdfg, base, ise_assignments, wired_inputs, visited)
        return None

    # Transient array → trace through its writing tasklet or copy edge
    if sym_name in inner_sdfg.arrays and inner_sdfg.arrays[sym_name].transient:
        for state in inner_sdfg.all_states():
            for nd in state.nodes():
                if not isinstance(nd, nodes.AccessNode) or nd.data != sym_name:
                    continue
                for in_edge in state.in_edges(nd):
                    if isinstance(in_edge.src, nodes.Tasklet):
                        tasklet = in_edge.src
                        inp_map: Dict[str, sp.Basic] = {}
                        for t_edge in state.in_edges(tasklet):
                            if isinstance(t_edge.src, nodes.AccessNode) and t_edge.dst_conn:
                                r = _resolve_symbol(inner_sdfg, t_edge.src.data,
                                                    ise_assignments, wired_inputs, visited)
                                if r is None:
                                    return None
                                inp_map[t_edge.dst_conn] = r
                        result = _tasklet_code_to_sympy(tasklet.code.as_string, inp_map)
                        if result is not None:
                            return result
                    elif isinstance(in_edge.src, nodes.AccessNode):
                        # Direct copy: trace through to the source
                        return _resolve_symbol(inner_sdfg, in_edge.src.data,
                                               ise_assignments, wired_inputs, visited)
        return None

    return None


def _resolve_condition_to_wired_inputs(
    inner_sdfg: SDFG,
    cond_expr: str,
    wired_inputs: set,
) -> Optional[sp.Basic]:
    """Resolve a condition expression to reference only wired input arrays.

    Traces interstate-edge symbol assignments and transient array
    computations backward through the data flow so that the returned
    SymPy expression contains only symbols that are wired NSDFG inputs.

    Returns ``None`` if resolution fails.
    """
    ise_assignments: Dict[str, str] = {}
    for e in inner_sdfg.all_interstate_edges():
        ise_assignments.update(e.data.assignments)

    parsed = dace.symbolic.pystr_to_symbolic(cond_expr, simplify=False)
    if not isinstance(parsed, sp.Basic):
        return None

    for sym in list(parsed.free_symbols):
        sym_name = str(sym)
        resolved = _resolve_symbol(inner_sdfg, sym_name, ise_assignments,
                                   wired_inputs, frozenset())
        if resolved is None:
            return None
        if resolved is not sym:
            parsed = parsed.subs(sym, resolved)

    # Verify all remaining symbols are wired inputs
    remaining = {str(s) for s in parsed.free_symbols}
    if not remaining.issubset(wired_inputs):
        return None
    return parsed


def _extend_nsdfg_to_outer_with_transients(
    inner_sdfg: SDFG,
    nsdfg_to_outer: Dict[str, str],
) -> Dict[str, str]:
    """Extend *nsdfg_to_outer* to include internal transients that can be
    traced back to wired inputs through simple data movement.

    Walks all dataflow edges in every state of *inner_sdfg* and propagates
    the mapping whenever an unknown transient is written from a known array
    via a direct ``AccessNode → AccessNode`` edge (copy / slice / index).
    """
    resolved: Dict[str, str] = dict(nsdfg_to_outer)
    all_states = list(inner_sdfg.all_states())
    changed = True
    while changed:
        changed = False
        for state in all_states:
            for edge in state.edges():
                if (isinstance(edge.src, nodes.AccessNode)
                        and isinstance(edge.dst, nodes.AccessNode)):
                    if edge.src.data in resolved and edge.dst.data not in resolved:
                        resolved[edge.dst.data] = resolved[edge.src.data]
                        changed = True
    return resolved


class IfElseMapToTileWhere(xf.SingleStateTransformation):
    """
    Replace a tiled map whose inner scope is a NestedSDFG with an if-else
    ConditionalBlock by a single ``TileIfElseOpLibraryNode``.

    The compound node's expansion executes both branches unconditionally
    on full tiles and uses a where-select to pick the correct result per
    element based on the condition mask.
    """

    outer_map_entry = xf.PatternNode(nodes.MapEntry)
    inner_map_entry = xf.PatternNode(nodes.MapEntry)
    nsdfg_node = xf.PatternNode(nodes.NestedSDFG)
    inner_map_exit = xf.PatternNode(nodes.MapExit)
    outer_map_exit = xf.PatternNode(nodes.MapExit)

    @classmethod
    def expressions(cls):  # type: ignore[override]
        """Return the pattern graph that triggers this transformation.

        DaCe matches this path-shaped subgraph pattern before invoking
        :meth:`can_be_applied` for the stricter semantic checks.

        Returns:
            A list containing a single path graph:
            ``outer_map_entry → inner_map_entry → nsdfg_node``
            ``→ inner_map_exit → outer_map_exit``.
        """
        return [sdutil.node_path_graph(
            cls.outer_map_entry,
            cls.inner_map_entry,
            cls.nsdfg_node,
            cls.inner_map_exit,
            cls.outer_map_exit,
        )]

    # ── applicability ────────────────────────────────────────────────

    def can_be_applied(self, graph: SDFGState, expr_index: int,
                       sdfg: SDFG, permissive: bool = False) -> bool:
        """Check whether this transformation is applicable to the matched subgraph.

        Validates the following conditions:

        * The inner scope contains only the :class:`~dace.sdfg.nodes.NestedSDFG`.
        * The outer scope contains only the inner map pair and the NestedSDFG.
        * The inner map is canonical (0-based, unit stride).
        * The NestedSDFG encodes an if-else ConditionalBlock pattern.
        * The condition expression can be parsed to a SymPy expression
          referencing only wired NSDFG input connectors.
        * Each branch contains exactly one supported element-wise tasklet.
        * Both branches write to the same output connector.

        Args:
            graph: The SDFG state containing the matched subgraph.
            expr_index: Index of the matched expression (always 0 here).
            sdfg: The top-level SDFG.
            permissive: Unused; present for API compatibility.

        Returns:
            ``True`` if all conditions are satisfied and the transformation
            can be applied.
        """
        outer_entry: nodes.MapEntry = self.outer_map_entry
        inner_entry: nodes.MapEntry = self.inner_map_entry
        nsdfg: nodes.NestedSDFG = self.nsdfg_node
        inner_exit: nodes.MapExit = self.inner_map_exit
        outer_exit: nodes.MapExit = self.outer_map_exit

        # Inner scope must contain only the NestedSDFG
        inner_scope = graph.scope_subgraph(inner_entry,
                                           include_entry=False,
                                           include_exit=False)
        if set(inner_scope.nodes()) != {nsdfg}:
            return False

        # Outer scope must contain only inner map + NestedSDFG
        outer_scope = graph.scope_subgraph(outer_entry,
                                           include_entry=False,
                                           include_exit=False)
        if set(outer_scope.nodes()) != {inner_entry, inner_exit, nsdfg}:
            return False

        # Inner map must be canonical (0-based, unit stride)
        if not is_canonical_inner_map(inner_entry.map):
            return False

        # Analyze the NestedSDFG for the if-else pattern
        analysis_result = self._analyze_nsdfg(nsdfg.sdfg)
        if analysis_result is None:
            return False

        _, cond_expr, true_branch_state, false_branch_state = analysis_result

        # Each condition symbol must be both declared and wired from outer scope.
        wired_inputs = {
            input_edge.dst_conn
            for input_edge in graph.in_edges(nsdfg)
            if input_edge.dst_conn is not None and input_edge.data.data is not None
        }

        nsdfg_arrays = set(nsdfg.sdfg.arrays.keys())
        condition_expression = _to_sympy_condition(cond_expr, nsdfg_arrays)
        if condition_expression is not None:
            # Direct condition: all symbols must be wired input arrays
            cond_syms = [str(s) for s in condition_expression.free_symbols]
            if any(s not in nsdfg.in_connectors for s in cond_syms):
                condition_expression = None
            elif any(s not in wired_inputs for s in cond_syms):
                condition_expression = None

        if condition_expression is None:
            # Fallback: resolve intermediates back to wired inputs
            condition_expression = _resolve_condition_to_wired_inputs(
                nsdfg.sdfg, cond_expr, wired_inputs)
            if condition_expression is None:
                return False

        # Each branch must have exactly one tasklet with one output,
        # OR be a direct copy (0 tasklets, AccessNode chain).
        for branch_state in (true_branch_state, false_branch_state):
            tasklets = [node for node in branch_state.nodes()
                        if isinstance(node, nodes.Tasklet)]
            if len(tasklets) == 0:
                # Copy branch: check for AccessNode → AccessNode chain
                if self._get_copy_branch_input(branch_state) is None:
                    return False
            elif len(tasklets) == 1:
                tasklet_node = tasklets[0]
                if len(tasklet_node.out_connectors) != 1:
                    return False
                tasklet_match = match_tasklet_to_tile_library_node(
                    branch_state, tasklet_node, MaskType.UNMASKED,
                    promote_scalars=True)
                if tasklet_match is None:
                    return False
            else:
                # Multi-tasklet chain (e.g., from SplitTasklets): check that
                # exactly one tasklet connects to the output and the chain
                # can be resolved to a SymPy expression over wired inputs.
                output_tasklet = self._find_output_tasklet(branch_state)
                if output_tasklet is None:
                    return False
                chain_expr = self._resolve_chain_expression(
                    branch_state, output_tasklet, nsdfg.sdfg, wired_inputs)
                if chain_expr is None:
                    return False

        # Both branches must write to the same output connector
        true_branch_output = self._get_branch_output_array(true_branch_state)
        false_branch_output = self._get_branch_output_array(false_branch_state)
        if true_branch_output is None or false_branch_output is None:
            return False
        if true_branch_output != false_branch_output:
            return False

        # The output must be an NSDFG output connector
        if true_branch_output not in nsdfg.out_connectors:
            return False

        # All branch operand arrays must have outer NSDFG connector mappings.
        # Internal transients (e.g. frontend-generated intermediates) lack
        # outer connectors and would cause a ValueError in apply().
        nsdfg_to_outer: Dict[str, str] = {}
        for e in graph.in_edges(nsdfg):
            if e.dst_conn and e.data.data:
                nsdfg_to_outer[e.dst_conn] = e.data.data
        for e in graph.out_edges(nsdfg):
            if e.src_conn and e.data.data:
                nsdfg_to_outer[e.src_conn] = e.data.data

        # Collect all NSDFG array names needed by condition + branches.
        needed_nsdfg_names: set = set()
        # Condition symbols
        cond_syms = {str(s) for s in condition_expression.free_symbols}
        needed_nsdfg_names.update(cond_syms)
        # Branch operand arrays
        for branch_state in (true_branch_state, false_branch_state):
            tasklets = [n for n in branch_state.nodes()
                        if isinstance(n, nodes.Tasklet)]
            if len(tasklets) == 0:
                copy_input = self._get_copy_branch_input(branch_state)
                if copy_input is not None:
                    needed_nsdfg_names.add(copy_input)
            elif len(tasklets) == 1:
                tasklet_node = tasklets[0]
                tasklet_match = match_tasklet_to_tile_library_node(
                    branch_state, tasklet_node, MaskType.UNMASKED,
                    promote_scalars=True)
                if tasklet_match is not None:
                    tc = tasklet_match.tasklet_classification
                    input_mapping = self._get_tasklet_input_mapping(
                        branch_state, tasklet_node)
                    if tc.rhs1 is not None and tc.constant1 is None:
                        arr = input_mapping.get(tc.rhs1)
                        if arr is not None:
                            needed_nsdfg_names.add(arr)
                    if tc.rhs2 is not None and tc.constant2 is None:
                        arr = input_mapping.get(tc.rhs2)
                        if arr is not None:
                            needed_nsdfg_names.add(arr)
            else:
                # Multi-tasklet chain: collect needed arrays from resolved expr
                output_tasklet = self._find_output_tasklet(branch_state)
                if output_tasklet is not None:
                    chain_expr = self._resolve_chain_expression(
                        branch_state, output_tasklet, nsdfg.sdfg, wired_inputs)
                    if chain_expr is not None:
                        for sym in chain_expr.free_symbols:
                            needed_nsdfg_names.add(str(sym))

        for name in needed_nsdfg_names:
            if name not in nsdfg_to_outer:
                return False

        return True

    # ── NestedSDFG analysis ──────────────────────────────────────────

    @staticmethod
    def _analyze_nsdfg(inner_sdfg: SDFG) \
            -> Optional[Tuple[ConditionalBlock, str, SDFGState, SDFGState]]:
        """Analyse a NestedSDFG for the if-else ConditionalBlock pattern.

        Expected CFG structure::

            empty_state --(assigns condition symbol)--> ConditionalBlock

        The ConditionalBlock must have exactly 2 branches: one with a
        condition and one with ``None`` (else).

        Args:
            inner_sdfg: The SDFG to analyse (from a NestedSDFG node).

        Returns:
            A 4-tuple ``(cond_block, cond_expr, true_state, false_state)``
            if the pattern matches, or ``None`` if it does not.
        """
        # Find ConditionalBlocks
        cond_blocks = [n for n in inner_sdfg.nodes()
                       if isinstance(n, ConditionalBlock)]
        if len(cond_blocks) != 1:
            return None
        cond_block = cond_blocks[0]

        if len(cond_block.branches) != 2:
            return None

        # Identify the condition expression from an incoming interstate edge
        incoming_interstate_edges = inner_sdfg.in_edges(cond_block)
        cond_expr = None
        cond_symbol = None

        # Determine which branch is "if" (has condition) and which is "else"
        (first_condition, first_body), (second_condition, second_body) = cond_block.branches
        if first_condition is not None and second_condition is None:
            cond_symbol = first_condition.as_string.strip()
            true_body, false_body = first_body, second_body
        elif first_condition is None and second_condition is not None:
            cond_symbol = second_condition.as_string.strip()
            true_body, false_body = second_body, first_body
        else:
            return None

        # Find the condition assignment in interstate edges
        for interstate_edge in incoming_interstate_edges:
            if cond_symbol in interstate_edge.data.assignments:
                cond_expr = interstate_edge.data.assignments[cond_symbol]
                break

        if cond_expr is None:
            return None

        # Each body must have exactly one state
        true_states = list(true_body.all_states())
        false_states = list(false_body.all_states())
        if len(true_states) != 1 or len(false_states) != 1:
            return None

        return cond_block, cond_expr, true_states[0], false_states[0]

    @staticmethod
    def _get_branch_output_array(state: SDFGState) -> Optional[str]:
        """Get the data name written by the branch.

        For tasklet branches, follows ``Tasklet → AccessNode → … → AccessNode``
        hops.  For copy branches (no tasklet), finds the sink AccessNode.

        Returns:
            The ``data`` name of the final access node, or ``None``.
        """
        tasklets = [node for node in state.nodes() if isinstance(node, nodes.Tasklet)]
        if len(tasklets) == 1:
            current = tasklets[0]
            visited: set = set()
            while True:
                out_edges = state.out_edges(current)
                access_successors = [e.dst for e in out_edges if isinstance(e.dst, nodes.AccessNode) and e.dst not in visited]
                if not access_successors:
                    break
                visited.add(current)
                current = access_successors[0]
            if isinstance(current, nodes.AccessNode):
                return current.data
            return None
        elif len(tasklets) == 0:
            # Copy branch: find the sink AccessNode
            access_nodes = [n for n in state.nodes()
                            if isinstance(n, nodes.AccessNode)]
            sinks = [n for n in access_nodes if state.out_degree(n) == 0]
            if len(sinks) == 1:
                return sinks[0].data
            return None
        else:
            # Multi-tasklet chain: find the sink AccessNode
            sinks = [n for n in state.nodes()
                     if isinstance(n, nodes.AccessNode) and state.out_degree(n) == 0]
            if len(sinks) == 1:
                return sinks[0].data
            return None

    @staticmethod
    def _get_copy_branch_input(state: SDFGState) -> Optional[str]:
        """Get the source array name for a copy-only branch (no tasklets).

        A copy branch has an ``AccessNode → … → AccessNode`` chain with
        no tasklets.  Returns the first (source) AccessNode's data name,
        or ``None`` if the pattern is not matched.
        """
        sources = state.source_nodes()
        if not sources:
            return None
        for src in sources:
            if isinstance(src, nodes.AccessNode):
                return src.data
        return None

    @staticmethod
    def _get_tasklet_input_mapping(state: SDFGState,
                                   tasklet: nodes.Tasklet) -> Dict[str, str]:
        """Map tasklet input connector names to NSDFG array names.

        Args:
            state: The :class:`~dace.sdfg.SDFGState` containing *tasklet*.
            tasklet: The :class:`~dace.sdfg.nodes.Tasklet` whose incoming
                edges are inspected.

        Returns:
            A dict ``{tasklet_connector_name: nsdfg_array_name}`` for every
            wired input connector.
        """
        input_mapping = {}
        for input_edge in state.in_edges(tasklet):
            if isinstance(input_edge.src, nodes.AccessNode) and input_edge.dst_conn is not None:
                input_mapping[input_edge.dst_conn] = input_edge.src.data
            elif input_edge.dst_conn is not None and input_edge.data.data is not None:
                input_mapping[input_edge.dst_conn] = input_edge.data.data
        return input_mapping

    @staticmethod
    def _find_output_tasklet(state: SDFGState) -> Optional[nodes.Tasklet]:
        """Find the tasklet that writes to the final (sink) AccessNode."""
        sinks = [n for n in state.nodes()
                 if isinstance(n, nodes.AccessNode) and state.out_degree(n) == 0]
        if len(sinks) != 1:
            return None
        sink = sinks[0]
        # Check direct edges to sink
        for edge in state.in_edges(sink):
            if isinstance(edge.src, nodes.Tasklet):
                return edge.src
        # Check one level of indirection (Tasklet → AccessNode → sink)
        for edge in state.in_edges(sink):
            if isinstance(edge.src, nodes.AccessNode):
                for inner_edge in state.in_edges(edge.src):
                    if isinstance(inner_edge.src, nodes.Tasklet):
                        return inner_edge.src
        return None

    @staticmethod
    def _resolve_chain_expression(
        state: SDFGState,
        output_tasklet: nodes.Tasklet,
        inner_sdfg: SDFG,
        wired_inputs: set,
    ) -> Optional[sp.Basic]:
        """Resolve a multi-tasklet chain to a SymPy expression over wired inputs.

        Used for branches where SplitTasklets has broken a single tasklet
        into a chain (e.g., constant generator -> operation).
        """
        ise_assignments: Dict[str, str] = {}
        for e in inner_sdfg.all_interstate_edges():
            ise_assignments.update(e.data.assignments)

        input_mapping: Dict[str, sp.Basic] = {}
        for edge in state.in_edges(output_tasklet):
            if not isinstance(edge.src, nodes.AccessNode) or edge.dst_conn is None:
                continue
            arr_name = edge.src.data
            resolved = _resolve_symbol(
                inner_sdfg, arr_name, ise_assignments, wired_inputs, frozenset())
            if resolved is None:
                return None
            input_mapping[edge.dst_conn] = resolved
        return _tasklet_code_to_sympy(output_tasklet.code.as_string, input_mapping)

    @staticmethod
    def _fix_staging_memlets(
        graph: SDFGState,
        sdfg: SDFG,
        compound_node: nodes.LibraryNode,
        out_tile_node: nodes.AccessNode,
        nsdfg_sym_mapping: Dict[str, object],
        inner_map_params: List[str],
        inner_map_ranges: list,
        tile_shape: tuple,
    ) -> None:
        """Reconstruct staging memlets when they are bounding boxes.

        After map collapse + tiling, memlets may be propagated to full-array
        bounding boxes (e.g., ``A[0:24, 0:20]``).  When the tile transient
        is smaller, these cause buffer overflows.  This method reconstructs
        correct per-tile subsets using the NSDFG's symbol_mapping.
        """
        from dace.subsets import Range as SubsetRange
        tile_volume = 1
        for s in tile_shape:
            tile_volume *= int(s) if isinstance(s, (int, sp.Integer)) else s

        # Filter out identity mappings (dimension params like FM → FM)
        # to keep only actual index variables (i → i + tile_i, j → j + ...).
        nsdfg_syms = sorted(
            k for k, v in nsdfg_sym_mapping.items()
            if str(k) != str(v)
        )

        def _to_sympy_val(val):
            if isinstance(val, sp.Basic):
                return val
            if hasattr(val, 'expr'):
                return val.expr
            return sp.sympify(val)

        def _compute_per_tile_subset(arr):
            ndim = len(arr.shape)
            if len(nsdfg_syms) != ndim:
                return None
            ranges = []
            for d, sym_name in enumerate(nsdfg_syms):
                expr = dace.symbolic.pystr_to_symbolic(
                    str(nsdfg_sym_mapping[sym_name]))
                inner_found = False
                for p_idx, p_name in enumerate(inner_map_params):
                    actual_sym = next(
                        (s for s in expr.free_symbols if str(s) == p_name),
                        None)
                    if actual_sym is not None:
                        inner_found = True
                        r_start = _to_sympy_val(inner_map_ranges[p_idx][0])
                        r_end = _to_sympy_val(inner_map_ranges[p_idx][1])
                        start_val = expr.subs(actual_sym, r_start)
                        end_val = expr.subs(actual_sym, r_end)
                        # Clamp end to array bound — the inner map range
                        # may not account for outer tile offsets, so the
                        # computed end can exceed the dimension size.
                        dim_bound = sp.sympify(arr.shape[d]) - 1
                        end_val = sp.Min(end_val, dim_bound)
                        ranges.append((start_val, end_val, 1))
                        break
                if not inner_found:
                    ranges.append((expr, expr, 1))
            return SubsetRange(ranges)

        def _needs_fix(memlet, arr):
            """Check if a staging memlet is a bounding box (too large)."""
            try:
                subset = primary_memlet_subset(memlet)
                if subset is None:
                    return True
                vol = subset.num_elements()
                return vol != tile_volume
            except TypeError:
                return True

        # Fix input staging: MapEntry → tile_transient
        for in_edge in graph.in_edges(compound_node):
            tile_node = in_edge.src
            if not isinstance(tile_node, nodes.AccessNode):
                continue
            if not sdfg.arrays.get(tile_node.data, type('', (), {'transient': False})).transient:
                continue
            for se in graph.in_edges(tile_node):
                if isinstance(se.src, nodes.MapEntry):
                    arr = sdfg.arrays.get(se.data.data)
                    if arr and _needs_fix(se.data, arr):
                        ns = _compute_per_tile_subset(arr)
                        if ns is not None:
                            se.data = with_primary_subset(se.data, ns)

        # Fix output staging: out_tile → MapExit
        for se in graph.out_edges(out_tile_node):
            if isinstance(se.dst, nodes.MapExit):
                arr = sdfg.arrays.get(se.data.data)
                if arr and _needs_fix(se.data, arr):
                    ns = _compute_per_tile_subset(arr)
                    if ns is not None:
                        se.data = with_primary_subset(se.data, ns)

    # ── apply ────────────────────────────────────────────────────────

    def apply(self, graph: SDFGState, sdfg: SDFG) -> None:  # type: ignore[override]
        """Apply the transformation: replace the if-else map with a tile compound node.

        Performs the following graph rewrite:

        1. Analyses the NestedSDFG to extract the condition and branch
           SymPy expressions.
        2. Derives the tile shape from the inner map ranges.
        3. Creates input tile transients and wires them from the outer map
           entry.
        4. Builds a :class:`~dace.libraries.cutile.nodes.if_else_op.TileIfElseOpLibraryNode`
           with the extracted expressions and connects all inputs.
        5. Creates an output tile transient and wires it to the outer map exit.
        6. Removes the original inner map entry/exit and NestedSDFG nodes.

        Args:
            graph: The SDFG state containing the matched subgraph.
            sdfg: The top-level SDFG.
        """
        outer_entry: nodes.MapEntry = self.outer_map_entry
        inner_entry: nodes.MapEntry = self.inner_map_entry
        nsdfg: nodes.NestedSDFG = self.nsdfg_node
        inner_exit: nodes.MapExit = self.inner_map_exit
        outer_exit: nodes.MapExit = self.outer_map_exit
        inner_sdfg = nsdfg.sdfg

        # Capture NSDFG symbol_mapping and inner map info for staging
        # memlet reconstruction (needed when memlets are bounding boxes).
        nsdfg_sym_mapping = dict(nsdfg.symbol_mapping)
        inner_map_params = list(inner_entry.map.params)
        inner_map_ranges = list(inner_entry.map.range.ranges)

        # ── 1. Analyze NestedSDFG ────────────────────────────────────
        analysis_result = self._analyze_nsdfg(inner_sdfg)
        assert analysis_result is not None
        _, cond_expr, true_branch_state, false_branch_state = analysis_result

        # ── 2. Tile shape from inner map ─────────────────────────────
        tile_shape = tuple(inner_entry.map.range.size())
        tile_subset = tile_subset_from_shape(tile_shape)

        # ── 3. Build NSDFG array → outer data mapping ───────────────
        # Maps nsdfg_internal_name → outer_data_name
        nsdfg_to_outer: Dict[str, str] = {}
        nsdfg_input_conns: set = set()

        # Input connectors
        for input_edge in graph.in_edges(nsdfg):
            if input_edge.dst_conn and input_edge.data.data:
                nsdfg_to_outer[input_edge.dst_conn] = input_edge.data.data
                nsdfg_input_conns.add(input_edge.dst_conn)
        # Output connectors map to the same outer array
        for output_edge in graph.out_edges(nsdfg):
            if output_edge.src_conn and output_edge.data.data:
                nsdfg_to_outer[output_edge.src_conn] = output_edge.data.data

        # ── 4. Create tile transients for *input* arrays ─────────────
        # Output-only arrays get their own tile in step 9; creating a
        # tile transient for them here would leave an isolated node.
        # outer_data_name → (tile_trans_name, tile_trans_node)
        outer_to_tile: Dict[str, Tuple[str, nodes.AccessNode]] = {}

        for nsdfg_conn, outer_name in nsdfg_to_outer.items():
            if nsdfg_conn not in nsdfg_input_conns:
                continue
            if outer_name in outer_to_tile:
                continue
            trans_name, trans_node = create_tile_transient(sdfg, graph, outer_name, tile_shape)
            outer_to_tile[outer_name] = (trans_name, trans_node)

        # Map nsdfg_internal_name → tile_trans_name (inputs only)
        nsdfg_to_tile: Dict[str, str] = {}
        for nsdfg_name, outer_name in nsdfg_to_outer.items():
            if outer_name in outer_to_tile:
                nsdfg_to_tile[nsdfg_name] = outer_to_tile[outer_name][0]

        # ── 5. Wire outer entry → tile transients (input staging) ────
        # We need to create edges from outer_entry to each tile transient.
        # Find the relevant outer->inner edges and reuse their memlets.
        wired_tiles: set = set()
        for input_edge in list(graph.in_edges(inner_entry)):
            # input_edge: outer_entry[src_conn] → inner_entry[dst_conn]
            if input_edge.src is not outer_entry:
                continue
            # Find the corresponding inner→nsdfg edge
            for inner_output_edge in graph.out_edges(inner_entry):
                if (inner_output_edge.dst is nsdfg
                        and inner_output_edge.data.data == input_edge.data.data):
                    outer_name = input_edge.data.data
                    if outer_name not in outer_to_tile:
                        continue
                    tile_name, tile_node = outer_to_tile[outer_name]
                    if tile_name in wired_tiles:
                        continue
                    wired_tiles.add(tile_name)

                    # Build staging memlet: preserve outer indexing
                    staging_memlet = copy.deepcopy(input_edge.data)
                    graph.add_edge(outer_entry, input_edge.src_conn,
                                   tile_node, None, staging_memlet)
                    break

        # ── 6. Convert condition and classify branch tasklets ───────
        wired_inputs = {
            input_edge.dst_conn
            for input_edge in graph.in_edges(nsdfg)
            if input_edge.dst_conn is not None and input_edge.data.data is not None
        }
        nsdfg_arrays = set(inner_sdfg.arrays.keys())
        condition_expression = _to_sympy_condition(cond_expr, nsdfg_arrays)
        if condition_expression is not None:
            cond_syms = [str(s) for s in condition_expression.free_symbols]
            if (any(s not in nsdfg.in_connectors for s in cond_syms)
                    or any(s not in wired_inputs for s in cond_syms)):
                condition_expression = None
        if condition_expression is None:
            condition_expression = _resolve_condition_to_wired_inputs(
                inner_sdfg, cond_expr, wired_inputs)
        assert condition_expression is not None, f"Unsupported condition: {cond_expr}"
        condition_symbol_names = sorted(
            str(free_symbol) for free_symbol in condition_expression.free_symbols
        )

        # Classify both branch tasklets (or detect copy/chain branches)
        branch_matches: Dict[str, Tuple[TaskletLibraryNodeMatch, nodes.Tasklet]] = {}   # "true"/"false" → (match, tasklet)
        copy_branches: Dict[str, str] = {}  # branch_name → nsdfg input array name
        chain_branches: Dict[str, sp.Basic] = {}  # branch_name → resolved SymPy expr over NSDFG array names
        for branch_name, branch_state in [("true", true_branch_state), ("false", false_branch_state)]:
            tasklets = [node for node in branch_state.nodes()
                        if isinstance(node, nodes.Tasklet)]
            if len(tasklets) == 1:
                tasklet_match = match_tasklet_to_tile_library_node(
                    branch_state, tasklets[0], MaskType.UNMASKED,
                    promote_scalars=True)
                assert tasklet_match is not None
                branch_matches[branch_name] = (tasklet_match, tasklets[0])
            elif len(tasklets) == 0:
                # Copy branch: AccessNode → AccessNode chain (identity)
                copy_input = self._get_copy_branch_input(branch_state)
                assert copy_input is not None, (
                    f"IfElseMapToTileWhere: {branch_name} branch has no "
                    f"tasklets and no valid copy chain")
                copy_branches[branch_name] = copy_input
            else:
                # Multi-tasklet chain (e.g., from SplitTasklets)
                output_tasklet = self._find_output_tasklet(branch_state)
                assert output_tasklet is not None, (
                    f"IfElseMapToTileWhere: {branch_name} branch has "
                    f"{len(tasklets)} tasklets but no output tasklet found")
                chain_expr = self._resolve_chain_expression(
                    branch_state, output_tasklet, inner_sdfg, wired_inputs)
                assert chain_expr is not None, (
                    f"IfElseMapToTileWhere: {branch_name} branch chain "
                    f"could not be resolved to a SymPy expression")
                chain_branches[branch_name] = chain_expr

        # ── 7. Build SymPy expressions for branches ─────────────────
        # Collect NSDFG arrays used by branch operands.
        needed_nsdfg_names = set(condition_symbol_names)
        # branch_name → {"rhs1": nsdfg_array, "rhs2": nsdfg_array} for non-constant operands
        branch_operand_arrays: Dict[str, Dict[str, str]] = {}

        for branch_name, branch_state in [("true", true_branch_state), ("false", false_branch_state)]:
            if branch_name in copy_branches:
                # Copy branch: the only operand is the input array
                copy_arr = copy_branches[branch_name]
                needed_nsdfg_names.add(copy_arr)
                branch_operand_arrays[branch_name] = {"rhs1": copy_arr}
                continue
            if branch_name in chain_branches:
                # Multi-tasklet chain: collect needed arrays from resolved expr
                for sym in chain_branches[branch_name].free_symbols:
                    needed_nsdfg_names.add(str(sym))
                continue
            tasklet_classification = branch_matches[branch_name][0].tasklet_classification
            tasklet_node = branch_matches[branch_name][1]
            input_mapping = self._get_tasklet_input_mapping(branch_state, tasklet_node)
            operand_arrays: Dict[str, str] = {}
            if (tasklet_classification.rhs1 is not None
                    and tasklet_classification.constant1 is None):
                rhs1_array_name = input_mapping.get(tasklet_classification.rhs1)
                if rhs1_array_name is not None:
                    operand_arrays["rhs1"] = rhs1_array_name
                    needed_nsdfg_names.add(rhs1_array_name)
            if (tasklet_classification.rhs2 is not None
                    and tasklet_classification.constant2 is None):
                rhs2_array_name = input_mapping.get(tasklet_classification.rhs2)
                if rhs2_array_name is not None:
                    operand_arrays["rhs2"] = rhs2_array_name
                    needed_nsdfg_names.add(rhs2_array_name)
            branch_operand_arrays[branch_name] = operand_arrays

        # Assign numbered connectors to unique outer arrays.
        outer_to_conn: Dict[str, str] = {}
        connector_index = 0
        for nsdfg_name in needed_nsdfg_names:
            outer_name = nsdfg_to_outer.get(nsdfg_name)
            if outer_name is not None and outer_name not in outer_to_conn:
                outer_to_conn[outer_name] = f"_in{connector_index}"
                connector_index += 1

        def _nsdfg_to_conn_sym(name: str) -> sp.Symbol:
            outer_name = nsdfg_to_outer.get(name)
            if outer_name is None:
                raise ValueError(
                    f"IfElseMapToTileWhere: NSDFG name '{name}' has no outer mapping."
                )
            return sp.Symbol(outer_to_conn[outer_name])

        # Build a SymPy expression for each branch.
        branch_exprs: Dict[str, sp.Basic] = {}
        for branch_name in ("true", "false"):
            if branch_name in copy_branches:
                # Copy branch: identity expression (output = input)
                operand_arrays = branch_operand_arrays[branch_name]
                branch_exprs[branch_name] = _nsdfg_to_conn_sym(
                    operand_arrays["rhs1"])
                continue
            if branch_name in chain_branches:
                # Multi-tasklet chain: remap NSDFG array names to connector symbols
                chain_expr = chain_branches[branch_name]
                subs = {}
                for sym in chain_expr.free_symbols:
                    subs[sym] = _nsdfg_to_conn_sym(str(sym))
                branch_exprs[branch_name] = chain_expr.xreplace(subs)
                continue
            tasklet_classification = branch_matches[branch_name][0].tasklet_classification
            operand_arrays = branch_operand_arrays[branch_name]
            # Left / first operand
            if tasklet_classification.constant1 is not None:
                left_operand = sp.sympify(tasklet_classification.constant1)
            elif "rhs1" in operand_arrays:
                left_operand = _nsdfg_to_conn_sym(operand_arrays["rhs1"])
            else:
                raise ValueError(
                    f"IfElseMapToTileWhere: {branch_name} branch has no left operand"
                )
            # Right / second operand (None for unary)
            right_operand: Optional[sp.Basic] = None
            if tasklet_classification.constant2 is not None:
                right_operand = sp.sympify(tasklet_classification.constant2)
            elif "rhs2" in operand_arrays:
                right_operand = _nsdfg_to_conn_sym(operand_arrays["rhs2"])
            branch_exprs[branch_name] = _build_sympy_expr(
                tasklet_classification.op, left_operand, right_operand)

        # Rewrite condition symbols from NSDFG names to connector names.
        condition_substitutions = {}
        for symbol in condition_expression.free_symbols:
            symbol_name = str(symbol)
            outer_name = nsdfg_to_outer.get(symbol_name)
            if outer_name is None:
                raise ValueError(
                    "IfElseMapToTileWhere: condition symbol "
                    f"'{symbol_name}' is not mapped to an outer array."
                )
            conn_name = outer_to_conn.get(outer_name)
            if conn_name is None:
                raise ValueError(
                    "IfElseMapToTileWhere: no input connector assigned for "
                    f"condition symbol '{symbol_name}' (outer '{outer_name}')."
                )
            condition_substitutions[symbol] = sp.Symbol(conn_name)
        node_condition = condition_expression.xreplace(condition_substitutions)

        # ── 8. Create TileIfElseOpLibraryNode ────────────────────────
        compound_node = TileIfElseOpLibraryNode(
            name="TileIfElseOp",
            condition=node_condition,
            true_expr=branch_exprs["true"],
            false_expr=branch_exprs["false"],
            tile_shape=list(tile_shape),
        )
        graph.add_node(compound_node)

        # Wire tile transients → compound node inputs
        for outer_name, conn_name in outer_to_conn.items():
            tile_name, tile_node = outer_to_tile[outer_name]
            graph.add_edge(tile_node, None, compound_node, conn_name,
                           memlet_with_primary_subset(tile_name,
                                                      tile_subset,
                                                      data_on_src=True))

        # ── 9. Connect output to outer exit ──────────────────────────
        output_nsdfg_array = self._get_branch_output_array(true_branch_state)
        if output_nsdfg_array is None:
            raise ValueError(
                "IfElseMapToTileWhere: true branch does not expose an output "
                "array to connect."
            )
        output_outer_name = nsdfg_to_outer[output_nsdfg_array]

        out_tile_name, out_tile_node = create_tile_transient(
            sdfg, graph, output_outer_name, tile_shape, suffix="_out_tile"
        )

        graph.add_edge(compound_node, "_out", out_tile_node, None,
                       memlet_with_primary_subset(out_tile_name,
                                                  tile_subset,
                                                  data_on_src=False))

        # Find the original nsdfg → inner_exit → outer_exit edge
        # and reuse its memlet for the output store
        store_memlet = None
        for output_edge in list(graph.out_edges(nsdfg)):
            if output_edge.dst is inner_exit and output_edge.src_conn == output_nsdfg_array:
                # Find inner_exit → outer_exit edge
                for outer_exit_edge in graph.out_edges(inner_exit):
                    if outer_exit_edge.data.data == output_outer_name:
                        store_memlet = copy.deepcopy(outer_exit_edge.data)
                        graph.add_edge(out_tile_node, None, outer_exit,
                                       outer_exit_edge.dst_conn, store_memlet)
                        break
                break
        if store_memlet is None:
            raise ValueError(
                f"IfElseMapToTileWhere: could not find output edge from "
                f"NestedSDFG to outer map exit for '{output_outer_name}'"
            )

        # ── 10. Remove original inner map and NestedSDFG ─────────────
        # Remove edges first
        for input_edge in list(graph.in_edges(inner_entry)):
            if input_edge.src is outer_entry:
                graph.remove_edge(input_edge)
        for output_edge in list(graph.out_edges(inner_exit)):
            if output_edge.dst is outer_exit:
                graph.remove_edge(output_edge)

        # Remove nodes
        graph.remove_node(nsdfg)
        graph.remove_node(inner_entry)
        graph.remove_node(inner_exit)

        # ── 11. Fix staging memlets for bounding-box cases ───────────
        # When memlets are propagated bounding boxes (e.g., A[0:24, 0:20]
        # for a 6-element tile), reconstruct correct per-tile subsets
        # using the NSDFG's symbol_mapping and inner map range.
        self._fix_staging_memlets(
            graph, sdfg, compound_node, out_tile_node,
            nsdfg_sym_mapping, inner_map_params, inner_map_ranges,
            tile_shape,
        )
