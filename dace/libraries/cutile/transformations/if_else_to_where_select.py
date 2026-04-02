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
from __future__ import annotations

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


# ── Transformation ───────────────────────────────────────────────────

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

        nsdfg_arrays = set(nsdfg.sdfg.arrays.keys())
        condition_expression = _to_sympy_condition(cond_expr, nsdfg_arrays)
        if condition_expression is None:
            return False
        condition_symbol_names = [
            str(free_symbol) for free_symbol in condition_expression.free_symbols
        ]
        if any(symbol_name not in nsdfg.in_connectors
               for symbol_name in condition_symbol_names):
            return False

        # Each condition symbol must be both declared and wired from outer scope.
        wired_inputs = {
            input_edge.dst_conn
            for input_edge in graph.in_edges(nsdfg)
            if input_edge.dst_conn is not None and input_edge.data.data is not None
        }
        if any(symbol_name not in wired_inputs
               for symbol_name in condition_symbol_names):
            return False

        # Each branch must have exactly one tasklet with one output
        for branch_state in (true_branch_state, false_branch_state):
            tasklets = [node for node in branch_state.nodes()
                        if isinstance(node, nodes.Tasklet)]
            if len(tasklets) != 1:
                return False
            tasklet_node = tasklets[0]
            if len(tasklet_node.out_connectors) != 1:
                return False
            # Classify using the NSDFG's internal state (promote
            # scalar types since these are element-wise ops inside a
            # NestedSDFG that will be lifted to tile-level).
            tasklet_match = match_tasklet_to_tile_library_node(
                branch_state, tasklet_node, MaskType.UNMASKED,
                promote_scalars=True)
            if tasklet_match is None:
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
        """Get the data name written by the single tasklet in a branch state.

        Args:
            state: The branch :class:`~dace.sdfg.SDFGState` containing
                exactly one :class:`~dace.sdfg.nodes.Tasklet`.

        Returns:
            The ``data`` name of the access node written by the tasklet,
            or ``None`` if the pattern is not matched.
        """
        tasklets = [node for node in state.nodes() if isinstance(node, nodes.Tasklet)]
        if len(tasklets) != 1:
            return None
        tasklet_node = tasklets[0]
        for output_edge in state.out_edges(tasklet_node):
            if isinstance(output_edge.dst, nodes.AccessNode):
                return output_edge.dst.data
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
        nsdfg_arrays = set(inner_sdfg.arrays.keys())
        condition_expression = _to_sympy_condition(cond_expr, nsdfg_arrays)
        assert condition_expression is not None, f"Unsupported condition: {cond_expr}"
        condition_symbol_names = sorted(
            str(free_symbol) for free_symbol in condition_expression.free_symbols
        )
        for symbol_name in condition_symbol_names:
            assert symbol_name in nsdfg.in_connectors, (
                "Condition may only reference NestedSDFG inputs; "
                f"got symbol '{symbol_name}' in '{cond_expr}'"
            )

        # Classify both branch tasklets
        branch_matches: Dict[str, Tuple[TaskletLibraryNodeMatch, nodes.Tasklet]] = {}   # "true"/"false" → (match, tasklet)
        for branch_name, branch_state in [("true", true_branch_state), ("false", false_branch_state)]:
            tasklets = [node for node in branch_state.nodes()
                        if isinstance(node, nodes.Tasklet)]
            assert len(tasklets) == 1
            tasklet_match = match_tasklet_to_tile_library_node(
                branch_state, tasklets[0], MaskType.UNMASKED,
                promote_scalars=True)
            assert tasklet_match is not None
            branch_matches[branch_name] = (tasklet_match, tasklets[0])

        # ── 7. Build SymPy expressions for branches ─────────────────
        # Collect NSDFG arrays used by branch operands.
        needed_nsdfg_names = set(condition_symbol_names)
        # branch_name → {"rhs1": nsdfg_array, "rhs2": nsdfg_array} for non-constant operands
        branch_operand_arrays: Dict[str, Dict[str, str]] = {}

        for branch_name, branch_state in [("true", true_branch_state), ("false", false_branch_state)]:
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
                           Memlet(data=tile_name, subset=tile_subset))

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
                       Memlet(data=out_tile_name, subset=tile_subset))

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
