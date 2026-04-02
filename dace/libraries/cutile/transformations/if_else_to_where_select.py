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
    cond_expr: str, nsdfg_arrays: set,
) -> Optional[sp.Basic]:
    """Convert a condition expression string to a SymPy expression.

    All free symbols in the resulting expression must map to NSDFG array
    names. Returns ``None`` if parsing fails or unknown symbols appear.
    """
    expr = dace.symbolic.pystr_to_symbolic(cond_expr, simplify=False)

    if not isinstance(expr, sp.Basic):
        return None

    symbols = {str(sym) for sym in expr.free_symbols}
    if not symbols.issubset(nsdfg_arrays):
        return None

    return expr


def _build_sympy_expr(
    op: str, left: sp.Basic, right: Optional[sp.Basic] = None,
) -> sp.Basic:
    """Build a SymPy expression from a classified operation.

    Parameters
    ----------
    op : str
        Operation string (e.g. ``"+"``, ``"abs"``, ``"sin"``).
    left : sp.Basic
        Left / first operand (Symbol or Number).
    right : sp.Basic or None
        Right / second operand.  ``None`` for unary operations.

    Returns
    -------
    sp.Basic
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
        info = self._analyze_nsdfg(nsdfg.sdfg)
        if info is None:
            return False

        cond_block, cond_expr, branch_true_state, branch_false_state = info

        nsdfg_arrays = set(nsdfg.sdfg.arrays.keys())
        cond_sympy = _to_sympy_condition(cond_expr, nsdfg_arrays)
        if cond_sympy is None:
            return False
        cond_symbols = [str(sym) for sym in cond_sympy.free_symbols]
        if any(sym not in nsdfg.in_connectors for sym in cond_symbols):
            return False

        # Each condition symbol must be both declared and wired from outer scope.
        wired_inputs = {
            ie.dst_conn
            for ie in graph.in_edges(nsdfg)
            if ie.dst_conn is not None and ie.data.data is not None
        }
        if any(sym not in wired_inputs for sym in cond_symbols):
            return False

        # Each branch must have exactly one tasklet with one output
        for bstate in (branch_true_state, branch_false_state):
            tasklets = [n for n in bstate.nodes()
                        if isinstance(n, nodes.Tasklet)]
            if len(tasklets) != 1:
                return False
            t = tasklets[0]
            if len(t.out_connectors) != 1:
                return False
            # Classify using the NSDFG's internal state (promote
            # scalar types since these are element-wise ops inside a
            # NestedSDFG that will be lifted to tile-level).
            match = match_tasklet_to_tile_library_node(
                bstate, t, MaskType.UNMASKED, promote_scalars=True)
            if match is None:
                return False

        # Both branches must write to the same output connector
        true_out = self._get_branch_output_array(branch_true_state)
        false_out = self._get_branch_output_array(branch_false_state)
        if true_out is None or false_out is None:
            return False
        if true_out != false_out:
            return False

        # The output must be an NSDFG output connector
        if true_out not in nsdfg.out_connectors:
            return False

        return True

    # ── NestedSDFG analysis ──────────────────────────────────────────

    @staticmethod
    def _analyze_nsdfg(inner_sdfg: SDFG) -> Optional[Tuple[
            ConditionalBlock, str, SDFGState, SDFGState]]:
        """
        Analyze a NestedSDFG for the if-else pattern.

        Returns ``(cond_block, cond_expr, true_state, false_state)`` or
        ``None`` if the pattern is not matched.

        Expected CFG structure::

            empty_state --(assigns condition symbol)--> ConditionalBlock

        The ConditionalBlock must have exactly 2 branches: one with a
        condition and one with ``None`` (else).
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
        ies = inner_sdfg.in_edges(cond_block)
        cond_expr = None
        cond_symbol = None

        # Determine which branch is "if" (has condition) and which is "else"
        (cond0, body0), (cond1, body1) = cond_block.branches
        if cond0 is not None and cond1 is None:
            cond_symbol = cond0.as_string.strip()
            true_body, false_body = body0, body1
        elif cond0 is None and cond1 is not None:
            cond_symbol = cond1.as_string.strip()
            true_body, false_body = body1, body0
        else:
            return None

        # Find the condition assignment in interstate edges
        for ie in ies:
            if cond_symbol in ie.data.assignments:
                cond_expr = ie.data.assignments[cond_symbol]
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
        """Get the data name written by the single tasklet in a branch state."""
        tasklets = [n for n in state.nodes() if isinstance(n, nodes.Tasklet)]
        if len(tasklets) != 1:
            return None
        t = tasklets[0]
        for oe in state.out_edges(t):
            if isinstance(oe.dst, nodes.AccessNode):
                return oe.dst.data
        return None

    @staticmethod
    def _get_tasklet_input_mapping(state: SDFGState,
                                   tasklet: nodes.Tasklet) -> Dict[str, str]:
        """
        Map tasklet input connector names to NSDFG array names.

        Returns ``{tasklet_conn: nsdfg_array_name}``.
        """
        mapping = {}
        for ie in state.in_edges(tasklet):
            if isinstance(ie.src, nodes.AccessNode) and ie.dst_conn is not None:
                mapping[ie.dst_conn] = ie.src.data
            elif ie.dst_conn is not None and ie.data.data is not None:
                mapping[ie.dst_conn] = ie.data.data
        return mapping

    # ── apply ────────────────────────────────────────────────────────

    def apply(self, graph: SDFGState, sdfg: SDFG):  # type: ignore[override]
        outer_entry: nodes.MapEntry = self.outer_map_entry
        inner_entry: nodes.MapEntry = self.inner_map_entry
        nsdfg: nodes.NestedSDFG = self.nsdfg_node
        inner_exit: nodes.MapExit = self.inner_map_exit
        outer_exit: nodes.MapExit = self.outer_map_exit
        inner_sdfg = nsdfg.sdfg

        # ── 1. Analyze NestedSDFG ────────────────────────────────────
        info = self._analyze_nsdfg(inner_sdfg)
        assert info is not None
        cond_block, cond_expr, true_state, false_state = info

        # ── 2. Tile shape from inner map ─────────────────────────────
        tile_shape = tuple(inner_entry.map.range.size())
        tile_subset = tile_subset_from_shape(tile_shape)

        # ── 3. Build NSDFG array → outer data mapping ───────────────
        # Maps nsdfg_internal_name → outer_data_name
        nsdfg_to_outer: Dict[str, str] = {}
        nsdfg_input_conns: set = set()

        # Input connectors
        for ie in graph.in_edges(nsdfg):
            if ie.dst_conn and ie.data.data:
                nsdfg_to_outer[ie.dst_conn] = ie.data.data
                nsdfg_input_conns.add(ie.dst_conn)
        # Output connectors map to the same outer array
        for oe in graph.out_edges(nsdfg):
            if oe.src_conn and oe.data.data:
                nsdfg_to_outer[oe.src_conn] = oe.data.data

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
        for ie in list(graph.in_edges(inner_entry)):
            # ie: outer_entry[src_conn] → inner_entry[dst_conn]
            if ie.src is not outer_entry:
                continue
            # Find the corresponding inner→nsdfg edge
            for ie2 in graph.out_edges(inner_entry):
                if ie2.dst is nsdfg and ie2.data.data == ie.data.data:
                    nsdfg_conn = ie2.dst_conn
                    outer_name = ie.data.data
                    if outer_name not in outer_to_tile:
                        continue
                    tile_name, tile_node = outer_to_tile[outer_name]
                    if tile_name in wired_tiles:
                        continue
                    wired_tiles.add(tile_name)

                    # Build staging memlet: preserve outer indexing
                    staging_memlet = copy.deepcopy(ie.data)
                    graph.add_edge(outer_entry, ie.src_conn,
                                   tile_node, None, staging_memlet)
                    break

        # ── 6. Convert condition and classify branch tasklets ───────
        nsdfg_arrays = set(inner_sdfg.arrays.keys())
        cond_sympy = _to_sympy_condition(cond_expr, nsdfg_arrays)
        assert cond_sympy is not None, f"Unsupported condition: {cond_expr}"
        cond_symbols = sorted(str(sym) for sym in cond_sympy.free_symbols)
        for sym in cond_symbols:
            assert sym in nsdfg.in_connectors, (
                "Condition may only reference NestedSDFG inputs; "
                f"got symbol '{sym}' in '{cond_expr}'"
            )

        # Classify both branch tasklets
        branch_cls = {}   # "true"/"false" → (match, tasklet)
        for tag, bstate in [("true", true_state), ("false", false_state)]:
            tasklets = [n for n in bstate.nodes()
                        if isinstance(n, nodes.Tasklet)]
            assert len(tasklets) == 1
            match = match_tasklet_to_tile_library_node(
                bstate, tasklets[0], MaskType.UNMASKED,
                promote_scalars=True)
            assert match is not None
            branch_cls[tag] = (match, tasklets[0])

        # ── 7. Build SymPy expressions for branches ─────────────────
        # Collect NSDFG arrays used by branch operands.
        needed_nsdfg_names: List[str] = []
        # tag → {"rhs1": nsdfg_array, "rhs2": nsdfg_array} for non-constant operands
        branch_operand_arrays: Dict[str, Dict[str, str]] = {}

        for tag, bstate in [("true", true_state), ("false", false_state)]:
            cls = branch_cls[tag][0].tasklet_classification
            tasklet = branch_cls[tag][1]
            inp_map = self._get_tasklet_input_mapping(bstate, tasklet)
            operand_arrays: Dict[str, str] = {}
            if cls.rhs1 is not None and cls.constant1 is None:
                arr = inp_map.get(cls.rhs1)
                if arr is not None:
                    operand_arrays["rhs1"] = arr
                    if arr not in needed_nsdfg_names:
                        needed_nsdfg_names.append(arr)
            if cls.rhs2 is not None and cls.constant2 is None:
                arr = inp_map.get(cls.rhs2)
                if arr is not None:
                    operand_arrays["rhs2"] = arr
                    if arr not in needed_nsdfg_names:
                        needed_nsdfg_names.append(arr)
            branch_operand_arrays[tag] = operand_arrays

        # Add condition symbols.
        for sym in cond_symbols:
            if sym not in needed_nsdfg_names:
                needed_nsdfg_names.append(sym)

        # Assign numbered connectors to unique outer arrays.
        outer_to_conn: Dict[str, str] = {}
        conn_idx = 0
        for nsdfg_name in needed_nsdfg_names:
            outer_name = nsdfg_to_outer.get(nsdfg_name)
            if outer_name is not None and outer_name not in outer_to_conn:
                outer_to_conn[outer_name] = f"_in{conn_idx}"
                conn_idx += 1

        def _nsdfg_to_conn_sym(name: str) -> sp.Symbol:
            return sp.Symbol(outer_to_conn[nsdfg_to_outer[name]])

        # Build a SymPy expression for each branch.
        branch_exprs: Dict[str, sp.Basic] = {}
        for tag in ("true", "false"):
            cls = branch_cls[tag][0].tasklet_classification
            oa = branch_operand_arrays[tag]
            # Left / first operand
            if cls.constant1 is not None:
                left = sp.sympify(cls.constant1)
            elif "rhs1" in oa:
                left = _nsdfg_to_conn_sym(oa["rhs1"])
            else:
                raise ValueError(
                    f"IfElseMapToTileWhere: {tag} branch has no left operand"
                )
            # Right / second operand (None for unary)
            right: Optional[sp.Basic] = None
            if cls.constant2 is not None:
                right = sp.sympify(cls.constant2)
            elif "rhs2" in oa:
                right = _nsdfg_to_conn_sym(oa["rhs2"])
            branch_exprs[tag] = _build_sympy_expr(cls.op, left, right)

        # Rewrite condition symbols from NSDFG names to connector names.
        cond_subs = {}
        for fsym in cond_sympy.free_symbols:
            sym = str(fsym)
            outer_name = nsdfg_to_outer.get(sym)
            if outer_name is None:
                raise ValueError(
                    "IfElseMapToTileWhere: condition symbol "
                    f"'{sym}' is not mapped to an outer array."
                )
            conn_name = outer_to_conn.get(outer_name)
            if conn_name is None:
                raise ValueError(
                    "IfElseMapToTileWhere: no input connector assigned for "
                    f"condition symbol '{sym}' (outer '{outer_name}')."
                )
            cond_subs[fsym] = sp.Symbol(conn_name)
        node_condition = cond_sympy.xreplace(cond_subs)

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
        output_nsdfg_array = self._get_branch_output_array(true_state)
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
        for oe in list(graph.out_edges(nsdfg)):
            if oe.dst is inner_exit and oe.src_conn == output_nsdfg_array:
                # Find inner_exit → outer_exit edge
                for oe2 in graph.out_edges(inner_exit):
                    if oe2.data.data == output_outer_name:
                        store_memlet = copy.deepcopy(oe2.data)
                        graph.add_edge(out_tile_node, None, outer_exit,
                                       oe2.dst_conn, store_memlet)
                        break
                break
        if store_memlet is None:
            raise ValueError(
                f"IfElseMapToTileWhere: could not find output edge from "
                f"NestedSDFG to outer map exit for '{output_outer_name}'"
            )

        # ── 10. Remove original inner map and NestedSDFG ─────────────
        # Remove edges first
        for e in list(graph.in_edges(inner_entry)):
            if e.src is outer_entry:
                graph.remove_edge(e)
        for e in list(graph.out_edges(inner_exit)):
            if e.dst is outer_exit:
                graph.remove_edge(e)

        # Remove nodes
        graph.remove_node(nsdfg)
        graph.remove_node(inner_entry)
        graph.remove_node(inner_exit)
