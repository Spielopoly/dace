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

import ast
import copy
import re
from typing import Dict, List, Optional, Tuple

import dace
from dace import Memlet, dtypes, subsets
from dace.sdfg import SDFG, SDFGState, nodes, utils as sdutil
from dace.sdfg.state import ConditionalBlock, ControlFlowRegion
from dace.transformation import transformation as xf

from dace.libraries.cutile.nodes.if_else_op import TileIfElseOpLibraryNode
from dace.libraries.cutile.op_registry import (
    match_tasklet_to_tile_library_node,
    MaskType,
)


# ── AST comparison-operator map ──────────────────────────────────────

_AST_CMP_OPS = {
    ast.Gt: ">", ast.Lt: "<", ast.GtE: ">=", ast.LtE: "<=",
    ast.Eq: "==", ast.NotEq: "!=",
}


# ── Helpers ──────────────────────────────────────────────────────────

def _nsdfg_array_refs_in_expr(expr_str: str, nsdfg_arrays: set) -> List[str]:
    """Return NSDFG array/scalar names referenced in *expr_str*."""
    refs = []
    for name in nsdfg_arrays:
        if re.search(r'\b' + re.escape(name) + r'\b', expr_str):
            refs.append(name)
    return refs


def _parse_condition_expr(
    cond_expr: str, nsdfg_arrays: set,
) -> Optional[Tuple[str, Optional[str], Optional[str], Optional[str]]]:
    """Parse a condition expression into comparison components.

    Returns ``(op, left_ref, right_ref, constant)`` where:

    * *op* is one of ``">", "<", ">=", "<=", "==", "!="``.
    * *left_ref* is the NSDFG array name on the left (always present).
    * *right_ref* is the NSDFG array name on the right, or ``None``
      when the right operand is a constant.
    * *constant* is the string representation of the constant operand,
      or ``None`` when the right operand is an array reference.

    Returns ``None`` if the expression cannot be parsed.
    """
    try:
        tree = ast.parse(cond_expr, mode="eval")
    except SyntaxError:
        return None

    body = tree.body
    if not isinstance(body, ast.Compare):
        return None
    if len(body.ops) != 1 or len(body.comparators) != 1:
        return None

    op_str = _AST_CMP_OPS.get(type(body.ops[0]))
    if op_str is None:
        return None

    # Left operand – must be an NSDFG array reference.
    if not isinstance(body.left, ast.Name) or body.left.id not in nsdfg_arrays:
        return None
    left_ref = body.left.id

    # Right operand – either another reference or a literal constant.
    right_node = body.comparators[0]
    if isinstance(right_node, ast.Name) and right_node.id in nsdfg_arrays:
        return op_str, left_ref, right_node.id, None
    if isinstance(right_node, ast.Constant):
        return op_str, left_ref, None, str(right_node.value)
    # Handle negative constants: ast.UnaryOp(op=USub, operand=Constant)
    if (isinstance(right_node, ast.UnaryOp)
            and isinstance(right_node.op, ast.USub)
            and isinstance(right_node.operand, ast.Constant)):
        return op_str, left_ref, None, str(-right_node.operand.value)

    return None


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
    def expressions(cls):
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
        for start, _, step in inner_entry.map.range:
            if start != 0 or step != 1:
                return False

        # Analyze the NestedSDFG for the if-else pattern
        info = self._analyze_nsdfg(nsdfg.sdfg)
        if info is None:
            return False

        cond_block, cond_expr, branch_true_state, branch_false_state = info

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

    def apply(self, graph: SDFGState, sdfg: SDFG):
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
        tile_subset = subsets.Range([(0, d - 1, 1) for d in tile_shape])

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
            data_desc = sdfg.arrays[outer_name]
            trans_name = sdfg._find_new_name(outer_name + "_tile")
            sdfg.add_transient(
                trans_name,
                shape=tile_shape,
                dtype=data_desc.dtype,
                storage=data_desc.storage,
                lifetime=dtypes.AllocationLifetime.Scope,
            )
            trans_node = graph.add_access(trans_name)
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

        # ── 6. Analyze condition and classify branch tasklets ─────
        nsdfg_arrays = set(inner_sdfg.arrays.keys())
        parsed = _parse_condition_expr(cond_expr, nsdfg_arrays)
        assert parsed is not None, f"Unsupported condition: {cond_expr}"
        cmp_op, cond_left, cond_right, cond_const = parsed

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

        true_cls = branch_cls["true"][0].tasklet_classification
        false_cls = branch_cls["false"][0].tasklet_classification

        # Determine effective constants per branch.  Unary negate (-A)
        # must be expressed as binary ``0 - A`` for the compound node
        # because "-" is in _BINARY_OPS and the validator expects rhs2.
        branch_const: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        for tag in ("true", "false"):
            cls = branch_cls[tag][0].tasklet_classification
            c1, c2 = cls.constant1, cls.constant2
            if (cls.op == "-" and cls.rhs2 is None
                    and c1 is None and c2 is None):
                # Unary negate → binary 0 - A
                c1 = "0"
            branch_const[tag] = (c1, c2)

        # ── 7. Build input_roles for the compound node ───────────────
        # Map NSDFG array names → list of roles they fulfil.
        nsdfg_roles: Dict[str, List[str]] = {}

        # Condition roles
        nsdfg_roles.setdefault(cond_left, []).append("cond_left")
        if cond_right is not None:
            nsdfg_roles.setdefault(cond_right, []).append("cond_right")

        # Branch roles (true / false)
        for tag, bstate in [("true", true_state), ("false", false_state)]:
            cls = branch_cls[tag][0].tasklet_classification
            tasklet = branch_cls[tag][1]
            inp_map = self._get_tasklet_input_mapping(bstate, tasklet)
            is_unary_negate = (cls.op == "-" and cls.rhs2 is None
                               and cls.constant1 is None
                               and cls.constant2 is None)
            if is_unary_negate:
                # rhs1 becomes rhs2 (binary 0 - A)
                if cls.rhs1 is not None:
                    arr = inp_map.get(cls.rhs1)
                    if arr is not None:
                        nsdfg_roles.setdefault(arr, []).append(
                            f"{tag}_rhs2")
            else:
                if cls.rhs1 is not None and cls.constant1 is None:
                    arr = inp_map.get(cls.rhs1)
                    if arr is not None:
                        nsdfg_roles.setdefault(arr, []).append(
                            f"{tag}_rhs1")
                if cls.rhs2 is not None and cls.constant2 is None:
                    arr = inp_map.get(cls.rhs2)
                    if arr is not None:
                        nsdfg_roles.setdefault(arr, []).append(
                            f"{tag}_rhs2")

        # Assign numbered connectors to unique outer arrays.
        outer_to_conn: Dict[str, str] = {}
        conn_idx = 0
        for nsdfg_name in nsdfg_roles:
            outer_name = nsdfg_to_outer.get(nsdfg_name)
            if outer_name is not None and outer_name not in outer_to_conn:
                outer_to_conn[outer_name] = f"_in{conn_idx}"
                conn_idx += 1

        # Build input_roles: connector → list of roles
        input_roles: Dict[str, List[str]] = {}
        for nsdfg_name, roles in nsdfg_roles.items():
            outer_name = nsdfg_to_outer.get(nsdfg_name)
            if outer_name is not None and outer_name in outer_to_conn:
                conn = outer_to_conn[outer_name]
                input_roles.setdefault(conn, []).extend(roles)

        # ── 8. Create TileIfElseOpLibraryNode ────────────────────────
        compound_node = TileIfElseOpLibraryNode(
            name="TileIfElseOp",
            cond_op=cmp_op,
            cond_constant=cond_const,
            true_op=true_cls.op,
            true_constant1=branch_const["true"][0],
            true_constant2=branch_const["true"][1],
            false_op=false_cls.op,
            false_constant1=branch_const["false"][0],
            false_constant2=branch_const["false"][1],
            tile_shape=list(tile_shape),
            input_roles=input_roles,
            num_inputs=len(outer_to_conn),
        )
        graph.add_node(compound_node)

        # Wire tile transients → compound node inputs
        for outer_name, conn_name in outer_to_conn.items():
            tile_name, tile_node = outer_to_tile[outer_name]
            graph.add_edge(tile_node, None, compound_node, conn_name,
                           Memlet(data=tile_name, subset=tile_subset))

        # ── 9. Connect output to outer exit ──────────────────────────
        output_nsdfg_array = self._get_branch_output_array(true_state)
        output_outer_name = nsdfg_to_outer[output_nsdfg_array]

        out_tile_name = sdfg._find_new_name(output_outer_name + "_out_tile")
        sdfg.add_transient(
            out_tile_name,
            shape=tile_shape,
            dtype=sdfg.arrays[output_outer_name].dtype,
            storage=sdfg.arrays[output_outer_name].storage,
            lifetime=dtypes.AllocationLifetime.Scope,
        )
        out_tile_node = graph.add_access(out_tile_name)

        graph.add_edge(compound_node, "_out", out_tile_node, None,
                       Memlet(data=out_tile_name, subset=tile_subset))

        # Find the original nsdfg → inner_exit → outer_exit edge
        # and reuse its memlet for the output store
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
