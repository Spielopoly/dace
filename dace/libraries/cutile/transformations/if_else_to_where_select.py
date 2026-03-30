"""
IfElseMapToTileWhere – transform tiled maps with if-else nested SDFGs to
tile-level unconditional branches + where-select.

Pattern (BEFORE, after MapTiling)::

    OuterMapEntry → InnerMapEntry → NestedSDFG(if-else) → InnerMapExit → OuterMapExit

Result (AFTER)::

    OuterMapEntry → [input tile transients]
                  → [TileOp(comparison) → cond_tile]
                  → [TileOp(true branch) → true_tile]
                  → [TileOp(false branch) → false_tile]
                  → [TileWhereSelect(cond, true, false) → output tile]
                  → OuterMapExit

Both branches are executed unconditionally on full tiles. A where-select node
picks the correct result per element based on the condition mask. This mirrors
``cuda.tile.where(cond, x, y)`` from NVIDIA's cuTile library.
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

from dace.libraries.cutile.nodes.op import TileOpLibraryNode
from dace.libraries.cutile.nodes.where_select import TileWhereSelectLibraryNode
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
    ConditionalBlock by tile-level operations and a where-select node.

    Both branches are executed unconditionally on full tiles.  A boolean
    condition tile is computed element-wise from the original branch
    condition, and a ``TileWhereSelectLibraryNode`` picks the correct
    result per element.
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

                    # Build staging memlet: preserve outer indexing,
                    # add other_subset for tile mapping
                    staging_memlet = copy.deepcopy(ie.data)
                    staging_memlet.other_subset = tile_subset
                    graph.add_edge(outer_entry, ie.src_conn,
                                   tile_node, None, staging_memlet)
                    break

        # ── 6. Create condition mask tile via TileOp comparison ────
        nsdfg_arrays = set(inner_sdfg.arrays.keys())
        parsed = _parse_condition_expr(cond_expr, nsdfg_arrays)
        assert parsed is not None, f"Unsupported condition: {cond_expr}"
        cmp_op, cond_left, cond_right, cond_const = parsed

        cond_tile_name = sdfg._find_new_name("cond_tile")
        sdfg.add_transient(
            cond_tile_name,
            shape=tile_shape,
            dtype=dace.bool,
            lifetime=dtypes.AllocationLifetime.Scope,
        )

        # Build the comparison TileOp library node.
        cmp_node = TileOpLibraryNode(
            name="TileOp_condition",
            op=cmp_op,
            tile_shape=list(tile_shape),
            constant2=cond_const,   # None when comparing two arrays
        )
        graph.add_node(cmp_node)

        # Connect left operand (_a) from existing tile transient.
        left_tile = nsdfg_to_tile[cond_left]
        left_node = outer_to_tile[nsdfg_to_outer[cond_left]][1]
        graph.add_edge(left_node, None, cmp_node, "_a",
                       Memlet(data=left_tile, subset=tile_subset))

        # Connect right operand (_b) or remove the connector.
        if cond_right is not None:
            right_tile = nsdfg_to_tile[cond_right]
            right_node = outer_to_tile[nsdfg_to_outer[cond_right]][1]
            graph.add_edge(right_node, None, cmp_node, "_b",
                           Memlet(data=right_tile, subset=tile_subset))
        else:
            # Right is a constant – remove unused _b connector.
            if "_b" in cmp_node.in_connectors:
                cmp_node.remove_in_connector("_b")

        # Connect output.
        cond_tile_write = graph.add_access(cond_tile_name)
        graph.add_edge(cmp_node, "_c", cond_tile_write, None,
                       Memlet(data=cond_tile_name, subset=tile_subset))

        # ── 7. Create TileOps for both branches ─────────────────────
        true_tile_name, true_tile_node = self._create_branch_tile_op(
            graph, sdfg, "true", true_state, inner_sdfg,
            nsdfg_to_tile, nsdfg_to_outer, outer_to_tile,
            tile_shape, tile_subset,
        )
        false_tile_name, false_tile_node = self._create_branch_tile_op(
            graph, sdfg, "false", false_state, inner_sdfg,
            nsdfg_to_tile, nsdfg_to_outer, outer_to_tile,
            tile_shape, tile_subset,
        )

        # ── 8. Create TileWhereSelect node ──────────────────────────
        where_node = TileWhereSelectLibraryNode(
            "TileWhereSelect", tile_shape=list(tile_shape))
        graph.add_node(where_node)

        # Connect condition - reuse the write node from the condition map
        graph.add_edge(cond_tile_write, None, where_node, "_cond",
                       Memlet(data=cond_tile_name, subset=tile_subset))

        # Connect true/false tiles - reuse result nodes from branch tile ops
        graph.add_edge(true_tile_node, None, where_node, "_x",
                       Memlet(data=true_tile_name, subset=tile_subset))
        graph.add_edge(false_tile_node, None, where_node, "_y",
                       Memlet(data=false_tile_name, subset=tile_subset))

        # ── 9. Connect output to outer exit ──────────────────────────
        # Find the output connector and original memlet
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

        graph.add_edge(where_node, "_c", out_tile_node, None,
                       Memlet(data=out_tile_name, subset=tile_subset))

        # Find the original nsdfg → inner_exit → outer_exit edge
        # and reuse its memlet for the output store
        for oe in list(graph.out_edges(nsdfg)):
            if oe.dst is inner_exit and oe.src_conn == output_nsdfg_array:
                # Find inner_exit → outer_exit edge
                for oe2 in graph.out_edges(inner_exit):
                    if oe2.data.data == output_outer_name:
                        store_memlet = copy.deepcopy(oe2.data)
                        store_memlet.other_subset = tile_subset
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

    # ── Branch TileOp creation ───────────────────────────────────────

    @staticmethod
    def _create_branch_tile_op(
        graph: SDFGState,
        sdfg: SDFG,
        branch_tag: str,
        branch_state: SDFGState,
        inner_sdfg: SDFG,
        nsdfg_to_tile: Dict[str, str],
        nsdfg_to_outer: Dict[str, str],
        outer_to_tile: Dict[str, Tuple[str, nodes.AccessNode]],
        tile_shape: tuple,
        tile_subset: subsets.Range,
    ) -> Tuple[str, nodes.AccessNode]:
        """
        Create a TileOp library node for one branch of the if-else.

        Returns ``(result_tile_name, result_tile_node)``.
        """
        # Find the tasklet in the branch state
        tasklets = [n for n in branch_state.nodes()
                    if isinstance(n, nodes.Tasklet)]
        assert len(tasklets) == 1
        tasklet = tasklets[0]

        # Classify the tasklet (promote scalars to array-level)
        match = match_tasklet_to_tile_library_node(
            branch_state, tasklet, MaskType.UNMASKED, promote_scalars=True)
        assert match is not None
        node_info = match.node_info
        classification = match.tasklet_classification

        # Create the TileOp library node
        lib_node = TileOpLibraryNode(
            name=f"TileOp_{branch_tag}",
            op=classification.op,
            tile_shape=list(tile_shape),
            constant1=classification.constant1,
            constant2=classification.constant2,
        )
        graph.add_node(lib_node)

        # Map tasklet input connectors to NSDFG arrays
        tasklet_input_map = IfElseMapToTileWhere._get_tasklet_input_mapping(
            branch_state, tasklet)

        # Connect inputs from tile transients
        # Map classification rhs to library node connectors
        rhs_to_lib = {}
        if classification.rhs1 is not None and node_info.rhs1 is not None:
            rhs_to_lib[classification.rhs1] = node_info.rhs1
        if classification.rhs2 is not None and node_info.rhs2 is not None:
            rhs_to_lib[classification.rhs2] = node_info.rhs2

        connected_lib_conns = set()
        for tasklet_conn, lib_conn in rhs_to_lib.items():
            if tasklet_conn not in tasklet_input_map:
                continue
            nsdfg_array = tasklet_input_map[tasklet_conn]
            tile_name = nsdfg_to_tile.get(nsdfg_array)
            if tile_name is None:
                continue

            if lib_conn in connected_lib_conns:
                continue
            connected_lib_conns.add(lib_conn)

            # Remove constant-replaced connectors
            if (lib_conn == node_info.rhs1 and classification.constant1 is not None):
                if lib_conn in lib_node.in_connectors:
                    lib_node.remove_in_connector(lib_conn)
                continue
            if (lib_conn == node_info.rhs2 and classification.constant2 is not None):
                if lib_conn in lib_node.in_connectors:
                    lib_node.remove_in_connector(lib_conn)
                continue

            # Reuse the existing tile transient node from outer_to_tile
            # so graph stays scope-reachable.
            outer_name = nsdfg_to_outer[nsdfg_array]
            tile_node = outer_to_tile[outer_name][1]
            graph.add_edge(tile_node, None, lib_node, lib_conn,
                           Memlet(data=tile_name, subset=tile_subset))

        # Remove unused input connectors for constants/unary ops
        if classification.constant1 is not None and node_info.rhs1:
            if node_info.rhs1 in lib_node.in_connectors:
                lib_node.remove_in_connector(node_info.rhs1)
        if classification.constant2 is not None and node_info.rhs2:
            if node_info.rhs2 in lib_node.in_connectors:
                lib_node.remove_in_connector(node_info.rhs2)
        if node_info.rhs2 is None:
            # Unary op: remove any extra binary connector
            for conn in list(lib_node.in_connectors):
                if conn not in connected_lib_conns:
                    lib_node.remove_in_connector(conn)

        # Create output tile transient
        output_nsdfg_array = IfElseMapToTileWhere._get_branch_output_array(
            branch_state)
        outer_name = nsdfg_to_outer[output_nsdfg_array]
        result_name = sdfg._find_new_name(f"{branch_tag}_tile")
        sdfg.add_transient(
            result_name,
            shape=tile_shape,
            dtype=sdfg.arrays[outer_name].dtype,
            storage=sdfg.arrays[outer_name].storage,
            lifetime=dtypes.AllocationLifetime.Scope,
        )
        result_node = graph.add_access(result_name)

        graph.add_edge(lib_node, node_info.out, result_node, None,
                       Memlet(data=result_name, subset=tile_subset))

        return result_name, result_node
