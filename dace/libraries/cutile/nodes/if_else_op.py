"""
TileIfElseOpLibraryNode – compound library node for conditional
element-wise tile operations.

Encapsulates the pattern::

    C = where(condition(...), true_op(...), false_op(...))

The expansion SDFG composes four inner library nodes:

1. ``TileOpLibraryNode`` (with ``expr``) evaluating the SymPy condition
2. ``TileOpLibraryNode`` for the true branch
3. ``TileOpLibraryNode`` for the false branch
4. ``TileWhereSelectLibraryNode`` for the final selection
"""
from __future__ import annotations

from typing import List, Optional, cast

import sympy as sp

import dace
from dace import library, properties
from dace.sdfg import SDFG, SDFGState, nodes
from dace.sdfg.nodes import LibraryNode
from dace.sdfg.validation import InvalidSDFGNodeError
from dace.transformation.transformation import ExpandTransformation

from .op import TileOpLibraryNode
from .where_select import TileWhereSelectLibraryNode
from ._base import _BINARY_OPS, _COMPARISON_OPS, _ALL_OPS


def _get_if_else_descriptors(node, state, sdfg):
    """Return *(in_descs, out_desc)* for a TileIfElseOpLibraryNode.

    *in_descs* maps connector name (``_in0``, ``_in1``, …) to the
    corresponding array descriptor from the parent SDFG.
    *out_desc* is the descriptor for ``_out``.
    """
    in_descs = {}
    out_desc = None
    for edge in state.in_edges(node):
        arr_name = edge.data.data
        if arr_name is None:
            continue
        conn = edge.dst_conn
        if conn is not None:
            in_descs[conn] = sdfg.arrays[arr_name]
    for edge in state.out_edges(node):
        arr_name = edge.data.data
        if arr_name is None:
            continue
        if edge.src_conn == "_out":
            out_desc = sdfg.arrays[arr_name]
    if out_desc is None:
        raise ValueError(
            f"TileIfElseOp expansion: _out not connected for node "
            f"'{node.name}'."
        )
    return in_descs, out_desc


def _required_roles(node):
    """Return the set of roles required by *node*'s configuration."""
    required = set()
    # True branch
    if node.true_constant1 is None:
        required.add("true_rhs1")
    if (node.true_op in _BINARY_OPS or node.true_op in _COMPARISON_OPS):
        if node.true_constant2 is None:
            required.add("true_rhs2")
    # False branch
    if node.false_constant1 is None:
        required.add("false_rhs1")
    if (node.false_op in _BINARY_OPS or node.false_op in _COMPARISON_OPS):
        if node.false_constant2 is None:
            required.add("false_rhs2")
    return required


# ── Library node ─────────────────────────────────────────────────────

@library.node
class TileIfElseOpLibraryNode(LibraryNode):
    """
    Compound library node for conditional element-wise tile operations.

    ::

        C = where(condition(...), true_op(...), false_op(...))

    The output connector is named ``_out`` (not ``_c``) to avoid
    name collisions with the ``_out`` connectors of expanded inner
    TileOp / TileWhereSelect tasklets inside the expansion SDFG.

    Connectors
    ----------
    _in0, _in1, …  (in)  : input tiles (count set at construction)
    _out            (out) : result tile
    """

    implementations: dict = {}
    default_implementation = "pure"

    condition = properties.Property(
        dtype=sp.Basic, default=None, allow_none=True,
        desc=(
            "SymPy boolean expression for the if-condition. "
            "Its free symbols must match input connector names "
            "(e.g., _in0, _in1)."
        ),
    )
    true_op = properties.Property(
        dtype=str, default="+",
        desc="Operation for the true branch.",
    )
    true_constant1 = properties.Property(
        dtype=str, default=None, allow_none=True,
        desc="True branch left/first constant operand.",
    )
    true_constant2 = properties.Property(
        dtype=str, default=None, allow_none=True,
        desc="True branch right/second constant operand.",
    )
    false_op = properties.Property(
        dtype=str, default="+",
        desc="Operation for the false branch.",
    )
    false_constant1 = properties.Property(
        dtype=str, default=None, allow_none=True,
        desc="False branch left/first constant operand.",
    )
    false_constant2 = properties.Property(
        dtype=str, default=None, allow_none=True,
        desc="False branch right/second constant operand.",
    )
    tile_shape = properties.ListProperty(
        element_type=int, default=None, allow_none=True,
        desc="Tile dimensions. None means inferred at expansion time.",
    )
    input_roles = properties.DictProperty(
        key_type=str, value_type=list,
        allow_none=True, default=None,
        desc="Maps connector name to list of roles, e.g. "
             "{'_in0': ['cond_left', 'true_rhs1']}.",
    )

    def __init__(self, name="TileIfElseOp", *,
                 condition=None,
                 true_op="+", true_constant1=None, true_constant2=None,
                 false_op="+", false_constant1=None, false_constant2=None,
                 tile_shape=None, input_roles=None, num_inputs=1, **kwargs):
        inputs = {f"_in{i}" for i in range(num_inputs)}
        super().__init__(name, inputs=inputs, outputs={"_out"}, **kwargs)
        self.condition = condition
        self.true_op = true_op
        self.true_constant1 = true_constant1
        self.true_constant2 = true_constant2
        self.false_op = false_op
        self.false_constant1 = false_constant1
        self.false_constant2 = false_constant2
        self.tile_shape = tile_shape
        self.input_roles = input_roles if input_roles is not None else {}

    def validate(self, sdfg: SDFG, state: SDFGState):
        sid = state.parent_graph.node_id(state)
        nid = state.node_id(self)

        # --- condition validity ---
        if self.condition is None:
            raise InvalidSDFGNodeError(
                f"TileIfElseOp '{self.name}': condition must be set.",
                sdfg=sdfg, state_id=sid, node_id=nid,
            )

        cond_symbols = {str(sym) for sym in self.condition.free_symbols}
        bad_symbols = cond_symbols - set(self.in_connectors)
        if bad_symbols:
            raise InvalidSDFGNodeError(
                f"TileIfElseOp '{self.name}': condition symbol(s) "
                f"{bad_symbols} do not match input connectors "
                f"{set(self.in_connectors)}.",
                sdfg=sdfg, state_id=sid, node_id=nid,
            )

        # --- operator validity ---
        if self.true_op not in _ALL_OPS:
            raise InvalidSDFGNodeError(
                f"TileIfElseOp '{self.name}': true_op '{self.true_op}' "
                f"not in {_ALL_OPS}",
                sdfg=sdfg, state_id=sid, node_id=nid,
            )
        if self.false_op not in _ALL_OPS:
            raise InvalidSDFGNodeError(
                f"TileIfElseOp '{self.name}': false_op '{self.false_op}' "
                f"not in {_ALL_OPS}",
                sdfg=sdfg, state_id=sid, node_id=nid,
            )

        # --- all input connectors must be connected ---
        connected_ins = set()
        for edge in state.in_edges(self):
            if edge.dst_conn is not None:
                connected_ins.add(edge.dst_conn)
        for conn in self.in_connectors:
            if conn not in connected_ins:
                raise InvalidSDFGNodeError(
                    f"TileIfElseOp '{self.name}': input connector "
                    f"{conn} not connected.",
                    sdfg=sdfg, state_id=sid, node_id=nid,
                )

        # --- output _out must be connected ---
        connected_outs = set()
        for edge in state.out_edges(self):
            if edge.src_conn is not None:
                connected_outs.add(edge.src_conn)
        if "_out" not in connected_outs:
            raise InvalidSDFGNodeError(
                f"TileIfElseOp '{self.name}': output _out not connected.",
                sdfg=sdfg, state_id=sid, node_id=nid,
            )

        # --- input_roles keys must reference valid input connectors ---
        role_map = self.input_roles or {}
        invalid_role_keys = set(role_map.keys()) - set(self.in_connectors)
        if invalid_role_keys:
            raise InvalidSDFGNodeError(
                f"TileIfElseOp '{self.name}': input_roles contain unknown "
                f"connector(s): {invalid_role_keys}",
                sdfg=sdfg, state_id=sid, node_id=nid,
            )
        dangling_role_keys = set(role_map.keys()) - connected_ins
        if dangling_role_keys:
            raise InvalidSDFGNodeError(
                f"TileIfElseOp '{self.name}': input_roles reference "
                f"unconnected connector(s): {dangling_role_keys}",
                sdfg=sdfg, state_id=sid, node_id=nid,
            )

        # --- each semantic role must be assigned at most once ---
        role_owner: dict[str, str] = {}
        duplicate_roles: set[str] = set()
        for conn, roles in role_map.items():
            for role in roles:
                if role in role_owner and role_owner[role] != conn:
                    duplicate_roles.add(role)
                else:
                    role_owner[role] = conn
        if duplicate_roles:
            raise InvalidSDFGNodeError(
                f"TileIfElseOp '{self.name}': role(s) assigned more than "
                f"once: {duplicate_roles}",
                sdfg=sdfg, state_id=sid, node_id=nid,
            )

        # --- input_roles must cover all required roles ---
        all_assigned = set()
        for roles in role_map.values():
            all_assigned.update(roles)
        missing = _required_roles(self) - all_assigned
        if missing:
            raise InvalidSDFGNodeError(
                f"TileIfElseOp '{self.name}': input_roles missing "
                f"required roles: {missing}",
                sdfg=sdfg, state_id=sid, node_id=nid,
            )


# ── Expansion ────────────────────────────────────────────────────────

@library.register_expansion(TileIfElseOpLibraryNode, "pure")  # type: ignore[arg-type]
class ExpandTileIfElseOpPure(ExpandTransformation):
    """Expand into an SDFG with four inner library nodes
    (condition_eval, true_branch, false_branch, where)."""

    environments: list = []

    @staticmethod
    def expansion(node: TileIfElseOpLibraryNode, state: SDFGState,
                  sdfg: SDFG) -> SDFG:
        node = cast(TileIfElseOpLibraryNode, node)
        in_descs, out_desc = _get_if_else_descriptors(node, state, sdfg)

        # ── inner SDFG skeleton ──────────────────────────────────────
        inner_sdfg = SDFG(node.name + "_sdfg")

        # Non-transient arrays for each input connector
        for conn, desc in in_descs.items():
            inner_sdfg.add_array(
                conn, shape=desc.shape, dtype=desc.dtype,
                strides=desc.strides, storage=desc.storage,
            )

        # Non-transient output (named _out to avoid clashing with the
        # _c connector on the inner TileOp / TileWhereSelect tasklets)
        inner_sdfg.add_array(
            "_out", shape=out_desc.shape, dtype=out_desc.dtype,
            strides=out_desc.strides, storage=out_desc.storage,
        )

        # Transient intermediates
        ref_shape = out_desc.shape
        storage = out_desc.storage
        inner_sdfg.add_array(
            "cond_tile", shape=ref_shape, dtype=dace.bool,
            transient=True, storage=storage,
        )
        inner_sdfg.add_array(
            "true_tile", shape=ref_shape, dtype=out_desc.dtype,
            transient=True, storage=storage,
        )
        inner_sdfg.add_array(
            "false_tile", shape=ref_shape, dtype=out_desc.dtype,
            transient=True, storage=storage,
        )

        inner_state = inner_sdfg.add_state(node.name + "_state")

        # ── reverse-map: role → connector name ───────────────────────
        role_to_input: dict[str, str] = {}
        for conn, roles in (node.input_roles or {}).items():
            for role in roles:
                role_to_input[role] = conn

        # ── inner library nodes ──────────────────────────────────────
        true_node = TileOpLibraryNode(
            "true_branch", op=node.true_op,
            tile_shape=node.tile_shape,
            constant1=node.true_constant1,
            constant2=node.true_constant2,
            out_connector="__true_out",
        )
        false_node = TileOpLibraryNode(
            "false_branch", op=node.false_op,
            tile_shape=node.tile_shape,
            constant1=node.false_constant1,
            constant2=node.false_constant2,
            out_connector="__false_out",
        )
        where_node = TileWhereSelectLibraryNode(
            "where", tile_shape=node.tile_shape,
        )

        # Build condition node using TileOpLibraryNode with an expr.
        #
        # The inner SDFG arrays are named '_in0', '_in1', etc.  DaCe
        # forbids tasklet connector names from matching SDFG array names,
        # so we rename the condition symbols to '_ci0', '_ci1', etc.
        # ('ci' = condition input) and wire back to the original arrays.
        if node.condition is None:
            raise ValueError(
                f"TileIfElseOp expansion: condition is missing for node "
                f"'{node.name}'."
            )

        cond_syms = sorted(
            str(s) for s in node.condition.free_symbols if isinstance(s, sp.Symbol)
        )
        # Map: original symbol name → renamed connector name
        cond_conn_rename = {s: f"_ci{i}" for i, s in enumerate(cond_syms)}
        cond_expr_renamed = node.condition.xreplace(
            {sp.Symbol(s): sp.Symbol(cond_conn_rename[s]) for s in cond_syms}
        )

        cond_node = TileOpLibraryNode(
            "condition_eval",
            expr=cond_expr_renamed,
            tile_shape=node.tile_shape,
            out_connector="__cond_out",
        )

        inner_state.add_node(cond_node)
        inner_state.add_node(true_node)
        inner_state.add_node(false_node)
        inner_state.add_node(where_node)

        # ── helper: wire _inX → sub-node connector via role ──────────
        def _wire_input(role, sub_node, sub_conn):
            if role not in role_to_input:
                return  # operand is a constant – no array connection
            in_conn = role_to_input[role]
            if in_conn not in inner_sdfg.arrays:
                raise ValueError(
                    f"TileIfElseOp expansion: role '{role}' references "
                    f"unknown input connector '{in_conn}' for node "
                    f"'{node.name}'."
                )
            access = inner_state.add_read(in_conn)
            desc = inner_sdfg.arrays[in_conn]
            inner_state.add_edge(
                access, None, sub_node, sub_conn,
                dace.Memlet.from_array(in_conn, desc),
            )

        # Wire condition inputs: renamed connector ← original inner SDFG array
        for orig_sym, ci_conn in sorted(cond_conn_rename.items()):
            cond_acc = inner_state.add_read(orig_sym)
            desc = inner_sdfg.arrays[orig_sym]
            inner_state.add_edge(
                cond_acc,
                None,
                cond_node,
                ci_conn,
                dace.Memlet.from_array(orig_sym, desc),
            )

        # Wire true branch
        _wire_input("true_rhs1", true_node, "_a")
        _wire_input("true_rhs2", true_node, "_b")

        # Wire false branch
        _wire_input("false_rhs1", false_node, "_a")
        _wire_input("false_rhs2", false_node, "_b")

        # ── sub-node outputs → transients → where-select ─────────────
        # Use a single access node per transient so that a clear
        # dataflow path (write → read) is visible within the state.
        cond_acc = inner_state.add_access("cond_tile")
        true_acc = inner_state.add_access("true_tile")
        false_acc = inner_state.add_access("false_tile")

        cond_arr = inner_sdfg.arrays["cond_tile"]
        true_arr = inner_sdfg.arrays["true_tile"]
        false_arr = inner_sdfg.arrays["false_tile"]

        # condition_eval → cond_tile → where._cond
        inner_state.add_edge(
            cond_node, "__cond_out", cond_acc, None,
            dace.Memlet.from_array("cond_tile", cond_arr),
        )
        inner_state.add_edge(
            cond_acc, None, where_node, "_cond",
            dace.Memlet.from_array("cond_tile", cond_arr),
        )

        # true_branch → true_tile → where._x
        inner_state.add_edge(
            true_node, "__true_out", true_acc, None,
            dace.Memlet.from_array("true_tile", true_arr),
        )
        inner_state.add_edge(
            true_acc, None, where_node, "_x",
            dace.Memlet.from_array("true_tile", true_arr),
        )

        # false_branch → false_tile → where._y
        inner_state.add_edge(
            false_node, "__false_out", false_acc, None,
            dace.Memlet.from_array("false_tile", false_arr),
        )
        inner_state.add_edge(
            false_acc, None, where_node, "_y",
            dace.Memlet.from_array("false_tile", false_arr),
        )

        # ── where → output ───────────────────────────────────────────
        out_access = inner_state.add_write("_out")
        inner_state.add_edge(
            where_node, "_c", out_access, None,
            dace.Memlet.from_array("_out", inner_sdfg.arrays["_out"]),
        )

        return inner_sdfg
