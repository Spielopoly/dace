"""
TileIfElseOpLibraryNode – compound library node for conditional
element-wise tile operations.

Encapsulates the pattern::

    C = where(A cond_op B_or_const, true_op(...), false_op(...))

The expansion SDFG composes four inner library nodes:

1. ``TileOpLibraryNode`` for the condition comparison
2. ``TileOpLibraryNode`` for the true branch
3. ``TileOpLibraryNode`` for the false branch
4. ``TileWhereSelectLibraryNode`` for the final selection
"""
from __future__ import annotations

from typing import List, Optional

import dace
from dace import library, properties
from dace.sdfg import SDFG, SDFGState, nodes
from dace.sdfg.nodes import LibraryNode
from dace.sdfg.validation import InvalidSDFGNodeError
from dace.transformation.transformation import ExpandTransformation

from .op import TileOpLibraryNode
from .where_select import TileWhereSelectLibraryNode
from ._base import _BINARY_OPS, _COMPARISON_OPS, _ALL_OPS


# Role name → (internal node label, connector name)
_ROLE_TO_NODE_CONN = {
    "cond_left": ("cmp", "_a"),
    "cond_right": ("cmp", "_b"),
    "true_rhs1": ("true_branch", "_a"),
    "true_rhs2": ("true_branch", "_b"),
    "false_rhs1": ("false_branch", "_a"),
    "false_rhs2": ("false_branch", "_b"),
}


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
    required = {"cond_left"}
    if node.cond_constant is None:
        required.add("cond_right")
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

        C = where(A cond_op B_or_const, true_op(...), false_op(...))

    The output connector is named ``_out`` (not ``_c``) to avoid
    name collisions with the ``_c`` connectors of expanded inner
    TileOp / TileWhereSelect tasklets inside the expansion SDFG.

    Connectors
    ----------
    _in0, _in1, …  (in)  : input tiles (count set at construction)
    _out            (out) : result tile
    """

    implementations: dict = {}
    default_implementation = "pure"

    cond_op = properties.Property(
        dtype=str, default=">",
        desc="Comparison operator for the condition.",
    )
    cond_constant = properties.Property(
        dtype=str, default=None, allow_none=True,
        desc="Constant for condition RHS. None means RHS comes from "
             "an input connector.",
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
                 cond_op=">", cond_constant=None,
                 true_op="+", true_constant1=None, true_constant2=None,
                 false_op="+", false_constant1=None, false_constant2=None,
                 tile_shape=None, input_roles=None, num_inputs=1, **kwargs):
        inputs = {f"_in{i}" for i in range(num_inputs)}
        super().__init__(name, inputs=inputs, outputs={"_out"}, **kwargs)
        self.cond_op = cond_op
        self.cond_constant = cond_constant
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

        # --- operator validity ---
        if self.cond_op not in _COMPARISON_OPS:
            raise InvalidSDFGNodeError(
                f"TileIfElseOp '{self.name}': cond_op '{self.cond_op}' "
                f"not in {_COMPARISON_OPS}",
                sdfg=sdfg, state_id=sid, node_id=nid,
            )
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

        # --- input_roles must cover all required roles ---
        all_assigned = set()
        for roles in (self.input_roles or {}).values():
            all_assigned.update(roles)
        missing = _required_roles(self) - all_assigned
        if missing:
            raise InvalidSDFGNodeError(
                f"TileIfElseOp '{self.name}': input_roles missing "
                f"required roles: {missing}",
                sdfg=sdfg, state_id=sid, node_id=nid,
            )


# ── Expansion ────────────────────────────────────────────────────────

@library.register_expansion(TileIfElseOpLibraryNode, "pure")
class ExpandTileIfElseOpPure(ExpandTransformation):
    """Expand into an SDFG with four inner library nodes
    (cmp, true_branch, false_branch, where)."""

    environments: list = []

    @staticmethod
    def expansion(node: TileIfElseOpLibraryNode, state: SDFGState,
                  sdfg: SDFG) -> SDFG:
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
        cmp_node = TileOpLibraryNode(
            "cmp", op=node.cond_op,
            tile_shape=node.tile_shape,
            constant2=node.cond_constant,
        )
        true_node = TileOpLibraryNode(
            "true_branch", op=node.true_op,
            tile_shape=node.tile_shape,
            constant1=node.true_constant1,
            constant2=node.true_constant2,
        )
        false_node = TileOpLibraryNode(
            "false_branch", op=node.false_op,
            tile_shape=node.tile_shape,
            constant1=node.false_constant1,
            constant2=node.false_constant2,
        )
        where_node = TileWhereSelectLibraryNode(
            "where", tile_shape=node.tile_shape,
        )

        inner_state.add_node(cmp_node)
        inner_state.add_node(true_node)
        inner_state.add_node(false_node)
        inner_state.add_node(where_node)

        # ── helper: wire _inX → sub-node connector via role ──────────
        def _wire_input(role, sub_node, sub_conn):
            if role not in role_to_input:
                return  # operand is a constant – no array connection
            in_conn = role_to_input[role]
            access = inner_state.add_read(in_conn)
            desc = inner_sdfg.arrays[in_conn]
            inner_state.add_edge(
                access, None, sub_node, sub_conn,
                dace.Memlet.from_array(in_conn, desc),
            )

        # Wire condition
        _wire_input("cond_left", cmp_node, "_a")
        _wire_input("cond_right", cmp_node, "_b")

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

        # cmp → cond_tile → where._cond
        inner_state.add_edge(
            cmp_node, "_c", cond_acc, None,
            dace.Memlet.from_array("cond_tile", cond_arr),
        )
        inner_state.add_edge(
            cond_acc, None, where_node, "_cond",
            dace.Memlet.from_array("cond_tile", cond_arr),
        )

        # true_branch → true_tile → where._x
        inner_state.add_edge(
            true_node, "_c", true_acc, None,
            dace.Memlet.from_array("true_tile", true_arr),
        )
        inner_state.add_edge(
            true_acc, None, where_node, "_x",
            dace.Memlet.from_array("true_tile", true_arr),
        )

        # false_branch → false_tile → where._y
        inner_state.add_edge(
            false_node, "_c", false_acc, None,
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
