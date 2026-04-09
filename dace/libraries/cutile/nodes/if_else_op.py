"""
TileIfElseOpLibraryNode – compound library node for conditional
element-wise tile operations.

Encapsulates the pattern::

    C = where(condition(...), true_expr(...), false_expr(...))

The expansion SDFG composes four inner library nodes:

1. ``TileOpLibraryNode`` (with ``expr``) evaluating the SymPy condition
2. ``TileOpLibraryNode`` for the true branch expression
3. ``TileOpLibraryNode`` for the false branch expression
4. ``TileWhereSelectLibraryNode`` for the final selection
"""
from typing import Dict, Optional, Tuple, cast

import sympy as sp

import dace
from dace import library, properties
from dace.sdfg import SDFG, SDFGState, nodes
from dace.sdfg.validation import InvalidSDFGNodeError
from dace.transformation.transformation import ExpandTransformation

from .op import TileOpLibraryNode
from .where_select import TileWhereSelectLibraryNode
from .base import TileNodeBase, get_all_input_descs


# ── Library node ─────────────────────────────────────────────────────

@library.node
class TileIfElseOpLibraryNode(TileNodeBase):
    """
    Compound library node for conditional element-wise tile operations.

    ::

        C = where(condition(...), true_expr(...), false_expr(...))

    The three SymPy expressions (``condition``, ``true_expr``,
    ``false_expr``) define the behaviour.  Input connectors are
    derived automatically from the union of free symbols across all
    three expressions.

    Connectors
    ----------
    _in0, _in1, …  (in)  : input tiles (derived from expressions)
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
    true_expr = properties.Property(
        dtype=sp.Basic, default=None, allow_none=True,
        desc=(
            "SymPy expression for the true branch. "
            "Its free symbols must be a subset of input connectors."
        ),
    )
    false_expr = properties.Property(
        dtype=sp.Basic, default=None, allow_none=True,
        desc=(
            "SymPy expression for the false branch. "
            "Its free symbols must be a subset of input connectors."
        ),
    )

    def __init__(self, name: str = "TileIfElseOp", *,
                 condition: Optional[sp.Basic] = None,
                 true_expr: Optional[sp.Basic] = None,
                 false_expr: Optional[sp.Basic] = None,
                 tile_shape: Optional[list] = None,
                 num_inputs: Optional[int] = None,
                 **kwargs) -> None:
        """Initialise the compound conditional element-wise tile operation node.

        Input connectors are derived automatically from the union of free
        symbols across all three SymPy expressions unless *num_inputs* is
        provided to create connectors ``_in0`` … ``_in{num_inputs-1}``
        explicitly.

        Args:
            name: Node display name in the SDFG (default
                ``"TileIfElseOp"``).
            condition: SymPy boolean expression for the if-condition.
            true_expr: SymPy expression for the true branch output.
            false_expr: SymPy expression for the false branch output.
            tile_shape: Fixed tile extents, or ``None`` to infer at expansion.
            num_inputs: When provided, create exactly this many input
                connectors (``_in0``, …, ``_in{num_inputs-1}``) regardless
                of the expressions.
            **kwargs: Forwarded to :class:`TileNodeBase`.
        """
        if num_inputs is not None:
            inputs = {f"_in{i}" for i in range(num_inputs)}
        else:
            all_symbols: set[str] = set()
            for expr in (condition, true_expr, false_expr):
                if expr is not None:
                    all_symbols.update(
                        str(s) for s in expr.free_symbols
                        if isinstance(s, sp.Symbol)
                    )
            inputs = all_symbols
        super().__init__(name, inputs=inputs, outputs={"_out"}, **kwargs)
        self.condition = condition
        self.true_expr = true_expr
        self.false_expr = false_expr
        self.tile_shape = tile_shape

    def validate(self, sdfg: SDFG, state: SDFGState) -> None:
        """Validate the if-else compound node before code generation.

        Checks that all connectors are wired, that ``condition``,
        ``true_expr``, and ``false_expr`` are all set, and that every free
        symbol in each expression corresponds to an input connector.

        Args:
            sdfg: The SDFG containing this node.
            state: The state containing this node.

        Raises:
            :class:`~dace.sdfg.validation.InvalidSDFGNodeError`: If any
                expression is ``None`` or a symbol does not match a connector.
        """
        self._validate_connectors_connected(sdfg, state, "TileIfElseOp")

        sid = state.parent_graph.node_id(state)
        nid = state.node_id(self)

        for attr_name in ("condition", "true_expr", "false_expr"):
            if getattr(self, attr_name) is None:
                raise InvalidSDFGNodeError(
                    f"TileIfElseOp '{self.name}': {attr_name} must be set.",
                    sdfg=sdfg, state_id=sid, node_id=nid,
                )

        in_conns = set(self.in_connectors)
        for attr_name in ("condition", "true_expr", "false_expr"):
            expr = getattr(self, attr_name)
            expr_syms = {str(s) for s in expr.free_symbols
                         if isinstance(s, sp.Symbol)}
            bad = expr_syms - in_conns
            if bad:
                raise InvalidSDFGNodeError(
                    f"TileIfElseOp '{self.name}': {attr_name} symbol(s) "
                    f"{bad} do not match input connectors {in_conns}.",
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
        """Expand the node into a nested SDFG with four inner library nodes.

        Builds an inner SDFG that evaluates ``condition``, ``true_expr``, and
        ``false_expr`` on full tiles using three :class:`TileOpLibraryNode`
        instances and then selects the correct result per element with a
        :class:`TileWhereSelectLibraryNode`.

        Args:
            node: The :class:`TileIfElseOpLibraryNode` to expand.
            state: The SDFG state containing *node*.
            sdfg: The SDFG owning the state.

        Returns:
            An inner :class:`~dace.sdfg.SDFG` that implements the conditional
            operation as a composition of tile library nodes.
        """
        in_descs, out_desc = get_all_input_descs(node, state, sdfg)

        # ── inner SDFG skeleton ──────────────────────────────────────
        inner_sdfg = SDFG(node.name + "_sdfg")

        # Non-transient arrays for each input connector
        for conn, desc in in_descs.items():
            inner_sdfg.add_array(
                conn, shape=desc.shape, dtype=desc.dtype,
                strides=desc.strides, storage=desc.storage,
            )

        # Non-transient output
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

        # ── helper: rename expression symbols with a prefix ──────────
        def _rename_expr(
            expr: sp.Basic, prefix: str
        ) -> Tuple[sp.Basic, Dict[str, str]]:
            """Rename free symbols in *expr* to ``{prefix}{i}``.

            Args:
                expr: The SymPy expression whose free symbols to rename.
                prefix: String prefix for the generated connector names.

            Returns:
                A 2-tuple ``(renamed_expr, rename_map)`` where *rename_map*
                maps each original symbol name to its generated name.
            """
            syms = sorted(
                str(s) for s in expr.free_symbols
                if isinstance(s, sp.Symbol)
            )
            rename_map = {s: f"{prefix}{i}" for i, s in enumerate(syms)}
            renamed = expr.xreplace(
                {sp.Symbol(s): sp.Symbol(rename_map[s]) for s in syms}
            )
            return renamed, rename_map

        # ── build inner library nodes ────────────────────────────────
        cond_expr_renamed, cond_rename = _rename_expr(node.condition, "_ci")
        true_expr_renamed, true_rename = _rename_expr(node.true_expr, "_ti")
        false_expr_renamed, false_rename = _rename_expr(node.false_expr, "_fi")

        cond_node = TileOpLibraryNode(
            "condition_eval",
            expr=cond_expr_renamed,
            tile_shape=node.tile_shape,
            out_connector="__cond_out",
        )
        true_node = TileOpLibraryNode(
            "true_branch",
            expr=true_expr_renamed,
            tile_shape=node.tile_shape,
            out_connector="__true_out",
        )
        false_node = TileOpLibraryNode(
            "false_branch",
            expr=false_expr_renamed,
            tile_shape=node.tile_shape,
            out_connector="__false_out",
        )
        where_node = TileWhereSelectLibraryNode(
            "where", tile_shape=node.tile_shape,
        )

        inner_state.add_node(cond_node)
        inner_state.add_node(true_node)
        inner_state.add_node(false_node)
        inner_state.add_node(where_node)

        # ── helper: wire renamed connectors to inner SDFG arrays ─────
        def _wire_sub_node(
            sub_node: TileOpLibraryNode, rename_map: Dict[str, str]
        ) -> None:
            """Wire a sub-node's renamed input connectors to inner SDFG access nodes.

            For each ``(original_symbol, new_connector)`` pair in *rename_map*,
            adds a read access node for the original inner SDFG array and
            connects it to the sub-node's renamed connector.

            Args:
                sub_node: The inner library node to wire.
                rename_map: Mapping from original symbol name to the renamed
                    connector name used in *sub_node*.
            """
            for orig_sym, new_conn in sorted(rename_map.items()):
                acc = inner_state.add_read(orig_sym)
                desc = inner_sdfg.arrays[orig_sym]
                inner_state.add_edge(
                    acc, None, sub_node, new_conn,
                    dace.Memlet.from_array(orig_sym, desc),
                )

        _wire_sub_node(cond_node, cond_rename)
        _wire_sub_node(true_node, true_rename)
        _wire_sub_node(false_node, false_rename)

        # ── sub-node outputs → transients → where-select ─────────────
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
