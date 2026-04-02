"""
Shared base class and helpers for cuTile element-wise operation library nodes.
"""
from __future__ import annotations

import math
from typing import Collection, List, Optional

import dace
import sympy as sp
from dace import properties
from dace.properties import make_properties
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.nodes import LibraryNode
from dace.sdfg.validation import InvalidSDFGNodeError
from dace.symbolic import symstr
from dace.sdfg import tasklet_utils


# ── Supported operations ─────────────────────────────────────────────
_BINARY_OPS = set(tasklet_utils._SUPPORTED_OPS)
_UNARY_OPS = set(tasklet_utils._UNARY_SYMBOLS.values()) | (tasklet_utils._SUPPORTED - _BINARY_OPS)
_ALL_OPS = _BINARY_OPS | _UNARY_OPS

# Dtypes accepted for mask / condition tiles
SUPPORTED_MASK_DTYPES = {
    dace.bool,
    dace.int8, dace.uint8, dace.int16, dace.uint16,
    dace.int32, dace.uint32, dace.int64, dace.uint64,
}


def _op_cpp_expr(op: str, left: str, right: str | None = None) -> str:
    """Return a C++ expression for a binary or unary operation."""
    if right is not None:
        return f"({left} {op} {right})"
    # Unary
    if op in ("-", "+"):
        return f"({op}{left})"
    return f"{op}({left})"


def _get_output_connector_name(node) -> str:
    """Return the single output connector name for *node*.

    cuTile op nodes are defined with exactly one output connector.
    """
    if len(node.out_connectors) != 1:
        raise ValueError(
            f"TileOp expansion: expected exactly one output connector for "
            f"node '{node.name}', got {set(node.out_connectors)}."
        )
    return next(iter(node.out_connectors))


# ── Expansion helpers ────────────────────────────────────────────────

def _get_tile_descriptors(node, state, sdfg):
    """Return (a_desc, b_desc, c_desc, m_desc, c_in_desc).

    a_desc / b_desc / m_desc / c_in_desc may be None when the
    corresponding connector is absent or replaced by a constant.
    """
    out_conn = _get_output_connector_name(node)
    a_desc = b_desc = c_desc = m_desc = c_in_desc = None
    for edge in state.in_edges(node):
        arr_name = edge.data.data
        if arr_name is None:
            continue
        if edge.dst_conn == "_a":
            a_desc = sdfg.arrays[arr_name]
        elif edge.dst_conn == "_b":
            b_desc = sdfg.arrays[arr_name]
        elif edge.dst_conn == "_m":
            m_desc = sdfg.arrays[arr_name]
        elif edge.dst_conn == "_c_in":
            c_in_desc = sdfg.arrays[arr_name]
    for edge in state.out_edges(node):
        arr_name = edge.data.data
        if arr_name is None:
            continue
        if edge.src_conn == out_conn:
            c_desc = sdfg.arrays[arr_name]
    if c_desc is None:
        raise ValueError(
            f"TileOp expansion: {out_conn} not connected for node '{node.name}'."
        )
    return a_desc, b_desc, c_desc, m_desc, c_in_desc


def _resolve_shape_and_scalar_form(node, ref_desc):
    """Return (shape, ndim, use_scalar_form) from the node's tile_shape or a reference descriptor."""
    tile_shape = node.tile_shape
    shape = tuple(tile_shape) if tile_shape is not None else ref_desc.shape
    ndim = len(shape)
    try:
        n_total = math.prod(int(s) for s in shape)
        use_scalar_form = (ndim == 0) or (n_total == 1)
    except (TypeError, ValueError):
        use_scalar_form = (ndim == 0)
    return shape, ndim, use_scalar_form


def _build_stride_decls(array_descs: dict[str, object]):
    """Build C++ stride declarations, index variable declarations, and index update lines.

    Returns (stride_decls, index_decls, index_updates) as strings ready
    for embedding in a C++ code template.
    """
    stride_decls = ""
    index_decls = ""
    index_updates = ""
    for key, desc in array_descs.items():
        stride_decls += f"const std::ptrdiff_t {key}_strides[ndim] = {{{', '.join(symstr(s) for s in desc.strides)}}};\n"
        index_decls += f"    std::size_t i{key} = 0;\n"
        index_updates += f"        i{key} += coord * {key}_strides[d];\n"
    return stride_decls, index_decls, index_updates


def _resolve_operands(constant1, constant2, is_binary):
    """Return (left_scalar, right_scalar, left_indexed, right_indexed) for C++ codegen."""
    left_scalar = constant1 if constant1 is not None else "_a"
    right_scalar = constant2 if constant2 is not None else ("_b" if is_binary else None)
    left_indexed = constant1 if constant1 is not None else "_a[ia]"
    right_indexed = constant2 if constant2 is not None else ("_b[ib]" if is_binary else None)
    return left_scalar, right_scalar, left_indexed, right_indexed


def _collect_array_descs(a_desc, b_desc):
    """Return dict of non-None operand descriptors keyed by short name."""
    descs: dict[str, object] = {}
    if a_desc is not None:
        descs["a"] = a_desc
    if b_desc is not None:
        descs["b"] = b_desc
    return descs


def _expr_connectors(expr) -> list:
    """Return sorted list of connector names derived from *expr*'s SymPy symbols.

    Every :class:`sympy.Symbol` in *expr*'s free symbols is treated as an
    array input connector (regardless of naming convention).
    """
    return sorted(str(s) for s in expr.free_symbols if isinstance(s, sp.Symbol))


def _get_all_input_descs(node, state, sdfg):
    """Return ``(input_descs, c_desc)`` for *node*.

    *input_descs* maps each input connector name to its array descriptor.
    *c_desc* is the descriptor for the output connector.
    """
    out_conn = _get_output_connector_name(node)
    input_descs = {}
    c_desc = None
    for edge in state.in_edges(node):
        arr_name = edge.data.data
        if arr_name is None:
            continue
        if edge.dst_conn is not None:
            input_descs[edge.dst_conn] = sdfg.arrays[arr_name]
    for edge in state.out_edges(node):
        arr_name = edge.data.data
        if arr_name is None:
            continue
        if edge.src_conn == out_conn:
            c_desc = sdfg.arrays[arr_name]
    if c_desc is None:
        raise ValueError(
            f"TileOp expansion: {out_conn} not connected for node '{node.name}'."
        )
    return input_descs, c_desc


def _build_multi_op_code(expr, input_descs: dict, c_desc, tile_shape, out_conn: str = "_out") -> str:
    """Generate C++ loop code for an arbitrary SymPy *expr*.

    Each key in *input_descs* maps a connector name (e.g. ``_a``, ``_in0``)
    to the array descriptor for that connector.  Numeric literals in the
    expression (SymPy ``Number`` nodes) are emitted as C++ literals by
    ``symstr``.
    """
    connectors = sorted(input_descs.keys())
    shape = tuple(tile_shape) if tile_shape is not None else c_desc.shape
    ndim = len(shape)
    try:
        n_total = math.prod(int(s) for s in shape)
        use_scalar_form = (ndim == 0) or (n_total == 1)
    except (TypeError, ValueError):
        use_scalar_form = (ndim == 0)

    # Strip leading '_' from connector names for C++ local variable names.
    def _lv(conn: str) -> str:
        return conn.lstrip("_")

    # Replace each connector symbol with a _val variable so the generated
    # C++ expression is readable and avoids pointer-name confusion.
    val_subs = {sp.Symbol(conn): sp.Symbol(f"{_lv(conn)}_val") for conn in connectors}
    cpp_expr_str = symstr(expr.xreplace(val_subs), cpp_mode=True)

    if use_scalar_form:
        val_reads = "".join(
            f"const auto {_lv(conn)}_val = {conn};\n" for conn in connectors
        )
        return f"{val_reads}{out_conn} = {cpp_expr_str};"

    shape_expr = ", ".join(symstr(s) for s in shape)
    c_strides_expr = ", ".join(symstr(s) for s in c_desc.strides)
    stride_decls = ""
    index_decls = ""
    index_updates = ""
    val_decls = ""
    for conn in connectors:
        desc = input_descs[conn]
        k = _lv(conn)
        strides_str = ", ".join(symstr(s) for s in desc.strides)
        stride_decls += f"const std::ptrdiff_t {k}_strides[ndim] = {{{strides_str}}};\n"
        index_decls += f"    std::size_t i{k} = 0;\n"
        index_updates += f"        i{k} += coord * {k}_strides[d];\n"
        val_decls += f"    const auto {k}_val = {conn}[i{k}];\n"

    return f"""
constexpr int ndim = {ndim};
const std::size_t shape[ndim] = {{{shape_expr}}};
{stride_decls}const std::ptrdiff_t c_strides[ndim] = {{{c_strides_expr}}};

std::size_t n = 1;
for (int d = 0; d < ndim; ++d) {{
    n *= shape[d];
}}
for (std::size_t i = 0; i < n; ++i) {{
    std::size_t rem = i;
{index_decls}    std::size_t ic = 0;
    for (int d = ndim - 1; d >= 0; --d) {{
        const auto extent = shape[d];
        const std::size_t coord = rem % extent;
        rem /= extent;
{index_updates}        ic += coord * c_strides[d];
    }}
{val_decls}    {out_conn}[ic] = {cpp_expr_str};
}}
"""


# ── Connector utilities ──────────────────────────────────────────────

def _get_connected_sets(node, state) -> tuple[set[str], set[str]]:
    """Return the sets of connected input and output connector names for *node*."""
    connected_ins: set[str] = set()
    for edge in state.in_edges(node):
        if edge.dst_conn is not None:
            connected_ins.add(edge.dst_conn)
    connected_outs: set[str] = set()
    for edge in state.out_edges(node):
        if edge.src_conn is not None:
            connected_outs.add(edge.src_conn)
    return connected_ins, connected_outs


# ── Base library node ────────────────────────────────────────────────

@make_properties
class _TileNodeBase(LibraryNode):
    """Abstract base for cuTile library nodes that operate on tiles.

    Provides the ``tile_shape`` property and a reusable
    ``_validate_connectors_connected`` helper.
    """

    tile_shape = properties.ListProperty(
        element_type=int,
        default=None,
        desc=(
            "Tile dimensions, e.g. [128] for a 1-D tile or [32, 32] for 2-D. "
            "0-D (scalar) tiles can be represented with an empty list []. "
            "When None, the shape is inferred from the incoming array descriptors "
            "at expansion time."
        ),
        allow_none=True,
    )

    def _validate_connectors_connected(self, sdfg: SDFG, state: SDFGState, label: str):
        """Check that every input and output connector is wired to an edge.

        Raises :class:`InvalidSDFGNodeError` for any unconnected connector.
        """
        connected_ins, connected_outs = _get_connected_sets(self, state)
        for conn in self.in_connectors:
            if conn not in connected_ins:
                raise InvalidSDFGNodeError(
                    f"{label} '{self.name}': input connector '{conn}' must be connected.",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )
        for conn in self.out_connectors:
            if conn not in connected_outs:
                raise InvalidSDFGNodeError(
                    f"{label} '{self.name}': output connector '{conn}' must be connected.",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )


@make_properties
class _TileOpBase(_TileNodeBase):
    """Abstract base for all cuTile element-wise operation library nodes.

    Provides the shared properties (``op``, ``constant1``,
    ``constant2``), common constructor logic,
    ``is_binary``, and a reusable ``_validate_common`` helper.

    Inherits ``tile_shape`` and ``_validate_connectors_connected`` from
    :class:`_TileNodeBase`.

    Concrete subclasses **must** still be decorated with ``@library.node``
    and define their own ``implementations`` / ``default_implementation``.
    """

    op = properties.Property(
        dtype=str,
        default="+",
        desc="Operation symbol, e.g. '+', '-', '*', '/', 'abs', 'sin', …",
    )

    expr = properties.Property(
        dtype=sp.Basic,
        default=None,
        allow_none=True,
        desc=(
            "Optional arbitrary SymPy expression for multi-op mode. "
            "When set, every SymPy Symbol in the expression becomes an "
            "input connector; ``op``, ``constant1``, and ``constant2`` "
            "are ignored.  Example: "
            "``sp.Symbol('_a') * (sp.Symbol('_b') + sp.Symbol('_d'))``."
        ),
    )

    constant1 = properties.Property(
        dtype=str,
        default=None,
        desc="Left / first constant operand. None means operand comes from _a.",
        allow_none=True,
    )

    constant2 = properties.Property(
        dtype=str,
        default=None,
        desc="Right / second constant operand. None means operand comes from _b (binary) or absent (unary).",
        allow_none=True,
    )

    def __init__(self, name: str, op: str = "+",
                 tile_shape: Optional[List[int]] = None,
                 constant1: Optional[str] = None,
                 constant2: Optional[str] = None,
                 expr=None,
                 out_connector: str = "_out",
                 extra_inputs: Collection[str] = frozenset(),
                 **kwargs):
        inputs: set[str] = set(extra_inputs)
        if expr is not None:
            # Multi-op mode: connectors derived from expression free symbols.
            inputs.update(_expr_connectors(expr))
        else:
            if constant1 is None:
                inputs.add("_a")
            # Binary ops get _b unless constant2 replaces the right operand.
            if (op in _BINARY_OPS) and constant2 is None:
                inputs.add("_b")

        super().__init__(
            name,
            inputs=inputs,
            outputs={out_connector},
            **kwargs,
        )
        self.op = op
        self.tile_shape = tile_shape
        self.constant1 = constant1
        self.constant2 = constant2
        self.expr = expr

    # ------------------------------------------------------------------
    @property
    def is_binary(self) -> bool:
        return self.constant2 is not None or "_b" in self.in_connectors

    def _validate_common(self, sdfg: SDFG, state: SDFGState, label: str):
        """Run the validation checks shared by all tile-op variants.

        *label* is used in error messages (e.g. ``"TileOp"``,
        ``"TileMaskedOp"``).
        """
        out_conn = _get_output_connector_name(self)

        if self.expr is not None:
            # Multi-op mode: validate expression symbols match connectors.
            connected_ins, connected_outs = _get_connected_sets(self, state)

            if out_conn not in connected_outs:
                raise InvalidSDFGNodeError(
                    f"{label} '{self.name}': output connector {out_conn} must be connected.",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )

            expr_conns = set(_expr_connectors(self.expr))
            if out_conn in expr_conns:
                raise InvalidSDFGNodeError(
                    f"{label} '{self.name}': '{out_conn}' cannot be used as an "
                    f"expression input symbol because it is reserved for the "
                    f"output connector.",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )
            bad = expr_conns - set(self.in_connectors)
            if bad:
                raise InvalidSDFGNodeError(
                    f"{label} '{self.name}': expr symbol(s) {bad} do not "
                    f"match input connectors {set(self.in_connectors)}.",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )
            for conn in self.in_connectors:
                if conn not in connected_ins:
                    raise InvalidSDFGNodeError(
                        f"{label} '{self.name}': input connector '{conn}' "
                        f"must be connected in multi-op mode.",
                        sdfg=sdfg,
                        state_id=state.parent_graph.node_id(state),
                        node_id=state.node_id(self),
                    )
        else:
            # Classic single-op mode: selective connector checks.
            connected_ins, connected_outs = _get_connected_sets(self, state)

            if out_conn not in connected_outs:
                raise InvalidSDFGNodeError(
                    f"{label} '{self.name}': output connector {out_conn} must be connected.",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )

            if self.op not in _ALL_OPS:
                raise InvalidSDFGNodeError(
                    f"{label} '{self.name}': unsupported op '{self.op}'. "
                    f"Supported: {_ALL_OPS}",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )

            if self.op not in _BINARY_OPS and self.is_binary:
                raise InvalidSDFGNodeError(
                    f"{label} '{self.name}': op '{self.op}' is unary-only "
                    f"but has binary connectors.",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )

            if self.constant1 is None and "_a" not in connected_ins:
                raise InvalidSDFGNodeError(
                    f"{label} '{self.name}': connector _a must be connected when constant1 is not set.",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )

            if "_b" in self.in_connectors and self.constant2 is None and "_b" not in connected_ins:
                raise InvalidSDFGNodeError(
                    f"{label} '{self.name}': connector _b must be connected when constant2 is not set.",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )
