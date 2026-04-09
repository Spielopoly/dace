"""
Shared base class and helpers for cuTile element-wise operation library nodes.
"""


import math
from typing import Collection, Dict, List, Optional, Set, Tuple

import dace
from dace.data.core import Data as DataDesc
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


def op_cpp_expr(op: str, left: str, right: Optional[str] = None) -> str:
    """Return a C++ expression string for a binary or unary operation.

    Args:
        op: Operation symbol or function name (e.g. ``"+"``, ``"abs"``).
        left: Left/first operand as a C++ expression string.
        right: Right/second operand string, or ``None`` for unary ops.

    Returns:
        A C++ expression string wrapping the operation.
    """
    if right is not None:
        return f"({left} {op} {right})"
    # Unary
    if op in ("-", "+"):
        return f"({op}{left})"
    return f"{op}({left})"


def get_output_connector_name(node: LibraryNode) -> str:
    """Return the single output connector name for *node*.

    Args:
        node: A cuTile library node with exactly one output connector.

    Returns:
        The name of the single output connector.

    Raises:
        ValueError: If the node has zero or more than one output connector.
    """
    if len(node.out_connectors) != 1:
        raise ValueError(
            f"TileOp expansion: expected exactly one output connector for "
            f"node '{node.name}', got {set(node.out_connectors)}."
        )
    return next(iter(node.out_connectors))


# ── Expansion helpers ────────────────────────────────────────────────

def get_tile_descriptors(
    node: LibraryNode,
    state: SDFGState,
    sdfg: SDFG,
) -> Tuple[Optional[DataDesc], Optional[DataDesc], DataDesc, Optional[DataDesc], Optional[DataDesc]]:
    """Return ``(a_desc, b_desc, c_desc, m_desc, c_in_desc)`` for *node*.

    Resolves array descriptors for the standard cuTile op connectors by
    walking the incoming and outgoing edges of *node*.

    Args:
        node: The cuTile library node to inspect.
        state: The SDFG state containing *node*.
        sdfg: The SDFG owning the state.

    Returns:
        A 5-tuple ``(a_desc, b_desc, c_desc, m_desc, c_in_desc)`` where
        ``a_desc``, ``b_desc``, ``m_desc``, and ``c_in_desc`` may be ``None``
        when the corresponding connector is absent or replaced by a constant.

    Raises:
        ValueError: If the output connector is not wired.
    """
    out_conn = get_output_connector_name(node)
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


def resolve_shape_and_scalar_form(
    node: LibraryNode,
    ref_desc: DataDesc,
) -> Tuple[tuple, int, bool]:
    """Return ``(shape, ndim, use_scalar_form)`` for *node*.

    Derives the tile shape from ``node.tile_shape`` when set, falling back
    to ``ref_desc.shape``.  A *scalar form* is used when the tile has zero
    or one total element (i.e. ``ndim == 0`` or product of shape is 1).

    Args:
        node: The cuTile library node whose ``tile_shape`` property to read.
        ref_desc: Fallback array descriptor used when ``node.tile_shape`` is
            ``None``.

    Returns:
        A 3-tuple ``(shape, ndim, use_scalar_form)`` where ``shape`` is a
        tuple of extents, ``ndim`` is its length, and ``use_scalar_form`` is
        ``True`` when the tile collapses to a single element.
    """
    tile_shape = node.tile_shape
    shape = tuple(tile_shape) if tile_shape is not None else ref_desc.shape
    ndim = len(shape)
    try:
        n_total = math.prod(int(s) for s in shape)
        use_scalar_form = (ndim == 0) or (n_total == 1)
    except (TypeError, ValueError):
        use_scalar_form = (ndim == 0)
    return shape, ndim, use_scalar_form


def get_tile_strides(desc: DataDesc, ndim: int):
    """Return the last *ndim* strides of *desc* (the tile-relevant ones)."""
    if ndim <= 0:
        return ()
    return desc.strides[-ndim:]


def build_stride_decls(array_descs: Dict[str, DataDesc], ndim: int) -> Tuple[str, str, str]:
    """Build C++ stride, index, and update declaration strings.

    Generates three code fragments used inside cuTile C++ expansion templates:
    stride constant arrays, index variable declarations, and per-dimension
    index increment lines.

    Args:
        array_descs: Mapping from short connector key (``"a"``, ``"b"``, …)
            to the corresponding array descriptor.
        ndim: Number of tile dimensions.  Only the last *ndim* strides of
            each descriptor are emitted.

    Returns:
        A 3-tuple ``(stride_decls, index_decls, index_updates)`` — each a
        multi-line C++ string ready for embedding in a code template.
    """
    stride_decls = ""
    index_decls = ""
    index_updates = ""
    for key, desc in array_descs.items():
        strides = get_tile_strides(desc, ndim)
        stride_decls += f"const std::ptrdiff_t {key}_strides[ndim] = {{{', '.join(symstr(s) for s in strides)}}};\n"
        index_decls += f"    std::size_t i{key} = 0;\n"
        index_updates += f"        i{key} += coord * {key}_strides[d];\n"
    return stride_decls, index_decls, index_updates


def resolve_operands(
    constant1: Optional[str],
    constant2: Optional[str],
    is_binary: bool,
) -> Tuple[str, Optional[str], str, Optional[str]]:
    """Return operand strings for scalar and indexed C++ code emitting.

    Each position may be a constant literal or a pointer/variable name.

    Args:
        constant1: Left/first constant literal, or ``None`` to use ``_a``.
        constant2: Right/second constant literal, or ``None`` to use ``_b``
            (binary) or nothing (unary).
        is_binary: Whether the operation takes two operands.

    Returns:
        A 4-tuple ``(left_scalar, right_scalar, left_indexed, right_indexed)``
        with C++ expression strings for both scalar and indexed access forms.
        ``right_scalar`` and ``right_indexed`` are ``None`` for unary ops.
    """
    left_scalar = constant1 if constant1 is not None else "_a"
    right_scalar = constant2 if constant2 is not None else ("_b" if is_binary else None)
    left_indexed = constant1 if constant1 is not None else "_a[ia]"
    right_indexed = constant2 if constant2 is not None else ("_b[ib]" if is_binary else None)
    return left_scalar, right_scalar, left_indexed, right_indexed


def collect_array_descs(
    a_desc: Optional[DataDesc],
    b_desc: Optional[DataDesc],
) -> Dict[str, DataDesc]:
    """Return a dict of non-``None`` operand descriptors keyed by short name.

    Args:
        a_desc: Descriptor for the ``_a`` (left/first) operand, or ``None``.
        b_desc: Descriptor for the ``_b`` (right/second) operand, or ``None``.

    Returns:
        A dict mapping ``"a"`` and/or ``"b"`` to their respective descriptors,
        omitting keys whose descriptor is ``None``.
    """
    descs: Dict[str, DataDesc] = {}
    if a_desc is not None:
        descs["a"] = a_desc
    if b_desc is not None:
        descs["b"] = b_desc
    return descs


def expr_connectors(expr: sp.Basic) -> List[str]:
    """Return a sorted list of connector names derived from *expr*'s free symbols.

    Every :class:`sympy.Symbol` in *expr*'s free symbols is treated as an
    array input connector (regardless of naming convention).

    Args:
        expr: A SymPy expression whose free symbols will be extracted.

    Returns:
        Sorted list of symbol name strings to use as connector names.
    """
    return sorted(str(s) for s in expr.free_symbols if isinstance(s, sp.Symbol))


def get_all_input_descs(
    node: LibraryNode,
    state: SDFGState,
    sdfg: SDFG,
) -> Tuple[Dict[str, DataDesc], DataDesc]:
    """Return ``(input_descs, c_desc)`` for *node*.

    Walks all edges of *node* to build a complete mapping of input connector
    names to their array descriptors and resolves the single output descriptor.

    Args:
        node: The cuTile library node to inspect.
        state: The SDFG state containing *node*.
        sdfg: The SDFG owning the state.

    Returns:
        A 2-tuple ``(input_descs, c_desc)`` where *input_descs* maps each
        input connector name to its array descriptor, and *c_desc* is the
        descriptor for the output connector.

    Raises:
        ValueError: If the output connector is not wired.
    """
    out_conn = get_output_connector_name(node)
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


def build_multi_op_code(
    expr: sp.Basic,
    input_descs: Dict[str, DataDesc],
    c_desc: DataDesc,
    tile_shape: Optional[List[int]],
    out_conn: str = "_out",
) -> str:
    """Generate C++ loop code for an arbitrary SymPy expression.

    Emits a C++ snippet that evaluates *expr* element-wise across a tile,
    reading from each connector in *input_descs* and writing to *out_conn*.
    Numeric literals in the expression (SymPy ``Number`` nodes) are emitted
    as C++ literals via ``symstr``.

    Args:
        expr: The SymPy expression to evaluate.  Its free symbols must match
            the keys of *input_descs*.
        input_descs: Mapping from connector name (e.g. ``"_a"``, ``"_in0"``)
            to the corresponding array descriptor.
        c_desc: Array descriptor for the output connector.
        tile_shape: Explicit tile extents, or ``None`` to use ``c_desc.shape``.
        out_conn: Name of the C++ output variable / connector (default
            ``"_out"``).

    Returns:
        A self-contained C++ code string suitable for use in a
        :class:`~dace.sdfg.nodes.Tasklet`.
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
    c_strides_expr = ", ".join(symstr(s) for s in get_tile_strides(c_desc, ndim))
    stride_decls = ""
    index_decls = ""
    index_updates = ""
    val_decls = ""
    for conn in connectors:
        desc = input_descs[conn]
        k = _lv(conn)
        strides_str = ", ".join(symstr(s) for s in get_tile_strides(desc, ndim))
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

def get_connected_sets(
    node: LibraryNode,
    state: SDFGState,
) -> Tuple[Set[str], Set[str]]:
    """Return the sets of connected input and output connector names for *node*.

    Args:
        node: The library node to inspect.
        state: The SDFG state containing *node*.

    Returns:
        A 2-tuple ``(connected_ins, connected_outs)`` of sets of connector
        name strings.
    """
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
class TileNodeBase(LibraryNode):
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

    def _validate_connectors_connected(self, sdfg: SDFG, state: SDFGState, label: str) -> None:
        """Check that every input and output connector is wired to an edge.

        Args:
            sdfg: The SDFG containing this node.
            state: The state containing this node.
            label: Human-readable node type name for error messages.

        Raises:
            :class:`dace.sdfg.validation.InvalidSDFGNodeError`: For any
                unconnected connector.
        """
        connected_ins, connected_outs = get_connected_sets(self, state)
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
class TileOpBase(TileNodeBase):
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
                 expr: Optional[sp.Basic] = None,
                 out_connector: str = "_out",
                 extra_inputs: Collection[str] = frozenset(),
                 **kwargs) -> None:
        """Initialise the base tile operation node.

        Derives input connectors automatically from *expr*'s free symbols
        (multi-op mode) or from the combination of *op* arity and the
        presence/absence of *constant1* / *constant2* (single-op mode).

        Args:
            name: Node display name in the SDFG.
            op: Operation symbol or function name (e.g. ``"+"``, ``"abs"``).
            tile_shape: Fixed tile extents, or ``None`` to infer at expansion.
            constant1: Literal C++ value replacing the left/first operand.
                When ``None`` the ``_a`` input connector is created.
            constant2: Literal C++ value replacing the right/second operand.
                When ``None`` the ``_b`` input connector is created for
                binary ops.
            expr: Optional SymPy expression for multi-op mode.  When set,
                all free symbols become input connectors and *op*, *constant1*,
                and *constant2* are ignored.
            out_connector: Name of the single output connector (default
                ``"_out"``).
            extra_inputs: Additional input connector names to add beyond those
                derived from *expr* / *op* / constants.
            **kwargs: Forwarded to :class:`~dace.sdfg.nodes.LibraryNode`.
        """
        inputs: Set[str] = set(extra_inputs)
        if expr is not None:
            # Multi-op mode: connectors derived from expression free symbols.
            inputs.update(expr_connectors(expr))
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
        """``True`` when the node has two operands (binary op).

        A node is considered binary when either ``constant2`` is set
        (constant right operand) or the ``_b`` input connector is present
        (array right operand).
        """
        return self.constant2 is not None or "_b" in self.in_connectors

    def _validate_common(self, sdfg: SDFG, state: SDFGState, label: str) -> None:
        """Run the validation checks shared by all tile-op variants.

        Args:
            sdfg: The SDFG containing this node.
            state: The state containing this node.
            label: Human-readable node type name used in error messages
                (e.g. ``"TileOp"``, ``"TileMaskedOp"``).

        Raises:
            :class:`~dace.sdfg.validation.InvalidSDFGNodeError`: If any
                validation constraint is violated.
        """
        out_conn = get_output_connector_name(self)

        if self.expr is not None:
            # Multi-op mode: validate expression symbols match connectors.
            connected_ins, connected_outs = get_connected_sets(self, state)

            if out_conn not in connected_outs:
                raise InvalidSDFGNodeError(
                    f"{label} '{self.name}': output connector {out_conn} must be connected.",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )

            expr_conns = set(expr_connectors(self.expr))
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
            connected_ins, connected_outs = get_connected_sets(self, state)

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
