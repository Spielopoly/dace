"""
Shared base class and helpers for cuTile element-wise operation library nodes.
"""
from __future__ import annotations

from typing import Collection, List, Optional

from dace import properties
from dace.properties import make_properties
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.nodes import LibraryNode
from dace.sdfg.validation import InvalidSDFGNodeError
from dace.symbolic import symstr


# ── Supported operations ─────────────────────────────────────────────
_BINARY_OPS = ["+", "-", "*", "/"]
_UNARY_OPS = ["-", "abs", "sin", "cos", "exp", "sqrt", "log"]
_ALL_OPS = sorted(set(_BINARY_OPS + _UNARY_OPS))


def _op_cpp_expr(op: str, left: str, right: str | None = None) -> str:
    """Return a C++ expression for a binary or unary operation."""
    if right is not None:
        return f"({left} {op} {right})"
    # Unary
    if op in ("-", "+"):
        return f"({op}{left})"
    return f"{op}({left})"


# ── Expansion helpers ────────────────────────────────────────────────

def _get_tile_descriptors(node, state, sdfg):
    """Return (a_desc, b_desc, c_desc, m_desc, c_in_desc).

    a_desc / b_desc / m_desc / c_in_desc may be None when the
    corresponding connector is absent or replaced by a constant.
    """
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
        if edge.src_conn == "_c":
            c_desc = sdfg.arrays[arr_name]
    if c_desc is None:
        raise ValueError(
            f"TileOp expansion: _c not connected for node '{node.name}'."
        )
    return a_desc, b_desc, c_desc, m_desc, c_in_desc


def _resolve_shape_and_scalar_form(node, ref_desc):
    """Return (shape, ndim, use_scalar_form) from the node's tile_shape or a reference descriptor."""
    tile_shape = node.tile_shape
    shape = tuple(tile_shape) if tile_shape is not None else ref_desc.shape
    ndim = len(shape)
    try:
        n_total = 1
        for s in shape:
            n_total *= int(s)
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


# ── Base library node ────────────────────────────────────────────────

@make_properties
class _TileOpBase(LibraryNode):
    """Abstract base for all cuTile element-wise operation library nodes.

    Provides the four shared properties (``op``, ``constant1``,
    ``constant2``, ``tile_shape``), common constructor logic,
    ``is_binary``, and a reusable ``_validate_common`` helper.

    Concrete subclasses **must** still be decorated with ``@library.node``
    and define their own ``implementations`` / ``default_implementation``.
    """

    op = properties.Property(
        dtype=str,
        default="+",
        desc="Operation symbol, e.g. '+', '-', '*', '/', 'abs', 'sin', …",
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

    def __init__(self, name: str, op: str = "+",
                 tile_shape: Optional[List[int]] = None,
                 constant1: Optional[str] = None,
                 constant2: Optional[str] = None,
                 extra_inputs: Collection[str] = frozenset(),
                 **kwargs):
        inputs: set[str] = set(extra_inputs)
        if constant1 is None:
            inputs.add("_a")
        # Binary ops get _b unless constant2 replaces the right operand.
        if op in _BINARY_OPS and constant2 is None:
            inputs.add("_b")

        super().__init__(
            name,
            inputs=inputs,
            outputs={"_c"},
            **kwargs,
        )
        self.op = op
        self.tile_shape = tile_shape
        self.constant1 = constant1
        self.constant2 = constant2

    # ------------------------------------------------------------------
    @property
    def is_binary(self) -> bool:
        return self.constant2 is not None or "_b" in self.in_connectors

    def _validate_common(self, sdfg: SDFG, state: SDFGState, label: str):
        """Run the validation checks shared by all tile-op variants.

        *label* is used in error messages (e.g. ``"TileOp"``,
        ``"TileMaskedOp"``).
        """
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

        # Find connected nodes in a single pass over edges
        in_nodes: dict[str, object] = {}
        for edge in state.in_edges(self):
            if edge.dst_conn is not None:
                in_nodes[edge.dst_conn] = edge.src
        out_nodes: dict[str, object] = {}
        for edge in state.out_edges(self):
            if edge.src_conn is not None:
                out_nodes[edge.src_conn] = edge.dst

        if "_c" not in out_nodes:
            raise InvalidSDFGNodeError(
                f"{label} '{self.name}': output connector _c must be connected.",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )

        if self.constant1 is None and "_a" not in in_nodes:
            raise InvalidSDFGNodeError(
                f"{label} '{self.name}': connector _a must be connected when constant1 is not set.",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )

        if "_b" in self.in_connectors and self.constant2 is None and "_b" not in in_nodes:
            raise InvalidSDFGNodeError(
                f"{label} '{self.name}': connector _b must be connected when constant2 is not set.",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )
