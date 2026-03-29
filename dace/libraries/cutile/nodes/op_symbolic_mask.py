"""
Symbolic-masked cuTile operation library node – unified binary and unary.

Like :class:`TileRuntimeMaskedOpLibraryNode` but instead of receiving a
runtime boolean mask tile, the mask condition is a **SymPy boolean
expression** evaluated per element during expansion.  This avoids
allocating and filling a separate mask tile at runtime.

The ``mask_condition`` property holds a SymPy expression that may
reference ``__m0``, ``__m1``, … (SymPy Symbol objects) for per-dimension
tile coordinates.  Conversion to C++ happens only at expansion time.
When the condition is false for a given element, the output is not written.
"""
from __future__ import annotations

from typing import List, Optional

import sympy as sp

import dace
from dace import dtypes, properties
from dace import library
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg.nodes import LibraryNode
from dace.sdfg.validation import InvalidSDFGNodeError
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from ..op_registry import TaskletType, MaskType, register_op
from .op import _op_cpp_expr, _get_tile_descriptors, _BINARY_OPS, _UNARY_OPS, _ALL_OPS


def _sympy_condition_to_cpp(cond: sp.Basic) -> str:
    """Convert a SymPy boolean/relational expression to a C++ string."""
    return symstr(cond, cpp_mode=True)


@library.node
class TileSymbolicMaskedOpLibraryNode(LibraryNode):
    """
    Element-wise masked tile op using a symbolic (compile-time) predicate.

    Binary:  if cond: C = (constant1 or _a) op (constant2 or _b)
    Unary:   if cond: C = op(constant1 or _a)

    Connectors
    ----------
    _a  (in, optional)  : left / first operand tile
    _b  (in, optional)  : right / second operand tile (binary only)
    _c  (out)           : result tile

    The ``mask_condition`` property holds a SymPy boolean expression that
    references ``sp.Symbol('__m0')``, ``sp.Symbol('__m1')``, … for
    per-dimension tile coordinates (0-based offsets within the bounding
    tile).  The expression is converted to C++ only at expansion time.
    When the condition evaluates to false, the output element is **not
    written**, preserving whatever value was already in the output memory.
    """

    implementations: dict = {}
    default_implementation = "pure"

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

    mask_condition = properties.Property(
        default=None,
        allow_none=True,
        desc=(
            "SymPy boolean expression using __m0, __m1, … symbols as "
            "per-dimension tile coordinates (0-based).  None means "
            "'always true' (no masking).  Converted to C++ at expansion."
        ),
        to_json=lambda obj: sp.srepr(obj) if obj is not None else None,
        from_json=lambda s, **kw: sp.sympify(s) if s is not None else None,
    )

    def __init__(self,
                 name: str = "TileSymbolicMaskedOp",
                 op: str = "+",
                 tile_shape: Optional[List[int]] = None,
                 constant1: Optional[str] = None,
                 constant2: Optional[str] = None,
                 mask_condition: Optional[sp.Basic] = None,
                 **kwargs):
        inputs: set[str] = set()
        if constant1 is None:
            inputs.add("_a")
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
        self.mask_condition = mask_condition

    @property
    def is_binary(self) -> bool:
        return self.constant2 is not None or "_b" in self.in_connectors

    def validate(self, sdfg: SDFG, state: SDFGState):
        if self.op not in _ALL_OPS:
            raise InvalidSDFGNodeError(
                f"TileSymbolicMaskedOp '{self.name}': unsupported op '{self.op}'. "
                f"Supported: {_ALL_OPS}",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )

        if self.op not in _BINARY_OPS and self.is_binary:
            raise InvalidSDFGNodeError(
                f"TileSymbolicMaskedOp '{self.name}': op '{self.op}' is unary-only "
                f"but has binary connectors.",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )

        c_node = None
        for edge in state.out_edges(self):
            if edge.src_conn == "_c":
                c_node = edge.dst
        if c_node is None:
            raise InvalidSDFGNodeError(
                f"TileSymbolicMaskedOp '{self.name}': output connector _c must be connected.",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )

        if self.constant1 is None:
            a_node = None
            for edge in state.in_edges(self):
                if edge.dst_conn == "_a":
                    a_node = edge.src
            if a_node is None:
                raise InvalidSDFGNodeError(
                    f"TileSymbolicMaskedOp '{self.name}': connector _a must be connected "
                    f"when constant1 is not set.",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )

        if "_b" in self.in_connectors and self.constant2 is None:
            b_node = None
            for edge in state.in_edges(self):
                if edge.dst_conn == "_b":
                    b_node = edge.src
            if b_node is None:
                raise InvalidSDFGNodeError(
                    f"TileSymbolicMaskedOp '{self.name}': connector _b must be connected "
                    f"when constant2 is not set.",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )


# ── C++ expansion ────────────────────────────────────────────────────

@library.register_expansion(TileSymbolicMaskedOpLibraryNode, "pure")
class ExpandTileSymbolicMaskedOpPure(ExpandTransformation):
    """Expand TileSymbolicMaskedOpLibraryNode into a C++ tasklet with inlined condition."""

    environments: list = []

    @staticmethod
    def expansion(node: TileSymbolicMaskedOpLibraryNode, state: SDFGState,
                  sdfg: SDFG) -> nodes.Tasklet:
        op = node.op
        constant1 = node.constant1
        constant2 = node.constant2
        mask_cond_sympy = node.mask_condition
        # Convert SymPy condition to C++ string for code generation.
        mask_condition = (
            _sympy_condition_to_cpp(mask_cond_sympy)
            if mask_cond_sympy is not None
            else "true"
        )

        a_desc, b_desc, c_desc = _get_tile_descriptors(node, state, sdfg)
        is_binary = (constant2 is not None) or (b_desc is not None)

        ref_desc = a_desc or b_desc or c_desc
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

        inputs: set[str] = set()
        if a_desc is not None:
            inputs.add("_a")
        if b_desc is not None:
            inputs.add("_b")

        # Determine operand values
        left_scalar = constant1 if constant1 is not None else "_a"
        right_scalar = constant2 if constant2 is not None else ("_b" if is_binary else None)
        left_indexed = constant1 if constant1 is not None else "_a[ia]"
        right_indexed = constant2 if constant2 is not None else ("_b[ib]" if is_binary else None)

        scalar_expr = _op_cpp_expr(op, left_scalar, right_scalar)
        indexed_expr = _op_cpp_expr(op, left_indexed, right_indexed)

        if use_scalar_form:
            # Declare coordinate variables at 0 for condition evaluation
            coord_decls = "\n".join(
                f"    constexpr std::ptrdiff_t __m{d} = 0;"
                for d in range(ndim)
            )
            code = f"""\
{{
{coord_decls}
    if ({mask_condition}) {{
        _c = {scalar_expr};
    }}
}}
"""
        else:
            shape_expr = ", ".join(symstr(s) for s in shape)
            c_strides_expr = ", ".join(symstr(s) for s in c_desc.strides)

            # Collect array descriptors for stride computation
            array_descs: dict[str, dace.data.Data] = {}
            if a_desc is not None:
                array_descs["a"] = a_desc
            if b_desc is not None:
                array_descs["b"] = b_desc

            stride_decls = ""
            index_decls = ""
            index_updates = ""
            for key, desc in array_descs.items():
                stride_decls += f"const std::ptrdiff_t {key}_strides[ndim] = {{{', '.join(symstr(s) for s in desc.strides)}}};\n"
                index_decls +=  f"    std::size_t i{key} = 0;\n"
                index_updates += f"        i{key} += coord * {key}_strides[d];\n"

            # Named coordinate aliases for the mask condition
            coord_aliases = "\n".join(
                f"    const std::ptrdiff_t __m{d} = (std::ptrdiff_t)__coords[{d}];"
                for d in range(ndim)
            )

            if not array_descs:
                # Both operands are constants
                code = f"""\
constexpr int ndim = {ndim};
const std::size_t shape[ndim] = {{{shape_expr}}};
const std::ptrdiff_t c_strides[ndim] = {{{c_strides_expr}}};
const auto _val = {indexed_expr};

std::size_t n = 1;
for (int d = 0; d < ndim; ++d) {{
    n *= shape[d];
}}
for (std::size_t i = 0; i < n; ++i) {{
    std::size_t rem = i;
    std::size_t ic = 0;
    std::size_t __coords[ndim];
    for (int d = ndim - 1; d >= 0; --d) {{
        const auto extent = shape[d];
        const std::size_t coord = rem % extent;
        rem /= extent;
        __coords[d] = coord;
        ic += coord * c_strides[d];
    }}
{coord_aliases}
    if ({mask_condition}) {{
        _c[ic] = _val;
    }}
}}
"""
            else:
                code = f"""\
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
    std::size_t __coords[ndim];
    for (int d = ndim - 1; d >= 0; --d) {{
        const auto extent = shape[d];
        const std::size_t coord = rem % extent;
        rem /= extent;
{index_updates}        ic += coord * c_strides[d];
        __coords[d] = coord;
    }}
{coord_aliases}
    if ({mask_condition}) {{
        _c[ic] = {indexed_expr};
    }}
}}
"""

        return nodes.Tasklet(
            label=node.name + "_cutile",
            inputs=inputs,
            outputs={"_c"},
            code=code,
            language=dtypes.Language.CPP,
        )


# ── Register all symbolic-masked ops ────────────────────────────────

_SYM_BINARY_DISPLAY_NAMES = {
    "+": "TileSymMaskedAdd", "-": "TileSymMaskedSubtract",
    "*": "TileSymMaskedMultiply", "/": "TileSymMaskedDivide",
}
_SYM_CONST_DISPLAY_NAMES = {
    "+": "TileSymMaskedConstAdd", "-": "TileSymMaskedConstSubtract",
    "*": "TileSymMaskedConstMultiply", "/": "TileSymMaskedConstDivide",
}
_SYM_SYMBOL_DISPLAY_NAMES = {
    "+": "TileSymMaskedSymbolAdd", "-": "TileSymMaskedSymbolSubtract",
    "*": "TileSymMaskedSymbolMultiply", "/": "TileSymMaskedSymbolDivide",
}
_SYM_UNARY_DISPLAY_NAMES = {
    "-": "TileSymMaskedNegate", "abs": "TileSymMaskedAbs",
    "sin": "TileSymMaskedSin", "cos": "TileSymMaskedCos",
    "exp": "TileSymMaskedExp", "sqrt": "TileSymMaskedSqrt",
    "log": "TileSymMaskedLog",
}

for _op in _BINARY_OPS:
    # Two-array symbolic-masked binary
    register_op(
        op=_op,
        tasklet_type=TaskletType.ARRAY_ARRAY,
        mask=MaskType.SYMBOLIC,
        node_type=TileSymbolicMaskedOpLibraryNode,
        node_name=_SYM_BINARY_DISPLAY_NAMES.get(_op, f"TileSymMaskedOp_{_op}"),
        out="_c",
        rhs1="_a",
        rhs2="_b",
    )
    # Array + constant symbolic-masked binary
    register_op(
        op=_op,
        tasklet_type=TaskletType.ARRAY_SYMBOL,
        mask=MaskType.SYMBOLIC,
        node_type=TileSymbolicMaskedOpLibraryNode,
        node_name=_SYM_CONST_DISPLAY_NAMES.get(_op, f"TileSymMaskedConstOp_{_op}"),
        out="_c",
        rhs1="_a",
        rhs2="_b",
    )
    # Two constants symbolic-masked binary
    register_op(
        op=_op,
        tasklet_type=TaskletType.SYMBOL_SYMBOL,
        mask=MaskType.SYMBOLIC,
        node_type=TileSymbolicMaskedOpLibraryNode,
        node_name=_SYM_SYMBOL_DISPLAY_NAMES.get(_op, f"TileSymMaskedSymOp_{_op}"),
        out="_c",
    )

for _op in _UNARY_OPS:
    # Array operand symbolic-masked unary
    register_op(
        op=_op,
        tasklet_type=TaskletType.UNARY_ARRAY,
        mask=MaskType.SYMBOLIC,
        node_type=TileSymbolicMaskedOpLibraryNode,
        node_name=_SYM_UNARY_DISPLAY_NAMES.get(_op, f"TileSymMaskedUnaryOp_{_op}"),
        out="_c",
        rhs1="_a",
    )
    # Constant operand symbolic-masked unary
    register_op(
        op=_op,
        tasklet_type=TaskletType.UNARY_SYMBOL,
        mask=MaskType.SYMBOLIC,
        node_type=TileSymbolicMaskedOpLibraryNode,
        node_name=_SYM_UNARY_DISPLAY_NAMES.get(_op, f"TileSymMaskedUnaryOp_{_op}") + "Const",
        out="_c",
    )
