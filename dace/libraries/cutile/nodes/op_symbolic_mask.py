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


from typing import List, Optional

import sympy as sp

import dace
from dace import dtypes, properties
from dace import library
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from ..op_registry import TaskletType, MaskType, register_op
from .base import (
    TileOpBase,
    expr_connectors,
    get_output_connector_name,
    op_cpp_expr, get_tile_descriptors, resolve_shape_and_scalar_form,
    build_stride_decls, resolve_operands, collect_array_descs,
    get_tile_strides,
    _BINARY_OPS, _UNARY_OPS,
)


def _sympy_condition_to_cpp(cond: sp.Basic) -> str:
    """Convert a SymPy boolean/relational expression to a C++ string.

    Args:
        cond: A SymPy boolean or relational expression.

    Returns:
        The equivalent C++ expression string.
    """
    return symstr(cond, cpp_mode=True)


@library.node
class TileSymbolicMaskedOpLibraryNode(TileOpBase):
    """
    Element-wise masked tile op using a symbolic (compile-time) predicate.

    Binary:  if cond: C = (constant1 or _a) op (constant2 or _b)
    Unary:   if cond: C = op(constant1 or _a)

    Connectors
    ----------
    _a  (in, optional)  : left / first operand tile
    _b  (in, optional)  : right / second operand tile (binary only)
    _out (out)          : result tile

    The ``mask_condition`` property holds a SymPy boolean expression that
    references ``sp.Symbol('__m0')``, ``sp.Symbol('__m1')``, … for
    per-dimension tile coordinates (0-based offsets within the bounding
    tile).  The expression is converted to C++ only at expansion time.
    When the condition evaluates to false, the output element is **not
    written**, preserving whatever value was already in the output memory.
    """

    implementations: dict = {}
    default_implementation = "pure"

    mask_condition = properties.Property(
        dtype=sp.Basic,
        default=None,
        allow_none=True,
        desc=(
            "SymPy boolean expression using __m0, __m1, … symbols as "
            "per-dimension tile coordinates (0-based).  None means "
            "'always true' (no masking).  Converted to C++ at expansion."
        ),
    )

    def __init__(self,
                 name: str = "TileSymbolicMaskedOp",
                 op: str = "+",
                 tile_shape: Optional[List[int]] = None,
                 constant1: Optional[str] = None,
                 constant2: Optional[str] = None,
                 expr=None,
                 out_connector: str = "_out",
                 mask_condition: Optional[sp.Basic] = None,
                 **kwargs) -> None:
        """Initialise the symbolic-masked element-wise tile operation node.

        Args:
            name: Node display name in the SDFG (default
                ``"TileSymbolicMaskedOp"``).
            op: Operation symbol or function name (e.g. ``"+"``, ``"abs"``).
            tile_shape: Fixed tile extents, or ``None`` to infer at expansion.
            constant1: Literal C++ value replacing the left/first operand.
                When ``None`` the ``_a`` input connector is created.
            constant2: Literal C++ value replacing the right/second operand.
                When ``None`` the ``_b`` input connector is created for binary
                ops.
            expr: Optional SymPy expression for multi-op mode.
            out_connector: Name of the single output connector (default
                ``"_out"``).
            mask_condition: SymPy boolean expression that uses ``__m0``,
                ``__m1``, … as per-dimension tile coordinates.  ``None``
                means *always true* (no masking).
            **kwargs: Forwarded to :class:`TileOpBase`.
        """
        super().__init__(name, op=op, tile_shape=tile_shape,
                         constant1=constant1, constant2=constant2,
                         expr=expr, out_connector=out_connector,
                         **kwargs)
        self.mask_condition = mask_condition

    def validate(self, sdfg: SDFG, state: SDFGState) -> None:
        """Validate the symbolic-masked op node before code generation.

        Args:
            sdfg: The SDFG containing this node.
            state: The state containing this node.

        Raises:
            :class:`dace.sdfg.validation.InvalidSDFGNodeError`: If any
                connector or operation constraint is violated.
        """
        self._validate_common(sdfg, state, "TileSymbolicMaskedOp")


# ── C++ expansion ────────────────────────────────────────────────────

@library.register_expansion(TileSymbolicMaskedOpLibraryNode, "pure")
class ExpandTileSymbolicMaskedOpPure(ExpandTransformation):
    """Expand TileSymbolicMaskedOpLibraryNode into a C++ tasklet with inlined condition."""

    environments: list = []

    @staticmethod
    def expansion(node: TileSymbolicMaskedOpLibraryNode, state: SDFGState,
                  sdfg: SDFG) -> nodes.Tasklet:
        """Expand the node into a C++ tasklet with an inlined symbolic condition.

        Converts ``mask_condition`` from a SymPy expression to a C++ boolean
        string and embeds it as a per-element guard in the generated loop.
        Elements that fail the condition are skipped (or restored from
        ``_c_in`` when that connector is present).

        Args:
            node: The :class:`TileSymbolicMaskedOpLibraryNode` to expand.
            state: The SDFG state containing *node*.
            sdfg: The SDFG owning the state.

        Returns:
            A :class:`dace.sdfg.nodes.Tasklet` implementing the
            symbolically-masked operation in C++.
        """
        out_conn = get_output_connector_name(node)
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

        a_desc, b_desc, c_desc, _, c_in_desc = get_tile_descriptors(node, state, sdfg)
        has_c_in = c_in_desc is not None

        if node.expr is not None:
            # ── Multi-op expression mode with symbolic mask ───────────────
            in_descs = {}
            for edge in state.in_edges(node):
                arr_name = edge.data.data
                if edge.dst_conn is None or arr_name is None:
                    continue
                in_descs[edge.dst_conn] = sdfg.arrays[arr_name]

            expr_inputs = expr_connectors(node.expr)
            missing = [c for c in expr_inputs if c not in in_descs]
            if missing:
                raise ValueError(
                    f"TileSymbolicMaskedOp expansion: missing expr input "
                    f"connector(s) {missing} for node '{node.name}'."
                )

            ref_desc = c_desc
            shape, ndim, use_scalar_form = resolve_shape_and_scalar_form(node, ref_desc)

            def _lv(conn: str) -> str:
                return conn.lstrip("_")

            val_subs = {sp.Symbol(conn): sp.Symbol(f"{_lv(conn)}_val")
                        for conn in expr_inputs}
            expr_cpp = symstr(node.expr.xreplace(val_subs), cpp_mode=True)

            inputs: set[str] = set(expr_inputs)
            if has_c_in:
                inputs.add("_c_in")

            if use_scalar_form:
                coord_decls = "\n".join(
                    f"    constexpr std::ptrdiff_t __m{d} = 0;"
                    for d in range(ndim)
                )
                val_reads = "".join(
                    f"    const auto {_lv(conn)}_val = {conn};\n"
                    for conn in expr_inputs
                )
                if has_c_in:
                    code = f"""\
{{
{coord_decls}
{val_reads}    if ({mask_condition}) {{
        {out_conn} = {expr_cpp};
    }} else {{
        {out_conn} = _c_in;
    }}
}}
"""
                else:
                    code = f"""\
{{
{coord_decls}
{val_reads}    if ({mask_condition}) {{
        {out_conn} = {expr_cpp};
    }}
}}
"""
            else:
                shape_expr = ", ".join(symstr(s) for s in shape)
                out_strides_expr = ", ".join(symstr(s) for s in get_tile_strides(c_desc, ndim))

                stride_decls = ""
                index_decls = ""
                index_updates = ""
                val_decls = ""
                for conn in expr_inputs:
                    desc = in_descs[conn]
                    key = _lv(conn)
                    stride_str = ", ".join(symstr(s) for s in get_tile_strides(desc, ndim))
                    stride_decls += (
                        f"const std::ptrdiff_t {key}_strides[ndim] = "
                        f"{{{stride_str}}};\n"
                    )
                    index_decls += f"    std::size_t i{key} = 0;\n"
                    index_updates += f"        i{key} += coord * {key}_strides[d];\n"
                    val_decls += f"    const auto {key}_val = {conn}[i{key}];\n"

                coord_aliases = "\n".join(
                    f"    const std::ptrdiff_t __m{d} = (std::ptrdiff_t)__coords[{d}];"
                    for d in range(ndim)
                )

                c_in_stride_decl = ""
                c_in_index_decl = ""
                c_in_index_update = ""
                c_in_else = ""
                if has_c_in and c_in_desc is not None:
                    c_in_strides = ", ".join(symstr(s) for s in get_tile_strides(c_in_desc, ndim))
                    c_in_stride_decl = f"const std::ptrdiff_t c_in_strides[ndim] = {{{c_in_strides}}};"
                    c_in_index_decl = "    std::size_t iin = 0;"
                    c_in_index_update = "        iin += coord * c_in_strides[d];"
                    c_in_else = f"else {{ {out_conn}[io] = _c_in[iin]; }}"

                code = f"""\
constexpr int ndim = {ndim};
const std::size_t shape[ndim] = {{{shape_expr}}};
{stride_decls}const std::ptrdiff_t out_strides[ndim] = {{{out_strides_expr}}};
{c_in_stride_decl}

std::size_t n = 1;
for (int d = 0; d < ndim; ++d) {{
    n *= shape[d];
}}

for (std::size_t i = 0; i < n; ++i) {{
    std::size_t rem = i;
{index_decls}    std::size_t io = 0;
{c_in_index_decl}
    std::size_t __coords[ndim];
    for (int d = ndim - 1; d >= 0; --d) {{
        const auto extent = shape[d];
        const std::size_t coord = rem % extent;
        rem /= extent;
{index_updates}        io += coord * out_strides[d];
{c_in_index_update}        __coords[d] = coord;
    }}
{coord_aliases}
    if ({mask_condition}) {{
{val_decls}        {out_conn}[io] = {expr_cpp};
    }}
    {c_in_else}
}}
"""

            return nodes.Tasklet(
                label=node.name + "_cutile",
                inputs=inputs,
                outputs={out_conn},
                code=code,
                language=dtypes.Language.CPP,
            )

        is_binary = (constant2 is not None) or (b_desc is not None)

        ref_desc = a_desc or b_desc or c_desc
        shape, ndim, use_scalar_form = resolve_shape_and_scalar_form(node, ref_desc)

        inputs: set[str] = set()
        if a_desc is not None:
            inputs.add("_a")
        if b_desc is not None:
            inputs.add("_b")
        if has_c_in:
            inputs.add("_c_in")

        # Determine operand values
        left_scalar, right_scalar, left_indexed, right_indexed = resolve_operands(
            constant1, constant2, is_binary)

        scalar_expr = op_cpp_expr(op, left_scalar, right_scalar)
        indexed_expr = op_cpp_expr(op, left_indexed, right_indexed)

        if use_scalar_form:
            # Declare coordinate variables at 0 for condition evaluation
            coord_decls = "\n".join(
                f"    constexpr std::ptrdiff_t __m{d} = 0;"
                for d in range(ndim)
            )
            if has_c_in:
                code = f"""\
{{
{coord_decls}
    if ({mask_condition}) {{
        {out_conn} = {scalar_expr};
    }} else {{
        {out_conn} = _c_in;
    }}
}}
"""
            else:
                code = f"""\
{{
{coord_decls}
    if ({mask_condition}) {{
        {out_conn} = {scalar_expr};
    }}
}}
"""
        else:
            shape_expr = ", ".join(symstr(s) for s in shape)
            c_strides_expr = ", ".join(symstr(s) for s in get_tile_strides(c_desc, ndim))

            # Collect array descriptors for stride computation
            array_descs = collect_array_descs(a_desc, b_desc)

            stride_decls, index_decls, index_updates = build_stride_decls(array_descs, ndim)

            # Named coordinate aliases for the mask condition
            coord_aliases = "\n".join(
                f"    const std::ptrdiff_t __m{d} = (std::ptrdiff_t)__coords[{d}];"
                for d in range(ndim)
            )

            c_in_stride_decl = ""
            c_in_index_decl = ""
            c_in_index_update = ""
            c_in_else = ""
            if has_c_in:
                c_in_stride_decl = f"const std::ptrdiff_t c_in_strides[ndim] = {{{', '.join(symstr(s) for s in get_tile_strides(c_in_desc, ndim))}}};"
                c_in_index_decl =   "    std::size_t iin = 0;"
                c_in_index_update = "        iin += coord * c_in_strides[d];"
                c_in_else =         f"else {{ {out_conn}[ic] = _c_in[iin]; }}"

            if not array_descs:
                # Both operands are constants
                code = f"""\
constexpr int ndim = {ndim};
const std::size_t shape[ndim] = {{{shape_expr}}};
const std::ptrdiff_t c_strides[ndim] = {{{c_strides_expr}}};
{c_in_stride_decl}
const auto _val = {indexed_expr};

std::size_t n = 1;
for (int d = 0; d < ndim; ++d) {{
    n *= shape[d];
}}
for (std::size_t i = 0; i < n; ++i) {{
    std::size_t rem = i;
    std::size_t ic = 0;
{c_in_index_decl}
    std::size_t __coords[ndim];
    for (int d = ndim - 1; d >= 0; --d) {{
        const auto extent = shape[d];
        const std::size_t coord = rem % extent;
        rem /= extent;
        __coords[d] = coord;
        ic += coord * c_strides[d];
{c_in_index_update}
    }}
{coord_aliases}
    if ({mask_condition}) {{
        {out_conn}[ic] = _val;
    }}
    {c_in_else}
}}
"""
            else:
                code = f"""\
constexpr int ndim = {ndim};
const std::size_t shape[ndim] = {{{shape_expr}}};
{stride_decls}const std::ptrdiff_t c_strides[ndim] = {{{c_strides_expr}}};
{c_in_stride_decl}

std::size_t n = 1;
for (int d = 0; d < ndim; ++d) {{
    n *= shape[d];
}}

for (std::size_t i = 0; i < n; ++i) {{
    std::size_t rem = i;
{index_decls}    std::size_t ic = 0;
{c_in_index_decl}
    std::size_t __coords[ndim];
    for (int d = ndim - 1; d >= 0; --d) {{
        const auto extent = shape[d];
        const std::size_t coord = rem % extent;
        rem /= extent;
{index_updates}        ic += coord * c_strides[d];
{c_in_index_update}        __coords[d] = coord;
    }}
{coord_aliases}
    if ({mask_condition}) {{
        {out_conn}[ic] = {indexed_expr};
    }}
    {c_in_else}
}}
"""

        return nodes.Tasklet(
            label=node.name + "_cutile",
            inputs=inputs,
            outputs={out_conn},
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
        out="_out",
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
        out="_out",
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
        out="_out",
    )

for _op in _UNARY_OPS:
    # Array operand symbolic-masked unary
    register_op(
        op=_op,
        tasklet_type=TaskletType.UNARY_ARRAY,
        mask=MaskType.SYMBOLIC,
        node_type=TileSymbolicMaskedOpLibraryNode,
        node_name=_SYM_UNARY_DISPLAY_NAMES.get(_op, f"TileSymMaskedUnaryOp_{_op}"),
        out="_out",
        rhs1="_a",
    )
    # Constant operand symbolic-masked unary
    register_op(
        op=_op,
        tasklet_type=TaskletType.UNARY_SYMBOL,
        mask=MaskType.SYMBOLIC,
        node_type=TileSymbolicMaskedOpLibraryNode,
        node_name=_SYM_UNARY_DISPLAY_NAMES.get(_op, f"TileSymMaskedUnaryOp_{_op}") + "Const",
        out="_out",
    )
