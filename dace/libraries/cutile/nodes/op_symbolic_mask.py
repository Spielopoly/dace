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
    op_cpp_expr, op_python_expression, get_tile_descriptors,
    resolve_shape_and_scalar_form,
    build_stride_decls_cpp, resolve_operands_cpp, resolve_operands_python,
    collect_array_descs,
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


def _sympy_condition_to_python(cond: sp.Basic) -> str:
    """Convert a SymPy boolean/relational expression to a Python string.

    Args:
        cond: A SymPy boolean or relational expression.

    Returns:
        The equivalent Python expression string.
    """
    return symstr(cond, cpp_mode=False)


def _build_coord_tile_code(shape: tuple, ndim: int, used_dims: set) -> str:
    """Generate Python code lines creating per-dimension coordinate tiles.

    For each dimension index *d* in *used_dims*, emits a line that assigns
    ``__m{d}`` to a tile whose values are the coordinate along that dimension,
    broadcast to the full tile *shape*.

    For 1-D tiles a simple ``ct.arange`` suffices; for N-D tiles the range is
    reshaped to have size 1 on every axis except *d*, then broadcast.

    Args:
        shape: Full tile shape tuple, e.g. ``(16,)`` or ``(8, 8)``.
        ndim: Number of dimensions (``len(shape)``).
        used_dims: Set of dimension indices referenced by the mask condition.

    Returns:
        A (possibly multi-line) Python code string defining ``__m{d}``
        variables for each dimension in *used_dims*.
    """
    lines = []
    for d in sorted(used_dims):
        if ndim == 1:
            lines.append(
                f"__m{d} = ct.arange({shape[d]}, dtype=ct.int32)"
            )
        else:
            # Reshape to (1, …, s_d, …, 1) then broadcast to full shape
            reshape_dims = tuple(shape[d] if i == d else 1 for i in range(ndim))
            reshape_str = ", ".join(str(x) for x in reshape_dims)
            shape_str = ", ".join(str(x) for x in shape)
            lines.append(
                f"__m{d} = ct.broadcast_to("
                f"ct.reshape(ct.arange({shape[d]}, dtype=ct.int32), "
                f"({reshape_str},)), ({shape_str},))"
            )
    return "\n".join(lines)


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
        left_scalar, right_scalar, left_indexed, right_indexed = resolve_operands_cpp(
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

            stride_decls, index_decls, index_updates = build_stride_decls_cpp(array_descs, ndim)

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

# ── cuTile Python expansion (``cutile_python``) ───────────────────────────

@library.register_expansion(TileSymbolicMaskedOpLibraryNode, "cutile_python")
class ExpandTileSymbolicMaskedOpCuTilePython(ExpandTransformation):
    """Expand TileSymbolicMaskedOpLibraryNode into a masked Python tasklet."""

    environments: list = []

    @staticmethod
    def expansion(
        node: TileSymbolicMaskedOpLibraryNode, state: SDFGState, sdfg: SDFG
    ) -> nodes.Tasklet:
        """Expand the node into a cuTile Python tasklet with a symbolic mask.

        Generates coordinate tiles via ``ct.arange`` for each dimension
        referenced in ``mask_condition``, evaluates the condition as a
        boolean tile, and uses ``ct.where`` to conditionally write results.

        Generated code pattern for an *n*-D tile::

            __m0 = ct.broadcast_to(ct.reshape(ct.arange(S0, dtype=ct.int32),
                                               (S0, 1, …)), (S0, S1, …))
            …
            __mask = (<condition string>)
            _out = ct.where(__mask, <operation>, _c_in)

        For a scalar tile (single element) coordinate variables are set to 0
        and the assignment uses a Python ternary expression.

        When ``mask_condition`` is ``None``, the operation is applied
        unconditionally and ``_c_in`` is not required.

        Args:
            node: The :class:`TileSymbolicMaskedOpLibraryNode` to expand.
            state: The SDFG state containing *node*.
            sdfg: The SDFG owning the state.

        Returns:
            A :class:`dace.sdfg.nodes.Tasklet` implementing the
            symbolically-masked operation in Python.

        Raises:
            ValueError: If ``_c_in`` is not connected and a mask condition is
                present.
        """
        out_conn = get_output_connector_name(node)
        a_desc, b_desc, c_desc, _, c_in_desc = get_tile_descriptors(
            node, state, sdfg)

        inputs: set[str] = set()
        if a_desc is not None:
            inputs.add("_a")
        if b_desc is not None:
            inputs.add("_b")

        # ── Build the operation expression ────────────────────────────────
        if node.expr is not None:
            expr_inputs = expr_connectors(node.expr)
            inputs.update(expr_inputs)
            inputs.discard(out_conn)
            base_expr = symstr(node.expr, cpp_mode=False)
        else:
            is_binary = (node.constant2 is not None) or (b_desc is not None)
            left, right, _, _ = resolve_operands_python(
                node.constant1, node.constant2, is_binary)
            base_expr = op_python_expression(node.op, left, right,
                                             ct_prefix=True)

        # ── Build the mask condition ──────────────────────────────────────
        mask_cond_sympy = node.mask_condition
        if mask_cond_sympy is None:
            # No mask — apply the operation unconditionally; _c_in is not needed
            code = f"{out_conn} = {base_expr}"
        else:
            if c_in_desc is None:
                raise ValueError(
                    "TileSymbolicMaskedOp cutile_python expansion requires "
                    "'_c_in' to preserve masked-off output lanes"
                )
            inputs.add("_c_in")

            ref_desc = a_desc or b_desc or c_desc
            shape, ndim, use_scalar_form = resolve_shape_and_scalar_form(
                node, ref_desc)
            mask_condition_py = _sympy_condition_to_python(mask_cond_sympy)

            if use_scalar_form:
                # Scalar tile: coordinates are 0, condition is a bool scalar
                coord_defs = "\n".join(
                    f"__m{d} = 0" for d in range(ndim)
                )
                lines = []
                if coord_defs:
                    lines.append(coord_defs)
                lines.append(f"__mask = ({mask_condition_py})")
                lines.append(
                    f"{out_conn} = {base_expr} if __mask else _c_in"
                )
                code = "\n".join(lines)
            else:
                # Determine which __m{d} symbols appear in the condition
                used_dims: set[int] = set()
                for sym in mask_cond_sympy.free_symbols:
                    name = str(sym)
                    if name.startswith("__m") and name[3:].isdigit():
                        d = int(name[3:])
                        if d >= ndim:
                            raise ValueError(
                                f"TileSymbolicMaskedOp '{node.name}': mask_condition "
                                f"references dimension __m{d} but tile has only {ndim} "
                                f"dimension(s) (shape={shape})"
                            )
                        used_dims.add(d)

                coord_code = _build_coord_tile_code(shape, ndim, used_dims)
                lines = []
                if coord_code:
                    lines.append(coord_code)
                lines.append(
                    f"{out_conn} = ct.where({mask_condition_py}, {base_expr}, _c_in)"
                )
                code = "\n".join(lines)

        return nodes.Tasklet(
            label=node.name + "_cutile_py",
            inputs=inputs,
            outputs={out_conn},
            code=code,
            language=dtypes.Language.Python,
        )



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
