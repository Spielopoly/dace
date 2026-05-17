"""
Masked cuTile operation library node – unified binary and unary.

Implements element-wise masked operations:
    Binary:
        if mask[idx]: C[idx] = OP(A[idx], B[idx])
        if mask[idx]: C[idx] = OP(A[idx], CONST2)
        if mask[idx]: C[idx] = OP(CONST1, B[idx])
        if mask[idx]: C[idx] = OP(CONST1, CONST2)
    Unary:
        if mask[idx]: C[idx] = OP(A[idx])
        if mask[idx]: C[idx] = OP(CONST1)

When the mask is false, the output element is left untouched.
``constant1`` replaces the left/first operand, ``constant2`` the right/second.
"""


from typing import List, Optional

import dace
import sympy as sp
from dace import dtypes
from dace import library
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg.validation import InvalidSDFGNodeError
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
    _BINARY_OPS, _UNARY_OPS, SUPPORTED_MASK_DTYPES,
)


@library.node
class TileRuntimeMaskedOpLibraryNode(TileOpBase):
    """
    Unified library node for masked element-wise operations on tiles.

    Binary:  if M: C = (constant1 or _a) op (constant2 or _b); else: C unchanged
    Unary:   if M: C = op(constant1 or _a); else: C unchanged

    Connectors
    ----------
    _a     (in, optional)  : left / first operand tile
    _b     (in, optional)  : right / second operand tile (binary only)
    _m     (in)            : boolean mask tile
    _c_in  (in, optional)  : initial C values used when mask is false
    _out   (out)           : result tile
    """

    implementations: dict = {}
    default_implementation = "pure"

    def __init__(self,
                 name: str = "TileMaskedOp",
                 op: str = "+",
                 tile_shape: Optional[List[int]] = None,
                 constant1: Optional[str] = None,
                 constant2: Optional[str] = None,
                 expr=None,
                 out_connector: str = "_out",
                 **kwargs) -> None:
        """Initialise the runtime-masked element-wise tile operation node.

        The ``_m`` connector for the boolean mask tile is always added as an
        input.  An optional ``_c_in`` connector can be wired to supply the
        original output values for masked-out lanes.

        Args:
            name: Node display name in the SDFG (default ``"TileMaskedOp"``).
            op: Operation symbol or function name (e.g. ``"+"``, ``"abs"``).
            tile_shape: Fixed tile extents, or ``None`` to infer at expansion.
            constant1: Literal C++ value replacing the left/first operand.
                When ``None`` the ``_a`` input connector is created.
            constant2: Literal C++ value replacing the right/second operand.
                When ``None`` the ``_b`` input connector is created for binary ops.
            expr: Optional SymPy expression for multi-op mode.
            out_connector: Name of the single output connector (default
                ``"_out"``).
            **kwargs: Forwarded to :class:`TileOpBase`.
        """
        super().__init__(name, op=op, tile_shape=tile_shape,
                         constant1=constant1, constant2=constant2,
                         expr=expr, out_connector=out_connector,
                         extra_inputs={"_m"}, **kwargs)

    def validate(self, sdfg: SDFG, state: SDFGState) -> None:
        """Validate the runtime-masked op node before code generation.

        Extends the common validation check with mask-specific constraints:
        the ``_m`` connector must be connected, and the shapes of all present
        operand tiles must match the output tile.  The mask dtype must also be
        boolean or an integer type.

        Args:
            sdfg: The SDFG containing this node.
            state: The state containing this node.

        Raises:
            :class:`dace.sdfg.validation.InvalidSDFGNodeError`: If any
                connector, shape, or dtype constraint is violated.
        """
        self._validate_common(sdfg, state, "TileMaskedOp")
        out_conn = get_output_connector_name(self)

        # Additional mask-specific validation
        m_node = c_node = None
        a_node = b_node = None
        for edge in state.in_edges(self):
            if edge.dst_conn == "_m":
                m_node = edge.src
            elif edge.dst_conn == "_a":
                a_node = edge.src
            elif edge.dst_conn == "_b":
                b_node = edge.src
        for edge in state.out_edges(self):
            if edge.src_conn == out_conn:
                c_node = edge.dst

        if m_node is None:
            raise InvalidSDFGNodeError(
                f"TileMaskedOp '{self.name}': connector _m must be connected.",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )

        if c_node is None:
            out_conn = get_output_connector_name(self)
            raise InvalidSDFGNodeError(
                f"TileMaskedOp '{self.name}': output connector '{out_conn}' must be connected.",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )

        c_desc = sdfg.arrays[c_node.data]
        m_desc = sdfg.arrays[m_node.data]

        if a_node is not None:
            a_desc = sdfg.arrays[a_node.data]
            if a_desc.shape != c_desc.shape:
                raise InvalidSDFGNodeError(
                    f"TileMaskedOp '{self.name}': shape mismatch — A={a_desc.shape}, C={c_desc.shape}",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )
        if b_node is not None:
            b_desc = sdfg.arrays[b_node.data]
            if b_desc.shape != c_desc.shape:
                raise InvalidSDFGNodeError(
                    f"TileMaskedOp '{self.name}': shape mismatch — B={b_desc.shape}, C={c_desc.shape}",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )
        if m_desc.shape != c_desc.shape:
            raise InvalidSDFGNodeError(
                f"TileMaskedOp '{self.name}': mask shape mismatch — M={m_desc.shape}, C={c_desc.shape}",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )

        if m_desc.dtype not in SUPPORTED_MASK_DTYPES:
            raise InvalidSDFGNodeError(
                f"TileMaskedOp '{self.name}': mask dtype must be bool or integer, got M={m_desc.dtype}",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )


# ── C++ expansion ────────────────────────────────────────────────────

@library.register_expansion(TileRuntimeMaskedOpLibraryNode, "pure")
class ExpandTileRuntimeMaskedOpPure(ExpandTransformation):
    """Expand TileRuntimeMaskedOpLibraryNode into a C++ tasklet."""

    environments: list = []

    @staticmethod
    def expansion(node: TileRuntimeMaskedOpLibraryNode, state: SDFGState,
                  sdfg: SDFG) -> nodes.Tasklet:
        """Expand the node into a C++ masked element-wise tasklet.

        Generates a C++ loop that applies the configured operation to every
        element of the input tile(s) where the mask is non-zero.  When a
        ``_c_in`` is connected, masked-out lanes copy from ``_c_in``;
        otherwise they are left unmodified in the output.

        Args:
            node: The :class:`TileRuntimeMaskedOpLibraryNode` to expand.
            state: The SDFG state containing *node*.
            sdfg: The SDFG owning the state.

        Returns:
            A :class:`dace.sdfg.nodes.Tasklet` implementing the masked
            operation in C++.
        """
        out_conn = get_output_connector_name(node)

        if node.expr is not None:
            # ── Multi-op expression mode with runtime mask ────────────────
            in_descs = {}
            out_desc = None
            for edge in state.in_edges(node):
                arr_name = edge.data.data
                if edge.dst_conn is None or arr_name is None:
                    continue
                in_descs[edge.dst_conn] = sdfg.arrays[arr_name]
            for edge in state.out_edges(node):
                arr_name = edge.data.data
                if edge.src_conn == out_conn and arr_name is not None:
                    out_desc = sdfg.arrays[arr_name]
            if out_desc is None:
                raise ValueError(
                    f"TileMaskedOp expansion: {out_conn} not connected for "
                    f"node '{node.name}'."
                )
            if "_m" not in in_descs:
                raise ValueError(
                    f"TileMaskedOp expansion: _m must be connected for node "
                    f"'{node.name}'."
                )

            expr_inputs = expr_connectors(node.expr)
            missing = [c for c in expr_inputs if c not in in_descs]
            if missing:
                raise ValueError(
                    f"TileMaskedOp expansion: missing expr input connector(s) "
                    f"{missing} for node '{node.name}'."
                )

            shape, ndim, use_scalar_form = resolve_shape_and_scalar_form(node, out_desc)
            m_desc = in_descs["_m"]
            has_c_in = "_c_in" in in_descs
            c_in_desc = in_descs.get("_c_in")

            def _lv(conn: str) -> str:
                return conn.lstrip("_")

            val_subs = {sp.Symbol(conn): sp.Symbol(f"{_lv(conn)}_val")
                        for conn in expr_inputs}
            expr_cpp = symstr(node.expr.xreplace(val_subs), cpp_mode=True)

            inputs: set[str] = {"_m", *expr_inputs}
            if has_c_in:
                inputs.add("_c_in")

            if use_scalar_form:
                val_reads = "".join(
                    f"const auto {_lv(conn)}_val = {conn};\n" for conn in expr_inputs
                )
                if has_c_in:
                    code = (
                        f"if (_m) {{\n"
                        f"{val_reads}"
                        f"{out_conn} = {expr_cpp};\n"
                        f"}} else {{ {out_conn} = _c_in; }}"
                    )
                else:
                    code = (
                        f"if (_m) {{\n"
                        f"{val_reads}"
                        f"{out_conn} = {expr_cpp};\n"
                        f"}}"
                    )
            else:
                shape_expr = ", ".join(symstr(s) for s in shape)
                m_strides_expr = ", ".join(symstr(s) for s in get_tile_strides(m_desc, ndim))
                out_strides_expr = ", ".join(symstr(s) for s in get_tile_strides(out_desc, ndim))

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

                code = f"""
constexpr int ndim = {ndim};
const std::size_t shape[ndim] = {{{shape_expr}}};
{stride_decls}const std::ptrdiff_t m_strides[ndim] = {{{m_strides_expr}}};
const std::ptrdiff_t out_strides[ndim] = {{{out_strides_expr}}};
{c_in_stride_decl}

std::size_t n = 1;
for (int d = 0; d < ndim; ++d) {{
    n *= shape[d];
}}
for (std::size_t i = 0; i < n; ++i) {{
    std::size_t rem = i;
{index_decls}    std::size_t im = 0;
    std::size_t io = 0;
{c_in_index_decl}
    for (int d = ndim - 1; d >= 0; --d) {{
        const auto extent = shape[d];
        const std::size_t coord = rem % extent;
        rem /= extent;
{index_updates}        im += coord * m_strides[d];
        io += coord * out_strides[d];
{c_in_index_update}
    }}
    if (_m[im]) {{
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

        op = node.op
        constant1 = node.constant1
        constant2 = node.constant2

        a_desc, b_desc, c_desc, m_desc, c_in_desc = get_tile_descriptors(
            node, state, sdfg)
        if m_desc is None:
            raise ValueError(
                f"TileMaskedOp expansion: _m must be connected for node '{node.name}'."
            )
        has_c_in = c_in_desc is not None
        is_binary = (constant2 is not None) or (b_desc is not None)

        ref_desc = a_desc or b_desc or c_desc
        shape, ndim, use_scalar_form = resolve_shape_and_scalar_form(node, ref_desc)

        inputs: set[str] = {"_m"}
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
            if has_c_in:
                code = f"if (_m) {{ {out_conn} = {scalar_expr}; }} else {{ {out_conn} = _c_in; }}"
            else:
                code = f"if (_m) {{ {out_conn} = {scalar_expr}; }}"
        else:
            shape_expr = ", ".join(symstr(s) for s in shape)
            m_strides_expr = ", ".join(symstr(s) for s in get_tile_strides(m_desc, ndim))
            c_strides_expr = ", ".join(symstr(s) for s in get_tile_strides(c_desc, ndim))

            # Collect array descriptors for stride computation
            array_descs = collect_array_descs(a_desc, b_desc)

            stride_decls, index_decls, index_updates = build_stride_decls(array_descs, ndim)

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
                # Both operands are constants – only mask + output iteration
                code = f"""
constexpr int ndim = {ndim};
const std::size_t shape[ndim] = {{{shape_expr}}};
const std::ptrdiff_t m_strides[ndim] = {{{m_strides_expr}}};
const std::ptrdiff_t c_strides[ndim] = {{{c_strides_expr}}};
{c_in_stride_decl}
const auto _val = {indexed_expr};

std::size_t n = 1;
for (int d = 0; d < ndim; ++d) {{
    n *= shape[d];
}}
for (std::size_t i = 0; i < n; ++i) {{
    std::size_t rem = i;
    std::size_t im = 0;
    std::size_t ic = 0;
{c_in_index_decl}
    for (int d = ndim - 1; d >= 0; --d) {{
        const auto extent = shape[d];
        const std::size_t coord = rem % extent;
        rem /= extent;
        im += coord * m_strides[d];
        ic += coord * c_strides[d];
{c_in_index_update}
    }}
    if (_m[im]) {{
        {out_conn}[ic] = _val;
    }}
    {c_in_else}
}}
"""
            else:
                code = f"""
constexpr int ndim = {ndim};
const std::size_t shape[ndim] = {{{shape_expr}}};
{stride_decls}const std::ptrdiff_t m_strides[ndim] = {{{m_strides_expr}}};
const std::ptrdiff_t c_strides[ndim] = {{{c_strides_expr}}};
{c_in_stride_decl}

std::size_t n = 1;
for (int d = 0; d < ndim; ++d) {{
    n *= shape[d];
}}

for (std::size_t i = 0; i < n; ++i) {{
    std::size_t rem = i;
{index_decls}    std::size_t im = 0;
    std::size_t ic = 0;
{c_in_index_decl}
    for (int d = ndim - 1; d >= 0; --d) {{
        const auto extent = shape[d];
        const std::size_t coord = rem % extent;
        rem /= extent;
{index_updates}        im += coord * m_strides[d];
        ic += coord * c_strides[d];
{c_in_index_update}
    }}
    if (_m[im]) {{
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


# ── Register all masked ops ─────────────────────────────────────────

# ── cuTile Python expansion (``cutile_python``) ───────────────────────────

@library.register_expansion(TileRuntimeMaskedOpLibraryNode, "cutile_python")
class ExpandTileRuntimeMaskedOpCuTilePython(ExpandTransformation):
    """Expand TileRuntimeMaskedOpLibraryNode into a simple Python tasklet."""

    environments: list = []

    @staticmethod
    def expansion(
        node: TileRuntimeMaskedOpLibraryNode, state: SDFGState, sdfg: SDFG
    ) -> nodes.Tasklet:
        out_conn = get_output_connector_name(node)
        a_desc, b_desc, c_desc, m_desc, c_in_desc = get_tile_descriptors(node, state, sdfg)

        if c_in_desc is None:
            raise ValueError(
                "TileRuntimeMaskedOp cutile_python expansion requires '_c_in' "
                "to preserve masked-off output lanes"
            )

        inputs: set = {"_m"}
        if a_desc is not None:
            inputs.add("_a")
        if b_desc is not None:
            inputs.add("_b")
        if c_in_desc is not None:
            inputs.add("_c_in")

        fallback = "_c_in"

        if node.expr is not None:
            from .base import expr_connectors
            inputs.update(expr_connectors(node.expr))
            inputs.discard(out_conn)
            base_expr = str(node.expr)
        else:
            left = node.constant1 if node.constant1 is not None else "_a"
            if node.constant2 is None and b_desc is None:
                if node.op in ("-", "+"):
                    base_expr = f"({node.op}{left})"
                elif node.op == "abs":
                    base_expr = f"abs({left})"
                elif node.op in ("sin", "cos", "exp", "sqrt", "log", "ceil", "floor"):
                    base_expr = f"ct.{node.op}({left})"
                else:
                    base_expr = f"{node.op}({left})"
            else:
                right = node.constant2 if node.constant2 is not None else "_b"
                base_expr = f"({left} {node.op} {right})"

        code = f"{out_conn} = ct.where(_m, {base_expr}, {fallback})"

        return nodes.Tasklet(
            label=node.name + "_cutile_py",
            inputs=inputs,
            outputs={out_conn},
            code=code,
            language=dtypes.Language.Python,
        )



_MASKED_BINARY_DISPLAY_NAMES = {"+": "TileMaskedAdd", "-": "TileMaskedSubtract",
                                "*": "TileMaskedMultiply", "/": "TileMaskedDivide"}
_MASKED_CONST_DISPLAY_NAMES = {
    "+": "TileMaskedConstAdd", "-": "TileMaskedConstSubtract",
    "*": "TileMaskedConstMultiply", "/": "TileMaskedConstDivide",
}
_MASKED_UNARY_DISPLAY_NAMES = {
    "-": "TileMaskedNegate", "abs": "TileMaskedAbs", "sin": "TileMaskedSin",
    "cos": "TileMaskedCos", "exp": "TileMaskedExp", "sqrt": "TileMaskedSqrt",
    "log": "TileMaskedLog",
}

for _op in _BINARY_OPS:
    # Two-array masked binary
    register_op(
        op=_op,
        tasklet_type=TaskletType.ARRAY_ARRAY,
        mask=MaskType.RUNTIME,
        node_type=TileRuntimeMaskedOpLibraryNode,
        node_name=_MASKED_BINARY_DISPLAY_NAMES.get(_op, f"TileMaskedOp_{_op}"),
        out="_out",
        rhs1="_a",
        rhs2="_b",
        mask_in="_m",
        out_in="_c_in",
    )
    # Array + constant masked binary
    register_op(
        op=_op,
        tasklet_type=TaskletType.ARRAY_SYMBOL,
        mask=MaskType.RUNTIME,
        node_type=TileRuntimeMaskedOpLibraryNode,
        node_name=_MASKED_CONST_DISPLAY_NAMES.get(_op, f"TileMaskedConstOp_{_op}"),
        out="_out",
        rhs1="_a",
        rhs2="_b",
        mask_in="_m",
        out_in="_c_in",
    )

for _op in _UNARY_OPS:
    register_op(
        op=_op,
        tasklet_type=TaskletType.UNARY_ARRAY,
        mask=MaskType.RUNTIME,
        node_type=TileRuntimeMaskedOpLibraryNode,
        node_name=_MASKED_UNARY_DISPLAY_NAMES.get(_op, f"TileMaskedUnaryOp_{_op}"),
        out="_out",
        rhs1="_a",
        mask_in="_m",
        out_in="_c_in",
    )
