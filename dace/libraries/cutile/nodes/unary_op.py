"""
TileUnaryOpLibraryNode for DaCe -> cuTile (TileIR) backend.

Represents element-wise unary operation on an array tile:
    C[subset] = OP(A[subset])          (array operand)
    C[subset] = OP(CONST)              (constant operand, no array connector)

The ``op`` property selects the operation:
  - Prefix operators: ``-``, ``+``
  - Function-style: ``abs``, ``sin``, ``cos``, ``exp``, ``sqrt``, ``log``
"""
from __future__ import annotations

import dace
from dace import dtypes, properties
from dace.sdfg import SDFG, SDFGState
from dace.sdfg import nodes
from dace.sdfg.nodes import LibraryNode
from dace import library
from dace.symbolic import symstr
from dace.transformation.transformation import ExpandTransformation
from dace.sdfg.validation import InvalidSDFGNodeError
from ..op_registry import register_op, MaskType, TaskletType


# ── Supported unary operations (add more entries to extend) ──────────
_UNARY_OPS = ["-", "abs", "sin", "cos", "exp", "sqrt", "log"]


def _unary_cpp_expr(op: str, operand: str) -> str:
    """Return a C++ expression for a unary operation."""
    if op in ("-", "+"):
        return f"({op}{operand})"
    return f"{op}({operand})"


@library.node
class TileUnaryOpLibraryNode(LibraryNode):
    """
    Generic library node for element-wise unary operations on a tile.

    C = op(A) or C = op(CONST)  where *op* is stored in the ``op`` property.

    Connectors (presence depends on whether constant replaces operand)
    ----------
    _a  (in, optional) : tile A — absent when constant is set
    _c  (out)          : tile C
    """

    implementations: dict = {}
    default_implementation = "pure"

    op = properties.Property(
        dtype=str,
        default="-",
        desc="Unary operation symbol, e.g. '-', 'abs', 'sin', 'exp', 'sqrt', 'log'.",
    )

    constant = properties.Property(
        dtype=str,
        default=None,
        desc="Constant operand. When set, no _a connector is needed.",
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

    def __init__(self, name: str = "TileUnaryOp", op: str = "-",
                 tile_shape: list[int] | None = None,
                 constant: str | None = None,
                 **kwargs):
        inputs: set[str] = set()
        if constant is None:
            inputs.add("_a")

        super().__init__(
            name,
            inputs=inputs,
            outputs={"_c"},
            **kwargs,
        )
        self.op = op
        self.tile_shape = tile_shape
        self.constant = constant

    def validate(self, sdfg: SDFG, state: SDFGState):
        if self.op not in _UNARY_OPS:
            raise InvalidSDFGNodeError(
                f"TileUnaryOp '{self.name}': unsupported op '{self.op}'. "
                f"Supported: {_UNARY_OPS}",
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
                f"TileUnaryOp '{self.name}': output connector _c must be connected.",
                sdfg=sdfg,
                state_id=state.parent_graph.node_id(state),
                node_id=state.node_id(self),
            )

        if self.constant is None:
            a_node = None
            for edge in state.in_edges(self):
                if edge.dst_conn == "_a":
                    a_node = edge.src
            if a_node is None:
                raise InvalidSDFGNodeError(
                    f"TileUnaryOp '{self.name}': connector _a must be connected when no constant is set.",
                    sdfg=sdfg,
                    state_id=state.parent_graph.node_id(state),
                    node_id=state.node_id(self),
                )


# ── C++ expansion ────────────────────────────────────────────────────

def _get_unary_tile_descriptors(node, state, sdfg):
    """Return (a_desc, c_desc) for a unary op node. a_desc may be None for constant ops."""
    a_desc = c_desc = None
    for edge in state.in_edges(node):
        if edge.dst_conn == "_a":
            a_desc = sdfg.arrays[edge.data.data]
    for edge in state.out_edges(node):
        if edge.src_conn == "_c":
            c_desc = sdfg.arrays[edge.data.data]
    if c_desc is None:
        raise ValueError(
            f"TileUnaryOp expansion: _c not connected for node '{node.name}'."
        )
    return a_desc, c_desc


@library.register_expansion(TileUnaryOpLibraryNode, "pure")
class ExpandTileUnaryOpPure(ExpandTransformation):
    """Expand TileUnaryOpLibraryNode into a C++ element-wise tasklet."""

    environments: list = []

    @staticmethod
    def expansion(node: TileUnaryOpLibraryNode, state: SDFGState, sdfg: SDFG) -> nodes.Tasklet:
        op = node.op
        constant = node.constant
        a_desc, c_desc = _get_unary_tile_descriptors(node, state, sdfg)

        ref_desc = a_desc or c_desc
        shape = tuple(node.tile_shape) if node.tile_shape is not None else ref_desc.shape
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

        if constant is not None:
            # Constant operand — fill tile with computed value
            expr = _unary_cpp_expr(op, constant)
            if use_scalar_form:
                code = f"_c = {expr};"
            else:
                shape_expr = ", ".join(symstr(s) for s in shape)
                c_strides_expr = ", ".join(symstr(s) for s in c_desc.strides)
                code = f"""
constexpr int ndim = {ndim};
const std::size_t shape[ndim] = {{{shape_expr}}};
const std::ptrdiff_t c_strides[ndim] = {{{c_strides_expr}}};
const auto _val = {expr};

std::size_t n = 1;
for (int d = 0; d < ndim; ++d) {{
    n *= shape[d];
}}
for (std::size_t i = 0; i < n; ++i) {{
    std::size_t rem = i;
    std::size_t ic = 0;
    for (int d = ndim - 1; d >= 0; --d) {{
        const auto extent = shape[d];
        const std::size_t coord = rem % extent;
        rem /= extent;
        ic += coord * c_strides[d];
    }}
    _c[ic] = _val;
}}
"""
        elif use_scalar_form:
            code = f"_c = {_unary_cpp_expr(op, '_a')};"
        else:
            shape_expr = ", ".join(symstr(s) for s in shape)
            a_strides_expr = ", ".join(symstr(s) for s in a_desc.strides)
            c_strides_expr = ", ".join(symstr(s) for s in c_desc.strides)

            code = f"""
constexpr int ndim = {ndim};
const std::size_t shape[ndim] = {{{shape_expr}}};
const std::ptrdiff_t a_strides[ndim] = {{{a_strides_expr}}};
const std::ptrdiff_t c_strides[ndim] = {{{c_strides_expr}}};

std::size_t n = 1;
for (int d = 0; d < ndim; ++d) {{
    n *= shape[d];
}}
for (std::size_t i = 0; i < n; ++i) {{
    std::size_t rem = i;
    std::size_t ia = 0;
    std::size_t ic = 0;
    for (int d = ndim - 1; d >= 0; --d) {{
        const auto extent = shape[d];
        const std::size_t coord = rem % extent;
        rem /= extent;
        ia += coord * a_strides[d];
        ic += coord * c_strides[d];
    }}
    _c[ic] = {_unary_cpp_expr(op, '_a[ia]')};
}}
"""
        return nodes.Tasklet(
            label=node.name + "_cutile",
            inputs=inputs,
            outputs={"_c"},
            code=code,
            language=dtypes.Language.CPP,
        )


# ── Register unary ops ──────────────────────────────────────────────

_UNARY_DISPLAY_NAMES = {
    "-": "TileNegate", "abs": "TileAbs", "sin": "TileSin",
    "cos": "TileCos", "exp": "TileExp", "sqrt": "TileSqrt", "log": "TileLog",
}

for _op in _UNARY_OPS:
    # Array operand
    register_op(
        op=_op,
        tasklet_type=TaskletType.UNARY_ARRAY,
        mask=MaskType.UNMASKED,
        node_type=TileUnaryOpLibraryNode,
        node_name=_UNARY_DISPLAY_NAMES.get(_op, f"TileUnaryOp_{_op}"),
        out="_c",
        rhs1="_a",
    )
    # Constant operand (UNARY_SYMBOL)
    register_op(
        op=_op,
        tasklet_type=TaskletType.UNARY_SYMBOL,
        mask=MaskType.UNMASKED,
        node_type=TileUnaryOpLibraryNode,
        node_name=_UNARY_DISPLAY_NAMES.get(_op, f"TileUnaryOp_{_op}") + "Const",
        out="_c",
    )
