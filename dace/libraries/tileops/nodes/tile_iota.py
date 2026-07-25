# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileIota`` — fill an integer (or scalar-numeric) tile with a per-lane
expression in the lane placeholders ``__l0..__l{K-1}``.

Used by the K-dim emitter / NSDFG-body promoter to materialize the
integer index tiles that feed :class:`TileLoad` / :class:`TileStore`,
plus any other per-lane affine fill (constant arange, diagonal index,
strided base). The expression may also reference additional input tiles
/ arrays declared via ``extra_inputs`` — the lane body reads them by
their connector names so callers can compose ``_src``- or ``_idx``-style
lookups without inventing new lib nodes.

The pure expansion lowers to a CPP tasklet with a K-fold nested scalar
loop body ``_dst[offset] = <expr>``. The K-dim contract (per the user
directive: "K-dim path can only emit tile ops or python single-element")
is satisfied at the IR level — the emitter / promoter places a
``TileIota`` lib node, not a raw CPP tasklet.
"""
from typing import Dict, Optional, Sequence, Set, Tuple

import dace
from dace import library, properties
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

from .._pure_codegen import nested_loops, tile_offset
from .tile_binop import _replace_symbol_operand, _symbol_operand_free_symbols


@library.expansion
class ExpandTileIotaPure(ExpandTransformation):
    """CPP tasklet emitting ``_dst[off] = <expr>`` over the K-fold lane loop."""

    environments = []

    @staticmethod
    def expansion(node: "TileIota", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        """Return a CPP tasklet that fills the tile lane by lane.

        K=0 / W=1 single-lane postamble: DaCe collapses ``Register
        Array(shape=(1,))`` transients to plain scalars, so the standard
        ``_dst[__l0] = expr`` body (with ``__l0 = 0``) would index into
        ``int64_t _dst;`` — a compile error. Detect the all-ones width
        case and emit the body WITHOUT the indexing wrappers: substitute
        each lane var ``__l<p>`` with ``0`` directly in ``expr``, and
        rewrite trailing ``_idx[0]`` / ``_src[0]`` extra-input reads to
        the bare connector name (the connector is also a scalar in this
        case). The result is a single CPP statement
        ``_dst = <expr_substituted>;``.

        :param node: The ``TileIota`` lib node being expanded.
        :param parent_state: State that owns the lib node (unused).
        :param parent_sdfg: SDFG that owns ``parent_state`` (unused).
        :returns: A CPP tasklet replacing the lib node in place.
        """
        widths = list(node.widths)
        inputs = {c: None for c in node.extra_inputs}
        if all(w == 1 for w in widths):
            expr = node.expr
            # Substitute lane vars with 0 (single lane).
            for p in range(len(widths)):
                expr = expr.replace(f"__l{p}", "0")
            # ``_idx[0]`` / ``_src[0]`` indexing on a (1,)-collapsed
            # scalar connector becomes the bare connector name.
            for conn in node.extra_inputs:
                expr = expr.replace(f"{conn}[0]", conn)
            body = f"_dst = {expr};"
            return nodes.Tasklet(
                label=f"{node.label}_pure",
                inputs=inputs,
                outputs={"_dst": None},
                code=body,
                language=dace.dtypes.Language.CPP,
            )
        off = tile_offset(widths)
        body = f"_dst[{off}] = {node.expr};"
        code = nested_loops(widths, body)
        return nodes.Tasklet(
            label=f"{node.label}_pure",
            inputs=inputs,
            outputs={"_dst": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )


@library.expansion
class ExpandTileIotaCutile(ExpandTransformation):
    """``cuda.tile``-Python expansion of :class:`TileIota`.

    Emits per-lane index arrays via ``ct.arange()`` for each dimension,
    broadcast to the full tile shape for K>=2, then evaluates the
    node's ``expr`` as a Python expression over those lane arrays.
    Extra inputs (``_idx``, ``_src``, ...) are tile/array connectors that
    broadcast element-wise in the cuTile expression.
    """

    environments = []

    @staticmethod
    def expansion(node: "TileIota", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        """Return a Python tasklet that fills the tile via cuTile ops.

        :param node: The ``TileIota`` lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: A Python-language tasklet replacing the lib node.
        """
        widths = list(node.widths)
        K = len(widths)
        inputs = {c: None for c in node.extra_inputs}

        # Derive the lane-array dtype from the ``_dst`` descriptor so the
        # declared descriptor dtype and the runtime tile dtype agree (the
        # lane-id tiles are declared int64; ct.arange defaults would yield
        # int32 tiles). Falls back to int32 when the descriptor is unknown.
        ct_dtype = "ct.int32"
        for out_edge in parent_state.out_edges(node):
            if out_edge.src_conn != "_dst" or out_edge.data is None or out_edge.data.data is None:
                continue
            dst_desc = parent_sdfg.arrays.get(out_edge.data.data)
            if dst_desc is not None:
                ct_dtype = f"ct.{dst_desc.dtype.to_string()}"
            break

        # Degenerate single-lane case: all widths are 1.
        if all(w == 1 for w in widths):
            expr = node.expr
            # Safe: K is capped at 3 in __init__, so __l{p} names never overlap.
            for p in range(K):
                expr = expr.replace(f"__l{p}", "0")
            for conn in node.extra_inputs:
                expr = expr.replace(f"{conn}[0]", conn)
            body = f"_dst = {expr}"
            return nodes.Tasklet(
                label=f"{node.label}_cutile",
                inputs=inputs,
                outputs={"_dst": None},
                code=body,
                language=dace.dtypes.Language.Python,
            )

        shape_tuple = ", ".join(str(w) for w in widths)
        lines = []

        # Per-dim lane-index arrays, broadcast to full tile shape for K>=2.
        if K == 1:
            lines.append(f"__l0 = ct.arange({widths[0]}, dtype={ct_dtype})")
        else:
            for k in range(K):
                slc = ["None"] * K
                slc[k] = ":"
                slc_str = "[" + ", ".join(slc) + "]"
                lines.append(f"__l{k} = ct.broadcast_to("
                             f"ct.arange({widths[k]}, dtype={ct_dtype}){slc_str}, "
                             f"({shape_tuple}))")

        # The expression uses __l0..__l{K-1} which are now cuTile arrays.
        lines.append(f"_dst = {node.expr}")

        return nodes.Tasklet(
            label=f"{node.label}_cutile",
            inputs=inputs,
            outputs={"_dst": None},
            code="\n".join(lines),
            language=dace.dtypes.Language.Python,
        )


@library.node
class TileIota(nodes.LibraryNode):
    """Per-lane affine / indirect fill of an integer tile.

    ``_dst[l_0, ..., l_{K-1}] = expr(l_0, ..., l_{K-1}, [extra_inputs])``
    where lane indices are spelled ``__l0..__l{K-1}`` inside ``expr``.

    Use cases (the K-dim path):

    * Affine gather/scatter index tiles: ``expr = "i + __l0"`` for a
      contiguous gather along dim 0; ``expr = "i + 2 * __l0"`` for a
      strided gather; arbitrary affine in the lane vars otherwise.
    * Indirect gather/scatter with a per-lane scalar lookup:
      ``extra_inputs = ("_idx",)``, ``expr = "_idx[__l0]"``.
    * Multi-dim indirect: ``extra_inputs = ("_src",)``,
      ``expr = "_src[<flat_offset(__l<p>)>]"``.

    The K-fold nested loop is CPP inside the pure expansion. The IR
    level above is a tile op — ``EmitTileOps`` / ``PromoteNSDFGBodyToTiles``
    never emit a raw CPP tasklet for these fills.

    :cvar implementations: Per-target expansions; ``"pure"`` is the
        CPP-loop correctness fallback. ``"cutile"`` emits the
        :mod:`cuda.tile`-Python equivalent (opt-in; the cuTile pipeline
        stamps it before expansion).
    :cvar default_implementation: ``"pure"``.
    """

    implementations = {"pure": ExpandTileIotaPure, "cutile": ExpandTileIotaCutile}
    default_implementation = "pure"

    target_isa = properties.Property(
        dtype=str,
        allow_none=False,
        default="SCALAR",
        desc="CPU target ISA the Auto-dispatch lowers to for K==1 "
        "(SCALAR | AVX512 | AVX2 | ARM_SVE | ARM_NEON | CUTILE); K>=2 is pure. "
        "Stamped by the VectorizeCPUMultiDim orchestrator before expansion.",
    )

    widths = properties.ListProperty(
        element_type=int,
        default=[],
        desc="Per-dim tile widths, innermost-last.",
    )
    expr = properties.Property(
        dtype=str,
        default="",
        desc="Per-lane body expression assigned to ``_dst[<offset>]``. "
        "Uses ``__l0..__l{K-1}`` for the lane indices and any "
        "name from ``extra_inputs`` for extra-input reads.",
    )
    extra_inputs = properties.ListProperty(
        element_type=str,
        default=[],
        desc="Extra input connectors the expression may read from "
        "(typical: ``_idx`` for a tile-shape lookup, ``_src`` for an "
        "outer-array boundary view).",
    )

    def __init__(
            self,
            name: str,
            widths: Tuple[int, ...],
            expr: str,
            extra_inputs: Sequence[str] = (),
            location: Optional[str] = None,
    ):
        """Construct a ``TileIota`` node.

        :param name: Node label.
        :param widths: Per-dim tile widths, innermost-last; length in
            ``{1, 2, 3}``.
        :param expr: Per-lane body expression (see class docstring).
        :param extra_inputs: Names of extra input connectors readable
            by ``expr``.
        :param location: Optional DaCe node location override.
        :raises ValueError: On invalid ``widths`` length or empty
            ``expr``.
        """
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"TileIota: widths length {len(widths)} not in {{1, 2, 3}}")
        if not expr:
            raise ValueError("TileIota: expr is required (non-empty per-lane body)")
        inputs = set(extra_inputs)
        super().__init__(name, location=location, inputs=inputs, outputs={"_dst"})
        self.widths = list(widths)
        self.expr = expr
        self.extra_inputs = list(extra_inputs)

    @property
    def free_symbols(self) -> Set[str]:
        """Runtime symbols used by :attr:`expr`.

        Lane placeholders and extra-input connectors are local to the
        expanded tasklet and therefore are not SDFG symbols.
        """
        local_names = set(self.extra_inputs)
        local_names.update(f"__l{d}" for d in range(len(self.widths)))
        return (super().free_symbols | _symbol_operand_free_symbols(self.expr)) - local_names

    def replace_dict(self, repl: Dict[str, str]) -> None:
        """Replace runtime symbols in :attr:`expr`."""
        local_names = set(self.extra_inputs)
        local_names.update(f"__l{d}" for d in range(len(self.widths)))
        symbol_repl = {key: value for key, value in repl.items() if key not in local_names}
        self.expr = _replace_symbol_operand(self.expr, symbol_repl)

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        """Confirm ``_dst`` and every declared extra input are connected.

        :param sdfg: SDFG that owns ``state``.
        :param state: State that owns ``self``.
        :raises ValueError: If a required connector is unconnected.
        """
        in_e = {e.dst_conn: e for e in state.in_edges(self) if e.dst_conn is not None}
        out_e = {e.src_conn: e for e in state.out_edges(self) if e.src_conn is not None}
        if "_dst" not in out_e:
            raise ValueError(f"{self.label}: required output '_dst' not connected")
        for c in self.extra_inputs:
            if c not in in_e:
                raise ValueError(f"{self.label}: declared extra input {c!r} not connected")
