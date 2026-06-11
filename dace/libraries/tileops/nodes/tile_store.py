# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileStore`` — write a K-dim tile back into a global array.

Symmetric to :class:`TileLoad`; the pure expansion emits a CPP tasklet
that walks the K-fold nested index space.
"""
from typing import Optional, Tuple

import dace
from dace import library, properties
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

from .._pure_codegen import nested_loops, offset_via_strides, tile_offset
from .. import _isa_codegen
from ..environments import TileOpsScalar, TileOpsAVX512, TileOpsAVX2, TileOpsNeon, TileOpsSVE


@library.expansion
class ExpandTileStorePure(ExpandTransformation):
    """Correctness-only CPP tasklet copying ``_src`` into the tile region of ``_dst``."""

    environments = []

    @staticmethod
    def expansion(node: "TileStore", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        """Return a CPP tasklet that copies ``_src`` into the tile
        region of the destination, optionally gated by ``_mask``.

        Destination offsets use the destination array's per-dim strides
        (read from the connector descriptor at expansion time) scaled
        by an optional :attr:`dim_strides` coefficient (defaulting to 1).

        :param node: The ``TileStore`` lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: A CPP tasklet replacing the lib node in place.
        """
        from dace.symbolic import symstr
        widths = list(node.widths)
        K = len(widths)
        dst_edge = next(e for e in parent_state.out_edges(node) if e.src_conn == "_dst")
        dst_arr = parent_sdfg.arrays[dst_edge.data.data]
        ndim = len(dst_arr.strides)
        # Step along the array dim each tile dim maps to (``dst_dims``);
        # default to the last K dims in order (a plain row-major tile).
        dims = list(node.dst_dims) if node.dst_dims else list(range(ndim - K, ndim))
        dst_strides = [symstr(dst_arr.strides[d]) for d in dims]
        coeff = list(node.dim_strides) if node.dim_strides else [1] * K
        dst_off = offset_via_strides(coeff, dst_strides)
        src_off = tile_offset(widths)
        # Resolve the per-lane source reference for each ``src_kind``:
        #   * ``Tile`` — the existing per-lane tile read.
        #   * ``Symbol`` — the literal / expression broadcast to every lane,
        #     cast to the destination dtype so a typed store resolves.
        #   * ``Scalar`` — a length-1 array read, broadcast to every lane.
        out_dtype = dst_arr.dtype.ctype
        if node.src_kind == "Symbol":
            src_ref = f"({out_dtype})({node.src_expr})"
        elif node.src_kind == "Scalar":
            src_ref = f"({out_dtype})(_src[0])"
        else:
            src_ref = f"_src[{src_off}]"
        if node.has_mask:
            body = f"if (_mask[{src_off}]) {{ _dst[{dst_off}] = {src_ref}; }}"
        else:
            body = f"_dst[{dst_off}] = {src_ref};"
        code = nested_loops(widths, body)
        inputs = (set() if node.src_kind == "Symbol" else {"_src"}) | ({"_mask"} if node.has_mask else set())
        return nodes.Tasklet(
            label=f"{node.label}_pure",
            inputs={c: None
                    for c in inputs},
            outputs={"_dst": None},
            code=code,
            language=dace.dtypes.Language.CPP,
        )


@library.expansion
class ExpandTileStoreCutile(ExpandTransformation):
    """TODO: docstring
    """

    environments = []

    @staticmethod
    def expansion(node: "TileStore", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> nodes.Tasklet:
        """Return a Python tasklet emitting ``ct.store`` / ``ct.scatter`` /
        a broadcast fill assignment.

        :param node: The lib node being expanded.
        :param parent_state: State that owns the lib node.
        :param parent_sdfg: SDFG that owns ``parent_state``.
        :returns: A Python-language tasklet replacing the lib node.
        """
        from dace.symbolic import symstr

        widths = tuple(node.widths)
        K = len(widths)

        dst_edge = next(e for e in parent_state.out_edges(node) if e.src_conn == "_dst")
        dst_arr = parent_sdfg.arrays[dst_edge.data.data]

        # Resolve the stored tile expression per src_kind (mirrors TileLoad).
        if node.src_kind == "Scalar":
            # Broadcast a single value. If the source comes from a global
            # length-1 array, load it first; otherwise the connector already
            # carries the scalar value.
            src_edge = next(e for e in parent_state.in_edges(node) if e.dst_conn == "_src")
            desc = parent_sdfg.arrays[src_edge.data.data]
            is_len1_array = (isinstance(desc, dace.data.Array)
                             and all(bool(dace.symbolic.simplify(s == 1)) for s in desc.shape))
            if is_len1_array:
                ref = f"ct.load(_src, index=({'0,' * len(desc.shape)}), shape=({'1,' * len(desc.shape)})).item()"
            else:
                ref = "_src.item()"
            tile_expr = f"ct.broadcast_to({ref}, {widths})"
        elif node.src_kind == "Symbol":
            tile_expr = f"ct.broadcast_to(({symstr(node.src_expr, cpp_mode=False)}), {widths})"
        elif node.src_kind == "Tile":
            tile_expr = "_src"
        else:
            raise ValueError(f"TileStore cutile expansion: unrecognized src_kind {node.src_kind!r}")

        inputs = (set() if node.src_kind == "Symbol" else {"_src"}) | ({"_mask"} if node.has_mask else set())

        # A widths-shaped transient destination is a tile-register fill
        # (e.g. the Symbol const-fill idiom), not a global-memory write:
        # tiles are SSA values in cuTile, so the fill is a plain assignment.
        is_tile_fill = bool(dst_arr.transient) and tuple(dst_arr.shape) == widths
        if is_tile_fill:
            if node.has_mask:
                tile_expr = f"ct.where(_mask, {tile_expr}, 0)"
            return nodes.Tasklet(
                label=f"{node.label}_cutile",
                inputs={c: None
                        for c in inputs},
                outputs={"_dst": None},
                code=f"_dst = {tile_expr}",
                language=dace.dtypes.Language.Python,
            )

        ndim = len(dst_arr.strides)
        # Array dim each tile dim maps to (``dst_dims``); default to the
        # last K dims in order (a plain row-major tile). cuTile indexing
        # always spans all array dims, so unused dims are pinned to 0.
        used_dimensions = tuple(node.dst_dims) if node.dst_dims else tuple(range(ndim - K, ndim))
        unused_dimensions = tuple(sorted(set(range(ndim)) - set(used_dimensions)))
        all_dimensions = unused_dimensions + used_dimensions
        if node.dim_strides:
            coeffs = tuple(node.dim_strides)
            if len(coeffs) != K:
                raise ValueError(f"TileStore cutile expansion: dim_strides length {len(coeffs)} != widths length {K}")
            is_default_coeffs = all(s == 1 for s in coeffs)
        else:
            coeffs = tuple(1 for _ in range(K))
            is_default_coeffs = True

        lines = [f"__pid{k} = ct.bid({k})" for k in range(K)]

        if is_default_coeffs and not node.has_mask:
            # Aligned block store. The stored tile must be in array-dim
            # order with singleton extents on unused dims: insert the
            # singleton axes first (tile axes then follow ``all_dimensions``
            # order), then permute into array order if needed.
            if ndim > K:
                expand_slicer = ", ".join(["None"] * len(unused_dimensions) + [":"] * K)
                tile_expr = f"{tile_expr}[{expand_slicer}]"
            if all_dimensions != tuple(sorted(all_dimensions)):
                permute_order = tuple(all_dimensions.index(d) for d in range(ndim))
                tile_expr = f"ct.permute({tile_expr}, axes={permute_order})"
            index_entries = []
            for d in range(ndim):
                if d in used_dimensions:
                    index_entries.append(f"__pid{used_dimensions.index(d)}")
                else:
                    index_entries.append("0")  # TODO: Where should we get the index for the unused dimensions?
            if tile_expr != "_src":
                lines.append(f"__tile = {tile_expr}")
                tile_expr = "__tile"
            lines.append(f"ct.store(_dst, index=({', '.join(index_entries)},), tile={tile_expr})")
        else:
            # General case: a lane mask and/or a non-unit per-tile-dim
            # coefficient ⇒ no aligned block tile, so build explicit
            # per-destination-dim index tiles and ct.scatter (the only
            # lane-masked write cuTile offers — L-store-nomask). Index
            # entries are built per tile axis, so the stored tile stays in
            # tile-dim order (no ct.permute on this path; mirrors ct.gather).
            idx_entries = []
            for d in range(ndim):
                if d in used_dimensions:
                    k = used_dimensions.index(d)
                    # global element index along this axis:
                    # (tile_start + lane) * coeff.
                    base = f"ct.arange({widths[k]}, dtype=ct.int32) + __pid{k} * {widths[k]}"
                    if coeffs[k] != 1:
                        base = f"({base}) * {coeffs[k]}"
                    if K == 1:
                        lines.append(f"__idx{k} = {base}")
                    else:
                        # place the W_k arange on tile axis k (singleton elsewhere)
                        slicer = ", ".join(":" if a == k else "None" for a in range(K))
                        lines.append(f"__idx{k} = ct.broadcast_to(({base})[{slicer}], {widths})")
                    idx_entries.append(f"__idx{k}")
                else:
                    idx_entries.append("0")  # fixed index 0 along unused destination dims
            if tile_expr != "_src":
                lines.append(f"__tile = {tile_expr}")
                tile_expr = "__tile"
            mask_kw = ", mask=_mask" if node.has_mask else ""
            lines.append(f"ct.scatter(_dst, ({', '.join(idx_entries)},), {tile_expr}{mask_kw})")

        return nodes.Tasklet(
            label=f"{node.label}_cutile",
            inputs={c: None
                    for c in inputs},
            outputs={"_dst": None},
            code="\n".join(lines),
            language=dace.dtypes.Language.Python,
        )


@library.expansion
class ExpandTileStoreScalar(ExpandTransformation):
    """K=1 scalar backend lowering (``dace/tile_ops/scalar.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsScalar]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_store_tasklet(node, parent_state, parent_sdfg, "scalar")


@library.expansion
class ExpandTileStoreAVX512(ExpandTransformation):
    """K=1 avx512 backend lowering (``dace/tile_ops/avx512.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsAVX512]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_store_tasklet(node, parent_state, parent_sdfg, "avx512")


@library.expansion
class ExpandTileStoreAVX2(ExpandTransformation):
    """K=1 avx2 backend lowering (``dace/tile_ops/avx2.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsAVX2]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_store_tasklet(node, parent_state, parent_sdfg, "avx2")


@library.expansion
class ExpandTileStoreNeon(ExpandTransformation):
    """K=1 neon backend lowering (``dace/tile_ops/arm_neon.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsNeon]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_store_tasklet(node, parent_state, parent_sdfg, "neon")


@library.expansion
class ExpandTileStoreSVE(ExpandTransformation):
    """K=1 sve backend lowering (``dace/tile_ops/arm_sve.h``); same call as
    the other ISA backends, differing only in the included header."""

    environments = [TileOpsSVE]

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        return _isa_codegen.make_store_tasklet(node, parent_state, parent_sdfg, "sve")


@library.node
class TileStore(nodes.LibraryNode):
    """Store a K-dim tile back into a global array.

    ``_src`` is the tile transient (``widths``-shaped); ``_dst`` carries
    the full memlet of the destination array with the out-edge's subset
    selecting the tile region. ``dim_strides`` records per-tile-dim
    strides into the destination view.
    """

    implementations = {
        "pure": ExpandTileStorePure,
        "cutile": ExpandTileStoreCutile,
        "scalar": ExpandTileStoreScalar,
        "avx512": ExpandTileStoreAVX512,
        "avx2": ExpandTileStoreAVX2,
        "neon": ExpandTileStoreNeon,
        "sve": ExpandTileStoreSVE
    }
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
    dim_strides = properties.ListProperty(
        element_type=int,
        default=[],
        desc="Per-tile-dim index coefficient; all 1s ⇒ unit step along each tile dim.",
    )
    dst_dims = properties.ListProperty(
        element_type=int,
        default=[],
        desc="Per-tile-dim destination-array dimension the tile dim maps to "
        "(innermost-last). Empty ⇒ the last K dims in order; a transposed / "
        "non-last mapping lists the actual array dims so the store steps along "
        "the correct axis.",
    )
    has_mask = properties.Property(
        dtype=bool,
        allow_none=False,
        default=False,
        desc="When True, the ``_mask`` input connector is required.",
    )
    src_kind = properties.Property(
        dtype=str,
        allow_none=False,
        default="Tile",
        desc="Source operand kind. 'Tile' (default) reads a ``widths``-shaped "
        "tile transient via ``_src``. 'Symbol' broadcasts ``src_expr`` (a "
        "symbolic expression / numeric literal) to every lane and omits the "
        "``_src`` connector. 'Scalar' broadcasts a length-1 array value read "
        "via ``_src``.",
    )
    src_expr = properties.Property(
        dtype=str,
        allow_none=True,
        default=None,
        desc="Symbolic expression embedded inline when ``src_kind=='Symbol'``; "
        "ignored otherwise.",
    )

    def __init__(self,
                 name: str,
                 widths: Tuple[int, ...],
                 dim_strides: Optional[Tuple[int, ...]] = None,
                 dst_dims: Optional[Tuple[int, ...]] = None,
                 has_mask: bool = False,
                 src_kind: str = "Tile",
                 src_expr: Optional[str] = None,
                 location: Optional[str] = None):
        """Construct a ``TileStore`` node.

        :param name: Node label.
        :param widths: Per-dim tile widths, innermost-last.
        :param dim_strides: Per-tile-dim stride coefficients; defaults
            to all 1s (contiguous).
        :param has_mask: When True, declare the ``_mask`` input.
        :param src_kind: Source operand shape — ``"Tile"`` (default),
            ``"Symbol"`` (broadcast ``src_expr`` to every lane; ``_src``
            omitted), or ``"Scalar"`` (broadcast a length-1 array read
            via ``_src``).
        :param src_expr: Required when ``src_kind == 'Symbol'``.
        :param location: Optional DaCe node location override.
        :raises ValueError: If ``widths`` is empty / longer than 3, if
            ``dim_strides`` length disagrees with ``widths``, or if
            ``src_kind`` is unsupported.
        """
        if not (1 <= len(widths) <= 3):
            raise ValueError(f"TileStore: widths must have length in {{1, 2, 3}}, got {widths!r}")
        if dim_strides is not None and len(dim_strides) != len(widths):
            raise ValueError(f"TileStore: dim_strides length {len(dim_strides)} != widths length {len(widths)}")
        if src_kind not in ("Tile", "Symbol", "Scalar"):
            raise ValueError(f"TileStore: src_kind must be one of {{'Tile', 'Symbol', 'Scalar'}}, got {src_kind!r}")
        if src_kind == "Symbol" and not src_expr:
            raise ValueError("TileStore: src_kind='Symbol' requires a non-empty src_expr")
        # ``Symbol`` source has no ``_src`` connector — the literal is
        # embedded inline at expansion time. ``Tile`` and ``Scalar`` both
        # read through ``_src``.
        inputs = (set() if src_kind == "Symbol" else {"_src"}) | ({"_mask"} if has_mask else set())
        super().__init__(name, location=location, inputs=inputs, outputs={"_dst"})
        self.widths = list(widths)
        self.dim_strides = list(dim_strides) if dim_strides else [1] * len(widths)
        self.dst_dims = list(dst_dims) if dst_dims else []
        self.has_mask = has_mask
        self.src_kind = src_kind
        self.src_expr = src_expr

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        """Check connectors.

        :param sdfg: SDFG that owns ``state``.
        :param state: State that owns ``self``.
        :raises ValueError: If a required connector is unconnected.
        """
        in_e = {e.dst_conn: e for e in state.in_edges(self) if e.dst_conn is not None}
        out_e = {e.src_conn: e for e in state.out_edges(self) if e.src_conn is not None}
        if self.src_kind != "Symbol" and "_src" not in in_e:
            raise ValueError(f"{self.label}: required input '_src' not connected (src_kind={self.src_kind!r})")
        if "_dst" not in out_e:
            raise ValueError(f"{self.label}: required output '_dst' not connected")
        if self.has_mask and "_mask" not in in_e:
            raise ValueError(f"{self.label}: has_mask=True but '_mask' not connected")
