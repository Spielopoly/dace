# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Shared CPP-codegen helpers for the tile-op lib nodes' ``pure`` expansions.

These helpers keep the bodies short and idiomatic: a K-fold nested
``for``-loop with per-dim lane indices (``__l0, __l1, ...``) rather
than the flattened linear-index-plus-decode shape. Each lib node's
pure expansion plugs in its own per-lane body via :func:`nested_loops`
and uses :func:`tile_offset` to flatten the tile transient's index
(register tiles are always row-major-contiguous).
"""
import numbers
from typing import List, Sequence

import sympy

import dace

# Legal ``_idx_<d>`` gather/scatter index dtypes (design section 10.4). Unsigned widths are
# accepted because CSR/COO index arrays are commonly ``uint32``; ``gather_lane_offset`` casts
# the read to ``long long`` so an unsigned index cannot wrap the signed address sum.
GATHER_INDEX_DTYPES = (dace.int32, dace.int64, dace.uint32, dace.uint64)


def ct_dtype_name(dtype) -> str:
    """``cuda.tile`` dtype attribute name for a DaCe typeclass.

    ``ct`` exposes numpy-style names (``ct.float64``, ``ct.int32``, ...);
    only ``bool`` is spelled ``bool_``.

    :param dtype: A :class:`dace.dtypes.typeclass`.
    :returns: The attribute name on the ``cuda.tile`` module.
    """
    name = dtype.as_numpy_dtype().name
    return "bool_" if name == "bool" else name


def constant_trip_count(width) -> bool:
    """True iff ``width`` is a compile-time-constant integer loop bound.

    A per-lane tile loop with a constant trip count (the register-tile /
    vector width, e.g. ``2`` for an ``fp16x2`` fill) is safe to force-unroll:
    the bound is known at code-gen time and never a runtime / symbolic value.
    This guard keeps the ``#pragma unroll`` (see :func:`nested_loops`) off any
    hypothetical symbolic-width loop, where an unroll pragma on a runtime bound
    is meaningless. Tile widths are ``ListProperty(element_type=int)`` today, so
    the common path is the plain ``int`` check; the sympy branch is defensive.

    :param width: A per-tile-dim width (``int`` in practice; a sympy expression
        is tolerated and accepted only when it is a concrete integer).
    :returns: ``True`` for a compile-time-constant integer width, else ``False``.
    """
    if isinstance(width, numbers.Integral):
        return True
    import sympy
    return isinstance(width, sympy.Basic) and bool(width.is_Integer)


def nested_loops(widths: Sequence[int], body: str, indent: str = "    ") -> str:
    """Wrap ``body`` in a K-fold nested for-loop iterating per-dim
    lane indices ``__l0, __l1, ...``.

    Each fixed-width lane loop is preceded by ``#pragma unroll`` (guarded by
    :func:`constant_trip_count`): the trip count is the compile-time
    register-tile width -- the vector width -- so a full unroll strips the loop
    overhead of a broadcast / scalar-fill (e.g. the ``fp16`` constant-multiply
    ``_c[__l] = float16(0.125)``) and lets the backend keep the tile in
    registers. This mirrors the CPU map-unroll pragma (``cpu.py`` ``#pragma
    unroll``) and the per-lane ``#pragma unroll`` the CUDA tile-op header emits;
    the same pure tasklet body is emitted verbatim by both the CPU and CUDA
    targets, so NVCC / Clang honour the pragma and GCC harmlessly ignores the
    unknown pragma.

    :param widths: Per-tile-dim widths, innermost-last.
    :param body: The per-lane C++ body (already trailing-``;`` if
        needed); may contain newlines (each line is indented).
    :param indent: One indent level (default 4 spaces).
    :returns: A C++ snippet with the nested loops + indented body.
    """
    K = len(widths)
    lines = []
    for d, w in enumerate(widths):
        if constant_trip_count(w):
            lines.append(f"{indent * d}#pragma unroll")
        lines.append(f"{indent * d}for (std::size_t __l{d} = 0; __l{d} < {w}; ++__l{d}) {{")
    for line in body.splitlines():
        lines.append(f"{indent * K}{line}")
    for d in reversed(range(K)):
        lines.append(f"{indent * d}}}")
    return "\n".join(lines)


def _enclosing_cutile_map(node, parent_state, parent_sdfg):
    """Return the ``MapEntry`` of the enclosing ``CuTile``-scheduled map.

    Walks the in-state scope chain of ``node`` and, when ``node`` lives in a
    NestedSDFG body, continues up through ``parent_nsdfg_node`` to the owning
    state.

    :param node: The tile-op library node being expanded.
    :param parent_state: The state that owns ``node``.
    :param parent_sdfg: The SDFG that owns ``parent_state``.
    :returns: The enclosing CuTile ``MapEntry``, or ``None`` if there is none.
    """
    from dace.sdfg import nodes as _nodes
    from dace import dtypes as _dtypes
    cur_node = node
    cur_state = parent_state
    cur_sdfg = parent_sdfg
    while cur_state is not None:
        scope = cur_state.scope_dict()
        m = scope.get(cur_node)
        while m is not None:
            if isinstance(m, _nodes.MapEntry) and m.map.schedule == _dtypes.ScheduleType.CuTile:
                return m
            m = scope.get(m)
        nsdfg_node = cur_sdfg.parent_nsdfg_node if cur_sdfg is not None else None
        if nsdfg_node is None or cur_sdfg.parent is None:
            break
        cur_node = nsdfg_node
        cur_state = cur_sdfg.parent
        cur_sdfg = cur_state.sdfg
    return None


# ---------------------------------------------------------------------------
# CuTile launch-grid folding (>3-D grid support)
#
# The ``cuda.tile`` runtime caps the launch grid at three axes (``ct.launch``
# takes a ``Dim3`` and ``ct.bid(axis)`` only accepts ``axis in {0, 1, 2}``).
# When a tiled ``CuTile`` map has more than three dimensions (e.g. 4-D
# ``softmax``, 5-D ``conv2d``), the extra grid dimensions are FOLDED onto the
# available axes and recovered inside the kernel via integer div/mod. The
# single canonical layout, shared by the map-entry codegen
# (``codegen/py/cutile_target.py``) and every tile-op ``cutile`` expansion, is:
#
#   * The two innermost map dimensions (``M-2`` and ``M-1``) map to grid axes
#     1 and 2 respectively.
#   * The remaining leading dimensions (``0 .. M-3``) are folded, row-major,
#     onto grid axis 0 (``x``, which carries the largest hardware grid-size
#     limit). Their block IDs are recovered from ``ct.bid(0)`` via div/mod.
#
# For ``M <= 3`` the identity mapping (``__pid{d} = ct.bid(d)``) is used, which
# is unchanged from the pre-folding behavior.
# ---------------------------------------------------------------------------


def cutile_grid_size_exprs(cutile_map_entry) -> List[str]:
    """Per-dimension grid size (tile count) expressions for a CuTile map.

    Each dimension's grid size is the number of tiles
    ``int_ceil(extent, step)`` where ``extent = end - start + 1`` (inclusive
    map-range end). Emitted as a structural integer ceil-division to keep both
    operands integral for the Python/cuTile backend (mirrors
    ``cutile_target._grid_exprs_from_map_entry``).

    :param cutile_map_entry: The enclosing CuTile ``MapEntry``.
    :returns: List of grid-size expression strings, one per map dimension.
    """
    from dace.symbolic import symstr
    exprs: List[str] = []
    for start, end, step in cutile_map_entry.map.range:
        extent_s = symstr(end - start + 1)
        step_s = symstr(step)
        exprs.append(extent_s if step_s == "1" else f"int_ceil({extent_s}, {step_s})")
    return exprs


def cutile_launch_grid_dims(grid_size_exprs: Sequence[str]) -> List[str]:
    """Fold per-dimension grid sizes onto the <=3-axis cuTile launch grid.

    :param grid_size_exprs: Per-dimension grid-size expressions in map order
        (dimension 0 outermost), length ``M``.
    :returns: List of at most three launch-grid dimension expressions passed to
        ``ct.launch``. For ``M <= 3`` the sizes are padded with ``"1"`` to a
        3-tuple; for ``M > 3`` the leading ``M-2`` dimensions are multiplied
        onto axis 0.
    """
    num_dims = len(grid_size_exprs)
    if num_dims <= 3:
        return (list(grid_size_exprs) + ["1", "1", "1"])[:3]
    folded = grid_size_exprs[:num_dims - 2]
    axis0 = " * ".join(f"({g})" for g in folded)
    return [axis0, grid_size_exprs[num_dims - 2], grid_size_exprs[num_dims - 1]]


def cutile_bid_expr(map_axis: int, num_map_dims: int, grid_size_exprs: Sequence[str]) -> str:
    """Block-ID expression for a single CuTile map grid axis (folding-aware).

    Returns the Python expression that evaluates, inside a cuTile kernel, to
    the block index of map dimension ``map_axis``. For ``num_map_dims <= 3``
    this is simply ``ct.bid(map_axis)``. For ``num_map_dims > 3`` the leading
    ``M-2`` dimensions are recovered from ``ct.bid(0)`` via div/mod and the two
    innermost dimensions read ``ct.bid(1)`` / ``ct.bid(2)`` (see the module
    header for the canonical layout).

    :param map_axis: The map dimension index (grid axis) to recover.
    :param num_map_dims: Total number of CuTile map dimensions (``M``).
    :param grid_size_exprs: Per-dimension grid-size expressions (length ``M``).
    :returns: A Python expression string yielding the block ID for ``map_axis``.
    """
    if num_map_dims <= 3:
        return f"ct.bid({map_axis})"
    if map_axis == num_map_dims - 2:
        return "ct.bid(1)"
    if map_axis == num_map_dims - 1:
        return "ct.bid(2)"
    # Folded leading dimension (0 <= map_axis <= M-3), recovered from ct.bid(0).
    g_axis = grid_size_exprs[map_axis]
    inner = grid_size_exprs[map_axis + 1:num_map_dims - 2]  # dims strictly inside, up to M-3
    if not inner:  # innermost folded dim (map_axis == M-3)
        return f"(ct.bid(0) % ({g_axis}))"
    divisor = " * ".join(f"({g})" for g in inner)
    if map_axis == 0:  # outermost folded dim: quotient is already < g_axis
        return f"(ct.bid(0) // ({divisor}))"
    return f"((ct.bid(0) // ({divisor})) % ({g_axis}))"


def cutile_bid_lines(node, parent_state, parent_sdfg, axes: Sequence[int]) -> List[str]:
    """Emit ``__pid{k} = <block-id>`` bindings for a tile op's grid axes.

    Resolves each tile-op tile dim's block ID against the enclosing CuTile map,
    applying the >3-D grid fold (see the module header) so the axis indices
    stay within the runtime's 3-axis cap. When there is no enclosing CuTile map
    or it has at most three dimensions, the direct ``ct.bid(axis)`` form is used
    (identical to the pre-folding behavior).

    :param node: The tile-op library node being expanded.
    :param parent_state: The state that owns ``node``.
    :param parent_sdfg: The SDFG that owns ``parent_state``.
    :param axes: Per-tile-dim grid axis (enclosing-map dimension index), or
        ``None`` for a dim no grid axis drives (its base is fully carried by
        the recovered offset; see :func:`cutile_tile_dim_bids`) — bound as
        ``__pid{k} = 0``.
    :returns: List of ``__pid{k} = ...`` statement strings, one per tile dim.
    """
    m = _enclosing_cutile_map(node, parent_state, parent_sdfg)
    if m is None or len(m.map.range) <= 3:
        return [f"__pid{k} = {'0' if axes[k] is None else f'ct.bid({axes[k]})'}" for k in range(len(axes))]
    num_map_dims = len(m.map.range)
    grid = cutile_grid_size_exprs(m)
    return [
        f"__pid{k} = {'0' if axes[k] is None else cutile_bid_expr(axes[k], num_map_dims, grid)}"
        for k in range(len(axes))
    ]


def cutile_grid_dim_offset(node, parent_state, parent_sdfg, K: int) -> int:
    """Number of enclosing CuTile-map grid dims that precede the K tile dims.

    The cuTile codegen binds ``__pid{d} = ct.bid(d)`` positionally for each
    dim of the enclosing ``CuTile`` map (``map.range[d]`` -> grid axis ``d``).
    A tile op runs inside the body of the innermost (tiled) loops; when the map
    also carries OUTER, non-tiled point dims (e.g. an ICON block loop ``jb``),
    those are the LEADING grid axes, so a tile op spanning ALL ``K`` tiled dims
    occupies grid axes ``offset .. offset + K - 1`` with
    ``offset = len(map.range) - K``.

    This is correct only for a tile op whose K dims ARE the innermost K (the
    common case: the iteration mask, the main data load/store). For a tile op
    that tiles a STRICT SUBSET of the tiled dims (e.g. a 1-D gather index tile
    that depends on a non-innermost loop) use :func:`cutile_tile_dim_bids`,
    which resolves each tile dim's grid axis individually.

    :param node: The tile-op library node being expanded.
    :param parent_state: The state that owns ``node``.
    :param parent_sdfg: The SDFG that owns ``parent_state``.
    :param K: The tile-op's tile-dim count (``len(widths)``).
    :returns: The grid-dim offset, or 0 when there is no enclosing CuTile map
        or it has exactly K dims (the common fully-tiled case, e.g. cuTile V1).
    """
    m = _enclosing_cutile_map(node, parent_state, parent_sdfg)
    return max(0, len(m.map.range) - K) if m is not None else 0


def cutile_tile_dim_bids(node, parent_state, parent_sdfg, used_dimensions: Sequence[int], src_begins: Sequence[str],
                         K: int) -> list:
    """Resolve the ``ct.bid`` grid axis for each of a load/store's K tile dims.

    The grid axis for tile dim ``d`` is the position, in the enclosing CuTile
    map's ``params``, of the iteration variable that tile dim ``d`` walks. That
    variable is the memlet begin of the source/dest array dim the tile dim maps
    to (``src_begins[used_dimensions[d]]`` -- e.g. ``A[jb, jk:jk+8]`` gives
    begin ``jk`` for the tiled dim). Matching it to ``map.params`` yields the
    exact grid axis even when the tile dim is NOT the innermost loop (the
    positional :func:`cutile_grid_dim_offset` is wrong in that case).

    Falls back to the trailing-K offset for any dim whose variable cannot be
    matched (e.g. a constant/broadcast dim, whose ``__pid`` is unused anyway),
    and entirely when there is no resolvable enclosing CuTile map.

    :param node: The tile-op library node being expanded.
    :param parent_state: The state that owns ``node``.
    :param parent_sdfg: The SDFG that owns ``parent_state``.
    :param used_dimensions: Per-tile-dim source/dest array dim index.
    :param src_begins: Per-source-dim memlet begin expression (as strings).
    :param K: The tile-op's tile-dim count.
    :returns: List of ``K`` entries: the grid-axis index, or ``None`` for a
        dim whose begin is driven by a non-grid variable (e.g. a sequential
        inner tile loop) — no ``__pid*W`` term applies there.
    """
    import dace.symbolic as _sym
    m = _enclosing_cutile_map(node, parent_state, parent_sdfg)
    offset = max(0, len(m.map.range) - K) if m is not None else 0
    fallback = [offset + d for d in range(K)]
    if m is None:
        return fallback
    params = [str(p) for p in m.map.params]
    bids = []
    for d in range(K):
        sd = used_dimensions[d] if d < len(used_dimensions) else None
        pos = None
        has_symbols = False
        if sd is not None and 0 <= sd < len(src_begins):
            try:
                syms = {str(s) for s in _sym.pystr_to_symbolic(str(src_begins[sd])).free_symbols}
            except Exception:  # noqa: BLE001 - non-symbolic begin -> use the raw string
                syms = {str(src_begins[sd])}
            has_symbols = any(not s.lstrip('-').isdigit() for s in syms)
            for i, p in enumerate(params):
                if p in syms:
                    pos = i
                    break
        if pos is None and has_symbols:
            # The begin is driven by a variable that is NOT a grid param
            # (e.g. a tile-strided SEQUENTIAL inner map inside the kernel,
            # from a dependent range like ``j = i+1 : N : W``).  No block id
            # advances this dim -- the full base is carried by the recovered
            # offset (:func:`cutile_tile_dim_offsets` leaves non-grid
            # variables in place) -- so the ``__pid*W`` term must vanish:
            # signal "no grid axis" and let :func:`cutile_bid_lines` bind
            # ``__pid{d} = 0``.  The positional fallback here would read a
            # WRONG grid axis (an outer point dim) and silently miscompile.
            bids.append(None)
        else:
            bids.append(pos if pos is not None else fallback[d])
    return bids


def cutile_tile_dim_offsets(node, parent_state, parent_sdfg, used_dimensions: Sequence[int], begins: Sequence[str],
                            K: int) -> list:
    """Return the constant per-tile-dim element offset carried by the memlet begin.

    A tiled load/store's per-dim memlet begin has the form ``<iter-var> + c`` --
    e.g. ``A[1:-1]`` yields begin ``__i0 + 1`` for the tiled dim, where ``__i0``
    is the block's element start (``__pid_k * W``) and ``c = 1`` is the constant
    offset of the slice. The block-aligned ``ct.load`` / ``ct.arange + __pid*W``
    index only reconstructs ``__i0`` and drops ``c``, so an offset slice
    (``A[1:-1]``, ``A[2:]``) reads/writes one (or more) elements too low. This
    helper recovers ``c`` for each tile dim by substituting every enclosing
    CuTile-map iteration variable in the begin with its range START -- what
    remains is the part of the address the block id does NOT advance and must
    be added back to the per-lane element index. For canonical 0-start maps the
    start substitution degenerates to the plain constant slice offset; for a
    nonzero-start map (e.g. the vectorizer's remainder map over
    ``4*int_floor(N,4):N``) the start itself becomes part of the offset, since
    ``__pid * W`` counts blocks relative to the range start.

    Map parameters whose range provably has a single iteration (e.g. a
    single-tile inner loop ``0:4:4``) are substituted with their constant
    value first — that is exact, not a guess. After that the begin must be
    affine in AT MOST ONE enclosing-map parameter (the dim's own iteration
    variable, the same one :func:`cutile_tile_dim_bids` resolves the block id
    from). A begin coupling several multi-iteration map parameters (e.g.
    ``A[__i0 + __i1]``) cannot be reconstructed from a single block id and
    raises ``NotImplementedError`` instead of silently producing a wrong
    offset.

    :param node: The tile-op library node being expanded.
    :param parent_state: The state that owns ``node``.
    :param parent_sdfg: The SDFG that owns ``parent_state``.
    :param used_dimensions: Per-tile-dim source/dest array dim index.
    :param begins: Per-source/dest-dim memlet begin expression (as strings).
    :param K: The tile-op's tile-dim count.
    :returns: List of ``K`` symbolic offsets (usually integers; ``0`` when the
        slice is anchored at the block-aligned start).
    :raises NotImplementedError: If a begin cannot be parsed or couples more
        than one enclosing-map parameter.
    """
    import dace.symbolic as _sym
    m = _enclosing_cutile_map(node, parent_state, parent_sdfg)
    params = []
    starts = []
    single_iter_subs = {}
    if m is not None:
        for p, rng in zip(m.map.params, m.map.range):
            try:
                p_sym = _sym.pystr_to_symbolic(str(p))
            except Exception:  # noqa: BLE001 - skip unparseable params
                params.append(None)
                starts.append(rng[0])
                continue
            params.append(p_sym)
            start, end, step = rng[0], rng[1], rng[2]
            starts.append(start)
            # A range with provably one iteration (extent <= step) pins the
            # parameter to its start value; substituting it is exact.
            try:
                if bool(_sym.simplify(end - start + 1 - step) <= 0):
                    single_iter_subs[p_sym] = start
            except (TypeError, ValueError):
                pass  # symbolic, not provably single-iteration
    param_set = {p for p in params if p is not None}
    offsets = []
    for d in range(K):
        sd = used_dimensions[d] if d < len(used_dimensions) else None
        if sd is None or sd < 0 or sd >= len(begins):
            offsets.append(0)
            continue
        try:
            begin = _sym.pystr_to_symbolic(str(begins[sd]))
        except Exception as ex:  # noqa: BLE001 - unusable begin must fail loudly
            raise NotImplementedError(f"{node.label}: cannot parse memlet begin {begins[sd]!r} for tile dim {d} "
                                      f"(array dim {sd}); refusing to guess the element offset.") from ex
        if single_iter_subs:
            begin = begin.subs(single_iter_subs)
        own = sorted((p for p in param_set if p in begin.free_symbols), key=str)
        if len(own) > 1:
            raise NotImplementedError(
                f"{node.label}: memlet begin {begins[sd]!r} for tile dim {d} (array dim {sd}) couples "
                f"multiple enclosing CuTile-map parameters {[str(p) for p in own]}; the block-aligned "
                f"index reconstruction supports at most one iteration variable per dim. Such accesses "
                f"should have been routed to the gather/scatter path by the detection passes.")
        if own:
            own_p = own[0]
            # ``__pid * W`` counts blocks relative to the range start, so the
            # begin evaluated AT the start is exactly the part the block id
            # does not advance (0-start maps degenerate to the slice constant).
            start = starts[params.index(own_p)]
            begin = begin.subs(own_p, start)
        remaining = _sym.simplify(begin)
        foreign = param_set & set(remaining.free_symbols)
        if foreign:
            raise NotImplementedError(
                f"{node.label}: element offset {remaining} for tile dim {d} (array dim {sd}) still "
                f"contains enclosing CuTile-map parameter(s) {[str(p) for p in sorted(foreign, key=str)]} "
                f"after removing the dim's own iteration variable; refusing to silently zero them.")
        offsets.append(remaining)
    return offsets


def cutile_offset_is_nonzero(offset) -> bool:
    """Whether a tile-dim element offset is provably non-zero.

    :param offset: A symbolic / numeric offset from :func:`cutile_tile_dim_offsets`.
    :returns: ``True`` if the offset does not simplify to ``0``.
    """
    import dace.symbolic as _sym
    try:
        return bool(_sym.simplify(offset) != 0)
    except Exception:  # noqa: BLE001 - be conservative: treat as offset present
        return offset != 0


def cutile_offset_block_shift(offset, width: int):
    """Block-index shift for a tile-dim element offset, if provably block-aligned.

    When the element offset is a provably nonnegative multiple of the tile
    width, an aligned ``ct.load`` / ``ct.store`` stays valid with the block
    index shifted by ``offset // width`` (fast path instead of per-element
    gather/scatter). Nonnegativity guards the undefined-behavior case of a
    tile lying entirely before the array start.

    :param offset: A symbolic / numeric offset from :func:`cutile_tile_dim_offsets`.
    :param width: The tile width of that dim.
    :returns: The symbolic block shift ``offset // width``, or ``None`` when
        divisibility / nonnegativity is not provable (caller must take the
        gather/scatter path).
    """
    import dace.symbolic as _sym
    try:
        q = _sym.simplify(_sym.pystr_to_symbolic(str(offset)) / int(width))
    except Exception:  # noqa: BLE001 - unprovable -> no fast path
        return None
    if getattr(q, "is_integer", None) and getattr(q, "is_nonnegative", None):
        return q
    return None


def tile_offset(widths: Sequence[int]) -> str:
    """Return the row-major flat offset expression for a register tile.

    For ``widths = (W_0, W_1, W_2)`` returns
    ``__l0 * (W_1*W_2) + __l1 * W_2 + __l2``. Always row-major because
    register-storage tile transients are contiguous by construction.

    :param widths: Per-tile-dim widths, innermost-last.
    :returns: The C++ offset expression.
    """
    K = len(widths)
    if K == 0:
        return "0"
    stride = 1
    parts = []
    for d in reversed(range(K)):
        if stride == 1:
            parts.append(f"__l{d}")
        else:
            parts.append(f"(__l{d} * {stride})")
        stride *= widths[d]
    return " + ".join(reversed(parts))


def offset_via_strides(
    coeffs: Sequence[int],
    strides: Sequence[str],
    replicate_factors: Sequence[int] = (),
    lane_index_exprs: Sequence[str] = ()) -> str:
    """Return the flat offset expression
    ``sum_d coeffs[d] * strides[d] * (__l<d> / replicate_factors[d])``.

    Used by ``TileLoad`` / ``TileStore`` to address the source / dest
    array's flat memory through its own per-dim strides scaled by the
    optional per-tile-dim ``dim_strides`` coefficient. When
    ``replicate_factors[d] > 1``, the per-dim lane index is divided by
    the replicate factor so ``k`` consecutive lanes index the same
    source element -- the within-dim group-broadcast lowering for the
    ``int_floor`` / ``int_ceil`` regime.

    Per-lane override: when ``lane_index_exprs[d]`` is a non-empty string,
    dim ``d`` uses it verbatim as the per-lane element offset *relative to
    the connector base* (the dim contributes ``(lane_index_exprs[d]) *
    strides[d]`` and the dim's ``coeffs`` / ``replicate_factors`` are
    bypassed). The ``TileLoad`` pure expansion supplies it for a
    non-dividing ``int_floor(c*iter + c0, D)`` (``W % D != 0`` or symbolic
    ``D``): the contracted-box broadcast ``_src[__l/D]`` is correct only
    when every tile starts on a phase boundary (``W % D == 0``), so the
    expansion instead passes the phase-aware ``(c*iter + c0 + c*__l)/D -
    (c*iter + c0)/D`` (relative to the box base ``&src[(c*iter+c0)/D]``).

    :param coeffs: Per-tile-dim integer coefficient (``1`` for
        contiguous; >1 for strided access).
    :param strides: Per-tile-dim source-array stride as a C++
        expression (typically the symbolic stride rendered with
        :func:`dace.symbolic.symstr`).
    :param replicate_factors: Per-tile-dim replicate factor (``1`` =
        each lane reads a distinct element; ``k > 1`` = ``k`` lanes
        share each element). Defaults to all-1 (no replication) when
        empty or omitted.
    :param lane_index_exprs: Per-tile-dim per-lane element offset C++
        expression (relative to the connector base); an empty / missing
        entry uses the standard ``coeff * stride * (__l / replicate)``
        addressing. Defaults to all-empty.
    :returns: The C++ offset expression, or ``"0"`` if K==0.
    """
    if not coeffs:
        return "0"
    parts = []
    for d, (c, s) in enumerate(zip(coeffs, strides)):
        if d < len(lane_index_exprs) and lane_index_exprs[d]:
            parts.append(f"(({lane_index_exprs[d]}) * ({s}))")
            continue
        lane = f"__l{d}"
        if d < len(replicate_factors):
            r = replicate_factors[d]
            # Symbolic replicate factors (e.g. ``DV`` in ``c[i // DV]``)
            # can't be compared via ``> 1`` (sympy raises TypeError).
            # Coerce to int when possible; symbolic falls through to the
            # runtime divisor emission -- ``__l / DV`` evaluates safely
            # at any DV >= 1.
            try:
                emit_div = int(r) > 1
            except (TypeError, ValueError):
                emit_div = True
            if emit_div:
                lane = f"({lane} / {r})"
        parts.append(f"({c} * ({s}) * {lane})")
    return " + ".join(parts)


def resolve_gather_deps(idx_shape, widths):
    """Find the sorted subset of tile dims an ``_idx_<d>`` index tile depends on.

    Implements the design section 9.2 lane-dependency lookup: given an
    ``_idx_<d>`` connector's descriptor shape and the lib node's tile widths,
    return the sorted tuple of tile dim indices ``deps_d`` the index tile
    varies over, or ``None`` if the shape cannot be reconciled with ``widths``.
    The special ``(1,)`` shape (scalar gather index, no lane dep) returns the
    empty tuple ``()``.

    **Index tiles are full-K-dim and resolve POSITIONALLY — ``ONE`` is NEVER
    collapsed** (user direction 2026-06-14: the markers must be preserved). A
    ``K``-dim index tile encodes its per-tile-dim dependencies *by position*:
    tile dim ``d`` is a dependency iff ``idx_shape[d]`` is neither literal ``1``
    nor the :data:`~dace.symbolic.ONE` broadcast marker (and a non-marker extent
    must equal ``widths[d]``). ``(W, ONE)`` (col gather, dep dim 0) and
    ``(ONE, W)`` (row gather, dep dim 1) are DISTINCT — collapsing both to
    ``(W,)`` would make equal-width tiles (``widths=(8, 8)``) ambiguous, the
    exact bug the ``ONE`` marker exists to prevent. The index-tile emitters
    (:meth:`InsertTileLoadStore._stage_array_read_tile`) therefore always emit
    the full-rank ``ONE``-padded form; a shape whose rank is not ``K`` (other
    than the scalar ``(1,)``) is rejected.

    :param idx_shape: The descriptor shape of an ``_idx_<d>`` connector
        (e.g. ``(4, 8)``, ``(4, ONE)``, ``(ONE, 8)``).
    :param widths: The lib node's full tile widths ``(W_0, ..., W_{K-1})``.
    :returns: Sorted tuple of tile dim indices, ``()`` for the scalar case,
        or ``None`` when the shape cannot be reconciled with ``widths``.
    """
    import dace
    # ``has_one_marker`` is True only for the ONE broadcast marker -- NOT a literal ``1``. A
    # literal ``1`` extent means a genuine width-1 tile dim (a real dependency). The two must
    # stay distinct here -- that disambiguation (broadcast vs a coincidental width-1 dep) is the
    # whole reason ONE is a symbol and not just ``1`` (user 2026-06-14), and it is what keeps the
    # index-tile rank aligned with the data tile (cuTile-faithful broadcast dims).
    from dace.symbolic import has_one_marker

    def _extent_eq(a, b):
        """Symbolic-safe extent equality."""
        try:
            return bool(dace.symbolic.simplify(a - b) == 0)
        except Exception:  # noqa: BLE001
            return a == b

    idx_shape = tuple(idx_shape)
    K = len(widths)
    # Scalar gather index (no lane dep): the legacy literal ``(1,)`` shape. A
    # K-dim all-``ONE`` shape is also scalar and falls out of the positional
    # loop below (every dim skipped -> empty deps).
    if idx_shape == (1, ):
        return ()
    # Full-K-dim positional resolution. The ONE markers are PRESERVED, never
    # collapsed: dim d is a dep iff its extent is not a 1/ONE broadcast marker.
    if len(idx_shape) != K:
        return None
    deps = []
    for d in range(K):
        if has_one_marker(idx_shape[d]):
            continue  # broadcast dim -- not a dependency
        if not _extent_eq(idx_shape[d], widths[d]):
            return None  # non-marker extent disagrees with the tile width
        deps.append(d)
    return tuple(deps)


def _no_ipow(expr):
    """Rewrite ``ipow(b, e)`` (dace's opaque integer-power function, e.g. from
    ``RelaxIntegerPowers``) back to ``b**e`` so sympy comparisons see through it.

    :param expr: A stride/shape entry (int or sympy expression).
    :returns: The expression with every ``ipow`` replaced by ``Pow``.
    """
    import sympy

    import dace
    if not isinstance(expr, sympy.Basic):
        return expr
    return expr.replace(dace.symbolic.ipow, lambda b, e: b**e)


def _strides_match_packed(shape, strides, order):
    """True when ``strides`` is the packed contiguous form for ``shape`` in
    ``order`` ("C" -- innermost-last, stride 1 on the last dim; or "F" --
    innermost-first, stride 1 on the first dim) with NO padding between dims.

    Symbolic shapes / strides are compared via sympy ``simplify == 0``, after
    normalizing ``ipow`` to ``Pow`` (``ipow(SM, 2)`` is otherwise opaque to
    sympy and a packed ``SM**2`` stride would be falsely rejected).

    :param shape: Tuple of dim sizes (may be symbolic).
    :param strides: Tuple of per-dim strides (may be symbolic).
    :param order: "C" or "F".
    :returns: ``True`` iff the layout is exactly packed in the requested order.
    """
    import dace
    if len(shape) != len(strides):
        return False
    if order == "C":
        order_range = range(len(shape) - 1, -1, -1)
    elif order == "F":
        order_range = range(len(shape))
    else:
        raise ValueError(f"order must be 'C' or 'F'; got {order!r}")
    expected = 1
    for d in order_range:
        try:
            # relax_ipow so the canonicalized packed-C stride ``ipow(N, 2)`` compares equal to
            # ``N*N``; the opaque ``ipow`` never simplifies against ``expected`` (heat3d).
            # Equalize before simplifying: a stride and a shape dim can carry two same-named symbol
            # INSTANCES (different dtype/assumptions) whose subtraction never cancels (channel_flow).
            diff = strides[d] - expected
            if isinstance(diff, sympy.Basic):
                diff = dace.symbolic.simplify(dace.symbolic.relax_ipow(dace.symbolic.equalize_symbol(diff)))
            if diff != 0:
                return False
        except Exception:  # noqa: BLE001 -- conservative refusal on un-comparable expressions.
            return False
        expected = expected * shape[d]
    return True


def validate_packed_layout(node_label, conn_name, desc):
    """Refuse any source / dest array whose stride pattern is neither packed C
    nor packed Fortran (design section 2.3).

    Padded layouts -- where strides exceed the product of inner dims -- raise
    :class:`NotImplementedError` until per-arch codegen support lands. 1-D
    arrays trivially satisfy both packings and are accepted iff their single
    stride is 1.

    :param node_label: Label of the calling lib node (for error messages).
    :param conn_name: Connector name carrying the array (typically ``_src``
        or ``_dst``).
    :param desc: The array descriptor (``dace.data.Data`` subclass) wired to
        the connector.
    :raises NotImplementedError: On non-packed-C non-packed-Fortran layout.
    """
    import dace
    if not isinstance(desc, dace.data.Array):
        return  # Scalars / Streams have no per-dim stride pattern to check.
    shape = tuple(desc.shape)
    strides = tuple(desc.strides)
    if len(shape) == 0:
        return
    if len(shape) == 1:
        try:
            if dace.symbolic.simplify(_no_ipow(strides[0]) - 1) != 0:
                raise NotImplementedError(f"{node_label}: {conn_name!r} has non-unit stride "
                                          f"{strides[0]} on its single dim; only packed layouts are "
                                          f"supported (section 2.3).")
        except NotImplementedError:
            raise
        except Exception:  # noqa: BLE001
            raise NotImplementedError(f"{node_label}: {conn_name!r} stride {strides[0]} could not be "
                                      f"verified against the packed-layout invariant (section 2.3).")
        return
    if not (_strides_match_packed(shape, strides, "C") or _strides_match_packed(shape, strides, "F")):
        raise NotImplementedError(f"{node_label}: {conn_name!r} has non-packed stride pattern "
                                  f"(shape={shape}, strides={strides}). Only packed-C and packed-"
                                  f"Fortran layouts are supported (section 2.3); padded layouts raise "
                                  f"NotImplementedError until codegen lands.")


def validate_mask_descriptor_lock(node_label, conn_name, desc, widths):
    """Refuse any mask descriptor that breaks the design section 10.2 lock.

    The locked shape: ``Array(shape=widths, dtype=bool_, storage=Register,
    transient=True)``. Anything else -- scalar masks, per-dim masks, non-bool
    predicates, non-Register storage, non-transient -- is rejected with a
    named error so the codegen never silently mis-emits.

    :param node_label: Label of the calling lib node (for error messages).
    :param conn_name: Connector name carrying the mask (typically ``_mask``
        or ``_o``).
    :param desc: The descriptor (``dace.data.Data`` subclass) of the array
        wired to the connector.
    :param widths: Tile widths ``(W_0, ..., W_{K-1})``.
    :raises ValueError: On any descriptor lock violation.
    """
    import dace
    if not isinstance(desc, dace.data.Array):
        raise ValueError(f"{node_label}: {conn_name!r} mask must be a dace.data.Array, "
                         f"got {type(desc).__name__}")
    if tuple(desc.shape) != tuple(widths):
        raise ValueError(f"{node_label}: {conn_name!r} mask shape {tuple(desc.shape)} must "
                         f"match widths {tuple(widths)} (section 10.2)")
    if desc.dtype != dace.bool_:
        raise ValueError(f"{node_label}: {conn_name!r} mask dtype {desc.dtype} must be bool_ "
                         f"(section 10.2)")
    if desc.storage != dace.dtypes.StorageType.Register and desc.storage != dace.dtypes.StorageType.CuTile_Tile:
        raise ValueError(f"{node_label}: {conn_name!r} mask storage {desc.storage} must be "
                         f"Register (section 10.2)")
    if not desc.transient:
        raise ValueError(f"{node_label}: {conn_name!r} mask must be transient (section 10.2)")


def gather_lane_offset(deps, widths, conn):
    """Build the row-major flat lane offset C expression into an ``_idx_<d>`` tile.

    Given ``deps = (p_0, ..., p_{n-1})`` (the tile dims the gather expression
    depends on) and the lib node's widths, returns the CPP expression
    ``conn[<flat offset>]`` where the flat offset is
    ``__l<p_0> * (W_<p_1> * W_<p_2> * ...) + __l<p_1> * (W_<p_2> * ...) + ... + __l<p_{n-1}>``.

    For the scalar case (``deps == ()``) returns ``conn[0]``.

    The read is cast to ``long long``: callers sum it with affine terms that may
    be negative, and an unsigned index dtype would make the whole sum wrap.

    :param deps: Sorted tuple of tile dim indices from :func:`resolve_gather_deps`.
    :param widths: Lib node's full tile widths.
    :param conn: The connector name (e.g. ``"_idx_0"``).
    :returns: A CPP expression string of the form ``(long long)conn[<offset>]``.
    """
    if not deps:
        return f"(long long)({conn}[0])"
    parts = []
    for i, p in enumerate(deps):
        inner = 1
        for q in deps[i + 1:]:
            inner *= widths[q]
        parts.append(f"__l{p}" if inner == 1 else f"(__l{p} * {inner})")
    return f"(long long)({conn}[{' + '.join(parts)}])"
