# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""cuTile lowering passes: tileops-anchored schedule, storage, and
implementation stamping for the Python/cuTile backend.

The six passes lower a tile-op SDFG — produced by
``VectorizeCPUMultiDim(target_isa="CUTILE", expand_tile_nodes=False)`` —
into the form the cuTile code generator (``dace/codegen/py/cutile_target.py``)
consumes. All stamping is anchored on the emitted ``tileops`` library nodes,
never on map-range heuristics (a pre-existing strided map such as
``A[i*2, j]`` must NOT become a kernel just because its step is > 1).

Required order::

    CuTileValidateTiles        # tile-op anchors exist; widths are powers of 2
    CuTileSetSchedules         # outermost enclosing map -> CuTile, inner -> Sequential
    CuTileSetTileStorage       # Register tile transients -> CuTile_Tile
    CuTileSetGlobalStorage     # kernel-touched non-transients -> GPU_Global
    CuTileInsertDataCopies     # (optional) clone GPU_Global non-transients, add copy states
    CuTileSetImplementations   # lib nodes -> target_isa="CUTILE", implementation="cutile"

followed by ``sdfg.expand_library_nodes()`` and
``sdfg.backend = dtypes.BackendLanguage.Python`` (single core-API calls,
performed by the ``VectorizeCuTile`` orchestrator or two explicit lines in a
manual recipe).

Out-of-order behavior (every pass cheaply validates its preconditions and
warns by default; ``strict=True`` raises instead):

- A pass run before the vectorizer finds no tile-op anchors -> warn/no-op.
- A storage pass run before ``CuTileSetSchedules`` finds no CuTile-scheduled
  map -> warn/no-op.
- ``CuTileSetImplementations`` run after ``expand_library_nodes()`` finds no
  library nodes -> warn/no-op.
- Every pass is idempotent: re-running the whole sequence changes nothing.

Exception: non-power-of-2 tile widths raise ``ValueError`` unconditionally
(a hard ``cuda.tile`` runtime requirement, not gated by ``strict``), as does
a tile-op node type that lacks a ``'cutile'`` implementation (loud failure
instead of a silent ``'pure'`` fallback).
"""
import warnings
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Type

from dace import SDFG, data, dtypes, memlet as mmlt, properties, transformation
from dace.sdfg import nodes
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl

#: Scope-dictionary cache type: per-state ``state.scope_dict()`` results.
_ScopeCache = Dict[SDFGState, Dict[nodes.Node, Optional[nodes.Node]]]


def _tile_node_types() -> Tuple[Type[nodes.LibraryNode], ...]:
    """Return the tuple of all tileops library-node classes.

    Imported lazily to keep the transformation package importable without
    pulling in the tileops library at module-import time.

    :returns: All tile-op ``LibraryNode`` classes exported by
        :mod:`dace.libraries.tileops.nodes` (including :class:`TileIota`).
    """
    from dace.libraries.tileops.nodes import (TileBinop, TileIota, TileLoad, TileMaskGen, TileReduce, TileStore,
                                              TileUnop)
    return (TileBinop, TileIota, TileLoad, TileMaskGen, TileReduce, TileStore, TileUnop)


def _collect_tile_nodes(sdfg: SDFG) -> List[Tuple[nodes.LibraryNode, SDFGState]]:
    """Collect every tileops library node in ``sdfg``, recursively.

    :param sdfg: SDFG to search (NestedSDFGs are included).
    :returns: List of ``(lib_node, state)`` pairs, where ``state`` is the
        SDFGState that owns the node.
    """
    tile_types = _tile_node_types()
    return [(node, graph) for node, graph in sdfg.all_nodes_recursive() if isinstance(node, tile_types)]


def _enclosing_map_chain(node: nodes.Node,
                         state: SDFGState,
                         scope_cache: Optional[_ScopeCache] = None) -> List[Tuple[nodes.MapEntry, SDFGState]]:
    """Walk the chain of enclosing map scopes of ``node``, across NestedSDFGs.

    The in-state scope chain comes from ``state.scope_dict()`` (cached per
    state in ``scope_cache``); when the owning SDFG is nested, the walk
    continues from its ``parent_nsdfg_node`` in the parent state, transitively
    up to the top-level SDFG.

    :param node: The dataflow node whose enclosing maps are collected.
    :param state: The SDFGState that owns ``node``.
    :param scope_cache: Optional mutable per-state cache of scope dictionaries
        (avoids recomputing ``scope_dict()`` for repeated walks).
    :returns: List of ``(map_entry, state)`` pairs, innermost-first /
        outermost-last; empty if ``node`` is not inside any map at any level.
    """
    if scope_cache is None:
        scope_cache = {}
    chain: List[Tuple[nodes.MapEntry, SDFGState]] = []
    current_node: nodes.Node = node
    current_state: SDFGState = state
    while True:
        scope_dict = scope_cache.get(current_state)
        if scope_dict is None:
            scope_dict = current_state.scope_dict()
            scope_cache[current_state] = scope_dict
        scope = scope_dict.get(current_node)
        while scope is not None:
            chain.append((scope, current_state))
            scope = scope_dict.get(scope)
        owning_sdfg = current_state.sdfg
        nsdfg_node = owning_sdfg.parent_nsdfg_node
        parent_state = owning_sdfg.parent
        if nsdfg_node is None or parent_state is None:
            break
        current_node = nsdfg_node
        current_state = parent_state
    return chain


def _iter_cutile_scopes(sdfg: SDFG) -> Iterator[Tuple[nodes.MapEntry, SDFGState]]:
    """Iterate over every CuTile-scheduled MapEntry in ``sdfg``, recursively.

    :param sdfg: SDFG to search (NestedSDFGs are included).
    :returns: Iterator of ``(map_entry, state)`` pairs.
    """
    for node, graph in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.CuTile:
            yield node, graph


def _is_power_of_two(value: int) -> bool:
    """Whether ``value`` is a positive power of two.

    :param value: The integer to test.
    :returns: ``True`` iff ``value`` is in ``{1, 2, 4, 8, ...}``.
    """
    return value > 0 and (value & (value - 1)) == 0


def _warn_or_raise(message: str, strict: bool) -> None:
    """Emit ``message`` as a ``UserWarning``, or raise when ``strict``.

    :param message: The diagnostic message.
    :param strict: When ``True``, raise ``ValueError(message)`` instead of
        warning.
    :raises ValueError: When ``strict`` is ``True``.
    """
    if strict:
        raise ValueError(message)
    warnings.warn(message)


@properties.make_properties
class _CuTileLoweringPass(ppl.Pass):
    """Shared base of the five cuTile lowering passes.

    Provides the ``strict`` knob (precondition violations warn by default and
    raise when ``strict=True``) and the common Pass plumbing.
    """

    CATEGORY: str = "Vectorization"

    strict = properties.Property(dtype=bool,
                                 default=False,
                                 desc="When True, precondition violations raise ValueError "
                                 "instead of emitting a UserWarning.")

    def __init__(self, strict: bool = False):
        """Initialize the pass.

        :param strict: When ``True``, precondition violations raise
            ``ValueError`` instead of emitting a ``UserWarning``.
        """
        super().__init__()
        self.strict = strict

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return bool(modified & ppl.Modifies.Everything)

    def depends_on(self) -> Set[Type[ppl.Pass]]:
        return set()


@properties.make_properties
@transformation.explicit_cf_compatible
class CuTileValidateTiles(_CuTileLoweringPass):
    """Precondition audit for the cuTile lowering sequence.

    Checks that (a) at least one tileops library node exists (the anchors all
    later passes stamp from), and (b) every anchor's ``widths`` are powers of
    two — a hard ``cuda.tile`` runtime requirement that raises ``ValueError``
    unconditionally (not gated by ``strict``).
    """

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Validate the tile-op anchors of ``sdfg``.

        :param sdfg: The SDFG to validate (not modified).
        :param pipeline_results: Unused pipeline results.
        :returns: The number of tileops anchors found, or ``None`` if there
            are none (after warning / raising per ``strict``).
        :raises ValueError: If any anchor has a non-power-of-2 tile width
            (always), or — when ``strict`` — if no anchors exist.
        """
        anchors = _collect_tile_nodes(sdfg)
        if not anchors:
            _warn_or_raise(
                "CuTileValidateTiles: no tileops library nodes found; run "
                "VectorizeCPUMultiDim(target_isa='CUTILE', expand_tile_nodes=False) first", self.strict)
            return None
        for node, _ in anchors:
            for width in node.widths:
                if not _is_power_of_two(width):
                    raise ValueError(f"{type(node).__name__} '{node.label}': tile width {width} in "
                                     f"widths={list(node.widths)} is not a power of 2 "
                                     "(cuda.tile runtime requirement)")
        return len(anchors)


@properties.make_properties
@transformation.explicit_cf_compatible
class CuTileSetSchedules(_CuTileLoweringPass):
    """Tileops-anchored schedule stamping.

    For every tileops anchor, the chain of enclosing maps (walked across
    NestedSDFG boundaries) is stamped: the outermost MapEntry becomes the
    cuTile kernel (``ScheduleType.CuTile``); every inner MapEntry in the chain
    becomes ``ScheduleType.Sequential``. Maps that enclose no anchor are left
    untouched; a trailing audit warns about any map that is neither CuTile,
    nor Sequential, nor inside a CuTile scope (a partially-vectorized SDFG —
    such maps are emitted as plain Python loops by the Python backend).
    """

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Stamp kernel and sequential schedules anchored on tile-op nodes.

        :param sdfg: The SDFG to stamp in place.
        :param pipeline_results: Unused pipeline results.
        :returns: The number of maps stamped ``ScheduleType.CuTile``, or
            ``None`` if there were no anchors / no enclosing maps.
        :raises ValueError: When ``strict`` and a precondition or the trailing
            audit fails.
        """
        anchors = _collect_tile_nodes(sdfg)
        if not anchors:
            _warn_or_raise(
                "CuTileSetSchedules: no tileops library nodes found; run "
                "VectorizeCPUMultiDim(target_isa='CUTILE', expand_tile_nodes=False) first", self.strict)
            return None

        scope_cache: _ScopeCache = {}
        cutile_entries: Set[nodes.MapEntry] = set()
        for node, state in anchors:
            chain = _enclosing_map_chain(node, state, scope_cache)
            if not chain:
                _warn_or_raise(
                    f"CuTileSetSchedules: tileops node '{node.label}' has no enclosing map at any "
                    "level; it cannot be placed inside a cuTile kernel", self.strict)
                continue
            for entry, _ in chain[:-1]:
                entry.map.schedule = dtypes.ScheduleType.Sequential
            outermost_entry = chain[-1][0]
            outermost_entry.map.schedule = dtypes.ScheduleType.CuTile
            cutile_entries.add(outermost_entry)

        # Trailing audit: maps untouched by the anchored stamping.
        for node, graph in sdfg.all_nodes_recursive():
            if not isinstance(node, nodes.MapEntry):
                continue
            if node.map.schedule in (dtypes.ScheduleType.CuTile, dtypes.ScheduleType.Sequential):
                continue
            chain = _enclosing_map_chain(node, graph, scope_cache)
            if any(entry.map.schedule == dtypes.ScheduleType.CuTile for entry, _ in chain):
                continue
            _warn_or_raise(
                f"CuTileSetSchedules: map '{node.map.label}' encloses no tileops node and keeps "
                f"schedule {node.map.schedule.name} (partially-vectorized SDFG; it will be emitted "
                "as plain Python loops by the Python backend)", self.strict)

        return len(cutile_entries) if cutile_entries else None


@properties.make_properties
@transformation.explicit_cf_compatible
class CuTileSetTileStorage(_CuTileLoweringPass):
    """Stamp tile/mask transients with ``StorageType.CuTile_Tile``.

    Three rules, applied in order:

    a. **Anchored:** every AccessNode adjacent to a tileops anchor whose
       descriptor (in its owning SDFG) is a transient ``data.Array`` with
       ``StorageType.Register`` becomes ``CuTile_Tile``. Scalars are left
       untouched (e.g. TileReduce-adjacent accumulator Scalars stay Register).
    b. **NSDFG boundary propagation:** for NestedSDFGs inside a CuTile scope,
       a ``CuTile_Tile`` array on one side of a connector stamps the
       descriptor on the other side too (outer -> inner unconditionally on
       Arrays — the connector view of a tile is a tile; inner -> outer only on
       Register Array transients). Iterated to a fixpoint.
    c. **Recursive fallback:** any remaining Register Array transient whose
       AccessNode sits (transitively through NSDFG parents) inside a CuTile
       scope becomes ``CuTile_Tile``; rule (b) is then re-run once more.
    """

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Stamp tile-array storage inside cuTile kernels.

        :param sdfg: The SDFG to stamp in place.
        :param pipeline_results: Unused pipeline results.
        :returns: The number of data descriptors whose storage was changed to
            ``CuTile_Tile``, or ``None`` if none were (e.g. on a repeated
            run, or when the precondition fails).
        :raises ValueError: When ``strict`` and no CuTile-scheduled map
            exists.
        """
        if next(_iter_cutile_scopes(sdfg), None) is None:
            _warn_or_raise("CuTileSetTileStorage: no CuTile-scheduled map found; run CuTileSetSchedules first",
                           self.strict)
            return None

        scope_cache: _ScopeCache = {}
        stamped: Set[int] = set()  # id() of changed descriptors

        def _stamp(desc: data.Data) -> None:
            """Set ``desc.storage = CuTile_Tile`` and record the change."""
            if desc.storage != dtypes.StorageType.CuTile_Tile:
                desc.storage = dtypes.StorageType.CuTile_Tile
                stamped.add(id(desc))

        # Rule (a): AccessNodes adjacent to tileops anchors.
        for node, state in _collect_tile_nodes(sdfg):
            owning_arrays = state.sdfg.arrays
            neighbors = [e.src for e in state.in_edges(node)] + [e.dst for e in state.out_edges(node)]
            for neighbor in neighbors:
                if not isinstance(neighbor, nodes.AccessNode):
                    continue
                desc = owning_arrays.get(neighbor.data)
                if (isinstance(desc, data.Array) and desc.transient
                        and desc.storage == dtypes.StorageType.Register):
                    _stamp(desc)

        # Rule (b): NSDFG boundary propagation (fixpoint).
        def _propagate_nsdfg_boundaries() -> None:
            """Sync CuTile_Tile storage across NestedSDFG connectors until a
            fixpoint is reached."""
            changed = True
            while changed:
                changed = False
                before = len(stamped)
                for node, graph in sdfg.all_nodes_recursive():
                    if not isinstance(node, nodes.NestedSDFG):
                        continue
                    chain = _enclosing_map_chain(node, graph, scope_cache)
                    if not any(entry.map.schedule == dtypes.ScheduleType.CuTile for entry, _ in chain):
                        continue
                    outer_arrays = graph.sdfg.arrays
                    inner_arrays = node.sdfg.arrays
                    boundary = ([(e.dst_conn, e) for e in graph.in_edges(node)] +
                                [(e.src_conn, e) for e in graph.out_edges(node)])
                    for conn, edge in boundary:
                        if conn is None or edge.data.data is None:
                            continue
                        outer_desc = outer_arrays.get(edge.data.data)
                        inner_desc = inner_arrays.get(conn)
                        if outer_desc is None or inner_desc is None:
                            continue
                        if (outer_desc.storage == dtypes.StorageType.CuTile_Tile and isinstance(inner_desc, data.Array)
                                and inner_desc.storage != dtypes.StorageType.CuTile_Tile):
                            _stamp(inner_desc)
                        elif (inner_desc.storage == dtypes.StorageType.CuTile_Tile
                              and isinstance(outer_desc, data.Array) and outer_desc.transient
                              and outer_desc.storage == dtypes.StorageType.Register):
                            _stamp(outer_desc)
                changed = len(stamped) > before

        _propagate_nsdfg_boundaries()

        # Rule (c): recursive fallback for Register Array transients inside
        # CuTile scopes (transitively through NSDFG parents).
        for node, graph in sdfg.all_nodes_recursive():
            if not isinstance(node, nodes.AccessNode):
                continue
            desc = graph.sdfg.arrays.get(node.data)
            if not (isinstance(desc, data.Array) and desc.transient
                    and desc.storage == dtypes.StorageType.Register):
                continue
            chain = _enclosing_map_chain(node, graph, scope_cache)
            if any(entry.map.schedule == dtypes.ScheduleType.CuTile for entry, _ in chain):
                _stamp(desc)

        # Rule (c) may have created new boundary mismatches; settle them.
        _propagate_nsdfg_boundaries()

        return len(stamped) if stamped else None


@properties.make_properties
@transformation.explicit_cf_compatible
class CuTileSetGlobalStorage(_CuTileLoweringPass):
    """Stamp kernel-touched non-transient Arrays with ``GPU_Global``.

    Only non-transients actually accessed inside a CuTile kernel scope
    (including transitively through NestedSDFG connectors) are stamped; other
    non-transients are left untouched (they resolve to host storage at
    codegen). Non-transient Scalars are left untouched as well — cuTile
    kernels take scalars as plain Python arguments. When a touched
    non-transient lives in a nested SDFG, the stamp is propagated outward
    through the NSDFG connector mapping up to the top-level argument array.

    Descriptors whose storage is already ``StorageType.CuTile_Tile`` are never
    overwritten (neither directly nor during outward propagation) — those are
    owned by :class:`CuTileSetTileStorage`. In particular, a non-transient
    tile-connector array inside a NestedSDFG (e.g. the connector view of a
    mask tile) keeps ``CuTile_Tile`` storage; skipped descriptors are not
    counted in the return value.
    """

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Stamp global-array storage for cuTile kernel operands.

        :param sdfg: The SDFG to stamp in place.
        :param pipeline_results: Unused pipeline results.
        :returns: The number of data descriptors whose storage was changed to
            ``GPU_Global``, or ``None`` if none were.
        :raises ValueError: When ``strict`` and no CuTile-scheduled map
            exists.
        """
        cutile_scopes = list(_iter_cutile_scopes(sdfg))
        if not cutile_scopes:
            _warn_or_raise("CuTileSetGlobalStorage: no CuTile-scheduled map found; run CuTileSetSchedules first",
                           self.strict)
            return None

        # Collect (owning_sdfg, data_name) pairs touched inside any kernel.
        touched: List[Tuple[SDFG, str]] = []
        seen: Set[Tuple[int, str]] = set()

        def _add(owning_sdfg: SDFG, name: Optional[str]) -> None:
            """Record a touched data name of ``owning_sdfg`` (deduplicated)."""
            if name is None:
                return
            key = (id(owning_sdfg), name)
            if key not in seen:
                seen.add(key)
                touched.append((owning_sdfg, name))

        def _collect_nsdfg(nsdfg_node: nodes.NestedSDFG) -> None:
            """Collect every data name accessed anywhere inside ``nsdfg_node``
            (all inner states execute inside the kernel), recursively."""
            inner_sdfg = nsdfg_node.sdfg
            for inner_state in inner_sdfg.states():
                for inner_node in inner_state.nodes():
                    if isinstance(inner_node, nodes.AccessNode):
                        _add(inner_sdfg, inner_node.data)
                    elif isinstance(inner_node, nodes.NestedSDFG):
                        _collect_nsdfg(inner_node)
                for inner_edge in inner_state.edges():
                    _add(inner_sdfg, inner_edge.data.data)

        for entry, state in cutile_scopes:
            owning_sdfg = state.sdfg
            subgraph = state.scope_subgraph(entry, include_entry=True, include_exit=True)
            for node in subgraph.nodes():
                if isinstance(node, nodes.AccessNode):
                    _add(owning_sdfg, node.data)
                elif isinstance(node, nodes.NestedSDFG):
                    _collect_nsdfg(node)
            for edge in subgraph.edges():
                # ``SubgraphView.edges()`` only yields edges with both
                # endpoints inside the scope, so the global container name is
                # captured here via the entry node's inner out-edge memlets
                # (whose ``.data`` references the global array), not via any
                # boundary edge crossing the scope.
                _add(owning_sdfg, edge.data.data)

        stamped: Set[int] = set()  # id() of changed descriptors

        def _stamp(desc: data.Data) -> None:
            """Set ``desc.storage = GPU_Global`` and record the change.

            Descriptors already ``CuTile_Tile`` (owned by
            :class:`CuTileSetTileStorage`) are left untouched and not counted.
            """
            if desc.storage in (dtypes.StorageType.GPU_Global, dtypes.StorageType.CuTile_Tile):
                return
            desc.storage = dtypes.StorageType.GPU_Global
            stamped.add(id(desc))

        for owning_sdfg, name in touched:
            desc = owning_sdfg.arrays.get(name)
            if desc is None or desc.transient or not isinstance(desc, data.Array):
                continue
            if desc.storage == dtypes.StorageType.CuTile_Tile:
                # Tile-connector view stamped by CuTileSetTileStorage; not a
                # global array — skip it (and do not propagate outward).
                continue
            _stamp(desc)
            # Propagate outward through NSDFG connector mappings up to the
            # top-level argument array.
            current_sdfg, current_name = owning_sdfg, name
            while current_sdfg.parent_nsdfg_node is not None and current_sdfg.parent is not None:
                nsdfg_node = current_sdfg.parent_nsdfg_node
                parent_state = current_sdfg.parent
                parent_sdfg = parent_state.sdfg
                outer_name: Optional[str] = None
                for edge in parent_state.in_edges(nsdfg_node):
                    if edge.dst_conn == current_name and edge.data.data is not None:
                        outer_name = edge.data.data
                        break
                if outer_name is None:
                    for edge in parent_state.out_edges(nsdfg_node):
                        if edge.src_conn == current_name and edge.data.data is not None:
                            outer_name = edge.data.data
                            break
                if outer_name is None:
                    break
                outer_desc = parent_sdfg.arrays.get(outer_name)
                if (outer_desc is None or outer_desc.transient or not isinstance(outer_desc, data.Array)
                        or outer_desc.storage == dtypes.StorageType.CuTile_Tile):
                    break
                _stamp(outer_desc)
                current_sdfg, current_name = parent_sdfg, outer_name

        return len(stamped) if stamped else None


@properties.make_properties
@transformation.explicit_cf_compatible
class CuTileSetImplementations(_CuTileLoweringPass):
    """Stamp every tileops node with the cuTile expansion, directly.

    Sets ``node.target_isa = "CUTILE"`` and ``node.implementation = "cutile"``
    on every tileops library node — no ``select_tile_implementation`` dispatch
    (cuTile selection is unconditional). A tile-op node type without a
    ``'cutile'`` implementation raises ``ValueError`` (loud failure instead of
    a silent ``'pure'`` fallback, which would emit CPP tasklets into a Python
    backend kernel). Must run before ``sdfg.expand_library_nodes()``.
    """

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Stamp the cuTile implementation on every tileops node.

        :param sdfg: The SDFG whose library nodes are stamped in place.
        :param pipeline_results: Unused pipeline results.
        :returns: The number of library nodes stamped, or ``None`` if there
            are none (e.g. when run after ``expand_library_nodes()``).
        :raises ValueError: If a tileops node lacks a ``'cutile'``
            implementation (always), or — when ``strict`` — if no tileops
            nodes exist.
        """
        anchors = _collect_tile_nodes(sdfg)
        if not anchors:
            _warn_or_raise(
                "CuTileSetImplementations: no tileops library nodes found (already expanded?); "
                "this pass must run before sdfg.expand_library_nodes()", self.strict)
            return None
        for node, _ in anchors:
            if "cutile" not in node.implementations:
                raise ValueError(f"{type(node).__name__} '{node.label}' has no 'cutile' implementation; "
                                 "a tileops node type must ship a cuTile expansion before it can be "
                                 "lowered for the cuTile backend")
            node.target_isa = "CUTILE"
            node.implementation = "cutile"
        return len(anchors)


@properties.make_properties
@transformation.explicit_cf_compatible
class CuTileInsertDataCopies(_CuTileLoweringPass):
    """Insert host-to-device and device-to-host copy states around cuTile
    computation, so callers can pass NumPy (host) arrays instead of CuPy
    (device) arrays.

    The pass operates on the top-level SDFG only and targets non-transient
    ``data.Array`` descriptors whose storage was stamped ``GPU_Global`` by
    :class:`CuTileSetGlobalStorage`. For each such array the pass:

    1. Creates a ``gpu_<name>`` transient clone with ``GPU_Global`` storage.
    2. Reverts the original descriptor to ``CPU_Heap`` (host-accessible).
    3. Replaces every in-graph reference (``AccessNode.data``, ``Memlet.data``,
       interstate edge expressions) to use the clone.
    4. Inserts a **copy-in state** before the current start block that copies
       every candidate array from host to device (conservative — avoids
       uninitialized device memory for partial writes).
    5. Inserts a **copy-out state** after all sink nodes that copies written
       candidates back from device to host.

    The pass is idempotent: a second run finds no ``GPU_Global`` non-transients
    (they were already cloned and reverted to ``CPU_Heap``) and returns
    ``None``.

    Must run after :class:`CuTileSetGlobalStorage` (the source of
    ``GPU_Global`` stamps on non-transients). Has no effect on transients,
    Scalars, or ``CuTile_Tile`` descriptors.
    """

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Clone GPU_Global non-transients and insert copy-in / copy-out states.

        :param sdfg: The top-level SDFG to transform in place.
        :param pipeline_results: Unused pipeline results.
        :returns: The number of arrays cloned (with copy states inserted), or
            ``None`` if nothing was done (no candidates, or already applied).
        :raises ValueError: When ``strict`` and no CuTile-scheduled map exists
            (the pass is pointless without kernel computation).
        """
        # Precondition: at least one CuTile-scheduled map must exist.
        if next(_iter_cutile_scopes(sdfg), None) is None:
            _warn_or_raise(
                "CuTileInsertDataCopies: no CuTile-scheduled map found; "
                "run CuTileSetSchedules first", self.strict)
            return None

        # -- Step 1: Identify candidates ---------------------------------
        # Non-transient Arrays with GPU_Global storage (stamped by
        # CuTileSetGlobalStorage).  Scalars and CuTile_Tile are skipped.
        candidates: Dict[str, data.Data] = {}
        for name, desc in sdfg.arrays.items():
            if (not desc.transient and isinstance(desc, data.Array)
                    and desc.storage == dtypes.StorageType.GPU_Global):
                candidates[name] = desc

        if not candidates:
            return None

        # -- Step 2: Classify as input / output --------------------------
        input_names: Set[str] = set()
        output_names: Set[str] = set()
        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode) and node.data in candidates:
                    if state.out_degree(node) > 0:
                        input_names.add(node.data)
                    if state.in_degree(node) > 0:
                        output_names.add(node.data)

        # Conservative: include all candidates in copy-in to avoid
        # uninitialized device memory for partial writes.
        copyin_names = set(candidates.keys())
        copyout_names = output_names

        # -- Step 3: Clone arrays ----------------------------------------
        cloned: Dict[str, str] = {}  # original name -> gpu clone name
        for name, desc in candidates.items():
            newdesc = desc.clone()
            newdesc.storage = dtypes.StorageType.GPU_Global
            newdesc.transient = True
            gpu_name = sdfg.add_datadesc('gpu_' + name, newdesc,
                                         find_new_name=True)
            cloned[name] = gpu_name

        # Revert originals to CPU_Heap.
        for name in cloned:
            sdfg.arrays[name].storage = dtypes.StorageType.CPU_Heap

        # -- Step 4: Replace all internal references ---------------------
        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, nodes.AccessNode) and node.data in cloned:
                    node.data = cloned[node.data]
            for edge in state.edges():
                if edge.data.data in cloned:
                    edge.data.data = cloned[edge.data.data]

        # Interstate edges (condition / assignment expressions).
        for edge in sdfg.all_interstate_edges():
            for orig, gpu in cloned.items():
                edge.data.replace(orig, gpu)

        # -- Step 5: Create copy-in state --------------------------------
        start_block = sdfg.start_block
        copyin_state = sdfg.add_state(sdfg.label + '_copyin')
        # Wire copyin -> old start block.
        sdfg.add_edge(copyin_state, start_block, InterstateEdge())
        # Make copyin the new start.
        sdfg.start_block = sdfg.node_id(copyin_state)
        for name in sorted(copyin_names):
            gpu_name = cloned[name]
            src = nodes.AccessNode(name)
            dst = nodes.AccessNode(gpu_name)
            copyin_state.add_node(src)
            copyin_state.add_node(dst)
            copyin_state.add_nedge(
                src, dst,
                mmlt.Memlet.from_array(name, sdfg.arrays[name]))

        # -- Step 6: Create copy-out state -------------------------------
        if copyout_names:
            copyout_state = sdfg.add_state(sdfg.label + '_copyout')
            # Connect every sink to the copy-out state.
            # Recompute sink nodes excluding the copyout state itself.
            for sink in sdfg.sink_nodes():
                if sink is not copyout_state:
                    sdfg.add_edge(sink, copyout_state, InterstateEdge())
            for name in sorted(copyout_names):
                gpu_name = cloned[name]
                src = nodes.AccessNode(gpu_name)
                dst = nodes.AccessNode(name)
                copyout_state.add_node(src)
                copyout_state.add_node(dst)
                copyout_state.add_nedge(
                    src, dst,
                    mmlt.Memlet.from_array(gpu_name, sdfg.arrays[gpu_name]))

        return len(cloned)
