

import copy
from typing import Optional

import dace
from dace import subsets, dtypes
from dace.memlet import Memlet
from dace.sdfg.state import ConditionalBlock, SDFGState
from dace.sdfg.construction_utils import duplicate_condition_across_top_level_nodes
from dace.sdfg import nodes


def _duplicate_condition_for_nested_nsdfg(nsdfg_sdfg: dace.SDFG) -> bool:
    """Duplicate single-branch conditional blocks in a nested SDFG.

    Args:
        nsdfg_sdfg: The inner SDFG of a :class:`~dace.sdfg.nodes.NestedSDFG`
            node.

    Returns:
        ``True`` if any normalization was applied.
    """
    applied = False
    for cfr in nsdfg_sdfg.all_control_flow_regions():
        for node in list(cfr.nodes()):
            if isinstance(node, ConditionalBlock) and len(node.branches) == 1:
                applied |= duplicate_condition_across_top_level_nodes(cfr, node)
    return applied

def duplicate_conditions_for_whole_sdfgs(sdfg: dace.SDFG) -> bool:
    """Duplicate single-branch conditional blocks in every nested SDFG.

    Args:
        sdfg: The top-level SDFG whose nested SDFGs will be normalized.

    Returns:
        ``True`` if any normalization was applied.
    """
    applied = False
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                applied |= _duplicate_condition_for_nested_nsdfg(node.sdfg)
    return applied


def tile_subset_from_shape(tile_shape: tuple[int, ...]) -> subsets.Range:
    """Create a contiguous subset range ``[0:d-1]`` for each dimension.

    Args:
        tile_shape: Shape of the tile.

    Returns:
        A :class:`~dace.subsets.Range` covering the full tile.
    """
    return subsets.Range([(0, d - 1, 1) for d in tile_shape])


def create_tile_transient(
    sdfg: dace.SDFG,
    graph: SDFGState,
    data_name: str,
    tile_shape: tuple[int, ...],
    suffix: str = "_tile",
) -> tuple[str, nodes.AccessNode]:
    """Create a tile-sized transient array and add its access node to *graph*.

    Args:
        sdfg: The SDFG that will own the new transient.
        graph: The state to which the new access node is added.
        data_name: Name of the original data container to derive the transient
            from.
        tile_shape: Shape of the tile transient.
        suffix: Suffix appended to *data_name* for the transient name
            (default ``"_tile"``).

    Returns:
        A 2-tuple ``(tile_name, access_node)`` — the name of the new
        transient and its :class:`~dace.sdfg.nodes.AccessNode`.
    """
    original_desc = sdfg.arrays[data_name]
    tile_name, _ = sdfg.add_transient(
        data_name + suffix,
        tile_shape,
        original_desc.dtype,
        storage=original_desc.storage,
        lifetime=dtypes.AllocationLifetime.Scope,
        find_new_name=True,
    )
    tile_node = nodes.AccessNode(tile_name)
    graph.add_node(tile_node)
    return tile_name, tile_node


def is_canonical_inner_map(inner_map: nodes.Map) -> bool:
    """Check whether an inner map has canonical ranges (start 0, stride 1).

    Args:
        inner_map: The :class:`~dace.sdfg.nodes.Map` to check.

    Returns:
        ``True`` if every dimension starts at ``0`` with stride ``1``.
    """
    for r_begin, _, r_stride in inner_map.range:
        if r_begin != 0 or r_stride != 1:
            return False
    return True


def primary_memlet_subset(memlet: Memlet) -> Optional[subsets.Subset]:
    """Return the memlet subset field through src/dst-oriented accessors."""
    if memlet._is_data_src is False:
        return memlet.dst_subset
    return memlet.src_subset


def with_primary_subset(memlet: Memlet, new_subset: subsets.Subset) -> Memlet:
    """Clone *memlet* and replace the direction-dependent primary subset.

    If the orientation is unresolved (``_is_data_src is None``), both
    ``src_subset`` and ``dst_subset`` are set to deep copies of
    ``new_subset``. This keeps the cloned memlet fully initialized and valid
    regardless of which side is later interpreted as primary.
    """
    updated = copy.deepcopy(memlet)
    if updated._is_data_src is False:
        updated.dst_subset = new_subset
    elif updated._is_data_src is True:
        updated.src_subset = new_subset
    else:
        # Orientation unresolved: initialize both sides to avoid half-initialized memlets.
        updated.src_subset = copy.deepcopy(new_subset)
        updated.dst_subset = copy.deepcopy(new_subset)
    return updated


def memlet_with_primary_subset(data_name: str,
                               primary_subset: subsets.Subset,
                               *,
                               data_on_src: Optional[bool] = None) -> Memlet:
    """Construct a memlet with explicit directional semantics."""
    memlet = Memlet(data=data_name)
    if data_on_src is True:
        memlet._is_data_src = True
        # Deep-copy so each memlet owns its subset object; sharing a single
        # Range across memlets fails SDFG validation's duplicate-subset check.
        memlet.src_subset = copy.deepcopy(primary_subset)
    elif data_on_src is False:
        memlet._is_data_src = False
        memlet.dst_subset = copy.deepcopy(primary_subset)
    else:
        memlet.src_subset = copy.deepcopy(primary_subset)
        memlet.dst_subset = copy.deepcopy(primary_subset)
    return memlet
