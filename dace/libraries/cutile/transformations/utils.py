from __future__ import annotations

import dace
from dace import subsets, dtypes
from dace.sdfg.state import ConditionalBlock, SDFGState
from dace.sdfg.construction_utils import duplicate_condition_across_top_level_nodes
from dace.sdfg import nodes


def _duplicate_condition_for_nested_nsdfg(nsdfg_sdfg: dace.SDFG) -> bool:
    """
    Parameters
    ----------
    nsdfg_sdfg : dace.SDFG
        The inner SDFG of a NestedSDFG node.

    Returns
    -------
    bool
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

    Parameters
    ----------
    sdfg : dace.SDFG
        The top-level SDFG whose nested SDFGs will be normalized.

    Returns
    -------
    bool
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

    Parameters
    ----------
    tile_shape : tuple[int, ...]
        Shape of the tile.

    Returns
    -------
    subsets.Range
        Range covering the full tile.
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

    Parameters
    ----------
    sdfg : dace.SDFG
        The SDFG that will own the new transient.
    graph : SDFGState
        The state to which the new access node is added.
    data_name : str
        Name of the original data container to derive the transient from.
    tile_shape : tuple[int, ...]
        Shape of the tile transient.
    suffix : str, optional
        Suffix appended to *data_name* for the transient name (default ``"_tile"``).

    Returns
    -------
    tuple[str, nodes.AccessNode]
        The name of the new transient and its access node.
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

    Parameters
    ----------
    inner_map : nodes.Map
        The map to check.

    Returns
    -------
    bool
        ``True`` if every dimension starts at 0 with stride 1.
    """
    for r_begin, _, r_stride in inner_map.range:
        if r_begin != 0 or r_stride != 1:
            return False
    return True
