import dace
from dace.sdfg.state import ConditionalBlock
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
    applied = False
    for state in sdfg.all_states():
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                applied |= _duplicate_condition_for_nested_nsdfg(node.sdfg)
    return applied