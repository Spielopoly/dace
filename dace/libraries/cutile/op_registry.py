"""
Operation registry for cuTile transformations.

Maps tasklet code patterns to cuTile library node classes. To add a new
operation, call ``register_op`` with the operation string, tasklet type,
mask type, library node class, and connector mapping.
"""
from __future__ import annotations
from typing import Dict, Optional, Type
from enum import Enum
from dataclasses import dataclass
from dace.sdfg.tasklet_utils import classify_tasklet, TaskletType
import dace
from dace.sdfg.nodes import LibraryNode, Tasklet

@dataclass(frozen=True)
class TaskletClassification:
    """
    - type (TaskletType): The classified tasklet type
    - lhs (str): Output connector name (left-hand side variable)
    - rhs1 (str or None):  Input connector/operand name left of the operator/first function argument
    - rhs2 (str or None): Input connector/operand name right of the operator/second function argument
    - constant1 (str or None): First constant/symbol value left of the operator/first function argument
    - constant2 (str or None): Second constant/symbol value right of the operator/second function argument
    - op (str): Operation symbol or function name
    """
    type: TaskletType
    lhs: str
    rhs1: Optional[str]
    rhs2: Optional[str]
    constant1: Optional[str]
    constant2: Optional[str]
    op: str

@dataclass(frozen=True)
class LibraryNodeInfo:
    type: Type[LibraryNode]
    node_name: str
    out: str # output connector name
    # input connector names, rhs1 from tasklet connects to rhs1 from library node, etc.
    # If None, the library node does not have that input.
    rhs1: Optional[str] = None
    rhs2: Optional[str] = None
    mask_in: Optional[str] = None  # name of the library node connector for the mask, if applicable
    out_in: Optional[str] = None  # name of the library node connector for the original output (for masked nodes), if applicable

@dataclass(frozen=True)
class TaskletLibraryNodeMatch:
    node_info: LibraryNodeInfo
    tasklet_classification: TaskletClassification

class MaskType(Enum):
    UNMASKED = "unmasked"
    RUNTIME = "runtime"

# Mapping from (operation, tasklet type, mask) to (library node class, library node name)
_OP_TO_LIBRARY_NODE: Dict[tuple[str, TaskletType, MaskType], LibraryNodeInfo] = {}


def register_op(op: str, tasklet_type: TaskletType, mask: MaskType, node_type: Type[LibraryNode], **library_node_kwargs):
    """
    Register a cuTile library node for a specific operation pattern.

    Parameters
    ----------
    op : str
        Operation symbol or function name to match (e.g., "+", "-", "*", "/").
    tasklet_type : TaskletType
        The classified tasklet type to match (e.g., ARRAY_ARRAY, UNARY_ARRAY, ARRAY_SYMBOL).
    mask : MaskType
        The mask type to match (e.g., UNMASKED, RUNTIME).
    node_type : Type[LibraryNode]
        The library node class to instantiate.
    library_node_kwargs :
        Keyword arguments used to construct a :class:`LibraryNodeInfo` instance. The following
        keys are expected:

        - ``node_name`` (str, required):
            Name of the library node to create (used as the node's label/name in the SDFG).
        - ``out`` (str, required):
            Name of the output connector of the library node that corresponds to the tasklet
            left-hand side (``lhs``).
        - ``rhs1`` (str, optional):
            Name of the first input connector of the library node. When present, the tasklet's
            ``rhs1`` connector is connected to this connector.
        - ``rhs2`` (str, optional):
            Name of the second input connector of the library node. When present, the tasklet's
            ``rhs2`` connector is connected to this connector.
        - ``mask_in`` (str, optional):
            Name of the input connector that receives a runtime boolean mask, for masked
            operations (e.g., when ``mask`` is :class:`MaskType.RUNTIME`). If not provided,
            the library node is assumed not to take a mask input.
        - ``out_in`` (str, optional):
            Name of an additional input connector that can receive the original output value
            for masked nodes (e.g., to implement "update where masked" semantics). If not
            provided, the library node is assumed not to take such an input.
    """
    _OP_TO_LIBRARY_NODE[(op, tasklet_type, mask)] = LibraryNodeInfo(type=node_type, **library_node_kwargs)


def match_tasklet_to_tile_library_node(state: dace.SDFGState, tasklet: Tasklet, mask: MaskType) -> Optional[TaskletLibraryNodeMatch]:
    """
    Match a tasklet to a tile library node class based on its code.

    Parameters
    ----------
    state : dace.SDFGState
        The state containing the tasklet.
    tasklet : dace.nodes.Tasklet
        The tasklet to match.
    mask : MaskType
        Select a masked library node variant
        Currently implemented masks are:
            - MaskType.UNMASKED (library node takes no mask argument)
            - MaskType.RUNTIME (library node takes an additional runtime boolean mask argument)

    Returns
    -------
    Optional[TaskletLibraryNodeMatch]
        A TaskletLibraryNodeMatch containing the matched library node class and tasklet classification if a match is found, otherwise None.
    """
    classification = TaskletClassification(**classify_tasklet(state, tasklet))
    
    key = (classification.op, classification.type, mask)
    if key in _OP_TO_LIBRARY_NODE:
        info = _OP_TO_LIBRARY_NODE[key]
        return TaskletLibraryNodeMatch(info, classification)
    else:
        return None