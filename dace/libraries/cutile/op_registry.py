"""
Operation registry for cuTile transformations.

Maps tasklet code patterns to cuTile library node classes. To add a new
operation, call ``register_op`` with the operation string, tasklet type,
mask type, library node class, and connector mapping.
"""
from typing import Dict, Optional, Type
from enum import Enum
from dataclasses import dataclass
from dace.sdfg.tasklet_utils import classify_tasklet, TaskletType
import dace
from dace.sdfg.nodes import LibraryNode, Tasklet

@dataclass(frozen=True)
class TaskletClassification:
    """Classified result of a single-statement tasklet.

    Attributes:
        type: The classified tasklet type.
        lhs: Output connector name (left-hand side variable).
        rhs1: Input connector / operand name for the left operand or first
            function argument, or ``None``.
        rhs2: Input connector / operand name for the right operand or second
            function argument, or ``None``.
        constant1: First constant value replacing the left operand, or ``None``.
        constant2: Second constant value replacing the right operand, or ``None``.
        op: Operation symbol or function name.
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
    """Registry entry mapping a tasklet op pattern to a library node type and its connector topology."""
    type: Type[LibraryNode]
    node_name: str
    out: str # output connector name
    # input connector names, rhs1 from tasklet connects to rhs1 from library node, etc.
    # If None, the library node does not have that input.
    rhs1: Optional[str] = None
    rhs2: Optional[str] = None
    constant1: Optional[str] = None  # first constant/symbol value (left operand), flows from TaskletClassification
    constant2: Optional[str] = None  # second constant/symbol value (right operand), flows from TaskletClassification
    mask_in: Optional[str] = None  # name of the library node connector for the mask, if applicable
    out_in: Optional[str] = None  # name of the library node connector for the original output (for masked nodes), if applicable

@dataclass(frozen=True)
class TaskletLibraryNodeMatch:
    """Matched pair of a registry entry and the tasklet classification that triggered it."""
    node_info: LibraryNodeInfo
    tasklet_classification: TaskletClassification

class MaskType(Enum):
    """Mask variant selector for cuTile library node matching.

    Determines which set of registered library nodes the op matcher considers
    when classifying a tasklet.
    """
    UNMASKED = "unmasked"
    RUNTIME = "runtime"
    SYMBOLIC = "symbolic"  # mask condition as a SymPy expression, evaluated at expansion time

# Mapping from (operation, tasklet type, mask) to LibraryNodeInfo
_OP_TO_LIBRARY_NODE: Dict[tuple[str, TaskletType, MaskType], LibraryNodeInfo] = {}


def register_op(op: str, tasklet_type: TaskletType, mask: MaskType, node_type: Type[LibraryNode], **library_node_kwargs):
    """Register a cuTile library node for a specific operation pattern.

    Args:
        op: Operation symbol or function name to match (e.g. ``"+"``,
            ``"-"``, ``"*"``, ``"/"``).
        tasklet_type: The classified tasklet type to match (e.g.
            ``ARRAY_ARRAY``, ``UNARY_ARRAY``, ``ARRAY_SYMBOL``).
        mask: The mask type to match (e.g. ``UNMASKED``, ``RUNTIME``).
        node_type: The library node class to instantiate.
        **library_node_kwargs: Keyword arguments forwarded to
            :class:`LibraryNodeInfo`.  Expected keys:

            - ``node_name`` (*str*, required): Label / name of the library
              node in the SDFG.
            - ``out`` (*str*, required): Output connector name corresponding
              to the tasklet LHS.
            - ``rhs1`` (*str*, optional): First input connector name.
            - ``rhs2`` (*str*, optional): Second input connector name.
            - ``mask_in`` (*str*, optional): Mask input connector for runtime
              masked ops.
            - ``out_in`` (*str*, optional): Pre-existing output connector for
              masked nodes that implement update-where semantics.
    """
    _OP_TO_LIBRARY_NODE[(op, tasklet_type, mask)] = LibraryNodeInfo(type=node_type, **library_node_kwargs)


# Mapping from scalar-level TaskletType to tile-level (array) equivalent.
# Used when matching scalar tasklets inside NestedSDFGs to tile library nodes.
_SCALAR_TO_ARRAY_TYPE = {
    TaskletType.SCALAR_SYMBOL: TaskletType.ARRAY_SYMBOL,
    TaskletType.SCALAR_SCALAR: TaskletType.ARRAY_ARRAY,
    TaskletType.SCALAR_ARRAY: TaskletType.SCALAR_ARRAY,      # already mixed
    TaskletType.UNARY_SCALAR: TaskletType.UNARY_ARRAY,
    TaskletType.ARRAY_SCALAR: TaskletType.ARRAY_ARRAY,
}


def match_tasklet_to_tile_library_node(state: dace.SDFGState, tasklet: Tasklet, mask: MaskType,
                                       promote_scalars: bool = False) -> Optional[TaskletLibraryNodeMatch]:
    """Match a tasklet to a tile library node class based on its code.

    Args:
        state: The state containing *tasklet*.
        tasklet: The :class:`~dace.sdfg.nodes.Tasklet` to match.
        mask: Selects the masked library node variant.  Supported values:

            - ``MaskType.UNMASKED`` – library node takes no mask argument.
            - ``MaskType.RUNTIME`` – library node takes an additional runtime
              boolean mask argument.
            - ``MaskType.SYMBOLIC`` – library node embeds a SymPy boolean
              predicate evaluated at expansion time.
        promote_scalars: When ``True``, scalar-level tasklet classifications
            (e.g. ``SCALAR_SYMBOL``) are promoted to their array-level
            equivalents (e.g. ``ARRAY_SYMBOL``) before the registry lookup.
            Useful when matching tasklets inside NestedSDFGs whose 0-D scalar
            data will be lifted to tile-level arrays.

    Returns:
        A :class:`TaskletLibraryNodeMatch` containing the matched library
        node class and tasklet classification, or ``None`` if no match is
        found.
    """
    classification = TaskletClassification(**classify_tasklet(state, tasklet))
    
    key = (classification.op, classification.type, mask)
    if key in _OP_TO_LIBRARY_NODE:
        info = _OP_TO_LIBRARY_NODE[key]
        return TaskletLibraryNodeMatch(info, classification)

    if promote_scalars:
        promoted = _SCALAR_TO_ARRAY_TYPE.get(classification.type)
        if promoted is not None:
            key2 = (classification.op, promoted, mask)
            if key2 in _OP_TO_LIBRARY_NODE:
                info = _OP_TO_LIBRARY_NODE[key2]
                return TaskletLibraryNodeMatch(info, classification)

    return None