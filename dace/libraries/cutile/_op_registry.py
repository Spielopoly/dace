"""
Operation registry for cuTile transformations.

Maps tasklet code patterns to cuTile library node classes. To add a new
operation, implement a matcher function and decorate it with @register_matcher.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Type

from dace import dtypes
from dace.sdfg.nodes import LibraryNode, Tasklet


@dataclass
class TileOpMatch:
    """Result of matching a tasklet to a tile operation."""
    library_node_class: Type[LibraryNode]
    in_conn_map: Dict[str, str]   # tasklet_conn → lib_node_conn
    out_conn_map: Dict[str, str]  # tasklet_conn → lib_node_conn
    op_name: str


_MATCHERS: List[Callable[[Tasklet], tuple[Optional[TileOpMatch], Optional[TileOpMatch]]]] = []


def register_matcher(func: Callable[[Tasklet], tuple[Optional[TileOpMatch], Optional[TileOpMatch]]]) -> Callable[[Tasklet], tuple[Optional[TileOpMatch], Optional[TileOpMatch]]]:
    """Decorator: register a function that matches tasklets to tile ops."""
    _MATCHERS.append(func)
    return func


def match_tasklet(tasklet: Tasklet) -> Optional[tuple[Optional[TileOpMatch], Optional[TileOpMatch]]]:
    """Try registered matchers in order; return first match or None.
    
    If a match is found, returns a tuple of (unmasked_match, masked_match), where each is either a TileOpMatch or None."""
    for matcher in _MATCHERS:
        result = matcher(tasklet)
        if result is not None and (result[0] is not None or result[1] is not None):
            return result
    return None


# ---------------------------------------------------------------------------
# Helpers for common patterns
# ---------------------------------------------------------------------------

def _parse_simple_binop(
    tasklet: Tasklet,
    op_ast_type: type,
    op_name: str,
    lib_node_class: Type[LibraryNode],
    masked_lib_node_class: Type[LibraryNode],
) -> tuple[Optional[TileOpMatch], Optional[TileOpMatch]]:
    """Match a tasklet whose code is ``out = in1 <op> in2``."""
    if tasklet.language != dtypes.Language.Python:
        return None, None
    code = tasklet.code.as_string.strip()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None, None

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return None, None
    assign = tree.body[0]
    if len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name):
        return None, None
    if not isinstance(assign.value, ast.BinOp):
        return None, None
    if not isinstance(assign.value.op, op_ast_type):
        return None, None
    if not isinstance(assign.value.left, ast.Name) or not isinstance(assign.value.right, ast.Name):
        return None, None

    out_name = assign.targets[0].id
    in1_name = assign.value.left.id
    in2_name = assign.value.right.id

    if out_name not in tasklet.out_connectors:
        return None, None
    if in1_name not in tasklet.in_connectors or in2_name not in tasklet.in_connectors:
        return None, None

    return TileOpMatch(
        library_node_class=lib_node_class,
        in_conn_map={in1_name: "_a", in2_name: "_b"},
        out_conn_map={out_name: "_c"},
        op_name=op_name,
    ), TileOpMatch(
        library_node_class=masked_lib_node_class,
        in_conn_map={in1_name: "_a", in2_name: "_b"},
        out_conn_map={out_name: "_c"},
        op_name=op_name,
    )



# ---------------------------------------------------------------------------
# Built-in matchers
# ---------------------------------------------------------------------------

@register_matcher
def match_add(tasklet: Tasklet) -> tuple[Optional[TileOpMatch], Optional[TileOpMatch]]:
    """Match ``c = a + b``."""
    # Avoid import loops
    from dace.libraries.cutile.nodes import TileAddLibraryNode, TileMaskedAddLibraryNode
    return _parse_simple_binop(tasklet, ast.Add, "add", TileAddLibraryNode, TileMaskedAddLibraryNode)

@register_matcher
def match_subtract(tasklet: Tasklet) -> tuple[Optional[TileOpMatch], Optional[TileOpMatch]]:
    """Match ``c = a - b``."""
    from dace.libraries.cutile.nodes import TileSubtractLibraryNode, TileMaskedSubtractLibraryNode
    return _parse_simple_binop(tasklet, ast.Sub, "subtract", TileSubtractLibraryNode, TileMaskedSubtractLibraryNode)
