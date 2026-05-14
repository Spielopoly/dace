"""Spec contract for cuTile Python backend.

Encodes and decodes operation specs as marker strings embedded in
Python-language Tasklets, enabling ``CuTilePythonCodeGen`` to generate
``@ct.kernel`` functions and ``ct.launch`` calls for each operation.
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Optional

# Marker prefix written into the Python tasklet body.  Must stay in sync with
# the ``_CUTILE_MARKER`` constant in ``dace.codegen.py.cutile_target``.
# Python comments are stripped by ast.parse, so we embed the spec as a
# variable assignment that survives the AST roundtrip.
CUTILE_MARKER: str = "__CUTILE_SPEC__"


@dataclass
class CuTileSpec:
    """All information needed by ``CuTilePythonCodeGen`` to emit a kernel."""

    version: int = 1
    #: One of "unmasked" | "runtime_mask" | "symbolic_mask" |
    #: "where_select" | "if_else"
    kind: str = "unmasked"
    #: Arithmetic/comparison operator symbol (e.g. ``"+"``, ``"abs"``).
    op: str = "+"
    #: Literal Python value replacing the left/first operand (optional).
    constant1: Optional[str] = None
    #: Literal Python value replacing the right/second operand (optional).
    constant2: Optional[str] = None
    #: Fixed tile extents, e.g. ``[16, 16]``.
    tile_shape: Optional[List[int]] = None
    #: Serialised SymPy condition string (symbolic_mask only).
    mask_condition: Optional[str] = None
    #: Number of tile dimensions.
    ndim: int = 1
    #: Serialised SymPy expression for multi-op (expression) mode.
    expr_str: Optional[str] = None
    #: if_else kind – condition expression string.
    cond_str: Optional[str] = None
    #: if_else kind – true-branch expression string.
    true_str: Optional[str] = None
    #: if_else kind – false-branch expression string.
    false_str: Optional[str] = None


def encode_spec(spec: CuTileSpec) -> str:
    """Serialise *spec* to a JSON string."""
    return json.dumps(asdict(spec))


def decode_spec(code: str) -> Optional["CuTileSpec"]:
    """Extract and deserialise a :class:`CuTileSpec` from *code*.

    Looks for an assignment ``__CUTILE_SPEC__ = '<json>'`` that survives the
    ``ast.parse`` / ``ast.unparse`` roundtrip used by :class:`CodeBlock`.
    Returns ``None`` if no marker is found.
    """
    import ast as _ast
    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return None
    for stmt in tree.body:
        if (
            isinstance(stmt, _ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], _ast.Name)
            and stmt.targets[0].id == CUTILE_MARKER
            and isinstance(stmt.value, _ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            return CuTileSpec(**json.loads(stmt.value.value))
    return None


def make_marker_code(spec: CuTileSpec) -> str:
    """Return the single-line marker string to embed in a tasklet body."""
    return f"{CUTILE_MARKER} = {repr(encode_spec(spec))}"
