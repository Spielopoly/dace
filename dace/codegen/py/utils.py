import ast
from typing import TYPE_CHECKING

from dace import data, dtypes, subsets, symbolic

_PY_CAST_TARGETS = {
    name: dtypes.PYTHON_TYPES.get(typeclass.type, f"dace.{name}")
    for typeclass, type_string in dtypes.TYPECLASS_TO_STRING.items() if (name := type_string.split("::")[-1])
}
# ``dace.bool`` is an alias of ``dace.bool_``. NumPy only exposes ``bool_``.
_PY_CAST_TARGETS["bool"] = "numpy.bool_"


class _BoundNameCollector(ast.NodeVisitor):
    """Collect names that can shadow serialized bare dtype calls."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.add(node.id)

    def visit_arg(self, node: ast.arg) -> None:
        self.names.add(node.arg)

    def visit_alias(self, node: ast.alias) -> None:
        self.names.add(node.asname or node.name.split(".", maxsplit=1)[0])

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name is not None:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name is not None:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name is not None:
            self.names.add(node.name)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest is not None:
            self.names.add(node.rest)
        self.generic_visit(node)


class _PythonDtypeCastRewriter(ast.NodeTransformer):
    """Make dtype calls safe for Python, NumPy, and CuPy scalar values."""

    def __init__(self, bound_names: set[str]) -> None:
        self.changed = False
        self._bound_names = bound_names

    def visit_Call(self, node: ast.Call) -> ast.expr:
        node = self.generic_visit(node)
        dtype_name = None
        if (isinstance(node.func, ast.Name) and node.func.id in _PY_CAST_TARGETS
                and node.func.id not in self._bound_names):
            dtype_name = node.func.id
        elif (isinstance(node.func, ast.Attribute) and node.func.attr in _PY_CAST_TARGETS
              and isinstance(node.func.value, ast.Name) and node.func.value.id in {"dace", "numpy", "np"}):
            dtype_name = node.func.attr
        if dtype_name is None or len(node.args) != 1 or node.keywords:
            return node
        self.changed = True

        def _value() -> ast.Name:
            return ast.Name(id="__dace_value", ctx=ast.Load())

        normalized = ast.IfExp(
            test=ast.Call(func=ast.Name(id="hasattr", ctx=ast.Load()),
                          args=[_value(), ast.Constant(value="item")],
                          keywords=[]),
            body=ast.Call(func=ast.Attribute(value=_value(), attr="item", ctx=ast.Load()), args=[], keywords=[]),
            orelse=_value(),
        )
        cast_target = ast.parse(_PY_CAST_TARGETS[dtype_name], mode="eval").body
        cast = ast.Call(func=cast_target, args=[normalized], keywords=[])
        cast_lambda = ast.Lambda(
            args=ast.arguments(posonlyargs=[],
                               args=[ast.arg(arg="__dace_value")],
                               vararg=None,
                               kwonlyargs=[],
                               kw_defaults=[],
                               kwarg=None,
                               defaults=[]),
            body=cast,
        )
        return ast.copy_location(ast.Call(func=cast_lambda, args=node.args, keywords=[]), node)


def rewrite_dtype_casts(body: str) -> str:
    """Qualify DaCe dtype calls and safely unwrap array scalar values."""
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return body
    bound_names = _BoundNameCollector()
    bound_names.visit(tree)
    rewriter = _PythonDtypeCastRewriter(bound_names.names)
    tree = rewriter.visit(tree)
    if not rewriter.changed:
        return body
    return ast.unparse(ast.fix_missing_locations(tree))


if TYPE_CHECKING:
    from dace.codegen.py.framecode import DaCePythonCodeGenerator
    from dace.codegen.py.target import PythonTargetCodeGenerator


def python_symbolic(expr) -> str:
    return symbolic.symstr(expr, cpp_mode=False)


def _python_slice_component(start, end, step) -> str:
    try:
        start_int = int(start)
        end_int = int(end)
        step_int = int(step)
    except (TypeError, ValueError):
        start_int = end_int = step_int = None

    if step_int is not None:
        if step_int == 1 and start_int == end_int:
            return str(start_int)
        stop_expr = str(end_int + 1) if step_int > 0 else str(end_int - 1)
        if step_int == 1:
            return f'{start_int}:{stop_expr}'
        return f'{start_int}:{stop_expr}:{step_int}'

    start_expr = python_symbolic(start)
    end_expr = python_symbolic(end)
    step_expr = python_symbolic(step)
    if step_expr == '1' and start_expr == end_expr:
        return start_expr
    stop_expr = f'(({end_expr}) + (1 if ({step_expr}) > 0 else -1))'
    if step_expr == '1':
        return f'{start_expr}:{stop_expr}'
    return f'{start_expr}:{stop_expr}:{step_expr}'


def subset_to_python_indices(desc: data.Data, subset: subsets.Subset | None) -> str:
    if subset is None:
        return ''

    if hasattr(desc, 'offset') and any(str(offset) != '0' for offset in desc.offset):
        subset = subset.offset_new(desc.offset, False)

    if isinstance(subset, subsets.Indices):
        parts = [python_symbolic(index) for index in subset.indices]
    elif isinstance(subset, subsets.Range):
        parts = [_python_slice_component(start, end, step) for start, end, step in subset.ranges]
    else:
        raise NotImplementedError(f'Unsupported subset type for Python backend: {type(subset).__name__}')

    if len(parts) == 1:
        return parts[0]
    return ', '.join(parts)


def data_access_expression(name: str,
                           desc: data.Data,
                           subset: subsets.Subset | None = None,
                           scalar_buffer: bool = False,
                           is_write: bool = False) -> str:
    if isinstance(desc, data.Scalar):
        if scalar_buffer:
            # Scalar buffers are 0-d numpy arrays. Reads use ``[()]`` so the
            # value is a numpy scalar (usable in cupy expressions); write
            # targets use ``[...]`` so assignment mutates the buffer in place.
            return f'{name}[...]' if is_write else f'{name}[()]'
        return name

    indices = subset_to_python_indices(desc, subset)
    if not indices:
        return name
    return f'{name}[{indices}]'


def numpy_array_expression(sdfg,
                           memlet,
                           with_brackets=True,
                           offset=None,
                           relative_offset=True,
                           packed_veclen=1,
                           use_other_subset=False,
                           indices=None,
                           referenced_array=None,
                           codegen: 'PythonTargetCodeGenerator | None' = None,
                           framecode: 'DaCePythonCodeGenerator | None' = None):
    del offset, relative_offset, packed_veclen, indices, framecode
    subset = memlet.other_subset if use_other_subset else memlet.subset
    desc = sdfg.arrays[memlet.data] if referenced_array is None else referenced_array
    name = codegen.ptr(memlet.data, desc, sdfg, subset=subset) if codegen is not None else memlet.data
    if not with_brackets:
        return subset_to_python_indices(desc, subset)
    return data_access_expression(name, desc, subset)


def numpy_offset_expression(d: data.Data, subset_in: subsets.Subset) -> str:
    return subset_to_python_indices(d, subset_in)
