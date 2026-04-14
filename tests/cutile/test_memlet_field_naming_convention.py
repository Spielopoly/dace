"""Guard against deprecated memlet field naming in cuTile library code."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, Optional


BANNED_MEMLET_KEYWORDS = {"subset", "other_subset"}
BANNED_DATA_ATTRIBUTE_ACCESS = {"subset", "other_subset"}


class _MemletNamingVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.memlet_ctor_names: set[str] = {"Memlet"}
        self.memlet_module_aliases: set[str] = {"dace"}
        self._scopes: list[set[str]] = [set()]
        self.violations: list[tuple[int, str]] = []

    def _push_scope(self) -> None:
        self._scopes.append(set())

    def _pop_scope(self) -> None:
        self._scopes.pop()

    def _is_memlet_var(self, name: str) -> bool:
        return any(name in scope for scope in reversed(self._scopes))

    def _set_memlet_var(self, name: str, is_memlet: bool) -> None:
        if is_memlet:
            self._scopes[-1].add(name)
        else:
            self._scopes[-1].discard(name)

    def _record(self, node: ast.AST, message: str) -> None:
        self.violations.append((node.lineno, message))

    def _is_memlet_constructor(self, func: ast.expr) -> bool:
        if isinstance(func, ast.Name):
            return func.id in self.memlet_ctor_names
        if isinstance(func, ast.Attribute) and func.attr == "Memlet":
            return isinstance(func.value, ast.Name) and func.value.id in self.memlet_module_aliases
        return False

    def _is_memlet_like_expr(self, expr: Optional[ast.expr]) -> bool:
        if expr is None:
            return False
        if isinstance(expr, ast.Name):
            return self._is_memlet_var(expr.id)
        if isinstance(expr, ast.Attribute):
            if expr.attr != "data":
                return False
            if isinstance(expr.value, ast.Name):
                return "edge" in expr.value.id
            return self._is_memlet_like_expr(expr.value)
        if isinstance(expr, ast.Call):
            return self._is_memlet_constructor(expr.func)
        return False

    @staticmethod
    def _name_targets(target: ast.expr) -> Iterable[str]:
        if isinstance(target, ast.Name):
            yield target.id
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                yield from _MemletNamingVisitor._name_targets(elt)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "dace":
                self.memlet_module_aliases.add(alias.asname or "dace")
            elif alias.name == "dace.memlet":
                self.memlet_module_aliases.add(alias.asname or "memlet")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "dace":
            for alias in node.names:
                local_name = alias.asname or alias.name
                if alias.name == "Memlet":
                    self.memlet_ctor_names.add(local_name)
                elif alias.name == "memlet":
                    self.memlet_module_aliases.add(local_name)
        elif node.module == "dace.memlet":
            for alias in node.names:
                if alias.name == "Memlet":
                    self.memlet_ctor_names.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._push_scope()
        self.generic_visit(node)
        self._pop_scope()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._push_scope()
        self.generic_visit(node)
        self._pop_scope()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._push_scope()
        self.generic_visit(node)
        self._pop_scope()

    def visit_Assign(self, node: ast.Assign) -> None:
        is_memlet_value = self._is_memlet_like_expr(node.value)
        for target in node.targets:
            for name in self._name_targets(target):
                self._set_memlet_var(name, is_memlet_value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        is_memlet_value = self._is_memlet_like_expr(node.value)
        for name in self._name_targets(node.target):
            self._set_memlet_var(name, is_memlet_value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._is_memlet_constructor(node.func):
            for kw in node.keywords:
                if kw.arg in BANNED_MEMLET_KEYWORDS:
                    self._record(node, f"Memlet keyword '{kw.arg}' is forbidden")

        if isinstance(node.func, ast.Name) and node.func.id in {"getattr", "setattr"} and len(node.args) >= 2:
            obj, attr_name = node.args[0], node.args[1]
            if isinstance(attr_name, ast.Constant) and attr_name.value in BANNED_DATA_ATTRIBUTE_ACCESS:
                if isinstance(attr_name.value, str) and self._is_memlet_like_expr(obj):
                    self._record(node, f"{node.func.id} with memlet field '{attr_name.value}' is forbidden")

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in BANNED_DATA_ATTRIBUTE_ACCESS and self._is_memlet_like_expr(node.value):
            self._record(node, f"Memlet field access '.{node.attr}' is forbidden")
        self.generic_visit(node)


def _collect_violations_from_source(source: str, filename: str = "<memory>") -> list[tuple[int, str]]:
    tree = ast.parse(source, filename=filename)
    visitor = _MemletNamingVisitor()
    visitor.visit(tree)
    return visitor.violations


def _collect_violations(file_path: Path) -> list[tuple[int, str]]:
    source = file_path.read_text(encoding="utf-8")
    return _collect_violations_from_source(source, filename=str(file_path))


def test_scanner_detects_constructor_aliases_and_memlet_var_usage() -> None:
    source = """
from dace import Memlet as M
import dace as d
from dace import memlet as dm

def f(edge):
    m = edge.data
    a = M(data='A', subset='0')
    b = d.Memlet(data='A')
    c = dm.Memlet(data='A')
    _ = m.subset
    _ = b.other_subset
    _ = getattr(m, 'subset')
    setattr(c, 'other_subset', None)
"""
    violations = _collect_violations_from_source(source)
    messages = [message for _, message in violations]

    assert any("keyword 'subset'" in msg for msg in messages)
    assert any(".subset" in msg for msg in messages)
    assert any(".other_subset" in msg for msg in messages)
    assert any("getattr" in msg and "subset" in msg for msg in messages)
    assert any("setattr" in msg and "other_subset" in msg for msg in messages)


def test_scanner_accepts_directional_memlet_fields() -> None:
    source = """
from dace import Memlet as M

def f(edge):
    m = edge.data
    x = M(data='A')
    m.src_subset = None
    m.dst_subset = None
    setattr(x, 'src_subset', None)
    _ = getattr(x, 'dst_subset')
"""
    assert _collect_violations_from_source(source) == []


def test_scanner_avoids_non_memlet_false_positive() -> None:
    source = """
class Foo:
    pass

def f(foo):
    foo.subset = 1
    _ = getattr(foo, 'other_subset')
"""
    assert _collect_violations_from_source(source) == []


def test_cutile_uses_directional_memlet_fields_only():
    repo_root = Path(__file__).resolve().parents[2]
    cutile_root = repo_root / "dace" / "libraries" / "cutile"

    all_violations: list[str] = []
    for py_file in sorted(cutile_root.rglob("*.py")):
        if "__pycache__" in py_file.parts:
            continue
        for line, message in _collect_violations(py_file):
            rel_path = py_file.relative_to(repo_root)
            all_violations.append(f"{rel_path}:{line}: {message}")

    assert not all_violations, (
        "Found deprecated memlet naming in cuTile library code:\n"
        + "\n".join(all_violations)
    )
