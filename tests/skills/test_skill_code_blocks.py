"""Bind skill examples against the installed xtrax package.

Every fenced Python block under ``agent_assets/skills/**/*.md`` is parsed with
:mod:`ast`. ``from xtrax... import X`` and ``xtrax.a.b`` names are resolved on
the installed package. Each resolved call is checked with
:meth:`inspect.Signature.bind` of its positional count plus keywords.

Partial calls use :meth:`inspect.Signature.bind_partial` instead. A call is
partial only when a direct argument is the ellipsis literal ``...``
(``Trainer(...)``, ``MemoryBudget(bytes=..., estimate=fn)``). That marker is
how a skill omits arguments it is not illustrating: ``...`` is not counted as
a positional, keyword names written ``name=...`` are still checked, and
omitted required parameters are allowed. Every other resolved call is
complete: missing required arguments and the wrong positional count fail.

Calls that unpack ``*args`` or ``**kwargs`` are not signature-checked. The
unpacked names are invisible; the callee is still resolved, so an unknown
attribute on that call is reported.

One forward pass tracks assignments. ``x = <call to a resolved xtrax class>``
binds ``x`` to that class, so ``x.method(...)`` and ``x.attr`` resolve on the
class (dataclass fields included). Method calls bind the unbound signature
with ``self`` / ``cls`` removed. ``Class(...) .method(...)`` is the same
rule. Any other assignment drops the name, so ``select_bucket = print`` is
not checked against ``select_bucket``.

Fences whose info string is ``python``, ``py``, or ``python3`` are scanned,
including openers of three or more backticks.

Skip a block by placing this HTML comment on the nearest non-blank line above
the fence (blank lines between the comment and the fence are allowed):

.. code-block:: markdown

    <!-- skill-check: skip -->
    ```python
    # illustrative fragment that is not a real call
    ```
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SKILLS = ROOT / "agent_assets" / "skills"
SKIP_MARKER = "<!-- skill-check: skip -->"

# Measured corpus: 70 checked blocks, 106 signature-checked calls.
# Floors are those counts minus slack (8 and 16) and stay above the audit
# minimums of 60 blocks and 80 calls.
MIN_CHECKED_BLOCKS = 62
MIN_CHECKED_CALLS = 90

_FENCE_OPEN = re.compile(r"^(`{3,})(python3|python|py)$", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"^`{3,}$")

_NOT_OURS = object()
_BROKEN = object()
_DATA = object()
_PLACEHOLDER = object()


@dataclass(frozen=True)
class CheckResult:
    """Failures plus the counts the non-vacuity guard asserts."""

    failures: list[str]
    blocks: int
    calls: int


@dataclass(frozen=True)
class _Instance:
    """Name bound to a constructed xtrax class. Attribute lookup uses ``cls``."""

    cls: type


@dataclass(frozen=True)
class _Bound:
    """Callable reached through an instance. Drop a leading ``self`` or ``cls``."""

    fn: object


def _skill_markdown(root: Path) -> list[Path]:
    return sorted(root.rglob("*.md"))


def _nearest_nonblank(lines: list[str], index: int) -> str | None:
    cursor = index - 1
    while cursor >= 0 and not lines[cursor].strip():
        cursor -= 1
    if cursor < 0:
        return None
    return lines[cursor].strip()


def iter_python_blocks(root: Path) -> list[tuple[Path, int, str, bool]]:
    """Return ``(path, first_content_line, source, skipped)`` for each Python fence.

    Info strings ``python``, ``py``, and ``python3`` match, and the opener may
    be three or more backticks. The closing fence must be at least as long as
    the opener. ``first_content_line`` is 1-based in ``path``. ``skipped`` is
    true when ``<!-- skill-check: skip -->`` is the nearest non-blank line
    above the fence.
    """
    blocks: list[tuple[Path, int, str, bool]] = []
    for path in _skill_markdown(root):
        lines = path.read_text(encoding="utf-8").splitlines()
        index = 0
        while index < len(lines):
            match = _FENCE_OPEN.match(lines[index].strip())
            if match is None:
                index += 1
                continue
            opener = len(match.group(1))
            skipped = _nearest_nonblank(lines, index) == SKIP_MARKER
            body: list[str] = []
            index += 1
            start = index + 1  # 1-based line of the first body line
            while index < len(lines):
                closing = lines[index].strip()
                if _FENCE_CLOSE.match(closing) and len(closing) >= opener:
                    break
                body.append(lines[index])
                index += 1
            blocks.append((path, start, textwrap.dedent("\n".join(body)), skipped))
            index += 1
    return blocks


def _location(path: Path, start_line: int, node: ast.AST) -> str:
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    lineno = getattr(node, "lineno", 1) or 1
    return f"{rel}:{start_line + lineno - 1}"


def _describe(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_describe(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return f"{_describe(node.func)}()"
    return type(node).__name__


def _is_xtrax_module(obj: object) -> bool:
    return isinstance(obj, ModuleType) and (
        obj.__name__ == "xtrax" or obj.__name__.startswith("xtrax.")
    )


def _is_xtrax_class(obj: object) -> bool:
    module = getattr(obj, "__module__", "")
    return (
        inspect.isclass(obj)
        and isinstance(module, str)
        and (module == "xtrax" or module.startswith("xtrax."))
    )


def _defines(cls: type, attr: str) -> bool:
    return any(attr in base.__dict__ for base in cls.__mro__ if base is not object)


def _lookup_class_attr(cls: type, attr: str) -> object | None:
    """Resolve ``attr`` on a class, including dataclass / annotation fields."""
    found: object | None
    try:
        found = getattr(cls, attr)
    except AttributeError:
        found = None
    else:
        # getattr(cls, "__call__") climbs onto ``type`` and returns the constructor.
        if attr == "__call__" and not _defines(cls, "__call__"):
            found = None
        if found is not None:
            return found
    if attr in getattr(cls, "__annotations__", {}):
        return _DATA
    if dataclasses.is_dataclass(cls):
        try:
            fields = dataclasses.fields(cls)
        except TypeError:
            fields = ()
        if any(field.name == attr for field in fields):
            return _DATA
    return None


def _explicit_dunder_call(cls: type) -> object | None:
    for base in cls.__mro__:
        if base is object:
            break
        candidate = base.__dict__.get("__call__")
        if candidate is not None:
            return candidate
    return None


def _assigned_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for elt in target.elts:
            names.extend(_assigned_names(elt))
        return names
    if isinstance(target, ast.Starred):
        return _assigned_names(target.value)
    return []


def _simple_names(targets: list[ast.expr]) -> list[str] | None:
    names: list[str] = []
    for target in targets:
        if not isinstance(target, ast.Name):
            return None
        names.append(target.id)
    return names


def _is_ellipsis(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is Ellipsis


def _has_unpacking(node: ast.Call) -> bool:
    if any(isinstance(arg, ast.Starred) for arg in node.args):
        return True
    return any(kw.arg is None for kw in node.keywords)


def _call_binding(node: ast.Call) -> tuple[int, dict[str, object], bool]:
    """Return ``(n_positional, keywords, partial)`` for signature binding.

    A direct ``...`` argument marks the call partial and is omitted from the
    positional count. ``name=...`` keeps the keyword and marks the call partial.
    """
    partial = False
    n_positional = 0
    for arg in node.args:
        if _is_ellipsis(arg):
            partial = True
            continue
        if isinstance(arg, ast.Starred):
            continue
        n_positional += 1
    keywords: dict[str, object] = {}
    for kw in node.keywords:
        if kw.arg is None:
            continue
        if _is_ellipsis(kw.value):
            partial = True
        keywords[kw.arg] = _PLACEHOLDER
    return n_positional, keywords, partial


class _Checker:
    def __init__(self, path: Path, start_line: int, failures: list[str]) -> None:
        self.path = path
        self.start_line = start_line
        self.failures = failures
        self.namespace: dict[str, object] = {}
        self.calls = 0

    def walk(self, tree: ast.AST) -> None:
        body = getattr(tree, "body", None)
        if isinstance(body, list):
            self._walk_body(body)

    def _walk_body(self, stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            self._walk_stmt(stmt)

    def _walk_child(self, stmts: list[ast.stmt]) -> None:
        saved = self.namespace
        self.namespace = dict(saved)
        try:
            self._walk_body(stmts)
        finally:
            self.namespace = saved

    def _walk_stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Import):
            self._bind_import(stmt)
            return
        if isinstance(stmt, ast.ImportFrom):
            self._bind_import_from(stmt)
            return
        if isinstance(stmt, ast.Assign):
            self._visit_expr(stmt.value)
            self._track_assign(stmt.targets, stmt.value)
            return
        if isinstance(stmt, ast.AnnAssign):
            if stmt.annotation is not None:
                self._visit_expr(stmt.annotation)
            if stmt.value is not None:
                self._visit_expr(stmt.value)
            self._track_assign([stmt.target], stmt.value)
            return
        if isinstance(stmt, ast.AugAssign):
            self._visit_expr(stmt.target)
            self._visit_expr(stmt.value)
            self._track_assign([stmt.target], None)
            return
        if isinstance(stmt, ast.Expr):
            self._visit_expr(stmt.value)
            return
        if isinstance(stmt, ast.Return):
            if stmt.value is not None:
                self._visit_expr(stmt.value)
            return
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._walk_function(stmt)
            return
        if isinstance(stmt, ast.ClassDef):
            self._walk_class(stmt)
            return
        if isinstance(stmt, ast.If):
            self._visit_expr(stmt.test)
            self._walk_body(stmt.body)
            self._walk_body(stmt.orelse)
            return
        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            self._visit_expr(stmt.iter)
            self._track_assign([stmt.target], None)
            self._walk_body(stmt.body)
            self._walk_body(stmt.orelse)
            return
        if isinstance(stmt, ast.While):
            self._visit_expr(stmt.test)
            self._walk_body(stmt.body)
            self._walk_body(stmt.orelse)
            return
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                self._visit_expr(item.context_expr)
                if item.optional_vars is not None:
                    self._track_assign([item.optional_vars], None)
            self._walk_body(stmt.body)
            return
        if isinstance(stmt, ast.Try):
            self._walk_body(stmt.body)
            for handler in stmt.handlers:
                if handler.type is not None:
                    self._visit_expr(handler.type)
                if handler.name is not None:
                    self.namespace.pop(handler.name, None)
                self._walk_body(handler.body)
            self._walk_body(stmt.orelse)
            self._walk_body(stmt.finalbody)
            return
        for child in ast.iter_child_nodes(stmt):
            if isinstance(child, ast.stmt):
                self._walk_stmt(child)
            elif isinstance(child, ast.expr):
                self._visit_expr(child)

    def _walk_function(self, stmt: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in stmt.decorator_list:
            self._visit_expr(decorator)
        self._visit_arguments(stmt.args)
        for child in (*stmt.args.posonlyargs, *stmt.args.args, *stmt.args.kwonlyargs):
            if child.annotation is not None:
                self._visit_expr(child.annotation)
        if stmt.returns is not None:
            self._visit_expr(stmt.returns)
        # branch: rebind-drop — a def shadows an imported name
        self.namespace.pop(stmt.name, None)
        self._walk_child(stmt.body)

    def _walk_class(self, stmt: ast.ClassDef) -> None:
        for decorator in stmt.decorator_list:
            self._visit_expr(decorator)
        for base in stmt.bases:
            self._visit_expr(base)
        for keyword in stmt.keywords:
            self._visit_expr(keyword.value)
        self.namespace.pop(stmt.name, None)
        self._walk_child(stmt.body)

    def _visit_arguments(self, args: ast.arguments) -> None:
        for default in args.defaults:
            self._visit_expr(default)
        for default in args.kw_defaults:
            if default is not None:
                self._visit_expr(default)

    def _visit_expr(self, node: ast.expr) -> None:
        if isinstance(node, ast.Call):
            self._visit_expr(node.func)
            for arg in node.args:
                self._visit_expr(arg)
            for keyword in node.keywords:
                self._visit_expr(keyword.value)
            self._bind_call(node)
            return
        if isinstance(node, ast.Attribute):
            self._visit_expr(node.value)
            self._resolve(node, report=True)
            return
        if isinstance(node, ast.NamedExpr):
            self._visit_expr(node.value)
            self._track_assign([node.target], node.value)
            return
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self._visit_expr(child)
            elif isinstance(child, ast.comprehension):
                self._visit_expr(child.iter)
                for test in child.ifs:
                    self._visit_expr(test)

    def _bind_import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name != "xtrax" and not alias.name.startswith("xtrax."):
                continue
            try:
                module = importlib.import_module(alias.name)
            except Exception as exc:  # noqa: BLE001 — report every failed skill import
                where = _location(self.path, self.start_line, node)
                self.failures.append(f"{where}: import {alias.name} failed: {exc}")
                continue
            if alias.asname:
                self.namespace[alias.asname] = module
            else:
                top = alias.name.split(".", 1)[0]
                self.namespace[top] = importlib.import_module(top)

    def _bind_import_from(self, node: ast.ImportFrom) -> None:
        module_name = node.module or ""
        if node.level or (module_name != "xtrax" and not module_name.startswith("xtrax.")):
            return
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 — report every failed skill import
            where = _location(self.path, self.start_line, node)
            self.failures.append(f"{where}: import {module_name} failed: {exc}")
            return
        for alias in node.names:
            # branch: alias — local name is ``asname`` when the import renames it
            local = alias.asname or alias.name
            if alias.name == "*":
                exported = getattr(module, "__all__", None)
                names = exported if exported is not None else dir(module)
                for name in names:
                    if name.startswith("_"):
                        continue
                    self.namespace[name] = getattr(module, name)
                continue
            try:
                self.namespace[local] = getattr(module, alias.name)
            except AttributeError:
                where = _location(self.path, self.start_line, node)
                self.failures.append(f"{where}: {module_name} has no attribute {alias.name!r}")

    def _track_assign(self, targets: list[ast.expr], value: ast.expr | None) -> None:
        names = _simple_names(targets)
        # branch: instance-assign — constructor calls bind the name to the class
        if names is not None and value is not None:
            constructed = self._constructed_class(value)
            if constructed is not None:
                for name in names:
                    self.namespace[name] = _Instance(constructed)
                return
        # branch: rebind-drop — any other assignment leaves the imported signature
        for target in targets:
            for name in _assigned_names(target):
                self.namespace.pop(name, None)

    def _constructed_class(self, node: ast.AST) -> type | None:
        if not isinstance(node, ast.Call):
            return None
        func = self._resolve(node.func, report=False)
        if _is_xtrax_class(func):
            assert isinstance(func, type)
            return func
        return None

    def _resolve(self, node: ast.AST, *, report: bool) -> object:
        if isinstance(node, ast.Name):
            return self.namespace.get(node.id, _NOT_OURS)
        # branch: chained-call — Class(...).attr is an instance of Class
        if isinstance(node, ast.Call):
            constructed = self._constructed_class(node)
            if constructed is not None:
                return _Instance(constructed)
            return _NOT_OURS
        if not isinstance(node, ast.Attribute):
            return _NOT_OURS
        base = self._resolve(node.value, report=False)
        return self._lookup(base, node.attr, node, report=report)

    def _lookup(self, base: object, attr: str, node: ast.Attribute, *, report: bool) -> object:
        if base is _NOT_OURS or base is _BROKEN:
            return base
        if isinstance(base, _Instance):
            found = _lookup_class_attr(base.cls, attr)
            if found is None:
                if report:
                    where = _location(self.path, self.start_line, node)
                    self.failures.append(
                        f"{where}: {_describe(node.value)} has no attribute {attr!r}"
                    )
                return _BROKEN
            if found is _DATA:
                return _DATA
            return _Bound(found)
        if inspect.isclass(base):
            found = _lookup_class_attr(base, attr)
            if found is None:
                if report:
                    where = _location(self.path, self.start_line, node)
                    self.failures.append(
                        f"{where}: {_describe(node.value)} has no attribute {attr!r}"
                    )
                return _BROKEN
            return found
        found = _lookup_object(base, attr)
        if found is None:
            if report:
                where = _location(self.path, self.start_line, node)
                self.failures.append(f"{where}: {_describe(node.value)} has no attribute {attr!r}")
            return _BROKEN
        return found

    def _bind_call(self, node: ast.Call) -> None:
        # branch: skip-unpacking — *args / **kwargs hide the real arguments
        if _has_unpacking(node):
            return
        target = self._resolve(node.func, report=False)
        if isinstance(target, _Instance):
            call = _explicit_dunder_call(target.cls)
            if call is None:
                where = _location(self.path, self.start_line, node)
                self.failures.append(f"{where}: instance of {target.cls.__name__} is not callable")
                return
            target = _Bound(call)
        if target is _NOT_OURS or target is _BROKEN:
            return
        fn = target.fn if isinstance(target, _Bound) else target
        if target is _DATA or not callable(fn):
            where = _location(self.path, self.start_line, node)
            self.failures.append(f"{where}: resolved name is not callable")
            return
        where = _location(self.path, self.start_line, node)
        n_positional, keywords, partial = _call_binding(node)
        try:
            signature = inspect.signature(fn, follow_wrapped=True)
        except (TypeError, ValueError) as exc:
            rendered = _render_call(n_positional, keywords)
            self.failures.append(f"{where}: {rendered} does not bind: {exc}")
            return
        if isinstance(target, _Bound):
            params = list(signature.parameters.values())
            if params and params[0].name in {"self", "cls"}:
                signature = signature.replace(parameters=params[1:])
        # branch: full-bind — positional arity and required arguments
        problems = _signature_problems(signature, n_positional, keywords, partial=partial)
        self.calls += 1
        if problems:
            rendered = _render_call(n_positional, keywords)
            detail = "; ".join(problems)
            self.failures.append(f"{where}: {rendered} does not bind: {detail}")


def _lookup_object(base: object, attr: str) -> object | None:
    try:
        return getattr(base, attr)
    except AttributeError:
        if _is_xtrax_module(base):
            dotted = f"{base.__name__}.{attr}"
            try:
                return importlib.import_module(dotted)
            except ModuleNotFoundError:
                return None
        return None


def _render_call(n_positional: int, keywords: dict[str, object]) -> str:
    parts: list[str] = []
    if n_positional:
        parts.append(f"{n_positional} positional")
    parts.extend(f"{name}=..." for name in keywords)
    return ", ".join(parts) if parts else "(no arguments)"


def _signature_problems(
    signature: inspect.Signature,
    n_positional: int,
    keywords: dict[str, object],
    *,
    partial: bool,
) -> list[str]:
    """Return bind errors. Unknown keywords are reported even when required args are missing."""
    has_var_keyword = any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    )
    problems: list[str] = []
    filtered = dict(keywords)
    if not has_var_keyword:
        unknown = [name for name in keywords if name not in signature.parameters]
        if unknown:
            rendered = ", ".join(repr(name) for name in unknown)
            problems.append(f"unexpected keyword argument(s): {rendered}")
            for name in unknown:
                filtered.pop(name, None)
    positional = [_PLACEHOLDER] * n_positional
    try:
        if partial:
            signature.bind_partial(*positional, **filtered)
        else:
            signature.bind(*positional, **filtered)
    except TypeError as exc:
        problems.append(str(exc))
    return problems


def analyze(root: Path) -> CheckResult:
    """Parse every Python fence under ``root`` and bind resolved xtrax calls."""
    failures: list[str] = []
    blocks = 0
    calls = 0
    for path, start_line, source, skipped in iter_python_blocks(root):
        if skipped or not source.strip():
            continue
        blocks += 1
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
            line = start_line + (exc.lineno or 1) - 1
            failures.append(f"{rel}:{line}: syntax error: {exc.msg}")
            continue
        checker = _Checker(path, start_line, failures)
        checker.walk(tree)
        calls += checker.calls
    return CheckResult(failures, blocks, calls)


def check_tree(root: Path) -> list[str]:
    """Return one message per unresolved xtrax name or unbound call."""
    return analyze(root).failures


def test_skill_python_blocks_bind_installed_signatures() -> None:
    failures = check_tree(SKILLS)
    assert not failures, "skill python blocks disagree with installed xtrax:\n" + "\n".join(
        failures
    )


def test_checked_volume_is_non_vacuous() -> None:
    """A fence-parsing regression must not pass an empty corpus.

    Floors are the measured corpus minus slack (70 blocks → 62, 106 calls →
    90), which stays above the audit minimums of 60 blocks and 80 calls.
    """
    result = analyze(SKILLS)
    assert result.blocks >= MIN_CHECKED_BLOCKS, result.blocks
    assert result.calls >= MIN_CHECKED_CALLS, result.calls


def test_pythonish_fences_are_recognized() -> None:
    """Info strings that look like Python are ones the checker scans."""
    unknown: list[str] = []
    fence = re.compile(r"^(\s*)(`{3,})(.*)$")
    for path in _skill_markdown(SKILLS):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            match = fence.match(line)
            if match is None:
                continue
            info = match.group(3).strip().split()
            if not info:
                continue
            token = info[0]
            if not token.lower().startswith("py"):
                continue
            opener = f"{match.group(2)}{token}"
            if _FENCE_OPEN.match(opener) is None:
                rel = path.relative_to(ROOT)
                unknown.append(f"{rel}:{lineno}: {match.group(3).strip()}")
    assert not unknown, "unrecognized python fence:\n" + "\n".join(unknown)


def test_skip_marker_drops_only_the_marked_fence(tmp_path: Path) -> None:
    skill = tmp_path / "demo" / "SKILL.md"
    skill.parent.mkdir()
    skill.write_text(
        "\n".join(
            [
                "```python",
                "from xtrax.tiling.bucket import select_bucket",
                "select_bucket(sequence_length=50, boundaries=(32, 64))",
                "```",
                "",
                "<!-- skill-check: skip -->",
                "```python",
                "from xtrax import LogicalMesh",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    failures = check_tree(tmp_path)
    assert len(failures) == 1
    assert "sequence_length" in failures[0]
    assert "LogicalMesh" not in failures[0]


def test_unknown_xtrax_import_is_reported(tmp_path: Path) -> None:
    skill = tmp_path / "demo.md"
    skill.write_text(
        "\n".join(
            [
                "```python",
                "from xtrax import LogicalMesh",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    failures = check_tree(tmp_path)
    assert len(failures) == 1
    assert "LogicalMesh" in failures[0]


def _write_block(tmp_path: Path, source: str, *, ticks: int = 3, info: str = "python") -> None:
    mark = "`" * ticks
    skill = tmp_path / "demo.md"
    skill.write_text(f"{mark}{info}\n{source}{mark}\n", encoding="utf-8")


_REJECTED: list[tuple[str, str]] = [
    (
        "from xtrax.tiling.bucket import select_bucket\nselect_bucket(1, 2, 3)\n",
        "too many positional",
    ),
    (
        "from xtrax.tiling.bucket import select_bucket\nselect_bucket(boundaries=(8,))\n",
        "missing a required argument",
    ),
    (
        "from xtrax.tiling.plan import BatchPlanner\nb = BatchPlanner()\nb.not_a_method()\n",
        "not_a_method",
    ),
    (
        "from xtrax.tiling.plan import BatchPlanner\nb = BatchPlanner()\nb.plan(nope_kw=1)\n",
        "nope_kw",
    ),
    (
        "from xtrax.tiling.plan import BatchPlanner\nBatchPlanner().plan(nope_kw=1)\n",
        "nope_kw",
    ),
    (
        "from xtrax.tiling.bucket import select_bucket as pick\npick(nope=1)\n",
        "nope",
    ),
]

_REJECTED_IDS = [
    "wrong-positional-count",
    "missing-required-arg",
    "instance-method-typo",
    "instance-method-bad-kwarg",
    "chained-call",
    "alias-import-bad-kwarg",
]

_ACCEPTED = [
    "from xtrax.tiling.bucket import select_bucket\nselect_bucket(3, (8,))\n",
    "from xtrax.tiling.bucket import select_bucket\nselect_bucket(length=3, boundaries=(8,))\n",
    "from xtrax.tiling.plan import BatchPlanner\nb = BatchPlanner()\nb.plan([])\n",
    "from xtrax.tiling.plan import AxisSpec\n"
    "spec = AxisSpec(name='batch', cardinality=4, default_batch_size=2)\n"
    "spec.bucket_boundaries\n",
    "from xtrax.tiling.plan import BatchPlanner\nBatchPlanner().plan([])\n",
    "from xtrax.tiling.bucket import select_bucket as pick\npick(3, (8,))\n",
    "from xtrax.tiling.bucket import select_bucket\nselect_bucket(*lengths, nope=1)\n",
    "from xtrax.tiling.bucket import select_bucket\nselect_bucket(1, **extra)\n",
    "from xtrax.tiling.bucket import select_bucket\n"
    "select_bucket = print\n"
    "select_bucket(not_a_real_parameter=1)\n",
    "from xtrax.training.trainer import Trainer\nTrainer(...)\n",
    "from xtrax.tiling.bucket import select_bucket\n"
    "def select_bucket(x): ...\n"
    "select_bucket(x=1)\n",
    "from xtrax.tiling.plan import AxisSpec\n"
    "class AxisSpec:\n"
    "    def __init__(self, x): ...\n"
    "AxisSpec(x=1)\n",
]

_ACCEPTED_IDS = [
    "positional-ok",
    "required-args-present",
    "instance-method-ok",
    "instance-attribute-ok",
    "chained-call-ok",
    "alias-import-ok",
    "starargs-unpack-skipped",
    "kwargs-unpack-skipped",
    "rebound-name-not-checked",
    "ellipsis-partial-ok",
    "def-shadows-import",
    "class-shadows-import",
]


@pytest.mark.parametrize(("source", "needle"), _REJECTED, ids=_REJECTED_IDS)
def test_wrong_examples_are_reported(tmp_path: Path, source: str, needle: str) -> None:
    _write_block(tmp_path, source)
    failures = check_tree(tmp_path)
    assert failures, source
    assert any(needle in failure for failure in failures), failures


@pytest.mark.parametrize("source", _ACCEPTED, ids=_ACCEPTED_IDS)
def test_sound_examples_bind(tmp_path: Path, source: str) -> None:
    _write_block(tmp_path, source)
    assert check_tree(tmp_path) == []


@pytest.mark.parametrize(
    ("ticks", "info"),
    [(3, "py"), (3, "python3"), (4, "python"), (5, "py")],
)
def test_python_fence_variants_are_scanned(tmp_path: Path, ticks: int, info: str) -> None:
    _write_block(
        tmp_path,
        "from xtrax.tiling.bucket import select_bucket\nselect_bucket(1, 2, 3)\n",
        ticks=ticks,
        info=info,
    )
    failures = check_tree(tmp_path)
    assert failures
    assert "too many positional" in failures[0]


def test_non_python_fence_is_ignored(tmp_path: Path) -> None:
    (tmp_path / "demo.md").write_text(
        "```bash\nfrom xtrax import LogicalMesh\n```\n",
        encoding="utf-8",
    )
    assert check_tree(tmp_path) == []


@pytest.mark.parametrize(
    "source",
    [
        "from xtrax.tiling.bucket import select_bucket\nselect_bucket(3, boundaries=(8,))\n",
    ],
)
def test_matching_keyword_binds(tmp_path: Path, source: str) -> None:
    _write_block(tmp_path, source)
    assert check_tree(tmp_path) == []
