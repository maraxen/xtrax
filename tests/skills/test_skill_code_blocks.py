"""Bind skill examples against the installed xtrax package.

Every fenced ``python`` block under ``agent_assets/skills/**/*.md`` is parsed
with :mod:`ast`. ``from xtrax... import X`` and ``xtrax.a.b`` names are resolved
on the installed package. A call to one of those callables that passes keyword
arguments is checked with :meth:`inspect.Signature.bind_partial`.

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
import importlib
import inspect
import textwrap
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SKILLS = ROOT / "agent_assets" / "skills"
SKIP_MARKER = "<!-- skill-check: skip -->"

_NOT_OURS = object()
_BROKEN = object()


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
    """Return ``(path, first_content_line, source, skipped)`` for each python fence.

    ``first_content_line`` is 1-based in ``path``. ``skipped`` is true when
    ``<!-- skill-check: skip -->`` is the nearest non-blank line above the fence.
    """
    blocks: list[tuple[Path, int, str, bool]] = []
    for path in _skill_markdown(root):
        lines = path.read_text(encoding="utf-8").splitlines()
        index = 0
        while index < len(lines):
            stripped = lines[index].strip()
            if stripped != "```python":
                index += 1
                continue
            skipped = _nearest_nonblank(lines, index) == SKIP_MARKER
            body: list[str] = []
            index += 1
            start = index + 1  # 1-based line of the first body line
            while index < len(lines) and lines[index].strip() != "```":
                body.append(lines[index])
                index += 1
            blocks.append((path, start, textwrap.dedent("\n".join(body)), skipped))
            index += 1
    return blocks


def _location(path: Path, start_line: int, node: ast.AST) -> str:
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    lineno = getattr(node, "lineno", 1) or 1
    return f"{rel}:{start_line + lineno - 1}"


def _is_xtrax_module(obj: object) -> bool:
    return isinstance(obj, ModuleType) and (
        obj.__name__ == "xtrax" or obj.__name__.startswith("xtrax.")
    )


def _lookup(base: object, attr: str) -> object | None:
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


def _bind_imports(
    tree: ast.AST, path: Path, start_line: int
) -> tuple[dict[str, object], list[str]]:
    namespace: dict[str, object] = {}
    failures: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name != "xtrax" and not alias.name.startswith("xtrax."):
                    continue
                try:
                    module = importlib.import_module(alias.name)
                except Exception as exc:  # noqa: BLE001 — report every failed skill import
                    where = _location(path, start_line, node)
                    failures.append(f"{where}: import {alias.name} failed: {exc}")
                    continue
                if alias.asname:
                    namespace[alias.asname] = module
                else:
                    top = alias.name.split(".", 1)[0]
                    namespace[top] = importlib.import_module(top)
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if node.level or (module_name != "xtrax" and not module_name.startswith("xtrax.")):
                continue
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001 — report every failed skill import
                where = _location(path, start_line, node)
                failures.append(f"{where}: import {module_name} failed: {exc}")
                continue
            for alias in node.names:
                local = alias.asname or alias.name
                if alias.name == "*":
                    exported = getattr(module, "__all__", None)
                    names = exported if exported is not None else dir(module)
                    for name in names:
                        if name.startswith("_"):
                            continue
                        namespace[name] = getattr(module, name)
                    continue
                try:
                    namespace[local] = getattr(module, alias.name)
                except AttributeError:
                    where = _location(path, start_line, node)
                    failures.append(f"{where}: {module_name} has no attribute {alias.name!r}")
    return namespace, failures


def _resolve(
    node: ast.AST,
    namespace: dict[str, object],
    path: Path,
    start_line: int,
    failures: list[str],
) -> object:
    if isinstance(node, ast.Name):
        if node.id in namespace:
            return namespace[node.id]
        return _NOT_OURS
    if not isinstance(node, ast.Attribute):
        return _NOT_OURS
    base = _resolve(node.value, namespace, path, start_line, failures)
    if base is _NOT_OURS or base is _BROKEN:
        return base
    found = _lookup(base, node.attr)
    if found is None:
        where = _location(path, start_line, node)
        failures.append(f"{where}: {ast.dump(node.value)} has no attribute {node.attr!r}")
        return _BROKEN
    return found


def _check_calls(
    tree: ast.AST,
    namespace: dict[str, object],
    path: Path,
    start_line: int,
) -> list[str]:
    failures: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            _resolve(node, namespace, path, start_line, failures)
        if not isinstance(node, ast.Call):
            continue
        keywords = {kw.arg: None for kw in node.keywords if kw.arg is not None}
        if not keywords:
            continue
        target = _resolve(node.func, namespace, path, start_line, failures)
        if target is _NOT_OURS or target is _BROKEN:
            continue
        where = _location(path, start_line, node)
        if not callable(target):
            failures.append(f"{where}: resolved name is not callable")
            continue
        try:
            signature = inspect.signature(target, follow_wrapped=True)
            signature.bind_partial(**keywords)
        except (TypeError, ValueError) as exc:
            rendered = ", ".join(f"{name}=..." for name in keywords)
            failures.append(f"{where}: {rendered} does not bind: {exc}")
    return failures


def check_tree(root: Path) -> list[str]:
    """Return one message per unresolved xtrax name or unbound keyword."""
    failures: list[str] = []
    for path, start_line, source, skipped in iter_python_blocks(root):
        if skipped or not source.strip():
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
            line = start_line + (exc.lineno or 1) - 1
            failures.append(f"{rel}:{line}: syntax error: {exc.msg}")
            continue
        _namespace, import_failures = _bind_imports(tree, path, start_line)
        failures.extend(import_failures)
        failures.extend(_check_calls(tree, _namespace, path, start_line))
    return failures


def test_skill_python_blocks_bind_installed_signatures() -> None:
    failures = check_tree(SKILLS)
    assert not failures, "skill python blocks disagree with installed xtrax:\n" + "\n".join(
        failures
    )


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


@pytest.mark.parametrize(
    "source",
    [
        "from xtrax.tiling.bucket import select_bucket\nselect_bucket(3, boundaries=(8,))\n",
    ],
)
def test_matching_keyword_binds(tmp_path: Path, source: str) -> None:
    skill = tmp_path / "demo.md"
    skill.write_text(f"```python\n{source}```\n", encoding="utf-8")
    assert check_tree(tmp_path) == []
