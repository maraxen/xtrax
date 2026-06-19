"""D3' added-types diff gate — LibCST merge-base public annotation check (#1589)."""

from __future__ import annotations

import ast
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import libcst as cst

from xtrax.devtools.emit import append_finding, emit_metric_finding
from xtrax.devtools.gates.type_hardening import DIMENSION, inspect_public_callable

DEFAULT_TARGET = Path("src/xtrax")
DiffStatus = Literal["pass", "fail", "skip"]


@dataclass(frozen=True, slots=True)
class DiffGateResult:
    status: DiffStatus
    merge_base: str | None
    skip_reason: str | None
    files_checked: int
    callables_checked: int
    violations: tuple[str, ...]
    findings_emitted: int


class _PublicFunctionCollector(cst.CSTVisitor):
    def __init__(self) -> None:
        self.class_stack: list[str] = []
        self.functions: dict[str, cst.FunctionDef | cst.AsyncFunctionDef] = {}

    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:
        if node.name.value.startswith("_"):
            return False
        self.class_stack.append(node.name.value)
        return True

    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        if self.class_stack and self.class_stack[-1] == node.name.value:
            self.class_stack.pop()

    def _record(self, node: cst.FunctionDef | cst.AsyncFunctionDef) -> None:
        name = node.name.value
        if name.startswith("_"):
            return
        qualname = ".".join([*self.class_stack, name])
        self.functions[qualname] = node

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        self._record(node)

    def visit_AsyncFunctionDef(self, node: cst.AsyncFunctionDef) -> None:
        self._record(node)


def _collect_public_functions(
    source: str,
) -> dict[str, cst.FunctionDef | cst.AsyncFunctionDef]:
    module = cst.parse_module(source)
    collector = _PublicFunctionCollector()
    module.visit(collector)
    return collector.functions


def _signature_changed(
    old_fn: cst.FunctionDef | cst.AsyncFunctionDef,
    new_fn: cst.FunctionDef | cst.AsyncFunctionDef,
) -> bool:
    if not old_fn.params.deep_equals(new_fn.params):
        return True
    old_returns = old_fn.returns
    new_returns = new_fn.returns
    if old_returns is None and new_returns is None:
        return False
    if old_returns is None or new_returns is None:
        return True
    return not old_returns.deep_equals(new_returns)


def _lookup_ast_callable(
    tree: ast.Module,
    qualname: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    parts = qualname.split(".")
    if len(parts) == 1:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == qualname:
                    return node
        return None

    class_name, method_name = parts[0], parts[1]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.name == method_name:
                        return child
    return None


def _git_run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def resolve_merge_base(repo: Path) -> str | None:
    """Resolve a merge-base SHA for diff comparison (CI-aware)."""
    repo = repo.resolve()
    base_ref = os.environ.get("GITHUB_BASE_REF")
    if base_ref:
        for candidate in (f"origin/{base_ref}", base_ref):
            result = _git_run(repo, "merge-base", "HEAD", candidate)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()

    event_before = os.environ.get("GITHUB_EVENT_BEFORE")
    if event_before and event_before != "0" * 40:
        verify = _git_run(repo, "cat-file", "-e", f"{event_before}^{{commit}}")
        if verify.returncode == 0:
            return event_before.strip()

    for candidate in ("origin/main", "main", "HEAD~1"):
        result = _git_run(repo, "merge-base", "HEAD", candidate)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return None


def list_changed_python_files(
    repo: Path,
    merge_base: str,
    *,
    target: Path,
) -> list[Path]:
    """Return changed ``.py`` files under ``target`` between merge-base and HEAD."""
    repo = repo.resolve()
    target = target.resolve()
    try:
        rel_target = target.relative_to(repo)
    except ValueError:
        rel_target = target
    prefix = f"{rel_target.as_posix()}/"
    result = _git_run(
        repo,
        "diff",
        "--name-only",
        f"{merge_base}..HEAD",
        "--",
        prefix,
    )
    if result.returncode != 0:
        return []
    files: list[Path] = []
    for line in result.stdout.splitlines():
        rel = line.strip()
        if not rel.endswith(".py") or rel.endswith("__init__.py"):
            continue
        path = repo / rel
        if path.is_file():
            files.append(path)
    return sorted(files)


def git_show_file(repo: Path, rev: str, rel_path: str) -> str | None:
    result = _git_run(repo, "show", f"{rev}:{rel_path}")
    if result.returncode != 0:
        return None
    return result.stdout


def diff_callables_to_audit(
    *,
    base_source: str | None,
    head_source: str,
) -> set[str]:
    """Return qualnames of added or signature-modified public callables."""
    head_map = _collect_public_functions(head_source)
    if base_source is None:
        return set(head_map)

    base_map = _collect_public_functions(base_source)
    touched: set[str] = set()
    for qualname, new_fn in head_map.items():
        old_fn = base_map.get(qualname)
        if old_fn is None or _signature_changed(old_fn, new_fn):
            touched.add(qualname)
    return touched


def audit_changed_callables(
    *,
    head_source: str,
    rel_path: str,
    qualnames: set[str],
) -> list[str]:
    tree = ast.parse(head_source, filename=rel_path)
    if not isinstance(tree, ast.Module):
        return [f"{rel_path}: expected module AST"]
    violations: list[str] = []
    for qualname in sorted(qualnames):
        fn_node = _lookup_ast_callable(tree, qualname)
        if fn_node is None:
            violations.append(f"{rel_path}: unable to locate callable `{qualname}`")
            continue
        for message in inspect_public_callable(fn_node, qualname=qualname):
            violations.append(f"{rel_path}: {message}")
    return violations


def run_added_types_diff_gate(
    repo_root: Path,
    *,
    target: Path = DEFAULT_TARGET,
    merge_base: str | None = None,
    audits_path: Path | None = None,
    run_id: str | None = None,
) -> DiffGateResult:
    """Run merge-base diff gate; skip loudly when merge-base cannot be resolved."""
    repo_root = repo_root.resolve()
    target_path = (repo_root / target).resolve() if not target.is_absolute() else target

    base = merge_base or resolve_merge_base(repo_root)
    if base is None:
        return DiffGateResult(
            status="skip",
            merge_base=None,
            skip_reason=(
                "merge-base unavailable (shallow clone?); "
                "configure fetch-depth:0 in CI"
            ),
            files_checked=0,
            callables_checked=0,
            violations=(),
            findings_emitted=0,
        )

    changed_files = list_changed_python_files(repo_root, base, target=target_path)
    all_violations: list[str] = []
    callables_checked = 0

    for path in changed_files:
        rel = path.relative_to(repo_root).as_posix()
        head_source = path.read_text(encoding="utf-8")
        base_source = git_show_file(repo_root, base, rel)
        touched = diff_callables_to_audit(
            base_source=base_source,
            head_source=head_source,
        )
        callables_checked += len(touched)
        all_violations.extend(
            audit_changed_callables(
                head_source=head_source,
                rel_path=rel,
                qualnames=touched,
            )
        )

    findings_emitted = 0
    if audits_path is not None:
        for violation in all_violations:
            record = emit_metric_finding(
                dim=DIMENSION,
                severity="major",
                file_line=violation.split(":", 1)[0],
                evidence=violation,
                rule_id="type_hardening.added_types_diff",
                symbol_qualname="",
                payload={"violation_kind": "added_types_diff"},
                run_id=run_id,
            )
            append_finding(record, audits_path=audits_path)
            findings_emitted += 1

    status: DiffStatus = "fail" if all_violations else "pass"
    return DiffGateResult(
        status=status,
        merge_base=base,
        skip_reason=None,
        files_checked=len(changed_files),
        callables_checked=callables_checked,
        violations=tuple(all_violations),
        findings_emitted=findings_emitted,
    )
