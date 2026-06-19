"""D6 API-ergonomics gate: param-sprawl AST scan + baseline ratchet (N2.7 / #1586)."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from xtrax.devtools.baseline import (
    DEFAULT_BASELINE_PATH,
    evaluate_metric,
    load_baseline,
    save_baseline,
    update_metric,
)
from xtrax.devtools.emit import append_finding, emit_metric_finding

METRIC_KEY = "api_ergonomics.param_sprawl_violation_count"
DIMENSION = "api_ergonomics"
RULE_ID = "api_ergonomics.param_sprawl"


@dataclass(frozen=True, slots=True)
class SprawlViolation:
    qualname: str
    file_line: str
    required_count: int


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    violation_count: int
    violations: tuple[SprawlViolation, ...]
    findings_emitted: int
    baseline_updated: bool
    metric_key: str = METRIC_KEY


def _is_public_callable(name: str) -> bool:
    if name in {"__init__", "__call__"}:
        return True
    return not name.startswith("_")


def _count_required_params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    positional: list[ast.arg] = []
    for arg in (*fn.args.posonlyargs, *fn.args.args):
        if arg.arg in {"self", "cls"}:
            continue
        positional.append(arg)

    defaults = [None] * (len(positional) - len(fn.args.defaults)) + list(fn.args.defaults)
    required = sum(1 for default in defaults if default is None)

    kw_defaults = fn.args.kw_defaults or []
    for arg, default in zip(fn.args.kwonlyargs, kw_defaults, strict=True):
        if arg.arg in {"self", "cls"}:
            continue
        if default is None:
            required += 1
    return required


def _file_line(path: Path, lineno: int) -> str:
    return f"{path}:{lineno}"


def _check_function(
    violations: list[SprawlViolation],
    path: Path,
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    qualname: str,
    public: bool,
    max_required: int,
) -> None:
    if not public:
        return
    required_count = _count_required_params(fn)
    if required_count > max_required:
        violations.append(
            SprawlViolation(
                qualname=qualname,
                file_line=_file_line(path, fn.lineno),
                required_count=required_count,
            )
        )


def _scan_class(
    violations: list[SprawlViolation],
    path: Path,
    node: ast.ClassDef,
    *,
    max_required: int,
    prefix: str = "",
) -> None:
    if not _is_public_callable(node.name):
        return
    class_qualname = f"{prefix}.{node.name}" if prefix else node.name
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _check_function(
                violations,
                path,
                child,
                qualname=f"{class_qualname}.{child.name}",
                public=_is_public_callable(child.name),
                max_required=max_required,
            )


def _scan_module(
    violations: list[SprawlViolation],
    path: Path,
    tree: ast.Module,
    *,
    max_required: int,
) -> None:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _check_function(
                violations,
                path,
                node,
                qualname=node.name,
                public=_is_public_callable(node.name),
                max_required=max_required,
            )
        elif isinstance(node, ast.ClassDef):
            _scan_class(violations, path, node, max_required=max_required)


def scan_param_sprawl(root: Path, max_required: int = 5) -> list[SprawlViolation]:
    """AST-scan public callables with more than ``max_required`` required parameters."""
    violations: list[SprawlViolation] = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        _scan_module(violations, path, tree, max_required=max_required)
    return violations


def run_api_ergonomics_gate(
    target: Path,
    audits_path: Path,
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    *,
    max_required: int = 5,
    run_id: str | None = None,
    write_baseline: bool = True,
) -> GateResult:
    """Scan param sprawl, emit findings, evaluate baseline ratchet."""
    violations = scan_param_sprawl(target.resolve(), max_required=max_required)
    violation_count = len(violations)

    emitted = 0
    for violation in violations:
        record = emit_metric_finding(
            dim=DIMENSION,
            severity="minor",
            file_line=violation.file_line,
            evidence=(
                f"{violation.qualname} has {violation.required_count} required "
                f"parameters (max {max_required})"
            ),
            rule_id=RULE_ID,
            symbol_qualname=violation.qualname,
            payload={
                "violation_kind": "param_sprawl",
                "required_count": violation.required_count,
                "max_required": max_required,
            },
            run_id=run_id,
        )
        append_finding(record, audits_path=audits_path)
        emitted += 1

    baseline = load_baseline(path=baseline_path)
    passes_gate, should_update = evaluate_metric(
        baseline,
        METRIC_KEY,
        float(violation_count),
    )
    baseline_updated = False
    if passes_gate and should_update and write_baseline:
        tightened = update_metric(
            baseline,
            METRIC_KEY,
            float(violation_count),
            "minimize",
        )
        save_baseline(tightened, path=baseline_path)
        baseline_updated = True

    return GateResult(
        passed=passes_gate,
        violation_count=violation_count,
        violations=tuple(violations),
        findings_emitted=emitted,
        baseline_updated=baseline_updated,
    )
