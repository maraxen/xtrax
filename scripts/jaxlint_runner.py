"""Runner for jaxlint and JL-PALLAS static lint rules (Debt #3197)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.lint_pallas import lint_pallas_file
from xtrax.jaxlint_runner import run_jaxlint_json as _base_run_jaxlint_json


def run_jaxlint_json(
    target: Path,
    *,
    root: Path,
    performance_only: bool = True,
    include_pallas: bool = True,
) -> list[dict[str, Any]]:
    findings = _base_run_jaxlint_json(target, root=root, performance_only=performance_only)
    if include_pallas:
        pallas_findings: list[dict[str, Any]] = []
        if target.is_file() and target.suffix == ".py":
            pallas_findings.extend(lint_pallas_file(target))
        elif target.is_dir():
            for py_file in target.rglob("*.py"):
                pallas_findings.extend(lint_pallas_file(py_file))
        findings.extend(pallas_findings)
    return findings
