"""Shared jaxlint subprocess runner for dimension gates."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from xtrax.devtools.emit import Severity


def map_severity(jaxlint_level: str) -> Severity:
    level = jaxlint_level.lower()
    if level == "error":
        return "major"
    if level == "warning":
        return "minor"
    return "info"


def run_jaxlint_json(
    target: Path,
    *,
    root: Path,
    performance_only: bool = True,
) -> list[dict[str, Any]]:
    cmd = [
        "uv",
        "run",
        "jaxlint",
        "check",
        "--format",
        "json",
    ]
    if performance_only:
        cmd.append("--no-doc")
    cmd.append(str(target))
    proc = subprocess.run(
        cmd,
        cwd=root,
        capture_output=True,
        text=True,
    )
    if not proc.stdout.strip():
        return []
    try:
        findings = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []
    if not isinstance(findings, list):
        return []
    return [f for f in findings if isinstance(f, dict)]


def file_line(finding: dict[str, Any]) -> str:
    path = str(finding.get("path", ""))
    line = finding.get("line", 0)
    return f"{path}:{line}"
