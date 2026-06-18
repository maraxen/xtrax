"""Shared interrogate subprocess runner for documentation gate."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_COVERAGE_RE = re.compile(r"actual:\s*(\d+(?:\.\d+)?)%")


def run_interrogate_coverage(package: Path, root: Path) -> float:
    """Run interrogate on ``package`` and return docstring coverage %."""
    cmd = [
        "uv",
        "run",
        "interrogate",
        str(package),
    ]
    proc = subprocess.run(
        cmd,
        cwd=root,
        capture_output=True,
        text=True,
    )
    output = f"{proc.stdout}\n{proc.stderr}"
    match = _COVERAGE_RE.search(output)
    if match is None:
        msg = (
            "interrogate output did not contain coverage percentage "
            f"(exit {proc.returncode}):\n{output}"
        )
        raise RuntimeError(msg)
    return float(match.group(1))
