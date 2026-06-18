#!/usr/bin/env python3
"""Run jaxlint and emit a JSON audit envelope (N0.4 foundation gate)."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT / "src" / "xtrax"
SCHEMA_VERSION = "audit_jaxlint_v0"


def _jaxlint_version() -> str:
    try:
        return importlib.metadata.version("jaxlint")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def run_jaxlint(
    target: Path,
    *,
    performance_only: bool,
) -> tuple[int, list[dict[str, Any]]]:
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
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if not proc.stdout.strip():
        return proc.returncode, []

    try:
        findings = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"jaxlint did not emit valid JSON (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        ) from exc

    if not isinstance(findings, list):
        raise SystemExit(f"expected JSON array from jaxlint, got {type(findings).__name__}")

    return proc.returncode, findings


def build_envelope(
    target: Path,
    *,
    performance_only: bool,
    exit_code: int,
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    errors = [f for f in findings if str(f.get("severity", "")).lower() == "error"]
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "jaxlint",
        "tool_version": _jaxlint_version(),
        "emitted_at": datetime.now(UTC).isoformat(),
        "target": str(target.relative_to(ROOT)),
        "mode": "performance_only" if performance_only else "full",
        "exit_code": exit_code,
        "finding_count": len(findings),
        "error_count": len(errors),
        "findings": findings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        nargs="?",
        default=str(DEFAULT_TARGET),
        help="Path to scan (default: src/xtrax)",
    )
    parser.add_argument(
        "--performance-only",
        action="store_true",
        default=True,
        help="Skip doc/math rules (--no-doc). Default for N0 foundation gate.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all jaxlint rule families (including documentation).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional path to write the envelope JSON",
    )
    args = parser.parse_args(argv)

    target = Path(args.target).resolve()
    performance_only = not args.full

    exit_code, findings = run_jaxlint(target, performance_only=performance_only)
    envelope = build_envelope(
        target,
        performance_only=performance_only,
        exit_code=exit_code,
        findings=findings,
    )

    payload = json.dumps(envelope, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)

    # Foundation gate: performance-only must be clean (no JL errors).
    if performance_only and envelope["error_count"] > 0:
        return 1
    return 0 if exit_code == 0 else exit_code


if __name__ == "__main__":
    sys.exit(main())
