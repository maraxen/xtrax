#!/usr/bin/env python3
"""Run jaxlint and emit a JSON audit envelope (N0.4 foundation gate)."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT / "src" / "xtrax"
DEFAULT_PORT_TARGET = ROOT / "port" / "port_target.toml"
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


def resolve_port_target(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    pyproject = ROOT / "pyproject.toml"
    if pyproject.is_file():
        pyproject_data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        configured = pyproject_data.get("tool", {}).get("port", {}).get("target")
        if configured:
            return (ROOT / configured).resolve()
    return DEFAULT_PORT_TARGET.resolve()


def load_manifest(port_root: Path, wave_id: str) -> dict[str, Any]:
    manifest_path = port_root / "manifests" / f"{wave_id}.toml"
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found for wave_id={wave_id!r}: {manifest_path}")
    return tomllib.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_targets_from_port_target(port_target_path: Path) -> list[Path]:
    port_config = tomllib.loads(port_target_path.read_text(encoding="utf-8"))
    port_section = port_config.get("port", {})
    wave_id = port_section.get("wave_id")
    if not isinstance(wave_id, str) or not wave_id:
        raise SystemExit("port_target.toml [port] wave_id is required")

    manifest = load_manifest(port_target_path.parent, wave_id)
    kernels = manifest.get("kernels")
    if not isinstance(kernels, list) or not kernels:
        raise SystemExit(f"manifest {wave_id} has no [[kernels]] entries")

    targets: list[Path] = []
    seen: set[Path] = set()
    for entry in kernels:
        if not isinstance(entry, dict):
            continue
        module_path = entry.get("module_path")
        if not isinstance(module_path, str) or not module_path:
            continue
        target = (ROOT / module_path).resolve()
        if target not in seen:
            seen.add(target)
            targets.append(target)
    if not targets:
        raise SystemExit(f"manifest {wave_id} has no module_path targets for jaxlint")
    return targets


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
        default=None,
        help="Path to scan (default: src/xtrax, unless --paths-from is set)",
    )
    parser.add_argument(
        "--paths-from",
        type=Path,
        default=None,
        help="Load jaxlint targets from port_target.toml wave manifest module_path entries",
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

    performance_only = not args.full

    if args.paths_from is not None:
        port_target = resolve_port_target(args.paths_from)
        if not port_target.is_file():
            raise SystemExit(f"port target not found: {port_target}")
        targets = resolve_targets_from_port_target(port_target)
    else:
        target_arg = args.target if args.target is not None else str(DEFAULT_TARGET)
        targets = [Path(target_arg).resolve()]

    combined_findings: list[dict[str, Any]] = []
    worst_exit = 0
    scanned: list[str] = []
    resolved_targets: list[Path] = []
    for target in targets:
        if not target.exists():
            raise SystemExit(f"jaxlint target not found: {target}")
        exit_code, findings = run_jaxlint(target, performance_only=performance_only)
        worst_exit = max(worst_exit, exit_code)
        rel = str(target.relative_to(ROOT))
        scanned.append(rel)
        resolved_targets.append(target)
        for finding in findings:
            tagged = dict(finding)
            tagged.setdefault("target", rel)
            combined_findings.append(tagged)

    envelope = build_envelope(
        resolved_targets[0],
        performance_only=performance_only,
        exit_code=worst_exit,
        findings=combined_findings,
    )
    if len(scanned) > 1:
        envelope["targets"] = scanned

    payload = json.dumps(envelope, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)

    # Foundation gate: performance-only passes when error_count is zero.
    if performance_only:
        return 1 if envelope["error_count"] > 0 else 0
    return 0 if worst_exit == 0 else worst_exit


if __name__ == "__main__":
    sys.exit(main())
