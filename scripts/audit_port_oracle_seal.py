#!/usr/bin/env python3
"""P0-ORACLE seal gate: reference headers, oracle_id lock, manifest freshness."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PORT_TARGET = ROOT / "port" / "port_target.toml"
SCHEMA_VERSION = "audit_port_oracle_seal_v0"
REFERENCE_BANNER = "# REFERENCE: DO NOT MODIFY"
ORACLE_ID_RE = re.compile(
    r"^ref:port/reference/(?P<kernel>[^:]+):(?P<version>[^:]+):sha256:(?P<hash>[0-9a-f]{64})$"
)


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


def load_port_target(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def load_manifest(port_root: Path, wave_id: str) -> tuple[dict[str, Any], Path]:
    manifest_path = port_root / "manifests" / f"{wave_id}.toml"
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found for wave_id={wave_id!r}: {manifest_path}")
    return tomllib.loads(manifest_path.read_text(encoding="utf-8")), manifest_path


def canonical_manifest_bytes(manifest_text: str) -> bytes:
    lines: list[str] = []
    for line in manifest_text.splitlines():
        if re.match(r"^\s*manifest_hash\s*=", line):
            continue
        lines.append(line)
    canonical = "\n".join(lines).rstrip() + "\n"
    return canonical.encode("utf-8")


def compute_manifest_hash(manifest_text: str) -> str:
    digest = hashlib.sha256(canonical_manifest_bytes(manifest_text)).hexdigest()
    return f"sha256:{digest}"


def compute_algo_content_hash(algo_path: Path) -> str:
    return hashlib.sha256(algo_path.read_bytes()).hexdigest()


def _check_reference_headers(reference_dir: Path) -> list[str]:
    failures: list[str] = []
    py_files = sorted(reference_dir.rglob("*.py"))
    if not py_files:
        failures.append(f"no reference .py files under {reference_dir}")
        return failures
    for path in py_files:
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if not text.startswith(REFERENCE_BANNER):
            rel = path.relative_to(ROOT)
            failures.append(f"{rel}: missing {REFERENCE_BANNER!r} header")
    return failures


def audit_oracle_seal(port_target_path: Path) -> tuple[dict[str, Any], int]:
    port_config = load_port_target(port_target_path)
    port_root = port_target_path.parent
    port_section = port_config.get("port", {})
    wave_id = port_section.get("wave_id")
    if not isinstance(wave_id, str) or not wave_id:
        raise SystemExit("port_target.toml [port] wave_id is required")

    oracle_id = port_section.get("oracle_id", "")
    reference_subtree = port_section.get("reference_subtree", "")
    failures: list[str] = []

    manifest, manifest_path = load_manifest(port_root, wave_id)
    manifest_text = manifest_path.read_text(encoding="utf-8")
    recorded_hash = manifest.get("manifest", {}).get("manifest_hash")
    expected_manifest_hash = compute_manifest_hash(manifest_text)
    if recorded_hash != expected_manifest_hash:
        failures.append(
            f"{manifest_path.relative_to(ROOT)}: manifest_hash mismatch "
            f"(recorded {recorded_hash!r}, expected {expected_manifest_hash!r})"
        )

    if not isinstance(reference_subtree, str) or not reference_subtree:
        failures.append("port_target [port] reference_subtree is required")
        reference_dir = None
    else:
        reference_dir = (port_root / reference_subtree).resolve()
        if not reference_dir.is_dir():
            failures.append(f"reference subtree missing: {reference_dir.relative_to(ROOT)}")

    algo_path: Path | None = None
    if reference_dir is not None and reference_dir.is_dir():
        algo_path = reference_dir / "algo.py"
        baseline_path = reference_dir / "baseline_io.json"
        if not algo_path.is_file():
            failures.append(f"reference algo.py missing: {algo_path.relative_to(ROOT)}")
        if not baseline_path.is_file():
            failures.append(
                f"reference baseline_io.json missing: {baseline_path.relative_to(ROOT)}"
            )
        failures.extend(_check_reference_headers(reference_dir))

    computed_hash: str | None = None
    if isinstance(oracle_id, str) and oracle_id:
        match = ORACLE_ID_RE.match(oracle_id)
        if match is None:
            failures.append(f"invalid oracle_id format: {oracle_id!r}")
        elif algo_path is not None and algo_path.is_file():
            computed_hash = compute_algo_content_hash(algo_path)
            if match.group("hash") != computed_hash:
                failures.append(
                    "oracle_id content hash mismatch "
                    f"(locked {match.group('hash')!r}, computed {computed_hash!r})"
                )
            kernel = match.group("kernel")
            if reference_dir is not None:
                expected_kernel = reference_dir.relative_to(port_root / "reference").as_posix()
                if kernel != expected_kernel:
                    failures.append(
                        f"oracle_id kernel {kernel!r} != reference subtree {expected_kernel!r}"
                    )
    else:
        failures.append("port_target [port] oracle_id is required")

    envelope: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "emitted_at": datetime.now(UTC).isoformat(),
        "port_target": str(port_target_path.relative_to(ROOT)),
        "wave_id": wave_id,
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "oracle_id": oracle_id,
        "computed_algo_hash": computed_hash,
        "manifest_hash_ok": recorded_hash == expected_manifest_hash,
        "failure_count": len(failures),
        "failures": failures,
    }
    return envelope, 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Path to port_target.toml (default: [tool.port].target or port/port_target.toml)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional path to write the envelope JSON",
    )
    args = parser.parse_args(argv)

    port_target = resolve_port_target(args.target)
    if not port_target.is_file():
        raise SystemExit(f"port target not found: {port_target}")

    envelope, exit_code = audit_oracle_seal(port_target)
    payload = json.dumps(envelope, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
