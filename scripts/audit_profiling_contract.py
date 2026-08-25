#!/usr/bin/env python3
"""Profiling contract gate CLI -- smoke-check xtrax.profiling without jax/GPU.

Complements tests/profiling/ as a standalone audit entrypoint following the
canonical gate recipe (ruff -> pytest -> scripts/audit_*.py). Checks:
  1. every committed ProbeRecord fixture round-trips (read + from_json);
  2. the claim-validity contract is alive: a stage0+stage1-only source set
     FAILS CLOSED on TERM_RANKING, and the full fixture set renders the
     claim-gated ranking table;
  3. the leaf-package seam rule holds: no prolix imports, no relative
     imports, no sibling-xtrax imports anywhere under src/xtrax/profiling/.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE_DIR = ROOT / "src" / "xtrax" / "profiling"
DEFAULT_FIXTURES_DIR = ROOT / "tests" / "profiling" / "fixtures" / "records"


def _check_leaf_rule(package_dir: Path) -> list[str]:
    failures: list[str] = []
    py_files = sorted(package_dir.rglob("*.py"))
    if not py_files:
        return [f"no .py files found under {package_dir}"]
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    failures.append(f"{path}: relative import with no module")
                    continue
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top == "prolix":
                    failures.append(f"{path}: imports prolix ({name!r})")
                elif top == "xtrax" and ".".join(name.split(".")[:2]) != ("xtrax.profiling"):
                    failures.append(
                        f"{path}: sibling-xtrax import ({name!r}) breaks "
                        "the dependency-free leaf rule"
                    )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=DEFAULT_FIXTURES_DIR,
        help="Directory of committed ProbeRecord JSON fixtures",
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=DEFAULT_PACKAGE_DIR,
        help="xtrax.profiling package directory for the leaf-rule scan",
    )
    args = parser.parse_args(argv)

    # Deferred imports: keep argparse/--help fast and dependency-light.
    from xtrax.profiling.claims import ClaimValidityError
    from xtrax.profiling.record import ProbeRecord
    from xtrax.profiling.report import render_report

    failures: list[str] = []

    fixtures = sorted(args.fixtures_dir.glob("*.json"))
    if not fixtures:
        failures.append(f"no fixtures found under {args.fixtures_dir}")
    else:
        for path in fixtures:
            try:
                rec = ProbeRecord.read(path)
                ProbeRecord.from_json(rec.to_json())
                del rec
            except Exception as exc:
                failures.append(f"fixture {path.name} failed round-trip: {exc}")

    by_stage = {}
    for path in fixtures:
        try:
            by_stage.setdefault(ProbeRecord.read(path).stage, []).append(path)
        except Exception:
            pass
    low_stage = [p for s in (0, 1) for p in by_stage.get(s, [])]
    high_stage = [p for s in (2, 3) for p in by_stage.get(s, [])]
    if low_stage and high_stage:
        try:
            render_report([*low_stage])
            failures.append(
                "claim gate DEAD: stage0/1-only set rendered a TERM_RANKING "
                "report instead of raising"
            )
        except ClaimValidityError:
            pass
        try:
            text = render_report([*low_stage, *high_stage])
            if "| scope | exclusive_seconds |" not in text:
                failures.append("rendered report missing expected table header")
        except ClaimValidityError as exc:
            failures.append(f"full fixture set failed to render: {exc}")

    failures.extend(_check_leaf_rule(args.package_dir))

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(
        f"profiling contract OK: {len(fixtures)} fixtures round-trip, "
        "claim gate fails closed, leaf rule holds"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
