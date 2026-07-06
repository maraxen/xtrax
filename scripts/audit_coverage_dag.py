#!/usr/bin/env python3
"""Distribution N6 coverage DAG manifest + per-tier baseline reporter (#1456 phase 1)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from xtrax.devtools.gates.test_rigor import parse_coverage_json, parse_pytest_summary

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "distribution" / "coverage_dag.toml"
DEFAULT_TIER = "tier1_core"


@dataclass(frozen=True)
class Tier:
    id: str
    description: str
    measure_coverage: bool
    uv_sync_extras: tuple[str, ...]
    pytest_args: tuple[str, ...]
    coverage_packages: tuple[str, ...] = ()
    coverage_omit: tuple[str, ...] = ()
    target_line_pct: float | None = None
    target_branch_pct: float | None = None
    enforce_line_pct: float | None = None
    enforce_branch_pct: float | None = None


@dataclass(frozen=True)
class CoverageDag:
    version: str
    state_path: str
    tiers: tuple[Tier, ...]


@dataclass(frozen=True)
class TierResult:
    tier_id: str
    measure_coverage: bool
    line_pct: float | None
    branch_pct: float | None
    tests_run: int
    tests_failed: int
    pytest_exit_code: int
    enforce_passed: bool | None = None
    enforce_failures: tuple[str, ...] = ()


def _optional_float(value: object, field: str, tier_id: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError(f"tiers[{tier_id!r}].{field} must be a number")
    return float(value)


def load_coverage_dag(config_path: Path) -> CoverageDag:
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    dag = data.get("dag")
    if not isinstance(dag, dict):
        raise ValueError(f"missing [dag] section in {config_path}")

    version = dag.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("dag.version must be a non-empty string")

    state_path = dag.get("state_path")
    if not isinstance(state_path, str) or not state_path:
        raise ValueError("dag.state_path must be a non-empty string")

    raw_tiers = data.get("tiers")
    if not isinstance(raw_tiers, list) or not raw_tiers:
        raise ValueError("coverage DAG must define at least one [[tiers]] entry")

    tiers: list[Tier] = []
    for raw in raw_tiers:
        if not isinstance(raw, dict):
            raise ValueError("each [[tiers]] entry must be a table")
        tier_id = raw.get("id")
        if not isinstance(tier_id, str) or not tier_id:
            raise ValueError("tiers[].id must be a non-empty string")

        description = raw.get("description", "")
        if not isinstance(description, str):
            raise ValueError(f"tiers[{tier_id!r}].description must be a string")

        measure_coverage = raw.get("measure_coverage")
        if not isinstance(measure_coverage, bool):
            raise ValueError(f"tiers[{tier_id!r}].measure_coverage must be a boolean")

        extras = raw.get("uv_sync_extras")
        if not isinstance(extras, list) or not extras:
            raise ValueError(f"tiers[{tier_id!r}].uv_sync_extras must be a non-empty list")
        if not all(isinstance(item, str) and item for item in extras):
            raise ValueError(f"tiers[{tier_id!r}].uv_sync_extras must contain strings")

        pytest_args = raw.get("pytest_args")
        if not isinstance(pytest_args, list) or not pytest_args:
            raise ValueError(f"tiers[{tier_id!r}].pytest_args must be a non-empty list")
        if not all(isinstance(item, str) and item for item in pytest_args):
            raise ValueError(f"tiers[{tier_id!r}].pytest_args must contain strings")

        packages = raw.get("coverage_packages", [])
        if packages is None:
            packages = []
        if not isinstance(packages, list):
            raise ValueError(f"tiers[{tier_id!r}].coverage_packages must be a list")
        if measure_coverage and not packages:
            raise ValueError(
                f"tiers[{tier_id!r}].coverage_packages required when measure_coverage=true"
            )
        if not all(isinstance(item, str) and item for item in packages):
            raise ValueError(f"tiers[{tier_id!r}].coverage_packages must contain strings")

        omit = raw.get("coverage_omit", [])
        if omit is None:
            omit = []
        if not isinstance(omit, list):
            raise ValueError(f"tiers[{tier_id!r}].coverage_omit must be a list")
        if not all(isinstance(item, str) and item for item in omit):
            raise ValueError(f"tiers[{tier_id!r}].coverage_omit must contain strings")

        tiers.append(
            Tier(
                id=tier_id,
                description=description,
                measure_coverage=measure_coverage,
                uv_sync_extras=tuple(extras),
                pytest_args=tuple(pytest_args),
                coverage_packages=tuple(packages),
                coverage_omit=tuple(omit),
                target_line_pct=_optional_float(
                    raw.get("target_line_pct"), "target_line_pct", tier_id
                ),
                target_branch_pct=_optional_float(
                    raw.get("target_branch_pct"), "target_branch_pct", tier_id
                ),
                enforce_line_pct=_optional_float(
                    raw.get("enforce_line_pct"), "enforce_line_pct", tier_id
                ),
                enforce_branch_pct=_optional_float(
                    raw.get("enforce_branch_pct"), "enforce_branch_pct", tier_id
                ),
            )
        )

    return CoverageDag(version=version, state_path=state_path, tiers=tuple(tiers))


def run_uv_sync(root: Path, extras: tuple[str, ...]) -> tuple[bool, str]:
    cmd = ["uv", "sync", *[f"--extra={extra}" for extra in extras]]
    result = subprocess.run(
        cmd,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or "uv sync failed"
        return False, stderr
    return True, ""


def write_coverage_config(path: Path, tier: Tier) -> None:
    """Write a tier-scoped coverage config (omit/source packages)."""
    lines = ["[run]", "branch = True"]
    if tier.coverage_packages:
        if len(tier.coverage_packages) == 1:
            lines.append(f"source = {tier.coverage_packages[0]}")
        else:
            lines.append("source =")
            lines.extend(f"    {package}" for package in tier.coverage_packages)
    if tier.coverage_omit:
        lines.append("omit =")
        lines.extend(f"    {pattern}" for pattern in tier.coverage_omit)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_tier_pytest(
    root: Path,
    tier: Tier,
) -> TierResult:
    sync_ok, sync_error = run_uv_sync(root, tier.uv_sync_extras)
    if not sync_ok:
        return TierResult(
            tier_id=tier.id,
            measure_coverage=tier.measure_coverage,
            line_pct=None,
            branch_pct=None,
            tests_run=0,
            tests_failed=1,
            pytest_exit_code=1,
        )

    env = {**os.environ, "PYTEST_ADDOPTS": ""}
    cmd = ["uv", "run", "pytest", *tier.pytest_args, "-o", "addopts="]
    cov_path: Path | None = None
    cov_config_path: Path | None = None

    if tier.measure_coverage:
        handle = tempfile.NamedTemporaryFile(
            suffix=".json",
            prefix="coverage-dag-",
            delete=False,
        )
        cov_path = Path(handle.name)
        handle.close()
        config_handle = tempfile.NamedTemporaryFile(
            suffix=".ini",
            prefix="coverage-dag-config-",
            delete=False,
        )
        cov_config_path = Path(config_handle.name)
        config_handle.close()
        write_coverage_config(cov_config_path, tier)
        for package in tier.coverage_packages:
            cmd.append(f"--cov={package}")
        cmd.extend(
            [
                "--cov-branch",
                f"--cov-config={cov_config_path}",
                f"--cov-report=json:{cov_path}",
            ]
        )

    try:
        result = subprocess.run(
            cmd,
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        combined = f"{result.stdout}\n{result.stderr}"
        tests_run, tests_failed = parse_pytest_summary(combined)
        if result.returncode != 0 or tests_failed > 0:
            tail = "\n".join(combined.strip().splitlines()[-40:])
            print(f"pytest output tail (tier {tier.id}):\n{tail}", file=sys.stderr)

        line_pct: float | None = None
        branch_pct: float | None = None
        if tier.measure_coverage:
            if cov_path is None or not cov_path.is_file():
                tests_failed = max(tests_failed, 1)
            else:
                line_pct, branch_pct = parse_coverage_json(cov_path)

        return TierResult(
            tier_id=tier.id,
            measure_coverage=tier.measure_coverage,
            line_pct=line_pct,
            branch_pct=branch_pct,
            tests_run=tests_run,
            tests_failed=tests_failed,
            pytest_exit_code=result.returncode,
        )
    finally:
        if cov_path is not None:
            cov_path.unlink(missing_ok=True)
        if cov_config_path is not None:
            cov_config_path.unlink(missing_ok=True)


def evaluate_enforce(tier: Tier, result: TierResult) -> TierResult:
    if not tier.measure_coverage:
        return result

    failures: list[str] = []
    if result.tests_failed > 0 or result.pytest_exit_code != 0:
        failures.append(
            f"pytest failed ({result.tests_failed} failures, exit {result.pytest_exit_code})"
        )
    if tier.enforce_line_pct is not None:
        if result.line_pct is None:
            failures.append("line coverage missing")
        elif result.line_pct < tier.enforce_line_pct:
            failures.append(
                f"line {result.line_pct:.1f}% < enforce floor {tier.enforce_line_pct:.1f}%"
            )
    if tier.enforce_branch_pct is not None:
        if result.branch_pct is None:
            failures.append("branch coverage missing")
        elif result.branch_pct < tier.enforce_branch_pct:
            failures.append(
                f"branch {result.branch_pct:.1f}% < enforce floor {tier.enforce_branch_pct:.1f}%"
            )

    if tier.enforce_line_pct is None and tier.enforce_branch_pct is None:
        enforce_passed = None
    else:
        enforce_passed = len(failures) == 0

    return TierResult(
        tier_id=result.tier_id,
        measure_coverage=result.measure_coverage,
        line_pct=result.line_pct,
        branch_pct=result.branch_pct,
        tests_run=result.tests_run,
        tests_failed=result.tests_failed,
        pytest_exit_code=result.pytest_exit_code,
        enforce_passed=enforce_passed,
        enforce_failures=tuple(failures),
    )


def build_state_payload(
    dag: CoverageDag,
    results: tuple[TierResult, ...],
) -> dict[str, object]:
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "dag_version": dag.version,
        "tiers": {
            result.tier_id: {
                "measure_coverage": result.measure_coverage,
                "line_pct": result.line_pct,
                "branch_pct": result.branch_pct,
                "tests_run": result.tests_run,
                "tests_failed": result.tests_failed,
                "pytest_exit_code": result.pytest_exit_code,
                "enforce_passed": result.enforce_passed,
                "enforce_failures": list(result.enforce_failures),
            }
            for result in results
        },
    }


def write_state(root: Path, dag: CoverageDag, payload: dict[str, object]) -> Path:
    state_path = root / dag.state_path
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return state_path


def format_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}%"


def print_results_table(results: tuple[TierResult, ...]) -> None:
    header = f"{'tier':<14} {'line':>8} {'branch':>8} {'failed':>8} {'cov':>5}"
    print(header)
    print("-" * len(header))
    for result in results:
        cov_flag = "yes" if result.measure_coverage else "no"
        print(
            f"{result.tier_id:<14} "
            f"{format_pct(result.line_pct):>8} "
            f"{format_pct(result.branch_pct):>8} "
            f"{result.tests_failed:>8} "
            f"{cov_flag:>5}"
        )


def audit_coverage_dag(
    *,
    root: Path,
    config_path: Path,
    tiers: tuple[Tier, ...],
    enforce_tier: str | None = None,
) -> tuple[bool, tuple[TierResult, ...], list[str]]:
    dag = load_coverage_dag(config_path)
    tier_by_id = {tier.id: tier for tier in dag.tiers}

    if enforce_tier is not None and enforce_tier not in tier_by_id:
        return False, (), [f"unknown enforce tier: {enforce_tier!r}"]

    results: list[TierResult] = []
    failures: list[str] = []

    for tier in tiers:
        result = run_tier_pytest(root, tier)
        if enforce_tier == tier.id:
            result = evaluate_enforce(tier, result)
            if result.enforce_passed is False:
                failures.extend(result.enforce_failures)
        results.append(result)

    payload = build_state_payload(dag, tuple(results))
    write_state(root, dag, payload)

    return len(failures) == 0, tuple(results), failures


def select_tiers(
    dag: CoverageDag,
    *,
    tier_id: str | None,
    all_tiers: bool,
) -> tuple[Tier, ...]:
    if all_tiers:
        return dag.tiers
    selected = tier_id or DEFAULT_TIER
    tier_by_id = {tier.id: tier for tier in dag.tiers}
    if selected not in tier_by_id:
        raise ValueError(f"unknown tier: {selected!r}")
    return (tier_by_id[selected],)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to coverage_dag.toml",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--tier",
        default=None,
        help=f"Measure a single tier (default: {DEFAULT_TIER})",
    )
    parser.add_argument(
        "--all-tiers",
        action="store_true",
        help="Measure every tier in the DAG",
    )
    parser.add_argument(
        "--enforce",
        metavar="TIER_ID",
        default=None,
        help="Fail when measured coverage is below enforce_* floors for this tier",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    config_path = args.config.resolve()
    dag = load_coverage_dag(config_path)

    try:
        tiers = select_tiers(dag, tier_id=args.tier, all_tiers=args.all_tiers)
    except ValueError as exc:
        print(f"FAIL: coverage DAG — {exc}", file=sys.stderr)
        return 1

    passed, results, failures = audit_coverage_dag(
        root=root,
        config_path=config_path,
        tiers=tiers,
        enforce_tier=args.enforce,
    )

    print("coverage DAG baseline report")
    print_results_table(results)
    state_path = root / dag.state_path
    print(f"state: {state_path.relative_to(root)}")

    if args.enforce and not passed:
        print("FAIL: coverage DAG enforce", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    if args.enforce is None:
        print("PASS: coverage DAG report-only (non-blocking)")
        return 0

    print("PASS: coverage DAG enforce")
    return 0


if __name__ == "__main__":
    sys.exit(main())
