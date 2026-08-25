#!/usr/bin/env python3
"""Distribution N10 epic-done release-readiness convergence audit (#1462)."""

from __future__ import annotations

import argparse
import functools
import json
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xtrax.run.freshness import (
    Attestation,
    FreshnessVerdict,
    ProbeResult,
    evaluate_freshness,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "distribution" / "release_readiness.toml"


@dataclass(frozen=True)
class BacklogItem:
    item_id: int
    slug: str
    title: str
    gate: str | None
    gate_type: str
    expected_status: str
    blocking: bool
    slow: bool
    notes: str
    attested_at: str | None = None
    ttl_days: float | None = None
    probe: str | None = None


@dataclass(frozen=True)
class AutomatedCheck:
    name: str
    recipe: str | None
    command: str | None
    category: str
    blocking: bool
    slow: bool


@dataclass(frozen=True)
class ReleaseReadinessConfig:
    version: str
    epic_id: int
    epic_title: str
    report_json: str
    report_markdown: str
    prerequisite_sync: tuple[str, ...]
    prerequisite_groups: tuple[str, ...]
    require_automated_pass: bool
    block_on_open_human_gates: bool
    backlog_items: tuple[BacklogItem, ...]
    automated_checks: tuple[AutomatedCheck, ...]
    ci_workflow: str
    publish_workflow: str
    docs_workflow: str
    required_ci_markers: tuple[str, ...]
    required_publish_markers: tuple[str, ...]


def load_release_readiness_config(config_path: Path) -> ReleaseReadinessConfig:
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    section = data.get("readiness")
    if not isinstance(section, dict):
        raise ValueError(f"missing [readiness] section in {config_path}")

    version = section.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("readiness.version must be a non-empty string")

    epic_id = section.get("epic_id")
    if not isinstance(epic_id, int):
        raise ValueError("readiness.epic_id must be an integer")

    epic_title = section.get("epic_title")
    if not isinstance(epic_title, str) or not epic_title:
        raise ValueError("readiness.epic_title must be a non-empty string")

    def _req_str(key: str) -> str:
        value = section.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"readiness.{key} must be a non-empty string")
        return value

    sync_extras = section.get("prerequisite_sync", [])
    if not isinstance(sync_extras, list):
        raise ValueError("readiness.prerequisite_sync must be a list")
    sync_groups = section.get("prerequisite_groups", [])
    if not isinstance(sync_groups, list):
        raise ValueError("readiness.prerequisite_groups must be a list")

    verdict = section.get("verdict", {})
    if not isinstance(verdict, dict):
        raise ValueError("readiness.verdict must be a table")
    require_automated = verdict.get("require_automated_pass", True)
    block_human = verdict.get("block_on_open_human_gates", True)
    if not isinstance(require_automated, bool) or not isinstance(block_human, bool):
        raise ValueError("readiness.verdict flags must be booleans")

    backlog_items: list[BacklogItem] = []
    raw_backlog = data.get("backlog_items")
    if not isinstance(raw_backlog, list) or not raw_backlog:
        raise ValueError("release_readiness must define backlog_items")
    for raw in raw_backlog:
        if not isinstance(raw, dict):
            raise ValueError("each backlog_items entry must be a table")
        item_id = raw.get("id")
        if not isinstance(item_id, int):
            raise ValueError("backlog_items[].id must be an integer")
        slug = raw.get("slug")
        title = raw.get("title")
        if not isinstance(slug, str) or not isinstance(title, str):
            raise ValueError("backlog_items[].slug and title must be strings")
        gate = raw.get("gate")
        if gate is not None and not isinstance(gate, str):
            raise ValueError("backlog_items[].gate must be a string when set")
        gate_type = raw.get("gate_type", "automated")
        if not isinstance(gate_type, str):
            raise ValueError("backlog_items[].gate_type must be a string")
        expected_status = raw.get("expected_status", "completed")
        if not isinstance(expected_status, str):
            raise ValueError("backlog_items[].expected_status must be a string")
        blocking = raw.get("blocking", True)
        slow = raw.get("slow", False)
        notes = raw.get("notes", "")
        if not isinstance(blocking, bool) or not isinstance(slow, bool):
            raise ValueError("backlog_items[].blocking/slow must be booleans")
        if not isinstance(notes, str):
            raise ValueError("backlog_items[].notes must be a string")
        attested_at = raw.get("attested_at")
        if attested_at is not None and not isinstance(attested_at, str):
            raise ValueError("backlog_items[].attested_at must be a string when set")
        ttl_days = raw.get("ttl_days")
        if ttl_days is not None and not isinstance(ttl_days, (int, float)):
            raise ValueError("backlog_items[].ttl_days must be a number when set")
        probe_name = raw.get("probe")
        if probe_name is not None and not isinstance(probe_name, str):
            raise ValueError("backlog_items[].probe must be a string when set")
        if gate_type == "human" and (attested_at is None or ttl_days is None):
            raise ValueError(
                f"backlog_items[{slug!r}]: gate_type='human' requires both "
                "attested_at and ttl_days (freshness primitive, T3-05/AC-X6)"
            )
        backlog_items.append(
            BacklogItem(
                item_id=item_id,
                slug=slug,
                title=title,
                gate=gate,
                gate_type=gate_type,
                expected_status=expected_status,
                blocking=blocking,
                slow=slow,
                notes=notes,
                attested_at=attested_at,
                ttl_days=float(ttl_days) if ttl_days is not None else None,
                probe=probe_name,
            )
        )

    automated_checks: list[AutomatedCheck] = []
    raw_checks = data.get("automated_checks")
    if not isinstance(raw_checks, list) or not raw_checks:
        raise ValueError("release_readiness must define automated_checks")
    for raw in raw_checks:
        if not isinstance(raw, dict):
            raise ValueError("each automated_checks entry must be a table")
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("automated_checks[].name must be a non-empty string")
        recipe = raw.get("recipe")
        command = raw.get("command")
        if (recipe is None) == (command is None):
            raise ValueError(
                f"automated_checks[{name!r}] requires exactly one of recipe or command"
            )
        category = raw.get("category", "general")
        if not isinstance(category, str):
            raise ValueError("automated_checks[].category must be a string")
        blocking = raw.get("blocking", True)
        slow = raw.get("slow", False)
        if not isinstance(blocking, bool) or not isinstance(slow, bool):
            raise ValueError("automated_checks[].blocking/slow must be booleans")
        automated_checks.append(
            AutomatedCheck(
                name=name,
                recipe=str(recipe) if recipe is not None else None,
                command=str(command) if command is not None else None,
                category=category,
                blocking=blocking,
                slow=slow,
            )
        )

    markers = section.get("workflow_markers", {})
    if not isinstance(markers, dict):
        raise ValueError("readiness.workflow_markers must be a table")

    def _marker_list(key: str) -> tuple[str, ...]:
        values = markers.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"readiness.workflow_markers.{key} must be a non-empty list")
        return tuple(str(item) for item in values)

    return ReleaseReadinessConfig(
        version=version,
        epic_id=epic_id,
        epic_title=epic_title,
        report_json=_req_str("report_json"),
        report_markdown=_req_str("report_markdown"),
        prerequisite_sync=tuple(str(item) for item in sync_extras),
        prerequisite_groups=tuple(str(item) for item in sync_groups),
        require_automated_pass=require_automated,
        block_on_open_human_gates=block_human,
        backlog_items=tuple(backlog_items),
        automated_checks=tuple(automated_checks),
        ci_workflow=str(markers.get("ci_workflow", ".github/workflows/ci.yml")),
        publish_workflow=str(markers.get("publish_workflow", ".github/workflows/publish.yml")),
        docs_workflow=str(markers.get("docs_workflow", ".github/workflows/docs.yml")),
        required_ci_markers=_marker_list("required_ci_markers"),
        required_publish_markers=_marker_list("required_publish_markers"),
    )


def run_prerequisite_sync(root: Path, config: ReleaseReadinessConfig) -> tuple[bool, str]:
    cmd = ["uv", "sync"]
    for extra in config.prerequisite_sync:
        cmd.append(f"--extra={extra}")
    for group in config.prerequisite_groups:
        cmd.append(f"--group={group}")
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


def run_check(
    root: Path,
    check: AutomatedCheck,
) -> dict[str, Any]:
    if check.recipe is not None:
        cmd = ["just", check.recipe]
    else:
        cmd = check.command.split() if check.command else []
    result = subprocess.run(
        cmd,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    tail = "\n".join(output.splitlines()[-12:]) if output else ""
    return {
        "name": check.name,
        "category": check.category,
        "blocking": check.blocking,
        "slow": check.slow,
        "command": " ".join(cmd),
        "passed": result.returncode == 0,
        "exit_code": result.returncode,
        "output_tail": tail,
    }


def verify_workflow_markers(
    root: Path,
    rel_path: str,
    markers: tuple[str, ...],
) -> list[str]:
    path = root / rel_path
    if not path.is_file():
        return [f"missing workflow file: {rel_path}"]
    text = path.read_text(encoding="utf-8")
    failures: list[str] = []
    for marker in markers:
        if marker not in text:
            failures.append(f"{rel_path} missing marker: {marker!r}")
    return failures


def load_coverage_state(root: Path) -> dict[str, Any]:
    state_path = root / ".praxia" / "coverage_last_measured.json"
    if not state_path.is_file():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


#: Named probes for the freshness primitive (T3-05/AC-X6), keyed by the
#: `[[backlog_items]].probe` TOML field. Each probe can only INVALIDATE an
#: attestation, never satisfy one — see xtrax.run.freshness.
def probe_pypi_and_git_tag(version: str, root: Path) -> ProbeResult:
    """Invalidate n9 (PyPI OIDC) if there's no matching git tag or PyPI release.

    The git-tag check is hermetic (no network). The PyPI check is opportunistic:
    a genuine 404 (no such release) invalidates, but any network failure is
    reported as `skipped`, never as an invalidation.
    """
    tag = f"v{version}"
    tag_result = subprocess.run(
        ["git", "tag", "-l", tag],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if tag_result.returncode != 0 or tag_result.stdout.strip() != tag:
        return ProbeResult(invalidated=True, reason=f"git tag {tag!r} not found locally")

    url = f"https://pypi.org/pypi/xtrax/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status == 200:
                return ProbeResult(invalidated=False)
            return ProbeResult(
                invalidated=True,
                reason=f"PyPI returned HTTP {response.status} for xtrax=={version}",
            )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return ProbeResult(invalidated=True, reason=f"PyPI has no release xtrax=={version}")
        return ProbeResult(invalidated=False, skipped=True, reason=f"PyPI probe HTTP error: {exc}")
    except (urllib.error.URLError, OSError) as exc:
        return ProbeResult(invalidated=False, skipped=True, reason=f"PyPI probe unreachable: {exc}")


PROBES: dict[str, Any] = {"pypi_and_git_tag": probe_pypi_and_git_tag}


def build_backlog_report(
    config: ReleaseReadinessConfig,
    check_results: dict[str, dict[str, Any]],
    *,
    root: Path,
    package_version: str,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in config.backlog_items:
        row: dict[str, Any] = {
            "id": item.item_id,
            "slug": item.slug,
            "title": item.title,
            "expected_status": item.expected_status,
            "blocking": item.blocking,
            "gate_type": item.gate_type,
            "notes": item.notes,
        }
        if item.gate_type == "human":
            if item.attested_at is None or item.ttl_days is None:
                raise ValueError(
                    f"backlog item {item.item_id} ({item.slug!r}): human gate missing "
                    "attested_at/ttl_days (should have been rejected at config load)"
                )
            probe_fn = None
            if item.probe is not None:
                if item.probe not in PROBES:
                    raise ValueError(
                        f"backlog item {item.item_id} ({item.slug!r}): unknown probe {item.probe!r}"
                    )
                named_probe = PROBES[item.probe]
                probe_fn = functools.partial(named_probe, package_version, root)
            verdict: FreshnessVerdict = evaluate_freshness(
                Attestation(
                    attested_at=item.attested_at,
                    ttl_days=item.ttl_days,
                    attested_by=item.slug,
                ),
                now=now,
                probe=probe_fn,
            )
            row["status"] = "completed" if verdict.fresh else "blocked"
            row["gate_passed"] = verdict.fresh
            row["freshness_reasons"] = list(verdict.reasons)
        elif item.gate_type == "meta":
            row["status"] = item.expected_status
            row["gate_passed"] = True
        elif item.gate:
            result = check_results.get(item.gate)
            if result is None:
                # Map gate recipe to check by running gate directly in backlog phase
                row["status"] = "not_run"
                row["gate_passed"] = False
            else:
                row["status"] = "completed" if result["passed"] else "failed"
                row["gate_passed"] = result["passed"]
        else:
            row["status"] = item.expected_status
            row["gate_passed"] = item.expected_status == "completed"
        rows.append(row)
    return rows


def compute_verdict(
    config: ReleaseReadinessConfig,
    *,
    automated_results: list[dict[str, Any]],
    backlog_rows: list[dict[str, Any]],
    workflow_failures: list[str],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if workflow_failures:
        reasons.extend(workflow_failures)

    failed_blocking = [
        item for item in automated_results if item["blocking"] and not item["passed"]
    ]
    if config.require_automated_pass and failed_blocking:
        for item in failed_blocking:
            reasons.append(f"automated check failed: {item['name']}")

    open_human = [
        row
        for row in backlog_rows
        if row["gate_type"] == "human" and row["blocking"] and row["status"] != "completed"
    ]
    if config.block_on_open_human_gates and open_human:
        for row in open_human:
            freshness_reasons = row.get("freshness_reasons") or []
            detail = f" ({'; '.join(freshness_reasons)})" if freshness_reasons else ""
            reasons.append(f"human gate open: #{row['id']} {row['slug']}{detail}")

    failed_backlog = [
        row
        for row in backlog_rows
        if row["blocking"]
        and row["gate_type"] != "human"
        and not row.get("gate_passed", False)
        and row["id"] != 1462
    ]
    for row in failed_backlog:
        reasons.append(f"backlog gate failed: #{row['id']} {row['slug']}")

    if reasons:
        if failed_blocking or failed_backlog or workflow_failures:
            return "BLOCKED_AUTOMATED", reasons
        return "BLOCKED_MANUAL", reasons
    return "READY", []


def render_markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# xtrax Release Readiness Report",
        "",
        f"- **Epic:** #{payload['epic_id']} {payload['epic_title']}",
        f"- **Generated:** {payload['timestamp']}",
        f"- **Verdict:** `{payload['verdict']}`",
        f"- **Package version:** `{payload.get('package_version', 'unknown')}`",
        "",
    ]
    if payload["verdict_reasons"]:
        lines.append("## Blockers")
        lines.extend(f"- {reason}" for reason in payload["verdict_reasons"])
        lines.append("")

    lines.extend(
        [
            "## Distribution backlog (N0-N10)",
            "",
            "| ID | Slug | Status | Gate | Blocking |",
            "|----|------|--------|------|----------|",
        ]
    )
    for row in payload["backlog"]:
        status = row.get("status", "")
        passed = "PASS" if row.get("gate_passed") else "FAIL"
        if row.get("gate_type") == "human":
            passed = "MANUAL"
        if row.get("gate_type") == "meta":
            passed = "META"
        lines.append(
            f"| {row['id']} | {row['slug']} | {status} | {passed} | "
            f"{'yes' if row['blocking'] else 'no'} |"
        )

    lines.extend(["", "## Automated checks", ""])
    for check in payload["automated_checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- **{check['name']}** ({check['category']}): {mark}")
        if not check["passed"] and check.get("output_tail"):
            lines.append(f"  ```\n  {check['output_tail']}\n  ```")

    coverage = payload.get("coverage", {})
    if coverage:
        lines.extend(["", "## Coverage state", ""])
        tiers = coverage.get("tiers", {})
        for tier_id, tier in tiers.items():
            line = tier.get("line_pct")
            branch = tier.get("branch_pct")
            if line is not None and branch is not None:
                lines.append(f"- `{tier_id}`: line {line:.1f}% / branch {branch:.1f}%")

    lines.extend(
        [
            "",
            "## Release policy",
            "",
            "- Do **not** push release tags until this report is `READY`.",
            "- Complete human gate #1454 (PyPI + TestPyPI Trusted Publisher) first.",
            "- Re-run: `just audit-release-readiness`",
            "",
        ]
    )
    return "\n".join(lines)


def read_package_version(root: Path) -> str:
    init_path = root / "src" / "xtrax" / "__init__.py"
    for line in init_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            return line.split("=", 1)[1].strip().strip("\"'")
    return "unknown"


def audit_release_readiness(
    root: Path,
    config_path: Path,
    *,
    quick: bool = False,
    skip_sync: bool = False,
) -> tuple[bool, dict[str, Any]]:
    config = load_release_readiness_config(config_path)
    failures: list[str] = []

    if not skip_sync:
        sync_ok, sync_error = run_prerequisite_sync(root, config)
        if not sync_ok:
            failures.append(f"prerequisite sync failed: {sync_error}")

    workflow_failures: list[str] = []
    workflow_failures.extend(
        verify_workflow_markers(root, config.ci_workflow, config.required_ci_markers)
    )
    workflow_failures.extend(
        verify_workflow_markers(root, config.publish_workflow, config.required_publish_markers)
    )
    if not (root / config.docs_workflow).is_file():
        workflow_failures.append(f"missing workflow file: {config.docs_workflow}")

    automated_results: list[dict[str, Any]] = []
    gate_results: dict[str, dict[str, Any]] = {}

    # Run backlog gate recipes first (dedupe with automated_checks)
    for item in config.backlog_items:
        if item.gate and item.gate_type == "automated":
            if item.gate in gate_results:
                continue
            if quick and item.slow:
                gate_results[item.gate] = {
                    "name": item.gate,
                    "passed": False,
                    "skipped": True,
                    "output_tail": "skipped in --quick mode",
                }
                continue
            result = run_check(
                root,
                AutomatedCheck(
                    name=item.gate,
                    recipe=item.gate,
                    command=None,
                    category="backlog",
                    blocking=item.blocking,
                    slow=item.slow,
                ),
            )
            gate_results[item.gate] = result
            automated_results.append(result)

    for check in config.automated_checks:
        if check.recipe and check.recipe in gate_results:
            automated_results.append(gate_results[check.recipe])
            continue
        if quick and check.slow:
            automated_results.append(
                {
                    "name": check.name,
                    "category": check.category,
                    "blocking": check.blocking,
                    "slow": True,
                    "command": check.recipe or check.command or "",
                    "passed": False,
                    "exit_code": 0,
                    "output_tail": "skipped in --quick mode",
                    "skipped": True,
                }
            )
            continue
        result = run_check(root, check)
        automated_results.append(result)
        if check.recipe:
            gate_results[check.recipe] = result

    package_version = read_package_version(root)
    backlog_rows = build_backlog_report(
        config, gate_results, root=root, package_version=package_version
    )
    for row in backlog_rows:
        row["gate"] = next(
            (item.gate for item in config.backlog_items if item.item_id == row["id"]),
            None,
        )

    verdict, verdict_reasons = compute_verdict(
        config,
        automated_results=automated_results,
        backlog_rows=backlog_rows,
        workflow_failures=workflow_failures,
    )
    if failures:
        verdict = "BLOCKED_AUTOMATED"
        verdict_reasons = failures + verdict_reasons

    payload: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "config_version": config.version,
        "epic_id": config.epic_id,
        "epic_title": config.epic_title,
        "package_version": package_version,
        "mode": "quick" if quick else "full",
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
        "workflow_failures": workflow_failures,
        "backlog": backlog_rows,
        "automated_checks": automated_results,
        "coverage": load_coverage_state(root),
        "release_policy": {
            "push_tags_allowed": verdict == "READY",
            "trigger_publish_workflow_allowed": verdict == "READY",
            "human_gate_1454_required": True,
        },
    }

    report_json = root / config.report_json
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report_md = root / config.report_markdown
    report_md.write_text(render_markdown_report(payload), encoding="utf-8")

    passed = verdict == "READY"
    return passed, payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to release_readiness.toml",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip slow checks (coverage tiers, docs build, full deterministic)",
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Skip uv sync prerequisite step",
    )
    args = parser.parse_args(argv)

    passed, payload = audit_release_readiness(
        root=args.root.resolve(),
        config_path=args.config.resolve(),
        quick=args.quick,
        skip_sync=args.skip_sync,
    )
    config = load_release_readiness_config(args.config.resolve())
    print(f"Release readiness verdict: {payload['verdict']}")
    print(f"JSON report: {args.root.resolve() / config.report_json}")
    print(f"Markdown report: {args.root.resolve() / config.report_markdown}")
    if passed:
        print("PASS: release readiness gate")
        return 0

    print("FAIL: release readiness gate", file=sys.stderr)
    for reason in payload["verdict_reasons"]:
        print(f"  - {reason}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
