"""Tests for distribution N10 release-readiness convergence gate (#1462)."""

from __future__ import annotations

import json
import subprocess
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.audit_release_readiness import (
    audit_release_readiness,
    build_backlog_report,
    compute_verdict,
    load_release_readiness_config,
    verify_workflow_markers,
)
from xtrax.run.freshness import ProbeResult

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "distribution" / "release_readiness.toml"


def test_load_release_readiness_config_reads_committed_toml() -> None:
    config = load_release_readiness_config(CONFIG_PATH)
    assert config.version == "0.1.0"
    assert config.epic_id == 1451
    assert len(config.backlog_items) == 11
    assert len(config.automated_checks) >= 8
    human = [item for item in config.backlog_items if item.gate_type == "human"]
    assert len(human) == 1
    assert human[0].item_id == 1454


def test_verify_workflow_markers_passes_on_publish_workflow() -> None:
    """The real publish workflow carries every marker the gate config requires.

    Both arguments come from the config rather than being restated here. The
    hardcoded pair this replaced kept asserting 'publish-testpypi' for two
    months after the TestPyPI job was deliberately removed, which is the exact
    drift a duplicated list invites.
    """
    config = load_release_readiness_config(CONFIG_PATH)
    failures = verify_workflow_markers(
        ROOT,
        config.publish_workflow,
        config.required_publish_markers,
    )
    assert failures == []


def test_human_gate_config_carries_attestation_fields() -> None:
    config = load_release_readiness_config(CONFIG_PATH)
    n9 = next(item for item in config.backlog_items if item.item_id == 1454)
    assert n9.attested_at == "2026-07-02T00:00:00Z"
    assert n9.ttl_days == 90.0
    assert n9.probe == "pypi_and_git_tag"


def test_human_gate_requires_attestation_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "release_readiness.toml"
    config_path.write_text(
        textwrap.dedent(
            """
            [readiness]
            version = "0.1.0"
            epic_id = 1
            epic_title = "test"
            report_json = ".praxia/report.json"
            report_markdown = ".praxia/report.md"

            [[backlog_items]]
            id = 1
            slug = "human_gate_missing_attestation"
            title = "test"
            gate_type = "human"
            blocking = true

            [[automated_checks]]
            name = "x"
            command = "true"
            category = "test"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="attested_at and ttl_days"):
        load_release_readiness_config(config_path)


def test_build_backlog_report_human_gate_fresh_within_ttl() -> None:
    config = load_release_readiness_config(CONFIG_PATH)
    now = datetime(2026, 7, 10, tzinfo=UTC)  # 8 days into the 90-day TTL

    with patch.dict(
        "scripts.audit_release_readiness.PROBES",
        {"pypi_and_git_tag": lambda version, root: ProbeResult(invalidated=False)},
    ):
        rows = build_backlog_report(config, {}, root=ROOT, package_version="0.3.0", now=now)

    n9 = next(row for row in rows if row["id"] == 1454)
    assert n9["status"] == "completed"
    assert n9["gate_passed"] is True
    assert n9["freshness_reasons"] == []


def test_build_backlog_report_human_gate_ttl_expired() -> None:
    config = load_release_readiness_config(CONFIG_PATH)
    now = datetime(2027, 1, 1, tzinfo=UTC)  # ~180 days after attestation, past the 90-day TTL

    with patch.dict(
        "scripts.audit_release_readiness.PROBES",
        {"pypi_and_git_tag": lambda version, root: ProbeResult(invalidated=False)},
    ):
        rows = build_backlog_report(config, {}, root=ROOT, package_version="0.3.0", now=now)

    n9 = next(row for row in rows if row["id"] == 1454)
    assert n9["status"] == "blocked"
    assert n9["gate_passed"] is False
    assert any("past TTL" in reason for reason in n9["freshness_reasons"])


def test_build_backlog_report_human_gate_invalidated_by_probe() -> None:
    config = load_release_readiness_config(CONFIG_PATH)
    now = datetime(2026, 7, 10, tzinfo=UTC)  # within TTL, but the probe invalidates it

    with patch.dict(
        "scripts.audit_release_readiness.PROBES",
        {
            "pypi_and_git_tag": lambda version, root: ProbeResult(
                invalidated=True, reason="git tag not found"
            )
        },
    ):
        rows = build_backlog_report(config, {}, root=ROOT, package_version="0.3.0", now=now)

    n9 = next(row for row in rows if row["id"] == 1454)
    assert n9["status"] == "blocked"
    assert n9["gate_passed"] is False
    assert any("git tag not found" in reason for reason in n9["freshness_reasons"])


def test_compute_verdict_blocks_on_human_gate() -> None:
    config = load_release_readiness_config(CONFIG_PATH)
    verdict, reasons = compute_verdict(
        config,
        automated_results=[{"name": "x", "blocking": True, "passed": True}],
        backlog_rows=[
            {
                "id": 1454,
                "slug": "n9_human_oidc",
                "gate_type": "human",
                "blocking": True,
                "status": "open",
            }
        ],
        workflow_failures=[],
    )
    assert verdict == "BLOCKED_MANUAL"
    assert any("human gate open" in reason for reason in reasons)


def test_audit_release_readiness_writes_report_with_mocks(tmp_path: Path) -> None:
    config_dir = tmp_path / "distribution"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "release_readiness.toml"
    config_path.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    for name in ("ci.yml", "publish.yml", "docs.yml"):
        path = tmp_path / ".github" / "workflows" / name
        path.write_text(
            textwrap.dedent(
                """
                audit-deterministic
                audit-coverage-tier1
                audit-coverage-tier2
                --doctest-modules src/xtrax/io/
                publish-pypi
                id-token: write
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
    (tmp_path / "src" / "xtrax").mkdir(parents=True)
    (tmp_path / "src" / "xtrax" / "__init__.py").write_text(
        '__version__ = "0.3.0"\n',
        encoding="utf-8",
    )

    def fake_run(cmd, cwd, capture_output, text, check):  # noqa: ANN001
        if cmd[:2] == ["uv", "sync"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 0, "PASS", "")

    with patch("scripts.audit_release_readiness.subprocess.run", side_effect=fake_run):
        passed, payload = audit_release_readiness(
            tmp_path,
            config_path,
            skip_sync=True,
            quick=False,
        )

    assert passed is False
    assert payload["verdict"] == "BLOCKED_MANUAL"
    report_path = tmp_path / ".praxia" / "release_readiness_report.json"
    assert report_path.is_file()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["epic_id"] == 1451
    assert len(saved["backlog"]) == 11


def test_main_cli_generates_report_on_repo(tmp_path: Path) -> None:
    """The CLI runs end to end on the real repo and writes a usable report.

    ``--report-root`` is not optional decoration here (#5003). This test runs
    inside the ``audit-deterministic`` chain, so without it every local run of
    that chain overwrote the *tracked* ``.praxia/release_readiness_report.json``
    -- replacing the committed post-publish ``mode=full`` / ``READY``
    attestation with this test's ``mode=quick`` / ``BLOCKED_INCOMPLETE`` output,
    whose "blockers" are all just --quick skips. A full release run rewrote it
    afterwards, so it self-corrected; an interrupted one left a misleading
    artifact that reads exactly like a real regression.
    """
    tracked_report = ROOT / ".praxia" / "release_readiness_report.json"
    before = tracked_report.read_bytes() if tracked_report.is_file() else None

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/audit_release_readiness.py",
            "--quick",
            "--skip-sync",
            "--report-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    # The tracked artifact is untouched -- the regression this test exists for.
    after = tracked_report.read_bytes() if tracked_report.is_file() else None
    assert after == before, (
        "running the readiness CLI rewrote the tracked release_readiness_report.json (#5003)"
    )

    # 0 = READY, 1 = a real verdict of not-ready. Anything else (2 = argparse,
    # >2 = traceback) means the CLI broke, which the old `returncode in (0, 1)`
    # inside an `or` could not distinguish from success.
    assert result.returncode in (0, 1), result.stderr

    written = tmp_path / ".praxia" / "release_readiness_report.json"
    assert written.is_file(), result.stderr
    payload = json.loads(written.read_text(encoding="utf-8"))

    # Assert the content the test claims to verify, not merely that a file
    # exists -- the old `is_file()` check passed on the committed file even if
    # the script had crashed before writing anything.
    assert payload["mode"] == "quick"
    assert payload["verdict"] in {
        "READY",
        "BLOCKED_AUTOMATED",
        "BLOCKED_MANUAL",
        "BLOCKED_INCOMPLETE",
    }
    assert payload["epic_id"] == 1451
    assert payload["package_version"]
    assert (tmp_path / ".praxia" / "release_readiness_report.md").is_file()
    assert "Release readiness verdict:" in result.stdout


def test_report_root_defaults_to_root(tmp_path: Path) -> None:
    """Omitting ``report_root`` keeps the release-time behaviour: write under root."""
    config_path = ROOT / "distribution" / "release_readiness.toml"
    config = load_release_readiness_config(config_path)

    def fake_run(cmd, cwd, capture_output, text, check):  # noqa: ANN001
        return subprocess.CompletedProcess(cmd, 0, "PASS", "")

    (tmp_path / "src" / "xtrax").mkdir(parents=True)
    (tmp_path / "src" / "xtrax" / "__init__.py").write_text(
        '__version__ = "0.0.1"\n', encoding="utf-8"
    )

    with patch("scripts.audit_release_readiness.subprocess.run", side_effect=fake_run):
        audit_release_readiness(tmp_path, config_path, skip_sync=True, quick=True)

    assert (tmp_path / config.report_json).is_file()
    assert (tmp_path / config.report_markdown).is_file()


def test_justfile_defines_audit_release_readiness() -> None:
    text = (ROOT / "Justfile").read_text(encoding="utf-8")
    assert "audit-release-readiness:" in text
