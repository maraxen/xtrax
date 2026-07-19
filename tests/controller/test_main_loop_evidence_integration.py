"""Integration tests for run_one_candidate_pass's post-run integrity/provenance gates
([GW-01], backlog id 3648) against a REAL bathos catalog.

Deliberately kept separate from test_main_loop.py: that file's own docstring states "no live
bathos/praxia infrastructure required for any test in this module" -- these tests genuinely
need the real `bathos` package (controller extra) and a real on-disk catalog (built via
bathos.catalog.init_catalog/write_run), so they must not be collected in CI tiers that only
install `--extra dev`. Mirrors test_bathos_library_wrappers_integration.py's own separation
rationale exactly.

These tests omit `evidence_candidate_fn`/`sidecar_drift_signal_fn` entirely, so
run_one_candidate_pass's REAL defaults (`get_evidence_candidate_for_run`/
`get_sidecar_drift_signal`) run against a real catalog -- proving the wiring end-to-end, not
just the injection seam (which test_main_loop.py's own stubbed tests already cover).
"""

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from controller.bathos_campaign_adapter import BathosCampaignAdapter
from controller.dispatch import MockDispatchBackend
from controller.main_loop import run_one_candidate_pass
from xtrax.loop.seed_gate import SeedTrialCounts
from xtrax.loop.sidecar_drift_gate import SidecarHashMismatchError
from xtrax.loop.stats_battery_gate import BathosStatsBatteryVerdict

bathos_catalog = pytest.importorskip("bathos.catalog")
bathos_schema = pytest.importorskip("bathos.schema")
pytest.importorskip("bathos.prereg")
pytest.importorskip("bathos.query")

_CANDIDATE_CONTENT = "candidate-source"
_VALID_SHA256 = hashlib.sha256(_CANDIDATE_CONTENT.encode("utf-8")).hexdigest()
_RUN_ID = "run-evidence-integration"


def _passing_stats_verdict(**_kwargs: Any) -> BathosStatsBatteryVerdict:
    return BathosStatsBatteryVerdict(
        verdict="pass",
        scipy_available=True,
        reasons=(),
        cohens_d=0.5,
        win_rate=0.8,
        breakdown_point=0.3,
        p_superiority=0.8,
        wilcoxon_p_value=0.01,
        icc=0.995,
        baseline_budget_equivalent=True,
    )


def _passing_seed_counts(
    db: Any, script_sha256: str, hypothesis_clause_id: str = ""
) -> SeedTrialCounts:
    return SeedTrialCounts(script_sha256=script_sha256, distinct_seed_count=5, trial_count=40)


def _ok_run_envelope(**extra: Any) -> dict[str, Any]:
    return {
        "ok": True,
        "error_code": None,
        "error": None,
        "resolution_hint": None,
        "script_path": "candidate.py",
        "run_id": _RUN_ID,
        "exit_code": 0,
        "success": True,
        **extra,
    }


class _RecordingTransport:
    def __init__(self, envelope: dict[str, Any]) -> None:
        self.envelope = envelope
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((tool_name, dict(arguments)))
        return self.envelope


def _write_run(
    catalog_dir: Path,
    *,
    manifest_sha256: str = "",
    manifest_path: str = "",
    sidecar_sha256: str = "",
) -> None:
    run = bathos_schema.Run(
        project_slug="test-project",
        command="candidate.py",
        argv=["python", "candidate.py"],
        git_hash="deadbeef",
        git_branch="main",
        git_dirty=False,
        id=_RUN_ID,
        timestamp=datetime.now(UTC),
        manifest_sha256=manifest_sha256,
        manifest_path=manifest_path,
        sidecar_sha256=sidecar_sha256,
    )
    bathos_catalog.write_run(run, catalog_dir)


def _run_pass(
    tmp_path: Path,
    catalog_dir: Path,
    *,
    campaign_mode: str = "confirmation",
    sidecar_drift_agent_mode: str = "collaborative",
):
    dispatch_backend = MockDispatchBackend(
        candidate_path=tmp_path / "candidate.py", candidate_content=_CANDIDATE_CONTENT
    )
    transport = _RecordingTransport(_ok_run_envelope())
    adapter = BathosCampaignAdapter(transport=transport, token="test-token")

    return run_one_candidate_pass(
        dispatch_backend,
        adapter,
        campaign_id="camp-1",
        campaign_mode=campaign_mode,
        candidate_static_fn=lambda path, root=None: None,
        catalog_dir=str(catalog_dir),
        sidecar_drift_agent_mode=sidecar_drift_agent_mode,
        stats_battery_kwargs={},
        stats_battery_fn=_passing_stats_verdict,
        seed_trial_counts_fn=_passing_seed_counts,
    )


class TestRealEvidenceGateEndToEnd:
    def test_valid_manifest_and_no_sidecar_drift_accepts_confirmation_campaign(
        self, tmp_path: Path
    ) -> None:
        catalog_dir = tmp_path / "catalog"
        bathos_catalog.init_catalog(catalog_dir)

        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text('{"real": "manifest"}')
        manifest_sha256 = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
        _write_run(catalog_dir, manifest_sha256=manifest_sha256, manifest_path=str(manifest_file))

        result = _run_pass(tmp_path, catalog_dir, campaign_mode="confirmation")

        assert result.gate_outcome.evidence_integrity.hard_blocked is False
        assert result.accepted is True

    def test_tampered_manifest_hard_blocks_confirmation_campaign(self, tmp_path: Path) -> None:
        catalog_dir = tmp_path / "catalog"
        bathos_catalog.init_catalog(catalog_dir)

        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text('{"real": "manifest"}')
        stale_sha256 = hashlib.sha256(b"different content").hexdigest()
        _write_run(catalog_dir, manifest_sha256=stale_sha256, manifest_path=str(manifest_file))

        result = _run_pass(tmp_path, catalog_dir, campaign_mode="confirmation")

        assert result.gate_outcome.evidence_integrity.hard_blocked is True
        assert result.accepted is False

    def test_tampered_manifest_is_advisory_only_for_exploration_campaign(
        self, tmp_path: Path
    ) -> None:
        catalog_dir = tmp_path / "catalog"
        bathos_catalog.init_catalog(catalog_dir)

        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text('{"real": "manifest"}')
        stale_sha256 = hashlib.sha256(b"different content").hexdigest()
        _write_run(catalog_dir, manifest_sha256=stale_sha256, manifest_path=str(manifest_file))

        result = _run_pass(tmp_path, catalog_dir, campaign_mode="exploration")

        assert result.gate_outcome.evidence_integrity.advisory is True
        assert result.gate_outcome.evidence_integrity.hard_blocked is False
        assert result.accepted is True

    def test_no_manifest_at_all_hard_blocks_confirmation_campaign(self, tmp_path: Path) -> None:
        """No manifest recorded (e.g. a `no_sidecar=True` exploratory run reused under a
        confirmation-mode campaign) is genuinely unverifiable -- correctly hard-blocked, not a
        bug: AC-19's own text is "unverifiable run excluded from evidence.\""""
        catalog_dir = tmp_path / "catalog"
        bathos_catalog.init_catalog(catalog_dir)
        _write_run(catalog_dir)

        result = _run_pass(tmp_path, catalog_dir, campaign_mode="confirmation")

        assert result.gate_outcome.evidence_integrity.hard_blocked is True
        assert result.accepted is False

    def test_sidecar_drift_under_autonomous_mode_raises_and_skips_gate_checks(
        self, tmp_path: Path
    ) -> None:
        catalog_dir = tmp_path / "catalog"
        bathos_catalog.init_catalog(catalog_dir)

        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text('{"real": "manifest"}')
        manifest_sha256 = hashlib.sha256(manifest_file.read_bytes()).hexdigest()

        script_path = tmp_path / "candidate.py"
        baseline_run = bathos_schema.Run(
            project_slug="test-project",
            command=str(script_path),
            argv=["python", str(script_path)],
            git_hash="deadbeef",
            git_branch="main",
            git_dirty=False,
            id="run-baseline",
            timestamp=datetime.now(UTC) - timedelta(hours=1),
            sidecar_sha256="baseline-hash",
        )
        current_run = bathos_schema.Run(
            project_slug="test-project",
            command=str(script_path),
            argv=["python", str(script_path)],
            git_hash="deadbeef",
            git_branch="main",
            git_dirty=False,
            id=_RUN_ID,
            timestamp=datetime.now(UTC),
            manifest_sha256=manifest_sha256,
            manifest_path=str(manifest_file),
            sidecar_sha256="drifted-hash",
        )
        bathos_catalog.write_run(baseline_run, catalog_dir)
        bathos_catalog.write_run(current_run, catalog_dir)

        with pytest.raises(SidecarHashMismatchError):
            _run_pass(
                tmp_path,
                catalog_dir,
                campaign_mode="confirmation",
                sidecar_drift_agent_mode="autonomous",
            )

    def test_sidecar_drift_under_collaborative_mode_only_warns(self, tmp_path: Path) -> None:
        catalog_dir = tmp_path / "catalog"
        bathos_catalog.init_catalog(catalog_dir)

        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text('{"real": "manifest"}')
        manifest_sha256 = hashlib.sha256(manifest_file.read_bytes()).hexdigest()

        script_path = tmp_path / "candidate.py"
        baseline_run = bathos_schema.Run(
            project_slug="test-project",
            command=str(script_path),
            argv=["python", str(script_path)],
            git_hash="deadbeef",
            git_branch="main",
            git_dirty=False,
            id="run-baseline",
            timestamp=datetime.now(UTC) - timedelta(hours=1),
            sidecar_sha256="baseline-hash",
        )
        current_run = bathos_schema.Run(
            project_slug="test-project",
            command=str(script_path),
            argv=["python", str(script_path)],
            git_hash="deadbeef",
            git_branch="main",
            git_dirty=False,
            id=_RUN_ID,
            timestamp=datetime.now(UTC),
            manifest_sha256=manifest_sha256,
            manifest_path=str(manifest_file),
            sidecar_sha256="drifted-hash",
        )
        bathos_catalog.write_run(baseline_run, catalog_dir)
        bathos_catalog.write_run(current_run, catalog_dir)

        result = _run_pass(
            tmp_path,
            catalog_dir,
            campaign_mode="confirmation",
            sidecar_drift_agent_mode="collaborative",
        )

        assert result.gate_outcome.evidence_integrity.sidecar_drift.should_warn is True
        assert result.gate_outcome.evidence_integrity.sidecar_drift.drifted is True
        # Sidecar drift under collaborative mode is a warn, not a hard block -- distinct from
        # the manifest-verification hard_blocked check above.
        assert result.gate_outcome.evidence_integrity.hard_blocked is False
