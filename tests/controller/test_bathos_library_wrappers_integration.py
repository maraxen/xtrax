"""Integration tests for controller.bathos_library_wrappers against real bathos (LC-07, AC-6;
[GW-01], backlog id 3648).

Deliberately kept separate from test_bathos_library_wrappers.py: that file mocks
sys.modules["bathos"] at import time (to test structure without requiring bathos
installed), which would poison these tests' real `import bathos.stats_gates` calls
if run in the same module. These tests require the real bathos package (controller
extra) and exercise both wrapper functions end-to-end against real return values --
closing the gap flagged in independent audit of PR #70: the mocked test suite never
actually invoked either wrapper function against real bathos behavior.
"""

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from controller.bathos_library_wrappers import (
    call_stats_battery_gate,
    get_evidence_candidate_for_run,
    get_seed_trial_counts,
    get_sidecar_drift_signal,
)
from xtrax.loop.attestation_evidence_gate import EvidenceCandidate
from xtrax.loop.seed_gate import SeedTrialCounts
from xtrax.loop.sidecar_drift_gate import SidecarDriftSignal
from xtrax.loop.stats_battery_gate import BathosStatsBatteryVerdict

# importorskip (not a bare `import duckdb`/`import bathos.stats_gates`) so this whole module
# skips cleanly in CI tiers that don't install the `controller` extra (e.g. tier1_core, which
# only installs `--extra dev` -- a bare import here would fail at collection time and take
# down that tier's entire coverage run, not just this file).
duckdb = pytest.importorskip("duckdb")
pytest.importorskip("bathos.stats_gates")
bathos_catalog = pytest.importorskip("bathos.catalog")
bathos_schema = pytest.importorskip("bathos.schema")


class TestCallStatsBatteryGateRealBathos:
    """call_stats_battery_gate against the real bathos.stats_gates.run_stats_battery."""

    def test_returns_bathos_stats_battery_verdict_with_real_values(self):
        verdict = call_stats_battery_gate(
            baseline_hpo_trials=10,
            candidate_hpo_trials=10,
            candidate_values=[1, 2, 3, 4, 5],
            baseline_values=[0.5, 1.5, 2.5, 3.5, 4.5],
        )

        assert isinstance(verdict, BathosStatsBatteryVerdict)
        assert verdict.verdict in ("pass", "confounded", "underpowered")
        assert verdict.scipy_available is True
        assert isinstance(verdict.reasons, tuple)
        assert verdict.win_rate == pytest.approx(1.0)
        assert verdict.baseline_budget_equivalent is True

    def test_identical_distributions_do_not_pass(self):
        """A candidate identical to baseline should not produce a 'pass' verdict."""
        verdict = call_stats_battery_gate(
            candidate_values=[1.0, 2.0, 3.0, 4.0, 5.0],
            baseline_values=[1.0, 2.0, 3.0, 4.0, 5.0],
        )

        assert verdict.verdict != "pass"
        # bathos treats ties as non-wins (strict >), so identical distributions win 0/5, not 0.5.
        assert verdict.win_rate == pytest.approx(0.0)

    def test_missing_required_stats_arrays_raises_from_bathos(self):
        """candidate_values/baseline_values are required by bathos itself, not optional."""
        with pytest.raises(TypeError, match="candidate_values"):
            call_stats_battery_gate(baseline_hpo_trials=10)


class TestGetSeedTrialCountsRealBathos:
    """get_seed_trial_counts against real bathos.campaigns count functions + real duckdb."""

    @pytest.fixture
    def db(self):
        conn = duckdb.connect(":memory:")
        conn.execute("CREATE TABLE runs (script_sha256 VARCHAR, seed INTEGER, run_id VARCHAR)")
        conn.execute(
            "INSERT INTO runs VALUES "
            "('abc', 1, 'r1'), ('abc', 2, 'r2'), ('abc', 2, 'r3'), "
            "('abc', NULL, 'r4'), ('other', 9, 'r5')"
        )
        yield conn
        conn.close()

    def test_counts_scoped_to_script_sha256(self, db):
        counts = get_seed_trial_counts(db, "abc", hypothesis_clause_id="h1")

        assert isinstance(counts, SeedTrialCounts)
        assert counts.script_sha256 == "abc"
        # distinct non-null seeds for 'abc': {1, 2} -> 2
        assert counts.distinct_seed_count == 2
        # total runs for 'abc': 4 (including the NULL-seed row)
        assert counts.trial_count == 4
        assert counts.hypothesis_clause_id == "h1"

    def test_unknown_script_sha256_returns_zero_counts(self, db):
        counts = get_seed_trial_counts(db, "does-not-exist")

        assert counts.distinct_seed_count == 0
        assert counts.trial_count == 0

    def test_hypothesis_clause_id_defaults_to_empty_string(self, db):
        counts = get_seed_trial_counts(db, "abc")
        assert counts.hypothesis_clause_id == ""


# ---------------------------------------------------------------------------
# get_evidence_candidate_for_run / get_sidecar_drift_signal against a REAL bathos catalog
# ([GW-01], backlog id 3648) -- a cool-tier (pure Parquet) catalog built via bathos's own
# bathos.catalog.init_catalog/write_run, not hand-rolled SQL, matching bathos's real
# get_run/verify_run_manifest/check_sidecar_drift code paths exactly (no bathos.db present,
# so _resolve_backend picks "cool" -- the list_runs/read_runs fallback both functions
# genuinely exercise).
# ---------------------------------------------------------------------------


def _make_run(
    *,
    run_id: str,
    command: str,
    timestamp: datetime,
    manifest_sha256: str = "",
    manifest_path: str = "",
    sidecar_sha256: str = "",
) -> "bathos_schema.Run":
    return bathos_schema.Run(
        project_slug="test-project",
        command=command,
        argv=["python", command],
        git_hash="deadbeef",
        git_branch="main",
        git_dirty=False,
        id=run_id,
        timestamp=timestamp,
        manifest_sha256=manifest_sha256,
        manifest_path=manifest_path,
        sidecar_sha256=sidecar_sha256,
    )


class TestGetEvidenceCandidateForRunRealBathos:
    def test_valid_manifest_verifies_true(self, tmp_path: Path):
        catalog_dir = tmp_path / "catalog"
        bathos_catalog.init_catalog(catalog_dir)

        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text('{"real": "manifest"}')
        manifest_sha256 = hashlib.sha256(manifest_file.read_bytes()).hexdigest()

        run = _make_run(
            run_id="run-valid-manifest",
            command="candidate.py",
            timestamp=datetime.now(UTC),
            manifest_sha256=manifest_sha256,
            manifest_path=str(manifest_file),
        )
        bathos_catalog.write_run(run, catalog_dir)

        candidate = get_evidence_candidate_for_run(
            "run-valid-manifest", catalog_dir=str(catalog_dir)
        )

        assert isinstance(candidate, EvidenceCandidate)
        assert candidate.run_id == "run-valid-manifest"
        assert candidate.manifest_verified is True
        assert candidate.stdout_verified is None

    def test_tampered_manifest_verifies_false(self, tmp_path: Path):
        catalog_dir = tmp_path / "catalog"
        bathos_catalog.init_catalog(catalog_dir)

        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text('{"real": "manifest"}')
        # Recorded hash deliberately does not match the file's real content -- simulates
        # post-hoc tampering (or a stale recorded hash).
        stale_sha256 = hashlib.sha256(b"different content").hexdigest()

        run = _make_run(
            run_id="run-tampered-manifest",
            command="candidate.py",
            timestamp=datetime.now(UTC),
            manifest_sha256=stale_sha256,
            manifest_path=str(manifest_file),
        )
        bathos_catalog.write_run(run, catalog_dir)

        candidate = get_evidence_candidate_for_run(
            "run-tampered-manifest", catalog_dir=str(catalog_dir)
        )

        assert candidate.manifest_verified is False

    def test_no_manifest_recorded_verifies_false(self, tmp_path: Path):
        catalog_dir = tmp_path / "catalog"
        bathos_catalog.init_catalog(catalog_dir)
        run = _make_run(
            run_id="run-no-manifest", command="candidate.py", timestamp=datetime.now(UTC)
        )
        bathos_catalog.write_run(run, catalog_dir)

        candidate = get_evidence_candidate_for_run("run-no-manifest", catalog_dir=str(catalog_dir))

        assert candidate.manifest_verified is False

    def test_unknown_run_id_verifies_false(self, tmp_path: Path):
        catalog_dir = tmp_path / "catalog"
        bathos_catalog.init_catalog(catalog_dir)

        candidate = get_evidence_candidate_for_run("does-not-exist", catalog_dir=str(catalog_dir))

        assert candidate.run_id == "does-not-exist"
        assert candidate.manifest_verified is False

    def test_stdout_verified_forwarded_verbatim(self, tmp_path: Path):
        catalog_dir = tmp_path / "catalog"
        bathos_catalog.init_catalog(catalog_dir)

        candidate = get_evidence_candidate_for_run(
            "does-not-exist", catalog_dir=str(catalog_dir), stdout_verified=True
        )

        assert candidate.stdout_verified is True


class TestGetSidecarDriftSignalRealBathos:
    def test_changed_sidecar_hash_drifts_true(self, tmp_path: Path):
        catalog_dir = tmp_path / "catalog"
        bathos_catalog.init_catalog(catalog_dir)
        script_path = tmp_path / "candidate.py"
        script_path.write_text("x = 1\n")
        now = datetime.now(UTC)

        baseline_run = _make_run(
            run_id="run-baseline",
            command=str(script_path),
            timestamp=now - timedelta(hours=1),
            sidecar_sha256="baseline-hash",
        )
        current_run = _make_run(
            run_id="run-current",
            command=str(script_path),
            timestamp=now,
            sidecar_sha256="different-hash",
        )
        bathos_catalog.write_run(baseline_run, catalog_dir)
        bathos_catalog.write_run(current_run, catalog_dir)

        signal = get_sidecar_drift_signal(
            script_path, "run-current", catalog_dir=str(catalog_dir)
        )

        assert isinstance(signal, SidecarDriftSignal)
        assert signal.drifted is True
        assert signal.current_sha256 == "different-hash"
        assert signal.script_id == str(script_path)

    def test_matching_sidecar_hash_does_not_drift(self, tmp_path: Path):
        catalog_dir = tmp_path / "catalog"
        bathos_catalog.init_catalog(catalog_dir)
        script_path = tmp_path / "candidate.py"
        script_path.write_text("x = 1\n")
        now = datetime.now(UTC)

        baseline_run = _make_run(
            run_id="run-baseline",
            command=str(script_path),
            timestamp=now - timedelta(hours=1),
            sidecar_sha256="same-hash",
        )
        current_run = _make_run(
            run_id="run-current",
            command=str(script_path),
            timestamp=now,
            sidecar_sha256="same-hash",
        )
        bathos_catalog.write_run(baseline_run, catalog_dir)
        bathos_catalog.write_run(current_run, catalog_dir)

        signal = get_sidecar_drift_signal(
            script_path, "run-current", catalog_dir=str(catalog_dir)
        )

        assert signal.drifted is False

    def test_unknown_run_id_yields_no_drift_and_empty_current_hash(self, tmp_path: Path):
        catalog_dir = tmp_path / "catalog"
        bathos_catalog.init_catalog(catalog_dir)

        signal = get_sidecar_drift_signal(
            tmp_path / "candidate.py", "does-not-exist", catalog_dir=str(catalog_dir)
        )

        assert signal.drifted is False
        assert signal.current_sha256 == ""
