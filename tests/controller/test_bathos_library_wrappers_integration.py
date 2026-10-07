"""Integration tests for controller.bathos_library_wrappers against real bathos (LC-07, AC-6).

Deliberately kept separate from test_bathos_library_wrappers.py: that file mocks
sys.modules["bathos"] at import time (to test structure without requiring bathos
installed), which would poison these tests' real `import bathos.stats_gates` calls
if run in the same module. These tests require the real bathos package (controller
extra) and exercise both wrapper functions end-to-end against real return values --
closing the gap flagged in independent audit of PR #70: the mocked test suite never
actually invoked either wrapper function against real bathos behavior.
"""

import pytest

from controller.bathos_library_wrappers import (
    call_stats_battery_gate,
    get_capability_probe_result,
    get_seed_trial_counts,
    get_sidecar_drift_signal,
)
from xtrax.loop.capability_probe_gate import CapabilityProbeResult
from xtrax.loop.seed_gate import SeedTrialCounts
from xtrax.loop.sidecar_drift_gate import SidecarDriftSignal
from xtrax.loop.stats_battery_gate import BathosStatsBatteryVerdict

# importorskip (not a bare `import duckdb`/`import bathos.stats_gates`) so this whole module
# skips cleanly in CI tiers that don't install the `controller` extra (e.g. tier1_core, which
# only installs `--extra dev` -- a bare import here would fail at collection time and take
# down that tier's entire coverage run, not just this file).
duckdb = pytest.importorskip("duckdb")
pytest.importorskip("bathos.stats_gates")
pytest.importorskip("bathos.capability")


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


class TestGetCapabilityProbeResultRealBathos:
    """get_capability_probe_result against the real bathos.capability.probe_capabilities."""

    def test_returns_capability_probe_result_with_real_values(self, tmp_path):
        """Test against a fresh empty tmp catalog dir."""
        result = get_capability_probe_result(catalog_dir=str(tmp_path))

        assert isinstance(result, CapabilityProbeResult)
        # AUDIT-REQUIRED (fixed): do NOT hardcode stats_battery_live is True -- scipy is not
        # installed in this venv (controller extra is only bathos>=0.13.0a1, no [stats]).
        # Assert the type instead of a specific value.
        assert isinstance(result.seed_live, bool)
        assert isinstance(result.stats_battery_live, bool)

    def test_default_catalog_dir_returns_capability_probe_result(self, tmp_path):
        """Test with default catalog_dir (empty string) using isolated temp directory.

        This test verifies the function works when called without an explicit catalog_dir
        by setting BTH_CATALOG_DIR env var to tmp_path, ensuring isolation from the real
        ~/.bth catalog directory.
        """
        import os

        # Set BTH_CATALOG_DIR to override the default catalog_dir resolution path
        old_env = os.environ.get("BTH_CATALOG_DIR")
        try:
            os.environ["BTH_CATALOG_DIR"] = str(tmp_path)
            result = get_capability_probe_result()

            assert isinstance(result, CapabilityProbeResult)
            assert isinstance(result.seed_live, bool)
            assert isinstance(result.stats_battery_live, bool)
        finally:
            # Restore original environment
            if old_env is None:
                os.environ.pop("BTH_CATALOG_DIR", None)
            else:
                os.environ["BTH_CATALOG_DIR"] = old_env


class TestGetSidecarDriftSignalRealBathos:
    """get_sidecar_drift_signal against the real bathos.prereg.check_sidecar_drift.

    This wrapper had no coverage at all before the tier4_controller gate existed: it is
    only reachable with the `controller` extra installed, and no gate installed it. Each
    test below names the branch it is here to pin, because the wrapper's four decisions
    (catalog default, hash lookup, lookup result, script id default) are what the gate
    measures -- covering only the happy path would leave the gate red on branch.

    Ground truth for the assertions: against an empty catalog directory, real bathos
    returns `check_sidecar_drift(...) == False` ("no prior baseline to compare against"),
    and `get_run(<unknown>) is None`. Both were confirmed against bathos 0.13.0a1 before
    these assertions were written.
    """

    @pytest.fixture
    def script(self, tmp_path):
        path = tmp_path / "candidate.py"
        path.write_text("print('candidate')\n")
        return path

    def test_explicit_catalog_and_hash_reports_no_drift(self, script, tmp_path):
        """catalog_dir set, hash supplied: no catalog lookup, drift False on an empty catalog."""
        signal = get_sidecar_drift_signal(
            script_path=str(script),
            catalog_dir=str(tmp_path),
            current_sidecar_sha256="abc123",
        )

        assert isinstance(signal, SidecarDriftSignal)
        # False here means "no prior baseline", not "hashes matched" -- bathos does not
        # distinguish the two and neither does the wrapper. See SidecarDriftSignal.drifted.
        assert signal.drifted is False
        assert signal.current_sha256 == "abc123"
        # script_id defaults to the script's stem, not its full path.
        assert signal.script_id == "candidate"
        # The wrapper never queries the baseline; it documents this as caller-owned.
        assert signal.first_run_sha256 == ""

    def test_explicit_script_id_overrides_the_stem_default(self, script, tmp_path):
        signal = get_sidecar_drift_signal(
            script_path=str(script),
            catalog_dir=str(tmp_path),
            current_sidecar_sha256="abc123",
            script_id="human-readable-id",
        )

        assert signal.script_id == "human-readable-id"

    def test_absent_hash_with_run_id_falls_back_to_a_catalog_lookup(self, script, tmp_path):
        """An empty hash plus a run_id takes the bathos.query.get_run path.

        The run does not exist in this empty catalog, so the lookup returns None and the
        hash stays empty -- the wrapper must not crash on the miss, which is the whole
        point of the `run_obj is not None` guard.
        """
        signal = get_sidecar_drift_signal(
            script_path=str(script),
            catalog_dir=str(tmp_path),
            current_sidecar_sha256="",
            run_id="no-such-run",
        )

        assert signal.drifted is False
        assert signal.current_sha256 == ""

    def test_absent_hash_without_run_id_skips_the_lookup_entirely(self, script, tmp_path):
        """Neither hash nor run_id: the `and run_id` short-circuit must skip the lookup."""
        signal = get_sidecar_drift_signal(
            script_path=str(script),
            catalog_dir=str(tmp_path),
            current_sidecar_sha256="",
        )

        assert signal.drifted is False
        assert signal.current_sha256 == ""

    def test_default_catalog_dir_resolves_under_home(self, script, tmp_path, monkeypatch):
        """An empty catalog_dir resolves to ~/.bth.

        Home is redirected at the pathlib level rather than through BTH_CATALOG_DIR,
        because the wrapper computes `Path.home() / ".bth"` itself and never consults
        that variable -- pointing the env var at tmp_path would leave this test reading
        the developer's real catalog while appearing to be isolated.
        """
        import pathlib

        monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))

        signal = get_sidecar_drift_signal(
            script_path=str(script),
            current_sidecar_sha256="abc123",
        )

        assert signal.drifted is False
        assert signal.current_sha256 == "abc123"
