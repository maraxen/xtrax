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

from controller.bathos_library_wrappers import call_stats_battery_gate, get_seed_trial_counts
from xtrax.loop.seed_gate import SeedTrialCounts
from xtrax.loop.stats_battery_gate import BathosStatsBatteryVerdict

# importorskip (not a bare `import duckdb`/`import bathos.stats_gates`) so this whole module
# skips cleanly in CI tiers that don't install the `controller` extra (e.g. tier1_core, which
# only installs `--extra dev` -- a bare import here would fail at collection time and take
# down that tier's entire coverage run, not just this file).
duckdb = pytest.importorskip("duckdb")
pytest.importorskip("bathos.stats_gates")


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
