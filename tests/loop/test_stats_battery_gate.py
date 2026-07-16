"""Tests for the T2-22 conclude-time stats-battery gate (#2181, AC-15)."""

import pytest

from xtrax.loop.stats_battery_gate import (
    BathosStatsBatteryVerdict,
    StatsBatteryGateInputError,
    assess_stats_battery_verdict,
)


def _verdict(verdict, *, reasons=()):
    return BathosStatsBatteryVerdict(
        verdict=verdict,
        scipy_available=verdict != "underpowered",
        reasons=reasons,
        cohens_d=0.3,
        win_rate=0.7,
        breakdown_point=0.25,
        p_superiority=0.8,
        wilcoxon_p_value=0.01,
        icc=0.995,
        baseline_budget_equivalent=True,
    )


class TestPassVerdictHonoredRegardlessOfMode:
    def test_exploration(self):
        decision = assess_stats_battery_verdict(_verdict("pass"), campaign_mode="exploration")
        assert decision.honored is True
        assert decision.hard_blocked is False
        assert decision.advisory is False

    def test_confirmation(self):
        decision = assess_stats_battery_verdict(_verdict("pass"), campaign_mode="confirmation")
        assert decision.honored is True
        assert decision.hard_blocked is False
        assert decision.advisory is False

    def test_sequential(self):
        decision = assess_stats_battery_verdict(_verdict("pass"), campaign_mode="sequential")
        assert decision.honored is True
        assert decision.hard_blocked is False
        assert decision.advisory is False


class TestDowngradeAdvisoryForExploration:
    def test_confounded(self):
        reasons = ("cohens_d below threshold",)
        decision = assess_stats_battery_verdict(
            _verdict("confounded", reasons=reasons), campaign_mode="exploration"
        )
        assert decision.honored is False
        assert decision.hard_blocked is False
        assert decision.advisory is True
        assert decision.reasons == reasons

    def test_underpowered(self):
        reasons = ("scipy was not installed",)
        decision = assess_stats_battery_verdict(
            _verdict("underpowered", reasons=reasons), campaign_mode="exploration"
        )
        assert decision.honored is False
        assert decision.hard_blocked is False
        assert decision.advisory is True
        assert decision.reasons == reasons


class TestDowngradeHardBlocksConfirmationAndSequential:
    def test_confounded_confirmation(self):
        decision = assess_stats_battery_verdict(
            _verdict("confounded"), campaign_mode="confirmation"
        )
        assert decision.honored is False
        assert decision.hard_blocked is True
        assert decision.advisory is False

    def test_underpowered_confirmation(self):
        decision = assess_stats_battery_verdict(
            _verdict("underpowered"), campaign_mode="confirmation"
        )
        assert decision.honored is False
        assert decision.hard_blocked is True
        assert decision.advisory is False

    def test_confounded_sequential(self):
        decision = assess_stats_battery_verdict(_verdict("confounded"), campaign_mode="sequential")
        assert decision.honored is False
        assert decision.hard_blocked is True
        assert decision.advisory is False

    def test_underpowered_sequential(self):
        decision = assess_stats_battery_verdict(
            _verdict("underpowered"), campaign_mode="sequential"
        )
        assert decision.honored is False
        assert decision.hard_blocked is True
        assert decision.advisory is False


class TestInvalidCampaignModeRaises:
    def test_unrecognized_mode_raises_regardless_of_verdict(self):
        with pytest.raises(StatsBatteryGateInputError, match="confirmatory"):
            assess_stats_battery_verdict(_verdict("pass"), campaign_mode="confirmatory")

    def test_unrecognized_mode_raises_on_downgrade_too(self):
        with pytest.raises(StatsBatteryGateInputError, match="confirmatory"):
            assess_stats_battery_verdict(_verdict("confounded"), campaign_mode="confirmatory")


class TestBathosStatsBatteryVerdict:
    def test_is_frozen(self):
        verdict = _verdict("pass")
        with pytest.raises(AttributeError):
            verdict.verdict = "confounded"
