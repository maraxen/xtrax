"""Tests for the T2-23 conclude-time seed/trial power-floor gate (#2181, AC-16, conclude half)."""

import pytest

from xtrax.loop.seed_gate import (
    MIN_SEEDS,
    MIN_TRIALS,
    SeedGateInputError,
    SeedTrialCounts,
    assess_seed_trial_floor,
)


def _counts(*, seeds: int, trials: int, clause: str = "") -> SeedTrialCounts:
    return SeedTrialCounts(
        script_sha256="deadbeef" * 8,
        distinct_seed_count=seeds,
        trial_count=trials,
        hypothesis_clause_id=clause,
    )


class TestFloorClearedHeldRegardlessOfMode:
    def test_exploration(self):
        decision = assess_seed_trial_floor(_counts(seeds=3, trials=29), campaign_mode="exploration")
        assert decision.held is True
        assert decision.hard_blocked is False
        assert decision.advisory is False

    def test_confirmation(self):
        decision = assess_seed_trial_floor(
            _counts(seeds=5, trials=40), campaign_mode="confirmation"
        )
        assert decision.held is True
        assert decision.hard_blocked is False
        assert decision.advisory is False

    def test_sequential(self):
        decision = assess_seed_trial_floor(_counts(seeds=3, trials=29), campaign_mode="sequential")
        assert decision.held is True
        assert decision.hard_blocked is False
        assert decision.advisory is False


class TestExactBoundaryValues:
    def test_exactly_three_seeds_and_29_trials_holds(self):
        decision = assess_seed_trial_floor(
            _counts(seeds=MIN_SEEDS, trials=MIN_TRIALS), campaign_mode="confirmation"
        )
        assert decision.held is True

    def test_two_seeds_29_trials_does_not_hold(self):
        decision = assess_seed_trial_floor(
            _counts(seeds=MIN_SEEDS - 1, trials=MIN_TRIALS), campaign_mode="confirmation"
        )
        assert decision.held is False
        assert decision.hard_blocked is True

    def test_three_seeds_28_trials_does_not_hold(self):
        decision = assess_seed_trial_floor(
            _counts(seeds=MIN_SEEDS, trials=MIN_TRIALS - 1), campaign_mode="confirmation"
        )
        assert decision.held is False
        assert decision.hard_blocked is True


class TestUnmetFloorAdvisoryForExploration:
    def test_zero_seeds_zero_trials(self):
        decision = assess_seed_trial_floor(_counts(seeds=0, trials=0), campaign_mode="exploration")
        assert decision.held is False
        assert decision.hard_blocked is False
        assert decision.advisory is True

    def test_below_both_thresholds(self):
        decision = assess_seed_trial_floor(_counts(seeds=1, trials=5), campaign_mode="exploration")
        assert decision.held is False
        assert decision.advisory is True
        assert decision.hard_blocked is False


class TestUnmetFloorHardBlocksConfirmationAndSequential:
    def test_zero_runs_confirmation(self):
        decision = assess_seed_trial_floor(_counts(seeds=0, trials=0), campaign_mode="confirmation")
        assert decision.held is False
        assert decision.hard_blocked is True
        assert decision.advisory is False

    def test_zero_runs_sequential(self):
        decision = assess_seed_trial_floor(_counts(seeds=0, trials=0), campaign_mode="sequential")
        assert decision.held is False
        assert decision.hard_blocked is True
        assert decision.advisory is False

    def test_enough_seeds_not_enough_trials_confirmation(self):
        decision = assess_seed_trial_floor(
            _counts(seeds=5, trials=10), campaign_mode="confirmation"
        )
        assert decision.held is False
        assert decision.hard_blocked is True

    def test_enough_trials_not_enough_seeds_sequential(self):
        decision = assess_seed_trial_floor(_counts(seeds=1, trials=50), campaign_mode="sequential")
        assert decision.held is False
        assert decision.hard_blocked is True


class TestInvalidCampaignModeRaises:
    def test_unrecognized_mode_raises_regardless_of_counts(self):
        with pytest.raises(SeedGateInputError, match="confirmatory"):
            assess_seed_trial_floor(_counts(seeds=3, trials=29), campaign_mode="confirmatory")

    def test_unrecognized_mode_raises_on_unmet_floor_too(self):
        with pytest.raises(SeedGateInputError, match="confirmatory"):
            assess_seed_trial_floor(_counts(seeds=0, trials=0), campaign_mode="confirmatory")


class TestMalformedCountsRaise:
    def test_negative_seed_count_raises(self):
        with pytest.raises(SeedGateInputError, match="non-negative"):
            assess_seed_trial_floor(_counts(seeds=-1, trials=10), campaign_mode="exploration")

    def test_negative_trial_count_raises(self):
        with pytest.raises(SeedGateInputError, match="non-negative"):
            assess_seed_trial_floor(_counts(seeds=1, trials=-1), campaign_mode="exploration")

    def test_seed_count_exceeding_trial_count_raises(self):
        with pytest.raises(SeedGateInputError, match="cannot exceed"):
            assess_seed_trial_floor(_counts(seeds=5, trials=2), campaign_mode="exploration")


class TestHypothesisClauseIdAndScriptShaThreadThrough:
    def test_fields_carried_into_decision(self):
        decision = assess_seed_trial_floor(
            _counts(seeds=3, trials=29, clause="clause-a"), campaign_mode="confirmation"
        )
        assert decision.hypothesis_clause_id == "clause-a"
        assert decision.script_sha256 == "deadbeef" * 8
        assert decision.distinct_seed_count == 3
        assert decision.trial_count == 29

    def test_default_clause_id_is_empty_string(self):
        decision = assess_seed_trial_floor(
            _counts(seeds=3, trials=29), campaign_mode="confirmation"
        )
        assert decision.hypothesis_clause_id == ""


class TestSeedTrialCounts:
    def test_is_frozen(self):
        counts = _counts(seeds=3, trials=29)
        with pytest.raises(AttributeError):
            counts.distinct_seed_count = 4
