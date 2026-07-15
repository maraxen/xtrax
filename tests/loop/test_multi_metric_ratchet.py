"""Tests for the multi-metric-regression ratchet (T2-17, #2181, AC-10, F7, PM-2)."""

import math

import pytest

from xtrax.loop.multi_metric_ratchet import (
    RatchetInputError,
    compute_ratchet_decision,
)


class TestCleanWinsAndLosses:
    def test_clean_win_on_all_metrics_is_improved(self) -> None:
        decision = compute_ratchet_decision(
            candidate_fitness={"accuracy": 0.95, "f1": 0.90},
            best_fitness={"accuracy": 0.80, "f1": 0.75},
            higher_is_better={"accuracy": True, "f1": True},
        )
        assert decision.improved is True
        assert decision.win_rate == pytest.approx(1.0)

    def test_clean_loss_on_all_metrics_is_not_improved(self) -> None:
        decision = compute_ratchet_decision(
            candidate_fitness={"accuracy": 0.60, "f1": 0.55},
            best_fitness={"accuracy": 0.80, "f1": 0.75},
            higher_is_better={"accuracy": True, "f1": True},
        )
        assert decision.improved is False
        assert decision.win_rate == pytest.approx(0.0)


class TestFasterButMoreMemoryFailureMode:
    def test_one_metric_improves_a_lot_another_regresses_is_not_improved(self) -> None:
        """AC-10's own cited example: a single-metric check would call this an improvement
        (speed dramatically better) while missing the memory regression. win_rate=0.5 < 0.6
        catches it without needing to weigh the metrics against each other."""
        decision = compute_ratchet_decision(
            candidate_fitness={"speed_ops_per_sec": 1000.0, "memory_mb": 800.0},
            best_fitness={"speed_ops_per_sec": 500.0, "memory_mb": 400.0},
            higher_is_better={"speed_ops_per_sec": True, "memory_mb": False},
        )
        assert decision.improved is False
        assert decision.win_rate == pytest.approx(0.5)


class TestBreakdownPointCatchesFragileWins:
    def test_fragile_majority_win_rate_has_low_breakdown_point(self) -> None:
        """3 of 5 metrics win (win_rate=0.6, exactly at threshold) -- but removing just ONE
        winning metric drops win_rate to 2/4=0.5 < 0.6, so breakdown_point = 1/5 = 0.2 -- exactly
        at its own threshold, not below it. Tighten to 4 metrics (win_rate=0.75) where removing
        one winner still keeps win_rate >= 0.6, to get a case that legitimately fails only via a
        SEPARATE, more fragile construction below.
        """
        decision = compute_ratchet_decision(
            candidate_fitness={"a": 1.0, "b": 1.0, "c": 1.0, "d": 0.0, "e": 0.0},
            best_fitness={"a": 0.0, "b": 0.0, "c": 0.0, "d": 1.0, "e": 1.0},
            higher_is_better={"a": True, "b": True, "c": True, "d": True, "e": True},
        )
        # win_rate = 3/5 = 0.6 (exactly at threshold)
        assert decision.win_rate == pytest.approx(0.6)
        # Removing 1 of the 3 winning metrics -> (3-1)/(5-1) = 0.5 < 0.6, so breakdown_point=1/5.
        assert decision.breakdown_point == pytest.approx(0.2)
        # 0.2 >= 0.2 threshold, so this specific case still passes breakdown_point -- but it's
        # AT the boundary, not comfortably above it; a genuinely more fragile case follows below.

    def test_barely_passing_win_rate_with_single_deciding_metric_fails_breakdown_point(
        self,
    ) -> None:
        """10 metrics, 6 win (win_rate=0.6, exactly at threshold). Removing just 1 winning
        metric drops win_rate to 5/9=0.556 < 0.6, so breakdown_point = 1/10 = 0.1 < 0.2 threshold
        -- improved=False even though win_rate alone would pass. This is the fragile-win case
        breakdown_point exists to catch."""
        candidate = {f"m{i}": (1.0 if i < 6 else 0.0) for i in range(10)}
        best = {f"m{i}": (0.0 if i < 6 else 1.0) for i in range(10)}
        higher_is_better = dict.fromkeys(candidate, True)

        decision = compute_ratchet_decision(candidate, best, higher_is_better=higher_is_better)

        assert decision.win_rate == pytest.approx(0.6)
        assert decision.breakdown_point == pytest.approx(0.1)
        assert decision.breakdown_point < 0.2
        assert decision.improved is False

    def test_already_below_win_rate_threshold_has_zero_breakdown_point(self) -> None:
        decision = compute_ratchet_decision(
            candidate_fitness={"a": 0.0, "b": 0.0, "c": 1.0},
            best_fitness={"a": 1.0, "b": 1.0, "c": 0.0},
            higher_is_better={"a": True, "b": True, "c": True},
        )
        assert decision.win_rate == pytest.approx(1 / 3)
        assert decision.breakdown_point == pytest.approx(0.0)
        assert decision.improved is False


class TestCohensDCatchesInconsistentMagnitude:
    def test_many_trivial_wins_hiding_one_catastrophic_regression_fails_cohens_d(self) -> None:
        """Cohen's d here is mean/stdev of the per-metric relative deltas -- a SCALE-INVARIANT
        consistency measure, not an absolute-magnitude one (uniform tiny wins actually give a
        LARGE d, see TestEdgeCases.test_identical_positive_relative_deltas_give_infinite_cohens_d).
        What it actually catches: 9 metrics each win by a trivial +1%, comfortably clearing
        win_rate (0.9 >= 0.6) and breakdown_point (0.8 >= 0.2, very robust to removing winners) --
        but the 10th metric regresses by -50%, dominating the mean and inflating the variance.
        win_rate/breakdown_point only count WHETHER each metric wins, not by how much; Cohen's d
        is sensitive to a catastrophic single-metric loss that a win-COUNT-based measure would
        treat identically to a trivial one.
        """
        candidate = {f"m{i}": 1.01 for i in range(9)}
        candidate["m9"] = 0.5
        best = {f"m{i}": 1.0 for i in range(10)}
        higher_is_better = dict.fromkeys(best, True)

        decision = compute_ratchet_decision(candidate, best, higher_is_better=higher_is_better)

        assert decision.win_rate == pytest.approx(0.9)
        assert decision.breakdown_point >= 0.2
        assert decision.cohens_d < 0.2
        assert decision.improved is False


class TestDirectionHandling:
    def test_lower_is_better_metric_with_smaller_candidate_value_counts_as_a_win(self) -> None:
        decision = compute_ratchet_decision(
            candidate_fitness={"loss": 0.1, "accuracy": 0.95},
            best_fitness={"loss": 0.5, "accuracy": 0.80},
            higher_is_better={"loss": False, "accuracy": True},
        )
        assert decision.per_metric_delta["loss"] > 0
        assert decision.win_rate == pytest.approx(1.0)
        assert decision.improved is True

    def test_lower_is_better_metric_with_larger_candidate_value_counts_as_a_loss(self) -> None:
        decision = compute_ratchet_decision(
            candidate_fitness={"loss": 0.9, "accuracy": 0.95},
            best_fitness={"loss": 0.5, "accuracy": 0.80},
            higher_is_better={"loss": False, "accuracy": True},
        )
        assert decision.per_metric_delta["loss"] < 0


class TestEdgeCases:
    def test_best_value_zero_falls_back_to_raw_signed_delta(self) -> None:
        decision = compute_ratchet_decision(
            candidate_fitness={"a": 5.0, "b": 1.0},
            best_fitness={"a": 0.0, "b": 1.0},
            higher_is_better={"a": True, "b": True},
        )
        assert decision.per_metric_delta["a"] == pytest.approx(5.0)

    def test_identical_positive_relative_deltas_give_infinite_cohens_d(self) -> None:
        decision = compute_ratchet_decision(
            candidate_fitness={"a": 2.0, "b": 2.0},
            best_fitness={"a": 1.0, "b": 1.0},
            higher_is_better={"a": True, "b": True},
        )
        assert decision.cohens_d == math.inf
        assert decision.improved is True

    def test_identical_nonpositive_relative_deltas_give_zero_cohens_d_not_nan(self) -> None:
        decision = compute_ratchet_decision(
            candidate_fitness={"a": 1.0, "b": 1.0},
            best_fitness={"a": 1.0, "b": 1.0},
            higher_is_better={"a": True, "b": True},
        )
        assert decision.cohens_d == 0.0
        assert not math.isnan(decision.cohens_d)
        assert decision.improved is False


class TestInputValidation:
    def test_mismatched_metric_keys_raises(self) -> None:
        with pytest.raises(RatchetInputError, match="same metric keys"):
            compute_ratchet_decision(
                candidate_fitness={"a": 1.0, "b": 2.0},
                best_fitness={"a": 1.0, "c": 2.0},
                higher_is_better={"a": True, "b": True},
            )

    def test_higher_is_better_missing_a_key_raises(self) -> None:
        with pytest.raises(RatchetInputError, match="same metric keys"):
            compute_ratchet_decision(
                candidate_fitness={"a": 1.0, "b": 2.0},
                best_fitness={"a": 1.0, "b": 2.0},
                higher_is_better={"a": True},
            )

    def test_single_metric_raises_cohens_d_degenerate(self) -> None:
        with pytest.raises(RatchetInputError, match="at least 2 metrics"):
            compute_ratchet_decision(
                candidate_fitness={"a": 1.0},
                best_fitness={"a": 0.5},
                higher_is_better={"a": True},
            )

    def test_zero_metrics_raises(self) -> None:
        with pytest.raises(RatchetInputError, match="at least 2 metrics"):
            compute_ratchet_decision(
                candidate_fitness={},
                best_fitness={},
                higher_is_better={},
            )
