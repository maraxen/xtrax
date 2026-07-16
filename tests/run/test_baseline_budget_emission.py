"""Tests for the T2-24 baseline-budget-equivalence emission (#2181, AC-17)."""

import math

import pytest

from xtrax.run.baseline_budget_emission import (
    BaselineBudgetCounts,
    BaselineBudgetCountsInputError,
    baseline_budget_counts,
)


class TestBaselineBudgetCounts:
    def test_all_values_supplied(self):
        counts = baseline_budget_counts(
            candidate_hpo_trials=10,
            baseline_hpo_trials=12,
            candidate_hpo_compute_budget=100.0,
            baseline_hpo_compute_budget=150.0,
        )
        assert counts == BaselineBudgetCounts(
            candidate_hpo_trials=10,
            baseline_hpo_trials=12,
            candidate_hpo_compute_budget=100.0,
            baseline_hpo_compute_budget=150.0,
        )

    def test_all_defaults_are_none(self):
        counts = baseline_budget_counts()
        assert counts == BaselineBudgetCounts(
            candidate_hpo_trials=None,
            baseline_hpo_trials=None,
            candidate_hpo_compute_budget=None,
            baseline_hpo_compute_budget=None,
        )

    def test_zero_counts_are_valid(self):
        # A candidate/baseline with zero trials recorded is meaningful (e.g. a
        # zero-HPO-budget baseline) and must not be conflated with "not tracked" (None).
        counts = baseline_budget_counts(
            candidate_hpo_trials=0,
            baseline_hpo_trials=0,
            candidate_hpo_compute_budget=0.0,
            baseline_hpo_compute_budget=0.0,
        )
        assert counts.candidate_hpo_trials == 0
        assert counts.baseline_hpo_trials == 0
        assert counts.candidate_hpo_compute_budget == 0.0
        assert counts.baseline_hpo_compute_budget == 0.0

    def test_partial_dimensions_supplied(self):
        # Only the trials dimension tracked; compute-budget dimension left None on both
        # sides -- matches bathos's own per-dimension opt-in stance.
        counts = baseline_budget_counts(candidate_hpo_trials=5, baseline_hpo_trials=5)
        assert counts.candidate_hpo_compute_budget is None
        assert counts.baseline_hpo_compute_budget is None

    def test_no_matching_baseline_recorded_is_all_none_not_an_error(self):
        # A candidate with no matching baseline recorded at all -- every field left at its
        # None default -- must construct cleanly; "nothing to compare" is bathos's own
        # documented stance, not a structural error here.
        counts = baseline_budget_counts(candidate_hpo_trials=20)
        assert counts.candidate_hpo_trials == 20
        assert counts.baseline_hpo_trials is None
        assert counts.candidate_hpo_compute_budget is None
        assert counts.baseline_hpo_compute_budget is None

    @pytest.mark.parametrize(
        ("field_name", "negative_value"),
        [
            ("candidate_hpo_trials", -1),
            ("baseline_hpo_trials", -1),
            ("candidate_hpo_compute_budget", -1.0),
            ("baseline_hpo_compute_budget", -1.0),
        ],
    )
    def test_negative_value_raises(self, field_name: str, negative_value: float):
        with pytest.raises(BaselineBudgetCountsInputError, match=field_name):
            baseline_budget_counts(**{field_name: negative_value})

    def test_negative_zero_float_is_not_negative(self):
        # -0.0 < 0 is False in Python; document that this is accepted, not a false-negative
        # gap in the validation.
        counts = baseline_budget_counts(candidate_hpo_compute_budget=-0.0)
        assert counts.candidate_hpo_compute_budget == 0.0

    @pytest.mark.parametrize(
        ("field_name", "non_finite_value"),
        [
            ("candidate_hpo_compute_budget", math.nan),
            ("baseline_hpo_compute_budget", math.nan),
            ("candidate_hpo_compute_budget", math.inf),
            ("baseline_hpo_compute_budget", math.inf),
            ("candidate_hpo_compute_budget", -math.inf),
            ("baseline_hpo_compute_budget", -math.inf),
        ],
    )
    def test_non_finite_value_raises(self, field_name: str, non_finite_value: float):
        # NaN/Infinity would otherwise silently pass bathos's own comparison (a NaN
        # comparison is always False), producing a false "equivalent" verdict instead of
        # AC-17's own fast/loud downgrade -- must be rejected here instead.
        with pytest.raises(BaselineBudgetCountsInputError, match=field_name):
            baseline_budget_counts(**{field_name: non_finite_value})


class TestAsStatsBatteryKwargs:
    def test_produces_the_exact_bathos_keyword_names(self):
        counts = BaselineBudgetCounts(
            candidate_hpo_trials=10,
            baseline_hpo_trials=12,
            candidate_hpo_compute_budget=100.0,
            baseline_hpo_compute_budget=150.0,
        )
        kwargs = counts.as_stats_battery_kwargs()
        assert kwargs == {
            "baseline_hpo_trials": 12,
            "candidate_hpo_trials": 10,
            "baseline_hpo_compute_budget": 150.0,
            "candidate_hpo_compute_budget": 100.0,
        }

    def test_none_fields_pass_through_as_none(self):
        counts = BaselineBudgetCounts()
        kwargs = counts.as_stats_battery_kwargs()
        assert kwargs == {
            "baseline_hpo_trials": None,
            "candidate_hpo_trials": None,
            "baseline_hpo_compute_budget": None,
            "candidate_hpo_compute_budget": None,
        }

    def test_kwargs_are_spreadable_into_a_function_call(self):
        # Simulates the real caller shape: `check_baseline_budget_equivalence(**kwargs)`
        # (bathos-side, B2-01) without importing bathos itself.
        def fake_check_baseline_budget_equivalence(
            *,
            baseline_hpo_trials,
            candidate_hpo_trials,
            baseline_hpo_compute_budget,
            candidate_hpo_compute_budget,
        ):
            return (
                baseline_hpo_trials,
                candidate_hpo_trials,
                baseline_hpo_compute_budget,
                candidate_hpo_compute_budget,
            )

        counts = BaselineBudgetCounts(candidate_hpo_trials=3, baseline_hpo_trials=5)
        result = fake_check_baseline_budget_equivalence(**counts.as_stats_battery_kwargs())
        assert result == (5, 3, None, None)
