"""Synthetic ground truth for the reference-parity harness."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from jax.scipy.stats import norm as norm_dist

from xtrax.testing import (
    InjectedSource,
    NegativeControl,
    NegativeControlError,
    SelfParityError,
    align_logits,
    assert_distinct_callables,
    collapsed_sampler_lane,
    distributional_lane,
    knob_coverage,
    map_token_ids,
    min_detectable_tv,
    order_from_randn,
    require_negative_control,
    teacher_forced_lane,
)

_ROOT = Path(__file__).resolve().parents[2]


def test_order_from_randn_eps_breaks_a_masked_unmasked_tie() -> None:
    # eps=1 ties (mask 1, |randn|=1) with (mask 0, |randn|=2): both scores are 2.
    # Stable argsort keeps the lower index. Dropping eps, or adding it after
    # the product, puts the unmasked site first.
    mask = np.array([1.0, 0.0])
    randn = np.array([1.0, -2.0])
    got = order_from_randn(mask, randn, eps=1.0)
    np.testing.assert_array_equal(got, np.array([0, 1]))


def test_order_from_randn_orders_several_unmasked_sites_by_abs_randn() -> None:
    # Row 0 is three zeros. Ascending eps*|randn| is index 2, 0, 1 (scores
    # 0.10, 0.25, 0.45). Row 1 scores are 0, 0.20, 0.10. Without eps, or with
    # eps added after the product, the three zeros stay in index order.
    mask = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    randn = np.array([[0.5, -0.9, 0.2], [0.0, 0.4, -0.2]])
    got = order_from_randn(mask, randn, eps=0.5)
    np.testing.assert_array_equal(got, np.array([[2, 0, 1], [0, 2, 1]]))


def test_order_from_randn_breaks_ties_toward_the_lower_index() -> None:
    mask = np.ones(4)
    randn = np.array([1.0, -1.0, 0.5, -0.5])
    got = order_from_randn(mask, randn, eps=0.0)
    np.testing.assert_array_equal(got, np.array([2, 3, 0, 1]))


def test_order_from_randn_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        order_from_randn(np.ones(2), np.ones(3), 1e-6)
    with pytest.raises(ValueError):
        order_from_randn(np.ones(2), np.ones(2), -1e-6)


def test_injected_source_feeds_the_same_arrays_to_both_callables() -> None:
    mask = np.array([1.0, 0.0, 1.0, 1.0])
    randn = np.array([0.4, -0.2, 0.8, -0.1])
    source = InjectedSource(
        order=order_from_randn(mask, randn, 1e-6),
        noise=np.array([0.2, -0.3]),
        uniform=np.array([0.1, 0.9]),
    )

    def candidate(x: int, *, order: np.ndarray, noise: np.ndarray, uniform: np.ndarray):
        return x, order, noise, uniform

    def reference(x: int, *, order: np.ndarray, noise: np.ndarray, uniform: np.ndarray):
        return x, np.asarray(order), np.asarray(noise), np.asarray(uniform)

    left = source.bind(candidate)(7)
    right = source.bind(reference)(7)
    other = np.array([3, 1, 2, 0])
    overridden = source.bind(candidate)(7, order=other)
    assert overridden[0] == 7
    np.testing.assert_array_equal(overridden[1], other)
    assert left[0] == right[0] == 7
    names = ("order", "noise", "uniform")
    provided = source.parameters()
    for part, name in enumerate(names, start=1):
        np.testing.assert_array_equal(left[part], right[part])
        np.testing.assert_array_equal(left[part], provided[name])


def _logits(vector: np.ndarray):
    def score(tokens: np.ndarray, position: int, known: np.ndarray) -> np.ndarray:
        del tokens, position, known
        return vector.copy()

    return score


def test_teacher_forced_lane_matches_on_both_directions() -> None:
    vector = np.array([0.25, -0.5, 1.0, 0.0])
    sequence = np.array([0, 1, 2, 3])
    order = np.array([2, 0, 3, 1])
    result = teacher_forced_lane(sequence, order, _logits(vector), _logits(vector))
    assert result.passed
    assert result.summary().startswith("PASS")
    assert [row.direction for row in result.positions].count("forward") == 4
    assert [row.direction for row in result.positions].count("reverse") == 4


def test_teacher_forced_lane_catches_a_swapped_alphabet() -> None:
    true = np.array([0.0, 2.0, 1.0, 5.0])
    swap = np.array([0, 2, 1, 3])
    sequence = np.array([0, 1, 2, 3])
    order = np.arange(4)

    def oracle(tokens: np.ndarray, position: int, known: np.ndarray) -> np.ndarray:
        del tokens, position, known
        out = np.zeros(4)
        out[swap] = true
        return out

    aligned = teacher_forced_lane(sequence, order, _logits(true), oracle, alphabet_map=swap)
    swapped = teacher_forced_lane(sequence, order, _logits(true), oracle)
    assert aligned.passed
    assert not swapped.passed
    assert swapped.summary().startswith("FAIL")


def test_teacher_forced_lane_maps_token_ids_before_the_oracle() -> None:
    seen: list[np.ndarray] = []
    swap = np.array([0, 2, 1, 3])

    def candidate(tokens: np.ndarray, position: int, known: np.ndarray) -> np.ndarray:
        del tokens, position, known
        return np.zeros(4)

    def oracle(tokens: np.ndarray, position: int, known: np.ndarray) -> np.ndarray:
        del position, known
        seen.append(tokens.copy())
        return np.zeros(4)

    sequence = np.array([1, 0])
    teacher_forced_lane(sequence, np.array([0, 1]), candidate, oracle, alphabet_map=swap)
    np.testing.assert_array_equal(seen[0], np.array([2, 0]))
    np.testing.assert_array_equal(map_token_ids(sequence, swap), np.array([2, 0]))


def test_teacher_forced_lane_reverse_direction_catches_order_dependent_bias() -> None:
    fixed = np.array([1.0, 0.0, 0.0, 0.0])

    def candidate(tokens: np.ndarray, position: int, known: np.ndarray) -> np.ndarray:
        del tokens, position
        if known[-1] and not known[0]:
            return fixed + 10.0
        return fixed.copy()

    result = teacher_forced_lane(np.arange(4), np.arange(4), candidate, _logits(fixed))
    assert not result.passed
    assert all(row.passed for row in result.positions if row.direction == "forward")
    assert any(not row.passed for row in result.positions if row.direction == "reverse")


def test_teacher_forced_lane_applies_a_3_cycle_forward_in_both_directions() -> None:
    cycle = np.array([1, 2, 0])
    sequence = np.array([0, 1, 2])
    order = np.array([2, 0, 1])
    expected_tokens = np.array([1, 2, 0])
    candidate_logits = np.array([3.0, 0.0, 1.0])
    oracle_logits = np.array([1.0, 3.0, 0.0])
    seen: list[np.ndarray] = []

    def candidate(tokens: np.ndarray, position: int, known: np.ndarray) -> np.ndarray:
        del tokens, position, known
        return candidate_logits.copy()

    def oracle(tokens: np.ndarray, position: int, known: np.ndarray) -> np.ndarray:
        del position, known
        seen.append(np.asarray(tokens).copy())
        if np.array_equal(tokens, expected_tokens):
            return oracle_logits.copy()
        return np.array([9.0, 9.0, 9.0])

    result = teacher_forced_lane(sequence, order, candidate, oracle, alphabet_map=cycle)
    assert result.passed
    assert all(row.passed for row in result.positions if row.direction == "forward")
    assert all(row.passed for row in result.positions if row.direction == "reverse")
    assert seen
    for tokens in seen:
        np.testing.assert_array_equal(tokens, expected_tokens)


def test_teacher_forced_lane_scorer_mutation_does_not_corrupt_the_next_step() -> None:
    def candidate(tokens: np.ndarray, position: int, known: np.ndarray) -> np.ndarray:
        del tokens, position
        logits = known.astype(np.float64)
        known[:] = True
        return logits

    def oracle(tokens: np.ndarray, position: int, known: np.ndarray) -> np.ndarray:
        del tokens, position
        logits = known.astype(np.float64)
        known[:] = True
        return logits

    result = teacher_forced_lane(np.array([0, 1]), np.array([0, 1]), candidate, oracle)
    assert result.passed
    assert all(row.passed for row in result.positions)


def test_teacher_forced_lane_refuses_self_comparison() -> None:
    def score(tokens: np.ndarray, position: int, known: np.ndarray) -> np.ndarray:
        raise AssertionError("scorer should not run")

    with pytest.raises(SelfParityError):
        teacher_forced_lane(np.array([0, 1]), np.array([0, 1]), score, score)


def test_assert_distinct_callables_catches_the_same_bound_method() -> None:
    class Box:
        def score(self) -> int:
            return 1

    box = Box()
    with pytest.raises(SelfParityError):
        assert_distinct_callables(box.score, box.score)
    assert_distinct_callables(Box().score, Box().score)
    assert_distinct_callables(lambda: 0, lambda: 1)


def test_align_logits_gathers_the_class_axis() -> None:
    logits = np.array([[10.0, 20.0, 30.0]])
    got = align_logits(logits, np.array([2, 0, 1]))
    np.testing.assert_array_equal(got, np.array([[30.0, 10.0, 20.0]]))


def test_collapsed_sampler_matches_clear_argmax() -> None:
    logits = np.array([[0.1, 0.2, 5.0], [1.0, -2.0, 0.0]])
    result = collapsed_sampler_lane(logits, logits.copy(), tie_margin=0.1)
    assert result.passed
    assert result.ties == ()
    assert result.mismatches == ()
    assert result.summary().startswith("PASS")


def test_collapsed_sampler_reports_near_ties_without_failing() -> None:
    candidate = np.array([0.0, 0.01])
    oracle = np.array([0.01, 0.0])
    result = collapsed_sampler_lane(candidate, oracle, tie_margin=0.05)
    assert result.passed
    assert result.mismatches == ()
    assert len(result.ties) == 1
    assert result.ties[0].disagreed
    assert result.ties[0].index == ()
    assert "1 near-tie" in result.summary()


def test_collapsed_sampler_fails_a_decisive_mismatch() -> None:
    candidate = np.array([[0.0, 3.0], [0.0, 0.01]])
    oracle = np.array([[3.0, 0.0], [0.01, 0.0]])
    result = collapsed_sampler_lane(candidate, oracle, tie_margin=0.05)
    assert not result.passed
    assert len(result.mismatches) == 1
    assert result.mismatches[0].index == (0,)
    assert result.summary().startswith("FAIL")


def test_collapsed_sampler_exact_tie_is_not_a_failure() -> None:
    candidate = np.array([1.0, 1.0, 0.0])
    oracle = np.array([0.0, 1.0, 1.0])
    result = collapsed_sampler_lane(candidate, oracle, tie_margin=0.0)
    assert result.passed
    assert result.ties[0].disagreed


def test_collapsed_sampler_near_tie_uses_the_smaller_margin() -> None:
    # Candidate margin 0.01, oracle margin 3. min is a tie; max is a mismatch.
    tight_candidate = np.array([0.0, 0.01])
    decisive_oracle = np.array([3.0, 0.0])
    small_candidate = collapsed_sampler_lane(tight_candidate, decisive_oracle, tie_margin=0.05)
    assert small_candidate.passed
    assert small_candidate.mismatches == ()
    assert len(small_candidate.ties) == 1
    assert small_candidate.ties[0].near_tie
    assert small_candidate.ties[0].candidate_margin == pytest.approx(0.01)
    assert small_candidate.ties[0].oracle_margin == pytest.approx(3.0)

    # Oracle margin 0.01, candidate margin 3. Candidate-margin-only would miss this.
    decisive_candidate = np.array([0.0, 3.0])
    tight_oracle = np.array([0.01, 0.0])
    small_oracle = collapsed_sampler_lane(decisive_candidate, tight_oracle, tie_margin=0.05)
    assert small_oracle.passed
    assert small_oracle.mismatches == ()
    assert len(small_oracle.ties) == 1
    assert small_oracle.ties[0].candidate_margin == pytest.approx(3.0)
    assert small_oracle.ties[0].oracle_margin == pytest.approx(0.01)


def test_collapsed_sampler_applies_a_3_cycle_forward() -> None:
    cycle = np.array([1, 2, 0])
    candidate = np.array([3.0, 0.0, 1.0])
    oracle = np.array([1.0, 3.0, 0.0])
    aligned = collapsed_sampler_lane(candidate, oracle, tie_margin=0.1, alphabet_map=cycle)
    assert aligned.passed
    assert aligned.mismatches == ()
    assert not collapsed_sampler_lane(candidate, oracle, tie_margin=0.1).passed


def test_collapsed_sampler_aligns_a_swapped_alphabet() -> None:
    swap = np.array([1, 0])
    candidate = np.array([0.0, 5.0])
    oracle = np.array([5.0, 0.0])
    assert collapsed_sampler_lane(candidate, oracle, tie_margin=0.1, alphabet_map=swap).passed
    assert not collapsed_sampler_lane(candidate, oracle, tie_margin=0.1).passed


def _fair_draws(n: int = 200) -> np.ndarray:
    return np.tile(np.array([0, 1], dtype=np.int64), n // 2)


def _probs() -> np.ndarray:
    return np.array([0.5, 0.5])


def test_distributional_lane_passes_only_with_a_rejecting_control() -> None:
    fair = _fair_draws()
    probs = _probs()

    def sampler(knobs: dict[str, int]) -> np.ndarray:
        if knobs["bias"] == 0:
            return fair
        return np.zeros(fair.shape[0], dtype=np.int64)

    control = require_negative_control(sampler, probs, {"bias": 0, "temperature": 1}, "bias", 1)
    result = distributional_lane(fair, probs, negative_control=control)
    assert control.rejected
    assert control.knob == "bias"
    assert result.passed
    assert result.verdict == "PASS"
    assert result.summary().startswith("PASS")
    assert result.steps[0].tv == pytest.approx(0.0)
    assert result.steps[0].p_value == pytest.approx(1.0)
    assert result.steps[0].min_detectable_tv == pytest.approx(
        min_detectable_tv(fair.shape[0], 2, 0.05, power=0.8)
    )


def test_distributional_lane_rejects_a_biased_sampler() -> None:
    fair = _fair_draws()
    biased = np.zeros(fair.shape[0], dtype=np.int64)
    draws = np.stack([fair, biased], axis=1)
    probs = np.array([[0.5, 0.5], [0.5, 0.5]])
    result = distributional_lane(draws, probs)
    assert result.verdict == "FAIL"
    assert not result.passed
    assert not result.steps[0].rejected
    assert result.steps[1].rejected
    assert result.steps[1].tv == pytest.approx(0.5)
    assert result.summary().startswith("FAIL")


def test_distributional_lane_refuses_pass_without_a_rejecting_control() -> None:
    fair = _fair_draws()
    probs = _probs()
    missing = distributional_lane(fair, probs)
    assert missing.verdict == "UNCONTROLLED"
    assert not missing.passed
    assert not missing.summary().startswith("PASS")


def test_hand_built_negative_control_is_refused() -> None:
    with pytest.raises(TypeError):
        NegativeControl("x", True, 0.0, 1.0)
    with pytest.raises(TypeError):
        NegativeControl("x", True, 0.0, 1.0, 100, 0.05)


def test_distributional_lane_requires_control_n_and_alpha_to_match() -> None:
    fair = _fair_draws()
    probs = _probs()

    def sampler(knobs: dict[str, int]) -> np.ndarray:
        if knobs["bias"] == 0:
            return fair
        return np.zeros(fair.shape[0], dtype=np.int64)

    control = require_negative_control(sampler, probs, {"bias": 0}, "bias", 1)
    assert control.n == fair.shape[0]
    assert control.alpha == pytest.approx(0.05)
    with pytest.raises(ValueError, match="n="):
        distributional_lane(fair[: fair.shape[0] // 2], probs, negative_control=control)
    with pytest.raises(ValueError, match="alpha"):
        distributional_lane(fair, probs, alpha=0.01, negative_control=control)


def test_chi_square_matches_counts_60_40() -> None:
    draws = np.array([0] * 60 + [1] * 40, dtype=np.int64)
    step = distributional_lane(draws, np.array([0.5, 0.5])).steps[0]
    assert step.chi2 == 4.0
    assert step.df == 1
    assert step.p_value == pytest.approx(0.0455, abs=1e-3)


def test_min_detectable_tv_ignores_zero_probability_classes() -> None:
    draws = _fair_draws()
    step = distributional_lane(draws, np.array([0.5, 0.5, 0.0])).steps[0]
    two = min_detectable_tv(draws.shape[0], 2, 0.05, power=0.8)
    three = min_detectable_tv(draws.shape[0], 3, 0.05, power=0.8)
    assert step.df == 1
    assert step.min_detectable_tv == pytest.approx(two)
    assert two != pytest.approx(three)


def test_distributional_power_at_the_reported_minimum_detectable_tv() -> None:
    # Nominal power is 0.8. The normal approximation lands near 0.75, so the
    # floor is widened to 0.60 (a halved chi-square statistic rejects near 0.45).
    n = 2000
    alpha = 0.05
    oracle = np.array([0.5, 0.5])
    null = np.random.default_rng(0).choice(2, size=n, p=oracle)
    null_step = distributional_lane(null, oracle, alpha=alpha).steps[0]
    assert not null_step.rejected
    shift = null_step.min_detectable_tv
    assert 0.0 < shift < 0.5
    perturbed = np.array([0.5 + shift, 0.5 - shift])
    reps = 60
    rejects = 0
    for index in range(reps):
        draws = np.random.default_rng(1_000 + index).choice(2, size=n, p=perturbed)
        step = distributional_lane(draws, oracle, alpha=alpha).steps[0]
        rejects += int(step.rejected)
    assert rejects / reps >= 0.60


def test_holm_keeps_twenty_seeded_null_steps() -> None:
    # Seed 1, n=800, T=20 has a raw p below 0.05 (an uncorrected lane FAILs)
    # and above alpha/T, so Holm keeps every null.
    n = 800
    steps = 20
    draws = np.random.default_rng(1).choice(2, size=(n, steps), p=np.array([0.5, 0.5]))
    probs = np.tile(np.array([0.5, 0.5]), (steps, 1))
    result = distributional_lane(draws, probs, alpha=0.05)
    assert min(step.p_value for step in result.steps) < 0.05
    assert all(not step.rejected for step in result.steps)
    assert result.verdict != "FAIL"


def test_require_negative_control_rejects_when_any_step_rejects() -> None:
    fair = _fair_draws()
    biased = np.zeros(fair.shape[0], dtype=np.int64)
    mixed = np.stack([fair, biased], axis=1)
    probs = np.array([[0.5, 0.5], [0.5, 0.5]])
    scored = distributional_lane(mixed, probs)
    assert [step.rejected for step in scored.steps] == [False, True]

    def sampler(knobs: dict[str, int]) -> np.ndarray:
        del knobs
        return mixed

    control = require_negative_control(sampler, probs, {"bias": 0}, "bias", 1)
    assert control.rejected
    assert control.n == fair.shape[0]
    assert control.alpha == pytest.approx(0.05)


def test_require_negative_control_asserts_the_perturbed_knob_rejects() -> None:
    fair = _fair_draws()

    def sampler(knobs: dict[str, int]) -> np.ndarray:
        del knobs
        return fair

    with pytest.raises(NegativeControlError, match="bias"):
        require_negative_control(sampler, _probs(), {"bias": 0}, "bias", 1)
    with pytest.raises(KeyError):
        require_negative_control(sampler, _probs(), {"bias": 0}, "temperature", 0)


def test_min_detectable_tv_follows_the_power_formula() -> None:
    # Chi-square 95th percentile at df=3 (NIST / published value).
    critical = 7.814727903251179
    df = 3
    n = 1000
    # power 0.5 => z = 0 => λ = critical - df, TV = 0.5 * sqrt(λ / n).
    expected = 0.5 * ((critical - df) / n) ** 0.5
    got = min_detectable_tv(n, n_classes=df + 1, alpha=0.05, power=0.5)
    assert got == pytest.approx(expected, rel=1e-3)

    z = float(norm_dist.ppf(0.8))
    noncentrality = 0.0
    for _ in range(50):
        scale = (2.0 * (df + 2.0 * noncentrality)) ** 0.5
        noncentrality = critical - df + z * scale
    expected_power = 0.5 * (noncentrality / n) ** 0.5
    assert min_detectable_tv(n, n_classes=4, alpha=0.05, power=0.8) == pytest.approx(
        expected_power, rel=1e-3
    )
    wide = min_detectable_tv(10_000, 4, 0.05, power=0.8)
    narrow = min_detectable_tv(100, 4, 0.05, power=0.8)
    assert wide < narrow
    assert min_detectable_tv(n, 4, 0.05, power=0.9) > min_detectable_tv(n, 4, 0.05, power=0.8)
    assert min_detectable_tv(n, 4, 0.01, power=0.8) > min_detectable_tv(n, 4, 0.05, power=0.8)


def test_knob_coverage_lists_unvaried_knobs() -> None:
    report = knob_coverage(["temperature", "top_k", "bias"], ["bias", "dropout"])
    assert report.unvaried == ("temperature", "top_k")
    assert report.varied == ("bias",)
    assert report.undeclared == ("dropout",)
    assert not report.complete
    covered = knob_coverage(["temperature", "bias"], ["bias", "temperature"])
    assert covered.complete
    assert covered.unvaried == ()


def test_testing_package_does_not_import_torch_or_scipy() -> None:
    root = _ROOT / "src" / "xtrax" / "testing"
    modules: list[str] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
    for name in modules:
        assert name.split(".")[0] not in {"torch", "scipy"}
