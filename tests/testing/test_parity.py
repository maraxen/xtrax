"""Synthetic ground truth for the reference-parity harness."""

from __future__ import annotations

import ast
from pathlib import Path

import jax.numpy as jnp
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


def test_order_from_randn_matches_numpy_stable_argsort() -> None:
    mask = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    randn = np.array([[0.2, -0.4, 0.1], [0.3, 0.5, -0.7]])
    eps = 1e-6
    scores = (mask + eps) * np.abs(randn)
    got = order_from_randn(mask, randn, eps)
    np.testing.assert_array_equal(got, np.argsort(scores, axis=-1, kind="stable"))
    np.testing.assert_array_equal(got, np.asarray(jnp.argsort(jnp.asarray(scores), axis=-1)))


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

    silent = NegativeControl(knob="bias", rejected=False, p_value=0.0, tv=1.0)
    assert distributional_lane(fair, probs, negative_control=silent).verdict == "UNCONTROLLED"
    lying = NegativeControl(knob="bias", rejected=True, p_value=0.9, tv=0.0)
    assert distributional_lane(fair, probs, negative_control=lying).verdict == "UNCONTROLLED"


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
