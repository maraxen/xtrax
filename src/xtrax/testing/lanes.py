"""Teacher-forced, collapsed, and distributional parity lanes.

Each lane compares a candidate to an oracle. The distributional lane reports
PASS only after a negative control has rejected a perturbed knob.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import chi2 as chi2_dist
from jax.scipy.stats import norm as norm_dist
from numpy.typing import ArrayLike

from xtrax.testing.guard import assert_distinct_callables

__all__ = [
    "CollapsedSamplerResult",
    "DistributionalResult",
    "KnobCoverage",
    "NegativeControl",
    "NegativeControlError",
    "Scorer",
    "StepDistribution",
    "TeacherForcedResult",
    "TieRecord",
    "align_logits",
    "collapsed_sampler_lane",
    "distributional_lane",
    "knob_coverage",
    "map_token_ids",
    "min_detectable_tv",
    "require_negative_control",
    "teacher_forced_lane",
]

# (tokens, position, known) -> conditional logits of shape (V,).
# ``known[i]`` is True where ``tokens[i]`` is already decoded. ``known[position]``
# is False. ``tokens`` are in that scorer's alphabet.
Scorer = Callable[[np.ndarray, int, np.ndarray], np.ndarray]

Verdict = Literal["PASS", "FAIL", "UNCONTROLLED"]


def align_logits(logits: ArrayLike, index_map: ArrayLike) -> np.ndarray:
    """Gather the class axis so result ``[..., i]`` is source class ``index_map[i]``.

    ``index_map[i]`` is the oracle class corresponding to candidate class ``i``.
    """
    values = np.asarray(logits)
    mapping = _as_index_map(index_map)
    if np.any(mapping < 0) or int(mapping.max()) >= values.shape[-1]:
        raise ValueError(
            f"index_map entries must address the class axis of shape {values.shape[-1]}"
        )
    return np.take(values, mapping, axis=-1)


def map_token_ids(tokens: ArrayLike, index_map: ArrayLike) -> np.ndarray:
    """Map token ids through ``index_map`` (candidate id -> oracle id)."""
    ids = np.asarray(tokens)
    mapping = _as_index_map(index_map)
    if ids.size and (np.any(ids < 0) or np.any(ids >= mapping.shape[0])):
        raise ValueError("token ids fall outside index_map")
    return mapping[ids]


def _as_index_map(index_map: ArrayLike) -> np.ndarray:
    mapping = np.asarray(index_map)
    if mapping.ndim != 1 or mapping.size == 0:
        raise ValueError("index_map must be a non-empty 1-D permutation")
    if not np.issubdtype(mapping.dtype, np.integer):
        raise ValueError("index_map must be an integer array")
    return mapping.astype(np.int64, copy=False)


def _as_permutation(index_map: ArrayLike, n_classes: int) -> np.ndarray:
    mapping = _as_index_map(index_map)
    if mapping.shape != (n_classes,):
        raise ValueError(f"alphabet_map length {mapping.shape[0]} != vocab {n_classes}")
    if np.any(mapping < 0) or np.any(mapping >= n_classes) or np.unique(mapping).size != n_classes:
        raise ValueError("alphabet_map must be a permutation of the class axis")
    return mapping


@dataclass(frozen=True)
class PositionComparison:
    """One conditional-logit comparison."""

    direction: Literal["forward", "reverse"]
    step: int
    position: int
    max_abs: float
    passed: bool


@dataclass(frozen=True)
class TeacherForcedResult:
    """Per-position conditional logits in the forward order and the reversed order.

    Attributes:
        passed: True when every position matches in both directions.
        positions: One record per step, forward steps then reverse steps.
        max_abs: Largest absolute logit difference across both directions.
        atol: Absolute tolerance used.
        rtol: Relative tolerance used.
    """

    passed: bool
    positions: tuple[PositionComparison, ...]
    max_abs: float
    atol: float
    rtol: float

    def summary(self) -> str:
        """One-line verdict with the largest logit gap."""
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"{verdict}: max|logit diff| = {self.max_abs:.3e} "
            f"(atol={self.atol:g}, rtol={self.rtol:g})"
        )


def teacher_forced_lane(
    sequence: ArrayLike,
    order: ArrayLike,
    candidate: Scorer,
    oracle: Scorer,
    *,
    alphabet_map: ArrayLike | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> TeacherForcedResult:
    """Compare conditional logits on one sequence, then on the reversed order.

    ``candidate`` and ``oracle`` are scorers ``(tokens, position, known) -> (V,)``.
    ``known`` is true on positions already decoded. The forward pass walks
    ``order``; the reverse pass walks ``order[::-1]``. Both directions have to
    match.

    ``alphabet_map[i]`` is the oracle class for candidate class ``i``. Token ids
    are mapped before the oracle call, and oracle logits are gathered back onto
    the candidate class axis with the same map. A swapped alphabet that is left
    unmapped shows up as a logit mismatch.

    Args:
        sequence: Candidate-alphabet token ids, shape ``(L,)``.
        order: Permutation of ``range(L)``, the decoding order.
        candidate: Candidate scorer.
        oracle: Oracle scorer. Must be a different callable from ``candidate``.
        alphabet_map: Permutation from candidate classes to oracle classes.
        atol: Absolute tolerance on logits.
        rtol: Relative tolerance on logits.

    Returns:
        Per-position diffs for both directions.
    """
    assert_distinct_callables(candidate, oracle)
    tokens = _as_token_sequence(sequence)
    positions = _as_order(order, tokens.shape[0])
    mapping: np.ndarray | None = None
    if alphabet_map is not None:
        mapping = _as_permutation(alphabet_map, int(np.asarray(alphabet_map).shape[0]))
        if np.any(tokens < 0) or np.any(tokens >= mapping.shape[0]):
            raise ValueError("sequence token ids fall outside alphabet_map")
    forward = _walk(tokens, positions, candidate, oracle, mapping, atol, rtol, "forward")
    reverse = _walk(tokens, positions[::-1], candidate, oracle, mapping, atol, rtol, "reverse")
    rows = tuple(forward + reverse)
    max_abs = max(row.max_abs for row in rows)
    return TeacherForcedResult(
        passed=all(row.passed for row in rows),
        positions=rows,
        max_abs=max_abs,
        atol=float(atol),
        rtol=float(rtol),
    )


def _as_token_sequence(sequence: ArrayLike) -> np.ndarray:
    tokens = np.asarray(sequence)
    if tokens.ndim != 1 or tokens.size == 0:
        raise ValueError("sequence must be a non-empty 1-D token-id array")
    if not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError("sequence must be integer token ids")
    return tokens.astype(np.int64, copy=False)


def _as_order(order: ArrayLike, length: int) -> np.ndarray:
    positions = np.asarray(order)
    if positions.shape != (length,) or not np.issubdtype(positions.dtype, np.integer):
        raise ValueError(f"order must be an integer permutation of shape ({length},)")
    positions = positions.astype(np.int64, copy=False)
    if np.sort(positions).tolist() != list(range(length)):
        raise ValueError("order must be a permutation of positions")
    return positions


def _walk(
    tokens: np.ndarray,
    positions: np.ndarray,
    candidate: Scorer,
    oracle: Scorer,
    mapping: np.ndarray | None,
    atol: float,
    rtol: float,
    direction: Literal["forward", "reverse"],
) -> list[PositionComparison]:
    known = np.zeros(tokens.shape[0], dtype=bool)
    oracle_tokens = tokens if mapping is None else map_token_ids(tokens, mapping)
    rows: list[PositionComparison] = []
    for step, position in enumerate(positions.tolist()):
        candidate_logits = _logits_at(candidate, tokens, position, known, mapping)
        oracle_logits = _logits_at(oracle, oracle_tokens, position, known, mapping)
        if mapping is not None:
            oracle_logits = align_logits(oracle_logits, mapping)
        if candidate_logits.shape != oracle_logits.shape:
            raise ValueError(
                f"logit shape {candidate_logits.shape} != oracle shape {oracle_logits.shape}"
            )
        diff = float(np.max(np.abs(candidate_logits - oracle_logits)))
        rows.append(
            PositionComparison(
                direction=direction,
                step=step,
                position=position,
                max_abs=diff,
                passed=bool(np.allclose(candidate_logits, oracle_logits, atol=atol, rtol=rtol)),
            )
        )
        known[position] = True
    return rows


def _logits_at(
    scorer: Scorer,
    tokens: np.ndarray,
    position: int,
    known: np.ndarray,
    mapping: np.ndarray | None,
) -> np.ndarray:
    logits = np.asarray(
        scorer(tokens.copy(), position, known.copy()),
        dtype=np.float64,
    )
    if logits.ndim != 1 or logits.size == 0:
        raise ValueError(f"scorer must return logits of shape (V,), got {logits.shape}")
    if mapping is not None and logits.shape != mapping.shape:
        raise ValueError(
            f"logits vocab {logits.shape[0]} != alphabet_map length {mapping.shape[0]}"
        )
    if not np.isfinite(logits).all():
        raise ValueError("logits must be finite")
    return logits


@dataclass(frozen=True)
class TieRecord:
    """Argmax comparison at one site, including the top-two logit margin."""

    index: tuple[int, ...]
    candidate_argmax: int
    oracle_argmax: int
    candidate_margin: float
    oracle_margin: float
    disagreed: bool
    near_tie: bool


@dataclass(frozen=True)
class CollapsedSamplerResult:
    """Temperature-0 argmax comparison.

    A site whose smaller top-two margin is within ``tie_margin`` is a near-tie:
    it is listed on ``ties`` and does not fail the lane, including when the
    argmax disagrees. A disagreement with both margins above ``tie_margin`` is a
    mismatch.

    Attributes:
        passed: True when ``mismatches`` is empty.
        tie_margin: Margin at or below which a site is a near-tie.
        ties: Near-tie sites, agreeing or disagreeing.
        mismatches: Decisive argmax disagreements.
    """

    passed: bool
    tie_margin: float
    ties: tuple[TieRecord, ...]
    mismatches: tuple[TieRecord, ...]

    def summary(self) -> str:
        """One-line verdict with mismatch and near-tie counts."""
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"{verdict}: {len(self.mismatches)} decisive mismatch(es), "
            f"{len(self.ties)} near-tie(s) within margin {self.tie_margin:g}"
        )


def collapsed_sampler_lane(
    candidate_logits: ArrayLike,
    oracle_logits: ArrayLike,
    *,
    tie_margin: float,
    alphabet_map: ArrayLike | None = None,
) -> CollapsedSamplerResult:
    """Compare temperature-0 argmax, and report near-ties instead of failing them.

    Argmax ties break toward the lower index (``numpy.argmax``). A site is a
    near-tie when ``min(candidate_margin, oracle_margin) <= tie_margin``, where
    each margin is the gap between the largest and second-largest logit.

    Args:
        candidate_logits: Candidate logits, shape ``(..., V)`` with ``V >= 2``.
        oracle_logits: Oracle logits in oracle class order, same shape after
            alignment.
        tie_margin: Non-negative gap that counts as a near-tie.
        alphabet_map: Optional permutation applied to the oracle class axis.

    Returns:
        Mismatches and near-tie records.
    """
    margin = float(tie_margin)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError(f"tie_margin must be finite and >= 0, got {tie_margin!r}")
    candidate = _finite_logits(candidate_logits, "candidate_logits")
    oracle = _finite_logits(oracle_logits, "oracle_logits")
    if alphabet_map is not None:
        mapping = _as_permutation(alphabet_map, candidate.shape[-1])
        oracle = align_logits(oracle, mapping)
    if candidate.shape != oracle.shape:
        raise ValueError(f"logit shape {candidate.shape} != oracle shape {oracle.shape}")
    if candidate.shape[-1] < 2:
        raise ValueError("collapsed comparison needs at least two classes")
    flat_candidate = candidate.reshape(-1, candidate.shape[-1])
    flat_oracle = oracle.reshape(-1, oracle.shape[-1])
    batch = candidate.shape[:-1]
    ties: list[TieRecord] = []
    mismatches: list[TieRecord] = []
    for flat_index in range(flat_candidate.shape[0]):
        index = ()
        if batch:
            index = tuple(int(part) for part in np.unravel_index(flat_index, batch))
        record = _tie_record(flat_candidate[flat_index], flat_oracle[flat_index], index, margin)
        if record.near_tie:
            ties.append(record)
        elif record.disagreed:
            mismatches.append(record)
    return CollapsedSamplerResult(
        passed=not mismatches,
        tie_margin=margin,
        ties=tuple(ties),
        mismatches=tuple(mismatches),
    )


def _finite_logits(logits: ArrayLike, name: str) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] == 0:
        raise ValueError(f"{name} must have a non-empty class axis")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    return values


def _tie_record(
    candidate: np.ndarray,
    oracle: np.ndarray,
    index: tuple[int, ...],
    tie_margin: float,
) -> TieRecord:
    candidate_argmax, candidate_margin = _argmax_margin(candidate)
    oracle_argmax, oracle_margin = _argmax_margin(oracle)
    return TieRecord(
        index=index,
        candidate_argmax=candidate_argmax,
        oracle_argmax=oracle_argmax,
        candidate_margin=candidate_margin,
        oracle_margin=oracle_margin,
        disagreed=candidate_argmax != oracle_argmax,
        near_tie=min(candidate_margin, oracle_margin) <= tie_margin,
    )


def _argmax_margin(logits: np.ndarray) -> tuple[int, float]:
    top = int(np.argmax(logits))
    masked = logits.copy()
    masked[top] = -np.inf
    second = int(np.argmax(masked))
    return top, float(logits[top] - logits[second])


@dataclass(frozen=True)
class StepDistribution:
    """Chi-square and total variation for one conditional step."""

    step: int
    chi2: float
    p_value: float
    tv: float
    min_detectable_tv: float
    rejected: bool
    df: int
    min_expected: float


@dataclass(frozen=True)
class NegativeControl:
    """A perturbed-knob run of the distributional lane.

    Attributes:
        knob: Name of the knob that was perturbed.
        rejected: True when the chi-square test rejected the oracle.
        p_value: Smallest step-wise p-value on the perturbed run.
        tv: Largest step-wise total variation on the perturbed run.
    """

    knob: str
    rejected: bool
    p_value: float
    tv: float


class NegativeControlError(AssertionError):
    """Perturbing the named knob left the lane matching the oracle."""


@dataclass(frozen=True)
class DistributionalResult:
    """Per-step conditional distribution versus oracle probabilities.

    ``verdict`` is ``PASS`` only when every step keeps the null (p >= alpha) and
    ``negative_control`` rejected at this alpha. A statistical match with no
    rejecting control is ``UNCONTROLLED``. A rejected null is ``FAIL``.

    Attributes:
        steps: One record per conditional step.
        alpha: Tail threshold.
        power: Power used for :func:`min_detectable_tv`.
        n: Draws per step.
        negative_control: Control attached by the caller, if any.
        verdict: ``PASS``, ``FAIL``, or ``UNCONTROLLED``.
    """

    steps: tuple[StepDistribution, ...]
    alpha: float
    power: float
    n: int
    negative_control: NegativeControl | None
    verdict: Verdict

    @property
    def passed(self) -> bool:
        """True only for an actual ``PASS`` verdict."""
        return self.verdict == "PASS"

    def summary(self) -> str:
        """One-line verdict. The word PASS appears only for a real pass."""
        min_p = min(step.p_value for step in self.steps)
        max_tv = max(step.tv for step in self.steps)
        if self.negative_control is None:
            control = "negative control not run"
        elif self.negative_control.rejected and self.negative_control.p_value < self.alpha:
            control = f"negative control {self.negative_control.knob!r} rejected"
        else:
            control = f"negative control {self.negative_control.knob!r} did not reject"
        return (
            f"{self.verdict}: min p={min_p:.3g} max TV={max_tv:.3g} alpha={self.alpha:g}; {control}"
        )


def min_detectable_tv(
    n: int,
    n_classes: int,
    alpha: float = 0.05,
    *,
    power: float = 0.8,
) -> float:
    """Smallest total variation detectable at ``power`` for this ``n`` and ``alpha``.

    The chi-square critical value is the central quantile at ``1 - alpha``
    (inverted from ``jax.scipy.stats.chi2.cdf``; that module has no ``ppf``).
    Non-centrality ``λ`` is the value whose normal approximation to the
    non-central chi-square — mean ``df + λ``, variance ``2(df + 2λ)`` — puts
    mass ``power`` above the critical value. ``df = n_classes - 1``.

    Among alternatives with a given total variation, the smallest chi-square
    divergence is ``4 TV²`` (Cauchy-Schwarz). The TV reported here is the one
    that still reaches ``λ`` under that least-favorable alternative:
    ``TV = 0.5 * sqrt(λ / n)``. A more concentrated error is detectable at a
    smaller TV. Default ``power`` is 0.8.

    Args:
        n: Number of draws.
        n_classes: Classes with positive oracle probability. At least 2.
        alpha: Test level in ``(0, 1)``.
        power: Target power in ``(0, 1)``.

    Returns:
        Minimum detectable total variation.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2, got {n_classes}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")
    df = n_classes - 1
    critical = _chi2_critical(alpha, df)
    z_power = float(norm_dist.ppf(power))
    noncentrality = 0.0
    for _ in range(50):
        scale = (2.0 * (df + 2.0 * noncentrality)) ** 0.5
        updated = critical - df + z_power * scale
        if updated < 0.0:
            updated = 0.0
        if abs(updated - noncentrality) <= 1e-8 * max(1.0, updated):
            noncentrality = updated
            break
        noncentrality = updated
    return 0.5 * (noncentrality / n) ** 0.5


def _chi2_critical(alpha: float, df: int) -> float:
    """Central chi-square quantile at probability ``1 - alpha``."""
    target = 1.0 - alpha
    z = float(norm_dist.ppf(target))
    df_f = float(df)
    scale = (2.0 / (9.0 * df_f)) ** 0.5
    cube = 1.0 - 2.0 / (9.0 * df_f) + z * scale
    guess = df_f * cube**3 if cube > 0.0 else df_f
    lo = 0.0
    hi = max(guess * 2.0, 1.0)
    expands = 0
    while float(chi2_dist.cdf(hi, df_f)) < target and expands < 40:
        hi *= 2.0
        expands += 1
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(chi2_dist.cdf(mid, df_f)) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def distributional_lane(
    draws: ArrayLike,
    oracle_probs: ArrayLike,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    negative_control: NegativeControl | None = None,
) -> DistributionalResult:
    """Compare per-step empirical conditionals to oracle probabilities.

    ``draws`` is ``(N,)`` token ids for one step, or ``(N, T)`` for ``T`` steps.
    ``oracle_probs`` is ``(K,)`` or ``(T, K)`` and each row sums to 1. The
    chi-square goodness-of-fit p-value comes from ``jax.scipy.stats.chi2.sf``.
    A step rejects when ``p < alpha``. Total variation is
    ``0.5 * sum(|empirical - oracle|)``.

    PASS requires every step to keep the null and a negative control whose
    ``rejected`` flag is set and whose p-value is below ``alpha``. Otherwise the
    verdict is FAIL (null rejected) or UNCONTROLLED (null kept, control missing
    or not rejecting).

    Args:
        draws: Integer token ids.
        oracle_probs: Oracle conditional probabilities.
        alpha: Tail threshold in ``(0, 1)``.
        power: Power passed to :func:`min_detectable_tv`.
        negative_control: Result of :func:`require_negative_control`, when run.

    Returns:
        Per-step statistics and a verdict that withholds PASS without a
        rejecting control.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    samples, probs = _draws_and_probs(draws, oracle_probs)
    n = int(samples.shape[0])
    steps = tuple(
        _score_step(samples[:, step], probs[step], step, alpha, power, n)
        for step in range(samples.shape[1])
    )
    stats_ok = all(not step.rejected for step in steps)
    control_ok = (
        negative_control is not None
        and negative_control.rejected
        and negative_control.p_value < alpha
    )
    if not stats_ok:
        verdict: Verdict = "FAIL"
    elif not control_ok:
        verdict = "UNCONTROLLED"
    else:
        verdict = "PASS"
    return DistributionalResult(
        steps=steps,
        alpha=float(alpha),
        power=float(power),
        n=n,
        negative_control=negative_control,
        verdict=verdict,
    )


def require_negative_control(
    sampler: Callable[[Mapping[str, Any]], np.ndarray],
    oracle_probs: ArrayLike,
    knobs: Mapping[str, Any],
    knob: str,
    perturbed_value: Any,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> NegativeControl:
    """Re-run ``sampler`` with ``knob`` perturbed and require the lane to reject.

    ``sampler`` receives a knob mapping and returns draws shaped like
    :func:`distributional_lane`. The returned control is what that lane needs
    before it can report PASS.

    Args:
        sampler: ``knobs -> draws``.
        oracle_probs: Oracle probabilities passed through to the lane.
        knobs: Baseline knob mapping. ``knob`` must already be a key.
        knob: Knob name to replace.
        perturbed_value: Replacement value.
        alpha: Tail threshold.
        power: Power passed through to the lane.

    Returns:
        A control with ``rejected=True``.

    Raises:
        KeyError: ``knob`` is not in ``knobs``.
        NegativeControlError: The perturbed sampler still matches the oracle.
    """
    if knob not in knobs:
        raise KeyError(knob)
    perturbed = dict(knobs)
    perturbed[knob] = perturbed_value
    scored = distributional_lane(sampler(perturbed), oracle_probs, alpha=alpha, power=power)
    min_p = min(step.p_value for step in scored.steps)
    max_tv = max(step.tv for step in scored.steps)
    rejected = any(step.rejected for step in scored.steps)
    if not rejected:
        raise NegativeControlError(
            f"perturbing knob {knob!r} to {perturbed_value!r} did not reject the oracle "
            f"(min p={min_p:.3g}, alpha={alpha})"
        )
    return NegativeControl(knob=knob, rejected=True, p_value=min_p, tv=max_tv)


def _draws_and_probs(draws: ArrayLike, oracle_probs: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(draws)
    if not np.issubdtype(samples.dtype, np.integer):
        raise ValueError("draws must be integer token ids")
    probs = np.asarray(oracle_probs, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
        if probs.ndim != 1:
            raise ValueError("oracle_probs for 1-D draws must have shape (K,)")
        probs = probs.reshape(1, -1)
    elif samples.ndim == 2:
        if probs.ndim != 2 or probs.shape[0] != samples.shape[1]:
            raise ValueError("oracle_probs for draws (N, T) must have shape (T, K)")
    else:
        raise ValueError("draws must have shape (N,) or (N, T)")
    if samples.shape[0] == 0:
        raise ValueError("draws must contain at least one sample")
    return samples.astype(np.int64, copy=False), probs


def _score_step(
    draws: np.ndarray,
    probs: np.ndarray,
    step: int,
    alpha: float,
    power: float,
    n: int,
) -> StepDistribution:
    if probs.ndim != 1:
        raise ValueError("each oracle distribution must have shape (K,)")
    if np.any(probs < 0.0) or not np.isfinite(probs).all() or abs(float(probs.sum()) - 1.0) > 1e-5:
        raise ValueError(f"oracle_probs[{step}] must be a finite probability vector")
    if np.any(draws < 0) or np.any(draws >= probs.shape[0]):
        raise ValueError(f"draws at step {step} fall outside the oracle alphabet")
    counts = np.bincount(draws, minlength=probs.shape[0]).astype(np.float64)
    positive = probs > 0.0
    n_positive = int(positive.sum())
    unexpected = bool(np.any(counts[~positive] > 0)) if np.any(~positive) else False
    if unexpected:
        stat = float("inf")
        p_value = 0.0
        df = max(n_positive - 1, 0)
        min_expected = 0.0
    else:
        expected = n * probs[positive]
        stat = float(np.sum((counts[positive] - expected) ** 2 / expected))
        df = n_positive - 1
        p_value = _chi2_sf(stat, df)
        min_expected = float(expected.min()) if expected.size else float("nan")
    empirical = counts / n
    tv = 0.5 * float(np.sum(np.abs(empirical - probs)))
    detectable = (
        min_detectable_tv(n, n_positive, alpha, power=power) if n_positive >= 2 else float("nan")
    )
    return StepDistribution(
        step=step,
        chi2=stat,
        p_value=p_value,
        tv=tv,
        min_detectable_tv=detectable,
        rejected=p_value < alpha,
        df=df,
        min_expected=min_expected,
    )


def _chi2_sf(stat: float, df: int) -> float:
    if df <= 0:
        return 0.0 if stat > 0.0 else 1.0
    if not np.isfinite(stat):
        return 0.0
    return float(chi2_dist.sf(jnp.asarray(stat), jnp.asarray(df)))


@dataclass(frozen=True)
class KnobCoverage:
    """Declared knob surface against the knobs actually varied.

    Attributes:
        declared: Knob names, first-seen order.
        varied: Names that were varied and were also declared.
        unvaried: Declared names absent from the varied set.
        undeclared: Varied names absent from the declared surface.
    """

    declared: tuple[str, ...]
    varied: tuple[str, ...]
    unvaried: tuple[str, ...]
    undeclared: tuple[str, ...]

    @property
    def complete(self) -> bool:
        """True when every declared knob was varied and every varied knob was declared."""
        return not self.unvaried and not self.undeclared


def knob_coverage(declared: Sequence[str], varied: Sequence[str]) -> KnobCoverage:
    """Report which declared knobs were varied against the oracle.

    Unvaried knobs stay in the evidence. Names are strings; duplicates collapse
    to the first occurrence.

    Args:
        declared: Knob surface the comparison claims to cover.
        varied: Knobs actually changed across oracle runs.

    Returns:
        Declared, varied, unvaried, and undeclared names.
    """
    declared_names = _knob_names(declared, "declared")
    varied_names = _knob_names(varied, "varied")
    declared_set = set(declared_names)
    varied_set = set(varied_names)
    return KnobCoverage(
        declared=declared_names,
        varied=tuple(name for name in varied_names if name in declared_set),
        unvaried=tuple(name for name in declared_names if name not in varied_set),
        undeclared=tuple(name for name in varied_names if name not in declared_set),
    )


def _knob_names(names: Sequence[str], label: str) -> tuple[str, ...]:
    if not all(isinstance(name, str) for name in names):
        raise TypeError(f"{label} knob names must be strings")
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return tuple(ordered)
