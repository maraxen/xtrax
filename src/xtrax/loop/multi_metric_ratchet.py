"""Multi-metric-regression ratchet (T2-17, #2181, AC-10, F7, PM-2).

AC-10's claim: a candidate's fitness dict is labeled an "improvement" over the current best only
if it wins by three simultaneous statistical criteria (Win Rate, Breakdown Point, Cohen's d) --
blocking silent semantic failures like "faster but more memory," where a single-metric check would
wrongly call a regression a win.

Grounding note (verified before writing any code, not guessed): WR/BP/Cohen's d are undefined in
this epic's three main design docs (design spec, research synthesis, DAG doc -- exhaustively
grepped, no formula, no citation). The real source is an earlier NLM-prequery research stage,
`.praxia/research/260702_nlm_prequery_roadmap.md:136-142`, which spells out the full CAMPAIGN-level
statistical battery this per-iteration gate is a simplified derivative of: "Effect size: Cohen's d
>= 0.2 minimum. Consistency: Win Rate >= 0.6 across tasks, or P(A>B) >= 0.75. Stability: Breakdown
Point >= 0.2 (>=20% of datasets must be removed to flip ranking)." **BP is Breakdown Point** -- a
standard robust-statistics concept (the minimum fraction of data points that must be adversarially
removed to flip a conclusion) -- not "Beneficial Proportion" (an earlier, wrong guess, corrected
before implementation). AutoSOTA's full paper text (already read for T2-03) uses none of these
terms; this statistical battery is this project's own synthesis, not drawn from an external paper.

Per `260702_roadmap-research-synthesis.md`, AC-10/F7 is explicitly "per-iteration, mechanical
(in-loop)" -- distinct from AC-15's campaign-level battery (>=3 seeds, real Wilcoxon/Friedman).
This module adapts the campaign-level thresholds to a SINGLE candidate-fitness-dict-vs-best-
fitness-dict comparison, aggregated across the dict's metric keys, not across seeds/tasks:

- Win Rate: fraction of metrics where the candidate wins (direction-corrected).
- Breakdown Point: the smallest fraction of *winning* metrics that would need to be adversarially
  removed to drop the win rate below its own threshold -- i.e. how robust the "candidate wins"
  verdict is to a small number of metrics being excluded or noisy.
- Cohen's d: mean/stdev of the per-metric signed RELATIVE deltas (direction-corrected, normalized
  by `abs(best_value)` so differently-scaled metrics -- e.g. accuracy in [0,1] vs. memory in MB --
  contribute comparably), treating the metrics themselves as the sample. This normalization step is
  this module's own extrapolation (not spelled out in the source doc for a single-comparison
  context) -- confirmed as a design decision, not silently assumed. Non-obvious consequence of
  mean/stdev being SCALE-INVARIANT: this is a measure of how CONSISTENT the per-metric deltas are,
  not their absolute magnitude -- many uniform trivial wins give a large (even infinite) d, while
  a large-magnitude but inconsistent set of deltas (e.g. nine metrics win by 1% each, a tenth
  regresses by 50%) can give a small or negative d even with a high win rate. This is deliberate:
  it's precisely what lets Cohen's d catch a catastrophic single-metric regression that
  win_rate/breakdown_point (which only count whether each metric wins, never by how much) would
  treat identically to a trivial one -- see the test suite's
  `TestCohensDCatchesInconsistentMagnitude` for a worked example, not a "detects trivial-magnitude
  wins" check as the name might suggest.

Design point: unlike every other `assert_*`-style T2-1x gate, this module is a **pure decision
function, not a raising gate** for the improvement/non-improvement outcome itself -- a candidate
not improving on the current best is the normal, common case in an evolutionary loop, not an
anomaly. Only structural input errors (mismatched metric key sets, or fewer than 2 metrics, which
makes Cohen's d degenerate) raise.

Extension seam (mirrors every other T2-0X module's stance): this module computes and returns a
`RatchetDecision`. It does NOT call into `xtrax.loop.ratchet_crash_atomicity` (T2-10) to actually
commit or reset -- unlike T2-14 (which composes T2-09's watchdog because it spawns a genuinely
separate, untrusted subprocess), T2-10's git actions are fast, synchronous, trusted local
operations with nothing to police here; the actual commit-or-reset sequencing stays a future loop
controller's job, the only thing that can safely order "decide -> commit -> advance the ref."
"""

import math
import statistics
from collections.abc import Mapping


class RatchetInputError(Exception):
    """Structural validation failure: mismatched metric keys, or fewer than 2 metrics.

    Distinct from a non-improvement verdict (`RatchetDecision.improved=False`, not an exception) --
    this is raised only when the inputs themselves are malformed.
    """


class RatchetDecision:
    """The composed AC-10 decision: whether the candidate is a genuine multi-metric improvement.

    `per_metric_delta` carries the signed, direction-corrected, relative delta for each metric
    (positive = candidate wins that metric) -- useful for a caller to log/diagnose why a candidate
    passed or failed, without re-deriving the per-metric math.
    """

    __slots__ = ("breakdown_point", "cohens_d", "improved", "per_metric_delta", "win_rate")

    def __init__(
        self,
        *,
        improved: bool,
        win_rate: float,
        breakdown_point: float,
        cohens_d: float,
        per_metric_delta: Mapping[str, float],
    ) -> None:
        self.improved = improved
        self.win_rate = win_rate
        self.breakdown_point = breakdown_point
        self.cohens_d = cohens_d
        self.per_metric_delta = per_metric_delta

    def __repr__(self) -> str:
        return (
            f"RatchetDecision(improved={self.improved!r}, win_rate={self.win_rate!r}, "
            f"breakdown_point={self.breakdown_point!r}, cohens_d={self.cohens_d!r})"
        )


def _relative_delta(candidate_value: float, best_value: float, higher_is_better: bool) -> float:
    raw_delta = candidate_value - best_value
    signed = raw_delta if higher_is_better else -raw_delta
    if best_value == 0:
        return signed
    return signed / abs(best_value)


def _breakdown_point(win_count: int, n: int, win_rate_threshold: float) -> float:
    """The smallest fraction of *winning* metrics that must be removed to drop the win rate
    below `win_rate_threshold`. 0.0 if the win rate is already below threshold (nothing to flip).
    Capped so the search never divides by zero (can't remove every remaining metric).
    """
    if n == 0:
        return 0.0
    win_rate = win_count / n
    if win_rate < win_rate_threshold:
        return 0.0

    max_removable = min(win_count, n - 1)
    for k in range(1, max_removable + 1):
        if (win_count - k) / (n - k) < win_rate_threshold:
            return k / n
    # Removing every winning metric short of the last remaining one still doesn't flip the
    # verdict -- maximally robust within what's computable (can't reduce n to 0).
    return max_removable / n


def compute_ratchet_decision(
    candidate_fitness: Mapping[str, float],
    best_fitness: Mapping[str, float],
    *,
    higher_is_better: Mapping[str, bool],
    win_rate_threshold: float = 0.6,
    breakdown_point_threshold: float = 0.2,
    cohens_d_threshold: float = 0.2,
) -> RatchetDecision:
    """The composed AC-10 gate: WR >= 0.6 AND BP >= 0.2 AND Cohen's d >= 0.2, over the fitness dict.

    Args:
        candidate_fitness: the candidate's fitness dict.
        best_fitness: the current best's fitness dict. Must share exactly the same keys as
            `candidate_fitness` and `higher_is_better`.
        higher_is_better: per-metric direction -- `True` if a larger value is better for that
            metric (e.g. accuracy), `False` if smaller is better (e.g. loss, latency, memory).
            Required explicitly; nothing in the fitness dict itself encodes direction.
        win_rate_threshold: default 0.6, per AC-10's own text.
        breakdown_point_threshold: default 0.2, per AC-10's own text.
        cohens_d_threshold: default 0.2, per AC-10's own text.

    Returns:
        A `RatchetDecision` -- `improved=False` is a normal outcome (the candidate simply didn't
        clear the bar), not an error.

    Raises:
        RatchetInputError: `candidate_fitness`, `best_fitness`, and `higher_is_better` don't share
            exactly the same key set, or there are fewer than 2 metrics (Cohen's d is degenerate
            for a single-point sample).
    """
    candidate_keys = set(candidate_fitness)
    best_keys = set(best_fitness)
    direction_keys = set(higher_is_better)
    if not (candidate_keys == best_keys == direction_keys):
        msg = (
            "candidate_fitness, best_fitness, and higher_is_better must share exactly the same "
            f"metric keys: candidate={sorted(candidate_keys)}, best={sorted(best_keys)}, "
            f"higher_is_better={sorted(direction_keys)}"
        )
        raise RatchetInputError(msg)

    metrics = sorted(candidate_keys)
    n = len(metrics)
    if n < 2:
        msg = f"need at least 2 metrics for a well-defined Cohen's d, got {n}: {metrics}"
        raise RatchetInputError(msg)

    deltas = {
        m: _relative_delta(candidate_fitness[m], best_fitness[m], higher_is_better[m])
        for m in metrics
    }

    win_count = sum(1 for d in deltas.values() if d > 0)
    win_rate = win_count / n

    breakdown_point = _breakdown_point(win_count, n, win_rate_threshold)

    delta_values = list(deltas.values())
    mean_delta = statistics.mean(delta_values)
    stdev_delta = statistics.stdev(delta_values)
    if stdev_delta == 0:
        cohens_d = math.inf if mean_delta > 0 else 0.0
    else:
        cohens_d = mean_delta / stdev_delta

    improved = (
        win_rate >= win_rate_threshold
        and breakdown_point >= breakdown_point_threshold
        and cohens_d >= cohens_d_threshold
    )

    return RatchetDecision(
        improved=improved,
        win_rate=win_rate,
        breakdown_point=breakdown_point,
        cohens_d=cohens_d,
        per_metric_delta=deltas,
    )


__all__ = [
    "RatchetDecision",
    "RatchetInputError",
    "compute_ratchet_decision",
]
