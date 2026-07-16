"""Baseline-budget-equivalence emission (T2-24, #2181, AC-17).

AC-17's DAG text (verbatim): "loop emits candidate + baseline HPO-trial/compute counts; conclude
checks the baseline received >= equal HPO trials/compute as the candidate. Success: baseline
budget >= candidate budget. Fast/loud: comparison verdict downgraded; loud in the campaign
report." Backlog-add note (same DAG entry): "Check lives in bathos B2-01 conclude gate."

Grounding (verified before writing any code, not guessed): `depends_on: [T2-22, B2-01]`, both
merged. Confirmed the comparison ITSELF already fully exists and is already wired through, exactly
as the DAG's own "check lives in bathos" note promised:
`bathos.stats_gates.check_baseline_budget_equivalence` (B2-01, bathos main, read at
`origin/main:src/bathos/stats_gates.py`) is a complete, tested function comparing
`baseline_hpo_trials`/`candidate_hpo_trials` and `baseline_hpo_compute_budget`/
`candidate_hpo_compute_budget`, and it is already folded into `run_stats_battery`'s composed
`verdict` -- a failed budget check alone downgrades `verdict` to `"confounded"` via that function's
internal `budget_pass`. T2-22's `xtrax.loop.stats_battery_gate.assess_stats_battery_verdict`
(already merged, this same session) already consumes that composed `verdict` through
`BathosStatsBatteryVerdict.verdict`. So the "conclude checks ..." half of AC-17's gate text is
already fully closed on the bathos side and already threaded through xtrax via T2-22 -- writing a
second comparison here would be pure duplication of `check_baseline_budget_equivalence`, exactly
what the backlog note's own "check lives in bathos" phrasing warns against. This module therefore
implements only the other half: "loop emits candidate + baseline HPO-trial/compute counts."

Scope correction (grounding came out differently from the initial framing on one point): the
initial framing assumed `bth run` already had CLI flags carrying these four counts, mirroring
`xtrax.run.component_binding.ComponentSidecarBinding.as_bth_run_flags()` (B2-08)'s
`--component-id`/`--component-sidecar-sha256` precedent. Grepped bathos's `cli.py` `run` command
directly: it has no `--baseline-hpo-trials`/`--candidate-hpo-trials`/
`--baseline-hpo-compute-budget`/`--candidate-hpo-compute-budget` flags, and no code path in
`bathos.runner.run_script` (or anywhere else in bathos) ever constructs a `Run` with
`baseline_hpo_trials`/`baseline_hpo_compute_budget` set to anything but their `None` default --
B2-02 added the `Run` schema columns, but wiring a CLI/runner input for them is a documented,
deliberate gap on the bathos side (`stats_gates.py`'s own module docstring: "conclude_campaign
wiring -- deliberately NOT done in this PR ... this module therefore exposes only the pure,
caller-supplied-arrays contract"). `run_stats_battery`/`check_baseline_budget_equivalence` take all
four counts as plain Python keyword arguments, not as `Run` row data read back from a catalog query
-- there is nothing shaped like a `bth run` argv flag to target on the bathos side for this data
yet. This module's emission target is therefore that function pair's actual keyword-argument
contract (`as_stats_battery_kwargs`), not a CLI argv list -- `component_binding.py`'s
`as_bth_run_flags` shape doesn't fit because the CLI surface it mirrors doesn't exist for this
data. If bathos later adds `bth run` flags for these fields, a caller can thread this dataclass's
fields through then; this module does not anticipate that shape speculatively.

Cross-repo integration seam (matches `xtrax.run.repro_floor`, `xtrax.loop.metrics_provenance`,
`xtrax.run.component_binding`, and `xtrax.loop.capability_probe_gate`/`stats_battery_gate` exactly
-- all already-merged #2181 items crossing into bathos): xtrax has no dependency on bathos, by
design. This module computes and returns a `BaselineBudgetCounts` -- the four HPO-trial/compute
values a caller passes straight through to `bathos.stats_gates.run_stats_battery`/
`check_baseline_budget_equivalence`. It does not call bathos itself, and it does not decide
pass/fail -- that comparison already lives entirely on the bathos side (see above).
"""

import math
from dataclasses import dataclass


class BaselineBudgetCountsInputError(Exception):
    """A supplied trial count or compute-budget value is negative or non-finite.

    Structurally meaningless (you cannot run a negative number of HPO trials or spend negative
    compute; NaN/Infinity cannot be compared meaningfully either -- and would silently pass
    bathos's own `check_baseline_budget_equivalence`, since a NaN comparison is always `False`,
    producing a false "equivalent" verdict instead of AC-17's own "fast/loud" downgrade) --
    distinct from bathos's own `check_baseline_budget_equivalence`, which never raises and instead
    treats a missing (`None`) value on either side of a dimension as "nothing to compare" (see
    `BaselineBudgetCounts`). This module raises only for a value that is present but out of range;
    mirrors `xtrax.loop.multi_metric_ratchet`'s identical `math.isfinite` guard on caller-supplied
    numeric values.
    """


@dataclass(frozen=True, slots=True)
class BaselineBudgetCounts:
    """Candidate + baseline HPO-trial/compute counts for AC-17's conclude-time comparison.

    Mirrors the four independently-optional parameters of
    `bathos.stats_gates.run_stats_battery`/`check_baseline_budget_equivalence` (B2-01, merged)
    verbatim -- this module does not invent a different shape. A `None` on a given dimension
    (trials or compute) means that dimension has nothing to compare, matching
    `check_baseline_budget_equivalence`'s own documented "a `None` on either side of a given
    dimension... skips that dimension's check" stance -- it is not an error to omit one, or even
    both, dimensions here.
    """

    candidate_hpo_trials: int | None = None
    baseline_hpo_trials: int | None = None
    candidate_hpo_compute_budget: float | None = None
    baseline_hpo_compute_budget: float | None = None

    def as_stats_battery_kwargs(self) -> dict[str, int | float | None]:
        """The exact keyword arguments a caller passes to
        `bathos.stats_gates.run_stats_battery`/`check_baseline_budget_equivalence` (both
        bathos-side, B2-01, merged). Pure dict construction -- this method does not call bathos or
        any subprocess itself.
        """
        return {
            "baseline_hpo_trials": self.baseline_hpo_trials,
            "candidate_hpo_trials": self.candidate_hpo_trials,
            "baseline_hpo_compute_budget": self.baseline_hpo_compute_budget,
            "candidate_hpo_compute_budget": self.candidate_hpo_compute_budget,
        }


def baseline_budget_counts(
    *,
    candidate_hpo_trials: int | None = None,
    baseline_hpo_trials: int | None = None,
    candidate_hpo_compute_budget: float | None = None,
    baseline_hpo_compute_budget: float | None = None,
) -> BaselineBudgetCounts:
    """Assemble a `BaselineBudgetCounts`, validating every supplied count/budget is non-negative.

    Args:
        candidate_hpo_trials: HPO trial count for the candidate arm. `None` if not tracked for
            this campaign.
        baseline_hpo_trials: HPO trial count for the baseline arm. `None` if not tracked for this
            campaign.
        candidate_hpo_compute_budget: HPO compute-budget value for the candidate arm
            (caller-defined unit -- GPU-hours, FLOPs, etc. -- must match whatever unit the
            campaign's own bookkeeping uses on both sides). `None` if not tracked for this
            campaign.
        baseline_hpo_compute_budget: HPO compute-budget value for the baseline arm, same unit
            convention as `candidate_hpo_compute_budget`. `None` if not tracked for this campaign.

    Returns:
        A `BaselineBudgetCounts` holding exactly the four supplied values, unmodified.

    Raises:
        BaselineBudgetCountsInputError: any supplied value is negative or non-finite (NaN/Inf).
    """
    for name, value in (
        ("candidate_hpo_trials", candidate_hpo_trials),
        ("baseline_hpo_trials", baseline_hpo_trials),
        ("candidate_hpo_compute_budget", candidate_hpo_compute_budget),
        ("baseline_hpo_compute_budget", baseline_hpo_compute_budget),
    ):
        if value is None:
            continue
        if not math.isfinite(value):
            msg = f"{name} must be finite (got {value}); NaN/Infinity is not meaningful"
            raise BaselineBudgetCountsInputError(msg)
        if value < 0:
            msg = (
                f"{name} must be >= 0 (got {value}); a negative trial count or compute budget "
                "is not meaningful"
            )
            raise BaselineBudgetCountsInputError(msg)

    return BaselineBudgetCounts(
        candidate_hpo_trials=candidate_hpo_trials,
        baseline_hpo_trials=baseline_hpo_trials,
        candidate_hpo_compute_budget=candidate_hpo_compute_budget,
        baseline_hpo_compute_budget=baseline_hpo_compute_budget,
    )


__all__ = [
    "BaselineBudgetCounts",
    "BaselineBudgetCountsInputError",
    "baseline_budget_counts",
]
