"""Direct bathos-library-import wrapper for stats-battery and seed-floor gates (LC-07, AC-6).

This module wraps pure-read/compute bathos functions directly as a Python library,
bypassing MCP entirely. It feeds results into xtrax's already-merged gate functions
(T2-22's stats_battery_gate and T2-23's seed_gate).

Grounding (per AC-6): bathos.stats_gates.run_stats_battery,
check_baseline_budget_equivalence, bathos.campaigns.count_seeds_for_script, and
count_runs_for_script are all pure functions over caller-supplied data or read-only
database queries — no write-path integrity concern. Controller living outside src/xtrax
violates no existing constraint. This matches #2181's "no bathos item blocks the walking
skeleton" principle.

Per AC-6's explicit requirement: NO MCP call appears anywhere in this item's code path.
Only direct Python imports (lazy-imported at function-call time, not module-import time,
for testability).
"""

from dataclasses import dataclass

# xtrax imports (always available)
from xtrax.loop.seed_gate import SeedTrialCounts
from xtrax.loop.stats_battery_gate import BathosStatsBatteryVerdict

# bathos imports are lazy — deferred to function-call time for testability.


@dataclass(frozen=True, slots=True)
class StatsBatteryResult:
    """Result of calling bathos.stats_gates.run_stats_battery directly.

    Maps bathos's returned verdict shape into xtrax's BathosStatsBatteryVerdict for
    consumption by xtrax.loop.stats_battery_gate.assess_stats_battery_verdict.
    """

    verdict: BathosStatsBatteryVerdict


def call_stats_battery_gate(
    baseline_hpo_trials: int | None = None,
    candidate_hpo_trials: int | None = None,
    baseline_hpo_compute_budget: float | None = None,
    candidate_hpo_compute_budget: float | None = None,
    **stats_arrays,
) -> BathosStatsBatteryVerdict:
    """Call bathos.stats_gates.run_stats_battery with caller-supplied kwargs.

    This wrapper accepts the same baseline/candidate HPO parameters as
    xtrax.run.baseline_budget_emission.BaselineBudgetCounts.as_stats_battery_kwargs(),
    plus the rest of run_stats_battery's own kwargs via **stats_arrays. Note that
    run_stats_battery's own `candidate_values`/`baseline_values` parameters are
    required (not optional) despite this wrapper's signature showing only optional
    HPO fields explicitly — they must be passed by keyword through **stats_arrays
    (e.g. seed_replicates, higher_is_better, alpha are the other real kwargs; see
    bathos.stats_gates.run_stats_battery's own signature for the complete list).

    Args:
        baseline_hpo_trials: Optional baseline HPO trial count.
        candidate_hpo_trials: Optional candidate HPO trial count.
        baseline_hpo_compute_budget: Optional baseline compute budget.
        candidate_hpo_compute_budget: Optional candidate compute budget.
        **stats_arrays: candidate_values/baseline_values (required by bathos) plus any
            other kwargs to pass directly to bathos.stats_gates.run_stats_battery.

    Returns:
        A BathosStatsBatteryVerdict (mirrors bathos.stats_gates.StatsBatteryVerdict verbatim)
        ready for consumption by xtrax.loop.stats_battery_gate.assess_stats_battery_verdict.

    Note:
        This function calls bathos directly as a pure Python library (no MCP, no subprocess).
        It performs no I/O or write operations.
    """
    # Lazy import — only at function-call time, not module-import time.
    import bathos.stats_gates

    # Call bathos directly — pure library import, no MCP.
    bathos_verdict = bathos.stats_gates.run_stats_battery(
        baseline_hpo_trials=baseline_hpo_trials,
        candidate_hpo_trials=candidate_hpo_trials,
        baseline_hpo_compute_budget=baseline_hpo_compute_budget,
        candidate_hpo_compute_budget=candidate_hpo_compute_budget,
        **stats_arrays,
    )

    # Thread the result into xtrax's BathosStatsBatteryVerdict shape.
    return BathosStatsBatteryVerdict(
        verdict=bathos_verdict.verdict,
        scipy_available=bathos_verdict.scipy_available,
        reasons=bathos_verdict.reasons,
        cohens_d=bathos_verdict.cohens_d,
        win_rate=bathos_verdict.win_rate,
        breakdown_point=bathos_verdict.breakdown_point,
        p_superiority=bathos_verdict.p_superiority,
        wilcoxon_p_value=bathos_verdict.wilcoxon_p_value,
        icc=bathos_verdict.icc,
        baseline_budget_equivalent=bathos_verdict.baseline_budget_equivalent,
    )


def get_seed_trial_counts(
    db,
    script_sha256: str,
    hypothesis_clause_id: str = "",
) -> SeedTrialCounts:
    """Query bathos for distinct seed count and trial count for a given script_sha256.

    This wrapper calls bathos.campaigns.count_seeds_for_script and
    count_runs_for_script directly, returning a SeedTrialCounts ready for consumption
    by xtrax.loop.seed_gate.assess_seed_trial_floor.

    Args:
        db: A bathos database connection (caller's responsibility to provide).
        script_sha256: The script's SHA256 hash to query counts for.
        hypothesis_clause_id: Optional label for logging/traceability; defaults to "".

    Returns:
        A SeedTrialCounts containing distinct_seed_count and trial_count, scoped to the
        given script_sha256, ready for assess_seed_trial_floor.

    Note:
        This function calls bathos directly as a pure Python library (no MCP, no subprocess).
        It performs read-only database queries only (no writes).
    """
    # Lazy import — only at function-call time, not module-import time.
    import bathos.campaigns

    # Call bathos directly — pure library import, no MCP.
    distinct_seed_count = bathos.campaigns.count_seeds_for_script(db, script_sha256)
    trial_count = bathos.campaigns.count_runs_for_script(db, script_sha256)

    # Construct and return the xtrax-side shape.
    return SeedTrialCounts(
        script_sha256=script_sha256,
        distinct_seed_count=distinct_seed_count,
        trial_count=trial_count,
        hypothesis_clause_id=hypothesis_clause_id,
    )


__all__ = [
    "call_stats_battery_gate",
    "get_seed_trial_counts",
    "StatsBatteryResult",
]
