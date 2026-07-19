"""Direct bathos-library-import wrapper for read-only #2181 gate data (LC-07, AC-6; [GW-01]).

This module wraps pure-read/compute bathos functions directly as a Python library,
bypassing MCP entirely. It feeds results into xtrax's already-merged gate functions
(T2-22's stats_battery_gate, T2-23's seed_gate, and -- added for [GW-01], backlog id
3648 -- T2-26's attestation_evidence_gate and T2-25's sidecar_drift_gate).

Grounding (per AC-6): bathos.stats_gates.run_stats_battery,
check_baseline_budget_equivalence, bathos.campaigns.count_seeds_for_script, and
count_runs_for_script are all pure functions over caller-supplied data or read-only
database queries — no write-path integrity concern. Controller living outside src/xtrax
violates no existing constraint. This matches #2181's "no bathos item blocks the walking
skeleton" principle. The same reasoning covers [GW-01]'s two additions:
bathos.query.get_run, bathos.prereg.verify_run_manifest, and
bathos.prereg.check_sidecar_drift are all read-only over an already-completed run's
catalog row -- no write-path integrity concern either.

Per AC-6's explicit requirement: NO MCP call appears anywhere in this item's code path.
Only direct Python imports (lazy-imported at function-call time, not module-import time,
for testability).
"""

import os
from dataclasses import dataclass
from pathlib import Path

# xtrax imports (always available)
from xtrax.loop.attestation_evidence_gate import EvidenceCandidate
from xtrax.loop.seed_gate import SeedTrialCounts
from xtrax.loop.sidecar_drift_gate import SidecarDriftSignal
from xtrax.loop.stats_battery_gate import BathosStatsBatteryVerdict

# bathos imports are lazy — deferred to function-call time for testability.

#: Mirrors bathos.mcp._get_catalog_dir's own resolution order exactly (param -> env var ->
#: bathos.config.default_catalog_dir()) -- re-implemented here rather than imported, matching
#: bathos_campaign_adapter.py's own _token_path precedent (that module independently
#: re-implements bathos.mcp_auth's private token-path resolution rather than reaching into
#: bathos.mcp's private helpers for a cross-repo caller).
_CATALOG_DIR_ENV_VAR = "BTH_CATALOG_DIR"


def _resolve_catalog_dir(catalog_dir: str) -> Path:
    """Resolve a catalog directory: explicit param, then env var, then bathos's own default."""
    if catalog_dir:
        return Path(catalog_dir)
    override = os.environ.get(_CATALOG_DIR_ENV_VAR)
    if override:
        return Path(override)
    import bathos.config

    return bathos.config.default_catalog_dir()


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


def get_evidence_candidate_for_run(
    run_id: str,
    *,
    catalog_dir: str = "",
    stdout_verified: bool | None = None,
) -> EvidenceCandidate:
    """Fetch `run_id`'s real bathos Run and verify its manifest, for [GW-01] (backlog id 3648).

    Calls bathos.query.get_run(run_id, catalog_dir) then bathos.prereg.verify_run_manifest(run)
    directly, returning an EvidenceCandidate ready for
    xtrax.loop.attestation_evidence_gate.admit_evidence.

    Args:
        run_id: the bathos run's real id (`CandidateRunResult.run_id`, populated from the
            `run` MCP tool's own envelope).
        catalog_dir: bathos catalog directory. Empty string resolves the same way bathos's
            own MCP layer does (env var, then bathos.config.default_catalog_dir()) -- see
            `_resolve_catalog_dir`.
        stdout_verified: caller-supplied result of independently re-hashing a stored stdout
            artifact against `Run.stdout_sha256` -- bathos does not persist raw stdout to
            re-hash against itself (verify_run_manifest's own docstring), so this module has
            no way to compute this itself. Defaults to `None` ("not recorded/not checked"),
            matching `Run.stdout_sha256`'s own nullable default.

    Returns:
        An EvidenceCandidate with `manifest_verified=False` (and no further attributes
        populated) if `run_id` names no run in this catalog at all -- an unfound run is, by
        construction, unverifiable evidence, not a structural error this wrapper raises for.

    Note:
        This function calls bathos directly as a pure Python library (no MCP, no subprocess).
        It performs read-only catalog queries and a local file hash -- no writes.
    """
    # Lazy import — only at function-call time, not module-import time.
    import bathos.prereg
    import bathos.query

    resolved_catalog_dir = _resolve_catalog_dir(catalog_dir)
    run = bathos.query.get_run(run_id, resolved_catalog_dir)
    manifest_verified = bathos.prereg.verify_run_manifest(run) if run is not None else False

    return EvidenceCandidate(
        run_id=run_id,
        manifest_verified=manifest_verified,
        stdout_verified=stdout_verified,
    )


def get_sidecar_drift_signal(
    script_path: Path,
    run_id: str,
    *,
    catalog_dir: str = "",
) -> SidecarDriftSignal:
    """Fetch `run_id`'s sidecar hash and check it against `script_path`'s first-run manifest.

    Calls bathos.query.get_run(run_id, catalog_dir) to get the just-completed run's own
    sidecar_sha256, then bathos.prereg.check_sidecar_drift(script_path, catalog_dir,
    current_sidecar_sha256) directly, returning a SidecarDriftSignal ready for
    xtrax.loop.sidecar_drift_gate.assert_sidecar_drift_reaction.

    Args:
        script_path: path to the candidate script bathos ran (matches
            check_sidecar_drift's own script-identity matching, which resolves this path
            and compares against each catalog run's recorded `command`).
        run_id: the bathos run's real id, used only to fetch this run's own
            `sidecar_sha256` (the "current" hash `check_sidecar_drift` compares against the
            script's first-run baseline).
        catalog_dir: bathos catalog directory, same resolution as
            `get_evidence_candidate_for_run`.

    Returns:
        A SidecarDriftSignal. `first_run_sha256` is left `""` -- `check_sidecar_drift`
        itself does not return the baseline hash it compared against, only whether they
        differ (message-building-only field, per `SidecarDriftSignal`'s own docstring).
        If `run_id` names no run in this catalog (or the run has no recorded sidecar hash),
        `current_sha256` is `""` and `drifted` is `False` -- matching
        `check_sidecar_drift`'s own "nothing to compare against yet" semantics for an empty
        `current_sidecar_sha256`.

    Note:
        This function calls bathos directly as a pure Python library (no MCP, no subprocess).
        It performs read-only catalog queries -- no writes.
    """
    # Lazy import — only at function-call time, not module-import time.
    import bathos.prereg
    import bathos.query

    resolved_catalog_dir = _resolve_catalog_dir(catalog_dir)
    run = bathos.query.get_run(run_id, resolved_catalog_dir)
    current_sha256 = run.sidecar_sha256 if run is not None else ""

    drifted = bathos.prereg.check_sidecar_drift(script_path, resolved_catalog_dir, current_sha256)

    return SidecarDriftSignal(
        drifted=drifted,
        script_id=str(script_path),
        first_run_sha256="",
        current_sha256=current_sha256,
    )


__all__ = [
    "call_stats_battery_gate",
    "get_evidence_candidate_for_run",
    "get_seed_trial_counts",
    "get_sidecar_drift_signal",
    "StatsBatteryResult",
]
