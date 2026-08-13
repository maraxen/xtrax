"""Direct bathos-library-import wrappers for gate functions (LC-07, AC-6, GW-01).

This module wraps pure-read/compute bathos functions directly as a Python library,
bypassing MCP entirely. It feeds results into xtrax's already-merged gate functions.

Grounding (per AC-6): bathos.stats_gates.run_stats_battery, check_baseline_budget_
equivalence, bathos.campaigns.count_seeds_for_script, count_runs_for_script,
bathos.prereg.verify_run_manifest, and bathos.prereg.check_sidecar_drift are all pure
functions over caller-supplied data or read-only database queries — no write-path
integrity concern. Controller living outside src/xtrax violates no existing constraint.
This matches #2181's "no bathos item blocks the walking skeleton" principle.

Per AC-6's explicit requirement: NO MCP call appears anywhere in this item's code path.
Only direct Python imports (lazy-imported at function-call time, not module-import time,
for testability).

GW-01 additions (step 2.1/2.2): get_evidence_candidate_for_run and get_sidecar_drift_
signal wrap attestation_evidence_gate and sidecar_drift_gate gate requirements; they
resolve Run objects via bathos.query.get_run and call the real bathos prereg verification
functions (verify_run_manifest, check_sidecar_drift).
"""

from dataclasses import dataclass

# xtrax imports (always available)
from xtrax.loop.attestation_evidence_gate import EvidenceCandidate
from xtrax.loop.capability_probe_gate import CapabilityProbeResult
from xtrax.loop.seed_gate import SeedTrialCounts
from xtrax.loop.sidecar_drift_gate import SidecarDriftSignal
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


def get_evidence_candidate_for_run(
    run_id: str,
    catalog_dir: str = "",
    stdout_verified: bool | None = None,
) -> EvidenceCandidate:
    """Verify a run's attestation evidence and return an EvidenceCandidate for admission.

    This wrapper resolves a Run object via bathos.query.get_run (step 1.2's grounding
    strategy), calls bathos.prereg.verify_run_manifest on it, and threads the result into
    xtrax.loop.attestation_evidence_gate.EvidenceCandidate for consumption by
    admit_evidence.

    Args:
        run_id: The bathos run ID to verify evidence for (obtained from CandidateRunResult).
        catalog_dir: bathos catalog directory (empty = use default). Forwarded to
            bathos.query.get_run.
        stdout_verified: Optional caller-supplied stdout verification result (True iff
            `Run.stdout_sha256` was recorded AND the caller's own re-hash matched it; False
            if recorded but mismatched; None if never recorded). Passed through to
            EvidenceCandidate verbatim -- see attestation_evidence_gate's module docstring
            for the integration seam.

    Returns:
        An EvidenceCandidate with run_id, manifest_verified (from verify_run_manifest),
        and stdout_verified (from the caller-supplied passthrough argument).

    Raises:
        ValueError: run_id is empty or the run was not found in the catalog.

    Note:
        This function calls bathos directly as a pure Python library (no MCP, no subprocess).
        It performs read-only database queries only (no writes). See the module docstring
        for the cross-repo integration seam (bathos.prereg verification, not xtrax-side
        verification).
    """
    # Lazy imports — only at function-call time, not module-import time.
    import bathos.prereg
    import bathos.query

    if not run_id:
        raise ValueError("run_id is required and cannot be empty")

    # Resolve the Run object via the step 1.2 strategy (direct bathos.query.get_run)
    from pathlib import Path as PathlibPath

    cat_dir = PathlibPath(catalog_dir) if catalog_dir else PathlibPath.home() / ".bth"
    run = bathos.query.get_run(run_id, cat_dir)
    if run is None:
        raise ValueError(f"run {run_id!r} not found in catalog at {cat_dir}")

    # Call verify_run_manifest (step 2.1)
    manifest_verified = bathos.prereg.verify_run_manifest(run)

    # Thread into xtrax's EvidenceCandidate shape
    return EvidenceCandidate(
        run_id=run_id,
        manifest_verified=manifest_verified,
        stdout_verified=stdout_verified,
    )


def get_sidecar_drift_signal(
    script_path: str,
    catalog_dir: str = "",
    current_sidecar_sha256: str = "",
    script_id: str = "",
) -> SidecarDriftSignal:
    """Check for sidecar drift and return a SidecarDriftSignal for AC-18 reaction.

    This wrapper calls bathos.prereg.check_sidecar_drift (step 2.2) and threads the result
    into xtrax.loop.sidecar_drift_gate.SidecarDriftSignal for consumption by
    assert_sidecar_drift_reaction.

    Args:
        script_path: Path to the candidate script to check sidecar drift for.
        catalog_dir: bathos catalog directory (empty = use default). Forwarded to
            bathos.prereg.check_sidecar_drift.
        current_sidecar_sha256: The sidecar SHA256 from the current run (typically from
            CandidateRunResult or a Run object). Required for meaningful drift detection.
        script_id: Optional human-readable script identifier (path or stem) for message
            building. If empty, script_path is used as the identifier.

    Returns:
        A SidecarDriftSignal with drifted (from check_sidecar_drift), script_id, and
        sidecar hashes (for message building).

    Note:
        This function calls bathos directly as a pure Python library (no MCP, no subprocess).
        It performs read-only database queries only (no writes). The actual bathos call
        (check_sidecar_drift) is a pure function with no write operations.
    """
    # Lazy import — only at function-call time, not module-import time.
    # Resolve catalog_dir
    from pathlib import Path as PathlibPath

    import bathos.prereg

    cat_dir = PathlibPath(catalog_dir) if catalog_dir else PathlibPath.home() / ".bth"
    script = PathlibPath(script_path)

    # Call check_sidecar_drift (step 2.2) — returns True iff the current run's sidecar
    # hash differs from the script's first-run manifest baseline.
    drifted = bathos.prereg.check_sidecar_drift(script, cat_dir, current_sidecar_sha256)

    # Use script_id if provided, else derive from script_path
    script_id_final = script_id if script_id else script.stem

    # Thread into xtrax's SidecarDriftSignal shape
    return SidecarDriftSignal(
        drifted=drifted,
        script_id=script_id_final,
        first_run_sha256="",  # Not queried by this wrapper (caller responsible)
        current_sha256=current_sidecar_sha256,
    )


def get_capability_probe_result(catalog_dir: str = "") -> CapabilityProbeResult:
    """Probe bathos capability and return a CapabilityProbeResult.

    This wrapper calls bathos.capability.probe_capabilities and threads the result
    into xtrax.loop.capability_probe_gate.CapabilityProbeResult for consumption by
    assert_capability_live (T2-27, AC-20).

    Args:
        catalog_dir: bathos catalog directory (empty = use default, resolved via
            bathos.config.default_catalog_dir()). Matches the resolution order used
            by bathos/mcp.py's _get_catalog_dir.

    Returns:
        A CapabilityProbeResult with seed_live and stats_battery_live booleans,
        ready for consumption by assert_capability_live.

    Note:
        This function calls bathos directly as a pure Python library (no MCP, no subprocess).
        It performs read-only database queries only (no writes). See capability_probe_gate's
        module docstring for the cross-repo integration seam (bathos.capability probe, not
        xtrax-side verification).
    """
    # Lazy imports — only at function-call time, not module-import time.
    import os
    from pathlib import Path as PathlibPath

    import bathos.capability
    import bathos.config

    # Resolve catalog_dir matching bathos/mcp.py's _get_catalog_dir order:
    # catalog_dir arg > BTH_CATALOG_DIR env > bathos.config.default_catalog_dir()
    if catalog_dir:
        resolved_dir = PathlibPath(catalog_dir)
    else:
        env_dir = os.environ.get("BTH_CATALOG_DIR", "")
        if env_dir:
            resolved_dir = PathlibPath(env_dir)
        else:
            resolved_dir = PathlibPath(bathos.config.default_catalog_dir())

    # Call probe_capabilities (T2-27's step 1)
    report = bathos.capability.probe_capabilities(resolved_dir)

    # Thread into xtrax's CapabilityProbeResult shape, mapping the returned
    # CapabilityReport's two independent liveness booleans exactly (see
    # capability_probe_gate's module docstring). Deliberately drop
    # missing_seed_columns/stats_unavailable_reason since CapabilityProbeResult's
    # fixed 2-field shape doesn't carry them.
    return CapabilityProbeResult(
        seed_live=report.seed_live,
        stats_battery_live=report.stats_battery_live,
    )


__all__ = [
    "call_stats_battery_gate",
    "get_capability_probe_result",
    "get_evidence_candidate_for_run",
    "get_seed_trial_counts",
    "get_sidecar_drift_signal",
    "StatsBatteryResult",
]
