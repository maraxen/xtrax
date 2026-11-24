"""ProbeRecord emission helpers for xtrax probe drivers.

Generic, domain-vocabulary-free: the caller supplies the full config dict
and scope labels (per D8 of
.praxia/docs/specs/260824_upstream-profiling-probe-tooling-from-prolix.md,
label vocab lives in drivers, never here). Adapted from prolix's
scripts/experiments/profile_b1_flash_vs_autodiff_forces.py::_emit_probe_record.

**The empty-attribution rule is load-bearing.** When a trace was captured
(scopes is not None) but every known label came back absent from it, the
attribution method map must be an EMPTY DICT, not None: scopes-without-
attribution and no-trace-at-all are different facts about a measurement.
Prolix's cluster array 20762713 lost this distinction to a `{...} or None`
collapse and failed at claim time; the behavior is pinned by
tests/profiling/test_emitters.py.
"""

__provenance__ = {
    "upstream": "prolix",
    "relationship": "derived",
    "waiver_reason": (
        "Adapted from prolix "
        "scripts/experiments/profile_b1_flash_vs_autodiff_forces.py::_emit_probe_record. "
        "Sibling modules and "
        ".praxia/docs/specs/260824_upstream-profiling-probe-tooling-from-prolix.md "
        "name branch wt-20260807-132628, which is not a commit sha or tag, and "
        "this repo records no SPDX licence for prolix."
    ),
}

import os
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

from xtrax.profiling.compare import DeltaVerdict
from xtrax.profiling.record import ProbeRecord, capture_git_sha

ATTRIBUTION_NAMED_SCOPE = "named_scope"
ATTRIBUTION_OP_NAME = "op_name"


def attribution_from_scopes(
    scopes: dict[str, tuple[float, int] | None] | None,
    *,
    method: str = ATTRIBUTION_NAMED_SCOPE,
) -> dict[str, str]:
    """Attribution-method map for `scopes`: one entry per NON-None value.

    Returns {} (never None) when every value is None -- that combination
    means "trace captured, labels expected but all absent", which ProbeRecord
    must keep distinguishable from "no trace captured" (scopes=None).
    """
    if scopes is None:
        return {}
    return {label: method for label, value in scopes.items() if value is not None}


def emit_probe_record(
    *,
    path: str | Path,
    probe_id: str,
    stage: int,
    n_atoms: int,
    platform: str,
    metrics: dict[str, float | int | str],
    scopes: dict[str, tuple[float, int] | None] | None = None,
    attribution_method: dict[str, str] | None = None,
    config: dict[str, str] | None = None,
    git_sha: str | None = None,
    repo_root: Path | None = None,
    bathos_run_id: str | None = None,
    bathos_campaign_id: str | None = None,
    capture_bathos_env: bool = True,
    num_replicas: int = 1,
    replica_id: int | None = None,
    bucket_index: int | None = None,
    bucket_capacity: int | None = None,
    x64_enabled: bool | None = None,
    device_kind: str | None = None,
    jax_version: str | None = None,
    jaxlib_version: str | None = None,
    xla_flags: str | None = None,
) -> ProbeRecord:
    """Construct, validate, write, and return a ProbeRecord in one call.

    Explicit-construction style per the contract (no decorator, no context
    manager): every argument remains visible at the call site. Provenance
    fields left as ``None`` delegate to ``ProbeRecord``'s ``default_factories``
    on this machine; passing explicit overrides allows stateless subprocess
    workers (e.g. Pass B diagnostic capture) to stamp their own environment.

    ``git_sha`` is resolved in this order: an explicit ``git_sha`` is used
    as-is; else ``repo_root`` (if given) is passed to git-sha capture; else the
    default factory resolves from the cwd's repository. Optional
    ``bathos_run_id`` / ``bathos_campaign_id`` (or ``BTH_RUN_ID`` /
    ``BTH_CAMPAIGN_ID`` when ``capture_bathos_env=True``) are recorded in
    ``config`` without importing ``bathos``.

    Raises ClaimValidityError on any contract violation (including the
    attribution-required-when-scopes rule); the record is written only after
    construction succeeded, so no invalid artifact can land on disk.
    """
    if git_sha is None and repo_root is not None:
        git_sha = capture_git_sha(repo_root)

    merged_config = dict(config or {})
    eff_run_id = (
        bathos_run_id.strip()
        if bathos_run_id is not None
        else (os.environ.get("BTH_RUN_ID", "").strip() if capture_bathos_env else "")
    )
    if eff_run_id:
        merged_config["bathos_run_id"] = eff_run_id
    eff_campaign_id = (
        bathos_campaign_id.strip()
        if bathos_campaign_id is not None
        else (os.environ.get("BTH_CAMPAIGN_ID", "").strip() if capture_bathos_env else "")
    )
    if eff_campaign_id:
        merged_config["bathos_campaign_id"] = eff_campaign_id

    provenance: dict[str, Any] = {}
    optional_prov: tuple[tuple[str, Any], ...] = (
        ("git_sha", git_sha),
        ("x64_enabled", x64_enabled),
        ("device_kind", device_kind),
        ("jax_version", jax_version),
        ("jaxlib_version", jaxlib_version),
        ("xla_flags", xla_flags),
    )
    for field_name, value in optional_prov:
        if value is not None:
            provenance[field_name] = value

    record = ProbeRecord(
        probe_id=probe_id,
        stage=stage,
        n_atoms=n_atoms,
        platform=platform,
        metrics=dict(metrics),
        scopes=scopes,
        attribution_method=attribution_method,
        config=merged_config,
        num_replicas=num_replicas,
        replica_id=replica_id,
        bucket_index=bucket_index,
        bucket_capacity=bucket_capacity,
        **provenance,
    )
    record.write(path)
    return record


def probe_record_to_bathos_metrics(
    record: ProbeRecord,
    verdict: DeltaVerdict | None = None,
) -> dict[str, float | str]:
    """Serialize ``ProbeRecord`` metrics and verdict summary for experiment tracking.

    Pure formatting helper with zero ``bathos`` imports so ``xtrax.profiling``
    stays dependency-free while feeding sidecar ``[outcomes]`` cleanly.
    """
    out: dict[str, float | str] = {key: float(val) for key, val in record.metrics.items()}
    out["probe_id"] = record.probe_id
    out["git_sha"] = record.git_sha
    out["platform"] = record.platform
    out["stage"] = float(record.stage)
    out["n_atoms"] = float(record.n_atoms)
    out["num_replicas"] = float(record.num_replicas)
    if record.device_kind is not None:
        out["device_kind"] = record.device_kind
    if "bathos_run_id" in record.config:
        out["bathos_run_id"] = record.config["bathos_run_id"]
    if "bathos_campaign_id" in record.config:
        out["bathos_campaign_id"] = record.config["bathos_campaign_id"]
    if verdict is not None:
        out["verdict_status"] = verdict.status.value
        out["promotable"] = "true" if verdict.promotable else "false"
        out["primary_metric"] = verdict.primary_metric
        if verdict.speedup_x is not None:
            out["speedup_x"] = float(verdict.speedup_x)
        if verdict.rel_delta_pct is not None:
            out["rel_delta_pct"] = float(verdict.rel_delta_pct)
        if verdict.noise_floor_x is not None:
            out["noise_floor_x"] = float(verdict.noise_floor_x)
    return out


def probe_record_field_names() -> tuple[str, ...]:
    """Field names of ProbeRecord, contract-order (driver introspection aid)."""
    return tuple(f.name for f in dataclass_fields(ProbeRecord))
