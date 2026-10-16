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

from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

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
) -> ProbeRecord:
    """Construct, validate, write, and return a ProbeRecord in one call.

    Explicit-construction style per the contract (no decorator, no context
    manager): every argument remains visible at the call site. Provenance
    fields (timestamp/jax versions/xla_flags/device_kind) are left to
    ProbeRecord's default_factories -- auto-captured on THIS machine.

    git_sha is resolved in this order: an explicit `git_sha` is used as-is;
    else `repo_root` (if given) is passed to the git-sha capture, anchoring it
    at that repository; else the default factory resolves from the cwd's
    repository. Pass `repo_root` when the driver's cwd is not the repository
    the measured code lives in.

    Raises ClaimValidityError on any contract violation (including the
    attribution-required-when-scopes rule); the record is written only after
    construction succeeded, so no invalid artifact can land on disk.
    """
    if git_sha is None and repo_root is not None:
        git_sha = capture_git_sha(repo_root)
    provenance: dict[str, Any] = {} if git_sha is None else {"git_sha": git_sha}
    record = ProbeRecord(
        probe_id=probe_id,
        stage=stage,
        n_atoms=n_atoms,
        platform=platform,
        metrics=dict(metrics),
        scopes=scopes,
        attribution_method=attribution_method,
        config=dict(config or {}),
        **provenance,
    )
    record.write(path)
    return record


def probe_record_field_names() -> tuple[str, ...]:
    """Field names of ProbeRecord, contract-order (driver introspection aid)."""
    return tuple(f.name for f in dataclass_fields(ProbeRecord))
