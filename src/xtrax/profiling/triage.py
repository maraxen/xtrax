"""Deterministic bottleneck classification over ProbeRecord bundles.

Maps production, diagnostic (``cmdbuf=off``), precision-pair, and
batch-scaling ``ProbeRecord`` telemetry into deterministically ranked
:class:`BottleneckFinding` items linked to ``xtrax-optimizing`` remediation
playbooks.

Zero module-level ``jax`` import and strictly scoped within
``xtrax.profiling.*`` per the profiling leaf-package contract.
"""

import dataclasses
import enum
import math

from xtrax.profiling.claims import (
    CONTRACT_VERSION,
    ClaimClass,
    ClaimValidityError,
    _reject_unverifiable_git_sha,
)
from xtrax.profiling.compare import _shas_match
from xtrax.profiling.kernel_probe import compute_precision_ratio
from xtrax.profiling.record import ProbeRecord
from xtrax.profiling.rules import EXEC_PER_ELEMENT_FLAG

_BATCH_SCALING_UNANIMITY: tuple[str, ...] = (
    "platform",
    "device_kind",
    "x64_enabled",
    "xla_flags",
    "jax_version",
    "jaxlib_version",
    "n_atoms",
    "bucket_capacity",
)
_VALID_BASIS = frozenset({"batch", "per_replica"})

_SEVERITY_RANK = {"blocking": 0, "high": 1, "medium": 2}
_CATEGORY_ORDER: dict[str, int] = {
    "recompile_drift": 0,
    "host_boundary_sync": 1,
    "fp64_pipe_saturation": 2,
    "spatial_dead_work": 3,
    "register_spill_pressure": 4,
    "atomic_contention": 5,
}


class BottleneckCategory(enum.Enum):
    """Canonical kernel and host-loop bottleneck classifications."""

    RECOMPILE_DRIFT = "recompile_drift"
    HOST_BOUNDARY_SYNC = "host_boundary_sync"
    FP64_PIPE_SATURATION = "fp64_pipe_saturation"
    SPATIAL_DEAD_WORK = "spatial_dead_work"
    REGISTER_SPILL_PRESSURE = "register_spill_pressure"
    ATOMIC_CONTENTION = "atomic_contention"


@dataclasses.dataclass(frozen=True, slots=True)
class TriageThresholds:
    """Decision thresholds for deterministic bottleneck triage."""

    exec_per_element_flag: float = EXEC_PER_ELEMENT_FLAG
    host_share_flag: float = 0.25
    mixed_over_single_flag: float = 1.35
    top_kernel_pct_flag: float = 30.0
    active_work_ratio_flag: float = 0.25
    ptx_regs_flag: int = 160
    ptx_vregs_flag: int = 256
    ptx_lmem_bytes_flag: int = 1
    batch_scaling_eff_flag: float = 0.50
    ctas_per_atomic_target_flag: float = 64.0
    low_fp64_device_substrings: tuple[str, ...] = (
        "l40",
        "l4",
        "rtx",
        "geforce",
        "4090",
        "5090",
        "rtx_pro_6000",
    )


@dataclasses.dataclass(frozen=True, slots=True)
class BottleneckFinding:
    """One ranked bottleneck finding emitted by :func:`triage_probe_bundle`."""

    category: BottleneckCategory
    severity: str
    target_kernel: str | None
    evidence: dict[str, float | str]
    summary: str
    playbook_ref: str
    threshold_status: str


def _top_scope_share(diag: ProbeRecord | None) -> tuple[str | None, float | None]:
    """Return ``(top_kernel_name, top_kernel_pct)`` from non-None ``diag.scopes``."""
    if diag is None or not diag.scopes:
        return None, None
    present = [(name, float(val[0])) for name, val in diag.scopes.items() if val is not None]
    if not present:
        return None, None
    total_exc = sum(sec for _, sec in present)
    top_name, top_sec = max(present, key=lambda item: (item[1], item[0]))
    top_pct = (100.0 * top_sec / total_exc) if total_exc > 0.0 else 0.0
    return top_name, top_pct


def _max_metric_across(
    records: tuple[ProbeRecord | None, ...],
    key: str,
) -> float | None:
    """Return maximum value of ``key`` across non-None records, or ``None`` if absent."""
    vals = [float(r.metrics[key]) for r in records if r is not None and key in r.metrics]
    return max(vals) if vals else None


def _rule_recompile_drift(prod: ProbeRecord) -> BottleneckFinding | None:
    """Flag repeated JIT recompilations detected in production probing."""
    n_comp = float(prod.metrics.get("n_compilations", 0.0))
    if n_comp <= 0.0:
        return None
    return BottleneckFinding(
        category=BottleneckCategory.RECOMPILE_DRIFT,
        severity="blocking",
        target_kernel=None,
        evidence={"n_compilations": n_comp},
        summary=f"Steady-state recompilations detected (n_compilations={n_comp:.3g} > 0.0).",
        playbook_ref=(
            "agent_assets/skills/xtrax-optimizing/SKILL.md"
            "#remediation-playbook-keyed-by-xtrax-profile-flags"
        ),
        threshold_status="calibrated",
    )


def _rule_host_boundary_sync(
    prod: ProbeRecord,
    thresholds: TriageThresholds,
) -> BottleneckFinding | None:
    """Flag host-boundary synchronization or high per-element dispatch overhead."""
    exec_pe = float(prod.metrics.get("exec_per_element", 0.0))
    total_s = float(prod.metrics.get("total_step_seconds", 0.0))
    host_s = float(prod.metrics.get("host_seconds", 0.0))
    host_share = (host_s / total_s) if total_s > 0.0 else 0.0
    exec_hit = exec_pe >= thresholds.exec_per_element_flag
    share_hit = total_s > 0.0 and host_share >= thresholds.host_share_flag
    if not (exec_hit or share_hit):
        return None
    status = "calibrated" if exec_hit else "provisional_gpu_heuristic"
    return BottleneckFinding(
        category=BottleneckCategory.HOST_BOUNDARY_SYNC,
        severity="high",
        target_kernel=None,
        evidence={"exec_per_element": exec_pe, "host_share": host_share},
        summary=(
            f"Host dispatch/sync overhead (exec_per_element={exec_pe:.4f}, "
            f"host_share={host_share:.1%})."
        ),
        playbook_ref=(
            "agent_assets/skills/xtrax-optimizing/references/tier1-host-boundary.md"
            "#decision-rules-verify-paths"
        ),
        threshold_status=status,
    )


def _rule_fp64_saturation(
    prod: ProbeRecord,
    diag: ProbeRecord | None,
    precision_pair: tuple[ProbeRecord, ProbeRecord] | None,
    thresholds: TriageThresholds,
) -> BottleneckFinding | None:
    """Flag FP64 pipe saturation or residual ``.f64`` virtual register pressure."""
    ratio: float | None = None
    if precision_pair is not None:
        ratio = compute_precision_ratio(*precision_pair)
    elif "mixed_over_single_ratio" in prod.metrics:
        ratio = float(prod.metrics["mixed_over_single_ratio"])
        if not math.isfinite(ratio) or ratio <= 0.0:
            raise ClaimValidityError(
                f"mixed_over_single_ratio must be finite and > 0.0, got {ratio!r}"
            )
    if ratio is None or ratio < thresholds.mixed_over_single_flag:
        return None

    f64_vregs = _max_metric_across((prod, diag), "ptx_f64_vregs_max") or 0.0
    dev = (prod.device_kind or "").lower()
    low_fp64_dev = any(sub in dev for sub in thresholds.low_fp64_device_substrings)
    if not (low_fp64_dev or f64_vregs > 0.0):
        return None

    top_name, _ = _top_scope_share(diag)
    return BottleneckFinding(
        category=BottleneckCategory.FP64_PIPE_SATURATION,
        severity="high",
        target_kernel=top_name,
        evidence={
            "mixed_over_single_ratio": ratio,
            "ptx_f64_vregs_max": f64_vregs,
            "device_kind": prod.device_kind or "",
        },
        summary=(
            f"Mixed precision slowdown {ratio:.2f}x with "
            f"device_kind={prod.device_kind!r} / f64_vregs={f64_vregs:.0f}."
        ),
        playbook_ref=(
            "agent_assets/skills/xtrax-optimizing/references/kernel-optimization-loop.md"
            "#fp64-pipe-saturation"
        ),
        threshold_status="provisional_gpu_heuristic",
    )


def _rule_spatial_dead_work(
    prod: ProbeRecord,
    diag: ProbeRecord | None,
    thresholds: TriageThresholds,
) -> BottleneckFinding | None:
    """Flag low active-work fraction caused by padding or tile-pair dead work."""
    if "active_work_ratio" not in prod.metrics:
        return None
    ratio = float(prod.metrics["active_work_ratio"])
    if not (0.0 <= ratio < thresholds.active_work_ratio_flag):
        return None
    top_name, top_pct = _top_scope_share(diag)
    if top_pct is not None and top_pct < thresholds.top_kernel_pct_flag:
        return None
    evidence: dict[str, float | str] = {"active_work_ratio": ratio}
    if top_pct is not None:
        evidence["top_kernel_pct"] = top_pct
    return BottleneckFinding(
        category=BottleneckCategory.SPATIAL_DEAD_WORK,
        severity="high",
        target_kernel=top_name,
        evidence=evidence,
        summary=(
            f"Low active work ratio ({ratio:.1%}) with dominant kernel "
            f"{top_name or '<unattributed>'}."
        ),
        playbook_ref=(
            "agent_assets/skills/xtrax-optimizing/references/"
            "tier3-pallas-megakernels-and-ragged-buckets.md"
            "#1-handling-ragged-axes-n_atoms-n_waters-and-ghost-atom-compaction"
        ),
        threshold_status="provisional_gpu_heuristic",
    )


def _rule_register_spill(
    prod: ProbeRecord,
    diag: ProbeRecord | None,
    thresholds: TriageThresholds,
) -> BottleneckFinding | None:
    """Flag PTX local-memory register spills or high register pressure."""
    records = (diag, prod)
    spill_bytes = _max_metric_across(records, "ptxas_spill_bytes_max")
    triton_spills = _max_metric_across(records, "triton_spills_max")
    lmem_bytes = _max_metric_across(records, "ptx_lmem_bytes_max")
    phys_regs = _max_metric_across(records, "ptxas_regs_max")
    vregs = _max_metric_across(records, "ptx_vregs_max")

    triggered = any(
        (
            spill_bytes is not None and spill_bytes >= thresholds.ptx_lmem_bytes_flag,
            triton_spills is not None and triton_spills >= thresholds.ptx_lmem_bytes_flag,
            lmem_bytes is not None and lmem_bytes >= thresholds.ptx_lmem_bytes_flag,
            phys_regs is not None and phys_regs >= thresholds.ptx_regs_flag,
            phys_regs is None and vregs is not None and vregs >= thresholds.ptx_vregs_flag,
        )
    )
    if not triggered:
        return None

    evidence: dict[str, float | str] = {}
    for key, val in (
        ("ptxas_spill_bytes_max", spill_bytes),
        ("triton_spills_max", triton_spills),
        ("ptx_lmem_bytes_max", lmem_bytes),
        ("ptxas_regs_max", phys_regs),
        ("ptx_vregs_max", vregs),
    ):
        if val is not None:
            evidence[key] = val
    top_name, _ = _top_scope_share(diag)
    return BottleneckFinding(
        category=BottleneckCategory.REGISTER_SPILL_PRESSURE,
        severity="high",
        target_kernel=top_name,
        evidence=evidence,
        summary="Register pressure / local memory spilling detected in compiled PTX/ptxas.",
        playbook_ref=(
            "agent_assets/skills/xtrax-optimizing/references/kernel-optimization-loop.md"
            "#register-spill-pressure"
        ),
        threshold_status="provisional_gpu_heuristic",
    )


def _validate_scaling_unanimity(lo_rec: ProbeRecord, hi_rec: ProbeRecord) -> None:
    """Validate version, stage, SHA, and hardware unanimity for batch scaling."""
    running_major = CONTRACT_VERSION.split(".")[0]
    lo_major = lo_rec.contract_version.split(".")[0]
    hi_major = hi_rec.contract_version.split(".")[0]
    if lo_major != hi_major or lo_major != running_major:
        raise ClaimValidityError("batch_scaling_pair contract_version major mismatch")
    if lo_rec.stage < 1 or hi_rec.stage < 1:
        raise ClaimValidityError("batch_scaling_pair requires stage >= 1")
    if lo_rec.platform == "gpu" and min(lo_rec.stage, hi_rec.stage) < 2:
        raise ClaimValidityError("GPU batch_scaling_pair requires stage >= 2")
    if lo_rec.stage != hi_rec.stage:
        raise ClaimValidityError("batch_scaling_pair stage mismatch")
    _reject_unverifiable_git_sha([lo_rec, hi_rec], ClaimClass.END_TO_END)
    if not _shas_match(lo_rec.git_sha, hi_rec.git_sha):
        raise ClaimValidityError("batch_scaling_pair git_sha mismatch")
    for attr in _BATCH_SCALING_UNANIMITY:
        if getattr(lo_rec, attr) != getattr(hi_rec, attr):
            raise ClaimValidityError(f"batch_scaling_pair mismatch on {attr!r}")
    for rec in (lo_rec, hi_rec):
        dmode = rec.config.get("diagnostic_mode", "none")
        if dmode not in ("", "none"):
            raise ClaimValidityError(f"batch_scaling_pair includes diagnostic record ({dmode!r})")


def _validate_batch_scaling_pair(lo_rec: ProbeRecord, hi_rec: ProbeRecord) -> tuple[float, str]:
    """Validate batch scaling pair invariants and return ``(eta_rep, basis)``."""
    _validate_scaling_unanimity(lo_rec, hi_rec)
    if not (1 <= lo_rec.num_replicas < hi_rec.num_replicas):
        raise ClaimValidityError(
            f"batch_scaling_pair requires 1 <= lo.num_replicas < hi.num_replicas, "
            f"got ({lo_rec.num_replicas}, {hi_rec.num_replicas})"
        )
    lo_basis = lo_rec.config["step_time_basis"] if "step_time_basis" in lo_rec.config else "batch"
    hi_basis = hi_rec.config["step_time_basis"] if "step_time_basis" in hi_rec.config else "batch"
    if lo_basis not in _VALID_BASIS or hi_basis not in _VALID_BASIS or lo_basis != hi_basis:
        raise ClaimValidityError(
            f"batch_scaling_pair invalid step_time_basis: {lo_basis!r} vs {hi_basis!r}"
        )
    if "total_step_seconds" not in lo_rec.metrics or "total_step_seconds" not in hi_rec.metrics:
        raise ClaimValidityError("batch_scaling_pair missing total_step_seconds")
    lo_val = float(lo_rec.metrics["total_step_seconds"])
    hi_val = float(hi_rec.metrics["total_step_seconds"])
    if not (math.isfinite(lo_val) and lo_val > 0.0 and math.isfinite(hi_val) and hi_val > 0.0):
        raise ClaimValidityError("batch_scaling_pair total_step_seconds must be finite > 0.0")

    t_rep_lo = (lo_val / lo_rec.num_replicas) if lo_basis == "batch" else lo_val
    t_rep_hi = (hi_val / hi_rec.num_replicas) if hi_basis == "batch" else hi_val
    return t_rep_lo / t_rep_hi, lo_basis


def _rule_atomic_contention(
    prod: ProbeRecord,
    diag: ProbeRecord | None,
    batch_scaling_pair: tuple[ProbeRecord, ProbeRecord] | None,
    thresholds: TriageThresholds,
) -> BottleneckFinding | None:
    """Flag excessive CTA fan-in onto atomic reduction targets."""
    ctas = _max_metric_across((prod, diag), "ctas_per_atomic_target")
    cta_hit = ctas is not None and ctas >= thresholds.ctas_per_atomic_target_flag
    eta_rep: float | None = None
    basis: str | None = None
    if batch_scaling_pair is not None:
        eta_rep, basis = _validate_batch_scaling_pair(*batch_scaling_pair)
    scale_hit = eta_rep is not None and eta_rep < thresholds.batch_scaling_eff_flag
    if not (cta_hit or scale_hit):
        return None

    evidence: dict[str, float | str] = {}
    if ctas is not None:
        evidence["ctas_per_atomic_target"] = ctas
    if eta_rep is not None:
        evidence["batch_scaling_eff"] = eta_rep
    if basis is not None:
        evidence["step_time_basis"] = basis
    top_name, _ = _top_scope_share(diag)
    return BottleneckFinding(
        category=BottleneckCategory.ATOMIC_CONTENTION,
        severity="high",
        target_kernel=top_name,
        evidence=evidence,
        summary=(
            "Atomic contention or poor multi-replica scaling detected "
            f"(eta_rep={eta_rep if eta_rep is not None else 'n/a'}, "
            f"ctas={ctas if ctas is not None else 'n/a'})."
        ),
        playbook_ref=(
            "agent_assets/skills/xtrax-optimizing/references/kernel-optimization-loop.md"
            "#atomic-contention"
        ),
        threshold_status="provisional_gpu_heuristic",
    )


def _finding_sort_key(finding: BottleneckFinding) -> tuple[int, int]:
    """Sort findings by descending severity and documented rule priority."""
    return (
        _SEVERITY_RANK.get(finding.severity, 99),
        _CATEGORY_ORDER.get(finding.category.value, 99),
    )


def triage_probe_bundle(
    prod: ProbeRecord,
    *,
    diag: ProbeRecord | None = None,
    precision_pair: tuple[ProbeRecord, ProbeRecord] | None = None,
    batch_scaling_pair: tuple[ProbeRecord, ProbeRecord] | None = None,
    thresholds: TriageThresholds | None = None,
) -> list[BottleneckFinding]:
    """Classify and deterministically rank bottlenecks across a ProbeRecord bundle.

    Args:
        prod: Production measurement ``ProbeRecord``.
        diag: Optional paired ``cmdbuf=off`` diagnostic ``ProbeRecord``.
        precision_pair: Optional ``(single, mixed)`` ``ProbeRecord`` pair.
        batch_scaling_pair: Optional ``(low_B, high_B)`` ``ProbeRecord`` pair.
        thresholds: Optional override thresholds.

    Returns:
        Sorted list of :class:`BottleneckFinding` items ordered by severity then
        canonical category order.
    """
    eff_thresholds = thresholds if thresholds is not None else TriageThresholds()
    candidates: list[BottleneckFinding | None] = [
        _rule_recompile_drift(prod),
        _rule_host_boundary_sync(prod, eff_thresholds),
        _rule_fp64_saturation(prod, diag, precision_pair, eff_thresholds),
        _rule_spatial_dead_work(prod, diag, eff_thresholds),
        _rule_register_spill(prod, diag, eff_thresholds),
        _rule_atomic_contention(prod, diag, batch_scaling_pair, eff_thresholds),
    ]
    findings = [item for item in candidates if item is not None]
    findings.sort(key=_finding_sort_key)
    return findings


__all__ = [
    "BottleneckCategory",
    "BottleneckFinding",
    "TriageThresholds",
    "triage_probe_bundle",
]
