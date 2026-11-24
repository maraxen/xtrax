"""Cross-commit ProbeRecord comparison, fidelity gating, and baseline ratchet.

Compares a baseline ``ProbeRecord`` (commit ``sha_A``) against a candidate
``ProbeRecord`` (commit ``sha_B``) while enforcing environment unanimity,
clean-git SHA provenance, fidelity/work-retention invariants, dispatch/compile
ceilings, GPU VRAM budgets, and robust MAD/IQR noise-floor gates.

Zero module-level ``jax`` import and strictly scoped within
``xtrax.profiling.*`` per the profiling leaf-package contract.
"""

import dataclasses
import enum
import hashlib
import json
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xtrax.profiling.claims import CONTRACT_VERSION, ClaimValidityError
from xtrax.profiling.record import ProbeRecord

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{7,40}(-dirty)?$")

_COMPARE_UNANIMITY_FIELDS: tuple[str, ...] = (
    "platform",
    "device_kind",
    "x64_enabled",
    "xla_flags",
    "jax_version",
    "jaxlib_version",
    "n_atoms",
    "num_replicas",
    "replica_id",
    "bucket_index",
    "bucket_capacity",
)

_PARITY_STATUS_KEYS: tuple[str, ...] = (
    "non_degeneracy_status",
    "corruption_control_status",
    "export_guardrail_status",
    "vjp_parity_status",
)

_NONNEG_PARITY_METRICS: tuple[str, ...] = (
    "parity_rel_err",
    "parity_max_abs_err",
    "parity_vjp_rel_err",
    "corruption_rel_err",
)

_VALID_BASIS = frozenset({"batch", "per_replica"})


class VerdictStatus(enum.Enum):
    """Disposition returned by :func:`compare_probe_pair`."""

    KEEP = "keep"
    ADVISORY_ONLY = "advisory_only"
    REVERT_REGRESSION = "revert_regression"
    REVERT_NOISE = "revert_noise"
    REJECT_PARITY = "reject_parity"
    REJECT_DISPATCH = "reject_dispatch"
    REJECT_VRAM = "reject_vram"
    REJECT_UNVERIFIABLE = "reject_unverifiable"


def _validate_optional_nonneg(value: float | None, name: str) -> None:
    """Validate that an optional threshold ``value`` is non-negative."""
    if value is None:
        return
    if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
        raise ClaimValidityError(f"{name} must be finite and >= 0.0, got {value!r}")


@dataclasses.dataclass(frozen=True, slots=True)
class ParityContract:
    """Numeric fidelity, negative-control, and work-retention gate settings.

    Attributes:
        max_rel_err: Maximum allowed ``parity_rel_err`` (``None`` disables).
        max_abs_err: Maximum allowed ``parity_max_abs_err`` (``None`` disables).
        max_vjp_rel_err: Maximum allowed ``parity_vjp_rel_err`` (``None`` disables).
        require_non_degeneracy: Require ``non_degeneracy_status == "pass"`` and
            positive ``output_l2_norm``.
        require_corruption_control: Require an active negative control turning red
            (``corruption_control_status == "pass"`` and error above threshold).
        min_corruption_rel_err: Optional extra lower bound for corruption error.
        require_export_ok: Require ``export_guardrail_status == "pass"`` with zero
            forbidden custom calls and positive artifact size.
        min_work_active_count: Optional absolute floor on ``work_active_count``.
        min_work_retention_ratio: Minimum allowed candidate/baseline
            ``work_active_count`` ratio (default ``0.99``).
        metric_ceilings: Upper bounds on named candidate metrics.
        metric_floors: Lower bounds on named candidate metrics.
    """

    max_rel_err: float | None = 1e-4
    max_abs_err: float | None = None
    max_vjp_rel_err: float | None = None
    require_non_degeneracy: bool = True
    require_corruption_control: bool = True
    min_corruption_rel_err: float | None = None
    require_export_ok: bool = False
    min_work_active_count: float | None = None
    min_work_retention_ratio: float = 0.99
    metric_ceilings: dict[str, float] = dataclasses.field(default_factory=dict)
    metric_floors: dict[str, float] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate parity tolerance fields on construction."""
        _validate_optional_nonneg(self.max_rel_err, "max_rel_err")
        _validate_optional_nonneg(self.max_abs_err, "max_abs_err")
        _validate_optional_nonneg(self.max_vjp_rel_err, "max_vjp_rel_err")
        _validate_optional_nonneg(self.min_corruption_rel_err, "min_corruption_rel_err")
        if self.min_work_active_count is not None:
            w = self.min_work_active_count
            if isinstance(w, bool) or not math.isfinite(w) or w <= 0.0:
                raise ClaimValidityError(
                    f"min_work_active_count must be finite and > 0.0, got {w!r}"
                )
        ratio = self.min_work_retention_ratio
        if isinstance(ratio, bool) or not math.isfinite(ratio) or not (0.0 < ratio <= 1.0):
            raise ClaimValidityError(
                f"min_work_retention_ratio must be finite in (0.0, 1.0], got {ratio!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class DeltaVerdict:
    """Audit-grade result of comparing a baseline and candidate ProbeRecord.

    Attributes:
        status: Classification from :class:`VerdictStatus`.
        promotable: ``True`` solely when ``status is VerdictStatus.KEEP``.
        primary_metric: Metric compared in Step 6.
        baseline_value: Baseline ``primary_metric`` value, if available.
        candidate_value: Candidate ``primary_metric`` value, if available.
        speedup_x: Ratio of improvement (> 1.0 means faster/better).
        rel_delta_pct: Signed candidate-minus-baseline percentage delta.
        noise_floor_x: Required relative speedup floor from MAD/IQR dispersion.
        baseline_probe_id: ``probe_id`` of the baseline record.
        candidate_probe_id: ``probe_id`` of the candidate record.
        baseline_sha: ``git_sha`` of the baseline record.
        candidate_sha: ``git_sha`` of the candidate record.
        reasons: Ordered tuple of gate/advisory diagnostics.
        metric_deltas: Signed ``candidate - baseline`` deltas across shared
            finite metrics in native units.
    """

    status: VerdictStatus
    promotable: bool
    primary_metric: str
    baseline_value: float | None
    candidate_value: float | None
    speedup_x: float | None
    rel_delta_pct: float | None
    noise_floor_x: float | None
    baseline_probe_id: str
    candidate_probe_id: str
    baseline_sha: str
    candidate_sha: str
    reasons: tuple[str, ...]
    metric_deltas: dict[str, float]


def _split_sha(sha: str) -> tuple[str, bool]:
    """Return ``(lowercase_hex_part, is_dirty)`` for a git SHA token."""
    cleaned = sha.strip().lower()
    if cleaned.endswith("-dirty"):
        return cleaned[: -len("-dirty")], True
    return cleaned, False


def _shas_match(sha_a: str, sha_b: str) -> bool:
    """Return ``True`` when two git SHAs identify the same commit and clean state."""
    hex_a, dirty_a = _split_sha(sha_a)
    hex_b, dirty_b = _split_sha(sha_b)
    if not hex_a or not hex_b or dirty_a != dirty_b:
        return False
    if hex_a == hex_b:
        return True
    shorter, longer = (hex_a, hex_b) if len(hex_a) <= len(hex_b) else (hex_b, hex_a)
    if len(shorter) < 7 or len(longer) != 40:
        return False
    return longer.startswith(shorter)


def _compute_metric_deltas(baseline: ProbeRecord, candidate: ProbeRecord) -> dict[str, float]:
    """Compute signed candidate minus baseline deltas for shared finite metrics."""
    deltas: dict[str, float] = {}
    for key in sorted(set(baseline.metrics) & set(candidate.metrics)):
        diff = float(candidate.metrics[key]) - float(baseline.metrics[key])
        if math.isfinite(diff):
            deltas[key] = diff
    return deltas


def _make_verdict(
    status: VerdictStatus,
    baseline: ProbeRecord,
    candidate: ProbeRecord,
    primary_metric: str,
    reasons: list[str],
    metric_deltas: dict[str, float],
    *,
    baseline_value: float | None = None,
    candidate_value: float | None = None,
    speedup_x: float | None = None,
    rel_delta_pct: float | None = None,
    noise_floor_x: float | None = None,
) -> DeltaVerdict:
    """Construct an immutable :class:`DeltaVerdict` with derived promotability."""
    promotable = status is VerdictStatus.KEEP
    return DeltaVerdict(
        status=status,
        promotable=promotable,
        primary_metric=primary_metric,
        baseline_value=baseline_value,
        candidate_value=candidate_value,
        speedup_x=speedup_x,
        rel_delta_pct=rel_delta_pct,
        noise_floor_x=noise_floor_x,
        baseline_probe_id=baseline.probe_id,
        candidate_probe_id=candidate.probe_id,
        baseline_sha=baseline.git_sha,
        candidate_sha=candidate.git_sha,
        reasons=tuple(reasons),
        metric_deltas=metric_deltas,
    )


def _validate_compare_params(
    *,
    require_parity: bool,
    parity: ParityContract | None,
    min_speedup: float,
    noise_sigma: float,
    max_compile_increase: float,
    max_dispatch_growth: float,
    max_vram_growth: float | None,
) -> None:
    """Validate :func:`compare_probe_pair` arguments (Step 0)."""
    if not require_parity and parity is not None:
        raise ClaimValidityError("parity cannot be passed when require_parity=False")
    if isinstance(min_speedup, bool) or not math.isfinite(min_speedup) or min_speedup < 1.0:
        raise ClaimValidityError(f"min_speedup must be finite and >= 1.0, got {min_speedup!r}")
    if isinstance(noise_sigma, bool) or not math.isfinite(noise_sigma) or noise_sigma < 0.0:
        raise ClaimValidityError(f"noise_sigma must be finite and >= 0.0, got {noise_sigma!r}")
    if (
        isinstance(max_compile_increase, bool)
        or not math.isfinite(max_compile_increase)
        or max_compile_increase < 0.0
    ):
        raise ClaimValidityError(
            f"max_compile_increase must be finite and >= 0.0, got {max_compile_increase!r}"
        )
    if (
        isinstance(max_dispatch_growth, bool)
        or not math.isfinite(max_dispatch_growth)
        or max_dispatch_growth < 1.0
    ):
        raise ClaimValidityError(
            f"max_dispatch_growth must be finite and >= 1.0, got {max_dispatch_growth!r}"
        )
    if max_vram_growth is not None and (
        isinstance(max_vram_growth, bool)
        or not math.isfinite(max_vram_growth)
        or max_vram_growth < 1.0
    ):
        raise ClaimValidityError(
            f"max_vram_growth must be finite and >= 1.0, got {max_vram_growth!r}"
        )


def _check_step_time_basis(
    baseline: ProbeRecord,
    candidate: ProbeRecord,
    primary_metric: str,
) -> str | None:
    """Validate matching ``step_time_basis`` when timing metrics or config tags are present."""
    basis_needed = (
        primary_metric == "total_step_seconds"
        or "step_time_basis" in baseline.config
        or "step_time_basis" in candidate.config
    )
    if not basis_needed:
        return None
    b_basis = (
        baseline.config["step_time_basis"] if "step_time_basis" in baseline.config else "batch"
    )
    c_basis = (
        candidate.config["step_time_basis"] if "step_time_basis" in candidate.config else "batch"
    )
    if b_basis not in _VALID_BASIS or c_basis not in _VALID_BASIS:
        return (
            f"invalid step_time_basis: baseline={b_basis!r}, candidate={c_basis!r} "
            f"(expected one of {sorted(_VALID_BASIS)})"
        )
    if b_basis != c_basis:
        return f"step_time_basis mismatch: baseline={b_basis!r} != candidate={c_basis!r}"
    return None


def _check_stage_and_diagnostic(baseline: ProbeRecord, candidate: ProbeRecord) -> str | None:
    """Verify comparison stages and ensure neither record is a diagnostic pass."""
    if baseline.stage < 1 or candidate.stage < 1:
        return f"stage >= 1 required (got baseline={baseline.stage}, candidate={candidate.stage})"
    if baseline.platform == "gpu" and min(baseline.stage, candidate.stage) < 2:
        return (
            f"GPU comparison requires stage >= 2 "
            f"(got baseline={baseline.stage}, candidate={candidate.stage})"
        )
    if baseline.stage != candidate.stage:
        return f"stage mismatch: baseline={baseline.stage} != candidate={candidate.stage}"
    for label, rec in (("baseline", baseline), ("candidate", candidate)):
        dmode = rec.config.get("diagnostic_mode", "none")
        if dmode not in ("", "none"):
            return f"{label} record is diagnostic (diagnostic_mode={dmode!r}); cannot compare"
    return None


def _check_hold_fixed(
    baseline: ProbeRecord,
    candidate: ProbeRecord,
    hold_fixed_config: tuple[str, ...],
) -> str | None:
    """Verify every required ``hold_fixed_config`` key matches identically."""
    for key in hold_fixed_config:
        if key not in baseline.config or key not in candidate.config:
            return f"hold_fixed_config key {key!r} missing from baseline or candidate config"
        if baseline.config[key] != candidate.config[key]:
            return (
                f"hold_fixed_config mismatch on {key!r}: "
                f"baseline={baseline.config[key]!r} != candidate={candidate.config[key]!r}"
            )
    return None


def _check_contract_and_env(
    baseline: ProbeRecord,
    candidate: ProbeRecord,
    *,
    hold_fixed_config: tuple[str, ...],
    primary_metric: str,
) -> str | None:
    """Verify contract version, environment unanimity, stage, and basis (Step 1)."""
    base_major = baseline.contract_version.split(".")[0]
    cand_major = candidate.contract_version.split(".")[0]
    running_major = CONTRACT_VERSION.split(".")[0]
    if base_major != cand_major or base_major != running_major:
        return (
            f"contract_version major mismatch: baseline={baseline.contract_version!r}, "
            f"candidate={candidate.contract_version!r}, running={CONTRACT_VERSION!r}"
        )

    for attr in _COMPARE_UNANIMITY_FIELDS:
        bval, cval = getattr(baseline, attr), getattr(candidate, attr)
        if bval != cval:
            return f"environment mismatch on {attr!r}: baseline={bval!r} != candidate={cval!r}"

    return (
        _check_stage_and_diagnostic(baseline, candidate)
        or _check_step_time_basis(baseline, candidate, primary_metric)
        or _check_hold_fixed(baseline, candidate, hold_fixed_config)
    )


def _check_git_provenance(
    baseline: ProbeRecord,
    candidate: ProbeRecord,
    *,
    allow_dirty: bool,
) -> tuple[bool, str | None, str | None]:
    """Validate clean/dirty git SHAs (Step 2) -> ``(dirty_capped, reject_reason, note)``."""
    for label, sha in (("baseline", baseline.git_sha), ("candidate", candidate.git_sha)):
        if not _GIT_SHA_RE.fullmatch(sha):
            return False, f"{label} git_sha {sha!r} failed strict hex/dirty verification", None
    any_dirty = baseline.git_sha.endswith("-dirty") or candidate.git_sha.endswith("-dirty")
    if any_dirty and not allow_dirty:
        return (
            False,
            f"dirty git_sha rejected with allow_dirty=False "
            f"(baseline={baseline.git_sha!r}, candidate={candidate.git_sha!r})",
            None,
        )
    if any_dirty:
        return True, None, "dirty_git_sha_advisory_only"
    return False, None, None


def _check_present_failure_witnesses(
    baseline: ProbeRecord,
    candidate: ProbeRecord,
    min_work_retention_ratio: float,
) -> tuple[VerdictStatus, str] | None:
    """Fail closed on explicit defect witnesses even when ``require_parity=False``."""
    for key in _PARITY_STATUS_KEYS:
        val = candidate.config.get(key)
        if val is not None and (val == "fail" or val not in ("pass", "not_measured")):
            return VerdictStatus.REJECT_PARITY, f"explicit parity witness failed: {key}={val!r}"
    if "output_l2_norm" in candidate.metrics and float(candidate.metrics["output_l2_norm"]) <= 0.0:
        return VerdictStatus.REJECT_PARITY, "candidate output_l2_norm <= 0.0 (degenerate output)"
    return _check_present_work_and_sign_witnesses(baseline, candidate, min_work_retention_ratio)


def _check_present_work_and_sign_witnesses(
    baseline: ProbeRecord,
    candidate: ProbeRecord,
    min_work_retention_ratio: float,
) -> tuple[VerdictStatus, str] | None:
    """Check present work counts and non-negative parity error witnesses."""
    base_work = baseline.metrics.get("work_active_count")
    cand_work = candidate.metrics.get("work_active_count")
    if base_work is not None and float(base_work) <= 0.0:
        return VerdictStatus.REJECT_UNVERIFIABLE, "baseline work_active_count <= 0.0"
    if cand_work is not None and float(cand_work) <= 0.0:
        return VerdictStatus.REJECT_PARITY, "candidate work_active_count <= 0.0"
    if base_work is not None and cand_work is not None:
        ratio = float(cand_work) / float(base_work)
        if ratio < min_work_retention_ratio:
            return (
                VerdictStatus.REJECT_PARITY,
                f"work_active_count dropped: {float(cand_work):.4g} / {float(base_work):.4g} = "
                f"{ratio:.4f} < {min_work_retention_ratio:.4f}",
            )
    for key in _NONNEG_PARITY_METRICS:
        val = candidate.metrics.get(key)
        if val is not None and float(val) < 0.0:
            return VerdictStatus.REJECT_PARITY, f"negative candidate parity metric {key}={val!r}"
    return None


def _check_scalar_parity_bounds(
    candidate: ProbeRecord,
    eff_parity: ParityContract,
) -> str | None:
    """Check scalar parity tolerances and VJP parity status."""
    bounds: tuple[tuple[str, float | None], ...] = (
        ("parity_rel_err", eff_parity.max_rel_err),
        ("parity_max_abs_err", eff_parity.max_abs_err),
        ("parity_vjp_rel_err", eff_parity.max_vjp_rel_err),
    )
    for key, ceiling in bounds:
        if ceiling is None:
            continue
        if key not in candidate.metrics:
            return f"required parity metric {key!r} missing from candidate"
        val = float(candidate.metrics[key])
        if not (0.0 <= val <= ceiling):
            return f"{key}={val:.6g} exceeds limit {ceiling:.6g}"
    vjp_status = candidate.config.get("vjp_parity_status")
    if vjp_status is not None and vjp_status != "pass":
        return f"vjp_parity_status={vjp_status!r} != 'pass'"
    return None


def _check_work_retention(
    baseline: ProbeRecord,
    candidate: ProbeRecord,
    eff_parity: ParityContract,
) -> str | None:
    """Check active-work retention ratio and absolute minimum floor."""
    if "work_active_count" in baseline.metrics:
        if "work_active_count" not in candidate.metrics:
            return "baseline records work_active_count, but candidate omitted it"
        ratio = float(candidate.metrics["work_active_count"]) / float(
            baseline.metrics["work_active_count"]
        )
        if ratio < eff_parity.min_work_retention_ratio:
            return (
                f"work_active_count retention {ratio:.4f} < "
                f"{eff_parity.min_work_retention_ratio:.4f}"
            )
    if eff_parity.min_work_active_count is not None:
        if "work_active_count" not in candidate.metrics:
            return "work_active_count missing for min_work_active_count check"
        cand_w = float(candidate.metrics["work_active_count"])
        if cand_w < eff_parity.min_work_active_count:
            return f"work_active_count {cand_w:.4g} < {eff_parity.min_work_active_count:.4g}"
    return None


def _check_custom_bounds(
    candidate: ProbeRecord,
    eff_parity: ParityContract,
) -> str | None:
    """Check user-declared ``metric_ceilings`` and ``metric_floors``."""
    for key, bound in eff_parity.metric_ceilings.items():
        if key not in candidate.metrics:
            return f"metric_ceilings key {key!r} missing from candidate"
        if float(candidate.metrics[key]) > bound:
            return f"{key}={candidate.metrics[key]:.6g} > ceiling {bound:.6g}"
    for key, bound in eff_parity.metric_floors.items():
        if key not in candidate.metrics:
            return f"metric_floors key {key!r} missing from candidate"
        if float(candidate.metrics[key]) < bound:
            return f"{key}={candidate.metrics[key]:.6g} < floor {bound:.6g}"
    return None


def _check_parity_thresholds_and_work(
    baseline: ProbeRecord,
    candidate: ProbeRecord,
    eff_parity: ParityContract,
) -> str | None:
    """Validate scalar errors, VJP parity, work retention, and custom bounds."""
    return (
        _check_scalar_parity_bounds(candidate, eff_parity)
        or _check_work_retention(baseline, candidate, eff_parity)
        or _check_custom_bounds(candidate, eff_parity)
    )


def _check_corruption_control(
    candidate: ProbeRecord,
    eff_parity: ParityContract,
) -> tuple[str | None, str | None]:
    """Return ``(error_msg, note)`` for negative-control red verification."""
    corr_status = candidate.config.get("corruption_control_status")
    has_corr_err = "corruption_rel_err" in candidate.metrics
    if not (eff_parity.require_corruption_control or corr_status == "pass" or has_corr_err):
        return None, "corruption_control_not_required"
    corr_thr = max(eff_parity.max_rel_err or 0.0, eff_parity.min_corruption_rel_err or 0.0)
    corr_rel = candidate.metrics.get("corruption_rel_err")
    corr_int = float(candidate.metrics.get("corruption_int_mismatches", 0.0))
    rel_ok = corr_rel is not None and float(corr_rel) > corr_thr
    if corr_status != "pass" or not (rel_ok or corr_int >= 1.0):
        return (
            f"corruption control failed (status={corr_status!r}, "
            f"corruption_rel_err={corr_rel!r}, threshold={corr_thr!r}, "
            f"corruption_int_mismatches={corr_int!r})",
            None,
        )
    return None, None


def _check_non_degen_and_export(
    candidate: ProbeRecord,
    eff_parity: ParityContract,
) -> str | None:
    """Verify non-degeneracy and optional StableHLO export guardrails."""
    if eff_parity.require_non_degeneracy:
        if candidate.config.get("non_degeneracy_status") != "pass":
            return "non_degeneracy_status != 'pass'"
        norm = candidate.metrics.get("output_l2_norm")
        if norm is None or float(norm) <= 0.0:
            return "output_l2_norm missing or <= 0.0"
    if eff_parity.require_export_ok:
        exp_status = candidate.config.get("export_guardrail_status")
        cc_count = candidate.metrics.get("export_custom_call_count")
        exp_bytes = float(candidate.metrics.get("export_artifact_bytes", 0.0))
        if exp_status != "pass" or cc_count != 0.0 or exp_bytes <= 0.0:
            return (
                f"export guardrail failed (status={exp_status!r}, "
                f"custom_call_count={cc_count!r}, artifact_bytes={exp_bytes!r})"
            )
    return None


def _evaluate_parity_gate(
    baseline: ProbeRecord,
    candidate: ProbeRecord,
    *,
    require_parity: bool,
    parity: ParityContract | None,
) -> tuple[bool, VerdictStatus | None, list[str]]:
    """Run Step 3 parity, non-degeneracy, corruption, export, and work gates."""
    eff_parity = parity if parity is not None else ParityContract()
    witness_err = _check_present_failure_witnesses(
        baseline, candidate, eff_parity.min_work_retention_ratio
    )
    if witness_err is not None:
        return False, witness_err[0], [witness_err[1]]
    if not require_parity:
        return True, None, ["parity_gate_bypassed"]

    notes: list[str] = []
    degen_or_exp = _check_non_degen_and_export(candidate, eff_parity)
    if degen_or_exp is not None:
        return False, VerdictStatus.REJECT_PARITY, [degen_or_exp]

    corr_err, corr_note = _check_corruption_control(candidate, eff_parity)
    if corr_note:
        notes.append(corr_note)
    if corr_err is not None:
        return False, VerdictStatus.REJECT_PARITY, [corr_err]

    err = _check_parity_thresholds_and_work(baseline, candidate, eff_parity)
    if err is not None:
        return False, VerdictStatus.REJECT_PARITY, [err]
    return False, None, notes


def _check_compile_ceiling(
    base_compiles: float,
    cand_compiles: float,
    max_compile_increase: float,
) -> tuple[VerdictStatus, str] | None:
    """Reject candidates with unexpected compilation overhead."""
    if base_compiles == 0.0:
        if cand_compiles > max_compile_increase:
            return (
                VerdictStatus.REJECT_DISPATCH,
                f"candidate n_compilations={cand_compiles:.4g} > "
                f"max_compile_increase={max_compile_increase:.4g}",
            )
        return None
    allowed = max_compile_increase > 0.0 and cand_compiles <= base_compiles + max_compile_increase
    if cand_compiles > 0.0 and not allowed:
        return (
            VerdictStatus.REJECT_DISPATCH,
            f"candidate n_compilations={cand_compiles:.4g} > 0 with baseline="
            f"{base_compiles:.4g} and max_compile_increase={max_compile_increase:.4g}",
        )
    return None


def _check_dispatch_growth(
    base_exec: float,
    cand_exec: float,
    max_dispatch_growth: float,
) -> tuple[VerdictStatus, str] | None:
    """Reject candidates whose execution count grows past ``max_dispatch_growth``."""
    growth = cand_exec / base_exec
    if growth > max_dispatch_growth:
        return (
            VerdictStatus.REJECT_DISPATCH,
            f"dispatch growth {cand_exec:.4g} / {base_exec:.4g} = "
            f"{growth:.4f}x > {max_dispatch_growth:.4f}x",
        )
    return None


def _check_dispatch_and_compile(
    baseline: ProbeRecord,
    candidate: ProbeRecord,
    *,
    max_compile_increase: float,
    max_dispatch_growth: float,
) -> tuple[VerdictStatus, str] | None:
    """Enforce zero steady-recompile and bounded dispatch growth (Step 4)."""
    for label, rec in (("baseline", baseline), ("candidate", candidate)):
        if "n_compilations" not in rec.metrics or float(rec.metrics["n_compilations"]) < 0.0:
            return (
                VerdictStatus.REJECT_UNVERIFIABLE,
                f"{label} missing valid n_compilations >= 0.0",
            )
        if "n_executions" not in rec.metrics or float(rec.metrics["n_executions"]) <= 0.0:
            return (
                VerdictStatus.REJECT_UNVERIFIABLE,
                f"{label} missing valid n_executions > 0.0",
            )

    compile_err = _check_compile_ceiling(
        float(baseline.metrics["n_compilations"]),
        float(candidate.metrics["n_compilations"]),
        max_compile_increase,
    )
    if compile_err is not None:
        return compile_err
    return _check_dispatch_growth(
        float(baseline.metrics["n_executions"]),
        float(candidate.metrics["n_executions"]),
        max_dispatch_growth,
    )


def _check_vram_gate(
    baseline: ProbeRecord,
    candidate: ProbeRecord,
    *,
    max_vram_growth: float | None,
) -> tuple[VerdictStatus | None, str | None]:
    """Enforce VRAM growth ceiling cleanly across GPU and CPU records (Step 5)."""
    if max_vram_growth is None:
        return None, None
    b_has = "peak_vram_bytes" in baseline.metrics
    c_has = "peak_vram_bytes" in candidate.metrics
    if baseline.platform == "gpu":
        if not b_has or not c_has:
            return (
                VerdictStatus.REJECT_UNVERIFIABLE,
                "GPU comparison requires peak_vram_bytes on both records",
            )
    else:
        if b_has != c_has:
            return (
                VerdictStatus.REJECT_UNVERIFIABLE,
                "CPU peak_vram_bytes present on only one record",
            )
        if not b_has and not c_has:
            return None, "cpu_peak_vram_bytes_absent_skipped"

    b_vram = float(baseline.metrics["peak_vram_bytes"])
    c_vram = float(candidate.metrics["peak_vram_bytes"])
    if b_vram <= 0.0 or c_vram <= 0.0:
        return (
            VerdictStatus.REJECT_UNVERIFIABLE,
            f"peak_vram_bytes must be > 0 (got {b_vram}, {c_vram})",
        )
    ratio = c_vram / b_vram
    if ratio > max_vram_growth:
        return (
            VerdictStatus.REJECT_VRAM,
            f"peak VRAM growth {ratio:.4f}x > {max_vram_growth:.4f}x "
            f"({c_vram:.0f} vs {b_vram:.0f} B)",
        )
    return None, None


def _validate_dispersion_nonneg(baseline: ProbeRecord, candidate: ProbeRecord) -> str | None:
    """Ensure any reported wall dispersion metrics are non-negative."""
    for label, rec in (("baseline", baseline), ("candidate", candidate)):
        for key in ("wall_mad_seconds", "wall_iqr_seconds"):
            if key in rec.metrics and float(rec.metrics[key]) < 0.0:
                return f"{label} negative dispersion {key}={rec.metrics[key]!r}"
    return None


def _paired_dispersion_ratio(
    baseline: ProbeRecord,
    candidate: ProbeRecord,
    key: str,
    base_val: float,
    cand_val: float,
) -> float | None:
    """Return paired relative quadrature dispersion when ``key`` exists in both records."""
    b_raw = baseline.metrics.get(key)
    c_raw = candidate.metrics.get(key)
    if b_raw is None or c_raw is None:
        return None
    bf, cf = float(b_raw), float(c_raw)
    if bf > 0.0 or cf > 0.0:
        return math.hypot(bf / base_val, cf / cand_val)
    return 0.0


def _compute_rel_noise(
    baseline: ProbeRecord,
    candidate: ProbeRecord,
    *,
    primary_metric: str,
    base_val: float,
    cand_val: float,
    noise_sigma: float,
) -> tuple[float | None, str | None, str | None]:
    """Return ``(rel_noise, unverifiable_error, advisory_note)`` for Step 6."""
    neg_err = _validate_dispersion_nonneg(baseline, candidate)
    if neg_err is not None:
        return None, neg_err, None
    if primary_metric != "total_step_seconds":
        return 0.0, None, None

    mad_ratio = _paired_dispersion_ratio(
        baseline, candidate, "wall_mad_seconds", base_val, cand_val
    )
    if mad_ratio is not None:
        return (noise_sigma * 1.4826 * mad_ratio if mad_ratio > 0.0 else 0.0), None, None

    iqr_ratio = _paired_dispersion_ratio(
        baseline, candidate, "wall_iqr_seconds", base_val, cand_val
    )
    if iqr_ratio is not None:
        return (noise_sigma * 0.7413 * iqr_ratio if iqr_ratio > 0.0 else 0.0), None, None

    return 0.0, None, "wall_dispersion_absent_using_min_speedup_floor"


def _classify_performance(
    *,
    base_val: float,
    cand_val: float,
    lower_is_better: bool,
    min_speedup: float,
    rel_noise: float,
    dirty_capped: bool,
    parity_bypassed: bool,
) -> tuple[VerdictStatus, float, float, float, str | None]:
    """Evaluate speedup against noise floor and promotion caps (Step 6)."""
    speedup_x = (base_val / cand_val) if lower_is_better else (cand_val / base_val)
    rel_delta_pct = (cand_val - base_val) / base_val * 100.0
    noise_floor_x = 1.0 + rel_noise
    required_speedup = max(min_speedup, noise_floor_x)
    reg_floor = max(0.0, 1.0 - rel_noise)

    if speedup_x < reg_floor:
        return (
            VerdictStatus.REVERT_REGRESSION,
            speedup_x,
            rel_delta_pct,
            noise_floor_x,
            f"regression: speedup_x={speedup_x:.4f} < {reg_floor:.4f}",
        )
    if speedup_x < required_speedup:
        return (
            VerdictStatus.REVERT_NOISE,
            speedup_x,
            rel_delta_pct,
            noise_floor_x,
            f"within noise/min floor: speedup_x={speedup_x:.4f} < {required_speedup:.4f}",
        )
    if dirty_capped or parity_bypassed:
        return VerdictStatus.ADVISORY_ONLY, speedup_x, rel_delta_pct, noise_floor_x, None
    return (
        VerdictStatus.KEEP,
        speedup_x,
        rel_delta_pct,
        noise_floor_x,
        f"promoted: speedup_x={speedup_x:.4f} >= {required_speedup:.4f}",
    )


def _run_resource_gates(
    baseline: ProbeRecord,
    candidate: ProbeRecord,
    primary_metric: str,
    metric_deltas: dict[str, float],
    reasons: list[str],
    *,
    max_compile_increase: float = 0.0,
    max_dispatch_growth: float = 1.05,
    max_vram_growth: float | None = 1.10,
) -> DeltaVerdict | None:
    """Enforce dispatch, compile, and VRAM budget gates before performance comparison."""
    disp_err = _check_dispatch_and_compile(
        baseline,
        candidate,
        max_compile_increase=max_compile_increase,
        max_dispatch_growth=max_dispatch_growth,
    )
    if disp_err is not None:
        return _make_verdict(
            disp_err[0],
            baseline,
            candidate,
            primary_metric,
            [*reasons, disp_err[1]],
            metric_deltas,
        )

    vram_status, vram_msg = _check_vram_gate(baseline, candidate, max_vram_growth=max_vram_growth)
    if vram_msg:
        reasons.append(vram_msg)
    if vram_status is not None:
        return _make_verdict(
            vram_status,
            baseline,
            candidate,
            primary_metric,
            reasons,
            metric_deltas,
        )
    return None


def _extract_primary_pair(
    baseline: ProbeRecord,
    candidate: ProbeRecord,
    primary_metric: str,
) -> tuple[float | None, float | None, str | None]:
    """Extract and validate strictly positive primary metric values from both records."""
    raw_b = baseline.metrics.get(primary_metric)
    raw_c = candidate.metrics.get(primary_metric)
    base_val = float(raw_b) if raw_b is not None else None
    cand_val = float(raw_c) if raw_c is not None else None
    if base_val is None or cand_val is None or base_val <= 0.0 or cand_val <= 0.0:
        return (
            base_val,
            cand_val,
            f"primary_metric {primary_metric!r} must be present and > 0.0 in both records",
        )
    return base_val, cand_val, None


def compare_probe_pair(
    baseline: ProbeRecord,
    candidate: ProbeRecord,
    *,
    hold_fixed_config: tuple[str, ...],
    primary_metric: str = "total_step_seconds",
    lower_is_better: bool = True,
    parity: ParityContract | None = None,
    require_parity: bool = True,
    min_speedup: float = 1.03,
    noise_sigma: float = 2.0,
    max_vram_growth: float | None = 1.10,
    max_compile_increase: float = 0.0,
    max_dispatch_growth: float = 1.05,
    allow_dirty: bool = False,
) -> DeltaVerdict:
    """Compare a baseline and candidate ProbeRecord across commits fail-closed.

    Args:
        baseline: Previously measured reference record (``sha_A``).
        candidate: New kernel-edit record (``sha_B``).
        hold_fixed_config: ``ProbeRecord.config`` keys required to match identically.
        primary_metric: Float metric in ``ProbeRecord.metrics`` to compare.
        lower_is_better: True for latencies/bytes; False for throughput metrics.
        parity: Explicit :class:`ParityContract` (must be ``None`` when
            ``require_parity=False``).
        require_parity: Enforce non-degeneracy, negative controls, and tolerances.
        min_speedup: Minimum relative improvement ratio (``>= 1.0``).
        noise_sigma: Multiplier on propagated relative MAD/IQR uncertainty.
        max_vram_growth: Maximum allowed ``peak_vram_bytes`` ratio (``None`` skips).
        max_compile_increase: Maximum steady-compile increase (default ``0.0``).
        max_dispatch_growth: Maximum ``n_executions`` ratio (default ``1.05``).
        allow_dirty: Permit ``-dirty`` git SHAs (still capped at ``ADVISORY_ONLY``).

    Returns:
        A :class:`DeltaVerdict` whose ``promotable`` flag is ``True`` only when
        ``status is VerdictStatus.KEEP``.

    Raises:
        ClaimValidityError: On invalid comparison parameter combinations.
    """
    _validate_compare_params(
        require_parity=require_parity,
        parity=parity,
        min_speedup=min_speedup,
        noise_sigma=noise_sigma,
        max_compile_increase=max_compile_increase,
        max_dispatch_growth=max_dispatch_growth,
        max_vram_growth=max_vram_growth,
    )
    metric_deltas = _compute_metric_deltas(baseline, candidate)

    env_err = _check_contract_and_env(
        baseline,
        candidate,
        hold_fixed_config=hold_fixed_config,
        primary_metric=primary_metric,
    )
    if env_err is not None:
        return _make_verdict(
            VerdictStatus.REJECT_UNVERIFIABLE,
            baseline,
            candidate,
            primary_metric,
            [env_err],
            metric_deltas,
        )

    dirty_capped, sha_err, sha_note = _check_git_provenance(
        baseline, candidate, allow_dirty=allow_dirty
    )
    if sha_err is not None:
        return _make_verdict(
            VerdictStatus.REJECT_UNVERIFIABLE,
            baseline,
            candidate,
            primary_metric,
            [sha_err],
            metric_deltas,
        )

    reasons: list[str] = [sha_note] if sha_note else []
    parity_bypassed, parity_status, parity_notes = _evaluate_parity_gate(
        baseline,
        candidate,
        require_parity=require_parity,
        parity=parity,
    )
    reasons.extend(parity_notes)
    if parity_status is not None:
        return _make_verdict(
            parity_status,
            baseline,
            candidate,
            primary_metric,
            reasons,
            metric_deltas,
        )

    resource_verdict = _run_resource_gates(
        baseline,
        candidate,
        primary_metric,
        metric_deltas,
        reasons,
        max_compile_increase=max_compile_increase,
        max_dispatch_growth=max_dispatch_growth,
        max_vram_growth=max_vram_growth,
    )
    if resource_verdict is not None:
        return resource_verdict

    base_val, cand_val, pair_err = _extract_primary_pair(baseline, candidate, primary_metric)
    if pair_err is not None or base_val is None or cand_val is None:
        return _make_verdict(
            VerdictStatus.REJECT_UNVERIFIABLE,
            baseline,
            candidate,
            primary_metric,
            [*reasons, pair_err or "invalid primary metric pair"],
            metric_deltas,
            baseline_value=base_val,
            candidate_value=cand_val,
        )

    rel_noise, disp_unverif, disp_note = _compute_rel_noise(
        baseline,
        candidate,
        primary_metric=primary_metric,
        base_val=base_val,
        cand_val=cand_val,
        noise_sigma=noise_sigma,
    )
    if disp_unverif is not None or rel_noise is None:
        return _make_verdict(
            VerdictStatus.REJECT_UNVERIFIABLE,
            baseline,
            candidate,
            primary_metric,
            [*reasons, disp_unverif or "invalid dispersion"],
            metric_deltas,
            baseline_value=base_val,
            candidate_value=cand_val,
        )
    if disp_note:
        reasons.append(disp_note)

    status, speedup_x, rel_delta_pct, noise_floor_x, perf_note = _classify_performance(
        base_val=base_val,
        cand_val=cand_val,
        lower_is_better=lower_is_better,
        min_speedup=min_speedup,
        rel_noise=rel_noise,
        dirty_capped=dirty_capped,
        parity_bypassed=parity_bypassed,
    )
    if perf_note:
        reasons.append(perf_note)

    return _make_verdict(
        status,
        baseline,
        candidate,
        primary_metric,
        reasons,
        metric_deltas,
        baseline_value=base_val,
        candidate_value=cand_val,
        speedup_x=speedup_x,
        rel_delta_pct=rel_delta_pct,
        noise_floor_x=noise_floor_x,
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write ``payload`` as indented JSON to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_ratchet_pointer(
    ptr: Path,
    verdict: DeltaVerdict,
    cand_path: Path,
    cand_rec: ProbeRecord,
    *,
    initialize: bool = False,
) -> Path:
    """Validate ratchet chain continuity and atomically advance ``ptr``."""
    if ptr.exists() and not initialize:
        try:
            prev = json.loads(ptr.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ClaimValidityError(f"existing pointer {ptr} is unreadable: {exc}") from exc
        if not isinstance(prev, dict):
            raise ClaimValidityError(f"existing pointer {ptr} must be a JSON object")
        if verdict.baseline_probe_id != prev.get("probe_id") or not _shas_match(
            verdict.baseline_sha, str(prev.get("git_sha", ""))
        ):
            raise ClaimValidityError(
                "ratchet chain continuity broken: verdict baseline "
                f"({verdict.baseline_probe_id!r}, {verdict.baseline_sha!r}) != pointer "
                f"({prev.get('probe_id')!r}, {prev.get('git_sha')!r})"
            )
    payload = {
        "baseline_record": str(cand_path),
        "git_sha": cand_rec.git_sha,
        "probe_id": cand_rec.probe_id,
        "updated_at": datetime.now(UTC).isoformat(),
        "speedup_x": verdict.speedup_x,
        "previous_sha": verdict.baseline_sha,
        "previous_probe_id": verdict.baseline_probe_id,
    }
    _atomic_write_json(ptr, payload)
    return ptr


def _write_rejected_sidecar(
    rejected_dir: str | Path | None,
    verdict: DeltaVerdict,
    cand_path: Path,
) -> Path:
    """Persist a non-promotable ``.verdict.json`` sidecar under ``rejected_dir``."""
    if rejected_dir is None:
        raise ClaimValidityError(
            f"rejected_dir is required to record non-promotable verdict {verdict.status.value!r}"
        )
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", verdict.candidate_probe_id).strip("_") or "probe"
    sha7 = _split_sha(verdict.candidate_sha)[0][:7] or "nosha"
    digest_src = (
        f"{cand_path}:{verdict.baseline_probe_id}:{verdict.baseline_sha}:"
        f"{verdict.candidate_probe_id}:{verdict.candidate_sha}:"
        f"{verdict.primary_metric}:{verdict.status.value}"
    )
    digest8 = hashlib.sha256(digest_src.encode("utf-8")).hexdigest()[:8]
    out_path = Path(rejected_dir) / f"{slug}_{sha7}_{digest8}.verdict.json"
    sidecar = {
        "status": verdict.status.value,
        "promotable": verdict.promotable,
        "primary_metric": verdict.primary_metric,
        "baseline_value": verdict.baseline_value,
        "candidate_value": verdict.candidate_value,
        "speedup_x": verdict.speedup_x,
        "rel_delta_pct": verdict.rel_delta_pct,
        "noise_floor_x": verdict.noise_floor_x,
        "baseline_probe_id": verdict.baseline_probe_id,
        "candidate_probe_id": verdict.candidate_probe_id,
        "baseline_sha": verdict.baseline_sha,
        "candidate_sha": verdict.candidate_sha,
        "candidate_record": str(cand_path),
        "reasons": list(verdict.reasons),
        "metric_deltas": dict(verdict.metric_deltas),
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    _atomic_write_json(out_path, sidecar)
    return out_path


def update_baseline_pointer(
    pointer_path: str | Path,
    verdict: DeltaVerdict,
    candidate_record_path: str | Path,
    *,
    rejected_dir: str | Path | None = None,
    initialize: bool = False,
) -> Path:
    """Advance a baseline pointer on ``KEEP``, or record a rejected verdict sidecar.

    Args:
        pointer_path: Path to the JSON baseline ratchet pointer file.
        verdict: Verdict produced by :func:`compare_probe_pair`.
        candidate_record_path: File path to the candidate ``ProbeRecord`` JSON.
        rejected_dir: Directory where ``.verdict.json`` sidecars are written when
            ``verdict`` is not promotable.
        initialize: Allow creating or overwriting a pointer without chaining from
            an existing baseline pointer.

    Returns:
        ``pointer_path`` when promoted, or the written ``.verdict.json`` path when
        not promotable.

    Raises:
        ClaimValidityError: If ``candidate_record_path`` mismatches ``verdict``,
            ratchet chain continuity breaks, or ``rejected_dir`` is omitted on a
            non-promotable verdict.
    """
    ptr = Path(pointer_path)
    cand_path = Path(candidate_record_path)
    cand_rec = ProbeRecord.read(cand_path)
    if cand_rec.probe_id != verdict.candidate_probe_id or not _shas_match(
        cand_rec.git_sha, verdict.candidate_sha
    ):
        raise ClaimValidityError(
            "candidate_record_path does not match verdict candidate provenance "
            f"(record=({cand_rec.probe_id!r}, {cand_rec.git_sha!r}), "
            f"verdict=({verdict.candidate_probe_id!r}, {verdict.candidate_sha!r}))"
        )

    if verdict.status is VerdictStatus.KEEP and verdict.promotable:
        return _write_ratchet_pointer(ptr, verdict, cand_path, cand_rec, initialize=initialize)
    return _write_rejected_sidecar(rejected_dir, verdict, cand_path)


__all__ = [
    "DeltaVerdict",
    "ParityContract",
    "VerdictStatus",
    "_check_present_failure_witnesses",
    "_shas_match",
    "compare_probe_pair",
    "update_baseline_pointer",
]
