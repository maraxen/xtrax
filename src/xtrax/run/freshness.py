"""TTL-attestation + invalidate-only-probe freshness primitive (T3-05/AC-X6, #3033).

Deliberately not release-readiness-specific (spec 260702_xtrax-workflows-as-a-praxia-plugin-packa):
a within-TTL attestation is the hermetic, offline loud-fail backstop; an opportunistic probe can
only INVALIDATE an attestation, never satisfy one. First consumer: replaces the hand-maintained
`expected_status` copied verbatim as live status in `scripts/audit_release_readiness.py`'s human
gates (closes debt #411's n9-OIDC staleness defect). Callers own what a gate means; this module
only answers "is this attestation still trustworthy".
"""

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

# Two independently-named switches so a blanket "skip checks" env var cannot silently
# disable both the TTL backstop and the probe at once — the exact PM-4 regression this
# primitive exists to prevent.
SKIP_TTL_ENV_VAR = "XTRAX_FRESHNESS_SKIP_TTL"
SKIP_PROBE_ENV_VAR = "XTRAX_FRESHNESS_SKIP_PROBE"


@dataclass(frozen=True, slots=True)
class Attestation:
    """A hermetic, offline claim that some fact was verified at `attested_at`."""

    attested_at: str  # ISO 8601; naive timestamps are assumed UTC
    ttl_days: float
    attested_by: str
    note: str = ""


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Result of an opportunistic, network-dependent check.

    A probe can only invalidate an attestation, never satisfy one: `skipped=True`
    (network unreachable, timeout, probe error) must never be treated as a pass.
    """

    invalidated: bool
    reason: str = ""
    skipped: bool = False


Probe = Callable[[], ProbeResult]


@dataclass(frozen=True, slots=True)
class FreshnessVerdict:
    fresh: bool
    ttl_expired: bool
    probe_invalidated: bool
    probe_skipped: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


def days_since(attested_at: str, *, now: datetime | None = None) -> float:
    """Fractional days between `attested_at` (ISO 8601) and `now` (default: current UTC)."""
    parsed = datetime.fromisoformat(attested_at)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    reference = now or datetime.now(UTC)
    delta = reference - parsed.astimezone(UTC)
    return max(delta.total_seconds() / 86_400.0, 0.0)


def _env_enabled(var: str, *, default: bool) -> bool:
    """Truthy value in `var` means "skip" (disabled); unset keeps `default`."""
    raw = os.environ.get(var)
    if raw is None:
        return default
    return raw.strip().lower() not in {"1", "true", "yes"}


def evaluate_freshness(
    attestation: Attestation,
    *,
    now: datetime | None = None,
    probe: Probe | None = None,
    ttl_backstop_enabled: bool | None = None,
    probe_enabled: bool | None = None,
) -> FreshnessVerdict:
    """Evaluate whether `attestation` is still trustworthy.

    `ttl_backstop_enabled`/`probe_enabled` default to True unless explicitly passed or
    overridden by their own independently-named env var (`SKIP_TTL_ENV_VAR` /
    `SKIP_PROBE_ENV_VAR`) — setting one never affects the other (PM-4).
    """
    resolved_now = now or datetime.now(UTC)
    ttl_on = (
        ttl_backstop_enabled
        if ttl_backstop_enabled is not None
        else _env_enabled(SKIP_TTL_ENV_VAR, default=True)
    )
    probe_on = (
        probe_enabled
        if probe_enabled is not None
        else _env_enabled(SKIP_PROBE_ENV_VAR, default=True)
    )

    reasons: list[str] = []
    ttl_expired = False
    probe_invalidated = False
    probe_skipped = False

    if ttl_on:
        age_days = days_since(attestation.attested_at, now=resolved_now)
        if age_days > attestation.ttl_days:
            ttl_expired = True
            reasons.append(
                f"attestation past TTL: {age_days:.1f}d old, ttl={attestation.ttl_days}d "
                f"(attested_by={attestation.attested_by!r})"
            )

    if probe_on and probe is not None:
        try:
            result = probe()
        except Exception as exc:
            result = ProbeResult(invalidated=False, skipped=True, reason=str(exc))
        if result.skipped:
            probe_skipped = True
        elif result.invalidated:
            probe_invalidated = True
            reasons.append(f"probe invalidated attestation: {result.reason}")

    fresh = not ttl_expired and not probe_invalidated
    return FreshnessVerdict(
        fresh=fresh,
        ttl_expired=ttl_expired,
        probe_invalidated=probe_invalidated,
        probe_skipped=probe_skipped,
        reasons=tuple(reasons),
    )


__all__ = [
    "SKIP_PROBE_ENV_VAR",
    "SKIP_TTL_ENV_VAR",
    "Attestation",
    "FreshnessVerdict",
    "Probe",
    "ProbeResult",
    "days_since",
    "evaluate_freshness",
]
