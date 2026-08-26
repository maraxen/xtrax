"""Tests for T3-05 TTL-attestation + invalidate-only-probe freshness primitive (#3033)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from xtrax.run.freshness import (
    SKIP_PROBE_ENV_VAR,
    SKIP_TTL_ENV_VAR,
    Attestation,
    ProbeResult,
    days_since,
    evaluate_freshness,
)

NOW = datetime(2026, 7, 10, tzinfo=UTC)


def _attestation(
    *, attested_at: str = "2026-07-01T00:00:00+00:00", ttl_days: float = 30.0
) -> Attestation:
    return Attestation(attested_at=attested_at, ttl_days=ttl_days, attested_by="tester")


def test_days_since_computes_fractional_days() -> None:
    assert days_since("2026-07-01T00:00:00+00:00", now=NOW) == pytest.approx(9.0)


def test_days_since_treats_naive_timestamp_as_utc() -> None:
    assert days_since("2026-07-01T00:00:00", now=NOW) == pytest.approx(9.0)


def test_within_ttl_and_no_probe_is_fresh() -> None:
    verdict = evaluate_freshness(_attestation(ttl_days=30.0), now=NOW, probe=None)

    assert verdict.fresh
    assert not verdict.ttl_expired
    assert not verdict.probe_invalidated
    assert verdict.reasons == ()


def test_past_ttl_fails_loud_even_with_no_probe() -> None:
    verdict = evaluate_freshness(_attestation(ttl_days=5.0), now=NOW, probe=None)

    assert not verdict.fresh
    assert verdict.ttl_expired
    assert not verdict.probe_invalidated
    assert any("past TTL" in reason for reason in verdict.reasons)


def test_probe_invalidates_an_otherwise_fresh_attestation() -> None:
    def probe() -> ProbeResult:
        return ProbeResult(invalidated=True, reason="PyPI release not found")

    verdict = evaluate_freshness(_attestation(ttl_days=30.0), now=NOW, probe=probe)

    assert not verdict.fresh
    assert not verdict.ttl_expired
    assert verdict.probe_invalidated
    assert any("PyPI release not found" in reason for reason in verdict.reasons)


def test_probe_cannot_satisfy_an_expired_attestation() -> None:
    def probe() -> ProbeResult:
        return ProbeResult(invalidated=False)

    verdict = evaluate_freshness(_attestation(ttl_days=5.0), now=NOW, probe=probe)

    assert not verdict.fresh
    assert verdict.ttl_expired


def test_probe_exception_is_treated_as_skipped_not_invalidating() -> None:
    def probe() -> ProbeResult:
        raise ConnectionError("network unreachable")

    verdict = evaluate_freshness(_attestation(ttl_days=30.0), now=NOW, probe=probe)

    assert verdict.fresh
    assert verdict.probe_skipped
    assert not verdict.probe_invalidated


def test_probe_skipped_result_is_treated_as_skipped_not_invalidating() -> None:
    def probe() -> ProbeResult:
        return ProbeResult(invalidated=False, skipped=True, reason="timeout")

    verdict = evaluate_freshness(_attestation(ttl_days=30.0), now=NOW, probe=probe)

    assert verdict.fresh
    assert verdict.probe_skipped
    assert not verdict.probe_invalidated


def test_disabling_ttl_backstop_explicitly_still_runs_probe() -> None:
    def probe() -> ProbeResult:
        return ProbeResult(invalidated=True, reason="revoked")

    verdict = evaluate_freshness(
        _attestation(ttl_days=5.0),
        now=NOW,
        probe=probe,
        ttl_backstop_enabled=False,
    )

    assert not verdict.fresh
    assert not verdict.ttl_expired  # backstop disabled, so this stays False...
    assert verdict.probe_invalidated  # ...but the probe still fired and invalidated


def test_disabling_probe_explicitly_still_runs_ttl_backstop() -> None:
    probe_called = False

    def probe() -> ProbeResult:
        nonlocal probe_called
        probe_called = True
        return ProbeResult(invalidated=True)

    verdict = evaluate_freshness(
        _attestation(ttl_days=5.0),
        now=NOW,
        probe=probe,
        probe_enabled=False,
    )

    assert not probe_called
    assert not verdict.fresh
    assert verdict.ttl_expired


def test_env_switches_are_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting one skip env var must not disable the other check (PM-4)."""
    probe_called = False

    def probe() -> ProbeResult:
        nonlocal probe_called
        probe_called = True
        return ProbeResult(invalidated=True, reason="revoked")

    monkeypatch.setenv(SKIP_PROBE_ENV_VAR, "1")
    monkeypatch.delenv(SKIP_TTL_ENV_VAR, raising=False)

    verdict = evaluate_freshness(_attestation(ttl_days=5.0), now=NOW, probe=probe)

    assert not probe_called
    assert not verdict.fresh
    assert verdict.ttl_expired  # TTL backstop still active; SKIP_PROBE alone didn't touch it


def test_a_single_blanket_skip_cannot_disable_both_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """There is no single env var that maps to both switches — verify neither name overlaps."""
    monkeypatch.setenv(SKIP_TTL_ENV_VAR, "1")

    def probe() -> ProbeResult:
        return ProbeResult(invalidated=True, reason="revoked")

    verdict = evaluate_freshness(_attestation(ttl_days=5.0), now=NOW, probe=probe)

    assert not verdict.fresh
    assert not verdict.ttl_expired  # TTL backstop skipped
    assert verdict.probe_invalidated  # but the probe (unrelated env var) still ran
