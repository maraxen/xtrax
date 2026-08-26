"""Promotion-to-main gate (T2-30, #2181, AC-23).

AC-23's policy (from `.praxia/docs/decisions/260714_2181-autoresearch-loop-constitution.md`):
evolved code proposed for promotion out of the sandbox lineage into `xtrax` `main` requires
Marielle's explicit engineering review and approval per promotion. Refused-by-default: absent
explicit approval, code stays in the sandbox lineage indefinitely. This is a per-promotion gate,
not a standing grant -- passing review once does not pre-approve a later promotion.

This module enforces the per-promotion check: given a candidate commit SHA, verify that an
explicit approval entry exists in the TOML gates file, and that its attestation is still fresh
(per `xtrax.run.freshness.Attestation`/`evaluate_freshness` semantics).

TOML schema: `[[gates]]` entries for `id="T2-30"` carry an `event_ref` field holding the
candidate commit SHA. No other gate type uses `event_ref` -- it is T2-30-specific, matching the
per-promotion semantics (gate c is tied to concrete commits, gates (b)/(d)/(e) are tied to
events/types rather than fixed SHAs).

Extension seam (mirrors `xtrax.loop.admission`'s and `xtrax.loop.closure_lock`'s stance): this
module raises on approval failure and stops there. It does not touch git/branches/merges -- it
only answers "is this promotion approved." A future loop controller catches
`PromotionNotApprovedError` and owns the actual promotion workflow (branch creation, cherry-pick,
merge, etc.). This module is the check; the controller is the action.
"""

import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from xtrax.run.freshness import Attestation, evaluate_freshness

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GATES_TOML = ROOT / ".praxia" / "loop_human_gates.toml"


class PromotionNotApprovedError(Exception):
    """Base for AC-23's promotion-refusal errors. Code stays in sandbox; do not retry."""


class NoMatchingApprovalError(PromotionNotApprovedError):
    """No approval entry found for this candidate commit SHA."""


class ApprovalExpiredError(PromotionNotApprovedError):
    """Approval entry exists but its attestation is not fresh (expired or unreviewed)."""


@dataclass(frozen=True, slots=True)
class GateEntry:
    """A single [[gates]] entry from the TOML file, cast to structured form.

    `event_ref` is the candidate commit SHA for T2-30 entries; optional for other gate types.
    The other fields mirror `xtrax.run.freshness.Attestation` plus gate metadata.
    """

    id: str
    event_ref: str | None
    attested_at: str
    ttl_days: float
    attested_by: str
    note: str = ""


def _parse_gate_entry(entry: dict) -> GateEntry:
    """Cast a TOML gate dict to GateEntry, extracting the fields we care about."""
    return GateEntry(
        id=entry.get("id", ""),
        event_ref=entry.get("event_ref"),
        attested_at=entry.get("attested_at", ""),
        ttl_days=float(entry.get("ttl_days", 0)),
        attested_by=entry.get("attested_by", ""),
        note=entry.get("note", ""),
    )


def assert_promotion_approved(
    candidate_commit_sha: str,
    *,
    toml_path: Path = DEFAULT_GATES_TOML,
    now: datetime | None = None,
) -> Attestation:
    """Verify that `candidate_commit_sha` has a fresh approval attestation in the gates file.

    Loads the TOML file, searches for `[[gates]]` entries where `id == "T2-30"` AND
    `event_ref == candidate_commit_sha`, checks the attestation's freshness via
    `evaluate_freshness`, and returns the attestation on success.

    Args:
        candidate_commit_sha: The commit SHA being proposed for promotion.
        toml_path: Path to the gates TOML file (default: `.praxia/loop_human_gates.toml`).
        now: Reference time for freshness check (default: current UTC).

    Returns:
        The fresh `Attestation` from the matching gate entry.

    Raises:
        NoMatchingApprovalError: No matching T2-30 entry found, or TOML file doesn't exist.
        ApprovalExpiredError: Matching entry found but attestation is not fresh.
    """
    try:
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError as exc:
        msg = f"gates file not found: {toml_path}"
        raise NoMatchingApprovalError(msg) from exc
    except Exception as exc:
        msg = f"failed to parse gates file: {exc}"
        raise NoMatchingApprovalError(msg) from exc

    gates = data.get("gates", [])
    if not isinstance(gates, list):
        msg = "gates TOML entry is not a list"
        raise NoMatchingApprovalError(msg)

    matching_entry = None
    for entry in gates:
        if not isinstance(entry, dict):
            continue
        parsed = _parse_gate_entry(entry)
        if parsed.id == "T2-30" and parsed.event_ref == candidate_commit_sha:
            matching_entry = parsed
            break

    if matching_entry is None:
        msg = f"no T2-30 approval found for commit {candidate_commit_sha}"
        raise NoMatchingApprovalError(msg)

    attestation = Attestation(
        attested_at=matching_entry.attested_at,
        ttl_days=matching_entry.ttl_days,
        attested_by=matching_entry.attested_by,
        note=matching_entry.note,
    )

    verdict = evaluate_freshness(attestation, now=now)
    if not verdict.fresh:
        msg = f"approval expired or not fresh: {'; '.join(verdict.reasons)}"
        raise ApprovalExpiredError(msg)

    return attestation


__all__ = [
    "ApprovalExpiredError",
    "DEFAULT_GATES_TOML",
    "GateEntry",
    "NoMatchingApprovalError",
    "PromotionNotApprovedError",
    "assert_promotion_approved",
]
