"""Campaign-approval gate (T2-32, #2181, AC-25).

AC-25's policy (from `.praxia/docs/decisions/260714_2181-autoresearch-loop-constitution.md`):
every campaign start requires Marielle's explicit approval before the campaign may run. The
external watchdog's kill authority (see §2) is always available regardless of campaign approval
status, and cannot be revoked by the loop under any circumstance, including mid-campaign.

This module enforces the per-campaign-start check: given a campaign ID, verify that an explicit
approval entry exists in the TOML gates file, and that its attestation is still fresh (per
`xtrax.devtools.freshness.Attestation`/`evaluate_freshness` semantics).

TOML schema: `[[gates]]` entries for `id="T2-32"` carry an `event_ref` field holding the
campaign's own identifier (a caller-supplied string). No other gate type uses `event_ref` -- it
is T2-32-specific, matching the per-campaign semantics (gate e is tied to specific campaign IDs,
not to standing runtime types).

Note on external watchdog separation: This module does NOT interact with or import
`xtrax.loop.external_stop_watchdog`. The watchdog's kill authority is implemented separately
and independently; the watchdog's ability to SIGKILL a campaign is unconditional and unrevokable
by the loop, even if this approval gate refuses. This module is purely the "may this campaign
START" half of AC-25 -- the separate, always-available kill authority is AC-13's responsibility.

Extension seam (mirrors `xtrax.loop.admission`'s and `xtrax.loop.closure_lock`'s stance): this
module raises on approval failure and stops there. It does not touch campaigns or workers -- it
only answers "is this campaign approved." A future loop controller catches
`CampaignNotApprovedError` and owns the actual campaign start workflow (worker submission,
result tracking, etc.). This module is the check; the controller is the action.
"""

import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from xtrax.devtools.freshness import Attestation, evaluate_freshness

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GATES_TOML = ROOT / ".praxia" / "loop_human_gates.toml"


class CampaignNotApprovedError(Exception):
    """Base for AC-25's campaign-refusal errors. Campaign cannot start; do not retry."""


class NoMatchingApprovalError(CampaignNotApprovedError):
    """No approval entry found for this campaign ID."""


class ApprovalExpiredError(CampaignNotApprovedError):
    """Approval entry exists but its attestation is not fresh (expired or unreviewed)."""


@dataclass(frozen=True, slots=True)
class GateEntry:
    """A single [[gates]] entry from the TOML file, cast to structured form.

    `event_ref` is the campaign ID for T2-32 entries; optional for other gate types.
    The other fields mirror `xtrax.devtools.freshness.Attestation` plus gate metadata.
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


def assert_campaign_approved(
    campaign_id: str,
    *,
    toml_path: Path = DEFAULT_GATES_TOML,
    now: datetime | None = None,
) -> Attestation:
    """Verify that `campaign_id` has a fresh approval attestation in the gates file.

    Loads the TOML file, searches for `[[gates]]` entries where `id == "T2-32"` AND
    `event_ref == campaign_id`, checks the attestation's freshness via
    `evaluate_freshness`, and returns the attestation on success.

    Args:
        campaign_id: The campaign's identifier (caller-supplied string).
        toml_path: Path to the gates TOML file (default: `.praxia/loop_human_gates.toml`).
        now: Reference time for freshness check (default: current UTC).

    Returns:
        The fresh `Attestation` from the matching gate entry.

    Raises:
        NoMatchingApprovalError: No matching T2-32 entry found, or TOML file doesn't exist.
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
        if parsed.id == "T2-32" and parsed.event_ref == campaign_id:
            matching_entry = parsed
            break

    if matching_entry is None:
        msg = f"no T2-32 approval found for campaign {campaign_id}"
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
    "CampaignNotApprovedError",
    "DEFAULT_GATES_TOML",
    "GateEntry",
    "NoMatchingApprovalError",
    "assert_campaign_approved",
]
