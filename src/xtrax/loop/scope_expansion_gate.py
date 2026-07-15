"""Scope-expansion gate (T2-31, #2181, AC-24).

AC-24's policy: any expansion of the loop's network access, tool allowlist, or sandbox
capabilities (including adding new evolve-block surface or new effectful tools) requires
Marielle's explicit approval before the expansion takes effect. Denied-by-default absent
that approval.

This module provides a checking mechanism that verifies a requested capability expansion
has a fresh, machine-checkable approval. Each specific expansion request requires its own
per-request approval, recorded in `.praxia/loop_human_gates.toml` as a `[[gates]]` entry
with `id = "T2-31"` and a caller-defined `event_ref` field (e.g., `"network:pypi.org"` or
`"tool:new_effectful_verb"` — the caller's responsibility to make it specific enough).

Extension seam (mirrors `xtrax.loop.admission`'s and `xtrax.loop.closure_lock`'s stance):
this module raises on failure and stops there. It does not decide what a human-in-the-loop
response looks like — a future loop controller catches `ScopeExpansionNotApprovedError`
and owns the escalation workflow.
"""

import tomllib
from pathlib import Path

from xtrax.devtools.freshness import Attestation, evaluate_freshness

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GATES_TOML = ROOT / ".praxia" / "loop_human_gates.toml"


class ScopeExpansionNotApprovedError(Exception):
    """Base for AC-24's approval-failure errors. The expansion must not proceed."""


class NoMatchingApprovalError(ScopeExpansionNotApprovedError):
    """No matching `id="T2-31"` entry with the requested `event_ref` was found."""


class ApprovalExpiredError(ScopeExpansionNotApprovedError):
    """A matching entry exists but its attestation is not fresh (past TTL or stale)."""


def assert_scope_expansion_approved(
    capability_name: str, *, toml_path: Path = DEFAULT_GATES_TOML
) -> Attestation:
    """Verify a scope-expansion request has fresh, machine-checkable approval.

    Loads `toml_path` and searches for a `[[gates]]` entry where `id == "T2-31"` and
    `event_ref == capability_name`. If found and the attestation is fresh (via
    `evaluate_freshness`), returns the attestation. Otherwise raises an exception.

    Args:
        capability_name: Caller-defined identifier for the specific capability being
            requested (e.g., "network:pypi.org", "tool:new_effectful_verb"). Must match
            the `event_ref` field in the TOML entry exactly.
        toml_path: Path to the gates TOML file. Defaults to `.praxia/loop_human_gates.toml`
            relative to repo root, but always overridable (primarily for tests).

    Returns:
        The fresh Attestation from the matching entry.

    Raises:
        NoMatchingApprovalError: No matching entry found, or TOML file does not exist.
        ApprovalExpiredError: Matching entry found but attestation is not fresh.
    """
    try:
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError as exc:
        msg = (
            f"gates TOML not found at {toml_path}; cannot verify approval for "
            f"capability={capability_name!r}"
        )
        raise NoMatchingApprovalError(msg) from exc
    except Exception as exc:
        msg = (
            f"failed to parse gates TOML at {toml_path}: {exc}; "
            f"cannot verify approval for capability={capability_name!r}"
        )
        raise NoMatchingApprovalError(msg) from exc

    gates = data.get("gates", [])
    for gate_entry in gates:
        if gate_entry.get("id") == "T2-31" and gate_entry.get("event_ref") == capability_name:
            attestation = Attestation(
                attested_at=gate_entry["attested_at"],
                ttl_days=gate_entry["ttl_days"],
                attested_by=gate_entry["attested_by"],
                note=gate_entry.get("note", ""),
            )
            verdict = evaluate_freshness(attestation)
            if not verdict.fresh:
                msg = (
                    f"approval for capability={capability_name!r} is stale/expired: "
                    f"{'; '.join(verdict.reasons)}"
                )
                raise ApprovalExpiredError(msg)
            return attestation

    msg = (
        f"no matching T2-31 approval found for capability={capability_name!r}; "
        f"scope expansion cannot proceed"
    )
    raise NoMatchingApprovalError(msg)


__all__ = [
    "ApprovalExpiredError",
    "DEFAULT_GATES_TOML",
    "NoMatchingApprovalError",
    "ScopeExpansionNotApprovedError",
    "assert_scope_expansion_approved",
]
