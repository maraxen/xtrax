"""Evaluator-change gate (T2-29, #2181, AC-22, gate b) -- standing runtime gate.

DAG text (T2-29, `.praxia/docs/roadmaps/research-epics/260702_02-dag-2181-autoresearch.md`): "any
change to evaluator code, splits, or metric definitions is gated by a human-approval node -- the
agent never approves its own judge. When it fires it forces a closure-hash re-lock (T2-05) before
the changed evaluator is trusted. Success: human sign-off + closure-hash re-lock (T2-05) before the
changed evaluator is trusted. Fast/loud: change rejected; loop halts until human sign-off + closure
hash re-lock."

Constitution policy (b) (`.praxia/docs/decisions/260714_2181-autoresearch-loop-constitution.md`,
already SIGNED/T2-28): "any change to evaluator code, test splits, or metric definitions requires
Marielle's explicit sign-off before the changed evaluator is trusted, and forces a closure-hash
re-lock (T2-05) of the new evaluator's complete closure... The agent never approves its own judge,
under any circumstance. This gate fires on every evaluator-change event, not once -- there is no
standing blanket approval."

Per-event scoping (the load-bearing design point): "not once -- there is no standing blanket
approval" means a fresh-but-unscoped T2-29 attestation would be wrong -- one approval could then
cover every future evaluator change forever, which is exactly the standing-blanket-approval the
constitution rules out. This module instead scopes each attestation to one specific evaluator-
change *event* via an `event_ref` field carrying the new closure's `ClosureManifest.closure_hash`
(`xtrax.loop.closure_lock`) -- a closure hash is a precise, content-addressed identifier for
"exactly this one evaluator-change event" (same code + splits + metric defs + pinned deps +
config), so an attestation for one hash can never be silently reused to approve a different change.

TOML schema extension (documented here only -- this module does not edit or own the real
`.praxia/loop_human_gates.toml` file, nor any shared schema-definition file): a `[[gates]]` entry
for this gate carries, in addition to the existing free-form fields other gates already use
(`id`, `ac`, `slug`, `title`, `attested_at`, `ttl_days`, `attested_by`, `note`, ...):

    [[gates]]
    id = "T2-29"
    event_ref = "<ClosureManifest.closure_hash of the newly-changed evaluator>"
    attested_at = "2026-07-15T00:00:00Z"   # ISO 8601
    ttl_days = 30
    attested_by = "Marielle Russo"
    note = "..."                            # optional, defaults to ""

`id` identifies the gate kind (T2-29 = evaluator change); `event_ref` identifies *which*
evaluator-change event this particular sign-off covers. Multiple `[[gates]]` entries with
`id = "T2-29"` and different `event_ref` values coexist freely -- each is a separate approved
event. A `[[gates]]` entry with `id = "T2-29"` but a stale `event_ref` (approving a *different*
evaluator change than the one currently proposed) does not satisfy this gate.

Composition seam: this module only *checks* for a matching, fresh sign-off -- it does not itself
perform the closure-hash re-lock the DAG text and constitution both require. That is
`xtrax.loop.closure_lock.build_closure_manifest`, already built and merged (T2-05). The intended
call sequence, owned by a future loop controller (mirrors `xtrax.loop.admission`'s and
`xtrax.loop.closure_lock`'s stance: this module raises and stops, it does not decide what happens
next):

    new_manifest = build_closure_manifest(...)  # hash the proposed new evaluator closure
    assert_evaluator_change_approved(new_manifest, toml_path=...)  # raises if unapproved/stale
    # only after both succeed does a caller treat `new_manifest` as the newly-locked closure.

No #2181 loop controller exists yet to own that swap-in step, so it is not built here -- inventing
one now would be speculative and likely wrong-shaped once the real controller lands.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xtrax.devtools.freshness import Attestation, evaluate_freshness
from xtrax.loop.closure_lock import ClosureManifest

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GATES_TOML = ROOT / ".praxia" / "loop_human_gates.toml"

_GATE_ID = "T2-29"


class EvaluatorChangeNotApprovedError(Exception):
    """Base for AC-22's evaluator-change-rejection errors. HALT the loop; do not retry."""


class NoMatchingApprovalError(EvaluatorChangeNotApprovedError):
    """No `[[gates]]` entry with `id="T2-29"` and this exact `event_ref` exists.

    Also raised (rather than letting `FileNotFoundError`/`tomllib.TOMLDecodeError` propagate
    unhandled) when the gates TOML is missing or malformed -- from this gate's perspective, an
    unreadable attestation file is indistinguishable from "no sign-off exists yet".
    """


class ApprovalExpiredError(EvaluatorChangeNotApprovedError):
    """A matching approval exists but its attestation is no longer fresh (TTL expired)."""


@dataclass(frozen=True, slots=True)
class _GateEntry:
    attested_at: str
    ttl_days: float
    attested_by: str
    note: str


def _load_gates(toml_path: Path) -> list[dict[str, Any]]:
    try:
        raw = toml_path.read_bytes()
    except (FileNotFoundError, OSError):
        return []
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except tomllib.TOMLDecodeError:
        return []
    gates = data.get("gates", [])
    if not isinstance(gates, list):
        return []
    return [g for g in gates if isinstance(g, dict)]


def _to_gate_entry(raw: dict[str, Any]) -> _GateEntry | None:
    try:
        return _GateEntry(
            attested_at=str(raw["attested_at"]),
            ttl_days=float(raw["ttl_days"]),
            attested_by=str(raw["attested_by"]),
            note=str(raw.get("note", "")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def assert_evaluator_change_approved(
    new_locked: ClosureManifest,
    *,
    toml_path: Path = DEFAULT_GATES_TOML,
) -> Attestation:
    """The composed AC-22 gate: a fresh, event-scoped sign-off for exactly `new_locked`.

    Scans `toml_path` for `[[gates]]` entries where `id == "T2-29"` and
    `event_ref == new_locked.closure_hash`. Of any matches, the most recent (by `attested_at`) is
    evaluated for freshness via `xtrax.devtools.freshness.evaluate_freshness`.

    Raises:
        NoMatchingApprovalError: no entry matches this exact `event_ref` (including: the TOML
            file doesn't exist, is malformed, or has no `id="T2-29"` entries at all).
        ApprovalExpiredError: a matching entry exists but its attestation has expired.

    Returns:
        The `Attestation` built from the matching (most recent) entry, once confirmed fresh.
    """
    matches = [
        raw
        for raw in _load_gates(toml_path)
        if raw.get("id") == _GATE_ID and raw.get("event_ref") == new_locked.closure_hash
    ]
    if not matches:
        msg = (
            f"no sign-off found for evaluator-change event event_ref="
            f"{new_locked.closure_hash!r} (gate id={_GATE_ID!r}) in {toml_path} -- "
            "the agent never approves its own judge; a human sign-off scoped to this exact "
            "evaluator-change event is required before it is trusted"
        )
        raise NoMatchingApprovalError(msg)

    entries = [entry for raw in matches if (entry := _to_gate_entry(raw)) is not None]
    if not entries:
        msg = (
            f"matching [[gates]] entry/entries for event_ref={new_locked.closure_hash!r} "
            f"(gate id={_GATE_ID!r}) in {toml_path} are malformed (missing/invalid "
            "attested_at/ttl_days/attested_by)"
        )
        raise NoMatchingApprovalError(msg)

    most_recent = max(entries, key=lambda e: e.attested_at)
    attestation = Attestation(
        attested_at=most_recent.attested_at,
        ttl_days=most_recent.ttl_days,
        attested_by=most_recent.attested_by,
        note=most_recent.note,
    )

    verdict = evaluate_freshness(attestation)
    if not verdict.fresh:
        msg = (
            f"evaluator-change sign-off for event_ref={new_locked.closure_hash!r} "
            f"(gate id={_GATE_ID!r}) has expired: {'; '.join(verdict.reasons)}"
        )
        raise ApprovalExpiredError(msg)

    return attestation


__all__ = [
    "DEFAULT_GATES_TOML",
    "ApprovalExpiredError",
    "EvaluatorChangeNotApprovedError",
    "NoMatchingApprovalError",
    "assert_evaluator_change_approved",
]
