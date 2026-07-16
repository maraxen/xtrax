"""Sidecar-drift reaction gate (T2-25, #2181, AC-18).

AC-18's claim (DAG text, `.praxia/docs/roadmaps/research-epics/260702_02-dag-2181-autoresearch.md`
section "T2-25 -- sidecar-drift reaction (AC-18)"): "across all runs of a script, the sidecar SHA
is identical to the first-run manifest; loop reacts to `SIDECAR_HASH_MISMATCH`. Success: sidecar
SHA stable across runs. Fast/loud: deny (autonomous mode) / warn (collaborative mode) -- cheapest
highest-leverage immutable-evaluator safeguard."

Grounding (verified before writing any code, `bathos/src/bathos/prereg.py`, commit `c6e87ef`,
bathos PR #14, "B2-04 sidecar drift detection (promote SIDECAR_HASH_MISMATCH)"): the DAG text's
`SIDECAR_HASH_MISMATCH` is a real, live `GateErrorCode.SIDECAR_HASH_MISMATCH` string-enum member
(value `"sidecar_hash_mismatch"`) -- not merely descriptive DAG prose, and not a bare literal
constant floating on its own. It is produced by `bathos.prereg.check_sidecar_drift(script_path,
catalog_dir, current_sidecar_sha256) -> bool`, which compares the current run's sidecar hash
against the script's *first-run manifest* baseline (its earliest catalog run, by timestamp, that
carries a non-empty `sidecar_sha256`) and returns `True` iff they differ. `check_sidecar_drift`
returns `False` both when the hashes match AND when no prior baseline exists yet (nothing to
compare against) -- bathos itself does not distinguish those two cases, and neither does this
module (see `SidecarDriftSignal.drifted` below).

Bathos's own `gate_check` (same file) already wires this into a deny-vs-warn branch keyed on
`bathos.prereg.AgentMode = Literal["collaborative", "autonomous"]` -- but that is the *bathos-side
pre-execution* gate, run when a script is actually launched through bathos's own runner. T2-25 is
the analogous reaction for the xtrax autoresearch loop itself, which -- per this epic's standing
cross-repo convention -- never calls bathos's `check_sidecar_drift` or imports `bathos.prereg`
directly. `SidecarDriftSignal` below is a plain, caller-supplied value carrying whatever
`check_sidecar_drift` (or an equivalent probe) already determined; the actual bathos call is a
future loop controller's job (or an orchestrating agent's), exactly the same seam as
`xtrax.loop.capability_probe_gate.CapabilityProbeResult` (T2-27) and
`xtrax.loop.stats_battery_gate.BathosStatsBatteryVerdict` (T2-22).

`AgentMode` mirrors `bathos.prereg.AgentMode`'s own literal (`"collaborative"`/`"autonomous"`)
verbatim, same spelling and order -- grepped `src/xtrax/loop/` and `src/xtrax/run/` first (per
this module's own dispatch instructions) and found no existing autonomous/collaborative mode
concept anywhere in xtrax; this is the first module needing it. It is a DISTINCT axis from
`CampaignMode` (`"exploration"`/`"confirmation"`/`"sequential"`, first introduced by
`capability_probe_gate.py`, T2-27) -- an agent can run collaboratively or autonomously in any
campaign mode; the two are orthogonal and this module does not conflate or combine them. Redefined
locally rather than imported from `capability_probe_gate` or `stats_battery_gate`, per "gates don't
import each other" (explicit in every prior T2-1x module's own docstring).

Shape decision (a) vs (b): neither pure shape fits alone, because "deny (autonomous) / warn
(collaborative)" is a genuine two-way branch, not a single refuse-or-proceed binary. The deny half
mirrors shape (a) (`capability_probe_gate.assert_capability_live`): a sidecar SHA changing across
runs of the *same script* is a structural anomaly -- something machine-verified as wrong (the
pre-registration silently changed after evidence started accumulating under it) -- not a normal,
expected outcome the way a downgraded statistics verdict is (`stats_battery_gate`'s own docstring
draws exactly this improved-vs-not-improved distinction; sidecar drift has no "hypothesis wasn't
supported" analogue). So this module raises on the deny path. But the warn half cannot be modeled
by raising at all -- AC-18 requires it to be non-blocking, so `assert_sidecar_drift_reaction` below
raises `SidecarHashMismatchError` only when `agent_mode == "autonomous"` and drift is detected, and
otherwise (no drift in either mode, or drift in `"collaborative"` mode) returns a
`SidecarDriftDecision` the caller can act on. A `SidecarDriftDecision` is therefore only ever
observed for a non-denying outcome (by construction -- the deny path never returns), so it carries
no `should_deny` field; `should_warn` is the only reaction flag it needs.

Extension seam (mirrors every other T2-1x gate's stance): this module raises or returns a decision
and stops there. It does not call `check_sidecar_drift` itself, does not decide what a future loop
controller does with a warn decision beyond returning it (e.g. whether/how to surface it in a
campaign report), and does not retry or re-probe bathos.
"""

from dataclasses import dataclass
from typing import Literal

AgentMode = Literal["collaborative", "autonomous"]

_VALID_MODES: frozenset[str] = frozenset({"collaborative", "autonomous"})


class SidecarDriftGateInputError(Exception):
    """`agent_mode` is not one of the two recognized `AgentMode` values.

    Distinct from a detected drift (`SidecarHashMismatchError` on deny, or a returned
    `SidecarDriftDecision` on warn/no-drift) -- this is raised only when the input itself is
    malformed.
    """


class SidecarHashMismatchError(Exception):
    """AC-18's deny path: `signal.drifted` is True and `agent_mode == "autonomous"`.

    Names the script and both hashes in its message. HALT the loop; do not retry -- the
    pre-registration itself is suspect, not the run.
    """


@dataclass(frozen=True, slots=True)
class SidecarDriftSignal:
    """The already-computed drift signal a caller obtained from bathos (see module docstring --
    this module does not call `bathos.prereg.check_sidecar_drift` itself).

    Attributes:
        drifted: `check_sidecar_drift(...)`'s own return value verbatim -- True iff the current
            run's sidecar hash differs from the script's first-run manifest baseline. False both
            when the hashes match AND when no prior baseline exists yet (bathos's own "nothing to
            compare against yet" case) -- this module does not need to distinguish those two,
            since neither warrants a reaction.
        script_id: a human-readable identifier for the script (path or stem), for messages only.
        first_run_sha256: the first-run manifest's sidecar hash, if the caller retrieved it (""
            if not) -- for message-building only, never compared here.
        current_sha256: the current run's sidecar hash, same caveat as `first_run_sha256`.
    """

    drifted: bool
    script_id: str
    first_run_sha256: str = ""
    current_sha256: str = ""


@dataclass(frozen=True, slots=True)
class SidecarDriftDecision:
    """The composed AC-18 decision for every outcome except the deny path (which raises instead
    of returning -- see module docstring).

    Attributes:
        drifted: `signal.drifted` verbatim.
        should_warn: True iff `drifted` and `agent_mode == "collaborative"` -- the caller should
            surface a loud, non-blocking warning (AC-18's own "warn (collaborative mode)" half).
            False when there was no drift to react to.
        reason: human-readable message naming the script and both hashes; `""` iff `drifted` is
            False.
    """

    drifted: bool
    should_warn: bool
    reason: str


def assert_sidecar_drift_reaction(
    signal: SidecarDriftSignal,
    *,
    agent_mode: AgentMode,
) -> SidecarDriftDecision:
    """The composed AC-18 gate: react to bathos's `SIDECAR_HASH_MISMATCH` signal.

    Args:
        signal: the already-computed drift signal (see module docstring -- this function does
            not call `bathos.prereg.check_sidecar_drift` itself).
        agent_mode: `"autonomous"` denies (raises) on drift; `"collaborative"` only warns
            (returns a decision, never raises for this reason).

    Returns:
        A `SidecarDriftDecision`. Reached only for a non-denying outcome: no drift (either mode),
        or drift under `"collaborative"` mode (`should_warn=True`).

    Raises:
        SidecarDriftGateInputError: `agent_mode` is not one of `"collaborative"`, `"autonomous"`.
        SidecarHashMismatchError: `signal.drifted` is True and `agent_mode == "autonomous"` -- a
            structural anomaly (mirrors `capability_probe_gate.assert_capability_live`'s own
            stance; see module docstring), not a normal outcome to merely advise around.
    """
    if agent_mode not in _VALID_MODES:
        msg = f"unrecognized agent_mode {agent_mode!r}; expected one of {sorted(_VALID_MODES)}"
        raise SidecarDriftGateInputError(msg)

    if not signal.drifted:
        return SidecarDriftDecision(drifted=False, should_warn=False, reason="")

    reason = (
        f"sidecar hash drift detected for {signal.script_id!r}: first-run manifest sha256="
        f"{signal.first_run_sha256!r}, current sha256={signal.current_sha256!r} -- "
        "pre-registration changed after evidence started accumulating under it"
    )
    if agent_mode == "autonomous":
        raise SidecarHashMismatchError(reason)

    return SidecarDriftDecision(drifted=True, should_warn=True, reason=reason)


__all__ = [
    "AgentMode",
    "SidecarDriftDecision",
    "SidecarDriftGateInputError",
    "SidecarDriftSignal",
    "SidecarHashMismatchError",
    "assert_sidecar_drift_reaction",
]
