"""Conclude-time statistical-battery gate (T2-22, #2181, AC-15).

AC-15's claim: at campaign conclude, the loop consumes the `bathos[stats]` battery verdict
(Wilcoxon/Friedman+Nemenyi, alpha=0.05 Holm; Cohen's d >= 0.2; WR >= 0.6 or P(A>B) >= 0.75; BP >=
0.2; ICC > 0.990). A confirmatory verdict is honored; a downgrade (`confounded`/`underpowered`)
hard-blocks confirmation/sequential campaigns but is only advisory for exploration.

Grounding (verified before writing any code): `depends_on: [T2-27, B2-01, B2-08]`, all merged.
`BathosStatsBatteryVerdict` mirrors `bathos.stats_gates.StatsBatteryVerdict`'s full shape exactly
(bathos PR #17, this epic, B2-01) -- this module does not invent a different shape for the
verdict. Confirmed (grepped `bathos.mcp`) that no MCP tool or CLI verb exposes
`run_stats_battery` remotely yet -- B2-01 shipped as a pure library function, callable only from
within bathos's own process. `bathos_verdict` is therefore a plain, caller-supplied value here,
the same seam as `capability_probe_gate.CapabilityProbeResult` (T2-27): whatever obtains the real
verdict (a future loop controller, or an orchestrating agent that can call bathos directly) does
so outside this module. Also confirmed: bathos's own `"underpowered"` verdict value already
covers "stats extra absent" as one of its two non-`"pass"` outcomes (bathos's own docstring:
`"underpowered": scipy was not installed...`) -- no separate handling needed for that DAG-text
clause; it falls out of the generic downgrade logic below.

`CampaignMode` mirrors `bathos.campaigns.create_campaign`'s own mode literal, same as
`capability_probe_gate.CampaignMode` (T2-27) -- redefined locally rather than imported, per every
prior `xtrax.loop` module's stated independence from its sibling gates (e.g.
`campaign_approval_gate.py`'s own docstring: does NOT import `external_stop_watchdog`).

Design point: unlike `capability_probe_gate.assert_capability_live`'s raise-based shape, this
module is a **pure decision function**, mirroring `multi_metric_ratchet.compute_ratchet_decision`
(T2-17) instead -- a downgraded statistical verdict is the campaign's own normal, expected outcome
when its hypothesis wasn't supported, not a structural anomaly the way "bathos capability isn't
live" is (T2-17's own docstring makes exactly this improved-vs-not-improved distinction). Only a
genuinely malformed `campaign_mode` raises.

Extension seam (mirrors every other T2-1x gate's stance): this module computes and returns a
`ConcludeStatsDecision`. It does not decide what a future loop controller does with a
`hard_blocked`/`advisory` decision beyond returning it, and does not implement the battery
computation or its remote retrieval itself.
"""

from dataclasses import dataclass
from typing import Literal

CampaignMode = Literal["exploration", "confirmation", "sequential"]
Verdict = Literal["pass", "confounded", "underpowered"]

_VALID_MODES: frozenset[str] = frozenset({"exploration", "confirmation", "sequential"})

#: Campaign modes AC-15 hard-blocks on a downgraded verdict. "exploration" is advisory-only.
_GATED_MODES: frozenset[str] = frozenset({"confirmation", "sequential"})


class StatsBatteryGateInputError(Exception):
    """`campaign_mode` is not one of the three recognized `CampaignMode` values.

    Distinct from a downgraded verdict (`ConcludeStatsDecision.honored=False`, not an exception)
    -- this is raised only when the input itself is malformed.
    """


@dataclass(frozen=True, slots=True)
class BathosStatsBatteryVerdict:
    """The already-computed battery verdict a caller obtained from bathos (mirrors
    `bathos.stats_gates.StatsBatteryVerdict`'s full shape verbatim -- see module docstring). This
    module does not call bathos itself.
    """

    verdict: Verdict
    scipy_available: bool
    reasons: tuple[str, ...]
    cohens_d: float
    win_rate: float
    breakdown_point: float
    p_superiority: float | None
    wilcoxon_p_value: float | None
    icc: float | None
    baseline_budget_equivalent: bool | None


@dataclass(frozen=True, slots=True)
class ConcludeStatsDecision:
    """The composed AC-15 decision at campaign conclude.

    Attributes:
        honored: `True` iff `bathos_verdict.verdict == "pass"` -- the confirmatory verdict is
            accepted.
        hard_blocked: `True` iff downgraded (`confounded`/`underpowered`) AND `campaign_mode` is
            `"confirmation"`/`"sequential"` -- the campaign must not report a pass. A future loop
            controller enforces this; this module only decides (mirrors every other pure-decision
            T2-1x module's stance).
        advisory: `True` iff downgraded AND `campaign_mode == "exploration"` -- proceed, but the
            caller should surface `reasons` loud in the campaign report, per AC-15's own
            "fast/loud" framing.
        reasons: `bathos_verdict.reasons` verbatim, passed through regardless of outcome so a
            caller can log full context whether honored, hard-blocked, or advisory.
    """

    honored: bool
    hard_blocked: bool
    advisory: bool
    reasons: tuple[str, ...]


def assess_stats_battery_verdict(
    bathos_verdict: BathosStatsBatteryVerdict,
    *,
    campaign_mode: CampaignMode,
) -> ConcludeStatsDecision:
    """The composed AC-15 gate: consume the battery verdict at campaign conclude.

    Args:
        bathos_verdict: the already-computed verdict (see module docstring -- this function does
            not call bathos itself).
        campaign_mode: the campaign's mode. `"confirmation"`/`"sequential"` hard-block on a
            downgraded verdict; `"exploration"` is advisory-only.

    Returns:
        A `ConcludeStatsDecision`. A downgraded verdict is a normal outcome (the campaign's own
        hypothesis simply wasn't supported), not an error -- this function never raises for it.

    Raises:
        StatsBatteryGateInputError: `campaign_mode` is not one of `"exploration"`,
            `"confirmation"`, `"sequential"`.
    """
    if campaign_mode not in _VALID_MODES:
        msg = (
            f"unrecognized campaign_mode {campaign_mode!r}; expected one of {sorted(_VALID_MODES)}"
        )
        raise StatsBatteryGateInputError(msg)

    honored = bathos_verdict.verdict == "pass"
    downgraded = not honored
    hard_blocked = downgraded and campaign_mode in _GATED_MODES
    advisory = downgraded and not hard_blocked

    return ConcludeStatsDecision(
        honored=honored,
        hard_blocked=hard_blocked,
        advisory=advisory,
        reasons=bathos_verdict.reasons,
    )


__all__ = [
    "BathosStatsBatteryVerdict",
    "CampaignMode",
    "ConcludeStatsDecision",
    "StatsBatteryGateInputError",
    "Verdict",
    "assess_stats_battery_verdict",
]
