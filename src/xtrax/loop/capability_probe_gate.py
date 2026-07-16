"""Capability-probe gate (T2-27, #2181, AC-20, PM-4).

AC-20's claim: on confirmatory-campaign start, the loop controller machine-probes bathos that
`Run.seed` + the statistical battery are **live** -- not a hand-maintained attestation that could
silently drift from reality. PM-4: refuse until bathos capability is machine-verified, never
assumed. Advisory/exploration campaigns are explicitly exempt and may proceed regardless.

Grounding (verified before writing any code): `depends_on: [T2-17, B2-06, T2-32]`, all merged.
`CapabilityProbeResult` mirrors `bathos.capability.CapabilityReport`'s two independent liveness
booleans exactly (bathos PR #18, this epic): `seed_live` (does the catalog's warm `runs` table
actually carry the `seed`/`baseline_hpo_trials`/`baseline_hpo_compute_budget` columns -- schema
migrations apply lazily via `bathos.compact.compact()`, so an unmigrated warm DB genuinely lacks
them) and `stats_battery_live` (does `scipy` -- the `bathos[stats]` extra -- import cleanly). This
module does not invent a different shape for "live."

`CampaignMode` mirrors `bathos.campaigns.create_campaign`'s own mode literal
(`"exploration"`/`"confirmation"`/`"sequential"`) verbatim -- no existing `xtrax.loop` module had
a campaign-mode concept before this one (grepped all prior T2-1x modules), so this is the first,
and matching bathos's own spelling avoids a translation-layer mismatch for whatever caller bridges
the two systems.

Cross-repo integration seam (matches `xtrax.run.repro_floor`, `xtrax.loop.metrics_provenance`, and
`xtrax.run.component_binding` exactly -- all already-merged #2181 items crossing into bathos):
xtrax has no dependency on bathos, by design. This module does NOT call bathos's
`capability_probe` MCP tool itself -- `probe_result` is a plain, caller-supplied value, assembled
from whatever the caller obtained by calling that tool (or an equivalent probe). The actual probe
call is a future loop controller's job (or an orchestrating agent's); this module is only the
refuse/proceed decision.

Extension seam (mirrors every other T2-1x gate's stance): this module raises on refusal and stops
there. It does not retry the probe, does not decide what a future loop controller does after a
refusal beyond raising loud, and does not implement the probe call itself.

Patch note (T2-22 design pass, #2181): `assert_capability_live` previously checked `campaign_mode`
only against `_GATED_MODES` -- any unrecognized string (a typo like `"confirmatory"`) fell through
to the same no-op as `"exploration"`, silently letting a malformed campaign proceed unguarded. This
directly contradicted PM-4's own "refuse until machine-verified, never assumed" principle. Now
validated against all three known `CampaignMode` values up front, raising
`CapabilityProbeInputError` (matches the `*InputError` naming convention used by
`multi_metric_ratchet.RatchetInputError` and `diversity_quota.DiversityQuotaInputError`) for
anything else.
"""

from dataclasses import dataclass
from typing import Literal

CampaignMode = Literal["exploration", "confirmation", "sequential"]

_VALID_MODES: frozenset[str] = frozenset({"exploration", "confirmation", "sequential"})

#: Campaign modes T2-27 actually gates. "exploration" always proceeds -- PM-4's own carve-out.
_GATED_MODES: frozenset[str] = frozenset({"confirmation", "sequential"})


class CapabilityProbeInputError(Exception):
    """`campaign_mode` is not one of the three recognized `CampaignMode` values.

    Distinct from a refused capability-live check (`CapabilityNotLiveError`) -- this is raised
    only when the input itself is malformed.
    """


class CapabilityNotLiveError(Exception):
    """AC-20's refusal error: bathos capability is not machine-verified live for a
    confirmatory/sequential campaign start. Names which check(s) failed in its message."""


@dataclass(frozen=True, slots=True)
class CapabilityProbeResult:
    """The two independent liveness checks a caller obtained from bathos (mirrors
    `bathos.capability.CapabilityReport`'s own shape -- see module docstring).

    Attributes:
        seed_live: whether the catalog's warm `runs` table carries the B2-02 seed columns.
        stats_battery_live: whether scipy (the `bathos[stats]` extra) is importable.
    """

    seed_live: bool
    stats_battery_live: bool


def assert_capability_live(
    probe_result: CapabilityProbeResult,
    *,
    campaign_mode: CampaignMode,
) -> None:
    """The composed AC-20 gate: refuse a confirmatory/sequential campaign start unless bathos
    capability is fully live.

    Args:
        probe_result: the already-probed liveness result (see module docstring -- this function
            does not call bathos itself).
        campaign_mode: the campaign's mode. Only `"confirmation"` and `"sequential"` are gated;
            `"exploration"` always proceeds (no-op), matching AC-20's own explicit carve-out.

    Raises:
        CapabilityProbeInputError: `campaign_mode` is not one of `"exploration"`,
            `"confirmation"`, `"sequential"`.
        CapabilityNotLiveError: `campaign_mode` is gated and `probe_result.seed_live` and/or
            `probe_result.stats_battery_live` is False -- the message names exactly which
            check(s) failed.
    """
    if campaign_mode not in _VALID_MODES:
        msg = (
            f"unrecognized campaign_mode {campaign_mode!r}; expected one of {sorted(_VALID_MODES)}"
        )
        raise CapabilityProbeInputError(msg)

    if campaign_mode not in _GATED_MODES:
        return

    missing = [
        name
        for name, live in (
            ("seed_live", probe_result.seed_live),
            ("stats_battery_live", probe_result.stats_battery_live),
        )
        if not live
    ]
    if missing:
        msg = (
            f"bathos capability not live for {campaign_mode!r} campaign start: "
            f"{', '.join(missing)} not live"
        )
        raise CapabilityNotLiveError(msg)


__all__ = [
    "CampaignMode",
    "CapabilityNotLiveError",
    "CapabilityProbeInputError",
    "CapabilityProbeResult",
    "assert_capability_live",
]
