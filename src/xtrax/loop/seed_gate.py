"""Conclude-time seed/trial power-floor gate (T2-23, #2181, AC-16, conclude half).

AC-16's claim: "loop emits `Run.seed` + campaign mode; conclude enforces >=3 seeds per
`script_sha256` AND N >= 29 trials per `script_sha256` per hypothesis clause (power floor for
P(A>B)>0.75 at beta=0.05). Success: confirmatory conclude only with >=3 seeds and N>=29 per
clause. Fast/loud: cannot conclude 'held' -> hard block; exploration campaigns -> advisory
anomaly only." The emission half (`Run.seed` surfacing) lives in `xtrax.run.seed_emission` --
see that module's docstring for why this is split in two: emission is a per-run, run-layer
concern (mirrors `xtrax.run.component_binding`'s own emission precedent), while this module's
count-and-decide logic is a per-conclude, loop-layer concern (mirrors
`xtrax.loop.stats_battery_gate`'s own conclude-time precedent) -- the two happen at genuinely
different points in a campaign's lifecycle and have no shared state.

Grounding (verified before writing any code, not guessed): `depends_on: [T2-22, B2-02]`, both
merged. B2-02 (bathos PR #15, origin/main commit 99af1c7) added `Run.seed`,
`Run.baseline_hpo_trials`, `Run.baseline_hpo_compute_budget` to the schema, AND two conclude-time
counting helpers in `bathos/campaigns.py`: `count_seeds_for_script(db, script_sha256) -> int`
(distinct non-null seeds for that script) and `count_runs_for_script(db, script_sha256) -> int`
(total runs for that script) -- that commit's own message states these two functions exist
specifically so a future conclude-time gate (this module) can enforce ">=3-seed ICC replication"
and "N>=29-per-script_sha256". This module's `SeedTrialCounts` mirrors those two return values
exactly, plain caller-supplied integers -- the same seam as every other T2-2x gate
(`capability_probe_gate.CapabilityProbeResult`, `stats_battery_gate.BathosStatsBatteryVerdict`):
whatever obtains the real counts (a future loop controller calling those two bathos functions once
per `script_sha256`) does so outside this module; this module does not call bathos itself.

The "held" terminology in the DAG's fast/loud clause ("cannot conclude 'held'") is not invented
here -- `hypothesis_status == "held"` is an existing bathos postmortem verdict value
(`bathos/postmortem.py`, confirmed on origin/main), the value a confirmatory campaign's hypothesis
is marked when its claim was supported. This module's `held` field names that same concept: does
THIS clause's own seed/trial count clear AC-16's floor, prior to and independent of whatever
verdict bathos ultimately records.

"Per hypothesis clause": bathos's own multi-hypothesis-clause concept
(`ClaimFile.union_gate_clauses`, `bathos/claim.py`, confirmed on origin/main) is internal
bookkeeping this module does not resolve or enumerate -- it is not exposed via any bathos function
xtrax would call. Rather than inventing a collection/enumeration API this repo has no concrete
shape for yet, this module evaluates ONE `(script_sha256, hypothesis_clause_id)` pair per call,
mirroring `stats_battery_gate.assess_stats_battery_verdict`'s own one-verdict-per-call
granularity; a caller checking a whole campaign's clause set calls this once per clause.
`hypothesis_clause_id` is threaded through to the returned decision for logging/traceability only
-- it does not affect the floor computation, which is defined purely by `distinct_seed_count`/
`trial_count`, both already scoped to one `script_sha256` by the bathos counting functions above.

N>=29 grounding: the DAG's own text states this literal number with no derivation. The same
number traces to `.praxia/research/260702_nlm_prequery_roadmap.md:141-142` ("N=29 trials/splits
to detect P(A>B)>0.75 at beta=0.05"), the same research note `xtrax.loop.multi_metric_ratchet`
(T2-17) already cites for its own WR/BP/Cohen's-d thresholds. That note states the number without
a first-principles derivation of its own (no power-analysis formula shown in either the DAG text
or the research note) -- taken as given from both sources, not independently re-derived here.

Design point (shape (b), pure decision function -- mirrors
`stats_battery_gate.assess_stats_battery_verdict` and
`multi_metric_ratchet.compute_ratchet_decision`'s own stated rationale): not yet clearing the
seed/trial floor is a campaign's ordinary, expected mid-flight state (most campaigns start at
zero seeds/trials and accumulate towards 3/29 over their own lifetime) -- not a structural
anomaly the way a missing bathos capability (T2-27) is. Only a genuinely malformed
`campaign_mode`, or a structurally impossible count (negative, or a distinct-seed count exceeding
the total trial count -- COUNT(DISTINCT seed) can never exceed COUNT(*) over the same rows),
raises.

Extension seam (mirrors every other T2-2x gate's stance): this module computes and returns a
`SeedTrialFloorDecision`. It does not decide what a future loop controller does with a
`hard_blocked`/`advisory` decision beyond returning it, does not call bathos's counting functions
itself, and does not resolve which hypothesis clauses exist for a campaign.
"""

from dataclasses import dataclass
from typing import Literal

CampaignMode = Literal["exploration", "confirmation", "sequential"]

_VALID_MODES: frozenset[str] = frozenset({"exploration", "confirmation", "sequential"})

#: Campaign modes AC-16 hard-blocks on an unmet seed/trial floor. "exploration" is advisory-only.
_GATED_MODES: frozenset[str] = frozenset({"confirmation", "sequential"})

#: AC-16's own literal thresholds (see module docstring for the N>=29 grounding note).
MIN_SEEDS: int = 3
MIN_TRIALS: int = 29


class SeedGateInputError(Exception):
    """`campaign_mode` is not a recognized `CampaignMode`, or a count is malformed.

    Distinct from an unmet seed/trial floor (`SeedTrialFloorDecision.held=False`, not an
    exception) -- this is raised only when the inputs themselves are malformed.
    """


@dataclass(frozen=True, slots=True)
class SeedTrialCounts:
    """The already-computed per-`script_sha256` counts a caller obtained from bathos (mirrors
    `bathos.campaigns.count_seeds_for_script`/`count_runs_for_script`'s return values exactly --
    see module docstring). This module does not call bathos itself.

    Attributes:
        script_sha256: the script these counts are scoped to (matches both bathos counting
            functions' own scoping).
        distinct_seed_count: `count_seeds_for_script`'s return value -- distinct non-null seeds
            recorded across runs of `script_sha256`.
        trial_count: `count_runs_for_script`'s return value -- total runs recorded for
            `script_sha256`.
        hypothesis_clause_id: optional caller-supplied label for which hypothesis clause this
            `(script_sha256, counts)` pair belongs to (see module docstring's "per hypothesis
            clause" note). Carried through to the decision for logging only; defaults to `""`
            for callers with only a single, unlabeled clause.
    """

    script_sha256: str
    distinct_seed_count: int
    trial_count: int
    hypothesis_clause_id: str = ""


@dataclass(frozen=True, slots=True)
class SeedTrialFloorDecision:
    """The composed AC-16 decision for one `(script_sha256, hypothesis_clause_id)` pair at
    campaign conclude.

    Attributes:
        held: `True` iff `distinct_seed_count >= MIN_SEEDS` AND `trial_count >= MIN_TRIALS` --
            AC-16's own floor is cleared for this clause.
        hard_blocked: `True` iff not `held` AND `campaign_mode` is `"confirmation"`/
            `"sequential"` -- the campaign must not report this clause's hypothesis as `"held"`
            (bathos's own postmortem verdict term). A future loop controller enforces this; this
            module only decides (mirrors every other pure-decision T2-2x module's stance).
        advisory: `True` iff not `held` AND `campaign_mode == "exploration"` -- proceed, but the
            caller should surface the shortfall loud in the campaign report, per AC-16's own
            "fast/loud" framing.
        script_sha256: threaded through from the input, for logging.
        hypothesis_clause_id: threaded through from the input, for logging.
        distinct_seed_count: threaded through from the input, for logging.
        trial_count: threaded through from the input, for logging.
    """

    held: bool
    hard_blocked: bool
    advisory: bool
    script_sha256: str
    hypothesis_clause_id: str
    distinct_seed_count: int
    trial_count: int


def assess_seed_trial_floor(
    counts: SeedTrialCounts,
    *,
    campaign_mode: CampaignMode,
) -> SeedTrialFloorDecision:
    """The composed AC-16 gate: enforce >=3 seeds AND N>=29 trials per `script_sha256` per clause.

    Args:
        counts: the already-computed per-`script_sha256` seed/trial counts (see module docstring
            -- this function does not call bathos itself).
        campaign_mode: the campaign's mode. `"confirmation"`/`"sequential"` hard-block when the
            floor isn't cleared; `"exploration"` is advisory-only.

    Returns:
        A `SeedTrialFloorDecision`. An unmet floor is a normal outcome (the campaign simply
        hasn't accumulated enough seeds/trials yet), not an error -- this function never raises
        for it.

    Raises:
        SeedGateInputError: `campaign_mode` is not one of `"exploration"`, `"confirmation"`,
            `"sequential"`; `counts.distinct_seed_count`/`counts.trial_count` is negative; or
            `counts.distinct_seed_count` exceeds `counts.trial_count` (a structural
            impossibility -- a count of distinct seed values can never exceed the total row
            count they were drawn from).
    """
    if campaign_mode not in _VALID_MODES:
        msg = (
            f"unrecognized campaign_mode {campaign_mode!r}; expected one of {sorted(_VALID_MODES)}"
        )
        raise SeedGateInputError(msg)

    if counts.distinct_seed_count < 0 or counts.trial_count < 0:
        msg = (
            "distinct_seed_count and trial_count must be non-negative, got "
            f"distinct_seed_count={counts.distinct_seed_count}, trial_count={counts.trial_count}"
        )
        raise SeedGateInputError(msg)

    if counts.distinct_seed_count > counts.trial_count:
        msg = (
            "distinct_seed_count cannot exceed trial_count (a count of distinct seed values can "
            f"never exceed the total row count it was drawn from), got "
            f"distinct_seed_count={counts.distinct_seed_count}, trial_count={counts.trial_count}"
        )
        raise SeedGateInputError(msg)

    held = counts.distinct_seed_count >= MIN_SEEDS and counts.trial_count >= MIN_TRIALS
    hard_blocked = not held and campaign_mode in _GATED_MODES
    advisory = not held and not hard_blocked

    return SeedTrialFloorDecision(
        held=held,
        hard_blocked=hard_blocked,
        advisory=advisory,
        script_sha256=counts.script_sha256,
        hypothesis_clause_id=counts.hypothesis_clause_id,
        distinct_seed_count=counts.distinct_seed_count,
        trial_count=counts.trial_count,
    )


__all__ = [
    "MIN_SEEDS",
    "MIN_TRIALS",
    "CampaignMode",
    "SeedGateInputError",
    "SeedTrialCounts",
    "SeedTrialFloorDecision",
    "assess_seed_trial_floor",
]
