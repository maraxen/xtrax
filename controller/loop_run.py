"""Error/retry policy + conclude-on-every-path guarantee (epic #3611 loop-controller, LC-11, AC-8c).

AC-8c's text (verbatim): "error/retry policy for partial-iteration failures + a task_id-bearing
telemetry/log hook (TRANSDUCTION convention), split out from an oversized original main-loop item
during adversarial review. AC: campaign_conclude (or an equivalent close-out call) fires on every
code path that exits the loop -- success, a caught per-candidate failure, or an uncaught exception
-- verified via a test that injects a failure mid-iteration (using LC-03's MockDispatchBackend
failure-injection modes) and asserts conclude still fires with an appropriate failure/aborted
status. This closes the exact audit-trail-corruption risk this epic's own brainstorm pre-mortem
predicted (partial failures leaving campaigns without a conclude call)." See
`.praxia/docs/specs/260716_loop-controller-epic-architecture-resolved.md` ("AC-8c", "finding 3").

This module provides one function, `run_campaign_loop`, which wraps LC-10's
`run_multi_iteration_loop` so that `campaign_adapter.campaign_conclude` is guaranteed to fire
exactly once on every path that exits the campaign -- success, a caught per-candidate dispatch
failure, or an uncaught exception -- and never silently swallows the original failure.

## Scoping decision: this item also owns `campaign_create` -- reasoning, not a guess

Neither LC-09 (`run_one_candidate_pass`) nor LC-10 (`run_multi_iteration_loop`) calls
`campaign_create` anywhere -- both explicitly scope themselves to "assume a campaign is already
open" (see both modules' own docstrings' "scoping decision" sections) and name `campaign_id` as a
required parameter. LC-10's own fixer flagged this explicitly during that PR: "`campaign_conclude`
is not called anywhere in this module... LC-11 must wrap `run_multi_iteration_loop` (or extend it)
to guarantee conclude fires on every exit path." AC-8c's own literal AC text only speaks to
"conclude fires on every path" -- it does not explicitly assign `campaign_create` to this item
either.

Checked directly (not assumed) whether some OTHER filed item claims `campaign_create`: queried the
live backlog for epic #3611 (`mcp__praxia__backlog list`). The only other item downstream of this
one is `[LC-12] AC-9 first real end-to-end integration smoke test` (backlog id 3622,
`depends_on: [AC-8a, AC-8b, AC-8c]`). Per the architecture spec's own AC-9 text (revised per
finding 12), LC-12's job is to *exercise* the already-built chain end-to-end against real
infrastructure ("one candidate that passes every gate, one candidate that fails a gate... at least
2 iterations") -- a testing/verification item, not a claim to own new production wiring. No filed
item anywhere in this epic calls `campaign_create`.

Practically: something has to open the campaign before `run_multi_iteration_loop` can use the
`campaign_id` it requires. Two options were weighed:

1. Leave `campaign_id` a required parameter here too (mirroring LC-10's own precedent exactly),
   pushing `campaign_create` to whichever future caller (LC-12, or a not-yet-designed CLI
   entrypoint) happens to invoke this function.
2. Make this function the single entry point that opens *and* guarantees-closes one campaign
   around one multi-iteration run.

(2) was chosen. Rationale: AC-8c is the item whose entire job is "guarantee campaign lifecycle
correctness around one run" -- splitting "open" into a different, not-yet-written caller while
this item owns "guarantee close" would recreate exactly the kind of ownership gap the epic's own
pre-mortem warned about (a lifecycle step with no clear owner). `BathosCampaignAdapter.campaign_
create`'s own docstring already says "Call once, at campaign start" -- this function is precisely
that start, immediately followed by the guaranteed-close region. LC-12 (the only other item that
could plausibly need this) is a *test* of this exact function, not a second production owner of
`campaign_create` -- there is no scope collision. If a future caller genuinely needs to attach to
an already-open campaign instead of creating a fresh one (e.g. resuming a crashed run), that is a
new, distinct capability this function does not claim to provide today, left to a future item
exactly as LC-10 left multi-iteration wiring to itself rather than guessing ahead.

## GW-03 addendum: campaign-approval standing gate wired in before `campaign_create` (T2-32, AC-25)

`campaign_approval_gate.assert_campaign_approved` (T2-32, AC-25, already built and merged) was
never wired into any production caller before this addendum -- verified directly: `grep` confirmed
`campaign_approval_gate` was imported only by its own test module anywhere in this repo. That
meant ANY caller of `run_campaign_loop` could start a real campaign against real bathos/praxia
infrastructure with zero human-approval check -- an ACTIVE safety gap (this exact unguarded path is
what LC-12's own real-infrastructure smoke test exercised to create real campaigns), not a latent
one, per the constitution's own policy text (`.praxia/docs/decisions/260714_2181-autoresearch-loop-
constitution.md`): "every campaign start requires Marielle's explicit approval before the campaign
may run... cannot be revoked by the loop under any circumstance."

Wired in as `campaign_approval_fn` (default: the real `assert_campaign_approved`), called with
`campaign_name` -- not `handle.campaign_id` -- as the approval-gate's own `campaign_id` argument.
This is a deliberate, load-bearing choice, not an oversight: `campaign_create` itself is what mints
the real bathos `campaign_id` (`envelope["campaign_id"]`, `bathos_campaign_adapter.py`), so no
bathos-side identifier exists yet at the point this gate must fire (strictly BEFORE
`campaign_create`, per [GW-03]'s own filing text). `campaign_name` is the only caller-known,
human-legible identifier available at that point -- and matches the gate module's own test fixtures
and TOML convention (`event_ref` values like `"campaign_001_initial"`, human-chosen slugs, not
bathos UUIDs). A human pre-approves a named campaign in `.praxia/loop_human_gates.toml` before this
function is ever invoked with that same name.

A `CampaignNotApprovedError` (or either subclass, `NoMatchingApprovalError`/`ApprovalExpiredError`)
fires before `campaign_create` -- no campaign is created, so nothing needs concluding, the same
"nothing to conclude" treatment as the `max_candidates < 1` `ValueError` (see the next section's
own note on this, extended below to cover this gate too).

The external watchdog's kill authority (already wired via LC-10/`start_watchdog`) remains available
regardless of approval status, unaffected by this change -- `campaign_approval_gate`'s own module
docstring is explicit that it is purely the "may this campaign START" half of AC-25; the separate,
always-available kill authority is AC-13's responsibility, never coupled to this gate.

**Operational requirement, not enforced by this code (audit finding, 2026-07-19):**
`assert_campaign_approved` matches on `(id="T2-32", event_ref==campaign_name)` plus TTL freshness
only -- it has no single-use consumption. One TOML entry therefore authorizes EVERY campaign start
using that same `campaign_name` for the entire `ttl_days` window, not just one. To preserve the
constitution's "every campaign start requires explicit approval" intent, `campaign_name` must be
minted uniquely per real, approved campaign start (e.g. a fresh random/timestamp suffix decided
*before* seeking approval, then approved and used exactly once) -- reusing an already-approved name
for a second, different campaign start silently rides the first approval. This module does not
mint or validate uniqueness itself; that responsibility sits with the caller.

## What "a caught per-candidate failure" means here, vs. "an uncaught exception" -- grounded, not
## invented

The AC's own text lists these as two distinct cases ("a caught per-candidate failure, **or** an
uncaught exception") implying different handling and potentially different `outcome_label`s. This
module draws that line at the `DispatchBackend` Protocol's own documented failure contract
(`controller/dispatch.py`'s docstring): `CandidateHandoffFailure`, `TimeoutError`, and `ValueError`
are the *only* exception types that Protocol promises a backend may raise from
`dispatch_candidate()` -- and, verified directly by grep, they are also the *only* place these
three types originate anywhere in the call chain `run_campaign_loop` -> `run_multi_iteration_loop`
-> `run_one_candidate_pass` sequences:

- `MockDispatchBackend`'s three failure-injection modes (`TIMEOUT`, `MALFORMED_COMPLETION`,
  `CANDIDATE_HANDOFF_FAILURE`) raise exactly these three types (`controller/dispatch.py`).
- `PraxiaDispatchBackend` (the real backend, LC-04) raises exactly the same three types for the
  same reasons (`controller/praxia_dispatch_backend.py`, grepped directly).
- Every other module in the chain -- `lineage_interim.py`, `bathos_campaign_adapter.py`,
  `stats_battery_gate.py`, `seed_gate.py`, `diversity_quota.py`, and (added by [GW-04]'s first
  slice) `xtrax.loop.candidate_static` -- raises its *own* distinct exception type
  (`MultiParentLineageUnsupportedError`, `BathosMcpToolError`/`BathosMcpTransportError`/
  `BathosTokenMissingError`, `StatsBatteryGateInputError`, `SeedGateInputError`,
  `DiversityQuotaInputError`, `CandidateStaticGateError`), none of which is a `ValueError`/
  `TimeoutError`/`CandidateHandoffFailure` subclass (all grepped and confirmed direct
  `Exception` subclasses) -- so `CandidateStaticGateError` falls to the same **uncaught
  exception** / `outcome_label="aborted"` bucket as the others in this list, with zero special-
  casing needed here.

So catching exactly `(CandidateHandoffFailure, TimeoutError, ValueError)` unambiguously identifies
"the dispatch step itself failed in one of its own documented ways" -- a **caught per-candidate
failure** -- concluded with `outcome_label="partial_failure"`. Everything else that is a normal
`Exception` (bathos MCP failures, gate-input errors, a genuine multi-parent-lineage rejection, a
malformed candidate source file for the diversity-quota window, or a genuine bug) is an
**uncaught exception** -- concluded with `outcome_label="aborted"`. `multi_iteration_loop.py`'s own
`max_candidates < 1` `ValueError` is deliberately never reachable inside this ambiguity: this
module re-validates `max_candidates` itself *before* calling `campaign_create` (see below), so
that specific `ValueError` propagates before any campaign exists, not through the categorized-vs-
uncaught branch at all. The same is true of a `CampaignNotApprovedError` (or either subclass) from
the campaign-approval gate ([GW-03], see its own addendum above) -- it, too, fires before
`campaign_create`, so it never reaches the categorized-vs-uncaught branch either.

**Why this is NOT literal automatic retry-and-continue.** The AC's own phrase "error/retry policy"
could mean "catch a per-candidate failure, retry that one candidate, and keep going" -- but that
capability does not exist at the granularity this function operates at. `run_multi_iteration_loop`
(LC-10, already merged, already tested -- see its own test suite's
`TestExceptionPropagatesAndWatchdogStillCleanedUp`) deliberately propagates every
`run_one_candidate_pass` exception immediately, aborting the *entire* multi-iteration run, not just
one candidate; it has no per-candidate retry hook. Building genuine single-candidate retry would
require either reopening LC-10's already-merged, already-tested contract (risking regressions this
item has no mandate to make) or duplicating its iteration loop here (diverging from the one tested
implementation). Retrying the *whole* multi-candidate run from scratch after a partial failure was
also considered and rejected: it would re-dispatch already-succeeded candidates a second time under
the same campaign, polluting bathos's own run history with duplicates -- worse than the failure it
would be working around. This module's actual "policy" is therefore a **classification** policy:
distinguish a known, documented dispatch-contract failure from a truly unexpected one, record that
distinction in the campaign's own `outcome_label`/conclusion text (so a human or downstream tooling
reviewing bathos campaign records can tell the two apart), and -- in both cases -- never swallow
the original exception. Genuine per-candidate retry is left as an explicitly out-of-scope future
capability for whichever item eventually grows `run_multi_iteration_loop` a retry hook.

## Never let a *secondary* conclude failure mask the *original* exception

Both failure branches call `campaign_adapter.campaign_conclude` from inside an `except` block. If
that call itself raises (e.g. bathos's MCP server is unreachable at the exact moment the run
failed), Python's normal exception-propagation would make *that* new exception the one escaping,
burying the original failure this whole item exists to make visible. `_conclude_best_effort` guards
against this explicitly: it catches any exception `campaign_conclude` itself raises, logs it loudly
(`_logger.exception`, plus a `campaign_conclude_failed` telemetry event) instead of letting it
propagate, so the enclosing `except` block's own bare `raise` always re-raises the exact original
exception object -- never a replacement, never silently dropped.

## Telemetry hook: a task_id-bearing callable, not a new logging subsystem

The AC's own text asks for "a task_id-bearing telemetry/log hook (TRANSDUCTION convention)". The
global TRANSDUCTION protocol (`~/.claude/CLAUDE.md`) requires every `transduction_log` call and
agent dispatch to carry a `task_id` (minted as `YYMMDD_<slug>` if none exists) -- but that
protocol's own `transduction_log` MCP tool is praxia-orchestration-scoped (agent sessions, sprint
work), not a dependency this repo's `controller/` production code should take on (grepped this
repo directly: no `task_id`-bearing log call or `transduction_log` reference exists anywhere in
`src/` or `controller/` before this item -- there is no existing in-repo convention to match
either). Rather than invent a heavier telemetry subsystem or bolt on an MCP dependency that doesn't
belong in this module, this follows LC-10's own precedent exactly: `on_leap_path_required` is
already an injectable `Callable[[LeapPathEvent], None]` hook with a structured-logging default.
`on_loop_event` here is the same pattern, one level up: a `Callable[[LoopEvent], None]` hook fired
at each campaign-lifecycle milestone (`campaign_created`, `campaign_concluded`,
`campaign_conclude_failed`), where every `LoopEvent` carries a `task_id` -- caller-supplied, or
minted via `_default_task_id` using the same `YYMMDD_<slug>` shape the TRANSDUCTION convention
already uses, scoped down to what one injectable log hook needs (a short, legible, collision-
resistant identifier for one campaign-loop run), not a claim that this *is* TRANSDUCTION logging.
The default handler (`_default_loop_event_handler`) does the one thing proportionate to a "hook":
structured logging via the standard `logging` module, so a human operator or downstream tooling can
grep a run's logs by `task_id` -- consistent with the "proportionate to what a hook implies" brief,
not an elaborate telemetry system.
"""

import logging
import re
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from controller.bathos_campaign_adapter import (
    BathosCampaignAdapter,
    CampaignConclusion,
    CampaignHandle,
)
from controller.bathos_library_wrappers import (
    call_stats_battery_gate,
    get_evidence_candidate_for_run,
    get_seed_trial_counts,
    get_sidecar_drift_signal,
)
from controller.dispatch import CandidateHandoffFailure, DispatchBackend
from controller.lineage_interim import CandidateParentage
from controller.main_loop import CampaignMode
from controller.multi_iteration_loop import (
    LeapPathEvent,
    MultiIterationLoopResult,
    _default_leap_path_handler,
    run_multi_iteration_loop,
)
from xtrax.devtools.freshness import Attestation
from xtrax.loop.attestation_evidence_gate import EvidenceCandidate
from xtrax.loop.campaign_approval_gate import DEFAULT_GATES_TOML, assert_campaign_approved
from xtrax.loop.candidate_static import assert_candidate_static
from xtrax.loop.external_stop_watchdog import WatchdogCriteria, WatchdogHandle, start_watchdog
from xtrax.loop.seed_gate import SeedTrialCounts
from xtrax.loop.sidecar_drift_gate import AgentMode as SidecarAgentMode
from xtrax.loop.sidecar_drift_gate import SidecarDriftSignal
from xtrax.loop.stats_battery_gate import BathosStatsBatteryVerdict

_logger = logging.getLogger(__name__)

# The DispatchBackend Protocol's own documented failure contract (controller/dispatch.py) --
# verified by grep to be the ONLY origin of these three exception types anywhere in the
# run_campaign_loop -> run_multi_iteration_loop -> run_one_candidate_pass call chain (see module
# docstring's "caught per-candidate failure" section). Anything else that is a plain `Exception`
# is classified as an uncaught exception instead.
_CAUGHT_PER_CANDIDATE_FAILURE_TYPES: tuple[type[Exception], ...] = (
    CandidateHandoffFailure,
    TimeoutError,
    ValueError,
)

LoopEventKind = Literal["campaign_created", "campaign_concluded", "campaign_conclude_failed"]


@dataclass(frozen=True, slots=True)
class LoopEvent:
    """One task_id-bearing campaign-lifecycle telemetry event (AC-8c's "telemetry/log hook").

    Attributes:
        task_id: identifies this `run_campaign_loop` invocation -- caller-supplied, or minted by
            `_default_task_id` (`YYMMDD_<slug>` shape, mirroring the TRANSDUCTION convention).
        kind: which lifecycle milestone this event marks.
        campaign_id: the bathos campaign this event concerns.
        outcome_label: the `outcome_label` passed to (or attempted on) `campaign_conclude` --
            `""` for `campaign_created` (no outcome yet).
    """

    task_id: str
    kind: LoopEventKind
    campaign_id: str
    outcome_label: str = ""


def _default_loop_event_handler(event: LoopEvent) -> None:
    """Default `on_loop_event` hook: structured logging, task_id-first (AC-8c's own text).

    See the module docstring's "Telemetry hook" section for why this is proportionate to what a
    "hook" implies, rather than an elaborate telemetry system.
    """
    _logger.info(
        "task_id=%s loop_event=%s campaign_id=%s outcome_label=%s",
        event.task_id,
        event.kind,
        event.campaign_id,
        event.outcome_label,
    )


@dataclass(frozen=True, slots=True)
class CampaignLoopResult:
    """The composed outcome of one successful `run_campaign_loop` call.

    Only returned on the success path -- both failure paths re-raise instead (see module
    docstring); there is no "partial" result to hand back on failure.

    Attributes:
        task_id: this run's task_id (caller-supplied or minted).
        campaign_id: the bathos campaign this run created and concluded.
        loop_result: the `MultiIterationLoopResult` from `run_multi_iteration_loop`.
        conclusion: the `CampaignConclusion` from the successful `campaign_conclude` call.
    """

    task_id: str
    campaign_id: str
    loop_result: MultiIterationLoopResult
    conclusion: CampaignConclusion


def _default_task_id(campaign_name: str) -> str:
    """Mint a TRANSDUCTION-convention task_id (`YYMMDD_<slug>`) when the caller supplies none.

    See the module docstring's "Telemetry hook" section: this reuses the one concrete, documented
    minting convention in this workspace (the global TRANSDUCTION protocol's own
    "mint one (`YYMMDD_<slug>`) if none exists" rule), scoped down to what a single injectable log
    hook needs -- a short, legible, collision-resistant identifier for one campaign-loop run. A
    random 8-hex-char suffix is appended (distinct from `CandidateHandoff`'s full-hash requirement
    -- this is a log-correlation id, not an integrity hash, so collision exposure is a much
    smaller concern) so two runs with the same campaign name on the same day still get distinct
    task_ids.
    """
    date_part = time.strftime("%y%m%d")
    slug = re.sub(r"[^a-z0-9]+", "-", campaign_name.strip().lower()).strip("-") or "campaign"
    suffix = uuid.uuid4().hex[:8]
    return f"{date_part}_{slug}-{suffix}"


def _conclude_best_effort(
    campaign_adapter: BathosCampaignAdapter,
    *,
    campaign_id: str,
    outcome_label: str,
    conclusion: str,
    task_id: str,
    on_loop_event: Callable[[LoopEvent], None],
) -> None:
    """Best-effort `campaign_conclude` call from a failure-handling `except` block.

    Deliberately swallows any exception `campaign_conclude` itself raises here (logging it loudly
    instead) so the caller's own bare `raise` -- re-raising the ORIGINAL exception the enclosing
    `except` block is handling -- is never masked by a *secondary* failure in this close-out call
    itself. See the module docstring's "Never let a secondary conclude failure mask the original
    exception" section.
    """
    try:
        campaign_adapter.campaign_conclude(campaign_id, outcome_label, conclusion=conclusion)
    except Exception:
        _logger.exception(
            "task_id=%s campaign_id=%s campaign_conclude itself failed while handling a %s "
            "outcome -- the ORIGINAL exception is still re-raised by the caller; this secondary "
            "failure is logged, not raised, so it cannot mask that original error",
            task_id,
            campaign_id,
            outcome_label,
        )
        on_loop_event(
            LoopEvent(
                task_id=task_id,
                kind="campaign_conclude_failed",
                campaign_id=campaign_id,
                outcome_label=outcome_label,
            )
        )
        return
    on_loop_event(
        LoopEvent(
            task_id=task_id,
            kind="campaign_concluded",
            campaign_id=campaign_id,
            outcome_label=outcome_label,
        )
    )


def run_campaign_loop(
    dispatch_backend: DispatchBackend,
    campaign_adapter: BathosCampaignAdapter,
    *,
    campaign_name: str,
    campaign_mode: CampaignMode,
    max_candidates: int,
    watchdog_criteria: WatchdogCriteria,
    campaign_question: str = "",
    campaign_hypothesis: str = "",
    parentage: CandidateParentage = CandidateParentage(),
    run_args: list[str] | None = None,
    output_paths: list[str] | None = None,
    tags: list[str] | None = None,
    agent_mode: str = "",
    no_sidecar: bool = False,
    stats_battery_kwargs: Mapping[str, Any],
    seed_trial_db: Any = None,
    hypothesis_clause_id: str = "",
    stats_battery_fn: Callable[..., BathosStatsBatteryVerdict] = call_stats_battery_gate,
    seed_trial_counts_fn: Callable[..., SeedTrialCounts] = get_seed_trial_counts,
    candidate_static_fn: Callable[..., None] = assert_candidate_static,
    candidate_static_root: Path | None = None,
    catalog_dir: str = "",
    evidence_candidate_fn: Callable[..., EvidenceCandidate] = get_evidence_candidate_for_run,
    stdout_verified: bool | None = None,
    sidecar_drift_signal_fn: Callable[..., SidecarDriftSignal] = get_sidecar_drift_signal,
    sidecar_drift_agent_mode: SidecarAgentMode = "collaborative",
    wall_clock_budget_seconds: float | None = None,
    time_fn: Callable[[], float] = time.monotonic,
    diversity_window_size: int = 5,
    on_leap_path_required: Callable[[LeapPathEvent], None] = _default_leap_path_handler,
    start_watchdog_fn: Callable[[int, WatchdogCriteria], WatchdogHandle] = start_watchdog,
    task_id: str = "",
    on_loop_event: Callable[[LoopEvent], None] = _default_loop_event_handler,
    campaign_approval_fn: Callable[..., Attestation] = assert_campaign_approved,
    campaign_approval_toml_path: Path = DEFAULT_GATES_TOML,
) -> CampaignLoopResult:
    """Open one bathos campaign, run the multi-iteration loop, and guarantee it concludes.

    Sequences `campaign_adapter.campaign_create` -> `run_multi_iteration_loop` (LC-10) ->
    `campaign_adapter.campaign_conclude`, with the conclude call guaranteed to fire on every path
    that exits after the campaign is created: normal completion, a caught per-candidate dispatch
    failure, or an uncaught exception. See the module docstring for the full grounding of every
    design choice below (campaign_create scoping, the caught-vs-uncaught classification, the
    secondary-failure guard, and the telemetry hook).

    Args:
        dispatch_backend: forwarded to `run_multi_iteration_loop` on every iteration.
        campaign_adapter: the `BathosCampaignAdapter` (LC-06) this function opens, runs, and
            concludes exactly one campaign through.
        campaign_name: forwarded to `campaign_adapter.campaign_create` as `name`.
        campaign_mode: forwarded to BOTH `campaign_adapter.campaign_create` (as `mode`) and
            `run_multi_iteration_loop` (as `campaign_mode`) -- a single mode governs both. Note
            bathos's `campaign_create` MCP tool currently rejects `"sequential"` (a known,
            documented bathos-side gap, AC-10) -- if the caller supplies it, `campaign_create`
            itself raises `BathosMcpToolError` *before* this function's guaranteed-conclude region
            begins (no campaign was created, so nothing needs concluding).
        max_candidates: forwarded to `run_multi_iteration_loop`. Validated (`>= 1`) by this
            function itself, before `campaign_create` is ever called -- see module docstring.
        watchdog_criteria: forwarded to `run_multi_iteration_loop`.
        campaign_question: forwarded to `campaign_adapter.campaign_create` as `question`.
        campaign_hypothesis: forwarded to `campaign_adapter.campaign_create` as `hypothesis`.
        parentage: forwarded to `run_multi_iteration_loop` unchanged.
        run_args: forwarded to `run_multi_iteration_loop` unchanged.
        output_paths: forwarded to `run_multi_iteration_loop` unchanged.
        tags: forwarded to `run_multi_iteration_loop` unchanged.
        agent_mode: forwarded to `run_multi_iteration_loop` unchanged.
        no_sidecar: forwarded to `run_multi_iteration_loop` unchanged.
        stats_battery_kwargs: forwarded to `run_multi_iteration_loop` unchanged.
        seed_trial_db: forwarded to `run_multi_iteration_loop` unchanged.
        hypothesis_clause_id: forwarded to `run_multi_iteration_loop` unchanged.
        stats_battery_fn: injection seam, forwarded to `run_multi_iteration_loop` (default: LC-07's
            real `call_stats_battery_gate`).
        seed_trial_counts_fn: injection seam, forwarded to `run_multi_iteration_loop` (default:
            LC-07's real `get_seed_trial_counts`).
        candidate_static_fn: injection seam, forwarded to `run_multi_iteration_loop` (default:
            T2-11's real `assert_candidate_static`).
        candidate_static_root: forwarded to `run_multi_iteration_loop` unchanged (jaxlint's
            subprocess root; default `None` lets jaxlint fall back to `Path.cwd()`).
        catalog_dir: forwarded to `run_multi_iteration_loop` unchanged (bathos catalog
            directory for `evidence_candidate_fn`/`sidecar_drift_signal_fn`).
        evidence_candidate_fn: injection seam, forwarded to `run_multi_iteration_loop`
            (default: [GW-01]'s real `get_evidence_candidate_for_run`).
        stdout_verified: forwarded to `run_multi_iteration_loop` unchanged.
        sidecar_drift_signal_fn: injection seam, forwarded to `run_multi_iteration_loop`
            (default: [GW-01]'s real `get_sidecar_drift_signal`).
        sidecar_drift_agent_mode: forwarded to `run_multi_iteration_loop` unchanged (default
            `"collaborative"` -- see `run_one_candidate_pass`'s own docstring).
        wall_clock_budget_seconds: forwarded to `run_multi_iteration_loop` unchanged.
        time_fn: injection seam, forwarded to `run_multi_iteration_loop` (default:
            `time.monotonic`).
        diversity_window_size: forwarded to `run_multi_iteration_loop` unchanged.
        on_leap_path_required: injection seam, forwarded to `run_multi_iteration_loop` (default:
            that module's own `_default_leap_path_handler`).
        start_watchdog_fn: injection seam, forwarded to `run_multi_iteration_loop` (default: the
            REAL `xtrax.loop.external_stop_watchdog.start_watchdog`). **Tests of this function
            must always override this** -- see `multi_iteration_loop.py`'s own module docstring
            for why the real one must never be reachable from a test.
        task_id: identifies this run for telemetry. `""` (default) mints one via
            `_default_task_id` (`YYMMDD_<slug>` shape).
        on_loop_event: injection seam -- invoked at each campaign-lifecycle milestone
            (`campaign_created`, `campaign_concluded`, `campaign_conclude_failed`). Defaults to
            `_default_loop_event_handler` (structured logging). See module docstring's "Telemetry
            hook" section.
        campaign_approval_fn: injection seam -- the human-approval gate (T2-32, AC-25; [GW-03])
            checked against `campaign_name` immediately before `campaign_create` is ever called.
            Defaults to the REAL `xtrax.loop.campaign_approval_gate.assert_campaign_approved`.
            **Tests of this function must always override this** with a stub returning a fake
            `Attestation` -- never let a test reach the real TOML-file-backed gate. See the module
            docstring's GW-03 addendum.
        campaign_approval_toml_path: forwarded to `campaign_approval_fn` as `toml_path` (default:
            `campaign_approval_gate.DEFAULT_GATES_TOML`, i.e. `.praxia/loop_human_gates.toml`).

    Returns:
        A `CampaignLoopResult` on successful completion (the only path that returns instead of
        raising).

    Raises:
        ValueError: `max_candidates < 1` -- raised before `campaign_create` is called; no
            campaign is created, so nothing needs concluding.
        CampaignNotApprovedError (NoMatchingApprovalError, ApprovalExpiredError): `campaign_name`
            has no fresh T2-32 approval entry in the gates file -- raised before `campaign_create`
            is called; no campaign is created, so nothing needs concluding (same treatment as the
            `max_candidates < 1` `ValueError` above; see the module docstring's GW-03 addendum).
        BathosMcpToolError, BathosMcpTransportError, BathosTokenMissingError: `campaign_create`
            itself failed -- raised before this function's guaranteed-conclude region begins.
        CandidateHandoffFailure, TimeoutError, ValueError: a caught per-candidate dispatch
            failure (see module docstring) -- `campaign_adapter.campaign_conclude` is called with
            `outcome_label="partial_failure"` before this is re-raised.
        Exception: any other exception raised by `run_multi_iteration_loop` (an uncaught
            exception, see module docstring) -- `campaign_adapter.campaign_conclude` is called
            with `outcome_label="aborted"` before this is re-raised.
    """
    if max_candidates < 1:
        msg = f"max_candidates must be >= 1, got {max_candidates}"
        raise ValueError(msg)

    resolved_task_id = task_id or _default_task_id(campaign_name)

    # Campaign-approval gate (T2-32, AC-25; [GW-03]) -- checked against campaign_name (the only
    # caller-known identifier available before bathos mints a real campaign_id) strictly before
    # campaign_create is ever called. See the module docstring's GW-03 addendum. A
    # CampaignNotApprovedError here (or either subclass) propagates before any campaign exists --
    # same "nothing to conclude" treatment as the max_candidates < 1 ValueError above.
    campaign_approval_fn(campaign_name, toml_path=campaign_approval_toml_path)

    handle: CampaignHandle = campaign_adapter.campaign_create(
        campaign_name,
        mode=campaign_mode,
        question=campaign_question,
        hypothesis=campaign_hypothesis,
    )
    campaign_id = handle.campaign_id
    on_loop_event(
        LoopEvent(task_id=resolved_task_id, kind="campaign_created", campaign_id=campaign_id)
    )

    try:
        loop_result = run_multi_iteration_loop(
            dispatch_backend,
            campaign_adapter,
            campaign_id=campaign_id,
            campaign_mode=campaign_mode,
            max_candidates=max_candidates,
            watchdog_criteria=watchdog_criteria,
            parentage=parentage,
            run_args=run_args,
            output_paths=output_paths,
            tags=tags,
            agent_mode=agent_mode,
            no_sidecar=no_sidecar,
            stats_battery_kwargs=stats_battery_kwargs,
            seed_trial_db=seed_trial_db,
            hypothesis_clause_id=hypothesis_clause_id,
            stats_battery_fn=stats_battery_fn,
            seed_trial_counts_fn=seed_trial_counts_fn,
            candidate_static_fn=candidate_static_fn,
            candidate_static_root=candidate_static_root,
            catalog_dir=catalog_dir,
            evidence_candidate_fn=evidence_candidate_fn,
            stdout_verified=stdout_verified,
            sidecar_drift_signal_fn=sidecar_drift_signal_fn,
            sidecar_drift_agent_mode=sidecar_drift_agent_mode,
            wall_clock_budget_seconds=wall_clock_budget_seconds,
            time_fn=time_fn,
            diversity_window_size=diversity_window_size,
            on_leap_path_required=on_leap_path_required,
            start_watchdog_fn=start_watchdog_fn,
        )
    except _CAUGHT_PER_CANDIDATE_FAILURE_TYPES as exc:
        _conclude_best_effort(
            campaign_adapter,
            campaign_id=campaign_id,
            outcome_label="partial_failure",
            conclusion=f"caught per-candidate dispatch failure ({type(exc).__name__}): {exc}",
            task_id=resolved_task_id,
            on_loop_event=on_loop_event,
        )
        raise
    except Exception as exc:
        _conclude_best_effort(
            campaign_adapter,
            campaign_id=campaign_id,
            outcome_label="aborted",
            conclusion=f"uncaught exception ({type(exc).__name__}): {exc}",
            task_id=resolved_task_id,
            on_loop_event=on_loop_event,
        )
        raise

    conclusion = campaign_adapter.campaign_conclude(
        campaign_id,
        "success",
        conclusion=(
            f"completed {len(loop_result.iterations)} candidate(s) "
            f"({loop_result.accepted_count} accepted), "
            f"termination_reason={loop_result.termination_reason}"
        ),
    )
    on_loop_event(
        LoopEvent(
            task_id=resolved_task_id,
            kind="campaign_concluded",
            campaign_id=campaign_id,
            outcome_label="success",
        )
    )
    return CampaignLoopResult(
        task_id=resolved_task_id,
        campaign_id=campaign_id,
        loop_result=loop_result,
        conclusion=conclusion,
    )


__all__ = [
    "CampaignLoopResult",
    "LoopEvent",
    "LoopEventKind",
    "run_campaign_loop",
]
