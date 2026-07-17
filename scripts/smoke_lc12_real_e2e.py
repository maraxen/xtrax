#!/usr/bin/env python3
"""Real end-to-end integration smoke test: controller vs. real praxia + real bathos (LC-12, AC-9).

AC-9's text (verbatim): "First real end-to-end integration smoke test against real praxia + real
bathos (no mocks). Must exercise, at minimum: one candidate that passes every gate, one candidate
that fails a gate and produces a needs_work/rejection verdict without corrupting campaign state,
and at least 2 iterations to exercise LC-10's multi-iteration path -- matching the rigor that
actually found 2 real bugs in xtrax PR #61's walking-skeleton smoke script, not a single lucky
happy-path run." See
`.praxia/docs/specs/260716_loop-controller-epic-architecture-resolved.md` ("AC-9", "finding 12")
and `.praxia/docs/roadmaps/loop-controller/260716_00-mandate.md`.

This is a ONE-OFF smoke-test script, mirroring `scripts/smoke_2181_walking_skeleton.py`'s own
precedent exactly: not a permanent module, not a pytest test, not registered as a CLI verb, not
wired into any CI job (`.github/workflows/ci.yml` was read directly before writing this: the
`wheel-smoke` job only greps a built wheel's file list for `port/`/`controller/` entries, and
`lint-format-type-test`'s `pytest` invocation only discovers files under `testpaths = ["tests",
"benchmarks"]` per `pyproject.toml` -- this file lives under `scripts/`, in neither path, and is
never imported by anything under those trees). Must be invoked deliberately, by a human, with the
`--i-understand-this-creates-real-bathos-state` flag (see below) -- there is no automatic trigger
anywhere in this repo that would run it.

## What this creates in REAL shared infrastructure -- read before running

This script calls `controller.loop_run.run_campaign_loop` (LC-11) twice against the REAL `bth-mcp`
server and the REAL, shared `~/.bth/catalog/bathos.db` catalog (the same catalog every other
bathos-using project on this machine shares) -- exactly two REAL bathos campaigns are created,
populated with real runs, and concluded. Every artifact this script creates is unambiguously
identifiable as a test fixture, never confusable with genuine research data:
  - Campaign names: `lc12-smoke-test-pass-<8-hex-suffix>` / `lc12-smoke-test-fail-<same-suffix>`.
  - `project_slug="xtrax-lc12-smoke"` on both the campaigns AND every run recorded under them
    (a project slug never used by any real xtrax/bathos research campaign).
  - Every run is tagged `["lc12-smoke-test", "epic-3611-lc12"]`.
No cleanup of the created campaigns/runs is performed by this script -- see the module's own
final report (printed at the end of a run, and the dispatching agent's own PR report) for the
exact `campaign_id`s so a human reviewer can inspect or delete them via `bth` directly.

## Design decisions grounded in real, verified constraints (not guessed)

1. **Two separate real campaigns, not one.** `run_campaign_loop`'s own docstring is explicit that
   a single `campaign_mode` value governs BOTH the real bathos campaign's `mode` (passed to
   `campaign_create`) AND every candidate's gate-assessment `campaign_mode` for the whole run.
   `xtrax.loop.stats_battery_gate`/`seed_gate` only hard-block under `"confirmation"`/
   `"sequential"` -- `"exploration"` is advisory-only for both gates, by design. Demonstrating a
   genuine, unambiguous "passes every gate" candidate (not one that merely isn't hard-blocked
   because the mode is lenient) needs a real stats-battery `verdict="pass"` (see calibration
   below); demonstrating a genuine hard-blocked rejection needs `campaign_mode="confirmation"`.
   Since one `run_campaign_loop` call can't mix both, this script makes two real, separate calls:
   one `"exploration"`-mode campaign (2 real candidates, genuinely strong stats, both accepted --
   also exercising LC-10's actual multi-iteration "keep going" loop) and one `"confirmation"`-mode
   campaign (1 real candidate, genuinely weak/indistinguishable-from-baseline stats, hard-blocked
   -- the AC's "needs_work/rejection verdict" case). `mode="sequential"` is never used anywhere in
   this script: confirmed directly against `bathos/src/bathos/mcp.py::campaign_create_tool`
   (`if mode not in ("exploration", "confirmation"): return {"error": ...}`) that bathos's own MCP
   tool rejects it outright -- a known, already-documented cross-repo gap (AC-10/LC-13), not
   something this script works around.

2. **A real seed/trial-floor "held=True" is not attempted for the passing candidate.** AC-16's own
   floor (`xtrax.loop.seed_gate.MIN_SEEDS=3`, `MIN_TRIALS=29`) requires 29 real prior runs of the
   IDENTICAL script_sha256 with >=3 distinct seeds to ever clear for a brand-new candidate script
   -- creating that much real state (29+ bathos run rows) purely to satisfy one gate in a smoke
   test would be disproportionate real-infrastructure footprint for what this item asks for, and
   would work against the "do not run this more than once or twice" hygiene rule this script was
   built under. The passing candidate's "every gate" claim is therefore: a REAL, genuinely
   confirmatory stats-battery verdict (`verdict="pass"`, not a mode-driven pass-through) PLUS a
   real, correctly-queried (but expectedly unmet, for a fresh script) seed/trial floor that is
   advisory-only under `"exploration"` mode by the gate's own design -- exactly the gate's intended
   behavior for an exploration-stage campaign, not a loophole. This is stated explicitly, not
   glossed over, per the fixer brief's own "actually check this" instruction.

3. **`PraxiaDispatchBackend` (LC-05) is the real `DispatchBackend`, used via its documented AC-4/1B
   fallback** (`attempt_primary_path=False`, the class's own default): the primary `write_staged_
   file` MCP path is confirmed unreachable in praxia's current merged state (that module's own
   docstring, and `.praxia/docs/research/260716_praxia-mcp-invocation-probe.md`) -- a known, filed
   cross-repo gap (debt id 746), not something this script forces around. The fallback still
   performs a REAL controller-side staging write + real SHA-256 self-attestation; it is real
   production code, not a test double. `PraxiaDispatchBackend.dispatch_candidate()` takes no
   arguments (content is fixed at construction, matching `DispatchBackend`'s own Protocol), so
   exercising >=2 genuinely different real candidates needs a small rotation wrapper
   (`_RotatingRealDispatchBackend` below) around several real `PraxiaDispatchBackend` instances --
   the wrapper only chooses which real backend answers each call; every `dispatch_candidate()`
   call is answered by real, production dispatch logic.

4. **Candidate scripts are deliberately silent (no stdout output).** Verified directly against
   `bathos/src/bathos/runner.py::run_script`: the candidate subprocess is spawned via
   `subprocess.Popen(argv, cwd=cwd, env=env)` with NO `stdout=`/`stderr=` redirection, so it
   inherits the `bth-mcp` server process's own stdout -- the exact file descriptor this script's
   own MCP client reads the JSON-RPC response stream from, line by line. A candidate that printed
   anything would race its own output against bathos's JSON-RPC response line on the same pipe,
   corrupting `_call_mcp_tool`'s `json.loads(line)` parse. This is a real, verified constraint of
   bathos's current MCP wiring (out of scope to fix here, cross-repo), worked around by making
   every candidate script a silent, side-effect-free, single-statement computation.

5. **No long-lived DuckDB connection is held across a real bathos MCP write.** DuckDB grants a
   read-only connection a shared file lock; a read-write connection needs an exclusive lock that
   cannot be granted while any shared lock is held elsewhere. Holding this script's own read-only
   catalog connection open for the whole run (naive approach) would risk blocking/breaking
   `bth-mcp`'s own real write connections for `run`/`campaign_create`/`campaign_conclude` --
   corrupting the very campaign state this item exists to safely create. `_short_lived_seed_trial_
   counts` (an injected `seed_trial_counts_fn`, still calling LC-07's real `get_seed_trial_counts`
   underneath -- not a mock) opens and closes a fresh read-only connection per call instead.

6. **`python` (bare) does not exist on PATH in this environment** (`which python` returns nothing;
   only `python3` and version-suffixed interpreters are installed) -- but bathos's real `run` tool
   hardcodes `argv = ["python", script_path] + args` (`bathos/src/bathos/mcp.py::run_tool`, not
   configurable by this caller). Unpatched, every real bathos `run` call in this script would fail
   at the subprocess-spawn layer before ever reaching the gate-check logic this smoke test exists
   to exercise. `_ensure_python_shim_on_path` fixes this locally (a `python -> sys.executable`
   symlink in a scratch dir, prepended to `PATH` for this process only), inherited by the `bth-mcp`
   subprocess this script spawns and then by the candidate subprocess *that* process spawns.

## Coherence check, not just "no exception" (per the fixer brief's explicit instruction)

After the `"confirmation"`-mode campaign concludes, this script independently re-reads the REAL
bathos catalog (a fresh, short-lived read-only DuckDB connection, opened after all real writes for
this script are done) and asserts: the campaign's own row is `status="concluded"` with a non-null
`concluded_at` and the expected `outcome_label`; exactly one `runs` row is attached to that
`campaign_id` (no orphaned/duplicate record); and that run's own `status="completed"` (not stuck
`"running"`). This is a genuine, independent readback of real state, not a re-assertion of
in-process return values.

## outcome_label is a campaign-lifecycle signal, not the candidate's own gate verdict

`run_campaign_loop` concludes with `outcome_label="success"` whenever `run_multi_iteration_loop`
returns without raising -- this is true for BOTH campaigns here, including the `"confirmation"`
one whose single candidate is gate-rejected: a rejected candidate is a normal, non-exceptional
`OneCandidatePassResult.accepted=False` value, not a caught/uncaught failure (see
`controller/main_loop.py`'s own docstring: gate checks return decisions, they do not raise). The
AC's "needs_work/rejection verdict" therefore lives in `OneCandidatePassResult.accepted`/
`GateOutcome.hard_blocked`, checked explicitly by this script below -- NOT in the bathos campaign's
own `outcome_label`, which correctly reports "the controller ran this campaign to completion
without an error," a true and distinct claim from "the candidate's hypothesis was confirmed."
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controller.bathos_campaign_adapter import BathosCampaignAdapter  # noqa: E402
from controller.bathos_library_wrappers import get_seed_trial_counts  # noqa: E402
from controller.dispatch import CandidateHandoff  # noqa: E402
from controller.loop_run import CampaignLoopResult, run_campaign_loop  # noqa: E402
from controller.praxia_dispatch_backend import PraxiaDispatchBackend  # noqa: E402
from xtrax.loop.external_stop_watchdog import WatchdogCriteria  # noqa: E402
from xtrax.loop.seed_gate import SeedTrialCounts  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("smoke_lc12_real_e2e")

_RUN_SUFFIX = uuid.uuid4().hex[:8]
_DATE_PART = time.strftime("%y%m%d")
_PROJECT_SLUG = "xtrax-lc12-smoke"
_PASS_CAMPAIGN_NAME = f"lc12-smoke-test-pass-{_RUN_SUFFIX}"
_FAIL_CAMPAIGN_NAME = f"lc12-smoke-test-fail-{_RUN_SUFFIX}"
_PASS_TASK_ID = f"{_DATE_PART}_lc12-real-e2e-pass-{_RUN_SUFFIX}"
_FAIL_TASK_ID = f"{_DATE_PART}_lc12-real-e2e-fail-{_RUN_SUFFIX}"
_TAGS = ["lc12-smoke-test", "epic-3611-lc12"]
_CATALOG_PATH = Path.home() / ".bth" / "catalog" / "bathos.db"

# Calibrated directly against the REAL bathos.stats_gates.run_stats_battery before trusting these
# for a test's pass/fail signal (BATHOS.md's "verify your measurement pipeline" rule) -- not
# assumed. Confirmed via a throwaway interpreter call (not persisted): candidate consistently
# ~2x baseline with low variance -> verdict="pass" (cohens_d=28.58, win_rate=1.0,
# p_superiority=1.0, wilcoxon_p=0.00049, breakdown_point=0.917); candidate statistically
# indistinguishable from baseline -> verdict="confounded" (cohens_d=-0.08, win_rate=0.5,
# p_superiority=0.476, wilcoxon_p=0.882, breakdown_point=0.0).
_GOOD_CANDIDATE_VALUES = [10.0, 10.2, 9.8, 10.1, 9.9, 10.3, 9.7, 10.0, 10.2, 9.9, 10.1, 10.0]
_GOOD_BASELINE_VALUES = [5.0, 5.2, 4.8, 5.1, 4.9, 5.3, 4.7, 5.0, 5.2, 4.9, 5.1, 5.0]
_BAD_CANDIDATE_VALUES = [5.1, 4.8, 5.3, 4.7, 5.2, 4.9, 5.0, 5.4, 4.6, 5.1, 4.9, 5.0]
_BAD_BASELINE_VALUES = [5.0, 5.2, 4.8, 5.1, 4.9, 5.3, 4.7, 5.0, 5.2, 4.9, 5.1, 5.0]


def _step(description: str) -> None:
    log.info("PASS: %s", description)


def _ensure_python_shim_on_path(shim_dir: Path) -> None:
    """Make bare ``python`` resolve to this interpreter for every subprocess this script spawns.

    See module docstring point 6: real bathos hardcodes ``argv = ["python", script_path]`` for
    every candidate run, but this environment has no bare ``python`` on PATH.
    """
    shim = shim_dir / "python"
    shim.symlink_to(sys.executable)
    os.environ["PATH"] = f"{shim_dir}{os.pathsep}{os.environ.get('PATH', '')}"


def _candidate_source(label: str, index: int) -> str:
    """A trivial, silent, side-effect-free candidate script. See module docstring point 4 for why
    it must never write to stdout.
    """
    return (
        f'"""LC-12 real-e2e smoke candidate ({label} #{index}) -- throwaway test artifact, not '
        "real research code. Deliberately produces no stdout output (see "
        'scripts/smoke_lc12_real_e2e.py\'s own module docstring, point 4, for why)."""\n\n'
        f"_value = {index} + 1  # trivial, silent computation only\n"
    )


class _RotatingRealDispatchBackend:
    """Cycles through several REAL `PraxiaDispatchBackend` instances, one per candidate.

    `PraxiaDispatchBackend.dispatch_candidate()` takes zero arguments (per `DispatchBackend`'s own
    Protocol) -- its candidate content is fixed at construction. Exercising a genuine
    multi-candidate run with textually distinct real candidates therefore needs this small
    rotation wrapper; every actual `dispatch_candidate()` call below is answered by a real
    `PraxiaDispatchBackend` running its real (AC-4/1B fallback) staging-write logic -- this wrapper
    only chooses which real backend answers the current call, it implements no dispatch/staging
    logic of its own.
    """

    def __init__(self, backends: list[PraxiaDispatchBackend]) -> None:
        self._backends = backends
        self._index = 0

    def dispatch_candidate(self) -> CandidateHandoff:
        backend = self._backends[self._index]
        self._index += 1
        return backend.dispatch_candidate()


def _short_lived_seed_trial_counts(
    _ignored_db: object, script_sha256: str, hypothesis_clause_id: str = ""
) -> SeedTrialCounts:
    """Injected `seed_trial_counts_fn`: real LC-07 `get_seed_trial_counts`, safe connection
    lifetime.

    See module docstring point 5: opens and closes a FRESH read-only DuckDB connection for this
    one query instead of holding one open across a real bathos MCP write, to avoid a real
    file-locking conflict with `bth-mcp`'s own write connections. `_ignored_db` exists only to
    match `get_seed_trial_counts`'s own positional call signature (`run_one_candidate_pass` always
    calls `seed_trial_counts_fn(seed_trial_db, ...)`); the real query logic below is LC-07's own
    `get_seed_trial_counts`, unmodified -- only the connection's lifetime differs from the
    default.
    """
    db = duckdb.connect(str(_CATALOG_PATH), read_only=True)
    try:
        return get_seed_trial_counts(db, script_sha256, hypothesis_clause_id)
    finally:
        db.close()


def _run_pass_campaign(
    workspace_root: Path, wall_clock_budget_seconds: float
) -> CampaignLoopResult:
    """Real `"exploration"`-mode campaign: 2 real candidates, genuinely strong stats (verdict
    literally `"pass"`), both accepted -- exercises LC-10's real multi-iteration "keep going" path.
    """
    backend = _RotatingRealDispatchBackend(
        [
            PraxiaDispatchBackend(
                _candidate_source("pass", i),
                staging_subdir="lc12_smoke_candidates",
                workspace_root=workspace_root,
            )
            for i in range(2)
        ]
    )
    campaign_adapter = BathosCampaignAdapter(project_slug=_PROJECT_SLUG)
    return run_campaign_loop(
        backend,
        campaign_adapter,
        campaign_name=_PASS_CAMPAIGN_NAME,
        campaign_mode="exploration",
        max_candidates=2,
        watchdog_criteria=WatchdogCriteria(wall_clock_budget_seconds=wall_clock_budget_seconds),
        campaign_question=(
            "LC-12 (backlog id 3622) real end-to-end smoke test: does a genuinely well-separated "
            "candidate clear the real stats-battery gate end-to-end against real bathos, across "
            "a genuine >=2-candidate multi-iteration run?"
        ),
        campaign_hypothesis=(
            "A candidate consistently ~2x baseline with low variance produces a real "
            "verdict='pass' from bathos.stats_gates.run_stats_battery."
        ),
        tags=list(_TAGS),
        no_sidecar=True,
        stats_battery_kwargs={
            "candidate_values": _GOOD_CANDIDATE_VALUES,
            "baseline_values": _GOOD_BASELINE_VALUES,
            "higher_is_better": True,
        },
        seed_trial_db=None,
        seed_trial_counts_fn=_short_lived_seed_trial_counts,
        hypothesis_clause_id="lc12-pass-clause",
        task_id=_PASS_TASK_ID,
    )


def _run_fail_campaign(
    workspace_root: Path, wall_clock_budget_seconds: float
) -> CampaignLoopResult:
    """Real `"confirmation"`-mode campaign: 1 real candidate, genuinely indistinguishable-from-
    baseline stats -- hard-blocked (the AC's "needs_work/rejection verdict" case), while the real
    bathos run itself still succeeds and the campaign still concludes cleanly.
    """
    backend = PraxiaDispatchBackend(
        _candidate_source("fail", 0),
        staging_subdir="lc12_smoke_candidates",
        workspace_root=workspace_root,
    )
    campaign_adapter = BathosCampaignAdapter(project_slug=_PROJECT_SLUG)
    return run_campaign_loop(
        backend,
        campaign_adapter,
        campaign_name=_FAIL_CAMPAIGN_NAME,
        campaign_mode="confirmation",
        max_candidates=1,
        watchdog_criteria=WatchdogCriteria(wall_clock_budget_seconds=wall_clock_budget_seconds),
        campaign_question=(
            "LC-12 (backlog id 3622) real end-to-end smoke test: does a genuinely "
            "indistinguishable-from-baseline candidate correctly hard-block under a real "
            "confirmation-mode campaign, without corrupting real campaign state?"
        ),
        campaign_hypothesis=(
            "A candidate statistically indistinguishable from baseline produces a real "
            "downgraded verdict (bathos.stats_gates.run_stats_battery) that hard-blocks under "
            "campaign_mode='confirmation'."
        ),
        tags=list(_TAGS),
        no_sidecar=True,
        stats_battery_kwargs={
            "candidate_values": _BAD_CANDIDATE_VALUES,
            "baseline_values": _BAD_BASELINE_VALUES,
            "higher_is_better": True,
        },
        seed_trial_db=None,
        seed_trial_counts_fn=_short_lived_seed_trial_counts,
        hypothesis_clause_id="lc12-fail-clause",
        task_id=_FAIL_TASK_ID,
    )


def _check_campaign_coherence(
    campaign_id: str, *, expected_run_count: int, expected_outcome_label: str
) -> None:
    """Independently re-read the REAL bathos catalog and assert the campaign+run state is
    coherent -- not merely "the code returned without raising" (per the fixer brief's explicit
    instruction). Opened AFTER all real writes for this script are complete (module docstring
    point 5: never held open across a real bathos MCP write).
    """
    db = duckdb.connect(str(_CATALOG_PATH), read_only=True)
    try:
        campaign_rows = db.execute(
            "SELECT status, outcome_label, concluded_at FROM campaigns WHERE id = ?",
            [campaign_id],
        ).fetchall()
        if not campaign_rows:
            msg = f"campaign {campaign_id} not found in the real bathos catalog after conclude"
            raise AssertionError(msg)
        status, outcome_label, concluded_at = campaign_rows[0]
        if status != "concluded":
            msg = f"campaign {campaign_id} status={status!r}, expected 'concluded'"
            raise AssertionError(msg)
        if outcome_label != expected_outcome_label:
            msg = (
                f"campaign {campaign_id} outcome_label={outcome_label!r}, expected "
                f"{expected_outcome_label!r}"
            )
            raise AssertionError(msg)
        if not concluded_at:
            msg = f"campaign {campaign_id} has no concluded_at timestamp"
            raise AssertionError(msg)

        run_rows = db.execute(
            "SELECT id, status FROM runs WHERE campaign_id = ?",
            [campaign_id],
        ).fetchall()
        if len(run_rows) != expected_run_count:
            msg = (
                f"campaign {campaign_id} has {len(run_rows)} run(s) recorded, expected "
                f"{expected_run_count} -- possible orphaned or missing run record"
            )
            raise AssertionError(msg)
        for run_id, run_status in run_rows:
            if run_status != "completed":
                msg = (
                    f"run {run_id} under campaign {campaign_id} has status={run_status!r}, "
                    "expected 'completed' -- a stuck or half-written run record"
                )
                raise AssertionError(msg)
    finally:
        db.close()


def _in_scratch_dir(
    wall_clock_budget_seconds: float,
    runner: Callable[[Path, float], CampaignLoopResult],
) -> CampaignLoopResult:
    """Run `runner` (one of the two `_run_*_campaign` functions) chdir'd into a fresh scratch
    tempdir, restoring the original cwd afterward. See module docstring point 4's neighboring
    concern: keeps every incidental side-effect directory (`.praxia/...`, `outputs/...`) bathos's
    real `run` creates out of this repo's own working tree.
    """
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="lc12_smoke_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        shim_dir = tmp_dir / "shim"
        shim_dir.mkdir()
        _ensure_python_shim_on_path(shim_dir)
        os.chdir(tmp_dir)
        try:
            return runner(tmp_dir, wall_clock_budget_seconds)
        finally:
            os.chdir(original_cwd)


def _assert_pass_campaign_result(result: CampaignLoopResult) -> None:
    iterations = result.loop_result.iterations
    if len(iterations) != 2:
        msg = f"expected 2 real iterations in the pass campaign, got {len(iterations)}"
        raise AssertionError(msg)
    for i, iteration in enumerate(iterations):
        if not iteration.run_result.success:
            msg = f"pass-campaign candidate {i} real bathos run did not succeed"
            raise AssertionError(msg)
        if not iteration.gate_outcome.stats_battery.honored:
            msg = (
                f"pass-campaign candidate {i} stats-battery verdict was not honored "
                f"(genuine 'pass' expected): {iteration.gate_outcome.stats_battery!r}"
            )
            raise AssertionError(msg)
        if iteration.gate_outcome.hard_blocked:
            msg = f"pass-campaign candidate {i} was unexpectedly hard-blocked"
            raise AssertionError(msg)
        if not iteration.accepted:
            msg = f"pass-campaign candidate {i} was unexpectedly not accepted"
            raise AssertionError(msg)
    _step(
        f"real 'exploration' campaign {result.campaign_id!r} ({_PASS_CAMPAIGN_NAME}): 2/2 real "
        "candidates genuinely pass every gate (verdict='pass', accepted=True) -- LC-10's "
        "multi-iteration path genuinely exercised"
    )


def _assert_fail_campaign_result(result: CampaignLoopResult) -> None:
    iterations = result.loop_result.iterations
    if len(iterations) != 1:
        msg = f"expected 1 real iteration in the fail campaign, got {len(iterations)}"
        raise AssertionError(msg)
    failing = iterations[0]
    if not failing.run_result.success:
        msg = (
            "fail-campaign candidate's real bathos run itself failed "
            "(expected the GATE to reject it, not the run)"
        )
        raise AssertionError(msg)
    if not failing.gate_outcome.hard_blocked:
        msg = "fail-campaign candidate was NOT hard-blocked -- calibration regressed"
        raise AssertionError(msg)
    if failing.accepted:
        msg = "fail-campaign candidate was unexpectedly accepted"
        raise AssertionError(msg)
    _step(
        f"real 'confirmation' campaign {result.campaign_id!r} ({_FAIL_CAMPAIGN_NAME}): the real "
        "candidate is genuinely gate-rejected (hard_blocked=True, accepted=False) -- a "
        "needs_work/rejection-shaped verdict, produced by a real, successful bathos run "
        f"(exit_code={failing.run_result.exit_code})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--i-understand-this-creates-real-bathos-state",
        action="store_true",
        dest="confirmed",
        help=(
            "Required. Confirms you understand this creates two real campaigns in the real, "
            "shared ~/.bth/catalog/bathos.db -- see this script's own module docstring."
        ),
    )
    parser.add_argument("--wall-clock-budget-seconds", type=float, default=180.0)
    args = parser.parse_args(argv)

    if not args.confirmed:
        parser.error(
            "refusing to run without --i-understand-this-creates-real-bathos-state "
            "(this script creates real state in the real, shared bathos catalog -- read the "
            "module docstring first)"
        )

    pass_result = _in_scratch_dir(args.wall_clock_budget_seconds, _run_pass_campaign)
    _assert_pass_campaign_result(pass_result)

    fail_result = _in_scratch_dir(args.wall_clock_budget_seconds, _run_fail_campaign)
    _assert_fail_campaign_result(fail_result)

    _check_campaign_coherence(
        fail_result.campaign_id, expected_run_count=1, expected_outcome_label="success"
    )
    _step(
        f"real bathos catalog readback confirms campaign {fail_result.campaign_id!r} is "
        "coherent after the gate-rejected candidate: status='concluded', exactly 1 run "
        "recorded, that run's own status='completed' (no orphaned/half-written record)"
    )

    _check_campaign_coherence(
        pass_result.campaign_id, expected_run_count=2, expected_outcome_label="success"
    )
    _step(
        f"real bathos catalog readback confirms campaign {pass_result.campaign_id!r} is "
        "coherent: status='concluded', exactly 2 runs recorded, both status='completed'"
    )

    log.info("")
    log.info("LC-12 REAL END-TO-END SMOKE TEST PASSED")
    log.info("pass campaign_id=%s name=%s", pass_result.campaign_id, _PASS_CAMPAIGN_NAME)
    log.info("fail campaign_id=%s name=%s", fail_result.campaign_id, _FAIL_CAMPAIGN_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
