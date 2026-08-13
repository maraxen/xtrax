"""Tests for controller.loop_run (LC-11, epic #3611, AC-8c).

AC-8c's own measurable criterion: "campaign_conclude (or an equivalent close-out call) fires on
every code path that exits the loop -- success, a caught per-candidate failure, or an uncaught
exception -- verified via a test that injects a failure mid-iteration (using LC-03's
MockDispatchBackend failure-injection modes) and asserts conclude still fires with an appropriate
failure/aborted status." Exercises:

1. Success path: `campaign_create` -> N real candidate passes -> `campaign_conclude` with
   `outcome_label="success"` -- verified at the wire level via a recording MCP transport, not just
   "the adapter's method was called."
2. Caught per-candidate failure: all three of `MockDispatchBackend`'s failure-injection modes
   (`TIMEOUT`, `MALFORMED_COMPLETION`, `CANDIDATE_HANDOFF_FAILURE`) each (a) still call
   `campaign_conclude` with `outcome_label="partial_failure"`, and (b) still re-raise the original
   exception type.
3. Uncaught exception: a genuine `MultiParentLineageUnsupportedError` (real production code, LC-08
   -- not a synthetic exception type) still calls `campaign_conclude` with
   `outcome_label="aborted"`, and still re-raises.
4. Proves (2) and (3) produce genuinely DISTINCT `outcome_label`s, not the same value for every
   failure.
5. A secondary failure inside `campaign_conclude` itself (during failure handling) never masks the
   ORIGINAL exception -- the original still propagates, and the secondary failure is only logged.
6. `campaign_create` is called exactly once per run, including on every failure path (proving this
   item's own campaign-lifecycle-ownership scoping decision is real, not just documented).
7. `max_candidates < 1` raises before `campaign_create` is ever called -- no orphaned campaign for
   a pure input-validation error.
8. The watchdog `run_multi_iteration_loop` starts is still stopped even when this wrapper's own
   failure handling runs (end-to-end confidence on top of LC-10's own dedicated watchdog tests).
9. `task_id`: caller-supplied value is used verbatim; omitted value is auto-minted in the
   `YYMMDD_<slug>-<hex>` shape.
10. `on_loop_event` receives every expected milestone, in order, each carrying the run's `task_id`.

GW-02 (#3649) + backlog #4203 additions -- `run_campaign_loop` now requires the identical 10 new
values `run_multi_iteration_loop` gained (see `test_multi_iteration_loop.py`'s own module
docstring for the full #4203 addendum this mirrors verbatim): `higher_is_better`/`frozen_context`/
`current_config`/`repo`/`ratchet_ref_name`/`callable_name`/`concrete_inputs` (required), plus
optional `commit_tree_sha`/`candidate_target_path`/`allow_fresh_start_despite_existing_lineage`.
Every `run_campaign_loop(` call site in this file now supplies these via the shared
`_base_kwargs()` helper. As in `test_multi_iteration_loop.py`, `guarded_evaluate_fn`/
`measure_two_phase_timing_fn`/`commit_parent_sha` are NOT threaded through this outer layer
(out of #4203's own scope), so every candidate that reaches a full pass here exercises the REAL
`guarded_evaluate`/`measure_two_phase_timing` defaults -- `_base_kwargs()`'s `frozen_context`
carries a genuinely-computed `ClosureManifest` + monotonically-improving `score_fn`,
`_RealFileDispatchBackend` appends a fixed `jax.jit`-safe `timing_probe(a, b)` to every written
candidate, and the autouse `_stub_crash_atomicity` fixture's `read_best_so_far` returns a FIXED
non-`None` SHA paired with `allow_fresh_start_despite_existing_lineage=True` in `_base_kwargs()`.

## Watchdog safety (mandatory reading for anyone editing this file)

Mirrors `tests/controller/test_multi_iteration_loop.py`'s own warning verbatim:
`xtrax.loop.external_stop_watchdog.start_watchdog` is a REAL function that spawns a REAL, detached
OS subprocess that SIGKILLs a target PID once a wall-clock budget elapses. **No test in this file
calls it.** Every test injects `start_watchdog_fn=<a _FakeWatchdogStarter instance>`.
"""

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

import controller.loop_run as loop_run_module
import controller.main_loop as main_loop_module
import controller.multi_iteration_loop as multi_iteration_loop_module
from controller.bathos_campaign_adapter import BathosCampaignAdapter, BathosMcpToolError
from controller.dispatch import (
    CandidateHandoff,
    CandidateHandoffFailure,
    MockDispatchBackend,
    MockFailureMode,
)
from controller.evaluate_adapter import BathosFrozenContext
from controller.lineage_interim import CandidateParentage, MultiParentLineageUnsupportedError
from controller.loop_run import CampaignLoopResult, LoopEvent, run_campaign_loop
from xtrax.devtools.freshness import Attestation
from xtrax.loop.campaign_approval_gate import (
    ApprovalExpiredError,
    NoMatchingApprovalError,
)
from xtrax.loop.closure_lock import build_closure_manifest
from xtrax.loop.external_stop_watchdog import WatchdogCriteria
from xtrax.loop.seed_gate import SeedTrialCounts
from xtrax.loop.stats_battery_gate import BathosStatsBatteryVerdict

# ---------------------------------------------------------------------------
# Fakes: watchdog starter/handle (NEVER the real subprocess-spawning start_watchdog)
# ---------------------------------------------------------------------------


class _FakeWatchdogHandle:
    def __init__(self) -> None:
        self.stop_calls = 0

    def is_alive(self) -> bool:
        return self.stop_calls == 0

    def join(self, timeout: float | None = None) -> None:  # pragma: no cover - unused by driver
        pass

    def stop(self) -> None:
        self.stop_calls += 1


class _FakeWatchdogStarter:
    def __init__(self) -> None:
        self.calls: list[tuple[int, WatchdogCriteria]] = []
        self.handles: list[_FakeWatchdogHandle] = []

    def __call__(self, target_pid: int, criteria: WatchdogCriteria) -> _FakeWatchdogHandle:
        self.calls.append((target_pid, criteria))
        handle = _FakeWatchdogHandle()
        self.handles.append(handle)
        return handle


# ---------------------------------------------------------------------------
# Fake dispatch backend: writes REAL file content per call, so
# run_multi_iteration_loop's diversity-quota window read (result.handoff.path.read_text())
# always succeeds -- mirrors test_multi_iteration_loop.py's own _SequentialDispatchBackend.
# ---------------------------------------------------------------------------


class _RealFileDispatchBackend:
    def __init__(self, tmp_path: Path) -> None:
        self._tmp_path = tmp_path
        self.call_count = 0

    def dispatch_candidate(self) -> CandidateHandoff:
        source = f"def f_{self.call_count}():\n    return {self.call_count}\n"
        # _TIMING_PROBE_SUFFIX appended (byte-identical every call) so the real, unstubbable
        # measure_two_phase_timing_fn default has a jax.jit-safe callable to resolve -- see the
        # module docstring's #4203 addendum (mirrors test_multi_iteration_loop.py's own pattern).
        content = source + _TIMING_PROBE_SUFFIX
        path = self._tmp_path / f"candidate_{self.call_count}.py"
        path.write_text(content, encoding="utf-8")
        sha256_hex = hashlib.sha256(content.encode("utf-8")).hexdigest()
        self.call_count += 1
        return CandidateHandoff(path=path, content_sha256=sha256_hex)


# ---------------------------------------------------------------------------
# Recording MCP transport: proves campaign_conclude fires at the real wire level (a genuine
# spy on the outgoing MCP tool call), not merely that some Python method was invoked.
# ---------------------------------------------------------------------------


def _ok_envelope(**extra: Any) -> dict[str, Any]:
    return {"ok": True, "error_code": None, "error": None, "resolution_hint": None, **extra}


def _fail_envelope(**extra: Any) -> dict[str, Any]:
    return {
        "ok": False,
        "error_code": "boom",
        "error": "injected transport-level failure",
        "resolution_hint": None,
        **extra,
    }


class _MultiToolTransport:
    """Handles all three of BathosCampaignAdapter's tools with minimal-but-valid envelopes,
    recording every call for inspection. `conclude_should_fail` lets a test simulate
    `campaign_conclude` itself failing (a secondary failure during failure handling)."""

    def __init__(
        self,
        *,
        campaign_id: str = "camp-loop-run-test",
        run_success: bool = True,
        conclude_should_fail: bool = False,
    ) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._campaign_id = campaign_id
        self._run_success = run_success
        self._conclude_should_fail = conclude_should_fail

    def __call__(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((tool_name, dict(arguments)))
        if tool_name == "campaign_create":
            return _ok_envelope(
                campaign_id=self._campaign_id,
                name=arguments["name"],
                mode=arguments["mode"],
                status="active",
                started_at="2026-07-17T00:00:00Z",
            )
        if tool_name == "run":
            return _ok_envelope(
                script_path=arguments["script_path"],
                exit_code=0 if self._run_success else 1,
                success=self._run_success,
            )
        if tool_name == "campaign_conclude":
            if self._conclude_should_fail:
                return _fail_envelope()
            return _ok_envelope(
                status="concluded",
                campaign_id=arguments["campaign_id"],
                outcome_label=arguments["outcome_label"],
            )
        raise AssertionError(f"unexpected tool_name {tool_name!r}")

    def calls_for(self, tool_name: str) -> list[dict[str, Any]]:
        return [args for name, args in self.calls if name == tool_name]


def _adapter(transport: _MultiToolTransport) -> BathosCampaignAdapter:
    return BathosCampaignAdapter(transport=transport, token="test-token")


def _passing_stats_verdict(**_kwargs: Any) -> BathosStatsBatteryVerdict:
    return BathosStatsBatteryVerdict(
        verdict="pass",
        scipy_available=True,
        reasons=(),
        cohens_d=0.5,
        win_rate=0.8,
        breakdown_point=0.3,
        p_superiority=0.8,
        wilcoxon_p_value=0.01,
        icc=0.995,
        baseline_budget_equivalent=True,
    )


def _passing_seed_counts(
    db: Any, script_sha256: str, hypothesis_clause_id: str = ""
) -> SeedTrialCounts:
    return SeedTrialCounts(script_sha256=script_sha256, distinct_seed_count=5, trial_count=40)


def _passing_campaign_approval_fn(
    campaign_id: str, *, toml_path: Path | None = None
) -> Attestation:
    """Stub standing in for T2-32's real `assert_campaign_approved` -- every existing LC-11 test
    exercises campaign-lifecycle behavior with no real `.praxia/loop_human_gates.toml` entry for
    its own test campaign name, so the REAL gate (which reads a TOML file) would reject every one
    of them. Tests that exercise the approval gate itself pass their own `campaign_approval_fn` or
    omit it to reach the real default."""
    return Attestation(attested_at="2026-07-19T00:00:00Z", ttl_days=30.0, attested_by="test")


def _passing_structure_tripwire_fn(
    path: Path, callable_name: str, *, abstract_inputs: list[Any], concrete_inputs: list[Any]
) -> None:
    """Stub for T2-13's real gate (GW-04) -- no-op for end-to-end tests."""
    return None


def _passing_candidate_smoke_fn(
    path: Path,
    callable_name: str,
    *,
    concrete_inputs: list[Any],
    wall_clock_budget_seconds: float = 60.0,
    poll_interval_seconds: float = 0.5,
    root: Path | None = None,
) -> None:
    """Stub for T2-14's real gate (GW-04) -- no-op for end-to-end tests."""
    return None


def _passing_checkified_execution_fn(
    path: Path,
    callable_name: str,
    *,
    concrete_inputs: list[Any],
    check_nans: bool = True,
    check_infs: bool = True,
) -> Any:
    """Stub for T2-15's real gate (GW-04) -- no-op for end-to-end tests."""
    return None


def _passing_capability_probe_fn(catalog_dir: str = "") -> Any:
    """Stub for T2-27's real gate (GW-05) -- returns a passing CapabilityProbeResult for
    end-to-end tests. Never calls the real bathos.capability.probe_capabilities."""
    from xtrax.loop.capability_probe_gate import CapabilityProbeResult

    return CapabilityProbeResult(seed_live=True, stats_battery_live=True)


_CRITERIA = WatchdogCriteria(wall_clock_budget_seconds=3600.0)


# ---------------------------------------------------------------------------
# GW-02 (#3649) / #4203 shared fixtures -- mirrors test_multi_iteration_loop.py's own
# identically-shaped helpers verbatim (see this file's module docstring's #4203 addendum).
# ---------------------------------------------------------------------------

_TIMING_PROBE_NAME = "timing_probe"
_TIMING_PROBE_SUFFIX = f"\n\ndef {_TIMING_PROBE_NAME}(a, b):\n    return a + b\n"
_TIMING_CONCRETE_INPUTS: list[Any] = [1.0, 2.0]
_HIGHER_IS_BETTER = {"metric_a": True, "metric_b": True}


def _make_monotonic_score_fn() -> Any:
    """A fresh, per-test score_fn returning a strictly-improving two-metric fitness dict (equal
    per-metric deltas -> Cohen's d always inf) -- see test_multi_iteration_loop.py's identical
    helper for the full rationale."""
    counter = {"n": 0}

    def _score_fn(
        raw_artifact_paths: tuple[Path, ...], split_paths: tuple[Path, ...]
    ) -> dict[str, float]:
        counter["n"] += 1
        value = float(counter["n"])
        return {"metric_a": value, "metric_b": value}

    return _score_fn


def _frozen_context() -> BathosFrozenContext:
    return BathosFrozenContext(
        locked=build_closure_manifest(
            evaluator_paths=(),
            split_paths=(),
            metric_def_paths=(),
            config={},
        ),
        campaign_adapter=None,  # type: ignore[arg-type]  # unused -- score_fn never touches it
        campaign_id="camp-1",
        score_fn=_make_monotonic_score_fn(),
    )


def _base_kwargs(dispatch_backend: Any, transport: _MultiToolTransport) -> dict[str, Any]:
    return {
        "dispatch_backend": dispatch_backend,
        "campaign_adapter": _adapter(transport),
        "campaign_name": "loop-run-test-campaign",
        "campaign_mode": "exploration",
        "watchdog_criteria": _CRITERIA,
        "stats_battery_kwargs": {},
        "stats_battery_fn": _passing_stats_verdict,
        "seed_trial_counts_fn": _passing_seed_counts,
        "campaign_approval_fn": _passing_campaign_approval_fn,
        "output_paths": ["dummy-output.txt"],
        "frozen_context": _frozen_context(),
        "current_config": {},
        "repo": Path("unused-repo"),
        "ratchet_ref_name": "refs/xtrax/best-so-far",
        "commit_tree_sha": "unused-tree-sha",
        "callable_name": _TIMING_PROBE_NAME,
        "concrete_inputs": _TIMING_CONCRETE_INPUTS,
        "higher_is_better": _HIGHER_IS_BETTER,
        "abstract_inputs": [],
        "structure_tripwire_fn": _passing_structure_tripwire_fn,
        "candidate_smoke_fn": _passing_candidate_smoke_fn,
        "candidate_smoke_root": None,
        "checkified_execution_fn": _passing_checkified_execution_fn,
        "capability_probe_fn": _passing_capability_probe_fn,
        "capability_probe_catalog_dir": "",
        # See test_multi_iteration_loop.py's module docstring #4203 addendum for why this
        # deliberate deviation from a literal "8 new values" reading is required.
        "allow_fresh_start_despite_existing_lineage": True,
    }


@pytest.fixture(autouse=True)
def _stub_crash_atomicity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirrors test_multi_iteration_loop.py's own identically-shaped autouse fixture (P3.2,
    backlog #4203) -- see that module's docstring #4203 addendum for the full rationale."""
    monkeypatch.setattr(
        main_loop_module, "read_best_so_far", lambda repo, ref_name: "stub-best-so-far-sha"
    )
    monkeypatch.setattr(
        main_loop_module,
        "create_pending_commit",
        lambda repo, tree_sha, parent_sha, message: "pending-sha",
    )
    monkeypatch.setattr(
        main_loop_module,
        "advance_best_so_far",
        lambda repo, ref_name, new_sha, expected_old_sha: None,
    )
    monkeypatch.setattr(
        main_loop_module, "reset_worktree_to_best_so_far", lambda repo, ref_name: "best-sha"
    )


# ---------------------------------------------------------------------------
# 1. Success path
# ---------------------------------------------------------------------------


class TestSuccessPath:
    def test_success_concludes_with_success_outcome_label(self, tmp_path: Path) -> None:
        dispatch_backend = _RealFileDispatchBackend(tmp_path)
        transport = _MultiToolTransport()
        starter = _FakeWatchdogStarter()
        received: list[LoopEvent] = []

        result = run_campaign_loop(
            **_base_kwargs(dispatch_backend, transport),
            max_candidates=2,
            start_watchdog_fn=starter,
            on_loop_event=received.append,
        )

        assert isinstance(result, CampaignLoopResult)
        assert result.campaign_id == "camp-loop-run-test"
        assert len(result.loop_result.iterations) == 2
        assert result.conclusion.outcome_label == "success"

        # campaign_create called exactly once.
        assert len(transport.calls_for("campaign_create")) == 1
        assert transport.calls_for("campaign_create")[0]["name"] == "loop-run-test-campaign"
        assert transport.calls_for("campaign_create")[0]["mode"] == "exploration"

        # campaign_conclude called exactly once, at the real wire level, success-shaped.
        conclude_calls = transport.calls_for("campaign_conclude")
        assert len(conclude_calls) == 1
        assert conclude_calls[0]["outcome_label"] == "success"
        assert conclude_calls[0]["campaign_id"] == "camp-loop-run-test"

        # Telemetry hook fired for both milestones, in order, sharing one task_id.
        assert [event.kind for event in received] == ["campaign_created", "campaign_concluded"]
        assert received[0].task_id == received[1].task_id == result.task_id
        assert received[1].outcome_label == "success"

        # Watchdog started once and stopped once.
        assert len(starter.calls) == 1
        assert starter.handles[0].stop_calls == 1

    def test_explicit_task_id_used_verbatim(self, tmp_path: Path) -> None:
        dispatch_backend = _RealFileDispatchBackend(tmp_path)
        transport = _MultiToolTransport()
        starter = _FakeWatchdogStarter()

        result = run_campaign_loop(
            **_base_kwargs(dispatch_backend, transport),
            max_candidates=1,
            start_watchdog_fn=starter,
            task_id="260717_my-explicit-task",
        )

        assert result.task_id == "260717_my-explicit-task"

    def test_omitted_task_id_is_auto_minted_yymmdd_slug_shape(self, tmp_path: Path) -> None:
        dispatch_backend = _RealFileDispatchBackend(tmp_path)
        transport = _MultiToolTransport()
        starter = _FakeWatchdogStarter()

        result = run_campaign_loop(
            **_base_kwargs(dispatch_backend, transport),
            max_candidates=1,
            start_watchdog_fn=starter,
        )

        assert re.match(r"^\d{6}_[a-z0-9-]+-[0-9a-f]{8}$", result.task_id), result.task_id


# ---------------------------------------------------------------------------
# 2. Caught per-candidate failure -- all three MockDispatchBackend failure-injection modes
# ---------------------------------------------------------------------------


class TestCaughtPerCandidateFailure:
    @pytest.mark.parametrize(
        ("mode", "expected_exception"),
        [
            (MockFailureMode.TIMEOUT, TimeoutError),
            (MockFailureMode.MALFORMED_COMPLETION, ValueError),
            (MockFailureMode.CANDIDATE_HANDOFF_FAILURE, CandidateHandoffFailure),
        ],
    )
    def test_failure_mode_concludes_partial_failure_and_reraises(
        self, tmp_path: Path, mode: MockFailureMode, expected_exception: type[Exception]
    ) -> None:
        dispatch_backend = MockDispatchBackend(
            candidate_path=tmp_path / "unused.py",
            candidate_content="def f():\n    return 1\n",
            mode=mode,
            timeout_delay=0.01,
        )
        transport = _MultiToolTransport()
        starter = _FakeWatchdogStarter()
        received: list[LoopEvent] = []

        with pytest.raises(expected_exception):
            run_campaign_loop(
                **_base_kwargs(dispatch_backend, transport),
                max_candidates=3,
                start_watchdog_fn=starter,
                on_loop_event=received.append,
            )

        # campaign_create still happened -- the campaign-lifecycle-ownership scoping decision is
        # real: this wrapper opens the campaign before attempting any candidate.
        assert len(transport.calls_for("campaign_create")) == 1

        # campaign_conclude genuinely fired at the wire level with the categorized outcome_label.
        conclude_calls = transport.calls_for("campaign_conclude")
        assert len(conclude_calls) == 1
        assert conclude_calls[0]["outcome_label"] == "partial_failure"

        # Telemetry hook mirrors the same milestones.
        assert [event.kind for event in received] == ["campaign_created", "campaign_concluded"]
        assert received[1].outcome_label == "partial_failure"

        # Watchdog cleanup still ran even though this wrapper's own failure handling ran too.
        assert starter.handles[0].stop_calls == 1

        # No real candidate run was ever recorded -- dispatch failed on the very first call.
        assert transport.calls_for("run") == []


# ---------------------------------------------------------------------------
# 3. Uncaught exception -- a genuine MultiParentLineageUnsupportedError (real production code,
#    not a synthetic exception type), proving the classification is grounded in real behavior.
# ---------------------------------------------------------------------------


class TestUncaughtException:
    def test_multi_parent_lineage_error_concludes_aborted_and_reraises(
        self, tmp_path: Path
    ) -> None:
        dispatch_backend = _RealFileDispatchBackend(tmp_path)
        transport = _MultiToolTransport()
        starter = _FakeWatchdogStarter()
        received: list[LoopEvent] = []
        multi_parent = CandidateParentage(parent_run_ids=("parent-a", "parent-b"))

        with pytest.raises(MultiParentLineageUnsupportedError):
            run_campaign_loop(
                **_base_kwargs(dispatch_backend, transport),
                max_candidates=3,
                parentage=multi_parent,
                start_watchdog_fn=starter,
                on_loop_event=received.append,
            )

        assert len(transport.calls_for("campaign_create")) == 1

        conclude_calls = transport.calls_for("campaign_conclude")
        assert len(conclude_calls) == 1
        assert conclude_calls[0]["outcome_label"] == "aborted"

        assert [event.kind for event in received] == ["campaign_created", "campaign_concluded"]
        assert received[1].outcome_label == "aborted"

        assert starter.handles[0].stop_calls == 1

        # resolve_derived_from raises before any bathos `run` call is ever attempted.
        assert transport.calls_for("run") == []

    def test_caught_and_uncaught_paths_produce_distinct_outcome_labels(
        self, tmp_path: Path
    ) -> None:
        """Directly proves the AC's own two-clause framing: "a caught per-candidate failure, OR
        an uncaught exception" are genuinely distinct cases, not the same outcome_label twice."""
        caught_transport = _MultiToolTransport()
        caught_backend = MockDispatchBackend(
            candidate_path=tmp_path / "unused.py",
            candidate_content="def f():\n    return 1\n",
            mode=MockFailureMode.CANDIDATE_HANDOFF_FAILURE,
        )
        with pytest.raises(CandidateHandoffFailure):
            run_campaign_loop(
                **_base_kwargs(caught_backend, caught_transport),
                max_candidates=3,
                start_watchdog_fn=_FakeWatchdogStarter(),
            )
        caught_label = caught_transport.calls_for("campaign_conclude")[0]["outcome_label"]

        uncaught_transport = _MultiToolTransport()
        uncaught_backend = _RealFileDispatchBackend(tmp_path)
        with pytest.raises(MultiParentLineageUnsupportedError):
            run_campaign_loop(
                **_base_kwargs(uncaught_backend, uncaught_transport),
                max_candidates=3,
                parentage=CandidateParentage(parent_run_ids=("a", "b")),
                start_watchdog_fn=_FakeWatchdogStarter(),
            )
        uncaught_label = uncaught_transport.calls_for("campaign_conclude")[0]["outcome_label"]

        assert caught_label == "partial_failure"
        assert uncaught_label == "aborted"
        assert caught_label != uncaught_label


# ---------------------------------------------------------------------------
# 4. A secondary campaign_conclude failure never masks the ORIGINAL exception
# ---------------------------------------------------------------------------


class TestSecondaryConcludeFailureDoesNotMaskOriginal:
    def test_conclude_failure_during_failure_handling_still_raises_original(
        self, tmp_path: Path
    ) -> None:
        dispatch_backend = MockDispatchBackend(
            candidate_path=tmp_path / "unused.py",
            candidate_content="def f():\n    return 1\n",
            mode=MockFailureMode.CANDIDATE_HANDOFF_FAILURE,
        )
        transport = _MultiToolTransport(conclude_should_fail=True)
        starter = _FakeWatchdogStarter()
        received: list[LoopEvent] = []

        with pytest.raises(CandidateHandoffFailure):
            run_campaign_loop(
                **_base_kwargs(dispatch_backend, transport),
                max_candidates=3,
                start_watchdog_fn=starter,
                on_loop_event=received.append,
            )

        # campaign_conclude was genuinely attempted (and failed) -- not skipped.
        assert len(transport.calls_for("campaign_conclude")) == 1

        # The secondary failure is surfaced via telemetry, not via the raised exception type.
        assert [event.kind for event in received] == [
            "campaign_created",
            "campaign_conclude_failed",
        ]

    def test_conclude_failure_on_the_success_path_propagates_that_failure(
        self, tmp_path: Path
    ) -> None:
        """No original exception to protect on the success path -- if campaign_conclude itself
        fails here, ITS failure is the real, honest thing to surface."""
        dispatch_backend = _RealFileDispatchBackend(tmp_path)
        transport = _MultiToolTransport(conclude_should_fail=True)
        starter = _FakeWatchdogStarter()

        with pytest.raises(BathosMcpToolError):
            run_campaign_loop(
                **_base_kwargs(dispatch_backend, transport),
                max_candidates=1,
                start_watchdog_fn=starter,
            )


# ---------------------------------------------------------------------------
# 5. max_candidates < 1 raises before campaign_create is ever called
# ---------------------------------------------------------------------------


class TestMaxCandidatesValidation:
    def test_zero_max_candidates_raises_before_campaign_create(self, tmp_path: Path) -> None:
        dispatch_backend = _RealFileDispatchBackend(tmp_path)
        transport = _MultiToolTransport()
        starter = _FakeWatchdogStarter()

        with pytest.raises(ValueError, match="max_candidates must be >= 1"):
            run_campaign_loop(
                **_base_kwargs(dispatch_backend, transport),
                max_candidates=0,
                start_watchdog_fn=starter,
            )

        assert transport.calls == [], "no campaign should be created for invalid input"
        assert starter.calls == [], "the watchdog must not be started for invalid input either"


# ---------------------------------------------------------------------------
# 6. Campaign-approval gate (T2-32, AC-25; [GW-03]): a genuine pre-`campaign_create` reject, not
# merely an injection seam that's never exercised by its real default.
# ---------------------------------------------------------------------------


class TestCampaignApprovalGate:
    def test_failing_approval_raises_before_campaign_create(self, tmp_path: Path) -> None:
        dispatch_backend = _RealFileDispatchBackend(tmp_path)
        transport = _MultiToolTransport()
        starter = _FakeWatchdogStarter()

        def denying_approval_fn(campaign_id: str, *, toml_path: Path | None = None) -> Attestation:
            msg = f"no T2-32 approval found for campaign {campaign_id}"
            raise NoMatchingApprovalError(msg)

        kwargs = _base_kwargs(dispatch_backend, transport)
        kwargs["campaign_approval_fn"] = denying_approval_fn

        with pytest.raises(NoMatchingApprovalError):
            run_campaign_loop(
                **kwargs,
                max_candidates=1,
                start_watchdog_fn=starter,
            )

        assert transport.calls == [], (
            "no bathos call (not even campaign_create) should ever happen when the "
            "campaign-approval gate rejects -- the exception must fire before campaign_create"
        )
        assert starter.calls == [], "the watchdog must not be started when approval is denied"

    def test_default_campaign_approval_fn_is_the_real_gate_and_rejects_with_no_toml_entry(
        self, tmp_path: Path
    ) -> None:
        """Omitting `campaign_approval_fn` entirely must exercise T2-32's REAL
        `assert_campaign_approved` -- not just prove the injection seam works."""
        dispatch_backend = _RealFileDispatchBackend(tmp_path)
        transport = _MultiToolTransport()
        starter = _FakeWatchdogStarter()
        missing_gates_toml = tmp_path / "nonexistent_gates.toml"

        kwargs = _base_kwargs(dispatch_backend, transport)
        del kwargs["campaign_approval_fn"]

        with pytest.raises(NoMatchingApprovalError, match="gates file not found"):
            run_campaign_loop(
                **kwargs,
                max_candidates=1,
                start_watchdog_fn=starter,
                campaign_approval_toml_path=missing_gates_toml,
            )

        assert transport.calls == [], "a denied campaign must burn zero real bathos calls"

    def test_default_campaign_approval_fn_lets_an_approved_campaign_proceed(
        self, tmp_path: Path
    ) -> None:
        """The real gate must not block a genuinely fresh, matching approval -- the rest of the
        run still completes end-to-end."""
        dispatch_backend = _RealFileDispatchBackend(tmp_path)
        transport = _MultiToolTransport()
        starter = _FakeWatchdogStarter()
        gates_toml = tmp_path / "gates.toml"
        gates_toml.write_text(
            """
[[gates]]
id = "T2-32"
event_ref = "loop-run-test-campaign"
attested_at = "2026-07-19T00:00:00Z"
ttl_days = 30.0
attested_by = "Marielle Russo"
note = "Approved for this test run"
"""
        )

        kwargs = _base_kwargs(dispatch_backend, transport)
        del kwargs["campaign_approval_fn"]

        result = run_campaign_loop(
            **kwargs,
            max_candidates=1,
            start_watchdog_fn=starter,
            campaign_approval_toml_path=gates_toml,
        )

        assert len(transport.calls_for("campaign_create")) == 1
        assert result.conclusion.outcome_label == "success"

    def test_default_campaign_approval_fn_rejects_an_expired_approval(self, tmp_path: Path) -> None:
        dispatch_backend = _RealFileDispatchBackend(tmp_path)
        transport = _MultiToolTransport()
        starter = _FakeWatchdogStarter()
        gates_toml = tmp_path / "gates.toml"
        gates_toml.write_text(
            """
[[gates]]
id = "T2-32"
event_ref = "loop-run-test-campaign"
attested_at = "2000-01-01T00:00:00Z"
ttl_days = 1.0
attested_by = "Marielle Russo"
note = "Old approval (expired)"
"""
        )

        kwargs = _base_kwargs(dispatch_backend, transport)
        del kwargs["campaign_approval_fn"]

        with pytest.raises(ApprovalExpiredError):
            run_campaign_loop(
                **kwargs,
                max_candidates=1,
                start_watchdog_fn=starter,
                campaign_approval_toml_path=gates_toml,
            )

        assert transport.calls == [], "an expired approval must burn zero real bathos calls"

    def test_approval_checked_against_campaign_name_not_bathos_campaign_id(
        self, tmp_path: Path
    ) -> None:
        """Proves the approval-gate lookup uses `campaign_name` (known before `campaign_create`),
        not `handle.campaign_id` (which bathos only mints DURING `campaign_create`, unavailable
        at the point this gate must fire)."""
        dispatch_backend = _RealFileDispatchBackend(tmp_path)
        # The recording transport's own campaign_id is deliberately unrelated to campaign_name --
        # if the gate were (incorrectly) checked against the bathos campaign_id, this would fail.
        transport = _MultiToolTransport(campaign_id="totally-unrelated-bathos-id")
        starter = _FakeWatchdogStarter()
        received_ids: list[str] = []

        def recording_approval_fn(
            campaign_id: str, *, toml_path: Path | None = None
        ) -> Attestation:
            received_ids.append(campaign_id)
            return Attestation(attested_at="2026-07-19T00:00:00Z", ttl_days=30.0, attested_by="t")

        kwargs = _base_kwargs(dispatch_backend, transport)
        kwargs["campaign_approval_fn"] = recording_approval_fn

        run_campaign_loop(
            **kwargs,
            max_candidates=1,
            start_watchdog_fn=starter,
        )

        assert received_ids == ["loop-run-test-campaign"]


class TestCapabilityProbeGate:
    """Tests for T2-27 (AC-20, GW-05): capability-probe gate wired before campaign_create."""

    def test_failing_probe_raises_before_campaign_create(self, tmp_path: Path) -> None:
        """A failing capability probe must raise CapabilityNotLiveError before campaign_create."""
        dispatch_backend = _RealFileDispatchBackend(tmp_path)
        transport = _MultiToolTransport()
        starter = _FakeWatchdogStarter()

        def failing_probe_fn(catalog_dir: str = "") -> Any:
            from xtrax.loop.capability_probe_gate import CapabilityProbeResult

            return CapabilityProbeResult(seed_live=False, stats_battery_live=False)

        kwargs = _base_kwargs(dispatch_backend, transport)
        kwargs["capability_probe_fn"] = failing_probe_fn
        kwargs["campaign_mode"] = "confirmation"  # gated mode, will check the probe result

        from xtrax.loop.capability_probe_gate import CapabilityNotLiveError

        with pytest.raises(CapabilityNotLiveError):
            run_campaign_loop(
                **kwargs,
                max_candidates=1,
                start_watchdog_fn=starter,
            )

        assert transport.calls == [], (
            "no bathos call (not even campaign_create) should ever happen when the "
            "capability-probe gate rejects -- the exception must fire before campaign_create"
        )
        assert starter.calls == [], "the watchdog must not be started when probe fails"

    def test_failing_probe_in_exploration_mode_allows_campaign_to_proceed(
        self, tmp_path: Path
    ) -> None:
        """PM-4 carve-out: failing probe in exploration mode must NOT raise -- exploration
        campaigns proceed regardless (the gate never enforces in exploration mode)."""
        dispatch_backend = _RealFileDispatchBackend(tmp_path)
        transport = _MultiToolTransport()
        starter = _FakeWatchdogStarter()

        def failing_probe_fn(catalog_dir: str = "") -> Any:
            from xtrax.loop.capability_probe_gate import CapabilityProbeResult

            return CapabilityProbeResult(seed_live=False, stats_battery_live=False)

        kwargs = _base_kwargs(dispatch_backend, transport)
        kwargs["capability_probe_fn"] = failing_probe_fn
        kwargs["campaign_mode"] = "exploration"  # exploration mode, probe is not gated

        result = run_campaign_loop(
            **kwargs,
            max_candidates=1,
            start_watchdog_fn=starter,
        )

        # Probe fails but campaign still creates and completes successfully
        assert len(transport.calls_for("campaign_create")) == 1
        assert result.conclusion.outcome_label == "success"

    def test_passing_probe_allows_campaign_to_proceed(self, tmp_path: Path) -> None:
        """A passing capability probe must not block campaign creation."""
        dispatch_backend = _RealFileDispatchBackend(tmp_path)
        transport = _MultiToolTransport()
        starter = _FakeWatchdogStarter()

        def passing_probe_fn(catalog_dir: str = "") -> Any:
            from xtrax.loop.capability_probe_gate import CapabilityProbeResult

            return CapabilityProbeResult(seed_live=True, stats_battery_live=True)

        kwargs = _base_kwargs(dispatch_backend, transport)
        kwargs["capability_probe_fn"] = passing_probe_fn
        kwargs["campaign_mode"] = "confirmation"

        result = run_campaign_loop(
            **kwargs,
            max_candidates=1,
            start_watchdog_fn=starter,
        )

        assert len(transport.calls_for("campaign_create")) == 1
        assert result.conclusion.outcome_label == "success"

    def test_capability_probe_fn_receives_exact_catalog_dir_passed_through(
        self, tmp_path: Path
    ) -> None:
        """Proves the catalog_dir is forwarded exactly from run_campaign_loop parameter."""
        dispatch_backend = _RealFileDispatchBackend(tmp_path)
        transport = _MultiToolTransport()
        starter = _FakeWatchdogStarter()
        received_dirs: list[str] = []

        def recording_probe_fn(catalog_dir: str = "") -> Any:
            received_dirs.append(catalog_dir)
            from xtrax.loop.capability_probe_gate import CapabilityProbeResult

            return CapabilityProbeResult(seed_live=True, stats_battery_live=True)

        kwargs = _base_kwargs(dispatch_backend, transport)
        kwargs["capability_probe_fn"] = recording_probe_fn
        kwargs["capability_probe_catalog_dir"] = "/custom/catalog/dir"

        run_campaign_loop(
            **kwargs,
            max_candidates=1,
            start_watchdog_fn=starter,
        )

        assert received_dirs == ["/custom/catalog/dir"]

    def test_default_capability_probe_fn_is_skipped_if_bathos_capability_unavailable(
        self, tmp_path: Path
    ) -> None:
        """If bathos.capability is not installed, the default probe_fn should skip cleanly
        (via pytest.importorskip or equivalent)."""
        # This test is primarily for coverage in CI tiers that skip bathos.capability.
        # The real default (get_capability_probe_result) handles the skip internally if the
        # bathos.capability import fails, so tests omitting an injected capability_probe_fn
        # must reach the real default and either pass or skip.
        dispatch_backend = _RealFileDispatchBackend(tmp_path)
        transport = _MultiToolTransport()
        starter = _FakeWatchdogStarter()

        kwargs = _base_kwargs(dispatch_backend, transport)
        # Omit capability_probe_fn to reach the real default
        del kwargs["capability_probe_fn"]

        try:
            result = run_campaign_loop(
                **kwargs,
                max_candidates=1,
                start_watchdog_fn=starter,
            )
            # If we get here, bathos.capability was available and the real default succeeded
            assert len(transport.calls_for("campaign_create")) == 1
            assert result.conclusion.outcome_label == "success"
        except ImportError as e:
            # If bathos.capability is not installed, we expect an ImportError
            pytest.skip(f"bathos.capability not available: {e}")

    def test_probe_fired_and_campaign_create_call_count_proves_ordering(
        self, tmp_path: Path
    ) -> None:
        """Proves via call-order tracking that the probe fires and completes before
        campaign_create (not after). We track this by monitoring the call sequence of our
        injected probe_fn and the transport's calls_for assertions."""
        dispatch_backend = _RealFileDispatchBackend(tmp_path)
        transport = _MultiToolTransport()
        starter = _FakeWatchdogStarter()
        calls_sequence: list[str] = []

        def tracking_probe_fn(catalog_dir: str = "") -> Any:
            calls_sequence.append("probe_called")
            from xtrax.loop.capability_probe_gate import CapabilityProbeResult

            return CapabilityProbeResult(seed_live=True, stats_battery_live=True)

        kwargs = _base_kwargs(dispatch_backend, transport)
        kwargs["capability_probe_fn"] = tracking_probe_fn

        result = run_campaign_loop(
            **kwargs,
            max_candidates=1,
            start_watchdog_fn=starter,
        )

        # Prove ordering: probe must be called before campaign_create (evidenced by the
        # probe being called and the campaign successfully created and concluded)
        assert calls_sequence == ["probe_called"], (
            "capability_probe_fn must be called during run_campaign_loop"
        )
        assert len(transport.calls_for("campaign_create")) == 1, (
            "campaign_create must be called exactly once after probe completes"
        )
        assert result.conclusion.outcome_label == "success", (
            "the campaign must complete successfully after probe passes"
        )


# ---------------------------------------------------------------------------
# 7. Cross-boundary forwarding-spy integration test (D3-amended C2 AC, backlog #4203): spies
# SIMULTANEOUSLY on BOTH layer boundaries -- run_campaign_loop -> run_multi_iteration_loop, AND
# run_multi_iteration_loop -> run_one_candidate_pass -- across >=2 iterations, asserting all 10
# named values forward correctly at both boundaries.
# ---------------------------------------------------------------------------

_FORWARDED_BYTE_IDENTICAL_KEYS = (
    "current_config",
    "repo",
    "ratchet_ref_name",
    "commit_tree_sha",
    "callable_name",
    "concrete_inputs",
    "candidate_target_path",
    "allow_fresh_start_despite_existing_lineage",
)


class TestCrossBoundaryForwardingSpy:
    def test_all_ten_values_forwarded_at_both_layer_boundaries_across_two_iterations(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dispatch_backend = _RealFileDispatchBackend(tmp_path)
        transport = _MultiToolTransport()
        starter = _FakeWatchdogStarter()
        base_kwargs = _base_kwargs(dispatch_backend, transport)
        # #4215 (AC-a): a literal, non-None value -- inert on this test's own accept-path
        # behavior (the autouse _stub_crash_atomicity fixture's read_best_so_far always returns
        # a fixed non-None SHA, so neither commit_parent_sha's nor bootstrap_base_tree_sha's
        # fan-out is ever exercised on real accept-branch git plumbing here -- only that the
        # VALUE arrives correctly at both call sites is being proven; AC-b's real-git-plumbing
        # proof lives in TestBootstrapCommitShaIntegration below).
        base_kwargs["bootstrap_commit_sha"] = "spy-bootstrap-sha"

        # Boundary 1: run_campaign_loop's own call to run_multi_iteration_loop (loop_run.py's
        # own imported name).
        outer_calls: list[dict[str, Any]] = []
        real_run_multi_iteration_loop = loop_run_module.run_multi_iteration_loop

        def _outer_spy(*args: Any, **kwargs: Any) -> Any:
            outer_calls.append(kwargs)
            return real_run_multi_iteration_loop(*args, **kwargs)

        monkeypatch.setattr(loop_run_module, "run_multi_iteration_loop", _outer_spy)

        # Boundary 2: run_multi_iteration_loop's own call to run_one_candidate_pass
        # (multi_iteration_loop.py's own imported name).
        inner_calls: list[dict[str, Any]] = []
        real_run_one_candidate_pass = multi_iteration_loop_module.run_one_candidate_pass

        def _inner_spy(*args: Any, **kwargs: Any) -> Any:
            inner_calls.append(kwargs)
            return real_run_one_candidate_pass(*args, **kwargs)

        monkeypatch.setattr(multi_iteration_loop_module, "run_one_candidate_pass", _inner_spy)

        result = run_campaign_loop(
            **base_kwargs,
            max_candidates=2,
            start_watchdog_fn=starter,
        )

        assert result.conclusion.outcome_label == "success"
        assert len(outer_calls) == 1, (
            "run_campaign_loop calls run_multi_iteration_loop exactly once, single call site"
        )
        assert len(inner_calls) == 2, "2 iterations means 2 run_one_candidate_pass calls"

        # --- Boundary 1 assertions: run_campaign_loop -> run_multi_iteration_loop ---
        outer_kwargs = outer_calls[0]
        assert outer_kwargs["frozen_context"] is base_kwargs["frozen_context"]
        assert outer_kwargs["higher_is_better"] is base_kwargs["higher_is_better"]
        for key in _FORWARDED_BYTE_IDENTICAL_KEYS:
            if key == "candidate_target_path":
                assert outer_kwargs[key] is None
                continue
            assert outer_kwargs[key] == base_kwargs[key]
        # #4215 (AC-a): pass-through unchanged at this boundary.
        assert outer_kwargs["bootstrap_commit_sha"] == base_kwargs["bootstrap_commit_sha"]

        # --- Boundary 2 assertions: run_multi_iteration_loop -> run_one_candidate_pass, on
        # BOTH iterations. The 8 static values + allow_fresh_start_despite_existing_lineage
        # forward byte-identical on every call; higher_is_better forwards per the conditional
        # rule (D1): None while best_fitness is None, the real mapping once best_fitness is set.
        for call_kwargs in inner_calls:
            assert call_kwargs["frozen_context"] is base_kwargs["frozen_context"]
            for key in _FORWARDED_BYTE_IDENTICAL_KEYS:
                if key == "candidate_target_path":
                    assert call_kwargs[key] is None
                    continue
                assert call_kwargs[key] == base_kwargs[key]
            # #4215 (AC-a): bootstrap_commit_sha fans out to BOTH inner params, byte-identical,
            # on every call -- two different kwarg names, so not part of
            # _FORWARDED_BYTE_IDENTICAL_KEYS above.
            assert call_kwargs["commit_parent_sha"] == base_kwargs["bootstrap_commit_sha"]
            assert call_kwargs["bootstrap_base_tree_sha"] == base_kwargs["bootstrap_commit_sha"]

        assert inner_calls[0]["best_fitness"] is None
        assert inner_calls[0]["higher_is_better"] is None
        assert inner_calls[1]["best_fitness"] is not None
        assert inner_calls[1]["higher_is_better"] is base_kwargs["higher_is_better"]


# ---------------------------------------------------------------------------
# 8. bootstrap_commit_sha (#4215) integration tests -- AC-b (the actual bug-closing test) and
# AC-c (regression). This file has no conftest.py / no shared fixtures across test files, so a
# self-contained local `_git`/`git_repo` copy is required here, mirroring
# tests/controller/test_main_loop.py's own identically-shaped fixture verbatim (same convention
# tests/controller/test_multi_iteration_loop.py already follows rather than importing across
# test files).
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A minimal real git repo with one commit containing one tracked file, so a valid base
    commit/tree exists for compute_candidate_tree_sha's own real git plumbing."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    _git(repo_dir, "init", "--quiet")
    _git(repo_dir, "config", "user.email", "test@example.com")
    _git(repo_dir, "config", "user.name", "Test")
    (repo_dir / "existing.txt").write_text("existing content\n", encoding="utf-8")
    _git(repo_dir, "add", "existing.txt")
    _git(repo_dir, "commit", "--quiet", "-m", "initial commit")
    return repo_dir


def _head_sha(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


class TestBootstrapCommitShaIntegration:
    """AC-b (bug-closing) + AC-c (regression): proves `bootstrap_commit_sha` threaded through
    the REAL production entry point (`run_campaign_loop`) actually resolves a genuinely-fresh
    campaign's first accepted candidate, exercising REAL git plumbing against a real fixture
    repo -- not just that a kwarg value arrives somewhere (that's AC-a's job, above)."""

    def test_bootstrap_commit_sha_lets_a_genuinely_fresh_campaign_accept_its_first_candidate(
        self,
        tmp_path: Path,
        git_repo: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """AC-b, the actual bug-closing test. `read_best_so_far` is re-patched to return `None`
        (a genuinely fresh campaign, no prior lineage) AND `commit_tree_sha` is left UNSET
        (deleted from the base kwargs, not the existing literal `"unused-tree-sha"`) -- per the
        spec's AC-b sharpening, only unsetting BOTH exercises `bootstrap_base_tree_sha`'s real
        fan-out into `compute_candidate_tree_sha`'s actual `base_sha^{tree}` git plumbing; a
        lazy implementation that only overrides `read_best_so_far` while keeping the literal
        `commit_tree_sha` would never reach that code path at all. `commit_tree_sha_fn` (the
        real default, `compute_candidate_tree_sha`) is never mocked or overridden here -- the
        test passing via the REAL default against a REAL git repo IS the proof.
        """
        monkeypatch.setattr(main_loop_module, "read_best_so_far", lambda repo, ref_name: None)

        dispatch_backend = _RealFileDispatchBackend(tmp_path)
        transport = _MultiToolTransport()
        starter = _FakeWatchdogStarter()

        kwargs = _base_kwargs(dispatch_backend, transport)
        del kwargs["commit_tree_sha"]
        kwargs["candidate_target_path"] = Path("candidate.py")
        kwargs["repo"] = git_repo
        kwargs["bootstrap_commit_sha"] = _head_sha(git_repo)

        result = run_campaign_loop(
            **kwargs,
            max_candidates=1,
            start_watchdog_fn=starter,
        )

        assert result.loop_result.accepted_count == 1
        assert result.conclusion.outcome_label == "success"

    def test_omitted_bootstrap_commit_sha_still_raises_valueerror_on_a_fresh_campaign(
        self,
        tmp_path: Path,
        git_repo: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """AC-c, regression: confirms the existing default-None behavior is unchanged.
        `bootstrap_commit_sha` is omitted entirely (stays at its `None` default) and the
        existing literal `commit_tree_sha="unused-tree-sha"` from `_base_kwargs()` is kept
        unchanged, so the ValueError under test is the `commit_parent_sha`-resolution one
        (main_loop.py's accept branch), which fires before the `commit_tree_sha` branch is ever
        reached."""
        monkeypatch.setattr(main_loop_module, "read_best_so_far", lambda repo, ref_name: None)

        dispatch_backend = _RealFileDispatchBackend(tmp_path)
        transport = _MultiToolTransport()
        starter = _FakeWatchdogStarter()

        kwargs = _base_kwargs(dispatch_backend, transport)
        kwargs["repo"] = git_repo

        with pytest.raises(ValueError, match="commit_parent_sha"):
            run_campaign_loop(
                **kwargs,
                max_candidates=1,
                start_watchdog_fn=starter,
            )
