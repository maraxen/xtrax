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

## Watchdog safety (mandatory reading for anyone editing this file)

Mirrors `tests/controller/test_multi_iteration_loop.py`'s own warning verbatim:
`xtrax.loop.external_stop_watchdog.start_watchdog` is a REAL function that spawns a REAL, detached
OS subprocess that SIGKILLs a target PID once a wall-clock budget elapses. **No test in this file
calls it.** Every test injects `start_watchdog_fn=<a _FakeWatchdogStarter instance>`.
"""

import hashlib
import re
from pathlib import Path
from typing import Any

import pytest

from controller.bathos_campaign_adapter import BathosCampaignAdapter, BathosMcpToolError
from controller.dispatch import (
    CandidateHandoff,
    CandidateHandoffFailure,
    MockDispatchBackend,
    MockFailureMode,
)
from controller.lineage_interim import CandidateParentage, MultiParentLineageUnsupportedError
from controller.loop_run import CampaignLoopResult, LoopEvent, run_campaign_loop
from xtrax.devtools.freshness import Attestation
from xtrax.loop.attestation_evidence_gate import EvidenceCandidate
from xtrax.loop.campaign_approval_gate import (
    ApprovalExpiredError,
    NoMatchingApprovalError,
)
from xtrax.loop.external_stop_watchdog import WatchdogCriteria
from xtrax.loop.seed_gate import SeedTrialCounts
from xtrax.loop.sidecar_drift_gate import SidecarDriftSignal
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
        path = self._tmp_path / f"candidate_{self.call_count}.py"
        path.write_text(source, encoding="utf-8")
        sha256_hex = hashlib.sha256(source.encode("utf-8")).hexdigest()
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


def _passing_evidence_candidate_fn(
    run_id: str, *, catalog_dir: str = "", stdout_verified: bool | None = None
) -> EvidenceCandidate:
    """Stub standing in for [GW-01]'s real `get_evidence_candidate_for_run` -- without this,
    every LC-11 test that lets a candidate genuinely proceed would reach the REAL bathos-backed
    default, which requires the `controller` extra (bathos installed) that this dev-only test
    file must not depend on."""
    return EvidenceCandidate(run_id=run_id, manifest_verified=True, stdout_verified=None)


def _passing_sidecar_drift_signal_fn(
    script_path: Path, run_id: str, *, catalog_dir: str = ""
) -> SidecarDriftSignal:
    """Stub standing in for [GW-01]'s real `get_sidecar_drift_signal` -- same rationale as
    `_passing_evidence_candidate_fn` above."""
    return SidecarDriftSignal(drifted=False, script_id=str(script_path))


_CRITERIA = WatchdogCriteria(wall_clock_budget_seconds=3600.0)


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
        "evidence_candidate_fn": _passing_evidence_candidate_fn,
        "sidecar_drift_signal_fn": _passing_sidecar_drift_signal_fn,
    }


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
