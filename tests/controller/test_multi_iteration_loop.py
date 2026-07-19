"""Tests for controller.multi_iteration_loop (LC-10, epic #3611, AC-8b).

AC-8b's own measurable criterion: "a real multi-iteration run (>=3 candidates) terminates
correctly on both a budget-exhaustion path and a normal-completion path." Exercises:

1. A genuine multi-iteration run (>=3 real `run_one_candidate_pass` calls) completing normally
   after `max_candidates` -- `termination_reason="normal_completion"`.
2. A genuine multi-iteration run (>=3 real candidates dispatched) stopping via the SOFT,
   in-process wall-clock budget check -- `termination_reason="budget_exhausted"` -- BEFORE
   `max_candidates` is reached, using an injected fake `time_fn`, never the real watchdog.
3. The diversity-quota Leap-Path response genuinely firing: an all-cosmetically-identical
   candidate sequence fills `assert_diversity_quota`'s rolling window with `False` flags,
   producing a real `leap_path_required=True` decision that reaches the injected
   `on_leap_path_required` hook -- proving the window-maintenance logic itself, not a mocked
   decision object (verified against the real `xtrax.loop.diversity_quota` functions directly,
   see the module-level sanity check below).
4. A genuinely-diverse candidate sequence (structurally different bodies, not just renamed
   variables) filling the same window WITHOUT firing Leap-Path -- the contrasting case, proving
   #3 isn't a tautology.
5. The watchdog-starting dependency is invoked exactly once per run, never once per candidate.
6. `max_candidates < 1` raises `ValueError` *before* the watchdog is ever started.
7. An uncaught exception from `run_one_candidate_pass` propagates unmodified, AND the watchdog
   this function itself started is still stopped (ordinary resource cleanup, not retry policy).

## Watchdog safety (mandatory reading for anyone editing this file)

`xtrax.loop.external_stop_watchdog.start_watchdog` is a REAL function that spawns a REAL,
detached OS subprocess that SIGKILLs a target PID once a wall-clock budget elapses (see that
module's own docstring). **No test in this file calls it.** Every test injects
`start_watchdog_fn=<a _FakeWatchdogStarter instance>`, which never spawns a subprocess -- it
returns an in-memory `_FakeWatchdogHandle` stub. The one place this file references the real
`start_watchdog` at all (`TestDefaultStartWatchdogFnIsTheRealOne`) does so via `is` identity
comparison against the function object itself, and is explicit in its own docstring that this
is an identity check, never a call.
"""

import hashlib
import inspect
from pathlib import Path
from typing import Any

import pytest

from controller.bathos_campaign_adapter import BathosCampaignAdapter
from controller.dispatch import CandidateHandoff, CandidateHandoffFailure, MockFailureMode
from controller.multi_iteration_loop import (
    LeapPathEvent,
    MultiIterationLoopResult,
    run_multi_iteration_loop,
)
from xtrax.loop.attestation_evidence_gate import EvidenceCandidate
from xtrax.loop.diversity_quota import is_structural_mutation
from xtrax.loop.external_stop_watchdog import WatchdogCriteria, start_watchdog
from xtrax.loop.seed_gate import SeedTrialCounts
from xtrax.loop.sidecar_drift_gate import SidecarDriftSignal
from xtrax.loop.stats_battery_gate import BathosStatsBatteryVerdict

# ---------------------------------------------------------------------------
# Module-level sanity check on the measurement pipeline this suite relies on
# (mirrors the diversity_quota.py grounding note: verified, not assumed, before trusting it
# for a test's pass/fail signal). Structurally distinct bodies (different control flow / op)
# must diff as structural; identical sources must not.
# ---------------------------------------------------------------------------

_IDENTICAL_SOURCE = "def f(a, b):\n    return a + b\n"
_DIVERSE_SOURCE_LOOP = (
    "def f(a, b):\n    result = 0\n    for _ in range(a):\n        result += b\n    return result\n"
)
_DIVERSE_SOURCES = [
    "def f(a, b):\n    return a + b\n",
    "def f(a, b):\n    if a > b:\n        return a\n    return b\n",
    _DIVERSE_SOURCE_LOOP,
    "def f(a, b):\n    return a * b\n",
]


def test_sanity_identical_sources_are_not_structural_mutations() -> None:
    assert is_structural_mutation(_IDENTICAL_SOURCE, _IDENTICAL_SOURCE) is False


def test_sanity_diverse_sources_are_structural_mutations() -> None:
    for previous, candidate in zip(_DIVERSE_SOURCES, _DIVERSE_SOURCES[1:], strict=False):
        assert is_structural_mutation(previous, candidate) is True


# ---------------------------------------------------------------------------
# Fakes: watchdog starter/handle (NEVER the real subprocess-spawning start_watchdog)
# ---------------------------------------------------------------------------


class _FakeWatchdogHandle:
    """In-memory stand-in for `WatchdogHandle` -- no subprocess anywhere."""

    def __init__(self) -> None:
        self.stop_calls = 0

    def is_alive(self) -> bool:
        return self.stop_calls == 0

    def join(self, timeout: float | None = None) -> None:  # pragma: no cover - unused by driver
        pass

    def stop(self) -> None:
        self.stop_calls += 1


class _FakeWatchdogStarter:
    """Records every `(target_pid, criteria)` call; returns a fresh `_FakeWatchdogHandle`.

    Never spawns a real process -- this is the only watchdog-starter every test in this file
    uses.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[int, WatchdogCriteria]] = []
        self.handles: list[_FakeWatchdogHandle] = []

    def __call__(self, target_pid: int, criteria: WatchdogCriteria) -> _FakeWatchdogHandle:
        self.calls.append((target_pid, criteria))
        handle = _FakeWatchdogHandle()
        self.handles.append(handle)
        return handle


# ---------------------------------------------------------------------------
# Fake dispatch backend: MockDispatchBackend (LC-03) is deterministic/fixed for its whole
# lifetime (same path+content every call), which cannot exercise a genuinely multi-candidate
# diversity-quota window. This test-only backend writes a REAL file per call (LC-03's own
# CandidateHandoff.path contract: "must be readable by controller") cycling through a
# caller-supplied list of source strings.
# ---------------------------------------------------------------------------


class _SequentialDispatchBackend:
    def __init__(self, tmp_path: Path, sources: list[str]) -> None:
        self._tmp_path = tmp_path
        self._sources = sources
        self.call_count = 0

    def dispatch_candidate(self) -> CandidateHandoff:
        source = self._sources[self.call_count % len(self._sources)]
        path = self._tmp_path / f"candidate_{self.call_count}.py"
        path.write_text(source, encoding="utf-8")
        sha256_hex = hashlib.sha256(source.encode("utf-8")).hexdigest()
        self.call_count += 1
        return CandidateHandoff(path=path, content_sha256=sha256_hex)


class _SingleShotFailingDispatchBackend:
    """Raises `CandidateHandoffFailure` on its first (and only expected) call."""

    def dispatch_candidate(self) -> CandidateHandoff:
        msg = "staging write failed: permission denied or disk full"
        raise CandidateHandoffFailure(msg)


# ---------------------------------------------------------------------------
# Bathos + gate helpers (mirrors tests/controller/test_main_loop.py's own conventions)
# ---------------------------------------------------------------------------


def _ok_envelope(**extra: Any) -> dict[str, Any]:
    return {"ok": True, "error_code": None, "error": None, "resolution_hint": None, **extra}


class _RecordingTransport:
    def __init__(self, envelope: dict[str, Any]) -> None:
        self.envelope = envelope
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((tool_name, dict(arguments)))
        return self.envelope


def _adapter() -> BathosCampaignAdapter:
    envelope = _ok_envelope(script_path="candidate.py", exit_code=0, success=True)
    transport = _RecordingTransport(envelope)
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


def _passing_evidence_candidate_fn(
    run_id: str, *, catalog_dir: str = "", stdout_verified: bool | None = None
) -> EvidenceCandidate:
    """Stub standing in for [GW-01]'s real `get_evidence_candidate_for_run` -- without this,
    every LC-10 test that lets a candidate genuinely proceed would reach the REAL bathos-backed
    default, which requires the `controller` extra (bathos installed) that this dev-only test
    file must not depend on."""
    return EvidenceCandidate(run_id=run_id, manifest_verified=True, stdout_verified=None)


def _passing_sidecar_drift_signal_fn(
    script_path: Path, run_id: str, *, catalog_dir: str = ""
) -> SidecarDriftSignal:
    """Stub standing in for [GW-01]'s real `get_sidecar_drift_signal` -- same rationale as
    `_passing_evidence_candidate_fn` above."""
    return SidecarDriftSignal(drifted=False, script_id=str(script_path))


def _base_kwargs(dispatch_backend: Any) -> dict[str, Any]:
    return {
        "dispatch_backend": dispatch_backend,
        "campaign_adapter": _adapter(),
        "campaign_id": "camp-1",
        "campaign_mode": "exploration",
        "stats_battery_kwargs": {},
        "stats_battery_fn": _passing_stats_verdict,
        "seed_trial_counts_fn": _passing_seed_counts,
        "evidence_candidate_fn": _passing_evidence_candidate_fn,
        "sidecar_drift_signal_fn": _passing_sidecar_drift_signal_fn,
    }


_CRITERIA = WatchdogCriteria(wall_clock_budget_seconds=3600.0)


# ---------------------------------------------------------------------------
# 1. Normal completion (>=3 candidates), no leap path (genuinely diverse sequence)
# ---------------------------------------------------------------------------


class TestNormalCompletion:
    def test_diverse_run_completes_normally_with_no_leap_path(self, tmp_path: Path) -> None:
        dispatch_backend = _SequentialDispatchBackend(tmp_path, _DIVERSE_SOURCES)
        starter = _FakeWatchdogStarter()

        result = run_multi_iteration_loop(
            **_base_kwargs(dispatch_backend),
            max_candidates=4,
            watchdog_criteria=_CRITERIA,
            diversity_window_size=3,
            start_watchdog_fn=starter,
        )

        assert isinstance(result, MultiIterationLoopResult)
        assert result.termination_reason == "normal_completion"
        assert len(result.iterations) == 4
        assert dispatch_backend.call_count == 4
        assert result.accepted_count == 4
        # Genuinely diverse candidates fill the window with True flags -> quota stays met.
        assert result.leap_path_events == ()
        # Watchdog started exactly once for the whole run, not once per candidate.
        assert len(starter.calls) == 1
        assert starter.handles[0].stop_calls == 1


# ---------------------------------------------------------------------------
# 2. Leap-Path response genuinely fires (all-identical candidate sources)
# ---------------------------------------------------------------------------


class TestLeapPathResponseGenuinelyFires:
    def test_identical_sources_fill_window_and_fire_leap_path(self, tmp_path: Path) -> None:
        dispatch_backend = _SequentialDispatchBackend(tmp_path, [_IDENTICAL_SOURCE])
        starter = _FakeWatchdogStarter()
        received_events: list[LeapPathEvent] = []

        result = run_multi_iteration_loop(
            **_base_kwargs(dispatch_backend),
            max_candidates=4,
            watchdog_criteria=_CRITERIA,
            diversity_window_size=3,
            on_leap_path_required=received_events.append,
            start_watchdog_fn=starter,
        )

        assert result.termination_reason == "normal_completion"
        assert len(result.iterations) == 4
        # Window fills to size 3 (all False) exactly when the 4th candidate (index 3) completes.
        assert len(result.leap_path_events) == 1
        event = result.leap_path_events[0]
        assert event.iteration_index == 3
        assert event.decision.leap_path_required is True
        assert event.decision.window == (False, False, False)
        # The injected hook received exactly the same event(s) the result carries.
        assert received_events == list(result.leap_path_events)

    def test_diverse_sources_never_fire_leap_path(self, tmp_path: Path) -> None:
        """Contrasting case: proves the previous test isn't a tautology of the hook always
        firing -- a genuinely-exploring sequence must NOT trigger Leap-Path."""
        dispatch_backend = _SequentialDispatchBackend(tmp_path, _DIVERSE_SOURCES)
        starter = _FakeWatchdogStarter()
        received_events: list[LeapPathEvent] = []

        result = run_multi_iteration_loop(
            **_base_kwargs(dispatch_backend),
            max_candidates=4,
            watchdog_criteria=_CRITERIA,
            diversity_window_size=3,
            on_leap_path_required=received_events.append,
            start_watchdog_fn=starter,
        )

        assert result.leap_path_events == ()
        assert received_events == []


# ---------------------------------------------------------------------------
# 3. Budget-exhaustion path (>=3 candidates dispatched, then stops before max_candidates)
# ---------------------------------------------------------------------------


class TestBudgetExhaustion:
    def test_soft_budget_stops_run_before_max_candidates(self, tmp_path: Path) -> None:
        dispatch_backend = _SequentialDispatchBackend(tmp_path, _DIVERSE_SOURCES)
        starter = _FakeWatchdogStarter()
        # Consumed in order: loop_start, then one pre-dispatch check per candidate_index.
        # 3 real candidates (indices 0,1,2) pass their budget check (elapsed < 10); the 4th
        # check (index 3) observes elapsed=11 >= 10 and stops -- never reaching a 4th dispatch.
        fake_clock = iter([0.0, 0.0, 2.0, 4.0, 11.0])

        def fake_time_fn() -> float:
            return next(fake_clock)

        result = run_multi_iteration_loop(
            **_base_kwargs(dispatch_backend),
            max_candidates=5,
            watchdog_criteria=_CRITERIA,
            wall_clock_budget_seconds=10.0,
            time_fn=fake_time_fn,
            start_watchdog_fn=starter,
        )

        assert result.termination_reason == "budget_exhausted"
        assert len(result.iterations) == 3
        assert dispatch_backend.call_count == 3, (
            "the loop must genuinely stop dispatching once the soft budget trips, not merely "
            "label a full run as budget_exhausted after the fact"
        )
        assert len(starter.calls) == 1
        assert starter.handles[0].stop_calls == 1

    def test_no_wall_clock_budget_never_exhausts(self, tmp_path: Path) -> None:
        """`wall_clock_budget_seconds=None` (the default) disables the soft check entirely --
        regression guard against an accidental always-on budget."""
        dispatch_backend = _SequentialDispatchBackend(tmp_path, _DIVERSE_SOURCES)
        starter = _FakeWatchdogStarter()

        result = run_multi_iteration_loop(
            **_base_kwargs(dispatch_backend),
            max_candidates=3,
            watchdog_criteria=_CRITERIA,
            start_watchdog_fn=starter,
        )

        assert result.termination_reason == "normal_completion"
        assert len(result.iterations) == 3


# ---------------------------------------------------------------------------
# 4. Watchdog started exactly once, not once per candidate (explicit, dedicated check)
# ---------------------------------------------------------------------------


class TestWatchdogStartedExactlyOncePerRun:
    def test_five_candidates_one_watchdog_start(self, tmp_path: Path) -> None:
        dispatch_backend = _SequentialDispatchBackend(tmp_path, _DIVERSE_SOURCES)
        starter = _FakeWatchdogStarter()

        run_multi_iteration_loop(
            **_base_kwargs(dispatch_backend),
            max_candidates=5,
            watchdog_criteria=_CRITERIA,
            start_watchdog_fn=starter,
        )

        assert len(starter.calls) == 1
        target_pid, criteria = starter.calls[0]
        assert isinstance(target_pid, int)
        assert criteria is _CRITERIA


# ---------------------------------------------------------------------------
# 5. Input validation: max_candidates < 1 raises before the watchdog is ever started
# ---------------------------------------------------------------------------


class TestMaxCandidatesValidation:
    def test_zero_max_candidates_raises_value_error_before_watchdog_start(
        self, tmp_path: Path
    ) -> None:
        dispatch_backend = _SequentialDispatchBackend(tmp_path, _DIVERSE_SOURCES)
        starter = _FakeWatchdogStarter()

        with pytest.raises(ValueError, match="max_candidates must be >= 1"):
            run_multi_iteration_loop(
                **_base_kwargs(dispatch_backend),
                max_candidates=0,
                watchdog_criteria=_CRITERIA,
                start_watchdog_fn=starter,
            )

        assert starter.calls == [], "the watchdog must not be started for invalid input"


# ---------------------------------------------------------------------------
# 6. Uncaught exception propagates unmodified; watchdog cleanup still runs (LC-11 boundary)
# ---------------------------------------------------------------------------


class TestExceptionPropagatesAndWatchdogStillCleanedUp:
    def test_dispatch_failure_propagates_and_watchdog_is_still_stopped(self) -> None:
        dispatch_backend = _SingleShotFailingDispatchBackend()
        starter = _FakeWatchdogStarter()

        with pytest.raises(CandidateHandoffFailure):
            run_multi_iteration_loop(
                **_base_kwargs(dispatch_backend),
                max_candidates=3,
                watchdog_criteria=_CRITERIA,
                start_watchdog_fn=starter,
            )

        assert len(starter.calls) == 1
        assert starter.handles[0].stop_calls == 1, (
            "the watchdog this function started must still be stopped in its own finally "
            "block even though the run itself raised -- this is resource cleanup, not "
            "error/retry policy (explicitly LC-11's scope, not implemented here)"
        )


# ---------------------------------------------------------------------------
# 7. The real start_watchdog is the documented default -- identity check ONLY, never called
# ---------------------------------------------------------------------------


class TestDefaultStartWatchdogFnIsTheRealOne:
    def test_default_parameter_is_the_real_start_watchdog_by_identity(self) -> None:
        """Confirms the wiring the module docstring promises ("defaulting to the real
        start_watchdog") WITHOUT ever invoking it -- an `is` identity check against the
        function object, not a call. No test anywhere in this file calls the real
        `start_watchdog`.
        """
        sig = inspect.signature(run_multi_iteration_loop)
        default = sig.parameters["start_watchdog_fn"].default
        assert default is start_watchdog


# ---------------------------------------------------------------------------
# 8. MockDispatchBackend interaction note from the module docstring, exercised directly
# ---------------------------------------------------------------------------


class TestMockDispatchBackendFileContentAssumption:
    def test_mock_dispatch_backend_alone_does_not_write_file_content(self, tmp_path: Path) -> None:
        """Documents (and locks in) the exact interaction the module docstring warns about:
        `MockDispatchBackend` computes `content_sha256` from `candidate_content` but never
        writes it to `candidate_path` -- callers of `run_multi_iteration_loop` using
        `MockDispatchBackend` directly must pre-write matching file content themselves for the
        diversity-quota window to observe anything (this module's own
        `_SequentialDispatchBackend` above does so deliberately)."""
        from controller.dispatch import MockDispatchBackend

        candidate_path = tmp_path / "candidate.py"
        backend = MockDispatchBackend(
            candidate_path=candidate_path,
            candidate_content="def f():\n    return 1\n",
            mode=MockFailureMode.NONE,
        )
        backend.dispatch_candidate()
        assert not candidate_path.exists()
