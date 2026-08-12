"""Tests for controller.main_loop (LC-09, epic #3611, AC-8a; GW-02 ratchet wiring, #3649).

AC-8a's own measurable criterion: "a single iteration completes end-to-end against
MockDispatchBackend." Exercises:

1. The full one-candidate-pass sequence genuinely executes in order (dispatch -> lineage-
   resolved bathos run -> gate checks), against `MockDispatchBackend` (LC-03) + an injected
   `BathosCampaignAdapter` transport (LC-06) + injected gate-wrapper stubs standing in for
   LC-07 -- no live bathos/praxia infrastructure required for any test in this module.
2. `derived_from` lineage resolution genuinely reaches both the bathos `run` MCP call's
   arguments and the composed result.
3. A candidate that clears both gates: `OneCandidatePassResult.accepted` is `True`.
4. A candidate that fails a gate (a `confirmation`/`sequential`-mode campaign with a
   downgraded stats verdict, or an unmet seed/trial floor): `accepted` is `False` and
   `gate_outcome.hard_blocked` is `True` -- proving the gate-check step's outcome is
   genuinely load-bearing, not just plumbed through and ignored.
5. A genuinely multi-parent `parentage` raises `MultiParentLineageUnsupportedError` before any
   bathos call is made (dispatch already happened; the bathos transport is never invoked).
6. A dispatch-backend failure (`CandidateHandoffFailure`) propagates unmodified; no bathos or
   gate call is ever attempted -- this module performs zero retry logic (LC-11's own scope).
7. `GateOutcome.hard_blocked`/`OneCandidatePassResult.accepted` unit-level truth tables.
8. The candidate-static gate (T2-11, AC-1; [GW-04] first slice) genuinely runs right after
   dispatch, before lineage resolution or any bathos call: a failing candidate raises
   `CandidateStaticGateError` with zero bathos calls made, and the real (non-injected) default
   `assert_candidate_static` is genuinely wired in, not just the injection seam.

GW-02 (backlog #3649) additions -- `run_one_candidate_pass` now wires in the real multi-metric
ratchet decision, crash-safe best-so-far lineage (accept and reject paths), and compile-time
exclusion; see `TestRatchetDecisionDrivesAcceptance`, `TestFirstCandidateSentinel`,
`TestBestFitnessHigherIsBetterMustBothBeSuppliedOrOmitted`, `TestGuardedEvaluateFnExactArgs`,
`TestClosureDriftPropagatesUncaught`, and `TestCompileTimeExcludedFromRatchetComparison` below.
Every test in this module now supplies the new required kwargs
(`frozen_context`/`current_config`/`repo`/`ratchet_ref_name`/`commit_tree_sha`/`callable_name`/
`concrete_inputs`) via the shared `_new_step_kwargs()` helper, with `guarded_evaluate_fn`/
`measure_two_phase_timing_fn` stubbed by default (the real defaults would hash real closure
files / `jax.jit`-compile a real candidate callable, neither of which exists for
`MockDispatchBackend`'s synthetic candidate paths) and T2-10 crash-atomicity calls stubbed via
the autouse `_stub_crash_atomicity` fixture (most tests exercise dispatch/lineage/gate/ratchet
behavior, not real git operations).
"""

import hashlib
import inspect
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import controller.main_loop as main_loop_module
from controller.bathos_campaign_adapter import (
    BathosCampaignAdapter,
    BathosMcpToolError,
    CandidateRunResult,
)
from controller.dispatch import (
    CandidateHandoff,
    CandidateHandoffFailure,
    MockDispatchBackend,
    MockFailureMode,
)
from controller.evaluate_adapter import BathosFrozenContext, score_raw_artifacts
from controller.lineage_interim import CandidateParentage, MultiParentLineageUnsupportedError
from controller.main_loop import (
    GateOutcome,
    OneCandidatePassResult,
    PriorBestSoFarLineageConflictError,
    compute_candidate_tree_sha,
    run_one_candidate_pass,
)
from xtrax.loop.candidate_static import CandidateStaticGateError
from xtrax.loop.closure_lock import ClosureHashMismatchError, ClosureManifest, UnlistedReadError
from xtrax.loop.compile_time_clock import TwoPhaseTiming
from xtrax.loop.multi_metric_ratchet import RatchetDecision
from xtrax.loop.seed_gate import SeedTrialCounts, SeedTrialFloorDecision
from xtrax.loop.stats_battery_gate import BathosStatsBatteryVerdict, ConcludeStatsDecision

_CANDIDATE_CONTENT = "candidate-source"
_VALID_SHA256 = hashlib.sha256(_CANDIDATE_CONTENT.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# GW-02 shared fixtures/helpers: every call to run_one_candidate_pass now needs
# frozen_context/current_config/repo/ratchet_ref_name/commit_tree_sha/callable_name/
# concrete_inputs (all required, no default). See module docstring.
# ---------------------------------------------------------------------------

_FROZEN_CONTEXT = BathosFrozenContext(
    locked=ClosureManifest(
        evaluator_paths=(),
        split_paths=(),
        metric_def_paths=(),
        pinned_deps_source=Path("uv.lock"),
        config={},
        closure_hash="unused-in-tests-guarded_evaluate_fn-is-stubbed",
    ),
    campaign_adapter=None,  # type: ignore[arg-type]  # unused: guarded_evaluate_fn is stubbed
    campaign_id="camp-1",
    score_fn=lambda *_args: {},
)

_PASSING_FITNESS = {"accuracy": 0.9, "loss": 0.1}
_BEST_FITNESS = {"accuracy": 0.9, "loss": 0.1}
_HIGHER_IS_BETTER = {"accuracy": True, "loss": False}
_WORSE_FITNESS = {"accuracy": 0.1, "loss": 0.9}


def _passing_guarded_evaluate_fn(*args: Any, **kwargs: Any) -> dict[str, float]:
    return dict(_PASSING_FITNESS)


def _passing_timing_fn(*args: Any, **kwargs: Any) -> TwoPhaseTiming:
    return TwoPhaseTiming(compile_time_seconds=0.0, runtime_seconds=0.0, result=None)


def _new_step_kwargs(**overrides: Any) -> dict[str, Any]:
    """Default GW-02 kwargs shared by every pre-GW-02 test in this module.

    Tests specifically exercising ratchet/crash-atomicity/compile-time behavior override the
    relevant keys (e.g. `guarded_evaluate_fn`, `best_fitness`/`higher_is_better`).
    """
    defaults: dict[str, Any] = {
        "frozen_context": _FROZEN_CONTEXT,
        "current_config": {},
        "guarded_evaluate_fn": _passing_guarded_evaluate_fn,
        "repo": Path("unused-repo"),
        "ratchet_ref_name": "refs/xtrax/best-so-far",
        "commit_tree_sha": "unused-tree-sha",
        "commit_parent_sha": "unused-parent-sha",
        "callable_name": "unused_callable",
        "concrete_inputs": [],
        "measure_two_phase_timing_fn": _passing_timing_fn,
    }
    defaults.update(overrides)
    return defaults


@pytest.fixture(autouse=True)
def _stub_crash_atomicity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub every T2-10 crash-atomicity call by default -- most tests in this module exercise
    dispatch/lineage/gate/ratchet behavior, not real git operations. Tests that specifically
    assert on these calls re-patch the relevant stub within their own test body (a later
    `monkeypatch.setattr` call overrides this fixture's earlier one for the rest of that test).
    """
    monkeypatch.setattr(main_loop_module, "read_best_so_far", lambda repo, ref_name: None)
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


def _passing_candidate_static_fn(path: Path, root: Path | None = None) -> None:
    """Stub standing in for T2-11's real `assert_candidate_static` -- every existing LC-09 test
    exercises dispatch/lineage/gate behavior against a `MockDispatchBackend` candidate_path that
    doesn't exist on disk (e.g. `Path("candidate.py")`), so the REAL gate (which imports the
    file) would reject every one of them. Tests that exercise the static gate itself pass their
    own `candidate_static_fn` instead of this stub."""
    return None


def _ok_envelope(**extra: Any) -> dict[str, Any]:
    """A bathos `traced_tool`-shaped success envelope, matching
    test_bathos_campaign_adapter.py's own convention."""
    return {"ok": True, "error_code": None, "error": None, "resolution_hint": None, **extra}


class _RecordingTransport:
    """Injection seam for `BathosCampaignAdapter` -- records every call it receives and
    returns a pre-programmed envelope. Optionally appends a label to a shared `order` list, so
    tests can assert this module's own call sequencing without live bathos infrastructure.
    """

    def __init__(self, envelope: dict[str, Any], order: list[str] | None = None) -> None:
        self.envelope = envelope
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._order = order

    def __call__(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._order is not None:
            self._order.append(f"bathos:{tool_name}")
        self.calls.append((tool_name, dict(arguments)))
        return self.envelope


def _mock_dispatch_backend(mode: MockFailureMode = MockFailureMode.NONE) -> MockDispatchBackend:
    return MockDispatchBackend(
        candidate_path=Path("candidate.py"),
        candidate_content=_CANDIDATE_CONTENT,
        mode=mode,
    )


def _run_envelope() -> dict[str, Any]:
    return _ok_envelope(script_path="candidate.py", exit_code=0, success=True)


def _passing_stats_verdict() -> BathosStatsBatteryVerdict:
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


def _downgraded_stats_verdict() -> BathosStatsBatteryVerdict:
    return BathosStatsBatteryVerdict(
        verdict="confounded",
        scipy_available=True,
        reasons=("insufficient effect size",),
        cohens_d=0.05,
        win_rate=0.5,
        breakdown_point=0.05,
        p_superiority=0.5,
        wilcoxon_p_value=0.4,
        icc=0.8,
        baseline_budget_equivalent=False,
    )


def _passing_seed_counts() -> SeedTrialCounts:
    return SeedTrialCounts(script_sha256=_VALID_SHA256, distinct_seed_count=5, trial_count=40)


def _failing_seed_counts() -> SeedTrialCounts:
    return SeedTrialCounts(script_sha256=_VALID_SHA256, distinct_seed_count=1, trial_count=5)


# ---------------------------------------------------------------------------
# Flagship AC-8a test: full sequence genuinely executes in order
# ---------------------------------------------------------------------------


class TestFullSequenceOrdering:
    def test_dispatch_then_run_then_gates_in_order(self) -> None:
        order: list[str] = []

        class _OrderTrackingDispatch:
            """Wraps MockDispatchBackend to record when dispatch actually happens, relative
            to the other steps -- proving this module doesn't call bathos/gates first."""

            def __init__(self) -> None:
                self._inner = _mock_dispatch_backend()

            def dispatch_candidate(self) -> CandidateHandoff:
                order.append("dispatch")
                return self._inner.dispatch_candidate()

        transport = _RecordingTransport(_run_envelope(), order=order)
        adapter = BathosCampaignAdapter(transport=transport, token="test-token")

        def stats_fn(**kwargs: Any) -> BathosStatsBatteryVerdict:
            order.append("stats_battery")
            return _passing_stats_verdict()

        def seed_fn(db: Any, script_sha256: str, hypothesis_clause_id: str = "") -> SeedTrialCounts:
            order.append("seed_trial")
            return _passing_seed_counts()

        def candidate_static_fn(path: Path, root: Path | None = None) -> None:
            order.append("candidate_static")

        result = run_one_candidate_pass(
            _OrderTrackingDispatch(),
            adapter,
            campaign_id="camp-1",
            campaign_mode="exploration",
            candidate_static_fn=candidate_static_fn,
            stats_battery_kwargs={},
            stats_battery_fn=stats_fn,
            seed_trial_counts_fn=seed_fn,
            output_paths=["artifact.json"],
            **_new_step_kwargs(),
        )

        expected_order = [
            "dispatch",
            "candidate_static",
            "bathos:run",
            "stats_battery",
            "seed_trial",
        ]
        assert order == expected_order, (
            "the one-candidate pass must sequence dispatch -> candidate-static gate -> bathos "
            f"run -> gate checks, in that order -- got {order}"
        )
        assert isinstance(result, OneCandidatePassResult)
        assert result.accepted is True

    def test_a_complete_pass_produces_a_composed_result_bundling_every_piece(self) -> None:
        """A single iteration completes end-to-end against MockDispatchBackend (AC-8a's own
        literal AC text) and the composed result carries every piece downstream items need."""
        transport = _RecordingTransport(_run_envelope())
        adapter = BathosCampaignAdapter(transport=transport, token="test-token")

        result = run_one_candidate_pass(
            _mock_dispatch_backend(),
            adapter,
            campaign_id="camp-1",
            campaign_mode="exploration",
            candidate_static_fn=_passing_candidate_static_fn,
            stats_battery_kwargs={},
            stats_battery_fn=lambda **kw: _passing_stats_verdict(),
            seed_trial_counts_fn=lambda db, script_sha256, hypothesis_clause_id="": (
                _passing_seed_counts()
            ),
            output_paths=["artifact.json"],
            **_new_step_kwargs(),
        )

        assert result.handoff.content_sha256 == _VALID_SHA256
        assert result.derived_from == ""
        assert isinstance(result.run_result, CandidateRunResult)
        assert result.run_result.success is True
        assert isinstance(result.gate_outcome, GateOutcome)
        assert result.gate_outcome.stats_battery.honored is True
        assert result.gate_outcome.seed_trial.held is True
        assert result.accepted is True


# ---------------------------------------------------------------------------
# derived_from lineage threads end-to-end
# ---------------------------------------------------------------------------


class TestDerivedFromThreadedEndToEnd:
    def test_single_parent_derived_from_reaches_bathos_run_call_and_result(self) -> None:
        transport = _RecordingTransport(_run_envelope())
        adapter = BathosCampaignAdapter(transport=transport, token="test-token")
        parentage = CandidateParentage(parent_run_ids=("parent-run-uuid-123",))

        result = run_one_candidate_pass(
            _mock_dispatch_backend(),
            adapter,
            campaign_id="camp-1",
            campaign_mode="exploration",
            candidate_static_fn=_passing_candidate_static_fn,
            parentage=parentage,
            stats_battery_kwargs={},
            stats_battery_fn=lambda **kw: _passing_stats_verdict(),
            seed_trial_counts_fn=lambda db, script_sha256, hypothesis_clause_id="": (
                _passing_seed_counts()
            ),
            output_paths=["artifact.json"],
            **_new_step_kwargs(),
        )

        assert result.derived_from == "parent-run-uuid-123"
        assert len(transport.calls) == 1
        tool_name, arguments = transport.calls[0]
        assert tool_name == "run"
        assert arguments["derived_from"] == "parent-run-uuid-123"
        assert arguments["campaign_id"] == "camp-1"

    def test_root_candidate_threads_empty_string_derived_from(self) -> None:
        transport = _RecordingTransport(_run_envelope())
        adapter = BathosCampaignAdapter(transport=transport, token="test-token")

        result = run_one_candidate_pass(
            _mock_dispatch_backend(),
            adapter,
            campaign_id="camp-1",
            campaign_mode="exploration",
            candidate_static_fn=_passing_candidate_static_fn,
            stats_battery_kwargs={},
            stats_battery_fn=lambda **kw: _passing_stats_verdict(),
            seed_trial_counts_fn=lambda db, script_sha256, hypothesis_clause_id="": (
                _passing_seed_counts()
            ),
            output_paths=["artifact.json"],
            **_new_step_kwargs(),
        )

        assert result.derived_from == ""
        assert transport.calls[0][1]["derived_from"] == ""


class TestSeedTrialCountsReceivesHandoffSha256:
    def test_script_sha256_from_handoff_and_hypothesis_clause_id_forwarded(self) -> None:
        received: dict[str, Any] = {}

        def seed_fn(db: Any, script_sha256: str, hypothesis_clause_id: str = "") -> SeedTrialCounts:
            received["script_sha256"] = script_sha256
            received["hypothesis_clause_id"] = hypothesis_clause_id
            return _passing_seed_counts()

        adapter = BathosCampaignAdapter(transport=_RecordingTransport(_run_envelope()), token="t")

        run_one_candidate_pass(
            _mock_dispatch_backend(),
            adapter,
            campaign_id="camp-1",
            campaign_mode="exploration",
            candidate_static_fn=_passing_candidate_static_fn,
            hypothesis_clause_id="clause-1",
            stats_battery_kwargs={},
            stats_battery_fn=lambda **kw: _passing_stats_verdict(),
            seed_trial_counts_fn=seed_fn,
            output_paths=["artifact.json"],
            **_new_step_kwargs(),
        )

        assert received["script_sha256"] == _VALID_SHA256
        assert received["hypothesis_clause_id"] == "clause-1"


# ---------------------------------------------------------------------------
# Gate checks are genuinely load-bearing, not plumbed-through-and-ignored
# ---------------------------------------------------------------------------


class TestGateCheckIsLoadBearing:
    def test_both_gates_clearing_accepts_a_confirmation_campaign(self) -> None:
        adapter = BathosCampaignAdapter(transport=_RecordingTransport(_run_envelope()), token="t")

        result = run_one_candidate_pass(
            _mock_dispatch_backend(),
            adapter,
            campaign_id="camp-1",
            campaign_mode="confirmation",
            candidate_static_fn=_passing_candidate_static_fn,
            stats_battery_kwargs={},
            stats_battery_fn=lambda **kw: _passing_stats_verdict(),
            seed_trial_counts_fn=lambda db, script_sha256, hypothesis_clause_id="": (
                _passing_seed_counts()
            ),
            output_paths=["artifact.json"],
            **_new_step_kwargs(),
        )

        assert result.gate_outcome.hard_blocked is False
        assert result.accepted is True

    def test_downgraded_stats_verdict_hard_blocks_confirmation_campaign(self) -> None:
        adapter = BathosCampaignAdapter(transport=_RecordingTransport(_run_envelope()), token="t")

        result = run_one_candidate_pass(
            _mock_dispatch_backend(),
            adapter,
            campaign_id="camp-1",
            campaign_mode="confirmation",
            candidate_static_fn=_passing_candidate_static_fn,
            stats_battery_kwargs={},
            stats_battery_fn=lambda **kw: _downgraded_stats_verdict(),
            seed_trial_counts_fn=lambda db, script_sha256, hypothesis_clause_id="": (
                _passing_seed_counts()
            ),
            output_paths=["artifact.json"],
            **_new_step_kwargs(),
        )

        assert result.gate_outcome.stats_battery.hard_blocked is True
        assert result.gate_outcome.hard_blocked is True
        assert result.accepted is False
        # The bathos run itself succeeded -- the rejection comes from the gate check, not
        # from a failed run. Proves the gate outcome, not just run success, drives acceptance.
        assert result.run_result.success is True

    def test_unmet_seed_trial_floor_hard_blocks_sequential_campaign(self) -> None:
        adapter = BathosCampaignAdapter(transport=_RecordingTransport(_run_envelope()), token="t")

        result = run_one_candidate_pass(
            _mock_dispatch_backend(),
            adapter,
            campaign_id="camp-1",
            campaign_mode="sequential",
            candidate_static_fn=_passing_candidate_static_fn,
            stats_battery_kwargs={},
            stats_battery_fn=lambda **kw: _passing_stats_verdict(),
            seed_trial_counts_fn=lambda db, script_sha256, hypothesis_clause_id="": (
                _failing_seed_counts()
            ),
            output_paths=["artifact.json"],
            **_new_step_kwargs(),
        )

        assert result.gate_outcome.seed_trial.hard_blocked is True
        assert result.gate_outcome.hard_blocked is True
        assert result.accepted is False

    def test_downgraded_verdict_is_advisory_only_for_exploration_campaign(self) -> None:
        """An advisory-only downgrade (exploration mode) must NOT flip acceptance -- both
        gate modules' own docstrings treat this as a normal, expected campaign state."""
        adapter = BathosCampaignAdapter(transport=_RecordingTransport(_run_envelope()), token="t")

        result = run_one_candidate_pass(
            _mock_dispatch_backend(),
            adapter,
            campaign_id="camp-1",
            campaign_mode="exploration",
            candidate_static_fn=_passing_candidate_static_fn,
            stats_battery_kwargs={},
            stats_battery_fn=lambda **kw: _downgraded_stats_verdict(),
            seed_trial_counts_fn=lambda db, script_sha256, hypothesis_clause_id="": (
                _failing_seed_counts()
            ),
            output_paths=["artifact.json"],
            **_new_step_kwargs(),
        )

        assert result.gate_outcome.stats_battery.advisory is True
        assert result.gate_outcome.seed_trial.advisory is True
        assert result.gate_outcome.hard_blocked is False
        assert result.accepted is True


# ---------------------------------------------------------------------------
# Multi-parent lineage fails loud, before any bathos call (AC-7's own requirement)
# ---------------------------------------------------------------------------


class TestMultiParentFailsLoudBeforeBathosCall:
    def test_genuine_multi_parent_raises_before_any_bathos_call(self) -> None:
        transport = _RecordingTransport(_run_envelope())
        adapter = BathosCampaignAdapter(transport=transport, token="t")
        parentage = CandidateParentage(parent_run_ids=("run-a", "run-b"))

        with pytest.raises(MultiParentLineageUnsupportedError):
            run_one_candidate_pass(
                _mock_dispatch_backend(),
                adapter,
                campaign_id="camp-1",
                campaign_mode="exploration",
                candidate_static_fn=_passing_candidate_static_fn,
                parentage=parentage,
                stats_battery_kwargs={},
                output_paths=["artifact.json"],
                **_new_step_kwargs(),
            )

        assert transport.calls == [], (
            "no bathos call should ever happen for a genuine multi-parent parentage -- the "
            "exception must fire before adapter.run is reached"
        )

    def test_duplicate_parent_is_not_multi_parent_and_proceeds(self) -> None:
        """The same parent listed twice is one real parent, not a multi-parent merge -- must
        not raise (regression guard mirroring test_lineage_interim.py's own distinction)."""
        transport = _RecordingTransport(_run_envelope())
        adapter = BathosCampaignAdapter(transport=transport, token="t")
        parentage = CandidateParentage(parent_run_ids=("run-a", "run-a"))

        result = run_one_candidate_pass(
            _mock_dispatch_backend(),
            adapter,
            campaign_id="camp-1",
            campaign_mode="exploration",
            candidate_static_fn=_passing_candidate_static_fn,
            parentage=parentage,
            stats_battery_kwargs={},
            stats_battery_fn=lambda **kw: _passing_stats_verdict(),
            seed_trial_counts_fn=lambda db, script_sha256, hypothesis_clause_id="": (
                _passing_seed_counts()
            ),
            output_paths=["artifact.json"],
            **_new_step_kwargs(),
        )

        assert result.derived_from == "run-a"
        assert len(transport.calls) == 1


# ---------------------------------------------------------------------------
# Dispatch failure propagates unmodified; zero retry logic (LC-11's own scope)
# ---------------------------------------------------------------------------


class TestDispatchFailurePropagatesWithNoRetry:
    def test_candidate_handoff_failure_propagates_and_skips_downstream_steps(self) -> None:
        dispatch_backend = _mock_dispatch_backend(mode=MockFailureMode.CANDIDATE_HANDOFF_FAILURE)
        transport = _RecordingTransport(_run_envelope())
        adapter = BathosCampaignAdapter(transport=transport, token="t")
        gate_calls: list[str] = []

        def stats_fn(**kwargs: Any) -> BathosStatsBatteryVerdict:
            gate_calls.append("stats")
            return _passing_stats_verdict()

        with pytest.raises(CandidateHandoffFailure):
            run_one_candidate_pass(
                dispatch_backend,
                adapter,
                campaign_id="camp-1",
                campaign_mode="exploration",
                candidate_static_fn=_passing_candidate_static_fn,
                stats_battery_kwargs={},
                stats_battery_fn=stats_fn,
                output_paths=["artifact.json"],
                **_new_step_kwargs(),
            )

        assert transport.calls == [], "no bathos call should happen after a dispatch failure"
        assert gate_calls == [], "no gate check should happen after a dispatch failure"

    def test_malformed_completion_raises_value_error_and_skips_downstream_steps(self) -> None:
        dispatch_backend = _mock_dispatch_backend(mode=MockFailureMode.MALFORMED_COMPLETION)
        transport = _RecordingTransport(_run_envelope())
        adapter = BathosCampaignAdapter(transport=transport, token="t")

        with pytest.raises(ValueError, match="malformed dispatch completion"):
            run_one_candidate_pass(
                dispatch_backend,
                adapter,
                campaign_id="camp-1",
                campaign_mode="exploration",
                candidate_static_fn=_passing_candidate_static_fn,
                stats_battery_kwargs={},
                output_paths=["artifact.json"],
                **_new_step_kwargs(),
            )

        assert transport.calls == []

    def test_timeout_raises_and_skips_downstream_steps(self) -> None:
        dispatch_backend = _mock_dispatch_backend(mode=MockFailureMode.TIMEOUT)
        dispatch_backend.timeout_delay = 0.01
        transport = _RecordingTransport(_run_envelope())
        adapter = BathosCampaignAdapter(transport=transport, token="t")

        with pytest.raises(TimeoutError):
            run_one_candidate_pass(
                dispatch_backend,
                adapter,
                campaign_id="camp-1",
                campaign_mode="exploration",
                candidate_static_fn=_passing_candidate_static_fn,
                stats_battery_kwargs={},
                output_paths=["artifact.json"],
                **_new_step_kwargs(),
            )

        assert transport.calls == []


# ---------------------------------------------------------------------------
# Bathos-run failure propagates unmodified; gate checks never attempted
# (audit finding on PR #73: only dispatch-side failure propagation was tested).
# ---------------------------------------------------------------------------


class TestBathosRunFailurePropagatesWithNoRetry:
    def test_bathos_mcp_tool_error_propagates_and_skips_gate_checks(self) -> None:
        failing_envelope = {
            "ok": False,
            "error_code": "script_not_found",
            "error": "candidate.py does not exist",
            "resolution_hint": None,
        }
        transport = _RecordingTransport(failing_envelope)
        adapter = BathosCampaignAdapter(transport=transport, token="t")
        gate_calls: list[str] = []

        def stats_fn(**kwargs: Any) -> BathosStatsBatteryVerdict:
            gate_calls.append("stats")
            return _passing_stats_verdict()

        def seed_fn(db: Any, script_sha256: str, hypothesis_clause_id: str = "") -> SeedTrialCounts:
            gate_calls.append("seed")
            return _passing_seed_counts()

        with pytest.raises(BathosMcpToolError, match="candidate.py does not exist"):
            run_one_candidate_pass(
                _mock_dispatch_backend(),
                adapter,
                campaign_id="camp-1",
                campaign_mode="exploration",
                candidate_static_fn=_passing_candidate_static_fn,
                stats_battery_kwargs={},
                stats_battery_fn=stats_fn,
                seed_trial_counts_fn=seed_fn,
                output_paths=["artifact.json"],
                **_new_step_kwargs(),
            )

        assert len(transport.calls) == 1, "the bathos run call itself should still be attempted"
        assert gate_calls == [], "no gate check should run after the bathos run call fails"


# ---------------------------------------------------------------------------
# stats_battery_kwargs / seed_trial_db forwarding (audit finding on PR #73: every
# existing test passed stats_battery_kwargs={} and never explicitly passed seed_trial_db).
# ---------------------------------------------------------------------------


class TestGateParameterForwarding:
    def test_non_empty_stats_battery_kwargs_reach_stats_battery_fn(self) -> None:
        transport = _RecordingTransport(_run_envelope())
        adapter = BathosCampaignAdapter(transport=transport, token="t")
        received_kwargs: dict[str, Any] = {}

        def stats_fn(**kwargs: Any) -> BathosStatsBatteryVerdict:
            received_kwargs.update(kwargs)
            return _passing_stats_verdict()

        run_one_candidate_pass(
            _mock_dispatch_backend(),
            adapter,
            campaign_id="camp-1",
            campaign_mode="exploration",
            candidate_static_fn=_passing_candidate_static_fn,
            stats_battery_kwargs={
                "candidate_values": [1, 2, 3],
                "baseline_values": [0.5, 1.5, 2.5],
                "baseline_hpo_trials": 10,
            },
            stats_battery_fn=stats_fn,
            seed_trial_counts_fn=lambda db, script_sha256, hypothesis_clause_id="": (
                _passing_seed_counts()
            ),
            output_paths=["artifact.json"],
            **_new_step_kwargs(),
        )

        assert received_kwargs == {
            "candidate_values": [1, 2, 3],
            "baseline_values": [0.5, 1.5, 2.5],
            "baseline_hpo_trials": 10,
        }

    def test_seed_trial_db_forwarded_to_seed_trial_counts_fn(self) -> None:
        transport = _RecordingTransport(_run_envelope())
        adapter = BathosCampaignAdapter(transport=transport, token="t")
        sentinel_db = object()
        received_db: list[Any] = []

        def seed_fn(db: Any, script_sha256: str, hypothesis_clause_id: str = "") -> SeedTrialCounts:
            received_db.append(db)
            return _passing_seed_counts()

        run_one_candidate_pass(
            _mock_dispatch_backend(),
            adapter,
            campaign_id="camp-1",
            campaign_mode="exploration",
            candidate_static_fn=_passing_candidate_static_fn,
            stats_battery_kwargs={},
            stats_battery_fn=lambda **kw: _passing_stats_verdict(),
            seed_trial_db=sentinel_db,
            seed_trial_counts_fn=seed_fn,
            output_paths=["artifact.json"],
            **_new_step_kwargs(),
        )

        assert received_db == [sentinel_db]


# ---------------------------------------------------------------------------
# GateOutcome.hard_blocked / OneCandidatePassResult.accepted: unit truth tables
# ---------------------------------------------------------------------------


def _stats_decision(*, hard_blocked: bool) -> ConcludeStatsDecision:
    return ConcludeStatsDecision(
        honored=not hard_blocked, hard_blocked=hard_blocked, advisory=False, reasons=()
    )


def _seed_decision(*, hard_blocked: bool) -> SeedTrialFloorDecision:
    return SeedTrialFloorDecision(
        held=not hard_blocked,
        hard_blocked=hard_blocked,
        advisory=False,
        script_sha256=_VALID_SHA256,
        hypothesis_clause_id="",
        distinct_seed_count=5,
        trial_count=40,
    )


class TestGateOutcomeHardBlocked:
    def test_neither_hard_blocked(self) -> None:
        outcome = GateOutcome(
            stats_battery=_stats_decision(hard_blocked=False),
            seed_trial=_seed_decision(hard_blocked=False),
        )
        assert outcome.hard_blocked is False

    def test_stats_hard_blocked_alone(self) -> None:
        outcome = GateOutcome(
            stats_battery=_stats_decision(hard_blocked=True),
            seed_trial=_seed_decision(hard_blocked=False),
        )
        assert outcome.hard_blocked is True

    def test_seed_hard_blocked_alone(self) -> None:
        outcome = GateOutcome(
            stats_battery=_stats_decision(hard_blocked=False),
            seed_trial=_seed_decision(hard_blocked=True),
        )
        assert outcome.hard_blocked is True

    def test_both_hard_blocked(self) -> None:
        outcome = GateOutcome(
            stats_battery=_stats_decision(hard_blocked=True),
            seed_trial=_seed_decision(hard_blocked=True),
        )
        assert outcome.hard_blocked is True


_SENTINEL_RATCHET_DECISION = RatchetDecision(
    improved=True, win_rate=1.0, breakdown_point=1.0, cohens_d=float("inf"), per_metric_delta={}
)

_REJECTING_RATCHET_DECISION = RatchetDecision(
    improved=False,
    win_rate=0.0,
    breakdown_point=0.0,
    cohens_d=-1.0,
    per_metric_delta={"loss": -1.0},
)


class TestOneCandidatePassResultAccepted:
    def _result(
        self, *, run_success: bool, hard_blocked: bool, improved: bool = True
    ) -> OneCandidatePassResult:
        return OneCandidatePassResult(
            handoff=CandidateHandoff(path=Path("c.py"), content_sha256=_VALID_SHA256),
            derived_from="",
            run_result=CandidateRunResult(script_path="c.py", exit_code=0, success=run_success),
            gate_outcome=GateOutcome(
                stats_battery=_stats_decision(hard_blocked=hard_blocked),
                seed_trial=_seed_decision(hard_blocked=False),
            ),
            ratchet_decision=(
                _SENTINEL_RATCHET_DECISION if improved else _REJECTING_RATCHET_DECISION
            ),
            fitness_dict=dict(_PASSING_FITNESS),
        )

    def test_run_failure_alone_rejects_even_with_clean_gates(self) -> None:
        result = self._result(run_success=False, hard_blocked=False)
        assert result.accepted is False

    def test_gate_hard_block_alone_rejects_even_with_successful_run(self) -> None:
        result = self._result(run_success=True, hard_blocked=True)
        assert result.accepted is False

    def test_success_and_clean_gates_accepts(self) -> None:
        result = self._result(run_success=True, hard_blocked=False)
        assert result.accepted is True

    def test_failure_and_hard_block_rejects(self) -> None:
        result = self._result(run_success=False, hard_blocked=True)
        assert result.accepted is False

    def test_ratchet_decision_not_improved_alone_rejects_even_with_success_and_clean_gates(
        self,
    ) -> None:
        """GW-02's own core fix: a candidate whose run succeeded and cleared both gates must
        still be rejected if the ratchet decision itself says it did not improve."""
        result = self._result(run_success=True, hard_blocked=False, improved=False)
        assert result.accepted is False

    def test_success_clean_gates_and_improved_ratchet_accepts(self) -> None:
        result = self._result(run_success=True, hard_blocked=False, improved=True)
        assert result.accepted is True


# ---------------------------------------------------------------------------
# Candidate-static gate (T2-11, AC-1; [GW-04] first slice): a genuine pre-dispatch reject,
# not merely an injection seam that's never exercised by its real default.
# ---------------------------------------------------------------------------


class TestCandidateStaticGate:
    def test_failing_candidate_static_raises_before_lineage_or_bathos_call(self) -> None:
        transport = _RecordingTransport(_run_envelope())
        adapter = BathosCampaignAdapter(transport=transport, token="t")
        parentage = CandidateParentage(parent_run_ids=("run-a", "run-b"))

        def failing_candidate_static_fn(path: Path, root: Path | None = None) -> None:
            msg = f"candidate {path} failed static checks"
            raise CandidateStaticGateError(msg)

        with pytest.raises(CandidateStaticGateError):
            run_one_candidate_pass(
                _mock_dispatch_backend(),
                adapter,
                campaign_id="camp-1",
                campaign_mode="exploration",
                # A genuinely multi-parent parentage would itself raise inside
                # resolve_derived_from -- passing it here proves candidate_static_fn fires
                # and raises *before* that lineage-resolution step is ever reached.
                parentage=parentage,
                candidate_static_fn=failing_candidate_static_fn,
                stats_battery_kwargs={},
                output_paths=["artifact.json"],
                **_new_step_kwargs(),
            )

        assert transport.calls == [], (
            "no bathos call should ever happen when the candidate-static gate rejects -- the "
            "exception must fire before adapter.run is reached"
        )

    def test_default_candidate_static_fn_is_the_real_gate_and_rejects_a_syntax_error(
        self, tmp_path: Path
    ) -> None:
        """Omitting `candidate_static_fn` entirely must exercise T2-11's REAL
        `assert_candidate_static` -- not just prove the injection seam works."""
        candidate_path = tmp_path / "bad_syntax.py"
        candidate_path.write_text("def f(:\n    pass")
        dispatch_backend = MockDispatchBackend(
            candidate_path=candidate_path,
            candidate_content=_CANDIDATE_CONTENT,
        )
        transport = _RecordingTransport(_run_envelope())
        adapter = BathosCampaignAdapter(transport=transport, token="t")

        with pytest.raises(CandidateStaticGateError, match="failed static checks"):
            run_one_candidate_pass(
                dispatch_backend,
                adapter,
                campaign_id="camp-1",
                campaign_mode="exploration",
                candidate_static_root=tmp_path,
                stats_battery_kwargs={},
                output_paths=["artifact.json"],
                **_new_step_kwargs(),
            )

        assert transport.calls == [], "a syntactically invalid candidate must burn zero real run"

    def test_default_candidate_static_fn_lets_a_clean_candidate_proceed(
        self, tmp_path: Path
    ) -> None:
        """The real gate must not block a genuinely clean candidate -- the rest of the pass
        still runs to completion end-to-end."""
        candidate_path = tmp_path / "clean.py"
        candidate_path.write_text('"""A minimal clean module."""\n')
        dispatch_backend = MockDispatchBackend(
            candidate_path=candidate_path,
            candidate_content=_CANDIDATE_CONTENT,
        )
        transport = _RecordingTransport(_run_envelope())
        adapter = BathosCampaignAdapter(transport=transport, token="t")

        result = run_one_candidate_pass(
            dispatch_backend,
            adapter,
            campaign_id="camp-1",
            campaign_mode="exploration",
            candidate_static_root=tmp_path,
            stats_battery_kwargs={},
            stats_battery_fn=lambda **kw: _passing_stats_verdict(),
            seed_trial_counts_fn=lambda db, script_sha256, hypothesis_clause_id="": (
                _passing_seed_counts()
            ),
            output_paths=["artifact.json"],
            **_new_step_kwargs(),
        )

        assert len(transport.calls) == 1
        assert result.accepted is True


# ---------------------------------------------------------------------------
# GW-02 (backlog #3649): the real multi-metric ratchet decision, crash-safe best-so-far
# lineage (accept and reject paths), and compile-time exclusion.
# ---------------------------------------------------------------------------


class TestRatchetRejectsWorseCandidateAndResetsWorktree:
    def test_strictly_worse_candidate_rejected_and_resets_worktree_exactly_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """VERIFY item 1: a candidate strictly worse on every metric than best_fitness ->
        accepted=False, reset_worktree_to_best_so_far invoked exactly once."""
        reset_calls: list[tuple[Path, str]] = []

        def spy_reset(repo: Path, ref_name: str) -> str:
            reset_calls.append((repo, ref_name))
            return "best-sha"

        monkeypatch.setattr(main_loop_module, "reset_worktree_to_best_so_far", spy_reset)

        adapter = BathosCampaignAdapter(transport=_RecordingTransport(_run_envelope()), token="t")

        def worse_guarded_evaluate_fn(*args: Any, **kwargs: Any) -> dict[str, float]:
            return dict(_WORSE_FITNESS)

        result = run_one_candidate_pass(
            _mock_dispatch_backend(),
            adapter,
            campaign_id="camp-1",
            campaign_mode="exploration",
            candidate_static_fn=_passing_candidate_static_fn,
            stats_battery_kwargs={},
            stats_battery_fn=lambda **kw: _passing_stats_verdict(),
            seed_trial_counts_fn=lambda db, script_sha256, hypothesis_clause_id="": (
                _passing_seed_counts()
            ),
            output_paths=["artifact.json"],
            **_new_step_kwargs(
                guarded_evaluate_fn=worse_guarded_evaluate_fn,
                best_fitness=_BEST_FITNESS,
                higher_is_better=_HIGHER_IS_BETTER,
            ),
        )

        assert result.ratchet_decision.improved is False
        assert result.accepted is False
        assert reset_calls == [(Path("unused-repo"), "refs/xtrax/best-so-far")]


class TestFirstCandidateSentinelAccept:
    def test_both_none_accepts_without_compute_ratchet_decision_and_falls_back_to_parent_sha(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """VERIFY item 2: best_fitness=None, higher_is_better=None -> accepted=True without
        compute_ratchet_decision being called; create_pending_commit/advance_best_so_far
        invoked with parent_sha falling back to commit_parent_sha and expected_old_sha=None."""
        compute_calls: list[Any] = []
        monkeypatch.setattr(
            main_loop_module,
            "compute_ratchet_decision",
            lambda *a, **kw: compute_calls.append((a, kw)),
        )

        create_calls: list[tuple[Any, Any, Any, Any]] = []
        advance_calls: list[tuple[Any, Any, Any, Any]] = []

        def spy_create(repo: Any, tree_sha: Any, parent_sha: Any, message: Any) -> str:
            create_calls.append((repo, tree_sha, parent_sha, message))
            return "pending-sha"

        def spy_advance(repo: Any, ref_name: Any, new_sha: Any, expected_old_sha: Any) -> None:
            advance_calls.append((repo, ref_name, new_sha, expected_old_sha))

        monkeypatch.setattr(main_loop_module, "create_pending_commit", spy_create)
        monkeypatch.setattr(main_loop_module, "advance_best_so_far", spy_advance)
        # read_best_so_far is already stubbed to return None by the autouse fixture.

        adapter = BathosCampaignAdapter(transport=_RecordingTransport(_run_envelope()), token="t")

        result = run_one_candidate_pass(
            _mock_dispatch_backend(),
            adapter,
            campaign_id="camp-1",
            campaign_mode="exploration",
            candidate_static_fn=_passing_candidate_static_fn,
            stats_battery_kwargs={},
            stats_battery_fn=lambda **kw: _passing_stats_verdict(),
            seed_trial_counts_fn=lambda db, script_sha256, hypothesis_clause_id="": (
                _passing_seed_counts()
            ),
            output_paths=["artifact.json"],
            **_new_step_kwargs(
                best_fitness=None,
                higher_is_better=None,
                commit_parent_sha="bootstrap-parent-sha",
            ),
        )

        assert result.accepted is True
        assert result.ratchet_decision.improved is True
        assert compute_calls == [], (
            "compute_ratchet_decision must never be called for the first candidate in a "
            "campaign -- the sentinel RatchetDecision is constructed directly"
        )
        assert len(create_calls) == 1
        _, _, parent_sha, _ = create_calls[0]
        assert parent_sha == "bootstrap-parent-sha"
        assert len(advance_calls) == 1
        _, _, _, expected_old_sha = advance_calls[0]
        assert expected_old_sha is None


class TestBestFitnessHigherIsBetterMustBothBeSuppliedOrOmitted:
    def test_one_none_one_not_raises_before_any_ratchet_or_crash_atomicity_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """VERIFY item 3: mismatched best_fitness/higher_is_better (one None, one not) raises
        ValueError before any ratchet or crash-atomicity call."""
        compute_calls: list[Any] = []
        create_calls: list[Any] = []
        advance_calls: list[Any] = []
        reset_calls: list[Any] = []
        monkeypatch.setattr(
            main_loop_module,
            "compute_ratchet_decision",
            lambda *a, **kw: compute_calls.append(1),
        )
        monkeypatch.setattr(
            main_loop_module, "create_pending_commit", lambda *a, **kw: create_calls.append(1)
        )
        monkeypatch.setattr(
            main_loop_module, "advance_best_so_far", lambda *a, **kw: advance_calls.append(1)
        )
        monkeypatch.setattr(
            main_loop_module,
            "reset_worktree_to_best_so_far",
            lambda *a, **kw: reset_calls.append(1),
        )

        adapter = BathosCampaignAdapter(transport=_RecordingTransport(_run_envelope()), token="t")

        with pytest.raises(
            ValueError,
            match="best_fitness and higher_is_better must both be supplied or both omitted",
        ):
            run_one_candidate_pass(
                _mock_dispatch_backend(),
                adapter,
                campaign_id="camp-1",
                campaign_mode="exploration",
                candidate_static_fn=_passing_candidate_static_fn,
                stats_battery_kwargs={},
                stats_battery_fn=lambda **kw: _passing_stats_verdict(),
                seed_trial_counts_fn=lambda db, script_sha256, hypothesis_clause_id="": (
                    _passing_seed_counts()
                ),
                output_paths=["artifact.json"],
                **_new_step_kwargs(best_fitness=_BEST_FITNESS, higher_is_better=None),
            )

        assert compute_calls == []
        assert create_calls == []
        assert advance_calls == []
        assert reset_calls == []


class TestGuardedEvaluateFnExactArgs:
    def test_guarded_evaluate_fn_called_with_step1_spec_args(self) -> None:
        """VERIFY item 4: guarded_evaluate_fn is called with the exact args spec'd in Step 1 --
        never anything read off CandidateRunResult."""
        captured: dict[str, Any] = {}

        def spy_guarded_evaluate_fn(
            locked: Any,
            evaluator: Any,
            frozen_context: Any,
            candidate: Any,
            *,
            current_config: Any,
            candidate_touched_paths: Any,
        ) -> dict[str, float]:
            captured["locked"] = locked
            captured["evaluator"] = evaluator
            captured["frozen_context"] = frozen_context
            captured["candidate"] = candidate
            captured["current_config"] = current_config
            captured["candidate_touched_paths"] = candidate_touched_paths
            return dict(_PASSING_FITNESS)

        adapter = BathosCampaignAdapter(transport=_RecordingTransport(_run_envelope()), token="t")
        sentinel_config = {"lr": 0.01}
        sentinel_touched = frozenset({Path("touched.py")})

        run_one_candidate_pass(
            _mock_dispatch_backend(),
            adapter,
            campaign_id="camp-1",
            campaign_mode="exploration",
            candidate_static_fn=_passing_candidate_static_fn,
            stats_battery_kwargs={},
            stats_battery_fn=lambda **kw: _passing_stats_verdict(),
            seed_trial_counts_fn=lambda db, script_sha256, hypothesis_clause_id="": (
                _passing_seed_counts()
            ),
            output_paths=["artifact.json", "artifact2.json"],
            **_new_step_kwargs(
                guarded_evaluate_fn=spy_guarded_evaluate_fn,
                current_config=sentinel_config,
                candidate_touched_paths=sentinel_touched,
            ),
        )

        assert captured["locked"] is _FROZEN_CONTEXT.locked
        assert captured["evaluator"] is score_raw_artifacts
        assert captured["frozen_context"] is _FROZEN_CONTEXT
        assert captured["candidate"] == ("artifact.json", "artifact2.json")
        assert captured["current_config"] is sentinel_config
        assert captured["candidate_touched_paths"] is sentinel_touched


class TestClosureDriftPropagatesUncaught:
    @pytest.mark.parametrize("exc_cls", [ClosureHashMismatchError, UnlistedReadError])
    def test_closure_drift_from_guarded_evaluate_fn_propagates_uncaught(
        self, exc_cls: type[Exception]
    ) -> None:
        """VERIFY item 5: ClosureHashMismatchError/UnlistedReadError injected from
        guarded_evaluate_fn propagates UNCAUGHT out of run_one_candidate_pass."""

        def failing_guarded_evaluate_fn(*args: Any, **kwargs: Any) -> dict[str, float]:
            msg = "closure drift injected for test"
            raise exc_cls(msg)

        adapter = BathosCampaignAdapter(transport=_RecordingTransport(_run_envelope()), token="t")

        with pytest.raises(exc_cls, match="closure drift injected for test"):
            run_one_candidate_pass(
                _mock_dispatch_backend(),
                adapter,
                campaign_id="camp-1",
                campaign_mode="exploration",
                candidate_static_fn=_passing_candidate_static_fn,
                stats_battery_kwargs={},
                output_paths=["artifact.json"],
                **_new_step_kwargs(guarded_evaluate_fn=failing_guarded_evaluate_fn),
            )


class TestCompileTimeExcludedFromRatchetComparison:
    def test_compile_time_seconds_absent_and_measure_called_with_handoff_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """VERIFY item 6: compile_time_seconds is absent from the dict passed to
        compute_ratchet_decision; measure_two_phase_timing is called with handoff.path (never a
        separately constructed path)."""
        ratchet_calls: list[tuple[dict[str, float], dict[str, float], dict[str, bool]]] = []

        def spy_compute_ratchet_decision(
            candidate_fitness: Mapping[str, float],
            best_fitness: Mapping[str, float],
            *,
            higher_is_better: Mapping[str, bool],
        ) -> RatchetDecision:
            ratchet_calls.append(
                (dict(candidate_fitness), dict(best_fitness), dict(higher_is_better))
            )
            return _SENTINEL_RATCHET_DECISION

        monkeypatch.setattr(
            main_loop_module, "compute_ratchet_decision", spy_compute_ratchet_decision
        )

        timing_calls: list[tuple[Any, str, list[Any]]] = []

        def spy_measure_two_phase_timing_fn(
            candidate_path: Path, callable_name: str, *, concrete_inputs: list[Any]
        ) -> TwoPhaseTiming:
            timing_calls.append((candidate_path, callable_name, concrete_inputs))
            return TwoPhaseTiming(compile_time_seconds=99.0, runtime_seconds=1.0, result=None)

        adapter = BathosCampaignAdapter(transport=_RecordingTransport(_run_envelope()), token="t")

        result = run_one_candidate_pass(
            _mock_dispatch_backend(),
            adapter,
            campaign_id="camp-1",
            campaign_mode="exploration",
            candidate_static_fn=_passing_candidate_static_fn,
            stats_battery_kwargs={},
            stats_battery_fn=lambda **kw: _passing_stats_verdict(),
            seed_trial_counts_fn=lambda db, script_sha256, hypothesis_clause_id="": (
                _passing_seed_counts()
            ),
            output_paths=["artifact.json"],
            **_new_step_kwargs(
                best_fitness=_BEST_FITNESS,
                higher_is_better=_HIGHER_IS_BETTER,
                measure_two_phase_timing_fn=spy_measure_two_phase_timing_fn,
                callable_name="candidate_fn",
                concrete_inputs=[1, 2, 3],
            ),
        )

        assert len(ratchet_calls) == 1
        candidate_fitness, _, _ = ratchet_calls[0]
        assert "compile_time_seconds" not in candidate_fitness
        assert candidate_fitness == _PASSING_FITNESS

        assert len(timing_calls) == 1
        candidate_path, callable_name, concrete_inputs = timing_calls[0]
        assert candidate_path == result.handoff.path
        assert callable_name == "candidate_fn"
        assert concrete_inputs == [1, 2, 3]


class TestGatesClearButRatchetRejects:
    def test_gates_clear_but_ratchet_not_improved_must_not_read_accepted_true(self) -> None:
        """VERIFY item 7: a candidate that clears stats_battery/seed_trial hard-blocked gates
        but whose ratchet_decision.improved is False must NOT have accepted=True."""

        def worse_guarded_evaluate_fn(*args: Any, **kwargs: Any) -> dict[str, float]:
            return dict(_WORSE_FITNESS)

        adapter = BathosCampaignAdapter(transport=_RecordingTransport(_run_envelope()), token="t")

        result = run_one_candidate_pass(
            _mock_dispatch_backend(),
            adapter,
            campaign_id="camp-1",
            campaign_mode="confirmation",
            candidate_static_fn=_passing_candidate_static_fn,
            stats_battery_kwargs={},
            stats_battery_fn=lambda **kw: _passing_stats_verdict(),
            seed_trial_counts_fn=lambda db, script_sha256, hypothesis_clause_id="": (
                _passing_seed_counts()
            ),
            output_paths=["artifact.json"],
            **_new_step_kwargs(
                guarded_evaluate_fn=worse_guarded_evaluate_fn,
                best_fitness=_BEST_FITNESS,
                higher_is_better=_HIGHER_IS_BETTER,
            ),
        )

        assert result.gate_outcome.hard_blocked is False, (
            "gates must genuinely clear for this test to prove ratchet is independently "
            "load-bearing, not just redundant with the existing gate checks"
        )
        assert result.ratchet_decision.improved is False
        assert result.accepted is False


# ---------------------------------------------------------------------------
# Backlog #4203 (Track A, P1.1-P1.7): fitness_dict field, commit_tree_sha_fn tree-substitution
# seam, candidate_target_path fail-fast validation, PriorBestSoFarLineageConflictError
# lineage-conflict guard, and the P1.6 lazy-invocation lock-in test.
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
    commit/tree exists for compute_candidate_tree_sha's own tests."""
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


def _ls_tree_entries(repo: Path, tree_sha: str) -> dict[str, str]:
    """path -> mode, for every entry in `git ls-tree -r <tree_sha>`."""
    entries: dict[str, str] = {}
    for line in _git(repo, "ls-tree", "-r", tree_sha).stdout.strip().splitlines():
        meta, path = line.split("\t")
        mode = meta.split()[0]
        entries[path] = mode
    return entries


class TestCandidateTargetPathValidation:
    """C7 (backlog #4203): candidate_target_path is optional, but required (fail-fast) when
    commit_tree_sha is not supplied directly."""

    def test_literal_commit_tree_sha_with_target_path_omitted_succeeds_fn_never_invoked(
        self,
    ) -> None:
        spy_calls: list[Any] = []

        def spy_commit_tree_sha_fn(
            handoff: CandidateHandoff, repo: Path, candidate_target_path: Path, base_sha: str | None
        ) -> str:
            spy_calls.append((handoff, repo, candidate_target_path, base_sha))
            return "should-not-be-reached"

        adapter = BathosCampaignAdapter(transport=_RecordingTransport(_run_envelope()), token="t")

        result = run_one_candidate_pass(
            _mock_dispatch_backend(),
            adapter,
            campaign_id="camp-1",
            campaign_mode="exploration",
            candidate_static_fn=_passing_candidate_static_fn,
            stats_battery_kwargs={},
            stats_battery_fn=lambda **kw: _passing_stats_verdict(),
            seed_trial_counts_fn=lambda db, script_sha256, hypothesis_clause_id="": (
                _passing_seed_counts()
            ),
            output_paths=["artifact.json"],
            # commit_tree_sha comes from _new_step_kwargs()'s literal 'unused-tree-sha' default;
            # candidate_target_path is deliberately left unset (omitted).
            **_new_step_kwargs(commit_tree_sha_fn=spy_commit_tree_sha_fn),
        )

        assert result.accepted is True
        assert spy_calls == [], (
            "commit_tree_sha_fn must never be invoked when commit_tree_sha is supplied literally"
        )

    def test_omitting_both_raises_value_error_before_any_dispatch_call(self) -> None:
        dispatch_calls: list[Any] = []

        class _TrackingDispatch:
            def dispatch_candidate(self) -> CandidateHandoff:
                dispatch_calls.append(1)
                return _mock_dispatch_backend().dispatch_candidate()

        adapter = BathosCampaignAdapter(transport=_RecordingTransport(_run_envelope()), token="t")

        with pytest.raises(ValueError, match="candidate_target_path is required"):
            run_one_candidate_pass(
                _TrackingDispatch(),
                adapter,
                campaign_id="camp-1",
                campaign_mode="exploration",
                candidate_static_fn=_passing_candidate_static_fn,
                stats_battery_kwargs={},
                output_paths=["artifact.json"],
                **_new_step_kwargs(commit_tree_sha=None),
            )

        assert dispatch_calls == [], (
            "the candidate_target_path validation must fire before dispatch_candidate() is "
            "ever called"
        )


class TestComputeCandidateTreeShaSignature:
    """C4 (backlog #4203): exactly 4 named params, in order."""

    def test_signature_has_exactly_four_named_params_in_order(self) -> None:
        sig = inspect.signature(compute_candidate_tree_sha)
        assert list(sig.parameters) == ["handoff", "repo", "candidate_target_path", "base_sha"]


class TestComputeCandidateTreeShaPlumbing:
    """C5/C9 (backlog #4203): whole-repo-tree substitution via a scratch GIT_INDEX_FILE."""

    def test_base_sha_none_raises_value_error_before_any_git_write(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        git_calls: list[Any] = []
        original_run_git = main_loop_module._run_git

        def spy_run_git(*args: Any, **kwargs: Any) -> Any:
            git_calls.append(args)
            return original_run_git(*args, **kwargs)

        monkeypatch.setattr(main_loop_module, "_run_git", spy_run_git)

        handoff = CandidateHandoff(path=git_repo / "existing.txt", content_sha256=_VALID_SHA256)

        with pytest.raises(ValueError, match="base_sha is None"):
            compute_candidate_tree_sha(handoff, git_repo, Path("new/candidate.py"), None)

        assert git_calls == [], "no git subcommand should run when base_sha is None"

    def test_new_path_resolves_tree_before_read_tree_writes_blob_before_update_index_mode_100644(
        self, git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """C5: rev-parse (base_sha^{tree}) before read-tree; hash-object -w before
        update-index --add --cacheinfo; resulting tree entry mode is 100644.
        C9: candidate_target_path is a genuinely NEW path (absent from the base tree) -- the
        resulting tree contains it alongside every pre-existing base-tree path unchanged."""
        base_sha = _head_sha(git_repo)
        candidate_file = tmp_path / "candidate_source.py"
        candidate_file.write_text("candidate content\n", encoding="utf-8")
        handoff = CandidateHandoff(path=candidate_file, content_sha256=_VALID_SHA256)

        subcommand_order: list[str] = []
        original_run_git = main_loop_module._run_git

        def spy_run_git(repo: Path, *args: str, **kwargs: Any) -> Any:
            subcommand_order.append(args[0])
            return original_run_git(repo, *args, **kwargs)

        monkeypatch.setattr(main_loop_module, "_run_git", spy_run_git)

        result_tree_sha = compute_candidate_tree_sha(
            handoff, git_repo, Path("new/candidate.py"), base_sha
        )

        assert subcommand_order == [
            "rev-parse",
            "read-tree",
            "hash-object",
            "update-index",
            "write-tree",
        ], (
            "base_sha^{tree} must be resolved before read-tree, and the blob must be written "
            "before it is staged via update-index"
        )

        entries = _ls_tree_entries(git_repo, result_tree_sha)
        assert entries["new/candidate.py"] == "100644"
        assert entries["existing.txt"] == "100644", (
            "every pre-existing base-tree path must remain unchanged"
        )
        assert entries != {"new/candidate.py": "100644"}, (
            "must never be a synthetic single-file tree"
        )

    def test_replacing_an_existing_path_updates_its_content_only(
        self, git_repo: Path, tmp_path: Path
    ) -> None:
        """git update-index --add does not distinguish new-path from replace-existing-path at
        the plumbing level (C9) -- confirm the replacement case also works correctly."""
        base_sha = _head_sha(git_repo)
        candidate_file = tmp_path / "candidate_source.py"
        candidate_file.write_text("replacement content\n", encoding="utf-8")
        handoff = CandidateHandoff(path=candidate_file, content_sha256=_VALID_SHA256)

        result_tree_sha = compute_candidate_tree_sha(
            handoff, git_repo, Path("existing.txt"), base_sha
        )

        entries = _ls_tree_entries(git_repo, result_tree_sha)
        assert entries == {"existing.txt": "100644"}
        blob_content = _git(git_repo, "cat-file", "-p", f"{result_tree_sha}:existing.txt").stdout
        assert blob_content == "replacement content\n"

    def test_real_index_never_mutated(self, git_repo: Path, tmp_path: Path) -> None:
        """The repo's real .git/index must never be read or written by
        compute_candidate_tree_sha."""
        real_index_path = git_repo / ".git" / "index"
        before = real_index_path.read_bytes() if real_index_path.exists() else None

        base_sha = _head_sha(git_repo)
        candidate_file = tmp_path / "candidate_source.py"
        candidate_file.write_text("candidate content\n", encoding="utf-8")
        handoff = CandidateHandoff(path=candidate_file, content_sha256=_VALID_SHA256)

        compute_candidate_tree_sha(handoff, git_repo, Path("new/candidate.py"), base_sha)

        after = real_index_path.read_bytes() if real_index_path.exists() else None
        assert before == after


class TestPriorBestSoFarLineageConflictGuard:
    """C6 (backlog #4203): a fresh-start call finding a prior best-so-far commit raises, unless
    explicitly overridden."""

    def test_raises_when_best_fitness_none_and_prior_lineage_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Per the TOML fixer_prompt / loop_sprint_plan.json's own explicit file:line anchor
        (P1.5: 'inserted in the best_fitness-is-None branch, currently lines 469-479'), this
        guard fires deep inside run_one_candidate_pass, AFTER dispatch/the bathos run/scoring
        have already executed -- not before them. (One spec doc's AC bullet [C6] independently
        says 'before any dispatch/bathos/git call', which is unsatisfiable simultaneously with
        the TOML/plan's own concrete placement instruction without restructuring the function;
        the TOML/plan's unambiguous line-anchored instruction is followed here, per this task's
        own stated precedence for resolving such conflicts.) What IS proven not to happen once
        the conflict is detected: neither of the accept-branch's own crash-atomicity git calls
        (create_pending_commit/advance_best_so_far) ever fires.
        """
        monkeypatch.setattr(
            main_loop_module, "read_best_so_far", lambda repo, ref_name: "conflicting-sha-123"
        )
        create_calls: list[Any] = []
        advance_calls: list[Any] = []
        monkeypatch.setattr(
            main_loop_module, "create_pending_commit", lambda *a, **kw: create_calls.append(1)
        )
        monkeypatch.setattr(
            main_loop_module, "advance_best_so_far", lambda *a, **kw: advance_calls.append(1)
        )
        transport = _RecordingTransport(_run_envelope())
        adapter = BathosCampaignAdapter(transport=transport, token="t")

        with pytest.raises(PriorBestSoFarLineageConflictError) as exc_info:
            run_one_candidate_pass(
                _mock_dispatch_backend(),
                adapter,
                campaign_id="camp-1",
                campaign_mode="exploration",
                candidate_static_fn=_passing_candidate_static_fn,
                stats_battery_kwargs={},
                output_paths=["artifact.json"],
                **_new_step_kwargs(best_fitness=None, higher_is_better=None),
            )

        msg = str(exc_info.value)
        assert "refs/xtrax/best-so-far" in msg
        assert "conflicting-sha-123" in msg
        assert "fresh" in msg.lower(), "message must steer the caller toward a fresh ref name"
        assert create_calls == [], (
            "create_pending_commit must never fire once the lineage conflict is detected"
        )
        assert advance_calls == [], (
            "advance_best_so_far must never fire once the lineage conflict is detected"
        )

    def test_allow_fresh_start_override_proceeds_with_automatic_accept_sentinel_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            main_loop_module, "read_best_so_far", lambda repo, ref_name: "conflicting-sha-123"
        )
        create_calls: list[Any] = []
        advance_calls: list[Any] = []

        def spy_create(repo: Any, tree_sha: Any, parent_sha: Any, message: Any) -> str:
            create_calls.append((repo, tree_sha, parent_sha, message))
            return "pending-sha"

        def spy_advance(repo: Any, ref_name: Any, new_sha: Any, expected_old_sha: Any) -> None:
            advance_calls.append((repo, ref_name, new_sha, expected_old_sha))

        monkeypatch.setattr(main_loop_module, "create_pending_commit", spy_create)
        monkeypatch.setattr(main_loop_module, "advance_best_so_far", spy_advance)

        adapter = BathosCampaignAdapter(transport=_RecordingTransport(_run_envelope()), token="t")

        result = run_one_candidate_pass(
            _mock_dispatch_backend(),
            adapter,
            campaign_id="camp-1",
            campaign_mode="exploration",
            candidate_static_fn=_passing_candidate_static_fn,
            stats_battery_kwargs={},
            stats_battery_fn=lambda **kw: _passing_stats_verdict(),
            seed_trial_counts_fn=lambda db, script_sha256, hypothesis_clause_id="": (
                _passing_seed_counts()
            ),
            output_paths=["artifact.json"],
            **_new_step_kwargs(
                best_fitness=None,
                higher_is_better=None,
                allow_fresh_start_despite_existing_lineage=True,
            ),
        )

        assert result.accepted is True
        assert result.ratchet_decision.improved is True
        assert len(create_calls) == 1
        assert len(advance_calls) == 1


class TestFitnessDictRoundTrip:
    """Base-spec AC bullet 1 (backlog #4203): fitness_dict round-trips byte-identically from
    guarded_evaluate_fn, and is never observed in a half-populated state on exception."""

    def test_fitness_dict_round_trips_byte_identical_from_guarded_evaluate_fn(self) -> None:
        adapter = BathosCampaignAdapter(transport=_RecordingTransport(_run_envelope()), token="t")

        result = run_one_candidate_pass(
            _mock_dispatch_backend(),
            adapter,
            campaign_id="camp-1",
            campaign_mode="exploration",
            candidate_static_fn=_passing_candidate_static_fn,
            stats_battery_kwargs={},
            stats_battery_fn=lambda **kw: _passing_stats_verdict(),
            seed_trial_counts_fn=lambda db, script_sha256, hypothesis_clause_id="": (
                _passing_seed_counts()
            ),
            output_paths=["artifact.json"],
            **_new_step_kwargs(),
        )

        assert result.fitness_dict == _PASSING_FITNESS

    def test_no_partial_result_constructed_when_guarded_evaluate_fn_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        construction_calls: list[Any] = []
        original_init = OneCandidatePassResult.__init__

        def spy_init(self: Any, *args: Any, **kwargs: Any) -> None:
            construction_calls.append((args, kwargs))
            original_init(self, *args, **kwargs)

        monkeypatch.setattr(OneCandidatePassResult, "__init__", spy_init)

        def failing_guarded_evaluate_fn(*args: Any, **kwargs: Any) -> dict[str, float]:
            msg = "boom mid-scoring"
            raise RuntimeError(msg)

        adapter = BathosCampaignAdapter(transport=_RecordingTransport(_run_envelope()), token="t")

        with pytest.raises(RuntimeError, match="boom mid-scoring"):
            run_one_candidate_pass(
                _mock_dispatch_backend(),
                adapter,
                campaign_id="camp-1",
                campaign_mode="exploration",
                candidate_static_fn=_passing_candidate_static_fn,
                stats_battery_kwargs={},
                output_paths=["artifact.json"],
                **_new_step_kwargs(guarded_evaluate_fn=failing_guarded_evaluate_fn),
            )

        assert construction_calls == [], (
            "no OneCandidatePassResult -- partial or otherwise -- may ever be constructed when "
            "guarded_evaluate_fn raises"
        )


class TestCommitTreeShaFnLazyInvocation:
    """P1.6 lock-in test (plan-audit-added): commit_tree_sha_fn must never be invoked for a
    candidate that is rejected, gate-blocked, or aborts via a guarded_evaluate_fn exception
    before ever reaching the accept branch."""

    def _spy_commit_tree_sha_fn(self, calls: list[Any]) -> Any:
        def _fn(
            handoff: CandidateHandoff, repo: Path, candidate_target_path: Path, base_sha: str | None
        ) -> str:
            calls.append((handoff, repo, candidate_target_path, base_sha))
            return "should-not-be-reached"

        return _fn

    def test_never_invoked_for_a_rejected_candidate(self) -> None:
        spy_calls: list[Any] = []

        def worse_guarded_evaluate_fn(*args: Any, **kwargs: Any) -> dict[str, float]:
            return dict(_WORSE_FITNESS)

        adapter = BathosCampaignAdapter(transport=_RecordingTransport(_run_envelope()), token="t")

        result = run_one_candidate_pass(
            _mock_dispatch_backend(),
            adapter,
            campaign_id="camp-1",
            campaign_mode="exploration",
            candidate_static_fn=_passing_candidate_static_fn,
            stats_battery_kwargs={},
            stats_battery_fn=lambda **kw: _passing_stats_verdict(),
            seed_trial_counts_fn=lambda db, script_sha256, hypothesis_clause_id="": (
                _passing_seed_counts()
            ),
            output_paths=["artifact.json"],
            **_new_step_kwargs(
                guarded_evaluate_fn=worse_guarded_evaluate_fn,
                best_fitness=_BEST_FITNESS,
                higher_is_better=_HIGHER_IS_BETTER,
                commit_tree_sha=None,
                candidate_target_path=Path("target.py"),
                commit_tree_sha_fn=self._spy_commit_tree_sha_fn(spy_calls),
            ),
        )

        assert result.accepted is False
        assert result.ratchet_decision.improved is False
        assert spy_calls == []

    def test_never_invoked_for_a_gate_blocked_and_ratchet_rejected_candidate(self) -> None:
        """Under the current (out-of-scope-to-restructure) control flow, the accept/reject
        crash-atomicity branch is driven solely by ratchet_decision.improved -- gate checks run
        strictly AFTER it (module docstring, step 3 vs step 2.7). A candidate that is
        gate-blocked but still ratchet-improved would therefore still reach the accept branch;
        this test combines gate-hard-block with a genuinely non-improved ratchet decision, the
        realistic "doubly rejected" case, to prove commit_tree_sha_fn is skipped."""
        spy_calls: list[Any] = []

        def worse_guarded_evaluate_fn(*args: Any, **kwargs: Any) -> dict[str, float]:
            return dict(_WORSE_FITNESS)

        adapter = BathosCampaignAdapter(transport=_RecordingTransport(_run_envelope()), token="t")

        result = run_one_candidate_pass(
            _mock_dispatch_backend(),
            adapter,
            campaign_id="camp-1",
            campaign_mode="confirmation",
            candidate_static_fn=_passing_candidate_static_fn,
            stats_battery_kwargs={},
            stats_battery_fn=lambda **kw: _downgraded_stats_verdict(),
            seed_trial_counts_fn=lambda db, script_sha256, hypothesis_clause_id="": (
                _passing_seed_counts()
            ),
            output_paths=["artifact.json"],
            **_new_step_kwargs(
                guarded_evaluate_fn=worse_guarded_evaluate_fn,
                best_fitness=_BEST_FITNESS,
                higher_is_better=_HIGHER_IS_BETTER,
                commit_tree_sha=None,
                candidate_target_path=Path("target.py"),
                commit_tree_sha_fn=self._spy_commit_tree_sha_fn(spy_calls),
            ),
        )

        assert result.gate_outcome.hard_blocked is True
        assert result.accepted is False
        assert spy_calls == []

    def test_never_invoked_when_guarded_evaluate_fn_raises_before_ratchet_decision(self) -> None:
        spy_calls: list[Any] = []

        def failing_guarded_evaluate_fn(*args: Any, **kwargs: Any) -> dict[str, float]:
            msg = "closure drift injected for test"
            raise ClosureHashMismatchError(msg)

        adapter = BathosCampaignAdapter(transport=_RecordingTransport(_run_envelope()), token="t")

        with pytest.raises(ClosureHashMismatchError):
            run_one_candidate_pass(
                _mock_dispatch_backend(),
                adapter,
                campaign_id="camp-1",
                campaign_mode="exploration",
                candidate_static_fn=_passing_candidate_static_fn,
                stats_battery_kwargs={},
                output_paths=["artifact.json"],
                **_new_step_kwargs(
                    guarded_evaluate_fn=failing_guarded_evaluate_fn,
                    commit_tree_sha=None,
                    candidate_target_path=Path("target.py"),
                    commit_tree_sha_fn=self._spy_commit_tree_sha_fn(spy_calls),
                ),
            )

        assert spy_calls == []
