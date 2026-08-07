"""Tests for controller.main_loop (LC-09, epic #3611, AC-8a).

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
"""

import hashlib
from pathlib import Path
from typing import Any

import pytest

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
from controller.lineage_interim import CandidateParentage, MultiParentLineageUnsupportedError
from controller.main_loop import GateOutcome, OneCandidatePassResult, run_one_candidate_pass
from xtrax.loop.candidate_static import CandidateStaticGateError
from xtrax.loop.seed_gate import SeedTrialCounts, SeedTrialFloorDecision
from xtrax.loop.stats_battery_gate import BathosStatsBatteryVerdict, ConcludeStatsDecision

_CANDIDATE_CONTENT = "candidate-source"
_VALID_SHA256 = hashlib.sha256(_CANDIDATE_CONTENT.encode("utf-8")).hexdigest()


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


class TestOneCandidatePassResultAccepted:
    def _result(self, *, run_success: bool, hard_blocked: bool) -> OneCandidatePassResult:
        return OneCandidatePassResult(
            handoff=CandidateHandoff(path=Path("c.py"), content_sha256=_VALID_SHA256),
            derived_from="",
            run_result=CandidateRunResult(script_path="c.py", exit_code=0, success=run_success),
            gate_outcome=GateOutcome(
                stats_battery=_stats_decision(hard_blocked=hard_blocked),
                seed_trial=_seed_decision(hard_blocked=False),
            ),
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
        )

        assert len(transport.calls) == 1
        assert result.accepted is True
