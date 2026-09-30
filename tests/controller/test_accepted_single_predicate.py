"""#4584 part 3 + gate move: one acceptance predicate drives the property, the best-so-far
lineage advance, and the probe record; the stats/seed gates run before the lineage step, so
a hard-blocked candidate can no longer become best-so-far.

This is the sign-off PR's behaviour: before it, a hard-blocked but improved candidate
advanced the best-so-far ref (the lineage read only `ratchet_decision.improved`), and the
probe record reported an advisory-only downgrade as not accepted.
"""

from pathlib import Path
from typing import Any

import pytest

import controller.main_loop as ml
from controller.bathos_campaign_adapter import BathosCampaignAdapter
from controller.main_loop import is_accepted, run_one_candidate_pass

# test_main_loop's autouse fixtures (git crash-atomicity stubs, metrics-provenance isolation
# with a non-empty run_id), registered here by import.
from tests.controller.test_main_loop import (  # noqa: F401
    _BEST_FITNESS,
    _HIGHER_IS_BETTER,
    _SENTINEL_RATCHET_DECISION,
    _downgraded_stats_verdict,
    _failing_seed_counts,
    _mock_dispatch_backend,
    _new_step_kwargs,
    _passing_candidate_static_fn,
    _passing_seed_counts,
    _passing_stats_verdict,
    _RecordingTransport,
    _run_envelope,
    _stub_crash_atomicity,
    _stub_metrics_provenance,
)


@pytest.mark.parametrize(
    ("run_success", "hard_blocked", "improved", "expected"),
    [
        (True, False, True, True),
        (False, False, True, False),
        (True, True, True, False),
        (True, False, False, False),
    ],
)
def test_the_predicate(run_success: bool, hard_blocked: bool, improved: bool, expected: bool):
    assert (
        is_accepted(run_success=run_success, hard_blocked=hard_blocked, improved=improved)
        is expected
    )


def _spy_lineage(
    monkeypatch: pytest.MonkeyPatch, *, prior: str | None = "prior-best-sha"
) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(ml, "create_pending_commit", lambda *a, **k: calls.append("commit") or "p")
    monkeypatch.setattr(ml, "advance_best_so_far", lambda *a, **k: calls.append("advance"))
    monkeypatch.setattr(ml, "reset_worktree_to_best_so_far", lambda *a, **k: calls.append("reset"))
    monkeypatch.setattr(ml, "read_best_so_far", lambda *a, **k: prior)
    return calls


def _pass(*, mode: str, stats: Any, seeds: Any, **step: Any):  # noqa: ANN202
    adapter = BathosCampaignAdapter(transport=_RecordingTransport(_run_envelope()), token="t")
    return run_one_candidate_pass(
        _mock_dispatch_backend(),
        adapter,
        campaign_id="camp-accept",
        campaign_mode=mode,
        candidate_static_fn=_passing_candidate_static_fn,
        stats_battery_kwargs={},
        stats_battery_fn=lambda **kw: stats,
        seed_trial_counts_fn=lambda db, sha, hypothesis_clause_id="": seeds,
        output_paths=["artifact.json"],
        **_new_step_kwargs(**step),
    )


def test_hard_blocked_but_improved_candidate_never_becomes_best_so_far(monkeypatch):
    """The headline defect: the lineage used to read only `improved`."""
    calls = _spy_lineage(monkeypatch)
    monkeypatch.setattr(ml, "compute_ratchet_decision", lambda *a, **k: _SENTINEL_RATCHET_DECISION)
    result = _pass(
        mode="confirmation",
        stats=_downgraded_stats_verdict(),
        seeds=_passing_seed_counts(),
        best_fitness=_BEST_FITNESS,
        higher_is_better=_HIGHER_IS_BETTER,
    )
    assert result.ratchet_decision.improved is True
    assert result.gate_outcome.hard_blocked is True
    assert result.accepted is False
    assert calls == ["reset"]  # rejected: reset to the prior best, no commit, no advance


def test_control_clean_gates_and_improved_candidate_advances(monkeypatch):
    """Positive control for the test above: same setup, gates clear -> it DOES advance."""
    calls = _spy_lineage(monkeypatch)
    monkeypatch.setattr(ml, "compute_ratchet_decision", lambda *a, **k: _SENTINEL_RATCHET_DECISION)
    result = _pass(
        mode="confirmation",
        stats=_passing_stats_verdict(),
        seeds=_passing_seed_counts(),
        best_fitness=_BEST_FITNESS,
        higher_is_better=_HIGHER_IS_BETTER,
    )
    assert result.accepted is True
    assert calls == ["commit", "advance"]


def test_hard_blocked_first_candidate_writes_and_resets_nothing(monkeypatch):
    """No best-so-far exists yet: nothing to reset to, and no ref is written."""
    calls = _spy_lineage(monkeypatch, prior=None)
    result = _pass(mode="sequential", stats=_passing_stats_verdict(), seeds=_failing_seed_counts())
    assert result.gate_outcome.hard_blocked is True
    assert result.accepted is False
    assert calls == []


def test_crash_resume_hard_block_resets_to_the_existing_ref(monkeypatch):
    """A real ref with best_fitness still None (crash-resume) is reset on reject: the
    condition is "a ref exists", not "best_fitness is set"."""
    calls = _spy_lineage(monkeypatch, prior="resumed-best-sha")
    result = _pass(
        mode="sequential",
        stats=_passing_stats_verdict(),
        seeds=_failing_seed_counts(),
        allow_fresh_start_despite_existing_lineage=True,
    )
    assert result.accepted is False
    assert calls == ["reset"]


def test_gates_run_before_the_lineage_step(monkeypatch):
    order: list[str] = []
    monkeypatch.setattr(ml, "create_pending_commit", lambda *a, **k: order.append("commit") or "p")
    monkeypatch.setattr(ml, "advance_best_so_far", lambda *a, **k: order.append("advance"))

    def stats_fn(**kw: Any):  # noqa: ANN202
        order.append("stats")
        return _passing_stats_verdict()

    adapter = BathosCampaignAdapter(transport=_RecordingTransport(_run_envelope()), token="t")
    run_one_candidate_pass(
        _mock_dispatch_backend(),
        adapter,
        campaign_id="camp-order",
        campaign_mode="exploration",
        candidate_static_fn=_passing_candidate_static_fn,
        stats_battery_kwargs={},
        stats_battery_fn=stats_fn,
        seed_trial_counts_fn=lambda db, sha, hypothesis_clause_id="": _passing_seed_counts(),
        output_paths=["artifact.json"],
        **_new_step_kwargs(),
    )
    assert order == ["stats", "commit", "advance"]


def test_probe_record_reports_the_same_acceptance_as_the_result(tmp_path: Path):
    """An advisory-only downgrade (exploration) is accepted by the property; the probe
    record used to say `accepted=false` for it (it read `honored and held`)."""
    from xtrax.profiling.record import ProbeRecord

    records = tmp_path / "records"
    adapter = BathosCampaignAdapter(transport=_RecordingTransport(_run_envelope()), token="t")
    result = run_one_candidate_pass(
        _mock_dispatch_backend(),
        adapter,
        campaign_id="camp-advisory",
        campaign_mode="exploration",
        candidate_static_fn=_passing_candidate_static_fn,
        stats_battery_kwargs={},
        stats_battery_fn=lambda **kw: _downgraded_stats_verdict(),
        seed_trial_counts_fn=lambda db, sha, hypothesis_clause_id="": _passing_seed_counts(),
        output_paths=["artifact.json"],
        **_new_step_kwargs(),
        probe_record_dir=records,
    )
    assert result.gate_outcome.stats_battery.advisory is True
    assert result.accepted is True
    (written,) = sorted(records.glob("pass_*.json"))
    assert ProbeRecord.read(written).config["accepted"] == "true"
