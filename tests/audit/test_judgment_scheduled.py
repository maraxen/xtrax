"""Tests for N5.2 judgment-track scheduled run + staleness (#1598)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from xtrax.devtools.baseline import load_baseline
from xtrax.devtools.judgment_scheduled import (
    DEFAULT_SCHEDULE_PATH,
    STALENESS_METRIC,
    JudgmentRunState,
    load_judgment_run_state,
    load_judgment_schedule,
    run_scheduled_judgment,
    save_judgment_run_state,
    staleness_days_since,
    update_staleness_baseline,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEDULE_PATH = ROOT / DEFAULT_SCHEDULE_PATH


def test_load_judgment_schedule_loads_committed_toml() -> None:
    schedule = load_judgment_schedule(SCHEDULE_PATH)
    assert schedule.version == "0.1.0"
    assert schedule.max_staleness_days == 21
    assert schedule.cron_hint == "0 6 * * 1"
    assert schedule.components == (
        "judgment_dispatch_emit",
        "refute_promote_self_test",
        "docs_judgment_self_test",
    )


def test_judgment_run_state_round_trip(tmp_path: Path) -> None:
    state_path = tmp_path / "judgment_last_run.toml"
    state = JudgmentRunState(
        last_run_at="2026-06-19T12:00:00+00:00",
        run_id="scheduled-test-run",
    )
    save_judgment_run_state(state, path=state_path)
    restored = load_judgment_run_state(state_path)
    assert restored == state


def test_staleness_days_since_computes_fractional_days() -> None:
    now = datetime(2026, 6, 21, 0, 0, 0, tzinfo=UTC)
    days = staleness_days_since("2026-06-19T00:00:00+00:00", now=now)
    assert days == pytest.approx(2.0)


@patch("xtrax.devtools.judgment_scheduled.run_docs_judgment_self_test")
@patch("xtrax.devtools.judgment_scheduled.run_refute_promote_self_test")
@patch("xtrax.devtools.judgment_scheduled.run_judgment_dispatch")
def test_run_scheduled_judgment_no_emit_records_state(
    mock_dispatch: object,
    mock_refute: object,
    mock_docs: object,
    tmp_path: Path,
) -> None:
    audits_path = tmp_path / "audits.jsonl"
    state_path = tmp_path / "judgment_last_run.toml"
    baseline_path = tmp_path / "audit_baseline.json"
    baseline_path.write_text(
        """{
  "schema_version": 1,
  "updated_at": "2026-06-19T00:00:00+00:00",
  "metrics": {
    "judgment.staleness_days": {
      "key": "judgment.staleness_days",
      "value": 0.0,
      "comparator": "minimize"
    }
  }
}
""",
        encoding="utf-8",
    )

    result = run_scheduled_judgment(
        audits_path,
        emit_dispatch=False,
        root=ROOT,
        schedule_path=SCHEDULE_PATH,
        state_path=state_path,
        baseline_path=baseline_path,
        dispatch_path=ROOT / "audit" / "judgment_dispatch.toml",
        rubrics_dir=ROOT / "audit" / "rubrics",
        routing_path=ROOT / "audit" / "routing.toml",
    )

    assert result.passed is True
    assert result.dispatch_emitted is False
    assert result.refute_promote_ok is True
    assert result.docs_judgment_ok is True
    assert result.staleness_passed is True
    assert result.staleness_days == pytest.approx(0.0, abs=0.01)
    mock_dispatch.assert_not_called()
    mock_refute.assert_called_once_with(root=ROOT)
    mock_docs.assert_called_once_with(root=ROOT)

    state = load_judgment_run_state(state_path)
    assert state is not None
    assert state.run_id == result.run_id

    baseline = load_baseline(baseline_path)
    assert baseline.metrics[STALENESS_METRIC].value == pytest.approx(
        0.0, abs=0.01
    )


def test_update_staleness_baseline_from_state(tmp_path: Path) -> None:
    state_path = tmp_path / "judgment_last_run.toml"
    baseline_path = tmp_path / "audit_baseline.json"
    baseline_path.write_text(
        """{
  "schema_version": 1,
  "updated_at": "2026-06-19T00:00:00+00:00",
  "metrics": {
    "judgment.staleness_days": {
      "key": "judgment.staleness_days",
      "value": 0.0,
      "comparator": "minimize"
    }
  }
}
""",
        encoding="utf-8",
    )
    save_judgment_run_state(
        JudgmentRunState(
            last_run_at="2026-06-10T00:00:00+00:00",
            run_id="stale-run",
        ),
        path=state_path,
    )

    with patch(
        "xtrax.devtools.judgment_scheduled.datetime"
    ) as mock_datetime:
        mock_datetime.now.return_value = datetime(
            2026, 6, 19, 0, 0, 0, tzinfo=UTC
        )
        mock_datetime.fromisoformat = datetime.fromisoformat
        observed = update_staleness_baseline(
            baseline_path,
            state_path=state_path,
            schedule_path=SCHEDULE_PATH,
        )

    assert observed == pytest.approx(9.0)
    baseline = load_baseline(baseline_path)
    assert baseline.metrics[STALENESS_METRIC].value == pytest.approx(0.0)


def test_update_staleness_baseline_without_state_uses_max(tmp_path: Path) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    baseline_path.write_text(
        """{
  "schema_version": 1,
  "updated_at": "2026-06-19T00:00:00+00:00",
  "metrics": {}
}
""",
        encoding="utf-8",
    )

    observed = update_staleness_baseline(
        baseline_path,
        state_path=tmp_path / "missing.toml",
        schedule_path=SCHEDULE_PATH,
    )

    assert observed == 21.0
    baseline = load_baseline(baseline_path)
    assert baseline.metrics[STALENESS_METRIC].value == 21.0
