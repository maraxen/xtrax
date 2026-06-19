"""Tests for N3.2 incremental ruff-select enablement schedule (#1594)."""

from __future__ import annotations

from pathlib import Path

import pytest

from xtrax.devtools.ruff_schedule import (
    DEFAULT_PYPROJECT_PATH,
    DEFAULT_SCHEDULE_PATH,
    load_ruff_schedule,
    next_pending_wave,
    read_pyproject_select,
    validate_ruff_schedule_sync,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEDULE_PATH = ROOT / DEFAULT_SCHEDULE_PATH
PYPROJECT_PATH = ROOT / DEFAULT_PYPROJECT_PATH


def test_load_ruff_schedule_loads_committed_toml() -> None:
    schedule = load_ruff_schedule(SCHEDULE_PATH)
    wave_ids = {wave.id for wave in schedule.waves}
    assert schedule.active_wave_id == "wave_baseline"
    assert "wave_baseline" in wave_ids
    assert "wave_doc" in wave_ids
    assert "wave_complexity" in wave_ids
    assert "wave_bugbear" in wave_ids
    assert len(schedule.waves) >= 4


def test_active_wave_matches_pyproject_select() -> None:
    schedule = load_ruff_schedule(SCHEDULE_PATH)
    pyproject_select = read_pyproject_select(PYPROJECT_PATH)
    active = next(wave for wave in schedule.waves if wave.id == schedule.active_wave_id)
    assert active.status == "active"
    assert active.select == pyproject_select
    validate_ruff_schedule_sync(
        schedule_path=SCHEDULE_PATH,
        pyproject_path=PYPROJECT_PATH,
    )


def test_next_pending_wave_returns_first_pending() -> None:
    pending = next_pending_wave(SCHEDULE_PATH)
    assert pending is not None
    assert pending.id == "wave_doc"
    assert pending.status == "pending"
    assert "D" in pending.select


def test_validate_ruff_schedule_sync_fails_on_drift(tmp_path: Path) -> None:
    drifted = tmp_path / "pyproject.toml"
    drifted.write_text(
        """
[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="active wave"):
        validate_ruff_schedule_sync(
            schedule_path=SCHEDULE_PATH,
            pyproject_path=drifted,
        )


def test_validate_ruff_schedule_sync_fails_on_schedule_drift(tmp_path: Path) -> None:
    broken = tmp_path / "ruff_enablement_schedule.toml"
    broken.write_text(
        """
[schedule]
schema_version = 1
version = "0.0.0"
active_wave_id = "wave_baseline"

[[waves]]
id = "wave_baseline"
status = "active"
select = ["E", "F"]
notes = "drifted"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="active wave"):
        validate_ruff_schedule_sync(
            schedule_path=broken,
            pyproject_path=PYPROJECT_PATH,
        )
