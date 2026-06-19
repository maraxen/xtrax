"""N3.2 incremental ruff-select enablement schedule (#1594)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SCHEDULE_PATH = Path("audit/ruff_enablement_schedule.toml")
DEFAULT_PYPROJECT_PATH = Path("pyproject.toml")


@dataclass(frozen=True, slots=True)
class RuffWave:
    id: str
    status: str
    select: frozenset[str]
    notes: str
    ties_to_dimension: str | None = None


@dataclass(frozen=True, slots=True)
class RuffSchedule:
    version: str
    active_wave_id: str
    waves: tuple[RuffWave, ...]


def load_ruff_schedule(path: Path = DEFAULT_SCHEDULE_PATH) -> RuffSchedule:
    """Load phased ruff rule rollout from ruff_enablement_schedule.toml."""
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    schedule = payload.get("schedule")
    if not isinstance(schedule, dict):
        msg = f"{path}: [schedule] section is required"
        raise ValueError(msg)
    raw_waves = payload.get("waves")
    if not isinstance(raw_waves, list):
        msg = f"{path}: [[waves]] list is required"
        raise ValueError(msg)
    waves: list[RuffWave] = []
    for item in raw_waves:
        if not isinstance(item, dict):
            msg = f"{path}: each [[waves]] entry must be a table"
            raise ValueError(msg)
        raw_select = item.get("select")
        if not isinstance(raw_select, list):
            msg = f"{path}: wave {item.get('id')!r} select must be a list"
            raise ValueError(msg)
        ties = item.get("ties_to_dimension")
        waves.append(
            RuffWave(
                id=str(item["id"]),
                status=str(item["status"]),
                select=frozenset(str(rule) for rule in raw_select),
                notes=str(item.get("notes", "")),
                ties_to_dimension=str(ties) if ties is not None else None,
            )
        )
    return RuffSchedule(
        version=str(schedule.get("version", "")),
        active_wave_id=str(schedule["active_wave_id"]),
        waves=tuple(waves),
    )


def read_pyproject_select(
    pyproject_path: Path = DEFAULT_PYPROJECT_PATH,
) -> frozenset[str]:
    """Read [tool.ruff.lint] select from pyproject.toml."""
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    lint = payload.get("tool", {}).get("ruff", {}).get("lint")
    if not isinstance(lint, dict):
        msg = f"{pyproject_path}: [tool.ruff.lint] section is required"
        raise ValueError(msg)
    raw_select = lint.get("select")
    if not isinstance(raw_select, list):
        msg = f"{pyproject_path}: [tool.ruff.lint] select must be a list"
        raise ValueError(msg)
    return frozenset(str(rule) for rule in raw_select)


def _active_wave(schedule: RuffSchedule) -> RuffWave:
    for wave in schedule.waves:
        if wave.id == schedule.active_wave_id:
            return wave
    msg = (
        f"active_wave_id={schedule.active_wave_id!r} "
        f"not found in [[waves]]"
    )
    raise ValueError(msg)


def validate_ruff_schedule_sync(
    schedule_path: Path = DEFAULT_SCHEDULE_PATH,
    pyproject_path: Path = DEFAULT_PYPROJECT_PATH,
) -> None:
    """Raise when active wave select mismatches pyproject [tool.ruff.lint] select."""
    schedule = load_ruff_schedule(schedule_path)
    active = _active_wave(schedule)
    pyproject_select = read_pyproject_select(pyproject_path)
    if active.select != pyproject_select:
        missing = sorted(active.select - pyproject_select)
        extra = sorted(pyproject_select - active.select)
        parts: list[str] = [
            f"active wave {active.id!r} select != pyproject select",
            f"  schedule: {sorted(active.select)}",
            f"  pyproject: {sorted(pyproject_select)}",
        ]
        if missing:
            parts.append(f"  missing from pyproject: {missing}")
        if extra:
            parts.append(f"  extra in pyproject: {extra}")
        raise ValueError("\n".join(parts))


def next_pending_wave(
    path: Path = DEFAULT_SCHEDULE_PATH,
) -> RuffWave | None:
    """Return the first pending wave in schedule order, or None."""
    schedule = load_ruff_schedule(path)
    for wave in schedule.waves:
        if wave.status == "pending":
            return wave
    return None


__all__ = [
    "DEFAULT_PYPROJECT_PATH",
    "DEFAULT_SCHEDULE_PATH",
    "RuffSchedule",
    "RuffWave",
    "load_ruff_schedule",
    "next_pending_wave",
    "read_pyproject_select",
    "validate_ruff_schedule_sync",
]
