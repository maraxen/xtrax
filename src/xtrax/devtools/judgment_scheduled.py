"""N5.2 judgment-track scheduled run + staleness metric (#1598)."""

from __future__ import annotations

import subprocess
import tomllib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from xtrax.devtools.baseline import (
    DEFAULT_BASELINE_PATH,
    load_baseline,
    save_baseline,
    update_metric,
)
from xtrax.devtools.judgment import run_judgment_dispatch, validate_judgment_wiring

DEFAULT_SCHEDULE_PATH = Path("audit/judgment_schedule.toml")
DEFAULT_STATE_PATH = Path(".praxia/judgment_last_run.toml")
STALENESS_METRIC = "judgment.staleness_days"


@dataclass(frozen=True, slots=True)
class JudgmentSchedule:
    version: str
    max_staleness_days: int
    cron_hint: str
    components: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class JudgmentRunState:
    last_run_at: str
    run_id: str


@dataclass(frozen=True, slots=True)
class ScheduledJudgmentResult:
    passed: bool
    run_id: str
    staleness_days: float
    staleness_passed: bool
    dispatch_emitted: bool
    refute_promote_ok: bool
    docs_judgment_ok: bool


def _format_toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        text = f"{value:.6g}"
        if "." not in text:
            return f"{text}.0"
        return text
    if isinstance(value, int):
        return f"{value}.0"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    msg = f"unsupported TOML value type: {type(value).__name__}"
    raise TypeError(msg)


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def load_judgment_schedule(
    path: Path = DEFAULT_SCHEDULE_PATH,
) -> JudgmentSchedule:
    """Load scheduled judgment settings from judgment_schedule.toml."""
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    schedule = payload.get("schedule")
    if not isinstance(schedule, dict):
        msg = f"{path}: [schedule] section is required"
        raise ValueError(msg)
    raw_components = payload.get("components", [])
    if not isinstance(raw_components, list):
        msg = f"{path}: [[components]] must be a list"
        raise ValueError(msg)
    components: list[str] = []
    for item in raw_components:
        if not isinstance(item, dict):
            msg = f"{path}: each [[components]] entry must be a table"
            raise ValueError(msg)
        components.append(str(item["id"]))
    return JudgmentSchedule(
        version=str(schedule.get("version", "0.0.0")),
        max_staleness_days=int(schedule["max_staleness_days"]),
        cron_hint=str(schedule["cron_hint"]),
        components=tuple(components),
    )


def load_judgment_run_state(
    path: Path = DEFAULT_STATE_PATH,
) -> JudgmentRunState | None:
    """Load last scheduled judgment run state, or None when never run."""
    if not path.exists():
        return None
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    state = payload.get("state")
    if not isinstance(state, dict):
        msg = f"{path}: [state] section is required"
        raise ValueError(msg)
    return JudgmentRunState(
        last_run_at=str(state["last_run_at"]),
        run_id=str(state["run_id"]),
    )


def save_judgment_run_state(
    state: JudgmentRunState,
    path: Path = DEFAULT_STATE_PATH,
) -> None:
    """Persist last scheduled judgment run timestamp and run_id."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "[state]",
        f"last_run_at = {_format_toml_value(state.last_run_at)}",
        f"run_id = {_format_toml_value(state.run_id)}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def staleness_days_since(
    last_run_at: str,
    *,
    now: datetime | None = None,
) -> float:
    """Return fractional days since last_run_at ISO timestamp."""
    parsed = datetime.fromisoformat(last_run_at)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    reference = now or datetime.now(UTC)
    delta = reference - parsed.astimezone(UTC)
    return max(delta.total_seconds() / 86_400.0, 0.0)


def _run_script(
    script: Path,
    args: list[str],
    *,
    cwd: Path,
) -> None:
    completed = subprocess.run(
        ["uv", "run", "python", str(script), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        output = f"{completed.stdout}{completed.stderr}".strip()
        msg = f"{script.name} failed (exit {completed.returncode}): {output}"
        raise RuntimeError(msg)


def run_refute_promote_self_test(*, root: Path) -> None:
    """Run refute-or-promote golden self-test via audit_refute_promote.py."""
    _run_script(
        root / "scripts" / "audit_refute_promote.py",
        ["--self-test"],
        cwd=root,
    )


def run_docs_judgment_self_test(*, root: Path) -> None:
    """Run docs-judgment stub semantic self-test via audit_docs_judgment.py."""
    _run_script(
        root / "scripts" / "audit_docs_judgment.py",
        ["--self-test", "--no-emit"],
        cwd=root,
    )


def update_staleness_baseline(
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    *,
    state_path: Path = DEFAULT_STATE_PATH,
    schedule_path: Path = DEFAULT_SCHEDULE_PATH,
) -> float:
    """Record judgment.staleness_days in baseline from last_run_at age."""
    state = load_judgment_run_state(state_path)
    if state is None:
        observed = float(load_judgment_schedule(schedule_path).max_staleness_days)
    else:
        observed = staleness_days_since(state.last_run_at)
    baseline = load_baseline(baseline_path)
    tightened = update_metric(baseline, STALENESS_METRIC, observed, "minimize")
    save_baseline(tightened, path=baseline_path)
    return observed


def run_scheduled_judgment(
    audits_path: Path,
    *,
    emit_dispatch: bool = True,
    root: Path | None = None,
    schedule_path: Path = DEFAULT_SCHEDULE_PATH,
    state_path: Path = DEFAULT_STATE_PATH,
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    dispatch_path: Path | None = None,
    rubrics_dir: Path | None = None,
    routing_path: Path | None = None,
) -> ScheduledJudgmentResult:
    """Validate judgment wiring, run self-tests, record last run, update staleness."""
    resolved_root = (root or Path.cwd()).resolve()
    schedule = load_judgment_schedule(schedule_path)
    resolved_run_id = str(uuid.uuid4())

    dispatch_kwargs: dict[str, Path] = {}
    if dispatch_path is not None:
        dispatch_kwargs["dispatch_path"] = dispatch_path
    if rubrics_dir is not None:
        dispatch_kwargs["rubrics_dir"] = rubrics_dir
    if routing_path is not None:
        dispatch_kwargs["routing_path"] = routing_path

    validate_judgment_wiring(**dispatch_kwargs)

    dispatch_emitted = False
    if emit_dispatch:
        run_judgment_dispatch(
            audits_path,
            emit_observations=True,
            run_id=resolved_run_id,
            **dispatch_kwargs,
        )
        dispatch_emitted = True

    refute_promote_ok = False
    docs_judgment_ok = False
    try:
        run_refute_promote_self_test(root=resolved_root)
        refute_promote_ok = True
        run_docs_judgment_self_test(root=resolved_root)
        docs_judgment_ok = True
    except RuntimeError:
        return ScheduledJudgmentResult(
            passed=False,
            run_id=resolved_run_id,
            staleness_days=0.0,
            staleness_passed=False,
            dispatch_emitted=dispatch_emitted,
            refute_promote_ok=refute_promote_ok,
            docs_judgment_ok=docs_judgment_ok,
        )

    now_iso = _utc_now_iso()
    save_judgment_run_state(
        JudgmentRunState(last_run_at=now_iso, run_id=resolved_run_id),
        path=state_path,
    )
    staleness_days = staleness_days_since(now_iso)
    staleness_passed = staleness_days <= float(schedule.max_staleness_days)
    update_staleness_baseline(
        baseline_path,
        state_path=state_path,
        schedule_path=schedule_path,
    )

    return ScheduledJudgmentResult(
        passed=True,
        run_id=resolved_run_id,
        staleness_days=staleness_days,
        staleness_passed=staleness_passed,
        dispatch_emitted=dispatch_emitted,
        refute_promote_ok=refute_promote_ok,
        docs_judgment_ok=docs_judgment_ok,
    )


__all__ = [
    "DEFAULT_SCHEDULE_PATH",
    "DEFAULT_STATE_PATH",
    "STALENESS_METRIC",
    "JudgmentRunState",
    "JudgmentSchedule",
    "ScheduledJudgmentResult",
    "load_judgment_run_state",
    "load_judgment_schedule",
    "run_docs_judgment_self_test",
    "run_refute_promote_self_test",
    "run_scheduled_judgment",
    "save_judgment_run_state",
    "staleness_days_since",
    "update_staleness_baseline",
]
