"""N4.1 judgment-track dispatch wiring — roster, validate, observation emit (#1590)."""

from __future__ import annotations

import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path

from xtrax.devtools.routing import DEFAULT_ROUTING_PATH, resolve_destination
from xtrax.devtools.rubrics import (
    DEFAULT_RUBRICS_DIR,
    RubricTable,
    load_all_rubrics,
    load_rubric,
)
from xtrax.findings import Severity, append_finding, emit_judgment_finding

DEFAULT_DISPATCH_PATH = Path("audit/judgment_dispatch.toml")


@dataclass(frozen=True, slots=True)
class JudgmentDispatchEntry:
    dimension: str
    agent_role: str
    rubric_path: Path
    default_severity: Severity
    label: str


@dataclass(frozen=True, slots=True)
class JudgmentDispatchRun:
    entry: JudgmentDispatchEntry
    destination: str
    anchor_quote: str
    finding_emitted: bool


@dataclass(frozen=True, slots=True)
class JudgmentResult:
    passed: bool
    entries: tuple[JudgmentDispatchRun, ...]
    findings_emitted: int
    destinations: dict[str, str]


def load_judgment_dispatch(
    path: Path = DEFAULT_DISPATCH_PATH,
) -> list[JudgmentDispatchEntry]:
    """Load dimension→agent roster from judgment_dispatch.toml."""
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    dispatch = payload.get("dispatch")
    if not isinstance(dispatch, dict):
        msg = f"{path}: [dispatch] section is required"
        raise ValueError(msg)
    raw_dimensions = payload.get("dimensions")
    if not isinstance(raw_dimensions, list):
        msg = f"{path}: [[dimensions]] list is required"
        raise ValueError(msg)
    entries: list[JudgmentDispatchEntry] = []
    for item in raw_dimensions:
        if not isinstance(item, dict):
            msg = f"{path}: each [[dimensions]] entry must be a table"
            raise ValueError(msg)
        entries.append(
            JudgmentDispatchEntry(
                dimension=str(item["dimension"]),
                agent_role=str(item["agent_role"]),
                rubric_path=Path(str(item["rubric_path"])),
                default_severity=item["default_severity"],
                label=str(item["label"]),
            )
        )
    return entries


def _anchor_quote_for_score(table: RubricTable, score: int) -> str:
    for anchor in table.anchors:
        if anchor.score == score:
            return anchor.criterion
    msg = f"rubric {table.dimension!r} missing anchor score={score}"
    raise ValueError(msg)


def validate_judgment_wiring(
    *,
    dispatch_path: Path = DEFAULT_DISPATCH_PATH,
    rubrics_dir: Path = DEFAULT_RUBRICS_DIR,
    routing_path: Path = DEFAULT_ROUTING_PATH,
) -> None:
    """Ensure every dispatch dimension has a rubric and non-empty agent_role."""
    entries = load_judgment_dispatch(dispatch_path)
    rubrics = load_all_rubrics(dir=rubrics_dir)
    errors: list[str] = []
    for entry in entries:
        if not entry.agent_role.strip():
            errors.append(f"{entry.dimension}: agent_role is empty")
        rubric_file = rubrics_dir / f"{entry.dimension}.toml"
        if entry.dimension not in rubrics:
            errors.append(
                f"{entry.dimension}: no rubric table in {rubrics_dir} (expected {rubric_file.name})"
            )
        elif entry.rubric_path.name != f"{entry.dimension}.toml":
            errors.append(
                f"{entry.dimension}: rubric_path {entry.rubric_path!s} "
                f"does not match dimension stem"
            )
        else:
            try:
                resolve_destination(
                    domain=entry.dimension,
                    track="judgment",
                    severity=entry.default_severity,
                    path=routing_path,
                )
            except ValueError as exc:
                errors.append(f"{entry.dimension}: routing unresolved — {exc}")
    if errors:
        bullet_lines = "\n".join(f"  - {e}" for e in errors)
        msg = f"judgment dispatch wiring failed:\n{bullet_lines}"
        raise ValueError(msg)


def run_judgment_dispatch(
    audits_path: Path,
    *,
    emit_observations: bool = True,
    run_id: str | None = None,
    dispatch_path: Path = DEFAULT_DISPATCH_PATH,
    rubrics_dir: Path = DEFAULT_RUBRICS_DIR,
    routing_path: Path = DEFAULT_ROUTING_PATH,
) -> JudgmentResult:
    """Validate wiring; optionally emit info-level armed observations."""
    validate_judgment_wiring(
        dispatch_path=dispatch_path,
        rubrics_dir=rubrics_dir,
        routing_path=routing_path,
    )
    resolved_run_id = run_id or str(uuid.uuid4())
    entries = load_judgment_dispatch(dispatch_path)
    runs: list[JudgmentDispatchRun] = []
    destinations: dict[str, str] = {}
    findings_emitted = 0

    for entry in entries:
        rubric_path = rubrics_dir / f"{entry.dimension}.toml"
        table = load_rubric(rubric_path)
        anchor_quote = _anchor_quote_for_score(table, score=3)
        destination = resolve_destination(
            domain=entry.dimension,
            track="judgment",
            severity=entry.default_severity,
            path=routing_path,
        )
        destinations[entry.dimension] = destination
        finding_emitted = False
        if emit_observations:
            record = emit_judgment_finding(
                dim=entry.dimension,
                severity=entry.default_severity,
                file_line=f"{entry.rubric_path}:dispatch",
                evidence=f"judgment dispatch armed for {entry.agent_role}",
                rubric_id=f"{entry.dimension}.dispatch",
                score=0,
                anchor_quote=anchor_quote,
                run_id=resolved_run_id,
            )
            append_finding(record, audits_path=audits_path)
            finding_emitted = True
            findings_emitted += 1
        runs.append(
            JudgmentDispatchRun(
                entry=entry,
                destination=destination,
                anchor_quote=anchor_quote,
                finding_emitted=finding_emitted,
            )
        )

    return JudgmentResult(
        passed=True,
        entries=tuple(runs),
        findings_emitted=findings_emitted,
        destinations=destinations,
    )


__all__ = [
    "DEFAULT_DISPATCH_PATH",
    "JudgmentDispatchEntry",
    "JudgmentDispatchRun",
    "JudgmentResult",
    "load_judgment_dispatch",
    "run_judgment_dispatch",
    "validate_judgment_wiring",
]
