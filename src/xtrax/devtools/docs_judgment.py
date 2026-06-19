"""N4.5 Documentation judgment: structural RubricScorer + semantic judge (#1596)."""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from xtrax.devtools.emit import append_finding, emit_judgment_finding
from xtrax.devtools.gates._interrogate import run_interrogate_coverage
from xtrax.devtools.gates._jaxlint import run_jaxlint_json as _run_jaxlint_json
from xtrax.devtools.gates.documentation import filter_jd_jm_errors
from xtrax.devtools.refute_promote import OBSERVATION_LABEL
from xtrax.devtools.rubrics import RubricTable, load_rubric

DEFAULT_JUDGMENT_PATH = Path("audit/docs_judgment.toml")

SemanticJudgeFn = Callable[["StructuralScore", str], int]
"""Return semantic score (1–5) given structural evidence and target file_line."""


@dataclass(frozen=True, slots=True)
class DocsJudgmentConfig:
    dimension: str
    agent_role: str
    pass_threshold: int
    rubric_path: Path
    structural_signals: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StructuralScore:
    score: int
    anchor_quote: str
    evidence: str


@dataclass(frozen=True, slots=True)
class DocsJudgmentResult:
    structural: StructuralScore
    semantic_score: int
    passed: bool
    finding_emitted: bool


def load_docs_judgment_config(
    path: Path = DEFAULT_JUDGMENT_PATH,
) -> DocsJudgmentConfig:
    """Load docs judgment settings from docs_judgment.toml."""
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    judgment = payload.get("judgment")
    if not isinstance(judgment, dict):
        msg = f"{path}: [judgment] section is required"
        raise ValueError(msg)
    raw_signals = judgment.get("structural_signals", [])
    if not isinstance(raw_signals, list):
        msg = f"{path}: structural_signals must be a list"
        raise ValueError(msg)
    return DocsJudgmentConfig(
        dimension=str(judgment["dimension"]),
        agent_role=str(judgment["agent_role"]),
        pass_threshold=int(judgment["pass_threshold"]),
        rubric_path=Path(str(judgment["rubric_path"])),
        structural_signals=tuple(str(item) for item in raw_signals),
    )


def _anchor_for_score(rubric: RubricTable, score: int) -> str:
    for anchor in rubric.anchors:
        if anchor.score == score:
            return anchor.criterion
    msg = f"rubric {rubric.dimension!r} missing anchor score={score}"
    raise ValueError(msg)


def score_structural_docs(
    jd_count: int,
    coverage_pct: float,
    rubric: RubricTable,
) -> StructuralScore:
    """Map jaxlint JD/JM count + interrogate coverage to rubric anchor 1–5."""
    if jd_count > 0:
        score = 1
    elif coverage_pct < 80.0:
        score = 2
    elif coverage_pct < 90.0:
        score = 3
    elif coverage_pct < 95.0:
        score = 4
    else:
        score = 5
    evidence = f"jd_jm_errors={jd_count}, interrogate_coverage_pct={coverage_pct:.1f}"
    return StructuralScore(
        score=score,
        anchor_quote=_anchor_for_score(rubric, score),
        evidence=evidence,
    )


def collect_structural_signals(
    target: Path,
    *,
    root: Path | None = None,
) -> tuple[int, float]:
    """Pull JD/JM error count and interrogate coverage for a scan target."""
    resolved_root = root or Path.cwd()
    resolved_target = target.resolve()
    coverage_pct = run_interrogate_coverage(resolved_target, resolved_root)
    raw_findings = _run_jaxlint_json(
        resolved_target,
        root=resolved_root,
        performance_only=False,
    )
    jd_errors = filter_jd_jm_errors(raw_findings)
    return len(jd_errors), coverage_pct


def stub_semantic_judge(structural: StructuralScore, file_line: str) -> int:
    """CI stub: pass through structural score (ignores file_line)."""
    _ = file_line
    return structural.score


def run_docs_judgment(
    target: Path,
    audits_path: Path,
    *,
    semantic_judge_fn: SemanticJudgeFn,
    run_id: str,
    config_path: Path = DEFAULT_JUDGMENT_PATH,
    root: Path | None = None,
    emit_finding: bool = True,
) -> DocsJudgmentResult:
    """Run two-stage docs judgment and optionally emit observation finding."""
    config = load_docs_judgment_config(config_path)
    rubric = load_rubric(config.rubric_path)
    jd_count, coverage_pct = collect_structural_signals(target, root=root)
    structural = score_structural_docs(jd_count, coverage_pct, rubric)
    file_line = str(target.resolve())
    semantic_score = semantic_judge_fn(structural, file_line)
    passed = semantic_score >= config.pass_threshold
    finding_emitted = False

    if emit_finding:
        record = emit_judgment_finding(
            dim=config.dimension,
            severity="info",
            file_line=file_line,
            evidence=(
                f"structural={structural.score}; semantic={semantic_score}; {structural.evidence}"
            ),
            rubric_id=f"{config.dimension}.docs_judgment",
            score=semantic_score,
            anchor_quote=structural.anchor_quote,
            payload={
                "label": OBSERVATION_LABEL,
                "structural_score": structural.score,
                "semantic_score": semantic_score,
                "rubric_scorer_evidence": structural.evidence,
                "protocol": "docs_judgment",
                "agent_role": config.agent_role,
            },
            run_id=run_id,
        )
        append_finding(record, audits_path=audits_path)
        finding_emitted = True

    return DocsJudgmentResult(
        structural=structural,
        semantic_score=semantic_score,
        passed=passed,
        finding_emitted=finding_emitted,
    )


__all__ = [
    "DEFAULT_JUDGMENT_PATH",
    "DocsJudgmentConfig",
    "DocsJudgmentResult",
    "SemanticJudgeFn",
    "StructuralScore",
    "collect_structural_signals",
    "load_docs_judgment_config",
    "run_docs_judgment",
    "score_structural_docs",
    "stub_semantic_judge",
]
