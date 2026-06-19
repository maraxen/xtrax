"""N4.4 Empirical-oracle (failing-pytest) promotion path (#1595)."""

from __future__ import annotations

import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

from xtrax.devtools.emit import append_finding, emit_judgment_finding
from xtrax.devtools.refute_promote import JudgmentCandidate

DEFAULT_ORACLE_PATH = Path("audit/empirical_oracle.toml")
BUG_LABEL = "bug"
DEFAULT_MAX_PROMOTIONS_PER_RUN = 3


@dataclass(frozen=True, slots=True)
class OracleConfig:
    version: str
    max_promotions_per_run: int
    promotion_requires: str
    executable_check: str


@dataclass(frozen=True, slots=True)
class PromotionRequest:
    finding_id: str
    candidate: JudgmentCandidate
    repro_test_path: Path


@dataclass(frozen=True, slots=True)
class ReproResult:
    passed_gate: bool
    exit_code: int
    output_snippet: str


@dataclass(frozen=True, slots=True)
class BugPromotionVerdict:
    promoted: bool
    label: str
    budget_remaining: int
    repro: ReproResult


def load_oracle_config(path: Path = DEFAULT_ORACLE_PATH) -> OracleConfig:
    """Load empirical-oracle promotion settings from empirical_oracle.toml."""
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    oracle = payload.get("oracle")
    if not isinstance(oracle, dict):
        msg = f"{path}: [oracle] section is required"
        raise ValueError(msg)
    return OracleConfig(
        version=str(oracle.get("version", "0.0.0")),
        max_promotions_per_run=int(
            oracle.get("max_promotions_per_run", DEFAULT_MAX_PROMOTIONS_PER_RUN)
        ),
        promotion_requires=str(oracle.get("promotion_requires", "failing_pytest")),
        executable_check=str(
            oracle.get(
                "executable_check",
                "is this finding reproducible as a failing test?",
            )
        ),
    )


def run_pytest_repro(test_path: Path, *, cwd: Path | None = None) -> ReproResult:
    """Run pytest repro; passed_gate is True when the test fails (non-zero exit)."""
    completed = subprocess.run(
        ["uv", "run", "pytest", str(test_path), "-q"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    output = f"{completed.stdout}{completed.stderr}".strip()
    snippet = output[-2000:] if len(output) > 2000 else output
    return ReproResult(
        passed_gate=completed.returncode != 0,
        exit_code=completed.returncode,
        output_snippet=snippet,
    )


def attempt_bug_promotion(
    request: PromotionRequest,
    audits_path: Path,
    *,
    budget: int,
    run_id: str,
    cwd: Path | None = None,
) -> BugPromotionVerdict:
    """Promote observation→bug when repro fails and budget allows."""
    repro = run_pytest_repro(request.repro_test_path, cwd=cwd)
    if budget <= 0 or not repro.passed_gate:
        return BugPromotionVerdict(
            promoted=False,
            label=BUG_LABEL,
            budget_remaining=budget,
            repro=repro,
        )

    candidate = request.candidate
    record = emit_judgment_finding(
        dim=candidate.dimension,
        severity=candidate.severity,
        file_line=candidate.file_line,
        evidence=candidate.evidence,
        rubric_id=candidate.rubric_id,
        score=candidate.score,
        anchor_quote=candidate.anchor_quote,
        symbol_qualname=candidate.symbol_qualname,
        payload={
            "label": BUG_LABEL,
            "promoted_from": request.finding_id,
            "repro_test_path": str(request.repro_test_path),
            "repro_exit_code": repro.exit_code,
            "protocol": "empirical_oracle",
        },
        run_id=run_id,
    )
    append_finding(record, audits_path=audits_path)
    return BugPromotionVerdict(
        promoted=True,
        label=BUG_LABEL,
        budget_remaining=budget - 1,
        repro=repro,
    )


__all__ = [
    "BUG_LABEL",
    "BugPromotionVerdict",
    "DEFAULT_MAX_PROMOTIONS_PER_RUN",
    "DEFAULT_ORACLE_PATH",
    "OracleConfig",
    "PromotionRequest",
    "ReproResult",
    "attempt_bug_promotion",
    "load_oracle_config",
    "run_pytest_repro",
]
