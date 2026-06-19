#!/usr/bin/env python3
"""N4.4 Empirical-oracle CLI — self-test repro or promote one JSON candidate."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from xtrax.devtools.empirical_oracle import (
    DEFAULT_ORACLE_PATH,
    PromotionRequest,
    attempt_bug_promotion,
    load_oracle_config,
    run_pytest_repro,
)
from xtrax.devtools.refute_promote import JudgmentCandidate

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDITS_PATH = ROOT / ".praxia" / "audits.jsonl"
REPRO_FIXTURE = ROOT / "tests" / "fixtures" / "audit_repro_fail.py"


def _candidate_from_json(data: dict[str, object]) -> JudgmentCandidate:
    return JudgmentCandidate(
        dimension=str(data["dimension"]),
        severity=data["severity"],  # type: ignore[arg-type]
        file_line=str(data["file_line"]),
        evidence=str(data["evidence"]),
        rubric_id=str(data["rubric_id"]),
        score=int(data.get("score", 0)),
        anchor_quote=str(data.get("anchor_quote", "")),
        symbol_qualname=str(data.get("symbol_qualname", "")),
    )


def _run_self_test(oracle_path: Path) -> int:
    config = load_oracle_config(oracle_path)
    repro = run_pytest_repro(REPRO_FIXTURE, cwd=ROOT)
    if not repro.passed_gate:
        print(
            f"FAIL: repro fixture did not fail (exit_code={repro.exit_code})",
            file=sys.stderr,
        )
        if repro.output_snippet:
            print(repro.output_snippet, file=sys.stderr)
        return 1
    print(
        f"PASS: empirical-oracle self-test "
        f"version={config.version} repro_exit_code={repro.exit_code}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-path",
        type=Path,
        default=ROOT / DEFAULT_ORACLE_PATH,
        help="empirical_oracle.toml path",
    )
    parser.add_argument(
        "--audits-path",
        type=Path,
        default=DEFAULT_AUDITS_PATH,
        help="JSONL path for promoted bug findings",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in failing pytest repro fixture",
    )
    parser.add_argument(
        "--promote-json",
        type=Path,
        help="Path to JSON with finding_id, candidate, repro_test_path",
    )
    parser.add_argument(
        "--budget",
        type=int,
        help="Promotion budget override (default: max_promotions_per_run from config)",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return _run_self_test(args.oracle_path)

    if args.promote_json is None:
        parser.error("one of --self-test or --promote-json is required")

    payload = json.loads(args.promote_json.read_text(encoding="utf-8"))
    candidate_data = payload.get("candidate")
    if not isinstance(candidate_data, dict):
        parser.error("--promote-json must include a candidate object")
    request = PromotionRequest(
        finding_id=str(payload["finding_id"]),
        candidate=_candidate_from_json(candidate_data),
        repro_test_path=Path(str(payload["repro_test_path"])),
    )
    config = load_oracle_config(args.oracle_path)
    budget = (
        args.budget
        if args.budget is not None
        else config.max_promotions_per_run
    )
    verdict = attempt_bug_promotion(
        request,
        args.audits_path,
        budget=budget,
        run_id=str(uuid.uuid4()),
        cwd=ROOT,
    )
    state = "promoted" if verdict.promoted else "skipped"
    print(
        f"PASS: empirical-oracle {state} "
        f"label={verdict.label} budget_remaining={verdict.budget_remaining} "
        f"repro_exit_code={verdict.repro.exit_code}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
