#!/usr/bin/env python3
"""N4.5 Docs judgment CLI — self-test with stub semantic judge."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from xtrax.devtools.docs_judgment import (
    DEFAULT_JUDGMENT_PATH,
    load_docs_judgment_config,
    run_docs_judgment,
    stub_semantic_judge,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT / "src" / "xtrax"
DEFAULT_AUDITS_PATH = ROOT / ".praxia" / "audits.jsonl"


def _run_self_test(
    *,
    target: Path,
    config_path: Path,
    audits_path: Path,
    no_emit: bool,
) -> int:
    config = load_docs_judgment_config(config_path)
    result = run_docs_judgment(
        target,
        audits_path,
        semantic_judge_fn=stub_semantic_judge,
        run_id=str(uuid.uuid4()),
        config_path=config_path,
        root=ROOT,
        emit_finding=not no_emit,
    )
    print(
        f"PASS: docs-judgment self-test "
        f"structural={result.structural.score} "
        f"semantic={result.semantic_score} "
        f"passed={result.passed} "
        f"threshold={config.pass_threshold} "
        f"finding_emitted={result.finding_emitted}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        nargs="?",
        default=str(DEFAULT_TARGET),
        help="Path to scan (default: src/xtrax)",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=ROOT / DEFAULT_JUDGMENT_PATH,
        help="docs_judgment.toml path",
    )
    parser.add_argument(
        "--audits-path",
        type=Path,
        default=DEFAULT_AUDITS_PATH,
        help="JSONL path for emitted judgment findings",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run pipeline with stub semantic judge (returns structural score)",
    )
    parser.add_argument(
        "--no-emit",
        action="store_true",
        help="Do not append judgment findings to audits JSONL",
    )
    args = parser.parse_args(argv)

    target = Path(args.target).resolve()
    if not target.exists():
        print(f"target not found: {target}", file=sys.stderr)
        return 2

    if not args.self_test:
        parser.error("--self-test is required")

    return _run_self_test(
        target=target,
        config_path=args.config_path,
        audits_path=args.audits_path,
        no_emit=args.no_emit,
    )


if __name__ == "__main__":
    sys.exit(main())
