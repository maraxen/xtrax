#!/usr/bin/env python3
"""N4.3 Refute-or-Promote CLI — self-test golden candidates or one JSON candidate."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from xtrax.devtools.refute_promote import (
    DEFAULT_PROTOCOL_PATH,
    JudgmentCandidate,
    promote_and_emit,
    resolve_persona_pair,
    run_refute_or_promote,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDITS_PATH = ROOT / ".praxia" / "audits.jsonl"


def _golden_candidates() -> list[tuple[JudgmentCandidate, bool, bool, bool]]:
    """(candidate, assert_pass, refute_kill, expect_promoted)."""
    return [
        (
            JudgmentCandidate(
                dimension="type_hardening",
                severity="info",
                file_line="src/xtrax/training/trainer.py:10",
                evidence="annotation coverage below rubric anchor",
                rubric_id="type_hardening.coverage",
                score=3,
                anchor_quote="whole-baseline annotation coverage ratchet",
            ),
            True,
            False,
            True,
        ),
        (
            JudgmentCandidate(
                dimension="correctness",
                severity="info",
                file_line="src/xtrax/engine/loop.py:88",
                evidence="metamorphic parity suspect on zero-input edge",
                rubric_id="correctness.metamorphic",
                score=4,
                anchor_quote="oracle-class correctness claim",
            ),
            True,
            True,
            False,
        ),
        (
            JudgmentCandidate(
                dimension="documentation",
                severity="info",
                file_line="src/xtrax/run/spec.py:1",
                evidence="public module missing module docstring",
                rubric_id="documentation.module_doc",
            ),
            False,
            False,
            False,
        ),
    ]


def _run_self_test(protocol_path: Path) -> int:
    promoted = 0
    dropped = 0
    for candidate, assert_pass, refute_kill, expect_promoted in _golden_candidates():
        assert_role, refute_role = resolve_persona_pair(
            candidate.dimension,
            path=protocol_path,
        )

        def assert_fn(_: JudgmentCandidate) -> bool:
            return assert_pass

        def refute_fn(_: JudgmentCandidate) -> bool:
            return refute_kill

        verdict = run_refute_or_promote(
            candidate,
            assert_fn=assert_fn,
            refute_fn=refute_fn,
            assert_role=assert_role,
            refute_role=refute_role,
        )
        if verdict.promoted != expect_promoted:
            print(
                f"FAIL: {candidate.dimension} promoted={verdict.promoted} "
                f"expected={expect_promoted}",
                file=sys.stderr,
            )
            return 1
        if verdict.promoted:
            promoted += 1
        else:
            dropped += 1
    print(f"PASS: refute-or-promote self-test promoted={promoted} dropped={dropped}")
    return 0


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-path",
        type=Path,
        default=ROOT / DEFAULT_PROTOCOL_PATH,
        help="refute_or_promote.toml persona protocol path",
    )
    parser.add_argument(
        "--audits-path",
        type=Path,
        default=DEFAULT_AUDITS_PATH,
        help="JSONL path for promoted observation findings",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run built-in golden candidates through stub assert/refute fns",
    )
    parser.add_argument(
        "--candidate-json",
        type=Path,
        help="Path to one JudgmentCandidate JSON object",
    )
    parser.add_argument(
        "--assert-pass",
        action="store_true",
        help="Stub assert_fn returns True (with --candidate-json)",
    )
    parser.add_argument(
        "--refute-kill",
        action="store_true",
        help="Stub refute_fn returns True (with --candidate-json)",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return _run_self_test(args.protocol_path)

    if args.candidate_json is None:
        parser.error("one of --self-test or --candidate-json is required")

    payload = json.loads(args.candidate_json.read_text(encoding="utf-8"))
    candidate = _candidate_from_json(payload)
    assert_role, refute_role = resolve_persona_pair(
        candidate.dimension,
        path=args.protocol_path,
    )
    verdict = run_refute_or_promote(
        candidate,
        assert_fn=lambda _: args.assert_pass,
        refute_fn=lambda _: args.refute_kill,
        assert_role=assert_role,
        refute_role=refute_role,
    )
    record = promote_and_emit(
        candidate,
        verdict,
        args.audits_path,
        run_id=str(uuid.uuid4()),
    )
    state = "promoted" if verdict.promoted else "dropped"
    emitted = "yes" if record is not None else "no"
    print(
        f"PASS: refute-or-promote {state} "
        f"assert_role={assert_role} refute_role={refute_role} emitted={emitted}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
