"""N4.3 Refute-or-Promote judgment reliability protocol (#1593)."""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from xtrax.findings import (
    AuditFinding,
    Severity,
    append_finding,
    emit_judgment_finding,
)

DEFAULT_PROTOCOL_PATH = Path("audit/refute_or_promote.toml")
OBSERVATION_LABEL = "observation"

AssertFn = Callable[["JudgmentCandidate"], bool]
"""Return True when the assert pass supports the candidate."""

RefuteFn = Callable[["JudgmentCandidate"], bool]
"""Return True when the refute pass kills the candidate."""


@dataclass(frozen=True, slots=True)
class JudgmentCandidate:
    dimension: str
    severity: Severity
    file_line: str
    evidence: str
    rubric_id: str
    score: int = 0
    anchor_quote: str = ""
    symbol_qualname: str = ""


@dataclass(frozen=True, slots=True)
class PersonaPair:
    dimension: str
    assert_role: str
    refute_role: str


@dataclass(frozen=True, slots=True)
class RefutePromoteVerdict:
    promoted: bool
    dropped: bool
    label: str
    assert_role: str
    refute_role: str
    phase_log: tuple[str, ...]


def load_refute_promote_protocol(
    path: Path = DEFAULT_PROTOCOL_PATH,
) -> list[PersonaPair]:
    """Load persona pairs from refute_or_promote.toml."""
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        msg = f"{path}: [protocol] section is required"
        raise ValueError(msg)
    raw_pairs = payload.get("persona_pairs")
    if not isinstance(raw_pairs, list):
        msg = f"{path}: [[persona_pairs]] list is required"
        raise ValueError(msg)
    pairs: list[PersonaPair] = []
    for item in raw_pairs:
        if not isinstance(item, dict):
            msg = f"{path}: each [[persona_pairs]] entry must be a table"
            raise ValueError(msg)
        pairs.append(
            PersonaPair(
                dimension=str(item["dimension"]),
                assert_role=str(item["assert_role"]),
                refute_role=str(item["refute_role"]),
            )
        )
    return pairs


def resolve_persona_pair(
    dimension: str,
    *,
    path: Path = DEFAULT_PROTOCOL_PATH,
) -> tuple[str, str]:
    """Resolve (assert_role, refute_role) for a dimension; falls back to '*'."""
    pairs = load_refute_promote_protocol(path)
    default_pair: PersonaPair | None = None
    exact_pair: PersonaPair | None = None
    for pair in pairs:
        if pair.dimension == "*":
            default_pair = pair
        elif pair.dimension == dimension:
            exact_pair = pair
    if exact_pair is not None:
        return exact_pair.assert_role, exact_pair.refute_role
    if default_pair is not None:
        return default_pair.assert_role, default_pair.refute_role
    msg = f"{path}: no persona pair for dimension={dimension!r} and no '*' default"
    raise ValueError(msg)


def run_refute_or_promote(
    candidate: JudgmentCandidate,
    *,
    assert_fn: AssertFn,
    refute_fn: RefuteFn,
    assert_role: str,
    refute_role: str,
) -> RefutePromoteVerdict:
    """Run assert → refute state machine.

    Promoted iff assert_fn returns True AND refute_fn returns False
    (refute did not kill the candidate).
    """
    phase_log: list[str] = []
    assert_passed = assert_fn(candidate)
    phase_log.append(f"assert:{assert_role}:{'pass' if assert_passed else 'fail'}")
    if not assert_passed:
        return RefutePromoteVerdict(
            promoted=False,
            dropped=True,
            label=OBSERVATION_LABEL,
            assert_role=assert_role,
            refute_role=refute_role,
            phase_log=tuple(phase_log),
        )

    refute_killed = refute_fn(candidate)
    phase_log.append(f"refute:{refute_role}:{'kill' if refute_killed else 'survive'}")
    promoted = not refute_killed
    return RefutePromoteVerdict(
        promoted=promoted,
        dropped=not promoted,
        label=OBSERVATION_LABEL,
        assert_role=assert_role,
        refute_role=refute_role,
        phase_log=tuple(phase_log),
    )


def promote_and_emit(
    candidate: JudgmentCandidate,
    verdict: RefutePromoteVerdict,
    audits_path: Path,
    *,
    run_id: str,
) -> AuditFinding | None:
    """Emit observation finding when candidate survived refute-or-promote."""
    if not verdict.promoted:
        return None
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
            "label": verdict.label,
            "assert_role": verdict.assert_role,
            "refute_role": verdict.refute_role,
            "protocol": "refute_or_promote",
            "phase_log": list(verdict.phase_log),
        },
        run_id=run_id,
    )
    append_finding(record, audits_path=audits_path)
    return record


__all__ = [
    "DEFAULT_PROTOCOL_PATH",
    "OBSERVATION_LABEL",
    "AssertFn",
    "JudgmentCandidate",
    "PersonaPair",
    "RefuteFn",
    "RefutePromoteVerdict",
    "load_refute_promote_protocol",
    "promote_and_emit",
    "resolve_persona_pair",
    "run_refute_or_promote",
]
