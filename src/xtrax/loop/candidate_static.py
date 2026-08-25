"""Candidate-static gate (T2-11, #2181, AC-1, F0).

AC-1's failure modes: a candidate source file cannot be imported (syntax error,
missing imports, etc.) or violates jaxlint's JL-series correctness checks.
This module guards against both, rejecting pre-compute rather than discovering
these issues during evaluation.

"Candidate" here is a scoped concept: a local Python source file path, not the
generic `Candidate` TypeVar in `xtrax.stages.evaluate` (which is untouched and
must not be imported by this module).

Extension seam (mirrors `xtrax.loop.admission`'s and `xtrax.loop.closure_lock`'s
stance): this module raises on failure and stops there. No loop controller exists
yet to own retry/escalate behavior, and no CLI verb should be invented here. The
loop controller will compose `assert_candidate_static` once before evaluation.
"""

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xtrax.jaxlint_runner import run_jaxlint_json


class CandidateStaticGateError(Exception):
    """Base for AC-1's rejection errors. The candidate fails static checks; do not evaluate."""


@dataclass(frozen=True, slots=True)
class CandidateStaticFailure:
    """A single static check failure: import error or jaxlint violation."""

    check: str
    message: str


@dataclass(frozen=True, slots=True)
class CandidateStaticResult:
    """Result of running both static checks on a candidate file."""

    candidate_path: Path
    failures: tuple[CandidateStaticFailure, ...]
    passed: bool

    def to_json_dict(self) -> dict[str, Any]:
        """Return structured envelope matching xtrax.cli.graph_verb precedent."""
        return {
            "schema_version": 1,
            "failure_count": len(self.failures),
            "failures": [{"check": f.check, "message": f.message} for f in self.failures],
        }


def _check_clean_import(candidate_path: Path) -> CandidateStaticFailure | None:
    """Try to import the candidate file; return failure if it raises, else None.

    Broad exception catch (mirroring _trace_probe.py line 94 pattern) captures
    any import failure: syntax error, missing dependency, ValueError, etc.
    """
    try:
        spec = importlib.util.spec_from_file_location(
            "__candidate_static_temp__",
            candidate_path,
        )
        if spec is None or spec.loader is None:
            return CandidateStaticFailure(
                check="import",
                message=f"failed to create module spec for {candidate_path}",
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 — gate aggregates import failures
        return CandidateStaticFailure(
            check="import",
            message=f"{type(exc).__name__}: {str(exc)}",
        )
    return None


def _filter_jl_errors(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep jaxlint findings whose rule_id starts with JL and severity is error."""
    errors: list[dict[str, Any]] = []
    for finding in findings:
        rule_id = str(finding.get("rule_id", ""))
        severity = str(finding.get("severity", "")).lower()
        if rule_id.startswith("JL") and severity == "error":
            errors.append(finding)
    return errors


def _check_jaxlint(candidate_path: Path, root: Path) -> list[CandidateStaticFailure]:
    """Run jaxlint on candidate; return list of JL-error failures (may be empty)."""
    raw_findings = run_jaxlint_json(candidate_path, root=root)
    jl_errors = _filter_jl_errors(raw_findings)

    failures = []
    for finding in jl_errors:
        rule_id = str(finding.get("rule_id", ""))
        message = str(finding.get("message", ""))
        line = finding.get("line")
        location = f"{candidate_path}:{line}" if isinstance(line, int) else str(candidate_path)

        failures.append(
            CandidateStaticFailure(
                check=f"jaxlint/{rule_id}",
                message=f"{location}: {message}",
            )
        )
    return failures


def check_candidate_static(
    candidate_path: Path,
    *,
    root: Path | None = None,
) -> CandidateStaticResult:
    """Run both static checks on a candidate file; return result (does not raise).

    Checks:
      1. Clean import: try to import the candidate file.
      2. Jaxlint: zero JL-series error-severity violations.

    Args:
        candidate_path: Path to a Python source file.
        root: Root directory for jaxlint subprocess (defaults to cwd).

    Returns:
        CandidateStaticResult with passed=True iff both checks pass, else False
        with failures listed.
    """
    resolved_root = root or Path.cwd()
    resolved_path = candidate_path.resolve()

    failures: list[CandidateStaticFailure] = []

    # Check 1: clean import
    import_failure = _check_clean_import(resolved_path)
    if import_failure:
        failures.append(import_failure)

    # Check 2: jaxlint JL errors
    jaxlint_failures = _check_jaxlint(resolved_path, resolved_root)
    failures.extend(jaxlint_failures)

    return CandidateStaticResult(
        candidate_path=resolved_path,
        failures=tuple(failures),
        passed=len(failures) == 0,
    )


def assert_candidate_static(
    candidate_path: Path,
    *,
    root: Path | None = None,
) -> None:
    """Composed AC-1 gate: clean import + zero JL-series errors.

    Raises:
        CandidateStaticGateError: if `check_candidate_static` returns
            passed=False, with failures embedded in the message.
    """
    result = check_candidate_static(candidate_path, root=root)
    if not result.passed:
        envelope = result.to_json_dict()
        msg = f"candidate {candidate_path} failed static checks: {json.dumps(envelope, indent=2)}"
        raise CandidateStaticGateError(msg)


__all__ = [
    "CandidateStaticFailure",
    "CandidateStaticGateError",
    "CandidateStaticResult",
    "assert_candidate_static",
    "check_candidate_static",
]
