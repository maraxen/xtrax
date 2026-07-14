"""Unit tests for conftest's max_discrepancy extraction (backlog #3493 sub-task).

Prior behavior: ``_emit_tier_result``'s FAIL branch (in
``pytest_runtest_makereport``) never passed ``max_discrepancy`` to
``_emit_tier_result``, so every FAIL verdict recorded in ``audits.jsonl``
carried ``max_discrepancy=None`` regardless of how large the actual numeric
drift was. This made the port-domain FAIL record useless for triage and — the
motivating case for backlog #3493 — left the S6 bathos harness-wrapper's
verdict sidecar with no real ``max_discrepancy`` to report either.

The fix adds:
  - ``_extract_max_discrepancy``: a best-effort parser of numpy's
    ``assert_allclose``/``assert_array_almost_equal`` failure message (the
    only kind of failure the graded parity tiers raise).
  - ``_max_discrepancy_for_call``: a thin adapter over a pytest
    ``CallInfo`` that the FAIL branch of ``pytest_runtest_makereport`` calls
    instead of hardcoding ``None`` — factored out specifically so this test
    can exercise the real call site without having to hand-drive a
    ``hookimpl(hookwrapper=True)`` generator (fragile / pytest-internals
    coupled).

These tests exercise the pure functions directly (loaded via importlib,
mirroring the existing ``_load_port_emit_module`` pattern in conftest.py
itself) so they do not depend on a live tier-test failure to run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

CONFTEST_PATH = Path(__file__).resolve().parent / "conftest.py"


def _load_conftest():
    spec = importlib.util.spec_from_file_location(
        "port_conftest_under_test", CONFTEST_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("port_conftest_under_test", module)
    spec.loader.exec_module(module)
    return module


conftest = _load_conftest()


# --- _extract_max_discrepancy -------------------------------------------------


def test_extract_max_discrepancy_none_without_exception() -> None:
    assert conftest._extract_max_discrepancy(None) is None


def test_extract_max_discrepancy_parses_legacy_numpy_message() -> None:
    exc = AssertionError(
        "\nNot equal to tolerance rtol=0.0001, atol=0.0001\n\n"
        "Mismatched elements: 1 / 4 (25%)\n"
        "Max absolute difference: 0.0012\n"
        "Max relative difference: 0.01\n"
    )
    assert conftest._extract_max_discrepancy(exc) == 0.0012


def test_extract_max_discrepancy_parses_modern_numpy_among_violations_message() -> None:
    # Newer numpy (>=2.x) phrasing: "Max absolute difference among violations: X"
    exc = AssertionError(
        "\nNot equal to tolerance rtol=0.0001, atol=0.0001\n\n"
        "Mismatched elements: 1 / 2 (50%)\n"
        "Max absolute difference among violations: 0.5\n"
        "Max relative difference among violations: 0.2\n"
    )
    assert conftest._extract_max_discrepancy(exc) == 0.5


def test_extract_max_discrepancy_parses_scientific_notation() -> None:
    exc = AssertionError("Max absolute difference: 1.23e-05\n")
    assert conftest._extract_max_discrepancy(exc) == 1.23e-05


def test_extract_max_discrepancy_none_when_no_match() -> None:
    exc = AssertionError("some unrelated failure with no numeric-drift phrasing")
    assert conftest._extract_max_discrepancy(exc) is None


def test_extract_max_discrepancy_matches_regardless_of_exception_type() -> None:
    # The FAIL branch passes whatever call.excinfo.value is; the parser only
    # inspects the message text, not the exception type.
    exc = RuntimeError("Max absolute difference: 0.5")
    assert conftest._extract_max_discrepancy(exc) == 0.5


# --- _max_discrepancy_for_call (the actual FAIL-branch call site) -------------


class _FakeExcInfo:
    def __init__(self, value: BaseException) -> None:
        self.value = value


class _FakeCallInfo:
    """Minimal stand-in for pytest.CallInfo — only .excinfo is read."""

    def __init__(self, excinfo: _FakeExcInfo | None) -> None:
        self.excinfo = excinfo


def test_max_discrepancy_for_call_none_when_no_excinfo() -> None:
    assert conftest._max_discrepancy_for_call(_FakeCallInfo(excinfo=None)) is None


def test_max_discrepancy_for_call_extracts_real_value() -> None:
    call = _FakeCallInfo(
        excinfo=_FakeExcInfo(
            AssertionError("Max absolute difference among violations: 0.00042\n")
        )
    )
    assert conftest._max_discrepancy_for_call(call) == 0.00042


def test_max_discrepancy_for_call_none_when_message_has_no_numeric_drift() -> None:
    call = _FakeCallInfo(excinfo=_FakeExcInfo(AssertionError("unrelated failure")))
    assert conftest._max_discrepancy_for_call(call) is None
