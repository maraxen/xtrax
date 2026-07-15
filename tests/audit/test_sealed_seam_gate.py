"""Standing gate T2-04/AC-E2: forbid mock/test-double EvaluateFn seam imports in src/xtrax/."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "xtrax"
sys.path.insert(0, str(ROOT / "scripts"))

from audit_compiler_boundary import scan  # noqa: E402
from audit_sealed_seam import ALLOWLIST, FORBIDDEN_PATTERNS, main  # noqa: E402


def test_src_xtrax_currently_has_zero_violations() -> None:
    assert scan(SRC, root=ROOT, allowlist=ALLOWLIST, patterns=FORBIDDEN_PATTERNS) == []


def test_detects_mock_evaluator_class(tmp_path: Path) -> None:
    (tmp_path / "offender.py").write_text("class MockEvaluator:\n    pass\n", encoding="utf-8")
    violations = scan(tmp_path, root=tmp_path, allowlist=frozenset(), patterns=FORBIDDEN_PATTERNS)
    assert len(violations) == 1
    assert violations[0].pattern_label == "mock-evaluator"


def test_detects_snake_case_fake_evaluator(tmp_path: Path) -> None:
    (tmp_path / "offender.py").write_text(
        "fake_evaluate_fn = lambda ctx, c: {}\n", encoding="utf-8"
    )
    violations = scan(tmp_path, root=tmp_path, allowlist=frozenset(), patterns=FORBIDDEN_PATTERNS)
    assert len(violations) == 1
    assert violations[0].pattern_label == "fake-evaluator"


def test_detects_stub_evaluator(tmp_path: Path) -> None:
    (tmp_path / "offender.py").write_text("stub_evaluator = None\n", encoding="utf-8")
    violations = scan(tmp_path, root=tmp_path, allowlist=frozenset(), patterns=FORBIDDEN_PATTERNS)
    assert len(violations) == 1
    assert violations[0].pattern_label == "stub-evaluator"


def test_detects_dummy_evaluation_class(tmp_path: Path) -> None:
    (tmp_path / "offender.py").write_text("class DummyEvaluation:\n    pass\n", encoding="utf-8")
    violations = scan(tmp_path, root=tmp_path, allowlist=frozenset(), patterns=FORBIDDEN_PATTERNS)
    assert len(violations) == 1
    assert violations[0].pattern_label == "dummy-evaluator"


def test_detects_unittest_mock_import(tmp_path: Path) -> None:
    (tmp_path / "offender.py").write_text(
        "from unittest.mock import MagicMock\n", encoding="utf-8"
    )
    violations = scan(tmp_path, root=tmp_path, allowlist=frozenset(), patterns=FORBIDDEN_PATTERNS)
    assert len(violations) == 1
    assert violations[0].pattern_label == "unittest-mock-import"


def test_word_boundary_rejects_unrelated_evaluat_usage(tmp_path: Path) -> None:
    """Plain `evaluate`/`Evaluator` usage -- with no mock/fake/stub/dummy prefix -- is the real
    EvaluateFn/SealedEvaluatorRegistry seam this gate exists to protect, and must NOT be flagged.
    """
    (tmp_path / "clean.py").write_text(
        "def evaluate_candidate(frozen_context, candidate):\n"
        "    return {'score': 1.0}\n"
        "class RealEvaluator:\n"
        "    pass\n",
        encoding="utf-8",
    )
    assert scan(tmp_path, root=tmp_path, allowlist=frozenset(), patterns=FORBIDDEN_PATTERNS) == []


def test_main_returns_1_and_prints_violation_to_stderr(tmp_path: Path, capsys: object) -> None:
    (tmp_path / "offender.py").write_text("class MockEvaluator:\n    pass\n", encoding="utf-8")
    exit_code = main([str(tmp_path)])
    assert exit_code == 1
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "offender.py:1" in captured.err
    assert "FAIL" in captured.err


def test_main_returns_0_and_prints_pass_when_clean(tmp_path: Path, capsys: object) -> None:
    (tmp_path / "clean.py").write_text("x = 1\n", encoding="utf-8")
    exit_code = main([str(tmp_path)])
    assert exit_code == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "PASS" in captured.out
