"""Standing gate LC-02/AC-1b: forbid bathos dependencies in src/xtrax/."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "xtrax"
sys.path.insert(0, str(ROOT / "scripts"))

from audit_bathos_independence import ALLOWLIST, FORBIDDEN_PATTERNS, main  # noqa: E402
from audit_compiler_boundary import scan  # noqa: E402


def test_src_xtrax_currently_has_zero_violations() -> None:
    assert scan(SRC, root=ROOT, allowlist=ALLOWLIST, patterns=FORBIDDEN_PATTERNS) == []


def test_detects_bathos_import(tmp_path: Path) -> None:
    (tmp_path / "offender.py").write_text("import bathos\n", encoding="utf-8")
    violations = scan(tmp_path, root=tmp_path, allowlist=frozenset(), patterns=FORBIDDEN_PATTERNS)
    assert len(violations) == 1
    assert violations[0].pattern_label == "bathos-import"


def test_detects_bathos_from_import(tmp_path: Path) -> None:
    (tmp_path / "offender.py").write_text(
        "from bathos.campaigns import create_campaign\n", encoding="utf-8"
    )
    violations = scan(tmp_path, root=tmp_path, allowlist=frozenset(), patterns=FORBIDDEN_PATTERNS)
    assert len(violations) == 1
    assert violations[0].pattern_label == "bathos-import"


def test_detects_mcp_bathos_tool_identifier(tmp_path: Path) -> None:
    (tmp_path / "offender.py").write_text(
        'result = call_tool("mcp__bathos__run")\n', encoding="utf-8"
    )
    violations = scan(tmp_path, root=tmp_path, allowlist=frozenset(), patterns=FORBIDDEN_PATTERNS)
    assert len(violations) == 1
    assert violations[0].pattern_label == "mcp-bathos-tool"


def test_word_boundary_rejects_xtrax_independence_docstring_references(tmp_path: Path) -> None:
    """This repo's own docstrings routinely cite xtrax's independence from bathos -- that
    entirely legitimate prose must NOT be flagged. This is the regression test for the pattern
    design choice (dependency signals, not a bare `bathos` substring) documented in the module
    docstring.
    """
    (tmp_path / "clean.py").write_text(
        '"""xtrax has no dependency on bathos, by design. '
        'See `src/xtrax/run/component_binding.py` for the full rationale."""\n',
        encoding="utf-8",
    )
    assert scan(tmp_path, root=tmp_path, allowlist=frozenset(), patterns=FORBIDDEN_PATTERNS) == []


def test_word_boundary_rejects_legitimate_prose_mentions(tmp_path: Path) -> None:
    """References to bathos in paths, comments about the bathos-xtrax bridge, or other
    non-import prose must NOT be flagged.
    """
    (tmp_path / "clean.py").write_text(
        '"""See `.praxia/docs/specs/260716_loop-controller-epic-architecture-resolved.md` '
        'for bathos MCP integration patterns."""\n',
        encoding="utf-8",
    )
    assert scan(tmp_path, root=tmp_path, allowlist=frozenset(), patterns=FORBIDDEN_PATTERNS) == []


def test_main_returns_1_and_prints_violation_to_stderr(tmp_path: Path, capsys: object) -> None:
    (tmp_path / "offender.py").write_text("import bathos\n", encoding="utf-8")
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
