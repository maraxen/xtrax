"""Standing gate T2-21/AC-28: forbid praxia-plugin-dispatch dependencies in src/xtrax/."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "xtrax"
sys.path.insert(0, str(ROOT / "scripts"))

from audit_compiler_boundary import scan  # noqa: E402
from audit_dispatch_independence import ALLOWLIST, FORBIDDEN_PATTERNS, main  # noqa: E402


def test_src_xtrax_currently_has_zero_violations() -> None:
    assert scan(SRC, root=ROOT, allowlist=ALLOWLIST, patterns=FORBIDDEN_PATTERNS) == []


def test_detects_praxia_import(tmp_path: Path) -> None:
    (tmp_path / "offender.py").write_text("import praxia\n", encoding="utf-8")
    violations = scan(tmp_path, root=tmp_path, allowlist=frozenset(), patterns=FORBIDDEN_PATTERNS)
    assert len(violations) == 1
    assert violations[0].pattern_label == "praxia-import"


def test_detects_praxia_from_import(tmp_path: Path) -> None:
    (tmp_path / "offender.py").write_text(
        "from praxia.mcp import client\n", encoding="utf-8"
    )
    violations = scan(tmp_path, root=tmp_path, allowlist=frozenset(), patterns=FORBIDDEN_PATTERNS)
    assert len(violations) == 1
    assert violations[0].pattern_label == "praxia-import"


def test_detects_mcp_praxia_tool_identifier(tmp_path: Path) -> None:
    (tmp_path / "offender.py").write_text(
        'result = call_tool("mcp__praxia__backlog")\n', encoding="utf-8"
    )
    violations = scan(tmp_path, root=tmp_path, allowlist=frozenset(), patterns=FORBIDDEN_PATTERNS)
    assert len(violations) == 1
    assert violations[0].pattern_label == "mcp-praxia-tool"


def test_detects_rig_run_underscore_variant(tmp_path: Path) -> None:
    (tmp_path / "offender.py").write_text("dispatch_via_rig_run(flow)\n", encoding="utf-8")
    violations = scan(tmp_path, root=tmp_path, allowlist=frozenset(), patterns=FORBIDDEN_PATTERNS)
    assert len(violations) == 1
    assert violations[0].pattern_label == "rig-run-dependency"


def test_detects_rig_run_hyphenated_variant(tmp_path: Path) -> None:
    (tmp_path / "offender.py").write_text('cmd = "praxia rig-run --flow x"\n', encoding="utf-8")
    violations = scan(tmp_path, root=tmp_path, allowlist=frozenset(), patterns=FORBIDDEN_PATTERNS)
    labels = {v.pattern_label for v in violations}
    assert "rig-run-dependency" in labels


def test_word_boundary_rejects_praxia_docs_path_references(tmp_path: Path) -> None:
    """This repo's own docstrings routinely cite `.praxia/docs/...` paths -- that entirely
    legitimate prose must NOT be flagged. This is the regression test for the pattern design
    choice (dependency signals, not a bare `praxia` substring) documented in the module docstring.
    """
    (tmp_path / "clean.py").write_text(
        '"""See `.praxia/docs/specs/260702_design-the-2181-agentic-algorithm-evolut.md` for '
        'the full rationale."""\n',
        encoding="utf-8",
    )
    assert scan(tmp_path, root=tmp_path, allowlist=frozenset(), patterns=FORBIDDEN_PATTERNS) == []


def test_word_boundary_rejects_unrelated_rig_usage(tmp_path: Path) -> None:
    """`rig` alone, or `rig` followed by an unrelated word, must NOT be flagged -- only the
    `rig_run`/`rig-run` compound is a real dispatch-identifier signal.
    """
    (tmp_path / "clean.py").write_text(
        "rig_teardown()\nrigging_config = {}\n", encoding="utf-8"
    )
    assert scan(tmp_path, root=tmp_path, allowlist=frozenset(), patterns=FORBIDDEN_PATTERNS) == []


def test_main_returns_1_and_prints_violation_to_stderr(tmp_path: Path, capsys: object) -> None:
    (tmp_path / "offender.py").write_text("import praxia\n", encoding="utf-8")
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
