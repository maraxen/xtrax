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


def test_word_boundary_rejects_the_exact_false_positive_shape_that_needed_anchoring(
    tmp_path: Path,
) -> None:
    """Regression test for the real false-positive an unanchored bathos-import pattern produced
    (caught by adversarial review before merge): ordinary English like "does not import bathos"
    or "obtained from bathos" contains the literal substring "import bathos"/"from bathos" without
    being a real Python import statement. Anchoring the pattern to logical line start
    (`^\\s*(?:import|from)\\s+bathos\\b`) is what makes this NOT a violation -- these sentences
    never start a line with `import`/`from`.
    """
    (tmp_path / "clean.py").write_text(
        '"""This module does not import bathos, by design -- every value it needs is obtained '
        'from bathos by the caller, one layer up."""\n',
        encoding="utf-8",
    )
    assert scan(tmp_path, root=tmp_path, allowlist=frozenset(), patterns=FORBIDDEN_PATTERNS) == []


def test_allowlist_entries_are_still_real_matches() -> None:
    """Ratchet: catches a stale ALLOWLIST entry pointing at text that no longer matches."""
    for rel_path, line_number in ALLOWLIST:
        lines = (ROOT / rel_path).read_text(encoding="utf-8").splitlines()
        line = lines[line_number - 1]
        assert any(pattern.search(line) for _, pattern in FORBIDDEN_PATTERNS), (
            f"stale allowlist entry {rel_path}:{line_number} matches no forbidden pattern"
        )


def test_allowlist_actually_suppresses_a_real_hit() -> None:
    """Proves ALLOWLIST isn't vacuous: without it, the same scan finds a real violation."""
    unfiltered = scan(SRC, root=ROOT, allowlist=frozenset(), patterns=FORBIDDEN_PATTERNS)
    assert len(unfiltered) == len(ALLOWLIST)
    assert {(v.path, v.line_number) for v in unfiltered} == ALLOWLIST


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
