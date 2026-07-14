"""Foundation gate T1-14/AC6: grep src/xtrax/ for UI/chain-map/plugin-state identifiers."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "xtrax"
sys.path.insert(0, str(ROOT / "scripts"))

from audit_compiler_boundary import ALLOWLIST, FORBIDDEN_PATTERNS, main, scan  # noqa: E402


def test_src_xtrax_currently_has_zero_violations() -> None:
    assert scan(SRC, root=ROOT) == []


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
    unfiltered = scan(SRC, root=ROOT, allowlist=frozenset())
    assert len(unfiltered) == len(ALLOWLIST)
    assert {(v.path, v.line_number) for v in unfiltered} == ALLOWLIST


def test_detects_chain_map_identifier(tmp_path: Path) -> None:
    (tmp_path / "offender.py").write_text("chain_map = {}\n", encoding="utf-8")
    violations = scan(tmp_path, root=tmp_path)
    assert len(violations) == 1
    assert violations[0].path == "offender.py"
    assert violations[0].line_number == 1
    assert violations[0].pattern_label == "chain_map/chain-map"


def test_detects_chain_map_hyphenated_variant(tmp_path: Path) -> None:
    (tmp_path / "offender.py").write_text('name = "chain-map-view"\n', encoding="utf-8")
    violations = scan(tmp_path, root=tmp_path)
    assert len(violations) == 1
    assert violations[0].pattern_label == "chain_map/chain-map"


def test_detects_mathjax_identifier(tmp_path: Path) -> None:
    (tmp_path / "offender.py").write_text("render_mathjax = True\n", encoding="utf-8")
    violations = scan(tmp_path, root=tmp_path)
    assert len(violations) == 1
    assert violations[0].pattern_label == "mathjax/math_jax"


def test_detects_plugin_state_identifier(tmp_path: Path) -> None:
    (tmp_path / "offender.py").write_text("plugin_state = None\n", encoding="utf-8")
    violations = scan(tmp_path, root=tmp_path)
    assert len(violations) == 1
    assert violations[0].pattern_label == "plugin_state/plugin-state"


def test_case_insensitive_matching(tmp_path: Path) -> None:
    (tmp_path / "offender.py").write_text(
        "CHAIN_MAP = {}\nMathJAX_ENABLED = True\n", encoding="utf-8"
    )
    violations = scan(tmp_path, root=tmp_path)
    labels = {v.pattern_label for v in violations}
    assert labels == {"chain_map/chain-map", "mathjax/math_jax"}


def test_word_boundary_rejects_same_word_extension(tmp_path: Path) -> None:
    """`mapper`/`statement` continue the word with more lowercase letters -- different words,
    not the forbidden term, so these must NOT be flagged.
    """
    (tmp_path / "clean.py").write_text(
        "chain_mapper_utility = 1\nplugin_statement_value = 2\n", encoding="utf-8"
    )
    assert scan(tmp_path, root=tmp_path) == []


def test_word_boundary_accepts_underscore_joined_compound_identifiers(tmp_path: Path) -> None:
    """`_`/case-transitions are valid identifier-component separators (Python convention) --
    unlike a same-word lowercase extension, these ARE flagged.
    """
    (tmp_path / "offender.py").write_text(
        "chain_map_view = 1\nplugin_state_machine = 2\nrender_mathjax_output = 3\n",
        encoding="utf-8",
    )
    violations = scan(tmp_path, root=tmp_path)
    labels = {v.pattern_label for v in violations}
    assert labels == {"chain_map/chain-map", "plugin_state/plugin-state", "mathjax/math_jax"}


def test_word_boundary_accepts_mid_identifier_camelcase_compounds(tmp_path: Path) -> None:
    """A lowercase->Uppercase transition mid-identifier is a valid boundary too, not just at
    string-start -- `someChainMapValue`/`getPluginStateNow` must be flagged, not silently missed.
    """
    (tmp_path / "offender.py").write_text(
        "someChainMapValue = 1\n"
        "def getPluginStateNow():\n"
        "    pass\n"
        "class NodeChainMapWidget:\n"
        "    pass\n",
        encoding="utf-8",
    )
    violations = scan(tmp_path, root=tmp_path)
    labels = {v.pattern_label for v in violations}
    assert labels == {"chain_map/chain-map", "plugin_state/plugin-state"}


def test_detects_math_jax_snake_case_variant(tmp_path: Path) -> None:
    """`mathjax`, like `chain_map`/`plugin_state`, must tolerate a `_`/`-` separator -- the
    idiomatic snake_case spelling `math_jax` must be caught, not just the fused `mathjax`.
    """
    (tmp_path / "offender.py").write_text("math_jax_enabled = True\n", encoding="utf-8")
    violations = scan(tmp_path, root=tmp_path)
    assert len(violations) == 1
    assert violations[0].pattern_label == "mathjax/math_jax"


def test_scans_toml_files_too(tmp_path: Path) -> None:
    (tmp_path / "schema.toml").write_text('id = "mathjax_label"\n', encoding="utf-8")
    violations = scan(tmp_path, root=tmp_path)
    assert len(violations) == 1
    assert violations[0].path == "schema.toml"


def test_clean_file_produces_no_violations(tmp_path: Path) -> None:
    (tmp_path / "clean.py").write_text("def infer_bundle(fn):\n    return fn\n", encoding="utf-8")
    assert scan(tmp_path, root=tmp_path) == []


def test_main_returns_1_and_prints_violation_to_stderr(tmp_path: Path, capsys: object) -> None:
    (tmp_path / "offender.py").write_text("chain_map = {}\n", encoding="utf-8")
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
