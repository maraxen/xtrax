"""Tests for scripts/assemble_changelog.py (changelog.d fragments -> release section)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from scripts.assemble_changelog import (
    CATEGORIES,
    ChangelogError,
    _split_unreleased,
    assemble,
    list_problems,
    load_fragments,
    main,
)

CHANGELOG = textwrap.dedent(
    """\
    # Changelog

    Intro text.

    ## [Unreleased]

    ### Added

    - Existing added bullet.

    ### Fixed

    Prose paragraph kept verbatim.

    - Existing fixed bullet.

    ## [0.1.0] - 2026-01-01

    ### Added

    - First release.
    """
)


def _setup(tmp_path: Path, changelog: str = CHANGELOG) -> tuple[Path, Path]:
    cl = tmp_path / "CHANGELOG.md"
    cl.write_text(changelog)
    frag = tmp_path / "changelog.d"
    frag.mkdir()
    (frag / "README.md").write_text("how to write fragments\n")
    (frag / ".gitkeep").write_text("")
    return cl, frag


def _run(tmp_path: Path, *extra: str) -> int:
    return main(
        [
            "--changelog",
            str(tmp_path / "CHANGELOG.md"),
            "--fragments-dir",
            str(tmp_path / "changelog.d"),
            *extra,
        ]
    )


def test_assembles_fragments_with_existing_unreleased_content(tmp_path: Path) -> None:
    cl, frag = _setup(tmp_path)
    (frag / "new-thing.added.md").write_text("- New added bullet.\n")
    (frag / "bug.fixed.md").write_text("- New fixed bullet.\n")
    (frag / "gone.removed.md").write_text("- Removed `old`.\n")
    assert _run(tmp_path, "--version", "0.2.0", "--date", "2026-02-03") == 0
    text = cl.read_text()

    # Emptied Unreleased sits directly above the new section, which sits above 0.1.0.
    assert text.index("## [Unreleased]") < text.index("## [0.2.0] - 2026-02-03")
    assert text.index("## [0.2.0] - 2026-02-03") < text.index("## [0.1.0] - 2026-01-01")
    unreleased = text[text.index("## [Unreleased]") : text.index("## [0.2.0]")]
    assert "###" not in unreleased and "Existing" not in unreleased

    section = text[text.index("## [0.2.0]") : text.index("## [0.1.0]")]
    # Existing bullets keep their category; new bullets join the same category tightly.
    assert "- Existing added bullet.\n- New added bullet.\n" in section
    assert (
        "Prose paragraph kept verbatim.\n\n- Existing fixed bullet.\n- New fixed bullet." in section
    )
    assert "### Removed\n\n- Removed `old`." in section
    # Untouched history.
    assert "## [0.1.0] - 2026-01-01\n\n### Added\n\n- First release.\n" in text
    assert text.startswith("# Changelog\n\nIntro text.\n\n## [Unreleased]")


def test_categories_follow_keep_a_changelog_order(tmp_path: Path) -> None:
    cl, frag = _setup(tmp_path, "# Changelog\n\n## [Unreleased]\n\n## [0.1.0] - 2026-01-01\n")
    for category in reversed(CATEGORIES):  # create in the wrong order on purpose
        (frag / f"x-{category}.{category}.md").write_text(f"- {category} entry\n")
    assert _run(tmp_path, "--version", "0.2.0", "--date", "2026-02-03") == 0
    headings = [line for line in cl.read_text().splitlines() if line.startswith("### ")]
    assert headings == [f"### {c.capitalize()}" for c in CATEGORIES]


def test_fragments_are_deleted_but_readme_and_gitkeep_stay(tmp_path: Path) -> None:
    _, frag = _setup(tmp_path)
    (frag / "a.added.md").write_text("- a\n")
    (frag / "b.changed.md").write_text("- b\n")
    assert _run(tmp_path, "--version", "0.2.0", "--date", "2026-02-03") == 0
    assert sorted(p.name for p in frag.iterdir()) == [".gitkeep", "README.md"]


def test_fragments_in_one_category_are_ordered_by_file_name(tmp_path: Path) -> None:
    cl, frag = _setup(tmp_path, "# Changelog\n\n## [Unreleased]\n")
    (frag / "b.added.md").write_text("- second\n")
    (frag / "a.added.md").write_text("- first\n")
    assert _run(tmp_path, "--version", "1.0", "--date", "2026-02-03") == 0
    assert "- first\n- second\n" in cl.read_text()


def test_wrapped_bullets_from_two_fragments_stay_one_tight_list(tmp_path: Path) -> None:
    # A wrapped bullet ends on an indented continuation line; the 0.4.1b5 cut put a
    # blank line between two such fragments (a loose list).
    cl, frag = _setup(tmp_path, "# Changelog\n\n## [Unreleased]\n")
    (frag / "a.fixed.md").write_text("- first bullet\n  wraps here.\n")
    (frag / "b.fixed.md").write_text("- second bullet\n  wraps too.\n")
    assert _run(tmp_path, "--version", "1.0", "--date", "2026-02-03") == 0
    assert "  wraps here.\n- second bullet\n" in cl.read_text()


@pytest.mark.parametrize(
    ("name", "content", "needle"),
    [
        ("no-category.md", "- x\n", "name must be"),
        ("thing.bogus.md", "- x\n", "unknown category 'bogus'"),
        ("thing.added.txt", "- x\n", "name must be"),
        ("Bad Slug.added.md", "- x\n", "name must be"),
        ("empty.added.md", "  \n\n", "file is empty"),
        ("heading.added.md", "### Added\n- x\n", "contains a heading line"),
    ],
)
def test_check_rejects_bad_fragments(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, name: str, content: str, needle: str
) -> None:
    _, frag = _setup(tmp_path)
    (frag / "good.added.md").write_text("- fine\n")
    (frag / name).write_text(content)
    assert list_problems(frag) and needle in list_problems(frag)[0]
    assert _run(tmp_path, "--check") == 1
    assert needle in caplog.text


def test_check_accepts_valid_fragments_and_an_empty_directory(tmp_path: Path) -> None:
    _, frag = _setup(tmp_path)
    assert _run(tmp_path, "--check") == 0
    (frag / "ok.security.md").write_text("- patched\n")
    assert _run(tmp_path, "--check") == 0


def test_assemble_refuses_an_invalid_fragment_and_changes_nothing(tmp_path: Path) -> None:
    cl, frag = _setup(tmp_path)
    (frag / "oops.bogus.md").write_text("- x\n")
    assert _run(tmp_path, "--version", "0.2.0", "--date", "2026-02-03") == 1
    assert cl.read_text() == CHANGELOG
    assert (frag / "oops.bogus.md").exists()


def test_duplicate_version_is_refused_and_changes_nothing(tmp_path: Path) -> None:
    cl, frag = _setup(tmp_path)
    (frag / "a.added.md").write_text("- a\n")
    assert _run(tmp_path, "--version", "0.1.0", "--date", "2026-02-03") == 1
    assert cl.read_text() == CHANGELOG
    assert (frag / "a.added.md").exists()


def test_missing_unreleased_heading_is_refused(tmp_path: Path) -> None:
    cl, frag = _setup(tmp_path, "# Changelog\n\n## [0.1.0] - 2026-01-01\n")
    (frag / "a.added.md").write_text("- a\n")
    with pytest.raises(ChangelogError, match=r"no '## \[Unreleased\]'"):
        assemble(cl.read_text(), load_fragments(frag), "0.2.0", "2026-02-03")
    assert _run(tmp_path, "--version", "0.2.0", "--date", "2026-02-03") == 1


def test_no_fragments_and_empty_unreleased_is_refused(tmp_path: Path) -> None:
    original = "# Changelog\n\n## [Unreleased]\n\n## [0.1.0] - 2026-01-01\n"
    cl, _ = _setup(tmp_path, original)
    assert _run(tmp_path, "--version", "0.2.0", "--date", "2026-02-03") == 1
    assert cl.read_text() == original


def test_no_fragments_but_existing_unreleased_content_is_cut(tmp_path: Path) -> None:
    cl, _ = _setup(tmp_path)
    assert _run(tmp_path, "--version", "0.2.0", "--date", "2026-02-03") == 0
    section = cl.read_text()
    assert "## [0.2.0] - 2026-02-03\n\n### Added\n\n- Existing added bullet." in section


def test_uncategorised_or_unknown_unreleased_content_is_refused(tmp_path: Path) -> None:
    stray = "# Changelog\n\n## [Unreleased]\n\nloose text\n\n### Added\n\n- a\n"
    with pytest.raises(ChangelogError, match="belongs to no category"):
        assemble(stray, [], "0.2.0", "2026-02-03")
    odd = "# Changelog\n\n## [Unreleased]\n\n### Misc\n\n- a\n"
    with pytest.raises(ChangelogError, match="unknown category '### Misc'"):
        assemble(odd, [], "0.2.0", "2026-02-03")


def test_bad_date_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ChangelogError, match="YYYY-MM-DD"):
        assemble(CHANGELOG, [], "0.2.0", "02/03/2026")


def test_the_html_note_under_unreleased_is_preserved_and_not_content(tmp_path: Path) -> None:
    text = "# Changelog\n\n## [Unreleased]\n\n<!-- note -->\n\n### Added\n\n- a\n"
    out = assemble(text, [], "0.2.0", "2026-02-03")
    assert "## [0.2.0] - 2026-02-03\n\n### Added\n\n- a" in out
    assert "<!-- note -->" not in out.split("## [0.2.0]")[1]


def test_repo_fragments_are_valid_and_unreleased_section_parses() -> None:
    """The real tree passes --check and its Unreleased section has only known categories."""
    root = Path(__file__).resolve().parents[2]
    assert list_problems(root / "changelog.d") == []
    _split_unreleased((root / "CHANGELOG.md").read_text(encoding="utf-8"))
