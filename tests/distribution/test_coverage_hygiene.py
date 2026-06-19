"""Tests for distribution N0 coverage hygiene gate (#1451)."""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

from scripts.audit_coverage_hygiene import (
    audit_coverage_hygiene,
    check_gitignore_patterns,
    find_forbidden_tracked_paths,
    gitignore_contains_pattern,
    load_hygiene_config,
    path_matches_forbidden_glob,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "distribution" / "coverage_hygiene.toml"


def _init_git_repo(repo_root: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )


def _write_config(repo_root: Path) -> Path:
    config_dir = repo_root / "distribution"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "coverage_hygiene.toml"
    config_path.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return config_path


def test_load_hygiene_config_reads_committed_toml() -> None:
    config = load_hygiene_config(CONFIG_PATH)
    assert config.version == "0.1.0"
    assert ".coverage" in config.required_gitignore_patterns
    assert "coverage.xml" in config.required_gitignore_patterns
    assert ".coverage_html/" in config.required_gitignore_patterns
    assert ".coverage_html/*" in config.forbidden_tracked_globs


@pytest.mark.parametrize(
    ("line", "pattern", "expected"),
    [
        (".coverage", ".coverage", True),
        ("**/.coverage", ".coverage", True),
        ("foo/.coverage", ".coverage", True),
        ("coverage.xml", "coverage.xml", True),
        (".coverage_html/", ".coverage_html/", True),
        ("htmlcov/", ".coverage", False),
    ],
)
def test_gitignore_contains_pattern(line: str, pattern: str, expected: bool) -> None:
    assert gitignore_contains_pattern([line], pattern) is expected


def test_check_gitignore_patterns_reports_missing_entries(tmp_path: Path) -> None:
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text(".coverage\n", encoding="utf-8")
    missing = check_gitignore_patterns(
        gitignore,
        (".coverage", "coverage.xml", ".coverage_html/"),
    )
    assert missing == ["coverage.xml", ".coverage_html/"]


@pytest.mark.parametrize(
    ("path", "glob", "expected"),
    [
        (".coverage", ".coverage", True),
        ("subdir/.coverage", ".coverage", True),
        ("coverage.xml", "coverage.xml", True),
        ("reports/coverage.xml", "coverage.xml", True),
        (".coverage_html/index.html", ".coverage_html/*", True),
        (".coverage_html", ".coverage_html/*", True),
        ("src/module.py", ".coverage", False),
    ],
)
def test_path_matches_forbidden_glob(path: str, glob: str, expected: bool) -> None:
    assert path_matches_forbidden_glob(path, glob) is expected


def test_find_forbidden_tracked_paths_collects_violations() -> None:
    tracked = [
        "src/xtrax/__init__.py",
        ".coverage",
        "coverage.xml",
        ".coverage_html/index.html",
    ]
    violations = find_forbidden_tracked_paths(
        tracked,
        (".coverage", "coverage.xml", ".coverage_html/*"),
    )
    assert violations == [".coverage", ".coverage_html/index.html", "coverage.xml"]


def test_audit_coverage_hygiene_passes_clean_tmp_repo(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / ".gitignore").write_text(
        textwrap.dedent(
            """
            .coverage
            coverage.xml
            .coverage_html/
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# ok\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    config_path = _write_config(tmp_path)
    passed, failures = audit_coverage_hygiene(
        root=tmp_path,
        config_path=config_path,
    )
    assert passed is True
    assert failures == []


def test_audit_coverage_hygiene_fails_on_tracked_coverage_artifact(
    tmp_path: Path,
) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / ".gitignore").write_text(
        ".coverage\ncoverage.xml\n.coverage_html/\n",
        encoding="utf-8",
    )
    (tmp_path / ".coverage").write_text("stale\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", ".gitignore"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "add", "-f", ".coverage"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "bad coverage"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    config_path = _write_config(tmp_path)
    passed, failures = audit_coverage_hygiene(
        root=tmp_path,
        config_path=config_path,
    )
    assert passed is False
    assert any("tracked coverage artifact: .coverage" in item for item in failures)


def test_audit_coverage_hygiene_fails_on_missing_gitignore_pattern(
    tmp_path: Path,
) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / ".gitignore").write_text(".coverage\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# ok\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    config_path = _write_config(tmp_path)
    passed, failures = audit_coverage_hygiene(
        root=tmp_path,
        config_path=config_path,
    )
    assert passed is False
    assert any(
        "missing required coverage pattern: 'coverage.xml'" in item for item in failures
    )
    assert any(
        "missing required coverage pattern: '.coverage_html/'" in item
        for item in failures
    )


def test_script_subprocess_on_repo_exits_zero() -> None:
    result = subprocess.run(
        ["uv", "run", "python", "scripts/audit_coverage_hygiene.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "PASS: coverage hygiene" in result.stdout
