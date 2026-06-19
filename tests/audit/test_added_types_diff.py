"""Tests for D3' added-types LibCST diff gate (#1589)."""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

from xtrax.devtools.gates.added_types_diff import (
    audit_changed_callables,
    diff_callables_to_audit,
    resolve_merge_base,
    run_added_types_diff_gate,
)
from xtrax.devtools.gates.type_hardening import inspect_public_callable

ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source).strip() + "\n", encoding="utf-8")


def test_inspect_public_callable_flags_missing_annotations() -> None:
    import ast

    tree = ast.parse(
        textwrap.dedent(
            """
            def bad(x, y: int):
                return x
            """
        )
    )
    fn = tree.body[0]
    assert isinstance(fn, ast.FunctionDef)
    violations = inspect_public_callable(fn, qualname="bad")
    assert any("parameter `x` missing annotation" in item for item in violations)
    assert any("missing return annotation" in item for item in violations)


def test_diff_callables_to_audit_new_file_requires_all_public() -> None:
    head = textwrap.dedent(
        """
        def typed(x: int) -> int:
            return x

        def untyped(y):
            return y
        """
    )
    touched = diff_callables_to_audit(base_source=None, head_source=head)
    assert touched == {"typed", "untyped"}


def test_diff_callables_to_audit_detects_signature_change() -> None:
    base = textwrap.dedent(
        """
        def fn(x: int) -> int:
            return x
        """
    )
    head = textwrap.dedent(
        """
        def fn(x: int, y) -> int:
            return x
        """
    )
    touched = diff_callables_to_audit(base_source=base, head_source=head)
    assert touched == {"fn"}


def test_diff_callables_to_audit_ignores_body_only_change() -> None:
    base = textwrap.dedent(
        """
        def fn(x: int) -> int:
            return x
        """
    )
    head = textwrap.dedent(
        """
        def fn(x: int) -> int:
            return x + 1
        """
    )
    touched = diff_callables_to_audit(base_source=base, head_source=head)
    assert touched == set()


def test_audit_changed_callables_reports_violations() -> None:
    head = textwrap.dedent(
        """
        def untyped(x, y: int):
            return x
        """
    )
    violations = audit_changed_callables(
        head_source=head,
        rel_path="pkg/sample.py",
        qualnames={"untyped"},
    )
    assert violations
    assert any("missing annotation" in item for item in violations)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    return repo


def test_run_added_types_diff_gate_passes_clean_diff(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    pkg = repo / "src" / "xtrax"
    _write(
        pkg / "typed.py",
        """
        def f(x: int) -> int:
            return x
        """,
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")

    _write(
        pkg / "typed.py",
        """
        def f(x: int) -> int:
            return x + 1
        """,
    )
    _write(
        pkg / "new_fn.py",
        """
        def g(x: int) -> int:
            return x
        """,
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "head")

    base = _git(repo, "rev-parse", "HEAD~1").stdout.strip()
    result = run_added_types_diff_gate(
        repo,
        target=Path("src/xtrax"),
        merge_base=base,
    )
    assert result.status == "pass"
    assert result.callables_checked == 1
    assert result.violations == ()


def test_run_added_types_diff_gate_fails_on_new_untyped(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    pkg = repo / "src" / "xtrax"
    _write(
        pkg / "base.py",
        """
        def keep(x: int) -> int:
            return x
        """,
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")

    _write(
        pkg / "bad.py",
        """
        def new_untyped(x, y: int):
            return x
        """,
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "head")

    base = _git(repo, "rev-parse", "HEAD~1").stdout.strip()
    result = run_added_types_diff_gate(
        repo,
        target=Path("src/xtrax"),
        merge_base=base,
    )
    assert result.status == "fail"
    assert any("new_untyped" in item for item in result.violations)


def test_run_added_types_diff_gate_skips_without_merge_base(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    result = run_added_types_diff_gate(
        repo,
        target=Path("src/xtrax"),
        merge_base=None,
    )
    assert result.status == "skip"
    assert result.skip_reason is not None


def test_resolve_merge_base_on_initialized_repo(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write(repo / "README.md", "hi\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    _write(repo / "README.md", "hello\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "second")
    base = resolve_merge_base(repo)
    assert base is not None


def test_audit_added_types_diff_cli_passes_on_repo(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    pkg = repo / "src" / "xtrax"
    _write(
        pkg / "typed.py",
        """
        def f(x: int) -> int:
            return x
        """,
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "only")
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(ROOT / "scripts" / "audit_added_types_diff.py"),
            "--repo-root",
            str(repo),
            "--base",
            _git(repo, "rev-parse", "HEAD").stdout.strip(),
            "--no-emit",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "PASS" in result.stdout
