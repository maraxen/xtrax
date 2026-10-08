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


def test_added_types_diff_ignores_nested_def_in_private_function() -> None:
    """AC-36: a public-named def nested inside a private function is not collected."""
    head = textwrap.dedent(
        """
        def build(x: int) -> int:
            return x

        def _topo(items):
            def visit(n):
                return n

            return visit
        """
    )
    touched = diff_callables_to_audit(base_source=None, head_source=head)
    violations = audit_changed_callables(
        head_source=head,
        rel_path="pkg/sample.py",
        qualnames=touched,
    )
    assert touched == {"build"}
    assert violations == []


def test_added_types_diff_ignores_closures_in_function_and_method() -> None:
    """AC-37: public-named closures inside a public fn and a public method are ignored."""
    head = textwrap.dedent(
        """
        def build(x: int) -> int:
            def inner(y: int) -> int:
                return y

            return inner(x)

        class Cls:
            def meth(self, x: int) -> int:
                def inner(y: int) -> int:
                    return y

                return inner(x)
        """
    )
    touched = diff_callables_to_audit(base_source=None, head_source=head)
    violations = audit_changed_callables(
        head_source=head,
        rel_path="pkg/sample.py",
        qualnames=touched,
    )
    assert touched == {"build", "Cls.meth"}
    assert violations == []


def test_added_types_diff_nested_only_change_does_not_flag_top_level() -> None:
    """AC-38: a signature change confined to a nested `helper` must not flag the

    top-level `helper` of the same name (the dict-overwrite collision, direction 1).
    """
    base = textwrap.dedent(
        """
        def helper(x: int) -> int:
            return x

        def outer() -> int:
            def helper(y: int) -> int:
                return y

            return helper(1)
        """
    )
    head = textwrap.dedent(
        """
        def helper(x: int) -> int:
            return x

        def outer() -> int:
            def helper(y: int, z: int) -> int:
                return y + z

            return helper(1, 2)
        """
    )
    touched = diff_callables_to_audit(base_source=base, head_source=head)
    assert touched == set()


def test_added_types_diff_top_level_only_change_is_detected() -> None:
    """AC-39 (OBJ-R1-11): a real signature change confined to the top-level `helper`

    must be flagged even though a same-named nested `helper` also exists. Pre-fix
    this is silent: both base and head collector maps hold the (unchanged) nested
    def under the same dict key, so the comparison sees no change at all.
    """
    base = textwrap.dedent(
        """
        def helper(x: int) -> int:
            return x

        def outer() -> int:
            def helper(y: int) -> int:
                return y

            return helper(1)
        """
    )
    head = textwrap.dedent(
        """
        def helper(x: int, y: int) -> int:
            return x + y

        def outer() -> int:
            def helper(y: int) -> int:
                return y

            return helper(1)
        """
    )
    touched = diff_callables_to_audit(base_source=base, head_source=head)
    assert touched == {"helper"}


def test_run_added_types_diff_gate_still_audits_real_changes(tmp_path: Path) -> None:
    """AC-40: control proving the fix does not pass by collecting nothing — a real

    changed+untyped top-level function and a real changed+untyped method are still
    audited, alongside the same nested-`helper` shape used in AC-38/AC-39.
    """
    repo = _init_repo(tmp_path)
    pkg = repo / "src" / "xtrax"
    _write(
        pkg / "sample.py",
        """
        def helper(x: int) -> int:
            return x

        def outer() -> int:
            def helper(y: int) -> int:
                return y

            return helper(1)

        class Cls:
            def worse(self, x: int) -> int:
                return x
        """,
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")

    _write(
        pkg / "sample.py",
        """
        def helper(x: int, y: int) -> int:
            return x + y

        def outer() -> int:
            def helper(y: int) -> int:
                return y

            return helper(1)

        def bad(x):
            return x

        class Cls:
            def worse(self, y):
                return y
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
    violation_text = "\n".join(result.violations)
    assert "bad" in violation_text
    assert "Cls.worse" in violation_text
    assert "unable to locate" not in violation_text
    assert "outer" not in violation_text


def test_added_types_diff_ignores_class_defined_inside_function() -> None:
    """AC-41: a class (and its methods) defined inside a public function is not collected."""
    head = textwrap.dedent(
        """
        def build(x: int) -> int:
            class Local:
                def m(self, y: int) -> int:
                    return y

            return Local().m(x)
        """
    )
    touched = diff_callables_to_audit(base_source=None, head_source=head)
    violations = audit_changed_callables(
        head_source=head,
        rel_path="pkg/sample.py",
        qualnames=touched,
    )
    assert touched == {"build"}
    assert violations == []


def test_added_types_diff_ignores_factory_pattern_nested_def() -> None:
    """AC-42 (accepted residual, spec section 6): a factory-built public API —

    `foo = _make_foo()` with a nested `def foo` — is invisible to the gate. This
    is asserted explicitly so the residual stays visible rather than silently
    re-appearing as a lookup failure.
    """
    head = textwrap.dedent(
        """
        def _make_foo():
            def foo(x: int) -> int:
                return x

            return foo

        foo = _make_foo()
        """
    )
    touched = diff_callables_to_audit(base_source=None, head_source=head)
    assert touched == set()


def test_added_types_diff_sweep_src_xtrax_has_no_unlocatable_callables() -> None:
    """AC-43: sweep every *.py under src/xtrax, mirroring the gate's own

    `__init__.py` skip (added_types_diff.py:164). Zero unresolvable qualnames
    post-fix; the pre-fix count is recorded in the fix commit, not asserted here.
    """
    src_root = ROOT / "src" / "xtrax"
    unresolvable: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        rel = path.relative_to(ROOT).as_posix()
        head_source = path.read_text(encoding="utf-8")
        touched = diff_callables_to_audit(base_source=None, head_source=head_source)
        violations = audit_changed_callables(
            head_source=head_source,
            rel_path=rel,
            qualnames=touched,
        )
        unresolvable.extend(v for v in violations if "unable to locate callable" in v)
    assert unresolvable == []


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
