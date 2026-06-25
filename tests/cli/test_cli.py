"""Integration tests for the installed xtrax console script (T8).

Tests validate end-to-end behavior of the installed `xtrax` CLI entry point
via subprocess, covering AC1 (help), AC5/AC8 (explain end-to-end), and
AC4 (fail-loud for undecorated functions).

These tests use `uv run xtrax` to exercise the real installed console script
as registered in [project.scripts], validating AC2 packaging.
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest

from xtrax.inference.config import AxisOverride, axis_config

# ---------------------------------------------------------------------------
# Module-level fixture function (used as --fn import target)
# ---------------------------------------------------------------------------

WORKTREE_ROOT = str(__import__("pathlib").Path(__file__).parent.parent.parent.resolve())


@axis_config(AxisOverride(name="batch", default_batch_size=2))
def decorated_fn(x):
    """A function decorated with axis_config for CLI integration testing."""
    return x * 2


def bare_fn(x):
    """An undecorated function — should cause @axis_config hint error."""
    return x * 2


# ---------------------------------------------------------------------------
# Test A — AC1: installed script help
# ---------------------------------------------------------------------------


class TestInstalledHelp:
    """AC1: `uv run xtrax --help` exits 0 and lists plan + explain."""

    def test_help_exits_zero(self):
        """AC1: `uv run xtrax --help` must exit with code 0."""
        result = subprocess.run(
            ["uv", "run", "xtrax", "--help"],
            capture_output=True,
            text=True,
            cwd=WORKTREE_ROOT,
        )
        assert result.returncode == 0, (
            f"Expected returncode 0, got {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_help_lists_plan(self):
        """AC1: --help output must include the 'plan' verb."""
        result = subprocess.run(
            ["uv", "run", "xtrax", "--help"],
            capture_output=True,
            text=True,
            cwd=WORKTREE_ROOT,
        )
        output = result.stdout + result.stderr
        assert "plan" in output, f"Expected 'plan' in --help output.\nFull output:\n{output}"

    def test_help_lists_explain(self):
        """AC1: --help output must include the 'explain' verb."""
        result = subprocess.run(
            ["uv", "run", "xtrax", "--help"],
            capture_output=True,
            text=True,
            cwd=WORKTREE_ROOT,
        )
        output = result.stdout + result.stderr
        assert "explain" in output, f"Expected 'explain' in --help output.\nFull output:\n{output}"

    def test_help_no_mangled_names(self):
        """AC1: --help output must NOT list 'plan-args' or 'explain-args'."""
        result = subprocess.run(
            ["uv", "run", "xtrax", "--help"],
            capture_output=True,
            text=True,
            cwd=WORKTREE_ROOT,
        )
        output = result.stdout + result.stderr
        assert "plan-args" not in output, (
            f"Mangled name 'plan-args' appeared in --help output.\n{output}"
        )
        assert "explain-args" not in output, (
            f"Mangled name 'explain-args' appeared in --help output.\n{output}"
        )


# ---------------------------------------------------------------------------
# Test B — AC5/AC8: installed script explain end-to-end
# ---------------------------------------------------------------------------


class TestInstalledExplain:
    """AC5/AC8: `uv run xtrax explain` produces valid JSON with _meta and plan stats."""

    def _run_explain(self, fn_path: str) -> subprocess.CompletedProcess:
        env = {**os.environ, "PYTHONPATH": WORKTREE_ROOT}
        return subprocess.run(
            [
                "uv",
                "run",
                "xtrax",
                "explain",
                "--fn",
                fn_path,
                "--shapes",
                "x=(4,)f32",
                "--fmt",
                "json",
            ],
            capture_output=True,
            text=True,
            cwd=WORKTREE_ROOT,
            env=env,
        )

    def test_explain_exits_zero(self):
        """AC5: explain with decorated_fn exits 0."""
        result = self._run_explain("tests.cli.test_cli:decorated_fn")
        assert result.returncode == 0, (
            f"Expected returncode 0, got {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_explain_stdout_is_valid_json(self):
        """AC5/AC8: explain stdout is parseable JSON."""
        result = self._run_explain("tests.cli.test_cli:decorated_fn")
        assert result.returncode == 0, (
            f"Subprocess failed with code {result.returncode}.\nstderr: {result.stderr}"
        )
        try:
            parsed = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            pytest.fail(f"stdout is not valid JSON: {e}\nstdout was:\n{result.stdout}")
        assert isinstance(parsed, dict), "Expected JSON object at top level"

    def test_explain_has_meta_key(self):
        """AC5/AC8: JSON output has '_meta' envelope key."""
        result = self._run_explain("tests.cli.test_cli:decorated_fn")
        assert result.returncode == 0
        parsed = json.loads(result.stdout)
        assert "_meta" in parsed, (
            f"Expected '_meta' key in JSON output. Got keys: {list(parsed.keys())}"
        )

    def test_explain_has_plan_stats_keys(self):
        """AC5/AC8: JSON output has all PlanStatsDict top-level keys."""
        result = self._run_explain("tests.cli.test_cli:decorated_fn")
        assert result.returncode == 0
        parsed = json.loads(result.stdout)
        expected_keys = {
            "axes",
            "strategy_counts",
            "total_axes",
            "memory_warnings",
            "dedup_stats",
            "bucket_stats",
        }
        missing = expected_keys - set(parsed.keys())
        assert not missing, (
            f"Missing PlanStatsDict keys: {missing}. Got keys: {list(parsed.keys())}"
        )


# ---------------------------------------------------------------------------
# Test C — AC4: fail-loud for undecorated function
# ---------------------------------------------------------------------------


class TestInstalledFailLoud:
    """AC4: undecorated functions produce a clean error mentioning @axis_config."""

    def test_bare_fn_nonzero_exit(self):
        """AC4: calling explain with a bare fn must exit nonzero."""
        env = {**os.environ, "PYTHONPATH": WORKTREE_ROOT}
        result = subprocess.run(
            [
                "uv",
                "run",
                "xtrax",
                "explain",
                "--fn",
                "tests.cli.test_cli:bare_fn",
                "--shapes",
                "x=(4,)f32",
                "--fmt",
                "json",
            ],
            capture_output=True,
            text=True,
            cwd=WORKTREE_ROOT,
            env=env,
        )
        assert result.returncode != 0, (
            f"Expected nonzero returncode for bare_fn, got {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_bare_fn_stderr_mentions_axis_config(self):
        """AC4: error output mentions @axis_config."""
        env = {**os.environ, "PYTHONPATH": WORKTREE_ROOT}
        result = subprocess.run(
            [
                "uv",
                "run",
                "xtrax",
                "explain",
                "--fn",
                "tests.cli.test_cli:bare_fn",
                "--shapes",
                "x=(4,)f32",
                "--fmt",
                "json",
            ],
            capture_output=True,
            text=True,
            cwd=WORKTREE_ROOT,
            env=env,
        )
        assert "axis_config" in result.stderr, (
            f"Expected '@axis_config' hint in stderr.\nstderr: {result.stderr}"
        )

    def test_bare_fn_no_traceback_in_stderr(self):
        """AC4: stderr must NOT contain a Python traceback (clean user-facing error)."""
        env = {**os.environ, "PYTHONPATH": WORKTREE_ROOT}
        result = subprocess.run(
            [
                "uv",
                "run",
                "xtrax",
                "explain",
                "--fn",
                "tests.cli.test_cli:bare_fn",
                "--shapes",
                "x=(4,)f32",
                "--fmt",
                "json",
            ],
            capture_output=True,
            text=True,
            cwd=WORKTREE_ROOT,
            env=env,
        )
        assert "Traceback" not in result.stderr, (
            f"Python traceback appeared in stderr — error is not clean.\nstderr: {result.stderr}"
        )
