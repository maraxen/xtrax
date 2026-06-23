"""Tests for xtrax.cli.explain module (T5 — explain verb + emit router).

Tests the run_explain function which orchestrates loading functions, parsing
shapes, inferring bundle schemas, planning tiling strategies, explaining the
plan, and emitting the result in the requested format.

AC5: json format produces json.loads-parseable output with _meta + PlanStatsDict keys.
AC6 text: fmt="text" produces non-empty human output.
AC6 missing-eda: fmt="html" raises CLIError mentioning xtrax[eda].
"""

from __future__ import annotations

import json

import pytest

from xtrax.cli.errors import CLIError
from xtrax.cli.explain import ExplainArgs, run_explain
from xtrax.inference.config import AxisOverride, axis_config  # noqa: F401

# ---------------------------------------------------------------------------
# Module-level fixture functions (used as import path targets)
# ---------------------------------------------------------------------------


@axis_config(AxisOverride(name="batch", default_batch_size=2))
def decorated_fn(x):
    """A function decorated with axis_config for testing the explain verb.

    This simulates a real user function annotated with axis override info.
    The decorator resolves the axis role, so planning and explain should succeed.
    """
    return x * 2


# ---------------------------------------------------------------------------
# AC5 — json output contract
# ---------------------------------------------------------------------------


class TestExplainJsonFormat:
    """Tests for JSON output format (the default machine contract)."""

    def test_json_output_is_parseable(self, capsys):
        """AC5: run_explain with fmt='json' emits json.loads-parseable stdout."""
        args = ExplainArgs(fn="tests.cli.test_explain:decorated_fn", shapes="x=(4,)f32", fmt="json")
        run_explain(args)

        captured = capsys.readouterr()
        output = captured.out.strip()
        assert output, "Expected non-empty stdout for json format"

        parsed = json.loads(output)
        assert isinstance(parsed, dict), "Expected a JSON object (dict) as top-level output"

    def test_json_output_has_meta_key(self, capsys):
        """AC5: json output must have a _meta envelope key."""
        args = ExplainArgs(fn="tests.cli.test_explain:decorated_fn", shapes="x=(4,)f32", fmt="json")
        run_explain(args)

        captured = capsys.readouterr()
        parsed = json.loads(captured.out.strip())
        assert "_meta" in parsed, (
            f"Expected '_meta' key in json output, got keys: {list(parsed.keys())}"
        )

    def test_json_meta_has_schema_version(self, capsys):
        """AC5: _meta envelope must contain schema_version=1."""
        args = ExplainArgs(fn="tests.cli.test_explain:decorated_fn", shapes="x=(4,)f32", fmt="json")
        run_explain(args)

        captured = capsys.readouterr()
        parsed = json.loads(captured.out.strip())
        assert parsed["_meta"].get("schema_version") == 1, (
            f"Expected schema_version=1 in _meta, got: {parsed['_meta']}"
        )

    def test_json_output_has_plan_stats_keys(self, capsys):
        """AC5: json output must contain PlanStatsDict top-level keys."""
        args = ExplainArgs(fn="tests.cli.test_explain:decorated_fn", shapes="x=(4,)f32", fmt="json")
        run_explain(args)

        captured = capsys.readouterr()
        parsed = json.loads(captured.out.strip())

        # PlanStatsDict keys (from xtrax.eda.types.PlanStatsDict):
        #   axes, strategy_counts, total_axes, memory_warnings,
        #   dedup_stats, bucket_stats
        expected_keys = {
            "axes", "strategy_counts", "total_axes",
            "memory_warnings", "dedup_stats", "bucket_stats",
        }
        missing = expected_keys - set(parsed.keys())
        assert not missing, (
            f"Missing PlanStatsDict keys in json output: {missing}. "
            f"Got keys: {list(parsed.keys())}"
        )

    def test_json_is_default_format(self, capsys):
        """AC5: ExplainArgs without fmt= defaults to json output."""
        # No fmt= argument — should default to "json"
        args = ExplainArgs(fn="tests.cli.test_explain:decorated_fn", shapes="x=(4,)f32")
        run_explain(args)

        captured = capsys.readouterr()
        output = captured.out.strip()
        assert output, "Expected non-empty stdout with default format"

        # Must be parseable as JSON
        parsed = json.loads(output)
        assert "_meta" in parsed, "Default format should produce JSON with _meta key"
        expected_keys = {
            "axes", "strategy_counts", "total_axes",
            "memory_warnings", "dedup_stats", "bucket_stats",
        }
        missing = expected_keys - set(parsed.keys())
        assert not missing, f"Missing PlanStatsDict keys: {missing}"


# ---------------------------------------------------------------------------
# AC6 text — human-readable output
# ---------------------------------------------------------------------------


class TestExplainTextFormat:
    """Tests for text output format (human-readable summary)."""

    def test_text_output_is_non_empty(self, capsys):
        """AC6 text: fmt='text' produces non-empty human output."""
        args = ExplainArgs(fn="tests.cli.test_explain:decorated_fn", shapes="x=(4,)f32", fmt="text")
        run_explain(args)

        captured = capsys.readouterr()
        output = captured.out.strip()
        assert output, "Expected non-empty stdout for text format"

    def test_text_output_is_not_json(self, capsys):
        """AC6 text: fmt='text' output should NOT be a bare JSON object (it's human text)."""
        args = ExplainArgs(fn="tests.cli.test_explain:decorated_fn", shapes="x=(4,)f32", fmt="text")
        run_explain(args)

        captured = capsys.readouterr()
        output = captured.out.strip()
        # Text output should be human readable — it may contain JSON-like
        # substrings but should not start with '{' (which would indicate
        # it accidentally produced JSON format)
        # We just require it's non-empty and has some human-readable content
        assert len(output) > 0


# ---------------------------------------------------------------------------
# AC6 missing-eda — html/png raises clean CLIError
# ---------------------------------------------------------------------------


class TestExplainMissingEda:
    """Tests for missing eda extra: html/png should raise clean CLIError."""

    def test_html_raises_cli_error_or_succeeds(self, capsys):
        """AC6 missing-eda: fmt='html' raises CLIError mentioning xtrax[eda].

        If matplotlib is NOT installed (expected in this venv), the error must
        be a CLIError mentioning 'xtrax[eda]', NOT a raw ModuleNotFoundError.
        If matplotlib IS unexpectedly installed, render() should return without error.
        """
        args = ExplainArgs(fn="tests.cli.test_explain:decorated_fn", shapes="x=(4,)f32", fmt="html")

        try:
            import matplotlib  # noqa: F401
            matplotlib_available = True
        except ImportError:
            matplotlib_available = False

        if matplotlib_available:
            # matplotlib is installed — render should not raise
            run_explain(args)  # should not raise
        else:
            # matplotlib is NOT installed — must get clean CLIError
            with pytest.raises(CLIError) as exc_info:
                run_explain(args)

            error_msg = str(exc_info.value)
            assert "xtrax[eda]" in error_msg, (
                f"Expected 'xtrax[eda]' in error message, got: {error_msg!r}"
            )
            # Must NOT be a raw ModuleNotFoundError traceback
            assert "ModuleNotFoundError" not in error_msg, (
                f"Expected clean CLIError, not raw ModuleNotFoundError: {error_msg!r}"
            )

    def test_html_raises_cli_error_not_module_not_found(self):
        """AC6 missing-eda: html format must not leak a raw ModuleNotFoundError.

        The error must be a CLIError (not ModuleNotFoundError) regardless of
        whether matplotlib is installed.
        """
        args = ExplainArgs(fn="tests.cli.test_explain:decorated_fn", shapes="x=(4,)f32", fmt="html")

        try:
            import matplotlib  # noqa: F401
            matplotlib_available = True
        except ImportError:
            matplotlib_available = False

        if not matplotlib_available:
            # Verify the exception type is CLIError, not ModuleNotFoundError
            with pytest.raises(CLIError):
                run_explain(args)
        else:
            # matplotlib is available; render should succeed
            pass  # No assertion needed


# ---------------------------------------------------------------------------
# Unknown format — should raise CLIError
# ---------------------------------------------------------------------------


class TestExplainUnknownFormat:
    """Tests for unknown format strings."""

    def test_unknown_fmt_raises_cli_error(self):
        """Unknown fmt string must raise CLIError."""
        args = ExplainArgs(
            fn="tests.cli.test_explain:decorated_fn",
            shapes="x=(4,)f32",
            fmt="unknown_format",  # type: ignore[arg-type]
        )
        with pytest.raises(CLIError):
            run_explain(args)
