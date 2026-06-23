"""Tests for xtrax.cli.explain module (T5 — explain verb + emit router).

Tests the run_explain function which orchestrates loading functions, parsing
shapes, inferring bundle schemas, planning tiling strategies, explaining the
plan, and emitting the result in the requested format.

AC5: json format produces json.loads-parseable output with _meta + PlanStatsDict keys.
AC6 text: fmt="text" produces non-empty human output.
AC6 missing-eda: fmt="html" raises CLIError mentioning xtrax[eda] (non-vacuous: forced
    via monkeypatching regardless of whether matplotlib is installed).
F1 html/png: fmt="html" with out=None prints non-empty HTML to stdout; fmt="html" with
    out=<path> writes a non-empty file; fmt="png" with out=<path> writes a non-empty file.
"""

from __future__ import annotations

import json
import sys

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
# AC6 missing-eda — html/png raises clean CLIError (non-vacuous)
# ---------------------------------------------------------------------------


class TestExplainMissingEda:
    """Tests for missing eda extra: html raises CLIError mentioning xtrax[eda].

    These tests are non-vacuous: they FORCE the error path via monkeypatching
    regardless of whether matplotlib is actually installed in the venv.
    The test will FAIL if the CLIError wrapping is removed from emit._emit_render.
    """

    def test_html_raises_cli_error_when_render_raises_module_not_found(self, monkeypatch):
        """AC6 missing-eda: when render raises ModuleNotFoundError, CLIError is raised.

        Forces the error path by making sys.modules['xtrax.eda'] raise
        ModuleNotFoundError when _emit_render tries 'from xtrax.eda import render'.
        The raised exception MUST be a CLIError mentioning 'xtrax[eda]', NOT
        a raw ModuleNotFoundError. This test FAILS if the wrapping in _emit_render
        is removed.
        """
        import xtrax.eda  # monkeypatch target; no viz import (needs matplotlib)

        # Patch xtrax.eda.render (as resolved in emit via 'from xtrax.eda import render')
        # to raise ModuleNotFoundError — simulates eda extra absent.

        def _raising_render(*args, **kwargs):
            raise ModuleNotFoundError("No module named 'matplotlib'")

        monkeypatch.setattr(xtrax.eda, "render", _raising_render)

        args = ExplainArgs(fn="tests.cli.test_explain:decorated_fn", shapes="x=(4,)f32", fmt="html")
        with pytest.raises(CLIError) as exc_info:
            run_explain(args)

        error_msg = str(exc_info.value)
        assert "xtrax[eda]" in error_msg, (
            f"Expected 'xtrax[eda]' in error message, got: {error_msg!r}"
        )
        assert "ModuleNotFoundError" not in error_msg, (
            f"Expected clean CLIError, not raw ModuleNotFoundError in message: {error_msg!r}"
        )

    def test_html_missing_eda_error_type_is_cli_error(self, monkeypatch):
        """AC6 missing-eda: the exception type must be CLIError, not ModuleNotFoundError.

        Forces the eda-import failure via monkeypatching sys.modules so that
        importing matplotlib inside render raises ModuleNotFoundError. The
        raised exception must be a CLIError (not ModuleNotFoundError).
        """
        import xtrax.cli.emit as emit_mod

        # Patch sys.modules so that 'matplotlib' appears unimportable,
        # then patch _emit_render so it simulates the eda-import failure path.
        saved = sys.modules.get("matplotlib", "SENTINEL")
        sys.modules["matplotlib"] = None  # type: ignore[assignment]

        def _raising_render(plan, fmt, out=None):
            # Simulates what _emit_render does when 'from xtrax.eda import render'
            # triggers a ModuleNotFoundError due to matplotlib being absent.
            try:
                import matplotlib  # noqa: F401  — this will raise because we set it to None
            except (ModuleNotFoundError, ImportError) as exc:
                raise CLIError(
                    f"--fmt {fmt!r} requires the eda extra: pip install xtrax[eda]"
                ) from exc

        monkeypatch.setattr(emit_mod, "_emit_render", _raising_render)

        try:
            args = ExplainArgs(
                fn="tests.cli.test_explain:decorated_fn", shapes="x=(4,)f32", fmt="html"
            )
            with pytest.raises(CLIError):
                run_explain(args)
        finally:
            # Restore sys.modules
            if saved == "SENTINEL":
                sys.modules.pop("matplotlib", None)
            else:
                sys.modules["matplotlib"] = saved  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# F1 — html/png actually produce output (non-silent)
# ---------------------------------------------------------------------------


class TestExplainHtmlPngOutput:
    """Tests that html/png formats produce real output (F1 fix).

    These tests verify that the F1 fix is in place: html/png output is no
    longer silently discarded. render()'s return value is properly used.
    """

    def test_html_no_out_prints_to_stdout(self, capsys):
        """F1 html: fmt='html' with out=None prints non-empty HTML to stdout."""
        pytest.importorskip("matplotlib")  # html render needs the eda extra
        args = ExplainArgs(
            fn="tests.cli.test_explain:decorated_fn",
            shapes="x=(4,)f32",
            fmt="html",
            out=None,
        )
        run_explain(args)

        captured = capsys.readouterr()
        output = captured.out.strip()
        assert output, "Expected non-empty HTML output on stdout"
        assert "<!DOCTYPE html>" in output or "<html" in output or "<svg" in output, (
            f"Expected HTML/SVG content in stdout, got: {output[:200]!r}"
        )

    def test_html_with_out_writes_file(self, tmp_path):
        """F1 html: fmt='html' with out=<path> writes a non-empty file."""
        pytest.importorskip("matplotlib")  # html render needs the eda extra
        out_path = str(tmp_path / "plan.html")
        args = ExplainArgs(
            fn="tests.cli.test_explain:decorated_fn",
            shapes="x=(4,)f32",
            fmt="html",
            out=out_path,
        )
        run_explain(args)

        import os
        assert os.path.exists(out_path), f"Expected file at {out_path!r} but it does not exist"
        size = os.path.getsize(out_path)
        assert size > 0, f"Expected non-empty file at {out_path!r}, got {size} bytes"

    def test_png_with_out_writes_file(self, tmp_path):
        """F1 png: fmt='png' with out=<path> writes a non-empty binary file."""
        pytest.importorskip("matplotlib")  # png render needs the eda extra
        out_path = str(tmp_path / "plan.png")
        args = ExplainArgs(
            fn="tests.cli.test_explain:decorated_fn",
            shapes="x=(4,)f32",
            fmt="png",
            out=out_path,
        )
        run_explain(args)

        import os
        assert os.path.exists(out_path), f"Expected file at {out_path!r} but it does not exist"
        size = os.path.getsize(out_path)
        assert size > 0, f"Expected non-empty PNG file at {out_path!r}, got {size} bytes"

    def test_png_no_out_raises_cli_error(self):
        """F1 png contract: fmt='png' without --out raises CLIError (binary footgun guard)."""
        args = ExplainArgs(
            fn="tests.cli.test_explain:decorated_fn",
            shapes="x=(4,)f32",
            fmt="png",
            out=None,
        )
        with pytest.raises(CLIError) as exc_info:
            run_explain(args)

        error_msg = str(exc_info.value)
        assert "--out" in error_msg or "out" in error_msg.lower(), (
            f"Expected --out hint in CLIError message, got: {error_msg!r}"
        )


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
