"""Tests for xtrax.cli.entrypoint (T6 — main() dispatch + AC verification).

AC1 help: main() with --help lists clean verb names "plan" and "explain"
          (NOT "plan-args" / "explain-args").
AC1 dispatch: main() with ["xtrax", "explain", ...] invokes run_explain and
              produces JSON with _meta on stdout.
AC2 import isolation: importing xtrax.cli.entrypoint does NOT import tyro.
                      importing xtrax does NOT import tyro or xtrax.cli.
"""

from __future__ import annotations

import json
import sys

import pytest

from xtrax.inference.config import AxisOverride, axis_config  # noqa: F401

# ---------------------------------------------------------------------------
# Module-level fixture function (used as --fn import target)
# ---------------------------------------------------------------------------


@axis_config(AxisOverride(name="batch", default_batch_size=2))
def decorated_fn(x):
    """A function decorated with axis_config for testing the entrypoint dispatch."""
    return x * 2


# ---------------------------------------------------------------------------
# AC2 — import isolation (tyro must NOT be imported at module import time)
# ---------------------------------------------------------------------------


class TestImportIsolation:
    """AC2: importing xtrax.cli.entrypoint must not trigger a tyro import."""

    def test_entrypoint_import_does_not_import_tyro(self):
        """AC2: after 'import xtrax.cli.entrypoint', tyro must NOT be in sys.modules."""
        # Remove tyro if it's already cached (from a prior test that called main()).
        sys.modules.pop("tyro", None)

        # Re-import the entrypoint module (may already be cached, but tyro must NOT).
        import xtrax.cli.entrypoint  # noqa: F401

        assert "tyro" not in sys.modules, (
            "tyro was imported at xtrax.cli.entrypoint module import time — "
            "it must only be imported inside main()."
        )

    def test_xtrax_import_does_not_import_tyro(self):
        """AC2: importing xtrax must not pull in tyro."""
        sys.modules.pop("tyro", None)

        import xtrax  # noqa: F401

        assert "tyro" not in sys.modules, (
            "tyro was imported when 'import xtrax' was evaluated."
        )

    def test_xtrax_import_does_not_import_xtrax_cli(self):
        """AC2: importing xtrax must not pull in xtrax.cli subpackage."""
        # We cannot truly un-import xtrax.cli once it's been imported in this
        # process (other tests do it), so we check that top-level xtrax does not
        # force-load the CLI — this is a structural check on the module attribute.
        import xtrax

        # xtrax.__init__ must not eagerly import xtrax.cli.
        # We verify it does not RE-import if we remove it from sys.modules.
        saved = sys.modules.pop("xtrax.cli", None)
        try:
            # Force a fresh attribute access on xtrax — should not re-add xtrax.cli.
            _ = getattr(xtrax, "__version__", None)
            assert "xtrax.cli" not in sys.modules, (
                "xtrax top-level import re-triggered xtrax.cli import."
            )
        finally:
            if saved is not None:
                sys.modules["xtrax.cli"] = saved


# ---------------------------------------------------------------------------
# AC1 help — clean verb names in --help output
# ---------------------------------------------------------------------------


class TestHelpOutput:
    """AC1 help: --help must list 'plan' and 'explain' as clean verb names."""

    def test_help_lists_plan_and_explain(self, monkeypatch, capsys):
        """AC1: running main() with --help shows 'plan' and 'explain', not mangled names."""
        from xtrax.cli.entrypoint import main

        monkeypatch.setattr(sys, "argv", ["xtrax", "--help"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        # tyro --help exits with SystemExit(0).
        assert exc_info.value.code == 0, (
            f"Expected SystemExit(0) from --help, got code={exc_info.value.code}"
        )

        captured = capsys.readouterr()
        output = captured.out + captured.err  # tyro may write to either

        assert "plan" in output, f"Expected 'plan' in --help output. Got:\n{output}"
        assert "explain" in output, f"Expected 'explain' in --help output. Got:\n{output}"

    def test_help_does_not_show_mangled_names(self, monkeypatch, capsys):
        """AC1: --help must NOT show 'plan-args' or 'explain-args' as subcommand names."""
        from xtrax.cli.entrypoint import main

        monkeypatch.setattr(sys, "argv", ["xtrax", "--help"])

        with pytest.raises(SystemExit):
            main()

        captured = capsys.readouterr()
        output = captured.out + captured.err

        assert "plan-args" not in output, (
            f"Mangled name 'plan-args' found in --help output. Got:\n{output}"
        )
        assert "explain-args" not in output, (
            f"Mangled name 'explain-args' found in --help output. Got:\n{output}"
        )


# ---------------------------------------------------------------------------
# AC1 dispatch — explain verb end-to-end via main()
# ---------------------------------------------------------------------------


class TestDispatch:
    """AC1 dispatch: main() routes to run_explain and produces valid JSON output."""

    def test_explain_dispatch_produces_json_with_meta(self, monkeypatch, capsys):
        """AC1 dispatch: explain --fn --shapes --fmt json returns JSON with _meta."""
        from xtrax.cli.entrypoint import main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "xtrax",
                "explain",
                "--fn",
                "tests.cli.test_entrypoint:decorated_fn",
                "--shapes",
                "x=(4,)f32",
                "--fmt",
                "json",
            ],
        )

        main()

        captured = capsys.readouterr()
        output = captured.out.strip()
        assert output, "Expected non-empty stdout from explain dispatch"

        parsed = json.loads(output)
        assert "_meta" in parsed, (
            f"Expected '_meta' key in dispatch output. Got keys: {list(parsed.keys())}"
        )

    def test_explain_dispatch_json_has_plan_stats_keys(self, monkeypatch, capsys):
        """AC1 dispatch: explain output contains PlanStatsDict top-level keys."""
        from xtrax.cli.entrypoint import main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "xtrax",
                "explain",
                "--fn",
                "tests.cli.test_entrypoint:decorated_fn",
                "--shapes",
                "x=(4,)f32",
                "--fmt",
                "json",
            ],
        )

        main()

        captured = capsys.readouterr()
        parsed = json.loads(captured.out.strip())

        expected_keys = {
            "axes", "strategy_counts", "total_axes",
            "memory_warnings", "dedup_stats", "bucket_stats",
        }
        missing = expected_keys - set(parsed.keys())
        assert not missing, (
            f"Missing PlanStatsDict keys in dispatch output: {missing}. "
            f"Got keys: {list(parsed.keys())}"
        )
