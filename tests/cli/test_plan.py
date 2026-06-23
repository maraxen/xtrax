"""Tests for xtrax.cli.plan module.

Tests the run_plan function which orchestrates loading functions, parsing shapes,
inferring bundle schemas, planning tiling strategies, and printing results.
"""

from __future__ import annotations

import pytest

from xtrax.cli.errors import CLIError
from xtrax.cli.plan import PlanArgs, run_plan
from xtrax.inference.config import AxisOverride, axis_config  # noqa: F401

# Module-level fixture functions (used as import path targets)


@axis_config(AxisOverride(name="batch", default_batch_size=2))
def decorated_fn(x):
    """A function decorated with axis_config for testing.

    This simulates a real user function that has been annotated with
    axis override information. The decorator resolves the axis role,
    so planning should succeed.
    """
    return x * 2


def bare_fn(x):
    """A bare function without @axis_config decorator.

    This simulates an undecorated user function. Planning should fail
    with an AmbiguousAxisError wrapped in CLIError, because the axis
    role is UNKNOWN and cannot be resolved.
    """
    return x * 2


class TestPlanHappy:
    """Tests for successful plan verb execution with decorated functions."""

    def test_run_plan_with_decorated_fn(self, capsys):
        """AC4 HAPPY: decorated_fn with resolved axis role succeeds.

        When a function is decorated with @axis_config, the axis role
        is KNOWN, so BatchPlanner.plan() should succeed and output a
        readable summary to stdout.
        """
        args = PlanArgs(fn="tests.cli.test_plan:decorated_fn", shapes="x=(4,)f32")
        run_plan(args)

        # Capture stdout
        captured = capsys.readouterr()
        output = captured.out

        # Verify output is non-empty and contains plan info
        assert len(output) > 0, "Expected non-empty stdout for successful plan"
        assert "batch" in output.lower() or "axis" in output.lower() or "plan" in output.lower(), (
            f"Expected plan info in output, got: {output}"
        )


class TestPlanFailLoud:
    """Tests for fail-loud behavior on undecorated functions."""

    def test_run_plan_with_bare_fn_raises_cli_error(self):
        """AC4 FAIL-LOUD: bare_fn with UNKNOWN role raises CLIError.

        When a function is not decorated, the axis role is UNKNOWN.
        BatchPlanner.plan() will raise AmbiguousAxisError, which should
        be caught and wrapped in a CLIError with a clean, actionable message
        that mentions @axis_config (not a raw traceback).
        """
        args = PlanArgs(fn="tests.cli.test_plan:bare_fn", shapes="x=(4,)f32")

        with pytest.raises(CLIError) as exc_info:
            run_plan(args)

        error_msg = str(exc_info.value)

        # Verify the message is clean and actionable
        assert "@axis_config" in error_msg or "axis_config" in error_msg, (
            f"Expected @axis_config hint in error message: {error_msg}"
        )

        # Verify it's not a raw AmbiguousAxisError traceback
        assert "Traceback" not in error_msg, (
            f"Expected clean error message, not raw traceback: {error_msg}"
        )
