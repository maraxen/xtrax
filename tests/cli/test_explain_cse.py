"""CLI acceptance tests for explain --report cse (spec 260825 §4.1)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("jax")

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
# tests/cli, placed on PYTHONPATH below so `_cse_demo.demo` resolves. The demo
# package is committed under tests/cli/_cse_demo/ rather than generated at test
# time, so the CLI subprocess and the assertions share one definition of the
# demo functions.
DEMO_DIR = Path(__file__).resolve().parent


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:  # type: ignore[name-defined]
    import os
    import subprocess

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_SRC), str(DEMO_DIR), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-c", "from xtrax.cli import main; main()", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


CSE_ARGS = [
    "explain",
    "--fn",
    "_cse_demo.demo:duplicated_compute",
    "--shapes",
    "x=(8)f32",
    "--report",
    "cse",
]


class TestExplainCseReport:
    def test_json_envelope_and_ac1(self) -> None:
        result = _run_cli([*CSE_ARGS, "--fmt", "json"])
        assert result.returncode == 0, result.stderr
        out = json.loads(result.stdout)
        assert out["_meta"]["schema_version"] == 1
        cr = out["cse_report"]
        assert cr["schema_version"] == 1
        by_prim = {d["primitive"]: d["eqn_count"] for d in cr["duplicates"]}
        assert by_prim == {"sin": 2, "mul": 2}
        assert cr["total_eqns"] == 7
        assert cr["duplicate_eqns"] == 4

    def test_text_format_clean_fn(self) -> None:
        result = _run_cli(
            [
                "explain",
                "--fn",
                "_cse_demo.demo:clean_compute",
                "--shapes",
                "x=(8)f32",
                "--report",
                "cse",
                "--fmt",
                "text",
            ]
        )
        assert result.returncode == 0, result.stderr
        assert "No duplicate subexpressions detected." in result.stdout

    def test_html_rejected_with_cli_error(self) -> None:
        from xtrax.cli.errors import CLIError  # noqa: F401  (contract reference)

        result = _run_cli([*CSE_ARGS, "--fmt", "html"])
        assert result.returncode == 1
        assert "not supported" in result.stderr

    def test_plan_report_still_default(self) -> None:
        """Backward compat: without --report, the plan path runs unchanged."""
        result = _run_cli(
            ["explain", "--fn", "_cse_demo.demo:annotated_compute", "--shapes", "x=(64,16)f32"]
        )
        assert result.returncode == 0, result.stderr
        out = json.loads(result.stdout)
        assert "axes" in out  # plan-stats payload shape
