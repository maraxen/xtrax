"""routing.toml's domain=port ``block_ci`` rows are actually enforced by CI (debt #423).

``audit/routing.toml`` routes a deterministic port tier FAIL to ``block_ci``, but no
code reads that row to block anything: the enforcement is the ``audit-port`` CI job
running ``just audit-port`` -> ``pytest port/tests/``, where a failing parity tier
fails the job. That chain was implicit and unverified. This module pins every link,
and proves the last one with a planted perturbation: a wrong ``chunked_map`` run through
the REAL port conftest and parity tests must make pytest exit non-zero, while the
unperturbed copy must pass (so the failure is the perturbation, not the harness).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[2]

_PLANT = '''
"""pytest plugin: make xtrax's chunked_map wrong before the parity tests import it."""
import xtrax.transforms.map as _m

_real = _m.chunked_map


def _wrong(fn, xs, *args, **kwargs):
    return _real(fn, xs, *args, **kwargs) + 1.0


def pytest_configure(config):
    _m.chunked_map = _wrong
'''


def _port_block_ci_rows() -> list[dict]:
    routes = tomllib.loads((ROOT / "audit" / "routing.toml").read_text(encoding="utf-8"))["routes"]
    return [r for r in routes if r["domain"] == "port" and r["destination"] == "block_ci"]


def test_port_block_ci_rows_exist_and_are_deterministic() -> None:
    """CI can only enforce what it can compute: block_ci must sit on the deterministic track."""
    rows = _port_block_ci_rows()
    assert rows, "routing.toml has no domain=port block_ci row -- nothing to enforce"
    assert {r["track"] for r in rows} == {"deterministic"}


def test_ci_runs_audit_port_on_port_and_source_changes() -> None:
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    job = workflow["jobs"]["audit-port"]
    assert any(step.get("run", "").strip() == "just audit-port" for step in job["steps"])
    filters = yaml.safe_load(
        next(
            s["with"]["filters"]
            for s in workflow["jobs"]["port-changes"]["steps"]
            if s.get("id") == "filter"
        )
    )
    assert {"port/**", "src/xtrax/**"} <= set(filters["port"])


def test_audit_port_recipe_runs_the_parity_tests() -> None:
    justfile = (ROOT / "Justfile").read_text(encoding="utf-8")
    deps = re.search(r"^audit-port:(.*)$", justfile, re.MULTILINE)
    assert deps is not None and "audit-port-parity" in deps.group(1).split()
    body = re.search(r"^audit-port-parity:\n((?:[ \t]+.*\n)+)", justfile, re.MULTILINE)
    assert body is not None and "pytest port/tests/" in body.group(1)


def _run_port_harness(tmp_path: Path, *, plant: bool) -> subprocess.CompletedProcess[str]:
    """Run the parity harness on a COPY of port/ so tier verdicts land in tmp, not the repo."""
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "port", root / "port", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "pyproject.toml", root / "pyproject.toml")
    args = [
        sys.executable,
        "-m",
        "pytest",
        "port/tests/test_parity_safe_map.py",
        "-q",
        "-o",
        "addopts=",
        "-p",
        "no:cacheprovider",
        "--rootdir",
        str(root),
    ]
    env = None
    if plant:
        (tmp_path / "plant_wrong_safe_map.py").write_text(_PLANT, encoding="utf-8")
        args += ["-p", "plant_wrong_safe_map"]
        import os

        env = {
            **os.environ,
            "PYTHONPATH": f"{tmp_path}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        }
    return subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=600, env=env)


def test_unperturbed_port_harness_passes(tmp_path: Path) -> None:
    """Positive control: the copied harness is sound, so a failure below is the plant."""
    result = _run_port_harness(tmp_path, plant=False)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert (tmp_path / "repo" / ".praxia" / "audits.jsonl").is_file()  # verdicts stayed in tmp


def test_planted_parity_failure_fails_the_harness(tmp_path: Path) -> None:
    """Negative control: a wrong chunked_map through the real conftest must fail pytest."""
    result = _run_port_harness(tmp_path, plant=True)
    assert result.returncode != 0, "a wrong chunked_map passed the parity harness"
    assert "FAILED" in result.stdout
    assert "Not equal to tolerance" in result.stdout  # failed ON parity, not on setup
    audits = (tmp_path / "repo" / ".praxia" / "audits.jsonl").read_text(encoding="utf-8")
    assert '"FAIL"' in audits  # the FAIL tier verdict was emitted, not swallowed
