"""Smoke tests for jaxlint JSON audit runner (N0.4 foundation gate)."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_jaxlint_json_runner_performance_mode_is_clean() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/audit_jaxlint_json.py",
            "--performance-only",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    envelope = json.loads(result.stdout)
    assert envelope["schema_version"] == "audit_jaxlint_v0"
    assert envelope["mode"] == "performance_only"
    assert envelope["error_count"] == 0
    assert envelope["target"] == "src/xtrax"


def test_jaxlint_json_runner_writes_output_file(tmp_path: Path) -> None:
    out = tmp_path / "jaxlint_audit.json"
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/audit_jaxlint_json.py",
            "-o",
            str(out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["finding_count"] >= 0
