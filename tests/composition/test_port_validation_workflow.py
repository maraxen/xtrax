"""Smoke tests for port_validation PCW template (#2278)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = ROOT / "agent_assets/workflows/port_validation.yaml"
JS_PATH = ROOT / ".claude/workflows/port-validation.js"
MAPPING_PATH = ROOT / "agent_assets/workflows/dw_mapping.toml"


def test_port_validation_yaml_exists_and_parses() -> None:
    assert YAML_PATH.is_file()
    text = YAML_PATH.read_text(encoding="utf-8")
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(text)
    assert data["name"] == "port_validation"
    role_steps = [n for n in data["nodes"] if n.get("kind") == "role_step"]
    assert len(role_steps) == 7
    ids = {n["id"] for n in role_steps}
    assert ids == {
        "p0_oracle",
        "p1_spec",
        "p1_5_topo",
        "p2_static",
        "p3_parity",
        "p4_emit",
        "p5_route",
    }
    assert data["budgets"]["max_total_rewinds"] == 6
    assert data["budgets"]["max_cost_usd"] == 12.0


def test_port_validation_js_exists_with_seven_phases() -> None:
    assert JS_PATH.is_file()
    js = JS_PATH.read_text(encoding="utf-8")
    assert "export const meta" in js
    assert "MAX_FIX_RETRIES = 2" in js
    assert "validatePortHookPayload" in js
    assert "port/docs/hook_schema_port_validation.md" in js
    assert "audit/routing.toml" in js
    meta_block = js.split("export const meta")[1].split("};")[0]
    meta_titles = re.findall(r"title:\s*\"([^\"]+)\"", meta_block)
    assert len(meta_titles) == 7


def test_dw_mapping_points_at_port_validation_js() -> None:
    assert MAPPING_PATH.is_file()
    text = MAPPING_PATH.read_text(encoding="utf-8")
    assert 'port_validation = "port-validation"' in text


def test_port_validation_yaml_emits_without_error() -> None:
    env = {**dict(**__import__("os").environ), "PRAXIA_WORKFLOWS_DIR": str(YAML_PATH.parent)}
    result = subprocess.run(
        ["praxia", "dw", "emit", "port_validation", "--stdout"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "export const meta" in result.stdout
