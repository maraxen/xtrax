"""Smoke tests for port_validation PCW template (#2278)."""

from __future__ import annotations

import re
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


def test_claude_pcw_artifact_still_named() -> None:
    """Claude-PCW JS + dw_mapping still name the port_validation artifact."""
    assert JS_PATH.is_file()
    mapping = MAPPING_PATH.read_text(encoding="utf-8")
    assert 'port_validation = "port-validation"' in mapping


# --- TestAcV33062 (backlog #4375 / T3-08 AC-V3) --------------------------------


def _acv33062_load_yaml() -> dict:
    """Same parse path as test_port_validation_yaml_exists_and_parses."""
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))


def _acv33062_js_meta_titles() -> list[str]:
    """Same meta-title extract as test_port_validation_js_exists_with_seven_phases."""
    js = JS_PATH.read_text(encoding="utf-8")
    meta_block = js.split("export const meta")[1].split("};")[0]
    return re.findall(r'title:\s*"([^"]+)"', meta_block)


class TestAcV33062:
    """AC-V3: YAML↔JS 1:1 titles, no strict tool_profile, no dw-emit in this module."""

    YAML_ID_TO_JS_TITLE = (
        ("p0_oracle", "P0-ORACLE"),
        ("p1_spec", "P1-SPEC"),
        ("p1_5_topo", "P1.5-TOPO"),
        ("p2_static", "P2-STATIC"),
        ("p3_parity", "P3-PARITY"),
        ("p4_emit", "P4-EMIT"),
        ("p5_route", "P5-ROUTE"),
    )

    def test_yaml_role_step_ids_map_1to1_to_js_meta_titles(self) -> None:
        data = _acv33062_load_yaml()
        role_steps = [n for n in data["nodes"] if n.get("kind") == "role_step"]
        yaml_ids = [n["id"] for n in role_steps]
        js_titles = _acv33062_js_meta_titles()
        expected_ids = [pair[0] for pair in self.YAML_ID_TO_JS_TITLE]
        expected_titles = [pair[1] for pair in self.YAML_ID_TO_JS_TITLE]
        assert yaml_ids == expected_ids
        assert js_titles == expected_titles
        assert list(zip(yaml_ids, js_titles, strict=True)) == list(self.YAML_ID_TO_JS_TITLE)

    def test_no_yaml_node_has_tool_profile_or_action_mode_strict(self) -> None:
        data = _acv33062_load_yaml()
        for node in data["nodes"]:
            assert "tool_profile" not in node, (
                f"TestAcV33062: node {node.get('id')!r} must not set tool_profile"
            )
            action_mode = node.get("action_mode")
            assert action_mode != "strict", (
                f"TestAcV33062: node {node.get('id')!r} must not set action_mode: strict "
                f"(absent or non-strict is OK; got {action_mode!r})"
            )

    def test_module_source_does_not_invoke_or_skip_dw(self) -> None:
        """Needles built at runtime so this method does not contain the banned text."""
        source = Path(__file__).read_text(encoding="utf-8")
        # Needles assembled so this method's own source does not contain them.
        dw_emit_needle = "dw" + " emit"
        argv_needle = '"dw", ' + '"emit"'
        unrecognized_needle = "unrecognized subcommand " + "'dw'"
        skip_call = "pytest" + ".skip"

        assert dw_emit_needle not in source, (
            "TestAcV33062: module must not invoke dw-emit (contiguous form)"
        )
        assert argv_needle not in source, (
            "TestAcV33062: module must not invoke dw-emit (argv list form)"
        )
        assert unrecognized_needle not in source, (
            "TestAcV33062: module must not special-case missing dw CLI; "
            "remaining dw-emit must FAIL not skip"
        )
        dw_gone_skip = skip_call in source and unrecognized_needle in source
        assert not dw_gone_skip, (
            "TestAcV33062: skip() must not be used for dw-gone "
            "(test_port_validation_yaml_emits_without_error still does; fixer replaces it)"
        )
