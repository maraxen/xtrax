"""Tests for port/bridge/composition_map.toml lint (composition-bridge-stub)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_LINT_PATH = ROOT / "scripts" / "lint_port_bridge_map.py"
_spec = importlib.util.spec_from_file_location("lint_port_bridge_map", _LINT_PATH)
assert _spec and _spec.loader
lint_port_bridge_map = importlib.util.module_from_spec(_spec)
sys.modules["lint_port_bridge_map"] = lint_port_bridge_map
_spec.loader.exec_module(lint_port_bridge_map)

DEFAULT_MAP = ROOT / "port" / "bridge" / "composition_map.toml"


def test_empty_map_passes() -> None:
    failures, exit_code = lint_port_bridge_map.lint_composition_map(DEFAULT_MAP)
    assert exit_code == 0
    assert failures == []


def test_empty_symbols_table_passes(tmp_path: Path) -> None:
    map_path = tmp_path / "composition_map.toml"
    map_path.write_text("[symbols]\n", encoding="utf-8")
    failures, exit_code = lint_port_bridge_map.lint_composition_map(map_path)
    assert exit_code == 0
    assert failures == []


def test_invalid_qualname_fails_when_map_populated(tmp_path: Path) -> None:
    map_path = tmp_path / "composition_map.toml"
    map_path.write_text(
        '[symbols]\n"xtrax.no.such.symbol" = "fake.node"\n',
        encoding="utf-8",
    )
    failures, exit_code = lint_port_bridge_map.lint_composition_map(map_path)
    assert exit_code == 1
    assert len(failures) == 1
    assert "xtrax.no.such.symbol" in failures[0]


def test_valid_qualname_passes_when_map_populated(tmp_path: Path) -> None:
    map_path = tmp_path / "composition_map.toml"
    map_path.write_text(
        '[symbols]\n"xtrax.transforms.map.safe_map" = "transforms.safe_map"\n',
        encoding="utf-8",
    )
    failures, exit_code = lint_port_bridge_map.lint_composition_map(map_path)
    assert exit_code == 0
    assert failures == []


def test_main_cli_empty_map() -> None:
    assert lint_port_bridge_map.main(["--map", str(DEFAULT_MAP)]) == 0


def test_main_cli_invalid_map(tmp_path: Path) -> None:
    map_path = tmp_path / "composition_map.toml"
    map_path.write_text(
        '[symbols]\n"xtrax.missing.kernel" = "node"\n',
        encoding="utf-8",
    )
    assert lint_port_bridge_map.main(["--map", str(map_path)]) == 1
