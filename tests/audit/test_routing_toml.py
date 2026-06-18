"""Minimal contract test for audit/routing.toml CC5 matrix (#2280)."""

from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2]
ROUTING_TOML = ROOT / "audit" / "routing.toml"

VALID_DESTINATIONS = frozenset(
    {"block_ci", "tombstone_eligible", "found_issues", "backlog_node"}
)


def _load_routing() -> dict:
    assert ROUTING_TOML.is_file(), f"missing {ROUTING_TOML}"
    return tomllib.loads(ROUTING_TOML.read_text(encoding="utf-8"))


def test_routing_toml_parses() -> None:
    data = _load_routing()
    assert "matrix" in data
    assert data["matrix"]["schema"] == "cc5-routing-v0"
    assert isinstance(data.get("routes"), list)
    assert data["routes"], "expected at least one [[routes]] row"


def test_port_domain_rows_have_valid_destinations() -> None:
    data = _load_routing()
    port_routes = [row for row in data["routes"] if row.get("domain") == "port"]
    assert len(port_routes) >= 3, "expected domain=port rows for tier FAIL, static WARN, observation"

    for row in port_routes:
        assert row.get("track") in {"deterministic", "judgment"}
        assert row.get("severity") in {"info", "minor", "major", "critical"}
        assert row.get("destination") in VALID_DESTINATIONS, row

    destinations = {row["destination"] for row in port_routes}
    assert "block_ci" in destinations or "backlog_node" in destinations
    assert "found_issues" in destinations
