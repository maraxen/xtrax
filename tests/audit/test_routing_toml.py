"""Minimal contract test for audit/routing.toml CC5 matrix (#2280)."""

import tomllib
from pathlib import Path

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
    assert len(port_routes) >= 3, (
        "expected domain=port rows for tier FAIL, static WARN, observation"
    )

    for row in port_routes:
        assert row.get("track") in {"deterministic", "judgment"}
        assert row.get("severity") in {"info", "minor", "major", "critical"}
        assert row.get("destination") in VALID_DESTINATIONS, row

    destinations = {row["destination"] for row in port_routes}
    assert "block_ci" in destinations or "backlog_node" in destinations
    assert "found_issues" in destinations


def test_matrix_version_bumped() -> None:
    data = _load_routing()
    assert data["matrix"]["version"] == "0.2.0"


def test_dimension_domain_rows_cover_track_severity_matrix() -> None:
    data = _load_routing()
    dimension_routes = [
        row for row in data["routes"] if row.get("domain") == "dimension"
    ]
    assert len(dimension_routes) == 8

    combos = {
        (row["track"], row["severity"])
        for row in dimension_routes
    }
    expected = {
        ("deterministic", sev) for sev in ("info", "minor", "major", "critical")
    } | {
        ("judgment", sev) for sev in ("info", "minor", "major", "critical")
    }
    assert combos == expected

    for row in dimension_routes:
        assert row.get("destination") in VALID_DESTINATIONS, row
        track, severity = row["track"], row["severity"]
        if track == "deterministic" and severity in {"critical", "major"}:
            assert row["destination"] == "block_ci"
        elif track == "judgment" and severity in {"critical", "major"}:
            assert row["destination"] == "backlog_node"
        else:
            assert row["destination"] == "found_issues"
