"""Contract tests for xtrax.devtools.routing resolve engine (#1579)."""

from pathlib import Path

import pytest

from xtrax.devtools.routing import (
    DEFAULT_ROUTING_PATH,
    load_routing_matrix,
    resolve_destination,
)

ROOT = Path(__file__).resolve().parents[2]


def test_load_routing_matrix_includes_port_and_dimension_rows() -> None:
    rows = load_routing_matrix(ROOT / DEFAULT_ROUTING_PATH)
    domains = {row.domain for row in rows}
    assert "port" in domains
    assert "dimension" in domains
    assert len([row for row in rows if row.domain == "dimension"]) == 8


def test_resolve_dimension_deterministic_major_blocks_ci() -> None:
    dest = resolve_destination(
        domain="dimension",
        track="deterministic",
        severity="major",
        path=ROOT / DEFAULT_ROUTING_PATH,
    )
    assert dest == "block_ci"


@pytest.mark.parametrize(
    ("track", "severity", "expected"),
    [
        ("deterministic", "critical", "block_ci"),
        ("deterministic", "major", "block_ci"),
        ("deterministic", "minor", "found_issues"),
        ("deterministic", "info", "found_issues"),
        ("judgment", "critical", "backlog_node"),
        ("judgment", "major", "backlog_node"),
        ("judgment", "minor", "found_issues"),
        ("judgment", "info", "found_issues"),
    ],
)
def test_resolve_dimension_matrix(track: str, severity: str, expected: str) -> None:
    dest = resolve_destination(
        domain="dimension",
        track=track,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        path=ROOT / DEFAULT_ROUTING_PATH,
    )
    assert dest == expected


def test_resolve_port_row_with_signals() -> None:
    dest = resolve_destination(
        domain="port",
        track="deterministic",
        severity="major",
        signals={"tier_verdict.status": "FAIL"},
        path=ROOT / DEFAULT_ROUTING_PATH,
    )
    assert dest == "block_ci"


def test_resolve_invalid_combo_raises(tmp_path: Path) -> None:
    empty_matrix = tmp_path / "routing.toml"
    empty_matrix.write_text(
        '[matrix]\nschema = "cc5-routing-v0"\nversion = "0.0.0"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no routing row"):
        resolve_destination(
            domain="port",
            track="deterministic",
            severity="critical",
            path=empty_matrix,
        )
