"""CC5 severity×track routing matrix loader and resolver (N1.3 / #1579)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from xtrax.devtools.emit import Severity, SourceTrack

DEFAULT_ROUTING_PATH = Path("audit/routing.toml")


@dataclass(frozen=True, slots=True)
class RouteRow:
    domain: str
    track: SourceTrack
    severity: Severity
    destination: str
    signal: str = ""
    condition: str = ""


def load_routing_matrix(path: Path = DEFAULT_ROUTING_PATH) -> list[RouteRow]:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    raw_routes = payload.get("routes", [])
    if not isinstance(raw_routes, list):
        msg = "routing.toml routes must be a list"
        raise ValueError(msg)
    rows: list[RouteRow] = []
    for item in raw_routes:
        if not isinstance(item, dict):
            msg = "each [[routes]] entry must be a table"
            raise ValueError(msg)
        rows.append(
            RouteRow(
                domain=str(item["domain"]),
                track=item["track"],
                severity=item["severity"],
                destination=str(item["destination"]),
                signal=str(item.get("signal", "")),
                condition=str(item.get("condition", "")),
            )
        )
    return rows


def _row_matches(
    row: RouteRow,
    track: SourceTrack,
    severity: Severity,
    signals: dict[str, str] | None,
) -> bool:
    if row.track != track or row.severity != severity:
        return False
    if row.signal:
        if signals is None:
            return False
        return signals.get(row.signal) == row.condition
    return True


def _specificity(row: RouteRow) -> int:
    return 2 if row.signal else 1


def resolve_destination(
    *,
    domain: str,
    track: SourceTrack,
    severity: Severity,
    signals: dict[str, str] | None = None,
    path: Path = DEFAULT_ROUTING_PATH,
) -> str:
    """Match the most specific row; fall back to domain=dimension when needed."""
    rows = load_routing_matrix(path)
    domains_to_try = [domain] if domain == "dimension" else [domain, "dimension"]

    for try_domain in domains_to_try:
        best: RouteRow | None = None
        best_spec = -1
        for row in rows:
            if row.domain != try_domain:
                continue
            if not _row_matches(row, track, severity, signals):
                continue
            spec = _specificity(row)
            if spec > best_spec:
                best = row
                best_spec = spec
        if best is not None:
            return best.destination

    msg = (
        f"no routing row for domain={domain!r}, track={track!r}, "
        f"severity={severity!r}, signals={signals!r}"
    )
    raise ValueError(msg)
