"""Judgment-track rubric anchor tables (N4.2 / #1591)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_RUBRICS_DIR = Path("audit/rubrics")


@dataclass(frozen=True, slots=True)
class RubricAnchor:
    score: int
    criterion: str
    evidence_hint: str


@dataclass(frozen=True, slots=True)
class RubricTable:
    dimension: str
    version: str
    dim_label: str
    anchors: tuple[RubricAnchor, ...]


def _parse_anchors(raw_anchors: object) -> tuple[RubricAnchor, ...]:
    if not isinstance(raw_anchors, list):
        msg = "anchors must be a list"
        raise ValueError(msg)
    anchors: list[RubricAnchor] = []
    for entry in raw_anchors:
        if not isinstance(entry, dict):
            msg = "each anchor must be a mapping"
            raise ValueError(msg)
        anchors.append(
            RubricAnchor(
                score=int(entry["score"]),
                criterion=str(entry["criterion"]),
                evidence_hint=str(entry["evidence_hint"]),
            )
        )
    return tuple(anchors)


def load_rubric(path: Path) -> RubricTable:
    """Load one rubric TOML file into a RubricTable."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    rubric = data.get("rubric")
    if not isinstance(rubric, dict):
        msg = f"{path}: [rubric] section is required"
        raise ValueError(msg)
    dimension = str(rubric["dimension"])
    stem = path.stem
    if dimension != stem:
        msg = f"{path}: dimension {dimension!r} must match filename stem {stem!r}"
        raise ValueError(msg)
    return RubricTable(
        dimension=dimension,
        version=str(rubric["version"]),
        dim_label=str(rubric["dim_label"]),
        anchors=_parse_anchors(data.get("anchors")),
    )


def load_all_rubrics(
    dir: Path = DEFAULT_RUBRICS_DIR,
) -> dict[str, RubricTable]:
    """Load every ``*.toml`` rubric in ``dir``, keyed by dimension slug."""
    if not dir.is_dir():
        msg = f"rubrics directory not found: {dir}"
        raise FileNotFoundError(msg)
    tables: dict[str, RubricTable] = {}
    for path in sorted(dir.glob("*.toml")):
        table = load_rubric(path)
        tables[table.dimension] = table
    return tables
