"""Tests for scripts/check_dependency_floors.py (debt #738)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.version import Version

from scripts.check_dependency_floors import declared_floors, floor_mismatches

ROOT = Path(__file__).resolve().parents[2]

_PYPROJECT = {
    "project": {
        "dependencies": ["jax>=0.10.2,<0.12", "numpy>=2.1", "unpinned"],
        "optional-dependencies": {"io": ["zarr>=3.0.8"], "dev": ["xtrax[io]", "pytest>=8"]},
    }
}


def test_declared_floors_reads_core_and_named_extras_only() -> None:
    assert declared_floors(_PYPROJECT, ["io"]) == {
        "jax": Version("0.10.2"),
        "numpy": Version("2.1"),
        "zarr": Version("3.0.8"),
    }


def test_self_referencing_extra_is_skipped() -> None:
    assert "xtrax" not in declared_floors(_PYPROJECT, ["dev"])


def test_unknown_extra_fails_loud() -> None:
    with pytest.raises(SystemExit, match="unknown extra"):
        declared_floors(_PYPROJECT, ["nope"])


def test_exact_floors_pass_and_trailing_zero_is_equal() -> None:
    floors = {"numpy": Version("2.1"), "jax": Version("0.10.2")}
    assert floor_mismatches(floors, {"numpy": "2.1.0", "jax": "0.10.2"}) == []


def test_unreachable_floor_fails() -> None:
    """Negative control: the numpy>=1.26 / zarr>=3.0 shape -- resolver landed above the floor."""
    problems = floor_mismatches({"numpy": Version("1.26")}, {"numpy": "2.1.0"})
    assert len(problems) == 1
    assert "unreachable" in problems[0]


def test_missing_distribution_fails() -> None:
    assert "not installed" in floor_mismatches({"zarr": Version("3.0.8")}, {"zarr": None})[0]


def test_repo_floors_parse() -> None:
    """The real pyproject's core + io + cli floors parse and include the pinned core."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    floors = declared_floors(data, ["io", "cli"])
    assert {"jax", "jaxlib", "equinox", "optax", "orbax-checkpoint", "numpy", "zarr"} <= set(floors)
