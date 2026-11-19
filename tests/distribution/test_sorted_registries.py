"""Export registries stay in plain ``sorted()`` order with no duplicates.

Parallel PRs used to append to the end of the same lists and conflict on that
one line.  Alphabetical order puts each addition on its own line.  The helper
:func:`_problems` is exercised by a negative control so the guard cannot go
vacuous.
"""

from __future__ import annotations

import ast
import tomllib
from collections.abc import Sequence
from pathlib import Path

import pytest

import xtrax

_ROOT = Path(__file__).resolve().parents[2]


def _problems(name: str, items: Sequence[str]) -> list[str]:
    """Return human-readable defects for *items* (empty when sorted and unique)."""
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            out.append(f"{name}: duplicate entry {item!r}")
        seen.add(item)
    ordered = sorted(items)
    for got, want in zip(items, ordered, strict=True):
        if got != want:
            out.append(f"{name}: not sorted; first out-of-place entry {got!r} (expected {want!r})")
            break
    return out


def _import_isolation_packages() -> list[str]:
    tree = ast.parse((_ROOT / "tests" / "test_import_isolation.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_PACKAGES" for t in node.targets
        ):
            return list(ast.literal_eval(node.value))
    raise AssertionError("_PACKAGES not found in tests/test_import_isolation.py")


def _public_api() -> dict:
    return tomllib.loads((_ROOT / "distribution" / "public_api.toml").read_text())["api"]


def _registries() -> dict[str, list[str]]:
    api = _public_api()
    return {
        "xtrax.__all__": list(xtrax.__all__),
        "xtrax._LAZY (keys)": list(xtrax._LAZY),
        "public_api.toml tier1_exports": list(api["tier1_exports"]),
        "public_api.toml smoke_exports": list(api["smoke_exports"]),
        "tests/test_import_isolation.py _PACKAGES": _import_isolation_packages(),
    }


@pytest.mark.parametrize("name", list(_registries()))
def test_registry_sorted_and_unique(name: str) -> None:
    items = _registries()[name]
    assert items, f"{name} is empty"
    assert _problems(name, items) == []


def test_checker_flags_unsorted_and_duplicates() -> None:
    """Negative control: the helper must reject bad lists."""
    assert _problems("x", ["b", "a"])
    assert _problems("x", ["a", "a"])
    assert _problems("x", ["a", "b", "c"]) == []
