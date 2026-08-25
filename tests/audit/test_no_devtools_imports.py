"""Shipped code must not import xtrax.devtools (unshipped tree).

Regression guard for the wheel breakage found by aminx dogfooding
(2026-08-25): xtrax.run.repro_floor + three loop gates imported
devtools.freshness/_jaxlint while pyproject excludes src/xtrax/devtools
from the wheel -- every downstream install hit ModuleNotFoundError.
The importlinter contract documents intent but did not flag these; this
AST scan is the enforcing gate (runs in audit-deterministic CI).
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "xtrax"
FORBIDDEN_ROOT = "xtrax.devtools"

# Omitted-from-wheel trees may freely import each other / devtools.
OMITTED = ("xtrax/devtools/", "xtrax/eda/")


def _imports_devtools(tree: ast.AST) -> list[str]:
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == FORBIDDEN_ROOT or alias.name.startswith(
                    FORBIDDEN_ROOT + "."
                ):
                    hits.append(f"{node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            level = node.level
            if level == 0 and (
                mod == FORBIDDEN_ROOT or mod.startswith(FORBIDDEN_ROOT + ".")
            ):
                hits.append(f"{node.lineno}: from {mod} import {', '.join(a.name for a in node.names)}")
    return hits


def test_shipped_tree_never_imports_devtools() -> None:
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if any(rel.startswith(prefix) for prefix in OMITTED):
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for hit in _imports_devtools(tree):
            offenders.append(f"{rel}: {hit}")
    assert not offenders, (
        "shipped modules import the unshipped xtrax.devtools tree "
        "(ModuleNotFoundError for every wheel consumer): "
        + "; ".join(offenders)
    )


def test_guard_detects_the_original_violation() -> None:
    # The guard must catch the exact shape that shipped broken in the first
    # place (import + from-import forms).
    probe = 'from xtrax.devtools.freshness import Attestation\n'
    assert _imports_devtools(ast.parse(probe))
    probe2 = "import xtrax.devtools.emit\n"
    assert _imports_devtools(ast.parse(probe2))
