#!/usr/bin/env python3
"""Gate: telemetry enforcement cannot be removed or silently skipped.

Two independent checks, both AST-based rather than grep-based so that a comment
or a docstring mentioning the right words cannot satisfy them.

1. **Engine still enforces.** Every method named in
   ``audit/telemetry_coverage.toml``'s ``[engine].enforcing_methods`` must call
   ``_resolve_ledger`` in its own body. This is the regression guard for the
   whole feature: delete the ledger wiring from ``Engine.fit`` and this fails,
   rather than the loss being noticed months later by an auditor holding a
   result nobody can reconstruct.

2. **Every CLI verb has a declared disposition.** Each key of
   ``xtrax.cli.registry.REGISTRY`` must appear in ``[verbs]``. The contract is
   not that every verb records -- ``plan`` and ``explain`` legitimately do not --
   but that the answer was decided rather than defaulted. Adding a verb that
   executes user code without thinking about telemetry is exactly the drift this
   catches, and an unknown verb fails closed.

Run directly, or via ``just audit-telemetry-coverage``. Exit code 1 on any
violation, so it can gate CI.
"""

from __future__ import annotations

import argparse
import ast
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "audit" / "telemetry_coverage.toml"
ENGINE_PATH = ROOT / "src" / "xtrax" / "engine" / "engine.py"
REGISTRY_PATH = ROOT / "src" / "xtrax" / "cli" / "registry.py"

LEDGER_RESOLVER = "_resolve_ledger"
VALID_DISPOSITIONS = frozenset({"records", "analysis_only", "ledger_admin"})


def _rel(path: Path) -> str:
    """Repo-relative path for messages, falling back to the absolute one.

    ``Path.relative_to`` raises for anything outside ROOT, which would make the
    gate crash instead of report when handed a path elsewhere (its own tests do
    exactly that). A lint gate that dies while formatting an error message is
    worse than the violation it found.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_contract(path: Path = CONTRACT_PATH) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _calls_resolver(node: ast.AST) -> bool:
    """Whether this function body contains a call to the ledger resolver."""
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name == LEDGER_RESOLVER:
                return True
    return False


def check_engine_enforces(contract: dict, engine_path: Path = ENGINE_PATH) -> list[str]:
    """Assert each declared method opens or accepts a ledger."""
    required = list(contract.get("engine", {}).get("enforcing_methods", []))
    if not required:
        return ["audit/telemetry_coverage.toml declares no [engine].enforcing_methods"]

    tree = ast.parse(engine_path.read_text(encoding="utf-8"), filename=str(engine_path))
    found: dict[str, bool] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in required:
            found[node.name] = _calls_resolver(node)

    problems: list[str] = []
    for name in required:
        if name not in found:
            problems.append(
                f"{_rel(engine_path)}: method {name!r} is declared as "
                "telemetry-enforcing but was not found -- was it renamed?"
            )
        elif not found[name]:
            problems.append(
                f"{_rel(engine_path)}: {name}() no longer calls "
                f"{LEDGER_RESOLVER}(). Telemetry enforcement has been removed: a run "
                "through this path would execute without producing a ledger row, and "
                "provenance cannot be captured retroactively."
            )
    return problems


def _registry_keys(registry_path: Path = REGISTRY_PATH) -> list[str]:
    """Read REGISTRY's keys statically.

    Parsed rather than imported: importing xtrax.cli pulls in jax and the whole
    verb surface, which makes a lint gate slow and couples it to runtime import
    health. The keys are string literals in a dict literal, so AST is exact here.
    """
    tree = ast.parse(registry_path.read_text(encoding="utf-8"), filename=str(registry_path))
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if not any(getattr(t, "id", None) == "REGISTRY" for t in targets):
            continue
        value = node.value
        if isinstance(value, ast.Dict):
            return [k.value for k in value.keys if isinstance(k, ast.Constant)]
    return []


def check_verbs_declared(contract: dict, registry_path: Path = REGISTRY_PATH) -> list[str]:
    """Assert every CLI verb has a reviewed telemetry disposition."""
    declared = contract.get("verbs", {})
    problems: list[str] = []

    for verb, disposition in declared.items():
        if disposition not in VALID_DISPOSITIONS:
            problems.append(
                f"audit/telemetry_coverage.toml: verb {verb!r} has unknown "
                f"disposition {disposition!r}; expected one of {sorted(VALID_DISPOSITIONS)}"
            )

    keys = _registry_keys(registry_path)
    if not keys:
        problems.append(
            f"{_rel(registry_path)}: could not read REGISTRY keys statically"
        )
        return problems

    for verb in keys:
        if verb not in declared:
            problems.append(
                f"CLI verb {verb!r} has no telemetry disposition. Add it to "
                "audit/telemetry_coverage.toml [verbs]: 'records' if it executes or "
                "lowers user code, 'analysis_only' if it does neither, "
                "'ledger_admin' if it operates on the ledger itself."
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    args = parser.parse_args(argv)

    contract = load_contract(args.contract)
    problems = check_engine_enforces(contract) + check_verbs_declared(contract)

    if problems:
        print("FAIL: telemetry coverage contract")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    verbs = contract.get("verbs", {})
    recording = sum(1 for d in verbs.values() if d == "records")
    print(
        f"PASS: telemetry coverage — {len(verbs)} verbs declared "
        f"({recording} recording); engine enforcement intact"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
