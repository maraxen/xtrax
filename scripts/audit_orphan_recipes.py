#!/usr/bin/env python3
"""Enumerate `audit-*` Justfile recipes that no CI workflow reaches (#5001).

This repo has 65 `audit-*` recipes and CI invokes six entry points. Everything the
six do not reach transitively is a gate that exists, is maintained, and is never run
-- so it is green in exactly the way an unplugged smoke detector is silent. Three
such gates were found broken this week alone once someone finally ran them:
`audit-jax-purity-gate` had a dead ruff path, `audit-docs-build` was both orphaned
and silently corrupting the shared venv, and `audit-release-readiness` had been red
since 2026-07-02 with nothing saying so.

The point of this script is to make that set *derived* rather than *listed*. A
hand-maintained exemption list of thirty untriaged entries would go stale the moment
someone added a recipe, and nothing would fail when it did. Here, a recipe that gets
wired into a workflow drops out of the list automatically, and a newly added one
appears in it automatically -- no edit to this file, and no way to forget.

## What counts as reachable

Entry points are discovered by scanning `.github/workflows/*.yml` for `just <recipe>`
invocations, not hardcoded. From those, reachability follows two kinds of edge:

  1. declared prerequisites (`audit-foundation: audit-imports ...`)
  2. `just <recipe>` lines inside a recipe body

The second matters more than it looks. `audit-deterministic` declares 19
prerequisites but its *body* calls 11 further recipes, so reading only the
dependency line understates CI coverage by more than a third. An earlier count in
this sprint made exactly that error.

## What this script does NOT claim

It reports which recipes are unreached. It says nothing about whether they pass --
that is the scheduled workflow's job (`.github/workflows/audit-orphans.yml`), and
until that workflow has run at least once, nobody knows. No expected-status data is
baked in here, because there is none yet; inventing it would be the same unfounded
green this script exists to expose.

## The one thing it does enforce

Every recipe a workflow names must exist in the Justfile. A workflow calling a
recipe that was renamed or deleted fails the job with an error nobody reads as
"gate missing", so this checks it directly and exits 1. Verified against the current
tree at the time of writing: all six entry points resolve.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = ROOT / ".github" / "workflows"

# `just <recipe>` as a shell word, on ONE line. The separator is `[^\S\n]` (horizontal
# whitespace) rather than `\s`, which matters: a `\s+` version of this pattern matched
# across the newline in
#
#     - name: Install just
#       run: uv tool install rust-just
#
# and reported a phantom `run` entry point from the prose word "just". Requiring the
# recipe name on the same line also rules out `run:` and other YAML keys, since the
# colon terminates the capture. A leading `-` is excluded so flags (`just --list`) are
# not read as recipe names.
JUST_CALL = re.compile(r"(?:^|[^\S\n]|[;&|(])just[^\S\n]+(?!-)([A-Za-z0-9_][A-Za-z0-9_-]*)\b")

AUDIT_PREFIX = "audit-"

# The scheduled workflow that RUNS the orphans must not count as coverage for them,
# or the list would empty itself out and the run would become a no-op that still
# looks busy. It invokes them through a shell variable today, so it contributes no
# entry points by accident -- but that is a property of one line of bash, not a
# guarantee. Excluding the file by name makes it structural.
SELF_WORKFLOW = "audit-orphans.yml"


def load_recipes(root: Path = ROOT) -> dict[str, dict]:
    """Return `just --dump` recipe records keyed by name.

    Uses just's own JSON dump rather than parsing the Justfile. Regex-parsing
    recipe headers gets dependency lists wrong on continuation lines and on
    parameterised recipes, and just already exposes the parsed form.
    """
    result = subprocess.run(
        ["just", "--dump", "--dump-format", "json"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or "just --dump failed"
        raise RuntimeError(f"could not dump Justfile: {stderr}")
    return json.loads(result.stdout)["recipes"]


def _body_calls(recipe: dict) -> set[str]:
    """Recipe names invoked as `just X` from inside a recipe body."""
    calls: set[str] = set()
    for line in recipe.get("body", []):
        # A body line is a list of fragments; interpolations are dicts, not strings.
        text = "".join(part for part in line if isinstance(part, str))
        calls.update(JUST_CALL.findall(text))
    return calls


def discover_ci_entrypoints(workflows_dir: Path = WORKFLOWS_DIR) -> dict[str, set[str]]:
    """Map recipe name -> set of workflow filenames that invoke it.

    Deliberately a text scan rather than a YAML parse: `run:` blocks are shell, and
    a recipe invocation can appear inside a multi-line block, a conditional, or a
    composite string. Scanning text catches all of those; the cost is that a recipe
    name appearing in a YAML comment would count as an entry point. That errs
    towards under-reporting orphans, which is the safe direction -- a false orphan
    would send someone chasing a gate that is in fact wired up.
    """
    found: dict[str, set[str]] = {}
    if not workflows_dir.is_dir():
        return found
    for path in sorted(workflows_dir.glob("*.yml")) + sorted(workflows_dir.glob("*.yaml")):
        if path.name == SELF_WORKFLOW:
            continue
        for name in JUST_CALL.findall(path.read_text(encoding="utf-8")):
            found.setdefault(name, set()).add(path.name)
    return found


def reachable_from(recipes: dict[str, dict], entrypoints: set[str]) -> set[str]:
    """Transitive closure over prerequisite edges and in-body `just X` calls."""
    seen: set[str] = set()
    stack = [name for name in entrypoints if name in recipes]
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        recipe = recipes[name]
        successors = {dep["recipe"] for dep in recipe.get("dependencies", [])}
        successors |= _body_calls(recipe)
        stack.extend(succ for succ in successors if succ in recipes and succ not in seen)
    return seen


def requires_arguments(recipe: dict) -> bool:
    """True if the recipe cannot be invoked as a bare `just <name>`.

    `star` (`*args`) and defaulted parameters are fine; a bare `singular` or a
    `plus` (`+args`) parameter is not.
    """
    for param in recipe.get("parameters", []):
        if param.get("default") is not None:
            continue
        if param.get("kind") in ("singular", "plus"):
            return True
    return False


def unreferenced_workflow_recipes(
    recipes: dict[str, dict], entrypoints: dict[str, set[str]]
) -> list[tuple[str, set[str]]]:
    """Recipes a workflow invokes that the Justfile does not define."""
    return sorted(
        (name, workflows) for name, workflows in entrypoints.items() if name not in recipes
    )


def orphan_audit_recipes(
    recipes: dict[str, dict], covered: set[str]
) -> tuple[list[str], list[str]]:
    """Split unreached `audit-*` recipes into runnable and argument-requiring."""
    runnable: list[str] = []
    needs_args: list[str] = []
    for name in sorted(recipes):
        if not name.startswith(AUDIT_PREFIX) or name in covered:
            continue
        if requires_arguments(recipes[name]):
            needs_args.append(name)
        else:
            runnable.append(name)
    return runnable, needs_args


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root")
    parser.add_argument(
        "--format",
        choices=("lines", "json", "report"),
        default="report",
        help="lines: one runnable orphan per line (for shell loops); "
        "json: full detail; report: human-readable summary",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    recipes = load_recipes(root)
    entrypoints = discover_ci_entrypoints(root / ".github" / "workflows")
    dangling = unreferenced_workflow_recipes(recipes, entrypoints)
    covered = reachable_from(recipes, set(entrypoints))
    runnable, needs_args = orphan_audit_recipes(recipes, covered)

    audit_total = sum(1 for name in recipes if name.startswith(AUDIT_PREFIX))
    audit_covered = sum(1 for name in covered if name.startswith(AUDIT_PREFIX))

    if args.format == "lines":
        for name in runnable:
            print(name)
    elif args.format == "json":
        print(
            json.dumps(
                {
                    "entrypoints": {k: sorted(v) for k, v in sorted(entrypoints.items())},
                    "audit_total": audit_total,
                    "audit_covered": audit_covered,
                    "orphans": runnable,
                    "orphans_needing_arguments": needs_args,
                    "dangling_workflow_recipes": {k: sorted(v) for k, v in dangling},
                },
                indent=2,
            )
        )
    else:
        print(f"CI entry points ({len(entrypoints)}):")
        for name, workflows in sorted(entrypoints.items()):
            print(f"  {name}  <- {', '.join(sorted(workflows))}")
        print(f"\naudit-* recipes: {audit_total} total, {audit_covered} reached by CI")
        print(f"\nUnreached and runnable ({len(runnable)}):")
        for name in runnable:
            print(f"  {name}")
        if needs_args:
            print(f"\nUnreached, needs arguments ({len(needs_args)}):")
            for name in needs_args:
                print(f"  {name}")

    if dangling:
        print("\nFAIL: workflow invokes recipes the Justfile does not define", file=sys.stderr)
        for name, workflows in dangling:
            print(f"  - {name} (in {', '.join(sorted(workflows))})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
