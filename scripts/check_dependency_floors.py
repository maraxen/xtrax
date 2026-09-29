"""Assert every declared dependency floor is the version actually installed.

Run inside an environment resolved with ``uv pip install --resolution
lowest-direct``. There, each direct dependency should land exactly on its
``>=`` lower bound; one that lands higher has a floor the resolver cannot reach
(no wheel for the supported Python, or a transitive constraint that forbids it),
so the declared floor is a claim nothing ever tests. Both cases fail loud.

Usage::

    python scripts/check_dependency_floors.py [--extra io --extra cli]

Exit 0 when every floored requirement (core + the named extras) is installed at
its floor; exit 1 listing each mismatch or missing distribution.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tomllib
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def declared_floors(pyproject: dict, extras: list[str]) -> dict[str, Version]:
    """Map each requirement name to its ``>=`` floor; requirements without one are skipped."""
    project = pyproject["project"]
    requirements = list(project.get("dependencies", []))
    optional = project.get("optional-dependencies", {})
    for extra in extras:
        if extra not in optional:
            msg = f"unknown extra {extra!r}; declared: {sorted(optional)}"
            raise SystemExit(msg)
        requirements.extend(optional[extra])
    floors: dict[str, Version] = {}
    for text in requirements:
        req = Requirement(text)
        if req.name.lower() == "xtrax":  # self-referencing extras (e.g. "xtrax[cli]")
            continue
        lower = [Version(s.version) for s in req.specifier if s.operator == ">="]
        if lower:
            floors[req.name.lower()] = max(lower)
    return floors


def floor_mismatches(floors: dict[str, Version], installed: dict[str, str | None]) -> list[str]:
    """Describe every floored requirement not installed at exactly its floor."""
    problems = []
    for name, floor in sorted(floors.items()):
        got = installed.get(name)
        if got is None:
            problems.append(f"{name}: declared >={floor} but not installed")
        elif Version(got) != floor:
            problems.append(
                f"{name}: declared >={floor} but lowest-direct resolved {got} -- the floor is "
                "unreachable; raise it to the lowest version that installs AND passes"
            )
    return problems


def _installed(names: list[str]) -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    for name in names:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--extra", action="append", default=[], help="also check this extra")
    parser.add_argument("--pyproject", type=Path, default=ROOT / "pyproject.toml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    floors = declared_floors(tomllib.loads(args.pyproject.read_text(encoding="utf-8")), args.extra)
    problems = floor_mismatches(floors, _installed(list(floors)))
    for name, floor in sorted(floors.items()):
        logger.info("floor %s>=%s", name, floor)
    if problems:
        for problem in problems:
            logger.error("FAIL %s", problem)
        return 1
    logger.info("PASS: all %d floors installed exactly", len(floors))
    return 0


if __name__ == "__main__":
    sys.exit(main())
