"""Every repo-relative path a Justfile recipe names must exist.

A recipe that names a missing file fails the moment it runs -- and if no gate runs that
recipe, nothing ever says so. `audit-jax-purity-gate` sat broken exactly that way: its
ruff line pointed at `src/xtrax/devtools/gates/_jaxlint.py` long after commit 0871837
renamed it to `src/xtrax/jaxlint_runner.py`, and because the recipe was in no chain, the
board stayed green over a recipe that could not run at all.

This test is cheap and total, so the class of failure closes rather than the one instance.
It lives in tests/audit/ because `audit-deterministic` runs that directory wholesale, which
is the property that makes it un-rottable.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JUSTFILE = ROOT / "Justfile"

# Directories whose contents are real, committed repo paths. Deliberately not a bare
# "anything that looks like a path": recipe lines also carry flag values, package names
# and URLs, and matching those would make this test noise rather than signal.
_TRACKED_DIRS = ("src", "tests", "scripts", "port", "benchmarks", "distribution")
_PATH_RE = re.compile(
    r"\b(?:" + "|".join(_TRACKED_DIRS) + r")/[A-Za-z0-9_./-]+\.(?:py|toml|json|md|yaml|yml)\b"
)


def _referenced_paths() -> set[str]:
    return set(_PATH_RE.findall(JUSTFILE.read_text(encoding="utf-8")))


def test_every_justfile_path_exists() -> None:
    referenced = _referenced_paths()
    assert referenced, "path regex matched nothing -- the test, not the Justfile, is broken"

    missing = sorted(path for path in referenced if not (ROOT / path).exists())

    assert not missing, (
        "Justfile recipes name files that do not exist:\n  "
        + "\n  ".join(missing)
        + "\nEither the file moved (update the recipe) or the recipe is dead (delete it)."
    )


def test_regex_would_catch_a_missing_path() -> None:
    """Negative control: prove the assertion above can actually fail.

    A test that only ever sees a passing repo cannot distinguish "all paths exist" from
    "the regex matched nothing useful" -- the exact vacuous-green shape this file exists
    to prevent, so it is worth one synthetic case.
    """
    sample = "    uv run ruff check src/xtrax/does_not_exist_xyz.py\n"
    found = _PATH_RE.findall(sample)

    assert found == ["src/xtrax/does_not_exist_xyz.py"]
    assert not (ROOT / found[0]).exists()
