"""Tests for the dependency-group/optional-extra alias contract (#4969, B3).

Root cause: `pyproject.toml` declares `dev` (and `eda`) in BOTH
`[dependency-groups]` and `[project.optional-dependencies]`, and the two copies had
drifted -- five packages (`beartype`, `chex`, `interrogate`, `jaxlint`, `libcst`)
existed only in the extra. `uv sync` is exact by default and prunes, so
`uv sync --group docs` kept the group and silently dropped those five (incident
#131).

The rule is deliberately NOT disjointness -- `dev`/`eda` are allowed, even
required, to appear in both tables. What's forbidden is the two copies diverging:
for any name present in both tables, the group's value must be exactly the
single-element alias `["xtrax[<name>]"]`. A name present in only one table (e.g.
`docs`, group-only) is unconstrained.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from scripts.audit_project_hygiene import check_group_extra_aliases

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = ROOT / "pyproject.toml"


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


def test_check_group_extra_aliases_flags_a_divergent_group() -> None:
    """A name declared in both tables whose group value is its own copy of the
    extra's packages, instead of the `xtrax[<name>]` alias, is flagged -- this is
    exactly the incident #131 shape: `dev`'s group used to list its own
    7-package copy alongside the extra's 15-package copy, and the two silently
    diverged.

    This is a synthetic fixture, not the real pyproject.toml: the real file was
    fixed by #4969, so a test that reads it can no longer exercise the flagging
    behavior -- it would trivially rot to "passes" the moment the fix landed
    (which it did). The checker's ability to flag a divergent input is the
    durable contract; this fixture keeps testing it regardless of the real
    repo's current state.
    """
    pyproject = {
        "dependency-groups": {"dev": ["complexipy>=5.6.1", "ruff>=0.15.22"]},
        "project": {
            "optional-dependencies": {
                "dev": ["complexipy>=5.6.1", "pytest>=8.0", "ruff>=0.4.0"],
            }
        },
    }

    failures = check_group_extra_aliases(pyproject)

    dev_failures = [item for item in failures if "].dev must be exactly" in item]
    assert len(dev_failures) == 1, failures
    assert "['xtrax[dev]']" in dev_failures[0]
    # The offending group's actual value must be visible in the message, so a
    # human/fixer can see exactly what's wrong without re-opening pyproject.toml.
    assert "complexipy" in dev_failures[0]


def test_check_group_extra_aliases_flags_each_divergent_name_independently() -> None:
    """Multiple divergent names in the same pyproject.toml each get their own,
    independently-identifiable failure -- e.g. both `dev` and `eda` diverging at
    once (the real #131 shape, before the #4969 fix) produces two failures, one
    per name, not a single combined one.
    """
    pyproject = {
        "dependency-groups": {
            "dev": ["ruff>=0.4.0"],
            "eda": ["pandas>=2.0", "matplotlib>=3.8"],
        },
        "project": {
            "optional-dependencies": {
                "dev": ["ruff>=0.4.0", "pytest>=8.0"],
                "eda": ["pandas>=2.0", "matplotlib>=3.8", "seaborn>=0.13"],
            }
        },
    }

    failures = check_group_extra_aliases(pyproject)

    assert len(failures) == 2, failures
    assert any("].dev must be exactly" in item and "['xtrax[dev]']" in item for item in failures), (
        failures
    )
    assert any("].eda must be exactly" in item and "['xtrax[eda]']" in item for item in failures), (
        failures
    )


def test_check_group_extra_aliases_ignores_group_only_name() -> None:
    """`docs` exists only in [dependency-groups] -- it is unconstrained by this
    contract and must never appear in its failures.
    """
    failures = check_group_extra_aliases(_pyproject())

    assert not any("].docs " in item for item in failures), failures


def test_check_group_extra_aliases_accepts_the_alias_shape() -> None:
    """A name declared in both tables as `xtrax[<name>]` in the group is fine --
    this is the shape #4969's landing fix converges `dev`/`eda` to.
    """
    pyproject = {
        "dependency-groups": {"dev": ["xtrax[dev]"], "docs": ["sphinx>=7"]},
        "project": {
            "optional-dependencies": {
                "dev": ["pytest>=8.0", "ruff>=0.4.0"],
            }
        },
    }

    failures = check_group_extra_aliases(pyproject)

    assert failures == []


def test_check_group_extra_aliases_rejects_partial_alias_list() -> None:
    """A group value that merely CONTAINS the alias among other entries is not the
    contract -- it must be exactly the single-element list.
    """
    pyproject = {
        "dependency-groups": {"dev": ["xtrax[dev]", "ruff>=0.4.0"]},
        "project": {"optional-dependencies": {"dev": ["ruff>=0.4.0"]}},
    }

    failures = check_group_extra_aliases(pyproject)

    assert len(failures) == 1
    assert "].dev must be exactly" in failures[0]


def test_check_group_extra_aliases_passes_on_repo() -> None:
    """The real pyproject.toml's `dev`/`eda` groups are now the self-referential
    aliases `["xtrax[dev]"]` / `["xtrax[eda]"]` (landed via #4969) -- the #131
    mechanism, two independently-maintained copies silently diverging, is
    structurally closed: a self-referential alias has no content of its own to
    diverge with.

    This checks `check_group_extra_aliases` directly rather than going through
    `audit_project_hygiene()` -- the overall-pass assertion for the whole gate
    already lives in `tests/distribution/test_project_hygiene.py`
    (`test_audit_project_hygiene_passes_on_repo`); duplicating it here would add
    nothing. This test is narrower and more useful: it isolates the B3
    alias-contract's own zero-failures state from the other checks the gate also
    runs.
    """
    failures = check_group_extra_aliases(_pyproject())

    assert failures == [], failures
