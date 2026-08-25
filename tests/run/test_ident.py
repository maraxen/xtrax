"""Tests for xtrax.run.ident: the stdlib-only run-id generator."""

from __future__ import annotations

import re

from xtrax.run.ident import new_run_id

RUN_ID_RE = re.compile(r"^run-[0-9a-f]{12}$")


def test_new_run_id_matches_canonical_format() -> None:
    """Generated ids are ``run-`` plus 12 lowercase hex chars (path/TOML-safe)."""
    assert RUN_ID_RE.match(new_run_id())


def test_new_run_id_distinct_consecutive() -> None:
    """Two consecutive generations never collide."""
    assert new_run_id() != new_run_id()


def test_new_run_id_hundred_unique() -> None:
    """Bulk uniqueness sanity: 100 draws, 100 distinct ids."""
    ids = {new_run_id() for _ in range(100)}
    assert len(ids) == 100
