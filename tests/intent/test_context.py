"""Tests for xtrax.intent.Context and intent error types."""

import dataclasses
import pickle

import jax
import pytest

from xtrax.intent import Context, SiteMismatchError, UnsatisfiableContextError
from xtrax.tiling import StaticWhile
from xtrax.tiling.budget import MemoryBudget


def test_context_is_frozen_and_hashable():
    ctx = Context()
    assert hash(ctx) == hash(Context())
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.chunk_size = 8


def test_export_target_requires_format():
    with pytest.raises(ValueError, match="export_format"):
        Context(target="export")
    assert Context(target="export", export_format="stablehlo").target == "export"


def test_chunk_size_positive():
    with pytest.raises(ValueError, match="chunk_size"):
        Context(chunk_size=0)


def test_errors_carry_site_path_constraint():
    e = UnsatisfiableContextError(
        "no strategy", site="dec", path="primal", constraint="export forbids X"
    )
    assert (e.site, e.path, e.constraint) == ("dec", "primal", "export forbids X")
    assert "dec" in str(e) and "primal" in str(e) and "export forbids X" in str(e)


def test_static_while_is_frozen():
    sw = StaticWhile(length=4)
    assert sw == StaticWhile(length=4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        sw.length = 8


def test_context_with_memory_budget_is_hashable():
    budget = MemoryBudget(bytes=1024, estimate=lambda decisions: 0)
    assert hash(Context(memory_budget=budget)) == hash(Context(memory_budget=budget))


def test_resolved_backend_prefers_explicit_backend():
    assert Context(backend="cpu").resolved_backend() == "cpu"
    assert Context().resolved_backend() == jax.default_backend()


@pytest.mark.parametrize("cls", [SiteMismatchError, UnsatisfiableContextError])
def test_errors_survive_pickling(cls):
    e = cls("bad", site="dec", path="diff", constraint="c")
    r = pickle.loads(pickle.dumps(e))
    assert type(r) is cls and str(r) == str(e)
    assert (r.site, r.path, r.constraint) == ("dec", "diff", "c")
