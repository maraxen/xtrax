"""Tests for the intent site session."""

import inspect

import jax.numpy as jnp
import pytest
from jax import ShapeDtypeStruct

from xtrax.intent import (
    Session,
    SiteDecision,
    SiteKey,
    SiteMismatchError,
    SiteRecord,
    activate,
    call_label,
    current_session,
)
from xtrax.tiling import Vmap


def sds(shape: tuple[int, ...]) -> ShapeDtypeStruct:
    return ShapeDtypeStruct(shape, jnp.float32)


def test_same_line_three_shapes_gives_three_ordinals():
    s = Session(mode="plan")
    with activate(s):
        keys = [s.enter_site("map", "m.py:10", (sds((n, 4)),), None, None) for n in (8, 64, 512)]
    assert [k.ordinal for k in keys] == [0, 1, 2] and len(set(keys)) == 3


def test_nested_scope_qualifies_path_and_parent():
    s = Session(mode="plan")
    with activate(s):
        outer = s.enter_site("loop", "m.py:1", (sds((4,)),), 16, None)
        with s.scope(outer, "diff"):
            inner = s.enter_site("map", "m.py:2", (sds((8, 4)),), None, None)
    assert inner.parent == outer and inner.path == "diff"


def test_exec_mode_rejects_unplanned_and_changed_avals():
    plan = Session(mode="plan")
    with activate(plan):
        plan.enter_site("map", "m.py:10", (sds((16, 4)),), None, None)
    run = Session(mode="exec", records=plan.records, decisions={})
    with activate(run), pytest.raises(SiteMismatchError, match="planned"):
        run.enter_site("map", "m.py:10", (sds((15, 4)),), None, None)


def test_no_session_outside_activate():
    assert current_session() is None


def test_current_session_default_used_only_when_inactive():
    fallback = Session(mode="plan")
    assert current_session(default=fallback) is fallback
    active = Session(mode="exec")
    with activate(active):
        assert current_session(default=fallback) is active


def test_exec_mode_unplanned_key_names_site_not_in_plan():
    run = Session(mode="exec", records=[], decisions={})
    with activate(run), pytest.raises(SiteMismatchError, match="site not in plan") as exc:
        run.enter_site("map", "m.py:10", (sds((16, 4)),), None, None)
    err = exc.value
    assert err.constraint == "site not in plan"
    assert err.site == "primal/m.py:10#0"
    assert err.path == "primal"
    assert run.records == []


def test_exec_mode_matching_avals_returns_planned_key():
    plan = Session(mode="plan")
    with activate(plan):
        first = plan.enter_site("map", "m.py:10", (sds((16, 4)),), None, None)
        second = plan.enter_site("map", "m.py:10", (sds((8, 4)),), None, None)
    planned = list(plan.records)
    run = Session(mode="exec", records=plan.records, decisions={})
    with activate(run):
        assert run.enter_site("map", "m.py:10", (sds((16, 4)),), None, None) == first
        assert run.enter_site("map", "m.py:10", (sds((8, 4)),), None, None) == second
    assert run.records == planned


def test_ordinals_are_per_parent_path_and_label():
    session = Session(mode="plan")
    with activate(session):
        first = session.enter_site("map", "m.py:1", (sds((2,)),), None, None)
        other = session.enter_site("map", "m.py:2", (sds((2,)),), None, None)
        again = session.enter_site("map", "m.py:1", (sds((3,)),), None, None)
    assert (first.ordinal, other.ordinal, again.ordinal) == (0, 0, 1)
    assert first.parent is None and first.path == "primal"


def test_plan_mode_appends_site_record():
    seen: list[int] = []

    def site_fn(value: int) -> int:
        seen.append(value)
        return value

    session = Session(mode="plan")
    with activate(session):
        key = session.enter_site("loop", "m.py:1", (sds((4,)),), 16, site_fn)
    assert len(session.records) == 1
    record = session.records[0]
    assert record.key == key
    assert record.kind == "loop"
    assert record.length == 16
    assert record.fn is site_fn
    assert record.avals == (sds((4,)),)


def test_exec_mode_rejects_aval_count_mismatch():
    plan = Session(mode="plan")
    with activate(plan):
        plan.enter_site("map", "m.py:10", (sds((16, 4)),), None, None)
    run = Session(mode="exec", records=plan.records, decisions={})
    traced = (sds((16, 4)), sds((4,)))
    with activate(run), pytest.raises(SiteMismatchError, match="planned") as exc:
        run.enter_site("map", "m.py:10", traced, None, None)
    assert "traced" in exc.value.constraint
    assert exc.value.path == "primal"


def test_exec_mode_rejects_dtype_mismatch():
    plan = Session(mode="plan")
    planned = (ShapeDtypeStruct((16, 4), jnp.float32),)
    with activate(plan):
        plan.enter_site("map", "m.py:10", planned, None, None)
    run = Session(mode="exec", records=plan.records, decisions={})
    traced = (ShapeDtypeStruct((16, 4), jnp.int32),)
    with activate(run), pytest.raises(SiteMismatchError, match="planned") as exc:
        run.enter_site("map", "m.py:10", traced, None, None)
    assert "traced" in exc.value.constraint


def test_activate_nests_and_restores():
    outer = Session(mode="plan")
    inner = Session(mode="plan")
    assert current_session() is None
    with activate(outer):
        assert current_session() is outer
        with pytest.raises(RuntimeError, match="boom"):
            with activate(inner):
                assert current_session() is inner
                raise RuntimeError("boom")
        assert current_session() is outer
    assert current_session() is None


def test_scopes_nest_as_a_stack():
    session = Session(mode="plan")
    with activate(session):
        outer = session.enter_site("loop", "m.py:1", (sds((4,)),), 4, None)
        with session.scope(outer, "diff"):
            mid = session.enter_site("map", "m.py:2", (sds((8,)),), None, None)
            with session.scope(mid, "primal"):
                inner = session.enter_site("map", "m.py:3", (sds((2,)),), None, None)
            after_inner = session.enter_site("map", "m.py:4", (sds((2,)),), None, None)
        after_outer = session.enter_site("map", "m.py:5", (sds((2,)),), None, None)
    assert inner.parent == mid and inner.path == "primal"
    assert after_inner.parent == outer and after_inner.path == "diff"
    assert after_outer.parent is None and after_outer.path == "primal"


def test_scope_pops_after_exception():
    session = Session(mode="plan")
    with activate(session):
        outer = session.enter_site("loop", "m.py:1", (sds((4,)),), 16, None)
        with pytest.raises(RuntimeError, match="scope-fail"):
            with session.scope(outer, "diff"):
                raise RuntimeError("scope-fail")
        after = session.enter_site("map", "m.py:3", (sds((2,)),), None, None)
    assert after.parent is None
    assert after.path == "primal"
    assert after.ordinal == 0


def test_decision_for_with_and_without_decisions():
    key = SiteKey(parent=None, path="primal", label="m.py:1", ordinal=0)
    decision = SiteDecision(
        key=key,
        kind="map",
        primal=Vmap(),
        diff=None,
        primal_reason="fits",
        diff_reason="no cotangent",
    )
    with_decisions = Session(mode="plan", decisions={key: decision})
    assert with_decisions.decision_for(key) is decision
    missing = SiteKey(parent=None, path="primal", label="m.py:2", ordinal=0)
    assert with_decisions.decision_for(missing) is None
    without = Session(mode="exec")
    assert without.decision_for(key) is None
    assert without.records == []
    assert with_decisions.records is not without.records


def test_call_label_uses_name_or_caller_file_line():
    assert call_label("step", 1) == "step"
    here = call_label(None, 1)
    assert here == f"test_sites.py:{inspect.currentframe().f_lineno - 1}"


def test_site_record_equality_ignores_fn():
    key = SiteKey(parent=None, path="primal", label="m.py:1", ordinal=0)
    avals = (sds((4,)),)
    left = SiteRecord(key, "map", avals, None, lambda x: x)
    right = SiteRecord(key, "map", avals, None, lambda x: x + 1)
    assert left == right
    assert hash(left) == hash(right)
    assert left != SiteRecord(key, "loop", avals, None, lambda x: x)
