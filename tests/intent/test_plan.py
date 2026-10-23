"""Tests for xtrax.intent.Plan, SiteDecision, and SiteKey."""

import json

import pytest

from xtrax.intent import PLAN_SCHEMA_VERSION, Context, Plan, SiteDecision, SiteKey
from xtrax.tiling import Bucket, ChunkedMap, Scan, StaticWhile, Vmap

dec_map = SiteDecision(
    key=SiteKey(parent=None, path="primal", label="model.py:20", ordinal=0),
    kind="map",
    primal=ChunkedMap(batch_size=32),
    diff=Vmap(),
    primal_reason="chunked map fits the memory budget",
    diff_reason="cotangent is a plain vmap",
)
_loop_parent = SiteKey(parent=None, path="primal", label="train.py:8", ordinal=0)
dec_loop = SiteDecision(
    key=SiteKey(parent=_loop_parent, path="diff", label="train.py:12", ordinal=1),
    kind="loop",
    primal=StaticWhile(length=8),
    diff=Scan(),
    primal_reason="fixed trip count",
    diff_reason="scan is reverse-mode safe",
)
p = Plan(PLAN_SCHEMA_VERSION, Context(), (dec_map, dec_loop))


def test_plan_json_round_trip():  # criterion 7 (serialisation)
    p = Plan(PLAN_SCHEMA_VERSION, Context(), (dec_map, dec_loop))
    assert Plan.from_json(p.to_json(), context=Context()) == p


def test_plan_table_lists_every_site_and_reason():
    t = p.table()
    assert str(dec_map.key) in t and dec_map.primal_reason in t and "ChunkedMap" in t
    assert str(dec_loop.key) in t and dec_loop.diff_reason in t and "StaticWhile" in t


def test_plan_is_hashable():
    assert hash(p) == hash(Plan.from_json(p.to_json(), context=Context()))


def test_unknown_schema_version_rejected():
    with pytest.raises(ValueError, match="schema_version"):
        Plan.from_json(
            p.to_json().replace('"schema_version": 1', '"schema_version": 99'),
            context=Context(),
        )


def test_site_decision_without_diff_round_trips():
    decision = SiteDecision(
        key=SiteKey(parent=None, path="primal", label="only.py:3", ordinal=1),
        kind="map",
        primal=Vmap(),
        diff=None,
        primal_reason="single batch fits",
        diff_reason="no cotangent requested",
        warnings=("narrow site",),
    )
    plan = Plan(PLAN_SCHEMA_VERSION, Context(), (decision,))
    restored = Plan.from_json(plan.to_json(), context=Context())
    assert restored == plan
    assert restored.decisions[0].diff is None
    assert restored.decisions[0].diff_reason == "no cotangent requested"


def test_site_key_str_renders_parent_path_label_ordinal():
    parent = SiteKey(parent=None, path="primal", label="outer.py:1", ordinal=0)
    key = SiteKey(parent=parent, path="diff", label="inner.py:4", ordinal=2)
    assert str(parent) == "primal/outer.py:1#0"
    assert str(key) == "primal/outer.py:1#0/diff/inner.py:4#2"


def test_scan_serialises_only_ordered_sinks():
    decision = SiteDecision(
        key=SiteKey(parent=None, path="primal", label="scan.py:2", ordinal=0),
        kind="loop",
        primal=Scan(ordered_sinks=False),
        diff=None,
        primal_reason="ordered sinks off",
        diff_reason="no cotangent requested",
    )
    raw = Plan(PLAN_SCHEMA_VERSION, Context(), (decision,)).to_json()
    assert '"ordered_sinks": false' in raw
    assert "transition" not in raw
    assert '"init"' not in raw
    restored = Plan.from_json(raw, context=Context())
    assert restored.decisions[0].primal == Scan(ordered_sinks=False)


def test_scan_with_callable_fields_cannot_serialise():
    key = SiteKey(parent=None, path="primal", label="scan.py:9", ordinal=0)
    with_transition = SiteDecision(
        key,
        "loop",
        Scan(transition=lambda carry, x: (carry, x)),
        None,
        "has transition",
        "no cotangent requested",
    )
    with_init = SiteDecision(
        key,
        "loop",
        Scan(init=0),
        None,
        "has init",
        "no cotangent requested",
    )
    for decision in (with_transition, with_init):
        plan = Plan(PLAN_SCHEMA_VERSION, Context(), (decision,))
        with pytest.raises(ValueError, match="Scan"):
            plan.to_json()


def test_unknown_strategy_type_names_the_type():
    decision = SiteDecision(
        key=SiteKey(parent=None, path="primal", label="bucket.py:1", ordinal=0),
        kind="map",
        primal=Bucket(boundaries=(4, 8)),
        diff=None,
        primal_reason="host buckets",
        diff_reason="no cotangent requested",
    )
    plan = Plan(PLAN_SCHEMA_VERSION, Context(), (decision,))
    with pytest.raises(ValueError, match="Bucket"):
        plan.to_json()


def test_from_json_uses_the_supplied_context():
    ctx = Context(chunk_size=8, backend="cpu")
    restored = Plan.from_json(p.to_json(), context=ctx)
    assert restored.context == ctx
    assert restored.decisions == p.decisions
    plan = Plan(PLAN_SCHEMA_VERSION, ctx, p.decisions)
    payload = json.loads(plan.to_json())
    assert payload["context"] == repr(plan.context)
    assert isinstance(payload["context"], str)


def test_by_key_indexes_every_decision():
    found = p.by_key()
    assert found[dec_map.key] == dec_map
    assert found[dec_loop.key] == dec_loop
    assert set(found) == {dec_map.key, dec_loop.key}


def test_by_key_rejects_duplicate_site_key():
    other = SiteDecision(
        key=dec_map.key,
        kind="map",
        primal=Vmap(),
        diff=None,
        primal_reason="second visit",
        diff_reason="no cotangent requested",
    )
    plan = Plan(PLAN_SCHEMA_VERSION, Context(), (dec_map, other))
    with pytest.raises(ValueError, match="duplicate SiteKey primal/model.py:20#0"):
        plan.by_key()


def _plan_payload() -> dict:
    return json.loads(p.to_json())


def test_from_json_unknown_strategy_type():
    payload = _plan_payload()
    payload["decisions"][0]["primal"]["type"] = "Bucket"
    with pytest.raises(ValueError, match="unknown strategy type 'Bucket'"):
        Plan.from_json(json.dumps(payload), context=Context())


def test_from_json_missing_strategy_type():
    payload = _plan_payload()
    del payload["decisions"][0]["primal"]["type"]
    with pytest.raises(ValueError, match="missing 'type'"):
        Plan.from_json(json.dumps(payload), context=Context())


def test_from_json_missing_decisions():
    payload = _plan_payload()
    del payload["decisions"]
    with pytest.raises(ValueError, match="missing 'decisions'"):
        Plan.from_json(json.dumps(payload), context=Context())


def test_from_json_missing_batch_size():
    payload = _plan_payload()
    del payload["decisions"][0]["primal"]["batch_size"]
    with pytest.raises(ValueError, match="missing 'batch_size'"):
        Plan.from_json(json.dumps(payload), context=Context())


def test_from_json_bad_path():
    payload = _plan_payload()
    payload["decisions"][0]["key"]["path"] = "nope"
    with pytest.raises(ValueError, match="path 'nope' is not 'primal' or 'diff'"):
        Plan.from_json(json.dumps(payload), context=Context())


def test_from_json_top_level_list():
    with pytest.raises(ValueError, match="plan JSON is not an object"):
        Plan.from_json("[]", context=Context())


def test_from_json_non_object_decision():
    payload = _plan_payload()
    payload["decisions"] = [None]
    with pytest.raises(ValueError, match="decision is not an object"):
        Plan.from_json(json.dumps(payload), context=Context())


def test_from_json_missing_key_field():
    payload = _plan_payload()
    del payload["decisions"][0]["key"]["label"]
    with pytest.raises(ValueError, match="missing 'label'"):
        Plan.from_json(json.dumps(payload), context=Context())


def test_from_json_invalid_text_is_json_decode_error():
    with pytest.raises(json.JSONDecodeError):
        Plan.from_json("{", context=Context())
