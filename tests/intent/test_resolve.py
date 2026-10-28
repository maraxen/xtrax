"""Tests for xtrax.intent.resolve."""

import jax
import jax.numpy as jnp
import pytest

from xtrax.intent import (
    PLAN_SCHEMA_VERSION,
    Context,
    IntentError,
    SiteKey,
    UnsatisfiableContextError,
    resolve,
)
from xtrax.intent._resolve import default_estimator
from xtrax.intent.sites import SiteRecord
from xtrax.tiling.budget import MemoryBudget
from xtrax.tiling.estimators import lowered_memory_estimate
from xtrax.tiling.plan import AxisDecision, AxisSpec, BatchPlanner
from xtrax.tiling.strategy import ChunkedMap, Scan, StaticWhile, Vmap, WhileCarry
from xtrax.transforms.map import chunked_map


def _key(label: str, *, parent: SiteKey | None = None, path: str = "primal") -> SiteKey:
    return SiteKey(parent=parent, path=path, label=label, ordinal=0)


def _map(
    label: str,
    cardinality: int,
    *,
    parent: SiteKey | None = None,
    path: str = "primal",
    fn=None,
) -> SiteRecord:
    return SiteRecord(
        key=_key(label, parent=parent, path=path),
        kind="map",
        avals=(jax.ShapeDtypeStruct((cardinality, 3), jnp.float32),),
        length=None,
        fn=fn,
    )


def _loop(
    label: str,
    length: int | None,
    *,
    parent: SiteKey | None = None,
    path: str = "primal",
) -> SiteRecord:
    return SiteRecord(
        key=_key(label, parent=parent, path=path),
        kind="loop",
        avals=(),
        length=length,
        fn=None,
    )


def axis_specs_for(records, chunk_size: int) -> list[AxisSpec]:
    """Map specs the resolver hands to BatchPlanner, in record order."""
    specs = []
    for record in records:
        if record.kind != "map" or record.key.path != "primal":
            continue
        cardinality = int(record.avals[0].shape[0])
        if cardinality <= 1:
            continue
        specs.append(
            AxisSpec(
                name=str(record.key),
                cardinality=cardinality,
                default_batch_size=chunk_size,
            )
        )
    return specs


def test_no_budget_all_maps_vmap_loops_static_while():
    """A missing budget keeps every map on Vmap and every fixed loop on StaticWhile.

    Fails if a map whose cardinality exceeds chunk_size is chunked without a budget,
    or if a fixed-length loop is planned as Scan.
    """
    records = [_map("batch", 64), _loop("steps", 8)]
    plan = resolve(records, Context())
    assert plan.schema_version == PLAN_SCHEMA_VERSION
    assert plan.context == Context()
    assert [decision.key for decision in plan.decisions] == [record.key for record in records]
    assert plan.decisions[0].primal == Vmap()
    assert plan.decisions[0].diff == Vmap()
    assert "same strategy" in plan.decisions[0].diff_reason
    assert plan.decisions[1].primal == StaticWhile(length=8)
    assert plan.decisions[1].diff == Scan()


def test_budget_matches_batchplanner_joint_mode():
    """Joint mode demotes map sites in record order, matching BatchPlanner."""
    records = [_map("first", 64), _loop("steps", 4), _map("second", 64)]

    def est(decisions):
        return sum(64 if isinstance(d.strategy, Vmap) else 8 for d in decisions)

    budget = MemoryBudget(bytes=100, estimate=est)
    plan = resolve(records, Context(memory_budget=budget, chunk_size=32))
    expected = BatchPlanner(budget=budget).plan(axis_specs_for(records, chunk_size=32))
    assert [d.primal for d in plan.decisions if d.kind == "map"] == [
        d.strategy for d in expected.decisions
    ]
    assert plan.decisions[1].primal == StaticWhile(length=4)


def test_infeasible_budget_names_site():
    """An unfittable joint plan names the largest map site and the planner message."""
    records = [_map("small", 8), _map("large", 128)]
    budget = MemoryBudget(bytes=1, estimate=lambda decisions: 100)
    with pytest.raises(UnsatisfiableContextError) as exc_info:
        resolve(records, Context(memory_budget=budget, chunk_size=4))
    err = exc_info.value
    assert err.path == "primal"
    assert err.site == "primal/large#0"
    assert "100" in err.constraint
    assert "1" in err.constraint


def test_export_non_stablehlo_loop_resolves_scan():
    """Export other than stablehlo plans a fixed loop as Scan on both paths."""
    plan = resolve([_loop("body", 4)], Context(target="export", export_format="onnx"))
    assert plan.decisions[0].primal == Scan()
    assert plan.decisions[0].diff == Scan()


def test_export_stablehlo_loop_keeps_static_while():
    """StableHLO export keeps a fixed loop on StaticWhile."""
    plan = resolve([_loop("body", 4)], Context(target="export", export_format="stablehlo"))
    assert plan.decisions[0].primal == StaticWhile(length=4)
    assert plan.decisions[0].diff == Scan()


def test_export_cond_loop_refused():
    """Export refuses a cond loop on every format, including stablehlo."""
    with pytest.raises(UnsatisfiableContextError) as exc_info:
        resolve([_loop("body", None)], Context(target="export", export_format="stablehlo"))
    err = exc_info.value
    assert err.constraint == "export forbids unbounded while_loop"
    assert err.path == "primal"
    assert "body" in err.site


def test_diff_unavailable_recorded():
    """A fixed loop with no diff path stores None and the caller-supplied reason."""
    reason = "cotangent was not requested"
    plan = resolve(
        [_loop("body", 4)],
        Context(),
        diff_available=False,
        diff_unavailable_reason=reason,
    )
    assert plan.decisions[0].primal == StaticWhile(length=4)
    assert plan.decisions[0].diff is None
    assert plan.decisions[0].diff_reason == reason


def test_scan_around_while_prefers_vmap_on_cuda():
    """On cuda, a chunked outer map around StaticWhile is put back on Vmap when it fits.

    The planner's own joint result for this budget is ChunkedMap. The hazard is what
    restores Vmap. Fails if the outer map stays chunked, or if the other map is restored too.
    """
    outer = _map("outer", 64)
    inner = _loop("inner", 4, parent=outer.key)
    other = _map("other", 64)
    records = [outer, inner, other]

    def est(decisions):
        total = 0
        for decision in decisions:
            vmap = isinstance(decision.strategy, Vmap)
            if decision.spec.name == str(outer.key):
                total += 30 if vmap else 20
            else:
                total += 40 if vmap else 10
        return total

    budget = MemoryBudget(bytes=50, estimate=est)
    planned = BatchPlanner(budget=budget).plan(axis_specs_for(records, 32))
    assert isinstance(planned.decisions[0].strategy, ChunkedMap)
    assert isinstance(planned.decisions[1].strategy, ChunkedMap)
    plan = resolve(records, Context(memory_budget=budget, chunk_size=32, backend="cuda"))
    assert plan.decisions[0].primal == Vmap()
    assert plan.decisions[0].diff == Vmap()
    assert plan.decisions[1].primal == StaticWhile(length=4)
    assert plan.decisions[2].primal == ChunkedMap(batch_size=32)


def test_scan_around_while_errors_when_vmap_exceeds_budget():
    """On cuda, the same hazard is an error when restoring Vmap does not fit."""
    outer = _map("outer", 64)
    records = [outer, _loop("inner", 4, parent=outer.key)]

    def est(decisions):
        return sum(100 if isinstance(d.strategy, Vmap) else 10 for d in decisions)

    budget = MemoryBudget(bytes=50, estimate=est)
    with pytest.raises(UnsatisfiableContextError) as exc_info:
        resolve(records, Context(memory_budget=budget, chunk_size=32, backend="cuda"))
    assert exc_info.value.constraint == "scan_around_while"
    assert "outer" in exc_info.value.site


def _two_while_maps_and_a_big_map():
    """Maps a and b each enclose a StaticWhile; map c encloses none and costs 1000 on Vmap."""
    a = _map("a", 64)
    b = _map("b", 64)
    records = [a, _loop("a_inner", 4, parent=a.key), b, _loop("b_inner", 4, parent=b.key)]
    records.append(_map("c", 64))
    costs = {"a": (100, 90), "b": (100, 90), "c": (1000, 10)}

    def est(decisions):
        total = 0
        for decision in decisions:
            vmap_cost, chunk_cost = costs[decision.spec.name.split("/")[-1].split("#")[0]]
            total += vmap_cost if isinstance(decision.strategy, Vmap) else chunk_cost
        return total

    return records, est


def test_scan_around_while_joint_budget_names_the_site_that_does_not_fit():
    """Switches are judged against the budget the earlier switches left, not one at a time.

    Planner result: a, b, c all ChunkedMap (90+90+10 = 190 <= 200). Switching a alone
    costs 100+90+10 = 200 and fits. Switching b as well costs 210. Judged alone each
    switch fits, so both used to switch and the plan exceeded the budget with no error (#6157).
    """
    records, est = _two_while_maps_and_a_big_map()
    budget = MemoryBudget(bytes=200, estimate=est)
    with pytest.raises(UnsatisfiableContextError) as exc_info:
        resolve(records, Context(memory_budget=budget, chunk_size=32, backend="cuda"))
    err = exc_info.value
    assert err.constraint == "scan_around_while"
    assert err.path == "primal"
    assert err.site == str(records[2].key)


def test_scan_around_while_switches_every_site_that_jointly_fits():
    """With 210 bytes both switches fit together, and the final plan stays within budget."""
    records, est = _two_while_maps_and_a_big_map()
    budget = MemoryBudget(bytes=210, estimate=est)
    plan = resolve(records, Context(memory_budget=budget, chunk_size=32, backend="cuda"))
    by_label = {d.key.label: d for d in plan.decisions}
    assert by_label["a"].primal == Vmap()
    assert by_label["b"].primal == Vmap()
    assert by_label["c"].primal == ChunkedMap(batch_size=32)
    assert 100 + 100 + 10 <= budget.bytes


def test_scan_around_while_budget_uses_overridden_sibling_strategies():
    """The remaining budget is computed from the strategies that will run, overrides included.

    At 1189 bytes the planner keeps c on Vmap (a, b chunked: 90+90+1000 = 1180), so a
    switch looked like 100+90+1000 > 1189. The caller overrides c to ChunkedMap, which
    frees 990 bytes: both switches then fit (100+100+10 = 210). Judging against the planner's
    c refused a switch that the final plan could afford.
    """
    records, est = _two_while_maps_and_a_big_map()
    budget = MemoryBudget(bytes=1189, estimate=est)
    plan = resolve(
        records,
        Context(memory_budget=budget, chunk_size=32, backend="cuda"),
        overrides={"c": (ChunkedMap(batch_size=32), ChunkedMap(batch_size=32))},
    )
    by_label = {d.key.label: d for d in plan.decisions}
    assert by_label["a"].primal == Vmap()
    assert by_label["b"].primal == Vmap()
    assert by_label["c"].primal == ChunkedMap(batch_size=32)


def test_scan_around_while_does_not_apply_off_cuda():
    """A non-cuda backend keeps the planner's ChunkedMap around a StaticWhile."""
    outer = _map("outer", 64)
    records = [outer, _loop("inner", 4, parent=outer.key)]

    def est(decisions):
        return sum(100 if isinstance(d.strategy, Vmap) else 10 for d in decisions)

    budget = MemoryBudget(bytes=50, estimate=est)
    plan = resolve(records, Context(memory_budget=budget, chunk_size=32, backend="cpu"))
    assert plan.decisions[0].primal == ChunkedMap(batch_size=32)
    assert plan.decisions[0].warnings == ()


def test_override_beats_hazard_with_warning():
    """An override keeps ChunkedMap and records that it bypassed scan_around_while."""
    outer = _map("outer", 64)
    records = [outer, _loop("inner", 4, parent=outer.key)]

    def est(decisions):
        return sum(100 if isinstance(d.strategy, Vmap) else 10 for d in decisions)

    budget = MemoryBudget(bytes=50, estimate=est)
    plan = resolve(
        records,
        Context(memory_budget=budget, chunk_size=32, backend="cuda"),
        overrides={"outer": (ChunkedMap(batch_size=32), ChunkedMap(batch_size=32))},
    )
    assert plan.decisions[0].primal == ChunkedMap(batch_size=32)
    assert plan.decisions[0].diff == ChunkedMap(batch_size=32)
    assert "override:outer bypassed scan_around_while" in plan.decisions[0].warnings


def test_override_exceeding_budget_raises():
    """Forcing two maps back to Vmap after planning exceeds the joint budget.

    The planner demotes until 8+64 fits in 100. Overriding both sites to Vmap
    costs 64+64. Fails while that override is applied with no second estimate.
    """
    records = [_map("first", 64), _map("second", 64)]

    def est(decisions):
        return sum(64 if isinstance(d.strategy, Vmap) else 8 for d in decisions)

    budget = MemoryBudget(bytes=100, estimate=est)
    with pytest.raises(UnsatisfiableContextError) as exc_info:
        resolve(
            records,
            Context(memory_budget=budget, chunk_size=32),
            overrides={
                "first": (Vmap(), Vmap()),
                "second": (Vmap(), Vmap()),
            },
        )
    err = exc_info.value
    assert err.path == "primal"
    assert err.site == str(records[0].key)
    assert err.constraint == "override exceeds memory budget: estimate 128 > 100"


def test_vmap_decision_batch_size_is_consistent_across_estimates():
    """Every estimate sees the same batch_size on a Vmap decision: ``chunk_size``.

    BatchPlanner records ``default_batch_size`` on a Vmap decision, and a Vmap runs
    the whole axis whatever that field says. The override re-check used to pass the
    cardinality (64), so a user estimator read two values for one strategy (#6158).
    """
    records = [_map("m", 64)]
    seen: set[int] = set()

    def est(decisions):
        seen.update(d.batch_size for d in decisions if isinstance(d.strategy, Vmap))
        return 1

    budget = MemoryBudget(bytes=10_000, estimate=est)
    plan = resolve(
        records,
        Context(memory_budget=budget, chunk_size=16),
        overrides={"m": (Vmap(), Vmap())},
    )
    assert plan.decisions[0].primal == Vmap()
    assert seen == {16}


def test_override_violating_context_raises():
    """Export other than stablehlo rejects an override that forces StaticWhile."""
    with pytest.raises(UnsatisfiableContextError) as exc_info:
        resolve(
            [_loop("body", 4)],
            Context(target="export", export_format="onnx"),
            overrides={"body": (StaticWhile(length=4), Scan())},
        )
    err = exc_info.value
    assert err.path == "primal"
    assert "body" in err.site
    assert "StaticWhile" in err.constraint


def test_diff_visit_copies_canonical_twin():
    """A diff-path record copies its primal-chain twin, including a rewritten parent chain."""
    outer = _map("outer", 4)
    inner = _loop("inner", 3, parent=outer.key)
    diff_inner = _loop("inner", 3, parent=_key("outer", path="diff"), path="diff")
    plan = resolve([outer, inner, diff_inner], Context())
    copied = plan.decisions[2]
    assert copied.key == diff_inner.key
    assert copied.primal == StaticWhile(length=3)
    assert copied.diff == Scan()
    assert copied.primal_reason.startswith(f"diff visit of {inner.key}: ")
    assert copied.diff_reason == plan.decisions[1].diff_reason
    assert copied.warnings == plan.decisions[1].warnings


def test_diff_visit_without_primal_twin_uses_kind_default():
    """A diff visit with no primal twin uses the map default and says so."""
    plan = resolve([_map("solo", 8, path="diff")], Context())
    decision = plan.decisions[0]
    assert decision.primal == Vmap()
    assert decision.diff == Vmap()
    assert decision.primal_reason == "no primal twin"
    assert decision.diff_reason == "no primal twin"


def test_unknown_override_label_raises():
    """An override whose label matches no record names that label."""
    with pytest.raises(ValueError, match="missing"):
        resolve([_map("batch", 4)], Context(), overrides={"missing": (Vmap(), Vmap())})


def test_loop_override_rejects_vmap():
    """A loop override that is not StaticWhile or Scan names the site."""
    with pytest.raises(IntentError) as exc_info:
        resolve([_loop("body", 4)], Context(), overrides={"body": (Vmap(), None)})
    assert exc_info.value.site == "body"


def test_cardinality_one_skips_planner():
    """A leading size of 1 is Vmap and is not a joint-budget axis.

    If it were planned, the size-64 site would be demoted: two Vmaps cost 128 and the
    only demotion candidate is the large site. Excluding the size-1 site leaves one
    Vmap, which fits in 100 bytes.
    """
    records = [_map("tiny", 1), _map("batch", 64)]

    def est(decisions):
        return sum(64 if isinstance(d.strategy, Vmap) else 8 for d in decisions)

    budget = MemoryBudget(bytes=100, estimate=est)
    plan = resolve(records, Context(memory_budget=budget, chunk_size=32))
    assert plan.decisions[0].primal == Vmap()
    assert plan.decisions[0].diff == Vmap()
    assert "cardinality" in plan.decisions[0].primal_reason
    assert plan.decisions[1].primal == Vmap()


def test_long_static_scan_warns_above_threshold():
    """A Scan longer than 256 warns; a Scan of length 256 does not."""
    exported = Context(target="export", export_format="onnx")
    long = resolve([_loop("body", 257)], exported)
    assert long.decisions[0].primal == Scan()
    assert "long_static_scan" in long.decisions[0].warnings
    boundary = resolve([_loop("body", 256)], exported)
    assert boundary.decisions[0].warnings == ()


def test_default_estimator_sums_lowered_memory():
    """default_estimator sums lowered_memory_estimate for Vmap and ChunkedMap, memoised."""

    def fn(x):
        return x + 1

    record = _map("batch", 4, fn=fn)
    spec = AxisSpec(name=str(record.key), cardinality=4, default_batch_size=2)
    vmap_decision = AxisDecision(spec, 4, "vmap", Vmap())
    chunk_decision = AxisDecision(spec, 2, "chunk", ChunkedMap(batch_size=2))
    estimate = default_estimator([record])
    vmap_bytes = lowered_memory_estimate(jax.vmap(fn), *record.avals)
    chunk_bytes = lowered_memory_estimate(
        lambda xs: chunked_map(fn, xs, batch_size=2),
        *record.avals,
    )
    assert estimate([vmap_decision]) == vmap_bytes
    assert estimate([vmap_decision]) == vmap_bytes
    assert estimate([chunk_decision]) == chunk_bytes
    assert estimate([vmap_decision, chunk_decision]) == vmap_bytes + chunk_bytes


def test_int_budget_uses_default_estimator():
    """An int memory_budget is planned by default_estimator.

    Printed on these two maps, both-Vmap is 192 bytes and chunking costs more,
    so 1000 keeps Vmap and 1 is unsatisfiable. The assertions are the plan and
    the error type.
    """

    def fn(x):
        return jnp.add(x, jnp.float32(1))

    records = [_map("first", 4, fn=fn), _map("second", 4, fn=fn)]
    plan = resolve(records, Context(memory_budget=1000))
    assert [decision.primal for decision in plan.decisions] == [Vmap(), Vmap()]
    with pytest.raises(UnsatisfiableContextError):
        resolve(records, Context(memory_budget=1))


def test_cond_loop_primal_is_plain_while():
    """A jit cond loop is WhileCarry and records no diff strategy."""
    plan = resolve([_loop("body", None)], Context())
    assert plan.decisions[0].primal == WhileCarry()
    assert plan.decisions[0].diff is None
    assert plan.decisions[0].diff_reason == (
        "conditional loop: jax.lax.while_loop (forward-mode only)"
    )


def test_chunked_override_around_while_records_bypass():
    """A cuda override to ChunkedMap around StaticWhile records the bypass.

    No budget would have chosen Vmap. The hazard must see the emitted
    ChunkedMap. Fails while warnings stay empty because the predicate ran
    on that automatic Vmap.
    """
    outer = _map("outer", 64)
    records = [outer, _loop("inner", 4, parent=outer.key)]
    plan = resolve(
        records,
        Context(backend="cuda"),
        overrides={"outer": (ChunkedMap(batch_size=32), ChunkedMap(batch_size=32))},
    )
    assert plan.decisions[0].primal == ChunkedMap(batch_size=32)
    assert plan.decisions[0].warnings == ("override:outer bypassed scan_around_while",)


def test_vmap_override_around_while_is_not_bypassed():
    """Overriding the outer map to Vmap emits Vmap and no hazard warning.

    Joint cost of that Vmap plus the other map's chunk is 30+10, which fits
    in 50. The automatic plan is ChunkedMap, so a predicate that ignores the
    override still appends the bypass string. Fails while that string is
    present on a Vmap decision.
    """
    outer = _map("outer", 64)
    other = _map("other", 64)
    records = [outer, _loop("inner", 4, parent=outer.key), other]

    def est(decisions):
        total = 0
        for decision in decisions:
            vmap = isinstance(decision.strategy, Vmap)
            if decision.spec.name == str(outer.key):
                total += 30 if vmap else 20
            else:
                total += 40 if vmap else 10
        return total

    budget = MemoryBudget(bytes=50, estimate=est)
    plan = resolve(
        records,
        Context(memory_budget=budget, chunk_size=32, backend="cuda"),
        overrides={"outer": (Vmap(), Vmap())},
    )
    assert plan.decisions[0].primal == Vmap()
    assert plan.decisions[0].warnings == ()
    assert plan.decisions[2].primal == ChunkedMap(batch_size=32)


def test_long_scan_override_records_bypass():
    """A jit override to Scan of length 257 records the long-scan bypass.

    The automatic primal is StaticWhile, so the warning stays empty until
    the predicate reads the emitted Scan. Export onnx with no override still
    warns long_static_scan for the same length.
    """
    plan = resolve(
        [_loop("body", 257)],
        Context(),
        overrides={"body": (Scan(), Scan())},
    )
    assert plan.decisions[0].primal == Scan()
    assert plan.decisions[0].warnings == ("override:body bypassed long_static_scan",)
    exported = resolve([_loop("body", 257)], Context(target="export", export_format="onnx"))
    assert exported.decisions[0].warnings == ("long_static_scan",)


def test_map_override_diff_reason_when_strategies_differ():
    """A map override whose diff is not the primal names the override.

    (Vmap, ChunkedMap(batch_size=2)) must not say the strategies match.
    Fails while diff_reason is the MVP same-strategy string.
    """
    plan = resolve(
        [_map("batch", 4)],
        Context(),
        overrides={"batch": (Vmap(), ChunkedMap(batch_size=2))},
    )
    decision = plan.decisions[0]
    assert decision.primal == Vmap()
    assert decision.diff == ChunkedMap(batch_size=2)
    assert decision.diff_reason == "override:batch"


def test_int_budget_multi_leaf_map_raises():
    """An int budget on a two-leaf map raises IntentError, not TypeError.

    The default estimator takes one positional array. Two leaves cannot be
    rebuilt into the element pytree. Fails while closure_convert raises
    TypeError, or while site, path, or constraint is wrong.
    """

    def body(tree):
        return tree["a"] + tree["b"]

    element = {"a": jnp.ones((3,), jnp.float32), "b": jnp.ones((3,), jnp.float32)}
    converted, _consts = jax.closure_convert(body, element)
    record = SiteRecord(
        key=_key("paired"),
        kind="map",
        avals=(
            jax.ShapeDtypeStruct((4, 3), jnp.float32),
            jax.ShapeDtypeStruct((4, 3), jnp.float32),
        ),
        length=None,
        fn=converted,
    )
    with pytest.raises(IntentError) as exc_info:
        resolve([record], Context(memory_budget=10**9))
    err = exc_info.value
    assert err.site == str(record.key)
    assert err.path == "primal"
    assert err.constraint == "default estimator takes one positional array"


def test_static_while_override_length_must_match_record():
    """A StaticWhile override whose length is not the site's length is refused.

    Fails while ``StaticWhile(length=99)`` is recorded for a length-4 loop.
    """
    with pytest.raises(IntentError) as exc_info:
        resolve(
            [_loop("body", 4)],
            Context(),
            overrides={"body": (StaticWhile(length=99), Scan())},
        )
    err = exc_info.value
    assert err.site == "body"
    assert err.path == "primal"
    assert "99" in err.constraint
    assert "4" in err.constraint


def test_loop_override_diff_none_records_scan_when_available():
    """A loop override with no diff strategy records Scan when differentiation runs.

    The JVP always scans. Fails while diff stays None or the reason is not
    ``override:body``.
    """
    plan = resolve(
        [_loop("body", 4)],
        Context(),
        overrides={"body": (StaticWhile(length=4), None)},
    )
    decision = plan.decisions[0]
    assert decision.primal == StaticWhile(length=4)
    assert decision.diff == Scan()
    assert decision.diff_reason == "override:body"


def test_loop_override_diff_none_stays_none_when_unavailable():
    """A loop override with no diff strategy stays None when diff was not traced.

    Fails while diff becomes Scan, or while the reason is not the caller-supplied
    unavailable reason.
    """
    reason = "cotangent was not requested"
    plan = resolve(
        [_loop("body", 4)],
        Context(),
        overrides={"body": (StaticWhile(length=4), None)},
        diff_available=False,
        diff_unavailable_reason=reason,
    )
    decision = plan.decisions[0]
    assert decision.primal == StaticWhile(length=4)
    assert decision.diff is None
    assert decision.diff_reason == reason


def test_scan_around_while_accepts_gpu_platform_name(monkeypatch):
    """``jax.default_backend() == "gpu"`` rewrites ChunkedMap around StaticWhile.

    ``"gpu"`` is JAX's platform name for CUDA and ROCm. Fails while the hazard only
    names ``"cuda"`` and the outer map stays chunked.
    """

    def _gpu() -> str:
        return "gpu"

    monkeypatch.setattr("xtrax.intent._resolve.jax.default_backend", _gpu)
    monkeypatch.setattr("xtrax.intent._compile.jax.default_backend", _gpu)
    monkeypatch.setattr("xtrax.intent.context.jax.default_backend", _gpu)
    outer = _map("outer", 64)
    inner = _loop("inner", 4, parent=outer.key)
    other = _map("other", 64)
    records = [outer, inner, other]

    def est(decisions):
        total = 0
        for decision in decisions:
            vmap = isinstance(decision.strategy, Vmap)
            if decision.spec.name == str(outer.key):
                total += 30 if vmap else 20
            else:
                total += 40 if vmap else 10
        return total

    budget = MemoryBudget(bytes=50, estimate=est)
    plan = resolve(records, Context(memory_budget=budget, chunk_size=32))
    assert plan.decisions[0].primal == Vmap()
    assert plan.decisions[0].diff == Vmap()
    assert plan.decisions[1].primal == StaticWhile(length=4)
    assert plan.decisions[2].primal == ChunkedMap(batch_size=32)


def test_cond_loop_override_allows_only_while_carry():
    """A cond loop accepts WhileCarry and refuses Scan and StaticWhile at plan time.

    A static-length loop still refuses WhileCarry. Fails while a Scan override
    is stored and the refusal waits until the loop runs.
    """
    with pytest.raises(IntentError) as scan_info:
        resolve([_loop("w", None)], Context(), overrides={"w": (Scan(), None)})
    scan_err = scan_info.value
    assert scan_err.site == "w"
    assert scan_err.path == "primal"
    assert scan_err.constraint == repr(Scan())

    static_while = StaticWhile(length=4)
    with pytest.raises(IntentError) as static_info:
        resolve([_loop("w", None)], Context(), overrides={"w": (static_while, None)})
    static_err = static_info.value
    assert static_err.site == "w"
    assert static_err.path == "primal"
    assert static_err.constraint == repr(static_while)

    plan = resolve([_loop("w", None)], Context(), overrides={"w": (WhileCarry(), None)})
    assert plan.decisions[0].primal == WhileCarry()
    assert plan.decisions[0].diff is None

    with pytest.raises(IntentError) as carry_info:
        resolve([_loop("body", 4)], Context(), overrides={"body": (WhileCarry(), None)})
    carry_err = carry_info.value
    assert carry_err.site == "body"
    assert carry_err.path == "primal"
    assert carry_err.constraint == repr(WhileCarry())


def test_cond_loop_diff_is_forward_mode_only():
    """A cond loop plans no diff strategy and names the plain while_loop.

    Fails while the diff path is Scan and the reason says it resolves to Scan.
    """
    plan = resolve([_loop("body", None)], Context())
    decision = plan.decisions[0]
    assert decision.primal == WhileCarry()
    assert decision.diff is None
    assert decision.diff_reason == ("conditional loop: jax.lax.while_loop (forward-mode only)")
