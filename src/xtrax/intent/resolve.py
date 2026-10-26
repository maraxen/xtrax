"""Resolve recorded intent sites into one :class:`~xtrax.intent.plan.Plan`.

Primal-chain records (every ``path`` in the key is ``"primal"``) are planned.
Map sites go through :class:`~xtrax.tiling.plan.BatchPlanner`. Loop sites follow
the context rules. Diff-path records copy their primal twin.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Literal

import jax

from xtrax.intent.context import Context
from xtrax.intent.errors import IntentError, UnsatisfiableContextError
from xtrax.intent.plan import PLAN_SCHEMA_VERSION, Plan, SiteDecision, SiteKey, canonical
from xtrax.intent.sites import SiteRecord
from xtrax.tiling.budget import BudgetInfeasibleError, MemoryBudget
from xtrax.tiling.estimators import lowered_memory_estimate
from xtrax.tiling.plan import AxisDecision, AxisSpec, BatchPlanner, vmap_decision_batch_size
from xtrax.tiling.strategy import ChunkedMap, Scan, StaticWhile, Vmap, WhileCarry
from xtrax.transforms.map import chunked_map

_MapStrategy = Vmap | ChunkedMap
_LoopPrimal = StaticWhile | Scan
_LoopOverride = StaticWhile | Scan | WhileCarry
SiteStrategyOverride = tuple[_MapStrategy | _LoopOverride, _MapStrategy | Scan | None]
_HazardView = SiteDecision | SiteRecord
HazardRule = tuple[
    str,
    Callable[[SiteDecision, Sequence[_HazardView], Context], bool],
    Literal["error", "warning"],
]

# ``"gpu"`` is JAX's platform name for CUDA and ROCm. The hazard is accepted
# for both until measured otherwise.
SCAN_AROUND_WHILE_BACKENDS = frozenset({"cuda", "gpu"})
LONG_SCAN_THRESHOLD = 256

_SAME_STRATEGY = "same strategy as primal in the MVP"
_NO_TWIN = "no primal twin"


def default_estimator(records: Sequence[SiteRecord]) -> Callable[[Sequence[AxisDecision]], int]:
    """Sum lowered memory of each planned map under its strategy.

    Each decision is matched to a record by ``AxisSpec.name == str(record.key)``.
    ``Vmap`` lowers ``jax.vmap`` of ``record.fn`` over the mapped pytree.
    Constants hoisted from that function are closed over as abstract inputs
    and are not mapped. ``ChunkedMap(b)`` lowers ``chunked_map`` the same
    way. Results are memoised per ``(key, strategy)``.

    Parameters
    ----------
    records : sequence of SiteRecord
        Records whose ``fn`` and ``avals`` the estimator may compile.

    Returns
    -------
    callable
        ``(decisions) -> bytes``. The total is the sum of the matched sites.
    """
    by_name: dict[str, SiteRecord] = {}
    for record in records:
        by_name.setdefault(str(record.key), record)
    cache: dict[tuple[SiteKey, _MapStrategy], int] = {}

    def estimate(decisions: Sequence[AxisDecision]) -> int:
        """Sum memoised lowered-memory bytes for ``decisions``.

        Parameters
        ----------
        decisions : sequence of AxisDecision
            Planner decisions, matched to records by axis name.

        Returns
        -------
        int
            Total estimated bytes.
        """
        total = 0
        for decision in decisions:
            record = by_name.get(decision.spec.name)
            if record is None:
                raise ValueError(f"no SiteRecord for axis {decision.spec.name!r}")
            strategy = decision.strategy
            if not isinstance(strategy, (Vmap, ChunkedMap)):
                raise ValueError(
                    "default_estimator strategy must be Vmap or ChunkedMap, "
                    f"got {type(strategy).__name__}"
                )
            cache_key = (record.key, strategy)
            cached = cache.get(cache_key)
            if cached is None:
                cached = _lowered_site(record, strategy)
                cache[cache_key] = cached
            total += cached
        return total

    return estimate


def resolve(
    records: Sequence[SiteRecord],
    ctx: Context,
    *,
    overrides: Mapping[str, SiteStrategyOverride] | None = None,
    diff_available: bool = True,
    diff_unavailable_reason: str = "",
) -> Plan:
    """Choose one :class:`~xtrax.intent.plan.SiteDecision` per record.

    Parameters
    ----------
    records : sequence of SiteRecord
        Sites in visit order. One decision is returned per record, in this order.
    ctx : Context
        Budget, export target, backend, and chunk size.
    overrides : mapping or None
        ``label -> (primal strategy, diff strategy or None)``. ``None`` means
        no overrides.
    diff_available : bool
        When ``False``, loop diff strategies are ``None``.
    diff_unavailable_reason : str
        Stored as ``diff_reason`` when a loop has no diff strategy.

    Returns
    -------
    Plan
        Schema :data:`~xtrax.intent.plan.PLAN_SCHEMA_VERSION` and ``ctx``.

    Raises
    ------
    ValueError
        An override label matches no record. The message names the label.
    IntentError
        An override strategy is illegal for the site kind. The error names the site.
    UnsatisfiableContextError
        The budget cannot be met, a map override makes the joint estimate
        exceed ``memory_budget``, export forbids the loop, or
        ``scan_around_while`` cannot move the outer map back to
        :class:`~xtrax.tiling.strategy.Vmap`.
    """
    override_map = {} if overrides is None else overrides
    _validate_overrides(records, override_map)
    budget = _budget_for(ctx, records)
    axis_decisions = _plan_maps(records, ctx, budget)
    chosen = {
        record.key: _chosen_decision(
            record,
            ctx,
            budget,
            axis_decisions,
            override_map,
            diff_available,
            diff_unavailable_reason,
        )
        for record in records
        if _is_primal_chain(record.key)
    }
    for record in records:
        if _is_primal_chain(record.key):
            _check_context(record, chosen[record.key].primal, ctx)
    final: dict[SiteKey, SiteDecision] = {}
    for record in records:
        if not _is_primal_chain(record.key):
            continue
        final[record.key] = _with_hazards(
            record,
            chosen,
            ctx,
            budget,
            axis_decisions,
            override_map,
            records,
            diff_available,
            diff_unavailable_reason,
        )
    _check_map_override_budget(records, final, ctx, budget, override_map)
    decisions = tuple(
        final[record.key]
        if _is_primal_chain(record.key)
        else _diff_visit(record, final, ctx, diff_available, diff_unavailable_reason)
        for record in records
    )
    return Plan(PLAN_SCHEMA_VERSION, ctx, decisions)


def _is_scan_around_while_backend(
    name: str,
    backends: frozenset[str] | None = None,
) -> bool:
    """Return whether ``name`` selects the scan-around-while hazard.

    ``"gpu"`` is JAX's platform name for CUDA and ROCm. The hazard is
    accepted for both until measured otherwise. ``"cuda"`` is the explicit
    device name.

    Parameters
    ----------
    name : str
        Backend from :meth:`~xtrax.intent.context.Context.resolved_backend`.
    backends : frozenset of str or None
        Set to consult. ``None`` reads :data:`SCAN_AROUND_WHILE_BACKENDS`
        at call time, so a test can replace that set.

    Returns
    -------
    bool
        ``True`` when ``name`` is in the selected set.
    """
    chosen = SCAN_AROUND_WHILE_BACKENDS if backends is None else backends
    return name in chosen


def _scan_around_while(
    decision: SiteDecision,
    ancestors: Sequence[_HazardView],
    ctx: Context,
) -> bool:
    """True when a chunked or scan primal encloses a StaticWhile on CUDA or ROCm.

    ``ancestors`` is the subtree: decisions for records whose canonical parent
    chain includes this site, plus the current :class:`SiteRecord`.
    ``"gpu"`` is JAX's platform name for both devices.
    """
    if not _is_scan_around_while_backend(ctx.resolved_backend()):
        return False
    if not isinstance(decision.primal, (ChunkedMap, Scan)):
        return False
    return any(
        isinstance(item, SiteDecision) and isinstance(item.primal, StaticWhile)
        for item in ancestors
    )


def _long_static_scan(
    decision: SiteDecision,
    ancestors: Sequence[_HazardView],
    ctx: Context,
) -> bool:
    """True when a loop primal is Scan and the record length exceeds 256."""
    del ctx
    if decision.kind != "loop" or not isinstance(decision.primal, Scan):
        return False
    for item in ancestors:
        if (
            isinstance(item, SiteRecord)
            and item.key == decision.key
            and isinstance(item.length, int)
        ):
            return item.length > LONG_SCAN_THRESHOLD
    return False


HAZARDS: tuple[HazardRule, ...] = (
    ("scan_around_while", _scan_around_while, "error"),
    ("long_static_scan", _long_static_scan, "warning"),
)


def _is_primal_chain(key: SiteKey) -> bool:
    """Return whether ``key`` is already its canonical primal-chain form."""
    return key == canonical(key)


def _ancestor_keys(key: SiteKey) -> tuple[SiteKey, ...]:
    """Canonical keys of ``key``'s parents, nearest parent first."""
    found: list[SiteKey] = []
    current = key.parent
    while current is not None:
        found.append(canonical(current))
        current = current.parent
    return tuple(found)


def _budget_for(ctx: Context, records: Sequence[SiteRecord]) -> MemoryBudget | None:
    """Return the joint budget, wrapping an int with :func:`default_estimator`."""
    raw = ctx.memory_budget
    if raw is None:
        return None
    if isinstance(raw, MemoryBudget):
        return raw
    return MemoryBudget(bytes=raw, estimate=default_estimator(records))


def _cardinality(record: SiteRecord) -> int:
    """Leading axis length of a map record."""
    return int(record.avals[0].shape[0])


def _planned_map(record: SiteRecord) -> bool:
    """Return whether this primal-chain map enters the batch planner."""
    return _is_primal_chain(record.key) and record.kind == "map" and _cardinality(record) > 1


def _plan_maps(
    records: Sequence[SiteRecord],
    ctx: Context,
    budget: MemoryBudget | None,
) -> tuple[AxisDecision, ...]:
    """Plan primal-chain map sites whose cardinality is greater than 1.

    Raises
    ------
    UnsatisfiableContextError
        Joint planning raises :class:`~xtrax.tiling.budget.BudgetInfeasibleError`.
        ``site`` is the largest planned map and ``constraint`` is the planner message.
    """
    if budget is None:
        return ()
    specs: list[AxisSpec] = []
    largest: SiteRecord | None = None
    largest_n = -1
    for record in records:
        if not _planned_map(record):
            continue
        cardinality = _cardinality(record)
        if cardinality > largest_n:
            largest_n = cardinality
            largest = record
        specs.append(
            AxisSpec(
                name=str(record.key),
                cardinality=cardinality,
                default_batch_size=ctx.chunk_size,
            )
        )
    if not specs:
        return ()
    try:
        return BatchPlanner(budget=budget).plan(specs).decisions
    except BudgetInfeasibleError as exc:
        if largest is None:  # pragma: no cover - specs come from the same records
            raise
        message = str(exc)
        raise UnsatisfiableContextError(
            message,
            site=str(largest.key),
            path="primal",
            constraint=message,
        ) from exc


def _loop_choice(
    record: SiteRecord,
    ctx: Context,
    *,
    diff_available: bool,
    diff_unavailable_reason: str,
) -> tuple[WhileCarry | _LoopPrimal, Scan | None, str, str]:
    """Primal and diff strategies for one loop, before overrides and hazards.

    A conditional loop (``record.length is None``) runs
    :func:`jax.lax.while_loop`. Its diff strategy is ``None``: forward mode
    is that while loop's JVP, and reverse mode is JAX's own error. No
    derivative strategy is added.
    """
    if record.length is None:
        primal: WhileCarry | _LoopPrimal = WhileCarry()
        primal_reason = "conditional loop is a plain while_loop"
        return (
            primal,
            None,
            primal_reason,
            "conditional loop: jax.lax.while_loop (forward-mode only)",
        )
    if ctx.target == "export" and ctx.export_format != "stablehlo":
        primal = Scan()
        primal_reason = "export format is not stablehlo; primal resolves to Scan"
    else:
        primal = StaticWhile(length=record.length)
        primal_reason = "fixed length resolves to StaticWhile"
    if diff_available:
        return primal, Scan(), primal_reason, "diff path resolves to Scan"
    return primal, None, primal_reason, diff_unavailable_reason


def _decision(
    record: SiteRecord,
    primal: Vmap | ChunkedMap | StaticWhile | Scan | WhileCarry,
    diff: Vmap | ChunkedMap | Scan | None,
    primal_reason: str,
    diff_reason: str,
) -> SiteDecision:
    """Build a decision with no warnings."""
    return SiteDecision(
        key=record.key,
        kind=record.kind,
        primal=primal,
        diff=diff,
        primal_reason=primal_reason,
        diff_reason=diff_reason,
    )


def _automatic_decision(
    record: SiteRecord,
    ctx: Context,
    budget: MemoryBudget | None,
    axis_decisions: Sequence[AxisDecision],
    *,
    diff_available: bool,
    diff_unavailable_reason: str,
) -> SiteDecision:
    """Strategy for one primal-chain record before overrides and hazards."""
    if record.kind == "loop":
        primal, diff, primal_reason, diff_reason = _loop_choice(
            record,
            ctx,
            diff_available=diff_available,
            diff_unavailable_reason=diff_unavailable_reason,
        )
        return _decision(record, primal, diff, primal_reason, diff_reason)
    cardinality = _cardinality(record)
    if cardinality <= 1:
        primal_map: _MapStrategy = Vmap()
        reason = "cardinality <= 1 resolves to Vmap"
    elif budget is None:
        primal_map = Vmap()
        reason = "no memory budget; cardinality selects Vmap"
    else:
        name = str(record.key)
        match = next(item for item in axis_decisions if item.spec.name == name)
        strategy = match.strategy
        if not isinstance(strategy, (Vmap, ChunkedMap)):
            raise IntentError(
                f"map primal strategy must be Vmap or ChunkedMap, got {type(strategy).__name__}",
                site=str(record.key),
                path="primal",
                constraint=repr(strategy),
            )
        primal_map = strategy
        reason = match.reasoning
    return _decision(record, primal_map, primal_map, reason, _SAME_STRATEGY)


def _no_twin_decision(
    record: SiteRecord,
    ctx: Context,
    *,
    diff_available: bool,
    diff_unavailable_reason: str,
) -> SiteDecision:
    """Per-kind default for a diff visit that has no primal twin."""
    if record.kind == "map":
        return _decision(record, Vmap(), Vmap(), _NO_TWIN, _NO_TWIN)
    primal, diff, _, _ = _loop_choice(
        record,
        ctx,
        diff_available=diff_available,
        diff_unavailable_reason=diff_unavailable_reason,
    )
    return _decision(record, primal, diff, _NO_TWIN, _NO_TWIN)


def _validate_overrides(
    records: Sequence[SiteRecord],
    overrides: Mapping[str, SiteStrategyOverride],
) -> None:
    """Reject unknown labels and strategies that do not match the site kind."""
    labels = {record.key.label for record in records}
    for label in overrides:
        if label not in labels:
            raise ValueError(f"override label {label!r} matches no record")
    for record in records:
        override = overrides.get(record.key.label)
        if override is not None:
            _check_override_type(record, override)


def _check_override_type(record: SiteRecord, override: SiteStrategyOverride) -> None:
    """Raise :class:`IntentError` when ``override`` is illegal for ``record``."""
    primal, diff = override
    site = record.key.label
    if record.kind == "map":
        if not isinstance(primal, (Vmap, ChunkedMap)):
            raise IntentError(
                f"map override must be Vmap or ChunkedMap, got {type(primal).__name__}",
                site=site,
                path="primal",
                constraint=repr(primal),
            )
        if diff is not None and not isinstance(diff, (Vmap, ChunkedMap)):
            raise IntentError(
                f"map override diff must be Vmap or ChunkedMap, got {type(diff).__name__}",
                site=site,
                path="diff",
                constraint=repr(diff),
            )
        return
    if record.length is None:
        if not isinstance(primal, WhileCarry):
            raise IntentError(
                f"cond loop override must be WhileCarry, got {type(primal).__name__}",
                site=site,
                path="primal",
                constraint=repr(primal),
            )
        if diff is not None:
            raise IntentError(
                f"cond loop override diff must be None, got {type(diff).__name__}",
                site=site,
                path="diff",
                constraint=repr(diff),
            )
        return
    if not isinstance(primal, (StaticWhile, Scan)):
        raise IntentError(
            f"loop override must be StaticWhile or Scan, got {type(primal).__name__}",
            site=site,
            path="primal",
            constraint=repr(primal),
        )
    if diff is not None and not isinstance(diff, Scan):
        raise IntentError(
            f"loop override diff must be Scan or None, got {type(diff).__name__}",
            site=site,
            path="diff",
            constraint=repr(diff),
        )


def _apply_override(
    record: SiteRecord,
    automatic: SiteDecision,
    override: SiteStrategyOverride,
    *,
    diff_available: bool,
    diff_unavailable_reason: str,
) -> SiteDecision:
    """Replace both paths of ``automatic`` with ``override``."""
    primal, diff = override
    label = record.key.label
    if record.kind == "map":
        used_diff = primal if diff is None else diff
        legal = isinstance(primal, (Vmap, ChunkedMap)) and isinstance(used_diff, (Vmap, ChunkedMap))
        if not legal:
            raise IntentError(
                f"map override must be Vmap or ChunkedMap, got {type(primal).__name__}",
                site=label,
                path="primal",
                constraint=repr(primal),
            )
        if diff is None or diff == primal:
            diff_reason = _SAME_STRATEGY
        else:
            diff_reason = f"override:{label}"
        return _decision(record, primal, used_diff, f"override:{label}", diff_reason)
    if record.length is None:
        if not isinstance(primal, WhileCarry):
            raise IntentError(
                f"cond loop override must be WhileCarry, got {type(primal).__name__}",
                site=label,
                path="primal",
                constraint=repr(primal),
            )
        if diff is not None:
            raise IntentError(
                f"cond loop override diff must be None, got {type(diff).__name__}",
                site=label,
                path="diff",
                constraint=repr(diff),
            )
        return _decision(
            record,
            primal,
            None,
            f"override:{label}",
            "conditional loop: jax.lax.while_loop (forward-mode only)",
        )
    if not isinstance(primal, (StaticWhile, Scan)):
        raise IntentError(
            f"loop override must be StaticWhile or Scan, got {type(primal).__name__}",
            site=label,
            path="primal",
            constraint=repr(primal),
        )
    if isinstance(primal, StaticWhile) and primal.length != record.length:
        raise IntentError(
            (
                f"StaticWhile override length {primal.length} does not match "
                f"site length {record.length}"
            ),
            site=label,
            path="primal",
            constraint=f"override length {primal.length}, site length {record.length}",
        )
    if diff is None:
        if diff_available:
            # The loop derivative always scans, including when the override
            # leaves the diff strategy unspecified.
            return _decision(
                record,
                primal,
                Scan(),
                f"override:{label}",
                f"override:{label}",
            )
        return _decision(record, primal, None, f"override:{label}", diff_unavailable_reason)
    if not isinstance(diff, Scan):
        raise IntentError(
            f"loop override diff must be Scan or None, got {type(diff).__name__}",
            site=label,
            path="diff",
            constraint=repr(diff),
        )
    del automatic
    return _decision(record, primal, diff, f"override:{label}", "diff path resolves to Scan")


def _chosen_decision(
    record: SiteRecord,
    ctx: Context,
    budget: MemoryBudget | None,
    axis_decisions: Sequence[AxisDecision],
    overrides: Mapping[str, SiteStrategyOverride],
    diff_available: bool,
    diff_unavailable_reason: str,
) -> SiteDecision:
    """Automatic decision with an override applied when the label has one."""
    automatic = _automatic_decision(
        record,
        ctx,
        budget,
        axis_decisions,
        diff_available=diff_available,
        diff_unavailable_reason=diff_unavailable_reason,
    )
    override = overrides.get(record.key.label)
    if override is None:
        return automatic
    return _apply_override(
        record,
        automatic,
        override,
        diff_available=diff_available,
        diff_unavailable_reason=diff_unavailable_reason,
    )


def _check_map_override_budget(
    records: Sequence[SiteRecord],
    decisions: Mapping[SiteKey, SiteDecision],
    ctx: Context,
    budget: MemoryBudget | None,
    overrides: Mapping[str, SiteStrategyOverride],
) -> None:
    """Reject a map override whose final joint estimate exceeds the budget.

    Specs stay in record order. Each decision uses the resolved primal.
    ``Vmap`` contributes :func:`~xtrax.tiling.plan.vmap_decision_batch_size`,
    the value ``BatchPlanner`` records; ``ChunkedMap`` contributes that
    strategy's batch size.

    Raises
    ------
    UnsatisfiableContextError
        ``budget`` is set, at least one primal-chain map was overridden, and
        the estimate exceeds ``budget.bytes``. ``site`` is the first such
        map's key, ``path`` is ``"primal"``, and ``constraint`` names both
        numbers.
    """
    if budget is None:
        return
    overridden = [
        record
        for record in records
        if _is_primal_chain(record.key)
        and record.kind == "map"
        and overrides.get(record.key.label) is not None
    ]
    if not overridden:
        return
    axis: list[AxisDecision] = []
    for record in records:
        if not _planned_map(record):
            continue
        primal = decisions[record.key].primal
        spec = AxisSpec(
            name=str(record.key),
            cardinality=_cardinality(record),
            default_batch_size=ctx.chunk_size,
        )
        if isinstance(primal, Vmap):
            batch_size = vmap_decision_batch_size(spec)
        elif isinstance(primal, ChunkedMap):
            batch_size = primal.batch_size
        else:
            raise IntentError(
                f"map primal strategy must be Vmap or ChunkedMap, got {type(primal).__name__}",
                site=str(record.key),
                path="primal",
                constraint=repr(primal),
            )
        axis.append(
            AxisDecision(
                spec=spec,
                batch_size=batch_size,
                reasoning=decisions[record.key].primal_reason,
                strategy=primal,
            )
        )
    estimate = budget.estimate(tuple(axis))
    if estimate > budget.bytes:
        constraint = f"override exceeds memory budget: estimate {estimate} > {budget.bytes}"
        raise UnsatisfiableContextError(
            constraint,
            site=str(overridden[0].key),
            path="primal",
            constraint=constraint,
        )


def _check_context(record: SiteRecord, primal: object, ctx: Context) -> None:
    """Raise when ``primal`` or a cond loop violates a hard context rule."""
    if record.kind == "loop" and record.length is None and ctx.target == "export":
        raise UnsatisfiableContextError(
            "export forbids unbounded while_loop",
            site=str(record.key),
            path="primal",
            constraint="export forbids unbounded while_loop",
        )
    if (
        ctx.target == "export"
        and ctx.export_format != "stablehlo"
        and isinstance(primal, StaticWhile)
    ):
        raise UnsatisfiableContextError(
            "export forbids StaticWhile when export_format is not stablehlo",
            site=str(record.key),
            path="primal",
            constraint="export forbids StaticWhile when export_format is not stablehlo",
        )


def _preview(
    record: SiteRecord,
    chosen: Mapping[SiteKey, SiteDecision],
    ctx: Context,
    *,
    diff_available: bool,
    diff_unavailable_reason: str,
) -> SiteDecision:
    """Strategy visible to a parent's hazard rule, before hazard rewrites."""
    if record.key in chosen:
        return chosen[record.key]
    twin = chosen.get(canonical(record.key))
    if twin is not None:
        return twin
    return _no_twin_decision(
        record,
        ctx,
        diff_available=diff_available,
        diff_unavailable_reason=diff_unavailable_reason,
    )


def _hazard_views(
    record: SiteRecord,
    records: Sequence[SiteRecord],
    chosen: Mapping[SiteKey, SiteDecision],
    ctx: Context,
    *,
    diff_available: bool,
    diff_unavailable_reason: str,
) -> tuple[_HazardView, ...]:
    """Subtree decisions plus ``record``, for hazard predicates."""
    outer = canonical(record.key)
    views: list[_HazardView] = []
    for other in records:
        if other.key == record.key:
            continue
        if outer not in _ancestor_keys(other.key):
            continue
        views.append(
            _preview(
                other,
                chosen,
                ctx,
                diff_available=diff_available,
                diff_unavailable_reason=diff_unavailable_reason,
            )
        )
    views.append(record)
    return tuple(views)


def _vmap_fits(
    record: SiteRecord,
    budget: MemoryBudget | None,
    axis_decisions: Sequence[AxisDecision],
) -> bool:
    """Return whether setting this map's axis to Vmap stays within the budget."""
    if budget is None:
        return True
    name = str(record.key)
    rewritten: list[AxisDecision] = []
    for decision in axis_decisions:
        if decision.spec.name == name:
            rewritten.append(
                replace(
                    decision,
                    strategy=Vmap(),
                    batch_size=vmap_decision_batch_size(decision.spec),
                )
            )
        else:
            rewritten.append(decision)
    return budget.estimate(tuple(rewritten)) <= budget.bytes


def _with_hazards(
    record: SiteRecord,
    chosen: Mapping[SiteKey, SiteDecision],
    ctx: Context,
    budget: MemoryBudget | None,
    axis_decisions: Sequence[AxisDecision],
    overrides: Mapping[str, SiteStrategyOverride],
    records: Sequence[SiteRecord],
    diff_available: bool,
    diff_unavailable_reason: str,
) -> SiteDecision:
    """Apply hazard rules to the decision that will be emitted."""
    decision = chosen[record.key]
    override = overrides.get(record.key.label)
    views = _hazard_views(
        record,
        records,
        chosen,
        ctx,
        diff_available=diff_available,
        diff_unavailable_reason=diff_unavailable_reason,
    )
    warnings = list(decision.warnings)
    for name, applies, severity in HAZARDS:
        if not applies(decision, views, ctx):
            continue
        if override is not None:
            warnings.append(f"override:{record.key.label} bypassed {name}")
            continue
        if name == "scan_around_while":
            fits = _vmap_fits(record, budget, axis_decisions)
            if fits and isinstance(decision.primal, ChunkedMap):
                decision = replace(
                    decision,
                    primal=Vmap(),
                    diff=Vmap(),
                    primal_reason="scan_around_while prefers Vmap",
                    diff_reason=_SAME_STRATEGY,
                )
                continue
            if not fits:
                raise UnsatisfiableContextError(
                    "scan_around_while",
                    site=str(record.key),
                    path="primal",
                    constraint="scan_around_while",
                )
            continue
        if severity == "warning":
            warnings.append(name)
            continue
        raise UnsatisfiableContextError(
            name,
            site=str(record.key),
            path="primal",
            constraint=name,
        )
    return replace(decision, warnings=tuple(warnings))


def _diff_visit(
    record: SiteRecord,
    final: Mapping[SiteKey, SiteDecision],
    ctx: Context,
    diff_available: bool,
    diff_unavailable_reason: str,
) -> SiteDecision:
    """Copy the canonical twin, or use the per-kind default when it is absent."""
    twin_key = canonical(record.key)
    twin = final.get(twin_key)
    if twin is None:
        decision = _no_twin_decision(
            record,
            ctx,
            diff_available=diff_available,
            diff_unavailable_reason=diff_unavailable_reason,
        )
        _check_context(record, decision.primal, ctx)
        return decision
    return replace(
        twin,
        key=record.key,
        primal_reason=f"diff visit of {twin_key}: {twin.primal_reason}",
    )


def _lowered_site(record: SiteRecord, strategy: _MapStrategy) -> int:
    """Lower one site under ``strategy`` and return XLA's peak-byte estimate.

    A record that carries ``xs_tree`` or hoisted constants is lowered as a
    function of the rebuilt ``xs`` pytree plus those constants. The mapped
    body closes over the constants, so they are not batch axes. A flat
    one-leaf record with no constants keeps ``jax.vmap(record.fn)``.
    """
    fn = record.fn
    if fn is None:
        raise IntentError(
            "default_estimator requires a site function",
            site=str(record.key),
            path="primal",
            constraint="record.fn is None",
        )
    if record.xs_tree is None and not record.const_avals:
        if len(record.avals) != 1:
            raise IntentError(
                "default estimator takes one positional array",
                site=str(record.key),
                path="primal",
                constraint="default estimator takes one positional array",
            )
        if isinstance(strategy, Vmap):
            return lowered_memory_estimate(jax.vmap(fn), *record.avals)
        batch_size = strategy.batch_size

        def _chunked(xs: object) -> object:
            return chunked_map(fn, xs, batch_size=batch_size)

        return lowered_memory_estimate(_chunked, *record.avals)

    xs = (
        record.avals[0]
        if record.xs_tree is None
        else jax.tree_util.tree_unflatten(record.xs_tree, record.avals)
    )
    consts = record.const_avals
    if isinstance(strategy, Vmap):

        def _vmapped(values: object, *closed: object) -> object:
            return jax.vmap(lambda x: fn(x, *closed))(values)

        return lowered_memory_estimate(_vmapped, xs, *consts)

    batch_size = strategy.batch_size

    def _chunked_closed(values: object, *closed: object) -> object:
        return chunked_map(lambda x: fn(x, *closed), values, batch_size=batch_size)

    return lowered_memory_estimate(_chunked_closed, xs, *consts)
