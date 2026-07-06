# Spec: joint-budget mode for BatchPlanner

**Task:** 260706_xtrax_assess · **Status:** approved-in-session (user + assistant review, this session) · **Scope:** xtrax only; aminx migration is a companion plan on the aminx side.

## Motivation

xtrax's `BatchPlanner.plan()` decides each axis independently (cardinality vs
batch size, optional per-axis memory override). Downstream consumers with a
total-memory constraint (aminx's planner is the motivating case) instead need
*joint* planning: start maximally parallel, demote axes one at a time until the
memory estimate for the whole plan fits a budget. Today that forces consumers to
fork the planner and maintain parallel strategy classes plus a translation
layer. Joint-budget planning is expressible purely in axes/cardinalities/
strategies/bytes — squarely inside xtrax's ownership boundary — so it belongs
in `BatchPlanner` as an optional mode, not as a consumer fork and not as a
pluggable-policy architecture.

## Design

### New declaration object (house pattern: CarrySpec/DedupSpec)

```python
@dataclass(frozen=True)
class MemoryBudget:
    bytes: int                                            # total plan budget, > 0
    estimate: Callable[[Sequence[AxisDecision]], int]     # joint estimator, returns bytes
```

- `__post_init__` validates: `bytes` is a positive `int` (bool rejected),
  `estimate` is callable. Fail loud at construction.
- Lives in `xtrax/tiling/budget.py` with `BudgetInfeasibleError(Exception)`.
- Exported from `xtrax.tiling` (NOT root `xtrax` — same tier as `CarrySpec`;
  no `distribution/public_api.toml` change).

### Planner integration

```python
BatchPlanner(memory_estimator=None, carry_specs=None, dedup_specs=None,
             heterogeneous_axes=None, budget=None)   # budget: MemoryBudget | None
```

- `budget` and `memory_estimator` are mutually exclusive → `ValueError` at
  construction. (The per-axis estimator's semantics — silent fallback, hidden
  `jax.devices()` limit read — must not leak into budget mode.)

### Algorithm (budget mode Phase 2)

Phases 0/0b/1 are unchanged and shared: CarrySpec → Scan (with the existing
heterogeneous-axes rejection), DedupSpec → DedupGather, UNKNOWN role →
`AmbiguousAxisError`. Additionally fixed in budget mode: `bucket_boundaries`
axes take Rule 1 (Bucket) as usual. All fixed decisions participate in the
estimate but are never demotion candidates.

Remaining ("eligible") axes:

1. **Initial assignment:** every eligible axis starts at `Vmap` — including
   `cardinality > default_batch_size` axes that the independent rules would
   have given SafeMap. The budget, not the cardinality heuristic, decides.
2. **Candidates:** eligible axes with `cardinality > default_batch_size`, in
   the order given. Axes with `cardinality <= default_batch_size` are excluded
   (demoting them is a memory no-op) and stay Vmap.
3. **Greedy loop:** compute `estimate(decisions)` over the full plan (spec
   order, fixed + current states). While over budget and candidates remain,
   demote the next candidate to `SafeMap(batch_size=default_batch_size)` and
   recompute. Stop at the first fit.
4. **Order contract:** demotion candidates are tried strictly in the order
   specs were given — the caller expresses demotion priority by ordering
   (axes you are most willing to sequentialize first). This reuses `plan()`'s
   existing documented order-preservation; no `axis_index` field.
5. **Infeasible:** candidates exhausted and still over budget →
   `BudgetInfeasibleError` naming the budget, the final estimate, and the
   per-axis strategy state. Never warn-and-continue.
6. **No silent fallback:** exceptions from `estimate` propagate unchanged.
   (Deliberate divergence from the per-axis `memory_estimator`, which swallows
   estimator errors — an existing wart budget mode must not inherit.)
7. **Divisibility:** demoting a non-divisible axis emits the existing Rule-5
   `RuntimeWarning` (deferred-failure contract at `make_axis_dispatch` time).
8. **Reasoning:** every budget-mode decision carries a reasoning string with
   the byte numbers (retained / demoted at step k with before→after estimate /
   no-op exclusion), so `xtrax plan` / `xtrax explain` show why each demotion
   happened. `AxisDecision`/`BatchPlan` shapes are unchanged → EDA works as-is.

Determinism: the plan is a pure function of `(specs, planner config)`.

### Native JAX tooling hooks (`xtrax/tiling/estimators.py`)

The planner stays estimator-agnostic — `MemoryBudget.estimate` is a plain
callable, which is precisely the seam where JAX's own memory accounting plugs
in. xtrax ships two building blocks so consumers hook native tools instead of
hand-rolling byte math:

- `device_memory_budget(fraction=0.9, device=None) -> int` — budget bytes from
  `Device.memory_stats()["bytes_limit"]` (the XLA allocator's real limit) with
  a safety fraction. Raises `RuntimeError` when the backend reports no stats —
  no silent 4 GiB default (deliberate divergence from the existing
  `_decide_strategy` device-limit read).
- `lowered_memory_estimate(fn, *abstract_args) -> int` — AOT-compiles `fn`
  from `ShapeDtypeStruct`s and reads `Compiled.memory_analysis()` (XLA's
  compile-time buffer assignment): argument + output + temp bytes. This is the
  compiler's own static memory plan, not a heuristic. Costs a compile per
  distinct shape signature; inside a greedy estimate, analyze a representative
  tile and scale analytically, or memoize on the decision signature.

Runtime *validation* (as opposed to planning) stays out of scope for this
slice: comparing the plan-time estimate against
`memory_stats()["peak_bytes_in_use"]` / `jax.profiler.device_memory_profile()`
after execution is an EDA-panel follow-up (estimated-vs-measured calibration).

## Acceptance criteria

- **AC1** `MemoryBudget` validates fields at construction (`bytes` positive int,
  bool rejected; `estimate` callable); importable from `xtrax.tiling`.
- **AC2** `BatchPlanner(budget=..., memory_estimator=...)` raises `ValueError`.
- **AC3** Under-budget plan: all eligible axes Vmap (including
  cardinality > batch_size), zero demotions, estimator called at least once.
- **AC4** Over-budget plan: axes demoted one at a time in given order; the
  estimate is recomputed after each demotion; the loop stops at the first fit
  (later candidates stay Vmap).
- **AC5** `cardinality <= default_batch_size` axes are never demotion
  candidates.
- **AC6** Carry/dedup/bucket axes are fixed, never demoted, and appear in the
  estimator's input; heterogeneous+carry rejection and the UNKNOWN-role guard
  behave identically in budget mode.
- **AC7** Infeasible plan raises `BudgetInfeasibleError`; message contains
  budget bytes, final estimate, and axis names with strategies.
- **AC8** An estimator that raises propagates its exception unchanged.
- **AC9** Demoting a non-divisible axis emits the Rule-5 `RuntimeWarning`.
- **AC10** Output decisions preserve spec order; all reasoning strings
  non-empty and budget-mode ones contain byte numbers.
- **AC11** CHANGELOG entry under [Unreleased]; no root public-API change.
- **AC12** `device_memory_budget` applies the fraction to `bytes_limit`,
  validates the fraction, and raises `RuntimeError` (not a silent default)
  when the device reports no memory stats.
- **AC13** `lowered_memory_estimate` returns XLA buffer-assignment bytes
  (argument + output + temp) for abstract inputs, scales with input size, and
  raises `RuntimeError` when the backend provides no memory analysis.

## Out of scope (recorded for future work)

- Batch-size refinement below `default_batch_size` (e.g., halving) as further
  demotion rungs — the two-rung Vmap→SafeMap ladder matches the motivating
  consumer; deeper ladders need a real use case first.
- A fully automatic joint estimator that composes `make_axis_dispatch` for
  each candidate decision set and runs `lowered_memory_estimate` on the
  resulting program — needs the dispatch-composition design and per-step
  compile caching; own slice.
- Estimated-vs-measured calibration: EDA panel comparing plan-time estimates
  against `memory_stats()["peak_bytes_in_use"]` /
  `jax.profiler.device_memory_profile()` after a real run.
- Reframing the per-axis `memory_estimator` as sugar over budget mode with a
  device-derived default budget (would also hoist the hidden
  `jax.devices()[0].memory_stats()` read out of `_decide_strategy`).
- aminx-side migration: axis registry feeds specs in demotion-priority order;
  parity harness asserting identical demotion sets on the real registry;
  deletion of aminx-native strategy classes and `_strategy_to_xtrax()`.
