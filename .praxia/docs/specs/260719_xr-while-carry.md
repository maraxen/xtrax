---
title: WhileCarry — a lax.while_loop-backed AxisStrategy
task_id: 260719_xr-while-carry
date: 260719
status: draft
source: prolix (.praxia/docs/specs/260718_xr-while-carry.md, task 260715_b1_physics_parity)
---

# WhileCarry — a `lax.while_loop`-backed `AxisStrategy`

## Motivation

`xtrax.tiling`'s sealed `AxisStrategy` union (`strategy.py:109`) is `Vmap | SafeMap | Scan | DedupGather | Bucket`. `Scan` compiles to `jax.lax.scan` via `JaxScanIterator` (`iterator.py:153`), whose contract is `(carry, x) -> (carry, y)`, exactly mirroring `lax.scan`: a known input sequence `x` is iterated over, and `y` (per-step output) is stacked into a trajectory.

A downstream consumer (**prolix**, a JAX molecular dynamics engine) has a real, already-shipped need this contract doesn't cover: an **inference** step loop (`EnsemblePlan.run(run_mode="inference")`, `prolix/api/ensemble_dispatch.py:dispatch_n_steps_inference`) that:

- Has **no external input sequence** — nothing to scan over, only the loop's own step counter.
- Wants **no per-step output collection** — only the final state matters (this is inference, not trajectory recording; recording uses a *separate*, already-`Scan`-wired dispatch path, `dispatch_n_steps` → `make_axis_dispatch(Scan(), ...)`).
- Is deliberately **not** `Scan`-based at all: prolix's own `B1-INFER` design doc found `lax.scan` unrolls per-step IR and measured 300-400x more compile time than `lax.while_loop` at production scale, and locked `lax.while_loop` for this path specifically because of that.

This is exactly `lax.while_loop`'s shape (`carry -> carry`, no `x`, no `y`) — and precisely why prolix's inference dispatch bypassed xtrax and used a hand-rolled carry (`_NLDispatchCarry`, a `NamedTuple {langevin_state, neighbor, did_overflow}`) instead. `grep -rn "while_loop" src/ .praxia/` in this repo returns zero hits — no `AxisStrategy` variant covers this shape today, and it has never been scoped here before.

### Why this belongs in xtrax, not just prolix

xtrax is genuinely load-bearing for the rest of prolix's `EnsemblePlan`: shape-bucket planning (`BatchPlanner`/`MemoryBudget`), `N_MOLS` dispatch (`Vmap`/`SafeMap`), duplicate-topology dedup (`DedupGather`), and the trajectory step-loop (`Scan`) are all xtrax-wired already. Only the carry-only inference loop isn't — a genuine, narrow gap in the `AxisStrategy` surface, not a modeling choice specific to prolix's physics. Any other xtrax consumer with an "iterate N times, keep only the final state, no per-step recording" loop (e.g. a fixed-point solver, an EM-refinement pass, any inference-only rollout) hits the identical gap.

## Reference implementation to generalize from (already shipped downstream)

`prolix/api/ensemble_plan.py`'s `_NLDispatchCarry` and `_run_single_inference`'s `_nl_step_fn` (landed 2026-07-18, prolix commit `7367f93`, verified in production use — zero ghost-position drift over 20 real steps with periodic neighbor-list updates every 3 steps; overflow-then-reallocate confirmed to recover cleanly) is a real, working instance of the pattern this strategy should generalize:

- **Periodic in-loop work gated by the loop's own step counter**, via `jax.lax.cond` (there: `neighbor.update()` every `nl_update_every` steps). `dispatch_n_steps_inference`'s step function has signature `(state, step_i) -> state`, reusing `lax.while_loop`'s own already-tracked iteration counter rather than threading a redundant counter through the carry itself — worth carrying into this strategy's own body-function convention below.
- **A field OR-accumulated across iterations without ever branching on it in-loop** (there: `did_overflow`), host-checked once after the loop returns.
- **Masked post-processing applied every iteration outside the "core" computation** (there: ghost-position/momentum pinning via `eqx.tree_at`) — orthogonal to the strategy itself, but confirms the carry can carry auxiliary state beyond what the "main" transition touches.

## Proposed design

### 1. New sealed-union member (`strategy.py`)

```python
@runtime_checkable
class WhileBodyFn(Protocol):
    """While-loop body: carry -> new_carry. No x (no input sequence), no y
    (no per-step output collection) -- narrower than ScanTransition."""

    def __call__(self, carry: Any) -> Any: ...


@runtime_checkable
class WhileCondFn(Protocol):
    """While-loop continuation predicate: carry -> bool (traced scalar)."""

    def __call__(self, carry: Any) -> Any: ...


@dataclass(frozen=True)
class WhileCarry:
    """Carry-only strategy: compiles to lax.while_loop. No per-step output
    collection -- only the final carry matters. For inference-only /
    fixed-point-style loops where recording a trajectory is either
    unwanted or handled by a separate Scan-based dispatch path.

    Not reverse-mode AD safe (lax.while_loop has no VJP rule) -- same
    caveat class as Scan's heterogeneous-axis restriction, just a
    different mechanism (Scan forbids the axis; WhileCarry forbids grad).
    """

    body: WhileBodyFn | None = None
    cond: WhileCondFn | None = None
    init: Any | None = None


AxisStrategy = Vmap | SafeMap | Scan | DedupGather | Bucket | WhileCarry
```

`cond` needs a default for the common "run exactly N steps" case — proposed as a small helper next to `WhileCarry` rather than a method on it (keeps the dataclass a plain, frozen data holder, consistent with the rest of the union):

```python
def fixed_step_count_cond(n_steps: int) -> WhileCondFn:
    """cond=fixed_step_count_cond(200) for `run exactly 200 steps`. Assumes
    the carry is a 2-tuple (step_i, state) or exposes a `.step_i` field --
    see WhileLoopIterator's carry-shape contract below."""
```

### 2. New iterator (`iterator.py`)

`JaxScanIterator.__call__` returns `(final_carry, ys)` — a **different arity** than what `WhileCarry` needs (no `ys` exists). Rather than overload one iterator's return shape on a runtime flag, `WhileLoopIterator` is a distinct class with its own, narrower contract:

```python
class WhileLoopIterator(eqx.Module):
    """Iterate via jax.lax.while_loop -- carry-bearing, no output collection.

    Unlike JaxScanIterator, returns only the final carry (no `ys`) --
    genuinely nothing to stack, since there's no per-step output.
    """

    def __call__(self, cond: Any, body: Any, init: Any) -> Any:
        """Apply fn using jax.lax.while_loop.

        Args:
            cond: Callable(carry) -> bool (traced scalar continuation predicate).
            body: Callable(carry) -> new_carry.
            init: Initial carry value.

        Returns:
            final_carry: The carry after the loop's condition first fails.
        """
        return jax.lax.while_loop(cond, body, init)
```

Carry-shape convention (mirroring prolix's `_nl_step_fn(state, step_i)` rather than threading a redundant counter through the carry): the iterator itself is agnostic to carry *shape* — `WhileCarry.body`/`.cond` close over whatever structure the caller needs (a bare state, or `(step_i, state)`), same as `Scan.transition` already does for `(carry, x)`. `fixed_step_count_cond` above documents the specific `(step_i, state)` convention it assumes; it is a convenience helper, not a structural requirement of `WhileLoopIterator` itself.

### 3. `make_axis_dispatch` (`dispatch.py`)

New branch, same shape as the existing `Scan`/`Vmap`/`SafeMap` branches:

```python
if isinstance(strategy, WhileCarry):
    return WhileLoopIterator()
```

Rejection logic mirrors `Scan`'s heterogeneous-axis check exactly — `lax.while_loop` has the identical static-carry-shape constraint across loop iterations that `lax.scan` has across scanned elements:

```python
if isinstance(strategy, WhileCarry):
    het_axes = heterogeneous_axes or set()
    if axis in het_axes:
        raise DispatchRejected(
            f"Cannot use WhileCarry strategy on {axis} axis: {axis} axis contains "
            "heterogeneous (variable-shape) state elements. lax.while_loop requires "
            "static carry shape across all iterations, same constraint as Scan.",
        )
```

`axis_dispatch`'s eager shim (the backward-compat path) gets a matching `elif isinstance(strategy, WhileCarry):` branch calling `jax.lax.while_loop(strategy.cond, strategy.body, strategy.init)` directly, mirroring how `Scan`'s eager branch calls `safe_scan` directly.

### 4. `CarrySpec` (`carry.py`) / `BatchPlanner.plan()` (`plan.py`)

`CarrySpec` today unconditionally pre-demotes to `Scan` (Phase 0, `plan.py:214-236`) — it has no notion of "this carry doesn't want per-step output." Rather than a wholly separate `WhileCarrySpec` type (duplicating `axis_name`/`init`/`ordered_sinks`), add one field to keep the declarative surface small:

```python
@dataclass(frozen=True)
class CarrySpec:
    axis_name: str
    init: Any
    transition: ScanTransition
    ordered_sinks: bool = True
    collect_outputs: bool = True  # NEW. False -> WhileCarry instead of Scan.
```

Phase 0's pre-demotion (`plan.py:214-236`) branches on this field:

```python
if spec.name in carry_by_name:
    cs = carry_by_name[spec.name]
    if spec.name in self.heterogeneous_axes:
        raise ValueError(...)  # unchanged
    if cs.collect_outputs:
        strategy = Scan(init=cs.init, transition=cs.transition, ordered_sinks=cs.ordered_sinks)
        reasoning = f"carry-bearing scan (CarrySpec declared for '{spec.name}')"
    else:
        strategy = WhileCarry(init=cs.init, body=cs.transition, cond=cs.cond)
        reasoning = f"carry-only while-loop (CarrySpec declared for '{spec.name}', collect_outputs=False)"
    decisions.append(AxisDecision(spec=spec, batch_size=1, reasoning=reasoning, strategy=strategy))
    continue
```

`cs.transition`'s type would need to accept either a `ScanTransition` (`(carry, x) -> (carry, y)`) or a `WhileBodyFn` (`carry -> carry`) depending on `collect_outputs` — this is the one place the two strategies' narrower/wider contracts have to reconcile at the `CarrySpec` boundary; worth a dedicated review pass in implementation, not resolved further here. A `cond: WhileCondFn | None = None` field is also needed on `CarrySpec` for the `collect_outputs=False` case (unused when `True`).

Default `collect_outputs=True` preserves every existing `CarrySpec` caller's behavior exactly — this is a strictly additive change to `AxisDecision`'s reachable strategy set, not a modification to any existing decision path.

## Non-goals

- Does not implement reverse-mode AD support for `WhileCarry` — `lax.while_loop` has no VJP rule in JAX itself; this is a hard upstream JAX constraint, not a design choice this spec can route around. Consumers needing gradients through a carry-only loop must use `Scan` (with `collect_outputs=True` and simply discarding `ys`) instead, same tradeoff `Scan`'s own heterogeneous-axis restriction already documents.
- Does not attempt to unify `WhileLoopIterator` and `JaxScanIterator` into one class — their return arities are genuinely different (`final_carry` vs. `(final_carry, ys)`), and forcing one shape would mean either a footgun default (`ys=None` silently) or a runtime branch on a static strategy choice that's already known at dispatch time.
- Does not migrate prolix's `_NLDispatchCarry`/`_nl_step_fn` to this strategy once implemented — that is a separate, prolix-side follow-up (already noted as a non-goal in prolix's own scoping doc), justified independently once `WhileCarry` is real and reviewed here.

## Status

Scoping only, written up in xtrax's own repo/conventions from prolix's draft (`.praxia/docs/decisions/` note, task `260715_b1_physics_parity`) after prolix's own scoping session confirmed the gap is real, non-speculative (grounded in a shipped, verified reference implementation), and structurally distinct from every existing `AxisStrategy` variant. Grounded directly against this repo's current `strategy.py`/`dispatch.py`/`iterator.py`/`carry.py`/`plan.py` (as of `main` @ `1d46c05`), not against prolix's paraphrase of them. No implementation attempted here — next step is review under xtrax's own process before any code lands.
