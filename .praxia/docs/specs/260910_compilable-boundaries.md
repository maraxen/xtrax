---
category: specs
title: "Compilable boundaries: classifying which hooks, taps, and sinks can cross the export boundary"
description: "Spec for an effect-classification scheme over Fuse/Tap/Sink and the 7 training Callback hooks — SplitTap makes taps exportable structurally, the DedupGather and Vmap-over-Scan boundary holes are closed by refusal, and JAX's anonymous host_callbacks refusal gains a named xtrax error"
task_id: 260910_compilable-boundaries
status: draft
---

# Specification: Compilable boundaries

## Overview

Give every boundary op — `Fuse`, `Tap`, `Sink`, and the seven `training.Callback` hooks — a
place in a decidable classification of *where its effect can live*: inside the compiled
artifact, lifted out of it and replayed from the artifact's output, or irreducibly host-only;
and make the one op currently rejected on a false premise (`Tap`) exportable by a structural
split rather than a declared precondition.

## The organizing question

An export target runs a compiled artifact with no Python. Every side effect in a plan must
therefore end up in exactly one of three places:

- **Lifted in** — the "effect" is not an effect at all; it is a pure computation the graph can
  carry verbatim. Nothing to do.
- **Lifted out and replayed** — the effect's entire observable content is a function of a value
  the graph already computes and returns, so the call is removed before tracing and the host
  re-derives it from the artifact's output.
- **Host-only** — the effect depends on something the artifact cannot produce (inter-call host
  state, wall-clock ordering, a file handle), so the plan must be refused rather than silently
  exported minus the effect.

`xtrax` already implements exactly one instance of the middle case: `AxisBoundary.materialize`
(`src/xtrax/stages/boundaries.py:99`), which strips a declared-materializing `Sink`
(`src/xtrax/export/pipeline.py:86-117`) and lets the caller read the values off the exported
output. This spec generalizes that from one hard-coded case to a stated classification, and
extends it to `Tap`.

---

## 1. Effect classification

### 1.1 The four classes

| Class | Meaning | Exporter action | Members |
|---|---|---|---|
| **PURE** | A JAX-traceable function of its input. No host call. | Keep in graph verbatim. | `Fuse`; `SplitTap.transform` (new) |
| **MATERIALIZABLE** | Host effect whose payload is exactly a value the exported callable already returns. | Remove the call before tracing; caller reads the output. | `Sink` with `materialize=True`; `SplitTap.observe` with `materialize=True` (new) |
| **HOST-ONLY** | Host effect whose payload or correctness is not recoverable from the output. | Refuse the plan, naming the axis and the op. | Every `Sink`/`Tap` not declared materializing; any legacy `Tap`; every boundary op on a `DedupGather` axis (§5.1) |
| **OUT-OF-TRACE** | Runs in host Python that was never inside a trace to begin with. | Nothing — there is no boundary to cross. | All 7 `training.Callback` hooks (§4) |

`PURE` and `OUT-OF-TRACE` are distinct even though neither requires exporter action: `PURE` ops
are *inside* the traced callable and constrain what may appear in the jaxpr; `OUT-OF-TRACE` ops
are outside it and constrain nothing.

**Measured, not assumed: a host callback never reaches IREE.** `jax.export.export(jax.jit(f))`
on a function containing an `io_callback` raises `NotImplementedError: serialization of
host_callbacks is not yet implemented` at `jax/_src/export/_export.py:1047` — before IREE is
called at all. The same function at the same shapes with the callback removed exported and
compiled clean, 9029 bytes (jax 0.11.1, IREE 3.11). `xtrax`'s export path is exactly that call
(`src/xtrax/export/pipeline.py:236`), and `grep -rn "host_callbacks" src/ tests/` returns zero
hits, so xtrax neither works around the restriction nor tests it today.

That measurement is the reason `materialize` and `_StrippedSink` exist: stripping is not an
optimization, it is the only way a boundary-bearing plan exports at all. The HOST-ONLY row is
therefore a measured refusal, not a design preference, and MATERIALIZABLE is the only route by
which a host effect's payload survives into an artifact.

### 1.2 What the caller declares, and what the validator checks

Declaration surface — **no new `AxisBoundary` field**:

| Caller writes | Declares |
|---|---|
| `AxisBoundary(fuse=f)` | `f` is PURE |
| `AxisBoundary(sink=s, materialize=True)` | `s` is MATERIALIZABLE (unchanged from today) |
| `AxisBoundary(tap=t, materialize=True)` where `t` satisfies `SplitTap` | `t.transform` is PURE, `t.observe` is MATERIALIZABLE (new) |
| anything else with a `tap` or `sink` under `export_safe=True` | HOST-ONLY → rejected |

Validator checks, all in `validate_plan_topology(..., export_safe=True)`
(`src/xtrax/stages/topology.py:241`) unless noted:

1. **Kind** — which of `fuse`/`tap`/`sink` are non-`None`.
2. **Tap shape** — `isinstance(boundary.tap, SplitTap)`, a `runtime_checkable` structural check
   for the presence of `transform`, `observe`, and `ordered`. A tap failing it is rejected
   whatever `materialize` says; a tap passing it is accepted only alongside `materialize=True`
   (AC-5).
3. **Something to materialize** — existing `MaterializeWithoutSinkError` (`topology.py:211`),
   narrowed by exactly one case. `topology.py:211` today reads
   `if materialize and boundary.sink is None`, which kills a materializing `SplitTap` the moment
   check 2 stops raising. It becomes:
   ```
   if materialize and boundary.sink is None and not isinstance(boundary.tap, SplitTap):
   ```
   `materialize=True` with neither a sink nor a `SplitTap` still raises, message unchanged
   (AC-7).
4. **No fuse on a materializing axis** — existing `MaterializeFuseConflictError`
   (`topology.py:37-52`), extended to cover a materializing tap for the identical reason
   (§2.4). The guard at `topology.py:229` today reads
   `if boundary.sink is not None and materialize and boundary.fuse is not None`; the
   `sink is not None` conjunct must go, leaving "a materializing axis carrying a `fuse`". Left
   in place, a tap-only materializing boundary with a `fuse` never reaches the error (AC-6).
5. **At most one materializing axis per plan** — existing `MultipleMaterializeAxesError`
   (`topology.py:322-330`); a tap-materializing axis counts against the same budget.
6. **Strategy compatibility** — `DedupGather` axes may carry no boundary op (§5.1). A
   `Vmap`-over-`Scan` plan on the unbatched-carry route may carry no materializing inner
   boundary (§5.3); that one is checked in the composer, because route selection depends on
   `scan_init`'s shape, which topology validation never sees.
7. **Residual callback** — a callback that survives the strip is caught at the
   `jax.export.export` call itself (`pipeline.py:236`), not by a jaxpr scan and not in
   `topology.py`. JAX already refuses it, measured above: `NotImplementedError: serialization
   of host_callbacks is not yet implemented` (`jax/_src/export/_export.py:1047`), raised before
   IREE. The deliverable is a named xtrax error carrying the plan's boundary-bearing axes and
   the pointer to `materialize=True`, in place of an internal JAX message that names neither
   (§8 Task 6).

`_check_export_boundary`'s `Raises:` docstring (`topology.py:188-191`) states the old kind rule
("tap present, or a sink not declared materializing") and must be rewritten to match checks 2-4.

### 1.3 Honest limits of the scheme

`materialize` today "is a precondition on `sink`, not a proof about it"
(`boundaries.py:115-118`). That statement is precise about *what kind* of unproven thing it is,
and the distinction matters for this spec:

- For a **sink**, the unproven part is **replay completeness**: does reading the exported output
  actually give the caller what the sink would have written? Graph correctness is not at risk —
  the sink's return value is discarded (`executor.py:129-130`), so deleting the call cannot
  change any traced value.
- For a **tap** under a hypothetical "declare it identity" design, the unproven part would be
  **graph correctness**: `tap`'s return value *replaces* the step output
  (`executor.py:127-128`), so stripping a tap that is not truly identity silently changes the
  exported program's numerics. That is a strictly worse failure than an incomplete replay, and
  it is why this spec does not extend `materialize`'s declare-and-trust pattern to taps.

**This spec therefore fixes the weakness for taps and inherits it for sinks.** Taps get a
structural split (§2), where the pure part is a separate callable that is traced and checked and
the effectful part is dropped — nothing is asserted about a callable the exporter cannot see.
Sinks keep the existing precondition, unchanged; see Out of Scope.

The residual-callback gate (check 7) narrows the sink weakness without closing it: a
"materializing" sink that leaves a callback in the trace fails with an xtrax error naming the
axis instead of a JAX-internal message naming nothing, but neither says anything about whether
the sink's replay is complete.

---

## 2. The Tap case

### 2.1 The current refusal, and why its stated reason does not support it

`topology.py:201-209` rejects any `Tap` under `export_safe=True`, unconditionally, with:

> A Tap is T -> T and participates in dataflow, so it cannot be stripped for export on any
> target.

`boundaries.py:112-113` states the same claim. The claim is **true but not load-bearing**. It
establishes that a tap is not *droppable* — you cannot delete `y = boundary.tap(y)` and keep the
graph's meaning, because the deleted call supplied `y`. It does not establish that a tap is not
*materializable*. Droppability and materializability are different properties, and every
materializing sink today is a case of the second without the first.

A `Tap` is `Sink + transform` — `boundaries.py:49-51` says "Identity transform with side effect…
Value continues downstream unchanged", and the executor's tests certify the transform is free to
be non-identity (§2.2). Either way the effect and the dataflow are separable in the concept: one
half computes a value, the other half looks at it. They are not separable in the current *type*:
`Tap.__call__` is one opaque callable that
does both, so the exporter has no handle on either half independently. That — not the T->T
signature — is the actual blocker.

### 2.2 `SplitTap`: separate the halves in the type

Add to `src/xtrax/stages/boundaries.py`:

```
@runtime_checkable
class SplitTap(Protocol, Generic[T]):
    ordered: bool
    def transform(self, x: T) -> T: ...   # pure JAX; stays in the graph
    def observe(self, x: T) -> None: ...  # host effect; lifted out at export
    def __call__(self, x: T) -> T: ...    # eager: y = transform(x); observe(y); return y
```

`SplitTap` is a distinct Protocol, not a widening of `Tap`. `Tap` is unchanged, and a legacy
`Tap` remains HOST-ONLY.

**The observation payload is the post-transform value.** This is the decision that makes the
whole mechanism cost nothing. `observe` is called on `y = transform(x)` — the same value that
continues downstream and gets stacked. So the stacked observation stream is *bit-identical to
`ys`*, the value the exported callable already returns. No second output slot, no executor
change, no shape-contract change. It degenerates to exactly the sink case, where "the executor's
per-step return value already equals what `sink` receives" (`boundaries.py:105-107`).

The cost of that decision: a tap that wants to observe its *input* while returning something
different is not expressible as a `SplitTap`. That shape is HOST-ONLY here.

**The narrowing cannot be warranted by `Tap`'s docstring**, which calls a tap an "identity
transform with side effect" (`boundaries.py:49-51`). The executor's tests certify the opposite:
`tests/stages/test_executor.py:142-149` (`test_scan_tap_transforms_y_not_carry`) pins
`ys == [0, 10, 20, 30]` for `tap=lambda y: y * 10`, and `:119-126` pins the same non-identity
behavior on the map path. A non-identity tap is the tested contract, so an argument that "nothing
conforming to the docstring loses expressiveness" rests on a docstring the tests falsify.

**The warrant is scope.** The narrowing applies only to the new export route. Legacy `Tap` is
untouched: same type, same eager semantics, same tests, still HOST-ONLY under `export_safe=True`.
`SplitTap` is purely additive, and no plan can export a tap at all today, so no working
configuration is narrowed — every existing tap keeps working exactly as it does now. Task 1
rewrites `boundaries.py:49-51` to describe what the tests certify rather than what the docstring
currently claims.

### 2.3 Export mechanism

Mirror `_StrippedSink` (`src/xtrax/export/pipeline.py:62-84`) with a transform-only stand-in:

```
class _ObserveStrippedTap:
    __slots__ = ("_transform", "ordered")
    def __init__(self, transform, ordered): ...
    def __call__(self, x):
        return self._transform(x)
```

`_boundaries_for_export` (`pipeline.py:86-117`) gains a branch: when
`getattr(boundary, "materialize", False)` and `boundary.tap` satisfies `SplitTap`, replace
`tap` via the same `dataclasses.replace` call already used at `pipeline.py:113`.

**`ordered` must be preserved on the stand-in**, for the reason already established for
`_StrippedSink` at `pipeline.py:62-74` and `pipeline.py:112`: the executor's branch selection
reads `.ordered` off the tap (`executor.py:107-113`, `executor.py:201`), so dropping the flag
flips an ordered `SafeMap` axis from `jax.lax.map(wrapped, xs)` onto
`safe_map(..., batch_size=...)`, which raises when the axis's cardinality **exceeds** the batch
size and is not divisible by it (`src/xtrax/transforms/map.py:33-37`). At or below the batch
size, `map.py:28-30` short-circuits to `jax.vmap` and no divisibility applies. A working
configuration would crash at export.

`_StrippedSink`'s docstring states that restriction without the `n > batch_size` qualifier
(`pipeline.py:69`). Task 4 adds the qualifier there and writes `_ObserveStrippedTap`'s docstring
with it.

### 2.4 Output-shape contract

**The exported callable's output pytree is unchanged.** For a one-axis plan with a
tap-materializing axis, `build_traceable_callable`'s returned callable produces exactly the
structure it would produce for the same plan with `boundary=None`: the (unfused) stacked `ys`.
This falls directly out of §2.2's post-transform decision.

The claim is scoped to one-axis plans deliberately. On the two-axis Vmap-over-Scan
unbatched-carry route it holds for a degenerate reason — that route returns the final carry with
or without a boundary — and there it is a symptom rather than a property (§5.3).

Interaction with the one-materialized-axis rule (`topology.py:322-330`): **unchanged, and a
tap-materializing axis counts toward the same budget of one.** Rationale, not deferral: the
composer accepts a one-axis plan or the certified Vmap-over-Scan two-axis shape, and in the
two-axis shape an outer boundary is refused outright (`composer.py:236-248`). At most one axis in
any composable plan can carry a boundary at all, so raising the limit would generalize into code
that cannot be reached.

Interaction with `fuse`: **a materializing tap on an axis that also has a `fuse` is rejected,
with the existing `MaterializeFuseConflictError`.** Same mechanism, same reason as for sinks
(`topology.py:44-47`): `_apply_fuse` (`executor.py:136-140`) collapses the stacked `ys` that the
observation payload *is*, and exposing both the pre-fuse stream and the fused value would require
the executor to return a second value.

### 2.5 Exactly what must change

**Error message — must change.** `topology.py:201-209`. Its current text asserts a claim that
becomes false ("cannot be stripped for export on any target"). New text must (a) name `SplitTap`
as the supported route, (b) keep the substring `has a Tap`, (c) state the actual reason for
refusing a legacy tap — the exporter cannot see the pure half, so stripping would change traced
values.

**`tests/stages/test_topology.py:286` `test_rejects_tap_even_when_materialize_is_set` — body
survives verbatim.** It constructs a legacy `_OrderedTap`, which remains HOST-ONLY, and matches
on `"has a Tap"`, which constraint (b) preserves. Two things around it must change:

- its inline comment at line 287, "A Tap is T -> T and feeds downstream; materialize never
  applies to it" — becomes false;
- `TestExportSafeRule3`'s class docstring at line 284, "a tap always rejects" — becomes false.

**`boundaries.py:112-113`** ("Never applies to `tap`: … not droppable on any target") must be
rewritten to distinguish droppable from materializable and point at `SplitTap`.

**Public docs — must change.** Three places state the old absolute:

- `docs/api/export.md:152-155` — "**`tap`** never crosses. A Tap is `T -> T` and feeds
  downstream, so it cannot be dropped on any target." Becomes the `SplitTap` +
  `materialize=True` route, with legacy `Tap` still refused.
- `agent_assets/skills/using-xtrax/references/run.md:243-245` — the `AxisBoundary` example
  annotates `tap=MyTap()` as "Identity + side effect (outside jit, via io_callback)". Gains the
  split form.
- `agent_assets/skills/using-xtrax/references/run.md:277` — the enumerated topology rules stop
  at rule 2 (`ordered=True` on a `Vmap` axis) and never mention the `export_safe=True` rules at
  all. Gains the tap/sink export rules including the `SplitTap` route.

**`executor.py`'s module docstring** gains no change: the executor is untouched by this spec.

**`src/xtrax/inference/ir_schema.py` is not affected.** `_axis_boundary_json_schema`
(`ir_schema.py:212-220`) is hand-written with literal `$ref`s and reflects over neither
`boundaries.__all__` nor `AxisBoundary`'s fields — it already omits `materialize`.
`_protocol_to_json_schema(Tap)` (`ir_schema.py:270`) is a function of `Tap` alone, which this
spec does not change, and a `SplitTap` still satisfies `isinstance(_, Tap)`. Noted here so a
fixer does not go looking for schema drift that cannot occur.

---

## 3. The `ordered` dimension

Ordering under `ordered=True` is a property of the *executed* graph: JEP-10657 token threading
creates an XLA data dependency between consecutive calls (`executor.py:22-36`). Strip the calls
and the tokens go with them. What remains is not "no ordering" — it is a different, weaker, and
often sufficient guarantee.

### 3.1 What a materialized boundary still guarantees

**Index order.** For a materializing axis, `ys[i]` is exactly the value the stripped
`observe`/`sink` would have received for element `i` of `xs`, for every one of the three
executable strategies:

| Strategy | Mechanism | Index order recoverable |
|---|---|---|
| `Scan` | `safe_scan` stacks per-step `y` in step order (`executor.py:247`) | Yes — stacked index == scan step |
| `SafeMap` | `jax.lax.map` / `safe_map` stack along the leading axis (`executor.py:211-213`) | Yes — stacked index == element index |
| `Vmap` | `jax.vmap` maps input axis 0 to output axis 0 (`executor.py:182`) | Yes — stacked index == lane index |
| `DedupGather` | boundary never fires (§5.1) | N/A — rejected |

The `Vmap` row is measured, not inferred: an unordered `io_callback` under `jax.vmap` fires
**once per lane**. `jax.vmap(f)(jnp.arange(4))` produced `n_calls=4` with
`shapes=[(), (), (), ()]` — four scalar calls, not one batched call. So the eager stream and the
materialized `ys` have the same cardinality on a `Vmap` axis, which is what makes AC-11's
cross-strategy comparison well-posed there.

**What is not recoverable: host-call order.** Any effect whose correctness depends on the
sequence of host calls rather than on the per-element payload — an append to a stream with an
implicit cursor, a monotone log sequence number, an accumulator living on the sink object — is
HOST-ONLY and must not be declared materializing. Nothing in the type system distinguishes these
from index-recoverable effects; this is precisely the replay-completeness precondition of §1.3.

### 3.2 Vmap

A `Vmap` axis has no step order at all, and `ordered=True` on a `Vmap` axis is rejected twice
over — at plan time (`topology.py:280-298`) and defensively at execution (`executor.py:171-180`)
— because JAX raises `ValueError: Cannot vmap ordered IO callback` regardless of nesting depth
(`executor.py:57-83`).

The consequence for materialization is a positive one: **under export, materializing is the only
way to observe per-element values off a `Vmap` axis.** The qualifier matters. Eagerly, an
unordered `Sink` or `Tap` on a `Vmap` axis is accepted today — `topology.py:280-298` rejects only
`ordered=True` — and fires once per lane (§3.1). Under `export_safe=True` that same unordered
sink is rejected unless declared materializing (`topology.py:219-227`), so materialization is the
export-side route and the only one. It yields index order (§3.1), a *stronger* guarantee than the
ordered io_callback route could ever deliver there, since that route is structurally unavailable.

**Rule 2 is not relaxed.** It would be tempting to accept `ordered=True` + `Vmap` when the
boundary materializes, since the callback is gone by the time the graph is built. This spec
refuses, because Rule 2 fires for every caller, not only export-safe ones
(`topology.py:280` sits above the `export_safe` early-continue at `topology.py:300-301`), and the
same `AxisBoundary` object serves both paths. Relaxing it under `export_safe` alone would admit a
plan that exports cleanly and cannot be run eagerly at all. A caller who wants observation off a
`Vmap` axis sets `ordered=False` and relies on index order.

### 3.3 `ordered` is inert in the artifact, but not free

After stripping, `ordered` has no semantic effect on the exported artifact — there is no callback
left to order. It still has a **lowering** effect, deliberately (§2.3): the stand-in preserves the
flag, so an ordered `SafeMap` axis is lowered through `jax.lax.map(wrapped, xs)` with
`strategy.batch_size` dropped (`executor.py:201-211`) — fully sequential, for an ordering
guarantee the artifact no longer provides.

This spec does not change that lowering (Out of Scope; the divisibility crash at
`transforms/map.py:33-37` is the reason the flag is preserved in the first place). It makes the
cost **visible**: `export_pipeline` emits a diagnostic through the existing
`ExportResult.diagnostics` / `_diagnostics_for` channel (`pipeline.py:59`, `pipeline.py:133-141`)
naming the axis, the `batch_size` that was ignored, and the fact that the ordering guarantee is
inert post-strip.

`_diagnostics_for` as written takes only a `CompileResult` (`pipeline.py:133`), which carries
neither the boundaries nor the strategies. The diagnostic needs both halves: `ordered` is read
off the boundary's tap or sink, and `batch_size` lives on the `SafeMap` strategy in
`decision.strategy`. So the emitting function takes **`plan.decisions` as well as the axis
boundaries**, alongside the compile result it takes today.

One entry per target is correct, not a duplication bug: `diagnostics` is a per-`ExportResult`
field (`pipeline.py:59`), populated per target inside the target loop (`pipeline.py:259`).

---

## 4. Hooks

### 4.1 Decision

**Option (b): classify the seven hooks and specify their relationship to the taxonomy; build no
bridge between `training.Callback` and `stages.AxisBoundary`.**

Reasoning:

- The hooks are already outside every trace. `Engine.fit` fires them in host Python around
  `trainer.step` (`src/xtrax/engine/engine.py:157-215`), and
  `src/xtrax/training/types.py:27` states the invariant: "All hooks run Python-side (outside JAX
  traces)." There is no compiled artifact for them to be inside of, so there is no boundary to
  cross — hence the `OUT-OF-TRACE` class in §1.1, which is a real classification result, not an
  evasion.
- A bridge (option a) would couple training instrumentation to per-axis pipeline boundaries with
  no shared consumer, no shared type, and no certification harness — the pattern
  `composer.py:157-204` explicitly warns against ("an uncertified extrapolation wearing a
  certified badge").
- Declaring them out of scope (option c) would drop a term the requirement names, and would leave
  `training/types.py:27`'s invariant an untested docstring assertion.

### 4.2 Classification

| Hook | Payload | Class | Why |
|---|---|---|---|
| `on_train_start(state)` | `ResumableState` | OUT-OF-TRACE, host-only | Fires once, outside the loop (`engine.py:157-158`). No traced payload. |
| `on_train_end(state)` | `ResumableState` | OUT-OF-TRACE, host-only | Fires in `finally` (`engine.py:212-215`), including on failure — a property only host code has. |
| `on_resume(state)` | `ResumableState` | OUT-OF-TRACE, host-only | Conditioned on a Python flag (`engine.py:160-162`). |
| `on_epoch_start(state, epoch)` | state + Python `int` | OUT-OF-TRACE, host-only | Epoch loop is host Python `for epoch in range(...)` (`engine.py:165`); `epoch` is never a traced value. |
| `on_epoch_end(state, epoch)` | state + Python `int` | OUT-OF-TRACE, host-only | Same; also gates checkpointing (`engine.py:201-203`). |
| `on_step_start(state)` | `ResumableState` | OUT-OF-TRACE, host-only | Carries the IR-capture trigger (`engine.py:177`, `telemetry/callback.py:114-115`), which is a compile-time host action by construction. |
| `on_step_end(state, metrics: dict[str, Array])` | state + traced-array metrics | OUT-OF-TRACE, **replay-shaped** | Its payload is entirely arrays produced by `trainer.step` and returned to host Python (`engine.py:184`). |

### 4.3 What "replay-shaped" means for `on_step_end`, and what this spec does about it

`on_step_end` is the one hook whose payload is the same *kind* of thing a MATERIALIZABLE boundary
produces: a stacked-or-scalar array the compiled step already returns. It is already consumed
after the step completes and already dispatched off the critical path
(`engine.py:186-192`, via `BoundedCallbackHandler`). So it is *already* a replay: if the training
step were ever exported, `metrics` would be an output slot and the hook would replay unchanged.

**The deliverable is a documented invariant plus two tests, not new machinery.** Concretely:

- Label each hook with its class in the `Callback` docstring (`training/types.py:24-38`), so the
  classification lives next to the protocol it describes.
- **Concreteness (AC-17).** Test the invariant at `training/types.py:27` that is today an
  unbacked assertion: a callback registered on a real `Engine.fit` run must receive **concrete**
  arrays. The hook body calls `numpy.asarray(v)` on every value in `metrics` and every array leaf
  of `state` and **appends the results to a module-level list**. The test body, after `fit`
  returns, asserts `len(recorded) == expected_step_count` and that every recorded leaf is a
  `numpy.ndarray`. Under a tracer, `numpy.asarray` raises
  `jax.errors.TracerArrayConversionError`. `isinstance(v, jax.core.Tracer)` is deliberately not
  used: it is a version-fragile private-ish surface, and this repo pins a JAX *range*
  (`PINNED_JAX_RANGE` at `src/xtrax/stages/_callback.py`, enforced at import,
  `_callback.py:110-111`), so a check that works on one version in the range and not another
  would be a latent false green.
- **Propagation (AC-18).** A hook that raises unconditionally must escape `fit`. Assert it on
  `on_epoch_start` (`engine.py:167-168`).

**`on_step_end` exceptions are swallowed by design, so neither assertion may be an `assert`
inside it.** `engine.py:186-192` submits the hook through `BoundedCallbackHandler.submit`, whose
`try/except Exception: logger.exception("callback error")` (`io.py:163-169`) absorbs everything
the hook raises — stated deliberately at `io.py:120-121` ("Exceptions in submitted coroutines are
logged but not propagated, allowing the training loop to continue"). A failed `assert` inside
`on_step_end` is logged and the test passes green. That is why AC-17 records from inside the hook
and asserts from outside it, and why the `len(recorded)` check is load-bearing rather than
decorative: it is the only thing that distinguishes "hook ran and its arrays were concrete" from
"hook never ran" and from "hook raised and was swallowed". `wait_all()` (`engine.py:195`)
guarantees every submitted hook has completed before `fit` returns, so the count is exact.

Six of the seven hooks are fired by direct call and do propagate — `on_train_start`
(`engine.py:157-158`), `on_resume` (`:161-162`), `on_epoch_start` (`:167-168`), `on_step_start`
(`:180-181`), `on_epoch_end` (`:198-199`), `on_train_end` (`:213-215`). `on_train_end` is
nonetheless the wrong venue for AC-18: it fires inside the `finally` (`engine.py:212-215`), so a
raise there would mask an in-flight training exception instead of demonstrating propagation.

**No shared type is introduced.** `Callback` and `AxisBoundary` remain fully separate seams. This
is an explicit decision, recorded here so a later reader does not mistake the shared vocabulary
for shared machinery.

---

## 5. The latent gaps found

`composer.py` routes a plan to exactly five callables. Anything else raises
`UnsupportedStrategyError` (`composer.py:128-134`), so the enumeration is complete and every
route below was audited for boundary handling:

| Route | Site | Boundary handling |
|---|---|---|
| `_run_map` (`Vmap`, `SafeMap`) | `composer.py:97` | Passes `boundary` to `execute_map_axis`. Correct. |
| `_run_scan` (`Scan`) | `composer.py:111` | Passes `boundary` to `execute_scan_axis`, returns `ys`. Correct. |
| `_run_dedup` (`DedupGather`) | `composer.py:123` | Passes **no** boundary. §5.1. |
| `_run_batched` (Vmap-over-Scan, batched carry) | `composer.py:273` | Applies tap and sink inline, returns the fused `ys`. Correct. |
| `_run_literal_vmap` (Vmap-over-Scan, unbatched carry) | `composer.py:279` | Fires the effects, returns `final_carry` only. §5.3. |

Two of the five mishandle boundaries. §5.2 is a third gap, in lowering rather than routing.

### 5.1 `DedupGather` axes silently discard their boundary — IN SCOPE

`compose_single_axis`'s `DedupGather` branch calls
`axis_dispatch(strategy, step_fn, xs)` (`src/xtrax/export/composer.py:119-126`) and passes no
boundary at all. `axis_dispatch` has no boundary parameter
(`src/xtrax/tiling/dispatch.py:120-122`), so a `DedupGather` axis's `fuse`, `tap`, and `sink`
never fire. `DedupGather` is nonetheless in `_EXPORTABLE_STRATEGIES` (`topology.py:175`), and
`_check_export_boundary` (`topology.py:178-239`) never looks at the strategy, so the boundary
passes validation and the artifact is built without it.

**The primary defect is a dropped `tap` or `fuse`.** Once §2.2 makes taps exportable, a
`DedupGather` axis's tap is silently absent from the graph: the artifact computes `fn(x)` where
the caller asked for `tap(fn(x))`. That is §1.3's graph-correctness class — strictly worse than
an incomplete replay, because the exported numerics differ from the eager ones. A dropped `fuse`
is the same class one level up: the artifact returns the unreduced stack, so its output shape
differs from what every other strategy produces for the identical plan.

**The sink case is secondary, and is a cardinality mismatch rather than value corruption.**
`axis_dispatch`'s dedup branch returns `gather_fn(deduped_ys, index_map)` at N positions
(`dispatch.py:169-178`), which is `[fn(xs[i]) for i in range(N)]` — every value is the right
value. What differs is how many: a conforming sink under `Scan`, `SafeMap`, or `Vmap` observes
one call per element the executor actually runs, and dedup runs K < N of them. So the
materialized output holds N correct values where the sink's own contract was K.

**The drop is universal, not export-specific.** `grep DedupGather src/xtrax/stages/` returns only
`topology.py:175`; the eager executor has no `DedupGather` route either, so an eager plan loses
its boundary the same way. This spec still scopes the rejection to `export_safe=True`: rejecting
outside it is a wider behavior change with no covering test and no demonstrated caller. Task 3
instead leaves an explanatory comment at `composer.py:119-126` recording the gap for whoever
wires eager dedup boundaries later.

**In scope**, on two grounds: it is a direct falsification of the invariant this spec extends
("the executor's per-step return value already equals what `sink` receives",
`boundaries.py:105-107`), so leaving it would make the new classification unsound on an
already-permitted strategy; and the fix is a rejection, which is cheap and fail-loud.
`_check_export_boundary` gains a strategy check: any non-`None` `fuse`/`tap`/`sink` on an axis
whose `type(decision.strategy).__name__` is `"DedupGather"` raises `PlanTopologyError`, with a
message naming the composer's missing boundary wiring and the workaround (move the boundary to
another axis, or apply the effect outside the exported function). That message text will be read
by a downstream `aminx` caller, so it must describe the drop accurately: the tap and fuse are
absent from the graph, and the sink's call count would not match its contract.

Neither in-repo `DedupGather` test attaches a boundary — `tests/stages/test_topology.py:379-380`
passes `axis_boundaries={}` and `tests/export/test_composer.py:123` leaves `boundary` at its
`None` default — so the rejection breaks no existing test.

### 5.2 Ordered `SafeMap` silently ignores `batch_size` — PARTIALLY IN SCOPE

`executor.py:211` runs `jax.lax.map(wrapped, xs)`, dropping `strategy.batch_size`, whenever the
axis has an ordered op. This is documented at length (`executor.py:38-55`,
`executor.py:203-210`) and is unconditional, not a corner case.

**In scope only as a diagnostic** (§3.3). The export path is where this is most wasteful — the
ordering guarantee is inert after stripping, yet the sequential lowering is retained — so the cost
is surfaced through `ExportResult.diagnostics`. **Out of scope as a fix:** dropping the flag on
the export stand-in would flip the lowering to `safe_map(..., batch_size=...)`, which raises when
the cardinality **exceeds** the batch size and is not divisible by it (`transforms/map.py:33-37`;
`map.py:28-30` short-circuits to `jax.vmap` at or below it) — exactly the crash `_StrippedSink`'s
docstring was written to prevent (`pipeline.py:62-74`, which states the restriction without that
qualifier). Whether it is removable by padding is Open Question 1.

### 5.3 Vmap-over-Scan's unbatched-carry route discards the materialized values — IN SCOPE

`compose_vmap_of_scan` has two routes, selected by `_init_is_batched(init, outer_n)`
(`composer.py:262`):

- **Batched carry.** `_run_batched` (`composer.py:273-277`) scans once over the batched carry,
  applies `tap` and `sink` inside `_batched_transition` (`composer.py:264-271`), and returns
  `_apply_fuse(ys, inner_boundary)`. The per-step values are in the output; materializing works.
- **Unbatched carry.** `_run_literal_vmap` (`composer.py:279-301`) maps `_lane` over the outer
  axis, and `_lane` (`composer.py:289-291`) does
  `final_carry, _ys = execute_scan_axis(fn, lane_init, inner_xs, inner_boundary)` and returns
  `final_carry` alone. The stacked `ys` are discarded, and `_apply_fuse` is never reached, so the
  inner `fuse` is dropped too.

`execute_scan_axis` still fires the inner sink or tap on every step of the literal route. So an
**eager** run of that plan gets its effects while the **exported** artifact returns only the
final carry, and materialization has nothing to read off the output. Which route runs depends on
the shape of `scan_init` — a value topology validation never receives.

**Fix: refuse the combination in the composer.** On the non-batched route, `compose_vmap_of_scan`
raises `MultiAxisCompositionError` when `getattr(inner_boundary, "materialize", False)`, naming
the batched-carry recipe (already documented at `composer.py:250-257`) as the workaround.
`materialize` is readable at that point: `_boundaries_for_export` rebuilds with
`dataclasses.replace` (`pipeline.py:113`), which preserves `materialize=True` on the stripped
boundary. Two comments become false and must be corrected in the same task —
`composer.py:324-325` ("this function does not know about `materialize`") and
`pipeline.py:232-233` ("The composer never sees `materialize` itself").

**Fixer trap.** §2.4's output-pytree claim survives this defect intact: with `boundary=None` the
literal route also returns `final_carry`, so a structural comparison passes while materialization
is broken. AC-10 and AC-11 are both scoped to one-axis plans, so neither can reach the two-axis
route at all. AC-14 is the only criterion that observes it.

Backlog **#5089** tracks this defect independently.

---

## 6. Backward compatibility

### 6.1 Plans that newly pass

| Was | Becomes | Site |
|---|---|---|
| `AxisBoundary(tap=<SplitTap>, materialize=True)` rejected under `export_safe=True` | accepted; tap's `observe` stripped, `transform` traced | `topology.py:201-209`, `pipeline.py:86-117` |

Every other currently-rejected combination stays rejected: a legacy `Tap` with
`materialize=True` (AC-4), a `SplitTap` **without** `materialize=True` (AC-5), and a `SplitTap`
on an axis with a `fuse` (AC-6).

The `SplitTap`-without-`materialize` case is load-bearing rather than pedantic. An implementation
written as `if tap is not None and not isinstance(tap, SplitTap): raise` lets it through
topology, and `_boundaries_for_export` strips only boundaries that declare `materialize`
(`pipeline.py:107`), so the tap survives unstripped and its `io_callback` reaches the trace —
where `jax.export.export` refuses it (§1.1). The rejection must be keyed on the pair, not on the
tap alone.

### 6.2 Plans that newly fail

| Was | Becomes | Site |
|---|---|---|
| Any `fuse`/`tap`/`sink` on a `DedupGather` axis accepted under `export_safe=True`, exporting an artifact that drops it — a tap absent from the graph, a fuse unapplied, a sink's N-vs-K call count | `PlanTopologyError` | `topology.py:178-239` (§5.1) |
| A two-axis Vmap-over-Scan plan with an unbatched `scan_init` and a materializing inner boundary, exporting an artifact that holds only the final carry | `MultiAxisCompositionError` naming the batched-carry recipe | `composer.py:279-301` (§5.3) |
| An exported callable whose post-strip trace still binds a host callback, failing with JAX's internal `NotImplementedError: serialization of host_callbacks is not yet implemented` (`jax/_src/export/_export.py:1047`), which names no axis | the same refusal, re-labelled as a named xtrax error naming the boundary-bearing axes and `materialize=True` | `pipeline.py:236` (§1.2 check 7) |

Rows 1 and 2 replace a silent wrong answer with a loud refusal. Row 3 changes no plan's outcome —
JAX already refuses — only the error a caller reads. The `DedupGather` change can break a
downstream caller (e.g. an `aminx` plan) whose export currently *succeeds*; the error message must
therefore name the workaround explicitly, not just the refusal.

### 6.3 `AxisBoundary` is not modified

**No field is added to `AxisBoundary`.** This is deliberate, and it avoids re-incurring the trap
`materialize`'s own docstring documents (`boundaries.py:124-128`): every field is
`eqx.field(static=True)`, i.e. treedef aux_data rather than a pytree leaf, so adding one
invalidates every `jax.jit` cache keyed on the old treedef and leaves an
`AxisBoundary` pickled before the field with no entry for it — a case that remains untested.
`SplitTap` is a `Protocol`, not a field; adding it changes no treedef and no pickle.

Any future extension that *does* add a field must carry that caveat forward verbatim.

### 6.4 Eager behavior

Unchanged for every existing plan. `boundaries.py:102-103` claims "Only `xtrax.export` reads
this", which is already inaccurate: `topology.py:199` reads `materialize` too, via `getattr`. The
conclusion survives on a corrected warrant — topology only *reads* the flag, and only under
`export_safe=True`, to decide whether to raise; it never acts on it, and `export_safe` defaults
to `False` (`topology.py:245`). Task 1 corrects that sentence to name the topology read.

The executor is not modified by this spec, and `SplitTap.__call__` gives an eager run of a split
tap the same `y = transform(x); observe(y); return y` sequence a conforming monolithic `Tap`
already performs.

---

## 7. Acceptance criteria

Each is independently testable and names the file it belongs in.

**AC-1** — `SplitTap` is exported from `xtrax.stages.boundaries`, is `runtime_checkable`, and
`isinstance` returns `True` for an object providing `transform`, `observe`, and `ordered`, and
`False` for one missing any of the three. *File:* `tests/stages/test_boundaries.py`

**AC-2** — Calling a `SplitTap` eagerly runs `transform` then `observe`, `observe` receives the
post-transform value (asserted by a `transform` that is not identity), and the return value is the
post-transform value. *File:* `tests/stages/test_boundaries.py`

**AC-3** — `validate_plan_topology([...], {"axis": AxisBoundary(tap=<SplitTap>,
materialize=True)}, export_safe=True)` does not raise. *File:* `tests/stages/test_topology.py`

**AC-4** — The same call with a legacy `Tap` (no `transform`/`observe`) raises
`PlanTopologyError` whose message contains both `"has a Tap"` and `"SplitTap"`. The existing
`test_rejects_tap_even_when_materialize_is_set` (`tests/stages/test_topology.py:286`) still passes
with an unmodified body. *File:* `tests/stages/test_topology.py`

**AC-5** — `AxisBoundary(tap=<SplitTap>)` **without** `materialize=True` under
`export_safe=True` still raises `PlanTopologyError`. A rejection keyed on the tap's type alone
would let this through, leaving an unstripped `io_callback` in the trace (§6.1). *File:*
`tests/stages/test_topology.py`

**AC-6** — `AxisBoundary(tap=<SplitTap>, fuse=<f>, materialize=True)` under `export_safe=True`
raises `MaterializeFuseConflictError`. The boundary carries **no sink**, which is what forces the
`sink is not None` conjunct out of `topology.py:229` (§1.2 check 4). *File:*
`tests/stages/test_topology.py`

**AC-7** — `AxisBoundary(materialize=True)` with neither a sink nor a tap still raises
`MaterializeWithoutSinkError`, with `"no sink"` still in the message. This is the case
`tests/stages/test_topology.py:321-325` already constructs; check 3's `SplitTap` exemption must
not delete that guard. *File:* `tests/stages/test_topology.py`

**AC-8** — Two axes each carrying a materializing boundary — one a `SplitTap` tap, one a sink —
raise `MultipleMaterializeAxesError` naming both axes. *File:* `tests/stages/test_topology.py`

**AC-9** — `_boundaries_for_export` replaces a materializing `SplitTap` with a stand-in whose
`ordered` equals the original's, whose call returns `transform(x)`, and which is not the original
object; and returns the input mapping **by identity** when no axis materializes. *File:*
`tests/export/test_boundary_stripping.py` (new)

**AC-10** — Output-arity invariance: for a one-axis plan, the output pytree structure
(`jax.tree_util.tree_structure`) and every leaf's shape/dtype of the callable built with a
materializing `SplitTap` boundary equal those of the callable built with `boundary=None`. *File:*
`tests/export/test_composer.py`

**AC-11** — Index-order recovery, parameterized over `Scan`, `SafeMap`, and `Vmap`. `observe` has
signature `observe(self, x: T) -> None` and receives no index, so the index is recovered from the
value instead: `transform` is `lambda x: x * 10 + 1` over `xs = jnp.arange(n)`, which is
injective, so `i = (v - 1) // 10`. The eager run's `observe` appends the **value only**; the test
recovers each index, asserts the recovered set equals `range(n)`, and asserts the stripped run's
`ys[i] == 10 * i + 1` for every `i`. This assumes nothing about firing order and reads identically
on all three strategies. For the `SafeMap` case, `n` must be **greater than `batch_size` and
divisible by it** — `transforms/map.py:28-30` short-circuits to `jax.vmap` when
`n <= batch_size`, which would silently re-run the `Vmap` case under a `SafeMap` label. *File:*
`tests/export/test_composer.py`

**AC-12** — **Fail-loud residual-callback gate, with both controls.** (a) *Negative control:* a
plan whose boundary is declared materializing but whose step function still binds a host callback
raises a named xtrax export error whose message names the plan's boundary-bearing axes and points
at `materialize=True`, and `compile_for_target` is never reached. (b) *Positive control:* the same
plan with the callback removed exports successfully. A gate that only ever sees case (b) would
pass vacuously; (a) is what makes it a gate. *File:*
`tests/export/test_pipeline_native_wasm32.py`

**AC-13** — `AxisBoundary(sink=..., materialize=True)` on a `DedupGather` axis under
`export_safe=True` raises `PlanTopologyError`, and the message names both `DedupGather` and a
workaround. A sibling case with `tap=` and one with `fuse=` raise identically. *File:*
`tests/stages/test_topology.py`

**AC-14** — Vmap-over-Scan route selection. A two-axis plan with an **unbatched** `scan_init` and
a materializing inner boundary raises `MultiAxisCompositionError`, and the message names the
batched-carry recipe. The same plan with a **batched** `scan_init` exports, and its output holds
the per-step values, not just the final carry. *File:* `tests/export/test_multi_axis.py`

**AC-15** — An ordered materializing boundary on a `SafeMap` axis with a non-`None` `batch_size`
produces an `ExportResult.diagnostics` entry naming the axis and the ignored `batch_size`; the
same plan with `ordered=False` produces no such entry. *File:*
`tests/export/test_pipeline_native_wasm32.py`

**AC-16** — The `Callback` docstring (`src/xtrax/training/types.py:24-38`) labels all seven hooks
with their class from §4.2, and a test asserts that each of the seven hook names appears in
`Callback.__doc__` alongside a class label — so a hook added later without a label fails the
gate. *File:* `tests/training/test_types.py`

**AC-17** — Hook payload concreteness. A callback on a real `Engine.fit` run calls
`numpy.asarray(v)` inside `on_step_end` for every value in `metrics` and every array leaf of
`state`, appending the results to a module-level list. After `fit` returns, the test asserts
`len(recorded) == expected_step_count` and that every recorded leaf is a `numpy.ndarray`. The
length assertion is the guard against a hook that never ran or that raised and was swallowed
(`io.py:163-169`); `wait_all()` (`engine.py:195`) makes the count exact by the time `fit`
returns. *File:* `tests/engine/test_engine.py`

**AC-18** — Hook exception propagation. An `on_epoch_start` that raises unconditionally
propagates out of `Engine.fit`. `on_train_end` must not be used — it fires in a `finally`
(`engine.py:212-215`), so a raise there masks an in-flight exception — and neither must
`on_step_end`, which is swallowed by design (§4.3). *File:* `tests/engine/test_engine.py`

**AC-19** — `AxisBoundary`'s field set is exactly `{"fuse", "tap", "sink", "materialize"}`,
pinning §6.3's no-new-field decision so a later change that adds one has to confront the
treedef/pickle caveat deliberately. *File:* `tests/stages/test_boundaries.py`

---

## 8. Task decomposition

Ordered by dependency. Each task is scoped for one fixer session.

**Task 1 — `SplitTap` protocol and docstring corrections.**
*Files:* `src/xtrax/stages/boundaries.py` (modify).
Add the `SplitTap` Protocol per §2.2 and to `__all__`. Add no field. Three docstring corrections:
`materialize`'s "Never applies to `tap`" paragraph (`boundaries.py:112-113`) per §2.5;
`Tap`'s "Identity transform with side effect" (`boundaries.py:49-51`), which the executor's tests
falsify (§2.2); and `materialize`'s "Only xtrax.export reads this" (`boundaries.py:102-103`),
which omits the `topology.py:199` read (§6.4).
*Gate:* AC-1, AC-2, AC-19 — `uv run --extra dev --extra io pytest tests/stages/test_boundaries.py -q`
*Scope:* ~45 LOC.

**Task 2 — Topology: accept a split tap, extend fuse conflict and the materialize budget.**
*Files:* `src/xtrax/stages/topology.py` (modify), `tests/stages/test_topology.py` (modify).
In `_check_export_boundary` (`topology.py:178-239`): replace the unconditional tap rejection
(`topology.py:201-209`) with the `isinstance(tap, SplitTap)` check plus the `materialize=True`
requirement (AC-5); rewrite the message per §2.5's three constraints; add the `SplitTap` exemption
to `topology.py:211` and drop the `sink is not None` conjunct from `topology.py:229`, both per
§1.2 checks 3-4; return `True` for a materializing tap so it counts toward `materializing_axes`
(`topology.py:264`, `topology.py:322-330`); rewrite the `Raises:` docstring
(`topology.py:188-191`). Update the comment at `tests/stages/test_topology.py:287` and the class
docstring at line 284.
*Depends on:* Task 1. *Gate:* AC-3, AC-4, AC-5, AC-6, AC-7, AC-8. *Scope:* ~80 LOC.

**Task 3 — Topology: reject boundary ops on `DedupGather` axes.**
*Files:* `src/xtrax/stages/topology.py` (modify), `tests/stages/test_topology.py` (modify),
`src/xtrax/export/composer.py` (modify — comment only).
Per §5.1. Leave an explanatory comment at `composer.py:119-126` recording that the branch passes
no boundary and that the eager path has no dedup boundary route either; change no composer
behavior there. Independent of Task 2's tap logic, but both touch `_check_export_boundary`, so
sequence after Task 2 to avoid a conflict.
*Depends on:* Task 2. *Gate:* AC-13. *Scope:* ~35 LOC.

**Task 4 — Export: `_ObserveStrippedTap` and the stripping branch.**
*Files:* `src/xtrax/export/pipeline.py` (modify),
`tests/export/test_boundary_stripping.py` (create).
Per §2.3, mirroring `_StrippedSink` (`pipeline.py:62-84`) including the `ordered` preservation and
its rationale. Extend `_boundaries_for_export` (`pipeline.py:86-117`), keeping the
identity-return-when-unchanged behavior at `pipeline.py:117`. Add the `n > batch_size` qualifier
to `_StrippedSink`'s docstring (`pipeline.py:69`) and write the new stand-in's docstring with it.
*Depends on:* Task 1. *Gate:* AC-9, AC-10, AC-11. *Scope:* ~75 LOC.

**Task 5 — Composer: refuse materializing on the unbatched Vmap-over-Scan route.**
*Files:* `src/xtrax/export/composer.py` (modify), `src/xtrax/export/pipeline.py` (modify —
comment only), `tests/export/test_multi_axis.py` (modify).
Per §5.3. In `compose_vmap_of_scan`, on the non-batched route (`composer.py:279`), raise
`MultiAxisCompositionError` when `getattr(inner_boundary, "materialize", False)`, naming the
batched-carry recipe as the workaround. Correct the two now-false comments:
`composer.py:324-325` and `pipeline.py:232-233`.
*Depends on:* Task 4. *Gate:* AC-14. *Scope:* ~35 LOC.

**Task 6 — Export: named residual-callback error.**
*Files:* `src/xtrax/export/pipeline.py` (modify),
`tests/export/test_pipeline_native_wasm32.py` (modify).
Per §1.2 check 7. Wrap the `jax.export.export(...)` call at `pipeline.py:236` in
`except NotImplementedError as exc:`; when `"host_callbacks" in str(exc)`, raise a named xtrax
export error naming the plan's boundary-bearing axes and pointing at `materialize=True`, chained
`from exc`; otherwise re-raise unchanged. No jaxpr scan, no `validate_export_safe` change, no
`xtrax.inference.memo` import — the refusal already exists in JAX and only its message is being
replaced.
*Depends on:* Task 4. *Gate:* AC-12 (both controls). *Scope:* ~25 LOC.

**Task 7 — Export: ordered-inert diagnostic.**
*Files:* `src/xtrax/export/pipeline.py` (modify),
`tests/export/test_pipeline_native_wasm32.py` (modify).
Extend `_diagnostics_for` (`pipeline.py:133-141`) or add a sibling taking `plan.decisions` **and**
the axis boundaries alongside the compile result, per §3.3 — `batch_size` lives on the `SafeMap`
strategy in `decision.strategy` and is not on `CompileResult`. Emit nothing when `ordered=False`
or `batch_size is None`. One entry per target is correct.
*Depends on:* Task 4. *Gate:* AC-15. *Scope:* ~45 LOC.

**Task 8 — Hooks: classify and pin.**
*Files:* `src/xtrax/training/types.py` (modify), `tests/training/test_types.py` (modify),
`tests/engine/test_engine.py` (modify).
Per §4.2 and §4.3. Docstring labels, plus the record-then-assert-outside concreteness test and the
`on_epoch_start` propagation test. No `src/xtrax/engine/` change: `on_step_end`'s swallowing is
deliberate (`io.py:120-121`) and stays.
*Depends on:* nothing. *Gate:* AC-16, AC-17, AC-18. *Scope:* ~65 LOC.

**Task 9 — Docs: the tap-never-crosses claim.**
*Files:* `docs/api/export.md` (modify),
`agent_assets/skills/using-xtrax/references/run.md` (modify).
Per §2.5's public-docs list: `export.md:152-155`, `run.md:243-245`, `run.md:277`.
*Depends on:* Task 2. *Gate:* `grep -n "cannot be dropped on any target" docs/ agent_assets/`
returns nothing, and `SplitTap` appears in both files. *Scope:* ~25 lines of prose.

---

## 9. Gates

Every command carries its `--extra` flags explicitly. `uv run` extras are **not sticky across
subprocess invocations** — an extra activated in one call is not present in the next, so omitting
the flags silently runs against a narrower environment.

```
uv run --extra dev --extra io pytest tests/export/ tests/stages/ -q
uv run --extra dev --extra io pytest tests/engine/ tests/training/ -q
uv run --extra dev --extra io pytest tests/export/ -q --cov=xtrax.export --cov-report=term-missing
just audit-coverage-tier1
uv run --extra dev ruff check . && uv run --extra dev ty check src/
just audit-deterministic
```

The second line is not optional: Task 8's criteria (AC-16, AC-17, AC-18) live in
`tests/training/` and `tests/engine/`, neither of which the first line reaches.

`just audit-deterministic` exiting 0 does not prove the suite is green — grep its log for
`FAILED` as well as reading the exit code.

`just audit-compiler-boundary` is **not** a gate for this spec. Despite the name it has nothing to
do with JAX compilation or export: it enforces that `src/xtrax/` stays free of UI, chain-map, and
plugin-state concerns (`.praxia/loop_priorities.toml:13`). This spec uses "export boundary"
throughout, which is the repo's existing term for the JAX/artifact boundary.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| `SplitTap.observe` on the post-transform value narrows the expressible tap set: a tap observing its input while returning something else cannot be split. | The narrowing applies only to the new export route. Legacy `Tap` is untouched, still works eagerly, still HOST-ONLY under `export_safe=True`; no tap exports today, so no working configuration loses anything. `Tap`'s identity docstring is *not* the warrant — the executor's tests certify non-identity taps (§2.2). Rollback: revert Task 1. |
| **No production consumer.** `grep` finds zero `Tap` implementations in `src/`. Every `Tap` in the repo is a test double, and not one has a side effect: `test_topology.py:87` `_OrderedTap` and `test_boundaries.py:119` `LoggingTap` are identity with no effect; `test_executor.py:120`, `:143`, and `test_multi_axis.py:280` are non-identity transforms; `test_executor.py:99` puts a `HostRecordSink` in the tap slot. So `SplitTap` is designed against test doubles, and the observe-half's ergonomics are unvalidated by any real caller. | Accepted, not mitigated away. The design costs nothing to carry (§2.2: no executor change, no shape change, no new field), and the sink half — which *does* have real implementations — is the load-bearing case. AC-11 exercises the observe half end-to-end so the route is at least executed. |
| The `DedupGather` rejection (§5.1) breaks a downstream plan that exports successfully today. | The current success is a silently wrong artifact — a dropped tap or fuse changes the graph, not just the replay. The message must name the workaround. Rollback: revert Task 3 alone — it touches no other task's code path. |
| Structural `isinstance(tap, SplitTap)` admits any object with attributes named `transform`/`observe` regardless of meaning. | Matches the module's deliberate duck-typed design (`topology.py:12-20`, `topology.py:199`) and the existing `runtime_checkable` `Tap`/`Sink`. A `transform` that is not pure is caught downstream by `jax.export.export` itself if it binds a callback (§1.1); a `transform` that is impure in some other way is not caught, and that is the residue of the duck-typed design. |
| The named residual-callback error (Task 6) swallows a `NotImplementedError` that meant something else. | The handler re-raises unchanged unless `"host_callbacks" in str(exc)`, and chains `from exc` so the original is never lost. It refuses nothing JAX would have accepted — the measurement in §1.1 shows the refusal is already unconditional. |
| The Vmap-over-Scan refusal (Task 5) rejects a two-axis plan that runs eagerly today. | Eagerly it still runs; only `export_pipeline` refuses, and only when the inner boundary declares `materialize`. The artifact it would otherwise produce holds none of the materialized values (§5.3), so the refusal replaces a silently empty result. Rollback: revert Task 5 alone. |
| AC-17 asserts concreteness via `numpy.asarray`, which could pass for a reason unrelated to tracing — or never run at all, since `on_step_end` exceptions are swallowed (`io.py:163-169`). | The `len(recorded) == expected_step_count` assertion runs **outside** the hook, after `fit`, so a hook that never ran or that raised and was logged fails the test. AC-18 is a separate criterion on a separate hook and does not underwrite AC-17. |

---

## 11. Out of scope

- **A bridge between `training.Callback` and `stages.AxisBoundary`.** Decided in §4.1 with
  reasoning; no shared type is introduced.
- **Relaxing Rule 2 (`ordered=True` + `Vmap`).** Decided in §3.2: it fires for eager callers too,
  and relaxing it under `export_safe` alone would admit plans that export but cannot run.
- **Lifting `MaterializeFuseConflictError` for sinks or taps.** Requires the executor to return a
  second value alongside the fused result (`topology.py:44-47`), which is an executor API change.
  The workaround stands: drop the `fuse` and reduce outside the exported function.
- **Multiple materialized axes / promoted output slots.** Unreachable while the composer accepts
  at most one boundary-bearing axis (`composer.py:236-248`, `composer.py:344-372`). §2.4.
- **Changing the ordered-`SafeMap` lowering.** §5.2 — diagnostic only.
- **Rejecting `DedupGather` boundaries outside `export_safe=True`.** §5.1 — the eager path drops
  them identically, but rejecting there is a wider behavior change with no covering test and no
  demonstrated caller. Task 3 leaves a comment at `composer.py:119-126` instead.
- **Making `_run_literal_vmap` materialize correctly.** §5.3 refuses the combination rather than
  threading the per-lane `ys` out of `_lane`, which would change that route's output pytree for
  every caller, not only materializing ones.
- **A jaxpr scan for residual callbacks.** §1.2 check 7 relies on `jax.export.export`'s own
  refusal (§1.1). A scan would duplicate a check JAX already performs, and would have to run
  against the *stripped* boundaries, which `validate_export_safe` does not receive
  (`pipeline.py:222-230` passes the un-stripped mapping, stripping happens after at
  `pipeline.py:234`).
- **Proving a materializing sink's replay is complete.** §1.3 — the precondition is unchanged for
  sinks; only the tap half is fixed structurally.
- **WASM32, Vulkan-SPIR-V, and Metal-SPIR-V targets.** `export_pipeline` already refuses VALIDATED
  targets outright (`pipeline.py:207-217`); this spec adds nothing target-specific.
- **`just audit-compiler-boundary`.** §9 — unrelated subsystem, name collision only.

---

## 12. Open questions

1. **Is `safe_map`'s cardinality-divisibility restriction (`transforms/map.py:33-37`) removable by
   padding?** If yes, the export path could drop the preserved `ordered` flag and recover real
   batching for materialized boundaries, converting §5.2's diagnostic into a fix. *Measurement:*
   pad `xs` to a multiple of `batch_size`, run, slice the result back, and compare against the
   `jax.lax.map(wrapped, xs)` output for a cardinality above the batch size and not divisible by
   it.
2. **Does `on_step_end` actually run out of order today?** `engine.py:186-195` submits each step's
   hook to a `BoundedCallbackHandler(max_concurrent=4)` and joins only at epoch end
   (`engine.py:195`), so on the face of the code step *i*'s hook need not complete before step
   *i+1*'s is submitted. Against that: `Callback`'s hooks are synchronous
   (`training/types.py:32-38`) and `fire_step_end` (`engine.py:189-190`) never suspends, so each
   submitted task runs to completion without yielding and the observable order is FIFO. Neither
   reading is a measurement. *Measurement:* a callback recording completion order with staggered
   sleeps across a multi-step epoch — which requires an **awaiting** hook body, since a
   synchronous `time.sleep` blocks the loop and cannot interleave. If interleaving is confirmed,
   §4.2's "replay-shaped" label for `on_step_end` should gain an explicit "unordered across steps"
   note, and the `Callback` docstring should say so.
3. **Does an `AxisBoundary` pickled before the `materialize` field load correctly?** Inherited
   untested caveat (`boundaries.py:124-128`). Unchanged by this spec, which adds no field (§6.3),
   but it remains open and would become urgent for any future field addition. *Measurement:*
   pickle a three-field `AxisBoundary` under an older revision and unpickle it under HEAD.

---

## References

- `src/xtrax/stages/boundaries.py` — `Fuse`/`Tap`/`Sink`/`AxisBoundary`; `materialize` docstring
  at 99-129.
- `src/xtrax/stages/executor.py` — application points 116-133, 143-217, 220-248; ordering
  mechanics in the module docstring 22-83.
- `src/xtrax/stages/topology.py` — `_check_export_boundary` 178-239, `validate_plan_topology`
  241-330, `_EXPORTABLE_STRATEGIES` 175.
- `src/xtrax/export/pipeline.py` — `_StrippedSink` 62-84, `_boundaries_for_export` 86-117,
  diagnostics 133-141, the safety-gate call 222-230, stripping 234, the `jax.export.export` call
  236, per-target diagnostics 259.
- `src/xtrax/export/composer.py` — strategy routing 59-134, `DedupGather` branch 119-126,
  outer-boundary refusal 236-248, `_init_is_batched` route selection 262, `_run_batched` 273-277,
  `_run_literal_vmap` 279-301, `build_traceable_callable` docstring 324-325.
- `src/xtrax/export/safety.py` — `validate_export_safe` 248-288.
- `src/xtrax/training/types.py` — `Callback` 24-38, synchronous hook signatures 32-38;
  `src/xtrax/engine/engine.py` — hook firing 152-215, `on_step_end` submission 186-192,
  `wait_all` 195, the `finally` 212-219; `src/xtrax/engine/io.py` — `BoundedCallbackHandler`
  116-174, the swallowing wrapper 163-169 and its stated rationale 120-121;
  `src/xtrax/telemetry/callback.py` — the one live implementation.
- `src/xtrax/stages/_callback.py` — JAX version and `io_callback` signature pins, 110-113.
- `src/xtrax/transforms/map.py:28-30` — the `n <= batch_size` vmap short-circuit;
  `:33-37` — the divisibility restriction that applies only above it.
- `src/xtrax/inference/ir_schema.py:212-220`, `:270` — the hand-written boundary schema, unaffected
  (§2.5).
- `jax/_src/export/_export.py:1047` — the `host_callbacks` serialization refusal (jax 0.11.1).
- `docs/api/export.md:150-155`, `agent_assets/skills/using-xtrax/references/run.md:243-245`,
  `:277` — the public statements of the tap-never-crosses claim.
- `tests/stages/test_topology.py:283-368` (Rule 3 pinning, including
  `test_rejects_materialize_without_sink` 321-325 and the boundary-free `DedupGather` case
  379-380), `tests/stages/test_executor.py:119-126`, `:142-149` (non-identity taps certified),
  `tests/stages/test_nested_ordering.py:59-177`, `tests/export/test_composer.py:66-150` (including
  the boundary-free `DedupGather` case at 123) — the pinning tests.
- Backlog **#5089** — the `_run_literal_vmap` materialization defect (§5.3), filed independently.
- `.praxia/docs/specs/260702_design-2174-next-slices-minimal-composit.md` — Fork-10, the
  resolution that put `tap`/`sink` inside the per-axis iterator body.
