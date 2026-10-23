---
title: 'Intent resolution: one pipeline, many runtimes'
description: Design for xtrax.loop/xtrax.map intent functions resolved per runtime context by xtrax.compile (two-pass, custom_jvp), with typed autobundling and the kernel seam as follow-ups
status: draft
task_id: 261023_equinox-decouple-pallas
date: '261023'
---
# Intent resolution: one pipeline, many runtimes

Brainstorm record: contemplex session `b9551d70`, artifact
`.praxia/docs/specs/261023_xtrax-as-a-pure-functional-pipeline-laye.md`.
Spike scripts: `scripts/experiments/intent_resolution_spike_261023/` (exploratory, not pre-registered;
feasibility only, jax 0.11.1 / equinox 0.13.8, CPU).

## 1. Intent

The user's words (261023):

> "what would be ideal would be a simple way to compose pipelines and specify their runtimes via one
> interface, but have multiple dispatch (e.g. while loop when autodiff not needed, vmap vs chunkedmap)"
>
> "my original vision was pure functional programming, where types get autobundled and resolved"

and they asked how AST/CST/jaxpr analysis can help. Earlier decisions in the same session: xtrax's
primitive layer is plain JAX (no Equinox required), and xtrax owns the reference/kernel seam while
kernels live with the models or in a sibling package.

The why, as evidenced in this repo: the right implementation of the same computation depends on
context JAX cannot see, and getting it wrong is expensive. A 512-step autoregressive decode compiled
as `lax.scan` took over 20 minutes of backend codegen on an L40S; as one `while_loop` it did not. A
`ChunkedMap` (a `lax.map`, i.e. a scan) around an inner `while_loop` hung compilation for an hour on
Blackwell. Pallas kernels cannot export and need hand-written VJPs. Today users pick these by hand,
per call site, and xtrax's planner only knows axis sizes and a memory budget.

## 2. Goals, non-goals, anti-goals

**Goals**
- Users write ordinary JAX: pure functions over pytrees, with `xtrax.loop` / `xtrax.map` where they
  mean "iterate" or "map", in the same call shapes as `lax.scan`/`fori_loop`/`while_loop` and
  `jax.vmap`/`lax.map`.
- One call, `xtrax.compile(fn, ctx)`, resolves every intent site to an implementation from the
  shapes and an explicit context, and returns an inspectable Plan.
- Differentiation is detected, never declared: a loop runs as one `while_loop` unless JAX
  differentiates it, then as a `scan`.
- Later phases: typed autobundling of stages; the kernel seam (`xtrax.op`) as one more intent.

**Non-goals** (orchestrator's call): a new IR or tracer replacing jaxpr; multi-host placement;
automatic kernel generation.

**Anti-goals**: resolution that changes results (every resolution must compute the same function as
the reference resolution, up to float reassociation); a DSL to learn instead of JAX; inferred
properties nobody checks (anything inferred is verified on the jaxpr or declared).

## 3. Constraints

- Public JAX API only for the intent functions (the private-primitive route already lost two names
  between minor versions during the spike).
- Unsatisfiable contexts fail at resolve time, before the execution trace, naming the site and the
  constraint (e.g. "site `decoder` (model.py:88): export target onnx forbids WhileCarry").
- The context is an explicit, frozen, hashable value. No ambient configuration decides anything.
- Reuse what exists: strategies (`tiling/strategy.py:59-126`), `BatchPlanner` (`tiling/plan.py:287,337`),
  `make_axis_dispatch` (`tiling/dispatch.py:31`), the stage executor (`stages/executor.py:195,281`),
  the export composer (`export/composer.py:60,129-135`) and the jaxpr toolkit
  (`profiling/jaxpr.py:52`, `inference/cse.py:488`, `export/safety.py:383,582`).
- One source of truth for batch sizes: the resolver calls `BatchPlanner`; it never re-derives sizes.
- Consumers (aminx, plegadx, prolix, demistify, denxity) keep working; nothing existing is removed in
  the MVP.

## 4. Decided design

### 4.1 Intent functions

**Module (orchestrator's call):** everything lives in a new module `xtrax.intent` —
`xtrax.intent.loop`, `xtrax.intent.map`, `xtrax.intent.compile`, `Context`, `Plan`. `xtrax.loop` is
already a subpackage (`src/xtrax/loop/`), so the top level does not re-export `loop`; below,
`xtrax.loop`/`xtrax.map`/`xtrax.compile` mean the `xtrax.intent` names.

| Intent | Call shape | Unresolved behaviour (outside `xtrax.compile`) |
|---|---|---|
| `xtrax.loop(body, init, *, length=None, xs=None, cond=None, name=None)` | `lax.scan` shape: `body(carry, x) -> (carry, y)`, returns `(carry, ys)`; `fori_loop`-like when only `length`; `while_loop`-like with `cond` (carry-only, `body(carry) -> carry`) | plain `lax.scan` / `while_loop`, identical results |
| `xtrax.map(f, xs, *, name=None)` | like `jax.vmap(f)(xs)` / `lax.map(f, xs, batch_size=…)` | plain `jax.vmap`, identical results |

Collected outputs (`ys`) on the while primal: the primal preallocates `ys` from `eval_shape` of one
step and writes with `lax.dynamic_update_index_in_dim`. Verified (spike `spike_loop_ys.py`): forward
jaxpr while/no scan and outputs equal `lax.scan`; `grad` jaxpr scan/no while and grads match; `jvp`
and `vmap(grad)` match; export round-trips with no `custom_call`. This is the decode shape
(per-step token outputs).

`xtrax.loop` with a static length is a `jax.custom_jvp` whose primal rule is a `while_loop` and whose
JVP rule is the JVP of the `scan` form. Verified (spike `spike_loop_customjvp.py`): undifferentiated
jaxpr has `while` and no `scan`; under `grad` the jaxpr has `scan` and no `while`, grads match a scan
reference; `jvp` matches; `jacrev(jacrev)` and `jacfwd(jacrev)` match; `vmap(grad)` matches;
`grad(checkpoint)` matches; `jax.export` round-trips with no `custom_call`. A loop with a
data-dependent `cond` is a plain `while_loop` and is not differentiable; that is JAX's own error and
stays loud.

Closures: a custom rule that closes over a differentiated value raises `UnexpectedTracerError`
(loud). `xtrax.loop` applies `jax.closure_convert` to the body internally, so users write normal
closures; verified correct gradients (spike `spike_loop_closure.py`).

Rejected (decision log): private `jax.extend.core.Primitive` rules (internals churn, no extra
capability); `jax.custom_vjp` (breaks forward-mode, which aminx uses in `aminx/host/runner.py` and
`aminx/utils/forward_jac.py`).

### 4.2 Context

```python
@dataclass(frozen=True)
class Context:
    memory_budget: MemoryBudget | None = None      # reuse tiling.budget.MemoryBudget
    target: Literal["jit", "export"] = "jit"
    export_format: str | None = None               # e.g. "stablehlo", "onnx"; required when target="export"
    backend: str | None = None                     # default: jax.default_backend()
    kernels: bool = False                          # follow-up (xtrax.op); inert in the MVP
```

No `differentiable` field: §4.1 makes it unnecessary. Field names are the builder's call; the set
of facts is not.

### 4.3 `xtrax.compile(fn, ctx)` and the Plan

Two passes, verified in `spike_two_pass.py`:

1. **Plan pass** (abstract, `jax.make_jaxpr`): each intent call records a site — key, input avals,
   parent site, path — and runs its default implementation.
2. **Resolve**: sites become `AxisSpec`s for `BatchPlanner` (joint-budget mode when a budget is
   given); context rules and hazard rules (§4.4) run over the site tree; the result is a Plan: per
   site, an existing strategy object for the **primal path** and one for the **differentiated path**,
   each with a one-line reason.
3. **Execution trace** (`jax.jit`): each intent call looks up its decision by key and checks its
   avals equal the plan's; the full site sequence must match; any mismatch raises before execution,
   naming the site.

Spike results: a Python loop calling one source line at three shapes got three distinct stable keys;
joint decisions (`[None, 32, 32]`) took effect (scan only at chunked sites); outputs equal plain
vmap; a trace-time side effect that changed a shape was caught before execution with the site named;
three calls produced one plan and one jit cache entry.

**Site keys are hierarchical and path-qualified**: `(parent key, path, user name or source location,
ordinal)` with `path ∈ {primal, diff, branch:i}`. Predicted (not yet spiked): under `grad`, the
custom_jvp rule re-traces a loop body through the scan form, so a site nested in a loop body is
visited on the `diff` path; flat keys like the spike's would report a false mismatch. The plan pass
therefore also traces the differentiated path when the function is called under differentiation
(builder's call how: e.g. `jax.linearize` of the abstract function), and validation is per path.

**Plan value**: frozen, hashable, versioned schema, serialisable (JSON), printable as a table
(site, path, strategy, reason), diffable across contexts, recorded in the run ledger, and
overridable by site name (`xtrax.compile(fn, ctx, overrides={"decoder": Scan()})`).

**Memory under differentiation**: because the two paths are separate rules, each gets its own
budgeted decision; the differentiated decision accounts for scan residuals (optionally with a
checkpoint policy). No tracer-type inspection.

**When the differentiated path is planned (orchestrator's call):** eagerly, best-effort. The plan
pass traces `fn` and, separately, `jax.linearize` of `fn` with respect to its inexact (floating)
leaves only; integer/bool leaves are held fixed. If linearization fails (e.g. a data-dependent
`cond` loop), every site on that path gets "no differentiated decision" with the JAX error as its
reason, and a later `grad` fails with JAX's own error. One Plan holds both paths, so the plan cache
is keyed by (function, avals, ctx), not by transform stack; `jax.jit`'s own cache separates the
transformed traces.

**Budget estimator (orchestrator's call):** `MemoryBudget.estimate` is a caller-supplied function
over the decision sequence (`tiling/budget.py:36-58`). When the user gives only a byte count,
`xtrax.compile` builds the estimator: for each site and candidate strategy it calls
`lowered_memory_estimate` (`tiling/estimators.py:72`) on that site's function at its recorded avals
(for the differentiated decision, on `jax.vjp` of it, which includes residuals), and sums over sites
that are live at the same time — siblings add, a nested site adds to its parent's per-iteration
footprint. Sites map to `AxisSpec`s in plan-pass visit order (pre-order over the site tree); that is
the order `BatchPlanner`'s joint-budget mode demotes in. A user-supplied estimator replaces this and
receives decisions with site names.

### 4.4 Rules the resolver applies

Context rules (lifted from where they live today, one shared table):
- `target="export"` forbids `Bucket` and `WhileCarry` (today refused later at `export/composer.py:129-135`).
- A data-dependent `cond` loop has no `diff` decision.

Hazard rules over the site tree (start with the two documented incidents; grow incrementally):
- a scan-family strategy (`Scan`, `ChunkedMap`) whose subtree contains a `while_loop` on a backend
  flagged for the scan-around-while hang;
- a static scan above a length threshold on the primal path (codegen blow-up) → primal decision is
  the while form.

One post-hoc check walks the execution jaxpr (`profiling/jaxpr.py:52`) and asserts no rule is
violated after resolution; it does not iterate.

**What each intent can resolve to (orchestrator's call).** `map` → `Vmap` or `ChunkedMap`. `loop` →
primal: static-length while (a new strategy, `StaticWhile`, carrying `ys`; existing `WhileCarry` is
carry-only and unbounded) or `Scan`; differentiated: `Scan`. `Bucket` (host-side) and `DedupGather`
(dispatched outside `make_axis_dispatch`, `tiling/dispatch.py:76-78`) are not emitted by intents in
the MVP.

**Export (orchestrator's call).** The composer refuses `WhileCarry` because its trip count is
unbounded (`export/composer.py:47,131`). A static-length loop is bounded, and StableHLO export of the
while primal already round-trips (spikes). So: `target="export"` + StableHLO allows `StaticWhile`;
formats that cannot express a while (to be confirmed per format, e.g. the ONNX route) resolve the
primal to `Scan`; a `cond` (unbounded) loop under export is refused at resolve time. The composer
learns `StaticWhile` in the same change.

**Precedence (orchestrator's call).** Context rules are hard: a decision or user override that
violates one is a resolve-time error. User overrides beat hazard rules; the Plan records the
override and the hazard it overrode as a warning. A hazard with no satisfying alternative under the
context is an error for the configurations documented as hangs or multi-minute codegen, and a
warning otherwise.

### 4.5 AST/CST and jaxpr: what each is for

- **jaxpr**: site discovery (plan pass), hazard verification, and (phase 2) verifying declared axis
  properties with `verify_axis_invariance` (`inference/invariance.py:66`, currently uncalled).
- **AST/CST**: never at runtime. Authoring-time libcst codemods from `lax.map`/`lax.scan`/
  `fori_loop`/`while_loop` to intent functions, plus lint for Python loops over axes. Source locations
  for site keys and error messages come from the call frame (`inspect`).

## 5. MVP acceptance criteria

1. **Given** `xtrax.loop(body, init, length=n)` inside `xtrax.compile(fn, ctx)`, **when** `fn` runs
   undifferentiated, **then** the execution jaxpr contains `while` and no `scan`, and the output equals
   the `lax.scan` reference.
2. **Given** the same function, **when** it is differentiated with `grad`, `jvp`, `jacfwd(jacrev)` and
   `vmap(grad)`, **then** every result equals the same transform of the `lax.scan` reference, and the
   gradient jaxpr contains `scan` and no `while`.
3. **Given** a loop body that closes over the parameters being differentiated, **when** `grad` is
   taken, **then** gradients equal the reference (closure conversion is internal).
4. **Given** `xtrax.map` sites whose combined estimate exceeds `ctx.memory_budget`, **when** compiled,
   **then** the Plan's batch sizes equal what `BatchPlanner` joint-budget mode returns for the same
   axes, and outputs equal plain `jax.vmap`.
5. **Given** intent sites nested two deep (loop-in-map and map-in-loop) and inside both `lax.cond`
   branches, **when** compiled and run with and without `grad` and under `jax.checkpoint`, **then** no
   site-validation error is raised.
6. **Given** a function whose second trace differs from the plan pass, or a context that forbids every
   strategy for a site, **when** compiled, **then** it raises before execution with the site's name or
   source location, its path, and the violated constraint.
7. **Given** a compiled function called repeatedly with the same shapes, **then** it plans once and
   records zero recompiles at steady state (`jax_log_compiles`); the Plan prints as a table and
   round-trips through JSON.
8. **Given** one real consumer workload, **when** ported to `xtrax.loop`/`xtrax.map`, **then** a
   pre-registered bathos run reports the outcome stated in its sidecar. Workload: aminx's
   autoregressive decode (`aminx/inference/decode/autoregressive.py`,
   `aminx/inference/sample_autoregressive.py`) — compile time and step time, scan vs resolved, on a
   GPU — plus peak memory under `grad` within the differentiated decision's budget.

## 6. Follow-ups (outside the MVP)

- **Kernel seam, `xtrax.op(reference, kernels=…)`**: `custom_jvp` whose primal is the planned kernel or
  the reference and whose JVP is the reference's; export always resolves to the reference. Spike
  evidence: `research/261023_pallas-block-sparse-spike.md`.
- **Typed autobundling**: stages are pure functions returning typed records; parameters are filled
  from upstream fields by name and checked by type (any ambiguity is an error); axis markers via
  `Annotated[...]` read with `get_type_hints(include_extras=True)` and verified on the jaxpr.
- **Codemods and lint** (libcst / ast-grep, following `codemods/safemap-to-chunkedmap/`).
- **On-disk Plan cache** keyed by function identity, avals and context, if plan-pass cost matters.
- **Equinox decoupling**: rescoped to the declaration objects (the plain-JAX pytree layer of
  `specs/261023_equinox-decoupling.md`); the training path is not a prerequisite.

## 7. Left to the builder

The Plan JSON schema, the hazard thresholds (backend list for the scan-around-while hang, scan
length for codegen), the error message wording, the `StaticWhile` field names, and per-format
confirmation of which export formats express a while.

## 8. Risks carried from the pre-mortem

| Risk | Mitigation in this design |
|---|---|
| Path-dependent site sequences produce false mismatches | hierarchical, path-qualified keys; criterion 5 |
| No adoption | drop-in call shapes; unresolved = plain JAX; criterion 8 measured on a real workload; codemods follow |
| Two sources of truth for batch sizes / export refusals | resolver calls BatchPlanner; export refusals become resolver rules in one table |
| Plan pass doubles trace time on large models | measure on the largest consumer model; plan cache follow-up |
| Opaque errors from inside custom rules | every error names site, path and constraint; criterion 6 |
