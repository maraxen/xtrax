---
title: 'Intent resolution MVP: implementation plan'
description: Task plan for xtrax.intent (loop, map, Context, Plan, compile) covering design criteria 1-7
status: draft
task_id: 261023_equinox-decouple-pallas
date: '261023'
---
# Intent Resolution MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `xtrax.intent` — `loop`, `map`, `Context`, `Plan`, `compile` — so one function resolves its iteration and mapping sites per runtime context, running loops as one `while_loop` unless JAX differentiates them.

**Architecture:** `loop` and `map` are `jax.custom_jvp` wrappers (primal rule = primal decision, JVP rule = JVP of the differentiated decision) that read per-site decisions from a site session. `compile(fn, ctx)` records sites in an abstract plan pass (plus a best-effort `jax.linearize` pass), resolves them with `BatchPlanner` and rule tables into a `Plan`, then runs `jax.jit` with the Plan applied and validates every site.

**Tech Stack:** Python 3.13, JAX (`>=0.10.2,<0.12`, tested on 0.11.1), existing `xtrax.tiling` (strategies, `BatchPlanner`, `MemoryBudget`, `lowered_memory_estimate`), pytest, ruff, ty, jaxlint.

**Spec:** `.praxia/docs/superpowers/specs/261023_intent-resolution-design.md` (approved 261023). Spikes it cites: `scripts/experiments/intent_resolution_spike_261023/` (`spike_loop_customjvp.py`, `spike_loop_ys.py`, `spike_loop_closure.py`, `spike_two_pass.py`) — read them before Tasks 4, 5 and 7; they are working prototypes of the mechanisms.

## Global Constraints

- Public JAX API only in `src/xtrax/intent/`: no `jax.interpreters.*`, `jax._src.*`, or `jax.extend.core.Primitive` (spec §3).
- `xtrax.intent` must not import `equinox` (plain-JAX layer, spec §1).
- Every resolution computes the same function as the reference resolution: outputs within `atol=1e-5` (float32) of `lax.scan` / `jax.vmap` references (spec §2 anti-goals).
- Unsatisfiable contexts and site mismatches raise before execution, naming site (user name or `file:line`), path, and constraint (spec §3).
- `Context` is a frozen, hashable dataclass; nothing reads ambient configuration (spec §3).
- Batch sizes come only from `BatchPlanner`; the resolver never computes them itself (spec §3).
- Module is `xtrax.intent`; do not add `loop` to `xtrax/__init__.py` (`xtrax.loop` is an existing subpackage). Mark the module experimental in its docstring and CHANGELOG.
- Local test runs are narrow (one file or `-k`); never the whole suite (`~/.claude/rules/local-compute-limits.md`). Prefix runs with `OMP_NUM_THREADS=4`.
- Before each commit: `uv run ruff check --no-fix <files>`, `uv run ruff format --check <files>`, `jaxlint check src/xtrax/intent`.
- Tests live in `tests/intent/` (tier-1 coverage counts them; `tests/audit/` would not).

## Review Focus

1. Pytree carries and outputs (tuple carry, dict `ys`, nested) through `loop` — a user expects any pytree `lax.scan` accepts. Pinned in Task 4 (`test_loop_pytree_carry_and_ys`).
2. Integer/bool leaves in a carry (step counters) under `grad` — expect correct float gradients and no float0 crash. Pinned in Task 4 (`test_loop_grad_with_int_counter`).
3. Degenerate extents: `length=0`, `length=1`, and `map` over a leading axis of size 1 (xtrax's own #2520 length-1 vmap miscompile). Pinned in Task 4 (`test_loop_degenerate_lengths`) and Task 5 (`test_map_leading_axis_one`).
4. Closures over Python scalars, non-differentiated arrays, and differentiated params together. Pinned in Task 4 (`test_loop_mixed_closures`).
5. The compiled function called with a new shape, and under an outer `jax.vmap` — expect a fresh plan per shape signature and correct batched results. Pinned in Task 7 (`test_compile_new_shape_replans`, `test_compile_under_outer_vmap`).

**Not in this plan** (spec items outside criteria 1–7): Plan recording in the run ledger; export-composer support for `StaticWhile`; `xtrax.op` kernel seam; typed autobundling; codemods/lint; on-disk Plan cache; criterion 8 (aminx decode run, gates release per the user's decision).

---

### Task 1: `StaticWhile`, `Context`, errors

**Files:**
- Modify: `src/xtrax/tiling/strategy.py` (add `StaticWhile` after `WhileCarry`)
- Modify: `src/xtrax/tiling/__init__.py` (export `StaticWhile`)
- Create: `src/xtrax/intent/__init__.py`, `src/xtrax/intent/context.py`, `src/xtrax/intent/errors.py`
- Test: `tests/intent/__init__.py`, `tests/intent/test_context.py`

**Interfaces:**
- Produces:
  - `StaticWhile` — `@dataclass(frozen=True) class StaticWhile: length: int` (static-length `while_loop` that may collect `ys`; bounded trip count, unlike `WhileCarry`).
  - `Context` — frozen dataclass, fields: `memory_budget: MemoryBudget | int | None = None` (an `int` is a byte count: the resolver builds the estimator, Task 6), `target: Literal["jit", "export"] = "jit"`, `export_format: str | None = None`, `backend: str | None = None`, `chunk_size: int = 32`, `kernels: bool = False`. `__post_init__` raises `ValueError` if `target == "export"` and `export_format is None`, or `chunk_size < 1`. `resolved_backend() -> str` returns `backend or jax.default_backend()`.
  - `IntentError(Exception)`; `SiteMismatchError(IntentError)`; `UnsatisfiableContextError(IntentError)`. Both subclasses take `(message: str, *, site: str, path: str, constraint: str)` and expose those as attributes; `str(err)` contains all three.
  - `xtrax/intent/__init__.py` re-exports `Context`, the three errors; later tasks add `loop`, `map`, `Plan`, `SiteDecision`, `compile`. Module docstring starts "Experimental."

- [ ] **Step 1: Write the failing tests**

```python
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
    e = UnsatisfiableContextError("no strategy", site="dec", path="primal", constraint="export forbids X")
    assert (e.site, e.path, e.constraint) == ("dec", "primal", "export forbids X")
    assert "dec" in str(e) and "primal" in str(e) and "export forbids X" in str(e)

def test_static_while_is_frozen():
    assert StaticWhile(length=4) == StaticWhile(length=4)
```

- [ ] **Step 2: Run to verify failure** — `OMP_NUM_THREADS=4 uv run pytest tests/intent/test_context.py -q` → FAIL (`ModuleNotFoundError: xtrax.intent`).
- [ ] **Step 3: Implement** the interfaces above.
- [ ] **Step 4: Run to verify pass** — same command → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(intent): Context, intent errors, StaticWhile strategy"`
---

### Task 2: `Plan` and `SiteDecision`

**Files:**
- Create: `src/xtrax/intent/plan.py`
- Test: `tests/intent/test_plan.py`

**Interfaces:**
- Produces:
  - `SiteKey` — frozen dataclass `(parent: SiteKey | None, path: Literal["primal", "diff"], label: str, ordinal: int)`; `label` is the user `name` or `"file.py:LINE"` of the intent call; `__str__` renders `parent/path/label#ordinal`. Lives in `plan.py` so decisions and sessions share it.
  - `SiteDecision` — frozen dataclass `(key: SiteKey, kind: str, primal: AxisStrategy | StaticWhile, diff: AxisStrategy | None, primal_reason: str, diff_reason: str, warnings: tuple[str, ...] = ())`. `diff is None` means "no differentiated decision" and `diff_reason` holds why.
  - `Plan` — frozen dataclass `(schema_version: int, context: Context, decisions: tuple[SiteDecision, ...])`; `PLAN_SCHEMA_VERSION = 1`; `by_key() -> Mapping[SiteKey, SiteDecision]`; `table() -> str` (one row per site: key, kind, primal, diff, reasons, warnings); `to_json() -> str`; `Plan.from_json(s, *, context: Context) -> Plan` (context is not serialised beyond `repr`; strategies serialise as `{"type": "ChunkedMap", "batch_size": 32}` etc. for `Vmap`, `ChunkedMap`, `Scan` (no callable fields), `StaticWhile`).
  - Re-export `Plan`, `SiteDecision` from `xtrax.intent`.

- [ ] **Step 1: Write the failing tests**

```python
def test_plan_json_round_trip():                    # criterion 7 (serialisation)
    p = Plan(PLAN_SCHEMA_VERSION, Context(), (dec_map, dec_loop))
    assert Plan.from_json(p.to_json(), context=Context()) == p

def test_plan_table_lists_every_site_and_reason():
    t = p.table()
    assert str(dec_map.key) in t and dec_map.primal_reason in t and "ChunkedMap" in t

def test_plan_is_hashable():
    assert hash(p) == hash(Plan.from_json(p.to_json(), context=Context()))

def test_unknown_schema_version_rejected():
    with pytest.raises(ValueError, match="schema_version"):
        Plan.from_json(p.to_json().replace('"schema_version": 1', '"schema_version": 99'), context=Context())
```

- [ ] **Step 2: Run to verify failure** — `OMP_NUM_THREADS=4 uv run pytest tests/intent/test_plan.py -q` → FAIL.
- [ ] **Step 3: Implement** `plan.py`.
- [ ] **Step 4: Run to verify pass** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(intent): Plan and SiteDecision with JSON round-trip"`
---

### Task 3: Site session

**Files:**
- Create: `src/xtrax/intent/sites.py`
- Test: `tests/intent/test_sites.py`

**Interfaces:**
- Produces:
  - Uses `SiteKey`, `SiteDecision` from Task 2.
  - `SiteRecord` — frozen dataclass `(key: SiteKey, kind: Literal["loop", "map"], avals: tuple[jax.ShapeDtypeStruct, ...], length: int | None, fn: Callable | None)` (`fn` = the closure-converted site function, used by the default estimator in Task 6; excluded from `__eq__`/`__hash__`).
  - `Session` — `mode: Literal["plan", "exec"]`; `records: list[SiteRecord]`; `decisions: Mapping[SiteKey, SiteDecision] | None` (from Task 2); methods `enter_site(kind, label, avals, length, fn) -> SiteKey` (computes the key from the current frame stack and per-(parent, path, label) ordinal counters, records in plan mode, validates in exec mode), `decision_for(key) -> SiteDecision | None`, `push(key, path)` / `pop()` context manager `scope(key, path)`.
  - Module functions: `current_session() -> Session | None` (a `contextvars.ContextVar`), `activate(session) -> contextmanager`, `call_label(name: str | None, stacklevel: int) -> str`.
- Rule the implementer cannot infer: an intent captures `current_session()` and its own `SiteKey` when it is traced, and its JVP rule re-enters that session with `activate(session)` and `session.scope(key, "diff")` before tracing the differentiated form. Under `jax.jit`, JAX calls the JVP rule lazily, after `compile`'s `activate` block has exited; without the capture, nested sites would silently see no session.
- In `exec` mode, `enter_site` raises `SiteMismatchError(site=str(key), path=key.path, constraint=...)` when the key was not planned or its avals differ from the planned avals (constraint text: `"planned <avals>, traced <avals>"` or `"site not in plan"`).

- [ ] **Step 1: Write the failing tests** (drive `Session` directly with fake sites; no JAX transforms needed)

```python
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
```

(`sds = jax.ShapeDtypeStruct` with `jnp.float32`.)

- [ ] **Step 2: Run to verify failure** — `OMP_NUM_THREADS=4 uv run pytest tests/intent/test_sites.py -q` → FAIL (import error).
- [ ] **Step 3: Implement** `sites.py` per the interfaces.
- [ ] **Step 4: Run to verify pass** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(intent): site keys and session with path-qualified validation"`
---

### Task 4: `loop`

**Files:**
- Create: `src/xtrax/intent/loop.py`
- Test: `tests/intent/test_loop.py`

**Interfaces:**
- Consumes: `current_session`, `activate`, `Session.enter_site`, `Session.decision_for`, `Session.scope`, `call_label` (Task 3); `StaticWhile` (Task 1); `Scan` (`xtrax.tiling.strategy`).
- Produces: `loop(body, init, *, length: int | None = None, xs=None, cond=None, name: str | None = None)`.
  - Scan form (`xs` and/or `length`): `body(carry, x) -> (carry, y)`; returns `(carry, ys)` exactly like `lax.scan`.
  - While form (`cond` given, no `length`/`xs`): `body(carry) -> carry`, returns `carry`; this is a plain `lax.while_loop` (not differentiable; JAX's own error stands). It still registers a site of kind `"loop"` with `length=None`.
  - Static forms are a `jax.custom_jvp` with `nondiff_argnums` for the closure-converted step and length: primal executes the site's primal decision (`StaticWhile` → `lax.while_loop` writing `ys` with `lax.dynamic_update_index_in_dim` into buffers preallocated from `jax.eval_shape` of one step; `Scan` → `lax.scan`); the JVP rule re-enters the captured session in scope `(key, "diff")` and returns `jax.jvp` of the `lax.scan` form. With no session or no decision, primal = `StaticWhile`, diff = `Scan`.
  - The body is passed through `jax.closure_convert(body, init_or_(init, x0))` internally; consts become explicit primal args so closures over differentiated values get gradients.
  - `xs` is threaded as a primal arg (indexed with `lax.dynamic_index_in_dim` in the while form).

- [ ] **Step 1: Write the failing tests** (references are plain `lax.scan`; `ok = lambda a, b: jnp.allclose(a, b, atol=1e-5)` over tree leaves)

```python
def test_loop_forward_is_while_not_scan():          # criterion 1
    jx = str(jax.make_jaxpr(f_loop)(x))
    assert "while[" in jx and "scan[" not in jx
    assert ok(jax.jit(f_loop)(x), f_scan_ref(x))

def test_loop_transforms_match_scan_reference():   # criterion 2
    assert ok(jax.grad(f_loop)(x), jax.grad(f_scan_ref)(x))
    assert ok(jax.jvp(f_loop, (x,), (t,))[1], jax.jvp(f_scan_ref, (x,), (t,))[1])
    assert ok(jax.jacfwd(jax.jacrev(f_loop))(x), jax.hessian(f_scan_ref)(x))
    assert ok(jax.vmap(jax.grad(f_loop))(xb), jax.vmap(jax.grad(f_scan_ref))(xb))
    gj = str(jax.make_jaxpr(jax.grad(f_loop))(x))
    assert "scan[" in gj and "while[" not in gj

def test_loop_closure_over_params_grad():          # criterion 3
    # body closes over w; grad wrt w equals lax.scan reference

def test_loop_collects_ys_like_scan():             # ys shape/dtype/values equal lax.scan, with xs and with length only
def test_loop_pytree_carry_and_ys():               # Review Focus 1: tuple carry, dict ys
def test_loop_grad_with_int_counter():             # Review Focus 2: carry (jnp.int32 step, float state); grad wrt float input matches reference
def test_loop_degenerate_lengths():                # Review Focus 3: length=0 returns init and empty ys; length=1 matches reference, both with and without grad
def test_loop_mixed_closures():                    # Review Focus 4: Python float + stop_gradient array + differentiated param
def test_loop_while_form_is_plain_while_loop():    # cond form returns carry equal to lax.while_loop
def test_loop_outside_compile_exports():           # jax.export(jax.jit(f_loop), platforms=["cpu"]) round-trips; "custom_call" not in mlir
```

Use body `lambda c, _: (jnp.tanh(c * w + 0.1), jnp.sum(c))`, `length=32`, `x = jnp.linspace(-1, 1, 5)` (the `spike_loop_ys.py` decode shape).

- [ ] **Step 2: Run to verify failure** — `OMP_NUM_THREADS=4 uv run pytest tests/intent/test_loop.py -q` → FAIL (import error).
- [ ] **Step 3: Implement `loop`** in `src/xtrax/intent/loop.py` per the interfaces; port the mechanism from `spike_loop_ys.py` and `spike_loop_closure.py`.
- [ ] **Step 4: Run to verify pass** → PASS (all 10).
- [ ] **Step 5: Commit** — `git commit -m "feat(intent): loop -- while_loop unless differentiated, scan under AD"`
---

### Task 5: `map`

**Files:**
- Create: `src/xtrax/intent/map.py`
- Test: `tests/intent/test_map.py`

**Interfaces:**
- Consumes: Task 3 session API; `Vmap`, `ChunkedMap` (`xtrax.tiling.strategy`); `chunked_map` (`xtrax.transforms.map`, which already guards length-1 axes and ragged remainders).
- Produces: `map(f, xs, *, name: str | None = None)` — maps `f` over the leading axis of every leaf of `xs`; `jax.custom_jvp` over the closure-converted `f`: primal executes the primal decision (`Vmap` → `jax.vmap`; `ChunkedMap(b)` → `chunked_map(f, xs, batch_size=b)`), JVP rule re-enters the captured session in scope `(key, "diff")` and returns `jax.jvp` of the diff decision's form. No session/decision → `Vmap` on both paths. Site kind `"map"`, `length=None`, avals = leaves of `xs`.

- [ ] **Step 1: Write the failing tests**

```python
def test_map_unresolved_equals_vmap():
    assert ok(map(layer, xs), jax.vmap(layer)(xs))

def test_map_chunked_decision_uses_lax_map():
    # inject a Session(mode="exec") whose decision for the site is SiteDecision(primal=ChunkedMap(32), diff=ChunkedMap(32), ...)
    # jaxpr of the call contains "scan["; result equals jax.vmap(layer)(xs) for xs of leading dim 100 (ragged remainder)

def test_map_grad_and_jvp_match_vmap():
def test_map_leading_axis_one():                    # Review Focus 3: leading dim 1 and 0, with and without grad
def test_map_closure_over_params_grad():
```

- [ ] **Step 2: Run to verify failure** — `OMP_NUM_THREADS=4 uv run pytest tests/intent/test_map.py -q` → FAIL.
- [ ] **Step 3: Implement `map`** in `src/xtrax/intent/map.py`. (Shadowing the builtin `map` is intentional inside this module; import it elsewhere as `from xtrax.intent import map as xmap` in tests.)
- [ ] **Step 4: Run to verify pass** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(intent): map -- vmap or chunked per decision, per AD path"`
---

### Task 6: Resolver

**Files:**
- Create: `src/xtrax/intent/resolve.py`
- Test: `tests/intent/test_resolve.py`

**Interfaces:**
- Consumes: `SiteRecord` (Task 3), `Context` and errors (Task 1), `SiteDecision`/`Plan` (Task 2), `BatchPlanner`, `AxisSpec`, `MemoryBudget`, `BudgetInfeasibleError` (`xtrax.tiling`), `lowered_memory_estimate` (`xtrax.tiling.estimators`).
- Produces: `resolve(records: Sequence[SiteRecord], ctx: Context, *, overrides: Mapping[str, SiteStrategyOverride] | None = None, diff_available: bool = True, diff_unavailable_reason: str = "") -> Plan`, where `SiteStrategyOverride = tuple[strategy_primal, strategy_diff | None]` keyed by site label (user `name`).
- Decisions the implementer cannot infer:
  - **Map sites → BatchPlanner:** one `AxisSpec(name=str(key), cardinality=<leading dim>, default_batch_size=...)` per map site in record order (pre-order; this is the demotion priority). Without `ctx.memory_budget`: `default_batch_size = cardinality` (→ `Vmap`). With it: `default_batch_size = ctx.chunk_size`, `BatchPlanner(budget=ctx.memory_budget)` (joint mode demotes to `ChunkedMap(chunk_size)`). The same strategy is used for primal and diff in the MVP; diff reason says so. `BudgetInfeasibleError` → `UnsatisfiableContextError(site=<largest site>, path="primal", constraint=<the planner's message>)`.
  - **Default estimator:** when `ctx.memory_budget` is an `int`, build `MemoryBudget(bytes=ctx.memory_budget, estimate=default_estimator(records))`. `default_estimator(records) -> Callable[[Sequence[AxisDecision]], int]` maps each decision (matched to its record by `AxisSpec.name == str(record.key)`) to `lowered_memory_estimate` of the site run under that strategy — `jax.vmap(record.fn)` for `Vmap`, `lambda xs: chunked_map(record.fn, xs, batch_size=b)` for `ChunkedMap(b)` — at `record.avals`, memoised per (key, strategy), and returns the sum. A user `MemoryBudget` is used as given. Criterion-4 tests use a deterministic user estimator.
  - **Loop sites:** primal `StaticWhile(length)` when `length is not None`; `Scan` when `ctx.target == "export"` and `ctx.export_format != "stablehlo"`; diff `Scan` when `diff_available`, else `None` with `diff_unavailable_reason`. `cond` loops (`length is None`): primal is the plain while; under `ctx.target == "export"` raise `UnsatisfiableContextError(constraint="export forbids unbounded while_loop")`.
  - **Hazard rules** (`HAZARDS: tuple[HazardRule, ...]`, each `(name, applies(decision, ancestors, ctx) -> bool, severity: Literal["error", "warning"])`):
    1. `scan_around_while` — a `ChunkedMap`/`Scan` primal decision whose subtree has a `StaticWhile` primal, on backends in `SCAN_AROUND_WHILE_BACKENDS = frozenset({"cuda"})` → prefer `Vmap` for the outer map if it fits the budget, else severity `"error"`.
    2. `long_static_scan` — a loop whose primal would be `Scan` with `length > LONG_SCAN_THRESHOLD = 256` → `"warning"`.
  - **Precedence:** context rules are hard (an override violating one raises `UnsatisfiableContextError`); overrides beat hazard rules and append `"override:<label> bypassed <hazard>"` to `warnings`.

- [ ] **Step 1: Write the failing tests** (build `SiteRecord`s directly)

```python
def test_no_budget_all_maps_vmap_loops_static_while():
def test_budget_matches_batchplanner_joint_mode():          # criterion 4
    est = lambda decisions: sum(64 if isinstance(d.strategy, Vmap) else 8 for d in decisions)
    budget = MemoryBudget(bytes=100, estimate=est)
    plan = resolve(records, Context(memory_budget=budget, chunk_size=32))
    expected = BatchPlanner(budget=budget).plan(axis_specs_for(records, chunk_size=32))
    assert [d.primal for d in plan.decisions if d.kind == "map"] == [d.strategy for d in expected.decisions]
def test_infeasible_budget_names_site():                    # criterion 6
def test_export_non_stablehlo_loop_resolves_scan():
def test_export_cond_loop_refused():                        # criterion 6
def test_diff_unavailable_recorded():
def test_scan_around_while_prefers_vmap_on_cuda():
def test_override_beats_hazard_with_warning():
def test_override_violating_context_raises():
```

- [ ] **Step 2: Run to verify failure** — `OMP_NUM_THREADS=4 uv run pytest tests/intent/test_resolve.py -q` → FAIL.
- [ ] **Step 3: Implement** `resolve.py`.
- [ ] **Step 4: Run to verify pass** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(intent): resolver over BatchPlanner with context and hazard rules"`
---

### Task 7: `compile`

**Files:**
- Create: `src/xtrax/intent/compile.py`
- Modify: `src/xtrax/intent/__init__.py` (export `loop`, `map`, `compile`, `Plan`, `SiteDecision`)
- Modify: `CHANGELOG.md` (`[Unreleased]` → Added: experimental `xtrax.intent`)
- Test: `tests/intent/test_compile.py`

**Interfaces:**
- Consumes: Tasks 1–6.
- Produces: `compile(fn, ctx: Context = Context(), *, overrides=None) -> Compiled`; `Compiled.__call__(*args, **kwargs)`; `Compiled.plan_for(*args, **kwargs) -> Plan` (plans without executing); `Compiled.plans: int` (number of plan passes run, for tests).
- Behaviour the implementer cannot infer:
  1. Cache key = `(jax.tree_util.tree_structure(args), tuple((leaf.shape, leaf.dtype) for leaf in leaves), ctx)`; one Plan and one `jax.jit` wrapper per key.
  2. Plan pass: abstract inputs = `jax.ShapeDtypeStruct` of every leaf (works for tracer args under an outer transform). Run `jax.make_jaxpr(fn)` under `activate(Session("plan"))`. Then, if any leaf is inexact, run `jax.make_jaxpr(lambda *a: jax.linearize(partial_over_inexact(fn), ...))` under a second plan session to record `diff`-path sites; on any exception set `diff_available=False` with the exception text as reason. Merge records (diff-path keys already carry `path="diff"`).
  3. `resolve(...)` → Plan.
  4. Execution: `jax.jit` of a wrapper that runs `fn` under `activate(Session("exec", records, plan.by_key()))` and, after `fn` returns, checks the primal-path key sequence equals the plan's primal-path sequence (else `SiteMismatchError(constraint="site sequence differs")`). Diff-path sites validate individually inside their re-entered session (Task 3).
  5. Post-hoc hazard check: on first execution per key, when `ctx.resolved_backend()` is in `SCAN_AROUND_WHILE_BACKENDS` (Task 6), take `jax.make_jaxpr` of the executed wrapper; for every equation from `xtrax.profiling.jaxpr.iter_jaxpr_eqns` whose `primitive.name == "scan"`, look through `iter_jaxpr_eqns(sub)` for each `sub` in `sub_jaxprs(eqn.params)`; any equation with `primitive.name == "while"` raises `IntentError` unless the Plan carries an `"override:… bypassed scan_around_while"` warning. Test with a monkeypatched backend set containing `"cpu"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_compile_unconstrained_equals_plain_jax():          # criteria 1, 4 (no budget)
def test_nested_sites_no_false_mismatch():                  # criterion 5
    # loop-in-map and map-in-loop, sites in both lax.cond branches;
    # run compiled fn plain, under jax.grad, under jax.checkpoint; no SiteMismatchError; values equal references
def test_nested_site_under_grad_uses_diff_decision():       # criterion 5: diff-path site resolved from the Plan, not defaults
def test_nonreproducible_trace_raises_before_execution():   # criterion 6 (trace-time side effect changes a shape; error names site and "planned")
def test_unsatisfiable_context_raises_at_compile_call():    # criterion 6
def test_plans_once_zero_recompiles():                      # criterion 7
    with jax.log_compiles(True):  # count via caplog
        for _ in range(3): f(x)
    assert f.plans == 1  # and no new compile logged after the first call
def test_plan_table_and_json_from_compiled():               # criterion 7
def test_compile_new_shape_replans():                       # Review Focus 5
def test_compile_under_outer_vmap():                        # Review Focus 5
def test_no_equinox_import():                               # subprocess: import xtrax.intent with an import hook that raises on "equinox"
def test_posthoc_scan_around_while_check():                 # backend set monkeypatched to {"cpu"}; override forcing ChunkedMap around a loop raises IntentError without the override warning, passes with it
```

- [ ] **Step 2: Run to verify failure** — `OMP_NUM_THREADS=4 uv run pytest tests/intent/test_compile.py -q` → FAIL.
- [ ] **Step 3: Implement** `compile.py`; prototype is `spike_two_pass.py`.
- [ ] **Step 4: Run to verify pass** — `OMP_NUM_THREADS=4 uv run pytest tests/intent -q` → PASS (whole `tests/intent/` directory is narrow enough).
- [ ] **Step 5: Lint and type-check** — `uv run ruff check --no-fix src/xtrax/intent tests/intent && uv run ruff format --check src/xtrax/intent tests/intent && jaxlint check src/xtrax/intent && uv run ty check src/xtrax/intent` → clean (ty: no new diagnostics vs `scripts/git-hooks/ty-baseline.txt`).
- [ ] **Step 6: Commit** — `git commit -m "feat(intent): compile -- two-pass plan, per-path site validation"`
