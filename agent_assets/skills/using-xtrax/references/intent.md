> Part of the `using-xtrax` skill (`agent_assets/skills/using-xtrax/SKILL.md`) — TIER-2 deep reference.

# xtrax.intent

> Experimental until criterion 8 (GPU run) passes; the BatchPlanner decision tree above remains the default.

`xtrax.intent` is experimental. — verify: `src/xtrax/intent/__init__.py:1`

Criterion 8 is the pre-registered GPU run that gates leaving experimental. — verify: `.praxia/docs/superpowers/specs/261023_intent-resolution-design.md:216-218`

The BatchPlanner decision tree in this skill stays the default. — verify: `agent_assets/skills/using-xtrax/SKILL.md:143-164`

Names below are the public export list. — verify: `src/xtrax/intent/__init__.py:20-40`

## Example

`compile(fn, ctx)` plans `fn` once per input signature and returns a callable. `plan_for` returns the `Plan` without running the jitted wrapper. `Plan.table` prints one row per site. — verify: `src/xtrax/intent/compile.py:739-762` — verify: `src/xtrax/intent/compile.py:656-672` — verify: `src/xtrax/intent/plan.py:153-183`

```python
import jax.numpy as jnp
from xtrax.intent import Context, compile, loop, map  # verify: src/xtrax/intent/__init__.py:11-15

def fn(xs):
    def step(carry, x):
        return carry + jnp.sum(x), carry

    carry, _ys = loop(step, jnp.float32(0.0), xs=xs, name="steps")  # verify: src/xtrax/intent/loop.py:338-346
    return map(lambda z: z + carry, xs, name="batch")  # verify: src/xtrax/intent/map.py:220-224

xs = jnp.ones((8, 4), dtype=jnp.float32)
planned = compile(fn, Context())  # verify: src/xtrax/intent/compile.py:739-741
print(planned.plan_for(xs).table())  # verify: src/xtrax/intent/compile.py:656
```

`table()` renders key, kind, primal, diff, both reasons, and warnings. — verify: `src/xtrax/intent/plan.py:153`

On this shape the default `Context` prints the rows below. — verify: `src/xtrax/intent/resolve.py:346-350` — verify: `src/xtrax/intent/resolve.py:394-396` — verify: `src/xtrax/intent/resolve.py:39`

```
key	kind	primal	diff	primal_reason	diff_reason	warnings
primal/steps#0	loop	StaticWhile(length=8)	Scan(ordered_sinks=True)	fixed length resolves to StaticWhile	diff path resolves to Scan
primal/batch#0	map	Vmap	Vmap	no memory budget; cardinality selects Vmap	same strategy as primal in the MVP
```

The loop length is the leading axis of `xs`. — verify: `src/xtrax/intent/loop.py:101-124`

A fixed length on `target="jit"` resolves to `StaticWhile`. — verify: `src/xtrax/intent/resolve.py:346-348`

The diff path resolves to `Scan`. — verify: `src/xtrax/intent/resolve.py:349-350`

`Scan` prints `ordered_sinks`. — verify: `src/xtrax/intent/plan.py:270-271` — verify: `src/xtrax/tiling/strategy.py:78`

A map with no memory budget and cardinality above 1 resolves to `Vmap`. — verify: `src/xtrax/intent/resolve.py:394-396`

The map diff reason is the shared MVP string. — verify: `src/xtrax/intent/resolve.py:39` — verify: `src/xtrax/intent/resolve.py:410`

## Context

`Context` is a frozen record. `resolved_backend` is the one place that falls back to the JAX default. — verify: `src/xtrax/intent/context.py:1-4` — verify: `src/xtrax/intent/context.py:62-75`

| Field | Stored | What it changes |
|---|---|---|
| `memory_budget` | `MemoryBudget`, an `int` byte count, or `None`. — verify: `src/xtrax/intent/context.py:22-23` — verify: `src/xtrax/intent/context.py:32` | `None` skips joint planning. An `int` becomes `MemoryBudget(bytes=..., estimate=default_estimator(records))`. A `MemoryBudget` is used as given and is the budget `BatchPlanner` plans against. — verify: `src/xtrax/intent/resolve.py:264-271` — verify: `src/xtrax/intent/resolve.py:297-319` |
| `target` | `"jit"` or `"export"`. Default `"jit"`. — verify: `src/xtrax/intent/context.py:24` — verify: `src/xtrax/intent/context.py:33` | `"export"` refuses a cond-form loop. With `export_format` other than `"stablehlo"`, a fixed-length loop primal resolves to `Scan`. — verify: `src/xtrax/intent/resolve.py:636-653` — verify: `src/xtrax/intent/resolve.py:343-345` |
| `export_format` | Format name, or `None`. — verify: `src/xtrax/intent/context.py:25-26` — verify: `src/xtrax/intent/context.py:34` | Required when `target` is `"export"`. `"stablehlo"` keeps a fixed-length loop on `StaticWhile`. — verify: `src/xtrax/intent/context.py:57-58` — verify: `src/xtrax/intent/resolve.py:343-348` |
| `backend` | Device backend name, or `None`. — verify: `src/xtrax/intent/context.py:27` — verify: `src/xtrax/intent/context.py:35` | `resolved_backend()` returns this name, or `jax.default_backend()` when it is `None`. `scan_around_while` applies when that name is in `SCAN_AROUND_WHILE_BACKENDS` (`"cuda"`). — verify: `src/xtrax/intent/context.py:62-75` — verify: `src/xtrax/intent/resolve.py:36` — verify: `src/xtrax/intent/resolve.py:214-215` |
| `chunk_size` | Positive int. Default `32`. — verify: `src/xtrax/intent/context.py:28` — verify: `src/xtrax/intent/context.py:36` — verify: `src/xtrax/intent/context.py:59-60` | `AxisSpec.default_batch_size` for each planned map. — verify: `src/xtrax/intent/resolve.py:310-314` |
| `kernels` | `bool`. Default `False`. Records whether kernel implementations are preferred. — verify: `src/xtrax/intent/context.py:29` — verify: `src/xtrax/intent/context.py:37` | Stored on the frozen record. Strategy selection reads `memory_budget`, `target`, `export_format`, `resolved_backend()`, and `chunk_size`. — verify: `src/xtrax/intent/context.py:17-37` — verify: `src/xtrax/intent/resolve.py:264-271` — verify: `src/xtrax/intent/resolve.py:332-351` |

## Dispatch

| Site | Resolution |
|---|---|
| Fixed-length `loop`, undifferentiated, `target="jit"` or `export_format="stablehlo"` | Primal `StaticWhile(length=...)`, run as one `while_loop`. — verify: `src/xtrax/intent/resolve.py:346-348` — verify: `src/xtrax/intent/loop.py:187-190` |
| That loop, differentiated | Diff strategy `Scan`. The JVP traces `lax.scan`. — verify: `src/xtrax/intent/resolve.py:349-350` — verify: `src/xtrax/intent/loop.py:270-290` |
| `loop(..., cond=)` (no `length`, no `xs`) | Primal `WhileCarry`, run as `lax.while_loop`. — verify: `src/xtrax/intent/resolve.py:340-342` — verify: `src/xtrax/intent/loop.py:325-335` — verify: `src/xtrax/intent/loop.py:401-402` |
| Fixed-length `loop` with `target="export"` and `export_format` other than `"stablehlo"` | Primal `Scan`. — verify: `src/xtrax/intent/resolve.py:343-345` |
| `map` with no memory budget, or leading size `<= 1` | `Vmap` on the primal and on the diff path. — verify: `src/xtrax/intent/resolve.py:391-396` — verify: `src/xtrax/intent/resolve.py:410` |
| `map` with leading size `> 1` and a memory budget | The `Vmap` or `ChunkedMap` that `BatchPlanner` returns for that axis. — verify: `src/xtrax/intent/resolve.py:284-319` — verify: `src/xtrax/intent/resolve.py:397-409` |
| `map` with no session decision | `Vmap` on both paths. — verify: `src/xtrax/intent/map.py:129-145` |

A planned map is a primal-chain map whose leading size is greater than 1. — verify: `src/xtrax/intent/resolve.py:279-281`

`SiteStrategyOverride` is `(primal strategy, diff strategy or None)`, keyed by the site label on `compile(..., overrides=...)`. — verify: `src/xtrax/intent/resolve.py:28` — verify: `src/xtrax/intent/__init__.py:17` — verify: `src/xtrax/intent/compile.py:753-755`

A map override is `Vmap` or `ChunkedMap` on each path. A loop override is `StaticWhile` or `Scan` on the primal, and `Scan` or `None` on the diff path. — verify: `src/xtrax/intent/resolve.py:447-480`

An unknown label raises `ValueError` naming that label. — verify: `src/xtrax/intent/resolve.py:438-440`

An override that trips a hazard records `override:<label> bypassed <name>` and keeps the override. — verify: `src/xtrax/intent/resolve.py:752-754`

## Errors

`IntentError` stores `site`, `path`, and `constraint`, and includes all three in the message. — verify: `src/xtrax/intent/errors.py:10-17` — verify: `src/xtrax/intent/errors.py:19-41`

`UnsatisfiableContextError` is that error when no strategy satisfies the context. — verify: `src/xtrax/intent/errors.py:92-93`

| Constraint it names | `site` |
|---|---|
| The `BatchPlanner` message, when the joint budget is infeasible. `path` is `"primal"`. — verify: `src/xtrax/intent/resolve.py:320-329` | The largest planned map's key. — verify: `src/xtrax/intent/resolve.py:324-327` |
| `override exceeds memory budget: estimate <n> > <bytes>`. — verify: `src/xtrax/intent/resolve.py:625-630` | The first overridden map's key. — verify: `src/xtrax/intent/resolve.py:628` |
| `export forbids unbounded while_loop`. — verify: `src/xtrax/intent/resolve.py:636-642` | That loop's key. — verify: `src/xtrax/intent/resolve.py:639` |
| `export forbids StaticWhile when export_format is not stablehlo`. — verify: `src/xtrax/intent/resolve.py:643-653` | That loop's key. — verify: `src/xtrax/intent/resolve.py:650` |
| `scan_around_while`, when moving the outer map back to `Vmap` exceeds the budget. — verify: `src/xtrax/intent/resolve.py:766-772` | That map's key. — verify: `src/xtrax/intent/resolve.py:769` |

`SiteMismatchError` is that error when a visit does not match the plan. — verify: `src/xtrax/intent/errors.py:88-89`

| What happened | What it names |
|---|---|
| Exec mode, and the key is absent. — verify: `src/xtrax/intent/sites.py:257-262` | `site` is `str(key)`, `path` is `key.path`, `constraint` is `"site not in plan"`. — verify: `src/xtrax/intent/sites.py:258-262` |
| A shape, dtype, or aval count differs. — verify: `src/xtrax/intent/sites.py:264-271` | `constraint` is `planned <avals>, traced <avals>`. — verify: `src/xtrax/intent/sites.py:265-270` |
| The primal site sequence at execution differs from the plan. — verify: `src/xtrax/intent/compile.py:484-491` | `site` is `"compile"`, `path` is `"primal"`, `constraint` is `"site sequence differs"`. — verify: `src/xtrax/intent/compile.py:486-490` |

## Limits

`import xtrax.intent` does not import equinox. Iterator names on `xtrax.tiling` load through `__getattr__`, so equinox stays unloaded until a caller touches an iterator name. — verify: `src/xtrax/tiling/__init__.py:134-139`

`test_no_equinox_import` blocks `equinox` on `sys.meta_path` and still imports `xtrax.intent`. — verify: `tests/intent/test_compile.py:271-284`

`default_estimator` lowers each planned map. A map site with one input leaf is the shape it accepts. A multi-leaf map whose `memory_budget` is an `int` raises `IntentError` with `constraint` `"default estimator takes one positional array"`. — verify: `src/xtrax/intent/resolve.py:60-66` — verify: `src/xtrax/intent/resolve.py:271` — verify: `src/xtrax/intent/resolve.py:822-828`

Pass a `MemoryBudget` whose `estimate` is your own function. An existing `MemoryBudget` is kept as given. — verify: `src/xtrax/intent/context.py:32` — verify: `src/xtrax/intent/resolve.py:269` — verify: `src/xtrax/tiling/budget.py:36-49`

Cond-form `loop` (`cond=`) resolves to `WhileCarry` and runs `lax.while_loop`. The `custom_jvp` belongs to the static runner. `WhileCarry` has no reverse-mode VJP. The plan records `Scan` on the diff path when the diff trace succeeds. Export refuses the form. — verify: `src/xtrax/intent/resolve.py:340-342` — verify: `src/xtrax/intent/loop.py:318-335` — verify: `src/xtrax/intent/loop.py:249-251` — verify: `src/xtrax/tiling/strategy.py:132` — verify: `src/xtrax/intent/resolve.py:349-350` — verify: `src/xtrax/intent/resolve.py:636-642`

The post-hoc scan-around-while check names `site="compile"`, `path="primal"`, and `constraint="scan_around_while"`. The resolve-time hazard names the individual site key. — verify: `src/xtrax/intent/compile.py:599-605` — verify: `src/xtrax/intent/resolve.py:767-771`

The same check runs when `resolved_backend()` is in `SCAN_AROUND_WHILE_BACKENDS` and the plan has no override-bypass warning. — verify: `src/xtrax/intent/compile.py:576-579`

Anti-goals held by this module: a resolution computes the same function as the reference resolution, up to float reassociation; callers write JAX; an inferred property is verified on the jaxpr or declared. — verify: `.praxia/docs/superpowers/specs/261023_intent-resolution-design.md:50-52`
