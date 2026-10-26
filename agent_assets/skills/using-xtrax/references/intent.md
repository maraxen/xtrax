> Part of the `using-xtrax` skill (`agent_assets/skills/using-xtrax/SKILL.md`) — TIER-2 deep reference.

# xtrax.intent

`xtrax.intent` is the first answer for a loop or map whose strategy should follow the caller's context. One call site runs as a `while_loop` when undifferentiated and as a `lax.scan` under AD, and a map runs as `Vmap` or as the `ChunkedMap` a memory budget requires. `BatchPlanner` is the layer `intent.map` plans with, and stays the direct API for a `BatchPlan` you build yourself. — verify: `agent_assets/skills/using-xtrax/SKILL.md:141-166`

Criterion 8 is the pre-registered GPU gate for leaving experimental. — verify: `.praxia/docs/superpowers/specs/261023_intent-resolution-design.md:216-218`

The user ruled criterion 8 met on equivalence on 261025, after the two runs below. The ruling is recorded on aminx PR #232, not in this repository.

On two pre-registered Blackwell (SM120) workloads, intent matched the hand-written forms exactly. It did so at equal compile and step cost, with gradients equal to a `lax.scan` reference and zero steady-state recompiles. The workloads were an autoregressive decode (aminx bathos run `3b93937a`) and Langevin trajectories under an outer map (run `c3c745a5`). Both runs returned `no_contrast`. The hand-picked scan form compiled as fast as the while form (1.03×), and `lax.map` around a `while_loop` compiled as fast as `vmap` around it (0.94×). The compile gap the `scan_around_while` rule guards against did not reproduce on current JAX. Read intent's value as one flag-free, differentiable call site, not as a compile-time win.

Names from `xtrax.intent` used below are in `__all__`. — verify: `src/xtrax/intent/__init__.py:25-46`

## Example

`compile(fn, ctx)` plans `fn` once per input signature and returns a callable. `plan_for` returns the `Plan` without running the jitted wrapper. `Plan.table` renders one row per site. — verify: `src/xtrax/intent/_compile.py:1037-1062` — verify: `src/xtrax/intent/_compile.py:943-959` — verify: `src/xtrax/intent/plan.py:173-205`

```python
import jax.numpy as jnp
from xtrax.intent import Context, compile, loop, map  # verify: src/xtrax/intent/__init__.py:16-20

def fn(xs):
    def step(carry, x):
        return carry + jnp.sum(x), carry

    carry, _ys = loop(step, jnp.float32(0.0), xs=xs, name="steps")  # verify: src/xtrax/intent/_loop.py:367-375
    return map(lambda z: z + carry, xs, name="batch")  # verify: src/xtrax/intent/_map.py:219-224

xs = jnp.ones((8, 4), dtype=jnp.float32)
planned = compile(fn, Context())  # verify: src/xtrax/intent/_compile.py:1037-1039
print(planned.plan_for(xs).table())  # verify: src/xtrax/intent/_compile.py:943
```

`table()` renders key, kind, primal, diff, both reasons, and warnings. A warning about the plan as a whole, such as an argument that could not be differentiated, follows the rows as a `warning:` line. — verify: `src/xtrax/intent/plan.py:184-188` — verify: `src/xtrax/intent/plan.py:204`

The rows below are a fixed-length loop (`StaticWhile`, diff `Scan`) and an unbudgeted map with cardinality above 1 (`Vmap` on both paths). — verify: `src/xtrax/intent/_resolve.py:375-382` — verify: `src/xtrax/intent/_resolve.py:426-428` — verify: `src/xtrax/intent/_resolve.py:40` — verify: `src/xtrax/intent/_resolve.py:442`

```
key	kind	primal	diff	primal_reason	diff_reason	warnings
primal/steps#0	loop	StaticWhile(length=8)	Scan(ordered_sinks=True)	fixed length resolves to StaticWhile	diff path resolves to Scan	
primal/batch#0	map	Vmap	Vmap	no memory budget; cardinality selects Vmap	same strategy as primal in the MVP	
```

The loop length is the leading axis of `xs`. — verify: `src/xtrax/intent/_loop.py:97-122`

A fixed length on `target="jit"` resolves to `StaticWhile`. — verify: `src/xtrax/intent/_resolve.py:378-380`

The diff path resolves to `Scan`. — verify: `src/xtrax/intent/_resolve.py:381-382`

`Scan` prints `ordered_sinks`. — verify: `src/xtrax/intent/plan.py:297-298` — verify: `src/xtrax/tiling/strategy.py:78`

A map with no memory budget and cardinality above 1 resolves to `Vmap`. — verify: `src/xtrax/intent/_resolve.py:426-428`

The map diff reason is the shared MVP string. — verify: `src/xtrax/intent/_resolve.py:40` — verify: `src/xtrax/intent/_resolve.py:442`

## Context

`Context` is a frozen record. `resolved_backend` is the one place that falls back to the JAX default. — verify: `src/xtrax/intent/context.py:1-4` — verify: `src/xtrax/intent/context.py:60-73`

| Field | Stored | What it changes |
|---|---|---|
| `memory_budget` | `MemoryBudget`, an `int` byte count, or `None`. — verify: `src/xtrax/intent/context.py:20-21` — verify: `src/xtrax/intent/context.py:30` | `None` skips joint planning. An `int` becomes `MemoryBudget(bytes=..., estimate=default_estimator(records))`. A `MemoryBudget` is used as given and is the budget `BatchPlanner` plans against. — verify: `src/xtrax/intent/_resolve.py:284-291` — verify: `src/xtrax/intent/_resolve.py:317-339` |
| `target` | `"jit"` or `"export"`. Default `"jit"`. — verify: `src/xtrax/intent/context.py:22` — verify: `src/xtrax/intent/context.py:31` | `"export"` refuses a cond-form loop. A fixed-length loop primal resolves to `Scan` when `target="export"` and `export_format` is other than `"stablehlo"`. — verify: `src/xtrax/intent/_resolve.py:741-747` — verify: `src/xtrax/intent/_resolve.py:375-377` |
| `export_format` | Format name, or `None`. — verify: `src/xtrax/intent/context.py:23-24` — verify: `src/xtrax/intent/context.py:32` | Required when `target` is `"export"`. `"stablehlo"` keeps a fixed-length loop on `StaticWhile`. — verify: `src/xtrax/intent/context.py:55-56` — verify: `src/xtrax/intent/_resolve.py:375-380` |
| `backend` | Device backend name, or `None`. — verify: `src/xtrax/intent/context.py:25` — verify: `src/xtrax/intent/context.py:33` | `resolved_backend()` returns this name, or `jax.default_backend()` when it is `None`. `scan_around_while` requires `_is_scan_around_while_backend` of that name. The set is `"cuda"` and `"gpu"`. `"gpu"` is JAX's platform name for CUDA and ROCm; the hazard is accepted for both until measured otherwise. The primal is `ChunkedMap` or `Scan`, and a subtree decision's primal is `StaticWhile`. — verify: `src/xtrax/intent/context.py:60-73` — verify: `src/xtrax/intent/_resolve.py:35-37` — verify: `src/xtrax/intent/_resolve.py:196-241` |
| `chunk_size` | Positive int. Default `32`. — verify: `src/xtrax/intent/context.py:26` — verify: `src/xtrax/intent/context.py:34` — verify: `src/xtrax/intent/context.py:57-58` | `AxisSpec.default_batch_size` for each planned map. — verify: `src/xtrax/intent/_resolve.py:329-335` |
| `kernels` | `bool`. Default `False`. Records whether kernel implementations are preferred. — verify: `src/xtrax/intent/context.py:27` — verify: `src/xtrax/intent/context.py:35` | Stored on the frozen record. Strategy selection reads `memory_budget`, `target`, `export_format`, `resolved_backend()`, and `chunk_size`. — verify: `src/xtrax/intent/context.py:15-35` — verify: `src/xtrax/intent/_resolve.py:234` — verify: `src/xtrax/intent/_resolve.py:286` — verify: `src/xtrax/intent/_resolve.py:333` — verify: `src/xtrax/intent/_resolve.py:375` |

## Dispatch

| Site | Resolution |
|---|---|
| Fixed-length `loop`, undifferentiated, `target="jit"` or `export_format="stablehlo"` | Primal `StaticWhile(length=...)`, run as one `while_loop` that collects `ys`. — verify: `src/xtrax/intent/_resolve.py:375-380` — verify: `src/xtrax/intent/_loop.py:185-201` |
| That loop, differentiated | Diff strategy `Scan`. The JVP always runs `jax.lax.scan`. For a `StaticWhile` primal it calls `_scan_collect`, which writes per-step outputs into a carried buffer laid out like the primal's. For a `Scan` primal it calls `_scan`. — verify: `src/xtrax/intent/_resolve.py:381-382` — verify: `src/xtrax/intent/_loop.py:311-320` — verify: `src/xtrax/intent/_loop.py:163-182` — verify: `src/xtrax/intent/_loop.py:142-150` |
| `loop(..., cond=)` (no `length`, no `xs`) | Primal `WhileCarry`, run as `lax.while_loop`. Diff is `None` (forward-mode only). — verify: `src/xtrax/intent/_resolve.py:366-374` — verify: `src/xtrax/intent/_loop.py:347-364` — verify: `src/xtrax/intent/_loop.py:388-394` |
| Fixed-length `loop` with `target="export"` and `export_format` other than `"stablehlo"` | Primal `Scan`. — verify: `src/xtrax/intent/_resolve.py:375-377` |
| `map` with no memory budget, or leading size `<= 1` | `Vmap` on the primal and on the diff path. — verify: `src/xtrax/intent/_resolve.py:422-428` — verify: `src/xtrax/intent/_resolve.py:442` |
| `map` with leading size `> 1` and a memory budget | The `Vmap` or `ChunkedMap` that `BatchPlanner` returns for that axis. — verify: `src/xtrax/intent/_resolve.py:304-339` — verify: `src/xtrax/intent/_resolve.py:429-441` |
| `map` with no session decision | `Vmap` on both paths. — verify: `src/xtrax/intent/_map.py:128-148` |

A planned map is a primal-chain map whose leading size is greater than 1. — verify: `src/xtrax/intent/_resolve.py:299-301`

`SiteStrategyOverride` is `(primal strategy, diff strategy or None)`, keyed by the site label on `compile(..., overrides=...)`. — verify: `src/xtrax/intent/_resolve.py:27` — verify: `src/xtrax/intent/__init__.py:19` — verify: `src/xtrax/intent/_compile.py:1041`

A map override primal is `Vmap` or `ChunkedMap`. Its diff is `Vmap`, `ChunkedMap`, or `None`. A fixed-length loop override primal is `StaticWhile` or `Scan`. Its diff is `Scan` or `None`. A cond loop's only legal primal override is `WhileCarry`; `Scan` and `StaticWhile` raise `IntentError` at plan time with `path="primal"`. — verify: `src/xtrax/intent/_resolve.py:479-528`

An unknown label raises `ValueError` naming that label. — verify: `src/xtrax/intent/_resolve.py:470-472`

A site with no `name` is labelled `module:line` from its caller, for example `models.encoder.layers:42`. Two files that give one such label in a session raise `IntentError`; pass `name=` to one of them. — verify: `src/xtrax/intent/sites.py:412` — verify: `src/xtrax/intent/sites.py:260-279`

An override that trips a hazard records `override:<label> bypassed <name>` and keeps the override. — verify: `src/xtrax/intent/_resolve.py:866-868`

## Errors

`IntentError` stores `site`, `path`, and `constraint`, and includes all three in the message. — verify: `src/xtrax/intent/errors.py:8-15` — verify: `src/xtrax/intent/errors.py:17-39`

`UnsatisfiableContextError` is that error when no strategy satisfies the context. — verify: `src/xtrax/intent/errors.py:90-91`

| Constraint it names | `site` |
|---|---|
| The `BatchPlanner` message, when the joint budget is infeasible. `path` is `"primal"`. — verify: `src/xtrax/intent/_resolve.py:340-349` | The largest planned map's key. — verify: `src/xtrax/intent/_resolve.py:346` |
| `override exceeds memory budget: estimate <n> > <bytes>`. — verify: `src/xtrax/intent/_resolve.py:729-736` | The first overridden map's key. — verify: `src/xtrax/intent/_resolve.py:733` |
| `export forbids unbounded while_loop`. — verify: `src/xtrax/intent/_resolve.py:741-747` | That loop's key. — verify: `src/xtrax/intent/_resolve.py:744` |
| `export forbids StaticWhile when export_format is not stablehlo`. — verify: `src/xtrax/intent/_resolve.py:748-758` | That loop's key. — verify: `src/xtrax/intent/_resolve.py:755` |
| `scan_around_while`, when moving the outer map back to `Vmap` exceeds the joint budget left after the switches of earlier sites. — verify: `src/xtrax/intent/_resolve.py:880-886` | That map's key. — verify: `src/xtrax/intent/_resolve.py:884` |

`SiteMismatchError` is that error when a site decision does not match the requested path. — verify: `src/xtrax/intent/errors.py:86-87`

| What happened | What it names |
|---|---|
| Exec mode, and the key is absent. — verify: `src/xtrax/intent/sites.py:173-174` — verify: `src/xtrax/intent/sites.py:294-300` | `site` is `str(key)`, `path` is `key.path`, `constraint` is `"site not in plan"`. — verify: `src/xtrax/intent/sites.py:295-300` |
| A shape, dtype, or aval count differs. — verify: `src/xtrax/intent/sites.py:301-308` — verify: `src/xtrax/intent/sites.py:454-459` | `constraint` is `planned <avals>, traced <avals>`. — verify: `src/xtrax/intent/sites.py:302-307` |
| The primal site sequence at execution differs from the plan. — verify: `src/xtrax/intent/_compile.py:699-705` | `site` is `"compile"`, `path` is `"primal"`, `constraint` is `"site sequence differs"`. — verify: `src/xtrax/intent/_compile.py:700-704` |

## Limits

`import xtrax.intent` does not import equinox. Iterator names on `xtrax.tiling` load through `__getattr__`, so equinox stays unloaded until a caller touches an iterator name. — verify: `src/xtrax/tiling/__init__.py:134-160`

`test_no_equinox_import` blocks `equinox` on `sys.meta_path` and still imports `xtrax.intent`. — verify: `tests/intent/test_compile.py:282-319`

`default_estimator` prices each planned map under each candidate strategy with a real ahead-of-time compile, which costs seconds per distinct shape. Under a tight budget it therefore compiles the `ChunkedMap` candidate too, before the `scan_around_while` rule applies. A map traced by `intent.map` lowers as a function of its `xs` pytree plus the hoisted constants. The `IntentError` with `constraint` `"default estimator takes one positional array"` applies only to a hand-built `SiteRecord` with no `xs_tree`, no constants, and more than one aval. — verify: `src/xtrax/intent/_resolve.py:81-98` — verify: `src/xtrax/tiling/estimators.py:72-99` — verify: `src/xtrax/intent/_map.py:277-285` — verify: `src/xtrax/intent/_resolve.py:942-948`

Pass a `MemoryBudget` whose `estimate` is your own function. An existing `MemoryBudget` is kept as given. — verify: `src/xtrax/intent/context.py:30` — verify: `src/xtrax/intent/_resolve.py:289-290` — verify: `src/xtrax/tiling/budget.py:36-49`

Cond-form `loop` (`cond=`) resolves to `WhileCarry` and runs `lax.while_loop`. The plan records `diff=None` with the reason `conditional loop: jax.lax.while_loop (forward-mode only)`. Forward mode is that while loop's JVP. Reverse mode raises JAX's own `while_loop` error. No derivative strategy is added. Export refuses the form. — verify: `src/xtrax/intent/_resolve.py:366-374` — verify: `src/xtrax/intent/_loop.py:347-364` — verify: `src/xtrax/intent/_loop.py:388-394`

The post-hoc scan-around-while check names `site="compile"`, `path="primal"`, and `constraint="scan_around_while"`. The resolve-time hazard names the individual site key. — verify: `src/xtrax/intent/_compile.py:887-892` — verify: `src/xtrax/intent/_resolve.py:881-886`

The same check traces before the jitted program runs, and caches the verdict per signature, pass and fail. It runs when `_is_scan_around_while_backend` accepts `resolved_backend()`. A bypass warning exempts only the site that carries it. — verify: `src/xtrax/intent/_compile.py:863-866` — verify: `src/xtrax/intent/_compile.py:993-1004`

Anti-goals held by this module: a resolution computes the same function as the reference resolution, up to float reassociation; callers write JAX; an inferred property is verified on the jaxpr or declared. — verify: `.praxia/docs/superpowers/specs/261023_intent-resolution-design.md:50-52`
