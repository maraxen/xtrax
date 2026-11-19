---
title: 'Intent MVP code-review fixes: waves A-C and re-review'
description: '/code-review xhigh on PR #228 found 15 issues; 11 fixed in three waves plus one follow-up, re-review 12/12 resolved; 5 parked as backlog.'
status: final
task_id: 261023_equinox-decouple-pallas
date: '261024'
verdict: 'PASS (re-review: 12 resolved, 0 not resolved, 1 new Minor parked as backlog #6165)'
base_sha: '715927d'
---
# Intent MVP code-review fixes: waves A-C and re-review

Branch `spec/equinox-decouple-261023`, PR #228. `/code-review xhigh` found 15 issues. At the
user's choice, 11 were fixed on this branch and 4 were parked as backlog. Each wave was a Cursor
fixer dispatched through praxia `surface_dispatch`. Every test it added failed before the fix,
and the controller then checked each wave.

| Wave | Commit | Findings |
|---|---|---|
| A: input handling | `1467140` | #2 estimator with closed-over consts, #5 Python scalars, #6 weak_type in plan and cache key, #13 empty `xs` |
| B: diff-path sites | `1b20118` | #3 linearize per input, #4 keep primal-pass diff sites, #11 canonical-twin fallback |
| C: plan vs runtime | `f5d5675` | #1 `gpu` backend name, #8 cond-loop overrides, #9 cond loops plan `diff=None`, #12 post-hoc before execution and per-site bypass, item 5 partial diff availability |

**Rulings**
- Parked: #7 joint budget after the scan_around_while rewrite (#6157), #10 Vmap batch_size in
  override re-estimates (#6158), #14 submodule shadowing (#6159), #15 basename site labels
  (#6160).
- Item 5 was added by the controller after checking Wave B. When any one argument failed, the
  plan showed `diff=None` while gradients w.r.t. the other arguments still ran a scan.
- Cond loops: no new derivative strategy. The plan states the loop is forward-mode only
  (`jax.lax.while_loop`), and reverse mode raises JAX's own error.
- `"gpu"` counts as hazardous, like `"cuda"`. JAX reports both CUDA and ROCm as `"gpu"`; ROCm
  has not been measured.
- New Minor from the re-review (partial-diff note only on loops with a diff strategy) is parked
  as #6165.
- Every dispatch was rejected at admission for ambient churn only (sandbox stub dotfiles, praxia
  telemetry/patches). HEAD was never reset.

The original findings (with failure scenarios) and the re-review follow, verbatim. Probe scripts
under `/tmp/claude-1000/` were throwaway.

---


## Original /code-review findings

```json
[
  {
    "file": "src/xtrax/intent/resolve.py",
    "line": 36,
    "summary": "SCAN_AROUND_WHILE_BACKENDS is {\"cuda\"}, but Context.resolved_backend() falls back to jax.default_backend(), which returns \"gpu\" on NVIDIA machines, so the scan_around_while rule (resolve) and the post-hoc check (compile.py:576) never fire with the default Context.",
    "failure_scenario": "On a CUDA box, compile(fn) with a budget that demotes an outer map to ChunkedMap around a StaticWhile loop: resolved_backend() returns 'gpu', which is not in {'cuda'}, so there is no Vmap rewrite, no error and no post-hoc rejection, and the scan-around-while pattern ships. Users who pass backend='gpu' (JAX's own name for it) are also unprotected; only an explicit backend='cuda' works."
  },
  {
    "file": "src/xtrax/intent/resolve.py",
    "line": 846,
    "summary": "default_estimator lowers jax.vmap(record.fn) with only the mapped argument, but record.fn is a closure_convert output that also takes the hoisted constants, so any int memory_budget crashes when a map body closes over a traced float value.",
    "failure_scenario": "compile(lambda xs, w: map(lambda z: jnp.sin(z * w), xs), Context(memory_budget=10**9)).plan_for(xs, w) raises TypeError 'inputs to the closure produced by closure_convert must have the same Pytree structure... got PyTreeDef(((), {}))' (reproduced). The same happens for any map nested in a loop whose body uses the carry, and for one-leaf pytree xs such as {'pos': arr}, which pass the len(avals)==1 guard but fail the tree-structure check."
  },
  {
    "file": "src/xtrax/intent/compile.py",
    "line": 384,
    "summary": "_trace_diff linearizes over every inexact input and swallows any exception, so one non-differentiable float input marks diff unavailable for the whole function and drops every diff-path record. jax.grad with respect to a different, differentiable argument then raises SiteMismatchError.",
    "failure_scenario": "fn(xs, w, aux) passes aux through jax.pure_callback and has map-in-loop sites using w. jax.grad(reference, argnums=1) works, but jax.grad(compile(fn), argnums=1) raises SiteMismatchError 'site not in plan (site=primal/outer_loop#0/diff/inner_map#0)' (reproduced). The plan also stores the callback's error text as the loop's diff_reason."
  },
  {
    "file": "src/xtrax/intent/compile.py",
    "line": 281,
    "summary": "_merge_records keeps only primal-chain records from the primal pass. Diff-path sites recorded there, when fn itself calls jax.grad (for example forces = -grad(energy)), are discarded and survive only if the separate diff pass runs and succeeds.",
    "failure_scenario": "fn(scale) with integer scale computes jax.grad(energy) of a loop with a nested map. _any_inexact is False, so the diff pass is skipped, the primal-pass record primal/l#0/diff/m#0 is dropped, and the first call raises SiteMismatchError 'site not in plan' (reproduced; the same fn with a float32 input works)."
  },
  {
    "file": "src/xtrax/intent/compile.py",
    "line": 154,
    "summary": "_leaf_signature, _abstract_tree, _any_inexact and loop._avals (loop.py:96) read leaf.shape and leaf.dtype on every leaf, so Python scalars, which lax.scan, vmap and jit all accept, crash under compile.",
    "failure_scenario": "compile(f)(xs, 2.0) raises AttributeError: 'float' object has no attribute 'shape' (reproduced). Inside a compiled function, loop(step, 0.0, xs=xs) also raises AttributeError, while the same loop outside a session and jax.lax.scan(step, 0.0, xs) both return 12.0 (reproduced). The tests always pass jnp.float32(0.0), which hides this."
  },
  {
    "file": "src/xtrax/intent/compile.py",
    "line": 137,
    "summary": "_abstract_tree builds ShapeDtypeStruct without weak_type, so the plan trace sees strong-typed inputs while the jit exec trace sees the caller's weak-typed ones. Dtype promotion differs between the two traces, and so do later site avals.",
    "failure_scenario": "fn(xs_bf16, s) with s = jnp.asarray(2.0) (weak ~float32): ys = map(lambda z: z * s, xs); map(jnp.sin, ys). Planning gives float32 ys, execution gives bfloat16, and the call raises SiteMismatchError 'planned float32, traced bfloat16' at primal/sin#0. Plain JAX returns bfloat16 (reproduced)."
  },
  {
    "file": "src/xtrax/intent/resolve.py",
    "line": 772,
    "summary": "scan_around_while checks each ChunkedMap's switch to Vmap on its own against the planner's original decisions. Several switches can therefore jointly exceed memory_budget, and the joint budget is never checked again unless overrides exist.",
    "failure_scenario": "On cuda, maps a and b each enclose a StaticWhile, and the costs are Vmap 100/100/1000 vs Chunked 90/90/10 with budget 200. The planner demotes all three (190). Switching a alone or b alone fits (200), so both become Vmap, and the emitted plan's joint estimate is 210 > 200 with no error (reproduced)."
  },
  {
    "file": "src/xtrax/intent/resolve.py",
    "line": 467,
    "summary": "Override checks accept Scan for a conditional loop (record.length is None), and WhileCarry is not an allowed override at all, so planning reports a Scan primal that the runtime refuses.",
    "failure_scenario": "compile(cond_fn, overrides={'w': (Scan(), None)}): plan_for prints 'Scan(ordered_sinks=True)' with reason override:w, then the call raises IntentError 'loop primal strategy must be WhileCarry, got Scan' from loop._cond_primal (reproduced). Any override on a cond loop is either rejected or fails at run time."
  },
  {
    "file": "src/xtrax/intent/resolve.py",
    "line": 350,
    "summary": "_loop_choice gives conditional loops diff=Scan() with reason 'diff path resolves to Scan', but _loop_cond runs a plain jax.lax.while_loop with no custom_jvp. The plan describes a derivative strategy that never runs.",
    "failure_scenario": "A cond loop under compile: jax.linearize in the diff pass succeeds, so the plan reports diff Scan. jax.grad(compiled) then fails with JAX's 'Reverse-mode differentiation does not work for lax.while_loop' error, and forward mode runs the while JVP rather than a scan. The plan contradicts what executes."
  },
  {
    "file": "src/xtrax/intent/resolve.py",
    "line": 617,
    "summary": "_check_map_override_budget re-estimates Vmap maps with batch_size=cardinality, but BatchPlanner (_budget_initial_decision) and _vmap_fits use batch_size=default_batch_size for Vmap. An estimator that reads batch_size gets different answers for the same plan.",
    "failure_scenario": "MemoryBudget estimate = sum(d.batch_size * bytes_per_row). Map B keeps Vmap within budget at batch_size=chunk_size. An override on map A then re-estimates B at batch_size=cardinality, the total exceeds the budget, and UnsatisfiableContextError blames A's override although nothing about B changed."
  },
  {
    "file": "src/xtrax/intent/_trace.py",
    "line": 113,
    "summary": "decision_for's diff fallback only flips the leaf key's path to 'primal' and leaves the parent chain alone. resolve treats twins by canonical(key), which rewrites the whole chain, so nested diff sites two or more levels deep miss their planned twin.",
    "failure_scenario": "A Session with decisions from resolve over primal records, under jax.grad of loop -> map -> loop. The inner loop's key (parent=(outer,diff,map), path=primal) does not fall back at all, and (parent=..diff.., path=diff) falls back to a non-canonical key. Both miss and silently run the defaults (StaticWhile/Vmap) instead of the planned decision, for example a budget-driven ChunkedMap."
  },
  {
    "file": "src/xtrax/intent/compile.py",
    "line": 702,
    "summary": "Compiled.__call__ runs the jitted program before _posthoc checks it, and marks the key as checked only when the check passes. A rejected program therefore fully executes, then raises, on every call. Separately, _bypass_warning turns the check off for the whole program when any single site's override bypassed the hazard.",
    "failure_scenario": "On cuda, the first call runs the slow scan-around-while program to completion, then raises IntentError, and each retry repeats the full run. Also, an override on map A that bypasses the hazard suppresses the post-hoc rejection of an unrelated map B's scan-around-while."
  },
  {
    "file": "src/xtrax/intent/loop.py",
    "line": 124,
    "summary": "_static_length indexes sizes[0] without checking for an empty list, so xs with no leaves and no length raises IndexError instead of jax.lax.scan's ValueError.",
    "failure_scenario": "loop(body, jnp.float32(0), xs=()) raises IndexError: list index out of range, where jax.lax.scan raises ValueError 'scan got no values to scan over and `length` not provided' (reproduced). The docstring promises scan's ValueError text."
  },
  {
    "file": "src/xtrax/intent/__init__.py",
    "line": 15,
    "summary": "Rebinding the package attributes compile and map to the functions hides the submodules xtrax.intent.compile and xtrax.intent.map from attribute access.",
    "failure_scenario": "import xtrax.intent.map as m gives the function, not the module (reproduced). monkeypatch.setattr('xtrax.intent.compile.SCAN_AROUND_WHILE_BACKENDS', ...) and mock.patch string targets fail to resolve. The new tests already work around this with sys.modules['xtrax.intent.compile']."
  },
  {
    "file": "src/xtrax/intent/sites.py",
    "line": 356,
    "summary": "call_label uses only os.path.basename of the caller's file, so sites in different modules that share a file name and line number get the same label, and label-keyed overrides and validation apply to both.",
    "failure_scenario": "models/encoder/layers.py:42 and models/decoder/layers.py:42 both label as 'layers.py:42'. overrides={'layers.py:42': (ChunkedMap(4), None)} meant for the encoder also rewrites the decoder map, and _check_map_override_budget attributes both to one override."
  }
]
```

---

## Scoped re-review: waves A–C (`715927d..f5d5675`)

HEAD `f5d5675`. Range is the three fix commits `1467140`, `1b20118`, `f5d5675`.
Parked findings #7, #10, #14, and #15 were not re-reviewed.

Probes: `OMP_NUM_THREADS=4 uv run --no-sync python /tmp/claude-1000/rr2_probes.py`
(15 passed, 0 failed). Each probe uses an input the fixer tests did not use.

## Tests

Each file was its own command:
`OMP_NUM_THREADS=4 uv run --no-sync pytest <file> -q --no-cov -p no:cacheprovider`.

| Command | Output |
| --- | --- |
| `tests/intent/test_compile.py` | `23 passed in 9.35s` |
| `tests/intent/test_resolve.py` | `33 passed in 0.74s` |
| `tests/intent/test_loop.py` | `26 passed in 9.25s` |
| `tests/intent/test_map.py` | `13 passed in 6.81s` |
| `tests/intent/test_sites.py` | `23 passed in 0.24s` |

`tests/intent/test_loop.py::test_loop_transforms_match_scan_reference` is in the 26 passed.

## Verdicts

### #2 closed-over constants — Resolved

Probe: a tuple of two `(6, 2)` arrays, a traced dict gain, and a traced bias vector, under `Context(memory_budget=10**9)`. `plan_for` succeeds, the call matches `jax.vmap`, and `_lowered_site` lowers both `Vmap` and `ChunkedMap(batch_size=2)` (the fixer test only checks the vmap path).

```
PASS #2 tuple xs + chunked consts: primal=Vmap() consts=2 vmap_bytes=156 chunk_bytes=288
```

### #5 Python scalars — Resolved

Probe: keyword `factor=3` (a Python int, not the fixer's positional `2.0`), and a compiled loop whose init is `(0.0, 0)`.

```
PASS #5 keyword int + tuple init: kw_int_dtype=float32 carry=(Array(66., dtype=float32), Array(4, dtype=int32, weak_type=True))
```

Both match `jax.vmap` / `jax.lax.scan`.

### #6 weak_type in the plan cache — Resolved

Probe: bfloat16 `(3, 5)` input and the scalar nested in a keyword dict, weak `jnp.asarray(1.5)` then strong `float32`. The fixer test uses a positional scalar and shape `(4, 3)`.

```
PASS #6 nested weak keyword: plans=2 weak=bfloat16 strong=float32
```

`plans` goes from 1 to 2, and both dtypes match plain JAX.

### #13 empty `xs` — Resolved

Probe: `xs={}`, `xs=[]`, `xs=((), {})`, and `xs=((),)` inside `compile`. The fixer test uses only `xs=()`.

```
PASS #13 empty pytrees: raised=['dict', 'list', 'nested']+compiled
```

Each raises `ValueError: scan got no values to scan over and `length` not provided.` No `IndexError`.

### #3 one nondiff input drops every diff site — Resolved

Probe: `fn(aux, rows, weight, *, token)` with `pure_callback` on the leading positional and on the keyword (`+1` / `-1`, not the fixer's trailing `* 2`). `rows` is `(3, 6)`.

```
PASS #3 nondiff first+keyword: diff path resolves to Scan; not differentiable w.r.t. aux (ValueError); not differentiable w.r.t. token (ValueError)
```

`jax.grad(..., argnums=2)` matches the scan/vmap reference, and `row_map` is present on the diff path. The reason does not contain the callback message.

### #4 primal-pass diff sites dropped for an exact input — Resolved

Probe: Python int `5` (not `jnp.int32(1)`) and an internal `jax.jacfwd` (not `jax.grad`), map label `jac_map`, rows `(3, 5)`.

```
PASS #4 python int jacfwd: plans=1 diff_sites=1
```

The value matches the scan/vmap `jacfwd`, and the plan keeps `jac_map` on the diff path.

### #11 canonical-twin fallback — Resolved

Probe: a three-level key `pad/primal → band/diff → cell/diff` (ordinal 1). The exact key stores `ChunkedMap(batch_size=7)` with `diff=None`. The canonical twin stores `ChunkedMap(batch_size=3)`. A leaf-only flip stores `Vmap`.

```
PASS #11 exact-key precedence: exact wins; leaf-flip ignored; primal key does not fall back
```

A logging dict shows the exact-key lookup is the only lookup (`looked == [deep]`). The fallback does not run when the exact key has a decision. A missing exact key still uses the canonical twin, not the leaf-flipped key. A primal key with no diff edge does not pick up an unrelated decision.

### #1 GPU backend name — Resolved

Probe: `Context(backend="gpu")` versus `backend="cpu"` on records the fixer test does not use (labels `pad` / `cell` / `side`, cardinality 12, chunk 4, budget 40, costs 22/14 and 36/9). The fixer test monkeypatches `jax.default_backend` and calls `resolve` with the cuda-test cost table.

```
PASS #1 explicit gpu backend: cpu=ChunkedMap(batch_size=4) gpu=Vmap() side=ChunkedMap(batch_size=4)
```

CPU keeps the planner's `ChunkedMap`. GPU rewrites only the map that encloses the `StaticWhile`, with reason `scan_around_while prefers Vmap`.

A second probe disables the resolver set and leaves compile's set as `{"cuda", "gpu"}`. `Context(backend="gpu")` still plans `ChunkedMap` and `Compiled.__call__` raises `constraint=scan_around_while` before a `jax.debug.callback` in the body runs.

```
PASS #1 gpu posthoc without resolve: gpu posthoc rejected before the body ran; resolve left ChunkedMap
```

### #8 cond-loop overrides — Resolved

Probe: `compile(...).plan_for` (not bare `resolve`) for a cond loop named `until`, threshold `jnp.max(carry) < 2.5`, init shape `(3,)`.

```
PASS #8 cond override via compile: diff=None reason=conditional loop: jax.lax.while_loop (forward-mode only)
```

`(WhileCarry(), Scan())` raises `IntentError` with `site="until"`, `path="diff"`, `constraint=repr(Scan())`. `StaticWhile(length=4)` raises with `path="primal"`. `(WhileCarry(), None)` plans `diff=None` with the forward-mode reason, and the call matches `jax.lax.while_loop`.

### #9 cond loops plan `diff=Scan()` — Resolved

Probe: body `carry + [0.4, -0.1]`, cond `carry[0] < 1.1`, init `[0.0, 0.9]`, label `shrink`. The fixer test uses `+ 1` and `sum < 3` on `zeros((2,))`.

```
PASS #9 cond plan vs while: reason=conditional loop: jax.lax.while_loop (forward-mode only) jaxpr_has_while=True
```

`plan_for` shows `WhileCarry` and `diff=None` with that reason. The compiled jaxpr contains `while[` and not `scan[`. `jax.jvp` matches `jax.lax.while_loop`. `jax.grad` raises JAX's `ValueError` mentioning `lax.while_loop`, not an `IntentError`.

### #12 post-hoc order and scope — Resolved

Probe: a legal `sin` map on shapes `(5, 2)` and `(7, 2)`; a rejected length-4 loop named `spin` on `(6, 3)`; then an override on `kept_map` (static while, length 2) beside `other_map` (cond while, limit `sum(x) + 0.5`). Compile's backend set includes `cpu` so the check actually traces. The fixer test uses `debug.callback` on `(8, 4)` / length 3 and names `left` / `right`.

```
PASS #12 posthoc cache and bypass: posthoc_calls=4 jaxprs=4 body_runs=0
```

The second call of the legal signature does not enter `_posthoc` and does not call `make_jaxpr` again. A new shape does. The rejected program raises `constraint=scan_around_while` before the body callback (`body_runs=0`). The second call raises the same exception object and does not trace again (`jaxpr` count stays at the first rejection). The `kept_map` override records `override:kept_map bypassed scan_around_while`, and the call still raises `scan_around_while` because `other_map` is not exempt.

### Item 5 partial diff availability — Resolved

Probe: two failing `pure_callback` arguments (`aux` and `extra`) plus a differentiable `weight`, rows `(4, 3)`. The fixer test names only `aux` on the `(4, 4)` `p4` function. A second function has both arguments failing. A third is cond-only, a fourth is map-only.

```
PASS item5 partial diff: loop_reason=diff path resolves to Scan; not differentiable w.r.t. aux (ValueError); not differentiable w.r.t. extra (ValueError) all_failed=aux: ValueError; extra: ValueError cond_diff=None cond_names_aux=False cond_reason=conditional loop: jax.lax.while_loop (forward-mode only) map_diff=Vmap() map_names_aux=False map_reason=same strategy as primal in the MVP
```

The static loop keeps `Scan`, names both failures, and `jax.grad` with respect to `weight` runs. When every inexact argument fails, `diff` is `None` and the reason is `aux: ValueError; extra: ValueError`. `plan_for` matches what runs for that loop.

The cond-only and map-only plans do not name `aux`. That is the minor issue below. It does not undo the loop case this item specified: the cond plan's `diff=None` is the strategy that runs.

## Regression checks

### Plan cost

One versus six inexact arguments on the same loop+map function (five of the six are `int32` in the one-argument case). Median of three `plan_for` calls, after a warmup plan:

```
PASS plan cost 1 vs 6: median_1=0.0181s median_6=0.0805s ratio=4.45
```

About 4.5× for 6× linearizations, both under a tenth of a second. Reasonable for a plan-once cache.

### Post-hoc cache

Covered by the #12 probe. A passing signature traces once. A rejected signature re-raises the cached `IntentError` and does not trace or execute again.

### No `while` under AD

`test_loop.py` passed, including `test_loop_transforms_match_scan_reference`. Independent probe: compiled static loop, length 5, init shape `(4,)`, `tanh(carry * 0.3 + 0.05)`.

```
PASS no while under grad: grad jaxpr has scan and no while
```

`jax.make_jaxpr(jax.grad(compiled))` contains `scan[` and not `while[`. The gradient matches `jax.lax.scan`.

### Fallback precedence

Covered by the #11 probe. `decision_for` returns the exact-key decision and does not consult the canonical twin when that decision exists, including when the exact decision's `diff` is `None` and the twin's strategy differs.

## New issue

### Minor — partial-diff note is dropped unless a loop has a diff strategy

`src/xtrax/intent/compile.py:584`

`_annotate_partial_diff` appends the failure note only when `decision.kind == "loop" and decision.diff is not None`. A cond loop stays on its forward-mode reason (`cond_names_aux=False`). A map-only plan keeps `diff=Vmap()` and `diff_reason="same strategy as primal in the MVP"` (`map_names_aux=False`). The static-loop case still names every failed argument. The strategies those other plans record are the ones that run; the failed argument is just absent from the plan text.

## Counts

12 Resolved (#1, #2, #3, #4, #5, #6, #8, #9, #11, #12, #13, item 5). 0 Not resolved. 1 Minor new issue (`compile.py:584`).
