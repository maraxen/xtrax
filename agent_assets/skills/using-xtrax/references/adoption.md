# Adopting xtrax in an existing library

Replace a local copy with the primitive below. Import paths are the ones the package actually exports.

## Symptom → primitive

| Local copy | xtrax |
|---|---|
| Hand-rolled `safe_map` / `lax.map` chunking | `chunked_map(fn, xs, batch_size=...)` in `xtrax.transforms.map`; strategy `ChunkedMap(batch_size=...)` in `xtrax.tiling` |
| Private length ladder | `BUCKET_LADDER` from `xtrax.export.rings`, then `select_bucket` / `bucketize` (`references/length-bucketing.md`) |
| Hand-rolled scan or while carry | `CarrySpec` (`xtrax.tiling.carry`). `collect_outputs=True` plans `Scan`. `collect_outputs=False` plans `WhileCarry` |
| Private device-budget / compile-memory math | `device_memory_budget` and `lowered_memory_estimate` in `xtrax.tiling.estimators` |
| Hand-rolled jax2onnx call | `convert_to_onnx` (`references/onnx-standalone.md`) |
| `np.unique` to build a `DedupSpec` | `synthesize_dedup_spec` (below) |

```python
import jax
import jax.numpy as jnp
from xtrax.tiling import lowered_memory_estimate

peak = lowered_memory_estimate(
    lambda x: x @ x,
    jax.ShapeDtypeStruct((8, 8), jnp.float32),
)
```

`device_memory_budget(fraction=0.9, device=None)` reads `device.memory_stats()["bytes_limit"]`. A device that does not report it raises `RuntimeError` (CPU does not). Pass an explicit `MemoryBudget` byte count in that case. `lowered_memory_estimate` lowers and compiles, then sums XLA's argument, output, and temp bytes. It raises `RuntimeError` when the backend returns no `memory_analysis()`.

Verify: `src/xtrax/transforms/map.py:8-33`, `src/xtrax/tiling/strategy.py:66-69`, `src/xtrax/tiling/carry.py:23-55`, `src/xtrax/tiling/estimators.py:27-97`.

## Alias lifetimes

`SafeMap`, `SafeMapIterator`, and `safe_map` existed for one release, 0.4.0a11, as `DeprecationWarning` aliases of `ChunkedMap`, `ChunkedMapIterator`, and `chunked_map`. They were removed in 0.4.0a12 and now raise `AttributeError` / `ImportError`.

Migrate with the ast-grep rules in `codemods/safemap-to-chunkedmap/` (`rules.yml` for Python, `rename_markdown.py` for Markdown). See that directory's README and the CHANGELOG `[0.4.0a12]` Removed entry.

Name-based strategy matching in `xtrax.stages.topology` and `xtrax.export.composer` still accepts the class name `"SafeMap"`, for a consumer's own duck-typed class of that name.

## What xtrax does not provide

| Need | What exists |
|---|---|
| An inference work-unit queue, or resume of a sampler / decoder from a cursor | Training resume only. `xtrax.cli.resume_verb.run_resume` loads an orbax checkpoint and calls `Engine.fit_sync(..., resume=True)` under ledger kind `KIND_TRAIN`. `xtrax.inference` is signature inference, CSE reports, and jaxpr memoization. No work-unit type lives under `src/xtrax/`. |
| A ledger kind for a free-standing sampling loop | Kinds are `KIND_TRAIN` (`Engine.fit`), `KIND_EVAL` (`Engine.evaluate`), and `KIND_EXPORT` (the export CLI). |

`Engine.evaluate` aggregates eval metrics. It is not a resumable inference runner.

## Telemetry

`Engine.fit` / `fit_sync` and `Engine.evaluate` open a ledger when `ledger` is omitted (`_resolve_ledger` → `RunLedger.open`). Opening fails closed: a row that cannot be written raises `LedgerUnavailableError` (`xtrax.telemetry`, a `RuntimeError`) and the run does not start.

`XTRAX_TELEMETRY_OPTOUT=1` still writes a row, with status `opted_out`, and the run is non-citable. Capture failures (git, IR export) degrade the row and record a reason; they do not abort the run.

```python
from xtrax.telemetry import LedgerUnavailableError, RunLedger
from xtrax.telemetry.record import KIND_TRAIN

assert issubclass(LedgerUnavailableError, RuntimeError)
# RunLedger.open(run_id, *, kind=KIND_TRAIN, root=None, derived_from=None, context=None)
```

Verify: `src/xtrax/telemetry/ledger.py:64-71` and `:205-214`, `src/xtrax/engine/engine.py:33-52` and `:88-109`, `src/xtrax/cli/resume_verb.py:107-118`.

## Inference-only loops

An autoregressive, sampling, or other inference-only loop uses the while-loop carry: `WhileCarry`, or `CarrySpec(..., collect_outputs=False, cond=...)`. Both compile to `jax.lax.while_loop` (one XLA `WhileOp`).

`Scan` / `CarrySpec(collect_outputs=True)` / `JaxScanIterator` compile to `jax.lax.scan`. Use those when you differentiate through the loop. `WhileCarry` has no reverse-mode VJP rule.

A 512-step autoregressive decode compiled as `lax.scan` took over 20 minutes of backend codegen on an L40S. The same loop as one `WhileOp` did not.

`fixed_step_count_cond(n)` treats the carry as a `(step_i, state)` tuple, or as an object with `.step_i`.

```python
import jax.numpy as jnp
from xtrax.tiling import (
    AxisSpec,
    BatchPlanner,
    CarrySpec,
    WhileCarry,
    axis_dispatch,
    fixed_step_count_cond,
)

def body(carry):
    step_i, total = carry
    return step_i + jnp.int32(1), total + step_i.astype(total.dtype)

init = (jnp.int32(0), jnp.float32(0.0))
cond = fixed_step_count_cond(4)
final = axis_dispatch(WhileCarry(body=body, cond=cond, init=init), None, None)

spec = AxisSpec(name="steps", cardinality=4, default_batch_size=4)
planned = BatchPlanner(carry_specs=[
    CarrySpec(axis_name="steps", init=init, transition=body, collect_outputs=False, cond=cond),
]).plan([spec])
assert isinstance(planned.decisions[0].strategy, WhileCarry)
```

`final` is `(step_i=4, total=6)` for steps 0..3. Verify: `src/xtrax/tiling/strategy.py:126-153`, `src/xtrax/tiling/carry.py:36-40`, `src/xtrax/tiling/dispatch.py:180-200`, `src/xtrax/tiling/plan.py:250-258`.

## ChunkedMap and an inner while_loop

`chunked_map` uses `jax.vmap` when `batch_size` is `None` or the leading axis is no longer than `batch_size`. Otherwise it calls `jax.lax.map(fn, xs, batch_size=batch_size)`, which is a scan. `ChunkedMapIterator` calls `chunked_map`. `axis_dispatch(ChunkedMap(...), ...)` does too.

A mapped function that itself calls `lax.while_loop` is a scan containing a while. On SM120 (Blackwell) that nest hung compile for an hour on a cluster job. Current xtrax has no transform that lifts the inner while out of `lax.map`.

Verify: `src/xtrax/transforms/map.py:28-33`, `src/xtrax/tiling/iterator.py:145-151`, `src/xtrax/tiling/dispatch.py:146-150`.

## Primitives consumers reimplement

### DedupSpec synthesis

`synthesize_dedup_spec` builds a byte-exact `DedupSpec`: each leaf becomes native bytes, and rows match on those bytes (`±0.0` stays distinct; bitwise-identical NaNs collapse). The spec's `axis_name` is always `"batch"`. A sample-stage duplication ratio below `threshold` (default `0.5`) returns `spec=None` (`"no_duplication"` or `"below_threshold"`). Exact `k > max_unique_k` (default `256`) returns `spec=None` with stage `"k_over_limit"`.

For an axis that is not named `"batch"`, copy the indices with `dataclasses.replace(spec, axis_name=...)`.

```python
from dataclasses import replace

import numpy as np
from xtrax.tiling.dedup_synthesis import synthesize_dedup_spec
from xtrax.tiling.plan import AxisSpec, BatchPlanner

rows = np.array(
    [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [3.0, 4.0], [1.0, 2.0]],
    dtype=np.float32,
)
result = synthesize_dedup_spec([rows])
spec = result.spec  # stage="synthesized", axis_name="batch", k=2
tokens = replace(spec, axis_name="tokens")
plan = BatchPlanner(dedup_specs=[spec]).plan([
    AxisSpec(name="batch", cardinality=6, default_batch_size=4, dedup_eligible=True),
])
```

`k_bucket` is not a `DedupSpec` field. `DedupSpec.to_dedup_gather` computes it with `get_k_bucket`. Verify: `src/xtrax/tiling/dedup_synthesis.py:219-289`.

### Loop-extent cost

`loop_bodies` and `extent_scaling_report` live in `xtrax.profiling.loop_scaling` (not re-exported from `xtrax.profiling`). Both trace with `make_jaxpr` and do not execute. `extent_scaling_report` traces at `extent` and `2 * extent` and flags a body whose per-iteration work ratio reaches `threshold` (default `1.5`).

```python
import jax
import jax.numpy as jnp
from xtrax.profiling.loop_scaling import extent_scaling_report, loop_bodies

def prog(xs):
    def body(c, x):
        return c + x, c
    return jax.lax.scan(body, jnp.float32(0), xs)

bodies = loop_bodies(prog, jnp.zeros((4,), dtype=jnp.float32))
report = extent_scaling_report(
    prog, lambda n: (jnp.zeros((n,), dtype=jnp.float32),), 4,
)
```

Verify: `src/xtrax/profiling/loop_scaling.py:181-218`.

### Gradient accumulation

`accumulate_grads` is `xtrax.training.grad.accumulate_grads` (not re-exported from `xtrax.training`). `microbatches` is a pre-stacked pytree whose leading axis is the microbatch count. It returns `(mean_grads, mean_loss)`.

```python
import jax.numpy as jnp
from xtrax.training.grad import accumulate_grads

params = {"w": jnp.ones((2,))}
micros = {"w": jnp.ones((3, 2))}
grads, mean_loss = accumulate_grads(
    lambda p, mb: jnp.sum((p["w"] - mb["w"]) ** 2),
    params,
    micros,
)
```

Verify: `src/xtrax/training/grad.py:11-16`.

### `adamw_with_schedule` defaults

```python
from xtrax.training.optim import adamw_with_schedule, no_bias_wd_mask
import inspect

sig = inspect.signature(adamw_with_schedule)
assert sig.parameters["wd_mask"].default is no_bias_wd_mask
assert sig.parameters["clip_norm"].default == 1.0
opt = adamw_with_schedule(1e-3, warmup_steps=10, total_steps=1000)
```

`no_bias_wd_mask` decays every leaf with `ndim != 1` and skips 1-D leaves (biases). `clip_norm=1.0` chains `optax.clip_by_global_norm(1.0)` in front of AdamW.

Both defaults move a step relative to `optax.adamw(lr, weight_decay=wd)` with a constant learning rate and no mask. Measured on CPU, one step at peak lr `1e-2`, `weight_decay=0.1`, `warmup_steps=0`, constant grad `10`: the 1-D bias moved by `-0.00999993` under the default mask and by `-0.01099993` when every leaf was decayed. With `weight_decay=0`, a uniform grad scale left the Adam step unchanged (the scale cancels), and a second step at a different global norm moved the weights to `-0.00975518` with `clip_norm=1.0` versus `-0.00732416` with `clip_norm=None`.

Verify: `src/xtrax/training/optim.py:10-22` and `:46-96`.

### Inference CSE and memo

`analyze_cse` and `memoize_jaxpr` are exported from `xtrax.inference`. `analyze_cse` traces once and reports duplicate jaxpr equations; it does not rewrite. `memoize_jaxpr` caches by leaf content. The wrapper has `memo_get_stats()`, `memo_reset()`, and `memo_rewrap()`.

```python
import jax
import jax.numpy as jnp
from xtrax.inference import analyze_cse, memoize_jaxpr

def dup(x):
    return jnp.sin(x) + jnp.sin(x)

report = analyze_cse(dup, [jax.ShapeDtypeStruct((4,), jnp.float32)])

@memoize_jaxpr
def score(x):
    return jnp.sum(x)

score(jnp.ones((4,)))
score(jnp.ones((4,)))
stats = score.memo_get_stats()
```

Verify: `src/xtrax/inference/cse.py:110-113`, `src/xtrax/inference/memo.py:1172-1196`.
