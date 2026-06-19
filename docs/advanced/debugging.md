# Debugging

Practical notes for diagnosing JAX tracing issues, numeric failures, and docs drift in xtrax.

## Static shapes and recompilation

xtrax treats batch and tiling thresholds as **static** wherever possible. If you change
`AxisSpec.default_batch_size`, `DataModule.batch_size`, or sparse `nse_budget` between
steps, JAX may recompile. When performance suddenly drops:

1. Confirm batch shapes are stable across the training loop.
2. Check whether a tiling plan changed cardinality or strategy between runs.
3. Use `jax.live_arrays()` / device memory profilers for unexpected buffer growth.

See {doc}`../architecture` for the static-shape rationale.

## Numeric safety

For gradient explosions or silent NaNs:

```python
from xtrax import create_train_step

trainer = create_train_step(loss_fn=loss, optimizer=optimizer, safety=True)
```

With `safety=True`, backward passes run under `jax.experimental.checkify` and raise on
the host when non-finite values appear in gradients.

Utility helpers in `xtrax.safety` (`safe_norm`, `safe_reciprocal`) avoid divide-by-zero
artifacts in custom losses.

## Type and import hygiene

- Public APIs are annotated for static analysis; run `ty check src/` locally.
- Tests install a scoped beartype import hook via `tests/conftest.py`. Set
  `XTRAX_DISABLE_BEARTYPE=1` to opt out when debugging hook-related failures.
- Prefer canonical imports documented in {doc}`../api/overview` (for example
  `from xtrax.engine.io import BoundedCallbackHandler` rather than re-export paths
  when the docs call out a boundary).

## Common issues

| Symptom | Likely cause | What to check |
|---------|--------------|---------------|
| `DispatchRejected` from tiling | Calling dispatch on a `DedupGather` strategy directly | Let `BatchPlanner` own dedup axes |
| `distributed=True` data error | `init_dist()` not called | Call `xtrax.distributed.init_dist()` before iterators |
| Docs example import fails | Optional extra missing | Install `pip install xtrax[eda]` for EDA pages |
| Coverage gate red on PR | New public API without annotations | Run `just audit-added-types-diff` |

## Logging and callbacks

`Engine` callbacks run Python-side outside JIT. Use `on_step_end` for lightweight metric
logging; keep heavy I/O async-friendly via `BoundedCallbackHandler` (see
{doc}`../api/output-sinks`).
