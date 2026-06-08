---
title: xtrax Sprint 5 — Coverage, Benchmarks, and Sparse Infrastructure
task_id: 260608_xtrax-s5-sparse
date: 260608
status: draft-r3
sprint_id: 5
depends_on: 260604_xtrax-spec
---

# Specification: xtrax Sprint 5

## 1. Purpose and Scope

Sprint 5 adds three capabilities:

**Phase 8 — Coverage Completion**: Bring test coverage from 88% to ≥95% by filling gaps in
`io/callbacks.py` (40%), `distributed/sharding.py` (67%), `stages/bundle.py` (80%),
`training/grad.py` (81%), and smaller gaps in five other modules. No new production code —
tests only.

**Phase 9 — Benchmarks**: Add a `benchmarks/` directory with pytest-benchmark fixtures for
xtrax's hot paths: training step throughput, gradient accumulation scaling, and tiling dispatch
overhead. Benchmarks must be runnable on CPU (no GPU required) and produce repeatable numbers.

**Phase 10 — Sparse Infrastructure**: Add `xtrax.sparse` — a `SparsePolicy` module and
`SparseMaskManager` that let callers apply structured sparsification with a fixed-nse budget
(preventing XLA retraces), step-scheduled mask updates, and a dense-fallback policy for
growth phases. Delegates actual sparse matmul to `jax.experimental.sparse`; owns only mask
management and recompile safety.

### What Sprint 5 is NOT

- Not a custom sparse GEMM implementation (delegate to JAX/cuSPARSE).
- Not a structured 2:4 sparsity implementation (requires Triton kernels; deferred).
- Not a sparse-aware Optax optimizer (deferred).
- No production code changes to existing modules for coverage — coverage is a test-only concern.

---

## 2. Package Layout Additions

```
src/xtrax/
  sparse/
    __init__.py          # exports: SparsePolicy, SparseMaskManager, SparseConfig
    policy.py            # SparsePolicy(eqx.Module)
    manager.py           # SparseMaskManager
    config.py            # SparseConfig dataclass (nse_budget, schedule, fallback_mode)

benchmarks/
  __init__.py
  conftest.py            # shared fixtures (tiny model, synthetic batch)
  bench_training_step.py # Trainer.step throughput
  bench_grad_accum.py    # accumulate_grads scaling (1, 2, 4, 8 microbatches)
  bench_tiling.py        # make_axis_dispatch overhead across strategies

tests/
  sparse/
    __init__.py
    test_policy.py
    test_manager.py
    test_config.py
```

---

## 3. Module Contracts

### 3.1 Phase 8: Coverage Targets

The following modules need additional tests. No production code changes.

#### `io/callbacks.py` (current: 40%)

`src/xtrax/io/callbacks.py` implements two things only: `async_indexed_stream` (an
async generator that prefetches from a blocking iterable via `asyncio.to_thread + Queue`)
and `BoundedCallbackHandler` (semaphore-bounded concurrent coroutine executor). There are
no concrete callback classes in this file. The `Callback` protocol lives in
`src/xtrax/training/types.py`.

The uncovered lines are:
- Lines 39-83: the entire body of `async_indexed_stream` (the producer task, the consumer
  loop, exception re-raise, and finally cleanup). The existing test file imports the module
  but does not exercise the generator body.
- Lines 121-122: the exception-logging branch inside `BoundedCallbackHandler.submit` —
  the `except Exception` handler that logs and suppresses the error.

Tests must cover (all async, use `pytest-asyncio`):

- `async_indexed_stream` with a normal list iterable → yields `(0, item0)`, `(1, item1)`, ...
  in order with correct indices
- `async_indexed_stream` with `buffer_size=1` still yields all items in order (tests queue
  backpressure)
- `async_indexed_stream` with an iterable that raises `RuntimeError` mid-way → the exception
  propagates from the generator (caller sees the RuntimeError on the next `async for` step)
- `BoundedCallbackHandler.submit` with a normal coroutine → `wait_all()` completes without error
- `BoundedCallbackHandler.submit` with a coroutine that raises `ValueError` → exception is
  swallowed (not propagated from `wait_all()`); verify via `caplog` that the error is logged
  at exception level
- `BoundedCallbackHandler.wait_all()` with no submitted tasks → returns immediately (no hang)
- `BoundedCallbackHandler(max_concurrent=1)` with two submitted coroutines → both complete
  (semaphore does not deadlock at concurrency=1)

#### `distributed/sharding.py` (current: 67%)

Uncovered: `apply_to_pytree`, `get_partition_spec` with no matching rule, `ShardingPolicy`
`__repr__`.

Tests must cover:

- `ShardingPolicy.get_partition_spec(path)` with no matching rule → returns `PartitionSpec()`
  (full replication fallback)
- `ShardingPolicy.apply_to_pytree(pytree)` with a simple dict pytree → returns a pytree of
  `PartitionSpec` objects with the same structure
- `ShardingPolicy` with empty `rules=()` → all paths get `PartitionSpec()`
- `repr(ShardingPolicy(...))` does not raise

#### `stages/bundle.py` (current: 80%)

Uncovered: `__init_subclass__` error paths for non-`Optional[Callable]` field types.

Tests must cover:

- Non-callable annotated field raises `TypeError` at class definition time (not at
  instantiation)
- `Optional[int]` annotated field raises `TypeError`
- `Optional[Callable]` annotated field does NOT raise
- `list[Callable]` annotated field raises `TypeError` (not Optional)

#### `training/grad.py` (current: 81%)

Uncovered: gradient accumulation with mismatched second-axis sizes, `filter_spec` propagation,
and the single-microbatch degenerate case.

Real signature: `accumulate_grads(loss_fn, params, microbatches, filter_spec=None)`
where `microbatches` is a **pre-stacked PyTree** whose leading axis = num_microbatches and
second axis = microbatch size. There is no `microbatch_sizes` parameter.

Tests must cover:

- `accumulate_grads` with `microbatches` whose leaves have unequal second-axis sizes
  (e.g., `(inputs_shape=(4,3), targets_shape=(5,3))` passed as a tuple/dict) → raises
  `ValueError`. Mirror the pattern in existing `test_grad.py:101-122`:
  ```python
  microbatches = (
      {"x": jnp.ones((4, 3)), "y": jnp.ones((4, 1))},   # leading=4, second=3/1
      {"x": jnp.ones((5, 3)), "y": jnp.ones((5, 1))},   # leading=5 != 4 → ValueError
  )
  # But microbatches must be a single pre-stacked pytree, not a list of dicts.
  # Correct pattern: a dict where different leaves have different second-axis sizes:
  x = {"a": jnp.ones((2, 4, 3)), "b": jnp.ones((2, 5, 3))}  # second axes 4 vs 5
  ```
  Use the mismatched-second-axis construction from `test_grad.py:101-122` as the template.
- `accumulate_grads` with custom `filter_spec=lambda x: x.ndim > 1` (excludes 1-D bias)
  → bias leaf in mean_grads is a zero array (not differentiable under filter_spec)
- `accumulate_grads` with `microbatches` having leading axis=1 (single microbatch) →
  `mean_grads` matches the single-batch gradient from `eqx.filter_value_and_grad` exactly

#### Minor gaps (target ≥95% per module)

- `checkpoint/orbax.py` line 94: cover the `step=None` → auto-latest path in `load_checkpoint`
- `tiling/dispatch.py` lines 51-54: cover the `Scan + init=None` rejection branch in
  `make_axis_dispatch` (raises `ValueError`); and line 63-64: the unknown-strategy `TypeError`
  branch (pass an object that is not a recognized `AxisStrategy`)
- `engine/engine.py` lines 82-84: cover the `checkpoint_dir=None` → no checkpoint branch
- `engine/io.py` line 111-115: cover the exception-logging branch in `BoundedCallbackHandler`
- `tiling/iterator.py` lines 87, 132: cover `BucketIterator` boundary and empty-batch paths
- `distributed/init.py` branches 75-96: cover coordinator-address-provided path

---

### 3.2 Phase 9: Benchmarks

All benchmarks use `pytest-benchmark` fixtures. Must be runnable with:

```bash
uv run pytest benchmarks/ --benchmark-only --benchmark-disable-gc
```

Benchmarks are NOT run in the default `uv run pytest tests/` invocation (separate directory).

#### `benchmarks/conftest.py`

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest
from xtrax.training.trainer import Trainer
from xtrax.training.types import ResumableState

@pytest.fixture
def tiny_model() -> eqx.Module:
    """2-layer MLP: Linear(64→64) + tanh + Linear(64→1). CPU-runnable. key=0."""

@pytest.fixture
def synthetic_batch() -> dict:
    """{"inputs": (32, 64) float32, "targets": (32, 1) float32} — fixed seed."""

@pytest.fixture
def trainer(tiny_model) -> Trainer:
    """Trainer with MSE loss (jnp.mean((pred - target)**2)) and Adam(1e-3)."""

@pytest.fixture
def trainer_state(tiny_model, trainer) -> ResumableState:
    """
    ResumableState with step=0, fresh key, tiny_model, and initialized opt_state.
    Constructed as:
        opt = optax.adam(1e-3)
        opt_state = opt.init(eqx.filter(tiny_model, eqx.is_array))
        return ResumableState(
            step=jnp.int32(0),
            key=jax.random.key(0),
            model=tiny_model,
            opt_state=opt_state,
        )
    """
```

Note: `Trainer` has no `init_state` method. State must be constructed explicitly as above.

#### `bench_training_step.py`

```python
def test_trainer_step_throughput(benchmark, trainer, trainer_state, synthetic_batch):
    """Measures Trainer.step throughput: steps/second after JIT warmup."""
    # Warmup: 3 steps outside benchmark loop to trigger JIT compilation
    state = trainer_state
    for _ in range(3):
        state, _ = trainer.step(state, synthetic_batch)
    # benchmark.pedantic: rounds=5, iterations=20
    # Reports: mean wall time per step
    def one_step():
        return trainer.step(state, synthetic_batch)
    benchmark.pedantic(one_step, rounds=5, iterations=20)
```

Metric: steps per second. Baseline: recorded on first run, stored as `.benchmark_baseline.json`
in project root (gitignored). Regression alert threshold: >20% slowdown vs baseline.

#### `bench_grad_accum.py`

```python
@pytest.mark.parametrize("n_microbatches", [1, 2, 4, 8])
def test_accumulate_grads_scaling(benchmark, n_microbatches, trainer, synthetic_batch):
    """Measures accumulate_grads wall time as microbatch count scales."""
```

Expected: linear scaling (not superlinear). Report scaling factor vs n_microbatches=1.

#### `bench_tiling.py`

```python
from xtrax.tiling.strategy import Vmap, SafeMap, DedupGather
from xtrax.tiling.dispatch import make_axis_dispatch

STRATEGIES = {
    "vmap": lambda: Vmap(),
    "safe_map": lambda: SafeMap(batch_size=8),
    "dedup": lambda: DedupGather(
        dedup_fn=lambda xs: (jnp.unique(xs, size=8, axis=0), jnp.zeros(32, dtype=jnp.int32)),
        gather_fn=lambda ys, idx: ys[idx],
        k_bucket=8,  # required field: max unique elements, power-of-2 padded
    ),
}
# DedupGather requires three fields: dedup_fn, gather_fn, k_bucket (int).
# Omitting k_bucket raises TypeError at dataclass construction.

@pytest.mark.parametrize("strategy_name", ["vmap", "safe_map", "dedup"])
def test_tiling_dispatch_overhead(benchmark, strategy_name):
    """Measures make_axis_dispatch setup overhead and per-call cost.
    strategy_name maps to a constructed AxisStrategy instance via STRATEGIES dict."""
    strategy = STRATEGIES[strategy_name]()
    xs = jnp.ones((32, 4))
    fn = lambda x: x * 2.0
    benchmark(make_axis_dispatch, strategy, fn, xs)
```

---

### 3.3 Phase 10: Sparse Infrastructure

#### `xtrax.sparse.config` — `SparseConfig`

```python
@dataclass(frozen=True)
class SparseConfig:
    nse_budget: int
    update_schedule: Callable[[int], bool]
    fallback_mode: Literal["dense_mask", "error"] = "dense_mask"
```

- `nse_budget`: maximum number of stored elements per sparse layer. Fixed at construction.
  Passed to `BCOO` as static `nse` — never changes after construction.
- `update_schedule`: callable from step → bool. Returns `True` on steps when the mask should
  be recomputed. If `False`, the existing mask is reused (no retrace).
- `fallback_mode`:
  - `"dense_mask"`: when true nonzeros exceed `nse_budget`, apply mask to dense tensor
    (no sparse format used). Correct but slower.
  - `"error"`: raise `ValueError` if true nonzeros exceed `nse_budget`.

#### `xtrax.sparse.policy` — `SparsePolicy`

```python
class SparsePolicy(eqx.Module):
    config: SparseConfig = eqx.field(static=True)

    def should_update(self, step: int) -> bool:
        """True if mask should be recomputed at this step. Static call — no trace."""
        return self.config.update_schedule(step)

    def make_mask(self, weights: Array, step: int) -> Array:
        """
        Compute a boolean mask selecting the top-nse_budget absolute values of `weights`.
        Returns a boolean Array of same shape as `weights`.
        Does NOT produce a BCOO array — that is SparseMaskManager's job.
        """

    def apply_mask(self, weights: Array, mask: Array) -> Array:
        """
        Apply mask to weights.
        If jnp.sum(mask) <= nse_budget: convert to BCOO with nse=nse_budget (padded).
        If jnp.sum(mask) > nse_budget and fallback_mode="dense_mask": return weights * mask.
        If jnp.sum(mask) > nse_budget and fallback_mode="error": raise ValueError.
        """
```

**JAX tracing contract**: `SparsePolicy` is `eqx.field(static=True)` when stored in a module.
`should_update` is always called Python-side (not inside jit). `make_mask` and `apply_mask`
ARE safe to call inside `filter_jit` when `nse_budget` is a static integer (it always is).

#### `xtrax.sparse.manager` — `SparseMaskManager`

```python
class SparseMaskManager:
    """
    NOT an eqx.Module — lives Python-side, manages masks across training steps.
    Holds a dict: layer_path -> current_mask (boolean Array).
    """

    def __init__(self, policy: SparsePolicy):
        self.policy = policy
        self._masks: dict[str, Array] = {}

    def step(
        self,
        params: PyTree,
        step: int,
        path_filter: Callable[[str], bool] = lambda _: True,
    ) -> PyTree:
        """
        If policy.should_update(step): recompute masks for all paths matching path_filter.
        Apply current masks to params via policy.apply_mask.
        Returns masked params pytree (same structure as input).
        On first call: always computes masks regardless of schedule.

        On no-update steps (schedule returns False): the previous mask is intentionally
        applied to the current (updated) params. Masks lag params by design until the
        next scheduled update — this is correct behavior, not a bug.
        """

    def current_masks(self) -> dict[str, Array]:
        """Returns a copy of the current mask dict."""
```

**Design decision**: `SparseMaskManager` is NOT an `eqx.Module` because masks are mutable
Python state that changes on a schedule. Making it an `eqx.Module` would require functional
update patterns incompatible with its role as a Python-side step controller.

#### Fixed-nse padding contract

When `apply_mask` converts a boolean mask to BCOO for a 2-D weight matrix of shape
`(rows, cols)`:

```python
from jax.experimental.sparse import BCOO

# argwhere on the 2D mask → indices of shape (nse_budget, 2)
# fill_value=0 means padded slots point to index (0, 0) — a valid position.
# Use a validity mask to zero those slots' data.
indices = jnp.argwhere(mask, size=config.nse_budget, fill_value=0)  # (nse_budget, 2)

# Count true nonzeros to identify which slots are real vs padded.
n_true = jnp.sum(mask)  # scalar int

# Mark slots beyond n_true as padded.
valid = jnp.arange(config.nse_budget) < n_true  # (nse_budget,) boolean

# Gather data at the (possibly padded) indices; zero out padded positions.
data = weights[indices[:, 0], indices[:, 1]]  # (nse_budget,) float
data = jnp.where(valid, data, jnp.zeros_like(data[0]))  # zero padded slots

# Build BCOO — nse is always nse_budget regardless of true sparsity.
bcoo = BCOO((data, indices), shape=weights.shape)
```

`jnp.argwhere(..., size=nse_budget)` — the `size` kwarg makes output shape static
(`nse_budget` rows), enabling JIT-compatibility. The data-zeroing for padded slots ensures
the BCOO array is numerically equivalent to `weights * mask` even though dummy index (0,0)
appears in the index array. The resulting `BCOO` always has exactly `nse_budget` stored
elements, so XLA sees a constant-shape problem regardless of true sparsity changes.

This complete recipe (argwhere → gather → zero-pad → BCOO) is the core recompile-safety
mechanism. The fixer must implement it exactly as specified — no variations.

#### Tests for Phase 10 (`tests/sparse/`)

**`test_config.py`**:
- `SparseConfig(nse_budget=4, update_schedule=lambda s: s % 2 == 0)` is frozen (immutable)
- `update_schedule(0)` → True; `update_schedule(1)` → False
- `fallback_mode` defaults to `"dense_mask"`

**`test_policy.py`**:
- `SparsePolicy.should_update(step=0)` calls through to `config.update_schedule`
- `make_mask(weights, step)` returns boolean Array, same shape as weights, with exactly
  `nse_budget` True values (or fewer if weights has fewer nonzeros)
- `apply_mask` with count ≤ budget: returns BCOO with `nse == nse_budget`, padded correctly
- `apply_mask` with count > budget and `fallback_mode="dense_mask"`: returns dense Array
- `apply_mask` with count > budget and `fallback_mode="error"`: raises `ValueError`
- BCOO output of `apply_mask` has same dense equivalent (via `.todense()`) as `weights * mask`

**`test_manager.py`**:
- First call to `step(params, step=0)` always recomputes masks
- `step(params, step=1)` with `update_schedule=lambda s: s % 2 == 0` → reuses step-0 masks
- `step(params, step=2)` → recomputes masks
- `current_masks()` returns the correct mask dict after `step()`
- `path_filter` excludes specified paths from masking
- Manager works with nested pytree params (dict of dicts)

---

## 4. Implementation Constraints

- All new code: ruff clean, no bare `python` (use `uv run python`).
- No Python-loop hot paths in `sparse/` on data axes.
- `jnp.argwhere(..., size=nse_budget)` MUST use the `size` kwarg — omitting it makes output
  shape data-dependent and triggers retrace.
- `SparseMaskManager.step()` MUST NOT be called inside `filter_jit`. It is Python-side only.
- `SparsePolicy.make_mask` and `apply_mask` MAY be called inside `filter_jit` (they are
  pure functions of static-nse inputs).
- Tests for Phase 10 must run on CPU with `jax.devices("cpu")` — no GPU required.
- Benchmarks in `benchmarks/` are NOT in `tests/` and NOT included in the default pytest run.

---

## 5. Phase Gates

**Phase 8 gate**: `uv run pytest tests/ --cov=xtrax --cov-report=term-missing` → total ≥95%;
each previously-gapped module individually ≥90%. ruff clean on all new test files.

**Phase 9 gate**: `uv run pytest benchmarks/ --benchmark-only --benchmark-disable-gc` runs
without error. All three benchmark files produce output. No assertion failures.

**Phase 10 gate**: `uv run pytest tests/sparse/ -v` all pass. `uv run pytest tests/ -v` full
suite still passes (no regressions). ruff clean on `src/xtrax/sparse/` and `tests/sparse/`.
Coverage on `src/xtrax/sparse/` ≥95%.

**Sprint gate**: all three phase gates pass simultaneously.

---

## 6. Acceptance Criteria

- `SparseConfig(nse_budget=10, update_schedule=lambda s: s % 5 == 0)` — `update_schedule`
  returns True at step 0, 5, 10; False at 1, 2, 3, 4, 6.
- `SparsePolicy.make_mask(weights_10x10, step=0)` with `nse_budget=20` → exactly 20 True values.
- `SparsePolicy.apply_mask(weights, mask_count_15)` with `nse_budget=20` → BCOO with nse=20;
  `.todense()` matches `weights * mask` within float32 precision.
- `SparsePolicy.apply_mask(weights, mask_count_25)` with `nse_budget=20, fallback="dense_mask"`
  → returns `weights * mask` as dense Array (not BCOO).
- `SparseMaskManager.step(params, step=0)` always applies masks regardless of schedule.
- `SparseMaskManager.step(params, step=1)` with `schedule=lambda s: s==0` → same masks as step 0.
- Coverage gate: `io/callbacks.py` ≥90%, `distributed/sharding.py` ≥90%, total ≥95%.
- `uv run pytest benchmarks/ --benchmark-only` completes without error.

---

## 7. Risk Table

| Risk | Severity | Mitigation |
|------|----------|------------|
| `jnp.argwhere(size=...)` behavior change across JAX versions | Medium | Pin JAX ≥0.4.36; test `nse` shape invariant explicitly |
| BCOO `fill_value=-1` dummy indices corrupt sparse matmul | Medium | Only `apply_mask` uses BCOO; caller must zero out dummy-indexed positions or treat output as mask-applied dense. Document as contract. |
| `SparseMaskManager` called inside jit (misuse) | Medium | Document "Python-side only" contract clearly. No runtime guard — `jax.core.cur_sublevel()` is a private internal API and unreliable across JAX versions. Misuse will manifest naturally: passing a traced array (JAX `Tracer`) to `jnp.sum(mask)` inside jit raises a `ConcretizationTypeError` with a clear message. |
| benchmark results non-reproducible on CI (no GPU, JIT variance) | Low | `--benchmark-disable-gc`, `--benchmark-warmup`, fixed seeds; report relative ratios not absolute |
| `io/callbacks.py` test requires running Engine.fit | Medium | Unit-test callbacks in isolation using mock trainer stub; do not depend on full Engine integration |

---

## 8. Open Questions

1. Should `SparseMaskManager` support checkpointing its mask state (for resumable sparse training)?
   — Deferred. Masks can be recomputed from params; checkpointing them is an optimization, not
   a correctness requirement.
2. Should `apply_mask` return a BCOO that is directly usable in `bcoo_dot_general`?
   — Yes for the non-fallback path. But callers must ensure their matmul uses the sparse path.
   xtrax does not wrap matmul calls — this remains the caller's responsibility.
3. Benchmark regression thresholds — hardcoded 20% or configurable?
   — Hardcoded for now. Configurable via `pyproject.toml [tool.benchmark]` in a future sprint.

---

## 9. References

- `jax.experimental.sparse` docs: https://jax.readthedocs.io/en/latest/jax.experimental.sparse.html
- `jnp.argwhere(size=...)`: JAX functional `argwhere` with static output size
- `pytest-benchmark`: https://pytest-benchmark.readthedocs.io/
- deep-research findings: session 47c9f585 (June 2026)
- Preceding spec: `.praxia/docs/specs/260604_xtrax-spec.md`
- task_id: `260608_xtrax-s5-sparse`
