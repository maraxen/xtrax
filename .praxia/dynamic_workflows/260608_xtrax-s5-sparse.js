// Sprint 5 runner — hand-authored (dw emit blocked: templates not globally installed, backlog #983)
// task_id: 260608_xtrax-s5-sparse   sprint_id: 5
//
// Sprint 5 scope:
//   Track A (Phase 8):  Coverage completion — 88% → ≥95%
//   Track B (Phase 9):  Benchmarks — pytest-benchmark fixtures for hot paths
//   Track C (Phase 10): Sparse infrastructure — SparsePolicy, SparseMaskManager, SparseConfig
//
// RACE SAFETY: sequential tracks (A → B → C). No parallel fixer race risk.

export const meta = {
  name: "260608_xtrax-s5-sparse",
  description: "Coverage completion (≥95%), pytest-benchmark suite, SparsePolicy/SparseMaskManager with fixed-nse recompile safety.",
  phases: [
    { title: "Track A — Phase 8: Coverage completion (#1311)" },
    { title: "Track B — Phase 9: Benchmarks (#1312)" },
    { title: "Track C — Phase 10: Sparse infrastructure (#1313)" },
  ],
};

const TASK_ID = "260608_xtrax-s5-sparse";
const MAX_FIX_RETRIES = 2;

const VERDICT_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["item_id", "verdict", "summary"],
  properties: {
    item_id: { type: "string" },
    verdict: { type: "string", enum: ["PASS", "NEEDS_WORK", "FAIL"] },
    summary: { type: "string" },
    issues: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["where", "problem", "fix"],
        properties: {
          where: { type: "string" },
          problem: { type: "string" },
          fix: { type: "string" },
        },
      },
    },
  },
};

const EMITTER_CTX = `xtrax sprint 5. Sprints 1–4 complete. Version 0.2.0 on main.\nSpec: .praxia/docs/specs/260608_xtrax-s5-sparse.md (oracle R3 approved).\ntask_id: 260608_xtrax-s5-sparse\nKey rules: uv run pytest; ruff clean before commit. No Python-loop hot paths on data axes.\npytest-cov and pytest-benchmark are already in [dependency-groups] dev.\nbenchmarks/ is NOT in testpaths — run separately with --benchmark-only.\n`;

const fixer = (prompt, label, phaseName) =>
  agent(`${prompt}\n\nWhen done, end your message with 'verdict: done' on its own line.`, {
    agentType: "fixer",
    label,
    phase: phaseName,
  });

const reviewer = (itemId, prompt, label, phaseName) =>
  agent(prompt, { agentType: "reviewer", label, phase: phaseName, schema: VERDICT_SCHEMA });

async function track(itemId, phaseName, fixerPrompt, reviewerPrompt) {
  log(`[${itemId}] implement`);
  await fixer(fixerPrompt, `fix:${itemId}`, phaseName);
  let verdict = await reviewer(itemId, reviewerPrompt, `review:${itemId}`, phaseName);
  for (let retry = 0; retry < MAX_FIX_RETRIES && verdict && verdict.verdict === "NEEDS_WORK"; retry++) {
    log(`[${itemId}] NEEDS_WORK — repair cycle ${retry + 1}/${MAX_FIX_RETRIES}`);
    const issues = (verdict.issues || [])
      .map((i) => `- ${i.where}: ${i.problem} -> ${i.fix}`)
      .join("\n");
    await fixer(
      `${fixerPrompt}\n\nA reviewer found issues — fix exactly these, nothing else:\n${issues}`,
      `fix:${itemId}:repair:${retry}`,
      phaseName
    );
    verdict = await reviewer(itemId, reviewerPrompt, `review:${itemId}:re:${retry}`, phaseName);
  }
  return verdict;
}

// ===== TRACK A — Phase 8: Coverage completion (#1311) =====================================
const trackA = () =>
  track(
    "1311",
    "Track A — Phase 8: Coverage completion (#1311)",
    `task_id: 260608_xtrax-s5-sparse
Spec: .praxia/docs/specs/260608_xtrax-s5-sparse.md §3.1

Read before editing:
  src/xtrax/io/callbacks.py
  src/xtrax/distributed/sharding.py
  src/xtrax/stages/bundle.py
  src/xtrax/training/grad.py
  src/xtrax/training/types.py
  tests/engine/test_io.py
  tests/distributed/test_sharding.py
  tests/training/test_grad.py
  tests/stages/test_bundle.py

## Coverage gaps to fill

### 1. tests/engine/test_io.py — io/callbacks.py gaps (lines 39-83, 121-122)

PRODUCTION BUG — fix this before writing tests (exception to tests-only rule for this file only):
  callbacks.py line 63: \`item = await asyncio.to_thread(queue.get)\`
  asyncio.Queue.get is a coroutine method. asyncio.to_thread runs a sync callable in a
  thread and returns its return value — calling queue.get() in a thread returns a coroutine
  object, NOT an item. Fix: change line 63 to \`item = await queue.get()\`
  The rest of Track A is tests-only; only callbacks.py:63 requires a production fix.

All async tests use pytest-asyncio (asyncio_mode="auto" already configured in pyproject.toml).

- test_async_indexed_stream_normal: \`async for\` over async_indexed_stream([10, 20, 30])
  yields (0,10), (1,20), (2,30) in order.
- test_async_indexed_stream_buffer_size_1: buffer_size=1, list of 5 items, all yielded in order.
- test_async_indexed_stream_exception_propagates: iterable is a generator that yields 2 items
  then raises RuntimeError("oops"). Verify RuntimeError propagates from the async for loop.
- test_bounded_handler_submit_normal: submit a coroutine that sets a flag. wait_all(). Flag is True.
- test_bounded_handler_submit_exception_logged: submit a coroutine that raises ValueError.
  wait_all() does NOT raise. Use caplog to verify exception logged at ERROR level.
  (The except branch is at callbacks.py:121-122 inside bounded_coro().)
- test_bounded_handler_wait_all_empty: BoundedCallbackHandler().wait_all() with no tasks completes immediately.
- test_bounded_handler_max_concurrent_1: BoundedCallbackHandler(max_concurrent=1). Submit two
  coroutines. wait_all(). Both complete without deadlock.

### 2. tests/distributed/test_sharding.py — sharding.py gaps

- test_get_partition_spec_no_match: ShardingPolicy with rules matching "weight".
  get_partition_spec("bias") → PartitionSpec() (full replication fallback).
- test_apply_to_pytree: apply_to_pytree({"encoder.weight": ..., "bias": ...}) returns
  pytree with matching PartitionSpec values per path.
- test_empty_rules: ShardingPolicy with empty rules, any path → PartitionSpec().
- test_repr_no_raise: repr(ShardingPolicy(...)) does not raise.

### 3. tests/stages/test_bundle.py — bundle.py __init_subclass__ error paths

- test_invalid_non_optional_callable_raises: field annotated as \`int\` → TypeError at class def.
- test_optional_int_raises: field annotated \`Optional[int]\` → TypeError at class def.
- test_list_callable_raises: field annotated \`list[Callable]\` → TypeError (not Optional).
- test_optional_callable_ok: field annotated \`Optional[Callable]\` does NOT raise (regression guard).

### 4. tests/training/test_grad.py — grad.py gaps

Real signature: accumulate_grads(loss_fn, params, microbatches, filter_spec=None)
microbatches is a pre-stacked PyTree — there is NO microbatch_sizes parameter.

- test_unequal_second_axis_raises: microbatches dict where leaves have different second-axis
  sizes (e.g. {"a": jnp.ones((2,4,3)), "b": jnp.ones((2,5,3))}) → ValueError (grad.py:71-77).
- test_filter_spec_excludes_bias: filter_spec=lambda x: isinstance(x, jax.Array) and x.ndim > 1.
  Bias leaf in returned grads is zeros.
- test_single_microbatch_matches_direct: leading axis=1 → mean_grads matches
  eqx.filter_value_and_grad result (atol=1e-6).

### 5. Minor gaps (one targeted test each)

- tests/checkpoint/test_orbax.py: test_load_checkpoint_step_none — save two checkpoints,
  load with step=None → loads latest. Verify step field matches.
- tests/tiling/test_dispatch.py: test_scan_none_init_raises (ValueError for Scan + init=None);
  test_unknown_strategy_raises (TypeError for non-AxisStrategy object).
- tests/engine/test_engine.py: test_no_checkpoint_dir — Engine.fit with checkpoint_dir=None
  completes without error; no checkpoint written.
- tests/tiling/test_iterator.py: cover BucketIterator boundary and empty-batch edge cases
  per existing BucketIterator implementation.
- tests/distributed/test_init.py: test_coordinator_address_provided — init_dist with
  coordinator_address="localhost:1234", num_processes=1 → no error.

Gate:
  uv run pytest tests/ --cov=xtrax --cov-report=term-missing -q
    → total ≥95%; io/callbacks.py ≥90%; distributed/sharding.py ≥90%
    → all 321+ tests pass
  ruff check tests/ → clean

${EMITTER_CTX}`,
    `task_id: 260608_xtrax-s5-sparse. Verify Phase 8 coverage completion:
1. Run: uv run pytest tests/ --cov=xtrax --cov-report=term-missing -q
   → total ≥95%; io/callbacks.py individually ≥90%; distributed/sharding.py individually ≥90%
2. async_indexed_stream exception test exercises callbacks.py:70-71 (isinstance re-raise).
3. BoundedCallbackHandler exception test exercises callbacks.py:121-122 (logger.exception);
   caplog confirms ERROR-level log; wait_all() does not raise.
4. Production change allowed ONLY for callbacks.py:63 — change \`asyncio.to_thread(queue.get)\`
   to \`queue.get()\`. No other src/ changes.
5. All pre-existing 321 tests still pass (no regressions).
6. ruff check src/xtrax tests/ → no violations.
`
  );

// ===== TRACK B — Phase 9: Benchmarks (#1312) ==============================================
const trackB = () =>
  track(
    "1312",
    "Track B — Phase 9: Benchmarks (#1312)",
    `task_id: 260608_xtrax-s5-sparse
Spec: .praxia/docs/specs/260608_xtrax-s5-sparse.md §3.2

Read before editing:
  src/xtrax/training/trainer.py
  src/xtrax/training/types.py
  src/xtrax/training/grad.py
  src/xtrax/tiling/dispatch.py
  src/xtrax/tiling/strategy.py

## Create benchmarks/ directory and files

### benchmarks/__init__.py — empty file

### benchmarks/conftest.py

\`\`\`python
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from xtrax.training.trainer import Trainer
from xtrax.training.types import ResumableState


class _TinyMLP(eqx.Module):
    layers: list

    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.layers = [
            eqx.nn.Linear(64, 64, key=k1),
            eqx.nn.Linear(64, 1, key=k2),
        ]

    def __call__(self, x):
        # eqx.nn.Linear operates on rank-1 input — vmap over the batch axis.
        def _forward(xi):
            xi = jax.nn.tanh(self.layers[0](xi))
            return self.layers[1](xi)
        return jax.vmap(_forward)(x)


@pytest.fixture(scope="module")
def tiny_model():
    return _TinyMLP(jax.random.key(0))


@pytest.fixture(scope="module")
def synthetic_batch():
    k1, k2 = jax.random.split(jax.random.key(42))
    return {
        "inputs": jax.random.normal(k1, (32, 64)),
        "targets": jax.random.normal(k2, (32, 1)),
    }


@pytest.fixture(scope="module")
def trainer(tiny_model):
    def mse(pred, target):
        return jnp.mean((pred - target) ** 2)
    return Trainer(loss_fn=mse, optimizer=optax.adam(1e-3))


@pytest.fixture(scope="module")
def trainer_state(tiny_model, trainer):
    # Trainer has no init_state method — build ResumableState explicitly.
    opt_state = trainer.optimizer.init(eqx.filter(tiny_model, eqx.is_array))
    return ResumableState(
        step=jnp.int32(0),
        key=jax.random.key(0),
        model=tiny_model,
        opt_state=opt_state,
    )
\`\`\`

### benchmarks/bench_training_step.py

\`\`\`python
def test_trainer_step_throughput(benchmark, trainer, trainer_state, synthetic_batch):
    state = trainer_state
    for _ in range(3):  # JIT warmup
        state, _ = trainer.step(state, synthetic_batch)

    def one_step():
        return trainer.step(state, synthetic_batch)

    benchmark.pedantic(one_step, rounds=5, iterations=20)
\`\`\`

### benchmarks/bench_grad_accum.py

\`\`\`python
import jax.numpy as jnp
import pytest
from xtrax.training.grad import accumulate_grads


@pytest.mark.parametrize("n_microbatches", [1, 2, 4, 8])
def test_accumulate_grads_scaling(benchmark, n_microbatches, tiny_model, synthetic_batch):
    inputs = jnp.stack([synthetic_batch["inputs"]] * n_microbatches)
    targets = jnp.stack([synthetic_batch["targets"]] * n_microbatches)
    microbatches = {"inputs": inputs, "targets": targets}

    def loss_fn(params, batch):
        pred = params(batch["inputs"])
        return jnp.mean((pred - batch["targets"]) ** 2)

    accumulate_grads(loss_fn, tiny_model, microbatches)  # warmup

    benchmark(accumulate_grads, loss_fn, tiny_model, microbatches)
\`\`\`

### benchmarks/bench_tiling.py

\`\`\`python
import jax.numpy as jnp
import pytest
from xtrax.tiling.dispatch import make_axis_dispatch
from xtrax.tiling.strategy import DedupGather, SafeMap, Vmap

STRATEGIES = {
    "vmap": lambda: Vmap(),
    "safe_map": lambda: SafeMap(batch_size=8),
    "dedup": lambda: DedupGather(
        dedup_fn=lambda xs: (jnp.unique(xs, size=8, axis=0), jnp.zeros(32, dtype=jnp.int32)),
        gather_fn=lambda ys, idx: ys[idx],
        k_bucket=8,  # required third field per strategy.py:57
    ),
}


@pytest.mark.parametrize("strategy_name", ["vmap", "safe_map", "dedup"])
def test_tiling_dispatch_overhead(benchmark, strategy_name):
    strategy = STRATEGIES[strategy_name]()
    xs = jnp.ones((32, 4))
    fn = lambda x: x * 2.0
    benchmark(make_axis_dispatch, strategy, fn, xs)
\`\`\`

Gate:
  uv run pytest benchmarks/ --benchmark-only --benchmark-disable-gc -q → exits 0
  uv run pytest tests/ -q → still passes (no regressions)
  ruff check benchmarks/ → clean

${EMITTER_CTX}`,
    `task_id: 260608_xtrax-s5-sparse. Verify Phase 9 benchmarks:
1. uv run pytest benchmarks/ --benchmark-only --benchmark-disable-gc -q → exits 0;
   benchmarks collected: 1 step + 4 grad_accum + 3 tiling parametrize = 8 total.
2. conftest trainer_state uses explicit ResumableState — no trainer.init_state.
3. _TinyMLP.__call__ wraps forward in jax.vmap(_forward)(x) for batched input.
4. DedupGather includes k_bucket=8 (required third field per strategy.py:57).
5. bench_grad_accum microbatches are pre-stacked pytree (jnp.stack, leading axis = n_microbatches).
6. uv run pytest tests/ -q → 321+ tests pass (no regressions).
7. ruff check benchmarks/ → clean.
`
  );

// ===== TRACK C — Phase 10: Sparse infrastructure (#1313) ===================================
const trackC = () =>
  track(
    "1313",
    "Track C — Phase 10: Sparse infrastructure (#1313)",
    `task_id: 260608_xtrax-s5-sparse
Spec: .praxia/docs/specs/260608_xtrax-s5-sparse.md §3.3

Read before editing:
  src/xtrax/training/types.py   (PyTree, Array aliases)
  pyproject.toml                (jax>=0.4.36 confirmed — argwhere size= kwarg supported)

## Create src/xtrax/sparse/

### src/xtrax/sparse/__init__.py
\`\`\`python
from xtrax.sparse.config import SparseConfig
from xtrax.sparse.policy import SparsePolicy
from xtrax.sparse.manager import SparseMaskManager

__all__ = ["SparseConfig", "SparsePolicy", "SparseMaskManager"]
\`\`\`

### src/xtrax/sparse/config.py
\`\`\`python
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SparseConfig:
    nse_budget: int
    update_schedule: Callable[[int], bool]
    fallback_mode: Literal["dense_mask", "error"] = "dense_mask"
\`\`\`

### src/xtrax/sparse/policy.py
\`\`\`python
from __future__ import annotations
from typing import Any
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from xtrax.sparse.config import SparseConfig

Array = jax.Array
PyTree = Any


class SparsePolicy(eqx.Module):
    config: SparseConfig = eqx.field(static=True)

    def should_update(self, step: int) -> bool:
        return self.config.update_schedule(step)

    def make_mask(self, weights: Array, step: int) -> Array:
        # argsort-based: returns EXACTLY nse_budget True values (no threshold-tie overflow).
        flat = jnp.abs(weights.ravel())
        n = min(self.config.nse_budget, flat.size)
        sorted_idx = jnp.argsort(flat)[-n:]
        mask_flat = jnp.zeros(flat.size, dtype=jnp.bool_).at[sorted_idx].set(True)
        return mask_flat.reshape(weights.shape)

    def apply_mask(self, weights: Array, mask: Array) -> Array | BCOO:
        # Eager Python-side only — NOT safe under filter_jit/jax.jit.
        # n_true branches on a traced Array under jit → TracerBoolConversionError.
        # Callers must invoke apply_mask outside of jit.
        n_true = jnp.sum(mask)
        if self.config.fallback_mode == "error" and n_true > self.config.nse_budget:
            raise ValueError(
                f"True nonzeros {int(n_true)} exceeds nse_budget {self.config.nse_budget}"
            )
        if n_true > self.config.nse_budget:
            return weights * mask  # dense_mask fallback
        # Fixed-nse BCOO: argwhere on 2D mask → (nse_budget, 2) indices.
        # fill_value=0 means padded slots point to (0,0); zeroed via validity mask.
        indices = jnp.argwhere(mask, size=self.config.nse_budget, fill_value=0)
        valid = jnp.arange(self.config.nse_budget) < n_true
        data = weights[indices[:, 0], indices[:, 1]]
        data = jnp.where(valid, data, jnp.zeros_like(data[0]))
        return BCOO((data, indices), shape=weights.shape)
\`\`\`

### src/xtrax/sparse/manager.py
\`\`\`python
from __future__ import annotations
from collections.abc import Callable
from typing import Any
import jax
from xtrax.sparse.policy import SparsePolicy

PyTree = Any


class SparseMaskManager:
    """Python-side mutable mask tracker. NOT an eqx.Module."""

    def __init__(self, policy: SparsePolicy) -> None:
        self.policy = policy
        self._masks: dict[str, jax.Array] = {}
        self._initialized: bool = False

    def step(
        self,
        params: PyTree,
        step: int,
        path_filter: Callable[[str], bool] = lambda _: True,
    ) -> PyTree:
        should_update = not self._initialized or self.policy.should_update(step)
        if should_update:
            for path, leaf in jax.tree_util.tree_leaves_with_path(params):
                path_str = ".".join(
                    p.key if hasattr(p, "key") else str(p) for p in path
                )
                if path_filter(path_str) and hasattr(leaf, "ndim") and leaf.ndim >= 2:
                    self._masks[path_str] = self.policy.make_mask(leaf, step)
            self._initialized = True

        # Old masks applied to current (updated) params on no-update steps — by design.
        def apply_leaf(path, leaf):
            path_str = ".".join(
                p.key if hasattr(p, "key") else str(p) for p in path
            )
            if path_str in self._masks:
                return self.policy.apply_mask(leaf, self._masks[path_str])
            return leaf

        return jax.tree_util.tree_map_with_path(apply_leaf, params)

    def current_masks(self) -> dict[str, jax.Array]:
        return dict(self._masks)
\`\`\`

## Create tests/sparse/

### tests/sparse/__init__.py — empty file

### tests/sparse/test_config.py
\`\`\`python
import pytest
from xtrax.sparse.config import SparseConfig

class TestSparseConfig:
    def test_frozen(self):
        cfg = SparseConfig(nse_budget=4, update_schedule=lambda s: s % 2 == 0)
        with pytest.raises((AttributeError, TypeError)):
            cfg.nse_budget = 8

    def test_schedule(self):
        cfg = SparseConfig(nse_budget=4, update_schedule=lambda s: s % 2 == 0)
        assert cfg.update_schedule(0) is True
        assert cfg.update_schedule(1) is False

    def test_default_fallback_mode(self):
        cfg = SparseConfig(nse_budget=4, update_schedule=lambda s: True)
        assert cfg.fallback_mode == "dense_mask"
\`\`\`

### tests/sparse/test_policy.py
\`\`\`python
import jax.numpy as jnp
import pytest
from jax.experimental.sparse import BCOO
from xtrax.sparse.config import SparseConfig
from xtrax.sparse.policy import SparsePolicy


def _make_policy(budget, schedule=None, fallback="dense_mask"):
    return SparsePolicy(config=SparseConfig(
        nse_budget=budget,
        update_schedule=schedule or (lambda s: True),
        fallback_mode=fallback,
    ))


class TestSparsePolicy:
    def test_should_update_delegates(self):
        p = _make_policy(4, schedule=lambda s: s == 0)
        assert p.should_update(0) is True
        assert p.should_update(1) is False

    def test_make_mask_shape_and_dtype(self):
        p = _make_policy(budget=4)
        w = jnp.arange(9, dtype=jnp.float32).reshape(3, 3)
        mask = p.make_mask(w, step=0)
        assert mask.shape == w.shape
        assert mask.dtype == jnp.bool_

    def test_apply_mask_bcoo_nse(self):
        p = _make_policy(budget=4)
        w = jnp.ones((3, 3))
        mask = jnp.array([[True, True, False], [True, True, False], [False, False, False]])
        result = p.apply_mask(w, mask)
        assert isinstance(result, BCOO)
        assert result.nse == 4

    def test_apply_mask_todense_equivalence(self):
        p = _make_policy(budget=4)
        w = jnp.arange(1, 10, dtype=jnp.float32).reshape(3, 3)
        mask = jnp.array([[True, True, False], [True, True, False], [False, False, False]])
        result = p.apply_mask(w, mask)
        assert jnp.allclose(result.todense(), w * mask, atol=1e-6)

    def test_apply_mask_dense_fallback(self):
        p = _make_policy(budget=2, fallback="dense_mask")
        w = jnp.ones((3, 3))
        mask = jnp.array([[True, True, False], [True, True, False], [False, False, False]])
        result = p.apply_mask(w, mask)
        assert not isinstance(result, BCOO)
        assert jnp.allclose(result, w * mask, atol=1e-6)

    def test_apply_mask_error_fallback_raises(self):
        p = _make_policy(budget=2, fallback="error")
        w = jnp.ones((3, 3))
        mask = jnp.array([[True, True, False], [True, True, False], [False, False, False]])
        with pytest.raises(ValueError, match="nse_budget"):
            p.apply_mask(w, mask)
\`\`\`

### tests/sparse/test_manager.py
\`\`\`python
import jax.numpy as jnp
from xtrax.sparse.config import SparseConfig
from xtrax.sparse.manager import SparseMaskManager
from xtrax.sparse.policy import SparsePolicy


def _make_manager(budget=4, schedule=None):
    cfg = SparseConfig(nse_budget=budget, update_schedule=schedule or (lambda s: True))
    return SparseMaskManager(policy=SparsePolicy(config=cfg))


class TestSparseMaskManager:
    def test_first_call_always_masks(self):
        mgr = _make_manager(budget=4, schedule=lambda s: False)
        params = {"w": jnp.ones((3, 3))}
        mgr.step(params, step=0)
        assert mgr.current_masks() != {}

    def test_no_update_reuses_masks(self):
        mgr = _make_manager(budget=4, schedule=lambda s: s == 0)
        params = {"w": jnp.arange(9, dtype=jnp.float32).reshape(3, 3)}
        mgr.step(params, step=0)
        keys_after_0 = set(mgr.current_masks())
        mgr.step(params, step=1)
        assert set(mgr.current_masks()) == keys_after_0

    def test_update_step_recomputes(self):
        mgr = _make_manager(budget=4, schedule=lambda s: s % 2 == 0)
        params = {"w": jnp.ones((3, 3))}
        mgr.step(params, step=0)
        mgr.step(params, step=1)
        mgr.step(params, step=2)
        assert mgr._initialized

    def test_path_filter_excludes(self):
        mgr = _make_manager(budget=4)
        params = {"weight": jnp.ones((3, 3)), "bias": jnp.ones((3, 3))}
        mgr.step(params, step=0, path_filter=lambda p: "weight" in p)
        masks = mgr.current_masks()
        assert any("weight" in k for k in masks)
        assert not any("bias" in k for k in masks)

    def test_current_masks_is_copy(self):
        mgr = _make_manager(budget=4)
        params = {"w": jnp.ones((3, 3))}
        mgr.step(params, step=0)
        assert mgr.current_masks() is not mgr.current_masks()
\`\`\`

Gate:
  uv run pytest tests/sparse/ -v → all pass
  uv run pytest tests/ -v → full suite passes (no regressions)
  uv run pytest tests/ --cov=xtrax --cov-report=term-missing -q → src/xtrax/sparse/ ≥95%
  ruff check src/xtrax/sparse/ tests/sparse/ → clean

${EMITTER_CTX}`,
    `task_id: 260608_xtrax-s5-sparse. Verify Phase 10 sparse infrastructure:
1. SparseConfig is a frozen dataclass; mutating nse_budget raises (AttributeError or TypeError).
2. SparsePolicy is eqx.Module with config marked eqx.field(static=True).
   apply_mask is EAGER PYTHON-SIDE ONLY — not JIT-safe due to data-dependent branch on traced n_true.
   Do not add JIT-safety requirements or tests.
3. make_mask returns EXACTLY nse_budget True values (argsort-based, no threshold-tie overflow).
   test_make_mask_shape_and_dtype: shape==(3,3), dtype==jnp.bool_, exactly 4 True for budget=4.
4. apply_mask with count≤budget: BCOO with nse==nse_budget; .todense() matches weights*mask (atol=1e-6).
5. apply_mask with count>budget, fallback=dense_mask: returns dense Array equal to weights*mask.
6. apply_mask with count>budget, fallback=error: raises ValueError containing "nse_budget".
7. BCOO construction uses jnp.argwhere(mask, size=nse_budget, fill_value=0) on 2D mask
   (not raveled); indices shape (nse_budget, 2); padded slots have zeroed data.
8. SparseMaskManager.step() first-call always masks regardless of schedule.
9. No-update steps reuse old masks applied to current (updated) params — by design.
10. path_filter correctly excludes specified paths from masking.
11. uv run pytest tests/sparse/ -v → all tests pass.
12. uv run pytest tests/ -v → 321+ tests pass, no regressions.
13. src/xtrax/sparse/ coverage ≥95%; ruff check src/xtrax/sparse/ tests/sparse/ → clean.
`
  );

log("xtrax Sprint 5: Coverage + Benchmarks + Sparse — Tracks A → B → C (sequential)");
const a = await trackA();
log(`[Track A] verdict: ${a && a.verdict}`);
const b = await trackB();
log(`[Track B] verdict: ${b && b.verdict}`);
const c = await trackC();
log(`[Track C] verdict: ${c && c.verdict}`);

return {
  task_id: TASK_ID,
  sprint_id: 5,
  verdicts: { "1311": a, "1312": b, "1313": c },
};
