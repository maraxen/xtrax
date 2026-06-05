// Sprint 2 runner — emitted by `praxia dw emit-sprint`, oracle-revised R2
// Source: .praxia/sprint_plans/sprint_plan.toml
// task_id: 260605_xtrax-s2-core   sprint_id: 2
//
// RACE SAFETY: writing chain (A→B→C→D→E→F→G→H→I→J→K→L) is STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time.
//
// ORACLE REVISION R2 (260605): Fixed C1 (SafetyTrainStep recompile), C2 (SafetyManager
//   eqx.Module contract + field names), C3 (Trainer dict return), I2 (optimizer.update
//   eqx.filter), I4 (WeightedLoss.weight static), I6 (safe_norm/reciprocal formula),
//   I7 (Callback on_resume + on_step_end), I3 (DataModule eqx.Module), M3 (weight_decay
//   default, partition_labels), M1 (file-path deviation note), PreemptionHandler SIGTERM.
//   Added dependency guards: I/J/K/L skip if upstream tracks fail.

export const meta = {
  name: "260605_xtrax-s2-core",
  description: "Safe ops, SafetyManager, PreemptionHandler, StageBundle protocols, ResumableState, Trainer, accumulate_grads, loss combinators, optimizer utilities, DataModule",
  phases: [
    { title: "Track A — Phase 3.1: safe_norm, safe_reciprocal (#1136)" },
    { title: "Track B — Phase 3.3: PreemptionHandler (#1138)" },
    { title: "Track C — Phase 3.4: Stage protocols — TransformFn, RollingFn, FuseFn (#1139)" },
    { title: "Track D — Phase 4.1: ResumableState, LossFunction, Callback (#1141)" },
    { title: "Track E — Phase 4.5: Optimizer utilities (#1145)" },
    { title: "Track F — Phase 4.7: DataModule + streaming (#1147)" },
    { title: "Track G — Phase 3.2: SafetyManager, with_safety (#1137)" },
    { title: "Track H — Phase 3.5: StageBundle (#1140)" },
    { title: "Track I — Phase 4.2: Trainer (#1142)" },
    { title: "Track J — Phase 4.4: Loss combinators (#1144)" },
    { title: "Track K — Phase 4.6: accumulate_grads (#1146)" },
    { title: "Track L — Phase 4.3: SafetyTrainStep, create_train_step (#1143)" },
  ],
};

const TASK_ID = "260605_xtrax-s2-core";
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

// Shared context for all writing tracks.
// FILE-PATH NOTE: Sprint 1 scaffold used different names than spec §2.
// Authoritative file paths for Sprint 2 (use these, not spec §2 paths):
//   safety/ops.py          <- spec says safety/numerics.py
//   safety/manager.py      <- spec says safety/checkify.py
//   training/types.py      <- spec splits into state.py / losses.py / callbacks.py
//   training/loss.py       <- spec says training/combinators.py
//   training/grad.py       <- spec says training/accumulate.py
//   data/pipeline.py       <- spec says data/streaming.py
const EMITTER_CTX = `xtrax sprint 2. Sprint 1 (scaffold + tiling) complete.
Spec: .praxia/docs/specs/260604_xtrax-spec.md (R3, oracle-approved). task_id: 260604_xtrax-shape
Key rules: no Python-loop hot paths; uv run pytest; ruff clean before commit.
All Phase 4 training code uses equinox (eqx.Module, filter_jit, filter_value_and_grad, apply_updates).
Optimizer code delegates entirely to optax — no reimplementation.
FILE PATHS (use these, not spec §2): safety/ops.py, safety/manager.py, training/types.py,
training/loss.py, training/grad.py, data/pipeline.py.
`;

// ---- per-track stage helpers ---------------------------------------------
const fixer = (prompt, label, phaseName) =>
  agent(prompt, {
    agentType: "fixer",
    label,
    phase: phaseName,
  });

const reviewer = (itemId, prompt, label, phaseName) =>
  agent(prompt, { agentType: "reviewer", label, phase: phaseName, schema: VERDICT_SCHEMA });

// Sequential implement->review with bounded NEEDS_WORK repair cycles.
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

const SKIPPED = (id, reason) => ({ item_id: id, verdict: "SKIPPED", summary: reason });
const passed = (v) => v?.verdict === "PASS";

// ===== TRACK A — Phase 3.1: safe_norm, safe_reciprocal (#1136) =========================
const trackA = () =>
  track(
    "1136",
    "Track A — Phase 3.1: safe_norm, safe_reciprocal (#1136)",
    `task_id: 260604_xtrax-shape
File: src/xtrax/safety/ops.py | Test: tests/safety/test_ops.py

Implement numerically stable ops in src/xtrax/safety/ops.py.
CRITICAL: use EXACTLY these formulas (spec §3.23) — any deviation fails review.

\`\`\`python
import jax.numpy as jnp
import jax

def safe_norm(x: jax.Array, axis=None, keepdims: bool = False, eps: float = 1e-8) -> jax.Array:
    # Mandated: jnp.sqrt(sum(x**2) + eps**2) — adds eps**2 (=1e-16), NOT finfo.eps
    return jnp.sqrt(jnp.sum(x**2, axis=axis, keepdims=keepdims) + eps**2)

def safe_reciprocal(x: jax.Array, eps: float = 1e-8) -> jax.Array:
    # Mandated: 1.0 / (x + eps) — smooth, unconditional; NO jnp.where branching
    return 1.0 / (x + eps)
\`\`\`

Update src/xtrax/safety/__init__.py to export both.

Write tests/safety/test_ops.py:
- safe_norm: grad at x=zeros is finite (not NaN) via jax.grad
- safe_norm uses eps**2: verify safe_norm(jnp.zeros(3)) == jnp.sqrt(1e-8**2) (not finfo.eps)
- safe_reciprocal(0.0) == 1/(0+1e-8) = 1e8 (smooth formula; NOT jnp.where)
- safe_reciprocal(2.0) ≈ 1/(2+1e-8) ≈ 0.5
- Both functions accept eps kwarg

Gate: \`uv run pytest tests/safety/test_ops.py -v\` pass; \`uv run ruff check src/xtrax/safety/ops.py\` clean.

${EMITTER_CTX}`,
    `task_id: 260604_xtrax-shape. Verify safe_norm and safe_reciprocal (spec §3.23):
1. safe_norm formula: jnp.sqrt(jnp.sum(x**2, ...) + eps**2) — VERIFY no finfo.eps, no linalg.norm
2. safe_norm has eps: float = 1e-8 parameter in signature
3. safe_reciprocal formula: 1.0 / (x + eps) — VERIFY no jnp.where, no conditional branching
4. safe_reciprocal has eps: float = 1e-8 parameter
5. jax.grad(lambda x: safe_norm(x))(jnp.zeros(3)) is finite (no NaN)
6. All tests pass; ruff clean
`,
  );

// ===== TRACK B — Phase 3.3: PreemptionHandler (#1138) =========================
const trackB = () =>
  track(
    "1138",
    "Track B — Phase 3.3: PreemptionHandler (#1138)",
    `task_id: 260604_xtrax-shape
File: src/xtrax/safety/preemption.py | Test: tests/safety/test_preemption.py

Implement PreemptionHandler in src/xtrax/safety/preemption.py (spec §3.25):

\`\`\`python
import signal
import threading
from collections.abc import Callable

class PreemptionHandler:
    def __init__(self, save_fn: Callable[[], None], rank: int = 0) -> None:
        self._save_fn = save_fn
        self._rank = rank
        self._registered = False
        self._preempted = threading.Event()
        self._saved = threading.Event()  # ensures save_fn called at most once

    @property
    def preempted(self) -> bool:
        return self._preempted.is_set()

    def register(self) -> None:
        if self._registered:
            return  # idempotent
        if self._rank == 0:
            signal.signal(signal.SIGUSR1, self._handle)
            signal.signal(signal.SIGTERM, self._handle)
        self._registered = True

    def _handle(self, signum, frame) -> None:
        self._preempted.set()
        if not self._saved.is_set():
            self._saved.set()
            self._save_fn()
\`\`\`

Write tests/safety/test_preemption.py:
- SIGUSR1 triggers save_fn exactly once (os.kill(os.getpid(), signal.SIGUSR1))
- SIGTERM also triggers save_fn (os.kill(os.getpid(), signal.SIGTERM))
- rank=1: save_fn NOT called on SIGUSR1 or SIGTERM
- register() idempotent (call twice, handler registered once)
- preempted property: False before signal, True after
- save_fn called at most once even if both SIGUSR1 and SIGTERM fire

Gate: \`uv run pytest tests/safety/test_preemption.py -v\` pass; \`uv run ruff check src/xtrax/safety/preemption.py\` clean.

${EMITTER_CTX}`,
    `task_id: 260604_xtrax-shape. Verify PreemptionHandler (spec §3.25):
1. SIGUSR1 and SIGTERM both trigger save_fn (both signals registered for rank=0)
2. rank=1: save_fn NOT called on either signal
3. register() idempotent — second call is a no-op
4. preempted property backed by threading.Event: False before signal, True after
5. save_fn called AT MOST ONCE even when both signals fire
6. All tests pass; ruff clean
`,
  );

// ===== TRACK C — Phase 3.4: Stage protocols (#1139) =========================
const trackC = () =>
  track(
    "1139",
    "Track C — Phase 3.4: Stage protocols — TransformFn, RollingFn, FuseFn (#1139)",
    `task_id: 260604_xtrax-shape
File: src/xtrax/stages/protocols.py | Test: tests/stages/test_protocols.py

Implement generic tier-1 protocols in src/xtrax/stages/protocols.py (spec §3.8):

\`\`\`python
from typing import Generic, Protocol, TypeVar, runtime_checkable

In = TypeVar("In")
Out = TypeVar("Out")
Carry = TypeVar("Carry")
PerItem = TypeVar("PerItem")
Combined = TypeVar("Combined")

@runtime_checkable
class TransformFn(Protocol[In, Out]):
    def __call__(self, x: In) -> Out: ...

@runtime_checkable
class RollingFn(Protocol[Carry, In, Out]):
    def __call__(self, carry: Carry, x: In) -> tuple[Carry, Out]: ...

@runtime_checkable
class FuseFn(Protocol[PerItem, Combined]):
    def __call__(self, items: PerItem) -> Combined: ...
\`\`\`

Update src/xtrax/stages/__init__.py.

Write tests/stages/test_protocols.py: lambdas pass isinstance for all three protocols.

Gate: \`uv run pytest tests/stages/test_protocols.py -v\` pass; ruff clean.

${EMITTER_CTX}`,
    `task_id: 260604_xtrax-shape. Verify Stage protocols (spec §3.8):
1. All three are @runtime_checkable
2. lambda x: x passes isinstance(fn, TransformFn)
3. lambda c, x: (c, x) passes isinstance(fn, RollingFn)
4. lambda items: items passes isinstance(fn, FuseFn)
5. All tests pass; ruff clean
`,
  );

// ===== TRACK D — Phase 4.1: ResumableState, LossFunction, Callback (#1141) =========================
const trackD = () =>
  track(
    "1141",
    "Track D — Phase 4.1: ResumableState, LossFunction, Callback (#1141)",
    `task_id: 260604_xtrax-shape
File: src/xtrax/training/types.py | Test: tests/training/test_types.py

Implement core training types in src/xtrax/training/types.py (spec §3.10–3.12).
CRITICAL: Callback must have exactly 7 hooks including on_resume; on_step_end takes metrics dict not bare Array.

\`\`\`python
from typing import Any, Protocol, runtime_checkable
import jax
import jax.numpy as jnp
import equinox as eqx

PyTree = Any
Array = jax.Array

@runtime_checkable
class LossFunction(Protocol):
    def __call__(self, predictions: PyTree, targets: PyTree) -> Array: ...

@runtime_checkable
class Callback(Protocol):
    def on_train_start(self, state: "ResumableState") -> None: ...
    def on_train_end(self, state: "ResumableState") -> None: ...
    def on_resume(self, state: "ResumableState") -> None: ...
    def on_epoch_start(self, state: "ResumableState", epoch: int) -> None: ...
    def on_epoch_end(self, state: "ResumableState", epoch: int) -> None: ...
    def on_step_start(self, state: "ResumableState") -> None: ...
    def on_step_end(self, state: "ResumableState", metrics: dict[str, Array]) -> None: ...
    # 7 hooks total; on_step_end takes metrics DICT (not bare loss Array)

class ResumableState(eqx.Module):
    step: Array          # jnp.int32 scalar — dynamic leaf, NOT static
    key: Array
    model: eqx.Module
    opt_state: PyTree
    extras: dict[str, PyTree] = eqx.field(default_factory=dict)
\`\`\`

Write tests: step is jnp.int32 Array and appears in eqx.filter(state, eqx.is_array);
LossFunction/Callback isinstance pass; Callback has exactly 7 hooks including on_resume;
on_step_end accepts (state, metrics: dict) not (state, loss: Array).

Gate: \`uv run pytest tests/training/test_types.py -v\` pass; ruff clean.

${EMITTER_CTX}`,
    `task_id: 260604_xtrax-shape. Verify training types (spec §3.10–3.12):
1. ResumableState.step is jnp.int32 Array (dynamic leaf): eqx.filter(state, eqx.is_array) INCLUDES step
2. LossFunction and Callback are @runtime_checkable Protocol; lambdas pass isinstance
3. Callback has EXACTLY 7 hooks: on_train_start, on_train_end, on_resume, on_epoch_start,
   on_epoch_end, on_step_start, on_step_end — on_resume MUST be present
4. on_step_end signature: (self, state, metrics: dict[str, Array]) — NOT loss: Array
5. All tests pass; ruff clean
`,
  );

// ===== TRACK E — Phase 4.5: Optimizer utilities (#1145) =========================
const trackE = () =>
  track(
    "1145",
    "Track E — Phase 4.5: Optimizer utilities (#1145)",
    `task_id: 260604_xtrax-shape
File: src/xtrax/training/optim.py | Test: tests/training/test_optim.py

Implement optimizer utilities in src/xtrax/training/optim.py (spec §3.16).
CRITICAL: weight_decay default is 1e-2 (NOT 1e-4). clip_norm=None returns base unchanged.

\`\`\`python
from collections.abc import Callable
from typing import Any
import jax
import equinox as eqx
import optax

PyTree = Any

def no_bias_wd_mask(params: PyTree) -> PyTree:
    # Mandated exactly: ndim != 1 -> True (apply wd), 1-D -> False (skip)
    return jax.tree.map(lambda x: x.ndim != 1, params)

def make_optimizer(
    base: optax.GradientTransformation,
    clip_norm: float | None = 1.0,
) -> optax.GradientTransformation:
    # clip_norm=None -> return base unchanged (no clipping)
    if clip_norm is None:
        return base
    # clip BEFORE base — mandatory order
    return optax.chain(optax.clip_by_global_norm(clip_norm), base)

def adamw_with_schedule(
    peak_lr: float,
    warmup_steps: int,
    total_steps: int,   # total steps INCLUDING warmup (NOT total-warmup)
    weight_decay: float = 1e-2,  # spec §3.16 default — NOT 1e-4
    clip_norm: float | None = 1.0,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    wd_mask: Callable = no_bias_wd_mask,
) -> optax.GradientTransformation:
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,  # total steps INCLUDING warmup — common footgun to avoid
        end_value=0.0,
    )
    base = optax.adamw(
        learning_rate=schedule, weight_decay=weight_decay,
        b1=b1, b2=b2, eps=eps,
        mask=wd_mask,  # mask IS always passed; mandatory
    )
    return make_optimizer(base, clip_norm=clip_norm)

def partition_labels(
    model: eqx.Module,
    frozen_filter: Callable[[Any], bool],
    frozen_label: str = "frozen",
    train_label: str = "train",
) -> PyTree:
    # Returns label pytree: each leaf is frozen_label or train_label.
    # Compatible with optax.partition(transforms, label_tree).
    return jax.tree.map(
        lambda leaf: frozen_label if frozen_filter(leaf) else train_label,
        model,
    )
\`\`\`

Write tests: no_bias_wd_mask({"w": ones((10,10)), "b": zeros(10)}) -> {"w": True, "b": False};
make_optimizer(base, clip_norm=None) returns base unchanged (same object);
make_optimizer chain order: clip_by_global_norm FIRST, then base;
adamw_with_schedule: weight_decay default is 1e-2; decay_steps=total_steps; mask passed;
partition_labels returns all frozen_label when frozen_filter=lambda _: True.

Gate: \`uv run pytest tests/training/test_optim.py -v\` pass; ruff clean.

${EMITTER_CTX}`,
    `task_id: 260604_xtrax-shape. Verify optimizer utilities (spec §3.16):
1. no_bias_wd_mask: {"w": shape(10,10), "b": shape(10)} -> {"w": True, "b": False}
2. make_optimizer(base, clip_norm=None) returns base UNCHANGED (same object, not a new chain)
3. make_optimizer(base, clip_norm=1.0): clip_by_global_norm FIRST in chain, then base
4. adamw_with_schedule: weight_decay default is 1e-2 (NOT 1e-4); decay_steps=total_steps
5. mask IS passed to adamw (verify wd_mask appears in optimizer via test)
6. partition_labels returns string-leaf pytree (all frozen_label when frozen_filter always True)
7. All tests pass; ruff clean
`,
  );

// ===== TRACK F — Phase 4.7: DataModule + streaming (#1147) =========================
const trackF = () =>
  track(
    "1147",
    "Track F — Phase 4.7: DataModule + streaming (#1147)",
    `task_id: 260604_xtrax-shape
Files: src/xtrax/data/module.py, src/xtrax/data/pipeline.py | Tests: tests/data/

Implement in src/xtrax/data/module.py (spec §3.19 — grain integration deferred to Phase 5/6):

\`\`\`python
from collections.abc import Callable, Iterator
from typing import Any
import equinox as eqx

# Module-level flag for distributed init state (real integration with init_dist deferred)
_dist_initialized: bool = False

def _mark_dist_initialized() -> None:
    """Call after init_dist() to allow DataModule iterators to proceed."""
    global _dist_initialized
    _dist_initialized = True

class DataModule(eqx.Module):
    dataset: Any
    batch_size: int = eqx.field(static=True)
    num_epochs: int | None = eqx.field(static=True)  # None = cycle indefinitely
    seed: int = eqx.field(static=True)
    distributed: bool = eqx.field(static=True)
    collate_fn: Callable | None = eqx.field(static=True, default=None)

    def train_iter(self) -> Iterator[Any]:
        if self.distributed and not _dist_initialized:
            raise RuntimeError(
                "DataModule: distributed=True requires init_dist() before train_iter()."
            )
        yield from self.dataset

    def eval_iter(self) -> Iterator[Any]:
        if self.distributed and not _dist_initialized:
            raise RuntimeError(
                "DataModule: distributed=True requires init_dist() before eval_iter()."
            )
        yield from self.dataset
\`\`\`

Implement in src/xtrax/data/pipeline.py (spec §3.20 — grain sharding deferred):

\`\`\`python
from collections.abc import AsyncIterator, Iterable
from typing import Any, TypeVar

T = TypeVar("T")

async def async_indexed_stream(iterable: Iterable[T]) -> AsyncIterator[tuple[int, T]]:
    for i, item in enumerate(iterable):
        yield i, item

def create_distributed_pipeline(
    dataset: Any,
    global_batch_size: int,
    num_devices: int,
    seed: int,
) -> Any:
    if global_batch_size % num_devices != 0:
        raise ValueError(
            f"create_distributed_pipeline: global_batch_size={global_batch_size} "
            f"must be divisible by num_devices={num_devices}."
        )
    per_device_batch = global_batch_size // num_devices
    return dataset  # stub — real grain sharding deferred to Phase 5/6
\`\`\`

Write tests: train_iter yields; eval_iter yields;
distributed=True without _mark_dist_initialized raises RuntimeError for BOTH train_iter and eval_iter;
_mark_dist_initialized() clears the guard (subsequent calls succeed);
create_distributed_pipeline raises for non-divisible; accepts seed param;
async_indexed_stream yields correct (index, item) pairs.

Gate: \`uv run pytest tests/data/ -v\` pass; ruff clean.

${EMITTER_CTX}`,
    `task_id: 260604_xtrax-shape. Verify DataModule and streaming (spec §3.19/3.20):
1. DataModule is an eqx.Module with static fields: batch_size, num_epochs, seed, distributed, collate_fn
2. train_iter() and eval_iter() yield from dataset
3. distributed=True without _mark_dist_initialized -> RuntimeError on BOTH train_iter AND eval_iter
4. _mark_dist_initialized() clears the guard (subsequent calls succeed)
5. create_distributed_pipeline raises ValueError for non-divisible; accepts seed: int param
6. async_indexed_stream yields (0, item0), (1, item1), ...
7. All tests pass; ruff clean
`,
  );

// ===== TRACK G — Phase 3.2: SafetyManager, with_safety (#1137) =========================
const trackG = () =>
  track(
    "1137",
    "Track G — Phase 3.2: SafetyManager, with_safety (#1137)",
    `task_id: 260604_xtrax-shape
File: src/xtrax/safety/manager.py | Test: tests/safety/test_manager.py

Implement in src/xtrax/safety/manager.py (spec §3.24).
CRITICAL: SafetyManager must be eqx.Module with 3 static fields; with_safety is alias for manager.wrap.
wrap() builds the jit-compiled checkified fn ONCE (not per call) to avoid recompilation.

\`\`\`python
from collections.abc import Callable
from typing import Any
import jax
import jax.experimental.checkify as checkify
import equinox as eqx

class SafetyManager(eqx.Module):
    enabled: bool = eqx.field(static=True)
    check_nans: bool = eqx.field(static=True)
    check_infs: bool = eqx.field(static=True)

    def wrap(self, fn: Callable) -> Callable:
        if not self.enabled:
            return fn  # strict identity: returned callable IS fn (no wrapper overhead)
        # Build jit-compiled checkified fn ONCE per wrap() call — NOT per invocation
        # float_checks covers NaN + Inf + overflow (both check_nans and check_infs)
        _checked_jit = jax.jit(checkify.checkify(fn, errors=checkify.float_checks))
        def wrapped_fn(*args: Any, **kwargs: Any) -> Any:
            err, result = _checked_jit(*args, **kwargs)
            err.throw()  # outside jit — raises Python exception on host
            return result
        return wrapped_fn

def with_safety(fn: Callable, manager: SafetyManager) -> Callable:
    """Convenience alias for manager.wrap(fn)."""
    return manager.wrap(fn)
\`\`\`

Write tests:
- SafetyManager(enabled=False, check_nans=False, check_infs=False): with_safety(fn, mgr) IS fn
  (strict identity via Python \`is\` operator — no wrapper)
- SafetyManager(enabled=True, check_nans=True, check_infs=True): wrapping a NaN-producing fn
  and calling it raises an exception on host
- SafetyManager is a valid eqx.Module (verify isinstance(mgr, eqx.Module))

Gate: \`uv run pytest tests/safety/test_manager.py -v\` pass; ruff clean.

${EMITTER_CTX}`,
    `task_id: 260604_xtrax-shape. Verify SafetyManager (spec §3.24):
1. SafetyManager is an eqx.Module with 3 eqx.field(static=True) fields: enabled, check_nans, check_infs
2. SafetyManager(enabled=False, ...): with_safety(fn, mgr) IS fn — strict Python identity (no wrapper)
3. SafetyManager(enabled=True, ...): NaN-producing fn raises Python exception after call (not silently)
4. with_safety(fn, mgr) is functionally identical to mgr.wrap(fn)
5. All tests pass; ruff clean
`,
  );

// ===== TRACK H — Phase 3.5: StageBundle (#1140) =========================
const trackH = () =>
  track(
    "1140",
    "Track H — Phase 3.5: StageBundle (#1140)",
    `task_id: 260604_xtrax-shape
File: src/xtrax/stages/bundle.py | Test: tests/stages/test_bundle.py

Implement StageBundle in src/xtrax/stages/bundle.py (spec §3.9):

\`\`\`python
from collections.abc import Callable
import equinox as eqx

class StageBundle(eqx.Module):
    """Typed bag of optional callable stage slots. Subclass and declare fields.
    Topology determined by non-None fields at Python dispatch level.
    WARNING: Do NOT call active_stages/has_stage inside JAX traces — Python-side only.
    No __call__ method — pure container.
    """

    def active_stages(self) -> list[str]:
        """Returns list[str] of field names with non-None callable values. Python-side only."""
        return [
            name for name, val in vars(self).items()
            if val is not None and callable(val)
        ]

    def has_stage(self, name: str) -> bool:
        """Returns True if named field is a non-None callable. Python-side only."""
        val = getattr(self, name, None)
        return val is not None and callable(val)
\`\`\`

Write tests: active_stages returns list[str] (names only, NOT dict, NOT callables);
has_stage True/False; valid eqx.Module pytree (jax.tree.leaves works);
no __call__ method on StageBundle base class.

Gate: \`uv run pytest tests/stages/test_bundle.py -v\` pass; ruff clean.

${EMITTER_CTX}`,
    `task_id: 260604_xtrax-shape. Verify StageBundle (spec §3.9):
1. active_stages() returns list[str] — names only, NOT a dict, NOT Callable values
2. has_stage() correct for present/absent slots
3. StageBundle subclass is valid eqx.Module (jax.tree.leaves works)
4. No __call__ method on StageBundle base class
5. All tests pass; ruff clean
`,
  );

// ===== TRACK I — Phase 4.2: Trainer (#1142) =========================
const trackI = () =>
  track(
    "1142",
    "Track I — Phase 4.2: Trainer (#1142)",
    `task_id: 260604_xtrax-shape
File: src/xtrax/training/trainer.py | Test: tests/training/test_trainer.py

Implement Trainer in src/xtrax/training/trainer.py (spec §3.13).
CRITICAL: (1) step returns dict[str, Array] NOT bare Array.
(2) optimizer.update 3rd arg MUST be eqx.filter(state.model, eqx.is_array) — NOT raw state.model.

\`\`\`python
from typing import Any
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from xtrax.training.types import LossFunction, ResumableState

class Trainer(eqx.Module):
    loss_fn: LossFunction
    optimizer: optax.GradientTransformation

    @eqx.filter_jit
    def step(
        self,
        state: ResumableState,
        batch: Any,
    ) -> tuple[ResumableState, dict[str, jax.Array]]:
        def loss_fn_inner(model):
            predictions = model(batch["inputs"])
            return self.loss_fn(predictions, batch["targets"])

        loss, grads = eqx.filter_value_and_grad(loss_fn_inner)(state.model)

        # MANDATORY: eqx.filter(state.model, eqx.is_array) as 3rd arg (spec §3.13 line 387).
        # grads from filter_value_and_grad are array-only; params must match structure.
        updates, new_opt_state = self.optimizer.update(
            grads, state.opt_state, eqx.filter(state.model, eqx.is_array)
        )
        new_model = eqx.apply_updates(state.model, updates)
        new_state = eqx.tree_at(
            lambda s: (s.model, s.opt_state, s.step),
            state,
            (new_model, new_opt_state, state.step + 1),
        )
        return new_state, {"loss": loss}  # metrics dict — at minimum {"loss": scalar}
\`\`\`

Write tests: step increments state.step by 1; step returns (ResumableState, dict) with
metrics["loss"] a scalar Array (NOT bare Array); loss decreases over 10 steps trivial regression
(linear model, MSE loss); verify optimizer.update called with eqx.filter(state.model, eqx.is_array)
as 3rd arg (use a mock GradientTransformation or check opt_state shape).

Gate: \`uv run pytest tests/training/test_trainer.py -v\` pass; ruff clean.

${EMITTER_CTX}`,
    `task_id: 260604_xtrax-shape. Verify Trainer (spec §3.13):
1. state.step increments by exactly 1 per step call
2. step returns (ResumableState, dict[str, Array]): metrics["loss"] is SCALAR Array, NOT bare Array
3. optimizer.update 3rd arg is eqx.filter(state.model, eqx.is_array) — NOT raw state.model
   (verify by testing with adamw that has weight decay — wrong arg would cause tree-structure error)
4. eqx.apply_updates used for model update (not manual assignment)
5. Loss decreases over 10 steps on trivial regression task
6. All tests pass; ruff clean
`,
  );

// ===== TRACK J — Phase 4.4: Loss combinators (#1144) =========================
const trackJ = () =>
  track(
    "1144",
    "Track J — Phase 4.4: Loss combinators (#1144)",
    `task_id: 260604_xtrax-shape
File: src/xtrax/training/loss.py | Test: tests/training/test_loss.py

Implement loss combinators in src/xtrax/training/loss.py (spec §3.15).
CRITICAL: WeightedLoss.weight MUST be eqx.field(static=True) Python float — NOT jax.Array.

\`\`\`python
from typing import Any
import jax
import jax.numpy as jnp
import equinox as eqx
from xtrax.training.types import LossFunction

class WeightedLoss(eqx.Module):
    loss_fn: LossFunction
    weight: float = eqx.field(static=True)  # static float — NOT jax.Array

    def __call__(self, predictions: Any, targets: Any) -> jax.Array:
        return self.weight * self.loss_fn(predictions, targets)

class MultiTaskLoss(eqx.Module):
    losses: tuple[WeightedLoss, ...]

    def __call__(
        self,
        predictions: tuple,
        targets: tuple,
    ) -> jax.Array:
        assert len(predictions) == len(targets) == len(self.losses), (
            f"MultiTaskLoss: length mismatch predictions={len(predictions)}, "
            f"targets={len(targets)}, losses={len(self.losses)}"
        )
        # Static-length tuple comprehension — unrolls at trace time (NOT a data-axis hot loop)
        return jnp.sum(jnp.stack([l(p, t) for l, p, t in zip(self.losses, predictions, targets)]))
\`\`\`

Write tests: WeightedLoss == weight*loss; MultiTaskLoss == sum of each weighted loss;
both pass isinstance(fn, LossFunction); tuple-length mismatch raises AssertionError;
weight is static: verify weight NOT in eqx.filter(wl, eqx.is_array).

Gate: \`uv run pytest tests/training/test_loss.py -v\` pass; ruff clean.

${EMITTER_CTX}`,
    `task_id: 260604_xtrax-shape. Verify loss combinators (spec §3.15):
1. WeightedLoss.weight is eqx.field(static=True) Python float: verify weight does NOT appear
   in eqx.filter(WeightedLoss(weight=2.0, loss_fn=...), eqx.is_array)
2. WeightedLoss(weight=2.0, loss_fn): output == 2.0 * loss_fn(p, t)
3. MultiTaskLoss: output == sum of each WeightedLoss applied to (p, t) pairs
4. Both pass isinstance(fn, LossFunction)
5. len mismatch raises AssertionError
6. losses field typed as tuple[WeightedLoss, ...] (not bare tuple)
7. All tests pass; ruff clean
`,
  );

// ===== TRACK K — Phase 4.6: accumulate_grads (#1146) =========================
const trackK = () =>
  track(
    "1146",
    "Track K — Phase 4.6: accumulate_grads (#1146)",
    `task_id: 260604_xtrax-shape
File: src/xtrax/training/grad.py | Test: tests/training/test_grad.py

Implement accumulate_grads in src/xtrax/training/grad.py using jax.lax.scan — NO Python for-loop.
microbatches is a pre-stacked PyTree with leading axis = num_microbatches (spec §3.17).

\`\`\`python
from collections.abc import Callable
from typing import Any
import jax
import jax.numpy as jnp
import equinox as eqx

def accumulate_grads(
    loss_fn: Callable[[Any, Any], jax.Array],
    params: Any,
    microbatches: Any,  # pre-stacked PyTree; leading axis = num_microbatches
    filter_spec: Callable | None = None,
) -> tuple[Any, jax.Array]:
    """Returns (mean_grads, mean_loss). Uses lax.scan — no Python for-loop."""
    if filter_spec is None:
        filter_spec = eqx.is_array

    def scan_fn(carry, microbatch):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(params, microbatch)
        return carry, (grads, loss)

    _, (all_grads, all_losses) = jax.lax.scan(scan_fn, None, microbatches)

    mean_grads = jax.tree.map(lambda g: jnp.mean(g, axis=0), all_grads)
    mean_loss = jnp.mean(all_losses)
    return mean_grads, mean_loss
\`\`\`

Write tests:
- Use jax.make_jaxpr to verify 'scan' appears in jaxpr (not a Python for-loop)
- Stack 4 equal microbatches of size 25 into shape (4, 25, ...) pytree;
  verify result matches full-batch grad within atol=1e-5, rtol=1e-5
- Returns (mean_grads, mean_loss) where mean_loss is a scalar Array

Gate: \`uv run pytest tests/training/test_grad.py -v\` pass; ruff clean.

${EMITTER_CTX}`,
    `task_id: 260604_xtrax-shape. Verify accumulate_grads (spec §3.17):
1. Implementation uses jax.lax.scan — verify via jax.make_jaxpr that 'scan' appears (NO Python for-loop)
2. With 4 equal microbatches of size 25 (pre-stacked as shape (4,25,...)):
   result matches full-batch grad within atol=1e-5, rtol=1e-5
3. Returns (mean_grads, mean_loss) where mean_loss is a scalar Array
4. All tests pass; ruff clean
`,
  );

// ===== TRACK L — Phase 4.3: SafetyTrainStep, create_train_step (#1143) =========================
const trackL = () =>
  track(
    "1143",
    "Track L — Phase 4.3: SafetyTrainStep, create_train_step (#1143)",
    `task_id: 260604_xtrax-shape
File: src/xtrax/training/step.py | Test: tests/training/test_step.py

Implement in src/xtrax/training/step.py (spec §3.14).
CRITICAL:
(1) Fields MUST be trainer/safety_manager (PUBLIC names per spec — not _trainer/_mgr).
(2) Do NOT call with_safety inside step() — that creates a new jitted wrapper every call.
(3) Use _step_jit (filter_jit'd) + step (Python wrapper that calls err.throw() outside jit).
(4) step returns dict[str, Array] matching Trainer.step interface.

\`\`\`python
from typing import Any
import jax
import jax.experimental.checkify as checkify
import equinox as eqx
import optax
from xtrax.training.types import LossFunction, ResumableState
from xtrax.training.trainer import Trainer
from xtrax.safety.manager import SafetyManager

class SafetyTrainStep(eqx.Module):
    trainer: Trainer           # PUBLIC field name (spec §3.14)
    safety_manager: SafetyManager  # PUBLIC field name (spec §3.14)

    @eqx.filter_jit
    def _step_jit(self, state: ResumableState, batch: Any):
        # JIT-compiled checkified step. Traced once per (state, batch) shape.
        # The inner Trainer.step filter_jit is stripped by JAX in jit-in-jit context;
        # checkify.checkify instruments the computation correctly.
        checked = checkify.checkify(self.trainer.step, errors=checkify.float_checks)
        return checked(state, batch)  # returns (err, (new_state, metrics))

    def step(
        self,
        state: ResumableState,
        batch: Any,
    ) -> tuple[ResumableState, dict[str, jax.Array]]:
        if not self.safety_manager.enabled:
            return self.trainer.step(state, batch)
        err, (new_state, metrics) = self._step_jit(state, batch)
        err.throw()  # OUTSIDE jit — raises Python exception on host
        return new_state, metrics

def create_train_step(
    loss_fn: LossFunction,
    optimizer: optax.GradientTransformation,
    safety: bool = False,
    safety_manager: SafetyManager | None = None,
) -> Trainer | SafetyTrainStep:
    trainer = Trainer(loss_fn=loss_fn, optimizer=optimizer)
    if not safety:
        return trainer
    if safety_manager is None:
        safety_manager = SafetyManager(enabled=True, check_nans=True, check_infs=True)
    return SafetyTrainStep(trainer=trainer, safety_manager=safety_manager)
\`\`\`

Write tests: create_train_step(safety=False) returns Trainer;
create_train_step(safety=True) returns SafetyTrainStep;
SafetyTrainStep has PUBLIC fields .trainer and .safety_manager (not _trainer/_mgr);
both expose .step(state, batch) -> (state, dict) where metrics["loss"] is Array (NOT bare Array);
SafetyTrainStep with NaN-producing loss raises on host (err.throw() fires);
create_train_step(safety=True, safety_manager=SafetyManager(enabled=False, check_nans=False, check_infs=False))
returns SafetyTrainStep with disabled safety (step delegates to trainer.step directly).

Gate: \`uv run pytest tests/training/test_step.py -v\` pass; ruff clean.

${EMITTER_CTX}`,
    `task_id: 260604_xtrax-shape. Verify SafetyTrainStep (spec §3.14):
1. safety=False -> returns Trainer instance
2. safety=True -> returns SafetyTrainStep instance
3. SafetyTrainStep has fields .trainer (NOT ._trainer) and .safety_manager (NOT ._mgr)
4. Both .step(state, batch) return (ResumableState, dict[str, Array]) — metrics["loss"] Array, NOT bare Array
5. SafetyTrainStep with NaN-producing loss raises Python exception on host (err.throw())
6. with_safety is NOT called inside step() or _step_jit() — checkify is inlined directly
7. create_train_step accepts safety_manager: SafetyManager | None = None kwarg
8. All tests pass; ruff clean
`,
  );

// ---- orchestrate: sequential chain with dependency guards ----
log("xtrax Sprint 2: Safety + Training Core — chain A->B->C->D->E->F->G->H->I->J->K->L (sequential, dependency-guarded)");

const a = await trackA();
const b = await trackB();
const c = await trackC();
const d = await trackD();
const e = await trackE();
const f = await trackF();
const g = await trackG();
const h = await trackH();

// I, J, K depend on D (training types / ResumableState)
const i = passed(d)
  ? await trackI()
  : SKIPPED("1142", "dep 1141 (ResumableState/types) not PASS");

const j = passed(d)
  ? await trackJ()
  : SKIPPED("1144", "dep 1141 (training types) not PASS");

const k = passed(d)
  ? await trackK()
  : SKIPPED("1146", "dep 1141 (training types) not PASS");

// L depends on D (types) + G (SafetyManager) + I (Trainer)
const l = (passed(d) && passed(g) && passed(i))
  ? await trackL()
  : SKIPPED("1143", "dep 1141/1137/1142 not all PASS");

return {
  task_id: TASK_ID,
  sprint_id: 2,
  verdicts: {
    "1136": a,
    "1138": b,
    "1139": c,
    "1141": d,
    "1145": e,
    "1147": f,
    "1137": g,
    "1140": h,
    "1142": i,
    "1144": j,
    "1146": k,
    "1143": l,
  },
};
