// Sprint 2 runner — emitted by `praxia dw emit-sprint`
// Source: .praxia/sprint_plans/sprint_plan.toml
// Regenerate: praxia dw emit-sprint sprint_plan.toml
// task_id: 260605_xtrax-s2-core   sprint_id: 2
//
// RACE SAFETY (memory: parallel fixers race on git-status scope checks in praxia):
//   the writing chain (A,B,C,D,E,F,G,H,I,J,K,L) runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time.

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

function extractVerdict(text) {
  const m = String(text ?? "").match(/verdict:\s*([a-z_]+)/i);
  return m ? m[1].toLowerCase() : "advance";
}

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

// Shared context for the writing tracks (from recon, task 260605_xtrax-s2-core).
const EMITTER_CTX = `xtrax sprint 2. Sprint 1 (scaffold + tiling) must be complete before this sprint runs.\nSpec: .praxia/docs/specs/260604_xtrax-spec.md (R3, oracle-approved).\ntask_id: 260604_xtrax-shape\nKey rules: no Python-loop hot paths; uv run pytest; ruff clean before commit.\nAll Phase 4 training code uses equinox (eqx.Module, filter_jit, filter_value_and_grad, apply_updates).\nOptimizer code delegates entirely to optax — no reimplementation.\n`;

// ---- per-track stage helpers ---------------------------------------------
const fixer = (prompt, label, phaseName) =>
  agent(`${prompt}\n\nWhen done, end your message with 'verdict: done' on its own line.`, {
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

// ===== TRACK A — Track A — Phase 3.1: safe_norm, safe_reciprocal (#1136) =========================
const trackA = () =>
  track(
    "1136",
    "Track A — Phase 3.1: safe_norm, safe_reciprocal (#1136)",
    `task_id: ${TASK_ID}. task_id: 260604_xtrax-shape\nFile: src/xtrax/safety/ops.py | Test: tests/safety/test_ops.py\n\nImplement numerically stable ops in src/xtrax/safety/ops.py:\n\n\`\`\`python\nimport jax.numpy as jnp\nimport jax\n\ndef safe_norm(x: jax.Array, axis: int | None = None, keepdims: bool = False) -> jax.Array:\n    # Use linalg.norm but prevent NaN gradient at zero by adding eps before sqrt\n    sq = jnp.sum(x * x, axis=axis, keepdims=keepdims)\n    return jnp.sqrt(sq + jnp.finfo(x.dtype).eps)\n\ndef safe_reciprocal(x: jax.Array) -> jax.Array:\n    eps = jnp.finfo(x.dtype).eps\n    return jnp.where(jnp.abs(x) < eps, 1.0 / eps, 1.0 / x)\n\`\`\`\n\nUpdate src/xtrax/safety/__init__.py to export both.\n\nWrite tests/safety/test_ops.py:\n- safe_norm: grad at x=zeros is finite (not NaN) via jax.grad\n- safe_reciprocal(0.0) == 1/eps (not inf, not NaN)\n- safe_reciprocal(2.0) ≈ 0.5\n\nGate: \`uv run pytest tests/safety/test_ops.py -v\` pass; \`uv run ruff check src/xtrax/safety/ops.py\` clean.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify safe_norm and safe_reciprocal:\n1. jax.grad(safe_norm)(jnp.zeros(3)) is finite (no NaN)\n2. safe_reciprocal(jnp.zeros(())) == 1/jnp.finfo(jnp.float32).eps\n3. safe_reciprocal(jnp.array(2.0)) ≈ 0.5\n4. All tests pass; ruff clean\n`,
  );

// ===== TRACK B — Track B — Phase 3.3: PreemptionHandler (#1138) =========================
const trackB = () =>
  track(
    "1138",
    "Track B — Phase 3.3: PreemptionHandler (#1138)",
    `task_id: ${TASK_ID}. task_id: 260604_xtrax-shape\nFile: src/xtrax/safety/preemption.py | Test: tests/safety/test_preemption.py\n\nImplement PreemptionHandler in src/xtrax/safety/preemption.py:\n\n\`\`\`python\nimport signal\nfrom collections.abc import Callable\n\nclass PreemptionHandler:\n    def __init__(self, save_fn: Callable, rank: int = 0) -> None:\n        self._save_fn = save_fn\n        self._rank = rank\n        self._registered = False\n\n    def register(self) -> None:\n        if self._registered:\n            return  # idempotent\n        if self._rank == 0:\n            signal.signal(signal.SIGUSR1, self._handle)\n        self._registered = True\n\n    def _handle(self, signum, frame):\n        self._save_fn()\n\`\`\`\n\nWrite tests/safety/test_preemption.py:\n- SIGUSR1 triggers save_fn once (send os.kill(os.getpid(), signal.SIGUSR1))\n- rank=1: save_fn NOT called on SIGUSR1\n- register() is idempotent (call twice, only registered once)\n\nGate: \`uv run pytest tests/safety/test_preemption.py -v\` pass; \`uv run ruff check src/xtrax/safety/preemption.py\` clean.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify PreemptionHandler:\n1. SIGUSR1 → save_fn called exactly once\n2. rank=1 → save_fn NOT called on SIGUSR1\n3. register() idempotent — second call is a no-op\n4. All tests pass; ruff clean\n`,
  );

// ===== TRACK C — Track C — Phase 3.4: Stage protocols — TransformFn, RollingFn, FuseFn (#1139) =========================
const trackC = () =>
  track(
    "1139",
    "Track C — Phase 3.4: Stage protocols — TransformFn, RollingFn, FuseFn (#1139)",
    `task_id: ${TASK_ID}. task_id: 260604_xtrax-shape\nFile: src/xtrax/stages/protocols.py | Test: tests/stages/test_protocols.py\n\nImplement generic tier-1 protocols in src/xtrax/stages/protocols.py:\n\n\`\`\`python\nfrom typing import Generic, Protocol, TypeVar, runtime_checkable\n\nIn = TypeVar("In")\nOut = TypeVar("Out")\nCarry = TypeVar("Carry")\nPerItem = TypeVar("PerItem")\nCombined = TypeVar("Combined")\n\n@runtime_checkable\nclass TransformFn(Protocol[In, Out]):\n    def __call__(self, x: In) -> Out: ...\n\n@runtime_checkable\nclass RollingFn(Protocol[Carry, In, Out]):\n    def __call__(self, carry: Carry, x: In) -> tuple[Carry, Out]: ...\n\n@runtime_checkable\nclass FuseFn(Protocol[PerItem, Combined]):\n    def __call__(self, items: PerItem) -> Combined: ...\n\`\`\`\n\nUpdate src/xtrax/stages/__init__.py.\n\nWrite tests/stages/test_protocols.py: lambdas pass isinstance for all three protocols.\n\nGate: \`uv run pytest tests/stages/test_protocols.py -v\` pass; ruff clean.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify Stage protocols:\n1. All three are runtime_checkable\n2. lambda x: x passes isinstance(fn, TransformFn)\n3. lambda c, x: (c, x) passes isinstance(fn, RollingFn)\n4. lambda items: items passes isinstance(fn, FuseFn)\n5. All tests pass; ruff clean\n`,
  );

// ===== TRACK D — Track D — Phase 4.1: ResumableState, LossFunction, Callback (#1141) =========================
const trackD = () =>
  track(
    "1141",
    "Track D — Phase 4.1: ResumableState, LossFunction, Callback (#1141)",
    `task_id: ${TASK_ID}. task_id: 260604_xtrax-shape\nFile: src/xtrax/training/types.py | Test: tests/training/test_types.py\n\nImplement core training types in src/xtrax/training/types.py:\n\n\`\`\`python\nfrom typing import Any, Protocol, runtime_checkable\nimport jax\nimport jax.numpy as jnp\nimport equinox as eqx\n\n@runtime_checkable\nclass LossFunction(Protocol):\n    def __call__(self, predictions: Any, targets: Any) -> jax.Array: ...\n\n@runtime_checkable\nclass Callback(Protocol):\n    def on_train_start(self, state: "ResumableState") -> None: ...\n    def on_train_end(self, state: "ResumableState") -> None: ...\n    def on_epoch_start(self, state: "ResumableState", epoch: int) -> None: ...\n    def on_epoch_end(self, state: "ResumableState", epoch: int) -> None: ...\n    def on_step_start(self, state: "ResumableState") -> None: ...\n    def on_step_end(self, state: "ResumableState", loss: jax.Array) -> None: ...\n\nclass ResumableState(eqx.Module):\n    step: jax.Array  # jnp.int32 scalar — DYNAMIC leaf, not static\n    key: jax.Array\n    model: eqx.Module\n    opt_state: Any\n    extras: dict[str, Any] = eqx.field(default_factory=dict)\n\`\`\`\n\nWrite tests: step is jnp.int32 Array (dynamic leaf, not static); LossFunction/Callback isinstance pass; ResumableState is valid eqx.Module.\n\nGate: \`uv run pytest tests/training/test_types.py -v\` pass; ruff clean.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify training types:\n1. ResumableState.step is jnp.int32 Array (dynamic JAX leaf, NOT static)\n2. Verify: \`eqx.filter(state, eqx.is_array)\` includes \`step\`\n3. LossFunction and Callback are runtime_checkable Protocol\n4. Lambdas pass isinstance checks\n5. All tests pass; ruff clean\n`,
  );

// ===== TRACK E — Track E — Phase 4.5: Optimizer utilities (#1145) =========================
const trackE = () =>
  track(
    "1145",
    "Track E — Phase 4.5: Optimizer utilities (#1145)",
    `task_id: ${TASK_ID}. task_id: 260604_xtrax-shape\nFile: src/xtrax/training/optim.py | Test: tests/training/test_optim.py\n\nImplement optimizer utilities in src/xtrax/training/optim.py:\n\n\`\`\`python\nfrom typing import Any\nimport jax\nimport optax\n\nPyTree = Any\n\ndef no_bias_wd_mask(params: PyTree) -> PyTree:\n    # Mandated exactly: ndim != 1 → True (apply weight decay), 1-D → False (skip)\n    return jax.tree.map(lambda x: x.ndim != 1, params)\n\ndef make_optimizer(base: optax.GradientTransformation, clip_norm: float = 1.0) -> optax.GradientTransformation:\n    # clip BEFORE base optimizer\n    return optax.chain(optax.clip_by_global_norm(clip_norm), base)\n\ndef adamw_with_schedule(\n    peak_lr: float,\n    warmup_steps: int,\n    total_steps: int,\n    weight_decay: float = 1e-4,\n    b1: float = 0.9,\n    b2: float = 0.999,\n    eps: float = 1e-8,\n    clip_norm: float = 1.0,\n    wd_mask: Any = None,\n) -> optax.GradientTransformation:\n    if wd_mask is None:\n        wd_mask = no_bias_wd_mask\n    schedule = optax.warmup_cosine_decay_schedule(\n        init_value=0.0, peak_value=peak_lr,\n        warmup_steps=warmup_steps,\n        decay_steps=total_steps,  # total steps, NOT total-warmup\n        end_value=0.0,\n    )\n    base = optax.adamw(\n        learning_rate=schedule, weight_decay=weight_decay,\n        b1=b1, b2=b2, eps=eps,\n        mask=wd_mask,  # mask IS always passed — mandatory\n    )\n    return make_optimizer(base, clip_norm=clip_norm)\n\ndef partition_labels(model: Any, label_fn: Any) -> Any:\n    return jax.tree.map(label_fn, model)\n\`\`\`\n\nWrite tests: no_bias_wd_mask({"w": ones((10,10)), "b": zeros(10)}) → {"w": True, "b": False}; make_optimizer chain order (clip first); adamw_with_schedule returns GradientTransformation; verify mask IS passed (use mock or check opt has wd_mask).\n\nGate: \`uv run pytest tests/training/test_optim.py -v\` pass; ruff clean.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify optimizer utilities:\n1. no_bias_wd_mask: {"w": shape(10,10), "b": shape(10)} → {"w": True, "b": False}\n2. make_optimizer: clip_by_global_norm FIRST in chain, then base\n3. adamw_with_schedule: decay_steps=total_steps (NOT total-warmup); mask IS passed\n4. All tests pass; ruff clean\n`,
  );

// ===== TRACK F — Track F — Phase 4.7: DataModule + streaming (#1147) =========================
const trackF = () =>
  track(
    "1147",
    "Track F — Phase 4.7: DataModule + streaming (#1147)",
    `task_id: ${TASK_ID}. task_id: 260604_xtrax-shape\nFiles: src/xtrax/data/module.py, src/xtrax/data/pipeline.py | Tests: tests/data/\n\nImplement in src/xtrax/data/module.py:\n\n\`\`\`python\nfrom collections.abc import Iterator\nfrom typing import Any\nimport jax.numpy as jnp\n\nclass DataModule:\n    def __init__(self, dataset: Any, batch_size: int, distributed: bool = False) -> None:\n        self._dataset = dataset\n        self._batch_size = batch_size\n        self._distributed = distributed\n        self._dist_initialized = False\n\n    def mark_distributed_initialized(self) -> None:\n        self._dist_initialized = True\n\n    def train_iter(self) -> Iterator[Any]:\n        if self._distributed and not self._dist_initialized:\n            raise RuntimeError(\n                "DataModule: distributed=True requires init_dist() before train_iter()."\n            )\n        yield from self._dataset\n\`\`\`\n\nImplement async_indexed_stream in src/xtrax/data/pipeline.py:\n\`\`\`python\nfrom collections.abc import AsyncIterator, Iterable\nfrom typing import Any, TypeVar\n\nT = TypeVar("T")\n\nasync def async_indexed_stream(iterable: Iterable[T]) -> AsyncIterator[tuple[int, T]]:\n    for i, item in enumerate(iterable):\n        yield i, item\n\`\`\`\n\nImplement create_distributed_pipeline:\n\`\`\`python\ndef create_distributed_pipeline(dataset: Any, global_batch_size: int, num_devices: int) -> Any:\n    if global_batch_size % num_devices != 0:\n        raise ValueError(\n            f"create_distributed_pipeline: global_batch_size={global_batch_size} "\n            f"must be divisible by num_devices={num_devices}."\n        )\n    per_device_batch = global_batch_size // num_devices\n    return dataset  # stub — real sharding via grain in future phases\n\`\`\`\n\nWrite tests: train_iter yields; distributed=True without init raises RuntimeError; create_distributed_pipeline raises for non-divisible; async_indexed_stream yields (index, item) pairs; exception from iterable re-raised.\n\nGate: \`uv run pytest tests/data/ -v\` pass; ruff clean.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify DataModule and streaming:\n1. DataModule.train_iter yields items correctly\n2. DataModule(distributed=True).train_iter() raises RuntimeError before mark_distributed_initialized()\n3. create_distributed_pipeline raises ValueError for non-divisible global_batch_size/num_devices\n4. async_indexed_stream yields correct (index, item) pairs\n5. Exception from iterable is re-raised (propagates to consumer)\n6. All tests pass; ruff clean\n`,
  );

// ===== TRACK G — Track G — Phase 3.2: SafetyManager, with_safety (#1137) =========================
const trackG = () =>
  track(
    "1137",
    "Track G — Phase 3.2: SafetyManager, with_safety (#1137)",
    `task_id: ${TASK_ID}. task_id: 260604_xtrax-shape\nFile: src/xtrax/safety/manager.py | Test: tests/safety/test_manager.py\n\nImplement in src/xtrax/safety/manager.py:\n\n\`\`\`python\nfrom collections.abc import Callable\nfrom dataclasses import dataclass\nfrom typing import Any\nimport jax\nimport jax.experimental.checkify as checkify\n\n@dataclass\nclass SafetyManager:\n    enabled: bool = True\n\ndef with_safety(fn: Callable, mgr: SafetyManager) -> Callable:\n    if not mgr.enabled:\n        return fn  # strict identity: return fn unchanged\n    checked_fn = checkify.checkify(fn, errors=checkify.float_checks)\n    @jax.jit\n    def wrapped(*args, **kwargs):\n        err, result = checked_fn(*args, **kwargs)\n        err.throw()\n        return result\n    return wrapped\n\`\`\`\n\nWrite tests:\n- SafetyManager(enabled=False) + with_safety(fn, mgr): returned callable IS fn (strict identity via \`is\`)\n- SafetyManager(enabled=True) wrapping a NaN-producing fn raises on host after jit call\n\nGate: \`uv run pytest tests/safety/test_manager.py -v\` pass; ruff clean.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify SafetyManager:\n1. SafetyManager(enabled=False): \`with_safety(fn, mgr) is fn\` — strict identity (not a wrapper)\n2. SafetyManager(enabled=True): NaN-producing jitted fn raises Python exception on host\n3. All tests pass; ruff clean\n`,
  );

// ===== TRACK H — Track H — Phase 3.5: StageBundle (#1140) =========================
const trackH = () =>
  track(
    "1140",
    "Track H — Phase 3.5: StageBundle (#1140)",
    `task_id: ${TASK_ID}. task_id: 260604_xtrax-shape\nFile: src/xtrax/stages/bundle.py | Test: tests/stages/test_bundle.py\n\nImplement StageBundle in src/xtrax/stages/bundle.py:\n\n\`\`\`python\nfrom collections.abc import Callable\nfrom typing import Any\nimport equinox as eqx\n\nclass StageBundle(eqx.Module):\n    # Base class for typed optional-slot bags.\n    # Subclasses declare optional Callable fields; topology inferred from non-None fields.\n    # NO __call__ method — caller decides how to sequence stages.\n    # WARNING: Do NOT branch on field presence inside jax.jit — caller precondition.\n\n    def active_stages(self) -> dict[str, Callable]:\n        return {\n            name: val\n            for name, val in self.__dict__.items()\n            if isinstance(val, Callable) and val is not None\n        }\n\n    def has_stage(self, name: str) -> bool:\n        val = getattr(self, name, None)\n        return val is not None and callable(val)\n\`\`\`\n\nWrite tests: active_stages returns correct non-None callables; has_stage True/False; valid eqx.Module pytree; no __call__ method.\n\nGate: \`uv run pytest tests/stages/test_bundle.py -v\` pass; ruff clean.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify StageBundle:\n1. active_stages() returns only non-None callable fields\n2. has_stage() correct for present/absent slots\n3. StageBundle subclass is valid eqx.Module (jax.tree.leaves works)\n4. No __call__ method on StageBundle\n5. All tests pass; ruff clean\n`,
  );

// ===== TRACK I — Track I — Phase 4.2: Trainer (#1142) =========================
const trackI = () =>
  track(
    "1142",
    "Track I — Phase 4.2: Trainer (#1142)",
    `task_id: ${TASK_ID}. task_id: 260604_xtrax-shape\nFile: src/xtrax/training/trainer.py | Test: tests/training/test_trainer.py\n\nImplement Trainer in src/xtrax/training/trainer.py:\n\n\`\`\`python\nfrom typing import Any\nimport jax\nimport jax.numpy as jnp\nimport equinox as eqx\nimport optax\nfrom xtrax.training.types import LossFunction, ResumableState\n\nclass Trainer(eqx.Module):\n    loss_fn: LossFunction\n    optimizer: optax.GradientTransformation\n\n    @eqx.filter_jit\n    def step(self, state: ResumableState, batch: Any) -> tuple[ResumableState, jax.Array]:\n        def loss_fn_inner(model):\n            predictions = model(batch["inputs"])\n            return self.loss_fn(predictions, batch["targets"])\n\n        loss, grads = eqx.filter_value_and_grad(loss_fn_inner)(state.model)\n        # Pass params as 3rd arg to optimizer.update — required for adamw and weight decay\n        updates, new_opt_state = self.optimizer.update(grads, state.opt_state, state.model)\n        new_model = eqx.apply_updates(state.model, updates)\n        new_state = eqx.tree_at(\n            lambda s: (s.model, s.opt_state, s.step),\n            state,\n            (new_model, new_opt_state, state.step + 1),\n        )\n        return new_state, loss\n\`\`\`\n\nWrite tests: step increments state.step by 1; loss decreases over 10 steps trivial regression (linear model, MSE); params passed as 3rd arg to optimizer.update (mock optax optimizer to verify).\n\nGate: \`uv run pytest tests/training/test_trainer.py -v\` pass; ruff clean.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify Trainer:\n1. state.step increments by exactly 1 per step call\n2. Loss decreases over 10 steps on trivial regression task\n3. optimizer.update called with (grads, opt_state, model) — 3rd arg is model/params\n4. eqx.apply_updates used (not manual parameter update)\n5. All tests pass; ruff clean\n`,
  );

// ===== TRACK J — Track J — Phase 4.4: Loss combinators (#1144) =========================
const trackJ = () =>
  track(
    "1144",
    "Track J — Phase 4.4: Loss combinators (#1144)",
    `task_id: ${TASK_ID}. task_id: 260604_xtrax-shape\nFile: src/xtrax/training/loss.py | Test: tests/training/test_loss.py\n\nImplement loss combinators in src/xtrax/training/loss.py:\n\n\`\`\`python\nfrom typing import Any\nimport jax\nimport jax.numpy as jnp\nimport equinox as eqx\nfrom xtrax.training.types import LossFunction\n\nclass WeightedLoss(eqx.Module):\n    loss_fn: LossFunction\n    weight: jax.Array\n\n    def __call__(self, predictions: Any, targets: Any) -> jax.Array:\n        return self.weight * self.loss_fn(predictions, targets)\n\nclass MultiTaskLoss(eqx.Module):\n    losses: tuple\n\n    def __call__(\n        self,\n        predictions: tuple,\n        targets: tuple,\n    ) -> jax.Array:\n        assert len(predictions) == len(targets) == len(self.losses), (\n            f"MultiTaskLoss: length mismatch predictions={len(predictions)}, "\n            f"targets={len(targets)}, losses={len(self.losses)}"\n        )\n        # Static-length tuple comprehension — unrolls at trace time, NOT a hot loop\n        return jnp.sum(jnp.stack([l(p, t) for l, p, t in zip(self.losses, predictions, targets)]))\n\`\`\`\n\nWrite tests: WeightedLoss == weight*loss; MultiTaskLoss == sum of weighted; both pass isinstance(LossFunction); tuple-length mismatch raises AssertionError.\n\nGate: \`uv run pytest tests/training/test_loss.py -v\` pass; ruff clean.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify loss combinators:\n1. WeightedLoss(weight=2.0, loss_fn): output == 2.0 * loss_fn(p, t)\n2. MultiTaskLoss: output == sum of each weighted loss applied to corresponding (p, t) pair\n3. Both pass isinstance(fn, LossFunction)\n4. len mismatch raises AssertionError\n5. Implementation uses list comprehension with zip (NOT jax.tree.map — wrong structure)\n6. All tests pass; ruff clean\n`,
  );

// ===== TRACK K — Track K — Phase 4.6: accumulate_grads (#1146) =========================
const trackK = () =>
  track(
    "1146",
    "Track K — Phase 4.6: accumulate_grads (#1146)",
    `task_id: ${TASK_ID}. task_id: 260604_xtrax-shape\nFile: src/xtrax/training/grad.py | Test: tests/training/test_grad.py\n\nImplement accumulate_grads in src/xtrax/training/grad.py using jax.lax.scan — NO Python for-loop:\n\n\`\`\`python\nfrom collections.abc import Callable\nfrom typing import Any\nimport jax\nimport jax.numpy as jnp\nimport equinox as eqx\n\ndef accumulate_grads(\n    loss_fn: Callable,\n    params: Any,\n    microbatches: list,\n    filter_spec: Any = None,\n) -> tuple[Any, jax.Array]:\n    if filter_spec is None:\n        filter_spec = eqx.is_array\n\n    # Validate all microbatches are equal size\n    sizes = [jax.tree.leaves(mb)[0].shape[0] for mb in microbatches]\n    if len(set(sizes)) > 1:\n        raise ValueError(\n            f"accumulate_grads: all microbatches must be equal size, got sizes={sizes}"\n        )\n\n    # Stack microbatches into a leading-axis pytree for lax.scan\n    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *microbatches)\n\n    def scan_fn(carry, microbatch):\n        loss, grads = eqx.filter_value_and_grad(loss_fn)(params, microbatch)\n        return carry, (grads, loss)\n\n    _, (all_grads, all_losses) = jax.lax.scan(scan_fn, None, stacked)\n\n    # Mean over microbatches\n    mean_grads = jax.tree.map(lambda g: jnp.mean(g, axis=0), all_grads)\n    mean_loss = jnp.mean(all_losses)\n    return mean_grads, mean_loss\n\`\`\`\n\nWrite tests: lax.scan appears in jaxpr (use jax.make_jaxpr); atol=1e-5 vs full-batch with 4 equal microbatches of size 25; unequal sizes raises ValueError.\n\nGate: \`uv run pytest tests/training/test_grad.py -v\` pass; ruff clean.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify accumulate_grads:\n1. Implementation uses jax.lax.scan — verify via jax.make_jaxpr that 'scan' appears (NO Python for-loop)\n2. With 4 equal microbatches of size 25: result matches full-batch grad within atol=1e-5\n3. Unequal microbatch sizes raises ValueError\n4. Returns (mean_grads, mean_loss)\n5. All tests pass; ruff clean\n`,
  );

// ===== TRACK L — Track L — Phase 4.3: SafetyTrainStep, create_train_step (#1143) =========================
const trackL = () =>
  track(
    "1143",
    "Track L — Phase 4.3: SafetyTrainStep, create_train_step (#1143)",
    `task_id: ${TASK_ID}. task_id: 260604_xtrax-shape\nFile: src/xtrax/training/step.py | Test: tests/training/test_step.py\n\nImplement in src/xtrax/training/step.py:\n\n\`\`\`python\nfrom typing import Any\nimport jax\nimport equinox as eqx\nimport optax\nfrom xtrax.training.types import LossFunction, ResumableState\nfrom xtrax.training.trainer import Trainer\nfrom xtrax.safety.manager import SafetyManager, with_safety\n\nclass SafetyTrainStep(eqx.Module):\n    _trainer: Trainer\n    _mgr: SafetyManager\n\n    def step(self, state: ResumableState, batch: Any) -> tuple[ResumableState, jax.Array]:\n        safe_step = with_safety(self._trainer.step, self._mgr)\n        return safe_step(state, batch)\n\ndef create_train_step(\n    loss_fn: LossFunction,\n    optimizer: optax.GradientTransformation,\n    safety: bool = False,\n) -> Trainer | SafetyTrainStep:\n    trainer = Trainer(loss_fn=loss_fn, optimizer=optimizer)\n    if not safety:\n        return trainer\n    return SafetyTrainStep(_trainer=trainer, _mgr=SafetyManager(enabled=True))\n\`\`\`\n\nWrite tests: create_train_step(safety=False) returns Trainer; create_train_step(safety=True) returns SafetyTrainStep; both expose .step(state, batch) signature; SafetyTrainStep detects NaN in loss.\n\nGate: \`uv run pytest tests/training/test_step.py -v\` pass; ruff clean.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify create_train_step:\n1. safety=False → returns Trainer instance\n2. safety=True → returns SafetyTrainStep instance\n3. Both have .step(state, batch) → (state, loss) signature\n4. SafetyTrainStep with NaN-producing loss raises on host\n5. All tests pass; ruff clean\n`,
  );

// ---- orchestrate: writing chain (A -> B -> C -> D -> E -> F -> G -> H -> I -> J -> K -> L, sequential) ----
log("xtrax Sprint 2: Safety + Training Core (Phase 3-4): writing chain (A -> B -> C -> D -> E -> F -> G -> H -> I -> J -> K -> L, sequential)");
const a = await trackA();
const b = await trackB();
const c = await trackC();
const d = await trackD();
const e = await trackE();
const f = await trackF();
const g = await trackG();
const h = await trackH();
const i = await trackI();
const j = await trackJ();
const k = await trackK();
const l = await trackL();

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
    "1143": l
  },
};
