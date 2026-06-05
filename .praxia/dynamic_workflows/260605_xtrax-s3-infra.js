// Sprint 3 runner — emitted by `praxia dw emit-sprint`
// Source: .praxia/sprint_plans/sprint_plan.toml
// Regenerate: praxia dw emit-sprint sprint_plan.toml
// task_id: 260605_xtrax-s3-infra   sprint_id: 3
//
// RACE SAFETY (memory: parallel fixers race on git-status scope checks in praxia):
//   the writing chain (A,B,C,D,E) runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time.

export const meta = {
  name: "260605_xtrax-s3-infra",
  description: "ShardingPolicy, init_dist, Orbax checkpoint wrappers, async_indexed_stream, BoundedCallbackHandler, Engine fit/eval",
  phases: [
    { title: "Track A — Phase 5.1: ShardingPolicy, mesh helpers (#1148)" },
    { title: "Track B — Phase 5.2: init_dist — idempotent distributed initializer (#1149)" },
    { title: "Track C — Phase 5.3: Checkpoint wrappers (#1150)" },
    { title: "Track D — Phase 6.1: async_indexed_stream, BoundedCallbackHandler (#1151)" },
    { title: "Track E — Phase 6.2: Engine (#1152)" },
  ],
};

const TASK_ID = "260605_xtrax-s3-infra";
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

// Shared context for the writing tracks (from recon, task 260605_xtrax-s3-infra).
const EMITTER_CTX = `xtrax sprint 3. Sprints 1 and 2 must be complete before this sprint.\nSpec: .praxia/docs/specs/260604_xtrax-spec.md (R3, oracle-approved).\ntask_id: 260604_xtrax-shape\nKey rules: no Python-loop hot paths; uv run pytest; ruff clean before commit.\nEngine uses eqx.nn.inference_mode for eval, asyncio.run for fit_sync.\nOrbax: always call wait_until_finished() after every save.\n`;

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

// ===== TRACK A — Track A — Phase 5.1: ShardingPolicy, mesh helpers (#1148) =========================
const trackA = () =>
  track(
    "1148",
    "Track A — Phase 5.1: ShardingPolicy, mesh helpers (#1148)",
    `task_id: ${TASK_ID}. task_id: 260604_xtrax-shape\nFile: src/xtrax/distributed/sharding.py | Test: tests/distributed/test_sharding.py\n\nImplement in src/xtrax/distributed/sharding.py:\n\n\`\`\`python\nimport re\nfrom dataclasses import dataclass, field\nfrom typing import Any\nimport jax\nfrom jax.sharding import PartitionSpec\n\n@dataclass\nclass ShardingPolicy:\n    # Rules: list of (regex_pattern, PartitionSpec) pairs.\n    # First match wins — order matters.\n    rules: list[tuple[str, PartitionSpec]] = field(default_factory=list)\n\n    def get_spec(self, path: str) -> PartitionSpec:\n        for pattern, spec in self.rules:\n            if re.search(pattern, path):\n                return spec\n        return PartitionSpec()  # default: replicated\n\ndef get_hardware_mesh_profile(device_type: str = "cpu") -> dict[str, Any]:\n    return {\n        "device_type": device_type,\n        "num_devices": len(jax.devices()),\n        "device_ids": [d.id for d in jax.devices()],\n    }\n\ndef get_device_mesh(mesh_shape: tuple[int, ...]) -> jax.sharding.Mesh:\n    devices = jax.devices()\n    total = 1\n    for s in mesh_shape:\n        total *= s\n    if total != len(devices):\n        raise ValueError(\n            f"get_device_mesh: mesh_shape product={total} != num_devices={len(devices)}"\n        )\n    import numpy as np\n    return jax.sharding.Mesh(np.array(devices).reshape(mesh_shape), axis_names=tuple(f"d{i}" for i in range(len(mesh_shape))))\n\`\`\`\n\nWrite tests: first-match-wins with overlapping patterns; CPU profile has required keys; shape mismatch raises ValueError.\n\nGate: \`uv run pytest tests/distributed/test_sharding.py -v\` pass; ruff clean.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify ShardingPolicy:\n1. First-match-wins: ShardingPolicy(rules=[("weight", PartSpec("d")), (".*", PartSpec())]) with path "encoder.weight" → PartSpec("d")\n2. get_hardware_mesh_profile returns dict with "device_type", "num_devices", "device_ids"\n3. get_device_mesh raises ValueError for wrong shape product\n4. All tests pass; ruff clean\n`,
  );

// ===== TRACK B — Track B — Phase 5.2: init_dist — idempotent distributed initializer (#1149) =========================
const trackB = () =>
  track(
    "1149",
    "Track B — Phase 5.2: init_dist — idempotent distributed initializer (#1149)",
    `task_id: ${TASK_ID}. task_id: 260604_xtrax-shape\nFile: src/xtrax/distributed/init.py | Test: tests/distributed/test_init.py\n\nImplement idempotent init_dist in src/xtrax/distributed/init.py:\n\n\`\`\`python\nimport os\nfrom dataclasses import dataclass\nfrom typing import Any\n\n@dataclass\nclass _DistState:\n    initialized: bool = False\n    config: dict[str, Any] | None = None\n\n_STATE = _DistState()\n\ndef init_dist(\n    num_processes: int | None = None,\n    process_id: int | None = None,\n    coordinator_address: str | None = None,\n) -> None:\n    global _STATE\n    config = {\n        "num_processes": num_processes,\n        "process_id": process_id,\n        "coordinator_address": coordinator_address,\n    }\n    if _STATE.initialized:\n        if _STATE.config != config:\n            raise RuntimeError(\n                f"init_dist: already initialized with different config. "\n                f"Previous: {_STATE.config}, New: {config}"\n            )\n        return  # idempotent same args\n    # Auto-detect SLURM environment\n    if num_processes is None:\n        num_processes = int(os.environ.get("SLURM_NTASKS", "1"))\n    if process_id is None:\n        process_id = int(os.environ.get("SLURM_PROCID", "0"))\n    if num_processes == 1:\n        _STATE.initialized = True\n        _STATE.config = config\n        return  # localhost single-process fallback\n    import jax\n    jax.distributed.initialize(\n        coordinator_address=coordinator_address or "localhost:1234",\n        num_processes=num_processes,\n        process_id=process_id,\n    )\n    _STATE.initialized = True\n    _STATE.config = config\n\ndef _reset_for_testing() -> None:\n    global _STATE\n    _STATE = _DistState()\n\`\`\`\n\nWrite tests: idempotent same args; different args raises RuntimeError; SLURM env mocked (monkeypatch); localhost fallback (num_processes=1).\n\nGate: \`uv run pytest tests/distributed/test_init.py -v\` pass; ruff clean.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify init_dist:\n1. Same args called twice: no error, initialized only once\n2. Different args: raises RuntimeError\n3. SLURM env (SLURM_NTASKS, SLURM_PROCID) used when args=None\n4. num_processes=1: localhost fallback, no jax.distributed.initialize call\n5. All tests pass; ruff clean\n`,
  );

// ===== TRACK C — Track C — Phase 5.3: Checkpoint wrappers (#1150) =========================
const trackC = () =>
  track(
    "1150",
    "Track C — Phase 5.3: Checkpoint wrappers (#1150)",
    `task_id: ${TASK_ID}. task_id: 260604_xtrax-shape\nFile: src/xtrax/checkpoint/orbax.py | Test: tests/checkpoint/test_orbax.py\n\nImplement save/load checkpoint wrappers in src/xtrax/checkpoint/orbax.py:\n\n\`\`\`python\nfrom pathlib import Path\nfrom typing import Any\nimport orbax.checkpoint as ocp\nfrom xtrax.training.types import ResumableState\n\ndef save_checkpoint(state: ResumableState, checkpoint_dir: str | Path, step: int | None = None) -> None:\n    # Must be called Python-side (outside JIT): int(state.step) would fail inside trace\n    step_int = int(state.step)\n    if step is not None:\n        step_int = step\n    checkpointer = ocp.CheckpointManager(str(checkpoint_dir), ocp.PyTreeCheckpointer())\n    checkpointer.save(step_int, state)\n    checkpointer.wait_until_finished()  # always wait — ensures durability\n\ndef load_checkpoint(checkpoint_dir: str | Path) -> ResumableState:\n    checkpoint_path = Path(checkpoint_dir)\n    if not checkpoint_path.exists() or not any(checkpoint_path.iterdir()):\n        raise FileNotFoundError(f"No checkpoint found at {checkpoint_dir}")\n    checkpointer = ocp.CheckpointManager(str(checkpoint_dir), ocp.PyTreeCheckpointer())\n    step = checkpointer.latest_step()\n    return checkpointer.restore(step)\n\`\`\`\n\nWrite tests: save/load round-trip preserves all ResumableState fields (step, key, model shapes); empty dir raises FileNotFoundError.\n\nGate: \`uv run pytest tests/checkpoint/test_orbax.py -v\` pass; ruff clean.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify checkpoint wrappers:\n1. save_checkpoint + load_checkpoint round-trip: all ResumableState fields bit-identical within JAX dtype precision\n2. load_checkpoint on empty/missing dir raises FileNotFoundError\n3. wait_until_finished() called after every save\n4. save_checkpoint uses int(state.step) — would fail inside JIT by design\n5. All tests pass; ruff clean\n`,
  );

// ===== TRACK D — Track D — Phase 6.1: async_indexed_stream, BoundedCallbackHandler (#1151) =========================
const trackD = () =>
  track(
    "1151",
    "Track D — Phase 6.1: async_indexed_stream, BoundedCallbackHandler (#1151)",
    `task_id: ${TASK_ID}. task_id: 260604_xtrax-shape\nFile: src/xtrax/engine/io.py | Test: tests/engine/test_io.py\n\nImplement in src/xtrax/engine/io.py:\n\n\`\`\`python\nimport asyncio\nfrom collections.abc import AsyncIterator, Callable, Iterable\nfrom typing import Any, TypeVar\n\nT = TypeVar("T")\n\nasync def async_indexed_stream(iterable: Iterable[T]) -> AsyncIterator[tuple[int, T]]:\n    # Exception from iterable re-raised on next yield\n    try:\n        for i, item in enumerate(iterable):\n            yield i, item\n    except Exception:\n        raise  # propagate to consumer\n\nclass BoundedCallbackHandler:\n    def __init__(self, max_concurrent: int = 4) -> None:\n        self._semaphore = asyncio.Semaphore(max_concurrent)\n        self._tasks: list[asyncio.Task] = []\n\n    async def submit(self, callback: Callable, *args: Any) -> None:\n        async with self._semaphore:\n            task = asyncio.create_task(self._run(callback, *args))\n            self._tasks.append(task)\n\n    async def _run(self, callback: Callable, *args: Any) -> None:\n        if asyncio.iscoroutinefunction(callback):\n            await callback(*args)\n        else:\n            callback(*args)\n\n    async def wait_all(self) -> None:\n        if self._tasks:\n            await asyncio.gather(*self._tasks)\n            self._tasks.clear()\n\`\`\`\n\nWrite tests: async_indexed_stream yields correct (index, item) pairs; exception from iterable propagates; BoundedCallbackHandler bounds concurrency; wait_all flushes pending callbacks.\n\nGate: \`uv run pytest tests/engine/test_io.py -v\` pass; ruff clean.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify async IO primitives:\n1. async_indexed_stream yields (0, item0), (1, item1), ... correctly\n2. Exception from iterable re-raised on next yield (propagates to consumer)\n3. BoundedCallbackHandler: semaphore bounds concurrent callbacks to max_concurrent\n4. wait_all() flushes all pending tasks\n5. All tests pass; ruff clean\n`,
  );

// ===== TRACK E — Track E — Phase 6.2: Engine (#1152) =========================
const trackE = () =>
  track(
    "1152",
    "Track E — Phase 6.2: Engine (#1152)",
    `task_id: ${TASK_ID}. task_id: 260604_xtrax-shape\nFile: src/xtrax/engine/engine.py | Test: tests/engine/test_engine.py\n\nImplement Engine in src/xtrax/engine/engine.py:\n\n\`\`\`python\nimport asyncio\nfrom pathlib import Path\nfrom typing import Any\nimport equinox as eqx\nfrom xtrax.training.types import Callback, ResumableState\nfrom xtrax.training.trainer import Trainer\nfrom xtrax.engine.io import BoundedCallbackHandler\nfrom xtrax.checkpoint.orbax import save_checkpoint\n\nclass Engine:\n    def __init__(\n        self,\n        trainer: Trainer,\n        callbacks: list[Callback] | None = None,\n        checkpoint_dir: str | Path | None = None,\n        validation_callbacks: list[Callback] | None = None,\n    ) -> None:\n        self._trainer = trainer\n        self._callbacks = callbacks or []\n        self._validation_callbacks = validation_callbacks or []\n        self._checkpoint_dir = checkpoint_dir\n        self._handler = BoundedCallbackHandler()\n\n    async def fit(self, state: ResumableState, data_iter: Any, num_epochs: int) -> ResumableState:\n        for cb in self._callbacks:\n            cb.on_train_start(state)\n        for epoch in range(num_epochs):\n            for cb in self._callbacks:\n                cb.on_epoch_start(state, epoch)\n            for batch in data_iter:\n                for cb in self._callbacks:\n                    cb.on_step_start(state)\n                state, loss = self._trainer.step(state, batch)\n                for cb in self._callbacks:\n                    cb.on_step_end(state, loss)\n            for cb in self._callbacks:\n                cb.on_epoch_end(state, epoch)\n            if self._checkpoint_dir is not None:\n                save_checkpoint(state, self._checkpoint_dir)\n        for cb in self._callbacks:\n            cb.on_train_end(state)\n        await self._handler.wait_all()\n        return state\n\n    def eval(self, state: ResumableState, data_iter: Any, metric_fns: dict) -> dict:\n        # Wrap model in inference_mode for eval — disables dropout/stochastic layers\n        eval_state = eqx.tree_at(lambda s: s.model, state, eqx.nn.inference_mode(state.model))\n        all_metrics = {k: [] for k in metric_fns}\n        for cb in self._validation_callbacks:\n            cb.on_epoch_start(eval_state, 0)\n        for batch in data_iter:\n            predictions = eval_state.model(batch["inputs"])\n            for name, fn in metric_fns.items():\n                all_metrics[name].append(fn(predictions, batch["targets"]))\n        for cb in self._validation_callbacks:\n            cb.on_epoch_end(eval_state, 0)\n        import jax.numpy as jnp\n        return {k: jnp.mean(jnp.stack(v)) for k, v in all_metrics.items()}\n\n    def fit_sync(self, state: ResumableState, data_iter: Any, num_epochs: int) -> ResumableState:\n        return asyncio.run(self.fit(state, data_iter, num_epochs))\n\`\`\`\n\nWrite tests:\n- fit over 2 epochs: state.step == num_batches * 2\n- All 7 Callback hooks fired in correct order (on_train_start ×1, on_epoch_start ×2, on_step_start ×N, on_step_end ×N, on_epoch_end ×2, on_train_end ×1)\n- eval: returns metrics dict; state.step unchanged; wraps model in inference_mode\n- eval fires validation_callbacks only\n- fit_sync works via asyncio.run wrapper\n\nGate: \`uv run pytest tests/engine/test_engine.py -v\` pass; ruff clean.\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. Verify Engine:\n1. fit over 2 epochs: state.step == num_batches * num_epochs\n2. Callback order: on_train_start ×1, on_epoch_start ×num_epochs, on_step_start ×(steps_per_epoch*epochs), on_step_end ×same, on_epoch_end ×num_epochs, on_train_end ×1\n3. eval: state.step unchanged; model wrapped in eqx.nn.inference_mode; returns {metric_name: scalar}\n4. eval only fires validation_callbacks, not self._callbacks\n5. fit_sync delegates to asyncio.run(self.fit(...))\n6. Checkpoint written after each epoch when checkpoint_dir set\n7. All tests pass; ruff clean\n`,
  );

// ---- orchestrate: writing chain (A -> B -> C -> D -> E, sequential) ----
log("xtrax Sprint 3: Distributed + Engine (Phase 5-6): writing chain (A -> B -> C -> D -> E, sequential)");
const a = await trackA();
const b = await trackB();
const c = await trackC();
const d = await trackD();
const e = await trackE();

return {
  task_id: TASK_ID,
  sprint_id: 3,
  verdicts: {
    "1148": a,
    "1149": b,
    "1150": c,
    "1151": d,
    "1152": e
  },
};
