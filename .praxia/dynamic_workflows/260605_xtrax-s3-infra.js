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
  description: "ShardingPolicy (eqx.Module, rules as tuple, get_partition_spec), init_dist (spec-aligned signature), Orbax checkpoint wrappers (modern API, no PyTreeCheckpointer), async_indexed_stream (prefetch + buffer_size), BoundedCallbackHandler (coroutine-based, exception logging), Engine (eqx.Module fields, DataModule data param, asyncio-first)",
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
const EMITTER_CTX = `xtrax sprint 3. Sprints 1 and 2 must be complete before this sprint.\nSpec: .praxia/docs/specs/260604_xtrax-spec.md (R3, oracle-approved). Sections: §3.18 Engine, §3.21 sharding, §3.22 init_dist, §3.26 checkpoint, §3.27 io callbacks.\ntask_id: 260605_xtrax-s3-infra\nKey rules: no Python-loop hot paths; uv run pytest; ruff clean before commit.\nFor Engine: use eqx.nn.inference_mode (verify exact API via Context7 against equinox ≥0.11 — could be eqx.nn.inference_mode(model) or eqx.tree_inference); asyncio.run for fit_sync.\nFor Orbax: ALWAYS call manager.wait_until_finished() after every save. VERIFY the correct orbax ≥0.6 API via Context7 before implementing — do NOT use ocp.PyTreeCheckpointer (deprecated ≥0.4).\nModule paths: repo layout (engine/io.py, distributed/init.py, distributed/sharding.py, checkpoint/orbax.py, engine/engine.py) is authoritative; spec §2 paths are stale.\n`;

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

// ===== TRACK A — Phase 5.1: ShardingPolicy, mesh helpers (#1148) =========================
const trackA = () =>
  track(
    "1148",
    "Track A — Phase 5.1: ShardingPolicy, mesh helpers (#1148)",
    `task_id: 260605_xtrax-s3-infra
File: src/xtrax/distributed/sharding.py | Test: tests/distributed/test_sharding.py

Read spec §3.21 for the authoritative interface. Key corrections vs prior draft:

1. ShardingPolicy MUST be an eqx.Module (not a dataclass).
   - Field: rules: tuple[tuple[str, PartitionSpec], ...] = eqx.field(static=True)
     (tuple of (regex_pattern_str, PartitionSpec) pairs; first match wins)
   - Method: get_partition_spec(self, path: str) -> PartitionSpec
     (NOT get_spec — the method is named get_partition_spec)
   - Method: apply_to_pytree(self, pytree) — traverse pytree leaves with jax.tree.map,
     returning same-structure pytree where each leaf is the PartitionSpec for that leaf's
     path (derive path from jax.tree_util.GetAttrKey / DictKey / etc.), or PartitionSpec()
     for unmatched leaves.

2. get_device_mesh MUST take (shape: tuple[int, ...], axis_names: tuple[str, ...]) -> Mesh.
   The prior version auto-generated d0,d1,... axis names — this is wrong; callers supply names.
   Validate: math.prod(shape) == len(jax.devices()), raise ValueError if not.

3. get_hardware_mesh_profile() — NO arguments (not device_type="cpu").
   Return exactly these keys:
     device_type: str  (e.g. "cpu", "gpu", "tpu")
     num_devices: int
     recommended_shape: tuple[int, ...]  (e.g. (1,) for single device, (8,) for 8 GPUs)
     recommended_axis_names: tuple[str, ...]  (e.g. ("batch",) for single-device; ("data",) for 8-GPU data-parallel)
   Never raises — CPU single-device fallback: shape=(1,), axis_names=("batch",).
   Do NOT return device_ids.

4. Update src/xtrax/distributed/__init__.py to re-export:
   ShardingPolicy, get_device_mesh, get_hardware_mesh_profile

DEVIATION NOTE (spec §2 vs repo): spec §2 may list a different module path; repo layout is authoritative.
Use jax.sharding.Mesh, jax.sharding.NamedSharding, jax.sharding.PartitionSpec.

Write tests:
- First-match-wins: ShardingPolicy with overlapping patterns, verify get_partition_spec returns correct spec
- get_device_mesh with matching shape/axis_names: returns Mesh with correct axis names
- get_device_mesh with wrong product: raises ValueError
- get_hardware_mesh_profile: returns all 4 required keys, no KeyError; never raises
- apply_to_pytree: simple dict pytree returns same-structure pytree of PartitionSpecs

Gate: uv run pytest tests/distributed/test_sharding.py -v pass; ruff clean.

${EMITTER_CTX}`,
    `task_id: 260605_xtrax-s3-infra. Verify ShardingPolicy and mesh helpers:
1. ShardingPolicy is an eqx.Module (not a dataclass). Field 'rules' is a tuple (not list) marked eqx.field(static=True). Rules hold (str, PartitionSpec) pairs.
2. Method is named get_partition_spec(self, path: str) (not get_spec). First-match-wins verified with overlapping patterns.
3. apply_to_pytree(self, pytree) exists and returns same-structure pytree of PartitionSpecs.
4. get_device_mesh(shape, axis_names) takes BOTH params. Axis names come from the parameter, not auto-generated. ValueError on shape/device count mismatch.
5. get_hardware_mesh_profile() takes NO args. Returns keys: device_type, num_devices, recommended_shape (tuple), recommended_axis_names (tuple). Does NOT return device_ids. Never raises.
6. src/xtrax/distributed/__init__.py re-exports ShardingPolicy, get_device_mesh, get_hardware_mesh_profile.
7. All tests pass; ruff clean.
`,
  );

// ===== TRACK B — Phase 5.2: init_dist — idempotent distributed initializer (#1149) =========================
const trackB = () =>
  track(
    "1149",
    "Track B — Phase 5.2: init_dist — idempotent distributed initializer (#1149)",
    `task_id: 260605_xtrax-s3-infra
File: src/xtrax/distributed/init.py | Test: tests/distributed/test_init.py

Read spec §3.22 for the authoritative interface. Key corrections vs prior draft:

1. init_dist parameter order MUST match spec §3.22:
   init_dist(coordinator_address=None, num_processes=None, process_id=None) -> None
   The prior draft had num_processes first — this is wrong.

2. After ANY successful initialization (including single-process fallback), call:
   from xtrax.data.module import _mark_dist_initialized
   _mark_dist_initialized()
   This is the Sprint 2 integration seam — without it, DataModule(distributed=True).train_iter() raises RuntimeError.

3. SLURM env discovery: also read SLURM_JOB_NODELIST to derive coordinator address when num_processes > 1 and coordinator_address is None. Pattern:
   nodelist = os.environ.get("SLURM_JOB_NODELIST", "")
   if nodelist: coordinator_address = derive_coordinator(nodelist)  # extract first node + port 1234

4. DEVIATION NOTE (spec §3.22 vs implementation): spec says localhost fallback CALLS jax.distributed.initialize with single-process config. However, calling jax.distributed.initialize with num_processes=1 can be unstable in some environments. Implementing as: skip the call for num_processes==1, call _mark_dist_initialized() and return. If future requirements need the single-process initialize call, this can be made a parameter. Document this deviation in a comment.

5. Idempotency: same args → no-op; different args → RuntimeError (already present in draft, keep).

6. Add _reset_for_testing() helper (keep from draft).

7. Update src/xtrax/distributed/__init__.py to re-export: init_dist, is_distributed (a simple helper: return _STATE.initialized).

Write tests:
- Same args twice: no error
- Different args: RuntimeError
- SLURM env vars (SLURM_NTASKS, SLURM_PROCID) auto-detected when args=None (monkeypatch os.environ)
- num_processes=1: returns without jax.distributed.initialize call; _mark_dist_initialized was called (verify via DataModule)
- Param order: coordinator_address is first keyword arg (not third)

Gate: uv run pytest tests/distributed/test_init.py -v pass; ruff clean.

${EMITTER_CTX}`,
    `task_id: 260605_xtrax-s3-infra. Verify init_dist:
1. Signature: init_dist(coordinator_address=None, num_processes=None, process_id=None) — coordinator_address is FIRST keyword arg.
2. Same args called twice: no error, initialized only once.
3. Different args after init: RuntimeError.
4. SLURM env (SLURM_NTASKS, SLURM_PROCID) auto-detected when args=None. SLURM_JOB_NODELIST consulted for coordinator address.
5. num_processes=1: no jax.distributed.initialize call (documented deviation from spec §3.22). _mark_dist_initialized() IS called.
6. After any successful init: DataModule(distributed=True).train_iter() does NOT raise RuntimeError. (Integration seam test: import DataModule, init_dist, call init_dist(num_processes=1), then verify train_iter succeeds.)
7. src/xtrax/distributed/__init__.py re-exports init_dist.
8. All tests pass; ruff clean.
`,
  );

// ===== TRACK C — Phase 5.3: Checkpoint wrappers (#1150) =========================
const trackC = () =>
  track(
    "1150",
    "Track C — Phase 5.3: Checkpoint wrappers (#1150)",
    `task_id: 260605_xtrax-s3-infra
File: src/xtrax/checkpoint/orbax.py | Test: tests/checkpoint/test_orbax.py

Read spec §3.26 for the authoritative interface. The prior draft used deprecated APIs and wrong signatures.

CRITICAL: Before writing any orbax code, load the orbax docs via Context7:
  - resolve-library-id: "orbax checkpoint" then query-docs with the relevant function names
  - Use orbax ≥0.6 API — ocp.PyTreeCheckpointer is DEPRECATED since orbax 0.4. Do NOT use it.
  - The correct modern API uses ocp.CheckpointManager with appropriate item handlers.

Implement THREE functions (spec §3.26):

1. get_checkpoint_manager(directory: str | Path, max_to_keep: int = 5, keep_period: int | None = None) -> CheckpointManager
   Creates and returns a CheckpointManager for the given directory.
   Use the appropriate orbax ≥0.6 API (verify via Context7).

2. save_checkpoint(manager: CheckpointManager, state: ResumableState, step: int | None = None) -> None
   - Extracts step: step_int = step if step is not None else int(state.step)
   - Saves via manager using the orbax ≥0.6 save API
   - ALWAYS calls manager.wait_until_finished() after save

3. load_checkpoint(manager: CheckpointManager, state_template: ResumableState, step: int | None = None) -> ResumableState
   - state_template is REQUIRED — orbax needs it to reconstruct the ResumableState pytree structure
   - If step is None, use manager.latest_step()
   - Returns a ResumableState (not a raw dict)

DEVIATION NOTE (spec §3.26 vs runtime): state_template must be provided because orbax cannot reconstruct eqx.Module pytrees without a target structure. This is required behavior, not a workaround.

Update src/xtrax/checkpoint/__init__.py to re-export: get_checkpoint_manager, save_checkpoint, load_checkpoint.

Write tests (use tmp_path fixture for directories):
- Round-trip: get_checkpoint_manager → save_checkpoint → load_checkpoint → verify all ResumableState fields (step as int32 scalar, model weights numerically identical within dtype precision)
- load_checkpoint on empty/missing dir: appropriate error raised
- wait_until_finished() called after save (can verify by checking state survives crash simulation or by checking step count)
- step override: save_checkpoint(manager, state, step=42) saves at step 42

Gate: uv run pytest tests/checkpoint/test_orbax.py -v pass; ruff clean.

${EMITTER_CTX}`,
    `task_id: 260605_xtrax-s3-infra. Verify checkpoint wrappers:
1. Three functions exist: get_checkpoint_manager(directory, max_to_keep, keep_period), save_checkpoint(manager, state, step=None), load_checkpoint(manager, state_template, step=None). No ocp.PyTreeCheckpointer usage.
2. save_checkpoint takes manager as FIRST arg (not checkpoint_dir). load_checkpoint takes manager + state_template.
3. Round-trip: get_checkpoint_manager → save → load returns ResumableState with step identical and model leaf shapes/dtypes preserved.
4. load_checkpoint uses state_template for structural restore (not returning raw dict).
5. wait_until_finished() called inside save_checkpoint.
6. step override param works: save at specified step, load recovers it.
7. src/xtrax/checkpoint/__init__.py re-exports all three functions.
8. All tests pass; ruff clean.
`,
  );

// ===== TRACK D — Phase 6.1: async_indexed_stream, BoundedCallbackHandler (#1151) =========================
const trackD = () =>
  track(
    "1151",
    "Track D — Phase 6.1: async_indexed_stream, BoundedCallbackHandler (#1151)",
    `task_id: 260605_xtrax-s3-infra
File: src/xtrax/engine/io.py | Test: tests/engine/test_io.py

Read spec §3.27 for the authoritative interface. Key corrections vs prior draft:

DEVIATION NOTE (module path): spec §3.27 places these in xtrax.io.callbacks; repo has engine/io.py as the stub. Implement in engine/io.py (repo layout is authoritative). Note: data/pipeline.py has a minimal async_indexed_stream with no buffer_size — this is a SEPARATE function; do not modify it. The engine/io.py version is the full spec §3.27 implementation.

1. async_indexed_stream(iterable: Iterable[T], buffer_size: int = 2) -> AsyncIterator[tuple[int, T]]
   - Prefetches up to buffer_size items AHEAD in a background thread using asyncio.to_thread
   - Pattern: producer task fills an asyncio.Queue(maxsize=buffer_size); consumer yields from queue
   - Exception from background thread is re-raised on next yield (propagates to consumer)
   - Must actually prefetch — not just enumerate in a loop. A slow iterable should have items ready.
   PRIOR DRAFT had no prefetch and no buffer_size — that is wrong.

2. class BoundedCallbackHandler:
   __init__(self, max_concurrent: int = 4) -> None

   async def submit(self, coro: Coroutine) -> None
   - Takes a COROUTINE object (not a callable + args)
   - Semaphore MUST be acquired INSIDE the async task, NOT in submit itself
     (So that submit returns immediately; the task waits for a slot before running)
   - Exceptions in the coro are LOGGED (not propagated) — use logging.exception(...)
     This prevents one failing callback from cancelling the training loop.

   async def wait_all(self) -> None
   - Awaits all pending tasks

Update src/xtrax/engine/__init__.py to re-export: BoundedCallbackHandler (Engine will be in engine.py).

Write tests:
- async_indexed_stream: yields (0, item0), (1, item1), ... correctly
- async_indexed_stream: exception from iterable re-raised on next yield
- BoundedCallbackHandler: submit(coro) takes a coroutine, not a callable
- BoundedCallbackHandler: with max_concurrent=2, submitting 5 long-running coros never has >2 in flight simultaneously
- BoundedCallbackHandler: exception in coro is logged but wait_all() completes without raising
- wait_all: flushes all pending tasks

Gate: uv run pytest tests/engine/test_io.py -v pass; ruff clean.

${EMITTER_CTX}`,
    `task_id: 260605_xtrax-s3-infra. Verify async IO primitives:
1. async_indexed_stream(iterable, buffer_size=2) — accepts buffer_size param. Prefetches using asyncio.to_thread + asyncio.Queue (NOT a trivial enumerate loop). Yields (0, item0), (1, item1), ...
2. Exception from background iterable thread propagates to async consumer on next yield.
3. BoundedCallbackHandler.submit(coro) — takes a Coroutine object (not callable+args).
4. Semaphore bounds concurrent EXECUTION (not just submission): submit 4 coros with max_concurrent=2, verify at most 2 run simultaneously (use asyncio.Event or sleep to control timing).
5. Exception in coro: logged (via logging module) but NOT propagated — wait_all() completes without raising even if a submitted coro raises.
6. wait_all() flushes all pending tasks.
7. All tests pass; ruff clean.
`,
  );

// ===== TRACK E — Phase 6.2: Engine (#1152) =========================
const trackE = () =>
  track(
    "1152",
    "Track E — Phase 6.2: Engine (#1152)",
    `task_id: 260605_xtrax-s3-infra
File: src/xtrax/engine/engine.py | Test: tests/engine/test_engine.py

Read spec §3.18 for the authoritative interface. This track has the most critical corrections.

CRITICAL CORRECTIONS:

1. Engine MUST be class Engine(eqx.Module) — not a plain Python class.
   Fields (all static — they hold non-array Python objects):
     trainer: Trainer | SafetyTrainStep = eqx.field(static=True)
     callbacks: tuple[Callback, ...] = eqx.field(static=True)
     validation_callbacks: tuple[Callback, ...] = eqx.field(static=True)

   checkpoint_dir is NOT a field — it is a parameter to fit().
   BoundedCallbackHandler is NOT a field — create one per fit() call.

   IMPORTANT: eqx.Module requires passing all fields in __init__ or using default factories.
   For eqx.Module, define __init__ explicitly OR use default values:
     class Engine(eqx.Module):
         trainer: Trainer | SafetyTrainStep
         callbacks: tuple[Callback, ...] = eqx.field(default=(), static=True)
         validation_callbacks: tuple[Callback, ...] = eqx.field(default=(), static=True)

   IMPORTANT: engine must accept BOTH Trainer AND SafetyTrainStep — import both:
     from xtrax.training.trainer import Trainer
     from xtrax.training.step import SafetyTrainStep
     from xtrax.training.types import Callback, ResumableState, LossFunction
     from xtrax.data.module import DataModule

2. fit signature:
   async def fit(self, state: ResumableState, data: DataModule, num_epochs: int, checkpoint_dir: str | Path | None = None) -> ResumableState
   - data is DataModule — iterate via data.train_iter(), NOT a raw data_iter argument
   - state, metrics = self.trainer.step(state, batch)  ← metrics is a dict, NOT loss
   - cb.on_step_end(state, metrics)  ← pass metrics dict (matches Callback.on_step_end(state, metrics: dict[str, Array]))
   - StopIteration from data.train_iter() should be caught and treated as epoch end
   - For checkpoint: create manager once per fit call (not per epoch):
       if checkpoint_dir is not None:
           from xtrax.checkpoint.orbax import get_checkpoint_manager, save_checkpoint
           manager = get_checkpoint_manager(checkpoint_dir)
     Then after each epoch: save_checkpoint(manager, state) — wait_until_finished() is inside save_checkpoint.
   - Dispatch callbacks asynchronously via BoundedCallbackHandler.submit() instead of direct calls where possible. At minimum, on_step_end should be dispatched async. on_epoch_end can be sync or async (spec §3.18 is flexible on dispatch granularity for non-step hooks).
   - on_train_start, on_epoch_start, on_step_start, on_step_end, on_epoch_end, on_train_end must all fire.

3. eval signature:
   async def eval(self, state: ResumableState, data: DataModule, loss_fn: LossFunction | None = None) -> dict[str, Array]
   - MUST be async def (not plain def)
   - data is DataModule — iterate via data.eval_iter()
   - Verify the correct Equinox inference API via Context7 (equinox ≥0.11):
     the call is eqx.nn.inference_mode(model) — but confirm the exact name
   - Aggregate metrics via jax.tree.map(jnp.mean, jnp.stack(list_of_metrics))
   - If loss_fn is provided, compute loss on each batch and include in metrics
   - Fires validation_callbacks only (not self.callbacks)

4. fit_sync:
   def fit_sync(self, state, data, num_epochs, checkpoint_dir=None):
       return asyncio.run(self.fit(state, data, num_epochs, checkpoint_dir))
   Wraps fit only. eval_sync is NOT specified by spec — do not add it.

5. Update src/xtrax/engine/__init__.py to re-export Engine.

Write tests:
- fit over 2 epochs with a simple DataModule: state.step == num_batches * 2
- Callback hook firing order and count: on_train_start×1, on_epoch_start×num_epochs, on_step_start×(steps*epochs), on_step_end×(steps*epochs), on_epoch_end×num_epochs, on_train_end×1
- on_step_end receives a dict (not a scalar) — specifically state and metrics dict
- eval: state.step unchanged; model wrapped in inference_mode; returns dict[str, Array]
- eval fires validation_callbacks only
- fit_sync delegates to asyncio.run(self.fit(...))
- Engine accepts SafetyTrainStep as trainer (polymorphic)

DEVIATION NOTE (spec §3.18 eqx.Module + static fields): callbacks/validation_callbacks hold Python callback objects (not JAX arrays) so they MUST be marked static. If not, Equinox will try to trace them as pytree leaves and fail.

Gate: uv run pytest tests/engine/test_engine.py -v pass; ruff clean.

${EMITTER_CTX}`,
    `task_id: 260605_xtrax-s3-infra. Verify Engine:
1. Engine is class Engine(eqx.Module). Fields: trainer (Trainer|SafetyTrainStep), callbacks (tuple, static), validation_callbacks (tuple, static). No checkpoint_dir field. No BoundedCallbackHandler field.
2. fit(self, state, data: DataModule, num_epochs, checkpoint_dir=None) — takes DataModule, not data_iter. async def.
3. trainer.step() result is unpacked as (state, metrics) — metrics is a dict. on_step_end receives metrics dict, not a scalar. Verified by checking Callback.on_step_end was called with a dict.
4. eval(self, state, data: DataModule, loss_fn=None) is async def (not plain def). Takes DataModule.
5. Checkpoint uses get_checkpoint_manager + save_checkpoint(manager, state) — NOT save_checkpoint(state, dir). Manager created once per fit call.
6. Engine accepts SafetyTrainStep as trainer without error (polymorphic).
7. fit_sync wraps fit via asyncio.run. eval_sync is absent.
8. All 6 Callback hooks fire in correct order during fit: on_train_start×1, on_epoch_start×E, on_step_start×(S*E), on_step_end×(S*E), on_epoch_end×E, on_train_end×1.
9. eval fires validation_callbacks, not training callbacks.
10. src/xtrax/engine/__init__.py re-exports Engine.
11. All tests pass; ruff clean.
`,
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
