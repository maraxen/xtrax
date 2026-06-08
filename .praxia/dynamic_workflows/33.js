// Sprint 7 runner — emitted by `praxia dw emit-sprint`
// Source: .praxia/sprint_plans/sprint_plan.toml
// Regenerate: praxia dw emit-sprint sprint_plan.toml
// task_id: 33   sprint_id: 7
//
// RACE SAFETY (memory: parallel fixers race on git-status scope checks in praxia):
//   the writing chain (C) runs STRICTLY SEQUENTIAL —
//   exactly one fixer touches the working tree at a time. Only the read-only
//   research/concurrent tracks (A,B) run concurrently.

export const meta = {
  name: "33",
  description: "Enable sparse weight application at inference time via sparsify_model functional transform, BucketIterator minimal implementation with static bucket padding, and integration tests verifying no-retrace guarantees and Engine.eval composition.",
  phases: [
    { title: "Track C — Integration tests (#1355)" },
    { title: "Track A — sparsify_model inference transform (#1353)" },
    { title: "Track B — BucketIterator minimal implementation (#1354)" },
  ],
};

const TASK_ID = "33";
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

// Shared context for the writing tracks (from recon, task 33).
const EMITTER_CTX = `xtrax Sprint 7. Sprint 6 complete; main clean at 96.3% coverage.\nSpec: .praxia/docs/specs/260608_inference-time-sparsification-in-xtrax-h.md\ntask_id: 260608_sparse-inference-brainstorm\nKey rules: uv run pytest; ruff clean before commit; no bare python.\n\nCRITICAL CONSTRAINTS:\n1. apply_mask is NOT jit-safe; must run Python-side only (branches on jnp.sum(mask) at runtime)\n2. SparsePolicy.apply_mask only supports 2D weight tensors (uses indices[:, 0], indices[:, 1])\n3. BCOO with fixed nse_budget has static indices shape — do not recompute inside jit\n4. Use is_leaf=lambda x: isinstance(x, BCOO) in tree_flatten to treat BCOO as atomic leaves\n5. xtrax does NOT own sparse matmul — BCOO is a container, callers handle matmul\n6. SparseMaskManager must NOT be called inside jit — it mutates Python-side dicts\n7. Non-2D leaves (biases, etc.) must be skipped with a warnings.warn, not crashed on\n`;

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

// ===== TRACK C — Track C — Integration tests (#1355) =========================
const trackC = () =>
  track(
    "1355",
    "Track C — Integration tests (#1355)",
    `task_id: ${TASK_ID}. You are writing integration tests for xtrax Sprint 7 (Track C).\ntask_id: 260608_sparse-inference-brainstorm\nSpec: .praxia/docs/specs/260608_inference-time-sparsification-in-xtrax-h.md (AC-6, AC-9)\n\n=== PREREQUISITE ===\nTrack A (sparsify_model) and Track B (BucketIterator) must be complete before this track.\n\n=== CONTEXT ===\nAC-6: No-retrace guarantee — sparsify_model + make_sparse_forward_fn + eqx.filter_jit should compile once and not recompile across repeated calls with same-shaped inputs\nAC-9: Engine.eval composition — sparsify_model applied to a model before Engine.eval runs without error; sparse BCOO leaves survive eqx.nn.inference_mode\n\nEngine location: src/xtrax/engine/engine.py\nEngine.eval at line 138; uses eqx.nn.inference_mode(state.model) at line 169\n\n=== TASK ===\nCreate tests/sparse/test_integration.py:\n\ntest_no_retrace_across_calls():\n    import jax\n    from xtrax.sparse.inference import sparsify_model, make_sparse_forward_fn\n    from xtrax.sparse.policy import SparsePolicy\n    from xtrax.sparse.config import SparseConfig\n    import equinox as eqx\n\n    key = jax.random.PRNGKey(0)\n    model = eqx.nn.Linear(8, 8, key=key)\n    config = SparseConfig(nse_budget=32)\n    policy = SparsePolicy(config=config)\n\n    sparse_model = sparsify_model(model, policy)\n    fn = make_sparse_forward_fn(lambda m, x: m(x), sparse_model)\n\n    # Python-side mutation inside eqx.filter_jit only executes during tracing,\n    # not on cached compiled dispatch — this is a reliable retrace detector.\n    # counter > 1 means assumption A1 (BCOO static closure prevents retrace) is FALSE.\n    call_count = {"n": 0}\n    @eqx.filter_jit\n    def forward(x):\n        call_count["n"] += 1\n        return fn(x)\n\n    x = jax.numpy.ones((8,))\n    out1 = forward(x)\n    out2 = forward(x)\n    out3 = forward(x)\n\n    # call_count["n"] should be 1 (traced once) — Python-side counter only increments on trace\n    # NOTE: eqx.filter_jit traces on first call; subsequent calls with same shape skip trace\n    assert call_count["n"] == 1, f"Expected 1 trace, got {call_count['n']} — retrace detected"\n    assert out1.shape == out2.shape == out3.shape\n\ntest_engine_eval_with_sparse_model():\n    # Verify sparsify_model output survives eqx.nn.inference_mode without error\n    # This tests AC-9: composition with Engine.eval pattern\n    import jax\n    import equinox as eqx\n    from xtrax.sparse.inference import sparsify_model\n    from xtrax.sparse.policy import SparsePolicy\n    from xtrax.sparse.config import SparseConfig\n\n    key = jax.random.PRNGKey(42)\n    model = eqx.nn.Linear(4, 4, key=key)\n    config = SparseConfig(nse_budget=8)\n    policy = SparsePolicy(config=config)\n\n    sparse_model = sparsify_model(model, policy)\n\n    # Apply inference_mode (same as Engine.eval does at line 169)\n    inference_model = eqx.nn.inference_mode(sparse_model)\n\n    # Verify BCOO leaves survive the transformation\n    import jax.numpy as jnp\n    from jax.experimental.sparse import BCOO\n    leaves = jax.tree_util.tree_leaves(inference_model, is_leaf=lambda x: isinstance(x, BCOO))\n    assert any(isinstance(leaf, BCOO) for leaf in leaves), "BCOO leaves lost after inference_mode"\n\n    # Run a forward pass\n    x = jnp.ones((4,))\n    @eqx.filter_jit\n    def forward(m, x):\n        return m(x)\n\n    out = forward(inference_model, x)\n    assert out.shape == (4,), f"Expected (4,), got {out.shape}"\n\nRun: uv run pytest tests/sparse/test_integration.py -v\nLint: uv run ruff check tests/sparse/test_integration.py --fix\n\n=== ACCEPTANCE CRITERIA ===\nAC-6: eqx.filter_jit traces the forward function exactly once across 3 same-shape calls — PASS/FAIL\nAC-9: sparsify_model output can be passed through eqx.nn.inference_mode; BCOO leaves survive; forward pass executes — PASS/FAIL\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. You are verifying Track C of xtrax Sprint 7 (integration tests for AC-6 and AC-9).\ntask_id: 260608_sparse-inference-brainstorm\nSpec: .praxia/docs/specs/260608_inference-time-sparsification-in-xtrax-h.md\n\nRun: uv run pytest tests/sparse/test_integration.py -v\nRun: uv run ruff check tests/sparse/test_integration.py\n\nVerify each AC:\nAC-6: test_no_retrace_across_calls passes; trace counter is 1 after 3 calls — PASS/FAIL\nAC-9: test_engine_eval_with_sparse_model passes; BCOO leaves present after inference_mode; forward pass returns correct shape — PASS/FAIL\n\nStructural check:\n- is_leaf=lambda x: isinstance(x, BCOO) used in tree_leaves call for AC-9 assertion — PASS/FAIL\n- eqx.filter_jit used (not jax.jit directly) — PASS/FAIL\n\nRun full suite to detect regressions: uv run pytest tests/sparse/ -q\nReport any new failures outside test_integration.py as regressions.\n\nEmit verdict: PASS or NEEDS_WORK with specific failures.\n`,
  );

// ===== TRACK A — Track A — sparsify_model inference transform (#1353) =========================
const trackA = () =>
  track(
    "1353",
    "Track A — sparsify_model inference transform (#1353)",
    `task_id: ${TASK_ID}. You are implementing the inference-time sparsification transform for xtrax (Sprint 7, Track A).\ntask_id: 260608_sparse-inference-brainstorm\nSpec: .praxia/docs/specs/260608_inference-time-sparsification-in-xtrax-h.md (AC-1 through AC-8)\n\n=== CONTEXT ===\n- Existing sparse infrastructure: src/xtrax/sparse/policy.py, config.py, manager.py\n- SparsePolicy.apply_mask: NOT jit-safe; 2D-only; uses jnp.argwhere with fixed size\n- SparsePolicy.make_mask: argsort-based, returns exactly nse_budget True values\n- BCOO registration: jax.experimental.sparse.BCOO is a registered JAX pytree\n  → tree_flatten without is_leaf splits BCOO into data+indices arrays\n  → MUST use is_leaf=lambda x: isinstance(x, BCOO) to treat BCOO as atomic\n- apply_mask 2D constraint: indices[:, 0], indices[:, 1] → ndim==2 required\n- Non-2D leaves (1D biases): warn and skip, do not crash\n\n=== TASK ===\nCreate src/xtrax/sparse/inference.py with:\n\n1. assert_not_tracing(leaves: list) -> None\n   - Iterate leaves; if any isinstance(leaf, jax.core.Tracer), raise RuntimeError:\n     "sparsify_model cannot be called inside jax.jit — call it before jit compilation"\n\n2. sparsify_model(model: eqx.Module, policy: SparsePolicy, leaf_filter=eqx.is_array) -> eqx.Module\n   - leaves, treedef = jax.tree_util.tree_flatten(model, is_leaf=lambda x: isinstance(x, BCOO))\n   - assert_not_tracing(leaves)\n   - Guard: if any(isinstance(leaf, BCOO) for leaf in leaves): raise ValueError("model already contains BCOO leaves — double sparsification detected")\n   - For each leaf (leaf_filter is the OUTER inclusion gate; ndim==2 is the inner sparsification gate):\n     * If NOT leaf_filter(leaf): new_leaf = leaf  (excluded entirely; no warning)\n     * Elif leaf_filter(leaf) and hasattr(leaf, "ndim") and leaf.ndim == 2:\n         mask = policy.make_mask(leaf, step=0)\n         new_leaf = policy.apply_mask(leaf, mask)\n     * Elif leaf_filter(leaf) and hasattr(leaf, "ndim") and leaf.ndim != 2:\n         warnings.warn(f"sparsify_model: skipping non-2D leaf (ndim={leaf.ndim}) — apply_mask requires 2D weights; leaf preserved as dense", stacklevel=2)\n         new_leaf = leaf\n     * Else: new_leaf = leaf\n   - return jax.tree_util.tree_unflatten(treedef, new_leaves)\n\n3. make_sparse_forward_fn(fn: Callable, sparse_model: eqx.Module) -> Callable\n   - Returns def _forward(inputs): return fn(sparse_model, inputs)\n   - IMPORTANT: Add this comment in the implementation body:\n     # Closing over sparse_model keeps BCOO leaves out of eqx.filter_jit's is_array\n     # partition. BCOO is not a jax.Array — passed as an argument it would be\n     # destructured into data+indices; as a closure constant it is treated as static.\n     # This is the load-bearing mechanism for AC-6 (no-retrace guarantee).\n   - Purpose: the closure pattern prevents filter_jit from re-partitioning BCOO on each call.\n\nExports: add sparsify_model, make_sparse_forward_fn to src/xtrax/sparse/__init__.py __all__\n\nCreate tests/sparse/test_inference.py:\n- test_sparsify_model_basic: eqx.nn.Linear(4, 4, key=...), SparseConfig(nse_budget=8), SparsePolicy(config), call sparsify_model; assert any leaf is BCOO instance\n- test_sparsify_model_todense_equals_masked_weight (AC-1 numerical): for each BCOO leaf in sparsified model, assert jnp.allclose(bcoo_leaf.todense(), original_dense_leaf * mask, atol=1e-6); compute mask = policy.make_mask(original_leaf, step=0) to reconstruct expected value; use jax.tree_util.tree_leaves with is_leaf=lambda x: isinstance(x, BCOO) to enumerate both original and sparse models in tandem\n- test_sparsify_model_no_double_sparsification: call sparsify_model twice; assert second call raises ValueError\n- test_sparsify_model_skips_non2d: model with 1D param; assert UserWarning emitted for non-2D\n- test_assert_not_tracing_outside_jit: assert_not_tracing([jnp.array(1.0)]) passes (no tracer)\n- test_sparsify_model_jit_guard: use jax.make_jaxpr or eqx.filter_jit callback to confirm raises RuntimeError when called inside jit (use eqx.filter_jit + closure that calls sparsify_model internally)\n- test_make_sparse_forward_fn_runs: sparsify model, wrap with make_sparse_forward_fn, call with dummy input; assert output shape\n- test_sparsify_model_bcoo_is_leaf_guard: create model whose first leaf is already BCOO; assert ValueError on double-sparsification path\n- test_sparsify_model_custom_leaf_filter (AC-8 parametric): (a) call with leaf_filter=lambda x: False → assert NO BCOO leaves in result (all leaves passed through unchanged); (b) call with leaf_filter=lambda x: hasattr(x, "ndim") and x.ndim >= 2 → assert same BCOO leaves as default (only 2D weights sparsified; apply_mask 2D-only constraint means non-2D filtered leaves warn and skip)\n\nRun: uv run pytest tests/sparse/test_inference.py -x -q\nLint: uv run ruff check src/xtrax/sparse/inference.py --fix\n\n=== ACCEPTANCE CRITERIA ===\nAC-1 (presence): sparsify_model returns eqx.Module with ≥1 BCOO leaf for a model with ≥1 2D weight\nAC-1 (numerical): each BCOO leaf's .todense() == original_weight * policy.make_mask(weight, step=0) within atol=1e-6\nAC-2: sparsify_model raises RuntimeError when called inside jax.jit\nAC-3: Double sparsification raises ValueError immediately\nAC-7: Non-2D leaves emit UserWarning and are preserved as dense arrays\nAC-8 (parametric): custom leaf_filter honored as outer inclusion gate; leaf_filter=lambda x: False yields no BCOO leaves\nAC-8 (forward fn): make_sparse_forward_fn closes over sparse_model; calling it returns correct output shape\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. You are verifying Track A of xtrax Sprint 7 (sparsify_model inference transform).\ntask_id: 260608_sparse-inference-brainstorm\nSpec: .praxia/docs/specs/260608_inference-time-sparsification-in-xtrax-h.md\n\nRun: uv run pytest tests/sparse/test_inference.py -v\nRun: uv run ruff check src/xtrax/sparse/inference.py\n\nVerify each AC:\nAC-1 (presence): At least one BCOO leaf present after sparsify_model on a model with 2D weights — PASS/FAIL\nAC-1 (numerical): BCOO leaf .todense() == original_weight * mask within atol=1e-6 (float32) — PASS/FAIL\nAC-2: RuntimeError raised when sparsify_model called inside jit — PASS/FAIL\nAC-3: ValueError on double sparsification — PASS/FAIL\nAC-7: UserWarning emitted for non-2D leaves; leaves preserved as dense (not None, not crash) — PASS/FAIL\nAC-8 (parametric): sparsify_model(model, policy, leaf_filter=lambda x: False) → zero BCOO leaves — PASS/FAIL\nAC-8 (forward fn): make_sparse_forward_fn returns callable; output shape matches original model — PASS/FAIL\n\nCheck src/xtrax/sparse/__init__.py: sparsify_model and make_sparse_forward_fn in __all__ — PASS/FAIL\n\nStructural checks:\n- leaf_filter is the OUTER inclusion gate (checked before ndim==2 branch) — PASS/FAIL\n- is_leaf=lambda x: isinstance(x, BCOO) used in tree_flatten call — PASS/FAIL\n- assert_not_tracing iterates leaves before any sparsification — PASS/FAIL\n- apply_mask called only for leaves that pass leaf_filter AND have ndim==2 — PASS/FAIL\n- make_sparse_forward_fn has closure comment explaining BCOO/filter_jit partition avoidance — PASS/FAIL\n\nEmit verdict: PASS (all ACs green, ruff clean) or NEEDS_WORK (list specific failures).\n`,
  );

// ===== TRACK B — Track B — BucketIterator minimal implementation (#1354) =========================
const trackB = () =>
  track(
    "1354",
    "Track B — BucketIterator minimal implementation (#1354)",
    `task_id: ${TASK_ID}. You are implementing BucketIterator.__iter__ for xtrax (Sprint 7, Track B).\ntask_id: 260608_sparse-inference-brainstorm\nSpec: .praxia/docs/specs/260608_inference-time-sparsification-in-xtrax-h.md (AC-4, AC-5)\n\n=== CONTEXT ===\nFile: src/xtrax/tiling/iterator.py\nBucketIterator is at line 90. The class already has:\n  - __init__ validating len(batch_sizes) == len(boundaries) + 1\n  - fields: boundaries: list[int], batch_sizes: list[int], fn: Callable, xs: Any\n  - __iter__ stub at line 121: yields nothing (yield from [])\n\nDesign intent:\n- boundaries define static bucket sizes (e.g. [128, 256, 512])\n- batch_sizes[i] is the batch size for bucket i (len(boundaries)+1 entries)\n- For a given input, find the SMALLEST bucket whose boundary >= seq_len; pad up to that boundary\n- Yield (result, original_length_mask) so caller can trim padding\n- Static bucket sizes prevent XLA recompilation on length variation\n- bucket_idx >= len(boundaries) means input exceeds ALL buckets: raise ValueError\n- EXACT boundary (seq_len == boundary) IS valid: pad_amount==0, no error\n\n=== TASK ===\nAdd these imports to the TOP-LEVEL of src/xtrax/tiling/iterator.py (NOT inline in __iter__):\n  import bisect\n  import warnings\n  (jax and jax.numpy are already imported at the top of the file — do NOT re-import them)\n\nReplace the __iter__ stub in src/xtrax/tiling/iterator.py with:\n\ndef __iter__(self):\n    leaves = jax.tree_util.tree_leaves(self.xs)\n    if not leaves:\n        return\n\n    seq_len = leaves[0].shape[0]\n    # bisect_left finds the first boundary >= seq_len (exact match is valid, pad_amount=0)\n    bucket_idx = bisect.bisect_left(self.boundaries, seq_len)\n\n    if bucket_idx >= len(self.boundaries):\n        max_bucket = self.boundaries[-1]\n        warnings.warn(\n            f"BucketIterator: input length {seq_len} exceeds maximum bucket size {max_bucket}; "\n            "this input cannot be processed without recompilation risk",\n            stacklevel=2,\n        )\n        raise ValueError(\n            f"Input length {seq_len} exceeds maximum bucket size {max_bucket}. "\n            "Add a larger boundary or use a different iterator."\n        )\n\n    bucket_size = self.boundaries[bucket_idx]\n    pad_amount = bucket_size - seq_len\n\n    padded_xs = jax.tree_util.tree_map(\n        lambda x: jnp.pad(x, [(0, pad_amount)] + [(0, 0)] * (x.ndim - 1)),\n        self.xs,\n    )\n\n    original_length_mask = jnp.arange(bucket_size) < seq_len\n    result = self.fn(padded_xs)\n    yield (result, original_length_mask)\n\nBisect correctness check:\n  seq_len=100, boundaries=[128,256,512] → bisect_left=0 → bucket_size=128 ✓\n  seq_len=128, boundaries=[128,256,512] → bisect_left=0 → bucket_size=128, pad_amount=0 ✓\n  seq_len=300, boundaries=[128,256,512] → bisect_left=2 → bucket_size=512 ✓\n  seq_len=512, boundaries=[128,256,512] → bisect_left=2 → bucket_size=512, pad_amount=0 ✓\n  seq_len=600, boundaries=[128,256,512] → bisect_left=3 → 3>=3 → raise ValueError ✓\n\nAdd tests to tests/tiling/test_iterator.py (or create if missing):\n- test_bucket_iterator_pads_to_boundary: input len=100, boundaries=[128,256], batch_sizes=[8,4,2]; verify yielded result comes from fn called on shape-128 input; original_length_mask[:100] all True, [100:] all False\n- test_bucket_iterator_exact_boundary: input len=128, boundaries=[128,256], batch_sizes=[8,4,2]; assert pad_amount==0 (bucket_size=128, seq_len=128); result shape (128,...) correct\n- test_bucket_iterator_exact_max_boundary: input len=256, boundaries=[128,256], batch_sizes=[8,4,2]; assert NO ValueError raised; pad_amount==0; verify result shape\n- test_bucket_iterator_exceeds_max_raises: input len=600, boundaries=[128,256,512], batch_sizes=[4,2,1,1]; assert ValueError raised AND UserWarning emitted\n- test_bucket_iterator_empty_xs_yields_nothing: empty pytree ({}); assert no yields\n\nRun: uv run pytest tests/tiling/test_iterator.py -x -q -k bucket\nLint: uv run ruff check src/xtrax/tiling/iterator.py --fix\n\n=== ACCEPTANCE CRITERIA ===\nAC-4: BucketIterator pads input to the smallest bucket >= seq_len via bisect.bisect_left; yields (result, mask) tuples; exact-boundary inputs (pad_amount==0) are accepted without error\nAC-5: Inputs exceeding max boundary emit UserWarning AND raise ValueError (no silent fallback)\n\n\n${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. You are verifying Track B of xtrax Sprint 7 (BucketIterator implementation).\ntask_id: 260608_sparse-inference-brainstorm\nSpec: .praxia/docs/specs/260608_inference-time-sparsification-in-xtrax-h.md\n\nRun: uv run pytest tests/tiling/test_iterator.py -v -k bucket\nRun: uv run ruff check src/xtrax/tiling/iterator.py\n\nVerify each AC:\nAC-4: BucketIterator yields (result, original_length_mask) tuples; mask shape == bucket_size; mask[:seq_len] all True — PASS/FAIL\nAC-4: bisect.bisect_LEFT used (not bisect_right, not linear scan) — verify by reading the implementation — PASS/FAIL\nAC-4: Exact-boundary input (seq_len == boundary) accepted; pad_amount==0; no error raised — PASS/FAIL\nAC-5: Input exceeding max boundary: both UserWarning emitted AND ValueError raised — PASS/FAIL\nAC-5: Empty xs pytree: iterator yields nothing (no crash) — PASS/FAIL\n\nStructural checks:\n- bisect and warnings imported at MODULE TOP-LEVEL (not inline in __iter__) — PASS/FAIL\n- jnp.pad call uses [(0, pad_amount)] + [(0, 0)] * (x.ndim - 1) for leading-axis padding only — PASS/FAIL\n- fn called on padded_xs (not original xs) — PASS/FAIL\n\nEmit verdict: PASS or NEEDS_WORK with specific failures.\n`,
  );

// ---- orchestrate: A+B concurrent (independent files), then C (depends on both) ----------
// Phase 1: A and B touch disjoint files and can run concurrently.
log("xtrax Sprint 7 Phase 1: Track A (sparsify_model) and Track B (BucketIterator) — concurrent");
const [resA, resB] = await Promise.all([trackA(), trackB()]);

// Phase 2: C imports from xtrax.sparse.inference (Track A) and tests BucketIterator (Track B).
// Must wait for Phase 1 to complete before the reviewer can run pytest successfully.
log("xtrax Sprint 7 Phase 2: Track C (integration tests) — sequential after A+B");
const resC = await trackC();

return {
  task_id: TASK_ID,
  sprint_id: 7,
  verdicts: {
    "1353": resA,
    "1354": resB,
    "1355": resC,
  },
};
