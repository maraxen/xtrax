// Sprint 8 runner — emitted from .praxia/sprint_plans/34.toml
// task_id: 34   sprint_id: 8
//
// All three tracks are CONCURRENT — disjoint files, no shared state.
// A: tests/sparse/test_manager.py (logger.debug test)
// B: tests/io/test_callbacks.py (TestAsyncIndexedStream rename)
// C: src/xtrax/sparse/inference.py + tests/sparse/test_inference.py (sparse_filter_jit + BCOO trap)

export const meta = {
  name: "34",
  description: "Sprint 8 deferred polish: (A) logger.debug test for SparseMaskManager skipped leaves, (B) TestAsyncIndexedStream rename in legacy test file, (C) sparse_filter_jit utility and BCOO destructuring trap note.",
  phases: [
    { title: "Track A — logger.debug test (#1373)" },
    { title: "Track B — TestAsyncIndexedStream rename (#1374)" },
    { title: "Track C — sparse_filter_jit + BCOO trap (#1376)" },
  ],
};

const TASK_ID = "34";
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

const EMITTER_CTX = `Sprint 7 complete — 38/38 sparse tests green, main clean at 5124a8c.
task_id: 260608_sparse-inference-brainstorm
Key rules: uv run pytest; ruff clean before commit; no bare python.

CRITICAL CONSTRAINTS:
1. _path_str and logger.debug already exist in manager.py — the test is the only missing piece.
2. TestAsyncIndexedStream in tests/io/test_callbacks.py is the LEGACY file (imports from xtrax.io.callbacks).
   tests/engine/test_io.py is CANONICAL. Rename legacy only; do NOT touch canonical.
3. BCOO is not a jax.Array — eqx.filter_jit without is_leaf guard will silently destructure it.
   sparse_filter_jit must pass is_leaf=lambda x: isinstance(x, BCOO) to eqx.filter_jit.
`;

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

// ===== TRACK A — logger.debug test for SparseMaskManager (#1373) =========================
const trackA = () =>
  track(
    "1373",
    "Track A — logger.debug test (#1373)",
    `task_id: ${TASK_ID}. You are adding a test for logger.debug on skipped leaves in SparseMaskManager (xtrax Sprint 8, Track A).
task_id: 260608_sparse-inference-brainstorm

=== CONTEXT ===
File: src/xtrax/sparse/manager.py
- _path_str(path) defined at line 15 — already extracted
- SparseMaskManager.step() at line 28; logger.debug("SparseMaskManager: skipping leaf %s", path_str) at line 41
- A leaf is skipped when: path_filter returns False OR leaf lacks ndim OR leaf.ndim < 2
- logger = logging.getLogger(__name__) => module name is "xtrax.sparse.manager"

File: tests/sparse/test_manager.py — has tests for step() but no test for logger.debug.

=== TASK ===
Add to tests/sparse/test_manager.py:

1. test_step_logs_debug_for_skipped_1d_leaf(caplog):
   - Create a minimal pytree with one 1D array (jnp.ones((4,)) — bias) and one 2D array (jnp.ones((4,4)) — weight)
   - Create SparseMaskManager with SparsePolicy(SparseConfig(nse_budget=8))
   - Call manager.step(params, step=0) inside caplog.at_level(logging.DEBUG, logger="xtrax.sparse.manager")
   - Assert at least one debug record contains "skipping leaf"

2. test_step_logs_debug_when_path_filter_excludes_leaf(caplog):
   - Pytree dict: {"weight": jnp.ones((4,4)), "bias": jnp.ones((4,))}
   - path_filter=lambda p: "weight" in p (bias is excluded by filter)
   - Assert debug message emitted for excluded leaf

caplog usage:
    def test_...(caplog):
        with caplog.at_level(logging.DEBUG, logger="xtrax.sparse.manager"):
            manager.step(...)
        assert any("skipping leaf" in r.message for r in caplog.records)

Run: uv run pytest tests/sparse/test_manager.py -x -q -k "debug or skip"
Lint: uv run ruff check tests/sparse/test_manager.py --fix

=== ACCEPTANCE CRITERIA ===
AC: both new tests pass; logger.debug emitted for each skipped leaf.

${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. You are verifying Track A of xtrax Sprint 8 (logger.debug test for SparseMaskManager).
task_id: 260608_sparse-inference-brainstorm

Run: uv run pytest tests/sparse/test_manager.py -v -k "debug or skip"
Run: uv run ruff check tests/sparse/test_manager.py

Verify:
- test_step_logs_debug_for_skipped_1d_leaf passes — PASS/FAIL
- test_step_logs_debug_when_path_filter_excludes_leaf passes — PASS/FAIL
- caplog captures logger "xtrax.sparse.manager" at DEBUG level — PASS/FAIL
- Assertion checks message contains "skipping leaf" — PASS/FAIL

Run full sparse suite: uv run pytest tests/sparse/ -q
Emit verdict: PASS or NEEDS_WORK with specific failures.`,
  );

// ===== TRACK B — TestAsyncIndexedStream rename (#1374) =========================
const trackB = () =>
  track(
    "1374",
    "Track B — TestAsyncIndexedStream rename (#1374)",
    `task_id: ${TASK_ID}. You are renaming TestAsyncIndexedStream in tests/io/test_callbacks.py (xtrax Sprint 8, Track B).
task_id: 260608_sparse-inference-brainstorm

=== CONTEXT ===
Sprint 6 consolidated async_indexed_stream into src/xtrax/engine/io.py.
Two classes share the name TestAsyncIndexedStream:
  - tests/engine/test_io.py:8  — CANONICAL (imports from xtrax.engine.io)
  - tests/io/test_callbacks.py:10 — LEGACY (imports from xtrax.io.callbacks re-export)

The legacy file validates the old import path still works as a shim. Rename it to reflect that.

=== TASK ===
In tests/io/test_callbacks.py ONLY:
1. Rename class TestAsyncIndexedStream -> TestAsyncIndexedStreamLegacyCallbacksShim
2. Update class docstring: "Tests for async_indexed_stream imported via the legacy xtrax.io.callbacks shim path."
3. Do NOT change any test method names or logic.
4. Do NOT touch tests/engine/test_io.py.

Run: uv run pytest tests/io/test_callbacks.py tests/engine/test_io.py -v
Lint: uv run ruff check tests/io/test_callbacks.py --fix

=== ACCEPTANCE CRITERIA ===
AC: tests/io/test_callbacks.py has TestAsyncIndexedStreamLegacyCallbacksShim (not TestAsyncIndexedStream); all tests in both files pass.

${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. You are verifying Track B of xtrax Sprint 8 (TestAsyncIndexedStream rename).
task_id: 260608_sparse-inference-brainstorm

Run: uv run pytest tests/io/test_callbacks.py tests/engine/test_io.py -v
Run: uv run ruff check tests/io/test_callbacks.py

Verify:
- tests/io/test_callbacks.py: no class TestAsyncIndexedStream — PASS/FAIL
- tests/io/test_callbacks.py: class TestAsyncIndexedStreamLegacyCallbacksShim present — PASS/FAIL
- All tests in tests/io/test_callbacks.py pass — PASS/FAIL
- tests/engine/test_io.py unchanged; all tests pass — PASS/FAIL

Emit verdict: PASS or NEEDS_WORK with specific failures.`,
  );

// ===== TRACK C — sparse_filter_jit + BCOO trap note (#1376) =========================
const trackC = () =>
  track(
    "1376",
    "Track C — sparse_filter_jit + BCOO trap (#1376)",
    `task_id: ${TASK_ID}. You are adding sparse_filter_jit and a BCOO trap note to src/xtrax/sparse/inference.py (xtrax Sprint 8, Track C).
task_id: 260608_sparse-inference-brainstorm

=== CONTEXT ===
File: src/xtrax/sparse/inference.py — exports: assert_not_tracing, sparsify_model, make_sparse_forward_fn

The BCOO destructuring trap:
- BCOO is NOT a jax.Array — it is a pytree node containing .data (jax.Array) and .indices (jax.Array)
- eqx.filter_jit without is_leaf=lambda x: isinstance(x, BCOO) will descend into BCOO nodes,
  partitioning .data and .indices as separate traced arrays — SILENT, no error raised
- This defeats retrace prevention: BCOO reconstructed from dynamic arrays on every call
- make_sparse_forward_fn closure sidesteps this entirely (preferred pattern)
- sparse_filter_jit provides a safe alternative when model must be passed as a jit argument

=== TASK ===
1. Add module-level docstring to inference.py (before imports):
"""
Inference-time sparsification utilities for xtrax.

Provides:
  - sparsify_model: functional transform — converts 2D weight leaves to BCOO format
  - make_sparse_forward_fn: closure helper — keeps BCOO leaves out of eqx.filter_jit's
    argument partition (the recommended composition pattern)
  - assert_not_tracing: guard that raises if called inside jax.jit
  - sparse_filter_jit: drop-in for eqx.filter_jit that passes is_leaf=lambda x: isinstance(x, BCOO),
    safe to use when a sparsified model is passed as a jit argument rather than a closure

BCOO destructuring trap:
  eqx.filter_jit without an is_leaf guard will silently descend into BCOO nodes and partition
  .data and .indices as separate traced arrays on every call. This defeats retrace prevention.
  Use make_sparse_forward_fn (closure pattern) to avoid this entirely, OR use sparse_filter_jit
  when you need to pass the sparsified model as a jit argument.
"""

2. Add sparse_filter_jit after make_sparse_forward_fn:

def sparse_filter_jit(fn: Callable, **kwargs) -> Callable:
    # BCOO is not a jax.Array — without is_leaf, filter_jit descends into BCOO nodes
    # and treats .data/.indices as separate traced arrays, silently breaking retrace prevention.
    return eqx.filter_jit(fn, default=eqx.is_array,
                           is_leaf=lambda x: isinstance(x, BCOO), **kwargs)

3. Add sparse_filter_jit to __all__ in src/xtrax/sparse/__init__.py.

4. Add test to tests/sparse/test_inference.py:
test_sparse_filter_jit_does_not_destructure_bcoo():
    key = jax.random.PRNGKey(0)
    model = eqx.nn.Linear(4, 4, key=key)
    config = SparseConfig(nse_budget=8)
    policy = SparsePolicy(config=config)
    sparse_model = sparsify_model(model, policy)
    call_count = {"n": 0}

    @sparse_filter_jit
    def forward(m, x):
        call_count["n"] += 1
        return m(x)

    x = jnp.ones((4,))
    out1 = forward(sparse_model, x)
    out2 = forward(sparse_model, x)

    assert call_count["n"] == 1, f"Expected 1 trace, got {call_count['n']} — BCOO destructuring triggered retrace"
    assert out1.shape == (4,)

Run: uv run pytest tests/sparse/test_inference.py -x -q -k "filter_jit"
Lint: uv run ruff check src/xtrax/sparse/inference.py --fix

=== ACCEPTANCE CRITERIA ===
AC: sparse_filter_jit exported from xtrax.sparse; test traces once across 2 calls; module docstring present.

${EMITTER_CTX}`,
    `task_id: ${TASK_ID}. You are verifying Track C of xtrax Sprint 8 (sparse_filter_jit + BCOO trap note).
task_id: 260608_sparse-inference-brainstorm

Run: uv run pytest tests/sparse/test_inference.py -v -k "filter_jit"
Run: uv run ruff check src/xtrax/sparse/inference.py src/xtrax/sparse/__init__.py

Verify:
- inference.py has module-level docstring mentioning "BCOO destructuring trap" — PASS/FAIL
- sparse_filter_jit defined; passes is_leaf=lambda x: isinstance(x, BCOO) to eqx.filter_jit — PASS/FAIL
- sparse_filter_jit in __all__ in src/xtrax/sparse/__init__.py — PASS/FAIL
- test_sparse_filter_jit_does_not_destructure_bcoo: trace count == 1 — PASS/FAIL

Run full suite: uv run pytest tests/sparse/ -q — report regressions.
Emit verdict: PASS or NEEDS_WORK with specific failures.`,
  );

// ---- orchestrate: all three tracks concurrent (disjoint files) ----------
log("xtrax Sprint 8: Tracks A, B, C — all concurrent (disjoint files)");
const [resA, resB, resC] = await Promise.all([trackA(), trackB(), trackC()]);

return {
  task_id: TASK_ID,
  sprint_id: 8,
  verdicts: {
    "1373": resA,
    "1374": resB,
    "1376": resC,
  },
};
