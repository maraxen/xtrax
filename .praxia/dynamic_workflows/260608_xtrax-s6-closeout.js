// Sprint 6 runner — hand-authored (dw emit blocked: templates not globally installed, backlog #983)
// task_id: 260608_xtrax-s6-closeout   sprint_id: 6
//
// Sprint 6 scope:
//   Track A (#1328): make_mask all-ties test + apply_mask (0,0)-aliasing test
//   Track B (#1326): Consolidate async_indexed_stream → engine/io.py
//   Track C (#1329): SparseMaskManager logger.debug + _path_str() helper
//
// RACE SAFETY: sequential tracks (A → B → C). No parallel fixer race risk.
// Oracle verdict (260608): approved sequential shape; spec_driven_dev rejected for Track A
// (spec already written inline); s5 track() pattern used throughout.

export const meta = {
  name: "260608_xtrax-s6-closeout",
  description: "Sprint 6 close-out: audit findings (#1328, #1329) + async consolidation (#1326).",
  phases: [
    { title: "Track A — make_mask all-ties + apply_mask aliasing tests (#1328)" },
    { title: "Track B — Consolidate async_indexed_stream → engine/io.py (#1326)" },
    { title: "Track C — SparseMaskManager logger.debug + _path_str() (#1329)" },
  ],
};

const TASK_ID = "260608_xtrax-s6-closeout";
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

const EMITTER_CTX = `xtrax Sprint 6. Sprint 5 complete; version 0.2.0 on main.\nAudit task_id: 260608_xtrax-s5-sparse; verdict was NEEDS_REVISION.\nKey rules: uv run pytest; ruff clean before commit.\nsparse/ lives at src/xtrax/sparse/; tests at tests/sparse/.\nPublic async API: src/xtrax/io/__init__.py re-exports async_indexed_stream and\nBoundedCallbackHandler from io/callbacks.py — consolidation must preserve this.\n`;

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

// ===== TRACK A — make_mask all-ties + apply_mask (0,0)-aliasing tests (#1328) ==============
const trackA = () =>
  track(
    "1328",
    "Track A — make_mask all-ties + apply_mask (0,0)-aliasing tests (#1328)",
    `task_id: 260608_xtrax-s6-closeout

Read before editing:
  src/xtrax/sparse/policy.py            (full file, 49 lines)
  tests/sparse/test_policy.py           (existing tests — do NOT remove any)

## Two edge-case tests to add to tests/sparse/test_policy.py

### 1. make_mask all-ties

make_mask (lines 22-28 of policy.py) uses jnp.argsort. When all abs values
are equal, argsort tie-breaking is implementation-defined. The invariant that
must hold regardless: exactly nse_budget True values in the output mask.

Add inside class TestSparsePolicy:

    def test_make_mask_all_ties_exact_budget(self):
        # All weights equal → tie-breaking is arbitrary but count must be exact.
        p = _make_policy(budget=4)
        w = jnp.ones((3, 3))          # 9 elements, all magnitude 1.0
        mask = p.make_mask(w, step=0)
        assert mask.shape == w.shape
        assert mask.dtype == jnp.bool_
        assert int(jnp.sum(mask)) == 4  # exactly nse_budget, not ≥

### 2. apply_mask (0,0)-aliasing with padding

apply_mask (lines 30-49 of policy.py) uses jnp.argwhere with fill_value=0.
When n_true < nse_budget, padded slots all point to index (0,0). Lines 46-48
zero those slots via the \`valid\` mask. The invariant: if mask[0,0] is False,
the (0,0) element in the dense output must be 0.0 — the padded alias must not
bleed the real weight value through.

Add inside class TestSparsePolicy:

    def test_apply_mask_padding_no_zero_zero_alias(self):
        # nse_budget=4, but only 2 True values in mask — 2 padding slots → (0,0).
        # weight[0,0] is non-zero but mask[0,0] is False.
        # The padded (0,0) alias must be zeroed, so todense()[0,0] == 0.0.
        p = _make_policy(budget=4)
        w = jnp.array([[5.0, 2.0, 3.0],
                        [0.0, 9.0, 0.0],
                        [0.0, 0.0, 7.0]])
        # Only (1,1) and (2,2) are True — 2 trues, 2 padding slots pointing to (0,0).
        mask = jnp.array([[False, False, False],
                           [False, True,  False],
                           [False, False, True]])
        result = p.apply_mask(w, mask)
        from jax.experimental.sparse import BCOO
        assert isinstance(result, BCOO)
        dense = result.todense()
        assert float(dense[0, 0]) == 0.0, f"(0,0) alias leaked: {dense[0,0]}"
        assert float(dense[1, 1]) == pytest.approx(9.0, abs=1e-6)
        assert float(dense[2, 2]) == pytest.approx(7.0, abs=1e-6)

Add \`import pytest\` to the test file imports if not already present.

Gate:
  uv run pytest tests/sparse/test_policy.py -v → all pass (including new tests)
  uv run pytest tests/ -q → full suite passes
  ruff check tests/sparse/ → clean

${EMITTER_CTX}`,
    `task_id: 260608_xtrax-s6-closeout. Verify #1328 edge-case tests:
1. test_make_mask_all_ties_exact_budget exists in TestSparsePolicy.
   - Inputs: jnp.ones((3,3)), budget=4.
   - int(jnp.sum(mask)) == 4 (exactly, not ≥ 4).
   - mask.shape == (3,3), mask.dtype == bool_.
2. test_apply_mask_padding_no_zero_zero_alias exists in TestSparsePolicy.
   - mask has 2 True values, budget=4 → 2 padding slots at (0,0).
   - weight[0,0] = 5.0 but mask[0,0] = False → dense[0,0] must be 0.0.
   - dense[1,1] ≈ 9.0, dense[2,2] ≈ 7.0.
3. No existing tests were removed or modified.
4. uv run pytest tests/sparse/test_policy.py -v → all pass.
5. uv run pytest tests/ -q → full suite passes, no regressions.
6. ruff check tests/sparse/ → clean.
`
  );

// ===== TRACK B — Consolidate async_indexed_stream → engine/io.py (#1326) ===================
const trackB = () =>
  track(
    "1326",
    "Track B — Consolidate async_indexed_stream → engine/io.py (#1326)",
    `task_id: 260608_xtrax-s6-closeout

Read before editing:
  src/xtrax/engine/io.py          (lines 1-116 — robust reference; KEEP AS-IS)
  src/xtrax/io/callbacks.py       (full file, 138 lines — buggy; replace body)
  src/xtrax/io/__init__.py        (lines 1-8 — public re-export; preserve API)
  src/xtrax/data/pipeline.py      (full file — stub to remove)
  src/xtrax/engine/engine.py      (line 23 — BoundedCallbackHandler import to repoint)

## Goal: single authoritative implementation in engine/io.py

The three copies:
  engine/io.py  — robust (asyncio.to_thread, proper _DONE/_ERROR sentinels)
  io/callbacks.py — buggy (producer has sync \`for item in iterator:\` at line 50,
                    blocking the event loop; uses queue._queue private API at line 51)
  data/pipeline.py — trivial enumerate stub (line 5 onward); not re-exported

Public contract (must not break):
  io/__init__.py:3 re-exports \`async_indexed_stream\` and \`BoundedCallbackHandler\`
  from \`xtrax.io.callbacks\` — downstream code that does
  \`from xtrax.io import async_indexed_stream\` must continue to work.

## Changes (edit only — no Write on existing files)

### 1. src/xtrax/io/callbacks.py — replace with thin re-export wrapper

Edit only. Replace the entire module body with:

\`\`\`python
"""Async IO utilities — thin re-export from canonical engine.io implementation."""

from xtrax.engine.io import BoundedCallbackHandler, async_indexed_stream

__all__ = ["async_indexed_stream", "BoundedCallbackHandler"]
\`\`\`

Keep the file; do not delete it (io/__init__.py imports from it).

### 2. src/xtrax/data/pipeline.py — remove async_indexed_stream stub

Edit only. If the file contains ONLY the async_indexed_stream stub and no other
symbols, replace the entire file body with:

\`\`\`python
"""Data pipeline utilities."""
\`\`\`

If data/pipeline.py has other exports, leave them intact and remove only
the async_indexed_stream function.

### 3. src/xtrax/engine/engine.py — repoint BoundedCallbackHandler import

Line 23 currently reads:
  from xtrax.io.callbacks import BoundedCallbackHandler

Change to:
  from xtrax.engine.io import BoundedCallbackHandler

### 4. src/xtrax/engine/io.py — no changes needed

### 5. Fix test regressions caused by the consolidation

After io/callbacks.py becomes a thin re-export and data/pipeline.py loses its stub,
two existing test modules break. Fix both as part of this track.

#### tests/data/test_pipeline.py — remove async_indexed_stream import and its test class

Line 3-6 currently imports async_indexed_stream from xtrax.data.pipeline.
After Track B removes that stub, collection fails with ImportError.

Edit only. Change the import block at lines 3-6 from:
  from xtrax.data.pipeline import (
      async_indexed_stream,
      create_distributed_pipeline,
  )
to:
  from xtrax.data.pipeline import create_distributed_pipeline

Remove the entire TestAsyncIndexedStream class (lines 9-19) — its subject no longer
lives at this path. The TestCreateDistributedPipeline class (lines 22-59) must remain
intact and unmodified.

#### tests/io/test_callbacks.py — fix logger name and log-message assertion

Line 73: caplog is scoped to logger="xtrax.io.callbacks". After consolidation, the
exception is logged by engine/io.py whose logger __name__ is "xtrax.engine.io".
Change line 73 from:
  with caplog.at_level(logging.ERROR, logger="xtrax.io.callbacks"):
to:
  with caplog.at_level(logging.ERROR, logger="xtrax.engine.io"):

Line 78: the current assertion is \`assert any("test error" in record.message ...)\`.
After consolidation, engine/io.py:154 logs \`logger.exception("callback error")\` —
the format string is "callback error", not the exception text. Change line 78 from:
  assert any("test error" in record.message for record in caplog.records)
to:
  assert any("callback error" in record.message for record in caplog.records)

Gate:
  uv run pytest tests/ -q → all pass (io/callbacks.py re-export chain intact)
  uv run python -c "from xtrax.io import async_indexed_stream, BoundedCallbackHandler; print('ok')"
  uv run python -c "from xtrax.engine.io import async_indexed_stream, BoundedCallbackHandler; print('ok')"
  ruff check src/xtrax/io/ src/xtrax/engine/ src/xtrax/data/ → clean

${EMITTER_CTX}`,
    `task_id: 260608_xtrax-s6-closeout. Verify #1326 consolidation:
1. Run: git diff HEAD -- src/xtrax/engine/io.py → empty (no changes to canonical source).
2. src/xtrax/io/callbacks.py is now a thin re-export wrapper — contains only imports
   from xtrax.engine.io and __all__; no function definitions remain.
3. src/xtrax/engine/engine.py:23 imports BoundedCallbackHandler from xtrax.engine.io.
4. src/xtrax/data/pipeline.py no longer defines async_indexed_stream.
5. Public API: uv run python -c "from xtrax.io import async_indexed_stream, BoundedCallbackHandler; print('ok')" → exits 0.
6. uv run pytest tests/ -q → all pass, no regressions.
7. ruff check src/xtrax/io/ src/xtrax/engine/ src/xtrax/data/ → clean.
`
  );

// ===== TRACK C — SparseMaskManager logger.debug + _path_str() helper (#1329) ===============
const trackC = () =>
  track(
    "1329",
    "Track C — SparseMaskManager logger.debug + _path_str() helper (#1329)",
    `task_id: 260608_xtrax-s6-closeout

Read before editing:
  src/xtrax/sparse/manager.py   (full file, 49 lines)

## Two cosmetic changes to src/xtrax/sparse/manager.py (edit only)

### 1. Extract _path_str() helper (DRY)

The path-string construction appears twice:
  Lines 30-32 (inside the update loop)
  Lines 39-41 (inside apply_leaf)

Both are:
  path_str = ".".join(p.key if hasattr(p, "key") else str(p) for p in path)

Add a module-level helper BEFORE the class definition:

    def _path_str(path) -> str:
        return ".".join(p.key if hasattr(p, "key") else str(p) for p in path)

Replace both inline constructions with \`_path_str(path)\`.

### 2. Add logger.debug for silently-skipped leaves

Add at module level after imports:
    import logging
    logger = logging.getLogger(__name__)

In the update branch (inside \`if should_update:\`), after computing path_str,
add an else branch that fires when the filter excludes a leaf or leaf.ndim < 2:

    for path, leaf in jax.tree_util.tree_leaves_with_path(params):
        path_str = _path_str(path)
        if path_filter(path_str) and hasattr(leaf, "ndim") and leaf.ndim >= 2:
            self._masks[path_str] = self.policy.make_mask(leaf, step)
        else:
            logger.debug("SparseMaskManager: skipping leaf %s", path_str)

No behavior changes — no new public methods, no signature changes.

Gate:
  uv run pytest tests/sparse/test_manager.py -v → all existing tests pass
  uv run pytest tests/ -q → full suite passes
  ruff check src/xtrax/sparse/manager.py → clean

${EMITTER_CTX}`,
    `task_id: 260608_xtrax-s6-closeout. Verify #1329 cosmetic cleanup:
1. _path_str() module-level helper defined before the class.
2. No inline path_str construction remains in step() — both sites use _path_str(path).
3. \`import logging\` and \`logger = logging.getLogger(__name__)\` present at module level.
4. Inside the update loop's else branch: logger.debug("SparseMaskManager: skipping leaf %s", path_str).
5. No behavior changes — no new public methods, no signature changes.
6. uv run pytest tests/sparse/test_manager.py -v → all pass.
7. uv run pytest tests/ -q → full suite passes, no regressions.
8. ruff check src/xtrax/sparse/manager.py → clean.
`
  );

log("xtrax Sprint 6: Audit close-out + async consolidation — Tracks A → B → C (sequential)");
const a = await trackA();
log(`[Track A #1328] verdict: ${a && a.verdict}`);
const b = await trackB();
log(`[Track B #1326] verdict: ${b && b.verdict}`);
const c = await trackC();
log(`[Track C #1329] verdict: ${c && c.verdict}`);

return {
  task_id: TASK_ID,
  sprint_id: 6,
  verdicts: { "1328": a, "1326": b, "1329": c },
};
