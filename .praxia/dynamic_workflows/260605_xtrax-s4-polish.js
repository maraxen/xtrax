// Sprint 4 runner — hand-authored (not emitted by dw emit)
// task_id: 260605_xtrax-s4-polish   sprint_id: 4
//
// Sprint 4 scope: Phase 7 Novel Additions
//   Task 7.1 (StageBundle __init_subclass__ validation) — ALREADY DONE in Sprint 2.
//              bundle.py lines 32-76 + test_bundle.py lines 193-224. No work needed.
//   Task 7.2 (MultiTaskLoss weight_schedule + semver 0.2.0) — this sprint's only track.
//
// RACE SAFETY: single-track workflow, no parallel fixer race risk.

export const meta = {
  name: "260605_xtrax-s4-polish",
  description: "MultiTaskLoss dynamic weight_schedule field (Callable[[int], Array] | None, step-applied); semver 0.1.0 → 0.2.0. Task 7.1 already complete.",
  phases: [
    { title: "Track A — Phase 7.2: MultiTaskLoss weight_schedule (#1153)" },
  ],
};

const TASK_ID = "260605_xtrax-s4-polish";
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

const EMITTER_CTX = `xtrax sprint 4. Sprints 1–3 complete.\nSpec: .praxia/docs/specs/260604_xtrax-spec.md §3.15, Phase 7 §5 Task 7.2.\ntask_id: 260605_xtrax-s4-polish\nKey rules: uv run pytest; ruff clean before commit. No Python-loop hot paths on data axes.\n`;

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

// ===== TRACK A — Phase 7.2: MultiTaskLoss weight_schedule (#1153) =========================
const trackA = () =>
  track(
    "1153",
    "Track A — Phase 7.2: MultiTaskLoss weight_schedule (#1153)",
    `task_id: 260605_xtrax-s4-polish
Files to modify:
  src/xtrax/training/loss.py
  tests/training/test_loss.py
  pyproject.toml

Read the current implementations first:
  src/xtrax/training/loss.py       (WeightedLoss + MultiTaskLoss as implemented in Sprint 2)
  tests/training/test_loss.py      (existing tests — must all still pass)
  pyproject.toml                   (current version = 0.1.0)

Read spec §3.15 and Phase 7 Task 7.2 in .praxia/docs/specs/260604_xtrax-spec.md.

## Task 7.2: MultiTaskLoss dynamic weight schedule

Add an optional dynamic weight schedule to MultiTaskLoss:

1. Add field to MultiTaskLoss (after the existing \`losses\` field):
     weight_schedule: Callable[[int], Array] | None = eqx.field(default=None, static=True)

   This field must be eqx.field(static=True) because it holds a Python callable (not a JAX
   array). A None value means "no schedule — use static weights as before."

2. Add required import at top of loss.py:
     from collections.abc import Callable

3. Modify MultiTaskLoss.__call__ to accept an optional step parameter:
     def __call__(
         self,
         predictions: tuple[PyTree, ...],
         targets: tuple[PyTree, ...],
         step: int = 0,
     ) -> Array:

   Inside __call__, after computing the sum of weighted losses:
     total = jnp.sum(jnp.stack([loss(pred, target) for loss, pred, target in zip(...)]))
     if self.weight_schedule is not None:
         total = total * self.weight_schedule(step)
     return total

   IMPORTANT: \`if self.weight_schedule is not None\` is a STATIC branch — weight_schedule
   is eqx.field(static=True), so this conditional is resolved at trace time, not JIT time.
   This is correct and does not break JAX tracing.

   DESIGN DECISION (spec is silent on HOW weight_schedule is applied):
   The spec defines the field type (Callable[[int], Array]) but does not specify the
   application semantics. The chosen interpretation is:
       total = total * self.weight_schedule(step)
   i.e., the schedule output is a scalar multiplier applied to the already-summed total loss.
   This is a deliberate design decision, not derived directly from spec. Document this
   choice with a one-line comment in the implementation:
       # weight_schedule returns a scalar multiplier; spec Task 7.2 is silent on semantics.
   The tests encode this same interpretation; if the interpretation changes, both must change.

4. Bump pyproject.toml version from "0.1.0" to "0.2.0".

## Constraints

- All existing tests in tests/training/test_loss.py MUST still pass unchanged.
  The step=0 default ensures backward compatibility.
- Do NOT change WeightedLoss at all.
- Do NOT change the LossFunction protocol in types.py.
- Existing MultiTaskLoss tests called as loss(predictions, targets) must still work.

## New tests to add in tests/training/test_loss.py

Add a test class or group of tests (after existing ones):

- test_weight_schedule_applied: Create MultiTaskLoss with weight_schedule=lambda step: jnp.array(2.0).
  Call loss(predictions, targets, step=0). Verify result == 2.0 * (sum of unscheduled loss).
- test_weight_schedule_step_dependent: weight_schedule=lambda step: jnp.array(float(step + 1)).
  Call with step=3. Verify result == 4.0 * unscheduled_loss.
- test_weight_schedule_none_unchanged: MultiTaskLoss with weight_schedule=None (default).
  Call with step=5. Verify result equals the same call without step (backward compat).
- test_weight_schedule_none_is_default: MultiTaskLoss(losses=...) with no weight_schedule kwarg.
  Verify loss.weight_schedule is None.

Gate: uv run pytest tests/training/test_loss.py -v pass; uv run pytest tests/ -v all pass; ruff clean on loss.py and test_loss.py.

${EMITTER_CTX}`,
    `task_id: 260605_xtrax-s4-polish. Verify MultiTaskLoss weight_schedule (Task 7.2):
1. MultiTaskLoss has field weight_schedule: Callable[[int], Array] | None = eqx.field(default=None, static=True). Field is marked static.
2. MultiTaskLoss.__call__ accepts step: int = 0. When weight_schedule is not None, result is weight_schedule(step) * sum_of_weighted_losses.
   Note: the spec is silent on HOW weight_schedule is applied. The fixer prompt explicitly
   documents the chosen interpretation (scalar multiplier on total loss) as a design decision,
   not a spec derivation. Accept this interpretation as stated — verify internal consistency,
   not that it matches a spec sentence that doesn't exist.
3. Backward compat: MultiTaskLoss(losses=...)(predictions, targets) with no step arg still works and matches Sprint 2 behavior.
4. weight_schedule=lambda step: jnp.array(2.0) doubles the loss vs no schedule.
5. step-dependent schedule: step=3 with schedule lambda s: s+1 gives 4× the unscheduled loss.
6. All original test_loss.py tests still pass (no regressions).
7. pyproject.toml version is "0.2.0".
8. ruff clean on loss.py; uv run pytest tests/ -v all pass (full suite, not just test_loss.py).
9. Protocol compatibility: LossFunction is @runtime_checkable with __call__ only. Adding
   step: int = 0 to MultiTaskLoss.__call__ does NOT break isinstance(loss, LossFunction) —
   runtime_checkable isinstance checks presence of __call__, not its signature arity. Verify
   no isinstance guard in types.py was changed and existing isinstance checks still work.
`,
  );

log("xtrax Sprint 4: Phase 7 Novel Additions — Track A only (Task 7.1 already done in Sprint 2)");
const a = await trackA();

return {
  task_id: TASK_ID,
  sprint_id: 4,
  verdicts: { "1153": a },
};
