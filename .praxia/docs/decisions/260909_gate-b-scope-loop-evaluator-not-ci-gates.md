---
title: Loop-constitution gate (b) covers the loop's evaluator closure, not repo CI gates
description: 'Marielle''s ruling: gate (b)''s ''evaluator code / test splits / metric definitions'' means the #2181 loop''s EvaluateFn closure; repo audit gates such as test_rigor are outside it and need no per-event attestation'
status: accepted
task_id: 260909_release-trail-sprint
date: '260909'
supersedes: ''
backlog_ids: ''
---
# Loop-constitution gate (b) covers the loop's evaluator closure, not repo CI gates


## Context

The `260909_release-trail-fail-loud` sprint rewrites
`src/xtrax/devtools/gates/test_rigor.py` so the audit gate reports and fails on
its own failure (#5021). The sprint specification asserted that this is a
loop-constitution gate-(b) event, and specified a `T2-29` attestation row plus a
closure-hash re-lock as a precondition for merging.

Gate (b) (`.praxia/docs/decisions/260714_2181-autoresearch-loop-constitution.md:45-52`)
reads:

> any change to evaluator code, test splits, or metric definitions requires
> Marielle's explicit sign-off before the changed evaluator is trusted, *and*
> forces a closure-hash re-lock (T2-05) of the new evaluator's complete closure
> (code + splits + metric defs + pinned deps + config). The agent never approves
> its own judge, under any circumstance. This gate fires on every
> evaluator-change *event*, not once — there is no standing blanket approval.

Read on its plain words alone, that covers `test_rigor.py`: it is the sole
producer of `test_rigor.line_coverage_pct` and `test_rigor.branch_coverage_pct`,
which a ratchet baseline enforces, so it authors metric definitions.

## The measurement

Three checks, all against the tree at `60b85e7`:

1. **`src/xtrax/loop/` never imports `test_rigor` or `xtrax.devtools.gates`.**
   `rg -n 'devtools|test_rigor' src/xtrax/loop/` returns two hits, both prose in
   docstrings describing a pattern being mirrored
   (`ratchet_crash_atomicity.py:78`, `candidate_smoke.py:40`). There is no call
   path from the loop to this gate.
2. **`evaluator_change_gate` is imported by exactly one file: its own test.**
   `rg -n 'evaluator_change_gate' --glob '!**/evaluator_change_gate.py'` returns
   only `tests/loop/test_evaluator_change_gate.py:9`. No production code calls
   the checker.
3. **`closure_lock.py`'s own docstring scopes the closure to the loop's fitness
   oracle** (`:1-24`): the failure mode it exists to prevent is "a candidate
   mutated an uncovered surface ... and the judge drifted undetected", the
   protected object is what an `EvaluateFn` consults per iteration, and it states
   plainly that "no #2181 loop controller exists yet" to act on a drift error.

## Decision

**Gate (b) covers the evaluator closure of the #2181 autonomous-evolution loop —
the `EvaluateFn` that judges evolved candidates and the closure `closure_lock`
locks around it. It does not cover repository CI and audit gates, including
`xtrax.devtools.gates.*`, the `audit-*` Justfile recipes, and
`.praxia/audit_baseline.json`.**

Consequently Phase C of the `260909` sprint fires no gate-(b) event. It needs no
`T2-29` attestation row and no closure-hash re-lock, and proceeds as an ordinary
reviewed pull request.

## Why this ruling and not the broader one

The broader reading is not absurd — it follows from the words in isolation. It is
rejected because it prices in permanent friction for no protective gain:

- The harm gate (b) names is a **judge that drifts while the agent keeps scoring
  against it**. That harm requires an autonomous scoring loop. There is no loop,
  and when there is one, `test_rigor` is not what it consults.
- Under the broader reading every change to any `audit-*` gate — a threshold
  nudge, a lint rule, a recipe rename — becomes a per-event human attestation
  with a 7-day TTL. That converts a targeted safeguard into a toll on ordinary
  repository maintenance, and a safeguard that fires constantly on low-stakes
  events is one that gets waved through on the high-stakes one.
- Attesting a CI gate as an evaluator would also make the first *live* gate-(b)
  row an example of the wrong thing, which is worse than having none.

## What did not change

- **The agent still never rules on this alone.** The constitution's "The agent
  never approves its own judge, under any circumstance" means an agent may not
  select the narrow reading in order to escape needing a sign-off. This ruling
  was put to Marielle as an open question with the evidence for both readings and
  a recommendation, and is recorded here because she made it.
- **Gate (b) is unchanged in force for what it covers.** When a `#2181` loop
  evaluator, its splits, or its metric definitions change, the sign-off and the
  closure-hash re-lock both still apply, per event, with no standing approval.
- **`evaluator_change_gate.py`'s `event_ref` contract is unchanged**: a gate-(b)
  attestation matches on `ClosureManifest.closure_hash` (`:148`), never a commit
  sha. That remains true for the first row that legitimately fires.

## Consequences

- Phase C of `260909_release-trail-fail-loud.md` drops sub-task C7 (the re-lock)
  and its `T2-29` precondition. The phase's other sub-tasks are unaffected.
- `.praxia/loop_human_gates.toml` gains no row from this sprint.
- A future change to a real loop evaluator should cite this document to confirm
  it is *in* scope, rather than re-deriving the boundary.
