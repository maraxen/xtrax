# T2-33 — island/population-search upgrade delta (scope only)

**Status:** DOCUMENTED DELTA, NOT IMPLEMENTED — implementation blocked on T2-31 (AC-24)
scope-expansion approval, which does not currently exist (confirmed against
`.praxia/loop_human_gates.toml` as of 2026-07-16: gates (b)-(e), including T2-31, have zero live
attestations — "no such event has happened yet").
**Governs:** epic #2181's P4-gated Phase-2 island/population-search upgrade (T2-33,
`.praxia/docs/roadmaps/research-epics/260702_02-dag-2181-autoresearch.md`).
**Supersedes:** the DAG's own one-line T2-33 sketch ("batch/generation eval, population state,
migration hooks, per-candidate resource accounting") on two specific points — see §3.

## 1. Purpose

T2-33 is the only item in the entire #2181 T2 DAG with no acceptance criteria
(`acs_covered: []`) — its own text calls it "a documented delta... not promised as zero-cost
drop-in." Its `depends_on: [T2-03, B2-03, T2-22, T2-31]` are now all merged/available, but that
does **not** mean island-phase entry is approved: T2-31 (`src/xtrax/loop/scope_expansion_gate.py`)
is a *mechanism* for recording approval, not an approval itself, and no island-phase entry has
been recorded. Per the DAG's own gate text, "island search drops in only after... the T2-31
(AC-24) scope-expansion gate approves island-phase entry (island search = a scope expansion)."

This document formalizes what that delta actually is, so a future implementer (or Marielle,
deciding whether to grant T2-31 approval) has a concrete spec to work from — without building any
code ahead of that approval, matching the item's own explicit fast/loud framing.

## 2. Grounding

Primary input: `.praxia/docs/research/260715_gear-autosota-full-read.md` (T2-03's full read of
GEAR, arXiv 2605.13874, and AutoSOTA, arXiv 2604.05550 — not title/abstract only). That note's §4
("Implication for T2-33's design") already did substantial design work; this document turns its
recommendations into an actionable spec. Cross-checked directly against:
- `src/xtrax/stages/evaluate.py` (T1-07, the sealed `EvaluateFn` seam)
- `src/xtrax/loop/multi_metric_ratchet.py` (T2-17, restart-from-best-K ratchet)
- `src/xtrax/loop/diversity_quota.py` (T2-18, Leap-Path quota / AST-structural-diff)
- `src/xtrax/loop/scope_expansion_gate.py` (T2-31, the approval-lookup mechanism)
- `.praxia/loop_human_gates.toml` (the live approval record — currently empty for T2-31)
- bathos's `campaign_edges`/`run_edges` tables (B2-03, merged, multi-parent lineage)

## 3. What does NOT need to change

- **`xtrax.stages.evaluate.EvaluateFn`'s seam is already sufficient.** Both GEAR's actual evaluator
  call (`(code_string) -> {bpb, vram_gb, params_m, success}`) and AutoSOTA's live accept/reject loop
  are single-candidate, scalar-dict-out — structurally identical to xtrax's existing sealed
  `(frozen_context, candidate) -> dict[str, float]` shape. Everything population/frontier-shaped in
  either reference system (UCB parent scores, novelty, promotion thresholds) is orchestration-layer
  bookkeeping computed *around* repeated single-candidate evaluate() calls, never a change to the
  call's own signature.
- **"Batch/generation eval"** (the DAG's original sketch) is dropped — neither GEAR nor AutoSOTA
  ever calls an evaluator with more than one candidate at a time. If per-candidate resource
  accounting across concurrently-running candidates is ever wanted, that is a scheduler/controller
  concern (how many single-candidate evaluate() calls run concurrently), not a change to the
  evaluate() call shape.

## 4. What the delta actually is

1. **Bounded elite frontier, not island migration.** Rescope "migration hooks" (the DAG's original
   sketch) to a GEAR-style bounded elite pool + UCB-style parent selection + rule-based promotion.
   This is cheaper than CodeEvolve-style island migration and extends, rather than replaces, the
   MVP's already-shipped mechanisms:
   - T2-17's `compute_ratchet_decision` (restart-from-best-K) is structurally the same idea as
     GEAR's permanently-reserved best-slot ratchet — one frontier slot that a new global best fills
     and is never evicted.
   - T2-18's `diversity_quota` (AST-based structural-diff/canonicalization, Leap-Path quota) is
     structurally the same idea as GEAR's novelty term (Jaccard distance between a candidate's
     description tokens and its most similar recent sibling) — T2-18's existing structural-diff
     machinery is a natural reuse point for a novelty score, not a reason to build a second,
     independent diffing mechanism.
2. **New population/frontier state** (real, non-zero cost — this document does not claim otherwise)
   would live in the loop controller / `frozen_context` layer, and never crosses the evaluate()
   seam:
   - per-node parent-usage count (how many times a node has served as a parent)
   - mean child-improvement over that node
   - a short natural-language "what was tried" description (input to the novelty/Jaccard term)
   - a composite UCB score (productivity + novelty + coverage), computed entirely by the controller
     from its own accumulated history — never passed into or computed by evaluate() itself
3. **Rule-based promotion**, matching GEAR's mechanism exactly: a new global best fills the reserved
   best slot; a candidate that beats the weakest current elite replaces it; otherwise the candidate
   is discarded. Each child is evaluated independently — no pairwise or batch scoring.
4. **Multi-parent campaign DAG** — bathos's B2-03 `campaign_edges`/`run_edges` tables (already
   merged this epic) already provide the `(child_id, parent_id)` multi-parent lineage primitive
   the DAG's own success clause calls for ("multi-parent campaign DAG assembled by bathos"). No new
   bathos-side work is needed for this half of T2-33.

## 5. Explicitly deferred beyond even this document

- **Literal CodeEvolve-style island migration** (periodic migration between isolated demes) — not
  recommended. The bounded-elite-frontier model above is both cheaper and closer to what's already
  shipped in P1/P2; this document does not re-open that choice, it's a considered recommendation,
  not an open question.
- **Any actual code** — no dataclasses, no UCB scoring function, no promotion-decision module, no
  controller wiring. All of it is blocked on T2-31 approval (§6).

## 6. Path to activation

If/when Marielle grants T2-31 approval (a `[[gates]] id="T2-31" event_ref="<phase2 capability
name>"` entry in `.praxia/loop_human_gates.toml`, following the exact pattern
`assert_scope_expansion_approved` already reads), a future implementation item should:

1. Define the frontier/population state dataclasses per §4.2.
2. Implement UCB parent-selection + promotion as a pure-decision-function module — matching this
   epic's established shape-(b) convention (`compute_ratchet_decision`, `assess_stats_battery_verdict`:
   never raise for a normal outcome, only for malformed input) — gated by
   `xtrax.loop.scope_expansion_gate.assert_scope_expansion_approved` before any island-search code
   path activates.
3. Reuse T2-18's existing AST-diff/canonicalization machinery for the novelty term rather than
   reimplementing structural comparison.
4. Consume bathos B2-03's `campaign_edges`/`run_edges` directly for multi-parent lineage — no new
   bathos work required.
