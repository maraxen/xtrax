---
title: 'Gate (b) ruling holds now that controller/ exists; #4584 parts 1/2/4 are not gate-(b) events'
description: 'Addendum to 260909: the gate-(b) closure is the declared evaluator/split/metric paths hashed by closure_lock; controller harness code outside it (main_loop acceptance plumbing) is not gate (b). Applied to #4584: pre-scoring halt, self-consistent CandidateRunResult and adapter docstring ship as an ordinary PR; reconciling `accepted` and moving the stats/seed gates ahead of the lineage advance goes in a separate PR for Marielle''s explicit sign-off.'
status: accepted
task_id: 260930_sprint-backlog-debt
date: '260930'
supersedes: ''
backlog_ids: '4584'
---
# Gate (b) ruling holds now that controller/ exists; #4584 parts 1/2/4 are not gate-(b) events

Addendum to `.praxia/docs/decisions/260909_gate-b-scope-loop-evaluator-not-ci-gates.md`.
It applies that ruling; it does not amend the loop constitution.

## Why an addendum

The 260909 ruling reasoned partly from "no #2181 loop controller exists yet".
`controller/` now exists. Left unstated, the next reader could take that sentence as
the ruling's premise and conclude it lapsed.

## What the gate-(b) closure is, in the code

`closure_lock.build_closure_manifest(evaluator_paths=, split_paths=,
metric_def_paths=, config=, pinned_deps_source=)` hashes exactly what the caller
declares. The controller builds it from `frozen_context.locked`
(`controller/evaluate_adapter.py`, re-locked at campaign start in
`controller/loop_run.py`). `controller/main_loop.py` is the harness that calls the
judge. It is not a member of that closure. Changing it trips no closure-hash lock,
and a campaign that did list a controller file would fail loudly, not drift.

## Ruling (Marielle, 2026-09-30)

1. The 260909 ruling stands with a controller present. Gate (b) covers the
   evaluator closure: the `EvaluateFn`, splits and metric definitions that
   `closure_lock` hashes. Harness code outside it is not a gate-(b) event.
2. Applied to backlog #4584:
   - **Not gate (b), an ordinary reviewed PR:**
     - part 1, halting before scoring when the run did not succeed;
     - part 2, making `CandidateRunResult` reject `success != (exit_code == 0)`;
     - part 4, correcting the adapter docstring.
     None of these changes the judge. Each can only refuse to score or accept.
   - **A separate PR with explicit sign-off:**
     - part 3, reconciling the three definitions of `accepted`;
     - moving the stats and seed gates ahead of the best-so-far lineage advance.
     Neither touches the hashed closure. Both define what counts as an accepted
     candidate, the closest thing in the controller to the judge's semantics, which
     the constitution says the agent must not settle for itself. Marielle approving
     that PR is the sign-off. No attestation row and no re-lock are needed, because
     nothing in the closure changes.

## Consequence to know

Part 1 is a HALT, not a rejection. A candidate whose bathos run fails now raises
`RawArtifactsUnavailableError` out of `run_one_candidate_pass`. `run_multi_iteration_loop`
does not catch it, so the campaign ends and concludes with `outcome_label="aborted"`.
The pre-bathos gates (smoke, checkified execution) already behave this way. Before,
the failed run was scored, recorded `accepted=False`, and the loop continued. Whether
a failed candidate should instead be a caught per-candidate failure that continues the
campaign is error/retry policy. That belongs to AC-8c (LC-11), not to this fix.
