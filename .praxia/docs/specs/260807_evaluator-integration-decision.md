---
title: Resolve #3657 — in-process evaluator integration for the #2181 loop-controller
task_id: 260807_evaluator-integration-brainstorm-orient
date: '260807'
status: draft
brainstorm_session: true
invest_overrides: []
---

# Resolve #3657 — in-process evaluator integration for the #2181 loop-controller

## Decision

**SPLIT_COMPUTE**: the bathos-dispatched candidate subprocess produces only **raw artifacts**
(predictions, checkpoints, logs) into `output_paths` — never a pre-computed summary fitness
value. `EvaluateFn` becomes the sole call site in `controller/`; its own implementation calls
`campaign_adapter.run(...)` (the existing, unchanged bathos dispatch) to obtain those raw
artifacts, and then performs the actual scoring arithmetic itself, in-process, comparing the
raw predictions against the locked ground-truth splits (`ClosureManifest.split_paths`).
`guarded_evaluate`'s `track_reads()`/`verify_closure()` wraps the whole call, covering both the
dispatch and the in-process scoring step.

This is a precision correction to the `WRAP` shape, not a fifth independent option. `WRAP`'s
original design (mandate that the wrapper re-read and re-hash the subprocess's own summary
number from `output_paths`) was unanimously refuted by three independent adversarial reviewers:
re-hashing a subprocess-computed summary only proves "the wrapper reported honestly," not that
the candidate's subprocess didn't fabricate that summary in the first place — exactly the
MLR-Bench-style self-reported-metric threat `metrics_provenance.py`'s own docstring names as the
reason AC-7/AC-8 exist. SPLIT_COMPUTE closes that specific gap by construction: the subprocess is
never trusted to compute or report a fitness *number* at all, only to produce data the wrapper
independently scores against locked, trusted ground truth. It requires zero new bathos schema,
zero new OS-level sandboxing primitive, and touches only xtrax/`controller/` — no second repo.

Two honest limits, carried forward rather than papered over: (a) it only works if the scoring
arithmetic is cheap enough to run in-process, after the expensive dispatched step — this needs
verification against a representative #2181 workload, not assumed; (b) it does not defend
against a candidate gaming the metric through *legitimate* means (degenerate-but-real
predictions exploiting a metric weakness, or peeking at held-out data to overfit rather than to
fabricate a reported number) — that is a metric-robustness/data-leakage problem orthogonal to
every architectural shape considered, including full cross-process observation.

## Acceptance Criteria

- Given a bathos-dispatched candidate subprocess, when it writes to `output_paths`, then
  `output_paths` must contain only raw artifacts (predictions, checkpoints, or logs), not a
  pre-computed summary fitness value.
- Given raw artifacts in `output_paths` and a locked `ClosureManifest`'s `split_paths`, when
  `EvaluateFn` computes a candidate's fitness dict, then the scoring arithmetic must execute
  inside the in-process wrapper, not inside the bathos-dispatched subprocess.
- Given `controller/main_loop.py::run_one_candidate_pass` needs a candidate's fitness, when that
  evaluation is dispatched, then `EvaluateFn` must be the sole call site through which bathos
  dispatch occurs.
- Given `EvaluateFn` is invoked with a `frozen_context` and `candidate`, when `EvaluateFn`
  executes, then `EvaluateFn`'s own implementation must call `campaign_adapter.run(...)` to
  perform the out-of-process bathos dispatch.
- Given `guarded_evaluate` wraps `evaluator(frozen_context, candidate)` inside `track_reads()`,
  when `EvaluateFn` performs the bathos dispatch and then reads the resulting raw artifacts to
  compute fitness, then `track_reads()` must remain the active audit context for the duration of
  both steps.
- Given a fitness dict computed by `EvaluateFn` from raw artifacts, when
  `evaluate_with_provenance` wraps that result in a `MetricsProvenanceRecord`, then the record's
  `evaluator_closure_hash` must trace to the `ClosureManifest` used for the in-process scoring
  step, not to any value reported by the candidate's subprocess.

## Decision Log

| Option | Verdict | Rationale |
|--------|---------|-----------|
| SPLIT_COMPUTE (subprocess produces raw artifacts only; wrapper scores in-process) | Selected | Closes the exact fabrication gap the adversarial reviewers found, by construction — the subprocess is never trusted with computing or reporting a fitness number. Zero new bathos schema, zero new OS primitive, single-repo (xtrax/`controller/`) change. |
| WRAP-as-originally-fixed (wrapper re-hashes the subprocess's own reported summary as the tracked event) | Rejected | Unanimously refuted by 3 independent adversarial reviewers: re-hashing a subprocess-computed summary only proves the wrapper reported honestly, not that the candidate didn't fabricate the underlying value. Delivers a strictly weaker guarantee than advertised. |
| REPLACE (`EvaluateFn` call fully replaces the bathos dispatch; no out-of-process run at all) | Rejected | Robust across every assumption-reversal tested. Breaks Fork 15/16's bundling of "evaluator invocation" and "bathos run/campaign emission" as ONE effectful action that must route through bathos MCP; discards bathos's tracking/attestation/campaign bookkeeping for every candidate unless reinvented in-process. |
| ALONGSIDE (`EvaluateFn` called in-process in addition to the existing bathos dispatch) | Rejected | Creates two independently-true evaluations of the same candidate with no defined relationship between the bathos exit-code result and the closure-locked fitness dict — duplication, not signal. |
| SUBPROCESS_BOUNDARY, naive (candidate's own script self-invokes `guarded_evaluate` on itself) | Rejected | Inverts AC-8's entire trust rationale — a self-reported metric with extra ceremony, exactly the MLR-Bench-style fabrication threat the mechanism exists to stop. |
| SUBPROCESS_BOUNDARY, trust-preserving (separate harness process observes the bathos-spawned subprocess directly) | Deferred | Would deliver a genuine observation-based guarantee, but requires net-new cross-process instrumentation (ptrace/eBPF/LD_PRELOAD-class) that exists nowhere in xtrax or bathos today, and would touch `bathos/runner.py` — a second repo. Leaves AC-7/AC-8 unenforced for the entire duration of that build. Viable if SPLIT_COMPUTE's in-process-scoring assumption breaks, or if leakage protection (not just fabrication) becomes a stated requirement. |
| Sandbox-prevent (Landlock/mount-namespace restricting the subprocess's file visibility at spawn time) | Deferred | Confirmed via direct code read (`bathos/runner.py:484`) to NOT close the fabrication gap at all — bathos already treats the subprocess's self-written result file as authoritative regardless of any sandbox layer; sandboxing prevents *leakage* (undeclared reads), not *fabrication* (a legitimate write of a fake value). Cheaper than ptrace/eBPF (stable unprivileged syscall API) if ever needed as a complementary leakage control, but not built today and not a substitute for SPLIT_COMPUTE. |

## Assumptions

| Assumption | Owner | Verification method |
|------------|-------|---------------------|
| Scoring arithmetic (comparing raw predictions against locked `split_paths`) is cheap enough to run in-process, inside the same wrapper call that performs the bathos dispatch, without becoming its own heavy-compute bottleneck | xtrax/#2181 implementer | Profile scoring cost against a representative #2181 candidate before wiring `EvaluateFn`'s adapter; if scoring itself needs heavy out-of-process compute, SPLIT_COMPUTE's core premise breaks and this decision must be revisited |
| Fork 15/16's bundling of "evaluator invocation" and "bathos run/campaign emission" requires both in the same call/process, not merely one attested transaction | xtrax architecture owner | Re-read Fork 15/16 (`.praxia/docs/specs/260702_design-the-2181-agentic-algorithm-evolut.md:54`) against the looser, transaction-level reading the assumption-reversal beat left open; confirm before the single-call-site requirement (AC-3/AC-4 above) locks in during implementation |
| SPLIT_COMPUTE's raw-artifact/in-process-scoring split does not itself introduce a new leakage surface (e.g. the wrapper's read of raw artifacts exposing something the candidate process couldn't see) | xtrax/#2181 implementer | Security-review pass on the wrapper's read path once the adapter is built, before enabling on real campaigns |

## TBDs

| Item | Owner | Resolution deadline |
|------|-------|---------------------|
| Concrete `FrozenContext`/`Candidate` types (still bare `TypeVar`s in `src/xtrax/stages/evaluate.py`) | xtrax implementer | Before `EvaluateFn` adapter implementation starts |
| Exact adapter mapping `CandidateHandoff{path, content_sha256}`/`CandidateRunResult{script_path, exit_code, success}` into `FrozenContext`/`Candidate` | xtrax implementer | Same as above |
| Behavior on a failed bathos run (`success=False` / non-zero `exit_code`) — whether `verify_closure()`/`track_reads()` still runs, whether a degenerate fitness dict is recorded, whether audit-hook state stays consistent for the next candidate's `guarded_evaluate` call | xtrax implementer | Before first production wiring lands |
| Wire the first production caller of `build_closure_manifest`/`verify_closure`/`guarded_evaluate` — confirmed zero callers exist anywhere today | xtrax implementer | Independent of, and prior to, this decision's implementation (per architecture advisor recommendation) |
| Whether sandbox-prevent (Landlock) or `SUBPROCESS_BOUNDARY` (trust-preserving) should be picked up later as leakage-specific hardening | xtrax architecture owner | Revisit only if leakage protection becomes a stated requirement, or if the in-process-scoring assumption above breaks |
| Cross-host/network-mount staleness in the wrapper's raw-artifact read (TOCTOU risk if bathos dispatch lands the candidate on a different host/network mount than the wrapper) | xtrax implementer | Before enabling on multi-host/cluster campaigns |

## Pre-mortem Record

- **Failure scenario (AI, primary — originally run against the pre-fix `WRAP` design):**
  A candidate's subprocess reads an undeclared validation split during its own execution and
  bakes an inflated result into a self-reported summary written to `output_paths`. The wrapper
  re-hashes that summary post-hoc, `closure_lock` reports `verified=True`, and the fabrication
  goes undetected for five months across an entire campaign, because `track_reads()`'s
  `sys.addaudithook` (process-scoped by construction, PEP 578) never observed the subprocess's
  actual execution — only the wrapper's own post-hoc read.
  **Mitigation addressed in spec: yes.** Under SPLIT_COMPUTE, the subprocess never computes or
  reports a fitness number at all — it produces only raw predictions, and the wrapper
  independently scores those predictions against locked, trusted `split_paths` it reads itself.
  A candidate can no longer simply fabricate a reported score; it would have to fabricate the
  raw predictions in a way that survives independent re-scoring against real ground truth, which
  is a materially harder bar.

- **Failure scenario (AI, additional — cross-host staleness, orthogonal to the fabrication gap):**
  Because bathos dispatch can land the candidate subprocess on a different host or a
  network-mounted path, the wrapper's read of `output_paths` can race the actual
  write-completion/sync of those paths (per this environment's own cluster conventions —
  Engaging via `sbatch`, rsync/rclone-mounted result paths documented as prone to propagation
  delay). This yields either a false "verified" over data that doesn't correspond to what was
  actually executed, or nondeterministic mismatches misread as flakiness rather than a real
  TOCTOU gap.
  **Mitigation addressed in spec: no.** SPLIT_COMPUTE does not change *when* or *from where* the
  wrapper reads raw artifacts relative to write-completion — this risk applies identically under
  SPLIT_COMPUTE as it did under the original `WRAP` design. Carried forward as a TBD above; must
  be resolved before enabling on multi-host/cluster campaigns.

## INVEST Gate

```
✓ Independent — no other backlog item blocks this; this decision's own implementation (AC-6-
  equivalent: controller/'s EvaluateFn adapter) is what wires the first production caller of
  closure_lock/metrics_provenance, which is currently unwired for unrelated reasons.
✓ Negotiable — the exact FrozenContext/Candidate types and bathos-failure-path behavior are
  already deferred to TBDs without weakening the core SPLIT_COMPUTE shape.
✓ Valuable — solves a trustworthy, non-fabricable fitness signal for #2181 researchers making
  candidate ratchet/promotion decisions; unblocks backlog #3649 (GW-02), which depends_on #3657.
✓ Estimable — single-repo (xtrax/controller/) adapter + call-pattern change, comparable scope to
  already-sized "extended" backlog items in this area.
✓ Small — 6 acceptance criteria (≤8).
✓ Testable — every criterion specifies an observable, falsifiable outcome; no forbidden vague
  terms.
```

No overrides required.
