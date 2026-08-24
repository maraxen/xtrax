---
title: Default provenance tracking at ZarrStagingSink for downstream consumers
task_id: 260824_default-sink-provenance-tracking
date: 260824
status: draft
brainstorm_session: true
invest_overrides: []
---

# Default Provenance Tracking at ZarrStagingSink

## Context

Prior-turn research established that xtrax's provenance today is fragmented across
`manifest.json`, `ClosureManifest`, `MetricsProvenanceRecord`, `ReproFloorResult`, and
`ZarrStagingSink.attrs`, linked only by an unenforced `run_id` convention — and that
`ZarrStagingSink` (`src/xtrax/run/zarr_sink.py`) has zero production callers today. This
session scopes a fix: default, auto-captured static run provenance stamped by the sink
itself, for downstream domain-library consumers (e.g. aminx) that construct
`ZarrStagingSink` directly. Modeled on bathos's `GitState` capture primitive
(`~/projects/bathos/src/bathos/git.py`) — narrowly: the safe git shellout + never-raise
fallback, not its myxcel multi-channel injection or PROV-JSON lineage export.

## Acceptance Criteria

- Given a downstream consumer constructs a `ZarrStagingSink` inside a real git checkout, when the sink is initialized, then it captures `git_sha`, `git_branch`, and `git_dirty` via a `git` shellout that never raises.
- Given a downstream consumer constructs a `ZarrStagingSink` where git state cannot be determined (no repo, missing `git` binary, or a failing shellout), when the sink is initialized, then it records `git_sha="unknown"` and emits a visible warning naming the reason, rather than failing silently.
- Given a downstream consumer constructs a `ZarrStagingSink` without a `run_id` available, when the sink is initialized, then construction raises an explicit error naming `run_id` as the missing required field.
- Given a `ZarrStagingSink` has completed at least one `drain()`, when the store's root group is inspected, then its `.attrs` contains the full core provenance record: `git_sha`, `git_branch`, `git_dirty`, `run_id`, and a `created_at` timestamp.
- Given a `ZarrStagingSink` drains a payload staged under a top-level key, when that top-level group is inspected, then its own `.attrs` contains a minimal provenance pointer (`run_id` and `git_sha` only), independent of the store's root group.
- Given a `SinkSpec` declares an extension schema for additional provenance fields, when a caller stages attrs that do not conform to that schema, then `stage()` or `drain()` raises a validation error rather than silently dropping or truncating the non-conforming fields.
- Given a `SinkSpec` declares no extension schema, when a caller stages arbitrary attrs alongside the core provenance fields, then those caller attrs are preserved exactly as `ZarrStagingSink` handles them today (no new validation applied to undeclared fields).
- Given `zarr.consolidate_metadata` has not yet been benchmarked against a representative multi-drain streaming workload, when this design ships, then `consolidate_metadata` is called at most once per completed run (not on every `drain()`), until benchmark data justifies more frequent consolidation.

## Decision Log

| Option | Verdict | Rationale |
|--------|---------|-----------|
| Consumer scope: downstream domain libraries (e.g. aminx) calling `ZarrStagingSink` directly | Selected | Matches where the actual write happens; CLI end-users already get `manifest.json` from a different entry point. |
| Consumer scope: CLI end-users of `xtrax run` | Rejected | Different entry point, already served by the existing always-write manifest. |
| Consumer scope: both, via one shared layer | Deferred | Reasonable follow-on once the sink-level mechanism lands and is proven; not needed to solve the identified gap now. |
| Concern: static run provenance only | Selected | xtrax has zero existing telemetry concept (confirmed: no hits for the word in the codebase); scoping in a live-metrics stream would balloon this into an unrelated feature. |
| Concern: static provenance + live telemetry | Rejected | Same reason — no existing telemetry substrate to build on; out of scope. |
| Automaticity: auto-injected inside `ZarrStagingSink` | Selected | Only mechanism that satisfies "by default" — can't be skipped short of not using the sink at all. |
| Automaticity: required call at RunSpec/Trainer construction | Rejected | Doesn't cover the (documented, real) case of a consumer using `ZarrStagingSink` directly without going through Trainer. |
| Automaticity: opt-in helper function | Rejected | Not enforceable; contradicts "by default." |
| Bathos adoption: `GitState`-equivalent (hash/branch/dirty, safe shellout, never-raise fallback) only | Selected | Directly reusable; matches xtrax's existing "no dependency on the orchestration tool built on top of it" posture (same reasoning `repro_floor.py` already documents for bathos generally). |
| Bathos adoption: + myxcel env/sidecar multi-channel injection | Deferred | Solves "no real .git checkout on a compute node," which exists for bathos's cluster use case but has no confirmed occurrence for xtrax consumers yet — revisit if/when that gap is hit in practice. |
| Bathos adoption: + W3C PROV-JSON multi-parent lineage export | Rejected | No identified need for multi-parent run lineage graphs at the xtrax layer; that's bathos's own concern. |
| Capture timing: once, at `ZarrStagingSink.__init__` | Selected | Correct semantics, not just cheaper — the running process's code identity is fixed at process start; a git change after that doesn't change what's executing. |
| Capture timing: fresh, on every `drain()` | Rejected | N subprocess calls for N drains, and doesn't reflect a real risk (already-loaded code doesn't change mid-run). |
| Capture timing: conditional re-shell on a dirty-state signal | Rejected | Added complexity with no identified need once init-time semantics were clarified. |
| Placement: root attrs + consolidated metadata only | Rejected | Pre-mortem-adjacent finding: orphans provenance the moment a top-level group is copied/exported out of the store on its own — a plausible, ordinary downstream workflow. |
| Placement: full record on every drained group | Rejected | Reintroduces the per-group attrs-write/inode duplication cost identified in the prior session's inode-pressure analysis. |
| Placement: hybrid — full record at root (+ consolidated metadata) plus a minimal pointer per top-level group only | Selected | Resolves the orphaning risk without reintroducing full per-group duplication; explicitly confirmed as acceptable resolution to the named pre-mortem risk. |
| Schema shape: reserved namespaced attrs key (e.g. `_xtrax_provenance`) | Rejected | Implicit reserved-word contract nothing enforces against caller key choice; superseded by the validated-schema direction. |
| Schema shape: flat fields merged into attrs | Rejected | Silent collision risk against caller-staged keys with the same name. |
| Schema shape: sink refuses colliding caller keys | Rejected | Superseded once a proper schema-validation direction was identified. |
| Schema shape: core required fields + caller-declared JSON-Schema-style extension, validated | Selected | Well-trodden pattern (JSON Schema); satisfies "verified parsable and schema validatable" while keeping fields beyond the core flexible per consumer. |
| Schema extension ownership: domain library registers once at import/setup | Rejected | Less flexible than per-`SinkSpec` declaration; not the direction chosen. |
| Schema extension ownership: caller declares per `SinkSpec` instance | Selected | Matches the JSON-Schema-declaration precedent directly; different runs of the same library can declare different extensions. |
| Schema extension ownership: separate decoupled validation pass, sink stays schema-agnostic | Rejected | Doesn't satisfy "fail fast and loud" at write time — validation should happen where the data is written, per the pre-mortem. |
| `closure_hash`: core required field | Rejected | Not universal — `ClosureManifest` is loop-controller/evaluator-lock machinery (epic #2181); a plain training consumer (e.g. aminx without `guarded_evaluate`) has no closure concept at all. |
| `closure_hash`: core required with a "no-closure" sentinel | Rejected | Forces an artificial value onto consumers who have no concept of a closure; adds noise without adding truth. |
| `closure_hash`: declared extension field | Selected | Present only when the consumer's context actually has a `ClosureManifest`; the enforced core stays `run_id` (universal to every `RunSpec`-based consumer) plus git state and timestamp. |

## Assumptions

| Assumption | Owner | Verification method |
|------------|-------|---------------------|
| `run_id` is obtainable by the caller at the point a `ZarrStagingSink` is constructed | xtrax maintainer | Audit real RunSpec → Trainer → Sink construction order once the plumbing mechanism (see TBDs) is designed; confirm no consumer path constructs a sink before a run_id exists. |
| A plain `git` shellout with a loud "unknown" fallback is sufficient for every current xtrax consumer execution environment | xtrax maintainer | Confirm against actual Engaging/cluster usage patterns before ruling out multi-channel (env/sidecar) injection permanently. |
| `zarr.consolidate_metadata`'s cost is low enough that once-per-run is not overly conservative, and high enough that per-drain would be a real problem | xtrax maintainer | Benchmark against a representative multi-drain streaming workload (see TBDs) before finalizing frequency. |

## TBDs

| Item | Owner | Resolution deadline |
|------|-------|---------------------|
| Concrete mechanism for RunSpec/Trainer construction to plumb `run_id` (and any future provenance context) down to wherever a `ZarrStagingSink` gets constructed, so consumers don't have to thread it manually | xtrax maintainer | Before implementation begins — this is what makes the "auto-injected, can't skip" premise actually true in a real RunSpec→Trainer→Sink call chain, not just at the sink's own boundary. |
| Benchmark data for `consolidate_metadata` cost across drain frequency | xtrax maintainer | Before the once-per-run default (AC8) is revisited toward a more frequent policy. |
| JSON-Schema validation implementation choice: pull in the `jsonschema` package vs. a minimal hand-rolled validator, weighed against xtrax's light-dependency posture (`zarr`/`io_callback` are already lazy-imported optional extras) | xtrax maintainer | Before implementing the extension-schema acceptance criteria. |
| Whether bathos's myxcel env/sidecar multi-channel git-provenance injection becomes necessary for xtrax | xtrax maintainer | Revisit if/when a consumer actually runs on a compute node without a real `.git` checkout. |

## Pre-mortem Record

- Failure scenario (user): A bug silently prevented recording of the actual git SHA/refs, or silently excluded custom extension fields/subschemas — the system degraded quietly instead of failing fast and loud.
- Failure scenario (AI): `consolidate_metadata` is called unconditionally on every `drain()` in a long streaming loop; write throughput silently degrades over months until it becomes the actual bottleneck in a production run, discovered only by accident.
- Mitigation addressed in spec: Yes. AC2 and AC3 make the two identified "loud failure" paths (unresolvable git state, missing `run_id`) explicit, testable acceptance criteria rather than implicit behavior. AC6 makes schema-validation failures loud (reject on non-conformance, never silently drop a field). AC8 directly gates the AI-added consolidation-cost scenario by deferring frequency-beyond-once-per-run to actual benchmark data rather than an assumption.

## INVEST Gate

```
✓ Independent — ZarrStagingSink already exists in main; no upstream blocker. Note: the
  RunSpec/Trainer plumbing mechanism (TBD) is a separate, not-yet-designed follow-on this
  spec deliberately does not resolve — the 8 ACs above are self-contained against
  ZarrStagingSink's own boundary regardless of how run_id reaches the caller.
✓ Negotiable — extension-schema adoption and consolidate_metadata frequency can both flex
  without losing the core provenance guarantee (ACs 1-5).
✓ Valuable — closes the fragmented/unenforced run_id↔zarr-store linkage gap identified for
  downstream domain-library consumers (e.g. aminx).
✓ Estimable — core capture + loud-fallback + hybrid placement + schema validation is a
  medium-sized, boundable change to one module (zarr_sink.py) plus a small schema-validation
  addition; the RunSpec/Trainer plumbing TBD would need its own separate estimate.
✓ Small — 8 acceptance criteria (at the ≤8 limit).
✓ Testable — every criterion is observable via attrs inspection or raised-error assertion;
  no forbidden vague terms.
```

All six dimensions pass without override.
