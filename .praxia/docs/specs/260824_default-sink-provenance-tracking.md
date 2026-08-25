---
title: Default provenance tracking at ZarrStagingSink for downstream consumers
task_id: 260824_default-sink-provenance-tracking
date: 260824
status: revised
brainstorm_session: true
adversarial_review: passed-with-revisions
invest_overrides: ["Small"]
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

This spec went through adversarial review (`praxia:spec-challenger` vs. `praxia:spec-defender`,
both Sonnet) after the brainstorming session closed. See **Adversarial Spec Review Record**
below. Acceptance criteria below are the **post-review revision** — 4 objections were conceded
outright, 5 more required a small addition the defender itself specified; both are folded in
directly rather than left as an addendum, since they're gaps an implementer would otherwise have
to guess at.

## Acceptance Criteria

- Given a downstream consumer constructs a `ZarrStagingSink` inside a real git checkout, when the sink is initialized, then the git-capture step captures `git_sha`, `git_branch`, and `git_dirty` via `git rev-parse HEAD` / `git rev-parse --abbrev-ref HEAD` / `git status --porcelain` (bathos's cited commands) wrapped in a broad `except Exception` (matching bathos's outer `capture_git_state`, not just its inner narrow catch) so that step alone never raises regardless of cause.
- Given a downstream consumer constructs a `ZarrStagingSink` where git state cannot be determined (no repo, missing `git` binary, or a failing shellout), when the sink is initialized, then it records `git_sha="unknown"` and calls `warnings.warn(...)` with message text naming which of the three causes applied, rather than failing silently.
- Given a `SinkSpec` gains a new required `run_id: str` field (there is no other constructor surface for it — `make_sink()` forwards `spec` unmodified, so no other code needs to change) and a downstream consumer constructs a `ZarrStagingSink` with `run_id` unset, when the sink is initialized, then construction raises an explicit error naming `run_id` as the missing required field.
- Given a `ZarrStagingSink` has completed at least one `drain()`, when the store's root group is inspected, then its `.attrs` contains the full core provenance record: `git_sha`, `git_branch`, `git_dirty`, `run_id`, and `created_at` as an ISO-8601 UTC string (`datetime.now(timezone.utc).isoformat()`), captured once at `__init__` and re-written idempotently (same value) on every subsequent `drain()`.
- Given a `ZarrStagingSink` drains a payload staged under a key (the full tuple passed to `stage()`, matching the group `drain()` already writes arrays to — not a first-path-segment subgrouping), when that key's group is inspected, then its own `.attrs` contains a minimal provenance pointer (`run_id` and `git_sha` only), independent of the store's root group.
- Given a caller stages `attrs` containing a key name that collides with a core provenance field name (`git_sha`, `git_branch`, `git_dirty`, `run_id`, `created_at`), when that key is staged, then `stage()` raises a validation error immediately — core provenance field names are reserved and a caller may not overwrite them by staging a same-named key.
- Given a `SinkSpec` declares an extension schema for additional provenance fields, when a caller calls `stage()` with `attrs` that do not conform to that schema, then `stage()` itself raises a validation error immediately (not deferred to `drain()`), so an invalid call cannot be masked by a later `stage()` call to the same key overwriting it before `drain()` ever runs.
- Given a `SinkSpec` declares no extension schema, when a caller stages arbitrary attrs alongside the core provenance fields, then those caller attrs are preserved exactly as `ZarrStagingSink` handles them today — standard JSON-Schema `additionalProperties`-permitted semantics mean only schema-declared keys are checked; any other key passes through untouched.
- Given `ZarrStagingSink` gains a new `finalize()` method (or context-manager `__exit__`) that a downstream consumer calls once at run end to signal run completion — the class has no such lifecycle signal today, so this is new public surface, not a reuse of an existing method — when `finalize()` runs, then it calls `zarr.consolidate_metadata()` exactly once, after which no further `drain()` calls are considered legitimate for that sink instance.
- Given two separate `ZarrStagingSink` instances are constructed against the same `output_dir` (legitimate today via the existing `mode='a'` append-open), when the second instance's `__init__` runs and the store's root `.attrs` already carries a different `run_id` than the second instance's own, then construction raises an explicit error rather than silently overwriting the first run's root record while earlier per-key pointers (from the prior criterion) still reference the first run.

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
| Bathos adoption: `GitState`-equivalent (hash/branch/dirty, safe shellout, never-raise fallback) only | Selected | Directly reusable; matches xtrax's existing "no dependency on the orchestration tool built on top of it" posture. Post-review: "safe shellout" and "never-raise fallback" are adopted as bathos's own two distinct layers (inner narrow-catch shellout + outer broad-catch wrapper), not conflated — see AC1. |
| Bathos adoption: + myxcel env/sidecar multi-channel injection | Deferred | Solves "no real .git checkout on a compute node," which has no confirmed occurrence for xtrax consumers yet — revisit if/when that gap is hit in practice. |
| Bathos adoption: + W3C PROV-JSON multi-parent lineage export | Rejected | No identified need for multi-parent run lineage graphs at the xtrax layer. |
| Capture timing: once, at `ZarrStagingSink.__init__` | Selected | Correct semantics — the running process's code identity is fixed at process start. `created_at` follows the same reasoning: captured once, re-written idempotently on every `drain()` (post-review, resolves C12's timing half). |
| Capture timing: fresh, on every `drain()` | Rejected | N subprocess calls for N drains, and doesn't reflect a real risk. |
| Capture timing: conditional re-shell on a dirty-state signal | Rejected | Added complexity with no identified need. |
| Placement: root attrs + consolidated metadata only | Rejected | Orphans provenance the moment a top-level group is copied/exported out of the store on its own. |
| Placement: full record on every drained group | Rejected | Reintroduces the per-group attrs-write/inode duplication cost from the prior session's inode-pressure analysis. |
| Placement: hybrid — full record at root (+ consolidated metadata) plus a minimal pointer per staged key's own group | Selected | Resolves the orphaning risk without full per-group duplication. Post-review: "top-level" means the full staged key tuple (the same group `drain()` already writes arrays to), not a first-path-segment subgrouping — clarified in ACs to remove the ambiguity C1 identified, even though the original text was defensible on a strict reading. |
| Schema shape: reserved namespaced attrs key (e.g. `_xtrax_provenance`) | Rejected | Implicit reserved-word contract nothing enforces against caller key choice. |
| Schema shape: flat fields merged into attrs | Rejected | Silent collision risk against caller-staged keys with the same name. |
| Schema shape: sink refuses colliding caller keys | Rejected (as originally scoped) → **partially re-adopted post-review** | Original rejection was about *caller-declared extension* keys; adversarial review (C5) found the *core* fields (git_sha/run_id/etc.) reintroduce the identical collision risk at the per-key-group attrs level once the core+extension design was selected. Resolution: core field names are reserved and collision raises (new AC), matching AC7's fail-loud posture — extension-field collision handling stays as originally decided (schema validation, not blanket refusal). |
| Schema shape: core required fields + caller-declared JSON-Schema-style extension, validated | Selected | Well-trodden pattern; satisfies "verified parsable and schema validatable" while keeping fields beyond the core flexible per consumer. |
| Schema extension ownership: domain library registers once at import/setup | Rejected | Less flexible than per-`SinkSpec` declaration. |
| Schema extension ownership: caller declares per `SinkSpec` instance | Selected | Different runs of the same library can declare different extensions. |
| Schema extension ownership: separate decoupled validation pass, sink stays schema-agnostic | Rejected | Doesn't satisfy "fail fast and loud" at write time. |
| `closure_hash`: core required field | Rejected | Not universal — `ClosureManifest` is loop-controller/evaluator-lock machinery (epic #2181); a plain training consumer has no closure concept at all. |
| `closure_hash`: core required with a "no-closure" sentinel | Rejected | Forces an artificial value onto consumers who have no concept of a closure. |
| `closure_hash`: declared extension field | Selected | Present only when the consumer's context actually has a `ClosureManifest`. |
| `run_id` constructor surface: new required `SinkSpec` field | Selected (post-review, closes C2) | `make_sink()` (sink.py:22-39) forwards `spec` unmodified to `ZarrStagingSink(spec)` — a `SinkSpec` field requires zero changes to that factory path, unlike a separate `__init__` kwarg which would. |
| AC6/AC7 validation timing: `stage()`-time vs. `drain()`-time | `stage()`-time selected (post-review, closes C4) | `drain()`-time validation permits an early invalid `stage()` call's attrs to be silently overwritten by a later valid `stage()` call to the same key (per the documented merge-overwrite semantics) before validation ever runs — masking the offending call. `stage()`-time validation makes that masking structurally impossible. |
| AC2 warning mechanism: unspecified vs. `warnings.warn` | `warnings.warn(msg, UserWarning, stacklevel=2)` selected (post-review, closes C7) | Neither the codebase (`zarr_sink.py`/`zarr_integrity.py` have zero precedent) nor the cited bathos file actually warns in this exact scenario (bathos's `_legacy_git_shellout`/`capture_git_state` silently return `None`/`_UNKNOWN`) — needed an explicit choice rather than inheriting one from the model. |
| Run-completion signal for AC8/consolidation: none vs. new `finalize()`/context-manager method | New `finalize()` method selected (post-review, closes C14 and, by construction, C16) | `ZarrStagingSink` has no lifecycle method today (`__init__`/`stage`/`take`/`drain`/`__len__` only) — "per completed run" presupposes an event the class can't currently observe. Placing the once-only `consolidate_metadata()` call inside `finalize()` also makes post-consolidation staleness (C16) structurally impossible, since no further `drain()` is legitimate after it. |
| Multi-run reuse of the same `output_dir`: silent overwrite vs. guard | Guard (raise on `run_id` mismatch at init) selected (post-review, closes C11) | `ZarrStagingSink` already opens with `mode='a'`, so multi-run reuse of one directory is real, pre-existing, supported behavior — a silent root-attrs overwrite would leave the store internally inconsistent (root claims run B, earlier per-key pointers still reference run A) with no way to detect it after the fact. |

> **260825 addendum (#457(1))**: the "both, via one shared layer" deferral has
> fired for the run CLI — `xtrax run` now derives its sink through
> `derive_sink_spec` (see `260824_runspec-trainer-run-id-plumbing.md`, status:
> adopted). The "CLI end-users | Rejected" row above is superseded for
> persistence provenance: CLI runs additionally get a zarr provenance store at
> `.xtrax/runs/<run_id>/metrics.zarr` alongside (not replacing) the manifest.

## Assumptions

| Assumption | Owner | Verification method |
|------------|-------|---------------------|
| `run_id` is obtainable by the caller at the point a `ZarrStagingSink` is constructed | xtrax maintainer | Audit real RunSpec → Trainer → Sink construction order once the plumbing mechanism (see TBDs) is designed; confirm no consumer path constructs a sink before a run_id exists. |
| A plain `git` shellout with a loud "unknown" fallback is sufficient for every current xtrax consumer execution environment | xtrax maintainer | Confirm against actual Engaging/cluster usage patterns before ruling out multi-channel (env/sidecar) injection permanently. |
| `zarr.consolidate_metadata`'s cost is low enough that exactly-once-at-finalize is not overly conservative | xtrax maintainer | Benchmark against a representative multi-drain streaming workload (see TBDs). |
| Requiring an explicit `finalize()` call is an acceptable API burden on every downstream consumer (vs. an implicit signal) | xtrax maintainer | Confirm with aminx (or whichever consumer adopts this first) that an explicit lifecycle call fits their call pattern before treating it as final. |

## TBDs

| Item | Owner | Resolution deadline |
|------|-------|---------------------|
| Concrete mechanism for RunSpec/Trainer construction to plumb `run_id` down to wherever a `ZarrStagingSink` gets constructed, so consumers don't have to thread it manually | xtrax maintainer | Before implementation begins. Note: this is now narrower than before — `run_id` is confirmed to land as a `SinkSpec` field (see Decision Log); this TBD is purely about the upstream RunSpec/Trainer wiring to populate that field, not the field's existence. |
| Benchmark data for `consolidate_metadata` cost, now specifically at the `finalize()` call site | xtrax maintainer | Before or shortly after first real consumer adoption — the once-per-run placement no longer needs the benchmark as a *precondition* (finalize() gives a safe default), but the data still informs whether that default is well-placed. |
| JSON-Schema validation implementation choice: `jsonschema` package vs. minimal hand-rolled validator | xtrax maintainer | Interim default (post-review, closes C3): ship with a minimal stdlib-only validator checking `type`/`required`/`properties` against the caller-declared schema dict; swap for the `jsonschema` package only if this TBD later resolves that way. Implementation is unblocked either way. |
| Whether bathos's myxcel env/sidecar multi-channel git-provenance injection becomes necessary for xtrax | xtrax maintainer | Revisit if/when a consumer actually runs on a compute node without a real `.git` checkout. |

## Pre-mortem Record

- Failure scenario (user): A bug silently prevented recording of the actual git SHA/refs, or silently excluded custom extension fields/subschemas — the system degraded quietly instead of failing fast and loud.
- Failure scenario (AI): `consolidate_metadata` is called unconditionally on every `drain()` in a long streaming loop; write throughput silently degrades over months until it becomes the actual bottleneck in a production run, discovered only by accident.
- Mitigation addressed in spec: Yes, and post-review more concretely: the git-unknown and missing-run_id paths are explicit ACs (1st/2nd/3rd criteria); the new collision-raise and stage()-time-validation criteria close two additional silent-degradation paths adversarial review found (core-field collision, drain()-time masking) that weren't covered by the original pre-mortem scenario but are the same failure shape; `finalize()` bounds `consolidate_metadata` to exactly once, directly gating the AI-added scenario without permitting the "never call it at all" loophole the original "at most once" wording left open.

## Adversarial Spec Review Record

Run 260824 via `praxia:spec-challenger` and `praxia:spec-defender` (both Sonnet), sequential (challenger first, defender responding to its specific objections — not independent parallel review).

**Challenger** (17 objections, overall verdict `not_ready`, confidence `high`): read the spec against the live `zarr_sink.py`/`sink.py`/`zarr_integrity.py` code and bathos's cited `git.py`, not just the spec prose. Found ambiguities in "top-level key" vs. the sink's actual nested-group model, constructor surfaces (`run_id`, extension schema) the spec's ACs assumed without specifying where they'd live, an unaddressed lifecycle gap for "per completed run," and several existing-but-unmentioned edge cases (`take()`'s silent attrs-discard, `mode='a'` multi-run reuse, core-vs-caller attrs collision).

**Defender** (Sonnet, same code + spec): rebutted 8 of 17 from the spec's own text or code the challenger read too narrowly (notably C13's "AC8 trivially satisfied by never calling consolidate_metadata" — the AC's literal purpose was always just capping frequency, not guaranteeing occurrence). Conceded 4 outright (C2 run_id location, C5 core-field collision, C7 warning mechanism, C14 lifecycle signal) and flagged 5 more needing one small concrete addition each rather than a full gap (C3 validator interim default, C11 multi-run guard, C12 timestamp format, C15 make_sink() no-op-by-construction, C16 folds into C14's fix). Overall verdict `needs_revision`, confidence `high` — explicitly narrower than the challenger's `not_ready`.

**Resolution:** all 9 real gaps (4 conceded + 5 partial) are folded directly into the Acceptance Criteria and Decision Log above, rather than left as a follow-up pass — each had a concrete, spec-writable answer, not an open research question. The 8 rebutted objections are not reflected as spec changes; three of them (C1, C4, C8) prompted small clarifying wording anyway, purely to remove ambiguity for a future reader even though the defender showed the original text was technically defensible.

## INVEST Gate (re-run post-revision)

```
✓ Independent — ZarrStagingSink already exists in main; no upstream blocker beyond this
  spec's own new surface (SinkSpec.run_id, SinkSpec extension schema, finalize()).
✓ Negotiable — extension-schema validator implementation and the multi-run guard's exact
  error type can both flex without losing the core provenance guarantee.
✓ Valuable — closes the fragmented/unenforced run_id↔zarr-store linkage gap for downstream
  domain-library consumers (e.g. aminx).
✓ Estimable — larger than the pre-review estimate: adversarial review surfaced a genuine new
  public-API needs (finalize()/lifecycle method, SinkSpec.run_id field, extension-schema
  field) beyond "stamp attrs on drain." Still boundable to zarr_sink.py + sink.py.
✗ Small — 10 acceptance criteria (was 8 pre-review; exceeds the ≤8 limit). The 2 added
  criteria (core-field collision raise, finalize()/consolidation-once) are not scope creep —
  each closes a conceded adversarial-review gap that would otherwise leave an implementer
  guessing. Recorded as an override rather than force-split, since splitting the collision
  rule or the lifecycle method into a separate spec would leave the remaining spec
  non-implementable on its own (both are load-bearing for AC3/AC5's stated guarantees).
✓ Testable — every criterion is observable via attrs inspection or raised-error assertion;
  no forbidden vague terms. Post-review, AC2's warning and AC8-successor's finalize() are now
  concrete enough to assert on (specific mechanism named), closing C7/C13's testability gaps.
```

`invest_overrides: ["Small"]` recorded in frontmatter per the gate's non-silent-bypass rule.
