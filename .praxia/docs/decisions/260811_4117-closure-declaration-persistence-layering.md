---
title: "#4117 scoping: closure-declaration persistence stays xtrax.cli-only, no controller/ import"
description: Resolves the open design question in backlog #4117's 260811 update -- xtrax.cli.manifest gains an optional, opaque closure-section parameter; it does not import or construct a ClosureManifest itself.
task_id: 260811_4117_closure_declaration_scoping
status: decided
date: '260811'
---
# #4117 scoping: closure-declaration persistence stays `xtrax.cli`-only

## Question this resolves

Backlog #4117's own 260811 update flagged one open design choice: "whether the manifest-derived
section becomes the source that constructs `BathosFrozenContext.locked`" -- i.e. should #4117
also wire the still-missing production caller of `build_closure_manifest`?

## Verified against current `origin/main` (fe16c2c)

- `xtrax.cli.config`/`manifest`/`run_verb`/`resume_verb` (the `xtrax run` verb) have zero
  references to `xtrax.loop.closure_lock`, `ClosureManifest`, or anything under `controller/`.
- `controller.evaluate_adapter.BathosFrozenContext` is constructed **only** in
  `tests/controller/test_evaluate_adapter.py` (8 sites, all literal/fake `ClosureManifest`s,
  e.g. `closure_hash="fake-hash"`) -- confirmed, not inferred: `grep -rn
  "BathosFrozenContext(" --include="*.py" .` outside `tests/` returns nothing.
- `controller/` is architecturally a separate package built *on top of* `xtrax` as a library
  (`.praxia/docs/roadmaps/loop-controller/260716_00-mandate.md`: "controller living outside
  src/xtrax importing xtrax as a library"). `xtrax.cli` importing `controller.evaluate_adapter`
  or `xtrax.loop.closure_lock` to build a `ClosureManifest` would invert that layering --
  `controller/` depends on `xtrax`, not the other way round, and nothing in the roadmap or the
  #2181 spec sanctions the reverse edge.

## Decision

**#4117 stays scoped to schema + write path only.** `write_manifest`/`write_manifest_dict`
(`src/xtrax/cli/manifest.py`) gain an **optional**, opaque closure-declaration parameter --
three plain lists of path strings (`evaluator_paths`/`split_paths`/`metric_def_paths`), matching
`ClosureManifest`'s own field names for the reader's sake but typed as `list[str] | None`, not
imported from `xtrax.loop.closure_lock`. `xtrax.cli` never imports `closure_lock` or
`controller.*`; it only accepts and serializes whatever three lists a caller passes in. Kept out
of `read_manifest`'s `required_fields` (per #4117's own highest-leverage note) -- no
`CURRENT_SCHEMA_VERSION` bump.

Wiring an actual production call to `build_closure_manifest` and threading its
`evaluator_paths`/`split_paths`/`metric_def_paths` into this new parameter is `controller/`-side
work -- the caller who already holds a real `ClosureManifest` in-process is the only place that
can supply real values. That is the still-missing "production caller" #4117's 260811 update
named, and it belongs to a **separate, controller/-scoped item** (filed as #4141, `depends_on:
[4117]`), not to #4117 itself.

## Why not fold it into #4117

Doing both in one item would require `xtrax.cli` to depend on `controller/`'s or
`xtrax.loop.closure_lock`'s types to know what a "closure declaration" even is, just to persist
three lists it never inspects. Keeping #4117 signature-only (plain strings, no closure_lock
import) means it is buildable and testable today with a fixture, exactly as #4093's own
disposition doc already assumed for the affigit-wire consumer side -- and it does not force a
decision about the controller-side production-caller design (single call site vs. baked into
`run_one_candidate_pass`'s post-dispatch path vs. a new campaign-start hook) that #4141 should
make on its own evidence, not as a rider on a schema PR.
