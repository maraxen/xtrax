---
title: Apply confirmed disposition for S4 (closure-declaration schema, not one write)
description: 'User-confirmed plan: amend #4093 and file a new backlog row for the real xtrax-side scope of S4, blocked on plugin:praxia:core MCP not connecting from the affigit corpus session'
status: blocked
task_id: 260807_affigit_triage_decisions
date: '260807'
---
# Apply confirmed disposition for S4 (closure-declaration schema, not one write)

## Context

Sibling doc: `affigit-core/.praxia/docs/handoffs/260807_apply-confirmed-disposition-for-staging-escalations-1-3-*.md`
(same `task_id`) has the full cross-repo context. Short version: affigit-wire's intake escalated
"file the xtrax-side half of S4" for owner decision since we own xtrax. A follow-up investigation
found the original framing ("one write at one call site persisting an already-computed value") was
wrong on both counts — verify this is still accurate against current `origin/main` before acting,
since the investigation is now a day old. The owner confirmed: file it, at the corrected scope.

**Why this doc exists instead of the mutation already being applied:** `plugin:praxia:core` would
not connect from the session where this was decided (rooted at `~/projects/affigit`, cwd-workspace
mismatch — see sibling doc for full diagnosis). This doc carries the exact plan for a session
rooted here, where the server does connect.

## Verify before acting (premise may have drifted — investigation was same-day but not same-hour)

Re-confirm against current `origin/main`:
1. No production call site exists for `build_closure_manifest` / `verify_closure` outside
   internal recompute, `__all__`, docstring samples, and a synthetic smoke script.
2. `TrainConfig` has no closure fields; `load_config`'s `require_sections` accepts only
   `model`/`optimizer`/`loss`/`data`; `write_manifest_dict` builds the manifest purely from
   `cfg_dict` with no other input channel.
3. xtrax #3657 is still open at P1, still unresolved on where closure enforcement belongs.

If any of these three has changed, stop and re-derive the plan below rather than applying it
as-is — it was scoped specifically against these three facts.

## Action items

### 1. Amend #4093's description

Correct it to state plainly: there are no production call sites for closure persistence today;
the spec's trigger moment ("the moment a run first calls `build_closure_manifest`") never occurs
in production. This is a schema change (new closure section in TOML/manifest), not a write path
retrofit.

**Add a structured outcome** (do not add a hard `depends_on` edge to #3657): #4093's verifier
should report `closure_declaration_absent` as a named outcome on every run where the closure
section is missing. This surfaces the gap continuously and applies pressure on #3657 without
blocking on it. (Cross-workspace `depends_on` edges do work in production, confirmed — declining
one here is a scheduling choice, not a capability limit. Available later at zero cost if the
verifier should ship as fully blocked instead.)

### 2. New backlog row — closure-declaration schema work

- **Priority: P2** (not P1 — xtrax's open backlog is already 22 P1 / 7 P2 / 1 P3; a second P1
  downstream of a blocked P1 (#3657) dilutes signal without accelerating anything).
- **Category/difficulty: extended, research** (schema change + call-site threading through
  `write_manifest`, `run.py`, `resume_verb.py`, plus a `read_manifest` required/optional decision
  — not mechanical).
- **`depends_on: [3657]`** (this IS the hard dependency — the schema work should genuinely wait
  for #3657 to resolve where closure enforcement belongs; only #4093's *verifier* avoids blocking
  on it, per above).
- **Highest-leverage design note to include in the item:** add the closure block as an *optional*
  manifest key and keep it out of `read_manifest`'s `required_fields`. This avoids a
  `CURRENT_SCHEMA_VERSION` bump entirely — a bump invalidates every existing run manifest and
  every existing user TOML, so it's worth designing around rather than accepting as a cost.

## Verification before closing this item

- `backlog(action="update", payload={id: 4093, description: <amended>})` succeeded and reads back
  via `backlog(action="list", ...)`.
- New backlog row exists with `depends_on: [3657]` present in `detail:"with_deps"` listing.
- Confirm #4093 has no hard dependency edge added (only the structured-outcome change).

## Outcome (260807, applied from a session rooted in xtrax)

Premise re-verified against current `origin/main` first, all three held:
1. Confirmed via `rg` — `build_closure_manifest`/`verify_closure`'s only callers are
   `evaluator_change_gate.py`'s docstring example, `closure_lock.py`'s own `__all__`, and
   `scripts/smoke_2181_walking_skeleton.py`. No production call site.
2. Confirmed by reading `src/xtrax/cli/config.py` and `src/xtrax/cli/manifest.py` directly —
   `TrainConfig` has exactly `schema_version/model/optimizer/loss/data/seed/num_epochs`, no closure
   field; `require_sections` is called with only `("model","optimizer","loss","data")`;
   `write_manifest_dict` builds the manifest purely from `cfg_dict`, and `read_manifest`'s
   `required_fields` list has no closure entry either.
3. Confirmed via `backlog(action="list", detail="with_deps")` — #3657 ([GW-01b]) is still open,
   P1, unresolved.

**Action item 2 (new backlog row) — DONE.** Filed as **#4112** in xtrax's own backlog:
priority P2, category `research`, difficulty `extended`, `depends_on: [3657]` (verified present
and correctly resolves as an existing open item; item reads `blocked: true` / `executable: false`
as a result, which is intended). Description carries the corrected S4 scope, the call-site list,
and the optional-manifest-key / no-schema-version-bump design note verbatim from this doc's action
item 2.

**Action item 1 (amend #4093) — BLOCKED, could not be applied from this session.** `backlog(action="list", ...)`
against xtrax's own workspace (`ws_09470f04-78a2-4a6e-9440-7635a265d4c1`) does not contain an item
#4093, and — more conclusively — the `add` call for #4112 ran this repo's own reference-existence
probe over the new item's text and returned:
```
{"literal":"#4093","kind":"referenced_item_id","claim_direction":"exists","existence":"absent","evidence":"no backlog item #4093"}
```
i.e. xtrax's backlog store itself, asked directly, says #4093 does not exist here. Cross-checked
`.praxia/identity.json`/`workspace.id` across the three affigit repos on disk: affigit-wire's
workspace id is `ws_f1c49978-f40e-44da-8ccc-2b95470cf9ed`, affigit-core's is
`ws_70353e2b-8d88-4cde-a6ef-54113f26f147` — neither matches xtrax's. #4093 was filed by the
affigit-wire intake session (per the sibling doc, "since we own xtrax" was the *rationale* for
filing it, not evidence of *where* it landed) and most likely lives in affigit-wire's backlog
store, not xtrax's.

This doc's own premise — "a session rooted here, where the server does connect" — conflated two
different failure modes. `plugin:praxia:core` connecting (true, verified: `backlog`/list/add all
worked fine above) is not the same as this session's workspace containing #4093's row. Also
attempted a `workspace` override on `list` pointed at affigit-wire's path — silently ignored, same
failure mode already on record for the `debt` tool ([[feedback_praxia-debt-tool-workspace-locked]]
in the operator's memory) and now confirmed to generalize to `backlog` as well. Did not attempt a
blind `update` on id 4093 as a workaround — an update payload with unknown current field values
risks clobbering fields on someone else's item in a workspace this session can't even read back to
verify.

**Recommended path forward:** amend #4093 from a session actually rooted in `~/projects/affigit-wire`
(or wherever #4093 concretely resolves — confirm via that workspace's own `backlog(action="list")`
before writing). The exact amendment text is unchanged from action item 1 above.
