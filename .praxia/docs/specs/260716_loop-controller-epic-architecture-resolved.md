---
session_id: cc8f100b
topic: Loop-controller epic architecture: resolving candidate hand-off, bathos call sequencing, and repo placement for xtrax's #2181 loop controller, with praxia as the pluggable MVP dispatch backend
task_type: architectural
winner: Composite resolved architecture: (1) Fork 1 = 1A refined — a generically-scoped praxia-side write tool (e.g. write_staged_file / write_candidate_source), matching the validate_workflow_yaml.rs precedent and xtrax's bathos_sidecar_ref convention, with DispatchBackend defining an abstract hand-off contract so pluggability lives at the Protocol boundary rather than inside PraxiaDispatchBackend's implementation; 1B kept only as a documented fallback if the praxia PR stalls. (2) Fork 2 = split resolution — controller imports bathos.stats_gates and count_seeds_for_script/count_runs_for_script directly as a Python library for the pure read/compute gaps (b)/(c), following #2181's "no bathos item blocks the walking skeleton" principle; but for (d) campaign_edges/run_edges (a write to bathos's own DB, bypassing MCP's validation surface), uses the already-supported single-parent derived_from as the interim mechanism and files a non-blocking bathos-side backlog item for a true multi-parent lineage MCP tool. (3) Fork 3 = confirmed same-repo controller/ top-level directory in xtrax, with wheel-exclusion CI extended (mirroring the port/ test) and a new controller optional-dependency group added to close the residual dependency-leakage gap.
created_at: 2026-07-16T18:59:24.517357+00:00
---

# Brainstorm: Loop-controller epic architecture: resolving candidate hand-off, bathos call sequencing, and repo placement for xtrax's #2181 loop controller, with praxia as the pluggable MVP dispatch backend

## Problem Frame
FIXED CONSTRAINTS (cannot change):
- The controller lives outside src/xtrax/ and imports xtrax as a library; xtrax never imports the controller or praxia. audit_dispatch_independence.py's scope (src/xtrax only) must remain satisfied by construction.
- DispatchBackend is a pluggable interface; PraxiaDispatchBackend is one concrete implementation, not baked in as a structural dependency of the controller's core loop logic.
- Invocation mechanics are settled: plain Python subprocess to praxia's real MCP entrypoint, confirmed empirically (260716 probe doc).
- bathos call sequencing skeleton is settled: campaign_create once, one run call per candidate per iteration (add_run_to_campaign handles attachment), campaign_conclude once at the end.
- Item granularity must match #2181's own backlog convention: [XX-NN] AC-N <short description>, each sized to one PR.
- This epic is narrower in scope than #2181 — it builds the controller that consumes #2181's already-merged gate modules; it does not redesign those gates.
- The bathos MCP server staleness (Finding 2e) is an environment/deployment issue, explicitly out of scope for this session's architecture decisions.
- mode="sequential" gap in campaign_create's MCP signature (Finding 2a) is a known bathos-side gap, not something this session resolves by rearchitecting xtrax/controller.

NEGOTIABLE:
- Where candidate-source hand-off logic lives (praxia vs. controller) — Finding 1's fork.
- Whether the controller imports bathos directly as a library for the three unwired gaps, vs. blocking on new bathos MCP tools — Finding 2's (b)/(c)/(d) fork.
- The exact shape/count of child backlog items, as long as each is PR-sized and dependency-ordered.
- Whether Finding 3's controller/ same-repo placement needs any refinement (e.g., exact CI gate wording) even though the core placement call is settled.

Confirm this frame?

## Idea Pool
- [user] Three decision areas, each with competing approaches:
- [user] FORK 1 — candidate hand-off mechanics:
- [user] 1A. Praxia-side write_candidate_source MCP tool: agent's completion is passed as a tool-call argument to a new praxia tool; the tool writes to a workspace-relative staging path and returns {path, content_sha256}; controller re-reads by path and re-verifies hash. Matches every existing precedent (validate_workflow_yaml.rs pattern) and xtrax's own bathos_sidecar_ref convention.
- [user] 1B. Controller-side parsing: PraxiaDispatchBackend itself extracts a fenced-code block from the raw completion text and writes it to disk under controller-owned path/naming logic; no new praxia tool needed.
- [user] FORK 2 — bathos gap wiring (stats battery / baseline-budget-equivalence, seed-floor counts, multi-parent lineage edges):
- [user] 2A. Controller imports bathos directly as a Python library for these three specific pure-function gaps, since the controller lives outside src/xtrax and this doesn't violate any existing constraint. Ships now, no cross-repo blocking dependency.
- [user] 2B. File blocking bathos-side backlog items (new MCP tools for run_stats_battery/check_baseline_budget_equivalence, count_seeds_for_script/count_runs_for_script, campaign_edges/run_edges) and wait, keeping the controller's only bathos-integration surface as MCP calls for architectural purity.
- [user] FORK 3 — repo placement (largely settled, confirming):
- [user] 3A. Same-repo controller/ top-level directory in xtrax, wheel-exclusion CI extended, new controller optional-dependency group added.
- [user] 3B. (steelman/counter-check only) Separate repo for controller — surfaced to confirm no real counter-argument survives scrutiny, not as a genuine competing option.
- [user] Adding orthogonal options and decomposition-level ideas surfaced by the research, since the forks aren't purely binary once you look at what actually varies:
- [user] FORK 1 — a third variant worth naming even though it's not favored: 1C. Hybrid — praxia tool exists but is generic (write_staged_file, not candidate-specific), so it isn't a praxia-repo dependency unique to this epic; other future dispatch backends could reuse the same generic tool if they also run through praxia's MCP surface, while backends that don't touch praxia at all (a hypothetical local-subprocess backend) still implement their own write path via the DispatchBackend interface. This preserves pluggability at the interface level while keeping the integrity-hash property for the praxia backend specifically.
- [user] FORK 2 — noting the three gaps (b/c/d) don't have to resolve identically: it's plausible stats-battery (b) and seed-floor (c) resolve one way while lineage edges (d) resolves another, since (d) is about mutable graph state (multi-parent lineage) written back to bathos, not just reading arrays for a gate check. A pure "read query, compute in caller" pattern (b, c) is architecturally different from a "write edges back to bathos's database" pattern (d).
- [user] DECOMPOSITION CANDIDATES (raw list, to be pruned/ordered later):
- [user] controller/ top-level scaffold + package structure + wheel-exclusion CI test (mirrors port/ test) + controller optional-dependency group in pyproject.
- [user] DispatchBackend Protocol/ABC definition + a MockDispatchBackend for tests, with no praxia dependency in this item.
- [user] PraxiaDispatchBackend concrete implementation wrapping the confirmed subprocess-to-MCP invocation mechanics.
- [user] Candidate hand-off mechanism per Fork 1's resolution (either the praxia tool PR, or the controller-side parser PR — depends on Fork 1 outcome).
- [user] bathos integration adapter module inside controller/ that wraps campaign_create/run/campaign_conclude MCP calls into one cohesive sequencing helper.
- [user] bathos gap-wiring item(s) per Fork 2's resolution — direct-import wrapper(s) for stats battery/baseline-budget-equivalence and seed-floor counts, scoped separately from lineage edges if they diverge.
- [user] lineage-edges wiring item, kept separate from (b)/(c) given the mutable-write distinction just raised.
- [user] controller main loop / orchestration entrypoint that sequences: campaign_create -> per-iteration (dispatch candidate -> hand off source -> bathos run -> gate checks) -> campaign_conclude, wired to the already-merged #2181 gate modules.
- [user] first real end-to-end integration item/smoke test proving one full iteration works against real praxia + real bathos (not mocks).
- [user] A few more orthogonal items worth surfacing before converging:
- [user] A small "file bathos-side gap backlog items" item may still be warranted even under a 2A resolution — not to block, but to record the debt: campaign_create MCP mode restriction (2a, sequential missing) and the bathos MCP server staleness (2e) are real gaps that need someone to own them, even if the controller doesn't wait on them. This should probably be a single lightweight cross-repo backlog-filing item, not an architecture task.
- [user] Error/retry policy for a failed dispatch call (praxia subprocess errors, timeout, malformed completion) is implied by "controller sequences dispatch + bathos calls" but wasn't named as its own item in the raw list — worth deciding whether it's folded into the main-loop item or split out.
- [user] Given TRANSDUCTION conventions elsewhere in this workspace, the controller's own iteration records probably need a task_id-bearing log/telemetry hook — likely folds into the main-loop item rather than being separate, but flagging it so it doesn't get silently dropped from scope.
- [user] Ready to converge.

## Decision Log
- [ACCEPT] Fork 1 - 1A: Praxia-side write_candidate_source MCP tool (refined per 1C: scoped generically, e.g. write_staged_file, with DispatchBackend defining an abstract hand-off contract so pluggability lives at the Protocol boundary): Matches every precedent found (validate_workflow_yaml.rs pattern) and xtrax's own bathos_sidecar_ref convention; gets integrity-hash-at-write-time for free. The "cuts against pluggability" objection dissolves once the hand-off contract is abstracted at the DispatchBackend interface level rather than baked into the controller's core loop -- PraxiaDispatchBackend's internal use of a praxia tool is an implementation detail, not a structural dependency other backends inherit. Cross-repo dependency risk on praxia's own review process is the same category of risk the epic already accepted by choosing PraxiaDispatchBackend as MVP backend; mitigated by scoping the tool generically for low review friction.
- [DEFER] Fork 1 - 1B: Controller-side parsing of fenced-code completion, no new praxia tool: Kept as documented fallback path only, not a parallel permanent option. If the praxia-side write_candidate_source PR stalls on praxia's own release process, the controller can fall back to parsing completions directly -- but this diverges from every found precedent and loses the free integrity-hash property, so it should not be the default.
- [ACCEPT] Fork 2 - (b)+(c): controller imports bathos.stats_gates (run_stats_battery, check_baseline_budget_equivalence) and the seed/run counting functions (count_seeds_for_script, count_runs_for_script) directly as a Python library: Both are pure read-and-compute functions over caller-supplied arrays / query results -- no write-path integrity concern. The controller lives outside src/xtrax so direct import violates no existing xtrax constraint. Follows the same "no bathos item may block the walking skeleton" principle #2181 already established -- ships now instead of blocking on new bathos MCP tools.
- [REJECT] Fork 2 - (d): campaign_edges/run_edges multi-parent lineage: Direct library import rejected for this gap specifically: unlike (b)/(c), this is a write to bathos's own database, bypassing MCP's tool-layer validation/audit surface -- a real integrity risk that the pure-compute gaps don't share. Interim resolution: use the already-supported single-parent derived_from on run (no bathos change needed, ships now); file a non-blocking bathos-side backlog item for a true campaign_edges/run_edges MCP tool as future work. This is a split resolution of Fork 2, not uniform 2A or 2B.
- [ACCEPT] Fork 3 - 3A: same-repo controller/ top-level directory in xtrax: Confirmed, no surviving counter-argument. Concrete evidence: existing wheel-exclusion CI template (port/ test) directly replicable for controller/; audit_dispatch_independence.py is structurally blind to controller/ (scoped to src/xtrax only); real same-repo precedent in praxia's own monorepo (apps/agentic-ui). 3B (separate repo) was a steelman check only and does not hold up -- no working separate-repo precedent exists anywhere in this workspace. One residual gap to address in decomposition: no CI gate yet stops a controller-only dependency leaking into xtrax's main dependency list instead of a controller optional-extras group.
- [ACCEPT] INVEST gate - S (Small) criterion: Explicit override: this session's deliverable is an epic-level architecture spec + decomposition (three fork resolutions plus a multi-item backlog breakdown), not a single small feature. S is expected to fail at the epic-synthesis level by design -- INVEST's Small criterion is deferred to and will be enforced on each individual child backlog item the decomposition produces, mirroring how epic #2181 itself was structured (one mandate doc, many small [XX-NN] AC-N children each sized to one PR). User (session driver) explicitly overrides the S failure to proceed to finalize.
- [ACCEPT] Final decomposition: pruned, ordered, PR-sized child backlog items for this epic (numbering note: xtrax's next-epic prefix appears to be T3 per the mandate doc's reference to already-merged "T3-01" workflow registration; exact numeric IDs should be confirmed against the live backlog system at filing time -- this session has no backlog tool access to verify collisions): Ordered, dependency-aware decomposition derived from the three fork resolutions above:
1. [T3-xx] AC-1 controller/ top-level scaffold -- package structure, wheel-exclusion CI test mirroring the existing port/ wheel test, and a new controller optional-dependency group in pyproject.toml (Fork 3 resolution). No dependencies -- first item.
2. [T3-xx] AC-2 DispatchBackend Protocol -- abstract interface including a CandidateHandoff{path, content_sha256} hand-off contract, plus a MockDispatchBackend for tests. Zero praxia dependency in this item. Depends on (1).
3. [T3-xx] AC-3 praxia-side write_staged_file/write_candidate_source MCP tool (praxia-repo PR, generically scoped per the 1C refinement, not candidate-specific) -- Fork 1 resolution. Independent of xtrax-side items; can run in parallel with (2).
4. [T3-xx] AC-4 PraxiaDispatchBackend concrete implementation -- wraps the confirmed subprocess-to-MCP invocation mechanics, calls the tool from (3), implements the CandidateHandoff contract from (2). Depends on (2) and (3); if (3) stalls, falls back to controller-side parsing (1B) as documented fallback.
5. [T3-xx] AC-5 bathos campaign-sequencing adapter -- wraps campaign_create/run/campaign_conclude MCP calls into one cohesive controller-owned helper. Depends on (1).
6. [T3-xx] AC-6 direct bathos-library-import wrapper for stats battery/baseline-budget-equivalence + seed-floor counts (gaps b/c) -- Fork 2 resolution part 1. Depends on (1).
7. [T3-xx] AC-7 lineage interim wiring -- single-parent derived_from usage in the controller for candidate parentage (gap d interim), plus filing a non-blocking bathos-side backlog item for a true campaign_edges/run_edges MCP tool. Depends on (1).
8. [T3-xx] AC-8 controller main loop / orchestration entrypoint -- sequences campaign_create -> per-iteration (dispatch candidate -> hand off source -> bathos run -> gate checks via #2181's already-merged modules) -> campaign_conclude; owns error/retry policy for partial-iteration failures and a task_id-bearing telemetry/log hook (TRANSDUCTION convention). Depends on (2), (4), (5), (6), (7).
9. [T3-xx] AC-9 first real end-to-end integration smoke test -- one full iteration against real praxia + real bathos (no mocks), proving the whole chain composes exactly as PR #61's walking-skeleton smoke script proved the gates compose in isolation. Depends on (8).
10. [T3-xx] AC-10 cross-repo gap backlog filing -- single lightweight item recording the non-blocking debt: campaign_create MCP mode="sequential" restriction (2a), stale bathos MCP server deployment (2e), and the deferred campaign_edges/run_edges MCP tool (from item 7). Not blocking; can be filed any time.

## Assumptions

## TBDs

## Pre-mortem Record
**User:** Six months out, definitive failure mode: the praxia-side write_staged_file tool from Fork 1 never landed cleanly — it shipped, but scoped too candidate-specific after all (schema drift under review pressure), so a second dispatch backend built later couldn't reuse it and had to bypass the DispatchBackend abstraction anyway, silently reintroducing the coupling the interface was designed to prevent. Separately, the derived_from single-parent interim for lineage edges (d) never got revisited — the "non-blocking backlog item" for campaign_edges/run_edges sat unprioritized on bathos's backlog, and by month six the controller's lineage data is quietly wrong for every multi-parent merge case (best-so-far candidates re-entering as parents), because nobody owned migrating off the interim once it shipped. The controller's own main-loop item absorbed error/retry policy and telemetry hooks without a clear owner, so failures partway through an iteration (e.g., bathos run succeeds but a gate check throws) leave campaigns in an inconsistent state with no conclude call, and campaign_conclude never fires for those runs — silently corrupting the audit trail the whole hand-off design was built to protect in the first place.
**AI:** _not recorded_

## Acceptance Criteria
**Given** FIXED CONSTRAINTS (cannot change):
- The controller lives outside src/xtrax/ and imports xtrax as a library; xtrax never imports the controller or praxia. audit_dispatch_independence.py's scope (src/xtrax only) must remain satisfied by construction.
- DispatchBackend is a pluggable interface; PraxiaDispatchBackend is one concrete implementation, not baked in as a structural dependency of the controller's core loop logic.
- Invocation mechanics are settled: plain Python subprocess to praxia's real MCP entrypoint, confirmed empirically (260716 probe doc).
- bathos call sequencing skeleton is settled: campaign_create once, one run call per candidate per iteration (add_run_to_campaign handles attachment), campaign_conclude once at the end.
- Item granularity must match #2181's own backlog convention: [XX-NN] AC-N <short description>, each sized to one PR.
- This epic is narrower in scope than #2181 — it builds the controller that consumes #2181's already-merged gate modules; it does not redesign those gates.
- The bathos MCP server staleness (Finding 2e) is an environment/deployment issue, explicitly out of scope for this session's architecture decisions.
- mode="sequential" gap in campaign_create's MCP signature (Finding 2a) is a known bathos-side gap, not something this session resolves by rearchitecting xtrax/controller.

NEGOTIABLE:
- Where candidate-source hand-off logic lives (praxia vs. controller) — Finding 1's fork.
- Whether the controller imports bathos directly as a library for the three unwired gaps, vs. blocking on new bathos MCP tools — Finding 2's (b)/(c)/(d) fork.
- The exact shape/count of child backlog items, as long as each is PR-sized and dependency-ordered.
- Whether Finding 3's controller/ same-repo placement needs any refinement (e.g., exact CI gate wording) even though the core placement call is settled.

Confirm this frame?
**When** implementing Composite resolved architecture: (1) Fork 1 = 1A refined — a generically-scoped praxia-side write tool (e.g. write_staged_file / write_candidate_source), matching the validate_workflow_yaml.rs precedent and xtrax's bathos_sidecar_ref convention, with DispatchBackend defining an abstract hand-off contract so pluggability lives at the Protocol boundary rather than inside PraxiaDispatchBackend's implementation; 1B kept only as a documented fallback if the praxia PR stalls. (2) Fork 2 = split resolution — controller imports bathos.stats_gates and count_seeds_for_script/count_runs_for_script directly as a Python library for the pure read/compute gaps (b)/(c), following #2181's "no bathos item blocks the walking skeleton" principle; but for (d) campaign_edges/run_edges (a write to bathos's own DB, bypassing MCP's validation surface), uses the already-supported single-parent derived_from as the interim mechanism and files a non-blocking bathos-side backlog item for a true multi-parent lineage MCP tool. (3) Fork 3 = confirmed same-repo controller/ top-level directory in xtrax, with wheel-exclusion CI extended (mirroring the port/ test) and a new controller optional-dependency group added to close the residual dependency-leakage gap.
**Then**
  - [ ] _add specific measurable criteria_ — superseded by the filled-in Acceptance Criteria in
        §Post-Brainstorm Adversarial Revision below; the decomposition there is this epic's real AC set.

---

## Post-Brainstorm Adversarial Revision

Two independent critic passes reviewed this finalized session before filing (matching #2181's own
"adversarially approved" step, right-sized for this narrower epic): a `contemplex-brainstorm-critic`
pass and an independent `praxia-spec-challenger` pass. **Both returned NEEDS_WORK / not_ready.**
Both critics' most severe findings independently verified against real source (not taken on faith)
before acting on them. This section documents the findings and the concrete revisions applied —
the original Idea Pool/Decision Log above is left untouched as the honest historical record of the
brainstorm itself.

### Findings acted on (blocking/FATAL)

1. **The mandate's own phase 5 (multi-iteration loop, T2-09 budget/watchdog wiring, T2-18
   diversity-quota Leap-Path triggering) was silently absent from the decomposition** — a
   "loop controller" that only drives one candidate through once isn't actually a loop. Verified
   directly against `.praxia/docs/roadmaps/loop-controller/260716_00-mandate.md` §6, item 5.
   **Fixed**: new item 8b below, wiring T2-09/T2-18 into the main loop explicitly.
2. **Item 7 (lineage interim) had no degrade-behavior or revisit trigger** for the exact risk the
   session's own pre-mortem named — "lineage data is quietly wrong for every multi-parent merge
   case... nobody owned migrating off the interim." Item 10's mitigation (an unprioritized backlog
   item) was literally the same pattern the pre-mortem already predicted would fail. **Fixed**:
   item 7 revised below with explicit fail-loud behavior + a concrete revisit trigger.
3. **Item 8 (main loop) absorbed 3 separable concerns and never committed to "campaign_conclude
   fires on every exit path"** — the exact audit-trail-corruption risk the pre-mortem named.
   **Fixed**: item 8 split into 8a (sequencing) / 8b (multi-iteration + budget/Leap-Path) / 8c
   (error/retry policy with an explicit conclude-on-every-path guarantee) below.
4. **No automated gate enforces "xtrax never depends on bathos"** (a pre-existing, docstring-only
   rule — `src/xtrax/run/repro_floor.py:29-30`, `src/xtrax/loop/prereg_match.py:11-19`), unlike the
   analogous praxia-independence gate. Verified directly: `scripts/audit_dispatch_independence.py`'s
   `FORBIDDEN_PATTERNS` covers only praxia-import/mcp-praxia-tool/rig-run-dependency, no bathos
   pattern; no `audit_bathos_*.py` exists anywhere in `scripts/`. Items 6/7 (the first code in this
   repo to wire bathos-consuming logic near existing gate modules) are exactly where this could
   drift silently. **Fixed**: new item 1b below.
5. **Item 7's dependency list was incomplete** — `derived_from` resolution and `add_run_to_campaign`
   both happen inside the *same* bathos `run_script` call that item 5's adapter wraps (verified
   directly: `bathos/src/bathos/runner.py`, `derived_from` resolved ~line 345, `add_run_to_campaign`
   called ~line 547, both inside one `run_script` invocation). Item 6 (gaps b/c) genuinely doesn't
   need item 5 since those bypass MCP entirely via direct library import — items 6 and 7 are NOT
   symmetric despite an identical stated dependency in the original decomposition. **Fixed**: item
   7's dependency corrected to `(1), (5)` below.
6. **The Acceptance Criteria section was a literal unfilled placeholder.** **Fixed**: the
   decomposition below, with each item's own concrete completion condition, IS this epic's AC set —
   stated explicitly rather than left as a dangling checkbox.

### Findings acted on (major)

7. Item 3 (`write_staged_file`) had no guard against the pre-mortem's own predicted failure
   ("scoped too candidate-specific after all, schema drift under review pressure"). **Fixed**: item
   3 now requires a generic field contract (no candidate-only fields) exercised by a second,
   non-candidate call-site before merge.
8. Fork 2(d)'s read-vs-write rationale was asserted without the same evidentiary standard the rest
   of the document holds itself to. Grounding it explicitly: `bathos.stats_gates.run_stats_battery`/
   `check_baseline_budget_equivalence` and `campaigns.count_seeds_for_script`/
   `count_runs_for_script` are pure functions over caller-supplied arrays / read-only queries — no
   write path exists in their signatures. `campaign_edges.add_campaign_edge`/`add_run_edge` (per
   the original bathos-sequencing research pass) write new rows directly, with no MCP tool
   validating the write — a genuinely different risk category, not just "newer/less stable."
9. Item 4's fallback trigger ("if (3) stalls") had no operational definition, and no reconciliation
   step existed between items 2/3's independently-built contract assumptions. **Fixed**: item 4
   revised below with a concrete trigger and an explicit reconciliation checkpoint.
10. The `validate_workflow_yaml.rs` precedent cited for item 2's `CandidateHandoff` contract has
    fail-soft semantics (silently omits `staged_path`/`content_sha256` on a write failure) and an
    8-hex-char (32-bit) hash-truncated filename — neither carried into the new contract. **Fixed**:
    item 2 revised below with an explicit failure variant and a longer hash requirement.
11. `MockDispatchBackend` (item 2) was named with no behavior contract, leaving item 8c's
    error/retry paths with no described place to actually be exercised in CI. **Fixed**: item 2
    revised below with a concrete mock behavior contract.
12. Item 9's description ("proving the whole chain composes exactly as PR #61...") was aspirational
    comparison language, satisfiable by one trivial happy-path run. **Fixed**: item 9 revised below
    to name concrete required exercise paths.

### Findings noted, not separately fixed (minor / already adequately covered)

Empty `Assumptions`/`TBDs` sections and the unfilled AI pre-mortem entry (both flagged by the
spec-challenger pass) are subsumed by this revision section directly documenting the assumptions
that mattered enough to act on; terminology nits (disambiguating "xtrax.loop" the existing package
from "the controller's main loop," and how a praxia-repo PR's completion maps to xtrax's own
backlog tracking) are worth a one-line clarification in the child items themselves at filing time,
not architecture-level rework.

## Final Decomposition (revised, dependency-ordered, PR-sized) — this epic's Acceptance Criteria

Numbering unchanged from the original session (`[T3-xx]` placeholders — confirm real IDs against
the live backlog at filing time; this session had no backlog-query tool to check collisions).
**Dependencies are cited by AC-label, not list position** — a first draft of this revision used
positional `(N)` references and got at least two of them wrong the moment items were inserted
(self-caught on re-read before filing; see note at the end of this section). AC-labels are stable
under insertion, matching the `depends_on: [T2-04, T1-07]`-style convention #2181's own DAG doc
already uses, for exactly this reason.

- **`[T3-xx] AC-1`** `controller/` top-level scaffold — package structure, wheel-exclusion CI test
  mirroring the existing `port/` wheel test, new `controller` optional-dependency group in
  `pyproject.toml`. **AC**: `uv build` produces a wheel with zero `controller/` entries, verified
  by the new test. `depends_on: []`.
- **`[T3-xx] AC-1b`** (new — finding 4) bathos-independence audit gate, mirroring
  `audit_dispatch_independence.py`'s exact pattern (import/mcp-tool/identifier grep), scoped to
  `src/xtrax` (matching the existing rule's own scope — `controller/` legitimately imports bathos
  and must NOT be scanned). **AC**: the new script exits 1 on an injected `import bathos` in
  `src/xtrax/`, exits 0 today (zero existing violations). **`depends_on: []`** — this gate protects
  `src/xtrax` as it exists today and needs nothing from `controller/`'s own scaffold to exist
  first; independent of and parallelizable with every other item here.
- **`[T3-xx] AC-2`** `DispatchBackend` Protocol — abstract interface including a
  `CandidateHandoff{path, content_sha256}` hand-off contract (revised per finding 10: an explicit
  `CandidateHandoffFailure` variant for a staging-write failure, and `content_sha256` computed
  from the FULL hash, not an 8-char prefix, given a long-running loop's collision exposure), plus
  a `MockDispatchBackend` (revised per finding 11: must support deterministic candidate return AND
  configurable failure-injection modes — timeout, malformed completion, a `CandidateHandoffFailure`
  — specifically so `AC-8c`'s error/retry paths are unit-testable without real infra). Zero
  praxia dependency. **AC**: `MockDispatchBackend`'s failure-injection modes are exercised by at
  least one test per mode. `depends_on: [AC-1]`.
- **`[T3-xx] AC-3`** praxia-side `write_staged_file`/`write_candidate_source` MCP tool (praxia-repo
  PR, generically scoped). **AC** (revised per finding 7): the tool's field contract excludes any
  candidate-specific field names, AND is exercised by a second, non-candidate call-site before
  being considered merge-ready — closing the pre-mortem's own predicted "scoped too specific"
  failure. `depends_on: []` — independent of every xtrax-side item; can run in parallel with `AC-2`.
- **`[T3-xx] AC-4`** `PraxiaDispatchBackend` concrete implementation — wraps the confirmed
  subprocess-to-MCP invocation mechanics, calls `AC-3`'s tool, implements `AC-2`'s
  `CandidateHandoff` contract. **AC** (revised per finding 9): before implementation starts, an
  explicit reconciliation step confirms `AC-3`'s actual merged tool schema matches `AC-2`'s
  `CandidateHandoff` contract exactly — owned by whoever picks up this item. Fallback trigger to
  controller-side parsing (1B, documented fallback) is concretely defined as: no praxia-side merge
  or explicit maintainer rejection within a stated review window (a specific number of weeks — set
  at filing time by whoever owns the praxia-repo relationship, not invented here); if triggered,
  the fallback must still produce a valid `CandidateHandoff`, computing `content_sha256` itself
  (a self-computed hash, not an independent integrity check — documented as a known, accepted
  trust-model difference from the primary path). `depends_on: [AC-2, AC-3]`.
- **`[T3-xx] AC-5`** bathos campaign-sequencing adapter — wraps `campaign_create`/`run`/
  `campaign_conclude` MCP calls into one cohesive controller-owned helper, including
  `derived_from` pass-through from the outset (closing finding 5's dependency gap preemptively —
  `AC-7` no longer needs to retrofit this). **AC**: the adapter's `run` wrapper accepts
  `derived_from` as a first-class parameter. `depends_on: [AC-1]`.
- **`[T3-xx] AC-6`** direct bathos-library-import wrapper for stats-battery/baseline-budget-
  equivalence + seed-floor counts (gaps b/c) — grounded per finding 8 (both are pure
  read/compute, no write path). **AC**: no MCP call appears anywhere in this item's code path.
  `depends_on: [AC-1]`.
- **`[T3-xx] AC-7`** lineage interim wiring (revised per finding 2) — single-parent `derived_from`
  usage via `AC-5`'s adapter for candidate parentage. **AC**: explicit fail-loud behavior — if the
  controller ever attempts to record a genuinely multi-parent merge (a ratcheted best-so-far
  candidate with more than one real parent) under this single-parent interim, it MUST raise a
  named exception, never silently pick one parent or silently drop lineage. Revisit trigger:
  re-evaluate migrating off this interim the first time that exception actually fires in a real
  campaign (not "any time," per the pre-mortem's own warning against an unprioritized backlog
  item). Also file the non-blocking bathos-side backlog item for a true `campaign_edges`/
  `run_edges` MCP tool. **`depends_on: [AC-1, AC-5]`** (corrected per finding 5 — a first draft of
  this revision wrote the wrong positional reference here; needs `AC-5` specifically, since
  `derived_from`'s pass-through lives in that adapter, not just any earlier item).
- **`[T3-xx] AC-8a`** controller main-loop sequencing — one-candidate pass: dispatch → hand off
  source → bathos `run` via `AC-5` → gate checks via #2181's already-merged modules. **AC**: a
  single iteration completes end-to-end against `MockDispatchBackend`.
  `depends_on: [AC-2, AC-4 (or its 1B fallback), AC-5, AC-6, AC-7]`.
- **`[T3-xx] AC-8b`** (new — finding 1) multi-iteration loop wiring: the actual "keep going"
  logic across candidates, external-stop-watchdog integration (T2-09's already-merged primitive
  supervises the whole run), diversity-quota Leap-Path triggering (T2-18's already-merged
  `assert_diversity_quota` decision — this item wires the *response* to a `leap_path_required`
  signal, which T2-18 itself never implements). **AC**: a real multi-iteration run (≥3
  candidates) terminates correctly on both a budget-exhaustion path and a normal-completion
  path. `depends_on: [AC-8a]`.
- **`[T3-xx] AC-8c`** (split from the original item 8, per finding 3) error/retry policy +
  task_id-bearing telemetry hook. **AC** (the pre-mortem's own concern, made concrete and
  testable): `campaign_conclude` (or an equivalent close-out call) fires on every code path that
  exits the loop — success, a caught per-candidate failure, or an uncaught exception — verified
  via a test that injects a failure mid-iteration (using `AC-2`'s `MockDispatchBackend`
  failure-injection modes) and asserts conclude still fires with an appropriate failure/aborted
  status. `depends_on: [AC-8a]`.
- **`[T3-xx] AC-9`** first real end-to-end integration smoke test against real praxia + real
  bathos (revised per finding 12 — no mocks, and must exercise, at minimum: one candidate that
  passes every gate, one candidate that fails a gate and produces a `needs_work`/rejection
  verdict without corrupting campaign state, and at least 2 iterations to exercise `AC-8b`'s
  multi-iteration path) — matching the rigor that actually found PR #61's 2 real bugs, not a
  single lucky happy-path run. `depends_on: [AC-8a, AC-8b, AC-8c]`.
- **`[T3-xx] AC-10`** cross-repo gap backlog filing — records `mode="sequential"` restriction on
  `campaign_create`'s MCP signature, the stale bathos MCP server deployment, and the deferred
  `campaign_edges`/`run_edges` MCP tool (cross-referenced from `AC-7`'s revisit trigger, not a
  duplicate unprioritized item). Not blocking; `depends_on: []`, can be filed any time.

**Self-check performed before filing**: re-derived every `depends_on` edge above by tracing what
each item's own AC text actually requires (not by trusting the first draft's positional-reference
version) — this caught the `AC-7`→`AC-5` error and confirmed `AC-1b` has no real dependency on
`AC-1`, both described inline above. No further inconsistencies found on this second pass.
