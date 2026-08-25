# Scope: upstreaming prolix's profiling + ProbeRecord tooling into xtrax

- **Date:** 2026-08-24
- **Status:** SCOPE (pre-implementation; no code moved)
- **Source of truth for the tooling:** `~/projects/prolix` branch `wt-20260807-132628`, tree `.claude/worktrees/wt-20260807-132628`
- **Governing spec on the prolix side:** `.praxia/docs/specs/260817_jax-profiling-optimization-workflow.md` (esp. §1 layered split, §P1 claim contract, §4 follow-ups, S8 trigger)
- **Coordination:** research performed by two Stealth/ox-alpha workers (`prolix-profiler-research`, `xtrax-integration-recon`) plus coordinator firsthand reads. No other model families used.

---

## 0. TL;DR

prolix's `scripts/profiling/` package was built dependency-free against a deliberately
"xtrax-shaped seam" — its own docstring names `xtrax.profiling` as the intended eventual
home, and prolix spec §4 defers that upstream behind the **S8 trigger** ("a second
consumer repo asks, OR the API survives two complete hunt loops unchanged"). This scoping
request is reasonably read as the first half of S8 firing.

What ports: 5 library modules (~850 LOC, stdlib-only at module scope, JAX imported lazily),
~1,150 LOC of tests, and one fixture README. What does not port: MD-domain scope-label
vocab, SLURM orchestration, bathos catalog joins (pattern only), and prolix's git-provenance
env var name. Estimated effort for phases A–C: **5–7 working days**. Phase A alone (core
port, green CI) is **1–2 days**.

The single most consequential decision below is **D1: land it as public package
`src/xtrax/profiling/`, NOT under `devtools/`** — devtools is excluded from the wheel,
and the entire point of the seam is that prolix can later consume it via a dependency bump.

## 1. Provenance and trigger status

Facts established from the prolix worktree:

1. `scripts/profiling/__init__.py` docstring: *"this is the seam intended for a possible
   later upstream to xtrax.profiling."* The no-prolix-imports rule is grep/AST-enforced by
   `tests/profiling/test_claim_contract.py` precisely so upstreaming is a move, not a rewrite.
2. Spec §4 follow-up row: *"Upstream `scripts/profiling/` *code* to `xtrax.profiling` …
   requires PR + alpha release + a prolix dependency bump to consume it. Deferred behind the
   S8 trigger: a second consumer repo asks, OR the API survives two complete hunt loops
   unchanged."*
3. Distinct from P9 (the `jax-profiling` *skill*, prose layer, planned at
   `agent_assets/skills/jax-profiling/SKILL.md` in xtrax). That skill exists today only as
   DRAFT in prolix (`.praxia/docs/specs/260820_jax-profiling-skill-DRAFT.md`); nothing has
   landed in xtrax (`agent_assets/skills/` contains only `using-xtrax`). P9 shipping was
   explicitly stated NOT to fire S8.
4. Governance note for this scope: user-requested adoption by xtrax satisfies "a second
   consumer repo asks" more directly than any speculative reading. Record this as the
   trigger event in both repos' backlogs rather than re-litigating it per phase. The prolix
   side must still do its own dependency-bump release whenever it starts consuming;
   nothing in xtrax blocks on that.

## 2. Inventory of what exists in prolix

### 2.1 Library (`scripts/profiling/`, ~854 LOC)

| Module | LOC | Role |
|---|---|---|
| `record.py` | 353 | `ProbeRecord` frozen dataclass: stage/scale-stamped profiling measurement with auto-captured provenance (git_sha, timestamp, x64_enabled, jax/jaxlib versions, XLA_FLAGS, device_kind), fail-closed JSON round-trip, `write/read/restamp_git_sha`. JAX imported lazily inside `default_factory` only. |
| `claims.py` | 277 | Machine-checked claim-validity contract v3.0: `ClaimClass` {STRUCTURAL, DISPATCH_COUNT, TERM_RANKING, END_TO_END}, REQUIRED_METRICS, fail-closed `select_sources`, `paired_configs` (axis × hold_fixed pairing), `assert_claim_supported` (stage≥2 GPU floor for TERM_RANKING via roofline argument; END_TO_END target declaration + `SCALE_EXTRAPOLATION_LIMIT=10.0`; unanimity on x64_enabled/xla_flags/device_kind/platform/git_sha; unverifiable-sha rejection). |
| `report.py` | 181 | Claim-gated Markdown bottleneck ranking from discovered records under `outputs/probiling/stage*/*.json` (sic: `outputs/profiling/`); skips `*_summary.json`/`coverage.json`; renders absent scopes as literal `"absent"`, never `0.0`; MIXED ATTRIBUTION banner when named_scope/op_name mix. |
| `trace.py` | 297 | Pure Perfetto-trace + compiled-HLO-text parser. Two-input attribution because named_scope labels do NOT surface as trace slices on this JAX install: executed events carry post-fusion thunk names in `args["hlo_op"]`; scope paths survive only in compiled HLO `op_name` metadata. Deepest-known-label attribution, transform-wrapper unwrapping (`vmap(jvp(label))`), depth cap 16. Also dispatch counters. |
| `__init__.py` | 36 | Public API: `CONTRACT_VERSION`, `SCALE_EXTRAPOLATION_LIMIT`, `ClaimClass`, `ClaimValidityError`, `ProbeRecord`, `assert_claim_supported`, `paired_configs`, `permitted_claims`, `select_sources`. |

Emission style: explicit construction + explicit `.write(path)`; no decorator or context
manager. Canonical emitter `_emit_probe_record` lives in
`scripts/experiments/profile_b1_flash_vs_autodiff_forces.py` (domain layer, not the library).
Hard-won pin: keep an EMPTY `attribution_method` dict when every scope is None — `or None`
there caused a real cluster failure (pinned by `tests/profiling/test_emit_probe_record.py`).

### 2.2 Tests (~1,152 LOC) and fixtures

- `test_claim_contract.py` (572): ~40 named tests mirroring spec P1 verbatim; includes the
  AST no-import check and relative-import rejection.
- `test_trace_parse.py` (233): pins parser behavior against a committed real CPU trace
  fixture (<1 MB budget) and regenerated-on-the-fly HLO text.
- `test_report.py` (85): discovery, column order, raise-over-stage0/1, banner logic.
- `test_emit_probe_record.py` (34): the empty-attribution-dict regression.
- `test_capture_git_sha.py` (18): provenance fallback chain.
- Fixtures generated ONLY via `ProbeRecord.write` (README documents regeneration).

### 2.3 Domain scripts (pattern-reference only, NOT ported as-is)

Stage 0 (`cost_analysis`, never executes), Stage 1 (CPU micro under `jax.profiler.trace`),
Stage 1-perturbation (instrumentation-doesn't-perturb A/B harness with IQR-sized pair count,
seeded bootstrap CIs, PASS/INSUFFICIENT_POWER/DETECTED_PERTURBATION verdicts), Stage 2
(SLURM/H200 config sweep under bathos campaigns), coverage aggregation (fails closed if
catalog unreachable), GPU op-ranking probe (warns when CUDA graphs swallow named_scope).

### 2.4 Output layout convention

```
outputs/profiling/stage<N>/<probe_id>.json     # one file per record, pretty JSON
outputs/profiling/stage*/hlo_as_text_*.txt     # compiled HLO beside its record
tests/profiling/fixtures/...                   # committed fixtures
```

## 3. What xtrax already has (and one collision)

Recon findings that shape the port:

1. **`src/xtrax/devtools/gates/_performance_probes.py` / `_trace_probe.py`**: "probe" in
   xtrax currently means chex `assert_max_traces` count gates + wall-time medians feeding
   the audit performance gate (`audit/performance_targets.toml`, `ProbeResult`,
   `GateResult`). This is a DIFFERENT concept from ProbeRecords (persistent, portable,
   claim-validity-checked measurement artifacts). See D4.
2. **No real profiling anywhere**: zero `jax.profiler`/xprof/tensorboard/tracemalloc usage
   in src/. Established idioms to reuse: `time.perf_counter` + `jax.block_until_ready()`
   (`loop/compile_time_clock.py`), median-of-N discipline (`gates/performance.py`).
3. **Wheel boundary**: `pyproject.toml` excludes `src/xtrax/devtools` from the wheel and
   coverage. Anything prolix must import cannot live there → D1.
4. **Style ratchets that bind new code**:
   - NO `from __future__ import annotations` (allowlist-gated by
     `tests/audit/test_no_future_annotations.py`); prolix's modules use it and must drop it.
   - Beartype+jaxtyping import hook runs over all `xtrax.*` during tests unless
     `XTRAX_DISABLE_BEARTYPE=1`; new annotations are runtime-checked.
   - Frozen-slots dataclasses, per-module exception hierarchies, TOML-under-`audit/` config
     loaders, grounding-first docstrings citing specs.
   - Canonical gate recipe (Justfile ~84-87): `ruff check` → targeted pytest →
     `scripts/audit_*_gate.py --no-write-baseline`. New tooling should ship a matching
     `audit-*` target + script wrapper.
5. **bathos hard boundary**: `src/xtrax` must never depend on bathos (enforced by
   `scripts/audit_bathos_independence.py`); only top-level `controller/` may. ProbeRecords
   are plain files so this is naturally satisfied; campaign attachment happens in controller.
6. **JSONL stream conventions** (`.praxia/*.jsonl` + locks) apply to loop telemetry, not to
   measurement artifacts. Keep one-JSON-file-per-record; see D6.

## 4. Design decisions

### D1 — Land as public `src/xtrax/profiling/` (not devtools)

Rationale: the pre-planned consumer is prolix via dependency bump ("alpha release +
dependency bump", spec §4). devtools is wheel-excluded and would foreclose exactly the
seam the package was shaped for. Subpackage-public (importable as `xtrax.profiling`) but
NOT re-exported from the top-level `xtrax/__init__` (respects N-series lazy-API direction;
decide `__getattr__` re-export later on demand).

Keep the package **self-contained**: grep-enforce "no imports from `xtrax.*` siblings"
inside `xtrax/profiling/` (the mirror image of prolix's no-prolix-imports test). It may
lazily import jax/jaxlib only. This preserves both directions of future reuse.

### D2 — Contract version stays `3.0`

The JSON schema is unchanged by the move: import paths and env-var names are not fields.
MAJOR bump rule stays documented in-module. If we ever rename/add REQUIRED_METRICS or
fields, follow P1's bump rule (removal of a required metric ⇒ MAJOR; plain-default field
addition ⇒ MINOR).

### D3 — Provenance env var rename

`PROLIX_GIT_SHA` → `XTRAX_GIT_SHA`. Keep the repo-root `.git_sha` sidecar convention and
the `-dirty`/`-unverified` sentinels and `restamp_git_sha` chain-of-custody (myxcel-style
cluster scratch without `.git` applies to xtrax cluster runs equally).

**Port bug to avoid:** `_REPO_ROOT = Path(__file__).resolve().parents[2]` is correct for
`scripts/profiling/record.py` but resolves to `<repo>/src` for
`src/xtrax/profiling/record.py`. Use `parents[3]` (or derive from an anchor like
`pyproject.toml` existence) and add a test asserting `_REPO_ROOT` contains `.git` or
`pyproject.toml`.

### D4 — Naming: ProbeRecord vs existing gate "probes"

Keep the name `ProbeRecord` (contract stability, cross-repo vocabulary). Document the
distinction in the package docstring: gate probes (`devtools/gates/_trace_probe.py`,
`ProbeResult`) are ephemeral pass/fail CI checks; ProbeRecords are durable evidence
artifacts evaluated against the claims contract. Do NOT rename existing gate symbols.

### D5 — Style conformance deltas from prolix source

Mechanical rewrites on port:
- Drop `from __future__ import annotations` (all five modules). Verify PEP 604 unions in
  dataclass field annotations resolve under the beartype hook (they do on py3.13).
- Carry over the `ty: ignore[invalid-argument-type]` comment on `dataclasses.fields(cls)`
  in `from_json` (documented ty mis-resolution).
- Convert module docstrings' prolix spec citations into xtrax-relative pointers to THIS
  document plus the prolix spec path (kept as historical rationale).
- Add grounding-first headers citing this spec, matching xtrax conventions.

### D6 — Artifact layout

Adopt `outputs/profiling/stage<N>/<probe_id>.json` unchanged (report discovery logic
depends on it). Fix report.py's cwd-sensitivity while porting: resolve default root from
`_REPO_ROOT`/explicit arg instead of `Path.cwd()`. Do NOT route records through
`.praxia/*.jsonl`: those are append-only loop telemetry streams; records are random-access
evidence files. Optional bridge (Phase C): emit an `AuditFinding` (dim="performance",
payload={probe_path}) into audits.jsonl when a claim assertion FAILS inside a gate context.

### D7 — bathos boundary

Unchanged: records are plain files; only `controller/` may attach them to bathos
campaigns/run metadata (hook next to `run_one_candidate_pass`'s injectable `*_fn`s /
`on_loop_event`). Coverage-aggregation-via-catalog is a pattern note only; any xtrax
version must live under controller/ and keep failing closed when the catalog is
unreachable.

### D8 — Domain vocab stays out of the core package

`KNOWN_LABELS` frozensets of MD scope names, the 5-key config vocab, protein/pme/grid
cells: all stay in probe drivers (xtrax defines its OWN label sets for sparse/tiling/
training kernels). Core package remains generic: `probe_id/stage/n_atoms/platform/metrics/
scopes/config/attribution_method`. `n_atoms` is the right general axis for xtrax too
(structural-biology workloads); do not generalize to `n_units` (would be a MAJOR contract
bump for zero benefit).

### D9 — JAX-version re-spike obligation

trace.py's event-name expectations (`args["hlo_op"]`, thunk naming, µs units,
command-buffer swallowing on GPU) were empirically resolved on prolix's pinned JAX
(0.10.2 fixtures). Before trusting Stage-1+ traces from xtrax runs, redo the short P4-style
presence-not-spelling spike on xtrax's installed JAX version and record the result here.
Budgeted inside Phase B.

**Spike RESULT (2026-08-24, xtrax venv):** jax/jaxlib 0.10.2 (identical major.minor to
prolix's fixtures). CPU-only jaxlib despite a machine GPU being present ("CUDA-enabled
jaxlib is not installed") -- xtrax probes stay Stage 0/1 until a CUDA jaxlib lands here.
Confirmed present, same shapes as prolix:
`CommonPjRtLoadedExecutable::ExecuteHelperOnSingleDevice` (1:1 with jitted calls);
`PjitFunction(<fn>)` at 2x per Python-level call (enter+exit bookkeeping);
`backend_compile_and_load` appears iff compilation happens inside the trace window
(pre-compiling outside yields 0 -- warm records legitimately have n_compilations=0);
executed-thunk events carry post-fusion names in `args["hlo_op"]`
(`broadcast_multiply_fusion`, `wrapped_reduce`) plus near-zero-duration `"end: "`
bookkeeping events; compiled HLO text carries
`op_name="jit(fn)/<named_scope>/<primitive>"` so scope paths survive ONLY in HLO text.
Perfetto export kwarg on this install is
`jax.profiler.trace(dir, create_perfetto_trace=True)` (not `create_perfetto_trace_file`).
Conclusion: trace.py's parsers are valid AS-IS on xtrax's current install; re-spike again
on any JAX upgrade.

## 5. Phased plan

### Phase A — core package port (est. 1–2 days)

1. Copy 5 modules verbatim → apply D3/D5 mechanical deltas.
2. Port tests to `tests/profiling/` (claim-contract, trace-parse, report, emit-regression,
   capture-git_sha). Rewrite the AST test's allowed-import constant
   (`scripts.profiling` → `xtrax.profiling`) and ADD the sibling-import prohibition from D1
   and the `_REPO_ROOT` sanity check.
3. Regenerate fixtures via `ProbeRecord.write`; rewrite fixtures README for xtrax commands.
4. New Justfile recipe + `scripts/audit_profiling_contract.py` wrapper following the
   canonical gate pattern; wire into CI workflow list.
5. Green: ruff + pytest + beartype-hooked import + ty clean.

Exit criteria: full ported suite passes in xtrax CI; package importable as
`xtrax.profiling`; wheel build excludes nothing unexpected (`pytest --collect-only` on
published-wheel simulation optional).

### Phase B — first xtrax-native probes (est. 2–3 days)

1. Define xtrax scope-label registry (start: training step, tiling strategies
   Vmap/SafeMap/DedupGather, sparse_filter_jit kernel, compile vs step split from
   `compile_time_clock`).
2. Generic `_emit_probe_record` helper adapted from prolix's domain emitter, parameterized
   on config vocab; unit-pin the empty-attribution-dict case.
3. Stage-0 cost-analysis script over one representative xtrax kernel; Stage-1 CPU micro
   harness (trace + HLO text + parse_scopes) over the same; D9 spike recorded.
4. Optional: persist pytest-benchmark stats from `benchmarks/conftest.py` as Stage-1
   records so benches leave durable artifacts.

Exit criteria: ≥3 committed example records under `outputs/profiling/`; `render_report`
produces a claim-gated table from them; perturbation harness demonstrated once on one
scope group.

**Phase B STATUS (2026-08-24):** core delivered, two items consciously deferred.
Delivered: xtrax scope-label registry (tiling_vmap / tiling_safemap / tiling_dedup_gather,
driver-local per D8); generic emitter `xtrax.profiling.emitters` with the
empty-attribution regression pinned by tests; Stage-0 cost-analysis driver over the three
tiling strategies (`scripts/prof_stage0_tiling_cost.py`, never executes); Stage-1 CPU
micro driver (`scripts/prof_stage1_tiling_micro.py`: one jitted program applying all
three strategies under named_scopes, trace + HLO-text attribution via
`scope_map_from_hlo_text`/`parse_scopes`/`parse_dispatch_counts`, warm-up outside the
timed window); 4 committed example records + HLO text under `outputs/profiling/`
(regeneration commands in its README); all three labels recovered with real non-zero
exclusive time and named_scope attribution. D9 spike result recorded above.

Deferred, with rationale:
1. *Claim-gated table from xtrax-native records* -- impossible honestly on this machine:
   render_report's TERM_RANKING gate requires Stage>=2 GPU sources and this box has a
   CPU-only jaxlib (D9 note). The fail-closed raise IS demonstrated live by the stage-1
   driver; table rendering itself stays covered by the fixture-based tests until GPU data
   exists.
2. *Perturbation harness demonstration* -- its purpose is to certify that instrumentation
   doesn't perturb a hot path backing a citable claim. No xtrax claim is backed by these
   records yet (nothing above STRUCTURAL/DISPATCH_COUNT), so there is no decision the
   harness would de-risk today. Revisit together with the first performance-backed claim
   or the perf-gate integration (Phase C).

### Phase C — integration surfaces (est. 2 days)

**Phase C STATUS (2026-08-24): delivered, with one scope adjustment.**
1. Dispatch tripwires: `_dispatch_probe.py` measures
   n_executions/n_compilations/n_jit_traces from a real traced run (one guarded callable
   built ONCE and reused -- run_trace_gate's per-call re-wrap recompiles inside the window
   and poisons n_compilations); `performance.py` ProbeSpec gains opt-in `max_compilations`
   /`max_jit_traces`/`emit_probe_record`; violations emit major findings
   (`violation_kind=dispatch_count`) and count into a NEW baseline metric
   `performance.dispatch_violation_count`. Ratchet semantics preserved: the dispatch metric
   is evaluated ONLY when some probe configures ceilings (otherwise bootstrap-on-missing-key
   would stamp 0.0 into every repo's baseline uninvited), so the repo's own
   performance_targets.toml + audit_baseline.json behave byte-identically pre/post Phase C
   (pinned by tests). Ceilings for the real sparse kernel deliberately NOT set yet --
   thresholds need data from CI runs first (scope doc Phase D item).
2. Controller hook: `run_one_candidate_pass(..., probe_record_dir=...)` writes one Stage-0
   provenance record per COMPLETED pass after all gates resolve: campaign_id, derived_from,
   handoff sha, host wall seconds, accepted/hard_blocked verdicts. Hard-blocked passes still
   record (forensics; outcome in config); exceptions leave no record. The only
   bathos-adjacent site permitted by D7.
3. Docs: `agent_assets/skills/using-xtrax/references/profiling.md` (TIER-2 reference,
   Verify-path citation style). Justfile `audit-profiling-contract` recipe extended to cover
   the Phase C files.

Scope adjustment vs original plan: baseline-ratchet entries for wall-time metrics were
NOT added beyond the dispatch metric -- adding ratchets without threshold data would
fabricate anchors; same reasoning as (1).


1. Extend `gates/performance.py` + `performance_targets.toml` with profiler-backed probe
   kinds (dispatch-count tripwire: n_compilations/n_jit_traces ceilings; optional
   wall-time→ProbeRecord emission), keeping existing ProbeResult plumbing intact.
2. Baseline-ratchet entries for any new metrics (via `devtools/baseline.py` comparators).
3. Controller hook: opt-in per-candidate ProbeRecord capture attached to run tags in
   `main_loop.run_one_candidate_pass` (only bathos-touching site).
4. Docs: a `references/profiling.md` under `agent_assets/skills/using-xtrax/` following the
   Verify-path citation style.

### Phase D — follow-ups (backlog items, not this scope)

- Land the P9 `jax-profiling` skill (fill its §Open section from Phase-B experience).
- CI dispatch-count tripwire promotion (fast job) once thresholds exist.
- GPU/Stage-2 story for xtrax (device matrix TBD; myxcel profiles already exist in-repo).
- prolix-side: dependency bump consuming `xtrax.profiling` and deletion of
  `scripts/profiling/` bodies (keep re-export shim one release).

## 6. Acceptance criteria

- [ ] `xtrax.profiling` importable, self-contained (no sibling imports; AST-tested).
- [ ] Ported contract suite green under beartype hook; fixtures regenerate from committed
      commands only.
- [ ] No `from __future__ import annotations` in any new file; ruff/ty clean.
- [ ] `render_report` refuses TERM_RANKING from stage<2 sources inside xtrax (smoke test).
- [ ] Wheel-included (not under devtools/); coverage omission decision recorded
      (recommend: DO measure coverage — it's now product code).
- [ ] This document updated with the D9 spike outcome before Stage-1 records are trusted.

## 7. Risks and open questions

| # | Item | Disposition |
|---|------|-------------|
| R1 | Trace-parser brittleness across JAX versions | D9 spike; presence-not-spelling asserts carried in ported tests |
| R2 | Beartype runtime-checks vs lazy jax imports in default_factories | Imports are function-local; hook only wraps annotations; low risk, verified in Phase A |
| R3 | Coverage-gate interaction: new public package raises coverage floor | Decide in Phase A whether to exempt stage-2-only paths; prefer measuring |
| R4 | Double-maintenance while prolix still owns its copy | Keep prolix shim re-exporting after its bump (Phase D); otherwise drift returns |
| Q1 | Does xtrax want records under git LFS/committed examples? | Recommend committing 2-3 tiny synthetic records as fixtures only; run outputs stay untracked |
| Q2 | Python floor: prolix fixtures pinned jax 0.10.2; confirm xtrax's jax lower bound parses identically | Part of D9 |
| Q3 | Should END_TO_END's SCALE_EXTRAPOLATION_LIMIT=10.0 keep its MD-derived justification text? | Keep constant + rewrite justification around xtrax workloads (tiling memory regimes) |

## Appendix — file-by-file port manifest

| prolix source | xtrax destination | Delta class |
|---|---|---|
| `scripts/profiling/__init__.py` | `src/xtrax/profiling/__init__.py` | imports, docstring (D1/D5) |
| `scripts/profiling/record.py` | `src/xtrax/profiling/record.py` | D3 env var, `_REPO_ROOT` fix, D5 |
| `scripts/profiling/claims.py` | `src/xtrax/profiling/claims.py` | D5 only |
| `scripts/profiling/report.py` | `src/xtrax/profiling/report.py` | cwd→root resolution (D6), D5 |
| `scripts/profiling/trace.py` | `src/xtrax/profiling/trace.py` | D5 (+D9 asserts where cheap) |
| `tests/profiling/*` | `tests/profiling/*` | import constants; add D1 AST rule + `_REPO_ROOT` test |
| `tests/profiling/fixtures/*` | `tests/profiling/fixtures/*` | regenerate; rewrite README |
| `_emit_probe_record` (experiments) | `src/xtrax/profiling/_emit.py` or probe driver util | generalize config vocab (D8) |
| stage0/1/perturbation explore scripts | `scripts/probes/` (new, xtrax-flavored) | pattern port, new label registry |
| SLURM/bathos stage-2 wrappers | NOT PORTED | controller-layer pattern only |

## Phase D progress (2026-08-25, resumed session)

**D-bench DONE**: benchmark wall-clock stats -> ProbeRecord bridge shipped.
Decision D10 (bench declaration protocol): pytest-benchmark runs have no
intrinsic stage/molecular scale, so benches DECLARE `xtrax_stage` /
`xtrax_n_atoms` / free-form `xtrax_*` config via `benchmark.extra_info`;
undeclared benches are never recorded (skipped-with-reason summary).
Stats schema pinned to installed Stats.fields: durations s->ms (`_ms`
suffix), counts passthrough, display-string composite `outliers` dropped
(it is "iqr;stddev" text, both components already numeric fields), unknown
fields abort loudly. Emission strictly opt-in via XTRAX_BENCH_RECORD_DIR;
off by default so local runs never dirty the tree. All three existing bench
modules now declare stage=1 / n_atoms=32 / scale_basis=batch_rows.
Verify: src/xtrax/profiling/bench.py, benchmarks/conftest.py,
tests/profiling/test_bench_records.py (incl. subprocess end-to-end).

Remaining deferred: CI-data dispatch ceilings, GPU Stage-2 story, prolix-side
dep bump, perturbation harness demo.

### Fidelity audit of the port (2026-08-25)

AST-level comparison of all five ported modules against prolix origin
(scripts/profiling @ wt-20260807-132628), after normalizing the documented
deltas (PROLIX_GIT_SHA->XTRAX_GIT_SHA, parents[2]->parents[3], slots=True,
future-annotations removal, scripts.profiling->xtrax.profiling) and stripping
docstrings: trace.py and __init__.py IDENTICAL; claims/record/report differ
in 6 top-level defs, ALL mapping to recorded decisions:

1. claims.assert_claim_supported: error-string wording ("see the prolix
   spec") -- provenance attribution only.
2. claims.paired_configs: list['ProbeRecord'] -> list[ProbeRecord] --
   consequence of the future-annotations ban (quoted ref became direct name).
3. record.ProbeRecord.metrics: dict[str, float] ->
   dict[str, float|int|str] -- the beartype-compat widening (pinned).
4. record._capture_timestamp + import: datetime.now(timezone.utc) ->
   datetime.now(UTC) -- py3.13 idiom, timezone.utc IS UTC; the ONLY
   divergence not pre-listed in a decision entry; cosmetic, zero behavior.
5. report._DEFAULT_DISCOVERY_ROOT + discover_records fallback: the D6 fix.

Conclusion: port is faithful; no undocumented semantic drift exists.
