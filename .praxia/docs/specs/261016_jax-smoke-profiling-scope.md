---
title: 'Smoke-stage JAX profiling: jaxlint host rules, xtrax profile verb, launch gate, jaxlint cisternal plugin'
description: 'Cross-repo scope for catching low-hanging JAX bottlenecks at L2: jaxlint owns static suspects, xtrax owns measurement, bathos owns the launch gate, jaxlint ships its edit hook as a cisternal plugin'
status: draft
task_id: 261016_jax-smoke-profiling-scope
date: '261016'
backlog_ids: ''
adversarial_review: ''
---
# Smoke-stage JAX profiling: jaxlint host rules, xtrax profile verb, launch gate, jaxlint cisternal plugin

## Purpose

Catch low-hanging JAX bottlenecks (host loops over device or host-numeric calls,
steady-state recompiles, per-iteration host syncs, loop bodies whose cost scales with
the loop) at the L2 smoke stage, before a long run launches. Motivating incident
(2026-10-16, denxity QB-MI): B=200 host-side NSB fits per target per replicate ran most
of a day on titanix with no production-shape profile taken first.

Scoped by four parallel read-only subagents plus one verification pass. All numbers
below are **spike results**, not findings; anything cited later gets re-run under a
bathos sidecar. Tags: [V] verified at file:line or by probe, [U] unverified.

## Separation of concerns

Test: **needs a callable plus example inputs → xtrax. Needs only source text or an IR
artifact → jaxlint.** jaxlint emits *suspects*; xtrax *measures* whether they matter.

| Concern | Owner |
|---|---|
| AST suspects in host code (bootstrap loops, fresh jit in loop, per-iteration sync) | jaxlint core (import-light, no JAX) |
| Rules over a lowered/exported IR handed in | `jaxlint[hlo]` |
| Lowering at declared shapes, compile/dispatch counts, device-vs-host split, L vs 2L loop scaling, wall estimate | xtrax `profiling/` + `xtrax profile` verb |
| Claim validity of a measurement; ProbeRecord schema | xtrax (exists: `profiling/claims.py`, `record.py`) |
| "This run cites a fresh profile" before launch | bathos (generic `[requires]` artifact gate) |
| Edit-time lint feedback to the agent | jaxlint, shipped as a cisternal plugin hook |
| L2 rule text, skills, remediation playbook | myxcel snippet, xtrax + jaxlint skills |

xtrax may feed IR it lowered into `jaxlint.hlo` rules; rule logic does not move into xtrax.

## Workstream 1: jaxlint host-side rules

**Current architecture** (origin/main d10ee00) [V]:
- Registry: `core/rules.py:8-29`. Severity is duplicated in each sensor's `Diagnostic(...)`.
- Wiring is manual: `sensors/__init__.py:13-20`.
- Sensors are per-file `ast.NodeVisitor`s with no cross-file index.
- Kernel detection is copy-pasted in `loops.py:38`, `purity.py:39` and `shapes.py:53`. It
  matches only attribute `jit`/`grad`, so it misses bare `@jit`, `@eqx.filter_jit` and
  `partial(jax.jit)`.
- No inline suppression comments.
- The disk cache is keyed on `__version__` (`cache.py:51`), so editable installs serve
  stale diagnostics after a rule change.
- Fixtures for JL003 onward are not referenced by any test.

**Key probe** [V, jax 0.11.1 / equinox 0.13.8]: re-wrapping the *same* function object
with `jax.jit`/`eqx.filter_jit` in a loop traces once; a fresh lambda/closure/partial per
iteration retraces every time.

**Calibration sweep** (1,619 files across xtrax, aminx, denxity, prolix, demistify,
plegadx, isochore; ~10 hand-classified hits per candidate):

| Candidate | Hits | Hand TP | Decision |
|---|---|---|---|
| E: bootstrap-style loop, `range(n_boot\|resampl\|replic\|n_perm\|n_draw\|...)` with non-trivial call + accumulation, device-agnostic | 25 | 2 strong + ~4 weak / 12 | **Ship first** (JL021, info). Only candidate that covers the incident class (numpy/scipy fits per replicate). |
| B1: jit-family wrap of a *fresh* callable in a loop, no loop-dependent free names | 10 | 0 confirmed avoidable; 3 unexamined | Ship (JL022, warning); precise if the dependence filter holds |
| C: `.item()` / `float()` / `np.asarray` / `device_get` / `block_until_ready` on device-tainted names per iteration | 209 ungated (34 in `src/`) | 0 strong, 2 weak / 10 | Ship info-level, `src/` only by default (JL023) |
| A: host loop calling a local jitted name / jnp op and appending | 20 local-jit (16 timing loops) | ~1/4 | Info only, local-file resolution + carried-state guard (JL020) |
| B2: `jax.jit(f)(x)` immediate call | 18 | 0/10 | Drop |
| B3: nested `@jit` def called in-function | 55 | 0/10 | Drop |
| D: shape-varying slice into jitted call | 0 | n/a | Drop |

Repo-wide bare-name jitted resolution is unusable (10/10 false positives from name
collisions), so resolution is local-file only. The denxity incident loop was not found
in the current tree [U: whether already fixed].

**Work items**

1. J0: shared `sensors/_jax_names.py` (import-alias resolver plus jitted-name table,
   covering bare `jit`, `eqx.filter_jit`, `partial(jax.jit)`); migrate the three copies.
   ~0.5 d.
2. J1: inline `# jaxlint: ignore[JLxxx]` suppression. ~0.5 d.
3. J2: parametrized fixture → expected-IDs harness covering every rule;
   `clean_minimal` stays at zero. ~0.5 d.
4. J3: cache key from rule-source hash, not `__version__`. S.
5. J4–J7: rules E, B1, C, A with positive + negative fixtures, `docs/rules.md` (both
   tables), SARIF help. ~0.5 d each. Each rule carries a `followup` field naming
   `xtrax profile` (see workstream 4).

## Workstream 2: `xtrax profile` verb

**Spike A: device vs host split on CPU** [V, jax 0.10.2]:
- Per-dispatch `ExecuteHelperOnSingleDevice` events and per-thunk events with
  `args.hlo_op` are present; thread names separate `python3` from the XLA client threads.
- A **2-way split** is derivable: device = union of `hlo_op` intervals; host+Python =
  window minus that union. A 3-way Python vs C++-dispatch split is not reliable.
- Do **not** use the existing `parse_hlo_op_times` for this: it sums durations, which
  overcounts on multithreaded CPU (17.2 ms summed in a 9.0 ms window). A new
  interval-union helper is needed.
- `python_tracer_level=0` cuts trace events ~5× with identical counts; tracer overhead ~11%.

**Spike B: negative control** (B=64, `4× tanh(h@W)` on 128×128) [V]:

| Variant | n_exec | Steady compiles | Untraced wall | exec / element |
|---|---|---|---|---|
| Host loop `x[b]` | 193 | 0 | 21.0 ms | 3.0–3.1 (B = 8…256) |
| `jax.vmap` | 1 | 0 | 2.75 ms | 0.004–0.125 |
| Chunked, C=16 | 9 | 0 | n/a | 0.141 |
| 3-stage jitted pipeline | 3 | 0 | n/a | 0.047 |

Steady-state recompiles are 0 in every variant, so **dispatch count is the
discriminator; the compile check alone misses a host loop.**

**Threshold**:
- Primary gate: `n_executions / cardinality(scale axis) < 0.5`. It is scale-free, and the
  gap is wide (≥3.0 vs ≤0.14).
- Optional per-entrypoint `max_executions` ceiling, declared the way `max_compilations` is
  in `audit/performance_targets.toml`.
- `max_compilations = 0` for steady units.
- No global absolute count.
- [U] The 0.5 is to be calibrated on real entrypoints (aminx sampler, `scripts/prof_*`)
  before it is fixed.

**Contract findings** [V]:
- `profiling/` is a leaf (AST-enforced, `tests/profiling/test_claim_contract.py:95-126`)
  and shipped code may not import `devtools`. So: measurement core in `profiling/`, verb
  in `cli/`, and the internal gate refactored onto the core.
- Inputs:
  - `--fn module:symbol --shapes ...` exists (`cli/shapes.py`) and `infer_bundle` gives
    axis cardinalities.
  - `plan_from_fn` raises on undecorated functions.
  - No config object (`TrainConfig`, `RunSpec`) carries full input shapes, so an
    `--inputs module:factory` hook taking `scale` is needed.
  - Zero-filled `--shapes` inputs may give degenerate `while_loop` trip counts [U].
- **Provenance bug**: `ProbeRecord._REPO_ROOT` is `parents[3]` of `record.py`, so a
  consumer's record is stamped with *xtrax's* HEAD; `emit_probe_record` takes no
  `git_sha`. This blocks the workstream-3 sha gate.
- Hardcoded output roots: `devtools/gates/performance.py:42` and
  `profiling/report.py` `_DEFAULT_DISCOVERY_ROOT` resolve into site-packages under a
  wheel install.
- Fit pre-check and extrapolation:
  - `jit(f).lower(ShapeDtypeStruct...).compile().memory_analysis()` works on CPU without
    allocating. `temp_size_in_bytes` read 0 on a fused toy [U for real programs].
  - If production does not fit, measure at scale S with
    `production / S <= SCALE_EXTRAPOLATION_LIMIT` (10), and label the wall estimate
    "extrapolated, CPU-relative".
  - Dispatch ratio and compile count are scale-free, so they stay valid at the smaller scale.

**Verb behaviour**:
- Runs a warm-up unit, then N steady units traced at production shape.
- Reports:
  - recompiles
  - exec/element
  - device/host split
  - compile seconds
  - `extent_scaling_report` at L and 2L
  - estimated total wall (per-unit × n_units)
- Emits a stage-1 ProbeRecord and a ranked markdown table.
  - `n_atoms` = scale-axis extent.
  - Metrics: `n_executions`, `n_compilations`, `n_jit_traces`, `total_step_seconds`,
    `compile_seconds`, `device_seconds`, `host_seconds`, `exec_per_element`.
- Exit codes: 0 pass, 1 flags raised, 2 could not measure. Exit 2 never reads as a pass.

**Work items**

1. X1: `profiling/unit_probe.py`: warm-up, traced steady units, interval-union split,
   `python_tracer_level=0`. M (~1 d).
2. X2: consumer git-sha provenance fix (resolve from cwd; `git_sha` parameter on
   `emit_probe_record`). S. **Prerequisite for B2.**
3. X3: output-dir resolver chain with source reporting: arg > `XTRAX_PROFILE_DIR` >
   `[tool.xtrax.profile]` > `~/.config/xtrax/config.toml` > print-only. Honour
   `RunSpec.output_root`. M.
4. X4: `cli/profile_verb.py` + `REGISTRY` entry; `--fn/--shapes/--inputs/--extent-axis`. M.
5. X5: dispatch-ratio rule + calibration on real entrypoints. S.
6. X6: fit pre-check + extrapolation path via `assert_claim_supported(END_TO_END, ...)`. M.
7. X7: shipped negative controls: host loop must FAIL, vmapped twin must PASS, plus a
   B=8 sensitivity check. S.
8. X8: refactor `devtools/gates/_dispatch_probe.py` onto X1, with gate behaviour unchanged. S.
9. X9: `xtrax profile --check <record> --sha HEAD` for the launch gate (reuses
   `claims._reject_unverifiable_git_sha`, `claims.py:68`). S.
10. X10: skills: `xtrax-probing` Quick Start → `xtrax profile`; remediation playbook in
    `xtrax-optimizing`; `using-xtrax/references/cli.md`. S.

## Workstream 3: launch gate and discipline

**Current state** [V, bathos bb53be1d; origin/main 986b7a95 is newer, so work from it]:
- `gate_check` (`prereg.py:474`) has one caller, `runner.py:917` (`bth run`). `bth submit`
  never calls it.
- Submit's reproduction prerequisite (`cli_cyclopts.py:763-797`) handles severity by stage:
  - It blocks an unmet prerequisite at validation/production and warns at
    exploration/calibration.
  - But it **skips when no sidecar is found, and warns and continues on exceptions**
    (795-796).
- Sidecars have no wall-time/budget field (`sidecar.py:359-368`), and unknown `[experiment]`
  keys only warn. So the "estimate in the sidecar" rule is prose-only today.
- bathos has no plugin/custom-rule mechanism; this is a bathos code change.
- A Bash PreToolUse hook is blind to MCP submits (`myxcel submit_job`, bathos `run`).
  Also, `~/.claude/hooks/jax-mem-guard.sh:71` passes `sbatch`/`myxcel`/`bth submit` through.

**Design**:
- bathos adds a domain-neutral `[requires]` table.
  - Fields: `artifact` glob, `json_key`, `must_equal = "HEAD"`,
    `reject_suffixes = ["-dirty", "-unverified"]`, `applies_if`.
  - Checked twice: (a) as a `bth submit` preflight before queueing or pushing, and
    (b) inside `gate_check`, so the compute-node `bth run` re-checks.
  - Stage severity copies the existing pattern: block at validation/production, warn at
    exploration.
  - It **fails closed**, unlike the existing submit gates.
- bathos adds a `[budget]` table (`est_wall_s`, `per_unit_s`, `n_units`, `probe_record`).
  The gate checks `est_wall_s` against the cited record within a tolerance.
- JAX workload detection:
  - An explicit `[requires] workload = "jax"` blocks; `bth new-experiment` scaffolds it.
  - An import-grep heuristic only warns.
  - Waiver: `[requires.waiver] reason = "..."`, logged in the run record. No global
    env-var bypass.
- xtrax supplies the record, `xtrax profile --check`, and the remediation hint string that
  bathos echoes verbatim.
- Optional Claude Code PreToolUse shim that calls `bth preflight` (the same code path).
  Its matcher covers `Bash` and `mcp__.*__(submit_job|run)`. It is a backstop, not the gate.

**Canonical sources to edit** (never the rendered copies):

| Item | Source |
|---|---|
| L1/L2/L3 rule | myxcel `agent_assets/snippets/cluster-rules.md` (L2 at line 35) |
| `using-xtrax`, `xtrax-probing`, `xtrax-optimizing` | xtrax `agent_assets/skills/` |
| `using-jax`, `using-jaxlint`, `jax-performance` (1-line stub) | jaxlint `agent_assets/skills/` |
| `jaxlint-jaxlint-always-on.md` | jaxlint `agent_assets/snippets/jaxlint-rules.md` |

Plugin hooks must be registered in **both** `agent_assets/hooks/manifest.toml` and
`.praxia/manifest.toml [[plugin.hook_specs]]` (praxia `CLAUDE.md:215-219`;
`praxia-config/src/manifest.rs:39`), or they export inert. Bump `[plugin].version`, or
`claude plugin update` is a no-op.

**Work items**
- B1: `[requires]`/`[budget]` parsing + validation. M.
- B2: submit preflight + `gate_check` integration. M-L, 3–4 d.
  - Fail-closed.
  - New `GateErrorCode`.
  - Waiver logging.
  - Positive and negative controls.
- B3: `new-experiment` scaffold + `using-bathos` doc. S.
- M1: myxcel L2 text ("smoke + `xtrax profile` at production shapes, record cited in
  `[budget]`") + version bump. S.
- P1: optional PreToolUse shim. S-M.

## Workstream 4: jaxlint as a cisternal agentic plugin

**Blockers** [V]:
1. jaxlint local `main` has unpushed work, plus a 133 MB file GitHub will reject.
   - `main` is 20 commits ahead of `origin/main` d10ee00, none pushed.
   - Commit 9b798c7 adds `.fastembed_cache/.../828e14...` at 133,093,490 bytes.
   - The working tree also has uncommitted sensor, skill and fixture edits.
   - Fix: rebuild a clean branch on `origin/main`, cherry-picking the plugin/export work
     without the cache, and gitignore `.fastembed_cache`. **Needs the user's call on the
     uncommitted edits.**
2. The manifest's `[plugin.mcp]` runs `jaxlint serve`, which does not exist; the entry
   point is `jaxlint-mcp`. The manifest also says 0.4.0 while `pyproject.toml` says 0.1.0a1.
3. Local cisternal is 60 commits behind `origin/main`; design against `origin/main`.

**Findings** [V]:
- cisternal's Claude hook export uses the hook `script` as a literal command
  (`export/hooks.py:29-58`). It bundles no file and has no timeout field, so the hook
  must be a console script on `PATH`.
- Rules/snippets come from praxia; cisternal only validates them (`manifest_extensions.py`).
- Reference implementation: maraxiom (maraxiom `.praxia/manifest.toml:143-153`).
  - Its `mrx-hitl-guard` hook is a stdlib-only console script.
  - It has an import-set test.
  - It mounts `plugin_app` by forwarding.
- contemplex and gemini-notebook-mcp hooks use paths that do not resolve elsewhere; do
  not copy them.
- `runner.py:13-16` imports `rich` eagerly.

**Design**:
- New console script `jaxlint-hook`, stdlib-only at import time, on PostToolUse
  `Write|Edit|MultiEdit`.
- Scope: a regex for a JAX import on the edited file → `run_checks` on that file only,
  using the disk cache.
- Output: new findings since the last check go back via `additionalContext`. Optional
  block on error-severity rules, behind a config flag.
- Failure policy: internal errors exit 0 with a one-line note. **Never exit 2 on a crash**;
  exit 2 is reserved for an intentional block.
- Budget: spike measured import 0.14 s, check 3–49 ms, CLI 0.22–0.37 s per edit.
- Telemetry: cisternal spans only when `CISTERNAL_TELEMETRY` is set.
- Packaging: plugin dependencies in an optional extra, so `jaxlint.core` stays import-light.
- Staleness: a content-hash check of the installed rule source, not the version string.
- Interface with workstream 1: the hook renders each rule's `followup` (e.g. "run
  `xtrax profile --fn ...`"), so new rules flow through with no hook change.
- Asset list:
  - skills: `using-jaxlint`, `using-jax`, `jax-performance`
  - the always-on snippet
  - the `jax-optimizer` agent
  - the MCP server (`jaxlint-mcp`)
  - the edit hook, registered in both praxia manifests

**Work items**
- C0: rebuild jaxlint main on origin/main without the 133 MB blob. S, needs the user.
- C1: fix the manifest MCP command and pick one version source of truth. S.
- C2: lazy `rich` import. S.
- C3: `jaxlint-hook` console script + hook_specs in both manifests + import-set test. M.
- C4: cisternal telemetry spans behind the env var. S.
- C5: export/validate via cisternal origin/main + `praxia plugin sync` dogfood, confirming
  the hook fires on a real edit. S-M.

## Sequencing

1. **Unblock**: C0 (user decision), X2 (provenance bug), bathos work rebased on
   origin/main.
2. **Parallel**: J0–J3 → J4 (rule E) first; X1 → X4 → X7 → X5; C1–C3.
3. **Integrate**: X9 + B1/B2 (the gate needs the sha fix and the `--check` helper); C3
   renders rule `followup`s.
4. **Discipline text last**, once the commands exist: M1, X10, jaxlint skills, B3.

Rough total: ≈3 d jaxlint rules, 5–6 d xtrax verb, 4–5 d bathos gate, 2–3 d jaxlint plugin.

## Decisions (user, 2026-10-16)

- **Severity policy:** the gate blocks at validation/production and **warns** at
  exploration/calibration.
- **Grandfathering:** sidecars of campaigns already running are **warn-only until a cutoff
  date**. The date is set when B2 ships; assumed ≈2 weeks after release, not yet fixed.
- **`.fastembed_cache`** is to be gitignored and excised from the jaxlint history rebuild
  (C0). A subagent is triaging the 20 unpushed commits.

## Decisions (user, 2026-10-17)

- **Cutoff:** a config value with no default, `.bth.toml [gate] requires_enforce_after`.
  Until it is set the gate only warns. The outermost `[gate]` within the git root wins.
- **Enforced stages:** an unmet requirement blocks at validation, production **and
  ablation** (bathos's confirmatory class). A sidecar with no stage counts as exploration
  and warns.
- **No discoverable sidecar:** `bth submit` scans every `.py` token and accepts
  `--sidecar PATH`. Under active enforcement, a submit with no sidecar found blocks.
- **Paths:** `[requires] artifact` and `[budget] probe_record` resolve relative to the
  project root and must stay inside it. Records go under `outputs/profiling/`.
- **P1:** built after B2, on the same `bth preflight` code path.
- **jaxlint version:** `pyproject.toml` is canonical; the manifest and bundle match it, and a
  test pins this (C1).
- **JL020:** `host-loop-jnp-ops` stays off by default.

## Open questions for the user

None. C0 is resolved: jaxlint `main` (4e40167) carries no `.fastembed_cache` blob, and the
directory is gitignored.
