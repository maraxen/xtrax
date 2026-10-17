---
name: xtrax-probing
description: "Use when a JAX timing, dispatch count, or compile count has to stay citable, when a benchmark result is being quoted past the scale that was measured, or when a performance gate needs a dispatch-count ceiling. Covers the xtrax profile smoke check, stored probe records, claim-validity checks, and bottleneck reports."
when_to_use: 'Triggers: a timing or dispatch count that a later claim will cite, benchmark result quoted beyond the scale that was measured, performance gate needs a dispatch-count or compile-count ceiling, bottleneck report from stored measurements, persist a benchmark so the number stays attachable to a claim.'
xtrax_version: 0.4.1b1
---
# xtrax-probing

## Purpose

Package every JAX measurement made in xtrax into a `ProbeRecord`: a frozen,
stage- and scale-stamped artifact whose fields state what the measurement may
be cited for. The claim-validity contract (`xtrax.profiling.claims`) turns
sets of records into verdicts -- fail-closed: an unsupported claim raises,
never silently passes.

Verify-paths (house convention): every rule below cites the module that owns
it. When this skill and the code disagree, the code wins -- then update this
skill.

## Non-Negotiables

1. Never hand-write record JSON. Construct via `ProbeRecord(...)` or
   `emit_probe_record(...)`; validation runs at construction and invalid
   artifacts never reach disk.
2. Never widen a guard to make a claim pass. Narrow the claim instead.
3. No first-party imports inside `xtrax.profiling` (leaf package, AST-
   enforced); jax is imported lazily only inside provenance factories.
4. New modules under `src/xtrax/profiling/` must not use future-annotations.

## Quick Start: `xtrax profile`

Run `xtrax profile` first, at the smoke stage, on ONE unit at its production
shape. It reports whether the unit recompiles at steady state and how many XLA
dispatches it makes per element of the scale axis. Verb: `src/xtrax/cli/profile_verb.py`.

Zero-filled inputs from a shape string (`name=(d0,...)dtype`):

```bash
xtrax profile --fn tests.cli._profile_fixtures:host_loop \
    --shapes "x=(64,128)f32" --extent-axis x --out none
```

Inputs from a factory `module:factory`, called as `factory(scale)`, which
returns `(args, kwargs)` or just `args`. The `--extent-axis` name must be a
kwarg of the result:

```bash
xtrax profile --fn tests.cli._profile_fixtures:vmapped_twin \
    --inputs tests.cli._profile_fixtures:make_inputs --scale 64 \
    --extent-axis x --out none
```

`--fn` and `--inputs` are import paths. The module must be importable from
the current directory. For a module in the repo, set `PYTHONPATH=.`. The
`tests.cli._profile_fixtures` examples above run from the repo root.

Reading the report:

- `steady compiles (window)`: backend compiles in the traced steady window.
  Must be 0. A non-zero value means the unit rebuilds its executable per call.
- `exec/element`: XLA executions per unit divided by the extent. Flag at
  `>= 0.5`, the value of `xtrax.profiling.rules.EXEC_PER_ELEMENT_FLAG`. A host
  loop measures about 3 per element; a batched unit measures well below 0.5.
  Steady compiles stay 0 for a host loop, so dispatch count is the check that
  catches it.
- `device s / unit` and `host s / unit`: a two-way split, device versus
  host plus Python. Host-dominated means the time is between dispatches.
  Timings are CPU-relative on CPU.
- `### Fit`: `fits: yes / no / unknown`. Device memory is read from the
  backend. CPU reports `unknown`, which is not a pass.
- `### Extrapolation`: present only with `--target-extent`. Prints the wall
  estimate at the production extent, marked extrapolated, or `wall estimate
  withheld` with the reason (see below).
- `### Flags`: one line per flag. `none` means the unit passes.

Exit codes: `0` pass, `1` flagged, `2` could not measure. Exit 2 is never a
pass. The report's `verdict:` line says which.

Record: each measurement writes a stage-1 ProbeRecord unless the output dir is
off. The report's `profile dir:` line names the directory and the layer that
chose it. The layers, first match wins (`src/xtrax/cli/profile_dir.py`):

1. `--out PATH`
2. `XTRAX_PROFILE_DIR`
3. `profile_dir` under `[tool.xtrax]` in the nearest `pyproject.toml`
4. `profile_dir` at the top level of `${XDG_CONFIG_HOME:-~/.config}/xtrax/config.toml`
5. `RunSpec.output_root`, when a run spec is given
6. off (print only)

`--out none` or an empty value disables the record at any layer. A relative
path is rejected. For a record in a scratch directory:

```bash
XTRAX_PROFILE_DIR=$TMPDIR/profiles xtrax profile --fn tests.cli._profile_fixtures:vmapped_twin \
    --shapes "x=(64,128)f32" --extent-axis x
```

Target extent: when the production extent does not fit in memory, measure
at a smaller extent and extrapolate with `--target-extent N`, where N is the
production extent. The measurement runs at the extent the inputs give. The
wall estimate is linear, and
it is printed only if `assert_claim_supported(END_TO_END)` accepts it: the
ratio must be within `SCALE_EXTRAPOLATION_LIMIT` (10x) and the git sha must be
clean. Otherwise the line reads `wall estimate withheld`. Withholding does not
change the exit code. The dispatch ratio and compile count are scale-free, so
they stay valid at the smaller extent. Without `--target-extent`, a unit that
the fit check marks `fits: no` exits 2.

Pre-launch gate: check a stored record against the revision that will run:

```bash
xtrax profile --check PATH/TO/profile_<fn>_<stamp>.json --sha HEAD
```

Exit 0 when the record's sha matches `--sha`, its provenance is clean, and its
flags pass. Exit 1 when it is stale or flagged. Exit 2 when it is unreadable,
lacks the metrics, has an unverifiable sha, or the revision does not resolve.
A record written from a dirty tree carries a `-dirty` sha and cannot back a
check, so commit before measuring anything that will gate a launch.
`--check` and `--fn` are mutually exclusive.

Negative control: the probe must still tell a host loop from a batched unit
on the platform in use. JAX profiler event names change across versions. Run
this on a new JAX version or platform:

```bash
uv run python -c "from xtrax.profiling.controls import run_negative_controls; r = run_negative_controls(); print(r); raise SystemExit(0 if r.held else 1)"
```

`run_negative_controls()` in `xtrax.profiling.controls` measures a host loop
(must be flagged) and a vmapped twin (must pass) at batch sizes 8 and 64. It
returns a `ControlsReport`, where `held` is True when all rules hold. The
module imports JAX, so it is not re-exported from `xtrax.profiling`.

## Stage / Claim Decision Table

| stage | platform | STRUCTURAL | DISPATCH_COUNT | TERM_RANKING | END_TO_END |
|---|---|---|---|---|---|
| 0 cost-analysis | any | yes | no | no | no |
| 1 micro-exec | cpu ok | yes | yes | no | no |
| 2+ device-measured | gpu required (+device_kind) | yes | yes | set-based | set-based |

TERM_RANKING additionally requires >=2 attributed scopes per source carrying
`total_step_seconds`, unanimity on x64_enabled/xla_flags/device_kind/
platform/git_sha, verifiable (clean) shas. END_TO_END requires a positive
target_n_atoms within SCALE_EXTRAPOLATION_LIMIT of min(source n_atoms).
Single-record checks go through `permitted_claims`; set-backed claims through
`assert_claim_supported` -- both raise `ClaimValidityError` on violation.

## Integration Surfaces (all opt-in, zero behavior change by default)

- **Performance gate**: per-probe `max_compilations` / `max_jit_traces`
  ceilings and `emit_probe_record = true`; the
  `performance.dispatch_violation_count` baseline ratchet activates only when
  some probe configures ceilings. A crashing probe becomes a counted major
  finding, never a gate crash.
- **Controller**: `run_one_candidate_pass(..., probe_record_dir=...)` writes
  one timestamped Stage-0 provenance record per completed pass; emission
  failures are contained and reported without discarding the pass.
- **pytest-benchmark**: see the bench protocol below; strict stats schema,
  collision refusal, loud skips.

## Lower-Level Probe API

The drivers and bench hooks below write records directly. Use them for
stage-0/1 studies and for benchmarks. Use `xtrax profile` for a single unit.

Stage-0 cost probe (no execution) and stage-1 micro-execution probe over the
tiling strategies; records land in `outputs/profiling/stage<N>/`, raw traces
stay gitignored:

```bash
XTRAX_GIT_SHA=$(git rev-parse HEAD) \
    uv run python scripts/prof_stage0_tiling_cost.py
XTRAX_GIT_SHA=$(git rev-parse HEAD) \
    uv run python scripts/prof_stage1_tiling_micro.py
```

Benchmark wall-clock persistence (opt-in; benches DECLARE their own
`xtrax_stage`/`xtrax_n_atoms` via `benchmark.extra_info`, undeclared benches
are skipped-with-reason):

```bash
XTRAX_GIT_SHA=$(git rev-parse HEAD) \
XTRAX_BENCH_RECORD_DIR=outputs/profiling/stage1 \
    uv run pytest benchmarks --benchmark-only
```

Contract + fixture audit:

```bash
just audit-profiling-contract
uv run python scripts/audit_profiling_contract.py
```

Claim-gated bottleneck report over `outputs/profiling/stage*/*.json`:
`scripts/prof_report.py` style entry via `xtrax.profiling.report`.

## Additional Resources

Load as needed -- do not read both up front:

- **`references/contract.md`** -- full ProbeRecord field/guard reference,
  claim rules with verify-paths, contract-versioning bump rule.
- **`references/workflows.md`** -- step-by-step workflows (drivers, bench
  declarations, gate/controller wiring), plus the ClaimValidityError
  catalogue: message -> cause -> fix.

Acting on a measurement -- deciding what to optimize and what evidence each
class of change owes -- is the `xtrax-optimizing` skill's job (three-tier
taxonomy + preregister-and-probe protocol; its drivers emit the records this
skill governs).
