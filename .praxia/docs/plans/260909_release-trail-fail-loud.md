---
category: plans
title: Fail loud where the a9 release trail found silent green
description: 'Sprint spec: fix zarr_content_digest determinism (#5013), consolidate the divergent dev dependency tables and the runtime-deps list (#4969, #4967), and make the test-rigor gate report its own failure (#5021) as the prerequisite for #5002'
status: draft
task_id: 260909_xtrax-sprint-spec
date: '260909'
sprint: '260909'
backlog_ids: '5013, 4969, 4967, 5021'
---

# Sprint: fail loud where the a9 release trail found silent green

## The finding

The 0.4.0a9 release trail turned over three places where this repo reports green
while the thing under the report is broken. Each has the same shape: a mechanism
that cannot fail, wrapped around a fact that is false.

| Site | Reports | Actually |
|---|---|---|
| `zarr_content_digest` (`src/xtrax/run/zarr_integrity.py:102`) | "deterministic content digest, unaffected by which process wrote the store" (:105-107) | two identical writes digest differently |
| `[dependency-groups].dev` vs `[project.optional-dependencies].dev` (`pyproject.toml:20-28` / `:33-49`) | one `dev` install | two sets, 7 vs 15 packages; `beartype` in only one |
| `audit-test-rigor-gate` (`Justfile:109-112`) | a `JSONDecodeError` traceback | a discarded pytest failure it captured and threw away, from a gate that would have reported PASS over a red suite anyway |

The dependency one has already been paid for once. `Justfile:231-234` records it:
the #131 beartype-stripping bug traced to these two tables, and the comment ends
"Consolidating them is the real fix -- backlog #4969." It was not consolidated.

Backlog rows for #5013, #4969, #4967 and #5021 were read directly by the
orchestrator this session; the option (a)/(b) framing in Phase A, the cross-host
argument, and the difficulty labels in `## Rubric` are verbatim from those rows.

### Measured, not inferred

- **`zarr_content_digest` folds every node's attrs into the hash**
  (`zarr_integrity.py:88-91`), with no skip mechanism anywhere in the module.
  `ZarrStagingSink.__init__` stamps `created_at = datetime.now(UTC).isoformat()`
  onto the root group (`zarr_sink.py:177-180`, written at `:223`/`:225-227`).
  Wall-clock. Two writes of byte-identical content one second apart therefore
  produce different digests, from a function whose docstring promises the
  opposite.
- **The root group is not the only site.** `drain()` stamps a
  `run_id`/`git_sha` pointer onto **every drained key's group**
  (`zarr_sink.py:347-352`). `SinkSpec.run_id` is required (`sink.py:26`) and
  `derive_sink_spec` falls through to `new_run_id()` (`sink.py:100`), which is
  `uuid4`-derived (`ident.py:13-19`). Two runs get different `run_id`s, so every
  non-root group differs too. **This breaks the root-only fix shape** — see A1.
- **Nothing tests the sink-written case.** `tests/run/test_zarr_integrity.py:22-30`
  builds stores by hand with `zarr.open_group`, never through `ZarrStagingSink`,
  so the three determinism tests (`:45`, `:60`, `:82`) all digest a store that
  carries no provenance attrs at all. `tests/run/test_repro_floor.py:27` does the
  same — `SinkSpec` and `ZarrStagingSink` appear nowhere in that file. The defect
  is invisible to both suites that exist to catch it.
- **`zarr_content_digest` and `update_zarr_node_digest` are both exported**
  (`run/__init__.py:26`, `:34`, `:51-52`) **and neither is a tier-1 export** —
  neither appears in `distribution/public_api.toml`. A keyword-only parameter
  breaks no pinned contract on either.
- **The two `dev` tables differ by nine packages.** `[dependency-groups].dev` has
  7 (`pyproject.toml:20-28`); `[project.optional-dependencies].dev` has 15
  (`:33-49`), including `beartype>=0.18.0` (`:42`), `jaxtyping`, `libcst`,
  `jaxlint`, `chex`, `interrogate`, `pyyaml`, `pytest`, `pytest-asyncio`.
  `eda` is *also* declared twice (`:30` as a group, `:50` as an extra) — same
  divergence class, currently identical content, one edit from diverging.
- **The Justfile depends on the default-synced group.** 151 `uv run` lines; only
  five carry `--extra` (`:5`, `:42`, `:43`, `:44`, `:306`). The other ~146 get
  `ruff`, `ty`, `pytest-cov`, `complexipy`, `import-linter` and `tyro` from
  `[dependency-groups].dev` being synced by default. **Deleting that group breaks
  `audit-deterministic` and `audit-project-hygiene` — Phase B's own gates.**
- **The alias shape resolves and fixes the #131 mechanism** (orchestrator dry-run
  spikes, uv 0.11.21, worktree restored afterwards). With
  `[dependency-groups] dev = ["xtrax[dev]"]` / `eda = ["xtrax[eda]"]`:
  `uv lock --check` resolves **218 packages, no error**; `uv sync --dry-run
  --group docs` removes **none** of `beartype`, `chex`, `interrogate`, `jaxlint`,
  `libcst`. The same command against today's tables removes **all five**. That is
  the #131 beartype-stripping bug reproduced by dry-run, and the alias closing it.
- **`uv run` does not prune, and never did.** `uv run --help` shows `--exact` as
  **opt-in** (inexact is the default: it adds what is missing and removes
  nothing). `uv sync --help` shows `--inexact` as opt-in (exact is the default:
  it prunes). So `uv sync` is the environment-replacing call and a bare `uv run`
  is not. `scripts/audit_docs_plumbing.py:164-167` and `Justfile:216-220` state
  this correctly. `scripts/audit_wave1_load_bearing.py:143-148` states the
  opposite and is wrong — see B8.
- **`grain>=0.2.0` is a hard runtime dependency (`pyproject.toml:7`) that is
  imported nowhere** — zero matches across `src/`, `tests/`, `scripts/`, `port/`
  (recon `260909_recon_4969_4967_dep_hygiene`). Its intended consumer,
  `src/xtrax/data/pipeline.py`, is a stub whose docstring says "real grain
  sharding deferred to Phase 5/6". `pytest-asyncio>=0.23` is likewise in the
  runtime `dependencies` array, and also in the dev extra (`:36`).
- **`test_rigor.py:90-95` creates the report file before pytest runs.**
  `tempfile.NamedTemporaryFile(delete=False)` produces a real, zero-byte file. So
  `cov_path.is_file()` at `:121` is **always True**, the missing-report branch at
  `:122-126` is unreachable dead code, and `:127` parses an empty file — the
  `json.JSONDecodeError` raised at `:49`. The pytest output is captured
  (`:110-118`) and assembled into `combined` (`:119`), which is referenced
  **only** inside the unreachable branch. Every byte of diagnostic is discarded.
- **The gate passes on coverage alone.** `test_rigor.py:180` is
  `passed = passes_line and passes_branch`. Neither `result.returncode` nor
  `stats.tests_failed` — both computed, both carried on `CoverageStats`
  (`:31-32`) — participates in the verdict. A suite that fails 300 tests but
  writes a valid coverage report exits 0.
- **No test in `tests/audit/test_test_rigor_gate.py` calls `run_pytest_coverage`.**
  All three gate tests patch it out (`:106-109`, `:154-157`, `:195-198`). The
  function has no coverage at all.
- **The coverage report was never written because coverage collected nothing.**
  A sibling artifact from the same orphan run, `audit-drift-check.log:40`, carries
  `CovReportWarning: Failed to generate report: No data to report`. That is the
  only mechanism consistent with the observed **zero-byte** file — pytest-cov
  emitted the warning and wrote no JSON. See C5.
- **`audit-bootstrap`'s failure is not environmental.** The CI artifact names
  `failed dimensions: api_ergonomics, structure_complexity` — a cascade of two
  real baseline regressions (recon `260909_recon_orphan_ci_logs`, confidence
  0.95; this corrects the earlier inference in `260909_recon_5002_orphan_gating`).
- **There is no independent triage to cross-check this slice against.** The
  zero-token `pm_triage` flow ran for ~20 minutes, exited 0, and appended **zero**
  records — its documented hollow-run failure mode. `.praxia/pm_triage_results.jsonl`
  does not exist. The slice rests on the four recon records and the rubric.

### The defects this already exposes

1. **`repro_floor` is broken by the digest defect, in xtrax's own tree.**
   `src/xtrax/run/repro_floor.py:125-135` re-runs a caller-supplied `compute(seed)`
   `rerun_count` times, digests each output, and reports `passed=False` on any
   mismatch. Every caller whose `compute` builds a `ZarrStagingSink` fails that
   check unconditionally — the module's entire premise, defeated by a timestamp.
2. **The consumer assertion is already written.** `demistify`'s
   `scripts/validation/zarr_io_parity.py:267-281` writes the same
   `PipelineResult` twice and asserts the digests match. It cannot pass.
3. **`audit-test-rigor-gate` has been hiding a real pytest failure**, and would
   have hidden it even without the crash — `test_rigor.py:180` never looks at the
   exit code. Whatever #5002 needs to gate on, nobody can see it until the gate
   both prints what it caught and fails on it.

## Rubric

`.praxia/sprint_rubric.toml` sets `ITEM_CAP = 3` and `DIFFICULTY_BUDGET = 6`
(`quick = 1`, `standard = 3`, `extended = 5`).

**This sprint touches four backlog rows across three work-streams**: #5013
(standard, 3), #4969 (standard, 3), #4967 (quick, 1), #5021 (quick, 1) — **8
points, 4 items.** Both the item cap and the difficulty budget are exceeded.
Calling #4967 "absorbed" would be relabelling four items as three, which is
exactly the move this spec exists to refuse.

The argument for taking it anyway: the rubric governs the **autonomous** loop, and
this is a human-run sprint — the previous one (`260903_controller-gate-gap.md`)
also exceeded it. #4967 edits the same `pyproject.toml` table #4969 already
rewrites; running it separately means editing that table twice and reviewing the
dependency surface twice. #5021 is the **prerequisite** for #5002 — without it
the next sprint's input does not exist.

**Every sub-task below is tagged with its backlog row in its heading**, so the
fallback cut is executable by heading rather than by judgement.

**Rubric-strict fallback cut: #5013 + #4969** (3 + 3 = 6, exactly at budget,
2 items). That drops **B4, B5, and all of Phase C** — which pushes #5002 out by
two sprints instead of one. Choose this if the budget is meant to bind.

It is also the only cut that fires **no loop-constitution gate**: Phase C is a
gate-(b) event and needs a `T2-29` attestation before merge (see Phase C), while
#5013 and #4969 touch no evaluator. A sprint that wants to run without a human
sign-off step is this cut.

**Decision point:** Marielle chooses the cut when approving this document (the
PR that lands it). Phase A and B1-B3/B6-B8 may start under either answer; B4,
B5 and Phase C do not open a branch until the full scope is confirmed. No
answer means the fallback cut.

## Phase A — make the digest mean what its docstring says (#5013)

Branch `fix/5013-zarr-digest-provenance`.

The fix is xtrax-side, option (a) from the #5013 row. The justification is not
"change the contract": `zarr_integrity.py:105-107` already promises the digest is
"unaffected by ... which process/session wrote the store, only by the store's
logical content". `run_id` and `created_at` in the digest **violate** the
documented contract. Excluding them by default restores it. Caller-side option
(b) is rejected: it would require every consumer to re-implement the exclusion,
and would leave xtrax's own `repro_floor` broken.

**A1 (#5013) — decide the skip scope, and do not make it root-only.** The #5013
row frames this as a root-group skip. The evidence above says root-only is
insufficient: `zarr_sink.py:347-352` writes `run_id`/`git_sha` onto every drained
key's group, and `ident.py:13-19` guarantees those differ between runs. A
root-only skip leaves every store with at least one non-root key still
non-deterministic — including demistify's, which stages under keys, and including
anything `repro_floor` measures. **This is the one place this spec deviates from
the item as filed, and it deviates on measurement.**

The shape to implement:

- **Root group:** skip all five names in `_CORE_PROVENANCE_FIELDS`
  (`zarr_sink.py:34`): `git_sha`, `git_branch`, `git_dirty`, `run_id`,
  `created_at`. All five are provenance; #5013's own cross-host argument (a store
  rebuilt on another machine has a different `git_branch`, and a dirty tree a
  different `git_dirty`) applies to the git trio exactly as the timing argument
  applies to `created_at`.
- **Non-root groups:** skip `run_id` and `git_sha` — the pointer pair
  `drain()` writes at `zarr_sink.py:349-350`.
- **Arrays:** skip nothing. `drain()` creates arrays with no attrs
  (`zarr_sink.py:334-341`); caller-staged attrs land on the *group*
  (`zarr_sink.py:344`).

**State the cost plainly, in the docstring and in review.** The exclusion is by
attr **name**, so it is unconditional: a hand-built store whose root group
carries its own meaningful attr called `git_branch`, `created_at`, `run_id`,
`git_sha` or `git_dirty` loses it from the default digest, and the same applies
to `run_id`/`git_sha` on any non-root group. Five names at root, two below it,
excluded whether or not a sink wrote them. That is the price. It is unreachable
*through the sink*, because `_validate_stage_attrs` (`zarr_sink.py:231-234`)
refuses caller-staged attrs colliding with `_CORE_PROVENANCE_FIELDS`; it is fully
reachable for a store built by hand. `include_provenance=True` is the escape
hatch.

**A marker or namespace would remove that cost, and is rejected for this sprint.**
The sink could write its provenance under a single namespaced attr (e.g.
`_xtrax_provenance = {...}`), or record alongside it the list of keys it stamped,
and the digest could then skip exactly the sink-written keys and nothing else.
That is the structurally correct answer and it is four lines from where the
pointer is written. It is out of scope here because the flat root-attr layout —
`git_sha`, `git_branch`, `git_dirty`, `run_id`, `created_at` as **top-level**
attrs — is the documented 0.4.0a7 contract (`CHANGELOG.md:147-158`) and is what
consumers read today. Changing it is a consumer-visible migration, not a digest
fix. **File it as a follow-up backlog item in the Phase A PR**, cross-referenced
from the docstring.

**A2 (#5013) — single-source the skip set.** Import `_CORE_PROVENANCE_FIELDS`
from `src/xtrax/run/zarr_sink.py:34`; do not re-declare the five names in
`zarr_integrity.py`. Derive the non-root pointer subset from the same frozenset
(`{"run_id", "git_sha"}`) rather than writing a second literal. Check the
import-linter contracts in `pyproject.toml:142-217` before adding the import —
both modules are inside `xtrax.run`, so no declared contract forbids it, but say
so in review rather than assume it.

**A3 (#5013) — the keyword, on both exported functions.** Add
`include_provenance: bool = False`, **keyword-only**, to `zarr_content_digest`
(`zarr_integrity.py:102`) **and** to `update_zarr_node_digest`
(`zarr_integrity.py:75`), which is separately exported at `run/__init__.py:26`
and `:51`. Keyword-only on both, so no existing three-positional-argument caller
of `update_zarr_node_digest` breaks. `update_zarr_node_digest` already receives
`path`, so `path == "/"` is the root test and no further argument is needed.
Pick this name and use it in both signatures; do not leave two candidate names in
the diff.

**A4 (#5013) — the tests.** Under `tests/run/test_zarr_integrity.py`, which today
has no `ZarrStagingSink` fixture at all (`:22-30`):

- Two separate `ZarrStagingSink` instances, **different `run_id`s**, writing
  identical staged content to two different paths, yield **equal** digests.
  *This test must fail against current code.* Confirm that before writing the
  fix; a version of it that passes today is testing the wrong thing.
- The same pair with `include_provenance=True` yields **different** digests.
- A hand-built store whose **root** group carries an attr named `run_id` has it
  excluded by default; the identical attr on a **non-root array** is still
  hashed. This pins the array/group asymmetry from A1.
- A hand-built store whose non-root **group** carries `run_id` has it excluded —
  asserted explicitly, as documentation of the A1 cost, not as an accident.

**A5 (#5013) — prove `repro_floor` un-breaks.** Add a test under `tests/run/` that
drives `run_repro_floor` (`repro_floor.py:125-135`) with a `compute` that builds a
real `ZarrStagingSink` per rerun and returns its path. **Use a fresh
`output_dir` per rerun**: `zarr_sink.py:215-222` raises `ValueError` ("Use a
fresh output_dir per run_id") when a directory already holds a different
`run_id`, so a fixture that reuses one directory gets an exception, not the
`passed=False` the test is meant to observe. It must report `passed=False`
before A and `passed=True` after. `tests/run/test_repro_floor.py`
builds stores with `zarr.open_group` (`:27`) and never mentions `SinkSpec` or
`ZarrStagingSink`, so this surface is entirely untested.

**A6 (#5013) — CHANGELOG.** A `### Fixed` entry under `[Unreleased]`
(`CHANGELOG.md:8`), naming the default behaviour change and the
`include_provenance` escape hatch. Note in the entry that `zarr_content_digest`
values computed before this change do not match values computed after it for any
sink-written store — that is the point, but a stored done-marker digest will now
mismatch.

**A7 (#5013) — update the shipped skill reference.**
`agent_assets/skills/using-xtrax/references/run.md:204` states the pre-A contract
verbatim: "sha256 over the store's paths, attrs, and array data; unaffected by
filesystem metadata or which process wrote it." After A the second clause is only
true *because of* an exclusion the line does not mention. Rewrite it to name the
exclusion and the keyword; check `:206` and `:210`, which list
`update_zarr_node_digest` and will need the keyword mentioned too. This is a task
rather than a nicety because that tree ships inside the wheel to every consumer,
and the only gate over it — `scripts/audit_project_hygiene.py:207-216` — checks
the `xtrax_version` marker in `SKILL.md`, not the prose in `references/`. Stale
prose there ships silently, to the agents that read it as the contract.

**Gate for Phase A:**

```bash
uv run --extra dev --extra io pytest tests/run/test_zarr_integrity.py tests/run/test_repro_floor.py -q
uv run --extra dev --extra io pytest tests/distribution/test_public_api.py -q
just audit-public-api          # Justfile:207-209
just audit-project-hygiene     # covers the A7 skill surface
```

All green, with the A4 two-sink test and the A5 `repro_floor` test both
demonstrated **red** against `origin/main` first. Record both red runs in the PR
body.

**Cross-repo note, not a task in this sprint:** demistify's
`pyproject.toml:39` declares `xtrax>=0.4.0a5` — a floor, not a pin — and its open
PR #38 (branch `chore/xtrax-0.4.0a9`) already migrates the call site, passing
`run_id=run_id or new_run_id()` at `zarr_io.py:367`. So the remaining work there
is not a code change: once Phase A ships in a release, #38 merges, and demistify
relocks, `demistify/scripts/validation/zarr_io_parity.py:267-281` passes with no
further consumer change. Do not edit demistify here.

## Phase B — one source of truth for dependencies (#4969, #4967)

Branch `chore/4969-dependency-single-source`. Independent of A; can run in
parallel.

**B1 (#4969) — make the group a thin alias of the extra. Do not delete it.** The
extras are canonical, because that is what CI passes (`ci.yml:59`, `:77`, `:120`,
`:148`, `:168`; `audit-orphans.yml:47`, `:82`; `audit-judgment.yml:22`). But the
Justfile is the other consumer, and ~146 of its 151 `uv run` lines carry no
`--extra` at all and depend on `[dependency-groups].dev` being synced by default.
Deleting the group breaks `audit-deterministic` and `audit-project-hygiene`,
which are Phase B's own gates.

```toml
[dependency-groups]
dev = ["xtrax[dev]"]
docs = ["sphinx>=7", "furo", "sphinx-autodoc-typehints", "myst-parser"]
eda = ["xtrax[eda]"]
```

A self-referential extra in a dependency-group is the one shape that makes the
group *structurally* unable to diverge: it has no content of its own. Measured on
uv 0.11.21 (dry-run spikes, worktree restored):

- `uv lock --check` under the alias resolves **218 packages, no error**.
- `uv sync --dry-run --group docs` under the alias removes **none** of
  `beartype`, `chex`, `interrogate`, `jaxlint`, `libcst`.
- The same command against today's tables removes **all five** — the #131
  mechanism, reproduced.

**Two consequences to carry in the PR, not discover in it:**

- **`uv.lock` must be regenerated** (`uv lock`) and committed in the *same* PR.
  The dependency graph changes twice here — the alias in B1/B2 and the `grain`
  move in B4.
- **`scripts/git-hooks/pre-push:33` runs `uv sync --frozen --extra dev --extra
  eda --extra io`.** `--frozen` hard-fails on lock drift, so a PR that changes
  `pyproject.toml` without relocking fails the developer's *first push*, before
  any gate runs. That is the hook working; say so in the PR body so it is not
  misread as a Phase B regression.

**Verification for B1 specifically:** re-run both dry-runs above after the edit
and paste the two package lists into the PR body. The alias is only worth taking
if the second list is empty.

**B2 (#4969) — alias `eda` the same way** (`pyproject.toml:30` becomes
`eda = ["xtrax[eda]"]`, extra at `:50` unchanged). It duplicates the extra with
currently-identical content; leaving it is leaving the same trap armed.
`[dependency-groups].docs` (`:29`) stays as-is — it has no extra counterpart, so
there is nothing for it to diverge from.

**B3 (#4969) — the contract test.** New file under `tests/distribution/`. The
rule is **not** disjointness — B1 and B2 deliberately put `dev` and `eda` in both
tables. The general form:

> For every name present in **both** `[dependency-groups]` and
> `[project.optional-dependencies]`, the group's value must be exactly the
> single-element list `["xtrax[<name>]"]`. A name present in only one table is
> unconstrained.

That admits the alias shape, forbids the divergent shape, and covers any future
name by construction. Wire it into `audit-project-hygiene` (`Justfile:248-251`),
which is already in `audit-deterministic`'s prerequisite list (`Justfile:324`) and
already parses `pyproject.toml` (`scripts/audit_project_hygiene.py:218-231`). Add
the same check to `audit_project_hygiene` (`:150-233`) so the script and the test
both enforce it, matching the existing recipe shape.

**B4 (#4967) — the runtime dependency list.** In the same `pyproject.toml` edit:

- Move `grain>=0.2.0` out of runtime `dependencies` (`:7`) into a new
  `data = ["grain>=0.2.0"]` extra, modelled on `io = ["zarr>=3.0"]` (`:52`).
- Remove `pytest-asyncio>=0.23` from runtime `dependencies` (`:7`). It stays in
  the dev extra (`:36`), which is where a test-only plugin belongs.
- Consumer check is already done and needs no re-run: `aminx` depends on
  `xtrax[io]`, `plegadx` on `xtrax[cli]`, and `jaxbeans`/`phyllo`/`proxide`
  depend on `grain` directly rather than through xtrax. No consumer references
  `xtrax[data]` (recon `260909_recon_4969_4967_dep_hygiene`).
- `### Changed` entry under `[Unreleased]` (`CHANGELOG.md:8`) naming **both**
  removals as consumer-visible: anyone relying on `pip install xtrax` to pull
  `grain` or `pytest-asyncio` must now ask for them.

**B5 (#4967) — the declared-vs-imported gate**, the hardening the #4967 row's
"adjacent hardening" paragraph proposes. A contract under `tests/distribution/`,
wired into the same `audit-project-hygiene` recipe as B3.

**Derive the distribution→import mapping; do not hand-write it.** A table in
`distribution/project_hygiene.toml` listing `orbax-checkpoint → orbax` and the
rest would be a second copy of `pyproject.toml:7`'s names with nothing keeping
the two in sync — the exact divergence class #4969 exists to kill. Use
`importlib.metadata.packages_distributions()` against the synced environment: it
returns import-name → `[distribution names]` for everything actually installed,
so the mapping is a fact about the environment rather than a claim in a config
file.

Check **both** directions:

- Every name in `[project].dependencies` has at least one top-level import under
  `src/`. **Walk nested imports too** — `jaxlib`'s only appearance in the tree is
  `import jaxlib` inside a function body at `src/xtrax/profiling/record.py:105`.
  A gate that walks only `tree.body` fails on a dependency that is genuinely
  used. Walk every `ast.Import` / `ast.ImportFrom` node at any depth and match the
  top-level module name.
- Every third-party top-level import under `src/` resolves to a declared runtime
  dependency **or** a declared extra. Exclude `sys.stdlib_module_names` and
  `xtrax` itself. This is the direction that actually breaks consumers — an
  import with no declaration anywhere is a `ModuleNotFoundError` at their install,
  not ours.

Lazy and optional imports guarded behind an extra are **expected** to resolve to
an extra rather than to `dependencies`; that is the correct outcome, not a
failure (`zarr` in `zarr_integrity.py:84`/`:113` is the model).

**`packages_distributions()` only sees what is installed, so state the policy
for what is not.** `audit-project-hygiene` runs under `dev` + `io`; it never
syncs `export`, `eda` or `controller`. Direction 2 will therefore meet imports
whose distribution is absent from the map — `iree` at
`src/xtrax/export/compile.py:74`, `:101`, `:248` (distributions
`iree-base-compiler` / `iree-base-runtime`, `pyproject.toml:55-65`), and
`matplotlib` / `seaborn` / `pandas` under `eda`. Resolve an import name that is
not in the map in this order, and fail only when all three miss:

1. the environment map (`packages_distributions()`);
2. normalised name-equality against every requirement name declared in
   `dependencies` **or** any extra (`pandas` → `pandas`);
3. an explicit `[import_name_overrides]` table in
   `distribution/project_hygiene.toml`, keyed by import name, for the residue
   where the import name and the distribution name differ **and** the
   distribution is not installed — today exactly `iree = ["iree-base-compiler",
   "iree-base-runtime"]`.

Keep step 3 from rotting: an override entry whose import name *is* resolvable
through step 1 or 2 fails the gate as stale. That bounds the hand-written table
to the un-installable residue and makes it self-cleaning, which is what
distinguishes it from the config-table map rejected above.

Demonstrate the gate **red** against `origin/main` — where `grain` is declared
and unimported — then green after B4.

Note for review: no existing contract covers either direction.
`tests/distribution/test_packaging_metadata.py` checks license, classifiers and
`py.typed`; `test_public_api.py` checks `__all__`/`_LAZY` consistency and forbids
eager root imports. Neither looks at dependency usage.

**B6 (#4969) — the five `uv sync` call sites, reviewed.** Three are legitimate
and get a comment; two change.

- `distribution/release_readiness.toml:7` — `prerequisite_sync = ["dev", "eda"]`
  **gains `io`**. Every CI job that syncs `dev` also syncs `io`; the readiness
  gate is the one that does not, and it runs `uv sync` through
  `scripts/audit_release_readiness.py:240`.
- `.github/workflows/docs.yml:28` — **optional consistency hygiene, not a defect
  fix.** Recommend taking it as one line: drop the `uv sync --group docs --extra
  eda` step and put the group and extra on the `uv run` at `:31`
  (`uv run --group docs --extra eda sphinx-build -W -n -b html docs docs/_build`),
  matching `scripts/audit_docs_plumbing.py:164-172`. There is no live bug here —
  `uv run` is inexact by default and prunes nothing, so line 31's bare invocation
  does **not** undo line 28's install. The value is one shape for one operation
  across the repo. Skip it without consequence if the diff is getting long.
- `scripts/audit_coverage_dag.py:160` — legitimate. It builds `--extra` flags
  per-tier from the coverage DAG config, which is the point of a per-tier
  environment. One-line comment so the next sweep does not re-litigate it.
- `scripts/audit_release_readiness.py:240` — legitimate. Reads extras and groups
  from config, builds additive flags, runs as the first step of a release gate on
  a fresh environment rather than mid-chain. Comment.
- `.github/workflows/ci.yml:186` — legitimate in intent, **no longer bare in
  effect once B1 lands.** The step is named "Build wheel (no dev extra)" (`:184`),
  but under the alias the default `dev` group resolves to the 15-package extra,
  so a bare `uv sync` there installs beartype, libcst, jaxlint, chex and the rest.
  Wheel contents and the `:189-210` assertions are unaffected; the step name
  becomes false and the job pays for packages it does not want. Add
  `--no-default-groups` to that `uv sync` so the step stays true to its name,
  and comment it: "bare on purpose; `--no-default-groups` because `dev` is now an
  alias of the extra (B1)".

**B7 (#4969) — update `Justfile:231-234`.** That comment states the root cause is
"still live". When B lands it is not. Rewrite it to say what the B3 contract now
enforces, rather than deleting the history — the paragraph above it (`:211-229`)
is the reason the whole docs-plumbing file runs per-commit.

**B8 (#4969) — correct `scripts/audit_wave1_load_bearing.py:143-148`.** That
docstring says a bare `uv run pytest` "resyncs the shared venv ... silently
uninstalling packages". It is wrong: `uv run` is inexact by default and prunes
nothing (`uv run --help`). The mechanism it describes is `uv sync`'s, correctly
documented at `Justfile:216-220`. Rewrite the paragraph to name `uv sync` as the
replacing call, and keep `_UV_SYNC_EXTRAS` (`:137`) — explicit extras remain
useful for *guaranteeing presence*, which is a different and real reason.

**Gate for Phase B:**

```bash
just audit-project-hygiene          # B3 + B5 contracts
just audit-docs-build-contract      # test_docs_plumbing.py:117 pins docs.yml's shape
just audit-deterministic            # expect exit 0
uv sync && uv run --extra dev --extra io pytest tests/distribution/ -q
```

Both new contracts demonstrated **red** against `origin/main` first;
`uv lock` regenerated and committed; both B1 dry-run package lists pasted into the
PR body.

## Phase C — make the test-rigor gate report its own failure (#5021)

Branch `fix/5021-test-rigor-fail-loud`. Independent of A and B.

The scope is the **gate**, not the suite. Do not try to fix why the full
`pytest tests/` run produces no coverage data. That is #5002's input, and this
sprint's job is to make it legible.

### Gate (b) — this phase needs Marielle's per-event sign-off

`.praxia/docs/decisions/260714_2181-autoresearch-loop-constitution.md:46-52`
(gate (b), AC-22): *"any change to evaluator code, test splits, or metric
definitions requires Marielle's explicit sign-off before the changed evaluator is
trusted ... This gate fires on every evaluator-change event, not once — there is
no standing blanket approval."*

C6 changes what the evaluator passes on. Under the plain words, so does the whole
of Phase C: `test_rigor.py` is the sole producer of `test_rigor.line_coverage_pct`
and `test_rigor.branch_coverage_pct`, and C1/C3 change the environment those
measurements are taken in. **Treat all of Phase C as one gate-(b) event.**

The quote above elides the policy's second clause, which binds equally:
"*and* forces a closure-hash re-lock (T2-05) of the new evaluator's complete
closure (code + splits + metric defs + pinned deps + config)" (`:49-51`). Phase C
therefore carries **two** obligations, not one: the sign-off row, and the re-lock
that produces the identifier the row must reference. C7 below is the re-lock.

**Approving this specification is not the sign-off.** The sign-off is a new
`[[gates]]` row in `.praxia/loop_human_gates.toml`, written when Marielle approves
the Phase C PR. It mirrors the T2-30 row already there for PR #96 (`:27-41`) in
field shape only — **its `event_ref` is not a commit sha.**
`src/xtrax/loop/evaluator_change_gate.py:148` matches a gate-(b) attestation on
`raw.get("event_ref") == new_locked.closure_hash`, and that module's docstring
(`:21-24`, `:33`) says why: a closure hash cannot be silently reused to approve a
different change, a commit sha can. Nothing calls that checker today, but writing
the first live gate-(b) row in a form the repo's own checker is structurally
guaranteed to reject would be the same "green over a false fact" this sprint
exists to remove. The PR is not mergeable before the row exists.

```toml
# T2-29 evaluator-change approval for the Phase C test-rigor gate rewrite.
# Per-event gate per AC-22: this attestation covers exactly the commit below.
[[gates]]
id = "T2-29"
ac = "AC-22"
slug = "evaluator_test_rigor_fail_loud"
title = "Evaluator change: test-rigor gate reports and fails on its own failure (gate b)"
event_ref = "<ClosureManifest.closure_hash from C7 -- NOT a commit sha>"
attested_at = "<ISO-8601 UTC at approval>"
ttl_days = 7
attested_by = "Marielle Russo"
note = "<approval quote + what changed: report-path reservation, returncode/tests_failed in the verdict, explicit extras>"
```

The header comment at `:9-13` says gates (b)-(e) get "its own attestation entry
here the first time its event actually fires". This is that first time for (b).

**C1 (#5021) — reserve the path without creating the file.** Replace the
`tempfile.NamedTemporaryFile(delete=False)` block at `test_rigor.py:90-95` with a
`tempfile.TemporaryDirectory` and a path inside it:

```python
with tempfile.TemporaryDirectory(prefix="xtrax-test-rigor-") as tmpdir:
    cov_path = Path(tmpdir) / "coverage.json"
```

**Everything that touches `cov_path` moves inside that `with` block**: lines
`:97-129` — the `env`/`cmd` construction, the `subprocess.run` call, the
`is_file()` check, and the `parse_coverage_json` call. Only the final
`return CoverageStats(...)` stays outside, since it depends solely on values
already extracted into locals. A swap-only edit that leaves `:97-129` at the old
indentation exits the context manager before pytest runs, so the directory is
gone when pytest-cov tries to write into it and the gate fails for "no such
directory" rather than for anything real.

The `finally: cov_path.unlink(missing_ok=True)` at `:128-129` goes away — the
context manager owns cleanup. With no file pre-created, `cov_path.is_file()` at
`:121` becomes a real test and the branch at `:122-126` becomes reachable for the
first time.

**C2 (#5021) — surface the evidence on both failure paths.** `combined` (`:119`)
already holds stdout and stderr and is thrown away on every path that matters.
Two paths need it:

- **Report missing** (`:122-126`): already builds the right message. It just was
  never reachable.
- **Report present but unparsable**: wrap the `parse_coverage_json` call at
  `:127` and raise `RuntimeError` naming `result.returncode`, the report's byte
  size, and `combined`. A truncated or empty report is a distinct failure from a
  missing one and should say so.

Neither path may raise a bare `json.JSONDecodeError` (`:49`) again.

**C3 (#5021) — explicit extras on the subprocess.** The command at `:98-109` is a
bare `uv run pytest`. Add `--extra dev --extra io` to the argv, as a module-level
constant rather than an inline literal, following
`scripts/audit_wave1_load_bearing.py:137`/`:151`.

**The rationale is presence, not pruning.** `uv run` is inexact by default and
removes nothing (see "Measured, not inferred"), so this is *not* a fix for the CI
failure — `audit-orphans.yml:82` syncs exactly `dev` + `io` immediately before
invoking each recipe, so the environment was already correct where the failure was
observed. The value is that the gate becomes correct when run from a narrower
local environment, where `zarr` or a dev tool may genuinely be absent. State that
reason in the code comment; do not restate the pruning claim.

Same edit, same reason, on **all six** bare `uv run` lines across both recipes,
`audit-test-rigor-gate` (`Justfile:109-112`) and `audit-test-rigor-gate-quick`
(`:114-117`) — three lines each, including the `ruff check` and `pytest` lines
that are textually identical between the two.

**C4 (#5021) — a test that would have caught it.**
`tests/audit/test_test_rigor_gate.py` patches `run_pytest_coverage` out of every
existing test (`:106-109`, `:154-157`, `:195-198`), so this function has no
coverage. Add a test that drives it for real, with the `uv run pytest` argv
replaced by a stub that exits non-zero and writes no report. Assert the raised
`RuntimeError` message contains both the non-zero exit code and the stub's stderr
text. Add a second variant whose stub writes a zero-byte report, asserting the C2
unparsable path. Both must fail against `origin/main` — the first with
`JSONDecodeError`, the second the same.

**C5 (#5021) — trigger the rerun and record the answer.** This sub-task is
CI-ops and backlog-write, not code: it needs `gh workflow run` (or the Actions
UI) and a backlog update, so route it to the orchestrator or a human, not to a
code-editing fixer. After C merges, trigger
`audit-orphans.yml` by `workflow_dispatch` (`:28`) and read the
`audit-test-rigor-gate` log. Two hypotheses remain:

- **(a)** the full suite genuinely fails or errors, so pytest-cov writes no usable
  report. The gate's message will now name a non-zero pytest exit and the failing
  tests.
- **(c)** the suite passes but **coverage collects nothing**, so pytest-cov emits
  `CovReportWarning: Failed to generate report: No data to report` and writes no
  JSON. This is the hypothesis the evidence favours: it is the only one consistent
  with a zero-byte report, and the identical warning appears in a sibling artifact
  from the same run (`audit-drift-check.log:40`). The gate's message will show a
  **zero** pytest exit with a missing or empty report.

If (c) holds, name the likely cause for whoever picks up #5002: some test in the
full walk runs pytest or coverage itself and clobbers the parent run's `.coverage`
data file. Instruct them to start from `rg -n -- '--cov|\.coverage|coverage\.'
tests/` and list the candidate call sites in the #5002 body.

*(An earlier hypothesis — that a bare `uv run` resynced the venv mid-run — is
dead. `uv run` prunes nothing, and `audit-orphans.yml:82` had already synced
`dev` + `io`.)*

Record which held in backlog **#5002**, with the artifact link. That is #5002's
input and this sprint's only deliverable into it.

**C6 (#5021) — fail on a red suite.** `test_rigor.py:180` is
`passed = passes_line and passes_branch`. Add `result.returncode != 0` and
`stats.tests_failed > 0` to the verdict, each carrying the evidence into the
failure message (exit code, failed-test count, `combined`). `CoverageStats` already
carries `tests_failed` (`:31-32`); `returncode` needs threading out of
`run_pytest_coverage` (`:111-118`) onto that dataclass, or onto a sibling field on
`GateResult` (`:35-44`).

Without this the gate reports PASS over a suite that fails 300 tests, provided the
coverage numbers clear the baseline. **This is the sub-task that makes Phase C a
gate-(b) event** — it changes what the evaluator passes on, not merely how it
reports.

**The evidence needs a carrier, and today there is none.** `combined` is a local
in `run_pytest_coverage` (`:119`); `CoverageStats` (`:27-32`) has no field for it;
`GateResult` (`:35-44`) has no message field; `run_test_rigor_gate` returns rather
than raises; and `scripts/audit_test_rigor_gate.py:72-80` prints a fixed line with
no slot for detail. Specify the carrier exactly:

- `CoverageStats` gains `returncode: int = 0` and `pytest_output: str = ""`.
  **Both defaulted**, because the dataclass is `frozen=True, slots=True` and is
  constructed positionally at five sites that would otherwise break:
  `tests/audit/test_test_rigor_gate.py:99`, `:147`, `:188`, `:247` and
  `tests/audit/test_bootstrap.py:92`. `returncode = 0` is also the only default
  that keeps the existing baseline-ratchet tests green under the new
  `returncode != 0` condition.
- `GateResult` gains `failure_detail: str = ""`, populated by
  `run_test_rigor_gate` whenever `passed` is `False` with the exit code,
  `tests_failed`, and the tail of `pytest_output`.
- `scripts/audit_test_rigor_gate.py:72-80` prints `failure_detail` on `FAIL`.

**C7 (#5021) — the closure-hash re-lock the constitution requires.** Gate (b)'s
second clause: the changed evaluator's closure is re-locked, and the attestation
references the lock. `src/xtrax/loop/closure_lock.py:98-105` already exposes
`build_closure_manifest(*, evaluator_paths, split_paths, metric_def_paths,
config, pinned_deps_source=...)` over arbitrary paths, so this needs no new
machinery:

```python
manifest = build_closure_manifest(
    evaluator_paths=(Path("src/xtrax/devtools/gates/test_rigor.py"),
                     Path("scripts/audit_test_rigor_gate.py")),
    split_paths=(),
    # The pass/fail thresholds are the ratchet baseline the CLI reads via
    # DEFAULT_BASELINE_PATH (src/xtrax/devtools/baseline.py:19).
    metric_def_paths=(Path(".praxia/audit_baseline.json"),),
    config={"extras": ["dev", "io"], "tests_path": "tests"},
)
manifest.closure_hash  # -> the T2-29 event_ref
```

Compute it on the PR's final head, record the hash and the input paths in the PR
body, and put the hash in the T2-29 row's `event_ref`. If the evaluator files
change again after the hash is taken, the hash is stale and the row does not
match — that is the mechanism working, recompute and re-attest.

**Gate for Phase C:**

```bash
uv run --extra dev --extra io pytest tests/audit/test_test_rigor_gate.py -q
just audit-test-rigor-gate-quick    # expect exit 0
```

Green, with both C4 tests demonstrated **red** against `origin/main`; a `T2-29`
row present in `.praxia/loop_human_gates.toml` whose `event_ref` is the C7
closure hash computed on the PR's final head; and
one `audit-orphans.yml` `workflow_dispatch` run completed with its
`audit-test-rigor-gate` outcome — hypothesis (a) or (c) — written into #5002.
**The phase is not done at merge — it is done when #5002 carries the answer.**

## Verification

Per phase, before any push:

```bash
uv run --extra dev --extra io pytest tests/run/test_zarr_integrity.py tests/run/ -k "digest or repro_floor" -q
uv run --extra dev --extra io pytest tests/distribution/ tests/audit/test_test_rigor_gate.py -q
uv run --extra dev ruff check . && uv run --extra dev ruff format --check . && uv run --extra dev ty check src/
just audit-deterministic         # stops at first failure; expect exit 0
```

`uv sync` replaces the environment; `uv run` does not. Every `uv sync` in a chain
is a hazard and every one above is deliberate. Carry `--extra` flags on `uv run`
calls anyway, to guarantee presence in a narrow local environment — not to prevent
pruning, which `uv run` does not do. A concurrent `uv sync` during a gate run
produces spurious failures; run gates solo.

Never run the whole suite locally. Each command above is narrowed to a path or a
`-k` selector for that reason.

Then CI: all jobs green, including `audit-release-readiness-contract`, which
Phase B's `release_readiness.toml:7` edit touches directly. Then the
`workflow_dispatch` rerun of `audit-orphans.yml`, read for its **job summary**,
not its badge — that workflow is green by design (`audit-orphans.yml:14-19`).

## Out of scope

- **#5002 (phase-2 orphan gating).** Pinning the passing set is dishonest until
  #5021 lands and one clean `audit-orphans.yml` rerun shows which of C5's
  hypotheses holds. Next sprint. Phase C feeds it.
- **#4966.** A policy decision, and the user's to make, not an agent's.
- **#4856** (WebGPU research) — deliberately deferred, still open by design.
- **#4584** (loop-constitution gate (b) for the controller) — a *different*
  gate-(b) event from Phase C's, and separately unresolved.
- **Cross-repo #5008 and #5012**, and any edit to the demistify or aminx trees.
- **The SafeMap rename (#3644).**
- **A marker/namespace layout for sink provenance** — see A1. Filed as a
  follow-up from the Phase A PR, not done here.
- **Four of the orphan run's five failures.** The run failed on:
  1. `audit-test-rigor-gate` — **this sprint, Phase C**;
  2. `audit-api-ergonomics-gate` — baseline regression,
     `param_sprawl_violation_count = 4`. Deferred, file-worthy.
  3. `audit-structure-complexity-gate` — baseline regression,
     `cognitive_complexity_max = 117`, 42 ruff complexity violations, 62 findings.
     Deferred, file-worthy.
  4. `audit-bootstrap` — not an independent failure: a cascade of (2) and (3),
     reporting `failed dimensions: api_ergonomics, structure_complexity`. Closes
     when they do. **Do not expect the Phase C fix to turn it green.**
  5. `audit-drift-check` — environmental. It needs a real `praxia plugin install`
     export to diff against and the praxia CLI is not installed in CI, exactly as
     the `Justfile:38-44` comment predicts. Local-only by design until T3-01's
     install step is wired into a CI job.
- **The stale remote branch `probe/webgpu-ci-adapter`** — deletion is blocked for
  me by the permission classifier.
