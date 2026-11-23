---
title: 'Migrate dependents onto xtrax.jax_cache.configure() (debt #3527)'
description: Per-project verdicts, call sites, pin plan, namespace decision and verification for moving aminx, demistify and aminx-ensemble onto xtrax.jax_cache.configure(); NOT-APPLICABLE evidence for denxity, aminx-lm, prolix, alynxr.
status: draft
task_id: 261123_xtrax-jax-cache-consumers
date: '261123'
backlog_ids: ''
adversarial_review: ''
---
# Migrate dependents onto xtrax.jax_cache.configure() (debt #3527)


task_id: `261123_xtrax-jax-cache-consumers` · xtrax debt **#3527** · source PR maraxen/xtrax#278 (released in `v0.4.1b9`, on PyPI)

## 1. Goal and scope

Replace hand-written JAX persistent-cache setup in **runtime entry points and library code** of xtrax
dependents with `xtrax.jax_cache.configure()` (or `resolve`/`validate`/`apply` across process pools),
so every project reads the same layers: explicit argument > `$XTRAX_JAX_CACHE_DIR` >
`$JAX_COMPILATION_CACHE_DIR` > `~/.config/xtrax/config.toml`, policy from `$XTRAX_JAX_CACHE` /
`[tool.xtrax] jax_compilation_cache`.

**Out of scope (owner decisions 2026-11-23):**

- Pre-registered experiment and benchmark scripts with a committed `.bth.toml` (they set cold/warm
  caches deliberately; their sidecars are receipts): demistify `scripts/perf/fit_compile_profile.py`,
  prolix `scripts/benchmarks/b1_init_exec.py`, aminx and aminx-ensemble
  `scripts/ebm/benchmarks/heterogeneous_batch_benchmark.py`, aminx `scripts/parity/*`,
  `scripts/analysis/laser_sample_chunk_width_cost.py`.
- Shell job scripts that export `JAX_COMPILATION_CACHE_DIR` (aminx `scripts/engaging/*.sbatch`,
  `.github/workflows/ci.yml:112`; aminx-ensemble `scripts/remote/with_env.sh:205`,
  `scripts/ebm/slurm/*.sbatch`). They keep working: that variable is layer 3 of `configure()`.
- Script-only dependents tev_design, biosensors, oaf, protamer, asr → xtrax debt **#3557**.
- denxity, aminx-lm, prolix, alynxr → per-project entry-point debts (section 3).

## 2. Assumptions

| # | Claim | Tag | Evidence |
|---|---|---|---|
| A1 | `jax_cache` ships in xtrax `0.4.1b9`, on PyPI | verified | `git tag --contains dec5a03e` → `v0.4.1b9`; PyPI release list ends `0.4.1b9` |
| A2 | 0.4.1b9 requires `jax>=0.10.2,<0.12`; compatible with all three ADOPT repos | verified | `v0.4.1b9:pyproject.toml` deps; aminx/aminx-ensemble `jax>=0.11.2,<0.12`, demistify `jax>=0.10.2,<0.11` |
| A3 | b8→b9 has no removals; only deprecation is `safe_scan` (DeprecationWarning) | verified | `v0.4.1b9:CHANGELOG.md` §0.4.1b9 |
| A4 | `configure()` works on jax 0.10.2 (demistify's lock): private `compilation_cache._cache` and `reset_cache` exist; after-compile switch resets; `none` clears | verified (throwaway spike, pass/fail only) | uv isolated env jax/jaxlib 0.10.2 + xtrax 0.4.1b9: entries under `jaxlib-0.10.2/`, reset logged, live dir `None` after off |
| A5 | When no layer sets a directory or a policy, `configure()` leaves JAX untouched. A non-empty `$JAX_COMPILATION_CACHE_DIR` **is** a layer (3) and is namespaced; the xtrax docstring (`jax_cache.py:42-43`) says otherwise and is wrong (xtrax debt #3560) | verified (R2 N2) | `jax_cache.py:282-283`, `:786-793` |
| A6 | An explicit `path` outranks a project `policy = "off"` | verified | `jax_cache.py:517-519` |
| A7 | Disable words apply only to `str`; a `Path("none")` is a directory named `none` | verified | `jax_cache.py:295-296` |
| A8 | Default namespace writes under `<dir>/jaxlib-<ver>`; one cold run per previously flat dir | verified | `jax_cache.py:617-623`; owner chose `auto` 2026-11-23 |
| A9 | Importing `aminx.cli` and `aminx.host.campaign` already imports jax, so `configure()` in their `main()` adds no import cost | verified | main-checkout venv, worktree src: `jax in sys.modules` True after import (5.3 s, 3.7 s) |
| A10 | demistify `cli.py` imports jax at module level | verified | `src/demistify/cli.py:32` |
| A11 | aminx-ensemble `cli.py` → `aminx_ensemble.mcp` imports jax at import | verified (by reading; R1 challenger) | `cli.py:7` → `mcp.py:26` → `protocol.py:9` `import jax` |
| A12 | demistify's spawn pool (`extract.py:419`) runs no JAX in workers | verified | `extract.py` imports numpy/mdtraj only (`:13-18`) |
| A13 | aminx `host/campaign.py` has its own `main(argv)` entry (planner/worker CLI) | verified | `campaign.py:~3315`, `configure_multiprocessing()` then argparse |
| A14 | aminx done-marker telemetry is diagnostic; resume compares `zarr_content_digest` | verified | `campaign.py:2046-2064` |
| A15 | alynxr runs no `jit` in `src/`; its ColabFold probe runs under a foreign interpreter without xtrax | verified | `git grep jax.jit src/` empty; `fold.py:264` `Install.python`; `probes/colabfold_run.py` has no xtrax import |
| A16 | denxity, aminx-lm, prolix define no `[project.scripts]` (prolix has a runnable `viewer_server.py`, not a console entry; tracked in #3555) | verified | `origin/main:pyproject.toml` |
| A17 | aminx and aminx-ensemble also ship MCP console entries that run JAX in-process | verified (R1 M1) | aminx `pyproject.toml:40` → `agent/mcp.py:61`, `sample` tool `agent/tools.py:288`; aminx-ensemble `pyproject.toml:63` → `mcp.py:1212`, `run_ensemble` `:956` |
| A18 | demistify hashes xtrax `tiling/`, `transforms/`, `intent/`, `profiling/`, `safety/` and the `safe_scan` module into `code_fingerprint`; b8→b9 changes `transforms/scan.py`, `transforms/__init__.py`, `tiling/dispatch.py`, so a b8-written store is not reused: `code_fingerprint` is in the store identity (`shard_identity.py:396`); an explicit store path raises `ShardStoreMismatch` (`pipeline.py:1088-1092`), `path="auto"` opens a new store with only a warning (`shard_sink.py:1040`) (corrected in P5 review of demistify#113) | verified (R1 M2) | `src/demistify/shard_identity.py:166-187`; `git diff --stat v0.4.1b8 v0.4.1b9`; owner: bump anyway, warn in PR (2026-11-23) |
| A19 | `configure()` walks every ancestor `pyproject.toml` from cwd for policy/knobs; a malformed one raises | verified (R1 M6) | `xtrax/config.py:326-352` |
| A20 | aminx `training/dataloading/preprocess.py:259` `mp.Pool` workers parse structures (numpy/msgpack); not a compile site | verified by reading; implementer re-greps the worker for `jit` | `preprocess.py:17-35,72,259` |

## 3. Verdicts

| Project | Verdict | Base | Evidence | Debt |
|---|---|---|---|---|
| aminx | **ADOPT** | `origin/main` 12e20f66 | `src/aminx/host/prep.py:191-192`; TODO at `src/aminx/utils/compile_telemetry.py:17-20` | #3527 |
| demistify | **ADOPT** (entry point only) | `origin/main` 462ca48 | `cli.py:1021` `main()`; only cache site is a pre-registered perf script | #3527 |
| aminx-ensemble | **ADOPT** (entry point only) | `origin/epic/aminx-ensemble` 359e41a0 (`origin/main` is the empty scaffold) | `cli.py:10` `main()`; cache path otherwise via env (`with_env.sh:205`) | #3527 |
| alynxr | NOT-APPLICABLE as drop-in | — | A15; resolver `fold.py:713-764` targets another machine's process; `portable_env()` deliberately leaves the directory to the target | #3556 |
| denxity | NOT-APPLICABLE (no entry point) | — | A16 | #3553 |
| aminx-lm | NOT-APPLICABLE (no entry point; vendored `xtrax-0.4.1b3` wheel) | — | A16; `[tool.uv.sources] xtrax = { path = "wheels/xtrax-0.4.1b3-py3-none-any.whl" }` | #3554 |
| prolix | NOT-APPLICABLE (no entry point) | — | A16 | #3555 |

## 4. Design per ADOPT project

**Lock rule (all three, amended in P3):** after `uv lock --upgrade-package xtrax`, no package other than
xtrax may change **version** or be added/removed. Marker re-simplification on unchanged versions
(uv rewrites `marker = ...` on e.g. `jaxlib`, `jax-cuda12-plugin`, `cuda-pathfinder` after a new
resolution) is accepted; list it in the PR body.

Rule for all three: call `configure()` **once at the start of every console entry point** (CLI and MCP),
before any compile, never at import time in library code. Do not pass `namespace` (owner: `auto`).

**Error policy (R1 M6, ruled):** catch nothing. A malformed `~/.config/xtrax/config.toml` or ancestor
`pyproject.toml` (A19), an unwritable directory or a missing `filelock` fails the command, including
non-JAX subcommands and `--help`. Tie-breaker: the owner's hard rule "fail loudly on a malformed file"
for derived-data configuration. The PR body states this. No lazy per-command configure.

### 4.1 aminx (`xtrax==0.4.1b8` → `==0.4.1b9`)

1. **Pin**: `pyproject.toml:27,50,70` → `==0.4.1b9`; `uv lock --upgrade-package xtrax`; commit `uv.lock`;
   the lock diff may move only xtrax (check every extra's resolution in the diff).
2. **Entry points** — `jax_cache.configure()` in each:
   `src/aminx/cli.py:2286` `main()` before `run_cli(app)`;
   `src/aminx/host/campaign.py` `main()` right after `configure_multiprocessing()`;
   `src/aminx/agent/mcp.py:61` `main()` after `cisternal.init()`, before `build_server().run()` (A17).
3. **`src/aminx/host/prep.py:191-192`**: replace the `jax.config.update` with
   `jax_cache.configure(path=spec.run_spec.io.cache_path)` when not None. When None, do nothing. Note: a
   library caller that calls `prep_*` directly without an entry point and without `cache_path` gets no
   xtrax layers; that matches today (JAX untouched) and is documented in the docstring.
4. **`src/aminx/utils/compile_telemetry.py`** (R1 M4): `dir` is the **live** `jax.config.jax_compilation_cache_dir`
   (`None` when unset, never the string `"None"`); `enabled = bool(dir)`; `source` is
   `jax_cache.current().source` when `current()` is not None **and** `current().dir` equals the live dir
   (or both are off), else `"jax.config (not set by xtrax jax_cache)"`. Same three keys, no schema bump.
   Remove the TODO.
5. **Tests**:
   - Autouse fixture in `tests/conftest.py` (or the touched files' conftest), per test:
     monkeypatch `XDG_CONFIG_HOME` and `HOME` to `tmp_path` subdirs so `~/.config/xtrax/config.toml`
     on the host is invisible (R2 N1); **no `chdir` in the autouse fixture** (R3 N3: demistify
     `tests/test_gaff2_checker.py:98-99` and aminx-ensemble `tests/aflf/test_config_default.py:14` rely on
     the repo-root cwd, whose `pyproject.toml` has no `[tool.xtrax]`); delete `XTRAX_JAX_CACHE*` and `JAX_COMPILATION_CACHE_DIR`; then
     `jax.config.update("jax_compilation_cache_dir", None)` explicitly (JAX read the env at import, so
     unsetting the variable does not clear the flag; the CI env at `ci.yml:112` sets it). Teardown:
     `jax._src.compilation_cache.reset_cache()`, `xtrax.jax_cache._applied = None`, restore the saved
     flag value. No test assumes the flag started at None without this fixture.
   - `tests/host/test_prep_cache.py`: expected dir is `cache_dir / f"jaxlib-{importlib.metadata.version('jaxlib')}"`.
     Replace the blanket `except Exception: pass` so a `JaxCacheDirError` / `JaxCacheConfigError` re-raises
     (catch only the mocking-induced error type, or assert on the cache before the failing call).
   - New entry-point tests, one per entry (cli, campaign, mcp): the entry modules call through the
     module attribute (`from xtrax import jax_cache; jax_cache.configure()`) so a monkeypatched
     `xtrax.jax_cache.configure` spy is seen; the aminx MCP test uses `pytest.importorskip` for the
     agent extra. Only the new entry-point, real-effect, off and fail-loud tests also
     `monkeypatch.chdir(tmp_path)`. Monkeypatch the spy, stub the dispatched app/server so nothing runs, call `main`, assert the spy ran once
     **before** the stub. Control: deleting the call turns each red.
   - Real-effect test (one, on `cli.main`): `XTRAX_JAX_CACHE_DIR=<tmp>` → live dir is `<tmp>/jaxlib-<ver>`
     **and** that directory exists on disk.
   - Fail-loud test: write a malformed `config.toml` under the patched `XDG_CONFIG_HOME/xtrax/` →
     `cli.main` raises `ValueError` naming the file.
     Off test: pre-set `jax.config.update("jax_compilation_cache_dir", <other tmp>)`, set
     `XTRAX_JAX_CACHE_DIR=none`, call `main` → live dir is `None`. This goes red if the call is deleted
     (the pre-set dir would remain), so the control is real.
   - `compile_telemetry`: (a) after `configure(path=tmp)` → dir is the namespaced path, source
     `explicit argument`; (b) after `configure(path="none")` → `{"enabled": False, "dir": None, ...}`;
     (c) live dir changed behind xtrax's back → fallback source.
6. Confirm by grep that `_worker_process_protein` (A20) never jits; if it does, the PR uses
   `resolve`/`validate` in the parent and `apply` via a pool initializer.

### 4.2 demistify (`xtrax[pack]>=0.4.1b8` → `>=0.4.1b9`; `eda` extra likewise)

1. Pin floor `pyproject.toml:42,63`; `uv lock --upgrade-package xtrax`; commit `uv.lock`; lock diff moves only xtrax.
2. `src/demistify/cli.py:1021` `main()`: `jax_cache.configure()` after `cisternal.init()`, before `run_app`.
3. Tests: the spy test, the real-effect test and the pre-set-then-`none` off test from 4.1.5, on
   `demistify.cli.main` with `run_app` stubbed; same autouse isolation fixture.
4. **PR body must state A18**: durable stores written under xtrax b8 refuse to resume after this lands
   (see A18 for the exact failure mode); finish or restart them.

### 4.3 aminx-ensemble (`xtrax[io,data]>=0.4.1b6,<0.5` → `>=0.4.1b9,<0.5`)

1. **Pin crosses xtrax 0.4.1b8, which removed `optax` and `orbax-checkpoint` from xtrax's core
   dependencies** (`v0.4.1b8:pyproject.toml`). aminx-ensemble's `src/aminx_ensemble/ebm/training.py:126`
   imports `optax` at module level, but the `ebm` extra (`pyproject.toml:21-24`) declares only
   `orbax-checkpoint`; add `"optax>=0.2.3"` to the `ebm` extra. (`aflf`/`afrw` already declare optax;
   `ebm`/`bioemu` already declare orbax.) aminx and demistify are already on b8 and declare what they import.
   Pin floor `pyproject.toml:11` (and the unpinned `xtrax[eda]` at `:58` stays as is); `uv lock --upgrade-package xtrax`;
   commit `uv.lock` (`scripts/remote/with_env.sh` runs `uv sync --frozen`). Expect a `DeprecationWarning`
   from `src/aminx_ensemble/ebm/_tiling.py:16,98` (`xtrax.safe_scan`); out of scope, noted in the PR.
2. Entry points: `src/aminx_ensemble/cli.py:10` `main()` before dispatching `app`; `src/aminx_ensemble/mcp.py:1212`
   `main()` before `server.run()` (A17).
3. Tests: spy tests for both mains, the real-effect and off tests on `cli.main`, same isolation fixture.
   PR targets `epic/aminx-ensemble`.

## 5. Old vs new behaviour

### 5.1 aminx `prep.py` (the only replaced setter)

| Case | Before | After |
|---|---|---|
| `cache_path` set, writable | flat `<dir>`, no mkdir, JAX creates it lazily | `<dir>/jaxlib-<ver>`, created and write-probed up front; first run cold once |
| `cache_path` unwritable | JAX logs a failed put per entry; run silently cold | `JaxCacheDirError` naming the dir at prep |
| `cache_path` relative | relative to cwd at first compile | relative to cwd at `configure()` (same in practice) |
| `cache_path == Path("none")` | directory `./none` | directory `./none/jaxlib-<ver>` (A7; unchanged semantics) |
| `cache_path` None | JAX untouched | untouched by prep; the entry point's `configure()` applied env/user config |
| set after a compile in the same process (several specs, different `cache_path`) | JAX keeps writing to the old dir (flag ignored once `_cache` exists) | `_cache` reset, warning, new dir used |
| explicit `cache_path` vs `policy="off"` from project **or** `$XTRAX_JAX_CACHE=off` | n/a | explicit wins (A6, `jax_cache.py:517-519`) |

### 5.2 Every entry point (aminx cli/campaign/mcp, demistify cli, aminx-ensemble cli/mcp)

| Case | Before | After |
|---|---|---|
| no layer sets a directory or policy (no env var, no user config, no `[tool.xtrax]` policy) | JAX default | JAX untouched (A5) |
| `$XTRAX_JAX_CACHE_DIR` or user config set | ignored | `<dir>/jaxlib-<ver>`; mkdir + 4 KiB write probe on **every** invocation, including planning and `--help` |
| `$JAX_COMPILATION_CACHE_DIR=<dir>` (myxcel, aminx CI `ci.yml:112`, `with_env.sh:205`, sbatch scripts) | JAX uses `<dir>` flat | xtrax reads it as layer 3 → `<dir>/jaxlib-<ver>`; first job per dir cold once |
| `$JAX_COMPILATION_CACHE_DIR=none` | JAX writes `./none` | off; flag cleared |
| malformed user config or ancestor `pyproject.toml` (A19), unwritable dir | runs | command fails with a named error (ruled, section 4) |
| existing tests that call a `main()` | no cache side effects | real `configure()` runs; isolated by the autouse fixture (4.1.5) |

## 6. Verification

- Narrow local runs only, threads capped: the touched test files, one at a time. No full suites.
- **Controls**: show in each PR body the spy tests and the off test red with the `configure()` call removed.
- CI on each PR; read failures from the logs, not the status.
- No bathos run: nothing here produces a finding.

## 7. Delivery

One branch `feat/261123-jax-cache-configure` and one PR per ADOPT repo (aminx → `main`, demistify → `main`,
aminx-ensemble → `epic/aminx-ensemble`). PR bodies name debt #3527, this spec, the section 5.2 behaviour
changes, and what CI does not cover (cluster runs: first job per cache dir is cold once under
`jaxlib-<ver>/`). Resolve #3527 when the three PRs merge; #3553–#3557 stay open.

## 8. Review log

- R1 (challenger, Sonnet, dsp_59100b09d0962a4c): 0 FATAL, 6 MAJOR, 6 MINOR. All six MAJOR accepted:
  M1 → A17 + MCP entries; M2 → A18 + PR warning (owner: bump now); M3 → test plan 4.1.5;
  M4 → telemetry reads the live dir; M5 → table 5.2; M6 → accepted as a behaviour change and ruled
  "fail loudly" (owner's hard rule), documented rather than made lazy. MINOR: A11 resolved; explicit vs
  `$XTRAX_JAX_CACHE=off` added to 5.1; library-caller note in 4.1.3; per-repo lock checks + committed locks;
  A20 pool; prolix viewer_server noted in A16.
- R2 (same challenger): all six R1 MAJOR resolved; 0 FATAL; 2 new MAJOR accepted: N1 → fixture isolates
  user config, cwd and the import-time JAX flag; N2 → A5 and 5.2 reworded (env var is layer 3, namespaced),
  xtrax docstring bug filed as #3560. MINORs accepted: module-attribute calls for spies, importorskip for
  the agent extra, a fail-loud test, on-disk check in the real-effect test.
- R3 (same challenger): N1, N2 resolved; 1 new MAJOR accepted: N3 → `chdir` moved out of the autouse
  fixture into the new tests only (cwd-dependent existing tests cited).
- P3 amendment (orchestrator, after implementer stop on lock diff, dsp_4890178511b0c950): lock rule
  reworded to versions-only; aminx-ensemble `ebm` extra gains `optax>=0.2.3` because the b6→b9 jump crosses
  xtrax b8's removal of optax/orbax from core dependencies.
- P5 (PR review): aminx#303 R1 1 MAJOR (host user config leaked into existing entry-point and subprocess
  tests) → suite-wide autouse `XDG_CONFIG_HOME` redirect + opt-in strict fixture; demistify#113 R1 1 MAJOR
  (wrong exception named for stale stores) → A18 and PR body corrected; aminx-ensemble#57 R1 no BLOCKER/MAJOR
  (mcp real-effect test added). Round 2 dispatched for #303 and #113.
