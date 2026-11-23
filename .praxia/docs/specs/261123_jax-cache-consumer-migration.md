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
| A5 | With nothing configured, `configure()` leaves JAX untouched (incl. an env `JAX_COMPILATION_CACHE_DIR`) | verified | `jax_cache.py:786-793` |
| A6 | An explicit `path` outranks a project `policy = "off"` | verified | `jax_cache.py:517-519` |
| A7 | Disable words apply only to `str`; a `Path("none")` is a directory named `none` | verified | `jax_cache.py:295-296` |
| A8 | Default namespace writes under `<dir>/jaxlib-<ver>`; one cold run per previously flat dir | verified | `jax_cache.py:617-623`; owner chose `auto` 2026-11-23 |
| A9 | Importing `aminx.cli` and `aminx.host.campaign` already imports jax, so `configure()` in their `main()` adds no import cost | verified | main-checkout venv, worktree src: `jax in sys.modules` True after import (5.3 s, 3.7 s) |
| A10 | demistify `cli.py` imports jax at module level | verified | `src/demistify/cli.py:32` |
| A11 | aminx-ensemble `cli.py` → `aminx_ensemble.mcp.app` imports jax at import | **unverified** | main-checkout venv lacks `cisternal`; implementer probes in the worktree env |
| A12 | demistify's spawn pool (`extract.py:419`) runs no JAX in workers | verified | `extract.py` imports numpy/mdtraj only (`:13-18`) |
| A13 | aminx `host/campaign.py` has its own `main(argv)` entry (planner/worker CLI) | verified | `campaign.py:~3315`, `configure_multiprocessing()` then argparse |
| A14 | aminx done-marker telemetry is diagnostic; resume compares `zarr_content_digest` | verified | `campaign.py:2046-2064` |
| A15 | alynxr runs no `jit` in `src/`; its ColabFold probe runs under a foreign interpreter without xtrax | verified | `git grep jax.jit src/` empty; `fold.py:264` `Install.python`; `probes/colabfold_run.py` has no xtrax import |
| A16 | denxity, aminx-lm, prolix define no `[project.scripts]` | verified | `origin/main:pyproject.toml` |

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

Rule for all three: call `configure()` **once at the start of each console entry point**, before any
compile, never at import time in library code. Do not pass `namespace` (owner: `auto`). Catch nothing:
a malformed config must fail loudly (`ValueError`, `JaxCacheConfigError`, `JaxCacheDirError`).

### 4.1 aminx (`xtrax==0.4.1b8` → `==0.4.1b9`)

1. **Pin**: `pyproject.toml:27,50,70` → `==0.4.1b9`; regenerate `uv.lock` (`uv lock --upgrade-package xtrax`); confirm no other package moves.
2. **Entry points**: `src/aminx/cli.py` `main()` (before `run_cli(app)`) and `src/aminx/host/campaign.py` `main()` (after `configure_multiprocessing()`): `jax_cache.configure()`.
3. **`src/aminx/host/prep.py:191-192`**: replace `jax.config.update("jax_compilation_cache_dir", str(cache_path))` with `jax_cache.configure(path=spec.run_spec.io.cache_path)` when not None. When None, do nothing (entry point already resolved).
4. **`src/aminx/utils/compile_telemetry.py`**: return the setting from `jax_cache.current()` when set: `{"enabled": s.enabled, "dir": str(s.dir) or None, "source": s.source}` (same three keys, no schema bump); fall back to the current `jax.config` read with `source = "jax.config (xtrax jax_cache not applied in this process)"` when `current()` is None. Remove the TODO.
5. **Tests** (`tests/host/test_prep_cache.py`): expected dir becomes `jax_cache.current().dir` and must equal `cache_dir / f"jaxlib-{jaxlib version}"`; the None-case test is unchanged. Add `tests/utils/test_compile_telemetry.py` (or extend the nearest existing file): after `configure(path=tmp)` the dict reports that dir and source `explicit argument`; with nothing applied it reports the fallback source. Reset xtrax module state between tests (`monkeypatch.setattr(jax_cache, "_applied", None)` and restore `jax.config`).

### 4.2 demistify (`xtrax[pack]>=0.4.1b8` → `>=0.4.1b9`; `eda` extra likewise)

1. Pin floor in `pyproject.toml:42,63`; `uv lock --upgrade-package xtrax`.
2. `src/demistify/cli.py` `main()`: `jax_cache.configure()` after `cisternal.init()`, before `run_app`.
3. Test: invoking `main([...a cheap command such as --help or version...])` with `XTRAX_JAX_CACHE_DIR=<tmp>` sets `jax.config.jax_compilation_cache_dir` to `<tmp>/jaxlib-<ver>`; with `XTRAX_JAX_CACHE_DIR=none` it stays unset.

### 4.3 aminx-ensemble (`xtrax[io,data]>=0.4.1b6,<0.5` → `>=0.4.1b9,<0.5`)

1. Pin floor `pyproject.toml:11`; `uv lock --upgrade-package xtrax`. Expect a `DeprecationWarning` from `src/aminx_ensemble/ebm/_tiling.py:16,98` (`xtrax.safe_scan`); not in scope, note in the PR.
2. `src/aminx_ensemble/cli.py` `main()`: `jax_cache.configure()` before dispatching `app`.
3. Probe A11 first; if `cli.py` import does not already import jax, record the added import time in the PR.
4. Test as 4.2.3. PR targets `epic/aminx-ensemble`.

## 5. Old vs new behaviour (aminx `prep.py`, the only replaced setter)

| Case | Before | After |
|---|---|---|
| `cache_path` set, writable | flat `<dir>`, no mkdir, JAX creates it lazily | `<dir>/jaxlib-<ver>`, created and write-probed up front; first run cold once |
| `cache_path` unwritable | JAX logs a failed put per entry; run silently cold | `JaxCacheDirError` naming the dir at prep |
| `cache_path` relative | relative to cwd at first compile | relative to cwd at `configure()` (same in practice) |
| `cache_path == Path("none")` | directory `./none` | directory `./none/jaxlib-<ver>` (A7; unchanged semantics) |
| `cache_path` None | JAX untouched | untouched by prep; entry point's `configure()` applied env/user config |
| set after a compile in the same process | JAX keeps writing to the old dir (flag ignored once `_cache` exists) | `_cache` reset, warning, new dir used |
| project `policy = "off"` + explicit `cache_path` | n/a | explicit wins (A6) |
| `$JAX_COMPILATION_CACHE_DIR=none` in env, no `cache_path` | JAX writes `./none` | xtrax clears the flag (off) |

## 6. Verification

- Narrow local runs only, threads capped: the touched test files, one at a time. No full suites.
- **Controls**: each new test must go red with the `configure()` call removed (show the red run in the PR body), and the `none` test must go red if the call ignores env (e.g. `configure(path=<tmp>)` hard-coded).
- CI on each PR; read failures from the logs, not the status.
- No bathos run: nothing here produces a finding.

## 7. Delivery

One branch `feat/261123-jax-cache-configure` and one PR per ADOPT repo (aminx → `main`, demistify → `main`,
aminx-ensemble → `epic/aminx-ensemble`). PR bodies name debt #3527, this spec, and what CI does not cover
(cluster runs: first job per cache dir is cold once under `jaxlib-<ver>/`).
Resolve #3527 when the three PRs merge; #3553–#3557 stay open.
