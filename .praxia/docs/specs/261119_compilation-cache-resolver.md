---
title: xtrax compilation-cache resolver
description: One layered resolver for the JAX persistent compilation cache dir, used by every xtrax entry point and consumer
status: draft
task_id: 261119_compilation-cache-resolver
date: '261119'
backlog_ids: ''
adversarial_review: 'spec-challenger (sonnet) 261119 NEEDS_WORK -> B1-B3, S1-S7, N1 folded in; S5 resolved by descoping max size'
---
# xtrax compilation-cache resolver

## Goal

Any xtrax project, and any consumer of xtrax, uses the JAX persistent compilation cache the same
way: one function decides the directory from layered configuration, applies it to JAX, and reports
which layer decided. A project opts in with one line of `pyproject.toml`; a machine opts in with one
line of `~/.config/xtrax/config.toml`; a job overrides with one env var.

## Current state (verified)

| Site | Behaviour | Evidence |
|---|---|---|
| campaign executor | explicit > `JAX_COMPILATION_CACHE_DIR` (`none`/empty = off) > off; `min_compile_time_secs=1.0` | `src/xtrax/campaign/executors/local.py:91-122` |
| sweep CLI (deprecated) | hardcodes `.xtrax/jax_cache`, relative to cwd | `src/xtrax/cli/sweep_verb.py:162` |
| engine / intent / run | no cache handling | grep of `src/` |
| alynxr | own resolver: arg > `ALYNXR_COMPILATION_CACHE_DIR` > `JAX_…` > registry | `alynxr/src/alynxr/fold.py:712` |
| aminx | `run_spec.io.cache_path` | `aminx/src/aminx/host/prep.py:177` |
| tev_design, protamer, jaxbeans | own env/flag handling, `min_compile_time_secs=0` | `run_design_grid.py:30`, `fit_production_library.py:148`, `jaxbeans/core/env.py:26` |

**Existing defect (verified, jax 0.11.1 spike):** JAX reads `JAX_COMPILATION_CACHE_DIR` itself at
import (`jax/_src/config.py:1626`, flag value printed at import). With the variable set to `none`,
the campaign executor's resolver returns `None` and does nothing, but JAX has already taken `none` as
a relative path and writes a cache into `./none/` under the cwd. Clearing the flag with
`jax.config.update("jax_compilation_cache_dir", None)` before the first compile prevents it
(spike: `./none` absent afterwards). No current test catches this: `tests/campaign/executors/
test_local.py:240-244` asserts only that `config.update` was called with a directory.

## JAX semantics the design relies on

All verified on jax 0.11.1, CPU, by spike; each becomes a required test (T4-T6), not a citation of
the spike.

- **V1 lazy init.** `_initialize_cache` (`jax/_src/compilation_cache.py:241-258`) runs at most once,
  and only sets `_cache_initialized` when a directory is set and caching is enabled. A compile that
  ran with no directory set does not lock anything: setting a directory afterwards takes effect for
  later compiles.
- **V2 silent ignore after init.** Once initialized with directory A, updating the flag to B is
  ignored: every later entry lands in A and B is never created. The flag value then lies about where
  entries go.
- **V3 public reset.** `jax.experimental.compilation_cache.compilation_cache.reset_cache` (re-export of
  `_src/compilation_cache.py:439`) returns the cache to the uninitialized state.
- **V4 threshold is write-side only.** `persistent_cache_min_compile_time_secs` is checked when an
  entry is written (`jax/_src/compiler.py:840`), not on read or in the key. Differing thresholds across
  projects change which entries get written, not whether existing entries hit. Standardizing it is
  for predictability (which kernels end up on disk, and how many files), not for correctness.

## Design

### Module

`src/xtrax/compilation_cache.py` (top-level, so leaf packages such as `profiling` are not forced to
import `xtrax.config`; the campaign executor and CLI import it).

```python
APP = "xtrax"
KEY_DIR = "compilation_cache_dir"
KEY_MIN_SECS = "compilation_cache_min_compile_secs"
ENV_VAR = "XTRAX_COMPILATION_CACHE_DIR"
JAX_ENV_VAR = "JAX_COMPILATION_CACHE_DIR"
OFF = "off"   # explicit-off sentinel accepted by `explicit=`

@dataclass(frozen=True)
class CacheDecision:
    directory: Path | None      # None = caching off
    source: CacheSource         # Literal: "arg" | "env_xtrax" | "env_jax" | "pyproject" | "user_config" | "off"
    source_path: Path | None    # file that decided (pyproject / user config), else None
    min_compile_secs: float
    min_compile_secs_source: str

def resolve_compilation_cache(explicit: str | Path | None = None, *, start=None) -> CacheDecision
def enable_compilation_cache(explicit=None, *, start=None, on_conflict="raise") -> CacheDecision
def apply_decision(decision: CacheDecision, *, on_conflict="raise") -> CacheDecision  # no re-resolution; used by workers
```

`source` values are a `Literal[...]` type, not free strings; the file path for the pyproject and
user-config layers goes in a separate `source_path: Path | None` field.

### Resolution order (first match wins)

1. `explicit` argument. `None` = not given. `""`, `"none"`, or `OFF` = off. A relative path resolves
   against cwd (it came from the caller, as in `profile_dir`).
2. `XTRAX_COMPILATION_CACHE_DIR` if **set at all**: empty/`none` = off (stops the search).
3. `JAX_COMPILATION_CACHE_DIR` if set at all: same. Kept as a layer because JAX already honours it
   (V-env); ignoring it would make xtrax and JAX disagree.
4. `compilation_cache_dir` under `[tool.xtrax]` in the nearest `pyproject.toml`.
5. `compilation_cache_dir` in `${XDG_CONFIG_HOME:-~/.config}/xtrax/config.toml`.
6. Off.

Env beats pyproject (user decision, 261119): env is the per-job override, pyproject the project
default. Layers 2-3 are handled by hand, **not** by `resolve_layered`'s env layer, because
`resolve_layered` treats an env `none` as "skip this layer and keep searching"
(`src/xtrax/config.py:188-191`), which would let `[tool.xtrax]` re-enable a cache the job switched
off. Layers 4-5 call `resolve_layered(APP, KEY_DIR, env_var=None, start=start, parse=_parse_dir)`
(same split as `src/xtrax/cli/profile_dir.py:80-87`).

**Off sentinels, every layer:** `""`, `none`, `off` (any case, stripped) mean off, via one
`_parse_dir` used for all layers (arg, both env vars, pyproject, user config). `explicit=Path("")`
is off, not cwd.

**Relative paths:** `XTRAX_COMPILATION_CACHE_DIR` and both config files must be absolute or
`~`-prefixed (`$VARS` expanded); a relative value raises `ValueError` naming the variable or file.
`JAX_COMPILATION_CACHE_DIR` keeps JAX's own semantics for compatibility (the current executor passes
relative values through, `local.py:100-103`): a relative value resolves against cwd at resolution
time and logs a WARNING. A malformed TOML raises (inherited from `resolve_layered`). No default
directory exists anywhere in library code.

`min_compile_secs`: explicit kwarg > `XTRAX_COMPILATION_CACHE_MIN_COMPILE_SECS` > `[tool.xtrax]` >
user config > `1.0` (the executor's current value). Resolved by `resolve_layered` directly (a number
has no `none`-means-off ambiguity).

### Applying the decision (`enable_compilation_cache`)

- **On:** `mkdir -p`; set `jax_compilation_cache_dir` and
  `jax_persistent_cache_min_compile_time_secs`.
- **Off is an action, but only pre-init:** if JAX's flag is non-empty, set it to `None`. This fixes
  the `./none` defect and makes "off from pyproject" win over an inherited `JAX_COMPILATION_CACHE_DIR`.
  Once the cache is initialized, clearing the flag does **not** stop writes
  (`compilation_cache.py:245-246` returns early and the built `_cache` stays), so on→off after init is
  a conflict like any other (below).
- **Initialization state:** read `getattr(jax._src.compilation_cache, "_cache_initialized", None)`.
  The public `is_initialized()` is deprecated and only reports `enable_compilation_cache`
  (`compilation_cache.py:428-436`), so it is unusable. If the attribute is missing (renamed in a future
  jax), log a WARNING and treat the state as "unknown, assume initialised". This catches a cache
  initialised by JAX's import-time env var or by a consumer's direct `jax.config.update`, which an
  xtrax-only bookkeeping heuristic would miss. A test pins the attribute's presence at the floor and
  ceiling of the supported jax range (`pyproject.toml:7`: `>=0.10.2,<0.12`).
- **Conflict (V2):** compare the new decision with the current JAX flag value, treating an empty flag
  as off. A conflict exists only when the cache is initialised and the two differ, which covers
  dir→other dir and dir→off. (off→dir never conflicts, because off never initialises.) Known limit: if
  something changed the flag after init, the flag no longer names the live directory (V2), so the
  error message says "flag value", not "live directory". When a conflict exists,
  `on_conflict="raise"` (default) raises `RuntimeError` naming both values, and `on_conflict="reset"`
  calls the public `reset_cache` (V3) and then applies, including applying off. Same decision:
  idempotent and silent.
- **Filesystem:** `mkdir(parents=True, exist_ok=True)` (concurrent workers race safely). If the path
  exists as a non-directory or is unwritable, raise `ValueError` naming the path and source; never
  silently degrade to off.
- **Multi-process / multi-host:** JAX writes cache entries only from `process_id == 0`
  (`compiler.py:835-838`); every process reads. In a multi-host job every host must resolve the same
  shared directory, or only host 0's kernels are reusable. The resolver does not attempt to enforce
  this; the docs page states it.
- Logs one INFO line on first application (DEBUG on idempotent repeats): `compilation cache:
  <dir|off> (source: <source>)`. Returns the decision so callers can write it into run records /
  bathos sidecars.

### Worker processes

Resolve **once in the parent**, then pass the **whole resolved `CacheDecision`** (dir or off, plus
`min_compile_secs`) to workers. It is a frozen, picklable dataclass. Today `cache_dir: str | None`
cannot tell "not given" from "off", so a parent that resolved off from `[tool.xtrax]` hands workers
`None` and each worker falls back to the inherited `JAX_COMPILATION_CACHE_DIR`. Passing only the
directory would also let `min_compile_secs` re-resolve from the worker's own cwd and env.

- `LocalPoolExecutor.cache_dir` accepts `str | Literal["off"] | None`. `run()` resolves it to a
  `CacheDecision` before spawning.
- New `apply_decision(decision)` applies a pre-resolved decision without re-resolving. It clears the
  worker's import-time JAX flag on off.
- The decision travels through `initargs` (`local.py:1019-1020`) **and** the per-task path
  (`_PoolTaskArgs` / the imap wrapper, `local.py:690-706`), because `_pool_worker_task` re-runs worker
  init on every task (`local.py:654`). The per-task call must be idempotent and silent (DEBUG). It
  uses `on_conflict="raise"`, but the same decision never conflicts.
- Workers are spawned (`local.py:1015`), so they re-import jax and re-read env (A9).
- Errors in the initializer surface as `BrokenProcessPool`. The parent therefore validates the
  directory (mkdir and writability) before spawning, so a bad path fails in the parent with a named
  error.
- The parent's `CacheDecision.source` is recorded once in the run summary.

### xtrax call sites

| Site | Change |
|---|---|
| `campaign/executors/local.py` | delete `_resolve_compilation_cache_dir`; use the module as above |
| `cli/sweep_verb.py:162` | call `enable_compilation_cache()`. **Behaviour change, deliberate:** the verb goes from always-on (`.xtrax/jax_cache`) to off unless configured. The verb is deprecated (removal 0.6.0); its `CHANGELOG` entry says how to keep the old behaviour. |
| `xtrax campaign run` | `--compilation-cache PATH\|off`, passed as `cache_dir` to the executor (resolved in `run()`) |
| `xtrax run` (`cli/run_verb.py:run_run`) | `--compilation-cache PATH\|off`; `run_run` calls `enable_compilation_cache(explicit=…)` before building the engine |
| introspection | no new verb (none exists in `cli/registry.py:44-57`). Both CLIs print the INFO decision line, and the decision (`dir`, `source`, `min_compile_secs`) is written into the run summary / record so it is visible after the fact |
| library entry points (`engine`, `intent`) | **not** auto-enabled. Libraries do not mutate global JAX config on import; consumers call `enable_compilation_cache()` once at their entry point. |

### Consumers (follow-up, not this PR)

One debt per consumer via the `xtrax-consumer-adoption` flow: replace each hand-rolled block with
`xtrax.compilation_cache.enable_compilation_cache(explicit=…)`. aminx passes `io.cache_path`;
alynxr keeps `ALYNXR_COMPILATION_CACHE_DIR` and its registry field as feeders for `explicit`.

### Storage

The cache writes many small files. The machine-level config on Engaging must point at
`/orcd/compute/so3/002/...`, not scratch (1,000,000-file cap). The docs page says so. A size cap is
out of scope here: callers who need one set JAX's own `jax_compilation_cache_max_size` directly. A
later revision can add a layered `compilation_cache_max_size_bytes` key with its own source field and
tests. Entries are keyed by JAX on jax/jaxlib version, backend and
device kind, so one directory shared across projects on one machine is safe. Sharing a directory
across machines with different GPUs is also safe but gives no reuse.

## Assumptions

| # | Assumption | Tag |
|---|---|---|
| A1 | JAX reads `JAX_COMPILATION_CACHE_DIR` at import | verified (spike; `config.py:1626`) |
| A2 | `none` in that env var makes JAX write `./none` | verified (spike) |
| A3 | Setting the flag to `None` before first compile prevents A2 | verified (spike) |
| A4 | Init is lazy and once-only; later dir change ignored | verified (spike; `compilation_cache.py:241-258`) |
| A5 | `reset_cache` is public in jax 0.11.1 | verified (`jax/experimental/compilation_cache/compilation_cache.py:15-17`) |
| A6 | Threshold only gates writes | verified (`compiler.py:840`) |
| A7 | `resolve_layered` env `none` skips layer | verified (`src/xtrax/config.py:188-191`) |
| A8 | `_cache_initialized` stays a stable private name across supported jax versions | **unverified** at the 0.10.2 floor; T9 pins it; missing ⇒ WARNING + assume initialised |
| A9 | Spawned pool workers re-import jax and re-read env | verified (`local.py:1015`, spawn context) |
| A11 | Clearing the flag after init does not stop writes | verified by code (`compilation_cache.py:245-246`); T6b tests it |
| A10 | jax lower bound of xtrax exposes A5's module | **unverified**; check `pyproject.toml` floor during implementation |

## Tests (write red first where marked)

- T1 resolution order: each layer wins over the ones below; source string correct; tmp `pyproject.toml`
  and `XDG_CONFIG_HOME`.
- T2 env `none`/empty/`off` in either env var stops the search even when `[tool.xtrax]` sets a dir;
  `none`/`off` in pyproject and user config mean off; `explicit=Path("")` is off.
- T3 relative path in `XTRAX_…` or a config file raises naming the source; relative
  `JAX_COMPILATION_CACHE_DIR` resolves against cwd with a WARNING; malformed TOML raises; path that is
  a file raises.
- **T4 (red first)** subprocess with `JAX_COMPILATION_CACHE_DIR=none`, call `enable_compilation_cache()`,
  compile (assert the jitted result, so a compile demonstrably ran); assert JAX flag is `None` and no
  `./none` under cwd. Fails on today's executor path.
- T5 subprocess: on, compile, assert entries appear in the resolved dir (min secs 0, min entry size -1).
- T6 subprocess: enable A, compile, enable B → `RuntimeError`; with `on_conflict="reset"`, entries land in B.
- T6b subprocess: enable A, compile, enable off → `RuntimeError`; with `on_conflict="reset"`, a further
  compile adds no entries to A.
- T7 pool executor, all with `min_compile_secs=0` and min entry size -1: (a) positive control: parent
  resolves a dir → entries appear; (b) parent resolves off from `[tool.xtrax]` while env
  `JAX_COMPILATION_CACHE_DIR` points at a real dir → zero entries there; (c) the per-task re-init logs
  at most one INFO line per worker.
- T8 sweep verb no longer creates `.xtrax/jax_cache` when unconfigured, and `CHANGELOG.md` carries
  the behaviour-change line.
- T9 `jax._src.compilation_cache` has `_cache_initialized` (runs in CI against the resolved jax; the
  floor is checked once in the implementation PR via `uv run --with jax==0.10.2`).
- Import-boundary audits (`tests/audit/test_no_devtools_imports.py`,
  `tests/audit/test_compiler_boundary_gate.py`, `tests/distribution/test_declared_vs_imported_deps.py`)
  run narrowly after the module lands.

Subprocess tests are CPU-only, narrow, and run with `-k` locally per `local-compute-limits`.

## Out of scope

Auto-enabling on `import xtrax`; migrating consumers (separate debts); XLA autotune caches
(`jax_persistent_cache_enable_xla_caches`) beyond passing JAX's default through.
