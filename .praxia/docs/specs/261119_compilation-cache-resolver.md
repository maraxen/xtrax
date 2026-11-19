---
title: xtrax compilation-cache resolver
description: One layered resolver for the JAX persistent compilation cache dir, used by every xtrax entry point and consumer
status: draft
task_id: 261119_compilation-cache-resolver
date: '261119'
backlog_ids: ''
adversarial_review: ''
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
    source: str                 # "arg" | "env:XTRAX_COMPILATION_CACHE_DIR" | "env:JAX_COMPILATION_CACHE_DIR"
                                # | "pyproject:<path>" | "user_config:<path>" | "off"
    min_compile_secs: float
    min_compile_secs_source: str

def resolve_compilation_cache(explicit: str | Path | None = None, *, start=None) -> CacheDecision
def enable_compilation_cache(explicit=None, *, start=None, on_conflict="raise") -> CacheDecision
```

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

Paths from env or config files must be absolute or `~`-prefixed; `$VARS` are expanded; a relative
path raises `ValueError` naming the variable or file. A malformed TOML raises (inherited from
`resolve_layered`). No default directory exists anywhere in library code.

`min_compile_secs`: explicit kwarg > `XTRAX_COMPILATION_CACHE_MIN_COMPILE_SECS` > `[tool.xtrax]` >
user config > `1.0` (the executor's current value). Resolved by `resolve_layered` directly (a number
has no `none`-means-off ambiguity).

### Applying the decision (`enable_compilation_cache`)

- **On:** `mkdir -p`; set `jax_compilation_cache_dir` and
  `jax_persistent_cache_min_compile_time_secs`.
- **Off is an action:** if JAX's flag is non-empty, set it to `None`. This fixes the `./none` defect
  and makes "off from pyproject" win over an inherited `JAX_COMPILATION_CACHE_DIR`.
- **Conflict (V2):** xtrax tracks the directory it last applied in a module global. If the cache is
  already initialized (`jax._src.compilation_cache._cache_initialized`, or, if private access is
  rejected in review, "xtrax applied a directory and a compile has since run") and the decision
  differs, `on_conflict="raise"` (default) raises `RuntimeError` naming both paths;
  `on_conflict="reset"` calls the public `reset_cache` (V3) then applies. Idempotent when the
  decision matches.
- Logs one INFO line: `compilation cache: <dir|off> (source: <source>)`. Returns the decision so
  callers can write it into run records / bathos sidecars.

### Worker processes

Resolve **once in the parent**, then pass the resolved directory **or the `OFF` sentinel** to
workers as `explicit`. Today `cache_dir: str | None` cannot tell "not given" from "off", so a parent
that resolved off from `[tool.xtrax]` hands workers `None` and each worker falls back to the
inherited `JAX_COMPILATION_CACHE_DIR`. Change `LocalPoolExecutor.cache_dir` to accept
`str | Literal["off"] | None` and have `run()` resolve `None` to a concrete value before spawning;
`_initialize_worker_process` calls `enable_compilation_cache(explicit=<resolved>)`, which also clears
the worker's import-time JAX flag on off. The parent's `CacheDecision.source` is recorded once in
the run summary.

### xtrax call sites

| Site | Change |
|---|---|
| `campaign/executors/local.py` | delete `_resolve_compilation_cache_dir`; use the module as above |
| `cli/sweep_verb.py:162` | call `enable_compilation_cache()`. **Behaviour change, deliberate:** the verb goes from always-on (`.xtrax/jax_cache`) to off unless configured. The verb is deprecated (removal 0.6.0); its `CHANGELOG` entry says how to keep the old behaviour. |
| `xtrax campaign run`, `xtrax run` CLIs | `--compilation-cache PATH|off`, passed as `explicit` |
| `xtrax config show` (or the nearest existing introspection verb) | print the decision and its source |
| library entry points (`engine`, `intent`) | **not** auto-enabled. Libraries do not mutate global JAX config on import; consumers call `enable_compilation_cache()` once at their entry point. |

### Consumers (follow-up, not this PR)

One debt per consumer via the `xtrax-consumer-adoption` flow: replace each hand-rolled block with
`xtrax.compilation_cache.enable_compilation_cache(explicit=…)`. aminx passes `io.cache_path`;
alynxr keeps `ALYNXR_COMPILATION_CACHE_DIR` and its registry field as feeders for `explicit`.

### Storage

The cache writes many small files. The machine-level config on Engaging must point at
`/orcd/compute/so3/002/...`, not scratch (1,000,000-file cap). The docs page says so; size limits use
JAX's own `jax_compilation_cache_max_size` (exposed as an optional `compilation_cache_max_size_bytes`
key resolved like `min_compile_secs`). Entries are keyed by JAX on jax/jaxlib version, backend and
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
| A8 | `_cache_initialized` stays a stable private name across supported jax versions | **unverified**; fallback heuristic specified above |
| A9 | Spawned pool workers re-import jax and re-read env | **unverified for this executor**; T7 covers it |
| A10 | jax lower bound of xtrax exposes A5's module | **unverified**; check `pyproject.toml` floor during implementation |

## Tests (write red first where marked)

- T1 resolution order: each layer wins over the ones below; source string correct; tmp `pyproject.toml`
  and `XDG_CONFIG_HOME`.
- T2 env `none`/empty in either env var stops the search even when `[tool.xtrax]` sets a dir.
- T3 relative path in env/config raises naming the source; malformed TOML raises.
- **T4 (red first)** subprocess with `JAX_COMPILATION_CACHE_DIR=none`, call `enable_compilation_cache()`,
  compile; assert JAX flag is `None` and no `./none` under cwd. Fails on today's executor path.
- T5 subprocess: on, compile, assert entries appear in the resolved dir (min secs 0, min entry size -1).
- T6 subprocess: enable A, compile, enable B → `RuntimeError`; with `on_conflict="reset"`, entries land in B.
- T7 pool executor, parent resolves off from `[tool.xtrax]` while env `JAX_COMPILATION_CACHE_DIR` is a
  real dir → workers write nothing there.
- T8 sweep verb no longer creates `.xtrax/jax_cache` when unconfigured.

Subprocess tests are CPU-only, narrow, and run with `-k` locally per `local-compute-limits`.

## Out of scope

Auto-enabling on `import xtrax`; migrating consumers (separate debts); XLA autotune caches
(`jax_persistent_cache_enable_xla_caches`) beyond passing JAX's default through.
