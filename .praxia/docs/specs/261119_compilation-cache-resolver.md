---
title: xtrax compilation-cache resolver
description: One layered resolver for the JAX persistent compilation cache dir, used by every xtrax entry point and consumer
status: draft
task_id: 261119_compilation-cache-resolver
date: '261119'
backlog_ids: ''
adversarial_review: 'spec-challenger (sonnet) 261119 NEEDS_WORK -> B1-B3, S1-S7, N1 folded in; S5 resolved by descoping max size. Rev 3 (placement/policy split, myxcel [env] contract, shared jax_cache) not yet re-reviewed'
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
KEY_DIR = "compilation_cache_dir"          # placement: user config only
KEY_POLICY = "compilation_cache"           # policy: "on" | "off"; pyproject and user config
KEY_MIN_SECS = "compilation_cache_min_compile_secs"
ENV_VAR = "XTRAX_COMPILATION_CACHE_DIR"
POLICY_ENV_VAR = "XTRAX_COMPILATION_CACHE"
JAX_ENV_VAR = "JAX_COMPILATION_CACHE_DIR"
OFF = "off"   # explicit-off sentinel accepted by `explicit=`

@dataclass(frozen=True)
class CacheDecision:
    directory: Path | None      # None = caching off
    source: CacheSource         # placement layer: "arg" | "env_xtrax" | "env_jax" | "user_config" | "none"
    source_path: Path | None    # user-config file that decided, else None
    policy: Literal["on", "off"]
    policy_source: PolicySource # "arg" | "env_xtrax" | "pyproject" | "user_config" | "default"
    min_compile_secs: float
    min_compile_secs_source: str

def resolve_compilation_cache(explicit: str | Path | None = None, *, start=None) -> CacheDecision
def enable_compilation_cache(explicit=None, *, start=None, on_conflict="raise") -> CacheDecision
def apply_decision(decision: CacheDecision, *, on_conflict="raise") -> CacheDecision  # no re-resolution; used by workers
```

`source` values are a `Literal[...]` type, not free strings; the file path for the pyproject and
user-config layers goes in a separate `source_path: Path | None` field.

### Two axes: placement (where) and policy (whether)

User decision 261119: **a committed `pyproject.toml` never holds a machine path.** myxcel pushes
the project tree to remotes, so an absolute path written on the laptop names the wrong place on
Engaging. Placement is a fact about the machine; policy is a fact about the project. They resolve
independently, and the cache is on only when policy is `on` **and** a placement exists.

**Placement (first match wins):**

1. `explicit` argument. `None` = not given. `""`, `"none"`, or `OFF` = off (this is also a policy-off;
   see below). A relative path resolves against cwd (it came from the caller, as in `profile_dir`).
2. `XTRAX_COMPILATION_CACHE_DIR` if **set at all**: empty/`none`/`off` = off (stops the search).
3. `JAX_COMPILATION_CACHE_DIR` if set at all: same. Kept as a layer because JAX already honours it
   (V-env); this is the variable myxcel's profile `[env]` injects on remotes (see "myxcel contract").
4. `compilation_cache_dir` in `${XDG_CONFIG_HOME:-~/.config}/xtrax/config.toml` (one per machine:
   laptop, titanix, Engaging login node).
5. None.

A `compilation_cache_dir` key found under `[tool.xtrax]` in `pyproject.toml` is **refused** with a
`ValueError` that names the file and points at the user config / myxcel profile.

Layers 2-3 are handled by hand, **not** by `resolve_layered`'s env layer, because `resolve_layered`
treats an env `none` as "skip this layer and keep searching" (`src/xtrax/config.py:188-191`), which
would let a lower layer re-enable a cache the job switched off. Layer 4 reads the user config file
directly with `xtrax.config`'s TOML helpers (`_user_config_path`, `_read_toml`), not through
`resolve_layered`, whose pyproject walk would otherwise supply placement.

**Policy (first match wins):**

1. `explicit`: `OFF`/`none`/`""` → off; any path → on.
2. `XTRAX_COMPILATION_CACHE` = `on` | `off` (a job-level override in either direction).
3. `compilation_cache = "on" | "off"` under `[tool.xtrax]` in the nearest `pyproject.toml`
   (via `resolve_layered`, `env_var=None`).
4. `compilation_cache` in the user config.
5. Default `on` (so a machine placement alone turns the cache on for every project, which is the
   point of a shared machine cache).

Consequence: a project that sets `compilation_cache = "off"` stays off on every machine, including
under a myxcel-injected `JAX_COMPILATION_CACHE_DIR`. Since JAX reads that variable at import, "off"
must actively clear the flag (already required below). Env still beats pyproject for a single run
via `XTRAX_COMPILATION_CACHE=on`.

**Off sentinels, every layer:** `""`, `none`, `off` (any case, stripped) mean off, via one
`_parse_dir` used for all layers (arg, both env vars, pyproject, user config). `explicit=Path("")`
is off, not cwd.

**Relative paths:** `XTRAX_COMPILATION_CACHE_DIR` and the user config must be absolute or
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

**Layout (user decision 261119): one shared directory per machine, named `jax_cache`.**

| Machine | Directory | Set by |
|---|---|---|
| Engaging | `/orcd/compute/so3/002/maarxaru/jax_cache` (beside `uv-cache`) | myxcel profile `[env]` for jobs; `~/.config/xtrax/config.toml` on the login node for non-myxcel runs |
| titanix | a local-disk path in `titanix.toml` `[env]` | same split |
| laptop (WSL) | `~/.cache/jax_cache` | `~/.config/xtrax/config.toml` |

The basename must be exactly `jax_cache`: myxcel's disk triage classifies JIT caches by basename
(`myxcel/src/myxcel/disk_triage_worker.py:50-51` matches `jax_cache` / `.jax_cache`, not
`jax-cache`), so the shared cache shows up as `jit-cache`, which is regenerable and safe to delete.
Pruning is all-or-nothing (delete the directory) until a size cap lands.

**Shared-filesystem safety (A14, A16).** One directory shared by concurrent array tasks is safe to
*run*: entries are written in place without a lock, but a reader that sees a partly written entry
gets a warning and recompiles. The one lasting hazard is a task **preempted or killed mid-write**. It
leaves a truncated `*-cache` file, and `put` never replaces an existing key (`lru_cache.py:146-147`),
so that kernel warns and recompiles on every later run until someone deletes the file. The window is
the duration of one `write_bytes` call, so this is rare, but `pi_so3` / preemptible partitions make
it plausible. Mitigations, in order:

1. Document it: `Error reading persistent compilation cache entry` in a job log means "delete the
   named entry" (or the whole `jax_cache` directory; it is regenerable).
2. Follow-up (not this PR): an `xtrax` maintenance helper that tries to deserialize each entry and
   removes the ones that fail, run occasionally as a myxcel job. It is file-count heavy, so it runs
   in a job, never on the login node.
3. Not adopted: per-node or per-job subdirectories. They avoid the hazard but give up the cross-job
   reuse that is the reason for a shared cache.

### myxcel contract: a generic profile `[env]` table

**Current state (verified):** myxcel has no machine-level env. `Profile`
(`myxcel/src/myxcel/config.py:63-105`) carries typed tables such as `[uv]`, which
`uv_placement.py` turns into job exports. Per-project env lives in `myxcel.toml` `[env_catalog]`
(`project_config.py:288-300`) and is merged into the preset by `resolve_preset`
(`submit.py:121-131`). Preset env wins over computed placement (`slurm.py:760-790`).

**Proposal (lands in myxcel, own spec/PR):** a profile `[env]` table holding machine facts, exported
into every job script that profile generates.

```toml
# ~/.config/myxcel/engaging.toml
[env]
JAX_COMPILATION_CACHE_DIR = "/orcd/compute/so3/002/maarxaru/jax_cache"
TRITON_CACHE_DIR = "/orcd/compute/so3/002/maarxaru/triton_cache"
HF_HOME = "/orcd/compute/so3/002/maarxaru/hf"
```

- **Precedence, lowest to highest:** profile `[env]` < project `[env_catalog]` < preset `env` <
  invocation overrides. This follows the existing convention that the profile owns machine facts and
  the project owns policy.
- **One owner per name:** a name that a typed table also sets, such as `UV_CACHE_DIR` from `[uv]`,
  raises `ConfigError` at profile load.
- **Reserved names:** the reserved set, currently `XLA_FLAGS` (`slurm.py:172`), and every `MYXCEL_*`
  name raise `ConfigError`.
- **Values:**
  - Strings only.
  - A leading `~` expands through `remote_home`, the same as `uv_placement._expand_remote`; it raises
    if `remote_home` is unset.
  - Values are exported `shlex`-quoted. There is no `$VAR` expansion, so a profile cannot smuggle in
    shell code.
- **Provenance:**
  - `MYXCEL_ENV_SOURCE=NAME=profile|project|preset,...` lists every name that came from or competed
    with profile `[env]`, the same shape as `MYXCEL_UV_SOURCE`.
  - xtrax's `CacheDecision.source == "env_jax"` plus that variable together say which file decided.
- **Byte-identical when unset:** a profile without `[env]` produces exactly today's script, the same
  guarantee `[uv]` makes.
- **Login shells:** `myxcel env-catalog <remote> <project> --shell` also emits the profile `[env]`
  lines, so a remote `.bashrc` can source the same values. Runs outside myxcel on that host then
  agree with myxcel jobs. Alternatively, `~/.config/xtrax/config.toml` on the remote does this for
  xtrax only.
- **Colab:** out of scope. VMs are ephemeral and the cache would not outlive the session.

**Why generic and not `[jax]`:** the JAX cache is one of several machine-level relocations the
triage reference already lists (JAX, Triton, HF, torch hub, pip). `[uv]` stays a typed table because
it computes values (lock-keyed venvs, link mode); plain "this variable is this path on this machine"
facts need no code of their own.

**How xtrax consumes it:** `JAX_COMPILATION_CACHE_DIR` is placement layer 3, and JAX itself also
reads it, so non-xtrax consumers (alynxr, tev_design, plain scripts) get the same cache with no code.
A project opts out with `[tool.xtrax] compilation_cache = "off"`, which beats the injected path
because policy and placement are separate axes. (Non-xtrax consumers opt out by preset `env`
`JAX_COMPILATION_CACHE_DIR=""`.)

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
| A12 | myxcel has no machine-level env table today | verified (`myxcel/src/myxcel/config.py:63-105`) |
| A13 | myxcel disk triage recognises `jax_cache` by basename only | verified (`disk_triage_worker.py:40-54`) |
| A14 | JAX's file cache writes entries atomically | **false** (verified by code): `LRUCache.put` writes in place with `write_bytes`, no temp+rename, and takes no lock unless `jax_compilation_cache_max_size` is set (`lru_cache.py:121-161`; default `-1`, `config.py:1657-1658`). See "Shared-filesystem safety" |
| A15 | Treating `JAX_COMPILATION_CACHE_DIR=""` as off holds in JAX too (empty flag = no cache) | verified by code (`compilation_cache.py:250-251`: `if not path: return`) |
| A16 | A torn or corrupt entry makes JAX warn and recompile, not crash | verified by code (`compiler.py:808-818`; `jax_raise_persistent_cache_errors` default False, `config.py:1490-1492`) |

## Tests (write red first where marked)

- T1 placement order and policy order, each independently: every layer wins over the ones below;
  source strings correct; tmp `pyproject.toml` and `XDG_CONFIG_HOME`.
- T2 env `none`/empty/`off` in either placement env var stops the placement search even when the user
  config sets a dir; `none`/`off` in the user config mean off; `explicit=Path("")` is off.
- T10 a `compilation_cache_dir` key under `[tool.xtrax]` raises `ValueError` naming the file.
- T11 policy beats placement: `[tool.xtrax] compilation_cache = "off"` with `JAX_COMPILATION_CACHE_DIR`
  set → decision off, JAX flag cleared, no entries (subprocess); `XTRAX_COMPILATION_CACHE=on` then
  re-enables it for that run.
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
