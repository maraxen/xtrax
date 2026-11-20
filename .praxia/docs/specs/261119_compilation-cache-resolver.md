---
title: xtrax compilation-cache resolver
description: One layered resolver for the JAX persistent compilation cache dir, used by every xtrax entry point and consumer
status: implemented
task_id: 261119_compilation-cache-resolver
date: '261119'
backlog_ids: ''
adversarial_review: 'R1 spec-challenger (claude sonnet) 261119 NEEDS_WORK -> B1-B3, S1-S7, N1 folded in; S5 descoped (max size). R2 cursor headless reviewer via surface_dispatch 74de3d1a (admitted, containment agent_sandbox) on rev 3 NEEDS_WORK -> B1 (local-seq + slurm array never apply a decision), S1-S7, N1 all verified against code and folded in as rev 4; rev 4 not yet re-reviewed'
---
# xtrax compilation-cache resolver

> **Implemented in `xtrax.jax_cache` (PR #278), 261119.** This spec was first built as a
> separate `xtrax.compilation_cache` module (branch `wt/261119-compilation-cache-spec`,
> `834e6f7`). PR #278 independently added `xtrax.jax_cache`, and the user chose to port this
> design onto it. Name mapping:
>
> | This spec | `xtrax.jax_cache` |
> |---|---|
> | `XTRAX_COMPILATION_CACHE_DIR` / `compilation_cache_dir` | `XTRAX_JAX_CACHE_DIR` / `jax_compilation_cache_dir` |
> | `XTRAX_COMPILATION_CACHE` / `compilation_cache` (policy) | `XTRAX_JAX_CACHE` / `jax_compilation_cache` |
> | `XTRAX_COMPILATION_CACHE_MIN_COMPILE_SECS` | `XTRAX_JAX_CACHE_MIN_COMPILE_SECS` (`jax_cache_min_compile_time_secs`) |
> | `CacheDecision` / `apply_decision` / `enable_compilation_cache` | `CacheSetting` / `apply` / `configure` |
> | `resolve_policy` | `policy_source` |
>
> **Differences kept from #278:**
> - Directories get a `jaxlib-<version>` namespace.
> - The write threshold stays at JAX's default unless configured.
> - A relative path in the user config is relative to that file.
> - An empty `JAX_COMPILATION_CACHE_DIR` counts as unset.
> - **A directory switch after the cache is in use warns and resets, instead of raising.**
>   That was #278's choice, kept to keep the port small. It replaces this spec's
>   `on_conflict="raise"` default.
>
> **Changes made to #278:**
> - The policy axis.
> - A pyproject directory is refused, and xtrax's own `pyproject.toml` no longer sets one.
> - `none`/`off` in `JAX_COMPILATION_CACHE_DIR` disables.
> - A configured off clears JAX's flag, resetting the cache first if it is in use.
> - The executor, Slurm and CLI wiring.

## Goal

Any xtrax project, and any consumer of xtrax, uses the JAX persistent compilation cache the same
way: one function decides the directory from layered configuration, applies it to JAX, and reports
which layer decided (for both where and whether).

- **A machine opts in** with one line of `~/.config/xtrax/config.toml`, or with its myxcel profile
  `[env]` for jobs myxcel submits.
- **A project opts out** with one line of `pyproject.toml` (`compilation_cache = "off"`).
- **A run overrides either** with one env var or flag.
- "Unconfigured" means no placement on that machine, so the cache is off.

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

### Separation of concerns (user-endorsed 261119)

Every setting is classified by who owns it before it gets a home:

| Kind | Example | Same on every machine? | Secret? | Home |
|---|---|---|---|---|
| Project policy | cache on/off, compile threshold, presets, uv venv mode | yes | no | committed: `[tool.xtrax]` in `pyproject.toml`, `myxcel.toml` |
| Machine fact | `jax_cache` path, flash vs scratch tier, `remote_home` | no (shared across projects) | no | per host, outside any repo: the myxcel profile (`[uv]`, proposed `[env]`) for remotes myxcel drives; `~/.config/<tool>/config.toml` for the laptop and for runs outside myxcel |
| Node-conditional flag | `XLA_FLAGS=--xla_gpu_shard_autotuning=false` on Blackwell nodes | no (varies per node in one job's partition) | no | myxcel's existing sbatch guard (`myxcel/src/myxcel/slurm.py:727-733`), which tests the node at run time; **not** profile `[env]`, which refuses `XLA_FLAGS` (`slurm.py:172`) |
| Per-run override | `--compilation-cache off`, `XTRAX_COMPILATION_CACHE=on` | n/a | no | CLI flag or env var, recorded in the run record |
| Secret | API tokens | n/a | yes | gitignored `.env` or keyring; never a myxcel profile `[env]` (exports are written into sbatch scripts on disk) |
| Shell convenience | `direnv` loading `.env` | n/a | maybe | `.envrc`, local interactive use only; not a source of truth |

Rules that follow:

- **Library code reads only:** explicit > env > `[tool.<pkg>]` (policy) > `~/.config/<pkg>` (machine).
  It never calls `load_dotenv`, because loading changes `os.environ` for the whole process behind the
  caller's back. Only entry scripts may load a `.env`.
- **`.env` is not for machine facts.** It is per checkout, so each worktree needs its own copy. It
  has no layering and no source reporting. Batch jobs do not load it.
- **myxcel pushes `.env` today.** No push exclude covers `.env` or `.envrc` (checked in both profiles
  and in `myxcel/src/myxcel/rsync.py`), so a laptop `.env` reaches the cluster.
- **myxcel's role:** it carries machine facts for remotes it drives into job env and records
  `MYXCEL_*_SOURCE`. Project policy and secrets stay out of it.
- **Why policy is a separate axis:** once myxcel has exported a value into env, a library cannot tell
  a machine default from a deliberate per-job override. A committed project "off" therefore has to
  resolve on its own axis to keep winning.

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

**Explicit argument across the axes:** `explicit=<path>` is both a placement and a policy-on. It
overrides a pyproject `compilation_cache = "off"` for that call. `explicit=OFF` is a policy-off and
wins over everything. T1b locks both cases.

**What "project off" guarantees, and its limit.** A project that sets `compilation_cache = "off"`
stays off in **every process that applies a decision before its first compile**. That covers every
xtrax entry point listed under "Every compiling process applies a decision", including under a
myxcel-injected `JAX_COMPILATION_CACHE_DIR`. JAX reads that variable at import
(`jax/_src/config.py:906`), so "off" must actively clear the flag (required below).

A process that never calls xtrax cannot be held to the project's policy: a plain JAX script, or
alynxr before it adopts xtrax. Under an injected `JAX_COMPILATION_CACHE_DIR` it uses the cache. Such
consumers opt out through their own env: a preset `env` setting `JAX_COMPILATION_CACHE_DIR=""`, or an
`XTRAX_COMPILATION_CACHE=off` that they honour after adoption. Env still beats pyproject for one run
via `XTRAX_COMPILATION_CACHE=on`.

**Parsers, one per axis:**

- **Placement values** go through `_parse_dir`. `""`, `none` and `off` (any case, stripped) mean off;
  anything else is a path. It applies to the explicit argument, both directory env vars and the user
  config. `explicit=Path("")` raises `ValueError` (implementation finding: `Path("") == Path(".")`,
  so the two cannot be told apart; pass `"off"` or an explicit path).

**Unconfigured is a no-op (implementation finding).** When neither axis was configured anywhere
(placement source `none` and policy source `default`), `apply_decision` leaves JAX's settings
alone. Otherwise a consumer that set JAX's cache itself, such as aminx from `io.cache_path`, would
have it cleared, or hit a conflict error, when it later ran an xtrax executor. A **configured** off
is still applied actively; `JAX_COMPILATION_CACHE_DIR=none` counts as configured, which keeps the
`./none` fix. `CacheDecision` also carries `placement`, the resolved placement before policy is
applied, so records can show an injected directory that policy held off.
- **Policy values** go through `_parse_policy`, case-insensitive:
  - `on` → on.
  - `off`, `none` and `""` → off.
  - Any other token raises `ValueError` naming the variable or file.

  It applies to `XTRAX_COMPILATION_CACHE`, pyproject `compilation_cache` and user-config
  `compilation_cache`.
- A pyproject `compilation_cache_dir` is never parsed. It is refused, as stated above.

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
- Logs one INFO line on first application (DEBUG on idempotent repeats), naming **both axes**:
  `compilation cache: <dir|off> (placement: <source>; policy: <on|off> from <policy_source>)`. A run
  kept off by project policy while a directory was injected must be distinguishable from a run with
  no placement at all. Returns the decision so callers can write it into run records / bathos
  sidecars.

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
- The parent's whole `CacheDecision` (directory, source, policy, policy_source, min_compile_secs) is
  recorded once in the run summary.

### Every compiling process applies a decision

The project-off guarantee holds only where a decision is applied before the first compile. Review
round 2 found two xtrax paths that apply none today:

- **`LocalSequentialExecutor`**, the default for `xtrax campaign run` (`src/xtrax/cli/campaign_verb.py:68`,
  instantiated at `:259`). It runs units in-process and never touches the cache. Fix: `run()` calls
  `enable_compilation_cache(cache_dir)` before the first unit. It takes the same
  `cache_dir: str | Literal["off"] | None` as the pool executor.
- **xtrax Slurm array tasks.** The array script is meant to work under `sbatch --export=NONE`
  (`src/xtrax/campaign/executors/slurm.py:210-231`) and exports only the variables it bakes
  (`slurm.py:478-491`). Each task runs `xtrax campaign run --row-index ... --executor local-seq`,
  which takes the `LocalSequentialExecutor` path above, so the fix there covers application.
  - **Correction from implementation:** the script bakes the generating interpreter's absolute path
    (`task_environment`, `slurm.py:221`). So it is generated on the cluster, not on the laptop as
    revision 4 said. The design is unchanged; the reason for not baking placement is the one below.
  - **Policy is baked only when a layer set it.** The generator resolves policy from the project's
    pyproject, or an `XTRAX_COMPILATION_CACHE` override, and exports `XTRAX_COMPILATION_CACHE=on|off`.
    The task's cwd may not see that pyproject. A default policy is not baked, so a node's user config
    can still decide.
  - **Placement is never baked from configuration.** The generating shell's env is dropped under
    `--export=NONE`. Baking its `JAX_COMPILATION_CACHE_DIR` would silently pin one shell's choice into
    every task, and the node's own config is the authority. Each task resolves placement on its node,
    from `~/.config/xtrax/config.toml`. **A20 is verified**: `HOME` is set under `--export=NONE`.
  - `SlurmArrayExecutor(compilation_cache=...)` (CLI `--compilation-cache`) takes an absolute cluster
    path, exported as `XTRAX_COMPILATION_CACHE_DIR` with policy on, or `off`, exported as
    `XTRAX_COMPILATION_CACHE=off`. A relative path is refused.
  - **myxcel profile `[env]` does not reach these tasks** when the script is submitted with
    `--export=NONE`, because myxcel's injection lives in myxcel's own generated script. Docs state
    this. The remote user config is the reliable placement for xtrax arrays.

### xtrax call sites

| Site | Change |
|---|---|
| `campaign/executors/local.py` | delete `_resolve_compilation_cache_dir`; pool executor as above; `LocalSequentialExecutor.run` calls `enable_compilation_cache` before the first unit |
| `campaign/executors/slurm.py` | bake `XTRAX_COMPILATION_CACHE=<resolved policy>` into the array script; optional `compilation_cache_dir` → `XTRAX_COMPILATION_CACHE_DIR`; never bake a resolved local placement |
| `cli/sweep_verb.py:162` | call `enable_compilation_cache()`. **Behaviour change, deliberate:** the verb stops writing `.xtrax/jax_cache` under cwd. It now uses the machine placement when one exists (policy default on), and is off when the machine has no placement. The verb is deprecated (removal 0.6.0); its `CHANGELOG` entry says this in those words and how to keep the old directory (`XTRAX_COMPILATION_CACHE_DIR=$PWD/.xtrax/jax_cache`). |
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
tests. Entries are keyed by JAX on the jaxlib version, the backend platform and the device kind
(verified: `jax/_src/cache_key.py:103,108,256,367`), so one directory shared across projects on one
machine is safe from mismatched entries. Sharing a directory across machines with different GPUs is
also safe but gives no reuse.

**Trust.** Cache entries are serialized executables that JAX deserializes and runs. Anyone who can
write the directory can plant code in every job that reads it. The shared `jax_cache` must therefore
be writable only by you (mode `0700`, or `0750` for a group you would run code from). It must not sit
under a setgid group-shared tree such as `so3_shared/` without checking the inherited group.
`enable_compilation_cache` logs a WARNING when the resolved directory is group- or world-writable. It
does not refuse, because some labs share caches on purpose.

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

- **Precedence, lowest to highest (revised in review round 2):** project `[env_catalog]` < profile
  `[env]` < preset `env` < invocation overrides.
  - The machine fact beats the committed catalog for the same name. A committed `myxcel.toml` is
    pushed to every remote, so a catalog path written on the laptop must not override the remote's
    own placement. This is the same "committed files never hold machine paths" rule as `[tool.xtrax]`.
  - Presets and invocation overrides stay above it, so a job can still opt out (for example
    `JAX_COMPILATION_CACHE_DIR=""`).
  - A name present in both the catalog and profile `[env]` logs a submit-time warning naming both
    values.
- **Names owned elsewhere are refused** at profile load with `ConfigError`:
  - everything uv placement writes: `UV_CACHE_DIR`, `UV_LINK_MODE`, `UV_PROJECT_ENVIRONMENT`,
    `MYXCEL_UV_READY`, `MYXCEL_UV_SOURCE` (`uv_placement.py`, and the scratch default at
    `slurm.py:776-779`);
  - the node-conditional reserved set, currently `XLA_FLAGS` (`slurm.py:172`). Node-conditional
    flags stay with the sbatch guard;
  - every other `MYXCEL_*` name.
- **Secret-looking names are refused** at profile load. A name refuses when, uppercased, it equals or
  ends with `TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL`, `CREDENTIALS`, or `_KEY`.
  - Example: `HF_TOKEN` is refused, while `HF_HOME` is allowed.
  - Reason: exported values are written into sbatch scripts on shared disk. The refusal message
    points at the cluster's own keyring or env.
- **Values:**
  - Strings only.
  - A leading `~` expands through `remote_home`, the same as `uv_placement._expand_remote`; it raises
    if `remote_home` is unset.
  - Values are exported `shlex`-quoted. There is no `$VAR` expansion, so a profile cannot smuggle in
    shell code.
- **Provenance:**
  - `MYXCEL_ENV_SOURCE=NAME=profile|project|preset|invocation,...` lists every name that came from or
    competed with profile `[env]`, the same shape as `MYXCEL_UV_SOURCE`. Invocation overrides get
    their own label.
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
| A17 | Cache key covers jaxlib version, backend platform, device kind | verified (`jax/_src/cache_key.py:103,108,256,367`) |
| A18 | `xtrax campaign run` defaults to `LocalSequentialExecutor`, which applies no cache decision | verified (`src/xtrax/cli/campaign_verb.py:68,259`) |
| A19 | xtrax Slurm array scripts target `--export=NONE` and export only baked variables | verified (`src/xtrax/campaign/executors/slurm.py:210-231,478-491`) |
| A20 | Under `--export=NONE`, a task still has `HOME` so `~/.config/xtrax/config.toml` resolves | **verified 261119** (Engaging `mit_quicktest` array job 29675457, tasks 0 and 1 on node1600: `SLURM_EXPORT_ENV=NONE`, `HOME=/home/maarxaru`, `XDG_CONFIG_HOME` unset, `~/.config` exists, `JAX_COMPILATION_CACHE_DIR` not inherited). Throwaway env probe, no finding cited beyond this row |

## Tests (write red first where marked)

- T1 placement order and policy order, each independently: every layer wins over the ones below;
  source strings correct; tmp `pyproject.toml` and `XDG_CONFIG_HOME`.
- T2 env `none`/empty/`off` in either placement env var stops the placement search even when the user
  config sets a dir; `none`/`off` in the user config mean off; `explicit=Path("")` raises.
- T1b cross-axis: `explicit=<path>` with pyproject `compilation_cache = "off"` → on at that path;
  `explicit=OFF` with every placement layer set → off.
- T1c `_parse_policy`: `ON`/`Off`/`none`/`""` parse; `yes` / `1` / a path raise `ValueError` naming
  the source.
- T10 a `compilation_cache_dir` key under `[tool.xtrax]` raises `ValueError` naming the file.
- T11 policy beats placement: `[tool.xtrax] compilation_cache = "off"` with `JAX_COMPILATION_CACHE_DIR`
  set → decision off, JAX flag cleared, no entries (subprocess); `XTRAX_COMPILATION_CACHE=on` then
  re-enables it for that run.
- T3 relative path in `XTRAX_…` or a config file raises naming the source; relative
  `JAX_COMPILATION_CACHE_DIR` resolves against cwd with a WARNING; malformed TOML raises; path that is
  a file raises.
- **T4 (red first)** subprocess with `JAX_COMPILATION_CACHE_DIR=none`, call `enable_compilation_cache()`,
  compile (assert the jitted result, so a compile demonstrably ran); assert JAX flag is `None` and no
  `./none` under cwd. Fails on today's executor path. Parametrized over `none`, `off` and `OFF`:
  JAX treats each as a truthy relative path (`compilation_cache.py:250`), so each must be cleared.
- T4b subprocess: `XTRAX_COMPILATION_CACHE_DIR=""` (placement off at layer 2) **and**
  `JAX_COMPILATION_CACHE_DIR=<real dir>`. After enabling, the decision directory is `None`, **JAX's
  flag is `None`**, and a compile writes no entries there. This catches an implementation that returns
  off but leaves JAX's import-time flag set.
- T5 subprocess: on, compile, assert entries appear in the resolved dir (min secs 0, min entry size -1).
- T6 subprocess: enable A, compile, enable B → `RuntimeError`; with `on_conflict="reset"`, entries land in B.
- T6b subprocess: enable A, compile, enable off → `RuntimeError`; with `on_conflict="reset"`, a further
  compile adds no entries to A.
- T7 pool executor, all with `min_compile_secs=0` and min entry size -1: (a) positive control: parent
  resolves a dir → entries appear; (b) parent resolves off from `[tool.xtrax]` while env
  `JAX_COMPILATION_CACHE_DIR` points at a real dir → zero entries there; (c) the per-task re-init logs
  at most one INFO line per worker.
- T7b `LocalSequentialExecutor`, same settings: pyproject off plus `JAX_COMPILATION_CACHE_DIR=<real dir>`
  → zero entries; a positive control with policy on → entries appear. Red on today's code, which never
  applies a decision on this path.
- T7c Slurm array script: the generated script exports `XTRAX_COMPILATION_CACHE=<policy resolved at
  generation>`, and exports **no** placement unless `compilation_cache_dir` was passed. Assert that
  with a user-config placement set on the generating host, that path does not appear in the script.
- T8 sweep verb, with `XTRAX_COMPILATION_CACHE_DIR`, `JAX_COMPILATION_CACHE_DIR` and `XTRAX_COMPILATION_CACHE`
  unset and `XDG_CONFIG_HOME` pointing at an empty tmp dir: no `.xtrax/jax_cache` is created, JAX's
  flag is `None`, and the run summary records placement `none`. With a user-config placement, entries
  land there instead. `CHANGELOG.md` carries the behaviour-change line.
- T12 decision record: a policy-off run under an injected placement logs and stores `policy: off`
  with `policy_source: pyproject`, distinguishable from a no-placement run.
- T13 group- or world-writable resolved directory → one WARNING naming the mode; `0700` → none.
- T9 `jax._src.compilation_cache` has `_cache_initialized` (runs in CI against the resolved jax; the
  floor is checked once in the implementation PR via `uv run --with jax==0.10.2`).
- Import-boundary audits (`tests/audit/test_no_devtools_imports.py`,
  `tests/audit/test_compiler_boundary_gate.py`, `tests/distribution/test_declared_vs_imported_deps.py`)
  run narrowly after the module lands.

Subprocess tests are CPU-only, narrow, and run with `-k` locally per `local-compute-limits`.

## Out of scope

Auto-enabling on `import xtrax`; migrating consumers (separate debts); XLA autotune caches
(`jax_persistent_cache_enable_xla_caches`) beyond passing JAX's default through.
