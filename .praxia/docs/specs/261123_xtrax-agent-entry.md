---
title: 'xtrax.agent: one startup for consumer CLI and MCP entry points'
description: 'Shared core for debts #3553-#3555: a small xtrax helper that runs xtrax.jax_cache.configure() then cisternal.init() in one fixed order, then dispatches a cyclopts app or a FastMCP server; per-repo specs build on it.'
status: converged
task_id: 261123_xtrax-entrypoint-cli-mcp
date: '261123'
backlog_ids: ''
adversarial_review: ''
---
# xtrax.agent: one startup for consumer CLI and MCP entry points


task_id: `261123_xtrax-entrypoint-cli-mcp` · shared core for debts **#3553** (denxity), **#3554** (aminx-lm),
**#3555** (prolix). #3556 (alynxr) does not use it: alynxr's CLI/MCP are already cisternal-wired, and its
own process compiles no JAX. Its only compile site is the ColabFold probe, which imports only the standard
library under ColabFold's interpreter (`probes/colabfold_run.py:28-39`). Owner decision 2026-11-23: "small xtrax helper".

## 1. Problem

Consumers that expose a CLI and/or an MCP server on cisternal (`@cisternal.tool` + `cisternal.wire`)
hand-roll the process start, each differently (all read at current `origin/main` / integration branch):

| Project | Entry | Startup today |
|---|---|---|
| demistify | `demistify.cli:main` | `configure_jax_cache()` (`jax_cache.configure()` + log); `JaxCacheConfigError`/`JaxCacheDirError` → stderr `demistify: jax cache configuration error: …` + `SystemExit(2)`; then `cisternal.init()`; then `run_app(app, argv)` (cli.py:1024-1037; jax_cache_setup.py:18-35). `run_app` calls `app(tokens, exit_on_error=False, print_error=True)` and maps `CycloptsError` → exit 2 (cli_support.py:226-257). A duplicate bare `configure()` from #113 is being removed in demistify#114. |
| aminx | `aminx.cli:main` | `jax_cache.configure()` → `run_cli(app)` (typer; telemetry inside run_cli) |
| aminx | `aminx.agent.mcp:main` | `require_agent_extra()` → `cisternal.init()` → `jax_cache.configure()` → `build_server().run()`; `build_server` installs `CisternalMiddleware` |
| aminx-ensemble | `cli:main` / `mcp:main` | `jax_cache.configure()` → `app(argv, result_action="return_value")` / `server.run()`; no `cisternal.init` |

The most deliberate precedent (demistify #111) configures the cache first, so a cache misconfiguration
becomes a usage error (exit 2) before telemetry starts. Three more projects (denxity, aminx-lm, prolix)
are about to write CLI and MCP entry points; without a shared start they will drift the same way.

## 2. Goal and scope

One small xtrax module that every consumer entry point calls. In a fixed order it applies the persistent
JAX compilation cache, initialises cisternal telemetry, and dispatches to a cyclopts app or a FastMCP
server. Cache errors become a clean CLI error. xtrax stays JAX-focused: cisternal is optional and is
imported only when telemetry is requested.

Non-goals:
- No tool registry, `wire()` wrapper, `CliContract` default or plugin manifest in xtrax. Those stay in each project (see §7).
- No change to cisternal.
- No migration of aminx, demistify or aminx-ensemble here. That is an optional follow-up.
- No MCP middleware: installing `CisternalMiddleware` stays the consumer's job, as aminx does in `build_server`.

## 3. Assumptions

| # | Claim | Tag | Evidence |
|---|---|---|---|
| A1 | xtrax has an optional `plugin` extra `cisternal>=0.1.7`, and a lazy-import + install-hint pattern (prints, exits 1) | verified | `pyproject.toml:136`; `src/xtrax/cli/plugin_verb.py:22-35` |
| A2 | `cisternal.init(log_dir, max_bytes, backup_count, exporters, heartbeat_interval, level)` is idempotent; the first call's settings win except `level` | verified | cisternal 0.1.11 `__init__.py:88-121` |
| A3 | `cisternal.init` starts a queue-listener thread and a heartbeat daemon thread; cisternal declares `fork` unsupported (use spawn/forkserver) | verified (challenger R1 C6) | cisternal `telemetry/pipeline.py:9-10,127,450-461` |
| A4 | `cisternal.wire(server, app=None, *, registry, expected, cli_contract, ...)` registers tools; it does not install middleware | verified | cisternal `registration/wired.py:408`; aminx installs `CisternalMiddleware` itself |
| A5 | cisternal requires `cyclopts>=4.18,<6` and `fastmcp>=4` | verified | cisternal `pyproject.toml:23-28` |
| A6 | `xtrax.jax_cache.configure()` raises `JaxCacheConfigError`/`JaxCacheDirError` on bad config or dir, `ValueError` on a malformed config file, and leaves JAX untouched when no layer is set | verified | `src/xtrax/jax_cache.py` (0.4.1b9; docstring fix xtrax#283) |
| A7 | a cyclopts `App` is called `app(tokens, **kw)` (`print_error`, `exit_on_error`, `result_action`, …), and by default it exits: an int return becomes the exit code, any other return is printed and exits 0, and a parse/usage error exits **1** (the default's name varies by version: `"print_non_int_sys_exit"` in 4.25.3, `"print_non_int_return_int_as_exit_code"` in 4.21.1; behavior identical, probed 261123); FastMCP `server.run(**kw)` takes transport arguments | cyclopts verified (R2); FastMCP `run` signature **unverified**, but the passthrough is duck-typed | cyclopts `core.py:1884-1898,2801` (demistify venv); demistify `cli_support.py:226-257`; aminx-ensemble `cli.py:10-14` |
| A12 | `xtrax[plugin]` (cisternal) is in xtrax's dev group, so tier-1 tests see cisternal installed | verified | `pyproject.toml:146-153` |
| A8 | a new `src/xtrax/*` module needs a `changelog.d` fragment; CI runs `assemble_changelog.py --check-new-modules --base-ref origin/main` | verified | `.github/workflows/ci.yml:94-97`; `scripts/assemble_changelog.py:206-304` |
| A9 | `tests/test_import_isolation.py` lists xtrax modules (sorted; `tests/distribution/test_sorted_registries.py`), and a new module is added there | verified (challenger R1 C8c) | `tests/test_import_isolation.py:19-41` |
| A10 | xtrax requires Python >=3.13, so the py39 node-core audits and the `__future__` ban do not apply to this module | verified | `pyproject.toml:6`; node-core audits target `xtrax` node cores only |
| A11 | consumers depend on the xtrax release that ships this (next beta after 0.4.1b9) | unverified | release is the owner's process |

## 4. Design

New public module `src/xtrax/agent.py`. It is not imported by `xtrax/__init__.py` and imports nothing
from cisternal, cyclopts or fastmcp at module level.

```python
def start(*, telemetry: bool = True, init_kwargs: Mapping[str, Any] | None = None,
          **cache_kwargs: Any) -> jax_cache.CacheSetting
def run_cli(app, argv: Sequence[str] | None = None, *, telemetry: bool = True,
            init_kwargs=None, app_kwargs: Mapping[str, Any] | None = None, **cache_kwargs) -> Any
def serve(server, *, telemetry: bool = True, init_kwargs=None,
          run_kwargs: Mapping[str, Any] | None = None, **cache_kwargs) -> None
```

**`start()`**, the ordering contract:
1. `setting = jax_cache.configure(**cache_kwargs)`. Errors propagate.
2. If `telemetry`: import cisternal lazily. A `ModuleNotFoundError` with `e.name == "cisternal"` becomes
   `MissingExtraError` (an `ImportError` subclass), whose message names `xtrax[plugin]` and matches
   `plugin_verb._MISSING_EXTRA` in style. Any other import failure, such as a broken fastmcp inside
   cisternal, propagates unchanged. Then call `cisternal.init(**(init_kwargs or {}))`.
3. Return `setting`.

Cache first follows demistify #111: a cache misconfiguration is a usage error and is reported before
telemetry threads start.

**`run_cli(app, argv)`** is the console-script entry:
- `start(...)`, then `return app(list(sys.argv[1:] if argv is None else argv), **(app_kwargs or {}))`.
- `start()` is composed of two private stages, `_configure_stage(**cache_kwargs)` (returns the
  `CacheSetting`) then `_telemetry_stage(telemetry, init_kwargs)`. `run_cli` calls them separately and
  wraps **only `_configure_stage`** (never telemetry, never the dispatch) in a catch of
  `(ValueError, OSError)`, so an `OSError` or `ValueError` from `cisternal.init` (an unwritable `log_dir`,
  bad `init_kwargs`) stays a traceback and is never reported as a cache error (R3 N5).
  `start()` itself is exactly `_configure_stage(...)` then `_telemetry_stage(...)` (the same two functions
  `run_cli`/`serve` call), so the order test and the stage split cannot drift (R4 MINOR).
  `JaxCacheConfigError` subclasses `ValueError` and `JaxCacheDirError` subclasses `OSError`
  (`jax_cache.py:171,175`), and a malformed config file raises a plain `ValueError` (R2 N1). It prints
  `<prog>: jax cache configuration error: <exc>` to stderr, where `prog` is the app's first name, else
  the `sys.argv[0]` basename, then raises `SystemExit(2)`, demistify's convention. Wrapping only
  `start()` keeps an app's own `ValueError`s as tracebacks.
- It catches `MissingExtraError`, prints its message and raises `SystemExit(1)`, the plugin-verb convention.
- Everything else propagates. Exit codes from the app itself are the app's or `app_kwargs`'s business:
  demistify-style strictness is `app_kwargs={"exit_on_error": False, "print_error": True}` plus the
  consumer's own wrapper. cyclopts' default `result_action` exits (int return = exit code; usage error = exit 1, so it never
  collides with the cache-error exit 2; probed on cyclopts 4.21.1 and 4.25.3, see A7), so `run_cli` *returns* the command's value only
  with `app_kwargs={"result_action": "return_value"}`; by default it exits like a console script should.

**`serve(server)`** = the same two stages then `server.run(**(run_kwargs or {}))`, with the same two
conversions: exit 2 around `_configure_stage` only, exit 1 for `MissingExtraError` from `_telemetry_stage`.

`start()` itself converts nothing: it is the library API. A caller that uses it directly, such as a
consumer with its own CLI runner (aminx's typer `run_cli`), gets the raw `ValueError`/`OSError`/
`MissingExtraError`. The module docstring says so (R2 N2).

`telemetry=False` skips cisternal entirely: no import, no init. That is for a consumer whose CLI runner
already owns telemetry (aminx's typer `run_cli`) or for scripts.

`cache_kwargs` pass unchanged to `jax_cache.configure` (e.g. `path=`). No defaults are added, so
namespace and policy stay xtrax.jax_cache's rules.

Duck-typed: no `isinstance` on cyclopts or FastMCP types.

**Documented constraints** (module docstring):
- With `telemetry=True`, the process runs cisternal threads, and cisternal declares `fork` unsupported.
  Consumers that use `multiprocessing` must use `spawn`/`forkserver` (denxity's harness already spawns;
  prolix to check).
- `telemetry=True` needs `xtrax[plugin]`, which pulls cisternal, cyclopts>=4.18,<6 and fastmcp>=4.
  Adopting the helper in a project that never initialised telemetry (aminx-ensemble today) adds that
  dependency and starts telemetry threads. That is intended for agent-facing entry points, and the
  project can opt out with `telemetry=False`.

Packaging: no new extra. Add a `changelog.d/<slug>.added.md` fragment and add `xtrax.agent` to
`tests/test_import_isolation.py` in sorted order.

## 5. Old vs new (for consumers that adopt it)

| Concern | Hand-rolled today | With `xtrax.agent` |
|---|---|---|
| order | varies | `jax_cache.configure` → `cisternal.init` → dispatch |
| telemetry init | sometimes missing (aminx-ensemble) | always, unless `telemetry=False` |
| cache misconfig | traceback (aminx, aminx-ensemble) or exit 2 (demistify) | stderr line + exit 2 |
| missing cisternal | project-specific | stderr hint naming `xtrax[plugin]` + exit 1 |
| app/server options | hand-written | `app_kwargs` / `run_kwargs` passthrough |
| telemetry options | `cisternal.init(...)` | `init_kwargs` |

## 6. Tests (`tests/test_agent.py`; tier-1 coverage floor applies)

Each patch targets the name `xtrax.agent` looks up: the module attribute `xtrax.jax_cache.configure`,
and `cisternal.init` via the imported module. Agent code calls `jax_cache.configure(...)` and
`cisternal.init(...)` through module attributes, never a `from`-imported function, so a spy sees the
real call.

- **Order:** spies record `["configure", "cisternal.init", "dispatch"]` for `run_cli` and `serve`. Control:
  swapping the two calls in the module turns it red.
- **Cache error → exit 2:** a `configure` spy raising `JaxCacheDirError` gives `SystemExit(2)`; stderr names
  the program; nothing is dispatched; `cisternal.init` is not called. The same holds with the **real**
  `configure` and a malformed `$XDG_CONFIG_HOME/xtrax/config.toml`, which raises a plain `ValueError`
  (R2 N1). Controls: an app that itself raises `ValueError` still gives a traceback, not exit 2; and a
  `cisternal.init` spy raising `OSError` (and, separately, `ValueError`) propagates as a traceback, with no
  "jax cache" text in stderr and no `SystemExit(2)` (R3 N5).
- **Missing cisternal → exit 1:** patch `builtins.__import__` so `import cisternal` raises
  `ModuleNotFoundError("…", name="cisternal")`, and assert in the test that this raised error has that
  `.name`. The result is exit 1 with `xtrax[plugin]` in stderr and nothing dispatched. The patch is used
  rather than `sys.modules["cisternal"] = None`, whose error `.name` is not guaranteed (R2 N4).
- **Broken dependency inside cisternal:** a `ModuleNotFoundError(name="fastmcp")` propagates unchanged and is
  not relabelled.
- **`telemetry=False`:** in a subprocess, call `xtrax.agent.start(telemetry=False)` (with `configure`
  neutralised by env: `XTRAX_JAX_CACHE_DIR=none`) and assert `"cisternal" not in sys.modules` afterwards.
  The positive control is the same subprocess with `telemetry=True` showing `"cisternal" in sys.modules`.
  It passes `init_kwargs={"log_dir": <tmp>}` so the real `cisternal.init` never writes `~/.cisternal/logs`
  (R2 N3). cisternal is in the dev group (A12), so this runs in tier-1. It skips with a reason only where
  cisternal is absent.
- **Passthrough:** `cache_kwargs` reach `configure` unchanged, `init_kwargs` reach `init`, `app_kwargs` reach
  `app(...)`, `run_kwargs` reach `run(...)`; `start()` returns configure's value; `argv=None` uses `sys.argv[1:]`.
  `run_cli` returns the app's value when given a stub callable or a real cyclopts app with
  `app_kwargs={"result_action": "return_value"}`. A real cyclopts app with defaults exits (A7).
- **Import lightness:** in a subprocess with cisternal, cyclopts and fastmcp installed (positive control:
  assert they import there), `import xtrax.agent` leaves all three out of `sys.modules`.
- **Isolation:** any test that reaches the real `configure` sets `XDG_CONFIG_HOME` to an empty tmp dir,
  clears `XTRAX_JAX_CACHE*` and `JAX_COMPILATION_CACHE_DIR`, and restores JAX's flag and `jax_cache._applied`.

## 7. What each per-repo spec still owns

This helper is about a dozen lines of glue. It does not by itself give denxity, aminx-lm or prolix a CLI.
Each per-repo spec must decide:
- **Operations:** the operations, with their I/O contract. denxity has no file I/O today; prolix's long runs are job-submitted.
- **Tools:** the `@cisternal.tool` functions and their registry name.
- **Wiring:** `wire(server, app, registry=..., expected=[...], cli_contract=...)`, including a `CliContract` (demistify uses `BASE_CONTRACT`, `cli.py:1013`).
- **Packaging:** console scripts and the `agent`/`plugin` extras; `agent_plugin.json` and skills.
- **Middleware:** `CisternalMiddleware` on the FastMCP server.
- **Tests:** an `EXPECTED_TOOLS` registry parity test, and a host-config-isolating test fixture (the #3527 lesson).
- **Process start:** `spawn`/`forkserver` where the project forks.

## 8. Delivery

Branch `feat/261123-xtrax-agent-entry`, one PR to xtrax `main`, then an xtrax beta release (owner's
process) before per-repo implementations land. Per-repo specs pin `xtrax[plugin]>=<that release>`.

## 9. Review log

- R1 (spec-challenger, Sonnet, dsp_8c2c814b58f64f97): 0 FATAL, 6 MAJOR, 4 MINOR; all accepted.
  - C1: demistify's real order is configure → init with exit 2; the order is flipped and the conversion added.
  - C2: `app_kwargs`.
  - C3: `ModuleNotFoundError` name check, exit 1.
  - C4: cost documented.
  - C5: `init_kwargs`.
  - C6: fork constraint documented.
  - C7: `run_kwargs`; middleware is the consumer's job.
  - C8: tests fixed (spy targets, positive controls, import-isolation registry).
  - C9: tags corrected; py39 audit out of scope.
  - C10: §7 added.
  - Side finding: demistify `main` configured twice after #111 and #113 merged independently, with 3 tests failing; fixed in demistify#114.
- R2 (same challenger): C1-C6 resolved. One new MAJOR, N1, accepted: `run_cli`/`serve` wrap only `start()` and
  catch `(ValueError, OSError)`, because a malformed config is a plain `ValueError`; a test was added. MINORs
  accepted: N2 (`start()` raw-error docstring), N3 (positive control writes logs to tmp), N4
  (`__import__` patch with an asserted `.name`; cisternal is in the dev group). A7: cyclopts verified;
  FastMCP `run` is still unverified (duck-typed).
- R3 (same challenger): N1 resolved. One new MAJOR, N5, accepted: the exit-2 catch covered `cisternal.init`
  too. `start()` is now two private stages, and `run_cli`/`serve` catch around the configure stage only.
  Control tests added.
- R4 (same challenger): N5 resolved; no FATAL/MAJOR. **Converged.** MINOR applied (`start()` reuses the two stages).
