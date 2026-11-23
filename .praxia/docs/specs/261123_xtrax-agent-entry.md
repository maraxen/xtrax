---
title: 'xtrax.agent: one startup for consumer CLI and MCP entry points'
description: 'Shared core for debts #3553-#3555: a small xtrax helper that runs cisternal.init() and xtrax.jax_cache.configure() in one fixed order, then dispatches a cyclopts app or a FastMCP server; per-repo specs build on it.'
status: draft
task_id: 261123_xtrax-entrypoint-cli-mcp
date: '261123'
backlog_ids: ''
adversarial_review: ''
---
# xtrax.agent: one startup for consumer CLI and MCP entry points


task_id: `261123_xtrax-entrypoint-cli-mcp` · shared core for debts **#3553** (denxity), **#3554** (aminx-lm),
**#3555** (prolix). #3556 (alynxr) does not use it: alynxr's CLI/MCP are already cisternal-wired and its
own process compiles no JAX (recon dsp_32d4ef3f508f25a9). Owner decision 2026-11-23: "small xtrax helper".

## 1. Problem

Three consumers already expose a CLI and/or an MCP server built on cisternal (`@cisternal.tool` +
`cisternal.wire(server, app, ...)`), and each hand-rolls the process start differently:

| Project | Entry | Startup order today |
|---|---|---|
| demistify | `demistify.cli:main` | `cisternal.init()` → `jax_cache.configure()` → `run_app(app, argv)` |
| aminx | `aminx.cli:main` | `jax_cache.configure()` → `run_cli(app)` (typer, telemetry inside) |
| aminx | `aminx.agent.mcp:main` | `require_agent_extra()` → `cisternal.init()` → `jax_cache.configure()` → `build_server().run()` |
| aminx-ensemble | `aminx_ensemble.cli:main` | `jax_cache.configure()` → `app(argv, result_action=...)` (no `cisternal.init`) |
| aminx-ensemble | `aminx_ensemble.mcp:main` | `jax_cache.configure()` → `server.run()` (no `cisternal.init`) |

Four more entry points are about to be written (denxity, aminx-lm, prolix CLI + MCP). Without a shared
start they will drift the same way: telemetry sometimes not initialised, cache configured before or
after telemetry, missing-extra errors worded differently.

## 2. Goal

One xtrax function family that every consumer entry point calls, which (in a fixed order) initialises
cisternal telemetry, applies the persistent JAX compilation cache, and dispatches to a cyclopts app or a
FastMCP server. xtrax stays JAX-focused: cisternal stays optional and is imported only when called.

Non-goals: no tool registry, no `wire()` wrapper, no CLI framework in xtrax; no change to cisternal; no
migration of aminx/demistify/aminx-ensemble in this spec (follow-up, optional).

## 3. Assumptions

| # | Claim | Tag | Evidence |
|---|---|---|---|
| A1 | xtrax already has an optional `plugin` extra = `cisternal>=0.1.7` and a lazy-import + install-hint pattern for it | verified | `pyproject.toml:118-136`; `src/xtrax/cli/plugin_verb.py:22-35` |
| A2 | `cisternal.init(log_dir=None, ..., level=None)` is idempotent | verified | cisternal 0.1.11 `src/cisternal/__init__.py:88-121` |
| A3 | `cisternal.wire(server, app=None, *, registry, expected, cli_contract, ...)` registers tools on a FastMCP server and a cyclopts app; it is the consumer's job, not this helper's | verified | cisternal `registration/wired.py:408` |
| A4 | cisternal requires `cyclopts>=4.18,<6` and `fastmcp>=4` | verified | cisternal `pyproject.toml:23-28` |
| A5 | `xtrax.jax_cache.configure()` resolves and applies the cache, fails loudly on malformed config, and leaves JAX untouched when no layer is set | verified | `src/xtrax/jax_cache.py` (0.4.1b9; docstring fix in xtrax#283) |
| A6 | consumers call a cyclopts app as `app(argv)` / `app(argv, result_action="return_value")` and a FastMCP server as `server.run()` | verified | aminx-ensemble `cli.py:10-14`, `mcp.py:1212-1214`; demistify `cli.py:1021-1029` |
| A7 | a new module under `src/xtrax/` needs a `changelog.d` fragment (CI "Changelog fragments & new module check") | verified | `.github/workflows/ci.yml:94` |
| A8 | a module imported by `import xtrax` must not import cisternal, cyclopts or fastmcp eagerly | verified (rule) | `plugin_verb.py` docstring; `import xtrax` stays light |
| A9 | consumers will depend on the release that ships this (next xtrax beta after 0.4.1b9) | unverified | needs an xtrax release; per-repo specs pin `>=` that version |

## 4. Design

New module `src/xtrax/agent.py` (public, not imported by `xtrax/__init__.py`):

```python
def start(*, telemetry: bool = True, **cache_kwargs) -> jax_cache.CacheSetting
def run_cli(app, argv=None, *, telemetry: bool = True, **cache_kwargs)
def serve(server, *, telemetry: bool = True, **cache_kwargs) -> None
```

- `start()` is the whole contract: (1) if `telemetry`, import cisternal lazily and call `cisternal.init()`
  — if cisternal is missing, raise `ImportError` with the same install-hint wording style as
  `plugin_verb._MISSING_EXTRA` naming the `plugin` extra; (2) call `jax_cache.configure(**cache_kwargs)`
  and return its `CacheSetting`. Telemetry first, so a cache configuration error is raised after
  telemetry is up (cisternal records process start). Errors propagate; nothing is caught.
- `run_cli(app, argv)` = `start(...)` then `return app(argv)` (cyclopts semantics; `argv=None` → `sys.argv[1:]`
  as cyclopts does). The helper does not translate exit codes; usage errors exit as cyclopts decides.
- `serve(server)` = `start(...)` then `server.run()` (stdio default, FastMCP semantics).
- `telemetry=False` skips cisternal entirely (for a consumer that runs telemetry inside its own CLI
  runner, e.g. aminx's typer `run_cli`).
- `cache_kwargs` pass straight to `jax_cache.configure` (e.g. `path=` for an explicit override); no
  defaults are added here, so the namespace and policy rules stay xtrax.jax_cache's.
- Duck-typed: no `isinstance` on cyclopts/FastMCP, so neither is imported by `xtrax.agent`.

Packaging: no new extra; document that `xtrax.agent` with `telemetry=True` needs `xtrax[plugin]`.
`changelog.d/<slug>.added.md` fragment. Module docstring explains when to use which function, with one
example each for a cyclopts CLI and a FastMCP server, and the ordering guarantee.

## 5. Old vs new (for consumers that adopt it)

| Concern | Hand-rolled today | With `xtrax.agent` |
|---|---|---|
| telemetry init | sometimes missing (aminx-ensemble) | always, unless `telemetry=False` |
| order | varies | `cisternal.init` → `jax_cache.configure` → dispatch |
| missing cisternal | project-specific error or none | `ImportError` naming `xtrax[plugin]` |
| cache override | each passes its own args | `**cache_kwargs` to `configure` |

## 6. Tests (in `tests/`, xtrax's tier-1 coverage floor applies)

- Order: monkeypatch `cisternal.init`, `xtrax.jax_cache.configure` and a stub app/server; `run_cli` and
  `serve` record `["cisternal.init", "configure", "dispatch"]`. Control: reordering in the module turns it red.
- `telemetry=False` never imports cisternal (assert `"cisternal" not in sys.modules` after a fresh
  import in a subprocess, or patch `importlib.import_module` to raise).
- Missing cisternal with `telemetry=True` → `ImportError` mentioning `xtrax[plugin]` and nothing dispatched.
- `cache_kwargs` reach `configure` unchanged; the returned value of `start()` is configure's return.
- `run_cli` returns the app's return value; `argv` passes through.
- A configure error propagates and nothing is dispatched.
- `import xtrax` and `import xtrax.agent` do not import cisternal, cyclopts or fastmcp (subprocess check).
- Isolation: tests that reach the real `configure` set `XDG_CONFIG_HOME` to an empty tmp dir, clear
  `XTRAX_JAX_CACHE*`, and restore JAX's flag and `jax_cache._applied`.

## 7. Delivery

Branch `feat/261123-xtrax-agent-entry`, one PR to xtrax `main`, then an xtrax beta release (owner's
release process) before the per-repo implementations land. Per-repo specs (denxity, aminx-lm, prolix)
reference this module and pin `xtrax[plugin]>=<that release>`.

## 8. Review log

(pending)
