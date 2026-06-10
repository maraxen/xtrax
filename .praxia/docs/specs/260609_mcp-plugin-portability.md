---
title: MCP Plugin Portability — Cross-Surface Installation Standard
task_id: 260609_mcp-plugin-portability
date: 260609
status: draft
brainstorm_session: true
invest_overrides: []
---

# MCP Plugin Portability — Cross-Surface Installation Standard

## Executive Summary

A three-layer approach to make every MCP server in the ecosystem independently
installable across all AI coding surfaces (Antigravity CLI, Cursor, Claude Code,
VS Code) without requiring praxia as a dependency.

**Layer 1 — Foundation**: Fix all missing entry points, align fastmcp to ≥3.3.1,
create `[mcp]` optional extras, standardize `<pkg>-mcp` naming convention.

**Layer 2 — Discovery**: Register MCPs via Python `entry_points` group
`mcp.servers` for programmatic discovery, and support `uv tool install` for
isolated PATH-based invocation.

**Layer 3 — Config Generation**: Each MCP gains `--emit-config <surface>` for
self-documenting registration. A shared `mcp-config-gen` library centralizes
surface format knowledge. Optional `mcp-init` workspace command scans and
generates all surface configs at once.

**Praxia**: Optional orchestrator — adds cross-MCP coordination, unified
excludeTools policy, and health monitoring when present. Zero coupling required.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   User Install UX                    │
│  uv tool install bathos   OR   pip install bathos[mcp]│
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│              Layer 1: Foundation                     │
│  • [project.scripts] entry point: <pkg>-mcp         │
│  • [project.optional-dependencies] mcp = [fastmcp]  │
│  • fastmcp >= 3.3.1 across all projects             │
│  • Naming: bth-mcp, mrx-mcp, mxl-mcp, ctxp-mcp     │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│              Layer 2: Discovery                      │
│  • [project.entry-points."mcp.servers"]              │
│    bathos = "bathos.mcp:mcp_server"                  │
│  • importlib.metadata.entry_points(group="mcp.servers")│
│  • uv tool install → binary on PATH                  │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│           Layer 3: Config Generation                 │
│  • <pkg>-mcp --emit-config <surface>                 │
│  • Shared lib: mcp-config-gen (surface format logic) │
│  • mcp-init: scan entry_points → generate all configs│
│  • Surfaces: antigravity, cursor, claude, vscode     │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│          Optional: Praxia Orchestration              │
│  • Unified excludeTools policy                       │
│  • Cross-MCP coordination                            │
│  • Health monitoring / daemon integration             │
│  • praxia export-mcp --surface <surface>             │
└─────────────────────────────────────────────────────┘
```

---

## Layer 1: Foundation — Detailed Design

### 1.1 Entry Point Standardization

Every Python MCP project must declare a console_scripts entry point following
the `<short-name>-mcp` convention:

```toml
[project.scripts]
bth-mcp = "bathos.mcp:mcp_server"       # bathos (already exists)
mrx-mcp = "maraxiom.mcp_server:main"    # maraxiom (ADD)
mxl-mcp = "myxcel.mcp_server:main"      # myxcel (ADD)
ctxp-mcp = "contemplex.mcp_server:main" # contemplex (already exists)
jaxlint-mcp = "jaxlint.mcp:main"        # jaxlint (verify)
```

For praxia (Rust): the compiled binary is already named `praxia-mcp`. No change needed.

### 1.2 fastmcp Version Alignment

All Python MCP projects standardize on `fastmcp >= 3.3.1`:

| Project | Current | Target | Migration effort |
|---------|---------|--------|-----------------|
| bathos | >=3.3.1 | >=3.3.1 | None |
| maraxiom | >=3.3.1 | >=3.3.1 | None |
| myxcel | >=0.4 | >=3.3.1 | **Medium** — API rewrite |
| contemplex | >=2.0,<3.0 | >=3.3.1 | **Medium** — remove upper bound, adapt API |
| jaxlint | unknown | >=3.3.1 | Verify and pin |

### 1.3 Optional MCP Extra

Move `fastmcp` out of core dependencies into an optional extra so the core
library (CLI, Python API) can be used without pulling in the MCP server:

```toml
[project.optional-dependencies]
mcp = ["fastmcp>=3.3.1"]
```

This enables:
- `pip install bathos` → CLI only
- `pip install bathos[mcp]` → CLI + MCP server
- `uv run --with bathos[mcp] bth-mcp` → ephemeral MCP run

### 1.4 Naming Convention Table

| Package | CLI command | MCP command | PyPI name |
|---------|------------|-------------|-----------|
| bathos | `bth` | `bth-mcp` | bathos |
| maraxiom | `mrx` | `mrx-mcp` | maraxiom |
| myxcel | `myxcel` | `mxl-mcp` | myxcel |
| contemplex | `ctxp` | `ctxp-mcp` | contemplex |
| jaxlint | `jaxlint` | `jaxlint-mcp` | jaxlint |
| praxia | `praxia` | `praxia-mcp` | N/A (Rust) |

---

## Layer 2: Discovery — Detailed Design

### 2.1 Python Entry Points Group

Each project registers its MCP server callable via the `mcp.servers` entry
point group:

```toml
[project.entry-points."mcp.servers"]
bathos = "bathos.mcp:mcp_server"
```

Discovery code (used by `mcp-init` and future tooling):

```python
from importlib.metadata import entry_points

def discover_mcp_servers() -> dict[str, str]:
    """Find all installed MCP servers via entry points."""
    eps = entry_points(group="mcp.servers")
    return {ep.name: ep.value for ep in eps}
```

### 2.2 uv Tool Install Pattern

For isolated installation (no virtualenv management):

```bash
# Install MCP as isolated tool
uv tool install bathos --with bathos[mcp]

# Now available on PATH
which bth-mcp  # → ~/.local/bin/bth-mcp
```

This is the recommended install method for users who want MCPs available
globally without polluting project environments.

---

## Layer 3: Config Generation — Detailed Design

### 3.1 `--emit-config` Flag

Every MCP gains an `--emit-config` CLI flag:

```bash
$ bth-mcp --emit-config antigravity
{
  "bathos": {
    "command": "bth-mcp",
    "args": [],
    "env": {}
  }
}

$ bth-mcp --emit-config cursor
{
  "bathos": {
    "command": "bth-mcp",
    "args": [],
    "env": {}
  }
}

$ praxia-mcp --emit-config antigravity
{
  "praxia": {
    "command": "praxia-mcp",
    "args": [],
    "env": {
      "DATABASE_URL": "postgresql:///praxia"
    },
    "excludeTools": ["dispatch", "tool_profile_info", ...]
  }
}
```

### 3.2 Shared Library: `mcp-config-gen`

A lightweight shared library (can live as a package or be vendored) that
encapsulates surface-specific config format knowledge:

```python
# mcp_config_gen/surfaces.py

SURFACE_FORMATS = {
    "antigravity": {
        "config_path": "~/.gemini/settings.json",
        "key_path": ["mcpServers"],
        "supports_exclude_tools": True,
    },
    "cursor": {
        "config_path": ".cursor/mcp.json",
        "key_path": ["mcpServers"],
        "supports_exclude_tools": False,
    },
    "claude": {
        "config_path": ".claude/mcp.json",
        "key_path": ["mcpServers"],
        "supports_exclude_tools": False,
    },
    "copilot": {
        "config_path": "~/.copilot/mcp-config.json",
        "key_path": ["mcpServers"],
        "supports_exclude_tools": False,
    },
    "vscode": {
        "config_path": ".vscode/settings.json",
        "key_path": ["mcp", "servers"],
        "supports_exclude_tools": False,
    },
}


def emit_config(
    name: str,
    command: str,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    exclude_tools: list[str] | None = None,
    surface: str = "antigravity",
) -> str:
    """Generate JSON config snippet for a given surface."""
    ...
```

### 3.3 `mcp-init` Workspace Command

Scans for installed MCPs and generates surface configs:

```bash
$ mcp-init --surfaces antigravity,cursor

Discovered MCP servers:
  ✓ bathos (bth-mcp)
  ✓ maraxiom (mrx-mcp)
  ✓ contemplex (ctxp-mcp)

Generated:
  ~/.gemini/settings.json  (3 servers added)
  .cursor/mcp.json         (3 servers added)
```

This command uses `importlib.metadata.entry_points(group="mcp.servers")` for
discovery and `mcp-config-gen` for format generation.

---

## Praxia Role: Optional Orchestrator

Praxia remains useful but never required:

| Capability | Without Praxia | With Praxia |
|-----------|---------------|-------------|
| MCP installation | `uv tool install pkg[mcp]` | Same |
| Config generation | `<pkg>-mcp --emit-config` | `praxia export-mcp --surface` |
| Tool exclusion | Manual per-surface config | Unified policy from praxia.toml |
| Health monitoring | None | Daemon health checks |
| Cross-MCP coordination | None | Orchestrated tool routing |
| Discovery | entry_points scan | entry_points + praxia registry |

---

## Per-Project Migration Checklist

### bathos (minimal changes)
- [ ] Move `fastmcp` from core deps to `[project.optional-dependencies] mcp`
- [ ] Add `[project.entry-points."mcp.servers"] bathos = "bathos.mcp:mcp_server"`
- [ ] Add `--emit-config` flag to `bth-mcp` entry point
- [ ] Verify `uv tool install bathos --with bathos[mcp]` works

### maraxiom (medium changes)
- [ ] Add `mrx-mcp = "maraxiom.mcp_server:main"` to `[project.scripts]`
- [ ] Move `fastmcp` from core deps to `[project.optional-dependencies] mcp`
- [ ] Add `[project.entry-points."mcp.servers"] maraxiom = "maraxiom.mcp_server:main"`
- [ ] Add `--emit-config` flag to mcp_server.py
- [ ] Publish to PyPI (if not already)

### myxcel (larger changes)
- [ ] Upgrade fastmcp from >=0.4 to >=3.3.1 — **API migration required**
- [ ] Add `mxl-mcp = "myxcel.mcp_server:main"` to `[project.scripts]`
- [ ] Move `fastmcp` to `[project.optional-dependencies] mcp`
- [ ] Add `[project.entry-points."mcp.servers"] myxcel = "myxcel.mcp_server:main"`
- [ ] Add `--emit-config` flag

### contemplex (medium changes)
- [ ] Remove `fastmcp<3.0` upper bound, pin `>=3.3.1`
- [ ] Adapt MCP server code to fastmcp 3.x API if needed
- [ ] Move `fastmcp` to `[project.optional-dependencies] mcp`
- [ ] Add `[project.entry-points."mcp.servers"] contemplex = "contemplex.mcp_server:main"`
- [ ] Add `--emit-config` flag

### jaxlint (small changes)
- [ ] Verify fastmcp version and pin >=3.3.1
- [ ] Add `[project.optional-dependencies] mcp` if not present
- [ ] Add `[project.entry-points."mcp.servers"] jaxlint = "jaxlint.mcp:main"`
- [ ] Add `--emit-config` flag

### praxia (Rust — different path)
- [ ] Add `--emit-config <surface>` flag to `praxia-mcp` binary
- [ ] Include default `excludeTools` list in emitted config
- [ ] Document `DATABASE_URL` env var requirement in emitted config

---

## Implementation Phasing

### Phase 1: Foundation Sprint (~2 days)
- Fix all missing entry points (maraxiom, myxcel)
- Align fastmcp versions across projects
- Create `[mcp]` optional extras
- Standardize naming convention
- **Exit criterion**: Every MCP invocable via `<pkg>-mcp` on PATH after install

### Phase 2: fastmcp Migration Sprint (~3-5 days)
- Migrate myxcel from fastmcp 0.4 to 3.x
- Migrate contemplex from fastmcp 2.x to 3.x
- Test all MCP servers against fastmcp 3.3.1+
- **Exit criterion**: All MCPs pass smoke tests on fastmcp 3.3.1

### Phase 3: Discovery & Config Gen (~3 days)
- Add `entry_points` group to all projects
- Implement `--emit-config` in shared mcp-config-gen library
- Wire `--emit-config` into each MCP's CLI
- Build `mcp-init` discovery + generation command
- **Exit criterion**: `mcp-init --surfaces antigravity,cursor` generates valid configs

### Phase 4: PyPI Publish & Documentation (~2 days)
- Publish all Python MCPs to PyPI
- Document install and registration for each surface
- Update README for each project with MCP install instructions
- **Exit criterion**: `uv tool install bathos --with bathos[mcp] && bth-mcp --emit-config antigravity` works from a clean machine

---

## Acceptance Criteria

- Given a Python MCP project has been published to PyPI, when a user runs
  `uv tool install <pkg> --with <pkg>[mcp]`, then the `<pkg>-mcp` command
  is available on PATH.

- Given a user runs `<pkg>-mcp --emit-config antigravity`, then the command
  outputs valid JSON matching the Antigravity CLI mcpServers schema.

- Given a user runs `<pkg>-mcp --emit-config cursor`, then the command
  outputs valid JSON matching the Cursor `.cursor/mcp.json` schema.

- Given a user runs `<pkg>-mcp --emit-config claude`, then the command
  outputs valid JSON matching the Claude Code `.claude/mcp.json` schema.

- Given multiple MCP packages are installed with the `mcp.servers` entry point
  group, when a user runs `mcp-init --surfaces antigravity`, then all
  discovered MCPs are added to `~/.gemini/settings.json`.

- Given an MCP project has `fastmcp` in `[project.optional-dependencies] mcp`,
  when a user runs `pip install <pkg>` without the `[mcp]` extra, then
  `fastmcp` is not installed.

- Given praxia is not installed, when a user installs and configures any
  individual MCP via `--emit-config`, then the MCP starts and responds to
  tool calls without error.

- Given praxia IS installed alongside other MCPs, when a user runs
  `praxia export-mcp --surface antigravity`, then praxia generates configs
  for all discovered MCPs with its orchestration policy applied.

---

## Decision Log

| Option | Verdict | Rationale |
|--------|---------|-----------|
| **Layered Composite (Foundation + Discovery + Config Gen)** | **Selected** | Best separation of concerns; each layer delivers standalone value; composable; doesn't introduce new central dependencies; enables future `mcp-install` CLI as optional add-on |
| CLI Installer Tool (`mcp-install`) | Deferred | Lowest friction UX but introduces a new central dependency before the foundation exists; can be built later on top of entry_points + mcp-config-gen |
| Praxia Plugin Export (fix gaps) | Incorporated | Praxia's `export-mcp` becomes a consumer of Layer 2+3 rather than the sole mechanism; removes coupling |
| Direct Function Dual-Decoration (`@app.command()` + `@mcp.tool()`) | **Rejected** | Code smell that couples CLI presentation (interactive CLI options, stdout, rich formatting) with MCP schema (JSON/Markdown returns, non-interactive, async stdio transport). Sticking to three-layer code architecture (Core Logic + CLI View + MCP View). |
| Unified CLI+MCP Base Library | **Deferred** | Not necessary for 6 projects. However, shared utilities (such as telemetry wrappers, base logger config, and config emission schemas) can be placed in `mcp-config-gen` library to prevent code duplication. |
| Self-Registration on First Run | Rejected | Security risk: MCP writing to host config files without explicit user consent; violates principle of least surprise |
| Central MCP Registry | Rejected | Premature for 6-server ecosystem; infrastructure overhead not justified; revisit if ecosystem grows to 20+ servers |
| Meta-Package (`mcp-suite`) | Rejected | Installs everything when user may only want one MCP; fights granularity; users can `pip install` individually |
| Monorepo MCP Workspace | Rejected | Fights existing polyrepo structure; coordination cost exceeds version-alignment benefit; version pinning achieves same goal |
| MCP-as-Container | Rejected | Massive overhead for stdio-based tools; Docker dependency inappropriate for lightweight dev tooling |


---

## Assumptions

| Assumption | Owner | Verification method |
|------------|-------|---------------------|
| All surfaces (AG, Cursor, Claude, VSCode) use `{ command, args?, env? }` over stdio | Marielle | Verify against current surface docs before Phase 3 |
| fastmcp 3.x API is stable enough to standardize on | Marielle | Check fastmcp changelog and release cadence |
| `uv tool install --with pkg[extra]` properly resolves optional deps | Marielle | Test with bathos before rolling out to all projects |
| `importlib.metadata.entry_points` works in uv-tool-installed environments | Marielle | Test discovery from isolated tool install |
| Surface config formats won't change breaking-ly in the next 6 months | N/A (external) | Monitor surface release notes; mcp-config-gen absorbs changes |

---

## TBDs

| Item | Owner | Resolution deadline |
|------|-------|---------------------|
| Exact fastmcp 3.x migration scope for myxcel (0.4 → 3.x delta) | Marielle | Before Phase 2 start |
| Whether `mcp-config-gen` should be a standalone PyPI package or vendored | Marielle | Phase 3 design |
| VS Code MCP server config format (less documented than others) | Marielle | Phase 3 research |
| Whether praxia `--emit-config` should include excludeTools from praxia.toml | Marielle | Phase 3 design |
| PyPI org/namespace strategy (individual packages vs. org namespace) | Marielle | Before Phase 4 |

---

## Pre-mortem Record

- **Failure scenario (fastmcp migration)**: myxcel and contemplex fastmcp upgrades
  require significant rewrites due to breaking API changes between major versions.
  Mitigation: budget explicit migration sprints with test coverage before/after.
  The spec accounts for this with dedicated Phase 2.

- **Failure scenario (uv tool isolation)**: MCPs installed via `uv tool install`
  can't access workspace-specific libraries or configs (e.g., praxia needs
  DATABASE_URL). Mitigation: `--emit-config` explicitly outputs required env vars
  so users configure them in the surface config. The spec documents env var
  requirements per MCP.

- **Failure scenario (N×M config maintenance)**: Surface config format changes
  break `--emit-config` across all MCPs. N MCPs × M surfaces = N×M maintenance
  points. Mitigation: extract config generation into shared `mcp-config-gen`
  library so format changes are fixed once. The spec designs for this with the
  shared library in Layer 3.

---

## INVEST Gate Report

```
✓ Independent — no blockers; each layer can be implemented incrementally
✓ Negotiable — Layer 3 (config gen) can be deferred; Layer 1 alone delivers value
✓ Valuable — enables cross-surface MCP installation without praxia coupling
✓ Estimable — ~10-12 days across 4 phases (see phasing above)
✗ Small — 8 acceptance criteria (at limit); 6 projects × 4 phases = significant scope
  → Acceptable: phases are independently deliverable; each phase is small
✓ Testable — all criteria specify observable outputs (JSON output, PATH availability, config file contents)

Decision: Passes. Small dimension at limit but phases provide natural decomposition.
```
