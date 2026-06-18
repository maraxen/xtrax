# Port validation (dev-only)

The `port/` subtree supports the [#2180](.praxia/docs/specs/260618_hmw-design-unified-implementation-valida.md) implementation validation pipeline: sealed reference oracles, graded parity tiers, static gates, and `domain=port` emit records.

**This tree is not shipped in the release wheel.** Only `src/xtrax/` is packaged. Port gates require a dev install.

## Dev-extra install

```bash
uv sync --extra dev
```

Port validation scripts and tests live under `port/` and are exercised locally via `just audit-port` (see `Justfile`) and in CI on PRs that touch `src/xtrax/` or `port/`.

## Layout

| Path | Purpose |
|------|---------|
| `port_target.toml` | Active port wave config (`wave_id`, oracle lock, capabilities, parity flags) |
| `reference/` | Sealed mathematical oracle (`# REFERENCE: DO NOT MODIFY`) |
| `tests/` | Graded parity harness (`test_parity_*.py` imports from `src/xtrax/`) |
| `emit/` | `port_emit.py` — appends `domain=port` records to `.praxia/audits.jsonl` |
| `manifests/` | P1.5 topo manifest artifacts (`manifest_hash`, topo-sorted qualnames) |
| `docs/` | Port-validation hook and workflow documentation |

Production translations land **in-place** in `src/xtrax/` (no `port/jax_port/` staging).

## Manifest resolution

Read `wave_id` from `port/port_target.toml` → load `port/manifests/<wave_id>.toml`. Manifests carry `manifest_hash` (SHA-256 of canonical TOML bytes) and `task_id` for freshness checks (AC-11).
