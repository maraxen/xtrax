# Composition bridge (optional, Phase 2)

`composition_map.toml` is an **opt-in** bridge between port validation and the composition graph ([#2174](https://github.com/maraxen/xtrax/issues/2174) Phase 2). It maps live `src/xtrax/` symbol qualnames to composition `node_id` values used in `.praxia/composition/`.

## v0.1 MVP

- The map ships **empty** (`[symbols]` with no entries).
- **No CI gate** in v0.1 — port waves do not require bridge completeness.
- Phase 1 port validation ignores composition metadata entirely.

## When to populate

Populate `[symbols]` when an integration port needs composition orchestration to reference a translated kernel. Each key must be a resolvable qualname under `xtrax.*`; each value is the target composition graph `node_id`.

## Lint

When `[symbols]` is non-empty, validate qualnames locally or in CI:

```bash
just audit-port-bridge
# or
uv run python scripts/lint_port_bridge_map.py
```

The linter exits `0` for an empty map and fails when any qualname does not resolve in `src/xtrax/`.
