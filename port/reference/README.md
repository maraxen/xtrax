# Sealed reference oracle (P0-ORACLE)

Vendored reference implementations live here. This subtree is the mathematical ground truth for port parity.

## Conventions

- Every reference source file **must** start with `# REFERENCE: DO NOT MODIFY`.
- Baseline I/O pairs are generated from vendored oracle execution — never from author-reported paper tables.
- Only the `reference-vendor` identity may write here; fixer dispatch scopes are read-only on `port/reference/`.
- `oracle_id` in `port/port_target.toml` locks the content hash: `ref:port/reference/<kernel>:v0.1.0:sha256:<hash>`.

## Layout (per kernel)

```
port/reference/<kernel>/
├── algo.py              # sealed reference implementation
└── baseline_io.json     # oracle-generated I/O pairs
```
