# Port emit (P4-EMIT)

`port_emit.py` appends `domain=port` records to `.praxia/audits.jsonl` after each parity tier completes.

- `finding_id = hash(dim + symbol_qualname + rule_id + tolerance_policy)` per audit-fw #1573 + N1.1 port amendment.
- Delegates to `xtrax.devtools.emit` when #1577 is importable; otherwise uses a local stub validated by `tests/contract/test_port_emit_schema.py`.

Implemented in backlog #2267.
