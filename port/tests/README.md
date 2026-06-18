# Port parity tests

Graded parity harness for implementation validation (P3-PARITY).

- Tests are named `test_parity_<kernel>.py` and import production symbols from `src/xtrax/` (in-place translation).
- `conftest.py` (added in a later wave) enforces blocking tier order T1→T2→T3→(T4 if `ad_critical`)→T5.
- T4 is skipped unless `port/port_target.toml` sets `ad_critical = true` with a non-empty `ad_critical_justification`.

Run with dev extra installed:

```bash
uv sync --extra dev
uv run pytest port/tests/ -v
```
