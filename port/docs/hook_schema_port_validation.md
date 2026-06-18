# Hook schema: `subagent-stop` / `port_validation` (v0.1)

Normative contract for P3-PARITY tier completion hooks. The PCW walker validates the **flat payload** only; optional wrapper keys are transport metadata.

**Sources:** spec Appendix A (`.praxia/docs/specs/260618_hmw-design-unified-implementation-valida.md`), design §1.5 (`.praxia/docs/designs/260618_2180_port_validation_design.md`).

## Normative flat payload

The walker unmarshals and validates this object. All fields are required.

| Field | Type | Description |
|-------|------|-------------|
| `tier_verdict` | `"PASS"` \| `"FAIL"` | Subagent-reported tier outcome |
| `port_parity_tier` | `"tier_1"` … `"tier_5"` | Graded parity tier that completed |
| `oracle_id` | string | Must match `oracle_id` lock in `port/port_target.toml` |
| `pytest_nodeid` | string | Full pytest node id, e.g. `port/tests/test_parity_<kernel>.py::test_tier_<n>` |
| `pytest_exit_code` | integer | Process exit code from the tier pytest invocation |
| `stdout_sha256` | string | SHA-256 (hex, lowercase) of the summary line (see below) |

```json
{
  "tier_verdict": "PASS",
  "port_parity_tier": "tier_3",
  "oracle_id": "ref:port/reference/safe_map:v0.1.0:sha256:52fd5458018d46d3c333287f803152eb38f66b618f540363521d825b518aea34",
  "pytest_nodeid": "port/tests/test_parity_safe_map.py::test_tier_3",
  "pytest_exit_code": 0,
  "stdout_sha256": "8209e56ae8f4cbcfa36112f13d52c1fb17dc7c0823fa6fd0ed4fdaa454189a1a"
}
```

Supervisor **aggregates** hook payloads; it does **not** originate tier verdicts.

## Optional wrapper metadata (transport only)

PCW / design transport may wrap the flat payload. The walker **ignores** outer keys and validates only the nested `payload` object.

```json
{
  "hook": "subagent-stop",
  "workflow": "port_validation",
  "phase": "P3-PARITY",
  "payload": {
    "tier_verdict": "PASS",
    "port_parity_tier": "tier_3",
    "oracle_id": "ref:port/reference/<kernel>:v0.1.0:sha256:<hash>",
    "pytest_nodeid": "port/tests/test_parity_<kernel>.py::test_tier_<n>",
    "pytest_exit_code": 0,
    "stdout_sha256": "<sha256 of UTF-8 normalized pytest summary line>"
  }
}
```

## `stdout_sha256` algorithm

1. Capture stdout from the tier pytest run.
2. Find the **final** line that matches `PASSED` or `FAILED` for the given `pytest_nodeid` (pytest short-summary format).
3. Normalize: use the line as emitted (no trailing newline in the hash input).
4. Compute SHA-256 over the UTF-8 encoding of that line; emit lowercase hex.

Example summary line (PASS):

```
port/tests/test_parity_safe_map.py::test_tier_3 PASSED [ 42%]
```

```python
import hashlib
line = "port/tests/test_parity_safe_map.py::test_tier_3 PASSED [ 42%]"
stdout_sha256 = hashlib.sha256(line.encode("utf-8")).hexdigest()
# → 8209e56ae8f4cbcfa36112f13d52c1fb17dc7c0823fa6fd0ed4fdaa454189a1a
```

## Walker FAIL conditions

Dispatch **FAIL** when **any** of the following hold (design §1.5):

1. **`tier_verdict == "PASS"` but `pytest_exit_code != 0`** — claimed pass with a failing process.
2. **`stdout_sha256` mismatch** — recomputed hash from captured stdout does not equal the payload value.
3. **`oracle_id` mismatch** — payload `oracle_id` ≠ `oracle_id` in `port/port_target.toml`.
4. **Hook PASS vs emit FAIL** — hook claims `tier_verdict: PASS` but the P4 emit record for the same tier reports `status: FAIL` (cross-check in P4-EMIT).

## Worked examples

### PASS — tier 3 parity subagent completion

Wrapper (transport) with normative inner payload:

```json
{
  "hook": "subagent-stop",
  "workflow": "port_validation",
  "phase": "P3-PARITY",
  "payload": {
    "tier_verdict": "PASS",
    "port_parity_tier": "tier_3",
    "oracle_id": "ref:port/reference/safe_map:v0.1.0:sha256:52fd5458018d46d3c333287f803152eb38f66b618f540363521d825b518aea34",
    "pytest_nodeid": "port/tests/test_parity_safe_map.py::test_tier_3",
    "pytest_exit_code": 0,
    "stdout_sha256": "8209e56ae8f4cbcfa36112f13d52c1fb17dc7c0823fa6fd0ed4fdaa454189a1a"
  }
}
```

Walker: PASS — exit code 0, hash matches summary line, `oracle_id` matches `port_target.toml`, emit agrees.

### FAIL — stdout hash mismatch (claimed PASS)

Subagent reports PASS but `stdout_sha256` does not match captured stdout (e.g. stale or wrong summary line):

```json
{
  "hook": "subagent-stop",
  "workflow": "port_validation",
  "phase": "P3-PARITY",
  "payload": {
    "tier_verdict": "PASS",
    "port_parity_tier": "tier_2",
    "oracle_id": "ref:port/reference/safe_map:v0.1.0:sha256:52fd5458018d46d3c333287f803152eb38f66b618f540363521d825b518aea34",
    "pytest_nodeid": "port/tests/test_parity_safe_map.py::test_tier_2",
    "pytest_exit_code": 0,
    "stdout_sha256": "0000000000000000000000000000000000000000000000000000000000000000"
  }
}
```

Captured stdout ends with:

```
port/tests/test_parity_safe_map.py::test_tier_2 FAILED [ 28%]
```

Correct hash would be `c02670524364ab168958fc3970d98906c6fca22d111c198531e1af4c819776fc`. Walker: **FAIL** (condition 2 — hash mismatch; also condition 1 if exit code were non-zero).

## Related `port_target.toml` fields

- **T4 timeout (AC-4):** `@pytest.mark.timeout(120)` per tier on CPU. `port/port_target.toml` may **lower** the budget, not raise it, without justification. T4 runs only when `[parity] ad_critical = true` with non-empty `ad_critical_justification`.
- **Trace-count baseline (AC-5):** `[parity] max_traces = 1` default. `scripts/audit_port_trace_count.py` runs `chex.assert_max_traces` on the jitted entrypoint qualname from the wave manifest.

See `port/port_target.toml` for the active wave template.
