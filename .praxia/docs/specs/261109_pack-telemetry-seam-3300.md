---
title: "xtrax Pack Telemetry Producer & Consumer Seam Specification (Debt #3300 Addendum)"
description: "Specification addendum for Debt #3300: producer-side timing and compression telemetry in xtrax.pack._node, strict stdlib isolation, schema compatibility with alynxr reader parity, and consumer-side Q5 ingestion in xtrax.eda.campaign.telemetry"
task_id: "261109_pack-telemetry-and-eda-pr-swarm"
category: "specs"
status: "complete"
revision: "1.0 (2026-11-09; mini-spec addendum closing debt #3300)"
owner_decisions: "top-level pack_telemetry record key; storage key set unchanged; 100% Python stdlib node core; transparent campaign EDA Q5 aggregation with graceful legacy fallback"
supersedes_scope_of: "xtrax debt #3300 (pack telemetry seam)"
---

# xtrax Pack Telemetry Producer & Consumer Seam Specification

## 1. Executive Summary & Objective

This specification addendum closes **xtrax Debt #3300** by formalizing the producer-side emission and consumer-side ingestion of per-unit pack telemetry.

The archive packing subsystem (`xtrax.pack`) converts loose filesystem directories of execution unit outputs into compact, seekable Zstandard archives (`members.tar.zst` + `members.idx.json`). Previously, packing wall-time and compression performance were uninstrumented, causing Question 5 (Secondary Run Telemetry) of the Campaign EDA engine (`xtrax.eda.campaign`) to display an uninstrumented placeholder banner:
```
Pack Telemetry: not instrumented (debt #3300)
```

This addendum defines:
1. **Producer-Side Telemetry in `xtrax.pack._node`**: Accurate measurement of pack wall time, raw member bytes, compressed pack bytes, compression ratio, member count, and compressor backend during `pack_unit`.
2. **Schema Compatibility & alynxr Reader Parity**: Storing telemetry in a dedicated top-level `record["pack_telemetry"]` object in `unit.json` / `done.json`, keeping `record["storage"]` strictly unchanged to preserve exact key set equality (`assert set(storage) == set(theirs)`) with alynxr unit fixtures.
3. **Strict Standard Library Isolation**: Maintaining 100% Python 3.9 standard library-only isolation in `src/xtrax/pack/_node.py` with zero external dependencies, preserving cluster batch script embeddability (`xtrax.pack.node_source()`).
4. **Consumer-Side Ingestion & Aggregation in `xtrax.eda.campaign`**: Ingesting per-unit telemetry across campaign units, computing whole-campaign aggregates (`total_raw_bytes`, `total_compressed_bytes`, `overall_compression_ratio`, `mean_pack_wall_ms`, `backends`), and dynamically replacing the uninstrumented banner.
5. **Campaign Aggregation States & Backward Compatibility**: Formally defining the four lifecycle states (`"instrumented"`, `"mixed"`, `"no_packed_units"`, `"uninstrumented"`) to guarantee graceful degradation on legacy or uninstrumented units.

---

## 2. Schema Compatibility & Architectural Invariants

### 2.1 The alynxr Storage Parity Invariant
The unit packing format was lifted from alynxr (`cd8e5a9`) to ensure universal interoperability across protein engineering repos. The alynxr test suite and cross-repo contracts assert that `record["storage"]` contains only the canonical alynxr storage fields:
```python
assert set(storage) == {
    "codec",
    "format",
    "index",
    "index_sha256",
    "n_members",
    "pack",
    "pack_sha256",
    "schema",
}
```
Injecting telemetry metrics directly into `record["storage"]` violates key set equality and breaks downstream alynxr readers.

### 2.2 Record Digest & Core Fields Immutability
`src/xtrax/pack/_node.py` computes canonical record identity over `_CORE_FIELDS`:
```python
_CORE_FIELDS = (
    "input_digest",
    "input_payload",
    "meta",
    "content_digest",
    "run_id",
    "env",
    "committed_at",
)
```
`compute_record_digest(record)` hashes only these core fields. By placing telemetry outside `_CORE_FIELDS` as a top-level key `record["pack_telemetry"]`, pack telemetry can be added, updated, or omitted without invalidating `record_digest` or failing `record_problem(record)`.

### 2.3 Unit Record Schema (`xtrax.pack_telemetry/1`)
When `pack_unit` packs an execution unit, `unit.json` (or `done.json` when `record_name="done.json"`) records:
```json
{
  "schema": "xtrax.unit/1",
  "content_digest": "...",
  "record_digest": "...",
  "storage": {
    "schema": "xtrax.unit_pack/1",
    "pack": "members.tar.zst",
    "index": "members.idx.json",
    "format": "tar.zst+frame-index/1",
    "codec": "zstd",
    "pack_sha256": "...",
    "index_sha256": "...",
    "n_members": 3
  },
  "pack_telemetry": {
    "schema": "xtrax.pack_telemetry/1",
    "pack_wall_ms": 12.45,
    "raw_bytes": 1048576,
    "compressed_bytes": 262144,
    "compression_ratio": 4.0,
    "member_count": 3,
    "backend": "zstandard"
  }
}
```

### 2.4 Telemetry Fields Contract
| Field | Type | Description |
|:---|:---:|:---|
| `schema` | `str` | Must equal `"xtrax.pack_telemetry/1"`. |
| `pack_wall_ms` | `float` | High-resolution wall-clock duration of packing and verification in milliseconds (`time.perf_counter()`). Strictly positive (`> 0.0`). |
| `raw_bytes` | `int` | Sum of uncompressed byte lengths across all packed member files (`sum(len(raw))`). |
| `compressed_bytes` | `int` | Exact byte length of the resulting `members.tar.zst` archive file on disk (`os.path.getsize()`). |
| `compression_ratio` | `float` | Effective compression ratio: `raw_bytes / compressed_bytes` (or `1.0` if `compressed_bytes == 0`). Rounded to 4 decimal places. |
| `member_count` | `int` | Total number of individual files packaged into the archive (`len(members)`). |
| `backend` | `str` | Active compression engine: `"zstandard"` (Python C-extension), `"zstd-cli"` (system CLI binary), or `"stdlib"` (PEP 784). |

---

## 3. Producer Implementation & Stdlib Isolation

### 3.1 Strict Stdlib Isolation in `_node.py`
`src/xtrax/pack/_node.py` is embedded into remote cluster batch scripts via `node_source()`. It must execute in minimal compute node environments running bare Python 3.9 without `xtrax`, `numpy`, or `cisternal`.
- All telemetry timing uses standard library `time.perf_counter()`.
- Uncompressed and compressed byte calculations use standard library `len(raw)` and `os.path.getsize(pack_path)`.
- Backend resolution maps internal `_codec()` names:
  ```python
  kind, _ = _codec()
  backend = "zstandard" if kind == "zstandard" else ("zstd-cli" if kind == "cli" else kind)
  ```
- No external packages are imported at top-level or inside function bodies.

### 3.2 Telemetry Bridge Outside the Node Boundary
When `pack_unit` is invoked from within the full `xtrax` Python library (`src/xtrax/pack/__init__.py`), a lightweight bridge allows emitting telemetry events into `xtrax.telemetry` or optional `cisternal` sinks if configured:
- `xtrax.pack.register_telemetry_hook(callback)` allows registering a listener callable.
- If `cisternal.telemetry.emit_event` is available, `xtrax.pack` delegates event delivery safely within a try-except block without introducing runtime dependencies.

---

## 4. Consumer Implementation & Campaign EDA Ingestion

### 4.1 Ingestion via Reconciliation Records
`xtrax.eda.campaign.reader.CampaignReader` reconciles persistence status for each manifest row.
1. When checking completion markers (`done.json` or `unit.json`), `reader.py` extracts `pack_telemetry` if present.
2. `ReconciliationRecord` carries `pack_telemetry: Mapping[str, Any] | None = None`.
3. In `xtrax.eda.campaign.telemetry.extract_campaign_telemetry`, per-unit records pass telemetry to the aggregation engine.

### 4.2 Campaign Aggregation States
Campaign-level pack telemetry transitions deterministically between four states:

```
                    ┌───────────────────────────┐
                    │  packed_count == 0?       │
                    └─────────────┬─────────────┘
                                  │
                   YES ───────────┴─────────── NO
                    │                           │
                    ▼                           ▼
         ┌─────────────────────┐    ┌───────────────────────────┐
         │   no_packed_units   │    │  instrumented_count == 0? │
         └─────────────────────┘    └─────────────┬─────────────┘
                                                  │
                                   YES ───────────┴─────────── NO
                                    │                           │
                                    ▼                           ▼
                         ┌─────────────────────┐    ┌───────────────────────────┐
                         │   uninstrumented    │    │ instrumented == packed?   │
                         └─────────────────────┘    └─────────────┬─────────────┘
                                                                  │
                                                   YES ───────────┴─────────── NO
                                                    │                           │
                                                    ▼                           ▼
                                         ┌─────────────────────┐    ┌─────────────────────┐
                                         │    instrumented     │    │        mixed        │
                                         └─────────────────────┘    └─────────────────────┘
```

1. **`"no_packed_units"`**: The campaign contains zero packed units (`packed_units == 0`). All units are loose directories or uncommitted.
   - `pack_telemetry_banner`: `"Pack Telemetry: no packed units in campaign"` (or `None`).
   - Summary statistics reflect zero packed bytes.
2. **`"instrumented"`**: One or more units are packed (`packed_units > 0`), and 100% of packed units provide valid `pack_telemetry`.
   - `pack_telemetry_banner`: Replaced with an instrumented summary string, e.g.:
     `"Pack Telemetry: instrumented (9 units, 3.85x compression, mean 14.2ms, zstandard)"`.
   - `pack_telemetry_status`: `"instrumented"`.
   - Full aggregate statistics populated.
3. **`"mixed"`**: Multiple units are packed (`packed_units > 0`), but only a subset ($0 < K < M$) have `pack_telemetry`.
   - `pack_telemetry_banner`: Graceful warning banner:
     `"Pack Telemetry: mixed (K of M units instrumented, debt #3300)"`.
   - `pack_telemetry_status`: `"mixed"`.
   - Partial aggregates computed over the $K$ instrumented units.
4. **`"uninstrumented"`**: Packed units exist (`packed_units > 0`), but zero units carry `pack_telemetry` (legacy units).
   - `pack_telemetry_banner`: `"Pack Telemetry: not instrumented (debt #3300)"`.
   - `pack_telemetry_status`: `"uninstrumented"`.
   - Preserves complete backward compatibility with existing tests and fixtures.

### 4.3 Aggregate Statistics
When `state` is `"instrumented"` or `"mixed"`, `extract_campaign_telemetry` computes:
- `total_raw_bytes`: `sum(u["raw_bytes"] for u in units)`
- `total_compressed_bytes`: `sum(u["compressed_bytes"] for u in units)`
- `overall_compression_ratio`: `total_raw_bytes / total_compressed_bytes` if `total_compressed_bytes > 0` else `1.0`
- `mean_pack_wall_ms`: arithmetic mean of `pack_wall_ms` across instrumented units
- `backends`: sorted list of distinct compression backends observed across units

### 4.4 Neutral Visualization & Artifact Updates
In `src/xtrax/eda/campaign/artifacts.py`:
- `summary.json`: Top-level `q5_telemetry` includes `pack_telemetry` block and aggregates.
- `specs/run_telemetry.vl.json` and `specs/q5_telemetry.vl.json`: Dynamic title reflecting status (`"Question 5: Secondary Run Telemetry (Status: instrumented)"`).
- `static/run_telemetry.svg` and `static/q5_telemetry.svg`: Rendered static vector graphics.
- `index.html`: When instrumented, `.telemetry-banner.telemetry-instrumented` displays an informational status bar with a green accent rather than an amber warning.

---

## 5. Verification & Testing Protocol

1. **Standalone Node Tests (`tests/pack/test_node_standalone.py`)**:
   - Verify `_node.py` imports strictly within `sys.stdlib_module_names`.
   - Verify source text matches `node_source()`.
2. **Pack Producer Tests (`tests/pack/test_pack.py`)**:
   - Verify `pack_unit()` writes `record["pack_telemetry"]` with schema `"xtrax.pack_telemetry/1"`.
   - Verify `pack_wall_ms > 0`, `raw_bytes >= compressed_bytes > 0`, `compression_ratio > 0`, and valid `backend`.
   - Verify `test_writes_what_alynxrs_reader_needs` still passes (`set(storage) == set(theirs)`).
   - Verify alynxr fixtures and tampering tests pass.
3. **Consumer & Q5 Tests (`tests/test_campaign_eda/test_coverage_telemetry.py`)**:
   - Test legacy uninstrumented units -> returns `PACK_TELEMETRY_BANNER` and status `"uninstrumented"`.
   - Test fully instrumented units -> banner replaced, status `"instrumented"`, aggregates verified.
   - Test mixed units -> status `"mixed"` and fallback banner.
   - Test campaign with zero packed units -> status `"no_packed_units"`.
4. **End-to-End Demo Campaign (`tests/test_campaign_eda/test_demo_campaign.py`)**:
   - Validate full 18-unit ubiquitin demo campaign with 9 packed units emitting telemetry.
   - Verify `summary.json`, `q5_telemetry.vl.json`, `q5_telemetry.svg`, `run_telemetry.vl.json`, and `run_telemetry.svg` are generated and hashed in `manifest.json`.
