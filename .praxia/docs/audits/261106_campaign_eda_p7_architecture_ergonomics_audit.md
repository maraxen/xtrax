---
category: audit
title: 'Phase P7 Swarm Review: Architecture, Plugin Ergonomics & Presentation Audit'
description: Independent verification of DomainPlugin protocols, zero-copy array immutability,
  exception quarantine, test suites, and Vega-Lite/HTML progressive rendering
task_id: 261106_xtrax-campaign-eda
status: complete
date: '2026-11-06'
agent: teamwork_critic_p7_1
verdict: APPROVE
---

# Independent Swarm Review Report: Architecture & Plugin Ergonomics (Phase P7)

**Reviewer**: `teamwork_critic_p7_1`  
**Role**: Independent Architecture, Plugin Ergonomics & Presentation Swarm Reviewer  
**Target Milestone**: Phase P7 Review (Campaign EDA Redesign, Revision 1.1)  
**Parent Agent**: `8c6c9b8f-26e4-41f2-9b35-adf51fb4ef0f`  
**Date**: 2026-11-06T21:51:00Z  
**Verdict**: **APPROVE**

---

## 1. Observation

### 1.1 Test Suite & Verification Controls Execution
1. **Targeted Pytest Suite (`test_plugins.py`, `test_coverage_telemetry.py`, `test_artifacts.py`)**:
   - Command: `uv run pytest tests/test_campaign_eda/test_plugins.py tests/test_campaign_eda/test_coverage_telemetry.py tests/test_campaign_eda/test_artifacts.py -v --no-cov`
   - Output: 9 passed, 12 warnings (PEP 585 deprecation in third-party beartype) in 0.77s.
2. **Complete Campaign EDA Test Suite (`tests/test_campaign_eda/`)**:
   - Command: `uv run pytest tests/test_campaign_eda/ -v --no-cov`
   - Output: 24 passed in 1.69s.
3. **Pre-Registered Formal Benchmark Script (`scripts/benchmarks/campaign_eda_benchmark.py`)**:
   - Command: `uv run python3 scripts/benchmarks/campaign_eda_benchmark.py --out /tmp/bench.json`
   - Output:
     - Control 1 (Positive Planted Knob Effect): Passed (`\hat{\Delta} = 1.488 \in [1.35, 1.65]`, $p < 0.05$).
     - Control 2 (Negative Null Campaign): Passed ($\text{FPR} = 0.04 \le 0.05$).
     - Control 3 (Vectorized Bootstrap Parity): Passed ($\max|\text{diff}| \le 2.66 \times 10^{-15} \le 10^{-12}$).
     - Control 4 (Storage Format Parity): Passed (exact SHA-256 and NumPy array equality).
     - Control 5 (Uncommitted Unit Refusal): Passed (100% exclusion of partial and failed units).
     - Control 6 (Small-Sample Guard): Passed ($U < 3 \implies \text{descriptive\_only} = \text{True}$).
     - Ubiquitin Demo Campaign: Passed (18 units, 288 sequences, runtime 0.083s $\le$ 60.0s budget).
4. **Linter & Code Hygiene**:
   - Command: `uv run ruff check src/xtrax/eda/campaign/ tests/test_campaign_eda/`
   - Output: `All checks passed!` (0 violations).
5. **Pack Format Regression Suite**:
   - Command: `uv run pytest tests/pack/test_pack.py -v --no-cov`
   - Output: 111 passed in 2.09s.

---

### 1.2 Domain Plugin Architecture & Quarantine Isolation
1. **DomainPlugin Protocol Definition**:
   - Location: `src/xtrax/eda/campaign/plugins.py:30-56`
   - Protocol declares `@runtime_checkable class DomainPlugin(Protocol)` with properties `domain_name: str`, `version: str`, and methods `extract_cell(store_path: Path, manifest_row: ManifestRow, sink_schema: SinkSchema) -> Sequence[LibraryCell]`, `compute_metrics(cell: LibraryCell) -> DomainMetricResult`.
2. **Dynamic Entry-Point Discovery**:
   - Location: `src/xtrax/eda/campaign/plugins.py:110-125`
   - Discovers plugins via `importlib.metadata.entry_points(group="xtrax.eda.domains")`.
   - Safely loads classes and verifies protocol conformance via `isinstance(plugin_inst, DomainPlugin)`.
   - Seamlessly merges dynamic plugins with explicit registry `_PLUGIN_REGISTRY`.
3. **Zero-Copy Array Immutability (MAJ-01)**:
   - Location: `src/xtrax/eda/campaign/plugins.py:58-74`
   - `enforce_cell_immutability()` sets `flags.writeable = False` on `cell.tokens`, `cell.mask`, `cell.logits`, `cell.scores`, and `cell.residues.real_mask`.
   - Verified that sliced views inherit read-only status (`b = a[1:3]; b.flags.writeable == False`), preventing in-place mutations at 0% memory copy overhead.
   - Tested in `tests/test_campaign_eda/test_plugins.py:56-74`.
4. **Defensive Quarantine Boundary (BLK-06)**:
   - Location: `src/xtrax/eda/campaign/plugins.py:128-160`
   - `execute_domain_plugin` wraps extraction and metric evaluation in `try: ... except (Exception, SystemExit) as exc:`.
   - Catches `SystemExit` (rogue process termination attempts) and records failure in `QuarantineRecord` via `quarantine_unit()`.
   - Tested in `tests/test_campaign_eda/test_plugins.py:75-128`.
5. **aminx Domain Plugin (`src/xtrax/eda/campaign/domains/aminx.py` & `/tmp/aminx/src/aminx/eda/plugin.py`)**:
   - Location: `src/xtrax/eda/campaign/domains/aminx.py:117-125, 137-155, 237-263`
   - Designable masking enforced as `real_mask & ~fixed_mask`.
   - Derived NLL fallback implemented via numerically stabilized log-sum-exp:
     $$\text{LSE}(Z_{l}) - Z_{l, s_l} = \max(Z_l) + \log\left(\sum_a \exp(Z_{l, a} - \max(Z_l))\right) - Z_{l, s_l}$$
   - Perplexity computed as $\exp(\text{mean}(\text{nll}))$.
   - Native recovery margin evaluates composition-preserving Fisher-Yates permutation of native sequence, flagging library collapse when $\text{margin} < 0.10$.
6. **demistify Domain Plugin (`src/xtrax/eda/campaign/domains/demistify.py` & `teamwork_worker_demistify_1/plugin.py`)**:
   - Location: `src/xtrax/eda/campaign/domains/demistify.py:108-120` & `teamwork_worker_demistify_1/plugin.py:192-280, 423-428`
   - Extracts `states_traj`, marginal entropy $S_i$, total entropy $S_{total}$, dense pairwise MI matrix, and MIST spanning tree edges.
   - Schema B graceful degradation: detects `result.zarr` without dense MI matrix, flags `matrix_fullness = False`, emits informational banner `"Schema B: dense MI matrix omitted; allosteric rankings derived from MIST spanning tree."`, and falls back to weighted tree node degree `der.hub_degree@1`.

---

### 1.3 Storage Parity & Refusal Subsystem
1. **Dual-Marker Discovery (BLK-07)**:
   - Location: `src/xtrax/eda/campaign/reader.py:39-56, 71-89, 243-263`
   - `get_marker_record_name()` checks for `done.json` (campaign runner marker) or `unit.json` (pack marker).
   - `read_done_marker()` transparently reads from `done.json`, `unit.json`, or `<unit_dir>.done.json`.
   - Forwards detected `record_name` to `xtrax.pack.read_unit_member()`.
2. **Storage Format Parity (Loose Files vs `.tar.zst` Archives)**:
   - Location: `tests/test_campaign_eda/test_controls.py:176-215`
   - Proven bit-identical SHA-256 digests and exact NumPy array values across loose and packed units.
3. **Strict Refusal of Uncommitted & Quarantined Units**:
   - Location: `src/xtrax/eda/campaign/reader.py:126-309, 319-344`
   - `reconcile_row()` strictly classifies units into `COMMITTED`, `PARTIAL` (active staging `.tmp` or active lease lock), `FAILED` (expired lease, missing marker, marker status failed, hash mismatch), or `MISSING`.
   - `get_committed_units()` filters out non-committed units and explicitly checks `quarantined_keys`, refusing quarantined units from downstream statistical pooling.
4. **100% Committed Visual Aggregation Rollup Rule (MAJ-04)**:
   - Location: `src/xtrax/eda/campaign/coverage.py:20-47`
   - `aggregate_visual_cell_status()` enforces:
     - `COMMITTED` iff 100% of constituent rows are `COMMITTED`.
     - `FAILED` if any constituent row is `FAILED` or quarantined.
     - `PARTIAL` if at least one row is `PARTIAL` (or mixed missing/committed without failures).
     - `MISSING` iff 100% of rows are `MISSING`.
   - Tested in `tests/test_campaign_eda/test_coverage_telemetry.py:9-39`.

---

### 1.4 Dependency Boundaries & Hygiene
1. **Core Package Dependency Footprint**:
   - Confirmed `src/xtrax/eda/campaign/` depends exclusively on Python standard library and NumPy (and internal `xtrax.campaign` / `xtrax.pack`).
2. **Optional Extras in `pyproject.toml`**:
   - Location: `pyproject.toml:71-73`
     ```toml
     eda-campaign = [ "duckdb>=1.0.0", "pyarrow>=14.0.0",]
     eda-viz = [ "altair>=5.0.0", "vl-convert-python>=1.3.0",]
     eda-all = [ "xtrax[eda-campaign]", "xtrax[eda-viz]",]
     ```
   - In `artifacts.py:503-556`, `pyarrow` is imported lazily with automatic fallback to JSON tables if absent.
   - In `artifacts.py:305-314`, `vl_convert` is imported lazily with automatic fallback to standalone vector SVG if absent.
3. **Cross-Import & Framework Isolation**:
   - Searched `src/xtrax/eda/` for `maraxiom`, `bathos`, and `cisternal`: 0 matches found.
   - Contract declared in `pyproject.toml:276-285`:
     ```toml
     [[tool.importlinter.contracts]]
     name = "Campaign EDA isolation from external frameworks"
     type = "forbidden"
     source_modules = ["xtrax.eda"]
     forbidden_modules = ["maraxiom", "bathos", "cisternal"]
     ```

---

### 1.5 Artifact Presentation Layer
1. **RFC 8259 JSON Sanitization (BLK-05)**:
   - Location: `src/xtrax/eda/campaign/artifacts.py:23-63`
   - `sanitize_for_rfc8259()` recursively coerces `NaN` and `+/-Inf` to `None` (`null`), and `-0.0` to `0.0`.
   - `dumps_canonical_json()` enforces `allow_nan=False`, guaranteeing strictly standard RFC 8259 JSON output.
2. **Vega-Lite Specifications with Null Filters**:
   - Location: `src/xtrax/eda/campaign/artifacts.py:92, 113, 132, 145, 195`
   - Every quantitative visualization specification includes explicit transform filters (e.g. `transform: [{"filter": "isValid(datum.diversity) && datum.diversity != null"}]`).
3. **Progressive Airgapped HTML Dashboard (MAJ-08)**:
   - Location: `src/xtrax/eda/campaign/artifacts.py:321-450`
   - Standalone pre-rendered static SVGs visible by default offline (`display: block`).
   - Unique DOM IDs per panel: `fallback-svg-{panel_name}` and `interactive-vis-{panel_name}`.
   - Deferred script loading (`<script defer>`).
   - Client-side 2000ms timeout guard to preserve static vector Reviewed Object offline if CDN is unreachable.
4. **Secondary Run View & Debt #3300 Pack Telemetry Seam Banner**:
   - Location: `src/xtrax/eda/campaign/telemetry.py:22-23, 93` & `src/xtrax/eda/campaign/artifacts.py:380-383`
   - Exports `PACK_TELEMETRY_BANNER = "Pack Telemetry: not instrumented (debt #3300)"`.
   - Displayed prominently in HTML dashboard with amber alert box styling.

---

## 2. Logic Chain

1. **Protocol & Isolation Integrity** [Observation 1.2]:
   - By defining `@runtime_checkable class DomainPlugin(Protocol)` and dynamically discovering implementations via `importlib.metadata.entry_points(group="xtrax.eda.domains")`, consumer repositories (`aminx`, `demistify`) plug in without core code modifications.
   - Enforcing `flags.writeable = False` on all ndarrays before passing `LibraryCell` to plugins guarantees zero-copy immutability (MAJ-01). Because sliced views in NumPy inherit `flags.writeable = False`, plugins cannot mutate inputs in place without triggering an immediate `ValueError`.
   - Catching `(Exception, SystemExit)` in `execute_domain_plugin` (BLK-06) prevents rogue or crashing plugins from terminating the parent analysis runner, while `QuarantineRecord` safely preserves diagnostic details.

2. **Storage Equivalence & Safety** [Observation 1.3]:
   - Transparent dual-marker discovery (`done.json` and `unit.json`) in `reader.py` resolves marker naming discrepancies between campaign runner units and packed units (BLK-07).
   - Control 4 demonstrates bit-identical reads across loose and packed units, ensuring analysis output is independent of physical archive state.
   - Restricting statistical pooling to `CellStatus.COMMITTED` units and filtering out quarantined keys guarantees uncommitted or corrupt data never contaminates scientific conclusions.
   - Aggregating grid cell statuses with the 100% committed rollup rule (MAJ-04) prevents incomplete sweep conditions from falsely masquerading as finished.

3. **Dependency Hygiene & Modular Decoupling** [Observation 1.4]:
   - Core campaign EDA relies strictly on Python stdlib and NumPy.
   - Heavy columnar and visualization packages (DuckDB, PyArrow, Altair, vl-convert) are cleanly isolated behind optional extras (`eda-campaign`, `eda-viz`, `eda-all`), with runtime fallbacks ensuring core functionality works in minimal environments.
   - Zero imports of `maraxiom`, `bathos`, or `cisternal` inside `xtrax.eda` preserve clean architectural separation.

4. **Standards-Compliant Visual Artifacts** [Observation 1.5]:
   - `sanitize_for_rfc8259()` with `allow_nan=False` eliminates invalid `NaN`/`Infinity` tokens, adhering strictly to RFC 8259 (BLK-05).
   - Vega-Lite null filters prevent rendering artifacts or browser console warnings on sparse data.
   - Airgapped progressive HTML dashboard with unique panel DOM container IDs (`fallback-svg-{panel_name}`) and deferred script loading ensures offline readability without layout shifts (MAJ-08).
   - Telemetry seam debt #3300 is transparently surfaced and bannered without conflating operational metrics with scientific findings.

---

## 3. Caveats / Tech Debt Items

1. **Debt #3300 Pack Telemetry Seam**:
   - `pack_events` telemetry is explicitly marked as `"not instrumented (debt #3300)"`. This is the intended design for Phase P5/P7 and is clearly bannered across telemetry JSON and the HTML dashboard.
2. **Third-Party Deprecation Warnings in Test Environment**:
   - Pytest execution emits minor `BeartypeDecorHintPep585DeprecationWarning` notices originating from upstream `beartype` typing helpers in Python 3.12/3.13. These warnings do not originate from `xtrax.eda.campaign` and do not impact functionality.

---

## 4. Conclusion

**Verdict: APPROVE**

The architectural boundaries, domain plugin ergonomics, storage parity mechanisms, and presentation layers in `src/xtrax/eda/campaign/` (alongside consumer plugins in `aminx` and `demistify`) strictly satisfy all requirements of Revision 1.1 specification:
- Protocol isolation, dynamic discovery, and defensive quarantine boundaries function without flaw.
- Array immutability (MAJ-01) and host resilience against `SystemExit` (BLK-06) are verified.
- Storage format parity, dual-marker resolution (BLK-07), uncommitted unit refusal, and the 100% committed visual rollup rule (MAJ-04) are deterministically enforced.
- Dependency boundaries remain strictly `stdlib + numpy` in core, with zero cross-imports to `maraxiom`, `bathos`, or `cisternal`.
- Presentation artifacts conform strictly to RFC 8259 (BLK-05) and progressive airgapped HTML rendering standards (MAJ-08).

All 24 targeted tests, the 6 synthetic verification controls, and the end-to-end Ubiquitin demo campaign pass with 100% success. The capability is approved for Phase P8 Victory Audit.

---

## 5. Verification Method

To independently reproduce this verification:

```bash
# 1. Run targeted Campaign EDA unit and integration tests (24 passed)
uv run pytest tests/test_campaign_eda/ -v --no-cov

# 2. Run the 6 pre-registered synthetic controls and Ubiquitin demo benchmark
uv run python3 scripts/benchmarks/campaign_eda_benchmark.py --out /tmp/bench.json

# 3. Verify linting compliance (0 errors)
uv run ruff check src/xtrax/eda/campaign/ tests/test_campaign_eda/

# 4. Run pack format regression tests (111 passed)
uv run pytest tests/pack/test_pack.py -v --no-cov

# 5. Verify zero cross-imports of external frameworks
grep -rnE "(maraxiom|bathos|cisternal)" src/xtrax/eda/campaign/
# Expected: 0 matches
```
