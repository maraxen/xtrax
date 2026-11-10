---
task_id: 261109_pack-telemetry-and-eda-pr-swarm
title: Three-PR Adversarial Swarm Code Review and Forensic Audit
date: 2026-11-09
verdict: REQUEST_CHANGES
repos:
  - xtrax (PR #269, feat/261106-campaign-eda)
  - aminx (PR #277, feat/261106-campaign-eda-plugin)
  - demistify (PR #91, feat/261106-campaign-eda-plugin)
reviewers:
  - Reviewer A (Scientific Fitness-for-Purpose & Statistical Rigor)
  - Reviewer B (Cross-Repo Contract, Plugin Boundary & Security/Quarantine)
  - Reviewer C (Storage Format Parity, Pack Telemetry & CI Hygiene)
---

# Three-PR Adversarial Swarm Code Review and Forensic Audit

**Task ID**: `261109_pack-telemetry-and-eda-pr-swarm`  
**Date**: 2026-11-09  
**Final Swarm Verdict**: **REQUEST_CHANGES**  
**Integrity Mode**: `development`  

---

## 1. Executive Summary & Findings Matrix

This document consolidates the findings of an adversarial, multi-agent swarm code review and forensic audit conducted across three interdependent pull requests:
1. **`xtrax` PR #269** (`feat/261106-campaign-eda` vs `main`): Campaign-level exploratory data analysis engine, Rao-Wu cluster ratio bootstrap, Best-of-$N$ order statistics recurrence, and pack telemetry ingestion seam (Debt #3300).
2. **`aminx` PR #277** (`feat/261106-campaign-eda-plugin` vs `main`): Generative protein sequence design domain plugin, sequence extraction, NLL evaluation, recovery margins, and positional entropy.
3. **`demistify` PR #91** (`feat/261106-campaign-eda-plugin` vs `main`): Conformational entropy and physical allosteric network domain plugin, MD trajectory parsing, CI-MIST mutual information, and packed Zarr storage.

Three specialized review agents evaluated the changeset concurrently across three independent analytical lenses:
- **Lens A (Scientific Fitness-for-Purpose & Statistical Rigor)**: Audit of the 5 scientist questions (Q1–Q5), cluster ratio bootstrap mathematics, replicate vs condition pooling, sample size accounting, and numeric vectorization.
- **Lens B (Cross-Repo Contract, Plugin Boundary & Security/Quarantine)**: Audit of `DomainPlugin` and `DomainMetricResult` protocols, entry-point discovery, zero-copy immutability, defensive quarantine boundaries, and `import-linter` isolation.
- **Lens C (Storage Format Parity, Pack Telemetry & CI Hygiene)**: Audit of loose vs packed `.tar.zst` storage parity, dual-marker resolution (`done.json` vs `unit.json`), `_node.py` stdlib isolation, Pack Telemetry seam design (Debt #3300), and multi-repo CI gates.

### 1.1 High-Level Audit Verdict

The consolidated verdict of the swarm review is **REQUEST_CHANGES**.

While the statistical foundations (the Rao-Wu cluster ratio bootstrap in `xtrax.eda.campaign.stats.bootstrap` and the Best-of-$N$ order statistics recurrence in `scores.py`) are mathematically sound, the cross-repo integration and execution layers suffer from **one CRITICAL INTEGRITY VIOLATION**, **seven BLOCKERS**, **thirteen MAJORS**, and **five MINORS**.

The critical finding (**F-LENB-15**) represents an explicit integrity violation: `src/xtrax/eda/campaign/domains/aminx.py` contains a hardcoded ubiquitin sequence fallback that injects fake experimental sequences whenever a campaign unit has missing data, bypassing quarantine and falsifying scientific metrics.

Furthermore, three orthogonal defects completely block campaign execution in production:
1. `aminx` emits `"score_nll"` and `"recovery_shuffled_native"`, whereas `xtrax` queries `"nll"` and `"recovery_shuffled"`, leading to silent data annihilation where 100% of energetic distributions and bootstrap uncertainty collapse to 0.0.
2. `aminx`'s storage loader assumes an uncompressed Zarr group and crashes with `GroupNotFoundError` on standard campaign units (loose `.npy` and packed `.tar.zst`), causing 100% of aminx units to enter quarantine.
3. `demistify` omits `record_name=record_name` when calling `is_packed` and `PackedIndex`, causing 100% of packed units marked with `done.json` to fail with `FileNotFoundError: No states_traj found in unit`.

### 1.2 Consolidated Findings Summary by Repository and Severity

| Repository | CRITICAL (Integrity) | BLOCKER | MAJOR | MINOR | Total |
|:---|:---:|:---:|:---:|:---:|:---:|
| **`xtrax`** | 1 | 2 | 7 | 2 | **12** |
| **`aminx`** | 0 | 3 | 3 | 2 | **8** |
| **`demistify`** | 0 | 2 | 3 | 2 | **7** |
| **Total** | **1** | **7** | **13** | **6** | **27** |

*(Note: Certain cross-cutting findings span multiple repositories; table counts reflect primary affected repos).*

---

### 1.3 Consolidated Findings Master Table

| ID | Title | Severity | Lens | Target Repo & Location | Summary of Defect |
|:---|:---|:---:|:---:|:---|:---|
| **F-LENB-15** | Hardcoded Ubiquitin Fallback Bypasses Defensive Quarantine | **CRITICAL (INTEGRITY)** | Lens B | `xtrax`: `src/xtrax/eda/campaign/domains/aminx.py:85-88` | Fabricates fake 18-mer ubiquitin sequence `MQIFVKTLTGKTITLEVE` when unit is empty, falsifying metrics and bypassing quarantine. |
| **F-BLK-01** | Metric & Flag Key Naming Mismatch Causes Silent Campaign Data Annihilation | **BLOCKER** | Lens A / B | `aminx`: `src/aminx/eda/plugin.py:598-605`<br>`xtrax`: `__init__.py:206, 244` | aminx emits `score_nll`/`recovery_shuffled_native`/`score_derived`; xtrax queries `nll`/`recovery_shuffled`/`is_score_derived`. All Q2 NLLs collapse to 0.0. |
| **F-BLK-02** | Storage Parity Failure in aminx `_load_sink_data` | **BLOCKER** | Lens A / B / C | `aminx`: `src/aminx/eda/plugin.py:200-234` | Calls `zarr.open_group` unconditionally on store paths. Crashes with `GroupNotFoundError` on loose `.npy` dirs and packed `.tar.zst`. |
| **F-BLK-03** | Dual-Marker Ingestion Bug in demistify `_read_member_bytes` / `_list_unit_relpaths` | **BLOCKER** | Lens A / B / C | `demistify`: `src/demistify/eda/plugin.py:116, 133` | Invokes `is_packed(store_path)` without `record_name=record_name`, defaulting to `unit.json`. Fails 100% of packed `done.json` units. |
| **F-BLK-04** | Dual-Marker Omission in demistify `PackedZarrStore` | **BLOCKER** | Lens A / C | `demistify`: `src/demistify/io/zarr_io.py:1087`<br>`src/demistify/unit_pack.py:488` | Instantiates `PackedIndex(self.unit_dir)` without `record_name`, raising `UnitPackRefused(R-U1)` on `done.json` Zarr units. |
| **F-BLK-05** | Workspace Working Tree Desynchronization in aminx | **BLOCKER** | Lens A / B / C | `aminx`: Local working tree (`/home/marielle/projects/aminx`) | Local branch sits on `main` (`86042c0e`) without PR #277 commits (`05174ed6`). Files missing on disk, plugin discovery fails, pytest fails. |
| **F-BLK-06** | `import-linter` Configuration Failure Crashes `lint-imports` | **BLOCKER** | Lens B | `xtrax`: `pyproject.toml:203-205` | Missing `include_external_packages = true` crashes linter when external forbidden modules (`maraxiom`, `bathos`, etc.) are declared. |
| **F-BLK-07** | Hardcoded Generative Metrics in xtrax Break Non-Generative Domains | **BLOCKER** | Lens A / B | `xtrax`: `src/xtrax/eda/campaign/__init__.py:184-250` | Hardcodes aminx generative metrics (`diversity`, `nll`). Demistify observables (`total_entropy`, `hub_score@1`) are discarded and zeroed. |
| **F-MAJ-01** | Artificial Truncation of `LibraryCell.sequences` at 100 Frames | **MAJOR** | Lens A / B | `demistify`: `src/demistify/eda/plugin.py:382`<br>`xtrax`: `__init__.py:242` | Sequences truncated to `min(T, 100)`. Distorts cluster ratio bootstrap weights by up to 1,000x; treats unequal MD runs as identical cluster sizes. |
| **F-MAJ-02** | Replicate Conflation with Knob Levels in Condition Derivation | **MAJOR** | Lens A | `xtrax`: `src/xtrax/eda/campaign/__init__.py:173, 205, 232` | Condition derived as `f"{arm}_T{temp}"`. Variations in checkpoints, policies, or noise are lumped together as stochastic replicates, inflating variance. |
| **F-MAJ-03** | Non-Vectorized Python Loops in NLL Evaluation & Positional Entropy | **MAJOR** | Lens A | `xtrax`: `domains/aminx.py:142-153, 205-210`<br>`aminx`: `src/aminx/eda/plugin.py:504-518` | Nested Python loops over $N \times L$ evaluating LogSumExp and column-by-column entropy instead of vectorized 2D NumPy array operations. |
| **F-MAJ-04** | Unguarded Zero-Sample Division and NaN Leaks in Bootstrap | **MAJOR** | Lens A | `xtrax`: `src/xtrax/eda/campaign/stats/bootstrap.py:48-54, 108, 164` | `np.sum(n_vec) == 0` triggers `ZeroDivisionError`; resampled zero-count clusters leak `NaN` into ratio denominator and `ci_95`. |
| **F-MAJ-05** | Per-Residue Track Arrays in `DomainMetricResult.tracks` Remain Mutable | **MAJOR** | Lens B | `xtrax`: `src/xtrax/eda/campaign/plugins.py:58-74, 146` | Immutability enforcement ignores `tracks`. Arrays remain `writeable=True`, allowing callers or export routines to mutate shared cached arrays. |
| **F-MAJ-06** | Quarantine Leaks on Discovery `SystemExit` and Exploding Properties | **MAJOR** | Lens B | `xtrax`: `src/xtrax/eda/campaign/plugins.py:121, 157` | `discover_domain_plugins()` fails to catch `SystemExit`. Exception handler accesses `plugin.domain_name` unguarded, crashing runner. |
| **F-MAJ-07** | Permissive `validate_schema` in Consumer Plugins & Missing Track Schema | **MAJOR** | Lens A / B | `aminx`: `plugin.py:89`<br>`demistify`: `plugin.py:73`<br>`xtrax`: `types.py:63-88` | Consumer plugins omit string key and boolean flag checks. `DomainMetricResult.validate_schema()` performs zero validation on `tracks`. |
| **F-MAJ-08** | Dataclass Duplication Breaks Nominal Type Compatibility | **MAJOR** | Lens B | `aminx`: `plugin.py:49-102`<br>`demistify`: `plugin.py:31-87` | Consumers duplicate `LibraryCell`, `DomainMetricResult` instead of importing from `xtrax.eda.campaign.types`. `isinstance` fails. |
| **F-MAJ-09** | Control 4 Storage Format Parity Mock Shortcut in Benchmark | **MAJOR** | Lens C | `xtrax`: `scripts/benchmarks/campaign_eda_benchmark.py:276-361` | Control 4 tests toy tarfile/zstd round-trip on dummy JSON. Never calls `pack_unit()`, `read_campaign_unit_member()`, or `extract_cell()`. |
| **F-MAJ-10** | Mock `uv` Sandbox Trap in Standalone Node Test | **MAJOR** | Lens C | `xtrax`: `tests/pack/test_node_standalone.py:48-56` | `_python39()` treats `"Mock uv for sandbox environment"` as an executable path without checking `Path.is_file()`, crashing `subprocess.run`. |
| **F-MAJ-11** | Ruff UP Rules Flag Python 3.9 Compatibility in `_node.py` | **MAJOR** | Lens C | `xtrax`: `pyproject.toml:23-28`<br>`src/xtrax/pack/_node.py` | Ruff `UP` rules flag Python 3.9 typing in standalone node (128 errors). Auto-fixing would destroy embedded cluster compatibility. |
| **F-MAJ-12** | 28 Ruff Lint Violations in `aminx.eda.plugin` | **MAJOR** | Lens C | `aminx`: `src/aminx/eda/plugin.py` | Fails CI checks with 28 violations (`ANN401`, `PLC0415`, `BLE001`, `ARG002`, `SIM102`, `PLR0915`). |
| **F-MAJ-13** | Architectural Boundary Gap: Linter Does Not Forbid Consumer Imports | **MAJOR** | Lens B | `xtrax`: `pyproject.toml:276-285` | Contract 4 forbids external frameworks but omits consumer libraries (`aminx`, `demistify`, `adaptyv`), risking reverse dependencies. |
| **F-MIN-01** | Omission of ANOVA / Feature Importance for Scientist Question 4 | **MINOR** | Lens A | `xtrax`: `src/xtrax/eda/campaign/` | Scientist Question 4 ("Knobs That Matter") is replaced entirely by sweep coverage reconciliation with zero variance attribution across knobs. |
| **F-MIN-02** | Stochastic Shuffled Native Baseline Seed in aminx | **MINOR** | Lens A | `aminx`: `src/aminx/eda/plugin.py:549` | Shuffled baseline uses `rng = np.random.default_rng(cell.seed)`, adding extraneous stochastic noise to `recovery_margin` across replicates. |
| **F-MIN-03** | Ruff Format Check Failures Across `demistify` and `xtrax` | **MINOR** | Lens C | `demistify`: 2 files<br>`xtrax`: 2 files | `ruff format --check` fails in CI. `xtrax/pyproject.toml` lacks `extend-exclude = [".agents"]`. |
| **F-MIN-04** | Missing Type Annotations / 56 `ty check` Diagnostics in demistify | **MINOR** | Lens C | `demistify`: `src/demistify/eda/plugin.py:244-277` | 56 diagnostics emitted under `ty check`: unimported `zarr.storage`, Zarr v3 indexing unions, and uncast float conversions. |
| **F-MIN-05** | Zero Test Coverage for Packed `done.json` Units in demistify | **MINOR** | Lens C | `demistify`: `tests/test_eda_plugin.py` | All test fixtures construct units with default `unit.json`, failing to exercise the dual-marker execution path in CI. |
| **F-MIN-06** | Entry-Point Discovery Missing Caching in Campaign EDA | **MINOR** | Lens B | `xtrax`: `src/xtrax/eda/campaign/plugins.py:110-125` | Uncached entry-point scan on every call adds filesystem overhead; lacks `_DOMAIN_PLUGIN_CACHE` and `reload_domain_plugins()`. |

---

## 2. Detailed Section: CRITICAL / INTEGRITY VIOLATION

### F-LENB-15: Hardcoded Ubiquitin Sequence Fallback in Builtin aminx Plugin Bypasses Defensive Quarantine

- **Severity**: **CRITICAL / INTEGRITY VIOLATION**
- **Lens**: Lens B (Cross-Repo Contract & Security/Quarantine)
- **Target Repository**: `xtrax` (PR #269)
- **File & Line Numbers**: `src/xtrax/eda/campaign/domains/aminx.py:85-88`

#### 2.1 Verbatim Code Quotation
```python
# src/xtrax/eda/campaign/domains/aminx.py:82-90
        sequences_list = []
        if isinstance(raw_seqs, list):
            sequences_list = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in raw_seqs]
        if not sequences_list:
            # Fallback for synthetic / demo units
            sequences_list = ["MQIFVKTLTGKTITLEVE"]
        
        # Build token array
        tokens_list = []
```

#### 2.2 Root-Cause Analysis
When reading an aminx campaign execution unit from disk, if `sequences.npy` is missing, unreadable, or empty, `sequences_list` evaluates to empty (`not sequences_list`). Instead of raising an exception or propagating a failure status to trigger unit quarantine, lines 85–88 inject a hardcoded 18-residue protein sequence: `"MQIFVKTLTGKTITLEVE"` (the N-terminus of human ubiquitin).

This fallback was ostensibly added as a shortcut for demo or mock units, but it resides directly inside the production `AminxDomainPlugin.extract_cell` execution path.

#### 2.3 Forensic Blast Radius & Integrity Implications
1. **Falsification of Experimental Data**: If an actual experimental unit fails on a cluster node (e.g. out of memory, GPU job timeout, empty output generation), the EDA pipeline does not report the failure. Instead, it fabricates a valid ubiquitin sequence, tokenizes it, and scores it.
2. **Defeating the Quarantine Boundary**: Spec 261106 §4.2 mandates that any unit failing ingestion or validation must be quarantined via `quarantine_unit(unit_key, domain_name, reason)`, recorded in `quarantine.json`, and surfaced in the Q4 Coverage View. This hardcoded fallback completely bypasses the quarantine boundary, allowing unexecuted or corrupted units to report passing status.
3. **Statistical Corruption**: The fabricated sequence enters downstream statistical analyses: Q1 sequence diversity, Q2 NLL energetics, and Q3 cluster ratio bootstrap. Point estimates and confidence intervals are computed over fake data.
4. **Violation of Praxia Integrity Mandate**: This violates the fundamental rule: *"DO NOT create dummy or facade implementations that produce correct-looking outputs without genuine logic. Every implementation must maintain real state and produce real behavior."*

#### 2.4 Required Remediation
1. Completely delete lines 85–88 from `src/xtrax/eda/campaign/domains/aminx.py`.
2. Replace with an explicit check that raises a descriptive exception when no valid sequences are extracted:
   ```python
   if not sequences_list:
       raise FileNotFoundError(
           f"No valid sequences found in unit {store_path} (manifest_row_hash={manifest_row.manifest_row_hash})"
       )
   ```
3. Verify that when an empty unit is encountered, `execute_domain_plugin` catches the `FileNotFoundError`, records a quarantine entry in `quarantine.json`, and excludes the unit from Q1–Q3 statistics while reflecting it in Q4 coverage reconciliation.

---

## 3. Detailed Section: BLOCKER Findings

### 3.1 F-BLK-01: Metric & Flag Key Naming Mismatch Causes Silent Campaign Data Annihilation
- **Deduplicated Identifiers**: `F-AMX-01` (Lens A), `F-LENB-01` (Lens B)
- **Severity**: **BLOCKER**
- **Repositories**: `aminx` (PR #277), `xtrax` (PR #269)
- **Locations**:
  - `aminx/src/aminx/eda/plugin.py:598-605`
  - `xtrax/src/xtrax/eda/campaign/__init__.py:206, 244`
  - `xtrax/src/xtrax/eda/campaign/artifacts.py:117, 126`

#### Root Cause & Code Evidence
In `aminx.eda.plugin:compute_metrics()`, the metric and flag dictionaries are constructed with non-canonical keys:
```python
# aminx/src/aminx/eda/plugin.py:598-605
metrics: dict[str, float] = {
    "diversity": mean_diversity,
    "recovery": mean_rec,
    "native_recovery": mean_rec,
    "recovery_shuffled_native": mean_shuf_rec,  # <-- Non-canonical key
    "score_nll": score_nll,                    # <-- Non-canonical key
    "mean_nll": score_nll,                     # <-- Non-canonical key
    "perplexity": perplexity,
}
flags["score_derived"] = bool(cell.is_score_derived)  # <-- Non-canonical key
```
Simultaneously, `xtrax.eda.campaign` hardcodes extraction of canonical keys `"nll"`, `"recovery_shuffled"`, and `"is_score_derived"`:
```python
# xtrax/src/xtrax/eda/campaign/__init__.py:206, 244
206: nll_val = metric_res.metrics.get("nll", 0.0)
244: unit_sums.append([n_samples * metric_res.metrics.get("nll", 0.0)])
```
Vega-Lite chart specifications in `artifacts.py` explicitly target fields `"nll"` and `"recovery_shuffled"`.

#### Blast Radius
Because `.get("nll", 0.0)` defaults to `0.0` when the key is absent, no runtime exception is raised. Instead, all NLL values across all units evaluate to `0.0`. Q2 energetic distributions collapse into degenerate delta functions at zero, and Q3 cluster ratio bootstraps compute ratios of zeroes. This is silent data annihilation.

#### Required Remediation
In `aminx/src/aminx/eda/plugin.py`, emit canonical keys `"nll"`, `"recovery_shuffled"`, and `"is_score_derived"` alongside legacy keys:
```python
metrics["nll"] = score_nll
metrics["score_nll"] = score_nll
metrics["mean_nll"] = score_nll
metrics["recovery_shuffled"] = mean_shuf_rec
metrics["recovery_shuffled_native"] = mean_shuf_rec
flags["is_score_derived"] = bool(cell.is_score_derived)
flags["score_derived"] = bool(cell.is_score_derived)
```

---

### 3.2 F-BLK-02: Storage Parity Failure in aminx `_load_sink_data`
- **Deduplicated Identifiers**: `F-AMX-02` (Lens A), `F-LENB-17` (Lens B), `F-C01` (Lens C)
- **Severity**: **BLOCKER**
- **Repository**: `aminx` (PR #277)
- **Location**: `src/aminx/eda/plugin.py:200-234`

#### Root Cause & Code Evidence
```python
# aminx/src/aminx/eda/plugin.py:200-215
def _load_sink_data(self, store_path: Path | Any, manifest_row: Any) -> Mapping[str, np.ndarray]:
    if isinstance(store_path, Mapping):
        return {k: np.asarray(v) for k, v in store_path.items()}
    import zarr
    path = Path(store_path)
    if path.is_file() and path.name.endswith(".npy"):
        return {"sequences": np.load(path)}
    # Open zarr group
    group = zarr.open_group(str(path), mode="r")
```
`_load_sink_data` assumes that if `store_path` is not a single `.npy` file, it must be an uncompressed Zarr group directory. In real campaigns, units are directories of loose `.npy` files (`sequences.npy`, `scores.npy`, `done.json`) or packed archives (`members.tar.zst`).

#### Empirical Proof of Failure
Calling `_load_sink_data` on a standard unit directory containing loose `.npy` files raises:
```
zarr.errors.GroupNotFoundError: GroupNotFoundError No group found in store at path ''
```
The exact same failure occurs on packed `.tar.zst` units. 100% of real aminx campaign units trigger an unhandled exception and are quarantined.

#### Required Remediation
Refactor `_load_sink_data` to transparently handle loose directories, packed `.tar.zst` archives, and in-memory mappings using `read_campaign_unit_member` (or `xtrax.pack.read_unit_member`):
1. Detect whether `store_path` is packed via `is_packed(store_path, record_name=...)`.
2. Read `sequences.npy`, `scores.npy`, and `logits.npy` by loading bytes directly into `np.load(io.BytesIO(raw_bytes))`.
3. Provide graceful fallback for `data.json` or loose array directories.

---

### 3.3 F-BLK-03: Dual-Marker Bug in demistify `_read_member_bytes` / `_list_unit_relpaths`
- **Deduplicated Identifiers**: `F-DEM-02` (Lens A), `F-LENB-16` (Lens B), `F-C02` (Lens C)
- **Severity**: **BLOCKER**
- **Repository**: `demistify` (PR #91)
- **Location**: `src/demistify/eda/plugin.py:116, 133`

#### Root Cause & Code Evidence
```python
# demistify/src/demistify/eda/plugin.py:112-138
def _read_member_bytes(store_path: Path, relpath: str) -> bytes:
    record_name = "done.json" if (store_path / "done.json").exists() else "unit.json"
    try:
        if is_packed(store_path):  # <-- BUG: record_name omitted!
            return read_unit_member(store_path, relpath, record_name=record_name)
...
def _list_unit_relpaths(store_path: Path) -> list[str]:
    record_name = "done.json" if (store_path / "done.json").exists() else "unit.json"
    try:
        if is_packed(store_path):  # <-- BUG: record_name omitted!
            return list_unit_members(store_path, record_name=record_name)
```
While `record_name` is correctly computed on lines 113 and 130, `is_packed(store_path)` is called without passing `record_name=record_name`. The function signature in `xtrax.pack` is `is_packed(unit_dir: PathLike, record_name: str = UNIT_RECORD)` where `UNIT_RECORD = "unit.json"`.

#### Blast Radius
For any campaign unit packed using `done.json`, `is_packed(store_path)` checks for `unit.json`, returns `False`, and falls through to loose file scanning (`rglob("*")`). `rglob` finds only container files: `['members.tar.zst', 'done.json', 'members.idx.json']`. No internal trajectory files are visible, causing `extract_cell` to raise `FileNotFoundError: No states_traj found in unit`. Ingestion fails 100% of the time for packed units in campaigns.

#### Required Remediation
Pass `record_name=record_name` on both calls:
```python
if is_packed(store_path, record_name=record_name):
```

---

### 3.4 F-BLK-04: Dual-Marker Bug in demistify `PackedZarrStore`
- **Deduplicated Identifiers**: `F-C03` (Lens C), `F-DEM-02` (Lens A)
- **Severity**: **BLOCKER**
- **Repository**: `demistify` (PR #91)
- **Location**: `src/demistify/io/zarr_io.py:1087`, `src/demistify/unit_pack.py:488`

#### Root Cause & Code Evidence
```python
# demistify/src/demistify/io/zarr_io.py:1079-1087
class PackedZarrStore(Store):
    def __init__(self, unit_dir: PathLike, store_prefix: str = "result.zarr") -> None:
        super().__init__(read_only=True)
        self.unit_dir = Path(unit_dir)
        ...
        from ..unit_pack import PackedIndex
        self._index = PackedIndex(self.unit_dir)  # <-- BUG: record_name omitted!
```
`PackedIndex` defaults to `record_name="unit.json"`. When a Schema B unit (`result.zarr`) is packed with `done.json`, `PackedIndex` raises:
```
xtrax.pack._node.UnitPackRefused: [R-U1] .../unit.json: unit has no record
```

#### Required Remediation
Add optional `record_name: str | None = None` to `PackedZarrStore.__init__`. If `None`, auto-detect:
```python
record_name = "done.json" if (self.unit_dir / "done.json").exists() else "unit.json"
self._index = PackedIndex(self.unit_dir, record_name=record_name)
```

---

### 3.5 F-BLK-05: Local Working Tree Desynchronization in aminx
- **Deduplicated Identifiers**: `F-AMX-07` (Lens A), `F-LENB-05` (Lens B), `F-C04` (Lens C)
- **Severity**: **BLOCKER**
- **Repository**: `aminx` (`/home/marielle/projects/aminx`)
- **Location**: Working tree git status and HEAD commit

#### Root Cause & Code Evidence
The local working tree for `aminx` is currently on branch `feat/261106-campaign-eda-plugin`, but pointing to commit `86042c0e` (`main`), rather than the remote PR #277 commit `05174ed6` (`origin/feat/261106-campaign-eda-plugin`).
Consequently:
- `src/aminx/eda/plugin.py` does not exist on disk.
- `tests/host/test_eda_plugin.py` does not exist on disk.
- `pyproject.toml` lacks entry point `[project.entry-points."xtrax.eda.domains"]`.
- `uv run pytest tests/host/test_eda_plugin.py` fails with code 4 ("file not found").
- Plugin discovery fails completely across all tools in the active environment.

#### Required Remediation
Fast-forward the local branch `feat/261106-campaign-eda-plugin` to match `origin/feat/261106-campaign-eda-plugin` (`05174ed6`):
```bash
cd /home/marielle/projects/aminx
git checkout feat/261106-campaign-eda-plugin
git merge --ff-only origin/feat/261106-campaign-eda-plugin
```

---

### 3.6 F-BLK-06: `import-linter` Configuration Failure Crashes `lint-imports` in xtrax
- **Deduplicated Identifiers**: `F-LENB-11` (Lens B)
- **Severity**: **BLOCKER**
- **Repository**: `xtrax` (PR #269)
- **Location**: `pyproject.toml:203-205`

#### Root Cause & Code Evidence
In `xtrax/pyproject.toml`:
```toml
[tool.importlinter]
root_package = "xtrax"
```
Contract 4 ("Campaign EDA isolation from external frameworks") declares external modules (`maraxiom`, `bathos`, `cisternal`) in `forbidden_modules`. When external modules are declared, `import-linter` requires `include_external_packages = true` at the top level. Omitting this key causes `lint-imports` to crash immediately on launch:
```
The top level configuration must have include_external_packages=True when there are external forbidden modules. (Exit code 1)
```

#### Required Remediation
Add `include_external_packages = true` under `[tool.importlinter]` in `xtrax/pyproject.toml`.

---

### 3.7 F-BLK-07: Hardcoded Generative Sequence Metrics in xtrax Break Non-Generative Domains
- **Deduplicated Identifiers**: `F-XTX-02` (Lens A), `F-LENB-02` (Lens B)
- **Severity**: **BLOCKER**
- **Repository**: `xtrax` (PR #269)
- **Location**: `src/xtrax/eda/campaign/__init__.py:184-250`

#### Root Cause & Code Evidence
In `run_campaign_eda`, the metric extraction loop hardcodes generative sequence metrics:
```python
# xtrax/src/xtrax/eda/campaign/__init__.py:184-189
q1_diversity = metric_res.metrics.get("diversity", 0.0)
q1_pos_ent = metric_res.metrics.get("positional_entropy", 0.0)
q1_comp_ent = metric_res.metrics.get("composition_entropy", 0.0)
q1_rec = metric_res.metrics.get("recovery", 0.0)
q1_margin = metric_res.metrics.get("recovery_margin", 0.0)
```
For physical simulation domains like `demistify`, the true observables are conformational entropy and allosteric network scores: `total_entropy`, `mean_residue_entropy`, `hub_score@1`, `der.hub_degree@1`. Because `xtrax` contains no domain metric mapping mechanism, all demistify observables are ignored, and Q1, Q2, and Q3 output pure zeroes.

#### Required Remediation
1. Introduce a domain metric mapping dictionary or domain protocol method `primary_observable() -> str` (e.g. mapping `total_entropy` to energetic score in Q2/Q3 for demistify).
2. Dynamically ingest all numeric keys present in `metric_res.metrics` into the per-cell and per-unit summary tables, ensuring no domain observable is silently dropped.

---

## 4. Detailed Section: MAJOR Findings

### 4.1 F-MAJ-01: Artificial Truncation of `LibraryCell.sequences` at min(T, 100) in demistify
- **Deduplicated Identifiers**: `F-DEM-01` (Lens A), `F-LENB-18` (Lens B)
- **Severity**: **MAJOR**
- **Repository**: `demistify` (PR #91), `xtrax` (PR #269)
- **Location**: `demistify/src/demistify/eda/plugin.py:382`, `xtrax/src/xtrax/eda/campaign/__init__.py:242`
- **Description**: In `DemistifyDomainPlugin.extract_cell`:
  ```python
  sequences = tuple(f"frame_{t}" for t in range(min(T, 100)))
  ```
  For an MD trajectory of length $T = 10,000$, `tokens.shape[0] == 10000`, but `len(cell.sequences) == 100`.
  In `xtrax/src/xtrax/eda/campaign/__init__.py:242`:
  ```python
  n_samples = len(cell.sequences)
  unit_counts.append(n_samples)
  ```
  `unit_counts` receives $100$ rather than $10,000$.
- **Statistical Distortion**: The Rao-Wu cluster ratio estimator calculates the weighted ratio:
  $$\hat{\theta} = \frac{\sum_{u=1}^U T_u}{\sum_{u=1}^U n_u}$$
  When $T$ varies across replicas (e.g. replica 1 ran $500$ frames, replica 2 ran $5,000$ frames), capping at $100$ treats both replicas as equal cluster sizes ($100$ vs $100$), corrupting cluster weighting and undercounting effective sample size by up to 1,000x.
- **Remediation**: In `demistify`, do not generate synthetic sequence strings for frames; set `sequences = ()` or `tuple(f"frame_{t}" for t in range(T))`. In `xtrax`, derive `n_samples` from `cell.tokens.shape[0]` (or `len(cell.sequences)` if tokens is None).

---

### 4.2 F-MAJ-02: Replicate Conflation with Knob Levels in xtrax Condition Derivation
- **Deduplicated Identifiers**: `F-XTX-01` (Lens A)
- **Severity**: **MAJOR**
- **Repository**: `xtrax` (PR #269)
- **Location**: `src/xtrax/eda/campaign/__init__.py:173, 205, 232`
- **Description**: Condition grouping is hardcoded to:
  ```python
  cond_str = f"{row.arm_id}_T{row.parameters.get('temperature', 'def')}"
  ```
  In sweeps that vary `checkpoint_id`, `fixed_policy`, `state_weight_profile`, `noise_level`, or `target_input`, these experimental knob dimensions are ignored. Rows with different checkpoints are lumped into identical `cond_str` groups. On line 238, `U = len(items)` pools these distinct experimental treatments as replicates of the same condition!
- **Scientific Impact**: Violates the global principle that replicas are stochastic repetitions of identical treatments. Conflating distinct treatments inflates within-condition variance and corrupts confidence intervals.
- **Remediation**: Dynamically derive condition keys by sorting and hashing/stringifying all parameter keys and manifest knob fields (excluding stochastic `seed`, `manifest_row_hash`, `job_id`, and `chunk_id`).

---

### 4.3 F-MAJ-03: Non-Vectorized Python Loops in NLL Evaluation & Positional Entropy
- **Deduplicated Identifiers**: `F-XTX-03` (Lens A), `F-AMX-05` (Lens A)
- **Severity**: **MAJOR**
- **Repositories**: `xtrax` (PR #269), `aminx` (PR #277)
- **Locations**:
  - `xtrax/src/xtrax/eda/campaign/domains/aminx.py:142-153, 205-210`
  - `aminx/src/aminx/eda/plugin.py:504-518`
- **Description**:
  1. `xtrax` lines 142–153 compute derived NLL using a double-nested Python loop: `for i in range(N): for pos in range(L):` evaluating LogSumExp per residue per sample.
  2. `xtrax` lines 205–210 re-implement positional entropy via `for pos in range(L):` instead of calling `compute_positional_entropy()`.
  3. `aminx` lines 504–518 compute positional entropy via `for pos in range(seq_len):` computing bincounts column-by-column.
- **Performance Impact**: For a production campaign with $N = 10,000$ and $L = 300$, a nested Python loop performs $3 \times 10^6$ interpreter iterations, incurring high CPU latency and violating JAX/NumPy vectorization standards.
- **Remediation**: Vectorize derived NLL with NumPy axis operations (`np.take_along_axis`, `scipy.special.logsumexp(axis=-1)`); replace column loops with 2D bincount histogramming.

---

### 4.4 F-MAJ-04: Unguarded Zero-Sample Division and NaN Leaks in Bootstrap
- **Deduplicated Identifiers**: `F-XTX-04` (Lens A)
- **Severity**: **MAJOR**
- **Repository**: `xtrax` (PR #269)
- **Location**: `src/xtrax/eda/campaign/stats/bootstrap.py:48-54, 108, 164-165`
- **Description**:
  1. Line 108: `point_estimates = np.sum(t_mat, axis=0) / float(np.sum(n_vec))` raises `ZeroDivisionError` if $\sum n = 0$.
  2. Line 54: In `vectorized_cluster_ratio_bootstrap()`, `return numerator / denominator` produces `NaN` if all units in a resample have $n_u = 0$. `NaN` propagates into `np.percentile`, returning `ci_95 = (nan, nan)`.
  3. Lines 164–165: `evaluate_contrast` computes differences of sums divided by counts without zero guards.
- **Remediation**: If $\sum n = 0$, return `descriptive_only=True` with a diagnostic warning; wrap bootstrap denominators with `np.where(denominator == 0, np.nan, denominator)`.

---

### 4.5 F-MAJ-05: Per-Residue Track Arrays in `DomainMetricResult.tracks` Remain Mutable
- **Deduplicated Identifiers**: `F-LENB-06` (Lens B), `F-LENB-07` (Lens B)
- **Severity**: **MAJOR**
- **Repositories**: `xtrax` (PR #269), `demistify` (PR #91)
- **Location**: `xtrax/src/xtrax/eda/campaign/plugins.py:58-74, 146`
- **Description**: `enforce_cell_immutability(cell)` sets `writeable = False` on tokens, mask, logits, and scores in `LibraryCell`. However, `metric_res.tracks` (containing per-residue entropy, allosteric hub scores, etc.) are returned with `flags.writeable = True`. Downstream visualization or export code can mutate these arrays in place, corrupting shared memory.
- **Remediation**: Add `enforce_metric_result_immutability(result: DomainMetricResult)` that iterates over `result.tracks.values()` and sets `track.flags.writeable = False`. Invoke it in `execute_domain_plugin`.

---

### 4.6 F-MAJ-06: Quarantine Leaks on Discovery `SystemExit` and Exploding Properties
- **Deduplicated Identifiers**: `F-LENB-08` (Lens B), `F-LENB-09` (Lens B), `F-LENB-10` (Lens B)
- **Severity**: **MAJOR**
- **Repository**: `xtrax` (PR #269)
- **Location**: `src/xtrax/eda/campaign/plugins.py:121, 143-159`
- **Description**:
  1. Line 121: `discover_domain_plugins()` catches `Exception`, but does not catch `SystemExit` or `BaseException` during `ep.load()`. A malformed third-party plugin can crash Python during discovery.
  2. Line 157: `quarantine_unit(unit_key, plugin.domain_name, reason)` accesses `plugin.domain_name` inside the exception handler. If `domain_name` raises an exception, the handler crashes and the error escapes uncaught.
  3. Line 143: If `extract_cell` returns `[]` without raising, the unit silently vanishes without a quarantine record.
- **Remediation**: Catch `(Exception, SystemExit)` in discovery; safely resolve domain name via `getattr(plugin, "domain_name", type(plugin).__name__)`; quarantine units returning 0 cells.

---

### 4.7 F-MAJ-07: Permissive `validate_schema` in Consumer Plugins & Missing Track Validation
- **Deduplicated Identifiers**: `F-AMX-04` (Lens A), `F-LENB-13` (Lens B), `F-LENB-14` (Lens B)
- **Severity**: **MAJOR**
- **Repositories**: `aminx` (PR #277), `demistify` (PR #91), `xtrax` (PR #269)
- **Locations**:
  - `aminx/src/aminx/eda/plugin.py:89`
  - `demistify/src/demistify/eda/plugin.py:73`
  - `xtrax/src/xtrax/eda/campaign/types.py:63-88`
- **Description**:
  1. Consumer `validate_schema()` only checks that metric values are numeric floats/ints. They omit checking that metric keys are `str`, and perform zero validation of `flags`.
  2. When consumer plugins construct a `DomainMetricResult` with non-string keys or non-boolean flags, downstream Parquet serialization crashes with `ArrowInvalid`.
  3. `DomainMetricResult.validate_schema()` in `xtrax` performs zero validation on `tracks`.
- **Remediation**: Align consumer `validate_schema()` with `xtrax`; validate `tracks` keys (`str`), values (`np.ndarray`), and finiteness in `xtrax.types.DomainMetricResult.validate_schema()`; have `xtrax.execute_domain_plugin` invoke `DomainMetricResult.validate_schema(metric_res)` directly.

---

### 4.8 F-MAJ-08: Dataclass Duplication Breaks Nominal Type Compatibility
- **Deduplicated Identifiers**: `F-LENB-03` (Lens B)
- **Severity**: **MAJOR**
- **Repositories**: `aminx` (PR #277), `demistify` (PR #91)
- **Locations**:
  - `aminx/src/aminx/eda/plugin.py:49-102`
  - `demistify/src/demistify/eda/plugin.py:31-87`
- **Description**: `aminx` and `demistify` duplicate definitions of `DomainMetricResult`, `LibraryCell`, and `ResidueTable` instead of importing from `xtrax.eda.campaign.types`. Consequently, `isinstance(metric_res, xtrax.DomainMetricResult)` evaluates to `False`.
- **Remediation**: Import authoritative dataclasses from `xtrax.eda.campaign.types`, providing fallback dataclasses only when `xtrax` is not installed.

---

### 4.9 F-MAJ-09: Control 4 Storage Format Parity Mock Shortcut in Benchmark
- **Deduplicated Identifiers**: `F-C05` (Lens C)
- **Severity**: **MAJOR (Rigor & Verification Gate)**
- **Repository**: `xtrax` (PR #269)
- **Location**: `scripts/benchmarks/campaign_eda_benchmark.py:276-361`
- **Description**: Pre-registered verification Control 4 is specified to verify bit-identical extraction between loose directories and packed archives using `xtrax.pack.pack_unit()`. In `campaign_eda_benchmark.py`, `run_control_4_storage_format_parity()` uses Python's `tarfile` and `zstandard` directly on toy dummy JSON files, completely bypassing `xtrax.pack`, `CampaignReader`, and domain plugins.
- **Remediation**: Rewrite `run_control_4_storage_format_parity()` to write a realistic unit, pack it using `xtrax.pack.pack_unit(..., record_name="done.json")`, and verify bit-identical member bytes via `read_campaign_unit_member()` and `extract_cell()`, matching `tests/test_campaign_eda/test_controls.py:176`.

---

### 4.10 F-MAJ-10: Mock `uv` Sandbox Trap in Standalone Node Test
- **Deduplicated Identifiers**: `F-C06` (Lens C)
- **Severity**: **MAJOR**
- **Repository**: `xtrax` (PR #269)
- **Location**: `tests/pack/test_node_standalone.py:48-56`
- **Description**: `_python39()` executes `uv python find 3.9`. When running inside a sandbox or container where `uv` is a mock binary, stdout is `"Mock uv for sandbox environment"`. `_python39()` treats this string as an executable path without verifying `Path(path).is_file()`, causing `subprocess.run` to crash with `FileNotFoundError`.
- **Remediation**: Update `_python39()` to check `proc.returncode == 0 and path and Path(path).is_file()`.

---

### 4.11 F-MAJ-11: Ruff UP Rules Flag Python 3.9 Compatibility in `_node.py`
- **Deduplicated Identifiers**: `F-C07` (Lens C)
- **Severity**: **MAJOR**
- **Repository**: `xtrax` (PR #269)
- **Location**: `pyproject.toml:23-28`, `src/xtrax/pack/_node.py`
- **Description**: `pyproject.toml` enables `UP` (pyupgrade) globally with `target-version = "py313"`. `src/xtrax/pack/_node.py` is strictly pinned to Python 3.9 compatibility for cluster node script embedding. Running `ruff check` produces 128 `UP` errors. Auto-fixing would corrupt Python 3.9 compatibility.
- **Remediation**: Add `"src/xtrax/pack/_node.py" = ["UP"]` and `"src/xtrax/run/writer_liveness.py" = ["UP"]` to `[tool.ruff.lint.per-file-ignores]` in `pyproject.toml`.

---

### 4.12 F-MAJ-12: 28 Ruff Lint Violations in `aminx.eda.plugin`
- **Deduplicated Identifiers**: `F-C08` (Lens C)
- **Severity**: **MAJOR**
- **Repository**: `aminx` (PR #277)
- **Location**: `src/aminx/eda/plugin.py` (commit `05174ed6`)
- **Description**: `src/aminx/eda/plugin.py` produces 28 violations of `ANN401` (Any typing), `PLC0415` (top-level imports), `BLE001` (blind except), `ARG002` (unused arguments), `SIM102` (nested ifs), and `PLR0915` (functions > 50 statements), failing aminx CI checks.
- **Remediation**: Refactor methods over 50 statements into helper functions, hoist imports, type annotations, and catch specific exceptions.

---

### 4.13 F-MAJ-13: Architectural Boundary Gap in `import-linter`
- **Deduplicated Identifiers**: `F-LENB-12` (Lens B)
- **Severity**: **MAJOR**
- **Repository**: `xtrax` (PR #269)
- **Location**: `pyproject.toml:276-285`
- **Description**: Import-linter Contract 4 forbids external frameworks (`maraxiom`, `bathos`, `cisternal`), but does not forbid downstream domain libraries (`aminx`, `demistify`, `adaptyv`, `plegadx`).
- **Remediation**: Add consumer packages (`aminx`, `demistify`, `adaptyv`, `plegadx`) to `forbidden_modules` in Contract 4.

---

## 5. Detailed Section: MINOR Findings

### 5.1 F-MIN-01: Omission of ANOVA / Feature Importance for Scientist Question 4
- **Deduplicated Identifiers**: `F-XTX-05` (Lens A)
- **Severity**: **MINOR**
- **Repository**: `xtrax` (PR #269)
- **Location**: `src/xtrax/eda/campaign/`
- **Description**: Scientist Question 4 ("Knobs That Matter / ANOVA & Feature Importance") was pre-registered to decompose variance across sweep parameters. In PR #269, sample-level ANOVA was rejected to prevent spurious significance under $N \to \infty$, but it was replaced entirely by sweep coverage reconciliation with zero variance attribution across knobs.
- **Remediation**: Document why sample-level ANOVA was rejected and compute replicate-level ($U$) effect size ranking across knobs in Q4.

---

### 5.2 F-MIN-02: Stochastic Shuffled Native Baseline Seed in aminx
- **Deduplicated Identifiers**: `F-AMX-06` (Lens A)
- **Severity**: **MINOR**
- **Repository**: `aminx` (PR #277)
- **Location**: `src/aminx/eda/plugin.py:549`
- **Description**: Shuffled native baseline uses `rng = np.random.default_rng(cell.seed)`, adding extraneous stochastic noise to `recovery_margin` across replicates of identical conditions.
- **Remediation**: Use a fixed deterministic seed (e.g. 42) matching `xtrax.eda.campaign.stats.composition`.

---

### 5.3 F-MIN-03: Ruff Format Check Failures Across `demistify` and `xtrax`
- **Deduplicated Identifiers**: `F-C09` (Lens C)
- **Severity**: **MINOR**
- **Repositories**: `demistify` (PR #91), `xtrax` (PR #269)
- **Locations**:
  - `demistify:src/demistify/eda/plugin.py`, `demistify:tests/test_eda_plugin.py`
  - `xtrax:scripts/empirical_challenge_harness.py`, `xtrax:tests/conftest.py`
- **Description**: `ruff format --check` fails in both repositories. In addition, `xtrax/pyproject.toml` lacks `extend-exclude = [".agents"]`, causing ruff to scan agent scratch directories.
- **Remediation**: Run `ruff format` and add `extend-exclude = [".agents"]` in `xtrax/pyproject.toml`.

---

### 5.4 F-MIN-04: Missing Type Annotations / 56 `ty check` Diagnostics in demistify
- **Deduplicated Identifiers**: `F-C10` (Lens C)
- **Severity**: **MINOR**
- **Repository**: `demistify` (PR #91)
- **Location**: `src/demistify/eda/plugin.py:244-277`
- **Description**: 56 diagnostics emitted under `ty check`: unimported `zarr.storage`, Zarr v3 indexing unions, and uncast float conversions.
- **Remediation**: Explicitly import `zarr.storage`, narrow Zarr group types before indexing, and cast attribute floats.

---

### 5.5 F-MIN-05: Zero Test Coverage for Packed `done.json` Units in demistify
- **Deduplicated Identifiers**: `F-C11` (Lens C)
- **Severity**: **MINOR**
- **Repository**: `demistify` (PR #91)
- **Location**: `tests/test_eda_plugin.py`
- **Description**: All test fixtures in `tests/test_eda_plugin.py` call `pack_unit` with default `unit.json`, leaving the `done.json` execution path unexercised and allowing the dual-marker bug to go undetected.
- **Remediation**: Add unit tests verifying extraction from packed Schema A and Schema B units marked with `done.json`.

---

### 5.6 F-MIN-06: Entry-Point Discovery Missing Caching in Campaign EDA
- **Deduplicated Identifiers**: `F-LENB-04` (Lens B)
- **Severity**: **MINOR**
- **Repository**: `xtrax` (PR #269)
- **Location**: `src/xtrax/eda/campaign/plugins.py:110-125`
- **Description**: `discover_domain_plugins()` executes an uncached `importlib.metadata.entry_points(group="xtrax.eda.domains")` call on every invocation, adding unnecessary filesystem overhead.
- **Remediation**: Introduce `_DOMAIN_PLUGIN_CACHE: dict[str, DomainPlugin] | None = None` and `reload_domain_plugins()`, matching `xtrax.eda.plugins`.

---

## 6. Pack Telemetry Seam Architectural Assessment (Debt #3300)

The swarm review evaluated the proposed architecture for Debt #3300 (`xtrax.pack` producer-side telemetry and `xtrax.eda.campaign.telemetry` consumer-side ingestion):

1. **Top-Level Record Placement (`record["pack_telemetry"]`)**:
   - `tests/pack/test_pack.py` asserts reader parity with `alynxr` via `assert set(storage) == set(theirs)`.
   - Injecting pack telemetry into `record["storage"]` violates this assertion.
   - `_CORE_FIELDS` in `_node.py` hashes only identity fields; placing telemetry at top-level `record["pack_telemetry"]` preserves `record_digest` invariance, passes `record_problem()`, and guarantees backward compatibility for legacy units.
2. **Strict Stdlib Isolation**:
   - `src/xtrax/pack/_node.py` MUST remain 100% Python stdlib-only. Timing and compression telemetry (`pack_wall_ms`, `raw_bytes`, `compressed_bytes`, `compression_ratio`, `member_count`, `backend`) can and must be gathered strictly via `time.perf_counter()`, `os.path.getsize()`, and `len(members)`.
3. **Consumer Ingestion States**:
   - `xtrax.eda.campaign.telemetry` must gracefully handle three distinct states:
     a. Fully instrumented (`"instrumented"`)
     b. Mixed legacy and instrumented
     c. Uninstrumented (`"no_packed_units"` or legacy units), displaying the graceful fallback banner.

---

## 7. Unified Remediation Action Plan

This section provides the actionable remediation plan across all three repositories, including exact files, changes, commit messages, and verification commands.

### 7.1 Repository: `xtrax` (PR #269, branch `feat/261106-campaign-eda`)

| Step | Target File | Action Required | Addressed Findings |
|:---:|:---|:---|:---:|
| **X1** | `src/xtrax/eda/campaign/domains/aminx.py` | Delete lines 85–88 (`sequences_list = ["MQIFVKTLTGKTITLEVE"]`). Raise `FileNotFoundError` when no valid sequences are extracted. Vectorize derived NLL with NumPy axis operations; reuse `compute_positional_entropy()`. | **F-LENB-15 (CRITICAL)**,<br>F-MAJ-03 |
| **X2** | `pyproject.toml` | 1. Add `include_external_packages = true` under `[tool.importlinter]`.<br>2. Add `aminx`, `demistify`, `adaptyv`, `plegadx` to Contract 4 `forbidden_modules`.<br>3. Add `"src/xtrax/pack/_node.py" = ["UP"]` and `"src/xtrax/run/writer_liveness.py" = ["UP"]` to `[tool.ruff.lint.per-file-ignores]`.<br>4. Add `extend-exclude = [".agents"]` under `[tool.ruff]`. | **F-BLK-06**,<br>F-MAJ-11,<br>F-MAJ-13,<br>F-MIN-03 |
| **X3** | `src/xtrax/eda/campaign/__init__.py` | 1. Dynamically construct condition keys from all manifest knob dimensions.<br>2. Derive `n_samples` from `cell.tokens.shape[0]` (fallback to `sequences`).<br>3. Support domain metric mapping / dynamic ingestion of non-generative domain observables (`total_entropy`, `hub_score@1`). | **F-BLK-07**,<br>F-MAJ-01,<br>F-MAJ-02 |
| **X4** | `src/xtrax/eda/campaign/plugins.py` | 1. Catch `(Exception, SystemExit)` in `discover_domain_plugins()`.<br>2. Add discovery cache `_DOMAIN_PLUGIN_CACHE` and `reload_domain_plugins()`.<br>3. Implement `enforce_metric_result_immutability()` setting `track.flags.writeable = False`.<br>4. Guard `plugin.domain_name` in quarantine handler; quarantine 0-cell returns. | F-MAJ-05,<br>F-MAJ-06,<br>F-MIN-06 |
| **X5** | `src/xtrax/eda/campaign/types.py` | Enhance `DomainMetricResult.validate_schema()` to validate that `tracks` keys are `str`, values are `np.ndarray`, and array values are finite numbers. | F-MAJ-07 |
| **X6** | `src/xtrax/eda/campaign/stats/bootstrap.py` | Add guards for $\sum n = 0$ in point estimates and bootstrap denominators, returning `descriptive_only=True` without NaN leaks or zero division. | F-MAJ-04 |
| **X7** | `scripts/benchmarks/campaign_eda_benchmark.py` | Rewrite `run_control_4_storage_format_parity()` to use `xtrax.pack.pack_unit(..., record_name="done.json")` and `read_campaign_unit_member()`. Add pre-registered `control_7_pack_telemetry_seam`. | F-MAJ-09 |
| **X8** | `tests/pack/test_node_standalone.py` | Update `_python39()` to verify `Path(path).is_file()`. | F-MAJ-10 |
| **X9** | Formatting | Run `uv run ruff format .` | F-MIN-03 |

**Commit Requirement (`xtrax`)**:
```bash
git commit -m "fix(eda): remediate M2 swarm audit findings across eda engine and pack seam

- Delete hardcoded ubiquitin fallback in builtin aminx domain (F-LENB-15)
- Fix import-linter external packages and add consumer isolation (F-BLK-06, F-MAJ-13)
- Fix condition grouping and sample size accounting (F-MAJ-01, F-MAJ-02)
- Enforce immutability on track arrays and guard quarantine handler (F-MAJ-05, F-MAJ-06)
- Guard zero counts in cluster bootstrap (F-MAJ-04)
- Fix mock uv detection in standalone node test (F-MAJ-10)
- Replace mock Control 4 with genuine pack_unit verification (F-MAJ-09)
- Ignore UP rules on Python 3.9 files in pyproject.toml (F-MAJ-11)

Task ID: 261109_pack-telemetry-and-eda-pr-swarm"
```

**Verification Commands (`xtrax`)**:
```bash
uv run pytest tests/pack/ tests/test_campaign_eda/ -v --no-cov
uv run ruff check .
uv run ruff format --check .
uv run ty check src/
uv run lint-imports
```

---

### 7.2 Repository: `aminx` (PR #277, branch `feat/261106-campaign-eda-plugin`)

| Step | Target File | Action Required | Addressed Findings |
|:---:|:---|:---|:---:|
| **A1** | Working Tree | Fast-forward local branch `feat/261106-campaign-eda-plugin` to `origin/feat/261106-campaign-eda-plugin` (`05174ed6`). | **F-BLK-05** |
| **A2** | `src/aminx/eda/plugin.py` | 1. Emit canonical keys `"nll"`, `"recovery_shuffled"`, and `"is_score_derived"` in `compute_metrics()`.<br>2. Refactor `_load_sink_data` to use `read_campaign_unit_member` or loose `.npy` + `read_unit_member` loader.<br>3. Vectorize positional entropy using 2D bincount histogramming.<br>4. Use fixed deterministic seed (42) for shuffled native baseline.<br>5. Align `DomainMetricResult.validate_schema()` with `xtrax.eda.campaign.types`.<br>6. Fix 28 Ruff lint violations (`ANN401`, `PLC0415`, `BLE001`, `ARG002`, `SIM102`, `PLR0915`). | **F-BLK-01**,<br>**F-BLK-02**,<br>F-MAJ-03,<br>F-MAJ-07,<br>F-MAJ-12,<br>F-MIN-02 |

**Commit Requirement (`aminx`)**:
```bash
git commit -m "fix(eda): remediate M2 swarm audit findings in aminx domain plugin

- Emit canonical keys 'nll', 'recovery_shuffled', and 'is_score_derived' (F-BLK-01)
- Support loose .npy and packed .tar.zst storage formats in _load_sink_data (F-BLK-02)
- Vectorize positional entropy computation (F-MAJ-03)
- Use deterministic seed for shuffled native baseline (F-MIN-02)
- Align validate_schema and fix 28 Ruff lint violations (F-MAJ-07, F-MAJ-12)

Task ID: 261109_pack-telemetry-and-eda-pr-swarm"
```

**Verification Commands (`aminx`)**:
```bash
uv run pytest tests/host/test_eda_plugin.py -v
uv run ruff check src/aminx/eda tests/host/test_eda_plugin.py
uv run ruff format --check src/aminx/eda tests/host/test_eda_plugin.py
```

---

### 7.3 Repository: `demistify` (PR #91, branch `feat/261106-campaign-eda-plugin`)

| Step | Target File | Action Required | Addressed Findings |
|:---:|:---|:---|:---:|
| **D1** | `src/demistify/eda/plugin.py` | 1. Pass `record_name=record_name` to `is_packed(store_path)` in `_read_member_bytes` and `_list_unit_relpaths`.<br>2. Populate `sequences` without arbitrary 100-frame truncation (or empty tuple).<br>3. Align `DomainMetricResult.validate_schema()` with `xtrax`.<br>4. Fix typing diagnostics under `ty check` (import `zarr.storage`, cast attribute floats). | **F-BLK-03**,<br>F-MAJ-01,<br>F-MAJ-07,<br>F-MIN-04 |
| **D2** | `src/demistify/io/zarr_io.py` | Add `record_name` resolution to `PackedZarrStore.__init__` and pass `record_name=record_name` to `PackedIndex`. | **F-BLK-04** |
| **D3** | `tests/test_eda_plugin.py` | Add unit tests for Schema A and Schema B packed units using `done.json` markers. | F-MIN-05 |
| **D4** | Formatting | Run `uv run ruff format src/demistify/eda tests/test_eda_plugin.py`. | F-MIN-03 |

**Commit Requirement (`demistify`)**:
```bash
git commit -m "fix(eda): remediate M2 swarm audit findings in demistify domain plugin

- Pass record_name to is_packed in _read_member_bytes and _list_unit_relpaths (F-BLK-03)
- Add dual-marker done.json support to PackedZarrStore (F-BLK-04)
- Remove 100-frame truncation on LibraryCell.sequences (F-MAJ-01)
- Align validate_schema and resolve ty check diagnostics (F-MAJ-07, F-MIN-04)
- Add tests for packed units with done.json markers (F-MIN-05)
- Reformat plugin files with ruff format (F-MIN-03)

Task ID: 261109_pack-telemetry-and-eda-pr-swarm"
```

**Verification Commands (`demistify`)**:
```bash
uv run pytest tests/test_eda_plugin.py -v
uv run ruff check src/demistify/eda tests/test_eda_plugin.py
uv run ruff format --check src/demistify/eda tests/test_eda_plugin.py
uv run ty check src/demistify/eda
```

---

## 8. Conclusion & Sign-Off

The multi-agent adversarial code review across `xtrax` PR #269, `aminx` PR #277, and `demistify` PR #91 has established that while the core statistical algorithms are sound, the current PR changesets cannot be merged due to critical integrity, contract, storage, and typing defects.

Remediation must proceed systematically across all three repositories following the Unified Remediation Action Plan outlined above. Once all remediations are committed and verified against the verification test suites, benchmark Control 4 and pre-registered Control 7 will be executed under Bathos to confirm end-to-end scientific and architectural integrity.
