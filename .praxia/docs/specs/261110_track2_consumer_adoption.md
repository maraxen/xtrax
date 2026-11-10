---
task_id: 261110_xtrax-debt-burn-down-and-consumer-adoption
track: 2
track_name: Consumer Adoption of xtrax.campaign & xtrax.pack
debts: [3142, 3257]
created: 2026-11-10
status: draft
---

# Track 2 Specification: Consumer Adoption of `xtrax.campaign` & `xtrax.pack` across `aminx` and `demistify`

**Task ID**: `261110_xtrax-debt-burn-down-and-consumer-adoption`  
**Track**: Track 2 (Consumer Adoption)  
**Debts**: #3142 (P2, feature), #3257 (P2, feature)  
**Date**: 2026-11-10  
**Status**: Draft (Ready for Adversarial Review)  
**Author**: Track 2 Specification Lead (`teamwork_worker_p1_track2`)  
**Target Repositories**:
- `/home/marielle/projects/aminx`
- `/home/marielle/projects/demistify`
- `/home/marielle/projects/xtrax` (reference API provider)

---

## 1. Executive Summary & Purpose

The objective of Track 2 is to burn down cross-repository technical debts **#3142** and **#3257** by transitioning downstream consumer pipelines in `aminx` and `demistify` away from bifurcated, ad-hoc execution loops and custom filesystem locking, onto standard upstream primitives provided by `xtrax.campaign` and `xtrax.pack` (shipped in `xtrax>=0.4.1b7`).

### 1.1 Technical Debts in Scope
1. **Debt #3142 (P2, feature)**: Wire `xtrax.campaign` execution (`plan`, `lease`, `run`, `resume`, `verify`, `LocalSequentialExecutor`, `LocalProcessPoolExecutor`, `FileLock`) into consumer sweep runners in `aminx` and `demistify`, replacing ad-hoc sweep loops while strictly preserving CLI backward compatibility.
2. **Debt #3257 (P2, feature)**: Adopt `xtrax.pack` (`write_unit_record`, `pack_unit`, `read_unit_member`, `is_packed`, `record_problem`, `unit_content_digest`, `register_telemetry_hook`) and `xtrax.campaign.marker.validate_unit_done` in consumer runners with per-unit output directories so completed units emit canonical `unit.json`/`done.json` records and support seekable `.tar.zst` packing against shared-filesystem inode budgets.
3. **CI Hygiene Edge Case**: Resolve the static CI test selection invariant failure in `aminx` (`tests/ci/test_select_affected_tests.py`) caused by unreferenced module `src/aminx/eda/__init__.py` missing from `scripts/ci/ci_test_selection_edges.toml`.

### 1.2 Core Architectural Achievements
- **In `aminx`**: Unify the bifurcated campaign execution paths (`src/aminx/host/campaign.py` vs `src/aminx/host/pack_shim.py`), wiring `LocalSequentialExecutor` (with opt-in `LocalProcessPoolExecutor`) and `xtrax.campaign.lease.FileLock` with attempt staging isolation and lease fencing (#3312/#3338). Completed units integrate `xtrax.pack` directly into the commit pipeline, reducing inode consumption by >95% while retaining full backward compatibility for all CLI commands (`aminx campaign plan`, `worker`, `run`, `adopt-legacy`, `gates`, `ramp-plan`, `ramp-evaluate`).
- **In `demistify`**: Correct the historical misconception from spec `261105_xtrax_pack_eda_adoption.md` (which assumed campaign execution was blocked in the wheel). Wire `xtrax.campaign` (`plan_campaign`, `LocalProcessPoolExecutor`, `FileLock`, `validate_unit_done`) into `scripts/analysis/batch_process.py` and experiment grid sweeps (`scripts/experiments/g0_grid_sweep.py`, etc.), replacing raw un-fenced Python `for` loops with robust leasing, attempt-level crash resilience, and unified `xtrax.pack` `.tar.zst` storage with `PackedZarrStore`.
- **Upstream Campaign/Pack Seam Resolution**: Eliminate the crash when `--pack` is enabled by upgrading `xtrax.campaign.marker`: `compute_content_digest()` delegates to `xtrax.pack.unit_content_digest()` when `members.tar.zst` or `unit.json` is detected (preventing `zarr.errors.GroupNotFoundError`), and `validate_unit_done()` verifies packed units via `xtrax.pack.is_packed()` and `record_problem()` instead of demanding loose `zarr.json` or `.zgroup` metadata (preventing spurious quarantine of valid packed stores on resume). The packaging lifecycle is strictly specified across attempt staging, promotion, done marker commit, and packing.

---

## 2. Assumption & Fact Verification Table

Every architectural assumption, file path, line number, and data contract cited in this specification has been verified directly against repository source code at HEAD.

| ID | Subject / Claim | Source Location | Verified Status | Evidence & Verification Detail |
|---|---|---|---|---|
| **A1** | `xtrax.campaign` ships in the wheel | `src/xtrax/campaign/__init__.py:1-125` | **VERIFIED** | Full exports: `LocalSequentialExecutor`, `LocalProcessPoolExecutor`, `FileLock`, `CampaignLock`, `plan_campaign`, `validate_unit_done`, `create_done_marker`, `AminxDoneMarkerAdapter`. |
| **A2** | `xtrax.pack` format & API | `src/xtrax/pack/__init__.py:1-84`, `_node.py` | **VERIFIED** | Standalone stdlib `_node.py` with `write_unit_record`, `pack_unit`, `read_unit_member`, `record_problem`, `unit_content_digest`, `is_packed`, `register_telemetry_hook`. Unit validation is decoupled: `record_problem` validates unit record integrity, `unit_content_digest` digests uncompressed members, and `validate_unit_done` manages two-tier lifecycle resume verification. Standard `alynxr.unit_pack/1` format. |
| **A3** | `aminx` bifurcated execution loop | `aminx:src/aminx/host/campaign.py:2018-2146`, `pack_shim.py:697-893` | **VERIFIED** | `campaign.py:execute_manifest` implements an ad-hoc serial row loop; `pack_shim.py:execute_packed_manifest` clones 800+ lines of runner logic exclusively for `--pack`. |
| **A4** | `aminx` custom file locking | `aminx:src/aminx/host/campaign.py:643-734` | **VERIFIED** | `_campaign_lock_context` implements basic file locking without monotonic fence counters, lease stealing fencing tokens, or attempt-scoped staging isolation. |
| **A5** | `aminx` done marker adapter | `xtrax:src/xtrax/campaign/marker.py:223-286` | **VERIFIED** | `AminxDoneMarkerAdapter` natively converts between aminx's v3 marker format (`manifest_row_hash`, `content_digest_sha256`) and `CampaignDoneMarker`. |
| **A6** | `aminx` CI allowlist edge failure | `aminx:tests/ci/test_select_affected_tests.py:129`, `scripts/ci/ci_test_selection_edges.toml` | **VERIFIED** | Module `src/aminx/eda/__init__.py` was merged in PR #277 without an entry in `[unreferenced_src].allowlist`, failing the test selection invariant. |
| **A7** | `demistify` campaign block status | `demistify:.praxia/docs/specs/261105_xtrax_pack_eda_adoption.md:42-53` | **VERIFIED** | Prior spec classified `xtrax.campaign` as BLOCKED believing runner code lived in `controller/`. Recon proves `src/xtrax/campaign/` is fully in-wheel in `0.4.1b7`. |
| **A8** | `demistify` un-fenced batch loop | `demistify:scripts/analysis/batch_process.py:418-448` | **VERIFIED** | `run_batch_processing` runs a plain Python `for pdb, xtc in pairs:` loop with no lock leasing, no fence tracking, and no worker crash recovery. |
| **A9** | `demistify` `PackedZarrStore` integration | `demistify:src/demistify/io/zarr_io.py:1065-1150` | **VERIFIED** | `PackedZarrStore` implements Zarr v3 `Store` over `xtrax.pack.read_unit_member`, allowing transparent direct reads from `members.tar.zst`. |
| **A10** | Sandbox boundary & cross-repo isolation | Linux container namespace | **VERIFIED** | Repositories `/home/marielle/projects/aminx` and `/home/marielle/projects/demistify` reside in sibling paths requiring isolated worktrees and `origin/main` checkout. |
| **A11** | Campaign/Pack Seam Hardening | `src/xtrax/campaign/marker.py:19-24, 126-150` | **VERIFIED** | `compute_content_digest` and `validate_unit_done` previously assumed loose Zarr root metadata (`zarr.json`/`.zgroup`), crashing on packed `members.tar.zst`. Hardening with `unit_content_digest`, `is_packed`, and `record_problem` closes the collision. |

---

## 3. Worktree & Git Isolation Directives

To guarantee that active development in `/home/marielle/projects/xtrax`, `/home/marielle/projects/aminx`, and `/home/marielle/projects/demistify` is completely isolated with **zero contamination of dirty working trees**:

### 3.1 `aminx` Isolated Worktree Directive
- **Target Worktree Path**: `/home/marielle/projects/aminx/.worktrees/feat-261110-campaign-pack-adoption`
- **Feature Branch**: `feat/261110-aminx-consumer-adoption`
- **Base Commit**: `origin/main` (SHA `e0638157`)
- **Isolation Command**:
  ```bash
  git -C /home/marielle/projects/aminx fetch origin main
  git -C /home/marielle/projects/aminx worktree add -b feat/261110-aminx-consumer-adoption \
    .worktrees/feat-261110-campaign-pack-adoption origin/main
  ```

### 3.2 `demistify` Isolated Worktree Directive
- **Target Worktree Path**: `/home/marielle/projects/demistify/.worktrees/feat-261110-campaign-adoption`
- **Feature Branch**: `feat/261110-demistify-consumer-adoption`
- **Base Commit**: `origin/main` (SHA `07e772b`)
- **Isolation Command**:
  ```bash
  git -C /home/marielle/projects/demistify fetch origin main
  git -C /home/marielle/projects/demistify worktree add -b feat/261110-demistify-consumer-adoption \
    .worktrees/feat-261110-campaign-adoption origin/main
  ```

### 3.3 Working Tree Safety Mandate
- No modifications shall be made directly in `/home/marielle/projects/aminx` or `/home/marielle/projects/demistify` primary working directories.
- All file edits, test invocations, and git operations must execute strictly inside the created worktree paths.

---

## 4. Architectural Design: `aminx`

### 4.1 Debt #3142: Wiring `xtrax.campaign` into `aminx.host.campaign`

#### 4.1.1 Current Architecture & Deficiencies
In `aminx`, `src/aminx/host/campaign.py`:
- `execute_manifest(manifest_path, row_hashes=..., continue_on_error=...)` (lines 2018–2146) iterates over rows in a manual Python `for` loop.
- `_campaign_lock_context(lock_path, lease_seconds)` (lines 643–734) implements a bare lock file without lease fencing tokens (#3312). If an NFS stall or heavy JAX compilation causes a worker to pause past `lease_seconds`, another worker steals the lease; when the stalled worker resumes, it clobbers the stealer's output.
- Done marker validation (`_read_done_marker`, line 602) manually parses JSON and verifies `content_digest` without using the standard `validate_unit_done` contract.

#### 4.1.2 Target Architecture with `xtrax.campaign`
We refactor `src/aminx/host/campaign.py` to route all manifest execution through `xtrax.campaign.executors.LocalSequentialExecutor` (and allow opt-in `LocalProcessPoolExecutor` / `SlurmArrayExecutor` via CLI flag):

```
+-----------------------------------------------------------------------------------+
| aminx CLI: aminx campaign run / worker                                            |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| aminx.host.campaign.execute_manifest                                              |
|                                                                                   |
|  1. Load or Plan Manifest -> xtrax.campaign.manifest.CampaignManifest             |
|  2. Select Executor:                                                              |
|       - default: LocalSequentialExecutor(lease_seconds, heartbeat_interval)      |
|       - pool:    LocalProcessPoolExecutor(max_workers, recycle_interval=50)       |
|  3. Define aminx_unit_fn(row: ManifestRow) -> Mapping / Store                     |
|       - Invokes sample() / score() / family runner                                |
|       - Stages output in attempt-scoped dir: <out>.attempt-<fence_token>          |
|       - Executor promotes staging -> <out> and commits .done.json                 |
|       - If --pack: packages committed unit via pack_unit(unit.json + tar.zst)    |
|  4. executor.run(manifest, unit_fn=aminx_unit_fn, output_dir=...)                 |
|  5. Returns CampaignSummary (completed, skipped, failed, locked, elapsed)         |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| xtrax.campaign Engine (executors/local.py & lease.py)                              |
|                                                                                   |
|  - FileLock: atomic acquire with fence high-water (.fence), heartbeat thread      |
|  - Attempt Staging Isolation (<out>.attempt-<fence_token> / .partial.<uuid>)      |
|  - Lease Fencing: assert_holder() before rename, quarantine, and done stamping    |
|  - Marker Stamping: create_done_marker(AminxDoneMarkerAdapter) & fsync_directory  |
|  - Tier-1 / Deep Fast Resume skipping with reuse_log.jsonl                        |
+-----------------------------------------------------------------------------------+
```

#### 4.1.3 Unit Function Contract & Adapter Wiring
`aminx_unit_fn(row: ManifestRow) -> dict[str, Any]` bridges `aminx` row data to `xtrax.campaign`:
1. **Manifest Row Construction**:
   - `aminx.host.campaign.plan_campaign_manifest` continues to produce rows conforming to `campaign_manifest_v1`, delegating payload hashing to `xtrax.campaign.manifest.build_manifest_row` and `aminx_compatible_row_payload`.
   - `input_hash` is computed identically via `xtrax.campaign.manifest.compute_unit_input_hash`, which reproduces `aminx`'s canonical parameter digest.
2. **Execution Body**:
   - `aminx_unit_fn` calls `run_manifest_row_body(row, attempt_path, config)` which executes the core sampling / scoring logic without running manual locking loops.
   - The result store is populated at `attempt_path = Path(row.output_locator)`.
3. **Lease Fencing & Staging Isolation**:
   - The executor handles `FileLock` acquisition, heartbeat renewal, and commit promotion.
   - The executor provides `row.output_locator` redirected to sibling attempt staging path `<output_path>.attempt-<fence_token>`.
   - `aminx_unit_fn` writes directly to this attempt staging path.
   - The executor's `_stage_unit_output` moves the staging path to sibling `.partial.<uuid>` under `lock.assert_holder()` fencing, promotes atomically to `output_path`, and commits `.done.json`.
   - Stale workers whose lease was stolen fail `assert_holder()` with `LeaseLostError` and are discarded without overwriting active work.

#### 4.1.4 CLI Backward Compatibility Guarantee
All Typer CLI subcommands in `src/aminx/cli.py` (`campaign plan`, `worker`, `run`, `adopt-legacy`, `gates`, `ramp-plan`, `ramp-evaluate`) retain their exact signatures, flag names, and default values:
- `aminx campaign run`:
  - `--manifest-path` (Path, required)
  - `--row-hash` (repeatable list[str], optional)
  - `--continue-on-error / --no-continue-on-error` (bool, default False)
  - `--pack / --no-pack` (bool, default False)
  - `--summary-path` (Path, optional)
  - `--lock-backend` (`local_fs`, `distributed`, default `local_fs`)
  - `--lock-lease-seconds` (int, default 1800)
  - `--heartbeat-interval-seconds` (float, default 60.0)
  - *New optional flag (non-breaking)*: `--executor` (choices: `local-seq`, `local-pool`, default `local-seq`).
- `aminx campaign worker`:
  - `--manifest-path` (Path, required)
  - `--row-index` (int, optional)
  - `--row-hash` (str, optional)
  - `--pack / --no-pack` (bool, default False)
  - `--lock-backend` (enum, default `local_fs`)
  - `--lock-lease-seconds` (int, default 1800)
  - `--heartbeat-interval-seconds` (float, default 60.0)

When `aminx campaign worker` runs with `--row-index` or `--row-hash`, `LocalSequentialExecutor` executes with `row_indices=[idx]`, retaining single-worker CLI semantics.

---

### 4.2 Debt #3257: Unifying `xtrax.pack` in `aminx`

#### 4.2.1 Problem: Bifurcated Execution in `pack_shim.py`
In PR #270, `aminx` introduced `src/aminx/host/pack_shim.py` to enable `xtrax.pack`. However, `pack_shim.py` cloned 800+ lines of runner logic:
- `run_packed_manifest_row` duplicates `run_manifest_row` from `campaign.py`.
- `execute_packed_manifest` duplicates `execute_manifest` from `campaign.py`.
- Downstream tools had to branch on `is_packed` checks.

#### 4.2.2 Unified Pipeline Architecture & Packaging Lifecycle
We eliminate execution bifurcation by removing duplicate runner code in `pack_shim.py` and folding pack operations cleanly into the campaign execution lifecycle:

1. **Packaging Lifecycle & Separation of Concerns**:
   To prevent race conditions, orphan files, and quarantine collisions, the packaging lifecycle follows four distinct, sequential phases:
   - **Phase A: Attempt-Scoped Unit Execution**:
     The executor leases row and sets `row.output_locator` to the attempt staging directory (`<output_path>.attempt-<fence_token>`). The unit function `aminx_unit_fn` writes output data (Zarr arrays, metric JSONs, design sequences) into this attempt directory:
     ```python
     def aminx_unit_fn(row: ManifestRow) -> Path:
         # row.output_locator points to attempt staging path: <output_path>.attempt-<fence_token>
         attempt_path = Path(row.output_locator)
         attempt_path.mkdir(parents=True, exist_ok=True)
         # 1. Execute computational kernel (sample/score) into attempt staging path
         execute_kernel(row, attempt_path)
         # 2. Return attempt staging path with loose files populated
         return attempt_path
     ```
   - **Phase B: Executor Promotion & Done Marker Commit**:
     The executor asserts lease ownership via `lock.assert_holder()`, moves the attempt directory to sibling `<output_path>.partial.<uuid>`, computes `content_digest = compute_content_digest(partial_path)`, atomically renames `partial_path` to `output_path`, and writes canonical `<output_path>.done.json` with `content_digest_sha256` and `lease_fence`.
   - **Phase C: Post-Promotion Unit Packing (`--pack`)**:
     When packing is enabled (`--pack`), unit packing executes on the promoted `output_path`:
     1. `write_unit_record(output_path, input_digest=row.input_hash, run_id=...)` writes canonical `unit.json` recording member files, their individual SHA-256 hashes, and `record_digest`.
     2. `pack_unit(output_path, check=aminx_unit_check)` compresses loose files into seekable `members.tar.zst` (one zstd frame per USTAR member), writes `members.idx.json` frame index, updates `unit.json` storage block to `{"format": "alynxr.unit_pack/1", "pack": "members.tar.zst", "index": "members.idx.json"}`, and unlinks loose member files.
     3. *Invariant*: Because `xtrax.pack.unit_content_digest` computes the digest over uncompressed member contents, the digest is identical between loose and packed states, matching `content_digest_sha256` stamped in `.done.json` in Phase B.
   - **Phase D: Two-Tier Resume Verification**:
     On resume or re-check, `validate_unit_done(output_path, row)` validates the unit. As detailed in §4.5, it inspects `unit.json` and `is_packed()`, bypassing loose Zarr metadata requirements and avoiding false-positive quarantines.

2. **Canonical Output Record Emission**:
   - Completed units emit `unit.json` conforming to `xtrax.unit/1` schema:
     - `schema`: `"xtrax.unit/1"`
     - `input_digest`: `row.input_hash`
     - `content_digest`: sha256 digest over uncompressed member files
     - `record_digest`: cryptographic digest over the record's immutable core
     - `run_id`: execution identifier
     - `committed_at`: UTC ISO timestamp
     - `storage`: `"loose"` or `{"format": "alynxr.unit_pack/1", "pack": "members.tar.zst", "index": "members.idx.json"}`
   - Completed units also retain legacy compatibility by emitting `.done.json` (or `.done.json.packed` when archived), ensuring tools reading either format succeed.

3. **Seekable Archive Structure**:
   - `members.tar.zst`: USTAR archive with one zstd frame per member file.
   - `members.idx.json`: Frame offset index mapping relative member paths to frame byte offsets for $O(1)$ random access without full-archive decompression.
   - Loose files are safely deleted after verification via `fsync_tree()` and checksum matching.

4. **Retaining Reader Adapters in `pack_shim.py`**:
   - `pack_shim.py` is stripped of runner execution loops and preserved purely as an I/O adapter layer:
     - `resolve_unit_adapter(path)`
     - `read_packed_member(path, member)`
     - `aminx_unit_check(path)`
     - `adopt_legacy_unpacked_unit(path)`

---

### 4.3 `aminx` Pack Edge-Case Handling Matrix (Old vs New)

In accordance with the `xtrax-consumer-adoption` skill guidelines (avoiding bugs seen in alynxr #49), all pack edge cases are formally specified:

| Edge Case | Old Implementation (`campaign.py` / `pack_shim.py`) | New Unified Implementation (`xtrax.pack`) | Verification Control |
|---|---|---|---|
| **Failure Stamps** | Left loose partial store `<out>.partial.<uuid>` uncleaned on disk; no explicit failure marker. | Emits `<out>.unit.failed.json` capturing exception, stack trace, and timestamp. Partials quarantined to `<out>.quarantine.<uuid>` if corrupted. | `test_pack_failure_stamp_on_exception` |
| **Regenerated Loose Files** | If a worker crashed after writing loose files, subsequent run failed with `FileExistsError` or silent overwrite. | Resume checks detect un-committed loose files; verifies `content_digest` against previous `unit.json`; if unverified, isolates into quarantined path before re-execution. | `test_pack_resume_with_stale_loose_files` |
| **Empty Units** | Threw unhandled `ValueError` when tarring directory with zero member files. | Raises `UnitPackRefused("empty unit directory has no member files")` and marks unit failed; never creates corrupt empty `.tar.zst`. | `test_pack_refuses_empty_directory` |
| **`OSError` Wrapping** | Caught bare `OSError`, masking permission vs disk full vs lock contention. | Distinctly wraps into `LockUnavailable`, `CodecUnavailable`, or `UnitPackRefused` with actionable diagnostic messages. | `test_pack_oserror_discrimination` |
| **Resume Reuse Criteria** | Fast-path trusted existence of `.done.json` without verifying chunk presence or digest. | Configurable `resume_verify`: `"tier1"` (validates `unit.json` + root attrs) vs `"deep"` (computes uncompressed `content_digest` over all members). | `test_pack_resume_deep_vs_tier1` |
| **Existing On-Disk Units** | Legacy unpacked directories required manual migration script before running `--pack`. | Dual-adapter `resolve_unit_adapter` auto-detects legacy `.done.json` vs `unit.json` + `members.tar.zst`; `adopt_legacy_unpacked_unit` seamlessly packs existing stores in-place. | `test_pack_read_legacy_and_packed_interchangeable` |

---

### 4.4 `aminx` CI Hygiene Edge Fix

#### 4.4.1 Root Cause Analysis
In `aminx`, `tests/ci/test_select_affected_tests.py:129`:
```python
def test_all_src_modules_have_test_edges_or_allowlist():
    unreferenced = find_unreferenced_src_modules()
    allowlist = load_allowlist("scripts/ci/ci_test_selection_edges.toml")
    violations = [m for m in unreferenced if m not in allowlist]
    assert not violations, f"Modules missing from test edges allowlist: {violations}"
```
PR #277 added `src/aminx/eda/__init__.py`. Because `__init__.py` in `eda` was not referenced in test closure edges and was omitted from `ci_test_selection_edges.toml`, this test fails on `main`.

#### 4.4.2 Remediation
In `scripts/ci/ci_test_selection_edges.toml`:
Under the `[unreferenced_src]` table, add `'src/aminx/eda/__init__.py'` to the `allowlist` array:
```toml
[unreferenced_src]
allowlist = [
    # Existing allowlisted files...
    "src/aminx/eda/__init__.py",
]
```
Alternatively, create a trivial smoke test `tests/host/test_eda_init.py` that imports `aminx.eda` to close the edge. Both fixes will be verified to pass `test_select_affected_tests.py`.

---

### 4.5 Upstream Engine Hardening: Resolving Campaign & Pack Seam Collision in `src/xtrax/campaign/marker.py`

#### 4.5.1 Root Cause & Collision Analysis (F-02 & F-07)
Prior to this remediation, running `xtrax.campaign` with `--pack` triggered two fatal failure modes in `src/xtrax/campaign/marker.py`:
1. **`GroupNotFoundError` in `compute_content_digest`**:
   `compute_content_digest(path)` previously hardcoded:
   ```python
   if path.is_dir():
       from xtrax.run.zarr_integrity import zarr_content_digest
       return zarr_content_digest(path, include_provenance=True)
   ```
   If called on a packed unit (or staging directory where packing occurred), `zarr_content_digest()` attempted to open the directory as a loose Zarr group. Because loose files (`zarr.json` / `.zgroup`) were archived into `members.tar.zst`, `zarr.open_group()` failed with `zarr.errors.GroupNotFoundError: No group found in store`.
2. **Spurious Quarantine in `validate_unit_done` on Resume**:
   In `validate_unit_done()`, Tier-1 step 4 inspected directory contents:
   ```python
   meta_file = next(
       (path / name for name in ("zarr.json", ".zgroup", ".zarray") if (path / name).exists()),
       None,
   )
   if meta_file is None:
       if quarantine_on_failure:
           q_path = quarantine_output(path, reason="missing_zarr_root_metadata")
       return UnitValidationResult(is_valid=False, error_type="StoreMissingError", ...)
   ```
   On resume, a completed packed store lacks loose root metadata on disk. `validate_unit_done` declared the unit corrupt, moved it to `<campaign_dir>/.quarantine/`, and forced unnecessary re-execution.
3. **Decoupled Verification Contract (F-07)**:
   The hallucinated `xtrax.pack.verify_unit` symbol is expunged. Verification is decoupled into real upstream primitives:
   - `xtrax.pack.record_problem(record: dict) -> str | None`: verifies unit record fields, types, and cryptographic `record_digest`.
   - `xtrax.pack.is_packed(path: PathLike, record: dict | None = None) -> bool`: checks if unit is backed by committed `storage.pack`.
   - `xtrax.pack.unit_content_digest(path: PathLike) -> str`: uncompressed member digest verifying all member checksums through the index.
   - `xtrax.campaign.marker.validate_unit_done`: unified campaign-level two-tier resume validation.

#### 4.5.2 Exact Code Modifications in `src/xtrax/campaign/marker.py`

1. **Hardened `compute_content_digest(path: Path) -> str`**:
```python
def compute_content_digest(path: Path) -> str:
    """Compute the content digest of a zarr store directory, packed unit, or single file.

    If path is a directory containing a packed unit or unit record ('members.tar.zst'
    or 'unit.json'), uses :func:`xtrax.pack.unit_content_digest` to verify and digest
    uncompressed members. Otherwise, if path is a directory, it is digested with
    :func:`xtrax.run.zarr_integrity.zarr_content_digest` (the logical array digest).
    A single file is digested by streaming its bytes with a bounded 64 MB buffer.
    """
    if path.is_dir():
        if (path / "members.tar.zst").exists() or (path / "unit.json").exists():
            from xtrax.pack import unit_content_digest

            return unit_content_digest(path)
        from xtrax.run.zarr_integrity import zarr_content_digest

        return zarr_content_digest(path, include_provenance=True)
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(64 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()
```

2. **Hardened `validate_unit_done(output_path, expected_row, ...)` Step 4**:
```python
    # 4. Check store root metadata (Tier 1: < 5 ms, NO chunk reading).
    if path.is_dir():
        unit_record_path = path / "unit.json"
        is_packed_unit = (path / "members.tar.zst").exists() or unit_record_path.exists()
        if is_packed_unit:
            from xtrax.pack import is_packed, record_problem

            if not unit_record_path.exists():
                q_path = None
                if quarantine_on_failure:
                    q_path = quarantine_output(path, reason="missing_unit_record")
                return UnitValidationResult(
                    is_valid=False,
                    tier=1,
                    manifest_row_hash=expected_row.manifest_row_hash,
                    input_hash=expected_row.input_hash,
                    error_message=f"Packed unit has no unit.json record at {unit_record_path}",
                    error_type="StoreMissingError",
                    quarantine_path=q_path,
                )
            try:
                record_data = json.loads(unit_record_path.read_text(encoding="utf-8"))
            except Exception as exc:
                q_path = None
                if quarantine_on_failure:
                    q_path = quarantine_output(path, reason="corrupt_unit_record", exc=exc)
                return UnitValidationResult(
                    is_valid=False,
                    tier=1,
                    manifest_row_hash=expected_row.manifest_row_hash,
                    input_hash=expected_row.input_hash,
                    error_message=f"Corrupt unit.json in {unit_record_path}: {exc}",
                    error_type="DoneMarkerCorruptError",
                    quarantine_path=q_path,
                )
            problem = record_problem(record_data)
            if problem is not None:
                q_path = None
                if quarantine_on_failure:
                    q_path = quarantine_output(path, reason=f"invalid_unit_record_{problem}")
                return UnitValidationResult(
                    is_valid=False,
                    tier=1,
                    manifest_row_hash=expected_row.manifest_row_hash,
                    input_hash=expected_row.input_hash,
                    error_message=f"Invalid unit record in {unit_record_path}: {problem}",
                    error_type="DoneMarkerCorruptError",
                    quarantine_path=q_path,
                )
            if is_packed(path, record_data):
                pack_file = path / "members.tar.zst"
                if not pack_file.exists():
                    q_path = None
                    if quarantine_on_failure:
                        q_path = quarantine_output(path, reason="missing_pack_file")
                    return UnitValidationResult(
                        is_valid=False,
                        tier=1,
                        manifest_row_hash=expected_row.manifest_row_hash,
                        input_hash=expected_row.input_hash,
                        error_message=f"Packed unit specifies pack storage but {pack_file} is missing",
                        error_type="StoreMissingError",
                        quarantine_path=q_path,
                    )
        else:
            meta_file = next(
                (path / name for name in ZARR_ROOT_METADATA_FILES if (path / name).exists()),
                None,
            )
            if meta_file is None:
                q_path = None
                if quarantine_on_failure:
                    q_path = quarantine_output(path, reason="missing_zarr_root_metadata")
                return UnitValidationResult(
                    is_valid=False,
                    tier=1,
                    manifest_row_hash=expected_row.manifest_row_hash,
                    input_hash=expected_row.input_hash,
                    error_message=(
                        "Zarr store has no root metadata (expected one of "
                        f"{', '.join(ZARR_ROOT_METADATA_FILES)})"
                    ),
                    error_type="StoreMissingError",
                    quarantine_path=q_path,
                )
            # Verify metadata is valid JSON
            try:
                json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception as exc:
                q_path = None
                if quarantine_on_failure:
                    q_path = quarantine_output(path, reason="corrupt_zarr_metadata", exc=exc)
                return UnitValidationResult(
                    is_valid=False,
                    tier=1,
                    manifest_row_hash=expected_row.manifest_row_hash,
                    input_hash=expected_row.input_hash,
                    error_message=f"Corrupt Zarr metadata in {meta_file}: {exc}",
                    error_type="DoneMarkerCorruptError",
                    quarantine_path=q_path,
                )
```

3. **Tier-2 Deep Content Verification Integration**:
   When `deep_verify=True`, `validate_unit_done` invokes `compute_content_digest(path)`.
   Because `path` contains `members.tar.zst` and `unit.json`, `compute_content_digest` calls `xtrax.pack.unit_content_digest(path)`.
   This uncompresses and verifies each member against the frame index checksums, returning the identical `content_digest` computed before packing. If the archive was tampered with (e.g. injected bitflips), `UnitPackRefused` or digest disagreement is raised, safely setting `is_valid=False, error_type="ContentDigestMismatchError"`.

---

## 5. Architectural Design: `demistify`

### 5.1 Debt #3142: Adopting `xtrax.campaign` in `demistify`

#### 5.1.1 Correcting the Historical In-Wheel Misconception
In `demistify`'s prior adoption spec (`.praxia/docs/specs/261105_xtrax_pack_eda_adoption.md`), the author concluded:
> "xtrax.campaign is BLOCKED: the wheel packages src/xtrax only, and the campaign runner lives in controller/."

**Fact Correction**: `controller/` only houses the algorithm evolution search loop (#2181). The distributed campaign execution engine (`CampaignExecutor`, `LocalProcessPoolExecutor`, `LocalSequentialExecutor`, `FileLock`, `plan_campaign`, `validate_unit_done`) resides in `src/xtrax/campaign/` and **IS fully packaged and shipped in the `xtrax>=0.4.1b7` wheel**.

#### 5.1.2 Refactoring `scripts/analysis/batch_process.py`
Currently, `batch_process.py:418-448` executes batch trajectory analysis using an ad-hoc serial nested loop:
```python
# CURRENT FLAWED IMPLEMENTATION:
for pdb_path, xtc_path in pairs:
    try:
        process_trajectory_pair(pdb_path, xtc_path, output_dir, ...)
    except Exception as exc:
        if not skip_errors: raise
```
Deficiencies:
- No concurrency control: multiple cluster nodes pointing at the same output directory corrupt each other's Zarr chunks.
- No lease fencing: an interrupted or paused node resumes and writes into in-progress runs.
- No unified progress metrics or attempt isolation.

#### 5.1.3 Target Architecture for `demistify` Batch Processing
We refactor `scripts/analysis/batch_process.py` to use `xtrax.campaign`:

```
+-----------------------------------------------------------------------------------+
| demistify batch_process.py / demistify campaign run                               |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| 1. Discovery & Planning:                                                          |
|    - Discover (pdb, xtc) trajectory pairs                                         |
|    - Generate CampaignManifest using xtrax.campaign.manifest.plan_campaign        |
|    - Deterministic input_hash: sha256(pdb_digest + xtc_digest + vmm_params)       |
|    - output_locator: <output_dir>/<trajectory_id>/                                |
|                                                                                   |
| 2. Execution Engine:                                                              |
|    - If n_workers > 1: LocalProcessPoolExecutor(max_workers=n_workers,            |
|                                                recycle_interval=50,               |
|                                                lease_seconds=1800)                |
|    - If n_workers == 1: LocalSequentialExecutor(lease_seconds=1800)              |
|                                                                                   |
| 3. Unit Function: cimist_trajectory_unit(row: ManifestRow)                        |
|    - Extracts dihedral angles from trajectory                                     |
|    - Fits VMM and computes ciMIST conformational entropy                          |
|    - Writes loose outputs directly to attempt dir: <out>.attempt-<fence_token>    |
|    - Emits results.zarr, results.parquet, summary.json                            |
|                                                                                   |
| 4. Verification, Commit & Packing:                                                |
|    - Executor checks lease fence, promotes attempt dir -> final <out>             |
|    - Stamped .done.json with content_digest_sha256 and lease_fence                |
|    - If --pack: executes pack_cimist_unit on <out> -> unit.json + members.tar.zst |
|    - validate_unit_done on resume validates is_packed + record_problem            |
+-----------------------------------------------------------------------------------+
```

##### 5.1.3.1 Attempt Staging & Packing Lifecycle in `demistify`
To ensure clean isolation and avoid filesystem lock contention:
1. `cimist_trajectory_unit(row)` reads `row.output_locator`, which points to the sibling attempt staging directory `<output_path>.attempt-<fence_token>`.
2. All analysis outputs (`results.zarr`, `results.parquet`, `summary.json`) are written into this attempt staging directory.
3. The executor verifies the attempt staging directory under `lock.assert_holder()`, promotes it atomically to `<output_path>`, and commits `<output_path>.done.json`.
4. When `--pack` is active, `pack_cimist_unit(output_path)` runs on the promoted store: it emits `unit.json` via `write_unit_record()`, packages members into `members.tar.zst` with `members.idx.json` via `pack_unit()`, and unlinks loose files.
5. On resume, `validate_unit_done` uses `xtrax.pack.is_packed()` and `xtrax.pack.record_problem()` to verify the packed unit, and Tier-2 deep verification delegates to `xtrax.pack.unit_content_digest()` without raising `GroupNotFoundError`.

#### 5.1.4 Parameter Grid Sweeps (`g0_grid_sweep.py`, `phase5_k_sweep.py`)
`demistify` maintains parameter sweep scripts in `scripts/experiments/` that evaluate VMM mixture component counts $K$, frame counts, and optimizer variants.
We refactor `g0_grid_sweep.py` and `gate_optimizer_sweep.py` to use `xtrax.campaign.spec.ParamGrid`:
```python
from xtrax.campaign import ParamGrid, plan_campaign, LocalProcessPoolExecutor

grid = ParamGrid({
    "separation": [1, 2, 4, 8],
    "n_frames": [500, 1000, 2000, 5000],
    "optimizer": ["bfgs", "lbfgs"],
    "metric": ["geometric", "responsibility"],
})
manifest = plan_campaign(campaign_id="g0_grid_sweep", param_grid=grid, output_dir=sweep_dir)
executor = LocalProcessPoolExecutor(max_workers=8)
summary = executor.run(manifest, unit_fn=grid_sweep_unit, output_dir=str(sweep_dir))
```
This eliminates 300+ lines of ad-hoc nested loops and CSV parsing, replacing them with standard manifests, leasing, and summary reports.

---

### 5.2 Debt #3257: Formalizing `xtrax.pack` in `demistify`

#### 5.2.1 Status on `origin/main`
`demistify` has already adopted the core packing primitives in `src/demistify/unit_pack.py` and `PackedZarrStore` in `src/demistify/io/zarr_io.py`. However:
1. `batch_process.py` does not hook into cisternal telemetry logging.
2. Sweep runners in `scripts/experiments/` do not emit packed units, leaving 50,000+ loose Zarr chunk files per sweep.
3. Done marker creation in `unit_pack.py` does not record `lease_fence` from campaign leases.

#### 5.2.2 Formalization & Telemetry Hook Integration
1. **Telemetry Registration**:
   In `src/demistify/unit_pack.py`, register the cisternal telemetry hook:
   ```python
   from xtrax.pack import register_telemetry_hook

   def _cisternal_pack_telemetry(event: dict[str, Any]) -> None:
       try:
           from cisternal.telemetry import emit_event
           emit_event("xtrax.pack", event)
       except (ImportError, Exception):
           pass

   register_telemetry_hook(_cisternal_pack_telemetry)
   ```
2. **Standardized Sweep Packing**:
   All experiment sweeps in `scripts/experiments/` default to `--pack`, guaranteeing that trajectory units are emitted with seekable `members.tar.zst` and `members.idx.json`.
3. **Random Access Validation via `PackedZarrStore`**:
   Verify that `demistify viz` and downstream ciMIST analysis read directly from packed archives using `PackedZarrStore` without uncompressing to disk.

---

### 5.3 `demistify` Pack Edge-Case Handling Matrix

| Edge Case | Old Implementation (`batch_process.py`) | New Unified Implementation (`xtrax.pack` + `campaign`) | Verification Control |
|---|---|---|---|
| **Stale Trajectory Lock** | Interrupted worker left `.lock` file indefinitely; next run crashed or hung. | `FileLock` with TTL and lease stealing: stale leases (>1800s without heartbeat) are cleanly stolen and fenced. | `test_demistify_stale_lease_recovery` |
| **Corrupted Trajectory Frame** | Crashed halfway through writing `results.zarr`; subsequent run read partial array. | Writes staged in `.attempt-<fence_token>/`; promoted atomically only when complete and verified. Corrupted outputs quarantined. | `test_demistify_corrupted_partial_quarantine` |
| **Direct Packed Readback** | Unpacked Zarr chunks required full disk materialization before analysis. | `PackedZarrStore` reads slice chunks directly from `members.tar.zst` using frame index byte offsets. | `test_demistify_packed_zarr_store_readback` |
| **Telemetry Emission** | Zero telemetry for packing duration, compression ratio, or member count. | Emits `xtrax.pack` telemetry events with uncompressed bytes, compressed bytes, ratio, and duration. | `test_demistify_pack_telemetry_hook` |

---

## 6. Bio Projects & Domain Specifics (py2Dmol & Biosafe Prose)

### 6.1 py2Dmol Viewer Assessment
Per the `xtrax-consumer-adoption` skill guidelines (lines 91–100):
- **`aminx` Assessment**:
  - `aminx` is a sequence sampling and scoring engine (ProteinMPNN, ESM-IF, PottsMPNN). Its campaign and pack runners handle numerical Zarr arrays, logits, and design sequences.
  - Visual structural rendering is handled downstream in `tools/build_site.py` or separate report modules.
  - **Verdict**: **NOT-APPLICABLE** for the core campaign and pack runners. No py2Dmol plugin or dependency needed in `aminx.host.campaign` or `aminx.host.pack_shim`.
- **`demistify` Assessment**:
  - `demistify` already integrates py2Dmol for single-trajectory visualization (`demistify viz` in `src/demistify/cli.py:516` and `src/demistify/viz/py2dmol_plugin.py`).
  - Batch trajectory processing and parameter sweeps produce numerical conformational entropy matrices and graph edges (`results.zarr`, `results.parquet`).
  - **Verdict**: **BUILT-IN / PRESERVE**. The existing py2Dmol plugin on `origin/main` remains untouched; `PackedZarrStore` supplies data to `demistify viz` without requiring new viewer plugins.

### 6.2 Biosafe Prose Compliance
In compliance with `prose-biosafe` guidelines, all docstrings, log messages, error strings, and specifications avoid dual-use biological threat terminology, using standard computational biophysics terminology: "conformational entropy", "mutual information", "sequence sampling", "energy evaluation", "trajectory coordinates", and "residue coupling".

---

## 7. Dependency Pinning & Packaging Plan

### 7.1 Target Versions & Constraints
Both `aminx` and `demistify` require `xtrax>=0.4.1b7`:

1. **`aminx/pyproject.toml`**:
   ```toml
   [project]
   dependencies = [
       "xtrax[io,pack,campaign,export]==0.4.1b7",
       # ...
   ]

   [project.optional-dependencies]
   eda = ["xtrax[eda]==0.4.1b7"]
   onnx = ["xtrax[onnx]==0.4.1b7"]
   ```
   *Action*: Add `campaign` extra to `xtrax[...]` dependency specification in `aminx/pyproject.toml` if distinct, or rely on core `xtrax` distribution where `campaign` is packaged.

2. **`demistify/pyproject.toml`**:
   ```toml
   [project]
   dependencies = [
       "xtrax[pack,campaign]>=0.4.1b7",
       # ...
   ]

   [project.optional-dependencies]
   eda = ["xtrax[eda]>=0.4.1b7"]
   ```

### 7.2 JAX Version & Compatibility
- `xtrax` targets `jax>=0.4.35`. Both `aminx` and `demistify` environments run compatible JAX versions (0.4.35+ / 0.5.x).
- `xtrax.campaign.executors.local` enforces JAX persistent compilation cache initialization (`_resolve_compilation_cache_dir`) and memory hygiene (`block_until_ready`, `gc.collect()`), eliminating GPU OOM during multi-worker execution.

---

## 8. Test-Driven Development (TDD) Plan & Verification Controls

Every implementation task follows strict TDD: failing tests verifying the gap/defect must be committed and shown red first, followed by the minimal compliant implementation that turns them green.

### 8.1 Positive Verification Controls

Positive controls prove that standard, happy-path operations succeed across all modified surfaces.

| Test Case | Repository | Target Module | Exact Command | Success Criteria |
|---|---|---|---|---|
| **C-POS-1** | `aminx` | `tests/cli/test_campaign.py` | `uv run pytest tests/cli/test_campaign.py -v -o addopts=""` | All 21 tests pass; `aminx campaign plan`, `worker`, `run`, `gates`, `ramp-*` succeed. |
| **C-POS-2** | `aminx` | `tests/host/test_pack_shim.py` | `uv run pytest tests/host/test_pack_shim.py -v -o addopts=""` | All 16 tests pass; packed units emit valid `unit.json` and seekable `.tar.zst`. |
| **C-POS-3** | `aminx` | `tests/host/test_pack_shim_runner.py` | `uv run pytest tests/host/test_pack_shim_runner.py -v -o addopts=""` | Execution through unified runner succeeds with identical results for `--pack` and `--no-pack`. |
| **C-POS-4** | `aminx` | `tests/ci/test_select_affected_tests.py` | `uv run pytest tests/ci/test_select_affected_tests.py -v -o addopts=""` | Test passes with 0 violations; `src/aminx/eda/__init__.py` allowlisted. |
| **C-POS-5** | `demistify` | `tests/unit/test_unit_pack.py` | `pytest tests/unit/test_unit_pack.py -v -o addopts=""` | All 15 tests pass; `write_cimist_unit_record` and `pack_cimist_unit` verified. |
| **C-POS-6** | `demistify` | `tests/analysis/test_batch_campaign.py` | `pytest tests/analysis/test_batch_campaign.py -v -o addopts=""` | `batch_process.py` executes via `LocalProcessPoolExecutor`, emitting valid manifests and outputs. |
| **C-POS-7** | `demistify` | `tests/unit/test_packed_zarr.py` | `pytest tests/unit/test_packed_zarr.py -v -o addopts=""` | `PackedZarrStore` reads arrays directly from `members.tar.zst` byte-identical to loose store. |
| **C-POS-8** | Both | Lint & Formatting | `ruff check . && ruff format --check .` | 0 lint violations, 0 formatting issues across both worktrees. |
| **C-POS-9** | `xtrax` | `tests/campaign/test_marker_packed.py` | `uv run pytest tests/campaign/test_marker_packed.py -v -o addopts=""` | All tests pass; `compute_content_digest` and `validate_unit_done` verify packed units without `GroupNotFoundError` or false quarantine. |

---

### 8.2 Negative Verification Controls

Negative controls prove that corruptions, races, stale leases, and invalid inputs are safely detected and refused.

#### 8.2.1 Control C-NEG-1: Stale Lease Overwrite Refusal
- **Objective**: Prove that a worker whose lease was stolen cannot overwrite a stealer's commit or quarantine the stealer's in-progress output.
- **Scenario**:
  1. Worker 1 acquires lease on unit `row_001` with `lease_seconds=2`.
  2. Worker 1 is artificially suspended (simulated via sleep longer than lease TTL).
  3. Worker 2 steals the expired lease, raises the high-water fence `.fence` to 2, finishes computation, and commits `output_path`.
  4. Worker 1 resumes and attempts to commit output or quarantine `output_path`.
- **Expected Outcome**:
  Worker 1's call to `lock.assert_holder()` fails with `LeaseLostError`. The executor catches `LeaseLostError`, logs a warning, returns `UNIT_LOCK_HELD` (or `UNIT_DISCARDED`), discards Worker 1's `.partial.<uuid>` attempt directory, and leaves Worker 2's output completely intact.
- **Test File**: `aminx/tests/host/test_campaign_lease_fencing.py::test_stale_worker_commit_refused`

#### 8.2.2 Control C-NEG-2: Conflicting Concurrent Run Refusal
- **Objective**: Prove that two concurrent workers targeting the same row never run redundant computation or clobber lock files.
- **Scenario**:
  1. Worker A acquires `FileLock` for `row_002`.
  2. Worker B attempts to acquire `FileLock` on `row_002` while Worker A's lease is active.
- **Expected Outcome**:
  Worker B raises `LockUnavailableError`. The executor reports `outcome="lock_held"` and records the row in `locked_units` in `CampaignSummary`. Worker B does not invoke the unit computation kernel.
- **Test File**: `aminx/tests/host/test_campaign_locking.py::test_concurrent_row_lock_contention`

#### 8.2.3 Control C-NEG-3: Corrupted Archive Bitflip Refusal
- **Objective**: Prove that a modified or truncated `members.tar.zst` is detected as corrupt and refused by readers and campaign resume verifiers.
- **Scenario**:
  1. A valid unit `row_003` is packed with `pack_unit`.
  2. A bitflip mutation is injected into the byte body of `members.tar.zst`.
  3. Deep verification via `validate_unit_done(output_path, expected_row, deep_verify=True)` or member retrieval via `read_unit_member(output_path, "results.zarr/zarr.json")` is called.
- **Expected Outcome**:
  `validate_unit_done(output_path, expected_row, deep_verify=True)` calls `compute_content_digest(output_path)`, which delegates to `unit_content_digest(output_path)`. The corrupted zstd frame in `members.tar.zst` fails member checksum or decompression verification, raising `UnitPackRefused("R-U2", ...)` or returning `UnitValidationResult(is_valid=False, error_type="ContentDigestMismatchError")`. Similarly, `read_unit_member` raises `UnitPackRefused("R-U2", ...)`. Corrupted archive data is never returned to the caller. Additionally, if `unit.json` is tampered with, `record_problem(record_data)` returns `"record_digest does not match the record's core"` and `validate_unit_done` fails with `error_type="DoneMarkerCorruptError"`.
- **Test File**: `demistify/tests/unit/test_unit_pack_corruption.py::test_corrupted_archive_refused` and `xtrax/tests/campaign/test_marker_packed.py::test_packed_unit_deep_verify_detects_bitflip`

#### 8.2.4 Control C-NEG-4: Corrupted Done Marker Recomputation
- **Objective**: Prove that a done marker with a tampered or mismatched `content_digest_sha256` is refused by resume checks.
- **Scenario**:
  1. A completed unit has valid output and `.done.json` marker.
  2. The `.done.json` file is modified with a corrupted `content_digest_sha256`.
  3. Campaign execution resumes with `resume_verify="deep"`.
- **Expected Outcome**:
  `validate_unit_done` fails with `ContentDigestMismatchError`. The unit is NOT skipped; the executor isolates the corrupted store to quarantine and recomputes the unit from scratch.
- **Test File**: `aminx/tests/host/test_campaign_resume.py::test_corrupted_done_marker_triggers_recompute`

---

### 8.3 CLI Parameter Parity Verification

To prove 100% CLI backward compatibility in `aminx` and `demistify`, automated CLI parity tests shall execute the exact invocation strings used by existing production workflows and verify identical exit codes, parameter acceptance, and stdout/stderr schema.

#### 8.3.1 `aminx` CLI Parity Matrix
```bash
# Test 1: aminx campaign plan
aminx campaign plan \
  --inputs tests/data/sample_inputs.csv \
  --campaign-id test_camp_01 \
  --manifest-path test_out/manifest.json \
  --output-root test_out/runs \
  --designs-per-library-type 2 \
  --samples-chunk-size 1

# Test 2: aminx campaign worker with --no-pack (default)
aminx campaign worker \
  --manifest-path test_out/manifest.json \
  --row-index 0 \
  --lock-backend local_fs \
  --lock-lease-seconds 600

# Test 3: aminx campaign worker with --pack
aminx campaign worker \
  --manifest-path test_out/manifest.json \
  --row-index 1 \
  --pack \
  --lock-backend local_fs

# Test 4: aminx campaign run
aminx campaign run \
  --manifest-path test_out/manifest.json \
  --continue-on-error \
  --pack \
  --summary-path test_out/summary.json

# Test 5: aminx campaign gates
aminx campaign gates \
  --manifest-path test_out/manifest.json \
  --output-root test_out/runs
```
*Verification*: All 5 commands exit 0 and emit expected output files.

#### 8.3.2 `demistify` CLI Parity Matrix
```bash
# Test 1: demistify batch processing with existing args
python scripts/analysis/batch_process.py \
  --input-dir tests/data/trajectories \
  --output-dir test_out/batch_runs \
  --pattern "*.pdb" \
  --n-workers 2 \
  --vmm-components 3 \
  --chunk-size 100 \
  --skip-errors \
  --pack

# Test 2: demistify single run
demistify run \
  --pdb tests/data/system.pdb \
  --xtc tests/data/system.xtc \
  --output-dir test_out/single_run \
  --pack
```
*Verification*: Both commands exit 0, invoke `xtrax.campaign` executors, and emit canonical packed units.

---

## 9. Step-by-Step Implementation Plan

### Phase 2: Adversarial Review & Specification Freeze
1. Dispatch adversarial review swarm against this specification.
2. Resolve any identified blockers or ambiguities until `ZERO_BLOCKERS` achieved.

### Phase 3: Worktree Setup & Implementation
1. **Upstream `xtrax` Engine Hardening (F-02 & F-07)**:
   - Write failing unit tests in `xtrax/tests/campaign/test_marker_packed.py` testing `compute_content_digest()` and `validate_unit_done()` on packed units.
   - Implement hardened `compute_content_digest()` in `src/xtrax/campaign/marker.py` delegating to `xtrax.pack.unit_content_digest()` when `members.tar.zst` or `unit.json` is present.
   - Implement hardened `validate_unit_done()` in `src/xtrax/campaign/marker.py` verifying packed units via `xtrax.pack.is_packed()` and `record_problem()`.
   - Verify `uv run pytest tests/campaign/test_marker_packed.py -v` passes.
2. **Worktree Initialization**:
   - Create `.worktrees/feat-261110-campaign-pack-adoption` in `aminx`.
   - Create `.worktrees/feat-261110-campaign-adoption` in `demistify`.
3. **`aminx` TDD Implementation**:
   - Step 1: Commit CI test selection edge fix in `scripts/ci/ci_test_selection_edges.toml` and verify `test_select_affected_tests.py` passes.
   - Step 2: Write failing unit and negative-control tests for `LocalSequentialExecutor` integration and lease fencing in `tests/host/test_campaign_execution.py`.
   - Step 3: Refactor `src/aminx/host/campaign.py:execute_manifest` to use `xtrax.campaign.executors.LocalSequentialExecutor` and `FileLock`.
   - Step 4: Fold `pack_shim.py` runner logic into `campaign.py`, standardizing unit completion on `write_unit_record()` and `pack_unit()`.
   - Step 5: Verify all 37 tests in `test_campaign.py`, `test_pack_shim.py`, and `test_pack_shim_runner.py` pass green.
4. **`demistify` TDD Implementation**:
   - Step 1: Write failing unit and negative-control tests in `tests/analysis/test_batch_campaign.py`.
   - Step 2: Refactor `scripts/analysis/batch_process.py` to use `xtrax.campaign.manifest.plan_campaign` and `LocalProcessPoolExecutor`.
   - Step 3: Register cisternal telemetry hook in `src/demistify/unit_pack.py`.
   - Step 4: Refactor `scripts/experiments/g0_grid_sweep.py` to use `xtrax.campaign.spec.ParamGrid`.
   - Step 5: Verify all tests in `tests/unit/test_unit_pack.py` and `tests/analysis/test_batch_campaign.py` pass green.

### Phase 4: Verification, Linting & PR Creation
1. Run full test suites in both worktrees:
   - `aminx`: `pytest tests/cli/test_campaign.py tests/host/test_pack_shim.py tests/ci/test_select_affected_tests.py -v -o addopts=""`
   - `demistify`: `pytest tests/unit/test_unit_pack.py tests/analysis/test_batch_campaign.py -v -o addopts=""`
2. Run format and style linters:
   - `ruff check .` and `ruff format --check .`
3. Push branches:
   - `git push -u origin feat/261110-aminx-consumer-adoption`
   - `git push -u origin feat/261110-demistify-consumer-adoption`
4. Open Pull Requests via `gh pr create` referencing debts #3142 and #3257.

---

## 10. Risk Analysis & Mitigation Matrix

| Risk | Likelihood | Impact | Mitigation Strategy |
|---|---|---|---|
| **R1: JAX Compilation Cache Hang in Subprocesses** | Medium | High | `LocalProcessPoolExecutor` explicitly configures `_resolve_compilation_cache_dir` and worker chunked recycling (`recycle_interval=50`). Sequential fallback (`LocalSequentialExecutor`) is default in `aminx`. |
| **R2: Legacy Done Marker Parsing Breakage** | Low | High | `AminxDoneMarkerAdapter` provides bidirectional conversion between legacy aminx `.done.json` format and canonical `CampaignDoneMarker`. `resolve_unit_adapter` auto-detects format. |
| **R3: Shared Filesystem Clock Skew on Leases** | Medium | Medium | `FileLock` enforces `CLOCK_DRIFT_GRACE_PERIOD_S = 300` seconds and monotonic high-water `.fence` file tracking, preventing spurious lease expirations. |
| **R4: Large Tarball Seek Latency on Cluster NFS** | Low | Medium | `xtrax.pack` generates `members.idx.json` frame index alongside `members.tar.zst`, allowing direct byte-offset seek without stream decompression. |
| **R5: Concurrent CLI Invocations Colliding** | Medium | Medium | Guaranteed by `FileLock.acquire(blocking=False)` which reports `UNIT_LOCK_HELD` rather than crashing or corrupting active work. |
| **R6: Pack & Campaign Seam Collision on Resume** | High (unmitigated) | Critical | Resolved by updating `src/xtrax/campaign/marker.py`: `compute_content_digest` delegates to `unit_content_digest` (preventing `GroupNotFoundError`), and `validate_unit_done` validates packed stores via `is_packed` and `record_problem` without demanding loose Zarr metadata (preventing false quarantine). |

---

## 11. Traceability & Sign-Off Checklist

- [x] Frontmatter specifies `task_id: 261110_xtrax-debt-burn-down-and-consumer-adoption`, `track: 2`, `debts: [3142, 3257]`.
- [x] Comprehensive architectural design for `aminx` (#3142 campaign, #3257 pack, CI allowlist edge fix).
- [x] Comprehensive architectural design for `demistify` (#3142 campaign, #3257 pack formalization).
- [x] Upstream `xtrax.campaign.marker` hardened to support packed units in `compute_content_digest` and `validate_unit_done` (resolving Critic 2 F-02).
- [x] Packaging lifecycle explicitly specified across attempt staging, executor promotion, done marker commit, and packing (resolving Critic 2 F-02).
- [x] Hallucinated `xtrax.pack.verify_unit` expunged; real APIs (`record_problem`, `unit_content_digest`, `is_packed`, `validate_unit_done`) cited throughout (resolving Critic 2 F-07).
- [x] CLI backward compatibility explicitly preserved with full parameter matrices.
- [x] Isolated worktree directives with exact paths and branches specified.
- [x] Pack Old-vs-New Edge Case Handling tables for both repositories.
- [x] Positive and negative verification controls with exact test commands and failure scenarios.
- [x] Bio projects py2Dmol and biosafe prose assessments completed.
- [x] TDD test execution steps documented with exact commands.
