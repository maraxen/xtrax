---
title: xtrax.campaign runner implementation epic (EPIC-261105-CAMPAIGN-RUNNER)
description: Track decomposition (T1-T5, GATE-01..11) for the campaign runner that shipped as PR #267; salvaged 261119 from an untracked worktree file as the historical record.
status: implemented
task_id: 261105_xtrax-campaign-runner
date: '261105'
salvaged_by: 261119_branch-triage
---
# Implementation Epic: `xtrax.campaign`
## Distributed Campaign Runner & Execution Engine

**Document ID**: `EPIC-261105-CAMPAIGN-RUNNER`  
**Task ID**: `261105_xtrax-campaign-runner`  
**Milestone**: P5 Implementation  
**Governing Specification**: `docs/spec/261105_campaign_runner_spec.md` (APPROVED & CONVERGED)  
**Author**: `teamwork_worker_p4_editor` (Specification Editor & Epic Creator)  
**Status**: APPROVED & READY FOR P5 IMPLEMENTATION  
**Date**: 2026-11-05  
**Target Path**: `docs/spec/261105_campaign_runner_epic.md`  

---

## 1. Executive Summary & Epic Objectives

This Epic decomposes the implementation of `xtrax.campaign` (the distributed, scheduler-agnostic campaign runner shipped directly in the `xtrax` wheel) into atomic, test-first tracks. 

The primary objective is the resolution of **Debt #3142**: eliminating over 3,000 lines of fragile, duplicated multi-node execution, leasing, and resume logic across `aminx`, `alynxr`, and `adaptyv` by providing an in-wheel, production-grade campaign execution substrate.

### Core Architectural Invariants
All implementation tracks must strictly preserve the three core invariants established and converged in `SPEC-261105-CAMPAIGN-RUNNER`:
1. **The Dispatch-Independence Gate**:
   $$\text{plan}(\text{CampaignSpec}) \longrightarrow \text{CampaignManifest}$$
   Planning is 100% pure host computation: 0 GPUs, 0 network side-effects, 0 child processes, deterministic canonical JSON hashing.
2. **aminx Manifest Slice Parity**:
   The domain parameter hash (`input_hash`) stored on each manifest row and done marker reproduces `aminx.host.campaign.compute_unit_input_hash` bit-for-bit against golden test vectors.
3. **Wheel Packaging Boundaries & Self-Sufficiency**:
   All public code resides strictly in `src/xtrax/campaign/` within the distributed wheel (`packages = ["src/xtrax"]`). There are zero imports of `controller/` or `bathos`.

---

## 2. Track Dependency & Phasing Graph

Implementation is organized into five decoupled tracks executed with strict Test-Driven Development (TDD).

```
                      ┌────────────────────────────────────────┐
                      │ Track 1: Core Schemas, Manifest        │
                      │ Planning & Canonical Hashing           │
                      └───────────────────┬────────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
┌────────────────────────────────────────┐ ┌────────────────────────────────────────┐
│ Track 2: Distributed Leasing,          │ │ Track 3: Executor Seam & Native        │
│ Heartbeat & Verified Resume            │ │ Schedulers                             │
└───────────────────┬────────────────────┘ └────────────────────┬───────────────────┘
                    │                                           │
                    └─────────────────────┬─────────────────────┘
                                          ▼
                      ┌────────────────────────────────────────┐
                      │ Track 4: CLI Surface &                 │
                      │ `xtrax sweep` Deprecation Shim         │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │ Track 5: End-to-End Acceptance Gate &  │
                      │ Concurrency Stress Verification        │
                      └────────────────────────────────────────┘
```

---

## 3. Detailed Track Breakdown

### Track 1: Core Schemas, Manifest Planning & Canonical Hashing
**Focus**: Pure host planning, declarative specification schemas, canonical JSON serialization, and unit identity vs aminx parameter hash separation.

- **Primary Source Files**:
  - `src/xtrax/campaign/__init__.py`
  - `src/xtrax/campaign/spec.py`
  - `src/xtrax/campaign/manifest.py`
- **Primary Test Files**:
  - `tests/campaign/test_spec.py`
  - `tests/campaign/test_canonical_json.py`
  - `tests/campaign/test_manifest.py`

#### Work Items & Tasks:
- [ ] **[T1-01] Declarative Specification Schemas (`spec.py`)**:
  - Implement `@dataclass(frozen=True, slots=True)` classes: `ParamGrid`, `ArmSpec`, `CampaignSpec`.
  - Enforce field validation: ensure `arms` non-empty, `chunk_size >= 1`, `samples_per_target >= 1`.
  - Implement serialization to/from dictionary and TOML configuration files.
  - *Test-First Gate*: `tests/campaign/test_spec.py` verifies initialization, immutability, validation errors on invalid parameters, and TOML roundtripping.

- [ ] **[T1-02] Canonical JSON Serialization & Normalization (`zarr_integrity.py` / `manifest.py`)**:
  - Implement `normalize_json_value(value: Any) -> Any`:
    - Normalize `float` and `np.floating` `-0.0` strictly to `0.0` (`0.0 if value == 0.0 else value`).
    - Sort dictionary items with `key=lambda kv: str(kv[0])` to prevent mixed-type key crashes (`TypeError`).
    - Recursively normalize nested dicts, lists, tuples, and numpy scalar/array types.
    - Normalize Unicode strings to NFC form.
  - Implement `canonical_json_bytes(value: Any) -> bytes` with `separators=(",", ":")`, `ensure_ascii=False`, `allow_nan=False`.
  - *Test-First Gate*: `tests/campaign/test_canonical_json.py` tests `-0.0` vs `0.0` hash equality, mixed string/int dict key sorting without crash, and rejection of non-finite floats (`NaN`, `Infinity`).

- [ ] **[T1-03] Manifest Row & Manifest Schemas (`manifest.py`)**:
  - Implement `ResourceRequirements`, `ManifestRow`, `ManifestMetadata`, `CampaignManifest`.
  - Include both `manifest_row_hash: str` (global unit identity) and `input_hash: str` (aminx parameter parity hash) on `ManifestRow`.
  - *Test-First Gate*: Verify schema versions (`"campaign_manifest_v1"`), slot constraints, and immutability.

- [ ] **[T1-04] Pure Host Manifest Planner (`manifest.py`)**:
  - Implement `compute_manifest_row_hash(payload: dict[str, Any]) -> str`.
  - Implement `compute_unit_input_hash(spec_dict: dict[str, Any], **kwargs: Any) -> str` matching `aminx.host.campaign.compute_unit_input_hash`.
  - Implement `plan_campaign(spec: CampaignSpec, output_manifest_path: str | Path | None = None) -> CampaignManifest`:
    - Inspect input parameters against `_CALLABLE_FIELDS`; immediately raise `SpecPartitionError` if callables are detected.
    - Expand parameter grid cartesian product across targets and arms.
    - Partition total samples into uniform chunks of size `spec.chunk_size`.
    - Compute deterministic static resource requirements (`ResourceRequirements`).
    - Compute `manifest_row_hash` and `input_hash` for each row.
    - Write formatted canonical JSON to disk if path provided.
  - *Test-First Gate*: `tests/campaign/test_manifest.py` validates 12-unit expansion without JAX imports, determinism across repeated calls, and `SpecPartitionError` negative control.

---

### Track 2: Distributed Leasing, Heartbeat & Verified Resume
**Focus**: Multi-node POSIX/NFS leasing, heartbeat CAS renewal, singleton commit token protocol, clock drift tolerance, Tier-1/Tier-2 resume verification, and defensive quarantine.

- **Primary Source Files**:
  - `src/xtrax/campaign/lease.py`
  - `src/xtrax/campaign/marker.py`
- **Primary Test Files**:
  - `tests/campaign/test_lease.py`
  - `tests/campaign/test_heartbeat.py`
  - `tests/campaign/test_commit_protocol.py`
  - `tests/campaign/test_marker.py`
  - `tests/campaign/test_quarantine.py`

#### Work Items & Tasks:
- [ ] **[T2-01] Atomic File Leasing & Stale Lease Stealing (`lease.py`)**:
  - Define `CLOCK_DRIFT_GRACE_PERIOD_S = 300`.
  - Implement `CampaignLock` dataclass with `acquired_at_unix_s` as immutable lease epoch.
  - Implement `fsync_directory(path: Path) -> None` catching and gracefully suppressing `EINVAL`, `ENOTSUP`, `EOPNOTSUPP`, and `EBADF` on NFS client directory file descriptors.
  - Implement atomic acquisition via `os.open(lock_path, O_CREAT | O_EXCL | O_WRONLY)`.
  - Implement stale lease stealing: verify both `now >= lease_expires + 300` and `now - mtime >= lease_seconds + 300`; write candidate steal file with new `acquired_at_unix_s` epoch, `os.replace`, and verify CAS readback.
  - *Test-First Gate*: `tests/campaign/test_lease.py` verifies mutual exclusion under concurrency, stale lease stealing past 300s margin, and rejection when within grace period.

- [ ] **[T2-02] Heartbeat Thread with CAS Expiration Guard (`lease.py`)**:
  - Implement `HeartbeatThread`:
    - Enforce pre-renewal expiry guard: `if time.time() >= self.current_lease_expires_at - 60.0: self.abort_event.set(); return False`.
    - Enforce pre-renewal readback verifying `owner_token`, `heartbeat_count`, and immutable `acquired_at_unix_s`.
    - Atomically stage renewal file, execute `os.replace`, call `fsync_directory`, and verify CAS post-readback.
    - Expose `abort_event` for worker thread monitoring.
  - *Test-First Gate*: `tests/campaign/test_heartbeat.py` simulates thread renewal, expiry guard triggering when delayed $> (lease - 60s)$, and CAS abort when lock is externally stolen.

- [ ] **[T2-03] Singleton Commit Token Protocol & Generation Sealing (`lease.py`)**:
  - Implement `FileLock.commit_output(...)`:
    1. Check if `<output_path>.done.json` exists; if so, discard attempt and return `StolenAttemptDiscarded`.
    2. Check `<generation_dir>/.generation_sealed`; raise `GenerationSealedError` if sealed.
    3. Atomically create singleton `<output_path>.commit` via `os.open(path, O_CREAT | O_EXCL | O_WRONLY)`. NO attempt UUID in path!
    4. Write commit metadata (`attempt_uuid`, `owner_token`, `committed_at`).
    5. Re-verify `<output_path>.lock` ownership, expiration, and `not heartbeat.abort_event.is_set()`.
    6. Promote `.partial.<attempt_uuid>` to `<output_path>` via `os.replace`.
    7. Write `<output_path>.done.json.tmp`, `fsync_file`, and `os.replace` to `<output_path>.done.json`.
    8. Durabilize parent directory and unlink `.commit` and `.lock`.
  - *Test-First Gate*: `tests/campaign/test_commit_protocol.py` asserts singleton mutual exclusion, zombie worker rejection, and generation sealing rejection.

- [ ] **[T2-04] Done Marker Schemas & Adapter Consolidation (`marker.py`)**:
  - Implement `ProducerInfo`, `CampaignDoneMarker` (`"campaign_done_marker_v3"`).
  - Implement `DoneMarkerAdapter` base protocol.
  - Implement `AminxDoneMarkerAdapter` (translating aminx v3 markers) and `AlynxrFragmentAdapter` (translating alynxr fold_stats records) directly in `src/xtrax/campaign/marker.py`, with zero references to phantom `unit_index_adapters.py`.
  - *Test-First Gate*: `tests/campaign/test_marker.py` tests parsing, validation, and adapter translation.

- [ ] **[T2-05] Decoupled Two-Tier Resume & Defensive Quarantine (`marker.py`)**:
  - Implement `validate_unit_done(output_path, expected_row: ManifestRow, deep_verify: bool = False) -> UnitValidationResult`:
    - Tier-1 Fast Resume (< 5 ms): checks `.done.json`, validates `manifest_row_hash` and `input_hash`, validates Zarr root metadata (`.zgroup`/`.zarray` header attributes) without reading chunk bytes.
    - Tier-2 Deep Verification (`deep_verify=True`): streams chunk bytes with bounded 64 MB buffer and recomputes `zarr_content_digest`.
  - Implement `quarantine_output(output_path: Path, reason: str, exc: Exception | None = None) -> Path`:
    - Handles `errno.ENOSPC` by renaming BOTH output store and `.done.json` in-place to `.corrupt.<uuid>`.
    - Handles `errno.EXDEV` by falling back to `shutil.move`.
  - *Test-First Gate*: `tests/campaign/test_quarantine.py` simulates bit rot, missing metadata, ENOSPC, EXDEV, and verifies Tier-1 executes in $< 5$ ms.

---

### Track 3: Executor Seam & Native Schedulers
**Focus**: Standardized `CampaignExecutor` protocol, local sequential execution with JAX memory hygiene, process pool execution with chunked recycling and persistent compilation cache, and pure host SLURM array script generation with sparse resume.

- **Primary Source Files**:
  - `src/xtrax/campaign/executors/protocol.py`
  - `src/xtrax/campaign/executors/local.py`
  - `src/xtrax/campaign/executors/slurm.py`
  - `src/xtrax/campaign/executors/__init__.py`
- **Primary Test Files**:
  - `tests/campaign/executors/test_protocol.py`
  - `tests/campaign/executors/test_local.py`
  - `tests/campaign/executors/test_slurm.py`

#### Work Items & Tasks:
- [ ] **[T3-01] `CampaignExecutor` Protocol & `CampaignSummary` (`executors/protocol.py`)**:
  - Define `CampaignSummary` dataclass: `campaign_id`, `total_units`, `completed_units`, `skipped_units`, `failed_units`, `elapsed_seconds`, `failures`.
  - Define `CampaignExecutor(Protocol)` with `run(...)` signature.
  - Define `cancel_tasks(task_ids: list[Any]) -> None` method on executor interface for straggler cancellation.
  - *Test-First Gate*: Protocol conformance tests with mock implementations.

- [ ] **[T3-02] `LocalSequentialExecutor` with Cache & Memory Hygiene (`executors/local.py`)**:
  - Implement single-process loop iterating through planned rows.
  - Integrate `FileLock`, skipping units that pass Tier-1 resume.
  - Memory Hygiene: After each unit, execute device tensor blocking (`block_until_ready`), `del state`, and `gc.collect()`.
  - Prohibit per-unit `jax.clear_caches()`; invoke cache clearing only upon arm backend or padding bucket transitions.
  - *Test-First Gate*: `tests/campaign/executors/test_local.py` tests sequential execution, error handling with `max_failures`, and memory stability.

- [ ] **[T3-03] `LocalProcessPoolExecutor` with Chunked Recycling & JAX Cache (`executors/local.py`)**:
  - Implement multiprocessing pool executor using `multiprocessing.get_context("spawn")`.
  - Configurable chunked recycling: `recycle_interval: int | None = 50`.
  - Worker process initializer: configure `jax.config.update("jax_compilation_cache_dir", ...)` and `jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)`.
  - *Test-First Gate*: Verify worker recycling after 50 units and constant RSS memory bounds.

- [ ] **[T3-04] `SlurmArrayExecutor` with Sparse Array Resume (`executors/slurm.py`)**:
  - Implement pure host script generator emitting portable `#SBATCH` submission script.
  - Implement `get_pending_row_indices(manifest, output_dir)` scanning `.done.json` markers.
  - Implement `format_slurm_array_indices(indices, max_concurrent)` converting sparse lists into range strings (e.g. `12,15,44-50%20`).
  - Boundary check: if `len(pending_indices) == 0`, exit cleanly without generating an invalid empty script.
  - Implement `cancel_tasks(omitted_indices)` issuing `scancel <job_id>_<task_id>`.
  - *Test-First Gate*: `tests/campaign/executors/test_slurm.py` tests range formatting, sparse resume script output, `bash -n` validation, and empty boundary handling.

---

### Track 4: CLI Surface & `xtrax sweep` Deprecation
**Focus**: Command-line interface verbs (`plan`, `run`, `status`, `verify`), registry registration, and seamless backward-compatibility shim for `xtrax sweep`.

- **Primary Source Files**:
  - `src/xtrax/cli/campaign_verb.py`
  - `src/xtrax/cli/sweep_verb.py`
  - `src/xtrax/cli/registry.py`
- **Primary Test Files**:
  - `tests/campaign/test_cli.py`
  - `tests/cli/test_sweep_deprecation.py`

#### Work Items & Tasks:
- [ ] **[T4-01] CLI Campaign Verbs (`cli/campaign_verb.py`)**:
  - Implement `xtrax campaign plan`: takes `campaign_spec.toml`, emits `manifest.json`.
  - Implement `xtrax campaign run`: takes `--manifest`, optional `--row-index` (for SLURM task execution), `--executor` (`local-seq`, `local-pool`, `slurm`), and `--require-approval`.
  - Implement `xtrax campaign status`: inspects output directory, reporting completed, running, failed, and quarantined units.
  - Implement `xtrax campaign verify`: runs Tier-1 or `--deep` Tier-2 audit across stores.
  - Integrate approval gate (T2-32): when `--require-approval` is passed, call `assert_campaign_approved(campaign_id)` against `.praxia/loop_human_gates.toml`.
  - Wire verbs into `src/xtrax/cli/registry.py`.
  - *Test-First Gate*: `tests/campaign/test_cli.py` tests subcommands, exit codes, and approval gate enforcement.

- [ ] **[T4-02] `xtrax sweep` Deprecation Compatibility Shim (`cli/sweep_verb.py`)**:
  - Refactor `src/xtrax/cli/sweep_verb.py` to wrap `xtrax.campaign`:
    - Emit `FutureWarning` indicating deprecation as of 0.5.0 and scheduled removal in 0.6.0.
    - Translate `[sweep.axes]` into single-arm `CampaignSpec`.
    - Plan manifest and execute via `LocalSequentialExecutor`.
  - *Test-First Gate*: `tests/cli/test_sweep_deprecation.py` tests that legacy TOML sweeps run to completion, produce expected outputs, and raise `FutureWarning`.

---

### Track 5: End-to-End Acceptance Gate & aminx Manifest Slice Parity
**Focus**: Golden vector parity verification against `aminx.host.campaign`, end-to-end multi-arm multi-target execution with real Zarr outputs, and multi-process concurrency stress testing.

- **Primary Test Files**:
  - `tests/campaign/test_aminx_parity.py`
  - `tests/campaign/test_e2e_campaign.py`
  - `tests/campaign/test_concurrency_stress.py`

#### Work Items & Tasks:
- [ ] **[T5-01] aminx Manifest Slice Parity Regression Suite (`test_aminx_parity.py`)**:
  - Encode golden test vectors pre-computed from production aminx campaigns.
  - Assert that `compute_unit_input_hash` and `ManifestRow.input_hash` match expected hash strings with 0 bit divergence.
  - Negative control: perturb float from `0.1` to `0.1000001` and verify hash flip.
  - *Verification Gate*: Bit-for-bit parity confirmed (AC-02).

- [ ] **[T5-02] End-to-End Multi-Arm Campaign Execution (`test_e2e_campaign.py`)**:
  - Construct a full end-to-end campaign with 2 targets, 2 arms (e.g. simulated Boltz-2 and Chai-1), and parameter axes.
  - Execute under `LocalProcessPoolExecutor` with real Zarr store writes and `.done.json` generation.
  - Verify all 12 Acceptance Criteria (AC-01 through AC-12) pass.
  - *Verification Gate*: End-to-end execution, resume, and output store integrity verified.

- [ ] **[T5-03] Multi-Process Concurrency & Crash Recovery Stress Test (`test_concurrency_stress.py`)**:
  - Spawn 8 concurrent worker processes competing for a 20-unit manifest over a shared directory.
  - Randomly inject SIGKILL / process termination mid-unit to simulate node preemption.
  - Verify stale lease stealing past 300s clock drift margin, heartbeat CAS abort, singleton commit token exclusion, and zero corrupt output stores.
  - *Verification Gate*: Zero tensor corruption, zero deadlocks, 100% verified completion.

---

## 4. Epistemic Rigor & Verification Gates

Each track must pass its designated synthetic positive and negative controls prior to merging:

| Gate ID | Target Track | Control Type | Verification Requirement |
| :--- | :--- | :--- | :--- |
| **GATE-01** | Track 1 | Positive Control | Byte-for-byte identical `manifest.json` on successive planning runs with 0 GPU allocations. |
| **GATE-02** | Track 1 | Negative Control | `SpecPartitionError` raised when callable or lambda is injected into `spec.parameters`. |
| **GATE-03** | Track 2 | Negative Control | Simultaneous commit attempts fail with `FileExistsError` on singleton `<output>.commit`. |
| **GATE-04** | Track 2 | Positive Control | Tier-1 Fast Resume checks done marker and Zarr metadata in $< 5$ ms. |
| **GATE-05** | Track 2 | Negative Control | Tampered chunk byte in Zarr store triggers relocation to `.quarantine/` under Tier-2 audit. |
| **GATE-06** | Track 2 | Edge Case Control | `quarantine_output` renames store and marker in-place to `.corrupt.<uuid>` on simulated `ENOSPC`. |
| **GATE-07** | Track 3 | Positive Control | `recycle_interval=50` keeps process RSS bounded within $\pm 2\%$ across 50 units. |
| **GATE-08** | Track 3 | Positive Control | `SlurmArrayExecutor` emits sparse range `#SBATCH --array=12,15,44-50%20` on resume. |
| **GATE-09** | Track 3 | Boundary Control | `SlurmArrayExecutor` returns early with completion message when `pending_indices == []`. |
| **GATE-10** | Track 4 | Deprecation Gate | Legacy `xtrax sweep` emits `FutureWarning` and completes execution via `LocalSequentialExecutor`. |
| **GATE-11** | Track 5 | Parity Gate | `test_aminx_parity.py` matches golden vectors bit-for-bit on `input_hash`. |

---
*Implementation Epic authored and approved by `teamwork_worker_p4_editor` for P5 implementation dispatch.*
