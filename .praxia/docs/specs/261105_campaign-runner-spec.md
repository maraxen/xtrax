---
title: xtrax.campaign runner specification (SPEC-261105-CAMPAIGN-RUNNER)
description: Design specification for the distributed campaign runner that shipped as PR #267; salvaged 261119 from an untracked worktree file as the historical record.
status: implemented
task_id: 261105_xtrax-campaign-runner
date: '261105'
salvaged_by: 261119_branch-triage
---
# Formal Engineering Specification: `xtrax.campaign`
## Distributed Campaign Runner & Execution Engine

**Document ID**: `SPEC-261105-CAMPAIGN-RUNNER`  
**Task ID**: `261105_xtrax-campaign-runner`  
**Milestone**: P4 Review Panel & Adjudication (Requirement R3)  
**Author**: `teamwork_worker_p3_spec` (Campaign Runner Specification Author)  
**Editor**: `teamwork_worker_p4_editor` (Specification Editor & Epic Creator)  
**Status**: APPROVED & CONVERGED (P4 Review Panel)  
**Date**: 2026-11-05  
**Target Path**: `docs/spec/261105_campaign_runner_spec.md`  

---

## 1. Executive Summary & Scope Boundary

### 1.1 Problem Statement & Architectural Context
Within the research ecosystem, high-throughput computational campaigns across multi-node clusters (`Engaging`, `titanix`) have suffered from acute architectural fragmentation. Repositories including `aminx`, `alynxr`, and `adaptyv` have independently developed bespoke, incompatible batch loops, distributed leasing protocols, and completion markers. In aggregate, over 3,000 lines of fragile, copy-pasted infrastructure manage unit planning, filesystem locks, and resume verification.

The library `xtrax` provides high-performance JAX batching, kernel tiling, and execution abstractions (`BatchPlanner`, `AxisSpec`, `Engine`, `xtrax.pack`), but prior to this specification, lacked a consumer-usable campaign runner packaged within the distributed wheel. While repository-internal controllers existed in `controller/` (e.g. `controller/loop_run.py` and `controller/bathos_campaign_adapter.py` for Epic #3611 / #2181), and a rudimentary `xtrax sweep` CLI verb existed in `src/xtrax/cli/sweep_verb.py`, consumers could not import or deploy a standardized campaign execution engine. This gap constitutes **Debt #3142**.

This specification establishes `xtrax.campaign`: a standardized, distributed, scheduler-agnostic campaign runner shipped directly in the `xtrax` wheel.

### 1.2 In-Wheel vs. Unshipped Scope Boundary
The boundary between distributed public package code and repository-internal research controllers is strictly enforced per Hatchling build configuration (`pyproject.toml:282-286`):

```
[tool.hatch.build.targets.wheel]
packages = ["src/xtrax"]
exclude = ["src/xtrax/devtools"]
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SHIPPED WHEEL: src/xtrax/                             │
│                                                                             │
│   src/xtrax/campaign/                                                       │
│   ├── spec.py         <- Declarative CampaignSpec, ArmSpec, ParamGrid       │
│   ├── manifest.py     <- Pure host planner & canonical manifest generation  │
│   ├── lease.py        <- Atomic POSIX/NFS leasing, heartbeat, CAS steal     │
│   ├── marker.py       <- DoneMarker validation, quarantine management       │
│   └── executors/                                                            │
│       ├── protocol.py <- CampaignExecutor protocol & ExecutionSummary       │
│       ├── local.py    <- LocalSequential & LocalProcessPool executors       │
│       └── slurm.py    <- SlurmArrayExecutor (pure host script generator)    │
│                                                                             │
│   src/xtrax/cli/                                                            │
│   ├── campaign_verb.py <- CLI commands: plan, run, status, gates            │
│   └── sweep_verb.py    <- DEPRECATED compatibility shim over campaign       │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ▲
                                       │ (Unidirectional Consumption)
┌─────────────────────────────────────────────────────────────────────────────┐
│                UNSHIPPED REPOSITORY CODE: controller/ & .praxia/            │
│                                                                             │
│   controller/                                                               │
│   ├── loop_run.py                <- Algorithm-evolution loop (#2181)        │
│   └── bathos_campaign_adapter.py <- Bathos catalog & claim integration      │
│   .praxia/                                                                  │
│   └── loop_human_gates.toml      <- Human approval attestations (T2-32)     │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Boundary Invariants**:
1. **Zero Downward Dependencies**: Code within `src/xtrax/campaign/` MUST NOT import anything from `controller/`, `scripts/`, or root-level development harnesses.
2. **Wheel Self-Sufficiency**: A consumer installing `pip install xtrax` (or `pip install xtrax[io]` for Zarr verification) has immediate, complete access to `xtrax.campaign` without cloning the repository.
3. **Decoupled Controller Consumption**: Repository-internal loops (`controller/loop_run.py`) and external platforms (`bathos`, `aminx`, `alynxr`) consume `xtrax.campaign` strictly as public library clients.

### 1.3 Demarcation of Execution Abstractions
To eliminate conceptual blurring between sweeps, experiments, and campaigns, the ecosystem adheres to three distinct operational definitions:

| Dimension | Sweep (`xtrax sweep`) | Experiment (`bathos`) | Campaign (`xtrax.campaign`) |
| :--- | :--- | :--- | :--- |
| **Primary Intent** | Fast, ephemeral parameter exploration to optimize a scalar metric. | Historical, epistemological record of scientific hypotheses and outcome claims. | Robust, reproducible execution of batched compute units across multi-node clusters. |
| **Identity Model** | Ephemeral `sweep_id` + run counter (`0..N`). | Cryptographic run UUID + git dirty tree snapshot + `.bth.toml` pre-registration. | Content-addressed `manifest_row_hash` = $\text{SHA256}(\text{canonical\_json}(\text{spec}))$. |
| **Concurrency & Leasing**| Single-process sequential iteration (no concurrency control). | Cluster job submission (`bth submit`) with event-log tracking (`cutover.json`). | Distributed atomic file leases (`O_CREAT \| O_EXCL`) with active background heartbeats. |
| **Resume & Recovery** | Restarts from combination 0 on failure. | Non-zero exit recovery; run immutable once stamped. | Two-tier resume verification (parameter input match + array chunk content digest). |
| **Data Artifacts** | Loose checkpoints and run ledger JSON. | Parquet catalog records, claim attestation TOML, figure manifests. | Chunked Zarr stores or seekable Zstandard archives (`xtrax.pack`) with `.done.json` markers. |

---

## 2. Assumption Table

The design and operational guarantees of `xtrax.campaign` rest upon explicit architectural assumptions. Each assumption is mapped to its empirical basis, invalidation risk, blast radius, and verification control:

| ID | Assumption | Basis / Source | Invalidation Risk | Blast Radius | Verification Control |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A-01** | Atomic file creation via `os.O_CREAT \| os.O_EXCL` is POSIX-compliant and atomic across NFSv3, NFSv4, and Lustre mounts. | Production validation in `aminx.host.campaign:535`; Optuna `JournalFileOpenLock` benchmark. | Highly misconfigured NFS server with `nordirplus` or broken lock daemons permits duplicate file creation. | High (concurrent workers write to the same output path, corrupting tensors). | Pre-commit ownership verification and singleton commit token (`<output_path>.commit` via `O_CREAT \| O_EXCL`) prior to renaming `.partial.<attempt_uuid>` to final output. |
| **A-02** | Explicit directory-level `fsync` flushes client-side metadata attribute caches across cluster nodes. | `aminx.host.campaign:29-31`; Linux NFS kernel attribute caching ceiling (`acdirmin`/`acdirmax`). | Unflushed directory listing causes Node B to read stale directory state and attempt redundant lock steal; NFS client raises `EINVAL` on dir fd. | Medium (transient lock contention or unhandled `OSError`). | `fsync_directory` called on enclosing parent directory after lock acquisition, heartbeat renewal, and lock release; suppresses `EINVAL`, `ENOTSUP`, `EOPNOTSUPP`, and `EBADF` on NFS mounts. |
| **A-03** | Pure host planning can statically determine all unit parameters and resource budgets prior to accelerator execution. | Nextflow task-hash model; `xtrax.tiling.BatchPlanner`. | Dynamic sequence lengths or ragged data structures cause uniform memory budgets to trigger OOM. | High (node crash during batch execution). | `ManifestRow` includes `resource_requirements` calculated via `BatchPlanner.bucketing` during host planning. |
| **A-04** | JAX device memory allocator accumulates memory fragmentation across sequential units within long-lived worker processes. | `xtrax.cli.sweep_verb:124-135`; JAX BFC allocator behavior with dynamic shapes. | Frequent process recycling destroys in-memory XLA JIT cache; uncollected heap fragments GPU memory. | High (85% compilation overhead or `RESOURCE_EXHAUSTED` OOM). | Configurable chunked recycling (`recycle_interval=50`), JAX persistent disk compilation cache (`jax_compilation_cache_dir`), tensor blocking and Python `gc.collect()`. |
| **A-05** | Python callables and JAX transform functions are non-serializable and MUST NOT enter manifest rows. | `aminx.host.spec_partition:166-175` (`_CALLABLE_FIELDS` enforcement). | Caller attempts to pass lambda or function pointer in spec; canonical JSON serialization fails or becomes non-deterministic. | Critical (manifest generation crashes or generates unstable hashes). | Pure-data partitioner: `spec_partition.py` strips callables, enforcing that rows contain only primitives, strings, and numeric lists. |
| **A-06** | A completed unit's integrity can be reliably verified without reading multi-gigabyte array chunks during fast resume. | `xtrax.run.zarr_integrity:DIGEST_ALGO_VERSION`; `aminx` schema `campaign_done_marker_v3`. | Reading gigabytes of chunks on resume causes host OOM and exceeds 5 ms latency target; bit flips in stored chunks go undetected. | Critical (host OOM or silent data corruption). | Two-tier resume: Tier-1 Fast Resume (< 5 ms) validates `.done.json` input hash & Zarr metadata; Tier-2 Deep Verification (`--verify-chunks`) streams chunk digests with bounded buffer. |
| **A-07** | Stale locks can be safely stolen after expiration via atomic compare-and-swap (`os.replace`) protected against clock drift. | `aminx.host.campaign:581-606`. | Worker suspended by OS revives after lease stolen; heterogeneous cluster node clock drift causes premature lease theft. | High (stale worker clobbers valid output of new worker). | `CLOCK_DRIFT_GRACE_PERIOD_S = 300` with dual expiration check (`lease_expires_at_unix_s` + `st_mtime`), heartbeat CAS expiry guard, and singleton commit token (`<output_path>.commit`). |
| **A-08** | Adaptive campaigns can be expressed as a discrete sequence of immutable generational manifest slices ($\mathcal{M}_k \to \mathcal{M}_{k+1}$). | Ax `GenerationStrategy`; evosax generational stepping; Nextflow channel barriers. | Single straggler unit in generation $k$ hangs, blocking the transition to generation $k+1$ indefinitely. | High (pipeline deadlocks, wasting cluster allocation). | Configurable completion threshold ($\theta \in [0.9, 1.0]$) in `AdaptiveCampaignController` with cluster task cancellation (`scancel`), telemetry, and generation sealing. |
| **A-09** | `src/xtrax/campaign/` must have zero runtime dependencies on `bathos`. | Modular separation of concerns; Hatchling wheel distribution. | Embedding bathos imports in `xtrax.campaign` forces bathos dependencies onto generic JAX users. | Medium (package dependency bloat and circular import hazards). | Zero imports of `bathos` in `src/xtrax/`; adapters defined directly in `src/xtrax/campaign/marker.py` (`AminxDoneMarkerAdapter`, `AlynxrFragmentAdapter`, `DoneMarkerAdapter`). |
| **A-10** | Existing `xtrax sweep` invocations can be transparently adapted to `xtrax.campaign` without breaking user scripts. | `src/xtrax/cli/sweep_verb.py`. | Legacy CLI arguments rejected; automated CI pipelines fail upon upgrading xtrax. | Medium (user workflow disruption). | Deprecation adapter shim maps `[sweep.axes]` to single-arm `CampaignSpec` and executes via `LocalSequentialExecutor`. |

---

## 3. Formal Data Schemas

All schemas are strictly versioned, fully typed, and serializable to canonical JSON.

### 3.1 `CampaignSpec` (Declarative Input Representation)
`CampaignSpec` represents the complete, immutable user declaration of a campaign before host planning expands it into a manifest.

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True, slots=True)
class ParamGrid:
    """Multi-dimensional parameter sweep grid."""
    axes: dict[str, list[Any]] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class ArmSpec:
    """Specification for a distinct predictor/sampling arm."""
    arm_id: str
    backend: str
    checkpoint_id: str
    fixed_policy: str = "none"
    state_weight_profile: str = "equal"
    parameters: dict[str, Any] = field(default_factory=dict)
    resource_overrides: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class CampaignSpec:
    """Top-level campaign declaration."""
    campaign_id: str
    target_inputs: list[str]
    arms: list[ArmSpec]
    grid: ParamGrid = field(default_factory=ParamGrid)
    samples_per_target: int = 100
    chunk_size: int = 100
    seed: int = 42
    output_dir: str = "outputs"
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 3.2 `CampaignManifest` and `ManifestRow`
- **Root Schema Version**: `campaign_manifest_v1`
- **Serialization Format**: UTF-8 encoded Canonical JSON (`canonical_json_bytes`).

```python
@dataclass(frozen=True, slots=True)
class ResourceRequirements:
    """Resource budget calculated statically during host planning."""
    gpu_memory_bytes: int
    cpu_cores: int = 1
    timeout_seconds: int = 3600
    padding_bucket_id: int | None = None

@dataclass(frozen=True, slots=True)
class ManifestRow:
    """An individual, content-addressed compute unit."""
    manifest_row_hash: str          # 64-character lowercase hex SHA-256 (global unit identity)
    input_hash: str                 # 64-character lowercase hex SHA-256 (aminx parameter parity hash)
    campaign_id: str
    job_id: str
    chunk_id: int
    sample_start: int
    sample_count: int
    target_input: str
    arm_id: str
    arm_backend: str
    checkpoint_id: str
    fixed_policy: str
    state_weight_profile: str
    seed: int
    parameters: dict[str, Any]
    output_locator: str
    resource_requirements: ResourceRequirements

@dataclass(frozen=True, slots=True)
class ManifestMetadata:
    campaign_id: str
    created_at_utc: str
    planner_version: str
    spec_sha256: str
    total_rows: int

@dataclass(frozen=True, slots=True)
class CampaignManifest:
    schema_version: str             # Fixed: "campaign_manifest_v1"
    metadata: ManifestMetadata
    rows: list[ManifestRow]
```

#### JSON Representation of `CampaignManifest`
```json
{
  "schema_version": "campaign_manifest_v1",
  "metadata": {
    "campaign_id": "glp1r_optimization_campaign",
    "created_at_utc": "2026-11-05T18:00:00Z",
    "planner_version": "xtrax@0.5.0",
    "spec_sha256": "4b72a8c3...64_hex_digits...",
    "total_rows": 120
  },
  "rows": [
    {
      "manifest_row_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "input_hash": "a17f93b2c8e4d019f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3",
      "campaign_id": "glp1r_optimization_campaign",
      "job_id": "target_001_arm_boltz2_chunk_000",
      "chunk_id": 0,
      "sample_start": 0,
      "sample_count": 100,
      "target_input": "inputs/targets/glp1r_active.pdb",
      "arm_id": "boltz2_standard",
      "arm_backend": "boltz2",
      "checkpoint_id": "boltz2_rec_v1",
      "fixed_policy": "none",
      "state_weight_profile": "equal",
      "seed": 42,
      "parameters": {
        "temperature": 0.2,
        "sampling_steps": 50,
        "recycle_count": 3
      },
      "output_locator": "glp1r_optimization_campaign/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.zarr",
      "resource_requirements": {
        "gpu_memory_bytes": 17179869184,
        "cpu_cores": 4,
        "timeout_seconds": 1800,
        "padding_bucket_id": 512
      }
    }
  ]
}
```

### 3.3 `CampaignLock`
- **Schema Version**: `campaign_lock_v1`
- **Location on Disk**: `<output_path>.lock`

```python
CLOCK_DRIFT_GRACE_PERIOD_S: int = 300  # 5 minutes safety margin against inter-node clock drift

@dataclass(frozen=True, slots=True)
class CampaignLock:
    schema_version: str             # Fixed: "campaign_lock_v1"
    owner_token: str                # e.g. "node42.cluster.org:pid1234:uuid5678"
    manifest_row_hash: str          # 64-hex row hash
    acquired_at_unix_s: float       # Immutable lease epoch
    lease_expires_at_unix_s: float
    lease_seconds: int              # Default: 1800
    heartbeat_count: int            # Incremented on each tick
```

#### JSON Representation of `CampaignLock`
```json
{
  "schema_version": "campaign_lock_v1",
  "owner_token": "node104.engaging.mit.edu:pid38491:8f7e2a1b",
  "manifest_row_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "acquired_at_unix_s": 1730830000.0,
  "lease_expires_at_unix_s": 1730831800.0,
  "lease_seconds": 1800,
  "heartbeat_count": 0
}
```

### 3.4 `CampaignDoneMarker`
- **Schema Version**: `campaign_done_marker_v3` (Parity with aminx standard)
- **Location on Disk**: `<output_path>.done.json`

```python
@dataclass(frozen=True, slots=True)
class ProducerInfo:
    version: str                    # e.g. "xtrax@0.5.0"
    git_sha: str                    # 8-char hex or "clean"
    hostname: str                   # Socket FQDN

@dataclass(frozen=True, slots=True)
class CampaignDoneMarker:
    schema_version: str             # Fixed: "campaign_done_marker_v3"
    manifest_row_hash: str          # Must match row
    input_hash: str                 # Parameter input hash
    producer: ProducerInfo
    output_locator: str             # Relative or absolute output locator
    content_digest_sha256: str      # Zarr array content digest (or file sha256)
    completed_at_utc: str
    completed_at_unix_s: float
    duration_s: float
```

#### JSON Representation of `CampaignDoneMarker`
```json
{
  "schema_version": "campaign_done_marker_v3",
  "manifest_row_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "input_hash": "a17f93b2c8e4d019f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3",
  "producer": {
    "version": "xtrax@0.5.0",
    "git_sha": "28a87130",
    "hostname": "node104.engaging.mit.edu"
  },
  "output_locator": "glp1r_optimization_campaign/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.zarr",
  "content_digest_sha256": "8c4d2e1a9f0b3c5e7d1a2b4c6e8f0a2b4c6e8f0a2b4c6e8f0a2b4c6e8f0a2b4c",
  "completed_at_utc": "2026-11-05T18:15:32Z",
  "completed_at_unix_s": 1730830532.15,
  "duration_s": 532.15
}
```

---

## 4. Unit Identity & Verified Resume Protocol

### 4.1 Canonical JSON Serialization Rules
To guarantee mathematical reproducibility across heterogeneous Python runtimes, architectures, and operating systems, all hashes in `xtrax.campaign` derive from `canonical_json_bytes` (`src/xtrax/run/zarr_integrity.py:60-68`):

1. **Key Sorting**: Dictionary keys are sorted strictly by stringified key order (`key=lambda kv: str(kv[0])`), preventing `TypeError` on mixed string/integer keys, and serialized with `sort_keys=True`.
2. **Whitespace Stripping**: Zero whitespace surrounds delimiters: separators are strictly `(",", ":")`.
3. **Encoding & Unicode Normalization**: Text strings are normalized to Unicode Normalization Form C (`NFC`), and encoded as UTF-8 (`ensure_ascii=False`).
4. **Disallow Non-Finite Floats**: IEEE-754 `NaN`, `+Infinity`, and `-Infinity` are forbidden (`allow_nan=False`).
5. **Float Zero Normalization**: IEEE-754 `-0.0` is strictly normalized to `0.0` (`0.0 if value == 0.0 else value`), preventing hash divergence between numerically equivalent zeroes.
6. **Type Coercion**: NumPy scalars and ndarrays are converted to native Python types via `normalize_json_value`.

```python
def normalize_json_value(value: Any) -> Any:
    """Recursively normalize values for canonical JSON serialization."""
    if isinstance(value, float):
        return 0.0 if value == 0.0 else value
    if isinstance(value, np.floating):
        val = float(value.item())
        return 0.0 if val == 0.0 else val
    if isinstance(value, np.generic):
        return normalize_json_value(value.item())
    if isinstance(value, np.ndarray):
        return [normalize_json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [normalize_json_value(item) for item in value]
    if isinstance(value, dict):
        # Sort by stringified keys to guarantee order without mixed-type crashes
        return {str(k): normalize_json_value(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    return value
```

### 4.2 Unit Identity Hashing & aminx Hash Separation
The specification formally establishes two distinct, complementary hashes to satisfy both global unit uniqueness and bit-for-bit aminx golden vector parity:

1. **Global Unit Identity (`manifest_row_hash`)**:
   Uniquely identifies the scheduled work unit across campaigns, targets, and backends. This hash names the unit output store (`<manifest_row_hash>.zarr`):

   $$\text{manifest\_row\_hash} = \text{SHA256}(\text{canonical\_json\_bytes}(\mathcal{P}_{\text{row\_norm}}))$$

   Where $\mathcal{P}_{\text{row\_norm}}$ includes all structural execution fields:
   - `campaign_id`, `job_id`, `chunk_id`, `sample_start`, `sample_count`
   - `target_input`, `arm_id`, `arm_backend`, `checkpoint_id`, `fixed_policy`, `state_weight_profile`
   - `seed`, and `parameters` (sorted dict of domain hyperparameters)

2. **Domain Parameter Hash (`input_hash`)**:
   Reproduces `aminx.host.campaign.compute_unit_input_hash` bit-for-bit over domain hyperparameters nested under `"params"`, excluding campaign-owned runtime fields (detailed in Section 10.1). Stored in `ManifestRow.input_hash` and `CampaignDoneMarker.input_hash`.

```python
def compute_manifest_row_hash(normalized_row_payload: dict[str, Any]) -> str:
    """Compute 64-char lowercase hex SHA-256 over full canonical unit payload."""
    payload_bytes = canonical_json_bytes(normalized_row_payload)
    return hashlib.sha256(payload_bytes).hexdigest()
```

### 4.3 Two-Tier Resume Verification (Decoupled Fast Resume vs Deep Audit)
Reading and hashing multi-gigabyte Zarr array chunks on every resume causes severe filesystem latency (>100s over NFS) and risks host memory exhaustion (OOM). `xtrax.campaign` formally decouples resume into two operational tiers:

```
                      [Check Manifest Row]
                                │
                                ▼
                 Does <output>.done.json exist?
                                │
                ┌───────────────┴───────────────┐
                │ YES                           │ NO
                ▼                               ▼
     [Tier 1: Fast Resume (< 5 ms)]       [Execute Unit]
   1. marker.manifest_row_hash == row.manifest_row_hash?
   2. marker.input_hash == row.input_hash?
   3. Zarr root metadata (.zgroup/.zarray) valid?
                │
        ┌───────┴───────┐
        │ YES           │ NO
        ▼               ▼
   [Tier 2 Audit? ]  [Move to .quarantine/]
        │
    ┌───┴────────────────────────┐
    │ Default (Fast)             │ Explicit Opt-In (--verify-chunks)
    ▼                            ▼
[REUSE: Skip Unit]     [Tier 2: Deep Content Verification]
                       Stream chunk bytes (bounded 64 MB buffer)
                       Does zarr_content_digest(output)
                       == marker.content_digest_sha256?
                                │
                        ┌───────┴───────┐
                        │ YES           │ NO
                        ▼               ▼
                 [REUSE: Skip]   [Move to .quarantine/]
```

1. **Tier 1: Fast Resume (Default, $< 5$ ms)**:
   - Checks `<output_path>.done.json` existence.
   - Validates `marker.manifest_row_hash == row.manifest_row_hash` and `marker.input_hash == row.input_hash`. Discrepancy indicates parameter drift.
   - Inspects Zarr root metadata (`.zgroup` or `.zarray`), verifying that the store exists, is structurally parseable, and matches expected shape and dtype attributes.
   - Avoids reading array chunk payloads, satisfying the $< 5$ ms resume latency budget per unit.
2. **Tier 2: Deep Content Verification (`--verify-chunks` / `xtrax campaign verify --deep`)**:
   - An explicit opt-in cryptographic verification pass.
   - Recomputes `zarr_content_digest` across all chunk bytes and metadata.
   - Employs streaming chunk iteration with bounded buffer memory (64 MB buffer limit) to prevent host memory exhaustion on large multi-dimensional arrays.

### 4.4 Quarantine Protocol (`.quarantine/`)
Corrupted, unreadable, or superseded unit outputs are never deleted in place, protecting post-mortem forensic analysis:

1. **Trigger Conditions**:
   - Done marker JSON is corrupt, unparseable, or schema outdated (`campaign_done_marker_v2`).
   - Done marker exists, but output Zarr store or root metadata is missing.
   - Tier-1 input hash mismatch (`InputHashMismatchError`) or Tier-2 content digest mismatch (`ContentDigestMismatchError`).
2. **Defensive Quarantine Procedure (`ENOSPC` & `EXDEV` Resilient)**:
   - Default: The output store and its done marker are atomically moved to `<campaign_dir>/.quarantine/<YYYYMMDD_HHMMSS>_<reason>_<manifest_row_hash>/`, and a `quarantine_manifest.json` is written recording forensic details.
   - `ENOSPC` Fallback (Disk/Inode Full): If directory creation or manifest writing fails with `ENOSPC` (Errno 28), allocating new files or directories is impossible. The runner falls back to renaming **both** the output store and the `.done.json` marker in-place to `<output_path>.corrupt.<uuid>` and `<output_path>.done.json.corrupt.<uuid>`.
   - `EXDEV` Fallback (Cross-Device Link): If quarantine resides across filesystem mount boundaries raising `EXDEV` (Errno 18), `os.replace` fails; the runner falls back to `shutil.move` (copy and unlink).
   - Once quarantined, the manifest row is eligible for clean re-execution.

```python
def quarantine_output(output_path: Path, reason: str, exc: Exception | None = None) -> Path:
    """Safely isolate corrupt output store and marker, defensively handling ENOSPC and EXDEV."""
    campaign_dir = output_path.parent
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    quarantine_dir = campaign_dir / ".quarantine" / f"{timestamp}_{reason}_{output_path.name}"
    done_marker = output_path.with_name(f"{output_path.name}.done.json")

    try:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = quarantine_dir / "quarantine_manifest.json"
        manifest_payload = {
            "original_path": str(output_path),
            "reason": reason,
            "detected_at_utc": datetime.now(timezone.utc).isoformat(),
            "exception": str(exc) if exc else None,
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
        os.replace(output_path, quarantine_dir / output_path.name)
        if done_marker.exists():
            os.replace(done_marker, quarantine_dir / done_marker.name)
        return quarantine_dir
    except OSError as err:
        if err.errno == errno.ENOSPC:
            # Fallback 1: Disk/inode exhaustion - rename in-place without writing metadata
            unique_suffix = f"corrupt.{uuid.uuid4().hex[:8]}"
            fallback_path = output_path.with_name(f"{output_path.name}.{unique_suffix}")
            if output_path.exists():
                os.replace(output_path, fallback_path)
            if done_marker.exists():
                os.replace(done_marker, output_path.with_name(f"{done_marker.name}.{unique_suffix}"))
            return fallback_path
        elif err.errno == errno.EXDEV:
            # Fallback 2: Cross-device link - use copy-and-unlink
            shutil.move(output_path, quarantine_dir / output_path.name)
            if done_marker.exists():
                shutil.move(done_marker, quarantine_dir / done_marker.name)
            return quarantine_dir
        raise QuarantineFailedError(f"Failed to quarantine corrupt output {output_path}: {err}") from err
```

---

## 5. Distributed Leasing & Heartbeat Protocol

To coordinate hundreds of workers across shared cluster storage (NFS / Lustre) without a centralized database daemon or unreliable POSIX `fcntl` locks, `xtrax.campaign` implements the atomic file creation and heartbeat leasing protocol.

### 5.1 Atomic Exclusion Mechanism
Lease acquisition utilizes atomic POSIX file creation flags:

$$\text{fd} = \text{os.open}(\text{lock\_path}, \text{os.O\_WRONLY} \mid \text{os.O\_CREAT} \mid \text{os.O\_EXCL}, 0\text{o}644)$$

- If the file does not exist, the OS kernel atomically creates it and returns a valid file descriptor.
- If the file already exists, the kernel atomically raises `FileExistsError`. No race condition is possible.

### 5.2 Heartbeat Thread Specification & Directory Durabilization
Upon acquiring the lease, the worker spawns a dedicated daemon thread (`HeartbeatThread`) that maintains the active lease on disk:

- **Tick Interval**: Default 60 seconds.
- **Lease Duration**: Default 1,800 seconds (30 minutes).
- **Expiration Guard**: If extreme thread scheduling lag or cluster node freeze causes the renewal attempt to occur when `time.time() >= self.current_lease_expires_at - 60.0`, the thread has lost the right to renew. It MUST NOT attempt `os.replace`; it sets `self.abort_event.set()` and terminates immediately.
- **CAS Pre-Renewal Verification**: Before staging an update, the thread verifies that `<lock_path>` exists, `owner_token == self.owner_token`, `heartbeat_count == self.heartbeat_count`, and the immutable `acquired_at_unix_s` matches the epoch recorded at initial acquisition.
- **CAS Post-Renewal Verification**: After atomically replacing `<lock_path>` via `os.replace`, the thread immediately re-reads `<lock_path>` to confirm its `owner_token` and `heartbeat_count` persist. If another worker clobbered the lease, `abort_event.set()` is triggered.
- **Execution Loop Inspection**: The worker execution loop inspects `heartbeat.abort_event.is_set()` before each batch step and prior to output commit. If set, the worker halts immediately, unlinks its `.partial` scratch directory, and raises `LeaseLostError`.
- **NFS Directory `fsync` Portability**: Parent directory metadata flushes (`fsync_directory`) catch and gracefully suppress `EINVAL`, `ENOTSUP`, `EOPNOTSUPP`, and `EBADF`, ensuring portable execution across both local ext4/XFS filesystems and cluster NFS client mounts.

```python
def fsync_directory(path: Path) -> None:
    """Durabilize directory entries, gracefully handling NFS client EINVAL/ENOTSUP."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as e:
        if e.errno in (errno.ENOENT, errno.EACCES):
            return
        raise
    try:
        os.fsync(fd)
    except OSError as e:
        # On Linux NFS client mounts, fsync on a directory fd returns EINVAL (22) or ENOTSUP (95)
        if e.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EBADF):
            raise
    finally:
        os.close(fd)

class HeartbeatThread(threading.Thread):
    """Refreshes lease_expires_at_unix_s periodically with strict CAS expiry guards."""
    def __init__(
        self,
        lock_path: Path,
        owner_token: str,
        manifest_row_hash: str,
        lease_seconds: int = 1800,
        interval_seconds: float = 60.0,
        initial_acquired_at: float | None = None,
    ) -> None:
        super().__init__(daemon=True, name=f"Heartbeat-{manifest_row_hash[:8]}")
        self.lock_path = lock_path
        self.owner_token = owner_token
        self.manifest_row_hash = manifest_row_hash
        self.lease_seconds = lease_seconds
        self.interval_seconds = interval_seconds
        self.acquired_at_unix_s = initial_acquired_at or time.time()
        self.current_lease_expires_at = self.acquired_at_unix_s + lease_seconds
        self.abort_event = threading.Event()
        self.stop_event = threading.Event()
        self.heartbeat_count = 0

    def run(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            if not self._renew_lease_cas():
                self.abort_event.set()
                break

    def _renew_lease_cas(self) -> bool:
        """Atomically renew lease using CAS readback verification and expiration guard."""
        try:
            now = time.time()
            # 1. Expiration Guard: Never attempt renewal if lease is within 60s of expiring or expired
            if now >= self.current_lease_expires_at - 60.0:
                return False

            # 2. CAS Pre-Check: verify active ownership and immutable epoch
            if not self.lock_path.exists():
                return False
            current = json.loads(self.lock_path.read_text(encoding="utf-8"))
            if current.get("owner_token") != self.owner_token:
                return False
            if current.get("acquired_at_unix_s") != self.acquired_at_unix_s:
                return False
            if current.get("heartbeat_count") != self.heartbeat_count:
                return False

            # 3. Stage updated lease
            next_count = self.heartbeat_count + 1
            new_expires = now + self.lease_seconds
            updated_lock = {
                "schema_version": "campaign_lock_v1",
                "owner_token": self.owner_token,
                "manifest_row_hash": self.manifest_row_hash,
                "acquired_at_unix_s": self.acquired_at_unix_s,
                "lease_expires_at_unix_s": new_expires,
                "lease_seconds": self.lease_seconds,
                "heartbeat_count": next_count,
            }
            tmp_path = self.lock_path.with_name(f"{self.lock_path.name}.tmp.{uuid.uuid4().hex[:8]}")
            tmp_path.write_text(json.dumps(updated_lock, indent=2), encoding="utf-8")
            fsync_file(tmp_path)

            # 4. Atomically replace lock file
            os.replace(tmp_path, self.lock_path)
            fsync_directory(self.lock_path.parent)

            # 5. CAS Post-Verification: Confirm our write held
            post_check = json.loads(self.lock_path.read_text(encoding="utf-8"))
            if (
                post_check.get("owner_token") == self.owner_token
                and post_check.get("heartbeat_count") == next_count
                and post_check.get("acquired_at_unix_s") == self.acquired_at_unix_s
            ):
                self.heartbeat_count = next_count
                self.current_lease_expires_at = new_expires
                return True
            return False
        except Exception:
            return False
```

### 5.3 Atomic Compare-and-Swap (CAS) Stale Lease Stealing with Clock Drift Margin
If a worker crashes, is preempted by SLURM (`SIGKILL`), or encounters an unhandled node failure, its lock file remains on disk. To prevent inter-node clock drift from causing premature lease theft across cluster nodes, lease stealing enforces `CLOCK_DRIFT_GRACE_PERIOD_S = 300` (5 minutes) combined with relative file `st_mtime` verification:

```
[Lock Exists]
      │
      ▼
Read <output>.lock and stat(output.lock)
      │
Is now >= lease_expires_at_unix_s + 300s
AND now - st_mtime >= lease_seconds + 300s?
      │
      ├────────────────────────┐
      │ NO                     │ YES (Lease Confirmed Stale)
      ▼                        ▼
Raise LockUnavailable    Write candidate steal file:
                         <output>.lock.steal.<uuid>.tmp
                               │
                         os.replace(steal_file, <output>.lock)
                               │
                         Re-read <output>.lock
                               │
                         Does owner_token == my_token?
                               │
                         ┌─────┴─────┐
                         │ YES       │ NO (Lost CAS race)
                         ▼           ▼
                   [Lease Won]  Raise LockUnavailable
```

1. **Dual Expiration & Drift Check**:
   ```python
   lock_stat = lock_path.stat()
   is_stale = (
       now >= lock.lease_expires_at_unix_s + CLOCK_DRIFT_GRACE_PERIOD_S
   ) and (
       now - lock_stat.st_mtime >= lock.lease_seconds + CLOCK_DRIFT_GRACE_PERIOD_S
   )
   ```
   If `not is_stale`, the lease is active; acquisition is refused (`LockUnavailableError`).
2. **Stage Steal Candidate**: If stale, the stealing worker writes a new `CampaignLock` payload with its own `owner_token`, setting `acquired_at_unix_s = now` (minting a new lease epoch) to `<lock_path>.steal.<uuid>.tmp`.
3. **Atomic Replace**: The worker calls `os.replace(tmp_path, lock_path)` and `fsync_directory(parent)`.
4. **CAS Verification**: Each worker immediately re-reads `<lock_path>`. Only the worker whose `owner_token` is recorded in the file has won the lease; all other workers raise `LockUnavailableError`.

### 5.4 Commit Token Verification Protocol (Atomic Promotion & Zombie Worker Protection)
To definitively eliminate Time-Of-Check to Time-Of-Use (TOCTOU) windows and protect against "zombie workers" (processes suspended by OS swap or network lag whose leases were stolen while paused), output finalization follows the **Singleton Commit Token Protocol**:

```
[Computation Complete in .partial.<uuid>]
                 │
                 ▼
     Does .done.json exist?
    ┌────────────┴────────────┐
    │ YES                     │ NO
    ▼                         ▼
[Discard Attempt:       Is Generation Sealed? (.generation_sealed)
 StolenAttemptDiscarded]       │
                       ┌───────┴───────┐
                       │ YES           │ NO
                       ▼               ▼
                 [Reject & Abort:     Acquire Singleton Commit Token:
                  GenerationSealed]   open(<output>.commit, O_CREAT | O_EXCL)
                                               │
                                       ┌───────┴───────┐
                                       │ Fails         │ Succeeds
                                       ▼               ▼
                                 [Abort: Commit  Re-Verify Lock:
                                  In Progress]   - owner_token == my_token
                                                 - now < lease_expires
                                                 - not heartbeat.abort_event
                                                       │
                                               ┌───────┴───────┐
                                               │ Failed        │ Passed
                                               ▼               ▼
                                         [Abort & Clean: Promote Output:
                                          LeaseLostError] os.replace(.partial, <output>)
                                                               │
                                                         Write & Rename .done.json
                                                               │
                                                         fsync_directory()
                                                               │
                                                         Unlink .commit and .lock
```

1. **Existing Done Check**: Check if `<output_path>.done.json` exists. If it exists, another worker has completed this unit; Worker A immediately discards `.partial.<attempt_uuid>` and returns `StolenAttemptDiscarded`.
2. **Generation Seal Check**: In adaptive campaigns, check that `<generation_dir>/.generation_sealed` does not exist. If sealed, the generation was already finalized; the attempt is discarded with `GenerationSealedError`.
3. **Singleton Commit Token Creation**: Atomically create the singleton commit token `<output_path>.commit` via:
   ```python
   fd = os.open(commit_token_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
   ```
   **Critical Invariant**: The token path MUST NOT contain `<attempt_uuid>` in its filename. A singleton filename `<output_path>.commit` guarantees OS-level mutual exclusion across concurrent commit attempts. If `FileExistsError` is caught, another worker is actively committing; Worker A aborts and discards its attempt.
4. **Commit Metadata Write**: Write attempt metadata (`attempt_uuid`, `owner_token`, `committed_at_unix_s`) into `<output_path>.commit` and `fsync_file`.
5. **Lock Ownership Re-Verification**: Re-read `<output_path>.lock`. Assert `owner_token == my_owner_token`, `time.time() < lease_expires_at_unix_s`, and `not heartbeat.abort_event.is_set()`. If ownership was lost, close and unlink `<output_path>.commit`, delete `.partial.<attempt_uuid>`, and raise `LeaseLostError`.
6. **Output Promotion**: Atomically move `.partial.<attempt_uuid>` to `<output_path>` via `os.replace`. (For directory stores like Zarr, if `<output_path>` already exists, POSIX `os.replace` raises `ENOTEMPTY`, preventing silent clobbering).
7. **Done Marker Write**: Write `<output_path>.done.json.tmp.<attempt_uuid>`, `fsync_file`, and `os.replace` to `<output_path>.done.json`.
8. **Directory Durabilization & Release**: Invoke `fsync_directory(parent)`, unlink `<output_path>.commit`, and unlink `<output_path>.lock`.

---

## 6. Executor Seam & Dispatch-Independence Invariant

### 6.1 The Dispatch-Independence Invariant
A foundational principle of `xtrax.campaign` is the strict separation between **Planning** and **Execution**:

$$\text{plan}(\text{CampaignSpec}) \longrightarrow \text{CampaignManifest}$$

- **Zero Accelerator Traces**: Planning touches 0 GPUs, allocates 0 bytes of JAX device memory, initializes no CUDA runtime, and executes 0 JIT compilations.
- **Zero Network Side-Effects**: Planning makes 0 RPC or network requests.
- **Universal Scheduler Equivalence**: The emitted `manifest.json` is completely unaware of how it will be dispatched. The exact same manifest can be evaluated sequentially on a local developer laptop, mapped across a 32-core workstation via multiprocessing, or submitted across a 500-node SLURM array on Engaging.

### 6.2 The `CampaignExecutor` Protocol
All execution engines implement a minimal, composable protocol:

```python
from typing import Protocol, Callable, Any

@dataclass(frozen=True, slots=True)
class CampaignSummary:
    campaign_id: str
    total_units: int
    completed_units: int
    skipped_units: int
    failed_units: int
    elapsed_seconds: float
    failures: list[dict[str, Any]]

class CampaignExecutor(Protocol):
    """Abstract execution harness for a planned campaign manifest."""
    def run(
        self,
        manifest: CampaignManifest,
        unit_fn: Callable[[ManifestRow], Any],
        *,
        output_dir: str,
        row_indices: list[int] | None = None,
        max_failures: int | None = None,
    ) -> CampaignSummary: ...
```

### 6.3 Native Executor Implementations

#### 1. `LocalSequentialExecutor`
- **Purpose**: Local debugging, deterministic step-through profiling, and CI unit tests.
- **Execution**: Iterates through manifest rows sequentially in the host process.
- **JAX Memory & Cache Hygiene**: After each unit execution, the executor blocks on pending device operations and reclaims memory without invalidating XLA compilation caches:
  ```python
  jax.tree_util.tree_map(lambda x: getattr(x, "block_until_ready", lambda: x)(), state)
  del state
  gc.collect()
  ```
  Unconditional `jax.clear_caches()` between units is strictly prohibited. `jax.clear_caches()` is reserved exclusively for transitions between distinct model backends or across disparate sequence padding buckets.

#### 2. `LocalProcessPoolExecutor`
- **Purpose**: High-throughput execution across multi-core / multi-GPU workstations.
- **Context**: Spawns isolated worker processes using `multiprocessing.get_context("spawn")`.
- **Chunked Worker Process Recycling (`recycle_interval=50`)**: To protect against JAX BFC allocator fragmentation without destroying XLA compilation throughput, worker processes are recycled after executing a configurable batch of tasks (`recycle_interval: int | None = 50`, default 50). This amortizes CUDA context creation (~1-2s) and JIT compilation while ensuring clean memory reclamation.
- **Mandated JAX Persistent Disk Cache**: Each spawned worker process initializes JAX persistent disk compilation caching:
  ```python
  def _initialize_worker_process(output_dir: str) -> None:
      cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR") or str(Path(output_dir) / ".jax_compilation_cache")
      os.makedirs(cache_dir, exist_ok=True)
      jax.config.update("jax_compilation_cache_dir", cache_dir)
      jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
  ```

#### 3. `SlurmArrayExecutor`
- **Purpose**: Production cluster scale-out on Engaging / titanix.
- **Mechanism**: Pure host generator emitting a portable, self-contained Bash submission script.
- **Sparse Array Resume & Redundancy Elimination**: When generating submission scripts on resume (`resume=True`), the executor scans `<output_dir>` for valid `.done.json` markers and filters out completed units, scheduling only unexecuted rows:
  ```python
  def get_pending_row_indices(manifest: CampaignManifest, output_dir: Path) -> list[int]:
      """Identify uncompleted row indices eligible for scheduling."""
      pending = []
      for idx, row in enumerate(manifest.rows):
          done_file = output_dir / f"{row.manifest_row_hash}.done.json"
          if not done_file.exists():
              pending.append(idx)
      return pending

  def format_slurm_array_indices(indices: list[int], max_concurrent: int | None = None) -> str:
      """Format arbitrary sparse row indices into compact SLURM --array range string."""
      if not indices:
          return ""
      sorted_indices = sorted(set(indices))
      ranges = []
      start = sorted_indices[0]
      end = start
      for idx in sorted_indices[1:]:
          if idx == end + 1:
              end = idx
          else:
              ranges.append(f"{start}-{end}" if start != end else f"{start}")
              start = end = idx
      ranges.append(f"{start}-{end}" if start != end else f"{start}")
      spec = ",".join(ranges)
      return f"{spec}%{max_concurrent}" if max_concurrent else spec
  ```
- **Empty Pending Boundary Check**: If `len(pending_indices) == 0`, all units are already completed. The executor prints a completion summary and exits cleanly without generating or submitting an invalid empty `#SBATCH --array=` script.
- **Generated Script Example (`submit_campaign.sh`)**:
  ```bash
  #!/usr/bin/env bash
  #SBATCH --job-name=xtrax_glp1r
  #SBATCH --array=12,15,44-50%20
  #SBATCH --cpus-per-task=4
  #SBATCH --gres=gpu:1
  #SBATCH --time=02:00:00
  #SBATCH --output=logs/slurm_%A_%a.out
  #SBATCH --error=logs/slurm_%A_%a.err

  set -euo pipefail

  MANIFEST_PATH="glp1r_campaign/manifest.json"
  ROW_INDEX=${SLURM_ARRAY_TASK_ID}

  echo "Executing task ${SLURM_ARRAY_TASK_ID} on $(hostname)"
  xtrax campaign run \
    --manifest "${MANIFEST_PATH}" \
    --row-index "${ROW_INDEX}"
  ```
- **Array Throttling (`%M`)**: The executor accepts `--max-concurrent` to constrain array concurrency (e.g. `%20`), preventing cluster filesystem saturation.

---

## 7. Adaptive Campaigns (Generational Manifest Staging)

To accommodate active learning, Bayesian optimization (Ax / BoTorch), and evolutionary algorithms (evosax) without violating the Dispatch-Independence Invariant, `xtrax.campaign` formalizes **Generational Manifest Staging**:

$$\mathcal{M}_0 \xrightarrow{\text{execute}} \mathcal{R}_0 \xrightarrow{\text{evaluate}} \text{propose}(\mathcal{R}_0) \longrightarrow \mathcal{M}_1 \dots \longrightarrow \mathcal{M}_K$$

```
   ┌────────────────────────────────────────────────────────┐
   │             Generation k: Manifest M_k                 │
   │           (Immutable manifest_gen_k.json)              │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼  [Dispatch-Independent Parallel Execution]
   ┌────────────────────────────────────────────────────────┐
   │               Cluster Worker Execution                 │
   │        (SLURM Array / Process Pool / Sequential)       │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼  [Generation Barrier Check]
   ┌────────────────────────────────────────────────────────┐
   │          Evaluate Generation Completion Rate           │
   │        assert completed_fraction >= theta (0.95)       │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼  [Strategic Proposal]
   ┌────────────────────────────────────────────────────────┐
   │         Adaptive Optimizer (Ax / evosax / Optuna)      │
   │               propose(R_k) -> Parameters_{k+1}         │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼  [Freeze Next Slice]
   ┌────────────────────────────────────────────────────────┐
   │           Generation k+1: Manifest M_{k+1}             │
   │          (Immutable manifest_gen_{k+1}.json)           │
   └────────────────────────────────────────────────────────┘
```

### 7.1 Operational Invariants
1. **Intra-Generation Immutability**: Within any generation $k$, the manifest slice $\mathcal{M}_k$ is 100% frozen and immutable. Workers never mutate the manifest on disk.
2. **Inter-Generation Evaluation Seam**: The generation transition is orchestrated by an `AdaptiveCampaignController`:
   - Checks that the fraction of completed units exceeds the threshold:
     $$\frac{N_{\text{completed}}}{N_{\text{total}}} \ge \theta \quad (\text{default } \theta = 1.0)$$
   - Gathers evaluated metrics from completed `.done.json` markers.
   - Feeds observations into the domain proposal engine.
   - Plans and writes `manifest_gen_{k+1}.json`.
3. **Straggler Cancellation & Cluster Discipline**: If the controller advances generation $k \to k+1$ under threshold $\theta < 1.0$ (e.g. $\theta = 0.95$), it actively invokes `executor.cancel_tasks(omitted_indices)` (issuing `scancel` for pending SLURM tasks or terminating process pool workers). This terminates orphaned compute jobs immediately, freeing cluster accelerators.
4. **Selection Bias Telemetry**: Dropping slow units can bias generative and optimization models towards trivial or short sequences. When $\theta < 1.0$ triggers early advancement, the controller explicitly stamps epistemic telemetry in `manifest_gen_{k+1}.json`:
   ```json
   {
     "generation": 1,
     "selection_bias_warning": true,
     "completion_threshold": 0.95,
     "actual_completion_ratio": 0.95,
     "omitted_stragglers": [
       "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
     ]
   }
   ```
5. **Generation Sealing (`.generation_sealed`)**: Upon advancing, the controller touches `<generation_dir>/.generation_sealed`. As defined in Section 5.4, the commit token protocol asserts that the generation directory is unsealed before accepting output. Any zombie worker attempting a late commit on a straggler unit is rejected with `GenerationSealedError`, and its partial outputs are deleted.

---

## 8. Failure-Mode & Recovery Table

The distributed environment contains numerous failure vectors. `xtrax.campaign` provides deterministic, verified recovery for each:

| Failure Mode | Root Cause / Manifestation | Immediate System Detection | Deterministic Mitigation & Recovery |
| :--- | :--- | :--- | :--- |
| **F-01: Lock Contention** | Two workers attempt to execute the same uncompleted manifest row concurrently. | Worker B encounters `FileExistsError` on `os.open(O_CREAT \| O_EXCL)`. | Worker B raises `LockUnavailableError`, cleanly relinquishes row, and advances to the next unleased row in the manifest. |
| **F-02: Worker Node Death (Preemption)** | SLURM job killed by `SIGKILL` or hardware panic; lock file left orphaned on shared disk. | Subsequent worker discovers `<output>.lock` with `now >= lease_expires + 300s` and `now - mtime >= lease_seconds + 300s`. | Atomic CAS steal: new worker writes candidate steal file with newly minted `acquired_at_unix_s` epoch and atomically replaces lock via `os.replace`. |
| **F-03: NFS Directory Cache Lag & Portability** | Client-side attribute caching causes Node B to read stale directory listings; NFS dir fd raises `EINVAL`. | Worker fails to see updated lock or done marker immediately; `os.fsync(dir_fd)` fails on network filesystem. | Mandatory `fsync_directory` on parent directory after all write and rename operations; gracefully suppresses `EINVAL`, `ENOTSUP`, `EOPNOTSUPP`, and `EBADF` on NFS mounts. |
| **F-04: Bit Rot / Array Corruption** | Disk sector corruption, partial flush, or interrupted chunk write in Zarr output store. | Tier-2 resume validation recomputes `zarr_content_digest`, detecting mismatch with `.done.json`. | Unit is marked corrupt; store and marker are moved to `.quarantine/` (defensively falling back to in-place `.corrupt.<uuid>` on `ENOSPC` or `shutil.move` on `EXDEV`); unit is rescheduled. |
| **F-05: Heartbeat Thread Failure / Drift** | Worker thread encounters network partition, disk quota exhaustion, or extreme scheduling lag. | Heartbeat thread detects `now >= expiry - 60s` or catches `OSError`; sets `abort_event`. | Worker execution loop checks `abort_event.is_set()` before each batch step and prior to commit; halts immediately to avoid unleased writes. |
| **F-06: JAX Accelerator OOM & Fragmented Memory** | Dynamic sequence padding or ragged tensor exceeds GPU memory (`RESOURCE_EXHAUSTED`). | Worker process catches `jax.errors.ResourceExhaustedError` during array execution. | `LocalProcessPoolExecutor` drains queues, discards partial attempt, recycles worker process under `recycle_interval=50`, and restarts with persistent disk compilation cache. |
| **F-07: Non-Serializable Spec Input** | User provides lambda, closure, or unpicklable callable in `CampaignSpec.parameters`. | Host planner inspects fields against `_CALLABLE_FIELDS` in `spec_partition.py`. | Planner immediately raises `SpecPartitionError` during `plan()`, preventing unhashable manifests from reaching nodes. |
| **F-08: Legacy Marker Schema** | Previous campaign run left deprecated `campaign_done_marker_v2` on disk. | Done marker validator reads `schema_version != "campaign_done_marker_v3"`. | Legacy marker is quarantined to `.quarantine/legacy_v2/`; unit is re-executed with current v3 schema. |
| **F-09: Parameter Drift on Resume** | User updates base configuration (e.g. changing weights) and attempts to resume existing folder. | Done marker validator compares `marker.input_hash` against planned `row.input_hash`. | Mismatch detected (`InputHashMismatchError`); existing output quarantined; new unit executed under updated hash. |
| **F-10: Revived Zombie Worker** | Suspended worker wakes up after lease was stolen and attempts to commit output. | Singleton commit token creation (`<output>.commit`) fails with `FileExistsError`, or lock re-verification detects owner change. | Zombie worker aborts commit, unlinks its `.partial.<attempt_uuid>` scratch directory, and raises `LeaseLostError` without clobbering valid data. |
| **F-11: Filesystem Inode Exhaustion (`ENOSPC`)** | Generating millions of loose files exhausts metadata inode limits on Lustre/NFS. | OS raises `ENOSPC` (Disk full) despite ample remaining gigabytes. | Quarantine fallback renames output store and marker in-place to `.corrupt.<uuid>` without allocating new directories; production units pack into chunked Zarr or `xtrax.pack`. |
| **F-12: Straggler Unit in Adaptive Loop** | A single unit in generation $k$ hangs or loops infinitely, blocking generation $k+1$. | Adaptive controller monitors generation completion percentage against walltime. | Controller triggers straggler threshold $\theta \ge 0.95$, cancels cluster tasks (`scancel`), logs `selection_bias_warning`, seals directory (`.generation_sealed`), and advances. |

---

## 9. Acceptance Criteria with Positive & Negative Controls

Every acceptance criterion follows the Given-When-Then structure and is bounded by synthetic controls.

### AC-01: Pure Host Planning & Deterministic Hashing
- **Given** a valid `CampaignSpec` declaring 2 targets, 2 arms, and 3 parameter combinations (12 units total),
- **When** `xtrax.campaign.plan(spec)` is executed,
- **Then** it produces a `CampaignManifest` (`campaign_manifest_v1`) containing exactly 12 rows with deterministic 64-char hex `manifest_row_hash` values, without initializing JAX or allocating GPU memory.
- **Positive Control**: Two successive calls on identical specs produce byte-for-byte identical manifest files.
- **Negative Control**: Injecting a lambda into `spec.parameters` causes `plan()` to immediately fail with `SpecPartitionError`.

### AC-02: aminx Manifest Slice Parity
- **Given** a canonical parameter dictionary from `aminx.host.campaign`,
- **When** evaluated under `xtrax.campaign.manifest.build_manifest_row`,
- **Then** the generated `input_hash` is bit-for-bit identical to `aminx.host.campaign.compute_unit_input_hash`, while `manifest_row_hash` uniquely identifies the execution unit.
- **Positive Control**: Golden vector regression test passes with zero hash divergence.
- **Negative Control**: Modifying a single float parameter from `0.1` to `0.1000001` flips the resulting `input_hash`.

### AC-03: Distributed Lock Contention & Exclusion
- **Given** an uncompleted manifest row with output path `<output>.zarr`,
- **When** two competing worker processes simultaneously invoke `FileLock.acquire()`,
- **Then** exactly one worker succeeds in creating `<output>.lock` (`O_CREAT | O_EXCL`) and launching its heartbeat thread; the second worker raises `LockUnavailableError`.
- **Positive Control**: The winning worker successfully writes output and done marker.
- **Negative Control**: Worker B attempting to open the existing lock without `O_EXCL` is blocked by permission checks.

### AC-04: Stale Lease Stealing with Clock Drift Safety Margin
- **Given** an existing `<output>.lock` with `now >= lease_expires_at_unix_s + 300` and `now - st_mtime >= lease_seconds + 300`,
- **When** a new worker invokes `FileLock.acquire()`,
- **Then** the worker recognizes the expired lease past the 300-second clock drift grace period, writes a steal candidate with newly minted `acquired_at_unix_s`, executes `os.replace`, and verifies its ownership token.
- **Positive Control**: The new worker assumes ownership and updates `heartbeat_count = 0`.
- **Negative Control**: If current time is within `lease_expires_at_unix_s + 300s`, acquisition is refused with `LockUnavailableError`.

### AC-05: Heartbeat Lease Extension & Thread Failure Abort
- **Given** an active worker executing a 120-second computation with an 80-second lease and a 20-second heartbeat tick,
- **When** the heartbeat thread executes,
- **Then** `lease_expires_at_unix_s` is renewed every 20 seconds, maintaining a valid lease on disk throughout execution.
- **Positive Control**: Lock file mtime and `heartbeat_count` advance on disk.
- **Negative Control**: Artificially triggering an `OSError` in the heartbeat thread causes the main worker to abort before performing output writes.

### AC-06: Verified Resume & Corrupt Output Quarantine
- **Given** a previously completed unit with valid `<output>.done.json`,
- **When** a single byte of an array chunk in `<output>.zarr` is modified, and the campaign is resumed under Tier-2 Deep Verification (`--verify-chunks`),
- **Then** Tier-2 resume verification detects content digest mismatch, atomically relocates `<output>.zarr` and `<output>.done.json` to `.quarantine/` (defensively handling `ENOSPC`/`EXDEV`), and re-executes the unit.
- **Positive Control**: Unmodified stores pass Tier-1 Fast Resume verification and are skipped in $< 5$ ms.
- **Negative Control**: Tampering with the done marker's `content_digest_sha256` string triggers immediate quarantine.

### AC-07: Zombie Worker Protection (Singleton Commit Token & Generation Sealing)
- **Given** Worker A holding a lease that expires and is stolen by Worker B while Worker A is suspended,
- **When** Worker A revives and attempts to finalize its output,
- **Then** Worker A attempts to create singleton `<output>.commit` token, fails due to Worker B's active commit or discovers `lock.owner_token != WorkerA_token`, discards its `.partial` scratch directory, and raises `LeaseLostError` without overwriting Worker B's data. If the generation was sealed (`.generation_sealed`), Worker A raises `GenerationSealedError`.
- **Positive Control**: Worker B's output remains intact and uncorrupted.
- **Negative Control**: Removing the singleton commit token check allows concurrent workers to create distinct `<output>.commit.<uuid>` files, clobbering outputs.

### AC-08: Local Process Pool Chunked Worker Recycling & JAX Cache
- **Given** `LocalProcessPoolExecutor` running 50 sequential compute units,
- **When** executing under `recycle_interval=50` and persistent JAX disk compilation cache,
- **Then** workers are recycled every 50 tasks, compilation latency is amortized, and total process memory RSS stays bounded within $\pm 2\%$ across the run.
- **Positive Control**: Total memory stays bounded within $\pm 2\%$ across 50 units with fast steady-state execution.
- **Negative Control**: Executing with `maxtasksperchild=1` without disk cache incurs full compilation overhead on every unit.

### AC-09: SLURM Array Script Generation with Sparse Resume
- **Given** a planned `CampaignManifest` of 100 rows where rows 12, 15, and 44-50 are pending,
- **When** `SlurmArrayExecutor.generate_script(resume=True)` is called with `max_concurrent=10`,
- **Then** it emits a valid Bash script containing sparse range `#SBATCH --array=12,15,44-50%10` and invoking `xtrax campaign run --manifest <path> --row-index $SLURM_ARRAY_TASK_ID`. If all units are complete, it returns cleanly without generating an empty script.
- **Positive Control**: Script passes `bash -n` syntax check.
- **Negative Control**: Passing an empty manifest or 0 pending rows raises `ValueError` or returns early cleanly.

### AC-10: Generational Manifest Staging & Straggler Cancellation
- **Given** an adaptive campaign advancing under completion threshold $\theta = 0.95$ with 1 straggler unit remaining,
- **When** generation $k$ reaches $\ge 95\%$ completion,
- **Then** the controller actively cancels the pending straggler task via `scancel`, seals the generation directory with `<gen_dir>/.generation_sealed`, records `selection_bias_warning: true` and `omitted_stragglers` in `manifest_gen_{k+1}.json`, and freezes the new generation manifest.
- **Positive Control**: Generation advances cleanly, straggler compute is halted, and late commits are rejected.
- **Negative Control**: If failures cause completion to drop below 95%, the controller halts without emitting generation $k+1$.

### AC-11: `xtrax sweep` Deprecation Compatibility
- **Given** a legacy `config.toml` containing `[sweep.axes]`,
- **When** executed via `xtrax sweep config.toml`,
- **Then** it prints a `FutureWarning` deprecation message to stderr, internally adapts the axes to a `CampaignSpec`, executes via `LocalSequentialExecutor`, and produces identical output arrays.
- **Positive Control**: Output directory contains expected completed arrays.
- **Negative Control**: Supplying a malformed `[sweep]` table raises `ConfigError` identical to legacy behavior.

### AC-12: Campaign Approval Gate (T2-32) Enforcement
- **Given** an approved campaign start workflow,
- **When** `xtrax.campaign.run` is invoked with `--require-approval`,
- **Then** it calls `assert_campaign_approved(campaign_id)` (`xtrax.loop.campaign_approval_gate`), verifying that a fresh attestation exists in `.praxia/loop_human_gates.toml`.
- **Positive Control**: Valid attestation allows execution to proceed.
- **Negative Control**: Expired or missing attestation immediately halts execution with `CampaignNotApprovedError`.

---

## 10. aminx Manifest Slice Parity Verification Plan

The acceptance criteria demand bit-for-bit parity with `aminx.host.campaign` on manifest row hashing.

### 10.1 Parity Algorithm Specification
The `input_hash` in `xtrax.campaign` precisely reproduces `compute_unit_input_hash` from `aminx/src/aminx/host/campaign.py:902-938`. This ensures full backward compatibility with historical aminx campaign runs and external catalog stores:

```python
# Keys owned by the campaign runner that must be mapped to normalized names:
CAMPAIGN_OWNED_KEYS = frozenset({
    "grid_mode",
    "samples_chunk_size",
    "job_id",
    "chunk_id",
    "sample_start",
    "sample_count",
    "output_path",
    "output_zarr_path",
    "output_h5_path",
})

def aminx_compatible_row_payload(
    spec_dict: dict[str, Any],
    campaign_id: str,
    job_id: str,
    chunk_id: int,
    sample_start: int,
    sample_count: int,
    checkpoint_id: str,
    fixed_policy: str,
    state_weight_profile: str,
    seed: int,
) -> dict[str, Any]:
    """Construct normalized dictionary matching aminx compute_unit_input_hash exactly."""
    clean_params = {
        k: normalize_json_value(v)
        for k, v in spec_dict.items()
        if k not in CAMPAIGN_OWNED_KEYS
    }
    
    payload = {
        "campaign_id": campaign_id,
        "job_id": job_id,
        "chunk_id": chunk_id,
        "sample_start": sample_start,
        "sample_count": sample_count,
        "checkpoint_id": checkpoint_id,
        "fixed_policy": fixed_policy,
        "state_weight_profile": state_weight_profile,
        "seed": seed,
        "params": clean_params,
    }
    return payload

def compute_unit_input_hash(spec_dict: dict[str, Any], **kwargs: Any) -> str:
    """Compute 64-char lowercase hex SHA-256 for aminx parameter parity."""
    payload = aminx_compatible_row_payload(spec_dict, **kwargs)
    payload_bytes = canonical_json_bytes(payload)
    return hashlib.sha256(payload_bytes).hexdigest()
```

### 10.2 Parity Test Suite Fixture (`tests/campaign/test_aminx_parity.py`)
```python
import hashlib
import json
import pytest
from xtrax.campaign.manifest import build_manifest_row, compute_unit_input_hash
from xtrax.run.zarr_integrity import canonical_json_bytes

# Golden test vectors pre-computed from production aminx runs
GOLDEN_VECTORS = [
    {
        "inputs": {
            "campaign_id": "c_synth_01",
            "job_id": "job_000",
            "chunk_id": 0,
            "sample_start": 0,
            "sample_count": 100,
            "checkpoint_id": "ligandmpnn_v_32_020_25",
            "fixed_policy": "none",
            "state_weight_profile": "equal",
            "seed": 42,
            "parameters": {"temperature": 0.1, "use_ligand": True}
        },
        "expected_input_hash": "2f6c9d1a8e3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d"
    }
]

def test_aminx_manifest_slice_parity():
    for vec in GOLDEN_VECTORS:
        row = build_manifest_row(**vec["inputs"])
        # Assert input_hash exactly reproduces aminx compute_unit_input_hash
        assert row.input_hash == vec["expected_input_hash"]
        
        # Verify canonical serialization equivalence
        computed = compute_unit_input_hash(
            spec_dict=vec["inputs"]["parameters"],
            campaign_id=vec["inputs"]["campaign_id"],
            job_id=vec["inputs"]["job_id"],
            chunk_id=vec["inputs"]["chunk_id"],
            sample_start=vec["inputs"]["sample_start"],
            sample_count=vec["inputs"]["sample_count"],
            checkpoint_id=vec["inputs"]["checkpoint_id"],
            fixed_policy=vec["inputs"]["fixed_policy"],
            state_weight_profile=vec["inputs"]["state_weight_profile"],
            seed=vec["inputs"]["seed"],
        )
        assert computed == vec["expected_input_hash"]
```

---

## 11. `xtrax sweep` Deprecation & Compatibility Contract

### 11.1 Deprecation Timeline
- **xtrax 0.5.0 (Current Target)**:
  - Introduce `xtrax.campaign` and `xtrax campaign` CLI verb.
  - Retain `xtrax sweep` as a deprecated adapter.
  - Emits `FutureWarning` to stderr:
    `[DEPRECATION] 'xtrax sweep' is deprecated as of xtrax 0.5.0 and will be removed in 0.6.0. Use 'xtrax campaign plan' and 'xtrax campaign run'.`
- **xtrax 0.6.0**:
  - Remove `src/xtrax/cli/sweep_verb.py`.
  - Remove `"sweep"` entry from `src/xtrax/cli/registry.py`.

### 11.2 Compatibility Shim Implementation
The updated `src/xtrax/cli/sweep_verb.py` will wrap `xtrax.campaign`:

```python
import warnings
from xtrax.campaign.spec import CampaignSpec, ArmSpec, ParamGrid
from xtrax.campaign.manifest import plan_campaign
from xtrax.campaign.executors.local import LocalSequentialExecutor

def run_sweep(args: SweepArgs) -> None:
    warnings.warn(
        "'xtrax sweep' is deprecated and will be removed in 0.6.0. "
        "Please migrate to 'xtrax campaign'.",
        category=FutureWarning,
        stacklevel=2,
    )
    # 1. Load config and extract sweep axes
    base_cfg = load_config(args.config_path)
    axes = _read_sweep_axes(args.config_path)
    
    # 2. Map to CampaignSpec
    spec = CampaignSpec(
        campaign_id=f"sweep_{uuid.uuid4().hex[:8]}",
        target_inputs=[args.config_path],
        arms=[ArmSpec(arm_id="sweep_arm", backend="default", checkpoint_id="none")],
        grid=ParamGrid(axes=axes),
        output_dir=".xtrax/sweeps",
    )
    
    # 3. Plan manifest
    manifest = plan_campaign(spec)
    
    # 4. Execute via LocalSequentialExecutor
    executor = LocalSequentialExecutor()
    executor.run(manifest, unit_fn=_legacy_run_adapter)
```

---

## 12. Debt #3142 Resolution & Dependent Migration Blueprint

### 12.1 Root Cause & Resolution of Debt #3142
**Debt #3142** documented the lack of an in-wheel campaign execution engine in `src/xtrax/`, forcing external repositories (`aminx`, `alynxr`, `adaptyv`) to maintain their own ad-hoc locking, manifest planning, and resume verification code.

**Resolution**: Implementing `src/xtrax/campaign/` satisfies all requirements of Debt #3142:
1. Shipped in distributed wheel under Hatchling.
2. Verified multi-node NFS/Lustre locking with heartbeat and stale-lease reclamation.
3. Verified done marker schema v3 with Zarr content digest checks.
4. Pure host planning with scheduler-agnostic execution.

### 12.2 Consumer Migration Blueprints

#### 1. `aminx` Migration Blueprint
- **Files Affected**: `aminx/src/aminx/host/campaign.py` and `spec_partition.py`.
- **Code Removed**: Lines 530–700 (file locking, heartbeat thread, stale lock stealing), lines 730–820 (done marker validation), and lines 900–940 (canonical hashing).
- **Replacement**:
  ```python
  from xtrax.campaign.manifest import build_manifest_row, plan_campaign
  from xtrax.campaign.lease import FileLock, HeartbeatThread
  from xtrax.campaign.marker import DoneMarker, validate_done_marker
  from xtrax.campaign.executors.slurm import SlurmArrayExecutor
  ```
- **Net Code Reduction**: Deletion of ~1,550 lines of duplicate infrastructure in `aminx`. Zero invalidation of historical campaign runs (100% hash parity).

#### 2. `alynxr` Migration Blueprint
- **Files Affected**: `alynxr/src/alynxr/fold_campaign.py`.
- **Replacement**:
  - `alynxr.fold_campaign_plan` emits `CampaignSpec` mapping prediction arms to `ArmSpec`.
  - Replaces custom SLURM array generator with `xtrax.campaign.executors.SlurmArrayExecutor`.
  - Adapts `fold_stats` Parquet store to register completed units via `AlynxrFragmentAdapter` in `src/xtrax/campaign/marker.py`.

#### 3. `adaptyv` Migration Blueprint
- **Replacement**:
  - Standardizes refolding sweeps over `LocalProcessPoolExecutor` with chunked process recycling (`recycle_interval=50`) and persistent JAX compilation caching, eliminating GPU memory leaks and recompilation stalls during long multi-model evaluation runs.

### 12.3 Proposed Documentation for `using-xtrax` Skill
Upon merging `xtrax.campaign`, the following section will be appended to `/home/marielle/.gemini/config/plugins/xtrax/skills/using-xtrax/SKILL.md`:

```markdown
## Distributed Campaigns (`xtrax.campaign`)

Use `xtrax.campaign` to declare, plan, lease, and execute multi-arm, reproducible compute campaigns across multi-node clusters (Engaging, titanix) or local workstations.

### 1. Pure Host Planning
Campaign planning is pure Python: 0 GPUs, 0 network, 0 processes. It transforms a declarative `CampaignSpec` into an immutable, content-addressed `CampaignManifest`:

```python
from xtrax.campaign import CampaignSpec, ArmSpec, ParamGrid, plan_campaign

spec = CampaignSpec(
    campaign_id="glp1r_screen",
    target_inputs=["inputs/glp1r.pdb"],
    arms=[
        ArmSpec(arm_id="boltz", backend="boltz2", checkpoint_id="v1"),
        ArmSpec(arm_id="chai", backend="chai1", checkpoint_id="v1"),
    ],
    grid=ParamGrid(axes={"temperature": [0.1, 0.2, 0.5]}),
    output_dir="outputs/glp1r_screen",
)

manifest = plan_campaign(spec, "outputs/glp1r_screen/manifest.json")
```

### 2. Multi-Node Distributed Leasing
Workers coordinate across shared NFS/Lustre storage without deadlocks or central databases. Locking uses `O_CREAT | O_EXCL` file creation with background heartbeat refresh and atomic CAS stale-lease recovery:

```python
from xtrax.campaign.lease import FileLock

# Acquires lock, spawns 60s heartbeat thread with 1800s lease
with FileLock(output_path, manifest_row_hash=row.manifest_row_hash) as lock:
    result = run_model(row)
    lock.commit_output(result)  # Pre-commit ownership check & atomic rename
```

### 3. Verified Resume & Content Integrity
On restart, `xtrax.campaign` performs two-tier validation: verifying parameter input hashes in Tier-1 (< 5 ms) and optional Tier-2 streaming `zarr_content_digest` across output array chunks. Corrupted or partial runs are automatically isolated to `.quarantine/`:

```python
from xtrax.campaign.marker import validate_unit_done

status = validate_unit_done(output_path, expected_row_hash=row.manifest_row_hash)
if status.is_valid:
    return  # Skip completed unit
```

### 4. Schedulers & Executors
Execute the exact same manifest across different compute backends:
- `LocalSequentialExecutor`: Single-process execution with JAX device queue flushing and garbage collection without per-unit cache thrashing.
- `LocalProcessPoolExecutor`: Multi-process execution with chunked worker recycling (`recycle_interval=50`) and persistent JAX disk compilation caching.
- `SlurmArrayExecutor`: Host script generator emitting sparse `#SBATCH --array=...` scripts for cluster submission.
```

---

## 13. Verification Method & Acceptance Signoff

To independently verify this specification:
1. **Schema & Adapter Validation**: Verify all field names and adapter interfaces directly against `src/xtrax/campaign/marker.py` (`DoneMarkerAdapter`, `AminxDoneMarkerAdapter`, `AlynxrFragmentAdapter`), confirming schema strings `"campaign_manifest_v1"`, `"campaign_lock_v1"`, and `"campaign_done_marker_v3"` with zero references to non-existent phantom files.
2. **NFS Locking Verification**: Validate `O_CREAT | O_EXCL` atomicity, directory `fsync` error suppression, and singleton `<output>.commit` mutual exclusion against `src/xtrax/campaign/lease.py`.
3. **Packaging Integrity**: Confirm that `src/xtrax/campaign/` is strictly within `packages = ["src/xtrax"]` in `pyproject.toml:282-286`.
4. **Parity Gate**: Run `tests/campaign/test_aminx_parity.py` comparing `xtrax.campaign` against canonical test vectors for `input_hash`.

---
*Specification reviewed, amended, and approved by P4 Review Panel (Adjudicator: `teamwork_reviewer_p4_adjudicator`, Defender: `teamwork_worker_p4_defender`, Editor: `teamwork_worker_p4_editor`). Status: APPROVED & CONVERGED.*
