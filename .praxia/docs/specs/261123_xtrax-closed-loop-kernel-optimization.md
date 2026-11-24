# Spec: `xtrax` Closed-Loop Kernel Optimization System (`261123`)

- **Status**: Cohered v4 (Adversarial Verdict: ACCEPT — Round 6 Final Sign-Off)
- **Target Repository**: `/home/marielle/projects/xtrax` (isolated worktree `/home/marielle/projects/xtrax/.claude/worktrees/kernel-opt-loop`, branch `feat/kernel-opt-loop` off `main`, never directly on `main`)
- **Target Committed Spec Path (once worktree is created)**: `.praxia/docs/specs/261123_xtrax-closed-loop-kernel-optimization.md`
- **Downstream Dogfood Consumer**: `prolix` (`v39-complete` worktree, Phase 2 after `xtrax` MVP lands)
- **Contract Compatibility**: Preserves `ProbeRecord` `CONTRACT_VERSION = "3.0"` (zero field additions/removals on `ProbeRecord`)

---

## 1. Problem Statement & Empirical Motivation

`xtrax.profiling` (`record.py`, `claims.py`, `emitters.py`, `unit_probe.py`, `trace.py`, `report.py`, `rules.py`) and its sibling skills (`xtrax-probing`, `xtrax-optimizing`, `xtrax-activation-parity`) already enforce provenance stamping, stage floors, single-commit paired configs, and negative controls. However, real GPU Pallas kernel optimization cycles (demonstrated in Prolix `v3.0 -> v3.4` across `NVIDIA L40S`, `H200`, `A100`, and `RTX PRO 6000 Blackwell`) exposed five structural gaps where agents fell back to ad-hoc JSONs and manual debugging:

1. **Cross-Commit Blind Spot**: `claims.py:_UNANIMITY_FIELDS` (`claims.py:54`) requires `git_sha` unanimity across compared records, so `xtrax.profiling` cannot evaluate a baseline commit (`sha_A`) against a kernel-edit candidate commit (`sha_B`).
2. **CUDA-Graph Attribution Collapse**: Under XLA's default `command_buffer`, Perfetto traces collapse a multi-kernel step into a single `command_buffer_*` thunk (`prolix/.praxia/docs/reference/260901_named-scope-attribution-limits.md:40-79`), and `named_scope` recovery dropped to 0.03% at production scale. Although `trace.py:165-179` supports `triton:<kernel_name>` custom-call attribution, `xtrax` lacks a standardized multi-pass GPU probe linking a production `cmdbuf=on` wall record with an isolated `cmdbuf=off` diagnostic kernel breakdown record.
3. **Missing Kernel & Algorithmic Efficiency Signals**: `unit_probe.py` does not record wall dispersion (`MAD`/`IQR`), peak device VRAM, PTX/`ptxas`/Triton register & spill counts, `mixed / single` precision ratios (1:64 FP64 cliff on `L40S` / workstation `RTX` Blackwell), or domain-supplied work-efficiency / atomic-contention counters.
4. **Fidelity & Non-Degeneracy Not Bound to Keep/Revert**: `BUG-TRITON-ATOM1` in Prolix `v3.4` produced an apparent 4x speedup (`0.14 ms/step`) by silently dropping off-diagonal neighbor tile pairs. `xtrax.testing.kernel_parity` is currently decoupled from `ProbeRecord` keep/revert verdicts and does not track work retention against baseline.
5. **Optional `bathos` Interop & Knowledge Drift**: `xtrax` must allow optional `bathos` run/campaign correlation without introducing a hard `bathos` dependency inside `src/xtrax/`. Additionally, un-upstreamed lessons `MK-50..MK-64` (`prolix/docs/plans/2026-10-28-xtrax-megakernel-composition-ledger.md:584-812`) and `v3.4` benchmark findings remain outside `xtrax-optimizing`.

---

## 2. Hard Architectural Invariants (Non-Negotiables)

Every design choice in this spec MUST satisfy all standing `xtrax` repository contracts:

1. **Leaf-Package Boundary** (`tests/profiling/test_claim_contract.py:84-127`, `scripts/audit_profiling_contract.py:26-52`):
   - Every module under `src/xtrax/profiling/` MUST have zero `prolix` imports, zero sibling `xtrax.*` imports (only `xtrax.profiling.*`), zero relative imports, and zero `bathos` imports.
   - **Zero module-level `import jax`**: EVERY module under `src/xtrax/profiling/` — explicitly including new modules `compare.py`, `kernel_probe.py`, `_diag_worker.py`, and `triage.py` — MUST omit module-level `import jax` (import `jax` lazily inside functions only where runtime execution requires it, i.e., `unit_probe.py`, `trace.py`, `kernel_probe.py`, and `_diag_worker.py`). `src/xtrax/profiling/__init__.py` remains unchanged so package import stays lightweight.
   - For runtime `beartype` compatibility (`record.py:239-243`, `tests/conftest.py`), modules inside `src/xtrax/profiling/` import `ProbeRecord` (and `DeltaVerdict` where annotated) directly at module top level (never gated behind `if TYPE_CHECKING:`), which remains 100% JAX-free because `record.py` and `compare.py` have no top-level `jax` import.
2. **Strict `bathos` & `praxia` Independence Under `src/xtrax/`** (`scripts/audit_bathos_independence.py:41-59`, `scripts/audit_dispatch_independence.py:50-54`, `distribution/coverage_dag.toml:12-37, 66-77`):
   - **Zero** `import bathos` or `from bathos` statements anywhere under `src/xtrax/` (including lazy inside-function imports!) and zero `mcp__bathos` strings.
   - **Zero** `import/from praxia`, `mcp__praxia`, or `rig[_-]run` substrings anywhere inside `src/xtrax/` files (including docstrings, comments, and `recommended_action` strings — `scripts/audit_dispatch_independence.py:50-54` scans raw file text with an empty `ALLOWLIST`).
   - All direct `bathos` library imports belong exclusively under `controller/` (covered by `tier4_controller` + `just audit-controller-types` and omitted from shipped `tier1_core` wheels).
   - Pure JSON/dict formatting helpers and environment-tag readers (`BTH_RUN_ID`, `BTH_CAMPAIGN_ID` via stdlib `os.environ`) require zero `bathos` import and live cleanly inside `src/xtrax/profiling/emitters.py`.
3. **No `from __future__ import annotations`** (`tests/audit/test_no_future_annotations.py:1-84`, Gate N0.2):
   - Every new file under `src/xtrax/` MUST omit `from __future__ import annotations`. New files under `controller/` should likewise omit `from __future__ import annotations`.
4. **`ProbeRecord` `CONTRACT_VERSION = "3.0"` & Dataclass Shape Compatibility** (`src/xtrax/profiling/record.py`, `src/xtrax/devtools/gates/dataclass_shape_diff.py:13-25`):
   - **Zero** fields added to or removed from `ProbeRecord`.
   - Continuous numeric measurements live in `ProbeRecord.metrics: dict[str, float]` (finite, non-boolean). Discrete statuses, hypotheses, step-time basis, and optional `bathos` IDs live in `ProbeRecord.config: dict[str, str]`.
   - Any defaulted fields added to existing shipped dataclasses (`UnitProbeResult`, `ProfileArgs`) MUST be appended strictly at the **end** of the dataclass field list so positional construction and `dataclass_shape_diff` remain unbroken.
5. **Telemetry Coverage Contract** (`audit/telemetry_coverage.toml:39`, `Justfile:56-59`):
   - Extending the existing `profile` CLI verb (declared `profile = "probes"`) preserves telemetry coverage with zero schema changes.
6. **Code Quality Gates (`complexity`, `api_ergonomics`, `interrogate`)** (`Justfile:107-133`):
   - Decompose multi-step evaluators into single-purpose helpers (`complexipy <= 15`, low cyclomatic complexity, `<= 5` required params on non-dataclass public callables, full docstrings across all modules/classes/functions/methods).

---

## 3. MVP Architecture & Module Specifications

### 3.1 `src/xtrax/profiling/compare.py` — Cross-Commit `DeltaVerdict` & Ratchet Gate

Purpose: Compare a baseline `ProbeRecord` (`sha_A`) against a candidate `ProbeRecord` (`sha_B`) across commits while enforcing strict environment unanimity, SHA verification, fidelity/work-retention, dispatch/compile, VRAM, and statistical noise-floor invariants. Zero `jax` imports; direct top-level import of `ProbeRecord`.

#### Public Types

```python
class VerdictStatus(enum.Enum):
    KEEP = "keep"
    ADVISORY_ONLY = "advisory_only"
    REVERT_REGRESSION = "revert_regression"
    REVERT_NOISE = "revert_noise"
    REJECT_PARITY = "reject_parity"
    REJECT_DISPATCH = "reject_dispatch"
    REJECT_VRAM = "reject_vram"
    REJECT_UNVERIFIABLE = "reject_unverifiable"

@dataclasses.dataclass(frozen=True, slots=True)
class ParityContract:
    max_rel_err: float | None = 1e-4
    max_abs_err: float | None = None
    max_vjp_rel_err: float | None = None
    require_non_degeneracy: bool = True
    require_corruption_control: bool = True
    min_corruption_rel_err: float | None = None
    require_export_ok: bool = False
    min_work_active_count: float | None = None
    min_work_retention_ratio: float = 0.99
    metric_ceilings: dict[str, float] = dataclasses.field(default_factory=dict)
    metric_floors: dict[str, float] = dataclasses.field(default_factory=dict)

@dataclasses.dataclass(frozen=True, slots=True)
class DeltaVerdict:
    status: VerdictStatus
    promotable: bool
    primary_metric: str
    baseline_value: float | None
    candidate_value: float | None
    speedup_x: float | None
    rel_delta_pct: float | None
    noise_floor_x: float | None
    baseline_probe_id: str
    candidate_probe_id: str
    baseline_sha: str
    candidate_sha: str
    reasons: tuple[str, ...]
    metric_deltas: dict[str, float]
```

`ParityContract.__post_init__` validates (raising `ClaimValidityError`): non-`None` `max_rel_err`, `max_abs_err`, `max_vjp_rel_err`, `min_corruption_rel_err` are finite and `>= 0.0`; non-`None` `min_work_active_count` is finite and `> 0.0`; `min_work_retention_ratio` is finite with `0.0 < min_work_retention_ratio <= 1.0`.

#### Public Function

```python
def compare_probe_pair(
    baseline: ProbeRecord,
    candidate: ProbeRecord,
    *,
    hold_fixed_config: tuple[str, ...],
    primary_metric: str = "total_step_seconds",
    lower_is_better: bool = True,
    parity: ParityContract | None = None,
    require_parity: bool = True,
    min_speedup: float = 1.03,
    noise_sigma: float = 2.0,
    max_vram_growth: float | None = 1.10,
    max_compile_increase: float = 0.0,
    max_dispatch_growth: float = 1.05,
    allow_dirty: bool = False,
) -> DeltaVerdict:
```

#### Exact Fail-Closed Evaluation Order (Decomposed Per-Step Helpers)

0. **Parameter Validation**: Raise `ClaimValidityError` if `(not require_parity and parity is not None)`, `min_speedup < 1.0`, `noise_sigma < 0.0`, `max_compile_increase < 0.0`, `max_dispatch_growth < 1.0`, or `(max_vram_growth is not None and max_vram_growth < 1.0)`.
1. **Contract & Environment Unanimity (excluding `git_sha`)**:
   - Require `contract_version` MAJOR match.
   - Require exact equality on: `platform`, `device_kind`, `x64_enabled`, `xla_flags`, `jax_version`, `jaxlib_version`, `n_atoms`, `num_replicas`, `replica_id`, `bucket_index`, `bucket_capacity`.
   - Stage floor & equality: require `stage >= 1` on all platforms (Stage-0 cost records never back execution comparisons), require `stage >= 2` when `platform == "gpu"`, and require `baseline.stage == candidate.stage`.
   - Diagnostic isolation: require `r.config.get("diagnostic_mode", "none") in ("", "none")` for **both** records (diagnostic `cmdbuf_off` records can NEVER be promoted or compared as execution baselines).
   - Step-time basis validation (when `primary_metric == "total_step_seconds"` OR `"step_time_basis"` is present in either record's `config`): resolve `base_basis = baseline.config["step_time_basis"] if "step_time_basis" in baseline.config else "batch"` (and symmetrically for `candidate`, so an explicit `""` is NOT masked as `"batch"`), require `base_basis in ("batch", "per_replica")`, `cand_basis in ("batch", "per_replica")`, and `base_basis == cand_basis`.
   - `hold_fixed_config`: require every key `k` in `hold_fixed_config` is present in **both** `baseline.config` and `candidate.config` and satisfies `baseline.config[k] == candidate.config[k]`.
   - Any failure -> `REJECT_UNVERIFIABLE` (`promotable=False`).
2. **Git Provenance Verification**:
   - Strict regex `_GIT_SHA_RE = re.compile(r"^[0-9a-f]{7,40}(-dirty)?$")` (matching `7–40` lowercase hex SHAs; rejecting `"unknown"`, `"-unverified"`, `"HEAD"`, `"local"`, empty strings).
   - Reject with `REJECT_UNVERIFIABLE` if either SHA fails `_GIT_SHA_RE.fullmatch(...)`.
   - If either SHA ends with `"-dirty"`:
     - If `allow_dirty=False` -> `REJECT_UNVERIFIABLE` (`promotable=False`).
     - If `allow_dirty=True` -> set `dirty_capped = True` and continue executing steps 3–6 so regressions still surface as `REJECT_*` / `REVERT_*`, while capping any otherwise-passing verdict at `ADVISORY_ONLY` (`promotable=False`).
3. **Fidelity, Non-Degeneracy & Work-Retention Gate** (runs before performance so dropped work can never register as a speedup):
   - Always run `_check_present_failure_witnesses(baseline, candidate)` first (even when `require_parity=False`), failing closed on any explicit defect:
     - Any key in `("non_degeneracy_status", "corruption_control_status", "export_guardrail_status", "vjp_parity_status")` present in `candidate.config` with value `"fail"` or an unrecognized value outside `("pass", "not_measured")` -> `REJECT_PARITY`.
     - `"output_l2_norm"` in `candidate.metrics` with `val <= 0.0` -> `REJECT_PARITY`.
     - `"work_active_count"` in `baseline.metrics` with `val <= 0.0` -> `REJECT_UNVERIFIABLE`.
     - `"work_active_count"` in `candidate.metrics` with `val <= 0.0` -> `REJECT_PARITY`.
     - `"work_active_count"` in both `baseline.metrics` (`> 0.0`) and `candidate.metrics` (`> 0.0`) with `cand_work / base_work < eff_min_work_retention_ratio` (using `ParityContract().min_work_retention_ratio = 0.99` when `require_parity=False`) -> `REJECT_PARITY`.
     - Any of `("parity_rel_err", "parity_max_abs_err", "parity_vjp_rel_err", "corruption_rel_err")` present in `candidate.metrics` with `val < 0.0` -> `REJECT_PARITY`.
   - If `require_parity=False`: set `parity_bypassed = True` (only waiving *absent / `"not_measured"`* parity evidence; caps any otherwise-passing verdict at `ADVISORY_ONLY`, `promotable=False`, with explicit advisory reason) and proceed to Step 4.
   - Otherwise (`eff_parity = parity if parity is not None else ParityContract()`):
     - **Numeric Non-Degeneracy** (when `eff_parity.require_non_degeneracy=True`): require `candidate.config.get("non_degeneracy_status") == "pass"` **AND** `"output_l2_norm"` in `candidate.metrics` with `val > 0.0`.
     - **Numeric Corruption Control**:
       - Note: `require_corruption_control` defaults to `True` on `ParityContract` to enforce negative-control hygiene against vacuous tolerance checks (`xtrax.testing.kernel_parity`). Callers without a corruption run may opt out explicitly via `ParityContract(require_corruption_control=False)` or CLI `--no-require-corruption-control`.
       - When `eff_parity.require_corruption_control=True`: require `candidate.config.get("corruption_control_status") == "pass"` **AND** EITHER (`"corruption_rel_err"` in `candidate.metrics` with `val > max(eff_parity.max_rel_err or 0.0, eff_parity.min_corruption_rel_err or 0.0)`) OR (`candidate.metrics.get("corruption_int_mismatches", 0.0) >= 1.0`).
       - When `eff_parity.require_corruption_control=False`: if `candidate.config.get("corruption_control_status")` is `"pass"` or `"corruption_rel_err"` is present, still enforce the threshold above; only absent / `"not_measured"` status is waived (recording `"corruption_control_not_required"` in `reasons`, still eligible for `KEEP` if numeric parity passes).
     - **Export Guardrail**: if `eff_parity.require_export_ok=True`, require `candidate.config.get("export_guardrail_status") == "pass"`, `candidate.metrics.get("export_custom_call_count") == 0.0`, and `candidate.metrics.get("export_artifact_bytes", 0.0) > 0.0`. (Any `"fail"` status is already caught fail-closed above.)
     - **Scalar Error & VJP Checks** (fail closed if required metric is missing):
       - `parity_rel_err` in `[0.0, eff_parity.max_rel_err]` when `eff_parity.max_rel_err is not None`
       - `parity_max_abs_err` in `[0.0, eff_parity.max_abs_err]` when `eff_parity.max_abs_err is not None`
       - `parity_vjp_rel_err` in `[0.0, eff_parity.max_vjp_rel_err]` when `eff_parity.max_vjp_rel_err is not None`; and whenever `"vjp_parity_status"` is present in `candidate.config`, require `"pass"`.
     - **Work-Retention Gate (`BUG-TRITON-ATOM1` Closure)**:
       - (Zero/negative baseline `work_active_count <= 0.0` already rejected as `REJECT_UNVERIFIABLE` above.)
       - If `"work_active_count"` in `baseline.metrics` (`> 0.0`): require `"work_active_count"` present in `candidate.metrics` (`> 0.0`) and `cand_work / base_work >= eff_parity.min_work_retention_ratio` (default `0.99`).
       - If `eff_parity.min_work_active_count is not None`: require `"work_active_count"` in `candidate.metrics` with `val >= eff_parity.min_work_active_count`.
     - **Custom Ceilings/Floors**: every key in `eff_parity.metric_ceilings` / `metric_floors` must exist on candidate and satisfy the bound.
     - Any parity failure -> `REJECT_PARITY` (`promotable=False`).
4. **Dispatch & Compile Gate (Strict, No Missing/Zero Bypass)**:
   - Require `"n_compilations"` present and `>= 0.0` in **both** records, and `"n_executions"` present and `> 0.0` in **both** records (if missing, negative, or `n_executions <= 0.0` -> `REJECT_UNVERIFIABLE`).
   - Require `cand_compiles <= max_compile_increase` (when `base_compiles == 0.0`), and if `base_compiles > 0.0`, forbid positive `cand_compiles` unless `max_compile_increase > 0.0 and cand_compiles <= base_compiles + max_compile_increase` (a broken baseline never excuses steady recompiles when `max_compile_increase == 0.0`) -> else `REJECT_DISPATCH`.
   - Require `cand_exec / base_exec <= max_dispatch_growth` -> else `REJECT_DISPATCH`.
5. **VRAM Gate**:
   - If `max_vram_growth is not None`:
     - On `platform == "gpu"`: require `"peak_vram_bytes"` present and `> 0.0` in **both** records (else `REJECT_UNVERIFIABLE`).
     - On `platform == "cpu"`: if present on only one record or `<= 0.0` -> `REJECT_UNVERIFIABLE`; if absent on both, skip VRAM check with advisory note.
     - When present in both: require `cand_vram / base_vram <= max_vram_growth` -> else `REJECT_VRAM`.
6. **Performance & Noise-Floor Gate**:
   - Require `primary_metric` present and `> 0.0` in both records (else `REJECT_UNVERIFIABLE`).
   - `speedup_x = base_val / cand_val` if `lower_is_better` else `cand_val / base_val`; `rel_delta_pct = (cand_val - base_val) / base_val * 100.0`.
   - Dispersion & relative noise floor (strictly scoped to `primary_metric == "total_step_seconds"`):
     - Validate any `"wall_mad_seconds"` or `"wall_iqr_seconds"` present on either record is `>= 0.0` (negative -> `REJECT_UNVERIFIABLE`).
     - When `primary_metric == "total_step_seconds"`:
       - If `"wall_mad_seconds"` in both and `(base_mad > 0.0 or cand_mad > 0.0)`:
         $$\text{rel\_noise} = \text{noise\_sigma} \cdot 1.4826 \cdot \sqrt{\left(\frac{\text{MAD}_{\text{base}}}{t_{\text{base}}}\right)^2 + \left(\frac{\text{MAD}_{\text{cand}}}{t_{\text{cand}}}\right)^2}$$
       - Else if `"wall_iqr_seconds"` in both and `(base_iqr > 0.0 or cand_iqr > 0.0)`:
         $$\text{rel\_noise} = \text{noise\_sigma} \cdot 0.7413 \cdot \sqrt{\left(\frac{\text{IQR}_{\text{base}}}{t_{\text{base}}}\right)^2 + \left(\frac{\text{IQR}_{\text{cand}}}{t_{\text{cand}}}\right)^2}$$
       - Else if `"wall_mad_seconds"` in both (both `0.0`): `rel_noise = 0.0`.
       - Else: `rel_noise = 0.0` (append advisory reason noting fallback to `min_speedup` floor).
     - When `primary_metric != "total_step_seconds"`: `rel_noise = 0.0` (wall dispersion is never divided by non-wall quantities like VRAM bytes).
   - `required_speedup = max(min_speedup, 1.0 + rel_noise)`.
   - If `speedup_x < max(0.0, 1.0 - rel_noise)` -> `REVERT_REGRESSION` (`promotable=False`).
   - Else if `speedup_x < required_speedup` -> `REVERT_NOISE` (`promotable=False`).
   - Else if `dirty_capped or parity_bypassed` -> `ADVISORY_ONLY` (`promotable=False`).
   - Else -> `KEEP` (`promotable=True`).

- **`DeltaVerdict.metric_deltas`**: `{k: float(candidate.metrics[k] - baseline.metrics[k]) for k in sorted(set(baseline.metrics) & set(candidate.metrics)) if math.isfinite(candidate.metrics[k] - baseline.metrics[k])}` — signed candidate-minus-baseline differences in native units across all shared finite metrics (populated on all verdicts, including pre-Step-6 early rejections, for auditability).

#### Baseline Ratchet Helper & Record Hygiene

```python
def update_baseline_pointer(
    pointer_path: str | Path,
    verdict: DeltaVerdict,
    candidate_record_path: str | Path,
    *,
    rejected_dir: str | Path | None = None,
    initialize: bool = False,
) -> Path:
```
- Reads `cand_rec = ProbeRecord.read(candidate_record_path)` and asserts `cand_rec.probe_id == verdict.candidate_probe_id` and `_shas_match(cand_rec.git_sha, verdict.candidate_sha)` (`ClaimValidityError` on mismatch).
- Note on `_shas_match(sha_a, sha_b)`: compares stripped lowercase hex parts; returns `True` iff equal OR (both clean/both dirty identically and one is a `>=7`-char prefix of a 40-char SHA).
- **Ratchet Chain Continuity**: when `verdict.status is VerdictStatus.KEEP and verdict.promotable`:
  - If `pointer_path.exists()` and `not initialize`: reads existing pointer JSON, requires `verdict.baseline_probe_id == prev["probe_id"]` and `_shas_match(verdict.baseline_sha, prev["git_sha"])`, raising `ClaimValidityError` on chain mismatch (prevents advancing a pointer using an unrelated/stale baseline!).
  - Atomically writes pointer JSON (`baseline_record`, `git_sha`, `probe_id`, `updated_at`, `speedup_x`, `previous_sha`, `previous_probe_id`) via temp file + `os.replace`.
- When NOT promotable: raises `ClaimValidityError` if `rejected_dir is None`; otherwise leaves `pointer_path` untouched and writes a collision-safe verdict sidecar `<slug>_<sha7>_<hash8>.verdict.json` into `rejected_dir` (recommended default: `outputs/profiling/rejected/`).
- **Discovery Isolation (`report.py` & `claims.py`)**:
  - Update `src/xtrax/profiling/report.py:34, 68-84`: split skip suffixes so directory globs ignore diagnostic/verdict artifacts while explicit `render_report([diag_path])` still renders `.diag.json` traces:
    ```python
    _DISCOVERY_SKIP_SUFFIXES = ("_summary.json", ".verdict.json", ".diag.json")
    _EXPLICIT_SKIP_SUFFIXES = ("_summary.json", ".verdict.json")
    _SKIP_SUFFIXES = _DISCOVERY_SKIP_SUFFIXES  # preserve module-level constant
    ```
    In `discover_records` (`report.py:68-83`), track `(path, from_glob: bool)` (`from_glob=True` for `base.glob("stage*/*.json")` and `p.glob("*.json")` when `p.is_dir()`, `from_glob=False` for explicit file paths) and check `path.name.endswith(_DISCOVERY_SKIP_SUFFIXES if from_glob else _EXPLICIT_SKIP_SUFFIXES)`.
  - Update `src/xtrax/profiling/claims.py:270-294`: `assert_claim_supported(..., ClaimClass.END_TO_END)` rejects any source whose `config.get("diagnostic_mode", "none") not in ("", "none")` with `ClaimValidityError`.

---

### 3.2 `src/xtrax/profiling/unit_probe.py` & `kernel_probe.py` — Multi-Pass GPU Probe, VRAM & PTX/`ptxas` Telemetry

#### 3.2.1 `UnitProbeResult` Backward-Compatible Dispersion & VRAM Extension (`unit_probe.py`)
Append **three** defaulted fields strictly **after** `dispatch_counts` (`src/xtrax/profiling/unit_probe.py:89`):
```python
dispatch_counts: dict[str, int] = field(default_factory=dict)
wall_mad_seconds: float = 0.0
wall_iqr_seconds: float = 0.0
peak_vram_bytes: int | None = None
```
- **Dispersion**: `wall_mad_seconds = median(|w_i - median(w)|)` for `n_wall >= 2` else `0.0`. `wall_iqr_seconds = max(0.0, q3 - q1)` via `statistics.quantiles(walls, n=4)` for `n_wall >= 4` else `0.0` (never substituting `max - min` range into an IQR field).
- **Device-Accurate `peak_vram_bytes`**: inspect execution device(s) of the final `call()` output pytree (falling back to inputs, then `jax.devices()[:1]`) where `platform == "gpu"`; query `dev.memory_stats()` assigning `stats = None` on exception (compliant with `ruff S110`), taking `max` over `stats.get("peak_bytes_in_use")`. Document explicitly that XLA `peak_bytes_in_use` is the process-lifetime device high-water mark (exact per-unit isolation requires subprocess isolation).
- **`UnitProbeResult.as_metrics()`**: adds `"wall_mad_seconds": float(self.wall_mad_seconds)`, `"wall_iqr_seconds": float(self.wall_iqr_seconds)`, and `"peak_vram_bytes": float(self.peak_vram_bytes)` when non-`None`. Update `tests/profiling/test_unit_probe.py:175-183` accordingly.

#### 3.2.2 PTX, `ptxas`, & Triton Kernel Telemetry Parsers (`kernel_probe.py`)
Pure-stdlib parsers over dumped `.ptx` files, `ptxas --verbose` stderr, and Triton cache metadata `.json` files (zero top-level `jax` import; zero GPU requirement; testable hermetically on fixtures):

```python
@dataclasses.dataclass(frozen=True, slots=True)
class PtxKernelStats:
    kernel_name: str
    vregs_by_type: dict[str, int]
    vregs_total_32bit_equiv: int
    f64_vregs: int
    ptx_local_bytes: int
    ptx_shared_bytes: int
    has_dynamic_extern_shared: bool
    ptx_file: str
    phys_regs: int | None = None
    spill_store_bytes: int | None = None
    spill_load_bytes: int | None = None
    stack_frame_bytes: int | None = None
    ptxas_smem_bytes: int | None = None
    triton_spills: int | None = None
```
- **`parse_ptx_text(ptx_text: str, *, ptx_file: str = "<memory>") -> list[PtxKernelStats]`**:
  - Strips `//` and `/* ... */` comments.
  - Scans `(?:^|\n)\s*(?:(?:\.visible|\.weak)\s+)?\.entry\s+([A-Za-z_.$][\w.$]*)\s*\(` with balanced `{ ... }` body extraction.
  - Parses vectorized `.reg .<type> %name<N>;` AND scalar `.reg .<type> %r0, %r1;` across all PTX register types (`.pred`, `.b8/.b16/.s8/.u8/.s16/.u16/.f16/.bf16`, `.b32/.s32/.u32/.f32`, `.b64/.s64/.u64/.f64`, `.b128`), clearly treating `.reg` counts as **virtual SSA declarations** (`vregs`), not physical hardware registers.
  - Computes `ptx_local_bytes` and `ptx_shared_bytes` across typed byte sizes + optional `[count]`; detects unsized `.extern .shared ... []` (`has_dynamic_extern_shared=True`).
- **`parse_ptxas_verbose(stderr_text: str) -> dict[str, PtxasKernelStats]`**: parses `ptxas info : Compiling entry function '<name>' for 'sm_...'` + `Used <R> registers, <S> bytes smem` / `Used <R> registers, <B> bytes stack frame, <S1> bytes spill stores, <S2> bytes spill loads` from `ptxas --verbose`.
- **`parse_triton_metadata_dir(cache_dir: str | Path) -> dict[str, TritonKernelMeta]`**: reads Triton `.json` cache sidecars (`n_regs`, `n_spills`, `shared`, `name`). Keeps `triton_spills` (`int`) strictly separate from `ptxas` exact `spill_store_bytes`/`spill_load_bytes` (only populating `phys_regs` from `n_regs` when `ptxas` did not run or failed).
- **`summarize_ptx_metrics(stats: list[PtxKernelStats]) -> dict[str, float]`**: emits `ptx_kernel_count`, `ptx_vregs_max`, `ptx_f64_vregs_max`, `ptx_lmem_bytes_max`, `ptx_lmem_bytes_total`, `ptx_smem_bytes_max`, plus when `ptxas` physical stats are present: `ptxas_regs_max`, `ptxas_spill_bytes_max` (and when Triton `triton_spills` / `phys_regs` are present: `ptxas_regs_max` if not yet set and `triton_spills_max`).

#### 3.2.3 Safe `XLA_FLAGS` Merging, `ptxas` Invocation, & Multi-Pass GPU Probe (`kernel_probe.py`, `_diag_worker.py`, `trace.py`)
1. **`merge_xla_flags(base_flags: str, overrides: dict[str, str]) -> str`**:
   - Tokenizes via `shlex.split(base_flags)`.
   - Removes any existing `--flag=val` or two-token `--flag val` matching keys in `overrides` or transient dump flags (`--xla_dump_to`, `--xla_dump_hlo_as_text`).
   - Appends deterministic `f"{k}={v}"` overrides and rejoins via `shlex.join`.
2. **`trace.py:594-631` fix**: extend `hlo_text_for(jitted: Any, *args: Any, **kwargs: Any) -> str` to accept `**kwargs` and pass through to `.lower(*args, **kwargs)`.
3. **Import & Isolation Contract (`kernel_probe.py` & `_diag_worker.py`)**:
   - **BOTH** `kernel_probe.py` and `_diag_worker.py` MUST have zero top-level `import jax` and zero `xtrax.cli` imports (strictly stdlib + `xtrax.profiling.*` at module scope; `_diag_worker.py` resolves `fn_spec`/`inputs_spec` purely via `importlib` matching `cli/loader.py`).
   - `_diag_worker.py` is a **stateless measurement worker** supporting `--mode {prod,diag,ptx}` and `--result-json <tmp_path>`; it NEVER emits `ProbeRecord`s to disk directly (avoiding XLA C++ stdout noise and preventing Pass B/Pass C write races).
4. **Deterministic 4-Stage Execution (`probe_kernel_dual_pass`)**:
   - **Stage 1 — Pass A (Production, `cmdbuf=on`)**:
     - When `fn_spec` + `inputs_spec` are given and `isolate_passes=True` on GPU, Pass A runs first inside `_diag_worker --mode prod` subprocess (`XLA_FLAGS` merged via `merge_xla_flags` stripping transient dumps, no command-buffer override) so GPU peak VRAM and timing have zero contamination from prior compilations and exits to release 100% VRAM before Pass B starts. Otherwise runs `probe_unit` in-process (passing child subprocesses `XLA_PYTHON_CLIENT_PREALLOCATE=false` when parent already holds a GPU context).
     - Resolves `diag_skipped_reason` (`"diag_disabled"` if `run_diag=False`, `"no_importable_spec"` if `fn_spec is None`) **before** emitting frozen `prod_record` (`diagnostic_mode = "none"`, `step_time_basis = "batch"`). Never mutates `prod_record` after construction.
   - **Stage 2 — Pass B (`_diag_worker --mode diag`, `cmdbuf=off` subprocess)**:
     - Runs `_diag_worker --mode diag --result-json <tmp>/pass_b.json` with `env["XLA_FLAGS"] = merge_xla_flags(base_xla_flags, {"--xla_gpu_enable_command_buffer": ""})` (clean, deterministic `xla_flags` — zero random temp paths inside `diag_record.xla_flags`!).
     - Captures Perfetto trace across `n_steady` calls; resolves HLO text cleanly via `_resolve_pass_b_hlo_text(fn, args, kwargs)` (`hlo_text_for` if `.lower()` is callable catching `(TypeError, ValueError, RuntimeError)`, else `hlo_text_for(jax.jit(fn), *args, **kwargs)` catching broad `Exception` `# noqa: BLE001`, returning `None` cleanly for unjittable host wrappers while stamping `hlo_attribution_skipped = "unjittable_wrapper"`). Falls back gracefully to `parse_hlo_op_times(events)` (`attribution_method[op] = "op_name"`).
     - Normalizes every scope's `exclusive_seconds` by `1.0 / n_steady` so `scopes` match per-unit `total_step_seconds` scale (`report.py` `pct_of_total` accuracy!). Records `"named_scope"` for matched `known_labels` and `"op_name"` for custom-call `triton:<kernel>` / unmapped device ops (`>= 1%` share), strictly satisfying `record.py:358`.
     - Writes raw telemetry (`scopes`, `attribution_method`, `metrics`, executed `triton:<kernel>` names, `compute_capability`, exact Pass B provenance fields) to `<tmp>/pass_b.json` and exits to release 100% GPU VRAM.
     - If Pass B exits non-zero, returns `DualPassProbeResult(..., diag_record=None, diag_skipped_reason=f"diag_subprocess_failed:exit_{rc}")` without modifying `prod_record`.
   - **Stage 3 — Pass C (Optional Compiler PTX / `ptxas` / Triton Subprocess when `run_diag=True and capture_ptx=True` & Pass B succeeded)**:
     - Inside a parent-owned `tempfile.TemporaryDirectory()` (`triton_cache/`, `triton_dump/`, `jax_cache/`, `xla_dump/`), spawns `_diag_worker --mode ptx --result-json <tmp>/pass_c.json` executing **1 cold compile+step** with:
       - `TRITON_CACHE_DIR=<tmp>/triton_cache`, `TRITON_KERNEL_DUMP=1`, `TRITON_DUMP_DIR=<tmp>/triton_dump`, `JAX_COMPILATION_CACHE_DIR=<tmp>/jax_cache`, and
       - `XLA_FLAGS=merge_xla_flags(base_xla_flags, {"--xla_gpu_enable_command_buffer": "", "--xla_dump_to": "<tmp>/xla_dump", "--xla_dump_hlo_as_text": "false"})`.
     - After worker exit (inside the `TemporaryDirectory()` context), parent runs `collect_pass_c_ptx_stats(...)` before deleting the temp directory:
       1. **Triton autotune filtering**: walk `triton_dump` + `triton_cache`; select `.ptx` files with sibling `<stem>.cubin` (winner artifacts) + `<stem>.json` (`n_regs`, `n_spills`, `shared`), discarding unbacked trial `.ptx` files whenever `.cubin`-backed winners exist; also scan `xla_dump/*.ptx` for XLA fusion kernels.
       2. **Executed-kernel filtering & deduplication**: when Pass B observed `triton:<name>` / kernel names, filter matched names if `>= 1` matches (else keep all deduplicated entries and tag `ptx_filter_status="unfiltered_name_mismatch"`). Deduplicate globally by `(kernel_name, sha256(normalized_entry_body))` preferring Triton `.cubin`-backed entries over XLA dumps, recording collapsed duplicate count in `ptx_dedup_count`.
       3. **`ptxas --verbose` invocation**: `find_ptxas()` checks `shutil.which("ptxas")`, `$CUDA_HOME/bin/ptxas`, `$CUDA_PATH/bin/ptxas`, `/usr/local/cuda/bin/ptxas`, and `nvidia.cuda_nvcc` `bin/ptxas`. Maps `compute_capability` via `re.fullmatch(r"\s*(\d+)\.(\d+)\s*", cc)` -> `sm_{int(M)}{int(m)}` (e.g. `"8.9"` -> `"sm_89"`, `"12.0"` -> `"sm_120"`). Invokes `subprocess.run([ptxas_bin, "--verbose", f"--gpu-name={arch}", "-o", os.devnull, str(ptx_path)], capture_output=True, text=True, timeout=30, check=False)` and parses `stderr + "\n" + stdout` via `parse_ptxas_verbose`. On missing `ptxas` or non-zero exit code, record warning on `DualPassProbeResult.ptxas_warnings` and fall back cleanly (`ptx_spill_source = "triton_meta"` if Triton `.json` parsed else `"ptx_only"`).
     - If Pass C fails or dumps zero `.ptx` files, Pass B telemetry is preserved (`ptx_capture_status = f"skipped:{reason}"`, `ptx_spill_source = "none"`).
   - **Stage 4 — Single Immutable `diag_record` Emission**:
     - Parent merges Pass B metrics/scopes + Pass C `summarize_ptx_metrics(ptx_stats)` + Pass B's clean provenance (`x64_enabled`, `device_kind`, `jax_version`, `jaxlib_version`, Pass B's clean `xla_flags` without `--xla_dump_to`, `git_sha`) + config tags (`diagnostic_mode="cmdbuf_off"`, `paired_prod_probe_id=prod_record.probe_id`, `diag_n_steady=str(n_steady)`, `ptx_capture_status`, `ptx_spill_source`, `ptx_dedup_count`, optional `ptx_filter_status`/`hlo_attribution_skipped`) and calls `emit_probe_record(...)` ONCE atomically to `<diag_dir>/<prod_record.probe_id>.diag.json` (default `outputs/profiling/diagnostics/`, outside `outputs/profiling/stage*/`).

#### 3.2.4 Precision Ratio Helper (`compute_precision_ratio`)
```python
def compute_precision_ratio(
    single_record: ProbeRecord,
    mixed_record: ProbeRecord,
    *,
    metric: str = "total_step_seconds",
    hold_fixed_config: tuple[str, ...] = (),
) -> float:
```
- Raises `ClaimValidityError` unless:
  - `contract_version` MAJOR match, `stage >= 1` (`>= 2` on GPU), `single_record.stage == mixed_record.stage`, and both `diagnostic_mode in ("", "none")`.
  - `_reject_unverifiable_git_sha` holds for both and `_shas_match(single_record.git_sha, mixed_record.git_sha)`.
  - Unanimity on `platform`, `device_kind`, `xla_flags`, `jax_version`, `jaxlib_version`, `n_atoms`, `num_replicas`, `replica_id`, `bucket_index`, `bucket_capacity`, `step_time_basis`, and `hold_fixed_config`.
  - **`mixed_record.x64_enabled is True`** (rejects silent FP32 truncation in `mixed`).
  - `single_record.x64_enabled` may be `False` or `True`, EXCEPT when `single_record.x64_enabled is True` (Prolix shared-process `jax_enable_x64=True` pattern): both records MUST carry `config["precision"] == "single"` and `"mixed"` respectively. Whenever `"precision"` is set on either record, enforce `"single"` vs `"mixed"`.
  - `single_val > 0.0` and `mixed_val > 0.0` (finite); returns `mixed_val / single_val`.

---

### 3.3 `src/xtrax/profiling/triage.py` — Deterministic Bottleneck Classifier

Purpose: Convert `(prod_record, diag_record, precision_pair, batch_scaling_pair)` into a deterministically ranked list of `BottleneckFinding`s. Zero `jax` imports; direct top-level import of `ProbeRecord`.

```python
class BottleneckCategory(enum.Enum):
    RECOMPILE_DRIFT = "recompile_drift"
    HOST_BOUNDARY_SYNC = "host_boundary_sync"
    FP64_PIPE_SATURATION = "fp64_pipe_saturation"
    SPATIAL_DEAD_WORK = "spatial_dead_work"
    REGISTER_SPILL_PRESSURE = "register_spill_pressure"
    ATOMIC_CONTENTION = "atomic_contention"

@dataclasses.dataclass(frozen=True, slots=True)
class TriageThresholds:
    exec_per_element_flag: float = 0.5                     # calibrated in rules.py
    host_share_flag: float = 0.25                          # provisional_gpu_heuristic
    mixed_over_single_flag: float = 1.35                   # provisional_gpu_heuristic
    top_kernel_pct_flag: float = 30.0                      # provisional_gpu_heuristic
    active_work_ratio_flag: float = 0.25                   # provisional_gpu_heuristic
    ptx_regs_flag: int = 160                               # physical ptxas/Triton regs heuristic
    ptx_vregs_flag: int = 256                              # fallback virtual PTX SSA regs heuristic
    ptx_lmem_bytes_flag: int = 1                           # any spill / local bytes > 0
    batch_scaling_eff_flag: float = 0.50                   # per-replica eta_rep threshold
    ctas_per_atomic_target_flag: float = 64.0              # provisional_gpu_heuristic
    low_fp64_device_substrings: tuple[str, ...] = (
        "l40", "l4", "rtx", "geforce", "4090", "5090", "rtx_pro_6000",
    )

@dataclasses.dataclass(frozen=True, slots=True)
class BottleneckFinding:
    category: BottleneckCategory
    severity: str                          # "blocking" | "high" | "medium"
    target_kernel: str | None
    evidence: dict[str, float | str]
    summary: str
    playbook_ref: str
    threshold_status: str                  # "calibrated" | "provisional_gpu_heuristic"
```

#### Exact Triage Rules

1. **`RECOMPILE_DRIFT` (`severity="blocking"`, `calibrated`)**: `prod.metrics.get("n_compilations", 0.0) > 0.0`.
   `playbook_ref`: `"agent_assets/skills/xtrax-optimizing/SKILL.md#remediation-playbook-keyed-by-xtrax-profile-flags"`
2. **`HOST_BOUNDARY_SYNC` (`severity="high"`)**: `exec_per_element >= exec_per_element_flag` (`calibrated`) OR (`total_step_seconds > 0.0` and `host_seconds / total_step_seconds >= host_share_flag` (`provisional_gpu_heuristic`)).
   `playbook_ref`: `"agent_assets/skills/xtrax-optimizing/references/tier1-host-boundary.md#decision-rules-verify-paths"`
3. **`FP64_PIPE_SATURATION` (`severity="high"`, `provisional_gpu_heuristic`)**: when `precision_pair is not None` or `"mixed_over_single_ratio"` is present in `prod.metrics`, resolve `mixed_over_single_ratio` via `compute_precision_ratio(*precision_pair)` (else `prod.metrics["mixed_over_single_ratio"]`), requiring finite `> 0.0` (raising `ClaimValidityError` if invalid); if neither is supplied, Rule 3 does not trigger. Compute `ptx_f64_vregs_max = max((r.metrics.get("ptx_f64_vregs_max", 0.0) for r in (prod, diag) if r is not None), default=0.0)` across both `prod` and `diag`. Fires when `mixed_over_single_ratio >= mixed_over_single_flag` AND (`device_kind` matches `low_fp64_device_substrings` OR `ptx_f64_vregs_max > 0.0`).
   `playbook_ref`: `"agent_assets/skills/xtrax-optimizing/references/kernel-optimization-loop.md#fp64-pipe-saturation"`
4. **`SPATIAL_DEAD_WORK` (`severity="high"`, `provisional_gpu_heuristic`)**: `active_work_ratio` present in `0.0 <= r < active_work_ratio_flag` AND (if `diag` has `>= 1` non-None scope, top kernel share `>= top_kernel_pct_flag`%).
   `playbook_ref`: `"agent_assets/skills/xtrax-optimizing/references/tier3-pallas-megakernels-and-ragged-buckets.md#1-handling-ragged-axes-n_atoms-n_waters-and-ghost-atom-compaction"`
5. **`REGISTER_SPILL_PRESSURE` (`severity="high"`, `provisional_gpu_heuristic`)**: checked across both `diag` and `prod`: `ptxas_spill_bytes_max >= ptx_lmem_bytes_flag` OR `triton_spills_max >= ptx_lmem_bytes_flag` OR `ptx_lmem_bytes_max >= ptx_lmem_bytes_flag` OR `ptxas_regs_max >= ptx_regs_flag` OR (`"ptxas_regs_max"` absent and `ptx_vregs_max >= ptx_vregs_flag`).
   `playbook_ref`: `"agent_assets/skills/xtrax-optimizing/references/kernel-optimization-loop.md#register-spill-pressure"`
6. **`ATOMIC_CONTENTION` (`severity="high"`, `provisional_gpu_heuristic`)**:
   - When `batch_scaling_pair = (lo_rec, hi_rec)` is provided, raise `ClaimValidityError` unless ALL hold:
     - `contract_version` MAJOR match across `(lo_rec, hi_rec)`;
     - `stage >= 1` (`>= 2` when `platform == "gpu"`) and `lo_rec.stage == hi_rec.stage`;
     - `_reject_unverifiable_git_sha([lo_rec, hi_rec], ClaimClass.END_TO_END)` AND `_shas_match(lo_rec.git_sha, hi_rec.git_sha)`;
     - Unanimity on `(platform, device_kind, x64_enabled, xla_flags, jax_version, jaxlib_version, n_atoms, bucket_capacity)` and both `diagnostic_mode in ("", "none")`;
     - `1 <= lo_rec.num_replicas < hi_rec.num_replicas`;
     - Matching `basis` in `("batch", "per_replica")` (empty `""` invalid);
     - `"total_step_seconds"` present, finite, and `> 0.0` in **both** `lo_rec.metrics` and `hi_rec.metrics`.
   - Compute per-replica latency `t_rep(B) = val / B` (if `basis == "batch"`) else `val` (if `basis == "per_replica"`), and **per-replica scaling efficiency**:
     $$\eta_{\text{rep}} = \frac{t_{\text{rep}}(B_{\text{lo}})}{t_{\text{rep}}(B_{\text{hi}})}$$
     (Concurrent scaling yields $\eta_{\text{rep}} > 1.0$; ideal saturated scaling yields $\eta_{\text{rep}} = 1.0$; contention serialization degrading per-replica latency $> 2\times$ yields $\eta_{\text{rep}} < 0.50$.)
   - Fires when (`ctas_per_atomic_target >= ctas_per_atomic_target_flag`) OR (`eta_rep < batch_scaling_eff_flag`).
   - `playbook_ref`: `"agent_assets/skills/xtrax-optimizing/references/kernel-optimization-loop.md#atomic-contention"`

---

### 3.4 Optional `bathos` Integration (Zero `bathos` Imports in `src/xtrax/`)

1. **`src/xtrax/profiling/emitters.py`** (leaf-clean; zero `bathos` import; zero `mcp__bathos`):
   - Append optional kwargs strictly after existing params in `emit_probe_record(...)`: `bathos_run_id: str | None = None`, `bathos_campaign_id: str | None = None`, `capture_bathos_env: bool = True`, `num_replicas: int = 1`, `replica_id: int | None = None`, `bucket_index: int | None = None`, `bucket_capacity: int | None = None`, `x64_enabled: bool | None = None`, `device_kind: str | None = None`, `jax_version: str | None = None`, `jaxlib_version: str | None = None`, `xla_flags: str | None = None` (the final 5 provenance overrides delegate to `ProbeRecord`'s `default_factory` when `None`, allowing Pass B subprocess provenance to populate `diag_record` accurately without inheriting parent `cmdbuf=on` `xla_flags`).
   - Resolve `bathos_run_id` / `bathos_campaign_id` from explicit kwarg or `os.environ.get("BTH_RUN_ID")` / `os.environ.get("BTH_CAMPAIGN_ID")` (non-empty only) into `config`.
   - Expose `probe_record_to_bathos_metrics(record: ProbeRecord, verdict: DeltaVerdict | None = None) -> dict[str, float | str]`.
2. **`controller/profiling_bathos.py`** (under unshipped `controller/` tree, covered by `tier4_controller` + `just audit-controller-types`):
   - `resolve_bathos_provenance(run_id: str, *, catalog_dir: str = "") -> dict[str, Any]`: lazily imports `bathos.config` and `bathos.query` inside the function body and queries `bathos.query.get_run`; raises clear `ImportError("Install xtrax[controller] to query bathos catalog")` if `bathos` is unavailable.

---

### 3.5 `src/xtrax/testing/kernel_parity.py` Bridge

Append structured telemetry helpers (preserving `ParityResult` and `VjpParityResult` shapes intact; zero `xtrax.profiling` import):
```python
@dataclass(frozen=True, slots=True)
class ProbeParityPayload:
    metrics: dict[str, float]
    config_statuses: dict[str, str]

def build_probe_parity_payload(
    forward: ParityResult,
    got_output: Any,
    reference_output: Any,
    *,
    rel: float = 1e-4,
    vjp_result: VjpParityResult | None = None,
    corrupted_output: Any | None = None,
    corruption_result: ParityResult | None = None,
) -> ProbeParityPayload:

def evaluate_kernel_parity_for_probe(
    kernel: Callable[..., Any],
    reference: Callable[..., Any],
    args: tuple[Any, ...],
    corrupt: Callable[[Callable[..., Any]], Callable[..., Any]],
    *,
    rel: float = 1e-4,
    vjp_kernel: VjpKernel | None = None,
    vjp_key: Any | None = None,
) -> ProbeParityPayload:
```
- **`build_probe_parity_payload` Exact Contract**:
  - Reject with `ValueError` if **both** `corrupted_output is not None` and `corruption_result is not None`.
  - Flatten `(got_output, reference_output)` via `jax.tree_util.tree_flatten`; check `structure_ok = (got_def == ref_def)` and leaf shape match.
  - **`output_l2_norm`**: float64 L2 norm across numeric/bool leaves of `got_output`.
  - **`parity_rel_err`** (aligned with `scale_relative_atol` at `kernel_parity.py:95-112`): `float("inf")` if not `structure_ok` or `forward.int_mismatches > 0`; else `max` over inexact leaves (`_is_inexact`) of `max(|g - w|) / max(1.0, max(|w|))` (`0.0` for empty leaves; `float("inf")` if any non-finite element exists; `0.0` if all leaves are integral/bool and `forward.int_mismatches == 0`).
  - **`non_degeneracy_status`**: `"pass"` iff `forward.ok and forward.int_mismatches == 0 and structure_ok and math.isfinite(forward.max_abs_err) and math.isfinite(parity_rel_err) and math.isfinite(output_l2_norm) and output_l2_norm > 0.0`, else `"fail"`.
  - **`vjp_result`** (when provided): `config_statuses["vjp_parity_status"] = "pass" if vjp_result.ok else "fail"`; `vjp_err = max(non-None rel_l2, default=0.0)` (if non-finite, force `vjp_parity_status = "fail"`).
  - **Corruption Control**:
    - If `corrupted_output is not None`: run `corr_res = compare_outputs(corrupted_output, reference_output, rel=rel)` and `corr_rel_err` via the same PyTree rel-err helper.
    - Else if `corruption_result is not None` (lets callers reuse `require_corruption_red` outputs): set `corr_res = corruption_result` and `corr_rel_err = float(corr_res.max_abs_err) / max(1.0, _tree_max_abs(reference_output))`.
    - When either corruption input is given: set `config_statuses["corruption_control_status"] = "pass"` iff `(not corr_res.ok) and ((math.isfinite(corr_rel_err) and corr_rel_err > rel) or corr_res.int_mismatches > 0)`, else `"fail"`; emit `"corruption_int_mismatches"` when `corr_res.int_mismatches > 0`.
    - When neither is given: explicitly set `config_statuses["corruption_control_status"] = "not_measured"` (never `"pass"`, and omit `"corruption_rel_err"`).
  - **`ProbeRecord.__post_init__` Finiteness Guard (`record.py:297-301`)**: `ProbeRecord` rejects `NaN`/`Inf` in `metrics`. Therefore ONLY insert `"parity_max_abs_err"`, `"parity_rel_err"`, `"output_l2_norm"`, `"parity_vjp_rel_err"`, `"corruption_rel_err"` into `payload.metrics` when `math.isfinite(val)` is `True`! Omit non-finite values, force the corresponding `*_status = "fail"`, and record `config_statuses["parity_nonfinite_metrics"] = ",".join(bad_keys)` so NaN/Inf divergence emits an auditable record that cleanly triggers `REJECT_PARITY` instead of crashing before persistence.
- **`evaluate_kernel_parity_for_probe`**: computes `got`, `ref`, `fwd = compare_outputs(got, ref, rel=rel)`, `corr = compare_outputs(corrupt(kernel)(*args), ref, rel=rel)`, and optional `vjp_parity(vjp_kernel, reference, args, key=vjp_key, rtol=rel, rel=rel)` (raises `ValueError` if `vjp_kernel is not None and vjp_key is None`), returning a complete `ProbeParityPayload` that passes `ParityContract(require_corruption_control=True)` out of the box when healthy and returns a structured `"fail"` payload on regression.

---

### 3.6 CLI Extensions (`src/xtrax/cli/profile_verb.py`)

Append all new defaulted fields **strictly after `sha: str | None = None`** (`src/xtrax/cli/profile_verb.py:125`) on `ProfileArgs`, using plain `bool` defaults (NEVER `bool | None = None`, which `tyro` renders as `--flag {None,True,False}` rather than clean `--flag` / `--no-flag` switches):
```python
compare: tuple[str, str] | None = None
triage: str | None = None
diag: str | None = None
precision_baseline: str | None = None
primary_metric: str | None = None
require_parity: bool = True
require_corruption_control: bool = True
max_rel_err: float | None = None
max_vjp_rel_err: float | None = None
min_work_retention_ratio: float | None = None
allow_dirty: bool = False
min_speedup: float | None = None
hold_fixed: str | None = None
bypass_hold_fixed: bool = False
ratchet_pointer: str | None = None
rejected_dir: str | None = None
from_triage_probe: str | None = None
triage_category: str | None = None
target_missed: bool = False
telemetry_summary: bool = False
telemetry_log_dir: str | None = None
```

- **CLI `--hold-fixed` / `--bypass-hold-fixed` Policy**:
  - Define provenance/telemetry/status exemptions (`_EXEMPT_DIFF_CONFIG_KEYS`, deliberately NOT exempting `"fn"` or user workload knobs):
    `frozenset({"probe_id", "bathos_run_id", "bathos_campaign_id", "paired_prod_probe_id", "diagnostic_mode", "diag_skipped_reason", "diag_n_steady", "ptx_capture_status", "ptx_spill_source", "ptx_dedup_count", "ptx_filter_status", "hlo_attribution_skipped", "step_time_basis", "non_degeneracy_status", "corruption_control_status", "export_guardrail_status", "vjp_parity_status", "parity_nonfinite_metrics"})`.
  - When `--hold-fixed KEY1,KEY2` is passed: split on `","`, strip whitespace; if blank or containing any empty token, fail `_not_measured(...)` exit `2`. Use `hold_fixed_config = tuple(tokens)`.
  - When `--bypass-hold-fixed` is passed: pass `hold_fixed_config = ()` AND force CLI verdict cap at `ADVISORY_ONLY` (`promotable=False`, reason `"hold_fixed_bypassed"`, exit `1`).
  - When **both omitted**: infer `hold_fixed_config = tuple(sorted((set(baseline.config) | set(candidate.config)) - _EXEMPT_DIFF_CONFIG_KEYS))` so workload identity (`fn`, `shapes`, `precision`, etc.) and all non-exempt config keys must match cleanly (`REJECT_UNVERIFIABLE` exit `2` on drift).
- **`--compare` to `ParityContract` & Shadow Feedback Wiring**:
  - When `args.require_parity` is `False`, pass `parity=None` into `compare_probe_pair(...)` so Step 0's `(not require_parity and parity is not None)` invariant is respected. Otherwise construct `ParityContract(...)` from `max_rel_err`, `max_vjp_rel_err`, `require_corruption_control`, and `min_work_retention_ratio`.
  - When `--from-triage-probe` + `--triage-category` are provided: resolve `subject_key = resolve_triage_subject_key(args.from_triage_probe)` and validate normalized category (`raw.strip().removeprefix("xtrax.triage.")` in `PROVISIONAL_TRIAGE_CATEGORIES` — using `removeprefix`, NEVER `.strip(...)`!) **before** comparison (exit `2` on unreadable path or unknown category). Only emit `emit_triage_shadow_label(...)` when `verdict.status` is terminal non-advisory (`KEEP -> True`; `REVERT_REGRESSION | REVERT_NOISE | REJECT_PARITY | REJECT_DISPATCH | REJECT_VRAM -> False`; skip on `REJECT_UNVERIFIABLE` or `ADVISORY_ONLY`).
- **Cross-mode validation (`run_profile`)**:
  - Wrap `run_profile` execution inside `profiling_telemetry_session(log_dir=args.telemetry_log_dir)` (`§3.9`), emitting operational M1 events **and** (in `--triage`) `emit_triage_shadow_verdicts(prod, findings)` alongside `emit_triage_telemetry(prod, findings, diag=diag, target_missed=args.target_missed)` before raising `SystemExit` (so `QueueListener` flushes cleanly).
  - Require exactly one active mode among `(args.fn is not None, args.check is not None, args.compare is not None, args.triage is not None, args.telemetry_summary)` via `_not_measured("give exactly one of --fn, --check, --compare, --triage, or --telemetry-summary")` (preserving `"exactly one"` test assertion and exit `2`).
  - Keep `--sha` error `"--sha is only valid with --check"`.
  - Do NOT reject default `n_steady`/`n_wall` values in `--check`/`--compare`/`--triage`/`--telemetry-summary` (`test_profile_verb.py:32-41`).
  - Reject any non-`None` measure-only flag (`shapes`, `inputs`, `extent_axis`, `scale`, `n_units`, `target_extent`, `out`) when `--fn` is omitted (`exit 2`).
  - Reject `rejected_dir is not None` unless `args.compare is not None` OR `args.telemetry_summary` (`exit 2`).
  - Reject any remaining compare-only flag (`primary_metric is not None`, `not require_parity`, `not require_corruption_control`, `max_rel_err is not None`, `max_vjp_rel_err is not None`, `min_work_retention_ratio is not None`, `allow_dirty`, `min_speedup is not None`, `hold_fixed is not None`, `bypass_hold_fixed`, `ratchet_pointer is not None`, `from_triage_probe is not None`, `triage_category is not None`) when `--compare` is omitted (`exit 2`).
  - Reject any triage-only flag (`diag is not None`, `precision_baseline is not None`, `target_missed`) when `--triage` is omitted (`exit 2`).
  - Inside `--compare`: reject `(from_triage_probe is None) != (triage_category is None)`, reject `hold_fixed is not None and bypass_hold_fixed`, and reject `not require_parity and (not require_corruption_control or max_rel_err is not None or max_vjp_rel_err is not None or min_work_retention_ratio is not None)` (`exit 2`).
  - `telemetry_log_dir` is permitted across all 5 modes.
  - Preserve `source.count("steady_state_flags(") == 2` (`tests/profiling/test_rules.py:67`).
- **Strict Exit Codes**:
  - `--compare`: `0` ONLY on `VerdictStatus.KEEP` (`promotable=True`); `1` on `ADVISORY_ONLY`, `REVERT_REGRESSION`, `REVERT_NOISE`, `REJECT_PARITY`, `REJECT_DISPATCH`, `REJECT_VRAM`; `2` on `REJECT_UNVERIFIABLE`, unreadable JSON, malformed flags, or ratchet chain mismatch.
  - `--triage`: `0` on 0 findings; `1` on `>= 1` findings; `2` on unreadable JSON or invalid cross-record inputs. Under an active telemetry session (or always on success), calls `emit_triage_telemetry(...)` + `emit_triage_shadow_verdicts(prod, findings)` and prints `shadow_subject_key: <key>` (`make_triage_subject_key(prod)`).
  - `--telemetry-summary`: `0` when 0 algedonic signals fire (including warm-up shadow gates or resolved empty log dir); `1` when `>= 1` algedonic signal fires; `2` when neither a telemetry log dir (`--telemetry-log-dir` / Layers 2–4) nor `--rejected-dir` resolves, an explicit `--telemetry-log-dir` / `--rejected-dir` does not exist, an `audit/shadow/triage_*.toml` schema fails (`GateCheckError` exit `3`), or matched `events.*.jsonl*` / `*.verdict.json` files contain unreadable/corrupt JSON/JSONL.

---

### 3.7 Skills & Knowledge Persistence (`agent_assets/skills/`)

(Ensure zero `xtrax_version:` frontmatter in any `SKILL.md` and never regenerate `src/xtrax/agent_plugin.json` on feature PRs.)

1. **`agent_assets/skills/xtrax-probing/` (`SKILL.md`, `references/contract.md`, `references/workflows.md`)**:
   - Document `compare_probe_pair` / `update_baseline_pointer` / `xtrax profile --compare`, `probe_kernel_dual_pass`, PTX/`ptxas`/Triton stats, `triage_probe_bundle` / `xtrax profile --triage`, `END_TO_END` diagnostic-mode refusal in `claims.py`, and optional `BTH_RUN_ID`/`BTH_CAMPAIGN_ID` config tagging.
   - Document operational telemetry (`--telemetry-log-dir`, `--telemetry-summary`, `--from-triage-probe <probe.json|subject_key> --triage-category <cat>`, `--target-missed`, `attest_triage_shadow_isolation`, `audit/shadow/triage_*.toml` + `cisternal shadow check audit/shadow/triage_<cat>.toml` passing explicit TOML paths because filenames differ from `xtrax.triage.<cat>` feature names), and the **Algedonic Stop & Pre-Cycle Observe Protocol** (explicitly referencing `algedonic-escalation` direct return-to-dispatcher — never citing removed `praxia-daemon` mechanisms).
2. **`agent_assets/skills/xtrax-optimizing/` (`SKILL.md`, NEW `references/kernel-optimization-loop.md`, UPDATED `references/tier3-pallas-megakernels-and-ragged-buckets.md`)**:
   - Update Non-Negotiable #2 for `paired_configs` (same SHA) vs `compare_probe_pair` (clean `baseline_sha -> candidate_sha`, `allow_dirty=True` capped at `ADVISORY_ONLY`).
   - Encode mandatory algedonic stop conditions (`instrument_degraded`, `>= 3` consecutive `REVERT_NOISE`, `parity_escape`, or urge to bypass gates via `--no-require-parity`/`--bypass-hold-fixed`, plus logging `triage_gap` bottlenecks to issue-tracker `staging`/`ideas` with `.diag.json` fixtures).
   - Write `references/kernel-optimization-loop.md` with exact heading anchors (`#fp64-pipe-saturation`, `#register-spill-pressure`, `#atomic-contention`) matching `triage.py`'s `playbook_ref` strings.
   - Upstream `MK-50..MK-64` from `prolix/docs/plans/2026-10-28-xtrax-megakernel-composition-ledger.md:584-812` + `v3.4` benchmark lessons cleanly into `tier3-pallas-megakernels-and-ragged-buckets.md`.
3. **`agent_assets/skills/xtrax-activation-parity/SKILL.md`**:
   - Add a distinct **Single-Kernel Pallas `REJECT_PARITY` Section (`K0..K3`)** (`assert_kernel_ran` + dtype/tag parity -> `interpret=True` vs GPU isolation -> `work_active_count` / `B=1` vs `B>1` bisection -> `vjp_parity` + `require_vjp_controls_red`), keeping `L0..L3` explicitly scoped to full-model checkpoint ports.

---

### 3.8 Verification & Negative Controls

1. **Updated Existing Test Files**:
   - `tests/profiling/test_unit_probe.py`: update `as_metrics()` exact-key set (`9` or `10` keys) + add dispersion (`n_wall=1,2,3,5`) & `peak_vram_bytes` device/mock tests.
   - `tests/cli/test_profile_verb.py`: keep all existing tests green (including zero unintended stderr/file side effects when Layer 5 telemetry is off!) + add `--compare` (`--no-require-corruption-control`, default `--hold-fixed` mismatch exit 2, `--bypass-hold-fixed` `ADVISORY_ONLY` exit 1, `--from-triage-probe` + `--triage-category` pairing for all 4 categories incl. `atomic_contention`/`register_spill_pressure` + `KEEP`/`REVERT_*`/`ADVISORY_ONLY` label rules, `KEEP` exit 0, `REJECT_UNVERIFIABLE`/chain error exit 2), `--triage` (verifies `shadow_subject_key` printed AND `shadow.verdict` records written + `--target-missed` cross-mode checks), and `--telemetry-summary` (`exit 0/1/2`, warm-up shadow gates, corrupt `events.*.jsonl*`, non-existent explicit paths) tests.
   - `tests/profiling/test_report.py`: verify `_DISCOVERY_SKIP_SUFFIXES` hides `.verdict.json`/`.diag.json` under directory discovery while explicit `render_report([diag_path])` renders `.diag.json` traces.
   - `tests/profiling/test_rules.py`: verify `steady_state_flags(` count stays 2.
   - `tests/test_import_isolation.py`: insert `"xtrax.profiling"`, `"xtrax.profiling.compare"`, `"xtrax.profiling.kernel_probe"`, `"xtrax.profiling.triage"`, and `"xtrax.telemetry.profiling_events"` in strict `sorted()` order into `_PACKAGES` (`tests/distribution/test_sorted_registries.py:60`).
2. **New Test Files**:
   - `tests/profiling/test_compare_and_triage.py`: `BUG-TRITON-ATOM1` work-drop control, baseline `work_active_count <= 0` (`REJECT_UNVERIFIABLE`), missing/negative parity/dispatch/VRAM controls, integer mismatch & non-finite NaN divergence (`parity_nonfinite_metrics` cleanly persists without crashing `ProbeRecord`), `evaluate_kernel_parity_for_probe` & `corruption_result` controls, `require_parity=False` still catching explicit `"fail"`/`work_active_count <= 0` while capping missing-parity at `ADVISORY_ONLY`, `step_time_basis` invalid string refusal, dirty-SHA `ADVISORY_ONLY` cap, ratchet chain-continuity refusal, `cmdbuf_off` comparison refusal & `claims.py` `END_TO_END` refusal, MAD (`1.4826`) / IQR (`0.7413`) & non-wall-metric guards, `compute_precision_ratio` `x64=False` rejection & Prolix shared-process `x64=True` test, `merge_xla_flags`, `hlo_text_for` `**kwargs`, PTX/`ptxas`/Triton parser & autotune dedup fixtures, `_diag_worker` zero top-level `jax` / `xtrax.cli` subprocess isolation test, all 6 triage categories including `batch_scaling_eff` ideal-scaling positive vs contention negative control & Rule 3/Rule 6 boundary tests, and `playbook_ref` link/anchor verification.
   - `tests/telemetry/test_profiling_events.py`:
     - **Keep `pytest.importorskip("cisternal.shadow")` strictly INSIDE the live-`cisternal.shadow` integration test functions (NEVER at module top level)** so `cisternal.shadow.mode_state.resolve_enforcer_symbol` imports `test_triage_shadow_writes_isolated` cleanly AND `tier1_core` passes even on `cisternal 0.1.7..0.1.10`.
     - Cover zero top-level `jax`/`cisternal`/`bathos` imports; Layer-5-off zero-stderr/zero-filesystem-write isolation; active-pipeline reuse vs explicit `log_dir` re-targeting; owned `init`/`shutdown_pipeline` JSONL flush; mocked fallback tests simulating absent `cisternal` and pre-`0.1.11` `cisternal` (`>=90%` line / `>=80%` branch coverage); `make_triage_subject_key` multi-iteration uniqueness (`N > 1`); targeted single-category `emit_triage_shadow_label`; all 6 `detect_algedonic_signals` positive & negative controls (prefix-matched colon-suffixed reasons, zero-negative `base_rate==1.0` suppression, warm-up suppression, `disabled` non-failure vs `failed`, `.verdict.json` dedup + chronological sorting, `bypass_corruption` non-erosion vs `bypass_parity`/`allow_dirty`, `n_cli_invocations` excluding `dual_pass`, and `top_kernel_pct` / `target_missed` `triage_gap`); live-`cisternal.shadow` tests (`load_mode_state`, `test_triage_shadow_writes_isolated` positive + failing leak control via `_run_isolation_audit`, `attest_triage_shadow_isolation` -> `evaluate_gate` PASS); and `flock`-guarded opt-in `.praxia/audits.jsonl` emission (verifying non-praxia repos do NOT get `.praxia/` created).
   - `tests/controller/test_profiling_bathos.py`: `resolve_bathos_provenance` mocked tests + `pytest.importorskip("bathos.query")` integration test (`tier4_controller`).
   - `tests/cli/test_entrypoint.py`: verify zero top-level `jax`/`tyro`/`bathos` imports on CLI entrypoint.
3. **Mandatory CI Gates (must all exit 0)**:
   - `just audit-bathos-independence`
   - `just audit-dispatch-independence`
   - `just audit-profiling-contract`
   - `just audit-no-future-annotations`
   - `just audit-telemetry-coverage`
   - `just audit-dataclass-shape-diff`
   - `just audit-added-types-diff`
   - `just audit-api-ergonomics-gate`
   - `just audit-structure-complexity-gate`
   - `just audit-documentation-gate`
   - `just audit-type-hardening-gate`
   - `just audit-project-hygiene`
   - `just audit-controller-types`
   - `uv run ruff check src/ controller/ tests/profiling tests/cli tests/testing tests/telemetry tests/controller`
   - `uv run ty check`
   - `uv run pytest tests/profiling tests/cli tests/testing tests/telemetry tests/audit tests/distribution tests/controller`

---

### 3.9 `src/xtrax/telemetry/profiling_events.py` & `audit/shadow/` — Operational Telemetry, Heuristic Calibration, & Algedonic Signals

Purpose: Record operational events about the kernel-optimization tools themselves (so broken instrumentation, noise cliffs, bypass erosion, parity escapes, miscalibrated heuristics, and missing bottleneck classes are caught by data rather than anecdotes), while keeping `src/xtrax/profiling/` 100% leaf-clean and making `cisternal` strictly optional (`xtrax[plugin]`). Zero `from __future__ import annotations`; zero top-level `jax`, `cisternal`, `tyro`, or `bathos` imports; zero `xtrax.eda` or `xtrax.devtools` imports; zero `import/from praxia`, `mcp__praxia`, or `rig[_-]run` substrings anywhere in the module (comments/docstrings/strings included, per `scripts/audit_dispatch_independence.py:50-54`). Direct top-level imports of `ProbeRecord`, `DeltaVerdict`, `BottleneckFinding`, and `AuditFinding` satisfy runtime `beartype`.

#### 3.9.1 Optional `cisternal` Lifecycle (`profiling_telemetry_session`)
Reuses `xtrax.telemetry.record.telemetry_opted_out()` (`XTRAX_TELEMETRY_OPTOUT`) and dynamic `importlib.import_module` loading (`src/xtrax/telemetry/record.py:407`):

```python
@contextlib.contextmanager
def profiling_telemetry_session(
    log_dir: str | Path | None = None,
    *,
    emit_events: bool = True,
) -> Iterator[TelemetrySessionStatus]:
```
1. If `not emit_events` or `telemetry_opted_out()`: yield inactive status immediately.
2. Dynamically import `cisternal` and `cisternal.telemetry.pipeline`; on `ImportError`, yield inactive status (`reason="cisternal_not_installed"`).
3. Inspect `active = pipeline.get_pipeline()`:
   - **When `log_dir is None`**:
     - If `active is not None`: reuse cleanly (`init_owned = False`; resolve `status.log_dir` from `active`'s `JsonlExporter._path.parent` if available; never shut down enclosing pipeline).
     - Else resolve Layers 2–4 via `cisternal.shadow.resolution.resolve_log_dir(cli_arg=None)` (`CISTERNAL_TELEMETRY_LOG_DIR`, `CISTERNAL_LOG_DIR`, `BTH_LOG_DIR`, `CTXP_LOG_DIR`, `pyproject.toml [tool.cisternal.telemetry]`, `~/.config/cisternal/config.toml`), or if `cisternal.shadow.resolution` is absent (`cisternal < 0.1.11`), inspect those 4 env vars directly (`os.getenv` + whitespace/`"none"` check).
     - **Sandbox & Pytest Hygiene Rule**: If Layer 5 resolves off (`LogDirOffError` / unconfigured / disabled) AND `os.getenv("XTRAX_PROFILE_TELEMETRY") != "1"`, do **NOT** call `cisternal.init()` (avoids `~/.cisternal/logs` sandbox errors, `[cisternal] Warning:` stderr noise breaking `test_profile_verb.py` `capsys`, and background threads). Yield inactive status (`reason="log_dir_off"`).
   - **When `log_dir is not None`** (explicit Layer-1 destination — never silently reuse a stale pipeline directory per `cisternal/telemetry/pipeline.py:385-395`):
     - `target_dir = Path(log_dir).resolve()`.
     - Inspect active JSONL exporter directories (`{Path(p).resolve().parent for exp in getattr(active, "_exporters", ()) if (p := getattr(exp, "_path", None)) is not None}`).
     - If `active is not None and target_dir in active_dirs`: reuse (`init_owned = False`).
     - Else: if `active is not None`, call `pipeline.shutdown_pipeline()` first to drain and close the prior pipeline before initializing at `target_dir` (`retargeted_from_active = True`).
4. When initializing (`target_dir` resolved): call `cisternal.init(log_dir=target_dir, heartbeat_interval=30.0)` (`init_owned = True`). In `finally:` when `init_owned=True`, call zero-arg `pipeline.shutdown_pipeline()` (`cisternal/telemetry/pipeline.py:472`) so buffered records drain and flush before `SystemExit`.
5. **Never-raise contract**: Any exception in init, emit, or shutdown records `status.degraded_reason` (no bare `pass`, compliant with `ruff S110`) and never fails the caller.

#### 3.9.2 Operational M1 Event Taxonomy
Emitted via `cisternal.emit_event(name, level=..., **fields)`. Because `cisternal` captures `Record.git_hash` asynchronously (`cisternal/telemetry/pipeline.py:443-454`), all SHAs used downstream are stored directly inside `fields` from `ProbeRecord`. Notice `n_cli_invocations` counts strictly the 5 top-level CLI mode events (`measure`, `check`, `compare`, `triage`, `not_measured`), excluding `dual_pass` and `parity_escape`:

| Event Name | Default Severity | Emitter Helper (all `<= 3` req params) | Key Fields |
|---|---|---|---|
| `xtrax.profile.measure` | `INFO` | `emit_measure_telemetry(fn, platform, device_kind, *, cardinality, n_flags, exit_code, duration_ms)` | `mode="fn"`, `fn`, `platform`, `device_kind`, `cardinality`, `n_flags`, `exit_code`, `duration_ms` |
| `xtrax.profile.check` | `INFO` (`WARNING` if stale) | `emit_check_telemetry(record_path, exit_code, *, stale, n_flags)` | `mode="check"`, `record_path`, `exit_code`, `stale`, `n_flags` |
| `xtrax.profile.not_measured` | `WARNING` | `emit_not_measured_telemetry(mode, reason_code, reason)` | `mode`, `reason_code`, `reason`, `exit_code=2` |
| `xtrax.profile.dual_pass` | `INFO` (`WARNING` if degraded) | `emit_dual_pass_telemetry(result)` | `prod_probe_id`, `device_kind`, `jax_version`, `pass_b_status` (`"ok"\|"disabled"\|"no_importable_spec"\|"failed"`), `pass_c_status` (`"ok"\|"disabled"\|"no_importable_spec"\|"skipped:<reason>"\|"failed"`), `ptx_spill_source`, `ptx_filter_status`, `hlo_attribution_skipped`, `ptxas_warning_count`, `diag_skipped_reason` |
| `xtrax.profile.compare` | `INFO` (`WARNING` on `ADVISORY_ONLY`/`REJECT_*`) | `emit_compare_telemetry(verdict, *, bypass_parity, bypass_corruption, bypass_hold_fixed, allow_dirty)` | `baseline_probe_id`, `candidate_probe_id`, `baseline_sha`, `candidate_sha`, `verdict_status`, `promotable`, `primary_metric`, `speedup_x`, `noise_floor_x`, `reasons`, `bypass_parity`, `bypass_corruption`, `bypass_hold_fixed`, `allow_dirty` |
| `xtrax.profile.triage` | `INFO` (`WARNING` on `unclassified_slowdown`) | `emit_triage_telemetry(prod, findings, *, diag=None, top_kernel_pct=None, target_missed=False)` | `prod_probe_id`, `shadow_subject_key` (`make_triage_subject_key(prod)`), `has_diag` (`diag is not None`), `categories`, `n_findings`, `top_kernel_pct`, `target_missed`, `unclassified_slowdown` (`len(findings) == 0 and (top_kernel_pct >= 30.0 or target_missed)`) |
| `xtrax.profile.parity_escape` | `CRITICAL` | `record_parity_escape(probe_id, commit_sha, reason, *, control_added="")` | `probe_id`, `commit_sha`, `reason`, `control_added` |

- **`top_kernel_pct` Derivation inside `emit_triage_telemetry`**: When `top_kernel_pct is None`, compute identically to `report.py:130-133`: over non-`None` entries `(sec, _)` in `diag.scopes` (when `diag is not None` and `diag.scopes`), let `total_exc = sum(float(sec) for ...)`, `max_exc = max((float(sec) for ...), default=0.0)`, and `top_kernel_pct = (100.0 * max_exc / total_exc) if total_exc > 0.0 else 0.0` (else `0.0`).

#### 3.9.3 `cisternal.shadow` Calibration for `provisional_gpu_heuristic` Triage Rules
Let `PROVISIONAL_TRIAGE_CATEGORIES = ("fp64_pipe_saturation", "register_spill_pressure", "spatial_dead_work", "atomic_contention")` and `TRIAGE_SHADOW_ENFORCER = "tests/telemetry/test_profiling_events.py::test_triage_shadow_writes_isolated"`.
Four committed mode-state TOMLs under `audit/shadow/`:
- `audit/shadow/triage_fp64_pipe_saturation.toml`
- `audit/shadow/triage_register_spill_pressure.toml`
- `audit/shadow/triage_spatial_dead_work.toml`
- `audit/shadow/triage_atomic_contention.toml`

Each strictly conforms to `cisternal.shadow.mode_state.ALLOWED_KEYS` (`mode_state.py:46-65`, zero unknown tables/keys) so `cisternal.shadow.load_mode_state(...)` validates with exit `0`:
- `[feature]`: `name = "xtrax.triage.<category>"`, `description`, `version = "0.1.0"`
- `[ladder]`: `current_stage = "shadow"`
- `[shadow]`: `traffic_fraction = 1.0`, `writes_isolated = true`, `writes_isolated_enforced_by = "tests/telemetry/test_profiling_events.py::test_triage_shadow_writes_isolated"`
- `[scoring]`: `positive_label = true`
- `[promotion]`: `target_stage = "advisory"`, `min_verdicts = 20`, `min_labels = 10`, `allowed_label_sources = ["frozen_gold", "owner_confirmed", "at_work"]`, `min_precision = 0.70`, `min_precision_lower_bound = 0.50`, `max_fp_rate = 0.25`, `isolation_violations_max = 0`, `threshold_basis`, `feature_version = "0.1.0"`

Key calibration & isolation APIs (degrading cleanly when `cisternal < 0.1.11` lacks `cisternal.shadow`):
- **`make_triage_subject_key(prod: ProbeRecord) -> str`** & **`resolve_triage_subject_key(probe_or_key: str | Path | ProbeRecord) -> str`**:
  - Computes per-measurement canonical 4-segment key `f"probe:{slug}:{sha12}:{digest12}"` where `slug = re.sub(r"[^A-Za-z0-9._-]+", "-", prod.probe_id).strip("-") or "probe"` (ensuring `:` in `"xtrax-profile:<fn>"` at `profile_verb.py:633` never introduces extra colons), `sha12 = re.sub(r"[^0-9a-fA-F-]+", "", prod.git_sha)[:12] or "nosha"`, and `digest12` is `sha256(json.dumps(..., sort_keys=True, separators=(",", ":")))[:12]` over `(probe_id, git_sha, timestamp, stage, n_atoms, platform, device_kind, x64_enabled, xla_flags, metrics, config)` so repeated runs accumulate `N >= 20` distinct subjects instead of overwriting `N = 1` in `cisternal.shadow.scoring.py:406, 417-418`.
  - `resolve_triage_subject_key` accepts either a literal `"probe:..."` key string (`startswith("probe:") and len > len("probe:")`) or a path to the triaged `ProbeRecord` JSON (re-deriving the exact key deterministically via `ProbeRecord.read`).
- **`emit_triage_shadow_verdicts(prod, findings, *, stage="shadow", feature_version="0.1.0") -> int`**: emits one `shadow.verdict(feature=f"xtrax.triage.{cat}", subject_key=make_triage_subject_key(prod), verdict=(cat in fired_cats), stage=stage, feature_version=feature_version, inputs_sha256=<full64>)` for each of the 4 provisional rules. Called by `run_profile`'s `--triage` branch as well as triage drivers.
- **`emit_triage_shadow_label(probe_or_key, category, label, *, label_source="at_work", labeller="agent") -> bool`**: normalizes via `category.strip().removeprefix("xtrax.triage.")` (NEVER `.strip(...)`!), validates `in PROVISIONAL_TRIAGE_CATEGORIES`, and emits ONE targeted `shadow.label(feature=f"xtrax.triage.{norm_category}", subject_key=resolve_triage_subject_key(probe_or_key), label=label, label_source=label_source, labeller=labeller)`. Never broadcasts across all 4 categories (preventing false `FN`/`TN`/`base_rate` corruption).
- **Confusion-Matrix Ground Truth Semantics**:
  - `TP`: rule fired (`True`) -> `--triage-category <cat>` fix achieved `KEEP` (`label=True`).
  - `FP`: rule fired (`True`) -> `--triage-category <cat>` fix reverted/rejected (`label=False`) or `owner_confirmed` `False`.
  - `TN`/`FN`: speculative interventions on non-triggering rules + golden/human negative & positive control labels (`label_source="frozen_gold"|"owner_confirmed"`).
- **`attest_triage_shadow_isolation(log_dir=None, *, stage="shadow") -> dict[str, bool]`**:
  - Inside `profiling_telemetry_session(log_dir=log_dir)` using a throwaway `TemporaryDirectory` `live_root`, calls shared `_run_isolation_audit(live_root=live_root, shadow_log_dir=shadow_sink, stage=stage, enforced_by=TRIAGE_SHADOW_ENFORCER, verdict_feature_version="__isolation_audit__")` with **`check_roots=[live_root, shadow_sink]`** (preventing `cisternal.shadow.isolation.prove_isolation` at `isolation.py:488-502` from snapshotting `.git`/`.venv`) across all 4 provisional features. Using `verdict_feature_version="__isolation_audit__"` ensures `cisternal.shadow.scoring:372-374, 401-404` records valid `shadow.isolation` evidence without inflating `[promotion] feature_version = "0.1.0"` `verdicts_total` in production `--telemetry-log-dir` directories (`finally:` flushes `shutdown_pipeline()` before `evaluate_gate` reads it).

#### 3.9.4 Reading Logs, Evaluating Shadow Gates, & Deterministic Algedonic Detector

```python
class AlgedonicSignalKind(enum.Enum):
    INSTRUMENT_DEGRADED = "instrument_degraded"
    NOISE_FLOOR_BLOWN = "noise_floor_blown"
    GUARDRAIL_EROSION = "guardrail_erosion"
    PARITY_ESCAPE = "parity_escape"
    HEURISTIC_MISCALIBRATED = "heuristic_miscalibrated"
    TRIAGE_GAP = "triage_gap"

@dataclasses.dataclass(frozen=True, slots=True)
class AlgedonicThresholds:
    min_degraded_dual_pass_per_env: int = 2
    max_consecutive_revert_noise: int = 3
    max_noise_floor_x: float = 1.10
    min_bypass_count: int = 2
    min_compare_for_rate: int = 5
    max_bypass_rate: float = 0.25
    min_invocations_for_error_rate: int = 5
    max_not_measured_rate: float = 0.30
    min_triage_gap_count: int = 2
    min_shadow_labels_for_demote: int = 10

@dataclasses.dataclass(frozen=True, slots=True)
class AlgedonicSignal:
    kind: AlgedonicSignalKind
    severity: str                    # "warning" | "error" | "critical"
    summary: str
    evidence: dict[str, Any]
    recommended_action: str
    demote_feature: str | None = None
```

1. **Strict Log/Verdict Readers & Safe Shadow Evaluation (`--telemetry-summary`)**:
   - `read_telemetry_events(log_dir: Path) -> list[dict[str, Any]]`: scans ONLY files matching `_LOG_FILE_RE = re.compile(r"(?P<base>.+\.jsonl)(?:\.(?P<rot>\d+))?")` sorted oldest-rotated first (`(base, -int(rot or 0))`, mirroring `cisternal.shadow.scoring:237, 294-328`), ignoring unrelated `.json` files; raises `TelemetryReadError` on any `OSError`, `json.JSONDecodeError`, or non-`dict` JSONL line.
   - `read_rejected_verdicts(rejected_dir: Path) -> list[dict[str, Any]]`: reads `*.verdict.json` sorted `(stat.st_mtime_ns, name)` and normalizes `verdict_status = str(v.get("verdict_status") or v.get("status") or "").lower()` (bridging `DeltaVerdict.status` vs `xtrax.profile.compare` `verdict_status`); raises `TelemetryReadError` on invalid JSON/non-`dict`.
   - `collect_triage_shadow_checks(log_dir: Path | None, *, repo_root: Path | None = None) -> list[dict[str, Any]]`: resolves `repo_root` (`audit/shadow/` ancestor fallback), evaluates each `triage_*.toml` cleanly: propagates `GateCheckError` with `exit_code == 3` (corrupt TOML -> exit `2`), and maps warm-up `GateCheckError(exit_code=2)` (`"REFUSAL:"` on 0 verdicts/labels at `mode_state.py:315-331`) to `status="uncalibrated"` without failing `--telemetry-summary`.
2. **`detect_algedonic_signals(events, *, verdicts=(), shadow_checks=(), thresholds=AlgedonicThresholds()) -> list[AlgedonicSignal]`**:
   - Normalizes `events` chronologically by `(_normalize_ts(ev.get("ts")), source_order_index)` and normalizes status across both compare events (`verdict_status`) and rejected `.verdict.json` sidecars (`status` -> `verdict_status` lowercase). Deduplicates `verdicts` against `events` by `(baseline_probe_id, candidate_probe_id, baseline_sha, candidate_sha, primary_metric, verdict_status)`.
   - Evaluates the 6 signals cleanly:
     1. `INSTRUMENT_DEGRADED` (`error`): `>= min_degraded_dual_pass_per_env` degraded `xtrax.profile.dual_pass` events sharing `(device_kind, jax_version)` where `pass_b_status == "failed"` OR `pass_c_status == "failed"` OR `pass_c_status.startswith("skipped:")` OR `int(ptxas_warning_count or 0) > 0` OR `ptx_filter_status == "unfiltered_name_mismatch"` OR `bool(hlo_attribution_skipped)`. Intentional `"disabled"` / `"no_importable_spec"` passes do NOT count.
     2. `NOISE_FLOOR_BLOWN` (`warning`): chronological longest consecutive streak of `REVERT_NOISE` comparisons `>= max_consecutive_revert_noise` OR any compare event with `noise_floor_x > max_noise_floor_x`.
     3. `GUARDRAIL_EROSION` (`error`): counts ONLY genuine guardrail bypasses `is_erosion_bypass = bool(bypass_parity or bypass_hold_fixed or allow_dirty)` (tracking `--no-require-corruption-control` separately as informational `corruption_optout_count`, NEVER failing at `min_bypass_count = 2`!). Fires when `erosion_bypass_count >= min_bypass_count`, OR (`n_compare >= min_compare_for_rate` and `erosion_bypass_count / n_compare > max_bypass_rate`), OR (`n_cli_invocations >= min_invocations_for_error_rate` and `n_not_measured / n_cli_invocations > max_not_measured_rate`).
     4. `PARITY_ESCAPE` (`critical`): `>= 1` `xtrax.profile.parity_escape` event -> recommends demoting autonomous ratchet to supervised/advisory, adding a mandatory `ParityContract` negative control, and running `bth blast-radius assess` when `bathos` is active.
     5. `HEURISTIC_MISCALIBRATED` (`warning`): for each `shadow_checks` entry, normalize blocking prefixes `prefix = str(r).split(":", 1)[0]`. Fires with `demote_feature = entry["feature"]` if `"isolation_violation_detected" in prefixes` OR (`metrics.get("labels_total", 0) >= thresholds.min_shadow_labels_for_demote` and (`bool(prefixes & {"precision_deficit", "precision_lower_bound_deficit", "fpr_exceeded", "recall_deficit", "recall_lower_bound_deficit"})` OR (`"precision_lower_bound_below_base_rate" in prefixes` and `(fp + tn) > 0` and `(tp + fn) > 0`))). Warm-up / undefined-denominator / zero-negative `base_rate==1.0` states do NOT trigger false demotion.
     6. `TRIAGE_GAP` (`warning`): `>= min_triage_gap_count` `xtrax.profile.triage` events with `unclassified_slowdown=True` -> recommends filing an issue-tracker backlog/ideas item with the `.diag.json` fixture to add the missing bottleneck rule. *(Note: zero `import/from praxia`, `mcp__praxia`, or `rig[_-]run` substrings in any string or docstring.)*
3. **Optional `.praxia/audits.jsonl` Writer (`append_algedonic_finding`)**:
   - Maps `"warning" -> "minor"`, `"error" -> "major"`, `"critical" -> "critical"` using `xtrax.findings.emit_metric_finding(dim="xtrax.profiling", ...)` and `xtrax.tombstone.is_tombstoned(...)`.
   - Resolves target `audits_path` ONLY when passed explicitly OR when `(repo_root or Path.cwd()) / ".praxia"` ALREADY exists as a directory (returns `None` cleanly in non-tracker repos — never auto-creates `.praxia/`).
   - Appends serialized `finding_to_dict(finding)` JSONL protected by `fcntl.flock(LOCK_EX)` (avoiding `xtrax.findings.append_finding`'s unlocked write hazard documented at `src/xtrax/telemetry/ledger.py:20-24`).

---

## Adversarial Coherence Audit Log (Round 1)

| ID | Severity | Summary of Issue Caught | Resolution in v2 |
|---|---|---|---|
| **B1** | `BLOCKING` | `src/xtrax/profiling_bathos.py` failed `audit_bathos_independence` & `tier1_core` coverage DAG | Removed `src/xtrax/profiling_bathos.py`; moved `resolve_bathos_provenance` to `controller/profiling_bathos.py` (`tier4_controller`); kept stdlib-only `probe_record_to_bathos_metrics` in `emitters.py`. |
| **B2** | `BLOCKING` | `UnitProbeResult` `as_metrics()` broke `test_unit_probe.py:175-183`; field count/order; `int` vs `float`; `n_wall<4` range/IQR mixup | Appended 3 defaulted fields after `dispatch_counts`; fixed `float(peak_vram_bytes)`; separated `n_wall >= 4` IQR from `n_wall < 4` (`0.0`); added multi-GPU safe `memory_stats()` & scheduled `test_unit_probe.py` update. |
| **B3** | `BLOCKING` | Unfinished/inverted `batch_scaling_eff` formulas; missing `1.4826 * MAD` & `primary_metric` guard on IQR | Specified `step_time_basis` + per-replica efficiency $\eta_{\text{rep}} = t_{\text{rep}}(B_{\text{lo}}) / t_{\text{rep}}(B_{\text{hi}})$; added `1.4826` MAD factor, guarded both MAD/IQR on `total_step_seconds`, rejected negative dispersion. |
| **B4** | `BLOCKING` | Fail-open `require_parity=False`, vacuous `ParityContract()`, spoofable statuses, `BUG-TRITON-ATOM1` work drop, missing Step 1/4/5/ratchet checks | Defaulted `require_parity=True`, capped bypass to `ADVISORY_ONLY`, required numeric witnesses + `min_work_retention_ratio=0.99`, tightened Steps 1/4/5, required ratchet chain continuity, isolated `.verdict.json`. |
| **B5** | `BLOCKING` | `XLA_FLAGS` naive append, `diag_record` leaking into `END_TO_END`, GPU OOM, `hlo_text_for` kwargs, `n_steady` scope scaling, PTX virtual vs `ptxas` physical regs | Added `shlex` `merge_xla_flags`, `END_TO_END` diagnostic rejection in `claims.py`, subprocess Pass A/B VRAM isolation, `hlo_text_for(**kwargs)`, `1/n_steady` scope normalization, `ptxas`/Triton metadata parsers. |
| **B6** | `BLOCKING` | `compute_precision_ratio` failed on Prolix's global `jax_enable_x64=True` for `single` and allowed `mixed` with `x64=False` | Enforced `mixed.x64_enabled is True`, required `precision="single"` vs `"mixed"` disambiguation when `single.x64_enabled is True`, added full unanimity + positive guards. |
| **M1** | `MAJOR` | `tyro` CLI pinned error strings (`test_profile_verb.py`), `ADVISORY_ONLY` exit 0, `--triage` exit codes, `ProfileArgs` order, `test_rules.py` count==2 | Appended fields after `sha`, preserved exact substrings & `test_rules.py` count, made `ADVISORY_ONLY` exit 1, validated cross-mode flags to exit 2. |
| **M2** | `MAJOR` | `_diag_worker` couldn't import `xtrax.cli.loader`; `.diag.json`/`.verdict.json` polluted `discover_records()` | Made `_diag_worker` stdlib-`importlib` pure; updated `report._SKIP_SUFFIXES` and isolated `diag_dir`/`rejected_dir`. |
| **M3** | `MAJOR` | `kernel_parity.ParityResult` & `VjpParityResult` lacked scalar metrics for `ParityContract` | Added `ProbeParityPayload` & `build_probe_parity_payload` in `src/xtrax/testing/kernel_parity.py`. |
| **M4** | `MAJOR` | Ledger ends at `MK-64` (no `MK-65..74`); broken `#remediation-playbook` anchor; `activation-parity` scope mismatch | Fixed range to `MK-50..MK-64`, fixed exact heading anchor & verified `playbook_ref`s, added `K0..K3` Pallas section to `xtrax-activation-parity`. |
| **M5** | `MAJOR` | `beartype` `TYPE_CHECKING` hazard, `complexipy<=15`, `interrogate`, `api_ergonomics` (`<=5` req params) | Enforced direct `ProbeRecord` imports, decomposed helpers, full docstrings, `<=3` required params on new public functions. |
| **Min 1–4** | `MINOR` | Missing test files in §3.7, missing multi-replica kwargs in `emitters.py`, `"blackwell"` false positive, frozen `prod_record` order | Expanded §3.7 test/audit matrix, added multi-replica kwargs to `emit_probe_record`, narrowed `low_fp64_device_substrings`, stamped `diag_skipped_reason` pre-construction. |

## Adversarial Coherence Audit Log (Round 2)

| ID | Severity | Summary of Issue Caught | Resolution in v3 |
|---|---|---|---|
| **R2-B1** | `BLOCKING` | Pass B vs Pass C ordering contradiction (immutable `diag_record`), missing `ptxas --verbose` invocation, Pass B `jax.jit(fn)` broad fallback | Made `_diag_worker` stateless (`--mode diag/ptx`, `--result-json`); parent merges Pass B + Pass C and writes single immutable `diag_record` once; added `find_ptxas`, `compute_capability_to_sm`, `ptxas --verbose` exec, Triton `.cubin`/`.json` dedup, `triton_spills`, and broad Pass B HLO fallback. |
| **R2-B2** | `BLOCKING` | Missing CLI `--hold-fixed` default policy; deadlock between `require_corruption_control=True`, `corrupted_output=None`, and `ProfileArgs` | Added `--no-hold-fixed` (`ADVISORY_ONLY`) + strict non-exempt config inference when omitted; added `evaluate_kernel_parity_for_probe`, `corruption_result` arg, `"not_measured"` explicit state, `--no-require-corruption-control`, and `--min-work-retention-ratio`. |
| **R2-M1** | `MAJOR` | `report.py:81-83` skipped `_SKIP_SUFFIXES` on explicit paths too, breaking `render_report([diag_path])` | Split `_DISCOVERY_SKIP_SUFFIXES` (glob paths) vs `_EXPLICIT_SKIP_SUFFIXES` (explicit files) inside `discover_records`. |
| **R2-M2** | `MAJOR` | `tyro` parses `require_parity: bool \| None` as `{None,True,False}` value tokens instead of `--flag`/`--no-flag` switches | Changed `ProfileArgs` booleans to plain `bool` defaults (`require_parity=True`, `require_corruption_control=True`, `no_hold_fixed=False`) with full cross-mode validation. |
| **R2-M3** | `MAJOR` | Residual gaps: `baseline work_active_count == 0.0`, `require_parity=False` ignoring `"fail"`, invalid `step_time_basis`, Rule 3/6 `triage.py`, lazy-JAX in `kernel_probe.py` & `test_import_isolation.py` | Enforced `<= 0.0 -> REJECT_UNVERIFIABLE`, `_check_present_failure_witnesses`, valid `step_time_basis` set, `prod+diag` FP64 vregs & full Rule 6 claim guards, zero top-level `jax` in `kernel_probe.py`/`_diag_worker.py` + `sorted()` `_PACKAGES`. |
| **R2-Min** | `MINOR` | `int_mismatches`/PyTree rel-err & `ProbeRecord` NaN finiteness crash in `build_probe_parity_payload`, undefined `metric_deltas`, `v3.5–v3.9` wording | Checked `forward.ok` + `int_mismatches`, guarded non-finite metrics via `parity_nonfinite_metrics`, specified `metric_deltas`, aligned prose to verified `v3.4` ledger. |

## Adversarial Coherence Audit Log (Round 3 & Round 4)

| ID | Severity | Summary of Issue Caught | Resolution in v4.1 |
|---|---|---|---|
| **R3-SignOff** | `INFO` | Draft v3 passed `VERDICT: ACCEPT` (0 BLOCKING, 0 MAJOR); 3 minor polish items flagged | Folded `--mode {prod,diag,ptx}`, `emit_probe_record` provenance kwargs, `rel=1e-4`, `parity=None`, Rule 3 absence check, and `bypass_hold_fixed` into v4/v4.1. |
| **R4-B1** | `BLOCKING` | Non-unique `subject_key=f"probe:{probe_id}"` collapsed all kernel iterations into `N=1` in `cisternal.shadow.scoring` | Added per-measurement deterministic `make_triage_subject_key` + `resolve_triage_subject_key`, printed in `--triage` and resolved in `--compare`. |
| **R4-B2** | `BLOCKING` | Broadcasting one `KEEP`/`REVERT` across all 4 rules corrupted `FN`, `TN`, `base_rate`, and `fp_rate` | Required `--triage-category` alongside `--from-triage-probe`; skipped labeling on `REJECT_UNVERIFIABLE`/`ADVISORY_ONLY`; documented clean negative-control collection. |
| **R4-B3** | `BLOCKING` | `evaluate_gate` colon-suffixed failure reasons missed exact match + zero-negative `base_rate==1.0` trap + missing production `shadow.isolation` records | Normalized `reason.split(":", 1)[0]`, exempted `(fp+tn)==0` `base_rate==1.0` from false demotion, and added `attest_triage_shadow_isolation` (`check_roots=[live_root, shadow_sink]`). |
| **R4-M1** | `MAJOR` | `--telemetry-summary` crashed exit `2` on `evaluate_gate` warm-up refusal & missed corrupt JSONL lines | Added `TelemetryReadError` strict line parsing + `GateCheckError(exit_code=2, "REFUSAL:")` `"uncalibrated"` mapping + `repo_root` resolution. |
| **R4-M2** | `MAJOR` | `pyproject.toml:136` `cisternal>=0.1.7` vs `tests/telemetry/` importing `cisternal.shadow` (`>=0.1.11`) | Moved `pytest.importorskip("cisternal.shadow")` strictly inside live-shadow tests; kept fallback unit tests + `tier1_core` coverage intact. |
| **R4-M3** | `MAJOR` | `NOISE_FLOOR_BLOWN` `.verdict.json` double-count/sort, `INSTRUMENT_DEGRADED` flagging `"disabled"`, `GUARDRAIL_EROSION` flagging `--no-require-corruption-control` & double-counting `dual_pass` | Deduplicated verdicts against chronological `events`, restricted degraded status checks to real failures, excluded `bypass_corruption` from error erosion, scoped `n_cli_invocations`. |
| **R4-M4** | `MAJOR` | `profiling_telemetry_session(log_dir=...)` reused stale pipeline dir + `audit_dispatch_independence.py:50-54` forbidden string trap | Added active-exporter dir mismatch reset in `profiling_telemetry_session`, scrubbed forbidden substrings across `src/xtrax/`, added `just audit-dispatch-independence` gate. |
| **R4-Min1..3** | `MINOR` | Log file glob order (`events.*.jsonl*`), `top_kernel_pct` / `--target-missed` wiring, `Record.git_hash` async `None` caveat | Mirrored `LOG_FILE_RE` rotation order, aligned `top_kernel_pct` with `report.py:130-133` + `--target-missed`, used `fields` SHAs only. |
| **R5-M1..M2** | `MAJOR` | `str.strip("xtrax.triage.")` mangled `atomic_contention`/`register_spill_pressure`; missing explicit `emit_triage_shadow_verdicts` call site in CLI `--triage` | Switched to `.removeprefix("xtrax.triage.")` in `§3.6` & `§3.9.3`; wired `emit_triage_shadow_verdicts(prod, findings)` explicitly into `run_profile`'s `--triage` branch. |
| **R5-Min1..4** | `MINOR` | Isolation audit synthetic verdict inflation, `.verdict.json` `"status"` key normalization, `probe_id` `:` slug normalization, `cisternal shadow check` TOML path invocation | Tagged isolation verdicts `feature_version="__isolation_audit__"`, normalized `verdict_status or status`, hyphenated `slug`, documented TOML path CLI invocation. |

## Round 6 Final Sign-Off (`VERDICT: ACCEPT`)

- Final Round 6 `xtrax-spec-challenger` verification on `Draft v4.2` returned **`VERDICT: ACCEPT`** (**0 `BLOCKING`**, **0 `MAJOR`**, **0 `MINOR`** remaining) across all `xtrax`, `cisternal`, and `prolix` interfaces and audit gates.
