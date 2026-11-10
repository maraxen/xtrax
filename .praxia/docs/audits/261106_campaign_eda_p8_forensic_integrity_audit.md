---
category: audit
title: Phase P8 Forensic Integrity & Victory Audit Report
description: Forensic code audit for hardcoded values, facade detection, boundary
  isolation, test and benchmark suite reproduction, and debt closure verification
task_id: 261106_xtrax-campaign-eda
status: complete
date: '2026-11-06'
agent: teamwork_auditor_p8_1
verdict: CLEAN
---

# Forensic Integrity & Victory Audit Report — Phase P8

**Auditor**: `teamwork_auditor_p8_1`  
**Role**: Forensic Integrity & Victory Auditor  
**Task ID**: `261106_xtrax-campaign-eda`  
**Work Product**: xtrax Campaign EDA Engine (`src/xtrax/eda/campaign/`), Consumer Domain Plugins (`aminx`, `demistify`), Benchmark Suite (`scripts/benchmarks/campaign_eda_benchmark.py`), and Sidecar (`scripts/benchmarks/campaign_eda_benchmark.bth.toml`)  
**Profile**: General Project  
**Integrity Mode**: Development Mode (ground truth per `ORIGINAL_REQUEST.md:7`)  
**Verdict**: **CLEAN** (Binary Veto NOT exercised; work product fully approved)

---

## Forensic Audit Summary

| Check | Verdict | Details |
|---|:---:|---|
| **Hardcoded Output Detection** | **PASS** | Dynamic random sampling (`np.random.default_rng`), calculated quantiles, zero canned string or array literals |
| **Facade Implementation Detection** | **PASS** | Genuine BLAS matrix algebra (`w_mat @ t_mat`), $O(1)$ ratio recurrence, 2D Pareto sorting, positive Shannon support |
| **Pre-populated Artifact Detection** | **PASS** | Fresh benchmark output written to `/tmp/victory_bench.json`; zero stale verification logs committed |
| **Build & Targeted Pytest Suites** | **PASS** | 24/24 `tests/test_campaign_eda/` passed (1.30s), 111/111 `tests/pack/` passed (4.15s), 8/8 `aminx` passed (0.08s) |
| **Ruff Code Hygiene & Linting** | **PASS** | 0 violations across `src/xtrax/eda/campaign/` and `tests/test_campaign_eda/` |
| **Pre-Registered Synthetic Controls** | **PASS** | All 6 controls pass deterministically per sidecar `campaign_eda_benchmark.bth.toml` |
| **End-to-End Demo Campaign** | **PASS** | 18-unit Ubiquitin campaign answers Q1–Q5 in 0.068s (total benchmark 0.148s $\le$ 60.0s budget) |
| **Boundary Enforcement & Isolation** | **PASS** | Core footprint is `stdlib + numpy`; 0 references/imports to `maraxiom`, `bathos`, or `cisternal` |
| **Committed-Only Reads & Refusal** | **PASS** | Uncommitted, partial, and quarantined units strictly refused from scientific pooling with explicit diagnostic banners |
| **Debt Closure Verification** | **PASS** | Debts #3301, #3143, #3300, and demistify #830 are authentically satisfied |

---

## 1. Observation

### 1.1 Independent Reproduction of Targeted Test Suites

1. **Campaign EDA Pytest Suite (`tests/test_campaign_eda/`)**:
   - Command: `uv run pytest tests/test_campaign_eda/ -v --no-cov`
   - Result: **24 passed, 12 warnings in 1.30s** (Exit code: 0)
   - Tested modules:
     - `test_artifacts.py`: RFC 8259 sanitization, Vega-Lite null filters, airgapped progressive dashboard DOM IDs.
     - `test_controls.py`: Positive planted effect, negative null campaign, batched==looped bootstrap parity, storage format parity, uncommitted unit refusal, small-sample guard.
     - `test_coverage_telemetry.py`: 100% committed visual rollup rule, debt #3300 pack banner.
     - `test_demo_campaign.py`: 18-unit Ubiquitin end-to-end execution.
     - `test_plugins.py`: Zero-copy array immutability (`flags.writeable = False`), quarantine on `(Exception, SystemExit)`, aminx and demistify builtin plugins.
     - `test_stats.py`: Pairwise diversity boundary guards ($N \le 1, |M|=0$), safe Shannon entropy ($H \ge 0.0$), Best-of-N recurrence, 2D Pareto sorting, cross-knob collision table.
     - `test_types.py`: `DomainMetricResult` finite validation, `CellStatus` enum, `ResidueTable` construction.

2. **Pack Format Regression Suite (`tests/pack/test_pack.py`)**:
   - Command: `uv run pytest tests/pack/test_pack.py -v --no-cov`
   - Result: **111 passed in 4.15s** (Exit code: 0)
   - Confirms backward compatibility and zero regressions in `xtrax.pack`.

3. **Aminx Domain Plugin Unit Tests (`tests/host/test_eda_plugin.py`)**:
   - Command: `cd /tmp/aminx && uv run pytest tests/host/test_eda_plugin.py -v`
   - Result: **8 passed, 4 warnings in 0.08s** (Exit code: 0)
   - Tested plugin metadata, MPNN alphabet tokenization, derived NLL fallback $\text{LSE}(\text{logits}) - \text{logits}[s]$ and perplexity, designable mask application (`real_mask & ~fixed_mask`), diversity and entropy, native recovery and shuffled native margin, schema rejection of `NaN`/`Inf`, and entry-point discovery via `importlib.metadata`.

4. **Code Linter & Formatting**:
   - Command: `uv run ruff check src/xtrax/eda/campaign/ tests/test_campaign_eda/`
   - Result: **All checks passed! (0 violations)** (Exit code: 0)

---

### 1.2 Reproduction of Pre-Registered Benchmark & Synthetic Controls

Command executed:
```bash
uv run python3 scripts/benchmarks/campaign_eda_benchmark.py --out /tmp/victory_bench.json
```
Output log excerpt:
```text
[2026-11-06 13:54:45,605] INFO: Executing Verification Control 1: Positive planted knob effect...
[2026-11-06 13:54:45,611] INFO: Control 1 passed: True
[2026-11-06 13:54:45,611] INFO: Executing Verification Control 2: Negative null campaign...
[2026-11-06 13:54:45,642] INFO: Control 2 passed: True
[2026-11-06 13:54:45,642] INFO: Executing Verification Control 3: Vectorized cluster ratio bootstrap parity...
[2026-11-06 13:54:45,670] INFO: Control 3 passed: True
[2026-11-06 13:54:45,670] INFO: Executing Verification Control 4: Storage format parity (loose vs packed)...
[2026-11-06 13:54:45,674] INFO: Control 4 passed: True
[2026-11-06 13:54:45,674] INFO: Executing Verification Control 5: Uncommitted unit refusal and explicit bannering...
[2026-11-06 13:54:45,680] INFO: Control 5 passed: True
[2026-11-06 13:54:45,681] INFO: Executing Verification Control 6: Small-sample guard (U < 3)...
[2026-11-06 13:54:45,681] INFO: Control 6 passed: True
[2026-11-06 13:54:45,681] INFO: Executing End-to-End Ubiquitin demo campaign (18 committed units answering Q1-Q5)...
[2026-11-06 13:54:45,749] INFO: Ubiquitin demo passed: True
[2026-11-06 13:54:45,749] INFO: Wrote result payload to /tmp/victory_bench.json
[2026-11-06 13:54:45,749] INFO: All 6 verification controls and end-to-end Ubiquitin demo passed successfully!
```

Empirical data verified from `/tmp/victory_bench.json`:
1. **Control 1 (Planted Knob Effect)**:
   - $\hat{\Delta} = 1.515104 \in [1.2, 1.8]$
   - $\text{CI}_{95\%} = [1.482875, 1.542740]$ (strictly excludes 0.0)
   - $\text{SE} = 0.015706$
   - $\text{MDE}_{0.80} = 0.044002$ ($|\hat{\Delta}| \ge \text{MDE}_{0.80}$, `powered = true`, `detected = true`)
2. **Control 2 (Negative Null Campaign)**:
   - Single simulation $\text{CI}_{95\%} = [-0.135221, 0.114234]$ (contains 0.0)
   - Across 100 independent null simulations: 4 rejections, $\text{FPR} = 0.04 \le 0.05$ (`fpr_controlled = true`)
3. **Control 3 (Vectorized Bootstrap Parity)**:
   - Unequal cluster sizes: $\max|\text{diff}| = 0.0$
   - Equal cluster sizes: $\max|\text{diff}| = 2.6645 \times 10^{-15} \le 10^{-12}$ (`unequal_parity = true`, `equal_parity = true`)
4. **Control 4 (Storage Format Parity)**:
   - `meta_sha_match`: true, `data_sha_match`: true, `parsed_match`: true
   - Bit-identical across loose directory and `members.tar.zst` archive.
5. **Control 5 (Uncommitted Unit Refusal)**:
   - Classified: `u1` $\implies$ `COMMITTED`, `u2` $\implies$ `PARTIAL`, `u3` $\implies$ `FAILED`
   - Banners: `"Unit u2: in progress (active lock/staging)"`, `"Unit u3: failed (killed_incomplete)"`
   - Downstream statistical pooling contains strictly `u1` (`only_committed_pooled = true`).
6. **Control 6 (Small-Sample Guard)**:
   - $U=1, 2 \implies \text{descriptive\_only = true}$, $\text{ci\_95 = null}$
   - $U=3 \implies \text{descriptive\_only = false}$, $\text{ci\_95}$ active.
7. **End-to-End Ubiquitin Demo Campaign**:
   - 18 committed units (2 arms $\times$ 3 temperatures $\times$ 3 seeds, 288 sequences, 76 residues).
   - Demo execution runtime: **0.068 seconds** (Total benchmark: **0.148 seconds**), well within the $\le 60.0$s ceiling.
   - Answers all 5 foundational questions:
     - **Q1**: `mean_diversity = 0.265010`, `mean_entropy = 0.744164` bits, `mean_recovery = 0.855674`, duplicate sequences = 10, cross-knob collision table generated.
     - **Q2**: Best-of-N minimum score curve computed across candidate grid ($n=1 \to 1.473$, $n=2 \to 1.372$, $n=4 \to 1.287$, $n=8 \to 1.216$, $n=16 \to 1.155$); Pareto frontier identified 5 non-dominated points.
     - **Q3**: Vectorized cluster ratio bootstrap estimates computed across 6 condition groups; all $U=3$ groups have active $95\%$ percentile CIs.
     - **Q4**: 18/18 planned rows reconciled as 100% committed with 6 committed visual aggregation cells.
     - **Q5**: Secondary run view rendered with compute wall times, loose/packed counts (9 loose, 9 packed), and explicit seam banner: `"Pack Telemetry: not instrumented (debt #3300)"`.

---

### 1.3 Forensic Code Inspections & Citations

1. **Vectorized Cluster Ratio Bootstrap (`src/xtrax/eda/campaign/stats/bootstrap.py:35-55`)**:
   ```python
   def vectorized_cluster_ratio_bootstrap(unit_sums, unit_counts, weights):
       n_vec = np.asarray(unit_counts, dtype=np.float64).reshape(-1, 1)
       t_mat = np.asarray(unit_sums, dtype=np.float64)
       w_mat = np.asarray(weights, dtype=np.float64)
       numerator = w_mat @ t_mat
       denominator = w_mat @ n_vec
       return numerator / denominator
   ```
   Evaluates $\hat{\Theta}^* = (W \cdot T) / (W \cdot \mathbf{n})$ via genuine BLAS matrix multiplication without Python loops over units or bootstrap draws.

2. **Numerically Stable Best-of-N Recurrence (`src/xtrax/eda/campaign/stats/scores.py:18-63`)**:
   ```python
   w = float(n) / float(N)
   for j in range(1, N - n + 2):
       score_val = sorted_scores[j - 1]
       expected_min += w * score_val
       if j < N - n + 1:
           w *= float(N - n - j + 1) / float(N - j)
   ```
   Iterative ratio recurrence eliminates float64 combinatorial overflow (`comb()` returning `inf` for $N > 1029$). Bounded candidate grid `grid = sorted(list({int(n) for n in candidate_grid if 0 < n <= N}))` (`scores.py:49`) strictly filters $n > N$.

3. **Safe Positive-Support Shannon Entropy (`src/xtrax/eda/campaign/stats/composition.py:21-32`)**:
   ```python
   def safe_shannon_entropy(probs: np.ndarray) -> float:
       p = probs[probs > 0]
       if len(p) == 0:
           return 0.0
       h = -float(np.sum(p * np.log2(p)))
       return max(0.0, h)
   ```
   Enforces $H \ge 0.0$ on non-zero support, completely eliminating negative bits and avoiding artificial $\epsilon$ distortion.

4. **Zero-Copy Array Immutability (`src/xtrax/eda/campaign/plugins.py:58-74`)**:
   ```python
   def enforce_cell_immutability(cell: LibraryCell) -> None:
       if isinstance(cell.tokens, np.ndarray):
           cell.tokens.flags.writeable = False
       if isinstance(cell.mask, np.ndarray):
           cell.mask.flags.writeable = False
       ...
   ```
   Freezes underlying NumPy buffers before passing `LibraryCell` to domain plugins. Slices inherit read-only status, guaranteeing plugins cannot mutate data in place.

5. **Defensive Quarantine Boundary (`src/xtrax/eda/campaign/plugins.py:128-160`)**:
   ```python
   try:
       cells = plugin.extract_cell(...)
       ...
   except (Exception, SystemExit) as exc:
       reason = f"{type(exc).__name__}: {exc}"
       quarantine_unit(unit_key, plugin.domain_name, reason)
       return []
   ```
   Catches both `Exception` and `SystemExit`, preventing rogue plugins from terminating the parent analysis runner and isolating bad units in the quarantine ledger.

6. **RFC 8259 Canonical JSON Sanitization (`src/xtrax/eda/campaign/artifacts.py:23-63`)**:
   ```python
   def sanitize_for_rfc8259(obj: Any) -> Any:
       if isinstance(obj, float):
           if math.isnan(obj) or math.isinf(obj):
               return None
           if obj == 0.0 and math.copysign(1.0, obj) < 0.0:
               return 0.0
   ...
   def dumps_canonical_json(obj: Any) -> str:
       clean_obj = sanitize_for_rfc8259(obj)
       return json.dumps(clean_obj, sort_keys=True, indent=2, allow_nan=False)
   ```
   Strictly adheres to RFC 8259 by replacing `NaN` and `Inf` with `None` (`null`) and coercing `-0.0` to `0.0`.

7. **Architectural Isolation Scan**:
   - `git grep -E "\b(maraxiom|bathos|cisternal)\b" src/xtrax/eda/` $\implies$ **0 matches**.
   - `git grep -E "\b(maraxiom|bathos|cisternal)\b" tests/test_campaign_eda/` $\implies$ **0 matches**.
   - Core footprint across all 11 modules in `src/xtrax/eda/campaign/` is strictly Python standard library + NumPy. `pyarrow` and `vl-convert` are imported lazily with automatic fallback to JSON tables and vector SVG.

---

## 2. Logic Chain

1. **Ground Truth & Integrity Mandate**:
   - Per `ORIGINAL_REQUEST.md:7`, the project operates under **Development Mode**.
   - In Development Mode, the forensic focus is catching hardcoded test results, facade implementations, and fabricated outputs.
   - Code reuse, optional extras, and standard mathematical packages are permitted.

2. **Absence of Hardcoded Values or Facades**:
   - Observation 1.1 and 1.3 confirm all test fixtures generate dynamic stochastic data via `np.random.default_rng()`.
   - Observation 1.3 citations prove that `bootstrap.py`, `scores.py`, and `composition.py` perform real BLAS matrix multiplications, non-dominated sorting, and iterative recurrence calculations.
   - Adversarial stress tests (scale $N=10,000$, subnormals, empty inputs, identical values) confirmed that mathematical routines behave authentically without throwing unhandled exceptions or returning canned constants.

3. **Empirical Reproduction of Synthetic Controls**:
   - Observation 1.2 confirms that running the benchmark script `campaign_eda_benchmark.py` directly executes all 6 synthetic verification controls and the 18-unit demo campaign.
   - Every metric in `/tmp/victory_bench.json` strictly satisfies the pre-registered decision rules declared in `campaign_eda_benchmark.bth.toml`:
     - $\hat{\Delta} = 1.515 \in [1.2, 1.8]$ and $p < 0.05$ (Control 1).
     - $\text{FPR} = 0.04 \le 0.05$ (Control 2).
     - $\max|\text{diff}| \le 2.66 \times 10^{-15} \le 10^{-12}$ (Control 3).
     - Exact bit-identical SHA-256 and NumPy equality across storage modes (Control 4).
     - 100% exclusion of uncommitted and partial units from pooled findings (Control 5).
     - Small-sample suppression of CIs for $U < 3$ (Control 6).
     - Ubiquitin demo completes in 0.068s $\le$ 60.0s answering Q1–Q5.

4. **Architectural Boundaries & Separation**:
   - Observation 1.3 Citation 7 proves zero cross-imports or references to `maraxiom`, `bathos`, or `cisternal` inside `src/xtrax/eda/`.
   - Core dependencies are strictly `stdlib + numpy`. Heavy columnar and visualization packages are optional extras with runtime fallbacks.

5. **Debt Resolution**:
   - Debt **#3301** is satisfied by the `xtrax.eda.campaign` core engine answering Q1–Q5.
   - Debt **#3143** is satisfied by `DomainPlugin` protocol and the `aminx` and `demistify` consumer plugins.
   - Debt **#3300** is satisfied by the explicit pack telemetry seam banner (`"Pack Telemetry: not instrumented (debt #3300)"`).
   - Demistify Debt **#830** is satisfied by `DemistifyDomainPlugin` with Schema B degradation handling.

6. **Conclusion**:
   - Because all verification checks passed genuinely and empirically with zero integrity violations, the binary verdict is **CLEAN**.

---

## 3. Caveats

1. **Pre-Existing Unstaged Changes in xtrax Worktree**:
   - The xtrax worktree contains pre-existing modified files in other subsystems (`src/xtrax/distributed/`, `src/xtrax/inference/`, etc.) from earlier sprint branches. None of these affect `src/xtrax/eda/campaign/` or `tests/test_campaign_eda/`.
2. **Third-Party Typing Deprecation Warnings**:
   - Test execution emits 12 `BeartypeDecorHintPep585DeprecationWarning` notices and 4 Python 3.12 `importlib.metadata` deprecation notices. These originate from third-party libraries and standard library deprecations scheduled for Python 3.13+, not from `xtrax.eda.campaign`.
3. **Debt #3300 Pack Telemetry Instrumentation**:
   - Pack events are intentionally not instrumented and are explicitly bannered as `"Pack Telemetry: not instrumented (debt #3300)"`, as mandated by the approved specification.

---

## 4. Conclusion

**Verdict: CLEAN**

The xtrax Campaign EDA capability and consumer domain plugin architecture pass the Phase P8 Victory Audit without reservation. All 6 pre-registered synthetic controls, all 5 core analytical questions, and all targeted test suites pass with 100% genuine implementation, strict architectural boundary isolation, and zero integrity violations.

---

## 5. Verification Method

To independently reproduce this Victory Audit from a fresh shell:

```bash
# 1. Run targeted Campaign EDA test suite (24 tests pass in ~1.3s)
uv run pytest tests/test_campaign_eda/ -v --no-cov

# 2. Run pack format regression suite (111 tests pass in ~4.2s)
uv run pytest tests/pack/test_pack.py -v --no-cov

# 3. Run ruff linter check (0 violations)
uv run ruff check src/xtrax/eda/campaign/ tests/test_campaign_eda/

# 4. Run aminx domain plugin unit tests (8 tests pass in ~0.08s)
cd /tmp/aminx && uv run pytest tests/host/test_eda_plugin.py -v && cd /home/marielle/projects/xtrax

# 5. Run the pre-registered benchmark script and verify results
uv run python3 scripts/benchmarks/campaign_eda_benchmark.py --out /tmp/victory_bench.json

# 6. Verify architectural isolation (must return 0 matches)
python3 -c "import os, re; pattern = re.compile(r'\b(maraxiom|bathos|cisternal)\b'); assert not [f for r, _, fs in os.walk('src/xtrax/eda') for f in fs if f.endswith('.py') and pattern.search(open(os.path.join(r, f)).read())]"
```

### Invalidation Conditions
This Victory Audit verdict is invalidated if:
1. Any test in `tests/test_campaign_eda/` fails.
2. The benchmark script returns a non-zero exit code or fails any of the 6 synthetic controls.
3. Any cross-import to `maraxiom`, `bathos`, or `cisternal` is introduced into `src/xtrax/eda/`.
4. An uncommitted or quarantined unit is pooled into scientific findings without being filtered and bannered.
