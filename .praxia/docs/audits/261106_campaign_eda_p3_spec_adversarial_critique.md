---
category: audit
title: 'Phase P3 Adversarial Stress-Test & Critique Report: xtrax Campaign EDA'
description: Adversarial stress-test and critique of the Campaign EDA specification
  (r1.0), identifying 7 blockers, 8 majors, and 3 minors including cluster ratio bootstrap
  bias and demo small-sample deadlock
task_id: 261106_xtrax-campaign-eda
status: complete
date: '2026-11-06'
agent: teamwork_challenger_1
verdict: CHALLENGE
target_specification: .praxia/docs/specs/261106_campaign-eda-redesign.md
---

# Phase P3 Adversarial Stress-Test & Critique Report: xtrax Campaign EDA

**Agent Identity**: `teamwork_challenger_1`  
**Role**: Campaign EDA Adversarial Challenger  
**Specification Under Review**: `.praxia/docs/specs/261106_campaign-eda-redesign.md` (r1.0)  
**Parent Orchestrator**: `8c6c9b8f-26e4-41f2-9b35-adf51fb4ef0f`  
**Timestamp**: 2026-11-06T20:25:00Z  

---

## 1. Observation

Direct empirical observations, file:line citations, mathematical formulas, and reproduction commands executed against the specification, codebase, and numerical environment:

### Observation 1.1: Vectorized Cluster Bootstrap Mathematical Bias on Unequal Cluster Sizes
- **Location**: `.praxia/docs/specs/261106_campaign-eda-redesign.md`, lines 468–477 (§5.3.2).
- **Verbatim Text**:
  ```python
  # Spec Eq: \hat{\Theta}^* = (1 / U) * (W @ Y)
  # W ~ Multinomial(U, [1/U, ..., 1/U])
  # Y in R^{U x D} where y_u is observable summary vector per unit
  ```
- **Empirical Execution**: Executed `uv run python` test with cluster sizes $N = [100, 10, 10]$ and cluster means $\mu = [10.0, 2.0, 2.0]$.
- **Result**:
  - True pooled population mean: $\frac{100 \times 10.0 + 10 \times 2.0 + 10 \times 2.0}{120} = 8.6667$.
  - Spec bootstrap estimator mean: $4.6923$.
  - Bias: $-3.9744$ (an error of $45.8\%$). The bootstrap distribution is severely shifted away from the true point estimator.

### Observation 1.2: Demo Campaign Replicate Count ($U=2$) Collides with Small-Sample Guard ($U < 3$)
- **Location**: Spec lines 480–483 (§5.3.3) vs. lines 668–671 (§8).
- **Verbatim Text (§5.3.3)**:
  > *"When replicate count $U < 3$: mark `descriptive_only = True`, suppress CI error bars in visualization specifications, and emit the warning: 'Warning: Replicate count U < 3; reporting descriptive point estimates only.'"*
- **Verbatim Text (§8)**:
  > *"Knob Grid: temperature: `[0.1, 0.3, 0.5]`; seed: `[1001, 1002]` (2 independent replication seeds)."*
- **Empirical Deduction**: Replicate count $U = 2$. Because $2 < 3$, every single cell in the real Ubiquitin demo campaign will trigger `descriptive_only = True` and suppress CI intervals, making it impossible for the demo to demonstrate Question 3's core deliverable or fulfill Acceptance Criterion AC-10.

### Observation 1.3: Best-of-$N$ Formula Float Overflow and $n > N$ Arithmetic Failure
- **Location**: Spec lines 442–446 (§5.2.2).
- **Verbatim Text**:
  $$\mathbb{E}\left[ \min_{i=1..n} S_i \right] = \sum_{j=1}^{N - n + 1} S_{(j)} \frac{\binom{N - j}{n - 1}}{\binom{N}{n}}$$
  > *"Evaluated across a geometric grid of candidate sizes $n \in \{1, 2, 4, 8, 16, 32, 64, \dots, N\}$."*
- **Empirical Execution**:
  1. For $N = 1100, n = 500$: `scipy.special.comb(1100, 500)` returns `inf`. `inf / inf` yields `nan` with `RuntimeWarning: invalid value encountered in scalar divide`.
  2. For $n > N$ (e.g. $N = 16$ in ubiquitin demo, while grid evaluates $n = 32, 64$): $\binom{N}{n} = 0$, causing either `ZeroDivisionError` or returning silent zero (as empty generator sum = 0.0), invalidating the score curve.

### Observation 1.4: Pairwise Identity Diversity Evaluates to `NaN` for $N \le 1$ or Fixed Residues ($|M|=0$)
- **Location**: Spec lines 392–403 (§5.1.1).
- **Verbatim Text**:
  ```python
  triu_indices = np.triu_indices(N, k=1)
  pairwise_identities = pid_matrix[triu_indices]
  mean_diversity = float(1.0 - np.mean(pairwise_identities))
  ```
- **Empirical Execution**:
  1. For $N = 1$: `triu_indices` has length 0. `np.mean([])` raises `RuntimeWarning: Mean of empty slice` and returns `nan`.
  2. For $|M| = 0$ (all residues fixed): `np.sum(match, axis=-1) / 0` raises `RuntimeWarning: invalid value encountered in divide` and returns `nan`.

### Observation 1.5: RFC 8259 JSON Non-Compliance and Browser Parser Crash
- **Location**: Spec lines 87 (§1.3), 177 (§3.1), and 540 (§6.1).
- **Verbatim Text**:
  > *"Canonical RFC 8259 JSON summaries... Floating-point normalization: Python `float` and NumPy floating values equal to `-0.0` are normalized strictly to `0.0`."*
- **Empirical Execution**: Python `json.dumps({"diversity": float("nan")})` emits `{"diversity": NaN}`. In strict RFC 8259 mode (`json.dumps(..., allow_nan=False)`), Python raises `ValueError: Out of range float values are not JSON compliant: nan`. In standard JavaScript engines (e.g. browser `JSON.parse()`), `JSON.parse('{"diversity": NaN}')` immediately throws `SyntaxError: Unexpected token N in JSON`.

### Observation 1.6: Domain Plugin Quarantine Bypassed by `SystemExit`
- **Location**: Spec lines 335–341 (§4.2).
- **Verbatim Text**:
  ```python
  try:
      cell_data = plugin.extract_cell(store_path, row, schema)
  except Exception as exc:
      logger.warning(...)
      self.quarantine_unit(...)
  ```
- **Empirical Execution**: A plugin raising `SystemExit("fatal error")` bypasses `except Exception:` completely, terminating the Python runtime immediately without quarantine.

### Observation 1.7: Record Name Disconnect Between `xtrax.pack` and `xtrax.campaign`
- **Location**: `src/xtrax/pack/_node.py:649` vs. `src/xtrax/campaign/marker.py:363`.
- **Verbatim Text**:
  - `read_unit_member(unit_dir: PathLike, relpath: str, *, record_name: str = UNIT_RECORD)` where `UNIT_RECORD = "unit.json"`.
  - In `src/xtrax/campaign/marker.py`: the completion record is named `<name>.done.json` with schema `"campaign_done_marker_v3"`.
- **Empirical Deduction**: If `read_unit_member` is invoked on a packed campaign unit without passing `record_name="done.json"`, `_read_record` returns `None`, the function falls through to the loose file read, and raises `FileNotFoundError` if the unit was swept.

### Observation 1.8: Missing Ubiquitin Target Fixture `tests/data/1ubq.pdb`
- **Location**: Spec line 664 (§8) and `P0_RECON_SYNTHESIS.md` line 76.
- **Empirical Execution**: Command `ls tests/data/1ubq.pdb` returned `No such file or directory`. `tests/data/` exists but contains test modules (`test_pipeline.py`, etc.), with zero PDB fixtures.

### Observation 1.9: Shannon Entropy Formula Negative Bit Generation
- **Location**: Spec lines 408 and 412 (§5.1.2).
- **Verbatim Text**:
  $$H(l) = -\sum_{c=0}^{19} p_l(c) \log_2 \left( p_l(c) + \epsilon \right)$$
- **Empirical Execution**: For a deterministic position ($p_0 = 1.0, p_{1..19} = 0.0$) with $\epsilon = 10^{-6}$, $-1.0 \times \log_2(1.0 + 10^{-6}) = -1.4427 \times 10^{-6}$ bits (negative entropy). For $\epsilon = 0$, evaluates to `nan` with `RuntimeWarning: divide by zero encountered in log2`.

### Observation 1.10: Airgapped HTML Parser Blocking and Duplicate DOM IDs
- **Location**: Spec lines 587–615 (§6.2).
- **Verbatim Text**: Script template defines `<div id="fallback-svg">` and `<div id="interactive-vis">` repeated across panels, with an external CDN script tag in `<head>`.
- **Empirical Deduction**: Synchronous `<script src="https://cdn...">` tags block the browser parser for 30–90 seconds on offline HPC nodes before TCP timeout. Duplicate IDs cause `document.getElementById('fallback-svg')` to return only the first panel's element, breaking 4 out of 5 panels.

---

## 2. Logic Chain

1. **Bootstrap Uncertainty Validity**:
   - The spec specifies estimating replicate uncertainty via $\hat{\Theta}^* = \frac{1}{U} (W \cdot Y)$ (Obs 1.1).
   - This formula assigns equal weight $\frac{1}{U}$ to each unit observable $\mathbf{y}_u$.
   - When units contain unequal sample sizes $N_u$ (arising from chunking, filtering, or varied sampling), the pooled mean is a ratio estimator $\frac{\sum N_u \mathbf{y}_u}{\sum N_u}$.
   - Because $W \cdot Y / U$ ignores $N_u$, the resampled distribution has an expected value $\frac{1}{U} \sum \mathbf{y}_u$, which deviates significantly from the true pooled sample mean (measured 46% bias in Obs 1.1).
   - Therefore, the spec's vectorized bootstrap formula is mathematically invalid for general campaigns and must be reformulated as a ratio estimator $\frac{W \cdot T}{W \cdot \mathbf{n}}$.

2. **Demo Feasibility & Acceptance Criteria Deadlock**:
   - The spec mandates that replicate counts $U < 3$ suppress CI intervals and set `descriptive_only = True` (Obs 1.2).
   - The spec simultaneously defines the official Ubiquitin demo campaign with $U = 2$ seeds (`seed: [1001, 1002]`) (Obs 1.2).
   - Because $2 < 3$, running the demo campaign will trigger interval suppression on all conditions.
   - Consequently, the demo campaign cannot answer Question 3 (honest uncertainty intervals) or satisfy Acceptance Criteria AC-10 and AC-16.

3. **Numerical Robustness & Serialization Integrity**:
   - Real-world campaigns frequently contain edge cells with $N = 1$ sample or completely fixed proteins $|M| = 0$.
   - In both cases, the spec's diversity formula evaluates to `NaN` (Obs 1.4).
   - Similarly, Best-of-N evaluates to `NaN` for $N > 1029$ or crashes when $n > N$ (Obs 1.3).
   - When metrics evaluate to `NaN` or `Inf`, serializing to JSON produces unquoted `NaN` values unless sanitized (Obs 1.5).
   - These non-standard tokens violate RFC 8259 and cause standard browser `JSON.parse()` to throw a fatal `SyntaxError`, completely disabling the interactive dashboard.

4. **Quarantine & Storage Boundaries**:
   - Defensive isolation relies on catching `Exception` (Obs 1.6).
   - In Python, `SystemExit` bypasses `Exception`, allowing a plugin to abort the runner process without logging a quarantine event.
   - Concurrently, storage parity relies on `xtrax.pack.read_unit_member`, which defaults to looking for `unit.json`, while the campaign runner writes `<output_path>.done.json` (Obs 1.7).
   - Without harmonizing the record filename, packed units cannot be read by Campaign EDA.

---

## 3. Categorized Adversarial Findings & Attack Vector Analysis

### [BLOCKER] Findings (Must be resolved before Phase P5 implementation begins)

#### BLK-01: Vectorized Cluster Bootstrap Bias on Unequal Cluster Sizes
- **Vector**: 1 (Question 3: Honest Uncertainty)
- **Vulnerability**: §5.3.2 computes $\hat{\Theta}^* = \frac{1}{U} (W \cdot Y)$, which assumes equal sample count across all replicate units. In practice, units often have unequal numbers of valid samples (e.g. chunking, invalid sequence filtering, or dynamic stopping). In this setting, the spec's formula produces massive statistical bias (demonstrated 46% error in Obs 1.1) and undercovers the true mean.
- **Counterexample**: Unit 1 has 100 samples with mean 10.0; Unit 2 has 10 samples with mean 2.0; Unit 3 has 10 samples with mean 2.0. True mean is 8.67. Spec bootstrap converges to 4.69.
- **Remediation Requirement**:
  Reformulate §5.3.2 to use the vectorized ratio cluster bootstrap:
  $$\hat{\Theta}^* = \frac{W \cdot T}{W \cdot \mathbf{n}}$$
  where $T \in \mathbb{R}^{U \times D}$ is the matrix of unit sum statistics ($T_{u, :} = \sum_{i=1}^{N_u} \mathbf{x}_{u, i}$), $\mathbf{n} = [N_1, \dots, N_U]^T \in \mathbb{N}^U$ is the sample size vector, and $W \in \mathbb{N}^{B \times U}$ is the multinomial cluster weight matrix. When $N_1 = \dots = N_U$, this reduces identically to $W \cdot Y / U$, but remains exact when cluster sizes vary.

#### BLK-02: Demonstration Campaign ($U=2$) Collides with Small-Sample Guard ($U < 3$)
- **Vector**: 1 (Question 3 & Demo Verification)
- **Vulnerability**: §5.3.3 mandates that replicate counts $U < 3$ suppress CI intervals and set `descriptive_only = True`. §8 pins the official Ubiquitin demo campaign to 2 replication seeds (`seed: [1001, 1002]`, $U=2$). Under this configuration, the demo campaign is mathematically prohibited from generating confidence intervals, causing the P8 victory audit to fail.
- **Remediation Requirement**:
  Update §8 to specify 3 replication seeds (`seed: [1001, 1002, 1003]`, $U=3$). Total workload becomes 18 units $\times$ 16 samples = 288 sequences, running in $< 45$ seconds on local CPU.

#### BLK-03: Best-of-$N$ Numerical Overflow ($N > 1029$) and Division by Zero ($n > N$)
- **Vector**: 1 (Question 2: Scores & Energetics)
- **Vulnerability**: The analytical formula in §5.2.2 evaluates $\binom{N - j}{n - 1} / \binom{N}{n}$. For $N > 1029$, $\binom{N}{n}$ overflows float64 to `inf`, yielding `inf / inf = nan`. For $n > N$ (evaluated on small pools like $N=16$), $\binom{N}{n} = 0$, producing division by zero or erroneous 0.0 values.
- **Remediation Requirement**:
  1. Mandate that the evaluation grid of candidate sizes is strictly bounded: $n \in \{2^k \mid 2^k \le N\} \cup \{N\}$.
  2. Evaluate order statistic weights in log-space using `scipy.special.gammaln` or iterative ratio recurrence:
     $$\omega_j(n, N) = \frac{n}{N} \prod_{k=0}^{j-2} \frac{N - n - k}{N - 1 - k}$$
     preventing intermediate float overflow.

#### BLK-04: Pairwise Diversity Division by Zero on $N \le 1$ or Fixed Scaffold ($|M|=0$)
- **Vector**: 1 (Question 1: Library Composition)
- **Vulnerability**: §5.1.1 takes the mean of the upper triangle of the pairwise identity matrix. If $N \le 1$, the upper triangle has 0 elements, producing `NaN`. If $|M| = 0$ (fully pinned protein), dividing by $|M|$ yields `NaN`.
- **Remediation Requirement**:
  Add explicit boundary guards:
  - If $N \le 1$: set `mean_diversity = 0.0`, `flag_single_sample = True`.
  - If $|M| = 0$: set `mean_diversity = 0.0`, `flag_no_designable = True`.

#### BLK-05: RFC 8259 JSON Non-Compliance on `NaN` / `Infinity` Crashing Client
- **Vector**: 5 (Edge Cases & Numerical Stability)
- **Vulnerability**: Any calculation yielding `NaN` or `Inf` emits unquoted `NaN` in Python's default `json.dumps()`. This violates RFC 8259 and crashes browser `JSON.parse()` with a fatal syntax error.
- **Remediation Requirement**:
  In `xtrax.eda.campaign.export`, implement a canonical JSON serializer that normalizes `-0.0` to `0.0`, converts `NaN` and `+/-Inf` strictly to `null`, and validates with `allow_nan=False`.

#### BLK-06: Domain Plugin Quarantine Escape via `SystemExit`
- **Vector**: 2 (Domain Plugin Architecture)
- **Vulnerability**: §4.2 catches `except Exception:`. If a buggy plugin executes `sys.exit()` or raises `SystemExit`, the Python process terminates immediately, bypassing quarantine.
- **Remediation Requirement**:
  Catch `(Exception, SystemExit)` in plugin wrappers:
  ```python
  try:
      cell_data = plugin.extract_cell(...)
  except (Exception, SystemExit) as exc:
      self.quarantine_unit(row.manifest_row_hash, plugin.domain_name, str(exc))
  ```

#### BLK-07: Storage Parity Filename Mismatch (`unit.json` vs. `done.json`)
- **Vector**: 3 (Storage Parity & Unit Index)
- **Vulnerability**: `xtrax.pack.read_unit_member()` defaults to `record_name="unit.json"`. The campaign runner records completion in `<output_path>.done.json` (`campaign_done_marker_v3`). Calling `read_unit_member` without explicitly overriding `record_name` fails to discover the pack and crashes with `FileNotFoundError` on swept units.
- **Remediation Requirement**:
  In `src/xtrax/eda/campaign/reader.py`, define a unified reader that inspects unit directory markers and explicitly passes `record_name="done.json"` (or detects `done.json` vs `unit.json`) to `read_unit_member()`.

---

### [MAJOR] Findings (Significant defects that impede operation or cause silent data corruption)

#### MAJ-01: Memory Duplication via Deep-Copy Isolation
- **Vector**: 2 (Plugin Quarantine & Memory)
- **Vulnerability**: §4.2 specifies "Deep-Copy Isolation" to prevent in-place mutation. For large campaigns with thousands of sequences and dense logits ($N \times L \times 21$), deep copying creates hundreds of megabytes of redundant allocations.
- **Remediation Requirement**:
  Replace `copy.deepcopy` with zero-copy immutability: set `array.flags.writeable = False` on all NumPy arrays in `LibraryCell`. This makes in-place mutation attempt raise `ValueError: assignment destination is read-only` at zero memory cost.

#### MAJ-02: PyArrow Export Crash on Non-Conforming Plugin Metric Types
- **Vector**: 2 (Plugin Quarantine & Table Export)
- **Vulnerability**: If a plugin emits a non-float metric value (string, None, list) in `DomainMetricResult.metrics`, PyArrow crashes during Parquet conversion with `ArrowInvalid`, terminating the EDA run outside the quarantine handler.
- **Remediation Requirement**:
  Validate `DomainMetricResult` strictly upon receipt: confirm that all metric values are `isinstance(v, (int, float))` and finite. If invalid, quarantine the unit immediately.

#### MAJ-03: Killed Slurm / OOM Tasks Misclassified as MISSING
- **Vector**: 1 (Question 4: Coverage Reconciliation)
- **Vulnerability**: A worker killed by SIGKILL (OOM or walltime limit) leaves an unsealed staging directory or an expired lock with no `done.json`. §5.4 classifies this as `MISSING` (as if it was never planned or started).
- **Remediation Requirement**:
  In §5.4, inspect the filesystem for abandoned staging directories (`*.tmp`), expired locks, or non-zero executor exit records. Classify these units as `FAILED` (or `KILLED_INCOMPLETE`), reserving `MISSING` exclusively for planned rows with zero disk traces.

#### MAJ-04: Conflation of Manifest Rows with Visual Grid Cells
- **Vector**: 1 (Question 4: Sweep Grid Visualization)
- **Vulnerability**: A Cartesian grid condition (e.g. arm $\times$ temperature) contains multiple manifest rows (seeds and chunks). §5.4 states that every row in $\mathcal{G}$ is plotted, without defining how multiple rows roll up into a single visual grid cell.
- **Remediation Requirement**:
  Specify the cell rollup function: a grid cell is `COMMITTED` iff 100% of its constituent seeds/chunks are committed; `PARTIAL` if at least one is in progress; `FAILED` if any row failed/quarantined; and `MISSING` if no row has started.

#### MAJ-05: Missing Ubiquitin Target Fixture `tests/data/1ubq.pdb`
- **Vector**: 3 & 4 (Fixtures & E2E aminx Demo)
- **Vulnerability**: §8 relies on `tests/data/1ubq.pdb` as an existing repository file. Empirical search confirmed this file does not exist in the repository.
- **Remediation Requirement**:
  In work package P5-09/P5-10, add a task to vendor the standard 76-residue Ubiquitin PDB fixture into `tests/fixtures/1ubq.pdb` (or embed synthetic coordinates), avoiding unresolvable `FileNotFoundError`.

#### MAJ-06: Undeclared Optional Extras in `pyproject.toml`
- **Vector**: 4 (Dependency Boundaries)
- **Vulnerability**: The spec relies on extras `xtrax[eda-campaign]` (DuckDB + PyArrow) and `xtrax[eda-viz]` (Altair + vl-convert) and entry-point group `[project.entry-points."xtrax.eda.domains"]`. None of these exist currently in `pyproject.toml`.
- **Remediation Requirement**:
  Explicitly include updating `pyproject.toml` in task P5-01.

#### MAJ-07: Shannon Entropy Formula Negative Entropy and Warning Generation
- **Vector**: 1 & 5 (Question 1 & Numerical Stability)
- **Vulnerability**: Formula $H = -\sum p \log_2(p + \epsilon)$ produces negative entropy for deterministic positions and triggers `RuntimeWarning` when evaluated naively with $\epsilon=0$.
- **Remediation Requirement**:
  Mandate safe evaluation via:
  ```python
  pos = p > 0
  H = -np.sum(p[pos] * np.log2(p[pos]))
  ```

#### MAJ-08: Airgapped HTML Parser Blocking and Duplicate DOM IDs
- **Vector**: 5 (Airgapped Execution & Dashboard)
- **Vulnerability**: Embedding synchronous CDN `<script>` tags blocks browser parsing for 30–90 seconds on airgapped nodes. Re-using `id="fallback-svg"` across multiple panels breaks `document.getElementById()`.
- **Remediation Requirement**:
  1. Add `defer` or `async` to CDN scripts.
  2. Use unique IDs per panel: `id="fallback-svg-{panel_name}"`.
  3. Ensure inline static SVGs are displayed immediately by default without requiring JavaScript execution.

---

### [MINOR] Findings (Ergonomics, documentation, and defense-in-depth)

#### MIN-01: Near-Duplicate and Cross-Knob Collision Scope Gap
- **Vector**: 1 (Question 1: Library Composition)
- **Observation**: Spec §5.1.3 defines exact duplicate rate within a cell ($1.0 - |\text{unique}|/N$), but omits near-duplicate rate (Hamming distance $\le 1$ or PID $\ge 95\%$) and cross-cell collisions across temperature levels.
- **Remediation**: Add a cross-knob collision table reporting sequences shared across different knob levels.

#### MIN-02: Pareto Sorting Directionality Ambiguity
- **Vector**: 1 (Question 2: Scores & Energetics)
- **Observation**: §5.2.3 proposes Pareto sorting on $(\bar{S}_\tau, \text{PID}_\tau)$ without specifying whether pairwise identity is minimized or maximized.
- **Remediation**: Clarify that both objectives are to be minimized: $\min(\text{NLL})$ and $\min(\text{PID})$ (minimizing pairwise identity maximizes diversity).

#### MIN-03: Missing Import-Linter Enforcement for Boundary F-10
- **Vector**: 4 (Dependency Boundaries)
- **Observation**: Boundary F-10 states zero imports of `maraxiom`, `bathos`, or `cisternal`. While enforced conceptually, `pyproject.toml` lacks an import-linter contract to prevent accidental regression.
- **Remediation**: Add a `tool.importlinter.contracts` block in `pyproject.toml` forbidding `maraxiom`, `bathos`, and `cisternal` across all `src/xtrax/` modules.

---

## 4. Caveats

- **Airgapped Browser Runtimes**: Empirical testing of browser timeout behavior was performed via JavaScript DOM parsing analysis and protocol inspection, rather than interactive manual browser clicking on an Engaging node.
- **Physical GPU Profiling**: Production-shape wall-time profiling for $N=10,000$ was verified via NumPy BLAS micro-benchmarks on local CPU; distributed multi-GPU scaling is deferred to Phase P5 execution.
- **Upstream Campaign Runner**: Spec review assumed `origin/feat/261105-campaign-runner-track5` merges cleanly as planned in Section 1.1.

---

## 5. Conclusion

The specification `261106_campaign-eda-redesign.md` provides a comprehensive architectural foundation for multi-knob biomolecular library analysis. However, it currently contains **7 critical BLOCKER defects** (chiefly mathematical bias in cluster bootstrap when unit sample sizes vary, a fatal parameter conflict between the demo campaign and the small-sample guard, float overflow in Best-of-N curves, `NaN` poisoning in pairwise diversity, RFC 8259 JSON violations, and quarantine escape via `SystemExit`) along with **8 MAJOR implementation risks**.

Proceeding directly to Phase P5 without addressing these defects will result in broken tests, numerical inaccuracies, and victory audit rejection.

**Final Recommendation**: **REVISE SPECIFICATION (P2)**. Incorporate the remediation requirements outlined above into Revision 1.1 of `.praxia/docs/specs/261106_campaign-eda-redesign.md` before approving the P3 adversarial gate.

---

## 6. Verification Method

To independently reproduce and verify the findings in this report, execute the following commands in the project workspace:

1. **Verify Vectorized Cluster Bootstrap Bias on Unequal Sample Sizes (BLK-01)**:
   ```bash
   uv run python -c "
   import numpy as np
   N = [100, 10, 10]; means = [10.0, 2.0, 2.0]
   true_mean = sum(n * m for n, m in zip(N, means)) / sum(N)
   Y = np.array(means).reshape(-1, 1); U = 3; B = 10000
   W = np.random.multinomial(U, [1/U]*U, size=B)
   theta_spec = (1.0 / U) * (W @ Y)
   print('True:', true_mean, 'Spec Bootstrap Mean:', theta_spec.mean())
   assert abs(true_mean - theta_spec.mean()) > 3.0, 'Bias not reproduced!'
   "
   ```

2. **Verify Best-of-$N$ Float Overflow for $N > 1029$ (BLK-03)**:
   ```bash
   uv run python -c "
   from scipy.special import comb
   import numpy as np
   c1 = comb(1099, 499); c2 = comb(1100, 500)
   ratio = c1 / c2
   print('comb(1100, 500):', c2, 'ratio:', ratio)
   assert np.isnan(ratio), 'Overflow to NaN not reproduced!'
   "
   ```

3. **Verify Diversity `NaN` on $N=1$ and Fixed Scaffold (BLK-04)**:
   ```bash
   uv run python -c "
   import numpy as np
   triu = np.triu_indices(1, k=1)
   val = float(1.0 - np.mean(np.zeros((1, 1))[triu]))
   print('Diversity for N=1:', val)
   assert np.isnan(val), 'NaN on N=1 not reproduced!'
   "
   ```

4. **Verify RFC 8259 JSON Non-Compliance (BLK-05)**:
   ```bash
   uv run python -c "
   import json
   try:
       json.dumps({'diversity': float('nan')}, allow_nan=False)
       assert False, 'Should have raised ValueError!'
   except ValueError as e:
       print('Reproduced strict RFC 8259 rejection:', e)
   "
   ```

5. **Verify `SystemExit` Quarantine Escape (BLK-06)**:
   ```bash
   uv run python -c "
   def bad_plugin(): raise SystemExit('escape')
   try:
       try: bad_plugin()
       except Exception: print('Caught by Exception')
   except SystemExit:
       print('SystemExit bypassed except Exception as predicted!')
   "
   ```

6. **Verify Missing Target Fixture `tests/data/1ubq.pdb` (MAJ-05)**:
   ```bash
   test ! -f tests/data/1ubq.pdb && echo "Verified: tests/data/1ubq.pdb does not exist!"
   ```
