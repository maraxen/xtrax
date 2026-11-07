---
title: "xtrax Campaign Exploratory Data Analysis (EDA): Pluggable Biomolecular Library Analytics, Vectorized Uncertainty Engine, and Neutral Artifact Export"
description: "Formal technical specification for xtrax Campaign EDA, providing knob-faceted sink statistics across finished and running campaigns, typed domain plugins (aminx, demistify), vectorized cluster bootstrap uncertainty, storage format parity (loose vs packed), and neutral export artifacts, closing debts #3301, #3143, #3300, and demistify #830"
task_id: "261106_xtrax-campaign-eda"
status: "FINAL r1.1 — Approved Technical Specification (Zero Open Blockers/Majors)"
revision: "1.1 (2026-11-06; formal technical specification approved after P3 adversarial review and adjudication; incorporating all 18 concrete remediations: BLK-01..07, MAJ-01..08, MIN-01..03; zero open blockers and zero open majors)"
owner_decisions: "xtrax engine + consumer metric plugins; spec against track5 campaign runner; plan EDA preserved as secondary telemetry view; real demo on aminx ubiquitin campaign"
supersedes_scope_of: "xtrax debt #3143 (EDA visualization), debt #3301 (campaign EDA integration), debt #3300 (pack telemetry seam), and demistify debt #830"
---

# xtrax Campaign Exploratory Data Analysis (EDA) Specification

## Executive Summary & Revision History

### Executive Summary
This specification defines the architecture, data models, domain plugin contracts, vectorized statistical inference engine, and neutral visualization export pipelines for **xtrax Campaign Exploratory Data Analysis (EDA)** (`xtrax.eda.campaign`).

The primary mission of Campaign EDA is to evaluate **what multi-dimensional knob configurations produced** across finished or running campaigns, answering five foundational scientific questions:
1. **Library Composition**: Domain-defined sequence/state diversity, positional Shannon entropy, amino acid composition entropy, target/wild-type recovery with margin controls, duplicate rates, and epistatic shifts across knob levels.
2. **Scores & Energetics**: Per-sample and per-arm Negative Log-Likelihood (NLL) distributions, derived NLL from logits, perplexity, order statistics (Best-of-$N$ and Top-$k$ enrichment curves), and score-versus-diversity Pareto frontiers.
3. **Knob Effects with Honest Uncertainty**: Rigorous hierarchical statistical pooling under the global `md-replicas-are-samples` rule, vectorized cluster bootstrap confidence intervals, Minimum Detectable Effect ($\text{MDE}_{0.80}$) power checks, and small-sample guards ($U < 3$).
4. **Coverage Reconciliation**: Full Cartesian planned grid reconciliation against committed sinks via the unit index, explicitly rendering committed, partial, failed, and missing cells without silent omission.
5. **Secondary Run Telemetry**: Per-unit compute wall times, cache reuse/resume audit trails, tiling plan memory footprints, and transparent bannering of uninstrumented pack telemetry seams (closing debt #3300).

Campaign EDA operates **read-only** exclusively over committed sinks indexed by `manifest_row_hash`, enforces bit-identical parity between loose files and packed archives (`xtrax.pack`), executes consumer domain logic (`aminx`, `demistify`) within quarantined plugin boundaries, and produces neutral, content-hashed export artifacts (canonical RFC 8259 JSON, tidy Parquet, Vega-Lite JSON specs, static SVGs, and progressive airgapped HTML dashboards).

### Debts and Scope Closure Registry
This specification directly remediates and formally closes the following tracked technical debts:
- **xtrax Debt #3143 (EDA Visualization)**: Redesigns `xtrax.eda` from narrowly explaining compiler batching plans (`BatchPlan`) into analyzing generated biomolecular libraries and physical ensembles across experimental campaigns, while preserving plan analysis as the secondary telemetry view.
- **xtrax Debt #3301 (Campaign Runner EDA Integration)**: Integrates Campaign EDA directly with the `xtrax.campaign` runner (`origin/feat/261105-campaign-runner-track5`), reading `CampaignManifest` specifications and resolving execution units via the lock-free `xtrax.run.unit_index`.
- **xtrax Debt #3300 (Pack Telemetry Uninstrumented Seam)**: Explicitly banners pack execution events as `"not instrumented (debt #3300)"` in telemetry views while exposing static archive status (`is_packed: true/false`) and providing full loose-vs-packed read parity.
- **demistify Debt #830 (Consumer Campaign Analysis)**: Provides a typed domain plugin for `demistify` to extract conformational entropy ($S_i, S_{total}$), mutual information matrices, MIST spanning tree edges, and microstate trajectories, with graceful degradation when Zarr stores omit dense MI matrices (Schema B).

### Revision History

| Rev | Date | Author / Agent | Changes & Status |
|---|---|---|---|
| **r1.0** | 2026-11-06 | `teamwork_worker_spec_1` | Formal technical specification synthesizing Phase P0 Reconnaissance (`P0_RECON_SYNTHESIS.md`), Phase P1 Prior Art Research (`261106_campaign_eda_prior_art_research.md`), prototype validation evidence (`campaign_library_eda_prototype_v2.py`), and the draft respec (`261106_campaign-library-eda-respec.md`). Prepared for Phase P3 Adversarial Review. |
| **r1.1** | 2026-11-06 | `teamwork_worker_editor_1` | Approved Technical Specification (Zero Open Blockers/Majors). Incorporated all 18 concrete adjudicated remediations from Phase P3 adversarial review: BLK-01 (cluster ratio bootstrap estimator for unequal cluster sizes), BLK-02 (ubiquitin demo replicate count $U=3$ [1001, 1002, 1003], 288 sequences; small-sample guard fixture), BLK-03 (Best-of-$N$ candidate grid bound $n \le N$ and iterative ratio recurrence avoiding float64 overflow), BLK-04 (pairwise diversity edge guards for $N \le 1$ and $|M|=0$), BLK-05 (recursive RFC 8259 JSON sanitization coercing `NaN`/`Inf` to `None`/`null` with Vega-Lite filters), BLK-06 (plugin quarantine exception tuple `(Exception, SystemExit)`), BLK-07 (`CampaignReader` dual-marker discovery for `done.json` and `unit.json`), MAJ-01 (zero-copy array immutability via `flags.writeable = False`), MAJ-02 (strict scalar type and finiteness validation for `DomainMetricResult`), MAJ-03 (classification of abandoned `.tmp` staging stores and expired locks as `FAILED` with `killed_incomplete`), MAJ-04 (Cartesian grid visual rollup rule: 100% committed requirement), MAJ-05 (vendoring 76-residue ubiquitin coordinates into `tests/fixtures/1ubq.pdb`), MAJ-06 (explicit declaration of `xtrax[eda-campaign]` and `xtrax[eda-viz]` extras in `pyproject.toml` in P5-01), MAJ-07 (positive-support Shannon entropy $H = -\sum_{p>0} p \log_2 p \ge 0.0$), MAJ-08 (deferred script loading and unique panel DOM IDs `fallback-svg-{panel_name}`), MIN-01 (cross-knob collision table), MIN-02 (Pareto optimization direction $\min(\text{NLL})$ and $\min(\text{PID})$), and MIN-03 (`tool.importlinter` boundary contracts in `pyproject.toml`). Zero open blockers and zero open majors. |

---

## 1. Architectural Governance & Decision Records

### 1.1 Decision 1 Superseding Record: Campaign-Level Tables Prioritized
In Revision 3.0 of the compiler EDA redesign (`.praxia/docs/specs/261028_xtrax-eda-redesign.md`, §2.1), **Decision 1** stated:
> *"Prioritize single-plan inspection panels first (`strategy`, `cardinality`, `dedup`, `bucket`, `memory`, `reasoning`). Defer cross-plan/campaign-level aggregation tables until dependent systems articulate concrete data schema requirements."*

#### Superseding Rationale
1. **Concrete Downstream Adoption**: The condition for deferral has been completely fulfilled. Two major consumer frameworks—`aminx` (generative protein sequence design) and `demistify` (conformational entropy and allostery in molecular dynamics)—have completed their initial adoptions (e.g. `aminx` PR #270) and now require multi-knob campaign exploratory data analysis to evaluate generative outputs.
2. **Separation of Concerns (Telemetry vs. Substance)**: Experience with early adoption demonstrated that displaying only `PlanStatsDict` for campaign runs is an architectural category error: compiler execution strategy describes *how* computation was dispatched, whereas scientific research requires evaluating *what* was produced (sequences, energetics, structural ensembles).
3. **Active Stacking Topology**: The campaign execution engine has landed in `origin/feat/261105-campaign-runner-track5` (`CampaignSpec`, `CampaignManifest`, `CampaignLock`, `CampaignDoneMarker`), and the unit index has merged into `origin/main` (PR #261, `src/xtrax/run/unit_index.py`). The persistent storage substrate is mature and ready for campaign analytics.

#### Formal Superseding Declaration
**Decision 1 is hereby SUPERSEDED**. Campaign-level exploratory tables, multi-dimensional knob faceting, and domain library statistics are elevated to **active, first-class, and prioritized** components of `xtrax.eda.campaign`. Single-plan inspection is formally repositioned as the **Secondary Run Telemetry View** (answering Question 5) within the unified campaign EDA dashboard.

```
+---------------------------------------------------------------------------------------+
|                              xtrax.eda Unified Scope                                  |
+---------------------------------------------------------------------------------------+
|  Primary Scientific Scope (xtrax.eda.campaign)                                        |
|  - Q1: Library Composition (Diversity, Entropy, Recovery, Duplicates, Masking)       |
|  - Q2: Scores & Energetics (NLL, Derived NLL, Perplexity, Best-of-N, Pareto)          |
|  - Q3: Honest Uncertainty (MD-Replicas Hierarchy, Cluster Bootstrap, MDE, Guards)     |
|  - Q4: Coverage Reconciliation (Committed, Partial, Failed, Missing Sweep Grid)      |
+---------------------------------------------------------------------------------------+
|  Secondary Telemetry Scope (xtrax.eda / Plan Telemetry)                               |
|  - Q5: Run Telemetry (Wall Time, Cache Reuse Logs, Tiling Plans, Debt #3300 Seam)    |
+---------------------------------------------------------------------------------------+
```

### 1.2 Decision 3 Restatement: Minimal Core Dependencies & Decoupled Extras
In accordance with Decision 3 of `261028_xtrax-eda-redesign.md`:
1. **Core Footprint**: Core `xtrax` is a foundational numerical compiler library. `xtrax.eda.campaign` core modules (`schema`, `reader`, `stats`, `types`, `domains.protocol`) MUST depend strictly and exclusively on the Python standard library and `numpy`.
2. **Columnar & Visualization Extras**: Heavy analytical engines, columnar formats, and graphic compilers reside strictly behind optional extras declared in `pyproject.toml` (delivered in work package P5-01):
   - `xtrax[eda-campaign]`: Includes `duckdb >= 1.0.0` and `pyarrow >= 14.0.0`.
   - `xtrax[eda-viz]`: Includes `altair >= 5.0.0` and `vl-convert-python >= 1.3.0`.
   - `xtrax[eda-all]`: Composite extra installing both sets.
   ```toml
   # pyproject.toml declaration (P5-01)
   [project.optional-dependencies]
   eda-campaign = [
       "duckdb>=1.0.0",
       "pyarrow>=14.0.0",
   ]
   eda-viz = [
       "altair>=5.0.0",
       "vl-convert-python>=1.3.0",
   ]
   eda-all = [
       "xtrax[eda-campaign,eda-viz]",
   ]

   [project.entry-points."xtrax.eda.domains"]
   aminx = "aminx.eda.plugin:AminxDomainPlugin"
   demistify = "demistify.eda.plugin:DemistifyDomainPlugin"
   ```
3. **Zero Framework Bridges in Core**: There shall be **zero internal imports** of external orchestration, tracking, or telemetry frameworks (`maraxiom`, `bathos`, `cisternal`) inside `src/xtrax/eda`. Integration with these frameworks occurs externally via neutral, content-addressed file artifacts.

### 1.3 Boundary F-10 Restatement: Strict Namespace & Package Isolation
In accordance with Finding F-10 of `261028_xtrax-eda-redesign.md`:
1. **Zero Cross-Imports**: There shall be **zero cross-imports** between `xtrax.eda` and `maraxiom.eda`.
   - `xtrax.eda` is a foundational compiler and campaign analysis library.
   - `maraxiom.eda` is an internal EDA manifest verification runner for slide decks and review pages.
   - Any attempt to import `maraxiom` inside `xtrax` or vice versa is a fatal architectural violation.
2. **Import-Linter Contract Enforcement**:
   Boundary F-10 is automatically verified by `import-linter` via `pyproject.toml` contract configuration (delivered in P5-01):
   ```toml
   [tool.importlinter]
   root_package = "xtrax"

   [[tool.importlinter.contracts]]
   name = "Campaign EDA isolation from external frameworks"
   type = "forbidden"
   source_modules = ["xtrax.eda"]
   forbidden_modules = [
       "maraxiom",
       "bathos",
       "cisternal",
   ]
   ```
3. **Universal Seam Protocol**: `xtrax.eda.campaign` exports self-contained, neutral, standardized artifacts:
   - Canonical RFC 8259 JSON summaries.
   - Partitioned Apache Parquet tidy tables.
   - Declarative Vega-Lite v5 JSON specifications.
   - Static vector SVGs rendered via headless compilers.
   - Cryptographic SHA-256 manifests.
   Downstream systems (`maraxiom`, `bathos`, web browsers) consume these artifacts exclusively via filesystem paths or CLI adapters.

---

## 2. Master Analytical Mapping: The 5 Scientific Questions

Campaign EDA must provide definitive, rigorous answers to five scientific questions across any knob factor or multi-factor combination ($\text{arm} \times \text{checkpoint} \times \text{policy} \times \text{parameters} \times \text{target}$):

| Master Question | Core Analytical Metric | Consumer Domain | Data Source / Artifact | Exact Schema Fields / Mathematical Operations | File & Line Evidence |
|---|---|---|---|---|---|
| **Q1: Library Composition** | Diversity (Mean Pairwise Identity) | `aminx` | Committed Zarr sink: `sequences` | Upper triangle of $(S[:, \text{None}] == S[\text{None}, :]) \land M$ averaged over designable residue count $|M|$; boundary guards for $N \le 1$ and $|M|=0$ returning 0.0 with flags | `prototype_v2.py:160-166` |
| **Q1: Library Composition** | Positional Shannon Entropy | `aminx` | Committed Zarr sink: `logits` | Softmax over token axis, safe positive-support bit entropy $H(l) = -\sum_{c: p_l(c)>0} p_l(c) \log_2 p_l(c)$ per designable residue $l$ | `prototype_v2.py:168-174` |
| **Q1: Library Composition** | Composition Shannon Entropy | `aminx` | Committed Zarr `sequences` or Parquet `sequence` | Global amino acid composition bit entropy $H_{\text{comp}} = -\sum_{c: \bar{p}(c)>0} \bar{p}(c) \log_2 \bar{p}(c)$ | `prototype_v2.py:343-346`, `campaign_sink.py:391` |
| **Q1: Library Composition** | Target Recovery vs. Native | `aminx` | Committed Zarr `sequences` + native residue table | Fraction of designable residues matching wild-type sequence: $\frac{1}{\|M\|} \sum_{l \in M} \mathbb{I}(s_l == s_l^{\text{WT}})$ | `prototype_v2.py:146-150` |
| **Q1: Library Composition** | Recovery Margin vs. Shuffled Native | `aminx` | Zarr `sequences` + shuffled wild-type target | $\text{Rec}(\mathbf{s}) - \text{Rec}_{\text{shuf}}(\mathbf{s})$; flag degenerate if margin $< 0.10$ | `prototype_v2.py:279-284, 348` |
| **Q1: Library Composition** | Duplicate & Cross-Knob Collision Rate | `aminx` | Committed Zarr `sequences` | Within-cell duplicate fraction $1.0 - \frac{\|\text{unique}(\mathbf{s})\|}{N}$; cross-knob collision matrix across temperature levels | `campaign_sink.py:391` |
| **Q1: Library Composition** | Designable Real-Residue Masking | `aminx` | Target length $L_{\text{true}}$ + Zarr `fixed_mask` | $M = \mathbf{m}_{\text{real}} \land \neg \mathbf{m}_{\text{fixed}}$; unmasked padding positions ($H \approx 4.04$ bits) strictly excluded | `prototype_v2.py:140-144` |
| **Q1: Library Composition** | Conformational State Trajectory | `demistify` | Committed sink: `states_traj.npy` or Zarr `states/states_traj` | Discrete microstate trajectory shape $(T, L)$ with microstate indices $0..K-1$ | `demistify:artifact-schema.md:12-25` |
| **Q2: Scores & Energetics** | Negative Log-Likelihood (NLL) | `aminx` | Parquet `score_nll` or Zarr `scores` array | Model-reported per-residue NLL: $-\frac{1}{\|M\|} \sum_{l \in M} \log p(s_l)$ | `campaign_sink.py:38, 393` |
| **Q2: Scores & Energetics** | Derived NLL & Perplexity | `aminx` | Committed Zarr `logits` + `sequences` | $\text{NLL}_{\text{der}} = \frac{1}{\|M\|} \sum_{l} (\text{LSE}(\mathbf{Z}_l) - Z_{l, s_l})$; $\text{Perplexity} = \exp(\min(\text{NLL}, 50))$ | `prototype_v2.py:152-158` |
| **Q2: Scores & Energetics** | Best-of-$N$ Enrichment Curve | `aminx` | Sorted candidate scores per condition | Order statistics: expected top score $\mathcal{E}(n) = \mathbb{E}[\min_{i=1..n} S_i]$ on bounded grid $n \in \{2^k \le N\} \cup \{N\}$ via iterative ratio recurrence $\omega_j(n, N)$ | Spec §5.2.2 |
| **Q2: Scores & Energetics** | Score vs. Diversity Pareto Frontier | `aminx` | Subsets filtered by score threshold $\tau$ | Non-dominated sorting minimizing both $\text{NLL}$ and $\text{PID}$ ($\min(\text{NLL}), \min(\text{PID})$) across score cutoffs | `prototype_v2.py:160`, Spec §5.2.3 |
| **Q2: Scores & Energetics** | Conformational Entropy & Allostery | `demistify` | `<sys>_summary.json`, `_mist_edges.csv`, `_mi_matrix.npy` | Residue entropy $S_i$, total MIST entropy $S_{\text{total}}$, pairwise MI matrix $I(X_i; X_j)$, MIST spanning tree edges | `demistify:artifact-schema.md:1-81` |
| **Q3: Honest Uncertainty** | Hierarchical Replicate Pooling | All | `ManifestRow.seed` + `ManifestRow.parameters` | Seeds pooled as samples of unconverged estimator per global `md-replicas-are-samples` rule | `md-analysis:SKILL.md:1-29` |
| **Q3: Honest Uncertainty** | Vectorized Cluster Ratio Bootstrap | All | Unit totals $T \in \mathbb{R}^{U \times D}$, counts $\mathbf{n} \in \mathbb{N}^U$, weights $W \in \mathbb{N}^{B \times U}$ | Fast BLAS projection: $\hat{\Theta}^* = \frac{W \cdot T}{(W \cdot \mathbf{n})[:, \text{None}]}$; empirical quantiles $[q_{\alpha/2}, q_{1-\alpha/2}]$ | `research_note:171-182`, Spec §5.3.2 |
| **Q3: Honest Uncertainty** | Small-Sample Guard ($U < 3$) | All | Replicate count $U$ in condition cell | When $U < 3$: mark `descriptive_only = True`, suppress CI intervals, emit warning banner | `nulls-and-uncertainty.md:45-60`, Spec §5.3.3 |
| **Q3: Honest Uncertainty** | Minimum Detectable Effect ($\text{MDE}$) | All | Standard Error of contrast $\text{SE}(\hat{\Delta})$ | $\text{MDE}_{0.80} \approx 2.802 \cdot \text{SE}(\hat{\Delta})$; if $\|\hat{\Delta}\| < \text{MDE}$, flag contrast underpowered | `research_note:183-196` |
| **Q4: Coverage** | Planned Cartesian Sweep Grid | All | `CampaignManifest.rows` | Full Cartesian product of planned parameter cells; visual cell is `COMMITTED` iff 100% of rows committed | `src/xtrax/campaign/manifest.py:232`, Spec §5.4 |
| **Q4: Coverage** | Committed Unit Verification | All | Unit Index Snapshot | `lookup(index, store, row.manifest_row_hash, row.input_hash)` returns `IndexReuse` | `src/xtrax/run/unit_index.py:755` |
| **Q4: Coverage** | Done Marker Verification | All | Done marker JSON (`done.json` or `unit.json`) | `validate_unit_done(out_dir, row)` verifies `content_digest_sha256` | `src/xtrax/campaign/marker.py:363`, Spec §3.3 |
| **Q4: Coverage** | Partial / In-Progress Cell | All | Filesystem leases | Active `CampaignLock` exists or staging directory `<store>.tmp` unsealed | `src/xtrax/campaign/lease.py:57` |
| **Q4: Coverage** | Failed / Corrupt Cell | All | Unit Index / Quarantine / FS scan | `lookup()` returns `Corrupt`, quarantined unit, or abandoned `*.tmp` / expired lock (`killed_incomplete`) | `src/xtrax/campaign/marker.py:274`, Spec §5.4 |
| **Q4: Coverage** | Missing Sweep Cell | All | Set difference | Planned row present in manifest with zero disk trace (never executed) | Set difference reconciliation |
| **Q5: Secondary Run View** | Unit Compute Wall Time | All | `CampaignDoneMarker` | Elapsed time `duration_s: float` and `completed_at_utc: str` | `src/xtrax/campaign/marker.py:87-101` |
| **Q5: Secondary Run View** | Cache Reuse Audit Trail | All | `<campaign>/reuse_log.jsonl` | Append-only log of resumed units: `manifest_row_hash`, `job_id`, `verify_mode` | `src/xtrax/campaign/executors/local.py:46` |
| **Q5: Secondary Run View** | Pack Telemetry Seam | All | `xtrax.pack` | **Uninstrumented Seam (Debt #3300)**: Must display `"not instrumented (debt #3300)"` | `src/xtrax/pack/_node.py` |
| **Q5: Secondary Run View** | Storage Format Discovery | All | `done.json` / `unit.json` / `_node.py` | Static check: `is_packed(unit_dir) -> bool` supporting dual markers | `src/xtrax/pack/_node.py:506`, Spec §3.4 |
| **Q5: Secondary Run View** | Compiler Tiling & Memory Stats | All | `BatchPlanLike` / `PlanStatsDict` | Strategy counts (`Vmap`, `Scan`), `memory_estimate_bytes`, `memory_warnings` | `src/xtrax/eda/types.py:65-81` |

---

## 3. Data Model, Schemas & Storage Parity

### 3.1 Campaign Execution Data Model
Campaign EDA reads the campaign plan directly from `CampaignManifest` (`src/xtrax/campaign/manifest.py`), which captures the planned Cartesian sweep grid.

```
+----------------------------------------------------------------------------------------+
|                                    CampaignManifest                                    |
+----------------------------------------------------------------------------------------+
| schema_version: "campaign_manifest_v1"                                                 |
| metadata: ManifestMetadata (campaign_id, planner_version, spec_sha256, total_rows)     |
| rows: list[ManifestRow]                                                                |
+----------------------------------------------------------------------------------------+
                                         │ 1..*
                                         ▼
+----------------------------------------------------------------------------------------+
|                                      ManifestRow                                       |
+----------------------------------------------------------------------------------------+
| manifest_row_hash: str (64-hex SHA-256 canonical identity)                             |
| input_hash: str        (64-hex SHA-256 aminx parameter parity hash)                   |
| campaign_id: str                                                                       |
| job_id: str                                                                            |
| chunk_id: int                                                                          |
| sample_start: int                                                                      |
| sample_count: int                                                                      |
| target_input: str                                                                      |
| arm_id: str                                                                            |
| arm_backend: str                                                                       |
| checkpoint_id: str                                                                     |
| fixed_policy: str                                                                      |
| state_weight_profile: str                                                              |
| seed: int                                                                              |
| parameters: dict[str, Any]                                                             |
| output_locator: str                                                                    |
| resource_requirements: ResourceRequirements                                           |
+----------------------------------------------------------------------------------------+
```

#### Canonical Hashing Specification
Every execution unit is identified by two 64-character lowercase hexadecimal SHA-256 digests:
1. `manifest_row_hash`: The global content-addressed identity of the unit, computed over the canonical JSON serialization of the normalized row payload:
   $$\text{manifest\_row\_hash} = \text{SHA256}\left( \text{canonical\_json\_bytes}(\mathcal{P}_{\text{row}}) \right)$$
   where $\mathcal{P}_{\text{row}}$ contains `campaign_id`, `job_id`, `chunk_id`, `sample_start`, `sample_count`, `target_input`, `arm_id`, `arm_backend`, `checkpoint_id`, `fixed_policy`, `state_weight_profile`, `seed`, and `parameters`.
2. `input_hash`: Computed for backwards compatibility with `aminx` parameter hashing, omitting campaign-owned bookkeeping keys (`job_id`, `chunk_id`, `output_path`, etc.).
3. **Canonical Normalization Rules (RFC 8259)**:
   - Floating-point normalization: Python `float` and NumPy floating values equal to `-0.0` are normalized strictly to `0.0`.
   - Non-finite value sanitization (RFC 8259 compliance): RFC 8259 strictly forbids `NaN`, `Infinity`, and `-Infinity` literals. In `xtrax.eda.campaign.export`, a recursive sanitization pass converts all `np.nan` and `math.isnan(v)` values to Python `None` (which serializes to standard JSON `null`), and converts `math.isinf(v)` to `None` with a diagnostic flag. Serializers execute strictly with `json.dumps(obj, allow_nan=False)` to guarantee syntax validity across all downstream JavaScript and standard JSON parsers.
   - String normalization: Unicode strings are normalized via `unicodedata.normalize("NFC", s)`.
   - Key sorting: Dictionary keys are stringified and sorted lexically.
   - Whitespace: Standard JSON separators `(',', ':')` without whitespace.

### 3.2 Knob Taxonomy
A central requirement of Campaign EDA is distinguishing independent experimental variables from stochastic replications:

1. **Row Knobs (Grid Axes & Facets)**:
   Experimental conditions planned at the manifest level:
   - Primary arm identifiers: `arm_id`, `arm_backend`, `checkpoint_id`.
   - Structural policies: `fixed_policy`, `state_weight_profile`.
   - Target configuration: `target_input` (PDB path or identifier).
   - User sweep parameters: Any parameter declared in `grid.axes` of `CampaignSpec` (e.g. `noise_level`, `step_count`, `model_variant`).
2. **Sink Knobs (In-Store Dimensions)**:
   Experimental factors evaluated within a single unit execution and stored along an array axis inside the committed sink (e.g. `temperature` in `aminx` stores).
   - Declared via `SinkSchema.bind` (e.g. `{"temperature": "temperatures"}`).
   - The EDA engine joins sink knobs with row knobs into a unified condition dictionary.
3. **Replicate Knobs (Stochastic Samples)**:
   - **`seed` is strictly a replicate sample**, NEVER a condition knob or factorial level.
   - In accordance with the global `md-replicas-are-samples` rule, different seeds under the same row/sink knob condition represent draws from an unconverged estimator and are pooled together to evaluate estimation variance.

### 3.3 Unit Index Integration & Done Markers
Campaign EDA does not scan directory trees searching for arbitrary files; it queries committed records via the lock-free unit index (`src/xtrax/run/unit_index.py`):

```python
from xtrax.run.unit_index import IndexSnapshot, lookup, IndexReuse, Corrupt, Missing, SchemaMismatch, IndexStale

# Load snapshot once across the entire campaign index root
index_snapshot: IndexSnapshot = load_index(campaign_index_root)

# Lookup unit by canonical manifest_row_hash and expected input_hash
result = lookup(
    index=index_snapshot,
    store=campaign_store_path,
    unit_key=row.manifest_row_hash,
    input_digest=row.input_hash,
    verify=True,
)
```

#### Lookup Result Classification & Bannering Rules
- `IndexReuse`: The unit is fully committed, verified, and eligible for statistical pooling.
- `Missing`: The unit was planned in the manifest but has not been executed. It is classified as `missing` in Question 4 coverage and omitted from statistical pooling.
- `Corrupt`: The unit's done marker or array content digest failed verification. The unit is classified as `failed` in Question 4 coverage, quarantined, and omitted from statistical pooling.
- `IndexStale` / `SchemaMismatch`: Flagged as execution defects and reported in coverage diagnostics.

#### Done Marker Verification Contract (`CampaignDoneMarker`)
For every committed unit, the completion marker located at `<unit_dir>/done.json` (or `<unit_dir>/unit.json` for packed units) must be verified:
- Schema version must equal `"campaign_done_marker_v3"` (or `"unit_pack_v1"` / `"campaign_done_marker_v3"` depending on whether written by the runner or pack tool).
- `manifest_row_hash` and `input_hash` must match the manifest row.
- `content_digest_sha256` must match the recomputed array digest of the store (Tier-2 deep verification).
- Telemetry fields `duration_s`, `completed_at_utc`, and `producer` are extracted for Question 5.

### 3.4 Storage Format Parity (LOOSE vs. PACKED) & Reader Harmonization
Units may exist on disk in two distinct storage layouts:
1. **Loose Directory**: Standard directory holding loose array files (e.g. Zarr group hierarchy, `.npy` files, `done.json`).
2. **Packed Archive (`xtrax.pack`)**: Compressed seekable archive `members.tar.zst` indexed by `members.idx.json` alongside a root `unit.json` record.

#### Storage Parity Invariant & Dual-Marker Reader
While `xtrax.pack._node.py` defaults to `record_name="unit.json"`, the campaign runner writes `<unit_dir>/done.json` (`campaign_done_marker_v3`). To eliminate record name disconnects and prevent `FileNotFoundError` on swept packed units, `CampaignReader` (`src/xtrax/eda/campaign/reader.py`) explicitly detects which marker exists and delegates accordingly:
```python
from pathlib import Path
from xtrax.pack import read_unit_member, is_packed

def read_campaign_unit_member(unit_dir: Path, relpath: str) -> bytes:
    """Transparently read member bytes from loose filesystem or compressed zstd archive frame.
    
    Inspects unit directory for either 'done.json' (campaign runner marker) or 'unit.json'
    (pack tool marker) before reading, preventing member lookup failures on swept units.
    """
    record_name = "done.json" if (unit_dir / "done.json").exists() else "unit.json"
    return read_unit_member(unit_dir, relpath, record_name=record_name)
```
- **Bit-Identical Parity**: The EDA reader MUST yield identical NumPy arrays, string tokens, and floating-point metrics regardless of whether a unit is loose or packed.
- **Testable Contract**: The test suite enforces that reading a fixture from a loose directory versus reading the identical fixture packed via `pack_unit()` produces bitwise-identical results across all 5 scientific questions.

---

## 4. Typed Domain Plugin Architecture (`xtrax.eda.domains`)

Because `xtrax` is a foundational compiler library, it does not hardcode biological or chemical domain semantics. Instead, domain knowledge is declared via a typed, versioned plugin interface.

### 4.1 Entry Point Group & Plugin Protocol
Plugins are registered via Python standard entry points under the group:
```
[project.entry-points."xtrax.eda.domains"]
aminx = "aminx.eda.plugin:AminxDomainPlugin"
demistify = "demistify.eda.plugin:DemistifyDomainPlugin"
```

#### Protocol Definition
```python
from typing import Protocol, runtime_checkable, Mapping, Sequence, Any
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True, slots=True)
class ResidueTable:
    """Canonical structural residue mapping."""
    native_tokens: str                # Normalized single-letter amino acid string
    chain_ids: tuple[str, ...]        # Chain identifier per residue
    res_numbers: tuple[int, ...]      # PDB author residue number per residue
    insertion_codes: tuple[str, ...]  # PDB insertion code per residue
    real_mask: np.ndarray             # Boolean array (True for real target residues)
    parser_name: str                  # Name of producing parser (e.g. "proxide")
    parser_version: str               # Version string of producing parser

@dataclass(frozen=True, slots=True)
class LibraryCell:
    """Standardized extracted data cell for a single experimental unit."""
    knobs: Mapping[str, Any]          # Merged row and sink knob dictionary
    seed: int                         # Stochastic replicate seed
    target: str                       # Target identifier
    tokens: np.ndarray                # Integer token array (N, L)
    sequences: tuple[str, ...]        # Decoded sequence strings (N,)
    logits: np.ndarray | None         # Unnormalized model logits (N, L, alphabet_size) or None
    scores: np.ndarray | None         # Model-reported scores (N,) or None
    is_score_derived: bool            # True if scores were derived host-side from logits
    residues: ResidueTable            # Target residue mapping table
    mask: np.ndarray                  # Boolean designable mask (L,)

@dataclass(frozen=True, slots=True)
class DomainMetricResult:
    """Named scalar metrics computed for a cell with strict type and finiteness contracts."""
    metrics: Mapping[str, float]      # Scalar metrics (e.g. {"diversity": 0.45, "rec": 0.62})
    flags: Mapping[str, bool]         # Boolean diagnostic flags (e.g. {"degenerate": False})
    tracks: Mapping[str, np.ndarray]  # Per-residue 1D arrays (e.g. {"entropy": H_l})

    def validate_schema(self) -> None:
        """Validate that all metric values are strictly numeric and finite.
        
        Enforces that every value in `metrics` satisfies `isinstance(v, (int, float))` and
        `math.isfinite(v)` (non-NaN and non-Inf). Violations trigger immediate unit quarantine,
        guaranteeing that downstream PyArrow Parquet table conversion never fails with ArrowInvalid.
        """
        for k, v in self.metrics.items():
            if not isinstance(v, (int, float)) or not np.isfinite(v):
                raise TypeError(
                    f"Domain metric {k!r} has invalid value {v!r} (type {type(v).__name__}); "
                    "metrics must be strictly finite floats or ints."
                )

@runtime_checkable
class DomainPlugin(Protocol):
    """Protocol for consumer domain analytical plugins."""
    
    @property
    def domain_name(self) -> str:
        """Unique identifier of the domain (e.g. 'aminx', 'demistify')."""
        ...
        
    @property
    def version(self) -> str:
        """Semantic version of the plugin implementation."""
        ...

    def extract_cell(
        self,
        store_path: Path,
        manifest_row: ManifestRow,
        sink_schema: SinkSchema,
    ) -> Sequence[LibraryCell]:
        """Extract one or more LibraryCells from a committed unit store."""
        ...

    def compute_metrics(self, cell: LibraryCell) -> DomainMetricResult:
        """Compute domain-specific scalar and per-residue metrics."""
        ...
```

### 4.2 Quarantine Execution & Immutability Isolation
To protect the EDA engine from third-party plugin bugs, unhandled crashes, or memory corruption:
1. **Zero-Copy Immutability Isolation**:
   To avoid duplicate heap allocations on large campaigns (where deep copying dense arrays like logits $\mathbb{R}^{N \times L \times 21}$ would consume hundreds of megabytes), arrays passed to or created within `LibraryCell` are enforced read-only via NumPy array flags:
   ```python
   def enforce_cell_immutability(cell: LibraryCell) -> None:
       """Set writeable=False on all underlying NumPy arrays for zero-copy immutability."""
       cell.tokens.flags.writeable = False
       cell.mask.flags.writeable = False
       if cell.logits is not None:
           cell.logits.flags.writeable = False
       if cell.scores is not None:
           cell.scores.flags.writeable = False
       if hasattr(cell.residues, "real_mask") and isinstance(cell.residues.real_mask, np.ndarray):
           cell.residues.real_mask.flags.writeable = False
   ```
   Any attempt by a plugin to mutate arrays in-place immediately raises `ValueError: assignment destination is read-only` at 0% memory copy overhead.
2. **Defensive Quarantine Protocol**:
   - Every call to `extract_cell()` and `compute_metrics()` is wrapped in a defensive exception handler catching both `Exception` and `SystemExit`.
   - Wrapping `(Exception, SystemExit)` prevents a buggy or rogue plugin from terminating the runner process via `sys.exit()`, ensuring robust isolation.
   - Outputs are validated against `DomainMetricResult.validate_schema()` to guarantee all metric values are finite numeric types before entering the Parquet export pipeline.
   ```python
   try:
       cells = plugin.extract_cell(store_path, row, schema)
       for cell in cells:
           enforce_cell_immutability(cell)
           metric_result = plugin.compute_metrics(cell)
           metric_result.validate_schema()
   except (Exception, SystemExit) as exc:
       logger.warning("Plugin %s failed on unit %s: %s", plugin.domain_name, row.manifest_row_hash, exc)
       self.quarantine_unit(row.manifest_row_hash, plugin.domain_name, str(exc))
       return []
   ```
   - Quarantined units are excluded from statistical pooling and rendered in Question 4 coverage dashboards with a prominent `QUARANTINED` diagnostic banner.
   - The public introspection function `xtrax.eda.campaign.get_quarantined_units() -> list[QuarantineRecord]` exposes all quarantined units for test assertions.

### 4.3 The `aminx` Domain Plugin
The `aminx` domain plugin extracts generative sequence libraries produced by models such as ProteinMPNN, SolubleMPNN, and LigandMPNN:

1. **Sink Extraction**:
   - Reads `sequences` array of shape `(N, [grid], [temperature], L)`.
   - Reads `logits` array of shape `(N, [grid], [temperature], L, 21)` if present.
   - Reads `fixed_mask` array of shape `(L,)` indicating non-designable pinned residues.
2. **Alphabet Normalization**:
   - Enforces explicit conversion between internal token orders:
     - MPNN alphabet: `ACDEFGHIKLMNPQRSTVWYX` (length 21).
     - AlphaFold/Proxide alphabet: `ARNDCQEGHILKMFPSTWYVX` (length 21).
   - Maps tokens to canonical single-letter strings prior to any string comparison.
3. **Residue Table Handling**:
   - Attempts to read embedded `residue_table` attribute from the Zarr group.
   - Fallback: Retrieves canonical residue table by parsing the target PDB file via the recorded parser version. Re-parsing is refused if the parser version differs from the producing build.
4. **Designable Masking**:
   - Evaluates real target residues $L_{\text{true}}$ from the residue table.
   - Constructs designable mask:
     $$\mathbf{m}_{\text{designable}} = \mathbf{m}_{\text{real}} \land \neg \mathbf{m}_{\text{fixed}}$$
   - Guarantees that unmasked padding tokens ($H \approx 4.04$ bits) never contaminate diversity, composition, or recovery calculations.

### 4.4 The `demistify` Domain Plugin
The `demistify` domain plugin extracts conformational entropy and allosteric mutual information networks from molecular dynamics simulations:

1. **Observables Extraction**:
   - Per-residue conformational entropy $S_i$ (nats).
   - Total MIST conformational entropy $S_{\text{total}}$ (nats).
   - Maximum Information Spanning Tree (MIST) edges from `_mist_edges.csv`.
   - Pairwise Mutual Information matrix $I(X_i; X_j)$ from `_mi_matrix.npy`.
   - Microstate trajectory `states_traj` from Zarr group `states/states_traj`.
2. **Schema B Degradation Protocol**:
   - In Schema B (`result.zarr`), the dense pairwise MI matrix is omitted to save disk space (`matrix_fullness: false`), retaining only MIST tree edges.
   - When encountering Schema B:
     - Computes allosteric hub metrics using tree node degree and edge weights instead of dense matrix sums.
     - Flags `matrix_fullness = False` and sets metric names with `der.hub_degree@1`.
     - Displays an informational banner: `"Schema B: dense MI matrix omitted; allosteric rankings derived from MIST spanning tree."`
     - Continues execution without raising an exception.

---

## 5. Vectorized Analytical Engine: Answering the 5 Questions

Campaign EDA enforces a strict **vectorization mandate**: all statistical calculations across units, seeds, bootstrap resamples, and knob cells MUST be vectorized in NumPy, Arrow, or DuckDB. Python `for` loops over individual samples, draws, or units are strictly prohibited.

### 5.1 Question 1: Library Composition Analytics

#### 5.1.1 Pairwise Identity & Hamming Diversity
For a cell containing $N$ sequence tokens $S \in \{0, \dots, 20\}^{N \times L}$ and designable boolean mask $M \in \mathbb{B}^{L}$ with $|M| = \sum M$:
1. Broadcast equality across sequence pairs:
   $$\text{Match}_{i, j, l} = (S_{i, l} == S_{j, l}) \quad \in \mathbb{B}^{N \times N \times L}$$
2. Mask and sum over designable positions:
   $$\text{PID}_{i, j} = \frac{1}{|M|} \sum_{l=1}^L \left( \text{Match}_{i, j, l} \land M_l \right) \quad \in [0, 1]^{N \times N}$$
3. **Vectorized Extraction with Strict Boundary Guards**:
   To prevent `NaN` division-by-zero or empty upper-triangle slice warnings when $N \le 1$ or $|M| = 0$ (e.g. fully pinned scaffolds):
   ```python
   flags: dict[str, bool] = {}
   if N <= 1:
       mean_diversity = 0.0
       flags["single_sample"] = True
   elif np.sum(designable_mask) == 0:
       mean_diversity = 0.0
       flags["zero_designable_positions"] = True
   else:
       triu_indices = np.triu_indices(N, k=1)
       mean_diversity = float(1.0 - np.mean(pid_matrix[triu_indices]))
   ```

#### 5.1.2 Positional & Composition Shannon Entropy
1. **Positional Shannon Entropy**:
   Computed per position $l \in M$ from model logits $\mathbf{Z} \in \mathbb{R}^{N \times L \times 21}$ or empirical token counts.
   Evaluated strictly over positive-support probabilities ($p_l(c) > 0$) to guarantee non-negative entropy ($H \ge 0.0$) and avoid numerical bias from additive $\epsilon$ offsets:
   $$p_l(c) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(S_{i, l} == c)$$
   $$H(l) = -\sum_{c: p_l(c) > 0} p_l(c) \log_2 p_l(c) \quad \text{(bits)}$$
   ```python
   pos = p > 0
   H_l = float(-np.sum(p[pos] * np.log2(p[pos]))) if np.any(pos) else 0.0
   ```
2. **Composition Shannon Entropy**:
   Measures global amino acid usage across the entire generated library over positive-support probabilities:
   $$\bar{p}(c) = \frac{1}{N \cdot |M|} \sum_{i=1}^N \sum_{l \in M} \mathbb{I}(S_{i, l} == c)$$
   $$H_{\text{comp}} = -\sum_{c: \bar{p}(c) > 0} \bar{p}(c) \log_2 \bar{p}(c) \quad \text{(bits)}$$
   - *Diagnostic Rule*: If $H_{\text{comp}} < 3.5$ bits (maximum 4.32 bits for 20 amino acids), the engine flags `library_collapse = True`.

#### 5.1.3 Target Recovery, Recovery Margin & Cross-Knob Collisions
1. Native target recovery:
   $$\text{Rec}(\mathbf{s}) = \frac{1}{|M|} \sum_{l \in M} \mathbb{I}(s_l == s_l^{\text{WT}})$$
2. Shuffled native baseline:
   - Generate a composition-preserving permutation of the wild-type sequence $\mathbf{s}^{\text{shuf}} = \pi(\mathbf{s}^{\text{WT}})$.
   - Compute recovery against shuffled sequence:
     $$\text{Rec}_{\text{shuf}}(\mathbf{s}) = \frac{1}{|M|} \sum_{l \in M} \mathbb{I}(s_l == s_l^{\text{shuf}})$$
3. **Recovery Margin & Degeneracy Guard**:
   $$\text{Margin} = \text{Rec}(\mathbf{s}) - \text{Rec}_{\text{shuf}}(\mathbf{s})$$
   - *Refusal Threshold*: If $\text{Margin} < 0.10$, the generator is operating near random chance given the background amino acid composition. The arm is flagged with `degenerate_arm = True` and highlighted in red on composition panels.
4. **Duplicate Rates & Cross-Knob Collision Matrix**:
   - Within-cell exact duplicate rate: $1.0 - \frac{|\text{unique}(\mathbf{s})|}{N}$.
   - Near-duplicate rate: fraction of sequence pairs within cell having Hamming distance $\le 1$ ($\text{PID} \ge 1.0 - \frac{1}{|M|}$).
   - **Cross-Knob Collision Table**: Evaluates sequence collisions across distinct knob levels (e.g. across temperature levels $T_a, T_b$ or arm variants):
     $$\text{Collision}(K_a, K_b) = \frac{|\mathcal{S}(K_a) \cap \mathcal{S}(K_b)|}{\min(|\mathcal{S}(K_a)|, |\mathcal{S}(K_b)|)}$$
     Outputs a symmetric cross-knob collision matrix reporting exact sequence overlaps and shared motifs across experimental conditions.

---

### 5.2 Question 2: Scores & Energetics Analytics

#### 5.2.1 NLL, Derived Fallback, and Perplexity
1. Model-reported Negative Log-Likelihood (NLL) per residue:
   $$\text{NLL}(\mathbf{s}) = -\frac{1}{|M|} \sum_{l \in M} \log p(s_l \mid \text{context})$$
2. **Derived Fallback from Logits**:
   When models output unnormalized logits $\mathbf{Z} \in \mathbb{R}^{L \times 21}$ but omit precomputed scores:
   $$\text{NLL}_{\text{der}}(\mathbf{s}) = \frac{1}{|M|} \sum_{l \in M} \left( \text{LogSumExp}(\mathbf{Z}_{l, :}) - Z_{l, s_l} \right)$$
   - The metric column is tagged as `der.nll@1?` to indicate host-derived provenance.
3. **Perplexity**:
   $$\text{Perplexity}(\mathbf{s}) = \exp\left( \min\left( \text{NLL}(\mathbf{s}), \, 50.0 \right) \right)$$

#### 5.2.2 Best-of-$N$ & Top-$k$ Enrichment Curves
Evaluates how expected top candidate performance scales with candidate library size $n \in [1, N]$:
1. For an ordered sequence of scores $S_{(1)} \le S_{(2)} \le \dots \le S_{(N)}$ in a condition cell:
2. **Candidate Grid Bound**:
   The candidate evaluation grid of subsample sizes is strictly bounded by available samples:
   $$n \in \{2^k \mid 2^k \le N\} \cup \{N\}$$
   Candidate subsample sizes $n > N$ are prohibited, preventing division-by-zero or empty combinatorial sums.
3. **Numerically Stable Order Statistic Recurrence**:
   The expected minimum score from a random subsample of size $n$ is evaluated analytically via order statistics:
   $$\mathbb{E}\left[ \min_{i=1..n} S_i \right] = \sum_{j=1}^{N - n + 1} S_{(j)} \cdot \omega_j(n, N)$$
   where the combinatorial weight $\omega_j(n, N) = \binom{N - j}{n - 1} / \binom{N}{n}$ is evaluated via a numerically stable iterative recurrence:
   $$\omega_1(n, N) = \frac{n}{N}$$
   $$\omega_j(n, N) = \omega_{j-1}(n, N) \cdot \frac{N - n - j + 2}{N - j + 1} \quad \text{for } j = 2, \dots, N - n + 1$$
   This evaluates strictly as products of ratios in $[0, 1]$, completely eliminating `scipy.special.comb` float64 overflow to `inf / inf = nan` for large libraries ($N > 1029$).
4. Demonstrates the diminishing returns of candidate scaling across temperature settings.

#### 5.2.3 Score-versus-Diversity Pareto Frontiers
1. Evaluates the multi-objective trade-off between sequence quality (low NLL) and library variation (high diversity).
2. For each condition, sweep score threshold percentiles $\tau \in [10\%, 20\%, \dots, 100\%]$.
3. For each subset $S_\tau = \{\mathbf{s} \mid \text{NLL}(\mathbf{s}) \le q_\tau\}$:
   - Compute mean score $\bar{S}_\tau$.
   - Compute mean pairwise identity $\text{PID}_\tau$.
4. **Pareto Sorting Direction**:
   Kung's non-dominated 2D sorting minimizes both objectives simultaneously:
   $$\min \bar{S}_\tau \quad \text{(minimize Negative Log-Likelihood / energy)}$$
   $$\min \text{PID}_\tau \quad \text{(minimize Pairwise Identity = maximize library diversity)}$$
   A candidate condition $A$ dominates $B$ if $\bar{S}_A \le \bar{S}_B$ and $\text{PID}_A \le \text{PID}_B$ with at least one strict inequality. The non-dominated set traces the Pareto frontier curve.

---

### 5.3 Question 3: Honest Uncertainty & Statistical Inference

#### 5.3.1 The Global MD-Replicas Hierarchy Rule
In computational biology and molecular simulations, treating generated sequences or trajectory frames as independent degrees of freedom ($N$) causes spurious statistical significance ($p \to 0$ as $N \to \infty$).
- **Formal Hierarchy**:
  $$\text{frames / sequences} \subset \text{seeds / replicas} \subset \text{build} \subset \text{condition}$$
- **Replicas are Samples, Not Experiments**: Seeds are draws from an unconverged estimator. Pooling (concatenation) across seeds within a condition defines the point estimator.
- **The Independent Unit**: The independent experimental unit $U$ for calculating variance and confidence intervals is the **seed** or **build**.

#### 5.3.2 Vectorized Cluster Ratio Bootstrap
To compute non-parametric confidence intervals without Gaussian assumptions, without Python loops, and without bias from unequal unit sample sizes (e.g. chunked, filtered, or variable-size units):
1. Let $U$ be the number of independent replicate units in a condition cell.
2. Let $N_u$ be the number of valid samples in replicate unit $u \in \{1, \dots, U\}$, forming sample count vector $\mathbf{n} = [N_1, \dots, N_U]^T \in \mathbb{N}^U$.
3. Let $T \in \mathbb{R}^{U \times D}$ be the matrix of unit sum statistics:
   $$T_{u, d} = \sum_{i=1}^{N_u} x_{u, i, d} = N_u \cdot y_{u, d}$$
   where $y_{u, d}$ is the observable mean of unit $u$ on metric $d$.
4. Define bootstrap iteration count $B = 2,000$.
5. Generate multinomial cluster resampling weight matrix $W \in \mathbb{N}^{B \times U}$:
   $$W_{b, :} \sim \text{Multinomial}\left(U, \, \left[\frac{1}{U}, \frac{1}{U}, \dots, \frac{1}{U}\right]\right)$$
6. Compute all $B$ bootstrap replication estimates via the vectorized **Cluster Ratio Estimator**:
   $$\hat{\Theta}^* = \frac{W \cdot T}{(W \cdot \mathbf{n})[:, \text{None}]} \quad \in \mathbb{R}^{B \times D}$$
   In Python/NumPy:
   ```python
   # T: (U, D) matrix of unit sum statistics (N_u * y_u)
   # n_vec: (U, 1) column vector of unit sample counts
   # W: (B, U) multinomial cluster resample weights
   numerator = W @ T                      # shape (B, D)
   denominator = W @ n_vec                # shape (B, 1)
   theta_boot = numerator / denominator   # shape (B, D)
   ```
   *Parity Proof*: When all units have equal sample count $N_u = N$:
   $$W \cdot \mathbf{n} = N \sum_{u=1}^U W_{b, u} = N \cdot U$$
   $$\hat{\Theta}^* = \frac{N \cdot (W \cdot Y)}{N \cdot U} = \frac{1}{U}(W \cdot Y)$$
   It reduces identically to unweighted pooling, while remaining mathematically unbiased when cluster sizes vary.
7. Compute empirical percentile confidence intervals along axis 0 at $[\alpha/2, 1 - \alpha/2]$ (e.g. 2.5% and 97.5%):
   $$\text{CI}_{95\%} = \left[ q_{0.025}(\hat{\Theta}^*), \, q_{0.975}(\hat{\Theta}^*) \right]$$
- **Microsecond Execution**: In NumPy/BLAS, this matrix ratio evaluates in $< 120$ microseconds for $U=10, B=2,000, D=5$.

#### 5.3.3 Small-Sample Guard ($U < 3$) & Demonstration Contrast
When replicate count $U < 3$:
- Resampling $U=2$ units yields only 3 distinct non-trivial weight allocations, producing a degenerate pseudo-interval.
- **Mandatory Guard**: When $U < 3$, the engine MUST set `descriptive_only = True`, suppress CI error bars in visualization specifications, and emit the warning: `"Warning: Replicate count U < 3; reporting descriptive point estimates only."`
- **Demonstration Contrast**: To ensure that the primary demonstration campaign actively exercises and displays full bootstrap confidence intervals, the official Ubiquitin demo campaign (§8) specifies $U = 3$ seeds (`[1001, 1002, 1003]`), satisfying $U \ge 3$. A dedicated test fixture with $U=1, 2$ in §7 / P5-09 verifies that the small-sample guard properly activates and suppresses intervals when $U < 3$.

#### 5.3.4 Minimum Detectable Effect ($\text{MDE}_{0.80}$) & Power Discipline
Before concluding that two knob settings have "no effect" ($\Delta = 0$):
1. For contrast estimate $\hat{\Delta} = \bar{Y}_B - \bar{Y}_A$ with standard error $\text{SE}(\hat{\Delta})$:
   $$\text{MDE}_{0.80} = (z_{0.975} + z_{0.80}) \cdot \text{SE}(\hat{\Delta}) = (1.960 + 0.842) \cdot \text{SE}(\hat{\Delta}) \approx 2.802 \cdot \text{SE}(\hat{\Delta})$$
2. **Inference Reporting Rules**:
   - If $|\hat{\Delta}| \ge \text{MDE}_{0.80}$ and $0 \notin \text{CI}_{95\%}$: Reported as **statistically detected and powered**.
   - If $|\hat{\Delta}| < \text{MDE}_{0.80}$ and $0 \in \text{CI}_{95\%}$: Reported as **indeterminate (underpowered)**. The report MUST emit: `"Indeterminate: contrast underpowered (observed effect < MDE)"`.
   - Declaring "no effect" without passing an MDE check or synthetic control is strictly prohibited.

---

### 5.4 Question 4: Coverage Reconciliation

The EDA engine evaluates execution completeness across the full planned Cartesian sweep grid:
1. Planned grid set: $\mathcal{G} = \{ \text{row.manifest\_row\_hash} \mid \text{row} \in \text{manifest.rows} \}$.
2. **Reconciliation States & Diagnostics**:
   - **Committed** (`COMMITTED`): Valid `IndexReuse` from unit index with verifying done marker (`done.json` or `unit.json`).
   - **Partial** (`PARTIAL`): Active, unexpired `CampaignLock` held on filesystem with actively updating heartbeat, or unsealed staging directory `<store>.tmp` actively being written.
   - **Failed** (`FAILED`): Lookup returned `Corrupt`, done marker content hash mismatch, output quarantined, or **abandoned staging directories (`*.tmp`), expired locks, or non-zero executor exit records**. Such abandoned/killed units (e.g. killed by Slurm walltime or OOM `SIGKILL`) are classified as `FAILED` with diagnostic reason `killed_incomplete`, preventing them from being falsely reported as unstarted.
   - **Missing** (`MISSING`): In planned grid $\mathcal{G}$ with strictly zero disk trace (never scheduled, queued, or started).
3. **Cartesian Grid Visual Cell Aggregation Rule**:
   A visual sweep grid condition (e.g. arm $\times$ temperature) typically aggregates multiple constituent manifest rows across replication seeds ($U$) or chunks ($K$):
   - A visual grid cell is **`COMMITTED` if and only if 100% of its planned constituent rows are committed**.
   - A visual grid cell is **`PARTIAL`** if at least one constituent row is in progress (`PARTIAL`) and no row has failed.
   - A visual grid cell is **`FAILED`** if any constituent row is `FAILED` (including `killed_incomplete`) or quarantined.
   - A visual grid cell is **`MISSING`** if 100% of constituent rows have zero disk trace.
4. **Explicit Visualization**:
   - In sweep grid charts (`vegalite.sweep_grid`), every cell in $\mathcal{G}$ is explicitly plotted.
   - Committed cells are drawn in solid status color (green or theme primary).
   - Missing cells are drawn as hatched grey blocks.
   - Partial cells are drawn in amber with an in-progress spinner badge.
   - Failed cells are drawn in bold red with hoverable error tooltips detailing the diagnostic failure reason (e.g. `killed_incomplete`, `corrupt_digest`, or plugin quarantine reason).
   - Dropping missing or failed cells from the visual matrix is strictly prohibited.

---

### 5.5 Question 5: Secondary Run View (Run Telemetry)

Compiler and execution telemetry is strictly isolated into a secondary dashboard tab to prevent conflation with scientific metrics:
1. **Wall-Time Profiling**:
   - Extracted from `CampaignDoneMarker.duration_s`.
   - Displays compute duration distributions across arm types, sequence lengths, and hardware workers.
2. **Reuse & Resume Logs**:
   - Ingests `<campaign_dir>/reuse_log.jsonl`.
   - Visualizes cache hit rates, re-verified units, and avoided compute hours.
3. **Debt #3300 Pack Telemetry Seam**:
   - In accordance with debt #3300, pack event telemetry is not yet instrumented inside the execution loop.
   - The Secondary Run View MUST display: `"Pack Telemetry: not instrumented (debt #3300)"`.
   - Simultaneously, static archive status (`is_packed: true/false`) is displayed per unit.
4. **Compiler Tiling & Memory Telemetry**:
   - Displays `BatchPlanLike` stats (`Vmap` vs. `Scan` vs. `ChunkedMap` strategy counts).
   - Displays `memory_estimate_bytes` and surfaces compiler `memory_warnings`.

---

## 6. Neutral Export Artifacts & Visualization Architecture

### 6.1 Canonical File Artifacts & RFC 8259 Sanitization
Campaign EDA compiles analysis results into a self-contained, content-addressed output bundle under `<campaign_dir>/eda/`:

```
<campaign_dir>/eda/
├── manifest.json                  # SHA-256 cryptographic manifest of all export files
├── summary.json                   # Canonical RFC 8259 campaign summary record
├── tables/
│   ├── library_metrics.parquet    # Tidy long-format table of all cell metrics
│   ├── coverage_grid.parquet      # Sweep grid status reconciliation table
│   └── run_telemetry.parquet      # Wall-time and reuse telemetry table
├── specs/
│   ├── library_composition.vl.json # Vega-Lite v5 declarative specification
│   ├── score_distributions.vl.json
│   ├── uncertainty_contrasts.vl.json
│   ├── sweep_coverage.vl.json
│   └── run_telemetry.vl.json
├── static/
│   ├── library_composition.svg    # Pre-rendered vector SVG (Reviewed Object)
│   ├── score_distributions.svg
│   ├── uncertainty_contrasts.svg
│   ├── sweep_coverage.svg
│   └── run_telemetry.svg
└── index.html                     # Standalone progressive airgapped HTML dashboard
```

#### Recursive RFC 8259 Sanitization Pass
To ensure universal compliance with RFC 8259 across all exported JSON artifacts (`summary.json`, `manifest.json`, and `specs/*.vl.json`), the serializer executes a recursive sanitization pass:
- All `np.nan` and `math.isnan(v)` values are coerced to Python `None` (which serializes to standard JSON `null`).
- All `math.isinf(v)` values are coerced to `None` with a diagnostic warning flag.
- Floating `-0.0` values are normalized strictly to `0.0`.
- The serialization pipeline executes strictly with `json.dumps(obj, allow_nan=False, indent=2)`, preventing browser `JSON.parse()` crashes.
- In Vega-Lite specs (`specs/*.vl.json`), data transforms include explicit null filters (e.g. `{"filter": "isValid(datum.score) && datum.score != null"}`) so missing or null metrics never distort quantitative scales.

#### Cryptographic Manifest Contract (`manifest.json`)
```json
{
  "schema_version": "xtrax_eda_manifest_v1",
  "campaign_id": "tev_design_ar_2026",
  "generated_at_utc": "2026-11-06T20:30:00Z",
  "engine_version": "0.4.1b7",
  "artifacts": {
    "summary.json": "3a7b8e...64hex",
    "tables/library_metrics.parquet": "9f1c2d...64hex",
    "static/library_composition.svg": "a8e4b1...64hex"
  }
}
```

### 6.2 Dual-Layer Visual Presentation: Reviewed Object Protocol
To support both rigorous publication review and interactive exploration:
1. **The Static SVG is the Reviewed Object**:
   - Every visualization panel is compiled to a pure static vector SVG using `vl-convert-python` (or headless Chromium).
   - The SVG content is hashed with SHA-256 and registered in `manifest.json`.
   - In scientific peer review and publication workflows, human reviewers inspect and promote the exact vector paths of the static SVG.
2. **Progressive Airgapped HTML Dashboard**:
   - The generated `index.html` file embeds the pre-rendered static SVGs directly inline within uniquely named panel containers: `id="fallback-svg-{panel_name}"`.
   - External library scripts use deferred loading (`<script defer src="...">`) to prevent blocking the HTML parser during network timeouts on airgapped HPC nodes.
   - **Static SVGs Visible by Default**: Static vector SVGs are styled `display: block` by default, guaranteeing immediate visibility even if JavaScript is completely disabled.
   - **2-Second Timeout Guard**: If external network access is blocked, Vega-Embed initialization aborts cleanly after 2 seconds, preserving the crisp inline static SVG.

```html
<!-- Example for panel: library_composition -->
<div class="panel-container" id="panel-library-composition">
  <!-- Static Reviewed Object Fallback (Always Visible by Default Offline) -->
  <div id="fallback-svg-library-composition" class="svg-container">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500">
      <!-- Pre-rendered vector paths -->
    </svg>
  </div>
  <!-- Interactive Explorer (Progressively Enhanced if JS & Vega Load) -->
  <div id="interactive-vis-library-composition" style="display: none;"></div>
</div>

<script defer>
  (function() {
    var panelName = "library-composition";
    var timeout = setTimeout(function() {
      console.warn("Vega CDN timed out (airgapped mode); static SVG preserved for panel: " + panelName);
    }, 2000);
    
    if (typeof vegaEmbed !== "undefined") {
      vegaEmbed('#interactive-vis-' + panelName, spec).then(function() {
        clearTimeout(timeout);
        var fallback = document.getElementById('fallback-svg-' + panelName);
        var interactive = document.getElementById('interactive-vis-' + panelName);
        if (fallback && interactive) {
          fallback.style.display = 'none';
          interactive.style.display = 'block';
        }
      }).catch(function(err) {
        console.error("Vega render failed; falling back to static SVG for panel: " + panelName, err);
      });
    }
  })();
</script>
```

---

## 7. Verification & Statistical Controls Suite

Before trusting Campaign EDA outputs, the analytical engine must pass a suite of five targeted verification controls:

### 7.1 Control 1: Synthetic Positive Control (`test_positive_planted_knob_effect`)
- **Objective**: Verify that the statistical inference engine detects a real knob effect without attenuation.
- **Fixture**: A synthetic campaign with 2 knob levels ($K_0, K_1$) across $U=6$ independent seeds. An effect $\Delta^* = +1.50$ is planted in $K_1$.
- **Pass Criteria**:
  1. Estimated contrast $\hat{\Delta} \in [1.35, 1.65]$.
  2. $0 \notin \text{CI}_{95\%}$ (effect statistically detected).
  3. $|\hat{\Delta}| \ge \text{MDE}_{0.80}$ (effect powered).

### 7.2 Control 2: Synthetic Negative Control (`test_negative_null_campaign_no_effect`)
- **Objective**: Verify that the engine controls false positives under a true null hypothesis.
- **Fixture**: A synthetic null campaign where $K_0$ and $K_1$ draw from identical distributions.
- **Pass Criteria**:
  1. $0 \in \text{CI}_{95\%}$.
  2. Across 100 independent null simulations, false positive rejection rate $\le 5\%$.

### 7.3 Control 3: Vectorization Parity (`test_batched_equals_looped`)
- **Objective**: Verify that the vectorized matrix cluster ratio bootstrap ($\hat{\Theta}^* = \frac{W \cdot T}{(W \cdot \mathbf{n})[:, \text{None}]}$) produces numerically identical results to an iterative Python loop over draws, across both equal and unequal cluster sizes.
- **Fixture**: $U=8$ units, unequal sample counts $N_u \in [10, 50, 100, 20, 80, 40, 60, 30]$, $D=4$ metrics, $B=2,000$ resamples with identical random seed.
- **Pass Criteria**:
  - `np.allclose(ci_vectorized, ci_looped, atol=1e-12) == True`.
  - When cluster sizes are equal ($N_u = N$), the ratio bootstrap matches the unweighted bootstrap identically: `np.allclose(theta_ratio, theta_unweighted, atol=1e-12) == True`.
  - The ratio bootstrap converges to the true pooled sample mean as cluster count scales, eliminating the permanent asymptotic bias of the unweighted estimator.

### 7.4 Control 4: Storage Parity (`test_packed_equals_loose_read`)
- **Objective**: Verify that packed units (`.tar.zst`) and loose directories produce bit-identical metrics.
- **Fixture**: A test unit written loose, then packed via `xtrax.pack.pack_unit()`.
- **Pass Criteria**:
  - Extracted `LibraryCell` objects, metric dictionaries, and SVG hashes match bit-for-bit:
    `hash(loose_cell) == hash(packed_cell)`.

### 7.5 Control 5: Uncommitted Unit Refusal (`test_uncommitted_refusal_and_banner`)
- **Objective**: Verify that uncommitted, partial, or corrupt units are excluded from statistical pooling.
- **Fixture**: A campaign containing 1 committed unit, 1 active-lease partial unit, and 1 corrupt done-marker unit.
- **Pass Criteria**:
  - Statistical tables pool only the 1 committed unit.
  - Question 4 coverage table explicitly classifies all 3 units (`COMMITTED`, `PARTIAL`, `FAILED`).
  - Warning banners are rendered for the partial and failed units.

### 7.6 Control 6: Small-Sample Guard Verification (`test_small_sample_guard_suppression`)
- **Objective**: Verify that condition cells with replicate counts $U < 3$ suppress confidence intervals and error bars.
- **Fixture**: Synthetic conditions with $U=1$ and $U=2$ independent seeds.
- **Pass Criteria**:
  1. Statistical records report `descriptive_only == True`.
  2. Point estimates are reported, but confidence interval bounds are emitted as `null`.
  3. Vega-Lite specifications omit error bar mark layers for the guarded condition.
  4. Diagnostic warning emitted: `"Warning: Replicate count U < 3; reporting descriptive point estimates only."`.

---

## 8. Real End-to-End aminx Demo Campaign

To validate production readiness, the test suite executes a real end-to-end `aminx` campaign on a local CPU/GPU runner:
- **Target Protein**: Ubiquitin (`tests/fixtures/1ubq.pdb`, 76 residues; vendored standard PDB fixture, single-chain, avoids parser chain reordering ambiguities).
- **Arms**:
  1. `proteinmpnn` (`proteinmpnn_v_48_020`)
  2. `solublempnn` (`solublempnn_v_48_020`)
- **Knob Grid**:
  - `temperature`: `[0.1, 0.3, 0.5]` (3 levels, showing progressive diversity vs. recovery trade-offs).
  - `seed`: `[1001, 1002, 1003]` (3 independent replication seeds, satisfying $U = 3 \ge 3$).
- **Execution & Output**:
  - 2 arms $\times$ 3 temperatures $\times$ 3 seeds = 18 total units $\times$ 16 samples = 288 generated sequences.
  - Sinks committed to disk in both loose and packed (`members.tar.zst`) formats.
  - Total local runtime: $< 45$ seconds on local CPU.
  - Replicate count $U=3$ guarantees that cluster ratio bootstrap confidence intervals, error bars, and MDE evaluations are fully active on real generated sequences, avoiding small-sample guard suppression.
  - Verified outputs: Full EDA bundle generated under `<campaign_dir>/eda/`, answering all 5 questions on real model outputs.

---

## 9. Discrete Fixer Tasks for Phase P5 Implementation

Implementation is partitioned into 10 modular, file-disjoint work packages to enable clean multi-agent execution:

```
+----------------------------------------------------------------------------------------+
|                                Phase P5 Implementation Plan                            |
+----------------------------------------------------------------------------------------+
|  [P5-01] Core Types & Schemas               --> src/xtrax/eda/campaign/schema.py       |
|  [P5-02] Storage Parity & Unit Reader       --> src/xtrax/eda/campaign/reader.py       |
|  [P5-03] Domain Plugin Registry & Protocol  --> src/xtrax/eda/campaign/domains/        |
|  [P5-04] aminx Domain Plugin                --> src/xtrax/eda/campaign/domains/aminx.py|
|  [P5-05] demistify Domain Plugin            --> src/xtrax/eda/campaign/domains/demist. |
|  [P5-06] Vectorized Statistics Engine       --> src/xtrax/eda/campaign/stats.py        |
|  [P5-07] Vega-Lite Specification Engine     --> src/xtrax/eda/campaign/spec.py         |
|  [P5-08] Neutral Export & HTML Renderer     --> src/xtrax/eda/campaign/export.py       |
|  [P5-09] Verification Controls Test Suite   --> tests/eda/campaign/test_controls.py    |
|  [P5-10] E2E aminx Demo & CLI Verb          --> src/xtrax/cli/campaign_eda_verb.py     |
+----------------------------------------------------------------------------------------+
```

### Work Package Details
1. **P5-01: Core Types, Schemas & Extras Declaration (`src/xtrax/eda/campaign/schema.py`, `pyproject.toml`)**
   - Implement `SinkArray`, `SinkSchema`, `LibraryCell`, `ResidueTable`, `DomainMetricResult`.
   - Implement `DomainMetricResult.validate_schema()` enforcing strict numeric finiteness (`isinstance(v, (int, float))` and `np.isfinite(v)`), quarantining invalid outputs to prevent PyArrow conversion failures (MAJ-02).
   - Core modules depend strictly and exclusively on `stdlib + numpy`.
   - Update `pyproject.toml` to declare optional extras `xtrax[eda-campaign]` (`duckdb >= 1.0.0`, `pyarrow >= 14.0.0`), `xtrax[eda-viz]` (`altair >= 5.0.0`, `vl-convert-python >= 1.3.0`), entry-point group `[project.entry-points."xtrax.eda.domains"]`, and automated `tool.importlinter` boundary contracts (MAJ-06, MIN-03).
2. **P5-02: Storage Parity & Unit Reader (`src/xtrax/eda/campaign/reader.py`)**
   - Implement `CampaignReader` integrating `unit_index.lookup()` and `xtrax.pack.read_unit_member()`.
   - Harmonize storage reading via `read_campaign_unit_member` to discover and accept both `done.json` and `unit.json` marker files, eliminating member lookup failures on swept packed units (BLK-07).
   - Inspect filesystem for abandoned staging directories (`*.tmp`), expired locks, or non-zero executor exit records, classifying them as `FAILED` with diagnostic reason `killed_incomplete` (MAJ-03).
   - Ensure bit-identical reads across loose directories and `members.tar.zst`.
3. **P5-03: Domain Plugin Registry & Immutability Isolation (`src/xtrax/eda/campaign/domains/`)**
   - Define `DomainPlugin` protocol and entry-point discovery under `xtrax.eda.domains`.
   - Replace deep copying with zero-copy immutability isolation by setting `flags.writeable = False` on all NumPy arrays in `LibraryCell` (MAJ-01).
   - Implement plugin quarantine execution handler wrapping `try: ... except (Exception, SystemExit) as exc:` to prevent `SystemExit` from terminating the runner process (BLK-06).
   - Validate metric outputs via `DomainMetricResult.validate_schema()` before accepting cells (MAJ-02).
4. **P5-04: aminx Domain Plugin (`src/xtrax/eda/campaign/domains/aminx.py`)**
   - Implement sequence extraction, MPNN alphabet normalization, native recovery, shuffled recovery margin, and derived NLL.
5. **P5-05: demistify Domain Plugin (`src/xtrax/eda/campaign/domains/demistify.py`)**
   - Implement ciMIST entropy, MI matrix, and MIST tree edge extraction.
   - Implement Schema B degradation handler (`matrix_fullness: false`).
6. **P5-06: Vectorized Statistics Engine (`src/xtrax/eda/campaign/stats.py`)**
   - Implement vectorized **Cluster Ratio Bootstrap**: $\hat{\Theta}^* = \frac{W \cdot T}{(W \cdot \mathbf{n})[:, \text{None}]}$ with $T_{u, d} = N_u \cdot y_{u, d}$ and $\mathbf{n} = [N_1, \dots, N_U]^T$, eliminating statistical bias across unequal cluster sizes (BLK-01).
   - Implement pairwise identity diversity with explicit boundary guards for $N \le 1$ and $|M|=0$ returning 0.0 with diagnostic flags (BLK-04).
   - Implement safe positive-support Shannon entropy $H = -\sum_{c: p(c)>0} p(c) \log_2 p(c) \ge 0.0$ bits without negative bits or divide-by-zero warnings (MAJ-07).
   - Implement Best-of-$N$ candidate grid bounding ($n \le N$) and iterative ratio recurrence $\omega_j(n, N)$ avoiding float64 overflow (BLK-03).
   - Implement Kung's non-dominated 2D Pareto sorting minimizing both $\text{NLL}$ and $\text{PID}$ ($\min(\text{NLL}), \min(\text{PID})$) (MIN-02).
   - Implement cross-knob sequence collision table across temperature levels and arms (MIN-01).
   - Implement small-sample guard ($U < 3$) suppressing CIs and setting `descriptive_only = True`.
   - Implement Minimum Detectable Effect ($\text{MDE}_{0.80}$) and power checks.
7. **P5-07: Vega-Lite Specification Engine (`src/xtrax/eda/campaign/spec.py`)**
   - Implement declarative Vega-Lite v5 JSON generators for the 5 scientific questions.
   - Add explicit null filter transforms in quantitative encodings (`datum.score != null`) (BLK-05).
   - Implement visual sweep grid cell aggregation rule: a cell is `COMMITTED` iff 100% of planned constituent rows are committed, `PARTIAL` if in progress, `FAILED` if any row failed or is quarantined, and `MISSING` if unstarted (MAJ-04).
8. **P5-08: Neutral Export & Standalone HTML Renderer (`src/xtrax/eda/campaign/export.py`, `viz.py`)**
   - Implement recursive RFC 8259 sanitization pass coercing `NaN` and `+/-Inf` to `None`/`null` and `-0.0` to `0.0`, validating with `json.dumps(obj, allow_nan=False)` (BLK-05).
   - Implement Parquet and SHA-256 manifest writers.
   - Implement standalone progressive HTML dashboard with deferred script loading (`<script defer>`), pre-rendered vector SVGs visible by default offline, unique DOM IDs per panel (`id="fallback-svg-{panel_name}"`), and 2-second timeout guard (MAJ-08).
9. **P5-09: Verification Controls Test Suite (`tests/eda/campaign/test_controls.py`, `tests/fixtures/1ubq.pdb`)**
   - Vendor standard 76-residue ubiquitin PDB coordinates into `tests/fixtures/1ubq.pdb` (MAJ-05).
   - Implement verification controls: synthetic positive, synthetic negative, vectorization parity (cluster ratio estimator across equal and unequal cluster sizes, BLK-01), storage parity (loose vs packed), uncommitted refusal.
   - Implement small-sample guard test fixture with $U=1$ and $U=2$ verifying interval suppression and banner emission (BLK-02).
10. **P5-10: E2E aminx Demo & CLI Verb (`src/xtrax/cli/campaign_eda_verb.py`)**
    - Implement `xtrax campaign eda <campaign_dir>` CLI command.
    - Wire ubiquitin demo campaign execution with $U = 3$ seeds (`[1001, 1002, 1003]`, 288 sequences) to ensure full bootstrap CIs and error bars are active on real data without small-sample guard suppression (BLK-02).
    - Validate output bundle generation and answers to all 5 questions.

---

## 10. Acceptance Criteria & Audit Checklist

The implementation shall be deemed complete and ready for production only when all of the following criteria are independently verified:

### Architecture & Boundaries
- [ ] **AC-01 (Decision 1 Superseded)**: Campaign-level tables are first-class, active, and fully implemented.
- [ ] **AC-02 (Decision 3 Minimal Core)**: Core modules depend strictly on `stdlib + numpy`; `duckdb`, `pyarrow`, and `altair` reside strictly behind optional extras declared in `pyproject.toml`.
- [ ] **AC-03 (F-10 Boundary)**: Zero cross-imports between `xtrax.eda` and `maraxiom.eda`; zero internal imports of `bathos`, `maraxiom`, or `cisternal`, enforced via `import-linter`.
- [ ] **AC-04 (Storage Parity)**: Loose directories and packed `.tar.zst` archives produce bit-identical data representations via dual-marker `CampaignReader`.
- [ ] **AC-05 (Committed Reads Only)**: Uncommitted, partial, or corrupt units are excluded from statistical pooling and bannered in coverage dashboards; abandoned staging stores classified as `FAILED` (`killed_incomplete`).
- [ ] **AC-06 (Plugin Quarantine & Immutability)**: Third-party domain plugin exceptions and `SystemExit` are quarantined without crashing the runner; zero-copy array immutability (`flags.writeable = False`) prevents state corruption.

### Scientific Analytical Questions
- [ ] **AC-07 (Q1 Composition)**: Pairwise identity diversity with boundary guards ($N \le 1, |M|=0$), safe positive-support Shannon entropy ($H \ge 0.0$), composition entropy, target recovery, duplicate rates, and cross-knob collision tables reported.
- [ ] **AC-08 (Q1 Masking & Margin)**: Unmasked padding tokens strictly excluded; recovery margin vs. shuffled native $< 0.10$ triggers degeneracy banner.
- [ ] **AC-09 (Q2 Scores)**: NLL distributions, derived NLL fallback from logits, perplexity, Best-of-$N$ curves with bounded candidate grid ($n \le N$) and ratio recurrence, and Pareto frontiers ($\min(\text{NLL}), \min(\text{PID})$) computed.
- [ ] **AC-10 (Q3 Uncertainty)**: Seeds pooled as samples per `md-replicas-are-samples`; vectorized cluster ratio bootstrap intervals reported without cluster-size bias; $U < 3$ triggers `descriptive_only = True`; underpowered contrasts flagged via MDE check.
- [ ] **AC-11 (Q4 Coverage)**: Sweep grid status explicitly differentiates committed, partial, failed (`killed_incomplete`), and missing cells; visual grid cells require 100% committed constituent rows; missing cells never dropped.
- [ ] **AC-12 (Q5 Telemetry)**: Per-unit wall times, cache reuse logs, and tiling plans displayed; pack telemetry explicitly bannered as `"not instrumented (debt #3300)"`.

### Verification Controls & Parity
- [ ] **AC-13 (Positive Control)**: Synthetic positive control captures planted knob effect ($\hat{\Delta} \in \text{CI}_{95\%}, 0 \notin \text{CI}_{95\%}$).
- [ ] **AC-14 (Negative Control)**: Synthetic null control yields $0 \in \text{CI}_{95\%}$ with $\le 5\%$ false positive rate.
- [ ] **AC-15 (Vectorization Parity)**: `np.allclose(ci_vectorized, ci_looped, atol=1e-12)` passes for all cluster ratio bootstrap statistics across equal and unequal cluster sizes.
- [ ] **AC-16 (E2E Demo)**: Ubiquitin aminx demo campaign ($U=3$ seeds [1001, 1002, 1003], 288 sequences, `tests/fixtures/1ubq.pdb`) runs locally, outputs valid EDA bundle, and passes end-to-end verification with active bootstrap intervals.
