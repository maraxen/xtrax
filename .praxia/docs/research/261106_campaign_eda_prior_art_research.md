---
task_id: 261106_xtrax-campaign-eda
title: "Research Note: Prior Art, Design Options, and Reference Implementations for xtrax Campaign EDA"
category: research
status: complete
date: 2026-11-06
author: "teamwork_explorer_3"
target_spec: ".praxia/docs/specs/261106_campaign-eda-redesign.md"
tags: [campaign, eda, prior-art, research, praxia]
---

# Research Note: Prior Art, Design Options, and Reference Implementations for xtrax Campaign EDA

- **Task**: `261106_xtrax-campaign-eda`
- **Date**: 2026-11-06
- **Author**: `teamwork_explorer_3` (Prior Art & Design Options Researcher)
- **Status**: Completed (Phase P1 Milestone)
- **Target Specification**: `.praxia/docs/specs/261106_campaign-eda-redesign.md`
- **Scope**: Comprehensive survey, prior art analysis, and design options evaluation across experiment tracking dashboards, factorial design of experiments (DOE), sequence/structure biomolecular generation metrics, columnar engines, grammar-of-graphics rendering, and local reference implementations.

---

## 1. Executive Summary & Verdict Registry

This research note investigates prior art and architectural design options to ground the implementation of Campaign Exploratory Data Analysis (EDA) in `xtrax`. The primary objective of Campaign EDA is to evaluate what multi-dimensional knob grids ($\text{arm} \times \text{checkpoint} \times \text{policy} \times \text{parameters} \times \text{seed} \times \text{target}$) produced across finished or running campaigns, answering five core scientific questions:
1. **Library Composition**: Diversity, per-position composition and Shannon entropy, target/wild-type recovery, duplicate rates, and shifts across knob levels.
2. **Scores**: Per-sample and per-arm score/NLL distributions, ranks, Best-of-$N$ / top-$k$ enrichment curves, and score-versus-diversity Pareto frontiers.
3. **Knob Effects with Honest Uncertainty**: Independent replication units respecting the hierarchical seed/build structure, vectorized cluster bootstrap intervals, Minimum Detectable Effect (MDE) calculations, and small-sample guards.
4. **Coverage**: Planned Cartesian sweep grid reconciliation against committed sinks, showing committed, partial, failed, and missing cells without silent omission.
5. **Secondary Run View**: Unit compute wall time, reuse/resume decisions, pack events telemetry (with debt #3300 bannering), and tiling plan memory budgets.

### Master Verdict Registry

| Area / Technology | Candidate / Concept | Verdict | Core Technical Rationale |
|---|---|---|---|
| **Dashboards** | Weights & Biases (W&B) | **ADAPT** (UX) / **REJECT** (Runtime) | Adapt parallel coordinates and dynamic multi-knob grouping; reject proprietary cloud server, SaaS dependency, network egress, and naive aggregation. |
| **Dashboards** | MLflow | **ADAPT** (Schema) / **REJECT** (Server) | Adapt run-vs-sweep hierarchy and artifact logging semantics; reject heavy SQL/HTTP tracking server and SQLite NFS locking hazards. |
| **Dashboards** | Aim | **ADAPT** (Queries) / **REJECT** (Engine) | Adapt sub-second slice-and-dice query semantics and hyperparameter grouping; reject C++/RocksDB binary dependencies and daemon process. |
| **Dashboards** | TensorBoard (HParams) | **ADAPT** (Layout) / **REJECT** (Format) | Adapt filesystem-first read-only consumption and HParams tabular layout; reject protobuf event files and unprincipled sample averaging. |
| **Factorial DOE** | Full & Fractional Factorial Grids | **ADOPT** | Adopt full Cartesian product representation (`CampaignManifest`) to model orthogonal knob interactions; explicitly visualize missing cells. |
| **Factorial DOE** | Response Surface Methodology (RSM) | **ADAPT** (Curves) / **REJECT** (Surfaces) | Adapt two-factor interaction plots; reject parametric polynomial surface smoothing in favor of empirical non-parametric pooled estimates. |
| **Factorial DOE** | Contrast Estimation ($\hat{\Delta}$) | **ADOPT** | Adopt linear contrasts and difference-in-differences to quantify marginal and joint knob effects. |
| **Uncertainty** | Classical Hypothesis Tests ($t$/ANOVA) | **REJECT** | Reject p-values over raw samples ($N \to \infty$ pseudoreplication); violates exchangeability and creates spurious statistical significance. |
| **Uncertainty** | Global MD-Replicas Hierarchy | **ADOPT** | Adopt hierarchy ($\text{frames} \subset \text{seeds} \subset \text{build} \subset \text{condition}$); seeds are samples of an unconverged estimator, never independent $N$. |
| **Uncertainty** | Vectorized Cluster Bootstrap ($W \cdot Y / U$) | **ADOPT** | Adopt multinomial matrix weighting in NumPy/Arrow for microsecond empirical percentile intervals without Python loops. |
| **Uncertainty** | Minimum Detectable Effect ($\text{MDE}_{0.80}$) | **ADOPT** | Adopt $\text{MDE}_{0.80} \approx 2.802 \cdot \text{SE}(\hat{\Delta})$; enforce power check before reporting "no effect". |
| **Uncertainty** | Small-Sample Guard ($U < 3$) | **ADOPT** | Adopt mandatory `descriptive_only = True` flag when replicate count $U < 3$; suppress degenerate confidence intervals. |
| **Biomolecular EDA** | Pairwise Identity & Hamming Diversity | **ADOPT** | Adopt vectorized upper-triangle matrix comparisons masked by real-residue designable masks. |
| **Biomolecular EDA** | Positional & Composition Shannon Entropy | **ADOPT** | Adopt bit-entropy ($H = -\sum p \log_2 p$) for positions and overall amino acid composition; detect library collapse. |
| **Biomolecular EDA** | Real-Residue Masking | **ADOPT** | Adopt strict masking (`real_mask & ~fixed_mask`); unmasked padding tokens ruin entropy calculations ($H \approx 4.04$ bits). |
| **Biomolecular EDA** | Target Recovery & Recovery Margin | **ADOPT** | Adopt recovery against WT and margin against shuffled native ($\text{recovery} - \text{recovery}_{shuf} \ge 0.10$). |
| **Biomolecular EDA** | Typed Alphabet Mappings | **ADOPT** | Enforce explicit conversion between MPNN order (`ACDEFGHIKLMNPQRSTVWYX`) and AlphaFold order (`ARNDCQEGHILKMFPSTWYVX`). |
| **Biomolecular EDA** | NLL & Perplexity Distributions | **ADOPT** | Adopt negative log-likelihood with derived fallback from logits ($\text{derived\_nll} = \text{LSE}(\text{logits}) - \text{logits}[s]$). |
| **Biomolecular EDA** | Best-of-$N$ & Top-$k$ Enrichment Curves | **ADOPT** | Adopt order statistics evaluating expected top performance $E[\min_{i=1..n} \text{score}_i]$ as candidate library size $n$ scales. |
| **Biomolecular EDA** | Score-Diversity Pareto Frontiers | **ADOPT** | Adopt non-dominated Pareto sorting to evaluate trade-offs between model confidence and sequence diversity. |
| **Biomolecular EDA** | Conformational Ensembles (ciMIST) | **ADOPT** (Metrics) / **ADAPT** (Schema B) | Adopt per-residue entropy $S_i$, total MIST entropy, and tree edges; adapt hub scoring when Schema B lacks dense MI matrix. |
| **Columnar Data** | DuckDB | **ADOPT** | Adopt DuckDB OLAP engine for zero-copy SQL analytics over partitioned Parquet datasets; place behind `xtrax[eda-campaign]`. |
| **Columnar Data** | Apache Arrow (PyArrow) | **ADOPT** | Adopt PyArrow as the in-memory columnar schema and Parquet I/O substrate behind `xtrax[eda-campaign]`. |
| **Columnar Data** | Polars | **REJECT** | Reject Polars as a direct dependency to eliminate bloat; DuckDB + PyArrow + NumPy cover 100% of required operations. |
| **Columnar Data** | Apache Parquet | **ADOPT** | Adopt Parquet as the canonical partitioned columnar on-disk storage format for campaign summary tables. |
| **Columnar Data** | NumPy Host Vectorization | **ADOPT** | Adopt NumPy for core mathematical transformations and matrix bootstrapping in the zero-extra base engine. |
| **Grammar of Graphics** | Vega-Lite | **ADOPT** | Adopt declarative JSON specification schema for all exploratory visualization panels. |
| **Grammar of Graphics** | Altair | **ADAPT** | Adapt Altair as an optional Pythonic spec builder behind extras; core engine emits pure JSON dictionaries without runtime Altair requirement. |
| **Grammar of Graphics** | Observable Plot | **REJECT** | Reject Observable Plot due to Node.js/browser runtime requirement and lack of standalone declarative JSON schema. |
| **Viz Architecture** | Static SVG Reviewed Object | **ADOPT** | Adopt static SVG (content-hashed with SHA-256) as the primary reviewed artifact; interactive Vega-Embed is an unhashed convenience layer. |
| **Viz Architecture** | Progressive HTML Airgapped Fallback | **ADOPT** | Adopt progressive HTML with inline SVG fallback and timeout guard to ensure visibility on airgapped HPC cluster nodes. |
| **Local References** | Maraxiom Review Pages | **ADOPT** (Artifacts) / **REJECT** (Import) | Adopt component taxonomy and neutral export artifact format; strictly maintain F-10 boundary (zero maraxiom imports in xtrax). |
| **Local References** | Bathos Campaigns | **ADOPT** (Controls) / **REJECT** (Import) | Adopt pre-registration and synthetic positive/negative control discipline; strictly maintain Decision 3 (zero bathos bridges in xtrax). |
| **Local References** | Denxity Nulls & Multiplicity | **ADOPT** | Adopt surrogate null pairings (`circular_shift_within_replica`, `permute_replicas`), Benjamini-Yekutieli correction, and degeneracy floors. |
| **Local References** | Alynxr `fold_stats_*` Store | **ADOPT** | Adopt 3-tier column namespace (`rep.*`, `der.*`, `gen.*`), `?` unverified column flag, and separation of raw array sinks from Parquet tables. |

---

## 2. Survey of Experiment & Sweep Dashboards

Modern ML tracking frameworks manage runs, hyperparameter sweeps, and artifact visualization. However, scientific design campaigns on high-performance computing (HPC) clusters introduce architectural constraints that eliminate typical SaaS or daemon-based tools:
- **Airgapped Compute**: Compute nodes on clusters like MIT Engaging or local HPC clusters have restricted or zero external Internet access. Tools requiring cloud connections or external CDNs fail catastrophically.
- **Shared Filesystem Concurrency**: Cluster jobs write simultaneously across NFS, Lustre, or GPFS mounts. File locking mechanisms designed for local SQLite or RocksDB fail, deadlock, or corrupt databases under multi-node NFS access.
- **Durable Scientific Record**: Scientific findings must be deterministic, content-addressed, and auditable. Ephemeral dashboards with dynamic, unversioned server state do not meet publication and review standards.

### 2.1 Weights & Biases (W&B)
- **Architecture**: Client-server architecture where a lightweight Python client (`wandb`) intercepts logging calls, buffers events to disk, and streams updates via HTTP/WebSocket to a centralized SaaS cloud or on-premise Docker server. Sweeps (`wandb.sweep`) operate via a central sweep server dispatching hyperparameter configurations to worker agents.
- **Exploratory Capabilities**:
  - *Parallel Coordinates Plot*: Excellent visualization of multi-parameter sweep grids, mapping continuous and categorical hyperparameter axes to vertical parallel lines connected by run polylines, colored by metric values.
  - *Grouping & Aggregation*: Enables grouping runs by arbitrary config keys (e.g. `model_family`, `learning_rate`) and dynamically computing mean, median, standard deviation, and bounding envelopes.
  - *Run Comparison*: Visual diffing of config dictionaries and overlay of performance curves.
- **Failure Modes & Hazards**:
  - *Network & Auth Ingress*: Requires external network connectivity and authentication tokens. Airgapped compute nodes refuse connections or hang on timeout.
  - *Unprincipled Aggregation*: Automatically averages metrics across runs without understanding experimental hierarchy (e.g., conflating independent biological builds with random seed replicates, or treating thousands of sequence samples as independent degrees of freedom).
  - *Silent Drop of Missing Grid Cells*: Parallel coordinates plots silently omit runs that failed, timed out, or crashed, presenting a misleadingly optimistic view of sweep success.
- **Verdicts**:
  - **ADAPT (UX Concepts)**: Adapt the parallel coordinates visual schema and multi-knob grouping semantics for multi-dimensional sweep visualization in Vega-Lite.
  - **REJECT (Runtime Engine)**: Reject the W&B backend, client daemon, SaaS dependencies, and naive aggregation algorithms.

### 2.2 MLflow
- **Architecture**: Open-source platform featuring an abstract tracking server backed by a relational database (SQLAlchemy: PostgreSQL, MySQL, SQLite) and an artifact store (S3, GCS, POSIX filesystem). Sweeps are represented as nested runs (`mlflow.start_run(nested=True)`), where a parent run encapsulates the sweep and child runs encapsulate individual trials.
- **Exploratory Capabilities**:
  - *Nested Hierarchy*: Explicit parent-child relationship between sweep manifests and individual unit executions.
  - *Param/Metric Separation*: Distinguishes immutable parameters (`log_param`) from dynamic scalar metrics (`log_metric`).
  - *Artifact Logging*: Binds arbitrary files (models, plots, tables) to run records.
- **Failure Modes & Hazards**:
  - *Filesystem Concurrency Deadlocks*: When run in serverless mode against a local directory or shared NFS (`file:///...`), MLflow uses SQLite. Under multi-node SLURM array execution, concurrent writes to SQLite over NFS trigger immediate `sqlite3.OperationalError: database is locked` or silent database corruption.
  - *Tracking Server Daemon Overhead*: Requiring users or cluster operators to spin up and maintain an `mlflow server` daemon is operationally brittle and unsuitable for self-contained, reproducible artifact directories.
  - *Flat Tabular Modeling*: Lacks native support for multi-dimensional tensor metrics, high-dimensional sequence distributions, or biological structure observables.
- **Verdicts**:
  - **ADAPT (Schema Semantics)**: Adapt the explicit parent-child (sweep manifest vs. unit execution) hierarchical data model and immutable parameter/metric separation.
  - **REJECT (Server Architecture)**: Reject the centralized tracking server, SQLite backend, and reliance on active background processes.

### 2.3 Aim
- **Architecture**: Open-source, Git-inspired ML experiment tracking system. Stores metric and parameter records in an embedded structure backed by RocksDB and Cython. Provides a custom query language (AimQL) for sub-second slicing and dicing.
- **Exploratory Capabilities**:
  - *Sub-second Multi-Run Querying*: AimQL (`run.hparams.temperature > 0.2 and run.metrics.loss[-1] < 1.5`) enables rapid exploration across tens of thousands of runs.
  - *Advanced Grouping & Faceting*: Allows dynamic grouping of charts by rows, columns, and colors based on complex hyperparameter combinations.
  - *Dense Metric Storage*: High-density storage optimized for high-frequency time-series logging.
- **Failure Modes & Hazards**:
  - *C++ / RocksDB Native Compilation*: Requires building and linking RocksDB C++ libraries. Causes installation failures and glibc compatibility issues across heterogeneous HPC cluster environments.
  - *Lock Contention on Distributed Filesystems*: RocksDB relies on single-process write locks (`LOCK` file). Multi-process SLURM tasks attempting to write to the same Aim repository fail immediately.
  - *Process Coupling*: Exploration requires launching a local web server (`aim up`). Does not natively generate static, standalone, airgapped report files (such as standalone SVGs or self-contained HTML).
- **Verdicts**:
  - **ADAPT (Query Ergonomics)**: Adapt the slice-and-dice query semantics and multi-dimensional faceting logic (re-implemented via DuckDB SQL).
  - **REJECT (Storage Engine)**: Reject the RocksDB storage engine, native compilation footprint, and daemon-based UI.

### 2.4 TensorBoard (HParams Plugin)
- **Architecture**: Filesystem-first log parser. Reads immutable, append-only event files (`tfevents`) containing Protocol Buffer records serialized to disk. The HParams plugin parses hyperparameter metadata and summary metrics directly from directory trees without a centralized database.
- **Exploratory Capabilities**:
  - *Parallel Coordinates & Scatter Plot Matrix*: Native HParams view displays parallel coordinates, scatter matrices, and sortable metric tables.
  - *Zero-Database Filesystem Consumption*: Operates entirely by scanning filesystem directories; no central tracking database to corrupt.
- **Failure Modes & Hazards**:
  - *Protobuf Overhead & Fragility*: Parsing massive protobuf event streams in Python is extremely slow and memory-intensive for large campaigns.
  - *Missing Grid Cells*: Missing or failed hyperparameter combinations are either completely hidden or cause visualization rendering errors.
  - *Absence of Statistical Inference*: Averages across runs without any replicate structure, honest confidence intervals, or power analysis.
- **Verdicts**:
  - **ADAPT (Filesystem-First Paradigm)**: Adapt the read-only, filesystem-first architecture where EDA parses existing committed directory artifacts without modifying execution state.
  - **REJECT (Event Protobufs & UI)**: Reject Protocol Buffer serialization, Java/C++ event parsers, and TensorBoard web interface.

---

## 3. Survey of Design of Experiments (DOE) & Factorial Analysis

Scientific evaluation of generative models and physical simulations requires formal Design of Experiments (DOE). In computational protein design and molecular dynamics, knobs (e.g. model checkpoint, sampling temperature, noise schedule, state-weight profiles, fixed residue pins) interact non-linearly. Evaluating knobs in isolation (one-factor-at-a-time, OFAT) completely obscures epistatic and synergic interactions.

### 3.1 Factorial Design, Main Effects, and Multi-Factor Interactions
- **Full Factorial Grid**: A planned sweep spanning the full Cartesian product of $D$ discrete knob factors:
  $$\mathcal{K} = K_1 \times K_2 \times \dots \times K_D$$
  where factor $d$ has $L_d$ levels, resulting in $N_{cells} = \prod_{d=1}^D L_d$ unique experimental conditions.
- **Main Effects**: The marginal effect of changing factor $K_i$ from level $a$ to level $b$, averaging over all other factors:
  $$\text{ME}(K_i, a \to b) = \frac{1}{|\mathcal{K}_{-i}|} \sum_{k \in \mathcal{K}_{-i}} \left( \mathbb{E}[Y \mid K_i=b, k] - \mathbb{E}[Y \mid K_i=a, k] \right)$$
- **Two-Factor Interactions**: Quantifies how the effect of factor $K_i$ depends on the level of factor $K_j$:
  $$\text{Int}(K_i, K_j) = \left( Y_{a_1, b_1} - Y_{a_1, b_0} \right) - \left( Y_{a_0, b_1} - Y_{a_0, b_0} \right)$$
  When $\text{Int}(K_i, K_j) \neq 0$, interaction plots show non-parallel response curves, indicating that optimal knob settings cannot be selected independently.
- **Hazard in Campaign Analysis**: High-dimensional grids suffer from missing cells due to execution timeouts, cluster preemptions, or out-of-memory failures. Conventional tools either silently omit missing cells (biasing main effects) or interpolate values (inventing data).
- **Verdict: ADOPT Full Factorial Modeling with Explicit Missingness Visualization**.
  - Enforce full Cartesian sweep planning via `CampaignManifest.rows`.
  - In EDA visualizations (e.g. `vegalite.sweep_grid`), missing or failed cells must be rendered explicitly as distinct visual marks (e.g. patterned grey cells with status banners), never dropped and never interpolated.

### 3.2 Response Surface Methodology (RSM) vs. Non-Parametric Ensembles
- **Response Surface Methodology**: Fits low-order polynomials (typically quadratic regression surfaces) over continuous design variables:
  $$Y = \beta_0 + \sum_{i} \beta_i X_i + \sum_{i} \beta_{ii} X_i^2 + \sum_{i < j} \beta_{ij} X_i X_j + \varepsilon$$
- **Limitations in Biomolecular Generation**:
  - Physical and generative landscapes (e.g. protein sequence likelihoods, free energies) are highly rugged and non-convex.
  - Quadratic surfaces smooth away sharp phase transitions (e.g., library collapse below a critical sampling temperature $T < 0.15$).
  - Parametric Gaussian error assumptions ($\varepsilon \sim \mathcal{N}(0, \sigma^2)$) fail drastically on bounded metrics (recovery $\in [0, 1]$, entropy $\ge 0$).
- **Verdict: ADAPT Interaction Curves / REJECT Parametric Quadratic Surfaces**.
  - Adopt empirical two-factor interaction plots (lines connecting discrete factor levels with empirical confidence intervals).
  - Reject parametric polynomial response surface regression in favor of empirical non-parametric pooled estimates.

### 3.3 Contrast Estimation & Honest Uncertainty Intervals

#### 3.3.1 The Pseudoreplication Trap in Generative and Simulation Regimes
In generative modeling (e.g., generating $N=10,000$ sequences per condition) and molecular dynamics simulations (e.g., collecting $N=500,000$ trajectory frames), classical hypothesis testing presents a severe methodological failure:
- **Spurious Significance**: As sample size $N \to \infty$, the standard error of the mean $\text{SE} = s / \sqrt{N} \to 0$. Any arbitrary, infinitesimal difference ($\Delta = 0.0001$) achieves $p < 10^{-15}$ under classical $t$-tests or ANOVAs, despite having zero scientific or biological relevance.
- **Violation of Exchangeability**: Sequences generated from a single stochastic run, or frames extracted from a single MD trajectory, are strongly autocorrelated and conditioned on identical PRNG seeds, starting structures, or model weights. Treating them as independent degrees of freedom ($N$) is blatant pseudoreplication.

#### 3.3.2 The Global MD-Replicas Hierarchy Rule
In accordance with established statistical rigor (`md-analysis:SKILL.md` and `nulls-and-uncertainty.md`):
- **Hierarchy of Levels**:
  $$\text{frames / sequences} \subset \text{seeds / replicas} \subset \text{build / starting structure} \subset \text{condition / knob setting}$$
- **Replicas are Samples, Not Experiments**: Seeds and replicas are draws from an unconverged estimator. Pooling (concatenation) across seeds within a condition is the point estimator.
- **Independent Unit of Contrast**: The independent experimental unit for a contrast between conditions is the highest unpooled level (typically the **seed** or **build**). Sample size $U$ is the number of independent units, never the number of generated sequences or trajectory frames.
- **Few-Sample Tests Prohibited**: Classical few-sample $t$-tests or per-site Welch tests treating seeds as $N$ ($N=2$ or $3$) are strictly prohibited.

#### 3.3.3 Vectorized Cluster Bootstrap
To estimate honest confidence intervals without relying on Gaussian normality assumptions or Python loops over units:
1. Let $U$ be the number of independent replication units in a condition cell, each producing an observable vector $Y_u \in \mathbb{R}^D$, forming matrix $Y \in \mathbb{R}^{U \times D}$.
2. Define a bootstrap iteration count $B$ (e.g., $B=2,000$).
3. Generate a cluster bootstrap resampling weight matrix $W \in \mathbb{N}^{B \times U}$, where each row is drawn independently from a multinomial distribution:
   $$W_{b, :} \sim \text{Multinomial}\left(U, \left[\frac{1}{U}, \frac{1}{U}, \dots, \frac{1}{U}\right]\right)$$
4. Compute all $B$ bootstrap replication estimates via a single vectorized matrix multiplication:
   $$\hat{\theta}^* = \frac{1}{U} \left( W \cdot Y \right) \in \mathbb{R}^{B \times D}$$
5. Extract empirical quantiles along axis 0 at $\alpha/2$ and $1 - \alpha/2$ (e.g. 2.5% and 97.5% for a 95% CI):
   $$\text{CI}_{1-\alpha} = \left[ q_{\alpha/2}(\hat{\theta}^*), \, q_{1-\alpha/2}(\hat{\theta}^*) \right]$$
This formulation executes in microseconds via NumPy/BLAS, completely eliminates Python iteration loops over draws or units, and honors the clustering structure.

#### 3.3.4 Minimum Detectable Effect (MDE) & Power Discipline
Before declaring "no effect" between two knob settings, the analysis must evaluate whether the campaign had sufficient statistical power to detect a meaningful difference.
- For a two-arm contrast with standard error $\text{SE}(\hat{\Delta})$ at significance level $\alpha = 0.05$ and power $1 - \beta = 0.80$:
  $$\text{MDE}_{0.80} = \left( z_{1 - \alpha/2} + z_{1 - \beta} \right) \cdot \text{SE}(\hat{\Delta}) = (1.960 + 0.842) \cdot \text{SE}(\hat{\Delta}) \approx 2.802 \cdot \text{SE}(\hat{\Delta})$$
- **Inference Rules**:
  1. If $|\hat{\Delta}| \ge \text{MDE}_{0.80}$ and $0 \notin \text{CI}_{95\%}$: The effect is **statistically detected and powered**.
  2. If $|\hat{\Delta}| < \text{MDE}_{0.80}$ and $0 \in \text{CI}_{95\%}$: The contrast is **underpowered**. The report MUST emit: `"Indeterminate: contrast underpowered (observed effect < MDE)"`, rather than claiming evidence of absence.
  3. Reporting "no effect" without an MDE check or positive control is strictly prohibited.

#### 3.3.5 Small-Sample Guard ($U < 3$)
When the number of independent units in a condition cell is small ($U < 3$):
- Resampling $U=2$ units yields only $\binom{2+2-1}{2} = 3$ unique non-trivial bootstrap weight allocations, producing a discrete, degenerate pseudo-distribution.
- **Mandatory Guard**: When $U < 3$, the statistical engine MUST flag `descriptive_only = True` and withhold confidence intervals, displaying only the empirical point estimate with a warning banner.

- **Verdict: ADOPT Vectorized Cluster Bootstrap, MDE Power Checks, and Small-Sample Guards; REJECT Classical Hypothesis Tests ($t$/ANOVA) over Raw Samples**.

---

## 4. Survey of Protein-Design & Sequence-Generation Library Analysis

Generative sequence models (e.g. ProteinMPNN, SolubleMPNN, ESM-IF, LigandMPNN) sample amino acid sequences conditioned on structural backbones and target contexts. Evaluating the resulting sequence libraries requires specific metrics designed to detect model pathologies, diversity collapse, and fitness-diversity trade-offs.

### 4.1 Sequence Diversity & Pairwise Identity Distributions
- **Definition**: Given $N$ generated sequences $\mathbf{s}_1, \dots, \mathbf{s}_N$ of length $L$, the pairwise identity between sequence $i$ and sequence $j$ over a designable residue mask $M \subseteq \{1, \dots, L\}$ is:
  $$\text{PID}(i, j) = \frac{1}{|M|} \sum_{l \in M} \mathbb{I}(s_{i, l} == s_{j, l})$$
- **Vectorized Computation**:
  - Convert token strings to integer array $S \in \{0, \dots, 20\}^{N \times L}$.
  - Compute pairwise identity matrix via broadcasting:
    $$\text{Match} = (S[:, \text{None}, :] == S[\text{None}, :, :]) \quad \in \mathbb{B}^{N \times N \times L}$$
  - Mask and average over residue axis:
    $$D = \frac{1}{|M|} \sum_{l \in M} \text{Match}[:, :, l] \quad \in [0, 1]^{N \times N}$$
  - Extract the strict upper triangle ($i < j$) to avoid diagonal identity ($1.0$) and symmetric duplication.
- **Diversity Summary**: Report mean pairwise identity, median, interquartile range (IQR), and full distribution histogram. Low diversity ($\text{PID} > 0.85$) signals sampling collapse.
- **Verdict: ADOPT Vectorized Pairwise Identity Distribution**.

### 4.2 Positional Shannon Entropy & Token Composition Entropy
- **Positional Shannon Entropy**: Measures sequence variability at each individual residue position $l \in M$:
  $$H(l) = -\sum_{c \in \mathcal{A}} p_l(c) \log_2 p_l(c) \quad \text{(bits)}$$
  where $\mathcal{A}$ is the 20-letter amino acid alphabet (max entropy $\log_2 20 \approx 4.32$ bits).
  - Can be computed either from empirical sample frequencies $p_l(c) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(s_{i, l} == c)$ or from model logits via softmax: $p_l(c) = \text{Softmax}(\mathbf{z}_{l})_c$.
  - Identifies functional constraints: structurally buried core residues exhibit $H(l) \approx 0-1$ bits, while surface-exposed loop positions exhibit $H(l) \approx 3-4$ bits.
- **Token Composition Entropy**: Measures the global amino acid frequency distribution across the entire generated library:
  $$H_{\text{comp}} = -\sum_{c \in \mathcal{A}} \bar{p}(c) \log_2 \bar{p}(c), \quad \text{where } \bar{p}(c) = \frac{1}{N \cdot |M|} \sum_{i=1}^N \sum_{l \in M} \mathbb{I}(s_{i, l} == c)$$
  - *Crucial Diagnostic*: In the TEV design AR campaign reconnaissance (`teamwork_explorer_1`), early LigandMPNN checkpoints exhibited severe library collapse ($H_{\text{comp}} \approx 3.3-3.4$ bits), disproportionately generating Leucine ($\sim 25\%$) and Alanine ($\sim 17\%$), whereas healthy models yield $H_{\text{comp}} > 4.0$ bits.
- **Verdict: ADOPT Positional and Composition Shannon Entropy**.

### 4.3 Real-Residue Masking & Padding Hazards
- **Hazard**: In batch inference, proteins of varying lengths are padded to a common tensor length $L_{\max}$ with pad tokens (e.g. index 20 or token `X`).
- **Observation**: Sampling over unmasked padding positions produces uniform pseudo-random token distributions with entropy $H \approx 4.04$ bits. If padding residues are included in library statistics, shorter proteins appear artificially hyper-diverse and exhibit spuriously diluted recovery.
- **Strict Masking Contract**:
  - Let $L_{\text{true}}$ be the true sequence length from the target PDB.
  - Let $\mathbf{m}_{\text{real}} \in \mathbb{B}^{L_{\max}}$ where $m_{\text{real}}[l] = \text{True}$ for $l < L_{\text{true}}$.
  - Let $\mathbf{m}_{\text{fixed}} \in \mathbb{B}^{L_{\max}}$ indicate fixed/pinned catalytic residues.
  - The designable mask MUST be evaluated as:
    $$\mathbf{m}_{\text{designable}} = \mathbf{m}_{\text{real}} \land \neg \mathbf{m}_{\text{fixed}}$$
  - All composition, diversity, recovery, and entropy metrics must evaluate exclusively over indices where $\mathbf{m}_{\text{designable}} == \text{True}$.
- **Verdict: ADOPT Strict Real-Residue Masking Contract**.

### 4.4 Target Recovery, Recovery Margin, and Alphabet Normalization
- **Native Recovery**: The fraction of designable residues matching the wild-type target sequence:
  $$\text{Rec}(\mathbf{s}) = \frac{1}{|M|} \sum_{l \in M} \mathbb{I}(s_l == s_l^{\text{WT}})$$
- **Recovery Margin vs. Shuffled Native**:
  - High native recovery alone can be a deceptive metric if the target protein has an abnormal composition (e.g., a collagen-like or coiled-coil sequence rich in Gly/Pro or Leu). A degenerate generator biased towards those tokens achieves high recovery purely by chance.
  - Define shuffled native recovery $\text{Rec}_{\text{shuf}}$ as recovery against a randomly permuted wild-type sequence with identical background amino acid composition.
  - Define Recovery Margin:
    $$\text{Margin} = \text{Rec}(\mathbf{s}) - \text{Rec}_{\text{shuf}}(\mathbf{s})$$
  - *Safety Threshold*: If $\text{Margin} < 0.10$, the generator is failing to capture sequence-structure specificity beyond background composition. The EDA engine must banner libraries failing this threshold.
- **Alphabet Normalization**:
  - Different structural biology frameworks order amino acid alphabets differently:
    - ProteinMPNN Order: `ACDEFGHIKLMNPQRSTVWYX`
    - AlphaFold / Proxide Order: `ARNDCQEGHILKMFPSTWYVX`
  - *Observed Bug*: In prototype campaigns, unmapped alphabet comparisons caused wild-type recovery to drop from $0.45$ to $0.14$ because token indices were directly compared without alphabet alignment.
  - *Requirement*: Sinks and domain plugins must declare their alphabet schema (`SinkSchema.alphabet`), and the EDA engine must normalize tokens to a single canonical alphabet before computing identity or recovery.
- **Verdict: ADOPT Native Recovery, Shuffled Native Margin, and Typed Alphabet Normalization**.

### 4.5 Score Distributions, Derived NLL, and Perplexity
- **Score Metrics**:
  - Model confidence is reported as Negative Log-Likelihood (NLL) per residue:
    $$\text{NLL}(\mathbf{s}) = -\frac{1}{|M|} \sum_{l \in M} \log p(s_l \mid \text{backbone}, \mathbf{s}_{<l})$$
  - Perplexity represents the effective number of choices per position:
    $$\text{Perplexity} = \exp\left( \min(\text{NLL}, 50.0) \right)$$
    (clamped at 50.0 to prevent floating-point overflow).
- **Derived Score Fallback**:
  - When consumer models output logits $\mathbf{Z} \in \mathbb{R}^{L \times |\mathcal{A}|}$ but omit precomputed score arrays, the domain plugin must derive NLL host-side:
    $$\text{NLL}_{\text{derived}} = \frac{1}{|M|} \sum_{l \in M} \left( \text{LogSumExp}(\mathbf{Z}_{l, :}) - \mathbf{Z}_{l, s_l} \right)$$
  - Derived scores must be explicitly flagged as `source = "derived"` (or tagged with a trailing `?`) to distinguish them from model-reported scores.
- **Verdict: ADOPT NLL and Perplexity Distributions with Derived Fallback**.

### 4.6 Order Statistics: Best-of-$N$ & Top-$k$ Enrichment Curves
- **Scientific Motivation**: In practical protein engineering, scientists do not test random unselected designs; they synthesize the top $k$ candidates from an in silico generated library of size $N$ (e.g. $k=1$ or $k=10$ from $N=1,000$).
- **The Metric**: How does expected top performance scale with candidate library size $n$?
  - For candidate sizes $n \in [1, N]$, compute the expected minimum score (or maximum recovery):
    $$\mathcal{E}(n) = \mathbb{E}\left[ \min_{i=1, \dots, n} \text{Score}_i \right]$$
  - Evaluated either analytically via order statistics or empirically by vectorized sub-sampling over $R=1,000$ draws without replacement.
  - Curves plot candidate pool size $n$ on the x-axis vs. top-performing score on the y-axis, demonstrating diminishing returns and comparing generator efficiency across temperatures and models.
- **Verdict: ADOPT Best-of-$N$ and Top-$k$ Enrichment Curves**.

### 4.7 Score-versus-Diversity Pareto Frontiers
- **The Generative Dilemma**: Generative sequence models exhibit a fundamental trade-off:
  - Low temperature ($T=0.1$): Produces excellent scores (low NLL, high confidence) but near-zero diversity (clones of the mode).
  - High temperature ($T=0.8$): Produces high diversity but poor scores (high NLL, non-viable sequences).
- **Pareto Frontier Formulation**:
  - For each knob condition, filter the library into subsets based on score thresholds $\tau$.
  - Evaluate subset pairwise diversity vs. mean score.
  - Identify non-dominated configurations (Pareto frontier): knob settings where no other configuration achieves both higher diversity and lower NLL.
- **Verdict: ADOPT Score-versus-Diversity Pareto Frontiers**.

### 4.8 Conformational Ensembles & Mutual Information (demistify / ciMIST)
In molecular dynamics ensembles (`demistify`), the library is not a sequence collection but a discrete conformational state trajectory $states\_traj \in \{0, \dots, K-1\}^{T \times L}$:
- **Observables**:
  - Marginal conformational entropy: $S_i = -\sum_{k} p_i(k) \ln p_i(k)$ (nats).
  - Total MIST conformational entropy: $S_{\text{total}} = \sum_i S_i - \sum_{(u,v) \in \mathcal{T}} I(u; v)$ (nats), where $\mathcal{T}$ is the maximum-weight spanning tree.
  - Pairwise Mutual Information matrix: $I(X_i; X_j)$.
  - Allosteric Hub Scores: $\sum_{j \neq i} I(X_i; X_j)$.
- **Schema Adaptation (Schema A vs. Schema B)**:
  - Schema A (loose files) includes `<sys>_mi_matrix.npy` (dense full matrix).
  - Schema B (Zarr) persists only MIST tree edges (`result.zarr/mist`); full matrix is omitted (`matrix_fullness: false`).
  - *Adaptation*: The demistify domain plugin must handle Schema B gracefully: when the dense matrix is absent, compute hub metrics from tree degree/weight, banner the degraded path, and avoid unhandled exceptions.
- **Verdict: ADOPT ciMIST Conformational Entropy & MIST Observables; ADAPT Hub Scoring for Schema B**.

---

## 5. Survey of Columnar Tools & High-Performance Vectorization

Campaign EDA requires aggregating, querying, and transforming metrics across hundreds of units and millions of generated sequences or frames. Traditional Python loops (`for row in manifest: for seq in sequences: ...`) are unacceptably slow, taking hours and crashing under memory limits. We evaluated columnar formats, zero-copy engines, and vectorization strategies.

### 5.1 Technology Comparison

| Feature / Criteria | DuckDB | Apache Arrow (PyArrow) | Polars | Apache Parquet | NumPy Vectorization |
|---|---|---|---|---|---|
| **Execution Model** | Vectorized Morsel-driven OLAP engine | In-memory columnar format & C++ kernels | Multithreaded Rust query engine | Columnar on-disk storage | C/BLAS array operations |
| **SQL Support** | Full ANSI SQL, window functions, PIVOT | Acero engine (limited) | SQL context (subset) | None (storage only) | None |
| **Zero-Copy Reads** | Direct read of Parquet & Arrow IPC | Memory-mapped IPC & Parquet buffers | Memory-mapped Arrow buffers | On-disk format | Native memory buffers |
| **Out-of-Core Processing** | Yes (streams larger-than-RAM data) | Partial (streaming RecordBatches) | Yes (streaming engine) | N/A | No (must fit in RAM) |
| **Dependency Footprint** | Single self-contained wheel | Pre-compiled C++ bindings | Large Rust binary wheel | File format standard | Universal standard (core) |
| **Packaging Boundary** | Optional extra (`xtrax[eda-campaign]`) | Optional extra (`xtrax[eda-campaign]`) | Optional extra (if adopted) | Serialized on disk | Core `xtrax` (zero extra) |

### 5.2 Deep-Dive Evaluation & Technical Rationale

#### 5.2.1 DuckDB
- **Evaluation**: DuckDB is an embeddable, in-process analytical SQL engine designed specifically for fast OLAP queries. It operates directly on Parquet files, Arrow tables, and Pandas/Polars DataFrames without copying data into internal database structures.
- **Strengths**:
  - Can execute complex multi-knob aggregations (`GROUP BY arm, checkpoint, temperature`) across thousands of partitioned Parquet files with a single SQL query: `SELECT * FROM read_parquet('campaign/predictions/**/*.parquet')`.
  - Supports advanced analytical operations: window functions (`DENSE_RANK() OVER (...)`), list aggregations, and dynamic PIVOT tables.
  - Streaming execution allows querying 50GB Parquet stores on a laptop with 8GB RAM without out-of-memory errors.
  - Self-contained binary wheel without external runtime or daemon dependencies.
- **Verdict: ADOPT DuckDB as the Primary Campaign SQL Query Engine** (behind `xtrax[eda-campaign]`).

#### 5.2.2 Apache Arrow & PyArrow
- **Evaluation**: Apache Arrow provides the universal in-memory columnar data standard. PyArrow provides fast C++ readers for Parquet files and memory-mapped IPC buffers.
- **Strengths**:
  - Serves as the zero-copy lingua franca between DuckDB, NumPy, and storage sinks.
  - Typed schema enforcement guarantees that missing values, string encodings, and nested array types are strictly validated.
  - Efficient slice-and-gather operations without Python object allocation.
- **Verdict: ADOPT Apache Arrow (PyArrow) as In-Memory Schema and Parquet I/O Layer** (behind `xtrax[eda-campaign]`).

#### 5.2.3 Polars
- **Evaluation**: Polars is an exceptionally fast DataFrame library built in Rust on top of Arrow.
- **Trade-off Analysis**:
  - While Polars has an expressive expression API, DuckDB already provides complete SQL querying, out-of-core streaming, and zero-copy Arrow integration.
  - Requiring both DuckDB and Polars introduces redundant dependencies and increases binary wheel footprint on cluster nodes.
  - Core `xtrax` philosophy favors minimal, orthogonal dependencies. DuckDB handles SQL and multi-table joins; NumPy handles array math; PyArrow handles storage schemas. Polars does not provide a capability that DuckDB and NumPy cannot achieve together.
- **Verdict: REJECT Polars as a Direct Dependency** (to avoid architectural redundancy and dependency bloat).

#### 5.2.4 Apache Parquet
- **Evaluation**: Industry-standard columnar storage format featuring dictionary encoding, bit-packing, run-length encoding (RLE), and Snappy/Zstandard compression.
- **Strengths**:
  - Columnar layout allows reading only requested columns (projection pushdown), dramatically reducing I/O when computing single-metric statistics across large campaigns.
  - Partitioned directory layout (`predictions/arm=proteinmpnn/part-0.parquet`) enables partition pruning (predicate pushdown), skipping irrelevant files during queries.
- **Verdict: ADOPT Apache Parquet as the Standard Tabular Output Format**.

#### 5.2.5 NumPy Host Vectorization
- **Evaluation**: Native array operations executed via BLAS/LAPACK.
- **Role in Core**: Core `xtrax` must remain functional with `stdlib + numpy` without requiring `[eda-campaign]` extras.
- **Operations**:
  - Pairwise identity broadcasting: `(S[:, None] == S[None, :])`.
  - Vectorized cluster bootstrap: $\hat{\theta}^* = \frac{1}{U}(W \cdot Y)$.
  - Analytical and empirical order statistics.
- **Verdict: ADOPT NumPy Host Vectorization for Core Statistical Computation**.

---

## 6. Survey of Grammar of Graphics & Visualization

Visualization in `xtrax` serves two distinct audiences: interactive human exploratory analysis in browsers and immutable, provenanced figure review for publication and decision-making.

### 6.1 Grammar of Graphics Engines

#### 6.1.1 Vega-Lite
- **Architecture**: High-level declarative grammar of graphics expressing visual encodings, scales, transforms, and interactive selections as pure JSON specifications.
- **Advantages**:
  - *Pure Declarative JSON*: The specification is data. It can be constructed, serialized, hashed, validated against JSON Schema, and transmitted without executing JavaScript or Python code.
  - *Rich Client-Side Interactivity*: Native support for interactive selections (`point`, `interval`), pan/zoom, brush-linking, and cross-filtering between coordinated panels.
  - *Composable Layout*: Composable concatenation operators (`vconcat`, `hconcat`, `facet`, `repeat`) enable multi-panel dashboard layouts.
- **Verdict: ADOPT Vega-Lite as the Core Visualization Grammar**.

#### 6.1.2 Altair
- **Architecture**: Python library providing a typed, Pythonic API that emits standard Vega-Lite JSON specifications.
- **Role**: Altair simplifies authoring complex Vega-Lite specifications in Python with IDE autocomplete and type validation.
- **Constraint**: Core `xtrax` should not require Altair at runtime to render dashboards. Core rendering can emit raw declarative JSON dicts directly.
- **Verdict: ADAPT Altair as an Optional Spec Generator** (available under extras, while core engine emits standard JSON dictionaries).

#### 6.1.3 Observable Plot
- **Architecture**: JavaScript-first grammar of graphics created by Mike Bostock (D3 creator) for rapid data visualization.
- **Limitations**:
  - Imperative JavaScript API that cannot be expressed as a standalone declarative JSON schema.
  - Requires a Node.js or browser runtime to construct and evaluate plots.
  - Cannot be validated or manipulated server-side in pure Python without headless browser orchestration.
- **Verdict: REJECT Observable Plot**.

### 6.2 Dual-Layer Architecture: Static SVG vs. Interactive HTML

#### 6.2.1 The Airgapped Cluster Hazard
High-performance computing clusters (e.g. MIT Engaging) run compute jobs on nodes isolated from external networks. If an EDA dashboard produces an HTML file that relies on external CDN script tags:
```html
<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
```
Opening this file on an airgapped workstation or viewing it without internet connectivity causes script loading to hang and time out, resulting in a blank white page.

#### 6.2.2 The Reviewed Object Protocol
In rigorous scientific pipelines (such as Maraxiom and Bathos):
- **The Static SVG is the Reviewed Object**: The exact pixels and vector paths of an SVG rendered via `vl-convert` are hashed (`SHA-256`). When a reviewer critiques, approves, or promotes a finding, they review the static SVG.
- **Interactive View as Convenience**: The interactive HTML / Vega-Embed view is an unhashed convenience layer for exploratory filtering. Critiques and promotions remain anchored to the static SVG asset.

#### 6.2.3 Progressive HTML Rendering with Timeout Guard
To reconcile interactivity with airgapped reliability, the EDA HTML view adopts a progressive rendering pattern:
1. Pre-render the full static vector SVG and embed it directly inline within the HTML document inside a fallback container:
   ```html
   <div id="fallback-svg">
     <svg ...><!-- Static pre-rendered vector SVG --></svg>
   </div>
   <div id="interactive-vis" style="display:none;"></div>
   ```
2. Include a 2-second timeout guard script: If external CDN scripts load successfully, the container toggles to the interactive Vega-Embed view. If CDN scripts fail or time out (airgapped mode), the interactive container is suppressed, and the static embedded SVG remains visible and crisp.

- **Verdict: ADOPT Dual-Layer Architecture (Static Hashed SVG + Progressive HTML with Timeout Guard)**.

---

## 7. Survey of Local Reference Implementations across Repositories

Within the local engineering ecosystem, several repositories have pioneered production patterns for experiment tracking, campaign stores, uncertainty modeling, and review pages. We survey their implementations to align `xtrax` with these established standards.

### 7.1 Maraxiom Review Pages (`maraxiom`)
- **Location**: `.praxia/docs/specs/261015_visual-first-experiment-review.md`, `maraxiom-review-pages:SKILL.md`.
- **Core Patterns**:
  - *Neutral Hashed Artifacts*: Every review page build generates neutral, content-addressed files: `<panel>.json` (canonical data), `<panel>.parquet` (tidy table), `<panel>.vl.json` (Vega-Lite spec), and `<panel>.svg` (rendered vector SVG), registered in a SHA-256 manifest.
  - *Standard Component Taxonomy*:
    - `vegalite.grouped_distribution`: Value distributions by group.
    - `vegalite.pooled_estimate`: Pooled estimates across conditions with block-bootstrap intervals and individual seed points.
    - `vegalite.sweep_grid`: Full parameter grid showing completion state; missing cells drawn explicitly, never interpolated.
    - `vegalite.residue_track`: Per-residue quantities along a protein chain.
  - *Inline EDA Manifest*: Declares `unit_columns`, `condition_column`, `value_column`, and `expected_units` (from the design, never from data).
  - *Bannering Policy*: Failing EDA checks render as visual warning banners across the panel; they never suppress the panel.
- **Architectural Boundary (F-10 / Decision 3)**:
  - Strict namespace separation: Zero imports of `maraxiom` inside `xtrax.eda`.
  - Maraxiom consumes `xtrax` EDA outputs exclusively via neutral file artifacts and external adapters.
- **Verdict: ADOPT Component Taxonomy, Neutral Artifact Manifest, and Bannering Policy; REJECT Internal Maraxiom Imports (Preserve F-10 Boundary)**.

### 7.2 Bathos Campaigns (`bathos`)
- **Location**: `bathos-campaigns:SKILL.md`, `using-bathos:SKILL.md`.
- **Core Patterns**:
  - *Pre-registration Discipline*: Experiments and confirmatory campaigns pre-register falsifiable claims and hypotheses in `.bth.toml` sidecars under `[outcomes]`.
  - *Synthetic Controls Discipline*: Every analytical instrument must be verified against synthetic controls:
    1. *Synthetic Positive Control*: Injects a known planted knob effect ($\Delta^*$) and verifies that the statistical detector captures it ($0 \notin \text{CI}_{95\%}$).
    2. *Synthetic Negative Control*: Evaluates a null campaign with identical distributions and verifies that false-positive detections do not exceed the nominal $\alpha$ rate ($0 \in \text{CI}_{95\%}$).
  - *Campaign Conclude & Union Gate*: Falsifiable claims are evaluated when a campaign concludes, verifying that all pre-registered outcomes hold simultaneously without cherry-picking.
- **Architectural Boundary**:
  - Zero imports of `bathos` inside `xtrax`. Bathos wraps execution scripts externally and tracks output directories via `bth run` and `figure_manifest.json`.
- **Verdict: ADOPT Pre-Registration and Synthetic Controls Discipline; REJECT Internal Bathos Imports (Preserve Decision 3)**.

### 7.3 Denxity (`denxity`)
- **Location**: `nulls-and-uncertainty.md`, `interpretation.md`, `xtrax.op` benchmarks.
- **Core Patterns**:
  - *Surrogate Null Distributions*: For mutual information and entropy statistics, arm choice dictates null pairing:
    - Conditional Arm ($I(X; Y \mid R)$): Paired with `circular_shift_within_replica` (preserves marginals and autocorrelation; destroys within-replica timing).
    - Pooled Arm ($I(X; Y)$): Paired with `permute_replicas` (preserves within-block structure; destroys replica identity).
  - *Multiplicity Correction*: MI tests across residue pairs exhibit strong spatial dependence. Standard Benjamini-Hochberg (BH) is invalid due to dependency violations; the framework enforces **Benjamini-Yekutieli (BY)** false discovery rate control.
  - *Degeneracy Floors*: Null distributions with fewer than 10 unique surrogate values are flagged `unavailable` rather than computing degenerate p-values.
- **Verdict: ADOPT Surrogate Null Pairings, Benjamini-Yekutieli Multiplicity Control, and Degeneracy Floor Rules**.

### 7.4 Alynxr `fold_stats_*` Parquet Store (`alynxr`)
- **Location**: `alynxr-fold-campaigns:SKILL.md`, `.praxia/docs/specs/261015_fold-campaigns-and-stats.md`.
- **Core Patterns**:
  - *Three-Tier Column Namespace*: Solves metric collision by qualifying every metric by entity and source:
    - `rep.<metric>`: Predictor-reported metric copied verbatim from output (e.g. `rep.ptm`, `rep.iptm`).
    - `der.<fn>@<ver>[params]`: Metric derived by a registered function at a pinned version (e.g. `der.ipsae_d0res@1[c=10]`).
    - `gen.<tool>.<metric>`: Generator-reported metric joined by `design_id`.
  - *Unverified Metric Flag (`?`)*: Derived metrics whose numerical parity against reference implementations has not been formally verified carry a trailing `?` in their column name (e.g. `der.plddt_mean?`), surviving export to CSV/DuckDB.
  - *Array Sink vs. Summary Table Separation*: High-dimensional structural arrays (Zarr stores, PDB coordinates, per-residue tensors) stay in dedicated unit array stores; scalar summaries and provenance metadata are indexed into partitioned Parquet tables.
  - *Content-Addressed Unit Identity*: Units are identified by a deterministic hash (`prediction_id` or `manifest_row_hash`) computed from canonical JSON payloads, never from filesystem paths.
- **Verdict: ADOPT Three-Tier Column Namespace (`rep.*`, `der.*`, `gen.*`), `?` Unverified Flag Convention, and Array-vs-Parquet Separation**.

---

## 8. Synthesis & Architectural Recommendations for Phase P2 Spec

Based on this comprehensive survey, we synthesize the foundational architecture recommendations for the upcoming Phase P2 Specification (`261106_campaign-eda-redesign.md`):

### 8.1 Architectural Principles
1. **Committed-Only Unit Index Reads**: The Campaign EDA engine must read exclusively from committed sinks via the unit index (`src/xtrax/run/unit_index.py`), keyed by `manifest_row_hash`. Uncommitted, partial, or quarantined units must be excluded from analytical pooling and bannered in coverage views.
2. **Strict Dependency & Packaging Discipline**:
   - `xtrax` core remains lightweight, depending strictly on `stdlib + numpy`.
   - All heavy columnar and visualization dependencies (`duckdb`, `pyarrow`, `altair`, `vl-convert`) reside strictly behind optional extras: `xtrax[eda-campaign]`.
   - Zero framework bridges: No internal imports of `maraxiom`, `bathos`, or `cisternal` inside `src/xtrax/eda`.
3. **Pluggable Domain Architecture**:
   - Deliver a typed entry-point interface (`xtrax.eda.domains`) allowing consumers (`aminx`, `demistify`) to define domain-specific library extractors, score extractors, and residue tables.
   - Domain plugins execute under deep-copy isolation, with error quarantine replacing failed plugin outputs with standardized diagnostic banners.
4. **Honest Statistical Inference Layer**:
   - Enforce the global hierarchical replica rule: seeds/replicas are samples of an unconverged estimator, never independent $N$.
   - Compute uncertainty intervals using vectorized cluster bootstrap ($W \cdot Y / U$) over independent units.
   - Compute Minimum Detectable Effect ($\text{MDE}_{0.80} \approx 2.802 \cdot \text{SE}$) before reporting null effects.
   - Guard against small sample degeneracies: flag `descriptive_only = True` when $U < 3$.
5. **Dual-Layer Visual Delivery**:
   - Export neutral hashed artifacts: canonical JSON, tidy Parquet, Vega-Lite JSON specifications, static SVG, and a SHA-256 manifest.
   - Embed pre-rendered vector SVG into progressive HTML with a 2-second CDN timeout guard to ensure flawless offline airgapped operation on cluster nodes.
6. **Transparent Secondary Run Telemetry**:
   - Isolate execution and computational telemetry (wall time, resume decisions, tiling memory) into a distinct secondary view.
   - Explicitly banner uninstrumented seams: display `"not instrumented (debt #3300)"` for pack events telemetry.

---

## 9. Verification & Evidence Citations

All findings, observations, and decisions in this research note are grounded in direct code inspection and established project skills:
1. `ORIGINAL_REQUEST.md`: Task definition and requirements R1–R4.
2. `P0_RECON_SYNTHESIS.md`: Master Question $\to$ Data Source $\to$ Field mapping table.
3. `feat/261105-campaign-runner-track5`:
   - `src/xtrax/campaign/manifest.py:232-482`: Manifest planning and `manifest_row_hash` computation.
   - `src/xtrax/campaign/marker.py:163-363`: Done marker verification and defensive quarantine.
   - `src/xtrax/pack/_node.py:328-672`: Packed vs loose bit-identical content digest and frame-level seek decompression.
   - `src/xtrax/run/unit_index.py:741-755`: Lock-free unit index lookup.
4. `aminx`:
   - `scripts/analysis/campaign_library_eda_prototype_v2.py:140-348`: Prototype TEV design AR campaign findings, alphabet alignment, and padding entropy traps.
5. `demistify`:
   - `demistify:analyzing-cimist-results/references/nulls-and-uncertainty.md:1-108`: Surrogate null pairings, multiplicity, and degeneracy floors.
   - `demistify:analyzing-cimist-results/references/artifact-schema.md:1-81`: Schema A, B, and C artifact specifications.
6. `naurmalade`:
   - `naurmalade:md-analysis/SKILL.md:1-29`: Global MD-replicas hierarchy rule and MDE controls.
7. `maraxiom`:
   - `maraxiom-review-pages/SKILL.md:1-100`: Static SVG reviewed object protocol and component taxonomy.
8. `alynxr`:
   - `alynxr-fold-campaigns/SKILL.md:341-500`: `fold_stats_*` Parquet store architecture, 3-tier namespaces, and `?` unverified column convention.
