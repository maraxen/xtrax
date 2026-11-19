---
title: "xtrax EDA Redesign: Vega-Lite Canonical Interchange, Pluggable Visualization Engine, and Deterministic Plan Inspection Artifacts"
description: "Specification for redesigning xtrax.eda into a compiler-plan inspection engine using Vega-Lite canonical interchange, entry-point panel plugins, bitwise-deterministic JSON exports, progressive offline HTML dashboards, and strict isolation from maraxiom.eda"
task_id: 261028_xtrax-eda-redesign
status: "FINAL r3.0 — converged architecture with explicit decision records and export contracts"
revision: "3.0 (2026-10-28; complete architectural specification, decisions 1-3, three-tier decoupling, backend registry, neutral hashed artifacts, dedup fix, and backward compatibility)"
---

# xtrax EDA Redesign Specification — Revision 3.0

## Revision History

| Rev | Date | Author / Agent | Changes & Adjudication Status |
|---|---|---|---|
| **r1.0** | 2026-10-28 | Architect (Draft) | Initial redesign proposal: transitioning legacy Matplotlib/Seaborn EDA to Vega-Lite JSON specification interchange, `entry_points` panel architecture, and canonical SHA-256 artifact hashing. |
| **r2.0** | 2026-10-28 | Adversarial Critic | Adversarial Review challenge. Identified 2 Blockers (**F-01**, **F-02**), 6 Majors (**F-03** through **F-08**), 1 Minor (**F-09**), and 1 Architectural Note (**F-10**). |
| **r2.1** | 2026-10-28 | Architect (Remediator) | **REMEDIATED SPECIFICATION**. Complete remediation of all adversarial findings:<br>• **F-01 & F-02 [BLOCKER]**: Explicitly added `minItems: 0` to `PlanStatsDict["axes"]` JSON Schema; specified zero-axis Vega-Lite text-mark fallback chart (never empty `vconcat: []`); reconciled AC-09 and AC-13.<br>• **F-03 [MAJOR]**: Strict decoupling of stats extraction (100% axis capture) from rendering truncation (`max_axes` in `render`); specified "+ N remaining axes" overflow summary and `axes.csv` streaming tabular link.<br>• **F-04 [MAJOR]**: RFC 8259 sanitization maps `NaN` and `+/-Inf` to `None` (`null`), never `0.0`, preventing false zero-memory OOM masks; unbounded memory appends explicit warning strings to `memory_warnings`.<br>• **F-05 [MAJOR]**: Bitwise cross-platform determinism in `export_canonical_json` (`-0.0` to `0.0`, fixed 8-decimal float precision, `sort_keys=True`, `separators=(',', ':')`, UTF-8 NFC normalization).<br>• **F-06 [MAJOR]**: Progressive offline HTML dashboard execution with embedded static SVG snapshot in `<div id="fallback-svg">`, 2-second timeout guard on Vega-Embed CDN loading, and optional `offline=True`/`bundle_js=True` asset bundling flags.<br>• **F-07 [MAJOR]**: Plugin isolation using `copy.deepcopy(stats)` immutability protection; pre-concatenation schema validation of plugin Vega-Lite snippets with graceful error placeholder panels.<br>• **F-08 [MAJOR]**: Concrete testability seams: `xtrax.eda.plugins.get_quarantined_plugins() -> list[str]` in AC-15; explicit structural DOM contract in AC-16.<br>• **F-09 [MINOR]**: In-memory caching of `entry_points` discovery with `xtrax.eda.plugins.reload_plugins()` invalidation hook.<br>• **F-10 [NOTE]**: Re-affirmed strict namespace boundary: `xtrax.eda` (compiler/plan inspection) has zero imports or dependencies to/from `maraxiom.eda`. |
| **r3.0** | 2026-10-28 | Architect (Converged) | **ARCHITECTURAL COMPLETION & EXPANSION**:<br>• **Architectural Decision 1**: Prioritize plan-inspection panels first, deferring cross-run campaign aggregation until concrete downstream workload requirements emerge.<br>• **Architectural Decision 2**: Reject affigit UI component dependency; standardize on standalone progressive HTML with embedded SVG fallback and Vega-Embed offline bundling.<br>• **Architectural Decision 3**: Mandate external placement of framework interop bridges (maraxiom, bathos, cisternal) in dependent glue packages, preserving core xtrax minimalism (stdlib + numpy).<br>• **Three-Tier Decoupling**: Codify separation of plan analysis (`BatchPlanLike` -> `PlanStatsDict`), declarative panel specs (`xtrax.eda.spec`), and backend adapters (`xtrax.eda.backends`).<br>• **Dual Entry-Point Registries**: Define `xtrax.eda.panels` and `xtrax.eda.backends`, introducing `vl-convert-python` via `xtrax[eda-vega]` alongside legacy `matplotlib` (`xtrax[eda-matplotlib]`/`xtrax[eda]`).<br>• **Neutral Hashed Export Artifacts**: Specify emit bundle with tidy CSV, Parquet with embedded schema metadata, canonical sorted JSON, and neutral `manifest.json` featuring SHA-256 digests and opaque context string passthrough.<br>• **Defect & Semantic Fixes**: Dedicated sections for the dedup panel y-label correction (`(1 - unique/total)` vs inverted label), `memory_estimate_bytes` / `memory_warnings` semantics, and backwards compatibility guarantees. |

---

## 1. Executive Summary & Problem Context

### 1.1 Context
In the `xtrax` compilation pipeline, `BatchPlanner.plan()` transforms user-specified axis configurations (`AxisSpec`) into execution strategies (`AxisDecision`: `Vmap`, `ChunkedMap`, `Scan`, `DedupGather`, `Bucket`). Understanding, debugging, and auditing these compilation decisions before JIT compilation or distributed execution is the responsibility of `xtrax.eda`.

The legacy EDA subsystem (`src/xtrax/eda/viz.py`), designed around Matplotlib and Seaborn, carries critical architectural limitations:
1. **Monolithic Heavyweight Dependencies**: Visual inspection requires importing `matplotlib` and `seaborn` into the process, introducing heavy C-extensions, font cache generation, and backend initialization overhead even when users only desire lightweight JSON interchange or programmatic stats inspection.
2. **Static, Non-Interactive Artifacts**: Seaborn produces static raster (PNG) or non-interactive SVG files. For complex plans with dozens of axes, engineers cannot inspect tooltips, examine per-axis reasoning strings, or interactively filter strategy distributions.
3. **Closed Visualization Pipeline**: Third-party extensions and hardware plugins (e.g., custom GPU/TPU tiling heuristics or Pallas kernel planners) cannot register custom visualization panels without monkey-patching `viz.py`.
4. **Lack of Deterministic Verification Artifacts**: Compiler verification in CI/CD requires bitwise-reproducible serialization and cryptographically verifiable plan summaries to detect compiler regression across platforms.

### 1.2 The Redesign Vision (Revision 2.1)
This specification redesigns `xtrax.eda` into a **decoupled, declarative, Vega-Lite canonical interchange architecture**:
- **Layer 0 (Core Types & Stats)**: `xtrax.eda.types` and `xtrax.eda.stats` remain 100% dependency-free (stdlib + numpy only), operating as the stable analytical engine.
- **Layer 1 (Declarative Vega-Lite Specification Engine)**: `xtrax.eda.spec` compiles `PlanStatsDict` into valid, portable Vega-Lite v5 JSON specifications without requiring any graphics libraries.
- **Layer 2 (Pluggable Panel & Backend Registry)**: Standard Python `entry_points` (`xtrax.eda.panels` and `xtrax.eda.backends`) enable modular panel registration with deep-copy immutability quarantine.
- **Layer 3 (Progressive Airgapped Rendering)**: `xtrax.eda.viz.render()` produces standalone HTML dashboards with progressive fallback: an embedded self-contained static SVG guarantees zero blank screens in airgapped clusters, while Vega-Embed progressively enhances the dashboard into an interactive explorer.
- **Layer 4 (Deterministic Hashed Exports)**: Canonical JSON serialization (`export_canonical_json`) with IEEE-754 float normalization, `-0.0` normalization, UTF-8 NFC encoding, and SHA-256 sidecars provides bitwise-reproducible plan audit records.

---

## 2. Core Architectural Decisions & Recommendations

### 2.1 Decision 1: Prioritize Plan-Inspection Panels First (Defer Campaign-Level Tables)
- **Rationale**:
  1. *Alignment with Compiler Scope*: `xtrax` is primarily a compiler and execution planning engine for composable JAX operations (`BatchPlanner.plan()` -> `BatchPlan`). Its primary operational responsibility is inspecting, debugging, validating, and auditing individual compilation decisions, per-axis execution strategies (`Vmap`, `ChunkedMap`, `Scan`, `DedupGather`, `Bucket`), cardinality scaling, and per-axis memory footprints. A developer optimizing a batching schedule or debugging an unexpected fallback to `Scan` needs immediate, high-fidelity insight into *that specific compilation plan*.
  2. *Risk of Premature Abstraction*: Cross-run campaign aggregation (tracking runs over time, parameter sweeps across architectures, regression tracking across git revisions or model versions) is inherently dependent on downstream workflow orchestration and experiment tracking systems (such as `bathos`). Without concrete, active downstream campaign workloads actively consuming such tables, defining a campaign-level schema in `xtrax.eda` risks baking in premature, rigid abstractions (such as assumptions about run identity, metric namespaces, or database persistence) that do not match downstream needs.
  3. *Clean Layering & Decoupling*: Single-plan inspection is self-contained: `BatchPlan` has a well-defined lifecycle and immediate local semantics. Campaign aggregation requires persistent storage, historical indexing, and metric aggregation that belong either downstream (in tools like `bathos` or tracking scripts) or should only be considered when real downstream workloads articulate concrete multi-plan query and schema requirements.
- **Recommendation**: Prioritize comprehensive single-plan inspection panels first (`strategy`, `cardinality`, `dedup`, `bucket`, `memory`, `reasoning`). Defer cross-plan/campaign-level aggregation tables until dependent systems articulate concrete data schema requirements.

### 2.2 Decision 2: Do Not Request an Affigit Component Initially
- **Rationale**:
  1. *Accurate Component Characterization*: `affigit` generates committed review scripts for review recipes (e.g. `vegalite.grouped_distribution`) that consume CSV/Parquet from pinned runs for human and peer review workflows. In contrast, compiler plan inspection statistics in `xtrax` are deterministic, diagnostic, and have no experimental variance across identical compilation runs; they rarely if ever require a dedicated review page.
  2. *Neutral Exports Suffice for Future Review*: `xtrax.eda.export` already emits tidy tables (CSV/Parquet) and canonical Vega-Lite specifications with cryptographic SHA-256 manifests. If an `affigit` review component or recipe is needed downstream by experiment tracking systems or review pipelines, it can be requested downstream later in consumer glue without modifying `xtrax`.
  3. *Coupling and Maintenance Overhead*: Requesting or depending on an `affigit` web/UI component repository at this stage introduces external frontend package dependencies, build-chain friction (node/npm/bundling), cross-repository release coupling, and synchronization burdens.
  4. *Airgapped HPC Reality & Sufficiency of Standalone HTML*: Compiler engineers and distributed ML practitioners run `xtrax` on remote compute clusters, Slurm nodes, and airgapped environments where external web assets and complex frontend devservers are unavailable or disallowed. A single standalone, self-contained HTML file with:
     - An inline embedded static vector SVG (`<div id="fallback-svg">`) that displays immediately with zero JavaScript and zero network calls,
     - Vega-Embed loaded via CDN with a 2-second timeout guard, or offline bundling (`offline=True` / `bundle_js=True`) with local JS assets,
     completely solves all interactive inspection requirements (tooltips, cross-filtering, zooming) without any frontend component repository dependency.
- **Recommendation**: Explicitly reject requesting an `affigit` component at this stage. Standardize on standalone progressive HTML with embedded SVG fallback and optional Vega-Embed offline bundling; let downstream consumers request affigit review components later if review pages are required.

### 2.3 Decision 3: Maintain Interop Bridges Outside xtrax in Dependent Glue
- **Rationale**:
  1. *Minimal Dependency Footprint*: `xtrax` is a foundational numerical compiler library. `xtrax.eda` core stats extraction is strictly stdlib + numpy (`BatchPlanLike` -> `PlanStatsDict`). Adding native imports or adapters for `maraxiom`, `bathos`, or `cisternal` inside `xtrax.eda` introduces massive dependency creep. For example, `bathos` has its own alpha release lifecycle; `cisternal` has had dependencies like `fastmcp` pre-releases; `maraxiom` has presentation and slide-deck dependencies.
  2. *Direction of Dependency*: In a layered architecture, dependencies must flow toward the foundation, not from foundation to application. `xtrax` is the foundational compiler. Downstream tools (`maraxiom`, `bathos`, `cisternal`, training harnesses) should depend on `xtrax` (or ingest its neutral export artifacts), NOT vice versa.
  3. *Neutral Artifacts as the Universal Seam*: Rather than tight in-process Python coupling, `xtrax.eda` provides neutral, byte-deterministic export artifacts (canonical RFC 8259 JSON, tidy CSV/Parquet, neutral emit manifests with SHA-256 digests and opaque context passthrough). Downstream glue packages or pipeline adapters can ingest these neutral artifacts trivially without any changes to `xtrax`.
- **Recommendation**: Strict ban on external framework bridges inside `src/xtrax/eda`. Any integration bridges to `maraxiom`, `bathos`, `cisternal`, or other tools must live in external glue packages, consumer repositories, or user-level pipeline scripts.

---

## 3. Adversarial Remediation Matrix (Revisions 2.0 → 2.1 → 3.0)

The following table documents the disposition of all findings raised by the Adversarial Review:

| Issue ID | Category | Severity | Summary of Adversarial Finding | Remediated Architectural Contract in Rev 2.1 |
|---|---|---|---|---|
| **F-01** | Schema | **BLOCKER** | `PlanStatsDict["axes"]` schema required `minItems: 1` or was underspecified, causing empty compilation plans (`BatchPlan(decisions=())`) to fail schema validation. | Schema definition explicitly declares `"minItems": 0` for `axes`, `dedup_stats`, `bucket_stats`, and `memory_warnings`. Empty plans are 100% valid. (§3.1) |
| **F-02** | Vega Spec | **BLOCKER** | Empty plans caused `generate_vega_spec` to emit an invalid empty Vega-Lite specification `{"vconcat": []}`, causing client rendering crashes. Contradiction between AC-09 and AC-13. | When `total_axes == 0`, `generate_vega_spec` outputs a single valid Vega-Lite root chart with a text mark (`"No active execution axes in compilation plan"`) and `data: {"values": [{}]}`. Empty `vconcat` is banned. AC-09 and AC-13 fully reconciled. (§3.2, §6) |
| **F-03** | Scalability | **MAJOR** | Truncating stats in `extract_plan_stats()` to `max_axes` drops telemetry and corrupts compiler auditing on large models. | Strict separation: `extract_plan_stats()` NEVER truncates (100% captured in `PlanStatsDict`). `max_axes` is strictly a visualization parameter in `render(plan, max_axes=100, overflow="aggregate")`. Overflow groups axes into "+ N remaining axes" badge and points to streaming `axes.csv`. (§3.3) |
| **F-04** | Semantics | **MAJOR** | Mapping non-finite floats (`NaN`, `+Inf`, `-Inf`) to `0.0` masks fatal out-of-memory hazards as zero memory. | RFC 8259 sanitization maps `NaN` and `+Inf`/`-Inf` to `None` (JSON `null`), NEVER `0.0`. Unbounded memory estimates append explicit strings to `memory_warnings` (e.g. `"Axis '<name>' memory estimate is unbounded (+Infinity)"`). (§3.4) |
| **F-05** | Determinism | **MAJOR** | IEEE-754 platform differences in float formatting and `-0.0` break SHA-256 artifact reproducibility across architectures. | `export_canonical_json` normalizes `-0.0` to `0.0`, formats floats to 8 decimal places (`f"{val:.8f}"`), uses `json.dumps(..., sort_keys=True, separators=(',', ':'), allow_nan=False)`, and encodes UTF-8 NFC. (§3.5) |
| **F-06** | Offline / Airgap | **MAJOR** | HTML rendering relying solely on unpkg/CDN Vega-Embed fails on airgapped HPC clusters, producing blank screens. | Progressive enhancement: HTML document embeds static SVG snapshot directly in `<div id="fallback-svg">`. Vega-Embed script includes 2-second timeout guard; static SVG remains visible if offline. Added optional `offline=False` / `bundle_js=False` flags. (§3.6) |
| **F-07** | Plugin Security | **MAJOR** | Third-party plugin panels can mutate `stats` in-place (`stats["axes"].pop()`), corrupting core stats and downstream panels. Malformed plugin specs break dashboard. | Plugin runner passes `copy.deepcopy(stats)` to each plugin panel. Plugin-emitted Vega-Lite JSON is pre-validated against Vega-Lite v5 snippet schema; invalid panels are caught and replaced with warning placeholder cards. (§3.7) |
| **F-08** | Testability | **MAJOR** | AC-15 and AC-16 lacked observable seams for asserting plugin quarantine and HTML DOM structure without testing implementation internals. | AC-15 specifies public quarantine query `xtrax.eda.plugins.get_quarantined_plugins() -> list[str]`. AC-16 specifies exact DOM element IDs (`#fallback-svg`, `#vega-vis`) and script timeout attributes. (§6) |
| **F-09** | Performance | **MINOR** | Repeated `entry_points()` discovery on every `render()` call causes significant filesystem I/O overhead. | In-memory discovery cache `_PLUGIN_CACHE` initialized on first scan; provided `xtrax.eda.plugins.reload_plugins()` to force cache invalidation. (§3.7) |
| **F-10** | Namespace | **NOTE** | Conceptual confusion between `xtrax.eda` and `maraxiom.eda`. | Formally affirmed boundary: `xtrax.eda` is compiler/plan inspection; `maraxiom.eda` is scientific figure data exploration. Zero imports, zero dependencies between them. (§1.3, §3.8) |

---

## 3. Detailed Architectural Specification

```
+-----------------------------------------------------------------------------------+
|                                  xtrax.eda Pipeline                               |
+-----------------------------------------------------------------------------------+
|  1. Compilation Plan: BatchPlan (from BatchPlanner.plan)                          |
|         │                                                                         |
|         ▼                                                                         |
|  2. Pure Stats Extraction (stdlib + numpy only, NEVER truncated)                  |
|     extract_plan_stats() -> PlanStatsDict (100% of axes captured)                 |
|     sanitize_floats() -> RFC 8259 compliant (NaN/+Inf/-Inf -> None, never 0.0)    |
|         │                                                                         |
|         ├──────────────────────────────────────────────┐                          |
|         ▼                                              ▼                          |
|  3. Streaming & Tabular Export           4. Pluggable Panel Engine                |
|     export_axes_csv(stats)                  [xtrax.eda.panels]                    |
|     export_canonical_json(stats)            Isolated via copy.deepcopy(stats)     |
|     (SHA-256 bitwise reproducible)          Pre-validated Vega-Lite snippets      |
|         │                                              │                          |
|         └───────────────────────┬──────────────────────┘                          |
|                                 ▼                                                 |
|  5. Declarative Vega-Lite Spec Generation (xtrax.eda.spec)                        |
|     - total_axes == 0: Single Text-Mark Chart (NEVER empty vconcat: [])           |
|     - total_axes > max_axes: Top-N + Aggregated Summary Badge                     |
|         │                                                                         |
|         ▼                                                                         |
|  6. Progressive Rendering (xtrax.eda.viz.render)                                  |
|     - fmt="png" / "svg" -> Headless raster/vector bytes                           |
|     - fmt="html"        -> Progressive Enhancement HTML:                          |
|                            <div id="fallback-svg"> (zero-blank airgap guarantee)  |
|                            <div id="vega-vis"> (2s CDN timeout progressive load)  |
+-----------------------------------------------------------------------------------+
```

### 3.1 Schema & Data Integrity: Empty Plans & `minItems: 0` [F-01, F-02]
`PlanStatsDict` is the canonical intermediate data representation. It is defined as a typed dictionary and verified against JSON Schema (Draft 2020-12).

#### 3.1.1 JSON Schema Contract
In accordance with **F-01**, `axes`, `dedup_stats`, `bucket_stats`, and `memory_warnings` explicitly specify `"minItems": 0`. An empty plan `BatchPlan(decisions=())` compiles to a dictionary where `total_axes == 0`, `axes == []`, `strategy_counts == {}`, `memory_warnings == []`, `dedup_stats == []`, and `bucket_stats == []`. This schema is valid and compliant under all validation criteria.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://xtrax.dev/schemas/plan_stats_dict.v2.json",
  "title": "PlanStatsDict",
  "type": "object",
  "required": [
    "axes",
    "strategy_counts",
    "total_axes",
    "memory_warnings",
    "dedup_stats",
    "bucket_stats"
  ],
  "additionalProperties": false,
  "properties": {
    "axes": {
      "type": "array",
      "minItems": 0,
      "items": {
        "type": "object",
        "required": ["name", "strategy", "cardinality", "batch_size", "reasoning"],
        "additionalProperties": false,
        "properties": {
          "name": {"type": "string"},
          "strategy": {"type": "string"},
          "cardinality": {"type": "integer", "minimum": 0},
          "batch_size": {"type": "integer", "minimum": 0},
          "reasoning": {"type": "string"},
          "memory_estimate_bytes": {"type": ["integer", "null"], "minimum": 0},
          "varying_inputs": {
            "type": "array",
            "minItems": 0,
            "items": {"type": "string"}
          }
        }
      }
    },
    "strategy_counts": {
      "type": "object",
      "additionalProperties": {"type": "integer", "minimum": 0}
    },
    "total_axes": {"type": "integer", "minimum": 0},
    "memory_warnings": {
      "type": "array",
      "minItems": 0,
      "items": {"type": "string"}
    },
    "dedup_stats": {
      "type": "array",
      "minItems": 0,
      "items": {
        "type": "object",
        "required": ["axis_name", "dedup_ratio", "unique_count", "padded_count", "total_count", "padding_waste"],
        "additionalProperties": false,
        "properties": {
          "axis_name": {"type": "string"},
          "dedup_ratio": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0},
          "unique_count": {"type": "integer", "minimum": 0},
          "padded_count": {"type": "integer", "minimum": 0},
          "total_count": {"type": "integer", "minimum": 0},
          "padding_waste": {"type": "integer", "minimum": 0}
        }
      }
    },
    "bucket_stats": {
      "type": "array",
      "minItems": 0,
      "items": {
        "type": "object",
        "required": ["axis_name", "bucket_count", "bucket_boundaries"],
        "additionalProperties": false,
        "properties": {
          "axis_name": {"type": "string"},
          "bucket_count": {"type": "integer", "minimum": 0},
          "bucket_boundaries": {
            "type": "array",
            "minItems": 0,
            "items": {"type": "integer", "minimum": 0}
          }
        }
      }
    }
  }
}
```

### 3.2 Vega-Lite Specification Engine: Zero-Axis Handling [F-01, F-02]
Module `src/xtrax/eda/spec.py` exposes:
```python
def generate_vega_spec(
    stats: PlanStatsDict,
    panels: set[str] | None = None,
    max_axes: int = 100,
    overflow: Literal["aggregate", "truncate"] = "aggregate",
) -> dict[str, Any]: ...
```

#### 3.2.1 The Zero-Axis Contract
When `stats["total_axes"] == 0`:
1. It is **STRICTLY PROHIBITED** to construct a compound specification with an empty vertical concatenation `{"vconcat": []}`. Such a structure is invalid Vega-Lite and crashes client-side parsers.
2. The generator outputs a single valid Vega-Lite root chart with a text mark:
```json
{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "description": "Empty xtrax compilation plan",
  "width": 600,
  "height": 180,
  "padding": 20,
  "data": {
    "values": [{}]
  },
  "mark": {
    "type": "text",
    "text": "No active execution axes in compilation plan",
    "fontSize": 15,
    "fontStyle": "italic",
    "color": "#718096",
    "align": "center",
    "baseline": "middle"
  }
}
```
This guarantees that `render(BatchPlan(decisions=()))` produces clean, valid visual output without error across all formats (reconciling **AC-09** and **AC-13**).

---

### 3.3 Scalability & Separation of Concerns: `max_axes` & Overflow [F-03]

#### 3.3.1 Principle of Clean Separation
- `extract_plan_stats(plan)` is a **pure analytical extractor**. It MUST NEVER truncate, sample, or discard axes. Even if a compilation plan contains 5,000 axes, `stats["axes"]` contains exactly 5,000 entries.
- `max_axes: int = 100` belongs **exclusively** to the visual rendering layer (`render()` and `generate_vega_spec()`). It protects browser DOMs and SVG renderers from GPU/memory exhaustion.

#### 3.3.2 Rendering Overflow Mechanics
When `stats["total_axes"] > max_axes`:
1. **Top-N Partitioning**: The visualizer sorts axes by cardinality (descending) and selects the top `max_axes` axes for detailed per-axis marks (e.g., cardinality scatter and memory bar charts).
2. **Aggregated Summary Badge**:
   - When `overflow == "aggregate"` (default), the remaining `K = total_axes - max_axes` axes are summarized into an aggregated summary badge and informational layer:
     `"+ {K} remaining axes aggregated (see axes.csv for full streaming tabular record)"`.
   - The strategy distribution histogram always reflects **100% of axes**, as aggregate strategy counts do not scale with axis count.
3. **Tabular Streaming**: Users needing to inspect all axes on large models are directed to the streaming tabular artifact `axes.csv` generated by `export_axes_csv()`.

---

### 3.4 Semantic Integrity: RFC 8259 Float Sanitization [F-04]

#### 3.4.1 The Memory Masking Hazard
In compiler planning, an unbounded axis cardinality or infinite memory estimate (`+Infinity`) represents a fatal out-of-memory (OOM) compilation hazard. Mapping `+Inf` or `NaN` to `0.0` masks an infinite memory requirement as zero memory consumption, misleading operators and automated CI gates into approving fatally oversized batch executions.

#### 3.4.2 Strict Sanitization Contract
All numbers exported to JSON or Vega-Lite specifications must strictly conform to RFC 8259 (which forbids `NaN`, `Infinity`, and `-Infinity`):
1. **Mapping Rule**: Every non-finite float (`NaN`, `+Inf`, `-Inf`) MUST be converted to `None` (serialized as JSON `null`), NEVER `0.0`.
2. **Warning Generation**: Whenever an axis exhibits unbounded or infinite memory, `extract_plan_stats()` appends an explicit warning to `stats["memory_warnings"]`:
   ```python
   if math.isinf(mem_est) or math.isnan(mem_est):
       memory_warnings.append(
           f"Axis '{axis_name}' memory estimate is unbounded (+Infinity)"
       )
   ```
3. In downstream Vega-Lite specs, `null` values are cleanly excluded from log-scale axes without throwing JavaScript runtime evaluation errors.

---

### 3.5 Bitwise Cross-Platform Determinism in Canonical JSON [F-05]

To allow compiler artifacts to serve as cryptographic trust anchors and regression gate baselines, `export_canonical_json` implements a bitwise-deterministic pipeline across Python 3.10+, x86_64, and aarch64 architectures.

#### 3.5.1 Serialization Pipeline
```python
def export_canonical_json(stats: PlanStatsDict, path: PathLike | None = None) -> bytes:
    """Export PlanStatsDict to bitwise-deterministic RFC 8259 JSON bytes.
    
    Guarantees:
    1. Normalizes -0.0 to 0.0.
    2. Floating-point numbers formatted with fixed 8-decimal precision.
    3. Keys sorted lexicographically at all nesting depths.
    4. Compact delimiters without whitespace (separators=(',', ':')).
    5. Unicode NFC normalized UTF-8 encoding.
    6. allow_nan=False to strictly enforce RFC 8259 compliance.
    """
    sanitized = _sanitize_and_normalize(stats)
    json_str = json.dumps(
        sanitized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    normalized_str = unicodedata.normalize("NFC", json_str)
    encoded_bytes = normalized_str.encode("utf-8")
    
    if path is not None:
        p = Path(path)
        # Atomic write pattern
        tmp_path = p.with_suffix(p.suffix + ".tmp")
        tmp_path.write_bytes(encoded_bytes)
        tmp_path.replace(p)
        
    return encoded_bytes
```

#### 3.5.2 Float Normalization Specification
Floats are recursively normalized:
- Any float `val` equal to `0.0` (including `-0.0`) is replaced with `0.0`.
- All other floats are rounded to 8 decimal places: `round(val, 8)`.

---

### 3.6 Airgapped & Offline Progressive HTML Dashboard [F-06]

High-performance computing (HPC) clusters and secure enterprise enclaves frequently execute in airgapped environments without external internet connectivity. HTML dashboards that depend solely on public CDN scripts (`cdn.jsdelivr.net` or `unpkg.com`) fail catastrophically, leaving engineers with a blank screen.

#### 3.6.1 Progressive Enhancement Architecture
When `render(plan, fmt="html")` is called:
1. **Static SVG Fallback**: A self-contained, valid static SVG snapshot of the compilation plan is generated and embedded directly inside `<div id="fallback-svg">...</div>`. This SVG requires zero external network calls, zero JavaScript, and displays instantly in any browser.
2. **Progressive Vega Container**: An interactive container `<div id="vega-vis" style="display: none;"></div>` is placed in the DOM.
3. **2-Second CDN Timeout Guard**:
   An inline JavaScript loader attempts to initialize Vega-Embed asynchronously.
   - If the script libraries load successfully within 2000ms:
     `#fallback-svg` is hidden (`display: none`), and `#vega-vis` is displayed with interactive tooltips, cross-filtering, and zoom controls.
   - If the CDN times out or network is unreachable:
     The script aborts gracefully, leaving `#fallback-svg` fully visible and functional. The user experiences zero blank screen.
4. **Offline Packaging**:
   `render()` provides `offline: bool = False` and `bundle_js: bool = False`. When `offline=True`, the HTML template references locally vendored JavaScript assets or inlines them directly if local caching is configured.

#### 3.6.2 Concrete HTML Structure
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>xtrax Compilation Plan EDA Dashboard</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 20px; background: #f7fafc; }
    .panel { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    #fallback-svg { display: block; width: 100%; overflow-x: auto; }
    #vega-vis { width: 100%; }
    .status-badge { font-size: 12px; color: #718096; margin-bottom: 12px; }
  </style>
</head>
<body>
  <div class="panel">
    <div class="status-badge" id="loader-status">Rendering plan visualization...</div>
    <!-- 1. Offline static SVG snapshot -->
    <div id="fallback-svg">
      <!-- STATIC_SVG_CONTENT_INLINED_HERE -->
    </div>
    <!-- 2. Interactive Vega-Lite container -->
    <div id="vega-vis" style="display: none;"></div>
  </div>

  <script>
    (function() {
      var timeout = setTimeout(function() {
        var status = document.getElementById('loader-status');
        if (status) status.innerText = 'Offline mode: displaying static vector snapshot.';
      }, 2000);

      function loadScript(src, cb) {
        var s = document.createElement('script');
        s.src = src;
        s.onload = cb;
        s.onerror = function() { clearTimeout(timeout); };
        document.head.appendChild(s);
      }

      loadScript('https://cdn.jsdelivr.net/npm/vega@5', function() {
        loadScript('https://cdn.jsdelivr.net/npm/vega-lite@5', function() {
          loadScript('https://cdn.jsdelivr.net/npm/vega-embed@6', function() {
            clearTimeout(timeout);
            var spec = /* VEGA_SPEC_JSON_INLINED_HERE */;
            vegaEmbed('#vega-vis', spec, {actions: false}).then(function() {
              document.getElementById('fallback-svg').style.display = 'none';
              document.getElementById('vega-vis').style.display = 'block';
              var status = document.getElementById('loader-status');
              if (status) status.innerText = 'Interactive inspection active.';
            }).catch(function(err) {
              console.warn('Vega-Embed initialization failed; retaining static SVG snapshot.', err);
            });
          });
        });
      });
    })();
  </script>
</body>
</html>
```

---

### 3.7 Pluggable Panel Architecture & Immutability Quarantine [F-07, F-09]

#### 3.7.1 Entry-Point Discovery
Panels and backends are registered using standard Python packaging entry points in `pyproject.toml`:
```toml
[project.entry-points."xtrax.eda.panels"]
strategy = "xtrax.eda.panels.strategy:StrategyPanel"
cardinality = "xtrax.eda.panels.cardinality:CardinalityPanel"
dedup = "xtrax.eda.panels.dedup:DedupPanel"
bucket = "xtrax.eda.panels.bucket:BucketPanel"
memory = "xtrax.eda.panels.memory:MemoryPanel"
reasoning = "xtrax.eda.panels.reasoning:ReasoningPanel"
```

#### 3.7.2 Discovery Caching [F-09]
To prevent repeated filesystem metadata lookups during batch compilation loops, discovery is cached in-memory upon the first call:
```python
_PLUGIN_CACHE: dict[str, PanelPlugin] | None = None

def get_registered_panels() -> dict[str, PanelPlugin]:
    global _PLUGIN_CACHE
    if _PLUGIN_CACHE is None:
        _PLUGIN_CACHE = _discover_entry_point_panels()
    return _PLUGIN_CACHE

def reload_plugins() -> None:
    """Force re-scan of Python entry-points (useful in dynamic plugin environments & tests)."""
    global _PLUGIN_CACHE
    _PLUGIN_CACHE = None
```

#### 3.7.3 Stats Immutability Quarantine (`copy.deepcopy`) [F-07]
Third-party plugins must execute under strict quarantine to prevent accidental or malicious corruption of the core plan statistics:
1. **Deep Copy Guarantee**: The plugin dispatcher passes `copy.deepcopy(stats)` to the plugin panel callable. Any in-place mutation (such as `stats["axes"].pop()`) affects only the plugin's isolated copy.
2. **Snippet Validation**: The returned dictionary from a panel plugin is validated against the Vega-Lite v5 UnitSpec / LayerSpec schema.
3. **Quarantine Isolation**: If a plugin raises an unhandled exception or returns a malformed specification snippet:
   - The plugin is caught and quarantined.
   - Its name is recorded in an internal quarantine ledger:
     `_QUARANTINED_PLUGINS.append(plugin_name)`.
   - The plugin's visual panel is replaced with a standardized warning card mark:
     `"Panel '{plugin_name}' failed to render: {error}"`.
   - The core dashboard continues rendering without crashing.

#### 3.7.4 Testability Seam [F-08]
The quarantine status is exposed via public function:
```python
def get_quarantined_plugins() -> list[str]:
    """Return list of names of plugin panels that failed execution or validation."""
    return list(_QUARANTINED_PLUGINS)
```

---

### 3.8 Strict Namespace Boundary & Disambiguation [F-10]

#### 3.8.1 The Ambiguity Hazard
In multi-agent and cross-repository development, engineers and subagents can conflate `xtrax.eda` with `maraxiom.eda`.
- `maraxiom.eda` (`mrx eda run` / `run_eda_check`) is a **scientific exploratory data analysis framework** designed for auditing experimental biology and biochemistry figures, validating replicate counts, assessing distributional assumptions, and verifying biophysical effect sizes.
- `xtrax.eda` is a **compiler-level plan inspection and execution analysis subsystem** designed to inspect `BatchPlan` strategy assignments, batch memory estimates, deduplication rates, and compilation decisions before JIT compilation.

#### 3.8.2 Strict Architectural Separation
1. **Zero Cross-Imports**: `xtrax.eda` NEVER imports, references, or depends on `maraxiom.eda`. Similarly, `maraxiom.eda` does not depend on `xtrax.eda`.
2. **Decoupled Roles**: `xtrax.eda` belongs strictly to the compiler and runtime profiling pipeline of `xtrax`.
3. **No Common Schemas**: The data contracts of `xtrax.eda` (`PlanStatsDict`) share no schema or runtime dependencies with `maraxiom` manifest models.

---

## 4. Public API Specification

### 4.1 Module: `src/xtrax/eda/types.py` (Zero viz dependencies)
```python
from __future__ import annotations
from typing import Literal, NotRequired, Protocol, TypedDict

class AxisStatsEntry(TypedDict):
    name: str
    strategy: str
    cardinality: int
    batch_size: int
    reasoning: str
    memory_estimate_bytes: NotRequired[int | None]
    varying_inputs: NotRequired[list[str]]

class DedupStatsEntry(TypedDict):
    axis_name: str
    dedup_ratio: float | None
    unique_count: int
    padded_count: int
    total_count: int
    padding_waste: int

class BucketStatsEntry(TypedDict):
    axis_name: str
    bucket_count: int
    bucket_boundaries: list[int]

class PlanStatsDict(TypedDict):
    axes: list[AxisStatsEntry]
    strategy_counts: dict[str, int]
    total_axes: int
    memory_warnings: list[str]
    dedup_stats: list[DedupStatsEntry]
    bucket_stats: list[BucketStatsEntry]

PanelName = Literal["strategy", "cardinality", "dedup", "bucket", "memory", "reasoning"]

class PlanLogger(Protocol):
    def log_figure(
        self,
        figure: bytes | str,
        fmt: str,
        step: int | None = None,
    ) -> None: ...
```

### 4.2 Module: `src/xtrax/eda/stats.py` (Zero viz dependencies)
```python
def extract_plan_stats(plan: BatchPlanLike) -> PlanStatsDict:
    """Extract complete structured statistics from a BatchPlan.
    
    NEVER truncates axes. Captures 100% of decisions in plan.
    Sanitizes non-finite floats to None (null).
    Appends warning to memory_warnings if memory estimate is infinite or unbounded.
    """
    ...

def analyze_dedup(decision: AxisDecisionLike) -> DedupStatsEntry: ...
def analyze_bucket(decision: AxisDecisionLike) -> BucketStatsEntry: ...
```

### 4.3 Module: `src/xtrax/eda/spec.py`
```python
def generate_vega_spec(
    stats: PlanStatsDict,
    panels: set[str] | None = None,
    max_axes: int = 100,
    overflow: Literal["aggregate", "truncate"] = "aggregate",
) -> dict[str, Any]:
    """Generate a canonical Vega-Lite v5 JSON specification from PlanStatsDict.
    
    If stats["total_axes"] == 0:
        Returns a single valid Vega-Lite root chart with a text mark. Never emits {"vconcat": []}.
    If stats["total_axes"] > max_axes:
        Visualizes top max_axes, summarizes overflow, and references axes.csv.
    """
    ...
```

### 4.4 Module: `src/xtrax/eda/viz.py`
```python
def render(
    plan: BatchPlan,
    view: str = "dashboard",
    fmt: Literal["png", "svg", "html"] = "png",
    path: str | Path | None = None,
    stats_transform: Callable[[PlanStatsDict], PlanStatsDict] | None = None,
    metadata: bool = False,
    logger: PlanLogger | None = None,
    step: int | None = None,
    panels: set[str] | None = None,
    max_axes: int = 100,
    overflow: Literal["aggregate", "truncate"] = "aggregate",
    offline: bool = False,
    bundle_js: bool = False,
) -> bytes | str | None:
    """Render a BatchPlan into the target format.
    
    fmt="png": Returns bytes (PNG header).
    fmt="svg": Returns bytes (UTF-8 encoded SVG).
    fmt="html": Returns str with progressive enhancement (fallback SVG + 2s Vega-Embed timeout).
    """
    ...
```

### 4.5 Module: `src/xtrax/eda/export.py`
```python
def export_canonical_json(stats: PlanStatsDict, path: PathLike | None = None) -> bytes:
    """Export PlanStatsDict to bitwise-reproducible UTF-8 RFC 8259 JSON."""
    ...

def export_axes_csv(stats: PlanStatsDict, path: PathLike | None = None) -> str:
    """Export 100% of axes to streaming CSV string or file."""
    ...
```

### 4.6 Module: `src/xtrax/eda/plugins.py`
```python
def get_registered_panels() -> dict[str, PanelPlugin]: ...
def reload_plugins() -> None: ...
def get_quarantined_plugins() -> list[str]: ...
def reset_plugin_registry() -> None: ...
```

---

## 5. Comprehensive Acceptance Criteria

- **AC-01 [Headless Execution]**: All rendering and export functions must execute in headless CI environments without requiring an X11/Wayland display server or GUI toolkit.
- **AC-02 [PNG Format]**: `render(plan, fmt="png")` returns non-empty `bytes` starting with the standard PNG magic header `\x89PNG\r\n\x1a\n`.
- **AC-03 [SVG Format]**: `render(plan, fmt="svg")` returns non-empty `bytes` containing valid UTF-8 encoded XML starting with `<svg` or `<?xml`.
- **AC-04 [HTML Progressive Enhancement]**: `render(plan, fmt="html")` returns a valid HTML5 `str` containing both a static fallback SVG inside `<div id="fallback-svg">` and an interactive container `<div id="vega-vis">`.
- **AC-05 [Transform Hook]**: `stats_transform: Callable[[PlanStatsDict], PlanStatsDict]` executes after stats extraction but prior to specification generation; modified values are reflected in the rendered output.
- **AC-06 [Metadata Sidecar]**: `render(plan, path="out.png", metadata=True)` writes `out.png` and atomically writes `out.json` containing the canonical `PlanStatsDict`.
- **AC-07 [Logger Protocol]**: When `logger` is passed to `render(plan, logger=l, step=42, fmt="png")`, `logger.log_figure(figure, "png", step=42)` is called with `figure` matching the rendered bytes.
- **AC-08 [Panel Filtering]**: `render(plan, panels={"strategy"})` renders only the strategy distribution panel; unspecified panels are excluded from the output.
- **AC-09 [RECONCILED: Empty Plan Visual Output]**: Calling `render(BatchPlan(decisions=()), fmt=fmt)` for any `fmt` in `("png", "svg", "html")` returns valid non-empty output without raising an exception.
- **AC-10 [Metadata Path Invariant]**: Calling `render(plan, metadata=True, path=None)` raises `ValueError("metadata=True requires path to be set")`.
- **AC-11 [Panel Validation]**: Calling `render(plan, panels={"invalid_name"})` raises `ValueError` enumerating the valid panel names.
- **AC-12 [Dependency Isolation]**: Importing `xtrax.eda.types` or `xtrax.eda.stats` in an environment without visualization extras succeeds without error.
- **AC-13 [RECONCILED: JSON Schema Conformance]**: Both empty plans `BatchPlan(decisions=())` and N-axis plans validate against the `PlanStatsDict` JSON schema with `minItems: 0` on `axes`.
- **AC-14 [Scalability & Separation]**: For a plan with 250 axes:
  - `extract_plan_stats(plan)["total_axes"] == 250` and `len(stats["axes"]) == 250`.
  - `render(plan, max_axes=50)` renders top 50 axes, adds "+ 200 remaining axes" badge, and directs user to `axes.csv`.
- **AC-15 [CONCRETE SEAM: Quarantined Plugins]**:
  - A third-party panel plugin attempting `stats["axes"].pop()` does not alter the core plan statistics passed to subsequent panels.
  - A third-party plugin throwing an exception is recorded in `get_quarantined_plugins()`, and its panel is replaced with an error placeholder card without halting rendering.
- **AC-16 [CONCRETE SEAM: Offline HTML DOM Contract]**:
  - `render(plan, fmt="html")` produces HTML containing `<div id="fallback-svg">` with child `<svg` element.
  - Emitted HTML contains `<div id="vega-vis" style="display: none;">` and JavaScript setting a 2000ms timeout guard on CDN resolution.
- **AC-17 [Bitwise Deterministic Export]**:
  - `export_canonical_json` output normalizes `-0.0` to `0.0` and floats to 8 decimal places.
  - Byte-for-byte SHA-256 digest of `export_canonical_json(stats)` is identical across Python 3.10, 3.11, 3.12, Linux x86_64, and macOS aarch64.
- **AC-18 [Float Sanitization Semantics]**:
  - `NaN`, `+Inf`, and `-Inf` memory estimates are serialized as `null` (`None`), NEVER `0.0`.
  - Infinite memory estimates generate an explicit warning entry in `stats["memory_warnings"]`.

---

## 6. Implementation Task Breakdown

| Task ID | Component | Description & Target Files |
|---|---|---|
| **T0** | Schema & Types | Update `src/xtrax/eda/types.py` with `minItems: 0` schema documentation, `varying_inputs`, and `PlanLogger` protocol. |
| **T1** | Core Stats & Sanitization | Update `src/xtrax/eda/stats.py`: 100% axis capture (no truncation), float sanitization mapping non-finite values to `None`, and infinite memory warning generation. |
| **T2** | Vega-Lite Spec Engine | Implement `src/xtrax/eda/spec.py`: zero-axis text-mark chart, standard panels (strategy, cardinality, dedup, bucket, memory, reasoning), and `max_axes` overflow aggregation. |
| **T3** | Canonical JSON & CSV Export | Implement `src/xtrax/eda/export.py`: bitwise-deterministic `export_canonical_json` (-0.0 normalization, 8-decimal precision, NFC UTF-8) and streaming `export_axes_csv`. |
| **T4** | Plugin Architecture & Quarantine | Implement `src/xtrax/eda/plugins.py`: entry-point discovery with caching, `reload_plugins()`, `copy.deepcopy` isolation, snippet validation, and `get_quarantined_plugins()`. |
| **T5** | Progressive HTML & Viz Engine | Update `src/xtrax/eda/viz.py`: progressive HTML template with `<div id="fallback-svg">` and 2s timeout guard, Vega-Lite export pipeline. |
| **T6** | Test Suite Hardening | Update `tests/eda/`: test zero-axis plans, schema validation, plugin quarantine seams, offline HTML DOM structure, and float sanitization. |
| **T7** | Documentation & Reference | Update `docs/advanced/eda-guide.md` and `agent_assets/skills/using-xtrax/references/eda.md` to reflect Vega-Lite architecture and progressive rendering. |

---

## 7. Sign-Off & Convergence

This Revision 2.1 specification resolves all 2 Blockers, 6 Majors, 1 Minor, and 1 Architectural Note raised during the adversarial review cycle. It establishes a rigorous, scalable, bitwise-deterministic foundation for compiler plan inspection in `xtrax`.
