> Part of the `using-xtrax` skill (`agent_assets/skills/using-xtrax/SKILL.md`) — TIER-2 deep reference.

# EDA: Plan Analysis and Visualization (10% of depth)

`xtrax.eda` provides compiler-plan inspection and verification tooling. It is decoupled into three tiers:
1. **Core Stats Extraction**: stdlib + numpy only, zero visualization dependencies.
2. **Declarative Specification**: compiles stats into portable Vega-Lite v5 JSON specifications.
3. **Pluggable Renderers & Deterministic Exports**: headless raster/vector generation, airgap-safe progressive HTML dashboards, and SHA-256 hashed artifact bundles.

### Disambiguation from `scientific-eda`
`xtrax.eda` is strictly a compiler and planning inspection subsystem for `BatchPlan` execution schedules, axis strategies, and per-axis memory footprints. It does not perform scientific data exploration or figure validation. For statistical analysis and data integrity verification behind scientific figures, consult the `scientific-eda` skill.

---

### Core Analysis (Zero Visualization Dependencies)

The analytical core extracts 100% of plan axes without truncation:

```python
from xtrax.eda.stats import extract_plan_stats  # verify: src/xtrax/eda/stats.py

plan = planner.plan([spec1, spec2, ...])
stats = extract_plan_stats(plan)

# stats is a PlanStatsDict:
# {
#   "axes": [
#       {
#           "name": "batch",
#           "strategy": "Vmap",
#           "cardinality": 100,
#           "batch_size": 32,
#           "reasoning": "Fits in memory",
#           "memory_estimate_bytes": None,
#           "varying_inputs": ["inputs"],
#       },
#       ...
#   ],
#   "strategy_counts": {"Vmap": 1, ...},
#   "total_axes": 1,
#   "memory_warnings": [],
#   "dedup_stats": [],
#   "bucket_stats": [],
# }
```

Verify: `src/xtrax/eda/stats.py:28`

#### Structural Protocols
`extract_plan_stats` accepts any object satisfying `BatchPlanLike` (`.decisions` sequence of `AxisDecisionLike` with `.spec`, `.batch_size`, `.reasoning`, `.strategy`). Strategy matching is protocol-narrowed via `DedupGatherLike` (`.k`, `.k_bucket`) and `BucketLike` (`.boundaries`). No inheritance from concrete classes is required.

Verify: `src/xtrax/eda/types.py:104-185`

#### explain_plan: Guaranteed Non-Empty Reasoning
Wrapper ensuring all axis reasoning fields are non-empty:

```python
from xtrax.eda.explain import explain_plan

stats = explain_plan(plan)
# All stats["axes"][i]["reasoning"] are guaranteed non-empty strings
```

Verify: `src/xtrax/eda/explain.py:14-41`

#### Per-Axis Strategy Analysis

```python
from xtrax.eda.stats import analyze_dedup, analyze_bucket

# Dedup analysis (computes dedup_ratio = 1 - unique/total)
dedup_stats = analyze_dedup(decision)

# Bucket analysis
bucket_stats = analyze_bucket(decision)
```

Verify: `src/xtrax/eda/stats.py:97-175`

---

### Declarative Vega-Lite Specification Engine

Compile plan statistics directly into a Vega-Lite v5 specification dictionary without graphics dependencies:

```python
from xtrax.eda.spec import generate_vega_spec

vl_spec = generate_vega_spec(stats, panels={"strategy", "cardinality"}, max_axes=100)
```

- When `stats["total_axes"] == 0`, produces a valid single text-mark chart (never empty `vconcat: []`).
- Sanitizes non-finite floats (`NaN`, `+Inf`, `-Inf`) to `None` (`null`) per RFC 8259.
- Visual truncation (`max_axes=100`, `overflow="aggregate"`) groups excess axes into an aggregated badge while preserving 100% axis telemetry in `stats`.

Verify: `src/xtrax/eda/spec.py:190-250`

---

### Visualization and Rendering (`render`)

`render()` produces raster bytes, vector bytes, or progressive HTML:

```python
from xtrax.eda import render

# 1. Default PNG raster rendering (returns bytes)
png_bytes = render(plan, fmt="png")

# 2. Vector SVG rendering (returns bytes)
svg_bytes = render(plan, fmt="svg")

# 3. Progressive HTML dashboard (returns str)
html_str = render(plan, fmt="html", backend="vega")

# 4. Save directly to file (returns None)
render(plan, path="plan_dashboard.html", fmt="html")
```

#### Signature and Options
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
    backend: Literal["auto", "vega", "matplotlib"] = "auto",
    max_axes: int = 100,
    overflow: Literal["aggregate", "truncate"] = "aggregate",
    offline: bool = False,
    bundle_js: bool = False,
) -> bytes | str | None: ...
```

- **Return Types**: Returns `bytes` for `"png"` / `"svg"` when `path=None`; returns `str` for `"html"` when `path=None`; returns `None` when `path` is specified.
- **Backends**:
  - `"matplotlib"` (legacy default): uses Seaborn/Matplotlib (`xtrax[eda-matplotlib]`).
  - `"vega"`: uses Vega-Lite specification engine and `vl-convert-python` (`xtrax[eda-vega]`).
- **Progressive Offline HTML**: HTML output embeds a static SVG in `<div id="fallback-svg">` alongside an interactive container `<div id="vega-vis">` with a 2-second timeout guard on CDN assets, guaranteeing immediate visibility on airgapped compute nodes.
- **PlanLogger Protocol**: Any remote logger implementing `.log_figure(figure, fmt, step)` can be passed to log figures to experiment trackers.

Verify: `src/xtrax/eda/viz.py:44-126`

---

### Deterministic Exports and Hashed Artifacts

Export bitwise-reproducible records and neutral tabular datasets:

```python
from xtrax.eda.export import (
    export_canonical_json,
    export_axes_csv,
    export_plan_artifacts,
    plan_to_dataframe,
)

# Canonical RFC 8259 JSON (-0.0 normalized, 8-decimal floats, sorted keys)
json_bytes = export_canonical_json(stats, path="plan.canonical.json")

# Streaming tidy CSV with 100% of axes
csv_str = export_axes_csv(stats, path="axes.csv")

# Complete self-verifying artifact bundle with SHA-256 manifest
manifest = export_plan_artifacts(stats, output_dir="./artifacts", prefix="plan", context="run-001")

# Convert stats to pandas DataFrame (requires xtrax[eda] or xtrax[eda-matplotlib])
df = plan_to_dataframe(stats)
```

Verify: `src/xtrax/eda/export.py`

---

### Extras Packaging Split

- **Core `xtrax`**: stdlib + numpy only. Core analysis (`extract_plan_stats`, `explain_plan`, `generate_vega_spec`) runs with zero visualization dependencies.
- **`xtrax[eda-vega]`**: installs `vl-convert-python` for headless Vega-Lite rasterization and vector rendering.
- **`xtrax[eda-matplotlib]`**: installs `pandas`, `matplotlib`, and `seaborn` for legacy visualization.
- **`xtrax[eda]`**: metapackage installing both visualization backends.
