---
session_id: d3821b15
task_id: 260615_eda-viz-api
task_type: constrained-technical
topic: xtrax EDA visualization API — revised post-critic
phase: post-synthesis (critic resolutions applied)
date: 2026-06-15
winner: Seaborn-backed render() API with PlanStats-first design
critic_verdict: NEEDS_WORK (two FATALs resolved; spec updated)
synthesizer_verdict: PASS
---

# xtrax EDA Visualization API — Revised Spec (post-critic)

## Fixed Constraints

1. `xtrax` core stays free of viz dependencies — the `eda` group is strictly optional.
2. `stats.py` works with stdlib + numpy only (no pandas, no matplotlib required).
3. Output formats PNG, SVG, HTML are all required targets.
4. Python 3.10+ type signatures required throughout.
5. Headless rendering is mandatory — API must be testable without a display.

---

## Dependency Group

```toml
[dependency-groups]
eda = ["pandas>=2.0", "matplotlib>=3.8", "seaborn>=0.13"]
```

Install: `pip install xtrax[eda]`

---

## Types (`src/xtrax/eda/types.py`)

No eda extras required for this module — uses only `typing` and stdlib.

```python
from __future__ import annotations
from typing import TypedDict, Literal
from typing import Protocol


class AxisStatsEntry(TypedDict):
    name: str
    strategy: str            # e.g. "Vmap", "SafeMap", "Scan"
    cardinality: int
    batch_size: int
    reasoning: str           # human-readable from AxisDecision.reasoning
    memory_estimate_bytes: int | None


class DedupStatsEntry(TypedDict):
    axis_name: str
    dedup_ratio: float       # 0.0–1.0; higher = more deduplication
    unique_count: int        # k (pre-padding)
    padded_count: int        # k_bucket (post-padding)
    total_count: int
    padding_waste: int       # padded_count - unique_count


class BucketStatsEntry(TypedDict):
    axis_name: str
    bucket_count: int
    bucket_boundaries: list[int]   # serializable to JSON


class PlanStatsDict(TypedDict):
    axes: list[AxisStatsEntry]
    strategy_counts: dict[str, int]    # "Vmap" -> count, "SafeMap" -> count, etc.
    total_axes: int
    memory_warnings: list[str]
    dedup_stats: list[DedupStatsEntry]
    bucket_stats: list[BucketStatsEntry]


# Exhaustive panel name enumeration
PanelName = Literal["strategy", "cardinality", "dedup", "bucket", "memory", "reasoning"]

_VALID_PANELS: frozenset[str] = frozenset(PanelName.__args__)  # type: ignore[attr-defined]


# Logger protocol — structural; xtrax never imports wandb or tensorboard
class PlanLogger(Protocol):
    def log_figure(
        self,
        figure: bytes | str,   # bytes for PNG/SVG; str for HTML
        fmt: str,              # "png" | "svg" | "html"
        step: int | None = None,
    ) -> None: ...
```

---

## Module Contracts

### `src/xtrax/eda/stats.py`

- **No eda extras required.** Importable from xtrax core without pandas, matplotlib, or seaborn.
- No imports of `xtrax.eda.viz` or `xtrax.eda.export` (one-way dep: viz imports stats, never reverse).

```python
from xtrax.tiling.plan import BatchPlan
from xtrax.eda.types import PlanStatsDict, DedupStatsEntry, BucketStatsEntry

def extract_plan_stats(plan: BatchPlan) -> PlanStatsDict:
    """Extract structured stats from a BatchPlan. No eda extras required."""
    ...

def analyze_dedup(decision: object) -> DedupStatsEntry:
    """Analyze deduplication characteristics of a single axis decision."""
    ...

def analyze_bucket(decision: object) -> BucketStatsEntry:
    """Analyze bucket boundaries of a single axis decision."""
    ...
```

### `src/xtrax/eda/explain.py`

- **No eda extras required.**
- Richer reasoning extraction; populates `reasoning` fields in `AxisStatsEntry`.

```python
from xtrax.tiling.plan import BatchPlan
from xtrax.eda.types import PlanStatsDict

def explain_plan(plan: BatchPlan) -> PlanStatsDict:
    """Full plan stats with reasoning text extracted from AxisDecision.reasoning."""
    ...
```

### `src/xtrax/eda/export.py`

- **Requires eda extras.** Raises `ImportError` with `pip install xtrax[eda]` message if pandas absent.
- Accepts `PlanStatsDict` (not `BatchPlan`) — pandas conversion decoupled from stats extraction.

```python
import pandas as pd
from xtrax.eda.types import PlanStatsDict

def plan_to_dataframe(stats: PlanStatsDict) -> pd.DataFrame:
    """Convert PlanStatsDict to a pandas DataFrame. Requires xtrax[eda]."""
    ...
```

### `src/xtrax/eda/viz.py`

- **Requires eda extras.** Module-level guard raises `ImportError` on import if seaborn absent.
- Uses matplotlib `Agg` backend for headless rendering.
- HTML note: `fmt="html"` embeds a matplotlib SVG in a minimal HTML template. If a plotly
  backend is added in future, `fmt="html"` will produce interactive plotly HTML. This is a
  documented semantic change, not an API signature breakage. Callers that parse or embed the
  raw HTML string must treat its internal structure as implementation-defined.

```python
from __future__ import annotations
from pathlib import Path
from typing import Callable, Literal

import matplotlib
matplotlib.use("Agg")  # headless

try:
    import seaborn  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "xtrax.eda.viz requires visualization extras. "
        "Install with: pip install xtrax[eda]"
    ) from exc

from xtrax.tiling.plan import BatchPlan
from xtrax.eda.types import PlanStatsDict, PlanLogger, PanelName, _VALID_PANELS


def render(
    plan: BatchPlan,
    view: str = "dashboard",
    fmt: Literal["png", "svg", "html"] = "png",
    path: str | Path | None = None,
    stats_transform: Callable[[PlanStatsDict], PlanStatsDict] | None = None,
    metadata: bool = False,
    logger: PlanLogger | None = None,
    step: int | None = None,
    panels: set[PanelName] | None = None,
) -> bytes | str | None:
    """
    Render a BatchPlan to PNG (bytes), SVG (bytes), or HTML (str).

    Return:
      fmt="png"/"svg", path=None -> bytes
      fmt="html",      path=None -> str
      path provided (any fmt)   -> None (written to disk)

    Raises:
      ValueError: metadata=True and path=None
      ValueError: unknown panel name in panels
      ImportError: (at import) if seaborn/matplotlib not installed
    """
    if metadata and path is None:
        raise ValueError("metadata=True requires path to be set")
    if panels is not None:
        unknown = panels - _VALID_PANELS
        if unknown:
            raise ValueError(
                f"Unknown panel(s): {unknown!r}. "
                f"Valid panels: {sorted(_VALID_PANELS)}"
            )
    ...
```

**Return type dispatch:**

| `fmt`    | `path=None` return | `path` provided return |
|----------|--------------------|------------------------|
| `"png"`  | `bytes` (PNG)      | `None`                 |
| `"svg"`  | `bytes` (SVG)      | `None`                 |
| `"html"` | `str` (HTML)       | `None`                 |

**Logger call:** `logger.log_figure(figure=result, fmt=fmt, step=step)` — `result` is `bytes` for PNG/SVG, `str` for HTML.

**Empty plan guard:** When `plan.decisions` is empty, return a non-empty placeholder figure without raising (guard all seaborn calls against empty DataFrames).

**Sidecar:** When `metadata=True` + `path` is set, writes `{path.stem}.json` after the primary artifact. Not atomic — consumers should check file existence before reading sidecar.

---

## Acceptance Criteria (12 total)

### Happy path

1. `render(plan, fmt="png")` returns `bytes` (PNG magic header `b'\x89PNG\r\n\x1a\n'`)
2. `render(plan, fmt="svg")` returns `bytes` (starts with `b"<svg"` or `b"<?xml"`)
3. `render(plan, fmt="html")` returns `str` containing `"<svg"`
4. `extract_plan_stats(plan)` importable and callable with no eda extras installed
5. `render(plan, stats_transform=fn)` applies `fn` to `PlanStatsDict` before rendering
6. `render(plan, path="out.png", metadata=True)` writes `out.png` + `out.json` (valid JSON)
7. `render(plan, logger=mock_logger, step=0)` calls `mock_logger.log_figure(figure, fmt="png", step=0)` where `figure: bytes`
8. `render(plan, panels={"strategy"})` omits cardinality panel from output

### Unhappy path (post-critic)

9. `render(BatchPlan(decisions=()))` returns non-empty `bytes`/`str` without raising
10. `render(plan, metadata=True, path=None)` raises `ValueError` with `"metadata=True requires path"`
11. `render(plan, panels={"invalid_panel"})` raises `ValueError` naming the bad panel and listing valid ones
12. `import xtrax.eda.viz` with seaborn absent raises `ImportError` with `"pip install xtrax[eda]"`

---

## Decision Log

| Option | Verdict | Rationale |
|---|---|---|
| VIZ-LIB-A: seaborn-only | **ACCEPT** | Polished plots, headless PNG/SVG via Agg, SVG-in-template HTML, single dep chain |
| VIZ-LIB-B: plotly-only | REJECT | kaleido breaks headless Docker CI (no chromium) — fixed constraint violation |
| VIZ-LIB-C: dual seaborn+plotly | REJECT | 2x maintenance; unjustified at v0.1 |
| VIZ-LIB-D: matplotlib-only | runner-up | Simpler dep chain; viable fallback if seaborn causes conflicts |
| OUTPUT-API-A: single `render()` | **ACCEPT** | Clean entry point, extensible, easy to mock |
| OUTPUT-API-B: format-specific fns | REJECT | Code duplication; awkward for format-agnostic pipelines |
| OUTPUT-API-C: `PlanFigure` class | REJECT | `.show()` env-detection leak; deferred |
| TRANSFORM-B: post-stats hook | **ACCEPT** | Pure function on stdlib+numpy TypedDict; no backend leak |
| TRANSFORM-C: post-render hook | REJECT | Leaks matplotlib/plotly type into user code |
| METADATA-B: JSON sidecar | **ACCEPT** | Format-agnostic; works for all three output formats |
| DASHBOARD-A: fixed layout + panel filter | **ACCEPT** | Predictable; testable; 80% case |
| DASHBOARD-B: composable registry | DEFER v0.2 | Overkill for v0.1 |
| INTEGRATION-A: `PlanLogger` protocol | **ACCEPT** | Structural protocol; xtrax never imports wandb/tb |
| STATS-LAYER: pure stats/viz separation | **ACCEPT** | `stats.py` importable without eda extras |
| ASSUMPTION-REVERSAL: `explain_plan()` primary | **ACCEPT** | Stable API surface is stats, not rendering |

### Critic resolutions

| Critique | Severity | Resolution |
|---|---|---|
| CRITIQUE-RETURN-TYPE: `bytes\|None` inconsistent with HTML→`str` | **FATAL** | Return type → `bytes \| str \| None`; explicit dispatch by `fmt` |
| CRITIQUE-LOGGER: `log_figure(bytes)` breaks on HTML | **FATAL** | Protocol → `log_figure(figure: bytes \| str, fmt: str, step: int \| None)` |
| CRITIQUE-TRANSFORM: no key schema for stats dict | MAJOR | `PlanStatsDict` TypedDict defined; transform typed `Callable[[PlanStatsDict], PlanStatsDict]` |
| CRITIQUE-SIDECAR: `metadata=True, path=None` unspecified | MAJOR | Raises `ValueError("metadata=True requires path to be set")` |
| CRITIQUE-STATS-LAYER: `extract_plan_stats()` return type unspecified | MAJOR | Explicitly typed `-> PlanStatsDict`; TypedDict is stdlib+numpy only |
| CRITIQUE-DASHBOARD: `panels=` is free-form string set | MAJOR | `PanelName = Literal[...]`; unknown panel raises `ValueError` |
| CRITIQUE-VIZ-LIB: reversibility claim misleading | MAJOR | Caveat added: `fmt="html"` semantics are implementation-defined across backends |
| CRITIQUE-ACCEPTANCE: empty `BatchPlan` not covered | MAJOR | Criterion 9 added; empty-plan guard required in impl |
| CRITIQUE-SIDECAR-CONCURRENCY: non-atomic sidecar write | MINOR | Documented; not fixed at API level |

---

## Deferred

- **DASHBOARD-B** (composable panel registry): v0.2 after adoption confirmed
- **`PlanFigure` abstraction**: revisit if notebook `.show()` demand surfaces
- **Atomic sidecar writes**: implementation discretion; document write order
- **plotly HTML backend**: future addition; `fmt="html"` semantics documented as backend-defined
- **tensorboard/wandb first-party adapters**: users provide thin `PlanLogger` adapters; not shipped in xtrax

---

## Pre-mortem Record

The design failed because seaborn's HTML output (SVG-in-template) was not interactive.
Users who expected tooltips wrote their own plotly wrappers. Over 6 months, three projects
independently wrote a plotly adapter for `PlanStatsDict`, and none used `render()`. The
stable part — `explain_plan()` and `PlanStatsDict` — was used everywhere.

Mitigation already in winner: `PlanStats`-first design makes the stable API surface correct.
`render()` is documented as a convenience function with no interactivity guarantee. The fix
for user-side interactivity is to pass `PlanStatsDict` directly to plotly — no xtrax API
change required.

---

## Amendments (post-adversarial review)

The following amendments resolve findings from the spec-challenger / spec-defender cycle.
All are targeted additions — no prior decisions are reversed.

### AMD-1: Dependency declaration syntax (BLOCKER — challenger C1)

The spec previously specified `[dependency-groups]` (PEP 735 / uv-native). This table does
not populate pip's bracket-extra syntax. Use `[project.optional-dependencies]` for pip
compatibility. Projects using uv may additionally mirror to `[dependency-groups]` for
`uv sync --group eda` support.

```toml
# pyproject.toml — pip-compatible extras (required)
[project.optional-dependencies]
eda = ["pandas>=2.0", "matplotlib>=3.8", "seaborn>=0.13"]

# Optional: uv-native group for `uv sync --group eda`
[dependency-groups]
eda = ["pandas>=2.0", "matplotlib>=3.8", "seaborn>=0.13"]
```

Install: `pip install xtrax[eda]` or `uv sync --group eda`

### AMD-2: Python version floor (BLOCKER — challenger C2)

Constraint 4 previously stated "Python 3.10+". The repository sets `requires-python = ">=3.13"`
and targets py313 in ruff. Correct floor: **Python 3.13+**. All type annotations may use 3.13
syntax (PEP 695, `type` statement) if desired; `from __future__ import annotations` is optional.

### AMD-3: `xtrax/eda/__init__.py` contract (BLOCKER — conceded by defender)

The `__init__.py` contract is load-bearing for the no-extras guarantee. A naive eager import
defeats the per-module guards. Required structure:

```python
# src/xtrax/eda/__init__.py
# Lazy-import contract: only stats/explain/types are imported at module level.
# viz and export are deferred — importing xtrax.eda without eda extras must succeed.

from xtrax.eda.stats import extract_plan_stats, analyze_dedup, analyze_bucket
from xtrax.eda.explain import explain_plan
from xtrax.eda.types import PlanStatsDict, PlanLogger, PanelName

def render(*args, **kwargs):
    """Lazy wrapper — defers viz import until first call."""
    from xtrax.eda.viz import render as _render
    return _render(*args, **kwargs)

def plan_to_dataframe(*args, **kwargs):
    """Lazy wrapper — defers export import until first call."""
    from xtrax.eda.export import plan_to_dataframe as _ptdf
    return _ptdf(*args, **kwargs)

__all__ = [
    "extract_plan_stats", "analyze_dedup", "analyze_bucket",
    "explain_plan", "render", "plan_to_dataframe",
    "PlanStatsDict", "PlanLogger", "PanelName",
]
```

### AMD-4: `analyze_dedup` / `analyze_bucket` parameter type (challenger C5, defender partial)

`analyze_dedup(decision: object)` and `analyze_bucket(decision: object)` are now typed as
`AxisDecision` (from `xtrax.tiling.plan`). Wrong-strategy behavior is specified:

- `analyze_dedup(decision)` where `decision.strategy` is not `DedupGather` → raises
  `TypeError(f"analyze_dedup requires DedupGather strategy; got {type(decision.strategy).__name__}")`
- `analyze_bucket(decision)` where `decision.strategy` is not `Bucket` → raises
  `TypeError(f"analyze_bucket requires Bucket strategy; got {type(decision.strategy).__name__}")`

Field sourcing for `DedupStatsEntry`:
- `unique_count` ← `DedupGather.k`
- `padded_count` ← `DedupGather.k_bucket`
- `total_count` ← `AxisDecision.spec.cardinality`
- `padding_waste` ← `k_bucket - k`
- `dedup_ratio` ← `1.0 - (k / spec.cardinality)` (0.0 if `spec.cardinality == 0`)

### AMD-5: Criterion 8 falsifiability seam (conceded by defender)

`render()` must embed `data-panel` attributes on SVG group elements so panel presence is
falsifiable without image parsing:

```xml
<g data-panel="strategy">...</g>
<g data-panel="cardinality">...</g>
```

**Amended Criterion 8:** `render(plan, panels={"strategy"})` returns SVG/HTML bytes/str
that contains `data-panel="strategy"` and does NOT contain `data-panel="cardinality"`.
Tests may assert via string search on the returned bytes/str (decode to str for PNG fallback
is not required — PNG tests use Criterion 8 only with SVG or HTML output).

### AMD-6: `stats_transform` output validation (conceded by defender)

`TypedDict` has no runtime enforcement. A transform returning malformed output currently
causes opaque errors inside seaborn calls. Required behavior:

After calling `stats_transform(stats)`, `render()` must validate the return value has the
required top-level keys (`axes`, `strategy_counts`, `total_axes`, `memory_warnings`,
`dedup_stats`, `bucket_stats`). If any required key is missing:

```python
raise TypeError(
    f"stats_transform must return PlanStatsDict with all required keys; "
    f"missing: {missing_keys!r}"
)
```

Value-type validation within each key is not required (too expensive for inner lists).
Key-presence validation is required and must fire before any seaborn call.

### AMD-7: Criterion 5 observability (defender partial)

Amended Criterion 5 is now testable via `strategy_counts` → bar chart data path:

**Amended Criterion 5:** `render(plan, fmt="svg", stats_transform=fn)` where `fn` modifies
`strategy_counts` (e.g., `fn = lambda s: {**s, "strategy_counts": {"Vmap": 99}}`) must
produce SVG output containing a bar or label reflecting the mutated value (e.g., a bar with
height proportional to 99, or a text label "99"). The strategy panel uses `strategy_counts`
directly as its data source — this contract must be documented in `viz.py`.

