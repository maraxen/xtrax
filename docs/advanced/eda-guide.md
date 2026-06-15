# EDA Guide

`xtrax.eda` provides a lightweight exploratory data analysis layer built on top of
`BatchPlan` — the output of `BatchPlanner.plan()`. It answers: *why did this axis get
this strategy?* and *what does the plan look like across all axes?*

## Installation

```bash
pip install xtrax[eda]         # full: seaborn, matplotlib, pandas
pip install xtrax              # core only: stats/explain still work
```

## Stats without extras

The stats and explain functions work with stdlib + numpy only — no extras required:

```python
from xtrax.tiling.plan import BatchPlanner, AxisSpec
from xtrax.eda import extract_plan_stats, explain_plan

specs = [
    AxisSpec("batch", cardinality=128, batch_size=32),
    AxisSpec("seq",   cardinality=512, batch_size=64,
             bucket_boundaries=(128, 256, 512)),
]
plan = BatchPlanner().plan(specs)

stats = extract_plan_stats(plan)
# stats["strategy_counts"]  -> {"SafeMap": 1, "Bucket": 1}
# stats["total_axes"]       -> 2
# stats["axes"][0]          -> {"name": "batch", "strategy": "SafeMap", ...}

# explain_plan guarantees non-empty reasoning strings
rich = explain_plan(plan)
for axis in rich["axes"]:
    print(f"{axis['name']}: {axis['reasoning']}")
```

## Rendering (requires extras)

```python
from xtrax.eda import render

# PNG to bytes (e.g. for notebook display)
img = render(plan, fmt="png")

# SVG to disk
render(plan, fmt="svg", path="plan.svg")

# Interactive-ready HTML (static SVG-in-HTML; see note below)
render(plan, fmt="html", path="plan.html")
```

```{note}
`fmt="html"` currently embeds a static matplotlib SVG in a minimal HTML template.
For interactive tooltips, pass `PlanStatsDict` directly to plotly — `render()` is
a convenience function, not a stability anchor for HTML semantics.
```

## Filtering panels

Six named panels are available:

```python
from xtrax.eda import render

# Show only the strategy distribution and cardinality scatter
render(plan, fmt="svg", panels={"strategy", "cardinality"})
```

Valid panel names: `"strategy"`, `"cardinality"`, `"dedup"`, `"bucket"`,
`"memory"`, `"reasoning"`.

## Transformation hook

Post-stats transforms let you annotate or filter before rendering:

```python
def highlight_large_axes(stats):
    stats = dict(stats)
    stats["axes"] = [
        {**ax, "reasoning": f"[LARGE] {ax['reasoning']}"}
        if ax["cardinality"] > 256 else ax
        for ax in stats["axes"]
    ]
    return stats

render(plan, stats_transform=highlight_large_axes, fmt="png")
```

The transform receives and must return a `PlanStatsDict`. Missing required keys
raise `TypeError` before any seaborn call.

## Metadata sidecar

```python
render(plan, path="plan.png", metadata=True)
# writes plan.png + plan.json (full PlanStatsDict as JSON)
```

`metadata=True` requires `path` to be set — raises `ValueError` otherwise.

## Logger integration (wandb / tensorboard)

`xtrax` never imports wandb or tensorboard. Provide a thin adapter:

```python
import wandb
from xtrax.eda.types import PlanLogger

class WandbLogger:
    def log_figure(self, figure: bytes | str, fmt: str, step: int | None = None):
        import io, PIL.Image
        if fmt == "png":
            img = PIL.Image.open(io.BytesIO(figure))
            wandb.log({"plan": wandb.Image(img)}, step=step)
        elif fmt == "html":
            wandb.log({"plan_html": wandb.Html(figure)}, step=step)

render(plan, logger=WandbLogger(), step=epoch)
```

## DataFrame export

```python
from xtrax.eda import extract_plan_stats, plan_to_dataframe

stats = extract_plan_stats(plan)
df = plan_to_dataframe(stats)
# one row per axis; columns: name, strategy, cardinality, batch_size, reasoning, ...
print(df.to_string())
```

## Dedup and bucket analysis

```python
from xtrax.eda.stats import analyze_dedup, analyze_bucket
from xtrax.tiling.strategy import DedupGather, Bucket

for decision in plan.decisions:
    if isinstance(decision.strategy, DedupGather):
        entry = analyze_dedup(decision)
        print(f"{decision.spec.name}: {entry['dedup_ratio']:.1%} redundancy, "
              f"padding waste = {entry['padding_waste']}")
    if isinstance(decision.strategy, Bucket):
        entry = analyze_bucket(decision)
        print(f"{decision.spec.name}: {entry['bucket_count']} buckets "
              f"at {entry['bucket_boundaries']}")
```
