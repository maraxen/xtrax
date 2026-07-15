> Part of the `using-xtrax` skill (`agent_assets/skills/using-xtrax/SKILL.md`) — TIER-2 deep reference.

# EDA: Plan Analysis and Visualization (10% of depth)

#### extract_plan_stats: Structured Analysis

Extract statistics from a `BatchPlan`:

```python
from xtrax.eda.stats import extract_plan_stats  # verify: src/xtrax/eda/stats.py

plan = planner.plan([spec1, spec2, ...])
stats = extract_plan_stats(plan)

# stats is a dict[str, Any] with:
# {
#   "axes": [
#       {"name": "batch", "cardinality": 100, "batch_size": 32, "reasoning": "...", ...},  # verify: src/xtrax/eda/types.py
#       ...
#   ],
#   ...
# }
```

Verify: `src/xtrax/eda/stats.py`

#### explain_plan: Guaranteed Non-Empty Reasoning

Wrapper around `extract_plan_stats` ensuring all reasoning fields are non-empty:

```python
from xtrax.eda.explain import explain_plan

stats = explain_plan(plan)
# All stats["axes"][i]["reasoning"] are guaranteed non-empty strings
```

Verify: `src/xtrax/eda/explain.py:14-41`

#### EDA-as-Planning-Audit Workflow

Before committing to a batching strategy, audit the plan:

```python
# Step 1: Build plan
plan = planner.plan([spec1, spec2, ...])

# Step 2: Inspect reasoning
stats = explain_plan(plan)
for axis in stats["axes"]:
    print(f"{axis['name']}: {axis['strategy']} ({axis['reasoning']})")

# Step 3: Visualize (if xtrax[eda] installed)
from xtrax.eda import render  # implemented in viz.py, re-exported via eda/__init__.py

html = render(plan)
with open("plan.html", "w") as f:
    f.write(html)
```

**Benefit**: Catch suboptimal strategy choices (e.g., SafeMap when Vmap would fit) before first JIT compilation.

#### analyze_dedup, analyze_bucket

Per-axis statistics:

```python
from xtrax.eda.stats import analyze_dedup, analyze_bucket

# Dedup analysis
dedup_stats = analyze_dedup(decision)  # For DedupGather decisions

# Bucket analysis
bucket_stats = analyze_bucket(decision)  # For Bucket decisions
```

Verify: `src/xtrax/eda/stats.py`

#### render: HTML Visualization

Generate interactive HTML plan visualization:

```python
from xtrax.eda import render  # implemented in viz.py, re-exported via eda/__init__.py

html = render(plan)
# html is a string of HTML
```

⚠ WARN: `render()` requires `pip install xtrax[eda]` (extras).  
Import is lazy — no error at module load time, but `render()` call will fail if extras not installed.

#### plan_to_dataframe: Pandas Export

Export plan stats to a pandas DataFrame:

```python
from xtrax.eda import plan_to_dataframe  # lazy re-export from eda/export.py, same pattern as render()

df = plan_to_dataframe(plan)
# DataFrame with columns: name, cardinality, strategy, batch_size, reasoning, ...
```

⚠ WARN: `plan_to_dataframe()` requires `pip install xtrax[eda]` (extras, same as `render()`).

Verify: `src/xtrax/eda/export.py:23` (implementation); `src/xtrax/eda/__init__.py:37-44` (lazy re-export wrapper). `from xtrax.eda.stats import plan_to_dataframe` does NOT work — `stats.py` does not define or import this symbol.
