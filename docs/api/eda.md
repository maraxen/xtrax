# EDA

```{note}
The EDA subpackage requires optional extras: `pip install xtrax[eda]`.
The stats and explain functions work without any extras installed.
```

## Overview

`xtrax.eda` provides exploratory data analysis primitives for inspecting and
visualizing `BatchPlan` outputs from the tiling subsystem. The design follows a
two-layer architecture:

- **Stats layer** (`xtrax.eda.stats`, `xtrax.eda.explain`) — stdlib + numpy only,
  importable from the xtrax core without extras.
- **Viz layer** (`xtrax.eda.viz`) — seaborn + matplotlib backend, requires
  `pip install xtrax[eda]`.

## Stats layer

```{automodule} xtrax.eda.stats
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} xtrax.eda.explain
:members:
:undoc-members:
:show-inheritance:
```

## Types

```{automodule} xtrax.eda.types
:members:
:undoc-members:
:show-inheritance:
```

## Visualization

Requires `pip install xtrax[eda]`.

```{automodule} xtrax.eda.viz
:members:
:undoc-members:
:show-inheritance:
```

## Export

Requires `pip install xtrax[eda]`.

```{automodule} xtrax.eda.export
:members:
:undoc-members:
:show-inheritance:
```
