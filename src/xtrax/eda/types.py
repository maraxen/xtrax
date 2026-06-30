"""Types for xtrax EDA visualization API.

All type definitions use only stdlib and numpy — no pandas, matplotlib, or seaborn.
This module is importable from xtrax core without eda extras.
"""

from collections.abc import Sequence
from typing import Literal, NotRequired, Protocol, TypedDict, runtime_checkable


class AxisStatsEntry(TypedDict):
    """Statistics for a single axis decision.

    Attributes:
        name: Human-readable axis name (e.g., "batch", "sequence").
        strategy: Selected strategy type as string (e.g., "Vmap",
            "SafeMap", "Scan", "DedupGather", "Bucket").
        cardinality: Number of elements along this axis.
        batch_size: Batch size used for this axis (if applicable).
        reasoning: Human-readable explanation of the decision.
        memory_estimate_bytes: Estimated memory consumption (in bytes)
            for this axis (optional).
    """

    name: str
    strategy: str
    cardinality: int
    batch_size: int
    reasoning: str
    memory_estimate_bytes: NotRequired[int | None]


class DedupStatsEntry(TypedDict):
    """Deduplication statistics for a single axis using DedupGather strategy.

    Attributes:
        axis_name: Name of the axis.
        dedup_ratio: 0.0–1.0; higher = more duplicates (more dedup benefit).
            Computed as 1 - (unique / total).
        unique_count: Number of unique elements (k, pre-padding).
        padded_count: Padded bucket size (k_bucket).
        total_count: Total elements in original axis (n).
        padding_waste: Number of padding elements inserted (k_bucket - k).
    """

    axis_name: str
    dedup_ratio: float
    unique_count: int
    padded_count: int
    total_count: int
    padding_waste: int


class BucketStatsEntry(TypedDict):
    """Bucketing statistics for a single axis using Bucket strategy.

    Attributes:
        axis_name: Name of the axis.
        bucket_count: Number of buckets (length of boundaries).
        bucket_boundaries: Sorted list of bucket boundary sizes.
    """

    axis_name: str
    bucket_count: int
    bucket_boundaries: list[int]


class PlanStatsDict(TypedDict):
    """Complete statistics extracted from a BatchPlan.

    Attributes:
        axes: List of statistics for each axis decision.
        strategy_counts: Histogram of strategy types used (e.g.,
            {"Vmap": 2, "SafeMap": 1}).
        total_axes: Total number of axes in the plan.
        memory_warnings: List of human-readable memory concerns (empty if none).
        dedup_stats: List of deduplication statistics (one per
            DedupGather axis, empty otherwise).
        bucket_stats: List of bucketing statistics (one per Bucket
            axis, empty otherwise).
    """

    axes: list[AxisStatsEntry]
    strategy_counts: dict[str, int]
    total_axes: int
    memory_warnings: list[str]
    dedup_stats: list[DedupStatsEntry]
    bucket_stats: list[BucketStatsEntry]


# Valid panel names for dashboard composition.
PanelName = Literal["strategy", "cardinality", "dedup", "bucket", "memory", "reasoning"]

# Frozenset of valid panel names (explicit enumeration to avoid runtime hacks).
_VALID_PANELS: frozenset[str] = frozenset(
    ("strategy", "cardinality", "dedup", "bucket", "memory", "reasoning")
)


@runtime_checkable
class AxisSpecLike(Protocol):
    """Structural protocol for the minimal axis-spec shape extract_plan_stats reads.

    xtrax.tiling.plan.AxisSpec satisfies this natively. Any other library's
    axis-spec type (e.g. a parallel BatchPlanner reimplementation) also
    satisfies it for free as long as it exposes `name`/`cardinality` — no
    inheritance or conversion required.
    """

    @property
    def name(self) -> str: ...

    @property
    def cardinality(self) -> int: ...


@runtime_checkable
class AxisDecisionLike(Protocol):
    """Structural protocol for the minimal axis-decision shape extract_plan_stats reads.

    xtrax.tiling.plan.AxisDecision satisfies this natively. `strategy` is
    typed as `object`, not `Any`: extract_plan_stats only ever reads
    `type(strategy).__name__` directly on it, then narrows to
    DedupGatherLike/BucketLike below (via isinstance against a
    @runtime_checkable Protocol, not a concrete class) before reading any
    strategy-specific fields -- so `object` is the sound, precise type here,
    not a type-checking escape hatch.
    """

    @property
    def spec(self) -> AxisSpecLike: ...

    @property
    def batch_size(self) -> int: ...

    @property
    def reasoning(self) -> str: ...

    @property
    def strategy(self) -> object: ...


@runtime_checkable
class BatchPlanLike(Protocol):
    """Structural protocol for the minimal batch-plan shape extract_plan_stats reads."""

    @property
    def decisions(self) -> Sequence[AxisDecisionLike]: ...


@runtime_checkable
class DedupGatherLike(Protocol):
    """Structural protocol narrowing `strategy: object` for dedup analysis.

    isinstance() against a @runtime_checkable Protocol (not the concrete
    xtrax.tiling.strategy.DedupGather class) lets any library's
    structurally-identical DedupGather-shaped strategy (e.g. a parallel
    BatchPlanner reimplementation with matching field names) be recognized
    and statically narrowed without inheriting from or importing xtrax's
    own class.
    """

    @property
    def k(self) -> int: ...

    @property
    def k_bucket(self) -> int: ...


@runtime_checkable
class BucketLike(Protocol):
    """Structural protocol narrowing `strategy: object` for bucket analysis.

    See DedupGatherLike docstring for why isinstance-against-Protocol (not
    a concrete class) is used here.
    """

    @property
    def boundaries(self) -> Sequence[int]: ...


@runtime_checkable
class PlanLogger(Protocol):
    """Structural protocol for plan visualization loggers (tensorboard, wandb, etc.).

    Any object implementing this protocol can be passed to render() to log figures
    to a remote system. Users adapt their logging framework by wrapping it with
    a thin callable.

    Example:
        class WandbLogger:
            def __init__(self, api):
                self.api = api

            def log_figure(
                self, figure: bytes | str, fmt: str,
                step: int | None = None
            ) -> None:
                # figure is bytes for PNG/SVG, str for HTML
                # fmt is "png", "svg", or "html"
                # step is optional epoch/iteration number
                if fmt == "html":
                    self.api.log({"plan_html": wandb.Html(figure)}, step=step)
                else:
                    self.api.log({"plan_image": wandb.Image(figure)}, step=step)
    """

    def log_figure(self, figure: bytes | str, fmt: str, step: int | None = None) -> None:
        """Log a rendered figure to a remote system.

        Args:
            figure: Rendered figure as bytes (PNG/SVG) or str (HTML).
            fmt: Output format ("png", "svg", or "html").
            step: Optional iteration/epoch number for log ordering.
        """
        ...


__all__ = [
    "AxisStatsEntry",
    "DedupStatsEntry",
    "BucketStatsEntry",
    "PlanStatsDict",
    "PanelName",
    "PlanLogger",
    "AxisSpecLike",
    "AxisDecisionLike",
    "BatchPlanLike",
    "DedupGatherLike",
    "BucketLike",
]
