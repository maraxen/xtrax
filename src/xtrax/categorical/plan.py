"""Tile-plan record for ``cooccurrence_counts``.

Declaration only. The planner (``plan_cooccurrence``) and the byte-budget
search land in task 1.5; until then callers cannot pass a plan.

Spec: .praxia/docs/specs/261015_exact-categorical-cooccurrence.md (§3, §5)
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CooccurrencePlan:
    """Tile sizes that a cooccurrence_counts call executes.

    Attributes:
        method: Method the plan runs, ``"matmul"`` or ``"scatter"``.
        r_tile: Replicates per ``lax.map`` step.
        t_chunk: Frames per ``lax.scan`` step.
        row_block: Variable rows of ``codes_a`` per block (dense layout).
        pair_tile: Pairs per ``lax.map`` step (pairs layout, scatter).
        r_padded: Replicate count after zero-weight padding.
        t_padded: Frame count after zero-weight padding.
        estimate_bytes: Planner's peak-memory estimate in bytes.
        budget_bytes: Byte budget the plan was sized against, or None.
    """

    method: Literal["matmul", "scatter"]
    r_tile: int
    t_chunk: int
    row_block: int
    pair_tile: int
    r_padded: int
    t_padded: int
    estimate_bytes: int
    budget_bytes: int | None
