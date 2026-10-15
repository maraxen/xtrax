"""Tile planner for ``cooccurrence_counts``.

``plan_cooccurrence`` is the only planner (spec §5, review M4). It returns exactly
the tile sizes that the kernel executes, so a recorded plan cannot diverge from
the run. It does not call ``BatchPlanner``, whose ``Vmap`` means full cardinality
(spec A34).

Spec: .praxia/docs/specs/261015_exact-categorical-cooccurrence.md (§3, §4.4, §5)
"""

import math
from dataclasses import dataclass
from typing import Literal

import jax

from xtrax.tiling.budget import BudgetInfeasibleError

# Method-selection crossovers (spec §4.4). The CPU value is provisional until the
# bathos-registered probe (task 1.6) re-derives it.
KK_CROSSOVER_CPU = 64
KK_CROSSOVER_GPU = 16

# Unbudgeted defaults (spec §5, byte_budget=None).
FRAME_CHUNK = 1024
PAIR_TILE = 256

# Budgeted search candidates (spec §5 steps 2 and 3).
_T_CANDIDATES = (4096, 2048, 1024, 512, 256, 128)
_SEARCH_T_CHUNK = 128
_PAIR_TILE_MAX = 4096
# Fixed overhead term of the estimate (spec §5).
_OVERHEAD_BYTES = 4096


@dataclass(frozen=True)
class CooccurrencePlan:
    """Tile sizes that a cooccurrence_counts call executes.

    Attributes:
        method: Method the plan runs, ``"matmul"`` or ``"scatter"``.
        r_tile: Replicates per ``lax.map`` step (its ``batch_size``).
        t_chunk: Frames per ``lax.scan`` step.
        row_block: Variable rows of ``codes_a`` per block. Pairs layout is not
            row-blocked, so there it equals ``Pa``.
        pair_tile: Pairs per ``lax.map`` step (scatter).
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


def select_method(
    method: Literal["auto", "matmul", "scatter"],
    layout: Literal["dense", "pairs"],
    n_pairs: int,
    n_dense_pairs: int,
    ka: int,
    kb: int,
    platform: str,
) -> tuple[str, str]:
    """Resolve ``method="auto"`` to a concrete method and its provenance rule.

    Spec §4.4. A pairs layout with fewer pairs than the dense ``Pa*Pb`` grid
    prefers ``scatter``, because ``matmul`` would compute the whole dense grid
    and gather rows. The spec's "<<" is unquantified, so ``n_pairs < Pa*Pb`` is
    the literal reading used here. Otherwise the platform crossover applies.

    Args:
        method: Caller's choice.
        layout: ``"dense"`` or ``"pairs"``.
        n_pairs: Number of pairs computed.
        n_dense_pairs: ``Pa*Pb``, the size of the dense grid.
        ka: Categories of ``a``.
        kb: Categories of ``b``.
        platform: ``jax.default_backend()``.

    Returns:
        ``(method, method_rule)``.

    Raises:
        ValueError: If ``method`` is not one of the three allowed values.
    """
    if method not in ("auto", "matmul", "scatter"):
        raise ValueError(f"method must be 'auto', 'matmul' or 'scatter', got {method!r}")
    if method != "auto":
        return method, "explicit"
    if layout == "pairs" and n_pairs < n_dense_pairs:
        rule = "cpu_provisional" if platform == "cpu" else "gpu_provisional"
        return "scatter", rule
    if platform == "cpu":
        chosen = "matmul" if ka * kb <= KK_CROSSOVER_CPU else "scatter"
        return chosen, "cpu_provisional"
    dense_small = layout == "dense" and ka * kb <= KK_CROSSOVER_GPU
    return ("matmul" if dense_small else "scatter"), "gpu_provisional"


def per_tile_bytes(
    *,
    method: Literal["matmul", "scatter"],
    n_a: int,
    k_a: int,
    n_b: int,
    k_b: int,
    row_block: int,
    t_chunk: int,
    r_tile: int,
    pair_tile: int,
    weight_itemsize: int,
) -> int:
    """Return the ``per_tile`` term of the spec §5 estimate (one scan step's temporaries).

    Args:
        method: ``"matmul"`` or ``"scatter"``.
        n_a: Variables ``Pa``.
        k_a: Categories ``Ka``.
        n_b: Variables ``Pb``.
        k_b: Categories ``Kb``.
        row_block: Variable rows per block.
        t_chunk: Frames per scan step.
        r_tile: Replicates per map step.
        pair_tile: Pairs per map step (scatter).
        weight_itemsize: Bytes per weight element.

    Returns:
        Bytes of per-step temporaries.
    """
    wb = weight_itemsize
    if method == "matmul":
        fa_blk = row_block * k_a
        fb = n_b * k_b
        return (
            r_tile * (t_chunk * (fa_blk + fb) * 4 + 2 * fa_blk * fb * 4 + t_chunk * wb)
            + t_chunk * (row_block + n_b) * 4
        )
    kk = k_a * k_b
    return r_tile * pair_tile * (2 * kk * 4 + 2 * t_chunk * 4) + t_chunk * (n_a + n_b) * 4


def estimate_cooccurrence_bytes(
    *,
    method: Literal["matmul", "scatter"],
    n_replicates: int,
    n_frames: int,
    n_a: int,
    k_a: int,
    n_b: int,
    k_b: int,
    row_block: int,
    n_pairs_blk: int,
    t_chunk: int,
    r_tile: int,
    pair_tile: int,
    weight_itemsize: int,
    code_itemsize: int,
    differentiable: bool = False,
    layout: Literal["dense", "pairs"] = "dense",
) -> int:
    """Estimate peak temporary plus resident bytes of one call (spec §5).

    Args:
        method: ``"matmul"`` or ``"scatter"``.
        n_replicates: Replicates ``R`` before padding.
        n_frames: Frames ``T`` before padding.
        n_a: Variables ``Pa``.
        k_a: Categories ``Ka``.
        n_b: Variables ``Pb``.
        k_b: Categories ``Kb``.
        row_block: Variable rows per block.
        n_pairs_blk: Pairs computed per row block.
        t_chunk: Frames per scan step.
        r_tile: Replicates per map step.
        pair_tile: Pairs per map step (scatter).
        weight_itemsize: Bytes per weight element.
        code_itemsize: Bytes per code element.
        differentiable: Add the backward-pass gather temporaries (spec §5):
            ``r_tile*row_block*Pb*t_chunk*4`` for the dense layout, or
            ``r_tile*pair_tile*t_chunk*4`` for the pairs layout.
        layout: ``"dense"`` or ``"pairs"``; selects the backward term.

    Returns:
        Estimated bytes, as in spec §5.
    """
    r_pad = -(-n_replicates // r_tile) * r_tile
    t_pad = -(-n_frames // t_chunk) * t_chunk
    wb, cb, ob = weight_itemsize, code_itemsize, 4
    inputs = (n_a + n_b) * t_pad * cb + r_pad * t_pad * wb + r_pad * (n_a + n_b) * 4
    if method == "matmul":
        fa_blk = row_block * k_a
        fb = n_b * k_b
        output = r_pad * (fa_blk * fb + fa_blk + fb + 1) * ob
    else:
        kk = k_a * k_b
        output = r_pad * (n_pairs_blk * kk + row_block * k_a + n_b * k_b + 1) * 4
    per_tile = per_tile_bytes(
        method=method,
        n_a=n_a,
        k_a=k_a,
        n_b=n_b,
        k_b=k_b,
        row_block=row_block,
        t_chunk=t_chunk,
        r_tile=r_tile,
        pair_tile=pair_tile,
        weight_itemsize=wb,
    )
    backward = 0
    if differentiable:
        joint_entries = row_block * n_b if layout == "dense" else pair_tile
        backward = r_tile * joint_entries * t_chunk * 4
    return int(inputs + output + per_tile + backward + _OVERHEAD_BYTES)


def device_budget_bytes(fraction: float = 0.5) -> int:
    """Return ``fraction`` of the first device's allocator limit (spec §5, A35).

    Mirrors ``xtrax.tiling.estimators.device_memory_budget`` without importing it,
    because ``xtrax.categorical`` may import only ``xtrax.tiling.budget``.

    Args:
        fraction: Share of ``bytes_limit`` to budget, in ``(0, 1]``.

    Returns:
        Budget in bytes.

    Raises:
        ValueError: If ``fraction`` is outside ``(0, 1]``.
        RuntimeError: If the device reports no ``bytes_limit``. Pass an int instead.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    stats = jax.devices()[0].memory_stats()
    if not stats or "bytes_limit" not in stats:
        raise RuntimeError(
            "device does not report memory_stats()['bytes_limit']; "
            "pass an explicit integer byte_budget instead"
        )
    return int(stats["bytes_limit"] * fraction)


def _resolve_budget(byte_budget: int | Literal["device"]) -> int:
    """Turn ``byte_budget`` into bytes; ``"device"`` queries the allocator."""
    if byte_budget == "device":
        return device_budget_bytes(fraction=0.5)
    if isinstance(byte_budget, bool) or not isinstance(byte_budget, int):
        raise TypeError(f"byte_budget must be an int, 'device' or None, got {byte_budget!r}")
    if byte_budget <= 0:
        raise ValueError(f"byte_budget must be positive, got {byte_budget}")
    return byte_budget


def _power_of_two_candidates(limit: int) -> list[int]:
    """Return powers of two from ``limit`` down to 1, descending, unique."""
    top = 1 << max(0, int(math.log2(max(limit, 1))))
    out: list[int] = []
    c = top
    while c >= 1:
        out.append(c)
        c //= 2
    return out


def plan_cooccurrence(
    *,
    n_replicates: int,
    n_frames: int,
    n_a: int,
    k_a: int,
    n_b: int,
    k_b: int,
    n_pairs: int,
    weight_itemsize: int,
    code_itemsize: int,
    strategy: Literal["exact_int", "float"] = "exact_int",
    differentiable: bool = False,
    method: Literal["auto", "matmul", "scatter"] = "auto",
    layout: Literal["dense", "pairs"] = "dense",
    byte_budget: int | Literal["device"] | None = None,
    platform: str | None = None,
) -> CooccurrencePlan:
    """Choose the tile sizes for one ``cooccurrence_counts`` call (spec §5).

    With ``byte_budget=None`` the unbudgeted defaults apply and the device is not
    queried: ``row_block=Pa``, ``t_chunk=min(1024, T)``, ``r_tile=1``,
    ``pair_tile=min(256, pairs per block)``. With an int budget, the deterministic
    search runs in this order: (1) largest ``row_block`` at ``r_tile=1``,
    ``t_chunk=128``; (2) largest ``t_chunk`` in ``[4096 .. 128]`` clamped to ``T``;
    (3) largest power-of-two ``pair_tile <= 4096`` (scatter); (4) ``r_tile=1`` on
    CPU, else the largest fitting power of two. If nothing fits, it raises.

    The budget is soft. Denxity documents 2-3x overshoot at small tiles
    (``_tiling.py:223-229``).

    Args:
        n_replicates: Replicates ``R``.
        n_frames: Frames ``T``.
        n_a: Variables ``Pa``.
        k_a: Categories ``Ka``.
        n_b: Variables ``Pb``.
        k_b: Categories ``Kb``.
        n_pairs: Pairs computed. For the dense layout this is ``Pa*Pb``.
        weight_itemsize: Bytes per weight element.
        code_itemsize: Bytes per code element.
        strategy: Must be ``"exact_int"``. ``"float"`` runs untiled and has no plan.
        differentiable: True for the ``custom_vjp`` path (spec §4.6). Its backward
            temporaries are added to the estimate. The kernel is the same int32 one.
        method: ``"auto"``, ``"matmul"`` or ``"scatter"``.
        layout: ``"dense"`` or ``"pairs"``.
        byte_budget: None, an int of bytes, or ``"device"``.
        platform: ``jax.default_backend()`` when None.

    Returns:
        The ``CooccurrencePlan`` the kernel executes.

    Raises:
        ValueError: For ``strategy="float"``, inconsistent sizes or an invalid budget.
        TypeError: For a budget that is neither an int, ``"device"`` nor None.
        BudgetInfeasibleError: If the full result alone, or the smallest tile
            plan, exceeds ``byte_budget``.
    """
    if strategy != "exact_int":
        raise ValueError("strategy='float' runs untiled and has no tile plan")
    if layout not in ("dense", "pairs"):
        raise ValueError(f"layout must be 'dense' or 'pairs', got {layout!r}")
    n_dense = n_a * n_b
    if layout == "dense" and n_pairs != n_dense:
        raise ValueError(f"dense layout has Pa*Pb={n_dense} pairs, got n_pairs={n_pairs}")
    if n_pairs < 1 or n_replicates < 1 or n_frames < 1:
        raise ValueError("n_pairs, n_replicates and n_frames must all be >= 1")
    if platform is None:
        platform = jax.default_backend()
    chosen, _ = select_method(method, layout, n_pairs, n_dense, k_a, k_b, platform)

    def pairs_per_block(row_block: int) -> int:
        return n_pairs if layout == "pairs" else row_block * n_b

    def estimate(row_block: int, t_chunk: int, pair_tile: int, r_tile: int) -> int:
        return estimate_cooccurrence_bytes(
            method=chosen,
            n_replicates=n_replicates,
            n_frames=n_frames,
            n_a=n_a,
            k_a=k_a,
            n_b=n_b,
            k_b=k_b,
            row_block=row_block,
            n_pairs_blk=pairs_per_block(row_block),
            t_chunk=t_chunk,
            r_tile=r_tile,
            pair_tile=pair_tile,
            weight_itemsize=weight_itemsize,
            code_itemsize=code_itemsize,
            differentiable=differentiable,
            layout=layout,
        )

    def build(row_block: int, t_chunk: int, pair_tile: int, r_tile: int, budget: int | None):
        return CooccurrencePlan(
            method=chosen,
            r_tile=r_tile,
            t_chunk=t_chunk,
            row_block=row_block,
            pair_tile=pair_tile,
            r_padded=-(-n_replicates // r_tile) * r_tile,
            t_padded=-(-n_frames // t_chunk) * t_chunk,
            estimate_bytes=estimate(row_block, t_chunk, pair_tile, r_tile),
            budget_bytes=budget,
        )

    if byte_budget is None:
        t_chunk = min(FRAME_CHUNK, n_frames)
        pair_tile = min(PAIR_TILE, pairs_per_block(n_a))
        return build(n_a, t_chunk, pair_tile, 1, None)

    budget = _resolve_budget(byte_budget)
    output_alone = n_replicates * (n_pairs * k_a * k_b + n_a * k_a + n_b * k_b + 1) * 4
    if output_alone > budget:
        raise BudgetInfeasibleError(
            f"the full result alone needs {output_alone} bytes, over byte_budget={budget}; "
            "split the variables with iter_cooccurrence_row_blocks"
        )

    t_search = min(_SEARCH_T_CHUNK, n_frames)
    t_candidates = sorted({min(c, n_frames) for c in _T_CANDIDATES}, reverse=True)

    def default_pair_tile(row_block: int) -> int:
        return min(PAIR_TILE, pairs_per_block(row_block))

    # (1) Largest row_block at r_tile=1, t_chunk=128.
    if layout == "dense":
        row_block = next(
            (
                rb
                for rb in range(n_a, 0, -1)
                if estimate(rb, t_search, default_pair_tile(rb), 1) <= budget
            ),
            None,
        )
    else:
        row_block = n_a
    minimal = estimate(1 if layout == "dense" else n_a, t_search, 1, 1)
    if row_block is None:
        raise BudgetInfeasibleError(
            f"no tile plan fits byte_budget={budget}: the smallest estimate is {minimal} bytes "
            "(row_block=1, t_chunk=128, pair_tile=1, r_tile=1)"
        )

    # (2) Largest t_chunk in [4096 .. 128], clamped to T.
    t_chunk = next(
        (
            c
            for c in t_candidates
            if c >= t_search and estimate(row_block, c, default_pair_tile(row_block), 1) <= budget
        ),
        None,
    )
    if t_chunk is None:
        raise BudgetInfeasibleError(
            f"no t_chunk in [{t_search}, {_T_CANDIDATES[0]}] fits byte_budget={budget} at "
            f"row_block={row_block}: the smallest estimate is {minimal} bytes"
        )

    # (3) Largest power-of-two pair_tile <= 4096 (scatter only).
    pair_tile = default_pair_tile(row_block)
    if chosen == "scatter":
        blk = pairs_per_block(row_block)
        pair_tile = next(
            (
                p
                for p in _power_of_two_candidates(min(_PAIR_TILE_MAX, blk))
                if estimate(row_block, t_chunk, p, 1) <= budget
            ),
            pair_tile,
        )

    # (4) r_tile: 1 on CPU; largest fitting power of two on GPU.
    r_tile = 1
    if platform != "cpu" and n_replicates > 1:
        r_tile = next(
            (
                r
                for r in _power_of_two_candidates(n_replicates)
                if estimate(row_block, t_chunk, pair_tile, r) <= budget
            ),
            1,
        )

    return build(row_block, t_chunk, pair_tile, r_tile, budget)
