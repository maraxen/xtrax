#!/usr/bin/env python3
"""Hit-loop timing for memoize_jaxpr cache performance baseline.

Measures the per-call overhead of cache hits on the current memo.py implementation.
Times three concurrent activities: the wrapped call (hit), build_key directly, and
_ensure_screened directly. Records the median microseconds per operation across repeats.

The output is one line: per_call_us=<median> key_us=<median> screen_us=<median>.
This is the baseline before refactoring; a later run records the after-refactor line
for comparison. No numeric threshold is set; this is a maintainability trace with no
correctness impact, and wall-clock noise on shared hardware makes a strict threshold flaky.

Usage:
    OMP_NUM_THREADS=4 uv run --extra dev python scripts/prof_memo_hit_loop.py \\
        [--calls 2000] [--repeats 5] [--n-scalars 4]
"""

from __future__ import annotations

import argparse
import logging
import statistics
import time
from typing import Any

import numpy as np

from xtrax.inference.memo import memoize_jaxpr

ROOT = __file__.split("scripts")[0].rstrip("/")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calls",
        type=int,
        default=2000,
        help="Number of calls per timing loop (default 2000)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of repeats (default 5)",
    )
    parser.add_argument(
        "--n-scalars",
        type=int,
        default=4,
        help="Number of scalar arguments (default 4)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG)

    # Define the function: f(x, *s) = x * 2.0 + sum(s)
    def f(x: Any, *s: int) -> Any:
        return x * 2.0 + sum(s)

    # Create test data
    x = np.ones((8,), np.float32)
    scalars = tuple(range(args.n_scalars))

    # Wrap with memoize_jaxpr
    wrapped = memoize_jaxpr(f)

    # One warm-up miss (cache is empty)
    wrapped(x, *scalars)

    # Collect timings for each repeat
    per_call_us_list = []
    key_us_list = []
    screen_us_list = []

    for _ in range(args.repeats):
        # Time loop (a): wrapped calls (hits)
        start = time.perf_counter()
        for _ in range(args.calls):
            wrapped(x, *scalars)
        elapsed_per_call = time.perf_counter() - start
        per_call_us = (elapsed_per_call * 1e6) / args.calls
        per_call_us_list.append(per_call_us)

        # Time loop (b): direct build_key calls
        # Get the core object from the wrapped callable
        core = wrapped._memo_core
        digest = core._ensure_screened((x, *scalars), {})
        start = time.perf_counter()
        for _ in range(args.calls):
            core.build_key(digest, (x, *scalars), {})
        elapsed_key = time.perf_counter() - start
        key_us = (elapsed_key * 1e6) / args.calls
        key_us_list.append(key_us)

        # Time loop (c): direct _ensure_screened calls
        start = time.perf_counter()
        for _ in range(args.calls):
            core._ensure_screened((x, *scalars), {})
        elapsed_screen = time.perf_counter() - start
        screen_us = (elapsed_screen * 1e6) / args.calls
        screen_us_list.append(screen_us)

    # Assert that loop (a) really hit
    stats = wrapped.memo_get_stats()
    expected_hits = args.repeats * args.calls
    actual_hits = stats["hits"]
    assert actual_hits == expected_hits, (
        f"Expected {expected_hits} hits, got {actual_hits}. Stats: {stats}"
    )

    # Compute and print the medians
    per_call_median = statistics.median(per_call_us_list)
    key_median = statistics.median(key_us_list)
    screen_median = statistics.median(screen_us_list)

    result = (
        f"per_call_us={per_call_median:.2f} key_us={key_median:.2f} screen_us={screen_median:.2f}"
    )
    print(result)

    return 0


if __name__ == "__main__":
    exit(main())
