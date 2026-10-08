# ruff: noqa -- a codemod fixture: its imports are deliberately unused
"""A module using ChunkedMap and chunked_map (docstring prose: rewritten)."""

from jax._src.util import safe_map as jax_safe_map  # JAX's own safe_map: untouched
from jax._src.util import safe_map
from xtrax.tiling import ChunkedMap, ChunkedMapIterator
from xtrax.transforms import chunked_map as xtrax_map

import jax._src.util as util
import xtrax

__all__ = ["ChunkedMap", "chunked_map"]
_LAZY = {"ChunkedMap": "xtrax.tiling", "chunked_map": "xtrax.transforms"}

BENCH_LABELS = {"safe_map": 1, "vmap": 2}  # a label, not an export name: untouched


def run(xs, fn):
    # ChunkedMap chunks; see chunked_map and ChunkedMapIterator.
    strategy = ChunkedMap(batch_size=8)
    it = ChunkedMapIterator(tile=8)
    a = xtrax.chunked_map(fn, xs, batch_size=4)
    b = xtrax_map(fn, xs, batch_size=4)
    c = util.safe_map(fn, xs)  # jax's: untouched
    d = jax_safe_map(fn, xs)
    if type(strategy).__name__ in ("Vmap", "ChunkedMap"):
        pass
    safe_map_count = 3  # a different identifier: untouched
    return strategy, it, a, b, c, d, safe_map_count


class TestSafeMapThing:  # a longer identifier: untouched
    pass
