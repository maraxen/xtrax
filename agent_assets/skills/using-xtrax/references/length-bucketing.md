# Length bucketing

One compiled executable per rung. `AxisSpec.bucket_boundaries` selects the `Bucket` strategy. `select_bucket` and `bucketize` run on the host, before `jit`, and the device only ever sees a rung length.

`BUCKET_LADDER` is defined in `xtrax.export.rings` and re-exported from `xtrax.export`. It is not in `xtrax.tiling`.

```python
import jax
import jax.numpy as jnp
import numpy as np
from xtrax.export.rings import BUCKET_LADDER
from xtrax.tiling import AxisSpec, BatchPlanner, Bucket, bucketize, select_bucket

spec = AxisSpec(
    name="seq",
    cardinality=1000,
    default_batch_size=32,
    bucket_boundaries=BUCKET_LADDER,  # (64, 128, 256, 512, 1024, 1536, 2048)
)
plan = BatchPlanner().plan([spec])
assert isinstance(plan.decisions[0].strategy, Bucket)

seq = np.arange(100, dtype=np.float32)
bucket = select_bucket(len(seq), boundaries=BUCKET_LADDER)  # 128, the rung, not an index
padded, mask = bucketize(seq, bucket_size=bucket)          # host NumPy; mask shape (128,)

@jax.jit
def step(x):
    return x * jnp.float32(2)

kept = np.asarray(step(jnp.asarray(padded)))[mask]         # (100,)
```

Verify: `src/xtrax/tiling/plan.py:41-44` (`bucket_boundaries`), `src/xtrax/tiling/strategy.py:105-122` (`Bucket`), `src/xtrax/tiling/bucket.py:31-60` (`select_bucket`), `src/xtrax/tiling/bucket.py:63-108` (`bucketize`), `src/xtrax/export/rings.py:98` (`BUCKET_LADDER`).

## Host pad, one compile per rung

`select_bucket(length, boundaries)` returns the smallest boundary `>= length`. A length past the last boundary raises `ValueError` (no silent extra shape). `bucketize(xs, bucket_size)` pads every leaf's leading axis with NumPy and returns `(padded_xs, original_length_mask)`.

The jitted step closes over the rung length only. A later call with a different raw length that lands on the same rung hits that executable.

⚠ WARN: `jnp.pad` inside the jitted function, applied to the raw length on each call, is shape-specialized. Each distinct length compiles again, and the rung cache never forms. Pad on the host with `bucketize`, then call the jitted step on the padded array.

`Bucket` is a plan descriptor. `axis_dispatch(Bucket(...), fn, xs)` raises `TypeError`. Pad on the host, then dispatch the per-rung compute with `Vmap` or `ChunkedMap`.

Verify: `src/xtrax/tiling/bucket.py:1-19`, `src/xtrax/tiling/dispatch.py:202-210`.
