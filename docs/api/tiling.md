# Tiling

```{automodule} xtrax.tiling
:members:
:undoc-members:
:show-inheritance:
```

## Dedup-spec synthesis and verification

`xtrax.tiling.dedup_synthesis` is not re-exported from `xtrax.tiling.__init__`;
import it directly.

- **`synthesize_dedup_spec`** auto-synthesizes a `DedupSpec` from batch evidence
  via a two-stage sample-then-exact algorithm.
- **`verify_dedup_spec`** checks claim (i), row-equality, of an existing
  `DedupSpec`: every row is compared **bitwise, per leaf, in its native byte
  layout** against its claimed canonical row — never a numeric tolerance,
  which would silently accept merging `±0.0` or `x`/`nextafter(x)`. It does
  **not** check claim (ii), compute-equivalence ("does re-running `fn` on the
  deduped rows reproduce `fn`'s full output"); that needs `fn` and is
  follow-up #5217.
- Row identity is exact **native bytes per leaf**: every leaf is converted to
  an `(N, B_l)` uint8 block in its own dtype's byte layout (not a promoted,
  concatenated float view), and blocks are concatenated along the feature
  axis. `bool` and `complex` get their own recipe; sub-byte integers are
  widened losslessly before comparison; sub-byte floats are refused.
- Verification costs O(N) device compute and transfers exactly one `(N, L)`
  boolean mismatch mask (`N * L` bytes, `L` = number of leaves) from device to
  host, regardless of leaf dtype.
- A row is reported bad if it mismatches its canonical row in **any** leaf
  (union semantics); `DedupSpecVerificationError.first_bad_row`/`n_bad`/
  `leaf_index` describe the first failure under that union.

```{automodule} xtrax.tiling.dedup_synthesis
:members:
:undoc-members:
:show-inheritance:
```
