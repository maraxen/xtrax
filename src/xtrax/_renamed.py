"""Name matching for the #3644 rename (SafeMap -> ChunkedMap).

The deprecated `SafeMap` / `SafeMapIterator` / `safe_map` aliases shipped for one release
(0.4.0a11) and are removed. Migrate with the ast-grep rules in
`codemods/safemap-to-chunkedmap/` (see its README).

What remains is the set of strategy class NAMES that name-based matching accepts for a
chunked map (stages/topology, export/composer). It still includes the legacy "SafeMap"
because a consumer can define its OWN duck-typed class of that name: aminx ships
`aminx.tiling.strategy.SafeMap` until its deprecation (aminx debt #2371). Drop it then
(xtrax #5680, step 4).
"""

CHUNKED_MAP_NAMES = ("ChunkedMap", "SafeMap")
