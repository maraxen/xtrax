# Codemod: `SafeMap` → `ChunkedMap` (xtrax #3644)

xtrax renamed its chunked-dispatch strategy:

| Old | New |
|---|---|
| `xtrax.tiling.SafeMap` | `xtrax.tiling.ChunkedMap` |
| `xtrax.tiling.SafeMapIterator` | `xtrax.tiling.ChunkedMapIterator` |
| `xtrax.transforms.safe_map` | `xtrax.transforms.chunked_map` |

Why: `safe_map` already means something else in JAX (`jax._src.util.safe_map`, a
length-checked map). "Safe" also suggested the checkify-based `xtrax.safety` subsystem.
The strategy is memory-bounded chunking over `jax.lax.map(batch_size=...)`, so the new
name says that.

The old names still import for **one release** as deprecated aliases, and each use raises
a `DeprecationWarning`. They are removed in the release after that.

## Run it

Python, structurally (needs [ast-grep](https://ast-grep.github.io/) ≥ 0.42). Start with a
report-only pass:

```bash
ast-grep scan --rule codemods/safemap-to-chunkedmap/rules.yml src tests
```

If it prints `warning[jax-safe-map-unaliased-import]`, alias that import before rewriting
(`from jax._src.util import safe_map as jax_safe_map`), and rename its call sites to match.
The identifier rule can't tell a later bare `safe_map(...)` in that file from xtrax's, so
it would rename it and leave a `NameError`. Then rewrite:

```bash
ast-grep scan --rule codemods/safemap-to-chunkedmap/rules.yml --update-all src tests
```

Markdown, by whole-word substitution (stdlib only):

```bash
python codemods/safemap-to-chunkedmap/rename_markdown.py --check docs README.md   # preview
python codemods/safemap-to-chunkedmap/rename_markdown.py docs README.md
```

Then run your import sorter and formatter. Renamed names sort differently, so
`ruff check --fix --select I` and `ruff format` will both have work to do. Both codemods
are idempotent, so running them again on migrated code changes nothing.

## What is rewritten, and what is deliberately not

Rewritten:

- the identifiers `SafeMap`, `SafeMapIterator` and `safe_map`, wherever they are
  imported, called, subclassed or referenced as `xtrax.safe_map`;
- the exact string literals `"SafeMap"` and `"SafeMapIterator"`. These are strategy names
  compared as text, as in `type(strategy).__name__ == "SafeMap"`, and names in `__all__`;
- the string `"safe_map"` only inside `__all__` or `_LAZY` assignments, where it is an export
  name;
- the class names inside longer strings (error messages, reasoning text), and all three
  names in comments and docstrings, as whole words.

Left alone:

- JAX's own `safe_map`, whether imported from a `jax…` module, reached as `….util.safe_map`,
  or named in prose ("JAX's own safe_map").
- Any other `"safe_map"` string, for example a benchmark label written into bench
  records, so history stays comparable.
- Longer identifiers such as `safe_map_count` or `TestSafeMapThing`.

Known limits:

- The JAX exclusion in prose (comments, docstrings, Markdown) applies to the whole
  line or comment. A line that mentions JAX's `safe_map` and xtrax's `SafeMap` together
  keeps both names, so fix those by hand.
- In Markdown, `from jax._src.util import safe_map` inside a code block is renamed. Only
  `util.safe_map` and "JAX's (own) safe_map" are recognised there, so check the preview.

`fixture_before.py` → `fixture_after.py` in this directory is the exact expected
transformation. xtrax's `tests/test_renamed_chunked_map.py` checks it, and checks that
a second run changes nothing.

## For code that dispatches on the strategy's class name

The deprecated alias returns the new class, so `type(SafeMap(...)).__name__` is
`"ChunkedMap"`. Code that dispatches on a strategy's class *name* has to accept the new
one before it upgrades xtrax. For example, aminx's `host/kernel_dispatch.py` does
`if strategy_name == "SafeMap":`. The string-literal rule above rewrites that line for you.
Otherwise the xtrax strategy falls through to the caller's fallback branch.

Inside xtrax, name-based matching (`stages.topology`, `export.composer`) accepts both names
for this release, so a consumer's own duck-typed class named `SafeMap` keeps working until
it migrates.
