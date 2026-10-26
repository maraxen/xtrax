# Changelog fragments

Feature PRs record their changelog entry as a file in this directory instead of
editing `## [Unreleased]` in `CHANGELOG.md` (two PRs editing the same section always
conflicted). The release PR assembles the fragments.

## Format

One file per change, named `<slug>.<category>.md`:

- `<slug>`: letters, digits, `-` and `_` (for example `merge-conflict-prevention`).
- `<category>`: one of `added`, `changed`, `deprecated`, `removed`, `fixed`,
  `security` (the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) categories).
- Content: Markdown bullet lines exactly as they should appear under that category's
  heading. No heading lines; the category comes from the file name.

Example, `changelog.d/struct-pytree.added.md`:

```markdown
- `xtrax.Struct` is a frozen-dataclass pytree base whose subclasses need no decorator.
```

Naming a public dataclass in a fragment acknowledges a shape change to it for the
`audit-dataclass-shape-diff` gate, as a `CHANGELOG.md` line does. Untracked fragment
files count.

## Checking and releasing

```bash
uv run python scripts/assemble_changelog.py --check      # CI runs this on every PR
uv run python scripts/assemble_changelog.py --version X.Y.Z --date YYYY-MM-DD   # release PR
```

The release command merges every fragment and the bullets already under
`## [Unreleased]` into a new `## [X.Y.Z] - DATE` section (Keep a Changelog category
order), leaves `## [Unreleased]` empty, and deletes the fragments. It refuses an
existing version, a missing `## [Unreleased]` heading, and a release with nothing in it.
