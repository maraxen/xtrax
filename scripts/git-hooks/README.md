# git-hooks

Versioned source of truth for local git hooks. Git hooks live in the shared
`.git/hooks/` directory (common across all worktrees of this repo, not
per-worktree), which is not itself version-controlled -- so the working
copies there can be lost on a fresh clone or if `.git/hooks/` is ever wiped.
This directory keeps a durable, reviewable copy.

## Install (or reinstall)

```bash
cp scripts/git-hooks/pre-push "$(git rev-parse --git-common-dir)/hooks/pre-push"
chmod +x "$(git rev-parse --git-common-dir)/hooks/pre-push"
```

## What's installed

- `pre-push` -- first syncs the CI-parity extras (`uv sync --extra dev --extra eda --extra io --quiet`, backlog #4398), then runs the same 3 cheap checks as the matching steps of `.github/workflows/ci.yml`'s `lint-format-type-test` job (`ruff check .`, `ruff format --check .`, `ty check src/`) before every `git push`. Skips
  the expensive coverage-gated pytest tiers -- those stay CI-only.

Since hooks are shared across worktrees, installing this once affects every
worktree's `git push`, not just the one it was installed from.

## ty-baseline.txt

`ty check src/` is ratcheted, not zero-tolerance: `ty-baseline.txt` lists
diagnostics that already existed repo-wide as of 260812 (in files unrelated
to any specific PR -- e.g. a stale cast in `devtools/tombstone.py`, a
type-narrowing gap in `inference/ir_schema.py`). The hook only fails on
diagnostics NOT already in that file, so it catches new regressions without
blocking every push on unrelated pre-existing debt.

If you fix a baseline entry, remove its line from `ty-baseline.txt` in the
same commit -- the hook doesn't require this, but a shrinking baseline is a
better long-term state than a static one.

Bypass for a single push if genuinely needed: `git push --no-verify`.
