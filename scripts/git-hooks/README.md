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

- `pre-push` -- first syncs the CI-parity extras (`uv sync --frozen --extra dev --extra eda --extra io`, backlog #4398), echoing how long the sync took, then runs the same 3 cheap checks as the matching steps of `.github/workflows/ci.yml`'s `lint-format-type-test` job (`ruff check .`, `ruff format --check .`, `ty check src/`) before every `git push`. Skips
  the expensive coverage-gated pytest tiers -- those stay CI-only.
- Side effect to know: the sync makes `.venv` match *exactly* those extras, so
  anything extra you installed locally (torch, sphinx, ...) is stripped from
  the venv on push; `--frozen` guarantees uv.lock itself is never rewritten.

Since hooks are shared across worktrees, installing this once affects every
worktree's `git push`, not just the one it was installed from.

## ty-baseline.txt

`ty check src/` is ratcheted against `ty-baseline.txt`: that file lists
diagnostics that already existed when the hook landed (in files unrelated to
any specific PR). The hook only fails on diagnostics NOT already in that
file, so it catches new regressions without blocking every push on unrelated
pre-existing debt. The baseline may be empty -- then the hook is exactly as
strict as CI's hard gate, which is a fine state, not a broken one.

If you fix a baseline entry, remove its line from `ty-baseline.txt` in the
same commit -- the hook doesn't require this, but a shrinking baseline is a
better long-term state than a static one.

Bypass for a single push if genuinely needed: `git push --no-verify`.
