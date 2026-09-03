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

### The hook does not cover `controller/`

`ty check src/` means `src/` -- `controller/` is a separate top-level tree, and
`[tool.ty.src] include` is scoped to `src/**/*.py`, so `controller/` is filtered out
before any path argument is even consulted. That is deliberate here: the hook's stated
job is to mirror `lint-format-type-test`, and covering `controller/` would mean
installing the `controller` extra (bathos plus ~62 transitive packages) on every local
push.

`controller/` is type-checked in CI instead, by the `controller-tests` job via
`just audit-controller-types`. **A green pre-push says nothing about `controller/`.**
The failure mode this note exists to prevent is subtler than a plain gap: a bare
`ty check controller/` does not error, it prints `WARN No python files found` and then
`All checks passed!` -- so the tree can look checked while nothing was read. The
`audit-controller-types` recipe passes an explicit `-c 'src.include=[...]'` override
for exactly that reason.

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
