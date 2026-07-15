"""Ratchet-crash-atomicity (T2-10, #2181, AC-14, PM-7).

PM-7's failure mode (`.praxia/docs/specs/260702_design-the-2181-agentic-algorithm-evolut.md`
line 76): "a corrupted mid-ratchet git state (crash between commit and reset) destroyed the
best-so-far lineage with no transaction boundary... git-as-memory had no crash-atomicity." This
module is the fix: it makes "ratcheting" best-so-far a two-step sequence built entirely out of
git's own object-database and ref-update atomicity, rather than a hand-rolled JSON/TOML "current
best" pointer file that could drift out of sync with git's real state. Step one,
`create_pending_commit`, writes a commit OBJECT via `git commit-tree` -- an object not yet
reachable from any ref, i.e. exactly the DAG text's "tmp-commit" -- and fsyncs it for durability.
Step two, `advance_best_so_far`, moves a ref onto that commit via `git update-ref`'s own
compare-and-swap old-value argument -- git's own atomic ref update, not a second bespoke
mechanism. Because each step is independently atomic at the OS/git level, a crash at any point
between them (mid-ratchet) leaves `read_best_so_far` returning either the pre-crash or the
post-crash SHA -- never a torn or partial state -- which is what makes recovery correct "for
free" from git's own guarantees, not something this module reconstructs by hand.

Extension seam (mirrors `xtrax.loop.admission`/`closure_lock`/`evaluator_completeness`/
`external_stop_watchdog`'s stance): this module raises on failure and stops there. It does not
decide what a future loop controller does on a `RefUpdateConflictError` -- retry? re-derive the
pending commit's parent against the new tip? escalate to a human? -- no #2181 loop controller
exists yet to validate that design against, and inventing one now would be speculative.
"""

import os
import subprocess
from pathlib import Path


class RatchetCrashAtomicityError(Exception):
    """Base for AC-14 crash-atomicity failures."""


class CommitCreationError(RatchetCrashAtomicityError):
    """`git commit-tree` failed to create the pending ("tmp-commit") commit object."""


class RefUpdateConflictError(RatchetCrashAtomicityError):
    """`git update-ref`'s compare-and-swap check failed.

    Either a concurrent writer moved the ref between the caller reading `expected_old_sha` and
    calling `advance_best_so_far`, or the caller's `expected_old_sha` was simply stale.
    """


def _git_run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def _git_dir(repo: Path) -> Path:
    result = _git_run(repo, "rev-parse", "--git-dir")
    if result.returncode != 0:
        msg = f"git rev-parse --git-dir failed in {repo}: {result.stderr.strip()}"
        raise CommitCreationError(msg)
    git_dir = Path(result.stdout.strip())
    if not git_dir.is_absolute():
        git_dir = (repo / git_dir).resolve()
    return git_dir


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_pending_commit_object(repo: Path, commit_sha: str) -> None:
    """fsync the newly-written loose commit object for durability (the "tmp-commit" half).

    Mirrors `xtrax.devtools.baseline.save_baseline`'s fsync discipline: fsync the object file
    itself, then fsync its containing directory so the directory entry from git's own internal
    write is durable too. If git deduplicated or packed the object instead of writing it loose
    (not expected right after a fresh `commit-tree`, but not this module's business to assume
    away), there is nothing loose to fsync and this is a no-op.
    """
    object_path = _git_dir(repo) / "objects" / commit_sha[:2] / commit_sha[2:]
    if not object_path.is_file():
        return
    _fsync_file(object_path)
    _fsync_file(object_path.parent)


def create_pending_commit(repo: Path, tree_sha: str, parent_sha: str, message: str) -> str:
    """Create a commit object (via `git commit-tree`) without moving any ref.

    This is the DAG text's "tmp-commit": the returned commit is a real object in git's object
    database, but it is unreachable from any branch/ref until a caller later passes its SHA to
    `advance_best_so_far`. The written loose object is fsynced before returning, so once this
    call returns, the object survives a crash regardless of whether the subsequent ref update
    ever happens.

    Raises:
        CommitCreationError: `git commit-tree` failed (e.g. bad `tree_sha`/`parent_sha`).
    """
    repo = repo.resolve()
    result = _git_run(repo, "commit-tree", tree_sha, "-p", parent_sha, "-m", message)
    if result.returncode != 0:
        msg = f"git commit-tree {tree_sha} -p {parent_sha} failed: {result.stderr.strip()}"
        raise CommitCreationError(msg)
    commit_sha = result.stdout.strip()
    if not commit_sha:
        msg = f"git commit-tree {tree_sha} -p {parent_sha} produced no commit SHA"
        raise CommitCreationError(msg)
    _fsync_pending_commit_object(repo, commit_sha)
    return commit_sha


def advance_best_so_far(
    repo: Path,
    ref_name: str,
    new_sha: str,
    expected_old_sha: str | None,
) -> None:
    """Atomically move `ref_name` to `new_sha`, iff its current value is `expected_old_sha`.

    This is the DAG text's "atomic ref update": `git update-ref <ref> <new> <old>` performs its
    own compare-and-swap, refusing the update (and leaving the ref untouched) if the ref's actual
    current value does not match `expected_old_sha`. Passing `expected_old_sha=None` asserts the
    ref must not currently exist (git's own convention: an empty old-value means "must not
    exist" -- see `git update-ref --help`), for the first-ever ratchet of a given ref.

    Raises:
        RefUpdateConflictError: the compare-and-swap check failed -- the ref's actual value did
            not match `expected_old_sha` (including: it already existed when `None` was passed).
    """
    repo = repo.resolve()
    old_value = expected_old_sha if expected_old_sha is not None else ""
    result = _git_run(repo, "update-ref", ref_name, new_sha, old_value)
    if result.returncode != 0:
        msg = (
            f"git update-ref {ref_name} {new_sha} (expected_old_sha={expected_old_sha!r}) "
            f"failed -- compare-and-swap check did not hold: {result.stderr.strip()}"
        )
        raise RefUpdateConflictError(msg)


def read_best_so_far(repo: Path, ref_name: str) -> str | None:
    """Return `ref_name`'s current SHA, or `None` if the ref does not exist.

    This is the recovery path (AC-14's "recovery restores best-so-far after an injected
    mid-ratchet crash"): because both `create_pending_commit` (object write) and
    `advance_best_so_far` (ref write) are each independently atomic, calling this after an
    interruption at any point always returns either the pre-crash or the post-crash SHA -- never
    a torn/partial state -- with no reconstruction logic needed here.
    """
    repo = repo.resolve()
    result = _git_run(repo, "rev-parse", "--verify", ref_name)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


__all__ = [
    "CommitCreationError",
    "RatchetCrashAtomicityError",
    "RefUpdateConflictError",
    "advance_best_so_far",
    "create_pending_commit",
    "read_best_so_far",
]
