"""Tests for ratchet-crash-atomicity (T2-10, #2181, AC-14, PM-7)."""

import subprocess
from pathlib import Path

import pytest

from xtrax.loop.ratchet_crash_atomicity import (
    CommitCreationError,
    RefUpdateConflictError,
    advance_best_so_far,
    create_pending_commit,
    read_best_so_far,
)

_REF_NAME = "refs/xtrax/best-so-far"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A minimal real git repo with one commit, so a valid parent/tree SHA exists."""
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(tmp_path, "add", "seed.txt")
    _git(tmp_path, "commit", "--quiet", "-m", "initial commit")
    return tmp_path


def _head_sha(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _head_tree_sha(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD^{tree}").stdout.strip()


class TestCreatePendingCommit:
    def test_creates_an_unreachable_commit_object(self, repo: Path) -> None:
        parent = _head_sha(repo)
        tree = _head_tree_sha(repo)

        pending_sha = create_pending_commit(repo, tree, parent, "pending candidate")

        # The object exists (cat-file succeeds) ...
        cat = _git(repo, "cat-file", "-t", pending_sha)
        assert cat.stdout.strip() == "commit"
        # ... but is not reachable from any ref yet -- no branch/ref advanced.
        branches = _git(repo, "branch", "--contains", pending_sha).stdout
        assert branches.strip() == ""

    def test_raises_on_bad_tree_sha(self, repo: Path) -> None:
        parent = _head_sha(repo)
        with pytest.raises(CommitCreationError):
            create_pending_commit(repo, "0" * 40, parent, "bad tree")


class TestRecoveryAfterMidRatchetCrash:
    def test_crash_between_commit_and_ref_update_preserves_old_lineage(self, repo: Path) -> None:
        """The core AC-14 claim: a crash between commit and reset never loses best-so-far."""
        old_sha = _head_sha(repo)
        tree = _head_tree_sha(repo)
        advance_best_so_far(repo, _REF_NAME, old_sha, expected_old_sha=None)

        # Simulate a mid-ratchet crash: create the pending commit object, but never call
        # advance_best_so_far -- as if the process died right between the two steps.
        create_pending_commit(repo, tree, old_sha, "candidate that never got ratcheted in")

        # Recovery: the ref was never touched, so best-so-far must still be the OLD sha.
        assert read_best_so_far(repo, _REF_NAME) == old_sha

    def test_full_cycle_advances_best_so_far_to_new_sha(self, repo: Path) -> None:
        old_sha = _head_sha(repo)
        tree = _head_tree_sha(repo)
        advance_best_so_far(repo, _REF_NAME, old_sha, expected_old_sha=None)

        new_sha = create_pending_commit(repo, tree, old_sha, "candidate that gets ratcheted in")
        advance_best_so_far(repo, _REF_NAME, new_sha, expected_old_sha=old_sha)

        assert read_best_so_far(repo, _REF_NAME) == new_sha


class TestAdvanceBestSoFarCasConflict:
    def test_stale_expected_old_sha_raises_and_leaves_ref_untouched(self, repo: Path) -> None:
        old_sha = _head_sha(repo)
        tree = _head_tree_sha(repo)
        advance_best_so_far(repo, _REF_NAME, old_sha, expected_old_sha=None)

        new_sha = create_pending_commit(repo, tree, old_sha, "candidate A")
        stale_sha = "1" * 40

        with pytest.raises(RefUpdateConflictError):
            advance_best_so_far(repo, _REF_NAME, new_sha, expected_old_sha=stale_sha)

        # The ref must be unchanged after the failed CAS -- still the old sha.
        assert read_best_so_far(repo, _REF_NAME) == old_sha


class TestFirstEverRefCreation:
    def test_expected_old_sha_none_succeeds_when_ref_absent(self, repo: Path) -> None:
        assert read_best_so_far(repo, _REF_NAME) is None
        sha = _head_sha(repo)

        advance_best_so_far(repo, _REF_NAME, sha, expected_old_sha=None)

        assert read_best_so_far(repo, _REF_NAME) == sha

    def test_expected_old_sha_none_fails_once_ref_already_exists(self, repo: Path) -> None:
        sha = _head_sha(repo)
        advance_best_so_far(repo, _REF_NAME, sha, expected_old_sha=None)

        tree = _head_tree_sha(repo)
        other_sha = create_pending_commit(repo, tree, sha, "another candidate")

        with pytest.raises(RefUpdateConflictError):
            advance_best_so_far(repo, _REF_NAME, other_sha, expected_old_sha=None)

        # Unchanged: the ref still points at the first-ever ratcheted sha.
        assert read_best_so_far(repo, _REF_NAME) == sha


class TestReadBestSoFar:
    def test_returns_none_for_a_ref_that_never_existed(self, repo: Path) -> None:
        assert read_best_so_far(repo, "refs/xtrax/never-created") is None
