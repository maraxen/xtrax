"""XTRAX_GIT_SHA / .git_sha file beat a missing .git on cluster scratch.

Ported from prolix tests/profiling/test_capture_git_sha.py; the env var was
renamed PROLIX_GIT_SHA -> XTRAX_GIT_SHA in the upstream port.
"""

from __future__ import annotations

from xtrax.profiling import record as rec


def test_capture_git_sha_from_env(monkeypatch):
    monkeypatch.setenv("XTRAX_GIT_SHA", "abc123deadbeef")
    assert rec._capture_git_sha() == "abc123deadbeef"


def test_capture_git_sha_from_file(monkeypatch, tmp_path):
    monkeypatch.delenv("XTRAX_GIT_SHA", raising=False)
    sha_file = tmp_path / ".git_sha"
    sha_file.write_text("cafebabeface\n")
    monkeypatch.setattr(rec, "_REPO_ROOT", tmp_path)
    assert rec._capture_git_sha() == "cafebabeface"


def test_repo_root_anchors_at_repository_root():
    """D3 port-bug guard: _REPO_ROOT must be the repository root, not <repo>/src.

    prolix's scripts/profiling layout needed parents[2]; src/xtrax/profiling
    needs parents[3]. A wrong depth silently breaks the `.git_sha` sidecar
    and `git rev-parse` fallback chain (records would stamp "unknown").
    """
    assert (rec._REPO_ROOT / "pyproject.toml").is_file(), (
        f"_REPO_ROOT resolved to {rec._REPO_ROOT}, which does not contain "
        "pyproject.toml -- the parents[N] depth no longer matches the "
        "package layout; fix _REPO_ROOT in xtrax/profiling/record.py"
    )
