"""XTRAX_GIT_SHA / .git_sha file beat a missing .git on cluster scratch.

Ported from prolix tests/profiling/test_capture_git_sha.py; the env var was
renamed PROLIX_GIT_SHA -> XTRAX_GIT_SHA in the upstream port.

Caller-repository resolution (spec 261016 X2): a ProbeRecord must carry the
commit of the repository its measurement was taken from, not the xtrax
checkout the module was imported from.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from xtrax.profiling import record as rec
from xtrax.profiling.emitters import emit_probe_record

_XTRAX_ROOT = Path(rec.__file__).resolve().parents[3]


def test_capture_git_sha_from_env(monkeypatch):
    monkeypatch.setenv("XTRAX_GIT_SHA", "abc123deadbeef")
    assert rec._capture_git_sha() == "abc123deadbeef"


def test_capture_git_sha_from_file(monkeypatch, tmp_path):
    monkeypatch.delenv("XTRAX_GIT_SHA", raising=False)
    sha_file = tmp_path / ".git_sha"
    sha_file.write_text("cafebabeface\n")
    monkeypatch.chdir(tmp_path)
    assert rec._capture_git_sha() == "cafebabeface"


# --- Caller-repository resolution (spec 261016 X2) --------------------------


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


@pytest.fixture
def isolated_git(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.delenv("XTRAX_GIT_SHA", raising=False)


def _make_repo(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "x",
        ],
        check=True,
    )
    return _git(path, "rev-parse", "HEAD")


def _xtrax_head() -> str:
    return _git(_XTRAX_ROOT, "rev-parse", "HEAD")


def test_cwd_repo_sha_wins_over_xtrax_checkout(isolated_git, monkeypatch, tmp_path):
    repo_a = tmp_path / "A"
    sha_a = _make_repo(repo_a)
    monkeypatch.chdir(repo_a)
    assert rec._capture_git_sha() == sha_a
    assert sha_a != _xtrax_head()


def test_cwd_subdirectory_resolves_to_repo_top_level(isolated_git, monkeypatch, tmp_path):
    repo_a = tmp_path / "A"
    sha_a = _make_repo(repo_a)
    sub = repo_a / "deep" / "dir"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    assert rec._capture_git_sha() == sha_a


def test_probe_record_built_in_repo_carries_repo_sha(isolated_git, monkeypatch, tmp_path):
    repo_a = tmp_path / "A"
    sha_a = _make_repo(repo_a)
    monkeypatch.chdir(repo_a)
    probe = rec.ProbeRecord(probe_id="p", stage=0, n_atoms=8, platform="cpu", metrics={"x": 1.0})
    assert probe.git_sha == sha_a


def test_explicit_git_sha_kwarg_bypasses_capture(isolated_git, monkeypatch, tmp_path):
    repo_a = tmp_path / "A"
    _make_repo(repo_a)
    monkeypatch.chdir(repo_a)
    probe = rec.ProbeRecord(
        probe_id="p", stage=0, n_atoms=8, platform="cpu", metrics={"x": 1.0}, git_sha="explicit"
    )
    assert probe.git_sha == "explicit"


def test_explicit_repo_root_beats_non_repo_cwd(isolated_git, monkeypatch, tmp_path):
    repo_a = tmp_path / "A"
    sha_a = _make_repo(repo_a)
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.chdir(plain)
    assert rec._capture_git_sha(repo_a) == sha_a


def test_env_var_wins_over_repo(isolated_git, monkeypatch, tmp_path):
    repo_a = tmp_path / "A"
    _make_repo(repo_a)
    monkeypatch.chdir(repo_a)
    monkeypatch.setenv("XTRAX_GIT_SHA", "  envsha  ")
    assert rec._capture_git_sha() == "envsha"


def test_dirty_repo_suffixed(isolated_git, monkeypatch, tmp_path):
    repo_a = tmp_path / "A"
    sha_a = _make_repo(repo_a)
    (repo_a / "untracked.txt").write_text("x")
    monkeypatch.chdir(repo_a)
    assert rec._capture_git_sha() == f"{sha_a}-dirty"


def test_git_sha_file_in_non_repo_cwd_is_honored(isolated_git, monkeypatch, tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / ".git_sha").write_text("abc123 extra\n")
    monkeypatch.chdir(plain)
    assert rec._capture_git_sha() == "abc123"


def test_non_repo_without_sha_file_is_unknown(isolated_git, monkeypatch, tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.chdir(plain)
    assert rec._capture_git_sha() == "unknown"


def test_emit_probe_record_explicit_git_sha(isolated_git, tmp_path):
    out = tmp_path / "r.json"
    emit_probe_record(
        path=out,
        probe_id="p",
        stage=0,
        n_atoms=8,
        platform="cpu",
        metrics={"x": 1.0},
        git_sha="deadbeef",
    )
    assert rec.ProbeRecord.read(out).git_sha == "deadbeef"


def test_emit_probe_record_repo_root(isolated_git, monkeypatch, tmp_path):
    repo_a = tmp_path / "A"
    sha_a = _make_repo(repo_a)
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.chdir(plain)
    out = tmp_path / "r.json"
    emit_probe_record(
        path=out,
        probe_id="p",
        stage=0,
        n_atoms=8,
        platform="cpu",
        metrics={"x": 1.0},
        repo_root=repo_a,
    )
    assert rec.ProbeRecord.read(out).git_sha == sha_a
