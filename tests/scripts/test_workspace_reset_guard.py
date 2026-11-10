"""Tests for scripts/git-hooks/pretool-git-reset-guard.sh (#2962)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK_PATH = REPO_ROOT / "scripts" / "git-hooks" / "pretool-git-reset-guard.sh"


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    tracked_file = path / "tracked.txt"
    tracked_file.write_text("initial content\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "--no-verify", "-m", "initial commit"],
        cwd=path,
        check=True,
        capture_output=True,
    )


def test_hook_script_exists() -> None:
    assert HOOK_PATH.is_file(), f"{HOOK_PATH} does not exist"


def test_guard_allows_non_bash_tool(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    payload = json.dumps({"tool_name": "Read", "tool_input": {"path": "tracked.txt"}})
    res = subprocess.run(
        ["bash", str(HOOK_PATH)],
        input=payload,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert res.returncode == 0


def test_guard_allows_safe_bash_commands(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "git status --porcelain"}})
    res = subprocess.run(
        ["bash", str(HOOK_PATH)],
        input=payload,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert res.returncode == 0


def test_guard_allows_clean_reset(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "git reset --hard HEAD"}})
    res = subprocess.run(
        ["bash", str(HOOK_PATH)],
        input=payload,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert res.returncode == 0


def test_guard_blocks_dirty_reset_and_creates_stash(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    # Dirty the working directory with modified file and untracked file
    (tmp_path / "tracked.txt").write_text("modified content\n", encoding="utf-8")
    (tmp_path / "untracked.txt").write_text("untracked content\n", encoding="utf-8")

    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "git reset --hard HEAD"}})
    res = subprocess.run(
        ["bash", str(HOOK_PATH)],
        input=payload,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert res.returncode == 2, (
        f"Expected exit 2, got {res.returncode}. Output: {res.stdout}\n{res.stderr}"
    )

    # Verify decision JSON was emitted
    decision_data = json.loads(res.stdout)
    assert (
        decision_data.get("decision") in ("block", "deny")
        or decision_data.get("permission") == "deny"
    )

    # Verify files are intact
    assert (tmp_path / "tracked.txt").read_text(encoding="utf-8") == "modified content\n"
    assert (tmp_path / "untracked.txt").read_text(encoding="utf-8") == "untracked content\n"

    # Verify safety stash log was created and contains valid stash OID
    log_file = tmp_path / ".praxia" / "safety_stashes.log"
    assert log_file.is_file()
    log_content = log_file.read_text(encoding="utf-8").strip()
    assert log_content
    stash_oid = log_content.split()[1]
    # Check stash object exists in git
    show_res = subprocess.run(
        ["git", "show", stash_oid], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert show_res.returncode == 0


def test_guard_override_env(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "tracked.txt").write_text("modified content\n", encoding="utf-8")
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "git reset --hard HEAD"}})
    env = dict(os.environ)
    env["PRAXIA_ALLOW_DESTRUCTIVE_RESET"] = "1"
    res = subprocess.run(
        ["bash", str(HOOK_PATH)],
        input=payload,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        check=False,
    )
    assert res.returncode == 0
