"""install_skills.py --check compares installed copies to the repo (#2596).

Every test points the target at tmp_path. HOME is a directory under tmp_path
so a fallback to ~/.claude stays off the real home directory.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO_SKILLS = ROOT / "agent_assets" / "skills"
SCRIPT = ROOT / "scripts" / "install_skills.py"


def _skill_names() -> list[str]:
    return sorted(
        path.name
        for path in REPO_SKILLS.iterdir()
        if path.is_dir() and (path / "SKILL.md").is_file()
    )


def _copy_skills(destination: Path, names: list[str] | None = None) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    selected = set(names) if names is not None else None
    for name in _skill_names():
        if selected is not None and name not in selected:
            continue
        shutil.copytree(REPO_SKILLS / name, destination / name)


def _set_version(skill_md: Path, version: str) -> None:
    text = skill_md.read_text(encoding="utf-8")
    updated, count = re.subn(
        r"^xtrax_version:\s*.*$",
        f"xtrax_version: {version}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    assert count == 1, skill_md
    skill_md.write_text(updated, encoding="utf-8")


def _run(
    args: list[str],
    *,
    home: Path,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HOME"] = str(home)
    env.pop("XTRAX_SKILLS_TARGET", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_check_fresh_install_is_silent(tmp_path: Path) -> None:
    target = tmp_path / "installed"
    _copy_skills(target)
    sentinel = target / "using-xtrax" / "SENTINEL"
    sentinel.write_text("keep", encoding="utf-8")

    result = _run(["--check", "--target", str(target)], home=tmp_path / "home")

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "home" / ".claude").exists()


def test_check_reports_missing_skill(tmp_path: Path) -> None:
    names = _skill_names()
    missing = names[0]
    target = tmp_path / "installed"
    _copy_skills(target, names[1:])

    result = _run(["--check", "--target", str(target)], home=tmp_path / "home")

    assert result.returncode != 0
    lines = result.stdout.splitlines()
    assert f"MISSING {missing}" in lines
    for name in names[1:]:
        assert f"MISSING {name}" not in lines
        assert not any(line.startswith(f"STALE {name} ") for line in lines)


def _repo_version(name: str) -> str:
    text = (REPO_SKILLS / name / "SKILL.md").read_text(encoding="utf-8")
    match = re.search(r"^xtrax_version:\s*(\S+)", text, flags=re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_check_reports_stale_version(tmp_path: Path) -> None:
    target = tmp_path / "installed"
    _copy_skills(target)
    older = _skill_names()[0]
    newer = _skill_names()[1]
    repo_version = _repo_version(older)
    _set_version(target / older / "SKILL.md", "0.4.0a1")
    _set_version(target / newer / "SKILL.md", "9.9.9")

    result = _run(["--check", "--target", str(target)], home=tmp_path / "home")

    assert result.returncode != 0
    lines = result.stdout.splitlines()
    assert f"STALE {older} installed=0.4.0a1 repo={repo_version}" in lines
    assert f"STALE {newer} installed=9.9.9 repo={_repo_version(newer)}" in lines
    assert not any(line.startswith("MISSING ") for line in lines)


def test_check_does_not_create_target_dir(tmp_path: Path) -> None:
    target = tmp_path / "absent"

    result = _run(["--check", "--target", str(target)], home=tmp_path / "home")

    assert result.returncode != 0
    assert not target.exists()
    assert any(line.startswith("MISSING ") for line in result.stdout.splitlines())


def test_check_target_env_and_argument(tmp_path: Path) -> None:
    home = tmp_path / "home"
    via_env = tmp_path / "via-env"
    via_arg = tmp_path / "via-arg"
    _copy_skills(via_env)
    via_arg.mkdir()

    from_env = _run(
        ["--check"],
        home=home,
        extra_env={"XTRAX_SKILLS_TARGET": str(via_env)},
    )
    assert from_env.returncode == 0, from_env.stdout + from_env.stderr
    assert from_env.stdout == ""
    assert not (home / ".claude").exists()

    from_arg = _run(
        ["--check", "--target", str(via_arg)],
        home=home,
        extra_env={"XTRAX_SKILLS_TARGET": str(via_env)},
    )
    assert from_arg.returncode != 0
    assert "MISSING using-xtrax" in from_arg.stdout.splitlines()
