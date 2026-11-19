"""Tests for scripts/assemble_changelog.py --check-new-modules gate (#3265)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "assemble_changelog.py"


def _init_repo_with_branch(path: Path) -> tuple[Path, Path]:
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

    src_dir = path / "src" / "xtrax"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "__init__.py").write_text("# root init\n", encoding="utf-8")
    (src_dir / "existing.py").write_text("# existing\n", encoding="utf-8")

    frag_dir = path / "changelog.d"
    frag_dir.mkdir(parents=True, exist_ok=True)
    (frag_dir / "README.md").write_text("# readme\n", encoding="utf-8")

    subprocess.run(["git", "add", "."], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "--no-verify", "-m", "initial commit"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "branch", "-M", "main"], cwd=path, check=True, capture_output=True)

    # Create feature branch
    subprocess.run(
        ["git", "checkout", "-b", "feature-test"], cwd=path, check=True, capture_output=True
    )
    return src_dir, frag_dir


def test_gate_passes_when_no_new_modules(tmp_path: Path) -> None:
    src_dir, _ = _init_repo_with_branch(tmp_path)
    # Modify existing file
    (src_dir / "existing.py").write_text("# updated\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "--no-verify", "-m", "update existing"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    res = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--check-new-modules", "--base-ref", "main"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0


def test_gate_fails_when_new_module_added_without_fragment(tmp_path: Path) -> None:
    src_dir, _ = _init_repo_with_branch(tmp_path)
    # Add new module
    (src_dir / "brand_new_feature.py").write_text("# feature\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "--no-verify", "-m", "add brand_new_feature"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    res = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--check-new-modules", "--base-ref", "main"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 1
    assert "brand_new_feature.py" in res.stderr or "brand_new_feature.py" in res.stdout


def test_gate_passes_when_new_module_and_fragment_added(tmp_path: Path) -> None:
    src_dir, frag_dir = _init_repo_with_branch(tmp_path)
    (src_dir / "brand_new_feature.py").write_text("# feature\n", encoding="utf-8")
    (frag_dir / "brand_new_feature.added.md").write_text(
        "- Add brand new feature\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "--no-verify", "-m", "add feature and fragment"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    res = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--check-new-modules", "--base-ref", "main"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0


def test_gate_subpackage_init_requires_fragment(tmp_path: Path) -> None:
    src_dir, _ = _init_repo_with_branch(tmp_path)
    subpkg = src_dir / "new_subpackage"
    subpkg.mkdir()
    (subpkg / "__init__.py").write_text("# subpkg init\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "--no-verify", "-m", "add subpkg init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    res = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--check-new-modules", "--base-ref", "main"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 1
    assert "new_subpackage/__init__.py" in (res.stderr + res.stdout)


def test_gate_raises_on_unresolvable_base(tmp_path: Path) -> None:
    _init_repo_with_branch(tmp_path)
    res = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--check-new-modules",
            "--base-ref",
            "nonexistent_ref_12345",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode != 0
