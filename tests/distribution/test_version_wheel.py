"""Tests for distribution N1 version contract + wheel gate (#1452)."""

from __future__ import annotations

import subprocess
import textwrap
import zipfile
from email.message import EmailMessage
from pathlib import Path

import pytest

from scripts.audit_version_wheel import (
    audit_version_wheel,
    find_built_wheel,
    load_version_contract,
    parse_init_version,
    read_hatch_version_path,
    read_wheel_metadata_version,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "distribution" / "version_contract.toml"


def _write_contract(repo_root: Path) -> Path:
    config_dir = repo_root / "distribution"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "version_contract.toml"
    config_path.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return config_path


def _write_minimal_pyproject(repo_root: Path, *, hatch_path: str) -> Path:
    pyproject = repo_root / "pyproject.toml"
    pyproject.write_text(
        textwrap.dedent(
            f"""
            [project]
            name = "xtrax"
            dynamic = ["version"]

            [build-system]
            requires = ["hatchling"]
            build-backend = "hatchling.build"

            [tool.hatch.version]
            path = "{hatch_path}"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return pyproject


def _write_init(
    repo_root: Path,
    version: str,
    *,
    relative: str = "src/xtrax/__init__.py",
) -> Path:
    init_path = repo_root / relative
    init_path.parent.mkdir(parents=True, exist_ok=True)
    init_path.write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    return init_path


def _make_wheel_with_version(tmp_path: Path, version: str) -> Path:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    wheel_path = dist_dir / f"xtrax-{version}-py3-none-any.whl"
    metadata = EmailMessage()
    metadata["Metadata-Version"] = "2.1"
    metadata["Name"] = "xtrax"
    metadata["Version"] = version
    with zipfile.ZipFile(wheel_path, "w") as archive:
        archive.writestr("xtrax-0.0.0.dist-info/METADATA", metadata.as_bytes())
    return wheel_path


def test_load_version_contract_reads_committed_toml() -> None:
    contract = load_version_contract(CONFIG_PATH)
    assert contract.version == "0.1.0"
    assert contract.version_source == "src/xtrax/__init__.py"
    assert contract.attribute == "__version__"
    assert contract.hatch_path_key == "tool.hatch.version.path"


def test_parse_init_version_reads_string_literal(tmp_path: Path) -> None:
    init_path = tmp_path / "__init__.py"
    init_path.write_text('__version__ = "0.3.0"\n', encoding="utf-8")
    assert parse_init_version(init_path) == "0.3.0"


def test_parse_init_version_rejects_non_literal(tmp_path: Path) -> None:
    init_path = tmp_path / "__init__.py"
    init_path.write_text(
        "__version__ = importlib.metadata.version('xtrax')\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be a string literal"):
        parse_init_version(init_path)


def test_read_hatch_version_path_reads_pyproject(tmp_path: Path) -> None:
    _write_minimal_pyproject(tmp_path, hatch_path="src/xtrax/__init__.py")
    pyproject_path = tmp_path / "pyproject.toml"
    assert read_hatch_version_path(pyproject_path) == "src/xtrax/__init__.py"


def test_find_built_wheel_returns_latest(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    older = dist_dir / "xtrax-0.1.0-py3-none-any.whl"
    newer = dist_dir / "xtrax-0.3.0-py3-none-any.whl"
    older.write_text("old\n", encoding="utf-8")
    newer.write_text("new\n", encoding="utf-8")
    assert find_built_wheel(dist_dir) == newer


def test_read_wheel_metadata_version(tmp_path: Path) -> None:
    wheel_path = _make_wheel_with_version(tmp_path, "0.3.0")
    assert read_wheel_metadata_version(wheel_path) == "0.3.0"


def test_audit_version_wheel_metadata_only_passes(tmp_path: Path) -> None:
    _write_init(tmp_path, "0.3.0")
    _write_minimal_pyproject(tmp_path, hatch_path="src/xtrax/__init__.py")
    config_path = _write_contract(tmp_path)

    passed, failures, version = audit_version_wheel(
        root=tmp_path,
        config_path=config_path,
        metadata_only=True,
    )
    assert passed is True
    assert failures == []
    assert version == "0.3.0"


def test_audit_version_wheel_metadata_only_fails_on_hatch_path_mismatch(
    tmp_path: Path,
) -> None:
    _write_init(tmp_path, "0.3.0")
    _write_minimal_pyproject(tmp_path, hatch_path="wrong/__init__.py")
    config_path = _write_contract(tmp_path)

    passed, failures, version = audit_version_wheel(
        root=tmp_path,
        config_path=config_path,
        metadata_only=True,
    )
    assert passed is False
    assert version == "0.3.0"
    assert any("hatch version path mismatch" in item for item in failures)


def test_audit_version_wheel_fails_on_wheel_version_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_init(tmp_path, "0.3.0")
    _write_minimal_pyproject(tmp_path, hatch_path="src/xtrax/__init__.py")
    config_path = _write_contract(tmp_path)
    _make_wheel_with_version(tmp_path, "9.9.9")

    monkeypatch.setattr(
        "scripts.audit_version_wheel.run_uv_build",
        lambda root: (True, ""),
    )

    passed, failures, version = audit_version_wheel(
        root=tmp_path,
        config_path=config_path,
        metadata_only=False,
    )
    assert passed is False
    assert version == "0.3.0"
    assert any("wheel METADATA Version mismatch" in item for item in failures)


def test_script_metadata_only_subprocess_exits_zero() -> None:
    result = subprocess.run(
        ["uv", "run", "python", "scripts/audit_version_wheel.py", "--metadata-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "PASS: version contract" in result.stdout
    assert "__version__='0.3.0'" in result.stdout
