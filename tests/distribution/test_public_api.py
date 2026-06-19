"""Tests for distribution N3 public API contract + lazy export gate (#1453)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.audit_public_api import (
    audit_public_api,
    check_forbid_eager_imports_at_root,
    load_public_api_contract,
    parse_init_all,
    parse_init_lazy_keys,
    smoke_resolve_exports,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "distribution" / "public_api.toml"


def _write_contract(repo_root: Path) -> Path:
    config_dir = repo_root / "distribution"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "public_api.toml"
    config_path.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return config_path


def _write_lazy_root_init(
    repo_root: Path,
    *,
    all_names: list[str],
    lazy_map: dict[str, str],
) -> Path:
    init_path = repo_root / "src" / "xtrax" / "__init__.py"
    init_path.parent.mkdir(parents=True, exist_ok=True)
    all_lines = ",\n    ".join(repr(name) for name in all_names)
    lazy_lines = ",\n    ".join(
        f"{name!r}: {module!r}" for name, module in lazy_map.items()
    )
    init_path.write_text(
        "\n".join(
            [
                '__version__ = "0.3.0"',
                "",
                "__all__ = [",
                f"    {all_lines},",
                "]",
                "",
                "_LAZY = {",
                f"    {lazy_lines},",
                "}",
                "",
                "def __getattr__(name):",
                "    if name in _LAZY:",
                "        import importlib",
                "        return getattr(importlib.import_module(_LAZY[name]), name)",
                (
                    '    raise AttributeError('
                    'f"module \'xtrax\' has no attribute {name!r}")'
                ),
                "",
                "def __dir__():",
                "    return sorted(__all__)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return init_path


def test_load_public_api_contract_reads_committed_toml() -> None:
    contract = load_public_api_contract(CONFIG_PATH)
    assert contract.version == "0.1.0"
    assert contract.lazy_root is True
    assert contract.root_module == "xtrax"
    assert contract.forbid_eager_imports_at_root is True
    assert "training" in contract.required_subpackage_inits
    assert "DedupGather" in contract.tier1_exports
    assert "DedupGather" in contract.smoke_exports


def test_parse_init_all_and_lazy_keys_match_committed_root() -> None:
    init_path = ROOT / "src" / "xtrax" / "__init__.py"
    assert set(parse_init_all(init_path)) == set(parse_init_lazy_keys(init_path))


def test_check_forbid_eager_imports_at_root_passes_committed_root() -> None:
    init_path = ROOT / "src" / "xtrax" / "__init__.py"
    assert check_forbid_eager_imports_at_root(init_path) == []


def test_check_forbid_eager_imports_at_root_rejects_eager_xtrax_import(
    tmp_path: Path,
) -> None:
    init_path = tmp_path / "__init__.py"
    init_path.write_text(
        "from xtrax.training import Trainer\n__all__ = ['Trainer']\n",
        encoding="utf-8",
    )
    failures = check_forbid_eager_imports_at_root(init_path)
    assert any("forbidden eager import" in item for item in failures)


def test_audit_public_api_passes_on_repo_root() -> None:
    passed, failures = audit_public_api(
        root=ROOT,
        config_path=CONFIG_PATH,
        smoke_imports=True,
    )
    assert passed is True
    assert failures == []


def test_dedup_gather_lazy_import() -> None:
    from xtrax import DedupGather

    assert DedupGather.__name__ == "DedupGather"


def test_audit_public_api_fails_on_all_lazy_mismatch(tmp_path: Path) -> None:
    _write_lazy_root_init(
        tmp_path,
        all_names=["Trainer", "Engine"],
        lazy_map={"Trainer": "xtrax.training"},
    )
    config_path = _write_contract(tmp_path)

    passed, failures = audit_public_api(
        root=tmp_path,
        config_path=config_path,
        smoke_imports=False,
    )
    assert passed is False
    assert any("__all__ / _LAZY mismatch" in item for item in failures)


def test_audit_public_api_fails_on_missing_subpackage_init(tmp_path: Path) -> None:
    contract = load_public_api_contract(CONFIG_PATH)
    _write_lazy_root_init(
        tmp_path,
        all_names=list(contract.tier1_exports),
        lazy_map={name: "xtrax.training" for name in contract.tier1_exports},
    )
    config_path = _write_contract(tmp_path)

    passed, failures = audit_public_api(
        root=tmp_path,
        config_path=config_path,
        smoke_imports=False,
    )
    assert passed is False
    assert any("missing subpackage init" in item for item in failures)


def test_smoke_resolve_exports_reports_missing_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModule:
        def __getattr__(self, name: str) -> object:
            raise AttributeError(name)

    monkeypatch.setattr(
        "scripts.audit_public_api.importlib.import_module",
        lambda name: FakeModule(),
    )
    failures = smoke_resolve_exports("xtrax", ("DedupGather",))
    assert any("AttributeError" in item for item in failures)


def test_script_subprocess_exits_zero() -> None:
    result = subprocess.run(
        ["uv", "run", "python", "scripts/audit_public_api.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "PASS: public API contract" in result.stdout
