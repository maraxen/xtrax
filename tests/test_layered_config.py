"""Tests for xtrax.config.resolve_layered and resolve_memory_budget."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from xtrax.config import resolve_layered, resolve_memory_budget
from xtrax.tiling.estimators import DEFAULT_DEVICE_MEMORY_BYTES

ENV = "XTRAX_TEST_LAYERED"
APP = "xtrax_layered_demo"


@pytest.fixture
def proj(tmp_path, monkeypatch):
    xdg = tmp_path / "xdg"
    xdg.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv(f"{APP.upper()}_MEMORY_BUDGET_BYTES", raising=False)
    root = tmp_path / "proj"
    root.mkdir()
    return root


def _pyproject(directory, body: str) -> None:
    (directory / "pyproject.toml").write_text(body)


def _user_config(body: str) -> None:
    path = Path(os.environ["XDG_CONFIG_HOME"]) / APP / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def test_arg_precedes_env(proj, monkeypatch) -> None:
    monkeypatch.setenv(ENV, "from-env")
    _pyproject(proj, f'[tool.{APP}]\ncolor = "from-project"\n')
    value, source = resolve_layered(APP, "color", arg="from-arg", env_var=ENV, start=proj)
    assert value == "from-arg"
    assert source == "arg"


def test_env_precedes_pyproject(proj, monkeypatch) -> None:
    monkeypatch.setenv(ENV, "from-env")
    _pyproject(proj, f'[tool.{APP}]\ncolor = "from-project"\n')
    value, source = resolve_layered(APP, "color", env_var=ENV, start=proj, default="fallback")
    assert value == "from-env"
    assert source == "env"


def test_pyproject_precedes_user_config(proj) -> None:
    _pyproject(proj, f'[tool.{APP}]\ncolor = "from-project"\n')
    _user_config('color = "from-user"\n')
    value, source = resolve_layered(APP, "color", env_var=ENV, start=proj, default="fallback")
    assert value == "from-project"
    assert source == "pyproject"


def test_user_config_precedes_default(proj) -> None:
    _user_config('color = "from-user"\n')
    value, source = resolve_layered(APP, "color", env_var=ENV, start=proj, default="fallback")
    assert value == "from-user"
    assert source == "user_config"


def test_default_when_nothing_is_configured(proj) -> None:
    value, source = resolve_layered(APP, "color", env_var=ENV, start=proj, default="fallback")
    assert value == "fallback"
    assert source == "default"


def test_source_label_for_each_layer(proj, monkeypatch) -> None:
    """Each deciding layer reports its own source string."""
    assert resolve_layered(APP, "color", arg=1, start=proj)[1] == "arg"

    monkeypatch.setenv(ENV, "env-value")
    assert resolve_layered(APP, "color", env_var=ENV, start=proj)[1] == "env"
    monkeypatch.delenv(ENV)

    _pyproject(proj, f"[tool.{APP}]\ncolor = 2\n")
    assert resolve_layered(APP, "color", env_var=ENV, start=proj)[1] == "pyproject"

    # A nearer table that lacks the key does not hide a user-config value,
    # and it does not inherit a parent project's key.
    parent = proj.parent
    _pyproject(parent, f'[tool.{APP}]\ncolor = "parent"\n')
    _pyproject(proj, f"[tool.{APP}]\nother = 1\n")
    _user_config('color = "from-user"\n')
    value, source = resolve_layered(APP, "color", env_var=ENV, start=proj, default="fallback")
    assert source == "user_config"
    assert value == "from-user"


def test_parent_pyproject_is_used_when_nearer_files_lack_the_table(proj) -> None:
    child = proj / "child"
    child.mkdir()
    _pyproject(child, '[project]\nname = "child"\n')
    _pyproject(proj, f'[tool.{APP}]\ncolor = "parent"\n')
    value, source = resolve_layered(APP, "color", start=child, default="fallback")
    assert value == "parent"
    assert source == "pyproject"


@pytest.mark.parametrize("raw", ["", "none", "NONE", "  None  "])
def test_empty_or_none_env_disables_the_layer(proj, monkeypatch, raw: str) -> None:
    monkeypatch.setenv(ENV, raw)
    _pyproject(proj, f'[tool.{APP}]\ncolor = "from-project"\n')
    value, source = resolve_layered(APP, "color", env_var=ENV, start=proj, default="fallback")
    assert value == "from-project"
    assert source == "pyproject"


def test_none_env_is_not_parsed(proj, monkeypatch) -> None:
    """'none' disables the env layer before parse runs."""
    monkeypatch.setenv(ENV, "none")
    value, source = resolve_layered(APP, "n", env_var=ENV, start=proj, parse=int, default=0)
    assert value == 0
    assert source == "default"


def test_parse_applies_to_env_not_to_arg(proj, monkeypatch) -> None:
    monkeypatch.setenv(ENV, "7")
    assert resolve_layered(APP, "n", arg="1", env_var=ENV, start=proj, parse=int) == ("1", "arg")
    assert resolve_layered(APP, "n", env_var=ENV, start=proj, parse=int) == (7, "env")


def test_malformed_pyproject_raises(proj) -> None:
    _pyproject(proj, "this is not [valid toml\n")
    with pytest.raises(ValueError, match="malformed TOML"):
        resolve_layered(APP, "color", start=proj, default="fallback")


def test_malformed_user_config_raises(proj) -> None:
    _user_config("this is not [valid toml\n")
    with pytest.raises(ValueError, match="malformed TOML"):
        resolve_layered(APP, "color", start=proj, default="fallback")


def test_resolve_memory_budget_uses_mocked_device(proj, monkeypatch) -> None:
    seen: dict[str, float] = {}

    def fake_budget(fraction: float = 0.9, device=None) -> int:
        seen["fraction"] = fraction
        return 1234

    monkeypatch.setattr("xtrax.tiling.estimators.device_memory_budget", fake_budget)
    value, source = resolve_memory_budget(APP, start=proj, headroom=0.5)
    assert value == 1234
    assert source == "device"
    assert seen["fraction"] == 0.5


def test_resolve_memory_budget_configured_value_skips_device(proj, monkeypatch) -> None:
    def fake_budget(fraction: float = 0.9, device=None) -> int:
        raise AssertionError("device layer should not be queried")

    monkeypatch.setattr("xtrax.tiling.estimators.device_memory_budget", fake_budget)
    monkeypatch.setenv(f"{APP.upper()}_MEMORY_BUDGET_BYTES", "555")
    value, source = resolve_memory_budget(APP, start=proj, headroom=0.5)
    assert value == 555
    assert source == "env"

    monkeypatch.delenv(f"{APP.upper()}_MEMORY_BUDGET_BYTES")
    _pyproject(proj, f"[tool.{APP}]\nmemory_budget_bytes = 777\n")
    value, source = resolve_memory_budget(APP, start=proj, headroom=0.25)
    assert value == 777
    assert source == "pyproject"


def test_resolve_memory_budget_device_failure_logs_default(proj, monkeypatch, caplog) -> None:
    import xtrax.config as config_mod

    def no_stats(fraction: float = 0.9, device=None) -> int:
        raise RuntimeError("no stats")

    monkeypatch.setattr("xtrax.tiling.estimators.device_memory_budget", no_stats)
    config_mod._memory_budget_default_logged = False
    with caplog.at_level(logging.INFO, logger="xtrax.config"):
        value, source = resolve_memory_budget(APP, start=proj, headroom=0.5)
    assert source == "default"
    assert value == int(DEFAULT_DEVICE_MEMORY_BYTES * 0.5)
    assert any("4 GiB" in record.message for record in caplog.records)


def test_resolve_memory_budget_arg_is_absolute(proj, monkeypatch) -> None:
    def fake_budget(fraction: float = 0.9, device=None) -> int:
        raise AssertionError("device layer should not be queried")

    monkeypatch.setattr("xtrax.tiling.estimators.device_memory_budget", fake_budget)
    value, source = resolve_memory_budget(APP, arg=4096, start=proj, headroom=0.5)
    assert value == 4096
    assert source == "arg"
