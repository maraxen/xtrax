"""Tests for xtrax.config.resolve_layered and resolve_memory_budget."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import pytest

from xtrax.config import _start_dir, resolve_layered, resolve_memory_budget
from xtrax.tiling.estimators import DEFAULT_DEVICE_MEMORY_BYTES

ENV = "XTRAX_TEST_LAYERED"
APP = "xtrax_layered_demo"


@pytest.fixture
def proj(tmp_path, monkeypatch):
    # Keep the pyproject walk inside the test tmp dir. An absolute start still
    # climbs real ancestors, but a forgotten start= uses this cwd.
    monkeypatch.chdir(tmp_path)
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


@pytest.mark.parametrize("arg", [0, ""])
def test_falsy_arg_wins_over_later_layers(proj, monkeypatch, arg) -> None:
    """0 and "" are explicit. Only None means the arg layer is unset."""
    monkeypatch.setenv(ENV, "from-env")
    _pyproject(proj, f'[tool.{APP}]\ncolor = "from-project"\n')
    value, source = resolve_layered(
        APP, "color", arg=arg, env_var=ENV, start=proj, default="fallback"
    )
    assert value == arg
    assert source == "arg"


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


def test_absent_user_config_key_falls_through_to_default(proj) -> None:
    """A user-config file that omits the key is not a hit."""
    _user_config('other = "present"\n')
    value, source = resolve_layered(APP, "color", env_var=ENV, start=proj, default="fallback")
    assert value == "fallback"
    assert source == "default"


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


def test_start_file_resolves_to_its_parent(proj) -> None:
    """A file start walks from the parent directory, not the file itself."""
    child = proj / "child"
    child.mkdir()
    marker = child / "note.txt"
    marker.write_text("x")
    _pyproject(child, f'[tool.{APP}]\ncolor = "from-parent"\n')
    _pyproject(proj, f'[tool.{APP}]\ncolor = "from-grandparent"\n')
    value, source = resolve_layered(APP, "color", start=marker, default="fallback")
    assert value == "from-parent"
    assert source == "pyproject"
    assert _start_dir(marker) == marker.resolve().parent


def test_default_start_is_cwd(proj, monkeypatch) -> None:
    _pyproject(proj, f'[tool.{APP}]\ncolor = "from-cwd"\n')
    monkeypatch.chdir(proj)
    value, source = resolve_layered(APP, "color", default="fallback")
    assert value == "from-cwd"
    assert source == "pyproject"


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


def test_parse_applies_to_pyproject_string(proj) -> None:
    _pyproject(proj, f'[tool.{APP}]\nn = "12"\n')
    assert resolve_layered(APP, "n", start=proj, parse=int, default=0) == (12, "pyproject")


def test_parse_applies_to_user_config_string(proj) -> None:
    _user_config('n = "12"\n')
    assert resolve_layered(APP, "n", start=proj, parse=int, default=0) == (12, "user_config")


def test_parse_failure_names_pyproject_path(proj) -> None:
    _pyproject(proj, f'[tool.{APP}]\nn = "nope"\n')
    path = proj.resolve() / "pyproject.toml"
    with pytest.raises(ValueError, match=re.escape(str(path))):
        resolve_layered(APP, "n", start=proj, parse=int, default=0)


def test_parse_failure_names_user_config_path(proj) -> None:
    _user_config('n = "nope"\n')
    path = Path(os.environ["XDG_CONFIG_HOME"]) / APP / "config.toml"
    with pytest.raises(ValueError, match=re.escape(str(path))):
        resolve_layered(APP, "n", start=proj, parse=int, default=0)


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
    def no_stats(fraction: float = 0.9, device=None) -> int:
        raise RuntimeError("no stats")

    monkeypatch.setattr("xtrax.tiling.estimators.device_memory_budget", no_stats)
    monkeypatch.setattr("xtrax.config._memory_budget_default_logged", False)
    with caplog.at_level(logging.INFO, logger="xtrax.config"):
        first_value, first_source = resolve_memory_budget(APP, start=proj, headroom=0.5)
        second_value, second_source = resolve_memory_budget(APP, start=proj, headroom=0.5)
    expected = int(DEFAULT_DEVICE_MEMORY_BYTES * 0.5)
    assert (first_value, first_source) == (expected, "default")
    assert (second_value, second_source) == (expected, "default")
    messages = [record.message for record in caplog.records if "4 GiB" in record.message]
    assert len(messages) == 1


def test_unrelated_device_exception_propagates(proj, monkeypatch) -> None:
    """Only RuntimeError (no bytes_limit) falls back. Other errors propagate."""

    def boom(fraction: float = 0.9, device=None) -> int:
        raise TypeError("allocator returned garbage")

    monkeypatch.setattr("xtrax.tiling.estimators.device_memory_budget", boom)
    with pytest.raises(TypeError, match="allocator returned garbage"):
        resolve_memory_budget(APP, start=proj)


def _toml_budget(value: object) -> str:
    if value is True:
        literal = "true"
    elif isinstance(value, str):
        literal = f'"{value}"'
    else:
        literal = str(value)
    return f"memory_budget_bytes = {literal}\n"


@pytest.mark.parametrize("layer", ["arg", "env", "pyproject", "user_config"])
@pytest.mark.parametrize("raw", [0, -1, True, "abc", 1.5])
def test_invalid_memory_budget_names_its_source(proj, monkeypatch, layer: str, raw: object) -> None:
    env_name = f"{APP.upper()}_MEMORY_BUDGET_BYTES"
    if layer == "arg":
        kwargs: dict = {"arg": raw}
        match = "argument"
    elif layer == "env":
        monkeypatch.setenv(env_name, "True" if raw is True else str(raw))
        kwargs = {}
        match = re.escape(f"${env_name}")
    elif layer == "pyproject":
        _pyproject(proj, f"[tool.{APP}]\n{_toml_budget(raw)}")
        kwargs = {}
        match = re.escape(str(proj.resolve() / "pyproject.toml"))
    else:
        _user_config(_toml_budget(raw))
        path = Path(os.environ["XDG_CONFIG_HOME"]) / APP / "config.toml"
        kwargs = {}
        match = re.escape(str(path))
    with pytest.raises(ValueError, match=match):
        resolve_memory_budget(APP, start=proj, **kwargs)


@pytest.mark.parametrize("xdg", [None, "", "   "])
def test_xdg_unset_or_empty_uses_home_config(proj, monkeypatch, tmp_path, xdg: str | None) -> None:
    home = tmp_path / "home"
    config = home / ".config" / APP / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text('color = "from-home"\n')
    monkeypatch.setenv("HOME", str(home))
    if xdg is None:
        monkeypatch.delenv("XDG_CONFIG_HOME")
    else:
        monkeypatch.setenv("XDG_CONFIG_HOME", xdg)
    value, source = resolve_layered(APP, "color", start=proj, default="fallback")
    assert value == "from-home"
    assert source == "user_config"


def test_resolve_memory_budget_arg_is_absolute(proj, monkeypatch) -> None:
    def fake_budget(fraction: float = 0.9, device=None) -> int:
        raise AssertionError("device layer should not be queried")

    monkeypatch.setattr("xtrax.tiling.estimators.device_memory_budget", fake_budget)
    value, source = resolve_memory_budget(APP, arg=4096, start=proj, headroom=0.5)
    assert value == 4096
    assert source == "arg"
