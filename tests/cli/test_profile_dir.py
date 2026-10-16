"""Tests for the profile output-dir resolver (spec 261016 X3).

Layers, first match wins: explicit arg > ``XTRAX_PROFILE_DIR`` > ``[tool.xtrax]
profile_dir`` in the nearest pyproject > ``~/.config/xtrax/config.toml``
``profile_dir`` > ``RunSpec.output_root`` > off (print-only).
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from xtrax.cli.profile_dir import ENV_VAR, KEY, resolve_profile_dir


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """No ambient env var and a user config dir that does not exist yet."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write_pyproject(directory: Path, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "pyproject.toml").write_text(body, encoding="utf-8")
    return directory


def _write_user_config(xdg: Path, body: str) -> None:
    (xdg / "xtrax").mkdir(parents=True, exist_ok=True)
    (xdg / "xtrax" / "config.toml").write_text(body, encoding="utf-8")


def test_key_names_are_stable():
    assert ENV_VAR == "XTRAX_PROFILE_DIR"
    assert KEY == "profile_dir"


def test_explicit_arg_wins_over_every_other_layer(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "env"))
    proj = _write_pyproject(tmp_path / "proj", f'[tool.xtrax]\n{KEY} = "{tmp_path / "pp"}"\n')
    run_spec = SimpleNamespace(output_root=tmp_path / "run")
    got = resolve_profile_dir(tmp_path / "arg", run_spec=run_spec, start=proj)
    assert got == (tmp_path / "arg", "arg")


def test_empty_explicit_arg_disables(tmp_path):
    assert resolve_profile_dir("") == (None, "arg")


def test_env_var_wins_over_pyproject_and_user_config(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "env"))
    proj = _write_pyproject(tmp_path / "proj", f'[tool.xtrax]\n{KEY} = "{tmp_path / "pp"}"\n')
    _write_user_config(tmp_path / "xdg", f'{KEY} = "{tmp_path / "user"}"\n')
    assert resolve_profile_dir(start=proj) == (tmp_path / "env", "env")


@pytest.mark.parametrize("value", ["", "   ", "none", "NONE"])
def test_env_var_empty_or_none_disables_with_env_source(monkeypatch, tmp_path, value):
    monkeypatch.setenv(ENV_VAR, value)
    proj = _write_pyproject(tmp_path / "proj", f'[tool.xtrax]\n{KEY} = "{tmp_path / "pp"}"\n')
    assert resolve_profile_dir(start=proj) == (None, "env")


def test_env_var_relative_path_fails_loudly(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "relative/dir")
    with pytest.raises(ValueError, match=ENV_VAR):
        resolve_profile_dir()


def test_pyproject_layer_is_used_when_no_arg_or_env(tmp_path):
    proj = _write_pyproject(tmp_path / "proj", f'[tool.xtrax]\n{KEY} = "{tmp_path / "pp"}"\n')
    assert resolve_profile_dir(start=proj) == (tmp_path / "pp", "pyproject")


def test_pyproject_is_found_by_walking_up(tmp_path):
    _write_pyproject(tmp_path / "proj", f'[tool.xtrax]\n{KEY} = "{tmp_path / "pp"}"\n')
    deep = tmp_path / "proj" / "a" / "b"
    deep.mkdir(parents=True)
    assert resolve_profile_dir(start=deep) == (tmp_path / "pp", "pyproject")


def test_pyproject_none_disables_and_does_not_fall_through(tmp_path):
    proj = _write_pyproject(tmp_path / "proj", f'[tool.xtrax]\n{KEY} = "none"\n')
    _write_user_config(tmp_path / "xdg", f'{KEY} = "{tmp_path / "user"}"\n')
    assert resolve_profile_dir(start=proj) == (None, "pyproject")


def test_user_config_layer_is_used_without_pyproject_or_env(tmp_path):
    _write_user_config(tmp_path / "xdg", f'{KEY} = "{tmp_path / "user"}"\n')
    empty = tmp_path / "empty"
    empty.mkdir()
    assert resolve_profile_dir(start=empty) == (tmp_path / "user", "user_config")


def test_pyproject_beats_user_config(tmp_path):
    proj = _write_pyproject(tmp_path / "proj", f'[tool.xtrax]\n{KEY} = "{tmp_path / "pp"}"\n')
    _write_user_config(tmp_path / "xdg", f'{KEY} = "{tmp_path / "user"}"\n')
    assert resolve_profile_dir(start=proj) == (tmp_path / "pp", "pyproject")


def test_run_spec_output_root_is_the_fallback(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    run_spec = SimpleNamespace(output_root=tmp_path / "run")
    assert resolve_profile_dir(run_spec=run_spec, start=empty) == (tmp_path / "run", "run_spec")


def test_run_spec_without_output_root_is_off(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    run_spec = SimpleNamespace(output_root=None)
    assert resolve_profile_dir(run_spec=run_spec, start=empty) == (None, "off")


def test_no_layer_set_is_off_print_only(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert resolve_profile_dir(start=empty) == (None, "off")


def test_malformed_pyproject_raises_naming_the_file(tmp_path):
    proj = _write_pyproject(tmp_path / "proj", "[tool.xtrax\n")
    with pytest.raises(ValueError, match="pyproject.toml"):
        resolve_profile_dir(start=proj)


def test_malformed_user_config_raises(tmp_path):
    _write_user_config(tmp_path / "xdg", "profile_dir = \n")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="config.toml"):
        resolve_profile_dir(start=empty)


def test_relative_pyproject_value_fails_loudly(tmp_path):
    proj = _write_pyproject(tmp_path / "proj", f'[tool.xtrax]\n{KEY} = "relative/dir"\n')
    with pytest.raises(ValueError, match="absolute"):
        resolve_profile_dir(start=proj)


def test_non_string_pyproject_value_fails_loudly(tmp_path):
    proj = _write_pyproject(tmp_path / "proj", f"[tool.xtrax]\n{KEY} = 3\n")
    with pytest.raises(ValueError, match="pyproject.toml"):
        resolve_profile_dir(start=proj)
