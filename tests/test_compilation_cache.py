"""Tests for xtrax.compilation_cache (spec .praxia/docs/specs/261119_compilation-cache-resolver.md).

Unit tests resolve in-process with a scrubbed environment. Tests that must observe what JAX
actually writes run a short CPU-only subprocess, because JAX reads JAX_COMPILATION_CACHE_DIR at
import and initialises its cache once per process.
"""

import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from xtrax import compilation_cache as cc
from xtrax.compilation_cache import (
    CacheDecision,
    apply_decision,
    resolve_compilation_cache,
    resolve_policy,
)

CACHE_VARS = (
    cc.ENV_VAR,
    cc.JAX_ENV_VAR,
    cc.POLICY_ENV_VAR,
    cc.MIN_SECS_ENV_VAR,
)


@pytest.fixture(autouse=True)
def _clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No cache env vars, an empty user-config home, and a cwd with no pyproject above it."""
    for name in CACHE_VARS:
        monkeypatch.delenv(name, raising=False)
    xdg = tmp_path / "xdg"
    xdg.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr(cc, "_last_applied", None)
    return work


def _user_config(text: str) -> Path:
    path = Path(os.environ["XDG_CONFIG_HOME"]) / "xtrax" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _pyproject(work: Path, text: str) -> Path:
    path = work / "pyproject.toml"
    path.write_text("[project]\nname = 'p'\n\n[tool.xtrax]\n" + text)
    return path


# --- placement and policy order (T1, T1b, T1c, T2, T3, T10) -----------------------------------


def test_unconfigured_is_none_default(_clean: Path) -> None:
    d = resolve_compilation_cache()
    assert (d.directory, d.placement, d.source) == (None, None, "none")
    assert (d.policy, d.policy_source) == ("on", "default")
    assert not d.configured


def test_placement_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    user_dir, jax_dir, xtrax_dir, arg_dir = (tmp_path / n for n in ("u", "j", "x", "a"))
    cfg = _user_config(f"compilation_cache_dir = '{user_dir}'\n")
    d = resolve_compilation_cache()
    assert (d.directory, d.source, d.source_path) == (user_dir, "user_config", cfg)

    monkeypatch.setenv(cc.JAX_ENV_VAR, str(jax_dir))
    assert resolve_compilation_cache().source == "env_jax"
    monkeypatch.setenv(cc.ENV_VAR, str(xtrax_dir))
    d = resolve_compilation_cache()
    assert (d.directory, d.source) == (xtrax_dir, "env_xtrax")
    d = resolve_compilation_cache(str(arg_dir))
    assert (d.directory, d.source, d.policy_source) == (arg_dir, "arg", "arg")


def test_policy_order(_clean: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _user_config("compilation_cache = 'off'\n")
    assert resolve_policy() == ("off", "user_config")
    _pyproject(_clean, "compilation_cache = 'on'\n")
    assert resolve_policy() == ("on", "pyproject")
    monkeypatch.setenv(cc.POLICY_ENV_VAR, "OFF")
    assert resolve_policy() == ("off", "env_xtrax")


def test_explicit_path_overrides_project_off_and_explicit_off_wins(
    _clean: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pyproject(_clean, "compilation_cache = 'off'\n")
    d = resolve_compilation_cache(str(tmp_path / "c"))
    assert (d.directory, d.policy, d.policy_source) == (tmp_path / "c", "on", "arg")

    monkeypatch.setenv(cc.ENV_VAR, str(tmp_path / "x"))
    _user_config(f"compilation_cache_dir = '{tmp_path / 'u'}'\n")
    d = resolve_compilation_cache("off")
    assert (d.directory, d.placement, d.policy) == (None, None, "off")


@pytest.mark.parametrize(
    ("raw", "expected"), [("ON", "on"), ("Off", "off"), ("none", "off"), ("", "off")]
)
def test_parse_policy_accepts(raw: str, expected: str) -> None:
    assert cc._parse_policy(raw) == expected


@pytest.mark.parametrize("raw", ["yes", "1", "/some/path"])
def test_parse_policy_refuses(raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(cc.POLICY_ENV_VAR, raw)
    with pytest.raises(ValueError, match=cc.POLICY_ENV_VAR):
        resolve_policy()


@pytest.mark.parametrize("var", [cc.ENV_VAR, cc.JAX_ENV_VAR])
@pytest.mark.parametrize("token", ["", "none", "OFF"])
def test_env_off_stops_the_search(
    var: str, token: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _user_config(f"compilation_cache_dir = '{tmp_path / 'u'}'\n")
    monkeypatch.setenv(var, token)
    d = resolve_compilation_cache()
    assert d.directory is None and d.configured


def test_user_config_off_token() -> None:
    _user_config("compilation_cache_dir = 'none'\n")
    d = resolve_compilation_cache()
    assert d.directory is None and d.source == "user_config"


def test_ambiguous_empty_path_is_refused() -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_compilation_cache(Path(""))


def test_relative_xtrax_env_and_user_config_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(cc.ENV_VAR, "rel/cache")
    with pytest.raises(ValueError, match=cc.ENV_VAR):
        resolve_compilation_cache()
    monkeypatch.delenv(cc.ENV_VAR)
    cfg = _user_config("compilation_cache_dir = 'rel'\n")
    with pytest.raises(ValueError, match=str(cfg)):
        resolve_compilation_cache()


def test_relative_jax_env_resolves_against_cwd_with_warning(
    _clean: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(cc.JAX_ENV_VAR, "rel")
    with caplog.at_level(logging.WARNING, logger=cc.__name__):
        d = resolve_compilation_cache()
    assert d.directory == _clean / "rel"
    assert "relative" in caplog.text


def test_malformed_user_config_raises() -> None:
    _user_config("compilation_cache_dir = [\n")
    with pytest.raises(ValueError, match="malformed TOML"):
        resolve_compilation_cache()


def test_pyproject_placement_is_refused(_clean: Path, tmp_path: Path) -> None:
    path = _pyproject(_clean, f"compilation_cache_dir = '{tmp_path / 'c'}'\n")
    with pytest.raises(ValueError, match=str(path)):
        resolve_compilation_cache()


def test_min_compile_secs_layers(_clean: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_compilation_cache().min_compile_secs == cc.DEFAULT_MIN_COMPILE_SECS
    _pyproject(_clean, "compilation_cache_min_compile_secs = 0\n")
    d = resolve_compilation_cache()
    assert (d.min_compile_secs, d.min_compile_secs_source) == (0.0, "pyproject")
    assert resolve_compilation_cache(min_compile_secs=2).min_compile_secs == 2.0


# --- record and apply (T12, T13, no-op) ------------------------------------------------------


def test_record_distinguishes_policy_off_from_no_placement(
    _clean: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(cc.JAX_ENV_VAR, str(tmp_path / "j"))
    _pyproject(_clean, "compilation_cache = 'off'\n")
    held_off = resolve_compilation_cache()
    assert held_off.directory is None
    assert held_off.as_record()["placement"] == str(tmp_path / "j")
    assert (held_off.policy, held_off.policy_source) == ("off", "pyproject")
    assert "policy: off from pyproject" in held_off.describe()
    json.dumps(held_off.as_record())

    monkeypatch.delenv(cc.JAX_ENV_VAR)
    (_clean / "pyproject.toml").unlink()
    assert resolve_compilation_cache().as_record()["placement"] is None


def test_apply_unconfigured_leaves_jax_alone() -> None:
    with patch("jax.config.update") as update:
        apply_decision(resolve_compilation_cache())
    update.assert_not_called()


def test_group_writable_dir_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    target = tmp_path / "shared"
    target.mkdir(mode=0o775)
    target.chmod(0o775)
    with caplog.at_level(logging.WARNING, logger=cc.__name__):
        cc.prepare_directory(target)
    assert "group- or world-writable" in caplog.text
    caplog.clear()
    target.chmod(0o700)
    with caplog.at_level(logging.WARNING, logger=cc.__name__):
        cc.prepare_directory(target)
    assert caplog.text == ""


def test_path_that_is_a_file_is_refused(tmp_path: Path) -> None:
    f = tmp_path / "file"
    f.write_text("x")
    with pytest.raises(ValueError, match="not a directory"):
        cc.prepare_directory(f)


def test_jax_exposes_cache_initialized_flag() -> None:
    """T9: the private flag the conflict check reads still exists in the resolved jax."""
    from jax._src import compilation_cache as jax_cc

    assert isinstance(getattr(jax_cc, "_cache_initialized", None), bool)


# --- what JAX actually writes (subprocess; T4, T4b, T5, T6, T6b, T11) -------------------------

_PRELUDE = textwrap.dedent(
    """
    import json, os, sys
    import jax, jax.numpy as jnp
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    from xtrax.compilation_cache import enable_compilation_cache

    def compile_once(k):
        return float(jax.jit(lambda x: x * k + 1.0)(jnp.ones(3)).sum())

    def entries(path):
        return sorted(os.listdir(path)) if os.path.isdir(path) else None
    """
)


def _run(code: str, cwd: Path, **env_extra: str) -> dict[str, Any]:
    env = {k: v for k, v in os.environ.items() if k not in CACHE_VARS}
    env.update({"JAX_PLATFORMS": "cpu", **env_extra})
    proc = subprocess.run(
        [sys.executable, "-c", _PRELUDE + textwrap.dedent(code)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("token", ["none", "off", "OFF"])
def test_jax_env_off_token_does_not_create_a_relative_cache(_clean: Path, token: str) -> None:
    """T4: JAX takes 'none'/'off' as a relative path; off must clear the flag before compiling."""
    out = _run(
        """
        d = enable_compilation_cache(min_compile_secs=0)
        val = compile_once(2.0)
        print(json.dumps({"flag": jax.config.jax_compilation_cache_dir, "val": val,
                          "dirs": sorted(os.listdir("."))}))
        """,
        _clean,
        JAX_COMPILATION_CACHE_DIR=token,
    )
    assert out["val"] == 9.0
    assert out["flag"] is None
    assert token not in out["dirs"]


def test_xtrax_env_off_clears_inherited_jax_dir(_clean: Path, tmp_path: Path) -> None:
    """T4b: placement off at layer 2 must clear JAX's import-time flag from layer 3."""
    jax_dir = tmp_path / "jaxdir"
    out = _run(
        f"""
        d = enable_compilation_cache(min_compile_secs=0)
        compile_once(3.0)
        print(json.dumps({{"dir": None if d.directory is None else str(d.directory),
                          "flag": jax.config.jax_compilation_cache_dir,
                          "entries": entries({str(jax_dir)!r})}}))
        """,
        _clean,
        XTRAX_COMPILATION_CACHE_DIR="",
        JAX_COMPILATION_CACHE_DIR=str(jax_dir),
    )
    assert out["dir"] is None and out["flag"] is None
    assert not out["entries"]


def test_enabled_cache_writes_entries(_clean: Path, tmp_path: Path) -> None:
    """T5: positive control for every 'no entries' assertion in this file."""
    target = tmp_path / "jax_cache"
    out = _run(
        f"""
        enable_compilation_cache({str(target)!r}, min_compile_secs=0)
        compile_once(4.0)
        print(json.dumps({{"entries": entries({str(target)!r})}}))
        """,
        _clean,
    )
    assert out["entries"]


def test_project_off_beats_injected_jax_dir(_clean: Path, tmp_path: Path) -> None:
    """T11: pyproject off holds under an injected JAX_COMPILATION_CACHE_DIR; env on reverts it."""
    jax_dir = tmp_path / "injected"
    _pyproject(_clean, "compilation_cache = 'off'\n")
    code = f"""
        d = enable_compilation_cache(min_compile_secs=0)
        compile_once(5.0)
        print(json.dumps({{"policy": d.policy, "flag": jax.config.jax_compilation_cache_dir,
                          "entries": entries({str(jax_dir)!r})}}))
        """
    off = _run(code, _clean, JAX_COMPILATION_CACHE_DIR=str(jax_dir))
    assert off["policy"] == "off" and off["flag"] is None and not off["entries"]
    on = _run(code, _clean, JAX_COMPILATION_CACHE_DIR=str(jax_dir), XTRAX_COMPILATION_CACHE="on")
    assert on["policy"] == "on" and on["entries"]


@pytest.mark.parametrize("second", ["B", "off"])
def test_change_after_init_raises_or_resets(_clean: Path, tmp_path: Path, second: str) -> None:
    """T6/T6b: after the cache initialises, a different decision conflicts; reset applies it."""
    a, b = tmp_path / "A", tmp_path / "B"
    new = str(b) if second == "B" else "off"
    out = _run(
        f"""
        enable_compilation_cache({str(a)!r}, min_compile_secs=0)
        compile_once(6.0)
        n_a = len(entries({str(a)!r}))
        try:
            enable_compilation_cache({new!r}, min_compile_secs=0)
            raised = False
        except RuntimeError:
            raised = True
        enable_compilation_cache({new!r}, min_compile_secs=0, on_conflict="reset")
        compile_once(7.0)
        print(json.dumps({{"raised": raised, "n_a_before": n_a,
                          "n_a_after": len(entries({str(a)!r})),
                          "b": entries({str(b)!r})}}))
        """,
        _clean,
    )
    assert out["raised"]
    assert out["n_a_after"] == out["n_a_before"]
    if second == "B":
        assert out["b"]


# --- executors and generated scripts (T7, T7b, T7c, T8) --------------------------------------


def _pool_cache_unit(row: Any) -> dict[str, Any]:
    """Pool unit: compile one kernel and report the worker's JAX cache flag."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    k = float(row.seed)
    val = float(jax.jit(lambda x: x * k + 1.0)(jnp.ones(3)).sum())
    return {"val": val, "flag": str(jax.config.jax_compilation_cache_dir)}


def test_pool_workers_receive_the_parent_decision(
    _clean: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T7: (a) positive control writes entries; (b) parent policy off beats inherited env."""
    from tests.campaign.executors.test_local import _create_mock_manifest
    from xtrax.campaign.executors.local import LocalProcessPoolExecutor

    on_dir = tmp_path / "on"
    decision = resolve_compilation_cache(str(on_dir), min_compile_secs=0)
    LocalProcessPoolExecutor(max_workers=1, cache_dir=decision).run(
        _create_mock_manifest(num_rows=2, campaign_id="cc_on"),
        _pool_cache_unit,
        output_dir=str(tmp_path / "out_on"),
    )
    assert any(on_dir.iterdir())

    injected = tmp_path / "injected"
    monkeypatch.setenv(cc.JAX_ENV_VAR, str(injected))
    _pyproject(_clean, "compilation_cache = 'off'\ncompilation_cache_min_compile_secs = 0\n")
    executor = LocalProcessPoolExecutor(max_workers=1)
    summary = executor.run(
        _create_mock_manifest(num_rows=2, campaign_id="cc_off"),
        _pool_cache_unit,
        output_dir=str(tmp_path / "out_off"),
    )
    assert summary.failed_units == 0
    assert executor.cache_decision is not None and executor.cache_decision.policy == "off"
    assert not injected.exists() or not any(injected.iterdir())


def test_sequential_executor_applies_a_decision_before_the_first_unit(tmp_path: Path) -> None:
    """T7b: the default campaign executor applies a decision before any unit compiles."""
    from tests.campaign.executors.test_local import _create_mock_manifest
    from xtrax.campaign.executors import local

    events: list[str] = []

    def fake_enable(cache: Any = None) -> CacheDecision:
        events.append(f"enable:{cache}")
        return resolve_compilation_cache()

    def unit(row: Any) -> dict[str, Any]:
        events.append("unit")
        return {"v": 1.0}

    with patch.object(local, "enable_compilation_cache", fake_enable):
        local.LocalSequentialExecutor(cache_dir="off").run(
            _create_mock_manifest(num_rows=2, campaign_id="cc_seq"),
            unit,
            output_dir=str(tmp_path / "out"),
        )
    assert events[0] == "enable:off"
    assert events.count("unit") == 2


def test_slurm_script_bakes_policy_never_local_placement(_clean: Path, tmp_path: Path) -> None:
    """T7c: policy is baked from the project; a user-config placement never reaches the script."""
    from tests.campaign.executors.test_slurm import _create_slurm_manifest, _write_manifest_file
    from xtrax.campaign.executors.slurm import SlurmArrayExecutor

    local_placement = tmp_path / "laptop_cache"
    _user_config(f"compilation_cache_dir = '{local_placement}'\n")
    manifest = _create_slurm_manifest(num_rows=2)
    mfile = _write_manifest_file(tmp_path, manifest)

    def script(**kwargs: Any) -> str:
        return SlurmArrayExecutor(**kwargs).generate_script(
            manifest=manifest,
            output_dir=tmp_path / "out",
            manifest_path=mfile,
            fn_spec="mypkg.units:predict",
        )

    text = script()
    assert str(local_placement) not in text
    assert f"export {cc.POLICY_ENV_VAR}=" not in text

    _pyproject(_clean, "compilation_cache = 'off'\n")
    text = script()
    assert f"export {cc.POLICY_ENV_VAR}=off" in text
    assert str(local_placement) not in text

    text = script(compilation_cache="/orcd/compute/x/jax_cache")
    assert f"export {cc.ENV_VAR}=/orcd/compute/x/jax_cache" in text
    with pytest.raises(ValueError, match="absolute"):
        SlurmArrayExecutor(compilation_cache="rel/cache")


def test_sweep_no_longer_writes_a_cwd_cache(_clean: Path) -> None:
    """T8: the deprecated sweep verb stops writing .xtrax/jax_cache when nothing is configured."""
    from tests.cli.test_sweep_verb import test_sweep_empty_axes

    mp = pytest.MonkeyPatch()
    try:
        test_sweep_empty_axes(_clean, mp)
    finally:
        mp.undo()
    assert not (_clean / ".xtrax" / "jax_cache").exists()
    changelog = Path(__file__).resolve().parents[1] / "changelog.d"
    assert any("jax_cache" in p.read_text() for p in changelog.glob("*.md"))
