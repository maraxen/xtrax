"""Policy axis and executor wiring for xtrax.jax_cache (spec 261119_compilation-cache-resolver).

Unit tests resolve in-process with a scrubbed environment. Tests that must observe what JAX
actually writes run a short CPU-only subprocess, because JAX reads JAX_COMPILATION_CACHE_DIR at
import and initialises its cache once per process.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from xtrax import jax_cache
from xtrax.jax_cache import (
    ENV_DIR,
    ENV_JAX_DIR,
    ENV_MIN_COMPILE_SECS,
    ENV_POLICY,
    CacheSetting,
    JaxCacheConfigError,
    policy_source,
    resolve,
)

_CACHE_VARS = (
    ENV_DIR,
    ENV_JAX_DIR,
    ENV_POLICY,
    ENV_MIN_COMPILE_SECS,
    "XTRAX_JAX_CACHE_MIN_ENTRY_BYTES",
    "XTRAX_JAX_CACHE_MAX_BYTES",
    "XTRAX_JAX_CACHE_NAMESPACE",
)


@pytest.fixture(autouse=True)
def work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No cache env vars, an empty user-config home, and a cwd with no pyproject above it."""
    for name in _CACHE_VARS:
        monkeypatch.delenv(name, raising=False)
    xdg = tmp_path / "xdg"
    xdg.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    root = tmp_path / "work"
    root.mkdir()
    monkeypatch.chdir(root)
    return root


def _user_config(body: str) -> Path:
    path = Path(os.environ["XDG_CONFIG_HOME"]) / "xtrax" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def _pyproject(root: Path, body: str) -> Path:
    path = root / "pyproject.toml"
    path.write_text("[project]\nname = 'p'\n\n[tool.xtrax]\n" + body)
    return path


# ---------------------------------------------------------------- policy axis


def test_policy_order(work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default on < user config < pyproject < env < argument."""
    assert policy_source(start=work) == ("on", "default")
    _user_config('jax_compilation_cache = "off"\n')
    assert policy_source(start=work)[0] == "off"
    _pyproject(work, 'jax_compilation_cache = "on"\n')
    assert policy_source(start=work)[1].startswith("project config")
    monkeypatch.setenv(ENV_POLICY, "OFF")
    assert policy_source(start=work) == ("off", f"${ENV_POLICY}")
    assert policy_source("on", start=work) == ("on", "explicit argument")


@pytest.mark.parametrize(("raw", "expected"), [("ON", "on"), ("Off", "off"), ("none", "off")])
def test_policy_words(raw: str, expected: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Policy words parse case-insensitively."""
    monkeypatch.setenv(ENV_POLICY, raw)
    assert policy_source()[0] == expected


@pytest.mark.parametrize("raw", ["yes", "1", "/a/path"])
def test_unknown_policy_word_raises(raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anything but on/off/none/empty raises naming the variable."""
    monkeypatch.setenv(ENV_POLICY, raw)
    with pytest.raises(JaxCacheConfigError, match=ENV_POLICY):
        policy_source()


def test_explicit_path_beats_project_off_and_explicit_off_beats_everything(
    work: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit directory is a policy-on for that call; an explicit off wins over every layer."""
    _pyproject(work, 'jax_compilation_cache = "off"\n')
    setting = resolve(tmp_path / "c", start=work)
    assert setting.enabled and setting.policy_source == "explicit argument"

    monkeypatch.setenv(ENV_DIR, str(tmp_path / "x"))
    _user_config(f'jax_compilation_cache_dir = "{tmp_path / "u"}"\n')
    off = resolve("off", start=work)
    assert (off.enabled, off.base_dir, off.policy) == (False, None, "off")


def test_record_distinguishes_policy_off_from_no_placement(
    work: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run held off by project policy keeps the injected directory in its record."""
    monkeypatch.setenv(ENV_JAX_DIR, str(tmp_path / "injected"))
    _pyproject(work, 'jax_compilation_cache = "off"\n')
    held = resolve(start=work)
    record = held.as_record()
    assert record["dir"] is None and record["base_dir"] == str(tmp_path / "injected")
    assert record["policy"] == "off" and record["policy_source"].startswith("project config")
    json.dumps(record)

    monkeypatch.delenv(ENV_JAX_DIR)
    (work / "pyproject.toml").unlink()
    assert resolve(start=work).as_record()["base_dir"] is None


def test_ambiguous_empty_path_is_refused(work: Path) -> None:
    """Path('') equals Path('.'), so neither can mean 'off'."""
    with pytest.raises(JaxCacheConfigError, match="ambiguous"):
        resolve(Path(""), start=work)


# ---------------------------------------------------------------- what JAX writes (subprocess)

_PRELUDE = textwrap.dedent(
    """
    import json, os
    import jax, jax.numpy as jnp
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    from xtrax import jax_cache

    def compile_once(k):
        return float(jax.jit(lambda x: x * k + 1.0)(jnp.ones(3)).sum())

    def files(path):
        return sorted(str(p) for p in __import__("pathlib").Path(path).rglob("*-cache"))
    """
)


def _run(code: str, cwd: Path, **env_extra: str) -> dict[str, Any]:
    env = {k: v for k, v in os.environ.items() if k not in _CACHE_VARS}
    env.update(
        {
            "JAX_PLATFORMS": "cpu",
            "XDG_CONFIG_HOME": os.environ["XDG_CONFIG_HOME"],
            ENV_MIN_COMPILE_SECS: "0",
            **env_extra,
        }
    )
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


@pytest.mark.parametrize("word", ["none", "off"])
def test_jax_env_disable_word_never_writes_a_relative_cache(work: Path, word: str) -> None:
    """Red on #278's head: JAX_COMPILATION_CACHE_DIR=none created ./none under the cwd."""
    out = _run(
        """
        jax_cache.configure()
        val = compile_once(2.0)
        print(json.dumps({"flag": jax.config.jax_compilation_cache_dir, "val": val,
                          "dirs": sorted(os.listdir("."))}))
        """,
        work,
        JAX_COMPILATION_CACHE_DIR=word,
    )
    assert out["val"] == 9.0  # a compile demonstrably ran
    assert not out["flag"]
    assert word not in out["dirs"]


def test_xtrax_env_off_clears_an_inherited_jax_directory(work: Path, tmp_path: Path) -> None:
    """Directory off at the xtrax layer must clear the flag JAX read from its own variable."""
    injected = tmp_path / "injected"
    out = _run(
        f"""
        s = jax_cache.configure()
        compile_once(3.0)
        print(json.dumps({{"enabled": s.enabled, "flag": jax.config.jax_compilation_cache_dir,
                          "files": files({str(injected)!r})}}))
        """,
        work,
        XTRAX_JAX_CACHE_DIR="",
        JAX_COMPILATION_CACHE_DIR=str(injected),
    )
    assert not out["enabled"] and not out["flag"] and not out["files"]


def test_project_off_beats_an_injected_jax_directory(work: Path, tmp_path: Path) -> None:
    """A project's off holds under a myxcel-style injected directory; env on reverts it.

    The second run is the positive control: the same compile does write entries.
    """
    injected = tmp_path / "injected"
    _pyproject(work, 'jax_compilation_cache = "off"\n')
    code = f"""
        s = jax_cache.configure()
        compile_once(5.0)
        print(json.dumps({{"policy": s.policy, "flag": jax.config.jax_compilation_cache_dir,
                          "files": files({str(injected)!r})}}))
        """
    off = _run(code, work, JAX_COMPILATION_CACHE_DIR=str(injected))
    assert off["policy"] == "off" and not off["flag"] and not off["files"]
    on = _run(code, work, JAX_COMPILATION_CACHE_DIR=str(injected), XTRAX_JAX_CACHE="on")
    assert on["policy"] == "on" and on["files"]


def test_off_after_a_compile_stops_writes(work: Path, tmp_path: Path) -> None:
    """Clearing the flag alone keeps writing to the built cache; apply() resets it first."""
    a = tmp_path / "A"
    out = _run(
        f"""
        jax_cache.configure({str(a)!r}, namespace="none")
        compile_once(6.0)
        before = len(files({str(a)!r}))
        jax_cache.configure("off")
        compile_once(7.0)
        print(json.dumps({{"before": before, "after": len(files({str(a)!r}))}}))
        """,
        work,
    )
    assert out["before"] > 0
    assert out["after"] == out["before"]


# ---------------------------------------------------------------- executors and scripts


def _pool_cache_unit(row: Any) -> dict[str, Any]:
    """Pool unit: compile one kernel and report the worker's JAX cache flag."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    k = float(row.seed)
    val = float(jax.jit(lambda x: x * k + 1.0)(jnp.ones(3)).sum())
    return {"val": val, "flag": str(jax.config.jax_compilation_cache_dir)}


def test_pool_workers_receive_the_parent_setting(
    work: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(a) positive control writes entries; (b) parent policy off beats the inherited env."""
    from tests.campaign.executors.test_local import _create_mock_manifest
    from xtrax.campaign.executors.local import LocalProcessPoolExecutor

    on_dir = tmp_path / "on"
    setting = resolve(on_dir, start=work, min_compile_time_secs=0, namespace="none")
    LocalProcessPoolExecutor(max_workers=1, cache_dir=setting).run(
        _create_mock_manifest(num_rows=2, campaign_id="jc_on"),
        _pool_cache_unit,
        output_dir=str(tmp_path / "out_on"),
    )
    assert list(on_dir.rglob("*-cache"))

    injected = tmp_path / "injected"
    monkeypatch.setenv(ENV_JAX_DIR, str(injected))
    monkeypatch.setenv(ENV_MIN_COMPILE_SECS, "0")
    _pyproject(work, 'jax_compilation_cache = "off"\n')
    executor = LocalProcessPoolExecutor(max_workers=1)
    summary = executor.run(
        _create_mock_manifest(num_rows=2, campaign_id="jc_off"),
        _pool_cache_unit,
        output_dir=str(tmp_path / "out_off"),
    )
    assert summary.failed_units == 0
    assert executor.cache_setting is not None and executor.cache_setting.policy == "off"
    assert not list(injected.rglob("*-cache"))


def test_sequential_executor_applies_a_setting_before_the_first_unit(tmp_path: Path) -> None:
    """The default campaign executor (and every Slurm task) applies a setting before any unit."""
    from tests.campaign.executors.test_local import _create_mock_manifest
    from xtrax.campaign.executors import local

    events: list[str] = []

    def fake_apply(setting: CacheSetting) -> CacheSetting:
        events.append(f"apply:{setting.policy}")
        return setting

    def unit(row: Any) -> dict[str, Any]:
        events.append("unit")
        return {"v": 1.0}

    with patch.object(local.jax_cache, "apply", fake_apply):
        local.LocalSequentialExecutor(cache_dir="off").run(
            _create_mock_manifest(num_rows=2, campaign_id="jc_seq"),
            unit,
            output_dir=str(tmp_path / "out"),
        )
    assert events[0] == "apply:off"
    assert events.count("unit") == 2


def test_slurm_script_bakes_policy_never_local_placement(work: Path, tmp_path: Path) -> None:
    """Policy is baked from the project; a user-config directory never reaches the script."""
    from tests.campaign.executors.test_slurm import _create_slurm_manifest, _write_manifest_file
    from xtrax.campaign.executors.slurm import SlurmArrayExecutor

    local_dir = tmp_path / "laptop_cache"
    _user_config(f'jax_compilation_cache_dir = "{local_dir}"\n')
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
    assert str(local_dir) not in text
    assert f"export {ENV_POLICY}=" not in text

    _pyproject(work, 'jax_compilation_cache = "off"\n')
    text = script()
    assert f"export {ENV_POLICY}=off" in text
    assert str(local_dir) not in text

    text = script(compilation_cache="/orcd/compute/x/jax_cache")
    assert f"export {ENV_DIR}=/orcd/compute/x/jax_cache" in text
    with pytest.raises(ValueError, match="absolute"):
        SlurmArrayExecutor(compilation_cache="rel/cache")


def test_sweep_no_longer_writes_a_cwd_cache(work: Path) -> None:
    """The deprecated sweep verb writes no .xtrax/jax_cache when nothing is configured."""
    from tests.cli.test_sweep_verb import test_sweep_empty_axes

    mp = pytest.MonkeyPatch()
    try:
        test_sweep_empty_axes(work, mp)
    finally:
        mp.undo()
    assert not (work / ".xtrax" / "jax_cache").exists()


def test_xtrax_pyproject_holds_no_cache_directory() -> None:
    """xtrax's own committed pyproject must not carry a machine path (it would be refused)."""
    import tomllib

    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text())
    assert "jax_compilation_cache_dir" not in data.get("tool", {}).get("xtrax", {})
    assert jax_cache.KEY_DIR == "jax_compilation_cache_dir"
