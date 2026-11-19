"""Tests for xtrax.jax_cache: layers, validation, idempotent configure, cross-process reuse."""

import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import jax
import pytest

from xtrax import jax_cache
from xtrax.jax_cache import (
    ENV_DIR,
    ENV_MAX_BYTES,
    ENV_MIN_COMPILE_SECS,
    ENV_NAMESPACE,
    JaxCacheConfigError,
    JaxCacheDirError,
    cache_dir_source,
    configure,
    host_cpu_fingerprint,
    resolve,
)

_ALL_ENV = (
    ENV_DIR,
    ENV_MIN_COMPILE_SECS,
    ENV_MAX_BYTES,
    ENV_NAMESPACE,
    "XTRAX_JAX_CACHE_MIN_ENTRY_BYTES",
    "JAX_COMPILATION_CACHE_DIR",
)


@pytest.fixture
def proj(tmp_path, monkeypatch):
    """A project directory inside tmp_path with no xtrax config anywhere on the walk."""
    monkeypatch.chdir(tmp_path)
    xdg = tmp_path / "xdg"
    xdg.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    for name in _ALL_ENV:
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "proj"
    root.mkdir()
    return root


def _pyproject(directory: Path, body: str) -> None:
    (directory / "pyproject.toml").write_text(body)


def _user_config(body: str) -> Path:
    path = Path(os.environ["XDG_CONFIG_HOME"]) / "xtrax" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


# ---------------------------------------------------------------- resolution layers


def test_unset_is_off(proj) -> None:
    assert cache_dir_source(start=proj) == (None, "unset")
    setting = resolve(start=proj)
    assert (setting.dir, setting.enabled, setting.source) == (None, False, "unset")


def test_explicit_argument_wins_over_everything(proj, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(ENV_DIR, str(tmp_path / "env"))
    _pyproject(proj, '[tool.xtrax]\njax_compilation_cache_dir = "proj-cache"\n')
    _user_config(f'jax_compilation_cache_dir = "{tmp_path / "user"}"\n')
    assert cache_dir_source(tmp_path / "arg", start=proj) == (
        tmp_path / "arg",
        "explicit argument",
    )


def test_env_wins_over_project_and_user(proj, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(ENV_DIR, str(tmp_path / "env"))
    _pyproject(proj, '[tool.xtrax]\njax_compilation_cache_dir = "proj-cache"\n')
    _user_config(f'jax_compilation_cache_dir = "{tmp_path / "user"}"\n')
    assert cache_dir_source(start=proj) == (tmp_path / "env", f"${ENV_DIR}")


def test_project_wins_over_user_and_is_relative_to_its_file(proj, tmp_path) -> None:
    _pyproject(proj, '[tool.xtrax]\njax_compilation_cache_dir = ".xtrax/jax_cache"\n')
    _user_config(f'jax_compilation_cache_dir = "{tmp_path / "user"}"\n')
    directory, source = cache_dir_source(start=proj)
    assert directory == proj / ".xtrax" / "jax_cache"
    assert source == f"project config {proj / 'pyproject.toml'}"


def test_nearest_pyproject_decides_not_a_parent(proj) -> None:
    _pyproject(proj, '[tool.xtrax]\njax_compilation_cache_dir = "outer"\n')
    inner = proj / "sub"
    inner.mkdir()
    _pyproject(inner, '[tool.xtrax]\njax_compilation_cache_dir = "inner"\n')
    assert cache_dir_source(start=inner)[0] == inner / "inner"


def test_user_config_is_the_fallback(proj, tmp_path) -> None:
    config = _user_config(f'jax_compilation_cache_dir = "{tmp_path / "user"}"\n')
    assert cache_dir_source(start=proj) == (tmp_path / "user", f"user config {config}")


def test_user_config_relative_path_is_relative_to_the_config_file(proj) -> None:
    config = _user_config('jax_compilation_cache_dir = "cache"\n')
    assert cache_dir_source(start=proj)[0] == config.parent / "cache"


def test_tilde_and_vars_expand(proj, monkeypatch) -> None:
    monkeypatch.setenv("XTRAX_TEST_SCRATCH", "/scratch/me")
    monkeypatch.setenv(ENV_DIR, "$XTRAX_TEST_SCRATCH/jax")
    assert cache_dir_source(start=proj)[0] == Path("/scratch/me/jax")
    monkeypatch.setenv(ENV_DIR, "~/jax")
    assert cache_dir_source(start=proj)[0] == Path.home() / "jax"


@pytest.mark.parametrize("value", ["none", "NONE", "", "  "])
def test_env_none_or_empty_disables_and_overrides_files(proj, monkeypatch, tmp_path, value) -> None:
    monkeypatch.setenv(ENV_DIR, value)
    _pyproject(proj, '[tool.xtrax]\njax_compilation_cache_dir = "proj-cache"\n')
    _user_config(f'jax_compilation_cache_dir = "{tmp_path / "user"}"\n')
    directory, source = cache_dir_source(start=proj)
    assert directory is None
    assert source == f"${ENV_DIR} (disabled)"
    assert resolve(start=proj).enabled is False


def test_explicit_none_disables_even_with_env(proj, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(ENV_DIR, str(tmp_path / "env"))
    assert cache_dir_source("none", start=proj) == (None, "explicit argument (disabled)")
    assert cache_dir_source("", start=proj) == (None, "explicit argument (disabled)")


def test_project_none_disables_and_overrides_user_config(proj, tmp_path) -> None:
    _pyproject(proj, '[tool.xtrax]\njax_compilation_cache_dir = "none"\n')
    _user_config(f'jax_compilation_cache_dir = "{tmp_path / "user"}"\n')
    directory, source = cache_dir_source(start=proj)
    assert directory is None and source.endswith("(disabled)")


def test_pyproject_without_the_key_falls_through_to_user_config(proj, tmp_path) -> None:
    _pyproject(proj, "[tool.xtrax]\nunrelated = 1\n")
    _user_config(f'jax_compilation_cache_dir = "{tmp_path / "user"}"\n')
    assert cache_dir_source(start=proj)[0] == tmp_path / "user"


# ---------------------------------------------------------------- failing loudly


def test_malformed_project_file_raises_naming_it(proj) -> None:
    _pyproject(proj, "[tool.xtrax\njax_compilation_cache_dir = oops\n")
    with pytest.raises(ValueError, match="pyproject.toml"):
        cache_dir_source(start=proj)


def test_malformed_user_file_raises_naming_it(proj) -> None:
    config = _user_config("this is = = not toml\n")
    with pytest.raises(ValueError, match=str(config)):
        cache_dir_source(start=proj)


def test_non_string_directory_raises(proj) -> None:
    _pyproject(proj, "[tool.xtrax]\njax_compilation_cache_dir = 7\n")
    with pytest.raises(JaxCacheConfigError, match="jax_compilation_cache_dir"):
        cache_dir_source(start=proj)


def test_malformed_file_is_not_silently_run_cold_by_configure(proj) -> None:
    _pyproject(proj, "[[[\n")
    with pytest.raises(ValueError):
        configure(start=proj)


# ---------------------------------------------------------------- knobs


def test_knobs_resolve_through_the_layers_and_record_sources(proj, monkeypatch) -> None:
    _pyproject(
        proj,
        "[tool.xtrax]\n"
        'jax_compilation_cache_dir = "c"\n'
        "jax_cache_min_compile_time_secs = 2.5\n"
        "jax_cache_max_size_bytes = 1000000\n",
    )
    monkeypatch.setenv(ENV_MIN_COMPILE_SECS, "0")
    setting = resolve(start=proj, min_entry_size_bytes=10)
    assert setting.min_compile_time_secs == 0.0  # env beats the project file
    assert setting.max_size_bytes == 1_000_000
    assert setting.min_entry_size_bytes == 10
    assert setting.knob_sources["min_compile_time_secs"] == f"${ENV_MIN_COMPILE_SECS}"
    assert setting.knob_sources["max_size_bytes"].startswith("project config")
    assert setting.knob_sources["min_entry_size_bytes"] == "explicit argument"


def test_unset_knobs_stay_none_so_jax_defaults_apply(proj) -> None:
    _pyproject(proj, '[tool.xtrax]\njax_compilation_cache_dir = "c"\n')
    setting = resolve(start=proj)
    assert (
        setting.min_compile_time_secs,
        setting.min_entry_size_bytes,
        setting.max_size_bytes,
    ) == (
        None,
        None,
        None,
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"min_compile_time_secs": -1.0}, "min_compile_time"),
        ({"max_size_bytes": -5}, "max_size"),
    ],
)
def test_negative_knobs_are_rejected(proj, kwargs, match) -> None:
    with pytest.raises(JaxCacheConfigError, match=match):
        resolve(start=proj, **kwargs)


def test_non_numeric_knob_in_env_is_rejected_naming_the_variable(proj, monkeypatch) -> None:
    monkeypatch.setenv(ENV_MIN_COMPILE_SECS, "soon")
    with pytest.raises(JaxCacheConfigError, match=ENV_MIN_COMPILE_SECS):
        resolve(start=proj)


def test_bad_namespace_is_rejected(proj) -> None:
    with pytest.raises(JaxCacheConfigError, match="jax_cache_namespace"):
        resolve(start=proj, namespace="per-user")


# ---------------------------------------------------------------- namespace


def test_default_namespace_is_the_jaxlib_version_only(proj) -> None:
    setting = resolve(tmp := proj / "base", start=proj)
    assert setting.base_dir == tmp
    assert setting.dir == tmp / f"jaxlib-{jax_cache._jaxlib_version()}"
    assert "cpu-" not in str(setting.dir)


def test_namespace_none_uses_the_directory_as_given(proj) -> None:
    setting = resolve(proj / "base", start=proj, namespace="none")
    assert setting.dir == proj / "base"


def test_namespace_cpu_flags_is_opt_in(proj) -> None:
    setting = resolve(proj / "base", start=proj, namespace="cpu-flags")
    assert setting.dir == (
        proj / "base" / f"jaxlib-{jax_cache._jaxlib_version()}" / f"cpu-{host_cpu_fingerprint()}"
    )


def _fake_cpuinfo(monkeypatch, tmp_path, text: str | None) -> None:
    """Make ``open('/proc/cpuinfo')`` read ``text`` (None: raise OSError)."""
    fixture = tmp_path / "cpuinfo"
    real_open = open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/cpuinfo":
            if text is None:
                raise OSError("masked")
            fixture.write_text(text)
            return real_open(fixture, *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)


def test_cpu_fingerprint_tracks_isa_flags_not_mitigation_flags(monkeypatch, tmp_path) -> None:
    _fake_cpuinfo(monkeypatch, tmp_path, "flags\t\t: sse avx avx2\n")
    base = host_cpu_fingerprint()
    _fake_cpuinfo(monkeypatch, tmp_path, "flags\t\t: sse avx avx2 md_clear ibpb stibp\n")
    assert host_cpu_fingerprint() == base  # kernel/microcode mitigation drift must not split
    _fake_cpuinfo(monkeypatch, tmp_path, "flags\t\t: sse avx avx2 avx512f\n")
    other = host_cpu_fingerprint()
    assert other != base and len(other) == len(base) == 8


def test_cpu_flags_namespace_fails_loudly_when_flags_are_unreadable(monkeypatch, tmp_path) -> None:
    _fake_cpuinfo(monkeypatch, tmp_path, None)
    with pytest.raises(JaxCacheConfigError, match="cpu-flags"):
        host_cpu_fingerprint()
    _fake_cpuinfo(monkeypatch, tmp_path, "model name\t: something\n")
    with pytest.raises(JaxCacheConfigError, match="cpu-flags"):
        host_cpu_fingerprint()


def test_resolve_never_imports_or_initialises_jax_backends(proj) -> None:
    code = (
        "import sys; from xtrax import jax_cache; jax_cache.resolve(); "
        "assert 'jax' not in sys.modules, 'resolve() imported jax'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=proj)
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------- configure (in-process)


@pytest.fixture
def restore_jax_state():
    """Put JAX's cache flags and xtrax's applied state back after a test that calls configure()."""
    from jax.experimental.compilation_cache import compilation_cache

    names = (
        "jax_compilation_cache_dir",
        "jax_persistent_cache_min_compile_time_secs",
        "jax_persistent_cache_min_entry_size_bytes",
        "jax_compilation_cache_max_size",
    )
    saved = {name: getattr(jax.config, name) for name in names}
    saved_applied = jax_cache._applied
    jax_cache._applied = None
    yield
    for name, value in saved.items():
        jax.config.update(name, value)
    compilation_cache.reset_cache()
    jax_cache._applied = saved_applied


def test_configure_applies_directory_and_knobs(proj, restore_jax_state, caplog) -> None:
    caplog.set_level(logging.INFO, logger="xtrax.jax_cache")
    setting = configure(
        proj / "cache",
        start=proj,
        min_compile_time_secs=0,
        min_entry_size_bytes=0,
        namespace="none",
    )
    assert setting.enabled and setting.dir == proj / "cache"
    assert setting.dir.is_dir()
    assert jax.config.jax_compilation_cache_dir == str(proj / "cache")
    assert jax.config.jax_persistent_cache_min_compile_time_secs == 0
    assert "decided by explicit argument" in caplog.text


def test_configure_is_idempotent(proj, restore_jax_state, monkeypatch) -> None:
    calls = []
    real = jax_cache._apply_to_jax
    monkeypatch.setattr(jax_cache, "_apply_to_jax", lambda s: (calls.append(s), real(s))[1])
    first = configure(proj / "cache", start=proj)
    second = configure(proj / "cache", start=proj)
    assert first == second
    assert len(calls) == 1


def _pretend_cache_started(monkeypatch) -> list[str]:
    """Make JAX look as if a compile already initialised its cache; record reset_cache calls."""
    from jax._src import compilation_cache

    resets: list[str] = []
    monkeypatch.setattr(compilation_cache, "_cache", SimpleNamespace(_path="dummy"))
    monkeypatch.setattr(compilation_cache, "reset_cache", lambda: resets.append("reset"))
    return resets


def test_configure_switches_directory_with_a_warning(
    proj, restore_jax_state, caplog, monkeypatch
) -> None:
    configure(proj / "a", start=proj, namespace="none")
    resets = _pretend_cache_started(monkeypatch)
    caplog.set_level(logging.WARNING, logger="xtrax.jax_cache")
    configure(proj / "b", start=proj, namespace="none")
    assert jax.config.jax_compilation_cache_dir == str(proj / "b")
    assert "already in use" in caplog.text
    assert resets == ["reset"]


def test_switch_before_any_compile_warns_nothing_and_does_not_reset(
    proj, restore_jax_state, caplog, monkeypatch
) -> None:
    from jax._src import compilation_cache

    monkeypatch.setattr(compilation_cache, "_cache", None)
    configure(proj / "a", start=proj, namespace="none")
    caplog.set_level(logging.WARNING, logger="xtrax.jax_cache")
    configure(proj / "b", start=proj, namespace="none")
    assert caplog.text == ""


def test_changing_only_max_size_after_start_resets_and_warns(
    proj, restore_jax_state, caplog, monkeypatch
) -> None:
    monkeypatch.setattr(jax_cache.importlib.util, "find_spec", lambda name: object())
    configure(proj / "a", start=proj, namespace="none", max_size_bytes=1_000_000)
    resets = _pretend_cache_started(monkeypatch)
    caplog.set_level(logging.WARNING, logger="xtrax.jax_cache")
    configure(proj / "a", start=proj, namespace="none", max_size_bytes=2_000_000)
    assert resets == ["reset"]
    assert "max_size_bytes=2000000" in caplog.text
    assert jax.config.jax_compilation_cache_max_size == 2_000_000


def test_idempotent_call_reapplies_when_jax_config_was_changed_elsewhere(
    proj, restore_jax_state
) -> None:
    configure(proj / "a", start=proj, namespace="none")
    jax.config.update("jax_compilation_cache_dir", str(proj / "elsewhere"))
    configure(proj / "a", start=proj, namespace="none")
    assert jax.config.jax_compilation_cache_dir == str(proj / "a")


def test_off_leaves_jaxs_own_setting_untouched(proj, restore_jax_state) -> None:
    jax.config.update("jax_compilation_cache_dir", str(proj / "set-by-jax-env"))
    setting = configure(start=proj)
    assert setting.enabled is False
    assert jax.config.jax_compilation_cache_dir == str(proj / "set-by-jax-env")
    assert configure("none", start=proj).enabled is False
    assert jax.config.jax_compilation_cache_dir == str(proj / "set-by-jax-env")


def test_unwritable_directory_fails_loudly(proj, restore_jax_state) -> None:
    blocker = proj / "a-file"
    blocker.write_text("not a directory")
    with pytest.raises(JaxCacheDirError, match="not writable"):
        configure(blocker / "cache", start=proj)
    assert jax.config.jax_compilation_cache_dir != str(blocker / "cache")


def test_world_writable_directory_warns(proj, restore_jax_state, caplog) -> None:
    shared = proj / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    caplog.set_level(logging.WARNING, logger="xtrax.jax_cache")
    configure(shared, start=proj, namespace="none")
    assert "world-writable" in caplog.text


@pytest.mark.parametrize("size", [10_000_000, 0])
def test_max_size_without_filelock_is_refused(proj, restore_jax_state, monkeypatch, size) -> None:
    monkeypatch.setattr(jax_cache.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(JaxCacheConfigError, match="filelock"):
        configure(proj / "cache", start=proj, max_size_bytes=size)


def test_unlimited_max_size_needs_no_filelock(proj, restore_jax_state, monkeypatch) -> None:
    monkeypatch.setattr(jax_cache.importlib.util, "find_spec", lambda name: None)
    assert configure(proj / "cache", start=proj, max_size_bytes=-1).enabled


# ---------------------------------------------------------------- review fixes


def test_unexpanded_variable_raises_naming_the_layer(proj, monkeypatch) -> None:
    monkeypatch.delenv("XTRAX_TEST_UNSET_SCRATCH", raising=False)
    config = _user_config('jax_compilation_cache_dir = "$XTRAX_TEST_UNSET_SCRATCH/jax_cache"\n')
    with pytest.raises(JaxCacheConfigError, match="unexpanded") as excinfo:
        cache_dir_source(start=proj)
    assert str(config) in str(excinfo.value)
    monkeypatch.setenv(ENV_DIR, "$XTRAX_TEST_UNSET_SCRATCH/jax_cache")
    with pytest.raises(JaxCacheConfigError, match=ENV_DIR):
        cache_dir_source(start=proj)
    with pytest.raises(JaxCacheConfigError, match="explicit argument"):
        cache_dir_source("~nosuchuser_xtrax/jax", start=proj)


def test_unexpanded_project_variable_names_the_pyproject(proj) -> None:
    _pyproject(proj, '[tool.xtrax]\njax_compilation_cache_dir = "$XTRAX_TEST_UNSET/c"\n')
    with pytest.raises(JaxCacheConfigError, match="pyproject.toml"):
        resolve(start=proj)


def test_jax_env_variable_beats_project_and_user_files(proj, monkeypatch, tmp_path) -> None:
    _pyproject(proj, '[tool.xtrax]\njax_compilation_cache_dir = "proj-cache"\n')
    _user_config(f'jax_compilation_cache_dir = "{tmp_path / "user"}"\n')
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", str(tmp_path / "jaxenv"))
    assert cache_dir_source(start=proj) == (tmp_path / "jaxenv", "$JAX_COMPILATION_CACHE_DIR")
    monkeypatch.setenv(ENV_DIR, str(tmp_path / "xtraxenv"))
    assert cache_dir_source(start=proj)[1] == f"${ENV_DIR}"  # xtrax's own variable is stronger
    monkeypatch.delenv(ENV_DIR)
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "")  # empty means unset, not disabled
    assert cache_dir_source(start=proj)[1].startswith("project config")


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_non_finite_float_knob_is_rejected(proj, monkeypatch, bad) -> None:
    monkeypatch.setenv(ENV_MIN_COMPILE_SECS, bad)
    with pytest.raises(JaxCacheConfigError, match=ENV_MIN_COMPILE_SECS):
        resolve(start=proj)


def test_configure_does_not_initialise_a_jax_backend(tmp_path) -> None:
    code = (
        "from xtrax import jax_cache; import jax; "
        f"jax_cache.configure({str(tmp_path / 'c')!r}); "
        "from jax._src import xla_bridge; "
        "assert not xla_bridge._backends, 'configure() initialised a backend'"
    )
    env = {k: v for k, v in os.environ.items() if k not in _ALL_ENV}
    env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path, env=env
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------- cross-process reuse

_CHILD = textwrap.dedent(
    """
    import json
    import jax
    import jax.numpy as jnp
    from jax import monitoring
    from xtrax import jax_cache

    events = []
    monitoring.register_event_listener(lambda name, **kw: events.append(name))
    setting = jax_cache.configure()

    @jax.jit
    def work(x):
        return jnp.tanh(x @ x.T).sum() + jnp.cumsum(x, axis=0).sum()

    work(jnp.ones((16, 16))).block_until_ready()
    print(json.dumps({
        "hits": events.count("/jax/compilation_cache/cache_hits"),
        "misses": events.count("/jax/compilation_cache/cache_misses"),
        "dir": str(setting.dir),
        "source": setting.source,
    }))
    """
)


def _run_child(cwd: Path, **env: str) -> dict:
    full = {k: v for k, v in os.environ.items() if k not in _ALL_ENV}
    full.pop("JAX_COMPILATION_CACHE_DIR", None)
    full.update({"JAX_PLATFORMS": "cpu", "XDG_CONFIG_HOME": str(cwd / "xdg"), **env})
    result = subprocess.run(
        [sys.executable, "-c", _CHILD],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=full,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_second_process_reuses_executables_from_the_shared_directory(tmp_path) -> None:
    cache = tmp_path / "cache"
    env = {ENV_DIR: str(cache), ENV_MIN_COMPILE_SECS: "0"}
    cold = _run_child(tmp_path, **env)
    warm = _run_child(tmp_path, **env)
    assert cold["hits"] == 0 and cold["misses"] > 0
    assert any(Path(cold["dir"]).iterdir()), "first process wrote nothing to the cache"
    assert warm["hits"] >= cold["misses"], (cold, warm)
    assert warm["misses"] == 0, (cold, warm)
    assert cold["source"] == f"${ENV_DIR}"


def _cache_files(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*-cache"))


def test_prune_corrupt_removes_truncated_and_empty_entries_only(tmp_path) -> None:
    cache = tmp_path / "cache"
    env = {ENV_DIR: str(cache), ENV_MIN_COMPILE_SECS: "0"}
    cold = _run_child(tmp_path, **env)
    entries = _cache_files(Path(cold["dir"]))
    assert len(entries) >= 2
    truncated, intact = entries[0], entries[1:]
    truncated.write_bytes(truncated.read_bytes()[: truncated.stat().st_size // 2])
    empty = Path(cold["dir"]) / "jit_dead-beef-cache"
    empty.write_bytes(b"")
    assert jax_cache.prune_corrupt(cache, min_age_secs=0, dry_run=True) == sorted(
        [truncated, empty]
    )
    assert truncated.exists() and empty.exists()  # dry run deletes nothing
    assert jax_cache.prune_corrupt(cache, min_age_secs=3600) == []  # too young: skipped
    assert jax_cache.prune_corrupt(cache, min_age_secs=0) == sorted([truncated, empty])
    assert not truncated.exists() and not empty.exists()
    assert all(path.exists() for path in intact)
    assert jax_cache.prune_corrupt(cache, min_age_secs=0) == []
    warm = _run_child(tmp_path, **env)
    assert warm["hits"] >= len(intact)


def test_prune_corrupt_without_a_configured_cache_raises(proj) -> None:
    with pytest.raises(JaxCacheConfigError, match="no JAX compilation cache"):
        jax_cache.prune_corrupt()


def test_disabled_cache_gives_no_reuse(tmp_path) -> None:
    env = {ENV_DIR: "none"}
    first = _run_child(tmp_path, **env)
    second = _run_child(tmp_path, **env)
    assert first["hits"] == second["hits"] == 0
