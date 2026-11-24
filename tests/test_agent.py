"""Tests for xtrax.agent: one startup (jax cache, then cisternal, then dispatch).

Patches target module attributes that ``xtrax.agent`` looks up at call time:
``xtrax.jax_cache.configure`` and ``cisternal.init``. The real ``configure`` is kept in
``_REAL_CONFIGURE`` so the isolation tests can restore it.
"""

from __future__ import annotations

import builtins
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import pytest

from xtrax import agent, jax_cache

_REAL_CONFIGURE = jax_cache.configure
_SENTINEL = object()


class Recorder:
    """Spies for jax_cache.configure and cisternal.init, plus a stub dispatch target."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.configure_kwargs: dict | None = None
        self.init_kwargs: dict | None = None
        self.configure_exc: BaseException | None = None
        self.init_exc: BaseException | None = None
        self.app_kwargs: dict | None = None
        self.run_kwargs: dict | None = None
        self.app_argv: list[str] | None = None


@pytest.fixture
def rec(monkeypatch) -> Recorder:
    """Spy configure; spy cisternal.init when cisternal is installed (it is in the dev group)."""
    r = Recorder()

    def fake_configure(*args, **kwargs):
        r.events.append("configure")
        r.configure_kwargs = kwargs
        if r.configure_exc is not None:
            raise r.configure_exc
        return _SENTINEL

    monkeypatch.setattr(jax_cache, "configure", fake_configure)

    cisternal = pytest.importorskip("cisternal")

    def fake_init(*args, **kwargs):
        r.events.append("cisternal.init")
        r.init_kwargs = kwargs
        if r.init_exc is not None:
            raise r.init_exc
        return None

    monkeypatch.setattr(cisternal, "init", fake_init)
    return r


class StubApp:
    """Stands in for a cyclopts App: callable with (tokens, **kw); has a name tuple."""

    name = ("demo",)

    def __init__(self, rec: Recorder, result=None, raises: BaseException | None = None):
        self.rec = rec
        self.result = result
        self.raises = raises

    def __call__(self, tokens, **kwargs):
        self.rec.events.append("dispatch")
        self.rec.app_argv = list(tokens)
        self.rec.app_kwargs = kwargs
        if self.raises is not None:
            raise self.raises
        return self.result


class StubServer:
    name = "demo-server"

    def __init__(self, rec: Recorder):
        self.rec = rec

    def run(self, **kwargs):
        self.rec.events.append("dispatch")
        self.rec.run_kwargs = kwargs


@pytest.fixture
def isolated_cache(monkeypatch, tmp_path):
    """Use the real jax_cache.configure with no env, a private XDG dir, and restored JAX state."""
    monkeypatch.setattr(jax_cache, "configure", _REAL_CONFIGURE)
    for var in list(os.environ):
        if var.startswith("XTRAX_JAX_CACHE") or var == "JAX_COMPILATION_CACHE_DIR":
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    names = ("jax_compilation_cache_dir",)
    saved = {name: getattr(jax.config, name) for name in names}
    saved_applied = jax_cache._applied
    yield tmp_path
    for name, value in saved.items():
        jax.config.update(name, value)
    jax_cache._applied = saved_applied


# ---- order: configure, then cisternal.init, then dispatch


def test_start_order_configure_before_init(rec):
    agent.start()
    assert rec.events == ["configure", "cisternal.init"]


def test_run_cli_order(rec):
    agent.run_cli(StubApp(rec), [], app_kwargs={"result_action": "return_value"})
    assert rec.events == ["configure", "cisternal.init", "dispatch"]


def test_serve_order(rec):
    agent.serve(StubServer(rec))
    assert rec.events == ["configure", "cisternal.init", "dispatch"]


# ---- cache error -> exit 2 (only from the configure stage)


def test_cache_dir_error_exits_2_and_skips_dispatch(rec, capsys):
    rec.configure_exc = jax_cache.JaxCacheDirError("cannot create /nope")
    with pytest.raises(SystemExit) as info:
        agent.run_cli(StubApp(rec), [])
    assert info.value.code == 2
    err = capsys.readouterr().err
    assert "demo: jax cache configuration error: cannot create /nope" in err
    assert rec.events == ["configure"]


def test_real_malformed_config_exits_2(rec, isolated_cache, capsys):
    cfg_dir = isolated_cache / "xdg" / "xtrax"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.toml").write_text("this is = = not toml\n")
    with pytest.raises(SystemExit) as info:
        agent.run_cli(StubApp(rec), [], telemetry=False)
    assert info.value.code == 2
    assert "jax cache configuration error" in capsys.readouterr().err
    assert rec.app_argv is None


def test_app_value_error_is_not_exit_2(rec):
    """Control: an app's own ValueError stays a traceback, not a cache-config exit 2."""
    app = StubApp(rec, raises=ValueError("app bug"))
    with pytest.raises(ValueError, match="app bug"):
        agent.run_cli(app, [])
    assert "dispatch" in rec.events


def test_init_oserror_propagates_not_exit_2(rec, capsys):
    """Control N5: an OSError from cisternal.init is a traceback, never exit 2."""
    rec.init_exc = OSError("log_dir not writable")
    with pytest.raises(OSError, match="log_dir not writable"):
        agent.run_cli(StubApp(rec), [])
    assert "jax cache" not in capsys.readouterr().err
    assert "dispatch" not in rec.events


def test_init_valueerror_propagates_not_exit_2(rec, capsys):
    """Control N5: a ValueError from cisternal.init is a traceback, never exit 2."""
    rec.init_exc = ValueError("bad init_kwargs")
    with pytest.raises(ValueError, match="bad init_kwargs"):
        agent.serve(StubServer(rec))
    assert "jax cache" not in capsys.readouterr().err
    assert "dispatch" not in rec.events


# ---- missing cisternal -> exit 1 with the xtrax[plugin] hint


def _block_cisternal(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cisternal" or name.startswith("cisternal."):
            err = ModuleNotFoundError(f"No module named {name!r}", name="cisternal")
            assert err.name == "cisternal"
            raise err
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_missing_cisternal_exits_1_with_hint(rec, monkeypatch, capsys):
    _block_cisternal(monkeypatch)
    with pytest.raises(SystemExit) as info:
        agent.run_cli(StubApp(rec), [])
    assert info.value.code == 1
    assert "xtrax[plugin]" in capsys.readouterr().err
    assert "dispatch" not in rec.events


def test_start_raises_missing_extra_error(rec, monkeypatch):
    _block_cisternal(monkeypatch)
    with pytest.raises(agent.MissingExtraError, match=r"xtrax\[plugin\]"):
        agent.start()
    assert isinstance(agent.MissingExtraError("x"), ImportError)


def test_broken_dependency_inside_cisternal_is_not_relabelled(rec, monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cisternal":
            raise ModuleNotFoundError("No module named 'fastmcp'", name="fastmcp")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ModuleNotFoundError) as info:
        agent.start()
    assert info.value.name == "fastmcp"
    assert not isinstance(info.value, agent.MissingExtraError)


# ---- telemetry=False never touches cisternal (subprocess: sys.modules is clean)


_PROBE = textwrap.dedent(
    """
    import sys
    from xtrax import agent
    agent.start(telemetry={telemetry}, init_kwargs={init_kwargs!r})
    print("cisternal" in sys.modules)
    """
)


def _run_probe(tmp_path: Path, telemetry: bool, init_kwargs: dict) -> str:
    env = {
        k: v
        for k, v in os.environ.items()
        if not (k.startswith("XTRAX_JAX_CACHE") or k == "JAX_COMPILATION_CACHE_DIR")
    }
    env["XTRAX_JAX_CACHE_DIR"] = "none"
    env["XDG_CONFIG_HOME"] = str(tmp_path / "xdg")
    code = _PROBE.format(telemetry=telemetry, init_kwargs=init_kwargs)
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip().splitlines()[-1]


def test_telemetry_false_does_not_load_cisternal(tmp_path):
    pytest.importorskip("cisternal")
    assert _run_probe(tmp_path, False, {}) == "False"


def test_telemetry_true_loads_cisternal_positive_control(tmp_path):
    pytest.importorskip("cisternal")
    # log_dir in tmp so the real cisternal.init never writes ~/.cisternal/logs
    assert _run_probe(tmp_path, True, {"log_dir": str(tmp_path / "logs")}) == "True"


# ---- passthrough


def test_cache_kwargs_reach_configure_and_start_returns_its_value(rec):
    result = agent.start(path="/somewhere", namespace="none")
    assert rec.configure_kwargs == {"path": "/somewhere", "namespace": "none"}
    assert result is _SENTINEL


def test_init_kwargs_reach_cisternal_init(rec):
    agent.start(init_kwargs={"log_dir": "/tmp/x", "level": 10})
    assert rec.init_kwargs == {"log_dir": "/tmp/x", "level": 10}


def test_app_kwargs_reach_app(rec):
    agent.run_cli(StubApp(rec), ["go"], app_kwargs={"exit_on_error": False, "print_error": True})
    assert rec.app_kwargs == {"exit_on_error": False, "print_error": True}
    assert rec.app_argv == ["go"]


def test_run_kwargs_reach_server_run(rec):
    agent.serve(StubServer(rec), run_kwargs={"transport": "stdio"})
    assert rec.run_kwargs == {"transport": "stdio"}


def test_argv_none_uses_sys_argv(rec, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "a", "b"])
    agent.run_cli(StubApp(rec), None, app_kwargs={"result_action": "return_value"})
    assert rec.app_argv == ["a", "b"]


def test_run_cli_returns_app_value(rec):
    app = StubApp(rec, result="ok")
    assert agent.run_cli(app, [], app_kwargs={"result_action": "return_value"}) == "ok"


def test_real_cyclopts_app_return_value_and_default_exit(rec):
    cyclopts = pytest.importorskip("cyclopts")
    app = cyclopts.App(name="demo")

    @app.default
    def main(x: int = 1) -> int:
        return x + 1

    got = agent.run_cli(
        app, ["--x", "2"], telemetry=False, app_kwargs={"result_action": "return_value"}
    )
    assert got == 3
    # Control: with cyclopts' default result action the command exits, as a console script should.
    with pytest.raises(SystemExit):
        agent.run_cli(app, ["--x", "2"], telemetry=False)


# ---- import lightness: importing xtrax.agent loads none of the optional stack


def test_import_agent_loads_no_optional_stack():
    for mod in ("cisternal", "cyclopts", "fastmcp"):
        pytest.importorskip(mod)
    code = textwrap.dedent(
        """
        import sys
        import xtrax.agent  # noqa: F401
        print(any(m in sys.modules for m in ("cisternal", "cyclopts", "fastmcp")))
        """
    )
    env = dict(os.environ)
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"
    # Positive control: the three modules really import in a fresh interpreter.
    ctrl = subprocess.run(
        [sys.executable, "-c", "import cisternal, cyclopts, fastmcp"],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert ctrl.returncode == 0, ctrl.stderr


# ---- audit follow-ups: start() converts nothing; serve() converts like run_cli


@pytest.mark.parametrize(
    "exc",
    [jax_cache.JaxCacheDirError("cannot create /nope"), jax_cache.JaxCacheConfigError("bad")],
)
def test_start_raises_cache_errors_unconverted(rec, exc):
    rec.configure_exc = exc
    with pytest.raises(type(exc)):
        agent.start()
    assert rec.events == ["configure"]


def test_serve_cache_error_exits_2_and_skips_run(rec, capsys):
    rec.configure_exc = jax_cache.JaxCacheDirError("cannot create /nope")
    with pytest.raises(SystemExit) as info:
        agent.serve(StubServer(rec))
    assert info.value.code == 2
    assert "jax cache configuration error: cannot create /nope" in capsys.readouterr().err
    assert rec.events == ["configure"]


def test_serve_missing_cisternal_exits_1_with_hint(rec, monkeypatch, capsys):
    _block_cisternal(monkeypatch)
    with pytest.raises(SystemExit) as info:
        agent.serve(StubServer(rec))
    assert info.value.code == 1
    assert "xtrax[plugin]" in capsys.readouterr().err
    assert "dispatch" not in rec.events


def test_run_cli_passes_cache_and_init_kwargs(rec):
    agent.run_cli(
        StubApp(rec),
        [],
        init_kwargs={"level": 10},
        app_kwargs={"result_action": "return_value"},
        path="/somewhere",
    )
    assert rec.configure_kwargs == {"path": "/somewhere"}
    assert rec.init_kwargs == {"level": 10}


def test_serve_passes_cache_and_init_kwargs(rec):
    agent.serve(StubServer(rec), init_kwargs={"level": 20}, namespace="none")
    assert rec.configure_kwargs == {"namespace": "none"}
    assert rec.init_kwargs == {"level": 20}
