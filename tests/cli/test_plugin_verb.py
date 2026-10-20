"""`xtrax plugin` forwards raw argv to cisternal's plugin sub-app; cisternal stays lazy."""

from __future__ import annotations

import subprocess
import sys

import pytest

from xtrax.cli import entrypoint


def _run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["xtrax", *argv])
    try:
        entrypoint.main()
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def test_plugin_forwards_argv_unparsed(monkeypatch: pytest.MonkeyPatch) -> None:
    cisternal_plugin = pytest.importorskip("cisternal.plugin")
    seen: dict[str, object] = {}

    def fake_plugin_app(spec):  # noqa: ANN001, ANN202
        seen["spec"] = spec

        def run(args: list[str]) -> None:
            seen["args"] = args

        return run

    monkeypatch.setattr(cisternal_plugin, "plugin_app", fake_plugin_app)
    argv = ["install", "--dry-run", "--help"]
    assert _run_main(monkeypatch, ["plugin", *argv]) == 0
    assert seen["args"] == argv
    spec = seen["spec"]
    assert (spec.name, spec.package, spec.cli) == ("xtrax", "xtrax", "xtrax")


def test_plugin_help_exits_0_and_lists_subcommands(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("cisternal.plugin")
    assert _run_main(monkeypatch, ["plugin", "--help"]) == 0
    out = capsys.readouterr().out
    for sub in ("install", "update", "export", "info"):
        assert sub in out


def test_plugin_without_extra_explains_and_exits_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A None entry in sys.modules makes `import cisternal.plugin` raise ImportError.
    monkeypatch.setitem(sys.modules, "cisternal", None)
    monkeypatch.setitem(sys.modules, "cisternal.plugin", None)
    assert _run_main(monkeypatch, ["plugin", "info"]) == 1
    assert "xtrax[plugin]" in capsys.readouterr().err


@pytest.mark.parametrize(
    "probe",
    [
        "import xtrax.cli.entrypoint, xtrax.cli.registry, xtrax.cli.plan",
        "import xtrax",
    ],
)
def test_other_entrypoints_do_not_import_cisternal(probe: str) -> None:
    code = f"import sys\n{probe}\nprint('cisternal' in sys.modules)\n"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_xtrax_help_advertises_plugin_verb(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """tyro cannot list the plugin verb (it is routed before tyro), so --help says it."""
    assert _run_main(monkeypatch, ["--help"]) == 0
    out = capsys.readouterr().out
    assert "xtrax plugin install" in out
    assert "xtrax[plugin]" in out


def test_plugin_with_no_checkout_and_no_snapshot_fails_cleanly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Outside a source checkout with no packaged snapshot, cisternal names the gap and exits 1."""
    import importlib.resources

    cisternal_app = pytest.importorskip("cisternal.plugin.app")

    class _NoSnapshot:
        def joinpath(self, *_parts: str) -> _NoSnapshot:
            return self

        def is_file(self) -> bool:
            return False

        def __str__(self) -> str:
            return "<no snapshot>"

    monkeypatch.setattr(cisternal_app, "find_checkout_manifest", lambda spec: None)
    monkeypatch.setattr(importlib.resources, "files", lambda _package: _NoSnapshot())
    caplog.set_level("ERROR")
    assert _run_main(monkeypatch, ["plugin", "install", "--dry-run"]) == 1
    assert "no plugin bundle" in caplog.text
