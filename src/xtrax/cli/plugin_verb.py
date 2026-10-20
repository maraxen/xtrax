"""`xtrax plugin install|update|export|info` -- cisternal's shared plugin sub-app.

Same surface as `bth plugin`, `mrx plugin` and `jaxlint plugin`. The raw argv (help flag
included) is forwarded to cisternal's ``plugin_app``, so cisternal owns the help text, the
error wording and the exit codes. The bundle resolves from the source checkout's
``.praxia/manifest.toml`` when run from one, else from the packaged
``xtrax/agent_plugin.json`` snapshot (regenerate with
``cisternal assets snapshot --out src/xtrax/agent_plugin.json``).

cisternal is the optional ``plugin`` extra and is imported only when this verb runs, so
``import xtrax`` and the other verbs never load it.

This verb is not in REGISTRY: tyro parses subcommands and cannot forward raw argv (for
``--help``), so ``entrypoint.main`` routes ``xtrax plugin ...`` here before tyro runs.
"""

from __future__ import annotations

import sys

PLUGIN_NAME = "xtrax"

_MISSING_EXTRA = (
    "`xtrax plugin` needs cisternal (Python >= 3.13), which is not installed.\n"
    "Install the extra, e.g.:  uv tool install --force 'xtrax[plugin]'"
)


def run_plugin(argv: list[str]) -> None:
    """Forward *argv* (everything after ``plugin``) to cisternal's plugin sub-app."""
    try:
        from cisternal.plugin import PluginSpec, plugin_app
    except ImportError:
        print(_MISSING_EXTRA, file=sys.stderr)
        raise SystemExit(1) from None
    spec = PluginSpec(name=PLUGIN_NAME, package="xtrax", cli="xtrax")
    plugin_app(spec)(list(argv))


__all__ = ["PLUGIN_NAME", "run_plugin"]
