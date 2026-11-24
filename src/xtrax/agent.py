"""One startup for consumer CLI and MCP entry points.

Every entry point runs the same stages in the same order: ``jax_cache.configure`` (the
persistent JAX compilation cache), then ``cisternal.init`` (telemetry), then dispatch.

Use ``run_cli`` for a cyclopts console script, ``serve`` for a FastMCP server, and ``start``
when the consumer owns its own dispatch (for example a typer runner that starts telemetry
itself, or a script; pass ``telemetry=False`` there).

    # cyclopts console script: cyclopts exits with the command's status by default
    def main() -> None:
        agent.run_cli(app)

    # in tests or library callers, get the command's return value instead of exiting
    result = agent.run_cli(app, ["fit", "x.npz"], app_kwargs={"result_action": "return_value"})

    # FastMCP
    def main() -> None:
        agent.serve(build_server())  # install CisternalMiddleware in build_server

Ordering: the cache is configured before telemetry starts, so a bad cache setting is reported
as a usage error before any cisternal thread exists. ``run_cli`` and ``serve`` print
``<prog>: jax cache configuration error: ...`` and exit 2 for a cache error, and print a hint
naming ``xtrax[plugin]`` and exit 1 when cisternal is missing. Errors from ``cisternal.init``
and from the app or server propagate unchanged.

``start()`` converts nothing: it raises ``JaxCacheConfigError``/``JaxCacheDirError`` and
``MissingExtraError`` directly.

Cost: ``telemetry=True`` needs the ``xtrax[plugin]`` extra (cisternal, cyclopts, fastmcp) and
starts cisternal threads. cisternal does not support ``fork``, so use ``spawn`` or
``forkserver`` for any ``multiprocessing`` in the process. This module imports nothing from
cisternal, cyclopts or fastmcp at module level.
"""

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from xtrax import jax_cache

__all__ = ["MissingExtraError", "run_cli", "serve", "start"]

_MISSING_EXTRA = (
    "`xtrax.agent` needs cisternal (Python >= 3.13), which is not installed.\n"
    "Install the extra, e.g.:  uv tool install --force 'xtrax[plugin]'"
)


class MissingExtraError(ImportError):
    """cisternal is not installed; the message names the ``xtrax[plugin]`` extra."""


def _configure_stage(**cache_kwargs: Any) -> jax_cache.CacheSetting:
    return jax_cache.configure(**cache_kwargs)


def _telemetry_stage(telemetry: bool, init_kwargs: Mapping[str, Any] | None) -> None:
    if not telemetry:
        return
    try:
        import cisternal
    except ModuleNotFoundError as exc:
        if exc.name == "cisternal":
            raise MissingExtraError(_MISSING_EXTRA) from exc
        raise
    cisternal.init(**(dict(init_kwargs) if init_kwargs else {}))


def _prog(obj: Any) -> str:
    name = getattr(obj, "name", None)
    if isinstance(name, str) and name:
        return name
    if isinstance(name, tuple) and name:
        return str(name[0])
    return Path(sys.argv[0]).name or "xtrax"


def _boot(
    obj: Any,
    *,
    telemetry: bool,
    init_kwargs: Mapping[str, Any] | None,
    cache_kwargs: Mapping[str, Any],
) -> None:
    try:
        _configure_stage(**cache_kwargs)
    except (ValueError, OSError) as exc:
        print(f"{_prog(obj)}: jax cache configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    try:
        _telemetry_stage(telemetry, init_kwargs)
    except MissingExtraError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


def start(
    *,
    telemetry: bool = True,
    init_kwargs: Mapping[str, Any] | None = None,
    **cache_kwargs: Any,
) -> jax_cache.CacheSetting:
    """Apply the jax cache, then start cisternal telemetry. Returns the cache setting.

    Errors propagate unconverted (see the module docstring).
    """
    setting = _configure_stage(**cache_kwargs)
    _telemetry_stage(telemetry, init_kwargs)
    return setting


def run_cli(
    app: Any,
    argv: Sequence[str] | None = None,
    *,
    telemetry: bool = True,
    init_kwargs: Mapping[str, Any] | None = None,
    app_kwargs: Mapping[str, Any] | None = None,
    **cache_kwargs: Any,
) -> Any:
    """Run startup, then ``app(argv, **app_kwargs)``. ``argv=None`` means ``sys.argv[1:]``."""
    _boot(app, telemetry=telemetry, init_kwargs=init_kwargs, cache_kwargs=cache_kwargs)
    tokens = list(sys.argv[1:] if argv is None else argv)
    return app(tokens, **(dict(app_kwargs) if app_kwargs else {}))


def serve(
    server: Any,
    *,
    telemetry: bool = True,
    init_kwargs: Mapping[str, Any] | None = None,
    run_kwargs: Mapping[str, Any] | None = None,
    **cache_kwargs: Any,
) -> None:
    """Run startup, then ``server.run(**run_kwargs)``."""
    _boot(server, telemetry=telemetry, init_kwargs=init_kwargs, cache_kwargs=cache_kwargs)
    server.run(**(dict(run_kwargs) if run_kwargs else {}))
