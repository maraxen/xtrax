"""Extensible plugin discovery and quarantine architecture for xtrax EDA.

Manages panel and backend entry points with deepcopy isolation, in-memory caching,
snippet validation, and testable quarantine tracking.
"""

from __future__ import annotations

import copy
import importlib.metadata
import logging
from typing import Any

from xtrax.eda.types import BackendAdapter, PanelPlugin, PlanStatsDict

logger = logging.getLogger(__name__)

_PANEL_CACHE: dict[str, PanelPlugin] | None = None
_BACKEND_CACHE: dict[str, BackendAdapter] | None = None
_QUARANTINED_PLUGINS: list[str] = []


def get_quarantined_plugins() -> list[str]:
    """Return list of plugin names that failed validation or raised exceptions."""
    return list(_QUARANTINED_PLUGINS)


def clear_quarantined_plugins() -> None:
    """Clear the quarantined plugins tracking list."""
    global _QUARANTINED_PLUGINS
    _QUARANTINED_PLUGINS = []


def reload_plugins() -> None:
    """Clear discovery caches to force re-scanning Python entry points."""
    global _PANEL_CACHE, _BACKEND_CACHE
    _PANEL_CACHE = None
    _BACKEND_CACHE = None
    clear_quarantined_plugins()


def _validate_panel_snippet(snippet: Any) -> bool:
    """Verify that a plugin-emitted snippet is a valid Vega-Lite specification dict."""
    if not isinstance(snippet, dict):
        return False
    valid_keys = {"mark", "layer", "concat", "vconcat", "hconcat"}
    if not any(k in snippet for k in valid_keys):
        return False
    return True


def _make_error_placeholder(plugin_name: str, error_msg: str) -> dict[str, Any]:
    """Produce a safe placeholder panel when a plugin fails or raises."""
    return {
        "title": f"Plugin Error: {plugin_name}",
        "width": 500,
        "height": 100,
        "data": {"values": [{}]},
        "mark": {
            "type": "text",
            "text": f"Panel '{plugin_name}' failed: {error_msg}",
            "color": "#e53e3e",
        },
    }


def execute_panel_plugin(
    plugin: PanelPlugin,
    stats: PlanStatsDict,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a panel plugin under strict deepcopy immutability quarantine.

    Args:
        plugin: The panel plugin instance.
        stats: Plan statistics dictionary.
        config: Optional configuration dictionary.

    Returns:
        Vega-Lite specification snippet, or an error placeholder if quarantined.
    """
    plugin_name = "unknown_plugin"
    try:
        plugin_name = getattr(plugin, "name", type(plugin).__name__)
        isolated_stats = copy.deepcopy(stats)
        cfg = copy.deepcopy(config) if config is not None else {}
        snippet = plugin.generate_spec(isolated_stats, cfg)
        if not _validate_panel_snippet(snippet):
            msg = "Emitted snippet is not a valid Vega-Lite panel dict"
            logger.warning("Plugin %s quarantined: %s", plugin_name, msg)
            if plugin_name not in _QUARANTINED_PLUGINS:
                _QUARANTINED_PLUGINS.append(plugin_name)
            return _make_error_placeholder(plugin_name, msg)
        return snippet
    except Exception as exc:
        msg = str(exc) or type(exc).__name__
        logger.exception("Plugin %s raised an unhandled exception: %s", plugin_name, msg)
        if plugin_name not in _QUARANTINED_PLUGINS:
            _QUARANTINED_PLUGINS.append(plugin_name)
        return _make_error_placeholder(plugin_name, msg)


def get_registered_panels() -> dict[str, PanelPlugin]:
    """Discover and return all registered panel plugins from Python entry points.

    Returns:
        Dictionary mapping panel names to PanelPlugin instances.
    """
    global _PANEL_CACHE
    if _PANEL_CACHE is not None:
        return _PANEL_CACHE

    panels: dict[str, PanelPlugin] = {}
    eps = importlib.metadata.entry_points()
    group_eps = (
        eps.select(group="xtrax.eda.panels")
        if hasattr(eps, "select")
        else eps.get("xtrax.eda.panels", [])
    )

    for ep in group_eps:
        try:
            plugin_cls = ep.load()
            instance = plugin_cls() if isinstance(plugin_cls, type) else plugin_cls
            panels[ep.name] = instance
        except Exception as exc:
            logger.warning("Failed to load panel plugin %s: %s", ep.name, exc)
            if ep.name not in _QUARANTINED_PLUGINS:
                _QUARANTINED_PLUGINS.append(ep.name)

    _PANEL_CACHE = panels
    return _PANEL_CACHE


def get_registered_backends() -> dict[str, BackendAdapter]:
    """Discover and return all registered backend adapters from Python entry points.

    Returns:
        Dictionary mapping backend names to BackendAdapter instances.
    """
    global _BACKEND_CACHE
    if _BACKEND_CACHE is not None:
        return _BACKEND_CACHE

    backends: dict[str, BackendAdapter] = {}
    eps = importlib.metadata.entry_points()
    group_eps = (
        eps.select(group="xtrax.eda.backends")
        if hasattr(eps, "select")
        else eps.get("xtrax.eda.backends", [])
    )

    for ep in group_eps:
        try:
            backend_cls = ep.load()
            instance = backend_cls() if isinstance(backend_cls, type) else backend_cls
            backends[ep.name] = instance
        except Exception as exc:
            logger.warning("Failed to load backend adapter %s: %s", ep.name, exc)

    _BACKEND_CACHE = backends
    return _BACKEND_CACHE
