"""Lazy function loader for CLI import paths.

This module provides load_fn(), which resolves import-path strings like
'module.path:symbol' into callables, deferring the actual import until
the function is called.

Stable public contract: `load_fn`/`CLIImportError` are domain-agnostic
(no coupling to TrainConfig, RunSpec, or any xtrax-specific shape) and are
exported at `xtrax.cli.__all__` for downstream packages to import directly —
unlike `TrainConfig`/`run_from_config` in `xtrax.cli.run`/`config`, which
stay cli-private (training-loop-specific). `graph-plan` (`xtrax.cli.graph_plan_verb`)
already relies on this same `module.path:symbol` convention to resolve a
graph node's `callable_ref` to the identical live function object `load_fn`
would resolve from a bare import-path string — this is a real shared
primitive within xtrax itself, not just incidentally reusable.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from xtrax.cli.errors import CLIImportError


def load_fn(path: str) -> Callable[..., Any]:
    """Resolve an import-path string to a callable.

    Parses a path string in the format 'module.path:symbol' and performs
    the actual import and attribute lookup at call time (not at module load).
    Returns the resolved callable directly.

    Args:
        path: Import-path string in format 'module.path:symbol'.
              For example: 'json:dumps', 'pathlib:Path', 'mylib.mod:my_func'.

    Returns:
        The resolved callable from the module.

    Raises:
        CLIImportError: If the path is malformed (no colon, empty parts),
                       if the module cannot be imported, or if the target
                       attribute does not exist.

    Example:
        >>> fn = load_fn('json:dumps')
        >>> fn({'key': 'value'})  # Returns '{"key": "value"}'
    """

    # Parse on the LAST ':'
    if ":" not in path:
        raise CLIImportError(
            f"malformed import path '{path}': no ':' separator found. "
            f"Expected format: 'module.path:symbol'"
        )

    parts = path.rsplit(":", 1)
    if len(parts) != 2:
        raise CLIImportError(f"malformed import path '{path}': split on ':' failed")

    module_name, attr_name = parts

    # Check for empty parts
    if not module_name:
        raise CLIImportError(f"malformed import path '{path}': empty module name")

    if not attr_name:
        raise CLIImportError(f"malformed import path '{path}': empty attribute name")

    # Perform the import at function-call time (not at module load)
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise CLIImportError(
            f"cannot import module '{module_name}' from path '{path}': {e}. "
            f"Is '{module_name}' installed and importable (on sys.path)?"
        ) from e

    try:
        func = getattr(mod, attr_name)
    except AttributeError as e:
        raise CLIImportError(
            f"module '{module_name}' has no attribute '{attr_name}' (from path '{path}')"
        ) from e

    return func
