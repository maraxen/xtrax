"""Output-dir resolver for the ``xtrax profile`` verb (spec 261016 X3).

Lives in ``cli/`` rather than ``profiling/``: the profiling package is an
AST-enforced leaf that may not import ``xtrax.config``.

Layers, first match wins, and the winning layer is reported:

1. The explicit argument (source ``"arg"``).
2. ``XTRAX_PROFILE_DIR`` (source ``"env"``).
3. ``profile_dir`` under ``[tool.xtrax]`` in the nearest ``pyproject.toml``
   (source ``"pyproject"``). The key is flat, not ``[tool.xtrax.profile]``:
   ``resolve_layered`` addresses one ``[tool.<app>]`` table and cannot reach a
   nested one without changing its contract.
4. ``profile_dir`` at the top level of ``${XDG_CONFIG_HOME:-~/.config}/xtrax/config.toml``
   (source ``"user_config"``).
5. ``RunSpec.output_root`` when a run spec is given (source ``"run_spec"``).
6. Off: no directory, print-only (source ``"off"``).

An empty value or ``none`` at any layer disables output. A relative path is
rejected, since the directory must not depend on the caller's cwd. No default
directory is ever chosen here.
"""

import os
from pathlib import Path
from typing import Any

from xtrax.config import resolve_layered

APP = "xtrax"
KEY = "profile_dir"
ENV_VAR = "XTRAX_PROFILE_DIR"


def _parse_dir(value: Any) -> Path | None:
    """Validate a configured directory. ``None`` means the layer disables output."""
    if not isinstance(value, str):
        raise TypeError(f"{KEY} must be a string path, got {type(value).__name__}")
    stripped = value.strip()
    if stripped == "" or stripped.lower() == "none":
        return None
    path = Path(stripped).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{KEY} must be absolute or start with '~', got {value!r}")
    return path


def resolve_profile_dir(
    explicit: str | Path | None = None,
    *,
    run_spec: Any = None,
    start: str | Path | None = None,
) -> tuple[Path | None, str]:
    """Resolve where a profile record is written, and which layer decided.

    Args:
        explicit: Directory passed by the caller (for example ``--out``). ``None``
            means not provided; an empty string or ``none`` disables output.
        run_spec: Optional object with an ``output_root`` attribute (a
            ``RunSpec``). Read only when no earlier layer decides.
        start: Directory to walk upward from when looking for ``pyproject.toml``.
            Defaults to the current directory.

    Returns:
        ``(directory, source)``. ``directory`` is ``None`` when output is
        disabled or off. ``source`` is one of ``"arg"``, ``"env"``,
        ``"pyproject"``, ``"user_config"``, ``"run_spec"``, ``"off"``.

    Raises:
        ValueError: A configured value is relative, the environment value is
            relative, or a TOML file on the walk is malformed. The message
            names the file or variable.
    """
    if explicit is not None:
        if isinstance(explicit, str) and explicit.strip().lower() in ("", "none"):
            return None, "arg"
        return Path(explicit).expanduser(), "arg"

    if ENV_VAR in os.environ:
        try:
            return _parse_dir(os.environ[ENV_VAR]), "env"
        except (TypeError, ValueError) as exc:
            raise ValueError(f"${ENV_VAR}: {exc}") from exc

    value, source = resolve_layered(APP, KEY, start=start, parse=_parse_dir, default=None)
    if source != "default":
        return value, source

    output_root = getattr(run_spec, "output_root", None) if run_spec is not None else None
    if output_root is not None:
        return Path(output_root), "run_spec"

    return None, "off"


__all__ = ["ENV_VAR", "KEY", "resolve_profile_dir"]
