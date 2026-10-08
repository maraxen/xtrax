"""Generic fail-loud TOML-config primitives (idea-003).

Domain-agnostic building blocks factored out of `xtrax.cli.config`'s
training-shaped `TrainConfig`/`load_config`: presence checks, custom-predicate
field validation, and schema-version classification, each raising a
caller-supplied ``error_cls`` rather than a fixed xtrax exception type. This
module has no dependency on `xtrax.cli` or `tyro` -- any consumer building
its own config-loading layer (not just xtrax's own CLI) can import it
directly.

``resolve_layered`` is the project-neutral five-layer lookup (argument, env,
``pyproject.toml``, per-machine user config, default). ``resolve_memory_budget``
builds a byte budget on that lookup and then the device allocator.

Spec: `.praxia/docs/specs/260715_generic-fail-loud-toml-to-dataclass-conf.md`
(contemplex session b93ead41). `xtrax.cli.config.load_config` composes these
primitives; see that module for the canonical (dog-fooded) usage.
"""

import logging
import os
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MEMORY_BUDGET_KEY = "memory_budget_bytes"
_memory_budget_default_logged = False


def load_toml_document(path: str, error_cls: type[Exception]) -> dict[str, Any]:
    """Parse a TOML file into a dict, raising ``error_cls`` on any IO/parse failure.

    Wraps both file-access failures (missing/unreadable path) and malformed-TOML
    failures into one caller-chosen exception type, so callers catch a single
    family regardless of which failure mode occurred.
    """
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except OSError as e:
        raise error_cls(f"cannot read config file '{path}': {e}") from e
    except tomllib.TOMLDecodeError as e:
        raise error_cls(f"malformed TOML in config file '{path}': {e}") from e


def require_sections(raw: dict, sections: Sequence[str], error_cls: type[Exception]) -> None:
    """Raise ``error_cls`` naming every section in ``sections`` missing from ``raw``.

    Collects and names all missing sections in one error (not just the first),
    so a caller fixing their config sees every problem at once.
    """
    missing = [section for section in sections if section not in raw]
    if missing:
        raise error_cls(f"missing required section(s): {', '.join(f'[{s}]' for s in missing)}")


def require_field(
    raw: dict, name: str, predicate: Callable[[Any], bool], error_cls: type[Exception]
) -> Any:
    """Extract ``raw[name]`` and validate it against an arbitrary predicate.

    Returns the value on success. Raises ``error_cls`` naming the field and the
    offending value on failure. ``predicate`` is an arbitrary
    ``Callable[[Any], bool]`` -- this covers custom validation (e.g. "must be a
    positive int") that a bare type-hint check cannot express.
    """
    value = raw.get(name)
    if not predicate(value):
        raise error_cls(f"field '{name}' failed validation, got: {value!r}")
    return value


@dataclass(frozen=True)
class SchemaVersionStatus:
    """Structured classification of a document's schema_version against ``current``.

    ``kind`` is an open string tag, not a fixed enum -- the extension seam for
    future states (e.g. a "deprecated but still supported" kind) is a new
    branch in `classify_schema_version`, not a change to any function's
    signature. Known kinds today: "ok", "missing", "mismatched",
    "newer_than_supported".
    """

    kind: str
    found: int | None
    current: int


def classify_schema_version(raw: dict, current: int) -> SchemaVersionStatus:
    """Classify ``raw``'s schema_version against ``current`` without raising.

    The public extension point: a caller needing custom handling for a status
    `check_schema_version` doesn't cover (e.g. warn-but-continue on a
    deprecated-but-supported version) can call this directly instead of only
    using `check_schema_version`'s raise-or-not wrapper.
    """
    found = raw.get("schema_version")
    if found is None:
        return SchemaVersionStatus(kind="missing", found=None, current=current)
    if not isinstance(found, int):
        return SchemaVersionStatus(kind="mismatched", found=None, current=current)
    if found > current:
        return SchemaVersionStatus(kind="newer_than_supported", found=found, current=current)
    if found != current:
        return SchemaVersionStatus(kind="mismatched", found=found, current=current)
    return SchemaVersionStatus(kind="ok", found=found, current=current)


_SCHEMA_VERSION_MESSAGES = {
    "missing": lambda s: "missing required field: schema_version",
    "mismatched": lambda s: f"schema_version mismatch: expected {s.current}, got {s.found!r}",
    "newer_than_supported": lambda s: (
        f"schema_version {s.found} is newer than the supported version {s.current}"
    ),
}


def check_schema_version(raw: dict, current: int, error_cls: type[Exception]) -> None:
    """Raise ``error_cls`` if ``raw``'s schema_version is missing, mismatched, or
    newer than ``current`` -- distinguishing each case in the message via
    `classify_schema_version`.
    """
    status = classify_schema_version(raw, current)
    if status.kind != "ok":
        raise error_cls(_SCHEMA_VERSION_MESSAGES[status.kind](status))


def resolve_layered(
    app: str,
    key: str,
    *,
    arg: Any = None,
    env_var: str | None = None,
    start: str | Path | None = None,
    parse: Callable[[Any], Any] | None = None,
    default: Any = None,
) -> tuple[Any, str]:
    """Resolve ``key`` through five layers and report which one decided.

    First match wins:

    1. ``arg`` when it is not ``None`` (source ``"arg"``). ``parse`` is not
       applied; ``None`` is the only "unset" sentinel, so a value of ``0`` or
       ``""`` is explicit.
    2. ``env_var`` when that variable is set (source ``"env"``). A value that
       is empty or ``none`` (any case, surrounding whitespace ignored) disables
       the layer and the search continues. The string passed on is stripped.
    3. ``key`` in the ``[tool.<app>]`` table of a ``pyproject.toml`` walked
       upward from ``start`` (default: the current directory). Source
       ``"pyproject"``. Files without that table are skipped. The nearest
       table ends the walk, so a nearer table that lacks ``key`` does not
       inherit the key from a parent project.
    4. Top-level ``key`` in ``${XDG_CONFIG_HOME:-~/.config}/<app>/config.toml``
       (source ``"user_config"``). The directory already names the app, so the
       key is not nested under another table.
    5. ``default`` (source ``"default"``).

    ``parse``, when given, is applied to the env, pyproject, and user-config
    values only. A malformed TOML file that this walk opens raises
    ``ValueError`` naming the path. A missing file is not an error.

    Args:
        app: Application name. Selects ``[tool.<app>]`` and the user-config
            directory.
        key: Field name inside that table, and the top-level key of the
            user-config file.
        arg: Explicit value. ``None`` means "not provided".
        env_var: Environment variable name. ``None`` skips the env layer.
        start: Directory (or a file, whose parent is used) to walk upward from.
        parse: Optional converter for file and env values.
        default: Value used when every configured layer is absent.

    Returns:
        ``(value, source)`` with ``source`` one of ``"arg"``, ``"env"``,
        ``"pyproject"``, ``"user_config"``, ``"default"``.

    Raises:
        ValueError: A TOML file on the walk is malformed or unreadable, or
            ``parse`` rejects the configured value. The message names the file
            or variable.
    """
    if arg is not None:
        return arg, "arg"

    if env_var is not None and env_var in os.environ:
        stripped = os.environ[env_var].strip()
        if stripped != "" and stripped.lower() != "none":
            return _apply_parse(stripped, parse, where=f"${env_var}"), "env"

    start_dir = _start_dir(start)
    from_project = _pyproject_value(app, key, start_dir)
    if from_project is not None:
        value, path = from_project
        where = f"{path}:[tool.{app}]"
        return _apply_parse(value, parse, where=where), "pyproject"

    user_path = _user_config_path(app)
    if user_path.is_file():
        data = _read_toml(user_path)
        if key in data:
            return _apply_parse(data[key], parse, where=str(user_path)), "user_config"

    return default, "default"


def resolve_memory_budget(
    app: str,
    *,
    arg: int | None = None,
    env_var: str | None = None,
    start: str | Path | None = None,
    headroom: float = 0.9,
) -> tuple[int, str]:
    """Resolve a memory budget in bytes for ``app``.

    Configured layers are absolute byte counts and are not scaled by
    ``headroom``. The key is ``memory_budget_bytes``: under ``[tool.<app>]``
    in the nearest ``pyproject.toml``, and as a top-level key in
    ``${XDG_CONFIG_HOME:-~/.config}/<app>/config.toml``. The environment
    variable defaults to ``<APP>_MEMORY_BUDGET_BYTES`` (``app`` uppercased).

    When no layer sets the key, the budget is
    ``device_memory_budget(fraction=headroom)`` (source ``"device"``). If the
    device does not report ``bytes_limit``, the budget is the documented
    4 GiB default times ``headroom`` (source ``"default"``) and that fallback
    is logged once. ``headroom`` must be in ``(0, 1]`` on the device path,
    matching ``device_memory_budget``.

    Args:
        app: Application name passed to ``resolve_layered``.
        arg: Explicit byte count. Wins outright. Must be a positive int.
        env_var: Environment variable. Defaults to
            ``<APP>_MEMORY_BUDGET_BYTES``.
        start: Start of the ``pyproject.toml`` walk.
        headroom: Fraction of the device limit, and of the 4 GiB default.
            Not applied to a configured absolute value.

    Returns:
        ``(bytes, source)``. ``source`` is one of ``resolve_layered``'s labels
        or ``"device"``.

    Raises:
        ValueError: A configured value is not a positive int, a TOML file on
            the walk is malformed, or ``headroom`` is outside ``(0, 1]`` when
            the device layer is reached.
    """
    global _memory_budget_default_logged
    if arg is not None:
        try:
            return _parse_memory_budget(arg), "arg"
        except ValueError as exc:
            raise ValueError(f"argument: {exc}") from exc

    if env_var is None:
        env_var = f"{app.upper()}_MEMORY_BUDGET_BYTES"
    value, source = resolve_layered(
        app,
        _MEMORY_BUDGET_KEY,
        env_var=env_var,
        start=start,
        parse=_parse_memory_budget,
        default=None,
    )
    if source != "default":
        return value, source

    from xtrax.tiling.estimators import DEFAULT_DEVICE_MEMORY_BYTES, device_memory_budget

    try:
        return device_memory_budget(fraction=headroom), "device"
    except RuntimeError:
        budget = int(DEFAULT_DEVICE_MEMORY_BYTES * headroom)
        if not _memory_budget_default_logged:
            _memory_budget_default_logged = True
            logger.info(
                "resolve_memory_budget(%s): device did not report bytes_limit; "
                "using the documented default of %s bytes (4 GiB * headroom %s).",
                app,
                budget,
                headroom,
            )
        return budget, "default"


def _apply_parse(value: Any, parse: Callable[[Any], Any] | None, *, where: str) -> Any:
    if parse is None:
        return value
    try:
        return parse(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: {exc}") from exc


def _start_dir(start: str | Path | None) -> Path:
    if start is None:
        origin = Path.cwd()
    else:
        origin = Path(start).expanduser()
    origin = origin.resolve()
    if origin.is_file():
        return origin.parent
    return origin


def _tool_table(data: dict[str, Any], app: str) -> dict[str, Any] | None:
    tool = data.get("tool")
    if not isinstance(tool, dict):
        return None
    table = tool.get(app)
    if not isinstance(table, dict):
        return None
    return table


def _pyproject_value(app: str, key: str, start_dir: Path) -> tuple[Any, Path] | None:
    """Nearest ``[tool.<app>]`` value, or None.

    Walks upward. Pyprojects without the table are skipped. The nearest table
    that exists ends the walk: a missing ``key`` there does not fall through
    to a parent project.
    """
    for directory in (start_dir, *start_dir.parents):
        path = directory / "pyproject.toml"
        if not path.is_file():
            continue
        table = _tool_table(_read_toml(path), app)
        if table is None:
            continue
        if key not in table:
            return None
        return table[key], path
    return None


def _user_config_path(app: str) -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / app / "config.toml"


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"malformed TOML in '{path}': {exc}") from exc
    except OSError as exc:
        raise ValueError(f"cannot read config file '{path}': {exc}") from exc


def _parse_memory_budget(value: Any) -> int:
    parsed = value
    if isinstance(parsed, str):
        try:
            parsed = int(parsed.strip())
        except ValueError as exc:
            raise ValueError(f"memory_budget_bytes must be a positive int, got {value!r}") from exc
    if isinstance(parsed, bool) or not isinstance(parsed, int) or parsed <= 0:
        raise ValueError(f"memory_budget_bytes must be a positive int, got {value!r}")
    return parsed


__all__ = [
    "SchemaVersionStatus",
    "check_schema_version",
    "classify_schema_version",
    "load_toml_document",
    "require_field",
    "require_sections",
    "resolve_layered",
    "resolve_memory_budget",
]
