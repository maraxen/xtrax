"""One resolver for the JAX persistent compilation cache: where (placement) and whether (policy).

Spec: ``.praxia/docs/specs/261119_compilation-cache-resolver.md``.

Placement is a fact about the machine; policy is a fact about the project. They resolve
independently, and the cache is on only when policy is ``on`` and a placement exists.

Placement, first match wins:

1. the ``explicit`` argument (``"off"``, ``"none"`` or ``""`` is off; a path is a placement and
   a policy-on);
2. ``XTRAX_COMPILATION_CACHE_DIR`` when set at all (``""``/``none``/``off`` stops the search: off);
3. ``JAX_COMPILATION_CACHE_DIR`` when set at all (same; JAX itself reads this one at import);
4. ``compilation_cache_dir`` in ``${XDG_CONFIG_HOME:-~/.config}/xtrax/config.toml``;
5. none.

A ``compilation_cache_dir`` under ``[tool.xtrax]`` in ``pyproject.toml`` is refused: a committed
file is pushed to remotes, where a path written on another machine names the wrong place.

Policy, first match wins: ``explicit`` > ``XTRAX_COMPILATION_CACHE`` (``on``/``off``) >
``compilation_cache`` under ``[tool.xtrax]`` > ``compilation_cache`` in the user config >
default ``on``.

When neither axis is configured anywhere, :func:`apply_decision` leaves JAX untouched, so a
caller that configured JAX itself keeps its setting. Any configured "off" is applied actively:
JAX reads ``JAX_COMPILATION_CACHE_DIR`` at import and takes ``none`` as a relative path, so off
clears its flag before the first compile.
"""

import logging
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from xtrax.config import (
    _pyproject_value,
    _read_toml,
    _start_dir,
    _user_config_path,
    resolve_layered,
)

logger = logging.getLogger(__name__)

APP = "xtrax"
KEY_DIR = "compilation_cache_dir"
KEY_POLICY = "compilation_cache"
KEY_MIN_SECS = "compilation_cache_min_compile_secs"
ENV_VAR = "XTRAX_COMPILATION_CACHE_DIR"
POLICY_ENV_VAR = "XTRAX_COMPILATION_CACHE"
MIN_SECS_ENV_VAR = "XTRAX_COMPILATION_CACHE_MIN_COMPILE_SECS"
JAX_ENV_VAR = "JAX_COMPILATION_CACHE_DIR"
OFF = "off"
DEFAULT_MIN_COMPILE_SECS = 1.0

_OFF_TOKENS = frozenset({"", "none", "off"})

CacheSource = Literal["arg", "env_xtrax", "env_jax", "user_config", "none"]
PolicySource = Literal["arg", "env_xtrax", "pyproject", "user_config", "default"]
Policy = Literal["on", "off"]
OnConflict = Literal["raise", "reset"]


@dataclass(frozen=True)
class CacheDecision:
    """A resolved cache decision. Frozen and picklable, so workers receive it unchanged.

    ``directory`` is the effective cache directory: ``None`` when either axis says off.
    ``placement`` is what the placement axis resolved, kept even when policy is off, so a run
    held off by project policy is distinguishable from a run with no placement at all.
    """

    directory: Path | None
    placement: Path | None
    source: CacheSource
    source_path: Path | None
    policy: Policy
    policy_source: PolicySource
    min_compile_secs: float
    min_compile_secs_source: str

    @property
    def configured(self) -> bool:
        """Whether any layer configured either axis.

        Returns:
            False when neither axis was configured anywhere; :func:`apply_decision` then
            leaves JAX's settings unchanged.
        """
        return not (self.source == "none" and self.policy_source == "default")

    def describe(self) -> str:
        """Summarise the decision on one line.

        Returns:
            The effective directory (or ``off``), the placement layer and the policy layer.
        """
        where = str(self.directory) if self.directory is not None else OFF
        placement = self.source if self.source_path is None else f"{self.source} {self.source_path}"
        return (
            f"compilation cache: {where} (placement: {placement}; "
            f"policy: {self.policy} from {self.policy_source})"
        )

    def as_record(self) -> dict[str, Any]:
        """Convert the decision for run summaries and sidecars.

        Returns:
            A JSON-ready dict; paths become strings.
        """
        record = asdict(self)
        for key in ("directory", "placement", "source_path"):
            if record[key] is not None:
                record[key] = str(record[key])
        return record


def _parse_dir(value: Any, where: str, *, allow_relative: bool) -> Path | None:
    """Parse a placement value. ``None`` means the layer says off."""
    if isinstance(value, Path):
        if str(value) in ("", "."):
            raise ValueError(
                f"{where}: Path('') and Path('.') are ambiguous; pass 'off' or an explicit path"
            )
        text = str(value)
    elif isinstance(value, str):
        text = value.strip()
        if text.lower() in _OFF_TOKENS:
            return None
    else:
        raise TypeError(f"{where}: {KEY_DIR} must be a string path, got {type(value).__name__}")
    path = Path(os.path.expandvars(text)).expanduser()
    if not path.is_absolute():
        if not allow_relative:
            raise ValueError(f"{where}: must be absolute or start with '~', got {value!r}")
        path = Path.cwd() / path
    return Path(os.path.normpath(path))


def _parse_policy(value: Any, where: str = "policy") -> Policy:
    """Parse a policy value: ``on``, or ``off``/``none``/empty (any case); TOML booleans too."""
    if isinstance(value, bool):
        return "on" if value else "off"
    if not isinstance(value, str):
        raise TypeError(f"{where}: {KEY_POLICY} must be 'on' or 'off', got {type(value).__name__}")
    token = value.strip().lower()
    if token == "on":
        return "on"
    if token in _OFF_TOKENS:
        return "off"
    raise ValueError(f"{where}: {KEY_POLICY} must be 'on' or 'off', got {value!r}")


def _parse_min_secs(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise TypeError(f"{KEY_MIN_SECS} must be a number, got {type(value).__name__}")
    secs = float(value)
    if secs < 0:
        raise ValueError(f"{KEY_MIN_SECS} must be >= 0, got {value!r}")
    return secs


def _refuse_pyproject_placement(start: str | Path | None) -> None:
    found = _pyproject_value(APP, KEY_DIR, _start_dir(start))
    if found is not None:
        _, path = found
        raise ValueError(
            f"{path}: [tool.{APP}] {KEY_DIR} is not allowed. A committed pyproject is pushed to "
            f"other machines, so it must not hold a machine path. Set {KEY_DIR} in "
            f"{_user_config_path(APP)} (or the remote's myxcel profile [env]); keep only "
            f"{KEY_POLICY} = 'on'|'off' in pyproject."
        )


def _resolve_placement(explicit: Any) -> tuple[Path | None, CacheSource, Path | None]:
    if explicit is not None:
        return _parse_dir(explicit, "explicit", allow_relative=True), "arg", None
    if ENV_VAR in os.environ:
        return (
            _parse_dir(os.environ[ENV_VAR], f"${ENV_VAR}", allow_relative=False),
            "env_xtrax",
            None,
        )
    if JAX_ENV_VAR in os.environ:
        raw = os.environ[JAX_ENV_VAR]
        path = _parse_dir(raw, f"${JAX_ENV_VAR}", allow_relative=True)
        if path is not None and not Path(raw.strip()).expanduser().is_absolute():
            logger.warning("$%s=%r is relative; resolved against cwd to %s", JAX_ENV_VAR, raw, path)
        return path, "env_jax", None
    user_path = _user_config_path(APP)
    if user_path.is_file():
        data = _read_toml(user_path)
        if KEY_DIR in data:
            return (
                _parse_dir(data[KEY_DIR], str(user_path), allow_relative=False),
                "user_config",
                user_path,
            )
    return None, "none", None


def resolve_policy(*, start: str | Path | None = None) -> tuple[Policy, PolicySource]:
    """Resolve the policy axis alone: env > pyproject > user config > default ``on``.

    Args:
        start: Directory to walk upward from when looking for ``pyproject.toml``.

    Returns:
        ``(policy, source)``.

    Raises:
        ValueError: an unknown policy token or a malformed TOML file.
    """
    if POLICY_ENV_VAR in os.environ:
        return _parse_policy(os.environ[POLICY_ENV_VAR], f"${POLICY_ENV_VAR}"), "env_xtrax"
    value, source = resolve_layered(
        APP, KEY_POLICY, env_var=None, start=start, parse=_parse_policy, default=None
    )
    if source == "pyproject":
        return value, "pyproject"
    if source == "user_config":
        return value, "user_config"
    return "on", "default"


def resolve_compilation_cache(
    explicit: str | Path | None = None,
    *,
    start: str | Path | None = None,
    min_compile_secs: float | None = None,
) -> CacheDecision:
    """Resolve both axes and the write threshold. Pure: touches neither JAX nor the filesystem.

    Args:
        explicit: Caller's choice. ``None`` = not given; ``"off"``/``"none"``/``""`` = off;
            a path = placement and policy-on (relative resolves against cwd).
        start: Directory to walk upward from when looking for ``pyproject.toml``.
        min_compile_secs: Write threshold; ``None`` resolves it from the layers.

    Returns:
        The resolved :class:`CacheDecision`.

    Raises:
        ValueError: a pyproject holds ``compilation_cache_dir``; a relative path in
            ``XTRAX_COMPILATION_CACHE_DIR`` or the user config; an unknown policy token;
            a malformed TOML file. The message names the source.
    """
    _refuse_pyproject_placement(start)
    placement, source, source_path = _resolve_placement(explicit)

    if explicit is not None:
        policy: Policy = "on" if placement is not None else "off"
        policy_source: PolicySource = "arg"
    else:
        policy, policy_source = resolve_policy(start=start)

    if min_compile_secs is not None:
        secs, secs_source = _parse_min_secs(min_compile_secs), "arg"
    else:
        secs, secs_source = resolve_layered(
            APP,
            KEY_MIN_SECS,
            env_var=MIN_SECS_ENV_VAR,
            start=start,
            parse=_parse_min_secs,
            default=DEFAULT_MIN_COMPILE_SECS,
        )

    return CacheDecision(
        directory=placement if policy == "on" else None,
        placement=placement,
        source=source,
        source_path=source_path,
        policy=policy,
        policy_source=policy_source,
        min_compile_secs=float(secs),
        min_compile_secs_source=secs_source,
    )


def _cache_initialized() -> bool:
    """Whether JAX's persistent cache is already built. Unknown counts as built."""
    try:
        from jax._src import compilation_cache as _jax_cc

        value = getattr(_jax_cc, "_cache_initialized", None)
    except Exception:  # noqa: BLE001 - any import trouble means "unknown"
        value = None
    if value is None:
        logger.warning(
            "cannot read jax._src.compilation_cache._cache_initialized; "
            "assuming the persistent cache is already initialised"
        )
        return True
    return bool(value)


def prepare_directory(directory: Path) -> None:
    """Create the cache directory and check it is usable; warn when others can write it.

    Args:
        directory: The resolved cache directory.

    Raises:
        ValueError: the path exists and is not a directory, or is not writable.
    """
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"compilation cache path {directory} exists and is not a directory")
    directory.mkdir(parents=True, exist_ok=True)
    if not os.access(directory, os.W_OK | os.X_OK):
        raise ValueError(f"compilation cache directory {directory} is not writable")
    mode = directory.stat().st_mode
    if mode & (stat.S_IWGRP | stat.S_IWOTH):
        logger.warning(
            "compilation cache directory %s is group- or world-writable (mode %o); anyone who can "
            "write it can plant executables that this process loads",
            directory,
            stat.S_IMODE(mode),
        )


_last_applied: CacheDecision | None = None


def apply_decision(decision: CacheDecision, *, on_conflict: OnConflict = "raise") -> CacheDecision:
    """Apply an already resolved decision to JAX's global config without re-resolving.

    Idempotent and quiet (DEBUG) when the same decision was applied before. An unconfigured
    decision leaves JAX untouched.

    Args:
        decision: A decision from :func:`resolve_compilation_cache`.
        on_conflict: What to do when JAX's cache is already initialised with a different
            directory: ``"raise"`` or ``"reset"`` (reset JAX's cache, then apply).

    Returns:
        ``decision``, unchanged.

    Raises:
        RuntimeError: a conflict under ``on_conflict="raise"``.
        ValueError: the directory is unusable (see :func:`prepare_directory`).
    """
    global _last_applied
    if on_conflict not in ("raise", "reset"):
        raise ValueError(f"on_conflict must be 'raise' or 'reset', got {on_conflict!r}")
    try:
        import jax
    except ImportError:
        return decision

    repeat = decision == _last_applied
    if not decision.configured:
        if not repeat:
            logger.debug("compilation cache: not configured; leaving JAX's settings unchanged")
        _last_applied = decision
        return decision

    current_raw = jax.config.jax_compilation_cache_dir
    current = str(current_raw) if current_raw else None
    target = str(decision.directory) if decision.directory is not None else None

    if target != current and current is not None and _cache_initialized():
        if on_conflict == "raise":
            raise RuntimeError(
                f"JAX's persistent compilation cache is already initialised (flag value "
                f"{current!r}); cannot switch it to {target or OFF!r}. Apply the decision before "
                f"the first compile, or pass on_conflict='reset'."
            )
        from jax.experimental.compilation_cache import compilation_cache as jax_cc

        jax_cc.reset_cache()

    if decision.directory is not None:
        prepare_directory(decision.directory)
        if target != current:
            jax.config.update("jax_compilation_cache_dir", target)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", decision.min_compile_secs)
    elif current is not None:
        jax.config.update("jax_compilation_cache_dir", None)

    if repeat:
        logger.debug(decision.describe())
    else:
        logger.info(decision.describe())
    _last_applied = decision
    return decision


def enable_compilation_cache(
    explicit: str | Path | None = None,
    *,
    start: str | Path | None = None,
    min_compile_secs: float | None = None,
    on_conflict: OnConflict = "raise",
) -> CacheDecision:
    """Resolve the cache decision and apply it to JAX. Call once at an entry point.

    Args:
        explicit: See :func:`resolve_compilation_cache`.
        start: See :func:`resolve_compilation_cache`.
        min_compile_secs: See :func:`resolve_compilation_cache`.
        on_conflict: See :func:`apply_decision`.

    Returns:
        The applied :class:`CacheDecision`.
    """
    decision = resolve_compilation_cache(explicit, start=start, min_compile_secs=min_compile_secs)
    return apply_decision(decision, on_conflict=on_conflict)


def decision_for(cache: "CacheDecision | str | Path | None") -> CacheDecision:
    """Normalise an executor's ``cache_dir`` argument to a resolved decision.

    Args:
        cache: A decision (returned as is), or an explicit argument to resolve.

    Returns:
        A resolved :class:`CacheDecision`.
    """
    if isinstance(cache, CacheDecision):
        return cache
    return resolve_compilation_cache(cache)


__all__ = [
    "DEFAULT_MIN_COMPILE_SECS",
    "ENV_VAR",
    "JAX_ENV_VAR",
    "KEY_DIR",
    "KEY_MIN_SECS",
    "KEY_POLICY",
    "MIN_SECS_ENV_VAR",
    "OFF",
    "POLICY_ENV_VAR",
    "CacheDecision",
    "apply_decision",
    "decision_for",
    "enable_compilation_cache",
    "prepare_directory",
    "resolve_compilation_cache",
    "resolve_policy",
]
