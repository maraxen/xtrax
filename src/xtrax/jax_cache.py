"""One configurable persistent JAX compilation cache for xtrax projects.

``configure()`` points JAX's persistent compilation cache at a directory chosen by
configuration, so every process of a campaign reuses the executables an earlier process
compiled instead of starting cold. Call it once at process entry, before the first
``jax.jit`` runs::

    from xtrax import jax_cache

    jax_cache.configure()

The directory is configuration and never a path baked into library code. Resolution order,
first match wins (the rule of ``isochore.traj_cache.cache_root_source``):

1. ``path``, the explicit argument (``"none"`` or ``""`` disables);
2. ``$XTRAX_JAX_CACHE_DIR`` if set at all (``none`` or empty disables and overrides files);
3. ``jax_compilation_cache_dir`` under ``[tool.xtrax]`` in the NEAREST ``pyproject.toml``
   at or above ``start`` (default: the current directory), so the consuming project wins;
4. ``jax_compilation_cache_dir`` in ``${XDG_CONFIG_HOME:-~/.config}/xtrax/config.toml``;
5. nothing: xtrax leaves the cache alone. JAX's own ``JAX_COMPILATION_CACHE_DIR`` is never
   overridden by an unset or disabled xtrax setting.

Values expand ``~`` and ``$VARS``. A relative path in a config file is relative to the
directory of that file; a relative explicit argument or environment value is relative to the
current directory. A config file that exists but cannot be parsed raises ``ValueError``
naming it; xtrax never silently runs cold.

Four more knobs follow the same layers. Unset means "leave JAX's default". Each is a
``[tool.xtrax]`` key (also a top-level key of the user config), an environment variable and an
argument of ``configure()``:

- ``jax_cache_min_compile_time_secs`` / ``XTRAX_JAX_CACHE_MIN_COMPILE_SECS`` /
  ``min_compile_time_secs`` sets ``jax_persistent_cache_min_compile_time_secs`` (JAX default
  1.0: cheaper compiles are never stored);
- ``jax_cache_min_entry_size_bytes`` / ``XTRAX_JAX_CACHE_MIN_ENTRY_BYTES`` /
  ``min_entry_size_bytes`` sets ``jax_persistent_cache_min_entry_size_bytes``;
- ``jax_cache_max_size_bytes`` / ``XTRAX_JAX_CACHE_MAX_BYTES`` / ``max_size_bytes`` sets
  ``jax_compilation_cache_max_size`` (LRU eviction; needs the ``filelock`` package);
- ``jax_cache_namespace`` / ``XTRAX_JAX_CACHE_NAMESPACE`` / ``namespace`` is ``auto``
  (default) or ``none`` and belongs to xtrax only.

Keying (read from ``jax._src.cache_key`` on jax 0.11.1): JAX already keys each entry by the
computation, the jaxlib version, the backend platform and ``platform_version``, the device
kind or topology fingerprint, the compile options and the XLA flags. A directory can
therefore hold entries for several jax/jaxlib versions and for CPU and GPU together: a
mismatch is a miss, never a wrong hit. What the key does NOT visibly carry is the host CPU
feature set (on CPU, ``platform_version`` is just ``"cpu"``). Executables built on a node with
different CPU features have loaded with ``Target machine feature ... is not supported``
warnings and SIGILL risk on a heterogeneous cluster, so by default (``namespace="auto"``)
xtrax adds ``jaxlib-<version>/cpu-<hash of host CPU flags>`` under the configured directory.
Both are computed without touching the JAX backend, so ``configure()`` never initialises
JAX. The jaxlib segment also makes pruning one version a single ``rm -r``.
``namespace="none"`` uses the directory as given.

Capacity: every entry is one file; with ``max_size`` set JAX also writes one ``-atime`` file
per entry (LRU eviction), so the file count doubles. On GPU the XLA per-fusion autotune cache
adds files under the same directory. On a filesystem with a file-count cap (not just a byte
quota), set ``jax_cache_min_compile_time_secs`` high enough that only expensive compiles are
stored, and ``jax_cache_max_size_bytes`` to bound growth.

Safety: JAX treats the cache as trusted code. A directory others can write to lets them run
code in your process, so ``configure()`` warns when the directory is world-writable. A
group-writable setgid directory on shared scratch is the usual intended case and is only
reported at INFO.
"""

import hashlib
import importlib.metadata
import importlib.util
import logging
import os
import platform
import stat
import tempfile
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xtrax.config import _pyproject_value, _read_toml, _start_dir, _user_config_path

logger = logging.getLogger(__name__)

APP = "xtrax"

ENV_DIR = "XTRAX_JAX_CACHE_DIR"
ENV_MIN_COMPILE_SECS = "XTRAX_JAX_CACHE_MIN_COMPILE_SECS"
ENV_MIN_ENTRY_BYTES = "XTRAX_JAX_CACHE_MIN_ENTRY_BYTES"
ENV_MAX_BYTES = "XTRAX_JAX_CACHE_MAX_BYTES"
ENV_NAMESPACE = "XTRAX_JAX_CACHE_NAMESPACE"

KEY_DIR = "jax_compilation_cache_dir"
KEY_MIN_COMPILE_SECS = "jax_cache_min_compile_time_secs"
KEY_MIN_ENTRY_BYTES = "jax_cache_min_entry_size_bytes"
KEY_MAX_BYTES = "jax_cache_max_size_bytes"
KEY_NAMESPACE = "jax_cache_namespace"

NAMESPACES = ("auto", "none")

_DISABLED = frozenset({"", "none"})


class JaxCacheConfigError(ValueError):
    """A configured cache value is not valid; the message names the file or variable."""


class JaxCacheDirError(OSError):
    """The resolved cache directory cannot be created or written."""


@dataclass(frozen=True)
class CacheSetting:
    """What ``resolve``/``configure`` decided.

    ``dir`` is the directory JAX writes to (namespace applied), ``None`` when ``enabled`` is False.
    ``base_dir`` is the configured directory before the namespace. ``source`` names the layer that
    chose the directory. ``knob_sources`` names the layer for each knob that was set.
    """

    dir: Path | None
    source: str
    enabled: bool
    base_dir: Path | None = None
    namespace: str = "auto"
    min_compile_time_secs: float | None = None
    min_entry_size_bytes: int | None = None
    max_size_bytes: int | None = None
    knob_sources: Mapping[str, str] = field(default_factory=dict)


def _expand(text: str) -> str:
    return os.path.expandvars(os.path.expanduser(text.strip()))


def _absolute(text: str, base: Path | None = None) -> Path:
    path = Path(_expand(text))
    if path.is_absolute():
        return path
    return Path(os.path.abspath((base or Path.cwd()) / path))


def _layered(
    key: str,
    env_var: str,
    arg: Any,
    start: Path,
) -> tuple[Any, str, Path | None]:
    """``(raw value, source, declaring file)``; raw ``None`` means no layer set ``key``.

    The first layer that sets the key decides, even when its value is ``none`` or empty: the
    caller reads that as "disabled here", which differs from ``None`` ("nothing set").
    """
    if arg is not None:
        return arg, "explicit argument", None
    if env_var in os.environ:
        return os.environ[env_var], f"${env_var}", None
    found = _pyproject_value(APP, key, start)
    if found is not None:
        return found[0], f"project config {found[1]}", found[1]
    user = _user_config_path(APP)
    if user.is_file():
        data = _read_toml(user)
        if key in data:
            return data[key], f"user config {user}", user
    return None, "unset", None


def _is_disabled(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in _DISABLED


def cache_dir_source(
    path: Path | str | None = None, start: Path | str | None = None
) -> tuple[Path | None, str]:
    """``(configured directory or None, which layer decided)``; see the module docstring.

    Pure: reads configuration only; never creates a directory and never imports JAX. The
    returned directory is the configured base, before the namespace segments ``configure``
    adds.
    """
    raw, source, origin = _layered(KEY_DIR, ENV_DIR, path, _start_dir(start))
    if raw is None:
        return None, source
    if isinstance(raw, Path):
        raw = str(raw)
    if not isinstance(raw, str):
        raise JaxCacheConfigError(
            f'{source}: {KEY_DIR!r} must be a string path or "none", got {raw!r}'
        )
    if _is_disabled(raw):
        return None, f"{source} (disabled)"
    return _absolute(raw, origin.parent if origin is not None else None), source


def _number_knob(
    key: str,
    env_var: str,
    arg: Any,
    start: Path,
    *,
    kind: Callable[[Any], Any],
    minimum: float,
    label: str,
) -> tuple[Any, str | None]:
    raw, source, _ = _layered(key, env_var, arg, start)
    if raw is None or _is_disabled(raw):
        return None, None
    if isinstance(raw, bool):
        raise JaxCacheConfigError(f"{source}: {key!r} must be a number, got {raw!r}")
    try:
        value = kind(raw.strip() if isinstance(raw, str) else raw)
    except (TypeError, ValueError) as exc:
        raise JaxCacheConfigError(f"{source}: {key!r} {label}, got {raw!r}") from exc
    if kind is int and isinstance(raw, float) and raw != value:
        raise JaxCacheConfigError(f"{source}: {key!r} {label}, got {raw!r}")
    if value < minimum:
        raise JaxCacheConfigError(f"{source}: {key!r} must be >= {minimum:g}, got {raw!r}")
    return value, source


def resolve(
    path: Path | str | None = None,
    *,
    start: Path | str | None = None,
    min_compile_time_secs: float | None = None,
    min_entry_size_bytes: int | None = None,
    max_size_bytes: int | None = None,
    namespace: str | None = None,
) -> CacheSetting:
    """Resolve the cache setting without touching JAX or the filesystem."""
    start_dir = _start_dir(start)
    base, source = cache_dir_source(path, start_dir)

    knob_sources: dict[str, str] = {}
    min_secs, src = _number_knob(
        KEY_MIN_COMPILE_SECS,
        ENV_MIN_COMPILE_SECS,
        min_compile_time_secs,
        start_dir,
        kind=float,
        minimum=0.0,
        label="must be a number of seconds",
    )
    if src:
        knob_sources["min_compile_time_secs"] = src
    min_entry, src = _number_knob(
        KEY_MIN_ENTRY_BYTES,
        ENV_MIN_ENTRY_BYTES,
        min_entry_size_bytes,
        start_dir,
        kind=int,
        minimum=-1,
        label="must be an integer byte count (-1 disables the size restriction)",
    )
    if src:
        knob_sources["min_entry_size_bytes"] = src
    max_size, src = _number_knob(
        KEY_MAX_BYTES,
        ENV_MAX_BYTES,
        max_size_bytes,
        start_dir,
        kind=int,
        minimum=-1,
        label="must be an integer byte count (-1 is unlimited)",
    )
    if src:
        knob_sources["max_size_bytes"] = src

    raw_ns, ns_source, _ = _layered(KEY_NAMESPACE, ENV_NAMESPACE, namespace, start_dir)
    if raw_ns is None or (isinstance(raw_ns, str) and raw_ns.strip() == ""):
        ns = "auto"
    else:
        ns = str(raw_ns).strip().lower()
        if ns not in NAMESPACES:
            raise JaxCacheConfigError(
                f"{ns_source}: {KEY_NAMESPACE!r} must be one of {NAMESPACES}, got {raw_ns!r}"
            )
        knob_sources["namespace"] = ns_source

    if base is None:
        return CacheSetting(
            dir=None,
            source=source,
            enabled=False,
            namespace=ns,
            min_compile_time_secs=min_secs,
            min_entry_size_bytes=min_entry,
            max_size_bytes=max_size,
            knob_sources=knob_sources,
        )
    return CacheSetting(
        dir=_namespaced(base, ns),
        source=source,
        enabled=True,
        base_dir=base,
        namespace=ns,
        min_compile_time_secs=min_secs,
        min_entry_size_bytes=min_entry,
        max_size_bytes=max_size,
        knob_sources=knob_sources,
    )


def host_cpu_fingerprint() -> str:
    """Eight hex digits that identify this host's CPU feature set; no JAX, no backend init.

    Hashes the ``flags`` (x86) or ``Features`` (arm) line of ``/proc/cpuinfo`` plus the
    machine type; falls back to ``platform.processor()`` where ``/proc/cpuinfo`` is unavailable.
    """
    flags = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                name, _, value = line.partition(":")
                if name.strip().lower() in {"flags", "features"}:
                    flags = " ".join(sorted(value.split()))
                    break
    except OSError:
        pass
    material = f"{platform.machine()}|{flags or platform.processor()}"
    return hashlib.sha256(material.encode()).hexdigest()[:8]


def _jaxlib_version() -> str:
    try:
        return importlib.metadata.version("jaxlib")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _namespaced(base: Path, namespace: str) -> Path:
    if namespace == "none":
        return base
    return base / f"jaxlib-{_jaxlib_version()}" / f"cpu-{host_cpu_fingerprint()}"


def _prepare_directory(directory: Path) -> None:
    """Create ``directory`` and prove it is writable; warn when others can write to it."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".xtrax_write_probe_"):
            pass
        mode = directory.stat().st_mode
    except OSError as exc:
        raise JaxCacheDirError(
            f"JAX compilation cache directory {directory} is not writable ({exc}); "
            f"fix the path or disable the cache with {ENV_DIR}=none"
        ) from exc
    if mode & stat.S_IWOTH:
        logger.warning(
            "JAX compilation cache directory %s is world-writable; JAX treats cache contents as "
            "trusted code, so anyone who can write there can run code in this process",
            directory,
        )
    elif mode & stat.S_ISGID:
        logger.info(
            "JAX compilation cache directory %s is setgid; new entries inherit its group", directory
        )


_LOCK = threading.Lock()
_applied: CacheSetting | None = None


def configure(
    path: Path | str | None = None,
    *,
    start: Path | str | None = None,
    min_compile_time_secs: float | None = None,
    min_entry_size_bytes: int | None = None,
    max_size_bytes: int | None = None,
    namespace: str | None = None,
) -> CacheSetting:
    """Resolve the cache setting and apply it to JAX. Idempotent; safe before or after import.

    Call it once at process entry, before the first ``jax.jit`` call. JAX reads the cache
    directory lazily at the first compile, so a call made before that takes effect with no
    further work. A call made after a compile that already initialised a cache at a different
    directory resets JAX's cache object (``compilation_cache.reset_cache``) and logs a
    warning; programs compiled earlier are not written retroactively. Calling again with the
    same resolved setting changes nothing.

    When nothing configures a directory (or it is disabled) xtrax leaves JAX untouched,
    including a ``JAX_COMPILATION_CACHE_DIR`` the environment already set.

    Raises:
        ValueError: a configuration file is malformed or a value is invalid; the message
            names the file or variable.
        JaxCacheDirError: the directory cannot be created or written.
        JaxCacheConfigError: ``max_size`` is set but the ``filelock`` package JAX needs for
            eviction is not installed.
    """
    global _applied
    setting = resolve(
        path,
        start=start,
        min_compile_time_secs=min_compile_time_secs,
        min_entry_size_bytes=min_entry_size_bytes,
        max_size_bytes=max_size_bytes,
        namespace=namespace,
    )
    with _LOCK:
        if not setting.enabled:
            logger.info("xtrax JAX compilation cache: off (%s)", setting.source)
            _note_jax_owned_cache()
            return setting
        if _applied == setting:
            return setting
        assert setting.dir is not None
        if (
            setting.max_size_bytes is not None
            and setting.max_size_bytes > 0
            and importlib.util.find_spec("filelock") is None
        ):
            raise JaxCacheConfigError(
                "jax_cache_max_size_bytes needs the 'filelock' package (JAX's LRU eviction); "
                "install filelock or leave the size limit unset"
            )
        _prepare_directory(setting.dir)
        _apply_to_jax(setting)
        _applied = setting
        logger.info(
            "xtrax JAX compilation cache: %s (decided by %s)%s",
            setting.dir,
            setting.source,
            _knob_summary(setting),
        )
        return setting


def _knob_summary(setting: CacheSetting) -> str:
    parts = []
    if setting.min_compile_time_secs is not None:
        parts.append(f"min_compile_time_secs={setting.min_compile_time_secs:g}")
    if setting.min_entry_size_bytes is not None:
        parts.append(f"min_entry_size_bytes={setting.min_entry_size_bytes}")
    if setting.max_size_bytes is not None:
        parts.append(f"max_size_bytes={setting.max_size_bytes}")
    return f" [{', '.join(parts)}]" if parts else ""


def _note_jax_owned_cache() -> None:
    import jax

    existing = jax.config.jax_compilation_cache_dir
    if existing:
        logger.info("JAX's own jax_compilation_cache_dir=%s is left untouched", existing)


def _apply_to_jax(setting: CacheSetting) -> None:
    import jax

    assert setting.dir is not None
    new_dir = str(setting.dir)
    existing = jax.config.jax_compilation_cache_dir
    if existing and existing != new_dir:
        from jax.experimental.compilation_cache import compilation_cache

        logger.warning(
            "JAX compilation cache was already pointed at %s; switching to %s and resetting "
            "JAX's cache object (programs compiled before this call are not written to the "
            "new directory)",
            existing,
            new_dir,
        )
        compilation_cache.reset_cache()
    jax.config.update("jax_compilation_cache_dir", new_dir)
    if setting.min_compile_time_secs is not None:
        jax.config.update(
            "jax_persistent_cache_min_compile_time_secs", setting.min_compile_time_secs
        )
    if setting.min_entry_size_bytes is not None:
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", setting.min_entry_size_bytes)
    if setting.max_size_bytes is not None:
        jax.config.update("jax_compilation_cache_max_size", setting.max_size_bytes)
