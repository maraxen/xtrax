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
3. ``$JAX_COMPILATION_CACHE_DIR`` if set and non-empty (JAX's own variable: an environment
   variable beats a file). The value is taken as JAX takes it, so ``none`` there is a
   directory named ``none``, not a disable word. xtrax applies its namespace under it, so an
   existing flat JAX directory goes cold once;
4. ``jax_compilation_cache_dir`` under ``[tool.xtrax]`` in the NEAREST ``pyproject.toml``
   that has an ``[tool.xtrax]`` table at or above ``start`` (default: the current
   directory);
5. ``jax_compilation_cache_dir`` in ``${XDG_CONFIG_HOME:-~/.config}/xtrax/config.toml``;
6. nothing: xtrax leaves the cache alone. JAX's own flag is never overridden by an unset or
   disabled xtrax setting.

Values expand ``~`` and ``$VARS``; a ``$VAR`` that is not set, or a ``~user`` that does not
exist, raises ``JaxCacheConfigError`` naming the layer rather than becoming a literal
directory. A relative path in a config file is relative to the directory of that file; a
relative explicit argument or environment value is relative to the current directory. A
config file that exists but cannot be parsed raises ``ValueError`` naming it; xtrax never
silently runs cold. Limitation: the ``pyproject.toml`` walk skips a nearer file that has no
``[tool.xtrax]`` table, so a nested subproject without one inherits its parent's directory.
Declare ``jax_compilation_cache_dir = "none"`` there to opt out. Adding a ``[tool.xtrax]``
table to a project also ends the walk for every other ``xtrax.config`` key.

Four more knobs follow the same layers (minus the JAX variable). Unset means "leave JAX's
default". Each is a ``[tool.xtrax]`` key (also a top-level key of the user config), an
environment variable and an argument of ``configure()``:

- ``jax_cache_min_compile_time_secs`` / ``XTRAX_JAX_CACHE_MIN_COMPILE_SECS`` /
  ``min_compile_time_secs`` sets ``jax_persistent_cache_min_compile_time_secs`` (JAX default
  1.0: cheaper compiles are never stored); finite and >= 0;
- ``jax_cache_min_entry_size_bytes`` / ``XTRAX_JAX_CACHE_MIN_ENTRY_BYTES`` /
  ``min_entry_size_bytes`` sets ``jax_persistent_cache_min_entry_size_bytes``;
- ``jax_cache_max_size_bytes`` / ``XTRAX_JAX_CACHE_MAX_BYTES`` / ``max_size_bytes`` sets
  ``jax_compilation_cache_max_size`` (LRU eviction; any value but -1 needs the ``filelock``
  package, and 0 disables caching outright);
- ``jax_cache_namespace`` / ``XTRAX_JAX_CACHE_NAMESPACE`` / ``namespace`` is ``auto``
  (default), ``none`` or ``cpu-flags`` and belongs to xtrax only.

Keying (read from ``jax._src.cache_key`` on jax 0.11.1): JAX keys each entry by the
computation, the jaxlib version, the backend platform and ``platform_version``, the compile
options, the XLA flags and an accelerator fingerprint. A directory can therefore hold
entries for several jax/jaxlib versions and for CPU and GPU together: a mismatch is a miss,
never a wrong hit. On CPU the accelerator fingerprint is
``xla_client.get_topology_for_devices(devices).fingerprint()``. An independent probe found it
unchanged by unrelated XLA flags and changed by every ``--xla_cpu_max_isa`` cap, so it
encodes the target CPU feature set and JAX already separates CPU entries by host features.
That evidence is indirect: no two physical hosts were compared. An earlier CPU-feature
mismatch on heterogeneous cluster nodes ("Target machine feature ... is not supported",
SIGILL risk) was seen on an older stack, and whether it can recur on this one is open.
Confirm by printing that fingerprint on two node types.

Namespaces, applied under the configured directory and computed without touching the JAX
backend, so ``configure()`` never initialises JAX:

- ``auto`` (default): ``jaxlib-<version>``. The version is already in JAX's key, so this
  costs no hits; it only makes pruning one version a single ``rm -r``.
- ``none``: the directory as given.
- ``cpu-flags``: ``jaxlib-<version>/cpu-<hash of the ISA flags in /proc/cpuinfo>``, a
  defence-in-depth opt-in for CPU executables shared across hosts with different CPUs. It
  hashes only ISA-relevant flags (``sse*``, ``avx*``, ``fma``, ``bmi*``, ``amx*``,
  ``f16c``, ``popcnt``, ``vnni``-like), not the whole flags line, so mitigation-flag
  drift does not split the cache. It splits GPU caches across hosts with different CPUs,
  so leave it off for GPU work. It raises when the CPU flags cannot be read.

Capacity: every entry is one file; with ``max_size`` set JAX also writes one ``-atime`` file
per entry (LRU eviction), so the file count doubles. On GPU XLA's per-fusion autotune cache
also writes files under the same directory, and ``min_compile_time_secs`` does not bound
them; set ``JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none`` to turn it off. On a filesystem
with a file-count cap (not just a byte quota), set ``jax_cache_min_compile_time_secs`` high
enough that only expensive compiles are stored, and ``jax_cache_max_size_bytes`` to bound
growth.

Write safety: JAX writes each entry with a plain ``write_bytes`` (no temporary file, no
rename; ``jax/_src/lru_cache.py``), and skips a key whose file already exists. A process
preempted mid-write therefore leaves a truncated entry that is a permanent miss (JAX turns
the failed read into a warning and recompiles). ``prune_corrupt()`` deletes JAX-key-named
entries that do not decode under zlib or zstd (JAX picks the codec in the writing process,
so the pruner accepts either and refuses, deleting nothing, when an entry needs a zstd
decoder it cannot import). Run it between campaign attempts, never while writers are active
(a file younger than ``min_age_secs`` is skipped for that reason).

Limits not handled here: ``configure()`` probes the directory by writing 4 KiB, which passes
on a setgid directory whose group is at its byte quota, after which JAX logs a failed ``put``
per entry and the run is silently cold. Treat "quota exceeded" warnings from JAX as a cold
cache. ``configure()`` also compares against the live ``jax.config`` only for the directory,
so a changed knob is applied but code that edits other JAX cache flags itself is not
reconciled.

Safety: JAX treats the cache as trusted code. A directory others can write to lets them run
code in your process, so ``configure()`` warns when the directory is world-writable. A
group-writable setgid directory on shared scratch is the usual intended case and is only
reported at INFO.
"""

import hashlib
import importlib.metadata
import importlib.util
import logging
import math
import os
import platform
import re
import stat
import tempfile
import threading
import zlib
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

ENV_JAX_DIR = "JAX_COMPILATION_CACHE_DIR"

NAMESPACES = ("auto", "none", "cpu-flags")

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


def _expand(text: str, source: str) -> str:
    """Expand ``~`` and ``$VARS``; raise when one is left unresolved (it would become a path)."""
    expanded = os.path.expandvars(os.path.expanduser(text.strip()))
    if "$" in expanded or expanded.startswith("~"):
        raise JaxCacheConfigError(
            f"{source}: {text!r} still contains an unexpanded variable or '~' after expansion "
            f"({expanded!r}); export the variable or write an absolute path"
        )
    return expanded


def _absolute(text: str, source: str, base: Path | None = None) -> Path:
    path = Path(_expand(text, source))
    if path.is_absolute():
        return path
    return Path(os.path.abspath((base or Path.cwd()) / path))


def _layered(
    key: str,
    env_var: str,
    arg: Any,
    start: Path,
    *,
    also_env: str | None = None,
) -> tuple[Any, str, Path | None]:
    """``(raw value, source, declaring file)``; raw ``None`` means no layer set ``key``.

    The first layer that sets the key decides, even when its value is ``none`` or empty: the
    caller reads that as "disabled here", which differs from ``None`` ("nothing set").
    ``also_env`` names a second environment variable consulted after ``env_var``; an empty
    value there counts as unset.
    """
    if arg is not None:
        return arg, "explicit argument", None
    if env_var in os.environ:
        return os.environ[env_var], f"${env_var}", None
    if also_env is not None and os.environ.get(also_env, "").strip():
        return os.environ[also_env], f"${also_env}", None
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
    raw, source, origin = _layered(KEY_DIR, ENV_DIR, path, _start_dir(start), also_env=ENV_JAX_DIR)
    if raw is None:
        return None, source
    if isinstance(raw, Path):
        raw = str(raw)
    if not isinstance(raw, str):
        raise JaxCacheConfigError(
            f'{source}: {KEY_DIR!r} must be a string path or "none", got {raw!r}'
        )
    if _is_disabled(raw) and source != f"${ENV_JAX_DIR}":  # JAX reads its own "none" as a path
        return None, f"{source} (disabled)"
    return _absolute(raw, source, origin.parent if origin is not None else None), source


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
    if isinstance(value, float) and not math.isfinite(value):
        raise JaxCacheConfigError(f"{source}: {key!r} must be finite, got {raw!r}")
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


_ISA_FLAG = re.compile(
    r"^(sse\w*|ssse3|avx\w*|fma\w*|bmi\w*|amx\w*|f16c|popcnt|movbe|abm|lzcnt|aes|pclmulqdq|"
    r"sha_ni|adx|neon|asimd\w*|sve\w*|fp16|bf16|i8mm|dotprod|fphp|asimdhp)$"
)


def host_cpu_fingerprint() -> str:
    """Eight hex digits of the host CPU's ISA feature set; no JAX, no backend init.

    Hashes the ISA-relevant tokens of the ``flags`` (x86) or ``Features`` (arm) line of
    ``/proc/cpuinfo`` plus the machine type. Mitigation and bookkeeping flags (``md_clear``,
    ``ibpb``, ...) are left out so they do not split the cache on identical hardware.

    Raises:
        JaxCacheConfigError: ``/proc/cpuinfo`` is unreadable or has no recognisable ISA flags,
            so every host would hash alike.
    """
    flags = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                name, _, value = line.partition(":")
                if name.strip().lower() in {"flags", "features"}:
                    kept = sorted(t for t in value.split() if _ISA_FLAG.match(t))
                    flags = " ".join(kept)
                    break
    except OSError:
        pass
    if not flags:
        raise JaxCacheConfigError(
            "namespace 'cpu-flags' needs readable ISA flags in /proc/cpuinfo; they are "
            "unavailable on this host (use namespace 'auto' or 'none')"
        )
    return hashlib.sha256(f"{platform.machine()}|{flags}".encode()).hexdigest()[:8]


def _jaxlib_version() -> str:
    try:
        return importlib.metadata.version("jaxlib")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _namespaced(base: Path, namespace: str) -> Path:
    if namespace == "none":
        return base
    versioned = base / f"jaxlib-{_jaxlib_version()}"
    if namespace == "cpu-flags":
        return versioned / f"cpu-{host_cpu_fingerprint()}"
    return versioned


def _prepare_directory(directory: Path) -> None:
    """Create ``directory`` and prove it is writable; warn when others can write to it."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".xtrax_write_probe_") as probe:
            probe.write(b"\0" * 4096)
            probe.flush()
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
        if _applied == setting and _live_dir() == str(setting.dir):
            return setting
        assert setting.dir is not None
        if (
            setting.max_size_bytes is not None
            and setting.max_size_bytes != -1
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


def _live_dir() -> str | None:
    import jax

    return jax.config.jax_compilation_cache_dir


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
    # jax._src is private; xtrax pins jax < 0.12 and the tests exercise these names.
    from jax._src import compilation_cache

    started = getattr(compilation_cache, "_cache", None) is not None
    previous_max = _applied.max_size_bytes if _applied is not None else None
    if started and existing and existing != new_dir:
        logger.warning(
            "JAX compilation cache was already in use at %s; switching to %s and resetting "
            "JAX's cache object (programs compiled before this call are not written to the "
            "new directory)",
            existing,
            new_dir,
        )
        compilation_cache.reset_cache()
    elif started and existing == new_dir and setting.max_size_bytes != previous_max:
        logger.warning(
            "JAX compilation cache at %s is already in use; resetting JAX's cache object so the "
            "new max_size_bytes=%s applies",
            new_dir,
            setting.max_size_bytes,
        )
        compilation_cache.reset_cache()
    elif existing and existing != new_dir:
        logger.info("replacing JAX's jax_compilation_cache_dir=%s with %s", existing, new_dir)
    jax.config.update("jax_compilation_cache_dir", new_dir)
    if setting.min_compile_time_secs is not None:
        jax.config.update(
            "jax_persistent_cache_min_compile_time_secs", setting.min_compile_time_secs
        )
    if setting.min_entry_size_bytes is not None:
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", setting.min_entry_size_bytes)
    if setting.max_size_bytes is not None:
        jax.config.update("jax_compilation_cache_max_size", setting.max_size_bytes)


class JaxCachePruneError(RuntimeError):
    """``prune_corrupt`` refused to run; nothing was deleted."""


# jax/_src/lru_cache.py names an entry "<key>-cache" (and, with eviction, "<key>-atime");
# jax/_src/cache_key.py builds the key as "<module name>-<64 hex digits of sha256>".
_ENTRY_NAME = re.compile(r"^.+-[0-9a-f]{64}-cache$")
_ENTRY_SUFFIX = "-cache"
_ATIME_SUFFIX = "-atime"
_TIME_PREFIX_BYTES = 4  # jax/_src/compilation_cache.py: big-endian compile time before payload
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_SCAN_DEPTH = 3  # <dir>/jaxlib-<ver>/cpu-<hash>/<entry>


def _zstd_decoder() -> Callable[[bytes], bytes] | None:
    """A zstd decompress function from whatever this process has, or None.

    JAX writes zstd when ``zstandard`` (or Python 3.14's ``compression.zstd``) is importable
    in the WRITING process and zlib otherwise, and the filename does not say which, so the
    pruner must not assume its own codec matches the writer's.
    """
    try:  # optional modules, resolved by name so type checkers need not know them
        return importlib.import_module("compression.zstd").decompress  # Python 3.14+
    except ImportError:
        pass
    try:
        return importlib.import_module("zstandard").ZstdDecompressor().decompress
    except ImportError:
        return None


def _classify_entry(data: bytes, zstd_decode: Callable[[bytes], bytes] | None) -> bool:
    """True when ``data`` decodes under any codec JAX could have used, False when it is corrupt.

    Raises:
        JaxCachePruneError: the entry has the zstd magic but no zstd decoder is importable, so
            its validity cannot be judged here. Deleting it would be a guess.
    """
    if not data:
        return False
    if data.startswith(_ZSTD_MAGIC):
        if zstd_decode is None:
            raise JaxCachePruneError(
                "found zstd-compressed cache entries but neither 'zstandard' nor "
                "compression.zstd is importable here; run prune_corrupt in an environment with "
                "the writer's codec. Nothing was deleted."
            )
        try:
            payload = zstd_decode(data)
        except Exception:  # zstandard.ZstdError or compression.zstd.ZstdError: a bad stream
            return False
    else:
        try:
            payload = zlib.decompress(data)
        except zlib.error:
            return False
    return len(payload) > _TIME_PREFIX_BYTES


def _entries(root: Path) -> list[Path]:
    """JAX-key-named ``*-cache`` files at most ``_SCAN_DEPTH`` levels below ``root``."""
    found: list[Path] = []
    for depth in range(_SCAN_DEPTH):
        pattern = "/".join(["*"] * depth + [f"*{_ENTRY_SUFFIX}"])
        found.extend(
            p
            for p in root.glob(pattern)
            if _ENTRY_NAME.match(p.name) and p.is_file() and not p.is_symlink()
        )
    return sorted(found)


def prune_corrupt(
    directory: Path | str | None = None,
    *,
    min_age_secs: float = 600.0,
    dry_run: bool = False,
) -> list[Path]:
    """Delete cache entries that do not decode (a writer preempted mid-write), return them.

    JAX writes entries non-atomically and never rewrites an existing key, so a truncated entry
    is a permanent miss that logs a warning in every process. Run this between campaign
    attempts, not while processes are writing: a file modified less than ``min_age_secs``
    ago is skipped because it may still be in flight.

    Only files named like JAX keys (``<name>-<64 hex>-cache``, at most three levels below
    ``directory``, symlinks skipped) are examined, and the matching ``-atime`` file is removed
    with a deleted entry. An entry counts as valid if the zlib decoder or, for the zstd magic,
    a zstd decoder accepts it; JAX picks the codec in the writing process, so one that this
    process cannot decode (zstd entries, no zstd package) makes the call refuse before it
    deletes anything. The scan runs to completion before any file is removed.

    Args:
        directory: A JAX cache directory. ``None`` uses the directory the current
            configuration resolves to (namespace included). An explicit directory must
            contain JAX-key-named entries or a ``jaxlib-*`` subdirectory.
        min_age_secs: Skip entries newer than this; 0 examines everything.
        dry_run: Report without deleting.

    Returns:
        The corrupt entry paths found (deleted unless ``dry_run``).

    Raises:
        JaxCacheConfigError: ``directory`` is ``None`` and no cache is configured, or the
            given directory does not look like a JAX cache.
        JaxCachePruneError: an entry needs a zstd decoder that is not importable.
    """
    import time

    if directory is None:
        setting = resolve()
        if setting.dir is None:
            raise JaxCacheConfigError(
                f"no JAX compilation cache is configured ({setting.source}); pass a directory"
            )
        root = setting.dir
        trusted = True
    else:
        root = Path(directory)
        trusted = False
    entries = _entries(root)
    if not trusted and not entries and not any(root.glob("jaxlib-*")):
        raise JaxCacheConfigError(
            f"{root} does not look like a JAX compilation cache (no '<name>-<64 hex>-cache' "
            "entries and no jaxlib-* subdirectory); refusing to prune it"
        )
    now = time.time()
    zstd_decode = _zstd_decoder()
    corrupt = [
        entry
        for entry in entries
        if not (min_age_secs > 0 and now - entry.stat().st_mtime < min_age_secs)
        and not _classify_entry(entry.read_bytes(), zstd_decode)
    ]
    if not dry_run:
        for entry in corrupt:
            entry.unlink(missing_ok=True)
            atime = entry.with_name(entry.name.removesuffix(_ENTRY_SUFFIX) + _ATIME_SUFFIX)
            atime.unlink(missing_ok=True)
    return corrupt
