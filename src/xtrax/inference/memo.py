"""memoize_jaxpr — content-keyed value cache for pure jitted callables (spec §4.2).

Opt-in value memoization around JAX-callable functions. The caller ATTESTS
purity by choosing to wrap; a static jaxpr screen raises MemoImpurityError for
DETECTABLE violations (stateful primitives, host callbacks, unkeyed random
usage). Documented blind spots: out-of-trace closure state, time/I/O, objects
with unstable traced representations.

Cache key (spec §4.2.2):
    (program_digest, pytree structure, per-leaf digests, salt, environment stamp)
- program_digest = sha256(normalized str(ClosedJaxpr) + folded const values in
  ascending const-var order). str() is deterministic but NOT injective over
  array constants; const folding closes that hole (OBJ-R2-02).
- leaf digests reuse update_array_digest's canonicalize+tobytes core plus a
  weak_type/dtype extension; Python scalars digest via (type-tag, repr);
  strings NFC-normalized; anything else -> MemoKeyUnsupportedLeafError.
- environment stamp bounds RNG-implementation and autotune/atomics drift.

Async safety: block_on_miss=True (default) blocks outputs before store;
False stores futures (pipelining mode — memory bound is entry-count only).

Spot-checking recomputes via the UNWRAPPED callable and compares numerically
(allclose): XLA autotuning/atomics legitimately produce bit variation.
"""

from __future__ import annotations

import hashlib
import threading
import unicodedata
import warnings
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast, overload

import jax
import numpy as np

from xtrax.inference.errors import (
    MemoDonationError,
    MemoImpurityError,
    MemoKeyUnsupportedLeafError,
    MemoMultiDeviceError,
    MemoStalenessError,
)

__all__ = [
    "MemoPolicy",
    "memoize_jaxpr",
]

_STAMP_OVERRIDE_ENV = "XTRAX_MEMO_STAMP_OVERRIDE"
_WARMUP_CALLS = 8


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------


def _require_stamp_override_env() -> None:
    import os

    if os.environ.get(_STAMP_OVERRIDE_ENV) != "1":
        raise ValueError(
            f"MemoPolicy._stamp_override is a TEST SEAM; set {_STAMP_OVERRIDE_ENV}=1 "
            "to use it outside tests (production keys must derive from the real "
            "environment)."
        )


@dataclass(frozen=True)
class MemoPolicy:
    """Tuning + safety policy for memoize_jaxpr (spec §4.2)."""

    max_entries: int = 128
    salt: str = ""
    spot_check_every: int = 0  # K=recompute every Kth call via unwrapped fn
    spot_check_rtol: float = 1e-5
    spot_check_atol: float = 1e-8
    copy_on_return: bool = False
    block_on_miss: bool = True
    slow_ratio_warn: float = 1.0
    _stamp_override: str | None = None

    def __post_init__(self) -> None:
        if self._stamp_override is not None:
            _require_stamp_override_env()
            if not self._stamp_override:
                raise ValueError("_stamp_override must be non-empty when provided")


# ---------------------------------------------------------------------------
# Key construction
# ---------------------------------------------------------------------------


def _environment_stamp() -> str:
    import jax

    backend = jax.default_backend()
    devices = jax.local_devices()
    device_kind = devices[0].device_kind if devices else "unknown"
    return "|".join(
        (
            jax.__version__,
            getattr(jax.lib, "xla_extension_version", "unknown"),
            backend,
            device_kind,
            str(devices[0].id if devices else -1),
        )
    )


def _leaf_digest(leaf: Any, sink: hashlib._Hash) -> None:
    """Fold one pytree leaf into the digest stream (spec §4.2.2 item 3)."""
    if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
        # House primitive core (zarr_integrity.update_array_digest recipe):
        # canonicalize then C-order bytes.
        arr = np.asarray(leaf)
        canon = np.ascontiguousarray(arr)
        sink.update(canon.dtype.name.encode())
        sink.update(repr(canon.shape).encode())
        sink.update(canon.tobytes(order="C"))
        weak = getattr(leaf, "weak_type", False)
        sink.update(f"|wt={bool(weak)}".encode())
        return
    if isinstance(leaf, (str, bytes)):
        tag = "bytes" if isinstance(leaf, bytes) else "str"
        val = leaf.decode("utf-8", "surrogatepass") if isinstance(leaf, bytes) else leaf
        norm = unicodedata.normalize("NFC", val)
        sink.update(f"{tag}:{norm}".encode("utf-8", "surrogatepass"))
        return
    if isinstance(leaf, (int, float, bool)):
        sink.update(f"{type(leaf).__name__}({leaf!r})".encode())
        return
    raise MemoKeyUnsupportedLeafError(
        f"Unsupported pytree leaf type {type(leaf).__name__!r} for memo key; "
        "admission restricted to arrays, scalars, bools and strings."
    )


def _key_digest(parts: tuple) -> str:
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, str):
            h.update(part.encode())
        elif isinstance(part, bytes):
            h.update(part)
        else:
            for leaf in part:
                _leaf_digest(leaf, h)
        h.update(b"\x1f")
    return h.hexdigest()


def _program_digest(closed) -> str:
    """sha256 over normalized program text PLUS folded const values."""
    h = hashlib.sha256()
    h.update(" ".join(str(closed).split()).encode())
    for const in closed.consts:  # ascending const-var declaration order
        _leaf_digest(np.asarray(const), h)
    return h.hexdigest()


def _pytree_leaves(args: tuple, kwargs: dict) -> list:
    return jax.tree_util.tree_leaves((args, kwargs))


def _structure_token(args: tuple, kwargs: dict) -> tuple:
    structure = jax.tree_util.tree_structure((args, kwargs))
    return (structure,)


# ---------------------------------------------------------------------------
# Impurity screen
# ---------------------------------------------------------------------------

_STATEFUL_PRIMITIVES = {"pjit_sprng_fold_in", "state_primal"}
_CALLBACK_PRIMITIVES = {
    "call",
    "pure_callback",
    "io_callback",
    "host_callback",
    "callback",
}
_RANDOM_PRIMITIVES = {"random_bits", "threefry2x32_p", "rng_bit_generator", "random_seed"}


def _screen_jaxpr(closed) -> None:
    """Raise MemoImpurityError on detectably impure primitives. Traversal walks
    with an explicit stack via _iter_subjaxprs exclusively, covering tuple/list-
    valued params generically and with no depth cap."""
    banned = _STATEFUL_PRIMITIVES | _CALLBACK_PRIMITIVES | _RANDOM_PRIMITIVES

    offenders: list[tuple[str, str]] = []  # (primitive_name, path) pairs
    stack: list[tuple[Any, str]] = [(closed.jaxpr, "jaxpr")]
    while stack:
        jaxpr_obj, jaxpr_path = stack.pop()
        for eqn in jaxpr_obj.eqns:
            eqn_path = f"{jaxpr_path}.{_eqn_label(eqn)}"
            name = eqn.primitive.name
            if name in banned:
                offenders.append((name, eqn_path))
            for param_name, param_val in eqn.params.items():
                for sub_path, sub_jaxpr in _iter_subjaxprs(param_val, param_name):
                    stack.append((sub_jaxpr, f"{eqn_path}.{sub_path}"))
    if offenders:
        names = sorted(set(name for name, _ in offenders))
        paths = ", ".join(path for _, path in offenders)
        raise MemoImpurityError(
            f"Function rejected by purity screen: stateful/callback/random "
            f"primitives present: {names}. If you believe this "
            "function is pure, restructure to avoid these primitives; wrapping "
            f"is the purity attestation.\nPaths: {paths}"
        )


# ---------------------------------------------------------------------------
# Donation screen (spec §3.3/§3.4)
# ---------------------------------------------------------------------------

_DonationSite = tuple[str, str, tuple[int, ...], tuple[int, ...]]


def _iter_subjaxprs(value: Any, path: str = ""):
    """Yield (path, subjaxpr) for every subjaxpr-like object (anything
    exposing ``.eqns``) reachable from ``value``, recursing into tuple/list
    values (e.g. ``lax.cond``'s ``branches``) with NO depth cap.

    Shared traversal used by both _screen_jaxpr and _screen_donation to cover
    tuple/list-valued params generically (e.g. lax.cond's branches) while
    respecting structural nesting at any depth.
    """
    stack: list[tuple[str, Any]] = [(path, value)]
    while stack:
        p, v = stack.pop()
        if hasattr(v, "eqns"):
            yield p, v
        elif isinstance(v, (tuple, list)):
            for i, item in enumerate(v):
                stack.append((f"{p}[{i}]" if p else f"[{i}]", item))


def _eqn_label(eqn) -> str:
    name = eqn.params.get("name")
    if name:
        return f"{eqn.primitive.name}[name={name}]"
    return eqn.primitive.name


def _wrapped_leaf_indices(eqn, operand_indices: tuple[int, ...], closed_invars) -> tuple[int, ...]:
    """Identity lookup only (not provenance tracing): for each donated
    operand of a TOP-LEVEL equation, find `j` such that the operand var IS
    (identity) `closed_invars[j]` — an index into the flattened pytree leaves
    of the wrapped function's positional args."""
    out: list[int] = []
    for i in operand_indices:
        operand = eqn.invars[i]
        for j, invar in enumerate(closed_invars):
            if operand is invar:
                out.append(j)
                break
    return tuple(out)


def _eqn_donation_sites(eqn, path: str, top_level: bool, closed_invars) -> list[_DonationSite]:
    """Both donation carriers (D3, P5): `donated_invars` (jit/pjit/scan/...)
    and `device_put`'s `copy_semantics` DONATE_INPUT element. Duck-typed on
    `.name` — the private `ArrayCopySemantics` type is never imported."""
    sites: list[_DonationSite] = []

    donated_invars = eqn.params.get("donated_invars")
    if donated_invars:
        idxs = tuple(i for i, d in enumerate(donated_invars) if d)
        if idxs:
            wrapped = _wrapped_leaf_indices(eqn, idxs, closed_invars) if top_level else ()
            sites.append((path, "donated_invars", idxs, wrapped))

    copy_semantics = eqn.params.get("copy_semantics")
    if copy_semantics:
        idxs = tuple(
            i for i, cs in enumerate(copy_semantics) if getattr(cs, "name", None) == "DONATE_INPUT"
        )
        if idxs:
            wrapped = _wrapped_leaf_indices(eqn, idxs, closed_invars) if top_level else ()
            sites.append((path, "copy_semantics", idxs, wrapped))

    return sites


def _donation_message(sites: tuple[_DonationSite, ...]) -> str:
    lines = [
        f"  - {path} (carrier={carrier}, eqn_operand_indices={op_idx}, "
        f"wrapped_input_leaf_indices={leaf_idx})"
        for path, carrier, op_idx, leaf_idx in sites
    ]
    return (
        "Function rejected by donation screen (spec §4.2 item 6): "
        "memoize_jaxpr never admits a function whose traced jaxpr carries a "
        "donation marker, at any depth. Sites:\n"
        + "\n".join(lines)
        + "\nRemedy: remove donate_argnums/donate_argnames/device_put(donate=True) "
        "from functions wrapped by memoize_jaxpr (spec §4.2 item 6)."
    )


def _screen_donation(closed) -> None:
    """Collect ALL donation-hazard sites (both carriers, at any depth), then
    raise once (§3.3 "conservative" rule). Traversal walks with an explicit
    stack via `_iter_subjaxprs` exclusively, so it covers tuple/list-valued
    params generically and has no depth cap."""
    sites: list[_DonationSite] = []
    closed_invars = closed.jaxpr.invars
    stack: list[tuple[Any, str, bool]] = [(closed.jaxpr, "jaxpr", True)]
    while stack:
        jaxpr_obj, jaxpr_path, top_level = stack.pop()
        for eqn in jaxpr_obj.eqns:
            eqn_path = f"{jaxpr_path}.{_eqn_label(eqn)}"
            sites.extend(_eqn_donation_sites(eqn, eqn_path, top_level, closed_invars))
            for param_name, param_val in eqn.params.items():
                for sub_path, sub_jaxpr in _iter_subjaxprs(param_val, param_name):
                    stack.append((sub_jaxpr, f"{eqn_path}.{sub_path}", False))
    if sites:
        raise MemoDonationError(_donation_message(tuple(sites)), sites=tuple(sites))


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


@dataclass
class _MemoEntry:
    value: Any  # output(s); future when block_on_miss=False
    ready: bool


@dataclass
class MemoStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_cached: int = 0
    last_hit_age: int = 0
    cum_op_seconds: float = 0.0
    cum_hash_seconds: float = 0.0
    calls: int = 0
    spot_check_mismatches: int = 0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class _MemoCore:
    """Per-wrapped-function cache state."""

    def __init__(self, fn: Callable, policy: MemoPolicy) -> None:
        if policy.max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if len(jax.local_devices()) > 1:
            raise MemoMultiDeviceError(
                "memoize_jaxpr v1 supports single-device sessions only "
                f"(found {len(jax.local_devices())} local devices). See spec N5."
            )
        self.fn = fn
        self.policy = policy
        self.lock = threading.Lock()
        self.cache: OrderedDict[str, _MemoEntry] = OrderedDict()
        self.stats = MemoStats()
        self.program_digest: str | None = None
        self.stamp: str = (
            policy._stamp_override if policy._stamp_override is not None else _environment_stamp()
        )
        self.screen_latched_error: MemoImpurityError | None = None
        self.calls_since_wrap = 0
        self.warned_slow = False

    # -- key handling ------------------------------------------------------

    def _ensure_program(self, args: tuple) -> None:
        if self.program_digest is None:

            def probe(*a):  # trace-only; no concrete execution of user code paths
                return self.fn(*a)

            try:
                closed = jax.make_jaxpr(probe)(*args)
            except TypeError as exc:
                raise MemoKeyUnsupportedLeafError(
                    "Argument cannot be traced as an abstract array (unsupported "
                    f"leaf type for memo keys): {exc}"
                ) from exc
            _screen_jaxpr(closed)
            _screen_donation(closed)
            self.program_digest = _program_digest(closed)

    def build_key(self, args: tuple, kwargs: dict) -> str:
        assert self.program_digest is not None, (
            "build_key requires _ensure_program to have run first"
        )
        h = hashlib.sha256()
        h.update(self.program_digest.encode())
        h.update(repr(_structure_token(args, kwargs)).encode())
        for leaf in _pytree_leaves(args, kwargs):
            _leaf_digest(leaf, h)
        h.update(self.policy.salt.encode())
        h.update(self.stamp.encode())
        return h.hexdigest()

    # -- LRU ---------------------------------------------------------------

    def _store(self, key: str, entry: _MemoEntry) -> None:
        while len(self.cache) >= self.policy.max_entries:
            _, evicted = self.cache.popitem(last=False)
            self.stats.evictions += 1
            if evicted.ready:
                self.stats.bytes_cached -= _entry_bytes(evicted.value)
        self.cache[key] = entry
        if entry.ready:
            self.stats.bytes_cached += _entry_bytes(entry.value)

    # -- call --------------------------------------------------------------

    def call(self, *args: Any, **kwargs: Any) -> Any:
        import time

        with self.lock:
            if self.screen_latched_error is not None:
                raise self.screen_latched_error
            self.calls_since_wrap += 1

        with self.lock:
            if self.stats.spot_check_mismatches > 0:
                raise MemoStalenessError("spot_check_mismatches > 0: poisoned until .memo_reset()")
        t0 = time.perf_counter()
        try:
            self._ensure_program(args)
        except MemoImpurityError as exc:
            with self.lock:
                self.screen_latched_error = exc  # latch (OBJ-R2-08)
            raise
        hash_seconds = time.perf_counter() - t0

        key = self.build_key(args, kwargs)

        with self.lock:
            entry = self.cache.get(key)
            if entry is not None:
                self.stats.hits += 1
                self.stats.last_hit_age = 0
                self.cache.move_to_end(key)
                next_call_number = self.stats.calls + 1
                do_spot = (
                    self.policy.spot_check_every > 0
                    and next_call_number % self.policy.spot_check_every == 0
                )
                if not do_spot:
                    value = self._finalize_hit(entry)
                    self.stats.calls += 1
                    return value
                value = self._finalize_hit(entry)  # ready before unlock
            else:
                do_spot = False
                value = None
        if do_spot:
            # calls counter incremented inside _maybe_spot_check_unlocked
            self._maybe_spot_check_unlocked(key)
            return value
        self.stats.misses += 1

        op_start = time.perf_counter()
        raw_out = self.fn(*args, **kwargs)
        if self.policy.block_on_miss:
            jax.block_until_ready(raw_out)
        ready = self.policy.block_on_miss
        op_seconds = time.perf_counter() - op_start

        # OBJ-R1-01/R1-13 (§3.4): the store keeps today's protection (a
        # defensive copy when copy_on_return=True), but the MISS return is
        # raw_out itself — not the cached object — so it may alias the
        # caller's argument exactly as the unwrapped fn would, and no
        # second copy is made.
        stored_value = _copy_array_leaves(raw_out) if self.policy.copy_on_return else raw_out
        with self.lock:
            self._store(key, _MemoEntry(value=stored_value, ready=ready))
            self.stats.cum_hash_seconds += hash_seconds
            self.stats.cum_op_seconds += op_seconds
            self.stats.calls += 1
            self._maybe_warn_slow()

        return raw_out

    # -- helpers -----------------------------------------------------------

    def _finalize_hit(self, entry: _MemoEntry) -> Any:
        if not entry.ready:
            jax.block_until_ready(entry.value)
            entry.ready = True
        # OBJ-R1-01 (§3.4): a hit — plain or spot-checked — returns a
        # defensive copy when copy_on_return=True, so a consumer that
        # donates/.delete()s the returned value cannot corrupt the cache
        # entry. Default (False): unchanged, identity return.
        if self.policy.copy_on_return:
            return _copy_array_leaves(entry.value)
        return entry.value

    def _record_op_time(self, t0: float) -> None:
        # Hit path: caller already holds the lock; just count the call.
        self.stats.calls += 1

    def _maybe_warn_slow(self) -> None:
        if self.policy.block_on_miss is False:
            return  # honest attribution impossible without readiness waits
        if self.warned_slow or self.stats.calls <= _WARMUP_CALLS:
            return
        if self.stats.cum_hash_seconds <= 0:
            return
        ratio = self.stats.cum_hash_seconds / max(self.stats.cum_op_seconds, 1e-12)
        if ratio > self.policy.slow_ratio_warn:
            self.warned_slow = True
            warnings.warn(
                f"memoize_jaxpr measured hash/op seconds ratio {ratio:.3g} "
                f"(> {self.policy.slow_ratio_warn}): key-building costs exceed "
                "compute — caching is currently SLOWER than recomputing for "
                "this workload.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _maybe_spot_check_unlocked(self, key: str) -> None:
        with self.lock:
            if self.stats.spot_check_mismatches > 0:
                raise MemoStalenessError("spot_check_mismatches > 0: poisoned until .memo_reset()")
            entry = self.cache.get(key)
            if entry is None:
                return
            cached_value = entry.value
        # Recompute OUTSIDE the lock via UNWRAPPED fn (fresh closure read).
        # Spot-check uses the SAME key => same inputs; we must recompute from
        # stored args. We deliberately store nothing beyond the value, so
        # spot-check replays only when the wrapper was given inputs; therefore
        # we stash the latest args on the core at call time.
        if self._last_args is None:
            return
        fresh = self.fn(*self._last_args)
        jax.block_until_ready(fresh)
        cached_flat = jax.tree_util.tree_leaves(cached_value)
        fresh_flat = jax.tree_util.tree_leaves(fresh)
        ok = len(cached_flat) == len(fresh_flat) and all(
            _numeric_equal(c, f, self.policy.spot_check_rtol, self.policy.spot_check_atol)
            for c, f in zip(cached_flat, fresh_flat)
        )
        with self.lock:
            if not ok:
                self.stats.spot_check_mismatches += 1
                evicted = self.cache.pop(key, None)
                if evicted is not None and evicted.ready:
                    self.stats.bytes_cached -= _entry_bytes(evicted.value)
                raise MemoStalenessError(
                    "Spot-check mismatch: cached output diverged from fresh "
                    "computation. Entry evicted; counter poisoned until "
                    ".memo_reset()."
                )

    _last_args: tuple | None = None

    def reset(self) -> None:
        with self.lock:
            self.stats.spot_check_mismatches = 0

    def rewrap(self) -> None:
        with self.lock:
            self.screen_latched_error = None


def _entry_bytes(value: Any) -> int:
    total = 0
    for leaf in _safe_leaves(value):
        if hasattr(leaf, "nbytes"):
            total += int(leaf.nbytes)
    return total


def _safe_leaves(value: Any) -> list:
    try:
        return jax.tree_util.tree_leaves(value)
    except Exception:
        return [value]


def _numeric_equal(a: Any, b: Any, rtol: float, atol: float) -> bool:
    import jax.numpy as jnp

    try:
        return bool(jnp.allclose(a, b, rtol=rtol, atol=atol))
    except Exception:
        return a is b


def _copy_array_leaves(value: Any) -> Any:
    """Copy `jax.Array`/`np.ndarray` leaves only (spec §3.4, AC-13). Other
    leaves are immutable Python scalars or strings and are returned as-is —
    calling `.copy()` on them would raise `AttributeError`."""

    def _copy_leaf(x: Any) -> Any:
        if isinstance(x, (jax.Array, np.ndarray)):
            return x.copy()
        return x

    return jax.tree_util.tree_map(_copy_leaf, value)


class MemoizedCallable(Protocol):
    """Interface exposed by the callable `memoize_jaxpr` returns.

    `wrap()` attaches these as instance attributes on a plain function object
    (not a class), so ty cannot infer them from `wrapped`'s own definition;
    `wrap()` casts its return value to this Protocol to declare the real
    contract instead of suppressing the resulting attribute errors.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def memo_get_stats(self) -> dict[str, Any]: ...
    def memo_reset(self) -> None: ...
    def memo_rewrap(self) -> None: ...

    _memo_core: _MemoCore
    _memo_stats_holder: dict[str, Any]


@overload
def memoize_jaxpr(fn: Callable, *, policy: MemoPolicy | None = None) -> MemoizedCallable: ...
@overload
def memoize_jaxpr(
    fn: None = None, *, policy: MemoPolicy | None = None
) -> Callable[[Callable], MemoizedCallable]: ...
def memoize_jaxpr(
    fn: Callable | None = None, *, policy: MemoPolicy | None = None
) -> MemoizedCallable | Callable[[Callable], MemoizedCallable]:
    """Decorator/wrapper adding a content-keyed value cache (opt-in attestation).

    Usage:
        @memoize_jaxpr
        def score(x): ...

        wrapped = memoize_jaxpr(score, policy=MemoPolicy(max_entries=64))

    Bare usage (`fn` given directly) returns the memoized callable. Called
    with only `policy=` (no `fn`), it returns a decorator — the
    `@memoize_jaxpr(policy=...)` form.

    The wrapped callable exposes ``.memo_get_stats()`` (dict), ``.memo_reset()``
    (clears poisoned spot-check counter) and ``.memo_rewrap()`` (clears a
    latched impurity error).
    """
    pol = policy if policy is not None else MemoPolicy()

    def _wrap(f: Callable) -> MemoizedCallable:
        import inspect

        core = _MemoCore(f, pol)
        stats_holder: dict[str, Any] = {}
        # Zero-arg (or all-default) callables can be screened at wrap time:
        if not inspect.signature(f).parameters:
            try:
                core._ensure_program(())
            except MemoImpurityError:
                # AC-9: bare `raise` preserves the exact exception type
                # (e.g. MemoDonationError), rather than downcasting to the
                # base class.
                raise

        @functools_wraps(f)
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            core._last_args = args
            result = core.call(*args, **kwargs)
            stats_holder["snapshot"] = dict(core.stats.as_dict())
            return result

        def _get_stats() -> dict[str, Any]:
            return dict(core.stats.as_dict())

        # Attribute assignment below is genuinely dynamic (monkey-patching a
        # plain function object); `Any` is the honest type for the write site.
        # The `cast` on return declares the actual static contract to callers.
        dynamic: Any = _wrapped
        dynamic.memo_get_stats = _get_stats
        dynamic.memo_reset = core.reset
        dynamic.memo_rewrap = core.rewrap
        dynamic._memo_core = core
        dynamic._memo_stats_holder = stats_holder
        return cast(MemoizedCallable, _wrapped)

    if fn is not None:
        return _wrap(fn)
    return _wrap


def functools_wraps(f: Callable) -> Callable[[Callable], Callable]:
    """Minimal functools.wraps replacement avoiding extra import surface."""
    import functools

    return functools.wraps(f)
