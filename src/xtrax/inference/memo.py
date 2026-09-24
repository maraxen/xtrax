"""memoize_jaxpr — content-keyed value cache for pure jitted callables (spec §4.2).

Opt-in value memoization around JAX-callable functions. The caller ATTESTS
purity by choosing to wrap; a static jaxpr screen raises MemoImpurityError for
DETECTABLE violations (stateful primitives, host callbacks, unkeyed random
usage). Documented blind spots: out-of-trace closure state, time/I/O, objects
with unstable traced representations.

Cache key (spec 260922 §3.1-§3.3):
    (screen_digest, pytree structure, per-leaf digests, salt, environment stamp)
- screen_digest is resolved PER CALL SIGNATURE (shape/dtype/treedef/kwargs, not
  just the first call's) by `_MemoCore._ensure_screened`, which traces and
  screens (purity + donation) either in ABSTRACT mode (array leaves AND
  exact-`int`/`float` scalars traced abstractly) or, on a fallback, in STATIC
  mode (scalars held static; only arrays traced). `bool`/enum/`str`/`bytes`
  leaves are always held static. screen_digest itself is
  sha256(normalized str(ClosedJaxpr) + folded const values in ascending
  const-var order) — see `_program_digest`.
- leaf digests reuse update_array_digest's canonicalize+tobytes core plus a
  container-type/weak_type/dtype extension; Python scalars digest via
  (type-tag, repr); `str`/`bytes` digest exactly (no normalization); anything
  else -> MemoKeyUnsupportedLeafError.
- environment stamp bounds RNG-implementation and autotune/atomics drift.

Async safety: block_on_miss=True (default) blocks outputs before store;
False stores futures (pipelining mode — memory bound is entry-count only).

Spot-checking recomputes via the UNWRAPPED callable and compares numerically
(allclose): XLA autotuning/atomics legitimately produce bit variation.
"""

from __future__ import annotations

import hashlib
import threading
import warnings
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NoReturn, Protocol, cast, overload

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
# Bound on _MemoCore._screened (spec §3.2). Read through the module global at
# USE TIME (never captured into __init__ or a default arg), so a test can
# `monkeypatch.setattr(memo_module, "_MAX_SCREENED_SIGNATURES", N)`.
_MAX_SCREENED_SIGNATURES = 1024
# Cached int bounds (T5 optimization)
_INT32_INFO = np.iinfo(np.int32)
_INT64_INFO = np.iinfo(np.int64)


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
    """Fold one pytree leaf into the digest stream (spec §3.3).

    Arrays fold the container type before the dtype (D-4, AC-13b), so an
    `np.ndarray` and an equal `jax.Array` digest differently. `str`/`bytes`
    digest EXACTLY, with no Unicode normalization (D-3, AC-6/AC-6b).
    """
    if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
        # House primitive core (zarr_integrity.update_array_digest recipe):
        # container type, then canonicalize + C-order bytes.
        arr = np.asarray(leaf)
        canon = np.ascontiguousarray(arr)
        sink.update(type(leaf).__qualname__.encode())
        sink.update(canon.dtype.name.encode())
        sink.update(repr(canon.shape).encode())
        sink.update(canon.tobytes(order="C"))
        weak = getattr(leaf, "weak_type", False)
        sink.update(f"|wt={bool(weak)}".encode())
        return
    if isinstance(leaf, bytes):
        # CR-1: length-prefixed so back-to-back leaf digests in one stream
        # (build_key has no separator between leaves) cannot alias across a
        # boundary, e.g. digest("a")+digest("str:b") == digest("astr:")+digest("b")
        # under the old unprefixed scheme.
        sink.update(b"bytes:%d:" % len(leaf) + leaf)
        return
    if isinstance(leaf, str):
        enc = leaf.encode("utf-8", "surrogatepass")
        sink.update(b"str:%d:" % len(enc) + enc)
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


# ---------------------------------------------------------------------------
# Per-signature leaf classification and trace modes (spec §3.1)
# ---------------------------------------------------------------------------


# Sentinels for the `_MemoCore._screened` table (spec §3.2). `_NEEDS_STATIC`
# is a persisted table VALUE (an ABSTRACT token whose trace needs STATIC
# fallback); `_MISSING` is only ever a local `dict.get` default and never
# stored, so the two are never confused.
class _NeedsStaticType:
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug aid only
        return "_NEEDS_STATIC"


_NEEDS_STATIC = _NeedsStaticType()
_MISSING = object()


def _fits_default_int(value: int, x64: bool | None = None) -> bool:
    """True iff `value` lies in the range of the canonical default int dtype
    (spec §3.1 item 2, G4/OBJ-R2-01). If `x64` is None, reads the live x64 setting;
    the default int dtype is int64 under x64 and int32 otherwise, which is what
    `canonicalize_dtype(np.int64)` returns."""
    if x64 is None:
        x64 = bool(jax.config.jax_enable_x64)
    info = _INT64_INFO if x64 else _INT32_INFO
    return info.min <= value <= info.max


def _static_exact_token(leaf: Any) -> str | bytes:
    """Exact-value token for a leaf that is always held static (spec §3.1
    item 3): repr() for numbers, exact (unnormalized) bytes for str/bytes.

    CR-1: unlike `_leaf_digest`, this does NOT need a length prefix. Its
    result is always embedded as one element of a Python tuple descriptor
    (`("static", type_name, exact_value_token)`), which is itself one element
    of the larger `descriptors` tuple compared/hashed as a structured Python
    object (never concatenated into a flat byte/string stream). Tuple
    equality is positional and structural, so two different leaves can never
    alias into the same descriptor regardless of their encoded length.
    """
    if isinstance(leaf, str):
        return leaf.encode("utf-8", "surrogatepass")
    if isinstance(leaf, bytes):
        return leaf
    return repr(leaf)


def _classify_leaf(leaf: Any, mode: str, x64: bool | None = None) -> tuple[str, tuple]:
    """Classify one flattened `(args, kwargs)` leaf for a screened-signature
    token (spec §3.1, in order):

    1. **arr** — has `.shape` and `.dtype` (covers numpy scalars, `jax.Array`,
       `np.ndarray`, `np.bool_`, ...). Always traced.
    2. **dyn** — `type(leaf) is float` or `type(leaf) is int` (EXACT type
       check, so `bool`/enum/numpy-scalar subclasses never land here). Traced
       in ABSTRACT mode; held static in STATIC mode.
    3. **static** — any other `bool`/`int`/`float`/`str`/`bytes` instance.
       Always held static (D-2).
    4. anything else raises `MemoKeyUnsupportedLeafError` before tracing.

    `mode` ("ABSTRACT" or "STATIC") only changes the descriptor for a "dyn"
    leaf; `kind` itself is mode-independent, so a caller can determine
    "has a traceable scalar" from either mode's classification.

    Returns `(kind, descriptor)`.
    """
    if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
        return "arr", (
            "arr",
            type(leaf).__qualname__,
            tuple(leaf.shape),
            np.dtype(leaf.dtype).name,
            bool(getattr(leaf, "weak_type", False)),
        )
    if type(leaf) is float:
        if mode == "ABSTRACT":
            return "dyn", ("dyn", "float")
        return "dyn", ("static", "float", repr(leaf))
    if type(leaf) is int:
        if mode == "ABSTRACT":
            return "dyn", ("dyn", "int", _fits_default_int(leaf, x64=x64))
        return "dyn", ("static", "int", repr(leaf))
    if isinstance(leaf, (bool, int, float, str, bytes)):
        return "static", ("static", type(leaf).__qualname__, _static_exact_token(leaf))
    raise MemoKeyUnsupportedLeafError(
        f"Unsupported pytree leaf type {type(leaf).__name__!r} for memo key; "
        "admission restricted to arrays, scalars, bools and strings."
    )


def _mode_token(
    leaves: list, treedef: Any, mode: str
) -> tuple[tuple[str, Any, tuple, bool], tuple[int, ...], bool]:
    """Classify every leaf under `mode` and return `(token, traced_positions,
    has_dyn)`. `traced_positions` is which flat-leaf indices get traced by
    `_trace_closed`: array leaves always; traceable scalars ("dyn") only in
    ABSTRACT mode (spec §3.1 "ABSTRACT/STATIC mode" paragraphs).

    CR-6: the token carries `bool(jax.config.jax_enable_x64)`, read per call.
    Without it, toggling the x64 flag between calls of the SAME signature
    reused the screened-signature table entry from before the toggle (the
    digest lookup never re-traces), serving a stale digest/key pair keyed to
    the wrong dtype's traced program.
    """
    x64 = bool(jax.config.jax_enable_x64)
    kinds: list[str] = []
    descriptors: list[tuple] = []
    for leaf in leaves:
        kind, descriptor = _classify_leaf(leaf, mode, x64=x64)
        kinds.append(kind)
        descriptors.append(descriptor)
    traced_kinds = ("arr", "dyn") if mode == "ABSTRACT" else ("arr",)
    traced = tuple(i for i, k in enumerate(kinds) if k in traced_kinds)
    tag = "A" if mode == "ABSTRACT" else "S"
    token = (tag, treedef, tuple(descriptors), x64)
    has_dyn = any(k == "dyn" for k in kinds)
    return token, traced, has_dyn


def _trace_closed(fn: Callable, leaves: list, treedef: Any, traced_positions: tuple[int, ...]):
    """Trace `fn` with only `traced_positions` abstracted; every other leaf
    stays closed-over as its concrete Python value (spec §3.1 "The trace
    calls a probe that takes only the traced leaves"). Raises whatever
    `jax.make_jaxpr` raises, uncaught — the caller wraps this call alone in
    try/except so screen errors are never mistaken for trace failures."""

    def probe(*traced_vals: Any) -> Any:
        full = list(leaves)
        for pos, val in zip(traced_positions, traced_vals):
            full[pos] = val
        a, k = jax.tree_util.tree_unflatten(treedef, full)
        return fn(*a, **k)

    return jax.make_jaxpr(probe)(*(leaves[p] for p in traced_positions))


def _raise_classified(exc: Exception) -> NoReturn:
    """Classify a failed FINAL trace (spec §3.1 "Classification of a failed
    final trace"): STATIC mode, or ABSTRACT mode when there are no traceable
    scalars. Raises `MemoKeyUnsupportedLeafError` chained `from exc`, or
    re-raises `exc` unchanged.

    CR-4: bare `IndexError` was too broad — a user function indexing a plain
    Python tuple/list out of range (a real bug, nothing to do with an array
    argument's value) was misreported as "consults the value of an array
    argument". The array-value idiom that legitimately raises an `IndexError`
    subclass is boolean indexing (`x[x > 0]`), which JAX raises as
    `jax.errors.NonConcreteBooleanIndexError` specifically — narrow the check
    to that. Any other `IndexError` falls through to the final `raise exc`
    below and propagates unchanged.
    """
    classified: tuple[type[Exception], ...] = (
        jax.errors.ConcretizationTypeError,
        jax.errors.TracerIntegerConversionError,
        jax.errors.TracerArrayConversionError,
    )
    if hasattr(jax.errors, "NonConcreteBooleanIndexError"):
        classified = (*classified, jax.errors.NonConcreteBooleanIndexError)
    if isinstance(exc, classified):
        raise MemoKeyUnsupportedLeafError(
            "Function consults the value of an array argument in Python "
            "(branching, indexing, or conversion), which cannot be screened."
        ) from exc
    if isinstance(exc, TypeError):
        raise MemoKeyUnsupportedLeafError(
            "Argument cannot be traced as an abstract array (unsupported "
            f"leaf type for memo keys): {exc}"
        ) from exc
    raise exc


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


# ---------------------------------------------------------------------------
# Donation screen (spec §3.3/§3.4)
# ---------------------------------------------------------------------------

_DonationSite = tuple[str, str, tuple[int, ...], tuple[int, ...]]


def _iter_subjaxprs(value: Any, path: str = ""):
    """Yield (path, subjaxpr) for every subjaxpr-like object (anything
    exposing ``.eqns``) reachable from ``value``, recursing into tuple/list
    values (e.g. ``lax.cond``'s ``branches``) with NO depth cap.

    Shared traversal used by _screen_program to cover tuple/list-valued params
    generically (e.g. lax.cond's branches) while respecting structural nesting
    at any depth.
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


def _wrapped_leaf_indices(
    eqn,
    operand_indices: tuple[int, ...],
    closed_invars,
    invar_to_leaf: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    """Identity lookup only (not provenance tracing): for each donated
    operand of a TOP-LEVEL equation, find `j` such that the operand var IS
    (identity) `closed_invars[j]` — an index into `closed.jaxpr.invars`.

    CR-3: `closed.jaxpr.invars` corresponds only to the TRACED flat leaves
    (`traced_positions` from `_mode_token`/`_trace_closed`), not to every
    flattened `(args, kwargs)` leaf, whenever some leaves are held static
    (bool/enum/str/bytes leaves, or scalars in STATIC mode). `invar_to_leaf`
    maps invar index `j` -> the true flat leaf index (`traced_positions[j]`).
    `None` means identity (every leaf was traced, e.g. an arrays-only call),
    which keeps `j == leaf index` and leaves existing array-only callers
    unaffected.
    """
    out: list[int] = []
    for i in operand_indices:
        operand = eqn.invars[i]
        for j, invar in enumerate(closed_invars):
            if operand is invar:
                out.append(invar_to_leaf[j] if invar_to_leaf is not None else j)
                break
    return tuple(out)


def _eqn_donation_sites(
    eqn,
    path: str,
    top_level: bool,
    closed_invars,
    invar_to_leaf: tuple[int, ...] | None = None,
) -> list[_DonationSite]:
    """Both donation carriers (D3, P5): `donated_invars` (jit/pjit/scan/...)
    and `device_put`'s `copy_semantics` DONATE_INPUT element. Duck-typed on
    `.name` — the private `ArrayCopySemantics` type is never imported."""
    sites: list[_DonationSite] = []

    donated_invars = eqn.params.get("donated_invars")
    if donated_invars:
        idxs = tuple(i for i, d in enumerate(donated_invars) if d)
        if idxs:
            wrapped = (
                _wrapped_leaf_indices(eqn, idxs, closed_invars, invar_to_leaf) if top_level else ()
            )
            sites.append((path, "donated_invars", idxs, wrapped))

    copy_semantics = eqn.params.get("copy_semantics")
    if copy_semantics:
        idxs = tuple(
            i for i, cs in enumerate(copy_semantics) if getattr(cs, "name", None) == "DONATE_INPUT"
        )
        if idxs:
            wrapped = (
                _wrapped_leaf_indices(eqn, idxs, closed_invars, invar_to_leaf) if top_level else ()
            )
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


def _screen_program(closed, invar_to_leaf: tuple[int, ...] | None = None) -> None:
    """Screen a closed jaxpr for impurity and donation hazards in one walk.

    Raises MemoImpurityError if the program contains detectably impure primitives
    (stateful, callback, or unkeyed random). Raises MemoDonationError if it contains
    donation markers at any depth. Traversal walks with an explicit stack via
    `_iter_subjaxprs` exclusively, so it covers tuple/list-valued params generically
    and has no depth cap.

    CR-3: `invar_to_leaf` (typically the caller's `traced_positions`) maps a
    top-level invar index to the true flat `(args, kwargs)` leaf index, for
    callers that traced fewer leaves than the flattened arg count (D-2
    static leaves). `None` (the default) keeps the old identity mapping.
    """
    banned = _STATEFUL_PRIMITIVES | _CALLBACK_PRIMITIVES | _RANDOM_PRIMITIVES

    offenders: list[tuple[str, str]] = []  # (primitive_name, path) pairs
    sites: list[_DonationSite] = []
    closed_invars = closed.jaxpr.invars
    stack: list[tuple[Any, str, bool]] = [(closed.jaxpr, "jaxpr", True)]

    while stack:
        jaxpr_obj, jaxpr_path, top_level = stack.pop()
        for eqn in jaxpr_obj.eqns:
            eqn_path = f"{jaxpr_path}.{_eqn_label(eqn)}"
            name = eqn.primitive.name
            if name in banned:
                offenders.append((name, eqn_path))
            sites.extend(
                _eqn_donation_sites(
                    eqn, eqn_path, top_level, closed_invars, invar_to_leaf if top_level else None
                )
            )
            for param_name, param_val in eqn.params.items():
                for sub_path, sub_jaxpr in _iter_subjaxprs(param_val, param_name):
                    stack.append((sub_jaxpr, f"{eqn_path}.{sub_path}", False))

    if offenders:
        names = sorted(set(name for name, _ in offenders))
        paths = ", ".join(path for _, path in offenders)
        raise MemoImpurityError(
            f"Function rejected by purity screen: stateful/callback/random "
            f"primitives present: {names}. If you believe this "
            "function is pure, restructure to avoid these primitives; wrapping "
            f"is the purity attestation.\nPaths: {paths}"
        )
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
        # Screened-signature table (spec §3.2): token -> digest, or
        # _NEEDS_STATIC while only the ABSTRACT attempt has been resolved.
        self._screened: OrderedDict[tuple, str | _NeedsStaticType] = OrderedDict()
        self.stamp: str = (
            policy._stamp_override if policy._stamp_override is not None else _environment_stamp()
        )
        self.screen_latched_error: MemoImpurityError | None = None
        self.calls_since_wrap = 0
        self.warned_slow = False

    # -- key handling ------------------------------------------------------

    def _insert(self, token: tuple, value: str | _NeedsStaticType) -> None:
        """Insert/refresh `token -> value` and evict LRU-oldest past the
        module cap (spec §3.2). Caller must hold `self.lock`. The cap is read
        as the bare module global on every call, so a test can monkeypatch it."""
        self._screened[token] = value
        self._screened.move_to_end(token)
        while len(self._screened) > _MAX_SCREENED_SIGNATURES:
            self._screened.popitem(last=False)

    def _resolve_static(
        self,
        leaves: list,
        treedef: Any,
        *,
        came_from_fallback: bool,
        abstract_token: tuple,
        abstract_exc: Exception | None = None,
    ) -> str:
        """§3.2 step 3: resolve via the STATIC token, tracing/screening on a
        miss. Only inserts `abstract_token -> _NEEDS_STATIC` if this call was
        reached via step 2's ABSTRACT-trace fallback (`came_from_fallback`).

        CR-5: `abstract_exc`, when this call came from the ABSTRACT-trace
        fallback, is the ABSTRACT-mode failure that triggered the retry. If
        the STATIC retry ALSO fails, that failure otherwise fully discards
        the ABSTRACT exception (§3.1's fallback rule retries on ANY
        exception, so the two failures can be unrelated bugs). We attach it
        as a note on the STATIC-mode exception before classifying/raising it,
        so it stays visible instead of vanishing.
        """
        static_token, static_traced, _ = _mode_token(leaves, treedef, "STATIC")

        with self.lock:
            sval = self._screened.get(static_token, _MISSING)
            if sval is not _MISSING:
                self._screened.move_to_end(static_token)

        if sval is not _MISSING:
            if came_from_fallback:
                with self.lock:
                    self._insert(abstract_token, _NEEDS_STATIC)
            return cast(str, sval)

        try:
            closed = _trace_closed(self.fn, leaves, treedef, static_traced)
        except Exception as exc:
            if abstract_exc is not None:
                exc.add_note(
                    "ABSTRACT-mode trace failed first: "
                    f"{type(abstract_exc).__name__}: {abstract_exc}"
                )
            _raise_classified(exc)  # never returns; no insert (§3.2 step 5)

        _screen_program(closed, static_traced)  # CR-3: invars == static_traced positions
        digest = _program_digest(closed)
        with self.lock:
            self._insert(static_token, digest)
            if came_from_fallback:
                self._insert(abstract_token, _NEEDS_STATIC)
        return digest

    def _ensure_screened_flat(self, leaves: list, treedef: Any) -> str:
        """Resolve the digest for THIS call's signature (spec §3.2),
        screening it (purity + donation) if this is the first time this
        signature has been seen. Takes pre-flattened leaves and treedef.
        Locked lookups/inserts; tracing and screening always run unlocked.
        Returns the digest read/produced; never re-reads the table afterward
        (a concurrent eviction could remove the entry)."""
        abstract_token, abstract_traced, has_dyn = _mode_token(leaves, treedef, "ABSTRACT")

        with self.lock:
            val = self._screened.get(abstract_token, _MISSING)
            if val is not _MISSING:
                self._screened.move_to_end(abstract_token)

        if val is not _MISSING and val is not _NEEDS_STATIC:
            return cast(str, val)
        if val is _NEEDS_STATIC:
            return self._resolve_static(
                leaves, treedef, came_from_fallback=False, abstract_token=abstract_token
            )

        # ABSTRACT miss: trace outside the lock.
        try:
            closed = _trace_closed(self.fn, leaves, treedef, abstract_traced)
        except Exception as exc:
            if has_dyn:
                return self._resolve_static(
                    leaves,
                    treedef,
                    came_from_fallback=True,
                    abstract_token=abstract_token,
                    abstract_exc=exc,
                )
            _raise_classified(exc)  # never returns; no insert (§3.2 step 5)

        _screen_program(closed, abstract_traced)  # CR-3: invars == abstract_traced positions
        digest = _program_digest(closed)
        with self.lock:
            self._insert(abstract_token, digest)
        return digest

    def _ensure_screened(self, args: tuple, kwargs: dict) -> str:
        """Resolve the digest for THIS call's signature (spec §3.2),
        screening it (purity + donation) if this is the first time this
        signature has been seen. Locked lookups/inserts; tracing and
        screening always run unlocked. Returns the digest read/produced;
        never re-reads the table afterward (a concurrent eviction could
        remove the entry)."""
        leaves, treedef = jax.tree_util.tree_flatten((args, kwargs))
        return self._ensure_screened_flat(leaves, treedef)

    def build_key(
        self,
        digest: str,
        args: tuple,
        kwargs: dict,
        *,
        leaves: list | None = None,
        treedef: Any = None,
    ) -> str:
        """Hash the digest THIS CALL resolved via `_ensure_screened`, then the
        structure, leaf digests, x64 flag, salt and stamp (spec §3.2). Runs
        outside `self.lock`, exactly as on main.

        If leaves and treedef are provided, uses them; otherwise computes them
        via tree_flatten.

        CR-6: `jax.config.jax_enable_x64` is folded in, read per call, so a
        cache entry stored under one x64 setting is never served to a call
        made under the other (arrays digest their own concrete dtype, but a
        Python scalar's OUTPUT dtype under x64 does not show up anywhere else
        in the key)."""
        if leaves is None or treedef is None:
            leaves, treedef = jax.tree_util.tree_flatten((args, kwargs))
        h = hashlib.sha256()
        h.update(digest.encode())
        h.update(repr((treedef,)).encode())
        for leaf in leaves:
            _leaf_digest(leaf, h)
        h.update(f"|x64={bool(jax.config.jax_enable_x64)}".encode())
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
        leaves, treedef = jax.tree_util.tree_flatten((args, kwargs))
        try:
            digest = self._ensure_screened_flat(leaves, treedef)
        except MemoImpurityError as exc:
            with self.lock:
                self.screen_latched_error = exc  # latch (OBJ-R2-08)
            raise
        hash_seconds = time.perf_counter() - t0

        key = self.build_key(digest, args, kwargs, leaves=leaves, treedef=treedef)

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
            # CR-2: calls counter is incremented exactly once inside
            # _maybe_spot_check_unlocked (both the ok and mismatch paths, and
            # the entry-evicted-before-recompute early return).
            self._maybe_spot_check_unlocked(key, args, kwargs)
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

    def _maybe_spot_check_unlocked(self, key: str, args: tuple, kwargs: dict) -> None:
        with self.lock:
            if self.stats.spot_check_mismatches > 0:
                raise MemoStalenessError("spot_check_mismatches > 0: poisoned until .memo_reset()")
            entry = self.cache.get(key)
            if entry is None:
                # CR-2: this call was still counted, even though the entry
                # was evicted out from under it before the recompute.
                self.stats.calls += 1
                return
            cached_value = entry.value
        # Recompute OUTSIDE the lock via UNWRAPPED fn (fresh closure read).
        # Replay uses this call's own (args, kwargs) (#5231).
        fresh = self.fn(*args, **kwargs)
        jax.block_until_ready(fresh)
        cached_flat = jax.tree_util.tree_leaves(cached_value)
        fresh_flat = jax.tree_util.tree_leaves(fresh)
        ok = len(cached_flat) == len(fresh_flat) and all(
            _numeric_equal(c, f, self.policy.spot_check_rtol, self.policy.spot_check_atol)
            for c, f in zip(cached_flat, fresh_flat)
        )
        with self.lock:
            # CR-2: a spot-checked hit is still a call, exactly once, whether
            # it matches or mismatches.
            self.stats.calls += 1
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
                core._ensure_screened((), {})
            except MemoImpurityError:
                # AC-9: bare `raise` preserves the exact exception type
                # (e.g. MemoDonationError), rather than downcasting to the
                # base class.
                raise

        @functools_wraps(f)
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
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
