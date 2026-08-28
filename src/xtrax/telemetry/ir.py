"""Capture the IR a run actually executed, at the compile boundary.

Three artifacts, in increasing distance from the source and decreasing
portability:

``jaxpr``          JAX's own semantic record. The most readable view, and the one
                   that survives a jaxlib upgrade with the clearest meaning.
``stablehlo``      What ``jax.export`` serialises: portable, replayable, and the
                   artifact a retrospective audit can actually re-run.
``optimized_hlo``  Post-XLA, device-specific. Measured at ~8 KB gzipped for a
                   96-layer model, so it is affordable despite being the largest
                   raw artifact -- but it is meaningless on different hardware,
                   which is why it is opt-in rather than default.

**Capture happens once per shape signature, never per step.** JAX compiles once
per signature, so hooking the compile boundary is both the cheapest and the
complete place to observe: every distinct computation passes through it exactly
once. Calling this from inside a step loop would multiply cost by the step count
and mint a blob per step, destroying the dedup property the store's cost model
depends on (``store.check_population`` exists to catch that mistake).

Every artifact is captured independently, and a failure in one degrades only
that one. ``jax.export`` legitimately refuses some traceable functions (custom
calls, unsupported primitives), and a run must never die because its observer
could not serialise it -- but the gap is recorded as an ``IRRef`` with
``mode="skipped"`` and a reason, never omitted.
"""

import os
from typing import Any

from xtrax.telemetry.record import IR_FULL, IR_HASH_ONLY, IR_SKIPPED, IRRef
from xtrax.telemetry.store import BlobStore, digest_of

IR_KIND_JAXPR = "jaxpr"
IR_KIND_STABLEHLO = "stablehlo"
IR_KIND_OPTIMIZED_HLO = "optimized_hlo"

# 50 MB, matching cisternal's DEFAULT_MAX_SNAPSHOT_BYTES. Measurements put a
# 96-layer model's whole IR set at 159 KB raw, so this cap is a runaway guard
# rather than an expected boundary.
DEFAULT_MAX_IR_BYTES = 50 * 1024 * 1024

# XTRAX_CAPTURE_IR values.
CAPTURE_FULL = "full"
CAPTURE_FULL_OPTIMIZED = "full+optimized"
CAPTURE_HASH = "hash"
CAPTURE_NONE = "none"
_VALID_CAPTURE_MODES = frozenset({CAPTURE_FULL, CAPTURE_FULL_OPTIMIZED, CAPTURE_HASH, CAPTURE_NONE})

_DEFAULT_KINDS = (IR_KIND_JAXPR, IR_KIND_STABLEHLO)


class IRCaptureMode:
    """Resolved capture policy for one run."""

    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.enabled = raw != CAPTURE_NONE
        self.store_text = raw in (CAPTURE_FULL, CAPTURE_FULL_OPTIMIZED)
        self.kinds: tuple[str, ...] = ()
        if self.enabled:
            self.kinds = _DEFAULT_KINDS
            if raw == CAPTURE_FULL_OPTIMIZED:
                self.kinds = (*_DEFAULT_KINDS, IR_KIND_OPTIMIZED_HLO)


def resolve_capture_mode(value: "str | None" = None) -> IRCaptureMode:
    """Resolve XTRAX_CAPTURE_IR into a policy, defaulting to ``full``.

    An unrecognised value falls back to ``full`` rather than raising: a typo in
    an environment variable should not stop a training run, and defaulting
    *towards* capture keeps the failure safe.
    """
    raw = (value if value is not None else os.environ.get("XTRAX_CAPTURE_IR") or "").strip().lower()
    if raw not in _VALID_CAPTURE_MODES:
        raw = CAPTURE_FULL
    return IRCaptureMode(raw)


def _require_text(text: "str | None", what: str) -> str:
    """Reject a None render rather than storing the literal string "None".

    ``mlir_module()`` and ``as_text()`` are both typed Optional. A None that
    slipped through would be stored as a four-byte blob that looks like a
    successful capture -- strictly worse than a skipped artifact, which at least
    says why.
    """
    if text is None:
        raise ValueError(f"{what} produced no text")
    return str(text)


def _render_jaxpr(fn: Any, args: "tuple[Any, ...]") -> str:  # noqa: ANN401
    import jax

    try:
        return str(jax.make_jaxpr(fn)(*args))
    except Exception:  # noqa: BLE001 - fall through to the equinox-aware path
        import equinox as eqx

        # An eqx.Module whose fields are all traceable flattens fine under plain
        # make_jaxpr (static fields ride along as pytree aux data). This path is
        # for the modules where that is not true -- a callable field, or a bool
        # that must stay static for control flow -- which is precisely what
        # eqx.filter_* exists to handle.
        return str(eqx.filter_make_jaxpr(fn)(*args)[0])


def _render_stablehlo(fn: Any, args: "tuple[Any, ...]") -> str:  # noqa: ANN401
    import jax

    try:
        # Same call shape as xtrax.cli.export.run_export, deliberately: one way
        # to lower to StableHLO in this codebase, not two that can drift.
        return _require_text(jax.export.export(jax.jit(fn))(*args).mlir_module(), "jax.export")
    except Exception:  # noqa: BLE001 - fall through to lowering directly
        import equinox as eqx

        # jax.export refuses some functions it can still lower, and it cannot
        # take an eqx.filter_jit wrapper at all. .lower().as_text() yields the
        # same pre-optimization StableHLO for both.
        lowered = eqx.filter_jit(fn).lower(*args)
        return _require_text(lowered.as_text(), "filter_jit lowering")


def _render_optimized_hlo(fn: Any, args: "tuple[Any, ...]") -> str:  # noqa: ANN401
    import jax

    # .compile() genuinely invokes XLA, so this is the expensive renderer and the
    # reason optimized HLO is opt-in. The idiom matches tiling/estimators.py and
    # profiling/trace.py. Note there is no equinox fallback: equinox's Compiled
    # wrapper exposes no as_text(), so a function that only lowers under
    # filter_jit degrades to a skipped artifact with that reason recorded.
    return _require_text(jax.jit(fn).lower(*args).compile().as_text(), "XLA compilation")


_RENDERERS = {
    IR_KIND_JAXPR: _render_jaxpr,
    IR_KIND_STABLEHLO: _render_stablehlo,
    IR_KIND_OPTIMIZED_HLO: _render_optimized_hlo,
}


def _capture_one(
    kind: str,
    fn: Any,  # noqa: ANN401
    args: "tuple[Any, ...]",
    *,
    store: "BlobStore | None",
    max_bytes: int,
    store_text: bool,
) -> IRRef:
    """Capture one artifact, degrading to a recorded reason on any failure."""
    renderer = _RENDERERS[kind]
    try:
        text = renderer(fn, args)
    except Exception as exc:  # noqa: BLE001 - observer must not kill the observed
        return IRRef(
            kind=kind,
            sha256="",
            bytes=0,
            mode=IR_SKIPPED,
            reason=f"{type(exc).__name__} while rendering {kind}: {exc}",
        )

    sha256, raw_len = digest_of(text)
    if not store_text or store is None:
        return IRRef(
            kind=kind,
            sha256=sha256,
            bytes=raw_len,
            mode=IR_HASH_ONLY,
            reason="capture mode records fingerprints only",
        )
    if raw_len > max_bytes:
        return IRRef(
            kind=kind,
            sha256=sha256,
            bytes=raw_len,
            mode=IR_HASH_ONLY,
            reason=f"{raw_len} bytes exceeds the {max_bytes}-byte capture cap",
        )
    try:
        store.put(text)
    except OSError as exc:
        return IRRef(
            kind=kind,
            sha256=sha256,
            bytes=raw_len,
            mode=IR_HASH_ONLY,
            reason=f"blob store write failed: {exc}",
        )
    return IRRef(kind=kind, sha256=sha256, bytes=raw_len, mode=IR_FULL)


def capture_ir(
    fn: Any,  # noqa: ANN401
    *abstract_inputs: Any,  # noqa: ANN401
    store: "BlobStore | None" = None,
    max_bytes: int = DEFAULT_MAX_IR_BYTES,
    mode: "IRCaptureMode | None" = None,
) -> "tuple[IRRef, ...]":
    """Capture the IR for ``fn`` at ``abstract_inputs``, once per shape signature.

    Returns one :class:`IRRef` per configured artifact. Never raises: an artifact
    that could not be rendered or stored comes back with ``mode="skipped"`` or
    ``mode="hash_only"`` and a reason, so the caller can downgrade the run's
    telemetry_status without losing the explanation.

    ``abstract_inputs`` may be concrete arrays or ``jax.ShapeDtypeStruct``; both
    trace identically, and only shape/dtype affect the resulting IR.
    """
    policy = mode if mode is not None else resolve_capture_mode()
    if not policy.enabled:
        return ()
    return tuple(
        _capture_one(
            kind,
            fn,
            abstract_inputs,
            store=store,
            max_bytes=max_bytes,
            store_text=policy.store_text,
        )
        for kind in policy.kinds
    )


def degraded_reason(refs: "tuple[IRRef, ...]") -> "str | None":
    """A single reason string if any artifact fell short of full capture.

    Returns None when every artifact was captured in full, which is the only
    case that leaves a run's telemetry_status at ``complete``.
    """
    partial = [ref for ref in refs if ref.mode != IR_FULL]
    if not partial:
        return None
    return "; ".join(f"{ref.kind}: {ref.reason or ref.mode}" for ref in partial)
