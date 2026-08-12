"""Eval-closure-invariant / full-closure SHA lock (T2-05, #2181, AC-7, F2, PM-1).

PM-1's failure mode (`.praxia/docs/specs/260702_design-the-2181-agentic-algorithm-evolut.md`):
an earlier design hashed the evaluator *file*, not its complete closure -- a candidate mutated
an uncovered surface (a splits path, a normalization constant, a dependency version, a data file
the evaluator imported) and the judge drifted undetected. This module is the fix: per iteration,
recompute a SHA-256 over the evaluator's COMPLETE CLOSURE (evaluator code + splits + metric-defs
+ pinned deps + config) and compare against the locked manifest, reject any candidate mutation of
a protected (closure-member) path, and flag any evaluator file read that wasn't declared in the
closure manifest.

Non-recoverable, by design (PM-1): a closure-lock violation is not a candidate rejection (which a
future loop might retry, log, or drop) -- it means the fitness oracle itself may have drifted, so
every `EvaluatorClosureDriftError` subclass here is meant to propagate and HALT, not be silently
caught and treated like an ordinary `EvaluateFn` result.

Extension seam (mirrors `xtrax.loop.admission`'s stance on `admit_candidate`): this module raises
on drift and stops there. It does not write an escalation artifact, notify anyone, or decide what
a human-in-the-loop response looks like -- no #2181 loop controller exists yet to define that, and
inventing one now (retry? campaign-ledger entry? paused campaign?) would be speculative and likely
wrong-shaped once the real controller lands. A future controller catches
`EvaluatorClosureDriftError` and owns the actual HALT+escalation workflow.

Read-tracking caveat: `track_reads` uses a single process-wide `sys.addaudithook` (installed once,
lazily) gated by a contextvar, because CPython audit hooks cannot be individually removed once
added (PEP 578) -- there is no way to install/tear down a hook per call. The hook is a cheap
early-return outside an active `track_reads()` context, so this is safe, but it does mean the hook
exists for the rest of the process once any caller has used `track_reads` once.
"""

import contextvars
import hashlib
import json
import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from xtrax.stages.evaluate import Candidate, EvaluateFn, FrozenContext

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PINNED_DEPS_SOURCE = ROOT / "uv.lock"


class EvaluatorClosureDriftError(Exception):
    """Base for AC-7's non-recoverable closure-lock violations. HALT the loop; do not retry.

    See module docstring: this is deliberately not a candidate-rejection result -- catching this
    broadly (rather than per-subclass) is the loop-level "stop everything" signal.
    """


class ClosureHashMismatchError(EvaluatorClosureDriftError):
    """The recomputed closure hash no longer matches the locked manifest (PM-1's core case)."""


class ProtectedPathMutatedError(EvaluatorClosureDriftError):
    """The candidate touched a path that is part of the locked evaluator closure."""


class UnlistedReadError(EvaluatorClosureDriftError):
    """The evaluator read a path that was not declared in the locked closure manifest."""


@dataclass(frozen=True, slots=True)
class ClosureManifest:
    """A locked snapshot of the evaluator's complete closure and its aggregate SHA-256.

    `evaluator_paths`/`split_paths`/`metric_def_paths` are the closure's declared file
    membership -- together they define both what gets hashed and what counts as "protected"
    (a candidate must not touch any of them) and "declared" (the evaluator may read only these).
    `config` is the run's config mapping, folded into the hash via the same canonicalization
    `xtrax.cli.hash.config_hash` uses (`json.dumps(..., sort_keys=True, default=str)`), but the
    full canonical JSON is folded into the aggregate digest rather than a separately-truncated
    12-char hash -- the aggregate closure hash is the security-relevant value here, so truncating
    an intermediate leg would only shrink collision resistance for no benefit.
    """

    evaluator_paths: tuple[Path, ...]
    split_paths: tuple[Path, ...]
    metric_def_paths: tuple[Path, ...]
    pinned_deps_source: Path
    config: Mapping[str, Any]
    closure_hash: str


def _canonical_config_bytes(config: Mapping[str, Any]) -> bytes:
    return json.dumps(config, sort_keys=True, default=str).encode()


def _resolved(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    return tuple(p.resolve() for p in paths)


def build_closure_manifest(
    *,
    evaluator_paths: tuple[Path, ...],
    split_paths: tuple[Path, ...],
    metric_def_paths: tuple[Path, ...],
    config: Mapping[str, Any],
    pinned_deps_source: Path = DEFAULT_PINNED_DEPS_SOURCE,
) -> ClosureManifest:
    """Read every declared closure member off disk now and compute the aggregate SHA-256.

    Reads are sorted (by resolved path) before hashing so the result is stable regardless of
    the order the caller passed paths in.
    """
    evaluator_paths = _resolved(evaluator_paths)
    split_paths = _resolved(split_paths)
    metric_def_paths = _resolved(metric_def_paths)
    pinned_deps_source = pinned_deps_source.resolve()

    digest = hashlib.sha256()
    all_declared = sorted({*evaluator_paths, *split_paths, *metric_def_paths})
    for path in all_declared:
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    digest.update(pinned_deps_source.read_bytes())
    digest.update(_canonical_config_bytes(config))

    return ClosureManifest(
        evaluator_paths=evaluator_paths,
        split_paths=split_paths,
        metric_def_paths=metric_def_paths,
        pinned_deps_source=pinned_deps_source,
        config=config,
        closure_hash=digest.hexdigest(),
    )


def _protected_paths(locked: ClosureManifest) -> frozenset[Path]:
    return frozenset({*locked.evaluator_paths, *locked.split_paths, *locked.metric_def_paths})


def verify_closure(
    locked: ClosureManifest,
    *,
    current_config: Mapping[str, Any],
    candidate_touched_paths: frozenset[Path] = frozenset(),
) -> None:
    """Recompute the closure hash from `locked`'s declared paths and `current_config`.

    `current_config` is deliberately a separate argument, not `locked.config` -- recomputing
    against the locked manifest's own stored config would make config drift undetectable by
    construction (the recomputed hash would always match). Passing the config actually in effect
    for this iteration is what makes this a real per-iteration check, not a tautology.

    Raises:
        ClosureHashMismatchError: the aggregate hash drifted, or a declared closure path no
            longer exists on disk (also treated as drift, not a bare `FileNotFoundError`).
        ProtectedPathMutatedError: `candidate_touched_paths` intersects the closure's own
            member paths.
    """
    try:
        current = build_closure_manifest(
            evaluator_paths=locked.evaluator_paths,
            split_paths=locked.split_paths,
            metric_def_paths=locked.metric_def_paths,
            config=current_config,
            pinned_deps_source=locked.pinned_deps_source,
        )
    except FileNotFoundError as exc:
        msg = f"closure member vanished from disk since locking: {exc}"
        raise ClosureHashMismatchError(msg) from exc

    if current.closure_hash != locked.closure_hash:
        msg = (
            f"closure hash drift: locked={locked.closure_hash} current={current.closure_hash} "
            "-- evaluator code, splits, metric-defs, pinned deps, or config changed since lock"
        )
        raise ClosureHashMismatchError(msg)

    touched_protected = frozenset(candidate_touched_paths) & _protected_paths(locked)
    if touched_protected:
        msg = f"candidate touched protected closure path(s): {sorted(touched_protected)}"
        raise ProtectedPathMutatedError(msg)


_read_sink: contextvars.ContextVar[set[Path] | None] = contextvars.ContextVar(
    "xtrax_loop_closure_lock_read_sink", default=None
)
_hook_installed = False


def _audit_open_hook(event: str, args: tuple[object, ...]) -> None:
    if event != "open":
        return
    sink = _read_sink.get()
    if sink is None:
        return
    path_arg = args[0] if args else None
    if isinstance(path_arg, str):
        sink.add(Path(path_arg).resolve())
    elif isinstance(path_arg, bytes):
        sink.add(Path(os.fsdecode(path_arg)).resolve())
    elif isinstance(path_arg, os.PathLike):
        # ty narrows a bare (unparameterized) os.PathLike to PathLike[object], which
        # doesn't structurally match fsdecode's PathLike[str] | PathLike[bytes]
        # overloads even though every real __fspath__ implementation returns str | bytes.
        path_like = cast("os.PathLike[str] | os.PathLike[bytes]", path_arg)
        sink.add(Path(os.fsdecode(path_like)).resolve())


def _ensure_hook_installed() -> None:
    global _hook_installed
    if not _hook_installed:
        sys.addaudithook(_audit_open_hook)
        _hook_installed = True


@contextmanager
def track_reads() -> Iterator[set[Path]]:
    """Yield a set that accumulates every file path opened while this context is active.

    Scoped via a contextvar, not by installing/removing the underlying audit hook (CPython
    cannot remove an audit hook once added -- see module docstring). Records both reads and
    writes; AC-7 only requires enumerating reads, but distinguishing open-mode robustly across
    both `open()` (str mode) and `os.open()` (int flags) is fragile, and for a non-recoverable
    safety gate an occasional false-positive-on-a-write is far cheaper than a missed read.
    """
    _ensure_hook_installed()
    observed: set[Path] = set()
    token = _read_sink.set(observed)
    try:
        yield observed
    finally:
        _read_sink.reset(token)


def guarded_evaluate(
    locked: ClosureManifest,
    evaluator: EvaluateFn[FrozenContext, Candidate],
    frozen_context: FrozenContext,
    candidate: Candidate,
    *,
    current_config: Mapping[str, Any],
    candidate_touched_paths: frozenset[Path] = frozenset(),
) -> dict[str, float]:
    """The composed AC-7 gate: hash check + protected-path check, then a read-enumerated call.

    Runs `verify_closure` (hash + protected-path) *before* invoking `evaluator` at all -- a
    drifted evaluator must never execute, not merely have its result discarded. Then calls
    `evaluator` under `track_reads` and raises `UnlistedReadError` if it opened anything outside
    the locked closure's declared paths.
    """
    verify_closure(
        locked, current_config=current_config, candidate_touched_paths=candidate_touched_paths
    )

    declared = _protected_paths(locked)
    with track_reads() as observed:
        result = evaluator(frozen_context, candidate)

    unlisted = frozenset(observed) - declared
    if unlisted:
        msg = f"evaluator read undeclared path(s): {sorted(unlisted)}"
        raise UnlistedReadError(msg)

    return result


__all__ = [
    "ClosureHashMismatchError",
    "ClosureManifest",
    "EvaluatorClosureDriftError",
    "ProtectedPathMutatedError",
    "UnlistedReadError",
    "build_closure_manifest",
    "guarded_evaluate",
    "track_reads",
    "verify_closure",
]
