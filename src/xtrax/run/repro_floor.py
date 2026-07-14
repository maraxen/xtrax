"""Seed-pinned re-run-N reproducibility floor (S5, spec §5.1, backlog #3498).

The no-oracle gate for a data product: when there is no independent oracle to
verify a computation against (bathos's `oracle_match` attestation kind), the
next-strongest evidence is that the computation is *byte-identically*
reproducible given its pinned seed. This module owns exactly that check --
re-run a caller-supplied computation ``rerun_count`` times with a **pinned**
seed, digest each run's Zarr output with
:func:`xtrax.run.zarr_integrity.zarr_content_digest`, and assert every digest
is identical (the first digest becomes the product's ``content_hash`` unless
one is otherwise fixed by the caller's own bookkeeping).

Scope (spec §5.1, owner decision: seed-pinning first)
------------------------------------------------------
This gate only proves determinism for **seedable** computations -- ones where
pinning a single integer seed makes every downstream random draw, thread
schedule, and reduction order reproduce identical bytes. It is **not**
applicable to genuinely non-seedable stochastic pipelines (e.g. hardware
nondeterministic reductions across devices, wall-clock-dependent outputs, or
processes with entropy sources outside the pinned seed) -- those need a
different, weaker gate that instead demonstrates *statistical* equivalence
across runs. That statistical-equivalence tier is explicitly a **named
follow-on**, not implemented here: do not silently downgrade this gate's
byte-identity requirement to a tolerance-based comparison; build a distinct
gate instead.

Cross-tool integration seam (bathos S4 attestation surface)
-------------------------------------------------------------
xtrax has no dependency on bathos (by design -- a compute library must not
depend on the orchestration/experiment-tracking tool built on top of it).
:func:`build_repro_floor_attestation_toml` therefore produces attestation
TOML text that mirrors ``bathos.attestation``'s ``[attestation]`` schema
(kind="repro_floor") *by convention*, not by import: field names, nesting,
and the "every rerun_digest == attested.content_hash" invariant are kept in
lockstep with `bathos/src/bathos/attestation.py`'s docstring by hand. The
actual cross-repo write happens one layer up, outside this module: the
caller (or an orchestrating agent) takes the file written by
:func:`write_repro_floor_attestation` and hands its path to bathos's
``attestation_register`` MCP tool (or ``bth attestation register <path>``
CLI) to anchor it into the bathos catalog. This module refuses
(``ValueError``) to build or write an attestation for a divergent result --
"no attestation" is the deny signal; there is no FAIL-attestation path here,
matching spec §5.1's "on digest divergence, deny (no attestation -> no
promotion)".
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from xtrax.devtools.freshness import Attestation
from xtrax.run.zarr_integrity import fsync_tree, zarr_content_digest


@dataclass(frozen=True, slots=True)
class ReproFloorResult:
    """Outcome of a seed-pinned re-run-N reproducibility check.

    ``content_hash`` is the reference digest every rerun is compared against
    -- by default the first rerun's digest, i.e. this loop treats its own
    first run as establishing the product's canonical content_hash. Pass an
    explicit ``content_hash`` to :func:`run_repro_floor` to instead check
    against an externally pinned reference (e.g. a hash recorded by a prior
    accepted run).
    """

    passed: bool
    seed_pin: int
    rerun_count: int
    rerun_digests: tuple[str, ...]
    content_hash: str
    run_id: str
    output_path: str
    mismatched_digests: tuple[str, ...] = field(default_factory=tuple)
    created_by: str = "xtrax.run.repro_floor"
    created_at: str = ""


def run_repro_floor(
    compute: Callable[[int], Path],
    *,
    seed: int,
    rerun_count: int,
    run_id: str,
    content_hash: str | None = None,
    created_by: str = "xtrax.run.repro_floor",
    now: datetime | None = None,
) -> ReproFloorResult:
    """Re-run ``compute(seed)`` ``rerun_count`` times with the seed pinned.

    Each call to ``compute`` must (re-)produce a Zarr store and return its
    root path; this function durabilizes it (:func:`fsync_tree`) and digests
    it (:func:`zarr_content_digest`) before moving to the next rerun --
    verification is only meaningful once every chunk/metadata file is
    actually on disk.

    Args:
        compute: Callable taking the pinned seed and returning the path to
            the Zarr store it produced. Called exactly ``rerun_count`` times
            with the *same* ``seed`` every time -- this is what "seed-pinned"
            means; ``compute`` is responsible for actually using the seed
            deterministically (out of scope for this function to enforce
            statically).
        seed: The seed pinned across every rerun.
        rerun_count: Number of times to re-run ``compute``. Must be >= 1.
        run_id: Identifier for the attested product (threaded into the
            eventual attestation's ``attested.run_id``).
        content_hash: Optional externally-pinned reference digest. Defaults
            to the first rerun's digest.
        created_by: Attribution recorded on the eventual attestation.
        now: Clock override for ``created_at`` (defaults to current UTC).

    Returns:
        ReproFloorResult. ``passed`` is True iff every rerun's digest equals
        ``content_hash``.

    Raises:
        ValueError: If ``rerun_count`` < 1.
    """
    if rerun_count < 1:
        msg = f"rerun_count must be >= 1, got {rerun_count}"
        raise ValueError(msg)

    digests: list[str] = []
    output_path: Path | None = None
    for _ in range(rerun_count):
        path = compute(seed)
        fsync_tree(path)
        digests.append(zarr_content_digest(path))
        output_path = path

    reference = content_hash if content_hash is not None else digests[0]
    mismatched = tuple(d for d in digests if d != reference)
    passed = not mismatched

    return ReproFloorResult(
        passed=passed,
        seed_pin=seed,
        rerun_count=rerun_count,
        rerun_digests=tuple(digests),
        content_hash=reference,
        run_id=run_id,
        output_path=str(output_path),
        mismatched_digests=mismatched,
        created_by=created_by,
        created_at=(now or datetime.now(UTC)).isoformat(),
    )


_TOML_NAMED_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def _toml_escape_basic_string(value: str) -> str:
    """Escape ``value`` for embedding in a TOML basic string (``"..."``).

    Implements the escaping subset of the TOML basic-string grammar
    (https://toml.io -- "Strings" -> basic strings) that matters for
    arbitrary caller-supplied text: backslash, double-quote, and control
    characters. Without this, a Windows-style path (backslashes) or a run_id
    containing a literal ``"`` corrupts the surrounding TOML syntax --
    exactly debt #628's failure mode (``build_repro_floor_attestation_toml``
    used to f-string-interpolate these fields with zero escaping).

    Single-pass over the *original* characters (not a chain of ``.replace``
    calls) so an escape sequence introduced for one character can never be
    misinterpreted as raw input needing further escaping.
    """
    out: list[str] = []
    for char in value:
        if char in _TOML_NAMED_ESCAPES:
            out.append(_TOML_NAMED_ESCAPES[char])
            continue
        codepoint = ord(char)
        if codepoint < 0x20 or codepoint == 0x7F:
            out.append(f"\\u{codepoint:04x}")
        else:
            out.append(char)
    return "".join(out)


def _toml_quote(value: str) -> str:
    """Wrap ``value`` in a properly-escaped TOML basic string literal."""
    return f'"{_toml_escape_basic_string(value)}"'


def _divergent_result_error(result: ReproFloorResult) -> ValueError:
    """Build the canonical "refusing to attest a divergent result" error.

    Shared by :func:`build_repro_floor_attestation_toml` and
    :func:`write_repro_floor_attestation` so both raise with identical
    wording -- single source of truth for the deny-signal message.
    """
    msg = (
        "refusing to build a repro_floor attestation for a divergent result "
        f"(mismatched_digests={result.mismatched_digests!r}); "
        "no attestation means no promotion"
    )
    return ValueError(msg)


def build_repro_floor_attestation_toml(result: ReproFloorResult) -> str:
    """Serialize a PASSing :class:`ReproFloorResult` into bathos's
    ``[attestation]`` (kind="repro_floor") TOML body -- see module docstring
    for the cross-tool integration seam this feeds.

    Every interpolated string field (``run_id``, ``output_path``,
    ``content_hash``, ``created_by``, ``created_at``, and each entry of
    ``rerun_digests``) is passed through :func:`_toml_quote` so that
    arbitrary caller-supplied text (e.g. a Windows-style path or a run_id
    containing a literal quote -- debt #628) produces valid, round-trippable
    TOML rather than corrupting the surrounding syntax.

    Raises:
        ValueError: If ``result.passed`` is False. Divergence must never
            produce an attestation -- no attestation means no promotion.
    """
    if not result.passed:
        raise _divergent_result_error(result)

    digests_toml = ", ".join(_toml_quote(d) for d in result.rerun_digests)
    return (
        "# Attestation (repro_floor)\n"
        "# Generated via xtrax.run.repro_floor -- register with bathos's\n"
        "# attestation_register MCP tool (or `bth attestation register <path>`)\n"
        "# to anchor into the bathos catalog (see module docstring).\n\n"
        "[attestation]\n"
        'kind = "repro_floor"\n'
        'verdict = "PASS"\n'
        f"attested = {{ run_id = {_toml_quote(result.run_id)}, "
        f"output_path = {_toml_quote(result.output_path)}, "
        f"content_hash = {_toml_quote(result.content_hash)} }}\n"
        f"seed_pin = {result.seed_pin}\n"
        f"rerun_count = {result.rerun_count}\n"
        f"rerun_digests = [{digests_toml}]\n"
        f"created_by = {_toml_quote(result.created_by)}\n"
        f"created_at = {_toml_quote(result.created_at)}\n"
    )


def write_repro_floor_attestation(result: ReproFloorResult, path: Path) -> Path:
    """Write the PASS attestation TOML for ``result`` to ``path``.

    Conventionally ``path`` ends in ``.attestation.bth.toml`` to match
    bathos's own naming, though nothing here enforces that. Creates parent
    directories as needed.

    Path-reuse / tombstoning (debt #644): nothing in this module enforces
    that ``path`` is unique per attempt -- a caller may legitimately reuse
    the same sidecar path across repeated repro-floor checks against the
    same artifact (e.g. a CI job re-running this gate over time). If an
    earlier call wrote a PASS attestation to ``path`` and a *later* call at
    that same ``path`` produces a divergent (denied) result, this function
    removes whatever file currently sits at ``path`` before raising --
    otherwise the stale, fully-valid-looking PASS attestation from the
    earlier good run would silently survive next to (or in place of) a
    denied rerun, which is worse than either tombstoning or refusing reuse
    outright: a caller (or bathos's ``attestation_register``) could later
    read that stale file and believe the *current* state still passes.
    "No attestation" must mean exactly that -- not "no *new* attestation,
    but the old one is still lying around."

    Args:
        result: A ReproFloorResult. Divergent results tombstone ``path``
            (see above) before raising.
        path: Destination file path.

    Returns:
        ``path``, for chaining, when ``result.passed`` is True.

    Raises:
        ValueError: If ``result.passed`` is False. Any pre-existing file at
            ``path`` is removed first (best-effort tombstone), then the
            same canonical error as :func:`build_repro_floor_attestation_toml`
            is raised -- no file is written or left behind in that case.
    """
    if not result.passed:
        if path.exists():
            path.unlink()
        raise _divergent_result_error(result)

    toml_text = build_repro_floor_attestation_toml(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(toml_text)
    return path


def repro_floor_freshness_attestation(
    result: ReproFloorResult,
    *,
    ttl_days: float,
) -> Attestation:
    """Wrap a PASSing repro_floor result as an
    :class:`xtrax.devtools.freshness.Attestation` so callers can gate reuse
    of the PASS verdict against a TTL (:func:`evaluate_freshness`) instead of
    treating it as permanently valid -- a repro_floor PASS recorded against
    since-changed code should eventually stop authorizing promotion on its
    own.

    Raises:
        ValueError: If ``result.passed`` is False.
    """
    if not result.passed:
        msg = "cannot construct a freshness attestation for a divergent repro_floor result"
        raise ValueError(msg)

    return Attestation(
        attested_at=result.created_at,
        ttl_days=ttl_days,
        attested_by=result.created_by,
        note=(
            f"repro_floor PASS: seed_pin={result.seed_pin} "
            f"rerun_count={result.rerun_count} content_hash={result.content_hash}"
        ),
    )


__all__ = [
    "ReproFloorResult",
    "build_repro_floor_attestation_toml",
    "repro_floor_freshness_attestation",
    "run_repro_floor",
    "write_repro_floor_attestation",
]
