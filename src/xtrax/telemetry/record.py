"""Schema for one run-ledger row: provenance + captured IR, schema-versioned.

Design lineage. This module is the run-level sibling of
``xtrax.profiling.record.ProbeRecord`` and deliberately keeps two of its
disciplines verbatim:

  1. *A provenance field a caller can forget is a provenance field that will be
     forgotten.* Every provenance value is captured by a ``default_factory``,
     never taken as a caller argument. A caller may override for tests, but
     cannot omit.
  2. Fail-closed deserialization. ``from_dict`` rejects unknown fields and
     refuses to reconstruct a row whose ``schema_version`` it does not
     understand, rather than silently applying this machine's own environment in
     place of what was actually recorded.

The env-capture factories themselves are *imported* from
``xtrax.profiling.record`` rather than reimplemented. That package's leaf
contract (no sibling xtrax imports, AST-enforced) is one-directional: it may not
import us, but we may import it. Copying those functions instead would fork
hard-won details -- ``normalize_device_kind``'s GH200-vs-H200 disambiguation
being the sharpest example.

Brittle where absence is unrecoverable, forgiving where it is not. Capture
degrades (a failed git shellout records ``provenance_source="degraded"``, never
raises: telemetry must not take down the run it observes), while *reading*
fails closed (unknown field or version drift raises). The one rule bridging
them is that no degradation may be silent -- ``__post_init__`` makes a
non-``complete`` status or a skipped IR artifact *unrepresentable* without a
reason string.

Why this exists at all: of 345 catalogued bathos runs, ``git_hash`` was
populated 345/345, yet only 40.6% still resolved to a commit and 92.2% ran on a
dirty tree (cisternal PR #32). Recording a SHA is not the same as being able to
reconstruct what ran, which is why ``provenance_source`` and the pinned-ref
fields below are part of the schema rather than an optional nicety.
"""

import dataclasses
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from xtrax.profiling.record import (
    capture_device_kind,
    capture_git_sha,
    capture_jax_version,
    capture_jaxlib_version,
    capture_timestamp,
    capture_x64_enabled,
    capture_xla_flags,
)

SCHEMA_VERSION = 1

# provenance_source values, most to least trustworthy.
PROVENANCE_CISTERNAL = "cisternal"
PROVENANCE_BUILTIN = "builtin"
PROVENANCE_DEGRADED = "degraded"

# telemetry_status values. Only "complete" is citable; see is_citable below.
STATUS_COMPLETE = "complete"
STATUS_DEGRADED = "degraded"
STATUS_OPTED_OUT = "opted_out"
STATUS_FAILED = "failed"
_VALID_STATUSES = frozenset({STATUS_COMPLETE, STATUS_DEGRADED, STATUS_OPTED_OUT, STATUS_FAILED})

# IR capture modes, mirroring cisternal's SNAPSHOT_FULL/METADATA_ONLY/NONE ladder.
IR_FULL = "full"
IR_HASH_ONLY = "hash_only"
IR_SKIPPED = "skipped"
_VALID_IR_MODES = frozenset({IR_FULL, IR_HASH_ONLY, IR_SKIPPED})

# Run kinds. "train" and "eval" are the two execution paths; "export" is an
# ahead-of-time lowering that produces an artifact without executing it.
KIND_TRAIN = "train"
KIND_EVAL = "eval"
KIND_EXPORT = "export"
_VALID_KINDS = frozenset({KIND_TRAIN, KIND_EVAL, KIND_EXPORT})

_GIT_TIMEOUT_SECONDS = 60


class SchemaVersionMismatchError(ValueError):
    """Raised when a ledger row's schema_version is not the running version.

    Loud by design, matching ``xtrax.findings.assert_schema_version_compatible``:
    a verdict computed under one schema is not guaranteed valid under another,
    so best-effort coercion would launder exactly the drift this ledger exists to
    make visible. See ``xtrax.telemetry.migrate`` for the forward path.
    """


class LedgerRecordError(ValueError):
    """Raised when a ledger row violates its own contract at construction."""


def _git(args: "list[str]", cwd: Path) -> "str | None":
    """Run a git command, returning stripped stdout or None on any failure.

    Returns None rather than raising: provenance capture must never be the thing
    that takes down the run it is observing. Callers distinguish "no value" from
    "clean empty value" by checking for None explicitly -- a distinction that
    matters most for ``git status --porcelain``, where empty means clean but
    None means we never found out.
    """
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.strip()


def _capture_hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "unknown"


def _capture_python_version() -> str:
    return sys.version.split()[0]


def _capture_platform() -> str:
    return platform.platform()


def _safe(factory: Any, fallback: Any) -> Any:  # noqa: ANN401 - generic capture shim
    """Call a capture factory, substituting a fallback if the environment lacks it.

    jax/jaxlib are hard xtrax dependencies, so these normally succeed. The guard
    exists so the ledger stays constructible in a stripped context (a CI gate, a
    compaction pass) without turning a missing optional import into a failed run.

    Never a bare ``pass``: the fallback is what gets *recorded*, so the gap is
    visible in the row rather than silently absent. ruff S110 forbids the
    alternative repo-wide.
    """
    try:
        return factory()
    except Exception:  # noqa: BLE001 - any import/runtime failure degrades identically
        return fallback


@dataclasses.dataclass(frozen=True, slots=True)
class IRRef:
    """A reference to one captured IR artifact in the content-addressed store.

    ``sha256`` is over the uncompressed IR text bytes, so identical IR across
    steps, across runs, and across machines collapses to a single blob. Measured
    on a 96-layer model: jaxpr + StableHLO + optimized HLO totalled 159 KB raw /
    14 KB gzipped, and IR size tracks op count rather than batch size -- which is
    why storing full text is affordable and an embedded DB is not needed.

    ``mode`` records whether the text was actually stored (``full``),
    fingerprinted only because it exceeded the size cap (``hash_only``), or not
    captured at all (``skipped``). A skipped artifact degrades the row's
    telemetry_status and can never be silently absent: ``__post_init__`` refuses
    to build one without a reason.
    """

    kind: str
    sha256: str
    bytes: int
    mode: str = IR_FULL
    reason: "str | None" = None

    def __post_init__(self) -> None:
        if self.mode not in _VALID_IR_MODES:
            raise LedgerRecordError(
                f"IRRef.mode must be one of {sorted(_VALID_IR_MODES)}, got {self.mode!r}"
            )
        if not self.kind or not self.kind.strip():
            raise LedgerRecordError("IRRef.kind must be a non-empty string")
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int) or self.bytes < 0:
            raise LedgerRecordError(f"IRRef.bytes must be a non-negative int, got {self.bytes!r}")
        if self.mode == IR_SKIPPED:
            if not (self.reason or "").strip():
                raise LedgerRecordError(
                    "IRRef.mode='skipped' requires a reason -- an artifact that was "
                    "not captured must say why, or the gap is indistinguishable from "
                    "an artifact that never existed"
                )
        elif len(self.sha256) != 64:
            raise LedgerRecordError(
                f"IRRef.sha256 must be a 64-char hex digest for mode={self.mode!r}, "
                f"got {self.sha256!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class RunProvenance:
    """Everything needed to reconstruct the environment and code a run executed.

    Constructed via :meth:`capture`, which is the ``default_factory`` on
    :class:`RunLedgerRecord`. Field defaults here are the *degraded* readings, so
    a partially-failed capture yields an explicit "we did not learn this" rather
    than a plausible-looking wrong value.
    """

    git_sha: str = "unknown"
    git_branch: "str | None" = None
    git_dirty: "bool | None" = None
    dirty_content_id: "str | None" = None
    remote_url: "str | None" = None
    submodule_state: "str | None" = None
    pinned_sha: "str | None" = None
    run_ref: "str | None" = None
    wip_ref: "str | None" = None
    snapshot_mode: "str | None" = None
    unpinned_reason: "str | None" = None
    provenance_source: str = PROVENANCE_DEGRADED
    jax_version: "str | None" = None
    jaxlib_version: "str | None" = None
    x64_enabled: "bool | None" = None
    xla_flags: str = ""
    device_kind: "str | None" = None
    hostname: str = "unknown"
    python_version: str = ""
    platform: str = ""

    @classmethod
    def capture(
        cls,
        *,
        run_id: "str | None" = None,
        cwd: "Path | None" = None,
        pin: bool = False,
    ) -> "RunProvenance":
        """Capture full provenance for the current process and working tree.

        ``pin=True`` additionally asks cisternal to create a durable per-run ref
        (and, on a dirty tree, a worktree snapshot commit) so the exact code
        state survives garbage collection and branch deletion. Pinning *writes*
        to the repository, so it is opt-in and performed once per run by
        ``RunLedger.open`` -- never on a bare record construction, and never in a
        per-step hook.

        Never raises. Every failure path degrades to a recorded value plus, where
        applicable, ``unpinned_reason``.
        """
        root = Path.cwd() if cwd is None else Path(cwd)
        env = {
            "jax_version": _safe(capture_jax_version, None),
            "jaxlib_version": _safe(capture_jaxlib_version, None),
            "x64_enabled": _safe(capture_x64_enabled, None),
            "xla_flags": _safe(capture_xla_flags, ""),
            "device_kind": _safe(capture_device_kind, None),
            "hostname": _capture_hostname(),
            "python_version": _capture_python_version(),
            "platform": _capture_platform(),
        }
        git = _capture_git_block(run_id=run_id, root=root, pin=pin)
        return cls(**git, **env)

    def to_dict(self) -> "dict[str, Any]":
        return dataclasses.asdict(self)


def _capture_repo_identity(root: Path) -> "dict[str, Any]":
    """Remote URL + submodule fingerprint.

    Both are gaps in cisternal's ``pin_run`` (which records neither), and both
    cost one cheap shellout. ``submodule_state`` is a digest rather than the raw
    listing so a repo with many submodules cannot bloat every ledger row; xtrax
    itself has a .gitmodules, so this is a live concern rather than a theoretical
    one. A repo with no submodules records None, not an empty-string digest --
    "no submodules" and "we did not check" must stay distinguishable.
    """
    remote = _git(["config", "--get", "remote.origin.url"], root)
    submodules = _git(["submodule", "status", "--recursive"], root)
    digest = None
    if submodules:
        digest = "sha256:" + hashlib.sha256(submodules.encode("utf-8")).hexdigest()
    return {"remote_url": remote or None, "submodule_state": digest}


def _capture_git_block(*, run_id: "str | None", root: Path, pin: bool) -> "dict[str, Any]":
    """Git provenance, preferring cisternal's durable capture when installed.

    Three tiers, each recorded honestly in ``provenance_source``:

      cisternal -- ``capture_git_state`` plus, when ``pin`` is set, ``pin_run``'s
        durable ref and dirty-worktree snapshot. This is the only tier where the
        recorded SHA is guaranteed to still resolve later.
      builtin   -- git shellouts via this module, falling back to
        ``xtrax.profiling.record.capture_git_sha`` (which honours XTRAX_GIT_SHA
        and a repo-root ``.git_sha`` file, both of which exist for cluster
        scratch dirs that have no ``.git``).
      degraded  -- HEAD could not be resolved at all.
    """
    identity = _capture_repo_identity(root)
    cisternal_block = _try_cisternal(run_id=run_id, root=root, pin=pin)
    if cisternal_block is not None:
        cisternal_block.update(identity)
        return cisternal_block

    sha = _git(["rev-parse", "HEAD"], root)
    if not sha:
        stamped = _safe(capture_git_sha, "unknown")
        source = PROVENANCE_BUILTIN if stamped and stamped != "unknown" else PROVENANCE_DEGRADED
        return {
            "git_sha": stamped or "unknown",
            "git_branch": None,
            "git_dirty": None,
            "dirty_content_id": None,
            "provenance_source": source,
            "unpinned_reason": "HEAD could not be resolved by git rev-parse",
            **identity,
        }

    porcelain = _git(["status", "--porcelain"], root)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    return {
        "git_sha": sha,
        "git_branch": branch,
        # None (not False) when the dirty check itself failed: a failed
        # `git status` is not evidence of a clean tree, and recording False
        # there would let two unverified runs trivially "agree" as clean.
        "git_dirty": None if porcelain is None else bool(porcelain),
        "dirty_content_id": None,
        "provenance_source": PROVENANCE_BUILTIN,
        "unpinned_reason": ("cisternal.provenance is not installed" if pin else None),
        **identity,
    }


def _try_cisternal(*, run_id: "str | None", root: Path, pin: bool) -> "dict[str, Any] | None":
    """Capture via cisternal.provenance, or None if it is unavailable.

    Imported lazily inside the function, the way ``run/zarr_sink.py`` treats
    zarr: cisternal is an optional extra (``xtrax[provenance]``), so its absence
    must cost a clean fallback, not an ImportError at module load.
    """
    try:
        # cisternal is an optional extra (xtrax[provenance]); an absent module
        # here is the designed fallback path, not a configuration error.
        from cisternal.provenance import (  # ty: ignore[unresolved-import]
            capture_git_state,
            pin_run,
        )
    except ImportError:
        return None

    try:
        state = capture_git_state(root)
    except Exception:  # noqa: BLE001 - cisternal degrades internally; belt and braces
        return None

    block: dict[str, Any] = {
        "git_sha": getattr(state, "hash", "unknown") or "unknown",
        "git_branch": getattr(state, "branch", None),
        "git_dirty": getattr(state, "dirty", None),
        "dirty_content_id": getattr(state, "dirty_content_id", None),
        "provenance_source": PROVENANCE_CISTERNAL,
        "pinned_sha": None,
        "run_ref": None,
        "wip_ref": None,
        "snapshot_mode": None,
        "unpinned_reason": None,
    }
    if not pin or not run_id:
        block["unpinned_reason"] = "pinning not requested for this record"
        return block

    try:
        result = pin_run(
            run_id=run_id,
            git_hash=block["git_sha"],
            git_branch=block["git_branch"] or "",
            dirty=bool(block["git_dirty"]),
            cwd=root,
        )
    except Exception as exc:  # noqa: BLE001 - a failed pin degrades, never fails the run
        block["unpinned_reason"] = f"pin_run raised {type(exc).__name__}: {exc}"
        return block

    block["run_ref"] = getattr(result, "run_ref", None)
    block["wip_ref"] = getattr(result, "wip_ref", None)
    block["snapshot_mode"] = getattr(result, "snapshot_mode", None)
    block["unpinned_reason"] = getattr(result, "unpinned_reason", None)
    # pin_run points the run ref at a worktree snapshot on a dirty tree, so the
    # pinned commit -- not necessarily HEAD -- is the reconstructable one.
    block["pinned_sha"] = getattr(result, "wip_commit", None) or block["git_sha"]
    return block


@dataclasses.dataclass(frozen=True, slots=True)
class RunLedgerRecord:
    """One append-only ledger row describing a single run.

    ``derived_from`` is a *single* parent run_id, matching
    ``controller/lineage_interim.py``'s existing single-parent contract rather
    than inventing a second, divergent lineage model. Rows therefore form a run
    DAG that is derivable on read, with no graph store required.
    """

    run_id: str
    kind: str
    derived_from: "str | None" = None
    telemetry_status: str = STATUS_COMPLETE
    status_reason: "str | None" = None
    ir: "tuple[IRRef, ...]" = ()
    schema_version: int = SCHEMA_VERSION
    ts: str = dataclasses.field(default_factory=capture_timestamp)
    provenance: RunProvenance = dataclasses.field(default_factory=RunProvenance.capture)

    def __post_init__(self) -> None:
        if not self.run_id or not self.run_id.strip():
            raise LedgerRecordError("run_id must be a non-empty string")
        if self.kind not in _VALID_KINDS:
            raise LedgerRecordError(
                f"kind must be one of {sorted(_VALID_KINDS)}, got {self.kind!r}"
            )
        if self.telemetry_status not in _VALID_STATUSES:
            raise LedgerRecordError(
                f"telemetry_status must be one of {sorted(_VALID_STATUSES)}, "
                f"got {self.telemetry_status!r}"
            )
        if self.derived_from is not None and not str(self.derived_from).strip():
            raise LedgerRecordError(
                "derived_from must be a non-empty run_id or None -- a blank string "
                "is not a parent, and recording one would fabricate lineage"
            )
        if self.derived_from == self.run_id:
            raise LedgerRecordError(
                f"derived_from must not equal run_id ({self.run_id!r}) -- a run "
                "cannot be its own parent"
            )
        if self.telemetry_status != STATUS_COMPLETE and not (self.status_reason or "").strip():
            raise LedgerRecordError(
                f"telemetry_status={self.telemetry_status!r} requires a status_reason; "
                "a non-complete row that does not say why is not auditable"
            )
        object.__setattr__(self, "ir", tuple(self.ir))

    @property
    def is_citable(self) -> bool:
        """Whether a result from this run may be cited.

        Only a ``complete`` row qualifies. This is the predicate the claim-time
        gate consults, and it is the enforcement tier that survives a caller who
        set XTRAX_TELEMETRY_OPTOUT: opting out of *capture* does not opt out of
        the *record*, it only makes the run non-citable.
        """
        return self.telemetry_status == STATUS_COMPLETE

    def to_dict(self) -> "dict[str, Any]":
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "kind": self.kind,
            "derived_from": self.derived_from,
            "ts": self.ts,
            "telemetry_status": self.telemetry_status,
            "status_reason": self.status_reason,
            "provenance": self.provenance.to_dict(),
            "ir": [dataclasses.asdict(ref) for ref in self.ir],
        }

    def to_json_line(self) -> str:
        """One-line JSON, sorted keys, newline-terminated -- the JSONL row form."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"

    @classmethod
    def from_dict(cls, raw: Any) -> "RunLedgerRecord":  # noqa: ANN401
        """Reconstruct a row, failing closed on version drift and unknown fields.

        Rule order matters and mirrors ``ProbeRecord.from_json``: the version
        check runs first, so a version-skewed row is diagnosed as version skew
        rather than misreported as field-level corruption. Callers wanting to
        read older rows go through ``xtrax.telemetry.migrate.upgrade_row``
        first; this constructor deliberately has no implicit upgrade path.

        ``raw`` is annotated ``Any``, not ``dict``, on purpose: it is a decoded
        JSON value of unproven shape (``json.loads`` happily returns a list, a
        string, or a number for a corrupted row), and the isinstance guard below
        is the thing that proves otherwise. Typing it as ``dict`` would assert
        the very property this method exists to verify, and would render that
        guard unreachable to a type checker while leaving it load-bearing at
        runtime.
        """
        if not isinstance(raw, dict):
            raise LedgerRecordError(
                f"ledger row must decode to an object, got {type(raw).__name__}"
            )
        version = raw.get("schema_version")
        if version != SCHEMA_VERSION:
            raise SchemaVersionMismatchError(
                f"ledger row schema_version={version!r} does not match the running "
                f"schema_version={SCHEMA_VERSION!r} -- refusing to reinterpret a row "
                "written under a different contract; see xtrax.telemetry.migrate"
            )
        known = {
            "schema_version",
            "run_id",
            "kind",
            "derived_from",
            "ts",
            "telemetry_status",
            "status_reason",
            "provenance",
            "ir",
        }
        unknown = set(raw) - known
        if unknown:
            raise LedgerRecordError(
                f"ledger row has unknown field(s) {sorted(unknown)} -- written by a "
                "newer contract than this code understands"
            )
        prov_raw = dict(raw.get("provenance") or {})
        prov_fields = {f.name for f in dataclasses.fields(RunProvenance)}
        unknown_prov = set(prov_raw) - prov_fields
        if unknown_prov:
            raise LedgerRecordError(
                f"ledger row provenance has unknown field(s) {sorted(unknown_prov)}"
            )
        try:
            ir = tuple(IRRef(**ref) for ref in raw.get("ir", ()))
        except TypeError as exc:
            raise LedgerRecordError(f"ledger row has a malformed ir entry: {exc}") from exc
        return cls(
            run_id=raw["run_id"],
            kind=raw["kind"],
            derived_from=raw.get("derived_from"),
            telemetry_status=raw.get("telemetry_status", STATUS_COMPLETE),
            status_reason=raw.get("status_reason"),
            ir=ir,
            schema_version=version,
            ts=raw["ts"],
            provenance=RunProvenance(**prov_raw),
        )

    @classmethod
    def from_json_line(cls, line: str) -> "RunLedgerRecord":
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LedgerRecordError(f"ledger row is not valid JSON: {exc}") from exc
        return cls.from_dict(raw)


def telemetry_opted_out() -> bool:
    """Whether XTRAX_TELEMETRY_OPTOUT is set to a truthy value.

    The opt-out suppresses *capture*, never the row: an opted-out run still
    writes a ledger entry, marked non-citable. Only explicit truthy spellings
    count, so XTRAX_TELEMETRY_OPTOUT=0 is not read as an opt-out.
    """
    return (os.environ.get("XTRAX_TELEMETRY_OPTOUT") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
