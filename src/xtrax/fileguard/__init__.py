"""File-count (inode) guard for shared scratch (spec 261030_ledger-pack-fileguard.md section 3.4).

On Engaging the binding limit is the per-user file count, not bytes. This package reads the
real quota and refuses a write that would push the projected count above a cap fraction of the
limit, re-reading the real figure every ``reread_every`` counted files so writers outside this
process stay visible. Lustre: ``lfs quota`` (refused, never replaced by a filesystem-wide
``df -i``). Elsewhere: the per-user ``quota -w`` row for the path's device plus the
filesystem ``df -i`` reading; a path with no per-user row is flagged with a
``quota_limit_unknown`` event, because a shared NFS server's ``df -i`` is never the user's cap.
When the target directory is setgid, the owning group's quota is enforced too. On MIT ORCD
(Engaging) the root-written ``~/orcd/.quota`` report supplies the per-user HOME, SCRATCH and POOL
file counts, which ``quota`` does not report for scratch or pool.

The logic lives in :mod:`xtrax.fileguard._node`, which is stdlib-only and Python 3.9 compatible
so a cluster script can embed it (:func:`node_source`) where xtrax is not installed. This module
adds the default emitter: ``cisternal.emit_event`` when the ``plugin`` extra is installed, else a
JSON line on the ``xtrax.fileguard`` logger.

Typical use::

    guard = FileCapGuard(out_dir, emit=default_emitter())
    for unit in units:
        guard.check(files_per_unit)  # raises QuotaRefused before the write
        write(unit)
"""

from pathlib import Path

from xtrax.fileguard._node import (
    LIMIT_UNKNOWN_EVENT,
    LUSTRE,
    ORCD_MAX_AGE_S,
    ORCD_REPORT_NAME,
    READING_EVENT,
    REFUSAL_EVENT,
    Emitter,
    FileCapGuard,
    LocationInfo,
    QuotaReading,
    QuotaRefused,
    QuotaUnreadable,
    Runner,
    check_concurrency_slack,
    default_orcd_dir,
    describe_location,
    fs_type,
    log_emitter,
    orcd_space_roots,
    parse_df_inodes,
    parse_lfs_quota,
    parse_orcd_report,
    parse_quota_rows,
    read_quota,
    read_quotas,
)

__all__ = [
    "LIMIT_UNKNOWN_EVENT",
    "LUSTRE",
    "ORCD_MAX_AGE_S",
    "ORCD_REPORT_NAME",
    "READING_EVENT",
    "REFUSAL_EVENT",
    "Emitter",
    "FileCapGuard",
    "LocationInfo",
    "QuotaReading",
    "QuotaRefused",
    "QuotaUnreadable",
    "Runner",
    "check_concurrency_slack",
    "default_emitter",
    "default_orcd_dir",
    "describe_location",
    "fs_type",
    "log_emitter",
    "node_source",
    "orcd_space_roots",
    "parse_df_inodes",
    "parse_lfs_quota",
    "parse_orcd_report",
    "parse_quota_rows",
    "read_quota",
    "read_quotas",
]


def default_emitter() -> Emitter:
    """``cisternal.emit_event`` when cisternal is importable, else :func:`log_emitter`."""
    try:
        import cisternal  # ty: ignore[unresolved-import]
    except ImportError:
        return log_emitter

    def _emit(name: str, fields: dict[str, object]) -> None:
        cisternal.emit_event(name, **fields)

    return _emit


def node_source() -> str:
    """Source text of the stdlib-only core, for embedding in a rendered cluster script."""
    return (Path(__file__).with_name("_node.py")).read_text(encoding="utf-8")
