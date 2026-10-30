"""File-count (inode) guard for shared scratch (spec 261030_ledger-pack-fileguard.md section 3.4).

On Engaging scratch the binding limit is the per-user file count, not bytes. This package reads
the real quota (``lfs quota`` on Lustre, never a filesystem-wide ``df -i`` there; ``df -i``
elsewhere), refuses a write that would push the projected count above a cap fraction of the
limit, and re-reads the real figure every ``reread_every`` counted files so writers outside this
process stay visible. When the target directory is setgid on Lustre, the owning group's quota is
enforced too.

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
    describe_location,
    fs_type,
    log_emitter,
    parse_df_inodes,
    parse_lfs_quota,
    read_quota,
    read_quotas,
)

__all__ = [
    "LIMIT_UNKNOWN_EVENT",
    "LUSTRE",
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
    "describe_location",
    "fs_type",
    "log_emitter",
    "node_source",
    "parse_df_inodes",
    "parse_lfs_quota",
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
