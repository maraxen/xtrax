"""Leaf constants shared by ``zarr_sink`` and ``zarr_integrity``.

Lives in its own import-free module so ``zarr_sink`` can depend on
``zarr_commit``/``digest`` (which depend on ``zarr_integrity``) without an import cycle.
"""

#: Core provenance field names written by the sink itself. Caller-staged
#: attrs may not use these names (collision raises at ``ZarrStagingSink.stage``).
#: Root attrs with these names are excluded from ``zarr_content_digest`` by
#: default. Non-root groups exclude only ``run_id`` and ``git_sha``.
CORE_PROVENANCE_FIELDS = frozenset(
    {
        "git_sha",
        "git_branch",
        "git_dirty",
        "run_id",
        "created_at",
        "producer",
        "xtrax_version",
    }
)

#: Root-only stamp excluded from the default content digest. Not reserved at
#: stage(): callers record their own ``seed`` attr on non-root groups.
ROOT_STAMPED_FIELDS = frozenset({"seed"})

#: Name recorded on a store root as the ``producer`` provenance attr.
PRODUCER_NAME = "xtrax"

#: Prefix reserved for xtrax-written attrs. Attrs whose key starts with this
#: prefix are excluded from ``zarr_content_digest`` by default (unless
#: ``include_provenance=True``), ensuring that internal sink bookkeeping
#: (stamped via ``ZarrStagingSink.stamp_reserved``) does not affect
#: content-based reproducibility.
RESERVED_ATTR_PREFIX = "xtrax."
