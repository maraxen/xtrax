"""Run-id generation (stdlib-only).

``new_run_id`` exists so drivers never block on naming: callers who need
meaningful or reproducible ids pass explicit ones instead (see
``derive_sink_spec`` precedence). Named ``new_run_id``, not
``generate_run_id``, because that name is taken by ``cli/run.py``'s
config-hash generator with incompatible semantics.
"""

from uuid import uuid4


def new_run_id() -> str:
    """Generate a fresh run id: ``run-`` + 12 lowercase hex chars.

    The charset is ``[0-9a-f]`` only -- ``uuid4().hex`` contains no dashes --
    so ids are path-safe, TOML-safe without escaping, and shell-safe.
    """
    return f"run-{uuid4().hex[:12]}"
