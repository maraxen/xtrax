"""xtrax CLI: command-line interface for batch plan visualization and analysis.

This package provides a Tyro-delegated CLI for xtrax, supporting commands like
`xtrax plan` and `xtrax explain` for analyzing and understanding batched
computation structures.

**Deferred (E2.2+):** The following verbs are planned for future releases:

- `run`: Execute a BatchPlan against a DataModule. Gating: DataModule factory
  interface and integration with the Trainer/Engine layer.
- `sweep`: Launch a parameter sweep campaign over a BatchPlan. Gating: grid-based
  sweep vs. bathos-campaign integration design.
- `resume`: Resume a previously interrupted run from a checkpoint. Gating:
  RunManifest schema and checkpoint/restore protocol.
- `export`: Export a plan or traced computation to external formats. Gating:
  decision on MLIR vs. serialized JAX export format and flatbuffers schema
  extension.
"""

from __future__ import annotations

from xtrax.cli.errors import CLIError, CLIImportError, ShapeParseError

__all__ = [
    "CLIError",
    "CLIImportError",
    "ShapeParseError",
    "main",
]


def main() -> None:
    """Console entrypoint — imports the tyro-backed dispatcher lazily.

    This function is the entry point for the xtrax CLI. It imports the
    actual CLI dispatcher from entrypoint.py on demand, keeping the
    xtrax.cli module tyro-free at import time. The real CLI implementation
    (entrypoint.py) is built in task T6.
    """
    from xtrax.cli.entrypoint import main as _main  # type: ignore[import-not-found]

    _main()
