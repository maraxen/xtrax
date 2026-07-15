"""xtrax CLI: command-line interface for batch plan visualization and analysis.

This package provides a Tyro-delegated CLI for xtrax, supporting commands like
`xtrax plan` and `xtrax explain` for analyzing and understanding batched
computation structures.
"""

from __future__ import annotations

from xtrax.cli.errors import CLIError, CLIImportError, ShapeParseError
from xtrax.cli.loader import load_fn

__all__ = [
    "CLIError",
    "CLIImportError",
    "ShapeParseError",
    "load_fn",
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
