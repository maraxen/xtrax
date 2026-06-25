"""xtrax CLI entrypoint (T6).

This module provides the main() function that dispatches to the correct
verb implementation via tyro subcommands. tyro is imported lazily inside
main() so that ``import xtrax.cli.entrypoint`` does NOT trigger a tyro
import — keeping the xtrax.cli namespace tyro-free at import time (AC2).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

from xtrax.cli.errors import CLIError
from xtrax.cli.registry import REGISTRY


def main() -> None:
    """CLI entrypoint: parse subcommand and dispatch to the appropriate verb.

    Builds a tyro subcommand dict from REGISTRY and dispatches the parsed
    Args instance to the matching run_fn. CLIError is caught at the top level
    and printed cleanly to stderr with SystemExit(1) — no tracebacks shown
    to the user.

    tyro is imported inside this function so that importing this module
    does not pull in tyro (AC2: import isolation).
    """
    # Lazy tyro import — AC2: keeps this module tyro-free at import time.
    import tyro

    for args_cls, _ in REGISTRY.values():
        mod = sys.modules[args_cls.__module__]
        if hasattr(mod, "tyro") and getattr(mod, "tyro") is None:
            setattr(mod, "tyro", tyro)

    # Build subcommand dict: {verb_name: ArgsClass}.
    # Typed as Callable[..., Any] to satisfy tyro's overload signature;
    # dataclass constructors are callables and tyro uses them as type annotations.
    subcommands: dict[str, Callable[..., Any]] = {
        name: args_cls for name, (args_cls, _fn) in REGISTRY.items()
    }

    try:
        selected = tyro.extras.subcommand_cli_from_dict(subcommands)

        # Dispatch: find the run_fn whose args_cls matches type(selected).
        selected_type = type(selected)
        for _name, (args_cls, run_fn) in REGISTRY.items():
            if args_cls is selected_type:
                run_fn(selected)
                return

        # Unreachable if REGISTRY is consistent — tyro only selects registered types.
        assert False, (  # noqa: B011
            f"No verb registered for args type {selected_type.__name__!r} — this is a bug in xtrax"
        )
    except CLIError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1) from None


__all__ = ["main"]
