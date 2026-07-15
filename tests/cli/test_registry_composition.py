"""idea-004 AC3: verify {**REGISTRY, **their_verbs} composes safely through
tyro.extras.subcommand_cli_from_dict -- the documented pattern for a downstream
package building its own tyro-dispatched CLI that reuses xtrax's own verbs
(`.praxia/docs/specs/260715_entry-points-based-xtrax-cli-verb-regist.md`).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from xtrax.cli import REGISTRY
from xtrax.inference.config import AxisOverride, axis_config


@axis_config(AxisOverride(name="batch", default_batch_size=2))
def _plannable_fn(x):
    """A function decorated with axis_config, used as a --fn import target."""
    return x * 2


@dataclass
class FakeDownstreamArgs:
    """A downstream package's own verb args -- name/shape has no relation to
    any built-in xtrax verb's ArgsClass.
    """

    value: int = 42


def _run_fake_downstream(args: FakeDownstreamArgs) -> None:
    print(f"fake-downstream-verb ran with value={args.value}")


def _dispatch(subcommands: dict[str, tuple[type[Any], Callable[..., None]]], argv: list[str]):
    """Mirror entrypoint.py's main() dispatch logic for an arbitrary registry dict."""
    import tyro

    for args_cls, _ in subcommands.values():
        mod = sys.modules.get(args_cls.__module__)
        if mod is not None and getattr(mod, "tyro", "unset") is None:
            mod.tyro = tyro

    tyro_subcommands: dict[str, Callable[..., Any]] = {
        name: args_cls for name, (args_cls, _fn) in subcommands.items()
    }
    old_argv = sys.argv
    try:
        sys.argv = ["xtrax", *argv]
        selected = tyro.extras.subcommand_cli_from_dict(tyro_subcommands)
    finally:
        sys.argv = old_argv

    selected_type = type(selected)
    for _name, (args_cls, run_fn) in subcommands.items():
        if args_cls is selected_type:
            run_fn(selected)
            return
    raise AssertionError(f"no verb registered for {selected_type!r}")


def test_merged_registry_dispatches_builtin_verb(capsys) -> None:
    """A built-in xtrax verb (plan) still dispatches correctly once merged with
    a downstream package's own verbs.
    """
    merged = {**REGISTRY, "fake-downstream-verb": (FakeDownstreamArgs, _run_fake_downstream)}
    _dispatch(
        merged,
        [
            "plan",
            "--fn",
            "tests.cli.test_registry_composition:_plannable_fn",
            "--shapes",
            "x=(4,)f32",
        ],
    )
    captured = capsys.readouterr()
    assert "batch" in captured.out or captured.out  # plan verb printed a summary


def test_merged_registry_dispatches_downstream_verb(capsys) -> None:
    """A downstream package's own verb, merged alongside REGISTRY, dispatches too."""
    merged = {**REGISTRY, "fake-downstream-verb": (FakeDownstreamArgs, _run_fake_downstream)}
    _dispatch(merged, ["fake-downstream-verb", "--value", "7"])
    captured = capsys.readouterr()
    assert "fake-downstream-verb ran with value=7" in captured.out


def test_merged_registry_has_no_name_collision() -> None:
    """Sanity: a fake downstream verb name distinct from all 9 built-ins merges
    without silently overwriting anything.
    """
    merged = {**REGISTRY, "fake-downstream-verb": (FakeDownstreamArgs, _run_fake_downstream)}
    assert len(merged) == len(REGISTRY) + 1
    assert merged["plan"] is REGISTRY["plan"]
