#!/usr/bin/env python3
"""T1-13 `graph author-bench` harness (#3068) -- the empirical gate deciding fork-12.

Compares T1-12's two authoring front-ends -- generate-then-validate (the shipped default,
`xtrax.composition.author.TemplateGenerator`) and Outlines constrained decoding (the smoke
demo, `scripts/smoke_outlines_constrained_decode.py`) -- over the ~10 fixed tasks in
`scripts/author_bench_fixtures.py`, reporting for each mode an ABSOLUTE structural-validity
yield and an ABSOLUTE ground-truth faithfulness rate (not just a mode-vs-mode delta), plus a
human-labeled faithfulness spot-check as a hard sub-gate.

Per PM2 (`.praxia/docs/research/260702_roadmap-research-synthesis.md:67`): a prior timeline
shipped this AC but dropped the faithfulness spot-check under sprint pressure, leaving only a
mode-vs-mode delta that reported "no degradation" while both modes silently authored
structurally-perfect graphs that computed the wrong thing. This script's fast/loud gate exists
specifically to make that failure mode impossible to ship silently: it exits 1 if the
human-labeled spot-check's pass rate falls below the AC's 80% bar.

Deliberately lives outside `src/xtrax/` and outside `REGISTRY` -- same placement rationale as
T1-12's smoke script, so `outlines`/`transformers`/`torch` never become reachable, even
transitively, from `src/xtrax/**`, and T1-12's existing loop-code gate
(`tests/audit/test_no_constrained_decode_in_loop.py`) needs no changes.

Usage:
    uv run python scripts/graph_author_bench.py
    uv run --extra authoring-smoke python scripts/graph_author_bench.py \\
        --include-constrained-decoding
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.author_bench_fixtures import (  # noqa: E402
    HUMAN_LABELED_SPOT_CHECK,
    TASKS,
    AuthorBenchTask,
)
from xtrax.composition.author import TemplateGenerator  # noqa: E402
from xtrax.composition.errors import (  # noqa: E402
    GraphConstructionError,
    GraphSerializationError,
    SchemaVersionError,
)
from xtrax.composition.graph import HostPrepGraph  # noqa: E402
from xtrax.composition.serialize import deserialize_graph  # noqa: E402
from xtrax.composition.validate import validate_graph  # noqa: E402

DEFAULT_OUT_PATH = ROOT / ".praxia" / "author_bench_report.json"
DEFAULT_CONSTRAINED_MODEL = "distilgpt2"

SPOT_CHECK_PASS_BAR = 0.80
FORK12_VALIDITY_MARGIN_PP = 0.10
FORK12_FAITHFULNESS_MARGIN_PP = 0.05

# Deserializing an arbitrary (possibly LLM-decoded, possibly degenerate) document is expected
# to fail in ordinary, documented ways -- not a harness bug.
_DESERIALIZE_ERRORS = (
    ValueError,
    SchemaVersionError,
    GraphSerializationError,
    GraphConstructionError,
)


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    structurally_valid: bool
    faithful: bool
    error: str | None = None


@dataclass(frozen=True)
class ModeResult:
    mode: str
    task_outcomes: tuple[TaskOutcome, ...]

    @property
    def structural_validity_yield(self) -> float:
        return sum(o.structurally_valid for o in self.task_outcomes) / len(self.task_outcomes)

    @property
    def faithfulness_rate(self) -> float:
        return sum(o.faithful for o in self.task_outcomes) / len(self.task_outcomes)


@dataclass(frozen=True)
class SpotCheckResult:
    n: int
    n_passed: int

    @property
    def pass_rate(self) -> float:
        return self.n_passed / self.n


def _import_path(fn: object) -> str:
    return f"{getattr(fn, '__module__', '?')}:{getattr(fn, '__qualname__', '?')}"


def _node_sequence(graph: HostPrepGraph) -> tuple[str, ...]:
    return tuple(_import_path(n.callable_ref) for n in graph.nodes)


def _score_graph(graph: HostPrepGraph, task: AuthorBenchTask) -> TaskOutcome:
    structurally_valid = validate_graph(graph, root=ROOT).passed
    faithful = _node_sequence(graph) == task.expected_sequence
    return TaskOutcome(task.task_id, structurally_valid, faithful)


def run_generate_then_validate_mode(tasks: tuple[AuthorBenchTask, ...] = TASKS) -> ModeResult:
    """Run T1-12's real, unmodified default path against every task.

    `TemplateGenerator` is blind to the task (no prompt input) -- a low faithfulness rate here
    is an expected, honest baseline finding, not a bug.
    """
    outcomes = []
    for idx, task in enumerate(tasks):
        graph = TemplateGenerator().generate(seed=idx, num_nodes=len(task.expected_sequence))
        outcomes.append(_score_graph(graph, task))
    return ModeResult("generate-then-validate", tuple(outcomes))


def _task_prompt(task: AuthorBenchTask) -> str:
    refs = ", ".join(task.expected_sequence)
    return (
        f"Author a minimal xtrax composition IR document that will {task.nl_prompt}. "
        f"Use only these callable_ref values, in this exact order: {refs}. No edges, no citations."
    )


def run_constrained_decoding_mode(
    tasks: tuple[AuthorBenchTask, ...] = TASKS, model_name: str = DEFAULT_CONSTRAINED_MODEL
) -> ModeResult:
    """Run the Outlines constrained-decoding smoke path against every task.

    A decode that fails to deserialize into a HostPrepGraph at all (malformed callable_ref,
    unknown schema_version, etc.) is scored as structurally invalid / unfaithful for that task
    rather than crashing the harness -- this is an expected outcome for a smoke-only path, not
    a harness bug (T1-12's own confirmed live run produced a schema_version=0, empty-nodes
    document, which fails deserialization outright).
    """
    from scripts import smoke_outlines_constrained_decode as smoke

    outcomes = []
    for task in tasks:
        try:
            document = smoke.run_smoke(model_name=model_name, prompt=_task_prompt(task))
            graph = deserialize_graph(document)
        except _DESERIALIZE_ERRORS as exc:
            outcomes.append(TaskOutcome(task.task_id, False, False, error=str(exc)))
            continue
        outcomes.append(_score_graph(graph, task))
    return ModeResult("constrained-decoding", tuple(outcomes))


def run_spot_check(
    items=HUMAN_LABELED_SPOT_CHECK, tasks: tuple[AuthorBenchTask, ...] = TASKS
) -> SpotCheckResult:
    """Execute the real callable chain for each hand-verified item and check the result.

    Sanity-checks the harness's OWN ground-truth arithmetic against synthetic ground truth
    (BATHOS measurement-verification rule) -- independent of either generator mode.
    """
    tasks_by_id = {t.task_id: t for t in tasks}
    n_passed = 0
    for item in items:
        task = tasks_by_id[item.task_id]
        value = item.sample_input
        for ref in task.expected_sequence:
            module_name, qualname = ref.split(":")
            fn = getattr(importlib.import_module(module_name), qualname)
            value = fn(value)
        if value == item.expected_output:
            n_passed += 1
    return SpotCheckResult(n=len(items), n_passed=n_passed)


def _spot_check_gate_passes(pass_rate: float) -> bool:
    return round(pass_rate, 9) >= SPOT_CHECK_PASS_BAR


def decide_fork12(gtv: ModeResult, cd: ModeResult | None) -> str:
    """Provisional fork-12 decision rule, verbatim from the AC.

    Adopt constrained decoding only if its structural-validity yield beats
    generate-then-validate's by >= 10pp AND its faithfulness is non-inferior within a 5pp
    margin. Revisable at benchmark design review.
    """
    if cd is None:
        return "UNDECIDED (constrained-decoding mode not run)"

    # Rates are ratios of small integer task counts (e.g. 6/10 - 5/10), which routinely land
    # on inexact binary floats (0.6 - 0.5 == 0.09999999999999998) -- round before comparing to
    # the pp margins so an exact-boundary result isn't misclassified by float noise.
    validity_delta = round(cd.structural_validity_yield - gtv.structural_validity_yield, 9)
    faithfulness_delta = round(cd.faithfulness_rate - gtv.faithfulness_rate, 9)

    if (
        validity_delta >= FORK12_VALIDITY_MARGIN_PP
        and faithfulness_delta >= -FORK12_FAITHFULNESS_MARGIN_PP
    ):
        return "ADOPT_CONSTRAINED_DECODING"
    return "KEEP_GENERATE_THEN_VALIDATE"


def _mode_result_to_dict(result: ModeResult) -> dict:
    return {
        "mode": result.mode,
        "structural_validity_yield": result.structural_validity_yield,
        "faithfulness_rate": result.faithfulness_rate,
        "task_outcomes": [asdict(o) for o in result.task_outcomes],
    }


def build_report(gtv: ModeResult, cd: ModeResult | None, spot_check: SpotCheckResult) -> dict:
    # Documents (not "structurally guards" -- ModeResult's rate properties are plain
    # sum/len divisions and can never actually be None) that PM2's delta-only failure mode
    # is prevented by construction: no code path in this module builds a report without both
    # absolute rate fields for the default mode.
    assert gtv.structural_validity_yield is not None
    assert gtv.faithfulness_rate is not None

    return {
        "schema_version": 1,
        "generate_then_validate": _mode_result_to_dict(gtv),
        "constrained_decoding": _mode_result_to_dict(cd) if cd is not None else None,
        "human_labeled_spot_check": {
            "n": spot_check.n,
            "n_passed": spot_check.n_passed,
            "pass_rate": spot_check.pass_rate,
            "pass_bar": SPOT_CHECK_PASS_BAR,
            "gate_passed": _spot_check_gate_passes(spot_check.pass_rate),
        },
        "fork12_decision": decide_fork12(gtv, cd),
    }


def _print_summary(report: dict) -> None:
    gtv = report["generate_then_validate"]
    print(
        f"generate-then-validate: structural_validity_yield={gtv['structural_validity_yield']:.0%} "
        f"faithfulness_rate={gtv['faithfulness_rate']:.0%}"
    )
    cd = report["constrained_decoding"]
    if cd is not None:
        print(
            f"constrained-decoding:   "
            f"structural_validity_yield={cd['structural_validity_yield']:.0%} "
            f"faithfulness_rate={cd['faithfulness_rate']:.0%}"
        )
    else:
        print("constrained-decoding:   not run")
    sc = report["human_labeled_spot_check"]
    print(
        f"human-labeled spot-check: {sc['n_passed']}/{sc['n']} "
        f"({sc['pass_rate']:.0%}, bar {sc['pass_bar']:.0%})"
    )
    print(f"fork-12 decision: {report['fork12_decision']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-constrained-decoding",
        action="store_true",
        help="Also run the Outlines constrained-decoding mode (requires the authoring-smoke extra)",
    )
    parser.add_argument("--constrained-model", default=DEFAULT_CONSTRAINED_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args(argv)

    gtv = run_generate_then_validate_mode()

    cd: ModeResult | None = None
    if args.include_constrained_decoding:
        try:
            cd = run_constrained_decoding_mode(model_name=args.constrained_model)
        except ImportError as exc:
            print(
                f"FAIL: --include-constrained-decoding was requested but outlines/transformers/"
                f"torch are not importable ({exc}). Install the authoring-smoke extra: "
                f"uv run --extra authoring-smoke python scripts/graph_author_bench.py "
                f"--include-constrained-decoding",
                file=sys.stderr,
            )
            return 1

    spot_check = run_spot_check()
    report = build_report(gtv, cd, spot_check)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    _print_summary(report)
    print(f"report written to {args.out}")

    if not _spot_check_gate_passes(spot_check.pass_rate):
        print(
            f"FAIL: human-labeled faithfulness spot-check below {SPOT_CHECK_PASS_BAR:.0%} pass bar "
            f"({spot_check.pass_rate:.0%})",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
