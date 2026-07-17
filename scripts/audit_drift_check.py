#!/usr/bin/env python3
"""T3-03 (backlog #3041, AC-X3 + AC-X3b): D3 drift `--check` with independent expectation.

xtrax-side drift gate diffing authored (post-SubFlow-rewrite) templates in
`agent_assets/workflows/` against their `~/.praxia/workflows/xtrax_<name>.yaml` exports.

Resolved MANDATORY by T3-06/AC-V1 (`.praxia/docs/research/260710_t3-06-session-init-drift-
verification.md`): praxia's free `SessionContext::init` drift-check only fires from an explicit
`praxia plugin export`/`install`/`uninstall` CLI call, never from ordinary session lifecycle --
so this gate is the only thing standing between an authored-template edit and a stale export
shipping. It is belt-and-suspenders for nothing; it is the gate.

AC-X3b (PM-3) is the load-bearing half of this script: the expected export is computed
INDEPENDENTLY, not by echoing praxia's own export/rewrite transform. Concretely:

  - This module never calls `praxia`, never imports any praxia crate/binding, and never reads
    the real export before deciding what the export SHOULD look like.
  - It reimplements the ORTHOGONAL-4/AC-P4 SubFlow-prefix rewrite from scratch in Python,
    against the documented `WorkflowNode::SubFlow` schema read directly from the praxia source
    (`crates/praxia-workflows/src/dsl.rs`: `#[serde(tag = "kind", rename_all = "snake_case")]`
    node with fields `id` + `template` -> YAML shape `kind: sub_flow` / `template: <name>`).
    Rust vs. Python, hand-written vs. (currently unimplemented upstream) -- zero shared code path.
  - The rewrite is AST-scoped (`yaml.compose`, a byte-position-aware node graph), not a substring
    search: only the `template:` scalar of a node whose own `kind:` scalar is exactly `sub_flow`
    is ever a rewrite candidate, and only when its value is an EXACT match against a name
    declared in THIS plugin's own `.praxia/manifest.toml` (never a fuzzy/substring match). A
    description or prompt field that happens to contain a sibling template's name as a substring
    is structurally unreachable by this code path -- it is simply never visited. This is the
    fix for PM-3's failure mode (a naive string-substitution rewrite corrupting an unrelated
    field, with the drift gate not noticing because it echoed the same broken transform on both
    sides).
  - Every code path that cannot confidently determine the expected form (parse failure,
    unexpected structure, an unresolvable SubFlow ref, an unsupported scalar style) raises
    `DriftComputationError` and the gate FAILS CLOSED (non-zero exit) -- it never falls back to
    treating "could not compute" as a pass.

`--check` is accepted for CLI-contract parity with praxia's `dw emit --check` convention this
gate is documented to mirror; the script has exactly one mode (diff and report, never write,
never heal) regardless of whether the flag is passed. D1 (auto-heal via `praxia plugin
export`/`install`) remains the local-dev convenience path and is out of scope here by design --
this script is read-only.
"""

from __future__ import annotations

import argparse
import difflib
import logging
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = ROOT / ".praxia" / "manifest.toml"
DEFAULT_EXPORTS_DIR = Path.home() / ".praxia" / "workflows"

# praxia's WorkflowNode enum tag for an inline sub-workflow reference
# (crates/praxia-workflows/src/dsl.rs, `#[serde(tag = "kind", rename_all = "snake_case")]`).
SUBFLOW_KIND = "sub_flow"

CheckStatus = Literal["match", "mismatch", "missing_export", "compute_error"]


class DriftComputationError(RuntimeError):
    """Raised when the expected export cannot be independently computed.

    AC-X3b's fail-closed clause: if D3 cannot compute the expected form, it must fail closed
    (non-zero exit) -- never exit 0 by echoing the real export or by silently skipping.
    """


@dataclass(frozen=True, slots=True)
class WorkflowEntry:
    name: str
    template_path: str


@dataclass(frozen=True, slots=True)
class ManifestData:
    plugin_name: str
    workflows: tuple[WorkflowEntry, ...]

    @property
    def known_names(self) -> frozenset[str]:
        """Bare (unprefixed) names this plugin itself declares -- the only legitimate SubFlow
        rewrite targets. Scoping to the plugin's OWN manifest (rather than e.g. any file that
        exists on disk) keeps the rewrite decision independently verifiable from xtrax's own
        source of truth, never inferred from praxia's output.
        """
        return frozenset(w.name for w in self.workflows)


@dataclass(frozen=True, slots=True)
class TemplateCheckResult:
    name: str
    template_path: str
    export_path: Path
    status: CheckStatus
    message: str
    diff: str = ""


def load_manifest(manifest_path: Path) -> ManifestData:
    """Parse `.praxia/manifest.toml`. Fails closed (raises) on any structural problem."""
    if not manifest_path.is_file():
        raise DriftComputationError(f"manifest not found: {manifest_path}")

    try:
        data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise DriftComputationError(f"manifest {manifest_path} is not valid TOML: {exc}") from exc

    try:
        plugin = data["plugin"]
        plugin_name = plugin["name"]
        raw_workflows = plugin["workflows"]
    except KeyError as exc:
        raise DriftComputationError(
            f"manifest {manifest_path} missing required key: {exc}"
        ) from exc

    if not isinstance(raw_workflows, list):
        raise DriftComputationError(f"manifest {manifest_path}: plugin.workflows is not a list")

    workflows: list[WorkflowEntry] = []
    for entry in raw_workflows:
        try:
            workflows.append(
                WorkflowEntry(name=entry["name"], template_path=entry["template_path"])
            )
        except (KeyError, TypeError) as exc:
            raise DriftComputationError(
                f"manifest {manifest_path}: malformed [[plugin.workflows]] entry {entry!r}: {exc}"
            ) from exc

    if not workflows:
        raise DriftComputationError(
            f"manifest {manifest_path} declares zero [[plugin.workflows]] entries"
        )

    return ManifestData(plugin_name=plugin_name, workflows=tuple(workflows))


def _splice(text: str, replacements: list[tuple[int, int, str]]) -> str:
    """Apply non-overlapping `(start, end, new_text)` byte-index replacements to `text`.

    Every byte outside a replacement span passes through completely untouched -- this is what
    makes a template with zero SubFlow nodes come back byte-identical to its source, and what
    keeps any rewrite scoped to exactly the spans that were found and validated.
    """
    if not replacements:
        return text

    ordered = sorted(replacements, key=lambda r: r[0])
    for (_, end, _new), (next_start, _, _next_new) in zip(ordered, ordered[1:], strict=False):
        if next_start < end:
            raise DriftComputationError("overlapping SubFlow rewrite spans -- refusing to splice")

    out: list[str] = []
    cursor = 0
    for start, end, new_text in ordered:
        if not (0 <= start <= end <= len(text)):
            raise DriftComputationError(f"invalid splice span ({start}, {end})")
        out.append(text[cursor:start])
        out.append(new_text)
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def compute_expected_export(
    source_text: str,
    plugin_name: str,
    known_names: frozenset[str],
) -> str:
    """Independently compute the post-SubFlow-rewrite expected export of `source_text`.

    See the module docstring for the independence argument. Summary of the scoping rule: a
    scalar is a rewrite candidate if and only if it is the `template:` value of a mapping that
    is itself an element of the top-level `nodes:` sequence AND that same mapping's `kind:`
    scalar is exactly `"sub_flow"`. Nothing else in the document is ever inspected for a
    rewrite -- description/prompt/id/role fields are structurally unreachable here.
    """
    try:
        root = yaml.compose(source_text)
    except yaml.YAMLError as exc:
        raise DriftComputationError(f"cannot parse template as YAML: {exc}") from exc

    if not isinstance(root, yaml.MappingNode):
        raise DriftComputationError("template top level is not a YAML mapping")

    nodes_seq: yaml.Node | None = None
    for key_node, value_node in root.value:
        if isinstance(key_node, yaml.ScalarNode) and key_node.value == "nodes":
            nodes_seq = value_node
            break

    if nodes_seq is None:
        raise DriftComputationError("template has no top-level 'nodes' key")
    if not isinstance(nodes_seq, yaml.SequenceNode):
        raise DriftComputationError("'nodes' key is not a YAML sequence")

    prefixed_known = {f"{plugin_name}_{name}" for name in known_names}
    replacements: list[tuple[int, int, str]] = []

    for item in nodes_seq.value:
        if not isinstance(item, yaml.MappingNode):
            raise DriftComputationError("a 'nodes' entry is not a YAML mapping")

        kind_value: str | None = None
        template_node: yaml.Node | None = None
        for key_node, value_node in item.value:
            if not isinstance(key_node, yaml.ScalarNode):
                continue
            if key_node.value == "kind" and isinstance(value_node, yaml.ScalarNode):
                kind_value = value_node.value
            elif key_node.value == "template":
                template_node = value_node

        if kind_value != SUBFLOW_KIND:
            continue

        if not isinstance(template_node, yaml.ScalarNode):
            raise DriftComputationError(
                f"a '{SUBFLOW_KIND}' node has no scalar 'template' field to rewrite"
            )

        ref = template_node.value

        if ref in prefixed_known:
            continue  # already prefixed -- idempotent, nothing to rewrite

        if ref not in known_names:
            raise DriftComputationError(
                f"SubFlow node references unknown template {ref!r} "
                f"(not a bare name declared in this plugin's own manifest: {sorted(known_names)})"
            )

        style = template_node.style
        prefixed = f"{plugin_name}_{ref}"
        if style is None:
            new_text = prefixed
        elif style in ('"', "'"):
            new_text = f"{style}{prefixed}{style}"
        else:
            raise DriftComputationError(
                f"SubFlow 'template' scalar has unsupported style {style!r}; refusing to guess"
            )

        replacements.append(
            (template_node.start_mark.index, template_node.end_mark.index, new_text)
        )

    return _splice(source_text, replacements)


def check_template(
    entry: WorkflowEntry,
    *,
    repo_root: Path,
    exports_dir: Path,
    plugin_name: str,
    known_names: frozenset[str],
) -> TemplateCheckResult:
    export_path = exports_dir / f"{plugin_name}_{entry.name}.yaml"
    source_path = repo_root / entry.template_path

    if not source_path.is_file():
        return TemplateCheckResult(
            name=entry.name,
            template_path=entry.template_path,
            export_path=export_path,
            status="compute_error",
            message=f"authored source not found: {source_path}",
        )

    source_text = source_path.read_text(encoding="utf-8")

    try:
        expected_text = compute_expected_export(source_text, plugin_name, known_names)
    except DriftComputationError as exc:
        return TemplateCheckResult(
            name=entry.name,
            template_path=entry.template_path,
            export_path=export_path,
            status="compute_error",
            message=f"cannot independently compute expected export for {entry.name!r}: {exc}",
        )

    if not export_path.is_file():
        return TemplateCheckResult(
            name=entry.name,
            template_path=entry.template_path,
            export_path=export_path,
            status="missing_export",
            message=(
                f"{entry.name!r} has not been installed/exported yet -- no file at "
                f"{export_path}; run `praxia plugin install` first (never treated as a pass)"
            ),
        )

    actual_text = export_path.read_text(encoding="utf-8")

    if actual_text == expected_text:
        return TemplateCheckResult(
            name=entry.name,
            template_path=entry.template_path,
            export_path=export_path,
            status="match",
            message="byte-identical",
        )

    diff = "".join(
        difflib.unified_diff(
            expected_text.splitlines(keepends=True),
            actual_text.splitlines(keepends=True),
            fromfile=f"expected/{plugin_name}_{entry.name}.yaml (independently computed)",
            tofile=str(export_path),
        )
    )
    return TemplateCheckResult(
        name=entry.name,
        template_path=entry.template_path,
        export_path=export_path,
        status="mismatch",
        message=f"drift detected in {entry.name!r}",
        diff=diff,
    )


def run_drift_check(
    *,
    manifest_path: Path,
    repo_root: Path,
    exports_dir: Path,
) -> tuple[bool, list[TemplateCheckResult]]:
    manifest = load_manifest(manifest_path)
    known_names = manifest.known_names
    results = [
        check_template(
            entry,
            repo_root=repo_root,
            exports_dir=exports_dir,
            plugin_name=manifest.plugin_name,
            known_names=known_names,
        )
        for entry in manifest.workflows
    ]
    passed = all(result.status == "match" for result in results)
    return passed, results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to the plugin manifest (default: .praxia/manifest.toml)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=ROOT,
        help="Root used to resolve manifest template_path entries (default: repo root)",
    )
    parser.add_argument(
        "--exports-dir",
        type=Path,
        default=DEFAULT_EXPORTS_DIR,
        help="Directory to diff exports against (default: ~/.praxia/workflows)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Run the drift check (mirrors praxia's `dw emit --check` convention). This script "
            "has exactly one mode -- diff and report, never write, never heal -- so this flag "
            "does not change behavior; it exists for CLI-contract parity with the convention "
            "it is documented to mirror."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        passed, results = run_drift_check(
            manifest_path=args.manifest,
            repo_root=args.repo_root,
            exports_dir=args.exports_dir,
        )
    except DriftComputationError as exc:
        print(f"FAIL: drift --check could not run: {exc}", file=sys.stderr)
        return 1

    if passed:
        print(f"PASS: drift --check ({len(results)} template(s) byte-identical)")
        return 0

    print("FAIL: drift --check", file=sys.stderr)
    for result in results:
        if result.status == "match":
            continue
        print(f"  - {result.name} [{result.status}]: {result.message}", file=sys.stderr)
        if result.diff:
            print(result.diff, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
