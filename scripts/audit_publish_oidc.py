#!/usr/bin/env python3
"""Distribution N7 publish.yml OIDC Trusted Publishing gate (#1461)."""

from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "distribution" / "publish_oidc.toml"


@dataclass(frozen=True)
class PublishOidcConfig:
    version: str
    workflow: str
    human_prerequisite_backlog: int
    tag_pattern: str
    require_workflow_dispatch: bool
    publish_requires_tag_push: bool
    build_markers: tuple[str, ...]
    testpypi_markers: tuple[str, ...]
    pypi_markers: tuple[str, ...]
    guard_markers: tuple[str, ...]
    forbidden_phrases: tuple[str, ...]
    contributing_markers: tuple[str, ...]


def load_publish_oidc_config(config_path: Path) -> PublishOidcConfig:
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    section = data.get("publish")
    if not isinstance(section, dict):
        raise ValueError(f"missing [publish] section in {config_path}")

    version = section.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("publish.version must be a non-empty string")

    workflow = section.get("workflow")
    if not isinstance(workflow, str) or not workflow:
        raise ValueError("publish.workflow must be a non-empty string")

    human_gate = section.get("human_prerequisite_backlog")
    if not isinstance(human_gate, int) or human_gate <= 0:
        raise ValueError("publish.human_prerequisite_backlog must be a positive int")

    triggers = section.get("triggers", {})
    if not isinstance(triggers, dict):
        raise ValueError("publish.triggers must be a table")
    tag_pattern = triggers.get("tag_pattern")
    if not isinstance(tag_pattern, str) or not tag_pattern:
        raise ValueError("publish.triggers.tag_pattern must be a non-empty string")
    require_dispatch = triggers.get("require_workflow_dispatch")
    publish_requires_tag = triggers.get("publish_requires_tag_push")
    if not isinstance(require_dispatch, bool) or not isinstance(
        publish_requires_tag, bool
    ):
        raise ValueError(
            "publish.triggers require_workflow_dispatch and "
            "publish_requires_tag_push must be booleans"
        )

    def _nested_markers(table_name: str) -> tuple[str, ...]:
        table = section.get(table_name, {})
        if not isinstance(table, dict):
            raise ValueError(f"publish.{table_name} must be a table")
        markers = table.get("markers")
        if not isinstance(markers, list) or not markers:
            raise ValueError(f"publish.{table_name}.markers must be a non-empty list")
        return tuple(str(item) for item in markers)

    forbidden_section = section.get("forbidden", {})
    if not isinstance(forbidden_section, dict):
        raise ValueError("publish.forbidden must be a table")
    phrases = forbidden_section.get("phrases")
    if not isinstance(phrases, list) or not phrases:
        raise ValueError("publish.forbidden.phrases must be a non-empty list")

    return PublishOidcConfig(
        version=version,
        workflow=workflow,
        human_prerequisite_backlog=human_gate,
        tag_pattern=tag_pattern,
        require_workflow_dispatch=require_dispatch,
        publish_requires_tag_push=publish_requires_tag,
        build_markers=_nested_markers("build_markers"),
        testpypi_markers=_nested_markers("testpypi_markers"),
        pypi_markers=_nested_markers("pypi_markers"),
        guard_markers=_nested_markers("guard_markers"),
        forbidden_phrases=tuple(str(item) for item in phrases),
        contributing_markers=_nested_markers("contributing_markers"),
    )


def _check_markers(
    failures: list[str],
    *,
    rel_path: str,
    text: str,
    markers: tuple[str, ...],
) -> None:
    for marker in markers:
        if marker not in text:
            failures.append(f"{rel_path} missing marker: {marker!r}")


def audit_publish_oidc(
    root: Path,
    config_path: Path,
) -> tuple[bool, list[str]]:
    config = load_publish_oidc_config(config_path)
    failures: list[str] = []

    workflow_path = root / config.workflow
    if not workflow_path.is_file():
        failures.append(f"missing publish workflow: {config.workflow}")
        workflow_text = ""
    else:
        workflow_text = workflow_path.read_text(encoding="utf-8")
        if f"- '{config.tag_pattern}'" not in workflow_text:
            failures.append(
                f"{config.workflow} missing tag trigger pattern {config.tag_pattern!r}"
            )
        if config.require_workflow_dispatch and (
            "workflow_dispatch" not in workflow_text
        ):
            failures.append(f"{config.workflow} missing workflow_dispatch trigger")
        _check_markers(
            failures,
            rel_path=config.workflow,
            text=workflow_text,
            markers=config.build_markers,
        )
        _check_markers(
            failures,
            rel_path=config.workflow,
            text=workflow_text,
            markers=config.testpypi_markers,
        )
        _check_markers(
            failures,
            rel_path=config.workflow,
            text=workflow_text,
            markers=config.pypi_markers,
        )
        if config.publish_requires_tag_push:
            _check_markers(
                failures,
                rel_path=config.workflow,
                text=workflow_text,
                markers=config.guard_markers,
            )
        for phrase in config.forbidden_phrases:
            if phrase in workflow_text:
                failures.append(
                    f"{config.workflow} contains forbidden phrase: {phrase!r}"
                )

    contributing = root / "CONTRIBUTING.md"
    if not contributing.is_file():
        failures.append("missing CONTRIBUTING.md release prerequisites")
    else:
        _check_markers(
            failures,
            rel_path="CONTRIBUTING.md",
            text=contributing.read_text(encoding="utf-8"),
            markers=config.contributing_markers,
        )

    return len(failures) == 0, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to publish_oidc.toml",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root",
    )
    args = parser.parse_args(argv)

    passed, failures = audit_publish_oidc(
        root=args.root.resolve(),
        config_path=args.config.resolve(),
    )
    if passed:
        print("PASS: publish OIDC gate")
        return 0

    print("FAIL: publish OIDC gate", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
