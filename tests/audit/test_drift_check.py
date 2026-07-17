"""Tests for T3-03 D3 drift `--check` gate (backlog #3041, AC-X3 + AC-X3b).

The single most important test here is `test_decoy_description_substring_is_not_rewritten`
(PM-3): it is the whole reason AC-X3b exists. Everything else establishes the surrounding
behavior (byte-identical pass, correct rewrite pass, bare-ref-left-in failure, fail-closed
paths for a missing export / malformed manifest / an unresolvable ref).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.audit_drift_check import (
    DriftComputationError,
    ManifestData,
    WorkflowEntry,
    compute_expected_export,
    load_manifest,
    main,
    run_drift_check,
)

ROOT = Path(__file__).resolve().parents[2]

PARENT_NO_SUBFLOW = """\
name: parent
version: "0.1.0"
nodes:
  - kind: role_step
    id: p0
    role: recon
    prompt_hint: |
      Do the recon step.
edges: []
"""

CHILD_TEMPLATE = """\
name: child
version: "0.1.0"
nodes:
  - kind: role_step
    id: c0
    role: fixer
    prompt_hint: fix it
edges: []
"""

PARENT_WITH_BARE_SUBFLOW = """\
name: parent
version: "0.1.0"
nodes:
  - kind: role_step
    id: p0
    role: recon
    prompt_hint: recon step
  - kind: sub_flow
    id: p1
    template: child
edges: []
"""

PARENT_WITH_DECOY_DESCRIPTION = """\
name: parent
version: "0.1.0"
nodes:
  - kind: role_step
    id: p0
    role: recon
    prompt_hint: |
      This step's description happens to mention child in passing -- it must never be rewritten.
  - kind: sub_flow
    id: p1
    template: child
edges: []
"""

PARENT_WITH_QUOTED_SUBFLOW = """\
name: parent
version: "0.1.0"
nodes:
  - kind: role_step
    id: p0
    role: recon
    prompt_hint: recon step
  - kind: sub_flow
    id: p1
    template: "child"
edges: []
"""

PARENT_WITH_ALREADY_PREFIXED_SUBFLOW = """\
name: parent
version: "0.1.0"
nodes:
  - kind: role_step
    id: p0
    role: recon
    prompt_hint: recon step
  - kind: sub_flow
    id: p1
    template: xtrax_child
edges: []
"""

PARENT_WITH_UNKNOWN_SUBFLOW = """\
name: parent
version: "0.1.0"
nodes:
  - kind: sub_flow
    id: p1
    template: not_a_declared_sibling
edges: []
"""


def _write_manifest(repo_root: Path, entries: list[tuple[str, str]]) -> Path:
    """entries: list of (name, template_path relative to repo_root)."""
    praxia_dir = repo_root / ".praxia"
    praxia_dir.mkdir(parents=True, exist_ok=True)
    lines = ["[plugin]", 'name = "xtrax"', 'version = "0.0.0"', ""]
    for name, template_path in entries:
        lines += [
            "[[plugin.workflows]]",
            f'name = "{name}"',
            f'template_path = "{template_path}"',
            "",
        ]
    manifest_path = praxia_dir / "manifest.toml"
    manifest_path.write_text("\n".join(lines), encoding="utf-8")
    return manifest_path


def _write_template(repo_root: Path, rel_path: str, text: str) -> Path:
    path = repo_root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# compute_expected_export: pure unit tests
# ---------------------------------------------------------------------------


def test_no_subflow_refs_returns_source_unchanged() -> None:
    expected = compute_expected_export(PARENT_NO_SUBFLOW, "xtrax", frozenset({"parent"}))
    assert expected == PARENT_NO_SUBFLOW


def test_bare_subflow_ref_is_rewritten_to_prefixed_name() -> None:
    expected = compute_expected_export(
        PARENT_WITH_BARE_SUBFLOW, "xtrax", frozenset({"parent", "child"})
    )
    assert "template: xtrax_child" in expected
    assert "template: child" not in expected
    # Nothing else in the document moved.
    assert expected.replace("template: xtrax_child", "template: child") == PARENT_WITH_BARE_SUBFLOW


def test_already_prefixed_ref_is_idempotent() -> None:
    expected = compute_expected_export(
        PARENT_WITH_ALREADY_PREFIXED_SUBFLOW, "xtrax", frozenset({"parent", "child"})
    )
    assert expected == PARENT_WITH_ALREADY_PREFIXED_SUBFLOW


def test_quoted_subflow_ref_preserves_quote_style() -> None:
    expected = compute_expected_export(
        PARENT_WITH_QUOTED_SUBFLOW, "xtrax", frozenset({"parent", "child"})
    )
    assert 'template: "xtrax_child"' in expected
    assert 'template: "child"' not in expected


def test_decoy_description_substring_is_not_rewritten() -> None:
    """PM-3 / AC-X3b: a description field that happens to contain a sibling template's name
    as a substring must NEVER be touched by the independent rewrite -- only the genuine
    `kind: sub_flow` node's `template:` field is a legitimate rewrite target.
    """
    expected = compute_expected_export(
        PARENT_WITH_DECOY_DESCRIPTION, "xtrax", frozenset({"parent", "child"})
    )
    assert "mention child in passing" in expected, "decoy description must survive untouched"
    assert "mention xtrax_child" not in expected
    assert "template: xtrax_child" in expected, "the genuine SubFlow ref must still be rewritten"


def test_unresolvable_subflow_ref_fails_closed() -> None:
    with pytest.raises(DriftComputationError, match="unknown template"):
        compute_expected_export(PARENT_WITH_UNKNOWN_SUBFLOW, "xtrax", frozenset({"parent"}))


def test_unparseable_yaml_fails_closed() -> None:
    with pytest.raises(DriftComputationError, match="cannot parse"):
        compute_expected_export("not: [valid: yaml: at: all", "xtrax", frozenset())


def test_missing_nodes_key_fails_closed() -> None:
    with pytest.raises(DriftComputationError, match="no top-level 'nodes' key"):
        compute_expected_export('name: parent\nversion: "0.1.0"\n', "xtrax", frozenset())


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------


def test_load_manifest_parses_entries(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, [("parent", "agent_assets/workflows/parent.yaml")])
    manifest = load_manifest(manifest_path)
    assert manifest.plugin_name == "xtrax"
    assert manifest.workflows == (
        WorkflowEntry(name="parent", template_path="agent_assets/workflows/parent.yaml"),
    )
    assert manifest.known_names == frozenset({"parent"})


def test_load_manifest_missing_file_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(DriftComputationError, match="not found"):
        load_manifest(tmp_path / "nope.toml")


def test_load_manifest_empty_workflows_fails_closed(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.toml"
    manifest_path.write_text('[plugin]\nname = "xtrax"\nworkflows = []\n', encoding="utf-8")
    # plugin.workflows parses fine but is empty -- zero declared templates, fail closed.
    with pytest.raises(DriftComputationError, match="zero"):
        load_manifest(manifest_path)


# ---------------------------------------------------------------------------
# run_drift_check / end-to-end scenarios
# ---------------------------------------------------------------------------


def test_case_a_no_subflow_refs_byte_identical_passes(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    exports_dir = tmp_path / "exports"
    exports_dir.mkdir(parents=True)

    manifest_path = _write_manifest(repo_root, [("parent", "agent_assets/workflows/parent.yaml")])
    _write_template(repo_root, "agent_assets/workflows/parent.yaml", PARENT_NO_SUBFLOW)
    (exports_dir / "xtrax_parent.yaml").write_text(PARENT_NO_SUBFLOW, encoding="utf-8")

    passed, results = run_drift_check(
        manifest_path=manifest_path, repo_root=repo_root, exports_dir=exports_dir
    )

    assert passed
    assert [r.status for r in results] == ["match"]


def test_case_b_correctly_rewritten_subflow_passes(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    exports_dir = tmp_path / "exports"
    exports_dir.mkdir(parents=True)

    manifest_path = _write_manifest(
        repo_root,
        [
            ("parent", "agent_assets/workflows/parent.yaml"),
            ("child", "agent_assets/workflows/child.yaml"),
        ],
    )
    _write_template(repo_root, "agent_assets/workflows/parent.yaml", PARENT_WITH_BARE_SUBFLOW)
    _write_template(repo_root, "agent_assets/workflows/child.yaml", CHILD_TEMPLATE)

    correctly_rewritten_parent = PARENT_WITH_BARE_SUBFLOW.replace(
        "template: child", "template: xtrax_child"
    )
    (exports_dir / "xtrax_parent.yaml").write_text(correctly_rewritten_parent, encoding="utf-8")
    (exports_dir / "xtrax_child.yaml").write_text(CHILD_TEMPLATE, encoding="utf-8")

    passed, results = run_drift_check(
        manifest_path=manifest_path, repo_root=repo_root, exports_dir=exports_dir
    )

    assert passed, [r.message for r in results if r.status != "match"]
    assert {r.status for r in results} == {"match"}


def test_case_c_bare_ref_left_unrewritten_fails_with_diff_naming_template(tmp_path: Path) -> None:
    """Mimics today's real praxia export behavior (`fs::copy`, no rewrite at all yet, since
    AC-P4/T3-12 is not implemented upstream): the SubFlow ref survives bare in the export.
    D3 must detect this as drift, not silently pass because both sides "agree".
    """
    repo_root = tmp_path / "repo"
    exports_dir = tmp_path / "exports"
    exports_dir.mkdir(parents=True)

    manifest_path = _write_manifest(
        repo_root,
        [
            ("parent", "agent_assets/workflows/parent.yaml"),
            ("child", "agent_assets/workflows/child.yaml"),
        ],
    )
    _write_template(repo_root, "agent_assets/workflows/parent.yaml", PARENT_WITH_BARE_SUBFLOW)
    _write_template(repo_root, "agent_assets/workflows/child.yaml", CHILD_TEMPLATE)

    # "Real" export is an unmodified copy -- the bare ref was never rewritten.
    (exports_dir / "xtrax_parent.yaml").write_text(PARENT_WITH_BARE_SUBFLOW, encoding="utf-8")
    (exports_dir / "xtrax_child.yaml").write_text(CHILD_TEMPLATE, encoding="utf-8")

    passed, results = run_drift_check(
        manifest_path=manifest_path, repo_root=repo_root, exports_dir=exports_dir
    )

    assert not passed
    by_name = {r.name: r for r in results}
    assert by_name["parent"].status == "mismatch"
    assert by_name["child"].status == "match"
    assert "parent" in by_name["parent"].diff
    assert "template: xtrax_child" in by_name["parent"].diff
    assert "template: child" in by_name["parent"].diff


def test_case_d_decoy_description_over_rewrite_detected(tmp_path: Path) -> None:
    """PM-3's actual failure mode, end to end: a buggy real export applied a naive
    string-substitution rewrite that also corrupted an unrelated description field. D3's
    independently-computed expectation never touches that field, so the mismatch is caught.
    """
    repo_root = tmp_path / "repo"
    exports_dir = tmp_path / "exports"
    exports_dir.mkdir(parents=True)

    manifest_path = _write_manifest(
        repo_root,
        [
            ("parent", "agent_assets/workflows/parent.yaml"),
            ("child", "agent_assets/workflows/child.yaml"),
        ],
    )
    _write_template(repo_root, "agent_assets/workflows/parent.yaml", PARENT_WITH_DECOY_DESCRIPTION)
    _write_template(repo_root, "agent_assets/workflows/child.yaml", CHILD_TEMPLATE)

    # Simulate a naive global string-substitution rewrite: it correctly rewrites the genuine
    # SubFlow ref AND incorrectly rewrites the "child" substring inside the description.
    buggy_export = PARENT_WITH_DECOY_DESCRIPTION.replace("child", "xtrax_child")
    (exports_dir / "xtrax_parent.yaml").write_text(buggy_export, encoding="utf-8")
    (exports_dir / "xtrax_child.yaml").write_text(CHILD_TEMPLATE, encoding="utf-8")

    passed, results = run_drift_check(
        manifest_path=manifest_path, repo_root=repo_root, exports_dir=exports_dir
    )

    assert not passed
    by_name = {r.name: r for r in results}
    assert by_name["parent"].status == "mismatch"
    assert "mention" in by_name["parent"].diff  # the corrupted description line is in the diff


def test_missing_export_fails_closed_never_treated_as_pass(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    exports_dir = tmp_path / "exports"
    exports_dir.mkdir(parents=True)

    manifest_path = _write_manifest(repo_root, [("parent", "agent_assets/workflows/parent.yaml")])
    _write_template(repo_root, "agent_assets/workflows/parent.yaml", PARENT_NO_SUBFLOW)
    # Deliberately do not write xtrax_parent.yaml into exports_dir.

    passed, results = run_drift_check(
        manifest_path=manifest_path, repo_root=repo_root, exports_dir=exports_dir
    )

    assert not passed
    assert results[0].status == "missing_export"


def test_unresolvable_ref_surfaces_as_compute_error_end_to_end(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    exports_dir = tmp_path / "exports"
    exports_dir.mkdir(parents=True)

    manifest_path = _write_manifest(repo_root, [("parent", "agent_assets/workflows/parent.yaml")])
    _write_template(repo_root, "agent_assets/workflows/parent.yaml", PARENT_WITH_UNKNOWN_SUBFLOW)

    passed, results = run_drift_check(
        manifest_path=manifest_path, repo_root=repo_root, exports_dir=exports_dir
    )

    assert not passed
    assert results[0].status == "compute_error"


def test_malformed_manifest_raises(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.toml"
    manifest_path.write_text('[plugin]\nname = "xtrax"\n', encoding="utf-8")
    with pytest.raises(DriftComputationError):
        run_drift_check(manifest_path=manifest_path, repo_root=tmp_path, exports_dir=tmp_path)


# ---------------------------------------------------------------------------
# CLI (main) tests
# ---------------------------------------------------------------------------


def test_main_exits_zero_on_pass(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo_root = tmp_path / "repo"
    exports_dir = tmp_path / "exports"
    exports_dir.mkdir(parents=True)
    manifest_path = _write_manifest(repo_root, [("parent", "agent_assets/workflows/parent.yaml")])
    _write_template(repo_root, "agent_assets/workflows/parent.yaml", PARENT_NO_SUBFLOW)
    (exports_dir / "xtrax_parent.yaml").write_text(PARENT_NO_SUBFLOW, encoding="utf-8")

    code = main(
        [
            "--check",
            "--manifest",
            str(manifest_path),
            "--repo-root",
            str(repo_root),
            "--exports-dir",
            str(exports_dir),
        ]
    )

    assert code == 0
    assert "PASS" in capsys.readouterr().out


def test_main_exits_one_with_diff_naming_drifted_template(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = tmp_path / "repo"
    exports_dir = tmp_path / "exports"
    exports_dir.mkdir(parents=True)
    manifest_path = _write_manifest(
        repo_root,
        [
            ("parent", "agent_assets/workflows/parent.yaml"),
            ("child", "agent_assets/workflows/child.yaml"),
        ],
    )
    _write_template(repo_root, "agent_assets/workflows/parent.yaml", PARENT_WITH_BARE_SUBFLOW)
    _write_template(repo_root, "agent_assets/workflows/child.yaml", CHILD_TEMPLATE)
    (exports_dir / "xtrax_parent.yaml").write_text(PARENT_WITH_BARE_SUBFLOW, encoding="utf-8")
    (exports_dir / "xtrax_child.yaml").write_text(CHILD_TEMPLATE, encoding="utf-8")

    code = main(
        [
            "--check",
            "--manifest",
            str(manifest_path),
            "--repo-root",
            str(repo_root),
            "--exports-dir",
            str(exports_dir),
        ]
    )

    err = capsys.readouterr().err
    assert code == 1
    assert "parent" in err
    assert "template: xtrax_child" in err


def test_main_exits_one_on_manifest_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--manifest", str(tmp_path / "does-not-exist.toml")])
    assert code == 1
    assert "FAIL" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Real-repo smoke test
# ---------------------------------------------------------------------------


def test_real_port_validation_template_computes_cleanly() -> None:
    """port_validation.yaml has zero SubFlow nodes today -- the independent computation must
    return it byte-for-byte unchanged, and must not raise (guards the actual repo asset the
    gate protects, not just fixtures).
    """
    manifest = load_manifest(ROOT / ".praxia" / "manifest.toml")
    entry = next(w for w in manifest.workflows if w.name == "port_validation")
    source_text = (ROOT / entry.template_path).read_text(encoding="utf-8")

    expected = compute_expected_export(source_text, manifest.plugin_name, manifest.known_names)

    assert expected == source_text


def test_manifest_data_and_workflow_entry_are_plain_dataclasses() -> None:
    # Cheap smoke test that the small data types stay easy to construct/compare in tests.
    entry = WorkflowEntry(name="x", template_path="y")
    manifest = ManifestData(plugin_name="xtrax", workflows=(entry,))
    assert manifest.known_names == frozenset({"x"})
