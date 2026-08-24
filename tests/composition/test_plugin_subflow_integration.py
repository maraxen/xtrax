"""First-consumer SubFlow integration test (T3-04, AC-X4, backlog id 3042, O-1).

De-risks xtrax being the first installed `[[plugin.workflows]]` consumer (open
question Q4: SubFlow prefixing untested in production). A temp ``~/.praxia`` is
built by actually invoking the real ``praxia plugin install`` CLI (never mocked)
against the fixture plugin at ``tests/fixtures/subflow_plugin/`` -- a two-template
parent/child pair, isolated from and never wired into the real repo-root
``.praxia/manifest.toml``. See that fixture's manifest.toml docstring for why it
is still named "xtrax": the export prefix must match the literal AC-X4 target
name ``xtrax_port_repair``.

This module also carries the decoy-description fixture reused by AC-P4 (T3-12):
the parent template's ``description`` field embeds the child's bare name
("port_repair") as ordinary prose, unrelated to the real SubFlow ref in the
``sub_flow`` node's ``template:`` field. AC-P4 now exists: authors write bare
sibling names and export rewrites only that dest ``template:`` field to the
prefixed form (`xtrax_port_repair`). PM-3's failure mode is a naive
string-substitution rewrite that also corrupts the decoy description. The
child template itself remains a byte-identical copy.

The `plugins.toml` schema asserted below (PM-5) was verified two ways, not
guessed: (1) read directly from praxia's own source --
`crates/praxia-workflows/src/registry.rs::PluginRegistryEntry` and
`crates/praxia-workflows/src/plugin_cli.rs::{read,write}_registry_raw` (top-level
`[plugins]` table, one `[plugins.<name>]` sub-table per plugin with `path`,
`hashes` (BTreeMap keyed by scope target, `_global` present today), and
`exported_files` (BTreeMap keyed by scope target, `_global` present today)) --
and (2) cross-checked against this machine's real, currently-installed
`~/.praxia/plugins.toml` (a live multi-plugin registry file, not xtrax-specific).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "tests/fixtures/subflow_plugin"
FIXTURE_MANIFEST_PATH = FIXTURE_DIR / ".praxia/manifest.toml"
PARENT_SOURCE_PATH = FIXTURE_DIR / "agent_assets/workflows/flow_parent.yaml"
CHILD_SOURCE_PATH = FIXTURE_DIR / "agent_assets/workflows/port_repair.yaml"

PLUGIN_NAME = "xtrax"
PARENT_BARE_NAME = "flow_parent"
CHILD_BARE_NAME = "port_repair"
PARENT_PREFIXED_NAME = f"{PLUGIN_NAME}_{PARENT_BARE_NAME}"
CHILD_PREFIXED_NAME = f"{PLUGIN_NAME}_{CHILD_BARE_NAME}"
DECOY_SUBSTRING = CHILD_BARE_NAME  # "port_repair" -- deliberately embedded in prose


def _load_fixture_manifest() -> dict[str, Any]:
    return tomllib.loads(FIXTURE_MANIFEST_PATH.read_text(encoding="utf-8"))


def _find_subflow_node(parsed_yaml: dict[str, Any]) -> dict[str, Any]:
    yaml_mod = pytest.importorskip("yaml")
    del yaml_mod  # only need the importorskip guard; parsed_yaml is passed in
    sub_flow_nodes = [n for n in parsed_yaml["nodes"] if n.get("kind") == "sub_flow"]
    assert len(sub_flow_nodes) == 1, (
        f"expected exactly one sub_flow node in {parsed_yaml.get('name')!r}, "
        f"found {len(sub_flow_nodes)}: {sub_flow_nodes}"
    )
    return sub_flow_nodes[0]


def _resolve_subflow_child(workflows_dir: Path, child_template_name: str) -> Path:
    """Mirror praxia's real FsTemplateRegistry tier-2 resolution rule.

    `template_registry.rs::FsTemplateRegistry::resolve` looks up `<dir>/<name>.yaml`
    in `~/.praxia/workflows/` (tier 2) among other tiers. Raises a loud,
    template-naming AssertionError (never a silent False) if the child is
    missing -- this is the AC-X4 "Fast/loud: unresolved ref -> test fails
    naming the missing prefixed template" behavior, made independently
    testable (see test_loud_failure_names_the_missing_prefixed_template below).
    """
    candidate = workflows_dir / f"{child_template_name}.yaml"
    assert candidate.is_file(), (
        f"SubFlow child template {child_template_name!r} did NOT resolve at the "
        f"real FsTemplateRegistry tier-2 path {candidate} -- the parent template's "
        f"sub_flow node references this exact prefixed name and cannot dispatch "
        f"it if the file is missing. (AC-X4 fast/loud failure mode.)"
    )
    return candidate


# ---------------------------------------------------------------------------
# Fixture self-checks (no install required) -- prove the fixture itself is
# well-formed and genuinely carries the decoy before trusting the install path.
# ---------------------------------------------------------------------------


def test_fixture_manifest_declares_parent_and_child() -> None:
    data = _load_fixture_manifest()
    assert data["plugin"]["name"] == PLUGIN_NAME
    workflows = data["plugin"]["workflows"]
    names = {w["name"] for w in workflows}
    assert names == {PARENT_BARE_NAME, CHILD_BARE_NAME}, (
        f"fixture manifest must declare exactly the parent+child pair, got {names}"
    )
    for entry in workflows:
        assert (FIXTURE_DIR / entry["template_path"]).is_file(), entry


class TestAcP43054:
    """AC-P4 / T3-12 (mandate 3054): authors write bare SubFlow names; export
    rewrites dest ``template:`` to the prefixed form.
    """

    def test_fixture_parent_authors_subflow_ref_in_post_install_prefixed_form(
        self,
    ) -> None:
        """AC-P4 now exists: authors write the bare sibling name; export
        rewrites dest. The fixture must author ``template: port_repair``.
        """
        yaml = pytest.importorskip("yaml")
        parsed = yaml.safe_load(PARENT_SOURCE_PATH.read_text(encoding="utf-8"))
        sub_flow_node = _find_subflow_node(parsed)
        assert sub_flow_node["template"] == CHILD_BARE_NAME, (
            f"fixture parent's sub_flow node must author the bare child name "
            f"{CHILD_BARE_NAME!r} (AC-P4 rewrites dest to {CHILD_PREFIXED_NAME!r}), "
            f"got {sub_flow_node['template']!r}"
        )


def test_fixture_decoy_description_contains_unrelated_child_name_substring() -> None:
    """The decoy fixture (PM-3, reused by AC-P4/T3-12) must genuinely be present:
    the *bare* child name as prose in an unrelated field, not as the real ref.
    """
    text = PARENT_SOURCE_PATH.read_text(encoding="utf-8")
    yaml = pytest.importorskip("yaml")
    parsed = yaml.safe_load(text)
    description = parsed["description"]
    assert DECOY_SUBSTRING in description, (
        f"decoy fixture missing: parent description must contain the bare "
        f"child-name substring {DECOY_SUBSTRING!r} as unrelated prose"
    )
    # The decoy occurrence must be the BARE name, never the prefixed ref form --
    # otherwise this isn't testing what PM-3 describes (an unrelated field that
    # happens to contain the child name, distinct from the real SubFlow ref slot).
    assert CHILD_PREFIXED_NAME not in description, (
        f"decoy description must not already contain the prefixed form "
        f"{CHILD_PREFIXED_NAME!r} -- that would make it indistinguishable from "
        f"a real ref and defeat the point of the decoy"
    )


def test_loud_failure_names_the_missing_prefixed_template() -> None:
    """Fast/loud (AC-X4): prove `_resolve_subflow_child` actually raises, with
    the missing template's name in the message, when resolution fails --
    exercised deterministically without needing to break the real install.
    """
    with tempfile.TemporaryDirectory() as empty_dir:
        with pytest.raises(AssertionError, match=CHILD_PREFIXED_NAME):
            _resolve_subflow_child(Path(empty_dir), CHILD_PREFIXED_NAME)


# ---------------------------------------------------------------------------
# Real install: the actual first-consumer integration test (O-1).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_install(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Runs the REAL `praxia plugin install` CLI once against the fixture
    plugin, into an isolated, disposable temp HOME. Never touches the real
    `~/.praxia` on this machine (HOME is overridden for the subprocess only).
    """
    tmp_home = tmp_path_factory.mktemp("t3_04_home")
    env = {**os.environ, "HOME": str(tmp_home)}
    # Pin PATH to PRAXIA_BIN's directory (or the first `praxia` on PATH) so a
    # stale later PATH entry cannot win. Do not hardcode a machine-local path.
    praxia_bin = os.environ.get("PRAXIA_BIN") or shutil.which("praxia")
    if praxia_bin:
        env["PATH"] = str(Path(praxia_bin).expanduser().resolve().parent) + os.pathsep + env.get(
            "PATH", ""
        )
    result = subprocess.run(
        ["praxia", "plugin", "install", str(FIXTURE_DIR)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"real `praxia plugin install {FIXTURE_DIR}` failed (exit "
        f"{result.returncode}): stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    workflows_dir = tmp_home / ".praxia/workflows"
    plugins_toml_path = tmp_home / ".praxia/plugins.toml"
    return {
        "tmp_home": tmp_home,
        "workflows_dir": workflows_dir,
        "plugins_toml_path": plugins_toml_path,
    }


@pytest.mark.skipif(shutil.which("praxia") is None, reason="praxia CLI not installed")
def test_child_resolves_via_post_install_prefixed_name(
    real_install: dict[str, Any],
) -> None:
    workflows_dir: Path = real_install["workflows_dir"]
    child_path = _resolve_subflow_child(workflows_dir, CHILD_PREFIXED_NAME)
    assert child_path.read_text(encoding="utf-8") == CHILD_SOURCE_PATH.read_text(
        encoding="utf-8"
    ), "installed child template must be byte-identical to its authored source"


@pytest.mark.skipif(shutil.which("praxia") is None, reason="praxia CLI not installed")
def test_parent_dispatches_child_via_prefixed_subflow_ref(
    real_install: dict[str, Any],
) -> None:
    """Success (AC-X4): "parent resolves + dispatches child." The installed
    parent's sub_flow node is proven to reference an actually-resolvable
    prefixed template -- i.e. is wired to dispatch it, per the real
    FsTemplateRegistry tier-2 (`~/.praxia/workflows/<name>.yaml`) resolution
    rule read directly from template_registry.rs.
    """
    workflows_dir: Path = real_install["workflows_dir"]
    parent_path = workflows_dir / f"{PARENT_PREFIXED_NAME}.yaml"
    assert parent_path.is_file(), (
        f"installed parent template {PARENT_PREFIXED_NAME!r} not found at {parent_path}"
    )

    yaml = pytest.importorskip("yaml")
    parsed_parent = yaml.safe_load(parent_path.read_text(encoding="utf-8"))
    sub_flow_node = _find_subflow_node(parsed_parent)
    referenced_child_name = sub_flow_node["template"]

    # This is the actual dispatch proof: resolve the exact name the installed
    # parent references, using the real tier-2 resolution rule. Fast/loud
    # naming the missing prefixed template if it doesn't resolve.
    _resolve_subflow_child(workflows_dir, referenced_child_name)
    assert referenced_child_name == CHILD_PREFIXED_NAME, (
        f"installed parent's sub_flow node must reference {CHILD_PREFIXED_NAME!r}, "
        f"got {referenced_child_name!r}"
    )


@pytest.mark.skipif(shutil.which("praxia") is None, reason="praxia CLI not installed")
def test_decoy_description_survives_export_unmodified(
    real_install: dict[str, Any],
) -> None:
    """PM-3 regression guard, reused by AC-P4/T3-12: the unrelated prose
    substring must be equal pre/post export. AC-P4 rewrites only the SubFlow
    ``template:`` field on dest (the parent is no longer a byte-identical
    copy); this assertion catches an over-broad rewrite that corrupts
    non-ref fields.
    """
    workflows_dir: Path = real_install["workflows_dir"]
    parent_path = workflows_dir / f"{PARENT_PREFIXED_NAME}.yaml"
    installed_text = parent_path.read_text(encoding="utf-8")
    source_text = PARENT_SOURCE_PATH.read_text(encoding="utf-8")

    yaml = pytest.importorskip("yaml")
    installed_description = yaml.safe_load(installed_text)["description"]
    source_description = yaml.safe_load(source_text)["description"]

    assert installed_description == source_description, (
        "decoy description was modified by export -- expected description "
        "equality (AC-P4 rewrites only the SubFlow template: field); if this "
        "fires, the rewrite is over-broad and corrupting unrelated fields "
        "(exactly the PM-3 failure mode)"
    )
    assert DECOY_SUBSTRING in installed_description, (
        f"decoy substring {DECOY_SUBSTRING!r} missing from installed description "
        f"-- fixture regressed"
    )
    assert CHILD_PREFIXED_NAME not in installed_description, (
        f"decoy description was over-rewritten to contain the prefixed form "
        f"{CHILD_PREFIXED_NAME!r} -- this is exactly the PM-3 corruption failure "
        f"mode (a naive substring rewrite touching an unrelated prose field)"
    )


@pytest.mark.skipif(shutil.which("praxia") is None, reason="praxia CLI not installed")
def test_installed_plugins_toml_matches_realistic_schema(
    real_install: dict[str, Any],
) -> None:
    """PM-5 gate: the plugins.toml schema fixtured here is grounded in praxia's
    real source (registry.rs::PluginRegistryEntry, plugin_cli.rs's
    read/write_registry_raw) and cross-checked against this machine's actual
    ~/.praxia/plugins.toml. Fails loud -- never passes by echo -- if the
    installed schema differs from what's fixtured below.
    """
    plugins_toml_path: Path = real_install["plugins_toml_path"]
    assert plugins_toml_path.is_file(), f"missing {plugins_toml_path}"

    data = tomllib.loads(plugins_toml_path.read_text(encoding="utf-8"))

    assert "plugins" in data, (
        "plugins.toml must have a top-level [plugins] table "
        "(praxia-workflows/src/plugin_cli.rs::write_registry_raw); "
        f"got top-level keys {sorted(data.keys())}"
    )
    plugins = data["plugins"]
    assert PLUGIN_NAME in plugins, (
        f"[plugins.{PLUGIN_NAME}] entry missing from installed plugins.toml; "
        f"got plugins {sorted(plugins.keys())}"
    )
    entry = plugins[PLUGIN_NAME]

    # PluginRegistryEntry's real field set (registry.rs): path, enabled, scope,
    # hashes, exported_files. enabled/scope are Option<..> and absent-by-default
    # on fresh install (never serialized when None by toml::Value::try_from),
    # so a fresh install's on-disk entry is exactly {path, hashes, exported_files}.
    known_fields = {"path", "enabled", "scope", "hashes", "exported_files"}
    unexpected = set(entry.keys()) - known_fields
    assert not unexpected, (
        f"[plugins.{PLUGIN_NAME}] has unfixtured field(s) {sorted(unexpected)} -- "
        f"the real plugins.toml schema has drifted from what this test fixtures "
        f"(PM-5: never pass against an unfixtured schema). Known fields: "
        f"{sorted(known_fields)}"
    )
    required_fields = {"path", "hashes", "exported_files"}
    missing = required_fields - set(entry.keys())
    assert not missing, (
        f"[plugins.{PLUGIN_NAME}] is missing required field(s) {sorted(missing)} "
        f"-- the real plugins.toml schema has drifted from what this test fixtures"
    )

    assert isinstance(entry["path"], str) and entry["path"], (
        f"[plugins.{PLUGIN_NAME}].path must be a non-empty string, got {entry['path']!r}"
    )

    hashes = entry["hashes"]
    assert isinstance(hashes, dict) and "_global" in hashes, (
        f"[plugins.{PLUGIN_NAME}.hashes] must be a table with a '_global' key "
        f"(fresh install always exports to the '_global' scope target per "
        f"plugin_cli.rs::install), got {hashes!r}"
    )
    assert hashes["_global"].startswith("sha256:"), (
        f"[plugins.{PLUGIN_NAME}.hashes._global] must be a 'sha256:<hex>' "
        f"composite hash (registry.rs::compute_composite_hash), got "
        f"{hashes['_global']!r}"
    )

    exported_files = entry["exported_files"]
    assert isinstance(exported_files, dict) and "_global" in exported_files, (
        f"[plugins.{PLUGIN_NAME}.exported_files] must be a table with a "
        f"'_global' key, got {exported_files!r}"
    )
    exported_global = exported_files["_global"]
    assert isinstance(exported_global, list) and len(exported_global) == 2, (
        f"[plugins.{PLUGIN_NAME}.exported_files._global] must list exactly the "
        f"2 fixture workflow exports (flow_parent + port_repair), got "
        f"{exported_global!r}"
    )
    exported_basenames = {Path(p).name for p in exported_global}
    assert exported_basenames == {
        f"{PARENT_PREFIXED_NAME}.yaml",
        f"{CHILD_PREFIXED_NAME}.yaml",
    }, (
        f"[plugins.{PLUGIN_NAME}.exported_files._global] basenames "
        f"{exported_basenames} do not match the expected prefixed exports"
    )
