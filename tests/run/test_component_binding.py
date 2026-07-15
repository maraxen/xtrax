"""Tests for the B2-08 xtrax-side component sidecar binding hook (#2181, AC-8)."""

import hashlib
from pathlib import Path

import pytest

from xtrax.composition.graph import HostPrepGraphNode
from xtrax.run.component_binding import (
    ComponentSidecarBinding,
    ComponentSidecarFileMissingError,
    ComponentSidecarRefMissingError,
    component_sidecar_binding,
)

VALID_METADATA = {"nl_description": "Prepare host-side tensors before JIT lowering."}


def _identity(x):
    return x


class TestComponentSidecarBinding:
    def test_binds_to_referenced_sidecar_file(self, tmp_path: Path):
        sidecar_path = tmp_path / "scripts" / "experiments" / "preprocess.bth.toml"
        sidecar_path.parent.mkdir(parents=True)
        sidecar_path.write_text('[experiment]\nhypothesis = "h"\n')

        node = HostPrepGraphNode(
            id="preprocess_node",
            callable_ref=_identity,
            metadata={
                **VALID_METADATA,
                "bathos_sidecar_ref": "scripts/experiments/preprocess.bth.toml",
            },
        )

        binding = component_sidecar_binding(node, tmp_path)
        assert binding.component_id == "preprocess_node"

        expected_sha256 = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
        assert binding.component_sidecar_sha256 == expected_sha256

    def test_hash_changes_when_sidecar_content_changes(self, tmp_path: Path):
        sidecar_path = tmp_path / "component.bth.toml"
        sidecar_path.write_text('[experiment]\nhypothesis = "original"\n')

        node = HostPrepGraphNode(
            id="node_a",
            callable_ref=_identity,
            metadata={**VALID_METADATA, "bathos_sidecar_ref": "component.bth.toml"},
        )
        binding_1 = component_sidecar_binding(node, tmp_path)

        sidecar_path.write_text('[experiment]\nhypothesis = "edited"\n')
        binding_2 = component_sidecar_binding(node, tmp_path)

        assert binding_1.component_sidecar_sha256 != binding_2.component_sidecar_sha256
        assert binding_1.component_id == binding_2.component_id

    def test_missing_bathos_sidecar_ref_raises(self, tmp_path: Path):
        node = HostPrepGraphNode(id="node_a", callable_ref=_identity, metadata=dict(VALID_METADATA))
        with pytest.raises(ComponentSidecarRefMissingError, match="node_a"):
            component_sidecar_binding(node, tmp_path)

    def test_empty_bathos_sidecar_ref_raises(self, tmp_path: Path):
        node = HostPrepGraphNode(
            id="node_a",
            callable_ref=_identity,
            metadata={**VALID_METADATA, "bathos_sidecar_ref": ""},
        )
        with pytest.raises(ComponentSidecarRefMissingError):
            component_sidecar_binding(node, tmp_path)

    def test_nonexistent_sidecar_file_raises(self, tmp_path: Path):
        node = HostPrepGraphNode(
            id="node_a",
            callable_ref=_identity,
            metadata={**VALID_METADATA, "bathos_sidecar_ref": "does_not_exist.bth.toml"},
        )
        with pytest.raises(ComponentSidecarFileMissingError, match="node_a"):
            component_sidecar_binding(node, tmp_path)

    def test_different_nodes_get_independent_bindings(self, tmp_path: Path):
        sidecar_a = tmp_path / "a.bth.toml"
        sidecar_a.write_text("a")
        sidecar_b = tmp_path / "b.bth.toml"
        sidecar_b.write_text("b")

        node_a = HostPrepGraphNode(
            id="node_a",
            callable_ref=_identity,
            metadata={**VALID_METADATA, "bathos_sidecar_ref": "a.bth.toml"},
        )
        node_b = HostPrepGraphNode(
            id="node_b",
            callable_ref=_identity,
            metadata={**VALID_METADATA, "bathos_sidecar_ref": "b.bth.toml"},
        )

        binding_a = component_sidecar_binding(node_a, tmp_path)
        binding_b = component_sidecar_binding(node_b, tmp_path)

        assert binding_a.component_id != binding_b.component_id
        assert binding_a.component_sidecar_sha256 != binding_b.component_sidecar_sha256


class TestAsBthRunFlags:
    def test_produces_correct_flag_pairs(self):
        binding = ComponentSidecarBinding(
            component_id="preprocess_node", component_sidecar_sha256="deadbeef" * 8
        )
        flags = binding.as_bth_run_flags()
        assert flags == [
            "--component-id",
            "preprocess_node",
            "--component-sidecar-sha256",
            "deadbeef" * 8,
        ]

    def test_flags_are_appendable_to_an_argv_list(self):
        binding = ComponentSidecarBinding(component_id="n1", component_sidecar_sha256="abc123")
        argv = ["bth", "run", "--campaign", "camp-1", *binding.as_bth_run_flags(), "script.py"]
        assert argv == [
            "bth",
            "run",
            "--campaign",
            "camp-1",
            "--component-id",
            "n1",
            "--component-sidecar-sha256",
            "abc123",
            "script.py",
        ]
