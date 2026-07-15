"""Component-level sidecar binding hook for the xtrax<->bathos bridge (B2-08, #2181, AC-8).

xtrax-side half of a dual-repo item: bathos owns the schema/API surface (`Run.component_id` /
`Run.component_sidecar_sha256` fields, `bathos.prereg.check_component_sidecar_drift`, `bth run
--component-id`/`--component-sidecar-sha256` CLI flags -- all merged to bathos main). This module
is the named xtrax-side sub-deliverable: "neither half closes alone" (B2-08's own DAG text).

Grounding: `HostPrepGraphNode.metadata` already has an optional `bathos_sidecar_ref` slot
(`xtrax/composition/node_metadata_schema.toml`, description "Path relative to repository root")
-- designed for exactly this purpose, never read by any code until this module. A node's
component-level bathos sidecar is whatever TOML file that slot names; `component_sidecar_sha256`
is the SHA256 of that file's raw bytes -- the same convention bathos's own
`prereg.resolve_sidecar` uses for script-path sidecars (`hashlib.sha256(p.read_bytes())`), just
keyed by a component ID instead of a script path (a `HostPrepGraphNode.id` is already unique
within its graph -- `HostPrepGraph.__post_init__` raises `GraphConstructionError` on duplicates
-- so no new ID scheme is needed).

Cross-tool integration seam (matches `xtrax.run.repro_floor` and
`xtrax.loop.metrics_provenance` EXACTLY -- both already-merged #2181 items crossing into bathos):
xtrax has no dependency on bathos (by design -- a compute library must not depend on the
orchestration/experiment-tracking tool built on top of it). This module computes the
`(component_id, component_sidecar_sha256)` pair a caller passes to `bth run --component-id
<component_id> --component-sidecar-sha256 <component_sidecar_sha256>` -- it does NOT itself shell
out to `bth` or import bathos. "xtrax stays truth-emitting, not gate-owning" is B2-08's own
framing for this exact division: xtrax emits the truth (the component ID + its sidecar's actual
current hash); bathos owns the gate (`check_component_sidecar_drift` decides whether that hash
drifted from the component's first-run manifest). The actual `bth run` invocation happens one
layer up, outside this module -- same as `repro_floor.write_repro_floor_attestation`'s caller
handing the written file to `bth attestation register`.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from xtrax.composition.graph import HostPrepGraphNode


class ComponentSidecarRefMissingError(Exception):
    """`node.metadata` has no `bathos_sidecar_ref` slot -- nothing to bind."""


class ComponentSidecarFileMissingError(Exception):
    """`bathos_sidecar_ref` names a path that doesn't exist under `repo_root`."""


@dataclass(frozen=True, slots=True)
class ComponentSidecarBinding:
    """The `(component_id, component_sidecar_sha256)` pair a caller passes to
    `bth run --component-id <component_id> --component-sidecar-sha256
    <component_sidecar_sha256>` (bathos-side B2-08, merged)."""

    component_id: str
    component_sidecar_sha256: str

    def as_bth_run_flags(self) -> list[str]:
        """The flag/value pairs a caller appends to its own `bth run` argv to bind that run
        to this component. Pure string formatting -- this method does not invoke `bth` or any
        subprocess itself."""
        return [
            "--component-id",
            self.component_id,
            "--component-sidecar-sha256",
            self.component_sidecar_sha256,
        ]


def component_sidecar_binding(node: HostPrepGraphNode, repo_root: Path) -> ComponentSidecarBinding:
    """Compute the bathos component-sidecar binding for a HostPrepGraphNode.

    Args:
        node: the composition node to bind. `component_id` is `node.id` verbatim.
        repo_root: repository root `node.metadata["bathos_sidecar_ref"]` is relative to.

    Raises:
        ComponentSidecarRefMissingError: `node.metadata` has no (or an empty) `bathos_sidecar_ref`
            slot.
        ComponentSidecarFileMissingError: the referenced path doesn't exist under `repo_root`.
    """
    sidecar_ref = node.metadata.get("bathos_sidecar_ref")
    if not sidecar_ref:
        msg = f"HostPrepGraphNode {node.id!r} has no bathos_sidecar_ref in its metadata"
        raise ComponentSidecarRefMissingError(msg)

    sidecar_path = repo_root / sidecar_ref
    if not sidecar_path.exists():
        msg = (
            f"bathos_sidecar_ref {sidecar_ref!r} for node {node.id!r} does not exist at "
            f"{sidecar_path}"
        )
        raise ComponentSidecarFileMissingError(msg)

    sha256 = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    return ComponentSidecarBinding(component_id=node.id, component_sidecar_sha256=sha256)


__all__ = [
    "ComponentSidecarBinding",
    "ComponentSidecarFileMissingError",
    "ComponentSidecarRefMissingError",
    "component_sidecar_binding",
]
