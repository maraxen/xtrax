"""Declarative specification schemas for xtrax.campaign.

Provides immutable, slotted dataclasses defining campaign structure,
parameter sweep grids, and arm configurations, with pure host validation,
callable-field partitioning, and TOML serialization.
"""

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class SpecPartitionError(ValueError):
    """Raised when non-serializable fields (e.g. callables) are detected in parameters."""


def _check_no_callables(obj: Any, path: str = "") -> None:
    """Recursively ensure that obj contains no callables, functions, or lambdas."""
    if callable(obj) and not isinstance(obj, type):
        target = f" at {path}" if path else ""
        raise SpecPartitionError(
            f"Callable or lambda detected{target}: {obj!r}. "
            "Campaign specifications must contain only pure, serializable data."
        )
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_no_callables(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            _check_no_callables(item, f"{path}[{idx}]")


_BARE_TOML_KEY = re.compile(r"[A-Za-z0-9_-]+")


def _format_toml_key(key: Any) -> str:
    """Format a TOML key, quoting anything that is not a bare key.

    An unquoted ``optimizer.lr`` is a dotted key and parses back as a nested table, so a
    key outside ``[A-Za-z0-9_-]+`` is emitted as a basic (quoted) string.
    """
    text = str(key)
    if _BARE_TOML_KEY.fullmatch(text):
        return text
    return _format_toml_value(text)


def _format_toml_value(val: Any) -> str:
    """Format primitive Python values into TOML syntax."""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if val == 0.0:
            return "0.0"
        return str(val)
    if isinstance(val, str):
        escaped = (
            val.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
        )
        return f'"{escaped}"'
    if isinstance(val, list):
        items = ", ".join(_format_toml_value(x) for x in val)
        return f"[{items}]"
    if isinstance(val, dict):
        items = ", ".join(
            f"{_format_toml_key(k)} = {_format_toml_value(v)}" for k, v in sorted(val.items())
        )
        return f"{{{items}}}"
    raise TypeError(f"Unsupported TOML type: {type(val)}")


@dataclass(frozen=True, slots=True)
class ParamGrid:
    """Multi-dimensional parameter sweep grid."""

    axes: dict[str, list[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from xtrax.campaign.manifest import canonical_json_bytes, normalize_json_value

        _check_no_callables(self.axes, path="ParamGrid.axes")
        # Unit identity is content-derived, so a repeated value on an axis would plan two
        # rows with the same hash writing the same store.
        for axis, values in self.axes.items():
            seen: set[bytes] = set()
            for value in values:
                key = canonical_json_bytes(normalize_json_value(value))
                if key in seen:
                    msg = f"ParamGrid axis {axis!r} repeats the value {value!r}"
                    raise ValueError(msg)
                seen.add(key)

    def to_dict(self) -> dict[str, Any]:
        return {"axes": {str(k): list(v) for k, v in self.axes.items()}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParamGrid":
        axes = data.get("axes", data)
        return cls(axes={str(k): list(v) for k, v in axes.items()})


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """Specification for a distinct predictor/sampling arm."""

    arm_id: str
    backend: str
    checkpoint_id: str
    fixed_policy: str = "none"
    state_weight_profile: str = "equal"
    parameters: dict[str, Any] = field(default_factory=dict)
    resource_overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.arm_id:
            raise ValueError("arm_id must be a non-empty string")
        if not self.backend:
            raise ValueError("backend must be a non-empty string")
        if not self.checkpoint_id:
            raise ValueError("checkpoint_id must be a non-empty string")
        _check_no_callables(self.parameters, path=f"ArmSpec({self.arm_id}).parameters")
        _check_no_callables(
            self.resource_overrides, path=f"ArmSpec({self.arm_id}).resource_overrides"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "backend": self.backend,
            "checkpoint_id": self.checkpoint_id,
            "fixed_policy": self.fixed_policy,
            "state_weight_profile": self.state_weight_profile,
            "parameters": dict(self.parameters),
            "resource_overrides": dict(self.resource_overrides),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArmSpec":
        return cls(
            arm_id=str(data["arm_id"]),
            backend=str(data["backend"]),
            checkpoint_id=str(data["checkpoint_id"]),
            fixed_policy=str(data.get("fixed_policy", "none")),
            state_weight_profile=str(data.get("state_weight_profile", "equal")),
            parameters=dict(data.get("parameters", {})),
            resource_overrides=dict(data.get("resource_overrides", {})),
        )


@dataclass(frozen=True, slots=True)
class CampaignSpec:
    """Top-level campaign declaration."""

    campaign_id: str
    target_inputs: list[str]
    arms: list[ArmSpec]
    grid: ParamGrid = field(default_factory=ParamGrid)
    samples_per_target: int = 100
    chunk_size: int = 100
    seed: int = 42
    output_dir: str = "outputs"
    metadata: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.campaign_id:
            raise ValueError("campaign_id must be a non-empty string")
        if not self.target_inputs:
            raise ValueError("target_inputs must be non-empty")
        if not self.arms:
            raise ValueError("arms must contain at least one ArmSpec")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if self.samples_per_target < 1:
            raise ValueError(f"samples_per_target must be >= 1, got {self.samples_per_target}")

        seen_targets: set[str] = set()
        for target in self.target_inputs:
            if target in seen_targets:
                msg = f"Duplicate target_input in CampaignSpec.target_inputs: {target!r}"
                raise ValueError(msg)
            seen_targets.add(target)

        seen_arm_ids: set[str] = set()
        for arm in self.arms:
            if arm.arm_id in seen_arm_ids:
                msg = f"Duplicate arm_id in CampaignSpec.arms: {arm.arm_id!r}"
                raise ValueError(msg)
            seen_arm_ids.add(arm.arm_id)

        _check_no_callables(self.metadata, path="CampaignSpec.metadata")
        _check_no_callables(self.parameters, path="CampaignSpec.parameters")

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "target_inputs": list(self.target_inputs),
            "arms": [arm.to_dict() for arm in self.arms],
            "grid": self.grid.to_dict(),
            "samples_per_target": self.samples_per_target,
            "chunk_size": self.chunk_size,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "metadata": dict(self.metadata),
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CampaignSpec":
        arms_raw = data.get("arms", [])
        arms = [a if isinstance(a, ArmSpec) else ArmSpec.from_dict(a) for a in arms_raw]
        grid_raw = data.get("grid", {})
        if isinstance(grid_raw, ParamGrid):
            grid = grid_raw
        elif isinstance(grid_raw, dict):
            grid = ParamGrid.from_dict(grid_raw)
        else:
            grid = ParamGrid()

        return cls(
            campaign_id=str(data["campaign_id"]),
            target_inputs=[str(t) for t in data.get("target_inputs", [])],
            arms=arms,
            grid=grid,
            samples_per_target=int(data.get("samples_per_target", 100)),
            chunk_size=int(data.get("chunk_size", 100)),
            seed=int(data.get("seed", 42)),
            output_dir=str(data.get("output_dir", "outputs")),
            metadata=dict(data.get("metadata", {})),
            parameters=dict(data.get("parameters", {})),
        )

    def to_toml(self, path: str | Path | None = None) -> str:
        """Serialize CampaignSpec to TOML format, optionally saving to path."""
        lines: list[str] = [
            f"campaign_id = {_format_toml_value(self.campaign_id)}",
            f"target_inputs = {_format_toml_value(list(self.target_inputs))}",
            f"samples_per_target = {self.samples_per_target}",
            f"chunk_size = {self.chunk_size}",
            f"seed = {self.seed}",
            f"output_dir = {_format_toml_value(self.output_dir)}",
        ]
        if self.parameters:
            lines.append(f"parameters = {_format_toml_value(self.parameters)}")

        if self.metadata:
            lines.append("")
            lines.append("[metadata]")
            for k, v in sorted(self.metadata.items()):
                lines.append(f"{_format_toml_key(k)} = {_format_toml_value(v)}")

        if self.grid.axes:
            lines.append("")
            lines.append("[grid.axes]")
            for k, v in sorted(self.grid.axes.items()):
                lines.append(f"{_format_toml_key(k)} = {_format_toml_value(v)}")

        for arm in self.arms:
            lines.append("")
            lines.append("[[arms]]")
            lines.append(f"arm_id = {_format_toml_value(arm.arm_id)}")
            lines.append(f"backend = {_format_toml_value(arm.backend)}")
            lines.append(f"checkpoint_id = {_format_toml_value(arm.checkpoint_id)}")
            lines.append(f"fixed_policy = {_format_toml_value(arm.fixed_policy)}")
            lines.append(f"state_weight_profile = {_format_toml_value(arm.state_weight_profile)}")
            if arm.parameters:
                lines.append(f"parameters = {_format_toml_value(arm.parameters)}")
            if arm.resource_overrides:
                lines.append(f"resource_overrides = {_format_toml_value(arm.resource_overrides)}")

        toml_text = "\n".join(lines) + "\n"
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(toml_text, encoding="utf-8")
        return toml_text

    @classmethod
    def from_toml(cls, path_or_str: str | Path) -> "CampaignSpec":
        """Parse a CampaignSpec from a TOML string or file path."""
        if isinstance(path_or_str, Path) or (
            isinstance(path_or_str, str) and "\n" not in path_or_str and Path(path_or_str).is_file()
        ):
            content = Path(path_or_str).read_text(encoding="utf-8")
        else:
            content = str(path_or_str)

        data = tomllib.loads(content)
        return cls.from_dict(data)
