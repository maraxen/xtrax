"""Node metadata slot schema loading + validation (T1-06, #3059).

Canonical home for `node_metadata_schema.toml` and its typed slot vocabulary
(`nl_description` required, `mathjax_label`/`citations`/`script_usage`/`audit_verdict`/
`bathos_sidecar_ref` optional). Promoted here from `scripts/load_capability_registry.py`
(which now re-exports these names) because `HostPrepGraphNode` (graph.py) -- production
code shipped in the wheel -- needs to enforce this schema, and `scripts/` is a devtools-only,
unpackaged directory (`pyproject.toml` ships only `src/xtrax`).

The schema TOML itself lives alongside this module (`node_metadata_schema.toml`) and is
loaded via `importlib.resources`, not a `Path(__file__).parents[N]` walk, so it resolves
correctly both from a repo checkout and from an installed wheel -- verified by building the
wheel and confirming the file needed an explicit `force-include` entry in `pyproject.toml`
to appear inside it at all (the only other non-`.py` file in the wheel, `py.typed`, needed
the same treatment).
"""

import re
import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(-[A-Za-z0-9.]+)?$")
KNOWN_SLOT_TYPES = frozenset({"string", "array", "object", "enum"})


@dataclass(frozen=True)
class SlotDefinition:
    id: str
    required: bool
    type: str
    min_length: int | None = None
    values: tuple[str, ...] | None = None
    fields: dict[str, str] | None = None
    item_type: str | None = None


@dataclass(frozen=True)
class NodeMetadataSchema:
    version: str
    schema_version: str
    slots: tuple[SlotDefinition, ...]


def _require_str(data: dict[str, Any], key: str, *, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: missing or empty string field '{key}'")
    return value.strip()


def _require_str_list(data: dict[str, Any], key: str, *, context: str) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context}: '{key}' must be a non-empty list")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{context}: '{key}' entries must be non-empty strings")
        out.append(item.strip())
    return out


def _parse_slot(raw: dict[str, Any], index: int) -> SlotDefinition:
    ctx = f"slots[{index}]"
    slot_id = _require_str(raw, "id", context=ctx)
    slot_type = _require_str(raw, "type", context=ctx)
    if slot_type not in KNOWN_SLOT_TYPES:
        raise ValueError(f"{ctx}: unknown slot type '{slot_type}'")

    required = raw.get("required")
    if not isinstance(required, bool):
        raise ValueError(f"{ctx}: 'required' must be a boolean")

    min_length: int | None = None
    if "min_length" in raw:
        min_length_raw = raw["min_length"]
        if not isinstance(min_length_raw, int) or min_length_raw < 1:
            raise ValueError(f"{ctx}: 'min_length' must be a positive integer")
        min_length = min_length_raw

    values: tuple[str, ...] | None = None
    if slot_type == "enum":
        raw_values = raw.get("values")
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError(f"{ctx}: enum slot requires non-empty 'values' list")
        values = tuple(_require_str_list(raw, "values", context=ctx))

    fields: dict[str, str] | None = None
    if slot_type == "object":
        raw_fields = raw.get("fields")
        if not isinstance(raw_fields, dict) or not raw_fields:
            raise ValueError(f"{ctx}: object slot requires non-empty 'fields' table")
        fields = {}
        for field_name, field_type in raw_fields.items():
            if not isinstance(field_name, str) or not field_name.strip():
                raise ValueError(f"{ctx}: invalid object field name")
            if field_type != "string":
                raise ValueError(f"{ctx}: unsupported field type '{field_type}' for '{field_name}'")
            fields[field_name.strip()] = field_type

    item_type: str | None = None
    if slot_type == "array":
        item_type = _require_str(raw, "item_type", context=ctx)
        if item_type != "citation":
            raise ValueError(f"{ctx}: unknown array item_type '{item_type}'")

    return SlotDefinition(
        id=slot_id,
        required=required,
        type=slot_type,
        min_length=min_length,
        values=values,
        fields=fields,
        item_type=item_type,
    )


def _default_schema_text() -> str:
    return (
        resources.files("xtrax.composition")
        .joinpath("node_metadata_schema.toml")
        .read_text(encoding="utf-8")
    )


def load_node_metadata_schema(path: Path | None = None) -> NodeMetadataSchema:
    text = path.read_text(encoding="utf-8") if path is not None else _default_schema_text()
    data = tomllib.loads(text)

    schema_raw = data.get("schema")
    if not isinstance(schema_raw, dict):
        raise ValueError("missing [schema] table")

    version = _require_str(schema_raw, "version", context="schema")
    if not SEMVER_RE.match(version):
        raise ValueError(f"schema: invalid semver '{version}'")
    schema_version = _require_str(schema_raw, "schema_version", context="schema")

    raw_slots = data.get("slots")
    if not isinstance(raw_slots, list) or not raw_slots:
        raise ValueError("slots must be a non-empty list")

    slots = tuple(_parse_slot(item, idx) for idx, item in enumerate(raw_slots))
    slot_ids = [slot.id for slot in slots]
    if len(slot_ids) != len(set(slot_ids)):
        raise ValueError("duplicate slot ids in node_metadata_schema")

    return NodeMetadataSchema(
        version=version,
        schema_version=schema_version,
        slots=slots,
    )


def _validate_citation(item: Any, *, context: str) -> None:
    if not isinstance(item, dict):
        raise ValueError(f"{context}: citation must be an object")
    doi = item.get("doi")
    url = item.get("url")
    text = item.get("text")
    has_doi = isinstance(doi, str) and doi.strip()
    has_url = isinstance(url, str) and url.strip()
    has_text = isinstance(text, str) and text.strip()
    if not (has_doi or has_url or has_text):
        raise ValueError(f"{context}: citation requires at least one of doi, url, text")


def _validate_slot_value(slot: SlotDefinition, value: Any, *, context: str) -> None:
    if slot.type == "string":
        if not isinstance(value, str):
            raise ValueError(f"{context}: expected string")
        if slot.min_length is not None and len(value.strip()) < slot.min_length:
            raise ValueError(f"{context}: string shorter than min_length {slot.min_length}")
        return

    if slot.type == "enum":
        if not isinstance(value, str) or value not in (slot.values or ()):
            allowed = ", ".join(slot.values or ())
            raise ValueError(f"{context}: expected one of [{allowed}]")
        return

    if slot.type == "array":
        if not isinstance(value, list):
            raise ValueError(f"{context}: expected array")
        if slot.item_type == "citation":
            for idx, item in enumerate(value):
                _validate_citation(item, context=f"{context}[{idx}]")
        return

    if slot.type == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{context}: expected object")
        if slot.fields:
            for field_name in slot.fields:
                field_value = value.get(field_name)
                if not isinstance(field_value, str) or not field_value.strip():
                    raise ValueError(f"{context}.{field_name}: expected non-empty string")
        return

    raise ValueError(f"{context}: unsupported slot type '{slot.type}'")


def validate_node_metadata(
    metadata: dict[str, Any],
    schema: NodeMetadataSchema | None = None,
) -> None:
    """Validate a node metadata dict against the typed schema."""
    resolved = schema or load_node_metadata_schema()
    slots_by_id = {slot.id: slot for slot in resolved.slots}
    known_ids = set(slots_by_id)

    unknown = set(metadata) - known_ids
    if unknown:
        raise ValueError(f"unknown metadata slots: {sorted(unknown)}")

    for slot in resolved.slots:
        if slot.required and slot.id not in metadata:
            raise ValueError(f"missing required slot '{slot.id}'")

    for key, value in metadata.items():
        _validate_slot_value(slots_by_id[key], value, context=key)


__all__ = [
    "KNOWN_SLOT_TYPES",
    "SEMVER_RE",
    "NodeMetadataSchema",
    "SlotDefinition",
    "load_node_metadata_schema",
    "validate_node_metadata",
]
