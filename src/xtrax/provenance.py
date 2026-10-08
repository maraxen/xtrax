"""Upstream origin of a module that carries code from another project.

A module declares its origin with ``__provenance__``, either a
:class:`Provenance` call or a dict of the same fields. The reader collects
those declarations by parsing source. It does not contact the network.

Profiling modules stay a leaf and cannot import this module, so they use the
dict spelling. Every other module uses ``Provenance(...)``.
"""

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

RELATIONSHIPS = frozenset({"vendored", "ported", "derived", "inspired"})
_FIELD_NAMES = frozenset({"upstream", "relationship", "revision", "licence", "waiver_reason"})
_SPDX_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-+]*$")
_DECLARATION = "__provenance__"

Form = Literal["call", "dict"]


def _blank(value: str | None) -> bool:
    return not isinstance(value, str) or not value.strip()


@dataclass(frozen=True, slots=True)
class Provenance:
    """Origin of one ported module.

    ``upstream`` is a repository URL or name. ``revision`` is a commit sha or
    a tag. ``licence`` is an SPDX identifier. ``relationship`` is one of
    ``vendored``, ``ported``, ``derived``, or ``inspired``. When a revision
    or a licence cannot be stated, ``waiver_reason`` records why.
    """

    upstream: str
    relationship: str
    revision: str | None = None
    licence: str | None = None
    waiver_reason: str | None = None

    def __post_init__(self) -> None:
        if _blank(self.upstream):
            raise ValueError("upstream is required")
        relationship = self.relationship.strip()
        if relationship not in RELATIONSHIPS:
            raise ValueError(
                f"relationship must be one of {sorted(RELATIONSHIPS)}, got {relationship!r}"
            )
        missing: list[str] = []
        if _blank(self.revision):
            missing.append("revision")
        if _blank(self.licence):
            missing.append("licence")
        if missing and _blank(self.waiver_reason):
            if len(missing) == 2:
                message = "revision and licence are required unless waiver_reason is set"
            else:
                message = f"{missing[0]} is required unless waiver_reason is set"
            raise ValueError(message)
        raw_licence = self.licence
        if isinstance(raw_licence, str) and raw_licence.strip():
            licence = raw_licence.strip()
        else:
            licence = None
        if licence is not None and _SPDX_ID.fullmatch(licence) is None:
            raise ValueError(f"licence must be an SPDX id, got {licence!r}")
        raw_revision = self.revision
        if isinstance(raw_revision, str) and raw_revision.strip():
            revision = raw_revision.strip()
        else:
            revision = None
        raw_waiver = self.waiver_reason
        if isinstance(raw_waiver, str) and raw_waiver.strip():
            waiver = raw_waiver.strip()
        else:
            waiver = None
        object.__setattr__(self, "upstream", self.upstream.strip())
        object.__setattr__(self, "relationship", relationship)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "licence", licence)
        object.__setattr__(self, "waiver_reason", waiver)


def _literal(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and (node.value is None or isinstance(node.value, str)):
        return node.value
    raise ValueError("provenance fields must be string literals or None")


def _is_provenance_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "Provenance"
    return isinstance(func, ast.Attribute) and func.attr == "Provenance"


def _fields_from_value(value: ast.expr) -> dict[str, str | None]:
    if isinstance(value, ast.Dict):
        fields: dict[str, str | None] = {}
        for key, item in zip(value.keys, value.values, strict=True):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                raise ValueError("provenance fields must be string literals or None")
            fields[key.value] = _literal(item)
    elif isinstance(value, ast.Call) and _is_provenance_call(value):
        if value.args:
            raise ValueError("provenance arguments must be keywords")
        fields = {}
        for keyword in value.keywords:
            name = keyword.arg
            if name is None:
                raise ValueError("provenance arguments must be keywords")
            fields[name] = _literal(keyword.value)
    else:
        raise ValueError("expected a Provenance(...) call or a dict")
    unknown = sorted(set(fields) - _FIELD_NAMES)
    if unknown:
        raise ValueError(f"unknown provenance fields: {unknown}")
    return fields


def _declaration_values(tree: ast.AST) -> list[ast.expr]:
    values: list[ast.expr] = []
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if _DECLARATION in names:
                values.append(node.value)
    return values


def _record_from_fields(fields: dict[str, str | None]) -> Provenance:
    upstream = fields.get("upstream")
    relationship = fields.get("relationship")
    return Provenance(
        upstream="" if upstream is None else upstream,
        relationship="" if relationship is None else relationship,
        revision=fields.get("revision"),
        licence=fields.get("licence"),
        waiver_reason=fields.get("waiver_reason"),
    )


def read_provenance(path: Path) -> Provenance | None:
    """Return the module's ``__provenance__`` record, or None when it has none."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = _declaration_values(tree)
    if not values:
        return None
    if len(values) > 1:
        raise ValueError("multiple __provenance__ assignments")
    return _record_from_fields(_fields_from_value(values[0]))


def _module_name(package_dir: Path, path: Path) -> str:
    parts = list(path.relative_to(package_dir.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def collect_provenance(package_dir: Path) -> dict[str, Provenance]:
    """Map dotted module names under ``package_dir`` to their provenance records."""
    found: dict[str, Provenance] = {}
    for path in sorted(package_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        record = read_provenance(path)
        if record is None:
            continue
        found[_module_name(package_dir, path)] = record
    return found


def _render_fields(record: Provenance, *, as_dict: bool) -> list[str]:
    pairs: list[tuple[str, str]] = [
        ("upstream", repr(record.upstream)),
        ("relationship", repr(record.relationship)),
    ]
    if record.revision is not None:
        pairs.append(("revision", repr(record.revision)))
    if record.licence is not None:
        pairs.append(("licence", repr(record.licence)))
    if record.waiver_reason is not None:
        pairs.append(("waiver_reason", repr(record.waiver_reason)))
    if as_dict:
        return [f'    "{name}": {value},' for name, value in pairs]
    return [f"    {name}={value}," for name, value in pairs]


def render_declaration(
    *,
    upstream: str,
    relationship: str,
    revision: str | None = None,
    licence: str | None = None,
    waiver_reason: str | None = None,
    form: Form = "call",
) -> str:
    """Render a ``__provenance__`` declaration. Raises when the record is invalid.

    ``form="dict"`` emits a literal dict, for modules that cannot import this one.
    The arguments are the whole input; nothing is fetched.
    """
    record = Provenance(
        upstream=upstream,
        relationship=relationship,
        revision=revision,
        licence=licence,
        waiver_reason=waiver_reason,
    )
    body = _render_fields(record, as_dict=form == "dict")
    if form == "dict":
        lines = ["__provenance__ = {", *body, "}"]
        return "\n".join(lines) + "\n"
    lines = ["__provenance__ = Provenance(", *body, ")"]
    rendered = "\n".join(lines) + "\n"
    return "from xtrax.provenance import Provenance\n\n" + rendered


__all__ = [
    "Provenance",
    "collect_provenance",
    "read_provenance",
    "render_declaration",
]
