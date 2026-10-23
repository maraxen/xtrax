"""Upstream origin of a module that carries code from another project.

A module declares its origin with ``__provenance__``, either a
:class:`Provenance` call or a dict of the same fields. The reader collects
those declarations by parsing source. It does not contact the network.

Profiling modules stay a leaf and cannot import this module, so they use the
dict spelling. Every other module uses ``Provenance(...)``.

:func:`resolve_distribution` covers the other direction: which installed
distribution a third-party package runs from, and whether the imported module
is actually part of it.
"""

import ast
import hashlib
import importlib.util
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

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
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            if isinstance(node.target, ast.Name) and node.target.id == _DECLARATION:
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


UNKNOWN = "unknown"
DISTRIBUTION_FIELDS = (
    "name",
    "version",
    "install_kind",
    "wheel_sha256",
    "vcs_commit",
    "source_url",
    "record_sha256",
    "module",
    "module_file",
    "module_in_dist",
)
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
# scheme://userinfo@ -- the userinfo part may carry a token.
_URL_USERINFO = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*://)[^/?#]*@")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _redact_userinfo(url: str) -> str:
    return _URL_USERINFO.sub(r"\1", url, count=1)


def _url_path(url: str) -> str:
    return url.split("#", 1)[0].split("?", 1)[0]


_MALFORMED: dict[str, object] = {}


def _direct_url(dist: metadata.Distribution) -> dict[str, object] | None:
    """The PEP 610 record; None when absent, ``_MALFORMED`` when unreadable."""
    text = dist.read_text("direct_url.json")
    if text is None:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return _MALFORMED
    return parsed if isinstance(parsed, dict) else _MALFORMED


def _archive_sha256(info: Mapping[Any, Any]) -> str | None:
    hashes = info.get("hashes")
    if isinstance(hashes, dict):
        value = hashes.get("sha256")
        if isinstance(value, str) and _SHA256_HEX.fullmatch(value):
            return value
    legacy = info.get("hash")
    if isinstance(legacy, str) and legacy.startswith("sha256="):
        value = legacy.removeprefix("sha256=")
        if _SHA256_HEX.fullmatch(value):
            return value
    return None


def _install_fields(direct: dict[str, object] | None) -> dict[str, str]:
    fields = {
        "install_kind": "index" if direct is None else UNKNOWN,
        "wheel_sha256": UNKNOWN,
        "vcs_commit": UNKNOWN,
        "source_url": UNKNOWN,
    }
    if direct is None or direct is _MALFORMED:
        return fields
    url = direct.get("url")
    if isinstance(url, str) and url:
        fields["source_url"] = _redact_userinfo(url)
    vcs = direct.get("vcs_info")
    archive = direct.get("archive_info")
    dir_info = direct.get("dir_info")
    if isinstance(vcs, dict):
        fields["install_kind"] = "vcs"
        commit = vcs.get("commit_id")
        if isinstance(commit, str) and commit:
            fields["vcs_commit"] = commit
    elif isinstance(archive, dict):
        fields["install_kind"] = "archive"
        sha = _archive_sha256(archive)
        if sha is not None and isinstance(url, str) and _url_path(url).endswith(".whl"):
            fields["wheel_sha256"] = sha
    elif isinstance(dir_info, dict):
        fields["install_kind"] = "editable" if dir_info.get("editable") is True else "directory"
    return fields


def _default_module(dist: metadata.Distribution) -> str | None:
    wanted = _normalize(dist.metadata["Name"] or "")
    candidates = sorted(
        module
        for module, owners in metadata.packages_distributions().items()
        if wanted in {_normalize(owner) for owner in owners}
    )
    preferred = wanted.replace("-", "_")
    if preferred in candidates:
        return preferred
    if len(candidates) == 1:
        return candidates[0]
    return None


def _module_file(module: str) -> str | None:
    loaded = sys.modules.get(module)
    if loaded is not None:
        location = getattr(loaded, "__file__", None)
        return location if isinstance(location, str) else None
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.has_location or spec.origin is None:
        return None
    return spec.origin


def _local_dir(url: str) -> Path | None:
    try:
        return Path.from_uri(url).resolve()
    except ValueError:
        return None


def _module_in_dist(
    dist: metadata.Distribution, install: dict[str, str], direct: dict[str, object] | None
) -> str:
    """Whether the module file is one the distribution installed: "true", "false" or "unknown"."""
    location = install["module_file"]
    if location == UNKNOWN:
        return UNKNOWN
    target = Path(location).resolve()
    if install["install_kind"] == "editable":
        url = direct.get("url") if direct is not None else None
        root = _local_dir(url) if isinstance(url, str) else None
        if root is None:
            return UNKNOWN
        return "true" if target.is_relative_to(root) else "false"
    files = dist.files
    if files is None:
        return UNKNOWN
    installed = {Path(str(dist.locate_file(entry))).resolve() for entry in files}
    return "true" if target in installed else "false"


def resolve_distribution(name: str, *, module: str | None = None) -> dict[str, str]:
    """Identify the installed distribution ``name`` and the module that would run from it.

    Returns one string per :data:`DISTRIBUTION_FIELDS` entry. A value that the
    installed metadata does not state is ``"unknown"``; nothing is inferred
    from the version string or fetched.

    ``install_kind`` is ``index`` (no PEP 610 record), ``vcs``, ``archive``,
    ``editable``, ``directory``, ``not_installed`` or ``unknown``.
    ``wheel_sha256`` and ``vcs_commit`` come from ``direct_url.json``;
    installers record a wheel hash only for some direct installs, so an index
    install reports ``unknown``. ``record_sha256`` is the sha256 of the
    installed ``RECORD`` file, which identifies the installed file set.
    ``source_url`` has any credentials removed.

    ``module`` defaults to the top-level module the distribution provides.
    ``module_file`` is taken from ``sys.modules`` when the module is imported,
    otherwise from ``importlib.util.find_spec``. ``module_in_dist`` is
    ``"true"`` when that file is one the distribution installed (or, for an
    editable install, lies under its source directory), ``"false"`` when a
    copy elsewhere shadows it, and ``"unknown"`` when either side is missing.

    Reads installed metadata only: no subprocess, no network, no writes.
    """
    if _blank(name):
        raise ValueError("distribution name is required")
    name = name.strip()
    record = dict.fromkeys(DISTRIBUTION_FIELDS, UNKNOWN)
    record["name"] = name
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        record["install_kind"] = "not_installed"
        if module is not None:
            record["module"] = module
            record["module_file"] = _module_file(module) or UNKNOWN
            if record["module_file"] != UNKNOWN:
                record["module_in_dist"] = "false"
        return record
    record["name"] = dist.metadata["Name"] or name
    record["version"] = dist.version or UNKNOWN
    direct = _direct_url(dist)
    record.update(_install_fields(direct))
    record_text = dist.read_text("RECORD")
    if record_text is not None:
        record["record_sha256"] = hashlib.sha256(record_text.encode("utf-8")).hexdigest()
    resolved_module = module if module is not None else _default_module(dist)
    if resolved_module is not None:
        record["module"] = resolved_module
        record["module_file"] = _module_file(resolved_module) or UNKNOWN
    record["module_in_dist"] = _module_in_dist(dist, record, direct)
    return record


__all__ = [
    "DISTRIBUTION_FIELDS",
    "UNKNOWN",
    "Provenance",
    "collect_provenance",
    "read_provenance",
    "render_declaration",
    "resolve_distribution",
]
