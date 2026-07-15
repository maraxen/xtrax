"""Generic fail-loud TOML-config primitives (idea-003).

Domain-agnostic building blocks factored out of `xtrax.cli.config`'s
training-shaped `TrainConfig`/`load_config`: presence checks, custom-predicate
field validation, and schema-version classification, each raising a
caller-supplied ``error_cls`` rather than a fixed xtrax exception type. This
module has no dependency on `xtrax.cli` or `tyro` -- any consumer building
its own config-loading layer (not just xtrax's own CLI) can import it
directly.

Spec: `.praxia/docs/specs/260715_generic-fail-loud-toml-to-dataclass-conf.md`
(contemplex session b93ead41). `xtrax.cli.config.load_config` composes these
primitives; see that module for the canonical (dog-fooded) usage.
"""

import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


def load_toml_document(path: str, error_cls: type[Exception]) -> dict[str, Any]:
    """Parse a TOML file into a dict, raising ``error_cls`` on any IO/parse failure.

    Wraps both file-access failures (missing/unreadable path) and malformed-TOML
    failures into one caller-chosen exception type, so callers catch a single
    family regardless of which failure mode occurred.
    """
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except OSError as e:
        raise error_cls(f"cannot read config file '{path}': {e}") from e
    except tomllib.TOMLDecodeError as e:
        raise error_cls(f"malformed TOML in config file '{path}': {e}") from e


def require_sections(raw: dict, sections: Sequence[str], error_cls: type[Exception]) -> None:
    """Raise ``error_cls`` naming every section in ``sections`` missing from ``raw``.

    Collects and names all missing sections in one error (not just the first),
    so a caller fixing their config sees every problem at once.
    """
    missing = [section for section in sections if section not in raw]
    if missing:
        raise error_cls(f"missing required section(s): {', '.join(f'[{s}]' for s in missing)}")


def require_field(
    raw: dict, name: str, predicate: Callable[[Any], bool], error_cls: type[Exception]
) -> Any:
    """Extract ``raw[name]`` and validate it against an arbitrary predicate.

    Returns the value on success. Raises ``error_cls`` naming the field and the
    offending value on failure. ``predicate`` is an arbitrary
    ``Callable[[Any], bool]`` -- this covers custom validation (e.g. "must be a
    positive int") that a bare type-hint check cannot express.
    """
    value = raw.get(name)
    if not predicate(value):
        raise error_cls(f"field '{name}' failed validation, got: {value!r}")
    return value


@dataclass(frozen=True)
class SchemaVersionStatus:
    """Structured classification of a document's schema_version against ``current``.

    ``kind`` is an open string tag, not a fixed enum -- the extension seam for
    future states (e.g. a "deprecated but still supported" kind) is a new
    branch in `classify_schema_version`, not a change to any function's
    signature. Known kinds today: "ok", "missing", "mismatched",
    "newer_than_supported".
    """

    kind: str
    found: int | None
    current: int


def classify_schema_version(raw: dict, current: int) -> SchemaVersionStatus:
    """Classify ``raw``'s schema_version against ``current`` without raising.

    The public extension point: a caller needing custom handling for a status
    `check_schema_version` doesn't cover (e.g. warn-but-continue on a
    deprecated-but-supported version) can call this directly instead of only
    using `check_schema_version`'s raise-or-not wrapper.
    """
    found = raw.get("schema_version")
    if found is None:
        return SchemaVersionStatus(kind="missing", found=None, current=current)
    if not isinstance(found, int):
        return SchemaVersionStatus(kind="mismatched", found=None, current=current)
    if found > current:
        return SchemaVersionStatus(kind="newer_than_supported", found=found, current=current)
    if found != current:
        return SchemaVersionStatus(kind="mismatched", found=found, current=current)
    return SchemaVersionStatus(kind="ok", found=found, current=current)


_SCHEMA_VERSION_MESSAGES = {
    "missing": lambda s: "missing required field: schema_version",
    "mismatched": lambda s: f"schema_version mismatch: expected {s.current}, got {s.found!r}",
    "newer_than_supported": lambda s: (
        f"schema_version {s.found} is newer than the supported version {s.current}"
    ),
}


def check_schema_version(raw: dict, current: int, error_cls: type[Exception]) -> None:
    """Raise ``error_cls`` if ``raw``'s schema_version is missing, mismatched, or
    newer than ``current`` -- distinguishing each case in the message via
    `classify_schema_version`.
    """
    status = classify_schema_version(raw, current)
    if status.kind != "ok":
        raise error_cls(_SCHEMA_VERSION_MESSAGES[status.kind](status))


__all__ = [
    "SchemaVersionStatus",
    "check_schema_version",
    "classify_schema_version",
    "load_toml_document",
    "require_field",
    "require_sections",
]
