"""Public dataclass field-shape diff gate (#5090).

`added_types_diff` audits the *annotations* of new or re-signatured public callables;
it never looks at a dataclass's fields, and `audit_public_api` only reads the root
`__all__`. So a public frozen dataclass that is part of a consumer-facing return type
(`ExportResult`, `SpirvValidationResult`, ...) could rename, drop or retype a field --
breaking every consumer's attribute access -- with no gate firing and no CHANGELOG
entry forced.

This gate diffs the field set of every public dataclass-like class in each changed
file against the merge-base. A consumer-breaking change is:

- a field removed or renamed (a rename is a removal plus an addition),
- a field's annotation changed,
- a field losing its default, or a new field without one (breaks construction),
- the class itself removed from a file that still exists.

Each one must be acknowledged: the class name has to appear on a line ADDED to
`CHANGELOG.md` since the merge-base. Adding a defaulted field is not breaking and is
not reported.

Only files that ship in the wheel are considered (read from pyproject.toml's hatch
wheel target), since a consumer cannot import anything else -- `src/xtrax/devtools`
is excluded there.

"Dataclass-like" is decided syntactically: decorated with `dataclass` (bare, called,
or `dataclasses.dataclass`), or subclassing `NamedTuple` or an Equinox `Module`.
Fields are the class body's annotated names, excluding private names and `ClassVar`.
"""

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from xtrax.devtools.gates.added_types_diff import (
    DEFAULT_TARGET,
    _git_run,
    git_show_file,
    list_changed_python_files,
    resolve_merge_base,
)
from xtrax.devtools.wheel_contents import load_wheel_contents

GateStatus = Literal["pass", "fail", "skip"]
CHANGELOG = "CHANGELOG.md"

_DATACLASS_DECORATORS = frozenset({"dataclass", "dataclasses.dataclass"})
_DATACLASS_BASES = frozenset({"NamedTuple", "typing.NamedTuple", "Module", "eqx.Module"})
_FIELD_CALLS = frozenset({"field", "dataclasses.field", "eqx.field", "equinox.field"})


@dataclass(frozen=True, slots=True)
class FieldShape:
    annotation: str
    has_default: bool


@dataclass(frozen=True, slots=True)
class ShapeChange:
    rel_path: str
    class_name: str
    kind: str  # removed-field | retyped-field | lost-default | required-field | removed-class
    detail: str

    def render(self) -> str:
        return f"{self.rel_path}: {self.class_name}: {self.kind}: {self.detail}"


@dataclass(frozen=True, slots=True)
class ShapeGateResult:
    status: GateStatus
    merge_base: str | None
    skip_reason: str | None
    files_checked: int
    changes: tuple[ShapeChange, ...]
    unacknowledged: tuple[ShapeChange, ...]


def _dotted(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    return ""


def _is_dataclass_like(cls: ast.ClassDef) -> bool:
    for deco in cls.decorator_list:
        target = deco.func if isinstance(deco, ast.Call) else deco
        if _dotted(target) in _DATACLASS_DECORATORS:
            return True
    return any(_dotted(base) in _DATACLASS_BASES for base in cls.bases)


def _has_default(value: ast.expr | None) -> bool:
    if value is None:
        return False
    if isinstance(value, ast.Call) and _dotted(value.func) in _FIELD_CALLS:
        return any(kw.arg in {"default", "default_factory"} for kw in value.keywords)
    return True


def collect_public_dataclass_fields(source: str) -> dict[str, dict[str, FieldShape]]:
    """Top-level public dataclass-like classes -> {field name: shape}, in body order."""
    tree = ast.parse(source)
    shapes: dict[str, dict[str, FieldShape]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
            continue
        if not _is_dataclass_like(node):
            continue
        fields: dict[str, FieldShape] = {}
        for stmt in node.body:
            if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
                continue
            name = stmt.target.id
            annotation = ast.unparse(stmt.annotation)
            if name.startswith("_") or annotation.split("[", 1)[0].endswith("ClassVar"):
                continue
            fields[name] = FieldShape(annotation=annotation, has_default=_has_default(stmt.value))
        shapes[node.name] = fields
    return shapes


def diff_public_dataclass_shapes(
    *, rel_path: str, base_source: str | None, head_source: str
) -> list[ShapeChange]:
    """Consumer-breaking field-shape changes between two versions of one file."""
    if base_source is None:  # a new file cannot break an existing consumer
        return []
    base = collect_public_dataclass_fields(base_source)
    head = collect_public_dataclass_fields(head_source)
    changes: list[ShapeChange] = []
    for cls, old_fields in base.items():
        if cls not in head:
            changes.append(ShapeChange(rel_path, cls, "removed-class", "no longer defined here"))
            continue
        new_fields = head[cls]
        for name, old in old_fields.items():
            new = new_fields.get(name)
            if new is None:
                changes.append(ShapeChange(rel_path, cls, "removed-field", name))
            elif new.annotation != old.annotation:
                changes.append(
                    ShapeChange(
                        rel_path,
                        cls,
                        "retyped-field",
                        f"{name}: {old.annotation} -> {new.annotation}",
                    )
                )
            elif old.has_default and not new.has_default:
                changes.append(ShapeChange(rel_path, cls, "lost-default", name))
        for name, new in new_fields.items():
            if name not in old_fields and not new.has_default:
                changes.append(ShapeChange(rel_path, cls, "required-field", name))
    return changes


def changelog_added_text(repo_root: Path, merge_base: str) -> str:
    """Lines added to CHANGELOG.md since the merge-base (empty if none or no file)."""
    result = _git_run(repo_root, "diff", "--unified=0", f"{merge_base}..HEAD", "--", CHANGELOG)
    if result.returncode != 0:
        return ""
    return "\n".join(
        line[1:]
        for line in result.stdout.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def acknowledged(change: ShapeChange, changelog_added: str) -> bool:
    return change.class_name in changelog_added


def run_dataclass_shape_gate(
    repo_root: Path,
    *,
    target: Path = DEFAULT_TARGET,
    merge_base: str | None = None,
) -> ShapeGateResult:
    """Diff public dataclass shapes against the merge-base; skip loudly without one."""
    repo_root = repo_root.resolve()
    target_path = (repo_root / target).resolve() if not target.is_absolute() else target
    base = merge_base or resolve_merge_base(repo_root)
    if base is None:
        return ShapeGateResult(
            status="skip",
            merge_base=None,
            skip_reason="merge-base unavailable (shallow clone?); configure fetch-depth:0 in CI",
            files_checked=0,
            changes=(),
            unacknowledged=(),
        )

    wheel = load_wheel_contents(repo_root / "pyproject.toml")
    changes: list[ShapeChange] = []
    files = [
        path
        for path in list_changed_python_files(repo_root, base, target=target_path)
        if wheel.ships(path.relative_to(repo_root).as_posix())
    ]
    for path in files:
        rel = path.relative_to(repo_root).as_posix()
        changes.extend(
            diff_public_dataclass_shapes(
                rel_path=rel,
                base_source=git_show_file(repo_root, base, rel),
                head_source=path.read_text(encoding="utf-8"),
            )
        )

    added = changelog_added_text(repo_root, base)
    unacked = tuple(c for c in changes if not acknowledged(c, added))
    return ShapeGateResult(
        status="fail" if unacked else "pass",
        merge_base=base,
        skip_reason=None,
        files_checked=len(files),
        changes=tuple(changes),
        unacknowledged=unacked,
    )
