"""#5090: public dataclass field-shape diff gate, and the shared wheel-contents reader."""

import subprocess
import textwrap
from pathlib import Path

import pytest

from xtrax.devtools.gates.dataclass_shape_diff import (
    collect_public_dataclass_fields,
    diff_public_dataclass_shapes,
    run_dataclass_shape_gate,
)
from xtrax.devtools.wheel_contents import load_wheel_contents

ROOT = Path(__file__).resolve().parents[2]

BASE = textwrap.dedent(
    """
    import dataclasses
    from dataclasses import dataclass, field
    from typing import ClassVar, NamedTuple
    import equinox as eqx

    @dataclass(frozen=True)
    class SpirvValidationResult:
        valid: bool
        adapter_type: str
        backend: str = "vulkan"
        _cache: int = 0
        KIND: ClassVar[str] = "spirv"

    @dataclasses.dataclass
    class Plain:
        n: int
        tags: list = field(default_factory=list)
        eager: int = field(repr=False)

    class Pair(NamedTuple):
        a: int
        b: int = 0

    class Strat(eqx.Module):
        batch_size: int = eqx.field(static=True, default=1)

    class NotADataclass:
        x: int

    @dataclass
    class _Private:
        y: int
    """
)


def _changes(head: str) -> list[tuple[str, str, str]]:
    return _changes_between(BASE, head)


def _changes_between(base: str, head: str) -> list[tuple[str, str, str]]:
    return [
        (c.class_name, c.kind, c.detail)
        for c in diff_public_dataclass_shapes(rel_path="m.py", base_source=base, head_source=head)
    ]


class TestCollect:
    def test_dataclass_forms_namedtuple_and_eqx_module_are_collected(self):
        shapes = collect_public_dataclass_fields(BASE)
        assert set(shapes) == {"SpirvValidationResult", "Plain", "Pair", "Strat"}

    def test_private_fields_and_classvars_are_excluded(self):
        fields = collect_public_dataclass_fields(BASE)["SpirvValidationResult"]
        assert list(fields) == ["valid", "adapter_type", "backend"]

    def test_default_detection_understands_field_calls(self):
        plain = collect_public_dataclass_fields(BASE)["Plain"]
        assert plain["tags"].has_default  # default_factory
        assert not plain["eager"].has_default  # field(repr=False) has no default
        assert collect_public_dataclass_fields(BASE)["Strat"]["batch_size"].has_default


class TestDiff:
    def test_unchanged_source_reports_nothing(self):
        assert _changes(BASE) == []

    def test_the_rename_from_the_item_is_caught(self):
        """#5090's own case: adapter_type -> validator on SpirvValidationResult."""
        head = BASE.replace("adapter_type: str", "validator: str")
        assert _changes(head) == [
            ("SpirvValidationResult", "removed-field", "adapter_type"),
            ("SpirvValidationResult", "required-field", "validator"),
        ]

    def test_retyped_field(self):
        head = BASE.replace("    valid: bool", "    valid: int")
        assert _changes(head) == [("SpirvValidationResult", "retyped-field", "valid: bool -> int")]

    def test_lost_default(self):
        head = BASE.replace('backend: str = "vulkan"', "backend: str")
        assert _changes(head) == [("SpirvValidationResult", "lost-default", "backend")]

    def test_new_defaulted_field_is_not_breaking(self):
        head = BASE.replace(
            '    backend: str = "vulkan"\n', '    backend: str = "vulkan"\n    extra: int = 0\n'
        )
        assert _changes(head) == []

    def test_removed_class(self):
        head = BASE.replace("class Pair(NamedTuple):", "class Renamed(NamedTuple):")
        assert ("Pair", "removed-class", "no longer defined here") in _changes(head)

    def test_private_and_non_dataclass_changes_are_ignored(self):
        head = BASE.replace("    x: int\n", "    x: str\n").replace("    y: int\n", "    z: int\n")
        head = head.replace("_cache: int = 0", "_cache: str = ''")
        assert _changes(head) == []

    def test_appending_a_defaulted_field_at_the_end_is_fine_but_inserting_is_not(self):
        appended = BASE.replace(
            '    backend: str = "vulkan"\n', '    backend: str = "vulkan"\n    extra: int = 0\n'
        )
        assert _changes(appended) == []
        inserted = BASE.replace(
            "    adapter_type: str\n", "    extra: int = 0\n    adapter_type: str\n"
        )
        kinds = {kind for _, kind, _ in _changes(inserted)}
        assert "reordered-fields" in kinds

    def test_namedtuple_reorder_is_breaking(self):
        head = BASE.replace(
            "        a: int\n        b: int = 0\n", "        b: int = 0\n        a: int\n"
        )
        head = BASE.replace("    a: int\n    b: int = 0\n", "    b: int = 0\n    a: int\n")
        assert ("Pair", "reordered-fields", "a 0->1, b 1->0") in _changes(head)

    def test_init_false_field_is_an_attribute_not_a_parameter(self):
        """Adding a derived `init=False` field is not a new required parameter."""
        head = BASE.replace(
            "    batch_size: int = eqx.field(static=True, default=1)\n",
            "    batch_size: int = eqx.field(static=True, default=1)\n"
            "    derived: int = eqx.field(init=False)\n",
        )
        assert _changes(head) == []

    def test_removing_an_init_false_field_still_breaks_attribute_access(self):
        base = BASE.replace(
            "    batch_size: int = eqx.field(static=True, default=1)\n",
            "    batch_size: int = eqx.field(static=True, default=1)\n"
            "    derived: int = eqx.field(init=False)\n",
        )
        changes = diff_public_dataclass_shapes(rel_path="m.py", base_source=base, head_source=BASE)
        assert [(c.class_name, c.kind, c.detail) for c in changes] == [
            ("Strat", "removed-field", "derived")
        ]

    @pytest.mark.parametrize(
        ("old", "new"),
        [
            ("backend: str = ", "backend: 'str' = "),
            ("    valid: bool\n", "    valid: Optional[bool]\n"),
        ],
        ids=["quoted-forward-ref", "optional"],
    )
    def test_spelling_only_annotation_rewrites_are_not_changes(self, old, new):
        base = (
            BASE.replace("    valid: bool\n", "    valid: bool | None\n")
            if "Optional" in new
            else BASE
        )
        head = (
            base.replace("    valid: bool | None\n", "    valid: Optional[bool]\n")
            if "Optional" in new
            else BASE.replace(old, new)
        )
        assert _changes_between(base, head) == []

    def test_new_file_cannot_break_a_consumer(self):
        assert (
            diff_public_dataclass_shapes(rel_path="m.py", base_source=None, head_source=BASE) == []
        )


class TestWheelContents:
    def test_real_pyproject_ships_src_but_not_devtools(self):
        wheel = load_wheel_contents(ROOT / "pyproject.toml")
        assert wheel.ships("src/xtrax/export/spirv.py")
        assert not wheel.ships("src/xtrax/devtools/gates/test_rigor.py")
        assert not wheel.ships("scripts/audit_public_api.py")
        assert not wheel.ships("src/xtraxy/other.py")  # prefix of a name is not a parent

    def test_glob_patterns_are_refused_not_guessed(self):
        data = {
            "tool": {
                "hatch": {
                    "build": {
                        "targets": {
                            "wheel": {"packages": ["src/xtrax"], "exclude": ["src/xtrax/**/_*.py"]}
                        }
                    }
                }
            }
        }
        with pytest.raises(ValueError, match="glob"):
            load_wheel_contents(data)

    def test_missing_packages_is_an_error(self):
        with pytest.raises(ValueError, match="no \\[tool.hatch"):
            load_wheel_contents({"tool": {}})


# --- end to end in a throwaway git repo (no dependency on this repo's refs) -------


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, str]:
    (tmp_path / "src" / "pkg" / "devtools").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.hatch.build.targets.wheel]\npackages = ["src/pkg"]\nexclude = ["src/pkg/devtools"]\n'
    )
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n")
    (tmp_path / "src" / "pkg" / "api.py").write_text(BASE)
    (tmp_path / "src" / "pkg" / "devtools" / "tool.py").write_text(BASE)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    _git(tmp_path, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base")
    return tmp_path, _git(tmp_path, "rev-parse", "HEAD")


def _commit(repo: Path, msg: str) -> None:
    _git(repo, "add", ".")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", msg)


def test_gate_fails_on_an_unacknowledged_rename_then_passes_once_changelogged(repo):
    root, base = repo
    api = root / "src" / "pkg" / "api.py"
    api.write_text(BASE.replace("adapter_type: str", "validator: str"))
    _commit(root, "rename")
    result = run_dataclass_shape_gate(root, target=Path("src/pkg"), merge_base=base)
    assert result.status == "fail"
    assert {c.kind for c in result.unacknowledged} == {"removed-field", "required-field"}

    (root / "CHANGELOG.md").write_text(
        "# Changelog\n- `SpirvValidationResult.adapter_type` renamed to `validator`.\n"
    )
    _commit(root, "changelog")
    result = run_dataclass_shape_gate(root, target=Path("src/pkg"), merge_base=base)
    assert result.status == "pass"
    assert len(result.changes) == 2  # still reported, now acknowledged


def test_changelog_ack_is_a_whole_word_match(repo):
    """`Result` is not acknowledged by a line about `ExportResult`."""
    root, base = repo
    api = root / "src" / "pkg" / "api.py"
    api.write_text(BASE + "\n@dataclass\nclass Result:\n    x: int\n")
    _commit(root, "add Result")
    base2 = _git(root, "rev-parse", "HEAD")
    api.write_text(BASE + "\n@dataclass\nclass Result:\n    y: int\n")
    (root / "CHANGELOG.md").write_text("# Changelog\n- `ExportResult` gained a field.\n")
    _commit(root, "rename Result.x")
    assert run_dataclass_shape_gate(root, target=Path("src/pkg"), merge_base=base2).status == "fail"


def test_deleting_a_module_is_a_removal(repo):
    root, base = repo
    _git(root, "rm", "-q", "src/pkg/api.py")
    _commit(root, "delete api")
    result = run_dataclass_shape_gate(root, target=Path("src/pkg"), merge_base=base)
    assert result.status == "fail"
    assert {c.kind for c in result.unacknowledged} == {"removed-class"}


def test_moving_a_module_is_a_removal_from_its_old_path(repo):
    root, base = repo
    _git(root, "mv", "src/pkg/api.py", "src/pkg/api2.py")
    _commit(root, "move api")
    result = run_dataclass_shape_gate(root, target=Path("src/pkg"), merge_base=base)
    assert result.status == "fail"
    assert {c.rel_path for c in result.unacknowledged} == {"src/pkg/api.py"}


def test_package_init_is_diffed(repo):
    root, base = repo
    init = root / "src" / "pkg" / "__init__.py"
    init.write_text(BASE)
    _commit(root, "add init")
    base2 = _git(root, "rev-parse", "HEAD")
    init.write_text(BASE.replace("adapter_type: str", "validator: str"))
    _commit(root, "rename in init")
    assert run_dataclass_shape_gate(root, target=Path("src/pkg"), merge_base=base2).status == "fail"


def test_gate_ignores_files_the_wheel_excludes(repo):
    root, base = repo
    tool = root / "src" / "pkg" / "devtools" / "tool.py"
    tool.write_text(BASE.replace("adapter_type: str", "validator: str"))
    _commit(root, "devtools-only change")
    result = run_dataclass_shape_gate(root, target=Path("src/pkg"), merge_base=base)
    assert result.status == "pass"
    assert result.changes == ()
