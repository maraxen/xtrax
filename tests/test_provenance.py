"""Provenance record, source reader, and declaration renderer (#2590, #2592)."""

import textwrap
from pathlib import Path

import pytest

from xtrax.provenance import (
    Provenance,
    collect_provenance,
    read_provenance,
    render_declaration,
)


def _complete(**overrides: object) -> Provenance:
    fields: dict[str, object] = {
        "upstream": "https://example.com/up",
        "relationship": "ported",
        "revision": "v1.2.3",
        "licence": "Apache-2.0",
    }
    fields.update(overrides)
    return Provenance(**fields)  # type: ignore[arg-type]


def test_provenance_constructs_with_sha_or_tag_and_spdx_id() -> None:
    record = _complete(revision="0123456789abcdef", relationship="vendored")
    assert record.upstream == "https://example.com/up"
    assert record.revision == "0123456789abcdef"
    assert record.licence == "Apache-2.0"
    assert record.relationship == "vendored"
    assert record.waiver_reason is None
    tagged = _complete(relationship="derived", revision="v0.1.0", licence="BSD-3-Clause")
    assert tagged.revision == "v0.1.0"
    assert tagged.licence == "BSD-3-Clause"


def test_missing_upstream_raises() -> None:
    with pytest.raises(ValueError, match="upstream is required"):
        _complete(upstream="  ")


def test_missing_relationship_raises() -> None:
    with pytest.raises(TypeError):
        Provenance(upstream="prolix", revision="v1", licence="MIT")  # type: ignore[call-arg]


def test_unknown_relationship_raises() -> None:
    with pytest.raises(ValueError, match="got 'forked'"):
        _complete(relationship="forked")


def test_blank_relationship_raises() -> None:
    with pytest.raises(ValueError, match="relationship must be one of"):
        _complete(relationship="")


def test_missing_revision_without_waiver_raises() -> None:
    with pytest.raises(ValueError, match="revision is required unless waiver_reason is set"):
        _complete(revision=None)


def test_missing_licence_without_waiver_raises() -> None:
    with pytest.raises(ValueError, match="licence is required unless waiver_reason is set"):
        _complete(licence=None)


def test_missing_revision_and_licence_without_waiver_raises() -> None:
    with pytest.raises(
        ValueError, match="revision and licence are required unless waiver_reason is set"
    ):
        _complete(revision=None, licence=None)


def test_blank_waiver_does_not_excuse_missing_fields() -> None:
    with pytest.raises(ValueError, match="waiver_reason"):
        _complete(revision=None, licence=None, waiver_reason="   ")


def test_waiver_stores_the_reason_and_leaves_unpinned_fields_empty() -> None:
    record = Provenance(
        upstream="prolix",
        relationship="ported",
        waiver_reason="branch wt-20260807-132628 is not a commit sha or tag",
    )
    assert record.revision is None
    assert record.licence is None
    assert record.waiver_reason == "branch wt-20260807-132628 is not a commit sha or tag"


def test_waiver_can_cover_only_the_revision() -> None:
    record = Provenance(
        upstream="prolix",
        relationship="inspired",
        licence="MIT",
        waiver_reason="revision is unpinnable",
    )
    assert record.revision is None
    assert record.licence == "MIT"
    assert record.waiver_reason == "revision is unpinnable"


def test_complete_record_may_also_carry_a_waiver_reason() -> None:
    record = _complete(waiver_reason="tag v1.2.3 is the upstream release, not a sha")
    assert record.revision == "v1.2.3"
    assert record.waiver_reason == "tag v1.2.3 is the upstream release, not a sha"


def test_non_spdx_licence_raises() -> None:
    with pytest.raises(ValueError, match="licence must be an SPDX id, got 'not a licence'"):
        _complete(licence="not a licence")


def test_reader_round_trips_a_provenance_call(tmp_path: Path) -> None:
    package = tmp_path / "xtrax"
    package.mkdir()
    (package / "leaf.py").write_text(
        textwrap.dedent(
            """\
            __provenance__ = Provenance(
                upstream="https://example.com/up",
                relationship="ported",
                revision="v1.2.3",
                licence="Apache-2.0",
            )
            """
        ),
        encoding="utf-8",
    )
    found = collect_provenance(package)
    assert list(found) == ["xtrax.leaf"]
    assert found["xtrax.leaf"] == _complete()


def test_reader_round_trips_a_dict_declaration(tmp_path: Path) -> None:
    package = tmp_path / "xtrax"
    nested = package / "profiling"
    nested.mkdir(parents=True)
    (nested / "__init__.py").write_text(
        textwrap.dedent(
            """\
            __provenance__ = {
                "upstream": "prolix",
                "relationship": "derived",
                "waiver_reason": "no sha and no SPDX id in this repo",
            }
            """
        ),
        encoding="utf-8",
    )
    found = collect_provenance(package)
    assert list(found) == ["xtrax.profiling"]
    assert found["xtrax.profiling"] == Provenance(
        upstream="prolix",
        relationship="derived",
        waiver_reason="no sha and no SPDX id in this repo",
    )


def test_reader_ignores_modules_without_a_declaration(tmp_path: Path) -> None:
    package = tmp_path / "xtrax"
    package.mkdir()
    (package / "plain.py").write_text("x = 1\n", encoding="utf-8")
    assert collect_provenance(package) == {}
    assert read_provenance(package / "plain.py") is None


def test_reader_rejects_non_literal_fields(tmp_path: Path) -> None:
    path = tmp_path / "leaf.py"
    path.write_text(
        '__provenance__ = Provenance(upstream="up", relationship=name)\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="string literals or None"):
        read_provenance(path)


def test_reader_rejects_unknown_dict_fields(tmp_path: Path) -> None:
    path = tmp_path / "leaf.py"
    path.write_text(
        '__provenance__ = {"upstream": "up", "relationship": "ported", "sha": "abc"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown provenance fields"):
        read_provenance(path)


def test_reader_rejects_duplicate_declarations(tmp_path: Path) -> None:
    path = tmp_path / "leaf.py"
    path.write_text(
        '__provenance__ = {"upstream": "up"}\n__provenance__ = {"upstream": "other"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="multiple __provenance__"):
        read_provenance(path)


def test_reader_accepts_a_qualified_provenance_call(tmp_path: Path) -> None:
    path = tmp_path / "leaf.py"
    path.write_text(
        textwrap.dedent(
            """\
            __provenance__ = xtrax.provenance.Provenance(
                upstream="https://example.com/up",
                relationship="ported",
                revision="v1",
                licence="MIT",
            )
            """
        ),
        encoding="utf-8",
    )
    record = read_provenance(path)
    assert record == Provenance(
        upstream="https://example.com/up",
        relationship="ported",
        revision="v1",
        licence="MIT",
    )


def test_reader_rejects_a_different_constructor(tmp_path: Path) -> None:
    path = tmp_path / "leaf.py"
    path.write_text('__provenance__ = Other(upstream="up")\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Provenance\\(\\.\\.\\.\\) call or a dict"):
        read_provenance(path)


def test_reader_rejects_positional_arguments(tmp_path: Path) -> None:
    path = tmp_path / "leaf.py"
    path.write_text('__provenance__ = Provenance("up", "ported")\n', encoding="utf-8")
    with pytest.raises(ValueError, match="keywords"):
        read_provenance(path)


def test_reader_rejects_starred_keywords(tmp_path: Path) -> None:
    path = tmp_path / "leaf.py"
    path.write_text("__provenance__ = Provenance(**fields)\n", encoding="utf-8")
    with pytest.raises(ValueError, match="keywords"):
        read_provenance(path)


def test_reader_rejects_a_non_string_dict_key(tmp_path: Path) -> None:
    path = tmp_path / "leaf.py"
    path.write_text("__provenance__ = {1: 'up'}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="string literals or None"):
        read_provenance(path)


def test_collect_skips_pycache(tmp_path: Path) -> None:
    package = tmp_path / "xtrax"
    cache = package / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "cached.py").write_text(
        '__provenance__ = {"upstream": "skip", "relationship": "ported", '
        '"revision": "v1", "licence": "MIT"}\n',
        encoding="utf-8",
    )
    assert collect_provenance(package) == {}


def test_reader_rejects_a_value_that_is_not_a_declaration(tmp_path: Path) -> None:
    path = tmp_path / "leaf.py"
    path.write_text("__provenance__ = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Provenance\\(\\.\\.\\.\\) call or a dict"):
        read_provenance(path)


def test_scaffold_emits_a_declaration_the_reader_accepts(tmp_path: Path) -> None:
    text = render_declaration(
        upstream="https://example.com/up",
        relationship="vendored",
        revision="0123456789abcdef",
        licence="MIT",
    )
    package = tmp_path / "xtrax"
    package.mkdir()
    (package / "leaf.py").write_text(text, encoding="utf-8")
    found = collect_provenance(package)
    assert found["xtrax.leaf"] == Provenance(
        upstream="https://example.com/up",
        relationship="vendored",
        revision="0123456789abcdef",
        licence="MIT",
    )
    dict_text = render_declaration(
        upstream="prolix",
        relationship="ported",
        waiver_reason="no sha and no SPDX id",
        form="dict",
    )
    assert "import" not in dict_text
    (package / "leaf.py").write_text(dict_text, encoding="utf-8")
    assert collect_provenance(package)["xtrax.leaf"].waiver_reason == "no sha and no SPDX id"


def test_scaffold_refuses_a_waiver_without_a_reason() -> None:
    with pytest.raises(ValueError, match="waiver_reason"):
        render_declaration(upstream="prolix", relationship="ported", revision="v1")
    with pytest.raises(ValueError, match="waiver_reason"):
        render_declaration(upstream="prolix", relationship="ported", waiver_reason="  ")
