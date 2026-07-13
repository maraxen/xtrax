"""Tests for T1-09 IR-schema single-source grep-gate (#3064)."""

from __future__ import annotations

from pathlib import Path

from scripts.audit_ir_schema_single_source import (
    audit_ir_schema_single_source,
    find_schema_definitions,
)


def test_clean_src_dir_passes(tmp_path: Path) -> None:
    src_dir = tmp_path / "xtrax"
    src_dir.mkdir()
    (src_dir / "harmless.py").write_text("x = 1\n", encoding="utf-8")

    allowed = src_dir / "inference" / "ir_schema.py"
    passed, hits = audit_ir_schema_single_source(src_dir, allowed)

    assert passed
    assert hits == []


def test_second_schema_definition_fails_naming_the_file(tmp_path: Path) -> None:
    src_dir = tmp_path / "xtrax"
    (src_dir / "inference").mkdir(parents=True)
    allowed = src_dir / "inference" / "ir_schema.py"
    allowed.write_text('SCHEMA = {"$schema": "..."}\n', encoding="utf-8")

    rogue = src_dir / "cli" / "hand_authored_schema.py"
    rogue.parent.mkdir(parents=True)
    rogue.write_text('SCHEMA = {"$schema": "a second, competing definition"}\n', encoding="utf-8")

    passed, hits = audit_ir_schema_single_source(src_dir, allowed)

    assert not passed
    assert len(hits) == 1
    assert str(rogue) in hits[0]
    assert hits[0].endswith(":1")


def test_missing_src_dir_passes_vacuously(tmp_path: Path) -> None:
    passed, hits = audit_ir_schema_single_source(
        tmp_path / "does-not-exist", tmp_path / "does-not-exist" / "ir_schema.py"
    )

    assert passed
    assert hits == []


def test_find_schema_definitions_ignores_prose_files(tmp_path: Path) -> None:
    src_dir = tmp_path / "xtrax"
    src_dir.mkdir()
    (src_dir / "notes.md").write_text(
        'mentions "$schema" but is prose, not code/data\n', encoding="utf-8"
    )

    hits = find_schema_definitions(src_dir)

    assert hits == []


def test_find_schema_definitions_scans_toml_and_json_data_files(tmp_path: Path) -> None:
    """xtrax.composition already ships a hand-authored non-.py schema-like file
    (node_metadata_schema.toml) -- a Python-only scan would miss the exact kind of parallel
    schema this gate exists to catch.
    """
    src_dir = tmp_path / "xtrax"
    (src_dir / "composition").mkdir(parents=True)
    rogue_toml = src_dir / "composition" / "hand_authored.toml"
    rogue_toml.write_text('schema = { "$schema" = "a second definition" }\n', encoding="utf-8")
    rogue_json = src_dir / "composition" / "hand_authored.json"
    rogue_json.write_text('{"$schema": "a second definition"}\n', encoding="utf-8")

    hits = find_schema_definitions(src_dir)

    assert any(str(rogue_toml) in hit for hit in hits)
    assert any(str(rogue_json) in hit for hit in hits)


def test_find_schema_definitions_catches_single_quote_spelling(tmp_path: Path) -> None:
    src_dir = tmp_path / "xtrax"
    src_dir.mkdir()
    rogue = src_dir / "rogue.py"
    rogue.write_text(
        "SCHEMA = {'$schema': 'a second, single-quoted definition'}\n", encoding="utf-8"
    )

    hits = find_schema_definitions(src_dir)

    assert any(str(rogue) in hit for hit in hits)


def test_real_src_tree_is_clean() -> None:
    """Guards the actual repo src/xtrax tree the gate protects, not just fixtures."""
    real_src_dir = Path(__file__).resolve().parents[2] / "src" / "xtrax"
    allowed = real_src_dir / "inference" / "ir_schema.py"

    passed, hits = audit_ir_schema_single_source(real_src_dir, allowed)

    assert passed, f"unexpected '\"$schema\"' hits outside {allowed}: {hits}"
