"""Tests for xtrax.config's domain-agnostic TOML-config primitives (idea-003).

Spec: `.praxia/docs/specs/260715_generic-fail-loud-toml-to-dataclass-conf.md`.
See tests/cli/test_config.py for the dog-fooded TrainConfig/load_config usage.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from xtrax.config import (
    SchemaVersionStatus,
    check_schema_version,
    classify_schema_version,
    load_toml_document,
    require_field,
    require_sections,
)


class LocalConfigError(Exception):
    """A caller-supplied error_cls, standing in for a downstream consumer's own type."""


def _write_toml(tmp_path, content: str) -> str:
    p = tmp_path / "config.toml"
    p.write_text(content)
    return str(p)


# ---------------------------------------------------------------------------
# load_toml_document
# ---------------------------------------------------------------------------


def test_load_toml_document_parses(tmp_path) -> None:
    p = _write_toml(tmp_path, "a = 1\nb = 2\n")
    raw = load_toml_document(p, LocalConfigError)
    assert raw == {"a": 1, "b": 2}


def test_load_toml_document_missing_file_raises(tmp_path) -> None:
    with pytest.raises(LocalConfigError, match="cannot read"):
        load_toml_document(str(tmp_path / "does_not_exist.toml"), LocalConfigError)


def test_load_toml_document_malformed_toml_raises(tmp_path) -> None:
    p = _write_toml(tmp_path, "this is not [valid toml")
    with pytest.raises(LocalConfigError, match="malformed TOML"):
        load_toml_document(p, LocalConfigError)


# ---------------------------------------------------------------------------
# require_sections
# ---------------------------------------------------------------------------


def test_require_sections_passes_when_all_present() -> None:
    require_sections({"a": 1, "b": 2}, ("a", "b"), LocalConfigError)  # no raise


def test_require_sections_names_single_missing() -> None:
    with pytest.raises(LocalConfigError, match=r"\[b\]"):
        require_sections({"a": 1}, ("a", "b"), LocalConfigError)


def test_require_sections_names_all_missing_not_just_first() -> None:
    with pytest.raises(LocalConfigError) as exc_info:
        require_sections({}, ("a", "b", "c"), LocalConfigError)
    message = str(exc_info.value)
    assert "[a]" in message
    assert "[b]" in message
    assert "[c]" in message


# ---------------------------------------------------------------------------
# require_field
# ---------------------------------------------------------------------------


def test_require_field_returns_value_on_success() -> None:
    value = require_field({"num_epochs": 3}, "num_epochs", lambda v: v > 0, LocalConfigError)
    assert value == 3


def test_require_field_names_field_and_value_on_failure() -> None:
    with pytest.raises(LocalConfigError, match=r"'num_epochs'.*-1"):
        require_field(
            {"num_epochs": -1},
            "num_epochs",
            lambda v: isinstance(v, int) and v > 0,
            LocalConfigError,
        )


def test_require_field_missing_field_fails_predicate() -> None:
    with pytest.raises(LocalConfigError, match="seed"):
        require_field({}, "seed", lambda v: isinstance(v, int), LocalConfigError)


# ---------------------------------------------------------------------------
# classify_schema_version / check_schema_version
# ---------------------------------------------------------------------------


def test_classify_schema_version_ok() -> None:
    status = classify_schema_version({"schema_version": 1}, current=1)
    assert status == SchemaVersionStatus(kind="ok", found=1, current=1)


def test_classify_schema_version_missing() -> None:
    status = classify_schema_version({}, current=1)
    assert status.kind == "missing"


def test_classify_schema_version_mismatched_older() -> None:
    status = classify_schema_version({"schema_version": 0}, current=1)
    assert status.kind == "mismatched"
    assert status.found == 0


def test_classify_schema_version_newer_than_supported() -> None:
    status = classify_schema_version({"schema_version": 5}, current=1)
    assert status.kind == "newer_than_supported"
    assert status.found == 5


def test_check_schema_version_raises_on_missing() -> None:
    with pytest.raises(LocalConfigError, match="schema_version"):
        check_schema_version({}, current=1, error_cls=LocalConfigError)


def test_check_schema_version_raises_on_mismatch() -> None:
    with pytest.raises(LocalConfigError, match="mismatch"):
        check_schema_version({"schema_version": 0}, current=1, error_cls=LocalConfigError)


def test_check_schema_version_raises_on_newer_than_supported() -> None:
    with pytest.raises(LocalConfigError, match="newer"):
        check_schema_version({"schema_version": 5}, current=1, error_cls=LocalConfigError)


def test_check_schema_version_passes_when_ok() -> None:
    check_schema_version({"schema_version": 1}, current=1, error_cls=LocalConfigError)  # no raise


# ---------------------------------------------------------------------------
# AC8 -- proof of generality: a non-training config shape, not TrainConfig
# ---------------------------------------------------------------------------


@dataclass
class ToyInferConfig:
    """A stand-in for a non-training (inference-shaped) consumer config --
    no optimizer/loss/num_epochs, unlike xtrax.cli.config.TrainConfig.
    """

    schema_version: int
    model: dict
    checkpoint: dict


def _load_toy_infer_config(path: str) -> ToyInferConfig:
    """A downstream-style loader composed entirely from xtrax.config primitives."""
    raw = load_toml_document(path, LocalConfigError)
    check_schema_version(raw, current=1, error_cls=LocalConfigError)
    require_sections(raw, ("model", "checkpoint"), LocalConfigError)
    return ToyInferConfig(
        schema_version=raw["schema_version"], model=raw["model"], checkpoint=raw["checkpoint"]
    )


def test_ac8_primitives_serve_a_non_training_config_shape(tmp_path) -> None:
    """The primitives work for a domain shape TrainConfig doesn't have
    (no optimizer/loss/num_epochs) -- proves the extraction is genuinely
    schema-agnostic, not just TrainConfig restated.
    """
    p = _write_toml(
        tmp_path,
        """
        schema_version = 1

        [model]
        path = "mypkg.models:make_model"

        [checkpoint]
        path = "/tmp/ckpt.orbax"
        """,
    )
    cfg = _load_toy_infer_config(p)
    assert cfg.schema_version == 1
    assert cfg.model["path"] == "mypkg.models:make_model"
    assert cfg.checkpoint["path"] == "/tmp/ckpt.orbax"


def test_ac8_toy_infer_config_missing_checkpoint_section_raises(tmp_path) -> None:
    p = _write_toml(
        tmp_path,
        """
        schema_version = 1

        [model]
        path = "mypkg.models:make_model"
        """,
    )
    with pytest.raises(LocalConfigError, match=r"\[checkpoint\]"):
        _load_toy_infer_config(p)
