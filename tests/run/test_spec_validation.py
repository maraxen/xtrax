"""Plain-Python validation backstops on SinkSpec and RunSpec.

The test env's beartype import hook rejects a wrongly typed constructor argument
before these backstops run, so type-level cases call the backstop directly on an
otherwise valid object -- the path production callers (no hook) actually take.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from xtrax.run.sink import GitProvenance, SinkSpec
from xtrax.run.spec import RunSpec

_PROV = {"git_sha": "abc", "git_branch": "main", "git_dirty": False}


class TestSinkSpecValidation:
    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"seed": True}, TypeError, "seed must be int"),
            ({"provenance": {"git_sha": 1, "git_branch": "m"}}, TypeError, r"\['git_sha'\]"),
            ({"provenance": {"git_sha": "a"}}, TypeError, r"\['git_branch'\]"),
            (
                {"provenance": {"git_sha": "a", "git_branch": "m", "git_dirty": "no"}},
                TypeError,
                "git_dirty",
            ),
            ({"level_schemas": {-1: {}}}, ValueError, "non-negative ints"),
            ({"level_schemas": {True: {}}}, ValueError, "non-negative ints"),
        ],
    )
    def test_bad_fields_are_rejected(
        self, kwargs: dict[str, object], error: type[Exception], match: str
    ) -> None:
        with pytest.raises(error, match=match):
            SinkSpec(run_id="r", format="memory", **kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("run_id", 5, "run_id must be str"),
            ("seed", 1.5, "seed must be int"),
            ("append", 1, "append must be bool"),
            ("provenance", "abc", "provenance must be"),
            ("level_schemas", {0: ["x"]}, "must be a mapping"),
        ],
    )
    def test_backstop_rejects_wrong_types(self, field: str, value: object, match: str) -> None:
        spec = SinkSpec(run_id="r", format="memory")
        setattr(spec, field, value)
        with pytest.raises(TypeError, match=match):
            spec.__post_init__()

    def test_append_with_create_or_join_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="incompatible"):
            SinkSpec(
                run_id="r",
                format="zarr",
                open_mode="create_or_join",
                store_identity={"k": 1},
                append=True,
            )

    @pytest.mark.parametrize(
        "provenance",
        [None, _PROV, GitProvenance(git_sha="a", git_branch="m", git_dirty=True), Path(".")],
        ids=["none", "mapping", "git-provenance", "path"],
    )
    def test_accepted_provenance_forms(self, provenance: object) -> None:
        spec = SinkSpec(run_id="r", format="memory", provenance=provenance)  # type: ignore[arg-type]
        assert spec.provenance is provenance

    def test_level_schemas_are_normalized_to_plain_dicts(self) -> None:
        schema = {"type": "object"}
        spec = SinkSpec(run_id="r", format="memory", level_schemas={0: schema})
        assert spec.level_schemas == {0: schema}
        assert spec.level_schemas[0] is not schema


class TestRunSpecValidation:
    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"device_count": 0}, ValueError, "device_count must be a positive int"),
            ({"device_count": True}, ValueError, "device_count must be a positive int"),
        ],
    )
    def test_bad_fields_are_rejected(
        self, kwargs: dict[str, object], error: type[Exception], match: str
    ) -> None:
        with pytest.raises(error, match=match):
            RunSpec(seed=0, axes=[], **kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("precision", 32, "precision must be a str"),
            ("output_root", "out", "output_root must be a pathlib.Path"),
            ("shard_lineage", ["a"], "shard_lineage must be a tuple"),
            ("shard_lineage", ("a", 1), "shard_lineage must be a tuple"),
        ],
    )
    def test_backstop_rejects_wrong_types(self, field: str, value: object, match: str) -> None:
        fields = {"device_count": None, "precision": None, "output_root": None}
        fields["shard_lineage"] = None
        fields[field] = value
        with pytest.raises(TypeError, match=match):
            RunSpec.__check_init__(SimpleNamespace(**fields))  # type: ignore[arg-type]

    def test_valid_execution_settings_are_kept(self) -> None:
        spec = RunSpec(
            seed=0,
            axes=[],
            output_root=Path("out"),
            device_count=2,
            precision="bf16",
            shard_lineage=("root", "s1"),
        )
        assert spec.device_count == 2
        assert spec.shard_lineage == ("root", "s1")

    def test_from_spec_is_identity(self) -> None:
        spec = RunSpec(seed=0, axes=[])
        assert RunSpec.from_spec(spec) is spec
