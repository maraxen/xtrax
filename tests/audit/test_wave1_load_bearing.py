"""Tests for T2-02's wave-1 load-bearing gate (#3070)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts.audit_wave1_load_bearing import (
    ROOT,
    audit_wave1_load_bearing,
    check_certifying_tests_green,
    check_importable,
    check_no_stub_markers,
)


def _write_fixture_module(tmp_path: Path, name: str, body: str) -> None:
    (tmp_path / f"{name}.py").write_text(body, encoding="utf-8")


class TestCheckImportable:
    def test_clean_module_with_real_symbols_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_fixture_module(tmp_path, "clean_mod", "def real_fn(x):\n    return x + 1\n")
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))

        hits = check_importable(["clean_mod"])

        assert hits == []

    def test_import_failure_fails_naming_the_module(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_fixture_module(tmp_path, "broken_mod", "this is not valid python (((\n")
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))

        hits = check_importable(["broken_mod"])

        assert len(hits) == 1
        assert "broken_mod" in hits[0]


class TestCheckNoStubMarkers:
    def test_missing_symbol_fails_naming_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_fixture_module(tmp_path, "missing_symbol_mod", "def real_fn(x):\n    return x\n")
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))
        monkeypatch.syspath_prepend(str(tmp_path))
        sys.modules.pop("missing_symbol_mod", None)

        hits = check_no_stub_markers({"missing_symbol_mod": ("does_not_exist",)})

        assert len(hits) == 1
        assert "missing_symbol_mod.does_not_exist" in hits[0]
        assert "not found" in hits[0]

    def test_notimplementederror_stub_body_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_fixture_module(
            tmp_path, "stub_raise_mod", "def stub_fn(x):\n    raise NotImplementedError\n"
        )
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))
        monkeypatch.syspath_prepend(str(tmp_path))
        sys.modules.pop("stub_raise_mod", None)

        hits = check_no_stub_markers({"stub_raise_mod": ("stub_fn",)})

        assert len(hits) == 1
        assert "stub_raise_mod.stub_fn" in hits[0]

    def test_ellipsis_stub_body_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_fixture_module(tmp_path, "stub_ellipsis_mod", "def stub_fn(x):\n    ...\n")
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))
        monkeypatch.syspath_prepend(str(tmp_path))
        sys.modules.pop("stub_ellipsis_mod", None)

        hits = check_no_stub_markers({"stub_ellipsis_mod": ("stub_fn",)})

        assert len(hits) == 1
        assert "stub_ellipsis_mod.stub_fn" in hits[0]

    def test_bare_pass_stub_body_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_fixture_module(tmp_path, "stub_pass_mod", "def stub_fn(x):\n    pass\n")
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))
        monkeypatch.syspath_prepend(str(tmp_path))
        sys.modules.pop("stub_pass_mod", None)

        hits = check_no_stub_markers({"stub_pass_mod": ("stub_fn",)})

        assert len(hits) == 1
        assert "stub_pass_mod.stub_fn" in hits[0]

    def test_real_body_with_docstring_passes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_fixture_module(
            tmp_path,
            "real_mod",
            'def real_fn(x):\n    """A real docstring."""\n    return x * 2\n',
        )
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))
        monkeypatch.syspath_prepend(str(tmp_path))
        sys.modules.pop("real_mod", None)

        hits = check_no_stub_markers({"real_mod": ("real_fn",)})

        assert hits == []

    def test_import_failure_fails_naming_the_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hits = check_no_stub_markers({"xtrax.this_module_does_not_exist": ("anything",)})

        assert len(hits) == 1
        assert "xtrax.this_module_does_not_exist" in hits[0]


class TestCheckCertifyingTestsGreen:
    def test_passing_test_file_produces_no_hits(self, tmp_path: Path) -> None:
        (tmp_path / "test_fake_pass.py").write_text(
            "def test_ok():\n    assert True\n", encoding="utf-8"
        )

        hits = check_certifying_tests_green(["test_fake_pass.py"], tmp_path)

        assert hits == []

    def test_certifying_test_failure_fails_naming_the_file(self, tmp_path: Path) -> None:
        (tmp_path / "test_fake_fail.py").write_text(
            "def test_bad():\n    assert False\n", encoding="utf-8"
        )

        hits = check_certifying_tests_green(["test_fake_fail.py"], tmp_path)

        assert len(hits) == 1
        assert "test_fake_fail.py" in hits[0]


def test_real_repo_wave1_deliverables_pass() -> None:
    """Guards the actual T1-04/05/07/08/11 deliverables the gate protects, not just fixtures.

    This is the test that proves those deliverables are load-bearing today, not merely that
    the gate's own mechanics work against synthetic fixtures.
    """
    passed, hits = audit_wave1_load_bearing(ROOT)

    assert passed, f"unexpected wave-1 load-bearing gate hits: {hits}"
