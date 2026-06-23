"""Tests for xtrax.cli.loader module.

Tests the load_fn() function which lazily resolves import-path strings
like 'json:dumps' into actual callables.
"""

from __future__ import annotations

import sys

import pytest

from xtrax.cli.errors import CLIImportError
from xtrax.cli.loader import load_fn


class TestLoadFnValid:
    """Tests for valid import paths."""

    def test_valid_stdlib_function(self) -> None:
        """load_fn resolves a valid stdlib function path."""
        # Use json.dumps from stdlib (no external fixture needed)
        fn = load_fn("json:dumps")

        # Verify it's the real function
        import json
        assert fn is json.dumps

        # Verify it actually works
        result = fn({"key": "value"})
        assert isinstance(result, str)

    def test_valid_class(self) -> None:
        """load_fn can resolve class paths too."""
        fn = load_fn("pathlib:Path")

        # Verify it's the real Path class
        from pathlib import Path
        assert fn is Path


class TestLoadFnLazyImport:
    """Tests that loader.py doesn't eagerly import target modules."""

    def test_import_xtrax_cli_loader_does_not_import_wave(self) -> None:
        """Importing xtrax.cli.loader should not import 'wave' module.

        This verifies lazy loading: xtrax.cli.loader must not import
        user modules at module-load time, only when load_fn() is called.
        """
        # Remove 'wave' if it's already loaded
        if 'wave' in sys.modules:
            del sys.modules['wave']

        # Import loader (should not import 'wave')
        import xtrax.cli.loader  # noqa: F401

        # Verify 'wave' is still not imported
        assert 'wave' not in sys.modules, (
            "Importing xtrax.cli.loader should not eagerly import 'wave'"
        )

        # Now call load_fn with wave (this should trigger the import)
        fn = load_fn("wave:open")

        # After calling load_fn, 'wave' should be loaded
        assert 'wave' in sys.modules

        # And fn should be the real wave.open
        import wave
        assert fn is wave.open


class TestLoadFnMalformed:
    """Tests for malformed import paths."""

    def test_no_colon(self) -> None:
        """load_fn raises CLIImportError for paths without ':'."""
        with pytest.raises(CLIImportError) as exc_info:
            load_fn("json")

        error_msg = str(exc_info.value)
        assert "json" in error_msg
        assert ":" in error_msg or "colon" in error_msg.lower()

    def test_empty_module_part(self) -> None:
        """load_fn raises CLIImportError for ':symbol' (empty module)."""
        with pytest.raises(CLIImportError) as exc_info:
            load_fn(":dumps")

        error_msg = str(exc_info.value)
        # Should mention the problematic path or empty module
        assert ":" in error_msg or "empty" in error_msg.lower()

    def test_empty_attribute_part(self) -> None:
        """load_fn raises CLIImportError for 'json:' (empty attribute)."""
        with pytest.raises(CLIImportError) as exc_info:
            load_fn("json:")

        error_msg = str(exc_info.value)
        # Should mention the problematic path or empty attribute
        assert "json:" in error_msg or "empty" in error_msg.lower()


class TestLoadFnBadModule:
    """Tests for non-existent modules."""

    def test_nonexistent_module(self) -> None:
        """load_fn raises CLIImportError for nonexistent modules."""
        with pytest.raises(CLIImportError) as exc_info:
            load_fn("no_such_pkg_xyz:foo")

        error_msg = str(exc_info.value)
        # Must mention the problematic path
        assert "no_such_pkg_xyz:foo" in error_msg
        # Should provide a helpful hint about sys.path
        assert "sys.path" in error_msg or "installed" in error_msg.lower()


class TestLoadFnBadAttribute:
    """Tests for missing attributes in valid modules."""

    def test_nonexistent_attribute(self) -> None:
        """load_fn raises CLIImportError for nonexistent attributes."""
        with pytest.raises(CLIImportError) as exc_info:
            load_fn("json:no_such_attr")

        error_msg = str(exc_info.value)
        # Must mention the problematic path
        assert "json:no_such_attr" in error_msg or "no_such_attr" in error_msg
