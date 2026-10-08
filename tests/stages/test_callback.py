"""Tests for the T1-03 vendored io_callback shim (#3007)."""

from __future__ import annotations

import inspect

import jax
import pytest

from xtrax.stages._callback import (
    PINNED_JAX_RANGE,
    IoCallbackSignatureError,
    _check_jax_version,
    _check_signature,
    _parse_version,
    io_callback,
)


class TestParseVersion:
    def test_parses_clean_dotted_version(self):
        assert _parse_version("0.10.2") == (0, 10, 2)

    def test_ignores_non_numeric_suffix(self):
        assert _parse_version("0.10.2.dev20260710") == (0, 10, 2)

    def test_stops_at_first_non_numeric_chunk(self):
        assert _parse_version("0.10.rc1") == (0, 10)


class TestCheckJaxVersion:
    def test_within_pinned_range_does_not_raise(self):
        _check_jax_version("0.10.2", PINNED_JAX_RANGE)
        _check_jax_version("0.10.99", PINNED_JAX_RANGE)

    def test_below_pinned_range_raises(self):
        with pytest.raises(IoCallbackSignatureError, match="outside the pinned range"):
            _check_jax_version("0.9.0", PINNED_JAX_RANGE)

    def test_at_or_above_upper_bound_raises(self):
        # Derived from the constant rather than hardcoded: this test previously pinned
        # "0.11.0" and silently became a no-op assertion the moment the range widened.
        at_upper_bound = ".".join(str(part) for part in PINNED_JAX_RANGE[1])
        with pytest.raises(IoCallbackSignatureError, match="outside the pinned range"):
            _check_jax_version(at_upper_bound, PINNED_JAX_RANGE)

    def test_real_installed_jax_is_within_pinned_range(self):
        """Guards the actual environment, not just fixtures (matches audit_substrate_lock's
        test_real_agent_assets_workflows_dir_is_clean pattern)."""
        _check_jax_version(jax.__version__)


class TestCheckSignature:
    _EXPECTED = ("callback", "result_shape_dtypes", "args", "sharding", "ordered", "kwargs")

    def _signature(self, param_names: tuple[str, ...], ordered_default: object = False):
        params = []
        seen_var_positional = False
        for name in param_names:
            if name == "args":
                params.append(inspect.Parameter(name, inspect.Parameter.VAR_POSITIONAL))
                seen_var_positional = True
            elif name == "kwargs":
                params.append(inspect.Parameter(name, inspect.Parameter.VAR_KEYWORD))
            elif name == "ordered":
                params.append(
                    inspect.Parameter(
                        name,
                        inspect.Parameter.KEYWORD_ONLY,
                        default=ordered_default,
                    )
                )
            elif seen_var_positional:
                # Anything after *args must be keyword-only (e.g. a renamed
                # 'ordered'/'sharding' param in a drift-simulation fixture).
                params.append(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None))
            else:
                params.append(inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD))
        return inspect.Signature(params)

    def test_matching_signature_does_not_raise(self):
        sig = self._signature(self._EXPECTED)
        _check_signature(sig, self._EXPECTED)

    def test_renamed_param_raises(self):
        drifted = ("callback", "result_shape_dtypes", "args", "sharding", "order", "kwargs")
        sig = self._signature(drifted)
        with pytest.raises(IoCallbackSignatureError, match="signature changed"):
            _check_signature(sig, self._EXPECTED)

    def test_removed_param_raises(self):
        drifted = ("callback", "result_shape_dtypes", "args", "kwargs")
        sig = self._signature(drifted)
        with pytest.raises(IoCallbackSignatureError, match="signature changed"):
            _check_signature(sig, self._EXPECTED)

    def test_changed_ordered_default_raises(self):
        sig = self._signature(self._EXPECTED, ordered_default=True)
        with pytest.raises(IoCallbackSignatureError, match="'ordered' parameter default"):
            _check_signature(sig, self._EXPECTED)

    def test_real_io_callback_signature_matches_today(self):
        """Guards the actual jax.experimental.io_callback shape, not just fixtures."""
        _check_signature(inspect.signature(jax.experimental.io_callback))


def test_shim_reexports_the_real_io_callback():
    assert io_callback is jax.experimental.io_callback


def test_import_succeeds_cleanly():
    import xtrax.stages._callback  # noqa: F401
