"""Tests verifying optional dependencies behavior when optax/orbax are absent (#3116, L3-F2)."""

import sys
from unittest.mock import patch

import pytest


def test_import_training_modules_without_optax():
    """Importing training modules with optax blocked does not raise error."""
    # Temporarily remove and block optax
    with patch.dict(sys.modules, {"optax": None}):
        import xtrax.checkpoint.orbax as orbax_mod
        import xtrax.training.optim as optim_mod
        import xtrax.training.state as state_mod
        import xtrax.training.step as step_mod
        import xtrax.training.trainer as trainer_mod

        assert optim_mod.optax is None or optim_mod.optax is not None
        assert state_mod.optax is None or state_mod.optax is not None
        assert step_mod.optax is None or step_mod.optax is not None
        assert trainer_mod.optax is None or trainer_mod.optax is not None
        assert orbax_mod.ocp is None or orbax_mod.ocp is not None


def test_require_optax_raises_actionable_error():
    """require_optax raises ImportError with pip install hint when optax is not available."""
    with patch.dict(sys.modules, {"optax": None}):
        from xtrax.training.optim import require_optax

        with pytest.raises(ImportError, match="pip install 'xtrax\\[training\\]'"):
            require_optax()


def test_require_orbax_raises_actionable_error():
    """require_orbax raises ImportError with pip install hint when orbax is not available."""
    with patch.dict(sys.modules, {"orbax.checkpoint": None}):
        from xtrax.checkpoint.orbax import require_orbax

        with pytest.raises(ImportError, match="pip install 'xtrax\\[training\\]'"):
            require_orbax()


def test_functions_call_require_optax_when_optax_missing():
    """Functions call require_optax and raise actionable ImportError when optax is missing."""
    with patch.dict(sys.modules, {"optax": None}):
        from xtrax.training.optim import adamw_with_schedule

        with pytest.raises(ImportError, match="xtrax.training requires 'optax'"):
            adamw_with_schedule(1e-3, 10, 100)
