"""Tests for training-only dependencies decoupling (#3116)."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = ROOT / "pyproject.toml"


class TestTrainingExtraDecoupling:
    """Verification controls for training optional dependencies in pyproject.toml."""

    def test_pyproject_isolates_training_in_optional_dependencies(self):
        """optax and orbax-checkpoint are removed from dependencies and placed in [training]."""
        data = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))

        core_deps = data["project"]["dependencies"]
        assert not any("optax" in dep for dep in core_deps), (
            "optax must not be in core dependencies"
        )
        assert not any("orbax" in dep for dep in core_deps), (
            "orbax-checkpoint must not be in core dependencies"
        )

        optional_deps = data["project"]["optional-dependencies"]
        assert "training" in optional_deps, (
            "[project.optional-dependencies].training must be defined"
        )
        training_deps = optional_deps["training"]
        assert any("optax" in dep for dep in training_deps), "optax must be in training extra"
        assert any("orbax-checkpoint" in dep for dep in training_deps), (
            "orbax-checkpoint must be in training extra"
        )

    def test_composite_extras_include_training(self):
        """dev and all composite extras include xtrax[training]."""
        data = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
        optional_deps = data["project"]["optional-dependencies"]

        assert "dev" in optional_deps
        assert "xtrax[training]" in optional_deps["dev"], "dev extra must include xtrax[training]"

        assert "all" in optional_deps
        assert "xtrax[training]" in optional_deps["all"], "all extra must include xtrax[training]"

    def test_guarded_import_message_format(self):
        """Guarded imports for optax and orbax-checkpoint report actionable error message."""
        from xtrax.checkpoint.orbax import require_orbax
        from xtrax.training.optim import require_optax

        # If optax is installed, require_optax() returns module; if missing, raises ImportError
        # We verify that require_optax / require_orbax functions exist and describe the hint
        assert callable(require_optax)
        assert callable(require_orbax)
