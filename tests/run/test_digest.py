"""Tests for xtrax.run.digest module."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from xtrax.run.digest import array_digest, canonical_digest, numerics_env, source_fingerprint


class TestCanonicalDigest:
    """Tests for canonical_digest."""

    def test_is_key_order_independent(self):
        """Verify that key order doesn't affect the digest."""
        payload1 = {"a": 1, "b": 2, "c": 3}
        payload2 = {"c": 3, "a": 1, "b": 2}
        assert canonical_digest(payload1) == canonical_digest(payload2)

    def test_changes_when_value_changes(self):
        """Verify that changing a value changes the digest."""
        payload1 = {"x": 1}
        payload2 = {"x": 2}
        assert canonical_digest(payload1) != canonical_digest(payload2)

    def test_deterministic(self):
        """Verify that the same payload always produces the same digest."""
        payload = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
        digest1 = canonical_digest(payload)
        digest2 = canonical_digest(payload)
        assert digest1 == digest2

    def test_nested_dict_order(self):
        """Verify that nested dicts are also order-independent."""
        payload1 = {"outer": {"a": 1, "b": 2}}
        payload2 = {"outer": {"b": 2, "a": 1}}
        assert canonical_digest(payload1) == canonical_digest(payload2)


class TestArrayDigest:
    """Tests for array_digest."""

    def test_numpy_arrays(self):
        """Verify that numpy arrays can be digested."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        digest = array_digest(arr)
        assert isinstance(digest, str)
        assert len(digest) == 64  # sha256 hex is 64 chars

    def test_jax_arrays(self):
        """Verify that JAX arrays can be digested."""
        try:
            import jax.numpy as jnp

            arr = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
            digest = array_digest(arr)
            assert isinstance(digest, str)
            assert len(digest) == 64
        except ImportError:
            pytest.skip("JAX not installed")

    def test_jax_array_equals_numpy_same_data(self):
        """Verify that JAX and numpy arrays with the same data produce the same digest."""
        try:
            import jax.numpy as jnp

            np_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            jax_arr = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
            assert array_digest(np_arr) == array_digest(jax_arr)
        except ImportError:
            pytest.skip("JAX not installed")

    def test_digest_changes_on_dtype_change(self):
        """Verify that changing dtype changes the digest."""
        arr_f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        arr_f64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        assert array_digest(arr_f32) != array_digest(arr_f64)

    def test_typed_prng_key(self):
        """Verify that typed PRNG keys are handled correctly."""
        try:
            import jax

            key = jax.random.key(0)
            digest = array_digest(key)
            # Should equal the digest of the key_data
            key_data = jax.random.key_data(key)
            assert digest == array_digest(key_data)
        except ImportError:
            pytest.skip("JAX not installed")

    def test_legacy_uint32_key(self):
        """Verify that legacy uint32 keys work."""
        key = np.array([1, 2], dtype=np.uint32)
        digest = array_digest(key)
        assert isinstance(digest, str)
        assert len(digest) == 64

    def test_deterministic(self):
        """Verify that the same array always produces the same digest."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        digest1 = array_digest(arr)
        digest2 = array_digest(arr)
        assert digest1 == digest2


class TestSourceFingerprint:
    """Tests for source_fingerprint."""

    def test_single_file(self, tmp_path: Path):
        """Verify that a single file can be fingerprinted."""
        py_file = tmp_path / "module.py"
        py_file.write_text("x = 1\n")
        digest = source_fingerprint({"test": py_file})
        assert isinstance(digest, str)
        assert len(digest) == 64

    def test_directory(self, tmp_path: Path):
        """Verify that a directory of .py files can be fingerprinted."""
        (tmp_path / "file1.py").write_text("x = 1\n")
        (tmp_path / "file2.py").write_text("y = 2\n")
        digest = source_fingerprint({"test": tmp_path})
        assert isinstance(digest, str)
        assert len(digest) == 64

    def test_deterministic(self, tmp_path: Path):
        """Verify that the same directory always produces the same digest."""
        (tmp_path / "file1.py").write_text("x = 1\n")
        (tmp_path / "file2.py").write_text("y = 2\n")
        digest1 = source_fingerprint({"test": tmp_path})
        digest2 = source_fingerprint({"test": tmp_path})
        assert digest1 == digest2

    def test_changes_when_file_changes(self, tmp_path: Path):
        """Verify that changing a file changes the digest."""
        py_file = tmp_path / "file.py"
        py_file.write_text("x = 1\n")
        digest1 = source_fingerprint({"test": tmp_path})
        py_file.write_text("x = 2\n")
        digest2 = source_fingerprint({"test": tmp_path})
        assert digest1 != digest2

    def test_label_prefix_prevents_collision(self, tmp_path: Path):
        """Verify that different labels prevent collisions from same file."""
        py_file = tmp_path / "file.py"
        py_file.write_text("x = 1\n")
        digest1 = source_fingerprint({"label1": py_file})
        digest2 = source_fingerprint({"label2": py_file})
        assert digest1 != digest2

    def test_ignores_pycache(self, tmp_path: Path):
        """Verify that __pycache__ directories are ignored."""
        (tmp_path / "file.py").write_text("x = 1\n")
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "file.pyc").write_text("compiled\n")
        # Should not change the digest
        digest1 = source_fingerprint({"test": tmp_path})
        (cache_dir / "file.pyc").write_text("different\n")
        digest2 = source_fingerprint({"test": tmp_path})
        assert digest1 == digest2

    def test_missing_path_raises_filenotfound(self, tmp_path: Path):
        """Verify that a missing path raises FileNotFoundError."""
        missing = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            source_fingerprint({"test": missing})

    def test_dir_with_no_py_raises_valueerror(self, tmp_path: Path):
        """Verify that a directory with no .py files raises ValueError."""
        (tmp_path / "file.txt").write_text("not python\n")
        with pytest.raises(ValueError, match="no .py files found"):
            source_fingerprint({"test": tmp_path})

    def test_empty_paths_raises_valueerror(self, tmp_path: Path):
        """Verify that no files overall raises ValueError."""
        with pytest.raises(ValueError, match="no .py files found overall"):
            source_fingerprint({})

    def test_single_file_equals_dir_with_same_file(self, tmp_path: Path):
        """Verify that a single file and a dir containing only that file produce
        the same fingerprint when the label and filename match.
        """
        # Create a single file
        file_path = tmp_path / "mymodule.py"
        file_path.write_text("x = 1\n")
        digest_single = source_fingerprint({"pkg": file_path})

        # Create a directory containing the same file
        dir_path = tmp_path / "subdir"
        dir_path.mkdir()
        (dir_path / "mymodule.py").write_text("x = 1\n")
        digest_dir = source_fingerprint({"pkg": dir_path})

        # Both should produce the same digest (entry name "pkg/mymodule.py" in both cases)
        assert digest_single == digest_dir


class TestNumericsEnv:
    """Tests for numerics_env."""

    def test_has_required_keys(self):
        """Verify that the result has the documented keys."""
        env = numerics_env()
        assert "numerics" in env
        assert "info" in env
        assert "platform" in env["numerics"]
        assert "device_kind" in env["numerics"]
        assert "XLA_FLAGS" in env["numerics"]
        assert "jax_default_matmul_precision" in env["numerics"]
        assert "xla_gpu_deterministic_ops" in env["numerics"]
        assert "hostname" in env["info"]
        assert "slurm_job_id" in env["info"]

    def test_types(self):
        """Verify that the result has the documented types."""
        env = numerics_env()
        assert isinstance(env["numerics"]["platform"], str)
        assert isinstance(env["numerics"]["device_kind"], str)
        assert isinstance(env["numerics"]["XLA_FLAGS"], str)
        assert isinstance(env["numerics"]["jax_default_matmul_precision"], str)
        assert isinstance(env["numerics"]["xla_gpu_deterministic_ops"], bool)
        assert isinstance(env["info"]["hostname"], str)
        assert env["info"]["slurm_job_id"] is None or isinstance(env["info"]["slurm_job_id"], str)

    def test_reads_xla_flags(self, monkeypatch):
        """Verify that XLA_FLAGS is read from environment."""
        monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_deterministic_ops=true")
        env = numerics_env()
        assert env["numerics"]["XLA_FLAGS"] == "--xla_gpu_deterministic_ops=true"
        assert env["numerics"]["xla_gpu_deterministic_ops"] is True

    def test_reads_slurm_job_id(self, monkeypatch):
        """Verify that SLURM_JOB_ID is read from environment."""
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        env = numerics_env()
        assert env["info"]["slurm_job_id"] == "12345"

    def test_slurm_job_id_none_when_unset(self, monkeypatch):
        """Verify that slurm_job_id is None when unset."""
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        env = numerics_env()
        assert env["info"]["slurm_job_id"] is None


class TestRuntimeImportsNoBeartype:
    """Regression tests: runtime imports must not require beartype (not a runtime dependency)."""

    def test_digest_and_zarr_sink_import_without_beartype(self):
        """Verify that xtrax.run.digest and zarr_sink import even when beartype is blocked.

        Regression: beartype is a dev-only dependency (in pyproject extras, not core
        dependencies), so these modules must not import it at runtime.
        """
        # Create a finder that blocks any beartype import
        code = """
import sys
import importlib.abc
import importlib.machinery

class BlockBeartype(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "beartype" or fullname.startswith("beartype."):
            raise ImportError(f"beartype is blocked in this test: {fullname}")
        return None

sys.meta_path.insert(0, BlockBeartype())

# Now try to import xtrax modules
try:
    import xtrax.run.digest
    import xtrax.run.zarr_sink
    print("SUCCESS: imports succeeded without beartype")
    sys.exit(0)
except ImportError as e:
    print(f"FAILED: {e}")
    sys.exit(1)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "SUCCESS" in result.stdout
