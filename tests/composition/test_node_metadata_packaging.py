"""Regression test for the wheel-packaging gap discovered during T1-06 (#3059).

`src/xtrax/composition/node_metadata_schema.toml` is a non-.py data file; hatchling's
`packages = ["src/xtrax"]` wheel target does NOT include non-.py files by default (the only
other one, `py.typed`, needed an explicit `force-include` entry). Without a matching
force-include entry for the schema, `xtrax.composition.node_metadata.load_node_metadata_schema()`
would raise FileNotFoundError for anyone who `pip install`s xtrax, invisible in local dev
(editable installs read straight from the checkout).
"""

import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_schema_toml_is_present_in_built_wheel() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = subprocess.run(
            ["uv", "build", "--wheel", "-o", tmp_dir],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

        wheels = list(Path(tmp_dir).glob("*.whl"))
        assert len(wheels) == 1, wheels

        with zipfile.ZipFile(wheels[0]) as archive:
            names = archive.namelist()
        assert "xtrax/composition/node_metadata_schema.toml" in names, names
