"""Negative control for T-X1: the same kill harness must observe tearing of in-place writes."""

from pathlib import Path

import zarr

from tests.run._zarr_commit_kill_harness import (
    _INPLACE_CHILD,
    _key_dirs,
    _kill_grid,
    _measure_duration,
)


class TestKillHarnessNegativeControl:
    """The same harness, against non-atomic in-place writes, must observe tearing."""

    @staticmethod
    def _is_torn(key_dir: Path) -> bool:
        try:
            group = zarr.open_group(str(key_dir), mode="r")
            return not group.attrs.get("done")
        except Exception:  # noqa: BLE001 - any unreadable/partial group is a torn write
            return True

    def test_inplace_writes_are_observed_torn(self, tmp_path: Path) -> None:
        script = tmp_path / "inplace_child.py"
        script.write_text(_INPLACE_CHILD)
        duration = _measure_duration(tmp_path, script)
        stores = _kill_grid(tmp_path, script, duration)

        torn_total = 0
        for store in stores:
            torn_total += sum(self._is_torn(d) for d in _key_dirs(store))
        assert torn_total >= 1, (
            f"kill harness cannot discriminate torn writes (D={duration:.2f}s): "
            "no in-place key was ever observed without its 'done' marker"
        )
        print(f"T-X1 control: D={duration:.2f}s, torn keys observed={torn_total}")
