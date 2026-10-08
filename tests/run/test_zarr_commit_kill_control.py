"""Negative control for T-X1: the same kill harness must observe tearing of in-place writes.

It must also observe COMPLETE in-place keys, so that both outcomes are known to be
distinguishable by the harness (a harness that only ever sees one outcome proves nothing).
"""

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
    def _is_complete(key_dir: Path) -> bool:
        """True iff the key's group opens cleanly and carries the ``done`` marker."""
        try:
            return bool(zarr.open_group(str(key_dir), mode="r").attrs.get("done"))
        except Exception:  # noqa: BLE001 - any unreadable/partial group is not complete
            return False

    def test_inplace_writes_are_observed_torn(self, tmp_path: Path) -> None:
        script = tmp_path / "inplace_child.py"
        script.write_text(_INPLACE_CHILD)
        duration = _measure_duration(tmp_path, script)
        stores = _kill_grid(tmp_path, script, duration)

        torn_total = 0
        complete_total = 0
        for store in stores:
            for key_dir in _key_dirs(store):
                complete = self._is_complete(key_dir)
                complete_total += complete
                torn_total += not complete
        assert torn_total >= 1, (
            f"kill harness cannot discriminate torn writes (D={duration:.2f}s): "
            "no in-place key was ever observed without its 'done' marker"
        )
        assert complete_total >= 1, (
            f"kill harness cannot discriminate complete writes (D={duration:.2f}s): "
            "no in-place key was ever observed WITH its 'done' marker, so a 'torn' verdict "
            "may just mean the marker is never readable"
        )
        print(
            f"T-X1 control: D={duration:.2f}s, torn keys observed={torn_total}, "
            f"complete keys observed={complete_total}"
        )
