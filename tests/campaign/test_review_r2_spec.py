"""Review round 2: duplicate grid values would plan colliding content-derived units."""

from __future__ import annotations

import pytest

from xtrax.campaign.spec import ParamGrid


def test_param_grid_rejects_repeated_axis_value() -> None:
    with pytest.raises(ValueError, match="repeats the value"):
        ParamGrid(axes={"temperature": [0.1, 0.2, 0.1]})


def test_param_grid_rejects_repeated_structured_value() -> None:
    with pytest.raises(ValueError, match="repeats the value"):
        ParamGrid(axes={"cfg": [{"a": 1, "b": 2}, {"b": 2, "a": 1}]})


def test_param_grid_accepts_distinct_values() -> None:
    ParamGrid(axes={"temperature": [0.1, 0.2], "n": [1, 2]})
