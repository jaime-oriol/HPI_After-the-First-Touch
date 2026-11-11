"""Tests for report_figures.repeatability: the split-half intervals.

The headline of Q3 is the Spearman-Brown figure. Both of its intervals come from team draws, the
full-length bounds are published beside the figure prose quotes, and a metric the run did not
interval publishes None rather than NaN, which would never compare equal and would leave --check
reporting the section stale on every run.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import report_figures as rf


def _rel() -> pd.DataFrame:
    return pd.DataFrame({"metric": ["aft_goals_per100", "corners_per_match"],
                         "half_r": [0.162, 0.586], "half_lo": [-0.050, 0.425],
                         "half_hi": [0.349, 0.692], "full_r_sb": [0.279, 0.739],
                         "full_lo": [-0.105, 0.596], "full_hi": [0.517, 0.818],
                         "n_splits": 500, "n_boot": 2000})


def test_repeatability_publishes_the_headline_interval(monkeypatch):
    monkeypatch.setattr(rf, "_load", lambda name: _rel() if "reliability" in name else None)
    out = rf.repeatability("corner")
    assert out["aft_goals_per100"]["half_ci"] == [-0.05, 0.349]
    assert out["aft_goals_per100"]["full_ci"] == [-0.105, 0.517]
    assert out["corners_per_match"]["full_ci"] == [0.596, 0.818]


def test_an_unintervalled_metric_publishes_none(monkeypatch):
    rel = _rel()
    rel.loc[0, ["half_lo", "half_hi", "full_lo", "full_hi"]] = np.nan
    monkeypatch.setattr(rf, "_load", lambda name: rel if "reliability" in name else None)
    out = rf.repeatability("corner")
    assert out["aft_goals_per100"]["half_ci"] is None
    assert out["aft_goals_per100"]["full_ci"] is None
