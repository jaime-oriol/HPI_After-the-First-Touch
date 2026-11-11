"""Tests for report_figures' reading of the refit draws: which reading decides, and the symmetry
check that licenses it.

A refit draw keeps about 63% of the distinct matches, so the cloud shifts with the learning curve
and its percentiles sit off the point. The normal reading keeps the point as the centre, the
pivotal one mirrors the shift and the percentile one carries it. On symmetric draws the centred
percentiles agree with the normal form, and the section has to notice when they do not.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import report_figures as rf

# Median 0.01, percentiles [-0.0185, 0.0385].
_SYMMETRIC = np.linspace(-0.02, 0.04, 601)


def test_on_draws_centred_on_the_point_every_reading_coincides():
    r = rf._refit_reading(0.01, _SYMMETRIC)
    for key in ("normal_ci", "adjusted_ci", "refit_ci", "centred_ci"):
        assert r[key] == pytest.approx([-0.0185, 0.0385], abs=1e-9)
    assert r["n_boot"] == 601 and r["skew"] == pytest.approx(0.0, abs=1e-9)


def test_a_shifted_cloud_moves_the_percentile_and_mirrors_the_pivotal_but_not_the_normal():
    """The same draws sitting 0.02 below a point of 0.03."""
    r = rf._refit_reading(0.03, _SYMMETRIC)
    assert r["normal_ci"] == pytest.approx([0.0015, 0.0585], abs=1e-9)
    assert r["centred_ci"] == pytest.approx(r["normal_ci"], abs=1e-9)
    assert r["refit_ci"] == pytest.approx([-0.0185, 0.0385], abs=1e-9)
    assert r["adjusted_ci"] == pytest.approx([0.0215, 0.0785], abs=1e-9)


def _skewed() -> np.ndarray:
    """A long right tail: 2.5th percentile 0, median 0.005, 97.5th 0.05."""
    return np.r_[np.zeros(5), np.full(90, 0.005), np.full(5, 0.05)]


def _draws_frame(contrasts: dict) -> pd.DataFrame:
    return pd.DataFrame([{"population": pop, "contrast": c, "point": point, "draw": i, "value": v}
                         for (pop, c), (point, d) in contrasts.items() for i, v in enumerate(d)])


def test_refit_readings_flags_a_contrast_whose_skew_flips_the_verdict(monkeypatch):
    """At a point of 0.01 the skewed draws give a normal form of [-0.015, 0.035], which covers
    zero, and centred percentiles of [0.005, 0.055], which do not: the symmetry the normal
    reading assumes fails there, and one such contrast is enough to break the contract."""
    frame = _draws_frame({("corner", "head_to_head"): (0.03, _SYMMETRIC),
                          ("corner", "step_360"): (0.01, _skewed())})
    monkeypatch.setattr(rf, "_load", lambda name: frame if name == "refit_draws.parquet" else None)
    out = rf.refit_readings()
    assert out["corner"]["head_to_head"]["verdict_agrees"] is True
    assert out["corner"]["step_360"]["verdict_agrees"] is False
    assert out["every_verdict_agrees"] is False


def test_refit_readings_is_empty_before_the_driver_has_run(monkeypatch):
    monkeypatch.setattr(rf, "_load", lambda name: None)
    assert rf.refit_readings() == {}
