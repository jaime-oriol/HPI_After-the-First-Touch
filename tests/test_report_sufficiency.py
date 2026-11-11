"""Tests for report_figures.sufficiency: which interval decides the freeze-frame verdict.

The fixed-fit intervals hold the out-of-fold probabilities and are too narrow, and the form this
repo reads is the refit one in normal shape, the point plus or minus half the refit width. A
published verdict decided on the narrow bound while the honest one sits beside it is the claim
leaning on the interval that flatters it, so when the record carries the refit block the boolean
is decided there, it says which one it used, and what the pivotal and percentile forms would have
said travels beside it.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import report_figures as rf

_STEPS = {"static_race-delivery": (-0.0010, -0.0030, -0.0001),
          "geometry360-delivery": (-0.0014, -0.0027, 0.00002),
          "gbt360-delivery": (-0.0006, -0.0020, 0.0010)}


def _record(refit: dict | None) -> pd.DataFrame:
    """A grant block whose fixed intervals say the static race beats the delivery rung. `refit`
    maps a decisive step to its (point, percentile lo, percentile hi); None leaves the block out."""
    rows = [{"block": "population", "key": k, "population": "grant", "n": n}
            for k, n in (("corners", 100), ("matches", 10), ("leagues", 5),
                         ("positive", 60), ("negative", 40))]
    rows.append({"block": "population", "key": "base_rate", "population": "grant", "n": 100,
                 "point": 0.6})
    rows.append({"block": "tuning", "key": "tau_snapshot", "population": "grant", "n": 100,
                 "point": 0.7})
    for key, (point, lo, hi) in _STEPS.items():
        rows.append({"block": "delta", "key": key, "population": "grant", "n": 100, "point": point,
                     "lo": lo, "hi": hi, "covers_zero": lo <= 0 <= hi, "p_wilcoxon": 0.5})
    for key, (point, p_lo, p_hi) in (refit or {}).items():
        rows.append({"block": "refit", "key": key, "population": "grant", "n": 300,
                     "point": point, "lo": 2 * point - p_hi, "hi": 2 * point - p_lo,
                     "step_lo": p_lo, "step_hi": p_hi, "covers_zero": True, "width_ratio": 1.8})
    return pd.DataFrame(rows)


def _all_cover(overrides: dict | None = None) -> dict:
    """Every decisive step's refit interval covering zero, with `overrides` replacing some."""
    base = {k: (p, p - 0.005, p + 0.003) for k, (p, _, _) in _STEPS.items()}
    return base | (overrides or {})


def test_the_verdict_is_decided_on_the_refit_normal_interval_when_the_record_has_one(monkeypatch):
    monkeypatch.setattr(rf, "_load", lambda name: _record(_all_cover()))
    g = rf.sufficiency()["grant"]
    assert g["headline_delta"]["covers_zero"] is False, "the fixture must disagree across forms"
    assert g["freeze_frame_beats_the_delivery_baseline"] is False
    assert g["verdict_interval"] == "refit_normal"
    assert set(g["refit"]) == {"headline_delta", "geometry_over_delivery",
                               "strong_learner_over_delivery"}


def test_a_refit_normal_interval_below_zero_makes_the_frame_beat_the_delivery(monkeypatch):
    geometry = {"geometry360-delivery": (-0.003, -0.005, -0.003)}
    monkeypatch.setattr(rf, "_load", lambda name: _record(_all_cover(geometry)))
    g = rf.sufficiency()["grant"]
    assert g["refit"]["geometry_over_delivery"]["normal_ci"] == [-0.004, -0.002]
    assert g["freeze_frame_beats_the_delivery_baseline"] is True


def test_the_pivotal_reading_travels_beside_the_one_that_decides(monkeypatch):
    """The corner case: pivotal wholly below zero, the normal and percentile forms covering it."""
    geometry = {"geometry360-delivery": (-0.00211, -0.00354, 0.0012)}
    monkeypatch.setattr(rf, "_load", lambda name: _record(_all_cover(geometry)))
    g = rf.sufficiency()["grant"]
    assert g["refit"]["geometry_over_delivery"]["adjusted_ci"] == [-0.00542, -0.00068]
    assert g["refit"]["geometry_over_delivery"]["normal_ci"] == [-0.00448, 0.00026]
    assert g["freeze_frame_beats_the_delivery_baseline"] is False
    assert g["beats_under_pivotal"] is True
    assert g["beats_under_percentile"] is False


def test_a_partial_refit_block_falls_back_to_the_fixed_interval_whole(monkeypatch):
    partial = _all_cover()
    del partial["gbt360-delivery"]
    monkeypatch.setattr(rf, "_load", lambda name: _record(partial))
    g = rf.sufficiency()["grant"]
    assert g["verdict_interval"] == "fixed_oof"
    assert g["freeze_frame_beats_the_delivery_baseline"] is True


def test_without_a_refit_block_the_verdict_says_it_used_the_fixed_interval(monkeypatch):
    monkeypatch.setattr(rf, "_load", lambda name: _record(None))
    g = rf.sufficiency()["grant"]
    assert g["freeze_frame_beats_the_delivery_baseline"] is True
    assert g["verdict_interval"] == "fixed_oof"
    assert "refit" not in g
