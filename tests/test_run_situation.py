"""Tests for run_situation: the radii record the situation constants are read against."""
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "src", _REPO / "pipeline"):
    sys.path.insert(0, str(p))

import run_situation as R                                             # noqa: E402
from constants import NEAR_RADII, STAKE_RADIUS                        # noqa: E402


def test_the_radii_record_reads_each_constant_on_its_own_substrate():
    """The drop over the rows with a drop, each ring over the bodies the rings score, and the
    staker distance over the contested drops that had a staker: a row without the substrate
    never enters the denominator."""
    t = pd.DataFrame({
        "drop_x": [100.0, 104.0, 110.0, None], "drop_y": [40.0, 38.0, 44.0, None],
        "n_scored": [10, 6, None, 4],
        "n_att_near3": [1, 0, None, 0], "n_def_near3": [1, 1, None, 0],
        "n_att_near5": [2, 1, None, 1], "n_def_near5": [2, 2, None, 0],
        "staker_dist": [2.0, float("nan"), 15.0, None]})
    r = R.radii_record(t).set_index(["block", "key"])
    assert r.loc[("drop", "median_x"), "value"] == 104.0 and r.loc[("drop", "median_x"), "n"] == 3
    inner, outer = (f"share_within_{v:g}" for v in NEAR_RADII)
    assert r.loc[("near", inner), "value"] == 3 / 20 and r.loc[("near", inner), "n"] == 20
    assert r.loc[("near", outer), "value"] == 8 / 20
    assert r.loc[("stake", "median"), "value"] == 8.5 and r.loc[("stake", "median"), "n"] == 2
    assert r.loc[("stake", f"share_within_{STAKE_RADIUS:g}"), "value"] == 0.5
