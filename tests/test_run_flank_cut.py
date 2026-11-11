"""Tests for run_flank_cut, the annex behind the paper's x >= 85 cut: the routes read per delivery,
the cut's paired difference, the write guard, and the path on real matches.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "src", _REPO / "pipeline", _REPO / "tests"):
    sys.path.insert(0, str(p))

import run_flank_cut as R                                                  # noqa: E402
from conftest import requires_sb                                          # noqa: E402


def test_the_routes_are_read_per_delivery_inside_the_window():
    rows = [{"set_piece_id": "a", "n_flick_on": 1, "n_prolongation": 0, "n_after_clearance": 1,
             "n_att_shots_in_window": 3, "n_direct": 1, "won_first_contact": True},
            {"set_piece_id": "b", "n_flick_on": 0, "n_prolongation": 0, "n_after_clearance": 0,
             "n_att_shots_in_window": 0, "n_direct": 0, "won_first_contact": None}]
    shots = [{"set_piece_id": "a", "segment": "direct", "xg": 0.1, "after_first_touch": False},
             {"set_piece_id": "a", "segment": "flick_on", "xg": 0.2, "after_first_touch": True},
             {"set_piece_id": "a", "segment": "after_clearance", "xg": 0.3, "after_first_touch": True},
             {"set_piece_id": "a", "segment": "late", "xg": 0.9, "after_first_touch": True}]
    d = R.per_delivery(rows, shots).set_index("set_piece_id")
    assert d.loc["a", "xg_win"] == pytest.approx(0.6)          # the late shot is out of the window
    assert d.loc["a", "xg_aft"] == pytest.approx(0.5) and d.loc["a", "xg_direct"] == pytest.approx(0.1)
    assert d.loc["a", ["p_shot", "p_direct", "p_aft", "p_prol", "p_clear"]].tolist() == [1.0] * 5
    assert d.loc["b", ["xg_win", "p_aft"]].tolist() == [0.0, 0.0] and np.isnan(d.loc["b", "won_fc"])


def test_a_cut_differences_below_minus_above_over_shared_draws():
    rng = np.random.default_rng(0)
    fk = pd.DataFrame({"match_id": np.repeat(np.arange(100), 4),
                       "delivery_x": rng.uniform(60.0, 120.0, 400)})
    below = (fk["delivery_x"] < 85.0).to_numpy()
    for k in R.CUT_MEASURES:
        fk[k] = below + rng.normal(0.0, 0.01, len(fk))
    rows = pd.DataFrame(R.cut_contrast(fk, 85.0, 200)).set_index("measure")
    assert rows["below_minus_above"].to_numpy() == pytest.approx(1.0, abs=0.01)
    assert (rows["lo"] > 0.9).all() and rows["n_below"].iloc[0] == int(below.sum())


@pytest.mark.parametrize("out", [[], ["--out-dir", str(_REPO / "data" / "derived" / "zones_F")]])
def test_a_limited_run_may_not_write_the_records(monkeypatch, out):
    """Neither in data/derived nor in zones_F, where the pass under partition F publishes."""
    def work(*a, **k):
        raise AssertionError("the run read the matches before refusing")

    monkeypatch.setattr(R.L, "list_matches", work)
    monkeypatch.setattr(sys, "argv", ["run_flank_cut.py", "--limit", "5", *out])
    with pytest.raises(SystemExit, match="may not write"):
        R.main()


@requires_sb
def test_the_annex_runs_on_real_matches(tmp_path):
    """The path: the lowered grammar, the placement and both tables, on a dozen matches."""
    proc = subprocess.run([sys.executable, "pipeline/run_flank_cut.py", "--limit", "12", "--boot",
                           "20", "--jobs", "2", "--out-dir", str(tmp_path)], cwd=_REPO,
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]
    bands = pd.read_parquet(tmp_path / "flank_cut_bands_limit12.parquet")
    cuts = pd.read_parquet(tmp_path / "flank_cut_contrasts_limit12.parquet")
    assert bands["band"].iloc[0] == "long_corners" and len(bands) > 1
    assert set(cuts["cut"]) == set(R.CUTS) and set(cuts["measure"]) == set(R.CUT_MEASURES)
    # the three jobs read the placement the same under either partition
    assert {"n_attack_cross", "n_stake_drop", "n_hold_back"} <= set(bands.columns)
