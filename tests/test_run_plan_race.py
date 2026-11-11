"""run_plan_race and tracking_clock on synthetic data: the clock offset is found where it was planted, the three cases are
the ones the sheet defines, an onset is read where the run starts and never from an estimated frame, and the share that
no relation would give is the product of the two sides' shares."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import run_plan_race as rpr  # noqa: E402
import tracking_clock as tc  # noqa: E402
from constants import FPS_TRACKING, PITCH_LENGTH_M, PITCH_WIDTH_M, PITCH_X_SB, PITCH_Y_SB  # noqa: E402


def _yards(x_m: float, y_m: float) -> list:
    """Tracking metres back to StatsBomb yards for a side attacking +x: the inverse of anchor_in_tracking at side +1."""
    return [PITCH_X_SB - (PITCH_LENGTH_M / 2 - x_m) * PITCH_X_SB / PITCH_LENGTH_M,
            PITCH_Y_SB / 2 - y_m * PITCH_Y_SB / PITCH_WIDTH_M]


def _events(times: np.ndarray, locations: list, kind: str) -> pd.DataFrame:
    return pd.DataFrame({"type.name": kind, "location": locations, "team.id": 1, "period": 1,
                         "timestamp": [f"00:{int(t // 60):02d}:{t % 60:06.3f}" for t in times]})


def test_the_clock_offset_is_found_where_it_was_planted():
    """Events logged `planted` seconds off the tracking clock, at the tracked ball's place: both anchors and their
    midpoint land on the plant, to one step of the grid."""
    t = np.arange(0, 600, 1 / FPS_TRACKING)
    x, y = 30 * np.sin(0.05 * t), 20 * np.cos(0.07 * t)
    ball = {1: np.array([t, x, y, np.ones_like(t)])}
    planted = -1.2
    te = np.arange(10, 590, 2.9)
    where = [_yards(np.interp(a, t, x), np.interp(a, t, y)) for a in te + planted]
    ev = pd.concat([_events(te, where, kind) for kind in tc.ANCHORS.values()])
    off = tc.clock_offsets(ev, ball, {1: True}, home_id=1)[1]
    step = float(np.diff(tc.SHIFTS)[0])
    assert all(abs(off[k] - planted) <= step + 1e-9 for k in ("pass", "receipt", "mid")), off


def test_the_three_cases_are_the_sheet_s():
    assert rpr.group("att", "att", "def") == "plan" and rpr.group("def", "def", "def") == "plan"
    assert rpr.group("att", "def", "att") == "flight" and rpr.group("att", "def", "def") == "race"


def test_an_onset_is_read_where_the_run_starts_and_never_from_an_estimated_frame():
    """Three players near a drop at the origin: one who starts running at it 0.4 s after the clearance, one the feed
    only estimates, and one already running when the window opens."""
    t_clear = 100.0
    ts = np.arange(t_clear - 2.0, t_clear + 3.0, 1 / FPS_TRACKING)
    toward = -np.array([12.0, 3.0]) / np.hypot(12.0, 3.0)
    frames = []
    for k, t in enumerate(ts):
        late = 10.0 - 3.0 * max(0.0, t - (t_clear + 0.4))
        early = np.array([12.0, 3.0]) + 2.0 * (t - ts[0]) * toward
        frames.append((t, k, [(True, "9", True, late, 0.0), (False, "4", False, 5.0, 5.0),
                              (True, "7", True, early[0], early[1])]))
    got = {o["jersey"]: o for o in rpr.onsets(frames, t_clear, np.zeros(2), home_is_att=True)}
    assert 0.4 <= got["9"]["onset_s"] <= 0.6 and got["9"]["fully_detected"] and not got["9"]["censored"]
    assert got["4"]["censored"] and got["4"]["onset_s"] == rpr.POST_S and not got["4"]["fully_detected"]
    assert got["7"]["already_moving"] and got["7"]["onset_s"] == pytest.approx(-rpr.PRE_S, abs=1 / FPS_TRACKING)
    assert (got["9"]["side"], got["4"]["side"]) == ("att", "def")


def test_the_flag_diagnostic_reads_only_the_corners_struck_from_the_flag():
    """The cross that delivers a short corner is struck from the wing, so its distance to a flag would move the median
    of a diagnostic that asks whether the aligned clock puts the ball at the corner flag when the corner is taken."""
    ok = pd.DataFrame({"anchor": ["kick", "kick", "kick", "cross"], "ball_to_flag_m": [1.0, 2.0, 3.0, 20.0]})
    d = rpr.flag_diagnostic(ok)
    assert d == {"block": "diagnostic", "key": "ball_to_flag_m", "n": 3, "point": 2.0}


def test_no_relation_is_the_product_of_the_two_sides_shares():
    rng = np.random.default_rng(0)
    t = pd.DataFrame({"match_id": np.repeat(np.arange(8), 5), "won": rng.random(40) < 0.6,
                      "near_kick": np.where(rng.random(40) < 0.7, "att", "def")})
    t["group_mid"] = np.where((t["near_kick"] == "att") == t["won"], "plan", "race")
    rec = pd.DataFrame(rpr.no_relation(t, boot=50)).set_index("key")
    w, n = t["won"].mean(), (t["near_kick"] == "att").mean()
    assert rec.loc["expected", "point"] == pytest.approx(w * n + (1 - w) * (1 - n))
    assert rec.loc["observed", "point"] == pytest.approx((t["group_mid"] == "plan").mean())
    assert rec.loc["diff", "point"] == pytest.approx(rec.loc["observed", "point"] - rec.loc["expected", "point"])
