"""Tests for kinematics — the CAUSAL velocity filter and the ball locator.

`causal_savgol` is the single anti-leakage guarantee behind every kinematic number the
project produces: nothing else checks that a velocity cannot see a future frame, and every
arrival time and reaction lag in s04 is computed from its output. The synthetic tests run on
a bare clone; the last one needs the tracking sample and skips cleanly without it.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import loader
from constants import MAX_PLAYER_SPEED_MS, SAVGOL_POLYORDER, SAVGOL_WINDOW_FRAMES
from kinematics import causal_savgol, compute_velocities, get_ball_pos
from conftest import requires_data

W, P = SAVGOL_WINDOW_FRAMES, SAVGOL_POLYORDER


def _frame() -> pd.DataFrame:
    """One tracking frame: a player and the ball."""
    return pd.DataFrame([
        {"player_id": "1_0", "is_ball": 0, "x_tracking": -10.0, "y_tracking": 0.0},
        {"player_id": None, "is_ball": 1, "x_tracking": 0.0, "y_tracking": 0.0}])


# ── The anti-leakage guarantee ──────────────────────────────────────────
def test_causal_savgol_cannot_see_the_future():
    """Changing a LATER sample must not move an EARLIER output.

    The whole point of the one-sided filter. scipy's centred savgol_filter would fail
    this, and every arrival time, reaction lag and velocity in s04 rests on it."""
    base = np.zeros(40)
    a = causal_savgol(base, window=W, polyorder=P)
    bumped = base.copy()
    bumped[25:] = 100.0                                    # a huge change, but only in the future
    b = causal_savgol(bumped, window=W, polyorder=P)
    assert np.allclose(a[:25], b[:25], equal_nan=True)     # nothing before the bump moved
    assert not np.allclose(a[25:], b[25:], equal_nan=True)  # and the filter is not simply inert


def test_a_centred_filter_fails_the_look_ahead_test():
    """The control that proves the test above discriminates rather than passing by
    construction: the same signal through scipy's CENTRED savgol_filter does move a
    sample before the bump, by exactly the (window-1)/2 frames its half-window reaches
    back. Without this, a filter silently swapped for the centred one could keep the
    look-ahead test green and nothing in the repo would notice."""
    base = np.zeros(40)
    bumped = base.copy()
    bumped[25:] = 100.0
    a, b = savgol_filter(base, W, P), savgol_filter(bumped, W, P)
    assert not np.allclose(a[:25], b[:25])                 # the centred filter leaks
    assert int(np.argmax(~np.isclose(a, b))) == 25 - (W - 1) // 2


def test_causal_savgol_keeps_gaps_undefined():
    """An incomplete or gapped trailing window stays NaN rather than being invented.

    Imputing 0 would be worse than a gap: a fabricated 0 m/s reads as a stationary
    player rather than as an unobserved one."""
    v = np.arange(20.0)
    out = causal_savgol(v, window=W, polyorder=P)
    assert np.isnan(out[:W - 1]).all()                     # the first window-1 samples are undefined
    assert np.isfinite(out[W:]).all()

    gapped = v.copy()
    gapped[10] = np.nan
    g = causal_savgol(gapped, window=W, polyorder=P)
    assert np.isnan(g[10:10 + W]).all()                    # every window containing the gap is undefined
    assert np.isfinite(g[10 + W:]).all()


def test_causal_savgol_recovers_a_constant():
    """A polyorder-1 fit reproduces a constant signal exactly where it is defined."""
    out = causal_savgol(np.full(30, 3.5), window=W, polyorder=P)
    assert np.allclose(out[W - 1:], 3.5)


def test_causal_savgol_reads_the_fit_at_the_newest_sample():
    """The trailing fit must be read at the window's NEWEST sample, not its oldest.

    A constant signal cannot tell the two apart, so this pins it on a ramp, where a
    polyorder-1 fit is exact: reading the newest sample returns v[n], reading the
    oldest returns v[n] - (window - 1). Both orderings are causal, so the existing
    look-ahead test cannot catch this either: the failure mode is a silent
    (window-1)-frame lag on every velocity the race stage consumes."""
    v = np.arange(40, dtype=float)
    out = causal_savgol(v, window=W, polyorder=P)
    assert np.allclose(out[W - 1:], v[W - 1:]), "velocity is lagged, not aligned"


# ── Velocities ──────────────────────────────────────────────────────────
def test_compute_velocities_caps_outliers_as_gaps():
    """A teleport is turned into a NaN gap, never into a 200 m/s velocity."""
    n = 40
    x = np.arange(n, dtype=float) * 0.2                    # 5 m/s at 25 fps
    x[30:] += 60.0                                         # a detection jump
    df = pd.DataFrame({"player_id": "1_0", "period_id": 1, "is_ball": 0,
                       "x_tracking": x, "y_tracking": 0.0})
    out = compute_velocities(df, window=W, polyorder=P, max_speed=12.0, fps=25)
    sp = out["speed"].to_numpy()
    assert np.nanmax(sp) <= 12.0 + 1e-9                    # a clean ramp does not overshoot the cap
    assert np.isnan(sp[30:36]).any()                       # the jump became a gap


# ── Ball position ───────────────────────────────────────────────────────
def test_missing_ball_is_reported_not_faked():
    """Broadcast tracking loses the ball often, and a fabricated position would put the
    whole surface in the wrong place without raising anything."""
    assert get_ball_pos(_frame()) is not None
    assert get_ball_pos(_frame().query("is_ball == 0")) is None


# ── Against real broadcast tracking ─────────────────────────────────────
@requires_data
def test_velocities_on_real_tracking(mid):
    """The whole module over one real match, which the synthetic tests cannot reach:
    partial detection, dropouts and a ball the feed keeps losing are the substrate this
    filter actually runs on.

    It also pins the figures compute_velocities' docstring states about this match, asserted
    exactly: a loader change that moved them would silently make the docstring false."""
    out = compute_velocities(loader.load_tracking(mid))

    ball, players = out[out["is_ball"] == 1], out[out["is_ball"] == 0]
    assert len(ball) > 0 and ball[["vx", "vy", "speed"]].isna().all().all()   # deliberate, see the docstring
    assert players["speed"].notna().mean() > 0.99
    assert (out["speed"].notna() == (out["vx"].notna() & out["vy"].notna())).all()

    # the cap bounds the RAW difference, not the filtered output, which overshoots it
    # on 62 rows here; asserting the count keeps that documented rather than surprising
    fast = players["speed"].dropna() > MAX_PLAYER_SPEED_MS
    assert int(fast.sum()) == 62 and round(players["speed"].max(), 2) == 14.70
    assert int(players["speed"].notna().sum()) == 3_211_443   # defined player speeds

    step = players.groupby(["player_id", "period_id"], dropna=False)["frame_count"].diff()
    gap = step.notna() & (step != 1)
    assert int(step.notna().sum()) == 3_212_606            # player-frame transitions
    assert int(gap.sum()) == 22                            # of them non-consecutive: 0.0007%
    assert int(step[gap].max()) == 42                      # longest dropout, in frames
