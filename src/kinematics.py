"""kinematics — velocities from tracking, and the causal filter the whole race rests on.

WHAT IS HERE RUNS ON EVERY CORNER s04 RACES. `compute_velocities` runs in
`SecondBallRace.__init__`, and `get_ball_pos` locates the ball at the kick frame, where s04's
flight test starts; `pitch_control` reads the ball through `get_ball_pos` too.

THE ONE-SIDED FILTER IS THE ANTI-LEAKAGE GUARANTEE, not a smoothing preference. Every arrival
time, every reaction lag and every velocity the project publishes passes through
`causal_savgol`, and scipy's centred `savgol_filter` would fit a window straddling the present
sample, so a player's speed at the first contact would carry frames from after it. That is the
future entering a feature read at T, which the anti-leakage rule bars outright.

Velocities are ALWAYS re-derived here rather than taken from the vendor's own vx/vy: their
smoothing is undocumented and may itself read future frames.

THE BALL ROW IS TAKEN AS THE VENDOR EMITS IT, and the vendor marks most of it as interpolated
where the race reads it. `visibility` is `VISIBLE` on a detected ball entry and `ESTIMATED` on an
interpolated one, and on the ball channel `confidence` is that same field under a second name,
`LOW` on exactly the `ESTIMATED` rows. The provider confirms that its ball
tracking is weak while the ball is off the ground and recommends no filter; a corner delivery is
airborne from the kick to the first contact, so the ball rows s04 reads at the contact and the
loose point are interpolated on most corners. Nothing here masks them: a mask would empty the
race, and the bound travels with what the race publishes instead (measured.md §race).

Coordinates are tracking metres, origin at the pitch centre, 105 x 68, at 25 fps.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_coeffs

from constants import (FPS_TRACKING, MAX_PLAYER_SPEED_MS, SAVGOL_POLYORDER,
                       SAVGOL_WINDOW_FRAMES)


# ── The causal filter ───────────────────────────────────────────────

def causal_savgol(v: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """One-sided (causal) Savitzky-Golay smoother: fits the polynomial to the
    window ENDING at each sample and reads it there, so no future frame leaks
    into the present value (the centred savgol_filter would, which
    definitions.md §10 bars). NaN-aware: a sample whose trailing window is not
    fully observed, at an edge or across a detection gap in broadcast tracking,
    stays NaN instead of being invented: a fabricated 0 m/s reads as a
    stationary player rather than as an unobserved one.

    The coefficient order is the subtle part. savgol_coeffs defaults to
    use="conv", so what comes back is ALREADY in convolution order and
    np.convolve applies it with pos=window-1 landing on the newest sample.
    Reversing it would still be causal, but would read the fit at the OLDEST
    sample of the window and so lag every velocity by window-1 frames, a
    failure the look-ahead test cannot see because both orderings are causal."""
    c = savgol_coeffs(window, polyorder, pos=window - 1)
    conv = np.convolve(np.nan_to_num(v), c, mode="full")[:len(v)]
    seen = np.convolve((~np.isnan(v)).astype(float), np.ones(window), mode="full")[:len(v)]
    conv[seen < window - 1e-9] = np.nan
    return conv


# ── Velocities ──────────────────────────────────────────────────────

def compute_velocities(
    tracking_df: pd.DataFrame,
    window: int = SAVGOL_WINDOW_FRAMES,
    polyorder: int = SAVGOL_POLYORDER,
    max_speed: float = MAX_PLAYER_SPEED_MS,
    fps: int = FPS_TRACKING,
) -> pd.DataFrame:
    """Add vx, vy, speed columns using a CAUSAL Savitzky-Golay filter.

    Computes smoothed velocities from x_tracking/y_tracking per player per
    period, one-sided so no future frame enters the present value, and turns
    raw differences above max_speed into NaN gaps rather than imputing them to 0.

    MAX_SPEED BOUNDS WHAT ENTERS THE FILTER, NOT WHAT LEAVES IT. The cap is
    applied to the raw finite difference, and a polyorder-1 fit read at the
    newest sample can overshoot its own inputs, so the returned speed is not
    itself bounded by max_speed. On match 4004219 that leaves 62 of 3,211,443
    defined player speeds above the max_speed cap, the largest 14.70 m/s.

    The finite difference assumes consecutive rows are consecutive frames, so a
    detection dropout is differenced as if it were one frame. Measured on match
    4004219 that is 22 of 3,212,606 player-frame transitions (0.0007%, longest
    gap 42 frames), and the max_speed cap turns those into NaN rather than a
    fake speed.

    THE BALL IS LEFT WITHOUT A VELOCITY ON PURPOSE. Its rows carry a null
    player_id, which groupby drops, so vx/vy/speed come back NaN there and the
    vendor's own values for the ball do not survive this call. Nothing reads
    them: pitch_control._extract_teams and s04's arrivals both filter on
    is_ball == 0, and the one place that does compute an arrow for the ball
    (render_race) hands it to plot_race, which drops it for the same reason.

    Args:
        tracking_df: Output of load_tracking.
        window: Savgol window size in frames (odd number).
        polyorder: Polynomial order for Savgol filter.
        max_speed: Bound on the RAW difference (m/s); a step above it becomes a
            NaN gap. It does not bound the returned speed, as above.
        fps: Tracking rate (dt = 1/fps); constants.FPS_TRACKING carries the
            measurement.

    Returns:
        tracking_df with added columns: vx, vy, speed (NaN where undefined).
    """
    df = tracking_df.copy()
    df["vx"] = np.nan
    df["vy"] = np.nan
    dt = 1.0 / fps

    for _, group in df.groupby(["player_id", "period_id"]):
        idx = group.index
        vx = (group["x_tracking"].diff() / dt).to_numpy()
        vy = (group["y_tracking"].diff() / dt).to_numpy()

        raw_speed = np.sqrt(vx**2 + vy**2)
        vx[raw_speed > max_speed] = np.nan
        vy[raw_speed > max_speed] = np.nan

        # a group with barely a window of detections is skipped whole, rather than
        # contributing an isolated sample surrounded by NaN
        if np.isfinite(vx).sum() > window:
            df.loc[idx, "vx"] = causal_savgol(vx, window, polyorder)
            df.loc[idx, "vy"] = causal_savgol(vy, window, polyorder)

    df["speed"] = np.sqrt(df["vx"]**2 + df["vy"]**2)
    return df


# ── Ball position in a frame ────────────────────────────────────────

def get_ball_pos(frame_data: pd.DataFrame) -> Optional[np.ndarray]:
    """Ball position in metres from one tracking frame, or None when no ball row exists.

    Reads the row whatever its `visibility`. An `ESTIMATED` ball is the vendor's interpolation
    rather than a detection, and it is the common case inside a corner's race window (module
    docstring). This returns the position the feed carries; what that bound means for a number
    is the caller's to state."""
    ball = frame_data[frame_data["is_ball"] == 1]
    if ball.empty:
        return None
    return np.array([ball.iloc[0]["x_tracking"], ball.iloc[0]["y_tracking"]])
