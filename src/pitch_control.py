"""pitch_control — Pitch control (Spearman PPCF) from tracking data.

Adapted from LaurieOnTracking (Friends-of-Tracking-Data-FoTD) for the Hudl
Broadcast X&Y long-format tracking (25 fps, meters). Kept as the cited Spearman
cross-check lineage for s04_race, which models its own velocity-aware
time-to-arrive and takes no surface from here.

WHAT THE PIPELINE ACTUALLY TAKES FROM THIS MODULE, so the boundary is readable
without tracing imports: `default_model_params`, and only its max_player_speed,
which s04, s04b_snapshot and run_first_contact use as the travel cap in their own
time-to-arrive. Everything else serves one line of render_race, which evaluates
the PPCF at the loose-ball point and PRINTS it as a console cross-check against
our own reach field. Nothing is drawn, and no published figure carries it. The
velocities the race is built on come from `kinematics`, never from here.

FIVE PROPERTIES OF THE RETURNED NUMBER, none of them visible at the call site:

  - It is PPCF_att alone, returned raw. The two teams sum to within
    model_converge_tol of 1 rather than to 1 (0.990-0.994 measured on a
    contested 11v11), because the integration stops at the first step that
    crosses the tolerance. The value is therefore low by up to one point.
  - It is NOT bounded above by 1 in a degenerate frame. Six or more attackers
    standing on the target with a long ball flight overshoot within a single
    Euler step; eleven of them return 1.89. Inherited from the reference, which
    returns the identical value, and not reachable at a loose-ball point, whose
    ball flight is short (the LOOSE_HORIZON_S note in constants has the distance).
  - A frame carrying no ball row is treated as "the ball is already there"
    (ball_tt = 0), which is the reference behaviour and moves the result by up
    to 0.1. get_ball_pos returns None often on broadcast tracking.
  - An empty attacking team gives 0.0 and an empty defending team 1.0. With BOTH
    empty the veto arithmetic is inf - inf and the answer is 0.0, so a frame with
    no players reads as "the defence owns it".
  - Only the defending keeper gets lambda_gk, as in the reference.

Coordinate convention: tracking x_tracking/y_tracking are meters centered at
(0,0), and all geometry here works in those meters.

Reference: Spearman 2018 "Beyond Expected Goals", by way of LaurieOnTracking.
The parameters below are that implementation's and NOT the paper's; see
default_model_params.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kinematics import get_ball_pos

# (pos, vel, is_gk) for one team, each row a player.
_TeamArrays = tuple[np.ndarray, np.ndarray, np.ndarray]


# ── MODEL PARAMETERS ────────────────────────────────────────────────

def default_model_params(time_to_control_veto: int = 3) -> dict[str, float]:
    """PPCF model parameters, as published by LaurieOnTracking.

    NOT the paper's values, and the difference is deliberate upstream: Spearman
    gives the defence an advantage of kappa = 1.72, while the reference
    implementation sets kappa = 1.0 "so that home & away players have same ball
    control probability". lambda_def inherits that choice. Speeds are m/s, times
    seconds, lambdas control-rate per second.

    time_to_control_veto sets how far ahead one team must be for the other's
    contribution to fall below 10^-veto, which lets the core skip the Euler
    integration for that target. The reference calls these 'short-cut'
    parameters and attributes them to itself rather than to the paper.
    """
    params = {
        "max_player_speed": 5.0,
        "reaction_time": 0.7,
        "tti_sigma": 0.45,
        "kappa_def": 1.0,
        "lambda_att": 4.3,
        "average_ball_speed": 15.0,
        "int_dt": 0.04,
        "max_int_time": 10.0,
        "model_converge_tol": 0.01,
    }
    params["lambda_def"] = params["lambda_att"] * params["kappa_def"]
    params["lambda_gk"] = params["lambda_def"] * 3.0                     # keeper controls faster (can catch)

    sigma_term = np.sqrt(3) * params["tti_sigma"] / np.pi
    params["time_to_control_att"] = (
        time_to_control_veto * np.log(10) * (sigma_term + 1 / params["lambda_att"])
    )
    params["time_to_control_def"] = (
        time_to_control_veto * np.log(10) * (sigma_term + 1 / params["lambda_def"])
    )
    return params


# ── CORE PPCF ALGORITHM ─────────────────────────────────────────────

def _ppcf_vectorized(
    targets: np.ndarray,
    att_pos: np.ndarray, att_vel: np.ndarray,
    def_pos: np.ndarray, def_vel: np.ndarray, def_gk: np.ndarray,
    ball_pos: np.ndarray,
    params: dict,
) -> np.ndarray:
    """PPCF_att for N targets at once, shape (N,).

    The same Euler forward integration as the reference's per-target loop, run
    over every target in one pass by broadcasting, and verified identical to it
    to 3.3e-16. The integration variable is tau = t - ball_tt, so each target is
    aligned to its own ball arrival rather than to a shared clock.

    Positions must already be finite; _extract_teams drops the ones that are not.
    """
    N = len(targets)
    vmax = params["max_player_speed"]
    rt = params["reaction_time"]
    lam_a = params["lambda_att"]
    dt = params["int_dt"]
    tol = params["model_converge_tol"]
    tc_a = params["time_to_control_att"]
    tc_d = params["time_to_control_def"]
    sig_c = np.pi / np.sqrt(3.0) / params["tti_sigma"]
    Pa, Pd = len(att_pos), len(def_pos)

    # (N,)
    if ball_pos is not None and not np.any(np.isnan(ball_pos)):
        ball_tt = np.linalg.norm(targets - ball_pos, axis=1) / params["average_ball_speed"]
    else:
        ball_tt = np.zeros(N)

    # Time-to-intercept: players × targets → (P, N)
    att_r = att_pos + att_vel * rt                                     # (Pa, 2)
    att_tti = rt + np.linalg.norm(
        targets[None] - att_r[:, None], axis=2,
    ) / vmax                                                           # (Pa, N)
    def_r = def_pos + def_vel * rt                                     # (Pd, 2)
    def_tti = rt + np.linalg.norm(
        targets[None] - def_r[:, None], axis=2,
    ) / vmax                                                           # (Pd, N)

    att_min = att_tti.min(axis=0) if Pa else np.full(N, np.inf)
    def_min = def_tti.min(axis=0) if Pd else np.full(N, np.inf)

    # Shortcuts: one team is far enough ahead that the other cannot matter, so the
    # Euler integration is skipped and the extreme returned exactly. Both masks true
    # at once would need tc_att + tc_def <= 0, so the defence-first order below is a
    # guard, not a tie-break.
    result = np.full(N, np.nan)
    def_dom = (att_min - np.maximum(ball_tt, def_min)) >= tc_d
    att_dom = (def_min - np.maximum(ball_tt, att_min)) >= tc_a
    result[def_dom] = 0.0
    result[att_dom & ~def_dom] = 1.0

    eidx = np.where(np.isnan(result))[0]
    if len(eidx) == 0:
        return result

    # ── Euler integration for contested targets ──
    M = len(eidx)
    e_btt = ball_tt[eidx]                                              # (M,)
    e_att = att_tti[:, eidx]                                           # (Pa, M)
    e_def = def_tti[:, eidx]                                           # (Pd, M)

    # Active player masks per target: (P, M)
    a_act = (e_att - att_min[eidx]) < tc_a
    d_act = (e_def - def_min[eidx]) < tc_d

    # Defender lambda (GK gets higher rate): (Pd, 1)
    d_lam = np.where(def_gk, params["lambda_gk"], params["lambda_def"])[:, None]

    tau_arr = np.arange(0, params["max_int_time"], dt)
    att_tti_rel = e_att - e_btt[None, :]
    def_tti_rel = e_def - e_btt[None, :]

    pa_cum = np.zeros((Pa, M))
    pd_cum = np.zeros((Pd, M))
    tot_a = np.zeros(M)
    tot_d = np.zeros(M)
    conv = np.zeros(M, dtype=bool)

    for tau in tau_arr:
        if conv.all():
            break
        live = ~conv
        rem = 1.0 - tot_a - tot_d                                      # remainder BEFORE this step, as the reference does
        with np.errstate(over='ignore'):
            p_a = 1.0 / (1.0 + np.exp(-sig_c * (tau - att_tti_rel)))
            p_d = 1.0 / (1.0 + np.exp(-sig_c * (tau - def_tti_rel)))
        # `live` is load-bearing: rem stays positive (~0.005) after a target converges,
        # so without it a settled target would keep accumulating.
        pa_cum += (rem * p_a * lam_a * a_act * live) * dt
        pd_cum += (rem * p_d * d_lam * d_act * live) * dt
        tot_a = pa_cum.sum(axis=0)
        tot_d = pd_cum.sum(axis=0)
        conv = (tot_a + tot_d) > (1.0 - tol)

    result[eidx] = tot_a
    return result


# ── INTERNAL HELPERS ────────────────────────────────────────────────

def _extract_teams(frame_data: pd.DataFrame, att_team_id: int) -> tuple[_TeamArrays, _TeamArrays]:
    """Split one frame into (pos, vel, is_gk) arrays for attackers and defenders.

    Players the broadcast did not locate are DROPPED, not carried as NaN. One NaN
    position makes att_tti.min() NaN, which switches off both veto branches and
    returns NaN for EVERY target rather than for that player, and the mask cannot
    catch it later because NaN * False is still NaN. Dropping reproduces the
    result the frame would give without that player, bit for bit.

    The is_ball filter is what makes the `!=` below safe: the ball row is the only
    one carrying a null team_id, and a null would otherwise count as a defender.
    """
    located = frame_data["x_tracking"].notna() & frame_data["y_tracking"].notna()
    players = frame_data[(frame_data["is_ball"] == 0) & located]
    att = players[players["team_id"] == att_team_id]
    def_ = players[players["team_id"] != att_team_id]

    def _arrays(df: pd.DataFrame) -> _TeamArrays:
        pos = df[["x_tracking", "y_tracking"]].values
        vel = df[["vx", "vy"]].fillna(0).values                 # a NaN gap means unknown, so treat as at rest
        is_gk = df["is_goalkeeper"].values.astype(bool)        # defending keepers use lambda_gk
        return pos, vel, is_gk

    return _arrays(att), _arrays(def_)


# ── POINT-WISE PPCF ────────────────────────────────────────────────

def ppcf_at_targets(
    frame_data: pd.DataFrame,
    targets_meters: np.ndarray,
    att_team_id: int,
    params: dict | None = None,
) -> np.ndarray:
    """PPCF for the attacking team at each of N target positions, shape (N,).

    targets_meters is (N, 2) in pitch-centred metres; a bare (2,) point raises.
    The ball is the frame's ball row, and its absence is not an error here: the
    module docstring lists that and the four other properties of this number that
    the signature does not show.
    """
    if params is None:
        params = default_model_params()

    (att_pos, att_vel, _), (def_pos, def_vel, def_gk) = _extract_teams(
        frame_data, att_team_id
    )

    return _ppcf_vectorized(
        targets_meters, att_pos, att_vel,
        def_pos, def_vel, def_gk, get_ball_pos(frame_data), params,
    )
