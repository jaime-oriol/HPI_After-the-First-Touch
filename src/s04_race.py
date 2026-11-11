"""s04_race — Stage 4 (RACE): the second-ball race from tracking.

The freeze-frame stages (s02) say WHO is committed; this stage, on continuous
tracking, says WHO GETS THERE, which form_answers.md Q2 assigns to the tracked matches.
For one corner it finds the first contact, then races every player to the loose
ball with a velocity-aware time-to-arrive (capped-speed travel after an
EMPIRICALLY MEASURED reaction lag, the time from first contact to first
detectable movement toward the ball) and turns those same arrival times into the
recovery probability via a soft-argmin. One model, fully our own: no third-party
pitch-control field enters the finding (the Spearman PPCF in pitch_control.py
stays as a cited motion-model lineage, printed at one point as a console
cross-check and never drawn).

Built on the granted Hudl Broadcast X&Y tracking (the 30-match Serie A sample):
2D positional data, no skeletons. That matters: an orientation-aware reach field
would need 3D shoulder angles this feed does not carry, so the arrival primitive
here is a transparent velocity-aware time-to-arrive (reaction lag measured from
tracking, then capped-speed travel).

THE BALL THIS STAGE RACES TO IS THE VENDOR'S, and the vendor marks most of it as interpolated
where it matters. Broadcast X&Y tracks the ball poorly while it is off the ground, as the
provider confirms, with no filter to recommend, and a corner delivery is airborne from
the kick to the first contact. So the three reads this stage makes off the ball channel, the
flight test, the contact and the loose-ball point, land on an `ESTIMATED` position on most of
the corners it publishes, and the recovery label read at the loose point does too. No
visibility mask is applied, because one would empty the race; the bound travels with every
figure that leaves here instead (measured.md §race carries the measured share).

Corners come from s01's grammar rather than a second predicate of our own, so the
two stages cannot disagree about what a corner is. Events are anchored to frames
on the vendor GAME clock (loader's period_elapsed_s); the video clock drifts ahead
of it in period 1 and lands the anchor before the kick. See _corner_frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import kinematics as kin
import pitch_control as pc
from constants import (CONTACT_RADIUS_M, FPS_TRACKING, LOOSE_HORIZON_S, MIN_DELIVERY_M,
                       PITCH_LENGTH_M, PITCH_WIDTH_M, PITCH_X_SB, PITCH_Y_SB,
                       RACE_SCAN_HORIZON_S, REACTION_MAX_S, REACTION_SPEED_THR,
                       RECOVERY_TAU_S)
from loader import attack_runs_rtl
from s01_phases import SetPieceGrammar
from validate import assert_no_group_leak


def find_corners(events: pd.DataFrame, kinds: tuple = ("corner",)) -> pd.DataFrame:
    """Deliveries for the race, delegated to s01's grammar so the two stages share
    ONE definition of a set piece (id, period, _t = seconds from period start, team.id).

    Corners by default; `kinds` races another s01 population through the same machinery.
    No second predicate here on purpose: a private copy would be free to drift from
    s01 and silently race a different set of deliveries than the pipeline labels."""
    return SetPieceGrammar(events, kinds=kinds).deliveries()


TAU_GRID = (0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


def anchor_in_tracking(anchor_sb: tuple, side: float) -> np.ndarray:
    """A fixed StatsBomb pitch point expressed in a corner's tracking frame, in metres.

    Events are logged on a normalised 120x80 grid with the attack towards x=120; tracking is
    metres on 105x68 about the centre spot, so the same pitch point sits at +x for one corner
    and -x for the next. `side` comes from attacking_side and multiplies BOTH axes, which is
    the 180-degree rotation rather than a mirror. The axes are rescaled separately because
    they do not share a factor (105/120 against 68/80), and both come from the pitch
    constants rather than a typed yard.

    The two y axes run opposite ways. StatsBomb's y grows from the attackers' left touchline
    to their right, and the tracking y of a team attacking towards +x grows towards their
    left, so the offset from the centre line changes sign on the way across: the ball at the
    kick frame of the tracked corners sits on the flank this puts the corner (test_s04_race).

    Exists so the sufficiency ladder (run_sufficiency) can race the tracked players to the
    same fixed point the freeze-frame arm is restricted to, which is the only way the step
    between them isolates kinematics instead of also carrying knowledge of where the ball
    came down."""
    from_goal_m = (PITCH_X_SB - float(anchor_sb[0])) * PITCH_LENGTH_M / PITCH_X_SB
    left_of_centre_m = (PITCH_Y_SB / 2.0 - float(anchor_sb[1])) * PITCH_WIDTH_M / PITCH_Y_SB
    return np.array([side * (PITCH_LENGTH_M / 2.0 - from_goal_m), side * left_of_centre_m])


def _softmax_p_att(tti: np.ndarray, is_att: np.ndarray, tau: float = RECOVERY_TAU_S) -> float:
    """P(attacking team wins the loose ball) as a soft-argmin over arrival times:
    a multinomial-logit choice on -tti (tau in seconds sets how much an arrival-
    time margin decides it). The soft version of the rule that picks the recoverer
    (argmin tti), so probability and predicted recoverer are ONE model, with no pitch
    control. Arrival-time -> probability lineage: Spearman OBSO / pitch control
    (this soft-argmin is the transparent velocity-aware time-to-arrive analogue)."""
    tti = np.asarray(tti, dtype=float)
    w = np.exp(-(tti - tti.min()) / tau)                 # min-shift keeps the exponential from underflowing
    return float(w[np.asarray(is_att, dtype=bool)].sum() / w.sum())


def _recovery_prob_att(race: pd.DataFrame, tau: float = RECOVERY_TAU_S) -> float:
    """Recovery probability for one race table (thin wrapper over _softmax_p_att)."""
    return _softmax_p_att(race["tti_s"].to_numpy(), race["is_att"].to_numpy(), tau)


def calibrate_tau(records: list[dict], taus: tuple[float, ...] = TAU_GRID, n_splits: int = 5,
                  default_tau: float = RECOVERY_TAU_S) -> dict:
    """Pick the soft-argmin temperature tau by minimizing out-of-fold log-loss on
    the ACTUAL recoverer (did the attacking team win the loose ball), GroupKFold
    by match, the same honest tuning S03 uses. Each corner's P(att) for a tau
    is a closed-form softmax over its own arrival times (no fitted weights), so an
    outer fold SELECTS tau on its train corners and SCORES it on the held-out ones;
    the adopted tau is the median fold choice. Degenerate (<2 corners, <2 match
    groups, or <2 of either class) -> keep default_tau and flag non-operative.

    Match disjointness is ASSERTED on every fold rather than trusted to the splitter, the
    way s03 asserts on each of its own split paths. The other half of that contract (a
    group key that never resolved, which compares unequal to itself and would slide through
    a set intersection on both sides at once) is checked UP FRONT instead, because sklearn
    rejects a NaN group before the fold loop starts and would otherwise raise first with a
    message about imputers.

    records: list of {tti: 1d array, is_att: bool array, y: bool|None, match_id, ...},
    one per raceable corner (see SecondBallRace.race_records). Concatenate across the
    30 tracked matches; a single match is correctly reported non-operative. Records
    with y=None (recoverer unknown, nobody tracked at the loose point) are usable
    for description downstream but carry no label, so calibration drops them here."""
    recs = [r for r in records if r is not None and r["y"] is not None
            and len(r["tti"]) >= 2 and np.isfinite(r["tti"]).all()]
    y = np.array([int(r["y"]) for r in recs])
    groups = np.array([r["match_id"] for r in recs])
    if len(recs) and pd.isna(groups).any():
        raise AssertionError(
            f"match_id is unresolved on {int(pd.isna(groups).sum())} of {len(groups)} "
            f"records, so the CV group is undefined and the split cannot be demonstrated")
    n_groups = int(len(np.unique(groups))) if len(recs) else 0
    if len(recs) < 2 or n_groups < 2 or y.sum() < 2 or (len(y) - y.sum()) < 2:
        return {"tau": default_tau, "oof_log_loss": float("nan"), "operative": False,
                "n": len(recs), "n_groups": n_groups, "fold_taus": []}

    def logloss(idx: np.ndarray, tau: float) -> float:
        """Binary log-loss over the corners in `idx` at temperature `tau`
        (each corner's P(att wins) is a closed-form softmax over its arrival times)."""
        p = np.clip([_softmax_p_att(recs[i]["tti"], recs[i]["is_att"], tau) for i in idx],
                    1e-6, 1 - 1e-6)
        yy = y[idx]
        return float(-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)))

    gkf = GroupKFold(n_splits=min(n_splits, n_groups))
    no_features = np.zeros((len(recs), 1))
    oof, fold_taus = [], []
    for tr, te in gkf.split(no_features, y, groups):
        assert_no_group_leak(groups, tr, te, "match_id")
        best = min(taus, key=lambda t: logloss(tr, t))
        fold_taus.append(best)
        oof.append(logloss(te, best))
    return {"tau": float(np.median(fold_taus)), "oof_log_loss": float(np.mean(oof)),
            "operative": True, "n": len(recs), "n_groups": n_groups, "fold_taus": fold_taus}


class SecondBallRace:
    """RACE stage over Broadcast X&Y tracking. One instance per match, reused per corner.

    Velocities are ALWAYS re-derived with the causal Savitzky-Golay rather than taken
    from the vendor's vx/vy: their smoothing is undocumented and may read future frames,
    which would leak the future into every arrival time computed here."""

    def __init__(self, tracking: pd.DataFrame, events: pd.DataFrame, params: dict | None = None,
                 tau: float = RECOVERY_TAU_S) -> None:
        self.ev = events
        self.trk = kin.compute_velocities(tracking, fps=FPS_TRACKING)
        # the only thing taken from pitch_control, and only its max_player_speed:
        # the travel cap below. No PPCF surface enters the race.
        self.params = params or pc.default_model_params()
        self.tau = tau

    # ── Locate the corner and its first contact ─────────────────────────
    def _corner_frame(self, corner: pd.Series) -> int:
        """Tracking frame_count nearest the corner event, on the vendor GAME clock.

        Measured, not assumed: in period 2 the two clocks coincide, but in period 1 the
        video clock drifts ahead of the game clock and the ball is detected at the
        anchored frame far less reliably: the video anchor lands before the kick, while
        the ball is still being placed and broadcast tracking has not picked it up."""
        period = int(corner["period"])
        target = float(corner["_t"])
        sub = self.trk[self.trk["period_id"] == period]
        return int(sub.loc[(sub["period_elapsed_s"] - target).abs().idxmin(), "frame_count"])

    def _window(self, period: int, f0: int) -> pd.DataFrame:
        """Tracking rows from the corner frame f0 to f0 + RACE_SCAN_HORIZON_S."""
        return self.trk[(self.trk["period_id"] == period) &
                        (self.trk["frame_count"].between(f0, f0 + int(RACE_SCAN_HORIZON_S * FPS_TRACKING)))]

    def first_contact(self, corner: pd.Series) -> dict | None:
        """First contact after the delivery has flown MIN_DELIVERY_M.

        None for short corners or no aerial contact in the window. The flight test is
        load-bearing: until the ball has actually left the spot the nearest body is the
        taker himself, which would register as a contact on every single corner.

        Every decision here reads the ball row as the feed emits it, and the ball is often
        `ESTIMATED` rather than `VISIBLE` on the contact and loose frames (measured.md §race
        carries the shares), which is the airborne phase the provider says its tracking
        handles worst. It bounds every arrival time, probability and label downstream, so
        the two flags travel with the contact and `race_records` persists them per race."""
        period = int(corner["period"])
        f0 = self._corner_frame(corner)
        win = self._window(period, f0)
        b0 = kin.get_ball_pos(win[win["frame_count"] == f0])
        if b0 is None:
            return None
        flew = False
        for f, g in win.groupby("frame_count"):
            ball = g[g["is_ball"] == 1]
            players = g[g["is_ball"] == 0]
            if ball.empty or players.empty:
                continue
            bxy = np.array([ball.iloc[0]["x_tracking"], ball.iloc[0]["y_tracking"]])
            if not flew:
                flew = np.linalg.norm(bxy - b0) >= MIN_DELIVERY_M
                if not flew:
                    continue
            d = np.hypot(players["x_tracking"] - bxy[0], players["y_tracking"] - bxy[1])
            if d.min() <= CONTACT_RADIUS_M:
                t = float(ball.iloc[0]["period_elapsed_s"])
                loose, loose_est = self._ball_at(period, int(f) + int(LOOSE_HORIZON_S * FPS_TRACKING), bxy)
                return {"frame": int(f), "t_s": t, "ball_xy": bxy, "loose_xy": loose,
                        "period": period,
                        "ball_estimated": bool(ball.iloc[0]["visibility"] == "ESTIMATED"),
                        "loose_estimated": loose_est}
        return None

    def _ball_at(self, period: int, frame: int, default: np.ndarray) -> tuple:
        """(ball position, whether that row is `ESTIMATED`) at a given frame, the loose-ball
        point LOOSE_HORIZON_S after contact, read as the feed emits it. With no ball row there the
        position falls back to `default` and the flag is None, so a fallback never reads as
        a detection."""
        g = self.trk[(self.trk["period_id"] == period) & (self.trk["frame_count"] == frame)]
        ball = g[g["is_ball"] == 1]
        if ball.empty:
            return default, None
        return (np.array([ball.iloc[0]["x_tracking"], ball.iloc[0]["y_tracking"]]),
                bool(ball.iloc[0]["visibility"] == "ESTIMATED"))

    # ── The race ────────────────────────────────────────────────────────
    def _reaction_lag(self, pid: str, period: int, fc: int, pos: np.ndarray, loose: np.ndarray) -> float:
        """Seconds from first contact until the player first moves toward the
        loose ball (velocity component > threshold). Capped; 0 if already going."""
        direction = loose - pos
        n = np.linalg.norm(direction)
        if n < 1e-6:
            return 0.0
        direction = direction / n
        track = self.trk[(self.trk["player_id"] == pid) & (self.trk["period_id"] == period) &
                         (self.trk["frame_count"].between(fc, fc + int(REACTION_MAX_S * FPS_TRACKING)))].sort_values("frame_count")
        for _, r in track.iterrows():
            v = np.array([r["vx"], r["vy"]])
            if np.isfinite(v).all() and float(v @ direction) > REACTION_SPEED_THR:
                return (int(r["frame_count"]) - fc) / FPS_TRACKING
        return REACTION_MAX_S

    def _arrivals(self, fc: dict, att: int, target: np.ndarray | None = None) -> pd.DataFrame:
        """Per-player race kinematics at first contact: the measured reaction lag
        toward the loose ball and the momentum drift during it. Shared so race()
        (arrival to the loose point) and reach_field() (arrival to every point)
        run the SAME model.

        `target` replaces the point the reaction lag is measured toward, and stays None on
        every path the published race takes. The sufficiency ladder (run_sufficiency) passes
        a fixed anchor instead, so that a tracked rung can be built which, like the
        freeze-frame arm it is compared against, does not know where this particular ball
        came down."""
        loose = fc["loose_xy"] if target is None else np.asarray(target, dtype=float)
        frame = self.trk[(self.trk["period_id"] == fc["period"]) & (self.trk["frame_count"] == fc["frame"])]
        players = frame[frame["is_ball"] == 0]
        rows = []
        for _, p in players.iterrows():
            pos = np.array([p["x_tracking"], p["y_tracking"]])
            vel = np.nan_to_num(np.array([p["vx"], p["vy"]], dtype=float))  # a NaN gap means no drift
            lag = self._reaction_lag(p["player_id"], fc["period"], fc["frame"], pos, loose)
            drift = pos + vel * lag
            rows.append({"player_id": p["player_id"], "team_id": p["team_id"],
                         "is_att": int(p["team_id"] == att),
                         "pos_x": float(pos[0]), "pos_y": float(pos[1]),
                         "reaction_lag_s": float(lag),
                         "drift_x": float(drift[0]), "drift_y": float(drift[1])})
        return pd.DataFrame(rows)

    def race(self, corner: pd.Series, fc: dict | None = None) -> pd.DataFrame | None:
        """Per-player race to the loose ball (None if no first contact). `fc` reuses
        an already-located first contact, as reach_field does: the window scan is the
        expensive step and summary() would otherwise pay for it twice.

        dist_to_loose measures from where the player STOOD and tti from where momentum
        drifted him to: the first is a diagnostic, only the second drives the model."""
        fc = self.first_contact(corner) if fc is None else fc
        if fc is None:
            return None
        loose = fc["loose_xy"]
        vmax = self.params["max_player_speed"]
        arr = self._arrivals(fc, corner["team.id"])
        dist_loose = np.hypot(arr["pos_x"] - loose[0], arr["pos_y"] - loose[1])
        tti = arr["reaction_lag_s"] + np.hypot(arr["drift_x"] - loose[0], arr["drift_y"] - loose[1]) / vmax
        out = pd.DataFrame({"player_id": arr["player_id"], "team_id": arr["team_id"],
                            "is_att": arr["is_att"], "dist_to_loose": dist_loose.astype(float),
                            "reaction_lag_s": arr["reaction_lag_s"], "tti_s": tti.astype(float)})
        out = out.sort_values("tti_s").reset_index(drop=True)
        out["pred_recoverer"] = (out.index == 0).astype(int)
        return out

    def reach_field(self, corner: pd.Series, fc: dict | None = None,
                    n_grid_x: int = 80, tau: float | None = None) -> tuple | None:
        """P(attacking team arrives first) over a pitch grid, from the SAME arrival
        model as race(): per-player measured reaction lag + momentum drift, then
        capped-speed travel to each point, fused by the soft-argmin. At the loose
        ball it reproduces summary()'s recovery_prob_att: one model, ours, no
        pitch control. Returns (p_att[ny, nx], xg[nx], yg[ny]) in meters-from-
        center, on linspace endpoints that span the touchlines exactly."""
        fc = self.first_contact(corner) if fc is None else fc
        if fc is None:
            return None
        tau = self.tau if tau is None else tau
        vmax = self.params["max_player_speed"]
        arr = self._arrivals(fc, corner["team.id"])
        drift = arr[["drift_x", "drift_y"]].to_numpy()
        lag = arr["reaction_lag_s"].to_numpy()
        is_att = arr["is_att"].to_numpy(dtype=bool)

        ny = int(round(n_grid_x * PITCH_WIDTH_M / PITCH_LENGTH_M))
        xg = np.linspace(-PITCH_LENGTH_M / 2, PITCH_LENGTH_M / 2, n_grid_x)
        yg = np.linspace(-PITCH_WIDTH_M / 2, PITCH_WIDTH_M / 2, ny)
        gx, gy = np.meshgrid(xg, yg)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

        d = np.linalg.norm(pts[None, :, :] - drift[:, None, :], axis=2)  # (P, G)
        tti = lag[:, None] + d / vmax                                    # (P, G)
        w = np.exp(-(tti - tti.min(axis=0, keepdims=True)) / tau)
        p_att = (w[is_att].sum(axis=0) / w.sum(axis=0)).reshape(ny, n_grid_x)
        return p_att, xg, yg

    def race_records(self, corners: pd.DataFrame, match_id: int,
                     anchor_sb: tuple | None = None,
                     at_delivery: bool = False) -> list[dict]:
        """Per-corner race records for this match: each raceable corner's player
        arrival times, attacker mask, the actual-recoverer label (None when nobody
        is tracked at the loose point: kept, flagged, never guessed), the corner
        id and contact time, and the match id (the CV group). Skips corners with no
        first contact or fewer than two racers. Concatenate across matches:
        calibrate_tau consumes the labelled subset, the driver describes them all.

        `anchor_sb` races everyone to a FIXED StatsBomb pitch point instead of to this
        corner's own loose ball, which is the anchored rung of the sufficiency ladder
        (run_sufficiency). None is the published behaviour and nothing on the run_race path
        passes it.

        `at_delivery` reads the positions at the kick instead of at the first contact and drops
        the kinematics, which is the rung that isolates what the 360 fails to see. It needs an
        anchor: there is nothing legitimate to race towards at the delivery otherwise.

        Every rung keeps the SAME corners and the SAME label. The label is read at the observed
        loose point in all of them, because a rung is a different predictor and not a
        different outcome, and a paired comparison needs the population to be identical.

        `ball_estimated_at_contact` and `ball_estimated_at_loose` carry the vendor's own
        visibility of the ball on the two frames the race reads it, so the bound every figure
        downstream inherits travels with the record rather than being asserted about it. The
        loose flag is None when no ball row exists there and the position fell back."""
        out = []
        vmax = self.params["max_player_speed"]
        for _, c in corners.iterrows():
            fc = self.first_contact(c)
            if fc is None:
                continue
            att = c["team.id"]
            target = (None if anchor_sb is None
                      else anchor_in_tracking(anchor_sb, self._attacking_side(fc, att)))
            if at_delivery:
                if target is None:
                    raise ValueError("at_delivery needs an anchor: at the kick there is no "
                                     "loose ball to race towards yet")
                arr = self._delivery_positions(c, att)
            else:
                arr = self._arrivals(fc, att, target)
            if len(arr) < 2:
                continue
            y = self._actual_recoverer(fc, att)
            loose = fc["loose_xy"] if target is None else target
            tti = (np.hypot(arr["pos_x"] - loose[0], arr["pos_y"] - loose[1]) / vmax).to_numpy() \
                if at_delivery else \
                (arr["reaction_lag_s"]
                 + np.hypot(arr["drift_x"] - loose[0], arr["drift_y"] - loose[1]) / vmax).to_numpy()
            out.append({"tti": tti, "is_att": arr["is_att"].to_numpy(dtype=bool),
                        "y": None if y is None else bool(y), "match_id": match_id,
                        "corner_id": c["id"], "t_contact_s": float(fc["t_s"]),
                        "period": int(fc["period"]),
                        "ball_estimated_at_contact": bool(fc["ball_estimated"]),
                        "ball_estimated_at_loose": fc["loose_estimated"]})
        return out

    def summary(self, corner: pd.Series, fc: dict | None = None) -> dict | None:
        """Race outcome: predicted recoverer, actual recoverer, and the recovery
        probability for the attacking team, all from our own arrival-time race
        (soft-argmin over tti), no pitch-control field.

        Pass `fc` when the caller already scanned the window, so a render that needs
        the contact frame too does not pay for the scan twice.

        `pred_matches_actual` is a SANITY flag, not an accuracy metric: prediction and
        label are both derived from the same tracking positions a second apart, so they
        are correlated by construction and this must never be quoted as validation."""
        fc = self.first_contact(corner) if fc is None else fc
        if fc is None:
            return None
        att = corner["team.id"]
        race = self.race(corner, fc)
        winner = race.iloc[0]
        actual = self._actual_recoverer(fc, att)
        return {"first_contact_s": fc["t_s"],
                "pred_recoverer_att": bool(winner["is_att"]),
                "pred_tti_s": float(winner["tti_s"]),
                "recovery_prob_att": _recovery_prob_att(race, self.tau),
                "actual_recoverer_att": actual,
                "pred_matches_actual": (actual is None) or (bool(winner["is_att"]) == actual)}

    def _delivery_positions(self, corner: pd.Series, att: int) -> pd.DataFrame:
        """Where every tracked player stands AT THE KICK, with no kinematics attached.

        The positional twin of the 360 freeze-frame, read from the same feed the race uses. It
        exists so the ladder can separate two things the anchored rung otherwise charges to one
        step: what the 360 fails to see, and what velocity and a measured lag are worth. No
        reaction lag and no momentum drift, because a snapshot has neither."""
        f0 = self._corner_frame(corner)
        frame = self.trk[(self.trk["period_id"] == int(corner["period"])) &
                         (self.trk["frame_count"] == f0)]
        players = frame[frame["is_ball"] == 0]
        return pd.DataFrame({"pos_x": players["x_tracking"].to_numpy(float),
                             "pos_y": players["y_tracking"].to_numpy(float),
                             "is_att": (players["team_id"] == att).astype(int).to_numpy()})

    def _attacking_side(self, fc: dict, att: int) -> float:
        """Which way this corner attacks on the tracking x axis, +1 or -1.

        Delegates to the loader's PER-PERIOD direction rather than reading this corner's own
        frame. That direction comes from the home keeper's mean x over the whole period, and
        it is the only rule on this feed with measured backing: the kickoff shape every
        published implementation uses reads a large share of the (match, period) slots wrong
        here (evidence.md §15), because the broadcast clock starts before kick-off
        (loader._directions_from_keeper_x). One decision per period, shared with the
        renders, rather than one per corner."""
        frame = self.trk[(self.trk["period_id"] == fc["period"]) &
                         (self.trk["frame_count"] == fc["frame"])]
        home = int(self.ev["home_team.home_team_id"].iloc[0])
        return -1.0 if attack_runs_rtl(frame, att, home) else 1.0

    def _actual_recoverer(self, fc: dict, att: int) -> bool | None:
        """Team of the player nearest the loose ball LOOSE_HORIZON_S after the contact. That point
        is the ball row as emitted, `ESTIMATED` on most published races (measured.md §race), so
        the label carries that bound as much as the prediction does."""
        frame = self.trk[(self.trk["period_id"] == fc["period"]) &
                         (self.trk["frame_count"] == fc["frame"] + int(LOOSE_HORIZON_S * FPS_TRACKING))]
        players = frame[frame["is_ball"] == 0]
        if players.empty:                                # nobody tracked there: unknown, never guessed
            return None
        d = np.hypot(players["x_tracking"] - fc["loose_xy"][0], players["y_tracking"] - fc["loose_xy"][1])
        return bool(players.loc[d.idxmin(), "team_id"] == att)
