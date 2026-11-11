"""s08_situation — Stage 8 (SITUATION): the freeze-frame at the second-ball contest.

The PLAN stage (s02) says who a team COMMITTED at the delivery; this stage says
what the drop actually looked like: after the first defensive clearing contact the
ball goes loose, and the next touch that contests it carries its OWN 360
freeze-frame: who surrounds the dropping ball, at what distances, with what local
control. That is the middle link of form_answers.md Q2's chain (plan -> SITUATION ->
race -> value). Coverage on the Big-5 grant, measured by pipeline/audit_360.py
(data/derived/audit360.parquet has the figures): most second-ball touches and
clearances carry a 360, and most cleared corners get a situation frame from one
source or the other.

Two coach-readable products, both anti-leakage (read only at/after the drop, never
folded back into the delivery features X_t of s02/s03):
  - the FOUR STATES, crossing whether the drop zone was STAKED in the plan (an s02
    stake-drop attacker, the frontal's job, within STAKE_RADIUS of where the next touch
    contested the ball) with whether the second ball was WON (the attacking team made
    that touch): won-as-planned / staked-but-lost / rescued / conceded-open. With no
    next touch there is no contest and `staked` stays empty;
  - local situation features around the drop (bodies within NEAR_RADII, nearest att/def,
    who gets there first, local control share).

Orientation note (load-bearing): StatsBomb logs every event in the ACTING team's
attacking frame, so a defender's clearance and the attacker's recovery live in
opposite frames. Everything here is re-oriented to the attacking corner team's
frame (attack -> x=120) by TEAM, not by location, so the drop point and the
delivery roles are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist

import loader_sb as _sb
from constants import NEAR_RADII, PITCH_X_SB, PITCH_Y_SB, STAKE_RADIUS
from s01_phases import SetPieceGrammar, first_contact, _is_xy
from s02_signature import CommitmentSignature

# Local-control sampling geometry, not a football quantity: how wide the disc around the
# drop is and how finely it is sampled. Fixed relative to the drop -> build the offsets ONCE.
_DISC_R = 6.0
_DISC_STEP = 1.0
_gx = np.arange(-_DISC_R, _DISC_R + _DISC_STEP, _DISC_STEP)
_DISC = np.array([(dx, dy) for dx in _gx for dy in _gx if dx * dx + dy * dy <= _DISC_R * _DISC_R])

# A restart is the ball out of play or a whistle: nothing loose was contested, so it wins nothing.
_RESTARTS = frozenset({"Throw-in", "Free Kick", "Goal Kick", "Corner", "Kick Off"})

# staked x won -> the four coach-readable states (hudl/form_answers.md Q2)
FOUR_STATES = {(True, True): "won_as_planned",
               (True, False): "staked_but_lost",
               (False, True): "rescued",
               (False, False): "conceded_open"}


def four_state(staked: bool, won: bool) -> str:
    """Cross the plan (was the drop staked) with the outcome (was the ball won)."""
    return FOUR_STATES[(bool(staked), bool(won))]


@dataclass
class SituationLabel:
    """One corner's second-ball situation, from the drop's own freeze-frame.

    The value trio prices what the drop PRODUCED: the xG of the after-clearance
    shot(s) inside the second-ball window (max, matching s01's after_clearance_xg),
    whether any scored, and how many there were. Pure Φ, computed from events
    strictly AFTER the drop, so it may sit next to the four states as an outcome
    and must never be folded back into any delivery-instant feature set."""
    set_piece_id: str
    cleared: bool
    has_next_touch: bool
    won_second_ball: Optional[bool]
    staked: Optional[bool]
    staker_dist: Optional[float]    # nearest staker to the contested drop; NaN with no staker
    n_stakers: Optional[int]        # stake-drop attackers at the kick; None without a delivery 360
    four_state: Optional[str]
    value_xg_after: float
    value_goal_after: bool
    n_after_shots: int
    drop_x: Optional[float]
    drop_y: Optional[float]
    sit_frame_source: Optional[str]
    # local situation features (None unless a situation 360 was available)
    n_att_near3: Optional[int]
    n_def_near3: Optional[int]
    n_att_near5: Optional[int]
    n_def_near5: Optional[int]
    att_min_dist: Optional[float]
    def_min_dist: Optional[float]
    first_to_drop_att: Optional[bool]
    local_control: Optional[float]
    n_seen: Optional[int]
    n_scored: Optional[int]         # the bodies the rings and the disc score: the frame, actor out


class DropSituation:
    """Stage 8 over one match's events + 360. One instance per match, reused per corner.

    `build_situation_table` is the only way a table is produced, because `staked` needs the
    delivery frame and only that path holds one. Looping `label` without it returns every row
    with `staked` and `four_state` null, a plausible table with the two columns this stage
    exists for silently missing.

    `_clear_and_next` is private by name and load-bearing in fact: `audit_360` reads it for
    the coverage audit, `audit_tracking` for the transition window, `run_first_contact` for the
    episode its first-contact frame belongs to and `run_duel_prices` for the touch whose OBV it
    prices, all deliberately, so that what counts as a clearance is one definition and not five.
    Its signature and its bounds are therefore API, and moving them moves those drivers' records
    as well as this stage's own table."""

    def __init__(self, events: pd.DataFrame, three_sixty: pd.DataFrame,
                 kinds: tuple = ("corner",)) -> None:
        self.g = SetPieceGrammar(events, kinds=kinds)
        self.fz = three_sixty
        self.ids = set(three_sixty["event_uuid"])

    # ── Locate the clearance and the touch that contests the loose ball ─
    def _clear_and_next(self, corner: pd.Series) -> tuple:
        """(clearance, next contesting touch, attacking team) for one corner.

        The next touch comes from s01's `first_contact` and never from a predicate written
        here. A row filed on the player who LOST the ball is not a touch, and a private copy
        of that rule does not inherit s01's corrections (evidence.md §15 has what one credited
        to the loser). One taxonomy, imported, is what keeps the two stages from drifting.

        The contest sits inside the episode: without that bound drops land past the window
        (evidence.md §15). A delivery with no first contact has no episode and so no clearance,
        which s01's `_clearances` decides for both stages. The contest is NOT bounded to the
        clearance's possession. A defender who collects the loose ball opens a new possession by
        doing so, and bounding the touch to the old one would drop those recoveries from the
        label, nearly all of them the defence's, reading a lost second ball as no second ball at
        all (evidence.md §15). A restart is no contest, so a next touch that is one leaves the
        label empty."""
        seq = self.g._sequence(corner)
        post = seq[seq["id"] != corner["id"]]
        att = corner["team.name"]
        others = [t for t in post["team.name"].dropna().unique() if t != att]
        deff = others[0] if others else None
        t0 = float(corner["_t"])
        fc = first_contact(post, t0 + self.g.SECOND_BALL_WINDOW_S)
        t_fc = None if fc is None else float(fc["_t"])
        clears = self.g._clearances(post, deff, t_fc)
        if clears.empty:
            return None, None, att
        clear = clears.iloc[0]
        after = post[(post["_t"] > clear["_t"]) & (post["_t"] <= self.g._episode_end(t_fc))]
        after = after[after["location"].map(_is_xy)]
        nxt = first_contact(after)
        if nxt is not None and nxt.get("pass.type.name") in _RESTARTS:
            nxt = None
        return clear, nxt, att

    def _oriented_frame(self, event_id: str, source_team: str, att: str) -> Optional[pd.DataFrame]:
        """That event's 360 -> tidy frame oriented to attack -> x=120, with
        team_id in ATTACKING-CORNER terms (1 = corner attacker), regardless of
        which team the event actor belonged to.

        Frame columns: x, y, team_id (1=att, 0=def), is_goalkeeper, is_actor. That is s02's
        schema minus `is_ball`: the freeze-frame carries no ball row, and s02 appends the
        corner spot as one because its geometry needs the kick position, which nothing here
        does. So this frame is NOT interchangeable with s02's, and a consumer that filters
        on `is_ball` raises on it. `is_actor` is carried because the actor of this event is
        the player who just touched the ball and so sits ON the drop point by construction;
        counting him as an ordinary body makes him the nearest body almost every time
        (s02_signature excludes him for the same reason)."""
        if event_id not in self.ids:
            return None
        fz = self.fz[self.fz["event_uuid"] == event_id]
        if fz.empty:
            return None
        flip = source_team != att
        rows = []
        for p in fz.iloc[0]["freeze_frame"]:
            x, y = float(p["location"][0]), float(p["location"][1])
            if flip:
                x, y = PITCH_X_SB - x, PITCH_Y_SB - y
            # 'teammate' is relative to the actor's team; corner-attacker iff teammate matches (source==att)
            is_att = int(bool(p.get("teammate")) == (source_team == att))
            rows.append({"x": x, "y": y, "team_id": is_att,
                         "is_goalkeeper": int(bool(p.get("keeper"))),
                         "is_actor": int(bool(p.get("actor")))})
        return pd.DataFrame(rows)

    @staticmethod
    def _orient_xy(xy: Any, team: str, att: str) -> Tuple[float, float]:
        return (PITCH_X_SB - xy[0], PITCH_Y_SB - xy[1]) if team != att else (float(xy[0]), float(xy[1]))

    def label(self, corner: pd.Series, delivery_frame: Optional[pd.DataFrame] = None) -> SituationLabel:
        """Segment one corner's second-ball situation. `delivery_frame` = the s02
        commitment frame (with a 'role' column) for the STAKED test; pass None to
        skip staked when the delivery has no 360. Staked is scored against the next
        touch only: the clearance places the features when nothing contests the ball,
        but a plan cannot be scored against a contest that did not happen.

        The drop point and the situation frame come from the SAME event so they describe
        one coherent instant: the next touch is preferred (positions as the ball is
        reached), the clearance is the fallback (positions as it goes loose), and if that
        one event has no 360 the features stay BLANK rather than borrowing the other's.
        Borrowing puts bodies from before the ball travelled against a drop point it only
        reached later, which is two instants in one row, so those rows are refused and the
        cost is reported rather than hidden: `measured.md` §situation carries what the rule
        admits (`situation_frame_usable_pct`, and its split by source) against the raw
        availability `audit_360` measures in §coverage_360. The clearance itself is located
        by s01's shared helper rather than re-derived here, because two copies of "what is a
        clearance" is how the two stages drift."""
        cid = corner["id"]
        clear, nxt, att = self._clear_and_next(corner)
        blank = dict.fromkeys(
            ("n_att_near3", "n_def_near3", "n_att_near5", "n_def_near5", "att_min_dist",
             "def_min_dist", "first_to_drop_att", "local_control", "n_seen", "n_scored"), None)

        if clear is None:
            return SituationLabel(cid, False, False, None, None, None, None, None, 0.0, False, 0,
                                  None, None, None, **blank)

        after = [s for s in self.g.shots(corner) if s.segment == "after_clearance"]
        value_xg = max((s.xg for s in after), default=0.0)
        value_goal = any(s.is_goal for s in after)

        won = None if nxt is None else bool(nxt["team.name"] == att)

        drop_src = nxt if nxt is not None else clear
        drop = self._orient_xy(drop_src["location"], drop_src["team.name"], att) \
            if _is_xy(drop_src.get("location")) else None

        staked, staker_dist, n_stakers = self._stake(delivery_frame,
                                                     drop if nxt is not None else None)
        fs = four_state(staked, won) if (staked is not None and won is not None) else None

        src, feats = None, blank
        if drop is not None and drop_src["id"] in self.ids:
            src = "next" if nxt is not None else "clearance"
            frame = self._oriented_frame(drop_src["id"], drop_src["team.name"], att)
            if frame is not None and len(frame):
                feats = self._features(frame, drop)

        return SituationLabel(
            cid, True, nxt is not None, won, staked, staker_dist, n_stakers, fs,
            float(value_xg), bool(value_goal), len(after),
            (None if drop is None else float(drop[0])), (None if drop is None else float(drop[1])),
            src, **feats)

    # ── STAKED: did the plan put a stake-drop attacker where the ball fell ─
    @staticmethod
    def _stake(delivery_frame: Optional[pd.DataFrame], drop: Optional[tuple]) -> tuple:
        """(staked, staker_dist, n_stakers): did the PLAN put a stake-drop attacker within
        STAKE_RADIUS of the contested drop, how near was the nearest, and how many were there.

        All three are None without a delivery 360. Without a contested drop the staker count
        is still known and the other two are None. With no staker, staked is False and the
        distance NaN, so a False tells nobody-staked from somebody-too-far."""
        if delivery_frame is None or "role" not in delivery_frame.columns:
            return None, None, None
        stakers = delivery_frame[delivery_frame["role"] == "stake_drop"]
        if drop is None:
            return None, None, len(stakers)
        if stakers.empty:
            return False, float("nan"), 0
        d = np.hypot(stakers["x"].to_numpy(float) - drop[0], stakers["y"].to_numpy(float) - drop[1])
        return bool(d.min() <= STAKE_RADIUS), float(d.min()), len(stakers)

    # ── Local situation features around the drop ────────────────────────
    @staticmethod
    def _features(frame: pd.DataFrame, drop: tuple) -> dict:
        """Who is around the drop: ring counts at NEAR_RADII, nearest body per side, and the
        attacking share of the pitch inside a disc centred on it.

        The disc stops at the lines. The ground behind the goal line is nobody's, and sampled
        anyway it goes to whoever stands nearest the line, so a drop near goal would score the
        keeper's side for space the ball cannot use (evidence.md §15 has how often).

        The event's ACTOR is excluded from both sides. He produced this event, so he stands
        on the drop by construction and otherwise wins `first_to_drop_att` whichever side he
        is on: measured over the grant, counting him more than doubles that flag and drags it
        up towards `won_second_ball`, the label it feeds, which is the shape of a feature
        reproducing its own target (evidence.md §15 carries the before and after). `n_seen`
        still counts the whole frame, being a coverage diagnostic rather than a feature;
        `n_scored` counts the bodies the rings score, the denominator NEAR_RADII is read on.

        The KEEPER is kept, and that is the opposite of what s02 does with the same body.
        s02._def() drops him because its counts ask who can contest a delivery and a keeper
        defends a line instead; here the question is who reaches a ball already loose in the
        six-yard box, which is a contest he is in. So he enters the rings, both minimum
        distances, the control disc and `first_to_drop_att`. Changing that is a modelling
        decision rather than a repair, so it is stated and pinned by test, not altered."""
        A = frame[(frame["team_id"] == 1) & (frame["is_actor"] == 0)]
        D = frame[(frame["team_id"] == 0) & (frame["is_actor"] == 0)]
        ax = np.c_[A["x"].to_numpy(float), A["y"].to_numpy(float)] if len(A) else np.empty((0, 2))
        dx = np.c_[D["x"].to_numpy(float), D["y"].to_numpy(float)] if len(D) else np.empty((0, 2))
        da = np.hypot(ax[:, 0] - drop[0], ax[:, 1] - drop[1]) if len(ax) else np.array([])
        dd = np.hypot(dx[:, 0] - drop[0], dx[:, 1] - drop[1]) if len(dx) else np.array([])
        a_min = float(da.min()) if len(da) else np.nan
        d_min = float(dd.min()) if len(dd) else np.nan
        disc = _DISC + np.asarray(drop)
        disc = disc[((disc >= 0) & (disc <= (PITCH_X_SB, PITCH_Y_SB))).all(axis=1)]
        pts = np.vstack([p for p in (ax, dx) if len(p)]) if (len(ax) or len(dx)) else np.empty((0, 2))
        is_att = np.r_[np.ones(len(ax)), np.zeros(len(dx))].astype(bool)
        if len(pts):
            nearest = cdist(disc, pts).argmin(axis=1)
            control = float(is_att[nearest].mean())
        else:
            control = np.nan
        # nearest body to the drop is an attacker: compare finite mins, and treat
        # an empty side as "the other side is first" (None only if the frame is empty)
        if not np.isfinite(a_min) and not np.isfinite(d_min):
            first = None
        elif not np.isfinite(d_min):
            first = True
        elif not np.isfinite(a_min):
            first = False
        else:
            first = bool(a_min < d_min)
        return {
            "n_att_near3": int((da <= NEAR_RADII[0]).sum()), "n_def_near3": int((dd <= NEAR_RADII[0]).sum()),
            "n_att_near5": int((da <= NEAR_RADII[1]).sum()), "n_def_near5": int((dd <= NEAR_RADII[1]).sum()),
            "att_min_dist": a_min, "def_min_dist": d_min, "first_to_drop_att": first,
            "local_control": control, "n_seen": int(len(frame)), "n_scored": int(len(da) + len(dd))}


def _match_situation_rows(mid: int, kinds: tuple = ("corner",)) -> list:
    """One match's situation rows (dicts): the build_situation_table worker."""
    ev = _sb.load_events(mid)
    fz = _sb.load_360(mid)
    ids = set(fz["event_uuid"])
    ds = DropSituation(ev, fz, kinds)
    rows = []
    for _, c in ds.g.deliveries().iterrows():
        # roles for the STAKED test, available only when the delivery itself has a 360
        delivery_frame = (CommitmentSignature.from_corner(c, fz).frame
                          if c["id"] in ids else None)
        lab = ds.label(c, delivery_frame)
        if not lab.cleared:
            continue
        rows.append({**asdict(lab), "match_id": mid})
    return rows


def build_situation_table(match_ids: list, n_jobs: int = 1,
                          kinds: tuple = ("corner",)) -> pd.DataFrame:
    """SITUATION table: one row per cleared delivery of `kinds` = second-ball situation + the
    four-states label (+ delivery 360 roles for STAKED when the delivery has a 360)
    + the value trio, plus match_id (the CV group). Enriches with situation features
    where the drop's own 360 is available; leaves them NaN (reported) where it is not.

    Workers return dicts and the parent builds ONE DataFrame in input-match order,
    so the parallel build is bit-identical to the serial one: per-match frames
    would infer dtypes per match, and row order decides GroupKFold membership."""
    if n_jobs == 1:
        chunks = [_match_situation_rows(m, kinds) for m in match_ids]
    else:
        chunks = Parallel(n_jobs=n_jobs)(delayed(_match_situation_rows)(m, kinds)
                                         for m in match_ids)
    return pd.DataFrame([r for rows in chunks for r in rows])
