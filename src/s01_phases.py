"""s01_phases — Stage 1: set-piece phase grammar + competing-outcomes labelling.

Machinery for the "After the First Touch" pipeline, run over the granted Hudl
StatsBomb events. It produces structure, not conclusions: from the event stream
alone it segments every corner and attacking free kick into

    delivery -> first contact -> second-ball window -> terminal outcome

and emits one label per delivery plus one row per attacking shot placed on the
frontier that gives the project its title.

THE FRONTIER (form_answers.md Q1: "when the first touch does not settle the play:
a defender heads the cross clear, a teammate flicks it on, a shot is blocked, the
keeper parries"). Every attacking shot lands in exactly one of five segments:

    direct        the shot IS the first touch
    flick_on      an intermediate ATTACKING header put it there
    prolongation  an intermediate attacking touch that was not a header
    after_clearance  a defensive clearing contact went first: the ball dropped
    late          outside the window, or the delivery produced no contact at all

The segment is derived from the EVENT CHAIN (type.name, team.name, player.id) and
never from `shot.key_pass_id`. That field is attached to the shot by the provider, so
any rule they use to assign it has already seen how the shot went. Measured on the
full grant (evidence.md §1): the chain closely reproduces the key-pass partition
on `direct` and exactly on `after_clearance`, so nothing is lost by refusing to read
it, and the partition depends on no annotation choice.

No 360 here: geometry and commitment features are Stage 2 (s02_signature).

TWO EVENT-TAXONOMY RULES, measured over the full grant (evidence.md §3
has the figures). `Ball Receipt*` is IN the on-ball set: it is the first real event
after a large share of corners, and leaving it out under-counts won_first_contact
and over-states time_to_first_contact_s. And `Goal Keeper` is SUBSETTED to live-ball
actions: taking every GK row counts Shot Faced (bookkeeping, no contact) and
Collected (the keeper ends the play), which over-counts `cleared`. A ball the
keeper caught is not a second ball.

An attacking free kick is a DELIVERY, not every free kick and not a restart. The
predicate never reads pass.end_location, which would define the population by what the
defence did to the ball, and carries no length bar: pass.length follows where the ball
was met (evidence.md §15), so a bar on it would select partly on the defence's first
touch (`_delivery_mask`).

A short or ground corner is a KICK and not a delivery. It is played to a team-mate, so the first
touch after it is a reception that says nothing about a contest, and every shot of the routine
would read as "after the first touch" of its own pass. Its delivery is the first cross the attacking
team plays in the same possession within SHORT_CROSS_WINDOW_S of the kick (`deliveries`, anchor
"cross"), and the first contact, the window and the segments are anchored there. A short corner with
no such cross puts no ball into the box: it keeps a row with no delivery (anchor "none", outcome
"no_delivery") so that a team's corners are still counted, and it has no shot, segment or model row.

Body part lives on a per-type column and there is NO duel.body_part, so `_is_header`
reads nothing on a Duel row. An aerial duel is still identifiable: `duel.type.name`
carries 'Aerial Lost', which is how first_contact() drops the loser. A last attacking
touch with no body part files the shot in `prolongation`; measured, that route barely
exists and only its passes could hide a header (definitions.md §15).

Reference: competing-outcomes / action-valuation lineage (VAEP, Decroos et al.
2019; methods_second_ball.md §1-§4); see docs/sota/VAEP.md.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional

import numpy as np
import pandas as pd

import constants
from constants import BOX_Y, FK_FLANK_MIN_X, SHORT_CORNER_MAX_LENGTH

# ── Event taxonomy (StatsBomb) the grammar reasons over ─────────────────
# `Foul Won` is deliberately NOT here: it is a whistle, not a touch, and it almost always
# credits the defending team (evidence.md §15), so admitting it biases the first-contact
# rate in one direction.
_ON_BALL = frozenset({"Ball Receipt*", "Pass", "Shot", "Clearance", "Block",
                      "Goal Keeper", "Duel", "Miscontrol", "Interception",
                      "Ball Recovery", "50/50", "Dribble"})

# Rows the export files on the player who did NOT come away with the ball, or on no
# contact at all. Spellings measured over 150 matches, never guessed.
_LOST_5050_OUTCOMES = frozenset({"Lost", "Success To Opposition"})
_GK_NO_CONTACT = frozenset({"Shot Faced", "Goal Conceded", "Penalty Conceded"})

# An own goal is filed as a PAIR: `Own Goal For` on the team that benefits, `Own Goal
# Against` on the team that conceded. Neither is a Shot, so a predicate written against
# "Shot" misses every one of them. The mirror row is bookkeeping and is never a touch.
_OWN_GOAL_FOR = "Own Goal For"
# The outcome an own goal carries on the frontier, where the readers find it (`is_own_goal`).
OWN_GOAL = "Own Goal"
# What may sit between an attacking action and the own goal it produced: the defence deflecting
# the ball, never holding it. A defender who received or passed it had the ball.
_DEFLECTING = frozenset({"Clearance", "Block", "Goal Keeper"})

_CLEAR_TYPES = frozenset({"Clearance", "Block"})
# Exact provider spellings: the export writes "Shot Saved to Post" with a lowercase "to", and a
# capitalised "To" matches nothing at all.
_LIVE_GK = frozenset({"Punch", "Save", "Shot Saved", "Keeper Sweeper", "Smother",
                      "Shot Saved to Post", "Shot Saved Off Target"})

# The TYPE alone does not decide it: the same keeper type splits both ways on outcome, and
# the `out` flag cannot arbitrate because it is almost never set on Goal Keeper rows. The
# outcomes are told apart by the rate at which an attacking touch follows within the window
# (evidence.md §15 has the rates). A WHITELIST, so an unseen provider spelling fails closed
# as dead.
_LIVE_GK_OUTCOMES = frozenset({"In Play Danger", "In Play Safe", "Clear", "Lost in Play"})

_BODY_PART_COLS = ("pass.body_part.name", "clearance.body_part.name",
                   "shot.body_part.name", "goalkeeper.body_part.name")

SEGMENTS = ("direct", "flick_on", "prolongation", "after_clearance", "late")
# PUBLIC so the drivers that need the roll-up import it instead of re-typing the three names.
AFTER_FIRST_TOUCH = frozenset({"flick_on", "prolongation", "after_clearance"})

# definitions.md §4. Which kind of touch the defence dealt with, inside after_clearance only.
ORIGINS = ("clearance_on_delivery", "clearance_on_shot", "clearance_on_touch")

OUTCOMES = ("shot_direct", "shot_flick_on", "shot_prolongation", "shot_after_clearance",
            "shot_late", "clear_counter", "clear_safe", "no_second_ball")
# The outcome of a short corner that no cross followed. It is not one of the competing outcomes: such a
# row has no delivery, so it never reaches the modelling table.
NO_DELIVERY = "no_delivery"
# Which event a label's delivery is: the kick itself, the cross that stands for a short or ground
# corner, or none when a short corner was followed by no cross.
DELIVERY_ANCHORS = ("kick", "cross", "none")

KINDS = ("corner", "free_kick_flank")


def _ts_to_s(ts: str) -> float:
    """StatsBomb 'HH:MM:SS.mmm' (within a period) -> seconds, for time arithmetic."""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _is_xy(v: Any) -> bool:
    """True if v is a usable [x, y] location (list/array of length 2, not a string)."""
    return hasattr(v, "__len__") and not isinstance(v, str) and len(v) == 2


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Column access that tolerates the sparse StatsBomb union schema."""
    return df[name] if name in df.columns else pd.Series(index=df.index, dtype="object")


def _num(v: Any) -> Optional[float]:
    """A scalar as float, or None when the provider left it out."""
    x = pd.to_numeric(v, errors="coerce")
    return None if pd.isna(x) else float(x)


def _str_or_none(v: Any) -> Optional[str]:
    """A scalar as str, or None. NaN becomes None so the column stays object, not float."""
    return None if v is None or pd.isna(v) else str(v)


def _body_part(row: pd.Series) -> Optional[str]:
    """First populated per-type body part on an event, or None."""
    for c in _BODY_PART_COLS:
        v = row.get(c)
        if isinstance(v, str):
            return v
    return None


def clearing_contacts(post: pd.DataFrame, defending_team: Optional[str],
                      before: Optional[float] = None) -> pd.DataFrame:
    """Defending-team contacts that leave the ball live and loose, in time order.

    The single source of truth for "the first touch did not settle it". s01's
    after-clearance test and s08's drop locator both reach it through
    `SetPieceGrammar._clearances`, so the two stages cannot drift apart on what a clearance is.

    A contact flagged `out` is EXCLUDED, because definitions.md defines the segment by a
    clearance "that leaves the ball live" and a ball off the pitch leaves nothing to
    contest. The flag is True-or-absent in the export, never False, so it is compared
    to True rather than tested for presence.

    The keeper branch cannot use that flag: `out` is inert on keeper rows (see
    _LIVE_GK_OUTCOMES). A keeper action is admitted on its OUTCOME instead, which is what
    actually separates a parry the attack can reach from a catch, a punch out of play or
    a ball touched behind for another corner.

    `before` bounds the search the same way first_contact is bounded: a clearance outside
    the episode is not this episode's clearance. Unbounded it can sit anywhere in the
    sequence (the longer window plus HORIZON_PAD_S) while the first contact is capped at
    SECOND_BALL_WINDOW_S, which drives at_risk_horizon_s negative (evidence.md §15)."""
    if defending_team is None or "type.name" not in post.columns:
        return post.iloc[0:0]
    t = post["type.name"]
    gk = _col(post, "goalkeeper.type.name")
    gko = _col(post, "goalkeeper.outcome.name")
    out = _col(post, "out").eq(True)
    keeper_live = (t == "Goal Keeper") & gk.isin(_LIVE_GK) & gko.isin(_LIVE_GK_OUTCOMES)
    live = (t.isin(_CLEAR_TYPES) & ~out) | keeper_live
    hit = post[live & (_col(post, "team.name") == defending_team)].sort_values("_t", kind="stable")
    return hit if before is None else hit[hit["_t"] <= before]


def genuine_touch(post: pd.DataFrame) -> pd.Series:
    """Rows that record a player meeting the ball: `_ON_BALL` minus the rows filed on no touch.

    One definition serves both of s01's questions, which touch came first and whether anything
    touched the ball between the delivery and a shot, so the two cannot disagree on what a touch
    is (definitions.md §3-§4). A Ball Receipt* carrying an outcome is a FAILED receipt (StatsBomb
    only ever writes 'Incomplete' there): the ball was not met."""
    t = _col(post, "type.name")
    failed = (t == "Ball Receipt*") & _col(post, "ball_receipt.outcome.name").notna()
    # An 'Aerial Lost' duel row is the LOSER's: every one of them carries a same-timestamp
    # aerial_won event on the opposing team (evidence.md §15), and duel.outcome.name is
    # never populated, so the type name is the outcome. The row records a player who
    # did NOT touch the ball, and the winner's own event (already on-ball, same instant)
    # is the touch. Same reasoning as the failed receipt above.
    lost_aerial = (t == "Duel") & (_col(post, "duel.type.name") == "Aerial Lost")
    # Same inversion, two more shapes the export uses: a 50/50 row is filed on each
    # contestant with its own outcome, and the losing side did not come away with the ball.
    lost_5050 = (t == "50/50") & _col(post, "50_50.outcome.name").isin(_LOST_5050_OUTCOMES)
    # Bookkeeping keeper rows carry no touch at all: `Shot Faced` is filed for facing a
    # shot, `Goal Conceded` for conceding one. Measured, they sit alone at their timestamp.
    gk_bookkeeping = ((t == "Goal Keeper")
                      & _col(post, "goalkeeper.type.name").isin(_GK_NO_CONTACT))
    return t.isin(_ON_BALL) & ~failed & ~lost_aerial & ~lost_5050 & ~gk_bookkeeping


def first_contact(post: pd.DataFrame, before: Optional[float] = None) -> Optional[pd.Series]:
    """Earliest genuine touch after the delivery, no later than `before`.

    The bound is load-bearing, not hygiene. A Shot is itself an on-ball touch, so
    without it a shot arriving twenty seconds later with nothing in between becomes
    its own 'first contact' and gets labelled `direct`, when the delivery in fact
    produced no contact and the ball had long left the episode."""
    if "type.name" not in post.columns:
        return None
    ct = post[genuine_touch(post)]
    if before is not None:
        ct = ct[ct["_t"] <= before]
    return ct.iloc[0] if len(ct) else None


def own_goal_source(between: pd.DataFrame, attacking: str) -> Optional[str]:
    """The attacking action an own goal is credited to: a shot's id, "delivery", or None.

    An own goal is not a shot, so the chain before it does not say what put the ball in the net,
    and the defender's own contact would file it after a clearance (evidence.md §2 has the
    counts). When only the defence deflecting the ball sits between the last attacking action and
    the own goal, that action produced it: the shot if it was one, the delivery itself if no
    attacker met the ball. An attacking pass, or a defender who had the ball, leaves it None and
    the chain stands. s01 and the falsification gate both place own goals through this."""
    touches = between[genuine_touch(between)]
    att = _col(touches, "team.name") == attacking
    last = float(touches.loc[att, "_t"].max()) if att.any() else -np.inf
    if not _col(touches[touches["_t"] > last], "type.name").isin(_DEFLECTING).all():
        return None
    if not att.any():
        return "delivery"
    shot = touches[att & (touches["_t"] == last) & (_col(touches, "type.name") == "Shot")]
    return str(shot["id"].iloc[0]) if len(shot) else None


def is_after_first_touch(segment: str) -> bool:
    """The Q1 roll-up.

    `late` is deliberately OUT. A shot four seconds and four touches after the
    delivery is still the second-ball episode; one at twelve seconds is open play
    from a set-piece possession. The wider figure stays recoverable from `segment`."""
    return segment in AFTER_FIRST_TOUCH


def is_own_goal(shots: pd.DataFrame) -> pd.Series:
    """The rows of a shot table that are own goals: goals the delivery produced, not shots.

    The readers of the frontier apply one rule through this: an own goal counts wherever goals are
    counted and in no ratio over shots or over xG, where a goal no model priced would lift its cell.
    The two that price a team's finishing against xG leave it out of every count, goals included, and
    the any-shot value of `run_aim_zone`'s robustness grid counts none as a shot (`definitions.md` §4).
    A table without the outcome column carries none."""
    if "shot_outcome" not in shots.columns:
        return pd.Series(False, index=shots.index)
    return shots["shot_outcome"].eq(OWN_GOAL)


def corner_side(y: Optional[float]) -> str:
    """The flag a set piece is taken from as the ATTACKING team sees it, "left" or "right".

    StatsBomb's y runs across the attacking frame from the attackers' left touchline (0) to their
    right (80). The swing settles that without trusting a convention: an inside-of-foot curl bends
    a left-footer's cross to his right, so it swings in only from the attackers' right, and the
    grant's left-footed inswingers come from y >= 40 (test_s01_phases re-counts it)."""
    return "right" if y is not None and y >= constants.PITCH_Y_MID_SB else "left"


@dataclass
class ShotSegment:
    """One attacking shot inside a set-piece sequence, placed on the frontier.

    The grain at which Q1's headline number is computed and at which the partition
    invariants are checked. Every field is T > delivery: LABEL SIDE ONLY, never a
    feature of the delivery-instant design matrix."""
    set_piece_id: str
    shot_id: str
    set_piece_type: str
    attacking_team: str
    segment: str
    after_first_touch: bool
    origin: Optional[str]
    t_since_delivery_s: float
    t_since_first_contact_s: Optional[float]
    t_since_clearance_s: Optional[float]
    n_touches_before: int
    chain_body_part: Optional[str]
    xg: float
    is_goal: bool
    shot_outcome: Optional[str]
    shot_body_part: Optional[str]
    provider_set_piece_phase: Optional[float]   # cross-check only, never a feature or a target


@dataclass
class SetPieceLabel:
    """One corner or attacking free kick, characterised from the event stream (no 360).

    Blocks are marked by the instant they are read at. Only the delivery block is
    T = delivery, and only it may reach the s03 design matrix.

    The aerial context that matters is the CLEARING contact, not the first contact:
    the first contact is usually a Ball Receipt*, which carries no body part and no
    aerial flag at all (evidence.md §9 has the coverage figures). The header
    that clears the ball is the one that decides where it drops."""
    set_piece_id: str
    set_piece_type: str
    possession: int
    period: int
    minute: int
    attacking_team: str
    defending_team: Optional[str]
    # which event is the delivery, and the corner kick it belongs to (`set_piece_id` is the delivery's)
    anchor: str
    kick_id: str
    kick_to_delivery_s: Optional[float]
    # delivery design (T = delivery)
    corner_side: str
    swing: str
    short_corner: bool
    taker_foot: Optional[str]
    # x, y and height are read at the kick. length, angle and end_x/y follow where the ball
    # was met, the first-contact point, so they stay out of X_t (definitions.md §5).
    delivery_x: Optional[float]
    delivery_y: Optional[float]
    delivery_length: Optional[float]
    delivery_angle: Optional[float]
    delivery_height: Optional[str]
    delivery_end_x: Optional[float]
    delivery_end_y: Optional[float]
    # first contact (T > delivery: label side)
    first_contact_type: Optional[str]
    first_contact_team: Optional[str]
    won_first_contact: Optional[bool]
    time_to_first_contact_s: Optional[float]
    first_contact_is_header: Optional[bool]
    first_contact_hops_rating: Optional[float]
    # attacking output on the frontier
    n_att_shots: int
    n_att_shots_in_window: int
    max_xg: float
    any_goal: bool
    after_first_touch: bool
    first_shot_segment: Optional[str]
    n_direct: int
    n_flick_on: int
    n_prolongation: int
    n_after_clearance: int
    n_late: int
    xg_after_first_touch: float
    after_clearance_shot: bool
    after_clearance_xg: float
    second_ball_time_s: Optional[float]
    # the loose-ball episode
    cleared: bool
    clearing_contact_type: Optional[str]
    # The keeper subtype the live whitelist already tested on. `clearing_contact_type` only
    # ever says "Goal Keeper", which cannot tell a punch from a save from a smother.
    clearing_gk_type: Optional[str]
    clearing_gk_outcome: Optional[str]
    clear_is_header: Optional[bool]
    clear_aerial_won: Optional[bool]
    time_to_clearance_s: Optional[float]
    at_risk_horizon_s: Optional[float]
    resolution_time_s: Optional[float]
    # defensive transition
    counter_shot: bool
    conceded_xg: float
    counter_time_s: Optional[float]
    # single competing-outcomes label
    outcome: str


class SetPieceGrammar:
    """Phase grammar over one match's events, for corners AND attacking free kicks.

    Pure event-stream reasoning; one instance per match, reused across its deliveries.

    The windows BIND to constants rather than redeclaring: the values and the reasoning
    that fixed them live in constants.py, and they stay class attributes because s08 and the
    drivers reach them through the class. The two 8.0s of the episode and the transition are
    still two decisions, not one, and neither may be collapsed into the other. The flank kind's
    taker-x floor binds the same way, so the annex of the cut lowers it on a subclass to read the
    bands below the paper's 85 with the same predicate, and a driver reads another cross window
    the same way."""

    SECOND_BALL_WINDOW_S = constants.SECOND_BALL_WINDOW_S    # episode, from FIRST CONTACT
    TRANSITION_WINDOW_S = constants.TRANSITION_WINDOW_S      # counter window, from the clearance
    SHORT_CROSS_WINDOW_S = constants.SHORT_CROSS_WINDOW_S    # a short corner's cross, from the KICK
    FLANK_MIN_X = FK_FLANK_MIN_X
    _HORIZON_PAD_S = constants.HORIZON_PAD_S

    def __init__(self, events: pd.DataFrame, kinds: tuple = ("corner",)) -> None:
        unknown = set(kinds) - set(KINDS)
        if unknown:
            raise ValueError(f"unknown set-piece kind(s) {sorted(unknown)}; valid: {KINDS}")
        ev = events.copy()
        # Seconds from the START OF THE PERIOD, not match-absolute: the StatsBomb
        # timestamp restarts each half. Every comparison below is scoped by period,
        # and this is the clock s04 anchors tracking frames on (loader.period_elapsed_s).
        ev["_t"] = ev["timestamp"].map(_ts_to_s)
        self.ev = ev.reset_index(drop=True)
        self.kinds = tuple(kinds)
        self._starts: Optional[pd.DataFrame] = None

    # ── Deliveries ──────────────────────────────────────────────────────
    def _delivery_mask(self, kind: str) -> pd.Series:
        """Rows that are a delivery of this kind.

        A corner is unambiguous. An attacking free kick is not: `pass.type.name ==
        'Free Kick'` alone is mostly restarts. `free_kick_flank`, the paper's population, is a
        kick taken from FLANK_MIN_X outside the box width and not along the ground, read
        entirely at the kick. It carries no length bar, which would read pass.length, where the
        ball was met (evidence.md §15), and which on that population drops the kicks the wall
        blocks far more often than short free kicks (constants.FK_FLANK_MIN_X has the count). A
        corner carries none either.

        The predicate does NOT read pass.end_location, because the end-in-box set is a strict
        subset whose complement differs sharply on both cleared and any-shot rates, so
        filtering on it would define the population by what the defence did. The sensitivity
        is measured in evidence.md §2.

        A direct free-kick SHOT is a Shot event, not a Pass, so it never enters here:
        it has no first touch to be "after"."""
        e = self.ev
        if kind == "corner":
            return e["pass.type.name"] == "Corner"
        x = e["location"].map(lambda v: float(v[0]) if _is_xy(v) else np.nan)
        y = e["location"].map(lambda v: float(v[1]) if _is_xy(v) else np.nan)
        m = (e["pass.type.name"] == "Free Kick") & (_col(e, "pass.height.name") != "Ground Pass")
        return m & (x >= self.FLANK_MIN_X) & ((y < BOX_Y[0]) | (y > BOX_Y[1]))

    def kicks(self) -> pd.DataFrame:
        """Every set-piece kick of the configured kinds, tagged and in match order.

        A short or ground corner is a kick here and not yet a delivery: `deliveries` replaces it by
        its cross."""
        e = self.ev
        cols = [c for c in ("id", "period", "minute", "_t", "possession",
                            "team.id", "team.name", "location", "pass.end_location",
                            "pass.height.name", "pass.technique.name",
                            "pass.body_part.name", "pass.length") if c in e.columns]
        parts = []
        for kind in self.kinds:
            m = self._delivery_mask(kind)
            if m.any():
                sub = e.loc[m, cols].copy()
                sub["set_piece_type"] = kind
                parts.append(sub)
        if not parts:
            out = e.iloc[0:0][cols].copy()
            out["set_piece_type"] = pd.Series(dtype="object")
            return out
        return pd.concat(parts).sort_values(["period", "_t"]).reset_index(drop=True)

    @staticmethod
    def is_short_corner(kick: pd.Series) -> bool:
        """A corner struck along the ground, or for less than SHORT_CORNER_MAX_LENGTH yards.

        It flags every corner not struck as an aerial delivery, so it reads "short or ground": some
        ground passes are driven into the box, and only the barred end location could tell them
        apart. A free kick is never flagged: one met that close was blocked by the wall far more
        often than played short (constants.FK_FLANK_MIN_X), and the length branch would carry that
        first touch into X_t."""
        length = pd.to_numeric(kick.get("pass.length"), errors="coerce")
        return bool(kick["set_piece_type"] == "corner"
                    and (kick.get("pass.height.name") == "Ground Pass"
                         or (pd.notna(length) and length < SHORT_CORNER_MAX_LENGTH)))

    def first_cross(self, kick: pd.Series) -> Optional[pd.Series]:
        """The attacking team's first cross after the kick, in the same possession and period, no
        later than SHORT_CROSS_WINDOW_S after it, or None.

        `pass.cross` is the provider's tag on the pass itself: it never reads where the ball ended,
        and the provider tags no corner kick as a cross, so a long corner is never its own cross."""
        e = self.ev
        m = ((e["period"] == kick["period"]) & (e["possession"] == kick["possession"])
             & (e["_t"] > kick["_t"]) & (e["_t"] <= kick["_t"] + self.SHORT_CROSS_WINDOW_S)
             & (_col(e, "team.name") == kick["team.name"]) & (_col(e, "type.name") == "Pass")
             & (_col(e, "pass.cross") == True))                                     # noqa: E712
        x = e[m].sort_values("_t", kind="stable")
        return x.iloc[0] if len(x) else None

    def deliveries(self) -> pd.DataFrame:
        """Every delivery of the configured kinds, tagged and in match order.

        A kick that is not a short or ground corner is its own delivery (`anchor` "kick"). A short or
        ground corner is replaced by its first cross, which carries the corner's id in `kick_id` and
        the seconds from the kick in `kick_to_delivery_s` (`anchor` "cross"); one with no cross is in
        `undelivered`. The id of a cross row is the cross event's, so a shot, a label and a 360 all
        key on the event the ball went in from."""
        kicks = self.kicks()
        if kicks.empty:
            return kicks.assign(anchor=pd.Series(dtype="object"), kick_id=pd.Series(dtype="object"),
                                kick_to_delivery_s=pd.Series(dtype="float64"))
        short = np.array([self.is_short_corner(k) for _, k in kicks.iterrows()])
        found = [(k, self.first_cross(k)) for _, k in kicks[short].iterrows()]
        found = [(k, x) for k, x in found if x is not None]
        crosses = self.ev.loc[[x.name for _, x in found], [c for c in kicks.columns if c != "set_piece_type"]]
        crosses = crosses.assign(set_piece_type="corner", anchor="cross", kick_id=[k["id"] for k, _ in found],
                                 kick_to_delivery_s=[float(x["_t"] - k["_t"]) for k, x in found])
        own = kicks[~short].assign(anchor="kick", kick_id=kicks.loc[~short, "id"], kick_to_delivery_s=0.0)
        return pd.concat([own, crosses]).sort_values(["period", "_t"]).reset_index(drop=True)

    def undelivered(self) -> pd.DataFrame:
        """The short or ground corners that no cross followed in time: kicks that put no ball into the box.

        They stay in the grain with `anchor` "none", so that a team's corners are counted."""
        kicks = self.kicks()
        cut = kicks[~kicks["id"].isin(self.deliveries()["kick_id"])]
        return cut.assign(anchor="none", kick_id=cut["id"], kick_to_delivery_s=np.nan).reset_index(drop=True)

    def corners(self) -> pd.DataFrame:
        """Corner deliveries only, in match order."""
        d = self.deliveries()
        return d[d["set_piece_type"] == "corner"].reset_index(drop=True) if len(d) else d

    # ── Per-delivery reasoning ──────────────────────────────────────────
    def _sequence(self, delivery: pd.Series) -> pd.DataFrame:
        """Events from the delivery forward, same period, within the horizon (the
        pad guarantees both windows fit even for a late-in-period set piece)."""
        e = self.ev
        horizon = (delivery["_t"] + max(self.SECOND_BALL_WINDOW_S, self.TRANSITION_WINDOW_S)
                   + self._HORIZON_PAD_S)
        seq = e[(e["period"] == delivery["period"]) & (e["_t"] >= delivery["_t"]) & (e["_t"] <= horizon)]
        return seq.sort_values("_t", kind="stable")

    def _next_delivery_t(self, delivery: pd.Series) -> float:
        """When the next set piece of either kind is struck in this period, or +inf.

        Over both kinds whatever the grammar was built for, so a table of one kind places a shot
        exactly where the table of both does."""
        s = self._starts
        if s is None:
            m = pd.concat([self._delivery_mask(k) for k in KINDS], axis=1).any(axis=1)
            s = self._starts = self.ev.loc[m, ["period", "_t"]]
        later = s.loc[(s["period"] == delivery["period"]) & (s["_t"] > delivery["_t"]), "_t"]
        return float(later.min()) if len(later) else np.inf

    def _episode_end(self, t_fc: Optional[float]) -> Optional[float]:
        """When this delivery's episode closes: one window after the FIRST CONTACT.

        The single place that answers it for the clearing contact and the shot
        segmentation; the first contact itself is searched within one window of the kick."""
        return None if t_fc is None else t_fc + self.SECOND_BALL_WINDOW_S

    def _clearances(self, post: pd.DataFrame, deff: Optional[str],
                    t_fc: Optional[float]) -> pd.DataFrame:
        """This delivery's clearing contacts: inside its episode, and none when it has no episode.

        With no first contact inside the window nothing opened an episode, and any clearance
        found later is later play: one inside the window would itself have been the first
        contact. `clearing_contacts` reads a missing bound as no bound, so the absent episode is
        decided here, once, for s01 and for s08, which locates its clearance through this."""
        end = self._episode_end(t_fc)
        return post.iloc[0:0] if end is None else clearing_contacts(post, deff, end)

    def _defending_team(self, seq: pd.DataFrame, attacking: str) -> Optional[str]:
        """The other team seen in the sequence (defenders have no team id here)."""
        others = [t for t in seq["team.name"].dropna().unique() if t != attacking]
        return others[0] if others else None

    def _segment_of(self, between: pd.DataFrame, attacking: str, shooter: Any,
                    t_shot: float, t_fc: Optional[float],
                    t_clear: Optional[float]) -> tuple:
        """(segment, body part of the last attacking touch) for one shot.

        Priority-ordered and TOTAL, so the five segments are mutually exclusive and
        exhaustive over a delivery's attacking shots.

        Two choices worth stating. The clearance dominates: after a defensive clearing
        contact the ball dropped, and definitions.md §4 defines the after-clearance segment
        by that contact regardless of who touched it next. And the shooter's own Ball
        Receipt* is not an intervening touch (it is the same action as the shot), without
        which nothing is ever `direct`, because a receipt precedes almost every shot.

        The window is anchored on FIRST CONTACT, not on the clearance. One clock covers all
        five segments, so a direct shot can be `late` too. And t_fc <= t_clear always, a
        clearing contact being on-ball itself, so the after-clearance test can only fire
        inside the episode. A touch here is `genuine_touch`'s, the one first_contact uses, so a
        failed receipt between the delivery and the shot leaves it direct."""
        end = self._episode_end(t_fc)
        if end is None or t_shot > end:
            return "late", None
        touches = between[genuine_touch(between)]
        own_receipt = ((_col(touches, "type.name") == "Ball Receipt*")
                       & (_col(touches, "player.id") == shooter))
        touches = touches[~own_receipt]
        att_touch = touches[_col(touches, "team.name") == attacking]
        bp = _body_part(att_touch.iloc[-1]) if len(att_touch) else None
        if t_clear is not None and t_clear < t_shot:
            return "after_clearance", bp
        if touches.empty:
            return "direct", None
        return ("flick_on" if bp == "Head" else "prolongation"), bp

    @staticmethod
    def _origin(segment: str, prior_shot: bool, clear_is_first_contact: bool,
                won_first_contact: Optional[bool]) -> Optional[str]:
        """Which kind of touch the defence dealt with, for an after-clearance shot.

        definitions.md §4, and exhaustive by construction: the clearing contact won the
        delivery's first contact, or an earlier attacking shot came in the same delivery
        (which also catches the rebound of a blocked shot), or neither.

        It belongs here and not in the descriptive driver because the prior-shot test needs
        the shot order, which this loop already has: anywhere downstream it has to be rebuilt
        by joining the shot grain back to the delivery grain. `won_first_contact` is compared
        against False rather than falsy, since it is None when no contact resolved and None
        must not pass."""
        if segment != "after_clearance":
            return None
        if clear_is_first_contact and won_first_contact is False:
            return ORIGINS[0]
        return ORIGINS[1] if prior_shot else ORIGINS[2]

    def shots(self, delivery: pd.Series) -> list:
        """Every attacking TERMINAL event this delivery produced, placed on the frontier.

        Shots and forced own goals both count, because both put the ball in the net and the
        frontier is about what the delivery produced, not about who got the credit. An own
        goal carries no shot-quality model, so its xg is 0.0 by an explicit choice rather
        than by a missing-value accident, and it has no body part to report. A predicate
        written against `type.name == "Shot"` alone scores them as nothing and puts
        goal-scoring corners in `clear_safe`.

        Each event belongs to ONE delivery, the last one struck before it. The horizon stops
        at the next set piece of either kind, or a corner taken inside the previous one's
        horizon shares the shots that follow it, and each counts twice: in its own segment and
        as the earlier delivery's `late` (evidence.md §2). An own goal takes the segment of the
        action that produced it (`own_goal_source`), because the defender's contact before it
        is the deflection, not a clearance the attack then played on. The shots after a short
        corner belong to its cross, and one struck between the kick and the cross belongs to no
        delivery. A short corner no cross followed has no delivery and so no shot."""
        att, t0, sid = delivery["team.name"], delivery["_t"], delivery["id"]
        if delivery.get("anchor", "kick") == "none":
            return []
        post = self._sequence(delivery)
        post = post[post["id"] != sid]
        deff = self._defending_team(post, att)
        fc = first_contact(post, t0 + self.SECOND_BALL_WINDOW_S)
        t_fc = float(fc["_t"]) if fc is not None else None
        clears = self._clearances(post, deff, t_fc)
        t_clear = float(clears["_t"].iloc[0]) if len(clears) else None
        won_fc = (fc["team.name"] == att) if fc is not None else None
        clear_is_fc = (t_fc is not None and t_clear is not None
                       and bool(np.isclose(t_fc, t_clear)))

        mine = (post["_t"] < self._next_delivery_t(delivery)) & (_col(post, "team.name") == att)
        shots = post[mine & (_col(post, "type.name") == "Shot")]
        owns = post[mine & (_col(post, "type.name") == _OWN_GOAL_FOR)]
        terminal = pd.concat([shots, owns]).sort_values("_t", kind="stable") if len(owns) else shots

        out, placed = [], {}
        for n_before, (_, s) in enumerate(terminal.iterrows()):
            ts = float(s["_t"])
            is_own = s.get("type.name") == _OWN_GOAL_FOR
            between = post[(post["_t"] > t0) & (post["_t"] < ts) & (post["id"] != s["id"])]
            seg, bp = self._segment_of(between, att, s.get("player.id"), ts, t_fc, t_clear)
            if is_own and seg != "late":
                src = own_goal_source(between, att)
                seg, bp = ("direct", None) if src == "delivery" else placed.get(src, (seg, bp))
            placed[s["id"]] = (seg, bp)
            xg = pd.to_numeric(s.get("shot.statsbomb_xg"), errors="coerce")
            out.append(ShotSegment(
                set_piece_id=sid, shot_id=s["id"], set_piece_type=delivery["set_piece_type"],
                attacking_team=att, segment=seg, after_first_touch=is_after_first_touch(seg),
                origin=self._origin(seg, n_before > 0, clear_is_fc, won_fc),
                t_since_delivery_s=ts - t0,
                t_since_first_contact_s=(ts - t_fc) if t_fc is not None else None,
                t_since_clearance_s=(ts - t_clear) if t_clear is not None and t_clear < ts else None,
                n_touches_before=int(genuine_touch(between).sum()),
                chain_body_part=bp,
                xg=0.0 if pd.isna(xg) else float(xg),
                is_goal=bool(is_own or s.get("shot.outcome.name") == "Goal"),
                shot_outcome=OWN_GOAL if is_own else s.get("shot.outcome.name"),
                shot_body_part=None if is_own else s.get("shot.body_part.name"),
                provider_set_piece_phase=pd.to_numeric(s.get("set_piece_phase"), errors="coerce")))
        return out

    def label(self, delivery: pd.Series) -> SetPieceLabel:
        """Segment one delivery and assign its competing-outcomes label.

        A counter needs more than a defending shot inside the window: it has to be in
        the POSSESSION CHAIN the clearance was in, or it is simply that team's next
        attack. Swept over every cleared corner (pipeline/sweep_counter.py): without that
        test most "counters" are already a different possession at every window, and past
        the shortest windows widening makes it worse. No window reaches the required
        contiguity.

        `delivery` is a row of `deliveries` (or `undelivered`), and its `anchor` says what the label
        is read from. On a kick the first contact, the window and the segments start at the kick; on
        a cross they start at the cross, and the short or ground corner before it only sets
        `short_corner`, which is then True for every row of a cross; with no delivery
        (`anchor` "none") nothing after the kick is read, and the row carries the kick's design with
        the outcome NO_DELIVERY.

        `short_corner` is `is_short_corner` of the kick. It never reads end_x, which is where the
        ball ENDED and so encodes what the defence did to it, and its length branch is the declared
        exception to the pass.length bar (definitions.md §5)."""
        att, t0, sid = delivery["team.name"], delivery["_t"], delivery["id"]
        anchor = delivery.get("anchor", "kick")
        if anchor not in DELIVERY_ANCHORS:
            raise ValueError(f"unknown anchor {anchor!r}; valid: {DELIVERY_ANCHORS}")
        seq = self._sequence(delivery)
        post = seq[seq["id"] != sid]
        deff = self._defending_team(post, att)
        if anchor == "none":
            post = post.iloc[0:0]

        # ── Delivery design (from the pass qualifiers) ──────────────────
        loc = delivery.get("location")
        cy = float(loc[1]) if _is_xy(loc) else None
        side = corner_side(cy)
        tech = delivery.get("pass.technique.name")
        swing = tech if tech in ("Inswinging", "Outswinging") else "short/straight"
        end = delivery.get("pass.end_location")
        end_x, end_y = (float(end[0]), float(end[1])) if _is_xy(end) else (None, None)
        length = pd.to_numeric(delivery.get("pass.length"), errors="coerce")
        short = anchor == "cross" or self.is_short_corner(delivery)

        # ── First contact and the clearing contact ──────────────────────
        fc = first_contact(post, t0 + self.SECOND_BALL_WINDOW_S)
        t_fc = float(fc["_t"]) if fc is not None else None
        clears = self._clearances(post, deff, t_fc)
        clear = clears.iloc[0] if len(clears) else None
        t_clear = float(clear["_t"]) if clear is not None else None
        horizon = (self.SECOND_BALL_WINDOW_S - (t_clear - t_fc)
                   if (t_clear is not None and t_fc is not None) else None)
        # Seconds from the clearance to the end of the set-piece possession, which resolves the
        # delivery: it can no longer produce an after-clearance shot. Measured over the WHOLE
        # sequence, so it routinely lands past the episode. Same possession test as the counter arm.
        t_res = None
        if t_clear is not None:
            after = post[(post["_t"] >= t_clear) & (post["possession"] != delivery["possession"])]
            if len(after):
                t_res = float(after["_t"].min()) - t_clear

        # ── The frontier ────────────────────────────────────────────────
        segs = self.shots(delivery)
        counts = {s: sum(1 for g in segs if g.segment == s) for s in SEGMENTS}
        in_win = [g for g in segs if g.segment != "late"]
        aft = [g for g in segs if g.after_first_touch]
        second = [g for g in segs if g.segment == "after_clearance"]
        first_seg = segs[0].segment if segs else None

        # ── Defensive transition ────────────────────────────────────────
        counter = post[(_col(post, "type.name") == "Shot") & (_col(post, "team.name") == deff)]
        if t_clear is not None and len(counter):
            counter = counter[(counter["_t"] >= t_clear)
                              & (counter["_t"] <= t_clear + self.TRANSITION_WINDOW_S)
                              & (counter["possession"] == clear["possession"])]
        else:
            counter = counter.iloc[0:0]
        conceded_xg = float(pd.to_numeric(_col(counter, "shot.statsbomb_xg"),
                                          errors="coerce").fillna(0.0).max()) if len(counter) else 0.0

        return SetPieceLabel(
            set_piece_id=sid, set_piece_type=delivery["set_piece_type"],
            possession=int(delivery["possession"]), period=int(delivery["period"]),
            minute=int(delivery["minute"]), attacking_team=att, defending_team=deff,
            anchor=anchor, kick_id=str(delivery.get("kick_id", sid)),
            kick_to_delivery_s=_num(delivery.get("kick_to_delivery_s", 0.0)),
            corner_side=side, swing=swing, short_corner=bool(short),
            taker_foot=delivery.get("pass.body_part.name"),
            delivery_x=(float(loc[0]) if _is_xy(loc) else None),
            delivery_y=cy,
            delivery_length=(None if pd.isna(length) else float(length)),
            delivery_angle=_num(delivery.get("pass.angle")),
            delivery_height=delivery.get("pass.height.name"),
            delivery_end_x=end_x, delivery_end_y=end_y,
            first_contact_type=(fc["type.name"] if fc is not None else None),
            first_contact_team=(fc["team.name"] if fc is not None else None),
            won_first_contact=((fc["team.name"] == att) if fc is not None else None),
            time_to_first_contact_s=((t_fc - t0) if t_fc is not None else None),
            first_contact_is_header=(_is_header(fc) if fc is not None else None),
            first_contact_hops_rating=(_aerial_rating(post, fc) if fc is not None else None),
            n_att_shots=len(segs), n_att_shots_in_window=len(in_win),
            max_xg=max((g.xg for g in segs), default=0.0),
            any_goal=any(g.is_goal for g in segs),
            after_first_touch=bool(aft), first_shot_segment=first_seg,
            n_direct=counts["direct"], n_flick_on=counts["flick_on"],
            n_prolongation=counts["prolongation"], n_after_clearance=counts["after_clearance"],
            n_late=counts["late"],
            xg_after_first_touch=max((g.xg for g in aft), default=0.0),
            after_clearance_shot=bool(second),
            after_clearance_xg=max((g.xg for g in second), default=0.0),
            second_ball_time_s=min((g.t_since_clearance_s for g in second
                                    if g.t_since_clearance_s is not None), default=None),
            cleared=clear is not None,
            clearing_contact_type=(clear["type.name"] if clear is not None else None),
            clearing_gk_type=_str_or_none(clear.get("goalkeeper.type.name")
                                          if clear is not None else None),
            clearing_gk_outcome=_str_or_none(clear.get("goalkeeper.outcome.name")
                                             if clear is not None else None),
            clear_is_header=(_is_header(clear) if clear is not None else None),
            clear_aerial_won=(_aerial_won(clear) if clear is not None else None),
            time_to_clearance_s=((t_clear - t0) if t_clear is not None else None),
            at_risk_horizon_s=horizon,
            resolution_time_s=t_res,
            counter_shot=bool(len(counter)), conceded_xg=conceded_xg,
            counter_time_s=(float(counter["_t"].min()) - t_clear if len(counter) else None),
            outcome=(NO_DELIVERY if anchor == "none"
                     else self._outcome(first_seg, bool(len(counter)), clear is not None)))

    @staticmethod
    def _outcome(first_shot_segment: Optional[str], counter_shot: bool,
                 cleared: bool) -> str:
        """One mutually-exclusive label per delivery, keyed on the segment of the
        FIRST attacking shot (temporal, so it is unambiguous when several arrive).

        There is no `goal` class. A goal label ranked above the segment would collapse
        first-touch and after-clearance goals into one bucket, which is the distinction
        this project exists to price. It stays recoverable as
        any_goal x first_shot_segment.

        The last rung is named for what it holds, no shot and no live clearance: no second
        ball. Mostly it is not the defence regaining anything (definitions.md §19)."""
        if first_shot_segment is not None:
            return f"shot_{first_shot_segment}"
        if counter_shot:
            return "clear_counter"
        if cleared:
            return "clear_safe"
        return "no_second_ball"

    def label_all(self) -> pd.DataFrame:
        """One row per delivery for this match, and one per short or ground corner that no cross
        followed (`anchor` "none"), in match order (empty frame if none).

        The rows with no delivery keep a team's corners counted: a count of corners taken is the
        denominator of every rate per corner, and a corner that put no ball into the box is still a
        corner. They carry no shot, no segment and the outcome NO_DELIVERY, and `shots_all` and the
        modelling table never hold them."""
        rows = [(c["period"], c["_t"], asdict(self.label(c)))
                for d in (self.deliveries(), self.undelivered()) for _, c in d.iterrows()]
        return pd.DataFrame([r for _, _, r in sorted(rows, key=lambda x: (x[0], x[1]))])

    def shots_all(self) -> pd.DataFrame:
        """One row per attacking shot for this match, at the frontier's own grain."""
        d = self.deliveries()
        rows = [asdict(g) for _, c in d.iterrows() for g in self.shots(c)]
        return pd.DataFrame(rows)


class CornerPhaseGrammar(SetPieceGrammar):
    """The grammar on corners alone, for the readers that take corners only
    (renders/render_signature_sb) and the tests that pin the corner path."""

    def __init__(self, events: pd.DataFrame) -> None:
        super().__init__(events, kinds=("corner",))


def _is_header(row: pd.Series) -> Optional[bool]:
    """Was this contact a header? None when the body part is not recorded.

    None rather than False, because the provider only writes a body part on some event
    types: coding "not recorded" as "not a header" would report 100% coverage over a
    column that is mostly unknown, and a null measured over a column nobody filled is a
    statement about coverage rather than about football."""
    bp = _body_part(row)
    return (bp == "Head") if bp is not None else None


def _aerial_won(row: pd.Series) -> Optional[bool]:
    """The provider's aerial-duel flag on a contact, or None.

    Never coerced to False: over corner clearances the column is True or MISSING and
    literally never False, so fillna(False) would conflate 'lost the aerial' with
    'was not an aerial at all'."""
    for c in ("clearance.aerial_won", "pass.aerial_won", "shot.aerial_won", "miscontrol.aerial_won"):
        v = row.get(c)
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
    return None


def _aerial_rating(post: pd.DataFrame, contact: pd.Series) -> Optional[float]:
    """HOPS aerial rating of the player who MADE this contact, or None.

    The provider files the pair on the loser's `Aerial Lost` row only:
    `duel.hops_rating` is that loser's, `duel.hops_opponent_rating` is the winner's.
    first_contact() returns the WINNER's own on-ball event, which carries no duel
    column at all, so reading `duel.hops_rating` straight off it is NaN on every row
    of the grant. The rating is recovered from the Aerial Lost row the other team
    filed at the same instant."""
    lost = post[(post["type.name"] == "Duel")
                & (_col(post, "duel.type.name") == "Aerial Lost")
                & (post["_t"] == float(contact["_t"]))
                & (_col(post, "team.name") != contact.get("team.name"))]
    if not len(lost):
        return None
    return pd.to_numeric(lost.iloc[0].get("duel.hops_opponent_rating"), errors="coerce")
