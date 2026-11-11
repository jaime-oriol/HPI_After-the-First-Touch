"""Tests for s01_phases — set-piece phase grammar + competing-outcomes labelling.

The builders below pin the predicate that decides the whole project, which segment a
shot lands in, on a synthetic event stream. Each of the five segments, the window
bound, the clearance predicate and the free-kick predicate is fixed exactly and runs
on a bare clone, without the granted sample and without checking ranges on whatever
the data happens to contain.
"""
import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from s01_phases import (
    CornerPhaseGrammar, NO_DELIVERY, SEGMENTS, SetPieceGrammar, SetPieceLabel, clearing_contacts,
    corner_side, first_contact, is_after_first_touch, _aerial_rating, _is_xy, _ts_to_s,
)
from conftest import requires_sb

_OUTCOMES = {"shot_direct", "shot_flick_on", "shot_prolongation", "shot_after_clearance",
             "shot_late", "clear_counter", "clear_safe", "no_second_ball"}
_SWINGS = {"Inswinging", "Outswinging", "short/straight"}

ATT, DEF = "ATT", "DEF"
WIN = SetPieceGrammar.SECOND_BALL_WINDOW_S
CROSS_WIN = SetPieceGrammar.SHORT_CROSS_WINDOW_S


# ── Synthetic event-stream builders ─────────────────────────────────────
def _ts(sec: float) -> str:
    return f"00:{int(sec // 60):02d}:{sec % 60:06.3f}"


def _row(t: float, typ: str, team: str = ATT, **kw) -> dict:
    row = {"id": kw.pop("id", f"e{t}_{typ}_{team}"), "type.name": typ, "team.name": team,
           "team.id": 1 if team == ATT else 2, "period": 1, "minute": 0, "possession": 1,
           "timestamp": _ts(t), "location": [110.0, 40.0], "player.id": kw.pop("player", 9.0)}
    row.update(kw)
    return row


def _corner(**kw) -> dict:
    return _row(0.0, "Pass", ATT, id="CORNER", **{
        "pass.type.name": "Corner", "location": [120.0, 0.1],
        "pass.end_location": [110.0, 40.0], "pass.height.name": "High Pass",
        "pass.length": 30.0, "pass.technique.name": "Inswinging", **kw})


def _free_kick(x: float = 70.0, length: float = 25.0, height: str = "High Pass",
               y: float = 40.0) -> dict:
    return _row(0.0, "Pass", ATT, id="FK", **{
        "pass.type.name": "Free Kick", "location": [x, y],
        "pass.end_location": [110.0, 40.0], "pass.height.name": height, "pass.length": length})


def _grammar(*rows, delivery=None, kinds=("corner",)) -> SetPieceGrammar:
    return SetPieceGrammar(pd.DataFrame([delivery or _corner(), *rows]), kinds=kinds)


def _segment(*rows) -> str:
    """The segment of the single attacking shot in a synthetic sequence."""
    g = _grammar(*rows)
    segs = g.shots(g.deliveries().iloc[0])
    assert len(segs) == 1
    return segs[0].segment


# ── Pure helpers ────────────────────────────────────────────────────────
def test_ts_to_s():
    """Timestamp 'HH:MM:SS.mmm' parses to the right number of seconds."""
    assert _ts_to_s("00:00:00.000") == 0.0
    assert _ts_to_s("00:06:13.138") == pytest.approx(373.138)
    assert _ts_to_s("01:02:03.500") == pytest.approx(3723.5)


def test_is_xy():
    """_is_xy accepts a length-2 list/tuple and rejects None / str / wrong length."""
    assert _is_xy([1.0, 2.0]) and _is_xy((120.0, 0.1))
    assert not _is_xy(None) and not _is_xy("xy") and not _is_xy([1.0, 2.0, 3.0])


def test_corner_side_names_the_attackers_side():
    """y = 0 is the attackers' left touchline in StatsBomb's attacking frame, so a corner from
    there is a left corner, and the label a delivery carries follows the same rule."""
    assert corner_side(0.1) == "left" and corner_side(79.9) == "right"
    g = _grammar()
    assert g.label(g.deliveries().iloc[0]).corner_side == "left"      # _corner() kicks from y 0.1


def test_corner_side_agrees_with_the_swing_on_the_grant():
    """The side label checked against physics rather than against a convention.

    An inside-of-foot curl bends a left-footer's cross to his right and a right-footer's to his
    left, so a corner swings in towards goal only from the flag on the kicking foot's opposite
    side. Recounted over every corner with a named foot and a curl, the label has to put the
    left-footed inswingers on the right and the right-footed ones on the left. It reads the kick
    y and applies `corner_side`, so it tests the rule itself rather than the label stored in a
    table on disk."""
    path = Path(__file__).resolve().parents[1] / "data" / "derived" / "eda_deliveries.parquet"
    if not path.exists():
        pytest.skip("build_table.py has not been run")
    d = pd.read_parquet(path, columns=["set_piece_type", "delivery_y", "taker_foot", "swing"])
    d = d[(d.set_piece_type == "corner") & d.swing.isin(["Inswinging", "Outswinging"])
          & d.taker_foot.isin(["Left Foot", "Right Foot"])]
    side = d.delivery_y.map(corner_side)
    inswing = d.swing == "Inswinging"
    for foot, towards_goal_from in (("Left Foot", "right"), ("Right Foot", "left")):
        m = d.taker_foot == foot
        assert inswing[m & (side == towards_goal_from)].mean() >= 0.99
        assert (~inswing)[m & (side != towards_goal_from)].mean() >= 0.99


def test_outcome_ladder():
    """One label per delivery, keyed on the FIRST shot's segment. The last rung is named for
    what it holds: no shot and no live clearance, so no second ball, which is mostly NOT the
    defence regaining the ball (definitions.md §19)."""
    f = SetPieceGrammar._outcome
    for seg in SEGMENTS:
        assert f(seg, True, True) == f"shot_{seg}"       # a shot outranks the transition labels
    assert f(None, True, True) == "clear_counter"
    assert f(None, False, True) == "clear_safe"
    assert f(None, False, False) == "no_second_ball"


def test_after_first_touch_excludes_late():
    """`late` is out of the Q1 roll-up: 12 s downstream is open play, not a second ball."""
    assert is_after_first_touch("flick_on") and is_after_first_touch("prolongation")
    assert is_after_first_touch("after_clearance")
    assert not is_after_first_touch("direct") and not is_after_first_touch("late")


# ── The five segments, pinned exactly ───────────────────────────────────
def test_segment_direct():
    """The shot IS the first touch: only the shooter's own receipt intervenes."""
    assert _segment(_row(1.0, "Ball Receipt*", ATT, player=9.0),
                    _row(1.2, "Shot", ATT, player=9.0, id="S")) == "direct"


def test_segment_flick_on():
    """An intermediate attacking HEADER makes it a flick-on."""
    assert _segment(_row(1.0, "Ball Receipt*", ATT, player=9.0),
                    _row(1.2, "Pass", ATT, player=9.0, **{"pass.body_part.name": "Head"}),
                    _row(2.0, "Ball Receipt*", ATT, player=10.0),
                    _row(2.2, "Shot", ATT, player=10.0, id="S")) == "flick_on"


def test_segment_prolongation():
    """The same chain with a FOOT touch is a prolongation, not a flick-on."""
    assert _segment(_row(1.0, "Ball Receipt*", ATT, player=9.0),
                    _row(1.2, "Pass", ATT, player=9.0, **{"pass.body_part.name": "Right Foot"}),
                    _row(2.0, "Ball Receipt*", ATT, player=10.0),
                    _row(2.2, "Shot", ATT, player=10.0, id="S")) == "prolongation"


def test_segment_after_clearance_and_its_priority():
    """A defensive clearing contact makes it after_clearance, and OUTRANKS the key-pass
    reading: after a clearance the intervening attacking touch is a recycle, not a
    prolongation of the delivery."""
    assert _segment(_row(1.0, "Ball Receipt*", DEF, player=4.0),
                    _row(1.2, "Clearance", DEF, player=4.0),
                    _row(3.0, "Shot", ATT, player=10.0, id="S")) == "after_clearance"
    assert _segment(_row(1.0, "Ball Receipt*", DEF, player=4.0),
                    _row(1.2, "Clearance", DEF, player=4.0),
                    _row(2.0, "Pass", ATT, player=9.0, **{"pass.body_part.name": "Head"}),
                    _row(3.0, "Shot", ATT, player=10.0, id="S")) == "after_clearance"


def test_segment_late_and_the_window_bound():
    """The window is anchored on FIRST CONTACT and its upper bound is load-bearing.

    Without the bound a shot from a later possession leaks in as a second ball. A DIRECT
    shot can be late too, which one clock covers and a clearance-anchored one cannot
    express at all."""
    fc = _row(1.0, "Ball Receipt*", ATT, player=9.0)
    assert _segment(fc, _row(1.0 + WIN, "Shot", ATT, player=9.0, id="S")) == "direct"
    assert _segment(fc, _row(1.0 + WIN + 0.01, "Shot", ATT, player=9.0, id="S")) == "late"


def test_a_row_filed_on_no_touch_between_delivery_and_shot_leaves_it_direct():
    """The segment reads the same touches first_contact does (definitions.md §3-§4).

    A failed receipt is a ball nobody met, so the shot behind it is still the first touch, and a
    failed receipt after a header does not hide the header. Counting every _ON_BALL row instead
    would file both cases as a prolongation."""
    missed = {"ball_receipt.outcome.name": "Incomplete"}
    assert _segment(_row(1.0, "Ball Receipt*", ATT, player=8.0, **missed),
                    _row(1.5, "Shot", ATT, player=10.0, id="S")) == "direct"
    assert _segment(_row(1.0, "Ball Receipt*", ATT, player=9.0),
                    _row(1.2, "Pass", ATT, player=9.0, **{"pass.body_part.name": "Head"}),
                    _row(2.0, "Ball Receipt*", ATT, player=10.0, **missed),
                    _row(2.2, "Shot", ATT, player=11.0, id="S")) == "flick_on"


def test_off_ball_events_do_not_count_as_the_first_contact():
    """Pressure is not a touch, so the shot behind it is still the first contact."""
    assert _segment(_row(1.0, "Pressure", DEF, player=4.0),
                    _row(2.0, "Shot", ATT, player=10.0, id="S")) == "direct"


def test_a_shot_long_after_an_untouched_delivery_is_late():
    """A Shot is itself an on-ball touch, so an unbounded first-contact search would
    make the shot its own first contact and label it `direct` no matter how late it
    arrived. The delivery's first contact has to happen inside the episode."""
    assert _segment(_row(WIN + 5.0, "Shot", ATT, player=10.0, id="S")) == "late"


def test_segments_are_mutually_exclusive_and_exhaustive():
    """Every attacking shot lands in exactly one segment (invariants I1 and I2)."""
    g = _grammar(_row(1.0, "Ball Receipt*", ATT, player=9.0),
                 _row(1.2, "Shot", ATT, player=9.0, id="S1"),
                 _row(2.0, "Clearance", DEF, player=4.0),
                 _row(3.0, "Shot", ATT, player=10.0, id="S2"),
                 _row(20.0, "Shot", ATT, player=11.0, id="S3"))
    lab = g.label(g.deliveries().iloc[0])
    counts = [lab.n_direct, lab.n_flick_on, lab.n_prolongation, lab.n_after_clearance, lab.n_late]
    assert sum(counts) == lab.n_att_shots == 3                       # I1
    assert sum(counts[:4]) == lab.n_att_shots_in_window == 2         # I2
    assert lab.n_direct == 1 and lab.n_after_clearance == 1 and lab.n_late == 1


# ── The origin of an after-clearance shot (definitions.md §4) ───────────
def _origins(*rows) -> dict:
    """shot_id -> origin for a synthetic sequence."""
    g = _grammar(*rows)
    return {s.shot_id: s.origin for s in g.shots(g.deliveries().iloc[0])}


def test_a_prior_shot_makes_the_next_one_clearance_on_shot():
    """A shot the defence deals with leaves a different second phase from a cleared cross.

    This is the path a coach reads differently: the loose ball exists because a shot was
    stopped, not because the delivery was headed away. Its share of the after-clearance corner
    shots is generated rather than written here, in `measured.md` §frontier, block `origin`,
    and the direct shot itself keeps no origin."""
    o = _origins(_row(1.0, "Ball Receipt*", ATT, player=9.0),
                 _row(1.2, "Shot", ATT, player=9.0, id="S1"),
                 _row(2.0, "Clearance", DEF, player=4.0),
                 _row(3.0, "Shot", ATT, player=10.0, id="S2"))
    assert o["S1"] is None
    assert o["S2"] == "clearance_on_shot"


def test_a_clearance_that_is_the_first_contact_is_clearance_on_delivery():
    """Nobody attacking reached the ball: the defence dealt with the delivery itself."""
    assert _origins(_row(1.0, "Clearance", DEF, player=4.0),
                    _row(3.0, "Shot", ATT, player=10.0, id="S"))["S"] == "clearance_on_delivery"


def test_an_intervening_touch_before_the_clearance_is_clearance_on_touch():
    """A knockdown or a duel went first, so it is neither of the other two."""
    assert _origins(_row(1.0, "Ball Receipt*", ATT, player=9.0),
                    _row(2.0, "Clearance", DEF, player=4.0),
                    _row(3.0, "Shot", ATT, player=10.0, id="S"))["S"] == "clearance_on_touch"


def test_the_keeper_subtype_survives_onto_the_label():
    """`clearing_contact_type` only ever says "Goal Keeper", which cannot tell a punch
    from a save from a smother. The subtype the live whitelist already tested on travels."""
    g = _grammar(_row(1.0, "Goal Keeper", DEF, player=1.0,
                      **{"goalkeeper.type.name": "Shot Saved",
                         "goalkeeper.outcome.name": "In Play Danger"}),
                 _row(3.0, "Shot", ATT, player=10.0, id="S"))
    lab = g.label(g.deliveries().iloc[0])
    assert lab.clearing_contact_type == "Goal Keeper"
    assert lab.clearing_gk_type == "Shot Saved"
    assert lab.clearing_gk_outcome == "In Play Danger"


def test_an_outfield_clearance_carries_no_keeper_subtype():
    """The two keeper-subtype columns stay empty rather than inheriting the outfield row."""
    g = _grammar(_row(1.0, "Clearance", DEF, player=4.0),
                 _row(3.0, "Shot", ATT, player=10.0, id="S"))
    lab = g.label(g.deliveries().iloc[0])
    assert lab.cleared is True
    assert lab.clearing_gk_type is None and lab.clearing_gk_outcome is None


# ── The clearance predicate ─────────────────────────────────────────────
@pytest.mark.parametrize("gk_type,gk_outcome,expected", [
    # the OUTCOME decides, because the same type splits both ways (evidence.md §15 has the
    # rate at which an attacking touch follows each outcome)
    ("Punch", "In Play Danger", True),
    ("Punch", "In Play Safe", True),
    ("Shot Saved", "In Play Danger", True),
    ("Punch", "Punched out", False),          # out of play
    ("Shot Saved", "Saved Twice", False),
    ("Shot Saved", "Touched Out", False),     # behind for another corner
    ("Shot Saved", "Success", False),         # caught
    ("Keeper Sweeper", "Claim", False),       # keeper has it
    # an unseen spelling, and a row with no outcome at all, must FAIL CLOSED as dead
    ("Punch", "Some Unseen Provider String", False),
    ("Punch", None, False),
    # the type gate still applies: bookkeeping rows never reach the outcome test
    ("Shot Faced", "In Play Danger", False),
    ("Collected", "In Play Danger", False),
])
def test_goalkeeper_clears_only_on_a_live_ball(gk_type, gk_outcome, expected):
    """A keeper action is a clearing contact only if its OUTCOME left the ball live.

    Type alone counts catches, punches out of play and balls touched behind as second-ball
    opportunities. The `out` flag cannot arbitrate here, since it is set on almost no Goal
    Keeper rows against about a quarter of Clearance/Block (evidence.md §15 has both rates).
    The whitelist fails closed so a new provider spelling cannot silently admit a dead ball,
    which is what the last two cases pin."""
    kw = {"goalkeeper.type.name": gk_type}
    if gk_outcome is not None:
        kw["goalkeeper.outcome.name"] = gk_outcome
    g = _grammar(_row(1.0, "Goal Keeper", DEF, player=1.0, **kw))
    assert g.label(g.deliveries().iloc[0]).cleared is expected


def test_an_own_goal_is_a_goal_on_the_frontier():
    """A forced own goal puts the ball in the net, so it belongs on the frontier.

    An own goal is filed as `Own Goal For` on the benefiting team, never as a Shot, so a
    predicate written against "Shot" scores it as nothing, and counting only Shot puts
    goal-scoring corners in `clear_safe` (evidence.md §2 has the counts). It carries no
    shot-quality model, so its xg is 0.0 by choice and it has no body part. No attacker met
    this delivery, so the goal is the delivery's own: `direct`, as the tests below pin."""
    g = _grammar(_row(1.0, "Clearance", DEF, player=4.0),
                 _row(3.0, "Own Goal For", ATT, id="OG", player=None))
    segs = g.shots(g.deliveries().iloc[0])
    assert len(segs) == 1
    og = segs[0]
    assert og.is_goal is True and og.shot_outcome == "Own Goal"
    assert og.xg == 0.0 and og.shot_body_part is None
    assert og.segment == "direct" and og.after_first_touch is False
    assert g.label(g.deliveries().iloc[0]).any_goal is True


# ── The own goal takes the phase of the action that produced it ─────────
_OG = dict(id="OG", player=None)
_SAVED = {"goalkeeper.type.name": "Shot Saved", "goalkeeper.outcome.name": "In Play Safe"}


def _placed(*rows) -> dict:
    """shot_id -> segment for a synthetic sequence after one corner."""
    g = _grammar(*rows)
    return {s.shot_id: s.segment for s in g.shots(g.deliveries().iloc[0])}


def test_an_own_goal_from_a_cross_turned_in_at_the_first_contact_is_direct():
    """No attacker met the ball: the defender's contact is the deflection, not a clearance."""
    assert _placed(_row(1.0, "Clearance", DEF, player=4.0),
                   _row(1.5, "Own Goal For", ATT, **_OG))["OG"] == "direct"


def test_an_own_goal_off_a_direct_shot_is_direct():
    """The keeper's parry or a block in between does not make the shot's goal a second ball."""
    p = _placed(_row(1.0, "Ball Receipt*", ATT, player=9.0),
                _row(1.0, "Shot", ATT, player=9.0, id="S", **{"shot.outcome.name": "Wayward"}),
                _row(1.4, "Goal Keeper", DEF, player=1.0, **_SAVED),
                _row(1.5, "Block", DEF, player=4.0),
                _row(2.0, "Own Goal For", ATT, **_OG))
    assert p["S"] == "direct" and p["OG"] == "direct"


def test_an_own_goal_off_a_later_shot_takes_that_shot_s_segment():
    """A prolongation shot deflected in is a prolongation goal, whatever the defence touched."""
    p = _placed(_row(1.0, "Ball Receipt*", ATT, player=9.0),
                _row(1.2, "Pass", ATT, player=9.0, **{"pass.body_part.name": "Right Foot"}),
                _row(2.0, "Ball Receipt*", ATT, player=10.0),
                _row(2.2, "Shot", ATT, player=10.0, id="S", **{"shot.outcome.name": "Saved"}),
                _row(2.6, "Goal Keeper", DEF, player=1.0, **_SAVED),
                _row(3.0, "Own Goal For", ATT, **_OG))
    assert p["S"] == "prolongation" and p["OG"] == "prolongation"


def test_an_own_goal_after_an_attacking_pass_keeps_the_chain():
    """An attacking touch that is not a shot produced nothing on its own: the chain decides."""
    assert _placed(_row(1.0, "Ball Receipt*", ATT, player=9.0),
                   _row(1.2, "Pass", ATT, player=9.0, **{"pass.body_part.name": "Right Foot"}),
                   _row(1.5, "Block", DEF, player=4.0),
                   _row(2.0, "Own Goal For", ATT, **_OG))["OG"] == "after_clearance"


def test_an_own_goal_after_the_defence_had_the_ball_keeps_the_chain():
    """A defender who received and passed it had the ball, so the delivery did not produce it."""
    assert _placed(_row(1.0, "Clearance", DEF, player=4.0),
                   _row(1.5, "Ball Receipt*", DEF, player=5.0),
                   _row(2.0, "Pass", DEF, player=5.0),
                   _row(3.0, "Own Goal For", ATT, **_OG))["OG"] == "after_clearance"


def test_a_late_own_goal_stays_late():
    """Outside the window it is open play, whatever produced it."""
    assert _placed(_row(1.0, "Clearance", DEF, player=4.0),
                   _row(1.0 + WIN + 1.0, "Own Goal For", ATT, **_OG))["OG"] == "late"


def test_the_label_follows_the_own_goal_s_segment():
    """The delivery's outcome is keyed on its first terminal event, so the cross turned in at the
    first contact labels the corner shot_direct, not shot_after_clearance."""
    g = _grammar(_row(1.0, "Clearance", DEF, player=4.0), _row(1.5, "Own Goal For", ATT, **_OG))
    lab = g.label(g.deliveries().iloc[0])
    assert lab.outcome == "shot_direct" and lab.after_clearance_shot is False and lab.any_goal is True


# ── A shot belongs to the last delivery struck before it ────────────────
def _second_corner(t: float) -> dict:
    return _row(t, "Pass", ATT, id="CORNER2", **{
        "pass.type.name": "Corner", "location": [120.0, 0.1], "pass.end_location": [110.0, 40.0],
        "pass.height.name": "High Pass", "pass.length": 30.0})


def test_a_shot_after_the_next_corner_is_that_corner_s_alone():
    """Inside the first corner's horizon, but struck after the second: counted once, under the second."""
    g = _grammar(_row(1.0, "Clearance", DEF, player=4.0, out=True), _second_corner(12.0),
                 _row(13.0, "Ball Receipt*", ATT, player=9.0), _row(13.1, "Shot", ATT, player=9.0, id="S"))
    d = g.deliveries()
    first, second = (d[d["id"] == k].iloc[0] for k in ("CORNER", "CORNER2"))
    assert [s.shot_id for s in g.shots(first)] == []
    assert [(s.shot_id, s.segment) for s in g.shots(second)] == [("S", "direct")]
    lab = g.label(first)
    assert lab.n_att_shots == 0 and lab.n_late == 0 and lab.outcome != "shot_late"


def test_the_cut_crosses_kinds_whatever_the_grammar_was_built_for():
    """A flank free kick's grammar still stops at the corner that follows it, so the flank table
    places a shot exactly where the table of both kinds does."""
    fk = _row(0.0, "Pass", ATT, id="FK", **{"pass.type.name": "Free Kick", "location": [95.0, 5.0],
                                           "pass.end_location": [110.0, 40.0], "pass.height.name": "High Pass",
                                           "pass.length": 30.0})
    rows = [_row(1.0, "Clearance", DEF, player=4.0, out=True), _second_corner(12.0),
            _row(13.0, "Ball Receipt*", ATT, player=9.0), _row(13.1, "Shot", ATT, player=9.0, id="S")]
    alone = SetPieceGrammar(pd.DataFrame([fk, *rows]), kinds=("free_kick_flank",))
    assert alone.shots_all().empty
    both = SetPieceGrammar(pd.DataFrame([fk, *rows]), kinds=("corner", "free_kick_flank"))
    assert list(both.shots_all()["set_piece_id"]) == ["CORNER2"]


def test_the_clearance_must_sit_inside_the_episode():
    """A clearing contact after the episode closes is not this episode's clearance.

    Both first_contact and clearing_contacts are capped at one window. Uncapped, `cleared`
    fires on an event anywhere in the sequence and at_risk_horizon_s goes negative
    (evidence.md §15), a time left in the episode that is already spent."""
    late_clear = _grammar(_row(1.0, "Ball Receipt*", ATT, player=9.0),
                          _row(20.0, "Clearance", DEF, player=4.0))
    lab = late_clear.label(late_clear.deliveries().iloc[0])
    assert lab.cleared is False                      # 19 s after the first contact
    assert lab.at_risk_horizon_s is None

    inside = _grammar(_row(1.0, "Ball Receipt*", ATT, player=9.0),
                      _row(4.0, "Clearance", DEF, player=4.0))
    lab2 = inside.label(inside.deliveries().iloc[0])
    assert lab2.cleared is True
    assert lab2.at_risk_horizon_s is not None and lab2.at_risk_horizon_s >= 0


def test_no_first_contact_means_no_clearance():
    """With no touch inside the window there is no episode, so a later clearance is later play.

    A clearance inside the window would itself have been the first contact, so any clearance a
    delivery without one can find sits past it. Searched with no bound, a clearance long after
    the window would mark the delivery cleared and move it from no_second_ball to clear_safe."""
    g = _grammar(_row(WIN + 7.0, "Clearance", DEF, player=4.0))
    lab = g.label(g.deliveries().iloc[0])
    assert lab.first_contact_type is None
    assert lab.cleared is False and lab.outcome == "no_second_ball"


def test_clearing_contacts_ignores_the_attacking_team():
    """An attacker's own clearance is not a defensive clearing contact."""
    post = pd.DataFrame([_row(1.0, "Clearance", ATT, player=9.0),
                         _row(2.0, "Clearance", DEF, player=4.0)]).assign(_t=[1.0, 2.0])
    assert list(clearing_contacts(post, DEF)["team.name"]) == [DEF]
    assert clearing_contacts(post, None).empty


# ── First contact ───────────────────────────────────────────────────────
def test_clearance_out_of_play_is_not_a_clearing_contact():
    """A clearance that puts the ball out does NOT leave it live.

    definitions.md defines the segment by a clearance "that leaves the ball live", and a
    ball off the pitch leaves nothing to contest, yet such a delivery would otherwise
    be labelled cleared and carry a full at-risk horizon."""
    live = _row(1.0, "Clearance", DEF, player=4.0)
    gone = _row(1.0, "Clearance", DEF, player=4.0, out=True)
    post_live = pd.DataFrame([live]).assign(_t=[1.0])
    post_gone = pd.DataFrame([gone]).assign(_t=[1.0])
    assert len(clearing_contacts(post_live, DEF)) == 1
    assert clearing_contacts(post_gone, DEF).empty


def test_live_gk_literals_exist_in_the_export():
    """Every _LIVE_GK entry must be a spelling the provider actually writes.

    A capitalised "To" ("Saved To Post", "Shot Saved To Post") matches nothing at all:
    the export writes "Shot Saved to Post"."""
    from s01_phases import _LIVE_GK
    assert "Shot Saved to Post" in _LIVE_GK
    assert not any("To Post" in v for v in _LIVE_GK)


def test_ball_receipt_is_the_first_contact():
    """Ball Receipt* is a genuine touch. Leaving it out nearly halves the measured
    won_first_contact rate and inflates time_to_first_contact_s (docs/evidence.md
    §3 carries the before/after)."""
    g = _grammar(_row(1.0, "Ball Receipt*", DEF, player=4.0),
                 _row(2.0, "Clearance", DEF, player=4.0))
    lab = g.label(g.deliveries().iloc[0])
    assert lab.first_contact_type == "Ball Receipt*"
    assert lab.won_first_contact is False
    assert lab.time_to_first_contact_s == pytest.approx(1.0)


def test_aerial_lost_duel_is_not_the_contact():
    """An 'Aerial Lost' duel row belongs to the player who LOST the header.

    duel.outcome.name is never populated on them, and every one carries a same-timestamp
    aerial_won event on the OPPOSING team (evidence.md §15). Returning that row credits
    the losing side with the first contact, which inverts won_first_contact on every
    aerial first touch."""
    post = pd.DataFrame([
        _row(1.0, "Duel", ATT, player=9.0, **{"duel.type.name": "Aerial Lost"}),
        _row(1.0, "Clearance", DEF, player=4.0)]).assign(_t=[1.0, 1.0])
    fc = first_contact(post)
    assert fc["type.name"] == "Clearance" and fc["team.name"] == DEF

    # a Tackle duel is self-describing (duel.outcome.name is populated) and stays a contact
    tk = pd.DataFrame([_row(1.0, "Duel", ATT, player=9.0,
                            **{"duel.type.name": "Tackle",
                               "duel.outcome.name": "Won"})]).assign(_t=[1.0])
    assert first_contact(tk)["type.name"] == "Duel"


def test_aerial_rating_reads_the_winner_not_the_loser():
    """first_contact_hops_rating is the rating of the player who WON the header.

    The provider files both ratings on the loser's `Aerial Lost` row: duel.hops_rating
    is that loser's and duel.hops_opponent_rating is the winner's, and no other row type
    carries either. Since first_contact() drops the loser's row, reading duel.hops_rating
    off the event it returns is NaN on every row. The rating is recovered from the Aerial
    Lost row the other team filed at the same instant, attached to the player who actually
    made the contact. What that recovers is generated, in `measured.md` §aerial
    `coverage_pct`, over the cleared deliveries rather than over every modelled corner."""
    post = pd.DataFrame([
        _row(1.0, "Duel", ATT, player=9.0, **{"duel.type.name": "Aerial Lost",
                                              "duel.hops_rating": 0.30,
                                              "duel.hops_opponent_rating": 0.80}),
        _row(1.0, "Clearance", DEF, player=4.0)]).assign(_t=[1.0, 1.0])
    fc = first_contact(post)
    assert fc["type.name"] == "Clearance"                  # the winner's own on-ball event
    assert _aerial_rating(post, fc) == 0.80                # the winner's rating, not the loser's 0.30

    # no aerial duel at that instant -> unknown, rather than a rating borrowed from elsewhere
    solo = pd.DataFrame([_row(1.0, "Clearance", DEF, player=4.0)]).assign(_t=[1.0])
    assert _aerial_rating(solo, first_contact(solo)) is None

    # a duel by the SAME team is not this contact's aerial, so it is not read either
    same = pd.DataFrame([
        _row(1.0, "Duel", DEF, player=4.0, **{"duel.type.name": "Aerial Lost",
                                              "duel.hops_opponent_rating": 0.80}),
        _row(1.0, "Clearance", DEF, player=5.0)]).assign(_t=[1.0, 1.0])
    assert _aerial_rating(same, first_contact(same)) is None


def test_failed_receipt_is_not_a_contact():
    """A Ball Receipt* carrying an outcome is a FAILED receipt: the ball was not met."""
    post = pd.DataFrame([
        _row(1.0, "Ball Receipt*", ATT, player=9.0, **{"ball_receipt.outcome.name": "Incomplete"}),
        _row(2.0, "Clearance", DEF, player=4.0)]).assign(_t=[1.0, 2.0])
    assert first_contact(post)["type.name"] == "Clearance"


@pytest.mark.parametrize("typ,kw", [
    ("Goal Keeper", {"goalkeeper.type.name": "Shot Faced"}),
    ("Goal Keeper", {"goalkeeper.type.name": "Goal Conceded"}),
    ("Goal Keeper", {"goalkeeper.type.name": "Penalty Conceded"}),
    ("50/50", {"50_50.outcome.name": "Lost"}),
    ("50/50", {"50_50.outcome.name": "Success To Opposition"}),
])
def test_a_row_filed_on_no_touch_is_not_the_first_contact(typ, kw):
    """definitions.md §3's rule, as a test rather than as prose: a first contact is a genuine
    touch, and a row filed on a player who never met the ball is not one.

    Two more shapes the export files on a player who did NOT come away with the ball: a keeper
    row recording only that he faced or conceded a shot, and the losing side of a 50/50. Both
    types are in _ON_BALL and both sit EARLIER than the touch that followed, so admitting either
    hands the first contact to the wrong team and inverts won_first_contact on those corners.

    Over the grant neither happens to be the earliest on-ball row, which is exactly why this
    needs pinning: the masks could stop working and no figure would move until they did."""
    post = pd.DataFrame([_row(1.0, typ, DEF, player=1.0, **kw),
                         _row(2.0, "Clearance", DEF, player=4.0)]).assign(_t=[1.0, 2.0])
    assert first_contact(post)["type.name"] == "Clearance"


def test_a_keeper_row_that_did_touch_the_ball_is_the_first_contact():
    """The control for the test above: the mask has to drop the bookkeeping rows and keep the
    keeper actions that met the ball, or it would pass by excluding every Goal Keeper row."""
    post = pd.DataFrame([_row(1.0, "Goal Keeper", DEF, player=1.0,
                              **{"goalkeeper.type.name": "Punch"}),
                         _row(2.0, "Clearance", DEF, player=4.0)]).assign(_t=[1.0, 2.0])
    assert first_contact(post)["type.name"] == "Goal Keeper"


# ── Flank free kicks ────────────────────────────────────────────────────
@pytest.mark.parametrize("x,y,length,height,kept", [
    (85.0, 10.0, 25.0, "High Pass", True),
    (84.9, 10.0, 25.0, "High Pass", False),     # short of the floor
    (90.0, 40.0, 25.0, "High Pass", False),     # central, inside the box width
    (90.0, 18.0, 25.0, "High Pass", False),     # the box-width line belongs to the box
    (90.0, 62.0, 25.0, "High Pass", False),
    (90.0, 17.9, 25.0, "High Pass", True),
    (90.0, 62.1, 25.0, "High Pass", True),
    (90.0, 10.0, 25.0, "Ground Pass", False),   # the height bar still applies
    (90.0, 10.0, 8.0, "High Pass", True),       # and no length bar: the wall's block stays in
])
def test_flank_free_kick_predicate(x, y, length, height, kept):
    """The paper's free-kick population: taken from FK_FLANK_MIN_X outside the box width, in the
    air, and with no length bar, because on this population the bar drops the kicks the wall
    blocks far more often than short free kicks (constants.FK_FLANK_MIN_X has the count)."""
    g = _grammar(delivery=_free_kick(x, length, height, y), kinds=("free_kick_flank",))
    assert (len(g.deliveries()) == 1) is kept
    if kept:
        assert g.deliveries()["set_piece_type"].iloc[0] == "free_kick_flank"


def test_the_flank_predicate_reads_the_kick_and_never_where_the_ball_ended():
    """Filtering on pass.end_location would define the population by what the DEFENCE did to the
    ball, so the same kick is a delivery wherever it ended."""
    for end in ([110.0, 40.0], [60.0, 5.0]):
        kick = {**_free_kick(90.0, y=10.0), "pass.end_location": end}
        assert len(_grammar(delivery=kick, kinds=("free_kick_flank",)).deliveries()) == 1


def test_the_flank_floor_is_a_class_attribute_the_annex_can_lower():
    """The annex of the cut reads the flank bands from x = 60 on a subclass that lowers the floor, so
    the kicks below 85 keep the paper's predicate and nothing else; the paper's grammar refuses them."""
    kick = pd.DataFrame([_free_kick(70.0, 8.0, "High Pass", 10.0)])
    assert SetPieceGrammar(kick, kinds=("free_kick_flank",)).deliveries().empty

    class Wide(SetPieceGrammar):
        FLANK_MIN_X = 60.0

    got = Wide(kick, kinds=("free_kick_flank",)).deliveries()
    assert len(got) == 1 and got["set_piece_type"].iloc[0] == "free_kick_flank"


def test_a_free_kick_is_never_flagged_short():
    """The short flag's length branch would read the wall on a free kick: met that early, it was
    blocked far more often than played short, and the flag sits in X_t. A corner keeps it."""
    fk = _grammar(delivery=_free_kick(90.0, 8.0, "High Pass", 10.0), kinds=("free_kick_flank",))
    assert fk.label(fk.deliveries().iloc[0]).short_corner is False
    corner = _grammar(delivery=_short())
    assert corner.label(corner.kicks().iloc[0]).short_corner is True


# ── A short or ground corner is delivered by its cross ──────────────────
def _short(**kw) -> dict:
    """A corner played to a team-mate eight yards away."""
    return _corner(**{"pass.length": 8.0, **kw})


def _cross(t: float, team: str = ATT, **kw) -> dict:
    """A pass the provider tags as a cross, struck from the attackers' right wing."""
    return _row(t, "Pass", team, **{"pass.cross": True, "pass.height.name": "High Pass",
                                    "pass.length": 25.0, "location": [105.0, 8.0], **kw})


def test_a_short_corner_is_delivered_by_its_first_cross():
    """The first touch after a short corner is the reception of its own pass, which says nothing
    about a contest, so the delivery is the cross: the first contact, the window and the segments
    start there, and the pass before it only sets `short_corner`."""
    g = _grammar(_row(1.5, "Ball Receipt*", ATT, player=7.0), _cross(3.0, id="CROSS", player=7.0),
                 _row(4.0, "Ball Receipt*", ATT, player=9.0),
                 _row(4.2, "Shot", ATT, player=9.0, id="S"), delivery=_short())
    d = g.deliveries()
    assert list(d["id"]) == ["CROSS"] and list(d["anchor"]) == ["cross"] and g.undelivered().empty
    assert list(d["kick_id"]) == ["CORNER"] and d["kick_to_delivery_s"].iloc[0] == pytest.approx(3.0)
    lab = g.label(d.iloc[0])
    assert (lab.set_piece_id, lab.kick_id, lab.anchor, lab.short_corner) == ("CROSS", "CORNER", "cross", True)
    assert lab.first_contact_type == "Ball Receipt*" and lab.time_to_first_contact_s == pytest.approx(1.0)
    assert [(s.shot_id, s.segment) for s in g.shots(d.iloc[0])] == [("S", "direct")]
    # read from the kick instead, the same shot is a prolongation of a pass to a team-mate
    kick = g.kicks().iloc[0]
    assert [s.segment for s in g.shots(kick)] == ["prolongation"]
    assert g.label(kick).time_to_first_contact_s == pytest.approx(1.5)


@pytest.mark.parametrize("t,delivered", [(CROSS_WIN, True), (CROSS_WIN + 0.01, False)])
def test_the_cross_must_come_within_the_bound_a_first_contact_has(t, delivered):
    """A short corner is delivered by a cross within the seconds a long corner has for its first
    contact, the bound included, and by none after it."""
    g = _grammar(_row(1.0, "Ball Receipt*", ATT, player=7.0), _cross(t, id="X", player=7.0),
                 delivery=_short())
    assert (len(g.deliveries()) == 1) is delivered and (g.undelivered().empty) is delivered


def test_a_short_corner_with_no_cross_keeps_a_row_with_no_delivery():
    """The corner is still a corner taken, so a count per team holds it, but it puts no ball into the
    box: no shot, no segment and no outcome that reaches the model."""
    g = _grammar(_row(1.0, "Ball Receipt*", ATT, player=7.0), _row(2.0, "Pressure", DEF, player=4.0),
                 _row(CROSS_WIN + 2.0, "Shot", ATT, player=7.0, id="S"), delivery=_short())
    assert g.deliveries().empty and list(g.undelivered()["id"]) == ["CORNER"]
    lab = g.label_all()
    assert len(lab) == 1
    r = lab.iloc[0]
    assert (r["anchor"], r["outcome"], r["n_att_shots"], bool(r["short_corner"])) == ("none", NO_DELIVERY, 0, True)
    assert r["defending_team"] == DEF and pd.isna(r["first_contact_type"]) and not r["cleared"]
    assert g.shots_all().empty


def test_the_cross_is_the_attacking_teams_first_in_the_same_possession():
    """A defending cross, a cross after the possession changed, an untagged pass and a later cross
    are all skipped."""
    g = _grammar(_cross(2.0, DEF, id="D"), _cross(1.5, ATT, possession=2, id="P2"),
                 _row(2.5, "Pass", ATT, id="NOT_A_CROSS"), _cross(3.0, id="FIRST"), _cross(5.0, id="LATER"),
                 delivery=_short())
    assert g.first_cross(g.kicks().iloc[0])["id"] == "FIRST"
    assert list(g.deliveries()["id"]) == ["FIRST"]


def test_a_pass_the_provider_does_not_tag_as_a_cross_delivers_nothing():
    g = _grammar(_row(2.0, "Pass", ATT, id="PASS", **{"pass.height.name": "High Pass"}), delivery=_short())
    assert g.first_cross(g.kicks().iloc[0]) is None and g.deliveries().empty


def test_a_long_corner_is_its_own_delivery_whatever_follows():
    """The provider tags no corner kick as a cross, so a long corner is never replaced by a later one."""
    g = _grammar(_cross(3.0, id="X"))
    d = g.deliveries()
    assert list(d["id"]) == ["CORNER"] and list(d["anchor"]) == ["kick"] and g.undelivered().empty
    assert list(d["kick_id"]) == ["CORNER"] and d["kick_to_delivery_s"].iloc[0] == 0.0


def test_a_shot_between_the_kick_and_the_cross_belongs_to_no_delivery():
    g = _grammar(_row(1.0, "Ball Receipt*", ATT, player=7.0), _row(1.5, "Shot", ATT, player=7.0, id="EARLY"),
                 _cross(3.0, id="CROSS", player=7.0), delivery=_short())
    assert g.shots_all().empty and g.label_all()["n_att_shots"].sum() == 0


def test_the_cross_window_is_a_class_attribute_a_driver_can_widen():
    ev = pd.DataFrame([_short(), _cross(CROSS_WIN + 2.0, id="LATE")])

    class Wide(SetPieceGrammar):
        SHORT_CROSS_WINDOW_S = CROSS_WIN + 4.0

    assert SetPieceGrammar(ev).deliveries().empty
    assert list(Wide(ev).deliveries()["id"]) == ["LATE"]


def test_label_all_counts_every_corner_once_in_match_order():
    g = _grammar(_second_corner(30.0), delivery=_short())
    lab = g.label_all()
    assert list(lab["set_piece_id"]) == ["CORNER", "CORNER2"] and list(lab["anchor"]) == ["none", "kick"]
    assert lab["set_piece_id"].is_unique


def test_an_unknown_kind_is_refused_rather_than_falling_into_a_free_kick_branch():
    """A kind the grammar does not build fails up front."""
    with pytest.raises(ValueError, match="unknown"):
        SetPieceGrammar(pd.DataFrame([_free_kick(90.0, y=10.0)]), kinds=("penalty",))


def test_corners_and_flank_free_kicks_are_tagged_and_separable():
    """form_answers.md Q2 promises both populations; they must never be silently pooled."""
    ev = pd.DataFrame([_corner(), _free_kick(90.0, y=10.0),
                       _row(1.0, "Clearance", DEF, player=4.0)])
    g = SetPieceGrammar(ev, kinds=("corner", "free_kick_flank"))
    assert set(g.deliveries()["set_piece_type"]) == {"corner", "free_kick_flank"}
    assert len(g.corners()) == 1
    assert (g.deliveries()["set_piece_type"] == "free_kick_flank").sum() == 1
    assert len(CornerPhaseGrammar(ev).deliveries()) == 1        # the corner-only view


# ── Grammar on the real sample ──────────────────────────────────────────
@requires_sb
def test_corners_found(sb_events):
    """Corners are extracted with the columns the labeler needs."""
    corners = CornerPhaseGrammar(sb_events).corners()
    assert len(corners) > 0
    assert {"id", "period", "team.name", "location", "set_piece_type"} <= set(corners.columns)


@requires_sb
def test_label_schema(sb_events):
    """label_all returns one row per corner taken, the deliveries and the short corners with none,
    with exactly the SetPieceLabel fields."""
    g = CornerPhaseGrammar(sb_events)
    df = g.label_all()
    assert len(df) == len(g.corners()) + len(g.undelivered()) == len(g.kicks())
    assert set(df.columns) == {f.name for f in dataclasses.fields(SetPieceLabel)}


@requires_sb
def test_label_invariants(sb_events):
    """The labels obey the anti-leakage invariants on real data."""
    g = CornerPhaseGrammar(sb_events)
    df = g.label_all()
    none = df["anchor"] == "none"
    assert df["anchor"].isin({"kick", "cross", "none"}).all()
    assert df.loc[~none, "outcome"].isin(_OUTCOMES).all() and (df.loc[none, "outcome"] == NO_DELIVERY).all()
    assert df.loc[none | (df["anchor"] == "cross"), "short_corner"].all()
    assert (df.loc[none, "n_att_shots"] == 0).all()
    assert df["swing"].isin(_SWINGS).all()
    assert df["corner_side"].isin({"left", "right"}).all()
    assert (~df["after_clearance_shot"] | (df["n_att_shots"] > 0)).all()
    assert (df["max_xg"] >= df["after_clearance_xg"] - 1e-9).all()
    assert ((df["after_clearance_xg"] > 0) <= df["after_clearance_shot"]).all()
    assert ((df["conceded_xg"] > 0) <= df["counter_shot"]).all()
    assert (df["max_xg"] >= 0).all() and (df["conceded_xg"] >= 0).all()
    assert (df["time_to_first_contact_s"].dropna() >= 0).all()

    # the partition holds on real data, not just on the synthetic fixtures
    seg_cols = ["n_direct", "n_flick_on", "n_prolongation", "n_after_clearance", "n_late"]
    assert (df[seg_cols].sum(axis=1) == df["n_att_shots"]).all()
    assert (df[seg_cols[:4]].sum(axis=1) == df["n_att_shots_in_window"]).all()
    assert (df["after_first_touch"] == (df[["n_flick_on", "n_prolongation",
                                            "n_after_clearance"]].sum(axis=1) > 0)).all()

    sp = df["after_clearance_shot"]
    assert df.loc[~sp, "second_ball_time_s"].isna().all()
    t2 = df.loc[sp, "second_ball_time_s"]
    assert ((t2 > 0) & (t2 <= WIN + 1e-9)).all()
    # a cleared corner's remaining risk time never exceeds the episode length
    hz = df.loc[df["cleared"], "at_risk_horizon_s"].dropna()
    assert (hz <= WIN + 1e-9).all()


@requires_sb
def test_shots_all_matches_the_label_counts(sb_events):
    """The shot-grain table and the delivery-grain counts describe the same shots."""
    g = CornerPhaseGrammar(sb_events)
    shots, labels = g.shots_all(), g.label_all()
    assert len(shots) == int(labels["n_att_shots"].sum())
    assert set(shots["segment"]) <= set(SEGMENTS)
    assert (shots["after_first_touch"] == shots["segment"].map(is_after_first_touch)).all()


@requires_sb
def test_deterministic(sb_events):
    """Labelling the same events twice yields an identical frame."""
    g = CornerPhaseGrammar(sb_events)
    pd.testing.assert_frame_equal(g.label_all(), g.label_all())


@requires_sb
def test_known_corner(sb_events):
    """A worked AC Milan corner (vs Napoli): an attacking shot, no goal."""
    df = CornerPhaseGrammar(sb_events).label_all()
    row = df[df["set_piece_id"] == "63ffdb9f-f92a-5f00-abcd-ae31a260858c"]
    assert len(row) == 1
    r = row.iloc[0]
    assert r["attacking_team"] == "AC Milan"
    assert r["n_att_shots"] >= 1
    assert not r["any_goal"]
    assert r["outcome"].startswith("shot_")
