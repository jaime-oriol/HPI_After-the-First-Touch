"""Tests for s08_situation — SITUATION stage (second-ball drop freeze-frame).

Synthetic tests pin the four-states logic, the team-based orientation, the staked
test and the local features exactly; the real-sample test runs the full table
builder over the cached Serie A sample and skips without it.
"""
from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from s08_situation import (
    DropSituation, SituationLabel, build_situation_table, four_state, FOUR_STATES,
)
# STAKE_RADIUS from its owner, not through s08's import of it: a re-export nobody declared
# is not API, and reading it from here keeps the test pinned to the value s08 actually uses.
from constants import PITCH_X_SB, PITCH_Y_SB, STAKE_RADIUS
from conftest import requires_sb
from test_s01_phases import ATT, DEF, WIN, _corner, _row

_STATES = {"won_as_planned", "staked_but_lost", "rescued", "conceded_open"}


# ── Four states (pure) ──────────────────────────────────────────────────
def test_four_state_map():
    """The staked x won cross maps to the four coach-readable states, and FOUR_STATES covers exactly them."""
    assert four_state(True, True) == "won_as_planned"      # staked the drop and won it -> held as planned
    assert four_state(True, False) == "staked_but_lost"    # staked but lost -> a speed problem, not the plan
    assert four_state(False, True) == "rescued"            # not staked yet won -> athletic, not repeatable
    assert four_state(False, False) == "conceded_open"     # not staked and lost -> a design failure
    assert set(FOUR_STATES.values()) == _STATES


# ── Orientation: flip by TEAM, not by location ──────────────────────────
def test_orient_xy_by_team():
    """_orient_xy flips a point into the attacking frame by TEAM (not by pitch location)."""
    xy = (10.0, 30.0)
    # same team as the corner attacker -> unchanged
    assert DropSituation._orient_xy(xy, "A", "A") == (10.0, 30.0)
    # defending team's event -> flipped into the attacking frame
    fx, fy = DropSituation._orient_xy(xy, "B", "A")
    assert (fx, fy) == (PITCH_X_SB - 10.0, PITCH_Y_SB - 30.0)    # point-reflected through the pitch centre


# ── STAKED: a plan stake-drop attacker standing at the drop ─────────────
def _delivery(roles_xy: list[tuple[str, float, float]]) -> pd.DataFrame:
    """Build a minimal delivery frame (x, y, role rows) for the STAKED test."""
    return pd.DataFrame([{"x": x, "y": y, "role": r} for (r, x, y) in roles_xy])


def test_stake_scores_the_nearest_staker_and_counts_them():
    """Staked only when a plan stake-drop attacker stands within STAKE_RADIUS of the drop. The
    distance travels so the radius can be re-measured, and the staker count tells a False with
    nobody staking from one with somebody too far; None throughout without a delivery 360."""
    dframe = _delivery([("stake_drop", 108.0, 40.0), ("attack_cross", 116.0, 40.0)])
    staked, dist, n = DropSituation._stake(dframe, (109.0, 41.0))
    assert staked is True and dist == pytest.approx(2 ** 0.5) and n == 1
    assert DropSituation._stake(dframe, (108.0 + STAKE_RADIUS + 5, 40.0)) == (False, STAKE_RADIUS + 5, 1)
    nobody = DropSituation._stake(_delivery([("attack_cross", 116.0, 40.0)]), (108.0, 40.0))
    assert nobody[0] is False and np.isnan(nobody[1]) and nobody[2] == 0
    assert DropSituation._stake(dframe, None) == (None, None, 1)       # no contested drop
    assert DropSituation._stake(None, (108.0, 40.0)) == (None, None, None)


# ── Local features around the drop ──────────────────────────────────────
def _frame(players: list[tuple]) -> pd.DataFrame:
    """Build a situation freeze-frame for the feature test. Rows are (x, y, team_id)
    or (x, y, team_id, is_actor); nobody is a keeper and nobody is the actor by default."""
    return pd.DataFrame([{"x": p[0], "y": p[1], "team_id": p[2], "is_goalkeeper": 0,
                          "is_actor": p[3] if len(p) > 3 else 0} for p in players])


def test_features_counts_and_control():
    """Local features count bodies in the 3/5 yd rings, pick the nearest side, and return a control share in [0,1]."""
    drop = (100.0, 40.0)
    frame = _frame([
        (101.0, 40.0, 1),      # attacker ~1 yd -> near3, nearest
        (104.0, 40.0, 1),      # attacker ~4 yd -> near5 only
        (100.0, 42.0, 0),      # defender ~2 yd -> near3
        (60.0, 40.0, 0),       # far defender
    ])
    f = DropSituation._features(frame, drop)
    assert f["n_att_near3"] == 1 and f["n_att_near5"] == 2               # one att in 3 yd, both in 5 yd
    assert f["n_def_near3"] == 1                                         # only the ~2 yd defender
    assert f["att_min_dist"] < f["def_min_dist"] and f["first_to_drop_att"] is True
    assert 0.0 <= f["local_control"] <= 1.0
    assert f["n_seen"] == 4 and f["n_scored"] == 4


def test_features_all_attackers_full_control():
    """With only attackers in the frame the drop disc is fully attacker-controlled and they get there first."""
    frame = _frame([(100.0, 40.0, 1), (101.0, 41.0, 1)])
    f = DropSituation._features(frame, (100.0, 40.0))
    assert f["local_control"] == 1.0 and f["first_to_drop_att"] is True  # every disc point is nearest an attacker


def test_the_control_disc_stops_at_the_goal_line():
    """The defender on his goal line is nearer the drop, but the attacker is nearer most of the
    playable ground around it. Sampled past the line, the disc would hand the defender the strip
    behind the goal and put the attacker under half."""
    f = DropSituation._features(_frame([(114.0, 40.0, 1), (120.0, 40.0, 0)]), (117.5, 40.0))
    assert f["first_to_drop_att"] is False
    assert f["local_control"] > 0.5


def test_features_exclude_the_actor():
    """The actor of the situation event is NOT counted as a body near the drop.

    He made the event, so he stands on the drop point by construction and would win
    first_to_drop_att for whichever side he is on, regardless of the contest."""
    drop = (100.0, 40.0)
    # the actor is a defender sitting exactly on the drop; the only real body is an attacker 2 yd away
    frame = _frame([(100.0, 40.0, 0, 1), (102.0, 40.0, 1, 0)])
    f = DropSituation._features(frame, drop)
    assert f["n_def_near3"] == 0                       # the actor does not occupy the ring
    assert f["n_att_near3"] == 1                       # the genuine attacker still does
    assert f["first_to_drop_att"] is True              # without the exclusion the actor would take this
    assert f["n_seen"] == 2                            # n_seen stays a coverage count over the whole frame
    assert f["n_scored"] == 1                          # the rings score the actor out

    # and with the actor left in, the verdict flips, which is the bug this pins
    stale = frame.assign(is_actor=0)
    assert DropSituation._features(stale, drop)["first_to_drop_att"] is False


def test_the_keeper_counts_as_a_defending_body_at_the_drop():
    """The keeper IS counted here, which is the opposite of what s02._def() does.

    s02 drops him because its counts ask who can contest a delivery and a keeper defends a
    line; at the drop the question is who reaches a loose ball in the six-yard box, which is
    a contest he is in. Pinned because it is a modelling choice with no test of its own: if
    someone later excludes him to match s02, this goes red and the change has to be argued."""
    drop = (114.0, 40.0)
    # the keeper is the only defending body, and he is nearer the drop than the attacker
    frame = _frame([(114.5, 40.0, 0), (117.0, 40.0, 1)])
    frame.loc[0, "is_goalkeeper"] = 1
    f = DropSituation._features(frame, drop)
    assert f["n_def_near3"] == 1                       # he occupies the ring
    assert f["first_to_drop_att"] is False             # and he wins the nearest-body test
    assert f["def_min_dist"] < f["att_min_dist"]


def test_oriented_frame_remaps_teammate_and_carries_the_actor():
    """_oriented_frame maps `teammate` into corner-attacker terms and keeps `actor`.

    The teammate flag is relative to the event's own actor, so on a DEFENDING team's
    event it means the opposite of corner-attacker. Get that backwards and every
    situation feature silently swaps sides."""
    fz = pd.DataFrame([{"event_uuid": "e1", "freeze_frame": [
        {"location": [100.0, 30.0], "teammate": True,  "keeper": False, "actor": True},
        {"location": [110.0, 50.0], "teammate": False, "keeper": False, "actor": False},
    ]}])
    ds = object.__new__(DropSituation)                  # the method reads only fz/ids
    ds.fz, ds.ids = fz, {"e1"}

    # event by the corner ATTACKER: teammate == attacker, and no flip
    own = ds._oriented_frame("e1", "A", "A")
    assert list(own["team_id"]) == [1, 0]
    assert list(own["is_actor"]) == [1, 0]
    assert (own.loc[0, "x"], own.loc[0, "y"]) == (100.0, 30.0)

    # event by the DEFENDING team: teammate now means defender, and the frame flips
    opp = ds._oriented_frame("e1", "B", "A")
    assert list(opp["team_id"]) == [0, 1]               # the remap, which is the load-bearing bit
    assert list(opp["is_actor"]) == [1, 0]              # the actor is still the actor
    assert (opp.loc[0, "x"], opp.loc[0, "y"]) == (PITCH_X_SB - 100.0, PITCH_Y_SB - 30.0)


# ── The next touch: bounded by the episode, not by the clearance's possession ──
def _drop_situation(*rows) -> DropSituation:
    """A DropSituation over a synthetic corner sequence and an empty 360."""
    fz = pd.DataFrame({"event_uuid": pd.Series(dtype=object),
                       "freeze_frame": pd.Series(dtype=object)})
    return DropSituation(pd.DataFrame([_corner(), *rows]), fz)


def test_a_clean_defensive_recovery_is_a_lost_second_ball():
    """A defender who collects the loose ball opens a new possession by doing so, and that is the
    attack losing the second ball. Bounding the next touch to the clearance's possession would
    drop exactly these rows from the label, nearly all of them the defence's (evidence.md §15)."""
    ds = _drop_situation(_row(1.0, "Clearance", DEF), _row(2.5, "Ball Recovery", DEF, possession=2))
    corner = ds.g.corners().iloc[0]
    nxt = ds._clear_and_next(corner)[1]
    assert nxt is not None and nxt["team.name"] == DEF
    assert ds.label(corner).won_second_ball is False


def test_a_restart_is_not_a_contest_for_the_loose_ball():
    """The ball out of play or a whistle leaves no second ball to win, so a restart as the next
    touch leaves the label empty rather than crediting whoever takes it."""
    ds = _drop_situation(_row(1.0, "Clearance", DEF),
                         _row(4.0, "Pass", ATT, possession=2, **{"pass.type.name": "Throw-in"}))
    corner = ds.g.corners().iloc[0]
    assert ds._clear_and_next(corner)[1] is None
    assert ds.label(corner).won_second_ball is None


def test_staked_is_empty_without_a_next_touch():
    """The staked test scores the plan against where the next touch contested the loose ball.
    With no next touch there is no contest, so `staked` stays empty instead of scoring the plan against the
    clearance; the clearance still places the drop the local features read."""
    ds = _drop_situation(_row(1.0, "Clearance", DEF, location=[10.0, 40.0]),
                         _row(4.0, "Pass", ATT, possession=2, **{"pass.type.name": "Throw-in"}))
    corner = ds.g.corners().iloc[0]
    lab = ds.label(corner, _delivery([("stake_drop", 108.0, 40.0)]))
    assert lab.has_next_touch is False and lab.won_second_ball is None
    assert (lab.drop_x, lab.drop_y) == (110.0, 40.0)
    assert lab.staked is None and lab.four_state is None


def test_a_delivery_nobody_touched_has_no_clearance_to_contest():
    """The clearance comes from s01's episode, so a delivery with no touch inside the window has
    none: a clearance long after the window is later play, and searched with no bound it would
    give the delivery a drop point and a second ball it never produced."""
    ds = _drop_situation(_row(WIN + 7.0, "Clearance", DEF), _row(WIN + 8.0, "Ball Receipt*", ATT))
    corner = ds.g.corners().iloc[0]
    assert ds._clear_and_next(corner)[0] is None
    assert ds.label(corner).cleared is False


# ── Real sample: the full builder over granted corners ────────────────
@requires_sb
def test_build_situation_table(sb_mid):
    """The full builder over a real granted match yields cleared-only rows with well-formed, att-oriented situation columns."""
    tab = build_situation_table([sb_mid])
    assert not tab.empty
    # Derived from the dataclass, not listed by hand: a hand-written list passes while a field
    # it forgot to name goes missing, and run_q2 and run_sufficiency select these columns by name.
    assert set(tab.columns) == {f.name for f in fields(SituationLabel)} | {"match_id"}
    assert tab["cleared"].all()
    # four_state, when known, is one of the four; and equals the staked x won cross
    known = tab[tab["four_state"].notna()]
    assert known["four_state"].isin(_STATES).all()
    for _, r in known.iterrows():
        assert r["four_state"] == four_state(bool(r["staked"]), bool(r["won_second_ball"]))
    # staked is known exactly where the delivery has a 360 and a next touch contests the ball,
    # and it is the nearest staker's distance read against the radius
    assert (tab["staked"].notna() == (tab["n_stakers"].notna() & tab["has_next_touch"])).all()
    scored = tab[tab["staked"].notna()]
    assert (scored["staked"].astype(bool) == (scored["staker_dist"] <= STAKE_RADIUS)).all()
    assert (scored["staker_dist"].isna() == (scored["n_stakers"] == 0)).all()
    # situation features, where present, are well-formed and oriented to att -> x=120
    seen = tab[tab["n_seen"].notna()]
    assert ((seen["local_control"] >= 0) & (seen["local_control"] <= 1)).all()
    assert (tab["drop_x"].dropna() >= 60).all()                          # drop is in the attacking half
    assert (tab["drop_x"].dropna() <= PITCH_X_SB).all()


@requires_sb
def test_value_trio_is_phi_and_self_consistent(sb_mid):
    """The value trio prices the drop and stays internally consistent.

    It is pure PHI (after-clearance shots only), so it must never be missing: a
    corner that produced nothing is a true 0.0, not a NaN. A goal implies a shot
    exists, and any positive xG implies at least one shot."""
    tab = build_situation_table([sb_mid])
    assert tab["value_xg_after"].notna().all() and tab["n_after_shots"].notna().all()
    assert (tab["value_xg_after"] >= 0).all() and (tab["value_xg_after"] <= 1).all()
    assert (tab.loc[tab["value_goal_after"], "n_after_shots"] >= 1).all()
    assert (tab.loc[tab["value_xg_after"] > 0, "n_after_shots"] >= 1).all()
    assert (tab.loc[tab["n_after_shots"] == 0, "value_xg_after"] == 0.0).all()


@requires_sb
def test_value_matches_s01_after_clearance_xg(sb_mid):
    """S08's price of the drop must BE s01's after_clearance_xg, not a second
    opinion on it. Both take the max xG over the same segment, so any drift here
    means the two stages disagree about what the drop produced."""
    from s01_phases import SetPieceGrammar
    import loader_sb

    sit = build_situation_table([sb_mid]).set_index("set_piece_id")
    lab = SetPieceGrammar(loader_sb.load_events(sb_mid), kinds=("corner",)).label_all()
    joined = sit.join(lab.set_index("set_piece_id")[["after_clearance_xg"]], how="inner")
    assert len(joined) == len(sit)
    assert (joined["value_xg_after"] - joined["after_clearance_xg"]).abs().max() == 0.0


@requires_sb
def test_build_situation_parallel_matches_serial(sb_mid):
    """The parallel build must reproduce the serial one exactly: row order decides
    GroupKFold membership, so a reordering here silently changes every fold."""
    pd.testing.assert_frame_equal(build_situation_table([sb_mid], n_jobs=1),
                                  build_situation_table([sb_mid], n_jobs=2))


@requires_sb
def test_the_default_situation_table_is_the_corner_population(sb_mid):
    """`kinds` must leave situation.parquet's population exactly where it was."""
    pd.testing.assert_frame_equal(build_situation_table([sb_mid]),
                                  build_situation_table([sb_mid], kinds=("corner",)))


@requires_sb
def test_build_situation_flank_parallel_matches_serial(flank_mids):
    """The flank free-kick build must reproduce the serial one exactly, with the same schema."""
    kinds = ("free_kick_flank",)
    a = build_situation_table(flank_mids, n_jobs=1, kinds=kinds)
    b = build_situation_table(flank_mids, n_jobs=2, kinds=kinds)
    assert len(a) > 0
    assert set(a.columns) == {f.name for f in fields(SituationLabel)} | {"match_id"}
    pd.testing.assert_frame_equal(a, b)


@requires_sb
def test_situation_deterministic(sb_mid):
    """Re-running the builder on the same match yields an identical table (no hidden randomness)."""
    tab = build_situation_table([sb_mid])
    pd.testing.assert_frame_equal(tab, build_situation_table([sb_mid]))
