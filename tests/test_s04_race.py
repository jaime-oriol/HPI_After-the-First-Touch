"""Tests for s04_race — second-ball race (time-to-arrive + reaction lag).

Pure helpers run everywhere; the race itself needs the granted Broadcast X&Y
tracking sample and skips cleanly without it. SecondBallRace is built once per
module (deriving causal velocities over a full match is the costly step)."""
import numpy as np
import pandas as pd
import pytest

import loader
import loader_sb
from s01_phases import CornerPhaseGrammar
from s04_race import (
    SecondBallRace, find_corners, _recovery_prob_att, anchor_in_tracking, calibrate_tau,
)
from constants import (DROP_ANCHOR_SB, PITCH_LENGTH_M, PITCH_X_SB, REACTION_MAX_S,
                       RECOVERY_TAU_S)
from conftest import requires_data, requires_sb


@requires_sb
def test_find_corners_default_is_the_corner_population(sb_events):
    """`kinds` must leave the published race population exactly where it was."""
    pd.testing.assert_frame_equal(find_corners(sb_events), find_corners(sb_events, kinds=("corner",)))
    pd.testing.assert_frame_equal(find_corners(sb_events), CornerPhaseGrammar(sb_events).corners())


@requires_sb
def test_find_corners_races_the_population_it_is_handed(flank_mids):
    fk = find_corners(loader_sb.load_events(flank_mids[0]), kinds=("free_kick_flank",))
    assert len(fk) >= 1 and set(fk["set_piece_type"]) == {"free_kick_flank"}


def test_recovery_prob_att():
    """Soft-argmin recovery prob: faster attacker -> higher P(att), ties -> 0.5."""
    fast = pd.DataFrame({"tti_s": [0.5, 1.5], "is_att": [1, 0]})
    slow = pd.DataFrame({"tti_s": [1.5, 0.5], "is_att": [1, 0]})
    tie = pd.DataFrame({"tti_s": [1.0, 1.0], "is_att": [1, 0]})
    p_fast, p_slow = _recovery_prob_att(fast), _recovery_prob_att(slow)
    assert 0.0 <= p_fast <= 1.0 and p_fast > 0.5 > p_slow
    assert abs(_recovery_prob_att(tie) - 0.5) < 1e-9


def test_calibrate_tau_gate():
    """One match group -> non-operative, keeps the default tau."""
    rec = [{"tti": np.array([0.5, 1.5]), "is_att": np.array([True, False]), "y": True, "match_id": 1},
           {"tti": np.array([1.5, 0.5]), "is_att": np.array([True, False]), "y": False, "match_id": 1}]
    out = calibrate_tau(rec)
    assert out["operative"] is False and out["tau"] == RECOVERY_TAU_S


def test_calibrate_tau_operative():
    """>=2 matches with both outcomes -> a tau is selected with a finite OOF loss."""
    recs = []
    for m in range(4):
        recs.append({"tti": np.array([0.4, 1.2]), "is_att": np.array([True, False]), "y": True, "match_id": m})
        recs.append({"tti": np.array([1.2, 0.4]), "is_att": np.array([True, False]), "y": False, "match_id": m})
    out = calibrate_tau(recs)
    assert out["operative"] is True and out["tau"] > 0 and np.isfinite(out["oof_log_loss"])


@pytest.fixture(scope="module")
def race_fx(mid):
    """SecondBallRace + its corners for the sample match (built once: the causal
    Savitzky-Golay pass over a full match is the expensive step)."""
    trk = loader.load_tracking(mid)
    ev = loader_sb.load_events(mid)
    return SecondBallRace(trk, ev), find_corners(ev), ev


@requires_data
def test_find_corners_matches_s01(race_fx):
    """S04 races exactly the corners S01 labels: same ids, not merely the same count.
    find_corners delegates to the grammar precisely so these cannot drift apart."""
    _, corn, ev = race_fx
    assert len(corn) > 0
    assert set(corn["id"]) == set(CornerPhaseGrammar(ev).corners()["id"])
    assert {"id", "period", "_t", "team.id"} <= set(corn.columns)


@requires_data
def test_anchor_lands_on_a_tracked_ball(race_fx):
    """Regression guard on the event->frame clock.

    The corner must be anchored on the vendor game clock (period_elapsed_s). The
    video clock drifts ahead of it in period 1, which lands the anchor before the
    kick, while the ball is still being placed and untracked, so the ball-present
    rate at the anchored frame is what exposes a wrong clock. That regression
    breaks no other assertion here, so it needs its own."""
    race, corn, _ = race_fx
    seen = 0
    for _, c in corn.iterrows():
        f0 = race._corner_frame(c)
        frame = race.trk[(race.trk["period_id"] == int(c["period"])) &
                         (race.trk["frame_count"] == f0)]
        seen += int((frame["is_ball"] == 1).any())
    assert seen / len(corn) >= 0.9


@requires_data
def test_race_table(race_fx):
    """The race table is sorted by tti with exactly one predicted recoverer and valid ranges."""
    race, corn, _ = race_fx
    r = next(t for t in (race.race(c) for _, c in corn.iterrows()) if t is not None)
    assert {"player_id", "team_id", "is_att", "dist_to_loose",
            "reaction_lag_s", "tti_s", "pred_recoverer"} <= set(r.columns)
    assert np.all(np.diff(r["tti_s"].values) >= -1e-9)
    assert r["pred_recoverer"].sum() == 1
    assert r["pred_recoverer"].iloc[0] == 1
    assert r["reaction_lag_s"].between(0, REACTION_MAX_S).all()
    assert (r["tti_s"] >= 0).all() and (r["dist_to_loose"] >= 0).all()
    assert not r[["tti_s", "reaction_lag_s", "dist_to_loose"]].isna().any().any()


@requires_data
def test_race_splits_both_teams(race_fx):
    """Both sides are raced: is_att must partition the frame by the corner's team.id,
    never collapse to one team (which is how a bad team-id join would show up)."""
    race, corn, _ = race_fx
    c = next(c for _, c in corn.iterrows() if race.race(c) is not None)
    r = race.race(c)
    assert set(r.loc[r["is_att"] == 1, "team_id"].unique()) == {c["team.id"]}
    assert (r["is_att"] == 0).any() and (r["is_att"] == 1).any()


@requires_data
def test_summary(race_fx):
    """A race summary reports a bounded recovery probability and sane timings."""
    race, corn, _ = race_fx
    s = next(x for x in (race.summary(c) for _, c in corn.iterrows()) if x is not None)
    assert 0.0 <= s["recovery_prob_att"] <= 1.0
    assert isinstance(s["pred_recoverer_att"], bool)
    assert s["pred_tti_s"] >= 0 and s["first_contact_s"] > 0


@requires_data
def test_all_corners_race(race_fx):
    """Most corners are raceable and every summary stays consistent.

    The share matters: a mis-anchored corner still returns a valid-looking table, so
    only the RATE exposes it. Measured 13/14 on this match with the correct clock,
    7/14 with the wrong one."""
    race, corn, _ = race_fx
    n = 0
    for _, c in corn.iterrows():
        s = race.summary(c)
        if s is None:
            continue
        n += 1
        assert 0.0 <= s["recovery_prob_att"] <= 1.0
        assert isinstance(s["pred_matches_actual"], bool)
    assert n / len(corn) >= 0.7


@requires_data
def test_reach_field_reproduces_summary_at_the_loose_ball(race_fx):
    """reach_field's docstring promises it IS race()'s model as a surface: "at the loose
    ball it reproduces summary()'s recovery_prob_att". The field is what render_race draws
    into a submittable figure, so the promise is checked here.

    The residual is grid discretisation only. Measured over this match on a 421-wide
    grid (0.25 m cells): max |diff| 0.034, mean 0.011 across 13 raceable corners. The
    bounds below sit just above that, so a formula drifting apart in one of the two
    code paths fails here while the sampling error does not."""
    race, corn, _ = race_fx
    diffs = []
    for _, c in corn.iterrows():
        fc = race.first_contact(c)
        if fc is None:
            continue
        s, out = race.summary(c, fc=fc), race.reach_field(c, fc=fc, n_grid_x=421)
        if s is None or out is None:
            continue
        p, xg, yg = out
        assert p.shape == (len(yg), len(xg))                 # the (ny, nx) layout ppcf shares
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        lx, ly = fc["loose_xy"]
        j, i = int(np.abs(xg - lx).argmin()), int(np.abs(yg - ly).argmin())
        diffs.append(abs(float(p[i, j]) - s["recovery_prob_att"]))

    assert diffs, "no raceable corner in the sample match"
    assert max(diffs) <= 0.05, f"reach_field and summary disagree at the loose ball: {max(diffs):.4f}"
    assert float(np.mean(diffs)) <= 0.02


def test_calibrate_tau_drops_unlabelled_records():
    """A y=None record (recoverer unknown) must be excluded from calibration, not
    crash it and not silently count as either class."""
    recs = []
    for m in range(4):
        recs.append({"tti": np.array([0.4, 1.2]), "is_att": np.array([True, False]), "y": True, "match_id": m})
        recs.append({"tti": np.array([1.2, 0.4]), "is_att": np.array([True, False]), "y": False, "match_id": m})
    unlabelled = {"tti": np.array([0.5, 0.6]), "is_att": np.array([True, False]), "y": None, "match_id": 0}
    out = calibrate_tau(recs + [unlabelled])
    assert out["operative"] is True and out["n"] == len(recs)


def test_calibrate_tau_refuses_a_split_it_cannot_demonstrate():
    """An unresolved match id must raise, not be tuned around.

    NaN compares unequal to itself, so a record whose group never resolved passes a set
    intersection on BOTH sides at once and the split reports green having checked nothing."""
    recs = []
    for m in range(4):
        recs.append({"tti": np.array([0.4, 1.2]), "is_att": np.array([True, False]),
                     "y": True, "match_id": m})
        recs.append({"tti": np.array([1.2, 0.4]), "is_att": np.array([True, False]),
                     "y": False, "match_id": m})
    recs.append({"tti": np.array([0.5, 0.9]), "is_att": np.array([True, False]),
                 "y": True, "match_id": float("nan")})
    with pytest.raises(AssertionError, match="cannot be demonstrated"):
        calibrate_tau(recs)


def test_anchor_lands_on_the_attacking_side_at_the_right_depth():
    """A fixed StatsBomb point placed into the tracking frame, both ways round.

    Tracking has no canonical attacking direction, so the anchor rotates with the corner and
    a sign error would put it behind the defending goal while still returning a plausible
    race. Depth is read off the constant: DROP_ANCHOR_SB sits 18.0 units short of the goal
    line, which is 15.75 m once rescaled, so re-fixing the anchor moves the 15.75 below with
    it. Both axes flip together, because the two halves are a 180-degree rotation of each
    other and not a mirror: a y that failed to flip would put the anchor on the wrong flank."""
    right = anchor_in_tracking(DROP_ANCHOR_SB, 1.0)
    left = anchor_in_tracking(DROP_ANCHOR_SB, -1.0)
    assert right[0] > 0 and left[0] < 0
    assert np.allclose(right, -left)
    assert right[1] != 0 and np.sign(right[1]) == -np.sign(left[1])
    assert PITCH_LENGTH_M / 2 - right[0] == pytest.approx(15.75, abs=0.01)


@requires_data
def test_the_anchor_puts_a_corner_on_the_flank_the_ball_is_on(race_fx):
    """StatsBomb's y and the tracking y run opposite ways, and the conversion has to know it.

    Placed through `anchor_in_tracking`, a corner's own spot has to land on the flank where the
    feed shows the ball at the kick. The rotation test above cannot see a mirrored y, because both
    halves flip together either way; this one reads the flank off the tracking. The ball at the
    kick frame is a single and often interpolated detection, so the bar is most corners, not all."""
    race, corn, ev = race_fx
    home = int(ev["home_team.home_team_id"].iloc[0])
    agree = []
    for _, c in corn.iterrows():
        f0 = race._corner_frame(c)
        frame = race.trk[(race.trk["period_id"] == int(c["period"])) &
                         (race.trk["frame_count"] == f0)]
        ball = frame[frame["is_ball"] == 1]
        if ball.empty:
            continue
        side = -1.0 if loader.attack_runs_rtl(frame, c["team.id"], home) else 1.0
        spot = anchor_in_tracking((PITCH_X_SB, float(c["location"][1])), side)
        agree.append(np.sign(spot[1]) == np.sign(float(ball.iloc[0]["y_tracking"])))
    assert len(agree) >= 5 and np.mean(agree) >= 0.8


@requires_data
def test_the_anchored_rung_takes_its_side_from_the_period_not_the_ball(race_fx):
    """The attacking side must be the loader's per-period direction, one decision per half.

    Two things this pins. The side has to agree with `attack_runs_rtl`, which the renders
    already use, so the repo cannot end up with two mechanisms for one quantity. And it has
    to be CONSTANT within a period: a rule read off each corner's own frame would wobble
    corner to corner, and the ball at the delivery frame is exactly that kind of single
    unreliable detection: measured, it sits a median 14.4 m from the nearest flag and on one
    corner by the centre circle."""
    race, corn, ev = race_fx
    home = int(ev["home_team.home_team_id"].iloc[0])
    seen: dict = {}
    for _, c in corn.iterrows():
        fc = race.first_contact(c)
        if fc is None:
            continue
        side = race._attacking_side(fc, c["team.id"])
        assert side in (1.0, -1.0)
        frame = race.trk[(race.trk["period_id"] == fc["period"]) &
                         (race.trk["frame_count"] == fc["frame"])]
        assert side == (-1.0 if loader.attack_runs_rtl(frame, c["team.id"], home) else 1.0)
        seen.setdefault((int(fc["period"]), int(c["team.id"])), side)
        assert seen[(int(fc["period"]), int(c["team.id"]))] == side, \
            "the attacking side changed within one period for the same team"


@requires_data
def test_the_anchored_rung_changes_the_race_and_the_default_does_not(race_fx, mid):
    """race_records(anchor_sb=...) must race the same corners to a different point.

    Two things at once, and both matter. The anchored rung has to actually move the arrival
    times, or the ladder's first step would compare an arm against itself; and passing
    nothing has to reproduce the published race exactly, because race_corners.parquet is
    built from it and every figure downstream is keyed to that file."""
    race, corn, _ = race_fx
    base = race.race_records(corn, mid)
    anchored = race.race_records(corn, mid, anchor_sb=DROP_ANCHOR_SB)
    assert [r["corner_id"] for r in base] == [r["corner_id"] for r in anchored]
    assert [r["y"] for r in base] == [r["y"] for r in anchored], "the label is not a rung"
    assert any(not np.allclose(a["tti"], b["tti"]) for a, b in zip(base, anchored, strict=True))
    again = race.race_records(corn, mid)
    assert all(np.array_equal(a["tti"], b["tti"]) for a, b in zip(base, again, strict=True))


@requires_data
def test_race_records_feed_calibrate_tau(race_fx, mid):
    """race_records emits the record shape calibrate_tau consumes, and a single
    match is correctly reported non-operative rather than silently tuned."""
    race, corn, _ = race_fx
    recs = race.race_records(corn, mid)
    assert len(recs) > 0
    r = recs[0]
    assert set(r) == {"tti", "is_att", "y", "match_id", "corner_id", "t_contact_s", "period",
                      "ball_estimated_at_contact", "ball_estimated_at_loose"}
    assert len(r["tti"]) == len(r["is_att"]) and np.isfinite(r["tti"]).all()
    assert calibrate_tau(recs)["operative"] is False


@requires_data
def test_first_contact_reports_the_balls_visibility_as_the_feed_marks_it(race_fx, mid):
    """The flag has to be the vendor's own `visibility` on the very row the contact was located
    on, never a proxy, and a missing ball at the loose frame has to read as unknown rather than
    as a detection. Both values occur on this match, so a flag that never fired would be as
    wrong as one that always did, and `race_records` has to carry what `first_contact` found."""
    race, corn, _ = race_fx
    flags: dict = {}
    for _, c in corn.iterrows():
        fc = race.first_contact(c)
        if fc is None:
            continue
        row = race.trk[(race.trk["period_id"] == fc["period"]) &
                       (race.trk["frame_count"] == fc["frame"]) & (race.trk["is_ball"] == 1)]
        assert fc["ball_estimated"] is bool(row["visibility"].iloc[0] == "ESTIMATED")
        assert fc["loose_estimated"] in (True, False, None)
        flags[c["id"]] = fc["ball_estimated"]
    assert any(flags.values()) and not all(flags.values()), \
        "both a detected and an interpolated contact are expected on this match"
    recs = race.race_records(corn, mid)
    assert recs and all(r["ball_estimated_at_contact"] is flags[r["corner_id"]] for r in recs)
    assert all(r["ball_estimated_at_loose"] in (True, False, None) for r in recs)


@requires_data
def test_the_delivery_rung_keeps_the_population_and_drops_the_kinematics(race_fx, mid):
    """Reading at the kick must change the predictor and nothing else.

    The ladder only means anything if every rung scores the SAME corners against the SAME
    label, so a paired difference is a difference of predictors. Three things pinned here: the
    corner set and the labels survive, the arrival times actually move (a rung identical to
    its neighbour would compare an arm with itself), and the delivery rung carries no reaction
    lag, which is what the step above it is supposed to be pricing."""
    race, corn, _ = race_fx
    contact = race.race_records(corn, mid, anchor_sb=DROP_ANCHOR_SB)
    kick = race.race_records(corn, mid, anchor_sb=DROP_ANCHOR_SB, at_delivery=True)
    assert [r["corner_id"] for r in contact] == [r["corner_id"] for r in kick]
    assert [r["y"] for r in contact] == [r["y"] for r in kick], "the label moved with the rung"
    assert any(not np.array_equal(a["tti"], b["tti"]) for a, b in zip(contact, kick, strict=True))
    # Racing from a standstill, every arrival time is pure travel: the shortest one is the
    # nearest body over the cap, with no lag added on top.
    vmax = race.params["max_player_speed"]
    for r in kick:
        assert r["tti"].min() * vmax < 60.0
        assert np.isfinite(r["tti"]).all() and (r["tti"] >= 0).all()


@requires_data
def test_the_delivery_rung_refuses_to_run_without_an_anchor(race_fx, mid):
    """At the kick there is no loose ball yet, so there is nothing legitimate to race to."""
    race, corn, _ = race_fx
    with pytest.raises(ValueError, match="needs an anchor"):
        race.race_records(corn, mid, at_delivery=True)
