"""Tests for run_first_contact: orientation, teams, order, and the reuse the driver promises.

The estimand is conditional on a frame read AFTER the delivery, so what these pin is that the
substitution is exactly a substitution: the same schema s02 publishes, the same race formula s04b
publishes, the same out-of-fold vectors the strong learner publishes, and the same bootstrap
validate publishes. Anything the driver computes on its own is a verdict rule, and that has its own
test here too.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "src", _REPO / "pipeline"):
    sys.path.insert(0, str(p))

from constants import PITCH_X_SB, PITCH_Y_SB                           # noqa: E402
from s02_signature import CommitmentSignature, build_frame              # noqa: E402
from s03_value import _THREE_SIXTY                                      # noqa: E402
from s04b_snapshot import snapshot_record                               # noqa: E402
from s08_situation import DropSituation                                 # noqa: E402
from test_s01_phases import ATT, DEF, _corner, _row                     # noqa: E402
from validate import refit_cluster_bootstrap                            # noqa: E402
import run_first_contact as fc                                          # noqa: E402


def _player(x: float, y: float, teammate: bool, actor: bool = False,
            keeper: bool = False) -> dict:
    return {"location": [x, y], "teammate": teammate, "actor": actor, "keeper": keeper}


def _situation(rows: list[dict], frames: dict) -> DropSituation:
    """A DropSituation over a synthetic corner sequence, with a 360 on the events in `frames`."""
    fz = pd.DataFrame([{"event_uuid": k, "freeze_frame": v, "visible_area": []}
                       for k, v in frames.items()])
    return DropSituation(pd.DataFrame([_corner(), *rows]), fz)


def test_t1_frame_drops_the_actor_and_turns_a_defending_event_into_the_attacking_frame():
    """A defender makes the first contact: the frame is in the defending team's orientation and
    `teammate` means defender. The T1 frame is the attacker's, the actor is out, and the corner spot is the
    single ball row s02's geometry expects."""
    fc_row = _row(1.0, "Clearance", DEF, id="FC", location=[10.0, 40.0])
    ds = _situation([fc_row], {"FC": [_player(10.0, 40.0, True, actor=True),
                                      _player(12.0, 35.0, True),
                                      _player(8.0, 45.0, False)]})
    corner = ds.g.corners().iloc[0]
    frame = fc.t1_frame(ds, pd.Series(fc_row), ATT, corner)
    bodies = frame[frame["is_ball"] == 0].reset_index(drop=True)
    assert len(bodies) == 2 and (frame["is_actor"] == 0).all()
    assert list(bodies["team_id"]) == [0, 1]
    assert (bodies.loc[0, "x"], bodies.loc[0, "y"]) == (PITCH_X_SB - 12.0, PITCH_Y_SB - 35.0)
    ball = frame[frame["is_ball"] == 1]
    assert len(ball) == 1 and (ball.iloc[0]["x"], ball.iloc[0]["y"]) == (120.0, 0.1)


def test_contact_facts_are_read_in_the_attacking_frame():
    """Who touched, when, and where, with the defender's location turned into the attacker's frame."""
    fc_row = pd.Series({**_row(1.25, "Clearance", DEF, location=[10.0, 30.0]), "_t": 1.25})
    f = fc.contact_facts(fc_row, 0.0, ATT)
    assert f == {"fc_won": 0.0, "fc_t_s": 1.25, "fc_x": PITCH_X_SB - 10.0, "fc_y": PITCH_Y_SB - 30.0}


def test_race_record_reproduces_snapshot_record_on_the_delivery_frame():
    """The T1 race is s04b's formula on another frame; on the delivery frame it must be s04b's."""
    fz = pd.DataFrame([{"event_uuid": "CORNER", "visible_area": [], "freeze_frame": [
        _player(120.0, 0.1, True, actor=True), _player(110.0, 38.0, True),
        _player(111.0, 42.0, False), _player(119.0, 40.0, False, keeper=True)]}])
    corner = pd.Series(_corner())
    ours = fc.race_record(build_frame(corner, fz)[0])
    theirs = snapshot_record(corner, fz)
    assert np.array_equal(ours["tti"], theirs["tti"])
    assert np.array_equal(ours["is_att"], theirs["is_att"])


def test_order_assertion_stops_the_run():
    """T1 <= clearance < drop, or the row is not the object the estimand describes."""
    fc.assert_order(1.0, 1.0, 2.0)
    for bad in ((2.0, 1.0, 3.0), (1.0, 2.0, 2.0)):
        with pytest.raises(SystemExit):
            fc.assert_order(*bad)


def test_t1_geometry_covers_every_published_frame_column():
    """The substitution replaces exactly the published 360 block, so it must produce all of it."""
    frame = pd.DataFrame([{"x": 110.0, "y": 40.0, "team_id": 1, "is_goalkeeper": 0, "is_actor": 0,
                           "is_ball": 0},
                          {"x": 112.0, "y": 36.0, "team_id": 0, "is_goalkeeper": 0, "is_actor": 0,
                           "is_ball": 0},
                          {"x": 120.0, "y": 0.1, "team_id": 1, "is_goalkeeper": 0, "is_actor": 0,
                           "is_ball": 1}])
    assert set(_THREE_SIXTY) <= set(CommitmentSignature(frame, None).row("x"))


def test_near_spot_attackers_counts_attackers_only():
    """The corner taker cannot be identified in a 360, so the step-0 count is attackers near the spot."""
    frame = pd.DataFrame({"x": [119.0, 119.5, 110.0, 120.0], "y": [1.0, 1.5, 40.0, 0.1],
                          "team_id": [1, 0, 1, 1], "is_ball": [0, 0, 0, 1]})
    assert fc.near_spot_attackers(frame, (120.0, 0.1)) == 1


def test_corner_row_reads_the_first_contact_s08_bounds_the_episode_with():
    """A receipt, a clearance and a recovery: T1 is the receipt, the drop is the recovery."""
    rows = [_row(1.0, "Ball Receipt*", ATT, id="FC"), _row(1.5, "Clearance", DEF, id="CL"),
            _row(3.0, "Ball Recovery", DEF, id="NX", possession=2)]
    ds = _situation(rows, {"FC": [_player(110.0, 40.0, True, actor=True),
                                  _player(108.0, 38.0, True), _player(109.0, 42.0, False)]})
    corner = ds.g.corners().iloc[0]
    r, rec = fc.corner_row(ds, corner)
    assert r["has_t1"] and r["t1_has_360"] and not r["t1_is_clear"]
    assert r["t1_to_clear_s"] == pytest.approx(0.5) and r["t1_to_drop_s"] == pytest.approx(2.0)
    assert r["n_bodies_t1"] == 2 and r["fc_won"] == 1.0
    assert rec is not None and rec["is_att"].tolist() == [True, False]


def test_gbt_oof_is_tuned_gbt_oof_averaged_over_seeds():
    """Twenty-five tasks instead of five at a time must not change a single probability."""
    from s04b_snapshot import tuned_gbt_oof
    rng = np.random.default_rng(1)
    X, g = rng.normal(size=(150, 3)), np.repeat(np.arange(15), 10)
    y = (X[:, 0] + rng.normal(scale=1.0, size=150) > 0).astype(int)
    ours = fc.gbt_oof(X, y, g, trials=2, seeds=2, jobs=2)
    theirs = np.mean([tuned_gbt_oof(X, y, g, n_trials=2, jobs=1, seed=sd)[0] for sd in range(2)],
                     axis=0)
    assert np.array_equal(ours, theirs)


def test_an_arm_nobody_reads_is_a_fit_not_made_and_the_other_arm_does_not_change(monkeypatch):
    """The refit draws read `gbt_1` only. Dropping `gbt360_D` from them halves a draw, and it is exact
    only if asking for one arm returns the same vector as asking for both: each arm is its own fit."""
    class Model:
        @staticmethod
        def _design(tab):
            return tab[["a", "b", "c"]].to_numpy(float), None

    rng = np.random.default_rng(2)
    frame = pd.DataFrame(rng.normal(size=(120, 3)), columns=["a", "b", "c"])
    t1 = frame.assign(a=frame["a"] + 1.0)
    y = (frame["a"] + rng.normal(size=120) > 0).to_numpy().astype(int)
    g = np.repeat(np.arange(12), 10)
    fits, real = [], fc.gbt_oof
    monkeypatch.setattr(fc, "gbt_oof", lambda *a: fits.append(a) or real(*a))
    both = fc.gbt_arm_losses(frame, t1, Model, y, g, ("gbt360_D", "gbt_1"), trials=2, seeds=2, jobs=1)
    one = fc.gbt_arm_losses(frame, t1, Model, y, g, ("gbt_1",), trials=2, seeds=2, jobs=1)
    assert list(both) == ["gbt360_D", "gbt_1"] and list(one) == ["gbt_1"] and len(fits) == 3
    assert np.array_equal(both["gbt_1"], one["gbt_1"])
    assert not np.array_equal(both["gbt360_D"], both["gbt_1"])


def test_step_means_asks_only_for_the_strong_learner_arms_its_pairs_read(monkeypatch):
    """The six decisions read `gbt_1`; the reported pair also reads `gbt360_D`; phase one reads neither."""
    asked = []

    def arms(frame, **kw):
        asked.append(kw["gbt_arms"])
        return {n: np.full(4, float(i)) for i, n in enumerate(
            ["delivery", "static_race_1", "geometry_1", "geometry360_D", "facts_1", "facts_geometry_1",
             "static_race_D", "gbt360_D", "gbt_1"])}

    monkeypatch.setattr(fc, "arm_losses", arms)
    frame = pd.DataFrame({"x": [0.0]})
    fc.step_means(frame, fc.DECISIONS, with_gbt=True)
    fc.step_means(frame, tuple(p for p in fc.DECISIONS if "gbt_1" not in p), with_gbt=False)
    fc.step_means(frame, fc.DECISIONS + fc.REPORTED, with_gbt=True)
    assert asked == [("gbt_1",), (), ("gbt360_D", "gbt_1")]


def test_decide_reads_the_preregistered_rules():
    """A stable pivotal interval below zero decides; one that changes side across seeds does not;
    V1 without the strong learner's refit cannot close as a null."""
    def row(key, below, bonf_hi=1.0):
        return {"key": key, "below_zero": below, "bonf_hi": bonf_hi}
    rows = [row("static_race_1-delivery", False), row("geometry_1-delivery", True, -0.1),
            row("static_race_1-static_race_D", True), row("geometry_1-geometry360_D", False),
            row("facts_geometry_1-facts_1", False)]
    out = fc.decide(rows, {"static_race_1-static_race_D": False}, gbt_refit=False)
    assert out["V1"]["state"] == "true" and out["V1"]["bonferroni_below_zero"] == ["geometry_1-delivery"]
    assert out["V2"]["state"].startswith("unstable") and out["V3"]["state"] == "false"
    rows[1] = row("geometry_1-delivery", False)
    assert fc.decide(rows, {}, gbt_refit=False)["V1"]["state"].startswith("undecided")
    assert fc.decide(rows, {}, gbt_refit=True)["V1"]["state"] == "false"


def test_mde_grid_is_the_power_curve_point_by_point():
    """In parallel or not, each candidate delta is power_curve's with that candidate's own seed. The
    grid sits where power is neither 0 nor 1, or a wrong seed would leave the curve unchanged."""
    from s04b_snapshot import power_curve
    rng = np.random.default_rng(5)
    diff, g = rng.normal(0.01, 0.05, size=200), np.repeat(np.arange(20), 10)
    grid = (0.004, 0.006, 0.008)
    _, ours = fc.mde_grid(diff, g, grid, jobs=2)
    theirs = pd.concat([power_curve(diff, g, (d,), n_boot=400, n_rep=300, seed=i)
                        for i, d in enumerate(grid)], ignore_index=True)
    assert ours.equals(theirs)


def test_resume_reuses_saved_draws_only_at_the_count_asked_for(monkeypatch, tmp_path):
    """--resume reads a seed's draws back from disk, here and in run_aim_zone, which imports this
    copy. A file of another length, a smoke's three draws left in the same directory, would set the
    verdict on them in silence, so it is drawn again; a file of the asked length is reused."""
    path = tmp_path / "first_contact_draws_seed0.npy"
    np.save(path, np.zeros((3, 2)))
    calls = []

    def redraw(table, stat, n, seed, jobs):
        calls.append(n)
        return np.zeros(2), np.ones((n, 2))

    monkeypatch.setattr(fc, "refit_draws", redraw)
    _, d = fc.seed_draws(pd.DataFrame(), None, 300, 0, 1, path, resume=True)
    assert calls == [300] and d.shape == (300, 2)
    point, d = fc.seed_draws(pd.DataFrame(), None, 300, 0, 1, path, resume=True)
    assert calls == [300] and point is None and d.shape == (300, 2)


@pytest.mark.parametrize("kind", [[], ["--kind", "free_kick_flank"]])
@pytest.mark.parametrize("dest", [[], ["--out-dir", str(_REPO / "data" / "derived" / "zones_F")]])
def test_the_published_record_is_written_only_at_the_registered_precision(monkeypatch, dest, kind):
    """V1-V3 are registered at 300 refit draws, five stability seeds of 2000 and the strong learner's
    budget, so a run at any other precision may not write either population's record, in
    data/derived or in zones_F where the pass under partition F publishes it, and refuses before the
    grant is read."""
    def work(*a, **k):
        raise AssertionError("the run did the work before refusing")

    monkeypatch.setattr(fc, "grant_table", work)
    monkeypatch.setattr(sys, "argv", ["run_first_contact.py", "--refit-boot", "3", *dest, *kind])
    with pytest.raises(SystemExit, match="may not write the published"):
        fc.main()


def test_the_flank_twin_reads_its_own_grant(monkeypatch, tmp_path):
    """--kind free_kick_flank reads that population's grant, the one run_sufficiency reads for it."""
    seen = {}

    def grant(limit, tag=""):
        seen["tag"] = tag
        raise SystemExit("grant read")

    monkeypatch.setattr(fc, "grant_table", grant)
    monkeypatch.setattr(sys, "argv", ["run_first_contact.py", "--kind", "free_kick_flank",
                                      "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="grant read"):
        fc.main()
    assert seen["tag"] == "_free_kick_flank"


def test_a_match_is_read_for_the_population_asked_for(monkeypatch):
    """The first contact of a flank free kick is s08's, read through a grammar of that population and
    its deliveries, as for a corner."""
    seen = {}

    class Situation:
        ids: set = set()

        def __init__(self, ev, fz, kinds=("corner",)):
            seen["kinds"] = kinds
            self.g = type("Grammar", (), {"deliveries": staticmethod(
                lambda: pd.DataFrame({"id": pd.Series([], dtype=object)}))})()

    monkeypatch.setattr(fc, "DropSituation", Situation)
    monkeypatch.setattr(fc.L, "load_events", lambda m: None)
    monkeypatch.setattr(fc.L, "load_360", lambda m: None)
    assert fc.match_rows(1, set(), "free_kick_flank") == ([], {}, {})
    assert seen["kinds"] == ("free_kick_flank",)


def test_refit_draws_reproduce_validate_at_the_nominal_level():
    """Keeping the draws must not change them: at 95% the bounds are validate's, bit for bit."""
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"match_id": np.repeat(np.arange(12), 5), "v": rng.normal(size=60)})
    stat = lambda f: np.array([f["v"].mean(), f["v"].median()])
    point, draws = fc.refit_draws(frame, stat, n_boot=40, seed=0, n_jobs=1)
    ref = refit_cluster_bootstrap(frame, stat, n_boot=40, seed=0, n_jobs=1)
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    assert np.array_equal(point, np.atleast_1d(ref.point))
    assert np.array_equal(lo, ref.lo) and np.array_equal(hi, ref.hi)
