"""Tests for eda_club_power: each estimator against a case whose answer is known, the own-goal rule,
the chained-shot rule, the split per team, C12a's reading and the parallel path."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "src", _REPO / "pipeline"):
    sys.path.insert(0, str(p))

import eda_club_power as C                                                 # noqa: E402


def test_the_icc_is_one_for_cluster_constants_and_near_zero_for_noise():
    rng = np.random.default_rng(0)
    cl = np.repeat(np.arange(200), 8)
    assert C.icc(np.repeat(rng.normal(size=200), 8), cl) == pytest.approx(1.0)
    assert abs(C.icc(rng.normal(size=1600), cl)) < 0.05


def test_the_icc_recovers_a_known_share_of_variance():
    rng = np.random.default_rng(1)
    cl = np.repeat(np.arange(1000), 10)
    x = np.repeat(rng.normal(0, 1.0, 1000), 10) + rng.normal(0, 3.0, 10000)
    assert C.icc(x, cl) == pytest.approx(1.0 / 10.0, abs=0.02)


def test_the_beta_binomial_fit_recovers_its_parameters():
    rng = np.random.default_rng(2)
    n = np.full(400, 300)
    k = rng.binomial(n, rng.beta(4.0, 36.0, 400))
    a, b = C.beta_binomial(k, n)
    assert a / (a + b) == pytest.approx(0.1, abs=0.01) and 20 < a + b < 70


def test_the_chained_rule_combines_shots_in_one_chance():
    out = C.chained(pd.Series([[0.3, 0.5], [], [0.6]]), 1.0)
    assert out == pytest.approx([1 - 0.7 * 0.5, 0.0, 0.6])
    assert C.chained(pd.Series([[0.6]]), 2.0)[0] == pytest.approx(1.0)


def _tables() -> tuple:
    deliv = pd.DataFrame({"set_piece_id": list("abc"), "league": "x", "attacking_team": "T",
                          "match_id": [1, 1, 2], "won_first_contact": [True, None, False]})
    shots = pd.DataFrame({"set_piece_id": ["a", "a", "b", "c"],
                          "segment": ["flick_on", "after_clearance", "direct", "prolongation"],
                          "shot_outcome": ["Saved", "Goal", "Goal", "Own Goal"],
                          "is_goal": [False, True, True, True], "xg": [0.1, 0.3, 0.2, 0.5]})
    return deliv, shots


def test_the_corner_table_leaves_own_goals_out_and_sums_the_route():
    t = C.corner_table(*_tables()).set_index("set_piece_id")
    assert t.loc["a", "aft_goal"] == 1.0 and t.loc["a", "aft_xg"] == pytest.approx(0.4)
    assert t.loc["c", "aft_goal"] == 0.0 and t.loc["c", "aft_shot"] == 0.0     # the own goal is out
    assert t.loc["b", "aft_shot"] == 0.0 and t.loc["b", "xg"] == pytest.approx(0.2)
    assert list(t["first_contact"]) == [1.0, 0.0, 0.0]


def test_more_lift_needs_fewer_corners_and_the_design_effect_scales_them():
    a = C.corners_needed(0.02, 0.14, 0.25, 1.0, True)
    b = C.corners_needed(0.02, 0.14, 1.00, 1.0, True)
    assert a > b > 0 and C.corners_needed(0.02, 0.14, 0.25, 2.0, True) == pytest.approx(2 * a)


def test_the_lift_the_formula_sees_inverts_it():
    m = {"mu": 0.18, "sd": 0.38}
    n = C.corners_needed(0.18, 0.38, 0.5, 1.3, True)
    assert C.matching_lift(m, n, True, 1.3) == pytest.approx(0.5, abs=1e-6)
    assert np.isnan(C.matching_lift({"mu": 0.6, "sd": 0.49}, 2.0, True, 1.0))


def test_a_rate_of_zero_has_no_lift_to_see():
    """A measure no delivery of the population scored on, as in a thin one, leaves nothing to lift:
    the corners and the lift are NaN, not a division by zero."""
    assert np.isnan(C.corners_needed(0.0, 0.0, 0.25, 1.0, True))
    assert np.isnan(C.corners_needed(0.0, 0.0, 0.25, 1.0, False))
    assert np.isnan(C.matching_lift({"mu": 0.0, "sd": 0.0}, 85.0, True, 1.0))


def test_a_continuous_measure_keeps_one_spread_so_any_lift_is_in_reach():
    n = C.corners_needed(0.02, 0.08, 1.0, 1.0, False)
    assert n == pytest.approx(C.Z80 ** 2 * 2 * 0.08 ** 2 / 0.02 ** 2)
    assert C.matching_lift({"mu": 0.02, "sd": 0.08}, 85.0, False, 1.0) == pytest.approx(
        C.Z80 * np.sqrt(2 * 0.08 ** 2 / 85.0) / 0.02, rel=1e-6)


def test_the_second_route_predicts_the_half_season_gap_on_binomial_corners():
    rng = np.random.default_rng(8)
    rows = [(f"t{i}", m) for i in range(80) for m in range(i * 40, i * 40 + 30) for _ in range(5)]
    t = pd.DataFrame(rows, columns=["attacking_team", "match_id"]).assign(league="x")
    for m in C.MEASURES:
        t[m] = (rng.random(len(t)) < 0.2).astype(float)
    measures = pd.DataFrame([{"measure": m, "mu": t[m].mean(), "sd": t[m].std()} for m in C.MEASURES])
    out = C.half_season_noise(t, measures, {m: 1.0 for m in C.MEASURES}, 30, 0)
    assert np.allclose(out["half_gap_sd"], out["half_gap_sd_predicted"], rtol=0.1)


def test_a_team_with_one_match_has_no_second_half_and_stays_out():
    """A thin population leaves teams with a single match, which no split can halve: the gap and its
    noise model read the teams found on both sides instead of failing on their unequal counts."""
    rng = np.random.default_rng(8)
    rows = [(f"t{i}", m) for i in range(20) for m in range(i * 40, i * 40 + 10) for _ in range(3)]
    t = pd.DataFrame(rows, columns=["attacking_team", "match_id"]).assign(league="x")
    t = pd.concat([t, t.iloc[:3].assign(attacking_team="zz", match_id=9999)], ignore_index=True)
    for m in C.MEASURES:
        t[m] = (rng.random(len(t)) < 0.2).astype(float)
    measures = pd.DataFrame([{"measure": m, "mu": t[m].mean(), "sd": t[m].std()} for m in C.MEASURES])
    out = C.half_season_noise(t, measures, {m: 1.0 for m in C.MEASURES}, 10, 0)
    assert np.isfinite(out["half_gap_sd_predicted"]).all()


def test_teams_that_do_not_vary_leave_no_luck_share():
    """No team scoring after the first touch, as in a thin population, leaves no variance between
    teams for luck to take a share of."""
    assert np.isnan(C.luck_share(0.001, 0.0))
    assert C.luck_share(0.001, 0.004) == pytest.approx(0.25)


def test_a_rate_of_zero_has_no_half_season_lift_either():
    t = pd.DataFrame([(f"t{i}", m) for i in range(4) for m in range(i * 10, i * 10 + 4)],
                     columns=["attacking_team", "match_id"]).assign(league="x")
    for m in C.MEASURES:
        t[m] = 0.0
    measures = pd.DataFrame([{"measure": m, "mu": 0.0, "sd": 0.0} for m in C.MEASURES])
    out = C.half_season_noise(t, measures, {m: 1.0 for m in C.MEASURES}, 3, 0)
    assert out["half_season_mde80_lift"].isna().all()


def test_the_simulated_variance_is_the_binomial_one_for_a_flat_chance():
    t = pd.DataFrame({"league": "x", "attacking_team": np.repeat([f"t{i}" for i in range(40)], 100)})
    onehot, _ = C.team_matrix(t)
    sims = C.simulated_variance(np.full(len(t), 0.1), onehot, 2000, [0])
    assert np.var(sims, axis=1, ddof=1).mean() == pytest.approx(0.1 * 0.9 / 100, rel=0.05)


def test_c12a_reads_inside_when_the_teams_differ_by_luck_alone():
    t = pd.DataFrame({"league": "x", "attacking_team": np.repeat([f"t{i}" for i in range(60)], 150)})
    onehot, _ = C.team_matrix(t)
    p = np.random.default_rng(3).uniform(0.0, 0.1, len(t))
    obs = C.simulated_variance(p, onehot, 1, [99])[0]
    assert C.c12a_reading(obs, C.simulated_variance(p, onehot, 2000, [4]))["inside"]
    assert not C.c12a_reading(obs * 3.0, C.simulated_variance(p, onehot, 2000, [4]))["inside"]


def test_a_split_puts_half_of_each_team_s_matches_on_each_side():
    t = pd.DataFrame({"league": "x", "attacking_team": ["A"] * 6 + ["B"] * 6,
                      "match_id": [1, 1, 2, 3, 4, 5] + [1, 2, 3, 4, 5, 5]})
    a = C.first_half(t, np.random.default_rng(5))
    for team, n in (("A", 5), ("B", 5)):
        sub = t[t["attacking_team"] == team].assign(a=a[t["attacking_team"] == team])
        assert sub.groupby("match_id")["a"].first().sum() == n // 2


def test_the_c12a_task_does_not_depend_on_the_worker_count():
    from joblib import Parallel, delayed
    t = pd.DataFrame({"league": np.repeat(["x", "y"], 600),
                      "attacking_team": np.repeat([f"t{i}" for i in range(12)], 100)})
    onehot, teams = C.team_matrix(t)
    p = np.random.default_rng(6).uniform(0, 0.1, len(t))
    obs = C.simulated_variance(p, onehot, 1, [7])[0]
    lg = teams.get_level_values(0).to_numpy()
    args = [(p, onehot, obs, lg, 300, s, "recalibrated") for s in range(3)]
    serial = [C._c12a_task(*a) for a in args]
    par = Parallel(n_jobs=3)(delayed(C._c12a_task)(*a) for a in args)
    assert serial == par


def test_the_flank_twin_reads_its_rows_of_the_shared_tables(monkeypatch, tmp_path):
    """--kind free_kick_flank reads the grain and the shots both populations share and keeps that
    population's rows."""
    deliv, shots = _tables()
    deliv = pd.concat([deliv.assign(set_piece_type="corner"),
                       deliv.assign(set_piece_id=list("xyz"), set_piece_type="free_kick_flank")])
    shots = pd.concat([shots.assign(set_piece_type="corner"),
                       shots.assign(set_piece_id=list("xxyz"), set_piece_type="free_kick_flank")])
    deliv.to_parquet(tmp_path / "grain.parquet")
    shots.to_parquet(tmp_path / "shots.parquet")
    seen = {}

    def table(d, s):
        seen.update(d=set(d["set_piece_id"]), s=set(s["set_piece_id"]))
        raise SystemExit("table built")

    monkeypatch.setattr(C, "corner_table", table)
    monkeypatch.setattr(sys, "argv", ["eda_club_power.py", "--kind", "free_kick_flank",
                                      "--deliveries", str(tmp_path / "grain.parquet"),
                                      "--shots", str(tmp_path / "shots.parquet"),
                                      "--out-dir", str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="table built"):
        C.main()
    assert seen == {"d": set("xyz"), "s": set("xyz")}


def test_another_precision_may_not_write_the_records(monkeypatch):
    def work(*a, **k):
        raise AssertionError("the run read the data before refusing")

    monkeypatch.setattr(C.pd, "read_parquet", work)
    monkeypatch.setattr(sys, "argv", ["eda_club_power.py", "--sims", "10"])
    with pytest.raises(SystemExit, match="published club_power"):
        C.main()
