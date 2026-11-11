"""Tests for eda_top_teams: the counts it builds on, the exact test, the selections and the draws.

What these pin is where a team comparison goes wrong without anything failing: an own goal credited
to a team's finishing, a corner classed by a shot that is not its first, a selection that sees the
half it is measured on, or a group rate averaged over teams when it is pooled over corners.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "pipeline"))

import eda_top_teams as tt                                   # noqa: E402


def _shots(rows: list) -> pd.DataFrame:
    """Shot rows (set_piece_id, team, match, segment, t, xg, goal, outcome) on one league."""
    return pd.DataFrame(rows, columns=["set_piece_id", "attacking_team", "match_id", "segment",
                                       "t_since_delivery_s", "xg", "is_goal", "shot_outcome"]
                        ).assign(league="L", set_piece_type="corner")


def test_own_goals_leave_every_count_and_each_shot_takes_its_route():
    s = tt.routes(_shots([("a", "X", 1, "direct", 1.0, 0.3, True, "Goal"),
                          ("a", "X", 1, "after_clearance", 3.0, 0.0, True, "Own Goal"),
                          ("b", "X", 1, "flick_on", 2.0, 0.2, False, "Saved"),
                          ("c", "X", 1, "late", 9.0, 0.1, False, "Off T")]))
    assert "Own Goal" not in set(s["shot_outcome"])
    assert dict(zip(s["set_piece_id"], s["route"], strict=True)) == {"a": "direct", "b": "aft",
                                                                      "c": "late"}


def test_a_corner_takes_the_route_of_its_first_shot_and_an_own_goal_alone_is_no_shot():
    """Each corner counts once, by the shot that came first; a corner whose only terminal event
    was an own goal produced no shot of the team's own."""
    deliv = pd.DataFrame({"set_piece_id": ["a", "b", "c"], "attacking_team": "X", "match_id": 1,
                          "league": "L"})
    s = tt.routes(_shots([("a", "X", 1, "after_clearance", 4.0, 0.1, False, "Saved"),
                          ("a", "X", 1, "direct", 1.0, 0.4, True, "Goal"),
                          ("b", "X", 1, "after_clearance", 2.0, 0.0, True, "Own Goal")]))
    factors = pd.DataFrame({r: [1.0] for r in tt.ROUTES},
                           index=pd.MultiIndex.from_tuples([("L", "X")], names=["league", "team"]))
    rows = tt.team_match_rows(deliv, s, factors).iloc[0]
    assert (rows["corners"], rows["corners_direct"], rows["corners_aft"], rows["corners_none"]) == (3, 1, 0, 2)
    assert rows["goals_total"] == 1 and rows["xg_aft"] == pytest.approx(0.1)


def test_the_recalibration_factor_of_a_team_never_reads_its_own_finishing():
    s = tt.routes(_shots([("a", "X", 1, "flick_on", 2.0, 0.5, True, "Goal"),
                          ("b", "X", 1, "flick_on", 2.0, 0.5, True, "Goal"),
                          ("c", "Y", 2, "flick_on", 2.0, 2.0, True, "Goal")]))
    teams = pd.MultiIndex.from_tuples([("L", "X"), ("L", "Y")], names=["league", "team"])
    f = tt.lto_factors(s, teams)["aft"]
    assert f[("L", "X")] == pytest.approx(1 / 2.0) and f[("L", "Y")] == pytest.approx(2 / 1.0)


def test_the_exact_test_is_the_poisson_binomial_counted_by_hand():
    """Two-sided, from the full distribution of goals over shots of unequal quality."""
    probs, k = np.array([0.1, 0.5, 0.3]), 2
    pmf = np.zeros(4)
    for hits in itertools.product([0, 1], repeat=3):
        pmf[sum(hits)] += np.prod([p if h else 1 - p for p, h in zip(probs, hits, strict=True)])
    assert tt.poisson_binomial_p(probs, k) == pytest.approx(min(1.0, 2 * min(pmf[:3].sum(), pmf[2:].sum())))
    assert tt.poisson_binomial_p(np.array([]), 0) == 1.0


def test_the_corner_goal_leaders_break_ties_by_xg_and_then_by_name():
    t = pd.DataFrame({"team": ["c", "a", "b", "d"], "goals_total": [5, 5, 5, 9],
                      "xg_total": [3.0, 4.0, 4.0, 1.0]})
    assert list(t.loc[tt.top_by(t, "goals_total", "xg_total", 3), "team"]) == ["a", "b", "d"]


def test_the_draws_and_the_permutations_stay_inside_their_strata():
    rng = np.random.default_rng(0)
    strata = np.array(["x", "x", "x", "y", "y"])
    c = tt.draw_counts(strata, 50, rng)
    assert (c[:, :3].sum(axis=1) == 3).all() and (c[:, 3:].sum(axis=1) == 2).all()
    member = np.array([True, False, False, True, False])
    p = tt.permuted(member, strata, 200, rng)
    assert (p[:, :3].sum(axis=1) == 1).all() and (p[:, 3:].sum(axis=1) == 1).all()


def test_the_split_half_null_relabels_teams_the_same_way_in_every_split():
    """Two splits choosing nine teams in common keep exactly nine in common under every null draw,
    which is the persistence an independent draw per split would destroy."""
    a = np.arange(30) < 10
    b = (np.arange(30) < 9) | (np.arange(30) == 10)
    relabel = np.array([np.random.default_rng(k).permutation(30) for k in range(50)])
    na, nb = tt.relabelled(a, relabel), tt.relabelled(b, relabel)
    assert (na.sum(axis=1) == 10).all() and ((na & nb).sum(axis=1) == 9).all()


def test_a_group_rate_is_pooled_over_its_corners_not_averaged_over_its_teams():
    num, den = np.array([1.0, 9.0, 2.0]), np.array([10.0, 30.0, 20.0])
    gap = tt.rate_gap(num, den, np.array([True, True, False]))[0]
    assert gap == pytest.approx(10.0 / 40.0 - 2.0 / 20.0)


def test_each_team_s_first_half_is_half_its_matches_and_only_the_team_s_own():
    """The selection sees the first half and the measure the second, match by match."""
    rows = pd.DataFrame({"league": "L", "team": ["X"] * 4 + ["Y"] * 3,
                         "match_id": [1, 2, 3, 4, 1, 2, 5]})
    games = pd.Series({("L", "X"): np.array([1, 2, 3, 4]), ("L", "Y"): np.array([1, 2, 5, 6])})
    m = tt.half_masks(rows, games, 40, 0)
    assert (m[:, :4].sum(axis=1) == 2).all()                  # 4 matches, 2 in the first half
    assert (m[:, 4:].sum(axis=1) <= 2).all()                  # Y's halves are its own four
    assert m[:, 4:].sum(axis=1).min() < m[:, 4:].sum(axis=1).max()


def _synthetic(n_leagues: int = 3, per: int = 8, n_matches: int = 6, seed: int = 0) -> tuple:
    """A season table, its (team, match) rows and the match lists, every metric column filled."""
    rng = np.random.default_rng(seed)
    rows, games = [], {}
    for li in range(n_leagues):
        for ti in range(per):
            lg, tm = f"L{li}", f"L{li}_t{ti}"
            games[(lg, tm)] = np.array([1000 * li + 100 * ti + j for j in range(n_matches)])
            for mid in games[(lg, tm)]:
                c = int(rng.integers(2, 9))
                xg = {r: float(rng.gamma(2.0, 0.01) * c) for r in tt.ROUTES}
                goals = {r: int(rng.poisson(xg[r])) for r in tt.ROUTES}
                shots = {r: int(goals[r] + rng.integers(0, 4)) for r in tt.ROUTES}
                rows.append({"league": lg, "team": tm, "match_id": int(mid), "corners": c,
                             "corners_aft": int(rng.integers(0, c + 1)),
                             "corners_direct": int(rng.integers(0, c + 1)),
                             **{f"xg_{r}": xg[r] for r in tt.ROUTES},
                             **{f"goals_{r}": goals[r] for r in tt.ROUTES},
                             **{f"shots_{r}": shots[r] for r in tt.ROUTES},
                             **{f"perf_{r}": goals[r] - xg[r] for r in tt.ROUTES},
                             **{f"perf_cal_{r}": goals[r] - 0.9 * xg[r] for r in tt.ROUTES},
                             "xg_total": sum(xg.values()), "goals_total": sum(goals.values())})
    rows = pd.DataFrame(rows)
    t = rows.drop(columns="match_id").groupby(["league", "team"]).sum().reset_index()
    t["matches"] = n_matches
    t["pos"] = t.groupby("league")["xg_total"].rank(ascending=False, method="first").astype(int)
    return t, t.copy(), rows, pd.Series(games)


def test_the_battery_over_the_cores_reproduces_the_serial_run_exactly():
    """The tasks are seeded by (seed, index) and single-threaded, so the worker count cannot move a
    single figure; and a season point is a fact of the data, the same at every seed."""
    t, t_long, rows, games = _synthetic()
    kw = dict(n_boot=60, n_perm=300, n_splits=3, n_split_perm=40, seeds=(0, 1))
    one, two = (tt.battery(t, t_long, rows, games, jobs=j, **kw)
                .sort_values(["seed", "group", "scope", "metric"]).reset_index(drop=True) for j in (1, 2))
    pd.testing.assert_frame_equal(one, two)
    season = one[(one["scope"] == "season") & (one["metric"] == tt.PRIMARY) & (one["group"] == "T2")]
    assert season["point"].nunique() == 1 and season["lo95"].nunique() == 2


def test_influence_leaves_each_member_out_once():
    t, _, _, _ = _synthetic()
    member = tt.groups(t)["G"][0]
    infl = tt.influence(t, member)
    assert len(infl) == member.sum() and set(infl["team"]) == set(t.loc[member, "team"])


def test_without_a_long_corner_table_the_long_scope_is_left_out():
    """A free kick is never short, so on the flank the long-corner scope would repeat the season."""
    t, t_long, _, _ = _synthetic()
    assert "long_corners" in {s["scope"] for s in tt.specs(t, t_long)}
    assert "long_corners" not in {s["scope"] for s in tt.specs(t, None)}


def test_the_flank_twin_reads_its_rows_of_the_shared_tables(monkeypatch, tmp_path):
    """--kind free_kick_flank reads the grain and the shots both populations share and keeps that
    population's rows: its shots, and the teams of its deliveries."""
    deliv = pd.DataFrame({"set_piece_id": list("abxy"), "league": "L", "match_id": 1,
                          "attacking_team": ["X", "X", "Y", "Y"], "short_corner": False,
                          "set_piece_type": ["corner"] * 2 + ["free_kick_flank"] * 2})
    shots = _shots([("a", "X", 1, "direct", 1.0, 0.3, True, "Goal"),
                    ("x", "Y", 1, "flick_on", 2.0, 0.2, False, "Saved")])
    shots.loc[1, "set_piece_type"] = "free_kick_flank"
    paths = {}
    for name, frame in (("deliveries", deliv), ("shots", shots), ("matches", pd.DataFrame({"m": [1]})),
                        ("standings", pd.DataFrame({"s": [1]}))):
        paths[name] = tmp_path / f"{name}.parquet"
        frame.to_parquet(paths[name])
    seen = {}

    def factors(s, teams):
        seen.update(s=set(s["set_piece_id"]), teams=list(teams))
        raise SystemExit("factors read")

    monkeypatch.setattr(tt, "lto_factors", factors)
    monkeypatch.setattr(sys, "argv", ["eda_top_teams.py", "--kind", "free_kick_flank",
                                      "--out-dir", str(tmp_path / "out"),
                                      *[a for k, p in paths.items() for a in (f"--{k}", str(p))]])
    with pytest.raises(SystemExit, match="factors read"):
        tt.main()
    assert seen == {"s": {"x"}, "teams": [("L", "Y")]}


def test_a_reduced_run_may_not_write_the_published_records(monkeypatch):
    def work(*a, **k):
        raise AssertionError("the run read its inputs before refusing")

    monkeypatch.setattr(tt.pd, "read_parquet", work)
    monkeypatch.setattr(sys, "argv", ["eda_top_teams.py", "--boot", "100"])
    with pytest.raises(SystemExit, match="published top_teams"):
        tt.main()
