"""Tests for eda_repeatability.split_half: what its published interval is an interval of.

The spread of the correlation over random partitions of the SAME teams is not a confidence interval:
it holds the teams fixed, and the teams are the sample. So the partition spread is not published
under the interval's name at all, and the interval that is published comes from drawing teams with
replacement inside their league and re-averaging over the same partitions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import eda_repeatability as E


@pytest.fixture
def two_matches() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Twelve teams in two leagues, two matches each, with corner volume that differs by team."""
    rng = np.random.default_rng(0)
    d_rows, s_rows = [], []
    for t in range(12):
        for mid in (100 * t, 100 * t + 1):
            for c in range(2 + t % 4 + mid % 3):
                sid = f"{t}-{mid}-{c}"
                d_rows.append({"league": "a" if t < 6 else "b", "attacking_team": f"T{t}",
                               "match_id": mid, "set_piece_id": sid})
                if rng.random() < 0.5:
                    s_rows.append({"attacking_team": f"T{t}", "match_id": mid, "set_piece_id": sid,
                                   "segment": rng.choice(["direct", "after_clearance"]),
                                   "is_goal": int(rng.random() < 0.2)})
    return pd.DataFrame(d_rows), pd.DataFrame(s_rows)


def test_no_partition_bracket_is_published_as_an_interval(two_matches):
    """Without resampling teams no interval is claimed; the partition spread is not one."""
    deliv, shots = two_matches
    rel = E.split_half(deliv, shots, n_splits=20, seed=0).set_index("metric")
    r = rel.loc["corners_per_match"]
    assert np.isfinite(r["half_r"]), "the fixture must leave the control measurable"
    assert np.isnan(r["half_lo"]) and np.isnan(r["half_hi"])


def test_the_team_bootstrap_has_width_and_moves_nothing_else(two_matches):
    deliv, shots = two_matches
    base = E.split_half(deliv, shots, n_splits=20, seed=0)
    boot = E.split_half(deliv, shots, n_splits=20, seed=0, n_boot=300)
    pd.testing.assert_series_equal(base["half_r"], boot["half_r"])
    pd.testing.assert_series_equal(base["full_r_sb"], boot["full_r_sb"])
    r = boot.set_index("metric").loc["corners_per_match"]
    assert r["half_lo"] < r["half_r"] < r["half_hi"]
    assert r["full_lo"] == pytest.approx(2 * r["half_lo"] / (1 + r["half_lo"]))
    assert r["full_hi"] == pytest.approx(2 * r["half_hi"] / (1 + r["half_hi"]))
    assert (boot["n_boot"] == 300).all()


def test_boot_cols_narrows_the_draws_without_moving_the_bounds_it_keeps(two_matches):
    deliv, shots = two_matches
    full = E.split_half(deliv, shots, n_splits=10, seed=0, n_boot=100).set_index("metric")
    part = E.split_half(deliv, shots, n_splits=10, seed=0, n_boot=100,
                        boot_cols=["corners_per_match"]).set_index("metric")
    # Equal to rounding, not to the bit: a narrower block changes the order numpy sums in.
    for b in ("half_lo", "half_hi"):
        assert part.loc["corners_per_match", b] == pytest.approx(
            full.loc["corners_per_match", b], abs=1e-12)
    assert part.drop(index="corners_per_match")[["half_lo", "half_hi"]].isna().all().all()


def test_a_draw_keeps_every_league_at_its_own_team_count():
    leagues = np.array(["a"] * 5 + ["b"] * 3 + ["c"] * 4)
    draws = E._team_draws(leagues, n_boot=50, rng=np.random.default_rng(1))
    assert draws.shape == (50, 12)
    for d in draws:
        assert (leagues[d] == "a").sum() == 5 and (leagues[d] == "c").sum() == 4


def test_a_run_off_the_recipe_s_precision_needs_out_dir(monkeypatch):
    """The dispersion and split-half records are what evidence.md §12 reads and what keeps a
    per-team ranking barred, so a run at fewer splits, simulations or team draws than the recipe
    may not write them into data/derived, and refuses before reading anything."""
    def work(*a, **k):
        raise AssertionError("the run did the work before refusing")

    monkeypatch.setattr(E, "kind_inputs", work)
    monkeypatch.setattr(sys, "argv", ["eda_repeatability.py", "--splits", "50"])
    with pytest.raises(SystemExit, match="may not write the published"):
        E.main()


def test_the_flags_name_the_grain_and_the_shots_read(monkeypatch, tmp_path):
    """The smoke reads the N-match grain and shots it built, never the grant's."""
    seen = []

    def read(path, *a, **k):
        seen.append(Path(path))
        if len(seen) == 2:
            raise RuntimeError("read")
        return pd.DataFrame()

    monkeypatch.setattr(E.pd, "read_parquet", read)
    monkeypatch.setattr(sys, "argv", ["eda_repeatability.py", "--deliveries", str(tmp_path / "d.parquet"),
                                      "--shots", str(tmp_path / "s.parquet"), "--out-dir", str(tmp_path)])
    with pytest.raises(RuntimeError, match="read"):
        E.main()
    assert seen == [tmp_path / "d.parquet", tmp_path / "s.parquet"]


def test_other_inputs_never_take_the_published_names(monkeypatch, tmp_path):
    monkeypatch.setattr(E.pd, "read_parquet", lambda *a, **k: pytest.fail("read before refusing"))
    monkeypatch.setattr(sys, "argv", ["eda_repeatability.py", "--deliveries", str(tmp_path / "d.parquet")])
    with pytest.raises(SystemExit, match="not the grant's"):
        E.main()


def _season(seed: int = 3) -> tuple:
    """(deliveries, shots, standings) of two leagues of six attacking sides, each league with a
    seventh side that only defends, so the team table carries a row the attack never fills."""
    rng = np.random.default_rng(seed)
    d_rows, s_rows = [], []
    for lg in ("a", "b"):
        for i in range(6):
            for m in range(5):
                mid = f"{lg}{i}{m}"
                for c in range(int(rng.integers(2, 8))):
                    sid = f"{mid}-{c}"
                    d_rows.append({"league": lg, "attacking_team": f"{lg}{i}", "match_id": mid,
                                   "defending_team": f"{lg}{(i + 1) % 6 if m else 'x'}",
                                   "set_piece_id": sid})
                    s_rows += [{"league": lg, "attacking_team": f"{lg}{i}", "match_id": mid,
                                "set_piece_id": sid, "is_goal": bool(rng.random() < 0.2),
                                "segment": rng.choice(["direct", "after_clearance", "flick_on", "late"]),
                                "xg": float(rng.uniform(0, 0.3)), "shot_outcome": "Saved"}
                               for _ in range(int(rng.random() < 0.4))]
    deliv = pd.DataFrame(d_rows)
    shots = pd.DataFrame(s_rows).merge(deliv[["set_piece_id", "defending_team"]], on="set_piece_id")
    teams = sorted(set(deliv["attacking_team"]) | set(deliv["defending_team"]))
    stand = pd.DataFrame({"league": [t[0] for t in teams], "team": teams, "team_ref": teams,
                          "pos": [int(t[1:]) + 1 if t[1:].isdigit() else 7 for t in teams],
                          "pts_ref": 0, "played": 5})
    return deliv, shots, stand


def test_the_dispersion_reads_the_team_table_s_totals_off_the_grain(monkeypatch, tmp_path):
    """The five totals the dispersion test reads are counted here off the grain and the shots, so
    this driver runs before eda_second_ball, whose team table reads the reliability written here.
    Every test must come out exactly as on that table, whose row order the simulation draws in."""
    import eda_second_ball as S
    deliv, shots, stand = _season()
    monkeypatch.setattr(S, "_DER", tmp_path)               # no reliability on disk: nothing shrinks
    table = S.team_table(deliv, shots, stand, "corner")
    totals = E.team_totals(deliv, shots)
    assert len(table) == len(totals) + 2                   # the two sides that only defend
    for num, den in E.DISPERSION:
        assert E.dispersion_test(totals, num, den, 200) == E.dispersion_test(table, num, den, 200)


def test_an_own_goal_is_a_goal_and_no_shot_in_the_team_totals(monkeypatch, tmp_path):
    """The team table and the dispersion test count the same way: an own goal adds to the goals,
    after-first-touch goals included, and to no shot count, and the two drivers still agree on every
    pair. Without the rule a team's shots would count a goal no model priced."""
    import eda_second_ball as S
    deliv, shots, stand = _season()
    team = shots.iloc[0]["attacking_team"]
    og = shots.iloc[[0]].assign(is_goal=True, xg=0.0, shot_outcome="Own Goal", segment="after_clearance")
    shots = pd.concat([shots, og], ignore_index=True)
    monkeypatch.setattr(S, "_DER", tmp_path)
    table, totals = S.team_table(deliv, shots, stand, "corner"), E.team_totals(deliv, shots)
    mine = shots["attacking_team"] == team
    row, tot = table[table["team"] == team].iloc[0], totals[totals["team"] == team].iloc[0]
    assert row["shots_for"] == tot["shots_for"] == (mine & (shots["shot_outcome"] != "Own Goal")).sum()
    assert row["goals_for"] == tot["goals_for"] == shots.loc[mine, "is_goal"].sum()
    assert row["og_goals_for"] == 1 and row["og_after_clearance_for"] == 1
    aft = mine & shots["segment"].isin(E.AFTER_FIRST_TOUCH)
    assert row["aft_goals_for"] == tot["aft_goals_for"] == shots.loc[aft, "is_goal"].sum()
    for num, den in E.DISPERSION:
        assert E.dispersion_test(totals, num, den, 200) == E.dispersion_test(table, num, den, 200)


def test_the_split_half_shot_metrics_ignore_an_own_goal(two_matches):
    """shots_per_corner and aft_shot_share are shot quantities: an own goal added to the shots
    moves neither."""
    deliv, shots = two_matches
    shots = shots.assign(shot_outcome="Saved")
    own = shots.iloc[:4].assign(shot_outcome="Own Goal", is_goal=1, segment="after_clearance")
    base = E.split_half(deliv, shots, n_splits=20, seed=0).set_index("metric")
    plus = E.split_half(deliv, pd.concat([shots, own], ignore_index=True), n_splits=20,
                        seed=0).set_index("metric")
    for m in ("corners_per_match", "shots_per_corner", "aft_shot_share"):
        assert plus.loc[m, "half_r"] == base.loc[m, "half_r"]


def _two_leagues(n_teams: int = 20, corners: int = 400) -> pd.DataFrame:
    """Two leagues whose rates differ and whose teams do not: every team scores at exactly its
    league's rate."""
    return pd.DataFrame([{"league": lg, "team": f"{lg}{i}", "corners_for": corners,
                          "aft_goals_for": round(corners * rate)}
                         for lg, rate in (("a", 0.01), ("b", 0.05)) for i in range(n_teams)])


def test_a_league_difference_is_not_a_team_difference():
    """One probability over both leagues reads the gap between them as team excess; the null that
    gives each league its own probability, which decides, does not."""
    r = E.dispersion_test(_two_leagues(), "aft_goals_for", "corners_for", 500)
    assert r["excess"] and not r["excess_by_league"]
    assert r["p_value_by_league"] > 0.05


def test_a_monte_carlo_p_value_never_reads_zero():
    """No null draw reaches the observed spread, and the p is still one in the draws plus one."""
    r = E.dispersion_test(_two_leagues(), "aft_goals_for", "corners_for", 500)
    assert r["p_value"] == pytest.approx(1 / 501)


def test_the_vectorised_spearman_is_scipy_on_the_pairs_both_halves_carry():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(7, 30)), rng.normal(size=(7, 30))
    a[0, 3], b[1, 5], a[2, :26] = np.nan, np.nan, np.nan
    got = E._spearman_rows(a, b)
    for i in range(7):
        m = np.isfinite(a[i]) & np.isfinite(b[i])
        want = stats.spearmanr(a[i, m], b[i, m])[0] if m.sum() > 5 else np.nan
        assert got[i] == pytest.approx(want, abs=1e-12, nan_ok=True)
