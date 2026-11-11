"""Tests for run_zone_comparison: every boundary of the six partitions, the canonical mirror of a flank
free kick, the blocks keeping every body, the population, the M1 rule and the parallel paths.

Each boundary is pinned on both sides of its line, so a `<` that becomes `<=` fails here and not in a
count nobody reads.
"""
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from conftest import requires_sb

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "src", _REPO / "pipeline"):
    sys.path.insert(0, str(p))

import run_zone_comparison as Z                                           # noqa: E402
from constants import SIX_YARD_Y                                          # noqa: E402
from s02b_aim import DESIGN_COLS                                          # noqa: E402
from zones import ZONES as IN_PLAY, cells, zones_f                        # noqa: E402

_EPS = 0.01


def _at(fn: Callable[[np.ndarray, np.ndarray], np.ndarray], pts: list) -> list:
    x, y = np.array(pts, float).T
    return list(fn(x, y))


def test_a_cuts_depth_at_the_registered_lines():
    assert _at(Z.HARD["A"], [(96 - _EPS, 40), (96, 70), (108 - _EPS, 5), (108, 40)]) == \
        ["hold_back", "stake_drop", "stake_drop", "attack_cross"]


def test_f_puts_every_line_where_the_laws_and_the_sheet_put_it():
    cases = {(114, 30): "six_yard", (114, 50): "six_yard", (114, 30 - _EPS): "near_post",
             (114, 50 + _EPS): "far_post", (114 - _EPS, 33): "near_post",
             (110, 36): "central", (110, 44): "central", (110, 36 - _EPS): "near_post",
             (110, 44 + _EPS): "far_post", (102, 40): "central", (102 - _EPS, 40): "frontal",
             (96, 40): "frontal", (96 - _EPS, 40): "rest", (99, 18): "frontal", (99, 62): "frontal",
             (99, 18 - _EPS): "rest", (99, 62 + _EPS): "rest", (102, 18): "near_post",
             (102, 18 - _EPS): "short", (102 - _EPS, 18 - _EPS): "rest", (119, 1): "short",
             (110, 62 + _EPS): "rest", (102, 62): "far_post"}
    assert _at(Z.zones_f, list(cases)) == list(cases.values())


def test_the_six_yard_box_is_derived_from_the_posts():
    assert SIX_YARD_Y == (30.0, 50.0)


def test_b_is_f_with_its_frontal_and_short_zones_as_rest():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(85, 120, 20000), rng.uniform(0, 80, 20000)
    f, b = Z.zones_f(x, y), Z.zones_b(x, y)
    keep = ~np.isin(f, ("frontal", "short"))
    assert (b[keep] == f[keep]).all() and (b[~keep] == "rest").all()


def test_d_puts_the_four_jobs_where_the_sheet_puts_them():
    cases = {(108, 18): "arrive", (108, 62): "arrive", (108, 62 + _EPS): "rest",
             (108 - _EPS, 40): "edge", (100, 20): "edge", (100, 60): "edge",
             (100 - _EPS, 40): "rest", (100, 20 - _EPS): "rest", (100, 60 + _EPS): "rest",
             (102, 18 - _EPS): "short", (102 - _EPS, 18 - _EPS): "rest", (96 - _EPS, 40): "hold",
             (96, 40): "rest", (119, 70): "rest"}
    assert _at(Z.zones_d, list(cases)) == list(cases.values())


def test_c_and_f_are_the_pipeline_s_own_partitions():
    """The comparison reads the two partitions in play from src/zones.py, so the one it chose and
    the one the pipeline counts cannot be drawn apart."""
    rng = np.random.default_rng(1)
    x, y = rng.uniform(85, 120, 5000), rng.uniform(0, 80, 5000)
    assert (Z.HARD["C"](x, y) == cells(x, y)).all() and Z.ZONES["C"] == IN_PLAY["C"]
    assert (Z.HARD["F"](x, y) == zones_f(x, y)).all() and Z.ZONES["F"] == IN_PLAY["F"]


def test_a_keeps_the_registered_depth_only_roles():
    """A is the partition the comparison registered on depth alone: unlike s02's jobs, which read
    the width, it keeps a body wide of the box past GOALMOUTH_X in `attack_cross`."""
    assert list(Z.HARD["A"](np.array([110.0, 110.0]), np.array([40.0, 5.0]))) == \
        ["attack_cross", "attack_cross"]


def test_the_table_s_own_zone_counts_leave_before_the_blocks_join():
    """A plan table carrying C and F counts under the blocks' names must not duplicate them, and
    the counts that stay are the ones recounted from the frames."""
    rng = np.random.default_rng(6)
    bodies = pd.DataFrame({"set_piece_id": rng.choice(list("ab"), 40),
                           "side": rng.choice(["att", "def"], 40),
                           "x": rng.uniform(95, 120, 40), "y": rng.uniform(0, 80, 40)})
    t = pd.DataFrame({"set_piece_id": list("ab"), "C_att_edge": [99.0, 99.0]})
    joined = Z.join_blocks(t, bodies, _e(), "corner")
    assert not joined.columns.duplicated().any() and (joined["C_att_edge"] < 99).all()


def test_every_zone_a_grid_can_emit_is_one_it_declares():
    rng = np.random.default_rng(2)
    x, y = rng.uniform(60, 120, 20000), rng.uniform(0, 80, 20000)
    for rep, fn in Z.HARD.items():
        assert set(fn(x, y)) == set(Z.ZONES[rep]), rep
        m = Z.zone_matrix(rep, x, y, None)
        assert m.shape == (len(x), len(Z.ZONES[rep])) and (m.sum(axis=1) == 1).all(), rep


def test_a_zone_a_grid_does_not_declare_stops_the_matrix(monkeypatch):
    monkeypatch.setitem(Z.HARD, "D", lambda x, y: np.full(len(x), "nowhere"))
    with pytest.raises(ValueError, match="does not declare"):
        Z.zone_matrix("D", np.array([100.0]), np.array([40.0]), None)


def _e() -> tuple:
    rng = np.random.default_rng(3)
    centres = [(116, 38), (110, 40), (112, 46), (101, 45), (100, 32), (103, 21)]
    att = pd.DataFrame(np.vstack([rng.normal(c, 1.0, (300, 2)) for c in centres]),
                       columns=["x", "y"])
    return Z.fit_e(att)


def test_e_names_its_components_nearest_the_goal_line_first():
    gmm, order = _e()
    assert np.all(np.diff(gmm.means_[order, 0]) <= 0)


def test_e_shares_a_body_in_front_between_components_and_counts_the_rest_whole():
    e = _e()
    x, y = np.array([110.0, 90.0, 110.0]), np.array([40.0, 40.0, 75.0])
    m = Z.zone_matrix("E", x, y, e)
    assert np.allclose(m.sum(axis=1), 1.0)
    assert m[0, -1] == 0.0 and m[1, -1] == 1.0 and m[2, -1] == 1.0


def test_every_block_keeps_every_body_and_an_empty_frame_counts_zero():
    rng = np.random.default_rng(4)
    n = 200
    bodies = pd.DataFrame({"set_piece_id": rng.choice(list("abcd"), n),
                           "side": rng.choice(["att", "def"], n),
                           "x": rng.uniform(85, 120, n), "y": rng.uniform(0, 80, n)})
    ids = pd.Series(list("abcde"))
    b = Z.blocks(bodies, ids, _e())
    assert list(b.columns) == sum(Z.COLS.values(), []) and len(b) == 5
    for rep in Z.REPS:
        for side in ("att", "def"):
            got = b[[f"{rep}_{side}_{z}" for z in Z.ZONES[rep]]].sum(axis=1).to_numpy()
            want = bodies[bodies["side"] == side].groupby("set_piece_id").size()
            assert np.allclose(got, want.reindex(list(ids), fill_value=0).to_numpy()), (rep, side)
    assert (b.iloc[4] == 0).all()


def _delivery(loc: tuple, players: list) -> tuple:
    d = pd.Series({"id": "k1", "location": list(loc)})
    fz = pd.DataFrame({"event_uuid": ["k1"], "freeze_frame": [[
        {"location": list(xy), "teammate": tm, "keeper": gk, "actor": ac}
        for xy, tm, gk, ac in players]]})
    return d, fz


def test_a_flank_free_kick_from_the_right_is_mirrored_onto_the_left():
    """A kick from y > 40 is read from the attackers' right, so its bodies land where the mirror
    image of the same kick from the left puts them; the taker and both keepers stay out."""
    players = [((95.0, 72.0), True, False, True),       # taker
               ((110.0, 50.0), True, False, False),     # attacker
               ((104.0, 30.0), True, False, False),     # attacker
               ((108.0, 45.0), False, False, False),    # defender
               ((119.0, 40.0), False, True, False),     # defending keeper
               ((30.0, 40.0), True, True, False)]       # attacking keeper
    right, found = Z.frame_bodies(*_delivery((95.0, 72.0), players))
    mirror = [((x, 80.0 - y), tm, gk, ac) for (x, y), tm, gk, ac in players]
    left, found_left = Z.frame_bodies(*_delivery((95.0, 8.0), mirror))
    assert right and not left
    assert sorted(found) == sorted(found_left)
    assert sorted(found) == [("att", 104.0, 50.0), ("att", 110.0, 30.0), ("def", 108.0, 35.0)]


def _tables() -> tuple:
    table = pd.DataFrame({"set_piece_id": list("abcde"), "match_id": [2, 1, 1, 3, 3],
                          "short_corner": [False, False, True, False, False],
                          "delivery_end_x": [116.0, 110.0, 105.0, np.nan, 117.0],
                          "delivery_end_y": [45.0, 38.0, 20.0, 40.0, 35.0],
                          "delivery_x": [120.0, 95.0, 120.0, 120.0, 120.0],
                          "delivery_y": [79.9, 5.0, 0.1, 0.1, 0.1],
                          "corner_side": ["right", "left", "left", "left", "left"]})
    info = pd.DataFrame({"set_piece_id": list("abcde"),
                         "kick_side_right": [True, False, False, False, False],
                         "won_first_contact": [True, None, False, True, False]})
    return table, info


def test_the_population_is_long_deliveries_with_an_end_in_the_canonical_frame():
    table, info = _tables()
    t = Z.population(table, info).set_index("set_piece_id")
    assert list(t.index) == ["b", "a", "e"]                # sorted by match, then id
    assert t.loc["a", "zone"] == "six_near"                # (116, 45) from the right lands at y = 35
    assert t.loc["a", "zone_fine"] == "six_near" and t.loc["a", "zone_coarse"] == "near"
    assert t.loc["a", "kick_y"] == pytest.approx(0.1) and t.loc["b", "kick_y"] == 5.0
    assert list(t["won"]) == [False, True, False]


def test_the_population_stops_on_a_side_or_a_first_contact_that_disagrees():
    table, info = _tables()
    bad = info.assign(kick_side_right=[False] * 5)
    with pytest.raises(SystemExit, match="kick side"):
        Z.population(table, bad)
    grain = info[["set_piece_id"]].assign(won_first_contact=[False, None, False, True, False])
    with pytest.raises(SystemExit, match="first contact"):
        Z.population(table, info, grain)
    Z.population(table, info, info[["set_piece_id", "won_first_contact"]])


def _refit(rows: dict) -> list:
    return [{"key": f"{R}-F", "bonf_hi": rows.get(R, (0.0, 0.01))[1],
             "point": rows.get(R, (0.0, 0.01))[0]} for R in Z.RIVALS]


def test_m1_replaces_f_only_on_a_clean_steady_bound_below_zero():
    clean = {R: True for R in Z.REPS}
    assert Z.decide(_refit({}), {}, clean)["keeps"] == "F"
    one = _refit({"C": (-0.004, -0.001)})
    assert Z.decide(one, {"C-F": True}, clean)["keeps"] == "C"
    assert Z.decide(one, {"C-F": False}, clean)["keeps"] == "F"
    assert Z.decide(one, {"C-F": True}, {**clean, "C": False})["keeps"] == "F"
    two = _refit({"C": (-0.004, -0.001), "E": (-0.006, -0.002)})
    v = Z.decide(two, {"C-F": True, "E-F": True}, clean)
    assert v["keeps"] == "E" and v["qualify"] == ["C", "E"]


def test_the_normal_form_is_the_point_plus_or_minus_half_the_percentile_width():
    rng = np.random.default_rng(5)
    draws = rng.normal(-0.01, 0.002, (4000, 1)) ** 3 * 1e4
    point = np.array([0.003])
    f = Z.forms(point, draws, 0.01)
    lo, hi = np.percentile(draws, [0.5, 99.5])
    assert f["n_lo"][0] == pytest.approx(point[0] - (hi - lo) / 2)
    assert f["n_hi"][0] == pytest.approx(point[0] + (hi - lo) / 2)
    assert f["p_hi"][0] == pytest.approx(2 * point[0] - lo)


def test_the_contrast_names_follow_the_differences():
    loss = {k: np.arange(3.0) + i for i, k in enumerate(("design", *Z.REPS))}
    assert list(Z.diffs(loss, "design")) == Z.keys("design")


def test_one_team_that_carries_a_difference_is_named():
    t = pd.DataFrame({"won": True, "attacking_team": ["x"] * 2 + ["y"] * 8,
                      "match_id": np.arange(10)})
    loss = {R: np.zeros(10) for R in Z.REPS}
    loss["A"] = np.array([-1.0, -1.0] + [0.01] * 8)
    row = next(r for r in Z.influence(t, "corner", loss)
               if r["key"] == "A-F" and r["unit"] == "attacking_team")
    assert row["who"] == "x" and row["flips"]


def test_no_input_reads_a_barred_field(monkeypatch):
    Z.assert_sources()
    monkeypatch.setitem(Z._SOURCE, "F_att_short", "pass.end_location")
    with pytest.raises(ValueError, match="barred"):
        Z.assert_sources()


@pytest.mark.parametrize("extra", [
    [], ["--robustness"], ["--out-dir", str(_REPO / "data" / "derived" / "auditoria_2026-09-30" / "zones")]])
def test_another_precision_may_not_write_the_comparison(monkeypatch, extra):
    """Neither the recipe's records in data/derived nor the plan's, and the robustness readings no
    more than the comparison."""
    def work(*a, **k):
        raise AssertionError("the run did the work before refusing")

    monkeypatch.setattr(Z, "build", work)
    monkeypatch.setattr(Z.pd, "read_parquet", work)
    monkeypatch.setattr(sys, "argv", ["run_zone_comparison.py", "--draws", "300", *extra])
    with pytest.raises(SystemExit, match="may not write"):
        Z.main()


def test_other_tables_may_not_write_the_comparison(monkeypatch, tmp_path):
    """Tables outside data/derived are not the published ones, so a run on them at the registered
    precision would still put its readings under the names the record is read from."""
    def work(*a, **k):
        raise AssertionError("the run did the work before refusing")

    monkeypatch.setattr(Z, "build", work)
    monkeypatch.setattr(Z.pd, "read_parquet", work)
    monkeypatch.setattr(sys, "argv", ["run_zone_comparison.py", "--tables", str(tmp_path)])
    with pytest.raises(SystemExit, match="not the published ones"):
        Z.main()


def test_each_population_is_checked_against_its_rows_of_the_combined_grain(monkeypatch, tmp_path):
    """The recipe writes one delivery grain for both populations, and each population's first
    contact is checked against its own rows of it, the flank's as the corner's."""
    table, info = _tables()
    twin = list("vwxyz")
    flank, finfo = table.assign(set_piece_id=twin), info.assign(set_piece_id=twin)
    table.to_parquet(tmp_path / "corners.parquet")
    flank.to_parquet(tmp_path / "corners_free_kick_flank.parquet")
    pd.concat([info.assign(set_piece_type="corner"),
               finfo.assign(set_piece_type="free_kick_flank",
                            won_first_contact=[False, None, False, True, False])]
              ).to_parquet(tmp_path / "eda_deliveries.parquet")
    built = {"corner": info, "free_kick_flank": finfo}
    no_bodies = pd.DataFrame(columns=["set_piece_id", "side", "x", "y"])
    monkeypatch.setattr(Z, "build", lambda t, kind, jobs: (built[kind], no_bodies))
    monkeypatch.setattr(sys, "argv", ["run_zone_comparison.py", "--counts-only", "--tables",
                                      str(tmp_path), "--out-dir", str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="first contact"):
        Z.main()


def test_the_robustness_readings_run_on_either_population(monkeypatch, tmp_path):
    """Registered on the corners, the readings added after the run read the flank free kicks the same
    way, every row and every saved draw named after its population."""
    t = _synthetic()
    n = len(Z.keys("design"))
    loss = {a: np.full(int(t["won"].sum()), float(i)) for i, a in enumerate(("design", *Z.REPS))}
    paths = []

    def draws(t, stat, k, seed, jobs, path, resume):
        paths.append(path.name)
        return None, np.zeros((k, n))

    monkeypatch.setattr(Z, "plan_losses", lambda t, *a, **k: loss)
    monkeypatch.setattr(Z, "seed_draws", draws)
    args = SimpleNamespace(boot=20, draws=4, jobs=1, resume=False)
    rows = Z.robustness(t, "free_kick_flank", args, tmp_path, "")
    assert {r["population"] for r in rows} == {"free_kick_flank"} and rows[-1]["key"] == "C confirmed"
    assert len(paths) == len(Z.ROBUSTNESS) and all("_free_kick_flank_" in p for p in paths)


def _synthetic(n_matches: int = 24, per: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_matches * per
    t = pd.DataFrame({"match_id": np.repeat(np.arange(n_matches), per),
                      "zone": rng.choice(["six_near", "six_far", "spot_central", "other"], n),
                      "won": rng.random(n) < 0.6, "kick_x": 120.0, "kick_y": rng.uniform(0, 1, n),
                      "swing": rng.choice(["Inswinging", "Outswinging"], n),
                      "corner_side": rng.choice(["left", "right"], n),
                      "taker_foot": "Right Foot", "delivery_height": "High Pass"})
    t["zone_base"] = t["zone"]
    t["zone_coarse"] = rng.choice(["near", "central", "far", "other"], n)
    t["league"] = rng.choice(["x", "y"], n)
    blocks = pd.DataFrame(rng.poisson(1.0, (n, len(sum(Z.COLS.values(), [])))).astype(float),
                          columns=sum(Z.COLS.values(), []))
    return pd.concat([t, blocks], axis=1)


def test_the_refit_draws_do_not_depend_on_the_worker_count():
    t = _synthetic()
    assert set(DESIGN_COLS) <= set(t.columns)
    p1, d1 = Z.refit_draws(t, Z.plan_stats, 2, 0, 1)
    p2, d2 = Z.refit_draws(t, Z.plan_stats, 2, 0, 2)
    assert np.array_equal(p1, p2) and np.array_equal(d1, d2)


def test_a_random_partition_is_another_partition_of_the_same_rows():
    t = _synthetic()
    a, b = Z.plan_stats(t), Z.plan_stats(t, 1)
    assert a.shape == b.shape == (len(Z.keys("design")),) and not np.array_equal(a, b)


def test_the_aim_can_be_read_on_another_grid():
    t = _synthetic()
    a, b = Z.plan_stats(t), Z.plan_stats(t, grid="coarse")
    assert a.shape == b.shape and not np.array_equal(a, b)


def test_the_league_enters_every_arm_and_an_undeclared_column_does_not():
    t = _synthetic(n_matches=6, per=5)
    assert Z.design(t, ("league",)).shape[1] == Z.design(t).shape[1] + 2
    with pytest.raises(ValueError, match="declared source"):
        Z.design(t, ("attacking_team",))


def test_each_added_reading_names_the_bound_that_confirms_c():
    assert [r[3] for r in Z.ROBUSTNESS] == ["bonf_hi", "hi", "hi"]
    assert [r[1] for r in Z.ROBUSTNESS] == ["base", "fine", "coarse"]
    assert Z.ROBUSTNESS[0][2] == ("league",)


@requires_sb
def test_the_frames_do_not_depend_on_the_worker_count(flank_mids):
    from s01_phases import SetPieceGrammar
    import loader_sb as L
    rows = [{"match_id": m, "set_piece_id": i} for m in flank_mids[:3]
            for i in SetPieceGrammar(L.load_events(m), kinds=("corner",)).deliveries()["id"]]
    table = pd.DataFrame(rows)
    i1, b1 = Z.build(table, "corner", 1)
    i2, b2 = Z.build(table, "corner", 3)
    assert len(i1) and len(b1)
    pd.testing.assert_frame_equal(i1, i2)
    pd.testing.assert_frame_equal(b1, b2)


def test_a_limited_run_reads_the_tables_the_smoke_built(tmp_path):
    """--limit N reads the N-match tables and grain a smoke run wrote, never the grant's cut down."""
    for name in ("corners", "corners_free_kick_flank", "eda_deliveries"):
        pd.DataFrame({"match_id": [7]}).to_parquet(tmp_path / f"{name}_limit3.parquet")
    grain, tables = Z.inputs(tmp_path, "_limit3")
    assert len(grain) == 1 and set(tables) == set(Z.POPULATIONS)
