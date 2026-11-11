"""Tests for run_zone_g: G1 and G2 cut the box where the sheet cuts it and C's cells outside it, every
arm counts every body once, the contrasts are each rival against C, the decision needs all of the sheet's
conditions for the rival it keeps, the screen's go and borderline are the sheet's, and the precision is
M1's.
"""
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "src", _REPO / "pipeline"):
    sys.path.insert(0, str(p))

import run_zone_comparison as Z                                           # noqa: E402
import run_zone_g as G                                                    # noqa: E402
from zones import cells, in_box                                           # noqa: E402


def _parts() -> dict:
    return G.partitions_of()


def _grid() -> tuple:
    x, y = np.meshgrid(np.arange(60.0, 120.01, 0.25), np.arange(0.0, 80.01, 0.25))
    return x.ravel(), y.ravel()


def test_g1_and_g2_cut_the_box_where_the_sheet_cuts_it():
    pts = [(116.5, 33.0), (116.5, 40.0), (116.5, 47.0), (110.0, 33.0), (110.0, 40.0), (110.0, 47.0),
           (106.0, 40.0), (103.0, 40.0), (116.5, 24.0), (104.0, 25.0), (110.0, 56.0), (99.0, 40.0),
           (99.0, 10.0), (110.0, 70.0), (80.0, 40.0)]
    x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    shared = ["six_near", "six_central", "six_far", "spot_near", "spot_central", "spot_far"]
    tail = ["near_side", "near_side", "far_side", "edge", "wide", "wide", "deep"]
    assert list(G.zones_g1(x, y)) == shared + ["front_central", "front_central"] + tail
    assert list(G.zones_g2(x, y)) == shared + ["spot_central", "edge"] + tail


def test_the_thirds_are_equal_and_span_the_six_yard_width():
    u0, u1 = G.SIX_YARD_Y
    assert np.isclose(G._T1 - u0, G._T2 - G._T1) and np.isclose(u1 - G._T2, G._T1 - u0)
    x, y = np.full(3, 116.0), np.array([G._T1 - 0.01, G._T1 + 0.01, G._T2 + 0.01])
    assert list(G.zones_g1(x, y)) == ["six_near", "six_central", "six_far"]


def test_outside_the_box_both_rivals_are_cs_cells():
    x, y = _grid()
    out = ~in_box(x, y)
    for label in (G.zones_g1, G.zones_g2):
        assert list(label(x[out], y[out])) == list(cells(x[out], y[out]))


def test_every_zone_a_rival_declares_is_one_it_emits_and_no_other():
    x, y = _grid()
    for label, zones in ((G.zones_g1, G.G1_ZONES), (G.zones_g2, G.G2_ZONES)):
        assert set(label(x, y)) == set(zones)
    assert (len(G.G1_ZONES), len(G.G2_ZONES)) == (12, 11)


def test_g2_differs_from_g1_only_in_the_central_block_behind_the_spot_line():
    x, y = _grid()
    a, b = G.zones_g1(x, y), G.zones_g2(x, y)
    u0, u1 = G.SIX_YARD_Y
    moved = a != b
    assert moved.any() and ((x[moved] >= 102) & (x[moved] < 108) & (y[moved] >= u0) & (y[moved] <= u1)).all()
    block = (x >= 102) & (x < 108) & (y >= u0) & (y <= u1)
    assert set(a[block]) == {"front_central"}
    assert set(b[block & (x < 105)]) == {"edge"} and set(b[block & (x >= 105)]) == {"spot_near", "spot_central", "spot_far"}


def test_the_cut_is_the_sheets_and_g2_is_the_switchs_own():
    """The sheet cuts G2 at 15 yd off the goal line, and since G2 was adopted the driver reads the
    switch's labeller and zones rather than copies of them."""
    import zones
    from constants import G2_EDGE_CUT
    assert G2_EDGE_CUT == 15.0 and G.zones_g2 is zones.zones_g2 and G.G2_ZONES == zones.ZONES["G2"]
    x = np.array([104.99, 105.0])
    assert list(G.zones_g2(x, np.full(2, 40.0))) == ["edge", "spot_central"]


def _bodies() -> pd.DataFrame:
    return pd.DataFrame({"set_piece_id": ["a", "a", "a", "b"], "side": ["att", "att", "def", "att"],
                         "x": [116.5, 103.0, 109.0, 80.0], "y": [33.0, 40.0, 41.0, 40.0]})


def test_every_body_lands_once_under_each_arm_and_an_empty_delivery_counts_zero():
    parts = _parts()
    out = G.blocks(_bodies(), pd.Series(["a", "b", "c"]), parts)
    assert G.G_ARMS == ("C", "G1", "G2")
    for arm, cols in G.columns(parts).items():
        for side, n in (("att", [2, 1, 0]), ("def", [1, 0, 0])):
            assert list(out[[c for c in cols if c.startswith(f"{arm}_{side}_")]].sum(axis=1)) == n
    assert out.loc[0, "G1_att_front_central"] == 1 and out.loc[0, "G2_att_edge"] == 1
    assert out.loc[0, "G1_def_spot_central"] == 1 and out.loc[0, "G2_def_spot_central"] == 1


def test_a_zone_a_partition_does_not_declare_stops_the_blocks():
    parts = {"G1": (lambda x, y: np.full(len(x), "nowhere", dtype=object), ("six_near",))}
    with pytest.raises(ValueError, match="does not declare"):
        G.blocks(_bodies(), pd.Series(["a"]), parts)


def test_the_tables_own_counts_of_c_leave_before_the_recount_joins():
    t = pd.DataFrame({"set_piece_id": ["a", "b"], "C_att_six_near": [99.0, 99.0]})
    out = G.join(t, _bodies(), _parts())
    assert out["C_att_six_near"].tolist() == [1.0, 0.0] and not out.columns.duplicated().any()


def test_the_sanity_count_stops_on_a_lost_body():
    parts = _parts()
    t = G.join(pd.DataFrame({"set_piece_id": ["a", "b"], "won": [True, False]}), _bodies(), parts)
    assert G.sanity(t, _bodies(), "corner", G.columns(parts))
    t.loc[0, "G2_att_six_near"] = 0.0
    with pytest.raises(SystemExit, match="loses or invents"):
        G.sanity(t, _bodies(), "corner", G.columns(parts))


def _synthetic(n_matches: int = 24, per: int = 12, seed: int = 0) -> pd.DataFrame:
    """run_zone_comparison's synthetic population, with the three arms' blocks."""
    rng = np.random.default_rng(seed)
    n = n_matches * per
    t = pd.DataFrame({"match_id": np.repeat(np.arange(n_matches), per),
                      "attacking_team": rng.choice(["a", "b", "c", "d"], n),
                      "zone": rng.choice(["six_near", "six_far", "spot_central", "other"], n),
                      "won": rng.random(n) < 0.6, "kick_x": 120.0, "kick_y": rng.uniform(0, 1, n),
                      "swing": rng.choice(["Inswinging", "Outswinging"], n),
                      "corner_side": rng.choice(["left", "right"], n),
                      "taker_foot": "Right Foot", "delivery_height": "High Pass"})
    t["zone_base"] = t["zone"]
    t["league"] = rng.choice(["x", "y"], n)
    cols = sum(G.columns(_parts()).values(), [])
    return pd.concat([t, pd.DataFrame(rng.poisson(1.0, (n, len(cols))).astype(float), columns=cols)],
                     axis=1)


def test_the_contrasts_are_each_rival_against_c_and_each_block_against_the_design():
    t, cols = _synthetic(), G.columns(_parts())
    d = G.diffs(G.plan_losses(t, cols), "design")
    stats = G.plan_stats(t, cols)
    assert tuple(d) == G.G_KEYS and np.allclose(stats, [v.mean() for v in d.values()])
    assert np.isclose(stats[0], stats[3] - stats[2]) and np.isclose(stats[1], stats[4] - stats[2])
    assert G.G_OUTCOME_KEYS == ("G1-C", "G2-C", "C-delivery", "G1-delivery", "G2-delivery")


def test_the_refit_draws_do_not_depend_on_the_worker_count():
    t = _synthetic()
    stat = partial(G.plan_stats, cols=G.columns(_parts()))
    p1, d1 = G.refit_draws(t, stat, 2, 0, 1)
    p2, d2 = G.refit_draws(t, stat, 2, 0, 2)
    assert np.array_equal(p1, p2) and np.array_equal(d1, d2)


_HELD = {r: {"league_in_design": True, "grid_fine": True, "grid_coarse": True} for r in G.G_RIVALS}
_CLEAN = {"C": True, "G1": True, "G2": True}
_REF = [{"key": "G1-C", "bonf_hi": -0.001, "point": -0.010}, {"key": "G2-C", "bonf_hi": -0.002, "point": -0.020}]


def test_the_rival_that_keeps_c_is_the_lowest_one_that_passes_every_condition():
    steady = {"G1": True, "G2": True}
    assert G.decide(_REF, steady, _CLEAN, _HELD)["replaces"] == "G2"
    assert G.decide(_REF, {"G1": True, "G2": False}, _CLEAN, _HELD)["replaces"] == "G1"
    assert G.decide(_REF, steady, {**_CLEAN, "G2": False}, _HELD)["replaces"] == "G1"
    above = [{"key": "G1-C", "bonf_hi": 0.001, "point": -0.01}, {"key": "G2-C", "bonf_hi": 0.002, "point": -0.02}]
    assert G.decide(above, steady, _CLEAN, _HELD)["replaces"] == "C"


def test_a_failed_reading_goes_to_jaime_and_a_run_with_no_seed_is_a_replication():
    steady = {"G1": True, "G2": True}
    failed = G.decide(_REF, steady, _CLEAN, {**_HELD, "G2": {**_HELD["G2"], "grid_fine": False}})
    assert failed["replaces"] == "G1" and failed["discuss"] == ["G2"]
    replication = G.decide(_REF, {}, _CLEAN, _HELD)
    assert replication["m1"] == ["G1", "G2"] and not replication["seeds_run"]


def _row(key: str, lo: float, hi: float, point: float) -> dict:
    return {"key": key, "bonf_lo": lo, "bonf_hi": hi, "point": point, "mde80": 0.02}


def test_the_screen_goes_on_a_bound_below_zero_with_a_clean_control_and_flags_a_borderline_one():
    ref = [_row("G1-C", -0.03, -0.002, -0.016), _row("G2-C", -0.0245, 0.0005, -0.012)]
    v = G.screen_verdict(ref, _CLEAN)
    assert v["go"] == ["G1"] and not v["G1"]["borderline"]
    assert not v["G2"]["go"] and v["G2"]["borderline"] and not v["G2"]["below_zero"]
    assert G.screen_verdict(ref, {**_CLEAN, "G1": False})["go"] == []


def test_the_readings_and_the_precision_are_m1s():
    assert G.ROBUSTNESS is Z.ROBUSTNESS
    assert (G._BOOT, G._REFIT_BOOT, G._SEEDS, G._STABILITY_BOOT) == (
        Z._BOOT, Z._REFIT_BOOT, Z._SEEDS, Z._STABILITY_BOOT)
    assert np.isclose(G.BONF_ALPHA, 0.01)


def test_the_screen_runs_end_to_end_on_a_small_population(monkeypatch, tmp_path):
    monkeypatch.setattr(G, "outcome_step", lambda *a, **k: [])
    args = SimpleNamespace(boot=20, draws=2, jobs=1, resume=False, refit_boot=2)
    rows = G.screen(_synthetic(), G.columns(_parts()), "corner", args, tmp_path, "")
    keys = {(r["block"], r["key"]) for r in rows}
    assert ("verdict", "screen") in keys and ("refit", "G1-C") in keys and ("control", "G2shuffled-design") in keys
    assert (tmp_path / "zone_g_draws_corner_seed0.npy").exists()


def _refuses(monkeypatch, argv: list, match: str) -> None:
    def work(*a, **k):
        raise AssertionError("the run read the tables before refusing")

    monkeypatch.setattr(G, "inputs", work)
    monkeypatch.setattr(sys, "argv", ["run_zone_g.py", *argv])
    with pytest.raises(SystemExit, match=match):
        G.main()


@pytest.mark.parametrize("extra", [["--draws", "200"], ["--seeds", "1"], ["--boot", "100"],
                                   ["--refit-boot", "10"]])
def test_another_precision_may_not_write_the_record(monkeypatch, extra):
    _refuses(monkeypatch, extra, "may not write")


def test_other_tables_may_not_write_the_record(monkeypatch, tmp_path):
    _refuses(monkeypatch, ["--tables", str(tmp_path)], "not the published ones")


def test_the_screen_may_run_at_another_precision(monkeypatch):
    def reached(*a, **k):
        raise RuntimeError("reached the tables")

    monkeypatch.setattr(G, "inputs", reached)
    monkeypatch.setattr(sys, "argv", ["run_zone_g.py", "--screen", "--draws", "300"])
    with pytest.raises(RuntimeError, match="reached the tables"):
        G.main()
