"""Tests for s02b_aim: the grid, the kick block, the aim model and the two decision statistics.

The statistics are pinned on synthetic corners whose answer is known by construction: an aim the
kick block states, and a defence that intercepts at the near post. D1 has to see the first, D2 the
second, and neither may see what is not there.
"""
import numpy as np
import pandas as pd
import pytest

import s02b_aim as A
from constants import ZONE_PARTITION
from zones import zone_names


def test_the_end_zones_fall_where_the_laws_put_the_lines():
    x = np.array([116.0, 110.0, 104.0, 104.0, 99.0, 90.0, 116.0])
    y = np.array([35.0, 40.0, 50.0, 10.0, 40.0, 40.0, 44.0])
    assert list(A.end_zones("base", x, y)) == ["six_near", "spot_central", "other", "other",
                                               "other", "other", "six_far"]
    assert list(A.end_zones("coarse", x, y)) == ["near", "central", "far", "other", "other",
                                                 "other", "far"]
    assert list(A.end_zones("fine", np.array([116.0, 104.0]), np.array([25.0, 55.0]))) == \
        ["six_near_out", "box_far_out"]


def test_every_zone_a_grid_can_emit_is_one_it_declares():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(80, 120, 5000), rng.uniform(0, 80, 5000)
    for grid, zones in A.GRIDS.items():
        assert set(A.end_zones(grid, x, y)) == set(zones), grid
        assert list(zones) == sorted(zones), "class order is the sorted zone list"


def test_no_input_reads_a_barred_field(monkeypatch):
    A.assert_sources()
    monkeypatch.setitem(A._COLUMN_SOURCE, A.KICK_COLS[0], "pass.end_location")
    with pytest.raises(ValueError, match="barred"):
        A.assert_sources()


def test_the_end_location_is_the_target_and_never_an_input():
    assert not any("end_location" in s for s in A._COLUMN_SOURCE.values())
    assert set(A._COLUMN_SOURCE) == (set(A.KICK_COLS) | set(A.DESIGN_COLS) | {"league"}
                                     | set(A.KICK_SPOT))


def test_a_column_added_to_both_arms_must_declare_its_source():
    t = _synthetic(n_matches=6, per=5)
    with pytest.raises(ValueError, match="declared source"):
        A.design(t, extra=("attacking_team",))
    t["league"] = "x"
    Xa, Xk = A.design(t, extra=("league",))
    assert Xa.shape[1] == A.design(t)[0].shape[1] + 1
    assert Xk.shape[1] == Xa.shape[1] + len(A.KICK_COLS)


def test_the_kick_spot_enters_both_arms_as_numbers():
    """A flank free kick is struck from where the foul was, so its design carries the canonical kick
    spot as two numbers in both arms, and the kick block is read net of it; a corner passes none and
    keeps the design A1 and A2 were registered on, bit for bit."""
    t = _synthetic(n_matches=20, per=10)
    t["kick_x"], t["kick_y"] = np.linspace(85.0, 119.0, len(t)), 10.0
    a0, k0 = A.design(t)
    a1, k1 = A.design(t, numeric=A.KICK_SPOT)
    assert np.array_equal(a1[:, :-2], a0) and np.array_equal(a1[:, -2:], t[list(A.KICK_SPOT)])
    assert np.array_equal(k1, np.c_[a1, t[A.KICK_COLS].to_numpy(float)])
    assert np.array_equal(A.design(t, numeric=())[1], k0)
    with pytest.raises(ValueError, match="declared source"):
        A.design(t, numeric=("delivery_end_x",))
    assert np.isfinite(A.statistics(t, numeric=A.KICK_SPOT)[0])


def _corner(y_flag: float, players: list) -> tuple:
    corner = pd.Series({"id": "c1", "location": [120.0, y_flag]})
    fz = pd.DataFrame({"event_uuid": ["c1"], "freeze_frame": [[
        {"location": list(xy), "teammate": tm, "keeper": gk, "actor": ac}
        for xy, tm, gk, ac in players]]})
    return corner, fz


# Where the kick block's three counted bodies stand under each partition.
_KICK_ZONES = {"C": ({"six_near": 1, "spot_far": 1}, {"six_central": 1}),
               "F": ({"six_yard": 1, "far_post": 1}, {"six_yard": 1}),
               "G2": ({"six_near": 1, "spot_far": 1}, {"six_central": 1})}


def test_the_kick_block_counts_jobs_and_outfield_defenders_only():
    """Taker and keepers out under the switch's partition; a far-flag corner lands in the same
    zones as its mirror image."""
    players = [((119.5, 0.5), True, False, True),      # taker
               ((116.0, 34.0), True, False, False),    # attacker, six-yard box short of the post
               ((110.0, 46.0), True, False, False),    # attacker, spot band past the far post
               ((116.0, 38.0), False, False, False),   # defender, six-yard box between the posts
               ((119.0, 40.0), False, True, False)]    # keeper
    near = A.kick_block(*_corner(0.5, players))
    mirrored = [((x, 80.0 - y), tm, gk, ac) for (x, y), tm, gk, ac in players]
    far = A.kick_block(*_corner(79.5, mirrored))
    for side, held in zip(("att", "def"), _KICK_ZONES[ZONE_PARTITION], strict=True):
        assert {z: near[f"{side}_{z}"] for z in zone_names() if near[f"{side}_{z}"]} == held
    assert near["att_box_n"] == 2 and near["def_box_n"] == 1
    assert not near["kick_side_right"] and far["kick_side_right"]
    assert {k: v for k, v in far.items() if k != "kick_side_right"} == \
           {k: v for k, v in near.items() if k != "kick_side_right"}


def _synthetic(n_matches: int = 60, per: int = 20, seed: int = 0, readable: bool = True,
               intercept: bool = True) -> pd.DataFrame:
    """Corners aimed near or far with equal value. The kick block states the aim when `readable`;
    when `intercept`, the defence wins most near-post-aimed balls short of the post and those carry
    little value, which is the bias D2 is built to catch."""
    rng = np.random.default_rng(seed)
    n = n_matches * per
    aim = rng.choice(["six_near", "six_far"], n)
    won = rng.random(n) < 0.4
    end = np.where(won, aim, np.where(intercept & (rng.random(n) < 0.7), "six_near", aim))
    xg = rng.exponential(0.05, n) * np.where(won, 1.5, 0.5)
    t = pd.DataFrame({"match_id": np.repeat(np.arange(n_matches), per), "zone": end, "won": won,
                      "xg": xg, "swing": rng.choice(["Inswinging", "Outswinging"], n),
                      "corner_side": rng.choice(["left", "right"], n), "taker_foot": "Right Foot",
                      "delivery_height": "High Pass"})
    signal = (aim == "six_near").astype(float) if readable else rng.random(n).round()
    for c in A.KICK_COLS:
        t[c] = rng.poisson(1.0, n).astype(float)
    t[A.KICK_COLS[0]] += 3 * signal
    t[A.KICK_COLS[1]] += 3 * (1 - signal)
    return t


def test_the_aim_model_scores_every_row_out_of_fold():
    t = _synthetic(n_matches=20, per=10)
    zones = A.GRIDS["base"]
    z = t["zone"].map({k: i for i, k in enumerate(zones)}).to_numpy()
    _, K = A.design(t)
    p = A.aim_oof(K, z, t["won"].to_numpy(bool), t["match_id"].to_numpy(), len(zones))
    assert p.shape == (len(t), len(zones)) and np.allclose(p.sum(1), 1.0)


def test_d1_sees_an_aim_the_kick_states_and_not_one_it_does_not():
    d1_read = A.statistics(_synthetic(readable=True))[0]
    d1_blind = A.statistics(_synthetic(readable=False))[0]
    assert d1_read < -0.1
    assert d1_blind > -0.02


def test_d2_sees_the_interception_bias_and_only_the_interception_bias():
    """With interceptions at the near post the end-zone map discounts it and the aim map does not;
    without them the two maps agree and D2 sits near zero."""
    d1, d2, naive, honest, _ = A.statistics(_synthetic(intercept=True))
    assert naive < 0 and d2 < 0 and abs(honest) < abs(naive)
    assert abs(A.statistics(_synthetic(intercept=False))[1]) < abs(d2)


def test_the_soft_map_recovers_zone_values_from_calibrated_probabilities():
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(3), 4000)
    mu = np.array([0.02, 0.05, 0.08])
    k = np.array([rng.choice(3, p=r) for r in p])
    v = mu[k] + rng.normal(0, 0.005, len(k))
    assert A.soft_contrast(v, p, 0, 2) == pytest.approx(-0.06, abs=0.01)
