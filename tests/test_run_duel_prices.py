"""Tests for run_duel_prices: points read in the canonical frame, the value each duel carries, the
price arithmetic, and C1a holding only where the confirmation sample repeats the development's order.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "src", _REPO / "pipeline"):
    sys.path.insert(0, str(p))

import run_duel_prices as D                                                # noqa: E402


def test_a_point_is_read_from_the_kick_s_side():
    """Whatever partition the switch names, a point from the attackers' right flag is labelled as
    its mirror image from the left."""
    right = D.zone_of(np.array([116.0, 110.0]), np.array([46.0, 50.0]), np.array([79.9, 79.9]))
    left = D.zone_of(np.array([116.0, 110.0]), np.array([34.0, 30.0]), np.array([0.1, 0.1]))
    assert list(right) == list(left)


def test_the_first_duel_carries_the_in_window_xg_and_drops_the_unresolved():
    deliv = pd.DataFrame({"set_piece_id": list("abc"), "league": "x", "match_id": [1, 1, 2],
                          "attacking_team": "T", "delivery_y": 79.9,
                          "delivery_end_x": [116.0, np.nan, 110.0], "delivery_end_y": [33.0, 40.0, 40.0],
                          "won_first_contact": [True, True, None]})
    shots = pd.DataFrame({"set_piece_id": ["a", "a", "a"], "segment": ["direct", "late", "after_clearance"],
                          "xg": [0.1, 0.5, 0.2]})
    f = D.first_duels(deliv, shots)
    assert list(f["set_piece_id"]) == ["a"] and f["value"].iloc[0] == pytest.approx(0.3)
    # the end point, mirrored by the kick's own side: (116, 33) from the right flag is (116, 47)
    assert f["zone"].iloc[0] == D.zone_of(np.array([116.0]), np.array([47.0]), np.array([0.1]))[0]
    assert bool(f["won"].iloc[0])


def test_the_second_duel_carries_both_readings_of_its_value():
    """The after-clearance xG s08 carries and the corner's after-first-touch xG the sheet names,
    side by side on the same cleared corners."""
    sit = pd.DataFrame({"set_piece_id": list("abc"), "match_id": [1, 1, 2],
                        "drop_x": [100.0, np.nan, 104.0], "drop_y": [40.0, 40.0, 30.0],
                        "won_second_ball": [True, True, None], "value_xg_after": [0.2, 0.3, 0.1]})
    deliv = pd.DataFrame({"set_piece_id": list("abc"), "league": "x", "attacking_team": "T",
                          "delivery_y": 0.1, "xg_after_first_touch": [0.5, 0.3, 0.1]})
    s = D.second_duels(sit, deliv)
    assert list(s["set_piece_id"]) == ["a"]
    assert s["value"].iloc[0] == pytest.approx(0.2) and s["value_aft"].iloc[0] == pytest.approx(0.5)


def test_the_list_runs_on_every_league_with_c1a_s_split_beside():
    """The list comes from every league, the project's scope; the development and confirmation
    samples C1a reads come beside it, then each league alone."""
    d = pd.DataFrame({"league": [*D.DEVELOPMENT, "serie_a", "la_liga"]})
    s = D.samples(sorted(d["league"].unique()))
    assert list(s)[:3] == ["all", "development", "confirmation"]
    assert len(s["all"](d)) == len(d)
    assert set(s["development"](d)["league"]) == set(D.DEVELOPMENT)
    assert set(s["confirmation"](d)["league"]) == {"serie_a", "la_liga"}


def test_obv_is_read_beside_the_second_ball_price_on_the_same_corners():
    """Reported, never deciding: OBV's price per zone beside ours, the Spearman of the two on draws
    both share, and the share of corners whose contesting touch carries an OBV."""
    zones = ("a", "b", "c", "d", "e")
    d = _duels(list(zones), seed=6)
    d["obv"] = 2.0 * d["value"]
    lost = d.index[~d["won"]][:10]
    d.loc[lost, "obv"] = np.nan
    r, table = D.obv_consistency(d, zones, 200)
    assert r["coverage"] == pytest.approx(1 - 10 / len(d))
    assert r["coverage_won"] == 1.0 and r["coverage_lost"] == pytest.approx(1 - 10 / (~d["won"]).sum())
    assert r["spearman"] == pytest.approx(1.0) and r["spearman_lo"] == pytest.approx(1.0)
    assert set(table["duel"]) == {"second_obv"}


def test_the_next_touch_s_obv_is_signed_for_the_attack(monkeypatch):
    """A touch worth +x to the defence is worth -x to the attack, a touch without OBV stays missing
    rather than zero, and a corner with no contesting touch has no row."""
    corners = pd.DataFrame({"id": list("abcd")})
    nxt = {"a": pd.Series({"team.name": "Att", "obv_total_net": 0.02}),
           "b": pd.Series({"team.name": "Def", "obv_total_net": 0.03}),
           "c": pd.Series({"team.name": "Att", "obv_total_net": None}), "d": None}

    class Stub:
        def __init__(self, *a, **k):
            self.g = type("Grammar", (), {"deliveries": staticmethod(lambda: corners)})()

        def _clear_and_next(self, c):
            return None, nxt[c["id"]], "Att"

    monkeypatch.setattr(D, "DropSituation", Stub)
    monkeypatch.setattr(D.L, "load_events", lambda m: None)
    monkeypatch.setattr(D.L, "load_360", lambda m: None)
    got = dict(D._next_touch_obv(1, "corner", set("abcd")))
    assert got["a"] == pytest.approx(0.02) and got["b"] == pytest.approx(-0.03)
    assert np.isnan(got["c"]) and "d" not in got


def test_the_price_is_won_minus_lost_and_the_what_if_scales_it():
    tot = np.zeros((2, 2, 2))                      # zones x (lost, won) x (n, sum)
    tot[0] = [[10, 1.0], [10, 3.0]]                # zone 0: lost 0.1, won 0.3
    tot[1] = [[30, 0.6], [10, 0.4]]                # zone 1: lost 0.02, won 0.04
    p = D.prices(tot)
    assert p["price"] == pytest.approx([0.2, 0.02])
    assert p["share"] == pytest.approx([1 / 3, 2 / 3])
    assert p["whatif_per100"] == pytest.approx(100 * p["share"] * 0.1 * p["price"])


def _duels(order: list, n_matches: int = 120, seed: int = 0) -> pd.DataFrame:
    """Corners whose price rises with the zone's place in `order`."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_matches):
        for _ in range(12):
            k = rng.integers(0, len(order))
            won = rng.random() < 0.4
            rows.append({"match_id": m, "attacking_team": f"t{m % 20}", "zone": order[k], "won": won,
                         "value": rng.exponential(0.01 + (0.05 * k if won else 0.0))})
    return pd.DataFrame(rows)


def test_c1a_holds_when_the_confirmation_repeats_the_order_and_not_when_it_reverses():
    zones = ("a", "b", "c", "d", "e")
    dev = _duels(list(zones), seed=1)
    assert D.c1a(dev, _duels(list(zones), seed=2), zones, 200, 0)["holds"]
    flipped = D.c1a(dev, _duels(list(zones)[::-1], seed=3), zones, 200, 0)
    assert not flipped["holds"] and flipped["spearman"] < 0


def test_c1a_is_undecided_rather_than_false_when_too_few_zones_are_left():
    """A Spearman over fewer than three zones says nothing, so the verdict is None, as the repo's
    degenerate cells are, and never a False that reads as a failed check; the decision follows."""
    zones = ("a", "b", "rare1", "rare2")
    r = D.c1a(_duels(["a", "b"], seed=7), _duels(["a", "b"], seed=8), zones, 50, 0)
    assert r["holds"] is None and np.isnan(r["spearman"])
    seeds = pd.DataFrame([{"holds": None}, {"holds": True}])
    assert D.c1a_decision(seeds) == {"C1a": None, "C1a_any_seed": True}
    assert D.c1a_decision(pd.DataFrame([{"holds": True}] * 2))["C1a"] is True


def test_the_readings_are_undecided_rather_than_raising_when_every_zone_is_thin():
    """A population whose second duels leave every zone thin reports no top zone and no Spearman,
    as C1a does under three zones, instead of failing on an empty argmax."""
    rows = [{"sample": "all", "duel": d, "zone": z, "thin": True, "price": 0.1}
            for d in ("second", "second_aft") for z in ("a", "b")]
    r = D.readings_agree(pd.DataFrame(rows))
    assert r["second_top_after_clearance"] is None and r["second_top_after_first_touch"] is None
    assert np.isnan(r["second_spearman"])


def test_a_thin_zone_is_named_and_left_out_of_the_correlation():
    zones = ("a", "b", "c", "d", "e", "rare")
    dev, conf = _duels(list(zones[:5]), seed=4), _duels(list(zones[:5]), seed=5)
    r = D.c1a(dev, conf, zones, 100, 0)
    assert "rare" in r["zones_thin"] and "rare" not in r["zones_used"]


def test_each_population_is_read_off_the_combined_grain(tmp_path):
    """The recipe writes one delivery grain for the paper's two populations (eda_build); each
    population takes its own rows of it, beside its own shots and situation table."""
    pd.DataFrame({"set_piece_id": list("abc"),
                  "set_piece_type": ["corner", "free_kick_flank", "corner"]}
                 ).to_parquet(tmp_path / "eda_deliveries.parquet")
    for kind, names in D._INPUTS.items():
        for name in names:
            pd.DataFrame({"kind": [kind]}).to_parquet(tmp_path / name)
    deliv, shots, sit = D.inputs(tmp_path, "free_kick_flank")
    assert list(deliv["set_piece_id"]) == ["b"]
    assert shots["kind"].tolist() == sit["kind"].tolist() == ["free_kick_flank"]


def test_a_limited_run_reads_the_tables_the_smoke_built(tmp_path):
    """--limit N reads the N-match grain, shots and situation a smoke run wrote, never the grant's."""
    pd.DataFrame({"set_piece_id": ["a"], "set_piece_type": ["corner"]}).to_parquet(
        tmp_path / "eda_deliveries_limit3.parquet")
    for name in D._INPUTS["corner"]:
        pd.DataFrame({"kind": ["corner"]}).to_parquet(
            tmp_path / name.replace(".parquet", "_limit3.parquet"))
    deliv, shots, sit = D.inputs(tmp_path, "corner", "_limit3")
    assert list(deliv["set_piece_id"]) == ["a"] and len(shots) == len(sit) == 1


# Both folders the published names are written into: data/derived, and zones_F for partition F.
_PUBLISHED = [[], ["--out-dir", str(_REPO / "data" / "derived" / "zones_F")]]


@pytest.mark.parametrize("out", _PUBLISHED)
def test_another_precision_may_not_write_the_records(monkeypatch, out):
    def work(*a, **k):
        raise AssertionError("the run read the data before refusing")

    monkeypatch.setattr(D.pd, "read_parquet", work)
    monkeypatch.setattr(sys, "argv", ["run_duel_prices.py", "--boot", "10", *out])
    with pytest.raises(SystemExit, match="published duel_prices"):
        D.main()


@pytest.mark.parametrize("out", _PUBLISHED)
def test_other_tables_may_not_write_the_records(monkeypatch, tmp_path, out):
    """Tables outside data/derived are not the published ones, so a run on them at the registered
    precision would still put its prices under the names the record is read from."""
    def work(*a, **k):
        raise AssertionError("the run read the data before refusing")

    monkeypatch.setattr(D.pd, "read_parquet", work)
    monkeypatch.setattr(sys, "argv", ["run_duel_prices.py", "--tables", str(tmp_path), *out])
    with pytest.raises(SystemExit, match="not the published ones"):
        D.main()


@pytest.mark.parametrize("out", _PUBLISHED)
def test_a_limited_run_may_not_write_the_records(monkeypatch, out):
    def work(*a, **k):
        raise AssertionError("the run read the data before refusing")

    monkeypatch.setattr(D.pd, "read_parquet", work)
    monkeypatch.setattr(sys, "argv", ["run_duel_prices.py", "--limit", "40", *out])
    with pytest.raises(SystemExit, match="N-match tables"):
        D.main()
