"""Tests for run_aim_zone: the population it builds, the wording it may use, and the verdict rules.

The verdicts are decided by code, so the rules are pinned here rather than trusted to a log: a
decision needs its pivotal interval below zero AND a steady bound across seeds, and A1 also needs a
clean shuffled control.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "src", _REPO / "pipeline"):
    sys.path.insert(0, str(p))

import run_aim_zone as R                                              # noqa: E402
from s02b_aim import KICK_COLS, KICK_SPOT                             # noqa: E402


def _refit(d1_below: bool, d2_below: bool, bonf: float = 0.01) -> list:
    return [{"key": "D1", "below_zero": d1_below, "bonf_hi": -bonf if d1_below else bonf},
            {"key": "D2", "below_zero": d2_below, "bonf_hi": bonf}]


def test_a1_needs_the_interval_a_steady_bound_and_a_clean_control():
    assert R.decide(_refit(True, True), {}, clean=True)["A1"]["state"] == "true"
    assert R.decide(_refit(True, True), {}, clean=False)["A1"]["state"] == "false"
    flipped = R.decide(_refit(True, True), {"D1": False}, clean=True)
    assert flipped["A1"]["state"].startswith("unstable")
    assert R.decide(_refit(False, True), {}, clean=True)["A1"]["state"] == "false"


def test_a2_does_not_hang_on_the_d1_control():
    v = R.decide(_refit(False, True), {}, clean=False)
    assert v["A2"]["state"] == "true" and v["A1"]["state"] == "false"
    flipped = R.decide(_refit(True, True), {"D2": False}, clean=True)
    assert flipped["A2"]["state"].startswith("unstable")


def test_the_bonferroni_reading_travels_beside_and_never_decides():
    v = R.decide(_refit(True, True, bonf=0.01), {}, clean=True)
    assert v["A2"]["state"] == "true" and not v["A2"]["bonferroni_below_zero"]


@pytest.mark.parametrize("lo, hi, word", [(-0.005, 0.008, "flat"),
                                          (-0.02, 0.005, "not distinguishable from zero"),
                                          (0.002, 0.03, "differs from zero"),
                                          (np.nan, 0.1, "no interval")])
def test_flat_is_written_only_inside_the_pre_registered_margin(lo, hi, word):
    assert R.EQUIVALENCE_XG == 0.01
    assert R.equivalence(lo, hi) == word


def _tables() -> tuple:
    corners = pd.DataFrame({"set_piece_id": list("abcde"), "match_id": [1, 1, 2, 2, 3],
                            "set_piece_type": "corner",
                            "short_corner": [False, False, True, False, False],
                            "delivery_end_x": [116.0, 110.0, 105.0, np.nan, 117.0],
                            "delivery_end_y": [45.0, 38.0, 20.0, 40.0, 35.0],
                            "delivery_x": 119.9, "delivery_y": [79.9, 0.1, 0.1, 0.1, 0.1],
                            "corner_side": ["right", "left", "left", "left", "left"]})
    dlv = pd.DataFrame({"set_piece_id": list("abcde"), "set_piece_type": "corner",
                        "won_first_contact": [True, None, False, True, False]})
    shots = pd.DataFrame({"set_piece_id": ["a", "a", "b", "e"],
                          "segment": ["direct", "late", "flick_on", "late"],
                          "after_first_touch": [False, True, True, True],
                          "xg": [0.1, 0.4, 0.05, 0.3]})
    block = pd.DataFrame({"set_piece_id": list("abcde"),
                          "kick_side_right": [True, False, False, False, False],
                          **{c: 0.0 for c in KICK_COLS}})
    return corners, dlv, shots, block


def test_the_population_is_long_corners_with_an_end_and_their_in_window_value():
    corners, dlv, shots, block = _tables()
    t = R.population(corners, dlv, shots, block).set_index("set_piece_id")
    assert list(t.index) == ["a", "b", "e"]
    assert t.loc["a", "zone"] == "six_near"      # (116, 45) from the far flag mirrors to y = 35
    assert t.loc["b", "zone"] == "spot_central"
    assert t.loc["a", "xg"] == pytest.approx(0.1) and t.loc["e", "xg"] == 0.0
    assert t.loc["b", "xg_aft"] == pytest.approx(0.05) and t.loc["a", "xg_aft"] == 0.0
    assert list(t["won"]) == [True, False, False]
    assert list(t["any_shot"]) == [1.0, 1.0, 0.0]


def test_an_own_goal_is_a_goal_and_no_shot_of_any_shot():
    """A delivery whose only in-window event is an own goal has no shot, and its zero xG moves nothing."""
    from s01_phases import OWN_GOAL
    corners, dlv, shots, block = _tables()
    shots = shots.assign(shot_outcome=["Saved", "Saved", OWN_GOAL, "Saved"], xg=[0.1, 0.4, 0.0, 0.3])
    t = R.population(corners, dlv, shots, block).set_index("set_piece_id")
    assert list(t["any_shot"]) == [1.0, 0.0, 0.0] and t.loc["b", "xg"] == 0.0


def test_the_flank_population_carries_the_canonical_kick_spot():
    """The flank twin reads its own rows of the combined grain, and each kick carries the spot it
    was struck from in the canonical frame, mirrored by the kick's side as the end point is."""
    corners, dlv, shots, block = _tables()
    fk = corners.assign(set_piece_type="free_kick_flank", short_corner=False,
                        delivery_x=[100.0, 90.0, 95.0, 96.0, 88.0],
                        delivery_y=[70.0, 5.0, 8.0, 9.0, 12.0])
    grain = pd.concat([dlv.assign(set_piece_type="free_kick_flank"),
                       dlv.assign(won_first_contact=None)])          # the corner rows beside
    t = R.population(fk, grain, shots, block, "free_kick_flank").set_index("set_piece_id")
    assert list(t.index) == ["a", "b", "c", "e"]
    assert list(t["won"]) == [True, False, False, False]
    assert t.loc["a", list(KICK_SPOT)].tolist() == [100.0, pytest.approx(10.0)]
    assert t.loc["b", list(KICK_SPOT)].tolist() == [90.0, pytest.approx(5.0)]


def test_only_the_flank_design_carries_the_kick_spot():
    """A corner is struck from the flag, so its design is the one A1 and A2 were registered on."""
    assert R.SPOT == {"corner": (), "free_kick_flank": KICK_SPOT}


def test_the_population_stops_when_the_frame_and_s01_disagree_on_the_side():
    corners, dlv, shots, block = _tables()
    block.loc[0, "kick_side_right"] = False
    with pytest.raises(SystemExit, match="corner side"):
        R.population(corners, dlv, shots, block)


def test_a_delivery_no_one_touched_sits_on_neither_side_of_the_first_contact():
    """'defence_first' means the defence made the first contact: a delivery no one touched has no
    contact to read, so it is counted apart instead of joining the defence's side."""
    from s02b_aim import GRIDS
    zones, n = GRIDS["base"], 36
    first = pd.Series([True, False, None] * (n // 3), dtype=object)
    t = pd.DataFrame({"match_id": np.repeat(np.arange(6), n // 6), "xg": np.linspace(0.0, 0.3, n),
                      "zone": np.resize(["six_near", "six_far"], n),
                      "won_first_contact": first, "won": first.eq(True)})
    pk = np.full((n, len(zones)), 1.0 / len(zones))
    rows = pd.DataFrame(R.by_first_contact(t, "confirmation", 20, (None, pk)))
    per_side = rows[rows["key"] == "naive"].set_index("grid")["n"]
    assert per_side["attack_won"] == per_side["defence_first"] == n // 3
    assert rows.loc[rows["grid"] == "no_contact", "n"].tolist() == [n // 3]


def test_the_draws_resume_by_run_first_contact_s_one_rule():
    """The two pre-registered drivers save and resume their refit draws through one function, so the
    rule that a saved file is reused only at the count asked for, pinned in test_run_first_contact,
    cannot drift between them."""
    import run_first_contact
    assert R.seed_draws is run_first_contact.seed_draws


@pytest.mark.parametrize("kind", [[], ["--kind", "free_kick_flank"]])
@pytest.mark.parametrize("dest", [[], ["--out-dir", str(_REPO / "data" / "derived" / "zones_F")]])
def test_the_published_record_is_written_only_at_the_registered_precision(monkeypatch, dest, kind):
    """A1 and A2 are registered at 300 refit draws and five stability seeds of 2000, so a run at any
    other precision may not write either population's record, in data/derived or in zones_F where
    the pass under partition F publishes it, and refuses before reading anything."""
    def work(*a, **k):
        raise AssertionError("the run read the data before refusing")

    monkeypatch.setattr(R.pd, "read_parquet", work)
    monkeypatch.setattr(R, "build", work)
    monkeypatch.setattr(sys, "argv", ["run_aim_zone.py", "--refit-boot", "3", *dest, *kind])
    with pytest.raises(SystemExit, match="may not write the published"):
        R.main()
