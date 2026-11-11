"""Tests for run_zone_s: both arms count every body once, S reads its frozen components in the box and
C's cells outside it, the contrasts are S against C, the decision needs all four of the sheet's
conditions, the readings and the precision are M1's, and another precision or other tables may not
write the record.
"""
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "src", _REPO / "pipeline"):
    sys.path.insert(0, str(p))

import run_zone_comparison as Z                                           # noqa: E402
import run_zone_s as R                                                    # noqa: E402

_PARAMS = {"k": 2, "names": ["six_near", "spot_central"], "weights": [0.5, 0.5],
           "means": [[116.0, 33.0], [110.0, 40.0]],
           "covariances": [[[4.0, 0.0], [0.0, 4.0]], [[4.0, 0.0], [0.0, 4.0]]]}


def _parts() -> dict:
    return R.partitions_of(_PARAMS)


def test_s_reads_its_components_in_the_box_and_cs_cells_outside_it():
    parts = _parts()
    x, y = np.array([116.5, 109.0, 99.0, 110.0, 80.0]), np.array([33.0, 41.0, 40.0, 70.0, 40.0])
    want = ["six_near", "spot_central", "edge", "wide", "deep"]
    assert list(parts["S"][0](x, y)) == want and list(parts["C"][0](x, y)) == want
    assert parts["S"][1] == ("six_near", "spot_central", "wide", "edge", "deep")


def _bodies() -> pd.DataFrame:
    return pd.DataFrame({"set_piece_id": ["a", "a", "a", "b"], "side": ["att", "att", "def", "att"],
                         "x": [116.5, 99.0, 109.0, 80.0], "y": [33.0, 40.0, 41.0, 40.0]})


def test_every_body_lands_once_under_each_arm_and_an_empty_delivery_counts_zero():
    parts = _parts()
    out = R.blocks(_bodies(), pd.Series(["a", "b", "c"]), parts)
    assert R.COMPARED == ("C", "S")
    for arm, cols in R.columns(parts).items():
        for side, n in (("att", [2, 1, 0]), ("def", [1, 0, 0])):
            assert list(out[[c for c in cols if c.startswith(f"{arm}_{side}_")]].sum(axis=1)) == n
    assert out.loc[0, "S_att_six_near"] == 1 and out.loc[0, "S_def_spot_central"] == 1


def test_a_zone_a_partition_does_not_declare_stops_the_blocks():
    parts = {"S": (lambda x, y: np.full(len(x), "nowhere", dtype=object), ("six_near",))}
    with pytest.raises(ValueError, match="does not declare"):
        R.blocks(_bodies(), pd.Series(["a"]), parts)


def test_the_tables_own_counts_of_c_leave_before_the_recount_joins():
    t = pd.DataFrame({"set_piece_id": ["a", "b"], "C_att_six_near": [99.0, 99.0]})
    out = R.join(t, _bodies(), _parts())
    assert out["C_att_six_near"].tolist() == [1.0, 0.0] and not out.columns.duplicated().any()


def test_the_sanity_count_stops_on_a_lost_body():
    parts = _parts()
    t = R.join(pd.DataFrame({"set_piece_id": ["a", "b"], "won": [True, False]}), _bodies(), parts)
    assert R.sanity(t, _bodies(), "corner", R.columns(parts))
    t.loc[0, "S_att_six_near"] = 0.0
    with pytest.raises(SystemExit, match="loses or invents"):
        R.sanity(t, _bodies(), "corner", R.columns(parts))


def _synthetic(n_matches: int = 24, per: int = 12, seed: int = 0) -> pd.DataFrame:
    """run_zone_comparison's synthetic population, with the two arms' blocks."""
    rng = np.random.default_rng(seed)
    n = n_matches * per
    t = pd.DataFrame({"match_id": np.repeat(np.arange(n_matches), per),
                      "zone": rng.choice(["six_near", "six_far", "spot_central", "other"], n),
                      "won": rng.random(n) < 0.6, "kick_x": 120.0, "kick_y": rng.uniform(0, 1, n),
                      "swing": rng.choice(["Inswinging", "Outswinging"], n),
                      "corner_side": rng.choice(["left", "right"], n),
                      "taker_foot": "Right Foot", "delivery_height": "High Pass"})
    t["zone_base"] = t["zone"]
    t["league"] = rng.choice(["x", "y"], n)
    cols = sum(R.columns(_parts()).values(), [])
    return pd.concat([t, pd.DataFrame(rng.poisson(1.0, (n, len(cols))).astype(float),
                                      columns=cols)], axis=1)


def test_the_contrasts_are_s_against_c_and_each_block_against_the_design():
    t, cols = _synthetic(), R.columns(_parts())
    d = R.diffs(R.plan_losses(t, cols), "design")
    stats = R.plan_stats(t, cols)
    assert tuple(d) == R.KEYS and np.allclose(stats, [v.mean() for v in d.values()])
    assert np.isclose(stats[0], stats[1] - stats[2])


def test_the_refit_draws_do_not_depend_on_the_worker_count():
    t = _synthetic()
    stat = partial(R.plan_stats, cols=R.columns(_parts()))
    p1, d1 = R.refit_draws(t, stat, 2, 0, 1)
    p2, d2 = R.refit_draws(t, stat, 2, 0, 2)
    assert np.array_equal(p1, p2) and np.array_equal(d1, d2)


def test_s_replaces_c_only_when_all_four_hold_and_a_failed_reading_goes_to_jaime():
    held = {"league_in_design": True, "grid_fine": True, "grid_coarse": True}
    clean, below = {"S": True, "C": True}, [{"key": "S-C", "bonf_hi": -0.001}]
    assert R.decide(below, True, clean, held)["replaces"]
    assert not R.decide([{"key": "S-C", "bonf_hi": 0.001}], True, clean, held)["replaces"]
    assert not R.decide(below, False, clean, held)["replaces"]
    assert not R.decide(below, True, {"S": False, "C": True}, held)["replaces"]
    failed = R.decide(below, True, clean, {**held, "grid_fine": False})
    assert not failed["replaces"] and failed["discuss"]
    replication = R.decide(below, None, clean, held)
    assert replication["m1"] and not replication["seeds_run"]


def test_the_readings_and_the_precision_are_m1s():
    assert R.READINGS == Z.ROBUSTNESS
    assert (R._BOOT, R._REFIT_BOOT, R._SEEDS, R._STABILITY_BOOT) == (
        Z._BOOT, Z._REFIT_BOOT, Z._SEEDS, Z._STABILITY_BOOT)
    assert np.isclose(R.BONF_ALPHA, 0.01)


def _refuses(monkeypatch, argv: list, match: str) -> None:
    def work(*a, **k):
        raise AssertionError("the run read the tables before refusing")

    monkeypatch.setattr(R, "inputs", work)
    monkeypatch.setattr(sys, "argv", ["run_zone_s.py", *argv])
    with pytest.raises(SystemExit, match=match):
        R.main()


@pytest.mark.parametrize("extra", [["--draws", "200"], ["--seeds", "1"], ["--boot", "100"],
                                   ["--refit-boot", "10"]])
def test_another_precision_may_not_write_the_record(monkeypatch, extra):
    _refuses(monkeypatch, extra, "may not write")


def test_other_tables_may_not_write_the_record(monkeypatch, tmp_path):
    _refuses(monkeypatch, ["--tables", str(tmp_path)], "not the published ones")
