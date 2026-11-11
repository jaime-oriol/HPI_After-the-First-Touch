"""Tests for run_tracking_curve: the fit reads a 1/√N curve as one, TRK1 refuses a curve that is
not, the projection inverts the curve, the draws are honest subsets and both statistics are the ones
their drivers publish."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "src", _REPO / "pipeline"):
    sys.path.insert(0, str(p))

import run_tracking_curve as T                                             # noqa: E402
from s04b_snapshot import score_arm                                       # noqa: E402


def _curve(slope: float, noise: float = 0.03, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [{"n_matches": n, "width": 0.4 * n ** slope * np.exp(rng.normal(0, noise)),
             "point": 0.1} for n in (10, 15, 20, 25) for _ in range(60)]
    return pd.DataFrame(rows + [{"n_matches": 30, "width": 0.4 * 30 ** slope, "point": 0.1}])


def test_the_fit_reads_a_one_over_root_n_curve_as_one():
    a, b = T.slope(_curve(-0.5))
    assert b == pytest.approx(-0.5, abs=0.03) and np.exp(a) == pytest.approx(0.4, rel=0.1)


def test_trk1_holds_on_root_n_and_refuses_a_flatter_curve():
    assert T.fit_curve(_curve(-0.5), 30, 0.1, 300, 0)["TRK1"]
    assert not T.fit_curve(_curve(-0.2), 30, 0.1, 300, 0)["TRK1"]


def test_the_projection_inverts_the_curve():
    f = T.fit_curve(_curve(-0.5, noise=1e-6), 30, 0.02, 50, 0)
    w30 = 0.4 * 30 ** -0.5
    assert f["n_star_sqrt"] == pytest.approx(30 * (w30 / 0.04) ** 2)
    assert f["n_star_fit"] == pytest.approx(f["n_star_sqrt"], rel=1e-3)


def test_the_draws_are_subsets_without_replacement_and_the_full_size_is_drawn_once():
    mids = np.arange(100, 130)
    d = T.draw_matches(mids, 10, 5, 0)
    assert len(d) == 5 and all(len(set(k)) == 10 and set(k) <= set(mids) for k in d)
    assert [list(k) for k in d] == [list(k) for k in T.draw_matches(mids, 10, 5, 0)]
    assert [list(k) for k in T.draw_matches(mids, 30, 5, 0)] == [list(mids)]


def _frame(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 240
    won = rng.random(n) < 0.6
    return pd.DataFrame({"corner_id": np.arange(n), "match_id": np.repeat(np.arange(24), 10),
                         "won": won, "p_att": np.clip(0.5 + 0.2 * (won - 0.5) + rng.normal(0, .15, n), .01, .99),
                         "p_snapshot": rng.uniform(0.2, 0.8, n),
                         "p_race": np.clip(0.5 + 0.3 * (won - 0.5) + rng.normal(0, .1, n), .01, .99)})


def test_s2_is_the_gap_in_the_race_probability_between_won_and_lost():
    f = _frame()
    point, lo, hi = T.s2_stat(f, 200, 0)
    assert point == pytest.approx(f.loc[f["won"], "p_att"].mean() - f.loc[~f["won"], "p_att"].mean())
    assert lo < point < hi


def test_s1_is_the_recalibrated_snapshot_minus_the_race():
    f = _frame()
    y, g = f["won"].to_numpy().astype(int), f["match_id"].to_numpy()
    want = (score_arm(f["p_snapshot"].to_numpy(), y, g)["row_recal"]
            - score_arm(f["p_race"].to_numpy(), y, g)["row_recal"]).mean()
    assert T.s1_stat(f, 200, 0)[0] == pytest.approx(want)


def test_a_draw_does_not_depend_on_the_worker_count():
    from joblib import Parallel, delayed
    f = _frame()
    keep = T.draw_matches(np.arange(24), 12, 3, 0)
    args = [("S1", f, k, 12, i, 100, 0) for i, k in enumerate(keep)]
    assert [T._task(*a) for a in args] == Parallel(n_jobs=3)(delayed(T._task)(*a) for a in args)


def test_a_population_under_the_floor_is_recorded_as_not_estimable(monkeypatch, tmp_path):
    """The flank free kicks in the tracked matches are too few for a curve: a statistic under the
    publication floor is recorded with its count and no curve, and the run ends cleanly."""
    small = _frame().iloc[:15]
    seen = []
    monkeypatch.setattr(T, "s1_frame", lambda der, jobs, kind: seen.append(kind) or small)
    monkeypatch.setattr(T, "s2_frame", lambda der, kind: seen.append(kind) or small)
    monkeypatch.setattr(sys, "argv", ["run_tracking_curve.py", "--kind", "free_kick_flank",
                                      "--out-dir", str(tmp_path)])
    T.main()
    assert seen == ["free_kick_flank"] * 2
    fit = pd.read_parquet(tmp_path / "tracking_curve_fit_free_kick_flank.parquet").set_index("stat")
    assert not fit["estimable"].any() and fit.loc["S1", "n_rows"] == 15
    assert pd.read_parquet(tmp_path / "tracking_curve_draws_free_kick_flank.parquet").empty


def test_another_precision_may_not_write_the_records(monkeypatch):
    def work(*a, **k):
        raise AssertionError("the run read the data before refusing")

    monkeypatch.setattr(T, "s1_frame", work)
    monkeypatch.setattr(sys, "argv", ["run_tracking_curve.py", "--draws", "3"])
    with pytest.raises(SystemExit, match="published tracking_curve"):
        T.main()


def test_other_tables_may_not_write_the_records(monkeypatch, tmp_path):
    """Tables outside data/derived are not the published ones, so a run on them at the registered
    precision would still put its numbers under the names the record is read from."""
    def work(*a, **k):
        raise AssertionError("the run read the data before refusing")

    monkeypatch.setattr(T, "s1_frame", work)
    monkeypatch.setattr(sys, "argv", ["run_tracking_curve.py", "--tables", str(tmp_path)])
    with pytest.raises(SystemExit, match="not the published ones"):
        T.main()
