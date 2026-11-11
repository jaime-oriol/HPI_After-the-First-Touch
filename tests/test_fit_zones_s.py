"""Tests for fit_zones_s, on synthetic clouds of points: the points are the attackers in the box, K
is the smallest within one standard error of the best and is measured on whole matches held out, the
components are named by C's cells from the goal line out, the labels are the mixture's in the box and
C's cells outside it, and frozen parameters are never written over unasked.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "src", _REPO / "pipeline"):
    sys.path.insert(0, str(p))

import fit_zones_s as S                                                    # noqa: E402


def _clouds(n_matches: int = 60, per: int = 12, seed: int = 0) -> pd.DataFrame:
    """Three tight clouds in the box (six-yard near, spot central, front far), every match drawing
    from all three, three bodies to a delivery."""
    rng = np.random.default_rng(seed)
    centres = np.array([[116.0, 33.0], [110.0, 40.0], [105.0, 52.0]])
    n = n_matches * per
    xy = centres[rng.integers(0, 3, n)] + rng.normal(scale=0.8, size=(n, 2))
    return pd.DataFrame({"set_piece_id": [f"k{i // 3}" for i in range(n)],
                         "match_id": np.repeat(np.arange(n_matches), per),
                         "x": xy[:, 0], "y": xy[:, 1]})


def test_the_points_are_the_attackers_in_the_box_on_the_populations_deliveries():
    t = pd.DataFrame({"set_piece_id": ["a", "b"], "match_id": [1, 2]})
    bodies = pd.DataFrame({"set_piece_id": ["a", "a", "a", "b", "c"],
                           "side": ["att", "att", "def", "att", "att"],
                           "x": [110.0, 98.0, 110.0, 115.0, 110.0],
                           "y": [40.0, 40.0, 40.0, 10.0, 40.0]})
    p = S.points(t, bodies)
    assert list(p["set_piece_id"]) == ["a"] and list(p["match_id"]) == [1]


def test_k_is_the_smallest_within_one_standard_error_of_the_best():
    se = [0.1] * 4
    assert S.choose_k(pd.DataFrame({"k": [2, 3, 4, 5], "score": [-5.0, -4.2, -3.95, -3.9],
                                    "se": se})) == 4
    assert S.choose_k(pd.DataFrame({"k": [2, 3, 4, 5], "score": [-3.0, -3.5, -3.6, -3.7],
                                    "se": se})) == 2


def test_the_curve_holds_out_whole_matches_and_finds_three_clouds(monkeypatch):
    monkeypatch.setattr(S, "K_RANGE", (2, 3, 4, 5))
    monkeypatch.setattr(S, "_RESTARTS", 2)
    p, real, seen = _clouds(), S.held_out, []

    def spy(k, xy, tr, te):
        seen.append((tr, te))
        return real(k, xy, tr, te)

    monkeypatch.setattr(S, "held_out", spy)
    curve = S.k_curve(p, 1)
    g = p["match_id"].to_numpy()
    assert len(seen) == len(S.K_RANGE) * S._FOLDS
    assert all(set(g[tr]).isdisjoint(g[te]) for tr, te in seen)
    assert S.choose_k(curve) == 3


def test_components_are_named_by_cs_cells_and_numbered_from_the_goal_line_out():
    means = np.array([[116.0, 33.0], [110.0, 40.0], [111.0, 41.0], [105.0, 52.0]])
    assert S.names(means) == ["six_near", "spot_central_2", "spot_central_1", "front_far"]


def test_the_fit_orders_its_components_by_cs_cells_and_serialises(monkeypatch):
    monkeypatch.setattr(S, "_RESTARTS", 2)
    params = S.fit(_clouds(), 3)
    assert params["names"] == ["six_near", "spot_central", "front_far"]
    assert np.isclose(sum(params["weights"]), 1.0)
    assert json.loads(json.dumps(params)) == params


def test_the_labels_are_the_mixtures_in_the_box_and_cs_cells_outside_it(monkeypatch):
    monkeypatch.setattr(S, "_RESTARTS", 2)
    gmm = S.mixture(3).fit(_clouds()[["x", "y"]].to_numpy(float))
    params = {"names": ["a", "b", "c"], "weights": gmm.weights_.tolist(),
              "means": gmm.means_.tolist(), "covariances": gmm.covariances_.tolist()}
    rng = np.random.default_rng(1)
    x, y = rng.uniform(102.0, 120.0, 500), rng.uniform(18.0, 62.0, 500)
    want = np.asarray(params["names"], dtype=object)[gmm.predict(np.c_[x, y])]
    assert list(S.label(params, x, y)) == list(want)
    assert list(S.label(params, np.array([98.0, 110.0, 80.0]),
                        np.array([40.0, 70.0, 40.0]))) == ["edge", "wide", "deep"]


def test_two_clouds_split_the_box_into_one_region_each():
    eye = np.eye(2).tolist()
    params = {"names": ["near", "far"], "weights": [0.5, 0.5],
              "means": [[110.0, 30.0], [110.0, 50.0]], "covariances": [eye, eye]}
    assert S.regions(params) == {"near": 1, "far": 1}


def test_halves_of_separated_clouds_agree(monkeypatch):
    monkeypatch.setattr(S, "_RESTARTS", 2)
    assert S.halves(_clouds(), 3) > 0.9


def _refuses(monkeypatch, argv: list, match: str) -> None:
    def work(*a, **k):
        raise AssertionError("the run read the tables before refusing")

    monkeypatch.setattr(S, "inputs", work)
    monkeypatch.setattr(sys, "argv", ["fit_zones_s.py", *argv])
    with pytest.raises(SystemExit, match=match):
        S.main()


def test_a_reduced_run_may_not_write_the_frozen_parameters(monkeypatch):
    _refuses(monkeypatch, ["--limit", "40"], "may not write")


def test_frozen_parameters_are_never_replaced_unasked(monkeypatch, tmp_path):
    frozen = tmp_path / "zones_s.json"
    frozen.write_text("{}\n")
    _refuses(monkeypatch, ["--params", str(frozen)], "frozen")
