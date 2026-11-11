"""Tests for pipeline/run_value — which model the calibration record describes.

`calibration.parquet` is quoted in prose as the calibration of the published head, so it has to be
measured on the head the driver adopts, after the nested tuning, not on the constructor's defaults.
The order is what fixes that, and the test pins the order: every fit the driver reports comes after
the tuned configuration is applied.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import run_value as rv


def test_the_tuned_configuration_is_applied_before_anything_is_fitted_or_calibrated(monkeypatch,
                                                                                    tmp_path):
    calls = []
    classes = ["clear_safe", "no_second_ball"]
    table = pd.DataFrame({"match_id": [1, 1, 2, 2], "target": ["clear_safe", "no_second_ball"] * 2})
    path = tmp_path / "corners_toy.parquet"
    table.to_parquet(path)

    cv = SimpleNamespace(class_counts={}, log_loss=0.0, brier=0.0, accuracy=0.0, smooth_ece=0.0,
                         ece=0.0, classes=classes)
    cal = SimpleNamespace(log_loss_before=0.0, log_loss_after=0.0, smooth_ece_before=0.0,
                          smooth_ece_after=0.0, brier_before=0.0, brier_after=0.0, ece_before=0.0,
                          ece_after=0.0, oof_proba_cal=np.full((4, 2), 0.5))

    class Model:
        lambda_counter = 0.0

        def tune_cv(self, *a, **k):
            calls.append("tune")
            return SimpleNamespace(n_outer=3, best_params={}, outer_log_loss=0.0,
                                   outer_log_loss_std=0.0)

        def apply_best_params(self, tune):
            calls.append("apply")
            return self

        def fit_cv(self, *a, **k):
            calls.append("fit")
            return cv

        def calibrate_cv(self, *a, **k):
            calls.append("calibrate")
            return cal

        def classwise_smooth_ece(self, *a, **k):
            return pd.DataFrame({"class": classes})

        def xsecondball(self, t):
            calls.append("xsb")
            return np.zeros(len(t))

        def hurdle_xg(self, t, *a, **k):
            return np.linspace(0.1, 0.2, len(t))

        def ablate(self, *a, **k):
            calls.append("ablate")
            return pd.DataFrame({"group": ["ALL"]})

    monkeypatch.setattr(rv, "CompetingOutcomesModel", Model)
    monkeypatch.setattr(sys, "argv", ["run_value.py", "--table", str(path)])
    rv.main()
    first_apply = calls.index("apply")
    assert all(calls.index(step) > first_apply for step in ("fit", "calibrate", "xsb", "ablate")), calls


def test_out_dir_writes_the_calibration_records_there(monkeypatch, tmp_path):
    """The pass under partition F writes the grant's records under their published names into its
    own folder, so --out-dir moves both records and nothing lands in data/derived."""
    path = tmp_path / "corners.parquet"
    pd.DataFrame({"match_id": [1, 1, 2, 2], "target": ["clear_safe", "no_second_ball"] * 2}
                 ).to_parquet(path)
    cv = SimpleNamespace(class_counts={}, log_loss=0.0, brier=0.0, accuracy=0.0, smooth_ece=0.0,
                         ece=0.0, classes=["clear_safe", "no_second_ball"])
    cal = SimpleNamespace(log_loss_before=0.0, log_loss_after=0.0, smooth_ece_before=0.0,
                          smooth_ece_after=0.0, brier_before=0.0, brier_after=0.0, ece_before=0.0,
                          ece_after=0.0, oof_proba_cal=np.full((4, 2), 0.5))
    model = SimpleNamespace(
        lambda_counter=0.0, apply_best_params=lambda t: None, fit_cv=lambda *a, **k: cv,
        tune_cv=lambda *a, **k: SimpleNamespace(n_outer=3, best_params={}, outer_log_loss=0.0,
                                                outer_log_loss_std=0.0),
        calibrate_cv=lambda *a, **k: cal, xsecondball=lambda t: np.zeros(len(t)),
        classwise_smooth_ece=lambda *a, **k: pd.DataFrame({"class": cv.classes}),
        hurdle_xg=lambda t, *a, **k: np.linspace(0.1, 0.2, len(t)),
        ablate=lambda *a, **k: pd.DataFrame({"group": ["ALL"]}))
    monkeypatch.setattr(rv, "CompetingOutcomesModel", lambda: model)
    monkeypatch.setattr(rv, "published_tag", lambda table: "")
    monkeypatch.setattr(rv, "assert_population", lambda *a: None)
    out = tmp_path / "zones_F"
    monkeypatch.setattr(sys, "argv", ["run_value.py", "--table", str(path), "--out-dir", str(out)])
    rv.main()
    assert {p.name for p in out.iterdir()} == {"calibration.parquet",
                                               "calibration_classwise.parquet"}


def test_a_missing_table_stops_instead_of_rebuilding_another(monkeypatch, tmp_path):
    """A --table that is not on disk is a wrong path, not a request to rebuild: the rebuild is the
    corner grant whatever the path named, so the run would model another table and say nothing.
    Only --build asks for it."""
    def rebuilt(*a, **k):
        raise AssertionError("rebuilt the corner grant for a table that was not asked for")

    monkeypatch.setattr(rv, "build_corner_table", rebuilt)
    monkeypatch.setattr(sys, "argv", ["run_value.py", "--table", str(tmp_path / "corners_typo.parquet")])
    with pytest.raises(SystemExit, match="absent"):
        rv.main()
