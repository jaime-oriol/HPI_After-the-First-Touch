"""Tests for report_figures.validation, heterogeneity block: the placebo the prose cites.

The placebo is built by `s03_value.step_heterogeneity` and persisted by `run_validation`; the
registry publishes it beside the real result, with `placebo_clean` decided by a rule fixed before
looking: neither permutation reaches p < 0.025 on the BLP slope or on the GATES gap, the Bonferroni
level the real verdict itself uses.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import report_figures as rf


def _record(placebo_ps: dict | None) -> pd.DataFrame:
    rows = [{"population": "corner", "block": "heterogeneity", "key": "blp", "delta": 0.86,
             "p_wilcoxon": 0.0001, "n_test": 20, "verdict": "heterogeneous"},
            {"population": "corner", "block": "heterogeneity", "key": "gates", "delta": 0.0074,
             "p_wilcoxon": 0.40, "n_test": 20, "verdict": "not actionable"}]
    for mode, (p_blp, p_gates) in (placebo_ps or {}).items():
        rows.append({"population": "corner", "block": "heterogeneity", "key": f"placebo_{mode}",
                     "delta": 0.01, "p_wilcoxon": p_blp, "ll_lo": 0.0002, "ll_hi": p_gates,
                     "n_test": 20, "verdict": "no detectable structure"})
    return pd.DataFrame(rows)


def test_the_placebo_is_published_and_clean_when_neither_permutation_fires(monkeypatch):
    rec = _record({"global": (1.0, 0.9), "within_match": (0.7, 1.0)})
    monkeypatch.setattr(rf, "_load", lambda name: rec if name.startswith("validation") else None)
    h = rf.validation()["corner"]["heterogeneity"]
    assert h["placebo"]["global"] == {"blp_slope": 0.01, "blp_p": 1.0, "gates_gap": 0.0002,
                                      "gates_p": 0.9}
    assert set(h["placebo"]) == {"global", "within_match"}
    assert h["placebo_clean"] is True


def test_one_permutation_below_the_level_makes_it_unclean(monkeypatch):
    rec = _record({"global": (1.0, 0.9), "within_match": (0.01, 1.0)})
    monkeypatch.setattr(rf, "_load", lambda name: rec if name.startswith("validation") else None)
    assert rf.validation()["corner"]["heterogeneity"]["placebo_clean"] is False


def test_a_record_without_a_placebo_publishes_none_rather_than_a_verdict(monkeypatch):
    monkeypatch.setattr(rf, "_load",
                        lambda name: _record(None) if name.startswith("validation") else None)
    h = rf.validation()["corner"]["heterogeneity"]
    assert "placebo" not in h and "placebo_clean" not in h
