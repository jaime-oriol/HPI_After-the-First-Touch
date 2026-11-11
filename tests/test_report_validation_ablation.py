"""Tests for report_figures.validation — which interval decides whether an ablation is a finding.

The fixed-fit interval holds the out-of-fold probabilities and is too narrow, and the widening is
not one factor across blocks (evidence.md §11). So a published count of the groups that cover zero
is read on each block's own refit interval, in normal form, when the run measured it; the pivotal
and percentile forms travel beside it, and the record says which interval it used.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import report_figures as rf

_FIXED = {"delivery": (0.0269, 0.0229, 0.0309), "edge": (-0.0003, -0.0006, -0.00002),
          "box": (-0.0002, -0.0006, 0.0002)}
# Pivotal bounds; the percentile ones sit 0.001 below, as the refit shift puts them.
_PIVOTAL = {"delivery": (0.02349, 0.03216), "edge": (-0.00057, 0.00102), "box": (-0.00048, 0.00100)}


def _only_validation(frame: pd.DataFrame):
    """A stub `_load` that serves the record and nothing else, as before run_refit_draws ran."""
    return lambda name: frame if name.startswith("validation") else None


def _record(with_refit: bool) -> pd.DataFrame:
    rows = [{"population": "corner", "block": "ablation", "key": k, "delta": d, "delta_lo": lo,
             "delta_hi": hi} for k, (d, lo, hi) in _FIXED.items()]
    if with_refit:
        rows += [{"population": "corner", "block": "ablation_refit", "key": k,
                  "delta": _FIXED[k][0], "delta_lo": lo, "delta_hi": hi,
                  "ll_lo": lo - 0.001, "ll_hi": hi - 0.001, "n_test": 300, "brier": 2.0}
                 for k, (lo, hi) in _PIVOTAL.items()]
    return pd.DataFrame(rows)


def test_the_count_is_read_on_each_blocks_refit_normal_interval(monkeypatch):
    monkeypatch.setattr(rf, "_load", _only_validation(_record(with_refit=True)))
    c = rf.validation()["corner"]
    edge = c["ablation"]["edge"]
    assert edge["ci"] == [-0.0006, -0.0], "the fixture must disagree across forms"
    assert edge["normal_ci"] == [-0.0011, 0.00049]
    assert edge["adjusted_ci"] == [-0.00057, 0.00102]
    assert edge["refit_ci"] == [-0.00157, 0.00002]
    assert edge["covers_zero"] is True
    assert c["ablation"]["delivery"]["covers_zero"] is False
    assert c["n_groups_covering_zero"] == 2
    assert c["ablation_interval"] == "refit_normal"


def test_without_the_refit_block_the_count_says_it_is_fixed(monkeypatch):
    monkeypatch.setattr(rf, "_load", _only_validation(_record(with_refit=False)))
    c = rf.validation()["corner"]
    assert c["n_groups_covering_zero"] == 1
    assert c["ablation_interval"] == "fixed_oof"
    assert "normal_ci" not in c["ablation"]["edge"]


def _ladder(m1: float, refit: tuple | None = None) -> pd.DataFrame:
    """A two-rung ladder whose delivery step has lower bound `m1`, the margin's anchor, and
    optionally the 360 step's refit row (point, pivotal lo, pivotal hi)."""
    steps = {"delivery": (m1 + 0.002, m1, m1 + 0.004), "full": (-0.0016, -0.0062, 0.0033)}
    rows = [{"population": "corner", "block": "ladder", "key": k, "log_loss": 1.4,
             "delta": d, "delta_lo": lo, "delta_hi": hi} for k, (d, lo, hi) in steps.items()]
    if refit is not None:
        d, lo, hi = refit
        rows.append({"population": "corner", "block": "refit", "key": "step", "delta": d,
                     "delta_lo": lo, "delta_hi": hi, "ll_lo": 2 * d - hi, "ll_hi": 2 * d - lo,
                     "n_test": 300, "brier": 1.5})
    return pd.DataFrame(rows)


def test_a_margin_anchored_on_a_non_positive_m1_is_published_as_undefined(monkeypatch):
    """With M1 not above zero there is no reference effect, and M2 < 0 makes both booleans true or
    false by construction: a population whose delivery step does not clear zero lands here. A
    verdict there is one nobody measured."""
    monkeypatch.setattr(rf, "_load", _only_validation(_ladder(-0.0015)))
    ni = rf.validation()["corner"]["non_inferiority"]
    assert ni["margin_defined"] is False
    assert ni["adds_less_than_m2"] is None and ni["two_sided_equivalent"] is None
    assert ni["upper_as_fraction_of_m1"] is None


def test_the_fixed_bound_decides_only_without_a_refit_row(monkeypatch):
    monkeypatch.setattr(rf, "_load", _only_validation(_ladder(0.0229)))
    ni = rf.validation()["corner"]["non_inferiority"]
    assert ni["m2"] == pytest.approx(0.00229)
    assert ni["ci_is_refit_normal"] is False
    assert ni["adds_less_than_m2"] is False
    assert ni["two_sided_equivalent"] is False
    assert ni["upper_as_fraction_of_m1"] == pytest.approx(0.144)


def test_the_refit_normal_bound_decides_and_can_pass(monkeypatch):
    """The corner case: the pivotal upper bound fails a tenth of M1, the normal one passes it."""
    monkeypatch.setattr(rf, "_load", _only_validation(_ladder(0.0331, (-0.00139, -0.00139, 0.00677))))
    ni = rf.validation()["corner"]["non_inferiority"]
    assert ni["ci_is_refit_normal"] is True
    assert ni["step_ci"] == [-0.00547, 0.00269]
    assert ni["adds_less_than_m2"] is True
    assert ni["two_sided_equivalent"] is False
    assert ni["upper_as_fraction_of_m1"] == pytest.approx(0.081)
