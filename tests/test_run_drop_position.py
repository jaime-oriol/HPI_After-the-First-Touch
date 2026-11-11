"""Tests for run_drop_position: does the adjustment actually adjust?

The failure that matters here is silent. A stratified estimate that quietly keeps comparing across
cells, or an adjusted one whose position terms do nothing, returns the raw gap wearing a different
name and nobody notices, because the number looks plausible either way. So the tests build data
where the answer is known by construction: a gap that is ENTIRELY position, which both adjusted
readings must send to zero, and a gap that is entirely real, which they must leave alone.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "src", _REPO / "pipeline"):
    sys.path.insert(0, str(p))

import run_drop_position as dp                                       # noqa: E402


def _frame(rows: list) -> pd.DataFrame:
    d = pd.DataFrame(rows)
    d["cell_x"] = (d["drop_x"] // 6.0).astype(int)
    d["cell_y"] = (d["drop_y"] // 6.0).astype(int)
    return d


def _confounded(n: int = 60) -> pd.DataFrame:
    """All of the gap is position: value depends on x alone, and the planned state sits where x is
    high three times in four, the rescued one where it is low. The raw gap is +0.04 and none of it
    survives inside a cell, so any honest adjustment has to send it to zero."""
    rows = []
    for i in range(n):
        for state, x, k in (("won_as_planned", 100.0, 3), ("won_as_planned", 70.0, 1),
                            ("rescued", 100.0, 1), ("rescued", 70.0, 3)):
            rows += [{"four_state": state, "drop_x": x, "drop_y": 40.0,
                      "value_xg_after": 0.10 if x == 100.0 else 0.02, "match_id": i % 12}] * k
    return _frame(rows)


def _real(n: int = 60) -> pd.DataFrame:
    """The gap is real and position is balanced: planned is worth +0.05 wherever it happens."""
    rows = []
    for i in range(n):
        for x in (70.0, 100.0):
            rows.append({"four_state": "won_as_planned", "drop_x": x, "drop_y": 40.0,
                         "value_xg_after": 0.10, "match_id": i % 12})
            rows.append({"four_state": "rescued", "drop_x": x, "drop_y": 40.0,
                         "value_xg_after": 0.05, "match_id": i % 12})
    return _frame(rows)


def test_a_gap_that_is_entirely_position_is_adjusted_away():
    """The raw contrast sees a premium that is not there; both adjusted readings must not."""
    d = _confounded()
    assert dp.raw_gap(d) == _approx(0.04), "the fixture must carry a gap that is all position"
    assert abs(dp.stratified_gap(d)) < 1e-9
    assert abs(dp.adjusted_gap(d)) < 1e-6


def test_a_real_gap_survives_both_adjustments():
    """An adjustment that also erases a genuine effect would be useless. Position is balanced here,
    so all three readings have to agree on +0.05."""
    d = _real()
    assert dp.raw_gap(d) == _approx(0.05)
    assert dp.stratified_gap(d) == _approx(0.05)
    assert dp.adjusted_gap(d) == _approx(0.05)


def test_the_confounded_case_really_is_confounded_when_the_overlap_is_removed():
    """Guard on the fixture itself: without cells holding both states the raw gap is the full
    position effect, which is what the test above must be able to remove."""
    d = _confounded()
    skewed = d[((d["four_state"] == "won_as_planned") & (d["drop_x"] == 100.0))
               | ((d["four_state"] == "rescued") & (d["drop_x"] == 70.0))]
    assert dp.raw_gap(skewed) == _approx(0.08)
    assert np.isnan(dp.stratified_gap(skewed)), "no cell holds both states, so nothing is estimable"


def test_usable_rows_counts_only_cells_holding_both_states():
    """The number that tells a shrinking effect apart from a shrinking sample."""
    d = _confounded()
    assert dp.usable_rows(d) == len(d)
    lonely = pd.concat([d, _frame([{"four_state": "won_as_planned", "drop_x": 10.0,
                                    "drop_y": 10.0, "value_xg_after": 0.5, "match_id": 0}])])
    assert dp.usable_rows(lonely) == len(d)


def test_a_degenerate_draw_returns_nan_rather_than_a_number():
    """Inside a bootstrap draw a state can vanish. NaN is dropped by the cluster bootstrap; a zero
    would be averaged in as if it were a measurement."""
    d = _real()
    one_state = d[d["four_state"] == "rescued"]
    assert np.isnan(dp.adjusted_gap(one_state))
    assert np.isnan(dp.stratified_gap(one_state))


def test_an_interval_that_is_not_finite_decides_nothing():
    """`nan <= 0 <= nan` is False, which would publish EXCLUDES ZERO where no interval exists."""
    assert dp.covers_zero(np.nan, np.nan) is None
    assert dp.covers_zero(-0.01, np.nan) is None
    assert dp.covers_zero(-0.01, 0.02) is True
    assert dp.covers_zero(0.01, 0.02) is False


def test_the_default_run_writes_the_cells_the_record_publishes(monkeypatch, tmp_path):
    """report_figures reads every cell present and closes H3 on whether any adjusted reading keeps
    the premium, so a run without --cell has to write the recipe's sweep: one cell alone would
    narrow that claim in silence. Run in-process on the grant's name, the derived directory moved."""
    _real(24).drop(columns=["cell_x", "cell_y"]).to_parquet(tmp_path / "situation.parquet")
    monkeypatch.setattr(dp, "_DER", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_drop_position.py", "--boot", "20"])
    dp.main()
    cells = set(pd.read_parquet(tmp_path / "drop_position.parquet")["cell_yd"])
    assert cells == {4.0, 6.0, 8.0, 12.0}


def test_another_sweep_may_not_write_the_published_record(monkeypatch, tmp_path):
    """Any other set of cells narrows or shifts the same claim, so it needs --out."""
    _real(24).drop(columns=["cell_x", "cell_y"]).to_parquet(tmp_path / "situation.parquet")
    monkeypatch.setattr(dp, "_DER", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_drop_position.py", "--boot", "20", "--cell", "5"])
    with pytest.raises(SystemExit, match="may not write the published"):
        dp.main()
    assert not (tmp_path / "drop_position.parquet").exists()


def _approx(v: float, tol: float = 1e-9) -> object:
    return pytest.approx(v, abs=tol)
