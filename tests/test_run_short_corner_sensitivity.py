"""Tests for run_short_corner_sensitivity: the arms, the cross build, and what the arms may share.

The populations must be treated identically, so nothing but the rows may change between them; a
version that re-selected C on one side would answer a different question and still print a
plausible table. The cross build must read what s01 delivers (the attacking team's FIRST cross of
the same possession, never one from a later possession or from the defence, which the grammar's own
tests pin) and keep what the short corners no cross reaches in time produced.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "src", _REPO / "pipeline"):
    sys.path.insert(0, str(p))

from s01_phases import SetPieceGrammar                              # noqa: E402
from test_s01_phases import ATT, _corner, _cross, _row               # noqa: E402
import run_short_corner_sensitivity as sc                           # noqa: E402


class _FakeModel:
    """Records what each arm is handed; tuning raises unless the test enables it."""
    calls: list = []

    def __init__(self, n_splits=5):
        pass

    def tune_cv(self, table, n_jobs=1):
        _FakeModel.calls.append("tuned")
        return {}

    def apply_best_params(self, t):
        pass

    def delivery_vs_360(self, table, n_boot=2000):
        _FakeModel.calls.append((len(table), n_boot))
        return {"delta": 0.01, "lo": 0.005, "hi": 0.015}


def test_contrast_passes_the_same_bootstrap_count_to_both_arms(monkeypatch):
    """An interval is not comparable with one built on a different resampling budget."""
    _FakeModel.calls = []
    monkeypatch.setattr(sc, "CompetingOutcomesModel", _FakeModel)
    t = pd.DataFrame({"match_id": np.repeat(np.arange(4), 5),
                      "short_corner": [True] * 5 + [False] * 15})
    a = sc.contrast(t, boot=777, tune=False, jobs=1)
    b = sc.contrast(t[~t["short_corner"]], boot=777, tune=False, jobs=1)
    assert _FakeModel.calls == [(20, 777), (15, 777)]
    assert a["n"] == 20 and b["n"] == 15 and a["matches"] == 4


def test_tuning_is_off_unless_asked(monkeypatch):
    """The default arm isolates the effect of dropping rows; re-selecting C would move two things."""
    _FakeModel.calls = []
    monkeypatch.setattr(sc, "CompetingOutcomesModel", _FakeModel)
    t = pd.DataFrame({"match_id": [1, 1, 2], "short_corner": [True, False, False]})
    sc.contrast(t, boot=10, tune=False, jobs=1)
    assert "tuned" not in _FakeModel.calls
    sc.contrast(t, boot=10, tune=True, jobs=1)
    assert "tuned" in _FakeModel.calls


def test_the_window_grid_does_not_repeat_the_window_the_table_is_built_on():
    """The arm at the grammar's window is the table's own short rows, so the grid around it holds others."""
    assert SetPieceGrammar.SHORT_CROSS_WINDOW_S not in sc.CROSS_WINDOWS_S
    assert min(sc.CROSS_WINDOWS_S) < SetPieceGrammar.SHORT_CROSS_WINDOW_S < max(sc.CROSS_WINDOWS_S)


def _second_short_corner() -> dict:
    return _row(100.0, "Pass", ATT, id="K2", **{
        "pass.type.name": "Corner", "location": [120.0, 0.1], "pass.end_location": [110.0, 40.0],
        "pass.height.name": "Ground Pass", "pass.length": 8.0, "possession": 2})


def test_the_cross_build_keeps_the_crosses_the_kicks_the_delays_and_what_no_cross_reached(monkeypatch):
    """Two short corners: the first crossed at 3 s, inside the window, the second at 10 s, past it but
    inside the horizon the other arms read. The second has no delivery in the table, and what its
    kick produced, a goal here, is counted beside the arms."""
    ev = pd.DataFrame([
        _corner(**{"pass.length": 8.0}), _cross(3.0, id="X1"), _second_short_corner(),
        _row(104.0, "Ball Receipt*", ATT, possession=2, player=9.0),
        _row(105.0, "Shot", ATT, possession=2, player=9.0, id="S", **{"shot.outcome.name": "Goal"}),
        _cross(110.0, id="X2", possession=2)])
    fz = pd.DataFrame({"event_uuid": ["CORNER", "X1", "K2", "X2"]})
    monkeypatch.setattr(sc.L, "load_events", lambda m: ev)
    monkeypatch.setattr(sc.L, "load_360", lambda m: fz)
    monkeypatch.setattr(sc, "delivery_row", lambda c, lab, frame, mid, league: {
        "set_piece_id": c["id"], "match_id": mid, "outcome": lab["outcome"], "short_corner": lab["short_corner"]})
    got = sc.cross_match(7, "serie_a", 15.0)
    assert got["n_short"] == 2 and got["delays"] == [3.0, 10.0]
    assert [(r["set_piece_id"], r["kick_id"], r["delay_s"]) for r in got["crosses"]] == [
        ("X1", "CORNER", 3.0), ("X2", "K2", 10.0)]
    assert [r["set_piece_id"] for r in got["kick_rows"]] == ["CORNER", "K2"]
    assert all(r["short_corner"] for r in got["crosses"] + got["kick_rows"])
    assert got["uncrossed"] == {"corners": 1, "shots": 1, "goals": 1, "goal_corners": 1}
    assert got["league"] == "serie_a" and got["goals_by_team"] == {ATT: 1}
    assert [r["set_piece_id"] for r in sc.cross_match(7, "serie_a", 5.0)["crosses"]] == ["X1"]


def test_a_subset_run_may_not_write_the_published_record(tmp_path):
    """A foreign table or a --limit build without --out must stop before any work."""
    other = tmp_path / "corners.parquet"
    pd.DataFrame({"short_corner": [True]}).to_parquet(other)
    for extra in (["--table", str(other)], ["--limit", "3"]):
        proc = subprocess.run([sys.executable, "pipeline/run_short_corner_sensitivity.py", *extra],
                              cwd=_REPO, capture_output=True, text=True, timeout=300)
        assert proc.returncode != 0
        assert "may not write the published" in proc.stderr
