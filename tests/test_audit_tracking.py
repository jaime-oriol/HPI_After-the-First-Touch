"""Tests for audit_tracking: the transition-window coverage reads the population it is asked for, and
one tracking load per match feeds every population, each into its own record.
"""
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "src", _REPO / "pipeline"):
    sys.path.insert(0, str(p))

import audit_tracking as A                                                 # noqa: E402


def test_the_coverage_reads_the_population_asked_for(monkeypatch):
    seen = {}

    class Situation:
        def __init__(self, ev, fz, kinds=("corner",)):
            seen["kinds"] = kinds
            self.g = type("Grammar", (), {"deliveries": staticmethod(
                lambda: pd.DataFrame({"id": pd.Series([], dtype=object)}))})()

    monkeypatch.setattr(A, "DropSituation", Situation)
    tracking = pd.DataFrame({"is_ball": pd.Series([], dtype=int)})
    assert A.transition_window_coverage(1, None, None, tracking, "free_kick_flank").empty
    assert seen["kinds"] == ("free_kick_flank",)


def test_one_tracking_load_feeds_every_population(monkeypatch, tmp_path):
    """Tracking is the costly read, so both populations share each match's frame, and each writes its
    own record under its own name."""
    loads = []
    monkeypatch.setattr(A, "cross_link_check", lambda: pd.DataFrame({"match_id": [1, 2]}))
    monkeypatch.setattr(A.L, "list_matches", lambda source=None: pd.DataFrame({"match_id": [1, 2, 3]}))
    monkeypatch.setattr(A.T, "load_tracking", lambda mid: loads.append(mid) or pd.DataFrame())
    monkeypatch.setattr(A.L, "load_events", lambda mid: None)
    monkeypatch.setattr(A.L, "load_360", lambda mid: None)
    monkeypatch.setattr(A, "coord_stats", lambda mid, trk: {})
    monkeypatch.setattr(A, "merge_coord", lambda parts: {})
    monkeypatch.setattr(A, "events_coordinate_check", lambda: {})
    monkeypatch.setattr(A, "report_coordinates", lambda coord, n: None)
    monkeypatch.setattr(A, "transition_window_coverage", lambda mid, ev, fz, trk, kind: pd.DataFrame(
        {"match_id": [mid], "corner_id": [f"{kind}-{mid}"], "visible_frac": [1.0],
         "high_conf_frac": [1.0], "mean_players_per_frame": [20.0], "ball_frame_frac": [1.0],
         "ball_visible_frac": [1.0]}))
    monkeypatch.setattr(sys, "argv", ["audit_tracking.py", "--limit", "2", "--kinds",
                                      "corner,free_kick_flank", "--out-dir", str(tmp_path)])
    A.main()
    assert loads == [1, 2]
    for name, kind in (("audit_tracking_limit2", "corner"),
                       ("audit_tracking_free_kick_flank_limit2", "free_kick_flank")):
        assert pd.read_parquet(tmp_path / f"{name}.parquet")["corner_id"].tolist() == [
            f"{kind}-1", f"{kind}-2"]
