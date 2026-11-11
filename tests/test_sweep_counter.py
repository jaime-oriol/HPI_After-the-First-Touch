"""Tests for pipeline/sweep_counter — the window sweep behind "the counter arm is not estimable".

The sweep counts a delivery as cleared on the same rule s01 labels it with, which closes on the
episode end: a clearance thirty seconds after the corner is a later passage of play, not this
delivery's. The pair below pins the rule from both sides: outside the episode is not a clearance
of this delivery, and inside it is.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import sweep_counter as sc
from conftest import requires_sb
from test_s01_phases import ATT, DEF, WIN, _corner, _row


def test_a_clearance_outside_the_episode_is_not_swept():
    """First contact at 1 s, so the episode closes one window later; a clearance past that is a
    later passage of play, and the delivery is not in the cleared set s01 would give."""
    ev = pd.DataFrame([_corner(), _row(1.0, "Ball Receipt*", ATT),
                       _row(1.0 + WIN + 5.0, "Clearance", DEF)])
    assert sc.sweep_events(ev, 0, ("corner",)) == []


def test_a_clearance_inside_the_episode_is_swept():
    """The control: the same corner cleared inside its episode is swept, once."""
    ev = pd.DataFrame([_corner(), _row(1.0, "Ball Receipt*", ATT), _row(2.0, "Clearance", DEF)])
    out = sc.sweep_events(ev, 0, ("corner",))
    assert len(out) == 1
    assert out[0]["set_piece_id"] == "CORNER"


def test_an_attacking_shot_later_in_the_half_does_not_outrank_the_counter():
    """Only the delivery's own attacking shot, the one its s01 label carries, takes priority over
    `clear_counter` in the outcome ladder. A shot minutes later in the same half is another
    passage of play and suppresses nothing."""
    ev = pd.DataFrame([_corner(), _row(1.0, "Ball Receipt*", ATT), _row(2.0, "Clearance", DEF),
                       _row(400.0, "Shot", ATT)])
    out = sc.sweep_events(ev, 0, ("corner",))
    assert len(out) == 1 and not out[0]["has_att_shot"]


def test_a_forced_own_goal_is_the_delivery_s_own_attacking_shot():
    """s01 counts a forced own goal as an attacking terminal event, so the ladder files the delivery
    under a shot class and a counter can never outrank it."""
    ev = pd.DataFrame([_corner(), _row(1.0, "Ball Receipt*", ATT), _row(2.0, "Clearance", DEF),
                       _row(3.0, "Own Goal For", ATT)])
    out = sc.sweep_events(ev, 0, ("corner",))
    assert len(out) == 1 and out[0]["has_att_shot"]


@requires_sb
def test_the_outranked_count_is_persisted_per_window(tmp_path, monkeypatch):
    """evidence.md §8 quotes how many swept deliveries also produced their own attacking shot, so the
    sweep persists it beside every window, where report_figures can publish it and a test bind it."""
    real = sc.L.list_matches
    monkeypatch.setattr(sc.L, "list_matches", lambda source=None: real(source).head(6))
    out = tmp_path / "sweep.parquet"
    monkeypatch.setattr(sys, "argv", ["sweep_counter.py", "--jobs", "1", "--out", str(out)])
    sc.main()
    s = pd.read_parquet(out)
    assert "outranked" in s.columns and (s["outranked"] <= s["events"]).all()


@requires_sb
def test_a_flank_sweep_never_lands_on_the_corner_name(tmp_path, monkeypatch):
    """`counter_sweep.parquet` carries the corner verdict report_figures publishes, so another
    population tags the default name, as --limit already does. In-process on six matches, with the
    repo root moved."""
    real = sc.L.list_matches
    monkeypatch.setattr(sc.L, "list_matches", lambda source=None: real(source).head(6))
    monkeypatch.setattr(sc, "_REPO", tmp_path)
    monkeypatch.setattr(sys, "argv", ["sweep_counter.py", "--kinds", "free_kick_flank", "--jobs", "1"])
    sc.main()
    der = tmp_path / "data" / "derived"
    assert not (der / "counter_sweep.parquet").exists()
    assert (der / "counter_sweep_free_kick_flank.parquet").exists()
