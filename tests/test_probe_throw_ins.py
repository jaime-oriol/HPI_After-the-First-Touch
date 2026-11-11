"""Tests for probe_throw_ins: the comparison's denominators come from the grammar, not from a literal.

The corner and free-kick rates beside the long throw divide shots by deliveries. Counted by the
same rule that builds the shots, the deliveries cannot drift apart from what they divide.
"""
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
for p in (_REPO / "src", _REPO / "pipeline"):
    sys.path.insert(0, str(p))

from s01_phases import SetPieceGrammar                                    # noqa: E402
from test_s01_phases import _corner, _free_kick, _ts                      # noqa: E402
import probe_throw_ins as pt                                              # noqa: E402


def test_the_denominators_are_counted_by_the_rule_that_builds_the_shots():
    """The paper's two populations: a free kick from the flank at x >= 85 is a delivery of the
    second one, a central one or one from deep is not. The corner always is."""
    events = pd.DataFrame([
        _corner(),
        {**_free_kick(x=95.0, y=5.0), "id": "FK_FLANK", "timestamp": _ts(30.0)},
        {**_free_kick(x=95.0, y=40.0), "id": "FK_CENTRAL", "timestamp": _ts(60.0)},
        {**_free_kick(x=70.0, y=5.0), "id": "FK_DEEP", "timestamp": _ts(90.0)},
    ])
    g = SetPieceGrammar(events, kinds=("corner", "free_kick_flank"))
    assert pt.delivery_counts(g) == {"corner": 1, "free_kick_flank": 1}


def test_a_match_with_no_set_piece_counts_zero_rather_than_failing():
    """A match with no delivery of a kind must contribute zero to the sum, not raise."""
    g = SetPieceGrammar(pd.DataFrame([_corner()]), kinds=("corner", "free_kick_flank"))
    assert pt.delivery_counts(g) == {"corner": 1, "free_kick_flank": 0}


def test_an_own_goal_is_a_goal_in_the_comparison_and_no_shot():
    """The line of a population counts its own goal in the goals per 100 and in no shot count,
    shot share or direct-shot count."""
    sub = pd.DataFrame({
        "match_id": [1, 1, 2, 2, 3],
        "segment": ["direct", "direct", "prolongation", "after_clearance", "after_clearance"],
        "after_first_touch": [False, False, True, True, True],
        "is_goal": [True, False, False, True, False],
        "shot_outcome": ["Goal", "Saved", "Off T", "Own Goal", "Blocked"],
        "xg": [0.4, 0.1, 0.05, 0.0, 0.02]})
    row = pt.comparison_row("corner", sub, 10, 20)
    assert (row["shots"], row["shots_per_100"]) == (4, 40.0)
    assert row["goals_per_100"] == 20.0
    assert row["direct_shots"] == 2
    assert row["aft_shot_pct"] == 50.0


def test_the_record_tag_keeps_every_other_run_off_the_published_name():
    """A subset, an override or another shot table each keep a run off throw_in_probe.parquet; a
    subset's tag already does, so its record is the `_limitN` one eda_throw_ins --limit N reads."""
    grant, other = pt.kind_inputs("corner")[1], Path("elsewhere/shots_two_limit200.parquet")
    assert pt.record_tag(None, False, grant) == ""
    assert pt.record_tag(None, False, other) == "_shots_two_limit200"
    assert pt.record_tag(150, False, other) == pt.record_tag(150, False, grant) == "_limit150"
    assert pt.record_tag(None, True, grant) == "_override"
