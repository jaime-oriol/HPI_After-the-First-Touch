"""Tests for pipeline/describe_frontier — the obs/exp cells the elementary layer publishes.

Two properties a published cell must have. A percentile interval of zero width is not a certainty
but a bootstrap that saw no variation at all, so it cannot exclude parity. And every block that
publishes `excludes_1` carries the exact Poisson bound beside it, so the two can be read against
each other on the cells where they disagree.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import describe_frontier as df


def _frame(n_matches: int = 30) -> pd.DataFrame:
    """One scoreless cell and one scoring cell, one shot each per match."""
    rng = np.random.default_rng(0)
    rows = []
    for m in range(n_matches):
        rows.append({"match_id": m, "key": "scoreless", "xg": 0.1, "is_goal": 0})
        rows.append({"match_id": m, "key": "scoring", "xg": 0.1, "is_goal": int(rng.random() < 0.3)})
    return pd.DataFrame(rows)


def test_a_zero_width_interval_excludes_nothing():
    """A cell with no goals resamples to 0 every time, so its percentile interval is [0, 0].

    Read naively that interval sits below 1 and "excludes parity", which would publish a cell with
    expected goals and none scored as mis-priced. A bootstrap that never moves carries no evidence
    either way."""
    out = df.clustered_oe(_frame(), "key", ["scoreless", "scoring"], boot=200)
    cell = out.loc["scoreless"]
    assert cell["lo"] == cell["hi"] == 0.0
    assert not cell["excludes_1"]


def test_the_origin_block_carries_the_exact_poisson_bound():
    """The origin block publishes poisson_lo/poisson_hi beside excludes_1, like every other block.

    Two of the cells whose verdict depends on which interval is read live in this block, so
    without the columns there is nothing to read that verdict against."""
    shots = _frame().assign(segment="after_clearance",
                            origin=lambda f: np.where(f["key"] == "scoring",
                                                      "clearance_on_touch", "clearance_on_shot"))
    out = df.origin_table(shots, boot=200)
    assert {"poisson_lo", "poisson_hi"} <= set(out.columns)
    assert out[["poisson_lo", "poisson_hi"]].notna().all().all()


def _with_own_goal(n_matches: int = 20) -> pd.DataFrame:
    """A direct and an after-clearance shot per match, the first match's clearance shot scoring,
    and one own goal after a clearance: a goal with xG 0.0 and no body part."""
    rows = []
    for m in range(n_matches):
        for seg, xg, goal in (("direct", 0.1, False), ("after_clearance", 0.2, m == 0)):
            rows.append({"match_id": m, "segment": seg, "xg": xg, "is_goal": goal,
                         "after_first_touch": seg == "after_clearance", "shot_outcome": "Saved",
                         "shot_body_part": "Right Foot", "t_since_delivery_s": 3.0,
                         "n_touches_before": 1})
    rows.append({**rows[1], "xg": 0.0, "is_goal": True, "shot_outcome": "Own Goal",
                 "shot_body_part": None})
    return pd.DataFrame(rows)


def test_an_own_goal_counts_as_a_goal_and_enters_no_ratio():
    """An own goal is a goal the delivery produced but not a shot: it is counted apart and in the
    share of goals, and never in a ratio over shots or over xG, where a goal no model priced would
    lift its cell."""
    shots = _with_own_goal()
    cell = df.segment_table(shots, boot=50).loc["after_clearance"]
    assert (cell["shots"], cell["goals"], cell["own_goals"]) == (20, 1, 1)
    assert cell["conv"] == 1 / 20 and np.isclose(cell["obs_exp"], 1 / 4.0)
    assert np.isclose(cell["goal_share"], 1.0)
    body = df.add_body_cut(shots)
    assert body.loc[shots["shot_outcome"].eq("Own Goal"), "body_cell"].isna().all()
    hl = df.headline(shots, boot=50)["point"]
    assert hl["share_shots_after_first_touch"] == 0.5 and hl["share_goals_after_first_touch"] == 1.0
    assert hl["conv_after_first_touch"] == 1 / 20
