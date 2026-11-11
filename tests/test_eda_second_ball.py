"""Tests for pipeline/eda_second_ball.py — the reliability guard on the team ranking.

Fully synthetic where the logic allows it (matching test_validate.py's philosophy):
the metric-name remap, the shrinkage and the reliability note are pure functions and
must not depend on eda_repeatability.py having been run.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import eda_second_ball as E


# ── _load_reliability: the metric-name remap onto team_table's own columns ──
def test_load_reliability_remaps_for_side_columns(monkeypatch, tmp_path):
    """Every metric except corners_per_match gets a `_for` suffix; the file is read
    from wherever _DER points, so a real run's artefact is never required."""
    rel = pd.DataFrame({"metric": ["corners_per_match", "goals_per100", "aft_goals_per100"],
                        "full_r_sb": [0.739, 0.247, 0.295]})
    rel.to_parquet(tmp_path / "eda_reliability_corner.parquet")
    monkeypatch.setattr(E, "_DER", tmp_path)

    out = E._load_reliability("corner")
    assert set(out.index) == {"corners_per_match", "goals_per100_for", "aft_goals_per100_for"}
    assert out.loc["aft_goals_per100_for", "full_r_sb"] == pytest.approx(0.295)


def test_kind_inputs_route_each_population_to_the_published_grains():
    """corner and free_kick_flank, the paper's two populations, share the published two-population
    grains; any other kind has none of its own and is refused."""
    from eda_build import kind_inputs
    for kind in ("corner", "free_kick_flank"):
        assert [p.name for p in kind_inputs(kind)] == ["eda_deliveries.parquet",
                                                       "shots_corner_free_kick_flank.parquet"]
    with pytest.raises(ValueError, match="penalty"):
        kind_inputs("penalty")


def test_load_reliability_reads_the_population_it_is_asked_for(monkeypatch, tmp_path):
    pd.DataFrame({"metric": ["aft_goals_per100"], "full_r_sb": [0.1]}).to_parquet(
        tmp_path / "eda_reliability_free_kick_flank.parquet")
    monkeypatch.setattr(E, "_DER", tmp_path)
    assert E._load_reliability("free_kick_flank").loc["aft_goals_per100_for", "full_r_sb"] == 0.1
    assert E._load_reliability("corner") is None


def test_load_reliability_none_when_never_run(monkeypatch, tmp_path):
    """No eda_repeatability.py output on disk -> None, never a crash or a silent zero."""
    monkeypatch.setattr(E, "_DER", tmp_path)
    assert E._load_reliability("corner") is None


# ── _shrink: Efron-Morris, and the order-preservation the printed warning claims ──
def test_shrink_pulls_toward_the_league_mean():
    rate = pd.Series([1.0, 5.0, 9.0])
    out = E._shrink(rate, reliability=0.3, league_rate=5.0)
    # 0.3*rate + 0.7*5.0
    assert list(out) == pytest.approx([3.8, 5.0, 6.2])


def test_shrink_is_order_preserving():
    """A single reliability shared by every team is a positive affine map, so the
    shrunk column can never reorder the raw one, only compress its spread. This is
    the exact claim eda_second_ball.py prints next to the shrunk column; if it broke,
    that message would be a lie."""
    rng = np.random.default_rng(0)
    rate = pd.Series(rng.normal(3.0, 1.5, size=50))
    shrunk = E._shrink(rate, reliability=0.29, league_rate=float(rate.mean()))
    assert list(rate.rank()) == list(shrunk.rank())


def test_shrink_reliability_one_is_the_identity():
    rate = pd.Series([1.0, 2.0, 3.0])
    assert list(E._shrink(rate, reliability=1.0, league_rate=99.0)) == pytest.approx(list(rate))


@pytest.mark.parametrize("r", [-0.146, -0.054, 0.0, 1.2])
def test_shrink_refuses_a_reliability_outside_the_unit_interval(r):
    """A thin population can measure a negative split-half reliability. Fed in, it would reverse
    the ranking it claims to keep; at zero every team collapses onto the mean, and above one the
    spread inflates."""
    with pytest.raises(ValueError, match="outside"):
        E._shrink(pd.Series([1.0, 2.0, 3.0]), reliability=r, league_rate=2.0)


# ── _reliability_note: never silent, always names which of the three cases ──
def test_reliability_note_missing_file():
    assert "UNMEASURED" in E._reliability_note(None, "aft_goals_per100_for")


def test_reliability_note_metric_not_covered():
    rel = pd.DataFrame({"full_r_sb": [0.8]}, index=["corners_per_match"])
    note = E._reliability_note(rel, "aft_goals_per100_ag")
    assert "UNMEASURED" in note and "aft_goals_per100_ag" in note


def test_reliability_note_below_floor():
    rel = pd.DataFrame({"full_r_sb": [0.295]}, index=["aft_goals_per100_for"])
    note = E._reliability_note(rel, "aft_goals_per100_for")
    assert "0.295" in note and "BELOW" in note


def test_reliability_note_clears_floor():
    rel = pd.DataFrame({"full_r_sb": [0.80]}, index=["corners_per_match"])
    note = E._reliability_note(rel, "corners_per_match")
    assert "0.800" in note and "publishable" in note


def test_reliability_floor_matches_koo_and_li_good_band():
    """Pinned so it cannot silently drift off the cited threshold."""
    assert E.RELIABILITY_FLOOR == 0.75


def test_an_unmeasurable_reliability_leaves_the_rate_unshrunk(tmp_path):
    """A split-half reliability that came out NaN, a population too thin for its halves to vary,
    measures nothing, so nothing is shrunk by it, as with one at or below zero."""
    from test_eda_repeatability import _season
    deliv, shots, stand = _season()
    pd.DataFrame({"metric": ["aft_goals_per100"], "full_r_sb": [np.nan]}).to_parquet(
        tmp_path / "eda_reliability_corner.parquet")
    t = E.team_table(deliv, shots, stand, "corner", tmp_path)
    assert t["aft_goals_per100_for_shrunk"].isna().all()


def test_the_goals_after_a_short_corner_no_cross_reached_make_the_teams_corner_goals_whole(tmp_path):
    """Such a corner has no delivery, so its goals are in no segment and not in `goals_for`: a team's corner goals told
    whole are the two, only the teams that scored one move, the shares keep their denominator, and an artefact that is not
    there leaves them unknown rather than zero."""
    from test_eda_repeatability import _season
    deliv, shots, stand = _season()
    t = E.team_table(deliv, shots, stand, "corner", tmp_path)
    assert t["goals_for_all"].isna().all()
    pd.DataFrame({"block": ["cross_delay", "uncrossed_team"], "key": ["crossed", "a|a0"], "n": [9, 2]}).to_parquet(
        tmp_path / "short_corner_sensitivity.parquet")
    whole = E.team_table(deliv, shots, stand, "corner", tmp_path)
    mine = whole["team"] == "a0"
    assert (whole.loc[mine, "goals_for_all"] == whole.loc[mine, "goals_for"] + 2).all()
    assert (whole.loc[~mine, "goals_for_all"] == whole.loc[~mine, "goals_for"]).all()
    assert (whole["aft_goals_for"] == t["aft_goals_for"]).all() and (whole["goals_for"] == t["goals_for"]).all()
    flank = E.team_table(deliv, shots, stand, "free_kick_flank", tmp_path)
    assert (flank["goals_for_all"] == flank["goals_for"]).all()


def test_load_reliability_reads_the_records_folder_it_is_given(tmp_path):
    pd.DataFrame({"metric": ["corners_per_match"], "full_r_sb": [0.5]}).to_parquet(
        tmp_path / "eda_reliability_corner.parquet")
    assert E._load_reliability("corner", tmp_path).loc["corners_per_match", "full_r_sb"] == 0.5


def test_the_flags_name_every_input_read(monkeypatch, tmp_path):
    """The smoke reads the grain and shots it built, and the standings its own earlier stage wrote,
    never the grant's."""
    seen = []

    def read(path, *a, **k):
        seen.append(Path(path))
        if len(seen) == 3:
            raise RuntimeError("read")
        return pd.DataFrame()

    monkeypatch.setattr(E.pd, "read_parquet", read)
    monkeypatch.setattr(sys, "argv", ["eda_second_ball.py", "--deliveries", str(tmp_path / "d.parquet"),
                                      "--shots", str(tmp_path / "s.parquet"), "--records", str(tmp_path),
                                      "--out-dir", str(tmp_path)])
    with pytest.raises(RuntimeError, match="read"):
        E.main()
    assert seen == [tmp_path / "d.parquet", tmp_path / "s.parquet", tmp_path / "eda_standings.parquet"]


def test_other_inputs_never_take_the_published_names(monkeypatch, tmp_path):
    monkeypatch.setattr(E.pd, "read_parquet", lambda *a, **k: pytest.fail("read before refusing"))
    monkeypatch.setattr(sys, "argv", ["eda_second_ball.py", "--records", str(tmp_path)])
    with pytest.raises(SystemExit, match="not the grant's"):
        E.main()


def test_a_run_off_the_recipe_s_precision_needs_out_dir(monkeypatch):
    """The team table and the position record are what report_figures reads, so a run at fewer
    draws than the recipe may not write them into data/derived, and refuses before reading."""
    monkeypatch.setattr(E.pd, "read_parquet", lambda *a, **k: pytest.fail("read before refusing"))
    monkeypatch.setattr(sys, "argv", ["eda_second_ball.py", "--boot", "100"])
    with pytest.raises(SystemExit, match="may not write the published"):
        E.main()
