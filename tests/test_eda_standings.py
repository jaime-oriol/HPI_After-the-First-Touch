"""Tests for pipeline/eda_standings — the club-name mapping between the two feeds.

The mapping is proposed by string similarity and proved by points, and the proof is only worth
something if it can fail: a mapping that put one club's points on another's row would ship a wrong
`pos` into `eda_standings.parquet` and every figure below it.

The one tolerance is real. Ligue 1 arrives one fixture short (refdata/README.md names it), so the
two clubs of that fixture sit short of the published table through no fault of the mapping, and a
guard that cannot tell that apart from a genuine mismatch is a guard nobody can leave switched on.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import eda_standings as es


def _table(rows: list) -> pd.DataFrame:
    """A reconciliation frame for one four-club league; 6 games each in a full round-robin."""
    return pd.DataFrame([dict(league="toy", team=t, team_ref=t, pos=i + 1,
                              pts_ref=pr, pts_derived=pd_, played=pl)
                         for i, (t, pr, pd_, pl) in enumerate(rows)])


def test_a_complete_season_that_agrees_is_clean():
    assert es.reconcile(_table([("a", 12, 12, 6), ("b", 9, 9, 6),
                                ("c", 6, 6, 6), ("d", 3, 3, 6)])).empty


def test_a_missing_fixture_excuses_at_most_three_points():
    """The real case: a club short one match may be short up to a win's worth and no more."""
    one_short = [("a", 12, 12, 6), ("b", 9, 9, 6), ("c", 7, 6, 5), ("d", 3, 3, 6)]
    assert es.reconcile(_table(one_short)).empty
    too_far = [("a", 12, 12, 6), ("b", 9, 9, 6), ("c", 11, 6, 5), ("d", 3, 3, 6)]
    assert list(es.reconcile(_table(too_far))["team"]) == ["c"]


def test_a_swapped_mapping_is_caught():
    """Two clubs whose published rows are exchanged. Both played every game, so neither gap has
    an excuse, and this is the failure a check that only printed would let through."""
    swapped = [("a", 9, 12, 6), ("b", 12, 9, 6), ("c", 6, 6, 6), ("d", 3, 3, 6)]
    assert set(es.reconcile(_table(swapped))["team"]) == {"a", "b"}


def test_deriving_more_points_than_published_is_never_excused():
    """A surplus cannot come from a fixture the export is missing, however many are missing."""
    surplus = [("a", 12, 12, 6), ("b", 9, 9, 6), ("c", 6, 9, 3), ("d", 3, 3, 6)]
    assert list(es.reconcile(_table(surplus))["team"]) == ["c"]


def test_the_published_table_is_versioned_not_hand_placed():
    """It lives in the versioned refdata/, so a fresh clone can run this stage."""
    assert es._REF.exists() and es._REF.parts[-2] == "refdata"
    ref = pd.read_csv(es._REF)
    assert len(ref) == 96 and set(ref.columns) == {"league", "pos", "team_ref", "pts"}
    assert ref.groupby("league")["pos"].apply(lambda s: sorted(s) == list(range(1, len(s) + 1))).all()


def test_derive_table_counts_a_draw_for_both_sides():
    m = pd.DataFrame([dict(league="toy", home_team="a", away_team="b", winner="Drawing"),
                      dict(league="toy", home_team="a", away_team="b", winner="a")])
    t = es.derive_table(m).set_index("team")
    assert t.loc["a", "pts"] == 4 and t.loc["b", "pts"] == 1
    assert t.loc["a", "played"] == 2 and t.loc["b", "d"] == 1


@pytest.mark.parametrize("raw,want", [("Athletic Club", "athletic club"),
                                      ("AS Roma", "roma"), ("1. FC Köln", "koln")])
def test_the_name_key_strips_what_differs_between_feeds(raw, want):
    assert es._key(raw) == want


def test_the_matches_flag_names_the_input(monkeypatch, tmp_path):
    """The smoke derives its table from the N-match results it built, not from the grant's."""
    seen = {}

    def read(path, *a, **k):
        seen["path"] = Path(path)
        raise RuntimeError("read")

    monkeypatch.setattr(es.pd, "read_parquet", read)
    monkeypatch.setattr(sys, "argv", ["eda_standings.py", "--matches", str(tmp_path / "m.parquet"),
                                      "--out", str(tmp_path / "s.parquet")])
    with pytest.raises(RuntimeError, match="read"):
        es.main()
    assert seen["path"] == tmp_path / "m.parquet"


def test_other_results_never_take_the_published_name(monkeypatch, tmp_path):
    """eda_standings.parquet is the season's table: results from elsewhere write it only under --out."""
    monkeypatch.setattr(es.pd, "read_parquet", lambda *a, **k: pytest.fail("read before refusing"))
    monkeypatch.setattr(sys, "argv", ["eda_standings.py", "--matches", str(tmp_path / "m.parquet")])
    with pytest.raises(SystemExit, match="pass --out"):
        es.main()
