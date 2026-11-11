"""Tests for pipeline/run_sufficiency — which run may write the record the verdict is read from."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import run_sufficiency as rs

_ZONES_F = Path(__file__).resolve().parents[1] / "data" / "derived" / "zones_F"


def test_a_limited_grant_is_the_one_the_smoke_built(tmp_path):
    """--limit N reads the N-match tables a smoke run wrote, never the grant's cut down: the grant's
    may come from another run, or not exist yet."""
    pd.DataFrame({"set_piece_id": [1, 2, 3], "match_id": [10, 10, 11]}).to_parquet(
        tmp_path / "corners_limit2.parquet")
    pd.DataFrame({"set_piece_id": [1, 2, 3], "won_second_ball": [True, None, False]}).to_parquet(
        tmp_path / "situation_limit2.parquet")
    t = rs.grant_table(2, "", tmp_path)
    assert list(t["set_piece_id"]) == [1, 3] and list(t["target"]) == ["won", "lost"]


def test_the_overlap_joins_the_race_to_its_own_folder_s_grant(tmp_path):
    """The tracked corners are joined to the grant of the folder the race lives in, whatever the
    grant arm reads: a smoke's grant subset holds no tracked match."""
    pd.DataFrame({"set_piece_id": [1, 2], "match_id": [10, 10], "league": ["x", "x"]}).to_parquet(
        tmp_path / "corners.parquet")
    pd.DataFrame({"set_piece_id": [1, 2], "won_second_ball": [True, False]}).to_parquet(
        tmp_path / "situation.parquet")
    pd.DataFrame({"corner_id": [1, 2], "match_id": [10, 10], "actual_att": [True, None],
                  "tau_s": [0.5, 0.5]}).to_parquet(tmp_path / "race_corners.parquet")
    prim, wide = rs.overlap_population("", tmp_path)
    assert list(prim["corner_id"]) == [1, 2] and list(wide["corner_id"]) == [1]


def test_the_grant_arm_and_the_overlap_read_their_own_folders(monkeypatch, tmp_path):
    """--limit N cuts the grant arm to the smoke's N-match tables, and --overlap-tables names the
    folder whose race, grant and 360 set the tracked comparison is built from."""
    seen = {}

    def grant(limit, tag="", der=rs._DER):
        seen["grant"] = (limit, der)
        return pd.DataFrame({"set_piece_id": [1], "match_id": [10], "won": [True], "league": ["x"]})

    def overlap(tag="", der=None):
        seen["overlap"] = der
        raise RuntimeError("read")

    monkeypatch.setattr(rs, "grant_table", grant)
    monkeypatch.setattr(rs, "overlap_population", overlap)
    monkeypatch.setattr(sys, "argv", ["run_sufficiency.py", "--limit", "3", "--overlap-tables",
                                      str(tmp_path), "--out", str(tmp_path / "s.parquet")])
    with pytest.raises(RuntimeError, match="read"):
        rs.main()
    assert seen == {"grant": (3, rs._DER), "overlap": tmp_path}


def test_another_overlap_never_takes_the_published_name(monkeypatch, tmp_path):
    """The record is the grant's: an overlap built from another folder writes it only under --out."""
    monkeypatch.setattr(rs, "grant_table", lambda *a, **k: pytest.fail("read before refusing"))
    monkeypatch.setattr(sys, "argv", ["run_sufficiency.py", "--overlap-tables", str(tmp_path)])
    with pytest.raises(SystemExit, match="not the published ones"):
        rs.main()


@pytest.mark.parametrize("dest", [[], ["--out", str(_ZONES_F / "sufficiency.parquet")]])
@pytest.mark.parametrize("extra", [["--refit-boot", "0"], ["--no-power"], ["--trials", "2"],
                                   ["--seeds", "1"], ["--boot", "200"]])
def test_the_published_record_is_written_only_at_the_recipe_s_precision(monkeypatch, extra, dest):
    """`freeze_frame_beats_the_delivery_baseline` is decided on the refit block, at the strong
    learner's budget, with the power curve beside it, so a run without one of them or at a smaller
    budget may not write sufficiency.parquet, in data/derived or in zones_F where the pass under
    partition F publishes it, and refuses before the grant is read."""
    def work(*a, **k):
        raise AssertionError("the run did the work before refusing")

    monkeypatch.setattr(rs, "grant_table", work)
    monkeypatch.setattr(sys, "argv", ["run_sufficiency.py", *extra, *dest])
    with pytest.raises(SystemExit, match="may not write the published"):
        rs.main()
