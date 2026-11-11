"""Tests for pipeline/run_q2 — where the race and the situation it joins are read from."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import run_q2 as q2


def test_both_halves_of_the_join_come_from_the_tables_folder(monkeypatch, tmp_path):
    """The smoke joins the race and the situation it built over the tracked sample, not the grant's."""
    seen = {}

    def join(tag="", der=None):
        seen["der"] = der
        raise RuntimeError("read")

    monkeypatch.setattr(q2, "load_join", join)
    monkeypatch.setattr(sys, "argv", ["run_q2.py", "--tables", str(tmp_path),
                                      "--out", str(tmp_path / "q2.parquet")])
    with pytest.raises(RuntimeError, match="read"):
        q2.main()
    assert seen["der"] == tmp_path


def test_tables_outside_data_derived_never_take_the_published_name(monkeypatch, tmp_path):
    """q2_overlap.parquet is the grant's record: another folder's tables write it only under --out."""
    monkeypatch.setattr(q2, "load_join", lambda *a, **k: pytest.fail("read before refusing"))
    monkeypatch.setattr(sys, "argv", ["run_q2.py", "--tables", str(tmp_path)])
    with pytest.raises(SystemExit, match="not the published ones"):
        q2.main()
