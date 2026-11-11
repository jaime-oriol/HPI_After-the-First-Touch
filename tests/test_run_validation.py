"""Tests for run_validation's write guard.

`validation.parquet` is the --by-type record and carries corner, free_kick_flank and pooled together,
so a run that is not --by-type must never reach it, and a missing table must stop rather than rebuild
a different one in silence. Both have to refuse BEFORE any model is fitted.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "pipeline/run_validation.py", *args], cwd=_REPO,
                          capture_output=True, text=True, timeout=300)


def test_a_corner_only_run_may_not_write_the_by_type_record(tmp_path):
    table = tmp_path / "corners.parquet"
    pd.DataFrame({"set_piece_type": ["corner"], "match_id": [1]}).to_parquet(table)
    proc = _run("--table", str(table))
    assert proc.returncode != 0
    assert "may not write the published validation.parquet" in proc.stderr
    assert "BASELINE LADDER" not in proc.stdout


def test_a_missing_table_stops_instead_of_rebuilding(tmp_path):
    proc = _run("--table", str(tmp_path / "absent.parquet"))
    assert proc.returncode != 0
    assert "absent" in proc.stderr


def test_a_by_type_run_off_another_table_may_not_write_the_record(tmp_path):
    """--by-type writes the record report_figures publishes, so only the grant's two-population
    table reaches it; a smaller one carrying both kinds stops before any model is fitted."""
    table = tmp_path / "corners_corner_free_kick_flank.parquet"
    pd.DataFrame({"set_piece_type": ["corner", "free_kick_flank"],
                  "match_id": [1, 2]}).to_parquet(table)
    proc = _run("--table", str(table), "--by-type")
    assert proc.returncode != 0
    assert "may not write the published validation.parquet" in proc.stderr
    assert "BASELINE LADDER" not in proc.stdout


def _main_on_the_grant_table(tmp_path: Path, monkeypatch, *flags: str) -> Path:
    """main() on the paper's two-population table with the derived directory moved and the model
    work stubbed, so what is under test is where it writes. Returns that directory."""
    spec = importlib.util.spec_from_file_location("_rv_guard", _REPO / "pipeline" / "run_validation.py")
    rv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rv)
    der = tmp_path / "derived"
    der.mkdir()
    table = der / "corners_corner_free_kick_flank.parquet"
    pd.DataFrame({"set_piece_type": ["corner", "free_kick_flank"],
                  "match_id": [1, 2]}).to_parquet(table)
    stub = pd.DataFrame({"population": ["stub"], "block": ["stub"]})
    monkeypatch.setattr(rv, "_OUT", der / "validation.parquet")
    monkeypatch.setattr(rv, "report", lambda *a, **k: stub)
    monkeypatch.setattr(rv, "pooling_contrast", lambda *a, **k: stub)
    monkeypatch.setattr(sys, "argv", ["run_validation.py", "--table", str(table), "--by-type",
                                      *flags])
    rv.main()
    return der


def test_the_paper_s_two_populations_write_the_record(tmp_path, monkeypatch):
    """The paper models the corners and the flank free kicks, so their grant table is the one
    --by-type run that reaches validation.parquet, the stage the recipe runs."""
    assert (_main_on_the_grant_table(tmp_path, monkeypatch) / "validation.parquet").exists()


def test_out_dir_writes_the_record_there(tmp_path, monkeypatch):
    """The pass under partition F writes the record under its published name into its own folder,
    and nothing beside the grant's."""
    out = tmp_path / "zones_F"
    der = _main_on_the_grant_table(tmp_path, monkeypatch, "--out-dir", str(out))
    assert [p.name for p in out.iterdir()] == ["validation.parquet"]
    assert not (der / "validation.parquet").exists()


def test_an_untuned_run_never_lands_on_the_record(tmp_path, monkeypatch):
    """--no-tune reports at the constructor defaults, another estimator from the tuned one the
    record publishes, so it writes a tagged name."""
    der = _main_on_the_grant_table(tmp_path, monkeypatch, "--no-tune")
    assert not (der / "validation.parquet").exists()
    assert (der / "validation_untuned.parquet").exists()
