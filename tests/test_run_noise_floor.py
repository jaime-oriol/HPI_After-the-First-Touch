"""run_noise_floor: the record names each block with the floor of its own width, the driver reads the corner rows and
writes the long schema, and a run that is not the grant's may not write the published record."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))
sys.path.insert(0, str(_REPO / "tests"))

import run_noise_floor as rnf                                       # noqa: E402
from s03_value import _BLOCKS                                      # noqa: E402
from test_s03_value import _synth                                  # noqa: E402


def _floors(widths: list) -> pd.DataFrame:
    return pd.DataFrame({"width": widths, "mean": [0.001 * w for w in widths], "sd": 0.0002, "p95": [0.002 * w for w in widths],
                         "max": [0.003 * w for w in widths], "n_draws": 30})


def test_each_block_is_named_with_the_floor_of_its_own_width():
    t = _synth()
    widths = sorted({sum(c in t.columns for c in cols) for cols in _BLOCKS.values()} - {0})
    rows = rnf.floor_rows(t, _floors(widths), 30, 3)
    assert set(rows.loc[rows["block"] == "floor", "n_cols"]) == set(widths)
    groups = rows[rows["block"] == "group"].set_index("key")
    assert set(groups.index) == {g for g, cols in _BLOCKS.items() if any(c in t.columns for c in cols)}
    for g, r in groups.iterrows():
        assert r["n_cols"] == sum(c in t.columns for c in _BLOCKS[g]) and r["p95"] == 0.002 * r["n_cols"]
    assert (rows["population"] == "corner").all() and (rows["n_draws"] == 30).all() and (rows["n_repeats"] == 3).all()


def test_the_driver_reads_the_corner_rows_and_writes_the_long_schema(monkeypatch, tmp_path):
    table = _synth().assign(set_piece_type="corner")
    table.loc[table.index[::5], "set_piece_type"] = "free_kick_flank"
    src = tmp_path / "corners.parquet"
    table.to_parquet(src)
    seen = {}
    widths = sorted({sum(c in table.columns for c in cols) for cols in _BLOCKS.values()} - {0})
    monkeypatch.setattr(rnf.CompetingOutcomesModel, "tune_cv", lambda self, t, **k: SimpleNamespace(best_params={"C": 1.0}))
    monkeypatch.setattr(rnf.CompetingOutcomesModel, "apply_best_params", lambda self, t: None)
    monkeypatch.setattr(rnf.CompetingOutcomesModel, "noise_floor",
                        lambda self, t, **k: seen.update(rows=len(t), **k) or _floors(widths))
    monkeypatch.setattr(sys, "argv", ["run_noise_floor.py", "--table", str(src), "--draws", "2", "--repeats", "1",
                                      "--jobs", "1", "--out-dir", str(tmp_path)])
    rnf.main()
    assert seen["rows"] == int((table["set_piece_type"] == "corner").sum()) and seen["n_draws"] == 2 and seen["n_repeats"] == 1
    assert set(pd.read_parquet(tmp_path / "noise_floor.parquet")["block"]) == {"floor", "group"}


def test_a_run_that_is_not_the_grant_s_may_not_write_the_published_record(tmp_path):
    other = tmp_path / "corners.parquet"
    pd.DataFrame({"set_piece_type": ["corner"]}).to_parquet(other)
    for extra in (["--table", str(other)], ["--draws", "3"]):
        proc = subprocess.run([sys.executable, "pipeline/run_noise_floor.py", *extra], cwd=_REPO, capture_output=True,
                              text=True, timeout=300)
        assert proc.returncode != 0 and "may not write the published" in proc.stderr
