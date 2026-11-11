"""Tests for run_refit_draws: the draws it keeps must be the published estimator's, and a reduced
run must never reach the grant's record.

Both refuse before any model is fitted, so neither needs the export.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "pipeline"))

import run_refit_draws as R                                           # noqa: E402


def test_a_reduced_run_may_not_write_the_grants_record():
    proc = subprocess.run([sys.executable, "pipeline/run_refit_draws.py", "--limit", "5"],
                          cwd=_REPO, capture_output=True, text=True, timeout=300)
    assert proc.returncode != 0
    assert "needs --out" in proc.stderr
    assert "REFIT DRAWS" not in proc.stdout


def test_the_draws_cover_the_paper_s_populations(monkeypatch, tmp_path):
    """Corners and flank free kicks from x >= 85, each and pooled, read off the one table both are
    validated on (run_validation --by-type), and their two sufficiency grants."""
    import pandas as pd
    pd.DataFrame({"match_id": [1, 1, 2, 2], "set_piece_type": ["corner", "free_kick_flank"] * 2}
                 ).to_parquet(tmp_path / R._TWO_POP_TABLE)
    monkeypatch.setattr(R, "_DER", tmp_path)
    frames = R.validation_frames(None)
    assert set(frames) == {"corner", "free_kick_flank", "pooled"}
    assert len(frames["pooled"]) == 4 and set(frames["corner"]["set_piece_type"]) == {"corner"}
    assert set(R._SUFF_TAGS) == {"corner", "free_kick_flank"}


def test_a_limited_run_reads_the_table_the_smoke_built(monkeypatch, tmp_path):
    """--limit N reads the N-match two-population table a smoke run wrote, never the grant's."""
    import pandas as pd
    stem = R._TWO_POP_TABLE.removesuffix(".parquet")
    pd.DataFrame({"match_id": [1, 2], "set_piece_type": ["corner", "free_kick_flank"]}
                 ).to_parquet(tmp_path / f"{stem}_limit2.parquet")
    monkeypatch.setattr(R, "_DER", tmp_path)
    assert len(R.validation_frames(2)["pooled"]) == 2


def test_the_check_reads_the_records_of_the_pass_it_draws_for(tmp_path):
    """The pass under partition F checks its draws against its own validation record in zones_F,
    not against the published one of partition C, whose points it would never reproduce."""
    import pandas as pd
    pd.DataFrame({"population": ["corner", "pooled"], "block": ["refit"] * 2}
                 ).to_parquet(tmp_path / "validation.parquet")
    assert list(R._published_validation("corner", tmp_path)["population"]) == ["corner"]


def test_check_stops_on_a_contrast_that_is_not_the_published_one():
    """Draws that do not reproduce the published point or percentiles belong to another
    estimator, and summarising them would publish its shape under this one's name."""
    draws = np.linspace(-1.0, 1.0, 401)
    pct = tuple(np.percentile(draws, [2.5, 97.5]))
    R._check("reproduces", 0.1, draws, 0.1, pct)
    R._check("no percentiles on record", 0.1, draws, 0.1, None)
    with pytest.raises(SystemExit, match="point"):
        R._check("moved point", 0.1 + 1e-6, draws, 0.1, pct)
    with pytest.raises(SystemExit, match="percentiles"):
        R._check("moved bound", 0.1, draws, 0.1, (pct[0] + 1e-6, pct[1]))
