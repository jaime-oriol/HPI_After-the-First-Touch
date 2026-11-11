"""Tests for pipeline/eda_throw_ins — which probe record the long-throw grain reads its cell from."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import eda_throw_ins as T


def test_a_limited_run_reads_the_cell_its_own_probe_adopted(monkeypatch, tmp_path):
    """--limit N reads the record `probe_throw_ins --limit N` wrote over the same first N matches,
    never the grant's, so the two counts can be held equal on a smoke run too."""
    seen = []

    def read(path, *a, **k):
        seen.append(Path(path))
        raise RuntimeError("read")

    monkeypatch.setattr(T.pd, "read_parquet", read)
    monkeypatch.setattr(T, "_DER", tmp_path)
    with pytest.raises(RuntimeError, match="read"):
        T.adopted_cell(150)
    assert seen == [tmp_path / "throw_in_probe_limit150.parquet"]
