"""The parallel schema/value dumps must equal the serial ones, exactly.

Both drivers reduce per-unit counters returned by workers. Summing counts is
associative, but two things are not: the first-seen example value each column carries,
and the key insertion order that becomes ordinal_position. Both depend on the reduce
running in a fixed order, so a refactor that lets joblib return out of order would
silently reorder the reference the data-provider doc quotes. This asserts it does not.
"""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from conftest import requires_data, requires_sb

_LIMIT = 4                  # enough matches/files to exercise the merge, cheap enough for CI
_FRAMES = 300               # tracking prefix: this test is about the reduce, not about coverage


def _driver(repo_root: Path, name: str) -> ModuleType:
    """Import a pipeline driver by path, as tests/test_drivers_smoke.py does."""
    path = repo_root / "pipeline" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


@requires_sb
@requires_data
def test_dump_schema_parallel_matches_serial(repo_root):
    ds = _driver(repo_root, "dump_schema")
    mids = ds.L.list_matches(None)["match_id"].tolist()[:_LIMIT]
    paths = sorted(ds.T.DATA_DIR.glob("*.jsonl"))[:_LIMIT]

    pd.testing.assert_frame_equal(ds.events_inventory(mids, 1), ds.events_inventory(mids, 2))
    pd.testing.assert_frame_equal(ds.tracking_inventory(paths, _FRAMES, 1),
                                  ds.tracking_inventory(paths, _FRAMES, 2))


@requires_sb
def test_dump_schema_descends_into_list_columns(repo_root):
    """freeze_frame and friends are opaque to a dotted flatten; their entry schema is
    the whole reason the doc needed a hand-written table."""
    ds = _driver(repo_root, "dump_schema")
    mids = ds.L.list_matches(None)["match_id"].tolist()[:_LIMIT]
    inv = ds.events_inventory(mids, 1)

    entry_tables = set(inv.loc[inv["table_name"].str.endswith("[]"), "table_name"])
    assert "events.freeze_frame[]" in entry_tables
    assert "events.shot.freeze_frame[]" in entry_tables

    top = set(inv.loc[inv["table_name"] == "events.freeze_frame[]", "column_name"])
    shot = set(inv.loc[inv["table_name"] == "events.shot.freeze_frame[]", "column_name"])
    # The trap the doc flags: one keeper-detection rule cannot serve both frames.
    assert "keeper" in top and "keeper" not in shot
    assert "position.name" in shot and "position.name" not in top


@requires_sb
@requires_data
def test_dump_values_parallel_matches_serial(repo_root):
    """All three populations at once: scalar columns, nested list fields, tracking."""
    dv = _driver(repo_root, "dump_values")
    ds = _driver(repo_root, "dump_schema")
    mids = dv.L.list_matches(None)["match_id"].tolist()[:_LIMIT]
    paths = sorted(dv.T.DATA_DIR.glob("*.jsonl"))[:2]

    inv = ds.events_inventory(mids, 1)
    ev = inv[inv["table_name"] == "events"]
    scalars = ev.loc[ev["dtype"] != "list", "column_name"].tolist()
    lists = ev.loc[ev["dtype"] == "list", "column_name"].tolist()

    v1, s1 = dv.build(mids, scalars, lists, paths, 1)
    v2, s2 = dv.build(mids, scalars, lists, paths, 2)
    pd.testing.assert_frame_equal(v1, v2)
    pd.testing.assert_frame_equal(s1, s2)
    assert {"events", "tracking_player"} <= set(s1["table_name"])
    assert any(t.endswith("[]") for t in s1["table_name"])


@requires_sb
def test_pass_type_name_domain_is_wider_than_the_set_piece_five(repo_root):
    """pass.type.name is documented as Corner/Free Kick/Kick Off/Goal Kick/Throw-in, but
    the grant also carries Recovery (the most frequent value) and Interception. It is the
    field SetPieceGrammar filters on, so its real domain is load-bearing."""
    dv = _driver(repo_root, "dump_values")
    mids = dv.L.list_matches(None)["match_id"].tolist()[:_LIMIT]
    _, summary = dv.build(mids, ["pass.type.name"], [], [], 1)

    row = summary.set_index("column_name").loc["pass.type.name"]
    assert row["kind"] == "categorical"
    assert row["cardinality"] > 5


@requires_data
def test_tracking_flush_does_not_change_the_counts(repo_root):
    """Buffering a whole 460MB tracking file costs several GB per worker, so the counters
    are flushed every few thousand frames. Counter addition is associative, so that has to
    be exact rather than approximate, and this pins it."""
    dv = _driver(repo_root, "dump_values")
    path = sorted(dv.T.DATA_DIR.glob("*.jsonl"))[0]
    often = dv._tracking_file_counts(path, flush_every=400)
    once = dv._tracking_file_counts(path, flush_every=10 ** 9)
    assert often.keys() == once.keys()
    for table in often:
        assert often[table].keys() == once[table].keys()
        for field in often[table]:
            assert often[table][field][0] == once[table][field][0]


@requires_sb
def test_dump_values_quantiles_are_exact_on_the_canonicalised_column(repo_root):
    """The counter is the histogram, so the reconstructed order statistics are the real
    ones, not an estimate off a retained sample."""
    dv = _driver(repo_root, "dump_values")
    q = dv._quantiles(dv.Counter({1.0: 10, 2.0: 10, 3.0: 10}))
    assert q["min"] == 1.0 and q["max"] == 3.0 and q["p50"] == 2.0
    assert q["mean"] == 2.0


def test_round_sig_keeps_four_significant_figures(repo_root):
    """Fixed decimals would flatten obv_* and explode ElapsedTime; significant figures
    bound cardinality at every scale, which is what removes the need to truncate."""
    dv = _driver(repo_root, "dump_values")
    got = dv._round_sig(dv.np.array([0.0, 0.0123456, 3456.78, -52.49137, 1e-6]))
    assert got[0] == 0.0
    assert got[1] == pytest.approx(0.01235, rel=1e-9)
    assert got[2] == pytest.approx(3457.0, rel=1e-9)
    assert got[3] == pytest.approx(-52.49, rel=1e-9)
    assert got[4] == pytest.approx(1e-6, rel=1e-9)


def test_round_sig_never_collides_identifiers(repo_root):
    """Ids reach the counter as floats, because pandas promotes an int column the moment
    it holds a null. Rounding those to four significant figures would merge distinct ids,
    a counter bug that reads exactly like a provider defect."""
    dv = _driver(repo_root, "dump_values")
    ids = dv.np.array([442123.0, 442124.0, 3208.0, 23901.0])
    assert dv._round_sig(ids).tolist() == ids.tolist()


# ── Which names a run may write ─────────────────────────────────────────
# columns.tsv, values.tsv and values_summary.tsv are the references data_provider.tex quotes, and
# schema_dump.parquet is dump_values' input. Each driver runs in-process on two matches, with its
# repo root moved, so a wrong name lands in tmp_path and not in the record.
def _two_matches(module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    real = module.L.list_matches
    monkeypatch.setattr(module.L, "list_matches", lambda source=None: real(source).head(2))
    monkeypatch.setattr(module, "_REPO", tmp_path)


@requires_sb
@requires_data
def test_a_capped_tracking_scan_never_lands_on_the_reference(repo_root, tmp_path, monkeypatch):
    """A frame cap reads a prefix of every tracking file, which cannot see a key only a later
    frame emits, so it tags both default names as --limit does."""
    ds = _driver(repo_root, "dump_schema")
    _two_matches(ds, monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["dump_schema.py", "--frames-per-file", "50"])
    ds.main()
    hudl, der = tmp_path / "docs" / "hudl", tmp_path / "data" / "derived"
    assert not (hudl / "columns.tsv").exists() and not (der / "schema_dump.parquet").exists()
    assert (hudl / "columns_frames50.tsv").exists() and (der / "schema_dump_frames50.parquet").exists()


@requires_sb
@requires_data
def test_a_schema_run_with_out_writes_nothing_under_the_record(repo_root, tmp_path, monkeypatch):
    """--out is how the dump runs without touching the record, so its parquet goes beside the TSV."""
    ds = _driver(repo_root, "dump_schema")
    _two_matches(ds, monkeypatch, tmp_path)
    out = tmp_path / "elsewhere" / "cols.tsv"
    monkeypatch.setattr(sys, "argv", ["dump_schema.py", "--limit", "2", "--frames-per-file", "50",
                                      "--out", str(out)])
    ds.main()
    assert out.exists() and out.with_suffix(".parquet").exists()
    assert not (tmp_path / "data").exists()


def _values_setup(repo_root: Path, monkeypatch: pytest.MonkeyPatch,
                  tmp_path: Path) -> tuple[ModuleType, Path]:
    """dump_values over two matches, reading an inventory of the same two matches as its input."""
    ds, dv = _driver(repo_root, "dump_schema"), _driver(repo_root, "dump_values")
    der = tmp_path / "data" / "derived"
    der.mkdir(parents=True)
    mids = ds.L.list_matches(None)["match_id"].tolist()[:2]
    ds.events_inventory(mids, 1).to_parquet(der / "schema_dump.parquet")
    _two_matches(dv, monkeypatch, tmp_path)
    return dv, der


@requires_sb
def test_a_values_scan_without_tracking_never_lands_on_the_reference(repo_root, tmp_path,
                                                                     monkeypatch):
    """values_summary.tsv carries the tracking tables the provider doc quotes, so a scan that skips
    them tags every name it writes."""
    dv, der = _values_setup(repo_root, monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["dump_values.py", "--no-tracking"])
    dv.main()
    hudl = tmp_path / "docs" / "hudl"
    assert not (hudl / "values.tsv").exists() and not (hudl / "values_summary.tsv").exists()
    assert not (der / "value_dump.parquet").exists() and not (der / "value_summary.parquet").exists()
    assert (hudl / "values_no_tracking_summary.tsv").exists()


@requires_sb
def test_a_values_run_with_out_writes_nothing_under_the_record(repo_root, tmp_path, monkeypatch):
    """--out is how the dump runs without touching the record, so both parquets go beside it."""
    dv, der = _values_setup(repo_root, monkeypatch, tmp_path)
    out = tmp_path / "elsewhere" / "vals.tsv"
    monkeypatch.setattr(sys, "argv", ["dump_values.py", "--no-tracking", "--out", str(out)])
    dv.main()
    assert out.exists() and out.with_suffix(".parquet").exists()
    assert out.with_name("vals_summary.parquet").exists()
    assert sorted(p.name for p in der.iterdir()) == ["schema_dump.parquet"]
