"""Tests for loader_sb — Hudl Statsbomb access (events + 360, sample + Big-5).

The cache tests are the important ones. A parquet round-trip turns the JSON's list
containers into numpy.ndarray, and every container filter in this repo is
`isinstance(v, list)`, so the failure mode is ZERO rows with no exception, not a
crash. Measured on 4004239 without the type restore: 3095 freeze-frames -> 0.
"""
import pandas as pd
import pytest

import loader_sb as L
from conftest import requires_sb


@requires_sb
def test_list_matches():
    """The Serie A sample lists all 30 tracked matches with team columns."""
    m = L.list_matches(L.SERIE_A_SAMPLE)
    assert len(m) == 30
    assert {"match_id", "home_team", "away_team"} <= set(m.columns)


@requires_sb
def test_list_matches_default_is_big5():
    """No-arg default is the full Big-5 grant, not a single sample."""
    m = L.list_matches()
    assert len(m) > len(L.list_matches(L.SERIE_A_SAMPLE))
    assert set(m["league"]) == set(L.BIG5_LEAGUES)


@requires_sb
def test_load_events(sb_events):
    """The event stream has the dotted columns the grammar reasons over."""
    assert len(sb_events) > 0
    for col in ("id", "type.name", "team.name", "possession", "timestamp", "pass.type.name"):
        assert col in sb_events.columns


@requires_sb
def test_column_pushdown_is_the_same_shape_cached_or_raw(sb_mid):
    """The same call must return the same columns whichever path serves it.

    dump_schema runs with the cache off; every other caller runs with it on. If the
    raw path ignored the pushdown, those two would silently disagree on shape."""
    cols = ["id", "type.name", "team.name"]
    cached = L.load_events(sb_mid, columns=cols, use_cache=True)
    raw = L.load_events(sb_mid, columns=cols, use_cache=False)
    assert list(cached.columns) == list(raw.columns) == cols
    assert len(cached) == len(raw)


@requires_sb
def test_load_360(sb_mid):
    """360 frames carry event_uuid + a non-empty freeze_frame of tagged players."""
    fz = L.load_360(sb_mid)
    assert len(fz) > 0
    assert {"event_uuid", "freeze_frame"} <= set(fz.columns)
    ff = fz.iloc[0]["freeze_frame"]
    assert isinstance(ff, list) and len(ff) > 0
    assert {"teammate", "keeper", "location"} <= set(ff[0].keys())


# ── The parquet cache, and the silent-zero it has to survive ────────────
@requires_sb
def test_cache_roundtrip_preserves_freeze_frame_lists(sb_mid):
    """The cached path must find exactly as many freeze-frames as the raw JSON.

    `assert n_raw > 0` comes FIRST on purpose: without it a regression that zeroed
    both sides would sail through on `n_cached == n_raw`."""
    raw = L.load_events(sb_mid, use_cache=False)
    L.clear_memo()
    cached = L.load_events(sb_mid, use_cache=True)

    pred = lambda v: isinstance(v, list) and len(v) > 0           # the production predicate, verbatim
    n_raw = int(raw["freeze_frame"].apply(pred).sum())
    assert n_raw > 0                                             # else this guard is vacuous
    assert int(cached["freeze_frame"].apply(pred).sum()) == n_raw

    ff = cached.loc[cached["freeze_frame"].apply(pred), "freeze_frame"].iloc[0]
    assert type(ff) is list                                      # `type is`: an ndarray is not a list subclass
    assert len(L.load_360(sb_mid)) == n_raw


@requires_sb
def test_cached_and_raw_events_agree(sb_mid):
    """Column set, order, row count, dtype and every scalar VALUE survive the round-trip.

    Nulls are compared as a mask rather than by value. Inside an object column the
    JSON path leaves None where parquet leaves nan, and pandas is about to start
    calling those unequal, but nothing in the pipeline can tell them apart
    (isinstance(v, str) is False for both, pd.notna catches both), so the mask is the
    honest comparison and the dtype check is what would catch a real drift."""
    raw = L.load_events(sb_mid, use_cache=False)
    L.clear_memo()
    cached = L.load_events(sb_mid, use_cache=True)
    assert list(raw.columns) == list(cached.columns) and len(raw) == len(cached)
    for c in (c for c in raw.columns if c not in L._container_cols(raw)):
        assert raw[c].dtype == cached[c].dtype, c
        assert raw[c].isna().equals(cached[c].isna()), c
        m = raw[c].notna()
        pd.testing.assert_series_equal(raw.loc[m, c], cached.loc[m, c])


@requires_sb
def test_columns_pushdown_never_poisons_the_memo(sb_mid):
    """A 3-column load_360 must not leave a 3-column frame under the match key.

    If it did, the next load_events would return it, CornerPhaseGrammar.corners()
    would find no 'pass.type.name' and the build would report ZERO corners for the
    match, silently: corners() returns an empty frame rather than raising."""
    L.clear_memo()
    L.load_360(sb_mid)
    assert L.load_events(sb_mid).shape[1] > 200


@requires_sb
def test_pushdown_returns_the_same_values_as_the_full_read(sb_mid):
    """A restricted read must equal the full read sliced to the same columns.

    Shape alone is not enough: a pushdown that saves time and moves one value is a
    defect, not an optimisation. Containers are checked by type and length because
    `_restore_containers` only converts the TOP level, so their nested payloads stay
    ndarray on both sides and element-wise equality would compare arrays."""
    cols = ["id", "type.name", "team.name", "minute", "second", "freeze_frame"]
    full = L.load_events(sb_mid)
    got = L.load_events(sb_mid, columns=cols)
    want = [c for c in cols if c in full.columns]
    assert list(got.columns) == want
    cont = set(L._container_cols(full))
    assert "freeze_frame" in cont                                # else the container half is vacuous
    for c in want:
        assert got[c].dtype == full[c].dtype, c
        assert got[c].isna().equals(full[c].isna()), c
        m = full[c].notna()
        assert m.sum() > 0, c                                    # a column of all-NaN proves nothing
        if c in cont:
            assert got.loc[m, c].map(type).eq(list).all(), c
            assert got.loc[m, c].map(len).equals(full.loc[m, c].map(len)), c
        else:
            pd.testing.assert_series_equal(got.loc[m, c], full.loc[m, c], check_exact=True)


@requires_sb
def test_pushdown_drops_names_absent_from_this_match(sb_mid):
    """The documented contract: the schema is sparse and varies per match, so a caller
    may ask for the full set on any match."""
    df = L.load_events(sb_mid, columns=["id", "type.name", "no.such.column"])
    assert list(df.columns) == ["id", "type.name"]


@requires_sb
def test_rebuild_cache_writes_what_it_reports_and_then_stops(tmp_path, monkeypatch, sb_mid, mid):
    """`rebuild_cache` is the only parallel entry point here, and nothing else in this
    file exercises it.

    The threading backend is deliberate: joblib's process backend re-imports the module
    in each worker, so the redirected EVENTS_CACHE would not reach them and the test
    would write into the real cache instead of tmp_path."""
    from joblib import parallel_config
    monkeypatch.setattr(L, "EVENTS_CACHE", tmp_path)
    mids = sorted({sb_mid, mid})
    with parallel_config(backend="threading"):
        assert L.rebuild_cache(mids, n_jobs=2, force=True) == len(mids)
    assert sorted(p.name for p in tmp_path.glob("*.parquet")) == [f"{m}.parquet" for m in mids]
    assert L.rebuild_cache(mids, n_jobs=1) == 0                  # what it just wrote is fresh


@requires_sb
def test_assert_list_containers_rejects_an_ndarray():
    """The loader boundary guard fires on the exact type parquet hands back."""
    import numpy as np
    bad = pd.DataFrame({"freeze_frame": [np.array([1.0, 2.0]), None]})
    with pytest.raises(TypeError, match="not list"):
        L._assert_list_containers(bad, ["freeze_frame"])
    L._assert_list_containers(pd.DataFrame({"freeze_frame": [[1.0, 2.0], None]}), ["freeze_frame"])


@requires_sb
@pytest.mark.parametrize("src", [None, "serie_a_sample", "premier_league"])
def test_catalog_matches_the_uncached_scan(src):
    """Cached and uncached catalogs are byte-identical, order included.

    Every driver slices this frame with --limit N. If the cached order ever drifted,
    --limit 20 would quietly describe a different 20 matches than the paper does."""
    pd.testing.assert_frame_equal(L.list_matches(src, use_cache=False),
                                  L.list_matches(src, use_cache=True))


@requires_sb
def test_build_catalog_force_reproduces_the_cached_catalog(tmp_path, monkeypatch):
    """The force path is what a changed tree triggers, and nothing else exercises it.

    CATALOG_CACHE is redirected so the rescan writes beside the test rather than
    rewriting the real catalog under data/derived/."""
    ref = L.build_catalog()
    monkeypatch.setattr(L, "CATALOG_CACHE", tmp_path / "catalog.parquet")
    forced = L.build_catalog(force=True)
    assert len(forced) > 0
    pd.testing.assert_frame_equal(ref.reset_index(drop=True), forced.reset_index(drop=True))
    assert (tmp_path / "catalog.parquet").exists()


@requires_sb
def test_catalog_invalidates_on_a_changed_tree():
    """The fingerprint covers size and mtime of every match file."""
    fp = L._tree_fingerprint()
    assert fp == L._tree_fingerprint()                           # stable when nothing moved
    assert len(fp) == 32                                         # blake2b-128 hex
