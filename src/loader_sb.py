"""loader_sb — Hudl Statsbomb events + 360 access (local granted data, no fetch).

Two local sources, same flat dotted-key schema (Hudl flattens it on export, so no
json_normalize): the 30-match Serie A tracked sample
(`data/serie_a_sample/events_360/`) and the five Big-5 2025/26 competition-seasons
(`data/big5_2025_26/<league>/events/`), Serie A nested inside both (the 30 sample
matches are a subset of the 380-match Serie A season). 360 (`freeze_frame`) is
embedded per event, not a separate endpoint: `load_360` is a filtered view over
`load_events`.

Backed by a per-match parquet cache under `data/derived/events/`. Measured on match
4004239 (median of 35 interleaved reps, warm page cache): a full JSON parse is 179 ms
against 69 ms through this loader, and a pushdown of only the 43 columns the corner
table build reads takes 36 ms, 5x less; the build itself reads the full frame. The
whole grant is 1.59 GB cached against 27 GB of JSON.

Pushdown cost is set by WHICH columns, not how many: 18 scalar columns cost 8.5 ms,
while `_FF_COLS` is three columns and 31 ms because one of them is `freeze_frame`.
Across a sweep of the whole grant (20 workers) reading only what the pipeline needs
is worth 1.8x. The ceiling is that modest because the container restore below runs
on both paths.

TWO TRAPS the cache has to survive:

  (a) the raw frame CANNOT be written. StatsBomb uses `{}` as the not-applicable
      sentinel on 14 container columns, and arrow refuses a column mixing dict and
      list: `ArrowInvalid: cannot mix list and non-list, non-null values` on
      related_events. So the cache stores a normalised frame ({} / [] -> None), and
      the loader restores list types on read; an empty container comes back as None.
      Verified lossless: across 20
      matches spanning all five leagues there is not one non-empty dict value.

  (b) on re-read pyarrow hands back numpy.ndarray where the JSON had list, and
      every container filter in this repo is `isinstance(v, list)`, which is False
      for an ndarray. Measured: 3095 freeze-frames became **0, with no exception
      raised**. Three defences, because a silent zero is the worst failure mode
      here: restore the type on read, assert it at the loader boundary with
      `type(v) is list`, and a regression test that first checks the raw side is
      non-empty so a 0 == 0 comparison cannot pass.

Set HUDL_NO_CACHE=1 (or pass use_cache=False) to force the raw JSON path.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

SERIE_A_SAMPLE = "serie_a_sample"
BIG5_LEAGUES = ("bundesliga", "la_liga", "ligue_1", "premier_league", "serie_a")
_SOURCES = (SERIE_A_SAMPLE,) + BIG5_LEAGUES

CACHE_DIR = DATA_DIR / "derived"
EVENTS_CACHE = CACHE_DIR / "events"
CATALOG_CACHE = CACHE_DIR / "catalog.parquet"
_CACHE_VERSION = 1                     # bump to invalidate every cached match at once
USE_CACHE = os.environ.get("HUDL_NO_CACHE") != "1"

_FF_COLS = ("id", "visible_area", "freeze_frame")     # the load_360 pushdown set
_MEMO_MAX = 2                          # full frames held per process; one costs ~80 MB resident


def _events_dir(source: str) -> Path:
    """Per-match events folder for a source name (a Big-5 league or the sample)."""
    if source == SERIE_A_SAMPLE:
        return DATA_DIR / "serie_a_sample" / "events_360"
    return DATA_DIR / "big5_2025_26" / source / "events"


def _match_id_of(path: Path) -> int:
    """Match id from a filename: bare `<id>.json` (big5) or `<id>_home_vs_away_date.json` (sample)."""
    return int(path.stem.split("_", 1)[0])


_INDEX: dict[int, Path] = {}


def _build_index() -> None:
    """Scan every source folder once (filenames only) and cache match_id -> path.

    The 30 sample matches also ship inside the Serie A season folder, so both copies
    claim the same match_id and the last source scanned (serie_a) wins: every stage reads
    the Big-5 copy. The two are different builds, not one file twice. They differ on
    possession numbering and play patterns in 29 of the 30 matches, on the provider's model
    outputs, and in one match on two events
    (data/derived/review_2026-09-23/data_provider_agent/c_duplicate_copies.log)."""
    if _INDEX:
        return
    for source in _SOURCES:
        d = _events_dir(source)
        if not d.exists():
            continue
        for p in d.glob("*.json"):
            _INDEX[_match_id_of(p)] = p


def _first_event(path: Path) -> dict:
    """Parse only the first event of a match file, which is enough for match metadata:
    every event carries the same match-level fields. 1 MB comfortably covers one event
    plus its freeze_frame."""
    with open(path) as f:
        buf = f.read(1 << 20)
    obj, _ = json.JSONDecoder().raw_decode(buf, idx=buf.index("{"))   # skip the array's '[' + whitespace
    return obj


# ── Container-column plumbing (traps a and b) ───────────────────────────
def _container_cols(df: pd.DataFrame) -> list:
    """Object columns holding dict/list values, the ones arrow cannot take raw."""
    return [c for c in df.columns
            if df[c].dtype == object and df[c].map(lambda v: isinstance(v, (dict, list))).any()]


def _normalise_for_arrow(df: pd.DataFrame) -> tuple:
    """`{}` / `[]` -> None on every container column. Returns (frame, container_cols)."""
    cols = _container_cols(df)
    out = df.copy()
    for c in cols:
        out[c] = out[c].map(lambda v: None if isinstance(v, (dict, list)) and len(v) == 0 else v)
    return out, cols


def _restore_containers(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """ndarray -> list on `cols`, undoing trap (b).

    Only the TOP level is converted. A nested payload (freeze_frame[i]["location"])
    stays an ndarray, deliberately: a deep restore costs 9x more (15 ms against 142 ms
    on match 4004239) and every consumer reads those duck-typed: `_is_xy` tests
    `hasattr(v, "__len__")` and build_frame does `float(p["location"][0])`."""
    for c in cols:
        df[c] = df[c].map(lambda v: list(v) if hasattr(v, "__len__")
                          and not isinstance(v, (str, dict, list)) else v)
    return df


def _assert_list_containers(df: pd.DataFrame, cols: list) -> None:
    """Loud postcondition at the loader boundary: a container column's first non-null
    value must be a plain list. One value decides for the column, since the restore
    converts all of it or none."""
    for c in cols:
        nn = df[c][df[c].notna()]
        if len(nn) and type(nn.iloc[0]) is not list:
            raise TypeError(
                f"loader_sb: '{c}' came back as {type(nn.iloc[0]).__name__}, not list. "
                f"Every `isinstance(v, list)` filter downstream (load_360, audit_360, "
                f"audit_tracking) would silently return ZERO rows. Delete {EVENTS_CACHE} "
                f"and rebuild.")


def _cache_path(match_id: int) -> Path:
    return EVENTS_CACHE / f"{match_id}.parquet"


def _cache_meta(path: Path) -> dict:
    md = pq.read_schema(path).metadata or {}
    return {k.decode(): v.decode() for k, v in md.items() if k.decode().startswith("hudl.")}


def _cache_is_fresh(match_id: int) -> bool:
    """Cached iff it exists, its version matches, and the source file is untouched."""
    p = _cache_path(match_id)
    src = _INDEX.get(match_id)
    if not p.exists() or src is None:
        return False
    try:
        m = _cache_meta(p)
    except Exception:
        return False
    st = src.stat()
    return (m.get("hudl.cache_version") == str(_CACHE_VERSION)
            and m.get("hudl.source_mtime_ns") == str(st.st_mtime_ns)
            and m.get("hudl.source_size") == str(st.st_size))


def _read_json(match_id: int) -> pd.DataFrame:
    with open(_INDEX[match_id]) as f:
        return pd.DataFrame(json.load(f))


def build_events_cache(match_id: int) -> Path:
    """Parse the JSON once, normalise, write atomically.

    Writes to a pid-suffixed temp then os.replace, so a killed run never leaves a
    half-written file that reads back as complete."""
    _build_index()
    src = _INDEX[match_id]
    df = _read_json(match_id)
    norm, cols = _normalise_for_arrow(df)
    tbl = pa.Table.from_pandas(norm, preserve_index=False)
    st = src.stat()
    meta = {b"hudl.cache_version": str(_CACHE_VERSION).encode(),
            b"hudl.match_id": str(match_id).encode(),
            b"hudl.source_mtime_ns": str(st.st_mtime_ns).encode(),
            b"hudl.source_size": str(st.st_size).encode(),
            b"hudl.container_cols": json.dumps(cols).encode(),
            b"hudl.n_rows": str(len(df)).encode()}
    tbl = tbl.replace_schema_metadata({**(tbl.schema.metadata or {}), **meta})
    EVENTS_CACHE.mkdir(parents=True, exist_ok=True)
    out = _cache_path(match_id)
    tmp = out.with_suffix(f".tmp.{os.getpid()}")
    pq.write_table(tbl, tmp, compression="zstd", compression_level=3)
    os.replace(tmp, out)
    return out


_MEMO: "OrderedDict[int, pd.DataFrame]" = OrderedDict()


def _memo_put(match_id: int, df: pd.DataFrame) -> None:
    """Hold a FULL frame only. A `columns=` read is never memoised: if load_360's
    3-column slice landed under the match key, the next load_events would return it,
    CornerPhaseGrammar.corners() would find no 'pass.type.name' and the build would
    report ZERO corners for that match without raising."""
    _MEMO[match_id] = df
    _MEMO.move_to_end(match_id)
    while len(_MEMO) > _MEMO_MAX:
        _MEMO.popitem(last=False)


def clear_memo() -> None:
    """Drop every memoised frame, so the next load_events reads the cache again."""
    _MEMO.clear()


def load_events(match_id: int, columns: list | None = None, use_cache: bool | None = None) -> pd.DataFrame:
    """Flat event stream for a match (dotted columns: type.name, pass.type.name, ...,
    freeze_frame/visible_area embedded).

    Args:
        match_id: StatsBomb match id.
        columns: parquet column pushdown. Names absent from this match's sparse
            union schema are dropped, so a caller may ask for the full s01 set on any
            match. A restricted read is never memoised.
        use_cache: None follows the module USE_CACHE. False forces the raw JSON,
            which dump_schema needs: through the cache it would document the cache's
            types rather than the vendor export's.
    """
    _build_index()
    use = USE_CACHE if use_cache is None else use_cache
    if not use:
        # the raw path must honour the pushdown too, and intersect the same way the
        # parquet branch does: otherwise HUDL_NO_CACHE=1 silently returns a
        # differently-shaped frame for the identical call.
        raw = _read_json(match_id)
        return raw if columns is None else raw[[c for c in columns if c in raw.columns]]

    if columns is None and match_id in _MEMO:
        _MEMO.move_to_end(match_id)
        return _MEMO[match_id].copy(deep=False)
    if not _cache_is_fresh(match_id):
        build_events_cache(match_id)

    p = _cache_path(match_id)
    cont = json.loads(_cache_meta(p).get("hudl.container_cols", "[]"))
    if columns is not None:
        names = set(pq.read_schema(p).names)
        df = pq.read_table(p, columns=[c for c in columns if c in names]).to_pandas()
        restored = [c for c in cont if c in df.columns]
        df = _restore_containers(df, restored)
        # load_360's own read is exactly this branch (a 3-column pushdown), and it is
        # named in the error message below on purpose: skipping the assert here would
        # leave the one caller most exposed to trap (b) with no loud guard at all.
        _assert_list_containers(df, restored)
        return df

    df = _restore_containers(pq.read_table(p).to_pandas(), cont)
    _assert_list_containers(df, cont)
    _memo_put(match_id, df)
    return df.copy(deep=False)


def load_360(match_id: int, use_cache: bool | None = None) -> pd.DataFrame:
    """360 freeze-frames view over load_events: one row per event that carries one.

    No second parse. If the full frame is already memoised this slices it for free;
    otherwise it is the 3-column pushdown this module's header times, rather than a re-read
    of everything.

    An absent freeze_frame/visible_area arrives as StatsBomb's `{}` sentinel on the raw
    path and as None through the cache; the filter keeps non-empty lists and nothing else."""
    use = USE_CACHE if use_cache is None else use_cache
    if use and match_id not in _MEMO:
        events = load_events(match_id, columns=list(_FF_COLS), use_cache=True)
    else:
        events = load_events(match_id, use_cache=use_cache)
    has_ff = events["freeze_frame"].apply(lambda v: isinstance(v, list) and len(v) > 0)
    fz = events.loc[has_ff, ["id", "visible_area", "freeze_frame"]].copy()
    return fz.rename(columns={"id": "event_uuid"})


def rebuild_cache(match_ids: list | None = None, n_jobs: int = 1, force: bool = False) -> int:
    """Warm the events cache. Returns how many matches were (re)written."""
    from joblib import Parallel, delayed
    _build_index()
    mids = list(_INDEX) if match_ids is None else list(match_ids)
    todo = [m for m in mids if force or not _cache_is_fresh(m)]
    if todo:
        Parallel(n_jobs=n_jobs, verbose=5)(delayed(build_events_cache)(m) for m in todo)
    return len(todo)


# ── Match catalog ───────────────────────────────────────────────────────
_CATALOG_VERSION = 1


def _tree_fingerprint() -> str:
    """blake2b-128 over (source, name, size, mtime_ns) of every match file.

    os.scandir only, no file is opened: 1781 entries in single-digit milliseconds, so
    the freshness check costs nothing and a full rebuild (1.5 s) is cheap enough that
    partial invalidation is not worth the risk of getting it wrong."""
    h = hashlib.blake2b(digest_size=16)
    for src in _SOURCES:
        d = _events_dir(src)
        if not d.exists():
            continue
        for e in sorted(os.scandir(d), key=lambda e: e.name):
            if e.name.endswith(".json"):
                st = e.stat()
                h.update(f"{src}/{e.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()


def build_catalog(force: bool = False) -> pd.DataFrame:
    """Parse the first event of every match once and cache the result.

    `_scan_ord` is the position in the canonical source-then-sorted-filename scan and
    it is load-bearing. list_matches ends with a non-stable sort_values("match_date")
    and every driver slices the result with --limit N, so if the cached row order
    drifted from the scan order, --limit 20 would silently select a DIFFERENT 20
    matches and every smoke-test figure would describe another substrate."""
    if not force and CATALOG_CACHE.exists():
        try:
            m = _cache_meta(CATALOG_CACHE)
            if (m.get("hudl.catalog_version") == str(_CATALOG_VERSION)
                    and m.get("hudl.tree_fingerprint") == _tree_fingerprint()):
                return pd.read_parquet(CATALOG_CACHE)
        except Exception:
            pass

    rows = []
    for src in _SOURCES:
        d = _events_dir(src)
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            e = _first_event(p)
            rows.append({"_scan_ord": len(rows), "match_id": _match_id_of(p),
                         "match_date": e.get("match_date"),
                         "home_team": e.get("home_team.home_team_name"),
                         "away_team": e.get("away_team.away_team_name"),
                         "league": src})
    cat = pd.DataFrame(rows)
    tbl = pa.Table.from_pandas(cat, preserve_index=False)
    tbl = tbl.replace_schema_metadata({
        **(tbl.schema.metadata or {}),
        b"hudl.catalog_version": str(_CATALOG_VERSION).encode(),
        b"hudl.tree_fingerprint": _tree_fingerprint().encode(),
        b"hudl.n_files": str(len(cat)).encode()})
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CATALOG_CACHE.with_suffix(f".tmp.{os.getpid()}")
    pq.write_table(tbl, tmp, compression="zstd", compression_level=3)
    os.replace(tmp, CATALOG_CACHE)
    return cat


def list_matches(source: str | None = None, use_cache: bool | None = None) -> pd.DataFrame:
    """Match catalog for a source (default: all five Big-5 leagues).

    `source` is one of BIG5_LEAGUES or SERIE_A_SAMPLE. Sorted by date so slicing the
    first N matches is deterministic (used by the pipeline drivers' --limit).
    Byte-identical to the uncached scan, and a test pins it because the row order is
    what --limit slices."""
    sources = BIG5_LEAGUES if source is None else (source,)
    use = USE_CACHE if use_cache is None else use_cache
    cols = ["match_id", "match_date", "home_team", "away_team", "league"]

    if use:
        cat = build_catalog()
        df = (cat[cat["league"].isin(sources)]
              .sort_values("_scan_ord", kind="stable")[cols])   # restore the scan order first
    else:
        rows = []
        for src in sources:
            for p in sorted(_events_dir(src).glob("*.json")):
                e = _first_event(p)
                rows.append({"match_id": _match_id_of(p), "match_date": e.get("match_date"),
                             "home_team": e.get("home_team.home_team_name"),
                             "away_team": e.get("away_team.away_team_name"), "league": src})
        df = pd.DataFrame(rows)
    return df.sort_values("match_date").reset_index(drop=True)
