"""dump_values — the value domain of every field in the granted Hudl export.

dump_schema.py answers "which fields exist and how often are they populated". This
answers "and what do they actually contain": the closed enumeration of every categorical
and the real range and shape of every numeric, including the proprietary families the
data-provider doc could only describe as inferred (set_piece_phase, obv_*, the
shot-geometry block, the HOPS duel ratings).

    python pipeline/dump_values.py --jobs 16      # the full grant
    python pipeline/dump_values.py --limit 20     # smoke test

Three populations, distinguished by table_name in the output:

  events              the scalar columns, read from the parquet cache
  events.<col>[]      the fields inside the list columns, read from the raw JSON
  tracking_*          the Broadcast X&Y fields, streamed from the raw JSONL

SOURCE, and why it differs per population. The scalar events pass reads the parquet
CACHE: what it measures is the value domain of scalar columns, for which the cache is
faithful (build_events_cache normalises the whole frame without dropping columns), and
the container pushdown skips most of the parquet's weight. The list pass cannot use it
(arrow round-trips a list-of-dict into a shape whose keys are no longer plain Python),
so it reads raw JSON, like dump_schema. Tracking has no cache at all.

CANONICALISATION, and why it is 4 significant figures. Counting raw floats would give a
counter with one key per distinct float; rounding to fixed decimals instead blows up on
wide-range columns (ElapsedTime) while flattening narrow ones (obv_*). Four significant
figures bounds cardinality to tens of thousands whatever the scale, so no truncation is
ever needed, which matters for more than memory: a cardinality cap would drop different
keys depending on how units fall across workers, and the parallel run would stop matching
the serial one. A pure function of the value keeps the merge exactly associative.
Quantiles are therefore exact to 4 significant figures, not estimates.

OUTPUT NAMES. values.tsv and its summary are the references data_provider.tex quotes, so every
flag that narrows the scan (--limit, --no-tracking, --schema) tags the default names, and with
--out the two parquets go beside the TSV: the dump can run without touching the record.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import loader_sb as L                                   # noqa: E402
import loader as T                                       # noqa: E402

_SIG_FIGS = 4
_TOP_N = 50
_QUANTILES = (0.01, 0.25, 0.5, 0.75, 0.99)

# Event-id lists have no value domain: every element is a distinct key, so counting them
# would build a counter with an entry per event that says nothing. Their length is the
# informative part.
_LEN_ONLY = {"related_events"}

# Coordinate lists in this export are [x, y] or [x, y, z]; index them so each axis keeps
# its own range. Anything longer is a variable-length run, where index i means nothing.
_INDEX_MAX_LEN = 3

_TRACK_ENTRY_LISTS = ("homePlayers", "awayPlayers", "balls", "referees")

# Frames between counter flushes in the tracking pass. Bounds peak memory per worker;
# see _tracking_file_counts for why holding a whole file is not an option.
_FLUSH_FRAMES = 5000


def _round_sig(a: np.ndarray) -> np.ndarray:
    """Canonicalise a numeric array: whole numbers exactly, everything else to
    _SIG_FIGS significant figures.

    Whole numbers are passed through because identifiers arrive as floats (pandas
    promotes an int column to float64 the moment it holds a null, so player.id.GK and
    team ids reach here as 442123.0), and rounding those to four significant figures
    would collide distinct ids into one bucket, which reads exactly like a provider defect.
    Counts, ids and second-resolution clocks are all whole, and their natural cardinality is
    already bounded, so nothing is lost by keeping them exact.
    """
    out = np.array(a, dtype=float, copy=True)
    frac = np.isfinite(out) & (out != np.floor(out))
    if frac.any():
        v = out[frac]
        mag = np.floor(np.log10(np.abs(v)))
        factor = np.power(10.0, _SIG_FIGS - 1 - mag)
        out[frac] = np.round(v * factor) / factor
    return out


def _counts_from_values(values: list) -> tuple[Counter, bool]:
    """(value -> count, is_numeric) for a plain Python list of scalars.

    Numeric-ness is decided on the first element, not per value: these lists come from one
    JSON field, and a field that mixes numbers with strings would be a vendor bug worth
    seeing as two separate keys rather than silently merged.
    """
    if not values:
        return Counter(), False
    first = values[0]
    numeric = isinstance(first, (int, float)) and not isinstance(first, bool)
    if numeric:
        arr = np.fromiter((v for v in values), dtype=float, count=len(values))
        vc = pd.Series(_round_sig(arr)).value_counts()
    else:
        vc = pd.Series(values, dtype=object).value_counts()
    return Counter(dict(zip(vc.index.tolist(), vc.to_numpy().tolist(), strict=True))), numeric


def _column_counts(s: pd.Series) -> tuple[Counter, bool]:
    """(value -> count, is_numeric) for one scalar column, nulls dropped.

    Dispatches on dtype rather than per value: a Python-level canonicaliser over every
    cell of the grant is the difference between minutes and hours. An object column holding floats
    (a nullable numeric that arrow handed back as object) is routed to the numeric path
    by inspecting one value, not by trying to convert the whole column blind.
    """
    s = s.dropna()
    if not len(s):
        return Counter(), False

    numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    if not numeric and s.dtype == object:
        first = s.iloc[0]
        if hasattr(first, "item"):
            first = first.item()
        numeric = isinstance(first, (int, float)) and not isinstance(first, bool)
        if numeric:
            s = pd.to_numeric(s, errors="coerce").dropna()
            if not len(s):
                return Counter(), True

    if numeric:
        vc = pd.Series(_round_sig(s.to_numpy(dtype=float))).value_counts()
    else:
        vc = s.value_counts()
    return Counter(dict(zip(vc.index.tolist(), vc.to_numpy().tolist(), strict=True))), numeric


# ───────────────────────── list columns and nested entries ─────────────────────────

def _emit(bucket: defaultdict[str, list], key: str, v: object) -> None:
    """Fan one JSON value out into the scalar buckets it contributes to.

    A dict recurses on dotted keys; a fixed-length numeric list (a [x, y] location) becomes
    one bucket per index, because the interesting thing about a freeze-frame location is the
    coordinate range, not the pair; anything else scalar lands as itself.
    """
    if v is None:
        return
    if isinstance(v, dict):
        for k, sub in v.items():
            _emit(bucket, f"{key}.{k}", sub)
    elif isinstance(v, list):
        bucket[f"{key}.len"].append(len(v))
        if not all(isinstance(e, (int, float)) and not isinstance(e, bool) for e in v):
            return
        if len(v) <= _INDEX_MAX_LEN:
            for i, e in enumerate(v):
                bucket[f"{key}.{i}"].append(e)
        else:
            # A variable-length coordinate run (visible_area's flat polygon) has no
            # stable meaning per index, so its values pool under one name and .len
            # carries the shape.
            bucket[key].extend(v)
    else:
        bucket[key].append(v)


def _list_column_values(ev: pd.DataFrame, columns: list[str]) -> dict[tuple, tuple[Counter, bool]]:
    """Value domain of everything inside the list columns of one match, keyed by
    (parent column, field) so the reduce can name the table without re-parsing the
    field: a dotted split cannot tell `tactics.lineup` from `freeze_frame.location`.

    Buckets are filled as plain Python lists and counted once per match: appending is cheap,
    while Counter.update per value over every freeze-frame entry of the grant is not.
    """
    out = {}
    for col in columns:
        if col not in ev.columns:
            continue
        bucket: defaultdict[str, list] = defaultdict(list)
        for v in ev[col]:
            if not isinstance(v, list) or not v:
                continue
            if col in _LEN_ONLY or not isinstance(v[0], dict):
                # A coordinate list is emitted whole, so [x, y] splits into .0 and .1
                # instead of pooling two different ranges under one name.
                _emit(bucket, col, v)
                continue
            bucket[f"{col}.len"].append(len(v))
            for entry in v:
                _emit(bucket, col, entry)
        for key, values in bucket.items():
            counts, numeric = _counts_from_values(values)
            if counts:
                out[(col, key)] = (counts, numeric)
    return out


# ─────────────────────────────────── workers ───────────────────────────────────

def _match_counts(match_id: int, scalar_cols: list[str],
                  list_cols: list[str]) -> tuple[dict, dict]:
    """One match's (scalar column counts, nested list-field counts). Worker body.

    Two reads on purpose: the scalar pass takes a column-pushdown parquet read, the list
    pass needs raw JSON because arrow does not hand a list-of-dict back as plain dicts.
    """
    ev = L.load_events(match_id, columns=scalar_cols)
    scalars = {}
    for col in ev.columns:
        counts, numeric = _column_counts(ev[col])
        if counts:
            scalars[col] = (counts, numeric)

    nested = {}
    if list_cols:
        raw = L.load_events(match_id, use_cache=False)
        nested = _list_column_values(raw, list_cols)
    return scalars, nested


def _flush(bucket: defaultdict[str, list], counted: dict) -> None:
    """Fold the buffered values into the running counters and drop the buffers."""
    for key, values in bucket.items():
        counts, numeric = _counts_from_values(values)
        if not counts:
            continue
        if key in counted:
            counted[key][0].update(counts)
        else:
            counted[key] = [counts, numeric]
    bucket.clear()


def _tracking_file_counts(path: Path, flush_every: int = _FLUSH_FRAMES) -> dict[str, dict]:
    """One tracking file's value domain, per inventory. Worker body: streams line by line.

    Buffers are folded into counters every `flush_every` frames rather than held for the
    whole file: buffering a whole file costs several GB per worker, because a Python list
    of floats is a list of boxed objects. Flushing is exact, not approximate: counter
    addition is associative, so the result is identical to counting once at the end, only
    bounded in memory.
    """
    counted = {name: {} for name in
               ("tracking_metadata", "tracking_frame", "tracking_player",
                "tracking_ball", "tracking_referee")}
    buckets = {name: defaultdict(list) for name in counted}
    entry_table = {"homePlayers": "tracking_player", "awayPlayers": "tracking_player",
                   "balls": "tracking_ball", "referees": "tracking_referee"}

    with open(path) as f:
        for k, v in json.loads(f.readline()).items():
            _emit(buckets["tracking_metadata"], k, v)
        for i, line in enumerate(f, 1):
            fr = json.loads(line)
            for k, v in fr.items():
                if k not in _TRACK_ENTRY_LISTS:
                    _emit(buckets["tracking_frame"], k, v)
            for side, table in entry_table.items():
                for entry in fr.get(side, []):
                    if entry:
                        for k, v in entry.items():
                            _emit(buckets[table], k, v)
            if i % flush_every == 0:
                for table in buckets:
                    _flush(buckets[table], counted[table])

    for table in buckets:
        _flush(buckets[table], counted[table])
    return {t: {k: (c, n) for k, (c, n) in d.items()} for t, d in counted.items()}


# ─────────────────────────────────── reduce ───────────────────────────────────

def _merge(totals: dict, is_num: dict, table: str, src: dict) -> None:
    """Fold one unit's counters into the running totals, keyed by (table, field)."""
    for field, (counts, numeric) in src.items():
        key = (table, field)
        if key not in totals:
            totals[key] = Counter()
            is_num[key] = numeric
        totals[key].update(counts)


def _quantiles(counts: Counter) -> dict:
    """Exact quantiles reconstructed from a numeric counter, no sample kept.

    The counter IS the histogram: sorting its keys once and walking the cumulative counts
    gives the true order statistics of the canonicalised column.
    """
    keys = np.array(sorted(counts.keys()), dtype=float)
    freq = np.array([counts[k] for k in keys], dtype=np.int64)
    cum = np.cumsum(freq)
    n = int(cum[-1])
    out = {"min": float(keys[0]), "max": float(keys[-1])}
    for q in _QUANTILES:
        out[f"p{int(q * 100)}"] = float(keys[int(np.searchsorted(cum, q * n, side="left"))])
    mean = float((keys * freq).sum() / n)
    out["mean"] = float(f"{mean:.{_SIG_FIGS}g}")
    return out


def _frames(totals: dict, is_num: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(top-N values per field, one summary row per field) from the merged counters."""
    value_rows, summary_rows = [], []
    for (table, field), counts in totals.items():
        n = int(sum(counts.values()))
        numeric = is_num[(table, field)]
        # most_common breaks count ties by insertion order, which is worker-dependent;
        # the secondary sort on the value itself makes the ranking reproducible.
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))
        for rank, (value, cnt) in enumerate(ranked[:_TOP_N], 1):
            value_rows.append({"table_name": table, "column_name": field, "rank": rank,
                               "value": str(value), "count": cnt,
                               "share_pct": round(100 * cnt / n, 4)})
        row = {"table_name": table, "column_name": field,
               "kind": "numeric" if numeric else "categorical",
               "n_populated": n, "cardinality": len(counts),
               "top_value": str(ranked[0][0]),
               "top_share_pct": round(100 * ranked[0][1] / n, 2),
               "closed_enumeration": (not numeric) and len(counts) <= _TOP_N}
        row.update(_quantiles(counts) if numeric else {})
        summary_rows.append(row)
    return pd.DataFrame(value_rows), pd.DataFrame(summary_rows)


def build(match_ids: list[int], scalar_cols: list[str], list_cols: list[str],
          tracking_paths: list[Path], n_jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Value domain over events, their nested list fields, and the tracking sample."""
    if n_jobs == 1:
        ev_results = [_match_counts(m, scalar_cols, list_cols) for m in match_ids]
        tr_results = [_tracking_file_counts(p) for p in tracking_paths]
    else:
        ev_results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_match_counts)(m, scalar_cols, list_cols) for m in match_ids)
        tr_results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_tracking_file_counts)(p) for p in tracking_paths)

    totals: dict = {}
    is_num: dict = {}
    # joblib preserves input order, so the reduce is catalog order either way.
    for scalars, nested in ev_results:
        _merge(totals, is_num, "events", scalars)
        for (col, field), payload in nested.items():
            _merge(totals, is_num, f"events.{col}[]", {field: payload})
    for per_table in tr_results:
        for table, counted in per_table.items():
            _merge(totals, is_num, table, counted)

    return _frames(totals, is_num)


def inventory_columns(schema_path: Path) -> tuple[list[str], list[str]]:
    """(scalar event columns, list event columns) read off dump_schema's inventory.
    Fails loudly rather than scanning a guessed subset."""
    if not schema_path.exists():
        raise SystemExit(f"missing {schema_path} -- run pipeline/dump_schema.py first")
    inv = pd.read_parquet(schema_path)
    ev = inv[inv["table_name"] == "events"]
    return (ev.loc[ev["dtype"] != "list", "column_name"].tolist(),
            ev.loc[ev["dtype"] == "list", "column_name"].tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N matches and N tracking files (smoke test)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="workers; --jobs N must give output identical to --jobs 1")
    ap.add_argument("--no-tracking", action="store_true",
                    help="skip the tracking pass (the events passes are much cheaper)")
    ap.add_argument("--schema", type=str, default=None,
                    help="dump_schema inventory parquet (default: data/derived/schema_dump.parquet)")
    ap.add_argument("--out", type=str, default=None,
                    help="output TSV for the top-N values, with the parquets beside it (default: "
                         "docs/hudl/values.tsv and data/derived/value_*.parquet, tagged by "
                         "--limit, --no-tracking and --schema)")
    args = ap.parse_args()

    schema_path = Path(args.schema) if args.schema else _REPO / "data" / "derived" / "schema_dump.parquet"
    scalar_cols, list_cols = inventory_columns(schema_path)

    catalog = L.list_matches(None)
    if args.limit:
        catalog = catalog.iloc[:args.limit]
    mids = catalog["match_id"].tolist()

    tracking_paths = [] if args.no_tracking else sorted(T.DATA_DIR.glob("*.jsonl"))
    if args.limit:
        tracking_paths = tracking_paths[:args.limit]

    print(f"VALUE DOMAIN SCAN -- {len(mids)} matches ({len(scalar_cols)} scalar + "
          f"{len(list_cols)} list columns), {len(tracking_paths)} tracking files, "
          f"{args.jobs} worker(s)")
    t0 = time.time()
    values, summary = build(mids, scalar_cols, list_cols, tracking_paths, args.jobs)
    print(f"  {len(summary)} fields with values, {len(values)} (field, value) rows, "
          f"{time.time() - t0:.1f}s")

    print("\nfields per table_name:")
    print(summary["table_name"].value_counts().to_string())
    n_cat = int((summary["kind"] == "categorical").sum())
    print(f"\ncategorical: {n_cat} ({int(summary['closed_enumeration'].sum())} fully "
          f"enumerated within top-{_TOP_N}) | numeric: {len(summary) - n_cat}")

    tag = ((f"_limit{args.limit}" if args.limit else "") + ("_no_tracking" if args.no_tracking else "")
           + (f"_{Path(args.schema).stem}" if args.schema else ""))
    out = Path(args.out) if args.out else _REPO / "docs" / "hudl" / f"values{tag}.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    values.to_csv(out, sep="\t", index=False)
    # Derived by suffixing the stem, never by substring-replacing it: under a custom
    # --out whose name has no "values" in it, a replace is a no-op and the summary
    # silently overwrites the values file.
    summary_out = out.with_name(f"{out.stem}_summary{out.suffix}")
    summary.to_csv(summary_out, sep="\t", index=False)
    print(f"\nsaved {out}  ({len(values)} rows)")
    print(f"saved {summary_out}  ({len(summary)} rows)")

    if args.out:
        cache, cache_summary = out.with_suffix(".parquet"), out.with_name(f"{out.stem}_summary.parquet")
    else:
        cache = _REPO / "data" / "derived" / f"value_dump{tag}.parquet"
        cache_summary = cache.with_name(f"value_summary{tag}.parquet")
    cache.parent.mkdir(parents=True, exist_ok=True)
    values.to_parquet(cache)
    summary.to_parquet(cache_summary)
    print(f"cached {cache}  [gitignored]")

    print(f"\ntotal wall-clock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
