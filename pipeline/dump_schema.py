"""dump_schema — full column/field inventory of the granted Hudl export.

Produces the data-provider doc's "Full column reference": every dotted column in
the Big-5 events export (union across the FULL grant, since StatsBomb's schema is
a genuine sparse union across event types: no single match exercises all
columns), the inner shape of every list-of-dict column, and every raw field in the
Broadcast X&Y tracking JSONL (metadata line + frame/player/ball/referee shapes).

    python pipeline/dump_schema.py --jobs 16       # the full grant
    python pipeline/dump_schema.py --limit 20      # smoke test

THE SENTINEL. StatsBomb/Hudl use an empty dict `{}` (sometimes an empty list `[]`) as
the "not applicable to this event type" marker on EVERY container-typed column, not
just freeze_frame/visible_area: confirmed on goalkeeper.end_location, pass.end_location,
carry.end_location and shot.end_location, where almost every row is the sentinel and
only a few carry a real [x, y] list. pandas' `.notna()` treats `{}` as present
because it is not NaN, so any dict/list column needs the extra emptiness check or its
pct_nonnull is silently inflated toward 100%. Scalar columns (float/str/bool) use real
NaN and do not need it, confirmed by contrast against a scalar column that sits at
its plausible coverage rather than being inflated toward 100%.

LIST COLUMNS. A dotted flatten stops dead at a list, so freeze_frame, tactics.lineup,
defensive_responsibility.players and the rest arrive as one opaque column whose entries
carry a schema of their own. Each such column gets its own inventory table, named
`events.<col>[]`, whose denominator is non-empty ENTRIES rather than events: one corner
delivery contributes a freeze-frame entry per visible player, so the two counts are not
comparable and must not share a table.

TRACKING SCOPE. All 30 files are read end to end, because a prefix cannot see a key only a
later match emits. VALUE-range auditing stays owned by audit_tracking.py and is not
duplicated here.

This driver reads the RAW JSON, never the parquet cache: it documents the vendor
export, and through the cache it would report the cache's own types and a `{}`-free
world instead.

DETERMINISM. Workers return per-unit stats and the parent reduces them in a fixed order
(catalog order for events, sorted path order for tracking), so --jobs N is byte-identical
to --jobs 1 down to ordinal_position and the first-seen example value.

OUTPUT NAMES. columns.tsv is the reference data_provider.tex quotes and schema_dump.parquet is
dump_values' input, so every flag that narrows the scan (--limit, --frames-per-file) tags both
default names, and with --out the parquet goes beside the TSV: the dump can run without touching
the record.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import Parallel, delayed

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import loader_sb as L                                   # noqa: E402
import loader as T                                       # noqa: E402


def _dtype_name(v: Any) -> str:
    """Python type name of one non-null value; pandas' object dtype is uninformative for JSON columns."""
    return type(v).__name__


def _example(v: Any, width: int = 80) -> str:
    """Truncated repr, cheap to compute during the same scan. Unwraps numpy scalars
    (np.float64(0.01) -> 0.01) for a readable cell; dtype already records the real type."""
    if hasattr(v, "item") and not isinstance(v, (list, dict)):
        try:
            v = v.item()
        except (ValueError, AttributeError):
            pass
    s = repr(v)
    return s if len(s) <= width else s[: width - 3] + "..."


def _count_populated(s: pd.Series) -> tuple[int, object]:
    """(count, first_populated_value) for one column, treating an empty dict/list
    as absent alongside real NaN. Two-tier for speed: the fast vectorized notna()
    first (one C-level pass); only if the surviving values are dict/list-typed
    (checked once, on the first survivor) does it pay the slower per-element
    empty-check, which scalar-typed columns (the vast majority) never pay."""
    nn = s[s.notna()]
    if not len(nn):
        return 0, None
    first = nn.iloc[0]
    if isinstance(first, (dict, list)):
        populated = nn[nn.map(lambda v: len(v) > 0)]
        if not len(populated):
            return 0, None
        return len(populated), populated.iloc[0]
    return len(nn), first


def _flatten(obj: dict, prefix: str = "") -> dict:
    """Recursive dict flatten with dotted keys, mirroring the events schema's own convention."""
    out = {}
    for k, v in obj.items():
        key = k if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _update_stats(stats: dict, obj: dict) -> None:
    """Accumulate {column: [nonnull_count, (dtype, example) or None]}, first-seen key order preserved."""
    for k, v in _flatten(obj).items():
        if k not in stats:
            stats[k] = [0, None]
        if v is not None:
            stats[k][0] += 1
            if stats[k][1] is None:
                stats[k][1] = (_dtype_name(v), _example(v))


def _merge_stats(dst: dict, src: dict) -> None:
    """Fold one unit's stats into the running total. Associative in the counts; the
    example is first-seen, so the CALLER owns the reduce order (see DETERMINISM)."""
    for k, (n, ex) in src.items():
        if k not in dst:
            dst[k] = [0, None]
        dst[k][0] += n
        if dst[k][1] is None and ex is not None:
            dst[k][1] = ex


def _rows_from_stats(table_name: str, stats: dict, total: int) -> list[dict]:
    rows = []
    for i, (col, (n, ex)) in enumerate(stats.items()):
        dtype, example = ex if ex else ("empty (never populated in scan)", "")
        rows.append({
            "table_name": table_name, "ordinal_position": i, "column_name": col,
            "dtype": dtype, "pct_nonnull": round(100 * n / total, 2) if total else 0.0,
            "example_value": example,
        })
    return rows


# ───────────────────────────────── events ─────────────────────────────────

def _is_dict_list(v: Any) -> bool:
    """A list whose entries carry their own schema, as opposed to [x, y] or a list of ids."""
    return isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict)


def _match_stats(match_id: int) -> tuple[dict, dict, int]:
    """One match's (column stats, per-list-column entry stats, row count). Worker body.

    Entry stats are keyed by the parent column and each carries its own entry count,
    because an entry denominator (freeze-frame players per event) is not the
    event denominator.
    """
    ev = L.load_events(match_id, use_cache=False)
    stats: dict = {}
    entry_stats: dict = {}
    for col in ev.columns:
        n, first = _count_populated(ev[col])
        stats[col] = [n, (_dtype_name(first), _example(first)) if n else None]
        if not (n and _is_dict_list(first)):
            continue
        es: dict = {}
        n_entries = 0
        for v in ev[col]:
            if not isinstance(v, list):
                continue
            for entry in v:
                if isinstance(entry, dict) and entry:
                    n_entries += 1
                    _update_stats(es, entry)
        entry_stats[col] = (es, n_entries)
    return stats, entry_stats, len(ev)


def events_inventory(match_ids: list[int], n_jobs: int) -> pd.DataFrame:
    """Union of every dotted column across the given matches, plus one table per
    list-of-dict column, with non-null rates and first-seen example values."""
    if n_jobs == 1:
        results = [_match_stats(m) for m in match_ids]
    else:
        results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_match_stats)(m) for m in match_ids)

    stats: dict = {}
    entry_totals: dict = {}
    total_rows = 0
    # joblib preserves input order, so this reduce runs in catalog order either way.
    for match_stats, match_entries, n_rows in results:
        _merge_stats(stats, match_stats)
        total_rows += n_rows
        for col, (es, n_entries) in match_entries.items():
            if col not in entry_totals:
                entry_totals[col] = [{}, 0]
            _merge_stats(entry_totals[col][0], es)
            entry_totals[col][1] += n_entries

    rows = _rows_from_stats("events", stats, total_rows)
    for col, (es, n_entries) in entry_totals.items():
        rows += _rows_from_stats(f"events.{col}[]", es, n_entries)
    return pd.DataFrame(rows)


# ──────────────────────────────── tracking ────────────────────────────────

def _tracking_file_stats(path: Path, max_frames: int | None) -> tuple:
    """One JSONL file's five inventories plus their counts. Worker body.

    Streams line by line and never materialises a frame: peak memory is one line
    plus the counters, so this scan is not bound by the 2-3 workers
    load_tracking needs.
    """
    meta_stats, frame_stats, player_stats, ball_stats, ref_stats = {}, {}, {}, {}, {}
    n_frames = n_players = n_balls = n_refs = 0
    with open(path) as f:
        _update_stats(meta_stats, json.loads(f.readline()))
        for i, line in enumerate(f):
            if max_frames is not None and i >= max_frames:
                break
            fr = json.loads(line)
            n_frames += 1
            frame_only = {k: v for k, v in fr.items()
                          if k not in ("homePlayers", "awayPlayers", "balls", "referees")}
            _update_stats(frame_stats, frame_only)
            for side in ("homePlayers", "awayPlayers"):
                for pl in fr.get(side, []):
                    if pl:
                        n_players += 1
                        _update_stats(player_stats, pl)
            for b in fr.get("balls", []):
                if b:
                    n_balls += 1
                    _update_stats(ball_stats, b)
            for r in fr.get("referees", []):
                if r:
                    n_refs += 1
                    _update_stats(ref_stats, r)
    return (meta_stats, frame_stats, player_stats, ball_stats, ref_stats,
            (1, n_frames, n_players, n_balls, n_refs))


def tracking_inventory(paths: list[Path], max_frames_per_file: int | None,
                       n_jobs: int) -> pd.DataFrame:
    """Raw JSONL field inventory: metadata line, frame-level, player/ball/referee entry shapes."""
    if n_jobs == 1:
        results = [_tracking_file_stats(p, max_frames_per_file) for p in paths]
    else:
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_tracking_file_stats)(p, max_frames_per_file) for p in paths)

    names = ["tracking_metadata", "tracking_frame", "tracking_player",
             "tracking_ball", "tracking_referee"]
    merged: list[dict] = [{} for _ in names]
    counts = [0] * len(names)
    # paths arrive sorted, and joblib preserves that order through the reduce.
    for res in results:
        for i in range(len(names)):
            _merge_stats(merged[i], res[i])
            counts[i] += res[5][i]

    rows = []
    for name, stats, total in zip(names, merged, counts, strict=True):
        rows += _rows_from_stats(name, stats, total)
    return pd.DataFrame(rows)


def main() -> None:
    """CLI: full-grant events column union (plus list-entry shapes) and a full tracking scan."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N Big-5 matches and N tracking files (smoke test)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="workers; --jobs N must give output identical to --jobs 1")
    ap.add_argument("--frames-per-file", type=int, default=None,
                    help="cap frames read per tracking file (default: the whole file)")
    ap.add_argument("--out", type=str, default=None,
                    help="output TSV, with its parquet beside it (default: docs/hudl/columns.tsv "
                         "and data/derived/schema_dump.parquet, tagged by --limit and "
                         "--frames-per-file)")
    args = ap.parse_args()

    catalog = L.list_matches(None)
    if args.limit:
        catalog = catalog.iloc[:args.limit]
    mids = catalog["match_id"].tolist()

    print(f"EVENTS SCHEMA SCAN -- {len(mids)} matches, {args.jobs} worker(s)")
    t0 = time.time()
    events_df = events_inventory(mids, args.jobs)
    n_event_cols = int((events_df["table_name"] == "events").sum())
    print(f"  {n_event_cols} distinct columns "
          f"(+{len(events_df) - n_event_cols} list-entry fields), {time.time() - t0:.1f}s")

    tracking_paths = sorted(T.DATA_DIR.glob("*.jsonl"))
    if args.limit:
        tracking_paths = tracking_paths[:args.limit]
    scope = "whole file" if args.frames_per_file is None else f"{args.frames_per_file} frames/file"
    print(f"\nTRACKING SCHEMA SCAN -- {len(tracking_paths)} files, {scope}")
    t1 = time.time()
    tracking_df = tracking_inventory(tracking_paths, args.frames_per_file, args.jobs)
    print(f"  {len(tracking_df)} distinct fields (metadata/frame/player/ball/referee), "
          f"{time.time() - t1:.1f}s")

    full = pd.concat([events_df, tracking_df], ignore_index=True)
    print("\nrows per table_name:")
    print(full["table_name"].value_counts().to_string())

    tag = (f"_limit{args.limit}" if args.limit else "") + (
        f"_frames{args.frames_per_file}" if args.frames_per_file is not None else "")
    out = Path(args.out) if args.out else _REPO / "docs" / "hudl" / f"columns{tag}.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out, sep="\t", index=False)
    print(f"\nsaved {out}  ({len(full)} rows)")

    cache = (out.with_suffix(".parquet") if args.out
             else _REPO / "data" / "derived" / f"schema_dump{tag}.parquet")
    cache.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(cache)
    print(f"cached {cache}  [gitignored]")

    print(f"\ntotal wall-clock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
