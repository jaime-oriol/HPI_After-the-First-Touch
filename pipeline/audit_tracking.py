"""audit_tracking — Broadcast X&Y audit of the tracked Serie A sample (30 matches).

Three checks for docs/hudl/data_provider.tex (methods_second_ball.md §8), all scoped
to the 30-match tracking sample (src/loader.py has no Big-5-wide tracking):

  1. CROSS-LINK CHECK — every tracking file pairs to an events match of the tracked
     Serie A sample via (home, away, date). Enumerated directly from the tracking
     directory (not via loader.py's private _PAIRING cache), so a file that fails to
     resolve is actually visible: an already-built _PAIRING dict would silently drop it.
     Hard-fails on any unmatched file (the acceptance bar is 0 unmatched).
  2. COORDINATE CONVENTION CHECK — tracking x/y (meters) vs loader.py's
     PITCH_LENGTH_M/PITCH_WIDTH_M (a SOFT bound: broadcast tracking can legitimately
     extrapolate slightly past the touchline), and StatsBomb event `location` (yards)
     vs the declared [0,120]x[0,80] convention (a HARD bound: any violation is a bug).
  3. TRANSITION-WINDOW COVERAGE — Broadcast X&Y detection density in
     [clear, clear+TRANSITION_WINDOW_S] after each delivery's first defensive
     clearance (the same clearance DropSituation._clear_and_next already defines,
     reused here, not re-derived), a validation-only signal and never a training input.
     --kinds covers several populations off one tracking pass, one record each.

src/loader.py stays unmodified: this script imports and reuses its private helpers
(_parse_tracking_filename, _canon) rather than adding new surface to that module.

    python pipeline/audit_tracking.py --kinds corner,free_kick_flank    # full 30-match run
    python pipeline/audit_tracking.py --limit 3        # smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))

import loader_sb as L                                   # noqa: E402
import loader as T                                       # noqa: E402
from build_table import kind_tag                         # noqa: E402
from constants import PITCH_X_SB, PITCH_Y_SB             # noqa: E402
from s01_phases import SetPieceGrammar                   # noqa: E402
from s08_situation import DropSituation                 # noqa: E402

# An audit threshold, not independently sourced for tracking-detection quality
# (methods_second_ball.md §8 reads the share that clears it).
_ADOPTED_VISIBLE_BAR = 0.80
# Beyond this multiple of the pitch half-dimension a point is a lost-track mis-detection, not soft
# extrapolation, so it is reported apart: a handful of them dominate a raw min/max.
_WILD_MULT = 3.0


def cross_link_check() -> pd.DataFrame:
    """One row per tracking file on disk, with the events match_id it resolves to
    (or None). Build-time check: the acceptance bar is 0 unmatched rows."""
    catalog = L.list_matches(L.SERIE_A_SAMPLE)
    by_key = {(r.home_team, r.away_team, r.match_date): int(r.match_id) for r in catalog.itertuples()}
    rows = []
    for p in sorted(T.DATA_DIR.glob("*.jsonl")):
        home, away, date = T._parse_tracking_filename(p)
        key = (T._canon(home), T._canon(away), date)
        rows.append({"file": p.name, "home": home, "away": away, "date": date,
                     "match_id": by_key.get(key)})
    return pd.DataFrame(rows)


def coord_stats(mid: int, trk: pd.DataFrame) -> dict:
    """One match's coordinate-bound counters, so the caller can accumulate them without
    holding the frame: this check and the transition-window one share ONE `load_tracking` per
    match, and thirty frames do not fit in memory at once."""
    xs, ys = trk["x_tracking"], trk["y_tracking"]
    oob = (xs.abs() > T.PITCH_LENGTH_M / 2) | (ys.abs() > T.PITCH_WIDTH_M / 2)
    # out-of-band once confidence=='LOW' rows are dropped: on players the field is constant HIGH,
    # so this removes only interpolated ball rows (methods_second_ball.md §8)
    conf_ok = trk["confidence"] != "LOW"
    wild = ((xs.abs() > _WILD_MULT * T.PITCH_LENGTH_M / 2)
            | (ys.abs() > _WILD_MULT * T.PITCH_WIDTH_M / 2))
    return {"x_min": float(xs.min()), "x_max": float(xs.max()),
            "y_min": float(ys.min()), "y_max": float(ys.max()),
            "rows": len(trk), "oob": int(oob.sum()), "oob_masked": int((oob & conf_ok).sum()),
            "wild": int(wild.sum()), "wild_match": mid if wild.any() else None,
            "wild_low_conf": int((wild & (trk["confidence"] == "LOW")).sum()),
            "wild_ball": int((wild & (trk["is_ball"] == 1)).sum())}


def events_coordinate_check(sample_big5_n: int = 50) -> dict:
    """StatsBomb event location (yards, HARD bound) across a random Big-5 sample."""
    big5 = L.list_matches(None)
    sample_ids = big5["match_id"].sample(n=min(sample_big5_n, len(big5)), random_state=0)
    ex_min = ex_max = ey_min = ey_max = None
    n_ev_rows = n_ev_oob = 0
    for mid in sample_ids:
        ev = L.load_events(int(mid))
        loc = ev["location"].dropna()
        loc = loc[loc.map(lambda v: isinstance(v, list) and len(v) == 2)]
        if not len(loc):
            continue
        xs = loc.map(lambda v: v[0])
        ys = loc.map(lambda v: v[1])
        ex_min = float(xs.min()) if ex_min is None else min(ex_min, float(xs.min()))
        ex_max = float(xs.max()) if ex_max is None else max(ex_max, float(xs.max()))
        ey_min = float(ys.min()) if ey_min is None else min(ey_min, float(ys.min()))
        ey_max = float(ys.max()) if ey_max is None else max(ey_max, float(ys.max()))
        oob = (xs < 0) | (xs > PITCH_X_SB) | (ys < 0) | (ys > PITCH_Y_SB)
        n_ev_oob += int(oob.sum())
        n_ev_rows += len(loc)

    return {"events_x_range": (ex_min, ex_max), "events_y_range": (ey_min, ey_max),
            "events_oob_rows": n_ev_oob, "events_rows": n_ev_rows,
            "events_sample_n": len(sample_ids)}


def merge_coord(parts: list) -> dict:
    """Fold the per-match counters into the tracking half of the coordinate report."""
    return {
        "tracking_x_range": (min(p["x_min"] for p in parts), max(p["x_max"] for p in parts)),
        "tracking_y_range": (min(p["y_min"] for p in parts), max(p["y_max"] for p in parts)),
        "tracking_rows": sum(p["rows"] for p in parts),
        "tracking_oob_rows": sum(p["oob"] for p in parts),
        "tracking_oob_rows_masked": sum(p["oob_masked"] for p in parts),
        "tracking_wild_rows": sum(p["wild"] for p in parts),
        "tracking_wild_low_conf": sum(p["wild_low_conf"] for p in parts),
        "tracking_wild_ball": sum(p["wild_ball"] for p in parts),
        "tracking_wild_matches": len({p["wild_match"] for p in parts} - {None}),
    }


def _in_window(t: pd.DataFrame, period: int, t0: float, t1: float) -> pd.Series:
    """Rows of `t` inside [t0, t1] of `period` on the vendor game clock."""
    return (t["period_id"] == period) & (t["period_elapsed_s"] >= t0) & (t["period_elapsed_s"] <= t1)


def transition_window_coverage(match_id: int, ev: pd.DataFrame, fz: pd.DataFrame,
                                tracking: pd.DataFrame, kind: str = "corner") -> pd.DataFrame:
    """Per cleared delivery of `kind`, Broadcast X&Y detection density in
    [clear._t, clear._t + TRANSITION_WINDOW_S], as the empirical distributions of
    visibility/confidence rather than an assumed 0-1 quality scale (both fields are
    categorical in the raw data).

    The window is cut on `period_elapsed_s`, the vendor GAME clock, because `clear._t`
    is a StatsBomb timestamp and those are the two that line up (loader.load_tracking
    documents the pair). The VIDEO clock starts before kick-off by a per-broadcast margin,
    so against an 8 s window it would sample period-1 frames from before the corner.

    The ball channel is audited beside the players, and separately, because the two fail
    differently: a player row is `ESTIMATED` when the camera lost him, a ball row when the
    vendor interpolated a ball it could not track, which the provider says is the airborne
    case. `ball_frame_frac` is the share of window frames carrying any ball row at all, and
    `ball_visible_frac` the share of those rows the vendor marks as detected."""
    ds = DropSituation(ev, fz, kinds=(kind,))
    players = tracking[tracking["is_ball"] == 0]
    balls = tracking[tracking["is_ball"] == 1]
    rows = []
    for _, c in ds.g.deliveries().iterrows():
        clear, _, _ = ds._clear_and_next(c)
        if clear is None:
            continue
        t0 = clear["_t"]
        t1 = t0 + SetPieceGrammar.TRANSITION_WINDOW_S
        period = clear["period"]
        win = players[_in_window(players, period, t0, t1)]
        bwin = balls[_in_window(balls, period, t0, t1)]
        n, n_frames = len(win), int(win["frame_count"].nunique())
        rows.append({
            "match_id": match_id, "corner_id": c["id"], "clear_t": float(t0), "period": int(period),
            "n_frame_rows": n,
            "visible_frac": float((win["visibility"] == "VISIBLE").mean()) if n else float("nan"),
            "high_conf_frac": float((win["confidence"] == "HIGH").mean()) if n else float("nan"),
            "mean_players_per_frame": float(win.groupby("frame_count").size().mean()) if n else float("nan"),
            "ball_frame_frac": float(len(bwin) / n_frames) if n_frames else float("nan"),
            "ball_visible_frac": (float((bwin["visibility"] == "VISIBLE").mean())
                                  if len(bwin) else float("nan")),
        })
    return pd.DataFrame(rows)


def report_coordinates(coord: dict, n_matches: int) -> None:
    """Print the coordinate convention check: the tracking half soft-bounded, the events hard."""
    print("COORDINATE CONVENTION CHECK\n")
    print(f"  tracking x range (m), RAW min/max: {coord['tracking_x_range']}  vs +-{T.PITCH_LENGTH_M / 2:.1f}  (soft bound)")
    print(f"  tracking y range (m), RAW min/max: {coord['tracking_y_range']}  vs +-{T.PITCH_WIDTH_M / 2:.1f}  (soft bound)")
    print(f"  tracking out-of-band rows (raw):              {coord['tracking_oob_rows']} / {coord['tracking_rows']} "
          f"({100 * coord['tracking_oob_rows'] / coord['tracking_rows']:.2f}%)")
    print(f"  tracking out-of-band rows (confidence-masked): {coord['tracking_oob_rows_masked']} / {coord['tracking_rows']} "
          f"({100 * coord['tracking_oob_rows_masked'] / coord['tracking_rows']:.2f}%)  "
          "-- excludes confidence=='LOW' rows")
    print(f"  WILD outliers (>{_WILD_MULT:.0f}x the soft bound -- not plausible extrapolation): "
          f"{coord['tracking_wild_rows']} row(s) across {coord['tracking_wild_matches']} of {n_matches} match(es) "
          f"({coord['tracking_wild_low_conf']} low-confidence, {coord['tracking_wild_ball']} on the ball channel)")
    print("    -- raw min/max above is DOMINATED by these wild rows; do not read it as the soft bound's overrun.")
    print(f"  events x range (yd):  {coord['events_x_range']}  vs [0, {PITCH_X_SB:.0f}]  "
          f"(HARD bound, n={coord['events_sample_n']} matches)")
    print(f"  events y range (yd):  {coord['events_y_range']}  vs [0, {PITCH_Y_SB:.0f}]  (HARD bound)")
    print(f"  events out-of-band rows: {coord['events_oob_rows']} / {coord['events_rows']}")
    if coord["events_oob_rows"]:
        print(f"  FLAG: StatsBomb event locations violate the declared "
              f"{PITCH_X_SB:.0f}x{PITCH_Y_SB:.0f} convention -- investigate.")
    print()


def main() -> None:
    """CLI: cross-link check -> coordinate check -> transition-window coverage per population."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N of the 30 sample matches (smoke test)")
    ap.add_argument("--kinds", type=str, default="corner",
                    help="s01 populations, comma-separated: one tracking pass feeds them all, and "
                         "each writes audit_tracking[_kind][_limitN].parquet")
    ap.add_argument("--out-dir", type=str, default=None, help="where to write (default: data/derived)")
    args = ap.parse_args()
    kinds = tuple(k.strip() for k in args.kinds.split(","))

    print("CROSS-LINK CHECK (tracking file -> events match, tracked Serie A sample)\n")
    link = cross_link_check()
    unmatched = link[link["match_id"].isna()]
    print(f"  tracking files: {len(link)}   matched: {int(link['match_id'].notna().sum())}   "
          f"unmatched: {len(unmatched)}")
    if len(unmatched):
        print("  UNMATCHED FILES:")
        for _, r in unmatched.iterrows():
            print(f"    {r['file']}  ({r['home']} vs {r['away']}, {r['date']})")
        raise SystemExit(f"cross-link check FAILED: {len(unmatched)} tracking file(s) did not pair "
                         "to an events match -- this blocks the doc's cross-linking claim.")
    print("  PASS: all tracking files paired.\n")

    # ONE load_tracking per match, feeding every check and every population, and the frame is
    # dropped before the next.
    catalog = L.list_matches(L.SERIE_A_SAMPLE)
    if args.limit:
        catalog = catalog.iloc[:args.limit]
    mids = [int(m) for m in catalog["match_id"]]
    coord_parts, tables = [], {k: [] for k in kinds}
    for mid in mids:
        trk = T.load_tracking(mid)
        coord_parts.append(coord_stats(mid, trk))
        ev, fz = L.load_events(mid), L.load_360(mid)
        for k in kinds:
            tables[k].append(transition_window_coverage(mid, ev, fz, trk, k))
        del trk
    report_coordinates({**merge_coord(coord_parts), **events_coordinate_check()}, len(mids))

    out_dir = Path(args.out_dir) if args.out_dir else _REPO / "data" / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)
    win_s = SetPieceGrammar.TRANSITION_WINDOW_S
    for k in kinds:
        print(f"TRANSITION-WINDOW COVERAGE, {k} (Broadcast X&Y, [clear, clear+{win_s:.0f}s])\n")
        tab = pd.concat(tables[k], ignore_index=True) if tables[k] else pd.DataFrame()
        if len(tab):
            print(f"  cleared deliveries covered: {len(tab)}  (matches: {len(mids)})")
            print(f"  mean visible_frac:    {tab['visible_frac'].mean():.3f}")
            print(f"  mean high_conf_frac:  {tab['high_conf_frac'].mean():.3f}")
            print(f"  mean players/frame:   {tab['mean_players_per_frame'].mean():.1f}")
            print(f"  ball: mean frame_frac {tab['ball_frame_frac'].mean():.3f}, "
                  f"mean visible_frac {tab['ball_visible_frac'].mean():.3f}  "
                  "(the vendor interpolates the airborne ball; the race reads it as emitted)")
            share_ge = float((tab["visible_frac"] >= _ADOPTED_VISIBLE_BAR).mean())
            print(f"  share of deliveries with visible_frac >= {_ADOPTED_VISIBLE_BAR:.0%} "
                  f"(an audit threshold, not independently sourced): {share_ge:.1%}")
        else:
            print("  no cleared deliveries in this subset.")
        out = out_dir / f"audit_tracking{kind_tag((k,))}{f'_limit{args.limit}' if args.limit else ''}.parquet"
        tab.to_parquet(out)
        print(f"\nsaved {out}  ({len(tab)} cleared deliveries)  [gitignored]\n")


if __name__ == "__main__":
    main()
