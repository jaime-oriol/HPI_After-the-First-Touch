"""audit_360 — freeze-frame availability gate (the freeze-frame re-check form_answers.md Q4 commits to).

Reports the 360 coverage the pipeline depends on, per competition and per match:
the DELIVERY freeze-frame (the plan, s02) and the second-ball drop frames (the
CLEARANCE and the NEXT touch, s08), plus corner/attacking-free-kick volume and
delivery visible-area/rest-defence occupancy: the data-provider audit numbers
for docs/hudl/data_provider.tex (methods_second_ball.md §8). form_answers.md Q4
commits to re-checking freeze-frame availability at second contacts on the granted
export, with event-geometry features as the pre-registered fallback. On the granted
export it is a green light (measured.md §coverage_360).

    python pipeline/audit_360.py --jobs 20                    # Big-5 aggregate (default)
    python pipeline/audit_360.py --per-match                  # + one row per match
    python pipeline/audit_360.py --per-league                 # + one row per league
    python pipeline/audit_360.py --limit 20 --per-league       # smoke test before a full run
    python pipeline/audit_360.py --jobs 20 --kinds free_kick_flank   # audit360_free_kick_flank
    python pipeline/audit_360.py --source serie_a_sample --out data/derived/audit.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import Parallel, delayed

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))

import loader_sb as L                                   # noqa: E402
from build_table import kind_tag                        # noqa: E402
# s02's lines, bound and not re-typed, so the audit describes the geometry the features use.
from constants import HALF_X, PITCH_X_SB, PITCH_Y_SB    # noqa: E402
from s01_phases import _is_xy                            # noqa: E402
from s08_situation import DropSituation                 # noqa: E402
from zones import in_box                                 # noqa: E402

_PITCH_AREA = PITCH_X_SB * PITCH_Y_SB
# The attacking-third heuristic for the free-kick volume count, not s01's grammar (see free_kicks).
_FK_ATTACKING_X = 80.0
# ±30% around the convergent 9.5-9.7 corners per match; definitions.md §11 carries its provenance.
_CORNER_RATE_BAND = (9.5 * 0.7, 9.7 * 1.3)
# Full-season match counts implied by each league's club count this season: 18 clubs play 306.
_FULL_SEASON_MATCHES = {"bundesliga": 306, "la_liga": 380, "ligue_1": 306,
                        "premier_league": 380, "serie_a": 380}


def free_kicks(events: pd.DataFrame, attacking_third_x: float = _FK_ATTACKING_X) -> pd.DataFrame:
    """Attacking free kicks (Pass + pass.type.name == 'Free Kick', taken from the
    attacking third, x >= attacking_third_x). A LOCAL HEURISTIC for the data-provider
    volume count only, never fed into CornerPhaseGrammar/DropSituation.

    It is not s01's population and must not be read as one. It applies neither the height bar
    nor the flank band of the `free_kick_flank` kind the grammar builds, so the two are
    different quantities counted over the same events; `definitions.md` §11 sets the two counts
    side by side."""
    e = events
    mask = e["pass.type.name"] == "Free Kick"
    x = e["location"].map(lambda v: float(v[0]) if _is_xy(v) else float("nan"))
    mask = mask & (x >= attacking_third_x)
    cols = [c for c in ("id", "period", "team.name", "location") if c in e.columns]
    return e.loc[mask, cols]


def _visible_area_pct(polygon: Any) -> float:
    """Shoelace polygon area / full-pitch area, as a percent. NaN if not a usable
    polygon (StatsBomb visible_area arrives as a flat [x1,y1,x2,y2,...] list)."""
    if not isinstance(polygon, list) or len(polygon) < 6:
        return float("nan")
    xs, ys = polygon[0::2], polygon[1::2]
    n = len(xs)
    area = sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n))
    return abs(area) / 2.0 / _PITCH_AREA * 100.0


def _zone_counts(freeze_frame: Any) -> tuple[int, int]:
    """(n_deep, n_box): 360 freeze-frame bodies (either team) in the rest-defence
    band (x < HALF_X) vs inside the box, its width included as s02's zones read it, at
    the delivery, which answers whether rest-defence players are reliably inside
    visible_area, not just box occupants."""
    if not isinstance(freeze_frame, list):
        return 0, 0
    pts = [(p["location"][0], p["location"][1]) for p in freeze_frame if _is_xy(p.get("location"))]
    return sum(1 for x, _ in pts if x < HALF_X), sum(1 for x, y in pts if in_box(x, y))


def audit_match(mid: int, kinds: tuple = ("corner",)) -> dict:
    """One match's freeze-frame coverage across the three windows that matter, for the deliveries
    of `kinds`, plus FK volume and delivery visible-area/rest-defence occupancy. The record's
    `corners` column counts those deliveries, whatever the population."""
    ev = L.load_events(mid)
    fz = L.load_360(mid)
    ids = set(fz["event_uuid"])
    fz_idx = fz.set_index("event_uuid")
    ds = DropSituation(ev, fz, kinds)
    corners = ds.g.deliveries()
    r = dict(match_id=mid, corners=len(corners), attacking_fk=len(free_kicks(ev)),
             delivery_360=0, cleared=0, clearance_360=0, next_touch=0, next_360=0, situation_any=0,
             va_pct_sum=0.0, va_ge80_count=0, n_360_corners=0, deep_seen_sum=0, box_seen_sum=0)
    for _, c in corners.iterrows():
        has_360 = c["id"] in ids
        r["delivery_360"] += int(has_360)
        if has_360:
            row = fz_idx.loc[c["id"]]
            va_pct = _visible_area_pct(row["visible_area"])
            if pd.notna(va_pct):
                r["va_pct_sum"] += va_pct
                r["va_ge80_count"] += int(va_pct >= 80.0)
                r["n_360_corners"] += 1
            n_deep, n_box = _zone_counts(row["freeze_frame"])
            r["deep_seen_sum"] += n_deep
            r["box_seen_sum"] += n_box
        clear, nxt, _ = ds._clear_and_next(c)
        if clear is None:
            continue
        r["cleared"] += 1
        r["clearance_360"] += int(clear["id"] in ids)
        if nxt is not None:
            r["next_touch"] += 1
            r["next_360"] += int(nxt["id"] in ids)
        r["situation_any"] += int((nxt is not None and nxt["id"] in ids) or clear["id"] in ids)
    return r


def _pct(num: float, den: float) -> str:
    """Format num/den as a 1-decimal percent, or a padded 'n/a' when the denominator is 0."""
    return f"{100 * num / den:.1f}%" if den else "  n/a"


def main() -> None:
    """CLI: audit 360 coverage + set-piece volume over a competition-season."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str, default=None,
                    help="loader_sb source (default: all five Big-5 leagues)")
    ap.add_argument("--per-match", action="store_true", help="print one row per match")
    ap.add_argument("--per-league", action="store_true", help="print one row per league")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N matches (smoke test)")
    ap.add_argument("--jobs", type=int, default=1, help="matches audited in parallel")
    ap.add_argument("--kinds", type=str, default="corner", help="the s01 population to audit")
    ap.add_argument("--out", type=str, default=None,
                    help="output parquet (default: data/derived/audit360.parquet, tagged by "
                         "--kinds, --source and --limit)")
    args = ap.parse_args()
    kinds = tuple(k.strip() for k in args.kinds.split(","))

    catalog = L.list_matches(args.source)
    if args.limit:
        catalog = catalog.iloc[:args.limit]
    mids = catalog["match_id"].tolist()
    # Dicts and ONE frame in the parent keep --jobs byte-identical: per-match frames infer dtypes.
    rows = ([audit_match(m, kinds) for m in mids] if args.jobs == 1 else
            Parallel(n_jobs=args.jobs, verbose=5)(delayed(audit_match)(m, kinds) for m in mids))
    df = pd.DataFrame(rows)
    tot = df.drop(columns="match_id").sum()

    # Two denominators on purpose: visible_area only accumulates where the polygon parses,
    # occupancy for every frame, and sharing one would skew the occupancy means.
    n360 = tot["n_360_corners"]
    n_frames = tot["delivery_360"]
    avg_va = tot["va_pct_sum"] / n360 if n360 else float("nan")
    avg_deep = tot["deep_seen_sum"] / n_frames if n_frames else float("nan")
    avg_box = tot["box_seen_sum"] / n_frames if n_frames else float("nan")

    print(f"360 COVERAGE AUDIT — source {args.source or 'big5 (all)'}  ({len(mids)} matches)\n")
    print(f"  corners:                    {int(tot['corners'])}")
    print(f"  attacking free kicks (heur, x>={_FK_ATTACKING_X:.0f}yd): {int(tot['attacking_fk'])}  [unverified]")
    print(f"  delivery 360 (the plan):    {int(tot['delivery_360'])}  ({_pct(tot['delivery_360'], tot['corners'])})")
    print(f"  visible_area (360 deliveries): mean {avg_va:.1f}% pitch, "
          f"{_pct(tot['va_ge80_count'], n360)} clear >=80%")
    print(f"  freeze-frame occupancy (360 deliveries): mean {avg_deep:.1f} bodies deep (x<{HALF_X:.0f}), "
          f"{avg_box:.1f} bodies in the box")
    print(f"  cleared (second-ball live): {int(tot['cleared'])}")
    print(f"  clearance 360:              {int(tot['clearance_360'])}  ({_pct(tot['clearance_360'], tot['cleared'])})")
    print(f"  next-touch exists:          {int(tot['next_touch'])}  ({_pct(tot['next_touch'], tot['cleared'])})")
    print(f"  next-touch 360 (the drop):  {int(tot['next_360'])}  ({_pct(tot['next_360'], tot['next_touch'])})")
    print(f"  situation frame available:  {int(tot['situation_any'])}  "
          f"({_pct(tot['situation_any'], tot['cleared'])} of cleared — next-touch or clearance 360)")

    if args.per_match:
        show = df.copy()
        show["deliv%"] = (100 * show["delivery_360"] / show["corners"].clip(lower=1)).round(0)
        show["sit%"] = (100 * show["situation_any"] / show["cleared"].clip(lower=1)).round(0)
        print("\nper match:")
        print(show[["match_id", "corners", "attacking_fk", "delivery_360", "deliv%",
                    "cleared", "situation_any", "sit%"]].to_string(index=False))

    if args.per_league:
        merged = df.merge(catalog[["match_id", "league"]], on="match_id")
        agg = merged.drop(columns="match_id").groupby("league").sum(numeric_only=True)
        agg["n_matches"] = merged.groupby("league").size()
        agg["corners_per_match"] = agg["corners"] / agg["n_matches"]
        agg["fk_per_match"] = agg["attacking_fk"] / agg["n_matches"]
        lo, hi = _CORNER_RATE_BAND
        print("\nper league:")
        for lg, row in agg.iterrows():
            flag = (f"  [FLAG: outside the {lo:.2f}-{hi:.2f}/match band]"
                    if not (lo <= row["corners_per_match"] <= hi) else "")
            print(f"  {lg:16s} n={int(row['n_matches']):4d}  corners={int(row['corners']):5d} "
                  f"({row['corners_per_match']:.2f}/match){flag}  "
                  f"fk={int(row['attacking_fk']):4d} ({row['fk_per_match']:.2f}/match, unverified)  "
                  f"deliv360={_pct(row['delivery_360'], row['corners'])}  "
                  f"sit={_pct(row['situation_any'], row['cleared'])}")
        print("\n  match-count reconciliation (granted vs current-club-count full season):")
        for lg, row in agg.iterrows():
            expected = _FULL_SEASON_MATCHES.get(lg)
            gap = f"  gap={expected - int(row['n_matches'])}" if expected else ""
            print(f"    {lg:16s} granted={int(row['n_matches']):4d}  expected={expected}{gap}")

    # A narrowed run (--kinds, --source, --limit) tags the default name: the grant's corner record is
    # what report_figures publishes (an explicit --out is honoured as given).
    tag = (kind_tag(kinds) + (f"_{args.source.lower()}" if args.source else "")
           + (f"_limit{args.limit}" if args.limit else ""))
    out = args.out or str(_REPO / "data" / "derived" / f"audit360{tag}.parquet")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"\nsaved {out}  ({len(df)} matches)  [gitignored]")

    print("\nNOTE: this is the coverage gate on every freeze-frame claim; re-run it after any loader change.")


if __name__ == "__main__":
    main()
