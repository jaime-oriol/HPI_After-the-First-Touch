"""build_table — build + cache the corner modelling table (S01 labels + S02 features).

Runs the freeze-frame pipeline over a dataset (default: all five Big-5 2025/26
leagues), prints the structural EDA (NO conclusions: distributions, 360 coverage,
sanity) and caches the S01+S02 joined table to data/derived/ for the value model.

One worker call per match yields the EDA rows and the table rows off the same cached
event frame, so the halves of the printed report cannot describe different corners.
Assembling the table here rather than through build_corner_table is also why this driver
calls s03's column guard itself: this is the table the pipeline publishes.

    python pipeline/build_table.py --limit 5 --jobs 1        # reference run
    python pipeline/build_table.py --limit 5 --jobs 4        # must match it exactly
    python pipeline/build_table.py --warm-cache --jobs 20     # the full grant
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import loader_sb as L                                   # noqa: E402
from constants import MIN_CELL, ZONE_PARTITION          # noqa: E402
from s01_phases import KINDS, SetPieceGrammar, is_own_goal   # noqa: E402
from s03_value import TARGET_MAP, _league_index, assert_declared_columns, corner_rows  # noqa: E402
from zones import zone_cols                             # noqa: E402

_DER = _REPO / "data" / "derived"
# Where the recipe's pass under partition F writes the published names (src/zones.py).
ZONES_F = _DER / "zones_F"


def kind_tag(kinds: tuple, default: tuple = ("corner",)) -> str:
    """The artefact suffix of a population: empty for the default, the kinds joined otherwise.

    The one naming rule every per-population artefact goes through, so a free-kick run can never
    land on a corner name."""
    return "" if tuple(kinds) == tuple(default) else "_" + "_".join(kinds)


def published_tag(table: str | Path) -> str | None:
    """The suffix of a grant table the pipeline publishes from, or None for any other table.

    `corners.parquet` gives "" and `corners_<kind>.parquet` gives "_<kind>" for one s01 kind. A
    --limit table, the two-population table and any path outside data/derived give None, and the
    model drivers write no record from those."""
    p = Path(table).resolve()
    return next((kind_tag((k,)) for k in KINDS
                 if p == (_DER / f"corners{kind_tag((k,))}.parquet").resolve()), None)


def published_dir(folder: str | Path) -> bool:
    """True for a folder the pipeline publishes records into: data/derived, and ZONES_F, where the
    same names are written by the pass under partition F. The guards that keep a reduced run off a
    published name read both through this."""
    return Path(folder).resolve() in {_DER.resolve(), ZONES_F.resolve()}


def assert_population(table: pd.DataFrame, tag: str) -> None:
    """Fail before modelling when a published table does not hold exactly the population its
    name declares: a record named for one kind and fitted on another reads as the first."""
    want, got = {tag.lstrip("_") or "corner"}, set(table["set_piece_type"].unique())
    if got != want:
        raise ValueError(f"a table named for {sorted(want)} holds {sorted(got)}")


def match_rows(match_id: int, league: str, kinds: tuple = ("corner",)) -> tuple:
    """(S01 label rows, shot-grain rows, S01+S02 joined rows) for one match.

    All three come from the same cached event frame, so the EDA, the frontier table
    and the modelling table cannot describe different corners. The shot grain
    is what the segment invariants are checked at and what the descriptive layer
    consumes; the delivery grain is what the model is fit on."""
    g = SetPieceGrammar(L.load_events(match_id), kinds=kinds)
    labels, shots = g.label_all(), g.shots_all()
    tag = {"match_id": match_id, "league": league}
    return ([{**r, **tag} for r in labels.to_dict("records")] if len(labels) else [],
            [{**r, **tag} for r in shots.to_dict("records")] if len(shots) else [],
            corner_rows(match_id, league, kinds))


def main() -> None:
    """Run S01 over every corner, join S02 on the 360-carrying subset, cache the table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="first N matches only")
    ap.add_argument("--source", type=str, default=None,
                    help="loader_sb source: a Big-5 league, SERIE_A_SAMPLE, or omit for all Big-5")
    ap.add_argument("--jobs", type=int, default=1,
                    help="worker processes; sized by peak RSS per worker, not by core count")
    ap.add_argument("--kinds", type=str, default="corner",
                    help=f"comma-separated s01 kinds from {', '.join(KINDS)}; each is validated "
                         f"SEPARATELY and pooling is a deliberate act")
    ap.add_argument("--warm-cache", action="store_true",
                    help="build the per-match parquet cache first")
    ap.add_argument("--out", type=str, default=None,
                    help="output parquet (default: corners.parquet, tagged by --kinds, --source "
                         "and --limit)")
    args = ap.parse_args()

    mids = L.list_matches(args.source)["match_id"].tolist()
    if args.limit:
        mids = mids[:args.limit]
    kinds = tuple(k.strip() for k in args.kinds.split(","))
    assert set(kinds) <= set(KINDS), f"unknown kind in {kinds}; valid: {KINDS}"
    leagues = _league_index()
    pairs = [(int(m), leagues.get(int(m))) for m in mids]

    # A narrowed run (--source, --limit) or a free-kick run never lands on the grant's corner table,
    # so each tags the DEFAULT name, as eda_build does (an explicit --out is honoured as given).
    tag = (kind_tag(kinds) + (f"_{args.source.lower()}" if args.source else "")
           + (f"_limit{args.limit}" if args.limit else ""))
    out = args.out or str(_DER / f"corners{tag}.parquet")

    if args.warm_cache:
        t = time.time()
        n = L.rebuild_cache([m for m, _ in pairs], n_jobs=args.jobs)
        print(f"cache: {n} match(es) written in {time.time() - t:.1f}s")

    t = time.time()
    if args.jobs == 1:
        chunks = [match_rows(m, lg, kinds) for m, lg in pairs]
    else:
        chunks = Parallel(n_jobs=args.jobs, verbose=5)(
            delayed(match_rows)(m, lg, kinds) for m, lg in pairs)

    # One DataFrame from all rows at once: per-match frames + concat infer dtypes per
    # match, and the parallel table would stop being bit-identical to the serial one.
    s01 = pd.DataFrame([r for lab, _, _ in chunks for r in lab])
    shots = pd.DataFrame([r for _, sh, _ in chunks for r in sh])
    tab = pd.DataFrame([r for _, _, joined in chunks for r in joined])
    if s01.empty:
        print(f"no corners found in {len(mids)} match(es) — nothing to cache")
        return

    # The rates below are over the rows that have a delivery: a short corner no cross followed is a
    # corner taken and has no first contact, shot or clearance to average.
    deliv = s01[s01["anchor"] != "none"]
    print(f"S01: {len(s01)} corners, {len(deliv)} with a delivery / {len(mids)} matches in "
          f"{time.time() - t:.1f}s ({args.jobs} worker(s))")
    print("  outcome:", (deliv["outcome"].value_counts(normalize=True) * 100).round(1).to_dict())
    print("  swing:  ", (deliv["swing"].value_counts(normalize=True) * 100).round(1).to_dict())
    print("  won first contact: %.1f%% | after-clearance shot: %.1f%% | cleared: %.1f%%" % (
        100 * deliv["won_first_contact"].mean(skipna=True),
        100 * deliv["after_clearance_shot"].mean(), 100 * deliv["cleared"].mean()))

    # The partition invariants: if either fails, every number downstream counts wrong shots.
    seg_cols = ["n_direct", "n_flick_on", "n_prolongation", "n_after_clearance", "n_late"]
    i1 = int(s01[seg_cols].to_numpy().sum()) == int(s01["n_att_shots"].sum()) == len(shots)
    i2 = int(s01[seg_cols[:4]].to_numpy().sum()) == int(s01["n_att_shots_in_window"].sum())
    print(f"\nfrontier: {len(shots)} terminal events, {int(is_own_goal(shots).sum())} of them own goals"
          f"   I1(sum==total) {i1}   I2(in-window) {i2}")
    if not (i1 and i2):
        raise SystemExit("segment partition is broken — refusing to cache")
    seg = shots["segment"].value_counts()
    goals = shots.groupby("segment")["is_goal"].sum()
    for s in seg.index:
        flag = f"  [<{MIN_CELL}, not publishable]" if seg[s] < MIN_CELL else ""
        print(f"    {s:14s} events={seg[s]:6d}  goals={int(goals[s]):4d}  "
              f"goals/event={100 * goals[s] / seg[s]:5.2f}%{flag}")
    aft = shots["after_first_touch"]
    print(f"  after the first touch: {100 * aft.mean():.1f}% of terminal events, "
          f"{100 * shots.loc[aft, 'is_goal'].sum() / max(shots['is_goal'].sum(), 1):.1f}% of goals")

    if tab.empty:
        print("\nS02 join: 0 corners with a delivery 360 — nothing to cache")
        return
    tab["target"] = tab["outcome"].map(TARGET_MAP)
    assert_declared_columns(tab)
    print(f"\nS02 join: {len(tab)} deliveries with 360 "
          f"({100 * len(tab) / len(deliv):.0f}% of deliveries carry a freeze-frame)")
    print("  jobs mean (box / frontal / held back):",
          tab[["n_attack_cross", "n_stake_drop", "n_hold_back"]].mean().round(2).to_dict())
    for side in ("att", "def"):
        print(f"  {side} per zone of {ZONE_PARTITION}:",
              tab[zone_cols(side)].mean().round(2).to_dict())
    print("  target:", tab["target"].value_counts().to_dict())
    print("  leagues:", tab["league"].value_counts().to_dict())

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tab.to_parquet(out)
    # A blind replace("corners", "shots") silently writes BOTH tables to the same path
    # whenever --out does not contain the word.
    p = Path(out)
    stem = p.stem.replace("corners", "shots") if "corners" in p.stem else f"{p.stem}_shots"
    shots_out = p.with_name(f"{stem}.parquet")
    shots.to_parquet(shots_out)
    print(f"\nsaved {out}  ({len(tab)} rows) and {shots_out}  ({len(shots)} shots)  [gitignored]")


if __name__ == "__main__":
    main()
