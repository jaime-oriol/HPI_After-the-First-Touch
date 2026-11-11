"""eda_throw_ins — the long-throw delivery grain, from the predicate probe_throw_ins adopted.

probe_throw_ins sweeps the predicate and runs the frontier on it but persists no delivery grain, so
the per-team long-throw count and the share carrying a delivery 360 come from here. It reuses the
probe's mask and its `adopt` rule on the probe's record, so the two cannot select different throws,
and it stops if the count disagrees with the one the probe published. A --limit N run reads the
record `probe_throw_ins --limit N` wrote over the same first N matches, so the check holds there too.

    python pipeline/eda_throw_ins.py --jobs 20
    python pipeline/eda_throw_ins.py --limit 150 --jobs 8       # smoke, after the probe's at 150
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
sys.path.insert(0, str(_REPO / "pipeline"))
_DER = _REPO / "data" / "derived"

import loader_sb as L                                    # noqa: E402
from probe_throw_ins import adopt, throw_in_mask         # noqa: E402
from s03_value import _league_index                      # noqa: E402


def adopted_cell(limit: int | None = None) -> tuple[tuple, int]:
    """((x_min, len_min, high_only), n) the probe's sweep adopted, read off its record; with `limit`,
    off the one its --limit run wrote."""
    g = pd.read_parquet(_DER / f"throw_in_probe{f'_limit{limit}' if limit else ''}.parquet")
    best = adopt(g[g["block"] == "grid"])
    return (float(best["x_min"]), float(best["len_min"]), bool(best["high_only"])), int(best["n"])


def match_rows(match_id: int, league: str, cell: tuple) -> list:
    """One match's long throws as dicts, with whether each one carries a delivery 360."""
    ev = L.load_events(match_id)
    ids = set(L.load_360(match_id)["event_uuid"])
    sel = ev[throw_in_mask(ev, *cell)]
    return [{"match_id": match_id, "league": league, "set_piece_id": r["id"],
             "attacking_team": r["team.name"], "delivery_360": r["id"] in ids}
            for _, r in sel.iterrows()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="first N matches only (smoke)")
    args = ap.parse_args()

    cell, n_probe = adopted_cell(args.limit)
    mids = L.list_matches()["match_id"].tolist()
    if args.limit:
        mids = mids[:args.limit]
    leagues = _league_index()
    t0 = time.time()
    chunks = Parallel(n_jobs=args.jobs)(delayed(match_rows)(int(m), leagues.get(int(m)), cell)
                                        for m in mids)
    tab = pd.DataFrame([r for chunk in chunks for r in chunk])
    per_team = tab.groupby(["league", "attacking_team"]).size()
    print(f"adopted cell {cell}: {len(tab)} long throws / {len(mids)} matches in "
          f"{time.time() - t0:.0f}s ({args.jobs} worker(s))")
    print(f"  per match {len(tab) / len(mids):.2f}   per team-season median {per_team.median():.0f} "
          f"min {per_team.min()} over {len(per_team)} teams   delivery 360 "
          f"{100 * tab['delivery_360'].mean():.2f}%")
    if len(tab) != n_probe:
        raise SystemExit(f"STOP: {len(tab)} throws against the probe's {n_probe} for the same cell")

    out = _DER / f"eda_throw_ins{f'_limit{args.limit}' if args.limit else ''}.parquet"
    tab.to_parquet(out)
    print(f"saved {out}  [gitignored]")


if __name__ == "__main__":
    main()
