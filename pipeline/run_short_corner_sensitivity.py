"""run_short_corner_sensitivity — does the headline contrast rest on the corners played short?

A short or ground corner is played to a team-mate, so s01 delivers it by its first cross, when one comes
within SHORT_CROSS_WINDOW_S of the kick, and the modelling table holds that cross: its 360 is the frame at
the cross and its label is anchored on it, so the first contact, the window and the outcome start where the
ball went into the box (definitions.md §3, constants.SHORT_CROSS_WINDOW_S). The contrast that carries the
argument (the delivery design against the whole 360 block, `s03.delivery_vs_360`, +delta = the delivery won)
is scored at the same regularisation, so that only the rows change, on:

    all             the modelling table: every long corner, and the cross of every short or ground corner
    long_only       the long corners alone
    short_cross_Ns  the crosses of the short or ground corners alone, taken within N seconds of the kick: the
                    window s01 fixes, and CROSS_WINDOWS_S around it
    kick_anchored   every corner at its own kick, a short or ground corner as the pass to a team-mate it is:
                    the rows the table would hold if the first touch after a short corner were the reception
                    of its own pass

The window is not fitted to the delays: it is the bound a first contact has after a long corner. The rule was
fixed before the run: the thesis holds on a population when its interval excludes zero on the delivery side
(`delivery_wins`). The short corners no cross reaches in time have no delivery and are not in any arm; what
they produced is counted beside the arms, in total and per team (`uncrossed_team`, "league|team"), which is how
a team's corner goals are told whole: its goals in the table and these.

`--tune` re-selects C inside each population instead. `run_validation --by-type` tunes each
population on its own rows with the ladder's grid (`s03.LADDER_TUNE`), while this driver holds C at
the class default or re-selects it over the full grid, so its figure for this contrast is not the
same estimator as these; the comparison that means something is the internal one.

    python pipeline/run_short_corner_sensitivity.py --jobs 20
    python pipeline/run_short_corner_sensitivity.py --jobs 20 --tune
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
_DER = _REPO / "data" / "derived"

import loader_sb as L                                                          # noqa: E402
from s01_phases import SetPieceGrammar                                         # noqa: E402
from s03_value import TARGET_MAP, CompetingOutcomesModel, _league_index, delivery_row   # noqa: E402

CROSS_WINDOWS_S = (5.0, 10.0, 15.0)


def contrast(table: pd.DataFrame, boot: int, tune: bool, jobs: int) -> dict:
    """delivery-vs-360 on one population, with C either re-selected or left at the class default."""
    m = CompetingOutcomesModel(n_splits=5)
    if tune:
        m.apply_best_params(m.tune_cv(table, n_jobs=jobs))
    return {"n": len(table), "matches": int(table["match_id"].nunique()),
            **m.delivery_vs_360(table, n_boot=boot)}


def cross_match(match_id: int, league: str, horizon: float) -> dict:
    """One match: the crosses after its short corners that come within `horizon` s and carry a 360, the
    short corners as the kicks they are, every first-cross delay, and what the short corners no cross
    reaches in time produced."""
    ev, fz = L.load_events(match_id), L.load_360(match_id)
    ids = set(fz["event_uuid"])
    g = SetPieceGrammar(ev, kinds=("corner",))
    primary = g.SHORT_CROSS_WINDOW_S
    g.SHORT_CROSS_WINDOW_S = np.inf     # every first cross of the possession, whenever it comes
    d = g.deliveries()
    crossed = d[d["anchor"] == "cross"]
    delay = dict(zip(crossed["kick_id"], crossed["kick_to_delivery_s"], strict=True))
    kicks = g.kicks()
    short = kicks[[g.is_short_corner(k) for _, k in kicks.iterrows()]]
    out = {"n_short": len(short), "league": league, "delays": [float(x) for x in delay.values()], "crosses": [],
           "kick_rows": [], "uncrossed": {"corners": 0, "shots": 0, "goals": 0, "goal_corners": 0},
           "goals_by_team": {}}
    for _, c in crossed.iterrows():
        if c["kick_to_delivery_s"] <= horizon and c["id"] in ids:
            out["crosses"].append({**delivery_row(c, asdict(g.label(c)), fz, match_id, league),
                                   "kick_id": c["kick_id"], "delay_s": float(c["kick_to_delivery_s"])})
    for _, k in short.iterrows():
        if k["id"] in ids:
            out["kick_rows"].append(delivery_row(k, asdict(g.label(k)), fz, match_id, league))
        if delay.get(k["id"], np.inf) > primary:
            shots, u = g.shots(k), out["uncrossed"]
            goals = sum(s.is_goal for s in shots)
            u["corners"] += 1
            u["shots"] += len(shots)
            u["goals"] += goals
            u["goal_corners"] += int(goals > 0)
            out["goals_by_team"][k["team.name"]] = out["goals_by_team"].get(k["team.name"], 0) + goals
    return out


def main() -> None:
    """Score the contrast on the table, on its long corners, on the crosses of the short ones and on the
    kicks they would be."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=8, help="the cross build, and the grid under --tune")
    ap.add_argument("--tune", action="store_true",
                    help="also re-select C inside each population")
    ap.add_argument("--table", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N matches for the cross build")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    t0 = time.time()

    grant = _DER / "corners.parquet"
    src = Path(args.table) if args.table else grant
    if args.out is None and (args.limit or src.resolve() != grant.resolve()):
        raise SystemExit("STOP: a subset run may not write the published "
                         "short_corner_sensitivity.parquet; pass --out")
    if not src.exists():
        raise SystemExit(f"STOP: {src.name} absent; build_table.py writes it")
    table = pd.read_parquet(src)
    short = table["short_corner"].astype(bool)
    print(f"{len(table)} deliveries, {int(short.sum())} the cross of a short or ground corner "
          f"({100 * short.mean():.1f}%)", flush=True)

    primary = SetPieceGrammar.SHORT_CROSS_WINDOW_S
    horizon = max(*CROSS_WINDOWS_S, primary)
    leagues = _league_index()
    mids = [int(m) for m in L.list_matches(None)["match_id"]][:args.limit]
    chunks = Parallel(n_jobs=args.jobs)(delayed(cross_match)(m, leagues[m], horizon) for m in mids)
    n_short = sum(c["n_short"] for c in chunks)
    delays = np.array([x for c in chunks for x in c["delays"]])
    entered = delays[delays <= primary]
    unc = {k: sum(c["uncrossed"][k] for c in chunks) for k in ("corners", "shots", "goals", "goal_corners")}
    by_team: dict = {}
    for c in chunks:
        for team, goals in c["goals_by_team"].items():
            by_team[f"{c['league']}|{team}"] = by_team.get(f"{c['league']}|{team}", 0) + goals
    if sum(by_team.values()) != unc["goals"]:
        raise SystemExit(f"STOP: the teams' goals ({sum(by_team.values())}) are not the total's ({unc['goals']})")
    crosses = pd.DataFrame([r for c in chunks for r in c["crosses"]])
    kicks = pd.DataFrame([r for c in chunks for r in c["kick_rows"]])
    for t in (crosses, kicks):
        t["target"] = t["outcome"].map(TARGET_MAP)
    crosses = crosses.sort_values(["match_id", "kick_id"]).reset_index(drop=True)
    print(f"{n_short} short corners over {len(mids)} matches, {len(delays)} followed by a cross, "
          f"{len(entered)} within {primary:.0f} s; the others, {unc['corners']}, made {unc['shots']} shots "
          f"and {unc['goals']} goals", flush=True)
    built = int((crosses["delay_s"] <= primary).sum())
    if built != int(short.sum()):
        raise SystemExit(f"STOP: {built} crosses within {primary:.0f} s carry a 360 here and the table holds "
                         f"{int(short.sum())}: the table is not this build's")

    long_ = table[~short].reset_index(drop=True)
    kick_anchored = (pd.concat([long_, kicks], ignore_index=True)
                     .sort_values("match_id", kind="stable").reset_index(drop=True))
    pops = [("all", table, np.nan), ("long_only", long_, np.nan), ("kick_anchored", kick_anchored, np.nan)]
    pops += [(f"short_cross_{w:.0f}s",
              (table[short] if w == primary else crosses[crosses["delay_s"] <= w]).reset_index(drop=True), w)
             for w in sorted(set(CROSS_WINDOWS_S) | {primary})]
    rows = [{"block": "cross_delay", "key": k, "n": n, "point": v} for k, n, v in (
        ("short_corners", n_short, np.nan), ("crossed", len(delays), np.nan),
        ("in_window", len(entered), np.nan), ("window_s", len(delays), primary),
        ("delay_median_s", len(entered), float(np.median(entered))),
        ("delay_p90_s", len(entered), float(np.quantile(entered, 0.9))),
        ("uncrossed_corners", unc["corners"], np.nan), ("uncrossed_shots", unc["shots"], np.nan),
        ("uncrossed_goals", unc["goals"], np.nan), ("uncrossed_goal_corners", unc["goal_corners"], np.nan))]
    rows += [{"block": "uncrossed_team", "key": key, "n": goals} for key, goals in sorted(by_team.items())]
    for tune in ([False, True] if args.tune else [False]):
        arm = "tuned" if tune else "fixed_C"
        for name, tab, window in pops:
            r = contrast(tab, args.boot, tune, args.jobs)
            excludes, wins = bool(r["lo"] > 0 or r["hi"] < 0), bool(r["lo"] > 0)
            tag = "   <- primary" if window == primary else ""
            print(f"  {name + '/' + arm:24s} n={r['n']:6d}  delta {r['delta']:+.5f} "
                  f"[{r['lo']:+.5f}, {r['hi']:+.5f}]  "
                  f"{'delivery wins' if wins else 'excludes zero' if excludes else 'covers zero'}"
                  f"{tag}", flush=True)
            rows.append({"block": arm, "key": name, "n": r["n"], "matches": r["matches"],
                         "point": r["delta"], "lo": r["lo"], "hi": r["hi"],
                         "excludes_zero": excludes, "delivery_wins": wins, "window_s": window,
                         "primary": bool(window == primary),
                         "short_share_pct": round(100 * float(short.mean()), 1)})

    out = Path(args.out) if args.out else _DER / "short_corner_sensitivity.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"\nwrote {out}  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
