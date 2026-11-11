"""run_flank_cut — where a flank free kick stops being a corner: the annex behind the x >= 85 cut.

The paper models corners and the flank free kicks from x >= 85 (definitions.md §11). This driver
measures why the floor sits there: the flank free kicks from x = 60 under the paper's predicate and
nothing else (outside the box width, not along the ground, no length bar), in 5-yard bands of the
taker's x. s01's grammar reads them through a subclass that lowers its flank floor, so a band below
85 holds the paper's kick taken further out, and long corners sit beside every band as the
reference.

    placement   the attackers' three jobs, which read the same under either partition, and the
                attackers and defenders per zone of the switch's partition at the kick, over the
                kicks carrying a delivery 360 (s02's counts, read in the kick's canonical frame)
    production  in-window chain xG and after-first-touch xG per delivery, match-clustered 95%
    mechanism   the share of deliveries with a shot, a direct one, one after the first touch, a
                flick-on or prolongation and an after-clearance one; the attack's first contact,
                over the deliveries that have one, its header share and median depth; the
                cleared rate
    cuts        at 85, the paper's, and at 90: below minus above for the after-first-touch rate and
                xG, the flick-on rate, the direct xG and the attack's first contact, paired over
                the same match draws

The rate per delivery is the primary reading and the after-first-touch share of shots sits beside
it, because a far kick almost never shoots direct and its share climbs whatever the touch produces
(definitions.md §8).

    python pipeline/run_flank_cut.py --jobs 20
    python pipeline/run_flank_cut.py --limit 40 --boot 200 --out-dir data/derived/smoke
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))
_DER = _REPO / "data" / "derived"

import loader_sb as L                                                   # noqa: E402
from build_table import published_dir                                  # noqa: E402
from constants import FK_FLANK_MIN_X, FK_MIN_X                         # noqa: E402
from s01_phases import SetPieceGrammar                                  # noqa: E402
from s02_signature import CommitmentSignature                           # noqa: E402
from s03_value import _league_index                                     # noqa: E402
from validate import cluster_bootstrap_mean, cluster_bootstrap_stats    # noqa: E402
from zones import JOBS, table_counts, zone_cols                         # noqa: E402

BAND = 5.0
JOBS_N = tuple(f"n_{j}" for j in JOBS)
CUTS = (FK_FLANK_MIN_X, 90.0)
CUT_MEASURES = ("p_aft", "xg_aft", "p_prol", "xg_direct", "won_fc")
_BOOT = 2000


class FlankFromHalf(SetPieceGrammar):
    """The paper's flank kind with its floor at the attacking half's line."""
    FLANK_MIN_X = FK_MIN_X


def _placement(d: pd.Series, fz: pd.DataFrame) -> dict:
    """Bodies per zone at the kick under both partitions, and the attackers' three jobs, which read
    the same under either: s02's subsets, counts and signature."""
    sig = CommitmentSignature.from_corner(d, fz)
    a, f = sig._att(), sig._def()
    kick_y = float(sig.frame.loc[sig.frame["is_ball"] == 1, "y"].iloc[0])
    return {**table_counts(a["x"].to_numpy(float), a["y"].to_numpy(float),
                           f["x"].to_numpy(float), f["y"].to_numpy(float), kick_y),
            **sig.signature()}


def match_rows(mid: int, league: str) -> tuple[list, list]:
    """(one row per corner and per flank free kick from x = 60, one per attacking shot) off one
    parse: s01's labels and shots, and the placement of every kick that carries a 360."""
    ev, fz = L.load_events(mid), L.load_360(mid)
    g = FlankFromHalf(ev, kinds=("corner", "free_kick_flank"))
    labels = g.label_all()
    if labels.empty:
        return [], []
    have = set(fz["event_uuid"])
    place = {d["id"]: _placement(d, fz) for _, d in g.deliveries().iterrows() if d["id"] in have}
    tag = {"match_id": mid, "league": league}
    rows = [{**r, **tag, **place.get(r["set_piece_id"], {}), "has_360": r["set_piece_id"] in place}
            for r in labels.to_dict("records")]
    return rows, [{**s, **tag} for s in g.shots_all().to_dict("records")]


def per_delivery(rows: list, shots: list) -> pd.DataFrame:
    """The delivery frame with the routes read per delivery: in-window xG in all, after the first
    touch and direct, and whether a delivery saw a shot of each kind."""
    d = pd.DataFrame(rows)
    s = pd.DataFrame(shots)
    s = s[s["segment"] != "late"]

    def xg(mask: pd.Series) -> pd.Series:
        return d["set_piece_id"].map(s[mask].groupby("set_piece_id")["xg"].sum()).fillna(0.0)

    d["xg_win"] = xg(pd.Series(True, index=s.index))
    d["xg_aft"] = xg(s["after_first_touch"].astype(bool))
    d["xg_direct"] = xg(s["segment"] == "direct")
    d["n_aft"] = d["n_flick_on"] + d["n_prolongation"] + d["n_after_clearance"]
    for name, n in (("p_shot", d["n_att_shots_in_window"]), ("p_direct", d["n_direct"]),
                    ("p_aft", d["n_aft"]), ("p_prol", d["n_flick_on"] + d["n_prolongation"]),
                    ("p_clear", d["n_after_clearance"])):
        d[name] = (n > 0).astype(float)
    d["won_fc"] = d["won_first_contact"].astype(float)
    return d


def band_stats(g: pd.DataFrame, boot: int) -> dict:
    """One band's production with match-clustered intervals, its mechanism and its placement."""
    m = g["match_id"].to_numpy()
    out: dict = {"n": len(g)}
    for col in ("xg_win", "xg_aft"):
        ci = cluster_bootstrap_mean(g[col].to_numpy(float), m, n_boot=boot)
        out.update({col: ci.point, f"{col}_lo": ci.lo, f"{col}_hi": ci.hi})
    for col in ("p_shot", "p_direct", "p_aft", "p_prol", "p_clear", "xg_direct", "won_fc"):
        out[col] = float(g[col].mean())
    out["aft_share"] = g["n_aft"].sum() / max(int(g["n_att_shots_in_window"].sum()), 1)
    out["fc_header"] = float(g["first_contact_is_header"].dropna().astype(float).mean())
    out["fc_x_median"] = float(g["delivery_end_x"].median())
    out["cleared"] = float(g["cleared"].astype(float).mean())
    seen = g[g["has_360"]]
    out["n_360"] = len(seen)
    out.update({c: float(seen[c].mean()) for c in (*JOBS_N, *zone_cols("att"), *zone_cols("def"),
                                                   "n_def_frontal")})
    return out


def cut_contrast(fk: pd.DataFrame, cut: float, boot: int) -> list:
    """Below minus above one cut, for each cut measure, over one set of match draws."""
    both = fk.assign(below=fk["delivery_x"] < cut)
    r = cluster_bootstrap_stats(both, {k: (lambda s, k=k: s.loc[s["below"], k].mean()
                                           - s.loc[~s["below"], k].mean())
                                       for k in CUT_MEASURES}, n_boot=boot)
    return [{"cut": cut, "measure": k, "below_minus_above": float(r.loc[k, "point"]),
             "lo": float(r.loc[k, "lo"]), "hi": float(r.loc[k, "hi"]),
             "n_below": int(both["below"].sum()), "n_above": int((~both["below"]).sum())}
            for k in CUT_MEASURES]


def main() -> None:
    """Every delivery of the grant through the lowered grammar, then the bands and the cuts."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8, help="workers reading the matches")
    ap.add_argument("--limit", type=int, default=None, help="first N matches: the path, not the numbers")
    ap.add_argument("--boot", type=int, default=_BOOT)
    ap.add_argument("--out-dir", type=str, default=str(_DER))
    args = ap.parse_args()
    out = Path(args.out_dir)
    if published_dir(out) and (args.limit or args.boot != _BOOT):
        raise SystemExit("STOP: a --limit or --boot run may not write the published flank_cut "
                         "records; pass --out-dir")

    mids = L.list_matches(None)["match_id"].tolist()[:args.limit]
    leagues = _league_index()
    t0 = time.time()
    chunks = Parallel(n_jobs=args.jobs)(delayed(match_rows)(int(m), leagues.get(int(m)))
                                        for m in mids)
    d = per_delivery([r for rows, _ in chunks for r in rows], [s for _, sh in chunks for s in sh])
    fk = d[d["set_piece_type"] == "free_kick_flank"].copy()
    fk["band"] = (np.floor(fk["delivery_x"] / BAND) * BAND).clip(upper=120.0 - BAND)
    corners = d[(d["set_piece_type"] == "corner") & ~d["short_corner"].astype(bool)]
    print(f"{len(mids)} matches in {time.time() - t0:.0f}s: {len(fk)} flank free kicks from "
          f"x = {FK_MIN_X:.0f} ({int((fk['delivery_x'] >= FK_FLANK_MIN_X).sum())} from the paper's "
          f"{FK_FLANK_MIN_X:.0f}), {len(corners)} long corners", flush=True)

    bands = pd.DataFrame([{"band": "long_corners", **band_stats(corners, args.boot)}]
                         + [{"band": f"{b:.0f}-{b + BAND:.0f}", **band_stats(g, args.boot)}
                            for b, g in fk.groupby("band")])
    cuts = pd.DataFrame([r for c in CUTS for r in cut_contrast(fk, c, args.boot)])
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 60)
    print("\nBANDS\n" + bands.round(4).to_string(index=False))
    print("\nCUTS, below minus above, paired over the same match draws\n"
          + cuts.round(4).to_string(index=False), flush=True)

    out.mkdir(parents=True, exist_ok=True)
    tail = f"_limit{args.limit}" if args.limit else ""
    bands.to_parquet(out / f"flank_cut_bands{tail}.parquet", index=False)
    cuts.to_parquet(out / f"flank_cut_contrasts{tail}.parquet", index=False)
    print(f"\nwrote flank_cut_bands{tail} and flank_cut_contrasts{tail} in {out}")


if __name__ == "__main__":
    main()
