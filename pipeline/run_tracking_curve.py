"""run_tracking_curve — what would more tracked matches buy?

Tracking is new at Hudl and part of what the event showcases, and the grant holds 30 tracked matches
that will not grow. This driver asks what the pipeline would gain with more, as a projection and
never as a fact. Pre-registered (data/derived/prereg_2026-10-01/tracking_curve.md).

TWO STATISTICS, each on the tracked corners whose second ball the event chain resolved:

    S1  the tracked rung of the sufficiency ladder: the per-row log loss of who recovers under the
        freeze-frame's static race minus under the tracked race, both recalibrated leaving one
        match out (run_sufficiency's overlap headline; positive: tracking predicts better)
    S2  the race's graded probability: mean P(att recovers) on won second balls minus on lost
        ones (run_q2; evidence.md §14)

THE CURVE: N tracked matches drawn without replacement, N in 10, 15, 20, 25 and all of them,
`--draws` draws per N, every draw rescored on its own matches (Platt refitted, intervals by match),
giving the statistic, its 95% interval width and the MDE at 80% (2.80 standard errors). The whole
curve runs at five seeds.

TRK1 decides whether the projection may be quoted: the slope of log width on log N, fitted over the
draws, has a 95% interval (draws resampled within each N) that holds −0.5, the 1/√N the projection
assumes, at every seed. The projection is the number of tracked matches at which the observed
effect's 95% half-width reaches the effect, on the fitted curve and on the 1/√N one, each with its
interval. The text never writes "would be worth more": it writes the N that would be needed.

--kind free_kick_flank reads the flank free kicks from x >= 85 the same way. A statistic whose
population falls under the publication floor (MIN_CELL) is recorded with its count as not estimable,
and draws no curve: the tracked matches hold few flank free kicks.

    python pipeline/run_tracking_curve.py --jobs 23
    python pipeline/run_tracking_curve.py --jobs 23 --kind free_kick_flank    # tracking_curve_*_<kind>
    python pipeline/run_tracking_curve.py --draws 20 --boot 200 --seeds 1 --out-dir data/derived/smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))
_DER = _REPO / "data" / "derived"

from build_table import kind_tag                                       # noqa: E402
from constants import MIN_CELL                                         # noqa: E402
from eda_build import COMBINED_KINDS                                   # noqa: E402
from run_q2 import load_join                                           # noqa: E402
from run_sufficiency import build_snapshot, fuse, overlap_population   # noqa: E402
from s04b_snapshot import paired_delta, score_arm                      # noqa: E402
from validate import Z80, Z95, cluster_bootstrap_stats                 # noqa: E402

SIZES = (10, 15, 20, 25)
# The registered precision, and the only one the published records are written at.
_CURVE_DRAWS, _DRAW_BOOT, _FIT_BOOT, _SEEDS = 200, 1000, 2000, 5


# ── The two populations ───────────────────────────────────────────────────────────
def s1_frame(der: Path, jobs: int, kind: str = "corner") -> pd.DataFrame:
    """Tracked deliveries with an event-chain second ball: who won it, the tracked race's
    P(att recovers) as race_corners publishes it, and the freeze-frame's at the same temperature."""
    tag = kind_tag((kind,))
    prim, _ = overlap_population(tag, der)
    want = prim.groupby("match_id")["corner_id"].agg(set).to_dict()
    records = build_snapshot(sorted(want), want, jobs, (kind,))
    tau = float(pd.read_parquet(der / f"race_corners{tag}.parquet")["tau_s"].iloc[0])
    out = prim.assign(p_snapshot=fuse(records, prim["corner_id"], tau), p_race=prim["p_att"])
    out = out.dropna(subset=["p_snapshot", "p_race"])
    print(f"S1: {len(out)} corners over {out['match_id'].nunique()} tracked matches", flush=True)
    return out[["corner_id", "match_id", "won", "p_snapshot", "p_race"]].reset_index(drop=True)


def s2_frame(der: Path, kind: str = "corner") -> pd.DataFrame:
    """run_q2's labelled overlap: the race's P(att recovers) and who won the second ball."""
    j = load_join(kind_tag((kind,)), der)
    lab = j[j["won_second_ball"].notna()].assign(won=lambda f: f["won_second_ball"].astype(bool))
    print(f"S2: {len(lab)} corners over {lab['match_id'].nunique()} tracked matches", flush=True)
    return lab[["corner_id", "match_id", "won", "p_att"]].reset_index(drop=True)


# ── One draw ──────────────────────────────────────────────────────────────────────
def s1_stat(f: pd.DataFrame, boot: int, seed: int) -> tuple[float, float, float]:
    """(point, lo, hi): snapshot minus race, both recalibrated on these matches alone."""
    y, g = f["won"].to_numpy().astype(int), f["match_id"].to_numpy()
    d = paired_delta(score_arm(f["p_snapshot"].to_numpy(float), y, g),
                     score_arm(f["p_race"].to_numpy(float), y, g), g, n_boot=boot, seed=seed)
    return d["delta"], d["lo"], d["hi"]


def s2_stat(f: pd.DataFrame, boot: int, seed: int) -> tuple[float, float, float]:
    """(point, lo, hi): mean P(att recovers) on won minus on lost, matches resampled."""
    gap = {"gap": lambda s: s.loc[s["won"], "p_att"].mean() - s.loc[~s["won"], "p_att"].mean()}
    r = cluster_bootstrap_stats(f, gap, n_boot=boot, seed=seed).loc["gap"]
    return float(r["point"]), float(r["lo"]), float(r["hi"])


STATS = {"S1": s1_stat, "S2": s2_stat}


def draw_matches(mids: np.ndarray, n: int, draws: int, seed: int) -> list:
    """`draws` subsets of n matches without replacement, or the whole set once at full size."""
    if n >= len(mids):
        return [np.sort(mids)]
    rng = np.random.default_rng([seed, n])
    return [np.sort(rng.choice(mids, n, replace=False)) for _ in range(draws)]


def _task(name: str, f: pd.DataFrame, keep: np.ndarray, n: int, i: int, boot: int, seed: int) -> dict:
    point, lo, hi = STATS[name](f[f["match_id"].isin(keep)].reset_index(drop=True), boot,
                                seed * 100000 + n * 1000 + i)
    w = hi - lo
    return {"stat": name, "seed": seed, "n_matches": n, "draw": i, "point": point, "lo": lo, "hi": hi,
            "width": w, "se": w / (2 * Z95), "mde80": Z80 * w / (2 * Z95)}


# ── The fit and the projection ────────────────────────────────────────────────────
def slope(rows: pd.DataFrame) -> tuple[float, float]:
    """(intercept, slope) of log width on log N by least squares over the draws."""
    ok = rows[np.isfinite(rows["width"]) & (rows["width"] > 0)]
    b, a = np.polyfit(np.log(ok["n_matches"]), np.log(ok["width"]), 1)
    return float(a), float(b)


def fit_curve(rows: pd.DataFrame, full_n: int, effect: float, n_boot: int, seed: int) -> dict:
    """The fitted slope with its interval over draws resampled within each N, TRK1, and the N at
    which the half-width reaches the effect, on the fitted curve and on 1/√N from the full width."""
    a, b = slope(rows)
    rng = np.random.default_rng([seed, 9])
    groups = [g for _, g in rows.groupby("n_matches")]
    boots = np.array([slope(pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups]))
                      for _ in range(n_boot)])
    b_lo, b_hi = np.percentile(boots[:, 1], [2.5, 97.5])
    w_full = float(rows.loc[rows["n_matches"] == full_n, "width"].mean())
    target = np.log(2 * abs(effect))
    n_fit = np.exp((target - boots[:, 0]) / boots[:, 1])
    return {"seed": seed, "intercept": a, "slope": b, "slope_lo": float(b_lo), "slope_hi": float(b_hi),
            "TRK1": bool(b_lo <= -0.5 <= b_hi), "effect": effect, "full_n": full_n, "full_width": w_full,
            "n_star_fit": float(np.exp((target - a) / b)),
            "n_star_fit_lo": float(np.percentile(n_fit, 2.5)),
            "n_star_fit_hi": float(np.percentile(n_fit, 97.5)),
            "n_star_sqrt": float(full_n * (w_full / (2 * abs(effect))) ** 2)}


# ── The run ───────────────────────────────────────────────────────────────────────
def main() -> None:
    """Both populations, the curve at every seed, the fit, TRK1 and the projection."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=str, default=str(_DER),
                    help="where corners, situation and race_corners live")
    ap.add_argument("--draws", type=int, default=_CURVE_DRAWS)
    ap.add_argument("--boot", type=int, default=_DRAW_BOOT, help="match draws inside each curve draw")
    ap.add_argument("--fit-boot", type=int, default=_FIT_BOOT)
    ap.add_argument("--seeds", type=int, default=_SEEDS)
    ap.add_argument("--jobs", type=int, default=20, help="workers; the output does not depend on it")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="where to write (default: data/derived, only at the registered precision)")
    ap.add_argument("--kind", choices=COMBINED_KINDS, default="corner",
                    help="the population: corners, or the flank free kicks from x >= 85")
    args = ap.parse_args()
    kt = kind_tag((args.kind,))
    out = Path(args.out_dir) if args.out_dir else _DER
    precision = (args.draws, args.boot, args.fit_boot, args.seeds)
    if out.resolve() == _DER.resolve() and precision != (_CURVE_DRAWS, _DRAW_BOOT, _FIT_BOOT, _SEEDS):
        raise SystemExit(f"STOP: the published tracking_curve records come only at (draws, boot, fit "
                         f"boot, seeds) = {(_CURVE_DRAWS, _DRAW_BOOT, _FIT_BOOT, _SEEDS)}; "
                         f"pass --out-dir")
    if out.resolve() == _DER.resolve() and Path(args.tables).resolve() != _DER.resolve():
        raise SystemExit("STOP: tables outside data/derived are not the published ones, so their "
                         "curve may not take the published names; pass --out-dir")

    der = Path(args.tables)
    frames = {"S1": s1_frame(der, args.jobs, args.kind), "S2": s2_frame(der, args.kind)}
    thin = {name: f for name, f in frames.items() if len(f) < MIN_CELL}
    for name, f in thin.items():
        print(f"{name}: {len(f)} rows over {f['match_id'].nunique()} matches, under the "
              f"n={MIN_CELL} publication floor: not estimable", flush=True)
    tasks = []
    for name, f in frames.items():
        if name in thin:
            continue
        mids = np.sort(f["match_id"].unique())
        for seed in range(args.seeds):
            for n in (*[s for s in SIZES if s < len(mids)], len(mids)):
                tasks += [delayed(_task)(name, f, keep, n, i, args.boot, seed)
                          for i, keep in enumerate(draw_matches(mids, n, args.draws, seed))]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        rows = pd.DataFrame(Parallel(n_jobs=args.jobs, verbose=2)(tasks),
                            columns=["stat", "seed", "n_matches", "draw", "point", "lo", "hi",
                                     "width", "se", "mde80"])

    fits = []
    for (name, seed), r in rows.groupby(["stat", "seed"]):
        full_n = int(r["n_matches"].max())
        effect = float(r.loc[r["n_matches"] == full_n, "point"].iloc[0])
        fits.append({"stat": name, "estimable": True, "n_rows": len(frames[name]),
                     **fit_curve(r, full_n, effect, args.fit_boot, seed)})
    fits += [{"stat": name, "estimable": False, "n_rows": len(f),
              "full_n": int(f["match_id"].nunique())} for name, f in thin.items()]
    fits = pd.DataFrame(fits)
    curve = rows.groupby(["stat", "seed", "n_matches"])[["width", "mde80", "point"]].mean()
    decision = {name: {"TRK1": bool(g["TRK1"].all()), "TRK1_any_seed": bool(g["TRK1"].any()),
                       "seeds": int(len(g))} for name, g in fits[fits["estimable"]].groupby("stat")}
    decision.update({name: {"TRK1": None, "estimable": False} for name in thin})

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print("\nTHE CURVE (mean over draws)\n" + curve.round(5).to_string())
    print("\nFIT, TRK1 AND PROJECTION by seed\n" + fits.round(4).to_string(index=False))
    print("\nDECISION\n" + "\n".join(f"  {k}: {v}" for k, v in decision.items()), flush=True)

    out.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out / f"tracking_curve_draws{kt}.parquet", index=False)
    fits.to_parquet(out / f"tracking_curve_fit{kt}.parquet", index=False)
    print(f"\nsaved the tracking_curve{kt} records in {out}")


if __name__ == "__main__":
    main()
