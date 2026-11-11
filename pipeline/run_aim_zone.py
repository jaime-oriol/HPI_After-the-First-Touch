"""run_aim_zone — does the delivery 360 carry the plan, and is the end-zone map biased?

The published nulls say the delivery freeze-frame adds nothing to the OUTCOME once the delivery
design is known. None of them asks whether it carries the INTENT. Attackers take their places for
where they have trained the ball to go, so the aim should be readable at the kick even though
`pass.end_location` is barred from X_t. And because that end location is where the defence got to
the ball on a defended delivery, a map of chain value by end zone would price the interception and
call it the zone. s02b_aim builds the pieces; this driver measures both claims.

POPULATION: long corners of the corner table with an end location and a delivery 360; the target
of the aim model is the canonical end zone, taught only on the corners the attack reached first.
Value is the in-window chain xG per corner (s01's shot grain, the `late` segment out). --kind
free_kick_flank reads the flank free kicks from x >= 85 the same way, with the canonical kick spot
in both arms' design (`SPOT`), as the paper's second population.

PRE-REGISTERED (data/derived/aim_zone_prereg_2026-09-21.md), before the four
leagues below were read. The Premier League is where the idea was developed, so it is reported
apart as the development sample and decides nothing. The confirmation sample is the other four
leagues, pooled, folds grouped by match.

    A1  the 360 carries the plan: D1 has its pivotal refit interval wholly below zero, keeps that
        side at every stability seed, and the shuffled kick block does not beat the design
    A2  the end-zone map is biased: D2 has its pivotal refit interval wholly below zero and keeps
        that side at every stability seed

(s02b_aim defines D1 and D2.) The kick block counts the zones of the partition the switch names
(src/zones.py). A1 and A2 were registered on corners and on C's twelve cells; under F the same
rules read F's zones, and on the flank free kicks they read that population.

The pivotal form is the one these rules were registered with, and it keeps deciding here although
every rule written after them reads the normal form (evidence.md §17); report_figures publishes the
normal form beside it, and neither that, the percentile form nor the Bonferroni level over the two
decisions decides. The honest map is reported, not decided: it is called flat only when its
pivotal interval sits inside the pre-registered margin of EQUIVALENCE_XG, and "not
distinguishable from zero" when it merely covers zero. Also reported and never deciding: both
maps inside each side of the first contact, with the aim's calibration, which is what separates
interception from the flattening a noisy aim causes; the robustness grid; each league on its
own; and the confirmation sample with the league in both arms' design.

    python pipeline/run_aim_zone.py --jobs 20
    python pipeline/run_aim_zone.py --jobs 20 --kind free_kick_flank       # aim_zone_<kind>
    python pipeline/run_aim_zone.py --limit 40 --boot 200 --refit-boot 3 --stability-seeds 1 \\
        --stability-boot 3 --out-dir data/derived/smoke     # the path, not the numbers
"""
from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))
_DER = _REPO / "data" / "derived"

import loader_sb as L                                                     # noqa: E402
from build_table import kind_tag, published_dir                           # noqa: E402
from run_first_contact import pivotal, seed_draws                         # noqa: E402
from s01_phases import SetPieceGrammar, is_own_goal                       # noqa: E402
from s02b_aim import (CONTRAST, GRIDS, KICK_COLS, KICK_SPOT, STATS,       # noqa: E402
                      aim_oof, assert_sources, contrast, design, end_zones, kick_block,
                      soft_contrast, statistics, zone_loss)
from validate import (cluster_bootstrap_mean, cluster_bootstrap_stats,    # noqa: E402
                      shuffle_features)
from zones import canonical_y                                             # noqa: E402

DEVELOPMENT = ("premier_league",)
# The numbers each population's design carries beside its categoricals: none on a corner, struck
# from the flag, so the design A1 and A2 were registered on stays as it was.
SPOT = {"corner": (), "free_kick_flank": KICK_SPOT}
AIM_DECISIONS = ("D1", "D2")
EQUIVALENCE_XG = 0.01
VALUES = ("xg", "xg_aft", "any_shot")
# Below this many matches GroupKFold's five outer folds, each with a three-way inner split on its
# attack-won rows, stop being estimable; a smaller sample is reported as skipped, not fitted.
_MIN_MATCHES = 15
# The registered precision, and the only one the published record is written at.
_BOOT, _REFIT_BOOT, _SEEDS, _STABILITY_BOOT = 2000, 300, 5, 2000


# ── The table ────────────────────────────────────────────────────────────
def match_block(mid: int, want: set, kind: str = "corner") -> list:
    """One match's kick blocks, one dict per wanted delivery of `kind` that carries a delivery 360."""
    ev, fz = L.load_events(mid), L.load_360(mid)
    ids = set(fz["event_uuid"])
    d = SetPieceGrammar(ev, kinds=(kind,)).deliveries()
    return [{"set_piece_id": c["id"], **kick_block(c, fz)}
            for _, c in d.iterrows() if c["id"] in want and c["id"] in ids]


def build(corners: pd.DataFrame, jobs: int, kind: str = "corner") -> pd.DataFrame:
    """Kick blocks over the table's matches, in sorted order, so the parallel build is the serial
    one: workers return dicts and the parent builds ONE frame."""
    mids = sorted(corners["match_id"].unique())
    want = {m: set(corners.loc[corners["match_id"] == m, "set_piece_id"]) for m in mids}
    t0 = time.time()
    chunks = ([match_block(m, want[m], kind) for m in mids] if jobs == 1 else
              Parallel(n_jobs=jobs)(delayed(match_block)(int(m), want[m], kind) for m in mids))
    rows = [r for ch in chunks for r in ch]
    print(f"kick blocks for {len(rows)} corners over {len(mids)} matches in "
          f"{time.time() - t0:.0f}s ({jobs} worker(s))", flush=True)
    return pd.DataFrame(rows)


def population(corners: pd.DataFrame, deliveries: pd.DataFrame, shots: pd.DataFrame,
               block: pd.DataFrame, kind: str = "corner") -> pd.DataFrame:
    """One population's long deliveries (a free kick is never short) with an end location and a
    kick block: their canonical end zone under every grid and kick spot, whether the attack made the
    first contact, and the three value measures. An own goal is a goal and no shot (s01_phases), so it
    adds to no `any_shot`, and its xG is zero."""
    t = corners[~corners["short_corner"].astype(bool) & corners["delivery_end_x"].notna()]
    dlv = deliveries.loc[deliveries["set_piece_type"] == kind, ["set_piece_id", "won_first_contact"]]
    t = t.merge(dlv, on="set_piece_id", how="inner").merge(block, on="set_piece_id", how="inner")
    right = (t["corner_side"] == "right").to_numpy()
    if not np.array_equal(right, t["kick_side_right"].to_numpy(bool)):
        raise SystemExit("STOP: the frame's corner side disagrees with s01's corner_side")
    ey = canonical_y(t["delivery_end_y"].to_numpy(float), right)
    for grid in GRIDS:
        t[f"zone_{grid}"] = end_zones(grid, t["delivery_end_x"].to_numpy(float), ey)
    t["zone"] = t["zone_base"]
    t["won"] = t["won_first_contact"].eq(True)
    t["kick_x"] = t["delivery_x"].astype(float)
    t["kick_y"] = canonical_y(t["delivery_y"].to_numpy(float), right)
    s = shots[shots["segment"] != "late"]
    t["xg"] = t["set_piece_id"].map(s.groupby("set_piece_id")["xg"].sum()).fillna(0.0)
    aft = s[s["after_first_touch"].astype(bool)]
    t["xg_aft"] = t["set_piece_id"].map(aft.groupby("set_piece_id")["xg"].sum()).fillna(0.0)
    t["any_shot"] = t["set_piece_id"].isin(s.loc[~is_own_goal(s), "set_piece_id"]).astype(float)
    return t.sort_values(["match_id", "set_piece_id"]).reset_index(drop=True)


# ── Fixed-fit readings ───────────────────────────────────────────────────
def aim_readability(t: pd.DataFrame, sample: str, numeric: tuple = ()) -> list:
    """How well each input set predicts the zone on attack-won corners, against the zones' base
    rates."""
    zones = GRIDS["base"]
    z = t["zone"].map({k: i for i, k in enumerate(zones)}).to_numpy()
    won, g = t["won"].to_numpy(bool), t["match_id"].to_numpy()
    A, K = design(t, numeric=numeric)
    prior = np.bincount(z[won], minlength=len(zones)) / won.sum()
    rows = [{"sample": sample, "block": "aim", "key": "prior", "n": int(won.sum()),
             "point": float(-np.log(np.clip(prior[z[won]], 1e-6, 1)).mean()),
             "top1": float(prior.max())}]
    for key, X in (("design", A), ("kick_block", K[:, A.shape[1]:]), ("design_kick_block", K)):
        p = aim_oof(X, z, won, g, len(zones))[won]
        rows.append({"sample": sample, "block": "aim", "key": key, "n": int(won.sum()),
                     "point": float(zone_loss(p, z[won]).mean()),
                     "top1": float((p.argmax(1) == z[won]).mean())})
    return rows


def fit_aim(t: pd.DataFrame, grid: str, extra: tuple = (),
            numeric: tuple = ()) -> tuple[np.ndarray, np.ndarray]:
    """(design-only, design + kick block) out-of-fold aim probabilities under one grid."""
    zones = GRIDS[grid]
    z = t[f"zone_{grid}"].map({k: i for i, k in enumerate(zones)}).to_numpy()
    won, g = t["won"].to_numpy(bool), t["match_id"].to_numpy()
    A, K = design(t, extra, numeric)
    return aim_oof(A, z, won, g, len(zones)), aim_oof(K, z, won, g, len(zones))


def fixed_intervals(t: pd.DataFrame, grid: str, value: str, boot: int,
                    fit: tuple | None = None, extra: tuple = (), numeric: tuple = ()) -> dict:
    """STATS on the fixed out-of-fold fit, each with its match-clustered percentile interval. The
    fit depends on the grid alone, so a caller sweeping value measures passes it once."""
    zones = GRIDS[grid]
    a, b = CONTRAST[grid]
    z = t[f"zone_{grid}"].map({k: i for i, k in enumerate(zones)}).to_numpy()
    won, g = t["won"].to_numpy(bool), t["match_id"].to_numpy()
    pa, pk = fit if fit is not None else fit_aim(t, grid, extra, numeric)
    d = zone_loss(pk, z) - zone_loss(pa, z)
    f = pd.DataFrame({"match_id": g, "v": t[value].to_numpy(float),
                      "end": t[f"zone_{grid}"].to_numpy(),
                      "aim": np.asarray(zones, dtype=object)[pk.argmax(1)]})
    P = pd.DataFrame(pk, columns=[f"p{j}" for j in range(len(zones))])
    f = pd.concat([f, P], axis=1)
    ja, jb = zones.index(a), zones.index(b)

    def soft(s: pd.DataFrame) -> float:
        return soft_contrast(s["v"].to_numpy(), s[P.columns].to_numpy(), ja, jb)

    out = cluster_bootstrap_stats(f, {
        "D2": lambda s: contrast(s["v"].to_numpy(), s["end"].to_numpy(), a, b)
                         - contrast(s["v"].to_numpy(), s["aim"].to_numpy(), a, b),
        "naive": lambda s: contrast(s["v"].to_numpy(), s["end"].to_numpy(), a, b),
        "honest": lambda s: contrast(s["v"].to_numpy(), s["aim"].to_numpy(), a, b),
        "soft": soft}, n_boot=boot)
    ci = cluster_bootstrap_mean(d[won], g[won], n_boot=boot)
    res = {"D1": (ci.point, ci.lo, ci.hi)}
    res.update({k: (float(r["point"]), float(r["lo"]), float(r["hi"])) for k, r in out.iterrows()})
    return res


def by_first_contact(t: pd.DataFrame, sample: str, boot: int, fit: tuple) -> list:
    """The naive and honest maps inside each side of the first contact, and the aim's calibration.

    A noisy argmax aim flattens any map read off it, so D2 alone cannot tell interception from
    attenuation. Where the attack made the first contact the end zone IS the aim and nothing was
    intercepted: the gap between the two maps there is the attenuation, measured. Calibration on
    those same rows is what the soft map's correction rests on.

    A delivery no one touched inside the window has no first contact, so it sits on neither side:
    its xG is zero and its end zone is where the ball went out, not where anyone met it. Its count
    rides beside the two sides as `no_contact`."""
    zones = GRIDS["base"]
    a, b = CONTRAST["base"]
    pk = fit[1]
    contact = t["won_first_contact"].notna().to_numpy()
    f = pd.DataFrame({"match_id": t["match_id"], "v": t["xg"].to_numpy(float),
                      "end": t["zone"].to_numpy(), "won": t["won"].to_numpy(bool),
                      "aim": np.asarray(zones, dtype=object)[pk.argmax(1)]})
    rows = [{"sample": sample, "block": "by_first_contact", "grid": "no_contact", "key": "n",
             "n": int((~contact).sum())}]
    for side, keep in (("attack_won", f["won"].to_numpy()),
                       ("defence_first", ~f["won"].to_numpy() & contact)):
        s = f[keep].reset_index(drop=True)
        out = cluster_bootstrap_stats(s, {
            "naive": lambda x: contrast(x["v"].to_numpy(), x["end"].to_numpy(), a, b),
            "honest": lambda x: contrast(x["v"].to_numpy(), x["aim"].to_numpy(), a, b),
            "gap": lambda x: contrast(x["v"].to_numpy(), x["end"].to_numpy(), a, b)
                             - contrast(x["v"].to_numpy(), x["aim"].to_numpy(), a, b)},
            n_boot=boot)
        rows += [{"sample": sample, "block": "by_first_contact", "grid": side, "key": k,
                  "n": len(s), "point": float(r["point"]), "lo": float(r["lo"]),
                  "hi": float(r["hi"])} for k, r in out.iterrows()]
    won = f["won"].to_numpy()
    for zone in (a, b):
        p = pk[won, zones.index(zone)]
        hit = (f["end"].to_numpy()[won] == zone).astype(float)
        edges = np.quantile(p, np.linspace(0, 1, 6))
        bin_ = np.clip(np.digitize(p, edges[1:-1]), 0, 4)
        rows += [{"sample": sample, "block": "calibration", "grid": zone, "key": f"q{k}",
                  "n": int((bin_ == k).sum()), "pred": float(p[bin_ == k].mean()),
                  "obs": float(hit[bin_ == k].mean())} for k in range(5)]
    return rows


def shuffled_control(t: pd.DataFrame, boot: int, numeric: tuple = ()) -> dict:
    """D1 with the kick block permuted jointly over every row: a block that still buys zone log
    loss there reads match or row structure the aim cannot carry, and no A1 reading means
    anything."""
    s = shuffle_features(t, KICK_COLS, mode="global", seed=0)
    lo_hi = fixed_intervals(s, "base", "xg", boot, numeric=numeric)["D1"]
    return {"point": lo_hi[0], "lo": lo_hi[1], "hi": lo_hi[2], "clean": bool(lo_hi[2] >= 0)}


def _robust_cell(t: pd.DataFrame, grid: str, value: str, within: bool, boot: int, fit: tuple,
                 sample: str) -> list:
    tv = t.copy()
    if within:
        tv[value] = tv[value] - tv.groupby("attacking_team")[value].transform("mean")
    res = fixed_intervals(tv, grid, value, boot, fit)
    return [{"sample": sample, "block": "robustness", "grid": grid, "value": value,
             "within_team": within, "key": k, "n": len(tv), "point": res[k][0], "lo": res[k][1],
             "hi": res[k][2]} for k in ("naive", "honest", "soft")]


def robustness(t: pd.DataFrame, sample: str, boot: int, jobs: int, numeric: tuple = ()) -> list:
    """The three maps under every grid and value measure, raw and demeaned within the attacking
    team. One aim fit per grid, then one task per (grid, value, within) cell."""
    fits = dict(zip(GRIDS, Parallel(n_jobs=min(jobs, len(GRIDS)))(
        delayed(fit_aim)(t, grid, (), numeric) for grid in GRIDS), strict=True))
    cells = [(grid, value, within) for grid in GRIDS for value in VALUES
             for within in (False, True)]
    parts = Parallel(n_jobs=min(jobs, len(cells)))(
        delayed(_robust_cell)(t, grid, value, within, boot, fits[grid], sample)
        for grid, value, within in cells)
    return [r for p in parts for r in p]


# ── Refit, stability and the verdicts ────────────────────────────────────
def _pivotal(point: np.ndarray, draws: np.ndarray, alpha: float) -> tuple:
    """run_first_contact.pivotal, or NaN bounds when no draw is finite whole, as on a thin smoke
    sample where a zone empties in every resample."""
    if not np.all(np.isfinite(draws), axis=1).any():
        return tuple(np.full(point.shape, np.nan) for _ in range(4))
    return pivotal(point, draws, alpha)


def refit_rows(point: np.ndarray, draws: np.ndarray, fixed: dict, sample: str) -> list:
    """Every statistic on the refit draws: pivotal decides, percentile and Bonferroni ride
    beside."""
    rows = []
    for i, key in enumerate(STATS):
        p_lo, p_hi, c_lo, c_hi = (v[i] for v in _pivotal(point, draws, 0.05))
        b_lo, b_hi, _, _ = (v[i] for v in _pivotal(point, draws, 0.05 / len(AIM_DECISIONS)))
        f_lo, f_hi = fixed[key][1], fixed[key][2]
        print(f"  {key:7s} {point[i]:+.5f} pivotal [{p_lo:+.5f}, {p_hi:+.5f}] percentile "
              f"[{c_lo:+.5f}, {c_hi:+.5f}] fixed [{f_lo:+.5f}, {f_hi:+.5f}]", flush=True)
        rows.append({"sample": sample, "block": "refit", "key": key,
                     "n": int(np.all(np.isfinite(draws), axis=1).sum()), "point": point[i],
                     "lo": p_lo, "hi": p_hi, "pct_lo": c_lo, "pct_hi": c_hi, "bonf_lo": b_lo,
                     "bonf_hi": b_hi, "fixed_lo": f_lo, "fixed_hi": f_hi,
                     "width_ratio": (c_hi - c_lo) / (f_hi - f_lo) if f_hi > f_lo else np.nan,
                     "below_zero": bool(p_hi < 0)})
    return rows


def equivalence(lo: float, hi: float) -> str:
    """The pre-registered wording for the honest map, off its pivotal interval."""
    if not np.isfinite([lo, hi]).all():
        return "no interval"
    if -EQUIVALENCE_XG <= lo and hi <= EQUIVALENCE_XG:
        return "flat"
    return "not distinguishable from zero" if lo <= 0 <= hi else "differs from zero"


def decide(refit: list, stable: dict, clean: bool) -> dict:
    """A1 and A2 by the rules in this module's docstring. A decisive bound that changes side
    across the stability seeds does not decide, whatever seed 0 says."""
    by = {r["key"]: r for r in refit}
    out = {}
    for name, key, needs_clean in (("A1", "D1", True), ("A2", "D2", False)):
        below, steady = by[key]["below_zero"], stable.get(key, True)
        state = ("unstable: does not decide" if not steady else
                 "true" if below and (clean or not needs_clean) else "false")
        out[name] = {"state": state, "by": key,
                     "bonferroni_below_zero": bool(by[key]["bonf_hi"] < 0)}
    return out


def run_sample(t: pd.DataFrame, sample: str, args: argparse.Namespace, out_dir: Path,
               tail: str) -> list:
    """Population, readability, fixed and refit statistics for one sample; stability, control,
    the league-in-design reading, robustness and the verdicts only where it decides."""
    confirm = sample == "confirmation"
    numeric = SPOT[args.kind]
    stat = partial(statistics, numeric=numeric)
    counts = t["zone"].value_counts().to_dict()
    print(f"\n{sample}: {len(t)} long corners / {t['match_id'].nunique()} matches, attack-won "
          f"{int(t['won'].sum())}; end zones {counts}", flush=True)
    rows = [{"sample": sample, "block": "population", "key": "long_corners", "n": len(t),
             "point": float(t["won"].mean()), "params": str(sorted(t["league"].unique()))}]
    rows += [{"sample": sample, "block": "zone_count", "key": k, "n": int(v)}
             for k, v in counts.items()]
    if t["match_id"].nunique() < _MIN_MATCHES:
        print(f"  skipped: under {_MIN_MATCHES} matches", flush=True)
        return rows + [{"sample": sample, "block": "skipped", "key": "too_few_matches"}]
    rows += aim_readability(t, sample, numeric)
    fit = fit_aim(t, "base", (), numeric)
    fixed = fixed_intervals(t, "base", "xg", args.boot, fit)
    rows += by_first_contact(t, sample, args.boot, fit)

    n0 = max(args.refit_boot, args.stability_boot if confirm and args.stability_seeds else 0)
    tag = f"{tail}_{sample}"
    point, d0 = seed_draws(t, stat, n0, 0, args.jobs,
                           out_dir / f"aim_zone_draws{tag}_seed0.npy", args.resume)
    expected = np.array([fixed[k][0] for k in STATS])
    if point is not None and not np.allclose(point, expected, rtol=0.0, atol=1e-9,
                                             equal_nan=True):
        raise SystemExit(f"STOP: the refit point {point} does not reproduce the fixed fit "
                         f"{expected}")
    print(f"REFIT ({sample}, {args.refit_boot} draws):", flush=True)
    ref = refit_rows(expected, d0[:args.refit_boot], fixed, sample)
    rows += ref
    honest = next(r for r in ref if r["key"] == "honest")
    rows.append({"sample": sample, "block": "equivalence", "key": "honest",
                 "margin": EQUIVALENCE_XG, "params": equivalence(honest["lo"], honest["hi"])})
    if not confirm:
        return rows

    stable = {}
    if args.stability_seeds:
        his = {0: _pivotal(expected, d0[:args.stability_boot], 0.05)[1]}
        for sd in range(1, args.stability_seeds):
            d_sd = seed_draws(t, stat, args.stability_boot, sd, args.jobs,
                              out_dir / f"aim_zone_draws{tag}_seed{sd}.npy", args.resume)[1]
            his[sd] = _pivotal(expected, d_sd, 0.05)[1]
        for key in AIM_DECISIONS:
            i = STATS.index(key)
            stable[key] = len({bool(his[sd][i] < 0) for sd in his}) == 1
            print(f"  {key} pivotal upper bound by seed "
                  f"{[round(float(his[sd][i]), 5) for sd in his]}"
                  f"{'' if stable[key] else '  CHANGES SIDE'}", flush=True)
            rows += [{"sample": sample, "block": "stability", "key": key, "seed": sd,
                      "n": args.stability_boot, "hi": float(his[sd][i]),
                      "below_zero": bool(his[sd][i] < 0)} for sd in his]

    ctrl = shuffled_control(t, args.boot, numeric)
    print(f"  shuffled kick block, D1: {ctrl['point']:+.5f} [{ctrl['lo']:+.5f}, {ctrl['hi']:+.5f}]"
          f"  {'clean' if ctrl['clean'] else 'LEAKAGE SUSPECTED'}", flush=True)
    rows.append({"sample": sample, "block": "control", "key": "D1/shuffled_kick_block", **ctrl})
    # Pooled leagues differ in zone base rates and possibly in how the 360 is tagged, and the
    # design carries no league, so the kick block could buy D1 by naming the league.
    net = fixed_intervals(t, "base", "xg", args.boot, extra=("league",), numeric=numeric)
    print("  league in both arms' design: " + "  ".join(f"{k} {net[k][0]:+.5f}" for k in STATS)
          + f"; D1 fixed [{net['D1'][1]:+.5f}, {net['D1'][2]:+.5f}]", flush=True)
    rows += [{"sample": sample, "block": "league_in_design", "key": k, "n": len(t),
              "point": net[k][0], "lo": net[k][1], "hi": net[k][2]} for k in STATS]
    rows += robustness(t, sample, args.boot, args.jobs, numeric)
    for name, v in decide(ref, stable, ctrl["clean"]).items():
        print(f"  {name}: {v['state']}  (Bonferroni below zero: {v['bonferroni_below_zero']})",
              flush=True)
        rows.append({"sample": sample, "block": "verdict", "key": name, "params": str(v)})
    return rows


def main() -> None:
    """Build the population, then the confirmation sample, the development one and each league."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None,
                    help="first N matches: the path, not the numbers")
    ap.add_argument("--boot", type=int, default=_BOOT, help="fixed-fit cluster-bootstrap draws")
    ap.add_argument("--refit-boot", type=int, default=_REFIT_BOOT, help="refit draws for the verdicts")
    ap.add_argument("--stability-seeds", type=int, default=_SEEDS)
    ap.add_argument("--stability-boot", type=int, default=_STABILITY_BOOT)
    ap.add_argument("--resume", action="store_true", help="reuse the per-seed draws already saved")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="where the artefacts land (default: data/derived)")
    ap.add_argument("--kind", choices=tuple(SPOT), default="corner",
                    help="the population: corners, or the flank free kicks from x >= 85")
    args = ap.parse_args()
    clock = time.time()
    assert_sources()
    out_dir = Path(args.out_dir) if args.out_dir else _DER
    kt = kind_tag((args.kind,))
    tail = f"_limit{args.limit}" if args.limit else ""
    precision = (args.boot, args.refit_boot, args.stability_seeds, args.stability_boot)
    if published_dir(out_dir) and not tail and precision != (_BOOT, _REFIT_BOOT, _SEEDS,
                                                             _STABILITY_BOOT):
        raise SystemExit(f"STOP: A1 and A2 are registered at {_REFIT_BOOT} refit draws and "
                         f"{_SEEDS} seeds of {_STABILITY_BOOT}, so a run at {precision} may not write "
                         f"the published aim_zone{kt}.parquet; pass --out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {"corners": _DER / f"corners{kt}{tail}.parquet",
             "eda_deliveries": _DER / f"eda_deliveries{tail}.parquet",
             "shots": _DER / f"shots{kt}{tail}.parquet"}
    for p in paths.values():
        if not p.exists():
            raise SystemExit(f"STOP: {p.name} absent; build_table.py and eda_build.py write it")
    corners = pd.read_parquet(paths["corners"])
    corners = corners[corners["set_piece_type"] == args.kind]
    block = build(corners, args.jobs, args.kind)
    t = population(corners, pd.read_parquet(paths["eda_deliveries"]),
                   pd.read_parquet(paths["shots"]), block, args.kind)

    dev = t["league"].isin(DEVELOPMENT)
    rows = run_sample(t[~dev].reset_index(drop=True), "confirmation", args, out_dir, kt + tail)
    rows += run_sample(t[dev].reset_index(drop=True), "development", args, out_dir, kt + tail)
    print("\nBY LEAGUE (fixed fit, reported):", flush=True)
    subs = {lg: s.reset_index(drop=True) for lg, s in t.groupby("league")
            if s["match_id"].nunique() >= _MIN_MATCHES}
    fitted = Parallel(n_jobs=max(1, min(args.jobs, len(subs))))(
        delayed(fixed_intervals)(s, "base", "xg", args.boot, None, (), SPOT[args.kind])
        for s in subs.values())
    for (lg, sub), res in zip(subs.items(), fitted, strict=True):
        print(f"  {lg:15s} " + "  ".join(f"{k} {res[k][0]:+.4f}" for k in STATS), flush=True)
        rows += [{"sample": lg, "block": "league", "key": k, "n": len(sub), "point": res[k][0],
                  "lo": res[k][1], "hi": res[k][2]} for k in STATS]

    out = out_dir / f"aim_zone{kt}{tail}.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"\nwrote {out}  [{time.time() - clock:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
