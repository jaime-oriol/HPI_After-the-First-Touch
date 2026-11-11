"""run_zone_g — does Opta Analyst's cut of the box read the plan better than C?

Pre-registered (data/derived/prereg_2026-10-13/zones_g.md) before G1 or G2 was scored.
Two rivals against C by M1 as zones.md defines it, in a driver of its own like run_zone_s: it reuses
run_zone_comparison's builders and readings and runs its own arms, so the registered runs are untouched.

G1 is Opta's box as its chart draws it: the six-yard box's width in three equal columns over three rows
(the six-yard line, the penalty spot's line, the front of the box), one cell on each side of it over the
whole depth, and C's `edge`, `wide` and `deep` outside. G2 runs the middle row on to 15 yd off the goal
line and gives the central block behind it to the `edge`, 12 cells against 11.

M1 FOR G. On the attack-won long corners, the out-of-fold zone log loss of s02b_aim's aim model on the
delivery design plus one block, each block counting the outfield attackers holding a job and the
outfield defenders per zone, both sides on one grid; Δ_R = loss(R) − loss(C), row by row. R replaces C
only if Δ_R's refit interval in normal form at 99% lies wholly below zero, its upper bound keeps that
side at all five stability seeds of 2000 match resamples, R's block shuffled over every row does not
beat the design, and the three readings C passed against F hold for R against C. Corners decide; flank
free kicks replicate without stability seeds.

THE SCREEN (--screen) is the sheet's stage 1: conditions 1 and 3 at one seed of 300 match resamples,
with everything reported beside them, to say whether the full run is worth its hours. It replaces
nothing. Reported, never deciding: each block against the design, the outcome step on s03's five
classes, the pivotal and percentile forms, the MDE at 80%, each league alone and left out, the
attacking team and the match whose leaving moves Δ_R furthest, and the sanity count.

    python pipeline/run_zone_g.py --counts-only                   # the sanity count, no model
    python pipeline/run_zone_g.py --timing --draws 46 --seeds 1 --refit-boot 23 \\
        --out-dir data/derived/smoke                              # cost per arm and draw, no score read
    python pipeline/run_zone_g.py --screen --draws 300 --out-dir data/derived/zones_g
    python pipeline/run_zone_g.py --jobs 23                       # the full run, after a go
"""
from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))
_DER = _REPO / "data" / "derived"

from build_table import published_dir                                         # noqa: E402
from constants import GOALMOUTH_X, HOLD_X, SIX_YARD_X, SIX_YARD_Y              # noqa: E402
from run_aim_zone import _MIN_MATCHES                                         # noqa: E402
from run_first_contact import refit_draws, seed_draws                         # noqa: E402
from run_validation import _tuned                                             # noqa: E402
from run_zone_comparison import (_BOOT, _REFIT_BOOT, _SEEDS, _STABILITY_BOOT,  # noqa: E402
                                 BONF_ALPHA, ROBUSTNESS, assert_sources, build, design,
                                 fixed, forms, inputs, population, refit_rows)
from s02b_aim import GRIDS, aim_oof, zone_loss                                # noqa: E402
from s03_value import _KICK_INTRINSIC, CompetingOutcomesModel                 # noqa: E402
from validate import cluster_bootstrap_mean, shuffle_features                 # noqa: E402
from zones import ZONES, cells, in_box, in_width, zones_g2                    # noqa: E402

G_RIVALS = ("G1", "G2")
G_ARMS = ("C", *G_RIVALS)
_PARTITIONS = 5
# The sheet's borderline: a bound this close to zero, as a share of its half-width, is within the sampling
# noise of an SD taken from 300 draws, 1/sqrt(2 * 299).
_BORDERLINE = 0.05
_U0, _U1 = SIX_YARD_Y                                   # the six-yard box's width, off the posts
_T1, _T2 = _U0 + (_U1 - _U0) / 3, _U0 + 2 * (_U1 - _U0) / 3   # Opta's thirds, equal
G1_ZONES = ("six_near", "six_central", "six_far", "spot_near", "spot_central", "spot_far",
            "front_central", "near_side", "far_side", *ZONES["C"][-3:])
# G2 is the switch's partition, so its labeller and its zones are src/zones.py's.
G2_ZONES = ZONES["G2"]


def keys(base: str) -> tuple:
    """Contrast names: each rival against C, then each arm against `base`."""
    return (*(f"{r}-C" for r in G_RIVALS), *(f"{a}-{base}" for a in G_ARMS))


G_KEYS, G_OUTCOME_KEYS = keys("design"), keys("delivery")


# ── The partitions and their blocks ──────────────────────────────────────
def _thirds(y: np.ndarray) -> np.ndarray:
    return np.where(y < _T1, "near", np.where(y < _T2, "central", "far"))


def _outside(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """C's three cells outside the box: `deep` short of HOLD_X, `edge` in front of the box inside its
    width, `wide` beside it."""
    return np.where(x < HOLD_X, "deep", np.where(in_width(y), "edge", "wide"))


def zones_g1(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Opta's box as drawn: thirds of the six-yard width by row, the front row one cell, and a cell
    on each side over the whole depth."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    row = np.where(x >= SIX_YARD_X, "six", np.where(x >= GOALMOUTH_X, "spot", "front"))
    cell = np.where(row == "front", "front_central", np.char.add(np.char.add(row, "_"), _thirds(y)))
    inner = np.where((y >= _U0) & (y <= _U1), cell, np.where(y < _U0, "near_side", "far_side"))
    return np.where(in_box(x, y), inner, _outside(x, y)).astype(object)


def partitions_of() -> dict:
    """{arm: (labeller on canonical points, its zones)}: C's twelve cells and the two rivals."""
    return {"C": (cells, ZONES["C"]), "G1": (zones_g1, G1_ZONES), "G2": (zones_g2, G2_ZONES)}


def columns(parts: dict) -> dict:
    return {a: [f"{a}_{s}_{z}" for s in ("att", "def") for z in zs] for a, (_, zs) in parts.items()}


def blocks(bodies: pd.DataFrame, ids: pd.Series, parts: dict) -> pd.DataFrame:
    """One row per delivery of `ids`, in that order: each arm's bodies per side and zone. A label a
    partition does not declare stops the run."""
    x, y = bodies["x"].to_numpy(float), bodies["y"].to_numpy(float)
    key = [bodies["set_piece_id"].to_numpy(), bodies["side"].to_numpy()]
    out = []
    for arm, (labeller, zs) in parts.items():
        lab = np.asarray(labeller(x, y), dtype=object)
        if undeclared := set(lab) - set(zs):
            raise ValueError(f"{arm} emitted zones it does not declare: {sorted(undeclared)}")
        one = (lab[:, None] == np.asarray(zs, dtype=object)[None, :]).astype(float)
        w = pd.DataFrame(one, columns=list(zs)).groupby(key).sum().unstack(level=1, fill_value=0.0)
        w.columns = [f"{arm}_{s}_{z}" for z, s in w.columns]
        out.append(w)
    cols = sum(columns(parts).values(), [])
    return (pd.concat(out, axis=1).reindex(index=ids.to_numpy(), columns=cols).fillna(0.0)
            .reset_index(drop=True))


def join(t: pd.DataFrame, bodies: pd.DataFrame, parts: dict) -> pd.DataFrame:
    """The population with every arm's counts recounted from the frames; the table's own copies of
    C's counts leave first."""
    cols = sum(columns(parts).values(), [])
    return pd.concat([t.drop(columns=[c for c in cols if c in t.columns]),
                      blocks(bodies, t["set_piece_id"], parts)], axis=1)


def sanity(t: pd.DataFrame, bodies: pd.DataFrame, pop: str, cols: dict) -> list:
    """Mean attackers and defenders per zone per delivery under each arm, before any model. Stops if
    a block loses or invents a body."""
    n = (bodies.groupby(["set_piece_id", "side"]).size().unstack(fill_value=0)
         .reindex(index=t["set_piece_id"], columns=["att", "def"], fill_value=0))
    rows = []
    for arm, cs in cols.items():
        for s in ("att", "def"):
            side = [c for c in cs if c.startswith(f"{arm}_{s}_")]
            gap = float(np.abs(t[side].sum(axis=1).to_numpy() - n[s].to_numpy()).max())
            if gap > 1e-9:
                raise SystemExit(f"STOP: {arm} loses or invents {s} bodies (largest gap {gap})")
            rows += [{"population": pop, "block": "sanity", "key": c, "n": len(t),
                      "point": float(t[c].mean()), "share": float((t[c] >= 0.5).mean())} for c in side]
    print(f"\nSANITY COUNT, {pop}: {len(t)} deliveries, attack won the first contact on "
          f"{int(t['won'].sum())}; every body counted once under C, G1 and G2", flush=True)
    return rows


# ── The models ───────────────────────────────────────────────────────────
def plan_losses(t: pd.DataFrame, cols: dict, shuffle_seed: int | None = None,
                arms: tuple = ("design", *G_ARMS), grid: str = "base", extra: tuple = ()) -> dict:
    """Per-row out-of-fold zone log loss on the attack-won rows, for the design alone and the design
    plus each block, every aim model rebuilt on `t`; the aim is the end zone under `grid`."""
    zones = GRIDS[grid]
    z = t[f"zone_{grid}"].map({k: i for i, k in enumerate(zones)}).to_numpy()
    won, g = t["won"].to_numpy(bool), t["match_id"].to_numpy()
    D = design(t, extra)
    X = {"design": D, **{a: np.c_[D, t[cols[a]].to_numpy(float)] for a in G_ARMS}}
    return {a: zone_loss(aim_oof(X[a], z, won, g, len(zones), shuffle_seed), z)[won] for a in arms}


def diffs(loss: dict, base: str) -> dict:
    """Each rival against C, then each arm against `base`, per row."""
    return {**{f"{r}-C": loss[r] - loss["C"] for r in G_RIVALS},
            **{f"{a}-{base}": loss[a] - loss[base] for a in G_ARMS}}


def plan_stats(t: pd.DataFrame, cols: dict, shuffle_seed: int | None = None, grid: str = "base",
               extra: tuple = ()) -> np.ndarray:
    """G_KEYS on one sample."""
    loss = plan_losses(t, cols, shuffle_seed, grid=grid, extra=extra)
    return np.array([d.mean() for d in diffs(loss, "design").values()])


def outcome_losses(t: pd.DataFrame, m: CompetingOutcomesModel, cols: dict) -> dict:
    """Per-row out-of-fold five-class log loss of s03's delivery rung and of the rung plus each block,
    at m's hyperparameters."""
    y = t["target"].to_numpy()
    arms = {"delivery": [], **{a: cols[a] for a in G_ARMS}}
    return {a: m._row_logloss(m.fit_cv(t, numeric=_KICK_INTRINSIC + c), y) for a, c in arms.items()}


def _outcome_stats(t: pd.DataFrame, m: CompetingOutcomesModel, cols: dict) -> np.ndarray:
    return np.array([d.mean() for d in diffs(outcome_losses(t, m, cols), "delivery").values()])


# ── Readings ─────────────────────────────────────────────────────────────
def shuffled_control(t: pd.DataFrame, cols: dict, pop: str, boot: int) -> tuple[list, dict]:
    """Each block permuted jointly over every row against the design alone, on the fixed fit. A
    shuffled block that still beats the design reads row structure no plan carries."""
    g = t.loc[t["won"], "match_id"].to_numpy()
    base = plan_losses(t, cols, arms=("design",))["design"]
    rows, clean = [], {}
    for a in G_ARMS:
        s = shuffle_features(t, cols[a], mode="global", seed=0)
        ci = cluster_bootstrap_mean(plan_losses(s, cols, arms=(a,))[a] - base, g, n_boot=boot)
        clean[a] = bool(ci.hi >= 0)
        rows.append({"population": pop, "block": "control", "key": f"{a}shuffled-design", "n": len(g),
                     "point": ci.point, "lo": ci.lo, "hi": ci.hi, "clean": clean[a]})
        print(f"  shuffled {a}: {ci.point:+.5f} [{ci.lo:+.5f}, {ci.hi:+.5f}] "
              f"{'clean' if clean[a] else 'BEATS THE DESIGN'}", flush=True)
    return rows, clean


def influence(t: pd.DataFrame, pop: str, loss: dict) -> list:
    """The attacking team, and the match, whose leaving moves each Δ_R furthest toward zero and past it."""
    w, rows = t.loc[t["won"]], []
    for r in G_RIVALS:
        d = loss[r] - loss["C"]
        for unit in ("attacking_team", "match_id"):
            s = pd.Series(d).groupby(w[unit].to_numpy()).agg(["sum", "count"])
            loo = (d.sum() - s["sum"]) / (len(d) - s["count"])
            j = (-np.sign(d.mean()) * (loo - d.mean())).idxmax()
            rows.append({"population": pop, "block": "influence", "key": f"{r}-C", "unit": unit,
                         "who": str(j), "point": float(d.mean()), "without": float(loo[j]),
                         "flips": bool(np.sign(loo[j]) != np.sign(d.mean()))})
    return rows


def _league_cell(t: pd.DataFrame, cols: dict, league: str, alone: bool) -> dict:
    sub = t[(t["league"] == league) == alone].reset_index(drop=True)
    return diffs(plan_losses(sub, cols), "design") if alone else dict(zip(
        G_KEYS, plan_stats(sub, cols), strict=True))


def by_league(t: pd.DataFrame, cols: dict, pop: str, boot: int, jobs: int) -> list:
    """Each league alone, with match-clustered 95% intervals on its fixed fit, and each left out."""
    cells_ = [(lg, alone) for lg in sorted(t["league"].unique()) for alone in (True, False)]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        out = Parallel(n_jobs=min(jobs, len(cells_)))(
            delayed(_league_cell)(t, cols, lg, alone) for lg, alone in cells_)
    rows = []
    for (lg, alone), res in zip(cells_, out, strict=True):
        if alone:
            g = t.loc[(t["league"] == lg) & t["won"], "match_id"].to_numpy()
            rows += fixed(pop, res, g, boot, block=f"league/{lg}")[0]
        else:
            rows += [{"population": pop, "block": f"league_out/{lg}", "key": k, "point": float(v)}
                     for k, v in res.items()]
    return rows


def other_partitions(t: pd.DataFrame, cols: dict, pop: str, point: np.ndarray, jobs: int) -> list:
    """Every contrast under grouped partitions dealt at random: how much of each point is the one
    partition GroupKFold draws."""
    with parallel_config(backend="loky", inner_max_num_threads=1):
        out = Parallel(n_jobs=min(jobs, _PARTITIONS))(
            delayed(plan_stats)(t, cols, s) for s in range(1, _PARTITIONS + 1))
    v = np.array(out)
    return [{"population": pop, "block": "partitions", "key": k, "n": len(v), "point": point[i],
             "lo": float(v[:, i].min()), "hi": float(v[:, i].max()), "mean": float(v[:, i].mean())}
            for i, k in enumerate(G_KEYS)]


def outcome_step(t: pd.DataFrame, cols: dict, pop: str, args: argparse.Namespace,
                 refit: bool = True) -> list:
    """s03's delivery rung against the rung plus each block on the five-class target, tuned as
    run_validation tunes and held fixed across the arms and the refit draws; the screen keeps the
    fixed fit and skips the refit."""
    m = _tuned(t, True, args.jobs)
    rows, fx = fixed(pop, diffs(outcome_losses(t, m, cols), "delivery"), t["match_id"].to_numpy(),
                     args.boot, block="outcome_fixed")
    if not refit:
        return rows
    expected = np.array([fx[k][0] for k in G_OUTCOME_KEYS])
    point, draws = refit_draws(t, partial(_outcome_stats, m=m, cols=cols), args.refit_boot, 0, args.jobs)
    if not np.allclose(point, expected, rtol=0.0, atol=1e-9):
        raise SystemExit(f"STOP: the outcome refit point {point} does not reproduce {expected}")
    return rows + refit_rows(pop, list(G_OUTCOME_KEYS), expected, draws, fx, block="outcome_refit")


def decide(ref: list, stable: dict, clean: dict, held: dict) -> dict:
    """zones_g.md: a rival replaces C when its 99% normal bound lies below zero, keeps that side at
    every stability seed, its shuffled block does not beat the design and the three readings hold;
    the lowest Δ of several. The first three without the readings go to the author with the numbers.
    `stable` empty means no seed was run, the replication, which M1 reads as not unstable."""
    by = {r["key"]: r for r in ref}
    below = [r for r in G_RIVALS if by[f"{r}-C"]["bonf_hi"] < 0]
    m1 = [r for r in below if stable.get(r, True) and clean[r]]
    ok = [r for r in m1 if all(held[r].values())]
    return {"replaces": min(ok, key=lambda r: by[f"{r}-C"]["point"]) if ok else "C", "m1": m1,
            "below_zero": below, "stable": stable, "seeds_run": bool(stable),
            "control_clean": {r: clean[r] for r in G_RIVALS}, "readings": held,
            "discuss": [r for r in m1 if r not in ok]}


def screen_verdict(ref: list, clean: dict) -> dict:
    """The sheet's stage 1: a rival goes to the full run when its 99% normal upper bound lies below zero
    and its shuffled block does not beat the design; a bound within _BORDERLINE of its half-width of
    zero is borderline, and the screen is repeated at 2000 draws before anything is read from it."""
    by, out = {r["key"]: r for r in ref}, {}
    for r in G_RIVALS:
        row = by[f"{r}-C"]
        half = (row["bonf_hi"] - row["bonf_lo"]) / 2
        out[r] = {"point": float(row["point"]), "bonf_hi": float(row["bonf_hi"]),
                  "mde80": float(row["mde80"]), "below_zero": bool(row["bonf_hi"] < 0),
                  "control_clean": clean[r], "borderline": bool(abs(row["bonf_hi"]) <= _BORDERLINE * half),
                  "go": bool(row["bonf_hi"] < 0 and clean[r])}
    return {"go": [r for r in G_RIVALS if out[r]["go"]], **out}


# ── The run ──────────────────────────────────────────────────────────────
def stability(t: pd.DataFrame, cols: dict, pop: str, point: np.ndarray, d0: np.ndarray,
              args: argparse.Namespace, path: Path) -> tuple[list, dict]:
    """Each Δ_R's 99% normal upper bound at every seed; steady when it keeps its side."""
    his = {0: forms(point, d0, BONF_ALPHA)["n_hi"]}
    stat = partial(plan_stats, cols=cols)
    for s in range(1, args.seeds):
        d = seed_draws(t, stat, args.draws, s, args.jobs,
                       path.with_name(path.name.replace("seed0", f"seed{s}")), args.resume)[1]
        his[s] = forms(point, d, BONF_ALPHA)["n_hi"]
    rows, stable = [], {}
    for i, r in enumerate(G_RIVALS):
        stable[r] = len({bool(h[i] < 0) for h in his.values()}) == 1
        print(f"  {r}-C 99% normal upper bound by seed {[round(float(h[i]), 5) for h in his.values()]}"
              f"{'' if stable[r] else '  CHANGES SIDE'}", flush=True)
        rows += [{"population": pop, "block": "stability", "key": f"{r}-C", "seed": s, "n": args.draws,
                  "hi": float(h[i]), "below_zero": bool(h[i] < 0)} for s, h in his.items()]
    return rows, stable


def readings(t: pd.DataFrame, cols: dict, pop: str, args: argparse.Namespace, out_dir: Path,
             tail: str) -> tuple[list, dict]:
    """The three readings C passed against F, read for each rival against C at one seed each."""
    rows, held = [], {r: {} for r in G_RIVALS}
    g = t.loc[t["won"], "match_id"].to_numpy()
    for name, grid, extra, bound in ROBUSTNESS:
        stat = partial(plan_stats, cols=cols, grid=grid, extra=extra)
        fx_rows, fx = fixed(pop, diffs(plan_losses(t, cols, grid=grid, extra=extra), "design"), g,
                            args.boot, block=f"{name}/fixed")
        point, d = seed_draws(t, stat, args.draws, 0, args.jobs,
                              out_dir / f"zone_g_draws{tail}_{pop}_{name}_seed0.npy", args.resume)
        expected = np.array([fx[k][0] for k in G_KEYS])
        if point is not None and not np.allclose(point, expected, rtol=0.0, atol=1e-9):
            raise SystemExit(f"STOP: {name}: the refit point {point} does not reproduce {expected}")
        ref = refit_rows(pop, list(G_KEYS), expected, d, fx, block=f"{name}/refit")
        for r in G_RIVALS:
            held[r][name] = bool(next(x for x in ref if x["key"] == f"{r}-C")[bound] < 0)
        rows += fx_rows + ref
    return rows, held


def _fixed_and_refit(t: pd.DataFrame, cols: dict, pop: str, args: argparse.Namespace,
                     path: Path) -> tuple[dict, list, list, np.ndarray, np.ndarray]:
    """The decision's contrasts on the fixed fit and on one seed of refit draws, which must reproduce
    the fixed fit's points."""
    loss = plan_losses(t, cols)
    rows, fx = fixed(pop, diffs(loss, "design"), t.loc[t["won"], "match_id"].to_numpy(), args.boot)
    point, d0 = seed_draws(t, partial(plan_stats, cols=cols), args.draws, 0, args.jobs, path, args.resume)
    expected = np.array([fx[k][0] for k in G_KEYS])
    if point is not None and not np.allclose(point, expected, rtol=0.0, atol=1e-9):
        raise SystemExit(f"STOP: the refit point {point} does not reproduce the fixed fit {expected}")
    return loss, rows, refit_rows(pop, list(G_KEYS), expected, d0, fx), expected, d0


def screen(t: pd.DataFrame, cols: dict, pop: str, args: argparse.Namespace, out_dir: Path,
           tail: str) -> list:
    """The sheet's stage 1 on one population: conditions 1 and 3 at one seed, and the cheap readings."""
    loss, rows, ref, _, _ = _fixed_and_refit(t, cols, pop, args, out_dir / f"zone_g_draws{tail}_{pop}_seed0.npy")
    ctrl, clean = shuffled_control(t, cols, pop, args.boot)
    rows += ref + ctrl + influence(t, pop, loss) + by_league(t, cols, pop, args.boot, args.jobs)
    rows += outcome_step(t, cols, pop, args, refit=False)
    verdict = screen_verdict(ref, clean)
    print(f"  SCREEN ({pop}): {verdict}", flush=True)
    return rows + [{"population": pop, "block": "verdict", "key": "screen", "params": str(verdict)}]


def compare(t: pd.DataFrame, cols: dict, pop: str, args: argparse.Namespace, out_dir: Path,
            tail: str, seeds: bool) -> list:
    """M1 for both rivals against C on one population, with every control and reported reading, the
    three readings and the verdict; stability seeds only where it decides."""
    path = out_dir / f"zone_g_draws{tail}_{pop}_seed0.npy"
    loss, rows, ref, expected, d0 = _fixed_and_refit(t, cols, pop, args, path)
    stab, stable = stability(t, cols, pop, expected, d0, args, path) if seeds else ([], {})
    ctrl, clean = shuffled_control(t, cols, pop, args.boot)
    read, held = readings(t, cols, pop, args, out_dir, tail)
    rows += ref + stab + ctrl + read + influence(t, pop, loss)
    rows += by_league(t, cols, pop, args.boot, args.jobs)
    rows += other_partitions(t, cols, pop, expected, args.jobs) + outcome_step(t, cols, pop, args)
    verdict = decide(ref, stable, clean, held)
    print(f"  G against C ({pop}{'' if seeds else ', replication: no stability seeds'}): {verdict}",
          flush=True)
    return rows + [{"population": pop, "block": "verdict", "key": "G", "params": str(verdict)}]


def timing(t: pd.DataFrame, cols: dict, pop: str, args: argparse.Namespace) -> list:
    """What each arm and each resample costs, measured blind: every score is dropped unread."""
    rows = []
    with threadpool_limits(1):
        for a in ("design", *G_ARMS):
            t0 = time.time()
            plan_losses(t, cols, arms=(a,))
            rows.append({"population": pop, "what": f"aim fit, {a}", "seconds": time.time() - t0})
    t0 = time.time()
    refit_draws(t, partial(plan_stats, cols=cols), args.draws, 0, args.jobs)
    rows.append({"population": pop, "what": f"aim refit, {args.draws} draws on {args.jobs} jobs",
                 "seconds": time.time() - t0})
    t0 = time.time()
    m = _tuned(t, True, args.jobs)
    rows.append({"population": pop, "what": "outcome tuning", "seconds": time.time() - t0})
    with threadpool_limits(1):
        t0 = time.time()
        outcome_losses(t, m, cols)
        rows.append({"population": pop, "what": "outcome fit, 4 arms", "seconds": time.time() - t0})
    for r in rows:
        print(f"  {pop:16s} {r['what']:42s} {r['seconds']:9.1f} s", flush=True)
    return rows


def main() -> None:
    """The frames and populations, every arm's counts and the sanity count; then the timing, the
    screen or the comparison."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=23)
    ap.add_argument("--limit", type=int, default=None, help="first N matches: the path, not the numbers")
    ap.add_argument("--tables", type=str, default=str(_DER))
    ap.add_argument("--out-dir", type=str, default=str(_DER))
    ap.add_argument("--boot", type=int, default=_BOOT, help="fixed-fit cluster-bootstrap draws")
    ap.add_argument("--refit-boot", type=int, default=_REFIT_BOOT, help="refit draws, outcome step")
    ap.add_argument("--seeds", type=int, default=_SEEDS, help="stability seeds, corners")
    ap.add_argument("--draws", type=int, default=_STABILITY_BOOT, help="refit draws per seed")
    ap.add_argument("--counts-only", action="store_true", help="the sanity count, no model")
    ap.add_argument("--timing", action="store_true", help="cost per arm and draw, no score kept")
    ap.add_argument("--screen", action="store_true", help="the sheet's stage 1, at the precision given")
    ap.add_argument("--resume", action="store_true", help="reuse the per-seed draws already saved")
    args = ap.parse_args()
    clock = time.time()
    assert_sources()
    out_dir = Path(args.out_dir)
    tail = f"_limit{args.limit}" if args.limit else ""
    precision = (args.boot, args.refit_boot, args.seeds, args.draws)
    if not tail and published_dir(out_dir) and not (args.counts_only or args.timing or args.screen) \
            and precision != (_BOOT, _REFIT_BOOT, _SEEDS, _STABILITY_BOOT):
        raise SystemExit(f"STOP: G against C is registered at {_SEEDS} seeds of {_STABILITY_BOOT} "
                         f"draws, so a run at {precision} may not write the published record; "
                         f"pass --out-dir")
    if not tail and published_dir(out_dir) and Path(args.tables).resolve() != _DER.resolve():
        raise SystemExit("STOP: tables outside data/derived are not the published ones, so their "
                         "readings may not take the published names; pass --out-dir")
    parts = partitions_of()
    cols = columns(parts)
    out_dir.mkdir(parents=True, exist_ok=True)
    grain, tables = inputs(Path(args.tables), tail)
    frames, rows = {}, []
    for pop, table in tables.items():
        info, bodies = build(table, pop, args.jobs)
        t = population(table, info, grain[grain["set_piece_type"] == pop])
        bodies = bodies[bodies["set_piece_id"].isin(t["set_piece_id"])]
        t = join(t, bodies, parts)
        frames[pop] = t
        rows += sanity(t, bodies, pop, cols)
    pd.DataFrame(rows).to_parquet(out_dir / f"zone_g_counts{tail}.parquet", index=False)
    if args.counts_only:
        print(f"\nwrote {out_dir / f'zone_g_counts{tail}.parquet'}  [{time.time() - clock:.0f}s]")
        return
    fit = {pop: t for pop, t in frames.items() if t.loc[t["won"], "match_id"].nunique() >= _MIN_MATCHES}
    if args.timing:
        cost = [r for pop, t in fit.items() for r in timing(t, cols, pop, args)]
        pd.DataFrame(cost).to_parquet(out_dir / f"zone_g_timing{tail}.parquet", index=False)
        print(f"\nwrote {out_dir / f'zone_g_timing{tail}.parquet'}  [{time.time() - clock:.0f}s]")
        return
    rows = []
    for pop, t in fit.items():
        print(f"\n{'SCREEN' if args.screen else 'G AGAINST C'}, {pop}", flush=True)
        rows += (screen(t, cols, pop, args, out_dir, tail) if args.screen
                 else compare(t, cols, pop, args, out_dir, tail, seeds=pop == "corner"))
    path = out_dir / f"zone_g_{'screen' if args.screen else 'comparison'}{tail}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"\nwrote {path}  [{time.time() - clock:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
