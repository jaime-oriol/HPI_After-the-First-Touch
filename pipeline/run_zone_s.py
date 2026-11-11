"""run_zone_s — does S, the box drawn from where the attackers stand, read the plan better than C?

Pre-registered (data/derived/prereg_2026-10-07/zones_s.md) before S was fitted. S is
fitted once by fit_zones_s and read here from its frozen parameters, and scored against C by M1 as
zones.md defines it, with C the reference and S the one rival. The registered comparison of A to F
is untouched: this driver reuses run_zone_comparison's builders and readings and runs its own arms.

M1 FOR S. On the attack-won long corners, the out-of-fold zone log loss of s02b_aim's aim model on
the delivery design plus one block, each block counting the outfield attackers holding a job and the
outfield defenders per zone, both sides on one grid; Δ_S = loss(S) − loss(C), row by row. S replaces
C only if all four hold: Δ_S's refit interval in normal form at 99% lies wholly below zero, M1's
level kept with one rival; its upper bound keeps that side at all five stability seeds of 2000 match
resamples; S's block shuffled over every row does not beat the design; and the three readings C
passed against F hold for S against C, the league in every arm's design at 99% and the aim on
s02b's fine and coarse grids at 95%, each at one seed. If the first three hold and a reading fails,
the decision goes to the author with the numbers. Corners decide; flank free kicks replicate without
stability seeds, and a contradiction is written down, not resolved by choosing.

Reported, never deciding: each block against the design alone, the outcome step on s03's five
classes, the pivotal and percentile forms, the MDE at 80%, each league alone and left out, the
attacking team and the match whose leaving moves Δ_S furthest, Δ_S under other grouped partitions,
the per-match sign test, and the sanity count.

    python pipeline/run_zone_s.py --counts-only                   # the sanity count, no model
    python pipeline/run_zone_s.py --timing --draws 46 --seeds 1 --refit-boot 23 \\
        --out-dir data/derived/smoke                              # cost per arm and draw, no score read
    python pipeline/run_zone_s.py --jobs 23
"""
from __future__ import annotations

import argparse
import json
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
from fit_zones_s import _PARAMS, label                                        # noqa: E402
from run_aim_zone import _MIN_MATCHES                                         # noqa: E402
from run_first_contact import refit_draws, seed_draws                         # noqa: E402
from run_validation import _tuned                                             # noqa: E402
from run_zone_comparison import (BONF_ALPHA, assert_sources, build, design,  # noqa: E402
                                 fixed, forms, inputs, population, refit_rows)
from s02b_aim import GRIDS, aim_oof, zone_loss                                # noqa: E402
from s03_value import _KICK_INTRINSIC, CompetingOutcomesModel                 # noqa: E402
from validate import cluster_bootstrap_mean, shuffle_features                 # noqa: E402
from zones import ZONES, cells                                                # noqa: E402

COMPARED = ("C", "S")
KEYS = ("S-C", "S-design", "C-design")
OUTCOME_KEYS = ("S-C", "S-delivery", "C-delivery")
# zones.md's precision (M-c), which the sheet keeps: five seeds of 2000 match resamples decide, the
# readings take one seed, the outcome step's refit 300 draws and the fixed-fit intervals 2000.
_BOOT, _REFIT_BOOT, _SEEDS, _STABILITY_BOOT = 2000, 300, 5, 2000
_PARTITIONS = 5
# The three readings C passed against F (zones.md), read for S against C: (name, the grid the aim
# is measured on, columns added to every arm's design, the refit bound that has to stay below zero).
READINGS = (("league_in_design", "base", ("league",), "bonf_hi"),
            ("grid_fine", "fine", (), "hi"), ("grid_coarse", "coarse", (), "hi"))


# ── The two partitions and their blocks ──────────────────────────────────
def partitions_of(params: dict) -> dict:
    """{arm: (labeller on canonical points, its zones)}: C's twelve cells, and S's components in
    the box with C's cells outside it."""
    return {"C": (cells, ZONES["C"]),
            "S": (partial(label, params), tuple(params["names"]) + ZONES["C"][-3:])}


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
    """The population with both arms' counts recounted from the frames; the table's own copies of
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
          f"{int(t['won'].sum())}; every body counted once under C and under S", flush=True)
    return rows


# ── The models ───────────────────────────────────────────────────────────
def plan_losses(t: pd.DataFrame, cols: dict, shuffle_seed: int | None = None,
                arms: tuple = ("design", *COMPARED), grid: str = "base", extra: tuple = ()) -> dict:
    """Per-row out-of-fold zone log loss on the attack-won rows, for the design alone and the design
    plus each block, every aim model rebuilt on `t`; the aim is the end zone under `grid`."""
    zones = GRIDS[grid]
    z = t[f"zone_{grid}"].map({k: i for i, k in enumerate(zones)}).to_numpy()
    won, g = t["won"].to_numpy(bool), t["match_id"].to_numpy()
    D = design(t, extra)
    X = {"design": D, **{a: np.c_[D, t[cols[a]].to_numpy(float)] for a in COMPARED}}
    return {a: zone_loss(aim_oof(X[a], z, won, g, len(zones), shuffle_seed), z)[won] for a in arms}


def diffs(loss: dict, base: str) -> dict:
    """S against C, then each block against `base`, per row."""
    return {"S-C": loss["S"] - loss["C"], f"S-{base}": loss["S"] - loss[base],
            f"C-{base}": loss["C"] - loss[base]}


def plan_stats(t: pd.DataFrame, cols: dict, shuffle_seed: int | None = None, grid: str = "base",
               extra: tuple = ()) -> np.ndarray:
    """KEYS on one sample."""
    loss = plan_losses(t, cols, shuffle_seed, grid=grid, extra=extra)
    return np.array([d.mean() for d in diffs(loss, "design").values()])


def outcome_losses(t: pd.DataFrame, m: CompetingOutcomesModel, cols: dict) -> dict:
    """Per-row out-of-fold five-class log loss of s03's delivery rung and of the rung plus each block,
    at m's hyperparameters."""
    y = t["target"].to_numpy()
    arms = {"delivery": [], **{a: cols[a] for a in COMPARED}}
    return {a: m._row_logloss(m.fit_cv(t, numeric=_KICK_INTRINSIC + c), y) for a, c in arms.items()}


# ── Readings ─────────────────────────────────────────────────────────────
def shuffled_control(t: pd.DataFrame, cols: dict, pop: str, boot: int) -> tuple[list, dict]:
    """Each block permuted jointly over every row against the design alone, on the fixed fit. A
    shuffled block that still beats the design reads row structure no plan carries."""
    g = t.loc[t["won"], "match_id"].to_numpy()
    base = plan_losses(t, cols, arms=("design",))["design"]
    rows, clean = [], {}
    for a in COMPARED:
        s = shuffle_features(t, cols[a], mode="global", seed=0)
        ci = cluster_bootstrap_mean(plan_losses(s, cols, arms=(a,))[a] - base, g, n_boot=boot)
        clean[a] = bool(ci.hi >= 0)
        rows.append({"population": pop, "block": "control", "key": f"{a}shuffled-design", "n": len(g),
                     "point": ci.point, "lo": ci.lo, "hi": ci.hi, "clean": clean[a]})
        print(f"  shuffled {a}: {ci.point:+.5f} [{ci.lo:+.5f}, {ci.hi:+.5f}] "
              f"{'clean' if clean[a] else 'BEATS THE DESIGN'}", flush=True)
    return rows, clean


def influence(t: pd.DataFrame, pop: str, loss: dict) -> list:
    """The attacking team, and the match, whose leaving moves Δ_S furthest toward zero and past it."""
    w, d, rows = t.loc[t["won"]], loss["S"] - loss["C"], []
    for unit in ("attacking_team", "match_id"):
        s = pd.Series(d).groupby(w[unit].to_numpy()).agg(["sum", "count"])
        loo = (d.sum() - s["sum"]) / (len(d) - s["count"])
        j = (-np.sign(d.mean()) * (loo - d.mean())).idxmax()
        rows.append({"population": pop, "block": "influence", "key": "S-C", "unit": unit,
                     "who": str(j), "point": float(d.mean()), "without": float(loo[j]),
                     "flips": bool(np.sign(loo[j]) != np.sign(d.mean()))})
    return rows


def _league_cell(t: pd.DataFrame, cols: dict, league: str, alone: bool) -> dict:
    sub = t[(t["league"] == league) == alone].reset_index(drop=True)
    return diffs(plan_losses(sub, cols), "design") if alone else dict(zip(
        KEYS, plan_stats(sub, cols), strict=True))


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
            for i, k in enumerate(KEYS)]


def outcome_step(t: pd.DataFrame, cols: dict, pop: str, args: argparse.Namespace) -> list:
    """s03's delivery rung against the rung plus each block on the five-class target, tuned as
    run_validation tunes and held fixed across the arms and the refit draws."""
    m = _tuned(t, True, args.jobs)
    d = outcome_losses(t, m, cols)
    rows, fx = fixed(pop, {"S-C": d["S"] - d["C"], "S-delivery": d["S"] - d["delivery"],
                           "C-delivery": d["C"] - d["delivery"]}, t["match_id"].to_numpy(),
                     args.boot, block="outcome_fixed")
    expected = np.array([fx[k][0] for k in OUTCOME_KEYS])

    def stat(s: pd.DataFrame, m: CompetingOutcomesModel = m) -> np.ndarray:
        o = outcome_losses(s, m, cols)
        return np.array([(o["S"] - o["C"]).mean(), (o["S"] - o["delivery"]).mean(),
                         (o["C"] - o["delivery"]).mean()])

    point, draws = refit_draws(t, stat, args.refit_boot, 0, args.jobs)
    if not np.allclose(point, expected, rtol=0.0, atol=1e-9):
        raise SystemExit(f"STOP: the outcome refit point {point} does not reproduce {expected}")
    return rows + refit_rows(pop, list(OUTCOME_KEYS), expected, draws, fx, block="outcome_refit")


def decide(ref: list, stable: bool | None, clean: dict, held: dict) -> dict:
    """zones_s.md: S replaces C when Δ_S's 99% normal bound lies below zero, keeps that side at every
    stability seed, S's shuffled block does not beat the design and the three readings hold. The
    first three without the readings go to the author with the numbers. `stable` None means no seed was
    run, the replication, which M1 reads as not unstable (run_zone_comparison.decide)."""
    below = bool(next(r for r in ref if r["key"] == "S-C")["bonf_hi"] < 0)
    m1 = below and stable is not False and clean["S"]
    return {"replaces": m1 and all(held.values()), "m1": m1, "below_zero": below, "stable": stable,
            "seeds_run": stable is not None, "control_clean": clean["S"], "readings": held,
            "discuss": m1 and not all(held.values())}


# ── The run ──────────────────────────────────────────────────────────────
def stability(t: pd.DataFrame, cols: dict, pop: str, point: np.ndarray, d0: np.ndarray,
              args: argparse.Namespace, path: Path) -> tuple[list, bool]:
    """Δ_S's 99% normal upper bound at every seed; steady when it keeps its side."""
    his = {0: float(forms(point, d0, BONF_ALPHA)["n_hi"][0])}
    stat = partial(plan_stats, cols=cols)
    for s in range(1, args.seeds):
        d = seed_draws(t, stat, args.draws, s, args.jobs,
                       path.with_name(path.name.replace("seed0", f"seed{s}")), args.resume)[1]
        his[s] = float(forms(point, d, BONF_ALPHA)["n_hi"][0])
    stable = len({h < 0 for h in his.values()}) == 1
    print(f"  S-C 99% normal upper bound by seed {[round(h, 5) for h in his.values()]}"
          f"{'' if stable else '  CHANGES SIDE'}", flush=True)
    return [{"population": pop, "block": "stability", "key": "S-C", "seed": s, "n": args.draws,
             "hi": h, "below_zero": bool(h < 0)} for s, h in his.items()], stable


def readings(t: pd.DataFrame, cols: dict, pop: str, args: argparse.Namespace, out_dir: Path,
             tail: str) -> tuple[list, dict]:
    """The three readings C passed against F, read for S against C at one seed each."""
    rows, held = [], {}
    g = t.loc[t["won"], "match_id"].to_numpy()
    for name, grid, extra, bound in READINGS:
        stat = partial(plan_stats, cols=cols, grid=grid, extra=extra)
        fx_rows, fx = fixed(pop, diffs(plan_losses(t, cols, grid=grid, extra=extra), "design"), g,
                            args.boot, block=f"{name}/fixed")
        point, d = seed_draws(t, stat, args.draws, 0, args.jobs,
                              out_dir / f"zone_s_draws{tail}_{pop}_{name}_seed0.npy", args.resume)
        expected = np.array([fx[k][0] for k in KEYS])
        if point is not None and not np.allclose(point, expected, rtol=0.0, atol=1e-9):
            raise SystemExit(f"STOP: {name}: the refit point {point} does not reproduce {expected}")
        ref = refit_rows(pop, list(KEYS), expected, d, fx, block=f"{name}/refit")
        held[name] = bool(next(r for r in ref if r["key"] == "S-C")[bound] < 0)
        rows += fx_rows + ref
    return rows, held


def compare(t: pd.DataFrame, cols: dict, pop: str, args: argparse.Namespace, out_dir: Path,
            tail: str, seeds: bool) -> list:
    """M1 for S against C on one population, with every control and reported reading, the three
    readings, and the verdict; stability seeds only where it decides."""
    loss = plan_losses(t, cols)
    rows, fx = fixed(pop, diffs(loss, "design"), t.loc[t["won"], "match_id"].to_numpy(), args.boot)
    path = out_dir / f"zone_s_draws{tail}_{pop}_seed0.npy"
    point, d0 = seed_draws(t, partial(plan_stats, cols=cols), args.draws, 0, args.jobs, path,
                           args.resume)
    expected = np.array([fx[k][0] for k in KEYS])
    if point is not None and not np.allclose(point, expected, rtol=0.0, atol=1e-9):
        raise SystemExit(f"STOP: the refit point {point} does not reproduce the fixed fit {expected}")
    ref = refit_rows(pop, list(KEYS), expected, d0, fx)
    stab, stable = stability(t, cols, pop, expected, d0, args, path) if seeds else ([], None)
    ctrl, clean = shuffled_control(t, cols, pop, args.boot)
    read, held = readings(t, cols, pop, args, out_dir, tail)
    rows += ref + stab + ctrl + read + influence(t, pop, loss)
    rows += by_league(t, cols, pop, args.boot, args.jobs)
    rows += other_partitions(t, cols, pop, expected, args.jobs) + outcome_step(t, cols, pop, args)
    verdict = decide(ref, stable, clean, held)
    print(f"  S against C ({pop}{'' if seeds else ', replication: no stability seeds'}): {verdict}",
          flush=True)
    return rows + [{"population": pop, "block": "verdict", "key": "S", "params": str(verdict)}]


def timing(t: pd.DataFrame, cols: dict, pop: str, args: argparse.Namespace) -> list:
    """What each arm and each resample costs, measured blind: every score is dropped unread."""
    rows = []
    with threadpool_limits(1):
        for a in ("design", *COMPARED):
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
        rows.append({"population": pop, "what": "outcome fit, 3 arms", "seconds": time.time() - t0})
    for r in rows:
        print(f"  {pop:16s} {r['what']:42s} {r['seconds']:9.1f} s", flush=True)
    return rows


def main() -> None:
    """S's parameters, the frames and populations, both arms' counts and the sanity count; then the
    timing or the comparison."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=23)
    ap.add_argument("--limit", type=int, default=None, help="first N matches: the path, not the numbers")
    ap.add_argument("--tables", type=str, default=str(_DER))
    ap.add_argument("--out-dir", type=str, default=str(_DER))
    ap.add_argument("--params", type=str, default=str(_PARAMS), help="S's frozen parameters")
    ap.add_argument("--boot", type=int, default=_BOOT, help="fixed-fit cluster-bootstrap draws")
    ap.add_argument("--refit-boot", type=int, default=_REFIT_BOOT, help="refit draws, outcome step")
    ap.add_argument("--seeds", type=int, default=_SEEDS, help="stability seeds, corners")
    ap.add_argument("--draws", type=int, default=_STABILITY_BOOT, help="refit draws per seed")
    ap.add_argument("--counts-only", action="store_true", help="the sanity count, no model")
    ap.add_argument("--timing", action="store_true", help="cost per arm and draw, no score kept")
    ap.add_argument("--resume", action="store_true", help="reuse the per-seed draws already saved")
    args = ap.parse_args()
    clock = time.time()
    assert_sources()
    out_dir = Path(args.out_dir)
    tail = f"_limit{args.limit}" if args.limit else ""
    precision = (args.boot, args.refit_boot, args.seeds, args.draws)
    if not tail and published_dir(out_dir) and not (args.counts_only or args.timing) and \
            precision != (_BOOT, _REFIT_BOOT, _SEEDS, _STABILITY_BOOT):
        raise SystemExit(f"STOP: S against C is registered at {_SEEDS} seeds of {_STABILITY_BOOT} "
                         f"draws, so a run at {precision} may not write the published record; "
                         f"pass --out-dir")
    if not tail and published_dir(out_dir) and Path(args.tables).resolve() != _DER.resolve():
        raise SystemExit("STOP: tables outside data/derived are not the published ones, so their "
                         "readings may not take the published names; pass --out-dir")
    parts = partitions_of(json.loads(Path(args.params).read_text()))
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
    pd.DataFrame(rows).to_parquet(out_dir / f"zone_s_counts{tail}.parquet", index=False)
    if args.counts_only:
        print(f"\nwrote {out_dir / f'zone_s_counts{tail}.parquet'}  [{time.time() - clock:.0f}s]")
        return
    fit = {pop: t for pop, t in frames.items() if t.loc[t["won"], "match_id"].nunique() >= _MIN_MATCHES}
    if args.timing:
        cost = [r for pop, t in fit.items() for r in timing(t, cols, pop, args)]
        pd.DataFrame(cost).to_parquet(out_dir / f"zone_s_timing{tail}.parquet", index=False)
        print(f"\nwrote {out_dir / f'zone_s_timing{tail}.parquet'}  [{time.time() - clock:.0f}s]")
        return
    rows = []
    for pop, t in fit.items():
        print(f"\nS AGAINST C, {pop}", flush=True)
        rows += compare(t, cols, pop, args, out_dir, tail, seeds=pop == "corner")
    path = out_dir / f"zone_s_comparison{tail}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"\nwrote {path}  [{time.time() - clock:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
