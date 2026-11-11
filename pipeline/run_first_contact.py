"""run_first_contact — does the 360 at the FIRST CONTACT say who wins the second ball?

The estimand is conditional: P(the attack wins the second ball | what was known at the kick, the
frame at the first contact T1). T1 comes after the delivery, so nothing here enters X_t or the
published `_COLUMN_SOURCE`. The T1 geometry replaces the delivery geometry under the same column
names, in memory and inside this driver only, and no table carrying it is written to disk.

T1 is s01's `first_contact`, called exactly as `s08._clear_and_next` calls it, and its frame is
read by `DropSituation._oriented_frame`, which turns the axes and the teammate flag into the corner
attacker's terms by team. The T1 actor is dropped before anything is measured: the actor stands on
the ball, and counting the actor doubles `first_to_drop_att` (evidence.md §15).

THE VERDICTS ARE DECIDED BY CODE, not read off a log. Each is a paired, match-clustered delta of
per-row log-loss on the refit interval in its PIVOTAL form, the one these rules were registered
with; every rule written after them reads the normal form (evidence.md §17), which report_figures
publishes beside this one. Neither that, the percentile form nor the Bonferroni level over the six
decisions decides. A verdict whose decisive bound changes sign across the five stability seeds does
not decide.

    V1  the T1 frame beats the delivery-location baseline: any T1 arm minus `delivery` below zero
    V2  it carries more than the kick frame: `static_race_1 - static_race_D` or
        `geometry_1 - geometry360_D` below zero; `gbt_1 - gbt360_D` is fixed-interval and reported
    V3  the positions add to who touched and where: `facts_geometry_1 - facts_1` below zero

The verdict's 300 draws are the first 300 of stability seed 0, the same generator in the same order,
so they are drawn once. Every seed's draws are saved as they finish, and `--resume` reuses them.
The grant this reads is `run_sufficiency.grant_table`. V1-V3 were registered on corners; --kind
free_kick_flank reads the flank free kicks from x >= 85 by the same rules, off that population's
grant, as the paper's second population.

    python pipeline/run_first_contact.py --step0 --jobs 20     # coverage, before any model
    python pipeline/run_first_contact.py --jobs 20             # the run
    python pipeline/run_first_contact.py --jobs 20 --kind free_kick_flank   # first_contact_<kind>
    python pipeline/run_first_contact.py --jobs 22 --gbt-refit # phase two, only if V1 is undecided

Loky workers do not inherit the warning filters set here, only the environment, so `main` sets
PYTHONWARNINGS before the first pool opens; without it the strong learner writes one LightGBM and one
Optuna notice per fit into the log. ConvergenceWarning still shows.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from matplotlib.path import Path as Polygon
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))
_DER = _REPO / "data" / "derived"

import loader_sb as L                                                   # noqa: E402
from build_table import kind_tag, published_dir                         # noqa: E402
from constants import DROP_ANCHOR_SB, PITCH_X_SB, PITCH_Y_SB            # noqa: E402
from eda_build import COMBINED_KINDS                                    # noqa: E402
from pitch_control import default_model_params                          # noqa: E402
from run_sufficiency import _won_proba, calibrated_tau, fuse, grant_table   # noqa: E402
from s01_phases import _is_xy, first_contact                            # noqa: E402
from s02_signature import CommitmentSignature, _orient, build_frame      # noqa: E402
from s03_value import (CompetingOutcomesModel, _KICK_INTRINSIC, _NUMERIC,   # noqa: E402
                       _THREE_SIXTY)
from s04b_snapshot import (_SB_TO_M, _gbt_outer_fold, mde, platt_oof,   # noqa: E402
                           power_curve, row_logloss, shuffled_control, snapshot_record)
from s08_situation import DropSituation                                 # noqa: E402
from validate import (_resample_rows, assert_no_group_leak,             # noqa: E402
                      cluster_bootstrap_mean, wilcoxon_paired)

_FACTS = ["fc_won", "fc_t_s", "fc_x", "fc_y"]
_T1, _T1_NO_TAKER = "__t1", "__t1nt"
# Fixed before any measurement: chosen, not measured.
_NEAR_SPOT, _NEAR_SPOT_SHARE = 5.0, 0.05
# V1 on three arms, V2 on two, V3 on one: the six decisions Bonferroni divides by.
DECISIONS = (("static_race_1", "delivery"), ("geometry_1", "delivery"), ("gbt_1", "delivery"),
             ("static_race_1", "static_race_D"), ("geometry_1", "geometry360_D"),
             ("facts_geometry_1", "facts_1"))
REPORTED = (("facts_1", "delivery"), ("gbt_1", "gbt360_D"))
_GBT_ARMS = ("gbt360_D", "gbt_1")
VERDICTS = {"V1": ("static_race_1-delivery", "geometry_1-delivery", "gbt_1-delivery"),
            "V2": ("static_race_1-static_race_D", "geometry_1-geometry360_D"),
            "V3": ("facts_geometry_1-facts_1",)}
_MDE_GRID = (0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015)
# The registered precision, and the only one the published record is written at.
_BOOT, _GBT_TRIALS, _GBT_SEEDS, _REFIT_BOOT, _SEEDS, _STABILITY_BOOT = 2000, 100, 5, 300, 5, 2000


# ── T1: the event, its frame, and what the contact itself says ──────────
def t1_event(ds: DropSituation, corner: pd.Series) -> tuple:
    """(first contact, clearance, next touch, attacking team, kick time), the first contact from
    the same call `_clear_and_next` makes, so the episode it bounds is the one this reads."""
    seq = ds.g._sequence(corner)
    post = seq[seq["id"] != corner["id"]]
    t0 = float(corner["_t"])
    fc = first_contact(post, t0 + ds.g.SECOND_BALL_WINDOW_S)
    clear, nxt, att = ds._clear_and_next(corner)
    return fc, clear, nxt, att, t0


def t1_frame(ds: DropSituation, fc: pd.Series, att: str, corner: pd.Series) -> pd.DataFrame | None:
    """The T1 frame in s02's schema: the actor out, and the corner spot as the one ball row."""
    f = ds._oriented_frame(fc["id"], fc["team.name"], att)
    if f is None or not len(f):
        return None
    orient = _orient(corner)
    bx, by = orient(float(corner["location"][0]), float(corner["location"][1]))
    ball = {"x": bx, "y": by, "team_id": 1, "is_goalkeeper": 0, "is_actor": 0, "is_ball": 1}
    return pd.concat([f[f["is_actor"] == 0].assign(is_ball=0), pd.DataFrame([ball])],
                     ignore_index=True)


def contact_facts(fc: pd.Series, t0: float, att: str) -> dict:
    """Who touched first, how long after the kick, and where, in the attacking frame."""
    xy = fc.get("location")
    x, y = DropSituation._orient_xy(xy, fc["team.name"], att) if _is_xy(xy) else (np.nan, np.nan)
    return {"fc_won": float(fc["team.name"] == att), "fc_t_s": float(fc["_t"]) - t0,
            "fc_x": float(x), "fc_y": float(y)}


def race_record(frame: pd.DataFrame, anchor_sb: tuple = DROP_ANCHOR_SB,
                vmax: float | None = None) -> dict | None:
    """s04b's static race on a frame already built: every non-ball body, the same units and cap."""
    bodies = frame[frame["is_ball"] == 0]
    if len(bodies) < 2:
        return None
    v = default_model_params()["max_player_speed"] if vmax is None else vmax
    xy = bodies[["x", "y"]].to_numpy(float)
    d = np.linalg.norm((xy - np.asarray(anchor_sb, dtype=float)) * _SB_TO_M, axis=1)
    return {"tti": d / v, "is_att": (bodies["team_id"].to_numpy() == 1),
            "n_bodies": int(len(bodies))}


def _near_spot(frame: pd.DataFrame, spot: tuple) -> np.ndarray:
    d = np.hypot(frame["x"].to_numpy(float) - spot[0], frame["y"].to_numpy(float) - spot[1])
    return (frame["is_ball"].to_numpy() == 0) & (frame["team_id"].to_numpy() == 1) & (d <= _NEAR_SPOT)


def near_spot_attackers(frame: pd.DataFrame, spot: tuple) -> int:
    """Attackers within _NEAR_SPOT of the corner spot: the taker is not identifiable in a 360."""
    return int(_near_spot(frame, spot).sum())


def anchor_in_view(ds: DropSituation, fc: pd.Series, att: str) -> bool | None:
    """Does T1's visible_area hold DROP_ANCHOR_SB? The polygon comes in the actor's frame."""
    va = ds.fz.loc[ds.fz["event_uuid"] == fc["id"], "visible_area"]
    poly = va.iloc[0] if len(va) else None
    if poly is None or len(poly) < 6:
        return None
    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
    if fc["team.name"] != att:
        pts = np.c_[PITCH_X_SB - pts[:, 0], PITCH_Y_SB - pts[:, 1]]
    return bool(Polygon(pts).contains_point(DROP_ANCHOR_SB))


def assert_order(t_fc: float, t_clear: float, t_drop: float) -> None:
    """T1 <= clearance < drop, the two invariants s01 and s08 declare; anything else stops the run."""
    if not t_fc <= t_clear < t_drop:
        raise SystemExit(f"STOP: first contact {t_fc} <= clearance {t_clear} < drop {t_drop} fails")


def corner_row(ds: DropSituation, corner: pd.Series) -> tuple[dict, dict | None]:
    """One corner's step-0 diagnostics, contact facts and T1 geometry, plus its T1 race record."""
    fc, clear, nxt, att, t0 = t1_event(ds, corner)
    r = {"set_piece_id": corner["id"], "has_t1": fc is not None, "t1_has_360": False,
         "t1_is_clear": None, "t1_to_clear_s": np.nan, "t1_to_drop_s": np.nan, "n_bodies_t1": 0,
         "n_bodies_delivery": np.nan, "near_spot_att": 0, "anchor_in_view": None,
         **dict.fromkeys(_FACTS, np.nan)}
    if corner["id"] in ds.ids:
        f = build_frame(corner, ds.fz)[0]
        r["n_bodies_delivery"] = int(((f["is_ball"] == 0) & (f["is_actor"] == 0)).sum())
    if fc is None:
        return r, None
    r.update(contact_facts(fc, t0, att))
    if clear is not None:
        r["t1_is_clear"] = bool(fc["id"] == clear["id"])
        r["t1_to_clear_s"] = float(clear["_t"]) - float(fc["_t"])
        if nxt is not None:
            assert_order(float(fc["_t"]), float(clear["_t"]), float(nxt["_t"]))
            r["t1_to_drop_s"] = float(nxt["_t"]) - float(fc["_t"])
    frame = t1_frame(ds, fc, att, corner) if fc["id"] in ds.ids else None
    if frame is None:
        return r, None
    spot = tuple(frame.loc[frame["is_ball"] == 1, ["x", "y"]].iloc[0])
    r.update(t1_has_360=True, n_bodies_t1=int((frame["is_ball"] == 0).sum()),
             near_spot_att=near_spot_attackers(frame, spot), anchor_in_view=anchor_in_view(ds, fc, att))
    if r["n_bodies_t1"] >= 2:
        for suffix, fr in ((_T1, frame), (_T1_NO_TAKER, frame[~_near_spot(frame, spot)])):
            geo = CommitmentSignature(fr, None).row(corner["id"])
            r.update({f"{c}{suffix}": geo[c] for c in _THREE_SIXTY})
    return r, race_record(frame)


def match_rows(mid: int, want: set, kind: str = "corner") -> tuple[list, dict, dict]:
    """One match: its rows, and the T1 and delivery race records keyed by delivery id."""
    ev, fz = L.load_events(mid), L.load_360(mid)
    ds = DropSituation(ev, fz, kinds=(kind,))
    d = ds.g.deliveries()
    rows, rec_1, rec_d = [], {}, {}
    for _, c in d[d["id"].isin(want)].iterrows():
        r, rec = corner_row(ds, c)
        rows.append({**r, "match_id": mid})
        if rec is not None:
            rec_1[c["id"]] = rec
        if c["id"] in ds.ids:
            s = snapshot_record(c, fz)
            if s is not None:
                rec_d[c["id"]] = {k: s[k] for k in ("tti", "is_att", "n_bodies")}
    return rows, rec_1, rec_d


# ── Arms and deltas, on whatever frame they are handed ──────────────────
def t1_view(frame: pd.DataFrame, suffix: str = _T1) -> pd.DataFrame:
    """The frame with the published 360 block replaced by the T1 one, under the same names."""
    return frame.drop(columns=_THREE_SIXTY).rename(columns={f"{c}{suffix}": c for c in _THREE_SIXTY})


def _require(table: pd.DataFrame, cols: list) -> None:
    missing = [c for c in cols if c not in table.columns]
    if missing:
        raise SystemExit(f"STOP: {missing} absent, and s03._design would drop them in silence")


def gbt_oof(X: np.ndarray, y: np.ndarray, g: np.ndarray, trials: int, seeds: int,
            jobs: int) -> np.ndarray:
    """s04b.tuned_gbt_oof averaged over seeds, with each (seed, outer fold) pair as one task.

    The same folds, the same per-fold nested search and the same seeds as `tuned_gbt_oof`, so every
    seed's out-of-fold vector is that function's, which a test pins; the average matches
    run_sufficiency's. What changes is the unit of parallelism: twenty-five tasks keep the machine
    busy where one seed's five folds at a time leave most of it idle."""
    folds = list(GroupKFold(n_splits=min(5, len(np.unique(g)))).split(X, y, g))
    for tr, te in folds:
        assert_no_group_leak(g, tr, te, "match_id")
    tasks = [(sd, k, tr, te) for sd in range(seeds) for k, (tr, te) in enumerate(folds)]
    done = Parallel(n_jobs=min(jobs, len(tasks)))(
        delayed(_gbt_outer_fold)(X, y, g, tr, te, trials, sd, k) for sd, k, tr, te in tasks)
    oof = np.zeros((seeds, len(y)))
    for (sd, _, _, _), (te, p, _) in zip(tasks, done, strict=True):
        oof[sd, te] = p
    return np.mean(list(oof), axis=0)


def gbt_arm_losses(frame: pd.DataFrame, t1: pd.DataFrame, m: CompetingOutcomesModel, y: np.ndarray,
                   g: np.ndarray, arms: tuple, trials: int, seeds: int, jobs: int) -> dict:
    """Per-row log-loss of the strong-learner arms asked for: `gbt360_D` on the delivery frame and
    `gbt_1` on the T1 view. Each is its own tuned fit, so an arm nobody asks for is a fit not made."""
    tables = {"gbt360_D": frame, "gbt_1": t1}
    out = {}
    for name in arms:
        X, _ = m._design(tables[name])
        out[name] = row_logloss(gbt_oof(X, y, g, trials, seeds, jobs), y)
    return out


def arm_losses(frame: pd.DataFrame, m: CompetingOutcomesModel, rec_1: dict, rec_d: dict,
               trials: int, seeds: int, with_gbt: bool, gbt_jobs: int = 1,
               suffix: str = _T1, gbt_arms: tuple = _GBT_ARMS) -> dict:
    """Per-row log-loss of every arm on `frame`. The heads keep the C the grid chose; the races
    re-select their temperature and the strong learner its search, as run_sufficiency does."""
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("default", category=ConvergenceWarning)
    y, g = frame["won"].to_numpy().astype(int), frame["match_id"].to_numpy()
    t1 = t1_view(frame, suffix)

    def head(tab: pd.DataFrame, numeric: list | None) -> np.ndarray:
        _require(tab, _NUMERIC if numeric is None else numeric)
        cv = m.fit_cv(tab, numeric=numeric, with_delivery_cats=True)
        return row_logloss(_won_proba(cv), y)

    out = {"delivery": head(frame, _KICK_INTRINSIC), "geometry360_D": head(frame, None),
           "geometry_1": head(t1, None), "facts_1": head(t1, _KICK_INTRINSIC + _FACTS),
           "facts_geometry_1": head(t1, _NUMERIC + _FACTS)}
    for name, rec in (("static_race_D", rec_d), ("static_race_1", rec_1)):
        tau = calibrated_tau(rec, frame)["tau"]
        out[name] = row_logloss(platt_oof(fuse(rec, frame["set_piece_id"], tau), y, g), y)
    if with_gbt:
        out.update(gbt_arm_losses(frame, t1, m, y, g, gbt_arms, trials, seeds, gbt_jobs))
    return out


def step_means(frame: pd.DataFrame, pairs: tuple, **kw: Any) -> np.ndarray:
    """The mean paired delta of each (a, b) in `pairs`, every arm rebuilt on `frame`.

    A strong-learner arm is refitted only if a pair reads it. The decisions never read `gbt360_D`
    (its pair with `gbt_1` is fixed-interval and reported), and refitting it in every draw doubled
    the cost of the draw without touching a single delta."""
    arms = tuple(a for a in _GBT_ARMS if any(a in p for p in pairs))
    losses = arm_losses(frame, **kw, gbt_arms=arms)
    return np.array([(losses[a] - losses[b]).mean() for a, b in pairs])


def refit_draws(frame: pd.DataFrame, stat_fn: Callable[[pd.DataFrame], np.ndarray],
                n_boot: int, seed: int, n_jobs: int,
                group_col: str = "match_id") -> tuple[np.ndarray, np.ndarray]:
    """validate.refit_cluster_bootstrap, keeping every draw in order so levels and prefixes are read
    off one set: (point, draws), a non-finite draw kept as a row of NaN rather than dropped.

    Same resamples, the same one-BLAS-thread rule in both branches, and `pivotal` applies the same
    whole-draw filter; a test pins that its 95% bounds are validate's bit for bit."""
    g = frame[group_col].to_numpy()
    if pd.isna(g).any():
        raise ValueError(f"{group_col} unresolved on {int(pd.isna(g).sum())} row(s)")

    def one(rows: np.ndarray) -> np.ndarray:
        return np.atleast_1d(np.asarray(stat_fn(frame.iloc[rows].reset_index(drop=True)), dtype=float))

    draws = list(_resample_rows(g, n_boot, seed))
    if n_jobs != 1:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            vals = Parallel(n_jobs=n_jobs)(delayed(one)(r) for r in draws)
    else:
        with threadpool_limits(1):
            vals = [one(r) for r in draws]
    point = np.atleast_1d(np.asarray(stat_fn(frame), dtype=float))
    out = np.array([v if v.shape == point.shape else np.full(point.shape, np.nan) for v in vals])
    return point, out


def mde_grid(diff: np.ndarray, g: np.ndarray, grid: tuple, jobs: int) -> tuple[float, pd.DataFrame]:
    """(MDE_80, power curve) with one task and one seed per candidate delta.

    s04b.power_curve draws every candidate from one generator in sequence, which pins the grid to a
    single core: over the grant's matches that is the slowest serial step of the run. Seeding each
    candidate on its own keeps every point reproducible and lets the grid run in parallel."""
    parts = Parallel(n_jobs=min(jobs, len(grid)))(
        delayed(power_curve)(diff, g, (d,), seed=i) for i, d in enumerate(grid))
    curve = pd.concat(parts, ignore_index=True)
    return mde(curve), curve


def pivotal(point: np.ndarray, draws: np.ndarray, alpha: float) -> tuple:
    """(pivotal lo, pivotal hi, percentile lo, percentile hi) at level 1 - alpha, over whole draws."""
    ok = draws[np.all(np.isfinite(draws), axis=1)]
    lo, hi = np.percentile(ok, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return 2 * point - hi, 2 * point - lo, lo, hi


def decide(rows: list, stable: dict, gbt_refit: bool) -> dict:
    """V1-V3 by the rules in this module's docstring, off the refit rows and each key's stability.

    A verdict is True when one of its steps has the pivotal 95% interval wholly below zero and that
    step's decisive bound keeps its side at every stability seed. V1 without the strong learner's
    refit cannot close as a null: the pre-registration sends it to phase two instead. The Bonferroni
    reading travels beside each verdict and never decides it."""
    by = {r["key"]: r for r in rows}
    out = {}
    for name, keys in VERDICTS.items():
        present = [k for k in keys if k in by]
        wins = [k for k in present if by[k]["below_zero"] and stable.get(k, True)]
        flips = [k for k in present if not stable.get(k, True)]
        if wins:
            state = "true"
        elif name == "V1" and not gbt_refit:
            state = "undecided: phase two (strong-learner refit)"
        elif flips:
            state = "unstable: does not decide"
        else:
            state = "false"
        out[name] = {"state": state, "by": wins, "unstable": flips,
                     "bonferroni_below_zero": [k for k in present if by[k]["bonf_hi"] < 0]}
    return out


# ── The run ──────────────────────────────────────────────────────────────
def build(grant: pd.DataFrame, jobs: int, kind: str = "corner") -> tuple[pd.DataFrame, dict, dict]:
    """The T1 table over the grant's matches, in input order so the parallel build is the serial one."""
    mids = sorted(grant["match_id"].unique())
    want = {m: set(grant.loc[grant["match_id"] == m, "set_piece_id"]) for m in mids}
    t0 = time.time()
    chunks = ([match_rows(m, want[m], kind) for m in mids] if jobs == 1 else
              Parallel(n_jobs=jobs, verbose=5)(delayed(match_rows)(m, want[m], kind) for m in mids))
    rows = [r for ch in chunks for r in ch[0]]
    rec_1 = {k: v for ch in chunks for k, v in ch[1].items()}
    rec_d = {k: v for ch in chunks for k, v in ch[2].items()}
    print(f"T1 frames read for {len(rows)} corners over {len(mids)} matches in "
          f"{time.time() - t0:.0f}s ({jobs} worker(s))", flush=True)
    return pd.DataFrame(rows), rec_1, rec_d


def step0(t1: pd.DataFrame, grant: pd.DataFrame) -> list:
    """Coverage before any model, total and by league; no threshold, the rows are what they are."""
    s = t1.merge(grant[["set_piece_id", "league"]], on="set_piece_id")
    s["usable"] = s["n_bodies_t1"] >= 2
    rows = []
    for lg, sub in [("all", s), *s.groupby("league")]:
        vals = {"grant": len(sub), "has_t1": int(sub["has_t1"].sum()),
                "t1_has_360": int(sub["t1_has_360"].sum()), "usable": int(sub["usable"].sum()),
                "anchor_in_view": int(sub["anchor_in_view"].eq(True).sum()),
                "t1_is_clear": int(sub["t1_is_clear"].eq(True).sum()),
                "near_spot_rows": int((sub.loc[sub["usable"], "near_spot_att"] > 0).sum())}
        print(f"  {lg:16s} " + "  ".join(f"{k} {v}" for k, v in vals.items()), flush=True)
        rows += [{"block": "step0", "key": f"{lg}/{k}", "n": v} for k, v in vals.items()]
    u = s[s["usable"]]
    for col in ("n_bodies_t1", "n_bodies_delivery", "t1_to_clear_s", "t1_to_drop_s", "fc_t_s"):
        med = float(u[col].median())
        print(f"  median {col}: {med:.3f}", flush=True)
        rows.append({"block": "step0", "key": f"median/{col}", "n": len(u), "point": med})
    return rows


def verdicts(point: np.ndarray, draws: np.ndarray, pairs: tuple, fixed: dict, tag: str) -> list:
    """Each decision's refit rows: the pivotal 95% interval that decides, the percentile and the
    Bonferroni-pivotal beside it, and the width against the fixed fit."""
    rows = []
    bonf = 0.05 / len(DECISIONS)
    for i, (a, b) in enumerate(pairs):
        p_lo, p_hi, c_lo, c_hi = (v[i] for v in pivotal(point, draws, 0.05))
        b_lo, b_hi, _, _ = (v[i] for v in pivotal(point, draws, bonf))
        key = f"{a}-{b}"
        f_lo, f_hi = fixed[key]
        print(f"  {key:34s} {point[i]:+.5f} pivotal [{p_lo:+.5f}, {p_hi:+.5f}] percentile "
              f"[{c_lo:+.5f}, {c_hi:+.5f}] bonferroni-pivotal [{b_lo:+.5f}, {b_hi:+.5f}]"
              f"{'  BEATS' if p_hi < 0 else ''}", flush=True)
        rows.append({"block": f"refit{tag}", "key": key, "n": int(np.isfinite(draws[:, i]).sum()),
                     "point": point[i], "lo": p_lo, "hi": p_hi, "pct_lo": c_lo, "pct_hi": c_hi,
                     "bonf_lo": b_lo, "bonf_hi": b_hi, "fixed_lo": f_lo, "fixed_hi": f_hi,
                     "width_ratio": (c_hi - c_lo) / (f_hi - f_lo) if f_hi > f_lo else np.nan,
                     "below_zero": bool(p_hi < 0)})
    return rows


def seed_draws(table: pd.DataFrame, stat: Callable[[pd.DataFrame], np.ndarray], n: int, seed: int,
               jobs: int, path: Path, resume: bool) -> tuple[np.ndarray | None, np.ndarray]:
    """One seed's refit draws, saved as soon as they finish and reused by --resume only when the
    saved file holds the number asked for; a file of another length is drawn again. run_aim_zone
    imports this copy, so the two pre-registered drivers resume by one rule."""
    if resume and path.exists():
        saved = np.load(path)
        if len(saved) == n:
            print(f"  seed {seed}: {path.name} reused", flush=True)
            return None, saved
        print(f"  seed {seed}: {path.name} holds {len(saved)} draws, not {n}: drawn again", flush=True)
    t0 = time.time()
    point, d = refit_draws(table, stat, n, seed, jobs)
    np.save(path, d)
    print(f"  seed {seed}: {n} draws in {time.time() - t0:.0f}s -> {path.name}", flush=True)
    return point, d


def main() -> None:
    """Step 0, then the arms, the deltas, the refit verdicts, their stability, controls and MDE."""
    os.environ.setdefault("PYTHONWARNINGS",
                          "ignore:X does not have valid feature names,ignore:Argument")
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("default", category=ConvergenceWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="first N matches: the path, not the numbers")
    ap.add_argument("--step0", action="store_true", help="coverage only, and stop")
    ap.add_argument("--boot", type=int, default=_BOOT, help="fixed-fit cluster-bootstrap draws")
    ap.add_argument("--trials", type=int, default=_GBT_TRIALS,
                    help="Optuna trials per outer fold, both LightGBM arms")
    ap.add_argument("--seeds", type=int, default=_GBT_SEEDS, help="strong-learner seeds")
    ap.add_argument("--refit-boot", type=int, default=_REFIT_BOOT, help="refit draws for the verdicts")
    ap.add_argument("--stability-seeds", type=int, default=_SEEDS)
    ap.add_argument("--stability-boot", type=int, default=_STABILITY_BOOT)
    ap.add_argument("--gbt-refit", action="store_true",
                    help="phase two: refit the strong learners too, when V1 stays undecided")
    ap.add_argument("--resume", action="store_true", help="reuse the per-seed draws already saved")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="where the artefacts land (default: data/derived)")
    ap.add_argument("--kind", choices=COMBINED_KINDS, default="corner",
                    help="the population: corners, or the flank free kicks from x >= 85")
    args = ap.parse_args()
    clock = time.time()
    out_dir = Path(args.out_dir) if args.out_dir else _DER
    kt = kind_tag((args.kind,))
    tail = kt + (f"_limit{args.limit}" if args.limit else "")
    tag = f"{tail}{'_gbt' if args.gbt_refit else ''}"
    precision = (args.boot, args.trials, args.seeds, args.refit_boot, args.stability_seeds,
                 args.stability_boot)
    if (not args.step0 and published_dir(out_dir) and not args.limit
            and precision != (_BOOT, _GBT_TRIALS, _GBT_SEEDS, _REFIT_BOOT, _SEEDS, _STABILITY_BOOT)):
        raise SystemExit(f"STOP: V1-V3 are registered at {_REFIT_BOOT} refit draws and {_SEEDS} "
                         f"seeds of {_STABILITY_BOOT}, with the strong learner's {_GBT_TRIALS} trials "
                         f"x {_GBT_SEEDS} seeds, so a run at {precision} may not write the published "
                         f"first_contact{tag}.parquet; pass --out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    def phase(name: str) -> None:
        print(f"\n[{time.time() - clock:7.0f}s] {name}", flush=True)

    phase("STEP 0: the grant and its first-contact frames")
    grant = grant_table(args.limit, kt)
    print(f"grant {len(grant)} {args.kind} deliveries / {grant['match_id'].nunique()} matches",
          flush=True)
    t1, rec_1, rec_d = build(grant, args.jobs, args.kind)
    rows = step0(t1, grant)
    geo_cols = [f"{c}{s}" for s in (_T1, _T1_NO_TAKER) for c in _THREE_SIXTY]
    t1.drop(columns=[c for c in geo_cols if c in t1.columns]).to_parquet(
        out_dir / f"first_contact_step0{tail}.parquet", index=False)
    if args.step0:
        pd.DataFrame(rows).to_parquet(out_dir / f"first_contact_coverage{tail}.parquet", index=False)
        return

    keep = t1[(t1["n_bodies_t1"] >= 2) & t1["set_piece_id"].isin(rec_1)
              & t1["set_piece_id"].isin(rec_d)]
    table = grant.merge(keep.drop(columns=["match_id"]), on="set_piece_id", how="inner")
    table = table.sort_values(["match_id", "set_piece_id"]).reset_index(drop=True)
    _require(table, [f"{c}{_T1}" for c in _THREE_SIXTY] + _FACTS)
    y, g = table["won"].to_numpy().astype(int), table["match_id"].to_numpy()
    near_share = float((table["near_spot_att"] > 0).mean())
    print(f"rows: {len(table)} corners / {table['match_id'].nunique()} matches, base rate "
          f"{y.mean():.4f}; attackers near the spot on {100 * near_share:.1f}% of rows", flush=True)
    rows += [{"block": "population", "key": "corners", "n": len(table), "point": float(y.mean())},
             {"block": "population", "key": "near_spot_share", "n": len(table), "point": near_share}]

    phase("TUNING, as run_sufficiency does it")
    m = CompetingOutcomesModel(n_splits=5)
    tune = m.tune_cv(table, n_jobs=args.jobs)
    m.apply_best_params(tune)
    print(f"  adopted {tune.best_params}", flush=True)
    rows.append({"block": "tuning", "key": "adopted", "n": tune.n, "point": tune.outer_log_loss,
                 "params": str(tune.best_params)})
    kw = dict(m=m, rec_1=rec_1, rec_d=rec_d, trials=args.trials, seeds=args.seeds)

    phase("ARMS, fixed fit")
    losses = arm_losses(table, **kw, with_gbt=True, gbt_jobs=args.jobs)
    for name, ll in losses.items():
        print(f"  {name:17s} {ll.mean():.5f}", flush=True)
        rows.append({"block": "arm", "key": name, "n": len(y), "point": float(ll.mean())})
    fixed = {}
    print("PAIRED DELTAS, fixed fit (a minus b; negative = a wins):", flush=True)
    for a, b in DECISIONS + REPORTED:
        d = losses[a] - losses[b]
        ci = cluster_bootstrap_mean(d, g, n_boot=args.boot)
        fixed[f"{a}-{b}"] = (ci.lo, ci.hi)
        print(f"  {a + '-' + b:34s} {ci.point:+.5f} [{ci.lo:+.5f}, {ci.hi:+.5f}]", flush=True)
        rows.append({"block": "delta", "key": f"{a}-{b}", "n": len(y), "point": ci.point,
                     "lo": ci.lo, "hi": ci.hi, "p_wilcoxon": wilcoxon_paired(d, g)["p_value"]})

    pairs = DECISIONS if args.gbt_refit else tuple(p for p in DECISIONS if "gbt_1" not in p)
    stat = lambda f: step_means(f, pairs, **kw, with_gbt=args.gbt_refit)
    expected = np.array([(losses[a] - losses[b]).mean() for a, b in pairs])
    phase(f"REFIT, seed 0 ({'with' if args.gbt_refit else 'without'} the strong learners)")
    n0 = max(args.refit_boot, args.stability_boot if args.stability_seeds else 0)
    point, d0 = seed_draws(table, stat, n0, 0, args.jobs,
                           out_dir / f"first_contact_draws{tag}_seed0.npy", args.resume)
    if point is not None and not np.allclose(point, expected, rtol=0.0, atol=1e-9):
        raise SystemExit(f"STOP: the refit point {point} does not reproduce the fixed deltas {expected}")
    ver = verdicts(expected, d0[:args.refit_boot], pairs, fixed, "")
    rows += ver

    stable = {}
    if args.stability_seeds:
        phase(f"STABILITY, {args.stability_seeds} seeds x {args.stability_boot} draws")
        hi = {}
        for sd in range(args.stability_seeds):
            d_sd = d0 if sd == 0 else seed_draws(
                table, stat, args.stability_boot, sd, args.jobs,
                out_dir / f"first_contact_draws{tag}_seed{sd}.npy", args.resume)[1]
            hi[sd] = pivotal(expected, d_sd[:args.stability_boot], 0.05)[1]
        for i, (a, b) in enumerate(pairs):
            key, sides = f"{a}-{b}", [bool(hi[sd][i] < 0) for sd in hi]
            stable[key] = len(set(sides)) == 1
            print(f"  {key:34s} pivotal upper bound {[round(float(hi[sd][i]), 5) for sd in hi]}"
                  f"{'' if stable[key] else '  CHANGES SIDE'}", flush=True)
            rows += [{"block": "stability", "key": key, "seed": sd, "n": args.stability_boot,
                      "hi": float(hi[sd][i]), "below_zero": bool(hi[sd][i] < 0)} for sd in hi]

    if near_share > _NEAR_SPOT_SHARE:
        nt = tuple(p for p in pairs if any(a in ("geometry_1", "gbt_1", "facts_geometry_1") for a in p))
        phase(f"NO-TAKER READING (attackers near the spot on {100 * near_share:.1f}% of rows)")
        stat_nt = lambda f: step_means(f, nt, **kw, with_gbt=args.gbt_refit, suffix=_T1_NO_TAKER)
        p_nt, d_nt = seed_draws(table, stat_nt, args.refit_boot, 0, args.jobs,
                                out_dir / f"first_contact_draws{tag}_no_taker_seed0.npy", args.resume)
        if p_nt is None:
            p_nt = stat_nt(table)
        rows += verdicts(p_nt, d_nt, nt, fixed, "_no_taker")

    phase("CONTROLS (a shuffled model beating its entropy means the harness leaks)")
    p_race_1 = fuse(rec_1, table["set_piece_id"], calibrated_tau(rec_1, table)["tau"])
    for mode in ("within_match", "global"):
        c = m.shuffled_control(t1_view(table), mode=mode)
        s = shuffled_control(p_race_1, y, g, mode=mode)
        print(f"  geometry_1/{mode}: {c['verdict']}   static_race_1/{mode}: "
              f"{'clean' if s['clean'] else 'LEAKAGE SUSPECTED'}", flush=True)
        rows += [{"block": "control", "key": f"geometry_1/{mode}", "clean": c["verdict"] == "clean"},
                 {"block": "control", "key": f"static_race_1/{mode}", "clean": s["clean"]}]

    phase("MDE, on the fixed-fit draws and widened by each step's refit ratio")
    ratio = {r["key"]: r["width_ratio"] for r in ver}
    for a, b in (("geometry_1", "delivery"), ("static_race_1", "delivery")):
        m80 = mde_grid(losses[a] - losses[b], g, _MDE_GRID, args.jobs)[0]
        adj = m80 * ratio.get(f"{a}-{b}", np.nan)
        print(f"  MDE_80 {a}-{b}: {m80}  adjusted {adj}", flush=True)
        rows.append({"block": "power", "key": f"mde_80/{a}-{b}", "n": len(y), "point": m80,
                     "hi": adj})

    phase("VERDICTS, by the pre-registered rules")
    for name, v in decide(ver, stable, args.gbt_refit).items():
        print(f"  {name}: {v['state']}  (by {v['by'] or '-'}; unstable {v['unstable'] or '-'}; "
              f"Bonferroni below zero {v['bonferroni_below_zero'] or '-'})", flush=True)
        rows.append({"block": "verdict", "key": name, "params": str(v)})

    out = out_dir / f"first_contact{tag}.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"\nwrote {out}  [{time.time() - clock:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
