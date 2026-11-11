"""run_zone_comparison — which partition of the box at the kick carries the plan?

s02 and s08 count the plan on one partition of the box at the kick, and this driver chooses it by the
rule pre-registered (data/derived/prereg_2026-10-01/zones.md) before any block below
was scored: F is the reference, and another replaces it only if it reads the plan clearly better. A
partition by depth alone, A below, counts attackers outside the box width as committed and cannot
tell the near post from the far one. src/zones.py keeps the two the comparison leaves in play.

THE SIX REPRESENTATIONS, in the canonical frame: attack towards x = 120 and the kick struck from the
y = 0 side, a flank free kick mirrored by its kick's y as a corner is. Each counts the outfield
attackers holding a job (taker out) and the outfield defenders (keepers out) per zone, both sides on
one grid, and a body in no zone as `rest`, so every block carries the same body totals and the arms
differ in the partition alone:

    A   the three roles by depth only, as the sheet registered them: x < 96, 96 <= x < 108, x >= 108
    B   StatsBomb 360 (Monte 2022): the six-yard box, the channel between the posts from the box
        edge to the six-yard line, and the rest of the box on the near and on the far side
    C   s02b_aim's twelve count cells
    D   Spielverlagerung's four jobs with width: arrive (x >= 108 inside the box width), the short
        option (x >= 102 outside it on the kick side), the edge rebound (100 <= x < 108,
        20 <= y <= 60) and staying back (x < 96)
    E   Shaw-style data zones: a six-component Gaussian mixture on the attackers in front of goal
        (x >= 96, y within 4 yd of the box width) over the long corners, each body's
        responsibilities summed per component. Positions only, fitted once on the corner population
        and applied unchanged to defenders and free kicks, like any fixed grid
    F   six zones: B's four, the frontal (96 <= x < 102 inside the box width) and the short
        zone (x >= 102, y < 18). B nests in F

M1, THE DECISION. Corners decide; flank free kicks replicate, and a contradiction is written down,
not resolved by choosing. On the attack-won rows, where the end zone is the aim, the out-of-fold
zone log loss of s02b_aim's aim model on the delivery design plus one block; the delivery design is
s02b_aim's categoricals and the kick spot, which on a free kick says where it was struck from and
would otherwise be credited to the block. Δ_R = loss(R) − loss(F) row by row. R replaces F only if
Δ_R's refit interval in normal form lies wholly below zero at the Bonferroni level over the five
comparisons (99%), that bound keeps its side at every stability seed, and R's block shuffled over
every row does not beat the design. Several qualify: the lowest Δ_R. Otherwise F stays.

Every contrast reads arm minus reference, so a negative one means the first-named arm predicts
better. Reported and never deciding: the pivotal and percentile forms beside the normal one, with
the skew of the draws; the MDE at 80% the Bonferroni rule could resolve; each block against the
design alone (s02b_aim's D1); the outcome step, s03's delivery rung against the rung plus each block
on the five-class target, at the hyperparameters run_validation tunes; each league alone and each
league left out; the largest shift one match or one attacking team makes; Δ under other grouped
partitions; the per-match sign test; and the free kicks.

The registered run read the plan's tables and wrote to its folder (data/derived/auditoria_2026-09-30);
the recipe runs it on its own tables, which compare_tables holds to the plan's. With --limit N it
reads the N-match tables and grain a smoke run built.

    python pipeline/run_zone_comparison.py --jobs 23
    python pipeline/run_zone_comparison.py --jobs 23 --robustness    # the readings added after it
    python pipeline/run_zone_comparison.py --counts-only             # the sanity count, no model
    python pipeline/run_zone_comparison.py --timing --draws 46 --seeds 1 --refit-boot 23 \\
        --out-dir data/derived/smoke      # cost per arm and draw, no score read
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
from scipy.stats import norm, skew
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))
_DER = _REPO / "data" / "derived"
_PLAN = _DER / "auditoria_2026-09-30"

import loader_sb as L                                                          # noqa: E402
from build_table import published_dir                                         # noqa: E402
from constants import (BOX_X, BOX_Y, EDGE_X, EDGE_Y_HALF, GOALMOUTH_X,         # noqa: E402
                       HOLD_X, PITCH_Y_MID_SB)
from run_aim_zone import _MIN_MATCHES                                          # noqa: E402
from run_first_contact import pivotal, refit_draws, seed_draws                 # noqa: E402
from run_validation import _tuned                                              # noqa: E402
from s01_phases import SetPieceGrammar                                         # noqa: E402
from s02_signature import ROLES, CommitmentSignature                           # noqa: E402
from s02b_aim import DESIGN_COLS, GRIDS, aim_oof, end_zones, zone_loss         # noqa: E402
from s03_value import (_BARRED_SOURCES, _COLUMN_SOURCE as _S03_SOURCE,         # noqa: E402
                       _KICK_INTRINSIC, CompetingOutcomesModel)
from validate import cluster_bootstrap_mean, shuffle_features, wilcoxon_paired  # noqa: E402
from zones import ZONES as _IN_PLAY, canonical_y, cells, in_width, zones_f     # noqa: E402

# Each population's table, named by its kind in s01's grammar, beside the delivery grain eda_build
# writes for both. Corners decide.
POPULATIONS = {"corner": "corners.parquet", "free_kick_flank": "corners_free_kick_flank.parquet"}
_DELIVERY_GRAIN = "eda_deliveries.parquet"

# E's region pad and size, as the sheet draws them; chosen, nothing was measured.
_E_Y_PAD, _E_K = 4.0, 6

REPS = ("A", "B", "C", "D", "E", "F")
RIVALS = REPS[:-1]
# C and F are the two src/zones.py keeps in play, read from there so the comparison and the
# pipeline cannot draw them apart; the other four live only here.
ZONES = {"A": ("attack_cross", "stake_drop", "hold_back"),
         "B": ("six_yard", "central", "near_post", "far_post", "rest"),
         "C": _IN_PLAY["C"], "D": ("arrive", "short", "edge", "hold", "rest"),
         "E": tuple(f"g{k}" for k in range(1, _E_K + 1)) + ("rest",), "F": _IN_PLAY["F"]}
COLS = {R: [f"{R}_{s}_{z}" for s in ("att", "def") for z in ZONES[R]] for R in REPS}
_SOURCE = {**{c: _S03_SOURCE[c] for c in DESIGN_COLS}, "kick_x": "location", "kick_y": "location",
           "league": "match catalogue", **{c: "freeze_frame@T" for R in REPS for c in COLS[R]}}

BONF_ALPHA = 0.05 / len(RIVALS)
# The two readings added after the registered run, before they were computed (zones.md, last
# section): (name, the grid the aim is measured on, columns added to every arm's design, the refit
# bound of C−F that has to stay below zero for C to be confirmed: `bonf_hi` at 99%, `hi` at 95%).
ROBUSTNESS = (("league_in_design", "base", ("league",), "bonf_hi"),
              ("grid_fine", "fine", (), "hi"), ("grid_coarse", "coarse", (), "hi"))
# The precision the sheet fixes (its M-c): five seeds of 2000 match resamples decide; the reported
# refits take 300, as every other in the repo; the fixed-fit intervals 2000. Grouped partitions
# dealt at random give the decisive Δ a second route.
_BOOT, _REFIT_BOOT, _SEEDS, _STABILITY_BOOT = 2000, 300, 5, 2000
_PARTITIONS = 5


def keys(base: str) -> list:
    """Contrast names, each rival against F and then each block against `base`."""
    return [f"{R}-F" for R in RIVALS] + [f"{R}-{base}" for R in REPS]


def assert_sources() -> None:
    """Raise when an input of either model is built on a field that reads the play's future."""
    bad = {c: s for c, s in _SOURCE.items() if any(b in s for b in _BARRED_SOURCES)}
    if bad:
        raise ValueError(f"inputs built on barred fields: {bad}")


# ── The partitions, canonical frame (C and F are src/zones.py's) ─────────
def zones_b(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """B, StatsBomb 360's four cells: F's box zones, with its frontal and short zones as rest."""
    z = zones_f(x, y)
    return np.where(np.isin(z, ("frontal", "short")), "rest", z)


def zones_d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """D, Spielverlagerung's four jobs with width; the edge band is half-open at GOALMOUTH_X so it
    cannot share a body with `arrive`."""
    edge = (x >= EDGE_X[0]) & (x < EDGE_X[1]) & (np.abs(y - PITCH_Y_MID_SB) <= EDGE_Y_HALF)
    return np.select([(x >= GOALMOUTH_X) & in_width(y), (x >= BOX_X) & (y < BOX_Y[0]), edge,
                      x < HOLD_X], ZONES["D"][:-1], "rest")


def _zones_a(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A, the three roles by depth only, as the sheet registered them: held back short of HOLD_X,
    attacking the cross from GOALMOUTH_X, staking the drop in between."""
    return np.where(x < HOLD_X, "hold_back", np.where(x >= GOALMOUTH_X, "attack_cross", "stake_drop"))


HARD = {"A": _zones_a, "B": zones_b, "C": cells, "D": zones_d, "F": zones_f}


def in_front(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """E's region: in front of goal and within _E_Y_PAD of the box width."""
    return (x >= HOLD_X) & (y >= BOX_Y[0] - _E_Y_PAD) & (y <= BOX_Y[1] + _E_Y_PAD)


def fit_e(att: pd.DataFrame) -> tuple[GaussianMixture, np.ndarray]:
    """E's mixture on the attackers in front of goal, and the order that names its components by
    mean, nearest the goal line first, so a column name does not hang on the fit's own order."""
    xy = att.loc[in_front(att["x"].to_numpy(float), att["y"].to_numpy(float)), ["x", "y"]]
    gmm = GaussianMixture(_E_K, random_state=0, n_init=4).fit(xy.to_numpy(float))
    return gmm, np.lexsort((gmm.means_[:, 1], -gmm.means_[:, 0]))


def zone_matrix(rep: str, x: np.ndarray, y: np.ndarray, e: tuple) -> np.ndarray:
    """(bodies, zones) membership: 0/1 under a grid, the mixture's responsibilities under E."""
    if rep == "E":
        gmm, order = e
        front, m = in_front(x, y), np.zeros((len(x), _E_K + 1))
        if front.any():
            m[front, :_E_K] = gmm.predict_proba(np.c_[x, y][front])[:, order]
        m[~front, _E_K] = 1.0
        return m
    lab = np.asarray(HARD[rep](x, y), dtype=object)
    undeclared = set(lab) - set(ZONES[rep])
    if undeclared:
        raise ValueError(f"{rep} emitted zones it does not declare: {sorted(undeclared)}")
    return (lab[:, None] == np.asarray(ZONES[rep], dtype=object)[None, :]).astype(float)


def blocks(bodies: pd.DataFrame, ids: pd.Series, e: tuple) -> pd.DataFrame:
    """One row per delivery of `ids`, in that order: each representation's bodies per side and zone."""
    x, y = bodies["x"].to_numpy(float), bodies["y"].to_numpy(float)
    key = [bodies["set_piece_id"].to_numpy(), bodies["side"].to_numpy()]
    parts = []
    for R in REPS:
        w = (pd.DataFrame(zone_matrix(R, x, y, e), columns=list(ZONES[R])).groupby(key).sum()
             .unstack(level=1, fill_value=0.0))
        w.columns = [f"{R}_{s}_{z}" for z, s in w.columns]
        parts.append(w)
    out = pd.concat(parts, axis=1).reindex(index=ids.to_numpy(), columns=sum(COLS.values(), []))
    return out.fillna(0.0).reset_index(drop=True)


def join_blocks(t: pd.DataFrame, bodies: pd.DataFrame, e: tuple, pop: str) -> pd.DataFrame:
    """The population with every representation's counts beside it, recounted from the frames. The
    plan's tables carry C's and F's counts under these very names (src/zones.py), so their copies
    leave first, and a column that still comes out twice stops the run."""
    block_cols = sum(COLS.values(), [])
    out = pd.concat([t.drop(columns=[c for c in block_cols if c in t.columns]),
                     blocks(bodies, t["set_piece_id"], e)], axis=1)
    if out.columns.duplicated().any():
        raise SystemExit(f"STOP: {pop} carries duplicated columns after the blocks join")
    return out


# ── The frames and the populations ───────────────────────────────────────
def inputs(tables: Path, tail: str = "") -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """The delivery grain and each population's table, from `tables`; `tail` names the N-match ones
    a smoke run built."""
    def read(name: str) -> pd.DataFrame:
        return pd.read_parquet(tables / name.replace(".parquet", f"{tail}.parquet"))

    return read(_DELIVERY_GRAIN), {pop: read(name) for pop, name in POPULATIONS.items()}


def frame_bodies(d: pd.Series, fz: pd.DataFrame) -> tuple[bool, list]:
    """One delivery's 360 off s02's frame and roles: whether it was struck from the attackers' right
    (the kick's y >= 40, s01's corner_side), and (side, x, y) for every outfield attacker holding a
    job and every outfield defender, mirrored onto the y = 0 side when it was."""
    f = CommitmentSignature.from_corner(d, fz).frame
    right = float(f.loc[f["is_ball"] == 1, "y"].iloc[0]) >= PITCH_Y_MID_SB
    out = []
    for side, m in (("att", f["role"].isin(ROLES)),
                    ("def", (f["role"] == "defender") & (f["is_goalkeeper"] == 0))):
        xs, ys = f.loc[m, "x"].to_numpy(float), canonical_y(f.loc[m, "y"].to_numpy(float), right)
        out += [(side, x, y) for x, y in zip(xs, ys, strict=True)]
    return right, out


def match_bodies(mid: int, kind: str, want: set) -> tuple[list, list]:
    """One match's wanted deliveries with a 360: the kick's side and who won the first contact
    (s01's label) per delivery, and its bodies in the canonical frame."""
    ev, fz = L.load_events(mid), L.load_360(mid)
    have = set(fz["event_uuid"])
    g = SetPieceGrammar(ev, kinds=(kind,))
    rows, bodies = [], []
    for _, c in g.deliveries().iterrows():
        if c["id"] not in want or c["id"] not in have:
            continue
        right, found = frame_bodies(c, fz)
        rows.append({"set_piece_id": c["id"], "kick_side_right": right,
                     "won_first_contact": g.label(c).won_first_contact})
        bodies += [(c["id"], *b) for b in found]
    return rows, bodies


def build(table: pd.DataFrame, kind: str, jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every match of the table in sorted order, so the parallel build is the serial one: workers
    return lists and the parent builds one frame of each."""
    mids = sorted(table["match_id"].unique())
    want = table.groupby("match_id")["set_piece_id"].agg(set).to_dict()
    t0 = time.time()
    chunks = ([match_bodies(int(m), kind, want[m]) for m in mids] if jobs == 1 else
              Parallel(n_jobs=jobs)(delayed(match_bodies)(int(m), kind, want[m]) for m in mids))
    info = pd.DataFrame([r for ch in chunks for r in ch[0]])
    bodies = pd.DataFrame([b for ch in chunks for b in ch[1]],
                          columns=["set_piece_id", "side", "x", "y"])
    print(f"{kind}: frames for {len(info)} deliveries over {len(mids)} matches in "
          f"{time.time() - t0:.0f}s ({jobs} worker(s))", flush=True)
    return info, bodies


def population(table: pd.DataFrame, info: pd.DataFrame,
               deliveries: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long deliveries with an end location and a frame: the canonical end zone of s02b's base grid
    (the aim model's target), whether the attack made the first contact, and the canonical kick
    spot. Each filter prints what it takes. The run stops if the frame's kick side disagrees with
    s01's, or if `deliveries` (eda_build's grain) disagrees on who made the first contact."""
    long_ = table[~table["short_corner"].astype(bool)]
    ended = long_[long_["delivery_end_x"].notna()]
    t = ended.merge(info, on="set_piece_id", how="inner", validate="1:1")
    print(f"  {len(table)} with a 360: {len(table) - len(long_)} short or ground, "
          f"{len(long_) - len(ended)} without an end location, {len(ended) - len(t)} without a "
          f"frame -> {len(t)}", flush=True)
    right = (t["corner_side"] == "right").to_numpy()
    if not np.array_equal(right, t["kick_side_right"].to_numpy(bool)):
        raise SystemExit("STOP: the frame's kick side disagrees with s01's corner_side")
    if deliveries is not None:
        ref = t[["set_piece_id"]].merge(deliveries[["set_piece_id", "won_first_contact"]],
                                        on="set_piece_id", how="left", validate="1:1")
        a, b = t["won_first_contact"], ref["won_first_contact"]
        if not (np.array_equal(a.eq(True), b.eq(True)) and np.array_equal(a.isna(), b.isna())):
            raise SystemExit("STOP: who made the first contact disagrees with the delivery grain")
    ey = canonical_y(t["delivery_end_y"].to_numpy(float), right)
    for grid in GRIDS:
        t[f"zone_{grid}"] = end_zones(grid, t["delivery_end_x"].to_numpy(float), ey)
    t["zone"] = t["zone_base"]
    t["won"] = t["won_first_contact"].eq(True)
    t["kick_x"] = t["delivery_x"].astype(float)
    t["kick_y"] = canonical_y(t["delivery_y"].to_numpy(float), right)
    return t.sort_values(["match_id", "set_piece_id"]).reset_index(drop=True)


def sanity(t: pd.DataFrame, bodies: pd.DataFrame, pop: str) -> list:
    """Mean attackers and defenders per zone per delivery, each representation, before any model.
    Stops if a block loses or invents a body. `share` is the share of deliveries with at least half
    a body in the zone, which for a grid is at least one."""
    n = (bodies.groupby(["set_piece_id", "side"]).size().unstack(fill_value=0)
         .reindex(index=t["set_piece_id"], columns=["att", "def"], fill_value=0))
    rows = []
    print(f"\nSANITY COUNT, {pop}: {len(t)} deliveries, attack won the first contact on "
          f"{int(t['won'].sum())}; bodies per delivery att {n['att'].mean():.2f}, "
          f"def {n['def'].mean():.2f}", flush=True)
    for R in REPS:
        line = []
        for s in ("att", "def"):
            cols = [f"{R}_{s}_{z}" for z in ZONES[R]]
            gap = float(np.abs(t[cols].sum(axis=1).to_numpy() - n[s].to_numpy()).max())
            if gap > 1e-9:
                raise SystemExit(f"STOP: {R} loses or invents {s} bodies (largest gap {gap})")
            rows += [{"population": pop, "block": "sanity", "key": f"{R}/{s}/{z}", "n": len(t),
                      "point": float(t[c].mean()), "share": float((t[c] >= 0.5).mean())}
                     for z, c in zip(ZONES[R], cols, strict=True)]
        for z in ZONES[R]:
            a, d = t[f"{R}_att_{z}"], t[f"{R}_def_{z}"]
            line.append(f"{z} {a.mean():.2f}/{d.mean():.2f}")
        print(f"  {R}  (att/def per delivery)  " + "  ".join(line), flush=True)
    return rows


# ── The models ───────────────────────────────────────────────────────────
def design(t: pd.DataFrame, extra: tuple = ()) -> np.ndarray:
    """The delivery design: s02b_aim's categoricals as dummies and the canonical kick spot. On a
    corner the spot is the flag; on a flank free kick it is where the kick was struck from, which a
    block would otherwise be credited with (s03_value.baseline_ladder keeps it in its rung so).
    `extra` adds categoricals to every arm, each with its declared source."""
    if undeclared := [c for c in extra if c not in _SOURCE]:
        raise ValueError(f"columns without a declared source: {undeclared}")
    D = pd.get_dummies(t[DESIGN_COLS + list(extra)].astype(str), dtype=float).to_numpy()
    return np.c_[D, t[["kick_x", "kick_y"]].to_numpy(float)]


def plan_losses(t: pd.DataFrame, shuffle_seed: int | None = None,
                arms: tuple = ("design", *REPS), grid: str = "base", extra: tuple = ()) -> dict:
    """Per-row out-of-fold zone log loss on the attack-won rows, for the design alone and the
    design plus each block, every aim model rebuilt on `t`; the aim is the end zone under `grid`."""
    zones = GRIDS[grid]
    z = t[f"zone_{grid}"].map({k: i for i, k in enumerate(zones)}).to_numpy()
    won, g = t["won"].to_numpy(bool), t["match_id"].to_numpy()
    D = design(t, extra)
    X = {"design": D, **{R: np.c_[D, t[COLS[R]].to_numpy(float)] for R in REPS}}
    return {a: zone_loss(aim_oof(X[a], z, won, g, len(zones), shuffle_seed), z)[won] for a in arms}


def outcome_losses(t: pd.DataFrame, m: CompetingOutcomesModel) -> dict:
    """Per-row out-of-fold five-class log loss of s03's delivery rung and of the rung plus each
    block, at m's hyperparameters, every model rebuilt on `t`."""
    y = t["target"].to_numpy()
    arms = {"delivery": [], **{R: COLS[R] for R in REPS}}
    return {a: m._row_logloss(m.fit_cv(t, numeric=_KICK_INTRINSIC + c), y) for a, c in arms.items()}


def diffs(loss: dict, base: str) -> dict:
    """Per-row differences, in the order of keys(base)."""
    return {**{f"{R}-F": loss[R] - loss["F"] for R in RIVALS},
            **{f"{R}-{base}": loss[R] - loss[base] for R in REPS}}


def plan_stats(t: pd.DataFrame, shuffle_seed: int | None = None, grid: str = "base",
               extra: tuple = ()) -> np.ndarray:
    """keys("design") on one sample."""
    loss = plan_losses(t, shuffle_seed, grid=grid, extra=extra)
    return np.array([d.mean() for d in diffs(loss, "design").values()])


def _outcome_stats(t: pd.DataFrame, m: CompetingOutcomesModel) -> np.ndarray:
    return np.array([d.mean() for d in diffs(outcome_losses(t, m), "delivery").values()])


# ── Readings ─────────────────────────────────────────────────────────────
def forms(point: np.ndarray, draws: np.ndarray, alpha: float) -> dict:
    """At level 1 - alpha over whole draws: normal (the point plus or minus half the percentile
    width, which decides), pivotal and percentile; NaN where no draw is finite whole."""
    ok = draws[np.all(np.isfinite(draws), axis=1)]
    if not len(ok):
        nan = np.full(point.shape, np.nan)
        return dict(n_lo=nan, n_hi=nan, p_lo=nan, p_hi=nan, c_lo=nan, c_hi=nan)
    p_lo, p_hi, c_lo, c_hi = pivotal(point, ok, alpha)
    half = (c_hi - c_lo) / 2
    return dict(n_lo=point - half, n_hi=point + half, p_lo=p_lo, p_hi=p_hi, c_lo=c_lo, c_hi=c_hi)


def fixed(pop: str, d: dict, g: np.ndarray, boot: int, block: str = "fixed") -> tuple[list, dict]:
    """Each contrast on the fixed fit: its match-clustered 95% interval, every one on the same
    resamples, and the per-match signed-rank p as a second route to its sign."""
    rows, out = [], {}
    for k, v in d.items():
        ci = cluster_bootstrap_mean(v, g, n_boot=boot)
        out[k] = (ci.point, ci.lo, ci.hi)
        rows.append({"population": pop, "block": block, "key": k, "n": len(v), "point": ci.point,
                     "lo": ci.lo, "hi": ci.hi, "p_wilcoxon": wilcoxon_paired(v, g)["p_value"]})
    return rows, out


def refit_rows(pop: str, names: list, point: np.ndarray, draws: np.ndarray, fx: dict,
               block: str = "refit") -> list:
    """Every contrast on its refit draws, normal at 95% and at the Bonferroni level with the
    pivotal and percentile forms beside, the draws' SD and skew, and the MDE at 80% the Bonferroni
    normal rule could resolve (half its width plus z_0.8 SDs)."""
    ok = draws[np.all(np.isfinite(draws), axis=1)]
    f95, fb = forms(point, ok, 0.05), forms(point, ok, BONF_ALPHA)
    sd = ok.std(axis=0, ddof=1) if len(ok) > 1 else np.full(point.shape, np.nan)
    rows = []
    print(f"  {block} ({pop}, {len(ok)} draws): point, normal 95%, normal {1 - BONF_ALPHA:.0%}, "
          f"pivotal {1 - BONF_ALPHA:.0%}", flush=True)
    for i, k in enumerate(names):
        rows.append({"population": pop, "block": block, "key": k, "n": len(ok), "point": point[i],
                     "lo": f95["n_lo"][i], "hi": f95["n_hi"][i], "pivotal_lo": f95["p_lo"][i],
                     "pivotal_hi": f95["p_hi"][i], "pct_lo": f95["c_lo"][i],
                     "pct_hi": f95["c_hi"][i], "bonf_lo": fb["n_lo"][i], "bonf_hi": fb["n_hi"][i],
                     "bonf_pivotal_lo": fb["p_lo"][i], "bonf_pivotal_hi": fb["p_hi"][i],
                     "bonf_pct_lo": fb["c_lo"][i], "bonf_pct_hi": fb["c_hi"][i],
                     "fixed_lo": fx[k][1], "fixed_hi": fx[k][2], "sd": sd[i],
                     "skew": float(skew(ok[:, i])) if len(ok) > 2 else np.nan,
                     "mde80": (fb["n_hi"][i] - point[i]) + norm.ppf(0.8) * sd[i]})
        print(f"    {k:12s} {point[i]:+.5f}  [{f95['n_lo'][i]:+.5f}, {f95['n_hi'][i]:+.5f}]  "
              f"[{fb['n_lo'][i]:+.5f}, {fb['n_hi'][i]:+.5f}]  "
              f"[{fb['p_lo'][i]:+.5f}, {fb['p_hi'][i]:+.5f}]", flush=True)
    return rows


def shuffled_control(t: pd.DataFrame, pop: str, boot: int) -> tuple[list, dict]:
    """Each block permuted jointly over every row against the design alone, on the fixed fit: a
    shuffled block that still beats the design (95% interval wholly below zero) reads row structure
    no plan carries, and its representation may not replace F."""
    g = t.loc[t["won"], "match_id"].to_numpy()
    base = plan_losses(t, arms=("design",))["design"]
    rows, clean = [], {}
    for R in REPS:
        s = shuffle_features(t, COLS[R], mode="global", seed=0)
        ci = cluster_bootstrap_mean(plan_losses(s, arms=(R,))[R] - base, g, n_boot=boot)
        clean[R] = bool(ci.hi >= 0)
        rows.append({"population": pop, "block": "control", "key": f"{R}shuffled-design",
                     "n": len(g), "point": ci.point, "lo": ci.lo, "hi": ci.hi,
                     "clean": clean[R]})
        print(f"  shuffled {R}: {ci.point:+.5f} [{ci.lo:+.5f}, {ci.hi:+.5f}] "
              f"{'clean' if clean[R] else 'BEATS THE DESIGN'}", flush=True)
    return rows, clean


def influence(t: pd.DataFrame, pop: str, loss: dict) -> list:
    """The attacking team, and the match, whose leaving moves each rival's Δ furthest toward zero
    and past it, on the fixed fit: the unit that could be carrying the result."""
    w, rows = t.loc[t["won"]], []
    for R in RIVALS:
        d = loss[R] - loss["F"]
        for unit in ("attacking_team", "match_id"):
            s = pd.Series(d).groupby(w[unit].to_numpy()).agg(["sum", "count"])
            loo = (d.sum() - s["sum"]) / (len(d) - s["count"])
            j = (-np.sign(d.mean()) * (loo - d.mean())).idxmax()
            rows.append({"population": pop, "block": "influence", "key": f"{R}-F", "unit": unit,
                         "who": str(j), "point": float(d.mean()), "without": float(loo[j]),
                         "flips": bool(np.sign(loo[j]) != np.sign(d.mean()))})
    return rows


def _league_cell(t: pd.DataFrame, league: str, alone: bool) -> dict:
    sub = t[(t["league"] == league) == alone].reset_index(drop=True)
    return diffs(plan_losses(sub), "design") if alone else dict(zip(
        keys("design"), plan_stats(sub), strict=True))


def by_league(t: pd.DataFrame, pop: str, boot: int, jobs: int) -> list:
    """Each league alone, with match-clustered 95% intervals on its fixed fit, and each league
    left out, the decision's points refitted without it."""
    cells_ = [(lg, alone) for lg in sorted(t["league"].unique()) for alone in (True, False)]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        out = Parallel(n_jobs=min(jobs, len(cells_)))(
            delayed(_league_cell)(t, lg, alone) for lg, alone in cells_)
    rows = []
    for (lg, alone), res in zip(cells_, out, strict=True):
        if alone:
            g = t.loc[(t["league"] == lg) & t["won"], "match_id"].to_numpy()
            rows += fixed(pop, res, g, boot, block=f"league/{lg}")[0]
        else:
            rows += [{"population": pop, "block": f"league_out/{lg}", "key": k, "point": float(v)}
                     for k, v in res.items()]
    return rows


def partitions(t: pd.DataFrame, pop: str, point: np.ndarray, jobs: int) -> list:
    """Every contrast under _PARTITIONS grouped partitions dealt at random, a second route to each
    decisive point: the spread says how much of it is the one partition GroupKFold draws."""
    with parallel_config(backend="loky", inner_max_num_threads=1):
        out = Parallel(n_jobs=min(jobs, _PARTITIONS))(
            delayed(plan_stats)(t, s) for s in range(1, _PARTITIONS + 1))
    v = np.array(out)
    return [{"population": pop, "block": "partitions", "key": k, "n": len(v), "point": point[i],
             "lo": float(v[:, i].min()), "hi": float(v[:, i].max()), "mean": float(v[:, i].mean())}
            for i, k in enumerate(keys("design"))]


def outcome_step(t: pd.DataFrame, pop: str, args: argparse.Namespace) -> list:
    """s03's delivery rung against the rung plus each block on the five-class target, tuned as
    run_validation tunes and held fixed across the arms and the refit draws."""
    m = _tuned(t, True, args.jobs)
    print(f"  outcome step ({pop}): hyperparameters {m.best_params_}", flush=True)
    rows, fx = fixed(pop, diffs(outcome_losses(t, m), "delivery"), t["match_id"].to_numpy(),
                     args.boot, block="outcome_fixed")
    point, draws = refit_draws(t, lambda s, m=m: _outcome_stats(s, m), args.refit_boot, 0,
                               args.jobs)
    expected = np.array([fx[k][0] for k in keys("delivery")])
    if not np.allclose(point, expected, rtol=0.0, atol=1e-9):
        raise SystemExit(f"STOP: the outcome refit point {point} does not reproduce {expected}")
    return rows + refit_rows(pop, keys("delivery"), expected, draws, fx, block="outcome_refit")


def decide(refit: list, stable: dict, clean: dict) -> dict:
    """M1: a rival qualifies when its Bonferroni normal interval lies wholly below zero, that bound
    keeps its side at every stability seed and its shuffled block does not beat the design; the
    lowest Δ among those replaces F, and F stays otherwise. `stable` empty means no seed was run."""
    by = {r["key"]: r for r in refit}
    below = [R for R in RIVALS if by[f"{R}-F"]["bonf_hi"] < 0]
    unstable = [R for R in below if not stable.get(f"{R}-F", True)]
    dirty = [R for R in below if not clean[R]]
    q = [R for R in below if R not in unstable and R not in dirty]
    return {"keeps": min(q, key=lambda R: by[f"{R}-F"]["point"]) if q else "F", "qualify": q,
            "below_zero": below, "unstable": unstable, "dirty_control": dirty,
            "seeds_run": bool(stable)}


# ── The run ──────────────────────────────────────────────────────────────
def stability(t: pd.DataFrame, pop: str, point: np.ndarray, d0: np.ndarray,
              args: argparse.Namespace, path: Path) -> tuple[list, dict]:
    """The Bonferroni normal upper bound of each rival at every seed; steady when it keeps its side."""
    his = {0: forms(point, d0, BONF_ALPHA)["n_hi"]}
    for s in range(1, args.seeds):
        d = seed_draws(t, plan_stats, args.draws, s, args.jobs,
                       path.with_name(path.name.replace("seed0", f"seed{s}")), args.resume)[1]
        his[s] = forms(point, d, BONF_ALPHA)["n_hi"]
    rows, stable = [], {}
    for i, k in enumerate(keys("design")[:len(RIVALS)]):
        stable[k] = len({bool(his[s][i] < 0) for s in his}) == 1
        print(f"  {k} Bonferroni normal upper bound by seed "
              f"{[round(float(his[s][i]), 5) for s in his]}"
              f"{'' if stable[k] else '  CHANGES SIDE'}", flush=True)
        rows += [{"population": pop, "block": "stability", "key": k, "seed": s, "n": args.draws,
                  "hi": float(his[s][i]), "below_zero": bool(his[s][i] < 0)} for s in his]
    return rows, stable


def compare(t: pd.DataFrame, pop: str, args: argparse.Namespace, out_dir: Path, tail: str,
            seeds: bool) -> list:
    """The decision's contrasts for one population, with every control and reported reading;
    stability seeds only where it decides."""
    loss = plan_losses(t)
    rows, fx = fixed(pop, diffs(loss, "design"), t.loc[t["won"], "match_id"].to_numpy(), args.boot)
    path = out_dir / f"zone_draws{tail}_{pop}_seed0.npy"
    point, d0 = seed_draws(t, plan_stats, args.draws, 0, args.jobs, path, args.resume)
    expected = np.array([fx[k][0] for k in keys("design")])
    if point is not None and not np.allclose(point, expected, rtol=0.0, atol=1e-9):
        raise SystemExit(f"STOP: the refit point {point} does not reproduce the fixed fit {expected}")
    ref = refit_rows(pop, keys("design"), expected, d0, fx)
    rows += ref
    stab, stable = stability(t, pop, expected, d0, args, path) if seeds else ([], {})
    ctrl, clean = shuffled_control(t, pop, args.boot)
    rows += stab + ctrl + influence(t, pop, loss) + by_league(t, pop, args.boot, args.jobs)
    rows += partitions(t, pop, expected, args.jobs) + outcome_step(t, pop, args)
    verdict = decide(ref, stable, clean)
    print(f"  M1 ({pop}{'' if seeds else ', replication: no stability seeds'}): {verdict}",
          flush=True)
    return rows + [{"population": pop, "block": "verdict", "key": "M1", "params": str(verdict)}]


def robustness(t: pd.DataFrame, pop: str, args: argparse.Namespace, out_dir: Path,
               tail: str) -> list:
    """The two readings added after the registered run (zones.md, last section), registered on the
    corners and read the same way on the flank free kicks: every contrast with the league in every
    arm's design, and with the aim on the fine and the coarse grid, each on its fixed fit and one
    seed of refit draws; then whether C is confirmed."""
    rows, held = [], {}
    g = t.loc[t["won"], "match_id"].to_numpy()
    for name, grid, extra, bound in ROBUSTNESS:
        print(f"\nROBUSTNESS, {pop}: {name}", flush=True)
        fx_rows, fx = fixed(pop, diffs(plan_losses(t, grid=grid, extra=extra), "design"), g,
                            args.boot, block=f"{name}/fixed")
        path = out_dir / f"zone_draws{tail}_{pop}_{name}_seed0.npy"
        point, d = seed_draws(t, partial(plan_stats, grid=grid, extra=extra), args.draws, 0,
                              args.jobs, path, args.resume)
        expected = np.array([fx[k][0] for k in keys("design")])
        if point is not None and not np.allclose(point, expected, rtol=0.0, atol=1e-9):
            raise SystemExit(f"STOP: {name}: the refit point {point} does not reproduce {expected}")
        ref = refit_rows(pop, keys("design"), expected, d, fx, block=f"{name}/refit")
        held[name] = bool(next(r for r in ref if r["key"] == "C-F")[bound] < 0)
        rows += fx_rows + ref
    verdict = {"confirmed": all(held.values()), **held}
    print(f"\nC confirmed ({pop}): {verdict}", flush=True)
    return rows + [{"population": pop, "block": "robustness", "key": "C confirmed",
                    "params": str(verdict)}]


def timing(t: pd.DataFrame, pop: str, args: argparse.Namespace) -> list:
    """What each arm and each resample costs, measured blind: every score is computed and dropped
    unread, so the precision can be fixed before any result is seen."""
    rows = []
    with threadpool_limits(1):
        for a in ("design", *REPS):
            t0 = time.time()
            plan_losses(t, arms=(a,))
            rows.append({"population": pop, "what": f"aim fit, {a}", "seconds": time.time() - t0})
    t0 = time.time()
    refit_draws(t, plan_stats, args.draws, 0, args.jobs)
    rows.append({"population": pop, "what": f"aim refit, {args.draws} draws on {args.jobs} jobs",
                 "seconds": time.time() - t0})
    t0 = time.time()
    m = _tuned(t, True, args.jobs)
    rows.append({"population": pop, "what": "outcome tuning", "seconds": time.time() - t0})
    with threadpool_limits(1):
        t0 = time.time()
        outcome_losses(t, m)
        rows.append({"population": pop, "what": "outcome fit, 7 arms",
                     "seconds": time.time() - t0})
    t0 = time.time()
    refit_draws(t, lambda s, m=m: _outcome_stats(s, m), args.refit_boot, 0, args.jobs)
    rows.append({"population": pop, "what": f"outcome refit, {args.refit_boot} draws on "
                 f"{args.jobs} jobs", "seconds": time.time() - t0})
    for r in rows:
        print(f"  {pop:16s} {r['what']:42s} {r['seconds']:9.1f} s", flush=True)
    return rows


def main() -> None:
    """Frames, populations, E's mixture and the sanity count; then the timing or the comparison."""
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
    ap.add_argument("--robustness", action="store_true",
                    help="the two readings added after the registered run, on both populations")
    ap.add_argument("--resume", action="store_true", help="reuse the per-seed draws already saved")
    args = ap.parse_args()
    clock = time.time()
    assert_sources()
    out_dir = Path(args.out_dir)
    tail = f"_limit{args.limit}" if args.limit else ""
    precision = (args.boot, args.refit_boot, args.seeds, args.draws)
    published = not tail and (published_dir(out_dir)
                              or out_dir.resolve() == (_PLAN / "zones").resolve())
    if published and not (args.counts_only or args.timing) and \
            precision != (_BOOT, _REFIT_BOOT, _SEEDS, _STABILITY_BOOT):
        raise SystemExit(f"STOP: M1 is registered at {_SEEDS} seeds of {_STABILITY_BOOT} draws, so a "
                         f"run at {precision} may not write the published zone_comparison or "
                         f"zone_robustness records; pass --out-dir")
    if not tail and published_dir(out_dir) and Path(args.tables).resolve() != _DER.resolve():
        raise SystemExit("STOP: tables outside data/derived are not the published ones, so their "
                         "readings may not take the published names; pass --out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = {}
    grain, tables = inputs(Path(args.tables), tail)
    for pop, table in tables.items():
        info, bodies = build(table, pop, args.jobs)
        t = population(table, info, grain[grain["set_piece_type"] == pop])
        frames[pop] = (t, bodies[bodies["set_piece_id"].isin(t["set_piece_id"])])
    b_c = frames["corner"][1]
    e = fit_e(b_c[b_c["side"] == "att"])
    gmm, order = e
    print("E, components by mean (x, y, weight): " + "  ".join(
        f"({gmm.means_[j, 0]:.1f}, {gmm.means_[j, 1]:.1f}, {gmm.weights_[j]:.3f})" for j in order),
        flush=True)
    rows = []
    for pop, (t, bodies) in frames.items():
        t = join_blocks(t, bodies, e, pop)
        frames[pop] = (t, bodies)
        rows += sanity(t, bodies, pop)
    rows += [{"population": "corner", "block": "e_component", "key": f"g{i + 1}",
              "point": float(gmm.weights_[j]), "lo": float(gmm.means_[j, 0]),
              "hi": float(gmm.means_[j, 1])} for i, j in enumerate(order)]
    pd.DataFrame(rows).to_parquet(out_dir / f"zone_counts{tail}.parquet", index=False)
    if args.counts_only:
        print(f"\nwrote {out_dir / f'zone_counts{tail}.parquet'}  [{time.time() - clock:.0f}s]")
        return
    fit = {pop: t for pop, (t, _) in frames.items()
           if t.loc[t["won"], "match_id"].nunique() >= _MIN_MATCHES}
    skipped = [{"population": pop, "block": "skipped", "key": "too_few_matches"}
               for pop in sorted(frames.keys() - fit.keys())]
    for r in skipped:
        print(f"\n{r['population']}: skipped, under {_MIN_MATCHES} matches with an attack-won row",
              flush=True)
    if args.robustness:
        path = out_dir / f"zone_robustness{tail}.parquet"
        rob = skipped + [r for pop, t in fit.items() for r in robustness(t, pop, args, out_dir, tail)]
        pd.DataFrame(rob).to_parquet(path, index=False)
        print(f"\nwrote {path}  [{time.time() - clock:.0f}s]", flush=True)
        return
    rows += skipped
    if args.timing:
        cost = [r for pop, t in fit.items() for r in timing(t, pop, args)]
        path = out_dir / f"zone_timing{tail}.parquet"
        pd.DataFrame(cost).to_parquet(path, index=False)
        print(f"\nwrote {path}  [{time.time() - clock:.0f}s]")
        return
    for pop, t in fit.items():
        print(f"\nCOMPARISON, {pop}", flush=True)
        rows += compare(t, pop, args, out_dir, tail, seeds=pop == "corner")
    path = out_dir / f"zone_comparison{tail}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"\nwrote {path}  [{time.time() - clock:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
