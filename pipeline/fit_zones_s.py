"""fit_zones_s — S, Shaw's zones drawn on the frame this grant has.

Shaw & Gopaladesikan draw their zones from tracking at two moments, two seconds before the kick and
one second after the first touch. The grant has one frame per delivery, the 360 at the kick, so S is
drawn there, at the moment this repo reads the plan: a Gaussian mixture on where the attackers with
the job of attacking the cross stand in the box, over M1's corner population, positions only and no
outcome. Pre-registered (data/derived/prereg_2026-10-07/zones_s.md) before it was fitted.

THE SIZE K is measured. For each K from 2 to 15, five folds by match: every held-out point is scored
by the mixture fitted on the other matches, and each match contributes the mean log density of its
points. S takes the smallest K whose score lies within one standard error, over matches, of the best
(the one-standard-error rule). Bodies in one delivery, and deliveries in one match, are not
independent draws, so the held-out likelihood is grouped and BIC is not used.

THE PARTITION is hard. A body in the box takes its component of highest posterior; outside it C's
cells (`edge`, `wide`, `deep`) stand unchanged, so S against C compares the partition of the box and
nothing else. One mixture labels both sides and both populations. Each component is named by the C
cell its mean falls in, numbered from the goal line out where two share one, and none is merged.
The parameters are frozen to refdata/zones_s.json with their provenance before any score, and the
diagnostics travel beside them: the curve of K, the components, whether each zone is one region on
a half-yard lattice, and the agreement between mixtures fitted on two halves of the matches.

    python pipeline/fit_zones_s.py --jobs 23
    python pipeline/fit_zones_s.py --limit 40 --out-dir data/derived/smoke \\
        --params data/derived/smoke/zones_s.json        # the path, not the numbers
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy import ndimage
from scipy.stats import multivariate_normal
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))
_DER = _REPO / "data" / "derived"
_PARAMS = _REPO / "refdata" / "zones_s.json"

from constants import BOX_X, BOX_Y, PITCH_X_SB                                # noqa: E402
from run_zone_comparison import build, inputs, population                     # noqa: E402
from validate import assert_no_group_leak                                     # noqa: E402
from zones import ZONES, cells, in_box                                        # noqa: E402

# Shaw's mixture on the targets is the ceiling (zones_s.md); the sheet fixes the folds, the restarts
# and the random state, so the fit is a function of the data.
K_RANGE = tuple(range(2, 16))
_FOLDS, _RESTARTS, _STATE = 5, 10, 0
_LATTICE = 0.5                     # yards: the step of the connectedness check


# ── The points ───────────────────────────────────────────────────────────
def points(t: pd.DataFrame, bodies: pd.DataFrame) -> pd.DataFrame:
    """The attackers holding a job who stand in the box at the kick, canonical frame, on the
    population's deliveries, each with its match: s02's `attack_cross` job is the box."""
    att = bodies[(bodies["side"] == "att") & bodies["set_piece_id"].isin(t["set_piece_id"])]
    att = att[in_box(att["x"].to_numpy(float), att["y"].to_numpy(float))]
    return att.merge(t[["set_piece_id", "match_id"]], on="set_piece_id", validate="m:1")


# ── The size ─────────────────────────────────────────────────────────────
def mixture(k: int) -> GaussianMixture:
    return GaussianMixture(k, covariance_type="full", n_init=_RESTARTS, random_state=_STATE)


def held_out(k: int, xy: np.ndarray, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    """The log density of the held-out points under a mixture of size k fitted on the rest."""
    return mixture(k).fit(xy[tr]).score_samples(xy[te])


def k_curve(p: pd.DataFrame, jobs: int) -> pd.DataFrame:
    """Each K's held-out score, the mean over matches of each match's mean log density, and its
    standard error over matches. One (K, fold) fit per task, one thread each."""
    xy, g = p[["x", "y"]].to_numpy(float), p["match_id"].to_numpy()
    folds = list(GroupKFold(_FOLDS).split(xy, groups=g))
    for tr, te in folds:
        assert_no_group_leak(g, tr, te, "match_id")
    work = [(k, tr, te) for k in K_RANGE for tr, te in folds]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        out = Parallel(n_jobs=max(1, min(jobs, len(work))))(
            delayed(held_out)(k, xy, tr, te) for k, tr, te in work)
    rows = []
    for k in K_RANGE:
        ll = np.empty(len(xy))
        for (kk, _, te), v in zip(work, out, strict=True):
            if kk == k:
                ll[te] = v
        per_match = pd.Series(ll).groupby(g).mean()
        rows.append({"k": k, "score": float(per_match.mean()), "matches": len(per_match),
                     "se": float(per_match.std(ddof=1) / np.sqrt(len(per_match)))})
    return pd.DataFrame(rows)


def choose_k(curve: pd.DataFrame) -> int:
    """The one-standard-error rule: the smallest K whose score lies within one standard error of
    the best K's."""
    best = curve.loc[curve["score"].idxmax()]
    return int(curve.loc[curve["score"] >= best["score"] - best["se"], "k"].min())


# ── The partition ────────────────────────────────────────────────────────
def names(means: np.ndarray) -> list:
    """Each component named by the C cell its mean falls in, numbered from the goal line out where
    two share one."""
    cell = cells(means[:, 0], means[:, 1])
    out = []
    for i, c in enumerate(cell):
        twins = sorted((j for j in range(len(cell)) if cell[j] == c), key=lambda j: -means[j, 0])
        out.append(str(c) if len(twins) == 1 else f"{c}_{twins.index(i) + 1}")
    return out


def fit(p: pd.DataFrame, k: int) -> dict:
    """The mixture of size k on every point, its components in C's order of cells and then from the
    goal line out: the parameters S is read from."""
    gmm = mixture(k).fit(p[["x", "y"]].to_numpy(float))
    cell = cells(gmm.means_[:, 0], gmm.means_[:, 1])
    order = sorted(range(k), key=lambda j: (ZONES["C"].index(cell[j]), -gmm.means_[j, 0]))
    means = gmm.means_[order]
    return {"k": k, "names": names(means), "weights": gmm.weights_[order].tolist(),
            "means": means.tolist(), "covariances": gmm.covariances_[order].tolist()}


def label(params: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """S on canonical points: in the box the component of highest posterior, the log weight plus
    the log density as GaussianMixture.predict reads it; outside the box C's cells."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    out = np.asarray(cells(x, y), dtype=object)
    box = in_box(x, y)
    if box.any():
        xy = np.c_[x[box], y[box]]
        lp = np.stack([np.log(w) + np.atleast_1d(multivariate_normal(m, c).logpdf(xy))
                       for w, m, c in zip(params["weights"], params["means"],
                                          params["covariances"], strict=True)], axis=1)
        out[box] = np.asarray(params["names"], dtype=object)[lp.argmax(axis=1)]
    return out


# ── The diagnostics ──────────────────────────────────────────────────────
def regions(params: dict) -> dict:
    """Each zone's number of connected regions on a half-yard lattice of the box."""
    xs = np.arange(BOX_X, PITCH_X_SB + 1e-9, _LATTICE)
    ys = np.arange(BOX_Y[0], BOX_Y[1] + 1e-9, _LATTICE)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    lab = label(params, X.ravel(), Y.ravel()).reshape(X.shape)
    return {z: int(ndimage.label(lab == z)[1]) for z in params["names"]}


def halves(p: pd.DataFrame, k: int) -> float:
    """The adjusted Rand index between the labels that two mixtures of size k give every point,
    each fitted on one half of the matches dealt at random."""
    mids = np.unique(p["match_id"])
    first = np.random.default_rng(_STATE).permutation(mids)[:len(mids) // 2]
    xy, a = p[["x", "y"]].to_numpy(float), p["match_id"].isin(first).to_numpy()
    return float(adjusted_rand_score(mixture(k).fit(xy[a]).predict(xy),
                                     mixture(k).fit(xy[~a]).predict(xy)))


def provenance(p: pd.DataFrame, tables: Path) -> dict:
    """What the parameters were fitted on, with a digest of the points that names the tables and
    the frames they came from."""
    h = hashlib.sha256()
    for col in ("set_piece_id", "match_id", "x", "y"):
        h.update(p[col].astype(str).str.cat(sep="|").encode())
    here = tables.resolve()
    return {"driver": "pipeline/fit_zones_s.py", "sheet": "data/derived/prereg_2026-10-07/zones_s.md",
            "tables": str(here.relative_to(_REPO)) if here.is_relative_to(_REPO) else str(tables),
            "points_sha256": h.hexdigest()[:32], "points": len(p),
            "deliveries": int(p["set_piece_id"].nunique()), "matches": int(p["match_id"].nunique()),
            "k_range": [K_RANGE[0], K_RANGE[-1]], "folds": _FOLDS, "restarts": _RESTARTS,
            "random_state": _STATE}


def main() -> None:
    """M1's corner population and its frames, the points, the curve of K, the fit, the diagnostics,
    and the parameters frozen where every later stage reads them."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=23)
    ap.add_argument("--limit", type=int, default=None, help="first N matches: the path, not the numbers")
    ap.add_argument("--tables", type=str, default=str(_DER))
    # Not zones_S: that is the folder the recipe's pass under S writes, and report_figures reads a
    # pass there whenever the folder exists.
    ap.add_argument("--out-dir", type=str, default=str(_DER / "zones_s_fit"))
    ap.add_argument("--params", type=str, default=str(_PARAMS))
    ap.add_argument("--overwrite", action="store_true", help="replace parameters already frozen")
    args = ap.parse_args()
    clock = time.time()
    tail = f"_limit{args.limit}" if args.limit else ""
    params_path, out_dir = Path(args.params), Path(args.out_dir)
    if tail and params_path.resolve() == _PARAMS.resolve():
        raise SystemExit("STOP: a run on N matches may not write S's parameters; pass --params")
    if params_path.exists() and not args.overwrite:
        raise SystemExit(f"STOP: {params_path} is frozen, and every score of S was read off it; "
                         f"--overwrite replaces it")
    grain, tables = inputs(Path(args.tables), tail)
    info, bodies = build(tables["corner"], "corner", args.jobs)
    t = population(tables["corner"], info, grain[grain["set_piece_type"] == "corner"])
    p = points(t, bodies)
    print(f"S: {len(p)} attackers in the box over {p['set_piece_id'].nunique()} long corners in "
          f"{p['match_id'].nunique()} matches", flush=True)
    curve = k_curve(p, args.jobs)
    k = choose_k(curve)
    print(curve.to_string(index=False) + f"\nK = {k} by the one-standard-error rule", flush=True)
    params = {**fit(p, k), "provenance": provenance(p, Path(args.tables))}
    reg, agree = regions(params), halves(p, k)
    print(f"regions per zone {reg}; agreement between halves (ARI) {agree:.3f}", flush=True)
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(json.dumps(params, indent=2) + "\n")
    out_dir.mkdir(parents=True, exist_ok=True)
    curve.to_parquet(out_dir / f"s_k_curve{tail}.parquet", index=False)
    comp = pd.DataFrame({"name": params["names"], "weight": params["weights"],
                         "mean_x": [m[0] for m in params["means"]],
                         "mean_y": [m[1] for m in params["means"]],
                         "cov_xx": [c[0][0] for c in params["covariances"]],
                         "cov_xy": [c[0][1] for c in params["covariances"]],
                         "cov_yy": [c[1][1] for c in params["covariances"]],
                         "regions": [reg[z] for z in params["names"]]})
    comp.assign(halves_ari=agree).to_parquet(out_dir / f"s_components{tail}.parquet", index=False)
    print(f"\nwrote {params_path} and {out_dir}  [{time.time() - clock:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
