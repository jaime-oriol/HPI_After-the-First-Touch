"""validate — the shared honest-validation layer.

Group splitters, the leakage guard, match-clustered bootstrap intervals and the
label-shuffle controls. Model-agnostic on purpose, because a guard that only one
stage runs is not a guard.

Every stage asserts on every split path it owns (s02b, s03, s04, s04b, s05). The drivers that
import this module take its interval helpers, and run_first_contact, run_frame_gpu and
run_positive_control also run the guard on folds of their own.

Two rules this module exists to enforce, the second from definitions.md §16:

  - every split is checked for leakage on EVERY grouping key it claims, and the
    check is executed, not assumed (assert_no_group_leak);
  - every headline figure carries an interval, and the interval resamples MATCHES,
    because corners inside a match share teams, coach, referee and scoreline.

What the default bootstrap does NOT cover: it holds the fitted out-of-fold
probabilities fixed and resamples only the evaluation rows, so it reflects
evaluation-set sampling error and nothing else. It does not propagate the variance
of refitting on a resampled training set, and it is therefore too NARROW. Say so
wherever it is printed, not only here.

`refit_cluster_bootstrap` is the one that does propagate it, by rebuilding the whole
model inside every draw, and `refit_cluster_draws` hands back those draws so their shape
can be checked before they are read in normal form. It is the honest interval and the
expensive one; the fixed-OOF version stays because most quantities in the repo are
reported against it and swapping the two silently would move published numbers without
saying so.

Two robustness tools sit alongside the bootstrap rather than replacing it.
`wilcoxon_paired` is a distribution-free test on the same paired per-match difference,
because an interval covering zero at p=0.4 and one covering zero at p=0.01 are
different findings that read identically without both. `seed_stability` re-runs a
seeded statistic across seeds and reports how many of its decimals survive, since a
resampling bound computed at one seed is a sample of size one from its own noise.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, wilcoxon

from constants import MIN_CELL

# Same shape as loader_sb.USE_CACHE: read once, compared strictly to "1".
INSPECT = os.environ.get("HUDL_INSPECT") == "1"

# The two-sided 95% normal quantile, for reading a refit interval as the point plus or minus
# this many SDs of its draws (refit_cluster_bootstrap says why that reading decides).
Z95 = float(norm.ppf(0.975))
# 80% power at two-sided 5%: a minimum detectable effect is this many standard errors.
Z80 = Z95 + float(norm.ppf(0.80))


@dataclass
class BootCI:
    """A point and its interval. The four quantities are floats for a scalar statistic and
    equal-length arrays for a vector one, so a caller that intervals several numbers off the
    same draws gets one object rather than one per element."""
    point: "float | np.ndarray"
    lo: "float | np.ndarray"
    hi: "float | np.ndarray"
    se: "float | np.ndarray"
    n_boot: int
    n_groups: int
    alpha: float
    seed: int
    method: str = "cluster-bootstrap-fixed-oof"


def leave_one_group_out(groups: np.ndarray) -> Iterator[tuple]:
    """(held_out_value, train_idx, test_idx) per distinct group, in sorted order.

    Sorted rather than sklearn's LeaveOneGroupOut because the fold table is printed
    in the paper and its row order has to be reproducible."""
    g = np.asarray(groups)
    for held in sorted(pd.unique(g)):
        te = np.flatnonzero(g == held)
        yield held, np.flatnonzero(g != held), te


def assert_no_group_leak(groups: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray,
                         name: str) -> None:
    """Raise if any group value appears on both sides of a split.

    A missing key raises too, and it has to be checked separately rather than left to the
    intersection below: NaN is not equal to itself, so a row whose group did not resolve
    slips through set intersection on BOTH sides at once and the guard reports green on a
    split it never checked. That is the exact failure this module exists to make impossible."""
    g = np.asarray(groups)
    used = np.concatenate([np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)])
    missing = int(pd.isna(g[used]).sum())
    if missing:
        raise AssertionError(
            f"{name} is missing on {missing} of {len(used)} split rows, so their side of "
            f"the split cannot be demonstrated: NaN compares unequal to itself and would "
            f"pass the disjointness test below unnoticed")
    shared = set(g[train_idx]) & set(g[test_idx])
    if shared:
        raise AssertionError(
            f"{name} leaks across the split: {sorted(map(str, shared))[:5]} "
            f"({len(shared)} shared of {len(set(g))} distinct values)")


def assert_split_disjoint(table: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray,
                          cols: tuple = ("match_id", "league")) -> None:
    """Run the leakage guard for every grouping key at once.

    Under a league-out split, match disjointness is implied only if `league`
    resolved for every row, which is exactly the failure this catches.

    A requested column missing from the table RAISES instead of being skipped, and
    an empty request raises too. A guard that quietly checks nothing because its key
    was renamed or dropped is worse than no guard: the suite still reports green
    while the split goes unchecked, which is the one failure this module exists to
    make impossible."""
    if not cols:
        raise ValueError("assert_split_disjoint: no grouping column requested, nothing checked")
    missing = [c for c in cols if c not in table.columns]
    if missing:
        raise KeyError(f"assert_split_disjoint: grouping column(s) {missing} not in table "
                       f"(has {sorted(table.columns)[:8]})")
    for c in cols:
        assert_no_group_leak(table[c].to_numpy(), train_idx, test_idx, c)


def _resample_rows(groups: np.ndarray, n_boot: int, seed: int) -> Iterator[np.ndarray]:
    """Row-index arrays for n_boot cluster resamples, drawn once and reused.

    Yielding from one generator keeps every statistic on the SAME draws, which is
    what makes a paired delta a valid difference rather than two marginal intervals."""
    uniq = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.concatenate([idx_of[g] for g in rng.choice(uniq, len(uniq), replace=True)])


def cluster_bootstrap_mean(values: np.ndarray, groups: np.ndarray, n_boot: int = 2000,
                           alpha: float = 0.05, seed: int = 0) -> BootCI:
    """Percentile interval on the mean of a per-row quantity, resampling groups.

    Use for anything that IS a row mean: per-row log-loss, per-row Brier, a 0/1 hit,
    xSecondBall, or the PAIRED per-row difference between two models. Feeding it the
    difference is the only correct way to interval a delta: two overlapping marginal
    intervals do not imply the difference is insignificant."""
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups)
    draws = [v[rows].mean() for rows in _resample_rows(g, n_boot, seed)]
    draws = np.asarray([d for d in draws if np.isfinite(d)])
    lo, hi = (np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
              if len(draws) else (np.nan, np.nan))
    return BootCI(float(v.mean()), float(lo), float(hi),
                  float(draws.std(ddof=1)) if len(draws) > 1 else float("nan"),
                  len(draws), int(len(np.unique(g))), alpha, seed)


def cluster_bootstrap_stats(frame: pd.DataFrame, stats: dict[str, Callable],
                            group_col: str = "match_id", n_boot: int = 2000,
                            alpha: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Several arbitrary statistics off ONE set of draws.

    For quantities that are not row means (a rate within a subset, an ECE, a ratio of
    sums). Each callable takes a resampled sub-frame and returns a scalar."""
    groups = frame[group_col].to_numpy()
    draws: dict[str, list] = {k: [] for k in stats}
    for rows in _resample_rows(groups, n_boot, seed):
        sub = frame.iloc[rows]
        for k, fn in stats.items():
            v = fn(sub)
            if np.isfinite(v):
                draws[k].append(v)
    recs = []
    for k, fn in stats.items():
        b = draws[k]
        lo, hi = (np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
                  if b else (np.nan, np.nan))
        recs.append(dict(stat=k, point=fn(frame), lo=lo, hi=hi, n_boot=len(b)))
    return pd.DataFrame(recs).set_index("stat")


def refit_cluster_draws(frame: pd.DataFrame, stat_fn: Callable, group_col: str = "match_id",
                        n_boot: int = 2000, seed: int = 0,
                        n_jobs: int = 1) -> tuple[np.ndarray, np.ndarray, int]:
    """(point, kept draws, groups) for a statistic REFIT inside every cluster resample.

    The engine of `refit_cluster_bootstrap`, exposed for a caller that needs the draws rather
    than their percentiles: their spread, or their shape, which is what says whether reading
    them in normal form is fair. `point` has one element per element of the statistic, and
    the draws one row per kept resample, in draw order."""
    g = frame[group_col].to_numpy()
    if pd.isna(g).any():
        raise ValueError(f"refit_cluster_draws: {group_col} unresolved on "
                         f"{int(pd.isna(g).sum())} row(s); the resample unit is undefined")

    def one(rows: np.ndarray) -> np.ndarray:
        sub = frame.iloc[rows].reset_index(drop=True)
        return np.atleast_1d(np.asarray(stat_fn(sub), dtype=float))

    draws = list(_resample_rows(g, n_boot, seed))
    # One BLAS thread per fit in BOTH branches, and that is not a speed knob. The parallel
    # branch caps its workers' pools, so a serial branch left at the machine's 24 threads
    # sums its dense algebra in a different order and the bound moves with the worker count.
    # An interval that moves with --jobs is not an interval of anything.
    if n_jobs != 1:
        from joblib import Parallel, delayed, parallel_config
        with parallel_config(backend="loky", inner_max_num_threads=1):
            vals = Parallel(n_jobs=n_jobs)(delayed(one)(r) for r in draws)
    else:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(1):
            vals = [one(r) for r in draws]

    point = np.atleast_1d(np.asarray(stat_fn(frame), dtype=float))
    # A draw is kept or dropped WHOLE. Keeping the finite half of a partly degenerate draw
    # would build each element's interval over a different set of matches, and the elements
    # would stop being the jointly resampled thing `_resample_rows` exists to make them.
    ok = np.array([v for v in vals if v.shape == point.shape and np.all(np.isfinite(v))])
    return point, ok.reshape(-1, point.size), int(len(np.unique(g)))


def refit_cluster_bootstrap(frame: pd.DataFrame, stat_fn: Callable, group_col: str = "match_id",
                            n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
                            n_jobs: int = 1) -> BootCI:
    """Cluster bootstrap that REFITS the whole model inside every resample.

    The interval `cluster_bootstrap_mean` returns holds the fitted out-of-fold
    probabilities fixed, so it covers evaluation-set sampling error and nothing else.
    This one hands `stat_fn` a resampled frame and lets it rebuild everything it needs
    (imputer, scaler, hyperparameter search, estimator), so the variance of having fit on
    a different sample of matches lands in the interval too. Anything fitted OUTSIDE the
    loop and merely applied inside it is a fixed-OOF bootstrap wearing this name.

    `lo` and `hi` are the percentiles of the draws, and their LOCATION is not to be read: a
    resample keeps about 63% of the distinct matches, which moves every draw by the learning
    curve rather than by the estimator's error. The spread is what the refit contributes, and
    `se` carries it (evidence.md §17).

    `stat_fn` may return a scalar or a fixed-length vector, and a vector is intervalled
    element by element off the SAME draws. That is what lets a caller interval a whole
    published curve from one refit per draw instead of one per element, which would be the
    same cost multiplied by the length of the curve and would score each element on a
    different resample. Those bounds stay POINTWISE. Each element covers at the nominal rate
    on its own, the chance that the whole vector covers at once is lower, and a caller reading
    the curve rather than one of its points needs a simultaneous band, which this does not
    build."""
    point, ok, n_groups = refit_cluster_draws(frame, stat_fn, group_col, n_boot, seed, n_jobs)
    lo, hi = (np.percentile(ok, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
              if len(ok) else (np.full(point.shape, np.nan), np.full(point.shape, np.nan)))
    se = (ok.std(axis=0, ddof=1) if len(ok) > 1 else np.full(point.shape, np.nan))
    # A scalar statistic reports plain floats, which is what every other caller unpacks.
    as_reported = ((lambda a: float(a[0])) if point.size == 1
                   else (lambda a: np.asarray(a, dtype=float)))
    return BootCI(as_reported(point), as_reported(lo), as_reported(hi), as_reported(se),
                  len(ok), n_groups, alpha, seed, method="cluster-bootstrap-refit")


def poisson_ratio_ci(k: int, e: float, alpha: float = 0.05) -> tuple:
    """Exact (Garwood) Poisson interval on k events, divided by the expectation e.

    The observed-over-expected diagnostic both frontier drivers report: a segment whose
    interval excludes 1.0 is one xG systematically mis-prices. sweep_counter reads it as a
    rate, with e the cleared deliveries. It lives here so every caller shares one bound."""
    if e <= 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else chi2.ppf(alpha / 2, 2 * k) / 2 / e
    return lo, chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2 / e


def wilcoxon_paired(values: np.ndarray, groups: np.ndarray, alpha: float = 0.05) -> dict:
    """Distribution-free sign test on a PAIRED per-row difference, clustered by match.

    Aggregates to one mean per match BEFORE testing. Corners inside a match share teams,
    coach, referee and scoreline, so a row-level signed-rank test would treat a match's
    dependent rows as independent observations, and when they move together its standard error
    comes out too small by up to the square root of the rows per match. The unit of
    independence is the match, here as everywhere else in this module.

    Reported NEXT TO the cluster bootstrap, not instead of it. They answer different
    questions: the bootstrap sizes the effect, this says whether its sign survives without
    assuming the differences are anything in particular. A delta whose interval covers zero
    and whose p is 0.4 is one finding; a delta whose interval covers zero and whose p is
    0.01 is a small effect measured precisely, and the two read identically without both."""
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups)
    per_match = np.array([np.nanmean(v[g == k]) for k in np.unique(g)])
    per_match = per_match[np.isfinite(per_match)]
    n = len(per_match)
    if n < 6 or np.allclose(per_match, 0.0):
        # Below ~6 clusters the exact signed-rank test cannot reach any p under 0.05, so a
        # "not significant" here would be a statement about the sample size, not the effect.
        return {"n_clusters": n, "statistic": float("nan"), "p_value": float("nan"),
                "significant": False, "note": "non-operative (too few clusters or all-zero)"}
    stat, p = wilcoxon(per_match, zero_method="wilcox", alternative="two-sided")
    return {"n_clusters": n, "statistic": float(stat), "p_value": float(p),
            "significant": bool(p < alpha), "median_per_match": float(np.median(per_match))}


def seed_stability(stat_fn: Callable, seeds: tuple = (0, 1, 2, 3, 4)) -> dict:
    """Run a seeded statistic once per seed and report how much the seed alone moves it.

    A resampling bound computed at one seed is a sample of size one from the procedure's own
    noise. If the third decimal moves with the seed, the third decimal is not a finding, and
    this is the cheapest way to know which digits are real before quoting them."""
    vals = np.array([float(stat_fn(s)) for s in seeds], dtype=float)
    ok = vals[np.isfinite(vals)]
    if not len(ok):
        return {"seeds": list(seeds), "mean": float("nan"), "spread": float("nan"),
                "stable_digits": 0}
    spread = float(ok.max() - ok.min())
    # How many decimals survive the seed: the first place where the spread shows up.
    digits = 0 if spread <= 0 else max(0, int(-np.floor(np.log10(spread))) - 1)
    return {"seeds": list(seeds), "mean": float(ok.mean()), "min": float(ok.min()),
            "max": float(ok.max()), "spread": spread,
            "stable_digits": 6 if spread == 0 else digits}


def shuffle_labels(y: np.ndarray, groups: np.ndarray | None = None,
                   mode: str = "global", seed: int = 0) -> np.ndarray:
    """Permute the target to build a control that must land at chance.

    Two modes, and both are needed. `global` permutes across the whole table and
    breaks X -> y entirely. `within_match` permutes only among rows of the same
    match, preserving each match's class composition: a global shuffle also destroys
    the within-match correlation of y, so it CANNOT detect a leak that travels
    through match identity, and that is the channel a grouped CV is most exposed to."""
    rng = np.random.default_rng(seed)
    out = np.array(y, copy=True)
    if mode == "global":
        rng.shuffle(out)
        return out
    if mode != "within_match" or groups is None:
        raise ValueError("mode must be 'global', or 'within_match' with groups")
    g = np.asarray(groups)
    for val in np.unique(g):
        idx = np.flatnonzero(g == val)
        out[idx] = rng.permutation(out[idx])
    return out


def shuffle_features(frame: pd.DataFrame, cols: Sequence[str], y: np.ndarray | None = None,
                     groups: np.ndarray | None = None, mode: str = "global",
                     seed: int = 0) -> pd.DataFrame:
    """Permute a BLOCK of feature columns JOINTLY, to control what that block buys.

    Joint, not per column: one permutation index moves all of `cols`, so a permuted row still
    holds one real corner's geometry and not a chimera. `shuffle_labels` beside it permutes
    the target and asks whether the HARNESS leaks; this asks what one BLOCK buys, at fixed
    design width, which a drop-column ablation cannot do.

      global        one permutation over every row, breaking block -> y outright.
      within_match  permutes inside each match, to ask if the block stands in for the match.
      within_class  permutes inside each level of `y`: Ojala & Garriga, JMLR 11 (2010)
                    1833-1863, test 2. It PRESERVES the class-conditional marginal, so a
                    block predicting through marginals alone survives untouched. Its null is
                    "no dependency beyond the per-class marginals", not "carries nothing".
    """
    if not len(cols):
        raise ValueError("cols is empty: there is no block to permute")
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f"columns absent from the frame: {missing}")
    rng = np.random.default_rng(seed)
    out = frame.copy()
    order = np.arange(len(frame))
    if mode == "global":
        order = rng.permutation(order)
    elif mode in ("within_match", "within_class"):
        key = groups if mode == "within_match" else y
        if key is None:
            raise ValueError(f"mode '{mode}' needs {'groups' if mode == 'within_match' else 'y'}")
        k = np.asarray(key)
        for val in np.unique(k):
            idx = np.flatnonzero(k == val)
            order[idx] = rng.permutation(idx)
    else:
        raise ValueError("mode must be 'global', 'within_match' or 'within_class'")
    # Column by column under ONE shared order, not a block assignment: a block of mixed
    # dtypes goes through an object array and would come back with bool columns upcast,
    # which _design then reads differently. This keeps every column's dtype.
    for c in cols:
        out[c] = frame[c].to_numpy()[order]
    return out


def _cluster_ols_slope(x: np.ndarray, y: np.ndarray, g: np.ndarray) -> tuple:
    """Slope of y on x with a CR0 cluster-robust standard error, clustered on g."""
    X = np.column_stack([np.ones(len(x)), np.asarray(x, dtype=float)])
    xtx = np.linalg.pinv(X.T @ X)
    beta = xtx @ (X.T @ y)
    resid = y - X @ beta
    meat = np.zeros((2, 2))
    for c in np.unique(g):
        u = X[g == c].T @ resid[g == c]
        meat += np.outer(u, u)
    v = xtx @ meat @ xtx
    return float(beta[1]), float(np.sqrt(max(v[1, 1], 0.0)))


def heterogeneity_blp(groups: np.ndarray,
                      cross_fit: Callable[[np.random.Generator], tuple],
                      n_reps: int = 20, n_bins: int = 5, seed: int = 0) -> dict:
    """Is a per-row model difference SYSTEMATIC, or noise around one number?

    Ask it BEFORE opening any subgroup, because it is what keeps the subgroups honest. The
    best linear predictor's slope says whether structure exists at all; the top-minus-bottom
    group gap says how much, and for whom. Both cluster on the group. BLP/GATES from
    Chernozhukov, Demirer, Duflo & Fernandez-Val (arXiv:1712.04802), median p over partitions
    with their factor two, honesty from Athey, Tibshirani & Wager (Ann. Statist. 47(2) 2019).

    `cross_fit(rng) -> (d, d_hat)`, one value per row from ONE honest partition, and the
    contract is that no row's d or d_hat came from a model trained on that row's fold. Scoring
    both off a single global cross-fit instead leaves d produced by models that saw the proxy's
    training rows, and the shared estimation noise can inflate the slope into a finding. Keeping
    the refit on the caller's side is also what keeps sklearn out of this module.

    TWO verdicts, because they answer different questions. `verdict` is whether structure
    EXISTS, over the two tests at Bonferroni 0.025 rather than whichever of them fires.
    `actionable` is whether to open cells, and it needs the GROUP contrast resolved: a
    significant slope with an unresolved gap says something varies without saying for whom. A
    slope near 1 means the proxy is CALIBRATED, never that the effect is large: the size is
    `gates_gap`, and the caller owns the margin it is compared against.

    ADAPTED, and the adaptation is what to declare: that literature estimates heterogeneous
    TREATMENT effects under unconfoundedness and there is no treatment here. What travels is
    the split-and-test machinery, not the causal reading."""
    g = np.asarray(groups)
    rng = np.random.default_rng(seed)
    blp, gates = [], []
    for _ in range(n_reps):
        d, dhat = (np.asarray(a, dtype=float) for a in cross_fit(rng))
        if len(d) != len(g) or len(dhat) != len(g):
            raise ValueError("cross_fit must return one d and one d_hat per row")
        if not (np.isfinite(d).all() and np.isfinite(dhat).all()) or np.ptp(dhat) == 0:
            continue
        b, se = _cluster_ols_slope(dhat, d, g)
        blp.append((b, 2 * norm.sf(abs(b) / se) if se > 0 else 1.0))
        edges = np.quantile(dhat, np.linspace(0, 1, n_bins + 1))
        top, bottom = dhat >= edges[-2], dhat <= edges[1]
        keep = top | bottom
        if top.sum() and bottom.sum() and keep.sum() >= 4:
            b2, se2 = _cluster_ols_slope(top[keep].astype(float), d[keep], g[keep])
            gates.append((b2, 2 * norm.sf(abs(b2) / se2) if se2 > 0 else 1.0))
    if not blp:
        return {"n_reps_used": 0, "verdict": "not estimable", "actionable": False}
    p_blp = float(min(1.0, 2 * np.median([p for _, p in blp])))
    p_gates = float(min(1.0, 2 * np.median([p for _, p in gates]))) if gates else float("nan")
    return {"n_reps_used": len(blp), "n_bins": n_bins,
            "blp_slope": float(np.median([b for b, _ in blp])), "blp_p": p_blp,
            "gates_gap": float(np.median([b for b, _ in gates])) if gates else float("nan"),
            "gates_p": p_gates,
            "verdict": ("heterogeneous" if min(p_blp, p_gates if gates else 1.0) < 0.025
                        else "no detectable structure"),
            "actionable": bool(gates and p_gates < 0.025)}


def partial_pooling_cells(values: np.ndarray, groups: np.ndarray, cells: np.ndarray,
                          min_cell: int = MIN_CELL) -> dict:
    """Shrink per-cell means toward the pooled one, and price the spread BETWEEN cells.

    The question a subgroup table cannot answer on its own: does this quantity vary across
    cells someone can NAME, or does every cell differ only as much as its own noise allows.
    Random effects under exchangeability, so a thin cell is pulled hardest and a fat one
    barely moves, which is what stops a small cell's accident from reading as a finding
    (Gelman, Hill & Yajima, JREE 5(2) 2012; Jones et al., Clinical Trials 8(2) 2011).

    tau is the magnitude and zero is a result: it says the cells are one cell wearing several
    names. DerSimonian-Laird for tau-squared, method of moments and closed form, with every
    cell's standard error CLUSTERED ON THE GROUP rather than counted per row.

    The VERDICT comes from Q against chi-square on k-1 df and never from tau clearing zero:
    DerSimonian-Laird returns a positive tau-squared whenever Q exceeds k-1, which happens
    about half the time under the null with few cells, so tau alone would call a finding on a Q
    that does not reach 0.05.

    Cells under `min_cell` rows are reported with their count and kept OUT of tau: below that
    the cell mean is noise, and its tiny standard error would enter the weights as near
    certainty."""
    v, g, c = np.asarray(values, dtype=float), np.asarray(groups), np.asarray(cells)
    rows = [{"cell": str(name), "n": int((c == name).sum()),
             "n_groups": int(pd.Series(g[c == name]).nunique()),
             "mean": float(v[c == name].mean()),
             "se": (_cluster_mean_se(v[c == name], g[c == name])
                    if (c == name).sum() >= min_cell else float("nan")),
             "below_floor": bool((c == name).sum() < min_cell)}
            for name in pd.unique(c)]
    tab = pd.DataFrame(rows)
    ok = tab[(~tab["below_floor"]) & np.isfinite(tab["se"]) & (tab["se"] > 0)]
    if len(ok) < 2:
        return {"cells": tab, "tau": float("nan"), "q_p": float("nan"),
                "verdict": "not estimable", "n_cells_used": int(len(ok))}
    w = 1.0 / ok["se"].to_numpy() ** 2
    mu = float((w * ok["mean"].to_numpy()).sum() / w.sum())
    q = float((w * (ok["mean"].to_numpy() - mu) ** 2).sum())
    k = len(ok)
    denom = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (q - (k - 1)) / denom) if denom > 0 else 0.0
    tab["shrunk"] = [(tau2 * r["mean"] + r["se"] ** 2 * mu) / (tau2 + r["se"] ** 2)
                     if np.isfinite(r["se"]) and r["se"] > 0 else np.nan
                     for _, r in tab.iterrows()]
    p = float(chi2.sf(q, k - 1))
    return {"cells": tab, "pooled": mu, "tau": float(np.sqrt(tau2)), "q": q, "df": k - 1,
            "q_p": p, "i2": float(max(0.0, (q - (k - 1)) / q)) if q > 0 else 0.0,
            "n_cells_used": k, "min_n": int(tab["n"].min()),
            "verdict": "cells differ" if p < 0.05 else "one cell wearing several names"}


def _cluster_mean_se(v: np.ndarray, g: np.ndarray) -> float:
    """Standard error of a mean with the rows clustered on g (CR0 on an intercept fit).

    Pinned by the two cases that define it: with every row its own cluster it reproduces
    sd/sqrt(n) to four places, and on rows carrying a strong within-cluster correlation it
    comes out 3.1x the naive figure. ONE cluster returns NaN rather than the 0.0 the algebra
    gives, because a single cluster says nothing about between-cluster spread and that zero
    would enter the pooling weights as certainty."""
    uniq = np.unique(g)
    if len(v) == 0 or len(uniq) < 2:
        return float("nan")
    mu = v.mean()
    meat = sum(float((v[g == c] - mu).sum()) ** 2 for c in uniq)
    return float(np.sqrt(meat) / len(v))


def inspect_design(X: np.ndarray | pd.DataFrame, names: list | None = None, label: str = "",
                   show: bool | None = None) -> dict:
    """Look at the design matrix BEFORE anything is fit on it.

    The pandas counterpart of the torch one-liner everyone writes by hand: dtype, shape,
    range, and the per-column facts a silent pipeline will otherwise swallow. It reports
    and never raises, because two of the things it flags are legitimate here: a column
    all-NaN inside one fold is handled by _fit_prep, and an intercept-only rung is a
    constant column on purpose. Raising would break rungs that are correct by design.

    Per column, not global. A global mean over this matrix is meaningless: the columns are
    metres, seconds and 0/1 flags at once, so their pooled mean is an artefact of the mix.
    The global range survives only as a magnitude alarm (a stray 1e9 shows up there).

    Four things it exists to catch, all of them silent in pandas, where `.mean()` skips
    NaN without saying so, unlike torch, where a NaN poisons the mean and announces itself:

      all_nan    a column with no observation at all; _fit_prep imputes it to 0, so the
                 feature is silently dead weight rather than an error (s03_value._fit_prep);
      constant   zero variance among finite values, which carries no signal and, unscaled,
                 makes StandardScaler divide by zero;
      has_inf    an infinity, which NaN-aware code does NOT catch and which propagates
                 through the scaler into every coefficient;
      nan_frac   how missing each column is, which decides whether its _isna indicator is
                 informative or just noise.
    """
    A = X.to_numpy(float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if names is None:
        names = list(X.columns) if isinstance(X, pd.DataFrame) else \
            [f"x{i}" for i in range(A.shape[1])]
    names = [str(c) for c in names]

    nan, inf = np.isnan(A), np.isinf(A)
    finite = ~(nan | inf)
    n_fin = finite.sum(axis=0)
    # Per-column stats over finite entries only; an empty column yields NaN, not a warning.
    var = np.array([A[finite[:, j], j].var() if n_fin[j] else np.nan
                    for j in range(A.shape[1])])
    frac = nan.mean(axis=0)
    fin_vals = A[finite]

    rep = {
        "label": label,
        "shape": tuple(int(v) for v in A.shape),
        "dtype": str(A.dtype),
        "n_nan": int(nan.sum()),
        "all_nan": [names[j] for j in range(A.shape[1]) if n_fin[j] == 0],
        "has_inf": [names[j] for j in range(A.shape[1]) if inf[:, j].any()],
        "constant": [names[j] for j in range(A.shape[1]) if n_fin[j] and var[j] == 0],
        "nan_frac": {names[j]: float(frac[j]) for j in range(A.shape[1]) if frac[j] > 0},
        "range": (float(fin_vals.min()), float(fin_vals.max())) if fin_vals.size
                 else (float("nan"), float("nan")),
    }
    if show is None:
        show = INSPECT
    if show:
        print(_format_inspection(rep))
    return rep


def _format_inspection(rep: dict, top: int = 5) -> str:
    """One compact block per design matrix, worst-offending columns first."""
    head = f"[inspect] {rep['label'] or 'design'}  shape={rep['shape']}  dtype={rep['dtype']}"
    lo, hi = rep["range"]
    lines = [head, f"  range [{lo:.4g}, {hi:.4g}]   nan cells: {rep['n_nan']}"]
    for key in ("all_nan", "has_inf", "constant"):
        if rep[key]:
            shown = ", ".join(rep[key][:top])
            more = f" (+{len(rep[key]) - top})" if len(rep[key]) > top else ""
            lines.append(f"  {key}: {shown}{more}")
    if rep["nan_frac"]:
        worst = sorted(rep["nan_frac"].items(), key=lambda kv: -kv[1])[:top]
        lines.append("  most missing: " +
                     ", ".join(f"{c} {f:.1%}" for c, f in worst))
    return "\n".join(lines)
