"""Tests for validate — splitters, the leakage guard, clustered intervals, controls.

Fully synthetic: this module is the thing that decides whether every other number
in the repo is honest, so its own tests must not depend on any data being present.
"""
import numpy as np
import pandas as pd
import pytest

from validate import (
    _format_inspection, assert_no_group_leak, assert_split_disjoint, cluster_bootstrap_mean,
    cluster_bootstrap_stats, inspect_design, leave_one_group_out, poisson_ratio_ci,
    heterogeneity_blp, partial_pooling_cells, refit_cluster_bootstrap, refit_cluster_draws,
    shuffle_features, shuffle_labels,
)


# ── Splitters and the guard ─────────────────────────────────────────────
def test_leave_one_group_out_partitions():
    """One fold per distinct group; each fold is disjoint and together they cover all rows."""
    g = np.array(list("aabbbcc"))
    folds = list(leave_one_group_out(g))
    assert [h for h, _, _ in folds] == ["a", "b", "c"]                # sorted, so the fold table is reproducible
    seen = np.concatenate([te for _, _, te in folds])
    assert sorted(seen) == list(range(len(g)))                        # every row held out exactly once
    for _, tr, te in folds:
        assert set(tr) & set(te) == set() and len(tr) + len(te) == len(g)


def test_leakage_guard_catches_a_shared_group():
    """A group value on both sides raises, and a clean split does not."""
    g = np.array([1, 1, 2, 2, 3, 3])
    assert_no_group_leak(g, np.array([0, 1, 2, 3]), np.array([4, 5]), "match_id")
    with pytest.raises(AssertionError, match="match_id leaks"):
        assert_no_group_leak(g, np.array([0, 1, 2]), np.array([3, 4, 5]), "match_id")


def test_leakage_guard_catches_a_group_key_that_did_not_resolve():
    """A NaN group value is the silent version of the same failure.

    NaN compares unequal to itself, so a row whose key is missing lands in neither side's
    set and the intersection comes back empty on a split that was never checked. The guard
    has to say so out loud instead of reporting green, which is the whole reason
    assert_split_disjoint refuses to check nothing."""
    g = np.array([1.0, np.nan, 2.0, np.nan])
    with pytest.raises(AssertionError, match="missing on 2 of 4 split rows"):
        assert_no_group_leak(g, np.array([0, 1]), np.array([2, 3]), "match_id")
    assert_no_group_leak(g, np.array([0]), np.array([2]), "match_id")   # rows off the split are fine


def test_split_disjoint_checks_every_key():
    """A split clean on match_id but leaking on league must still fail.

    This is the league-out failure mode: the folds are match-disjoint by
    construction, so a guard that only checks match_id would pass while a league
    straddled both sides."""
    t = pd.DataFrame({"match_id": [1, 2, 3, 4], "league": ["a", "a", "b", "b"]})
    assert_split_disjoint(t, np.array([0, 1]), np.array([2, 3]))
    with pytest.raises(AssertionError, match="league leaks"):
        assert_split_disjoint(t, np.array([0, 1, 2]), np.array([3]))


def test_split_disjoint_refuses_to_check_nothing():
    """A missing or empty grouping key must RAISE, never skip.

    The dangerous failure is not a loud error, it is a guard that passes because
    the column it checks was renamed away: the split then goes unchecked while the
    suite still reports green."""
    t = pd.DataFrame({"match_id": [1, 2, 3, 4]})
    tr, te = np.array([0, 1]), np.array([2, 3])
    with pytest.raises(KeyError, match="league"):
        assert_split_disjoint(t, tr, te)                       # league absent, requested by default
    with pytest.raises(ValueError, match="nothing checked"):
        assert_split_disjoint(t, tr, te, cols=())
    assert_split_disjoint(t, tr, te, cols=("match_id",))       # asking only for what exists is fine


# ── Clustered bootstrap ─────────────────────────────────────────────────
def test_bootstrap_point_is_the_sample_mean():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    ci = cluster_bootstrap_mean(v, np.array([1, 1, 2, 2]), n_boot=100, seed=0)
    assert ci.point == pytest.approx(2.5) and ci.lo <= ci.point <= ci.hi
    assert ci.n_groups == 2 and ci.n_boot == 100


def test_clustering_widens_the_interval_when_rows_correlate():
    """Resampling clusters must be WIDER than resampling rows under within-cluster
    correlation. That gap is the entire reason the group is the match: if the two
    agreed, every interval in the repo would be understating its uncertainty."""
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(40), 25)                                  # 40 clusters of 25 rows
    v = np.repeat(rng.normal(size=40), 25) + 0.05 * rng.normal(size=1000)   # nearly all variance BETWEEN clusters
    clustered = cluster_bootstrap_mean(v, g, n_boot=500, seed=1)
    by_row = cluster_bootstrap_mean(v, np.arange(len(v)), n_boot=500, seed=1)
    assert (clustered.hi - clustered.lo) > 3 * (by_row.hi - by_row.lo)


def test_bootstrap_covers_the_truth_at_about_the_nominal_rate():
    """Empirical coverage of the 95% interval on clustered data with a known mean."""
    rng = np.random.default_rng(7)
    hits = 0
    sims = 120
    for _ in range(sims):
        cl = rng.normal(0.0, 1.0, size=30)                            # cluster effects, true mean 0
        g = np.repeat(np.arange(30), 10)
        v = np.repeat(cl, 10) + rng.normal(0.0, 0.3, size=300)
        ci = cluster_bootstrap_mean(v, g, n_boot=200, seed=0)
        hits += int(ci.lo <= 0.0 <= ci.hi)
    assert 0.85 <= hits / sims <= 1.0                                 # loose band; a broken interval falls far below


def test_bootstrap_stats_shares_one_set_of_draws():
    """Two statistics and their difference come off the same resamples, so the
    difference interval is a paired one rather than two marginals subtracted."""
    df = pd.DataFrame({"match_id": np.repeat(np.arange(20), 5),
                       "a": np.arange(100, dtype=float),
                       "b": np.arange(100, dtype=float) * 2})
    out = cluster_bootstrap_stats(
        df, {"mean_a": lambda s: s["a"].mean(), "mean_b": lambda s: s["b"].mean(),
             "delta": lambda s: s["b"].mean() - s["a"].mean()},
        n_boot=200, seed=0)
    assert set(out.index) == {"mean_a", "mean_b", "delta"}
    assert out.loc["delta", "point"] == pytest.approx(
        out.loc["mean_b", "point"] - out.loc["mean_a", "point"])
    assert out.loc["delta", "lo"] <= out.loc["delta", "point"] <= out.loc["delta", "hi"]


# ── Refit bootstrap ─────────────────────────────────────────────────────
def _clustered(seed: int = 0, n_groups: int = 40, per: int = 10) -> pd.DataFrame:
    """Rows correlated within match, which is the case the whole module is built for."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"match_id": np.repeat(np.arange(n_groups), per),
                         "v": np.repeat(rng.normal(size=n_groups), per)
                         + 0.1 * rng.normal(size=n_groups * per)})


def _regression_frame(seed: int = 0, n_groups: int = 15, per: int = 8, p: int = 6) -> pd.DataFrame:
    """A design the fit can actually move on: correlated rows within match, few matches."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_groups * per, p))
    y = (X @ rng.normal(size=p) + np.repeat(rng.normal(0, 1.5, n_groups), per)
         + rng.normal(0, 1.0, n_groups * per))
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    df["y"], df["match_id"] = y, np.repeat(np.arange(n_groups), per)
    return df


def _oof_squared_error(df: pd.DataFrame) -> np.ndarray:
    """Per-row leave-one-match-out squared error of an OLS fit. The cheapest thing with the
    shape that matters: a model refit on other matches and scored on this one."""
    X = df[[c for c in df.columns if c.startswith("x")]].to_numpy()
    y, g = df["y"].to_numpy(), df["match_id"].to_numpy()
    out = np.zeros(len(y))
    for held in np.unique(g):
        tr = g != held
        beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        out[~tr] = (y[~tr] - X[~tr] @ beta) ** 2
    return out


def test_refit_is_wider_than_holding_the_fit_fixed():
    """The entire reason this function exists, on the same data and the same draws.

    The fixed-OOF interval resamples rows around out-of-fold losses computed ONCE; the refit
    re-cross-fits inside every draw, so a draw that happens to hold a different set of
    matches gets a different model and the spread of that lands in the interval. Measured
    here at 1.25x wider, and between 1.15x and 1.34x over six data seeds, so the 1.10 bar is
    below the whole observed range and both seeds are fixed: the number is deterministic
    rather than a draw that can wander under the bar."""
    df = _regression_frame()
    refit = refit_cluster_bootstrap(df, lambda s: float(_oof_squared_error(s).mean()),
                                    n_boot=400, seed=0)
    fixed = cluster_bootstrap_mean(_oof_squared_error(df), df["match_id"].to_numpy(),
                                   n_boot=400, seed=0)
    assert (refit.hi - refit.lo) > 1.10 * (fixed.hi - fixed.lo)
    assert refit.method == "cluster-bootstrap-refit" and refit.n_groups == 15


def test_refit_hands_the_callable_only_the_resampled_rows():
    """Anything fitted outside the loop makes this a fixed-OOF bootstrap wearing the name,
    so the callable must never see a row the draw did not select. Match blocks are drawn
    whole and with replacement, so the sizes come in multiples of the block."""
    df = _clustered(n_groups=12, per=5)
    seen = []
    refit_cluster_bootstrap(df, lambda s: (seen.append(s), float(s["v"].mean()))[1],
                            n_boot=25, seed=3)
    assert len(seen) == 26                                 # 25 draws + the point estimate
    for s in seen[:-1]:
        assert len(s) == len(df)                           # 12 blocks of 5, drawn with replacement
        assert s["match_id"].nunique() <= 12                # a draw repeats matches, never invents one
    assert seen[-1].equals(df)                             # the point is the full frame


def test_refit_refuses_an_unresolved_group_key():
    """A NaN match_id makes the resample unit undefined, and NaN != NaN would let it slip
    through the disjointness check unnoticed rather than loudly."""
    df = _clustered(n_groups=6, per=3)
    df.loc[0, "match_id"] = np.nan
    with pytest.raises(ValueError, match="match_id unresolved"):
        refit_cluster_bootstrap(df, lambda s: float(s["v"].mean()), n_boot=5, seed=0)


def _mean_v(sub: pd.DataFrame) -> float:
    """Module level so loky can ship it to a worker; a lambda would not pickle."""
    return float(sub["v"].mean())


def test_refit_is_identical_serial_and_parallel():
    """The n_jobs branch must return the draw-for-draw same interval as the serial one.

    joblib only distributes the work: the resample indices come from ONE seeded generator
    before the fan-out, so a difference here would mean the draws themselves moved. That is
    the same determinism the corner build asserts between its serial and parallel paths, and
    nothing else in the tree exercises this branch even though it is the one the expensive
    published refit runs on."""
    df = _clustered(n_groups=10, per=4)
    serial = refit_cluster_bootstrap(df, _mean_v, n_boot=20, seed=1)
    parallel = refit_cluster_bootstrap(df, _mean_v, n_boot=20, seed=1, n_jobs=2)
    assert (parallel.point, parallel.lo, parallel.hi) == (serial.point, serial.lo, serial.hi)
    assert parallel.n_boot == serial.n_boot == 20
    assert parallel.method == "cluster-bootstrap-refit"


def test_the_bootstrap_summarises_exactly_the_draws_it_exposes():
    """run_refit_draws persists these draws and report_figures reads the normal form off them,
    so they must be the ones behind the published percentiles, not a second resample that
    happens to share the seed."""
    df = _clustered(n_groups=10, per=4)
    point, draws, n_groups = refit_cluster_draws(df, _mean_v, n_boot=50, seed=4)
    ci = refit_cluster_bootstrap(df, _mean_v, n_boot=50, seed=4)
    assert draws.shape == (ci.n_boot, 1) and n_groups == ci.n_groups == 10
    assert ci.point == point[0]
    assert (ci.lo, ci.hi) == tuple(np.percentile(draws[:, 0], [2.5, 97.5]))
    assert ci.se == draws[:, 0].std(ddof=1)


def _blas_threads(sub: pd.DataFrame) -> float:
    """Threads the BLAS pool offers this draw. Module level so loky can ship it."""
    from threadpoolctl import threadpool_info
    return float(max([d["num_threads"] for d in threadpool_info()] or [1]))


def test_both_branches_draw_under_the_same_thread_budget():
    """The two branches must do their arithmetic the same way, not merely start from the
    same draws.

    The parallel branch caps each worker to one BLAS thread. A serial branch left on the
    machine's full pool then reduces the same dense algebra in a different order, and the
    bound moves while the point does not: measured over 20 draws of a per-bin summary,
    --jobs 1 and --jobs 12 agreed exactly on the point and differed by 1.2e-10 on the
    interval. An interval that moves with the worker count is not an interval of the
    procedure it names.

    Asserted on the thread budget itself rather than on two bounds agreeing, because a
    test-sized matrix stays under OpenBLAS's own threading threshold and would agree either
    way, so it would pass on the broken code and prove nothing."""
    df = _clustered(n_groups=6, per=3)
    serial = refit_cluster_bootstrap(df, _blas_threads, n_boot=3, seed=0)
    parallel = refit_cluster_bootstrap(df, _blas_threads, n_boot=3, seed=0, n_jobs=2)
    assert serial.lo == parallel.lo == 1.0, "a draw ran on more than one BLAS thread"


def test_refit_intervals_a_vector_off_exactly_the_draws_a_scalar_would_use():
    """A vector statistic must give element for element what N scalar runs give.

    This is what makes intervalling a whole published shape one refit per draw instead of
    one per element: if the two disagreed, the elements would be resampled independently
    and their joint variation would be gone. Both directions are pinned here: the scalar
    path still returns plain floats, which every other caller in the repo depends on."""
    df = _clustered(n_groups=12, per=5)
    parts = (lambda s: s["v"].mean(), lambda s: s["v"].std(), lambda s: s["v"].max())
    vec = refit_cluster_bootstrap(df, lambda s: [f(s) for f in parts], n_boot=60, seed=2)
    assert vec.point.shape == vec.lo.shape == vec.hi.shape == vec.se.shape == (3,)
    for k, fn in enumerate(parts):
        one = refit_cluster_bootstrap(df, fn, n_boot=60, seed=2)
        assert isinstance(one.point, float) and isinstance(one.lo, float)
        assert (one.point, one.lo, one.hi) == pytest.approx(
            (vec.point[k], vec.lo[k], vec.hi[k]))


def test_a_draw_with_one_non_finite_element_is_dropped_whole():
    """Half a draw is not a draw. Keeping the finite elements of a degenerate resample
    would build each element's interval over a different set of matches, which is exactly
    what drawing every statistic from one generator exists to prevent."""
    df = _clustered(n_groups=8, per=4)
    seen = {"n": 0}

    def stat(sub: pd.DataFrame) -> list:
        seen["n"] += 1
        return [1.0, np.nan if seen["n"] == 3 else 2.0]

    ci = refit_cluster_bootstrap(df, stat, n_boot=10, seed=0)
    assert ci.n_boot == 9                                  # 10 draws, the third dropped entire
    assert ci.lo.shape == ci.hi.shape == (2,)
    assert ci.point == pytest.approx([1.0, 2.0])           # the point is the full frame, still finite


# ── The Poisson obs/exp interval ────────────────────────────────────────
def test_poisson_interval_matches_its_closed_form_at_two_degrees_of_freedom():
    """Garwood on k events is built from chi-square quantiles, and at 2 df the chi-square
    IS an exponential: chi2.ppf(p, 2) = -2·ln(1-p). So both ends have an elementary closed
    form that does not go through the same scipy call the implementation uses.

    Upper end at k=0: -ln(alpha/2). Lower end at k=1: -ln(1-alpha/2)."""
    lo0, hi0 = poisson_ratio_ci(0, 1.0)
    assert lo0 == 0.0                                      # no events -> nothing to bound from below
    assert hi0 == pytest.approx(-np.log(0.025), rel=1e-12)     # 3.6889
    lo1, _ = poisson_ratio_ci(1, 1.0)
    assert lo1 == pytest.approx(-np.log(0.975), rel=1e-12)     # 0.025318


def test_poisson_interval_inverts_the_poisson_tails_it_claims_to():
    """The definition, checked against a different distribution than the one it is written
    with. An exact Poisson interval is the set of rates the observation does not reject:
    P(X <= k | hi) = alpha/2 and P(X >= k | lo) = alpha/2."""
    from scipy.stats import poisson
    for k in (1, 4, 17, 60):
        lo, hi = poisson_ratio_ci(k, 1.0)
        assert poisson.cdf(k, hi) == pytest.approx(0.025, abs=1e-9)
        assert poisson.sf(k - 1, lo) == pytest.approx(0.025, abs=1e-9)


def test_poisson_interval_brackets_the_ratio_and_grows_with_the_count():
    """obs/exp has to sit inside its own interval, and both ends move up with k: a segment
    is only mis-priced if the interval EXCLUDES 1.0, which is unreadable if either fails."""
    e = 12.5
    prev_lo, prev_hi = -1.0, -1.0
    for k in range(0, 40):
        lo, hi = poisson_ratio_ci(k, e)
        assert lo <= k / e <= hi
        assert lo >= prev_lo and hi > prev_hi               # monotone in k at fixed exposure
        prev_lo, prev_hi = lo, hi


def test_poisson_interval_is_nan_when_there_is_no_exposure():
    """Zero or negative expectation is not a rate of anything; returning 0/0 as a number
    would put a fabricated bound next to a real one in the frontier table."""
    for e in (0.0, -1.0, -0.001):
        lo, hi = poisson_ratio_ci(5, e)
        assert np.isnan(lo) and np.isnan(hi)


def test_poisson_interval_covers_a_known_rate_at_or_above_the_nominal_level():
    """Empirical coverage on draws from a rate the test knows, the same shape as the
    bootstrap coverage test above.

    The band is one-sided on purpose. Garwood is the EXACT interval, and exactness on a
    discrete distribution means conservative: the count cannot land on the boundary, so
    coverage sits at or above 95% and never at it. A version that under-covers is broken;
    one that over-covers is doing what it says. The seed is fixed, so this is a
    deterministic number rather than a draw that can wander over the band."""
    rng = np.random.default_rng(11)
    e, lam, sims = 25.0, 1.0, 2000
    ks = rng.poisson(lam * e, size=sims)
    hits = sum(lo <= lam <= hi for lo, hi in (poisson_ratio_ci(int(k), e) for k in ks))
    assert 0.95 <= hits / sims <= 1.0


# ── Controls ────────────────────────────────────────────────────────────
def test_global_shuffle_keeps_the_class_mix():
    y = np.array(["a"] * 30 + ["b"] * 70)
    s = shuffle_labels(y, seed=0)
    assert sorted(s) == sorted(y) and not np.array_equal(s, y)


def test_within_match_shuffle_preserves_every_match_composition():
    """The whole point of the second mode: per-match class counts survive exactly,
    so a leak travelling through match identity is not destroyed by the control."""
    rng = np.random.default_rng(2)
    g = np.repeat(np.arange(10), 12)
    y = rng.choice(["a", "b", "c"], size=120)
    s = shuffle_labels(y, groups=g, mode="within_match", seed=0)
    for m in np.unique(g):
        assert sorted(y[g == m]) == sorted(s[g == m])
    assert not np.array_equal(s, y)


def test_within_match_shuffle_requires_groups():
    with pytest.raises(ValueError):
        shuffle_labels(np.array(["a", "b"]), mode="within_match")


def _block_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                         "flag": rng.random(n) < 0.5, "cat": rng.choice(list("xyz"), n),
                         "keep": np.arange(n)})


def test_feature_shuffle_moves_the_block_together_and_nothing_else():
    """Joint, and dtype-preserving. Permuting each column on its own order would build rows
    that never happened, and a block assignment across mixed dtypes upcasts the bool, which
    the design then reads differently."""
    f = _block_frame()
    cols = ["a", "b", "flag", "cat"]
    o = shuffle_features(f, cols, mode="global", seed=1)
    assert list(o.dtypes.astype(str)) == list(f.dtypes.astype(str))
    assert o["keep"].equals(f["keep"])
    assert np.allclose(np.sort(o["a"]), np.sort(f["a"]))
    pairs = set(map(tuple, f[["a", "b"]].to_numpy()))
    assert all(tuple(r) in pairs for r in o[["a", "b"]].to_numpy())


def test_within_class_shuffle_keeps_the_class_conditional_marginal():
    """The property that makes Ojala & Garriga test 2 a NARROWER null than it reads as: a
    block predicting through its per-class marginals alone survives this control untouched,
    so a clean result here is not evidence that the block carries nothing."""
    rng = np.random.default_rng(3)
    f = _block_frame()
    y = rng.choice(["a", "b"], size=len(f))
    o = shuffle_features(f, ["a"], y=y, mode="within_class", seed=1)
    for c in np.unique(y):
        assert np.allclose(np.sort(o.loc[y == c, "a"]), np.sort(f.loc[y == c, "a"]))


def _planted(slope: float, seed: int = 0):
    """(groups, cross_fit) where every row's value is `slope` times a known predictor."""
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(80), 10)
    x = rng.normal(size=len(g))
    v = slope * x + rng.normal(0.0, 1.0, size=len(g))
    return g, (lambda r: (v + r.normal(0.0, 1e-9, size=len(g)), x))


def test_heterogeneity_recovers_a_planted_slope():
    """The positive control. Without it a null verdict is unreadable: a test that cannot see
    structure returns the same answer as data that has none."""
    g, cf = _planted(0.8, seed=0)
    r = heterogeneity_blp(g, cf, n_reps=12, seed=1)
    assert r["verdict"] == "heterogeneous" and r["actionable"]
    assert abs(r["blp_slope"] - 0.8) < 0.2 and r["blp_p"] < 0.025


def test_heterogeneity_stays_silent_on_noise():
    g, cf = _planted(0.0, seed=3)
    r = heterogeneity_blp(g, cf, n_reps=12, seed=1)
    assert r["verdict"] == "no detectable structure" and not r["actionable"]


def test_heterogeneity_refuses_a_cross_fit_of_the_wrong_length():
    """The contract is one d and one d_hat per ROW. A cross_fit that returns only the test
    half of a split is the single-split version of this test, and it would otherwise line up
    silently against the wrong groups."""
    g, _ = _planted(0.8, seed=0)
    with pytest.raises(ValueError):
        heterogeneity_blp(g, lambda r: (np.zeros(5), np.zeros(5)), n_reps=2)


def test_pooling_calls_identical_cells_one_cell():
    """Cells drawn from one distribution have to come back as one cell wearing several names,
    or every later subgroup claim is unreadable. The verdict is checked and tau is NOT: this
    draw returns tau=0.08 under a null with four cells, which is the whole reason the verdict
    reads Q rather than tau."""
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(60), 8)
    v = rng.normal(size=len(g))
    cells = np.array(["a", "b", "c", "d"] * (len(g) // 4))
    r = partial_pooling_cells(v, g, cells)
    assert r["verdict"] == "one cell wearing several names" and r["q_p"] > 0.05
    assert r["cells"]["shrunk"].std() < r["cells"]["mean"].std()     # shrinkage pulled them in


def test_pooling_finds_cells_that_really_differ():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(60), 8)
    cells = np.array(["a", "b"] * (len(g) // 2))
    v = np.where(cells == "a", 1.0, -1.0) + rng.normal(0.0, 0.3, size=len(g))
    r = partial_pooling_cells(v, g, cells)
    assert r["verdict"] == "cells differ" and r["q_p"] < 0.05 and r["tau"] > 0


def test_pooling_verdict_comes_from_q_and_not_from_tau_clearing_zero():
    """DerSimonian-Laird returns a positive tau-squared whenever Q exceeds k-1, which happens
    about half the time under the null with few cells, so tau>0 is not evidence of anything.
    The largest cell effect this project has found is that case (evidence.md §17)."""
    rng = np.random.default_rng(7)
    g = np.repeat(np.arange(60), 8)
    cells = np.array(["a", "b"] * (len(g) // 2))
    seen = [partial_pooling_cells(rng.normal(size=len(g)), g, cells) for _ in range(40)]
    positive_tau = [r for r in seen if r["tau"] > 0]
    assert positive_tau, "the null should still produce tau>0 sometimes"
    assert all(r["verdict"] == "one cell wearing several names"
               for r in positive_tau if r["q_p"] >= 0.05)


def test_pooling_keeps_thin_cells_out_of_tau_but_still_reports_them():
    rng = np.random.default_rng(2)
    g = np.repeat(np.arange(60), 8)
    cells = np.array(["big"] * (len(g) - 5) + ["thin"] * 5)
    r = partial_pooling_cells(rng.normal(size=len(g)), g, cells, min_cell=20)
    thin = r["cells"].set_index("cell").loc["thin"]
    assert bool(thin["below_floor"]) and int(thin["n"]) == 5
    assert r["verdict"] == "not estimable"


def test_feature_shuffle_refuses_an_empty_or_unknown_block():
    f = _block_frame()
    with pytest.raises(ValueError):
        shuffle_features(f, [])
    with pytest.raises(KeyError):
        shuffle_features(f, ["nope"])
    with pytest.raises(ValueError):
        shuffle_features(f, ["a"], mode="within_class")


# ── Wilcoxon and seed stability ─────────────────────────────────────────
def test_wilcoxon_clusters_by_match_rather_than_counting_rows():
    """Nine corners from one match are not nine independent observations.

    The same signed differences, once as 8 matches and once as 8 rows spread over 72,
    must give the SAME p: the test sees match means either way. A row-level test would
    see 72 observations and report a far smaller p for identical evidence."""
    from validate import wilcoxon_paired
    d = np.array([0.02, 0.03, 0.01, 0.04, 0.02, 0.05, 0.03, 0.02])
    few = wilcoxon_paired(d, np.arange(8))
    many = wilcoxon_paired(np.repeat(d, 9), np.repeat(np.arange(8), 9))
    assert many["n_clusters"] == 8
    assert many["p_value"] == pytest.approx(few["p_value"])


def test_wilcoxon_separates_a_null_effect_from_a_small_precise_one():
    """The reason it is reported next to the bootstrap. Both of these have a tiny mean;
    one is noise around zero and the other is consistently positive, and an interval that
    covers zero looks the same on both."""
    from validate import wilcoxon_paired
    rng = np.random.default_rng(0)
    g = np.arange(40)
    noise = rng.normal(0.0, 0.05, 40)
    small = np.full(40, 0.004) + rng.normal(0.0, 0.001, 40)
    assert not wilcoxon_paired(noise, g)["significant"]
    assert wilcoxon_paired(small, g)["significant"]


def test_wilcoxon_is_non_operative_rather_than_wrong_on_too_few_clusters():
    """Under about six clusters the exact test cannot reach p < 0.05 at all, so a
    'not significant' would describe the sample size and not the effect."""
    from validate import wilcoxon_paired
    r = wilcoxon_paired(np.array([0.1, 0.2, 0.3]), np.arange(3))
    assert np.isnan(r["p_value"]) and not r["significant"] and "non-operative" in r["note"]


def test_seed_stability_reports_how_many_decimals_survive_the_seed():
    from validate import seed_stability
    steady = seed_stability(lambda s: 0.1234 + 1e-6 * s)
    jumpy = seed_stability(lambda s: 0.1 + 0.02 * s)
    assert steady["stable_digits"] > jumpy["stable_digits"]
    assert steady["spread"] < jumpy["spread"]
    assert seed_stability(lambda s: 0.5)["stable_digits"] == 6


def test_seed_stability_runs_every_seed_it_reports():
    from validate import seed_stability
    seen = []
    seed_stability(lambda s: (seen.append(s), 1.0)[1], seeds=(3, 7, 11))
    assert seen == [3, 7, 11]


# ── Design-matrix inspection ────────────────────────────────────────────

def _sick_frame() -> pd.DataFrame:
    """One column per pathology the inspector exists to name, plus a healthy one."""
    return pd.DataFrame({
        "healthy": [1.0, 2.0, 3.0, 4.0],
        "all_nan": [np.nan] * 4,
        "half_nan": [1.0, np.nan, np.nan, 2.0],
        "constant": [7.0] * 4,
        "infinite": [1.0, np.inf, 2.0, 3.0],
    })


def test_inspect_names_every_pathology_and_spares_the_healthy_column():
    r = inspect_design(_sick_frame(), show=False)
    assert r["all_nan"] == ["all_nan"]
    assert r["has_inf"] == ["infinite"]
    assert r["constant"] == ["constant"]
    assert r["nan_frac"] == {"all_nan": 1.0, "half_nan": 0.5}
    assert r["shape"] == (4, 5)


def test_inspect_reports_rather_than_raises():
    """The design deliberately does NOT raise: an all-NaN column inside a fold is
    handled by _fit_prep, and the intercept-only rung is a constant column on purpose.
    A guard here would fail rungs that are correct."""
    r = inspect_design(_sick_frame(), show=False)          # must not raise
    assert r["n_nan"] == 6                                # 4 all-NaN + 2 half-NaN


def test_inspect_range_ignores_the_infinity_it_flags():
    """min/max are taken over finite cells, so one inf cannot swallow the whole range."""
    lo, hi = inspect_design(_sick_frame(), show=False)["range"]
    assert (lo, hi) == (1.0, 7.0)


def test_inspect_reads_an_array_and_a_frame_the_same_way():
    """_design hands it a bare ndarray plus column names; the tests hand it a frame."""
    df = _sick_frame()
    a = inspect_design(df.to_numpy(float), list(df.columns), show=False)
    b = inspect_design(df, show=False)
    assert a == b


def test_formatted_block_names_every_pathology_and_truncates_the_long_lists():
    """The printed side of the inspector. Every other test here passes show=False, so
    without this one the formatter is the only code path in the module that nothing runs,
    and a report nobody can read is the same as no report."""
    text = _format_inspection(inspect_design(_sick_frame(), label="rung", show=False))
    assert "rung" in text and "shape=(4, 5)" in text
    assert "all_nan: all_nan" in text and "has_inf: infinite" in text
    assert "constant: constant" in text
    assert "range [1, 7]" in text                            # finite cells only, the inf excluded
    assert "most missing: all_nan 100.0%, half_nan 50.0%" in text     # worst first
    wide = pd.DataFrame({f"c{i}": [np.nan] * 3 for i in range(9)})
    assert "(+4)" in _format_inspection(inspect_design(wide, show=False))   # 9 columns, top=5


def test_inspect_prints_only_when_asked():
    """show=True is the switch HUDL_INSPECT flips; the default must stay silent, because
    _design is called once per rung per ablation group and would drown the run."""
    df = _sick_frame()
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        inspect_design(df, label="loud", show=True)
        inspect_design(df, label="quiet", show=False)
    assert "loud" in buf.getvalue() and "quiet" not in buf.getvalue()
