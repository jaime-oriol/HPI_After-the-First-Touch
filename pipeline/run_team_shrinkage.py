"""run_team_shrinkage — is any team's after-first-touch corner rate distinguishable from its league?

The per-team rate that opens the argument is a ratio of two small numbers. Ranking on it is
already barred by the split-half reliability evidence.md §12 publishes, but "unreliable" is not
the same claim as "this team is not better", and a reader is owed the second one measured rather
than asserted. This fits the hierarchical model that answers it, and reports what survives once
sampling noise is discounted.

    aft_goals_t ~ Binomial(corners_t, p_t)
    logit(p_t)  = mu + a_league[l(t)] + a_team[t]

Binomial-logit, team nested in league, non-centred, NUTS. The structure is the Soccer Factor Model
of Andorra & Göbel (arXiv 2412.05911). The prior scales are FIXED in advance rather than read off
the same data; S2 anchors them in the empirical between-group SD instead and S1 doubles them, so
what the priors are worth is measured instead of argued.

Three verdicts, fixed before the first fit:

    V1  there is between-team signal at all   mean sigma_team beats the within-league permutation
    V2  the named case differs from its league   its a_team HDI excludes zero
    V3  the named case is top-5 with confidence  P(rank <= 5) > 0.5

DIAGNOSTICS DECIDE WHAT IS REPORTED. M1 with any divergence, r_hat >= 1.01 or ess_bulk <= 400
reports its non-convergence and no result, an outcome the run ends on cleanly, and a population with
no goal at all has nothing to shrink and ends the same way; S1 and S2 publish
their figure beside their own convergence flag; the permutation null holds only the fits that
converged and says how many. TARGET_ACCEPT differs from the pre-registered value, and evidence.md
§12 records why. --kind free_kick_flank fits the same model on the per-team table of the flank free
kicks from x >= 85, the paper's second population.

    python pipeline/run_team_shrinkage.py
    python pipeline/run_team_shrinkage.py --kind free_kick_flank   # team_shrinkage_<kind>
    python pipeline/run_team_shrinkage.py --smoke   # the path and the clock, not the verdict
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import logit
from scipy.stats import spearmanr

if TYPE_CHECKING:
    import xarray as xr

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))
_DER = _REPO / "data" / "derived"

from build_table import kind_tag                                    # noqa: E402
from eda_build import COMBINED_KINDS                                # noqa: E402
from eda_second_ball import NAMED_CASES as CASES                    # noqa: E402

TARGET_ACCEPT = 0.99
# r_hat, ess_bulk and divergences: the three that decide whether a fit is reportable at all.
R_HAT_MAX, ESS_MIN = 1.01, 400


def log(msg: str) -> None:
    print(msg, flush=True)


def empirical_logit_sd(k: np.ndarray, n: np.ndarray, groups: np.ndarray) -> float:
    """Between-group SD of the logit rate, with the Haldane correction (+0.5)."""
    df = pd.DataFrame({"k": k, "n": n, "g": groups}).groupby("g")[["k", "n"]].sum()
    return float(np.std(logit((df["k"] + 0.5) / (df["n"] + 1.0)), ddof=1))


def build(k: np.ndarray, n: np.ndarray, lg: np.ndarray, n_lg: int, mu0: float,
          s_league: float, s_team: float) -> pm.Model:
    """The binomial-logit model, non-centred so the sampler does not fight the funnel."""
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=mu0, sigma=1.0)
        sigma_league = pm.HalfNormal("sigma_league", sigma=s_league)
        sigma_team = pm.HalfNormal("sigma_team", sigma=s_team)
        a_league = pm.Deterministic("a_league",
                                    pm.Normal("z_league", 0.0, 1.0, shape=n_lg) * sigma_league)
        a_team = pm.Deterministic("a_team", pm.Normal("z_team", 0.0, 1.0, shape=len(k)) * sigma_team)
        p = pm.Deterministic("p", pm.math.sigmoid(mu + a_league[lg] + a_team))
        pm.Binomial("goals", n=n, p=p, observed=k)
    return model


def fit(model: pm.Model, draws: int, tune: int, chains: int, seed: int) -> tuple:
    """Sample, and return the draws with the three diagnostics that decide reportability."""
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=chains,
                          target_accept=TARGET_ACCEPT,
                          random_seed=seed, progressbar=False, compute_convergence_checks=False)
    s = az.summary(idata, var_names=["mu", "sigma_league", "sigma_team", "a_league", "a_team"])
    diag = {"max_r_hat": float(s["r_hat"].max()), "min_ess_bulk": float(s["ess_bulk"].min()),
            "divergences": int(idata.sample_stats["diverging"].sum())}
    diag["converged"] = bool(diag["max_r_hat"] < R_HAT_MAX and diag["min_ess_bulk"] > ESS_MIN
                             and diag["divergences"] == 0)
    return idata, diag


def write_record_without_fit(rows: list, out: Path) -> None:
    """The record of a population no model was reported for: its counts and diagnostics, and an empty
    per-team and permutation table, so nobody takes the tables of an earlier fit for this one's."""
    pd.DataFrame(rows).to_parquet(out, index=False)
    for side in ("teams", "perm"):
        pd.DataFrame().to_parquet(out.with_name(f"{out.stem}_{side}{out.suffix}"))


def hdi95(draws: xr.DataArray) -> np.ndarray:
    """The 95% highest-density interval of each team, (teams, 2). arviz 1 names the level `prob` and returns
    the array itself; arviz 0 names it `hdi_prob` and returns a Dataset keyed by the variable."""
    if int(az.__version__.split(".")[0]) >= 1:
        return az.hdi(draws, prob=0.95).values
    return az.hdi(draws, hdi_prob=0.95)[draws.name].values


def team_table(idata: az.InferenceData, teams: pd.DataFrame) -> pd.DataFrame:
    """Per-team posterior rate, its own effect, and the rank distribution that rank hides.

    The rank is computed inside every posterior draw and then summarised, never by ordering the
    posterior means: ordering the means throws away exactly the uncertainty this model exists to
    show, and would hand back a ranking as confident-looking as the raw one."""
    p = idata.posterior["p"].stack(s=("chain", "draw")).values          # (teams, samples)
    a = idata.posterior["a_team"].stack(s=("chain", "draw")).values
    ranks = (-p).argsort(axis=0).argsort(axis=0) + 1                    # 1 = highest rate
    hdi_p, hdi_a = hdi95(idata.posterior["p"]), hdi95(idata.posterior["a_team"])
    out = teams[["league", "team", "corners_for", "aft_goals_for"]].copy()
    out["raw_rate"] = out["aft_goals_for"] / out["corners_for"]
    out["p_mean"], out["p_lo"], out["p_hi"] = p.mean(axis=1), hdi_p[:, 0], hdi_p[:, 1]
    out["a_team_lo"], out["a_team_hi"] = hdi_a[:, 0], hdi_a[:, 1]
    out["p_a_team_pos"] = (a > 0).mean(axis=1)
    out["rank_median"] = np.median(ranks, axis=1)
    out["rank_lo"], out["rank_hi"] = np.percentile(ranks, [2.5, 97.5], axis=1)
    out["p_top5"] = (ranks <= 5).mean(axis=1)
    return out


def permutation_p(sigmas: np.ndarray, real: float) -> float | None:
    """The (1 + #at least as extreme) / (1 + n) permutation p-value, or None on an empty null.

    The +1 on both sides is what keeps a permutation p-value from ever reaching zero: with n
    permutations it cannot go below 1/(n+1), and that floor is reported beside it rather than left
    for a reader to infer from a suspiciously round number."""
    if not len(sigmas):
        return None
    return float((1 + (sigmas >= real).sum()) / (1 + len(sigmas)))


def verdicts(case: dict, p_perm: float | None) -> dict:
    """V1-V3 by the rules in this module's docstring; V1 is None when nothing converged.

    V2 asks whether the HDI excludes zero on EITHER side, because a team measurably worse than its
    league is as much a finding as one measurably better."""
    return {"V1_between_team_signal": None if p_perm is None else bool(p_perm < 0.05),
            "V2_case_distinguishable": bool(case["a_team_hdi"][0] > 0 or case["a_team_hdi"][1] < 0),
            "V3_case_top5_confident": bool(case["p_top5"] > 0.5)}


def main() -> None:
    """Fit M1, price the named cases, run the two prior sensitivities and the permutation null."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="short chains and two permutations")
    ap.add_argument("--teams", type=str, default=None, help="the per-team table eda_second_ball writes")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--kind", choices=COMBINED_KINDS, default="corner",
                    help="the population whose per-team table eda_second_ball writes")
    args = ap.parse_args()
    draws, tune, chains, n_perm, perm_draws = ((200, 200, 2, 2, 200) if args.smoke
                                               else (2000, 2000, 4, 20, 1000))
    stem = f"team_shrinkage{kind_tag((args.kind,))}{'_smoke' if args.smoke else ''}"
    out = Path(args.out) if args.out else _DER / f"{stem}.parquet"
    grant = _DER / f"eda_teams_{args.kind}.parquet"
    src = Path(args.teams) if args.teams else grant
    if args.out is None and src.resolve() != grant.resolve():
        raise SystemExit(f"STOP: {src.name} is not the {args.kind} team table, so it may not write "
                         f"the published {out.name}; pass --out")
    if not src.exists():
        raise SystemExit(f"STOP: {src.name} absent; eda_second_ball.py --kind {args.kind} writes it")

    teams = pd.read_parquet(src).reset_index(drop=True)
    k = teams["aft_goals_for"].to_numpy(int)
    n = teams["corners_for"].to_numpy(int)
    leagues = teams["league"].to_numpy()
    lg_names = sorted(set(leagues))
    lg = np.array([lg_names.index(v) for v in leagues])
    log(f"{len(teams)} teams, {len(lg_names)} leagues, {k.sum()} aft goals / {n.sum()} corners, "
        f"pooled rate {k.sum() / n.sum():.4f}")
    rows = [{"block": "population", "key": "teams", "n": len(teams), "point": float(k.sum() / n.sum())},
            {"block": "population", "key": "goals", "n": int(k.sum())},
            {"block": "population", "key": "corners", "n": int(n.sum())}]
    if not k.sum():
        # The pooled logit is minus infinity and the sampler cannot start: a thin subset of the flank
        # free kicks gets here. The record says so, like a fit that did not converge.
        write_record_without_fit(rows, out)
        log("no after-first-touch goal in this population: nothing to shrink; the record says so")
        return
    mu0 = float(logit(k.sum() / n.sum()))

    t0 = time.time()
    m1 = build(k, n, lg, len(lg_names), mu0, 0.5, 0.5)
    # Prior predictive check (Gabry et al. 2019): the mass the fixed scales put on per-corner rates
    # far above the pooled one. Recorded beside the fit; no verdict reads it.
    with m1:
        prior = pm.sample_prior_predictive(draws=1000, random_seed=1)
    pr = prior.prior["p"].values.ravel()
    rows += [{"block": "prior_predictive", "key": "share_rate_below_0.10",
              "point": float((pr < 0.10).mean())},
             {"block": "prior_predictive", "key": "share_rate_above_0.20",
              "point": float((pr > 0.20).mean())}]
    idata, diag = fit(m1, draws, tune, chains, seed=42)
    log(f"M1: {diag}  [{time.time() - t0:.1f}s]")
    rows += [{"block": "diagnostics", "key": f"M1/{key}", "point": float(v)}
             for key, v in diag.items()]
    if not diag["converged"] and not args.smoke:
        write_record_without_fit(rows, out)
        log("M1 did not converge: no result is reported (pre-registration); the record says so")
        return

    with m1:
        ppc = pm.sample_posterior_predictive(idata, random_seed=42, progressbar=False)
    g = ppc.posterior_predictive["goals"].stack(s=("chain", "draw")).values
    lo, hi = np.percentile(g, [2.5, 97.5], axis=1)
    rows.append({"block": "diagnostics", "key": "M1/ppc_coverage_95",
                 "point": float(((k >= lo) & (k <= hi)).mean())})
    tab = team_table(idata, teams)

    p = idata.posterior["p"].stack(s=("chain", "draw")).values
    idx = {t: int(np.flatnonzero(teams["team"].to_numpy() == t)[0]) for t in CASES}
    cases = {}
    for t, i in idx.items():
        r = tab.iloc[i]
        cases[t] = {"a_team_hdi": [float(r["a_team_lo"]), float(r["a_team_hi"])],
                    "p_top5": float(r["p_top5"])}
        log(f"{t}: raw {r['raw_rate']:.4f}, posterior {r['p_mean']:.4f} "
            f"[{r['p_lo']:.4f}, {r['p_hi']:.4f}], rank {r['rank_median']:.0f} "
            f"[{r['rank_lo']:.0f}, {r['rank_hi']:.0f}], P(top5) {r['p_top5']:.4f}")
        rows += [{"block": "case", "key": f"{t}/{col}", "n": int(r["corners_for"]),
                  "point": float(r[col])}
                 for col in ("raw_rate", "p_mean", "p_lo", "p_hi", "a_team_lo", "a_team_hi",
                             "p_a_team_pos", "rank_median", "rank_lo", "rank_hi", "p_top5")]
    rows.append({"block": "case", "key": "p_first_above_second",
                 "point": float((p[idx[CASES[0]]] > p[idx[CASES[1]]]).mean())})

    raw_rank = tab["raw_rate"].rank(ascending=False)
    post_rank = tab["p_mean"].rank(ascending=False)
    rows += [{"block": "shrinkage", "key": "spearman_raw_vs_posterior",
              "point": float(spearmanr(tab["raw_rate"], tab["p_mean"])[0])},
             {"block": "shrinkage", "key": "teams_moving_over_10_places",
              "n": int(((raw_rank - post_rank).abs() > 10).sum())}]

    for name, (s_lg, s_tm) in {"S1": (1.0, 1.0),
                               "S2": (empirical_logit_sd(k, n, leagues),
                                      empirical_logit_sd(k, n, teams["team"].to_numpy()))}.items():
        idata_s, diag_s = fit(build(k, n, lg, len(lg_names), mu0, s_lg, s_tm), draws, tune, chains, 43)
        tab_s = team_table(idata_s, teams)
        log(f"{name}: converged {diag_s['converged']}, "
            f"P(top5) {tab_s.iloc[idx[CASES[0]]]['p_top5']:.4f}")
        rows += [{"block": "sensitivity", "key": f"{name}/converged",
                  "point": float(diag_s["converged"])},
                 {"block": "sensitivity", "key": f"{name}/prior_scale_league", "point": float(s_lg)},
                 {"block": "sensitivity", "key": f"{name}/prior_scale_team", "point": float(s_tm)},
                 {"block": "sensitivity", "key": f"{name}/case_p_top5",
                  "point": float(tab_s.iloc[idx[CASES[0]]]["p_top5"])},
                 {"block": "sensitivity", "key": f"{name}/spearman_with_M1",
                  "point": float(spearmanr(tab_s["p_mean"], tab["p_mean"])[0])}]

    # Exact reallocation of each league's goals over its corners: the hypergeometric is the
    # permutation of individual corner outcomes, not a resample of teams.
    rng = np.random.default_rng(2026)
    perm, t0 = [], time.time()
    for j in range(n_perm):
        k_perm = np.zeros_like(k)
        for li in range(len(lg_names)):
            members = np.flatnonzero(lg == li)
            k_perm[members] = rng.multivariate_hypergeometric(n[members], int(k[members].sum()))
        # A model per permutation: pm.set_data on the observed node left a None logp factor in
        # PyMC 5.25, and a rebuild costs seconds on this many rows.
        idata_p, diag_p = fit(build(k_perm, n, lg, len(lg_names), mu0, 0.5, 0.5),
                              perm_draws, perm_draws, chains, seed=100 + j)
        perm.append({"sigma_team_mean": float(idata_p.posterior["sigma_team"].mean()), **diag_p})
    perm = pd.DataFrame(perm)
    real = float(idata.posterior["sigma_team"].mean())
    null = perm.loc[perm["converged"], "sigma_team_mean"].to_numpy()
    p_perm = permutation_p(null, real)
    # The convergence filter can drop fits from either end of the null. `p_perm_all` keeps them all:
    # not the pre-registered rule, never decides, shows which way the filter moved it.
    everything = perm["sigma_team_mean"].to_numpy()
    p_all = permutation_p(everything, real)
    log(f"null: {len(null)} of {len(perm)} converged, real {real:.4f}, p_perm {p_perm}, "
        f"floor {1 / (1 + len(null)):.4f}, p over all {len(perm)}: {p_all:.4f}  "
        f"[{time.time() - t0:.1f}s]")
    rows += [{"block": "null", "key": "n_perm", "n": int(len(perm))},
             {"block": "null", "key": "n_converged", "n": int(len(null))},
             {"block": "null", "key": "sigma_team_real", "point": real},
             {"block": "null", "key": "p_perm", "point": p_perm},
             {"block": "null", "key": "p_floor", "point": round(1 / (1 + len(null)), 4)},
             {"block": "null", "key": "p_perm_all", "point": p_all, "n": int(len(everything))}]

    for key, state in verdicts(cases[CASES[0]], p_perm).items():
        log(f"  {key}: {state}")
        rows.append({"block": "verdict", "key": key, "params": str(state)})

    pd.DataFrame(rows).to_parquet(out, index=False)
    tab.to_parquet(out.with_name(f"{out.stem}_teams{out.suffix}"), index=False)
    perm.to_parquet(out.with_name(f"{out.stem}_perm{out.suffix}"), index=False)
    log(f"wrote {out}, its per-team table and its permutation table")


if __name__ == "__main__":
    main()
