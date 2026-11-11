"""eda_club_power — what can a club measure about its own corners, and how long does it take?

A club changes a routine and wants to know whether it works. This driver turns the per-corner noise
of five measures into the corners, and the seasons, a club needs to see a given lift, and says which
measure to read. Pre-registered (data/derived/prereg_2026-10-01/c12_club_principles.md).

THE MEASURES, per corner, own goals out of every count as in eda_top_teams: after-first-touch goal
(at least one), after-first-touch xG (summed over the shots, eda_top_teams' convention),
after-first-touch shot (at least one), total corner xG, and the attack's first contact, a delivery
that reaches no one counting as not won. For each: the base rate and spread; the intraclass
correlation between teams, and between matches inside a team (the clustering a club's own
before-and-after carries), each with a cluster-resampled interval; the Beta-Binomial
overdispersion κ of team rates on the binary measures; and the split-half reliability over the
teams, corner volume beside it as the positive control (eda_repeatability).

POWER, 80% at two-sided 5%, for a lift of 25%, 50% and 100%: between two teams, and before and after
a change inside one team, each in corners and in team-seasons at the median volume. A second route
checks the formula against the data: the spread of the gap between two random halves of each
team's season, the noise a club's own half-season before-and-after carries.

C12a decides whether the power figures can be quoted: under "finishing is luck once the chances are
known", a team's after-first-touch goals are a Poisson-binomial over its corners, each scoring with
1 − Π(1 − xG) over its shots (the chained-shot rule, sota.md `Anzer_Bauer_xG_2021`), the xG
recalibrated to the grant's own goals over xG on that route. The observed between-team variance of
the goal rate has to sit inside the 95% interval of that variance at five seeds. A κ fitted on the
goals themselves would make that check pass by construction, so it conditions on the chances alone,
the sheet's recorded deviation, and κ travels beside it with the raw-xG version. --kind
free_kick_flank reads the flank free kicks from x >= 85 the same way, the paper's second population.

    python pipeline/eda_club_power.py --jobs 23
    python pipeline/eda_club_power.py --jobs 23 --kind free_kick_flank     # club_power_*_<kind>
    python pipeline/eda_club_power.py --sims 400 --boot 50 --splits 20 --seeds 2 --out-dir data/derived/smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy import optimize, stats

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))
_DER = _REPO / "data" / "derived"

from build_table import kind_tag                        # noqa: E402
from eda_build import COMBINED_KINDS, kind_inputs       # noqa: E402
from eda_repeatability import _team_draws, split_half   # noqa: E402
from s01_phases import AFTER_FIRST_TOUCH                # noqa: E402
from validate import Z80                                # noqa: E402

MEASURES = ("aft_goal", "aft_xg", "aft_shot", "xg", "first_contact")
BINARY = ("aft_goal", "aft_shot", "first_contact")
LIFTS = (0.25, 0.50, 1.00)
# The registered precision, and the only one the published records are written at: 10000 simulated
# seasons per seed for C12a, 2000 cluster draws for every interval (§12's), 500 match splits.
_SEASON_SIMS, _CLUSTER_DRAWS, _SPLITS, _SEEDS = 10000, 2000, 500, 5
_SIM_CHUNK = 500


# ── The per-corner table ──────────────────────────────────────────────────────────
def corner_table(deliv: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """One row per corner: the five measures and the chained after-first-touch goal probability."""
    s = shots[shots["shot_outcome"] != "Own Goal"]
    aft = s[s["segment"].isin(AFTER_FIRST_TOUCH)]
    by = aft.groupby("set_piece_id")
    t = deliv[["set_piece_id", "league", "attacking_team", "match_id"]].copy()
    t["aft_goal"] = t["set_piece_id"].map(by["is_goal"].max()).fillna(0).astype(float)
    t["aft_xg"] = t["set_piece_id"].map(by["xg"].sum()).fillna(0.0)
    t["aft_shot"] = t["set_piece_id"].isin(aft["set_piece_id"]).astype(float)
    t["xg"] = t["set_piece_id"].map(s.groupby("set_piece_id")["xg"].sum()).fillna(0.0)
    t["first_contact"] = deliv["won_first_contact"].eq(True).astype(float).to_numpy()
    t["aft_xg_list"] = t["set_piece_id"].map(by["xg"].agg(list))
    t["aft_xg_list"] = t["aft_xg_list"].apply(lambda v: v if isinstance(v, list) else [])
    return t.reset_index(drop=True)


def chained(xg_lists: pd.Series, factor: float = 1.0) -> np.ndarray:
    """Per corner, the chance of at least one goal from its shots: 1 − Π(1 − factor·xG)."""
    return np.array([1.0 - np.prod(1.0 - np.clip(factor * np.asarray(v, float), 0.0, 1.0))
                     for v in xg_lists])


# ── Clustering, overdispersion, power ─────────────────────────────────────────────
def icc(x: np.ndarray, clusters: np.ndarray) -> float:
    """One-way ANOVA intraclass correlation for unbalanced clusters, untruncated."""
    g = pd.Series(x).groupby(clusters)
    k, n, sizes = g.ngroups, len(x), g.size().to_numpy(float)
    m0 = (n - (sizes ** 2).sum() / n) / (k - 1)
    msb = (sizes * (g.mean().to_numpy() - x.mean()) ** 2).sum() / (k - 1)
    msw = ((x - g.transform("mean").to_numpy()) ** 2).sum() / (n - k)
    return float((msb - msw) / (msb + (m0 - 1) * msw))


def icc_draws(x: np.ndarray, clusters: np.ndarray, n_boot: int, seed: list) -> np.ndarray:
    """The ICC over clusters resampled with replacement, each draw relabelling its copies."""
    rng, uniq = np.random.default_rng(seed), pd.unique(clusters)
    idx = {c: np.flatnonzero(clusters == c) for c in uniq}
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = [idx[uniq[j]] for j in pick]
        out[b] = icc(x[np.concatenate(rows)], np.repeat(np.arange(len(rows)), [len(r) for r in rows]))
    return out


def beta_binomial(k: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    """(α, β) maximising the Beta-Binomial likelihood of k successes in n trials per team."""
    p = k.sum() / n.sum()
    start = np.log([max(p, 1e-3) * 50.0, max(1 - p, 1e-3) * 50.0])

    def nll(theta: np.ndarray) -> float:
        a, b = np.exp(theta)
        return -float(stats.betabinom.logpmf(k, n, a, b).sum())

    a, b = np.exp(optimize.minimize(nll, start, method="Nelder-Mead",
                                    options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000}).x)
    return float(a), float(b)


def corners_needed(mu: float, sd: float, lift: float, deff: float, binary: bool) -> float:
    """Corners per arm for 80% power at two-sided 5% to see `lift` on a rate of `mu`: the two
    binomial variances of the classic two-proportion test on a binary measure, one common spread on
    a continuous one (letting it grow with the mean would let the noise keep pace with any lift).
    A rate of zero has no lift to see, so the corners are NaN."""
    mu2 = mu * (1 + lift)
    if mu2 == mu:
        return np.nan
    var = mu * (1 - mu) + mu2 * (1 - mu2) if binary else 2 * sd ** 2
    return float(deff * Z80 ** 2 * var / (mu2 - mu) ** 2)


# ── C12a: the between-team variance under luck given the chances ─────────────────
def team_matrix(t: pd.DataFrame) -> tuple[np.ndarray, pd.MultiIndex]:
    """(n_corners, n_teams) one-hot of each corner's team, and the team index."""
    teams = pd.MultiIndex.from_frame(t[["league", "attacking_team"]].drop_duplicates()
                                     .sort_values(["league", "attacking_team"]))
    ix = teams.get_indexer(pd.MultiIndex.from_frame(t[["league", "attacking_team"]]))
    out = np.zeros((len(t), len(teams)))
    out[np.arange(len(t)), ix] = 1.0
    return out, teams


def simulated_variance(p: np.ndarray, onehot: np.ndarray, n_sims: int, seed: list) -> np.ndarray:
    """(n_sims, n_teams) simulated goal rates per team: every corner scores with its p, drawn in
    chunks; a column left out later gives the leave-one-team-out reading from the same draws."""
    rng, n = np.random.default_rng(seed), onehot.sum(axis=0)
    out = [((rng.random((min(_SIM_CHUNK, n_sims - i), len(p))) < p) @ onehot) / n
           for i in range(0, n_sims, _SIM_CHUNK)]
    return np.vstack(out)


def c12a_reading(obs: np.ndarray, sims: np.ndarray, keep: np.ndarray | None = None) -> dict:
    """Observed between-team variance of the rate against the simulated interval."""
    keep = np.ones(len(obs), bool) if keep is None else keep
    v_obs, v = float(np.var(obs[keep], ddof=1)), np.var(sims[:, keep], axis=1, ddof=1)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return {"v_obs": v_obs, "v_lo": float(lo), "v_hi": float(hi), "v_mean": float(v.mean()),
            "inside": bool(lo <= v_obs <= hi), "p_upper": float((1 + (v >= v_obs).sum()) / (1 + len(v)))}


def luck_share(luck: float, v_obs: float) -> float:
    """The share of the observed between-team variance that luck alone gives, NaN when the teams do
    not vary at all (no team scoring after the first touch)."""
    return luck / v_obs if v_obs > 0 else np.nan


def _c12a_task(p: np.ndarray, onehot: np.ndarray, obs: np.ndarray, leagues: np.ndarray,
               n_sims: int, seed: int, version: str) -> list:
    """One seed's C12a reading over every team; seed 0 also carries each league and each team left
    out, from the same draws."""
    sims = simulated_variance(p, onehot, n_sims, [seed, 0 if version == "recalibrated" else 1])
    rows = [{"seed": seed, "version": version, "scope": "all", **c12a_reading(obs, sims)}]
    if seed == 0:
        rows += [{"seed": 0, "version": version, "scope": f"without_{lg}",
                  **c12a_reading(obs, sims, leagues != lg)} for lg in pd.unique(leagues)]
        rows += [{"seed": 0, "version": version, "scope": f"without_team_{j}",
                  **c12a_reading(obs, sims, np.arange(len(obs)) != j)} for j in range(len(obs))]
    return rows


# ── The per-measure figures ───────────────────────────────────────────────────────
def _measure_task(t: pd.DataFrame, m: str, n_boot: int, seed: int) -> dict:
    """Spread, both intraclass correlations with their intervals, κ on a binary measure."""
    x = t[m].to_numpy(float)
    team = (t["league"] + "|" + t["attacking_team"]).to_numpy()
    within = x - pd.Series(x).groupby(team).transform("mean").to_numpy()
    tm = (team + "|" + t["match_id"].astype(str)).to_numpy()
    out = {"measure": m, "mu": float(x.mean()), "sd": float(x.std(ddof=1)),
           "icc_team": icc(x, team), "icc_match_in_team": icc(within, tm),
           "corners_per_team_match": float(pd.Series(tm).value_counts().mean())}
    for name, v, cl, k in (("icc_team", x, team, 1), ("icc_match_in_team", within, tm, 2)):
        d = icc_draws(v, cl, n_boot, [seed, k, MEASURES.index(m)])
        out[f"{name}_lo"], out[f"{name}_hi"] = np.percentile(d, [2.5, 97.5])
    if m in BINARY:
        g = t.groupby(["league", "attacking_team"])[m].agg(["sum", "size"])
        k_, n_ = g["sum"].to_numpy(), g["size"].to_numpy()
        a, b = beta_binomial(k_, n_)
        draws = _team_draws(g.index.get_level_values(0).to_numpy(), n_boot,
                            np.random.default_rng([seed, 3, MEASURES.index(m)]))
        kap = np.array([sum(beta_binomial(k_[d], n_[d])) for d in draws])
        lo, hi = np.percentile(kap, [2.5, 97.5])
        # κ has no ceiling where the teams show no overdispersion; ρ = 1/(1 + κ), the share of the
        # variance between teams, is bounded and reads the same draws.
        out.update({"kappa": a + b, "kappa_lo": float(lo), "kappa_hi": float(hi),
                    "rho": 1.0 / (1.0 + a + b), "rho_lo": float(1.0 / (1.0 + hi)),
                    "rho_hi": float(1.0 / (1.0 + lo))})
    return out


def power_rows(m: dict, season: pd.Series) -> list:
    """Corners and team-seasons to see each lift, between two teams and before and after inside one,
    with the clustering of corners inside a team's match in the design effect."""
    deff = 1 + (m["corners_per_team_match"] - 1) * max(m["icc_match_in_team"], 0.0)
    rows = []
    for lift in LIFTS:
        n = corners_needed(m["mu"], m["sd"], lift, deff, m["measure"] in BINARY)
        rows.append({"measure": m["measure"], "lift": lift, "deff": deff, "corners_per_arm": n,
                     "seasons_between_teams": n / season.median(),
                     "seasons_before_after": 2 * n / season.median(),
                     "seasons_before_after_p10": 2 * n / season.quantile(0.1),
                     "seasons_before_after_p90": 2 * n / season.quantile(0.9)})
    return rows


def first_half(t: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """True where a corner's match falls in its own team's first half of one random split, half of
    each team's matches rounded down. A match is split per team, so the two sides of one match may
    land in different halves."""
    team = t["league"] + "|" + t["attacking_team"]
    first = {f"{tm}|{mid}" for tm, mids in t.groupby(team)["match_id"].unique().items()
             for mid in rng.permutation(mids)[:len(mids) // 2]}
    return (team + "|" + t["match_id"].astype(str)).isin(first).to_numpy()


def half_season_noise(t: pd.DataFrame, measures: pd.DataFrame, deffs: dict, n_splits: int,
                      seed: int) -> pd.DataFrame:
    """The second route to the power: per measure, the SD over teams of the gap between two random
    halves of each team's matches, against the SD the noise model the formula uses predicts (each
    team's halves, the design effect, the variance at the base rate), and the lift at 80% power a
    half-season before-and-after can see on each route. A team with one match has no second half,
    so both routes read the teams found on both sides of each split."""
    rng = np.random.default_rng([seed, 4])
    team = t["league"] + "|" + t["attacking_team"]
    gaps = {m: [] for m in MEASURES}
    n_a, n_b = [], []
    for _ in range(n_splits):
        a = first_half(t, rng)
        rate_a = t[a].groupby(team[a])[list(MEASURES)].mean()
        rate_b = t[~a].groupby(team[~a])[list(MEASURES)].mean()
        both = rate_a.index.intersection(rate_b.index)
        n_a.append(team[a].value_counts().reindex(both).to_numpy(float))
        n_b.append(team[~a].value_counts().reindex(both).to_numpy(float))
        for m in MEASURES:
            gaps[m].append(float((rate_a.loc[both, m] - rate_b.loc[both, m]).std(ddof=1)))
    inv = float(np.mean([np.mean(1 / x + 1 / y) for x, y in zip(n_a, n_b, strict=True)]))
    rows = []
    for m in measures.to_dict("records"):
        var0 = m["mu"] * (1 - m["mu"]) if m["measure"] in BINARY else m["sd"] ** 2
        sd_obs = float(np.mean(gaps[m["measure"]]))
        rows.append({"measure": m["measure"], "half_gap_sd": sd_obs,
                     "half_gap_sd_predicted": float(np.sqrt(deffs[m["measure"]] * var0 * inv)),
                     "half_season_mde80_lift": (float(Z80 * sd_obs / m["mu"]) if m["mu"] > 0
                                                else np.nan)})
    return pd.DataFrame(rows)


def matching_lift(m: dict, corners: float, binary: bool, deff: float) -> float:
    """The lift the formula sees at 80% with `corners` per arm, for the second route to compare; a
    binary rate cannot be lifted past 1, so its search stops short of it, NaN if out of reach or if
    the rate is zero."""
    if not m["mu"] > 0:
        return np.nan
    top = min(50.0, 1.0 / m["mu"] - 1.0 - 1e-6) if binary else 50.0
    def shortfall(lift: float) -> float:
        return corners_needed(m["mu"], m["sd"], lift, deff, binary) - corners

    return float(optimize.brentq(shortfall, 1e-4, top)) if shortfall(top) < 0 else np.nan


# ── The run ───────────────────────────────────────────────────────────────────────
def main() -> None:
    """Corner table, per-measure figures, power by two routes, C12a at every seed, reliability."""
    ap = argparse.ArgumentParser()
    d_def, s_def = kind_inputs("corner")
    ap.add_argument("--deliveries", type=str, default=str(d_def))
    ap.add_argument("--shots", type=str, default=str(s_def))
    ap.add_argument("--sims", type=int, default=_SEASON_SIMS)
    ap.add_argument("--boot", type=int, default=_CLUSTER_DRAWS)
    ap.add_argument("--splits", type=int, default=_SPLITS)
    ap.add_argument("--seeds", type=int, default=_SEEDS)
    ap.add_argument("--jobs", type=int, default=20, help="workers; the output does not depend on it")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="where to write (default: data/derived, only from the grant's tables at the "
                         "registered precision)")
    ap.add_argument("--kind", choices=COMBINED_KINDS, default="corner",
                    help="the population: corners, or the flank free kicks from x >= 85")
    args = ap.parse_args()
    kt = kind_tag((args.kind,))
    out = Path(args.out_dir) if args.out_dir else _DER
    precision = (args.sims, args.boot, args.splits, args.seeds)
    inputs = [Path(p).resolve() for p in (args.deliveries, args.shots)]
    if out.resolve() == _DER.resolve() and (
            precision != (_SEASON_SIMS, _CLUSTER_DRAWS, _SPLITS, _SEEDS)
            or inputs != [p.resolve() for p in (d_def, s_def)]):
        raise SystemExit(f"STOP: the published club_power records come only from the grant's tables at "
                         f"(sims, boot, splits, seeds) = {(_SEASON_SIMS, _CLUSTER_DRAWS, _SPLITS, _SEEDS)}; "
                         f"pass --out-dir")

    deliv = pd.read_parquet(args.deliveries)
    deliv = deliv[deliv["set_piece_type"] == args.kind].reset_index(drop=True)
    shots = pd.read_parquet(args.shots)
    shots = shots[shots["set_piece_type"] == args.kind]
    t = corner_table(deliv, shots)
    season = t.groupby(["league", "attacking_team"]).size()
    print(f"{len(t)} {args.kind} deliveries, {t['match_id'].nunique()} matches, {len(season)} "
          f"team-seasons; per team-season median {season.median():.1f} (p10 "
          f"{season.quantile(.1):.1f}, p90 {season.quantile(.9):.1f})", flush=True)
    aft = shots[(shots["shot_outcome"] != "Own Goal") & shots["segment"].isin(AFTER_FIRST_TOUCH)]
    factor = float(aft["is_goal"].sum() / aft["xg"].sum())
    print(f"after-first-touch goals over xG over the grant: {factor:.3f} "
          f"({int(aft['is_goal'].sum())} goals, {aft['xg'].sum():.1f} xG)", flush=True)

    onehot, teams = team_matrix(t)
    obs = (t["aft_goal"].to_numpy() @ onehot) / onehot.sum(axis=0)
    leagues = teams.get_level_values(0).to_numpy()
    probs = {"recalibrated": chained(t["aft_xg_list"], factor), "raw_xg": chained(t["aft_xg_list"])}
    tasks = [delayed(_measure_task)(t, m, args.boot, 0) for m in MEASURES]
    tasks += [delayed(_c12a_task)(probs[v], onehot, obs, leagues, args.sims, s, v)
              for v in probs for s in range(args.seeds)]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        done = Parallel(n_jobs=args.jobs, verbose=2)(tasks)
    measures = pd.DataFrame(done[:len(MEASURES)])
    c12a = pd.DataFrame([r for d in done[len(MEASURES):] for r in d])
    team_names = {f"without_team_{j}": f"{lg}|{tm}" for j, (lg, tm) in enumerate(teams)}
    c12a["scope"] = c12a["scope"].replace(team_names)

    power = pd.DataFrame([r for m in measures.to_dict("records") for r in power_rows(m, season)])
    deffs = power.groupby("measure")["deff"].first().to_dict()
    noise = half_season_noise(t, measures, deffs, args.splits, 0)
    half = season.median() / 2
    noise["formula_lift_at_half_season"] = [
        matching_lift(m, half, m["measure"] in BINARY, deffs[m["measure"]])
        for m in measures.to_dict("records")]

    p = probs["recalibrated"]
    luck = float(np.mean(((p * (1 - p)) @ onehot) / onehot.sum(axis=0) ** 2))
    main_rows = c12a[(c12a["scope"] == "all") & (c12a["version"] == "recalibrated")]
    decision = {"C12a": bool(main_rows["inside"].all()),
                "C12a_any_seed": bool(main_rows["inside"].any()), "seeds": int(len(main_rows)),
                "v_obs": float(main_rows["v_obs"].iloc[0]),
                "luck_share": luck_share(luck, float(main_rows["v_obs"].iloc[0])),
                "raw_xg_inside_all_seeds": bool(c12a[(c12a["scope"] == "all")
                                                     & (c12a["version"] == "raw_xg")]["inside"].all()),
                "aft_goals_over_xg": factor, "season_median": float(season.median())}

    scores = t[["attacking_team", "match_id", *MEASURES]]
    rel = split_half(deliv, shots, args.splits, scores=scores, n_boot=args.boot,
                     boot_cols=["corners_per_match", *MEASURES])

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print("\nPER MEASURE\n" + measures.round(4).to_string(index=False))
    print("\nPOWER (80%, two-sided 5%)\n" + power.round(2).to_string(index=False))
    print("\nSECOND ROUTE, the half-season gap\n" + noise.round(4).to_string(index=False))
    print("\nC12a by seed\n" + c12a[c12a["scope"] == "all"].round(6).to_string(index=False))
    print("\nC12a, each league left out (seed 0)\n" + c12a[c12a["scope"].str.startswith("without_")]
          .round(6).to_string(index=False))
    loo = c12a[c12a["scope"].str.contains(r"\|") & (c12a["version"] == "recalibrated")]
    print(f"\nC12a, each team left out (seed 0): inside {int(loo['inside'].sum())} of {len(loo)}; v_obs "
          f"{loo['v_obs'].min():.6f} to {loo['v_obs'].max():.6f}")
    print("\nRELIABILITY (split-half, Spearman-Brown)\n" + rel.round(3).to_string(index=False))
    print("\nDECISION\n" + "\n".join(f"  {k:28s} {v}" for k, v in decision.items()), flush=True)

    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("measures", measures), ("power", power), ("half_season", noise),
                        ("c12a", c12a), ("reliability", rel), ("decision", pd.DataFrame([decision]))):
        frame.to_parquet(out / f"club_power_{name}{kt}.parquet", index=False)
    print(f"\nsaved the six club_power{kt} records in {out}")


if __name__ == "__main__":
    main()
