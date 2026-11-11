"""eda_top_teams — do the best sides, and the best corner sides, live after the first touch?

The paper opens on Arsenal (argument.md §2): most of its corner goals came after the first contact.
This driver asks it of two groups fixed before looking (data/derived/prereg_2026-10-01/top_teams.md):
the top 2 of each league by final position (group T) and the 10 teams of the grant with the most
corner goals (group G), each against the rest, by route: direct, after the first touch, late.

Own goals are out of every count. The frontier counts them as goals the delivery produced
(evidence.md §2); here they would be credited to a team's finishing. A team rate is pooled over the
team's corners, and the teams are the sample of every interval (evidence.md §12), resampled inside
their group. The sheet's registered draws go inside each league and group as well, which leaves
group T two teams per cell, where a bootstrap halves the between-team variance (one team leaves
none); that reading travels beside as `*_registered` and decides nothing. Group G is chosen on
goals, so its goal-based contrasts are inflated by construction, and its decision reads the
split-half version: chosen on one half of each team's matches and measured on the other, where the
membership moves from split to split and the draws resample teams over the whole grant. Its null
relabels the teams once per draw, the same way in every split, which keeps how persistently a team
is chosen.

Performance is goals against the shot-level xG, tested exactly (Poisson-binomial), and again against
xG recalibrated by route to the goals over xG of every other team: the provider's xG over-prices the
direct set-piece shot (evidence.md §2), so a raw under-performance on direct shots is partly the
provider's, and leaving the team out keeps its own finishing out of its yardstick.

Every contrast runs at five stability seeds, as the registered rules of evidence.md §20 and §21 do,
and a decision holds only when its decisive bound keeps its side at all five. The battery is a set of
seeded tasks, each single-threaded, so a run over the cores reproduces the serial one exactly.

The sheet registered corners. --kind free_kick_flank runs the same battery on the flank free kicks
from x >= 85, the paper's second population, without the long-corner scope a free kick never needs.

    python pipeline/eda_top_teams.py --jobs 23
    python pipeline/eda_top_teams.py --jobs 23 --kind free_kick_flank     # top_teams_*_<kind>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy import stats

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "pipeline"))
_DER = _REPO / "data" / "derived"

from build_table import kind_tag                        # noqa: E402
from constants import RELIABILITY_FLOOR                 # noqa: E402
from eda_build import COMBINED_KINDS, kind_inputs       # noqa: E402
from eda_repeatability import _BOOT as _REL_BOOT        # noqa: E402  §12's draws, §12's interval
from eda_repeatability import split_half                # noqa: E402
from s01_phases import AFTER_FIRST_TOUCH                # noqa: E402
from validate import Z80, Z95                           # noqa: E402

# The registered precision, and the only one the published records are written at.
_TEAM_DRAWS, _PERMS, _SPLITS, _SPLIT_PERMS, _SEEDS = 10000, 100000, 500, 10000, 5
_PERM_CHUNK = 20000
ROUTES = ("direct", "aft", "late")
TOP_T, TOP_G, CUTS = 2, 10, (1, 2, 3, 4)
# T2's margin: a quarter of the grant's pooled rate, fixed in the sheet.
MARGIN_FRACTION = 0.25
PRIMARY = "aft_xg_per_corner"
# name -> (numerator, denominator), each summed over a group's teams before dividing.
RATES = {
    "aft_xg_per_corner": ("xg_aft", "corners"),
    "direct_xg_per_corner": ("xg_direct", "corners"),
    "late_xg_per_corner": ("xg_late", "corners"),
    "xg_per_corner": ("xg_total", "corners"),
    "aft_xg_share": ("xg_aft", "xg_total"),
    "corners_per_match": ("corners", "matches"),
    "aft_corner_share": ("corners_aft", "corners"),
    "direct_corner_share": ("corners_direct", "corners"),
    "aft_shots_per_corner": ("shots_aft", "corners"),
    "direct_shots_per_corner": ("shots_direct", "corners"),
    "aft_goals_per_corner": ("goals_aft", "corners"),
    "direct_goals_per_corner": ("goals_direct", "corners"),
    "aft_perf_per_corner": ("perf_aft", "corners"),
    "direct_perf_per_corner": ("perf_direct", "corners"),
    "aft_perf_cal_per_corner": ("perf_cal_aft", "corners"),
    "direct_perf_cal_per_corner": ("perf_cal_direct", "corners"),
}
SPLIT_RATES = ("aft_xg_per_corner", "direct_xg_per_corner", "aft_xg_share",
               "aft_perf_cal_per_corner")


# ── The season table ─────────────────────────────────────────────────────────────
def routes(shots: pd.DataFrame) -> pd.DataFrame:
    """Corner shots without own goals, each on its route."""
    s = shots[shots["shot_outcome"] != "Own Goal"].copy()
    s["route"] = np.select([s["segment"] == "direct", s["segment"].isin(AFTER_FIRST_TOUCH)],
                           ["direct", "aft"], "late")
    return s


def lto_factors(s: pd.DataFrame, teams: pd.MultiIndex) -> pd.DataFrame:
    """Goals over xG on each route over every OTHER team, one row per team, so the factor that
    recalibrates a team's xG never reads that team's own finishing."""
    per = s.groupby(["league", "attacking_team", "route"])[["is_goal", "xg"]].sum().unstack(
        "route", fill_value=0.0).reindex(columns=pd.MultiIndex.from_product([["is_goal", "xg"], ROUTES]),
                                         fill_value=0.0)
    per.index.names = ["league", "team"]
    per = per.reindex(teams, fill_value=0.0)
    tot = per.sum()
    return pd.DataFrame({r: (tot[("is_goal", r)] - per[("is_goal", r)])
                         / (tot[("xg", r)] - per[("xg", r)]) for r in ROUTES}, index=teams)


def poisson_binomial_p(probs: np.ndarray, k: int) -> float:
    """Two-sided exact p-value of k goals from independent shots scoring with `probs`."""
    pmf = np.array([1.0])
    for p in probs:
        pmf = np.convolve(pmf, [1.0 - p, p])
    return float(min(1.0, 2.0 * min(pmf[:k + 1].sum(), pmf[k:].sum())))


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty(len(p))
    out[order] = q.clip(max=1.0)
    return out


def team_match_rows(deliv: pd.DataFrame, s: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Per (team, match): corners by the route of their first shot, and shots, xG, goals and
    performance by route, the unit the season table and the split-half halves both sum."""
    first = (s.sort_values("t_since_delivery_s", kind="stable")
             .groupby("set_piece_id")["route"].first())
    d = deliv.assign(first_route=deliv["set_piece_id"].map(first).fillna("none"))
    key = ["league", "attacking_team", "match_id"]
    out = d.groupby(key).size().rename("corners").to_frame()
    for r in (*ROUTES, "none"):
        out[f"corners_{r}"] = d[d["first_route"] == r].groupby(key).size()
    for r in ROUTES:
        g = s[s["route"] == r].groupby(key)
        out[f"shots_{r}"], out[f"xg_{r}"], out[f"goals_{r}"] = g.size(), g["xg"].sum(), g["is_goal"].sum()
    out = out.fillna(0.0).reset_index().rename(columns={"attacking_team": "team"})
    f = factors.reindex(pd.MultiIndex.from_frame(out[["league", "team"]]))
    for r in ROUTES:
        out[f"perf_{r}"] = out[f"goals_{r}"] - out[f"xg_{r}"]
        out[f"perf_cal_{r}"] = out[f"goals_{r}"] - f[r].to_numpy() * out[f"xg_{r}"]
    out["xg_total"] = out[[f"xg_{r}" for r in ROUTES]].sum(axis=1)
    out["goals_total"] = out[[f"goals_{r}" for r in ROUTES]].sum(axis=1)
    return out


def played(matches: pd.DataFrame) -> pd.Series:
    """Matches each team played, read off the match results rather than off its corners."""
    both = pd.concat([matches[["league", side]].set_axis(["league", "team"], axis=1)
                      for side in ("home_team", "away_team")])
    return both.groupby(["league", "team"]).size()


def season_table(rows: pd.DataFrame, s: pd.DataFrame, matches: pd.DataFrame,
                 stand: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """One row per team: the season sums, its finishing position, and the exact test per route,
    raw and recalibrated, with its q-value over every team."""
    t = rows.drop(columns="match_id").groupby(["league", "team"]).sum()
    t["matches"] = played(matches)
    t = t.reset_index().merge(stand[["league", "team", "pos"]], on=["league", "team"],
                              how="left", validate="1:1")
    assert t[["pos", "matches"]].notna().all().all(), "a team without a position or a match count"
    shots = {k: g for k, g in s.groupby(["league", "attacking_team", "route"])}
    f = factors.reindex(pd.MultiIndex.from_frame(t[["league", "team"]]))
    for r in ROUTES:
        for tag, scale in (("", np.ones(len(t))), ("cal_", f[r].to_numpy())):
            p = np.array([poisson_binomial_p(np.clip(shots[(lg, tm, r)]["xg"].to_numpy() * k, 0, 1),
                                             int(shots[(lg, tm, r)]["is_goal"].sum()))
                          if (lg, tm, r) in shots else 1.0
                          for lg, tm, k in zip(t["league"], t["team"], scale, strict=True)])
            t[f"p_{tag}{r}"], t[f"q_{tag}{r}"] = p, bh(p)
    return t


def top_by(t: pd.DataFrame, col: str, tie: str, n: int) -> np.ndarray:
    """The n teams highest on `col`, ties broken by `tie` and then by name."""
    order = t.sort_values([col, tie, "team"], ascending=[False, False, True]).index[:n]
    return t.index.isin(order)


# ── Contrasts ────────────────────────────────────────────────────────────────────
def draw_counts(strata: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n_teams): how often each team is drawn, teams resampled inside each stratum."""
    c = np.zeros((n_boot, len(strata)))
    rows = np.arange(n_boot)[:, None]
    for s in pd.unique(strata):
        pos = np.flatnonzero(strata == s)
        np.add.at(c, (rows, pos[rng.integers(0, len(pos), size=(n_boot, len(pos)))]), 1.0)
    return c


def permuted(member: np.ndarray, strata: np.ndarray, n_perm: int,
             rng: np.random.Generator) -> np.ndarray:
    """(n_perm, n_teams) memberships with the labels shuffled inside each stratum."""
    keys, out = rng.random((n_perm, len(member))), np.zeros((n_perm, len(member)), dtype=bool)
    for s in pd.unique(strata):
        pos = np.flatnonzero(strata == s)
        out[:, pos] = keys[:, pos].argsort(axis=1).argsort(axis=1) < int(member[pos].sum())
    return out


def rate_gap(num: np.ndarray, den: np.ndarray, member: np.ndarray,
             w: np.ndarray | None = None) -> np.ndarray:
    """rate(member) minus rate(rest), each pooled over its teams; a row of `w` weights the teams of
    one draw, and a row of `member` is one membership."""
    m = np.atleast_2d(member).astype(float)
    w = np.ones_like(m) if w is None else w
    inside, outside = w * m, w * (1.0 - m)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (inside @ num) / (inside @ den) - (outside @ num) / (outside @ den)


def perm_gaps(num: np.ndarray, den: np.ndarray, member: np.ndarray, strata: np.ndarray,
              n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """The gap under `n_perm` memberships shuffled inside each stratum, drawn in chunks."""
    return np.concatenate([rate_gap(num, den, permuted(member, strata, min(_PERM_CHUNK, n_perm - i), rng))
                           for i in range(0, n_perm, _PERM_CHUNK)] or [np.array([])])


def summarise(point: float, boot: np.ndarray, perm: np.ndarray | None) -> dict:
    """Point; percentile intervals at 95%, 90% and Bonferroni over the two contrasts; the normal
    form beside them with the skew of the draws; MDE at 80% and the permutation p-value."""
    b = boot[np.isfinite(boot)]
    q, sd = np.percentile(b, [2.5, 97.5, 5.0, 95.0, 1.25, 98.75]), float(np.std(b))
    p = (np.nan if perm is None or not len(perm)
         else float((1 + np.sum(np.abs(perm) >= abs(point))) / (1 + len(perm))))
    return {"point": point, "lo95": q[0], "hi95": q[1], "lo90": q[2], "hi90": q[3],
            "lo_bonf": q[4], "hi_bonf": q[5], "lo95_normal": point - Z95 * sd,
            "hi95_normal": point + Z95 * sd, "skew": float(stats.skew(b)),
            "mde80": Z80 * sd, "p_perm": p}


def contrast(t: pd.DataFrame, metric: str, member: np.ndarray, perm_strata: np.ndarray,
             n_boot: int, n_perm: int, seed: list, registered: np.ndarray | None = None) -> dict:
    """One group against the rest on one metric: teams drawn inside their group, the permutation
    null, and, where given, the registered draws inside finer strata, reported beside."""
    num, den = (t[c].to_numpy(float) for c in RATES[metric])
    rng = np.random.default_rng(seed)
    point = float(rate_gap(num, den, member)[0])
    boot = rate_gap(num, den, member, draw_counts(member.astype(str), n_boot, rng))
    out = {"metric": metric, **summarise(point, boot, perm_gaps(num, den, member, perm_strata,
                                                                 n_perm, rng))}
    if registered is not None:
        reg = rate_gap(num, den, member, draw_counts(registered, n_boot, rng))
        q = np.nanpercentile(reg, [2.5, 97.5, 5.0, 95.0])
        out.update({"lo95_registered": q[0], "hi95_registered": q[1],
                    "lo90_registered": q[2], "hi90_registered": q[3]})
    return out


def groups(t: pd.DataFrame) -> dict:
    """name -> (membership, formed by the table): the two registered groups and their variants."""
    out = {f"T{k}": ((t["pos"] <= k).to_numpy(), True) for k in CUTS}
    out["G"] = (top_by(t, "goals_total", "xg_total", TOP_G), False)
    out["G_xg"] = (top_by(t, "xg_total", "goals_total", TOP_G), False)
    return out


def specs(t: pd.DataFrame, t_long: pd.DataFrame | None) -> list:
    """Every season contrast of one seed: the two groups on every metric, then their robustness
    (the other cuts, long corners only where `t_long` is given, each league left out)."""
    out = []

    def add(group: str, scope: str, table: pd.DataFrame, metrics: tuple, member: np.ndarray,
            by_table: bool, role: str = "") -> None:
        lg = table["league"].to_numpy().astype(str)
        for metric in metrics:
            out.append({"group": group, "scope": scope, "metric": metric, "table": table,
                        "role": role or ("primary" if metric == PRIMARY else "secondary"),
                        "member": member, "perm": lg if by_table else np.zeros(len(table)),
                        "registered": np.char.add(lg, member.astype(str)) if by_table else None})

    g = groups(t)
    for name, (member, by_table) in g.items():
        main = name in (f"T{TOP_T}", "G")
        add(name, "season", t, tuple(RATES) if main else (PRIMARY,), member, by_table,
            "" if main else "robustness")
    keys = pd.MultiIndex.from_frame(t[["league", "team"]])
    for name in (f"T{TOP_T}", "G"):
        member, by_table = g[name]
        if t_long is not None:
            add(name, "long_corners", t_long, (PRIMARY,),
                pd.MultiIndex.from_frame(t_long[["league", "team"]]).isin(keys[member]), by_table,
                "robustness")
        for lg in pd.unique(t["league"]):
            keep = (t["league"] != lg).to_numpy()
            add(name, f"without_{lg}", t[keep].reset_index(drop=True), (PRIMARY,), member[keep],
                by_table, "robustness")
    return out


def _season_task(spec: dict, n_boot: int, n_perm: int, seed: int, index: int) -> dict:
    r = contrast(spec["table"], spec["metric"], spec["member"], spec["perm"], n_boot, n_perm,
                 [seed, index], spec["registered"])
    return {"seed": seed, "group": spec["group"], "scope": spec["scope"], "role": spec["role"], **r}


def influence(t: pd.DataFrame, member: np.ndarray) -> pd.DataFrame:
    """The primary contrast with each member left out in turn, so no single club carries a group."""
    num, den = (t[c].to_numpy(float) for c in RATES[PRIMARY])
    out = []
    for i in np.flatnonzero(member):
        keep = np.arange(len(t)) != i
        out.append({"team": t.loc[i, "team"], "league": t.loc[i, "league"],
                    "gap_without": float(rate_gap(num[keep], den[keep], member[keep])[0])})
    return pd.DataFrame(out)


# ── The split-half version of group G, and the match-clustered reading ─────────────
def _onehot(rows: pd.DataFrame, t: pd.DataFrame) -> np.ndarray:
    """(n_rows, n_teams): which season-table team each (team, match) row belongs to."""
    ix = pd.MultiIndex.from_frame(t[["league", "team"]]).get_indexer(
        pd.MultiIndex.from_frame(rows[["league", "team"]]))
    assert (ix >= 0).all(), "a (team, match) row whose team the season table does not carry"
    out = np.zeros((len(rows), len(t)))
    out[np.arange(len(rows)), ix] = 1.0
    return out


def half_masks(rows: pd.DataFrame, games: pd.Series, n_splits: int, seed: int) -> np.ndarray:
    """(n_splits, n_rows): True where the row's match falls in its team's first half that split.

    Each team's matches are permuted on their own and the first half of them, rounded down, is
    the half the selection may see; the match is the unit, as in eda_repeatability."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n_splits, len(rows)), dtype=bool)
    for (lg, tm), mids in games.items():
        mids = np.asarray(mids)
        ranks = rng.random((n_splits, len(mids))).argsort(axis=1).argsort(axis=1)
        first = pd.Series(range(len(mids)), index=mids)
        idx = np.flatnonzero((rows["league"] == lg).to_numpy() & (rows["team"] == tm).to_numpy())
        out[:, idx] = ranks[:, first[rows["match_id"].to_numpy()[idx]].to_numpy()] < len(mids) // 2
    return out


def split_half_gap(rows: pd.DataFrame, t: pd.DataFrame, games: pd.Series, n_splits: int,
                   n_boot: int, n_perm: int, seed: int) -> list:
    """Group G chosen on one half of each team's matches and measured on the other.

    Per split the 10 teams with the most corner goals in the first half (ties by corner xG, then
    name) are measured on the second; the point averages the splits. The team draws re-average
    over the same splits. The null relabels the teams once per draw and applies that relabelling
    to every split: drawing 10 teams afresh in each split would average 500 independent nulls into
    one far narrower than the statistic it is read against, whose chosen teams persist from split
    to split. A draw that holds no member of a split leaves that split out of its average."""
    a_mask, onehot = half_masks(rows, games, n_splits, seed), _onehot(rows, t)

    def sums(col: str, mask: np.ndarray) -> np.ndarray:
        return (mask * rows[col].to_numpy(float)) @ onehot             # (n_splits, n_teams)

    goals_a, xg_a = sums("goals_total", a_mask), sums("xg_total", a_mask)
    name_rank = t["team"].rank(method="first").to_numpy()
    member = np.zeros((n_splits, len(t)), dtype=bool)
    for i in range(n_splits):
        member[i, np.lexsort((name_rank, -xg_a[i], -goals_a[i]))[:TOP_G]] = True

    rng = np.random.default_rng([seed, 1])
    counts = draw_counts(np.zeros(len(t)), n_boot, rng)
    half_b = {m: [sums(c, ~a_mask) for c in RATES[m]] for m in SPLIT_RATES}
    point = {m: [] for m in SPLIT_RATES}
    boot_sum = {m: np.zeros(n_boot) for m in SPLIT_RATES}
    boot_n = {m: np.zeros(n_boot) for m in SPLIT_RATES}
    perm_sum = {m: np.zeros(n_perm) for m in SPLIT_RATES}
    relabel = np.array([rng.permutation(len(t)) for _ in range(n_perm)]) if n_perm else None
    for i in range(n_splits):
        null = relabelled(member[i], relabel) if n_perm else None
        for m in SPLIT_RATES:
            num, den = half_b[m][0][i], half_b[m][1][i]
            point[m].append(rate_gap(num, den, member[i])[0])
            b = rate_gap(num, den, member[i], counts)
            boot_sum[m] += np.nan_to_num(b)
            boot_n[m] += np.isfinite(b)
            if null is not None:
                perm_sum[m] += rate_gap(num, den, null)
    return [{"seed": seed, "group": "G", "scope": "split_half",
             "role": "primary" if m == PRIMARY else "secondary",
             **summarise(float(np.mean(point[m])), boot_sum[m] / boot_n[m],
                         perm_sum[m] / n_splits if n_perm else None), "metric": m}
            for m in SPLIT_RATES]


def relabelled(member: np.ndarray, relabel: np.ndarray) -> np.ndarray:
    """(n_draws, n_teams): one split's membership carried over by each draw's relabelling of teams."""
    return member[relabel]


def match_clustered(rows: pd.DataFrame, t: pd.DataFrame, member: np.ndarray, n_boot: int,
                    seed: int) -> dict:
    """The primary contrast with matches resampled instead of teams, beside the team reading."""
    mids = pd.Index(pd.unique(rows["match_id"]))
    idx, onehot = mids.get_indexer(rows["match_id"]), _onehot(rows, t)
    rng, m, gaps = np.random.default_rng([seed, 2]), member.astype(float), []
    for start in range(0, n_boot, 1000):
        c = draw_counts(np.zeros(len(mids)), min(1000, n_boot - start), rng)[:, idx]
        num, den = ((c * rows[col].to_numpy(float)) @ onehot for col in RATES[PRIMARY])
        with np.errstate(invalid="ignore", divide="ignore"):
            gaps.append((num @ m) / (den @ m) - (num @ (1 - m)) / (den @ (1 - m)))
    lo, hi, lo90, hi90 = np.nanpercentile(np.concatenate(gaps), [2.5, 97.5, 5.0, 95.0])
    return {"seed": seed, "group": f"T{TOP_T}", "scope": "match_clustered", "role": "robustness",
            "metric": PRIMARY, "lo95": lo, "hi95": hi, "lo90": lo90, "hi90": hi90}


def battery(t: pd.DataFrame, t_long: pd.DataFrame | None, rows: pd.DataFrame, games: pd.Series,
            n_boot: int, n_perm: int, n_splits: int, n_split_perm: int, seeds: tuple,
            jobs: int) -> pd.DataFrame:
    """Every contrast at every seed, as seeded single-threaded tasks over `jobs` workers."""
    sp, member_t = specs(t, t_long), groups(t)[f"T{TOP_T}"][0]
    tasks = [delayed(_season_task)(s, n_boot, n_perm, seed, i)
             for seed in seeds for i, s in enumerate(sp)]
    tasks += [delayed(split_half_gap)(rows, t, games, n_splits, n_boot, n_split_perm, seed)
              for seed in seeds]
    tasks += [delayed(match_clustered)(rows, t, member_t, n_boot, seed) for seed in seeds]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        done = Parallel(n_jobs=jobs, verbose=2)(tasks)
    res = pd.DataFrame([r for d in done for r in (d if isinstance(d, list) else [d])])
    secondary = res[(res["scope"] == "season") & (res["role"] == "secondary")]
    for _, idx in secondary.groupby(["seed", "group"]).groups.items():
        res.loc[idx, "q_bh"] = bh(res.loc[idx, "p_perm"].to_numpy())
    return res


def decide(res: pd.DataFrame, pooled: float) -> dict:
    """The registered decisions, each holding only if its decisive bound keeps its side at every
    stability seed, with the Bonferroni reading and the per-seed bounds beside."""
    margin = MARGIN_FRACTION * pooled
    T = res[(res["group"] == f"T{TOP_T}") & (res["scope"] == "season") & (res["metric"] == PRIMARY)]
    G = res[(res["group"] == "G") & (res["scope"] == "split_half") & (res["metric"] == PRIMARY)]
    M = res[(res["scope"] == "match_clustered")]
    inside = lambda lo, hi: (lo > -margin) & (hi < margin)
    return {
        "T1_registered": bool((T["lo95_registered"] > 0).all()),
        "T2_registered": bool(inside(T["lo90_registered"], T["hi90_registered"]).all()),
        "T2_match_clustered": bool(inside(M["lo90"], M["hi90"]).all()),
        "pooled_rate": pooled, "margin": margin, "seeds": int(T["seed"].nunique()),
        "T1": bool((T["lo95"] > 0).all()), "T1_any_seed": bool((T["lo95"] > 0).any()),
        "T1_bonferroni": bool((T["lo_bonf"] > 0).all()),
        "T2": bool(inside(T["lo90"], T["hi90"]).all()),
        "T2_any_seed": bool(inside(T["lo90"], T["hi90"]).any()),
        "T2_bonferroni": bool(inside(T["lo95"], T["hi95"]).all()),
        "G1": bool((G["lo95"] > 0).all()), "G1_any_seed": bool((G["lo95"] > 0).any()),
        "G1_bonferroni": bool((G["lo_bonf"] > 0).all()),
        "T_point": float(T["point"].iloc[0]), "T_lo95_by_seed": T["lo95"].round(6).tolist(),
        "T_hi95_by_seed": T["hi95"].round(6).tolist(),
        "G_point_by_seed": G["point"].round(6).tolist(), "G_lo95_by_seed": G["lo95"].round(6).tolist(),
    }


def main() -> None:
    """Build the season table, run the battery at every seed, read the decisions, write records."""
    ap = argparse.ArgumentParser()
    d_def, s_def = kind_inputs("corner")
    m_def = _DER / "eda_matches.parquet"
    ap.add_argument("--deliveries", type=str, default=str(d_def))
    ap.add_argument("--shots", type=str, default=str(s_def))
    ap.add_argument("--matches", type=str, default=str(m_def))
    ap.add_argument("--standings", type=str, default=str(_DER / "eda_standings.parquet"))
    ap.add_argument("--boot", type=int, default=_TEAM_DRAWS)
    ap.add_argument("--perms", type=int, default=_PERMS)
    ap.add_argument("--splits", type=int, default=_SPLITS)
    ap.add_argument("--split-perms", type=int, default=_SPLIT_PERMS)
    ap.add_argument("--seeds", type=int, default=_SEEDS)
    ap.add_argument("--jobs", type=int, default=20,
                    help="workers for the seeded tasks; the output does not depend on it")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="where to write (default: data/derived, only from the grant's tables at "
                         "the registered precision)")
    ap.add_argument("--kind", choices=COMBINED_KINDS, default="corner",
                    help="the population: corners, or the flank free kicks from x >= 85")
    args = ap.parse_args()
    kt = kind_tag((args.kind,))
    out = Path(args.out_dir) if args.out_dir else _DER
    precision = (args.boot, args.perms, args.splits, args.split_perms, args.seeds)
    inputs = [Path(p).resolve() for p in (args.deliveries, args.shots, args.matches)]
    if out.resolve() == _DER.resolve() and (
            precision != (_TEAM_DRAWS, _PERMS, _SPLITS, _SPLIT_PERMS, _SEEDS)
            or inputs != [p.resolve() for p in (d_def, s_def, m_def)]):
        raise SystemExit(f"STOP: the published top_teams records come only from the grant's tables at "
                         f"(boot, perms, splits, split perms, seeds) = "
                         f"{(_TEAM_DRAWS, _PERMS, _SPLITS, _SPLIT_PERMS, _SEEDS)}; pass --out-dir")

    deliv = pd.read_parquet(args.deliveries)
    deliv = deliv[deliv["set_piece_type"] == args.kind]
    shots = pd.read_parquet(args.shots)
    s = routes(shots[shots["set_piece_type"] == args.kind])
    matches, stand = pd.read_parquet(args.matches), pd.read_parquet(args.standings)
    teams = pd.MultiIndex.from_frame(
        deliv[["league", "attacking_team"]].drop_duplicates().sort_values(["league", "attacking_team"])
        .set_axis(["league", "team"], axis=1))
    factors = lto_factors(s, teams)
    rows = team_match_rows(deliv, s, factors)
    t = season_table(rows, s, matches, stand, factors)
    t_long = None
    if args.kind == "corner":
        long_d = deliv[~deliv["short_corner"].astype(bool)]
        long_s = s[s["set_piece_id"].isin(long_d["set_piece_id"])]
        t_long = season_table(team_match_rows(long_d, long_s, factors), long_s, matches, stand,
                              factors)
    games = pd.concat([matches[["league", c, "match_id"]].set_axis(["league", "team", "match_id"], axis=1)
                       for c in ("home_team", "away_team")]).groupby(["league", "team"])["match_id"].unique()
    pooled_f = s.groupby("route")["is_goal"].sum() / s.groupby("route")["xg"].sum()
    print(f"{len(t)} teams, {int(t['corners'].sum())} {args.kind} deliveries, {len(s)} shots without "
          f"own goals; goals over xG by route over the grant: {pooled_f.round(3).to_dict()}")

    seeds = tuple(range(args.seeds))
    res = battery(t, t_long, rows, games, args.boot, args.perms, args.splits, args.split_perms,
                  seeds, args.jobs)
    g = groups(t)
    for name, (member, _) in g.items():
        t[f"in_{name}"] = member
    infl = pd.concat([influence(t, g[name][0]).assign(group=name) for name in (f"T{TOP_T}", "G")])
    num, den = (t[c].to_numpy(float) for c in RATES[PRIMARY])
    member_t = g[f"T{TOP_T}"][0]
    by_league = pd.DataFrame([{"league": lg, "corners": float(x["corners"].sum()),
                               "gap": float(rate_gap(x["xg_aft"].to_numpy(float), x["corners"].to_numpy(float),
                                                     x[f"in_T{TOP_T}"].to_numpy())[0])}
                              for lg, x in t.groupby("league")])
    decisions = decide(res, float(num.sum() / den.sum()))
    rates = num / den
    decisions.update({
        "T_team_weighted_gap": float(rates[member_t].mean() - rates[~member_t].mean()),
        "T_within_league_weighted_gap": float(np.average(by_league["gap"], weights=by_league["corners"])),
        "influence_range_T": [float(infl.loc[infl.group == f"T{TOP_T}", "gap_without"].min()),
                              float(infl.loc[infl.group == f"T{TOP_T}", "gap_without"].max())],
        "influence_range_G": [float(infl.loc[infl.group == "G", "gap_without"].min()),
                              float(infl.loc[infl.group == "G", "gap_without"].max())]})

    scores = deliv[["attacking_team", "match_id", "set_piece_id"]].assign(
        aft_xg=deliv["set_piece_id"].map(s[s["route"] == "aft"].groupby("set_piece_id")["xg"].sum()).fillna(0.0),
        xg=deliv["set_piece_id"].map(s.groupby("set_piece_id")["xg"].sum()).fillna(0.0)
    ).drop(columns="set_piece_id")
    rel = split_half(deliv, s, args.splits, scores=scores, n_boot=_REL_BOOT,
                     boot_cols=["corners_per_match", "aft_xg", "xg"])
    decisions["aft_xg_full_reliability"] = float(rel.set_index("metric").loc["aft_xg", "full_r_sb"])
    decisions["per_team_ranking_allowed"] = decisions["aft_xg_full_reliability"] >= RELIABILITY_FLOOR

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    view = res[res["seed"] == 0][["group", "scope", "role", "metric", "point", "lo95", "hi95",
                                  "lo95_normal", "hi95_normal", "skew", "mde80", "p_perm", "q_bh"]]
    print("\nCONTRASTS, seed 0, group minus rest (rates pooled over each group's corners)")
    print(view.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\nT within each league:\n" + by_league.to_string(index=False))
    print("\nINFLUENCE, the primary gap with each member left out:\n" + infl.to_string(index=False))
    print("\nDECISIONS (each holds only if it holds at every seed)")
    for k, v in decisions.items():
        print(f"  {k:30s} {v}")
    print("\nRELIABILITY (split-half, Spearman-Brown to the full season)")
    print(rel.round(3).to_string(index=False))

    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("table", t), ("contrasts", res), ("influence", infl),
                        ("by_league", by_league), ("decisions", pd.DataFrame([decisions])),
                        ("reliability", rel)):
        frame.to_parquet(out / f"top_teams_{name}{kt}.parquet")
    print(f"\nsaved the six top_teams{kt} records in {out}")


if __name__ == "__main__":
    main()
